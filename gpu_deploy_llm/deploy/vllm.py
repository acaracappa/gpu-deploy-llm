"""vLLM deployment logic via SSH.

Supports two deployment modes:
1. Docker (preferred) - uses vllm/vllm-openai container
2. Pip (fallback) - installs vllm via pip and runs directly

Security requirements implemented:
- P0: vLLM API key generation and configuration
- P0: SSH key handled in memory only (via SSHConnection)
- P1: Pinned Docker image version
- P1: Non-root container user
"""

import logging
import secrets
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from gpu_deploy_llm.ssh.connection import SSHConnection
from gpu_deploy_llm.models.registry import Quantization
from gpu_deploy_llm.utils.errors import DeploymentError

logger = logging.getLogger(__name__)

# Pinned vLLM image version (P1 security requirement)
VLLM_IMAGE = "vllm/vllm-openai:v0.4.0"

# Pinned vLLM pip version
VLLM_PIP_VERSION = "0.4.0"

# Container configuration
CONTAINER_NAME = "vllm-server"
VLLM_PORT = 8000


class DeploymentMode(str, Enum):
    """Deployment mode for vLLM."""
    DOCKER = "docker"
    PIP = "pip"
    AUTO = "auto"  # Try Docker first, fallback to pip


@dataclass
class DeploymentConfig:
    """Configuration for vLLM deployment."""

    model_id: str
    quantization: Quantization = Quantization.NONE
    gpu_count: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    api_key: Optional[str] = None  # Auto-generated if not provided
    mode: DeploymentMode = DeploymentMode.PIP  # Default to pip - more reliable on cloud GPUs

    # Docker settings
    image: str = VLLM_IMAGE
    container_name: str = CONTAINER_NAME
    port: int = VLLM_PORT

    def __post_init__(self):
        # Generate secure API key if not provided (P0 security requirement)
        if not self.api_key:
            self.api_key = secrets.token_urlsafe(32)


@dataclass
class DeploymentResult:
    """Result of vLLM deployment."""

    success: bool
    endpoint: str = ""
    api_key: str = ""
    container_name: str = ""
    mode: DeploymentMode = DeploymentMode.DOCKER
    error: Optional[str] = None
    container_logs: str = ""
    process_pid: Optional[int] = None  # For pip mode

    # Timing info for diagnostics
    pull_duration_seconds: float = 0.0
    start_duration_seconds: float = 0.0


class VLLMDeployer:
    """Deploys vLLM on a remote GPU instance via SSH.

    Supports two deployment modes:
    - Docker (preferred): Uses containerized vLLM
    - Pip (fallback): Installs vLLM via pip and runs directly
    """

    def __init__(self, ssh: SSHConnection):
        """Initialize deployer with SSH connection.

        Args:
            ssh: Active SSH connection to GPU instance
        """
        self.ssh = ssh

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy vLLM with the specified configuration.

        Args:
            config: Deployment configuration

        Returns:
            DeploymentResult with endpoint and API key
        """
        # Determine deployment mode
        use_docker = False
        use_pip = False

        if config.mode == DeploymentMode.DOCKER:
            use_docker = True
        elif config.mode == DeploymentMode.PIP:
            use_pip = True
        else:  # AUTO mode
            logger.info("Checking Docker availability...")
            if await self.ssh.check_docker():
                use_docker = True
                logger.info("Docker available, using container deployment")
            else:
                use_pip = True
                logger.info("Docker not available, using pip deployment")

        if use_docker:
            return await self._deploy_docker(config)
        else:
            return await self._deploy_pip(config)

    async def _deploy_docker(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy vLLM using Docker container."""
        import time

        result = DeploymentResult(
            success=False,
            api_key=config.api_key or "",
            container_name=config.container_name,
            mode=DeploymentMode.DOCKER,
        )

        try:
            # Step 1: Stop any existing container
            await self._stop_existing_container(config.container_name)

            # Step 2: Pull vLLM image
            logger.info(f"Pulling Docker image: {config.image}")
            pull_start = time.time()
            await self._pull_image(config.image)
            result.pull_duration_seconds = time.time() - pull_start
            logger.info(f"Image pulled in {result.pull_duration_seconds:.1f}s")

            # Step 3: Start vLLM container
            logger.info(f"Starting vLLM container with model: {config.model_id}")
            start_start = time.time()
            await self._start_container(config)
            result.start_duration_seconds = time.time() - start_start
            logger.info(f"Container started in {result.start_duration_seconds:.1f}s")

            # Step 4: Wait for container to be running
            await self._wait_for_container(config.container_name)

            # Build endpoint URL
            result.endpoint = f"http://{self.ssh.host}:{config.port}/v1"
            result.success = True

            logger.info(f"vLLM deployed successfully at {result.endpoint}")

        except DeploymentError:
            # Cleanup container before re-raising
            try:
                await self._stop_existing_container(config.container_name)
            except Exception as cleanup_err:
                logger.warning(f"Container cleanup failed: {cleanup_err}")
            raise
        except Exception as e:
            result.error = str(e)
            logger.error(f"Docker deployment failed: {e}")

            # Collect container logs on failure
            try:
                result.container_logs = await self.ssh.get_container_logs(
                    config.container_name
                )
            except Exception as log_err:
                logger.warning(f"Failed to collect container logs: {log_err}")

            # Cleanup container
            try:
                await self._stop_existing_container(config.container_name)
            except Exception as cleanup_err:
                logger.warning(f"Container cleanup failed: {cleanup_err}")

            raise DeploymentError(str(e), stage="deploy_docker")

        return result

    async def _deploy_pip(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy vLLM using pip installation."""
        import time

        result = DeploymentResult(
            success=False,
            api_key=config.api_key or "",
            mode=DeploymentMode.PIP,
        )
        started_pid: Optional[int] = None

        try:
            # Step 1: Check Python availability
            logger.info("Checking Python availability...")
            python_cmd = await self._find_python()
            logger.info(f"Using Python: {python_cmd}")

            # Step 2: Kill any existing vLLM process
            await self._kill_existing_vllm()

            # Step 3: Install vLLM via pip
            logger.info("Installing vLLM via pip...")
            pull_start = time.time()
            await self._install_vllm_pip(python_cmd)
            result.pull_duration_seconds = time.time() - pull_start
            logger.info(f"vLLM installed in {result.pull_duration_seconds:.1f}s")

            # Step 4: Start vLLM server
            logger.info(f"Starting vLLM server with model: {config.model_id}")
            start_start = time.time()
            pid = await self._start_vllm_process(config, python_cmd)
            started_pid = pid
            result.process_pid = pid
            result.start_duration_seconds = time.time() - start_start
            logger.info(f"vLLM process started (PID: {pid})")

            # Build endpoint URL
            result.endpoint = f"http://{self.ssh.host}:{config.port}/v1"
            result.success = True

            logger.info(f"vLLM deployed successfully at {result.endpoint}")

        except DeploymentError:
            # Cleanup process if it was started
            if started_pid is not None:
                await self._kill_vllm_process(started_pid)
            raise
        except Exception as e:
            result.error = str(e)
            logger.error(f"Pip deployment failed: {e}")
            # Cleanup process if it was started
            if started_pid is not None:
                await self._kill_vllm_process(started_pid)
            raise DeploymentError(str(e), stage="deploy_pip")

        return result

    async def _find_python(self) -> str:
        """Find suitable Python command."""
        for cmd in ["python3", "python"]:
            result = await self.ssh.run(f"{cmd} --version")
            if result.success:
                return cmd

        raise DeploymentError("Python not found on remote host", stage="find_python")

    async def _kill_existing_vllm(self) -> None:
        """Kill any existing vLLM processes."""
        await self.ssh.run("pkill -f 'vllm.entrypoints' || true")
        await self.ssh.run("pkill -f 'vllm serve' || true")

    async def _kill_vllm_process(self, pid: int) -> None:
        """Kill a specific vLLM process by PID."""
        try:
            await self.ssh.run(f"kill -9 {pid} 2>/dev/null || true")
        except Exception as e:
            logger.warning(f"Failed to kill PID {pid}: {e}")

    async def _ensure_pip(self, python_cmd: str) -> None:
        """Ensure pip is available, install if needed."""
        result = await self.ssh.run(f"{python_cmd} -m pip --version")
        if result.success:
            return

        logger.info("pip not found, attempting to install...")

        # Try ensurepip first
        result = await self.ssh.run(f"{python_cmd} -m ensurepip --upgrade")
        if result.success:
            logger.info("pip installed via ensurepip")
            return

        # Try get-pip.py
        result = await self.ssh.run(
            f"curl -sS https://bootstrap.pypa.io/get-pip.py | {python_cmd}",
            timeout=120,
        )
        if result.success:
            logger.info("pip installed via get-pip.py")
            return

        # Try apt-get (Debian/Ubuntu)
        result = await self.ssh.run(
            "apt-get update && apt-get install -y python3-pip",
            timeout=120,
        )
        if result.success:
            logger.info("pip installed via apt-get")
            return

        raise DeploymentError(
            "Could not install pip on remote host",
            stage="install_pip",
        )

    async def _detect_cuda_version(self) -> str:
        """Detect CUDA version from nvidia-smi to select correct PyTorch wheel."""
        # Try to get CUDA version from nvidia-smi
        result = await self.ssh.run(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"
        )

        if result.success and result.stdout.strip():
            driver_version = result.stdout.strip()
            try:
                major = int(driver_version.split('.')[0])
                # Driver 525+ supports CUDA 12.x, older drivers use CUDA 11.8
                if major >= 525:
                    logger.info(f"Driver {driver_version} detected, using CUDA 12.1")
                    return "cu121"
                else:
                    logger.info(f"Driver {driver_version} detected, using CUDA 11.8")
                    return "cu118"
            except (ValueError, IndexError):
                pass

        # Default to CUDA 12.1 (most common on recent cloud GPUs)
        logger.info("Could not detect driver version, defaulting to CUDA 12.1")
        return "cu121"

    async def _install_vllm_pip(self, python_cmd: str) -> None:
        """Install vLLM via pip with all required dependencies.

        Installation order is critical:
        1. Detect CUDA version from driver
        2. Install PyTorch with correct CUDA support
        3. Verify PyTorch CUDA is working
        4. Install numpy < 2.0 (vLLM/transformers compatibility)
        5. Install vLLM
        6. Verify vLLM imports correctly
        """
        # Ensure pip is available
        await self._ensure_pip(python_cmd)

        # Step 1: Detect CUDA version
        cuda_version = await self._detect_cuda_version()

        # Step 2: Install PyTorch with CUDA support first (required by vLLM)
        logger.info(f"Installing PyTorch with CUDA support ({cuda_version})...")
        torch_result = await self.ssh.run(
            f"{python_cmd} -m pip install torch --index-url https://download.pytorch.org/whl/{cuda_version}",
            timeout=600,  # 10 min timeout for PyTorch download
        )

        if not torch_result.success:
            logger.warning(f"PyTorch {cuda_version} install failed, trying cu118: {torch_result.stderr}")
            torch_result = await self.ssh.run(
                f"{python_cmd} -m pip install torch --index-url https://download.pytorch.org/whl/cu118",
                timeout=600,
            )

        if not torch_result.success:
            logger.warning(f"PyTorch CUDA install failed, trying default: {torch_result.stderr}")
            torch_result = await self.ssh.run(
                f"{python_cmd} -m pip install torch",
                timeout=600,
            )

        # Step 3: Verify PyTorch CUDA is working
        logger.info("Verifying PyTorch CUDA...")
        verify_torch = await self.ssh.run(
            f'{python_cmd} -c "import torch; print(f\'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}\')"'
        )
        if verify_torch.success:
            logger.info(f"PyTorch verification: {verify_torch.stdout.strip()}")
            if "CUDA: False" in verify_torch.stdout:
                logger.warning("PyTorch installed but CUDA not available - model loading may fail")
        else:
            raise DeploymentError(
                f"PyTorch installation failed: {verify_torch.stderr}",
                stage="install_pytorch",
            )

        # Step 4: Install compatible numpy version (vLLM/transformers need numpy < 2.0)
        # numpy 2.0 removed numpy.lib.function_base which breaks many ML libraries
        logger.info("Installing compatible numpy (<2.0)...")
        numpy_result = await self.ssh.run(
            f"{python_cmd} -m pip install 'numpy>=1.26,<2.0'",
            timeout=120,
        )
        if not numpy_result.success:
            logger.warning(f"numpy install warning: {numpy_result.stderr}")

        # Verify numpy version
        verify_numpy = await self.ssh.run(
            f'{python_cmd} -c "import numpy; print(f\'numpy {{numpy.__version__}}\')"'
        )
        if verify_numpy.success:
            logger.info(f"numpy verification: {verify_numpy.stdout.strip()}")

        # Step 5: Install vLLM
        logger.info(f"Installing vLLM {VLLM_PIP_VERSION}...")
        result = await self.ssh.run(
            f"{python_cmd} -m pip install vllm=={VLLM_PIP_VERSION}",
            timeout=600,  # 10 min timeout
        )

        if not result.success:
            # Try without version pinning as fallback
            logger.warning(f"Pinned version failed, trying latest: {result.stderr}")
            result = await self.ssh.run(
                f"{python_cmd} -m pip install vllm",
                timeout=600,
            )

        if not result.success:
            raise DeploymentError(
                f"Failed to install vLLM: {result.stderr}",
                stage="install_vllm",
            )

        # Step 6: Verify vLLM imports correctly
        logger.info("Verifying vLLM installation...")
        verify_vllm = await self.ssh.run(
            f'{python_cmd} -c "from vllm import LLM; print(\'vLLM import OK\')"',
            timeout=60,
        )
        if not verify_vllm.success:
            # Get more details about the import error
            detail_result = await self.ssh.run(
                f'{python_cmd} -c "import vllm" 2>&1 | tail -10'
            )
            raise DeploymentError(
                f"vLLM installation verification failed: {detail_result.stdout or verify_vllm.stderr}",
                stage="verify_vllm",
            )
        logger.info("vLLM installation verified successfully")

    async def _start_vllm_process(
        self,
        config: DeploymentConfig,
        python_cmd: str,
    ) -> int:
        """Start vLLM server process in background with startup verification.

        Verifies:
        1. Process starts and stays alive for 5 seconds
        2. Process doesn't crash immediately (common with dependency issues)
        """
        import asyncio

        # Build vLLM command
        # Use VLLM_API_KEY environment variable instead of --api-key to avoid exposing key in process list
        # Use 'env' to set environment variable before nohup runs the command
        cmd_parts = [
            "nohup",
            "env",
            f"VLLM_API_KEY={shlex.quote(config.api_key)}",
            python_cmd,
            "-m", "vllm.entrypoints.openai.api_server",
            f"--model {shlex.quote(config.model_id)}",
            f"--port {config.port}",
            "--host 0.0.0.0",
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        ]

        # Add quantization if specified
        if config.quantization != Quantization.NONE:
            cmd_parts.append(f"--quantization {shlex.quote(config.quantization.value)}")

        # Add tensor parallelism for multi-GPU
        if config.gpu_count > 1:
            cmd_parts.append(f"--tensor-parallel-size {config.gpu_count}")

        # Add max model length if specified
        if config.max_model_len:
            cmd_parts.append(f"--max-model-len {config.max_model_len}")

        # Redirect output and run in background
        cmd_parts.append("> /tmp/vllm.log 2>&1 &")

        command = " ".join(cmd_parts)
        logger.debug(f"vLLM command: {command}")

        result = await self.ssh.run(command)
        if not result.success:
            raise DeploymentError(
                f"Failed to start vLLM: {result.stderr}",
                stage="start_vllm",
            )

        # Wait briefly for process to start
        await asyncio.sleep(3)

        # Get initial PID using pgrep -n (newest) for more reliable detection
        pid_result = await self.ssh.run("pgrep -n -f 'vllm.entrypoints'")
        if not (pid_result.success and pid_result.stdout.strip()):
            # Process didn't start - get logs
            log_result = await self.ssh.run("tail -30 /tmp/vllm.log 2>/dev/null")
            raise DeploymentError(
                f"vLLM process failed to start. Logs:\n{log_result.stdout}",
                stage="start_vllm",
            )

        initial_pid = pid_result.stdout.strip()
        logger.info(f"vLLM process started with PID {initial_pid}")

        # Wait and verify process stays alive (catches immediate crashes)
        await asyncio.sleep(5)

        # Check if our specific PID is still running
        pid_check = await self.ssh.run(f"kill -0 {initial_pid} 2>/dev/null && echo 'alive'")
        if not (pid_check.success and "alive" in pid_check.stdout):
            # Process died - get logs to understand why
            log_result = await self.ssh.run("tail -50 /tmp/vllm.log 2>/dev/null")
            raise DeploymentError(
                f"vLLM process (PID {initial_pid}) crashed immediately after starting. Logs:\n{log_result.stdout}",
                stage="start_vllm",
            )

        # Check if new vLLM processes spawned (indicates restart loop)
        pid_result = await self.ssh.run("pgrep -n -f 'vllm.entrypoints'")
        if pid_result.success and pid_result.stdout.strip():
            newest_pid = pid_result.stdout.strip()
            if newest_pid != initial_pid:
                log_result = await self.ssh.run("tail -30 /tmp/vllm.log 2>/dev/null")
                raise DeploymentError(
                    f"vLLM process restarting (new PID {newest_pid} detected after starting {initial_pid}). "
                    f"This usually indicates a dependency issue. Logs:\n{log_result.stdout}",
                    stage="start_vllm",
                )

        logger.info(f"vLLM process verified stable (PID {initial_pid})")
        return int(initial_pid)

    async def get_vllm_logs(self, lines: int = 50) -> str:
        """Get vLLM logs (works for both Docker and pip modes)."""
        # Try Docker first
        containers = await self.ssh.get_running_containers()
        if CONTAINER_NAME in containers:
            return await self.ssh.get_container_logs(CONTAINER_NAME)

        # Try pip mode logs
        result = await self.ssh.run(f"tail -{lines} /tmp/vllm.log 2>/dev/null || echo 'No logs found'")
        return result.stdout

    async def get_gpu_memory_usage(self) -> str:
        """Get GPU memory usage from nvidia-smi."""
        result = await self.ssh.run(
            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null"
        )
        if result.success and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 3:
                used, total, util = [p.strip() for p in parts[:3]]
                return f"GPU: {used}MB/{total}MB ({util}% util)"
        return "GPU: unknown"

    async def get_vllm_status(self) -> dict:
        """Get comprehensive vLLM status for debugging."""
        status = {
            "process_running": False,
            "pid": None,
            "gpu_memory": "unknown",
            "last_log_lines": [],
        }

        # Check if process is running
        pid_result = await self.ssh.run("pgrep -f 'vllm.entrypoints' | head -1")
        if pid_result.success and pid_result.stdout.strip():
            status["process_running"] = True
            status["pid"] = pid_result.stdout.strip()

        # Get GPU memory
        status["gpu_memory"] = await self.get_gpu_memory_usage()

        # Get last few log lines
        log_result = await self.ssh.run("tail -5 /tmp/vllm.log 2>/dev/null")
        if log_result.success:
            status["last_log_lines"] = [l for l in log_result.stdout.strip().split("\n") if l]

        return status

    async def _stop_existing_container(self, container_name: str) -> None:
        """Stop and remove existing container if present."""
        # Check if container exists
        result = await self.ssh.run(
            f"docker ps -a --filter name=^{container_name}$ --format '{{{{.Names}}}}'"
        )

        if result.success and container_name in result.stdout:
            logger.info(f"Stopping existing container: {container_name}")
            await self.ssh.run(f"docker stop {container_name}")
            await self.ssh.run(f"docker rm {container_name}")

    async def _pull_image(self, image: str) -> None:
        """Pull Docker image."""
        result = await self.ssh.run(f"docker pull {image}", timeout=600)  # 10 min timeout
        if not result.success:
            raise DeploymentError(
                f"Failed to pull image {image}: {result.stderr}",
                stage="pull_image",
            )

    async def _start_container(self, config: DeploymentConfig) -> None:
        """Start vLLM container with secure configuration."""
        # Build docker run command with security controls
        # Use VLLM_API_KEY environment variable instead of --api-key to avoid exposing key in process list
        cmd_parts = [
            "docker run -d",
            "--gpus all",
            "--ipc=host",  # Required for vLLM/PyTorch shared memory
            "--user 1000:1000",  # P1: Non-root container
            f"-e VLLM_API_KEY={shlex.quote(config.api_key)}",  # P0: API key via env var (not visible in ps)
            f"-p {config.port}:{VLLM_PORT}",
            f"--name {shlex.quote(config.container_name)}",
            config.image,
            f"--model {shlex.quote(config.model_id)}",
            "--host 0.0.0.0",
            f"--port {VLLM_PORT}",
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        ]

        # Add quantization if specified
        if config.quantization != Quantization.NONE:
            cmd_parts.append(f"--quantization {shlex.quote(config.quantization.value)}")

        # Add tensor parallelism for multi-GPU
        if config.gpu_count > 1:
            cmd_parts.append(f"--tensor-parallel-size {config.gpu_count}")

        # Add max model length if specified
        if config.max_model_len:
            cmd_parts.append(f"--max-model-len {config.max_model_len}")

        command = " ".join(cmd_parts)
        logger.debug(f"Docker command: {command}")

        result = await self.ssh.run(command, timeout=120)
        if not result.success:
            raise DeploymentError(
                f"Failed to start container: {result.stderr}",
                stage="start_container",
            )

    async def _wait_for_container(
        self,
        container_name: str,
        timeout: int = 30,
        max_ssh_retries: int = 3,
    ) -> None:
        """Wait for container to be running."""
        import asyncio
        from gpu_deploy_llm.utils.errors import SSHConnectionError

        ssh_errors = 0
        for _ in range(timeout):
            try:
                result = await self.ssh.run(
                    f"docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(container_name)}"
                )
                # Reset SSH error count on successful command
                ssh_errors = 0
                if result.success and "true" in result.stdout.lower():
                    return
            except SSHConnectionError as e:
                ssh_errors += 1
                if ssh_errors >= max_ssh_retries:
                    raise DeploymentError(
                        f"SSH connection failed {max_ssh_retries} times while waiting for container: {e}",
                        stage="wait_container",
                    )
                logger.warning(f"Transient SSH error ({ssh_errors}/{max_ssh_retries}): {e}")
            await asyncio.sleep(1)

        raise DeploymentError(
            f"Container {container_name} not running after {timeout}s",
            stage="wait_container",
        )

    async def get_container_status(self, container_name: str = CONTAINER_NAME) -> dict:
        """Get container status information."""
        result = await self.ssh.run(
            f"docker inspect {container_name} --format "
            "'{{{{.State.Status}}}} {{{{.State.StartedAt}}}} {{{{.State.Health.Status}}}}'"
        )

        if not result.success:
            return {"status": "not_found", "error": result.stderr}

        parts = result.stdout.strip().split()
        return {
            "status": parts[0] if len(parts) > 0 else "unknown",
            "started_at": parts[1] if len(parts) > 1 else "",
            "health": parts[2] if len(parts) > 2 else "unknown",
        }

    async def stop(self, container_name: str = CONTAINER_NAME) -> bool:
        """Stop vLLM (works for both Docker and pip modes).

        Args:
            container_name: Container name to stop (Docker mode)

        Returns:
            True if stopped successfully
        """
        stopped = False

        # Try Docker first
        containers = await self.ssh.get_running_containers()
        if container_name in containers:
            result = await self.ssh.run(f"docker stop {container_name}")
            if result.success:
                await self.ssh.run(f"docker rm {container_name}")
                logger.info(f"Container {container_name} stopped and removed")
                stopped = True

        # Also kill any pip-mode processes
        result = await self.ssh.run("pkill -f 'vllm.entrypoints' || true")
        if result.success:
            logger.info("vLLM process killed")
            stopped = True

        return stopped
