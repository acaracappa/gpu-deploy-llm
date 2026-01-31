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
    mode: DeploymentMode = DeploymentMode.AUTO

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
            raise
        except Exception as e:
            result.error = str(e)
            logger.error(f"Docker deployment failed: {e}")

            # Collect container logs on failure
            try:
                result.container_logs = await self.ssh.get_container_logs(
                    config.container_name
                )
            except Exception:
                pass

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
            result.process_pid = pid
            result.start_duration_seconds = time.time() - start_start
            logger.info(f"vLLM process started (PID: {pid})")

            # Build endpoint URL
            result.endpoint = f"http://{self.ssh.host}:{config.port}/v1"
            result.success = True

            logger.info(f"vLLM deployed successfully at {result.endpoint}")

        except DeploymentError:
            raise
        except Exception as e:
            result.error = str(e)
            logger.error(f"Pip deployment failed: {e}")
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

    async def _install_vllm_pip(self, python_cmd: str) -> None:
        """Install vLLM via pip."""
        # Ensure pip is available
        await self._ensure_pip(python_cmd)

        # Install vLLM with timeout for large package
        result = await self.ssh.run(
            f"{python_cmd} -m pip install --upgrade vllm=={VLLM_PIP_VERSION}",
            timeout=600,  # 10 min timeout
        )

        if not result.success:
            # Try without version pinning as fallback
            logger.warning(f"Pinned version failed, trying latest: {result.stderr}")
            result = await self.ssh.run(
                f"{python_cmd} -m pip install --upgrade vllm",
                timeout=600,
            )

        if not result.success:
            raise DeploymentError(
                f"Failed to install vLLM: {result.stderr}",
                stage="install_vllm",
            )

    async def _start_vllm_process(
        self,
        config: DeploymentConfig,
        python_cmd: str,
    ) -> int:
        """Start vLLM server process in background."""
        # Build vLLM command
        cmd_parts = [
            f"VLLM_API_KEY={config.api_key}",
            "nohup",
            python_cmd,
            "-m", "vllm.entrypoints.openai.api_server",
            f"--model {config.model_id}",
            f"--port {config.port}",
            "--host 0.0.0.0",
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
            "--api-key-env VLLM_API_KEY",
        ]

        # Add quantization if specified
        if config.quantization != Quantization.NONE:
            cmd_parts.append(f"--quantization {config.quantization.value}")

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

        # Get PID
        import asyncio
        await asyncio.sleep(2)  # Give it time to start

        pid_result = await self.ssh.run("pgrep -f 'vllm.entrypoints' | head -1")
        if pid_result.success and pid_result.stdout.strip():
            return int(pid_result.stdout.strip())

        # Check if process failed to start
        log_result = await self.ssh.run("cat /tmp/vllm.log | tail -20")
        raise DeploymentError(
            f"vLLM process failed to start. Log: {log_result.stdout}",
            stage="start_vllm",
        )

    async def get_vllm_logs(self) -> str:
        """Get vLLM logs (works for both Docker and pip modes)."""
        # Try Docker first
        containers = await self.ssh.get_running_containers()
        if CONTAINER_NAME in containers:
            return await self.ssh.get_container_logs(CONTAINER_NAME)

        # Try pip mode logs
        result = await self.ssh.run("cat /tmp/vllm.log 2>/dev/null || echo 'No logs found'")
        return result.stdout

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
        cmd_parts = [
            "docker run -d",
            "--gpus all",
            "--ipc=host",  # Required for vLLM/PyTorch shared memory
            "--user 1000:1000",  # P1: Non-root container
            f"-p {config.port}:{VLLM_PORT}",
            f"-e VLLM_API_KEY={config.api_key}",  # P0: API key
            f"--name {config.container_name}",
            config.image,
            f"--model {config.model_id}",
            "--api-key-env VLLM_API_KEY",  # P0: Enforce API key
            "--host 0.0.0.0",
            f"--port {VLLM_PORT}",
            f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        ]

        # Add quantization if specified
        if config.quantization != Quantization.NONE:
            cmd_parts.append(f"--quantization {config.quantization.value}")

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
    ) -> None:
        """Wait for container to be running."""
        import asyncio

        for _ in range(timeout):
            result = await self.ssh.run(
                f"docker inspect -f '{{{{.State.Running}}}}' {container_name}"
            )
            if result.success and "true" in result.stdout.lower():
                return
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
