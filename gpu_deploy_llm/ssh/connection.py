"""SSH connection and command execution.

Matches patterns from cloud-gpu-shopper:
- internal/ssh/executor.go - SSH execution patterns
- internal/ssh/gpu_status.go - nvidia-smi parsing
"""

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, List, Tuple

import asyncssh

from gpu_deploy_llm.utils.errors import SSHConnectionError

logger = logging.getLogger(__name__)

# Timeouts matching shopper's ssh/executor.go
DEFAULT_CONNECT_TIMEOUT = 30  # DefaultExecutorConnectTimeout
DEFAULT_COMMAND_TIMEOUT = 60  # DefaultExecutorCommandTimeout


@dataclass
class GPUStatus:
    """GPU status from nvidia-smi.

    Matches internal/ssh/gpu_status.go GPUStatus struct.
    """

    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int
    temperature_c: int
    power_draw_w: int

    def is_healthy(self) -> bool:
        """Check if GPU is in healthy state."""
        return (
            self.temperature_c < 90 and self.memory_used_mb < self.memory_total_mb
        )

    @property
    def memory_free_mb(self) -> int:
        """Get free memory in MB."""
        return self.memory_total_mb - self.memory_used_mb

    @property
    def memory_used_pct(self) -> float:
        """Get memory usage percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


@dataclass
class CommandResult:
    """Result of SSH command execution."""

    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class SSHConnection:
    """SSH connection manager with secure key handling.

    Uses temporary files for SSH keys with 0600 permissions.
    Keys are deleted after connection is closed.

    Usage:
        async with SSHConnection(host, port, user, private_key) as ssh:
            result = await ssh.run("nvidia-smi")
            if result.success:
                print(result.stdout)
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        private_key: str,
        connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
        command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ):
        """Initialize SSH connection parameters.

        Args:
            host: SSH hostname or IP
            port: SSH port
            user: SSH username
            private_key: SSH private key content (PEM format)
            connect_timeout: Connection timeout in seconds
            command_timeout: Command execution timeout in seconds
        """
        self.host = host
        self.port = port
        self.user = user
        self._private_key = private_key
        self.connect_timeout = connect_timeout
        self.command_timeout = command_timeout
        self._conn: Optional[asyncssh.SSHClientConnection] = None
        self._key_file: Optional[str] = None

    async def __aenter__(self) -> "SSHConnection":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self) -> None:
        """Establish SSH connection."""
        # Write key to secure temporary file
        fd, self._key_file = tempfile.mkstemp(prefix="ssh_key_", suffix=".pem")
        try:
            os.chmod(self._key_file, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(self._private_key)
        except Exception:
            os.close(fd)
            if self._key_file and os.path.exists(self._key_file):
                os.unlink(self._key_file)
            raise

        try:
            # Load key from file
            key = asyncssh.read_private_key(self._key_file)

            # Connect
            self._conn = await asyncio.wait_for(
                asyncssh.connect(
                    self.host,
                    port=self.port,
                    username=self.user,
                    client_keys=[key],
                    known_hosts=None,  # Accept any host key (cloud instances)
                ),
                timeout=self.connect_timeout,
            )
            logger.info(f"SSH connected to {self.user}@{self.host}:{self.port}")
        except asyncio.TimeoutError:
            raise SSHConnectionError(
                f"SSH connection timeout ({self.connect_timeout}s)",
                host=self.host,
                port=self.port,
            )
        except asyncssh.Error as e:
            raise SSHConnectionError(
                f"SSH connection failed: {e}",
                host=self.host,
                port=self.port,
            )
        except Exception as e:
            raise SSHConnectionError(
                f"SSH connection error: {e}",
                host=self.host,
                port=self.port,
            )

    async def close(self) -> None:
        """Close SSH connection and cleanup key file."""
        if self._conn:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None

        # Securely delete key file
        if self._key_file and os.path.exists(self._key_file):
            try:
                os.unlink(self._key_file)
                logger.debug(f"Deleted temporary key file: {self._key_file}")
            except Exception as e:
                logger.warning(f"Failed to delete key file: {e}")
            self._key_file = None

    async def run(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> CommandResult:
        """Execute a command over SSH.

        Args:
            command: Command to execute
            timeout: Command timeout (default: self.command_timeout)

        Returns:
            CommandResult with stdout, stderr, exit_code
        """
        if not self._conn:
            raise SSHConnectionError("Not connected", host=self.host, port=self.port)

        timeout = timeout or self.command_timeout

        try:
            result = await asyncio.wait_for(
                self._conn.run(command, check=False),
                timeout=timeout,
            )
            return CommandResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=result.exit_status or 0,
            )
        except asyncio.TimeoutError:
            raise SSHConnectionError(
                f"Command timeout ({timeout}s): {command[:50]}...",
                host=self.host,
                port=self.port,
            )
        except asyncssh.Error as e:
            raise SSHConnectionError(
                f"Command failed: {e}",
                host=self.host,
                port=self.port,
            )

    async def run_checked(
        self,
        command: str,
        timeout: Optional[int] = None,
    ) -> str:
        """Execute a command and raise on non-zero exit.

        Args:
            command: Command to execute
            timeout: Command timeout

        Returns:
            Command stdout

        Raises:
            SSHConnectionError: On non-zero exit code
        """
        result = await self.run(command, timeout)
        if not result.success:
            raise SSHConnectionError(
                f"Command failed (exit {result.exit_code}): {result.stderr}",
                host=self.host,
                port=self.port,
            )
        return result.stdout

    async def get_gpu_status(self) -> List[GPUStatus]:
        """Get GPU status via nvidia-smi.

        Uses same query format as shopper (internal/ssh/gpu_status.go).

        Returns:
            List of GPUStatus objects
        """
        # Same query as shopper's gpu_status.go
        command = (
            "nvidia-smi --query-gpu=name,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu,power.draw "
            "--format=csv,noheader,nounits"
        )

        result = await self.run(command)
        if not result.success:
            raise SSHConnectionError(
                f"nvidia-smi failed: {result.stderr}",
                host=self.host,
                port=self.port,
            )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 6:
                logger.warning(f"Unexpected nvidia-smi output: {line}")
                continue

            try:
                gpus.append(
                    GPUStatus(
                        name=parts[0],
                        memory_used_mb=int(float(parts[1])),
                        memory_total_mb=int(float(parts[2])),
                        utilization_pct=int(float(parts[3])),
                        temperature_c=int(float(parts[4])),
                        power_draw_w=int(float(parts[5])),
                    )
                )
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse nvidia-smi line '{line}': {e}")

        return gpus

    async def check_docker(self) -> bool:
        """Check if Docker is available."""
        result = await self.run("docker --version")
        return result.success

    async def get_running_containers(self) -> List[str]:
        """Get list of running container names."""
        result = await self.run("docker ps --format '{{.Names}}'")
        if not result.success:
            return []
        return [name.strip() for name in result.stdout.strip().split("\n") if name.strip()]

    async def get_container_logs(
        self,
        container_name: str,
        tail: int = 100,
    ) -> str:
        """Get container logs.

        Args:
            container_name: Container name
            tail: Number of lines to retrieve

        Returns:
            Container logs
        """
        result = await self.run(f"docker logs --tail {tail} {container_name}")
        # Combine stdout and stderr (docker logs goes to stderr for some containers)
        return result.stdout + result.stderr

    async def get_processes(self) -> str:
        """Get process listing."""
        result = await self.run("ps aux | head -50")
        return result.stdout if result.success else ""

    async def get_network_info(self) -> str:
        """Get network information."""
        result = await self.run("netstat -tlnp 2>/dev/null || ss -tlnp")
        return result.stdout if result.success else ""

    async def get_disk_usage(self) -> str:
        """Get disk usage."""
        result = await self.run("df -h")
        return result.stdout if result.success else ""

    async def get_memory_info(self) -> str:
        """Get memory information."""
        result = await self.run("free -h")
        return result.stdout if result.success else ""

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        result = await self.run(f"test -f {path} && echo 'exists'")
        return result.success and "exists" in result.stdout
