"""Host diagnostics via SSH.

Provides comprehensive visibility into the remote host:
- OS/distro identification
- GPU status and driver info
- Docker availability
- Python environment
- System resources
- Running processes
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class OSInfo:
    """Operating system information."""
    name: str = "unknown"
    version: str = "unknown"
    distro: str = "unknown"
    kernel: str = "unknown"
    arch: str = "unknown"


@dataclass
class GPUInfo:
    """GPU information."""
    name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    temperature_c: int = 0
    utilization_pct: int = 0


@dataclass
class DockerInfo:
    """Docker availability info."""
    available: bool = False
    version: str = ""
    running_containers: int = 0
    error: Optional[str] = None


@dataclass
class PythonInfo:
    """Python environment info."""
    available: bool = False
    version: str = ""
    path: str = ""
    pip_available: bool = False
    pip_version: str = ""
    vllm_installed: bool = False
    vllm_version: str = ""


@dataclass
class SystemResources:
    """System resource info."""
    cpu_cores: int = 0
    memory_total_gb: float = 0
    memory_available_gb: float = 0
    disk_total_gb: float = 0
    disk_available_gb: float = 0


@dataclass
class HostDiagnostics:
    """Complete host diagnostics."""

    # Connection status
    ssh_connected: bool = False
    shell_functional: bool = False
    connection_error: Optional[str] = None

    # System info
    hostname: str = ""
    os_info: OSInfo = field(default_factory=OSInfo)
    uptime: str = ""

    # GPU
    gpus: List[GPUInfo] = field(default_factory=list)
    nvidia_smi_available: bool = False

    # Docker
    docker: DockerInfo = field(default_factory=DockerInfo)

    # Python
    python: PythonInfo = field(default_factory=PythonInfo)

    # Resources
    resources: SystemResources = field(default_factory=SystemResources)

    # Network
    can_reach_internet: bool = False
    can_reach_huggingface: bool = False

    # Raw command outputs for debugging
    raw_outputs: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ssh_connected": self.ssh_connected,
            "shell_functional": self.shell_functional,
            "connection_error": self.connection_error,
            "hostname": self.hostname,
            "os": {
                "name": self.os_info.name,
                "version": self.os_info.version,
                "distro": self.os_info.distro,
                "kernel": self.os_info.kernel,
                "arch": self.os_info.arch,
            },
            "uptime": self.uptime,
            "gpus": [
                {
                    "name": g.name,
                    "driver": g.driver_version,
                    "cuda": g.cuda_version,
                    "memory_total_mb": g.memory_total_mb,
                    "memory_used_mb": g.memory_used_mb,
                    "temp_c": g.temperature_c,
                    "util_pct": g.utilization_pct,
                }
                for g in self.gpus
            ],
            "nvidia_smi": self.nvidia_smi_available,
            "docker": {
                "available": self.docker.available,
                "version": self.docker.version,
                "containers": self.docker.running_containers,
                "error": self.docker.error,
            },
            "python": {
                "available": self.python.available,
                "version": self.python.version,
                "pip": self.python.pip_available,
                "vllm_installed": self.python.vllm_installed,
                "vllm_version": self.python.vllm_version,
            },
            "resources": {
                "cpu_cores": self.resources.cpu_cores,
                "memory_total_gb": round(self.resources.memory_total_gb, 1),
                "memory_available_gb": round(self.resources.memory_available_gb, 1),
                "disk_total_gb": round(self.resources.disk_total_gb, 1),
                "disk_available_gb": round(self.resources.disk_available_gb, 1),
            },
            "network": {
                "internet": self.can_reach_internet,
                "huggingface": self.can_reach_huggingface,
            },
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = []

        # Connection
        if self.ssh_connected and self.shell_functional:
            lines.append(f"✓ SSH connected to {self.hostname}")
        else:
            lines.append(f"✗ SSH connection failed: {self.connection_error}")
            return "\n".join(lines)

        # OS
        lines.append(f"  OS: {self.os_info.distro} {self.os_info.version} ({self.os_info.kernel})")

        # GPUs
        if self.gpus:
            for i, gpu in enumerate(self.gpus):
                lines.append(f"  GPU {i}: {gpu.name} ({gpu.memory_total_mb}MB, {gpu.temperature_c}°C)")
            lines.append(f"  Driver: {self.gpus[0].driver_version}, CUDA: {self.gpus[0].cuda_version}")
        else:
            lines.append("  GPU: Not detected")

        # Docker (optional - we use pip deployment by default)
        if self.docker.available:
            lines.append(f"  Docker: {self.docker.version} ({self.docker.running_containers} containers)")
        else:
            lines.append(f"  Docker: Not available - {self.docker.error} (not required)")

        # Python
        if self.python.available:
            vllm = f", vLLM {self.python.vllm_version}" if self.python.vllm_installed else ""
            lines.append(f"  Python: {self.python.version}{vllm}")
        else:
            lines.append("  Python: Not available")

        # Resources
        lines.append(f"  Memory: {self.resources.memory_available_gb:.1f}/{self.resources.memory_total_gb:.1f} GB available")
        lines.append(f"  Disk: {self.resources.disk_available_gb:.1f}/{self.resources.disk_total_gb:.1f} GB available")

        # Network
        net_status = []
        if self.can_reach_internet:
            net_status.append("internet ✓")
        if self.can_reach_huggingface:
            net_status.append("huggingface ✓")
        if net_status:
            lines.append(f"  Network: {', '.join(net_status)}")

        return "\n".join(lines)


class HostDiagnosticsCollector:
    """Collects diagnostics from a remote host via SSH."""

    def __init__(self, ssh_connection):
        """Initialize with an SSH connection.

        Args:
            ssh_connection: SSHConnection instance
        """
        self.ssh = ssh_connection

    async def collect_all(self, progress_callback=None) -> HostDiagnostics:
        """Collect all diagnostics.

        Args:
            progress_callback: Optional async callback(step, message) for progress updates

        Returns:
            HostDiagnostics with all collected information
        """
        diag = HostDiagnostics()

        async def log(step: str, msg: str):
            logger.info(f"[diag] {step}: {msg}")
            if progress_callback:
                await progress_callback(step, msg)

        # Test basic connectivity
        await log("shell", "Testing shell access...")
        try:
            result = await self.ssh.run("echo 'shell_test_ok'")
            if result.success and "shell_test_ok" in result.stdout:
                diag.ssh_connected = True
                diag.shell_functional = True
                await log("shell", "Shell functional ✓")
            else:
                diag.connection_error = f"Shell test failed: {result.stderr}"
                await log("shell", f"Shell test failed: {result.stderr}")
                return diag
        except Exception as e:
            diag.connection_error = str(e)
            await log("shell", f"Connection error: {e}")
            return diag

        # Hostname
        await log("hostname", "Getting hostname...")
        result = await self.ssh.run("hostname")
        if result.success:
            diag.hostname = result.stdout.strip()
            diag.raw_outputs["hostname"] = result.stdout

        # OS Info
        await log("os", "Getting OS information...")
        diag.os_info = await self._get_os_info()

        # Uptime
        result = await self.ssh.run("uptime -p 2>/dev/null || uptime")
        if result.success:
            diag.uptime = result.stdout.strip()

        # GPU Info
        await log("gpu", "Checking GPU status...")
        diag.nvidia_smi_available, diag.gpus = await self._get_gpu_info()
        if diag.gpus:
            await log("gpu", f"Found {len(diag.gpus)} GPU(s): {diag.gpus[0].name}")
        else:
            await log("gpu", "No GPUs detected via nvidia-smi")

        # Docker
        await log("docker", "Checking Docker availability...")
        diag.docker = await self._get_docker_info()
        if diag.docker.available:
            await log("docker", f"Docker {diag.docker.version} available")
        else:
            await log("docker", f"Docker not available: {diag.docker.error}")

        # Python
        await log("python", "Checking Python environment...")
        diag.python = await self._get_python_info()
        if diag.python.available:
            await log("python", f"Python {diag.python.version} available")
        else:
            await log("python", "Python not available")

        # System resources
        await log("resources", "Checking system resources...")
        diag.resources = await self._get_system_resources()

        # Network connectivity
        await log("network", "Testing network connectivity...")
        diag.can_reach_internet = await self._check_connectivity("8.8.8.8")
        diag.can_reach_huggingface = await self._check_connectivity("huggingface.co", use_curl=True)

        await log("complete", "Diagnostics collection complete")
        return diag

    async def _get_os_info(self) -> OSInfo:
        """Get OS information."""
        info = OSInfo()

        # Try /etc/os-release first
        result = await self.ssh.run("cat /etc/os-release 2>/dev/null")
        if result.success:
            for line in result.stdout.split("\n"):
                if line.startswith("NAME="):
                    info.name = line.split("=", 1)[1].strip('"')
                elif line.startswith("VERSION="):
                    info.version = line.split("=", 1)[1].strip('"')
                elif line.startswith("ID="):
                    info.distro = line.split("=", 1)[1].strip('"')

        # Kernel version
        result = await self.ssh.run("uname -r")
        if result.success:
            info.kernel = result.stdout.strip()

        # Architecture
        result = await self.ssh.run("uname -m")
        if result.success:
            info.arch = result.stdout.strip()

        return info

    async def _get_gpu_info(self) -> tuple:
        """Get GPU information via nvidia-smi."""
        gpus = []

        # Check if nvidia-smi is available
        result = await self.ssh.run("which nvidia-smi")
        if not result.success:
            return False, []

        # Get GPU info
        result = await self.ssh.run(
            "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,utilization.gpu "
            "--format=csv,noheader,nounits"
        )

        if not result.success:
            return True, []  # nvidia-smi exists but query failed

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpu = GPUInfo(
                    name=parts[0],
                    driver_version=parts[1],
                    memory_total_mb=int(float(parts[2])) if parts[2] else 0,
                    memory_used_mb=int(float(parts[3])) if parts[3] else 0,
                    temperature_c=int(float(parts[4])) if parts[4] else 0,
                    utilization_pct=int(float(parts[5])) if parts[5] else 0,
                )
                gpus.append(gpu)

        # Get CUDA version
        if gpus:
            result = await self.ssh.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
            result2 = await self.ssh.run("nvcc --version 2>/dev/null | grep 'release' | awk '{print $5}' | tr -d ','")
            if result2.success and result2.stdout.strip():
                for gpu in gpus:
                    gpu.cuda_version = result2.stdout.strip()

        return True, gpus

    async def _get_docker_info(self) -> DockerInfo:
        """Get Docker information."""
        info = DockerInfo()

        # Check if docker is available
        result = await self.ssh.run("docker --version 2>&1")
        if not result.success or "not found" in result.stdout.lower():
            info.error = "Docker not installed"
            return info

        # Parse version
        if "Docker version" in result.stdout:
            info.available = True
            try:
                info.version = result.stdout.split("Docker version")[1].split(",")[0].strip()
            except:
                info.version = "unknown"

        # Check if docker daemon is running
        result = await self.ssh.run("docker ps 2>&1")
        if not result.success:
            if "permission denied" in result.stdout.lower() or "permission denied" in result.stderr.lower():
                info.error = "Permission denied (user not in docker group)"
            elif "connect" in result.stdout.lower() or "connect" in result.stderr.lower():
                info.error = "Docker daemon not running"
            else:
                info.error = result.stderr or result.stdout
            info.available = False
            return info

        # Count running containers
        result = await self.ssh.run("docker ps -q | wc -l")
        if result.success:
            try:
                info.running_containers = int(result.stdout.strip())
            except:
                pass

        return info

    async def _get_python_info(self) -> PythonInfo:
        """Get Python environment information."""
        info = PythonInfo()

        # Find Python
        for cmd in ["python3", "python"]:
            result = await self.ssh.run(f"{cmd} --version 2>&1")
            if result.success and "Python" in result.stdout:
                info.available = True
                info.version = result.stdout.replace("Python", "").strip()

                # Get path
                result2 = await self.ssh.run(f"which {cmd}")
                if result2.success:
                    info.path = result2.stdout.strip()
                break

        if not info.available:
            return info

        # Check pip
        result = await self.ssh.run(f"{info.path} -m pip --version 2>&1")
        if result.success and "pip" in result.stdout:
            info.pip_available = True
            try:
                info.pip_version = result.stdout.split("pip")[1].split("from")[0].strip()
            except:
                info.pip_version = "unknown"

        # Check vLLM
        result = await self.ssh.run(f"{info.path} -c \"import vllm; print(vllm.__version__)\" 2>&1")
        if result.success and result.returncode == 0:
            info.vllm_installed = True
            info.vllm_version = result.stdout.strip()

        return info

    async def _get_system_resources(self) -> SystemResources:
        """Get system resource information."""
        res = SystemResources()

        # CPU cores
        result = await self.ssh.run("nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null")
        if result.success:
            try:
                res.cpu_cores = int(result.stdout.strip())
            except:
                pass

        # Memory
        result = await self.ssh.run("free -b 2>/dev/null | grep Mem")
        if result.success:
            parts = result.stdout.split()
            if len(parts) >= 4:
                try:
                    res.memory_total_gb = int(parts[1]) / (1024**3)
                    res.memory_available_gb = int(parts[3]) / (1024**3) if len(parts) > 6 else int(parts[3]) / (1024**3)
                except:
                    pass

        # Disk
        result = await self.ssh.run("df -B1 / 2>/dev/null | tail -1")
        if result.success:
            parts = result.stdout.split()
            if len(parts) >= 4:
                try:
                    res.disk_total_gb = int(parts[1]) / (1024**3)
                    res.disk_available_gb = int(parts[3]) / (1024**3)
                except:
                    pass

        return res

    async def _check_connectivity(self, host: str, use_curl: bool = False) -> bool:
        """Check network connectivity to a host."""
        if use_curl:
            result = await self.ssh.run(f"curl -s --connect-timeout 5 -o /dev/null -w '%{{http_code}}' https://{host}", timeout=10)
            return result.success and result.stdout.strip().startswith("2")
        else:
            result = await self.ssh.run(f"ping -c 1 -W 3 {host} 2>/dev/null", timeout=10)
            return result.success
