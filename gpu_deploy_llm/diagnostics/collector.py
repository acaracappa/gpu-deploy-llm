"""Diagnostic snapshot collection for debugging cloud-gpu-shopper.

Snapshots are collected at key checkpoints:
- shopper_ready: After /ready returns 200
- session_created: Immediately after POST /sessions
- session_running: When status transitions to running
- ssh_connected: Initial instance state
- deployment_started: Before vLLM deploy
- deployment_complete: After successful deploy
- error: On any failure (ALWAYS collect before cleanup)

Output: ./diagnostics/<session_id>/<timestamp>_<label>.json
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Checkpoint(str, Enum):
    """Diagnostic checkpoint labels."""

    SHOPPER_READY = "shopper_ready"
    SESSION_CREATED = "session_created"
    SESSION_RUNNING = "session_running"
    SSH_CONNECTED = "ssh_connected"
    DEPLOYMENT_STARTED = "deployment_started"
    DEPLOYMENT_COMPLETE = "deployment_complete"
    ERROR = "error"


@dataclass
class APITrace:
    """Trace of an API call."""

    method: str
    path: str
    status: int
    duration_ms: float
    request_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TimingInfo:
    """Timing information for diagnostics."""

    time_since_creation: str = ""
    time_since_last_checkpoint: str = ""
    provisioning_duration: str = ""


@dataclass
class ShopperState:
    """State information from shopper."""

    session_diagnostics: Optional[Dict] = None
    shopper_health: Optional[Dict] = None


@dataclass
class InstanceState:
    """State information from the GPU instance."""

    nvidia_smi: str = ""
    processes: str = ""
    network: str = ""
    container_logs: str = ""
    disk_usage: str = ""
    memory_info: str = ""
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostInfo:
    """Cost information."""

    session_cost_so_far: float = 0.0


@dataclass
class DiagnosticSnapshot:
    """Complete diagnostic snapshot at a checkpoint."""

    timestamp: str
    checkpoint: str
    session_id: str

    timing: TimingInfo = field(default_factory=TimingInfo)
    shopper_state: ShopperState = field(default_factory=ShopperState)
    api_trace: List[APITrace] = field(default_factory=list)
    instance_state: InstanceState = field(default_factory=InstanceState)
    costs: CostInfo = field(default_factory=CostInfo)
    errors: List[str] = field(default_factory=list)

    # Additional metadata
    provider: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    model_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class DiagnosticCollector:
    """Collects and stores diagnostic snapshots.

    Usage:
        collector = DiagnosticCollector(session_id="sess-123")

        # Collect at checkpoints
        await collector.collect(
            Checkpoint.SESSION_CREATED,
            shopper_client=client,
            ssh=None,  # Not connected yet
        )

        await collector.collect(
            Checkpoint.SSH_CONNECTED,
            shopper_client=client,
            ssh=ssh_connection,
        )
    """

    def __init__(
        self,
        session_id: str,
        output_dir: str = "./diagnostics",
        model_id: str = "",
        provider: str = "",
        gpu_type: str = "",
        gpu_count: int = 0,
    ):
        """Initialize collector.

        Args:
            session_id: Session ID for organizing snapshots
            output_dir: Base output directory
            model_id: Model being deployed
            provider: GPU provider name
            gpu_type: GPU type
            gpu_count: Number of GPUs
        """
        self.session_id = session_id
        self.output_dir = Path(output_dir) / session_id
        self.model_id = model_id
        self.provider = provider
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count

        self._creation_time = datetime.utcnow()
        self._last_checkpoint_time = self._creation_time
        self._api_traces: List[APITrace] = []
        self._snapshots: List[DiagnosticSnapshot] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_api_trace(
        self,
        method: str,
        path: str,
        status: int,
        duration_ms: float,
        request_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Add an API call trace."""
        self._api_traces.append(
            APITrace(
                method=method,
                path=path,
                status=status,
                duration_ms=duration_ms,
                request_id=request_id,
                error=error,
            )
        )

    async def collect(
        self,
        checkpoint: Checkpoint,
        shopper_client=None,
        ssh=None,
        error: Optional[str] = None,
        container_name: str = "vllm-server",
    ) -> DiagnosticSnapshot:
        """Collect diagnostic snapshot at a checkpoint.

        Args:
            checkpoint: Checkpoint label
            shopper_client: ShopperClient instance (optional)
            ssh: SSHConnection instance (optional)
            error: Error message if this is an error checkpoint
            container_name: Container name for log collection

        Returns:
            DiagnosticSnapshot
        """
        now = datetime.utcnow()

        snapshot = DiagnosticSnapshot(
            timestamp=now.isoformat() + "Z",
            checkpoint=checkpoint.value,
            session_id=self.session_id,
            provider=self.provider,
            gpu_type=self.gpu_type,
            gpu_count=self.gpu_count,
            model_id=self.model_id,
        )

        # Timing info
        snapshot.timing = TimingInfo(
            time_since_creation=self._format_duration(now - self._creation_time),
            time_since_last_checkpoint=self._format_duration(
                now - self._last_checkpoint_time
            ),
        )

        # API traces (copy and clear)
        snapshot.api_trace = self._api_traces.copy()

        # Collect shopper state
        if shopper_client:
            snapshot.shopper_state = await self._collect_shopper_state(shopper_client)

        # Collect instance state via SSH
        if ssh:
            snapshot.instance_state = await self._collect_instance_state(
                ssh, container_name
            )

        # Add error if present
        if error:
            snapshot.errors.append(error)

        # Update timing
        self._last_checkpoint_time = now

        # Save snapshot
        self._snapshots.append(snapshot)
        self._save_snapshot(snapshot)

        logger.info(f"Collected diagnostic snapshot: {checkpoint.value}")
        return snapshot

    async def _collect_shopper_state(self, client) -> ShopperState:
        """Collect state from shopper API."""
        state = ShopperState()

        try:
            # Get session diagnostics
            diag = await client.get_session_diagnostics(self.session_id)
            state.session_diagnostics = diag.model_dump() if hasattr(diag, 'model_dump') else asdict(diag)
        except Exception as e:
            logger.debug(f"Failed to get session diagnostics: {e}")

        try:
            # Get shopper health
            health = await client.health_check()
            state.shopper_health = health.model_dump() if hasattr(health, 'model_dump') else asdict(health)
        except Exception as e:
            logger.debug(f"Failed to get shopper health: {e}")

        return state

    async def _collect_instance_state(
        self,
        ssh,
        container_name: str,
    ) -> InstanceState:
        """Collect state from GPU instance via SSH."""
        state = InstanceState()

        try:
            # GPU status
            gpus = await ssh.get_gpu_status()
            state.nvidia_smi = "\n".join(
                f"{g.name}: {g.memory_used_mb}/{g.memory_total_mb}MB, "
                f"{g.utilization_pct}% util, {g.temperature_c}°C"
                for g in gpus
            )
        except Exception as e:
            state.nvidia_smi = f"Error: {e}"

        try:
            state.processes = await ssh.get_processes()
        except Exception as e:
            state.processes = f"Error: {e}"

        try:
            state.network = await ssh.get_network_info()
        except Exception as e:
            state.network = f"Error: {e}"

        try:
            state.container_logs = await ssh.get_container_logs(container_name)
        except Exception as e:
            state.container_logs = f"Error: {e}"

        try:
            state.disk_usage = await ssh.get_disk_usage()
        except Exception as e:
            state.disk_usage = f"Error: {e}"

        try:
            state.memory_info = await ssh.get_memory_info()
        except Exception as e:
            state.memory_info = f"Error: {e}"

        return state

    def _save_snapshot(self, snapshot: DiagnosticSnapshot) -> Path:
        """Save snapshot to file."""
        filename = f"{snapshot.timestamp.replace(':', '-')}_{snapshot.checkpoint}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write(snapshot.to_json())

        logger.debug(f"Saved snapshot to {filepath}")
        return filepath

    @staticmethod
    def _format_duration(delta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def get_all_snapshots(self) -> List[DiagnosticSnapshot]:
        """Get all collected snapshots."""
        return self._snapshots.copy()

    def get_output_path(self) -> Path:
        """Get the output directory path."""
        return self.output_dir
