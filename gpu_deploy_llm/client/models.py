"""Pydantic models for cloud-gpu-shopper API types.

Based on:
- pkg/models/session.go - Session model, status constants
- pkg/models/gpu.go - GPU offer model structure
- internal/api/handlers.go - Request/response types
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Session lifecycle statuses.

    Lifecycle: pending → provisioning → running → stopping → stopped
                            ↓
                          failed
    """

    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class StoragePolicy(str, Enum):
    """Storage policy for session cleanup."""

    PRESERVE = "preserve"
    DESTROY = "destroy"


class LaunchMode(str, Enum):
    """Launch mode for session deployment.

    SSH mode: Interactive SSH access, deploy workloads manually via pip/scripts
    Entrypoint mode: Pre-built Docker image runs automatically (recommended)
    """

    SSH = "ssh"
    ENTRYPOINT = "entrypoint"


class WorkloadType(str, Enum):
    """Workload type for the session."""

    LLM = "llm"
    LLM_VLLM = "llm_vllm"
    LLM_TGI = "llm_tgi"
    TRAINING = "training"
    BATCH = "batch"
    INTERACTIVE = "interactive"


class GPUOffer(BaseModel):
    """GPU offer from cloud-gpu-shopper inventory.

    Matches pkg/models/gpu.go GPUOffer struct.
    """

    id: str = Field(..., description="Unique offer ID")
    provider: str = Field(..., description="Provider name (vastai, tensordock)")
    gpu_type: str = Field(..., description="GPU model name")
    gpu_count: int = Field(..., description="Number of GPUs")
    vram_gb: float = Field(..., description="VRAM per GPU in GB")
    price_per_hour: float = Field(..., description="Hourly price in USD")
    region: Optional[str] = Field(None, description="Geographic region")
    availability: Optional[str] = Field(None, description="Availability status")

    # Additional metadata that may be present
    cpu_cores: Optional[int] = None
    ram_gb: Optional[float] = None
    disk_gb: Optional[float] = None
    bandwidth_gbps: Optional[float] = None


class Session(BaseModel):
    """Session from cloud-gpu-shopper.

    Matches pkg/models/session.go Session struct (response version).
    """

    id: str = Field(..., description="Unique session ID")
    consumer_id: str = Field(..., description="Consumer identifier")
    status: SessionStatus = Field(..., description="Current session status")
    provider: str = Field(..., description="Provider name")
    offer_id: Optional[str] = Field(None, description="Original offer ID")

    # GPU details
    gpu_type: str = Field(..., description="GPU model name")
    gpu_count: int = Field(..., description="Number of GPUs")

    # SSH connection details (may be empty until running)
    ssh_host: Optional[str] = Field(None, description="SSH hostname/IP")
    ssh_port: Optional[int] = Field(None, description="SSH port")
    ssh_user: Optional[str] = Field(None, description="SSH username")

    # API endpoint details (entrypoint mode)
    launch_mode: Optional[str] = Field(None, description="Launch mode: ssh or entrypoint")
    api_endpoint: Optional[str] = Field(None, description="Full URL to API (entrypoint mode)")
    api_port: Optional[int] = Field(None, description="Mapped API port (entrypoint mode)")
    model_id: Optional[str] = Field(None, description="HuggingFace model ID")

    # Timing
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

    # Cost
    price_per_hour: float = Field(0.0, description="Hourly price")

    # Error info (present when status is FAILED)
    error: Optional[str] = Field(None, description="Error message if failed")

    # Workload configuration
    workload_type: Optional[str] = Field(None, description="Workload type")
    storage_policy: Optional[StoragePolicy] = Field(None, description="Storage policy")

    @property
    def is_terminal(self) -> bool:
        """Check if session is in a terminal state."""
        return self.status in (SessionStatus.STOPPED, SessionStatus.FAILED)

    @property
    def is_connectable(self) -> bool:
        """Check if session has SSH connection details."""
        return bool(self.ssh_host and self.ssh_port and self.ssh_user)


class InventoryFilter(BaseModel):
    """Filter parameters for inventory query."""

    min_vram: Optional[float] = Field(None, description="Minimum VRAM in GB")
    max_price: Optional[float] = Field(None, description="Maximum hourly price")
    provider: Optional[str] = Field(None, description="Filter by provider")
    gpu_type: Optional[str] = Field(None, description="Filter by GPU type")
    min_gpu_count: Optional[int] = Field(None, description="Minimum GPU count")


class CreateSessionRequest(BaseModel):
    """Request body for POST /api/v1/sessions.

    Matches internal/api/handlers.go CreateSessionRequest struct.

    Two deployment modes:
    1. SSH mode (default): Get SSH access, deploy workloads manually
    2. Entrypoint mode: Pre-built Docker image runs automatically (recommended)

    For entrypoint mode with vLLM:
        CreateSessionRequest(
            launch_mode=LaunchMode.ENTRYPOINT,
            docker_image="vllm/vllm-openai:latest",
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            exposed_ports=[8000],
            ...
        )
    """

    offer_id: str = Field(..., description="ID of the offer to provision")
    consumer_id: str = Field(..., description="Consumer identifier with version")
    workload_type: str = Field(
        "llm_vllm", description="Workload type (required: llm_vllm)"
    )
    reservation_hours: int = Field(
        1,
        ge=1,
        le=12,
        description="Reservation duration (1-12 hours)",
    )
    storage_policy: StoragePolicy = Field(
        StoragePolicy.DESTROY,
        description="Storage policy on cleanup",
    )
    idle_threshold_minutes: Optional[int] = Field(
        None, description="Idle timeout in minutes"
    )

    # Entrypoint mode configuration
    launch_mode: Optional[LaunchMode] = Field(
        None,
        description="Launch mode: 'ssh' (default) or 'entrypoint' (Docker-based)",
    )
    docker_image: Optional[str] = Field(
        None,
        description="Docker image for entrypoint mode (e.g., 'vllm/vllm-openai:latest')",
    )
    model_id: Optional[str] = Field(
        None,
        description="HuggingFace model ID for entrypoint mode",
    )
    exposed_ports: Optional[List[int]] = Field(
        None,
        description="Ports to expose (e.g., [8000] for vLLM)",
    )
    quantization: Optional[str] = Field(
        None,
        description="Quantization method (awq, gptq)",
    )


class CreateSessionResponse(BaseModel):
    """Response from POST /api/v1/sessions.

    IMPORTANT: ssh_private_key is ONLY returned here - capture immediately!
    """

    session: Session = Field(..., description="Created session")
    ssh_private_key: str = Field(
        ...,
        description="SSH private key (ONLY returned at creation!)",
    )


class SessionDiagnostics(BaseModel):
    """Diagnostics response from GET /api/v1/sessions/:id/diagnostics.

    Note: GPU health checks require client-side SSH access since
    the private key is not stored server-side.
    """

    session_id: str
    status: SessionStatus
    provider: str
    gpu_type: str
    gpu_count: int
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: Optional[str] = None
    uptime: Optional[str] = None
    time_to_expiry: Optional[str] = None
    ssh_available: Optional[bool] = None
    note: Optional[str] = None


class CostInfo(BaseModel):
    """Cost information from GET /api/v1/costs."""

    session_id: Optional[str] = None
    consumer_id: Optional[str] = None
    total_cost: float = Field(..., description="Total cost in USD")
    duration_hours: float = Field(..., description="Total duration in hours")
    breakdown: Optional[List[dict]] = Field(
        None, description="Per-session cost breakdown"
    )


class ExtendSessionRequest(BaseModel):
    """Request body for POST /api/v1/sessions/:id/extend."""

    additional_hours: int = Field(
        ...,
        ge=1,
        le=12,
        description="Hours to add (1-12)",
    )


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    version: Optional[str] = None
    providers: Optional[dict] = None


class ReadyResponse(BaseModel):
    """Response from GET /ready."""

    ready: bool
    message: Optional[str] = None
