"""REST client for cloud-gpu-shopper API.

Implements all endpoints from the API:
- /health - Service health check
- /ready - Readiness check (503 during startup sweep)
- /api/v1/inventory - List GPU offers
- /api/v1/sessions - Session management
- /api/v1/costs - Cost tracking
"""

import logging
from typing import Optional, List
from urllib.parse import urljoin

import httpx

from gpu_deploy_llm import __version__
from gpu_deploy_llm.utils.errors import (
    ShopperAPIError,
    StaleInventoryError,
    DuplicateSessionError,
    SessionFailedError,
    RateLimitError,
    ShopperNotReadyError,
)
from .models import (
    GPUOffer,
    Session,
    SessionStatus,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionDiagnostics,
    CostInfo,
    InventoryFilter,
    HealthResponse,
    ReadyResponse,
    ExtendSessionRequest,
    LaunchMode,
    WorkloadType,
    StoragePolicy,
)

logger = logging.getLogger(__name__)

# Default consumer ID with version for debugging
DEFAULT_CONSUMER_ID = f"gpu-deploy-llm/v{__version__}"

# Timeouts matching shopper's SSH executor (internal/ssh/executor.go)
DEFAULT_CONNECT_TIMEOUT = 30.0  # DefaultExecutorConnectTimeout
DEFAULT_COMMAND_TIMEOUT = 60.0  # DefaultExecutorCommandTimeout


class ShopperClient:
    """REST client for cloud-gpu-shopper API.

    Usage:
        async with ShopperClient("http://localhost:8080") as client:
            # Check service is ready
            await client.wait_for_ready()

            # Query inventory
            offers = await client.get_inventory(min_vram=16)

            # Create session
            response = await client.create_session(
                offer_id=offers[0].id,
                reservation_hours=2,
            )
            # IMPORTANT: Capture SSH key immediately!
            ssh_key = response.ssh_private_key
    """

    def __init__(
        self,
        base_url: str,
        consumer_id: str = DEFAULT_CONSUMER_ID,
        timeout: float = DEFAULT_COMMAND_TIMEOUT,
        debug: bool = False,
    ):
        """Initialize the shopper client.

        Args:
            base_url: Base URL of cloud-gpu-shopper (e.g., "http://localhost:8080")
            consumer_id: Consumer identifier (default: gpu-deploy-llm/v{version})
            timeout: Request timeout in seconds
            debug: Enable verbose request/response logging
        """
        self.base_url = base_url.rstrip("/")
        self.consumer_id = consumer_id
        self.timeout = timeout
        self.debug = debug
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "ShopperClient":
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=DEFAULT_CONNECT_TIMEOUT),
                headers={
                    "User-Agent": f"gpu-deploy-llm/{__version__}",
                    "Accept": "application/json",
                },
            )
            return self
        except Exception:
            # Ensure partial client is cleaned up
            if self._client:
                await self._client.aclose()
                self._client = None
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        """Explicitly close the client (for non-context-manager usage)."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return self._client

    def _log_request(self, method: str, path: str, **kwargs):
        """Log request details if debug mode enabled."""
        if self.debug:
            logger.info(f"Request: {method} {path}")
            if kwargs.get("json"):
                logger.info(f"Body: {kwargs['json']}")
            if kwargs.get("params"):
                logger.info(f"Params: {kwargs['params']}")

    def _log_response(self, response: httpx.Response):
        """Log response details if debug mode enabled."""
        if self.debug:
            logger.info(f"Response: {response.status_code}")
            try:
                logger.info(f"Body: {response.json()}")
            except Exception:
                logger.info(f"Body: {response.text[:500]}")

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP request with error handling."""
        self._log_request(method, path, **kwargs)

        try:
            response = await self.client.request(method, path, **kwargs)
        except httpx.TimeoutException as e:
            raise ShopperAPIError(
                f"Request timeout: {method} {path}",
                status_code=None,
            ) from e
        except httpx.ConnectError as e:
            raise ShopperAPIError(
                f"Connection failed: {self.base_url}",
                status_code=None,
            ) from e
        except httpx.RequestError as e:
            raise ShopperAPIError(
                f"Request error: {e}",
                status_code=None,
            ) from e

        self._log_response(response)

        return response

    def _handle_error(self, response: httpx.Response, context: str = ""):
        """Handle error responses from the API."""
        request_id = response.headers.get("X-Request-ID")

        try:
            data = response.json()
        except Exception:
            data = {"message": response.text}

        error_type = data.get("error_type")
        message = data.get("message", data.get("error", str(data)))

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_after_int = None
            if retry_after:
                try:
                    retry_after_int = int(retry_after)
                except ValueError:
                    logger.warning(f"Invalid Retry-After header: {retry_after}")
            raise RateLimitError(
                message=message,
                retry_after=retry_after_int,
                request_id=request_id,
            )

        if response.status_code == 503:
            if error_type == "stale_inventory":
                raise StaleInventoryError(
                    message=message,
                    offer_id=data.get("offer_id", ""),
                    request_id=request_id,
                )
            if data.get("ready") is False:
                raise ShopperNotReadyError(
                    message=message,
                    request_id=request_id,
                )

        if response.status_code == 409:
            raise DuplicateSessionError(
                message=message,
                existing_session_id=data.get("existing_session_id", ""),
                request_id=request_id,
            )

        raise ShopperAPIError(
            message=f"{context}: {message}" if context else message,
            status_code=response.status_code,
            request_id=request_id,
        )

    # Health & Readiness

    async def health_check(self) -> HealthResponse:
        """Check service health (GET /health)."""
        response = await self._request("GET", "/health")
        if response.status_code != 200:
            self._handle_error(response, "Health check failed")
        return HealthResponse(**response.json())

    async def ready_check(self) -> ReadyResponse:
        """Check service readiness (GET /ready).

        Returns 503 during startup sweep - use wait_for_ready() instead.
        """
        response = await self._request("GET", "/ready")
        if response.status_code == 503:
            data = response.json()
            return ReadyResponse(ready=False, message=data.get("message"))
        if response.status_code != 200:
            self._handle_error(response, "Ready check failed")
        return ReadyResponse(**response.json())

    async def wait_for_ready(
        self,
        timeout: float = 120.0,
        interval: float = 5.0,
    ) -> bool:
        """Wait for shopper to be ready.

        Polls /ready until it returns 200 or timeout.

        Args:
            timeout: Maximum wait time in seconds
            interval: Poll interval in seconds

        Returns:
            True if ready, raises TimeoutError otherwise
        """
        import asyncio
        import time

        start_time = time.time()
        deadline = start_time + timeout

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"Shopper not ready after {timeout}s")

            try:
                result = await self.ready_check()
                if result.ready:
                    return True
                logger.info(f"Shopper not ready: {result.message}")
            except asyncio.CancelledError:
                raise  # Re-raise CancelledError, don't swallow it
            except ShopperNotReadyError as e:
                logger.info(f"Shopper not ready: {e}")
            except Exception as e:
                logger.warning(f"Ready check error: {e}")

            # Sleep for interval or remaining time, whichever is smaller
            sleep_time = min(interval, remaining)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    # Inventory

    async def get_inventory(
        self,
        min_vram: Optional[float] = None,
        max_price: Optional[float] = None,
        provider: Optional[str] = None,
        gpu_type: Optional[str] = None,
        min_gpu_count: Optional[int] = None,
    ) -> List[GPUOffer]:
        """Get available GPU offers (GET /api/v1/inventory).

        Args:
            min_vram: Minimum VRAM in GB
            max_price: Maximum hourly price in USD
            provider: Filter by provider (vastai, tensordock)
            gpu_type: Filter by GPU type
            min_gpu_count: Minimum number of GPUs

        Returns:
            List of available GPU offers
        """
        params = {}
        if min_vram is not None:
            params["min_vram"] = int(min_vram)  # API expects integer
        if max_price is not None:
            params["max_price"] = max_price
        if provider is not None:
            params["provider"] = provider
        if gpu_type is not None:
            params["gpu_type"] = gpu_type
        if min_gpu_count is not None:
            params["min_gpu_count"] = min_gpu_count

        response = await self._request("GET", "/api/v1/inventory", params=params)

        if response.status_code != 200:
            self._handle_error(response, "Failed to get inventory")

        data = response.json()
        offers = data.get("offers", data) if isinstance(data, dict) else data

        return [GPUOffer(**offer) for offer in offers]

    # Session Management

    async def create_session(
        self,
        offer_id: str,
        reservation_hours: int = 1,
        workload_type: str = "llm_vllm",
        storage_policy: StoragePolicy = StoragePolicy.DESTROY,
        idle_threshold_minutes: Optional[int] = None,
        # Entrypoint mode parameters
        launch_mode: Optional[str] = None,
        docker_image: Optional[str] = None,
        model_id: Optional[str] = None,
        exposed_ports: Optional[List[int]] = None,
        quantization: Optional[str] = None,
    ) -> CreateSessionResponse:
        """Create a new session (POST /api/v1/sessions).

        IMPORTANT: The SSH private key is ONLY returned here!
        Capture response.ssh_private_key immediately.

        Two modes available:
        1. SSH mode (default): Get SSH access, deploy manually
        2. Entrypoint mode: Pre-built Docker image runs automatically

        For entrypoint mode with vLLM Docker image (recommended for instances
        without general internet access):
            await client.create_session(
                offer_id="...",
                launch_mode="entrypoint",
                docker_image="vllm/vllm-openai:latest",
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                exposed_ports=[8000],
            )

        Args:
            offer_id: ID of the offer to provision
            reservation_hours: Duration in hours (1-12)
            workload_type: Workload type (default: llm_vllm)
            storage_policy: Storage cleanup policy (preserve/destroy)
            idle_threshold_minutes: Optional idle timeout
            launch_mode: "ssh" (default) or "entrypoint" (Docker-based)
            docker_image: Docker image for entrypoint mode
            model_id: HuggingFace model ID for entrypoint mode
            exposed_ports: Ports to expose (e.g., [8000] for vLLM)
            quantization: Quantization method (awq, gptq)

        Returns:
            CreateSessionResponse with session and SSH private key
        """
        request = CreateSessionRequest(
            offer_id=offer_id,
            consumer_id=self.consumer_id,
            workload_type=workload_type,
            reservation_hours=reservation_hours,
            storage_policy=storage_policy,
            idle_threshold_minutes=idle_threshold_minutes,
            launch_mode=launch_mode,
            docker_image=docker_image,
            model_id=model_id,
            exposed_ports=exposed_ports,
            quantization=quantization,
        )

        response = await self._request(
            "POST",
            "/api/v1/sessions",
            json=request.model_dump(exclude_none=True),
        )

        if response.status_code not in (200, 201):
            self._handle_error(response, "Failed to create session")

        data = response.json()
        session_data = data.get("session")
        if session_data is None:
            raise ShopperAPIError(
                "Response missing 'session' key",
                status_code=response.status_code,
            )
        return CreateSessionResponse(
            session=Session(**session_data),
            ssh_private_key=data.get("ssh_private_key", ""),
        )

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session ID to prevent path traversal."""
        if not session_id:
            raise ValueError("session_id cannot be empty")
        if "/" in session_id or "\\" in session_id:
            raise ValueError("session_id contains invalid characters")
        if session_id in (".", ".."):
            raise ValueError("session_id cannot be '.' or '..'")

    async def get_session(self, session_id: str) -> Session:
        """Get session details (GET /api/v1/sessions/:id).

        Args:
            session_id: Session ID

        Returns:
            Session object
        """
        self._validate_session_id(session_id)
        response = await self._request("GET", f"/api/v1/sessions/{session_id}")

        if response.status_code == 404:
            raise ShopperAPIError(
                f"Session not found: {session_id}",
                status_code=404,
            )

        if response.status_code != 200:
            self._handle_error(response, f"Failed to get session {session_id}")

        return Session(**response.json())

    async def list_sessions(
        self,
        consumer_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Session]:
        """List sessions (GET /api/v1/sessions).

        Args:
            consumer_id: Filter by consumer ID
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of sessions
        """
        params = {}
        if consumer_id is not None:
            params["consumer_id"] = consumer_id
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit

        response = await self._request("GET", "/api/v1/sessions", params=params)

        if response.status_code != 200:
            self._handle_error(response, "Failed to list sessions")

        data = response.json()
        sessions = data.get("sessions", data) if isinstance(data, dict) else data

        return [Session(**s) for s in sessions]

    async def signal_done(self, session_id: str) -> Session:
        """Signal session completion (POST /api/v1/sessions/:id/done).

        Triggers graceful cleanup by the shopper.

        Args:
            session_id: Session ID

        Returns:
            Updated session (may be minimal for 204 responses)
        """
        self._validate_session_id(session_id)
        response = await self._request("POST", f"/api/v1/sessions/{session_id}/done")

        if response.status_code == 404:
            raise ShopperAPIError(
                f"Session not found: {session_id}",
                status_code=404,
            )

        if response.status_code not in (200, 202, 204):
            self._handle_error(response, f"Failed to signal done for {session_id}")

        # Handle 204 No Content response
        if response.status_code == 204:
            return Session(
                id=session_id,
                consumer_id=self.consumer_id,
                status=SessionStatus.STOPPING,
                provider="unknown",
                gpu_type="unknown",
                gpu_count=0,
                price_per_hour=0,
            )

        # API may return just a message instead of full session
        data = response.json()
        if "message" in data and "id" not in data:
            return Session(
                id=data.get("session_id", session_id),
                consumer_id=self.consumer_id,
                status=SessionStatus.STOPPING,
                provider="unknown",
                gpu_type="unknown",
                gpu_count=0,
                price_per_hour=0,
            )

        return Session(**data)

    async def force_destroy(self, session_id: str) -> Session:
        """Force destroy session (DELETE /api/v1/sessions/:id).

        Immediate cleanup - use for error recovery.

        Args:
            session_id: Session ID

        Returns:
            Updated session
        """
        self._validate_session_id(session_id)
        response = await self._request("DELETE", f"/api/v1/sessions/{session_id}")

        if response.status_code == 404:
            raise ShopperAPIError(
                f"Session not found: {session_id}",
                status_code=404,
            )

        if response.status_code not in (200, 202, 204):
            self._handle_error(response, f"Failed to destroy session {session_id}")

        if response.status_code == 204:
            # No content, return minimal session
            return Session(
                id=session_id,
                consumer_id=self.consumer_id,
                status=SessionStatus.STOPPED,
                provider="unknown",
                gpu_type="unknown",
                gpu_count=0,
                price_per_hour=0,
            )

        # API may return just a message instead of full session
        data = response.json()
        if "message" in data and "id" not in data:
            return Session(
                id=data.get("session_id", session_id),
                consumer_id=self.consumer_id,
                status=SessionStatus.STOPPED,
                provider="unknown",
                gpu_type="unknown",
                gpu_count=0,
                price_per_hour=0,
            )

        return Session(**data)

    async def extend_session(
        self,
        session_id: str,
        additional_hours: int,
    ) -> Session:
        """Extend session reservation (POST /api/v1/sessions/:id/extend).

        Args:
            session_id: Session ID
            additional_hours: Hours to add (1-12)

        Returns:
            Updated session with new expiry
        """
        self._validate_session_id(session_id)
        request = ExtendSessionRequest(additional_hours=additional_hours)

        response = await self._request(
            "POST",
            f"/api/v1/sessions/{session_id}/extend",
            json=request.model_dump(),
        )

        if response.status_code == 404:
            raise ShopperAPIError(
                f"Session not found: {session_id}",
                status_code=404,
            )

        if response.status_code != 200:
            self._handle_error(response, f"Failed to extend session {session_id}")

        return Session(**response.json())

    async def get_session_diagnostics(self, session_id: str) -> SessionDiagnostics:
        """Get session diagnostics (GET /api/v1/sessions/:id/diagnostics).

        Note: GPU health checks require client-side SSH since
        the private key is not stored server-side.

        Args:
            session_id: Session ID

        Returns:
            Session diagnostics
        """
        self._validate_session_id(session_id)
        response = await self._request(
            "GET", f"/api/v1/sessions/{session_id}/diagnostics"
        )

        if response.status_code == 404:
            raise ShopperAPIError(
                f"Session not found: {session_id}",
                status_code=404,
            )

        if response.status_code != 200:
            self._handle_error(
                response, f"Failed to get diagnostics for {session_id}"
            )

        return SessionDiagnostics(**response.json())

    # Cost Tracking

    async def get_costs(
        self,
        session_id: Optional[str] = None,
        consumer_id: Optional[str] = None,
    ) -> CostInfo:
        """Get cost information (GET /api/v1/costs).

        Args:
            session_id: Filter by session ID
            consumer_id: Filter by consumer ID

        Returns:
            Cost information
        """
        params = {}
        if session_id is not None:
            params["session_id"] = session_id
        if consumer_id is not None:
            params["consumer_id"] = consumer_id

        response = await self._request("GET", "/api/v1/costs", params=params)

        if response.status_code != 200:
            self._handle_error(response, "Failed to get costs")

        return CostInfo(**response.json())

    # Convenience Methods

    async def wait_for_running(
        self,
        session_id: str,
        timeout: float = 8 * 60,  # 8 minutes (DefaultSSHVerifyTimeout)
        initial_interval: float = 15.0,  # DefaultSSHCheckInterval
        max_interval: float = 60.0,  # DefaultSSHMaxInterval
        multiplier: float = 1.5,  # DefaultSSHBackoffMultiplier
        provider: Optional[str] = None,
    ) -> Session:
        """Wait for session to reach running status.

        Uses polling strategy matching shopper's provisioner.

        Args:
            session_id: Session ID to poll
            timeout: Maximum wait time (default: 8 minutes)
            initial_interval: Initial poll interval
            max_interval: Maximum poll interval
            multiplier: Backoff multiplier
            provider: Provider name for TensorDock delay

        Returns:
            Session in running status

        Raises:
            SessionFailedError: If session fails
            SessionStoppedError: If session stops unexpectedly
            TimeoutError: If timeout exceeded
        """
        import asyncio
        import time
        from gpu_deploy_llm.utils.errors import SessionStoppedError

        # TensorDock-specific cloud-init delay
        if provider == "tensordock":
            logger.info("TensorDock detected, waiting 45s for cloud-init...")
            await asyncio.sleep(45)

        interval = initial_interval
        deadline = time.time() + timeout

        while time.time() < deadline:
            session = await self.get_session(session_id)

            if session.status == SessionStatus.RUNNING:
                return session

            if session.status == SessionStatus.FAILED:
                raise SessionFailedError(
                    message=f"Session {session_id} failed: {session.error}",
                    session_id=session_id,
                    error=session.error or "Unknown error",
                )

            if session.status == SessionStatus.STOPPED:
                raise SessionStoppedError(
                    message=f"Session {session_id} stopped unexpectedly",
                    session_id=session_id,
                )

            logger.info(f"Session {session_id} status: {session.status}")
            await asyncio.sleep(interval)
            interval = min(interval * multiplier, max_interval)

        raise TimeoutError(f"Session {session_id} did not become running in {timeout}s")
