"""Error hierarchy for GPU Deploy LLM.

Mirrors error types from cloud-gpu-shopper for proper error handling.
"""

from typing import Optional


class GPUDeployError(Exception):
    """Base exception for all GPU Deploy LLM errors."""

    pass


class ShopperAPIError(GPUDeployError):
    """Base exception for cloud-gpu-shopper API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class StaleInventoryError(ShopperAPIError):
    """Raised when selected offer is no longer available (HTTP 503).

    Response includes:
    - error_type: "stale_inventory"
    - retry_suggested: true
    - offer_id: the stale offer ID
    """

    def __init__(
        self,
        message: str,
        offer_id: str,
        status_code: int = 503,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, status_code, request_id)
        self.offer_id = offer_id


class DuplicateSessionError(ShopperAPIError):
    """Raised when consumer already has active session for an offer (HTTP 409)."""

    def __init__(
        self,
        message: str,
        existing_session_id: str,
        status_code: int = 409,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, status_code, request_id)
        self.existing_session_id = existing_session_id


class SessionFailedError(ShopperAPIError):
    """Raised when session status transitions to 'failed'.

    This occurs on SSH verification timeout or provider creation failures.
    """

    def __init__(
        self,
        message: str,
        session_id: str,
        error: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, status_code, request_id)
        self.session_id = session_id
        self.error = error


class SessionStoppedError(ShopperAPIError):
    """Raised when session unexpectedly transitions to 'stopped'."""

    def __init__(
        self,
        message: str,
        session_id: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, status_code, request_id)
        self.session_id = session_id


class NoAvailableOffersError(GPUDeployError):
    """Raised when no GPU offers meet the requirements."""

    def __init__(self, message: str = "No available offers meet the requirements"):
        super().__init__(message)


class ProvisioningFailed(GPUDeployError):
    """Raised when provisioning fails after all retry attempts."""

    def __init__(self, message: str, attempts: int = 0):
        super().__init__(message)
        self.attempts = attempts


class SSHConnectionError(GPUDeployError):
    """Raised when SSH connection fails."""

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        super().__init__(message)
        self.host = host
        self.port = port


class DeploymentError(GPUDeployError):
    """Raised when vLLM deployment fails."""

    def __init__(self, message: str, stage: Optional[str] = None):
        super().__init__(message)
        self.stage = stage


class VerificationError(GPUDeployError):
    """Raised when deployment verification fails."""

    def __init__(self, message: str, check: Optional[str] = None):
        super().__init__(message)
        self.check = check


class ShopperNotReadyError(ShopperAPIError):
    """Raised when shopper /ready returns 503 (startup sweep in progress)."""

    def __init__(
        self,
        message: str = "Shopper not ready (startup sweep in progress)",
        status_code: int = 503,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, status_code, request_id)


class RateLimitError(ShopperAPIError):
    """Raised when rate limited by the shopper API (HTTP 429)."""

    def __init__(
        self,
        message: str = "Rate limited by shopper API",
        retry_after: Optional[int] = None,
        status_code: int = 429,
        request_id: Optional[str] = None,
    ):
        super().__init__(message, status_code, request_id)
        self.retry_after = retry_after
