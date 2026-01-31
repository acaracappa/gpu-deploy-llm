"""Tests for error hierarchy."""

import pytest
from gpu_deploy_llm.utils.errors import (
    GPUDeployError,
    ShopperAPIError,
    StaleInventoryError,
    DuplicateSessionError,
    SessionFailedError,
    SessionStoppedError,
    NoAvailableOffersError,
    ProvisioningFailed,
    SSHConnectionError,
    DeploymentError,
    VerificationError,
    RateLimitError,
    ShopperNotReadyError,
)


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """All errors should inherit from GPUDeployError."""
        errors = [
            ShopperAPIError("test"),
            StaleInventoryError("test", "offer-1"),
            DuplicateSessionError("test", "session-1"),
            SessionFailedError("test", "session-1", "error"),
            NoAvailableOffersError(),
            ProvisioningFailed("test"),
            SSHConnectionError("test"),
            DeploymentError("test"),
            VerificationError("test"),
        ]
        for error in errors:
            assert isinstance(error, GPUDeployError)

    def test_api_errors_inherit_from_shopper_error(self):
        """API errors should inherit from ShopperAPIError."""
        errors = [
            StaleInventoryError("test", "offer-1"),
            DuplicateSessionError("test", "session-1"),
            SessionFailedError("test", "session-1", "error"),
            RateLimitError(),
            ShopperNotReadyError(),
        ]
        for error in errors:
            assert isinstance(error, ShopperAPIError)


class TestStaleInventoryError:
    """Tests for StaleInventoryError."""

    def test_captures_offer_id(self):
        """Should capture the stale offer ID."""
        error = StaleInventoryError("Offer stale", "offer-123")
        assert error.offer_id == "offer-123"
        assert error.status_code == 503

    def test_message(self):
        """Should have correct message."""
        error = StaleInventoryError("Offer no longer available", "offer-123")
        assert "Offer no longer available" in str(error)


class TestDuplicateSessionError:
    """Tests for DuplicateSessionError."""

    def test_captures_existing_session(self):
        """Should capture existing session ID."""
        error = DuplicateSessionError("Duplicate", "existing-session-456")
        assert error.existing_session_id == "existing-session-456"
        assert error.status_code == 409


class TestSessionFailedError:
    """Tests for SessionFailedError."""

    def test_captures_session_and_error(self):
        """Should capture session ID and error details."""
        error = SessionFailedError(
            "Session failed",
            session_id="session-789",
            error="SSH verification timeout",
        )
        assert error.session_id == "session-789"
        assert error.error == "SSH verification timeout"


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_default_values(self):
        """Should have correct defaults."""
        error = RateLimitError()
        assert error.status_code == 429
        assert error.retry_after is None

    def test_with_retry_after(self):
        """Should capture retry-after."""
        error = RateLimitError(retry_after=30)
        assert error.retry_after == 30


class TestSSHConnectionError:
    """Tests for SSHConnectionError."""

    def test_captures_connection_details(self):
        """Should capture host and port."""
        error = SSHConnectionError(
            "Connection refused",
            host="192.168.1.100",
            port=22,
        )
        assert error.host == "192.168.1.100"
        assert error.port == 22


class TestDeploymentError:
    """Tests for DeploymentError."""

    def test_captures_stage(self):
        """Should capture deployment stage."""
        error = DeploymentError("Failed to pull image", stage="pull_image")
        assert error.stage == "pull_image"


class TestProvisioningFailed:
    """Tests for ProvisioningFailed."""

    def test_captures_attempts(self):
        """Should capture number of attempts."""
        error = ProvisioningFailed("Max retries exceeded", attempts=3)
        assert error.attempts == 3
