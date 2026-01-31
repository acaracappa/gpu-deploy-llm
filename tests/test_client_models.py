"""Tests for client Pydantic models."""

import pytest
from datetime import datetime
from gpu_deploy_llm.client.models import (
    GPUOffer,
    Session,
    SessionStatus,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionDiagnostics,
    CostInfo,
    InventoryFilter,
    StoragePolicy,
)


class TestGPUOffer:
    """Tests for GPUOffer model."""

    def test_create_minimal(self):
        """Should create with required fields only."""
        offer = GPUOffer(
            id="offer-123",
            provider="vastai",
            gpu_type="RTX 4090",
            gpu_count=1,
            vram_gb=24.0,
            price_per_hour=0.45,
        )
        assert offer.id == "offer-123"
        assert offer.provider == "vastai"
        assert offer.gpu_type == "RTX 4090"
        assert offer.region is None

    def test_create_full(self):
        """Should create with all fields."""
        offer = GPUOffer(
            id="offer-456",
            provider="tensordock",
            gpu_type="A100",
            gpu_count=2,
            vram_gb=80.0,
            price_per_hour=1.50,
            region="us-east",
            cpu_cores=16,
            ram_gb=128,
            disk_gb=500,
        )
        assert offer.region == "us-east"
        assert offer.cpu_cores == 16


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Should create session with required fields."""
        session = Session(
            id="sess-123",
            consumer_id="gpu-deploy-llm/v0.1.0",
            status=SessionStatus.RUNNING,
            provider="vastai",
            offer_id="offer-123",
            gpu_type="RTX 4090",
            gpu_count=1,
            price_per_hour=0.45,
            created_at=datetime.utcnow(),
        )
        assert session.status == SessionStatus.RUNNING
        assert not session.is_terminal

    def test_is_terminal_stopped(self):
        """Stopped session should be terminal."""
        session = Session(
            id="sess-123",
            consumer_id="test",
            status=SessionStatus.STOPPED,
            provider="vastai",
            offer_id="offer-123",
            gpu_type="RTX 4090",
            gpu_count=1,
            price_per_hour=0.45,
            created_at=datetime.utcnow(),
        )
        assert session.is_terminal

    def test_is_terminal_failed(self):
        """Failed session should be terminal."""
        session = Session(
            id="sess-123",
            consumer_id="test",
            status=SessionStatus.FAILED,
            provider="vastai",
            offer_id="offer-123",
            gpu_type="RTX 4090",
            gpu_count=1,
            price_per_hour=0.45,
            created_at=datetime.utcnow(),
            error="SSH timeout",
        )
        assert session.is_terminal
        assert session.error == "SSH timeout"

    def test_is_connectable(self):
        """Should check SSH connection details."""
        # Not connectable without SSH details
        session = Session(
            id="sess-123",
            consumer_id="test",
            status=SessionStatus.RUNNING,
            provider="vastai",
            offer_id="offer-123",
            gpu_type="RTX 4090",
            gpu_count=1,
            price_per_hour=0.45,
            created_at=datetime.utcnow(),
        )
        assert not session.is_connectable

        # Connectable with SSH details
        session.ssh_host = "192.168.1.100"
        session.ssh_port = 22
        session.ssh_user = "root"
        assert session.is_connectable


class TestCreateSessionRequest:
    """Tests for CreateSessionRequest model."""

    def test_create_minimal(self):
        """Should create with defaults."""
        request = CreateSessionRequest(
            offer_id="offer-123",
            consumer_id="gpu-deploy-llm/v0.1.0",
        )
        assert request.workload_type == "llm_vllm"
        assert request.reservation_hours == 1
        assert request.storage_policy == StoragePolicy.DESTROY

    def test_reservation_hours_validation(self):
        """Should validate reservation hours range."""
        # Valid range
        request = CreateSessionRequest(
            offer_id="offer-123",
            consumer_id="test",
            reservation_hours=12,
        )
        assert request.reservation_hours == 12

        # Invalid: too low
        with pytest.raises(ValueError):
            CreateSessionRequest(
                offer_id="offer-123",
                consumer_id="test",
                reservation_hours=0,
            )

        # Invalid: too high
        with pytest.raises(ValueError):
            CreateSessionRequest(
                offer_id="offer-123",
                consumer_id="test",
                reservation_hours=13,
            )


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_statuses(self):
        """Should have all expected statuses."""
        statuses = [s.value for s in SessionStatus]
        assert "pending" in statuses
        assert "provisioning" in statuses
        assert "running" in statuses
        assert "stopping" in statuses
        assert "stopped" in statuses
        assert "failed" in statuses

    def test_status_from_string(self):
        """Should create from string value."""
        status = SessionStatus("running")
        assert status == SessionStatus.RUNNING


class TestInventoryFilter:
    """Tests for InventoryFilter model."""

    def test_create_empty(self):
        """Should create with no filters."""
        filter = InventoryFilter()
        assert filter.min_vram is None
        assert filter.max_price is None

    def test_create_with_filters(self):
        """Should create with filters."""
        filter = InventoryFilter(
            min_vram=16.0,
            max_price=0.50,
            provider="vastai",
            min_gpu_count=2,
        )
        assert filter.min_vram == 16.0
        assert filter.max_price == 0.50
        assert filter.provider == "vastai"
        assert filter.min_gpu_count == 2
