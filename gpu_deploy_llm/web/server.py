"""FastAPI server with WebSocket for real-time test dashboard.

Provides:
- GET / - Serves the single-page dashboard
- GET /api/status - Current service and session status
- POST /api/test/run - Start a deployment test
- WS /ws - Real-time event stream
"""

import asyncio
import json
import logging
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from gpu_deploy_llm import __version__
from gpu_deploy_llm.client import ShopperClient, SessionStatus
from gpu_deploy_llm.models import calculate_requirements, Quantization, list_models
from gpu_deploy_llm.models.requirements import select_best_offer
from gpu_deploy_llm.ssh import SSHConnection, HostDiagnosticsCollector
from gpu_deploy_llm.deploy import (
    VLLMDeployer, DeploymentConfig, HealthChecker, Benchmarker,
    BenchmarkStore, create_benchmark_record,
)
from gpu_deploy_llm.diagnostics import DiagnosticCollector, Checkpoint
from gpu_deploy_llm.utils.errors import StaleInventoryError

logger = logging.getLogger(__name__)

# Global state
_shopper_url: str = "http://localhost:8080"
_debug: bool = False
_active_connections: Set[WebSocket] = set()
_current_test: Optional["TestRunner"] = None
_current_task: Optional[asyncio.Task] = None
_test_lock: Optional[asyncio.Lock] = None  # Initialized in lifespan
_connections_lock: Optional[asyncio.Lock] = None  # Initialized in lifespan
MAX_WEBSOCKET_CONNECTIONS = 100


def _get_test_lock() -> asyncio.Lock:
    """Get test lock, raising if not initialized."""
    if _test_lock is None:
        raise RuntimeError("Server not initialized - test_lock is None")
    return _test_lock


def _get_connections_lock() -> asyncio.Lock:
    """Get connections lock, raising if not initialized."""
    if _connections_lock is None:
        raise RuntimeError("Server not initialized - connections_lock is None")
    return _connections_lock


class EventType(str, Enum):
    """WebSocket event types."""

    STATUS = "status"
    LOG = "log"
    SESSION = "session"
    SNAPSHOT = "snapshot"
    RESULT = "result"
    ERROR = "error"


@dataclass
class WebSocketEvent:
    """Event sent over WebSocket."""

    type: EventType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def to_json(self) -> str:
        return json.dumps({"type": self.type.value, **self.data, "timestamp": self.timestamp})


class TestConfig(BaseModel):
    """Configuration for a deployment test."""

    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    quantization: str = "none"
    max_price: float = 0.50
    reservation_hours: int = 1
    provider: Optional[str] = None
    skip_verification: bool = False
    auto_cleanup: bool = True

    @field_validator("max_price")
    @classmethod
    def validate_max_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_price must be positive")
        if v > 100:
            raise ValueError("max_price exceeds maximum ($100/hr)")
        return v

    @field_validator("reservation_hours")
    @classmethod
    def validate_reservation_hours(cls, v: int) -> int:
        if v < 1:
            raise ValueError("reservation_hours must be at least 1")
        if v > 24:
            raise ValueError("reservation_hours exceeds maximum (24)")
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        valid = {"none", "awq", "gptq", "squeezellm", "fp8"}
        if v.lower() not in valid:
            raise ValueError(f"quantization must be one of: {', '.join(valid)}")
        return v.lower()


class TestRunner:
    """Runs a deployment test and emits events."""

    STEPS = [
        "calculate_requirements",
        "query_inventory",
        "select_offer",
        "create_session",
        "wait_for_running",
        "connect_ssh",
        "diagnose_host",
        "deploy_vllm",
        "verify_deployment",
        "benchmark",
        "cleanup",
    ]

    def __init__(self, config: TestConfig, shopper_url: str, debug: bool = False):
        self.config = config
        self.shopper_url = shopper_url
        self.debug = debug
        self.current_step = 0
        self.session_id: Optional[str] = None
        self.ssh_key: Optional[str] = None
        self.passed = False
        self.error: Optional[str] = None
        self._cancelled = False

    async def run(self) -> None:
        """Run the test and emit events."""
        try:
            if self._cancelled:
                await self._log("info", "Test cancelled")
                return

            await self._emit_status("calculate_requirements", "running")
            await self._log("info", f"Calculating requirements for {self.config.model_id}")

            quant = Quantization(self.config.quantization)
            requirements = calculate_requirements(self.config.model_id, quant)

            await self._log(
                "info",
                f"Requirements: {requirements.min_vram_gb}GB VRAM, {requirements.min_gpu_count} GPU(s)",
            )
            await self._emit_status("calculate_requirements", "complete")

            if self._cancelled:
                await self._log("info", "Test cancelled")
                return

            async with ShopperClient(self.shopper_url, debug=self.debug) as client:
                # Check shopper health
                await self._log("info", "Checking shopper health...")
                try:
                    await client.wait_for_ready(timeout=30)
                    await self._log("info", "Shopper ready")
                except Exception as e:
                    await self._log("error", f"Shopper not ready: {e}")
                    raise

                # Query inventory
                await self._emit_status("query_inventory", "running")
                offers = await client.get_inventory(
                    min_vram=requirements.total_vram_needed,
                    max_price=self.config.max_price,
                    provider=self.config.provider,
                    min_gpu_count=requirements.min_gpu_count,
                )
                await self._log("info", f"Found {len(offers)} offers")
                await self._emit_status("query_inventory", "complete")

                if self._cancelled:
                    await self._log("info", "Test cancelled")
                    return

                if not offers:
                    raise Exception("No suitable GPU offers found")

                # Select offer
                await self._emit_status("select_offer", "running")
                offer = select_best_offer(offers, requirements, self.config.max_price)
                if not offer:
                    raise Exception("No offers meet requirements")

                await self._log(
                    "info",
                    f"Selected: {offer.gpu_type} @ ${offer.price_per_hour:.3f}/hr ({offer.provider})",
                )
                await self._emit_status("select_offer", "complete")

                # Create session with stale inventory retry
                await self._emit_status("create_session", "running")
                excluded = set()
                session = None

                for attempt in range(3):
                    available = [o for o in offers if o.id not in excluded]
                    if not available:
                        raise Exception("No offers available after retries")

                    offer = min(available, key=lambda o: o.price_per_hour)

                    try:
                        response = await client.create_session(
                            offer_id=offer.id,
                            reservation_hours=self.config.reservation_hours,
                        )
                        session = response.session
                        self.session_id = session.id
                        self.ssh_key = response.ssh_private_key
                        await self._log("info", f"Session created: {session.id}")
                        await self._emit_session(session)
                        break
                    except StaleInventoryError as e:
                        await self._log("warning", f"Offer stale, retrying... (attempt {attempt + 1}/3)")
                        excluded.add(e.offer_id)

                if not session:
                    raise Exception("Failed to create session after retries")

                await self._emit_status("create_session", "complete")

                if self._cancelled:
                    await self._log("info", "Test cancelled")
                    return

                # Wait for running
                await self._emit_status("wait_for_running", "running")
                session = await client.wait_for_running(
                    session.id,
                    provider=session.provider,
                )
                await self._log("info", f"Session running: {session.ssh_host}:{session.ssh_port}")
                await self._emit_session(session)
                await self._emit_status("wait_for_running", "complete")

                if self._cancelled:
                    await self._log("info", "Test cancelled")
                    return

                # Connect SSH
                await self._emit_status("connect_ssh", "running")
                async with SSHConnection(
                    host=session.ssh_host,
                    port=session.ssh_port,
                    user=session.ssh_user,
                    private_key=self.ssh_key,
                ) as ssh:
                    await self._log("info", "SSH connected")
                    await self._emit_status("connect_ssh", "complete")

                    if self._cancelled:
                        await self._log("info", "Test cancelled")
                        return

                    # Diagnose host
                    await self._emit_status("diagnose_host", "running")
                    await self._log("info", "Running host diagnostics...")

                    async def diag_progress(step: str, msg: str):
                        await self._log("info", f"  [{step}] {msg}")

                    diagnostics_collector = HostDiagnosticsCollector(ssh)
                    host_diag = await diagnostics_collector.collect_all(progress_callback=diag_progress)

                    # Log summary
                    if host_diag.shell_functional:
                        await self._log("info", f"Host: {host_diag.hostname}")
                        await self._log("info", f"  OS: {host_diag.os_info.distro} {host_diag.os_info.version}")
                        await self._log("info", f"  Kernel: {host_diag.os_info.kernel}")

                        if host_diag.gpus:
                            for i, gpu in enumerate(host_diag.gpus):
                                await self._log(
                                    "info",
                                    f"  GPU {i}: {gpu.name} ({gpu.memory_total_mb}MB, {gpu.temperature_c}°C, driver {gpu.driver_version})"
                                )
                        else:
                            await self._log("warning", "  No GPUs detected!")

                        if host_diag.docker.available:
                            await self._log("info", f"  Docker: {host_diag.docker.version}")
                        else:
                            await self._log("warning", f"  Docker: Not available - {host_diag.docker.error}")

                        if host_diag.python.available:
                            vllm_info = f", vLLM {host_diag.python.vllm_version}" if host_diag.python.vllm_installed else ""
                            await self._log("info", f"  Python: {host_diag.python.version}{vllm_info}")
                        else:
                            await self._log("warning", "  Python: Not available")

                        await self._log(
                            "info",
                            f"  Resources: {host_diag.resources.memory_available_gb:.1f}GB RAM, "
                            f"{host_diag.resources.disk_available_gb:.1f}GB disk available"
                        )

                        net_status = []
                        if host_diag.can_reach_internet:
                            net_status.append("internet ✓")
                        else:
                            net_status.append("internet ✗")
                        if host_diag.can_reach_huggingface:
                            net_status.append("huggingface ✓")
                        else:
                            net_status.append("huggingface ✗")
                        await self._log("info", f"  Network: {', '.join(net_status)}")

                        await self._emit_status("diagnose_host", "complete")
                    else:
                        await self._log("error", f"Shell not functional: {host_diag.connection_error}")
                        await self._emit_status("diagnose_host", "failed")
                        raise Exception(f"Host diagnostics failed: {host_diag.connection_error}")

                    if self._cancelled:
                        await self._log("info", "Test cancelled")
                        return

                    # Deploy vLLM
                    await self._emit_status("deploy_vllm", "running")
                    deployer = VLLMDeployer(ssh)
                    deploy_config = DeploymentConfig(
                        model_id=requirements.model_id,
                        quantization=quant,
                        gpu_count=session.gpu_count,
                    )

                    await self._log("info", "Deploying vLLM (this may take several minutes)...")
                    deploy_result = await deployer.deploy(deploy_config)

                    if not deploy_result.success:
                        raise Exception(f"Deployment failed: {deploy_result.error}")

                    await self._log("info", f"vLLM deployed at {deploy_result.endpoint}")
                    await self._emit_status("deploy_vllm", "complete")

                    if self._cancelled:
                        await self._log("info", "Test cancelled")
                        return

                    # Verify deployment
                    if not self.config.skip_verification:
                        await self._emit_status("verify_deployment", "running")

                        checker = HealthChecker(
                            ssh=ssh,
                            endpoint=deploy_result.endpoint,
                            api_key=deploy_result.api_key,
                        )

                        await self._log("info", "Waiting for model to load...")

                        async def log_progress(elapsed: float, message: str):
                            await self._log("info", f"[{elapsed:.0f}s] {message}")

                        try:
                            await checker.wait_for_ready(
                                timeout=300,
                                interval=10.0,
                                progress_callback=log_progress,
                            )
                        except TimeoutError:
                            await self._log("warning", "Model loading timed out")

                        await self._log("info", "Running verification checks...")
                        verification = await checker.verify_all()

                        for check in verification.checks:
                            level = "info" if check.status.value == "passed" else "error"
                            await self._log(level, f"{check.name}: {check.status.value} - {check.message}")

                        await self._emit_status("verify_deployment", "complete")
                    else:
                        await self._emit_status("verify_deployment", "skipped")

                    if self._cancelled:
                        await self._log("info", "Test cancelled")
                        return

                    # Benchmark (inside SSH context)
                    await self._emit_status("benchmark", "running")
                    await self._log("info", "Running benchmark with test prompts...")

                    try:
                        async with Benchmarker(
                            ssh=ssh,
                            api_key=deploy_result.api_key,
                            model_id=requirements.model_id,
                        ) as benchmarker:
                            benchmark_result = await benchmarker.run_benchmark()

                            if benchmark_result.success:
                                await self._log(
                                    "info",
                                    f"Benchmark complete: {benchmark_result.avg_tokens_per_second:.1f} tokens/sec avg, "
                                    f"TTFT: {benchmark_result.avg_time_to_first_token_ms:.0f}ms"
                                )

                                # Log individual prompt results
                                for pr in benchmark_result.prompt_results:
                                    status = "passed" if pr.matches_expected or pr.matches_expected is None else "failed"
                                    match_info = ""
                                    if pr.matches_expected is not None:
                                        match_info = f" [{'match' if pr.matches_expected else 'no match'}]"
                                    await self._log(
                                        "info" if status == "passed" else "warning",
                                        f"  {pr.prompt_id}: {pr.tokens_per_second:.1f} tps, "
                                        f"{pr.time_to_first_token_ms:.0f}ms TTFT{match_info}"
                                    )

                                # Log response quality
                                if benchmark_result.prompts_with_expected > 0:
                                    await self._log(
                                        "info",
                                        f"Response quality: {benchmark_result.prompts_matching_expected}/{benchmark_result.prompts_with_expected} "
                                        f"prompts matched expected ({benchmark_result.match_rate*100:.0f}%)"
                                    )

                                if benchmark_result.throughput_tokens_per_second > 0:
                                    await self._log(
                                        "info",
                                        f"Throughput test: {benchmark_result.throughput_tokens_per_second:.1f} tokens/sec"
                                    )

                                # Save benchmark to store
                                try:
                                    store = BenchmarkStore()
                                    record = create_benchmark_record(
                                        session_id=session.id,
                                        model_id=requirements.model_id,
                                        quantization=self.config.quantization,
                                        provider=session.provider,
                                        gpu_type=session.gpu_type,
                                        gpu_count=session.gpu_count,
                                        gpu_vram_mb=0,  # TODO: get from diagnostics
                                        price_per_hour=session.price_per_hour,
                                        benchmark_result=benchmark_result,
                                    )
                                    store.save(record)
                                    await self._log(
                                        "info",
                                        f"Benchmark saved: {record.id} "
                                        f"({record.tokens_per_dollar:.1f} tokens/$/hr, "
                                        f"${record.cost_per_million_tokens:.4f}/1M tokens)"
                                    )
                                except Exception as e:
                                    await self._log("warning", f"Failed to save benchmark: {e}")

                                await self._emit_status("benchmark", "complete")
                            else:
                                await self._log("warning", f"Benchmark had failures: {benchmark_result.error}")
                                await self._emit_status("benchmark", "complete")

                    except Exception as e:
                        await self._log("warning", f"Benchmark failed: {e}")
                        await self._emit_status("benchmark", "failed")

                # Cleanup (outside SSH context - uses shopper client, not SSH)
                if self.config.auto_cleanup:
                    await self._emit_status("cleanup", "running")
                    await self._log("info", "Cleaning up session...")
                    await client.signal_done(session.id)
                    await self._log("info", "Session cleanup initiated")
                    await self._emit_status("cleanup", "complete")
                else:
                    await self._emit_status("cleanup", "skipped")
                    await self._log("info", f"Session left running: {session.id}")

            self.passed = True
            await self._emit_result(True)

        except Exception as e:
            self.error = str(e)
            await self._log("error", f"Test failed: {e}")
            await self._emit_result(False, str(e))

            # Cleanup on error
            if self.session_id and self.config.auto_cleanup:
                try:
                    async with ShopperClient(self.shopper_url) as client:
                        await client.force_destroy(self.session_id)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup session {self.session_id}: {cleanup_error}")
                    await self._log("warning", f"Session cleanup failed: {cleanup_error}")

    async def _emit_status(self, step: str, status: str) -> None:
        """Emit step status event."""
        step_index = self.STEPS.index(step) if step in self.STEPS else 0
        event = WebSocketEvent(
            type=EventType.STATUS,
            data={
                "step": step,
                "status": status,
                "step_index": step_index,
                "total_steps": len(self.STEPS),
            },
        )
        await broadcast_event(event)

    async def _log(self, level: str, message: str) -> None:
        """Emit log event and print to console."""
        # Print to console for debugging
        print(f"[TEST][{level.upper()}] {message}")

        event = WebSocketEvent(
            type=EventType.LOG,
            data={"level": level, "message": message},
        )
        await broadcast_event(event)

    async def _emit_session(self, session) -> None:
        """Emit session update event."""
        event = WebSocketEvent(
            type=EventType.SESSION,
            data={
                "id": session.id,
                "status": session.status.value,
                "provider": session.provider,
                "gpu_type": session.gpu_type,
                "gpu_count": session.gpu_count,
                "price_per_hour": session.price_per_hour,
                "ssh_host": session.ssh_host,
                "ssh_port": session.ssh_port,
            },
        )
        await broadcast_event(event)

    async def _emit_result(self, passed: bool, error: Optional[str] = None) -> None:
        """Emit test result event."""
        event = WebSocketEvent(
            type=EventType.RESULT,
            data={"passed": passed, "error": error},
        )
        await broadcast_event(event)


async def broadcast_event(event: WebSocketEvent) -> None:
    """Broadcast event to all connected WebSocket clients."""
    async with _get_connections_lock():
        connections_snapshot = set(_active_connections)

    disconnected = set()
    for ws in connections_snapshot:
        try:
            await ws.send_text(event.to_json())
        except Exception:
            disconnected.add(ws)

    if disconnected:
        async with _get_connections_lock():
            # Use discard() to safely handle concurrent modifications
            for ws in disconnected:
                _active_connections.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context."""
    global _test_lock, _connections_lock
    _test_lock = asyncio.Lock()
    _connections_lock = asyncio.Lock()
    logger.info(f"Starting GPU Deploy LLM Dashboard v{__version__}")
    yield
    logger.info("Shutting down dashboard")


app = FastAPI(
    title="GPU Deploy LLM Dashboard",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML."""
    static_path = Path(__file__).parent / "static" / "index.html"
    try:
        if static_path.exists():
            return HTMLResponse(static_path.read_text())
    except IOError as e:
        logger.warning(f"Failed to read dashboard HTML: {e}")

    # Fallback inline HTML if static file missing or read failed
    return HTMLResponse(get_inline_dashboard())


@app.get("/api/status")
async def get_status():
    """Get current service and session status."""
    status = {
        "version": __version__,
        "shopper_url": _shopper_url,
        "shopper": {"healthy": False, "ready": False},
        "providers": {},
        "active_session": None,
        "current_test": None,
    }

    try:
        async with ShopperClient(_shopper_url, debug=_debug) as client:
            # Health check
            try:
                health = await client.health_check()
                status["shopper"]["healthy"] = True
                status["shopper"]["status"] = health.status
            except Exception as e:
                logger.debug(f"Health check failed: {e}")

            # Ready check
            try:
                ready = await client.ready_check()
                status["shopper"]["ready"] = ready.ready
            except Exception as e:
                logger.debug(f"Ready check failed: {e}")

            # Get inventory counts per provider
            try:
                offers = await client.get_inventory()
                providers = {}
                for offer in offers:
                    if offer.provider not in providers:
                        providers[offer.provider] = 0
                    providers[offer.provider] += 1
                status["providers"] = providers
            except Exception as e:
                logger.debug(f"Inventory check failed: {e}")

    except Exception as e:
        status["error"] = str(e)

    async with _get_test_lock():
        if _current_test:
            status["current_test"] = {
                "session_id": _current_test.session_id,
                "current_step": _current_test.current_step,
            }

    return JSONResponse(status)


@app.get("/api/models")
async def get_models():
    """Get list of supported models."""
    models = list_models()
    return JSONResponse([
        {
            "id": m.model_id,
            "name": m.name,
            "vram_fp16": m.vram_fp16_gb,
            "vram_4bit": m.vram_4bit_gb,
            "description": m.description,
        }
        for m in models
    ])


@app.get("/api/sessions")
async def get_sessions(limit: int = Query(default=50, ge=1, le=500)):
    """Get session history from the shopper.

    Returns sessions created by gpu-deploy-llm, newest first.
    """
    try:
        async with ShopperClient(_shopper_url, debug=_debug) as client:
            # Filter by our consumer_id (includes version) to only show gpu-deploy-llm sessions
            sessions = await client.list_sessions(
                consumer_id=f"gpu-deploy-llm/v{__version__}",
                limit=limit
            )

            # Convert to response format, sorted by created_at descending
            result = []
            for s in sessions:
                result.append({
                    "id": s.id,
                    "status": s.status.value if hasattr(s.status, 'value') else s.status,
                    "provider": s.provider,
                    "gpu_type": s.gpu_type,
                    "gpu_count": s.gpu_count,
                    "price_per_hour": s.price_per_hour,
                    "ssh_host": s.ssh_host,
                    "ssh_port": s.ssh_port,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                })

            # Sort by created_at descending (newest first)
            result.sort(key=lambda x: x.get("created_at") or "", reverse=True)

            return JSONResponse({"sessions": result})
    except Exception as e:
        logger.error(f"Failed to fetch sessions: {e}")
        return JSONResponse({"sessions": [], "error": str(e)})


@app.get("/api/benchmarks")
async def get_benchmarks(
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    gpu_type: Optional[str] = None,
    limit: int = 50,
):
    """Get benchmark history.

    Args:
        model_id: Filter by model
        provider: Filter by provider
        gpu_type: Filter by GPU type
        limit: Max results
    """
    try:
        store = BenchmarkStore()
        results = store.list(
            model_id=model_id,
            provider=provider,
            gpu_type=gpu_type,
            limit=limit,
        )
        return JSONResponse({"benchmarks": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Failed to fetch benchmarks: {e}")
        return JSONResponse({"benchmarks": [], "error": str(e)})


@app.get("/api/benchmarks/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """Get a specific benchmark record with full details."""
    try:
        store = BenchmarkStore()
        record = store.get(benchmark_id)
        if not record:
            raise HTTPException(404, "Benchmark not found")
        return JSONResponse(record.to_dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch benchmark: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/benchmarks/compare/{model_id}")
async def compare_benchmarks(model_id: str):
    """Get performance comparison across GPU types for a model."""
    try:
        store = BenchmarkStore()
        comparison = store.get_comparison(model_id)
        return JSONResponse(comparison)
    except Exception as e:
        logger.error(f"Failed to compare benchmarks: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/benchmarks/recommend/{model_id}")
async def recommend_config(
    model_id: str,
    optimize_for: str = "balanced",  # cost, speed, balanced
    max_price: Optional[float] = None,
):
    """Get recommended configuration for a model based on historical benchmarks.

    Args:
        model_id: Model to get recommendation for
        optimize_for: "cost" (best value), "speed" (fastest), or "balanced"
        max_price: Optional maximum price per hour
    """
    try:
        store = BenchmarkStore()
        recommendation = store.get_best_config(
            model_id=model_id,
            optimize_for=optimize_for,
            max_price=max_price,
        )
        if not recommendation:
            return JSONResponse({
                "recommendation": None,
                "message": f"No benchmark data for model {model_id}",
            })
        return JSONResponse(recommendation)
    except Exception as e:
        logger.error(f"Failed to get recommendation: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/test/run")
async def run_test(config: TestConfig):
    """Start a deployment test."""
    global _current_test, _current_task

    async with _get_test_lock():
        if _current_test and not _current_test.passed and not _current_test.error:
            raise HTTPException(400, "Test already running")

        _current_test = TestRunner(config, _shopper_url, _debug)
        _current_task = asyncio.create_task(_current_test.run())

        def _task_done_callback(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
                if exc:
                    logger.error(f"Test task failed: {exc}")
            except asyncio.CancelledError:
                logger.info("Test task was cancelled")

        _current_task.add_done_callback(_task_done_callback)

    return JSONResponse({"status": "started", "model": config.model_id})


@app.post("/api/test/stop")
async def stop_test():
    """Stop the current test."""
    global _current_test, _current_task

    async with _get_test_lock():
        if _current_test:
            _current_test._cancelled = True
        if _current_task and not _current_task.done():
            _current_task.cancel()
        _current_test = None
        _current_task = None

    return JSONResponse({"status": "stopped"})


class CleanupRequest(BaseModel):
    """Request to cleanup a session."""
    session_id: Optional[str] = None


@app.post("/api/session/cleanup")
async def cleanup_session(request: CleanupRequest):
    """Manually cleanup a session.

    If session_id is provided, cleanup that session.
    Otherwise, cleanup the current test's session if one exists.
    """
    global _current_test

    # Determine which session to cleanup
    target_session_id = request.session_id
    async with _get_test_lock():
        if not target_session_id and _current_test:
            target_session_id = _current_test.session_id

    if not target_session_id:
        raise HTTPException(400, "No session to cleanup")

    cleanup_status = "cleanup_initiated"
    error_message = None

    try:
        async with ShopperClient(_shopper_url, debug=_debug) as client:
            # Try force destroy first for faster cleanup
            await client.force_destroy(target_session_id)
    except Exception as e:
        # Log the error but don't fail - the session might already be gone
        error_message = str(e)
        cleanup_status = "cleanup_attempted"
        logger.warning(f"Cleanup request for {target_session_id} had error: {e}")

    # Always clear local state and emit event
    async with _get_test_lock():
        if _current_test and _current_test.session_id == target_session_id:
            _current_test = None

    # Emit cleanup event
    msg = f"Session {target_session_id[:12]}... cleanup initiated"
    if error_message:
        msg += f" (warning: {error_message})"

    event = WebSocketEvent(
        type=EventType.LOG,
        data={"level": "info" if not error_message else "warning", "message": msg},
    )
    await broadcast_event(event)

    return JSONResponse({
        "status": cleanup_status,
        "session_id": target_session_id,
        "error": error_message
    })


@app.post("/api/session/dismiss")
async def dismiss_session(request: CleanupRequest):
    """Dismiss a session from local tracking without cleanup.

    Use this when a session is stuck/phantom and cleanup fails.
    """
    global _current_test

    target_session_id = request.session_id
    if not target_session_id:
        raise HTTPException(400, "No session_id provided")

    # Clear local state
    async with _get_test_lock():
        if _current_test and _current_test.session_id == target_session_id:
            _current_test = None

    # Emit dismissal event
    event = WebSocketEvent(
        type=EventType.LOG,
        data={"level": "info", "message": f"Session {target_session_id[:12]}... dismissed from tracking"},
    )
    await broadcast_event(event)

    # Also emit a session update to mark it as dismissed
    event = WebSocketEvent(
        type=EventType.SESSION,
        data={
            "id": target_session_id,
            "status": "dismissed",
            "provider": "",
            "gpu_type": "",
            "gpu_count": 0,
            "price_per_hour": 0,
        },
    )
    await broadcast_event(event)

    return JSONResponse({
        "status": "dismissed",
        "session_id": target_session_id
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    # Check connection limit outside the lock to minimize lock hold time
    async with _get_connections_lock():
        if len(_active_connections) >= MAX_WEBSOCKET_CONNECTIONS:
            await websocket.close(code=1013)  # Try Again Later
            return

    # Accept WebSocket outside the lock (I/O operation)
    await websocket.accept()

    # Add to connections under lock
    async with _get_connections_lock():
        _active_connections.add(websocket)

    try:
        # Send initial status - send as dict directly to avoid double-encoding
        status_response = await get_status()
        status_data = json.loads(status_response.body.decode())
        await websocket.send_json({"type": "init", "status": status_data})

        # Keep connection alive
        while True:
            try:
                # Wait for messages (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_text("ping")

    except WebSocketDisconnect:
        pass
    finally:
        async with _get_connections_lock():
            _active_connections.discard(websocket)


def get_inline_dashboard() -> str:
    """Return inline dashboard HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU Deploy LLM - Dashboard</title>
    <style>
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #e8e8e8;
            --text-secondary: #a0a0a0;
            --accent: #00d9ff;
            --success: #00ff88;
            --warning: #ffaa00;
            --error: #ff4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--bg-card);
        }
        .header h1 { font-size: 1.5rem; }
        .btn {
            background: var(--accent);
            color: var(--bg-primary);
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: var(--bg-secondary);
            border-radius: 10px;
            padding: 20px;
        }
        .card h2 {
            font-size: 1rem;
            margin-bottom: 15px;
            color: var(--text-secondary);
        }
        .status-item {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 10px;
        }
        .status-dot.green { background: var(--success); }
        .status-dot.yellow { background: var(--warning); }
        .status-dot.red { background: var(--error); }
        .status-dot.gray { background: var(--text-secondary); }
        .progress-bar {
            height: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s;
        }
        .step {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        .step-icon {
            width: 20px;
            height: 20px;
            margin-right: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .step.complete .step-icon::before { content: "✓"; color: var(--success); }
        .step.running .step-icon::before { content: "◐"; color: var(--accent); animation: spin 1s linear infinite; }
        .step.pending .step-icon::before { content: "○"; color: var(--text-secondary); }
        .step.failed .step-icon::before { content: "✗"; color: var(--error); }
        @keyframes spin { to { transform: rotate(360deg); } }
        .logs {
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.85rem;
        }
        .log-entry { margin-bottom: 5px; }
        .log-entry.info { color: var(--text-primary); }
        .log-entry.warning { color: var(--warning); }
        .log-entry.error { color: var(--error); }
        .log-time { color: var(--text-secondary); margin-right: 10px; }
        .session-info { font-size: 0.9rem; }
        .session-info div { margin-bottom: 8px; }
        .session-info label { color: var(--text-secondary); }
        select, input {
            background: var(--bg-card);
            border: 1px solid var(--bg-card);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            width: 100%;
        }
        .config-row { display: flex; gap: 10px; margin-bottom: 10px; }
        .config-row > * { flex: 1; }
    </style>
</head>
<body>
    <div class="header">
        <h1>GPU Deploy LLM - Test Dashboard</h1>
        <button class="btn" id="runBtn" onclick="runTest()">Run Test ▶</button>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Service Status</h2>
            <div id="serviceStatus">
                <div class="status-item">
                    <span class="status-dot gray" id="shopperDot"></span>
                    <span>Cloud-GPU-Shopper: <span id="shopperStatus">Checking...</span></span>
                </div>
                <div class="status-item" style="margin-left: 20px;">
                    <span>└─ /health: <span id="healthStatus">-</span></span>
                </div>
                <div class="status-item" style="margin-left: 20px;">
                    <span>└─ /ready: <span id="readyStatus">-</span></span>
                </div>
                <div class="status-item" style="margin-top: 15px;">
                    <span class="status-dot gray" id="vastaiDot"></span>
                    <span>Vast.ai: <span id="vastaiStatus">-</span></span>
                </div>
                <div class="status-item">
                    <span class="status-dot gray" id="tensordockDot"></span>
                    <span>TensorDock: <span id="tensordockStatus">-</span></span>
                </div>
            </div>
            <div id="sessionInfo" class="session-info" style="margin-top: 20px; display: none;">
                <h2 style="margin-bottom: 10px;">Active Session</h2>
                <div><label>ID:</label> <span id="sessionId">-</span></div>
                <div><label>Status:</label> <span id="sessionStatus">-</span></div>
                <div><label>Provider:</label> <span id="sessionProvider">-</span></div>
                <div><label>GPU:</label> <span id="sessionGpu">-</span></div>
                <div><label>Cost:</label> <span id="sessionCost">-</span></div>
            </div>
        </div>

        <div class="card">
            <h2>Test Progress</h2>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%"></div>
            </div>
            <div id="steps">
                <div class="step pending" data-step="calculate_requirements"><span class="step-icon"></span>Calculate requirements</div>
                <div class="step pending" data-step="query_inventory"><span class="step-icon"></span>Query inventory</div>
                <div class="step pending" data-step="select_offer"><span class="step-icon"></span>Select offer</div>
                <div class="step pending" data-step="create_session"><span class="step-icon"></span>Create session</div>
                <div class="step pending" data-step="wait_for_running"><span class="step-icon"></span>Wait for running</div>
                <div class="step pending" data-step="connect_ssh"><span class="step-icon"></span>Connect SSH</div>
                <div class="step pending" data-step="deploy_vllm"><span class="step-icon"></span>Deploy vLLM</div>
                <div class="step pending" data-step="verify_deployment"><span class="step-icon"></span>Verify deployment</div>
                <div class="step pending" data-step="cleanup"><span class="step-icon"></span>Cleanup</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Test Configuration</h2>
        <div class="config-row">
            <div>
                <label>Model</label>
                <select id="modelSelect">
                    <option value="TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama 1.1B (fastest)</option>
                    <option value="microsoft/phi-2">Phi-2</option>
                    <option value="Qwen/Qwen2-1.5B-Instruct">Qwen2 1.5B</option>
                    <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral 7B</option>
                </select>
            </div>
            <div>
                <label>Quantization</label>
                <select id="quantSelect">
                    <option value="none">None (FP16)</option>
                    <option value="awq">AWQ (4-bit)</option>
                    <option value="gptq">GPTQ (4-bit)</option>
                </select>
            </div>
            <div>
                <label>Max Price ($/hr)</label>
                <input type="number" id="maxPrice" value="0.50" step="0.05" min="0.1">
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Live Logs</h2>
        <div class="logs" id="logs"></div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                reconnectAttempts = 0;
                addLog('info', 'Connected to dashboard');
                fetchStatus();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleEvent(data);
            };

            ws.onclose = () => {
                addLog('warning', 'Disconnected, reconnecting...');
                const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
                reconnectAttempts++;
                setTimeout(connect, delay);
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
            };
        }

        function handleEvent(data) {
            switch (data.type) {
                case 'status':
                    updateStep(data.step, data.status);
                    updateProgress(data.step_index, data.total_steps);
                    break;
                case 'log':
                    addLog(data.level, data.message);
                    break;
                case 'session':
                    updateSession(data);
                    break;
                case 'result':
                    handleResult(data);
                    break;
                case 'init':
                    const status = JSON.parse(data.status);
                    updateServiceStatus(status);
                    break;
            }
        }

        function updateStep(step, status) {
            const stepEl = document.querySelector(`[data-step="${step}"]`);
            if (stepEl) {
                stepEl.className = `step ${status}`;
            }
        }

        function updateProgress(current, total) {
            const pct = ((current + 1) / total) * 100;
            document.getElementById('progressBar').style.width = `${pct}%`;
        }

        function addLog(level, message) {
            const logs = document.getElementById('logs');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = `log-entry ${level}`;
            entry.innerHTML = `<span class="log-time">${time}</span>${message}`;
            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;
        }

        function updateSession(data) {
            document.getElementById('sessionInfo').style.display = 'block';
            document.getElementById('sessionId').textContent = data.id;
            document.getElementById('sessionStatus').textContent = data.status;
            document.getElementById('sessionProvider').textContent = data.provider;
            document.getElementById('sessionGpu').textContent = `${data.gpu_type} x${data.gpu_count}`;
            document.getElementById('sessionCost').textContent = `$${data.price_per_hour.toFixed(3)}/hr`;
        }

        function handleResult(data) {
            const btn = document.getElementById('runBtn');
            btn.disabled = false;
            btn.textContent = 'Run Test ▶';

            if (data.passed) {
                addLog('info', '✓ Test PASSED');
            } else {
                addLog('error', `✗ Test FAILED: ${data.error}`);
            }
        }

        function updateServiceStatus(status) {
            const shopperDot = document.getElementById('shopperDot');
            const shopperStatus = document.getElementById('shopperStatus');

            if (status.shopper.healthy && status.shopper.ready) {
                shopperDot.className = 'status-dot green';
                shopperStatus.textContent = 'Connected';
            } else if (status.shopper.healthy) {
                shopperDot.className = 'status-dot yellow';
                shopperStatus.textContent = 'Starting...';
            } else {
                shopperDot.className = 'status-dot red';
                shopperStatus.textContent = 'Disconnected';
            }

            document.getElementById('healthStatus').textContent = status.shopper.healthy ? '200 OK' : 'Failed';
            document.getElementById('readyStatus').textContent = status.shopper.ready ? '200 OK' : '503';

            // Provider status
            const providers = status.providers || {};
            updateProvider('vastai', providers.vastai);
            updateProvider('tensordock', providers.tensordock);
        }

        function updateProvider(name, count) {
            const dot = document.getElementById(`${name}Dot`);
            const status = document.getElementById(`${name}Status`);
            if (count > 0) {
                dot.className = 'status-dot green';
                status.textContent = `Available (${count})`;
            } else {
                dot.className = 'status-dot gray';
                status.textContent = 'No offers';
            }
        }

        async function fetchStatus() {
            try {
                const resp = await fetch('/api/status');
                const status = await resp.json();
                updateServiceStatus(status);
            } catch (e) {
                console.error('Failed to fetch status:', e);
            }
        }

        async function runTest() {
            const btn = document.getElementById('runBtn');
            btn.disabled = true;
            btn.textContent = 'Running...';

            // Reset steps
            document.querySelectorAll('.step').forEach(el => {
                el.className = 'step pending';
            });
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('logs').innerHTML = '';

            const config = {
                model_id: document.getElementById('modelSelect').value,
                quantization: document.getElementById('quantSelect').value,
                max_price: parseFloat(document.getElementById('maxPrice').value),
                reservation_hours: 1,
                skip_verification: false,
                auto_cleanup: true
            };

            try {
                await fetch('/api/test/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                addLog('info', 'Test started...');
            } catch (e) {
                addLog('error', `Failed to start test: ${e}`);
                btn.disabled = false;
                btn.textContent = 'Run Test ▶';
            }
        }

        // Start connection
        connect();

        // Periodic status refresh
        setInterval(fetchStatus, 30000);
    </script>
</body>
</html>'''


def run_server(
    port: int = 8080,
    shopper_url: str = "http://localhost:8080",
    debug: bool = False,
    open_browser: bool = True,
):
    """Run the web server."""
    global _shopper_url, _debug
    _shopper_url = shopper_url
    _debug = debug

    import uvicorn

    if open_browser:
        webbrowser.open(f"http://localhost:{port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
