"""Verification and health checks for vLLM deployments.

Verification checks:
1. GPU Status - nvidia-smi shows healthy GPUs
2. Container Running - docker ps shows vllm-server
3. Health Endpoint - GET /health returns 200
4. Model Loaded - GET /v1/models returns model info
5. Inference Test - POST /v1/completions generates output

Two modes of operation:
1. SSH-curl mode (HealthChecker): Runs curl via SSH on remote instance
2. HTTP mode (HealthCheckerHTTP): Uses httpx with SSH port forwarding
"""

import asyncio
import logging
import shlex
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from gpu_deploy_llm.ssh.connection import SSHConnection, GPUStatus
from gpu_deploy_llm.utils.errors import VerificationError

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Status of a health check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str = ""
    details: Optional[dict] = None
    duration_ms: float = 0.0


@dataclass
class VerificationResult:
    """Result of full deployment verification."""

    passed: bool
    checks: List[HealthCheckResult] = field(default_factory=list)
    gpu_status: List[GPUStatus] = field(default_factory=list)
    model_info: Optional[dict] = None
    inference_result: Optional[str] = None
    total_duration_ms: float = 0.0

    @property
    def summary(self) -> str:
        """Get summary of verification results."""
        passed = sum(1 for c in self.checks if c.status == CheckStatus.PASSED)
        failed = sum(1 for c in self.checks if c.status == CheckStatus.FAILED)
        return f"{passed}/{len(self.checks)} checks passed, {failed} failed"


class HealthChecker:
    """Health checker for vLLM deployments."""

    def __init__(
        self,
        ssh: SSHConnection,
        endpoint: str,
        api_key: str,
        container_name: str = "vllm-server",
        timeout: float = 30.0,
    ):
        """Initialize health checker.

        Args:
            ssh: SSH connection to GPU instance
            endpoint: vLLM API endpoint (e.g., http://host:8000/v1)
            api_key: vLLM API key for authentication
            container_name: Name of vLLM container
            timeout: HTTP request timeout
        """
        self.ssh = ssh
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.container_name = container_name
        self.timeout = timeout

    async def _get_gpu_memory(self) -> str:
        """Get GPU memory usage from remote node."""
        try:
            result = await self.ssh.run(
                "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null"
            )
            if result.success and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    used, total = [p.strip() for p in parts[:2]]
                    try:
                        total_int = int(total)
                        used_int = int(used)
                        pct = int(used_int / total_int * 100) if total_int > 0 else 0
                        return f"GPU mem: {used}MB/{total}MB ({pct}%)"
                    except (ValueError, ZeroDivisionError):
                        return f"GPU mem: {used}MB/{total}MB (calc error)"
        except Exception:
            pass
        return "GPU mem: unknown"

    async def _get_remote_debug_info(self) -> str:
        """Get debug info from remote node when vLLM not responding."""
        try:
            # Check if vLLM process is running
            pid_result = await self.ssh.run("pgrep -f 'vllm.entrypoints' | head -1")
            if pid_result.success and pid_result.stdout.strip():
                pid = pid_result.stdout.strip()

                # Check if port 8000 is listening
                port_result = await self.ssh.run("ss -tlnp | grep :8000 || netstat -tlnp 2>/dev/null | grep :8000 || echo 'port not listening'")
                port_status = "port 8000 OPEN" if ":8000" in port_result.stdout else "port 8000 NOT listening"

                # Get last log line
                log_result = await self.ssh.run("tail -1 /tmp/vllm.log 2>/dev/null")
                last_log = log_result.stdout.strip()[:80] if log_result.success else ""

                gpu_mem = await self._get_gpu_memory()
                return f"PID {pid}, {port_status}, {gpu_mem}, log: {last_log}"
            else:
                # Process not running - check why
                log_result = await self.ssh.run("tail -5 /tmp/vllm.log 2>/dev/null")
                if log_result.success and log_result.stdout.strip():
                    # Extract key error info
                    log_lines = log_result.stdout.strip().split('\n')
                    last_lines = ' | '.join(l[:60] for l in log_lines[-3:])
                    return f"Process not running! Logs: {last_lines}"
                return "Process not running, no logs"
        except Exception as e:
            return f"debug error: {e}"

    async def verify_all(
        self,
        skip_inference: bool = False,
    ) -> VerificationResult:
        """Run all verification checks.

        Args:
            skip_inference: Skip inference test (faster)

        Returns:
            VerificationResult with all check results
        """
        import time

        start_time = time.time()
        result = VerificationResult(passed=True)

        # Run checks in sequence
        checks = [
            self.check_gpu_status,
            self.check_container_running,
            self.check_health_endpoint,
            self.check_model_loaded,
        ]

        if not skip_inference:
            checks.append(self.check_inference)

        for check_func in checks:
            check_result = await check_func()
            result.checks.append(check_result)

            if check_result.status == CheckStatus.FAILED:
                result.passed = False

            # Store additional data
            if check_result.name == "gpu_status" and check_result.details:
                result.gpu_status = check_result.details.get("gpus", [])
            elif check_result.name == "model_loaded" and check_result.details:
                result.model_info = check_result.details
            elif check_result.name == "inference" and check_result.details:
                result.inference_result = check_result.details.get("output")

        result.total_duration_ms = (time.time() - start_time) * 1000
        return result

    async def check_gpu_status(self) -> HealthCheckResult:
        """Check GPU health via nvidia-smi."""
        import time

        start = time.time()

        try:
            gpus = await self.ssh.get_gpu_status()

            if not gpus:
                return HealthCheckResult(
                    name="gpu_status",
                    status=CheckStatus.FAILED,
                    message="No GPUs found",
                    duration_ms=(time.time() - start) * 1000,
                )

            unhealthy = [g for g in gpus if not g.is_healthy()]
            if unhealthy:
                return HealthCheckResult(
                    name="gpu_status",
                    status=CheckStatus.FAILED,
                    message=f"{len(unhealthy)} GPU(s) unhealthy",
                    details={"gpus": gpus, "unhealthy": unhealthy},
                    duration_ms=(time.time() - start) * 1000,
                )

            return HealthCheckResult(
                name="gpu_status",
                status=CheckStatus.PASSED,
                message=f"{len(gpus)} GPU(s) healthy",
                details={"gpus": gpus},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="gpu_status",
                status=CheckStatus.FAILED,
                message=f"GPU check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_container_running(self) -> HealthCheckResult:
        """Check if vLLM is running (container or process)."""
        import time

        start = time.time()

        try:
            # First check for Docker container
            containers = await self.ssh.get_running_containers()
            if self.container_name in containers:
                return HealthCheckResult(
                    name="container_running",
                    status=CheckStatus.PASSED,
                    message=f"Container '{self.container_name}' is running",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Check for pip mode process
            pid_result = await self.ssh.run("pgrep -f 'vllm.entrypoints' | head -1")
            if pid_result.success and pid_result.stdout.strip():
                pid = pid_result.stdout.strip()
                return HealthCheckResult(
                    name="container_running",
                    status=CheckStatus.PASSED,
                    message=f"vLLM process running (PID {pid})",
                    details={"pid": pid, "mode": "pip"},
                    duration_ms=(time.time() - start) * 1000,
                )

            # Neither container nor process found - check logs for why
            log_result = await self.ssh.run("tail -5 /tmp/vllm.log 2>/dev/null")
            last_logs = log_result.stdout.strip() if log_result.success else "no logs"

            return HealthCheckResult(
                name="container_running",
                status=CheckStatus.FAILED,
                message=f"vLLM not running (no container or process)",
                details={"last_logs": last_logs},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="container_running",
                status=CheckStatus.FAILED,
                message=f"vLLM check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_health_endpoint(self) -> HealthCheckResult:
        """Check vLLM /health endpoint via SSH (runs curl on remote instance)."""
        import time

        start = time.time()
        # Use localhost on remote instance since vLLM binds to 0.0.0.0
        port = 8000  # vLLM port

        try:
            # Run curl on remote instance via SSH
            result = await self.ssh.run(
                f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health",
                timeout=int(self.timeout),
            )

            if result.success and result.stdout.strip() == "200":
                return HealthCheckResult(
                    name="health_endpoint",
                    status=CheckStatus.PASSED,
                    message="Health endpoint returned 200",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Try to get response body for debugging
            body_result = await self.ssh.run(
                f"curl -s http://localhost:{port}/health 2>&1 | head -c 500",
                timeout=int(self.timeout),
            )
            body = body_result.stdout if body_result.success else ""

            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Health endpoint returned {result.stdout.strip() or 'no response'}",
                details={"status_code": result.stdout.strip(), "body": body},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Health endpoint check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_model_loaded(self) -> HealthCheckResult:
        """Check if model is loaded via /v1/models (runs curl on remote instance)."""
        import time
        import json as json_module

        start = time.time()
        port = 8000  # vLLM port

        try:
            # Run curl on remote instance via SSH
            # Use environment variable to avoid exposing API key in process list
            result = await self.ssh.run(
                f"API_KEY={shlex.quote(self.api_key)} "
                f"curl -s -H \"Authorization: Bearer $API_KEY\" http://localhost:{port}/v1/models",
                timeout=int(self.timeout),
            )

            if result.success and result.stdout.strip():
                try:
                    data = json_module.loads(result.stdout)
                    models = data.get("data", [])

                    if models:
                        return HealthCheckResult(
                            name="model_loaded",
                            status=CheckStatus.PASSED,
                            message=f"Model loaded: {models[0].get('id', 'unknown')}",
                            details={"models": models},
                            duration_ms=(time.time() - start) * 1000,
                        )

                    return HealthCheckResult(
                        name="model_loaded",
                        status=CheckStatus.FAILED,
                        message="No models loaded",
                        details=data,
                        duration_ms=(time.time() - start) * 1000,
                    )
                except json_module.JSONDecodeError:
                    return HealthCheckResult(
                        name="model_loaded",
                        status=CheckStatus.FAILED,
                        message=f"Invalid JSON response: {result.stdout[:100]}",
                        duration_ms=(time.time() - start) * 1000,
                    )

            # Get debug info when no response
            debug_info = await self._get_remote_debug_info()
            return HealthCheckResult(
                name="model_loaded",
                status=CheckStatus.FAILED,
                message=f"Models endpoint failed: {result.stderr or 'no response'} - {debug_info}",
                details={"stderr": result.stderr, "debug": debug_info},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="model_loaded",
                status=CheckStatus.FAILED,
                message=f"Model check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_inference(self) -> HealthCheckResult:
        """Test inference via /v1/completions (runs curl on remote instance)."""
        import time
        import json as json_module

        start = time.time()
        port = 8000  # vLLM port

        try:
            # Build the JSON payload - escape for shell
            payload = json_module.dumps({
                "model": "default",
                "prompt": "Hello, I am",
                "max_tokens": 10,
                "temperature": 0.0,
            })

            # Run curl on remote instance via SSH (longer timeout for inference)
            # Use environment variable to avoid exposing API key in process list
            result = await self.ssh.run(
                f"API_KEY={shlex.quote(self.api_key)} "
                f"curl -s -X POST -H \"Authorization: Bearer $API_KEY\" "
                f"-H 'Content-Type: application/json' "
                f"-d '{payload}' http://localhost:{port}/v1/completions",
                timeout=60,  # Longer timeout for inference
            )

            if result.success and result.stdout.strip():
                try:
                    data = json_module.loads(result.stdout)
                    choices = data.get("choices", [])

                    if choices and choices[0].get("text"):
                        output = choices[0]["text"]
                        return HealthCheckResult(
                            name="inference",
                            status=CheckStatus.PASSED,
                            message="Inference successful",
                            details={"output": output, "usage": data.get("usage")},
                            duration_ms=(time.time() - start) * 1000,
                        )

                    return HealthCheckResult(
                        name="inference",
                        status=CheckStatus.FAILED,
                        message="Inference returned no output",
                        details=data,
                        duration_ms=(time.time() - start) * 1000,
                    )
                except json_module.JSONDecodeError:
                    return HealthCheckResult(
                        name="inference",
                        status=CheckStatus.FAILED,
                        message=f"Invalid JSON response: {result.stdout[:100]}",
                        duration_ms=(time.time() - start) * 1000,
                    )

            return HealthCheckResult(
                name="inference",
                status=CheckStatus.FAILED,
                message=f"Inference failed: {result.stderr or 'no response'}",
                details={
                    "stderr": result.stderr,
                    "stdout": result.stdout[:500] if result.stdout else "",
                },
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="inference",
                status=CheckStatus.FAILED,
                message=f"Inference check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def wait_for_ready(
        self,
        timeout: float = 300.0,  # 5 minutes for model loading
        interval: float = 10.0,
        progress_callback=None,
    ) -> bool:
        """Wait for vLLM to be ready (model loaded).

        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            progress_callback: Optional async callback(elapsed_secs, message) for progress updates

        Returns:
            True if ready, raises TimeoutError otherwise
        """
        import time

        start_time = time.time()
        deadline = start_time + timeout
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            elapsed = time.time() - start_time

            try:
                # Check health endpoint first
                health_result = await self.check_health_endpoint()
                health_status = "up" if health_result.status == CheckStatus.PASSED else "waiting"

                # Check model loaded
                result = await self.check_model_loaded()

                if result.status == CheckStatus.PASSED:
                    if progress_callback:
                        await progress_callback(elapsed, f"Model loaded successfully after {elapsed:.0f}s")
                    return True

                # Detailed progress message
                msg = f"Health: {health_status}, Model: {result.message}"
                logger.info(f"[{elapsed:.0f}s] {msg}")

                if progress_callback:
                    await progress_callback(elapsed, msg)

            except Exception as e:
                # Get remote debug info when check fails
                remote_info = await self._get_remote_debug_info()
                msg = f"Check error: {type(e).__name__}: {e} - {remote_info}"
                logger.warning(f"[{elapsed:.0f}s] {msg}")
                if progress_callback:
                    await progress_callback(elapsed, msg)

            await asyncio.sleep(interval)

        raise TimeoutError(f"vLLM not ready after {timeout}s")


class HealthCheckerHTTP:
    """HTTP-based health checker for vLLM deployments.

    This checker uses direct HTTP requests via httpx, intended for use
    with SSH port forwarding. It provides cleaner error handling and
    supports streaming responses.

    Usage with port forwarding:
        async with ssh.forward_local_port(0, 8000) as local_port:
            checker = HealthCheckerHTTP(
                endpoint=f"http://localhost:{local_port}",
                api_key=api_key,
            )
            await checker.wait_for_ready()
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout: float = 30.0,
    ):
        """Initialize HTTP health checker.

        Args:
            endpoint: vLLM API endpoint (e.g., http://localhost:8000)
            api_key: vLLM API key for authentication
            timeout: HTTP request timeout
        """
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for HealthCheckerHTTP")

        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "HealthCheckerHTTP":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client. Required if not using context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        WARNING: If not using this class as a context manager (async with),
        you must call close() when done to avoid resource leaks.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def check_health_endpoint(self) -> HealthCheckResult:
        """Check vLLM /health endpoint via HTTP."""
        import time

        start = time.time()
        client = self._get_client()

        try:
            response = await client.get(f"{self.endpoint}/health")

            if response.status_code == 200:
                return HealthCheckResult(
                    name="health_endpoint",
                    status=CheckStatus.PASSED,
                    message="Health endpoint returned 200",
                    duration_ms=(time.time() - start) * 1000,
                )

            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Health endpoint returned {response.status_code}",
                details={"status_code": response.status_code, "body": response.text[:500]},
                duration_ms=(time.time() - start) * 1000,
            )

        except httpx.ConnectError as e:
            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Connection failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )
        except httpx.TimeoutException:
            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Request timeout ({self.timeout}s)",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Health check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_model_loaded(self) -> HealthCheckResult:
        """Check if model is loaded via /v1/models."""
        import time

        start = time.time()
        client = self._get_client()
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = await client.get(f"{self.endpoint}/v1/models", headers=headers)

            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])

                if models:
                    return HealthCheckResult(
                        name="model_loaded",
                        status=CheckStatus.PASSED,
                        message=f"Model loaded: {models[0].get('id', 'unknown')}",
                        details={"models": models},
                        duration_ms=(time.time() - start) * 1000,
                    )

                return HealthCheckResult(
                    name="model_loaded",
                    status=CheckStatus.FAILED,
                    message="No models loaded",
                    details=data,
                    duration_ms=(time.time() - start) * 1000,
                )

            return HealthCheckResult(
                name="model_loaded",
                status=CheckStatus.FAILED,
                message=f"Models endpoint returned {response.status_code}",
                details={"status_code": response.status_code},
                duration_ms=(time.time() - start) * 1000,
            )

        except httpx.ConnectError:
            return HealthCheckResult(
                name="model_loaded",
                status=CheckStatus.FAILED,
                message="Connection failed - vLLM not responding",
                duration_ms=(time.time() - start) * 1000,
            )
        except httpx.TimeoutException:
            return HealthCheckResult(
                name="model_loaded",
                status=CheckStatus.FAILED,
                message=f"Request timeout ({self.timeout}s)",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="model_loaded",
                status=CheckStatus.FAILED,
                message=f"Model check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_inference(self) -> HealthCheckResult:
        """Test inference via /v1/completions."""
        import time

        start = time.time()
        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "default",
            "prompt": "Hello, I am",
            "max_tokens": 10,
            "temperature": 0.0,
        }

        try:
            # Use longer timeout for inference
            response = await client.post(
                f"{self.endpoint}/v1/completions",
                headers=headers,
                json=payload,
                timeout=60.0,
            )

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])

                if choices and choices[0].get("text"):
                    output = choices[0]["text"]
                    return HealthCheckResult(
                        name="inference",
                        status=CheckStatus.PASSED,
                        message="Inference successful",
                        details={"output": output, "usage": data.get("usage")},
                        duration_ms=(time.time() - start) * 1000,
                    )

                return HealthCheckResult(
                    name="inference",
                    status=CheckStatus.FAILED,
                    message="Inference returned no output",
                    details=data,
                    duration_ms=(time.time() - start) * 1000,
                )

            return HealthCheckResult(
                name="inference",
                status=CheckStatus.FAILED,
                message=f"Inference endpoint returned {response.status_code}",
                details={"status_code": response.status_code, "body": response.text[:500]},
                duration_ms=(time.time() - start) * 1000,
            )

        except httpx.ConnectError:
            return HealthCheckResult(
                name="inference",
                status=CheckStatus.FAILED,
                message="Connection failed",
                duration_ms=(time.time() - start) * 1000,
            )
        except httpx.TimeoutException:
            return HealthCheckResult(
                name="inference",
                status=CheckStatus.FAILED,
                message="Inference timeout (60s)",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name="inference",
                status=CheckStatus.FAILED,
                message=f"Inference check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def wait_for_ready(
        self,
        timeout: float = 300.0,
        interval: float = 10.0,
        progress_callback=None,
    ) -> bool:
        """Wait for vLLM to be ready (model loaded).

        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            progress_callback: Optional async callback(elapsed_secs, message)

        Returns:
            True if ready, raises TimeoutError otherwise
        """
        import time

        start_time = time.time()
        deadline = start_time + timeout

        while time.time() < deadline:
            elapsed = time.time() - start_time

            # Check health endpoint first
            health_result = await self.check_health_endpoint()
            health_status = "up" if health_result.status == CheckStatus.PASSED else "waiting"

            # Check model loaded
            model_result = await self.check_model_loaded()

            if model_result.status == CheckStatus.PASSED:
                if progress_callback:
                    await progress_callback(elapsed, f"Model loaded after {elapsed:.0f}s")
                return True

            # Progress message
            msg = f"Health: {health_status}, Model: {model_result.message}"
            logger.info(f"[{elapsed:.0f}s] {msg}")

            if progress_callback:
                await progress_callback(elapsed, msg)

            await asyncio.sleep(interval)

        raise TimeoutError(f"vLLM not ready after {timeout}s")
