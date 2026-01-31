"""Verification and health checks for vLLM deployments.

Verification checks:
1. GPU Status - nvidia-smi shows healthy GPUs
2. Container Running - docker ps shows vllm-server
3. Health Endpoint - GET /health returns 200
4. Model Loaded - GET /v1/models returns model info
5. Inference Test - POST /v1/completions generates output
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

import httpx

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
        """Check if vLLM container is running."""
        import time

        start = time.time()

        try:
            containers = await self.ssh.get_running_containers()

            if self.container_name in containers:
                return HealthCheckResult(
                    name="container_running",
                    status=CheckStatus.PASSED,
                    message=f"Container '{self.container_name}' is running",
                    duration_ms=(time.time() - start) * 1000,
                )

            return HealthCheckResult(
                name="container_running",
                status=CheckStatus.FAILED,
                message=f"Container '{self.container_name}' not running",
                details={"running_containers": containers},
                duration_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return HealthCheckResult(
                name="container_running",
                status=CheckStatus.FAILED,
                message=f"Container check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_health_endpoint(self) -> HealthCheckResult:
        """Check vLLM /health endpoint."""
        import time

        start = time.time()
        # vLLM health endpoint is at root /health, not /v1/health
        health_url = self.endpoint.replace("/v1", "") + "/health"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(health_url)

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

        except Exception as e:
            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Health endpoint check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def check_model_loaded(self) -> HealthCheckResult:
        """Check if model is loaded via /v1/models."""
        import time

        start = time.time()
        models_url = f"{self.endpoint}/models"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    models_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )

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
        completions_url = f"{self.endpoint}/completions"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for inference
                response = await client.post(
                    completions_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "default",  # vLLM uses the loaded model
                        "prompt": "Hello, I am",
                        "max_tokens": 10,
                        "temperature": 0.0,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])

                    if choices and choices[0].get("text"):
                        output = choices[0]["text"]
                        return HealthCheckResult(
                            name="inference",
                            status=CheckStatus.PASSED,
                            message=f"Inference successful",
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
                    message=f"Inference returned {response.status_code}",
                    details={
                        "status_code": response.status_code,
                        "body": response.text[:500],
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
    ) -> bool:
        """Wait for vLLM to be ready (model loaded).

        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if ready, raises TimeoutError otherwise
        """
        import time

        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                result = await self.check_model_loaded()
                if result.status == CheckStatus.PASSED:
                    return True
                logger.info(f"Waiting for model to load... ({result.message})")
            except Exception as e:
                logger.debug(f"Model check error: {e}")

            await asyncio.sleep(interval)

        raise TimeoutError(f"vLLM not ready after {timeout}s")
