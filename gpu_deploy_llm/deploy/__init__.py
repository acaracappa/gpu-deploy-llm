"""vLLM deployment, verification, and benchmarking.

Two deployment modes are supported:

1. SSH mode (VLLMDeployer):
   - SSH into instance, install vLLM via pip or Docker
   - More control over deployment
   - Requires internet access for pip install

2. Docker entrypoint mode (DockerEntrypointDeployer):
   - Shopper starts pre-built Docker image automatically
   - No internet access required on instance
   - Recommended for instances without general internet
"""

from .vllm import VLLMDeployer, DeploymentConfig, DeploymentResult, DeploymentMode
from .health import (
    HealthChecker,
    HealthCheckerHTTP,
    HealthCheckResult,
    VerificationResult,
    CheckStatus,
)
from .benchmark import Benchmarker, BenchmarkResult, PromptResult, run_benchmark
from .benchmark_store import BenchmarkStore, BenchmarkRecord, create_benchmark_record
from .docker_entrypoint import (
    DockerEntrypointDeployer,
    DockerEntrypointConfig,
    DockerEntrypointResult,
    deploy_vllm_entrypoint,
    VLLM_IMAGE,
    TGI_IMAGE,
    OLLAMA_IMAGE,
)

__all__ = [
    # SSH-based deployment
    "VLLMDeployer",
    "DeploymentConfig",
    "DeploymentResult",
    "DeploymentMode",
    # Docker entrypoint deployment (recommended)
    "DockerEntrypointDeployer",
    "DockerEntrypointConfig",
    "DockerEntrypointResult",
    "deploy_vllm_entrypoint",
    "VLLM_IMAGE",
    "TGI_IMAGE",
    "OLLAMA_IMAGE",
    # Health checking
    "HealthChecker",
    "HealthCheckerHTTP",
    "HealthCheckResult",
    "VerificationResult",
    "CheckStatus",
    # Benchmarking
    "Benchmarker",
    "BenchmarkResult",
    "PromptResult",
    "run_benchmark",
    "BenchmarkStore",
    "BenchmarkRecord",
    "create_benchmark_record",
]
