"""Docker entrypoint-based deployment via cloud-gpu-shopper.

This deployment mode uses the shopper's entrypoint feature to start
pre-built Docker images (vLLM, TGI) directly without SSH installation.

Benefits:
- No internet access required on instance (image is pulled by provider)
- No pip installation (uses pre-built container)
- Faster startup (no dependency installation)
- Works on instances that block general internet but whitelist container registries

Usage:
    # Create session with entrypoint mode
    response = await client.create_session(
        offer_id="...",
        launch_mode="entrypoint",
        docker_image="vllm/vllm-openai:latest",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        exposed_ports=[8000],
    )

    # vLLM starts automatically, just verify it's ready
    deployer = DockerEntrypointDeployer(client)
    result = await deployer.wait_for_ready(session)
"""

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from typing import Optional, List

import httpx

from gpu_deploy_llm.client.shopper import ShopperClient
from gpu_deploy_llm.client.models import (
    Session,
    SessionStatus,
    LaunchMode,
    CreateSessionResponse,
)
from gpu_deploy_llm.utils.errors import DeploymentError

logger = logging.getLogger(__name__)

# Pre-built Docker images (matching cloud-gpu-shopper constants)
VLLM_IMAGE = "vllm/vllm-openai:latest"
TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:latest"
OLLAMA_IMAGE = "ollama/ollama:latest"

# Default ports
VLLM_PORT = 8000
TGI_PORT = 80


@dataclass
class DockerEntrypointConfig:
    """Configuration for Docker entrypoint deployment."""

    model_id: str
    docker_image: str = VLLM_IMAGE
    port: int = VLLM_PORT
    quantization: Optional[str] = None

    # Shopper configuration
    reservation_hours: int = 1
    storage_policy: str = "destroy"

    @classmethod
    def vllm(
        cls,
        model_id: str,
        quantization: Optional[str] = None,
        reservation_hours: int = 1,
    ) -> "DockerEntrypointConfig":
        """Create config for vLLM deployment."""
        return cls(
            model_id=model_id,
            docker_image=VLLM_IMAGE,
            port=VLLM_PORT,
            quantization=quantization,
            reservation_hours=reservation_hours,
        )

    @classmethod
    def tgi(
        cls,
        model_id: str,
        quantization: Optional[str] = None,
        reservation_hours: int = 1,
    ) -> "DockerEntrypointConfig":
        """Create config for TGI deployment."""
        return cls(
            model_id=model_id,
            docker_image=TGI_IMAGE,
            port=TGI_PORT,
            quantization=quantization,
            reservation_hours=reservation_hours,
        )


@dataclass
class DockerEntrypointResult:
    """Result of Docker entrypoint deployment."""

    success: bool
    session_id: str = ""
    endpoint: str = ""
    api_key: str = ""  # Generated API key for vLLM
    model_id: str = ""
    error: Optional[str] = None

    # SSH details (still provided for debugging)
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: Optional[str] = None
    ssh_private_key: str = ""


class DockerEntrypointDeployer:
    """Deploy LLM inference servers using shopper's Docker entrypoint mode.

    This deployer creates sessions with pre-built Docker images that start
    automatically, eliminating the need for SSH-based installation.

    The shopper handles:
    - Provisioning the GPU instance
    - Starting the Docker container with the specified image
    - Exposing the API port

    This deployer handles:
    - Creating the session with correct parameters
    - Waiting for the API to become available
    - Health checking the deployment
    """

    def __init__(
        self,
        client: ShopperClient,
        http_timeout: float = 30.0,
    ):
        """Initialize the deployer.

        Args:
            client: Shopper client for session management
            http_timeout: Timeout for HTTP health checks
        """
        self.client = client
        self.http_timeout = http_timeout
        self._http_client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "DockerEntrypointDeployer":
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.http_timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def http_client(self) -> httpx.AsyncClient:
        if not self._http_client:
            raise RuntimeError("Deployer not initialized. Use 'async with' context.")
        return self._http_client

    async def deploy(
        self,
        offer_id: str,
        config: DockerEntrypointConfig,
    ) -> DockerEntrypointResult:
        """Deploy using Docker entrypoint mode.

        Args:
            offer_id: ID of the GPU offer to provision
            config: Deployment configuration

        Returns:
            DockerEntrypointResult with endpoint and connection details
        """
        result = DockerEntrypointResult(
            success=False,
            model_id=config.model_id,
        )

        try:
            # Step 1: Create session with entrypoint mode
            logger.info(f"Creating entrypoint session for {config.model_id}")
            response = await self.client.create_session(
                offer_id=offer_id,
                reservation_hours=config.reservation_hours,
                workload_type="llm_vllm",
                storage_policy=config.storage_policy,
                launch_mode="entrypoint",
                docker_image=config.docker_image,
                model_id=config.model_id,
                exposed_ports=[config.port],
                quantization=config.quantization,
            )

            result.session_id = response.session.id
            result.ssh_private_key = response.ssh_private_key

            logger.info(f"Session created: {result.session_id}")

            # Step 2: Wait for session to be running
            session = await self.client.wait_for_running(
                session_id=result.session_id,
                provider=response.session.provider,
            )

            result.ssh_host = session.ssh_host
            result.ssh_port = session.ssh_port
            result.ssh_user = session.ssh_user

            # Step 3: Get API endpoint
            # Use api_endpoint if provided, otherwise construct from SSH host
            if session.api_endpoint:
                result.endpoint = session.api_endpoint
            elif session.ssh_host:
                # Construct endpoint from SSH host
                api_port = session.api_port or config.port
                result.endpoint = f"http://{session.ssh_host}:{api_port}/v1"
            else:
                raise DeploymentError(
                    "No API endpoint or SSH host available",
                    stage="get_endpoint",
                )

            logger.info(f"API endpoint: {result.endpoint}")

            # Step 4: Wait for API to be ready
            logger.info("Waiting for API to be ready...")
            await self._wait_for_api(result.endpoint)

            result.success = True
            logger.info(f"Deployment successful: {result.endpoint}")

        except DeploymentError:
            raise
        except Exception as e:
            result.error = str(e)
            logger.error(f"Deployment failed: {e}")
            raise DeploymentError(str(e), stage="deploy_entrypoint")

        return result

    async def _wait_for_api(
        self,
        endpoint: str,
        timeout: float = 300.0,  # 5 minutes
        interval: float = 10.0,
    ) -> None:
        """Wait for the API to be ready.

        Polls the /v1/models endpoint until it returns successfully.
        """
        import time

        deadline = time.time() + timeout
        last_error = None

        while time.time() < deadline:
            try:
                # Try to get models list
                response = await self.http_client.get(f"{endpoint}/models")
                if response.status_code == 200:
                    logger.info("API is ready")
                    return
                elif response.status_code == 401:
                    # API key required but endpoint is up
                    logger.info("API is ready (auth required)")
                    return
                else:
                    last_error = f"HTTP {response.status_code}"
            except httpx.ConnectError:
                last_error = "Connection refused"
            except httpx.TimeoutException:
                last_error = "Connection timeout"
            except Exception as e:
                last_error = str(e)

            logger.debug(f"API not ready: {last_error}")
            await asyncio.sleep(interval)

        raise DeploymentError(
            f"API not ready after {timeout}s: {last_error}",
            stage="wait_for_api",
        )

    async def verify_deployment(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """Verify deployment is working correctly.

        Args:
            endpoint: API endpoint URL
            api_key: Optional API key for authenticated endpoints

        Returns:
            Verification results
        """
        results = {
            "models_endpoint": False,
            "model_loaded": False,
            "inference_works": False,
            "model_name": None,
            "error": None,
        }

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            # Check /v1/models
            response = await self.http_client.get(
                f"{endpoint}/models",
                headers=headers,
            )
            if response.status_code == 200:
                results["models_endpoint"] = True
                data = response.json()
                if data.get("data"):
                    results["model_loaded"] = True
                    results["model_name"] = data["data"][0].get("id")
            else:
                results["error"] = f"Models endpoint: HTTP {response.status_code}"
                return results

            # Try a simple completion
            response = await self.http_client.post(
                f"{endpoint}/completions",
                headers=headers,
                json={
                    "model": results["model_name"],
                    "prompt": "Hello",
                    "max_tokens": 5,
                },
                timeout=60.0,
            )
            if response.status_code == 200:
                results["inference_works"] = True
            else:
                results["error"] = f"Completion: HTTP {response.status_code}"

        except Exception as e:
            results["error"] = str(e)

        return results


async def deploy_vllm_entrypoint(
    client: ShopperClient,
    offer_id: str,
    model_id: str,
    quantization: Optional[str] = None,
    reservation_hours: int = 1,
) -> DockerEntrypointResult:
    """Convenience function to deploy vLLM using Docker entrypoint mode.

    This is the recommended deployment method for instances that:
    - Don't have general internet access
    - Have Docker pre-installed
    - Support container registry access

    Args:
        client: Shopper client
        offer_id: GPU offer ID
        model_id: HuggingFace model ID
        quantization: Optional quantization method
        reservation_hours: Session duration

    Returns:
        DockerEntrypointResult with deployment details
    """
    config = DockerEntrypointConfig.vllm(
        model_id=model_id,
        quantization=quantization,
        reservation_hours=reservation_hours,
    )

    async with DockerEntrypointDeployer(client) as deployer:
        return await deployer.deploy(offer_id, config)
