"""Ollama deployment - simple LLM server via SSH.

Ollama is much simpler than vLLM:
- Single binary installation (curl | sh)
- Built-in model registry (no HuggingFace dependency)
- OpenAI-compatible API at /v1/chat/completions
- Uses GGUF quantized models (smaller, faster to download)

Usage:
    async with SSHConnection(...) as ssh:
        deployer = OllamaDeployer(ssh)

        # Deploy and get local port
        result = await deployer.deploy("tinyllama")

        # Access via SSH tunnel at localhost
        async with ssh.forward_local_port(11434, 11434) as local_port:
            # Now accessible at http://localhost:{local_port}/v1/chat/completions
            response = httpx.post(
                f"http://localhost:{local_port}/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]}
            )
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from gpu_deploy_llm.ssh.connection import SSHConnection
from gpu_deploy_llm.utils.errors import DeploymentError

logger = logging.getLogger(__name__)

# Ollama default port
OLLAMA_PORT = 11434

# Small models that work on limited VRAM
SMALL_MODELS = [
    "tinyllama",      # 1.1B, ~1GB
    "phi",            # 2.7B, ~2GB
    "gemma:2b",       # 2B, ~2GB
    "qwen:1.8b",      # 1.8B, ~1GB
    "llama3.2:1b",    # 1B, ~1GB
]


@dataclass
class OllamaResult:
    """Result of Ollama deployment."""

    success: bool
    model: str = ""
    port: int = OLLAMA_PORT
    error: Optional[str] = None

    # Timing
    install_seconds: float = 0.0
    model_pull_seconds: float = 0.0


class OllamaDeployer:
    """Deploy Ollama LLM server via SSH.

    Ollama is simpler than vLLM because:
    - Single curl command to install
    - Uses its own model registry (not HuggingFace)
    - Smaller quantized models (GGUF)
    - Works on most instances with internet
    """

    def __init__(self, ssh: SSHConnection):
        self.ssh = ssh

    async def deploy(
        self,
        model: str = "tinyllama",
        timeout: float = 300.0,
    ) -> OllamaResult:
        """Deploy Ollama and pull a model.

        Args:
            model: Model to pull (e.g., "tinyllama", "phi", "gemma:2b")
            timeout: Total timeout for installation and model pull

        Returns:
            OllamaResult with deployment status
        """
        import time

        result = OllamaResult(success=False, model=model)

        try:
            # Step 1: Check if Ollama is already installed
            logger.info("Checking for existing Ollama installation...")
            check = await self.ssh.run("which ollama || echo 'not found'")

            if "not found" in check.stdout:
                # Step 2: Install Ollama
                logger.info("Installing Ollama...")
                start = time.time()

                install = await self.ssh.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    timeout=120,
                )

                result.install_seconds = time.time() - start

                if not install.success:
                    raise DeploymentError(
                        f"Failed to install Ollama: {install.stderr}",
                        stage="install_ollama",
                    )

                logger.info(f"Ollama installed in {result.install_seconds:.1f}s")
            else:
                logger.info("Ollama already installed")

            # Step 3: Start Ollama server in background
            logger.info("Starting Ollama server...")

            # Kill any existing Ollama process
            await self.ssh.run("pkill ollama || true")
            await asyncio.sleep(1)

            # Start server in background
            serve = await self.ssh.run(
                f"nohup ollama serve > /tmp/ollama.log 2>&1 &",
            )

            # Wait for server to be ready
            await asyncio.sleep(3)

            # Check if running
            running = await self.ssh.run("pgrep ollama")
            if not running.success:
                logs = await self.ssh.run("tail -20 /tmp/ollama.log")
                raise DeploymentError(
                    f"Ollama server failed to start. Logs:\n{logs.stdout}",
                    stage="start_ollama",
                )

            logger.info("Ollama server running")

            # Step 4: Pull the model
            logger.info(f"Pulling model: {model}")
            start = time.time()

            pull = await self.ssh.run(
                f"ollama pull {model}",
                timeout=int(timeout),
            )

            result.model_pull_seconds = time.time() - start

            if not pull.success:
                raise DeploymentError(
                    f"Failed to pull model {model}: {pull.stderr}",
                    stage="pull_model",
                )

            logger.info(f"Model pulled in {result.model_pull_seconds:.1f}s")

            # Step 5: Verify model is ready
            logger.info("Verifying model...")
            verify = await self.ssh.run(f"ollama list | grep {model}")

            if not verify.success:
                raise DeploymentError(
                    f"Model {model} not found after pull",
                    stage="verify_model",
                )

            result.success = True
            logger.info(f"Ollama deployment complete. Model: {model}, Port: {OLLAMA_PORT}")

        except DeploymentError:
            raise
        except Exception as e:
            result.error = str(e)
            raise DeploymentError(str(e), stage="deploy_ollama")

        return result

    async def chat(
        self,
        model: str,
        message: str,
        timeout: float = 60.0,
    ) -> str:
        """Send a chat message and get response (via SSH).

        For testing without port forwarding.
        """
        cmd = f'''curl -s http://localhost:{OLLAMA_PORT}/api/chat -d '{{"model": "{model}", "messages": [{{"role": "user", "content": "{message}"}}], "stream": false}}' '''

        result = await self.ssh.run(cmd, timeout=int(timeout))

        if not result.success:
            raise DeploymentError(
                f"Chat failed: {result.stderr}",
                stage="chat",
            )

        return result.stdout

    async def get_models(self) -> list:
        """List available models."""
        result = await self.ssh.run("ollama list 2>/dev/null || echo 'error'")

        if "error" in result.stdout or not result.success:
            return []

        models = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if parts:
                    models.append(parts[0])

        return models

    async def stop(self) -> bool:
        """Stop Ollama server."""
        result = await self.ssh.run("pkill ollama || true")
        logger.info("Ollama server stopped")
        return True

    async def get_logs(self, lines: int = 50) -> str:
        """Get Ollama logs."""
        result = await self.ssh.run(f"tail -{lines} /tmp/ollama.log 2>/dev/null || echo 'No logs'")
        return result.stdout


async def deploy_ollama_with_tunnel(
    ssh: SSHConnection,
    model: str = "tinyllama",
    local_port: int = 11434,
) -> tuple:
    """Deploy Ollama and set up local port forwarding.

    Returns a context manager for the tunnel and deployment result.

    Usage:
        async with SSHConnection(...) as ssh:
            result, tunnel_ctx = await deploy_ollama_with_tunnel(ssh, "tinyllama")

            async with tunnel_ctx as port:
                # Access at http://localhost:{port}/v1/chat/completions
                response = httpx.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [...]}
                )
    """
    deployer = OllamaDeployer(ssh)
    result = await deployer.deploy(model)

    # Create tunnel context
    tunnel_ctx = ssh.forward_local_port(local_port, OLLAMA_PORT)

    return result, tunnel_ctx
