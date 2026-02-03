"""Test vLLM deployment on Vast.ai using entrypoint mode.

This script:
1. Finds a suitable GPU offer on Vast.ai
2. Creates a session with vLLM Docker image (entrypoint mode)
3. Sets up SSH port forwarding to access locally
4. Tests the OpenAI-compatible API
"""

import asyncio
import logging
import sys

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def main():
    # Import here to ensure venv is active
    from gpu_deploy_llm.client.shopper import ShopperClient
    from gpu_deploy_llm.ssh.connection import SSHConnection

    VLLM_IMAGE = "vllm/vllm-openai:latest"
    VLLM_PORT = 8000

    shopper_url = "http://localhost:8080"  # cloud-gpu-shopper URL
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for testing
    max_price = 0.50  # $/hr
    local_port = 8000  # Local port for forwarding

    logger.info("=" * 60)
    logger.info("vLLM Deployment Test (Entrypoint Mode)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_id}")
    logger.info(f"Image: {VLLM_IMAGE}")
    logger.info(f"Max price: ${max_price}/hr")
    logger.info("")

    try:
        async with ShopperClient(shopper_url) as client:
            # Step 1: Check shopper health
            logger.info("Step 1: Checking shopper health...")
            try:
                await client.wait_for_ready(timeout=30)
                logger.info("  Shopper ready")
            except Exception as e:
                logger.error(f"  Shopper not ready: {e}")
                logger.error("  Make sure cloud-gpu-shopper is running")
                return 1

            # Step 2: Find GPU offers
            logger.info("Step 2: Finding GPU offers...")
            offers = await client.get_inventory(
                min_vram=8,  # TinyLlama needs ~4GB, use 8 for safety
                max_price=max_price,
                provider="vastai",  # Vast.ai only
                min_gpu_count=1,
            )
            logger.info(f"  Found {len(offers)} offers")

            if not offers:
                logger.error("  No suitable GPU offers found")
                return 1

            # Filter for more reliable GPU types and pick cheapest
            reliable_gpus = ["RTX 3090", "RTX 4090", "RTX 4080", "A100", "A6000"]
            preferred = [o for o in offers if any(g in o.gpu_type for g in reliable_gpus)]
            if preferred:
                offer = min(preferred, key=lambda o: o.price_per_hour)
            else:
                offer = min(offers, key=lambda o: o.price_per_hour)
            logger.info(f"  Selected: {offer.gpu_type} @ ${offer.price_per_hour:.3f}/hr")

            # Step 3: Create session with SSH mode (faster provisioning)
            logger.info("Step 3: Creating session (SSH mode)...")
            response = await client.create_session(
                offer_id=offer.id,
                reservation_hours=1,
                launch_mode="ssh",  # SSH mode - we'll install vLLM via pip
            )
            session = response.session
            ssh_key = response.ssh_private_key
            logger.info(f"  Session created: {session.id}")

            try:
                # Step 4: Wait for session to be running
                logger.info("Step 4: Waiting for session to start...")
                session = await client.wait_for_running(
                    session.id,
                    provider=session.provider,
                )
                logger.info(f"  Session running at {session.ssh_host}:{session.ssh_port}")

                # Step 5: Connect via SSH
                logger.info("Step 5: Connecting via SSH...")
                async with SSHConnection(
                    host=session.ssh_host,
                    port=session.ssh_port,
                    user=session.ssh_user,
                    private_key=ssh_key,
                ) as ssh:
                    logger.info("  SSH connected")

                    # Step 6: Install and start vLLM via pip
                    logger.info("Step 6: Setting up environment...")

                    # Check what's available
                    result = await ssh.run("python3 --version && which python3")
                    logger.info(f"  {result.stdout.strip()}")

                    # Check for pip
                    result = await ssh.run("python3 -m pip --version")
                    if result.exit_code != 0:
                        logger.info("  pip not found, installing...")
                        result = await ssh.run("apt-get update && apt-get install -y python3-pip", timeout=120)
                        if result.exit_code != 0:
                            # Try get-pip.py
                            result = await ssh.run("curl -sS https://bootstrap.pypa.io/get-pip.py | python3", timeout=120)
                            if result.exit_code != 0:
                                logger.error(f"  Failed to install pip: {result.stderr}")
                                return 1
                        logger.info("  pip installed")

                    # Install vLLM (this can take a few minutes)
                    logger.info("  Installing vLLM via pip (this may take a few minutes)...")
                    result = await ssh.run("python3 -m pip install vllm", timeout=600)
                    if result.exit_code != 0:
                        logger.error(f"  Failed to install vLLM: {result.stderr}")
                        return 1
                    logger.info("  vLLM installed")

                    # Check CUDA and GPU
                    logger.info("  Checking GPU...")
                    result = await ssh.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
                    logger.info(f"  GPU: {result.stdout.strip()}")

                    # Start vLLM server in background (use v0 engine for compatibility)
                    logger.info(f"  Starting vLLM server for {model_id}...")
                    vllm_cmd = (
                        f"VLLM_USE_V1=0 nohup python3 -m vllm.entrypoints.openai.api_server "
                        f"--model {model_id} "
                        f"--host 0.0.0.0 "
                        f"--port {VLLM_PORT} "
                        f"--trust-remote-code "
                        f"--dtype auto "
                        f"> /tmp/vllm.log 2>&1 &"
                    )
                    result = await ssh.run(vllm_cmd)
                    await asyncio.sleep(10)  # Give it more time to start

                    # Check if it's running
                    result = await ssh.run("pgrep -f 'vllm.entrypoints' || echo 'not running'")
                    if "not running" in result.stdout:
                        # Check logs for errors
                        result = await ssh.run("cat /tmp/vllm.log")
                        logger.error(f"  vLLM failed to start. Logs:\n{result.stdout[:2000]}")
                        return 1
                    logger.info("  vLLM server started")

                    # Step 7: Set up port forwarding
                    logger.info(f"Step 7: Setting up port forwarding: localhost:{local_port} -> remote:{VLLM_PORT}")

                    async with ssh.forward_local_port(local_port, VLLM_PORT) as actual_port:
                        logger.info(f"  Port forwarding active on localhost:{actual_port}")

                        logger.info("")
                        logger.info("vLLM API endpoints (OpenAI-compatible):")
                        logger.info(f"  http://localhost:{actual_port}/v1/models")
                        logger.info(f"  http://localhost:{actual_port}/v1/completions")
                        logger.info(f"  http://localhost:{actual_port}/v1/chat/completions")
                        logger.info("")
                        logger.info("Example curl:")
                        logger.info(f'  curl http://localhost:{actual_port}/v1/chat/completions \\')
                        logger.info('    -H "Content-Type: application/json" \\')
                        logger.info('    -d \'{"model": "' + model_id + '", "messages": [{"role": "user", "content": "Hello!"}]}\'')
                        logger.info("")

                        # Step 8: Wait for model to load and test
                        logger.info("Step 8: Waiting for model to load...")
                        async with httpx.AsyncClient(timeout=300) as http:
                            # Poll /v1/models until model is ready
                            for attempt in range(60):  # 10 minutes max (model download can take time)
                                try:
                                    resp = await http.get(f"http://localhost:{actual_port}/v1/models")
                                    if resp.status_code == 200:
                                        data = resp.json()
                                        if data.get("data"):
                                            model_name = data["data"][0].get("id", "unknown")
                                            logger.info(f"  Model loaded: {model_name}")
                                            break
                                except Exception as e:
                                    logger.info(f"  Waiting... ({attempt + 1}/60) - {e}")
                                    # Every 5 attempts, check vLLM logs
                                    if attempt > 0 and attempt % 5 == 0:
                                        result = await ssh.run("tail -10 /tmp/vllm.log 2>/dev/null || echo 'no logs'")
                                        if result.stdout.strip() and result.stdout.strip() != "no logs":
                                            logger.info(f"  vLLM log: {result.stdout.strip()[:200]}")
                                await asyncio.sleep(10)
                            else:
                                logger.error("  Model did not load in time")
                                return 1

                            # Test generation
                            logger.info("Step 9: Testing chat completion...")
                            resp = await http.post(
                                f"http://localhost:{actual_port}/v1/chat/completions",
                                json={
                                    "model": model_id,
                                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                                    "max_tokens": 50
                                },
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                                logger.info(f"  Response: {content[:200]}")
                                logger.info("")
                                logger.info("SUCCESS! vLLM is working.")
                            else:
                                logger.error(f"  Generation failed: {resp.status_code} - {resp.text}")
                                return 1

                        # Keep connection open for manual testing
                        logger.info("")
                        logger.info("Press Ctrl+C to cleanup and exit...")
                        try:
                            while True:
                                await asyncio.sleep(60)
                        except KeyboardInterrupt:
                            logger.info("Interrupted, cleaning up...")

            finally:
                # Cleanup
                logger.info("Cleaning up session...")
                try:
                    await client.signal_done(session.id)
                    logger.info("  Session cleanup initiated")
                except Exception as e:
                    logger.warning(f"  Cleanup error: {e}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
