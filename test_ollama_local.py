#!/usr/bin/env python3
"""Test Ollama deployment with local port forwarding.

This script:
1. Gets a GPU instance from Vast.ai via cloud-gpu-shopper
2. Deploys Ollama via SSH
3. Sets up SSH port forwarding to access locally
4. Tests the LLM from your local machine
5. Cleans up

Usage:
    python test_ollama_local.py

The LLM will be accessible at http://localhost:11434 while the script runs.
"""

import asyncio
import logging
import sys

import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    from gpu_deploy_llm.client.shopper import ShopperClient
    from gpu_deploy_llm.ssh.connection import SSHConnection
    from gpu_deploy_llm.deploy.ollama import OllamaDeployer, OLLAMA_PORT

    shopper_url = "http://localhost:8080"
    model = "tinyllama"  # Small model, ~1GB
    local_port = 11434

    session_id = None
    ssh_key = None

    async with ShopperClient(shopper_url, debug=True) as client:
        try:
            # Step 1: Check shopper is ready
            logger.info("Checking shopper readiness...")
            await client.wait_for_ready(timeout=30)
            logger.info("Shopper is ready")

            # Step 2: Get inventory
            logger.info("Getting GPU offers...")
            offers = await client.get_inventory(
                min_vram=8,
                provider="vastai",
                max_price=0.50,
            )

            if not offers:
                logger.error("No suitable GPU offers found")
                return

            # Sort by price
            offers.sort(key=lambda x: x.price_per_hour)
            offer = offers[0]
            logger.info(f"Selected: {offer.gpu_type} @ ${offer.price_per_hour}/hr")

            # Step 3: Create session
            logger.info("Creating session...")
            response = await client.create_session(
                offer_id=offer.id,
                reservation_hours=1,
                workload_type="llm_vllm",
            )

            session_id = response.session.id
            ssh_key = response.ssh_private_key
            logger.info(f"Session created: {session_id}")

            # Step 4: Wait for running
            logger.info("Waiting for session to be running...")
            session = await client.wait_for_running(
                session_id,
                provider=response.session.provider,
            )
            logger.info(f"Session running at {session.ssh_host}:{session.ssh_port}")

            # Step 5: Connect via SSH
            logger.info("Connecting via SSH...")
            async with SSHConnection(
                host=session.ssh_host,
                port=session.ssh_port,
                user=session.ssh_user,
                private_key=ssh_key,
            ) as ssh:
                # Check GPU
                gpu = await ssh.get_gpu_status()
                if gpu:
                    logger.info(f"GPU: {gpu[0].name} ({gpu[0].memory_total_mb}MB)")

                # Step 6: Deploy Ollama
                logger.info(f"Deploying Ollama with model: {model}")
                deployer = OllamaDeployer(ssh)
                result = await deployer.deploy(model)

                if not result.success:
                    logger.error(f"Deployment failed: {result.error}")
                    return

                logger.info(f"Ollama deployed successfully")
                logger.info(f"  Install time: {result.install_seconds:.1f}s")
                logger.info(f"  Model pull time: {result.model_pull_seconds:.1f}s")

                # Step 7: Set up port forwarding
                logger.info(f"Setting up port forwarding: localhost:{local_port} -> remote:{OLLAMA_PORT}")

                async with ssh.forward_local_port(local_port, OLLAMA_PORT) as actual_port:
                    logger.info(f"Port forwarding active on localhost:{actual_port}")
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info("LLM is now accessible at:")
                    logger.info(f"  http://localhost:{actual_port}/v1/chat/completions")
                    logger.info("")
                    logger.info("Try it with curl:")
                    logger.info(f'  curl http://localhost:{actual_port}/v1/chat/completions \\')
                    logger.info(f'    -d \'{{"model": "{model}", "messages": [{{"role": "user", "content": "Hello!"}}]}}\'')
                    logger.info("=" * 60)
                    logger.info("")

                    # Step 8: Test the API
                    logger.info("Testing API...")
                    async with httpx.AsyncClient() as http:
                        try:
                            # Test models endpoint
                            resp = await http.get(
                                f"http://localhost:{actual_port}/api/tags",
                                timeout=10.0,
                            )
                            if resp.status_code == 200:
                                logger.info("Models endpoint: OK")
                                models = resp.json().get("models", [])
                                for m in models:
                                    logger.info(f"  - {m.get('name')}")

                            # Test chat
                            logger.info("Testing chat completion...")
                            resp = await http.post(
                                f"http://localhost:{actual_port}/api/chat",
                                json={
                                    "model": model,
                                    "messages": [{"role": "user", "content": "Say hello in one word."}],
                                    "stream": False,
                                },
                                timeout=60.0,
                            )

                            if resp.status_code == 200:
                                data = resp.json()
                                content = data.get("message", {}).get("content", "")
                                logger.info(f"Chat response: {content[:100]}")
                            else:
                                logger.warning(f"Chat returned: {resp.status_code}")

                        except Exception as e:
                            logger.error(f"API test failed: {e}")

                    # Keep tunnel open for manual testing
                    logger.info("")
                    logger.info("Press Ctrl+C to stop and cleanup...")
                    try:
                        while True:
                            await asyncio.sleep(10)
                    except KeyboardInterrupt:
                        logger.info("Stopping...")

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            if session_id:
                logger.info(f"Cleaning up session {session_id}...")
                try:
                    await client.signal_done(session_id)
                    logger.info("Session cleanup initiated")
                except Exception as e:
                    logger.warning(f"Cleanup failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
