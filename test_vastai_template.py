#!/usr/bin/env python3
"""Test Vast.ai vLLM template deployment.

Uses Vast.ai's vLLM template which:
- Starts vLLM automatically via VLLM_MODEL env var
- Exposes port 8000 for API access
- Uses SSH tunnel to access locally

Usage:
    # First, rebuild and restart the shopper:
    cd ../cloud-gpu-shopper && go build ./cmd/server && ./server

    # Then run this test:
    python test_vastai_template.py
"""

import asyncio
import logging
import sys

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    from gpu_deploy_llm.client.shopper import ShopperClient
    from gpu_deploy_llm.ssh.connection import SSHConnection

    shopper_url = "http://localhost:8080"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for testing
    local_port = 8000  # vLLM default port

    session_id = None
    ssh_key = None

    async with ShopperClient(shopper_url, debug=True) as client:
        try:
            # Step 1: Check shopper is ready
            logger.info("Checking shopper readiness...")
            await client.wait_for_ready(timeout=30)
            logger.info("Shopper is ready")

            # Step 2: Get Vast.ai inventory
            logger.info("Getting Vast.ai GPU offers...")
            offers = await client.get_inventory(
                min_vram=8,
                provider="vastai",
                max_price=0.50,
            )

            if not offers:
                logger.error("No suitable Vast.ai GPU offers found")
                return

            # Sort by price and pick cheapest
            offers.sort(key=lambda x: x.price_per_hour)
            offer = offers[0]
            logger.info(f"Selected: {offer.gpu_type} @ ${offer.price_per_hour}/hr (ID: {offer.id})")

            # Step 3: Create session with entrypoint mode (vLLM template)
            logger.info(f"Creating session with vLLM template for model: {model_id}")
            response = await client.create_session(
                offer_id=offer.id,
                reservation_hours=1,
                workload_type="llm_vllm",
                # Entrypoint mode parameters
                launch_mode="entrypoint",
                docker_image="vllm/vllm-openai:latest",
                model_id=model_id,
                exposed_ports=[8000],
            )

            session_id = response.session.id
            ssh_key = response.ssh_private_key
            logger.info(f"Session created: {session_id}")
            logger.info(f"SSH key received: {len(ssh_key)} bytes")

            # Step 4: Wait for session to be running
            logger.info("Waiting for session to be running...")
            session = await client.wait_for_running(
                session_id,
                provider=response.session.provider,
            )

            logger.info(f"Session running!")
            logger.info(f"  SSH: {session.ssh_host}:{session.ssh_port}")
            if session.api_endpoint:
                logger.info(f"  API: {session.api_endpoint}")

            # Step 5: Connect via SSH and check status
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

                # Check environment variables
                env_result = await ssh.run("env | grep VLLM")
                logger.info(f"vLLM env vars:\n{env_result.stdout}")

                # Check if vLLM is running
                ps_result = await ssh.run("ps aux | grep -i vllm | grep -v grep")
                if ps_result.stdout.strip():
                    logger.info("vLLM process is running!")
                    logger.info(ps_result.stdout[:200])
                else:
                    logger.warning("vLLM process not found yet")

                # Check logs
                log_result = await ssh.run("cat /var/log/vllm.log 2>/dev/null || cat /tmp/vllm.log 2>/dev/null || echo 'No logs found'")
                logger.info(f"vLLM logs:\n{log_result.stdout[:500]}")

                # Step 6: Set up port forwarding to access API locally
                logger.info(f"Setting up port forwarding: localhost:{local_port} -> remote:8000")

                async with ssh.forward_local_port(local_port, 8000) as actual_port:
                    logger.info(f"Port forwarding active on localhost:{actual_port}")
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info("vLLM API accessible at:")
                    logger.info(f"  http://localhost:{actual_port}/v1/models")
                    logger.info(f"  http://localhost:{actual_port}/v1/completions")
                    logger.info("")
                    logger.info("Test with:")
                    logger.info(f"  curl http://localhost:{actual_port}/v1/models")
                    logger.info("=" * 60)
                    logger.info("")

                    # Wait for vLLM to be ready (may take a while to load model)
                    logger.info("Waiting for vLLM API to be ready...")
                    async with httpx.AsyncClient() as http:
                        for attempt in range(30):  # 5 minutes max
                            try:
                                resp = await http.get(
                                    f"http://localhost:{actual_port}/v1/models",
                                    timeout=10.0,
                                )
                                if resp.status_code == 200:
                                    logger.info("vLLM API is ready!")
                                    data = resp.json()
                                    logger.info(f"Models: {data}")
                                    break
                                elif resp.status_code == 401:
                                    logger.info("vLLM API ready (requires auth)")
                                    break
                            except (httpx.ConnectError, httpx.TimeoutException):
                                pass

                            logger.info(f"Waiting for vLLM... (attempt {attempt + 1}/30)")
                            await asyncio.sleep(10)
                        else:
                            logger.warning("vLLM API not ready after 5 minutes")

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
