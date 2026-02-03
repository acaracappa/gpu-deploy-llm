# GPU Deploy LLM - Bug Fix Plan

## Summary of Issues Found

During testing, we identified several bugs preventing successful vLLM deployment:

| Bug | Severity | Status |
|-----|----------|--------|
| vLLM dependency installation (PyTorch, numpy) | Critical | Partial fix applied |
| Port 8000 not listening after vLLM starts | Critical | Under investigation |
| SSH verification timeouts from shopper | High | Not fixed |
| Stale inventory handling | Medium | Partial retry logic exists |
| Public endpoint exposure unclear | Medium | Architecture clarification needed |

---

## Bug 1: vLLM Dependency Installation Failures

### Symptoms
```
[TEST][INFO] [0s] Health: waiting, Model: Models endpoint failed: no response - PID 466, port 8000 NOT listening, GPU mem: 2MB/12288MB (0%), log: PyTorch was not found
[TEST][INFO] [12s] Health: waiting, Model: Models endpoint failed: no response - PID 526, port 8000 NOT listening, GPU mem: 2MB/12288MB (0%), log: ModuleNotFoundError: No module named 'numpy.lib.function_base'
```

### Root Cause
1. `pip install vllm` doesn't install PyTorch with CUDA support by default
2. numpy 2.x removed `numpy.lib.function_base` which vLLM/transformers depend on
3. vLLM process starts but crashes immediately, spawning new PIDs repeatedly

### Current Fix (Partial)
Added to `vllm.py:_install_vllm_pip()`:
```python
# Step 1: Install PyTorch with CUDA support
await self.ssh.run(f"{python_cmd} -m pip install torch --index-url https://download.pytorch.org/whl/cu121")

# Step 2: Install compatible numpy
await self.ssh.run(f"{python_cmd} -m pip install 'numpy<2.0'")

# Step 3: Install vLLM
await self.ssh.run(f"{python_cmd} -m pip install vllm=={VLLM_PIP_VERSION}")
```

### Remaining Work
- [ ] Add verification after each install step (check import works)
- [ ] Add CUDA version detection to select correct PyTorch wheel (cu118 vs cu121)
- [ ] Add fallback to pre-built vLLM wheel if pip install fails
- [ ] Consider using a requirements.txt with pinned versions

### Fix Implementation

**File:** `gpu_deploy_llm/deploy/vllm.py`

```python
async def _install_vllm_pip(self, python_cmd: str) -> None:
    """Install vLLM via pip with proper dependency ordering."""

    # Detect CUDA version
    cuda_result = await self.ssh.run("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1")
    driver_version = cuda_result.stdout.strip() if cuda_result.success else ""

    # Map driver to CUDA version (simplified)
    # Driver 525+ -> CUDA 12.x, Driver 470-525 -> CUDA 11.8
    cuda_index = "cu121"  # Default to CUDA 12.1
    if driver_version:
        major = int(driver_version.split('.')[0])
        if major < 525:
            cuda_index = "cu118"

    logger.info(f"Detected driver {driver_version}, using PyTorch {cuda_index}")

    # Step 1: Install PyTorch with CUDA support
    logger.info("Installing PyTorch with CUDA support...")
    torch_result = await self.ssh.run(
        f"{python_cmd} -m pip install torch --index-url https://download.pytorch.org/whl/{cuda_index}",
        timeout=600,
    )
    if not torch_result.success:
        raise DeploymentError(f"PyTorch installation failed: {torch_result.stderr}")

    # Verify PyTorch CUDA works
    verify_result = await self.ssh.run(
        f'{python_cmd} -c "import torch; assert torch.cuda.is_available(), \'CUDA not available\'"'
    )
    if not verify_result.success:
        raise DeploymentError(f"PyTorch CUDA verification failed: {verify_result.stderr}")
    logger.info("PyTorch CUDA verified")

    # Step 2: Install compatible numpy (before vLLM)
    logger.info("Installing compatible numpy...")
    numpy_result = await self.ssh.run(
        f"{python_cmd} -m pip install 'numpy>=1.26,<2.0'",
        timeout=120,
    )
    if not numpy_result.success:
        logger.warning(f"numpy install warning: {numpy_result.stderr}")

    # Step 3: Install vLLM
    logger.info(f"Installing vLLM {VLLM_PIP_VERSION}...")
    vllm_result = await self.ssh.run(
        f"{python_cmd} -m pip install vllm=={VLLM_PIP_VERSION}",
        timeout=600,
    )
    if not vllm_result.success:
        raise DeploymentError(f"vLLM installation failed: {vllm_result.stderr}")

    # Verify vLLM imports
    verify_vllm = await self.ssh.run(
        f'{python_cmd} -c "from vllm import LLM; print(\'vLLM OK\')"'
    )
    if not verify_vllm.success:
        raise DeploymentError(f"vLLM verification failed: {verify_vllm.stderr}")

    logger.info("vLLM installation verified")
```

---

## Bug 2: Port 8000 Not Listening After vLLM Starts

### Symptoms
- vLLM process has a PID but port 8000 is NOT listening
- GPU memory stays at 0% (model not loaded)
- Process keeps restarting with new PIDs

### Root Cause
vLLM process crashes immediately after starting due to:
1. Missing dependencies (see Bug 1)
2. Insufficient GPU memory for model
3. Model download failures (network issues)

### Fix Implementation

**File:** `gpu_deploy_llm/deploy/vllm.py`

Add startup verification with better error capture:

```python
async def _start_vllm_process(self, python_cmd: str) -> None:
    """Start vLLM process with better error handling."""

    # Start vLLM in background
    start_cmd = f"""
nohup {python_cmd} -m vllm.entrypoints.openai.api_server \\
    --model {self.model_id} \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --api-key {self.api_key} \\
    {self._get_quantization_args()} \\
    > /tmp/vllm.log 2>&1 &
echo $!
"""
    result = await self.ssh.run(start_cmd)
    if not result.success:
        raise DeploymentError(f"Failed to start vLLM: {result.stderr}")

    pid = result.stdout.strip()
    logger.info(f"vLLM started with PID {pid}")

    # Wait briefly then check if process is still alive
    await asyncio.sleep(5)

    alive_check = await self.ssh.run(f"kill -0 {pid} 2>/dev/null && echo 'alive' || echo 'dead'")
    if "dead" in alive_check.stdout:
        # Process died - get logs
        log_result = await self.ssh.run("tail -50 /tmp/vllm.log")
        raise DeploymentError(
            f"vLLM process {pid} died immediately. Logs:\n{log_result.stdout}"
        )

    # Check if port is binding (may take a few seconds)
    for attempt in range(6):  # 30 seconds total
        await asyncio.sleep(5)
        port_check = await self.ssh.run("ss -tlnp | grep :8000")
        if ":8000" in port_check.stdout:
            logger.info("Port 8000 is now listening")
            return

        # Check process still alive
        alive_check = await self.ssh.run(f"kill -0 {pid} 2>/dev/null && echo 'alive' || echo 'dead'")
        if "dead" in alive_check.stdout:
            log_result = await self.ssh.run("tail -50 /tmp/vllm.log")
            raise DeploymentError(
                f"vLLM process died during startup. Logs:\n{log_result.stdout}"
            )

    # Process alive but port not listening - log what's happening
    log_result = await self.ssh.run("tail -20 /tmp/vllm.log")
    logger.warning(f"Port not listening after 30s, but process alive. Logs:\n{log_result.stdout}")
```

---

## Bug 3: SSH Verification Timeouts from Shopper

### Symptoms
```
Session status: failed
Error: SSH verification timeout
```
This happens before we even get to connect - the shopper service fails to verify SSH.

### Root Cause
1. Vast.ai/TensorDock instance takes longer than 8 minutes to become SSH-ready
2. Network/firewall issues on provider side
3. Instance provisioning stuck

### Fix Implementation

This is primarily a shopper-side issue, but we can improve our handling:

**File:** `gpu_deploy_llm/client/shopper.py`

```python
async def create_session_with_retry(
    self,
    offer_id: str,
    consumer_id: str,
    workload_type: str,
    reservation_hours: int = 1,
    max_retries: int = 3,
    excluded_providers: Optional[set] = None,
) -> CreateSessionResponse:
    """Create session with automatic retry on different offers."""

    excluded_offers = set()
    excluded_providers = excluded_providers or set()
    last_error = None

    for attempt in range(max_retries):
        try:
            # Get fresh inventory
            inventory = await self.get_inventory()

            # Filter offers
            suitable = [
                o for o in inventory.offers
                if o.id not in excluded_offers
                and o.provider not in excluded_providers
            ]

            if not suitable:
                raise NoAvailableOffersError("No suitable offers after exclusions")

            # Select cheapest
            offer = min(suitable, key=lambda o: o.price_per_hour)

            logger.info(f"Attempt {attempt + 1}: Trying {offer.gpu_type} @ ${offer.price_per_hour}/hr ({offer.provider})")

            response = await self.create_session(
                offer_id=offer.id,
                consumer_id=consumer_id,
                workload_type=workload_type,
                reservation_hours=reservation_hours,
            )

            # Wait for session to become running
            session = await self.wait_for_session_running(
                response.session.id,
                timeout=600,  # 10 minutes
            )

            if session.status == "failed":
                logger.warning(f"Session failed: {session.error}")
                excluded_offers.add(offer.id)

                # If SSH timeout on this provider, try different provider
                if "SSH" in (session.error or ""):
                    logger.info(f"Excluding provider {offer.provider} due to SSH issues")
                    excluded_providers.add(offer.provider)

                last_error = SessionFailedError(session.error)
                continue

            return response

        except StaleInventoryError as e:
            excluded_offers.add(e.offer_id)
            last_error = e
            continue

    raise last_error or ProvisioningError("Max retries exceeded")
```

---

## Bug 4: Stale Inventory Handling

### Symptoms
```
Error: stale_inventory - The selected offer is no longer available
```

### Root Cause
Time gap between inventory query and session creation allows offers to be claimed.

### Current State
Partial retry logic exists but needs improvement.

### Fix Implementation

**File:** `gpu_deploy_llm/web/server.py`

Improve the test runner to handle stale inventory:

```python
async def _run_test(self) -> None:
    """Run deployment test with better error handling."""

    max_provision_attempts = 3
    excluded_offers = set()

    for attempt in range(max_provision_attempts):
        try:
            # Query inventory
            await self._emit_status("query_inventory", "running")
            inventory = await self.shopper.get_inventory(
                min_vram=requirements.vram_gb,
                min_gpu_count=requirements.gpu_count,
            )

            # Filter out previously failed offers
            available = [o for o in inventory.offers if o.id not in excluded_offers]

            if not available:
                raise NoAvailableOffersError(
                    f"No offers available (excluded {len(excluded_offers)} failed offers)"
                )

            # Select offer
            offer = self._select_offer(available)
            await self._log("info", f"Selected: {offer.gpu_type} @ ${offer.price_per_hour}/hr ({offer.provider})")

            # Create session
            await self._emit_status("create_session", "running")
            response = await self.shopper.create_session(...)

            # Success - continue with deployment
            break

        except StaleInventoryError as e:
            await self._log("warning", f"Offer {e.offer_id} no longer available, retrying...")
            excluded_offers.add(e.offer_id)
            if attempt == max_provision_attempts - 1:
                raise
            continue
```

---

## Bug 5: Public Endpoint Exposure

### Current Understanding

**Vast.ai Architecture:**
- SSH connection goes through proxy: `ssh5.vast.ai:12345`
- The proxy only forwards SSH (port 22), NOT arbitrary ports
- Port 8000 on the instance is NOT accessible via `ssh5.vast.ai:8000`

### Solution: SSH Port Forwarding (IMPLEMENTED)

**Since we already have SSH working, we can use SSH local port forwarding to tunnel the vLLM endpoint back to the test suite.** This is the cleanest approach:

1. No additional software needed on remote instance
2. Works through Vast.ai/TensorDock SSH proxies
3. Secure - traffic is encrypted through SSH tunnel

### Implementation (DONE)

Added `forward_local_port()` method to `SSHConnection` class:

**File:** `gpu_deploy_llm/ssh/connection.py`

```python
async with SSHConnection(host, port, user, private_key) as ssh:
    # Forward local port 8000 to remote port 8000
    async with ssh.forward_local_port(8000, 8000) as local_port:
        # Now can access vLLM at http://localhost:{local_port}/v1/models
        # using httpx or any HTTP client
        response = httpx.get(
            f"http://localhost:{local_port}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
```

### Remaining Work: Update Test Suite to Use Port Forwarding

**File:** `gpu_deploy_llm/web/server.py`

Change health checks and benchmarks to use port forwarding instead of SSH curl:

```python
async with SSHConnection(...) as ssh:
    # Deploy vLLM...

    # Set up port forward for health checks and benchmarks
    async with ssh.forward_local_port(0, 8000) as local_port:
        endpoint = f"http://localhost:{local_port}"

        # Health checks can now use httpx directly
        health_checker = HealthChecker(
            endpoint=endpoint,
            api_key=api_key,
        )

        # Wait for model to load
        await health_checker.wait_for_ready()

        # Run benchmarks with direct HTTP access
        benchmarker = Benchmarker(
            endpoint=endpoint,
            api_key=api_key,
        )
        results = await benchmarker.run_suite()
```

### Benefits of This Approach

| Aspect | SSH Curl (Current) | SSH Port Forward (New) |
|--------|-------------------|------------------------|
| Health checks | Works | Works |
| Benchmarks | Requires shell escaping | Clean httpx calls |
| Latency | Extra SSH command per request | Single SSH tunnel |
| Error handling | Parse curl output | Standard HTTP errors |
| Streaming | Not possible | Works |

### Alternative: cloudflared Tunnel (For External Access)

If users need to access the endpoint from machines without SSH access, cloudflared can create a public URL:

```python
async def _setup_public_endpoint(self) -> Optional[str]:
    """Set up public endpoint using cloudflared tunnel (optional)."""

    # Install cloudflared
    install_cmd = """
curl -L --output /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i /tmp/cloudflared.deb 2>/dev/null || apt-get install -f -y
"""
    await self.ssh.run(install_cmd, timeout=120)

    # Start tunnel
    tunnel_cmd = """
nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflared.log 2>&1 &
sleep 5
grep -o 'https://[^[:space:]]*\.trycloudflare\.com' /tmp/cloudflared.log | head -1
"""
    result = await self.ssh.run(tunnel_cmd, timeout=30)

    if result.success and "trycloudflare.com" in result.stdout:
        return result.stdout.strip()
    return None
```

---

## Implementation Priority

| Priority | Bug | Effort | Impact | Status |
|----------|-----|--------|--------|--------|
| P0 | Bug 1: vLLM dependencies | Medium | Critical - nothing works without this | **DONE** - CUDA detection, PyTorch/numpy install, verification |
| P0 | Bug 2: Port not listening | Medium | Critical - need to verify deployment | **DONE** - Startup verification, crash detection |
| P1 | Bug 3: SSH timeout handling | Low | High - improves reliability | Not started |
| P1 | Bug 4: Stale inventory | Low | Medium - reduces failed attempts | Partial |
| P1 | Bug 5: SSH port forwarding | Low | High - enables proper testing | **DONE** - `forward_local_port()` in SSHConnection |
| P2 | Bug 6: Refactor health/benchmark to use port forward | Medium | High - cleaner code | **DONE** - Added `HealthCheckerHTTP` class |

---

## Bug 6: Refactor Health/Benchmark to Use Port Forwarding

### Current State

Health checks and benchmarks currently work by running curl via SSH:

```python
# health.py - current approach
result = await self.ssh.run(
    f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health",
    timeout=int(self.timeout),
)
```

This works but has downsides:
- Shell escaping is error-prone
- Can't stream responses
- Error handling is more complex
- Extra latency per request

### Proposed Refactor

Use the new `SSHConnection.forward_local_port()` to create a tunnel, then use standard httpx:

**File:** `gpu_deploy_llm/deploy/health.py`

```python
class HealthChecker:
    """Health checker for vLLM deployments."""

    def __init__(
        self,
        endpoint: str,  # Now a local URL like http://localhost:8000
        api_key: str,
        timeout: float = 30.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def check_health_endpoint(self) -> HealthCheckResult:
        """Check vLLM /health endpoint."""
        try:
            response = await self._client.get(f"{self.endpoint}/health")
            if response.status_code == 200:
                return HealthCheckResult(
                    name="health_endpoint",
                    status=CheckStatus.PASSED,
                    message="Health endpoint returned 200",
                )
            # ... error handling
        except httpx.RequestError as e:
            return HealthCheckResult(
                name="health_endpoint",
                status=CheckStatus.FAILED,
                message=f"Request failed: {e}",
            )
```

**File:** `gpu_deploy_llm/web/server.py`

```python
async def _run_test(self) -> None:
    """Run deployment test."""
    # ... provision session, connect SSH, deploy vLLM ...

    async with SSHConnection(...) as ssh:
        # Deploy vLLM
        deployer = VLLMDeployer(ssh=ssh, model_id=model_id, ...)
        await deployer.deploy()

        # Set up port forward for health checks
        async with ssh.forward_local_port(0, 8000) as local_port:
            endpoint = f"http://localhost:{local_port}"
            await self._log("info", f"Tunnel established: {endpoint}")

            # Health checker uses direct HTTP
            health_checker = HealthChecker(
                endpoint=endpoint,
                api_key=api_key,
            )

            await self._emit_status("wait_for_model", "running")
            await health_checker.wait_for_ready(
                timeout=300,
                progress_callback=self._health_progress,
            )

            # Benchmark uses direct HTTP
            if self.config.run_benchmark:
                benchmarker = Benchmarker(
                    endpoint=endpoint,
                    api_key=api_key,
                )
                results = await benchmarker.run_suite()
```

### Migration Path

1. Keep existing SSH-curl approach as fallback
2. Add port-forward approach as primary
3. Remove SSH-curl code once port-forward is proven

---

## Testing Plan

After fixes are implemented:

1. **Unit test dependency installation**
   - Mock SSH and verify correct commands sent
   - Test CUDA version detection

2. **Integration test with TinyLlama**
   - Deploy to Vast.ai
   - Verify health endpoint responds
   - Run inference test

3. **Retry logic test**
   - Simulate stale inventory
   - Verify automatic retry with different offer

4. **Provider comparison**
   - Test on both Vast.ai and TensorDock
   - Document provider-specific quirks

---

## Quick Start After Fixes

```bash
# Start dashboard
cd /Users/avc/Documents/gpu-deploy-llm
python -m gpu_deploy_llm.web.server --port 8081

# Or run CLI test
python -m gpu_deploy_llm deploy TinyLlama/TinyLlama-1.1B-Chat-v1.0 --provider vastai
```
