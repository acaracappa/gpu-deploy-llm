# GPU Deploy LLM - Ops Bug Report

**Generated:** 2026-01-31
**Analyzed File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py

---

## Process Tracing Summary

**Running Process Found:**
- PID: 14391
- Command: `/opt/homebrew/Cellar/python@3.13/3.13.3_1/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python -m gpu_deploy_llm.cli --shopper-url http://localhost:8080 web --port 8081`
- CPU: 0.0%
- Memory: 0.4% (~102MB RSS)
- Status: SN (sleeping, low priority)
- Started: 3:03 PM
- Runtime: ~5 minutes

**Note:** Unable to check for zombie processes or read service logs due to permission restrictions during analysis.

---

## Bug List

### BUG-001: Race Condition in Global State Mutation
**Severity:** Critical
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 715-720, 733-737

**Description:**
The global `_current_test` variable is modified without any synchronization primitives (locks, semaphores). Multiple concurrent requests to `/api/test/run` or `/api/test/stop` can cause race conditions where:
- A test could be partially overwritten
- Test state could become inconsistent
- Memory corruption could occur in edge cases

**Code:**
```python
@app.post("/api/test/run")
async def run_test(config: TestConfig):
    global _current_test
    if _current_test and not _current_test.passed and not _current_test.error:
        raise HTTPException(400, "Test already running")
    _current_test = TestRunner(config, _shopper_url, _debug)  # Race condition here
```

**Suggested Fix:**
Use an `asyncio.Lock` to protect access to `_current_test`:
```python
_test_lock = asyncio.Lock()

@app.post("/api/test/run")
async def run_test(config: TestConfig):
    async with _test_lock:
        global _current_test
        if _current_test and not _current_test.passed and not _current_test.error:
            raise HTTPException(400, "Test already running")
        _current_test = TestRunner(config, _shopper_url, _debug)
```

---

### BUG-002: WebSocket Connection Set Modification During Iteration
**Severity:** High
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 479-488

**Description:**
The `broadcast_event` function iterates over `_active_connections` and then modifies it (via `difference_update`). While Python sets are thread-safe for single operations, this pattern can cause issues with concurrent WebSocket connections/disconnections during broadcast.

**Code:**
```python
async def broadcast_event(event: WebSocketEvent) -> None:
    disconnected = set()
    for ws in _active_connections:  # Iteration
        try:
            await ws.send_text(event.to_json())
        except Exception:
            disconnected.add(ws)
    _active_connections.difference_update(disconnected)  # Modification
```

**Suggested Fix:**
Create a copy of the set before iteration or use a lock:
```python
async def broadcast_event(event: WebSocketEvent) -> None:
    disconnected = set()
    connections_snapshot = set(_active_connections)  # Create snapshot
    for ws in connections_snapshot:
        try:
            await ws.send_text(event.to_json())
        except Exception:
            disconnected.add(ws)
    _active_connections.difference_update(disconnected)
```

---

### BUG-003: Unbounded WebSocket Connection Set (Potential Memory Leak)
**Severity:** High
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 43, 844, 865

**Description:**
The `_active_connections` set grows without bound. While connections are removed on normal disconnect (line 865), certain failure modes (client crashes, network issues) may leave stale connections in the set that never get removed.

**Code:**
```python
_active_connections: Set[WebSocket] = set()
# ...
await websocket.accept()
_active_connections.add(websocket)  # Always adds
# ...
finally:
    _active_connections.discard(websocket)  # May not always execute
```

**Suggested Fix:**
1. Implement a maximum connection limit
2. Add periodic cleanup of stale connections
3. Use weak references for connection tracking

```python
MAX_CONNECTIONS = 100
# Add connection limit check
if len(_active_connections) >= MAX_CONNECTIONS:
    await websocket.close(code=1013)  # Try again later
    return
```

---

### BUG-004: Fire-and-Forget Async Task Without Error Handling
**Severity:** High
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 723

**Description:**
The test is run as a fire-and-forget task using `asyncio.create_task()`. If the task raises an unhandled exception, it will be silently lost and may cause the test to appear stuck.

**Code:**
```python
asyncio.create_task(_current_test.run())  # No error handling
```

**Suggested Fix:**
Add exception logging callback:
```python
task = asyncio.create_task(_current_test.run())
task.add_done_callback(lambda t: logger.error(f"Test task failed: {t.exception()}") if t.exception() else None)
```

---

### BUG-005: Swallowed Exception in Cleanup Handler
**Severity:** Medium
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 424-429

**Description:**
In the error cleanup path, exceptions from `force_destroy` are completely silenced with a bare `except Exception: pass`. This makes debugging cleanup failures impossible.

**Code:**
```python
if self.session_id and self.config.auto_cleanup:
    try:
        async with ShopperClient(self.shopper_url) as client:
            await client.force_destroy(self.session_id)
    except Exception:
        pass  # Silent failure - BAD
```

**Suggested Fix:**
Log the exception even if not re-raising:
```python
except Exception as e:
    logger.warning(f"Cleanup failed for session {self.session_id}: {e}")
```

---

### BUG-006: Missing Cancelled Flag Check
**Severity:** Medium
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 108, 110-429

**Description:**
The `TestRunner` class has a `_cancelled` flag that is set in `stop_test()` (line 734), but the `run()` method never checks this flag. A stopped test will continue running until completion.

**Code:**
```python
self._cancelled = False  # Line 108 - defined
# ... run() method never checks _cancelled
```

**Suggested Fix:**
Add cancellation checks at appropriate points in the run() method:
```python
async def run(self) -> None:
    try:
        if self._cancelled:
            return
        # ... continue with steps, checking _cancelled periodically
```

---

### BUG-007: Silent Exception Swallowing in Status Endpoint
**Severity:** Medium
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 532-544

**Description:**
Multiple try/except blocks with bare `pass` statements make debugging connectivity issues very difficult. The status endpoint silently fails for health, ready, and inventory checks.

**Code:**
```python
try:
    health = await client.health_check()
    status["shopper"]["healthy"] = True
except Exception:
    pass  # Silent failure
```

**Suggested Fix:**
Log exceptions at debug level:
```python
except Exception as e:
    logger.debug(f"Health check failed: {e}")
```

---

### BUG-008: Deprecation Warning - datetime.utcnow()
**Severity:** Low
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 64

**Description:**
`datetime.utcnow()` is deprecated in Python 3.12+. The code should use timezone-aware datetimes.

**Code:**
```python
timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
```

**Suggested Fix:**
```python
from datetime import datetime, timezone
timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

---

### BUG-009: Missing Validation on Test Configuration
**Severity:** Low
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 70-80

**Description:**
The `TestConfig` model lacks validation constraints. Invalid values (negative prices, zero reservation hours) are accepted without validation.

**Code:**
```python
class TestConfig(BaseModel):
    max_price: float = 0.50  # No minimum validation
    reservation_hours: int = 1  # No minimum validation
```

**Suggested Fix:**
Add Pydantic validators:
```python
from pydantic import field_validator

class TestConfig(BaseModel):
    max_price: float = 0.50
    reservation_hours: int = 1

    @field_validator('max_price')
    @classmethod
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('max_price must be positive')
        return v
```

---

### BUG-010: WebSocket Initial Status Sends Raw Bytes
**Severity:** Low
**File:** /Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py
**Line:** 848-849

**Description:**
The WebSocket endpoint decodes the JSON response body and wraps it in another JSON object, which is inefficient and could cause encoding issues.

**Code:**
```python
status = await get_status()
await websocket.send_json({"type": "init", "status": status.body.decode()})
```

**Suggested Fix:**
Return the data directly rather than re-serializing:
```python
status_data = await get_status_data()  # Return dict, not JSONResponse
await websocket.send_json({"type": "init", "status": status_data})
```

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 1     |
| High     | 3     |
| Medium   | 3     |
| Low      | 3     |
| **Total**| **10**|

### Priority Actions
1. **Immediate:** Fix BUG-001 (race condition) - can cause data corruption
2. **High Priority:** Fix BUG-002, BUG-003, BUG-004 - reliability issues
3. **Medium Priority:** Fix BUG-005, BUG-006, BUG-007 - debugging/maintenance issues
4. **Low Priority:** Fix BUG-008, BUG-009, BUG-010 - code quality improvements

---

## Notes

- Log file analysis could not be completed due to permission restrictions
- Process zombie check could not be completed due to permission restrictions
- Git status check could not be completed (not a git repo issue)
