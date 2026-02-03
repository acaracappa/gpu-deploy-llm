# GPU Deploy LLM Dashboard - Integration Issues Report

**Generated:** 2026-01-31
**Agent:** Integration Test Agent
**Services Tested:**
- Dashboard: http://localhost:8081
- Cloud-GPU-Shopper: http://localhost:8080

---

## Test Summary

| Test Category | Status | Issues Found |
|---------------|--------|--------------|
| WebSocket Connection | NEEDS_VERIFICATION | 3 |
| API Integration | NEEDS_VERIFICATION | 4 |
| State Synchronization | NEEDS_VERIFICATION | 5 |
| Error Handling | NEEDS_VERIFICATION | 3 |

---

## 1. WebSocket Connection Tests

### INT-WS-001: WebSocket Reconnection Race Condition

**Component:** `gpu_deploy_llm/web/static/index.html` (line 556-578)

**Steps to Reproduce:**
1. Connect to dashboard at http://localhost:8081
2. Stop the dashboard server
3. Restart the dashboard server
4. Observe WebSocket reconnection behavior

**Expected Behavior:**
WebSocket should reconnect cleanly and receive fresh state via `init` message.

**Potential Issue:**
The reconnection logic uses exponential backoff but does NOT clear `sessionLog` or `benchmarksBySession` state before reconnecting. This could cause duplicate entries if the server sends `init` data that overlaps with locally cached data.

**Code Reference:**
```javascript
ws.onclose = () => {
    updateConnectionStatus(false);
    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
    reconnectAttempts++;
    setTimeout(connect, delay);
};
```

**Recommendation:** Clear local state or deduplicate on reconnect.

---

### INT-WS-002: WebSocket Ping/Pong Not Synchronized with Server

**Component:** `gpu_deploy_llm/web/server.py` (line 840-865) and `static/index.html` (line 568)

**Steps to Reproduce:**
1. Open dashboard
2. Wait for WebSocket timeout (30 seconds server-side)
3. Monitor network traffic

**Expected Behavior:**
Client and server should maintain synchronized keepalive.

**Potential Issue:**
Server sends "ping" after 30s timeout, expects "pong". Client responds to "ping" with "pong" but the timing mismatch could cause connection drops if client is slow.

**Code Reference (Server):**
```python
data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
if data == "ping":
    await websocket.send_text("pong")
```

**Code Reference (Client):**
```javascript
if (event.data === 'ping') { ws.send('pong'); return; }
```

**Recommendation:** Implement bidirectional heartbeat with client-initiated pings.

---

### INT-WS-003: Event JSON Parse Error Silently Ignored

**Component:** `gpu_deploy_llm/web/static/index.html` (line 569)

**Steps to Reproduce:**
1. Send malformed JSON via WebSocket (inject via dev tools)
2. Observe console for errors

**Expected Behavior:**
Graceful error handling with user notification.

**Actual Behavior:**
Parse errors are silently caught and ignored:
```javascript
try { handleEvent(JSON.parse(event.data)); } catch (e) {}
```

**Recommendation:** Add error logging and potentially show connection quality indicator.

---

## 2. API Integration Tests

### INT-API-001: Shopper Connection Timeout Too Short

**Component:** `gpu_deploy_llm/client/shopper.py` (line 46-48)

**Steps to Reproduce:**
1. Start shopper with slow startup
2. Call `/api/status` from dashboard
3. Observe timeout behavior

**Expected Behavior:**
Reasonable timeout for initial connection.

**Potential Issue:**
Default connect timeout is 30s but command timeout is 60s. During shopper startup sweep, `/ready` returns 503 for up to 2 minutes. The `wait_for_ready` in TestRunner has 30s timeout (line 129) which may be too short.

**Code Reference:**
```python
await client.wait_for_ready(timeout=30)  # server.py line 129
```

vs

```python
async def wait_for_ready(self, timeout: float = 120.0, ...):  # shopper.py line 214
```

**Recommendation:** Align timeouts - use 120s default in TestRunner.

---

### INT-API-002: Session Filtering Includes Version in Consumer ID

**Component:** `gpu_deploy_llm/web/server.py` (line 594-597)

**Steps to Reproduce:**
1. Run dashboard version 1.0.0, create sessions
2. Upgrade to version 1.0.1
3. Fetch sessions via `/api/sessions`

**Expected Behavior:**
Should see all gpu-deploy-llm sessions regardless of version.

**Actual Behavior:**
Only sessions from current version are shown:
```python
sessions = await client.list_sessions(
    consumer_id=f"gpu-deploy-llm/v{__version__}",  # Version-specific!
    limit=limit
)
```

**Recommendation:** Filter by prefix "gpu-deploy-llm/" instead of exact version match.

---

### INT-API-003: Dashboard Sessions Endpoint Missing Error Details

**Component:** `gpu_deploy_llm/web/server.py` (line 586-621)

**Steps to Reproduce:**
1. Create a session that fails
2. Fetch via `/api/sessions`
3. Check if error message is included

**Expected Behavior:**
Session error message should be included in response.

**Actual Behavior:**
The `error` field from Session model is NOT included in the response mapping (lines 603-612).

**Code Reference:**
```python
result.append({
    "id": s.id,
    "status": s.status.value,
    # ... other fields
    # MISSING: "error": s.error
})
```

**Recommendation:** Add `"error": s.error` to response mapping.

---

### INT-API-004: Benchmark Store Path Not Configurable

**Component:** `gpu_deploy_llm/web/server.py` (line 640)

**Steps to Reproduce:**
1. Run dashboard in different environments
2. Check where benchmarks are stored

**Expected Behavior:**
Benchmark storage location should be configurable.

**Potential Issue:**
`BenchmarkStore()` is called without path parameter. Need to verify default path is appropriate for production use.

**Recommendation:** Make benchmark store path configurable via environment variable or server config.

---

## 3. State Synchronization Tests

### INT-SYNC-001: Dashboard-Shopper Session State Mismatch (Phantom Sessions)

**Component:** `gpu_deploy_llm/web/static/index.html` (line 615-643)

**Steps to Reproduce:**
1. Create session via dashboard
2. Manually delete session via shopper API
3. Refresh dashboard sessions tab

**Expected Behavior:**
Session should disappear from dashboard.

**Potential Issue:**
Local `sessionLog` maintains session state that may persist after shopper removes the session. The `fetchSessions` function REPLACES local sessionLog but tries to preserve `result` state:

```javascript
const existingResults = {};
sessionLog.forEach(s => { if (s.result) existingResults[s.id] = s.result; });
```

If a session is deleted from shopper but had a local `result` state, it won't appear in the new fetch but the `existingResults` map still holds orphaned data.

**Recommendation:** Clear orphaned local state when session not found in shopper.

---

### INT-SYNC-002: Benchmark-Session Association Fragile

**Component:** `gpu_deploy_llm/web/static/index.html` (line 923-925)

**Steps to Reproduce:**
1. Run benchmark for session A
2. Session A stops/fails
3. Re-run test creating session B
4. Check benchmark association

**Expected Behavior:**
Benchmarks correctly linked to sessions.

**Potential Issue:**
Benchmarks are indexed by session_id:
```javascript
benchmarksBySession = {};
allBenchmarks.forEach(b => {
    benchmarksBySession[b.session_id] = b;
});
```

If same session_id is reused (unlikely but possible with provider), benchmark data could be overwritten.

**Recommendation:** Index by benchmark_id with session_id as lookup key.

---

### INT-SYNC-003: Real-time Session Updates Not Deduplicated

**Component:** `gpu_deploy_llm/web/static/index.html` (line 615-643)

**Steps to Reproduce:**
1. Create session
2. Receive WebSocket session update
3. Simultaneously call fetchSessions()
4. Check for duplicate entries

**Expected Behavior:**
Single session entry regardless of update source.

**Potential Issue:**
`updateSession` (WebSocket) and `fetchSessions` (polling) both modify `sessionLog`. Race condition could cause:
1. WebSocket adds session to sessionLog
2. fetchSessions replaces sessionLog
3. WebSocket update arrives for same session
4. Session appears twice briefly

**Code shows deduplication attempt but timing window exists:**
```javascript
const existingIdx = sessionLog.findIndex(s => s.id === data.id);
```

**Recommendation:** Add mutex/lock or use session ID as map key instead of array.

---

### INT-SYNC-004: Status Polling Interval Mismatch

**Component:** `gpu_deploy_llm/web/static/index.html` (line 1047-1049)

**Steps to Reproduce:**
1. Monitor network traffic
2. Observe polling intervals

**Potential Issue:**
Multiple polling intervals running concurrently:
```javascript
setInterval(fetchStatus, 30000);      // 30s
setInterval(fetchSessions, 10000);    // 10s
setInterval(fetchBenchmarks, 15000);  // 15s
```

This creates uneven server load and potential race conditions.

**Recommendation:** Consolidate into single poll with staggered sub-requests or use WebSocket push for all updates.

---

### INT-SYNC-005: currentSessionId Not Cleared on Test Completion

**Component:** `gpu_deploy_llm/web/static/index.html` (line 801-830)

**Steps to Reproduce:**
1. Run test to completion
2. Start new test
3. Check currentSessionId state

**Expected Behavior:**
currentSessionId should be cleared and set to new session.

**Potential Issue:**
On test success, currentSessionId is cleared (line 821). On test failure with cleanup, it's cleared in `cleanupSessionById`. But if test fails WITHOUT auto_cleanup, currentSessionId remains set to old session ID, potentially causing subsequent operations to affect wrong session.

**Recommendation:** Always clear currentSessionId in handleResult before starting new test.

---

## 4. Error Handling Tests

### INT-ERR-001: Shopper Unreachable Error Not User-Friendly

**Component:** `gpu_deploy_llm/web/server.py` (line 517-567)

**Steps to Reproduce:**
1. Stop cloud-gpu-shopper
2. Load dashboard
3. Check status display

**Expected Behavior:**
Clear message that shopper is offline with troubleshooting steps.

**Actual Behavior:**
Shows "Offline" status but no guidance. The error is only logged server-side:
```python
except Exception as e:
    status["error"] = str(e)  # Raw exception message
```

**Recommendation:** Add user-friendly error messages and suggested actions.

---

### INT-ERR-002: Invalid Session ID Returns Generic 404

**Component:** `gpu_deploy_llm/client/shopper.py` (line 343-363)

**Steps to Reproduce:**
1. Call `/api/session/cleanup` with invalid session ID
2. Check error response

**Expected Behavior:**
Clear error message indicating session not found.

**Actual Behavior:**
`ShopperAPIError` with "Session not found" but dashboard's `cleanupSessionById` catches all errors and shows generic message:
```javascript
addLog('error', `Cleanup failed: ${data.detail || 'Unknown error'}`);
```

**Recommendation:** Pass through structured error codes for better UI handling.

---

### INT-ERR-003: WebSocket Broadcast Failure Not Reported

**Component:** `gpu_deploy_llm/web/server.py` (line 479-488)

**Steps to Reproduce:**
1. Connect multiple WebSocket clients
2. Disconnect one client abruptly
3. Trigger event broadcast

**Expected Behavior:**
Failed sends should be logged for debugging.

**Actual Behavior:**
Exceptions are silently caught and client removed:
```python
try:
    await ws.send_text(event.to_json())
except Exception:
    disconnected.add(ws)
```

**Recommendation:** Add debug logging for broadcast failures.

---

## Recommendations Summary

### High Priority
1. **INT-SYNC-001**: Implement proper session state reconciliation
2. **INT-API-002**: Fix version-specific consumer_id filtering
3. **INT-API-001**: Align timeout configurations

### Medium Priority
4. **INT-WS-001**: Clear local state on WebSocket reconnect
5. **INT-SYNC-003**: Use map-based session storage to prevent duplicates
6. **INT-API-003**: Include error field in session response

### Low Priority
7. **INT-WS-003**: Add error logging for JSON parse failures
8. **INT-SYNC-004**: Consolidate polling intervals
9. **INT-ERR-001**: Improve user-facing error messages

---

## Test Verification Commands

When services are available, verify issues with these commands:

```bash
# Test 1: WebSocket Connection
wscat -c ws://localhost:8081/ws

# Test 2: API Status
curl -s http://localhost:8081/api/status | jq .

# Test 3: Shopper Health
curl -s http://localhost:8080/health | jq .

# Test 4: Compare Sessions
diff <(curl -s http://localhost:8081/api/sessions | jq '.sessions[].id' | sort) \
     <(curl -s http://localhost:8080/api/v1/sessions | jq '.[].id' | sort)

# Test 5: Invalid Session Cleanup
curl -s -X POST http://localhost:8081/api/session/cleanup \
  -H "Content-Type: application/json" \
  -d '{"session_id":"invalid-session-id-12345"}'

# Test 6: Shopper Unreachable Simulation
# Stop shopper, then:
curl -s http://localhost:8081/api/status | jq '.shopper'
```

---

## Files Analyzed

| File | Purpose | Lines |
|------|---------|-------|
| `/Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/server.py` | FastAPI server with WebSocket | 1303 |
| `/Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/web/static/index.html` | Dashboard SPA | 1054 |
| `/Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/client/shopper.py` | Shopper REST client | 643 |
| `/Users/avc/Documents/gpu-deploy-llm/gpu_deploy_llm/client/models.py` | Pydantic models | 212 |

---

*Report generated by Integration Test Agent*
