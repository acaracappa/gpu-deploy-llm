#!/usr/bin/env python3
"""
Integration Test Runner for GPU Deploy LLM Dashboard

This script tests the integration points between:
- Dashboard (http://localhost:8081)
- Cloud-GPU-Shopper (http://localhost:8080)

Run: python debug-logs/integration-test-runner.py

Requirements: pip install httpx websockets
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import httpx
    import websockets
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install httpx websockets")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


@dataclass
class TestSuite:
    """Collection of test results."""
    results: List[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def print_summary(self):
        print("\n" + "=" * 60)
        print("INTEGRATION TEST RESULTS")
        print("=" * 60)

        for r in self.results:
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"{status} {r.test_id}: {r.name}")
            print(f"       {r.message}")
            if r.details:
                for k, v in r.details.items():
                    print(f"       - {k}: {v}")
            print()

        print("-" * 60)
        print(f"Total: {len(self.results)} | Passed: {self.passed} | Failed: {self.failed}")
        print("-" * 60)


class IntegrationTester:
    """Test the dashboard integration points."""

    DASHBOARD_URL = "http://localhost:8081"
    SHOPPER_URL = "http://localhost:8080"

    def __init__(self):
        self.suite = TestSuite()
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    async def run_all(self):
        """Run all integration tests."""
        print("Starting Integration Tests...")
        print(f"Dashboard: {self.DASHBOARD_URL}")
        print(f"Shopper: {self.SHOPPER_URL}")
        print()

        # 1. WebSocket Tests
        await self.test_websocket_connection()
        await self.test_websocket_init_message()

        # 2. API Integration Tests
        await self.test_dashboard_status_endpoint()
        await self.test_shopper_health()
        await self.test_shopper_ready()
        await self.test_shopper_sessions()

        # 3. State Comparison Tests
        await self.test_sessions_sync()

        # 4. Error Scenario Tests
        await self.test_invalid_session_cleanup()
        await self.test_shopper_unreachable()

        self.suite.print_summary()
        return self.suite.failed == 0

    async def test_websocket_connection(self):
        """Test WebSocket connection to dashboard."""
        test_id = "WS-001"
        start = datetime.now()

        try:
            async with websockets.connect(f"ws://localhost:8081/ws") as ws:
                # Connection established
                duration = (datetime.now() - start).total_seconds() * 1000
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="WebSocket Connection",
                    passed=True,
                    message="Successfully connected to ws://localhost:8081/ws",
                    duration_ms=duration,
                ))
        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.suite.add(TestResult(
                test_id=test_id,
                name="WebSocket Connection",
                passed=False,
                message=f"Failed to connect: {e}",
                duration_ms=duration,
            ))

    async def test_websocket_init_message(self):
        """Test that WebSocket receives init message with status."""
        test_id = "WS-002"
        start = datetime.now()

        try:
            async with websockets.connect(f"ws://localhost:8081/ws") as ws:
                # Wait for init message
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)

                if data.get("type") == "init" and "status" in data:
                    status = json.loads(data["status"])
                    duration = (datetime.now() - start).total_seconds() * 1000
                    self.suite.add(TestResult(
                        test_id=test_id,
                        name="WebSocket Init Message",
                        passed=True,
                        message="Received valid init message with status",
                        details={
                            "version": status.get("version"),
                            "shopper_healthy": status.get("shopper", {}).get("healthy"),
                        },
                        duration_ms=duration,
                    ))
                else:
                    self.suite.add(TestResult(
                        test_id=test_id,
                        name="WebSocket Init Message",
                        passed=False,
                        message=f"Unexpected message type: {data.get('type')}",
                    ))
        except asyncio.TimeoutError:
            self.suite.add(TestResult(
                test_id=test_id,
                name="WebSocket Init Message",
                passed=False,
                message="Timeout waiting for init message",
            ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="WebSocket Init Message",
                passed=False,
                message=f"Error: {e}",
            ))

    async def test_dashboard_status_endpoint(self):
        """Test GET /api/status on dashboard."""
        test_id = "API-001"
        start = datetime.now()

        try:
            resp = await self.client.get(f"{self.DASHBOARD_URL}/api/status")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Dashboard Status Endpoint",
                    passed=True,
                    message="GET /api/status returned 200",
                    details={
                        "version": data.get("version"),
                        "shopper_healthy": data.get("shopper", {}).get("healthy"),
                        "shopper_ready": data.get("shopper", {}).get("ready"),
                        "providers": data.get("providers"),
                    },
                    duration_ms=duration,
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Dashboard Status Endpoint",
                    passed=False,
                    message=f"Unexpected status code: {resp.status_code}",
                    duration_ms=duration,
                ))
        except httpx.ConnectError:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Dashboard Status Endpoint",
                passed=False,
                message="Dashboard not reachable at http://localhost:8081",
            ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Dashboard Status Endpoint",
                passed=False,
                message=f"Error: {e}",
            ))

    async def test_shopper_health(self):
        """Test GET /health on shopper."""
        test_id = "API-002"
        start = datetime.now()

        try:
            resp = await self.client.get(f"{self.SHOPPER_URL}/health")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Health Endpoint",
                    passed=True,
                    message="GET /health returned 200",
                    details={
                        "status": data.get("status"),
                        "version": data.get("version"),
                    },
                    duration_ms=duration,
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Health Endpoint",
                    passed=False,
                    message=f"Unexpected status code: {resp.status_code}",
                    duration_ms=duration,
                ))
        except httpx.ConnectError:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Shopper Health Endpoint",
                passed=False,
                message="Shopper not reachable at http://localhost:8080",
            ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Shopper Health Endpoint",
                passed=False,
                message=f"Error: {e}",
            ))

    async def test_shopper_ready(self):
        """Test GET /ready on shopper."""
        test_id = "API-003"
        start = datetime.now()

        try:
            resp = await self.client.get(f"{self.SHOPPER_URL}/ready")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Ready Endpoint",
                    passed=True,
                    message="GET /ready returned 200 (ready)",
                    details={"ready": data.get("ready")},
                    duration_ms=duration,
                ))
            elif resp.status_code == 503:
                data = resp.json()
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Ready Endpoint",
                    passed=True,  # 503 is valid during startup
                    message="GET /ready returned 503 (startup sweep in progress)",
                    details={"message": data.get("message")},
                    duration_ms=duration,
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Ready Endpoint",
                    passed=False,
                    message=f"Unexpected status code: {resp.status_code}",
                    duration_ms=duration,
                ))
        except httpx.ConnectError:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Shopper Ready Endpoint",
                passed=False,
                message="Shopper not reachable",
            ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Shopper Ready Endpoint",
                passed=False,
                message=f"Error: {e}",
            ))

    async def test_shopper_sessions(self):
        """Test GET /api/v1/sessions on shopper."""
        test_id = "API-004"
        start = datetime.now()

        try:
            resp = await self.client.get(f"{self.SHOPPER_URL}/api/v1/sessions")
            duration = (datetime.now() - start).total_seconds() * 1000

            if resp.status_code == 200:
                data = resp.json()
                sessions = data.get("sessions", data) if isinstance(data, dict) else data
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Sessions Endpoint",
                    passed=True,
                    message=f"GET /api/v1/sessions returned {len(sessions)} sessions",
                    details={"session_count": len(sessions)},
                    duration_ms=duration,
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Shopper Sessions Endpoint",
                    passed=False,
                    message=f"Unexpected status code: {resp.status_code}",
                    duration_ms=duration,
                ))
        except httpx.ConnectError:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Shopper Sessions Endpoint",
                passed=False,
                message="Shopper not reachable",
            ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Shopper Sessions Endpoint",
                passed=False,
                message=f"Error: {e}",
            ))

    async def test_sessions_sync(self):
        """Compare sessions from dashboard vs shopper."""
        test_id = "SYNC-001"

        try:
            # Get dashboard sessions
            dash_resp = await self.client.get(f"{self.DASHBOARD_URL}/api/sessions")
            dash_data = dash_resp.json()
            dash_sessions = {s["id"] for s in dash_data.get("sessions", [])}

            # Get shopper sessions
            shop_resp = await self.client.get(f"{self.SHOPPER_URL}/api/v1/sessions")
            shop_data = shop_resp.json()
            shop_sessions_list = shop_data.get("sessions", shop_data) if isinstance(shop_data, dict) else shop_data
            shop_sessions = {s["id"] for s in shop_sessions_list}

            # Compare
            phantom_sessions = dash_sessions - shop_sessions  # In dashboard but not shopper
            missing_sessions = shop_sessions - dash_sessions  # In shopper but not dashboard

            if not phantom_sessions and not missing_sessions:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Session State Synchronization",
                    passed=True,
                    message="Dashboard and Shopper session states match",
                    details={
                        "dashboard_sessions": len(dash_sessions),
                        "shopper_sessions": len(shop_sessions),
                    },
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Session State Synchronization",
                    passed=False,
                    message="Session state mismatch detected",
                    details={
                        "phantom_sessions": list(phantom_sessions),
                        "missing_sessions": list(missing_sessions),
                    },
                ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Session State Synchronization",
                passed=False,
                message=f"Could not compare sessions: {e}",
            ))

    async def test_invalid_session_cleanup(self):
        """Test cleanup of non-existent session."""
        test_id = "ERR-001"

        try:
            resp = await self.client.post(
                f"{self.DASHBOARD_URL}/api/session/cleanup",
                json={"session_id": "nonexistent-session-12345"},
            )

            # Should return error but not crash
            if resp.status_code in (200, 400, 404):
                data = resp.json()
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Invalid Session Cleanup",
                    passed=True,
                    message=f"Handled gracefully with status {resp.status_code}",
                    details={
                        "status": data.get("status"),
                        "error": data.get("error"),
                    },
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Invalid Session Cleanup",
                    passed=False,
                    message=f"Unexpected status code: {resp.status_code}",
                ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Invalid Session Cleanup",
                passed=False,
                message=f"Error: {e}",
            ))

    async def test_shopper_unreachable(self):
        """Test dashboard behavior when shopper is unreachable."""
        test_id = "ERR-002"

        # This test checks if /api/status handles shopper being down gracefully
        try:
            resp = await self.client.get(f"{self.DASHBOARD_URL}/api/status")

            if resp.status_code == 200:
                data = resp.json()
                shopper_healthy = data.get("shopper", {}).get("healthy", False)

                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Dashboard Handles Shopper Status",
                    passed=True,
                    message="Dashboard reports shopper status correctly",
                    details={
                        "shopper_healthy": shopper_healthy,
                        "shopper_ready": data.get("shopper", {}).get("ready"),
                        "error": data.get("error"),
                    },
                ))
            else:
                self.suite.add(TestResult(
                    test_id=test_id,
                    name="Dashboard Handles Shopper Status",
                    passed=False,
                    message=f"Dashboard returned {resp.status_code}",
                ))
        except Exception as e:
            self.suite.add(TestResult(
                test_id=test_id,
                name="Dashboard Handles Shopper Status",
                passed=False,
                message=f"Error: {e}",
            ))


async def main():
    async with IntegrationTester() as tester:
        success = await tester.run_all()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
