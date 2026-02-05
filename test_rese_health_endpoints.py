#!/usr/bin/env python3
"""
Test script for RESE Phase Health Endpoints

This script tests all health check endpoints for:
1. Phase I: Epistemic Audit (port 8001)
2. Phase II: Isomorphic Mapping (port 8002)
3. Phase III: MCTS Refinement (port 8003)
4. Phase IV: Architecture Assembly (port 8004)
5. Aggregate Health (port 8000)

Tests:
- GET /health - Liveness check
- GET /ready - Readiness check
- GET /metrics - Metrics endpoint
- GET / - Root endpoint

Author: RESE Team
Created: 2026-02-04
"""

import asyncio
import aiohttp
from typing import Dict, Any, List
from datetime import datetime
import json
import sys


# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


# Phase configurations
PHASES = {
    "Phase I (Epistemic Audit)": {"url": "http://localhost:8001", "port": 8001},
    "Phase II (Isomorphic Mapping)": {"url": "http://localhost:8002", "port": 8002},
    "Phase III (MCTS Refinement)": {"url": "http://localhost:8003", "port": 8003},
    "Phase IV (Architecture Assembly)": {"url": "http://localhost:8004", "port": 8004},
    "Aggregate Health": {"url": "http://localhost:8000", "port": 8000},
}

ENDPOINTS = ["/", "/health", "/ready", "/metrics"]


async def test_endpoint(session: aiohttp.ClientSession, phase_name: str, base_url: str, endpoint: str) -> Dict[str, Any]:
    """
    Test a single endpoint.

    Args:
        session: aiohttp session
        phase_name: Name of the phase
        base_url: Base URL of the phase API
        endpoint: Endpoint to test

    Returns:
        Test result dictionary
    """
    url = f"{base_url}{endpoint}"
    start_time = asyncio.get_event_loop().time()

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
            response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

            try:
                data = await response.json()
            except:
                data = await response.text()

            return {
                "phase": phase_name,
                "endpoint": endpoint,
                "url": url,
                "status_code": response.status,
                "response_time_ms": round(response_time_ms, 2),
                "success": response.status == 200,
                "data": data,
                "error": None,
            }

    except asyncio.TimeoutError:
        return {
            "phase": phase_name,
            "endpoint": endpoint,
            "url": url,
            "status_code": None,
            "response_time_ms": 5000.0,
            "success": False,
            "data": None,
            "error": "Timeout after 5000ms",
        }

    except Exception as e:
        return {
            "phase": phase_name,
            "endpoint": endpoint,
            "url": url,
            "status_code": None,
            "response_time_ms": 0,
            "success": False,
            "data": None,
            "error": str(e),
        }


async def test_phase(phase_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Test all endpoints for a single phase.

    Args:
        phase_name: Name of the phase
        config: Phase configuration with URL and port

    Returns:
        List of test results
    """
    base_url = config["url"]
    results = []

    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing {phase_name}{Colors.RESET}")
    print(f"URL: {base_url}")
    print("-" * 80)

    async with aiohttp.ClientSession() as session:
        for endpoint in ENDPOINTS:
            result = await test_endpoint(session, phase_name, base_url, endpoint)
            results.append(result)

            # Print result
            status_color = Colors.GREEN if result["success"] else Colors.RED
            status_icon = "[OK]" if result["success"] else "[FAIL]"

            print(f"{status_color}{status_icon} {Colors.BOLD}{endpoint}{Colors.RESET} "
                  f"({result['status_code']}): {result['response_time_ms']}ms")

            if not result["success"] and result["error"]:
                print(f"  {Colors.RED}Error: {result['error']}{Colors.RESET}")

            # Show data preview for successful requests
            if result["success"] and result["data"] and endpoint != "/":
                data_str = json.dumps(result["data"], indent=2)
                if len(data_str) > 200:
                    data_str = data_str[:200] + "..."
                print(f"  {Colors.YELLOW}Response preview:{Colors.RESET}")
                print(f"  {data_str}")

    return results


async def test_all_phases():
    """
    Test all phases and endpoints.
    """
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}RESE Phase Health Endpoints Test{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []
    total_tests = 0
    total_passed = 0

    for phase_name, config in PHASES.items():
        results = await test_phase(phase_name, config)
        all_results.extend(results)

        # Count successes
        for result in results:
            total_tests += 1
            if result["success"]:
                total_passed += 1

    # Print summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}Test Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.RESET}")

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    summary_color = Colors.GREEN if success_rate == 100 else Colors.YELLOW if success_rate >= 50 else Colors.RED

    print(f"{summary_color}{Colors.BOLD}Total Tests: {total_passed}/{total_tests} ({success_rate:.1f}%){Colors.RESET}")

    # Print failed tests
    failed_tests = [r for r in all_results if not r["success"]]
    if failed_tests:
        print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.RESET}")
        for result in failed_tests:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {result['phase']} - {result['endpoint']}")
            if result["error"]:
                print(f"    Error: {result['error']}")

    # Print response time statistics
    response_times = [r["response_time_ms"] for r in all_results if r["success"]]
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        print(f"\n{Colors.BLUE}{Colors.BOLD}Response Time Statistics:{Colors.RESET}")
        print(f"  Average: {avg_response_time:.2f}ms")
        print(f"  Min: {min_response_time:.2f}ms")
        print(f"  Max: {max_response_time:.2f}ms")

    return all_results


def main():
    """
    Main entry point.
    """
    try:
        results = asyncio.run(test_all_phases())

        # Exit with error code if any tests failed
        failed_count = sum(1 for r in results if not r["success"])
        if failed_count > 0:
            sys.exit(1)
        else:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! [OK]{Colors.RESET}")
            sys.exit(0)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}Test suite error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
