#!/usr/bin/env python3
"""
Quick verification script for RESE Health APIs

This script starts a single health API and verifies it works.

Usage:
    python verify_rese_health_apis.py phase1

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
import time
import subprocess
import urllib.request
import json
from pathlib import Path


def verify_phase(phase: str):
    """
    Verify a single phase health API.

    Args:
        phase: Phase key (phase1, phase2, phase3, phase4, aggregate)
    """
    # Phase configurations
    configs = {
        "phase1": {
            "script": "glue/adapters/rese-phase1/src/health_api.py",
            "port": "8001",
            "name": "Phase I (Epistemic Audit)",
        },
        "phase2": {
            "script": "glue/adapters/rese-phase2/src/health_api.py",
            "port": "8002",
            "name": "Phase II (Isomorphic Mapping)",
        },
        "phase3": {
            "script": "glue/adapters/rese-phase3/src/health_api.py",
            "port": "8003",
            "name": "Phase III (MCTS Refinement)",
        },
        "phase4": {
            "script": "glue/adapters/rese-phase4/src/health_api.py",
            "port": "8004",
            "name": "Phase IV (Architecture Assembly)",
        },
        "aggregate": {
            "script": "glue/adapters/rese-integration/health/aggregate_health.py",
            "port": "8000",
            "name": "Aggregate Health",
        },
    }

    if phase not in configs:
        print(f"Error: Unknown phase '{phase}'")
        print(f"Valid phases: {', '.join(configs.keys())}")
        return False

    config = configs[phase]
    base_dir = Path(__file__).parent
    script_path = base_dir / config["script"]

    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False

    print("=" * 80)
    print(f"Verifying {config['name']}")
    print("=" * 80)
    print(f"Script: {config['script']}")
    print(f"Port: {config['port']}")
    print()

    # Set environment
    env = os.environ.copy()

    if phase == "phase1":
        env["PHASE1_HEALTH_PORT"] = config["port"]
    elif phase == "phase2":
        env["PHASE2_HEALTH_PORT"] = config["port"]
    elif phase == "phase3":
        env["PHASE3_HEALTH_PORT"] = config["port"]
    elif phase == "phase4":
        env["PHASE4_HEALTH_PORT"] = config["port"]
    elif phase == "aggregate":
        env["AGGREGATE_HEALTH_PORT"] = config["port"]

    # Start process
    print("Starting health API...")
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(f"Started with PID: {process.pid}")
    print()

    # Wait for startup
    print("Waiting for API to start (5 seconds)...")
    time.sleep(5)

    # Test endpoints
    base_url = f"http://localhost:{config['port']}"
    endpoints = ["/", "/health", "/ready"]

    all_passed = True

    for endpoint in endpoints:
        url = base_url + endpoint
        print(f"Testing {url}...")

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    print(f"  ✓ Status 200 OK")
                    print(f"  Response keys: {list(data.keys())}")
                else:
                    print(f"  ✗ Status {response.status}")
                    all_passed = False
        except urllib.error.HTTPError as e:
            print(f"  ✗ HTTP Error: {e.code}")
            all_passed = False
        except urllib.error.URLError as e:
            print(f"  ✗ URL Error: {e.reason}")
            all_passed = False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_passed = False

        print()

    # Shutdown
    print("=" * 80)
    print("Stopping health API...")
    process.terminate()

    try:
        process.wait(timeout=5)
        print("✓ Stopped cleanly")
    except subprocess.TimeoutExpired:
        print("✗ Force killing")
        process.kill()

    print("=" * 80)

    if all_passed:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python verify_rese_health_apis.py <phase>")
        print("Phases: phase1, phase2, phase3, phase4, aggregate")
        sys.exit(1)

    phase = sys.argv[1].lower()

    try:
        success = verify_phase(phase)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
