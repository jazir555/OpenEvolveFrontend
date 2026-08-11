#!/usr/bin/env python3
"""
Startup script for RESE Health APIs

This script starts all health check APIs for RESE phases in separate processes.

Usage:
    python start_rese_health_apis.py [--phases PHASE1,PHASE2,...] [--test]

Options:
    --phases: Comma-separated list of phases to start (default: all)
              Options: phase1, phase2, phase3, phase4, aggregate, all
    --test:   Run tests after starting APIs

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict


# Phase configurations
PHASE_CONFIGS = {
    "phase1": {
        "name": "Phase I (Epistemic Audit)",
        "script": "glue/adapters/rese-phase1/src/health_api.py",
        "port_env": "PHASE1_HEALTH_PORT",
        "default_port": "8001",
    },
    "phase2": {
        "name": "Phase II (Isomorphic Mapping)",
        "script": "glue/adapters/rese-phase2/src/health_api.py",
        "port_env": "PHASE2_HEALTH_PORT",
        "default_port": "8002",
    },
    "phase3": {
        "name": "Phase III (MCTS Refinement)",
        "script": "glue/adapters/rese-phase3/src/health_api.py",
        "port_env": "PHASE3_HEALTH_PORT",
        "default_port": "8003",
    },
    "phase4": {
        "name": "Phase IV (Architecture Assembly)",
        "script": "glue/adapters/rese-phase4/src/health_api.py",
        "port_env": "PHASE4_HEALTH_PORT",
        "default_port": "8004",
    },
    "aggregate": {
        "name": "Aggregate Health",
        "script": "glue/adapters/rese-integration/health/aggregate_health.py",
        "port_env": "AGGREGATE_HEALTH_PORT",
        "default_port": "8000",
    },
}


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def print_phase_status(phase: str, config: Dict, status: str, color_code: str = "\033[0m"):
    """Print phase startup status."""
    print(f"{color_code}[*] {config['name']}: {status}{color_code}")


def start_phase_process(phase: str, config: Dict, base_dir: Path) -> subprocess.Popen:
    """
    Start a health API process for a phase.

    Args:
        phase: Phase key (e.g., "phase1")
        config: Phase configuration
        base_dir: Base directory

    Returns:
        Subprocess object
    """
    script_path = base_dir / config["script"]

    # Set environment variables
    env = os.environ.copy()
    env[config["port_env"]] = config["default_port"]

    # Start process
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process


def start_phases(phases_to_start: List[str], base_dir: Path) -> Dict[str, subprocess.Popen]:
    """
    Start health API processes for specified phases.

    Args:
        phases_to_start: List of phase keys to start
        base_dir: Base directory

    Returns:
        Dictionary mapping phase keys to subprocess objects
    """
    processes = {}

    print_header("Starting RESE Health APIs")

    for phase in phases_to_start:
        if phase not in PHASE_CONFIGS:
            print(f"\033[91m[!] Unknown phase: {phase}\033[0m")
            continue

        config = PHASE_CONFIGS[phase]

        print(f"\033[94m[*] Starting {config['name']}...\033[0m")
        print(f"    Script: {config['script']}")
        print(f"    Port: {config['default_port']}")

        try:
            process = start_phase_process(phase, config, base_dir)
            processes[phase] = process
            print(f"\033[92m[[OK]] Started {config['name']} (PID: {process.pid})\033[0m")

            # Give it time to start
            time.sleep(1)

        except Exception as e:
            print(f"\033[91m[[FAIL]] Failed to start {config['name']}: {e}\033[0m")

    return processes


def wait_for_startup(phases: List[str], timeout: int = 10):
    """
    Wait for APIs to be ready.

    Args:
        phases: List of phase keys
        timeout: Timeout in seconds
    """
    print_header("Waiting for APIs to be ready")

    import aiohttp
    import asyncio

    async def check_ready():
        urls = {
            "phase1": "http://localhost:8001/health",
            "phase2": "http://localhost:8002/health",
            "phase3": "http://localhost:8003/health",
            "phase4": "http://localhost:8004/health",
            "aggregate": "http://localhost:8000/health",
        }

        async with aiohttp.ClientSession() as session:
            for phase in phases:
                if phase not in urls:
                    continue

                url = urls[phase]
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            config = PHASE_CONFIGS[phase]
                            print(f"\033[92m[[OK]] {config['name']} is ready\033[0m")
                        else:
                            config = PHASE_CONFIGS[phase]
                            print(f"\033[93m[!] {config['name']} returned {response.status}\033[0m")
                except Exception as e:
                    config = PHASE_CONFIGS[phase]
                    print(f"\033[91m[[FAIL]] {config['name']} not ready: {e}\033[0m")

    try:
        asyncio.run(check_ready())
    except ImportError:
        print("\033[93m[!] aiohttp not installed, skipping readiness check\033[0m")


def run_tests():
    """
    Run health endpoint tests.
    """
    print_header("Running Health Endpoint Tests")

    test_script = Path(__file__).parent / "test_rese_health_endpoints.py"

    if not test_script.exists():
        print(f"\033[91m[[FAIL]] Test script not found: {test_script}\033[0m")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=Path(__file__).parent,
            env=os.environ.copy()
        )

        return result.returncode == 0

    except Exception as e:
        print(f"\033[91m[[FAIL]] Failed to run tests: {e}\033[0m")
        return False


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Start RESE Health APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start all health APIs
    python start_rese_health_apis.py

    # Start specific phases
    python start_rese_health_apis.py --phases phase1,phase2

    # Start and test
    python start_rese_health_apis.py --test

    # Start only aggregate health
    python start_rese_health_apis.py --phases aggregate
        """
    )

    parser.add_argument(
        "--phases",
        type=str,
        default="all",
        help="Comma-separated list of phases to start (phase1,phase2,phase3,phase4,aggregate,all)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests after starting APIs"
    )

    args = parser.parse_args()

    # Determine which phases to start
    if args.phases.lower() == "all":
        phases_to_start = list(PHASE_CONFIGS.keys())
    else:
        phases_to_start = [p.strip().lower() for p in args.phases.split(",")]
        # Validate phase names
        for phase in phases_to_start:
            if phase not in PHASE_CONFIGS:
                print(f"\033[91mError: Unknown phase '{phase}'\033[0m")
                print(f"Valid phases: {', '.join(PHASE_CONFIGS.keys())}")
                sys.exit(1)

    # Get base directory
    base_dir = Path(__file__).parent

    # Start phases
    processes = start_phases(phases_to_start, base_dir)

    if not processes:
        print("\033[91m[[FAIL]] No phases started\033[0m")
        sys.exit(1)

    # Wait for startup
    wait_for_startup(phases_to_start)

    # Run tests if requested
    if args.test:
        test_passed = run_tests()

        if not test_passed:
            print_header("Tests Failed")
            print("\033[91mSome tests failed. APIs are still running.\033[0m")

    # Print status
    print_header("Health APIs Running")
    print(f"\033[92mStarted {len(processes)} phase(s)\033[0m\n")

    for phase, process in processes.items():
        config = PHASE_CONFIGS[phase]
        print(f"  {config['name']}:")
        print(f"    PID: {process.pid}")
        print(f"    Health: http://localhost:{config['default_port']}/health")
        print(f"    Docs: http://localhost:{config['default_port']}/docs")
        print()

    print("Press Ctrl+C to stop all APIs")

    # Keep script running
    try:
        for process in processes.values():
            process.wait()
    except KeyboardInterrupt:
        print_header("Stopping Health APIs")
        for phase, process in processes.items():
            config = PHASE_CONFIGS[phase]
            print(f"\033[94m[*] Stopping {config['name']}...\033[0m")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"\033[92m[[OK]] Stopped {config['name']}\033[0m")
            except subprocess.TimeoutExpired:
                print(f"\033[91m[[FAIL]] Force killing {config['name']}\033[0m")
                process.kill()

        print(f"\033[92m[[OK]] All APIs stopped\033[0m")


if __name__ == "__main__":
    main()
