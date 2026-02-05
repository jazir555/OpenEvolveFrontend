#!/usr/bin/env python3
"""
Initialize Baseline Results

Creates baseline performance metrics by running all benchmarks.
This baseline will be used for future performance comparisons.

Usage:
    python init_baseline.py

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Script paths
BENCHMARK_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_DIR / "results"
BASELINE_FILE = RESULTS_DIR / "baseline.json"

BENCHMARK_SCRIPTS = [
    ("phase1", BENCHMARK_DIR / "benchmark_phase1.py"),
    ("phase2", BENCHMARK_DIR / "benchmark_phase2.py"),
    ("phase3", BENCHMARK_DIR / "benchmark_phase3.py"),
    ("phase4", BENCHMARK_DIR / "benchmark_phase4.py"),
    ("full_pipeline", BENCHMARK_DIR / "benchmark_full_pipeline.py"),
]


def run_benchmark(script_path: Path) -> dict:
    """Run a benchmark script and return results."""
    print(f"\nRunning {script_path.name}...")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BENCHMARK_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        print(result.stdout)

        # Find most recent result file
        result_files = sorted(
            RESULTS_DIR.glob(f"{script_path.stem}_benchmark_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        if result_files:
            with open(result_files[0], 'r') as f:
                return json.load(f)

    except subprocess.TimeoutExpired:
        print("Error: Benchmark timed out")
    except Exception as e:
        print(f"Error: {e}")

    return None


def main():
    """Initialize baseline by running all benchmarks."""
    print("=" * 70)
    print("RESE Baseline Initialization")
    print("=" * 70)
    print("This will run all benchmarks and save results as baseline.")
    print("Expected runtime: 5-10 minutes\n")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Run all benchmarks
    baseline_results = {
        "baseline": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": {}
    }

    for phase, script_path in BENCHMARK_SCRIPTS:
        results = run_benchmark(script_path)

        if results:
            baseline_results["benchmarks"][phase] = results
            print(f"[OK] {phase} benchmarks completed")
        else:
            print(f"[FAIL] {phase} benchmarks failed")

    # Save baseline
    print("\n" + "=" * 70)
    print("Saving baseline...")
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline_results, f, indent=2)

    print(f"[OK] Baseline saved to: {BASELINE_FILE}")
    print("\nBaseline initialization complete!")
    print("You can now use --compare-baseline to compare future runs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
