#!/usr/bin/env python3
"""
RESE Phase I Performance Benchmarks

Benchmarks Phase I (Epistemic Audit) operations:
- Constraint hardening time
- Assumption mining throughput (assumptions/second)
- Red team protocol execution time

Tests with varying problem sizes:
- Small: 10 assumptions
- Medium: 100 assumptions
- Large: 1000 assumptions

Outputs: JSON with timing statistics (min, max, mean, median, std dev)

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
import json
import time
import statistics
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any
from pathlib import Path

# Add phase1 adapter to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase1" / "src"))

try:
    from phase1_executor import (
        EpistemicAuditExecutor,
        Phase1Config,
        TacitAssumption,
    )
except ImportError as e:
    print(f"Error importing Phase I executor: {e}")
    print("Make sure phase1_executor.py is available")
    sys.exit(1)


# ============================================================================
# BENCHMARK DATA GENERATORS
# ============================================================================

def generate_failure_patterns(count: int) -> List[Dict[str, Any]]:
    """Generate synthetic failure patterns for testing.

    Args:
        count: Number of patterns to generate

    Returns:
        List of failure pattern dictionaries
    """
    patterns = []
    for i in range(count):
        patterns.append({
            "pattern_id": str(uuid.uuid4()),
            "pattern_description": f"Failure pattern {i} related to lattice defects and loading ratio",
            "failure_rate": 0.5 + (i % 50) / 100.0,  # Varying failure rates
            "data_points": 100 + i * 10,
        })
    return patterns


def generate_problem_description(size: str = "medium") -> str:
    """Generate a problem description of specified size.

    Args:
        size: "small", "medium", or "large"

    Returns:
        Problem description string
    """
    base = """
    Material science problem: It is impossible to strengthen lattice defects beyond
    the theoretical limit when loading ratio exceeds critical threshold.
    """

    multipliers = {
        "small": 1,
        "medium": 5,
        "large": 20
    }

    multiplier = multipliers.get(size, 5)
    return (base + "\n") * multiplier


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_constraint_hardening(
    executor: EpistemicAuditExecutor,
    problem_size: str,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark constraint hardening performance.

    Measures:
    - Time to harden constraints
    - Number of constraints extracted

    Args:
        executor: Phase I executor instance
        problem_size: Size of problem ("small", "medium", "large")
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Constraint Hardening ({problem_size})...")

    problem = generate_problem_description(problem_size)
    timings_ms = []

    for i in range(iterations):
        start = time.perf_counter()

        # Hardening only (without full audit)
        constraints = executor.constraint_hardener.harden_constraints(
            problem_description=problem,
            correlation_id=str(uuid.uuid4()),
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({len(constraints)} constraints)")

    # Calculate statistics
    results = {
        "benchmark": "constraint_hardening",
        "problem_size": problem_size,
        "iterations": iterations,
        "constraints_count": len(constraints),
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "constraints_per_second": round(len(constraints) / (statistics.mean(timings_ms) / 1000), 2)
        }
    }

    return results


def benchmark_assumption_mining(
    executor: EpistemicAuditExecutor,
    assumption_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark assumption mining performance.

    Measures:
    - Time to mine assumptions
    - Assumptions per second throughput

    Args:
        executor: Phase I executor instance
        assumption_count: Number of failure patterns (assumptions to mine)
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Assumption Mining ({assumption_count} patterns)...")

    failure_patterns = generate_failure_patterns(assumption_count)
    timings_ms = []
    assumption_counts = []

    for i in range(iterations):
        start = time.perf_counter()

        # Mining only
        assumptions = executor.assumption_miner.mine_assumptions(
            failure_patterns=failure_patterns,
            correlation_id=str(uuid.uuid4()),
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)
        assumption_counts.append(len(assumptions))

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({len(assumptions)} assumptions)")

    # Calculate statistics
    avg_assumptions = statistics.mean(assumption_counts)
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "assumption_mining",
        "input_patterns": assumption_count,
        "iterations": iterations,
        "assumptions_mined": {
            "min": min(assumption_counts),
            "max": max(assumption_counts),
            "mean": round(avg_assumptions, 1),
        },
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "assumptions_per_second": round(avg_assumptions / avg_time_sec, 2)
        }
    }

    return results


def benchmark_red_team_protocol(
    executor: EpistemicAuditExecutor,
    assumption_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark red team protocol performance.

    Measures:
    - Time to execute red team attacks
    - Hypotheses tested per second

    Args:
        executor: Phase I executor instance
        assumption_count: Number of assumptions to test
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Red Team Protocol ({assumption_count} assumptions)...")

    # Generate synthetic assumptions
    assumptions = [
        TacitAssumption(
            id=str(uuid.uuid4()),
            description=f"Assumption {i} about parameter behavior",
            source_pattern=f"Pattern {i}",
            confidence_score=0.5 + (i % 50) / 100.0,
            supporting_evidence_count=10 + i,
        )
        for i in range(assumption_count)
    ]

    timings_ms = []
    falsification_counts = []

    for i in range(iterations):
        start = time.perf_counter()

        # Red team protocol
        falsification_results = executor.red_team.attack_hypotheses(
            assumptions=assumptions,
            constraints=[],
            correlation_id=str(uuid.uuid4()),
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)
        falsification_counts.append(len(falsification_results))

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({len(falsification_results)} falsifications)")

    # Calculate statistics
    avg_falsifications = statistics.mean(falsification_counts)
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "red_team_protocol",
        "assumptions_tested": assumption_count,
        "iterations": iterations,
        "falsifications": {
            "min": min(falsification_counts),
            "max": max(falsification_counts),
            "mean": round(avg_falsifications, 1),
        },
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "assumptions_per_second": round(assumption_count / avg_time_sec, 2)
        }
    }

    return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_phase1_benchmarks() -> Dict[str, Any]:
    """Run all Phase I benchmarks.

    Returns:
        Complete benchmark results
    """
    print("=" * 70)
    print("RESE Phase I Performance Benchmarks")
    print("=" * 70)

    # Initialize executor with default config
    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config)

    results = {
        "phase": "phase1_epistemic_audit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": []
    }

    # Benchmark 1: Constraint Hardening
    print("\n" + "=" * 70)
    print("Benchmark Suite 1: Constraint Hardening")
    print("=" * 70)

    for size in ["small", "medium", "large"]:
        result = benchmark_constraint_hardening(executor, size, iterations=5)
        results["benchmarks"].append(result)

    # Benchmark 2: Assumption Mining
    print("\n" + "=" * 70)
    print("Benchmark Suite 2: Assumption Mining")
    print("=" * 70)

    for count in [10, 100, 1000]:
        result = benchmark_assumption_mining(executor, count, iterations=5)
        results["benchmarks"].append(result)

    # Benchmark 3: Red Team Protocol
    print("\n" + "=" * 70)
    print("Benchmark Suite 3: Red Team Protocol")
    print("=" * 70)

    for count in [10, 100, 1000]:
        result = benchmark_red_team_protocol(executor, count, iterations=5)
        results["benchmarks"].append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("Phase I Benchmark Summary")
    print("=" * 70)

    for benchmark in results["benchmarks"]:
        print(f"\n{benchmark['benchmark'].upper()} ({benchmark.get('problem_size', benchmark.get('input_patterns', benchmark.get('assumptions_tested', 'N/A')))}):")
        print(f"  Mean Time: {benchmark['timings_ms']['mean']}ms")
        print(f"  Throughput: {benchmark['throughput']}")

    return results


def save_results(results: Dict[str, Any], output_dir: str = None):
    """Save benchmark results to JSON file.

    Args:
        results: Benchmark results dictionary
        output_dir: Output directory (defaults to results/)
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent / "results")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"phase1_benchmark_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run benchmarks
    results = run_phase1_benchmarks()

    # Save results
    save_results(results)

    print("\n" + "=" * 70)
    print("Phase I Benchmarks Complete!")
    print("=" * 70)
