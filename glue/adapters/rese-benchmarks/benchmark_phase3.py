#!/usr/bin/env python3
"""
RESE Phase III Performance Benchmarks

Benchmarks Phase III (MCTS Search) operations:
- MCTS iteration throughput (iterations/second)
- Tree node creation rate
- Convergence detection speed

Tests with varying iteration counts:
- Small: 100 iterations
- Medium: 1000 iterations
- Large: 10000 iterations

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
import math
import random
from datetime import datetime, timezone
from typing import Dict, List, Any, Callable
from pathlib import Path

# Add phase3 adapter to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase3" / "src"))

try:
    from phase3_executor import (
        MCTSSearchExecutor,
        Phase3Config,
        Hypothesis,
        SearchTreeNode,
        MCTSNodeState,
        ExplorationStrategy,
    )
except ImportError as e:
    print(f"Error importing Phase III executor: {e}")
    print("Make sure phase3_executor.py is available")
    sys.exit(1)


# ============================================================================
# BENCHMARK HELPERS
# ============================================================================

def generate_hypothesis(statement: str) -> Hypothesis:
    """Generate a synthetic hypothesis for testing.

    Args:
        statement: Hypothesis statement

    Returns:
        Hypothesis object
    """
    return Hypothesis(
        hypothesis_id=str(uuid.uuid4()),
        statement=statement,
        confidence=random.uniform(0.5, 0.9),
        status="validated",
        evidence_count=random.randint(5, 20),
        metadata={"generated": True}
    )


def simple_reward_function(hypothesis: Hypothesis) -> float:
    """Simple reward function for benchmarking.

    Args:
        hypothesis: Hypothesis to evaluate

    Returns:
        Reward value [0.0, 1.0]
    """
    return hypothesis.confidence + random.uniform(-0.05, 0.05)


def hypothesis_generator_factory(count: int = 3) -> Callable[[], List[Hypothesis]]:
    """Factory for creating hypothesis generators.

    Args:
        count: Number of hypotheses to generate

    Returns:
        Hypothesis generator function
    """
    def generate() -> List[Hypothesis]:
        return [
            generate_hypothesis(f"Test hypothesis {i}")
            for i in range(count)
        ]
    return generate


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_mcts_iterations(
    executor: MCTSSearchExecutor,
    iteration_count: int,
    iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark MCTS iteration throughput.

    Measures:
    - Time to complete MCTS iterations
    - Iterations per second
    - Tree statistics

    Args:
        executor: Phase III executor instance
        iteration_count: Number of MCTS iterations to run
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking MCTS Iterations ({iteration_count} iterations)...")

    # Create root hypothesis
    root_hypothesis = generate_hypothesis("Root hypothesis for benchmark")

    timings_ms = []
    tree_stats_list = []

    for i in range(iterations):
        start = time.perf_counter()

        # Create temporary executor with custom iteration count
        temp_config = Phase3Config.from_env()
        temp_config.iterations = iteration_count
        temp_executor = MCTSSearchExecutor(config=temp_config, logger=None)

        # Run search
        result, error = temp_executor.execute_search(
            root_hypothesis=root_hypothesis,
            hypothesis_generator=hypothesis_generator_factory(count=3),
            reward_function=simple_reward_function
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)

        if result:
            tree_stats = {
                "total_nodes": result.total_nodes,
                "max_depth": result.max_depth,
                "iterations_completed": result.iterations,
            }
            tree_stats_list.append(tree_stats)

            print(f"  Iteration {i+1}: {elapsed:.2f}ms ({result.iterations} iters, {result.total_nodes} nodes)")

    # Calculate statistics
    avg_time_sec = statistics.mean(timings_ms) / 1000

    # Average tree stats
    avg_nodes = statistics.mean([s["total_nodes"] for s in tree_stats_list]) if tree_stats_list else 0
    avg_depth = statistics.mean([s["max_depth"] for s in tree_stats_list]) if tree_stats_list else 0

    results = {
        "benchmark": "mcts_iterations",
        "iteration_count": iteration_count,
        "iterations": iterations,
        "tree_stats": {
            "avg_nodes": round(avg_nodes, 1),
            "avg_depth": round(avg_depth, 1),
        },
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "iterations_per_second": round(iteration_count / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


def benchmark_tree_node_creation(
    executor: MCTSSearchExecutor,
    node_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark tree node creation rate.

    Measures:
    - Time to create tree nodes
    - Nodes per second

    Args:
        executor: Phase III executor instance
        node_count: Number of nodes to create
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Tree Node Creation ({node_count} nodes)...")

    timings_ms = []

    for i in range(iterations):
        start = time.perf_counter()

        # Build tree with specified node count
        root_hypothesis = generate_hypothesis("Root")
        root_node = executor.tree_builder.build_root(root_hypothesis)

        created = 1  # Root node
        current_nodes = [root_node]

        while created < node_count and current_nodes:
            parent = current_nodes.pop(0)

            # Generate child hypotheses
            children_count = min(3, node_count - created)
            child_hypotheses = [
                generate_hypothesis(f"Node {created + j}")
                for j in range(children_count)
            ]

            # Expand node
            new_nodes = executor.tree_builder.expand_node(
                parent,
                child_hypotheses
            )

            created += len(new_nodes)
            current_nodes.extend(new_nodes)

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({created} nodes)")

    # Calculate statistics
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "tree_node_creation",
        "node_count": node_count,
        "iterations": iterations,
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "nodes_per_second": round(node_count / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


def benchmark_convergence_detection(
    executor: MCTSSearchExecutor,
    window_size: int,
    iterations: int = 10
) -> Dict[str, Any]:
    """Benchmark convergence detection speed.

    Measures:
    - Time to check convergence
    - Convergence checks per second

    Args:
        executor: Phase III executor instance
        window_size: ACI window size
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Convergence Detection (window_size={window_size})...")

    timings_us = []

    # Populate convergence history
    for i in range(window_size):
        executor.convergence_detector.update(
            iteration=i,
            best_confidence=0.7 + (i % 10) / 100.0,
            best_reward=0.6 + (i % 10) / 100.0
        )

    for i in range(iterations):
        start = time.perf_counter()

        # Check convergence
        is_converged, aci_value = executor.convergence_detector.check_convergence()

        elapsed_us = (time.perf_counter() - start) * 1_000_000
        timings_us.append(elapsed_us)

        if i == 0 or (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}: {elapsed_us:.2f}μs (converged={is_converged}, aci={aci_value})")

    # Calculate statistics
    avg_time_us = statistics.mean(timings_us)
    avg_time_sec = avg_time_us / 1_000_000

    results = {
        "benchmark": "convergence_detection",
        "window_size": window_size,
        "iterations": iterations,
        "timings_us": {
            "min": round(min(timings_us), 2),
            "max": round(max(timings_us), 2),
            "mean": round(avg_time_us, 2),
            "median": round(statistics.median(timings_us), 2),
            "stdev": round(statistics.stdev(timings_us), 2) if len(timings_us) > 1 else 0.0,
        },
        "throughput": {
            "checks_per_second": round(1.0 / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_phase3_benchmarks() -> Dict[str, Any]:
    """Run all Phase III benchmarks.

    Returns:
        Complete benchmark results
    """
    print("=" * 70)
    print("RESE Phase III Performance Benchmarks")
    print("=" * 70)

    # Initialize executor with default config
    config = Phase3Config.from_env()
    executor = MCTSSearchExecutor(config=config)

    results = {
        "phase": "phase3_mcts_search",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": []
    }

    # Benchmark 1: MCTS Iterations
    print("\n" + "=" * 70)
    print("Benchmark Suite 1: MCTS Iterations")
    print("=" * 70)

    for iter_count in [100, 1000, 10000]:
        result = benchmark_mcts_iterations(executor, iter_count, iterations=3)
        results["benchmarks"].append(result)

    # Benchmark 2: Tree Node Creation
    print("\n" + "=" * 70)
    print("Benchmark Suite 2: Tree Node Creation")
    print("=" * 70)

    for node_count in [100, 1000, 5000]:
        result = benchmark_tree_node_creation(executor, node_count, iterations=5)
        results["benchmarks"].append(result)

    # Benchmark 3: Convergence Detection
    print("\n" + "=" * 70)
    print("Benchmark Suite 3: Convergence Detection")
    print("=" * 70)

    for window_size in [50, 100, 200]:
        result = benchmark_convergence_detection(executor, window_size, iterations=10)
        results["benchmarks"].append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("Phase III Benchmark Summary")
    print("=" * 70)

    for benchmark in results["benchmarks"]:
        print(f"\n{benchmark['benchmark'].upper()} ({benchmark.get('iteration_count', benchmark.get('node_count', benchmark.get('window_size', 'N/A')))}):")
        print(f"  Mean Time: {benchmark['timings_ms']['mean'] if 'mean' in benchmark['timings_ms'] else benchmark['timings_us']['mean']}{'ms' if 'mean' in benchmark['timings_ms'] else 'μs'}")
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
    filename = f"phase3_benchmark_{timestamp}.json"
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
    results = run_phase3_benchmarks()

    # Save results
    save_results(results)

    print("\n" + "=" * 70)
    print("Phase III Benchmarks Complete!")
    print("=" * 70)
