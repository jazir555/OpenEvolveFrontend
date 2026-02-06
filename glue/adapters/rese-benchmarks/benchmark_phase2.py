#!/usr/bin/env python3
"""
RESE Phase II Performance Benchmarks

Benchmarks Phase II (Isomorphic Mapping) operations:
- Isomorphic mapping computation time
- I_mech score calculation performance
- Cross-domain pattern matching speed

Tests with varying target domain counts:
- Small: 1 target domain
- Medium: 5 target domains
- Large: 10 target domains

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
from collections import defaultdict

# Add phase2 adapter to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase2" / "src"))

try:
    from phase2_executor import (
        IsomorphicMappingExecutor,
        Phase2Config,
        FunctionalDependencyGraph,
        FunctionalDependency,
    )
except ImportError as e:
    print(f"Error importing Phase II executor: {e}")
    print("Make sure phase2_executor.py is available")
    sys.exit(1)


# ============================================================================
# BENCHMARK DATA GENERATORS
# ============================================================================

def generate_fdg(domain: str, node_count: int = 20) -> FunctionalDependencyGraph:
    """Generate a synthetic Functional Dependency Graph.

    Args:
        domain: Domain name
        node_count: Number of nodes in the graph

    Returns:
        FunctionalDependencyGraph
    """
    nodes = [f"node_{i}_{domain}" for i in range(node_count)]

    dependencies = []
    adjacency_list = {node: [] for node in nodes}

    # Create random dependencies (avoid cycles for simplicity)
    for i in range(len(nodes) - 1):
        source = nodes[i]
        target = nodes[i + 1]
        dep = FunctionalDependency(
            source=source,
            target=target,
            relationship_type="causal",
            strength=0.7,
            domain=domain
        )
        dependencies.append(dep)
        adjacency_list[source].append(target)

    return FunctionalDependencyGraph(
        domain=domain,
        nodes=nodes,
        dependencies=dependencies,
        adjacency_list=adjacency_list
    )


def generate_target_fdgs(count: int) -> List[FunctionalDependencyGraph]:
    """Generate multiple target domain FDGs.

    Args:
        count: Number of target domains

    Returns:
        List of FunctionalDependencyGraphs
    """
    domains = ["physics", "biology", "economics", "computer_science",
               "chemistry", "mathematics", "engineering", "medicine",
               "psychology", "sociology"]

    fdgs = []
    for i in range(min(count, len(domains))):
        fdg = generate_fdg(domains[i], node_count=15 + i * 2)
        fdgs.append(fdg)

    return fdgs


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_isomorphic_mapping(
    executor: IsomorphicMappingExecutor,
    target_domain_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark isomorphic mapping computation.

    Measures:
    - Time to find isomorphic mappings
    - Number of mappings found
    - Mappings per second throughput

    Args:
        executor: Phase II executor instance
        target_domain_count: Number of target domains
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Isomorphic Mapping ({target_domain_count} target domains)...")

    # Generate source and target FDGs
    source_fdg = generate_fdg("source", node_count=20)
    target_fdgs = generate_target_fdgs(target_domain_count)

    timings_ms = []
    mapping_counts = []

    for i in range(iterations):
        start = time.perf_counter()

        # Find isomorphic mappings
        mappings = executor.cross_domain_mapper.find_isomorphic_mappings(
            source_fdg=source_fdg,
            target_fdgs=target_fdgs
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)
        mapping_counts.append(len(mappings))

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({len(mappings)} mappings)")

    # Calculate statistics
    avg_mappings = statistics.mean(mapping_counts)
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "isomorphic_mapping",
        "target_domains": target_domain_count,
        "iterations": iterations,
        "mappings_found": {
            "min": min(mapping_counts),
            "max": max(mapping_counts),
            "mean": round(avg_mappings, 1),
        },
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "mappings_per_second": round(avg_mappings / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


def benchmark_imech_score(
    executor: IsomorphicMappingExecutor,
    node_count: int,
    iterations: int = 100
) -> Dict[str, Any]:
    """Benchmark I_mech score calculation.

    Measures:
    - Time to calculate I_mech score
    - Scores per second throughput

    Args:
        executor: Phase II executor instance
        node_count: Number of nodes in FDGs
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking I_mech Score Calculation (node_count={node_count})...")

    # Generate two FDGs
    fdg1 = generate_fdg("domain1", node_count=node_count)
    fdg2 = generate_fdg("domain2", node_count=node_count)

    timings_us = []  # Microseconds for more precision

    for i in range(iterations):
        start = time.perf_counter()

        # Calculate I_mech score
        score = executor.cross_domain_mapper.compute_imech_score(
            source_fdg=fdg1,
            target_fdg=fdg2
        )

        elapsed_us = (time.perf_counter() - start) * 1_000_000
        timings_us.append(elapsed_us)

        if i == 0 or (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}: {elapsed_us:.2f}μs (score={score:.3f})")

    # Calculate statistics
    avg_time_us = statistics.mean(timings_us)
    avg_time_sec = avg_time_us / 1_000_000

    results = {
        "benchmark": "imech_score_calculation",
        "node_count": node_count,
        "iterations": iterations,
        "timings_us": {
            "min": round(min(timings_us), 2),
            "max": round(max(timings_us), 2),
            "mean": round(avg_time_us, 2),
            "median": round(statistics.median(timings_us), 2),
            "stdev": round(statistics.stdev(timings_us), 2) if len(timings_us) > 1 else 0.0,
        },
        "throughput": {
            "scores_per_second": round(1.0 / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


def benchmark_cross_domain_patterns(
    executor: IsomorphicMappingExecutor,
    target_domain_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark cross-domain pattern matching.

    Measures:
    - Time to identify cross-domain patterns
    - Patterns found per second

    Args:
        executor: Phase II executor instance
        target_domain_count: Number of target domains
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Cross-Domain Pattern Matching ({target_domain_count} domains)...")

    # Generate source and target FDGs
    source_fdg = generate_fdg("source", node_count=30)
    target_fdgs = generate_target_fdgs(target_domain_count)

    timings_ms = []
    pattern_counts = []

    for i in range(iterations):
        start = time.perf_counter()

        # Identify cross-domain patterns
        patterns = executor._identify_cross_domain_patterns(
            source_fdg=source_fdg,
            target_fdgs=target_fdgs
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)
        pattern_counts.append(len(patterns))

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({len(patterns)} patterns)")

    # Calculate statistics
    avg_patterns = statistics.mean(pattern_counts)
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "cross_domain_pattern_matching",
        "target_domains": target_domain_count,
        "iterations": iterations,
        "patterns_found": {
            "min": min(pattern_counts),
            "max": max(pattern_counts),
            "mean": round(avg_patterns, 1),
        },
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "patterns_per_second": round(avg_patterns / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_phase2_benchmarks() -> Dict[str, Any]:
    """Run all Phase II benchmarks.

    Returns:
        Complete benchmark results
    """
    print("=" * 70)
    print("RESE Phase II Performance Benchmarks")
    print("=" * 70)

    # Initialize executor with default config
    config = Phase2Config.from_env()
    executor = IsomorphicMappingExecutor(config)

    results = {
        "phase": "phase2_isomorphic_mapping",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": []
    }

    # Benchmark 1: Isomorphic Mapping
    print("\n" + "=" * 70)
    print("Benchmark Suite 1: Isomorphic Mapping")
    print("=" * 70)

    for domain_count in [1, 5, 10]:
        result = benchmark_isomorphic_mapping(executor, domain_count, iterations=5)
        results["benchmarks"].append(result)

    # Benchmark 2: I_mech Score Calculation
    print("\n" + "=" * 70)
    print("Benchmark Suite 2: I_mech Score Calculation")
    print("=" * 70)

    for node_count in [10, 20, 50]:
        result = benchmark_imech_score(executor, node_count, iterations=100)
        results["benchmarks"].append(result)

    # Benchmark 3: Cross-Domain Pattern Matching
    print("\n" + "=" * 70)
    print("Benchmark Suite 3: Cross-Domain Pattern Matching")
    print("=" * 70)

    for domain_count in [1, 5, 10]:
        result = benchmark_cross_domain_patterns(executor, domain_count, iterations=5)
        results["benchmarks"].append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("Phase II Benchmark Summary")
    print("=" * 70)

    for benchmark in results["benchmarks"]:
        print(f"\n{benchmark['benchmark'].upper()} ({benchmark.get('target_domains', benchmark.get('node_count', 'N/A'))}):")
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
    filename = f"phase2_benchmark_{timestamp}.json"
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
    results = run_phase2_benchmarks()

    # Save results
    save_results(results)

    print("\n" + "=" * 70)
    print("Phase II Benchmarks Complete!")
    print("=" * 70)
