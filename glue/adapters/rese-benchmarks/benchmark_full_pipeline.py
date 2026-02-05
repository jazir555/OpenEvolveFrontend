#!/usr/bin/env python3
"""
RESE Full Pipeline End-to-End Benchmark

Benchmarks the complete RESE pipeline:
- Total pipeline execution time
- Per-phase timing breakdown
- Memory usage per phase
- End-to-end throughput

Tests with three problem complexities:
- Simple: Small problem, few assumptions, limited iterations
- Medium: Moderate problem, balanced parameters
- Complex: Large problem, many assumptions, extensive iterations

Outputs: Comprehensive JSON report with all metrics

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
import json
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add all phase adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase1" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase2" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase3" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase4" / "src"))

try:
    from phase1_executor import (
        EpistemicAuditExecutor,
        Phase1Config,
    )
    from phase2_executor import (
        IsomorphicMappingExecutor,
        Phase2Config,
    )
    from phase3_executor import (
        MCTSSearchExecutor,
        Phase3Config,
        Hypothesis,
    )
    from phase4_executor import (
        ArchitectureAssemblyExecutor,
        Phase4Config,
    )
except ImportError as e:
    print(f"Error importing RESE executors: {e}")
    print("Make sure all phase executors are available")
    sys.exit(1)


# ============================================================================
# PROBLEM COMPLEXITY CONFIGURATIONS
# ============================================================================

PROBLEM_CONFIGS = {
    "simple": {
        "phase1": {
            "assumption_count": 10,
            "constraint_complexity": "small",
        },
        "phase2": {
            "target_domains": 2,
            "node_count": 10,
        },
        "phase3": {
            "iterations": 100,
            "max_depth": 10,
        },
        "phase4": {
            "paradigm_shifts": 5,
        },
    },
    "medium": {
        "phase1": {
            "assumption_count": 100,
            "constraint_complexity": "medium",
        },
        "phase2": {
            "target_domains": 5,
            "node_count": 20,
        },
        "phase3": {
            "iterations": 1000,
            "max_depth": 15,
        },
        "phase4": {
            "paradigm_shifts": 20,
        },
    },
    "complex": {
        "phase1": {
            "assumption_count": 1000,
            "constraint_complexity": "large",
        },
        "phase2": {
            "target_domains": 10,
            "node_count": 50,
        },
        "phase3": {
            "iterations": 5000,
            "max_depth": 20,
        },
        "phase4": {
            "paradigm_shifts": 50,
        },
    },
}


# ============================================================================
# DATA GENERATORS
# ============================================================================

def generate_failure_patterns(count: int) -> List[Dict[str, Any]]:
    """Generate failure patterns for Phase I."""
    return [
        {
            "pattern_id": str(uuid.uuid4()),
            "pattern_description": f"Failure pattern {i}",
            "failure_rate": 0.5 + (i % 50) / 100.0,
            "data_points": 100 + i * 10,
        }
        for i in range(count)
    ]


def generate_problem_description(complexity: str) -> str:
    """Generate problem description based on complexity."""
    multipliers = {"simple": 1, "medium": 5, "large": 20}
    mult = multipliers.get(complexity, 5)

    base = """
    Material science problem: It is impossible to strengthen lattice defects beyond
    the theoretical limit when loading ratio exceeds critical threshold.
    """

    return (base + "\n") * mult


def generate_hypothesis(statement: str) -> Hypothesis:
    """Generate a hypothesis for Phase III."""
    import random
    return Hypothesis(
        hypothesis_id=str(uuid.uuid4()),
        statement=statement,
        confidence=random.uniform(0.5, 0.9),
        status="validated",
        evidence_count=random.randint(5, 20),
        metadata={"generated": True}
    )


# ============================================================================
# MEMORY TRACKING
# ============================================================================

class MemoryTracker:
    """Context manager for tracking memory usage."""

    def __init__(self):
        self.peak_memory_mb = 0.0

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory_mb = peak / 1024 / 1024  # Convert to MB
        tracemalloc.stop()


# ============================================================================
# PHASE EXECUTORS
# ============================================================================

def execute_phase1(
    config: Dict[str, Any],
    problem_description: str,
    failure_patterns: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], float, float]:
    """Execute Phase I: Epistemic Audit.

    Returns:
        Tuple of (result, execution_time_ms, memory_mb)
    """
    # Configure
    phase1_config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(phase1_config)

    # Execute with memory tracking
    with MemoryTracker() as mem_tracker:
        start = time.perf_counter()

        result = executor.perform_audit(
            problem_description=problem_description,
            failure_patterns=failure_patterns,
            correlation_id=str(uuid.uuid4()),
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

    return result.to_dict(), elapsed_ms, mem_tracker.peak_memory_mb


def execute_phase2(
    config: Dict[str, Any],
    phase1_result: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Execute Phase II: Isomorphic Mapping.

    Returns:
        Tuple of (result, execution_time_ms, memory_mb)
    """
    # Configure
    phase2_config = Phase2Config.from_env()
    executor = IsomorphicMappingExecutor(phase2_config)

    # Execute with memory tracking
    with MemoryTracker() as mem_tracker:
        start = time.perf_counter()

        result = executor.execute_phase2(
            source_domain="materials_science",
            problem_description=phase1_result.get("problem_description", ""),
            target_domains=None,  # Use defaults
            constraints=None,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

    # Convert to dict (using to_dict if available, or vars)
    if hasattr(result, 'to_dict'):
        result_dict = result.to_dict()
    else:
        result_dict = vars(result)

    return result_dict, elapsed_ms, mem_tracker.peak_memory_mb


def execute_phase3(
    config: Dict[str, Any],
    phase2_result: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Execute Phase III: MCTS Search.

    Returns:
        Tuple of (result, execution_time_ms, memory_mb)
    """
    import random

    # Configure
    phase3_config = Phase3Config.from_env()
    phase3_config.iterations = config["iterations"]
    phase3_config.max_depth = config["max_depth"]

    executor = MCTSSearchExecutor(config=phase3_config, logger=None)

    # Create root hypothesis
    root_hypothesis = generate_hypothesis("Root hypothesis from Phase II")

    # Create simple generators
    def hypothesis_generator():
        return [
            generate_hypothesis(f"Child hypothesis {i}")
            for i in range(3)
        ]

    def reward_function(hypothesis):
        return hypothesis.confidence + random.uniform(-0.05, 0.05)

    # Execute with memory tracking
    with MemoryTracker() as mem_tracker:
        start = time.perf_counter()

        result, error = executor.execute_search(
            root_hypothesis=root_hypothesis,
            hypothesis_generator=hypothesis_generator,
            reward_function=reward_function
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

    # Convert to dict
    result_dict = {
        "search_id": result.search_id if result else None,
        "best_hypothesis": {
            "hypothesis_id": result.best_hypothesis.hypothesis_id if result and result.best_hypothesis else None,
            "confidence": result.best_hypothesis.confidence if result and result.best_hypothesis else 0.0,
        } if result and result.best_hypothesis else None,
        "iterations": result.iterations if result else 0,
        "total_nodes": result.total_nodes if result else 0,
        "max_depth": result.max_depth if result else 0,
        "convergence_reached": result.convergence_reached if result else False,
        "execution_time_ms": result.execution_time_ms if result else 0.0,
    }

    return result_dict, elapsed_ms, mem_tracker.peak_memory_mb


def execute_phase4(
    config: Dict[str, Any],
    phase1_result: Dict[str, Any],
    phase2_result: Dict[str, Any],
    phase3_result: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """Execute Phase IV: Architecture Assembly.

    Returns:
        Tuple of (result, execution_time_ms, memory_mb)
    """
    # Configure
    phase4_config = Phase4Config.from_env()
    executor = ArchitectureAssemblyExecutor(phase4_config)

    # Generate patterns
    shift_count = config.get("paradigm_shifts", 10)
    phase1_patterns = [{"pattern_id": str(uuid.uuid4()), "type": "structural", "confidence": 0.7} for _ in range(shift_count)]
    phase2_patterns = [{"pattern_id": str(uuid.uuid4()), "type": "functional", "confidence": 0.75} for _ in range(shift_count)]
    phase3_patterns = [{"pattern_id": str(uuid.uuid4()), "type": "behavioral", "confidence": 0.8} for _ in range(shift_count)]

    # Execute with memory tracking
    with MemoryTracker() as mem_tracker:
        start = time.perf_counter()

        assembly = executor.execute(
            phase1_result=phase1_result,
            phase2_result=phase2_result,
            phase3_result=phase3_result,
            phase1_patterns=phase1_patterns,
            phase2_patterns=phase2_patterns,
            phase3_patterns=phase3_patterns,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

    # Convert to dict
    result_dict = {
        "assembly_id": assembly.assembly_id,
        "confidence": assembly.confidence,
        "paradigm_shifts_count": len(assembly.paradigm_shifts),
        "aci_reduction": assembly.aci_reduction_achieved,
        "status": assembly.status.value if hasattr(assembly.status, 'value') else str(assembly.status),
    }

    return result_dict, elapsed_ms, mem_tracker.peak_memory_mb


# ============================================================================
# FULL PIPELINE BENCHMARK
# ============================================================================

def benchmark_full_pipeline(
    complexity: str,
    iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark the full RESE pipeline.

    Args:
        complexity: Problem complexity ("simple", "medium", "complex")
        iterations: Number of benchmark iterations

    Returns:
        Complete benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking Full Pipeline ({complexity.upper()} complexity)")
    print('=' * 70)

    config = PROBLEM_CONFIGS[complexity]
    all_results = []

    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---")

        # Generate data
        problem_description = generate_problem_description(
            config["phase1"]["constraint_complexity"]
        )
        failure_patterns = generate_failure_patterns(
            config["phase1"]["assumption_count"]
        )

        # Execute pipeline
        pipeline_timings = []
        phase_results = {}

        # Phase I
        print("\n[Phase I] Epistemic Audit...")
        p1_result, p1_time, p1_mem = execute_phase1(
            config["phase1"],
            problem_description,
            failure_patterns
        )
        phase_results["phase1"] = p1_result
        pipeline_timings.append({
            "phase": "phase1",
            "time_ms": p1_time,
            "memory_mb": p1_mem,
        })
        print(f"  Completed in {p1_time:.2f}ms, {p1_mem:.2f}MB")

        # Phase II
        print("\n[Phase II] Isomorphic Mapping...")
        p2_result, p2_time, p2_mem = execute_phase2(
            config["phase2"],
            p1_result
        )
        phase_results["phase2"] = p2_result
        pipeline_timings.append({
            "phase": "phase2",
            "time_ms": p2_time,
            "memory_mb": p2_mem,
        })
        print(f"  Completed in {p2_time:.2f}ms, {p2_mem:.2f}MB")

        # Phase III
        print("\n[Phase III] MCTS Search...")
        p3_result, p3_time, p3_mem = execute_phase3(
            config["phase3"],
            p2_result
        )
        phase_results["phase3"] = p3_result
        pipeline_timings.append({
            "phase": "phase3",
            "time_ms": p3_time,
            "memory_mb": p3_mem,
        })
        print(f"  Completed in {p3_time:.2f}ms, {p3_mem:.2f}MB")

        # Phase IV
        print("\n[Phase IV] Architecture Assembly...")
        p4_result, p4_time, p4_mem = execute_phase4(
            config["phase4"],
            p1_result,
            p2_result,
            p3_result
        )
        phase_results["phase4"] = p4_result
        pipeline_timings.append({
            "phase": "phase4",
            "time_ms": p4_time,
            "memory_mb": p4_mem,
        })
        print(f"  Completed in {p4_time:.2f}ms, {p4_mem:.2f}MB")

        # Calculate totals
        total_time = sum(t["time_ms"] for t in pipeline_timings)
        max_memory = max(t["memory_mb"] for t in pipeline_timings)

        print(f"\nTotal Pipeline Time: {total_time:.2f}ms")
        print(f"Peak Memory Usage: {max_memory:.2f}MB")

        all_results.append({
            "iteration": i + 1,
            "phase_timings": pipeline_timings,
            "total_time_ms": total_time,
            "peak_memory_mb": max_memory,
        })

    # Calculate aggregate statistics
    total_times = [r["total_time_ms"] for r in all_results]
    peak_memories = [r["peak_memory_mb"] for r in all_results]

    # Phase-wise aggregates
    phase_stats = {}
    for phase in ["phase1", "phase2", "phase3", "phase4"]:
        phase_times = [
            next(t["time_ms"] for t in r["phase_timings"] if t["phase"] == phase)
            for r in all_results
        ]
        phase_memories = [
            next(t["memory_mb"] for t in r["phase_timings"] if t["phase"] == phase)
            for r in all_results
        ]

        phase_stats[phase] = {
            "time_ms": {
                "min": round(min(phase_times), 2),
                "max": round(max(phase_times), 2),
                "mean": round(sum(phase_times) / len(phase_times), 2),
                "total": round(sum(phase_times), 2),
            },
            "memory_mb": {
                "min": round(min(phase_memories), 2),
                "max": round(max(phase_memories), 2),
                "mean": round(sum(phase_memories) / len(phase_memories), 2),
            },
        }

    results = {
        "benchmark": "full_pipeline",
        "complexity": complexity,
        "iterations": iterations,
        "config": config,
        "aggregate_statistics": {
            "total_time_ms": {
                "min": round(min(total_times), 2),
                "max": round(max(total_times), 2),
                "mean": round(sum(total_times) / len(total_times), 2),
            },
            "peak_memory_mb": {
                "min": round(min(peak_memories), 2),
                "max": round(max(peak_memories), 2),
                "mean": round(sum(peak_memories) / len(peak_memories), 2),
            },
        },
        "phase_statistics": phase_stats,
        "detailed_results": all_results,
    }

    return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_full_pipeline_benchmarks() -> Dict[str, Any]:
    """Run full pipeline benchmarks for all complexities.

    Returns:
        Complete benchmark results
    """
    print("=" * 70)
    print("RESE Full Pipeline End-to-End Benchmarks")
    print("=" * 70)

    results = {
        "benchmark_suite": "full_pipeline_end_to_end",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": []
    }

    # Run benchmarks for each complexity
    for complexity in ["simple", "medium", "complex"]:
        result = benchmark_full_pipeline(complexity, iterations=3)
        results["benchmarks"].append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("Full Pipeline Benchmark Summary")
    print("=" * 70)

    for benchmark in results["benchmarks"]:
        complexity = benchmark["complexity"].upper()
        total_time = benchmark["aggregate_statistics"]["total_time_ms"]["mean"]
        peak_mem = benchmark["aggregate_statistics"]["peak_memory_mb"]["mean"]

        print(f"\n{complexity} Complexity:")
        print(f"  Total Time: {total_time:.2f}ms")
        print(f"  Peak Memory: {peak_mem:.2f}MB")

        print("\n  Phase Breakdown:")
        for phase, stats in benchmark["phase_statistics"].items():
            print(f"    {phase}: {stats['time_ms']['mean']:.2f}ms, {stats['memory_mb']['mean']:.2f}MB")

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
    filename = f"full_pipeline_benchmark_{timestamp}.json"
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
    results = run_full_pipeline_benchmarks()

    # Save results
    save_results(results)

    print("\n" + "=" * 70)
    print("Full Pipeline Benchmarks Complete!")
    print("=" * 70)
