#!/usr/bin/env python3
"""
RESE Phase IV Performance Benchmarks

Benchmarks Phase IV (Architecture Assembly) operations:
- Architecture assembly time
- Knowledge integration speed
- Validation processing time

Tests with varying paradigm shift counts:
- Small: 1 paradigm shift
- Medium: 10 paradigm shifts
- Large: 100 paradigm shifts

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

# Add phase4 adapter to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rese-phase4" / "src"))

try:
    from phase4_executor import (
        ArchitectureAssemblyExecutor,
        ParadigmShiftAssembler,
        KnowledgeIntegrator,
        ArchitectureValidator,
        Phase4Config,
        ParadigmShift,
        SynthesizedKnowledge,
        ValidationLevel,
        IntegrationStrategy,
    )
except ImportError as e:
    print(f"Error importing Phase IV executor: {e}")
    print("Make sure phase4_executor.py is available")
    sys.exit(1)


# ============================================================================
# BENCHMARK DATA GENERATORS
# ============================================================================

def generate_patterns(count: int, phase: int) -> List[Dict[str, Any]]:
    """Generate synthetic patterns for testing.

    Args:
        count: Number of patterns to generate
        phase: Phase number (1, 2, or 3)

    Returns:
        List of pattern dictionaries
    """
    patterns = []
    for i in range(count):
        patterns.append({
            "pattern_id": str(uuid.uuid4()),
            "type": ["structural", "functional", "behavioral"][i % 3],
            "source_phase": phase,
            "confidence": 0.5 + (i % 50) / 100.0,
            "description": f"Pattern {i} from Phase {phase}",
            "transformation_rules": [
                {"rule_id": str(uuid.uuid4()), "type": "mapping"},
            ],
        })
    return patterns


def generate_paradigm_shifts(count: int) -> List[ParadigmShift]:
    """Generate synthetic paradigm shifts for testing.

    Args:
        count: Number of paradigm shifts to generate

    Returns:
        List of ParadigmShift objects
    """
    shifts = []
    for i in range(count):
        shift = ParadigmShift(
            shift_id=str(uuid.uuid4()),
            shift_type=["structural", "functional", "ontological"][i % 3],
            description=f"Paradigm shift {i}",
            source_patterns=[str(uuid.uuid4()) for _ in range(3)],
            phase1_contributions=generate_patterns(1, 1) if i % 2 == 0 else [],
            phase2_contributions=generate_patterns(1, 2) if i % 3 == 0 else [],
            phase3_contributions=generate_patterns(1, 3),
            transformation_rules=[],
            confidence=0.6 + (i % 40) / 100.0,
            validation_status="pending",
        )
        shifts.append(shift)
    return shifts


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

def benchmark_architecture_assembly(
    assembler: ParadigmShiftAssembler,
    paradigm_shift_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark architecture assembly time.

    Measures:
    - Time to assemble paradigm shifts
    - Paradigm shifts per second

    Args:
        assembler: ParadigmShiftAssembler instance
        paradigm_shift_count: Number of paradigm shifts to assemble
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Architecture Assembly ({paradigm_shift_count} shifts)...")

    # Generate patterns
    phase1_patterns = generate_patterns(paradigm_shift_count, 1)
    phase2_patterns = generate_patterns(paradigm_shift_count, 2)
    phase3_patterns = generate_patterns(paradigm_shift_count, 3)

    timings_ms = []
    shift_counts = []

    for i in range(iterations):
        start = time.perf_counter()

        # Assemble paradigm shifts
        shifts = assembler.assemble(
            phase1_patterns=phase1_patterns,
            phase2_patterns=phase2_patterns,
            phase3_patterns=phase3_patterns
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)
        shift_counts.append(len(shifts))

        print(f"  Iteration {i+1}: {elapsed:.2f}ms ({len(shifts)} shifts)")

    # Calculate statistics
    avg_shifts = statistics.mean(shift_counts)
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "architecture_assembly",
        "paradigm_shifts": paradigm_shift_count,
        "iterations": iterations,
        "shifts_assembled": {
            "min": min(shift_counts),
            "max": max(shift_counts),
            "mean": round(avg_shifts, 1),
        },
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "shifts_per_second": round(avg_shifts / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


def benchmark_knowledge_integration(
    integrator: KnowledgeIntegrator,
    paradigm_shift_count: int,
    iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark knowledge integration speed.

    Measures:
    - Time to integrate knowledge
    - Integration operations per second

    Args:
        integrator: KnowledgeIntegrator instance
        paradigm_shift_count: Number of paradigm shifts
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Knowledge Integration ({paradigm_shift_count} shifts)...")

    # Generate paradigm shifts
    paradigm_shifts = generate_paradigm_shifts(paradigm_shift_count)

    timings_ms = []

    for i in range(iterations):
        start = time.perf_counter()

        # Integrate knowledge
        knowledge = integrator.integrate(
            phase1_result=None,  # Simplified for benchmark
            phase2_result=None,
            phase3_result=None,
            paradigm_shifts=paradigm_shifts
        )

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)

        print(f"  Iteration {i+1}: {elapsed:.2f}ms (confidence={knowledge.confidence:.2f})")

    # Calculate statistics
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "knowledge_integration",
        "paradigm_shifts": paradigm_shift_count,
        "iterations": iterations,
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "integrations_per_second": round(1.0 / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


def benchmark_validation_processing(
    validator: ArchitectureValidator,
    paradigm_shift_count: int,
    iterations: int = 10
) -> Dict[str, Any]:
    """Benchmark validation processing time.

    Measures:
    - Time to validate assembly
    - Validations per second

    Args:
        validator: ArchitectureValidator instance
        paradigm_shift_count: Number of paradigm shifts in assembly
        iterations: Number of benchmark iterations

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking Validation Processing ({paradigm_shift_count} shifts)...")

    # Generate a mock assembly
    from phase4_executor import ArchitectureAssembly, AssemblyStatus

    paradigm_shifts = generate_paradigm_shifts(paradigm_shift_count)

    knowledge = SynthesizedKnowledge(
        knowledge_id=str(uuid.uuid4()),
        knowledge_type="benchmark",
        description="Benchmark knowledge",
        source_phase1=None,
        source_phase2=None,
        source_phase3=None,
        paradigm_shifts=paradigm_shifts,
        integration_strategy=IntegrationStrategy.MERGED,
        synthesis_rules=[],
        confidence=0.8,
        completeness=1.0,
        consistency=0.9,
    )

    assembly = ArchitectureAssembly(
        assembly_id=str(uuid.uuid4()),
        synthesized_knowledge=knowledge,
        paradigm_shifts=paradigm_shifts,
        final_architecture={},
        aci_reduction_achieved=0.25,
        confidence=0.8,
        validation_level=ValidationLevel.STANDARD,
        status=AssemblyStatus.VALIDATED,
    )

    timings_ms = []

    for i in range(iterations):
        start = time.perf_counter()

        # Validate assembly
        is_valid, validation_results = validator.validate(assembly)

        elapsed = (time.perf_counter() - start) * 1000
        timings_ms.append(elapsed)

        if i == 0 or (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}: {elapsed:.2f}ms (valid={is_valid}, checks={len(validation_results)})")

    # Calculate statistics
    avg_time_sec = statistics.mean(timings_ms) / 1000

    results = {
        "benchmark": "validation_processing",
        "paradigm_shifts": paradigm_shift_count,
        "iterations": iterations,
        "timings_ms": {
            "min": round(min(timings_ms), 2),
            "max": round(max(timings_ms), 2),
            "mean": round(statistics.mean(timings_ms), 2),
            "median": round(statistics.median(timings_ms), 2),
            "stdev": round(statistics.stdev(timings_ms), 2) if len(timings_ms) > 1 else 0.0,
        },
        "throughput": {
            "validations_per_second": round(1.0 / avg_time_sec, 2) if avg_time_sec > 0 else 0.0
        }
    }

    return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_phase4_benchmarks() -> Dict[str, Any]:
    """Run all Phase IV benchmarks.

    Returns:
        Complete benchmark results
    """
    print("=" * 70)
    print("RESE Phase IV Performance Benchmarks")
    print("=" * 70)

    # Initialize executor with default config
    config = Phase4Config.from_env()
    from phase4_executor import StructuredLogger

    logger = StructuredLogger("benchmark")
    assembler = ParadigmShiftAssembler(config, logger)
    integrator = KnowledgeIntegrator(config, logger)
    validator = ArchitectureValidator(config, logger)

    results = {
        "phase": "phase4_architecture_assembly",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": []
    }

    # Benchmark 1: Architecture Assembly
    print("\n" + "=" * 70)
    print("Benchmark Suite 1: Architecture Assembly")
    print("=" * 70)

    for shift_count in [1, 10, 100]:
        result = benchmark_architecture_assembly(assembler, shift_count, iterations=5)
        results["benchmarks"].append(result)

    # Benchmark 2: Knowledge Integration
    print("\n" + "=" * 70)
    print("Benchmark Suite 2: Knowledge Integration")
    print("=" * 70)

    for shift_count in [1, 10, 100]:
        result = benchmark_knowledge_integration(integrator, shift_count, iterations=5)
        results["benchmarks"].append(result)

    # Benchmark 3: Validation Processing
    print("\n" + "=" * 70)
    print("Benchmark Suite 3: Validation Processing")
    print("=" * 70)

    for shift_count in [1, 10, 100]:
        result = benchmark_validation_processing(validator, shift_count, iterations=10)
        results["benchmarks"].append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("Phase IV Benchmark Summary")
    print("=" * 70)

    for benchmark in results["benchmarks"]:
        print(f"\n{benchmark['benchmark'].upper()} ({benchmark.get('paradigm_shifts', 'N/A')}):")
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
    filename = f"phase4_benchmark_{timestamp}.json"
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
    results = run_phase4_benchmarks()

    # Save results
    save_results(results)

    print("\n" + "=" * 70)
    print("Phase IV Benchmarks Complete!")
    print("=" * 70)
