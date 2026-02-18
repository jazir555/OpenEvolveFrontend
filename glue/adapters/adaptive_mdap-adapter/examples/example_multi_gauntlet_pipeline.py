#!/usr/bin/env python3
"""
Example: Multi-Gauntlet Verification Pipeline

This example demonstrates how to create and execute a multi-gauntlet
pipeline for rigorous solution verification.

Usage:
    cd examples
    python example_multi_gauntlet_pipeline.py
"""

import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")

from src import get_advanced_gauntlet_integration, GauntletType, GauntletSeverity


def main():
    """Demonstrate multi-gauntlet pipeline."""
    print("=" * 70)
    print("  EXAMPLE: Multi-Gauntlet Verification Pipeline")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get advanced gauntlet integration
    gauntlet = get_advanced_gauntlet_integration()

    # Sample solution to verify
    solution = {
        "approach": "Distributed consensus using Raft",
        "description": "Implement Raft consensus algorithm for leader election",
        "code_snippets": ["def elect_leader(nodes): ...", "def append_log(entry): ..."],
        "test_coverage": 0.95,
        "performance_metrics": {"latency_p99_ms": 50, "throughput_qps": 10000}
    }

    print("Solution to Verify:")
    print("-" * 70)
    print(f"Approach: {solution['approach']}")
    print(f"Description: {solution['description']}")
    print(f"Test Coverage: {solution['test_coverage']:.1%}")
    print(f"Latency (P99): {solution['performance_metrics']['latency_p99_ms']}ms")
    print("-" * 70)

    # Create gauntlet pipeline
    print("\nCreating gauntlet pipeline...\n")

    pipeline = gauntlet.create_gauntlet_pipeline(
        complexity_score=0.85,  # High complexity
        base_gauntlet_type=GauntletType.ADVERSARIAL,
        include_cross_validation=True,
        severity=GauntletSeverity.HARDCORE
    )

    print(f"Pipeline Configuration:")
    print(f"  Total Gauntlets: {len(pipeline.gauntlets)}")
    print(f"  Execution Mode: {pipeline.execution_mode}")
    print(f"  Aggregation Strategy: {pipeline.aggregation_strategy}")

    print("\nGauntlets in Pipeline:")
    print("-" * 70)

    for i, g in enumerate(pipeline.gauntlets, 1):
        status_icon = "[REQUIRED]" if g.required else "[OPTIONAL]"
        print(f"\n{i}. {status_icon} {g.gauntlet_type.value}")
        print(f"   Severity: {g.severity.value}")
        print(f"   Description: {g.description[:80]}...")

    # Execute pipeline
    print("\n" + "=" * 70)
    print("Executing Gauntlet Pipeline")
    print("=" * 70)

    result = gauntlet.execute_pipeline(
        pipeline=pipeline,
        solution=solution,
        context={"domain": "distributed_systems", "workflow_type": "sovereign"}
    )

    print(f"\nExecution Complete")
    print(f"Total Gauntlets: {result.total_gauntlets}")
    print(f"Passed: {result.passed_gauntlets}")
    print(f"Failed: {result.failed_gauntlets}")
    print(f"Skipped: {result.skipped_gauntlets}")
    print(f"Overall Pass: {result.overall_pass}")
    print(f"Aggregate Score: {result.aggregate_score:.3f}")
    print(f"Execution Time: {result.execution_time_ms:.0f}ms")

    if result.gauntlet_results:
        print("\nDetailed Results:")
        print("-" * 70)

        for gr in result.gauntlet_results:
            status = "[PASS]" if gr.passed else "[FAIL]"
            print(f"\n{status} {gr.gauntlet_type}")
            print(f"  Score: {gr.score:.3f}")
            print(f"  Reasoning: {gr.reasoning[:80]}...")

            if gr.red_flags:
                print(f"  Red Flags: {len(gr.red_flags)}")
                for flag in gr.red_flags[:2]:
                    print(f"    - {flag}")

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Return exit code based on overall pass
    return 0 if result.overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
