#!/usr/bin/env python3
"""
Example: Advanced Problem Decomposition

This example demonstrates the advanced OpenEvolve integration's ability
to decompose complex problems into manageable sub-problems.

Usage:
    cd examples
    python example_advanced_decomposition.py
"""

import os
import sys
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# Set environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")

from src import get_advanced_openevolve_integration


def main():
    """Demonstrate advanced problem decomposition."""
    print("=" * 70)
    print("  EXAMPLE: Advanced Problem Decomposition")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")

    # Get advanced integration
    advanced = get_advanced_openevolve_integration()

    # Complex problem to decompose
    problem_statement = """
    Design and implement a globally distributed, fault-tolerant database system
    with the following requirements:
    - Multi-region active-active replication
    - Consistent hashing for data distribution
    - Automatic failover and recovery
    - Strong consistency with low latency
    - Horizontal scalability to petabytes
    """

    print("Problem Statement:")
    print("-" * 70)
    print(problem_statement.strip())
    print("-" * 70)

    # Decompose the problem
    print("\nDecomposing problem...\n")

    decomposition = advanced.decompose_problem(
        workflow_id="decomposition_example",
        problem_statement=problem_statement,
        workflow_type="sovereign",
        max_depth=3
    )

    print(f"Decomposition Strategy: {decomposition.decomposition_strategy}")
    print(f"Total Sub-Problems: {len(decomposition.sub_problems)}")
    print(f"Recommended Parallelization: {decomposition.recommended_parallelization}")

    print("\nSub-Problems:")
    print("-" * 70)

    for i, sub in enumerate(decomposition.sub_problems, 1):
        print(f"\n{i}. {sub['description'][:80]}...")
        print(f"   Complexity: {sub['complexity']:.3f}")
        print(f"   Estimated Effort: {sub.get('estimated_effort', 'N/A')}")
        print(f"   Dependencies: {len(sub.get('dependencies', []))}")

    # Team selection
    print("\n" + "=" * 70)
    print("Team Selection")
    print("=" * 70)

    team_selection = advanced.select_teams_for_stage(
        workflow_id="decomposition_example",
        stage="solving",
        workflow_type="sovereign",
        complexity_score=0.8
    )

    print(f"\nStage: {team_selection.stage}")
    print(f"Workflow Type: {team_selection.workflow_type}")
    print(f"Complexity Score: {team_selection.complexity_score:.3f}")
    print(f"Estimated Cost: ${team_selection.estimated_cost:.2f}")

    print("\nRecommended Teams:")
    for team_name, team_info in team_selection.recommended_teams.items():
        print(f"  - {team_name}: {team_info['agents']} agents")
        print(f"    Reasoning: {team_info['reasoning']}")

    # Resource optimization
    print("\n" + "=" * 70)
    print("Resource Optimization")
    print("=" * 70)

    optimization = advanced.optimize_resources(
        workflow_id="decomposition_example",
        stage="solving",
        complexity_score=0.8,
        estimated_duration_ms=300000
    )

    print(f"\nCPU Allocation: {optimization.cpu_allocation} cores")
    print(f"Memory Allocation: {optimization.memory_allocation_mb}MB")
    print(f"Timeout: {optimization.timeout_ms}ms")
    print(f"Cost Savings: {optimization.estimated_cost_savings:.1%}")

    print("\n" + "=" * 70)
    print("  EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nEnd Time: {datetime.now(timezone.utc).isoformat()}\n")


if __name__ == "__main__":
    main()
