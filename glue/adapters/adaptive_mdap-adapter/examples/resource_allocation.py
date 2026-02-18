"""
Example: Resource Allocation Based on Complexity

Demonstrates how to use the Adaptive MDAP Adapter to allocate
computational resources based on problem complexity.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    CanonicalComplexityScore,
    TaskStatus,
    AdaptiveMDAPAdapterConfig
)


def main():
    """Run resource allocation example."""

    # Get adapter instance
    adapter = get_adapter()

    print("=" * 60)
    print("Adaptive MDAP Adapter - Resource Allocation")
    print("=" * 60)
    print()

    # Define problems with different complexities
    problems = [
        {
            "name": "Simple Task",
            "description": "Write a hello world function",
            "domain": "programming",
            "depth": 1,
            "dependencies": [],
        },
        {
            "name": "Medium Task",
            "description": "Build a web scraper with error handling and data export",
            "domain": "data_engineering",
            "depth": 3,
            "dependencies=["html-parser", "storage", "error-handler"],
        },
        {
            "name": "Complex Task",
            "description": "Design a machine learning pipeline with feature engineering, model training, hyperparameter optimization, and production deployment",
            "domain": "machine_learning",
            "depth": 5,
            "dependencies=["data-ingestion", "feature-store", "training", "deployment", "monitoring"],
        },
    ]

    for problem_info in problems:
        print(f"Problem: {problem_info['name']}")
        print("-" * 60)

        # Create subproblem
        subproblem = CanonicalSubProblem(
            id=f"task-{problem_info['name'].lower().replace(' ', '-')}",
            description=problem_info['description'],
            domain=problem_info['domain'],
            depth=problem_info['depth'],
            dependencies=problem_info['dependencies'],
        )

        # Analyze complexity
        analysis_response = adapter.analyze_complexity(subproblem)

        if analysis_response.status != TaskStatus.COMPLETED:
            print(f"✗ Complexity analysis failed: {analysis_response.error}")
            print()
            continue

        # Get resource allocation based on complexity
        allocation_response = adapter.allocate_resources(
            complexity_score=analysis_response.complexity_score
        )

        if allocation_response.status == TaskStatus.COMPLETED:
            strategy = allocation_response.strategy

            print(f"Complexity Score: {analysis_response.complexity_score.overall_score:.2f}")
            print(f"Strategy: {strategy.strategy}")
            print(f"  Agents: {strategy.n_agents}")
            print(f"  K-Ahead: {strategy.k_ahead}")
            print(f"  Max Retries: {strategy.max_retries}")
            print(f"  Timeout: {strategy.timeout_ms}ms")

            # Interpret the strategy
            if strategy.n_agents == 1:
                print("  Interpretation: Simple direct solving")
            elif strategy.n_agents <= 3:
                print("  Interpretation: MDAP light multi-agent")
            else:
                print("  Interpretation: MAKER ultra multi-agent voting")
        else:
            print(f"✗ Resource allocation failed: {allocation_response.error}")

        print()

    # Show health and metrics
    print("=" * 60)
    print("Adapter Metrics")
    print("=" * 60)

    health = adapter.health_check()
    metrics = health['metrics']

    print(f"Total Requests: {metrics['requests_total']}")
    print(f"Successful: {metrics['requests_success']}")
    print(f"Failed: {metrics['requests_failed']}")
    print(f"Circuit Breaker Trips: {metrics['circuit_breaker_trips']}")


if __name__ == "__main__":
    main()
