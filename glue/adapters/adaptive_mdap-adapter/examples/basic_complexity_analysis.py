"""
Example: Basic Complexity Analysis

Demonstrates how to use the Adaptive MDAP Adapter to analyze
the complexity of subproblems.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_mdap_adapter import (
    get_adapter,
    CanonicalSubProblem,
    TaskStatus,
    AdaptiveMDAPAdapterConfig
)


def main():
    """Run basic complexity analysis example."""

    # Configure adapter (could also load from environment)
    config = AdaptiveMDAPAdapterConfig(
        timeout_ms=5000,
        max_retries=2,
        log_level="INFO"
    )

    # Get adapter instance
    adapter = get_adapter(config)

    print("=" * 60)
    print("Adaptive MDAP Adapter - Basic Complexity Analysis")
    print("=" * 60)
    print()

    # Example 1: Simple problem
    print("Example 1: Simple Problem")
    print("-" * 60)

    simple_subproblem = CanonicalSubProblem(
        id="simple-001",
        description="Create a simple addition function",
        domain="programming",
        depth=1,
        dependencies=[],
        metadata={"language": "python"}
    )

    response = adapter.analyze_complexity(
        subproblem=simple_subproblem,
        correlation_id="example-001"
    )

    if response.status == TaskStatus.COMPLETED:
        print(f"✓ Status: {response.status.value}")
        print(f"  Overall Complexity: {response.complexity_score.overall_score:.2f}")
        print(f"  Text Length Score: {response.complexity_score.text_length_score:.2f}")
        print(f"  Dependency Score: {response.complexity_score.dependency_score:.2f}")
        print(f"  Depth Score: {response.complexity_score.depth_score:.2f}")
        print(f"  Execution Time: {response.execution_time_ms}ms")
    else:
        print(f"✗ Error: {response.error}")

    print()

    # Example 2: Medium complexity problem
    print("Example 2: Medium Complexity Problem")
    print("-" * 60)

    medium_subproblem = CanonicalSubProblem(
        id="medium-001",
        description="Implement a REST API with authentication and database integration",
        domain="web_development",
        depth=3,
        dependencies=["database-setup", "auth-module"],
        metadata={"framework": "fastapi", "database": "postgresql"}
    )

    response = adapter.analyze_complexity(
        subproblem=medium_subproblem,
        correlation_id="example-002"
    )

    if response.status == TaskStatus.COMPLETED:
        print(f"✓ Status: {response.status.value}")
        print(f"  Overall Complexity: {response.complexity_score.overall_score:.2f}")
        print(f"  Text Length Score: {response.complexity_score.text_length_score:.2f}")
        print(f"  Dependency Score: {response.complexity_score.dependency_score:.2f}")
        print(f"  Depth Score: {response.complexity_score.depth_score:.2f}")
        print(f"  Execution Time: {response.execution_time_ms}ms")
    else:
        print(f"✗ Error: {response.error}")

    print()

    # Example 3: High complexity problem
    print("Example 3: High Complexity Problem")
    print("-" * 60)

    complex_subproblem = CanonicalSubProblem(
        id="complex-001",
        description="Design and implement a distributed consensus algorithm for a blockchain system with Byzantine fault tolerance, cryptographic verification, and network partition handling",
        domain="distributed_systems",
        depth=5,
        dependencies=["cryptography", "networking", "consensus-core", "verification"],
        metadata={"protocol": "raft", "tolerance": "byzantine"}
    )

    response = adapter.analyze_complexity(
        subproblem=complex_subproblem,
        correlation_id="example-003"
    )

    if response.status == TaskStatus.COMPLETED:
        print(f"✓ Status: {response.status.value}")
        print(f"  Overall Complexity: {response.complexity_score.overall_score:.2f}")
        print(f"  Text Length Score: {response.complexity_score.text_length_score:.2f}")
        print(f"  Dependency Score: {response.complexity_score.dependency_score:.2f}")
        print(f"  Depth Score: {response.complexity_score.depth_score:.2f}")
        print(f"  Execution Time: {response.execution_time_ms}ms")
    else:
        print(f"✗ Error: {response.error}")

    print()

    # Health check
    print("=" * 60)
    print("Adapter Health Check")
    print("=" * 60)

    health = adapter.health_check()
    print(f"Status: {health['status']}")
    print(f"Circuit Breaker: {health['circuit_breaker_state']}")
    print(f"MDAP Available: {health['mdap_available']}")
    print(f"Metrics: {health['metrics']}")


if __name__ == "__main__":
    main()
