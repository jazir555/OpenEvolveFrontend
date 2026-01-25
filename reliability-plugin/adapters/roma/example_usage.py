"""
ROMA Reliability Adapter - Example Usage
========================================

This script demonstrates how to use the ROMA Reliability Adapter
to solve problems with LMQL constraints and Guardrails validation.

Prerequisites:
- ROMA MCP tools installed and available
- LMQL adapter (optional, for constraint enforcement)
- Guardrails adapter (optional, for validation)

Author: OpenEvolve Team
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reliability_plugin.adapters.roma import (
    RomaReliabilityAdapter,
    solve_with_constraints,
    analyze_with_constraints
)
from reliability_plugin.adapters.roma.config import (
    create_constraints,
    RomaAdapterConfig,
    set_config
)


def example_1_basic_usage():
    """
    Example 1: Basic usage with convenience function.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    result = solve_with_constraints(
        task="Solve the traveling salesman problem for 5 cities",
        max_depth=3
    )

    if result.success:
        print(f"✓ Task solved successfully")
        print(f"  Layers used: {result.layers_used}")
        print(f"  ROMA status: {result.metadata.get('roma_status', 'N/A')}")

        if result.has_violations():
            print(f"  Constraint violations: {result.constraint_violations}")

        if result.has_validation_failures():
            print(f"  Validation failures: {result.validation_failures}")
    else:
        print(f"✗ Task failed: {result.error}")


def example_2_constrained_decomposition():
    """
    Example 2: Solve with detailed LMQL constraints.
    """
    print("\n" + "="*60)
    print("Example 2: Constrained Decomposition")
    print("="*60)

    # Build constraints using fluent builder
    constraints = create_constraints() \
        .with_max_depth(4) \
        .with_max_subtasks(12) \
        .with_subtask_token_limit(600) \
        .require_json() \
        .build()

    result = solve_with_constraints(
        task="Design a microservices architecture for an e-commerce platform",
        max_depth=4,
        constraints=constraints,
        execution_mode="event_driven"
    )

    if result.success:
        print(f"✓ Architecture design completed")
        print(f"  Layers used: {result.layers_used}")
        print(f"  Remediations applied: {result.remediation_applied}")

        # Show ROMA result
        if result.result:
            print(f"  Execution mode: {result.result.get('execution_mode')}")
            print(f"  Max depth: {result.result.get('max_depth')}")
    else:
        print(f"✗ Design failed: {result.error}")


def example_3_analysis_mode():
    """
    Example 3: Analyze a problem without solving.
    """
    print("\n" + "="*60)
    print("Example 3: Analysis Mode")
    print("="*60)

    adapter = RomaReliabilityAdapter()

    result = adapter.analyze_with_constraints(
        task="Analyze the complexity of implementing a distributed cache",
        analysis_type="decomposition",
        max_depth=2
    )

    if result.success:
        print(f"✓ Analysis completed")
        print(f"  Layers used: {result.layers_used}")
        print(f"  Analysis type: {result.metadata.get('analysis_type')}")

        # Show analysis
        if result.analysis:
            print(f"  Max depth found: {result.analysis.get('max_depth')}")
    else:
        print(f"✗ Analysis failed: {result.error}")


def example_4_verification_and_critique():
    """
    Example 4: Verify and critique a solution.
    """
    print("\n" + "="*60)
    print("Example 4: Verification and Critique")
    print("="*60)

    adapter = RomaReliabilityAdapter()

    # Solution to verify
    solution = "Implement Redis cache with TTL of 1 hour for all user sessions"

    # Verify the solution
    verification = adapter.verify_with_constraints(
        solution=solution,
        original_task="Design a session management system",
        verification_criteria=["correctness", "completeness", "security"]
    )

    if verification.success:
        print(f"✓ Verification completed")
        if verification.result:
            print(f"  Verification result: {verification.result.get('verification', 'N/A')[:100]}...")
    else:
        print(f"✗ Verification failed: {verification.error}")

    # Critique the solution
    critique = adapter.critique_with_constraints(
        solution=solution,
        original_task="Design a session management system",
        critique_focus="security"
    )

    if critique.success:
        print(f"✓ Critique completed")
        if critique.result:
            print(f"  Critique focus: {critique.result.get('critique_focus')}")
    else:
        print(f"✗ Critique failed: {critique.error}")


def example_5_custom_configuration():
    """
    Example 5: Use custom configuration.
    """
    print("\n" + "="*60)
    print("Example 5: Custom Configuration")
    print("="*60)

    # Create custom configuration
    config = RomaAdapterConfig(
        enabled=True,
        lmql_enabled=True,
        guardrails_enabled=True,
        max_depth_default=5,
        execution_mode_default="recursive",
        constraint_defaults={
            "max_depth": 5,
            "max_subtasks": 20,
            "subtask_token_limit": 1000
        }
    )

    # Validate configuration
    errors = config.validate()
    if errors:
        print(f"✗ Configuration errors: {errors}")
        return

    # Set configuration
    set_config(config)

    # Create adapter with custom config
    adapter = RomaReliabilityAdapter(config=config)

    # Solve with custom defaults
    result = adapter.solve_with_constraints(
        task="Plan a complete CI/CD pipeline implementation",
        max_depth=5
    )

    if result.success:
        print(f"✓ Task solved with custom configuration")
        print(f"  Layers used: {result.layers_used}")
    else:
        print(f"✗ Task failed: {result.error}")


def example_6_health_check():
    """
    Example 6: Check adapter health and status.
    """
    print("\n" + "="*60)
    print("Example 6: Health Check")
    print("="*60)

    adapter = RomaReliabilityAdapter()

    # Get basic status
    status = adapter.get_status()
    print(f"ROMA available: {status['roma_available']}")
    print(f"LMQL available: {status['lmql_available']}")
    print(f"Guardrails available: {status['guardrails_available']}")

    # Detailed health check
    health = adapter.health_check()
    print(f"\nAdapter healthy: {health['adapter_healthy']}")

    print("\nComponent health:")
    for component, component_health in health['components'].items():
        healthy = component_health.get('healthy', False)
        status_icon = "✓" if healthy else "✗"
        print(f"  {status_icon} {component}: {component_health.get('message', 'OK' if healthy else 'Not available')}")


def example_7_error_handling():
    """
    Example 7: Demonstrate error handling.
    """
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)

    # Try with invalid input (too short)
    result = solve_with_constraints(
        task="Hi",
        max_depth=3
    )

    if not result.success:
        print(f"✗ Task failed as expected")
        print(f"  Error: {result.error}")

        # Check if it's an input validation error
        if "Input validation failed" in result.error:
            print(f"  Type: Input validation error")

        # Check validation failures
        if result.validation_failures:
            print(f"  Validation failures: {result.validation_failures}")


def example_8_parallel_solving():
    """
    Example 8: Solve multiple tasks in parallel.
    """
    print("\n" + "="*60)
    print("Example 8: Parallel Solving")
    print("="*60)

    adapter = RomaReliabilityAdapter()

    tasks = [
        "Design a REST API for user authentication",
        "Implement a database migration strategy",
        "Set up monitoring and alerting"
    ]

    results = []
    for task in tasks:
        result = adapter.solve_with_constraints(
            task=task,
            max_depth=2,
            constraints={"max_subtasks": 5}
        )
        results.append(result)

    # Show results
    for i, result in enumerate(results, 1):
        status = "✓" if result.success else "✗"
        print(f"{status} Task {i}: {result.task[:50]}...")
        if result.success:
            print(f"   Layers: {result.layers_used}")
        else:
            print(f"   Error: {result.error}")


def main():
    """
    Run all examples.
    """
    print("\n" + "="*60)
    print("ROMA Reliability Adapter - Example Usage")
    print("="*60)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Constrained Decomposition", example_2_constrained_decomposition),
        ("Analysis Mode", example_3_analysis_mode),
        ("Verification and Critique", example_4_verification_and_critique),
        ("Custom Configuration", example_5_custom_configuration),
        ("Health Check", example_6_health_check),
        ("Error Handling", example_7_error_handling),
        ("Parallel Solving", example_8_parallel_solving),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Examples Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
