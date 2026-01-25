"""
MDAP Reliability Adapter - Usage Examples
==========================================

This file demonstrates how to use the MDAP Reliability Adapter
to add Guardrails validation to MDAP voting without modifying MDAP core.

Author: OpenEvolve Team
Version: 1.0.0
"""

from reliability_plugin.adapters.mdap import (
    MDAPReliabilityAdapter,
    create_mdap_adapter,
    solve_with_guardrails
)
import json


# =============================================================================
# EXAMPLE 1: Basic Usage
# =============================================================================

def example_1_basic_usage():
    """Basic MDAP solve with validation"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Create adapter
    adapter = create_mdap_adapter()

    # Check status
    status = adapter.get_status()
    print(f"\nStatus:")
    print(f"  MDAP Available: {status['mdap_available']}")
    print(f"  Guardrails Available: {status['guardrails_available']}")

    # Solve with validation
    result = adapter.solve_with_validation(
        task="What is 2 + 2? Provide the answer with confidence.",
        mdap_k_ahead=3,
        validators=["vote_format", "json_structure"]
    )

    print(f"\nResult:")
    print(f"  Success: {result.success}")

    if result.success:
        print(f"  Statistics: {result.statistics}")
        if result.result:
            print(f"  Result (truncated): {json.dumps(result.result, indent=2)[:200]}...")
    else:
        print(f"  Error: {result.error}")
        if result.validation_failures:
            print(f"  Validation Failures: {result.validation_failures}")


# =============================================================================
# EXAMPLE 2: Vote Validation
# =============================================================================

def example_2_vote_validation():
    """Validate individual MDAP votes"""
    print("\n" + "=" * 60)
    print("Example 2: Vote Validation")
    print("=" * 60)

    adapter = create_mdap_adapter()

    # Test votes
    test_votes = [
        {"decision": "APPROVE", "confidence": 0.9, "reason": "Correct answer"},
        {"decision": "REJECT", "confidence": 0.1},
        {"invalid": "format"},
        "MALICIOUS <script>alert('xss')</script> CONTENT"
    ]

    for i, vote in enumerate(test_votes):
        print(f"\n--- Vote {i+1} ---")
        print(f"Input: {json.dumps(vote) if isinstance(vote, dict) else vote}")

        validation = adapter.verify_vote(
            vote=vote,
            validators=["json_structure", "required_fields", "malicious_patterns"]
        )

        print(f"Valid: {validation.is_valid}")
        print(f"Remediated: {validation.remediated}")

        if validation.failures:
            print(f"Failures: {validation.failures}")

        if validation.remediated:
            print(f"Remediated Vote: {json.dumps(validation.vote) if isinstance(validation.vote, dict) else validation.vote}")


# =============================================================================
# EXAMPLE 3: Convenience Function
# =============================================================================

def example_3_convenience_function():
    """One-off solve with guardrails"""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Function")
    print("=" * 60)

    # One-off solve
    result = solve_with_guardrails(
        task="Explain quantum computing in one sentence.",
        mdap_k_ahead=5,
        validators=["vote_format", "json_structure"]
    )

    print(f"\nSuccess: {result.success}")
    print(f"Statistics: {result.statistics}")

    if result.success:
        print(f"\nSolution found with {result.statistics['total_votes']} total votes")


# =============================================================================
# EXAMPLE 4: Error Handling
# =============================================================================

def example_4_error_handling():
    """Demonstrate error handling"""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling")
    print("=" * 60)

    adapter = create_mdap_adapter()

    # Test 1: Invalid mdap_k_ahead
    print("\n--- Test 1: Invalid mdap_k_ahead ---")
    result = adapter.solve_with_validation(
        task="Test task",
        mdap_k_ahead=25  # Invalid: must be 2-20
    )
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 2: Empty task
    print("\n--- Test 2: Empty task ---")
    result = adapter.solve_with_validation(
        task="",  # Invalid: empty string
        mdap_k_ahead=3
    )
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 3: Very long task
    print("\n--- Test 3: Very long task ---")
    long_task = "Test " * 5000  # > 10000 characters
    result = adapter.solve_with_validation(
        task=long_task,
        mdap_k_ahead=3
    )
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")


# =============================================================================
# EXAMPLE 5: Statistics Tracking
# =============================================================================

def example_5_statistics():
    """Track adapter statistics"""
    print("\n" + "=" * 60)
    print("Example 5: Statistics Tracking")
    print("=" * 60)

    adapter = create_mdap_adapter()

    # Reset statistics
    adapter.reset_statistics()
    print("Statistics reset")

    # Perform multiple solves
    tasks = [
        "What is 2 + 2?",
        "What is 3 + 3?",
        "What is 4 + 4?"
    ]

    for task in tasks:
        result = adapter.solve_with_validation(
            task=task,
            mdap_k_ahead=3
        )
        print(f"\nTask: {task}")
        print(f"  Success: {result.success}")

    # Get statistics
    stats = adapter.get_statistics()
    print(f"\n--- Cumulative Statistics ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")


# =============================================================================
# EXAMPLE 6: Configuration
# =============================================================================

def example_6_configuration():
    """Custom configuration"""
    print("\n" + "=" * 60)
    print("Example 6: Configuration")
    print("=" * 60)

    # Get current configuration
    adapter = create_mdap_adapter()
    status = adapter.get_status()

    print(f"\nCurrent Configuration:")
    print(f"  Guardrails Enabled: {status['config']['guardrails_enabled']}")
    print(f"  Validators: {status['config']['validators']}")
    print(f"  On-Fail Strategy: {status['config']['on_fail_strategy']}")
    print(f"  Max Retries: {status['config']['max_retries']}")


# =============================================================================
# EXAMPLE 7: Batch Processing
# =============================================================================

def example_7_batch_processing():
    """Process multiple tasks"""
    print("\n" + "=" * 60)
    print("Example 7: Batch Processing")
    print("=" * 60)

    adapter = create_mdap_adapter()

    tasks = [
        "Calculate 5 + 7",
        "Calculate 10 - 3",
        "Calculate 6 * 8"
    ]

    results = []
    for i, task in enumerate(tasks):
        print(f"\nProcessing task {i+1}: {task}")

        result = adapter.solve_with_validation(
            task=task,
            mdap_k_ahead=3,
            validators=["vote_format"]
        )

        results.append(result)
        print(f"  Success: {result.success}")

    # Summary
    successful = sum(1 for r in results if r.success)
    print(f"\n--- Summary ---")
    print(f"  Total Tasks: {len(tasks)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(tasks) - successful}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("MDAP Reliability Adapter - Usage Examples")
    print("=" * 60)

    try:
        # Run examples
        example_1_basic_usage()
        example_2_vote_validation()
        example_3_convenience_function()
        example_4_error_handling()
        example_5_statistics()
        example_6_configuration()
        example_7_batch_processing()

        print("\n" + "=" * 60)
        print("All Examples Complete")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
