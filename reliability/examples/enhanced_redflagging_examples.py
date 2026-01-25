"""
Enhanced Red Flagging System - Complete Examples

This file demonstrates the multi-layered red flagging system with LMQL
and Guardrails integration.
"""

import sys
import json
from typing import Dict, Any, List

sys.path.append("../")
sys.path.append("../../")

from reliability.enhanced_redflagger import (
    EnhancedRedFlagger,
    EnhancedRedFlagRules,
    RedFlag,
    RedFlagSeverity,
    create_enhanced_redflagger
)
from reliability.lmql_adapter import get_default_adapter
from reliability.guardrails_adapter import create_adapter

# Try to import MDAP adapter, handle gracefully if unavailable
try:
    from reliability_plugin.adapters.mdap.mdap_reliability_adapter import (
        MDAPReliabilityAdapter,
        solve_with_redflagging
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    print("Note: MDAP adapter not available - Example 4 will be skipped")

# =============================================================================
# EXAMPLE 1: Basic Enhanced Red Flagging
# ============================================================================

def example_1_basic_redflagging():
    """Example 1: Basic enhanced red flagging"""
    print("=" * 70)
    print("EXAMPLE 1: Basic Enhanced Red Flagging")
    print("=" * 70)

    # Create enhanced red flagger with default rules
    flagger = create_enhanced_redflagger()

    # Test output with red flags
    flagged_output = """
    This response contains a secret API key: sk-1234567890abcdef
    and some toxic language that should be flagged.
    """

    # Check for red flags
    is_flagged, flags = flagger.check_for_red_flags(
        raw_text=flagged_output,
        candidate={"response": flagged_output},
        schema=None
    )

    print(f"\nIs Flagged: {is_flagged}")
    print(f"Number of Flags: {len(flags)}")

    for flag in flags:
        print(f"\n  🚩 {flag.category}")
        print(f"     Severity: {flag.severity.value}")
        print(f"     Message: {flag.message}")
        print(f"     Remediation: {flag.remediation or 'N/A'}")

    # Get statistics
    stats = flagger.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total Checks: {stats['total_checks']}")
    print(f"  Critical Flags: {stats['critical_flags']}")
    print(f"  High Flags: {stats['high_flags']}")

    return flagger, flags


# =============================================================================
# EXAMPLE 2: Custom Red Flag Rules
# ============================================================================

def example_2_custom_rules():
    """Example 2: Custom red flag rules"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Red Flag Rules")
    print("=" * 70)

    # Create custom rules
    custom_rules = EnhancedRedFlagRules(
        max_tokens=500,
        max_characters=3000,
        min_confidence=0.7,
        confidence_threshold=0.7,

        # Enable LMQL pre-generation
        enable_lmql_constraints=True,
        lmql_max_retries=2,

        # Custom Guardrails validators
        guardrails_validators=[
            "toxic_language",
            "pii_filter",
            "secrets_detection",
            "competitor_check"  # Custom validator
        ],

        # Forbidden content
        forbidden_keywords=["confidential", "internal", "do_not_share"],

        # Required content
        required_keywords=["solution", "answer"],

        # Format requirement
        required_format="json",
        require_schema_match=True,

        # Thresholds
        toxicity_threshold=0.9,  # More strict
        pii_detection_strict=True
    )

    # Create flagger with custom rules
    flagger = EnhancedRedFlagger(rules=custom_rules)

    # Test output
    test_output = {
        "solution": "This is the answer to your problem.",
        "confidence": 0.8
    }

    is_flagged, flags = flagger.check_for_red_flags(
        raw_text=json.dumps(test_output),
        candidate=test_output,
        schema={"type": "object"}
    )

    print(f"\nCustom Rules Check:")
    print(f"  Is Flagged: {is_flagged}")
    print(f"  Flags: {len(flags)}")

    for flag in flags:
        print(f"    - {flag.category}: {flag.message}")

    return flagger


# =============================================================================
# EXAMPLE 3: LMQL Pre-Generation Constraints
# ============================================================================

def example_3_lmql_constraints():
    """Example 3: LMQL pre-generation constraints"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: LMQL Pre-Generation Constraints")
    print("=" * 70)

    # Create flagger with LMQL enabled
    rules = EnhancedRedFlagRules(
        enable_lmql_constraints=True,
        max_tokens=300,
        max_characters=2000,
        forbidden_keywords=["harmful", "illegal"],
        required_format="json",
        confidence_threshold=0.6
    )

    flagger = EnhancedRedFlagger(rules=rules)

    # Get LMQL constraints for pre-generation
    constraints = flagger.get_lmql_constraints()

    print(f"\n🔒 Generated {len(constraints)} LMQL Constraints:")

    for i, constraint in enumerate(constraints, 1):
        print(f"\n  Constraint {i}:")
        print(f"    Type: {constraint.type.value}")
        print(f"    Field: {constraint.field}")
        print(f"    Description: {constraint.description}")
        if constraint.min_value is not None:
            print(f"    Min Value: {constraint.min_value}")
        if constraint.max_value is not None:
            print(f"    Max Value: {constraint.max_value}")
        if constraint.max_length is not None:
            print(f"    Max Length: {constraint.max_length}")

    # Show how to use with LMQL adapter
    print(f"\n💡 Usage with LMQL Adapter:")
    print(f"```python")
    print(f"lmql_adapter = get_default_adapter()")
    print(f"result = lmql_adapter.constrained_generation(")
    print(f"    prompt='Generate response',")
    print(f"    constraints=constraints")
    print(f")")
    print(f"```")

    return flagger, constraints


# =============================================================================
# EXAMPLE 4: Integration with MDAP
# ============================================================================

def example_4_mdap_integration():
    """Example 4: Integration with MDAP adapter"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Integration with MDAP Adapter")
    print("=" * 70)

    if not MDAP_AVAILABLE:
        print("\n⚠️  MDAP adapter not available - skipping this example")
        print("   To enable: Install reliability-plugin with MDAP support")
        return None, None

    # Method 1: Using convenience function
    print("\n📌 Method 1: Convenience Function")
    print("-" * 70)

    result = solve_with_redflagging(
        task="What is 2 + 2?",
        mdap_k_ahead=3,
        use_lmql_constraints=True,
        use_enhanced_validation=True
    )

    print(f"\nSuccess: {result['success']}")
    print(f"Red Flags: {result['red_flag_count']}")
    print(f"Layers Used: {result['layers_used']}")

    if result['red_flags']:
        print("\n🚩 Red Flags Detected:")
        for flag in result['red_flags'][:3]:  # Show first 3
            print(f"  - {flag['category']}: {flag['message']}")

    print(f"\n📊 Flagging Statistics:")
    stats = result['flagging_statistics']
    print(f"  Total Checks: {stats['total_checks']}")
    print(f"  Critical Flags: {stats['critical_flags']}")
    print(f"  High Flags: {stats['high_flags']}")
    if 'prevention_rate' in stats:
        print(f"  Prevention Rate: {stats['prevention_rate']:.2%}")

    # Method 2: Using adapter directly
    print("\n\n📌 Method 2: Direct Adapter Usage")
    print("-" * 70)

    adapter = MDAPReliabilityAdapter()

    result = adapter.solve_with_enhanced_redflagging(
        task="Generate a secure JSON response",
        mdap_k_ahead=5,
        use_lmql_constraints=True,
        use_enhanced_validation=True,
        schema={"type": "object", "required": ["answer"]}
    )

    print(f"\nSuccess: {result['success']}")
    print(f"Result: {result.get('result', 'N/A')}")

    return adapter, result


# =============================================================================
# EXAMPLE 5: Severity-Based Handling
# ============================================================================

def example_5_severity_handling():
    """Example 5: Severity-based handling of red flags"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Severity-Based Red Flag Handling")
    print("=" * 70)

    flagger = create_enhanced_redflagger()

    # Test outputs with different severity levels
    test_cases = [
        ("Critical", "Here is a secret password: my_password_123"),
        ("High", "This is harmful content with toxic language"),
        ("Medium", "This response is too long and exceeds the token limit"),
        ("Low", "Minor formatting issue")
    ]

    for severity_level, test_output in test_cases:
        print(f"\n{severity_level} Severity Test:")
        print(f"  Output: {test_output[:50]}...")

        is_flagged, flags = flagger.check_for_red_flags(
            raw_text=test_output,
            candidate={"text": test_output}
        )

        # Categorize by severity
        critical = [f for f in flags if f.severity == RedFlagSeverity.CRITICAL]
        high = [f for f in flags if f.severity == RedFlagSeverity.HIGH]
        medium = [f for f in flags if f.severity == RedFlagSeverity.MEDIUM]
        low = [f for f in flags if f.severity == RedFlagSeverity.LOW]

        print(f"  Flags: {len(flags)} (CRITICAL: {len(critical)}, HIGH: {len(high)}, MEDIUM: {len(medium)}, LOW: {len(low)})")

        # Show handling based on severity
        if critical or high:
            print(f"  ✋ ACTION: Reject output (severity too high)")
        elif medium:
            print(f"  ⚠️  ACTION: Flag for review (medium severity)")
        elif low:
            print(f"  ℹ️  ACTION: Log warning (low severity)")
        else:
            print(f"  ✅ ACTION: Accept output")


# =============================================================================
# EXAMPLE 6: Statistics and Monitoring
# ============================================================================

def example_6_statistics():
    """Example 6: Statistics and monitoring"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Statistics and Monitoring")
    print("=" * 70)

    flagger = create_enhanced_redflagger()

    # Run multiple checks
    test_outputs = [
        "This is a normal response without any issues.",
        "This has an API_KEY: sk-1234567890",
        "This is toxic and should be rejected.",
        "Another normal response."
    ]

    print("\n🔍 Running validation checks...")
    for i, output in enumerate(test_outputs, 1):
        is_flagged, flags = flagger.check_for_red_flags(
            raw_text=output,
            candidate={"text": output}
        )
        print(f"  Check {i}: Flagged={is_flagged}, Flags={len(flags)}")

    # Get comprehensive statistics
    stats = flagger.get_statistics()

    print("\n📊 Comprehensive Statistics:")
    print(f"  Total Checks: {stats['total_checks']}")
    print(f"  Pre-Generation Preventions: {stats['pre_generation_preventions']}")
    print(f"  Post-Generation Flags: {stats['post_generation_flags']}")
    print(f"  Remediated Outputs: {stats['remediated_outputs']}")
    print(f"  Rejected Outputs: {stats['rejected_outputs']}")
    print(f"\n  By Severity:")
    print(f"    Critical: {stats['critical_flags']}")
    print(f"    High: {stats['high_flags']}")
    print(f"    Medium: {stats['medium_flags']}")
    print(f"    Low: {stats['low_flags']}")
    print(f"\n  Rates:")
    print(f"    Flag Rate: {stats['flag_rate']:.2%}")
    print(f"    Prevention Rate: {stats['prevention_rate']:.2%}")
    print(f"    Availability:")
    print(f"      LMQL: {stats['lmql_available']}")
    print(f"      Guardrails: {stats['guardrails_available']}")


# =============================================================================
# EXAMPLE 7: Real-World Use Case
# ============================================================================

def example_7_real_world():
    """Example 7: Real-world use case - Secure MDAPI voting"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Real-World Use Case - Secure MDAP Voting")
    print("=" * 70)

    if not MDAP_AVAILABLE:
        print("\n⚠️  MDAP adapter not available - skipping this example")
        print("   To enable: Install reliability-plugin with MDAP support")
        return None

    # Scenario: MDAP voting with enhanced security
    print("\n📋 Scenario: Multi-agent voting with security validation")
    print("-" * 70)

    # Create adapter with enhanced red flagging
    adapter = MDAPReliabilityAdapter()

    # Configure for secure voting
    result = adapter.solve_with_enhanced_redflagging(
        task="Generate a secure API response for user authentication",
        mdap_k_ahead=5,
        use_lmql_constraints=True,  # Prevent bad content during generation
        use_enhanced_validation=True,  # Validate all votes
        schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.5}
            },
            "required": ["action", "confidence"]
        }
    )

    print(f"\n✅ Success: {result['success']}")
    print(f"📊 Red Flags: {result['red_flag_count']}")
    print(f"🔧 Layers: {', '.join(result['layers_used'])}")

    # Show detailed red flag information
    if result['red_flags']:
        print(f"\n🚩 Detailed Red Flag Information:")
        for flag in result['red_flags']:
            print(f"\n  Category: {flag['category']}")
            print(f"  Severity: {flag['severity']}")
            print(f"  Message: {flag['message']}")
            print(f"  Validator: {flag.get('validator', 'N/A')}")
            print(f"  Remediation: {flag.get('remediation', 'N/A')}")

    # Show effectiveness metrics
    stats = result['flagging_statistics']
    print(f"\n📈 Effectiveness Metrics:")
    print(f"  Total Checks: {stats['total_checks']}")
    print(f"  Flags Caught: {stats['post_generation_flags']}")
    if stats['total_checks'] > 0:
        print(f"  Remediation Rate: {stats['remediated_outputs'] / stats['total_checks'] * 100:.1f}%")
        print(f"  Rejection Rate: {stats['rejected_outputs'] / stats['total_checks'] * 100:.1f}%")

    return result


# =============================================================================
# MAIN RUNNER
# ============================================================================

def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "ENHANCED RED FLAGGING EXAMPLES" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    examples = [
        ("Basic Enhanced Red Flagging", example_1_basic_redflagging),
        ("Custom Red Flag Rules", example_2_custom_rules),
        ("LMQL Pre-Generation Constraints", example_3_lmql_constraints),
        ("Integration with MDAP", example_4_mdap_integration),
        ("Severity-Based Handling", example_5_severity_handling),
        ("Statistics and Monitoring", example_6_statistics),
        ("Real-World Use Case", example_7_real_world)
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            print(f"\n\n{'=' * 70}")
            print(f"EXAMPLE {i}: {name}")
            print(f"{'=' * 70}")
            func()
        except Exception as e:
            print(f"\n❌ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n\n{'=' * 70}")
    print("✅ ALL EXAMPLES COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
