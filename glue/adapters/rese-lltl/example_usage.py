"""
RESE LLTL Example Usage

Demonstrates how to use the Logic-to-Loss Translation Layer adapter.

This example shows:
1. Creating symbolic constraints
2. Translating constraints to loss functions
3. Detecting contradictions
4. Using the adapter in a workflow

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Any
import json

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
glue_lib = project_root / "glue" / "lib"
adapter_src = project_root / "glue" / "adapters" / "rese-lltl" / "src"

sys.path.insert(0, str(glue_lib))
sys.path.insert(0, str(adapter_src))

from lltl_adapter import LLTLAdapter


# ============================================================================
# DEFINE CONSTRAINT STRUCTURE
# ============================================================================

@dataclass
class SymbolicConstraint:
    """
    A symbolic constraint to be translated to loss function.

    This is a simplified version matching the RESE canonical schema.
    """
    constraint_id: str
    type: str  # "hard" or "soft"
    category: str  # "logical", "causal", "temporal", "spatial", "resource", "epistemic"
    description: str
    expression: str
    dependencies: List[str]
    priority: float
    confidence: float


# ============================================================================
# EXAMPLE 1: SIMPLE TRANSLATION
# ============================================================================

def example_1_simple_translation():
    """Example 1: Translate a single constraint."""
    print("=" * 80)
    print("EXAMPLE 1: Single Constraint Translation")
    print("=" * 80)

    # Create adapter
    print("\n1. Initializing LLTL adapter...")
    adapter = LLTLAdapter()

    # Create a simple constraint
    constraint = SymbolicConstraint(
        constraint_id="example-001",
        type="hard",
        category="logical",
        description="Variable X must be greater than 5",
        expression="x > 5",
        dependencies=[],
        priority=1.0,
        confidence=0.95
    )

    print(f"   Created constraint: {constraint.constraint_id}")
    print(f"   Description: {constraint.description}")
    print(f"   Expression: {constraint.expression}")

    # Encode the constraint
    print("\n2. Encoding constraint...")
    encoded, error = adapter.encode_single(constraint)

    if error:
        print(f"   ERROR: {error}")
        return

    print(f"   [OK] Encoded successfully")
    print(f"   Constraint ID: {encoded['constraint_id']}")
    print(f"   Feature vector dimension: {len(encoded['feature_vector'])}")
    print(f"   Feature vector (first 10): {encoded['feature_vector'][:10]}")
    print(f"   Structural encoding: {encoded['structural_encoding']}")

    # Get stats
    print("\n3. Adapter statistics:")
    stats = adapter.get_stats()
    cache_stats = stats['translator_stats']['encoder_cache']
    print(f"   Cache hits: {cache_stats['cache_hits']}")
    print(f"   Cache misses: {cache_stats['cache_misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")


# ============================================================================
# EXAMPLE 2: MULTIPLE CONSTRAINTS
# ============================================================================

def example_2_multiple_constraints():
    """Example 2: Translate multiple constraints."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Multiple Constraints Translation")
    print("=" * 80)

    # Create adapter
    print("\n1. Initializing LLTL adapter...")
    adapter = LLTLAdapter()

    # Create multiple constraints
    constraints = [
        SymbolicConstraint(
            constraint_id="example-002-a",
            type="hard",
            category="logical",
            description="X must be greater than 5",
            expression="x > 5",
            dependencies=[],
            priority=1.0,
            confidence=0.95
        ),
        SymbolicConstraint(
            constraint_id="example-002-b",
            type="soft",
            category="causal",
            description="Y should be proportional to X",
            expression="y = 2 * x",
            dependencies=["example-002-a"],
            priority=0.8,
            confidence=0.7
        ),
        SymbolicConstraint(
            constraint_id="example-002-c",
            type="hard",
            category="temporal",
            description="Z must be less than current time",
            expression="z < now()",
            dependencies=[],
            priority=0.9,
            confidence=0.85
        )
    ]

    print(f"   Created {len(constraints)} constraints:")
    for c in constraints:
        print(f"   - {c.constraint_id}: {c.description}")

    # Translate all constraints
    print("\n2. Translating constraints...")
    result, error = adapter.translate_constraints(constraints, timeout_ms=10000)

    if error:
        print(f"   ERROR: {error}")
        return

    print(f"   [OK] Translation completed successfully")
    print(f"   Input constraints: {result['input_constraints']}")
    print(f"   Encoded constraints: {result['encoded_constraints']}")
    print(f"   Loss functions: {result['loss_functions']}")
    print(f"   Contradictions detected: {result['contradictions_detected']}")
    print(f"   Duration: {result['duration_ms']:.2f}ms")

    # Show combined loss
    print("\n3. Combined loss function:")
    combined = result['combined_loss']
    print(f"   Combination strategy: {combined['combination_strategy']}")
    print(f"   Total weight: {combined['total_weight']:.4f}")
    print(f"   Individual weights: {[f'{w:.4f}' for w in combined['weights']]}")


# ============================================================================
# EXAMPLE 3: CONTRADICTION DETECTION
# ============================================================================

def example_3_contradiction_detection():
    """Example 3: Detect contradictions between constraints."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Contradiction Detection")
    print("=" * 80)

    # Create adapter
    print("\n1. Initializing LLTL adapter...")
    adapter = LLTLAdapter()

    # Create potentially contradictory constraints
    constraints = [
        SymbolicConstraint(
            constraint_id="example-003-a",
            type="hard",
            category="logical",
            description="X must be greater than 10",
            expression="x > 10",
            dependencies=[],
            priority=1.0,
            confidence=0.95
        ),
        SymbolicConstraint(
            constraint_id="example-003-b",
            type="hard",
            category="logical",
            description="X must be less than 5",
            expression="x < 5",
            dependencies=[],
            priority=1.0,
            confidence=0.95
        ),
        SymbolicConstraint(
            constraint_id="example-003-c",
            type="hard",
            category="logical",
            description="X equals 7",
            expression="x == 7",
            dependencies=[],
            priority=1.0,
            confidence=0.95
        )
    ]

    print(f"   Created {len(constraints)} potentially contradictory constraints:")
    for c in constraints:
        print(f"   - {c.constraint_id}: {c.description} ({c.expression})")

    # Detect contradictions
    print("\n2. Detecting contradictions...")
    contradictions, error = adapter.detect_contradictions(constraints)

    if error:
        print(f"   WARNING: {error}")

    print(f"   Found {len(contradictions)} potential contradictions")

    if contradictions:
        print("\n3. Contradiction details:")
        for i, contradiction in enumerate(contradictions, 1):
            print(f"   Contradiction {i}:")
            print(f"   - Between: {contradiction['constraint1_id']} and {contradiction['constraint2_id']}")
            print(f"   - Type: {contradiction['type']}")
            print(f"   - Confidence: {contradiction['confidence']}")
    else:
        print("\n3. No direct contradictions detected (naive implementation)")
        print("   Note: Current DITO uses naive O(n²) detection")
        print("   Full contradiction detection requires Tier 6 optimizations")


# ============================================================================
# EXAMPLE 4: HEALTH CHECK AND STATISTICS
# ============================================================================

def example_4_health_and_stats():
    """Example 4: Health check and statistics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Health Check and Statistics")
    print("=" * 80)

    # Create adapter
    print("\n1. Initializing adapter...")
    adapter = LLTLAdapter()

    # Health check
    print("\n2. Performing health check...")
    is_healthy, message = adapter.health_check()

    if is_healthy:
        print(f"   [OK] Adapter is healthy: {message}")
    else:
        print(f"   [FAIL] Adapter is unhealthy: {message}")
        return

    # Get detailed statistics
    print("\n3. Detailed statistics:")
    stats = adapter.get_stats()

    print("\n   Configuration:")
    print(f"   - Encoding dimension: {stats['adapter_config']['encoding']['encoding_dim']}")
    print(f"   - Cache size: {stats['adapter_config']['encoding']['cache_size']}")
    print(f"   - Default loss type: {stats['adapter_config']['loss']['default_type']}")
    print(f"   - Combination strategy: {stats['adapter_config']['loss']['combination_strategy']}")
    print(f"   - Timeout: {stats['adapter_config']['timeout_ms']}ms")

    print("\n   Runtime Statistics:")
    translator_stats = stats['translator_stats']
    print(f"   - Encoder cache hits: {translator_stats['encoder_cache']['cache_hits']}")
    print(f"   - Encoder cache misses: {translator_stats['encoder_cache']['cache_misses']}")
    print(f"   - Encoder hit rate: {translator_stats['encoder_cache']['hit_rate']:.2%}")
    print(f"   - Cache size: {translator_stats['encoder_cache']['cache_size']}")
    print(f"   - Contradictions tracked: {translator_stats['dito_contradictions']}")


# ============================================================================
# EXAMPLE 5: ERROR HANDLING
# ============================================================================

def example_5_error_handling():
    """Example 5: Error handling and edge cases."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Error Handling")
    print("=" * 80)

    # Create adapter
    print("\n1. Initializing adapter...")
    adapter = LLTLAdapter()

    # Test 1: Empty constraint list
    print("\n2. Test: Empty constraint list")
    result, error = adapter.translate_constraints([])

    if error:
        print(f"   [OK] Correctly handled empty list: {error}")
    else:
        print(f"   [FAIL] Should have returned error for empty list")

    # Test 2: Invalid constraint structure
    print("\n3. Test: Invalid constraint structure")
    invalid_constraint = {"invalid": "data"}

    result, error = adapter.translate_constraints([invalid_constraint])

    if error:
        print(f"   [OK] Correctly handled invalid constraint")
        print(f"   Error: {error}")
    else:
        print(f"   [FAIL] Should have returned error for invalid constraint")

    # Test 3: Timeout handling
    print("\n4. Test: Very short timeout")
    constraints = [
        SymbolicConstraint(
            constraint_id="test-timeout",
            type="hard",
            category="logical",
            description="Test timeout",
            expression="x > 0",
            dependencies=[],
            priority=1.0,
            confidence=1.0
        )
    ]

    result, error = adapter.translate_constraints(constraints, timeout_ms=1)

    if error:
        print(f"   [OK] Timeout handling works: {error}")
    else:
        print(f"   Note: Operation completed within 1ms timeout")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("RESE LOGIC-TO-LOSS TRANSLATION LAYER - EXAMPLE USAGE")
    print("=" * 80)
    print("\nThis script demonstrates the LLTL adapter capabilities.")
    print("Following CLAUDE.md principles:")
    print("- Law of Configuration Explicitness: All config via env vars")
    print("- Law of Idempotency: Cached translations")
    print("- Circuit Breaker: Graceful failure handling")
    print("- Structured Logging: JSON logs with correlation_id")
    print("")

    try:
        # Run examples
        example_1_simple_translation()
        example_2_multiple_constraints()
        example_3_contradiction_detection()
        example_4_health_and_stats()
        example_5_error_handling()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY [OK]")
        print("=" * 80)
        print("\nThe LLTL adapter is ready for integration!")
        print("\nNext steps:")
        print("1. Review the adapter API in glue/adapters/rese-lltl/README.md")
        print("2. Run the probe script: bash glue/adapters/rese-lltl/probes/check_lltl.sh")
        print("3. Integrate with your RESE pipeline")
        print("")

    except Exception as e:
        print(f"\n[FAIL] Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
