"""
Integration test for RESE Phase II adapter

Tests the full adapter interface and API.
"""

import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "schemas"))

# Set required env vars
os.environ["PHASE2_MAX_TARGET_DOMAINS"] = "10"
os.environ["PHASE2_IMECH_THRESHOLD"] = "0.7"
os.environ["PHASE2_PATTERN_THRESHOLD"] = "0.6"
os.environ["PHASE2_TIMEOUT_MS"] = "20000"
os.environ["PHASE2_MAX_MAPPINGS"] = "50"
os.environ["PHASE2_ENABLE_CONSTRAINT_INVERSION"] = "true"
os.environ["PHASE2_SEARCH_DEPTH"] = "5"

from phase2_adapter import Phase2Adapter


def test_adapter_basic():
    """Test basic adapter functionality."""
    print("Testing Phase II Adapter...")

    # Create adapter
    adapter = Phase2Adapter()

    # Check health
    health = adapter.get_health()
    assert health["status"] == "healthy"
    print(f"✓ Adapter healthy: {health}")

    # Execute Phase II
    request = {
        "source_domain": "physics",
        "problem_description": "Energy conservation in closed system with wave propagation",
        "target_domains": ["biology", "economics"],
        "constraints": ["energy is conserved", "momentum is conserved"],
        "context": {"temperature": "high"}
    }

    result = adapter.execute_phase2(request)

    # Verify result
    assert "result_id" in result
    assert result["source_domain"] == "physics"
    assert len(result["target_domains"]) == 2
    assert "mappings" in result
    assert "summary" in result

    print(f"✓ Phase II execution successful")
    print(f"  - Result ID: {result['result_id']}")
    print(f"  - Mappings found: {result['summary']['mapping_count']}")
    print(f"  - Patterns found: {result['summary']['pattern_count']}")
    print(f"  - Inverted constraints: {result['summary']['inverted_count']}")
    print(f"  - Best I_mech: {result['summary']['best_imech_score']:.2f}")
    print(f"  - Execution time: {result['execution_time_ms']:.2f}ms")

    # Check best mapping
    if result["best_mapping"]:
        print(f"✓ Best mapping found:")
        print(f"  - Target: {result['best_mapping']['target_domain']}")
        print(f"  - I_mech: {result['best_mapping']['i_mech_score']:.2f}")
        print(f"  - Confidence: {result['best_mapping']['confidence']:.2f}")

    # Check inverted constraints
    if result["inverted_constraints"]:
        print(f"✓ Inverted constraints:")
        for inv in result["inverted_constraints"]:
            print(f"  - {inv['inverted'][:80]}...")

    print("\n✓ All integration tests passed!")


def test_adapter_error_handling():
    """Test adapter error handling."""
    print("\nTesting error handling...")

    adapter = Phase2Adapter()

    # Invalid request (missing source_domain)
    try:
        result = adapter.execute_phase2({
            "problem_description": "Test"
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Validation error caught: {str(e)[:60]}...")

    # Check DLQ
    dlq = adapter.get_dlq_contents()
    print(f"✓ DLQ size: {len(dlq)}")

    print("\n✓ Error handling tests passed!")


if __name__ == "__main__":
    test_adapter_basic()
    test_adapter_error_handling()
    print("\n" + "="*60)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("="*60)
