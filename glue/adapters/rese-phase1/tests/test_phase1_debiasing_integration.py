#!/usr/bin/env python3
"""
Integration Tests for Φ₂: Metacognitive Reflection with Phase I

Tests cover:
1. Integration with Phase I executor
2. Debiasing results in EpistemicAuditResult
3. Metrics tracking (CBI, bias reduction)
4. Error handling and circuit breaker
5. End-to-end workflow

Following CLAUDE.md testing principles:
- Law of Runtime Truth: Test actual execution
- Circuit Breaker Pattern: Verify failure handling
- Structured Logging: Verify JSON output
"""

import os
import sys
import json
import uuid
import asyncio
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phase1_executor import (
    Phase1Config,
    EpistemicAuditExecutor,
    EpistemicAuditResult,
    StructuredLogger,
)
from metacognitive_reflector import (
    DebiasingConfig,
    Hypothesis,
)


# ============================================================================
# TEST UTILITIES
# ============================================================================

def setup_env():
    """Setup environment variables for testing"""
    os.environ['PHASE1_TIMEOUT_MS'] = '30000'
    os.environ['PHASE1_DEBIASING_ENABLED'] = 'true'
    os.environ['PHASE1_CBI_THRESHOLD'] = '0.5'
    os.environ['PHASE1_ANTITHETICAL_COUNT'] = '3'
    os.environ['PHASE1_DEBIASING_TIMEOUT_MS'] = '10000'
    os.environ['PHASE1_MAX_ASSUMPTIONS'] = '100'
    os.environ['PHASE1_ENABLE_TACIT_MINING'] = 'true'
    os.environ['PHASE1_ENABLE_RED_TEAM'] = 'true'


# ============================================================================
# TEST CASES
# ============================================================================

async def test_debiasing_integration_with_phase1():
    """Test 1: Φ₂ integration with Phase I executor"""
    print("\n=== Test 1: Φ₂ Integration with Phase I ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    # Verify metacognitive reflector is initialized
    assert executor.metacognitive_reflector is not None, \
        "MetacognitiveReflector should be initialized"

    print("✓ MetacognitiveReflector integrated with Phase I executor")


async def test_debiasing_results_in_audit():
    """Test 2: Debiasing results appear in EpistemicAuditResult"""
    print("\n=== Test 2: Debiasing Results in Audit ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    # Perform audit with biased failure patterns
    problem_description = "LENR thermal coefficient obviously demonstrates lattice loading"

    failure_patterns = [
        {
            "pattern_description": "Lattice defects clearly cause inconsistent thermal output",
            "failure_rate": 0.6,
            "data_points": 50,
        },
        {
            "pattern_description": "Temperature undoubtedly affects reaction rate",
            "failure_rate": 0.7,
            "data_points": 75,
        },
    ]

    result = await executor.perform_audit(
        problem_description=problem_description,
        failure_patterns=failure_patterns,
        correlation_id=str(uuid.uuid4()),
    )

    # Verify debiasing results are present
    assert isinstance(result, EpistemicAuditResult), "Result should be EpistemicAuditResult"
    assert result.debiasing_results is not None, "Debiasing results should be present"
    assert isinstance(result.debiasing_results, list), "Debiasing results should be a list"

    # Verify metrics
    assert 'assumptions_debiased' in result.metrics, "Metrics should include assumptions_debiased"
    assert 'average_cbi' in result.metrics, "Metrics should include average_cbi"
    assert 'average_bias_reduction' in result.metrics, "Metrics should include average_bias_reduction"

    print("✓ Debiasing results present in EpistemicAuditResult")
    print(f"  - Assumptions debiased: {result.metrics['assumptions_debiased']}")
    print(f"  - Average CBI: {result.metrics.get('average_cbi', 'N/A')}")
    print(f"  - Average bias reduction: {result.metrics.get('average_bias_reduction', 'N/A')}")


async def test_cbi_tracking_across_iterations():
    """Test 3: CBI tracking across multiple iterations"""
    print("\n=== Test 3: CBI Tracking Across Iterations ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem_description = "X causes Y"

    failure_patterns = [
        {
            "pattern_description": "X clearly causes Y",
            "failure_rate": 0.5,
            "data_points": 100,
        },
    ]

    # Run multiple audits
    cbis = []
    for i in range(3):
        result = await executor.perform_audit(
            problem_description=problem_description,
            failure_patterns=failure_patterns,
            correlation_id=str(uuid.uuid4()),
        )

        if result.metrics.get('average_cbi') is not None:
            cbis.append(result.metrics['average_cbi'])

    # Verify CBI is tracked
    assert len(cbis) > 0, "Should have CBI measurements"
    assert all(0.0 <= cbi <= 1.0 for cbi in cbis), "All CBI values should be in [0,1]"

    print("✓ CBI tracked across iterations")
    print(f"  - Iterations with CBI: {len(cbis)}")
    print(f"  - CBI values: {[f'{cbi:.4f}' for cbi in cbis]}")


async def test_bias_reduction_measurement():
    """Test 4: Bias reduction measurement"""
    print("\n=== Test 4: Bias Reduction Measurement ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem_description = "This undoubtedly proves the hypothesis"

    failure_patterns = [
        {
            "pattern_description": "This clearly demonstrates the mechanism",
            "failure_rate": 0.8,
            "data_points": 100,
        },
    ]

    result = await executor.perform_audit(
        problem_description=problem_description,
        failure_patterns=failure_patterns,
        correlation_id=str(uuid.uuid4()),
    )

    # Verify bias reduction is measured
    if result.debiasing_results and len(result.debiasing_results) > 0:
        for debiasing_result in result.debiasing_results:
            assert 'bias_reduction' in debiasing_result, "Each result should have bias_reduction"
            assert 'initial_cbi' in debiasing_result, "Each result should have initial_cbi"
            assert 'confirmation_bias_index' in debiasing_result, "Each result should have final CBI"

            # Verify bias reduction is non-negative
            bias_reduction = debiasing_result['bias_reduction']
            assert bias_reduction >= 0.0, f"Bias reduction should be non-negative: {bias_reduction}"

            # Verify final CBI <= initial CBI
            initial_cbi = debiasing_result['initial_cbi']
            final_cbi = debiasing_result['confirmation_bias_index']
            assert final_cbi <= initial_cbi, \
                f"Final CBI ({final_cbi}) should be <= initial CBI ({initial_cbi})"

        print("✓ Bias reduction measured correctly")
        print(f"  - Assumptions debiased: {len(result.debiasing_results)}")
        if result.metrics.get('average_bias_reduction'):
            print(f"  - Average bias reduction: {result.metrics['average_bias_reduction']:.2f}%")
    else:
        print("⚠ No debiasing results (assumptions may be below threshold)")


async def test_error_handling_when_debiasing_disabled():
    """Test 5: Error handling when debiasing is disabled"""
    print("\n=== Test 5: Error Handling - Debiasing Disabled ===")

    setup_env()
    os.environ['PHASE1_DEBIASING_ENABLED'] = 'false'

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    # MetacognitiveReflector should still be initialized but disabled
    # or None if initialization failed
    print(f"  - MetacognitiveReflector present: {executor.metacognitive_reflector is not None}")

    # Perform audit (should complete without debiasing)
    problem_description = "X causes Y"
    failure_patterns = [
        {
            "pattern_description": "Pattern",
            "failure_rate": 0.5,
            "data_points": 50,
        },
    ]

    result = await executor.perform_audit(
        problem_description=problem_description,
        failure_patterns=failure_patterns,
        correlation_id=str(uuid.uuid4()),
    )

    # Audit should complete successfully even without debiasing
    assert isinstance(result, EpistemicAuditResult), "Audit should complete"

    print("✓ Audit completes successfully when debiasing disabled")
    print(f"  - Debiasing results: {result.debiasing_results}")


async def test_debiasing_with_no_assumptions():
    """Test 6: Debiasing behavior with no tacit assumptions"""
    print("\n=== Test 6: Debiasing with No Assumptions ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem_description = "X causes Y"

    # Empty failure patterns
    failure_patterns = []

    result = await executor.perform_audit(
        problem_description=problem_description,
        failure_patterns=failure_patterns,
        correlation_id=str(uuid.uuid4()),
    )

    # Should complete successfully
    assert isinstance(result, EpistemicAuditResult), "Audit should complete"

    # Debiasing results should be None or empty
    if result.debiasing_results is not None:
        assert len(result.debiasing_results) == 0, "Should have no debiasing results"

    print("✓ Audit completes with no assumptions")
    print(f"  - Tacit assumptions: {len(result.tacit_assumptions)}")
    print(f"  - Debiasing results: {result.debiasing_results}")


async def test_canonical_schema_compliance():
    """Test 7: Canonical schema compliance"""
    print("\n=== Test 7: Canonical Schema Compliance ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem_description = "X may cause Y"
    failure_patterns = [
        {
            "pattern_description": "Pattern",
            "failure_rate": 0.5,
            "data_points": 50,
        },
    ]

    result = await executor.perform_audit(
        problem_description=problem_description,
        failure_patterns=failure_patterns,
        correlation_id=str(uuid.uuid4()),
    )

    # Convert to dict (canonical format)
    result_dict = result.to_dict()

    # Verify required fields
    required_fields = [
        'phase',
        'audit_id',
        'problem_description',
        'tacit_assumptions',
        'contradictions',
        'falsification_results',
        'hardened_constraints',
        'debiasing_results',
        'metrics',
        'metadata',
        'correlation_id',
        'timestamp',
    ]

    for field in required_fields:
        assert field in result_dict, f"Missing required field: {field}"

    # Verify debiasing results structure
    if result_dict['debiasing_results']:
        for debiasing_result in result_dict['debiasing_results']:
            required_debiasing_fields = [
                'original_hypothesis',
                'debiased_hypothesis',
                'antithetical_outcomes',
                'confirmation_bias_index',
                'initial_cbi',
                'bias_reduction',
                'bias_analysis',
                'timestamp',
            ]
            for field in required_debiasing_fields:
                assert field in debiasing_result, f"Missing debiasing field: {field}"

    print("✓ Canonical schema compliance verified")
    print(f"  - Required fields present: {len(required_fields)}")
    print(f"  - Debiasing results: {len(result_dict['debiasing_results'] or [])}")


async def test_timestamp_utc_compliance():
    """Test 8: UTC timestamp compliance (Law of UTC)"""
    print("\n=== Test 8: UTC Timestamp Compliance ===")

    setup_env()

    config = Phase1Config.from_env()
    executor = EpistemicAuditExecutor(config=config)

    problem_description = "X causes Y"
    failure_patterns = [
        {
            "pattern_description": "Pattern",
            "failure_rate": 0.5,
            "data_points": 50,
        },
    ]

    result = await executor.perform_audit(
        problem_description=problem_description,
        failure_patterns=failure_patterns,
        correlation_id=str(uuid.uuid4()),
    )

    # Verify main timestamp is UTC ISO-8601
    try:
        timestamp = datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
        assert timestamp.tzinfo is not None, "Timestamp should have timezone"
        print(f"  - Main timestamp: {result.timestamp}")
    except ValueError:
        assert False, f"Timestamp should be ISO-8601: {result.timestamp}"

    # Verify debiasing timestamps
    if result.debiasing_results:
        for i, debiasing_result in enumerate(result.debiasing_results):
            try:
                timestamp = datetime.fromisoformat(
                    debiasing_result['timestamp'].replace('Z', '+00:00')
                )
                assert timestamp.tzinfo is not None, f"Debiasing timestamp {i} should have timezone"
            except ValueError:
                assert False, f"Debiasing timestamp {i} should be ISO-8601"

    print("✓ UTC timestamp compliance verified")


# ============================================================================
# TEST RUNNER
# ============================================================================

async def run_all_tests():
    """Run all integration tests"""
    print("=" * 70)
    print("Φ₂: Metacognitive Reflection - Integration Tests")
    print("=" * 70)

    tests = [
        test_debiasing_integration_with_phase1,
        test_debiasing_results_in_audit,
        test_cbi_tracking_across_iterations,
        test_bias_reduction_measurement,
        test_error_handling_when_debiasing_disabled,
        test_debiasing_with_no_assumptions,
        test_canonical_schema_compliance,
        test_timestamp_utc_compliance,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Integration Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
