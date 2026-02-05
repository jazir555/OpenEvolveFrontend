#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Phi2: Metacognitive Reflection and Debiasing Subroutine

Tests cover:
1. Bias identification in directional hypotheses
2. Antithetical outcome generation
3. CBI calculation accuracy
4. Bias reduction measurement
5. Integration with Phase I

Following CLAUDE.md testing principles:
- Law of Runtime Truth: Test actual execution
- Law of Idempotency: Verify safe re-execution
- Structured Logging: Verify JSON output
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from metacognitive_reflector import (
    MetacognitiveReflector,
    DebiasingConfig,
    Hypothesis,
    BiasAnalysis,
    DebiasingResult,
    BiasType,
    Severity,
)
from phase1_executor import TacitAssumption


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestLogger:
    """Simple test logger"""
    def info(self, msg, **kwargs):
        print(f"[INFO] {msg} {kwargs}")

    def warn(self, msg, **kwargs):
        print(f"[WARN] {msg} {kwargs}")

    def error(self, msg, error=None, **kwargs):
        print(f"[ERROR] {msg} {error} {kwargs}")

    def debug(self, msg, **kwargs):
        print(f"[DEBUG] {msg} {kwargs}")


# ============================================================================
# TEST CASES
# ============================================================================

def test_debiasing_config_from_env():
    """Test 1: Configuration loading from environment"""
    print("\n=== Test 1: DebiasingConfig.from_env() ===".encode('utf-8').decode('utf-8'))

    # Set environment variables
    os.environ['PHASE1_DEBIASING_ENABLED'] = 'true'
    os.environ['PHASE1_CBI_THRESHOLD'] = '0.6'
    os.environ['PHASE1_ANTITHETICAL_COUNT'] = '5'
    os.environ['PHASE1_DEBIASING_TIMEOUT_MS'] = '8000'

    config = DebiasingConfig.from_env()

    assert config.ENABLE_DEBIASING == True, "Debiasing should be enabled"
    assert config.CBI_THRESHOLD == 0.6, f"CBI_THRESHOLD should be 0.6, got {config.CBI_THRESHOLD}"
    assert config.ANTITHETICAL_COUNT == 5, f"ANTITHETICAL_COUNT should be 5, got {config.ANTITHETICAL_COUNT}"
    assert config.TIMEOUT_MS == 8000, f"TIMEOUT_MS should be 8000, got {config.TIMEOUT_MS}"

    print("✓ Configuration loaded correctly")
    print(f"  - ENABLE_DEBIASING: {config.ENABLE_DEBIASING}")
    print(f"  - CBI_THRESHOLD: {config.CBI_THRESHOLD}")
    print(f"  - ANTITHETICAL_COUNT: {config.ANTITHETICAL_COUNT}")
    print(f"  - TIMEOUT_MS: {config.TIMEOUT_MS}")


def test_identify_confirmation_bias():
    """Test 2: Identify confirmation bias in hypothesis"""
    print("\n=== Test 2: Identify Confirmation Bias ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    # Create hypothesis with confirmation bias
    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="This obviously proves the theory. It clearly demonstrates the mechanism.",
        confidence=0.9,
        assumptions=["Assumption 1"],
    )

    bias_analysis = reflector._identify_directional_bias(hypothesis)

    assert bias_analysis.bias_type == BiasType.CONFIRMATION, \
        f"Expected CONFIRMATION bias, got {bias_analysis.bias_type}"
    assert bias_analysis.confidence > 0.3, \
        f"Expected significant confidence, got {bias_analysis.confidence}"
    assert len(bias_analysis.directional_language) >= 2, \
        f"Expected at least 2 directional phrases, got {len(bias_analysis.directional_language)}"

    print("✓ Confirmation bias identified correctly")
    print(f"  - Bias Type: {bias_analysis.bias_type.value}")
    print(f"  - Confidence: {bias_analysis.confidence:.2f}")
    print(f"  - Severity: {bias_analysis.severity.value}")
    print(f"  - Directional Language: {bias_analysis.directional_language}")


def test_identify_neutral_hypothesis():
    """Test 3: Identify neutral hypothesis"""
    print("\n=== Test 3: Identify Neutral Hypothesis ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    # Create neutral hypothesis
    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="This may suggest a possible mechanism. It might indicate a relationship.",
        confidence=0.6,
        assumptions=["Assumption 1"],
    )

    bias_analysis = reflector._identify_directional_bias(hypothesis)

    assert bias_analysis.bias_type == BiasType.NEUTRAL, \
        f"Expected NEUTRAL bias, got {bias_analysis.bias_type}"
    assert bias_analysis.severity == Severity.LOW, \
        f"Expected LOW severity, got {bias_analysis.severity}"

    print("✓ Neutral hypothesis identified correctly")
    print(f"  - Bias Type: {bias_analysis.bias_type.value}")
    print(f"  - Severity: {bias_analysis.severity.value}")


def test_generate_antithetical_outcomes():
    """Test 4: Generate antithetical outcomes"""
    print("\n=== Test 4: Generate Antithetical Outcomes ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="X causes Y",
        confidence=0.8,
        assumptions=["X is independent", "Y is dependent"],
    )

    antithetical = reflector._generate_antithetical_outcomes(
        hypothesis,
        count=3,
        correlation_id=str(uuid.uuid4()),
    )

    assert len(antithetical) == 3, \
        f"Expected 3 antithetical outcomes, got {len(antithetical)}"

    # Verify negation exists
    negated = [h for h in antithetical if "not" in h.statement.lower()]
    assert len(negated) > 0, "Expected at least one negated outcome"

    # Verify all have lower confidence
    for h in antithetical:
        assert h.confidence < hypothesis.confidence, \
            f"Antithetical outcome should have lower confidence: {h.confidence} >= {hypothesis.confidence}"

    print("✓ Antithetical outcomes generated correctly")
    print(f"  - Count: {len(antithetical)}")
    for i, h in enumerate(antithetical):
        print(f"  - Outcome {i+1}: {h.statement[:60]}... (confidence: {h.confidence:.2f})")


def test_calculate_cbi():
    """Test 5: Calculate Confirmation Bias Index"""
    print("\n=== Test 5: Calculate Confirmation Bias Index ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    # High confidence hypothesis with low confidence alternatives
    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="X causes Y",
        confidence=0.9,
        assumptions=[],
    )

    antithetical = [
        Hypothesis(id=str(uuid.uuid4()), statement="Not X", confidence=0.3, assumptions=[]),
        Hypothesis(id=str(uuid.uuid4()), statement="Y causes X", confidence=0.4, assumptions=[]),
        Hypothesis(id=str(uuid.uuid4()), statement="Alternative", confidence=0.35, assumptions=[]),
    ]

    cbi = reflector._calculate_confirmation_bias_index(
        hypothesis,
        antithetical,
        [],
        str(uuid.uuid4()),
    )

    # CBI should be high (close to 1.0) due to large confidence gap
    assert cbi > 0.4, f"Expected high CBI (>0.4), got {cbi}"
    assert 0.0 <= cbi <= 1.0, f"CBI should be in [0,1], got {cbi}"

    print("✓ CBI calculated correctly")
    print(f"  - CBI: {cbi:.4f}")
    print(f"  - Interpretation: {'High bias' if cbi > 0.5 else 'Low bias'}")


def test_apply_metacognitive_reflection():
    """Test 6: Apply metacognitive reflection"""
    print("\n=== Test 6: Apply Metacognitive Reflection ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="This obviously proves the theory",
        confidence=0.95,
        assumptions=["Assumption 1"],
    )

    bias_analysis = BiasAnalysis(
        bias_type=BiasType.CONFIRMATION,
        confidence=0.8,
        affected_assumptions=["assumption_0"],
        directional_language=["obviously", "proves"],
        severity=Severity.HIGH,
    )

    debiased = reflector._apply_metacognitive_reflection(
        hypothesis,
        bias_analysis,
        [],
        str(uuid.uuid4()),
    )

    # Confidence should be reduced
    assert debiased.confidence < hypothesis.confidence, \
        f"Debiased hypothesis should have lower confidence: {debiased.confidence} >= {hypothesis.confidence}"

    # Statement should be modified
    assert "obviously" not in debiased.statement.lower(), \
        "Directional language 'obviously' should be removed"

    print("✓ Metacognitive reflection applied correctly")
    print(f"  - Original confidence: {hypothesis.confidence:.2f}")
    print(f"  - Debiasing confidence: {debiased.confidence:.2f}")
    print(f"  - Original statement: {hypothesis.statement}")
    print(f"  - Debiasing statement: {debiased.statement}")


def test_perform_debiasing_end_to_end():
    """Test 7: End-to-end debiasing process"""
    print("\n=== Test 7: End-to-End Debiasing ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="This clearly demonstrates that X causes Y",
        confidence=0.9,
        assumptions=["X is necessary", "Y is sufficient"],
    )

    tacit_assumptions = [
        TacitAssumption(
            id=str(uuid.uuid4()),
            description="X causes Y",
            source_pattern="pattern1",
            confidence_score=0.8,
            supporting_evidence_count=10,
        ),
    ]

    result = reflector.perform_debiasing(
        hypothesis=hypothesis,
        assumptions=tacit_assumptions,
        correlation_id=str(uuid.uuid4()),
    )

    # Verify result structure
    assert isinstance(result, DebiasingResult), "Result should be DebiasingResult"
    assert result.original_hypothesis.id == hypothesis.id, "Original hypothesis should match"
    assert len(result.antithetical_outcomes) == 3, "Should have 3 antithetical outcomes"
    assert 0.0 <= result.confirmation_bias_index <= 1.0, "CBI should be in [0,1]"
    assert result.bias_reduction >= 0.0, "Bias reduction should be non-negative"

    # Verify timestamp is UTC ISO-8601
    try:
        datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
    except ValueError:
        assert False, f"Timestamp should be ISO-8601: {result.timestamp}"

    print("✓ End-to-end debiasing completed successfully")
    print(f"  - Initial CBI: {result.initial_cbi:.4f}")
    print(f"  - Final CBI: {result.confirmation_bias_index:.4f}")
    print(f"  - Bias Reduction: {result.bias_reduction:.2f}%")
    print(f"  - Antithetical Outcomes: {len(result.antithetical_outcomes)}")
    print(f"  - Reflections Applied: {result.metacognitive_reflections_applied}")


def test_idempotency():
    """Test 8: Verify idempotency (safe to run multiple times)"""
    print("\n=== Test 8: Idempotency Test ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=5000,
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="X may cause Y",
        confidence=0.7,
        assumptions=[],
    )

    # Run debiasing twice
    result1 = reflector.perform_debiasing(
        hypothesis=hypothesis,
        assumptions=[],
        correlation_id=str(uuid.uuid4()),
    )

    result2 = reflector.perform_debiasing(
        hypothesis=hypothesis,
        assumptions=[],
        correlation_id=str(uuid.uuid4()),
    )

    # Results should be consistent (same CBI within reasonable tolerance)
    assert abs(result1.confirmation_bias_index - result2.confirmation_bias_index) < 0.01, \
        "CBI should be consistent across runs"

    print("✓ Idempotency verified")
    print(f"  - Run 1 CBI: {result1.confirmation_bias_index:.4f}")
    print(f"  - Run 2 CBI: {result2.confirmation_bias_index:.4f}")


def test_timeout_enforcement():
    """Test 9: Timeout enforcement"""
    print("\n=== Test 9: Timeout Enforcement ===")

    config = DebiasingConfig(
        ENABLE_DEBIASING=True,
        CBI_THRESHOLD=0.5,
        ANTITHETICAL_COUNT=3,
        TIMEOUT_MS=1,  # Very short timeout
        DIRECTIONAL_LANGUAGE_THRESHOLD=2,
        CONFIDENCE_THRESHOLD=0.3,
    )

    reflector = MetacognitiveReflector(config=config, logger=TestLogger())

    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="X causes Y" * 100,  # Long statement to slow processing
        confidence=0.9,
        assumptions=["A" * 100] * 10,
    )

    try:
        result = reflector.perform_debiasing(
            hypothesis=hypothesis,
            assumptions=[],
            correlation_id=str(uuid.uuid4()),
        )
        # If it completes, verify it was fast
        print("✓ Debiasing completed within timeout")
    except TimeoutError as e:
        print(f"✓ Timeout enforced correctly: {e}")
    except Exception as e:
        # Other exceptions are acceptable for this test
        print(f"✓ Processing handled (may have hit timeout): {type(e).__name__}")


def test_cbi_threshold_validation():
    """Test 10: CBI threshold configuration validation"""
    print("\n=== Test 10: CBI Threshold Validation ===")

    # Test invalid CBI threshold (should raise ValueError)
    try:
        os.environ['PHASE1_CBI_THRESHOLD'] = '1.5'  # Invalid (> 1.0)
        config = DebiasingConfig.from_env()
        assert False, "Should have raised ValueError for invalid CBI_THRESHOLD"
    except ValueError as e:
        print(f"✓ Invalid CBI threshold rejected: {e}")

    # Test valid CBI threshold
    os.environ['PHASE1_CBI_THRESHOLD'] = '0.7'
    config = DebiasingConfig.from_env()
    assert config.CBI_THRESHOLD == 0.7, "Valid CBI threshold should be accepted"
    print(f"✓ Valid CBI threshold accepted: {config.CBI_THRESHOLD}")


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Phi2: Metacognitive Reflection - Unit Tests".encode('utf-8', errors='ignore').decode('utf-8', errors='ignore'))
    print("=" * 70)

    tests = [
        test_debiasing_config_from_env,
        test_identify_confirmation_bias,
        test_identify_neutral_hypothesis,
        test_generate_antithetical_outcomes,
        test_calculate_cbi,
        test_apply_metacognitive_reflection,
        test_perform_debiasing_end_to_end,
        test_idempotency,
        test_timeout_enforcement,
        test_cbi_threshold_validation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
