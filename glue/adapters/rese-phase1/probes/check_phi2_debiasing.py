#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe Φ₂: Metacognitive Reflection (Debiasing) API

Following CLAUDE.md Law of Runtime Truth:
- Trust execution, not documentation
- Verify Φ₂ debiasing works before relying on it
- Test bias identification, antithetical generation, and CBI calculation
"""

import sys
import os
import traceback
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

print("=" * 70)
print("Φ₂: Metacognitive Reflection - API Probe")
print("=" * 70)

# Test 1: Import metacognitive reflector
print("\n[TEST 1] Importing MetacognitiveReflector...")
try:
    from metacognitive_reflector import (
        MetacognitiveReflector,
        DebiasingConfig,
        Hypothesis,
        BiasAnalysis,
        DebiasingResult,
        BiasType,
        Severity,
    )
    print(f"  [PASS] MetacognitiveReflector imported")
except ImportError as e:
    print(f"  [FAIL] Failed to import: {e}")
    sys.exit(1)

# Test 2: Import bias metrics
print("\n[TEST 2] Importing BiasMetricsTracker...")
try:
    from bias_metrics import (
        BiasMetricsTracker,
        BiasMeasurement,
        BiasMetricsSummary,
        BiasTrend,
        calculate_cbi,
        calculate_bias_reduction,
    )
    print(f"  [PASS] BiasMetricsTracker imported")
except ImportError as e:
    print(f"  [FAIL] Failed to import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 3: Load configuration from environment
print("\n[TEST 3] Loading configuration from environment...")
try:
    # Set test environment variables
    os.environ['PHASE1_DEBIASING_ENABLED'] = 'true'
    os.environ['PHASE1_CBI_THRESHOLD'] = '0.5'
    os.environ['PHASE1_ANTITHETICAL_COUNT'] = '3'
    os.environ['PHASE1_DEBIASING_TIMEOUT_MS'] = '5000'

    config = DebiasingConfig.from_env()
    print(f"  [PASS] Configuration loaded")
    print(f"  [INFO] ENABLE_DEBIASING: {config.ENABLE_DEBIASING}")
    print(f"  [INFO] CBI_THRESHOLD: {config.CBI_THRESHOLD}")
    print(f"  [INFO] ANTITHETICAL_COUNT: {config.ANTITHETICAL_COUNT}")
except Exception as e:
    print(f"  [FAIL] Failed to load configuration: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Create reflector instance
print("\n[TEST 4] Creating MetacognitiveReflector instance...")
try:
    reflector = MetacognitiveReflector(config=config)
    print(f"  [PASS] Reflector instance created")
    print(f"  [INFO] Config: {reflector.get_stats()}")
except Exception as e:
    print(f"  [FAIL] Failed to create reflector: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Identify confirmation bias
print("\n[TEST 5] Testing bias identification...")
try:
    hypothesis_biased = Hypothesis(
        id=str(uuid.uuid4()),
        statement="This obviously proves the theory. It clearly demonstrates the mechanism.",
        confidence=0.9,
        assumptions=["Assumption 1"],
    )

    bias_analysis = reflector._identify_directional_bias(hypothesis_biased)

    print(f"  [INFO] Bias Type: {bias_analysis.bias_type.value}")
    print(f"  [INFO] Confidence: {bias_analysis.confidence:.2f}")
    print(f"  [INFO] Severity: {bias_analysis.severity.value}")
    print(f"  [INFO] Directional Language: {bias_analysis.directional_language}")

    assert bias_analysis.bias_type == BiasType.CONFIRMATION, "Should detect confirmation bias"
    assert len(bias_analysis.directional_language) >= 2, "Should detect directional phrases"

    print(f"  [PASS] Bias identification working correctly")
except Exception as e:
    print(f"  [FAIL] Bias identification failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Generate antithetical outcomes
print("\n[TEST 6] Testing antithetical outcome generation...")
try:
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

    print(f"  [INFO] Generated {len(antithetical)} antithetical outcomes")
    for i, outcome in enumerate(antithetical):
        print(f"  [INFO]   {i+1}. {outcome.statement[:60]}... (confidence: {outcome.confidence:.2f})")

    assert len(antithetical) == 3, "Should generate 3 outcomes"
    assert all(h.confidence < hypothesis.confidence for h in antithetical), \
        "Antithetical outcomes should have lower confidence"

    print(f"  [PASS] Antithetical generation working correctly")
except Exception as e:
    print(f"  [FAIL] Antithetical generation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Calculate CBI
print("\n[TEST 7] Testing Confirmation Bias Index calculation...")
try:
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

    print(f"  [INFO] CBI: {cbi:.4f}")
    print(f"  [INFO] P(H|E): {hypothesis.confidence:.2f}")
    print(f"  [INFO] P(H̄|E): {sum(h.confidence for h in antithetical) / len(antithetical):.2f}")

    assert 0.0 <= cbi <= 1.0, "CBI should be in [0,1]"
    assert cbi > 0.3, "Should detect high bias in this example"

    print(f"  [PASS] CBI calculation working correctly")
except Exception as e:
    print(f"  [FAIL] CBI calculation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Apply metacognitive reflection
print("\n[TEST 8] Testing metacognitive reflection application...")
try:
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

    print(f"  [INFO] Original: {hypothesis.statement}")
    print(f"  [INFO] Debiasing: {debiased.statement}")
    print(f"  [INFO] Original confidence: {hypothesis.confidence:.2f}")
    print(f"  [INFO] Debiasing confidence: {debiased.confidence:.2f}")

    assert debiased.confidence < hypothesis.confidence, "Confidence should be reduced"
    assert "obviously" not in debiased.statement.lower(), "Directional language should be removed"

    print(f"  [PASS] Metacognitive reflection working correctly")
except Exception as e:
    print(f"  [FAIL] Metacognitive reflection failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 9: End-to-end debiasing
print("\n[TEST 9] Testing end-to-end debiasing process...")
try:
    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement="This clearly demonstrates that X causes Y",
        confidence=0.9,
        assumptions=["X is necessary", "Y is sufficient"],
    )

    result = reflector.perform_debiasing(
        hypothesis=hypothesis,
        assumptions=[],
        correlation_id=str(uuid.uuid4()),
    )

    print(f"  [INFO] Initial CBI: {result.initial_cbi:.4f}")
    print(f"  [INFO] Final CBI: {result.confirmation_bias_index:.4f}")
    print(f"  [INFO] Bias Reduction: {result.bias_reduction:.2f}%")
    print(f"  [INFO] Antithetical Outcomes: {len(result.antithetical_outcomes)}")
    print(f"  [INFO] Reflections Applied: {result.metacognitive_reflections_applied}")

    assert isinstance(result, DebiasingResult), "Should return DebiasingResult"
    assert result.confirmation_bias_index <= result.initial_cbi, \
        "Final CBI should be <= initial CBI"

    print(f"  [PASS] End-to-end debiasing working correctly")
except Exception as e:
    print(f"  [FAIL] End-to-end debiasing failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 10: Bias metrics tracking
print("\n[TEST 10] Testing bias metrics tracking...")
try:
    tracker = BiasMetricsTracker()

    # Record some measurements
    for epoch in range(1, 4):
        tracker.record_measurement(
            epoch=epoch,
            confirmation_bias_index=0.7 - (epoch * 0.1),
            initial_cbi=0.8,
            bias_reduction=10.0 + (epoch * 5),
            hypotheses_count=5,
            correlation_id=str(uuid.uuid4()),
        )

    summary = tracker.calculate_summary()

    print(f"  [INFO] Total Epochs: {summary.total_epochs}")
    print(f"  [INFO] Current CBI: {summary.current_cbi:.4f}")
    print(f"  [INFO] Average CBI: {summary.average_cbi:.4f}")
    print(f"  [INFO] Trend: {summary.cbi_trend.value}")

    assert summary.total_epochs == 3, "Should record 3 epochs"
    assert summary.cbi_trend == BiasTrend.IMPROVING, "Should detect improving trend"

    print(f"  [PASS] Bias metrics tracking working correctly")
except Exception as e:
    print(f"  [FAIL] Bias metrics tracking failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 11: CBI threshold validation
print("\n[TEST 11] Testing CBI threshold validation...")
try:
    threshold_check = tracker.check_thresholds(0.6)

    print(f"  [INFO] Status: {threshold_check['status']}")
    print(f"  [INFO] CBI: {threshold_check['cbi']:.4f}")
    print(f"  [INFO] Warning Threshold: {threshold_check['warning_threshold']}")
    print(f"  [INFO] Critical Threshold: {threshold_check['critical_threshold']}")

    assert 'status' in threshold_check, "Should have status"
    assert 'alerts' in threshold_check, "Should have alerts"

    print(f"  [PASS] CBI threshold validation working correctly")
except Exception as e:
    print(f"  [FAIL] CBI threshold validation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 12: Utility functions
print("\n[TEST 12] Testing utility functions...")
try:
    # Test CBI calculation
    cbi = calculate_cbi(0.9, [0.3, 0.4, 0.35])
    print(f"  [INFO] calculate_cbi(0.9, [0.3, 0.4, 0.35]) = {cbi:.4f}")
    assert 0.0 <= cbi <= 1.0, "CBI should be in [0,1]"

    # Test bias reduction calculation
    reduction = calculate_bias_reduction(0.8, 0.5)
    print(f"  [INFO] calculate_bias_reduction(0.8, 0.5) = {reduction:.2f}%")
    assert reduction > 0, "Should calculate positive reduction"

    print(f"  [PASS] Utility functions working correctly")
except Exception as e:
    print(f"  [FAIL] Utility functions failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL Φ₂ DEBIASING API PROBES PASSED")
print("=" * 70)
print("\nConclusion: Φ₂ Metacognitive Reflection is fully operational")
print("  - Bias identification: Working")
print("  - Antithetical generation: Working")
print("  - CBI calculation: Working")
print("  - Metacognitive reflection: Working")
print("  - End-to-end debiasing: Working")
print("  - Bias metrics tracking: Working")
print("  - Threshold validation: Working")
print("  - Utility functions: Working")
print("\n✓ Φ₂ is ready for production use")
