"""
Standalone test for OpenEvolve-Only Mode (no dependencies)
"""

import asyncio
import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strategy_recommender import (
    EnsembleStrategySelector,
    LoongFlowChecker,
    ProblemCharacteristics,
    EvaluationCost,
    ComplexityLevel,
)


@pytest.mark.asyncio
async def test_loongflow_checker():
    """Test LoongFlow availability checker"""
    print("=" * 60)
    print("TEST: LoongFlowChecker")
    print("=" * 60)

    # Check availability
    is_available = LoongFlowChecker.is_available()
    print(f"[OK] LoongFlow available: {is_available}")

    # Verify returns bool
    assert isinstance(is_available, bool), "Should return boolean"
    print("[OK] Returns boolean type")

    # Test reset
    LoongFlowChecker.reset()
    is_available2 = LoongFlowChecker.is_available()
    print(f"[OK] Reset works: {is_available2}")

    print("\n[PASS] All LoongFlowChecker tests passed!\n")


@pytest.mark.asyncio
async def test_selector_initialization():
    """Test selector initialization with LoongFlow disabled"""
    print("=" * 60)
    print("TEST: Selector Initialization (OpenEvolve-Only)")
    print("=" * 60)

    # Create selector with LoongFlow disabled
    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )

    print(f"[OK] Selector initialized")
    print(f"  - enable_loongflow: {selector.enable_loongflow}")
    print(f"  - loongflow_available: {selector.loongflow_available}")

    assert selector.enable_loongflow is False, "Should be disabled"
    assert selector.loongflow_available is False, "Should not be available"
    print("[OK] LoongFlow properly disabled")

    # Check available modes
    modes = selector.get_available_modes()
    print(f"[OK] Available modes: {modes}")
    assert "pes" not in modes, "PES should not be available"
    assert "qd" in modes, "QD should be available"
    assert "mo" in modes, "MO should be available"
    print("[OK] Mode list correct (no PES)")

    print("\n[PASS] All initialization tests passed!\n")


@pytest.mark.asyncio
async def test_openevolve_rule_based():
    """Test OpenEvolve-only rule-based prediction"""
    print("=" * 60)
    print("TEST: OpenEvolve Rule-Based Prediction")
    print("=" * 60)

    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )

    # Test Case 1: Multi-objective → MO mode
    print("\n1. Multi-objective problem:")
    problem_chars = ProblemCharacteristics(
        domain="finance",
        complexity=ComplexityLevel.HIGH,
        evaluation_cost=EvaluationCost.EXPENSIVE,
        has_multiple_objectives=True,
        requires_diversity=False,
        requires_robustness=False,
        constraint_count=2,
        estimated_iterations=50
    )

    prediction = await selector._openevolve_rule_based(
        problem_chars,
        "finance"
    )

    system_value = prediction.system.value if hasattr(prediction.system, 'value') else prediction.system
    mode_value = prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode

    print(f"   System: {system_value}")
    print(f"   Mode: {mode_value}")
    print(f"   Confidence: {prediction.confidence:.2f}")
    print(f"   Reasoning: {prediction.reasoning}")

    assert system_value == "openevolve", "Should recommend OpenEvolve"
    assert mode_value == "mo", "Should recommend MO for multi-objective"
    print("   [OK] Correctly selects MO mode")

    # Test Case 2: Diversity → QD mode
    print("\n2. Diversity problem:")
    problem_chars = ProblemCharacteristics(
        domain="science",
        complexity=ComplexityLevel.MEDIUM,
        evaluation_cost=EvaluationCost.MODERATE,
        has_multiple_objectives=False,
        requires_diversity=True,
        requires_robustness=False,
        constraint_count=1,
        estimated_iterations=30
    )

    prediction = await selector._openevolve_rule_based(
        problem_chars,
        "science"
    )

    print(f"   System: {prediction.system.value if hasattr(prediction.system, 'value') else prediction.system}")
    print(f"   Mode: {prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode}")
    print(f"   Confidence: {prediction.confidence:.2f}")

    system_value = prediction.system.value if hasattr(prediction.system, 'value') else prediction.system
    assert system_value == "openevolve", "Should recommend OpenEvolve"
    mode_value = prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode
    assert mode_value == "qd", "Should recommend QD for diversity"
    print("   [OK] Correctly selects QD mode")

    # Test Case 3: Robustness → Adversarial mode
    print("\n3. Robustness problem:")
    problem_chars = ProblemCharacteristics(
        domain="engineering",
        complexity=ComplexityLevel.HIGH,
        evaluation_cost=EvaluationCost.EXPENSIVE,
        has_multiple_objectives=False,
        requires_diversity=False,
        requires_robustness=True,
        constraint_count=3,
        estimated_iterations=100
    )

    prediction = await selector._openevolve_rule_based(
        problem_chars,
        "engineering"
    )

    print(f"   System: {prediction.system.value if hasattr(prediction.system, 'value') else prediction.system}")
    print(f"   Mode: {prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode}")
    print(f"   Confidence: {prediction.confidence:.2f}")

    system_value = prediction.system.value if hasattr(prediction.system, 'value') else prediction.system
    assert system_value == "openevolve", "Should recommend OpenEvolve"
    mode_value = prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode
    assert mode_value == "adversarial", "Should recommend Adversarial for robustness"
    print("   [OK] Correctly selects Adversarial mode")

    # Test Case 4: Default → Standard mode
    print("\n4. Default problem:")
    problem_chars = ProblemCharacteristics(
        domain="web",
        complexity=ComplexityLevel.LOW,
        evaluation_cost=EvaluationCost.CHEAP,
        has_multiple_objectives=False,
        requires_diversity=False,
        requires_robustness=False,
        constraint_count=0,
        estimated_iterations=200
    )

    prediction = await selector._openevolve_rule_based(
        problem_chars,
        "web"
    )

    print(f"   System: {prediction.system.value if hasattr(prediction.system, 'value') else prediction.system}")
    print(f"   Mode: {prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode}")
    print(f"   Confidence: {prediction.confidence:.2f}")

    system_value = prediction.system.value if hasattr(prediction.system, 'value') else prediction.system
    assert system_value == "openevolve", "Should recommend OpenEvolve"
    mode_value = prediction.mode.value if hasattr(prediction.mode, 'value') else prediction.mode
    assert mode_value == "standard", "Should recommend Standard as default"
    print("   [OK] Correctly selects Standard mode")

    print("\n[PASS] All rule-based tests passed!\n")


@pytest.mark.asyncio
async def test_full_recommendation():
    """Test full recommendation in OpenEvolve-only mode"""
    print("=" * 60)
    print("TEST: Full Recommendation (OpenEvolve-Only)")
    print("=" * 60)

    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )

    # Test recommendation
    prediction = await selector.recommend_with_ensemble(
        problem_description="Optimize portfolio allocation for risk-adjusted returns",
        domain="finance",
        constraints={
            "objectives": ["maximize_returns", "minimize_risk"],
            "constraints": ["budget_limit"],
            "time_limit_seconds": 60
        },
        enable_loongflow=False
    )

    system, mode = prediction.strategy

    print(f"\nRecommended Strategy:")
    print(f"  System: {system.value if hasattr(system, 'value') else system}")
    print(f"  Mode: {mode.value if hasattr(mode, 'value') else mode}")
    print(f"  Point Estimate: {prediction.point_estimate:.2%}")
    print(f"  Confidence Interval: [{prediction.confidence_interval[0]:.2%}, {prediction.confidence_interval[1]:.2%}]")
    print(f"  Confidence Level: {prediction.confidence_level:.1%}")
    print(f"  Methods Used: {prediction.prediction_methods}")
    print(f"  Agreement: {(1.0 - prediction.disagreement_ratio):.1%}")

    # Verify OpenEvolve-only (check both enum and string)
    system_value = system.value if hasattr(system, 'value') else system
    mode_value = mode.value if hasattr(mode, 'value') else mode

    assert system_value == "openevolve", f"Should recommend OpenEvolve system, got {system_value}"
    assert mode_value in ["qd", "mo", "adversarial", "standard"], f"Should be OpenEvolve mode, got {mode_value}"
    assert "pes" not in str(mode_value), f"Should not recommend PES (LoongFlow), got {mode_value}"

    # Verify prediction structure
    assert prediction.point_estimate >= 0.0, "Point estimate should be non-negative"
    assert prediction.confidence_interval[0] <= prediction.point_estimate, "Lower bound should be below estimate"
    assert prediction.confidence_interval[1] >= prediction.point_estimate, "Upper bound should be above estimate"

    print("\n[OK] Recommendation structure valid")
    print("[OK] OpenEvolve-only mode working correctly")

    print("\n[PASS] Full recommendation test passed!\n")


@pytest.mark.asyncio
async def test_convenience_method():
    """Test convenience method for OpenEvolve-only"""
    print("=" * 60)
    print("TEST: Convenience Method (recommend_openevolve_only)")
    print("=" * 60)

    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=True  # Try to enable
    )

    # Use convenience method
    prediction = await selector.recommend_openevolve_only(
        problem_description="Design robust bridge structure",
        domain="engineering",
        constraints={
            "objectives": ["minimize_weight", "maximize_strength"],
            "safety_critical": True
        }
    )

    system, mode = prediction.strategy

    print(f"\nRecommended Strategy:")
    print(f"  System: {system}")
    print(f"  Mode: {mode}")

    system_value = system.value if hasattr(system, 'value') else system
    mode_value = mode.value if hasattr(mode, 'value') else mode
    # Should be OpenEvolve-only despite enable_loongflow=True
    assert system_value == "openevolve", "Should recommend OpenEvolve"
    assert mode_value in ["qd", "mo", "adversarial", "standard"], "Should be OpenEvolve mode"

    print("\n[OK] Convenience method works correctly")
    print("\n[PASS] Convenience method test passed!\n")


@pytest.mark.asyncio
async def test_cold_start():
    """Test cold start handling in OpenEvolve-only mode"""
    print("=" * 60)
    print("TEST: Cold Start (OpenEvolve-Only)")
    print("=" * 60)

    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )

    problem_chars = ProblemCharacteristics(
        domain="science",
        complexity=ComplexityLevel.MEDIUM,
        evaluation_cost=EvaluationCost.MODERATE,
        has_multiple_objectives=False,
        requires_diversity=True,
        requires_robustness=False,
        constraint_count=1,
        estimated_iterations=30
    )

    prediction = await selector.handle_cold_start(
        problem_chars=problem_chars,
        domain="science",
        enable_loongflow=False
    )

    system, mode = prediction.strategy

    print(f"\nCold Start Recommendation:")
    print(f"  System: {system}")
    print(f"  Mode: {mode}")
    print(f"  Point Estimate: {prediction.point_estimate:.2%}")
    print(f"  Confidence Level: {prediction.confidence_level:.1%}")

    system_value = system.value if hasattr(system, 'value') else system
    mode_value = mode.value if hasattr(mode, 'value') else mode
    # Should still return valid prediction
    assert system_value == "openevolve", "Should recommend OpenEvolve"
    assert prediction.confidence_level < 1.0, "Should have lower confidence in cold start"

    print("\n[OK] Cold start handling works correctly")
    print("\n[PASS] Cold start test passed!\n")


@pytest.mark.asyncio
async def test_mode_determination():
    """Test LoongFlow usage determination logic"""
    print("=" * 60)
    print("TEST: LoongFlow Usage Determination")
    print("=" * 60)

    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )

    # Test 1: Config disabled
    result = selector._determine_loongflow_usage(enable_loongflow=None)
    print(f"1. Config disabled: {result}")
    assert result is False, "Should be False when disabled in config"

    # Test 2: Runtime override to disable
    result = selector._determine_loongflow_usage(enable_loongflow=False)
    print(f"2. Runtime override (False): {result}")
    assert result is False, "Should be False with runtime override"

    # Test 3: Runtime override to enable (should check availability)
    result = selector._determine_loongflow_usage(enable_loongflow=True)
    print(f"3. Runtime override (True): {result}")
    # If LoongFlow unavailable, should be False
    if not LoongFlowChecker.is_available():
        assert result is False, "Should be False when LoongFlow unavailable"

    print("\n[OK] LoongFlow usage determination works correctly")
    print("\n[PASS] Mode determination test passed!\n")


async def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OPENEVOLVE-ONLY MODE TEST SUITE")
    print("=" * 60 + "\n")

    try:
        await test_loongflow_checker()
        await test_selector_initialization()
        await test_openevolve_rule_based()
        await test_full_recommendation()
        await test_convenience_method()
        await test_cold_start()
        await test_mode_determination()

        print("=" * 60)
        print("ALL TESTS PASSED! [PASS]")
        print("=" * 60)
        print("\nOpenEvolve-only mode is working correctly!")
        print("The strategy selector can operate without LoongFlow.")

        return 0

    except AssertionError as e:
        print("\n" + "=" * 60)
        print("TEST FAILED! [FAIL]")
        print("=" * 60)
        print(f"\nError: {e}")
        return 1

    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST ERROR! [FAIL]")
        print("=" * 60)
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
