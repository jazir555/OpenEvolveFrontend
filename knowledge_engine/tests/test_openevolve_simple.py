"""
Simplified test for OpenEvolve-Only Mode
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strategy_recommender import (
    EnsembleStrategySelector,
    LoongFlowChecker,
    ProblemCharacteristics,
    EvaluationCost,
    ComplexityLevel,
)


def enum_value(enum_obj):
    """Helper to get enum value"""
    return enum_obj.value if hasattr(enum_obj, 'value') else enum_obj


async def main():
    """Run simplified tests"""
    print("\n" + "=" * 60)
    print("OPENEVOLVE-ONLY MODE TEST SUITE")
    print("=" * 60 + "\n")

    # Test 1: LoongFlow Checker
    print("[TEST 1] LoongFlow Checker")
    is_available = LoongFlowChecker.is_available()
    print(f"  LoongFlow available: {is_available}")
    print(f"  Type: {type(is_available)}")
    assert isinstance(is_available, bool)
    print("  [PASS]\n")

    # Test 2: Selector initialization
    print("[TEST 2] Selector Initialization (OpenEvolve-Only)")
    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )
    print(f"  enable_loongflow: {selector.enable_loongflow}")
    print(f"  loongflow_available: {selector.loongflow_available}")
    assert selector.enable_loongflow is False
    assert selector.loongflow_available is False
    print("  [PASS]\n")

    # Test 3: Available modes
    print("[TEST 3] Available Modes")
    modes = selector.get_available_modes()
    print(f"  Modes: {modes}")
    assert "pes" not in modes
    assert "qd" in modes
    assert "mo" in modes
    print("  [PASS]\n")

    # Test 4: OpenEvolve rule-based - Multi-objective
    print("[TEST 4] Rule-Based: Multi-objective")
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

    prediction = await selector._openevolve_rule_based(problem_chars, "finance")
    print(f"  System: {enum_value(prediction.system)}")
    print(f"  Mode: {enum_value(prediction.mode)}")
    print(f"  Confidence: {prediction.confidence:.2f}")

    assert enum_value(prediction.system) == "openevolve"
    assert enum_value(prediction.mode) == "mo"
    print("  [PASS]\n")

    # Test 5: OpenEvolve rule-based - Diversity
    print("[TEST 5] Rule-Based: Diversity")
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

    prediction = await selector._openevolve_rule_based(problem_chars, "science")
    print(f"  System: {enum_value(prediction.system)}")
    print(f"  Mode: {enum_value(prediction.mode)}")

    assert enum_value(prediction.system) == "openevolve"
    assert enum_value(prediction.mode) == "qd"
    print("  [PASS]\n")

    # Test 6: OpenEvolve rule-based - Robustness
    print("[TEST 6] Rule-Based: Robustness")
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

    prediction = await selector._openevolve_rule_based(problem_chars, "engineering")
    print(f"  System: {enum_value(prediction.system)}")
    print(f"  Mode: {enum_value(prediction.mode)}")

    assert enum_value(prediction.system) == "openevolve"
    assert enum_value(prediction.mode) == "adversarial"
    print("  [PASS]\n")

    # Test 7: OpenEvolve rule-based - Default
    print("[TEST 7] Rule-Based: Default")
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

    prediction = await selector._openevolve_rule_based(problem_chars, "web")
    print(f"  System: {enum_value(prediction.system)}")
    print(f"  Mode: {enum_value(prediction.mode)}")

    assert enum_value(prediction.system) == "openevolve"
    assert enum_value(prediction.mode) == "standard"
    print("  [PASS]\n")

    # Test 8: Full recommendation
    print("[TEST 8] Full Recommendation")
    prediction = await selector.recommend_with_ensemble(
        problem_description="Optimize portfolio allocation",
        domain="finance",
        constraints={"objectives": ["maximize_returns", "minimize_risk"]},
        enable_loongflow=False
    )

    # The strategy is actually: ((system_enum, mode_enum), agreement)
    (strategy_system, strategy_mode), agreement = prediction.strategy
    system_val = enum_value(strategy_system)
    mode_val = enum_value(strategy_mode)

    print(f"  Strategy tuple: {prediction.strategy}")
    print(f"  System: {system_val}")
    print(f"  Mode: {mode_val}")
    print(f"  Point Estimate: {prediction.point_estimate:.2%}")
    print(f"  Confidence: {prediction.confidence_level:.1%}")
    print(f"  Agreement: {agreement:.1%}")

    assert system_val == "openevolve"
    assert mode_val in ["qd", "mo", "adversarial", "standard"]
    assert "pes" not in str(mode_val)
    print("  [PASS]\n")

    # Test 9: Convenience method
    print("[TEST 9] Convenience Method (recommend_openevolve_only)")
    prediction = await selector.recommend_openevolve_only(
        problem_description="Design robust bridge",
        domain="engineering",
        constraints={"safety_critical": True}
    )

    (strategy_system, strategy_mode), agreement = prediction.strategy
    system_val = enum_value(strategy_system)
    mode_val = enum_value(strategy_mode)

    print(f"  System: {system_val}")
    print(f"  Mode: {mode_val}")

    assert system_val == "openevolve"
    assert mode_val in ["qd", "mo", "adversarial", "standard"]
    print("  [PASS]\n")

    # Test 10: Cold start
    print("[TEST 10] Cold Start")
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

    # handle_cold_start returns EnsemblePrediction with strategy=(system, mode)
    strategy_system, strategy_mode = prediction.strategy
    system_val = enum_value(strategy_system)
    mode_val = enum_value(strategy_mode)

    print(f"  System: {system_val}")
    print(f"  Mode: {mode_val}")
    print(f"  Point Estimate: {prediction.point_estimate:.2%}")
    print(f"  Confidence: {prediction.confidence_level:.1%}")

    assert system_val == "openevolve"
    assert prediction.confidence_level < 1.0
    print("  [PASS]\n")

    # All tests passed
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nOpenEvolve-only mode is working correctly!")
    print("The strategy selector can operate without LoongFlow.")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
