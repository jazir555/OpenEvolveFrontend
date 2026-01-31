"""
Example Usage of Strategy Recommender

This file demonstrates real-world usage of the AI-powered strategy recommender
for all 6 target domains.
"""

import asyncio
from knowledge_engine.core.strategy_recommender import (
    StrategyRecommender,
    recommend_evolutionary_strategy
)


# ============================================================================
# EXAMPLE 1: Finance Domain - Portfolio Optimization
# ============================================================================

async def finance_example():
    """Example: Portfolio allocation optimization"""
    print("=" * 80)
    print("EXAMPLE 1: Finance - Portfolio Optimization")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = """
    Optimize portfolio allocation across 5 technology stocks.
    Maximize Sharpe ratio (return/risk).
    Constraints: No asset > 40% allocation, min 5% per asset.
    Use 3-year backtest for evaluation.
    """

    constraints = {
        "objectives": ["return", "risk", "sharpe_ratio"],
        "constraints": [
            "no_short_selling",
            "max_allocation_0.4",
            "min_allocation_0.05"
        ],
        "time_limit_seconds": 300  # 5 min per backtest
    }

    recommendation = await recommender.recommend_strategy(
        problem, "finance", constraints
    )

    print(f"\nRecommended: {recommendation.recommended_system} - {recommendation.recommended_mode}")
    print(f"Confidence: {recommendation.confidence:.1%}")
    print(f"\nReasoning:")
    print(f"  {recommendation.reasoning.primary_reason}")
    print(f"\nExpected Performance:")
    print(f"  Iterations: {recommendation.expected_performance.expected_iterations}")
    print(f"  Time: {recommendation.expected_performance.expected_time_seconds:.1f}s")
    print(f"  Success: {recommendation.expected_performance.success_probability:.1%}")

    return recommendation


# ============================================================================
# EXAMPLE 2: Trading Domain - Strategy Development
# ============================================================================

async def trading_example():
    """Example: Trading strategy development"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Trading - Strategy Development")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = """
    Develop algorithmic trading strategy for S&P 500.
    Use technical indicators (RSI, MACD, moving averages).
    Test across bull, bear, and sideways market regimes.
    Need robustness to market regime changes.
    """

    constraints = {
        "objectives": ["total_return", "sharpe_ratio", "max_drawdown"],
        "safety_critical": True,  # Financial safety
        "time_limit_seconds": 600  # 10 min per backtest
    }

    recommendation = await recommender.recommend_strategy(
        problem, "trading", constraints
    )

    print(f"\nRecommended: {recommendation.recommended_system} - {recommendation.recommended_mode}")
    print(f"Reason: {recommendation.reasoning.primary_reason}")

    if recommendation.recommended_mode == "adversarial":
        print("\nNote: Adversarial mode recommended for robustness testing")
        print("  - Red team: Simulates adverse market conditions")
        print("  - Blue team: Develops robust strategy")

    return recommendation


# ============================================================================
# EXAMPLE 3: Science Domain - Experimental Design
# ============================================================================

async def science_example():
    """Example: Scientific experiment optimization"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Science - Experimental Design")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = """
    Optimize chemical reaction conditions for maximum yield.
    Variables: Temperature (100-500°C), Pressure (1-100 atm), Catalyst (0-10%).
    Each experiment requires molecular dynamics simulation (~15 min).
    Goal: Maximize yield while minimizing cost.
    """

    constraints = {
        "objectives": ["yield", "cost"],
        "constraints": ["max_temperature_500"],
        "time_limit_seconds": 900  # 15 min simulation
    }

    recommendation = await recommender.recommend_strategy(
        problem, "science", constraints
    )

    print(f"\nRecommended: {recommendation.recommended_system} - {recommendation.recommended_mode}")
    print(f"\nEvaluation Cost: VERY EXPENSIVE (simulations)")
    print(f"PES Advantage: ~60% fewer experiments needed")
    print(f"  Expected: {recommendation.expected_performance.expected_iterations} experiments")
    print(f"  vs Standard: ~{int(recommendation.expected_performance.expected_iterations / 0.4)} experiments")

    return recommendation


# ============================================================================
# EXAMPLE 4: Engineering Domain - Structural Optimization
# ============================================================================

async def engineering_example():
    """Example: Engineering design optimization"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Engineering - Structural Optimization")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = """
    Optimize truss bridge design for minimum weight.
    Must support 1000kg load with safety factor of 2.0.
    Design variables: Beam lengths, cross-sections, material.
    Each design requires FEA simulation.
    """

    constraints = {
        "objectives": ["weight", "safety_factor"],
        "constraints": [
            "max_stress_250_mpa",
            "max_deflection_10mm"
        ],
        "time_limit_seconds": 600,  # 10 min FEA
        "safety_critical": True
    }

    recommendation = await recommender.recommend_strategy(
        problem, "engineering", constraints
    )

    print(f"\nRecommended: {recommendation.recommended_system} - {recommendation.recommended_mode}")
    print(f"\nSafety Critical: YES")
    print(f"Robustness Testing: {recommendation.problem_analysis.requires_robustness}")

    # Check if adversarial in alternatives
    for alt in recommendation.alternatives:
        if alt.mode == "adversarial":
            print(f"\nAlternative: {alt.system} - {alt.mode}")
            print(f"  Reason: {alt.reason}")
            print(f"  When to use: {alt.when_to_use}")

    return recommendation


# ============================================================================
# EXAMPLE 5: Pharma Domain - Molecular Optimization
# ============================================================================

async def pharma_example():
    """Example: Molecular structure optimization"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Pharma - Molecular Optimization")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = """
    Optimize small molecule structure for drug target.
    Target: Kinase inhibitor for cancer treatment.
    Objectives: Maximize binding affinity, minimize toxicity.
    Each evaluation requires molecular docking simulation (~20 min).
    """

    constraints = {
        "objectives": ["binding_affinity", "toxicity", "solubility"],
        "constraints": ["lipinski_rule_of_5"],
        "time_limit_seconds": 1200  # 20 min docking
    }

    recommendation = await recommender.recommend_strategy(
        problem, "pharma", constraints
    )

    print(f"\nRecommended: {recommendation.recommended_system} - {recommendation.recommended_mode}")
    print(f"\nDiverse Solutions: {recommendation.problem_analysis.requires_diversity}")
    print(f"  QD Mode explores diverse chemical space")

    if recommendation.recommended_mode == "qd":
        print(f"\nQD Configuration:")
        print(f"  Grid resolution: {recommendation.config_overrides.get('grid_resolution', 10)}")
        print(f"  Archive size: {recommendation.config_overrides.get('archive_size', 1000)}")

    return recommendation


# ============================================================================
# EXAMPLE 6: Web Domain - Landing Page Optimization
# ============================================================================

async def web_example():
    """Example: Web design optimization"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Web - Landing Page Optimization")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = """
    Optimize landing page design for conversion.
    Variables: Button color, placement, headline text, hero image.
    Use Google Lighthouse for performance scoring.
    """

    constraints = {
        "objectives": ["conversion_rate", "performance"],
        "time_limit_seconds": 5  # Very fast evaluation
    }

    recommendation = await recommender.recommend_strategy(
        problem, "web", constraints
    )

    print(f"\nRecommended: {recommendation.recommended_system} - {recommendation.recommended_mode}")
    print(f"\nEvaluation Cost: CHEAP (Lighthouse takes ~5 seconds)")
    print(f"Expected iterations: {recommendation.expected_performance.expected_iterations}")
    print(f"  Can afford many iterations with fast evaluation")

    return recommendation


# ============================================================================
# EXAMPLE 7: Learning from Results
# ============================================================================

async def learning_example():
    """Example: Learning from completed evolutionary runs"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Learning from Results")
    print("=" * 80)

    recommender = StrategyRecommender()

    # Get initial recommendation
    problem = "Optimize portfolio allocation"
    rec1 = await recommender.recommend_strategy(
        problem, "finance", {"time_limit_seconds": 300}
    )

    print(f"\nInitial Recommendation: {rec1.recommended_mode}")
    print(f"Confidence: {rec1.confidence:.1%}")

    # Simulate running evolutionary algorithm
    print("\nRunning evolutionary algorithm...")
    await asyncio.sleep(1)  # Simulate work

    # Learn from result
    result = {
        "run_id": "finance_run_001",
        "domain": "finance",
        "strategy_used": rec1.recommended_mode,
        "mode_used": rec1.recommended_mode,
        "complexity": "high",
        "final_score": 0.88,  # Actual achieved score
        "predicted_score": 0.82,  # Was this predicted?
        "iterations": 28,
        "evaluations": 28,
        "diversity_score": 0.65,
        "evaluation_cost": "expensive"
    }

    await recommender.learn_from_run(result)
    print(f"Learned from run: Score = {result['final_score']}")
    print(f"Prediction accuracy: {abs(result['final_score'] - result['predicted_score']):.3f}")

    # Get new recommendation (should be more informed)
    rec2 = await recommender.recommend_strategy(
        problem, "finance", {"time_limit_seconds": 300}
    )

    print(f"\nUpdated Recommendation: {rec2.recommended_mode}")
    print(f"Confidence: {rec2.confidence:.1%}")

    # Show learning impact
    if len(recommender.recommendation_accuracy) > 0:
        avg_accuracy = sum(recommender.recommendation_accuracy) / len(recommender.recommendation_accuracy)
        print(f"\nRecommender Accuracy: {avg_accuracy:.1%}")


# ============================================================================
# EXAMPLE 8: Comparing All Strategies
# ============================================================================

async def comparison_example():
    """Example: Compare all strategies for a problem"""
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Strategy Comparison")
    print("=" * 80)

    recommender = StrategyRecommender()

    problem = "Optimize portfolio allocation with multiple objectives"
    constraints = {
        "objectives": ["return", "risk", "liquidity"],
        "time_limit_seconds": 300
    }

    recommendation = await recommender.recommend_strategy(
        problem, "finance", constraints
    )

    print(f"\nProblem: {problem}")
    print(f"\nStrategy Ranking:")

    for i, strategy in enumerate(recommendation.ranking[:5], 1):
        print(f"\n{i}. {strategy.system.upper()} - {strategy.mode.upper()}")
        print(f"   Score: {strategy.score:.1f}/100")
        print(f"   Confidence: {strategy.confidence:.1%}")
        print(f"   Pros:")
        for pro in strategy.pros[:2]:
            print(f"     + {pro}")
        if strategy.cons:
            print(f"   Cons:")
            for con in strategy.cons[:2]:
                print(f"     - {con}")

    print(f"\n✅ SELECTED: {recommendation.recommended_system} - {recommendation.recommended_mode}")


# ============================================================================
# MAIN RUNNER
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("STRATEGY RECOMMENDER - REAL-WORLD EXAMPLES")
    print("=" * 80)

    # Run examples
    await finance_example()
    await trading_example()
    await science_example()
    await engineering_example()
    await pharma_example()
    await web_example()
    await learning_example()
    await comparison_example()

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. PES (LoongFlow) best for expensive evaluations")
    print("2. QD (OpenEvolve) best for diversity exploration")
    print("3. MO (OpenEvolve) best for multiple objectives")
    print("4. Adversarial (OpenEvolve) best for robustness")
    print("5. Standard (OpenEvolve) best for simple problems")
    print("6. Recommender learns from past runs to improve accuracy")


if __name__ == "__main__":
    asyncio.run(main())
