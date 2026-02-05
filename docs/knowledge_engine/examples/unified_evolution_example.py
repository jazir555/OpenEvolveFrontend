"""
Unified Evolution Knowledge Extraction System - Comprehensive Example

This example demonstrates complete usage of the unified knowledge extraction
system for analyzing OpenEvolve and LoongFlow dual runs.

Scenario: Portfolio Optimization for Finance Domain

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import json
from datetime import datetime, UTC
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.integrations.unified_evolution_integration import (
    UnifiedEvolutionKnowledgeExtractor,
    DualRunAnalysis
)


# ========================================================================
# MOCK DATA GENERATORS
# ========================================================================

def generate_openevolve_finance_result():
    """
    Generate realistic OpenEvolve result for portfolio optimization

    Simulates a MAP-Elites run with 5 islands exploring risk-return space
    """
    return {
        "best_solution": """
def optimize_portfolio(returns, risk_tolerance=0.1):
    import numpy as np

    # Calculate covariance matrix
    cov = np.cov(returns.T)

    # Use equal-risk contribution approach
    n = len(returns[0])
    weights = np.ones(n) / n  # Start with equal weights

    # Iteratively adjust for risk parity
    for _ in range(10):
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        marginal_risk = np.dot(cov, weights) / portfolio_risk
        weights = weights / marginal_risk
        weights = weights / weights.sum()

    return weights
        """,
        "best_fitness": 1.45,  # Sharpe ratio
        "best_iteration": 78,
        "total_iterations": 150,
        "total_evaluations": 1500,  # 10 evaluations per iteration
        "total_time": 450.0,  # 7.5 minutes
        "history": [
            {
                "iteration": i,
                "fitness": 0.8 + 0.65 * (1 - np.exp(-i / 40)),
                "population_diversity": 0.9 - 0.3 * (i / 150)
            }
            for i in range(0, 151, 10)
        ],
        "archive": {
            "coverage": 0.72,
            "occupancy_rate": 0.68,
            "occupancy": {
                "(0,0)": "portfolio_low_risk",
                "(1,1)": "portfolio_balanced",
                "(2,2)": "portfolio_high_return"
            },
            "solutions": [
                {
                    "id": "sol_1",
                    "fitness": 1.45,
                    "features": (3, 4),
                    "characteristics": {"risk": 0.12, "return": 0.18}
                },
                {
                    "id": "sol_2",
                    "fitness": 1.38,
                    "features": (5, 2),
                    "characteristics": {"risk": 0.15, "return": 0.22}
                },
                {
                    "id": "sol_3",
                    "fitness": 1.32,
                    "features": (1, 6),
                    "characteristics": {"risk": 0.09, "return": 0.14}
                }
            ]
        },
        "config": {
            "population_size": 1000,
            "num_islands": 5,
            "feature_dimensions": ["risk", "return"],
            "feature_bins": 10,
            "migration_interval": 50,
            "exploration_ratio": 0.2,
            "exploitation_ratio": 0.7
        },
        "llm_calls": 300,  # 2 per iteration (generate + evaluate)
        "tokens_used": 120000
    }


def generate_loongflow_finance_result():
    """
    Generate realistic LoongFlow PES result for portfolio optimization

    Simulates Plan-Execute-Summarize approach with directed search
    """
    return {
        "best_solution": """
def optimize_portfolio(returns, risk_tolerance=0.1):
    import numpy as np
    from scipy.optimize import minimize

    # Calculate expected returns and covariance
    mu = np.mean(returns, axis=0)
    cov = np.cov(returns.T)

    # Objective: maximize Sharpe ratio
    def objective(weights):
        portfolio_return = np.dot(mu, weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -portfolio_return / portfolio_risk

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        {'type': 'ineq', 'fun': lambda w: w}  # Long-only
    ]

    # Initial guess: equal weights
    n = len(mu)
    x0 = np.ones(n) / n

    # Optimize
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)
    return result.x
        """,
        "best_fitness": 1.52,  # Slightly better Sharpe ratio
        "total_iterations": 50,  # Much fewer iterations!
        "total_evaluations": 600,  # ~60% fewer than OpenEvolve
        "total_time": 350.0,  # ~6 minutes
        "convergence_generation": 42,
        "sample_efficiency": 0.00253,  # fitness per evaluation
        "generations": [
            {
                "plan": {
                    "strategy": "Use mean-variance optimization with scipy",
                    "reasoning": "Markowitz portfolio theory provides optimal risk-return tradeoff",
                    "action_steps": [
                        "Calculate expected returns and covariance matrix",
                        "Define Sharpe ratio maximization objective",
                        "Add constraints for full investment and long-only positions",
                        "Use SLSQP optimizer for constrained optimization"
                    ]
                },
                "execution": {
                    "approach": "Direct optimization with scipy.minimize",
                    "early_stopped": True,
                    "iterations": 2,  # Converged quickly
                    "evaluations_per_iteration": 3
                },
                "summary": {
                    "insight": "Constrained optimization outperforms evolutionary search for this problem",
                    "reflection": "Planning phase identified optimal mathematical approach",
                    "recommendation": "Use analytical methods when problem structure is well-understood"
                }
            }
            for _ in range(10)
        ],
        "evolutionary_tree": {
            "root_id": "root_portfolio",
            "num_generations": 10,
            "branching_factor": 3.0,
            "best_path": ["root", "analytical_approach", "constrained_opt", "final_solution"],
            "solutions": [
                {
                    "id": "final_solution",
                    "fitness": 1.52,
                    "generation": 8,
                    "ancestry": ["root", "analytical_approach", "constrained_opt"]
                }
            ]
        },
        "summaries": [
            {
                "insight": "Planning reduces wasted evaluations by 60%",
                "evidence": "PES required 600 evaluations vs 1500 for traditional EA"
            },
            {
                "insight": "Domain knowledge (portfolio theory) is crucial",
                "evidence": "Analytical solution found in generation 8"
            },
            {
                "insight": "Early stopping on convergence improves efficiency",
                "evidence": "Stopped after 2 iterations once solution found"
            }
        ],
        "metrics": {
            "total_evaluations": 600,
            "best_fitness": 1.52,
            "convergence_generation": 42,
            "improvement_rate": 0.015,
            "sample_efficiency": 0.00253
        },
        "llm_calls": 150,  # 3 per generation (plan + execute + summarize)
        "tokens_used": 95000
    }


# ========================================================================
# DEMO FUNCTIONS
# ========================================================================

async def demo_basic_usage():
    """Demonstrate basic usage of the unified extractor"""
    print("=" * 80)
    print("DEMO 1: Basic Usage")
    print("=" * 80)

    # Initialize extractor
    extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=None)

    # Generate mock results
    oe_result = generate_openevolve_finance_result()
    lf_result = generate_loongflow_finance_result()

    print("\n[*] Running dual-run analysis...")
    print(f"   Domain: Finance")
    print(f"   Problem: Portfolio optimization for maximum Sharpe ratio")

    # Extract and analyze
    analysis = await extractor.extract_dual_run_knowledge(
        openevolve_result=oe_result,
        loongflow_result=lf_result,
        domain="finance",
        problem="Portfolio optimization for maximum Sharpe ratio"
    )

    # Display results
    print(f"\n[OK] Analysis Complete: {analysis.run_id}")

    # Performance comparison
    perf = analysis.performance_comparison
    print(f"\n[STATS] Performance Comparison:")
    print(f"   Overall Winner: {perf.overall_winner.upper()}")
    print(f"   Confidence: {perf.confidence * 100:.1f}%")

    print(f"\n   Convergence Speed:")
    print(f"   - OpenEvolve: {perf.convergence_speed['openevolve']} iterations")
    print(f"   - LoongFlow: {perf.convergence_speed['loongflow']} iterations")
    print(f"   - Ratio: {perf.convergence_speed['ratio']:.2f}x")

    print(f"\n   Solution Quality:")
    print(f"   - OpenEvolve: {perf.solution_quality['openevolve']:.3f} Sharpe")
    print(f"   - LoongFlow: {perf.solution_quality['loongflow']:.3f} Sharpe")

    print(f"\n   Evaluation Efficiency:")
    print(f"   - OpenEvolve: {perf.evaluation_efficiency['openevolve']:.6f}")
    print(f"   - LoongFlow: {perf.evaluation_efficiency['loongflow']:.6f}")
    print(f"   - LoongFlow is {(1/perf.evaluation_efficiency['ratio'] - 1) * 100:.1f}% more efficient")

    return analysis


async def demo_artifact_exploration(analysis: DualRunAnalysis):
    """Demonstrate artifact exploration"""
    print("\n" + "=" * 80)
    print("DEMO 2: Artifact Exploration")
    print("=" * 80)

    print("\n📦 OpenEvolve Artifacts:")
    for artifact in analysis.openevolve_artifacts:
        print(f"\n   Type: {artifact.artifact_type}")
        print(f"   Confidence: {artifact.confidence * 100:.1f}%")

        if artifact.artifact_type == "map_elites_archive":
            print(f"   Archive Coverage: {artifact.content['archive_coverage'] * 100:.1f}%")
            print(f"   Diverse Solutions: {len(artifact.content['diverse_solutions'])}")

    print("\n📦 LoongFlow Artifacts:")
    for artifact in analysis.loongflow_artifacts:
        print(f"\n   Type: {artifact.artifact_type}")
        print(f"   Confidence: {artifact.confidence * 100:.1f}%")

        if artifact.artifact_type == "pes_patterns":
            print(f"   PES Generations: {artifact.content['num_generations']}")
            print(f"   Planning Strategies: {len(artifact.content['planning_strategies'])}")


async def demo_best_practices(analysis: DualRunAnalysis):
    """Demonstrate best practice analysis"""
    print("\n" + "=" * 80)
    print("DEMO 3: Best Practices")
    print("=" * 80)

    print(f"\n💡 Identified Best Practices (Top 5):")
    for i, practice in enumerate(analysis.best_practices[:5], 1):
        print(f"\n   {i}. {practice.practice}")
        print(f"      Source: {practice.source_system}")
        print(f"      Confidence: {practice.confidence * 100:.1f}%")


async def demo_synergy_opportunities(analysis: DualRunAnalysis):
    """Demonstrate synergy opportunity detection"""
    print("\n" + "=" * 80)
    print("DEMO 4: Synergy Opportunities")
    print("=" * 80)

    print(f"\n🔗 Cross-System Synergy Opportunities (Top 5):")

    top_opportunities = sorted(
        analysis.synergy_opportunities,
        key=lambda o: o.priority,
        reverse=True
    )[:5]

    for i, opp in enumerate(top_opportunities, 1):
        print(f"\n   {i}. {opp.description}")
        print(f"      Type: {opp.opportunity_type}")
        print(f"      From: {opp.source_system} -> To: {opp.target_system}")
        print(f"      Expected Improvement: {opp.expected_improvement * 100:.1f}%")
        print(f"      Complexity: {opp.implementation_complexity}")
        print(f"      Priority: {opp.priority:.1f}/100")


async def demo_hybrid_recommendation(analysis: DualRunAnalysis):
    """Demonstrate hybrid strategy recommendation"""
    print("\n" + "=" * 80)
    print("DEMO 5: Hybrid Strategy Recommendation")
    print("=" * 80)

    rec = analysis.hybrid_recommendation

    print(f"\n🎯 Recommended Strategy: {rec.recommended_mode.upper()}")
    print(f"   Confidence: {rec.confidence * 100:.1f}%")
    print(f"   Expected Improvement: {rec.expected_improvement * 100:.1f}%")

    print(f"\n📋 Rationale:")
    print(f"   {rec.rationale}")

    print(f"\n⚙️  Configuration:")
    for key, value in rec.configuration.items():
        print(f"   {key}: {value}")

    print(f"\n[WARN]  Risk Factors:")
    for risk in rec.risk_factors:
        print(f"   - {risk}")


async def demo_serialization(analysis: DualRunAnalysis):
    """Demonstrate serialization and export"""
    print("\n" + "=" * 80)
    print("DEMO 6: Serialization and Export")
    print("=" * 80)

    # Convert to dictionary
    analysis_dict = analysis.to_dict()

    print(f"\n💾 Serialization:")
    print(f"   Keys: {list(analysis_dict.keys())}")

    # Export to JSON
    output_file = Path("dual_run_analysis_example.json")
    with open(output_file, 'w') as f:
        json.dump(analysis_dict, f, indent=2, default=str)

    print(f"   Exported to: {output_file.absolute()}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")


async def demo_knowledge_fusion(analysis: DualRunAnalysis):
    """Demonstrate knowledge fusion"""
    print("\n" + "=" * 80)
    print("DEMO 7: Knowledge Fusion")
    print("=" * 80)

    extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=None)

    # Fuse insights from both systems
    fused_insights = await extractor.fuse_evolutionary_insights(
        analysis.openevolve_artifacts,
        analysis.loongflow_artifacts
    )

    print(f"\n🔮 Fused Insights ({len(fused_insights)} total):")

    for insight in fused_insights:
        print(f"\n   Type: {insight.artifact_type}")
        print(f"   Source: {insight.source_system}")
        print(f"   Confidence: {insight.confidence * 100:.1f}%")

        if "description" in insight.content:
            print(f"   Description: {insight.content['description']}")


# ========================================================================
# MAIN DEMO
# ========================================================================

async def main():
    """Run complete demo"""
    print("\n" + "=" * 80)
    print("     UNIFIED EVOLUTION KNOWLEDGE EXTRACTION SYSTEM - DEMO")
    print("=" * 80)
    print("\nScenario: Portfolio Optimization in Finance Domain")
    print("Comparing OpenEvolve (MAP-Elites) vs LoongFlow (PES)")

    try:
        # Demo 1: Basic usage
        analysis = await demo_basic_usage()

        # Demo 2: Artifact exploration
        await demo_artifact_exploration(analysis)

        # Demo 3: Best practices
        await demo_best_practices(analysis)

        # Demo 4: Synergy opportunities
        await demo_synergy_opportunities(analysis)

        # Demo 5: Hybrid recommendation
        await demo_hybrid_recommendation(analysis)

        # Demo 6: Serialization
        await demo_serialization(analysis)

        # Demo 7: Knowledge fusion
        await demo_knowledge_fusion(analysis)

        # Final summary
        print("\n" + "=" * 80)
        print("     DEMO COMPLETE")
        print("=" * 80)

        print("\n📊 Key Takeaways:")
        print(f"   * LoongFlow achieved {analysis.performance_comparison.evaluation_efficiency['loongflow'] / analysis.performance_comparison.evaluation_efficiency['openevolve']:.1f}x better evaluation efficiency")
        print(f"   * LoongFlow used 60% fewer evaluations")
        print(f"   * {len(analysis.synergy_opportunities)} cross-system synergy opportunities identified")
        print(f"   * Recommended: {analysis.hybrid_recommendation.recommended_mode.upper()} mode for this domain")

        print("\n✨ The unified system successfully:")
        print("   [OK] Extracted knowledge from both evolutionary systems")
        print("   [OK] Compared performance across 6 dimensions")
        print("   [OK] Identified actionable best practices")
        print("   [OK] Detected high-value synergy opportunities")
        print("   [OK] Generated data-driven hybrid strategy recommendation")

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
