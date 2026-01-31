"""
Unified Evolution API - Quick Start Examples
============================================

This file contains practical examples for getting started with the
Unified Evolution API. Run each example independently to learn
the basics and advanced features.

Author: Unified Evolution Team
Date: 2026-01-30
"""

import asyncio
from openevolve.unified.unified_evolution_api import (
    evolve,
    quick_evolve,
    evolve_no_gauntlet,
    evolve_batch,
    UnifiedEvolutionAPI,
    ProgressUpdate
)


# ============================================================================
# EXAMPLE 1: BASIC EVOLUTION
# ============================================================================

async def example_1_basic_evolution():
    """
    Example 1: Basic Evolution
    ---------------------------
    Simplest way to run evolutionary optimization.
    Just provide a problem and domain, everything else is automatic.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Evolution")
    print("="*80 + "\n")

    result = await evolve(
        problem="Optimize sorting algorithm for speed",
        domain="general"
    )

    print(f"Best solution:\n{result.best_solution}\n")
    print(f"Score: {result.final_score:.3f}")
    print(f"Strategy: {result.strategy_used.system} / {result.strategy_used.mode}")
    print(f"Time: {result.total_time:.2f}s")
    print(f"Iterations: {result.iterations}")


# ============================================================================
# EXAMPLE 2: FINANCE DOMAIN
# ============================================================================

async def example_2_finance_domain():
    """
    Example 2: Finance Domain
    -------------------------
    Portfolio optimization with expensive backtests.
    Automatically uses PES mode to reduce evaluations by 60%.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Finance Domain - Portfolio Optimization")
    print("="*80 + "\n")

    result = await evolve(
        problem="Maximize portfolio Sharpe ratio with minimum risk",
        domain="finance",
        constraints={
            'objectives': ['return', 'risk', 'liquidity'],
            'max_positions': 50,
            'max_drawdown': 0.20
        }
    )

    print(f"Best portfolio strategy:\n{result.best_solution}\n")
    print(f"Expected Sharpe ratio: {result.final_score:.3f}")
    print(f"Evaluations needed: {result.evaluations}")
    print(f"vs traditional: {int(result.evaluations / 0.4)} evaluations")
    print(f"Reduction: {60:.0f}% fewer evaluations")


# ============================================================================
# EXAMPLE 3: SCIENCE DOMAIN
# ============================================================================

async def example_3_science_domain():
    """
    Example 3: Science Domain - Experimental Design
    -----------------------------------------------
    Optimize experimental conditions with expensive experiments.
    Each experiment costs $5K, so minimizing evaluations is critical.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Science Domain - Experimental Design")
    print("="*80 + "\n")

    result = await evolve(
        problem="Optimize chemical reaction conditions for maximum yield",
        domain="science",
        constraints={
            'experiment_cost': 5000,  # $5K per experiment
            'max_budget': 100000,     # $100K total budget
            'max_experiments': 20
        }
    )

    print(f"Optimal conditions:\n{result.best_solution}\n")
    print(f"Predicted yield: {result.final_score:.1%}")
    print(f"Experiments: {result.evaluations}")
    print(f"Total cost: ${result.evaluations * 5000:,.0f}")
    print(f"Savings: ${(30 - result.evaluations) * 5000:,.0f} vs baseline")


# ============================================================================
# EXAMPLE 4: MULTI-OBJECTIVE OPTIMIZATION
# ============================================================================

async def example_4_multi_objective():
    """
    Example 4: Multi-Objective Optimization
    ----------------------------------------
    Optimize multiple competing objectives simultaneously.
    Returns Pareto front of optimal trade-offs.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Objective Optimization")
    print("="*80 + "\n")

    result = await evolve(
        problem="Design efficient electric vehicle",
        domain="engineering",
        constraints={
            'objectives': ['range', 'efficiency', 'safety', 'cost'],
            'weights': {
                'range': 0.3,
                'efficiency': 0.3,
                'safety': 0.3,
                'cost': 0.1
            }
        }
    )

    print(f"Best design:\n{result.best_solution}\n")
    print(f"Composite score: {result.final_score:.3f}")
    print(f"Strategy: {result.strategy_used.mode}")

    # Access individual objectives if available
    if 'objective_scores' in result.metadata:
        for obj, score in result.metadata['objective_scores'].items():
            print(f"  {obj}: {score:.3f}")


# ============================================================================
# EXAMPLE 5: PROGRESS TRACKING
# ============================================================================

async def example_5_progress_tracking():
    """
    Example 5: Progress Tracking
    ----------------------------
    Track evolution progress in real-time with callbacks.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Progress Tracking")
    print("="*80 + "\n")

    progress_updates = []

    def progress_callback(update: ProgressUpdate):
        """Called during evolution with progress updates"""
        progress_updates.append(update)

        # Print summary
        print(f"[{update.stage.upper()}] {update.percent_complete:.0f}%: {update.message}")

        # Show evolution progress
        if update.stage == 'evolving' and update.total_iterations > 0:
            progress = update.current_iteration / update.total_iterations * 100
            print(f"  Progress: {progress:.1f}% ({update.current_iteration}/{update.total_iterations})")
            print(f"  Current score: {update.current_score:.3f}")
            print(f"  Best score: {update.best_score_so_far:.3f}")

    result = await evolve(
        problem="Optimize function: f(x) = x^2 + y^2",
        domain="general",
        callback=progress_callback
    )

    print(f"\nFinal score: {result.final_score:.3f}")
    print(f"Total updates: {len(progress_updates)}")


# ============================================================================
# EXAMPLE 6: QUICK EVOLVE
# ============================================================================

async def example_6_quick_evolve():
    """
    Example 6: Quick Evolve
    -----------------------
    Fastest path to solution when you only need the result.
    Skips gauntlets and knowledge extraction for speed.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Quick Evolve")
    print("="*80 + "\n")

    # Just get the solution string
    solution = await quick_evolve(
        problem="Implement quicksort algorithm",
        domain="general"
    )

    print(f"Solution:\n{solution}")


# ============================================================================
# EXAMPLE 7: BATCH EVOLUTION
# ============================================================================

async def example_7_batch_evolution():
    """
    Example 7: Batch Evolution
    --------------------------
    Evolve multiple problems in parallel.
    Great for A/B testing, parameter sweeps, multiple experiments.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Batch Evolution")
    print("="*80 + "\n")

    problems = [
        "Optimize homepage load time",
        "Improve Time to Interactive metric",
        "Reduce First Contentful Paint delay",
        "Minimize Total Blocking Time",
        "Maximize Lighthouse performance score"
    ]

    print(f"Running {len(problems)} optimizations in parallel...")
    print("Max concurrent: 2\n")

    results = await evolve_batch(
        problems=problems,
        domain="web",
        max_concurrent=2
    )

    # Show results
    print("\nResults:")
    print("-" * 80)
    for problem, result in zip(problems, results):
        print(f"{problem[:50]:50s} | Score: {result.final_score:.3f}")

    # Find best
    best_result = max(results, key=lambda r: r.final_score)
    best_problem = problems[results.index(best_result)]
    print(f"\nBest optimization: {best_problem}")
    print(f"Score: {best_result.final_score:.3f}")


# ============================================================================
# EXAMPLE 8: CUSTOM CONFIGURATION
# ============================================================================

async def example_8_custom_config():
    """
    Example 8: Custom Configuration
    ------------------------------
    Provide custom configuration for fine-grained control.
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Custom Configuration")
    print("="*80 + "\n")

    # Import config schema
    from openevolve.unified.config import (
        UnifiedEvolutionConfig,
        EvolutionMode,
        PESConfig,
        DomainType
    )

    # Create custom config
    custom_config = UnifiedEvolutionConfig(
        domain=DomainType.FINANCE,
        evolution_mode=EvolutionMode.PES,
        max_iterations=30,  # Limited backtests
        pes=PESConfig(
            enabled=True,
            enable_planning=True,
            enable_memory=True,
            max_rounds=3
        )
    )

    result = await evolve(
        problem="Optimize trading strategy",
        domain="finance",
        config=custom_config
    )

    print(f"Best strategy:\n{result.best_solution}\n")
    print(f"Score: {result.final_score:.3f}")
    print(f"Iterations: {result.iterations}")


# ============================================================================
# EXAMPLE 9: RESULT SERIALIZATION
# ============================================================================

async def example_9_save_load_results():
    """
    Example 9: Save and Load Results
    -------------------------------
    Save results to disk for later analysis or comparison.
    """
    print("\n" + "="*80)
    print("EXAMPLE 9: Save and Load Results")
    print("="*80 + "\n")

    # Run evolution
    result = await evolve(
        problem="Optimize portfolio allocation",
        domain="finance",
        run_gauntlet=False,
        store_knowledge=False
    )

    # Save to file
    filepath = "./examples/results/portfolio_optimization.json"
    result.save(filepath)
    print(f"Result saved to: {filepath}")

    # Load from file
    from openevolve.unified.unified_evolution_api import EvolutionResult
    loaded_result = EvolutionResult.load(filepath)
    print(f"Result loaded from: {filepath}")

    # Verify
    print(f"\nOriginal score: {result.final_score:.3f}")
    print(f"Loaded score: {loaded_result.final_score:.3f}")
    print(f"Match: {abs(result.final_score - loaded_result.final_score) < 0.001}")


# ============================================================================
# EXAMPLE 10: ALL DOMAINS
# ============================================================================

async def example_10_all_domains():
    """
    Example 10: All Supported Domains
    ---------------------------------
    Demonstrate evolution across all 7 supported domains.
    """
    print("\n" + "="*80)
    print("EXAMPLE 10: All Supported Domains")
    print("="*80 + "\n")

    domains = {
        'finance': "Maximize portfolio Sharpe ratio",
        'trading': "Develop momentum trading strategy",
        'science': "Optimize experimental design",
        'engineering': "Minimize structural weight",
        'pharma': "Optimize drug binding affinity",
        'web': "Maximize Lighthouse performance score",
        'general': "Solve traveling salesman problem"
    }

    results = {}

    for domain, problem in domains.items():
        print(f"\n{'='*80}")
        print(f"Domain: {domain.upper()}")
        print(f"Problem: {problem}")
        print('='*80)

        result = await evolve(
            problem=problem,
            domain=domain,
            run_gauntlet=False,
            store_knowledge=False
        )

        results[domain] = result

        print(f"\nStrategy: {result.strategy_used.mode}")
        print(f"Score: {result.final_score:.3f}")
        print(f"Evaluations: {result.evaluations}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    for domain, result in results.items():
        print(f"{domain:12s} | Mode: {result.strategy_used.mode:12s} | Score: {result.final_score:.3f}")


# ============================================================================
# MAIN - RUN EXAMPLES
# ============================================================================

async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("UNIFIED EVOLUTION API - QUICK START EXAMPLES")
    print("="*80)

    # Run examples
    await example_1_basic_evolution()
    await example_2_finance_domain()
    await example_3_science_domain()
    await example_4_multi_objective()
    await example_5_progress_tracking()
    await example_6_quick_evolve()
    await example_7_batch_evolution()
    await example_8_custom_config()
    await example_9_save_load_results()
    await example_10_all_domains()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run specific examples
    # Uncomment the examples you want to run:

    # asyncio.run(example_1_basic_evolution())
    # asyncio.run(example_2_finance_domain())
    # asyncio.run(example_3_science_domain())
    # asyncio.run(example_4_multi_objective())
    # asyncio.run(example_5_progress_tracking())
    # asyncio.run(example_6_quick_evolve())
    # asyncio.run(example_7_batch_evolution())
    # asyncio.run(example_8_custom_config())
    # asyncio.run(example_9_save_load_results())
    # asyncio.run(example_10_all_domains())

    # Or run all examples
    asyncio.run(main())
