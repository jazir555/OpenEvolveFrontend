"""
Simple Financial Evolution Example

Demonstrates basic usage of the Financial Evolution Bridge
to evolve crisis-surviving trading strategies.
"""

import asyncio
import logging
from datetime import datetime

from knowledge_engine.finance import (
    FinancialEvolutionAgent,
    EvolutionObjective,
    EvolutionBudget,
    CrisisType,
    Strategy,
    StrategyType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Run simple financial evolution example"""

    logger.info("=" * 60)
    logger.info("Financial Evolution Example")
    logger.info("=" * 60)

    # ========================================================================
    # Step 1: Initialize Agent
    # ========================================================================

    logger.info("\n[Step 1] Initializing Financial Evolution Agent...")

    config = {
        "backtester": {
            "data_source": "CRSP_SIMULATED",  # Use simulated data for demo
            "include_delisted": True,
            "adjust_for_splits": True,
            "adjust_for_dividends": True
        },
        "fitness": {
            "sharpe_weight": 2.0,
            "drawdown_weight": -5.0,
            "wealth_weight": 3.0,
            "crisis_weight": 5.0,  # Prioritize crisis survival
            "delisting_weight": -10.0
        },
        "memory": {
            # Uncomment to persist memory
            # "persistence_path": "./financial_memory.json"
        }
    }

    agent = FinancialEvolutionAgent(config=config)

    logger.info("[OK] Agent initialized")

    # ========================================================================
    # Step 2: Define Evolution Objective
    # ========================================================================

    logger.info("\n[Step 2] Defining evolution objective...")

    objective = EvolutionObjective(
        universe="survivorship_free_equities_2000_2026",
        crisis_periods=[
            ("2007-09-01", "2009-03-31", CrisisType.GFC),
            ("2020-02-01", "2020-04-30", CrisisType.COVID)
        ],
        survival_constraints={
            "max_drawdown": 0.30,  # Max 30% loss
            "min_equity_final": 1.0,  # At least break even
            "delisting_penalty": -1000
        }
    )

    logger.info(f"[OK] Objective defined:")
    logger.info(f"  - Crisis periods: {len(objective.crisis_periods)}")
    logger.info(f"  - Max drawdown: {objective.survival_constraints.max_drawdown:.1%}")
    logger.info(f"  - Min final wealth: {objective.survival_constraints.min_equity_final:.1%}")

    # ========================================================================
    # Step 3: Define Budget
    # ========================================================================

    logger.info("\n[Step 3] Defining budget constraints...")

    budget = EvolutionBudget(
        iterations=10,  # Small for demo
        cost_cap=20,  # $20 max
        strategies_per_iteration=10
    )

    logger.info(f"[OK] Budget defined:")
    logger.info(f"  - Max iterations: {budget.iterations}")
    logger.info(f"  - Cost cap: ${budget.cost_cap}")
    logger.info(f"  - Strategies per iteration: {budget.strategies_per_iteration}")

    # ========================================================================
    # Step 4: Run Evolution
    # ========================================================================

    logger.info("\n[Step 4] Running evolution...")
    logger.info("This may take a few minutes...")

    start_time = datetime.now()

    result = await agent.evolve_strategies(
        objective=objective,
        budget=budget
    )

    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info("[OK] Evolution complete!")

    # ========================================================================
    # Step 5: Analyze Results
    # ========================================================================

    logger.info("\n[Step 5] Analyzing results...")
    logger.info("=" * 60)

    # Summary statistics
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Iterations completed: {result.iterations_completed}")
    logger.info(f"  Total cost: ${result.final_cost:.2f}")
    logger.info(f"  Execution time: {elapsed:.1f} seconds")
    logger.info(f"  Best fitness: {result.best_fitness:.2f}")
    logger.info(f"  Converged: {result.converged}")

    # Best strategies
    logger.info(f"\n🏆 Top {min(3, len(result.best_strategies))} Strategies:")
    for i, strategy in enumerate(result.best_strategies[:3], 1):
        logger.info(f"\n  [{i}] {strategy.strategy_id}")
        logger.info(f"      Type: {strategy.strategy_type}")
        logger.info(f"      Description: {strategy.description}")
        logger.info(f"      Parameters: {strategy.parameters}")

    # Lessons learned
    logger.info(f"\n💡 Lessons Learned: {len(result.lessons_learned)}")

    # Group lessons by crisis
    crisis_lessons = {}
    for lesson in result.lessons_learned:
        if lesson.crisis not in crisis_lessons:
            crisis_lessons[lesson.crisis] = []
        crisis_lessons[lesson.crisis].append(lesson)

    for crisis, lessons in crisis_lessons.items():
        successful = sum(1 for l in lessons if l.successful)
        logger.info(f"\n  {crisis}:")
        logger.info(f"    Total lessons: {len(lessons)}")
        logger.info(f"    Successful: {successful}")
        logger.info(f"    Failed: {len(lessons) - successful}")

        # Show top lesson
        if lessons:
            top_lesson = max(lessons, key=lambda l: abs(l.boost_amount))
            logger.info(f"    Key insight: {top_lesson.lesson}")

    # ========================================================================
    # Step 6: Evaluate Single Strategy (Bonus)
    # ========================================================================

    logger.info("\n[Step 6] Evaluating single strategy (bonus)...")

    if result.best_strategies:
        best_strategy = result.best_strategies[0]

        # Evaluate on full period
        result_detail, score = await agent.evaluate_strategy(
            strategy=best_strategy,
            period="2000-01-01:2026-12-31"
        )

        logger.info(f"[OK] Strategy evaluation complete:")
        logger.info(f"  Sharpe ratio: {result_detail.sharpe_ratio:.2f}")
        logger.info(f"  Max drawdown: {result_detail.max_drawdown:.1%}")
        logger.info(f"  Final wealth: {result_detail.final_wealth:.2f}x")
        logger.info(f"  Total trades: {result_detail.total_trades}")
        logger.info(f"  Win rate: {result_detail.win_rate:.1%}")
        logger.info(f"  Fitness score: {score.total_score:.2f}")

        # Component scores
        logger.info(f"\n  Fitness Components:")
        for component, value in score.components.items():
            logger.info(f"    {component}: {value:.3f}")

    # ========================================================================
    # Step 7: Memory Analysis (Bonus)
    # ========================================================================

    logger.info("\n[Step 7] Memory analysis (bonus)...")

    # Crisis statistics
    stats = agent.memory.get_crisis_statistics()

    if stats:
        logger.info(f"\n📈 Crisis Statistics:")
        for crisis, data in stats.items():
            logger.info(f"\n  {crisis}:")
            logger.info(f"    Total lessons: {data['total_lessons']}")
            logger.info(f"    Success rate: {data['success_rate']:.1%}")
            logger.info(f"    Avg boost: {data['avg_boost']:.3f}")

    # ========================================================================
    # Done!
    # ========================================================================

    logger.info("\n" + "=" * 60)
    logger.info("[OK] Example complete!")
    logger.info("=" * 60)

    # Save results (optional)
    logger.info("\n💾 Tip: Enable memory persistence to save lessons:")
    logger.info('  config["memory"]["persistence_path"] = "./financial_memory.json"')


async def example_single_strategy():
    """Example: Evaluate a single custom strategy"""

    logger.info("\n" + "=" * 60)
    logger.info("Single Strategy Evaluation Example")
    logger.info("=" * 60)

    agent = FinancialEvolutionAgent()

    # Define custom momentum strategy
    strategy = Strategy(
        strategy_id="custom_momentum_6m",
        strategy_type=StrategyType.MOMENTUM,
        parameters={
            "lookback": 6,  # 6-month lookback
            "alpha": 0.005,  # Small base return
            "beta": 1.1  # Slightly above market exposure
        },
        description="6-month momentum strategy with market beta",
        entry_conditions=[
            "6-month return > 0",
            "Above 200-day moving average"
        ],
        exit_conditions=[
            "6-month return < 0",
            "10% stop loss"
        ],
        risk_constraints={
            "max_position_size": 0.05,
            "max_total_exposure": 1.0
        }
    )

    logger.info(f"\nStrategy: {strategy.strategy_id}")
    logger.info(f"Type: {strategy.strategy_type}")
    logger.info(f"Parameters: {strategy.parameters}")

    # Evaluate
    result, score = await agent.evaluate_strategy(
        strategy=strategy,
        period="2000-01-01:2026-12-31"
    )

    logger.info(f"\nResults:")
    logger.info(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"  Max drawdown: {result.max_drawdown:.1%}")
    logger.info(f"  Final wealth: {result.final_wealth:.2f}x")
    logger.info(f"  Fitness: {score.total_score:.2f}")

    logger.info(f"\nFitness breakdown:")
    logger.info(f"  Base score: {score.base_score:.2f}")
    logger.info(f"  Learned boost: {score.learned_boost:.2f}")


if __name__ == "__main__":
    # Run main evolution example
    asyncio.run(main())

    # Run single strategy example
    # asyncio.run(example_single_strategy())
