#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Integration Example for Trading Strategy Evolution System

This example demonstrates:
1. Setting up the complete system
2. Running continuous evolution
3. Monitoring progress
4. Extracting insights
5. Deploying strategies

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path

# Import components
from openevolve.agents.trading import (
    TradingEvolver,
    RLMGenerator,
    VariantManager,
    JudgePanel,
    CausalModeler,
    Adversary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_single_cycle():
    """Example: Run a single evolution cycle."""
    logger.info("=== Example 1: Single Evolution Cycle ===")

    # Initialize evolver
    evolver = TradingEvolver(
        max_variants=5,
        max_parallel_variants=3,
        evolution_interval=timedelta(minutes=30),
        backtest_days=90,
        live_trading_enabled=False
    )

    # Run single cycle
    logger.info("Starting evolution cycle...")
    state = await evolver.run_evolution_cycle()

    # Display results
    logger.info(f"Generation: {state.generation}")
    logger.info(f"Best Strategy: {state.best_strategy_id}")
    logger.info(f"Best Fitness: {state.best_fitness:.4f}")
    logger.info(f"Population Size: {len(state.population)}")

    # Get top strategies
    top_strategies = await evolver.get_top_strategies(top_n=3)

    logger.info("\n=== Top 3 Strategies ===")
    for i, strat in enumerate(top_strategies, 1):
        logger.info(f"\n#{i}: {strat['strategy']['name']}")
        logger.info(f"  Type: {strat['strategy']['strategy_type']}")
        logger.info(f"  Fitness: {strat['fitness']:.4f}")
        logger.info(f"  Return: {strat['performance']['total_return']:.2%}")
        logger.info(f"  Sharpe: {strat['performance']['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {strat['performance']['max_drawdown']:.2%}")

    return evolver


async def example_continuous_evolution():
    """Example: Run continuous evolution with monitoring."""
    logger.info("=== Example 2: Continuous Evolution ===")

    evolver = TradingEvolver(
        max_variants=10,
        evolution_interval=timedelta(seconds=30),  # Short for demo
        live_trading_enabled=False
    )

    # Run for 3 cycles
    logger.info("Starting continuous evolution (3 cycles)...")

    for cycle in range(3):
        logger.info(f"\n--- Cycle {cycle + 1} ---")

        await evolver.run_evolution_cycle()

        # Get summary
        summary = await evolver.get_evolution_summary()

        logger.info(f"Generation: {summary['generation']}")
        logger.info(f"Best Fitness: {summary['best_fitness']:.4f}")
        logger.info(f"Knowledge Artifacts: {summary['knowledge_artifacts']}")
        logger.info(f"Causal Models: {summary['causal_models']}")

        # Show top strategy
        top = await evolver.get_top_strategies(top_n=1)
        if top:
            logger.info(f"Top Strategy: {top[0]['strategy']['name']}")
            logger.info(f"  Fitness: {top[0]['fitness']:.4f}")

    return evolver


async def example_component_usage():
    """Example: Using individual components."""
    logger.info("=== Example 3: Component Usage ===")

    # Initialize components
    generator = RLMGenerator()
    manager = VariantManager(max_variants=5)
    panel = JudgePanel()
    adversary = Adversary()
    modeler = CausalModeler()

    # Define market regime
    market_regime = {
        "regime": "bull",
        "volatility": "low",
        "trend": "upward"
    }

    logger.info(f"Market Regime: {market_regime}")

    # Step 1: Generate strategies
    logger.info("\n--- Generating Strategies ---")
    strategies = await generator.generate_strategies(
        market_regime=market_regime,
        num_ideas=3
    )

    for strategy in strategies:
        logger.info(f"  - {strategy.name}: {strategy.description}")

    # Step 2: Add to variant manager and test
    logger.info("\n--- Paper Trading ---")
    for strategy in strategies:
        variant = await manager.add_strategy(strategy)
        logger.info(f"  Testing {variant.name}...")

        performance = await manager.paper_trade_variant(variant.variant_id)

        logger.info(f"    Return: {performance.total_return:.2%}")
        logger.info(f"    Sharpe: {performance.sharpe_ratio:.2f}")
        logger.info(f"    Max DD: {performance.max_drawdown:.2%}")

    # Step 3: Judge panel evaluation
    logger.info("\n--- Judge Panel Evaluation ---")
    variants = await manager.get_active_variants()

    for variant in variants[:2]:  # Evaluate first 2
        performance = await manager.get_performance(variant.variant_id)

        evaluations = await panel.evaluate_strategy(
            variant, performance, market_regime
        )

        aggregate = panel.aggregate_evaluations(evaluations)

        logger.info(f"\n  {variant.name}:")
        logger.info(f"    Overall Score: {aggregate['overall_score']:.3f}")
        logger.info(f"    Consensus: {aggregate['consensus']:.3f}")
        logger.info(f"    Recommendation: {aggregate['recommendation']}")

        if aggregate['concerns']:
            logger.info(f"    Concerns: {aggregate['concerns'][:2]}")

    # Step 4: Adversarial testing
    logger.info("\n--- Adversarial Testing ---")
    top_variant = (await manager.get_top_variants(top_n=1))[0]

    result = await adversary.test_strategy(
        top_variant,
        market_conditions=["bull", "bear", "crisis"]
    )

    logger.info(f"  Robustness Score: {result['robustness_score']:.3f}")
    logger.info(f"  Failure Modes: {len(result['failure_modes'])}")
    logger.info(f"  Recommendations: {result['recommendations'][:2]}")

    # Step 5: Causal learning
    logger.info("\n--- Causal Learning ---")
    performance_history = await manager.get_performance_history(
        variants[0].variant_id
    )

    if performance_history:
        causal_model = await modeler.learn_from_outcomes(
            strategy=strategies[0],
            performance_history=performance_history,
            market_context=market_regime
        )

        insights = await modeler.extract_insights(causal_model)

        logger.info(f"  Insights: {len(insights)}")
        for insight in insights[:3]:
            if insight['type'] == 'causal_insight':
                logger.info(f"    - {insight['insight']}")
                logger.info(f"      Strength: {insight['strength']:.3f}")

    # Step 6: Evolution
    logger.info("\n--- Evolving New Variants ---")
    top_variants = await manager.get_top_variants(top_n=2)

    children = await manager.evolve_variants(
        parent_variants=top_variants,
        num_children=3
    )

    logger.info(f"  Created {len(children)} child variants")
    for child in children:
        logger.info(f"    - {child.name} (gen {child.generation})")


async def example_custom_workflow():
    """Example: Custom workflow with specific requirements."""
    logger.info("=== Example 4: Custom Workflow ===")

    # Custom configuration
    config = {
        "evolution": {
            "max_variants": 8,
            "backtest_days": 180,
            "min_trades": 30
        },
        "risk": {
            "max_drawdown": 0.20,
            "min_sharpe": 1.0,
            "stop_loss_pct": 0.05
        }
    }

    # Initialize with config
    manager = VariantManager(
        max_variants=config["evolution"]["max_variants"],
        backtest_days=config["evolution"]["backtest_days"]
    )

    generator = RLMGenerator()
    panel = JudgePanel()

    # Generate strategies for specific regime
    market_regime = {"regime": "sideways", "volatility": "medium"}

    logger.info(f"Generating strategies for {market_regime['regime']} regime...")

    strategies = await generator.generate_strategies(
        market_regime=market_regime,
        num_ideas=5
    )

    # Filter strategies that meet criteria
    qualified_strategies = []

    for strategy in strategies:
        variant = await manager.add_strategy(strategy)
        performance = await manager.paper_trade_variant(variant.variant_id)

        # Check if meets risk criteria
        if (performance.max_drawdown <= config["risk"]["max_drawdown"] and
            performance.sharpe_ratio >= config["risk"]["min_sharpe"]):

            # Get judge approval
            evaluations = await panel.evaluate_strategy(
                variant, performance, market_regime
            )
            aggregate = panel.aggregate_evaluations(evaluations)

            if aggregate["recommendation"] in ["approve", "conditional"]:
                qualified_strategies.append({
                    "strategy": strategy,
                    "variant": variant,
                    "performance": performance,
                    "judge_score": aggregate["overall_score"]
                })

    logger.info(f"\nQualified Strategies: {len(qualified_strategies)}")

    # Sort by judge score
    qualified_strategies.sort(key=lambda x: x["judge_score"], reverse=True)

    for i, qual in enumerate(qualified_strategies, 1):
        logger.info(f"\n#{i}: {qual['strategy'].name}")
        logger.info(f"  Judge Score: {qual['judge_score']:.3f}")
        logger.info(f"  Return: {qual['performance'].total_return:.2%}")
        logger.info(f"  Sharpe: {qual['performance'].sharpe_ratio:.2f}")

    return qualified_strategies


async def example_knowledge_integration():
    """Example: Using knowledge engine for persistent learning."""
    logger.info("=== Example 5: Knowledge Integration ===")

    # Note: This example requires knowledge engine to be available
    # from knowledge_engine import KnowledgeEngine

    # For demonstration, we'll show the structure

    logger.info("Knowledge engine integration allows:")
    logger.info("  1. Persistent storage of learned causal models")
    logger.info("  2. Cross-run knowledge accumulation")
    logger.info("  3. Strategy lineage tracking")
    logger.info("  4. Performance pattern recognition")

    # Example structure:
    """
    ke = KnowledgeEngine(
        neo4j_uri="bolt://localhost:7687",
        qdrant_host="localhost",
        qdrant_port=6333
    )

    evolver = TradingEvolver(
        knowledge_engine=ke,
        max_variants=10
    )

    # Learnings are automatically persisted
    await evolver.run_evolution_cycle()

    # Knowledge persists across restarts
    # Previous learnings inform new strategy generation
    """


async def main():
    """Run all examples."""
    examples = [
        ("Single Cycle", example_single_cycle),
        ("Continuous Evolution", example_continuous_evolution),
        ("Component Usage", example_component_usage),
        ("Custom Workflow", example_custom_workflow),
        # ("Knowledge Integration", example_knowledge_integration)
    ]

    print("=" * 60)
    print("Trading Strategy Evolution System - Integration Examples")
    print("=" * 60)

    for name, example_func in examples:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}\n")

        try:
            await example_func()
            await asyncio.sleep(1)  # Brief pause between examples
        except Exception as e:
            logger.error(f"Error in {name}: {e}", exc_info=True)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
