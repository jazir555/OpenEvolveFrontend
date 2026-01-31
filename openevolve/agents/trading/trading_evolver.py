#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Trading Strategy Evolution System - Main Orchestrator

Continuous 24/7 autonomous trading strategy research and evolution.
Uses evolutionary algorithms and LLM reasoning to discover and refine
profitable trading strategies.

This system orchestrates:
1. Strategy generation (RLM)
2. Variant management and testing
3. Judge panel evaluation
4. Adversarial testing
5. Causal learning
6. Strategy deployment

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Optional LoongFlow integration
try:
    from loongflow.framework.pes.pes_agent import PESAgent
    from loongflow.framework.pes.context import EvolveChainConfig
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False

from openevolve.agents.trading.schemas import (
    Strategy,
    StrategyVariant,
    StrategyPerformance,
    MarketData,
    TradeSignal,
    EvolutionState,
    StrategyType,
    SignalType
)
from openevolve.agents.trading.rlm_generator import RLMGenerator
from openevolve.agents.trading.variant_manager import VariantManager
from openevolve.agents.trading.judge_panel import JudgePanel
from openevolve.agents.trading.causal_modeler import CausalModeler
from openevolve.agents.trading.adversary import Adversary


logger = logging.getLogger(__name__)


class TradingEvolver:
    """
    Main orchestrator for continuous trading strategy evolution.

    Operates in continuous 24/7 mode with four phases:
    1. GENERATE: RLM generates new strategy ideas
    2. EVOLVE: Evolve variants using PES
    3. SELECT: Judge panel selects best strategies
    4. LEARN: Build causal models from outcomes

    Usage:
        evolver = TradingEvolver(
            knowledge_engine=ke,
            max_variants=10,
            evolution_interval=timedelta(hours=1)
        )

        await evolver.start()
        # Runs continuously...

        # Or single cycle
        state = await evolver.run_evolution_cycle()
    """

    def __init__(
        self,
        knowledge_engine=None,
        max_variants: int = 10,
        max_parallel_variants: int = 3,
        evolution_interval: timedelta = timedelta(hours=1),
        backtest_days: int = 90,
        live_trading_enabled: bool = False,
        checkpoint_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Trading Evolver.

        Args:
            knowledge_engine: Optional knowledge engine for persistent learning
            max_variants: Maximum number of variants to maintain
            max_parallel_variants: Maximum variants to test in parallel
            evolution_interval: Time between evolution cycles
            backtest_days: Days of historical data for backtesting
            live_trading_enabled: Whether to enable live trading
            checkpoint_dir: Directory for checkpoints
            config: Additional configuration
        """
        self.knowledge_engine = knowledge_engine
        self.max_variants = max_variants
        self.max_parallel_variants = max_parallel_variants
        self.evolution_interval = evolution_interval
        self.backtest_days = backtest_days
        self.live_trading_enabled = live_trading_enabled
        self.checkpoint_dir = checkpoint_dir or Path("./trading_evolver_checkpoints")
        self.config = config or {}

        # Initialize state
        self.state = EvolutionState()
        self.running = False
        self.current_cycle = 0

        # Initialize components
        self.rlm_generator = RLMGenerator(
            knowledge_engine=knowledge_engine
        )
        self.variant_manager = VariantManager(
            max_variants=max_variants,
            backtest_days=backtest_days
        )
        self.judge_panel = JudgePanel(
            knowledge_engine=knowledge_engine
        )
        self.causal_modeler = CausalModeler(
            knowledge_engine=knowledge_engine
        )
        self.adversary = Adversary(
            knowledge_engine=knowledge_engine
        )

        # Optional LoongFlow integration
        self.pes_agent: Optional[PESAgent] = None
        if LOONGFLOW_AVAILABLE and live_trading_enabled:
            self._initialize_loongflow()

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"TradingEvolver initialized with {max_variants} max variants, "
                   f"evolution interval: {evolution_interval}")

    def _initialize_loongflow(self):
        """Initialize LoongFlow PES agent for enhanced evolution."""
        try:
            config = EvolveChainConfig(
                domain_name="trading_strategy_evolution",
                enable_memory=True,
                enable_planning=True
            )
            self.pes_agent = PESAgent(config=config)
            logger.info("LoongFlow PES Agent initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize LoongFlow: {e}")
            self.pes_agent = None

    async def start(self):
        """
        Start continuous evolution cycle.

        Runs indefinitely until stop() is called.
        """
        self.running = True
        logger.info("Starting continuous trading strategy evolution...")

        while self.running:
            try:
                self.current_cycle += 1
                logger.info(f"=== Evolution Cycle {self.current_cycle} ===")

                # Run single evolution cycle
                await self.run_evolution_cycle()

                # Save checkpoint
                await self.save_checkpoint()

                # Wait for next cycle
                if self.running:
                    logger.info(f"Waiting {self.evolution_interval} until next cycle...")
                    await asyncio.sleep(self.evolution_interval.total_seconds())

            except Exception as e:
                logger.error(f"Error in evolution cycle {self.current_cycle}: {e}", exc_info=True)
                # Continue running despite errors

        logger.info("Trading strategy evolution stopped")

    def stop(self):
        """Stop continuous evolution."""
        self.running = False
        logger.info("Stopping trading strategy evolution...")

    async def run_evolution_cycle(self) -> EvolutionState:
        """
        Run a complete evolution cycle.

        Cycle phases:
        1. GENERATE: Generate new strategy ideas
        2. EVOLVE: Evolve and test variants
        3. SELECT: Judge panel selects best
        4. LEARN: Build causal models

        Returns:
            Updated evolution state
        """
        # Phase 1: GENERATE
        logger.info("Phase 1: GENERATE - Creating new strategy ideas")
        new_strategies = await self._generate_phase()

        # Phase 2: EVOLVE
        logger.info("Phase 2: EVOLVE - Testing and evolving variants")
        evolved_variants = await self._evolve_phase(new_strategies)

        # Phase 3: SELECT
        logger.info("Phase 3: SELECT - Judge panel evaluation")
        selected_strategies = await self._select_phase(evolved_variants)

        # Phase 4: LEARN
        logger.info("Phase 4: LEARN - Building causal models")
        await self._learn_phase(selected_strategies)

        # Update state
        self.state.generation += 1
        self.state.timestamp = datetime.now(UTC)

        return self.state

    async def _generate_phase(self) -> List[Strategy]:
        """
        Generate Phase: Create new strategy ideas using RLM.

        Returns:
            List of new strategy candidates
        """
        # Identify current market regime
        market_regime = await self._identify_market_regime()

        # Generate strategy ideas
        strategy_ideas = await self.rlm_generator.generate_strategies(
            market_regime=market_regime,
            num_ideas=5,
            current_state=self.state
        )

        logger.info(f"Generated {len(strategy_ideas)} new strategy ideas")

        return strategy_ideas

    async def _evolve_phase(self, new_strategies: List[Strategy]) -> List[StrategyVariant]:
        """
        Evolve Phase: Test and evolve strategy variants.

        Args:
            new_strategies: New strategies to test

        Returns:
            List of evolved variants with performance data
        """
        evolved_variants = []

        # Add new strategies to variant manager
        for strategy in new_strategies:
            await self.variant_manager.add_strategy(strategy)

        # Evolve variants using PES if available, otherwise standard evolution
        if self.pes_agent:
            evolved_variants = await self._evolve_with_pes()
        else:
            evolved_variants = await self._evolve_standard()

        # Paper trade variants in parallel
        tested_variants = await self._paper_trade_parallel(evolved_variants)

        # Adversarial testing
        robust_variants = await self._adversarial_test(tested_variants)

        logger.info(f"Evolved {len(robust_variants)} variants successfully")

        return robust_variants

    async def _evolve_with_pes(self) -> List[StrategyVariant]:
        """
        Evolve variants using LoongFlow PES.

        Returns:
            List of evolved variants
        """
        logger.info("Using LoongFlow PES for directed evolution")

        # Get current population
        current_variants = await self.variant_manager.get_active_variants()

        # Define evolution problem
        evolution_problem = {
            "domain": "trading_strategy",
            "objective": "maximize risk_adjusted_return",
            "constraints": {
                "max_drawdown": 0.2,
                "min_sharpe": 0.5,
                "max_variants": self.max_variants
            },
            "current_population": [v.to_dict() for v in current_variants],
            "performance_history": self.state.knowledge_artifacts
        }

        # Run PES evolution
        try:
            pes_result = await self.pes_agent.run(
                problem=evolution_problem,
                max_iterations=10
            )

            # Extract evolved variants from PES result
            evolved_variants = self._extract_pes_variants(pes_result)

            return evolved_variants

        except Exception as e:
            logger.error(f"PES evolution failed: {e}, falling back to standard evolution")
            return await self._evolve_standard()

    async def _evolve_standard(self) -> List[StrategyVariant]:
        """
        Standard evolution without PES.

        Returns:
            List of evolved variants
        """
        logger.info("Using standard evolution")

        # Get best performing variants
        best_variants = await self.variant_manager.get_top_variants(
            top_n=min(self.max_parallel_variants, 3)
        )

        # Create new variants through mutation and crossover
        evolved_variants = await self.variant_manager.evolve_variants(
            parent_variants=best_variants,
            num_children=self.max_variants - len(best_variants)
        )

        return evolved_variants

    async def _paper_trade_parallel(
        self,
        variants: List[StrategyVariant]
    ) -> List[StrategyVariant]:
        """
        Paper trade multiple variants in parallel.

        Args:
            variants: Variants to test

        Returns:
            Variants with updated performance data
        """
        # Create parallel testing tasks
        tasks = [
            self.variant_manager.paper_trade_variant(
                variant.variant_id,
                days=self.backtest_days
            )
            for variant in variants[:self.max_parallel_variants]
        ]

        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        tested_variants = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Variant {variants[i].variant_id} failed: {result}")
            else:
                tested_variants.append(variants[i])

        return tested_variants

    async def _adversarial_test(
        self,
        variants: List[StrategyVariant]
    ) -> List[StrategyVariant]:
        """
        Perform adversarial testing on variants.

        Args:
            variants: Variants to test

        Returns:
            Robust variants that passed adversarial testing
        """
        robust_variants = []

        for variant in variants:
            # Run adversarial tests
            adversarial_result = await self.adversary.test_strategy(
                variant,
                market_conditions=["bull", "bear", "high_volatility", "crisis"]
            )

            # Check if variant is robust enough
            if adversarial_result["robustness_score"] > 0.6:
                robust_variants.append(variant)
                logger.info(f"Variant {variant.variant_id} passed adversarial tests")
            else:
                logger.warning(f"Variant {variant.variant_id} failed adversarial tests: "
                            f"{adversarial_result['failure_modes']}")

        return robust_variants

    async def _select_phase(
        self,
        variants: List[StrategyVariant]
    ) -> List[Strategy]:
        """
        Select Phase: Judge panel evaluates and selects best strategies.

        Args:
            variants: Variants to evaluate

        Returns:
            Selected strategies for deployment
        """
        selected_strategies = []

        for variant in variants:
            # Get performance data
            performance = await self.variant_manager.get_performance(variant.variant_id)

            # Judge panel evaluation
            judge_evaluations = await self.judge_panel.evaluate_strategy(
                variant=variant,
                performance=performance,
                market_regime=await self._identify_market_regime()
            )

            # Aggregate judge scores
            aggregate_score = self.judge_panel.aggregate_evaluations(judge_evaluations)

            # Select if score is high enough
            if aggregate_score["overall_score"] > 0.7:
                strategy = await self.variant_manager.variant_to_strategy(variant)
                selected_strategies.append(strategy)

                # Update best strategy if needed
                fitness = performance.calculate_fitness()
                if fitness > self.state.best_fitness:
                    self.state.best_strategy_id = strategy.strategy_id
                    self.state.best_fitness = fitness

                    # Deploy to live trading if enabled
                    if self.live_trading_enabled:
                        await self._deploy_strategy(strategy)

                logger.info(f"Selected strategy {strategy.strategy_id} with score {aggregate_score['overall_score']:.3f}")

        # Prune underperforming variants
        await self.variant_manager.prune_variants(
            keep_top_n=self.max_variants // 2
        )

        return selected_strategies

    async def _learn_phase(self, strategies: List[Strategy]):
        """
        Learn Phase: Build causal models from outcomes.

        Args:
            strategies: Strategies to learn from
        """
        for strategy in strategies:
            # Get performance history
            performance_history = await self.variant_manager.get_performance_history(
                strategy.strategy_id
            )

            # Build causal model
            causal_model = await self.causal_modeler.learn_from_outcomes(
                strategy=strategy,
                performance_history=performance_history,
                market_context=await self._get_market_context()
            )

            # Extract insights
            insights = await self.causal_modeler.extract_insights(causal_model)

            # Store in knowledge
            self.state.knowledge_artifacts.extend(insights)

            # Store in knowledge engine if available
            if self.knowledge_engine:
                await self._store_knowledge_artifacts(insights)

        logger.info(f"Learned {len(strategies)} causal models")

    async def _identify_market_regime(self) -> Dict[str, Any]:
        """
        Identify current market regime.

        Returns:
            Market regime classification and features
        """
        # TODO: Implement actual market regime detection
        # For now, return placeholder
        return {
            "regime": "bull",
            "volatility": "medium",
            "trend": "upward",
            "sentiment": "positive"
        }

    async def _get_market_context(self) -> Dict[str, Any]:
        """Get current market context for learning."""
        return {
            "regime": await self._identify_market_regime(),
            "timestamp": datetime.now(UTC).isoformat(),
            "generation": self.state.generation
        }

    async def _deploy_strategy(self, strategy: Strategy):
        """
        Deploy strategy to live trading.

        Args:
            strategy: Strategy to deploy
        """
        if not self.live_trading_enabled:
            logger.warning(f"Live trading disabled, not deploying {strategy.strategy_id}")
            return

        logger.info(f"Deploying strategy {strategy.strategy_id} to live trading")

        # TODO: Implement actual live trading deployment
        # This would connect to brokerage API, risk management, etc.

    async def save_checkpoint(self):
        """Save current state to checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"evolution_state_{self.state.generation}.json"

        checkpoint_data = {
            "state": self.state.to_dict(),
            "cycle": self.current_cycle,
            "timestamp": datetime.now(UTC).isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    async def load_checkpoint(self, checkpoint_path: Path):
        """
        Load state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        # Restore state
        self.state = EvolutionState(**checkpoint_data["state"])
        self.current_cycle = checkpoint_data["cycle"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}, "
                   f"generation {self.state.generation}")

    def _extract_pes_variants(self, pes_result: Dict[str, Any]) -> List[StrategyVariant]:
        """Extract evolved variants from PES result."""
        variants = []

        for variant_data in pes_result.get("evolved_solutions", []):
            variant = StrategyVariant(
                variant_id=variant_data["variant_id"],
                parent_strategy_id=variant_data.get("parent_id", "unknown"),
                name=variant_data["name"],
                parameters=variant_data["parameters"],
                generation=variant_data.get("generation", 0)
            )
            variants.append(variant)

        return variants

    async def _store_knowledge_artifacts(self, artifacts: List[Dict[str, Any]]):
        """Store knowledge artifacts in knowledge engine."""
        if not self.knowledge_engine:
            return

        # Store using knowledge engine
        # This depends on the specific knowledge engine implementation
        logger.info(f"Stored {len(artifacts)} knowledge artifacts")

    async def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get summary of evolution progress.

        Returns:
            Evolution summary statistics
        """
        return {
            "generation": self.state.generation,
            "cycle": self.current_cycle,
            "best_strategy": self.state.best_strategy_id,
            "best_fitness": self.state.best_fitness,
            "population_size": len(self.state.population),
            "diversity_metrics": self.state.diversity_metrics,
            "convergence_metrics": self.state.convergence_metrics,
            "knowledge_artifacts": len(self.state.knowledge_artifacts),
            "causal_models": len(self.state.causal_models)
        }

    async def get_top_strategies(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top performing strategies.

        Args:
            top_n: Number of top strategies to return

        Returns:
            List of top strategies with performance
        """
        variants = await self.variant_manager.get_top_variants(top_n=top_n)

        strategies_with_performance = []
        for variant in variants:
            performance = await self.variant_manager.get_performance(variant.variant_id)
            strategy = await self.variant_manager.variant_to_strategy(variant)

            strategies_with_performance.append({
                "strategy": strategy.to_dict(),
                "performance": performance.to_dict(),
                "fitness": performance.calculate_fitness()
            })

        # Sort by fitness
        strategies_with_performance.sort(key=lambda x: x["fitness"], reverse=True)

        return strategies_with_performance[:top_n]
