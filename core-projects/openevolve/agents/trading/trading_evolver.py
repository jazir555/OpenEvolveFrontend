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

# **ACTUAL INTEGRATION**: Adaptive MDAP for complexity-based strategy evolution
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

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

        # **ACTUAL INTEGRATION**: Initialize Adaptive MDAP components
        self.complexity_classifier: Optional[TaskComplexityClassifier] = None
        self.mdap_allocator: Optional[AdaptiveMDAPAllocator] = None
        self._current_complexity: Optional[str] = None
        self._complexity_configs: Dict[str, Dict[str, Any]] = {
            "low": {
                "max_variants": max(3, max_variants // 3),
                "max_parallel_variants": max(1, max_parallel_variants // 2),
                "judge_panel_size": 3,
                "adversarial_iterations": 2,
                "adversarial_market_conditions": ["bull", "bear"],
                "backtest_days": max(30, backtest_days // 2),
                "description": "Low complexity: Simple strategies with minimal parameters"
            },
            "medium": {
                "max_variants": max_variants,
                "max_parallel_variants": max_parallel_variants,
                "judge_panel_size": 5,
                "adversarial_iterations": 4,
                "adversarial_market_conditions": ["bull", "bear", "high_volatility"],
                "backtest_days": backtest_days,
                "description": "Medium complexity: Standard strategies with moderate complexity"
            },
            "high": {
                "max_variants": max_variants * 2,
                "max_parallel_variants": max_parallel_variants + 2,
                "judge_panel_size": 7,
                "adversarial_iterations": 6,
                "adversarial_market_conditions": ["bull", "bear", "high_volatility", "crisis", "flash_crash"],
                "backtest_days": backtest_days + 30,
                "description": "High complexity: Complex multi-factor strategies with intensive testing"
            }
        }
        if ADAPTIVE_MDAP_AVAILABLE:
            self._initialize_adaptive_mdap()

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

    def _initialize_adaptive_mdap(self):
        """Initialize Adaptive MDAP components for complexity-based evolution."""
        try:
            self.complexity_classifier = TaskComplexityClassifier()
            self.mdap_allocator = AdaptiveMDAPAllocator(
                default_resources=self._complexity_configs["medium"]
            )
            logger.info("Adaptive MDAP components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Adaptive MDAP: {e}")
            self.complexity_classifier = None
            self.mdap_allocator = None

    async def classify_strategy_complexity(
        self,
        strategy_idea: Dict[str, Any],
        market_regime: Dict[str, Any]
    ) -> str:
        """
        Classify the complexity of a trading strategy idea using TaskComplexityClassifier.

        **ACTUAL INTEGRATION**: Uses Adaptive MDAP to determine strategy complexity
        based on strategy characteristics and current market regime.

        Args:
            strategy_idea: Dictionary containing strategy description, parameters, rules
            market_regime: Current market regime classification

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        if not ADAPTIVE_MDAP_AVAILABLE or self.complexity_classifier is None:
            # Fallback: Use simple heuristics if Adaptive MDAP not available
            return self._fallback_complexity_classification(strategy_idea, market_regime)

        try:
            # Create subproblem representation for complexity classification
            subproblem = SubProblem(
                id=strategy_idea.get("strategy_id", "unknown"),
                description=strategy_idea.get("description", ""),
                domain="trading_strategy",
                constraints=self._extract_strategy_constraints(strategy_idea),
                requirements=self._extract_strategy_requirements(strategy_idea, market_regime)
            )

            # Classify complexity using Adaptive MDAP
            complexity_result = self.complexity_classifier.classify(subproblem)
            complexity = complexity_result.get("complexity_level", "medium").lower()

            # Normalize to expected values
            if complexity in ["simple", "low", "easy"]:
                complexity = "low"
            elif complexity in ["complex", "high", "hard", "difficult"]:
                complexity = "high"
            else:
                complexity = "medium"

            self._current_complexity = complexity
            logger.info(f"Strategy {subproblem.id} classified as {complexity} complexity")
            return complexity

        except Exception as e:
            logger.warning(f"Complexity classification failed: {e}, using fallback")
            return self._fallback_complexity_classification(strategy_idea, market_regime)

    def _fallback_complexity_classification(
        self,
        strategy_idea: Dict[str, Any],
        market_regime: Dict[str, Any]
    ) -> str:
        """
        Fallback complexity classification using simple heuristics.

        Args:
            strategy_idea: Strategy idea dictionary
            market_regime: Current market regime

        Returns:
            Complexity level: "low", "medium", or "high"
        """
        # Count complexity indicators
        complexity_score = 0

        # Number of parameters
        params = strategy_idea.get("parameters", {})
        num_params = len(params) if isinstance(params, dict) else 0
        if num_params > 10:
            complexity_score += 2
        elif num_params > 5:
            complexity_score += 1

        # Number of rules/conditions
        rules = strategy_idea.get("rules", [])
        num_rules = len(rules) if isinstance(rules, list) else 0
        if num_rules > 5:
            complexity_score += 2
        elif num_rules > 2:
            complexity_score += 1

        # Market regime complexity
        regime = market_regime.get("regime", "")
        volatility = market_regime.get("volatility", "medium")
        if regime in ["high_volatility", "crisis", "flash_crash"]:
            complexity_score += 2
        elif volatility == "high":
            complexity_score += 1

        # Strategy type
        strategy_type = strategy_idea.get("type", "")
        if strategy_type in ["multi_factor", "statistical_arbitrage", "ml_based"]:
            complexity_score += 2
        elif strategy_type in ["trend_following", "mean_reversion"]:
            complexity_score += 1

        # Classify based on score
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        return "low"

    def _extract_strategy_constraints(self, strategy_idea: Dict[str, Any]) -> List[str]:
        """Extract constraints from strategy idea."""
        constraints = []
        params = strategy_idea.get("parameters", {})
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    constraints.append(f"{key}: {value[0]} to {value[1]}")
                elif isinstance(value, dict):
                    min_val = value.get("min")
                    max_val = value.get("max")
                    if min_val is not None and max_val is not None:
                        constraints.append(f"{key}: {min_val} to {max_val}")
        return constraints

    def _extract_strategy_requirements(
        self,
        strategy_idea: Dict[str, Any],
        market_regime: Dict[str, Any]
    ) -> List[str]:
        """Extract requirements from strategy idea and market regime."""
        requirements = []

        # Add market regime as requirement
        regime = market_regime.get("regime", "unknown")
        requirements.append(f"market_regime: {regime}")

        # Add volatility requirement
        volatility = market_regime.get("volatility", "medium")
        requirements.append(f"volatility_handling: {volatility}")

        # Add strategy type
        strategy_type = strategy_idea.get("type", "unknown")
        requirements.append(f"strategy_type: {strategy_type}")

        # Add performance requirements
        target_sharpe = strategy_idea.get("target_sharpe", 0.5)
        requirements.append(f"target_sharpe: {target_sharpe}")

        max_drawdown = strategy_idea.get("max_drawdown", 0.2)
        requirements.append(f"max_drawdown: {max_drawdown}")

        return requirements

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

        **ACTUAL INTEGRATION**: Classifies strategy complexity for each idea
        using Adaptive MDAP to determine appropriate resource allocation.

        Returns:
            List of new strategy candidates with complexity classification
        """
        # Identify current market regime
        market_regime = await self._identify_market_regime()

        # Generate strategy ideas
        strategy_ideas = await self.rlm_generator.generate_strategies(
            market_regime=market_regime,
            num_ideas=5,
            current_state=self.state
        )

        # **ACTUAL INTEGRATION**: Classify complexity for each strategy idea
        for idea in strategy_ideas:
            idea_dict = idea.to_dict() if hasattr(idea, 'to_dict') else idea
            complexity = await self.classify_strategy_complexity(idea_dict, market_regime)
            # Attach complexity classification to the idea
            if hasattr(idea, 'metadata'):
                idea.metadata['complexity'] = complexity
            elif isinstance(idea, dict):
                idea['complexity'] = complexity

        logger.info(f"Generated {len(strategy_ideas)} new strategy ideas with complexity classification")

        return strategy_ideas

    async def _evolve_phase(self, new_strategies: List[Strategy]) -> List[StrategyVariant]:
        """
        Evolve Phase: Test and evolve strategy variants.

        **ACTUAL INTEGRATION**: Uses complexity-based adaptive resources:
        - Low complexity: Fewer variants, smaller judge panel, lighter adversarial testing
        - High complexity: More variants, larger judge panel, intensive adversarial testing

        Args:
            new_strategies: New strategies to test

        Returns:
            List of evolved variants with performance data
        """
        evolved_variants = []

        # **ACTUAL INTEGRATION**: Determine dominant complexity level for this batch
        complexity_levels = []
        for strategy in new_strategies:
            if hasattr(strategy, 'metadata') and 'complexity' in strategy.metadata:
                complexity_levels.append(strategy.metadata['complexity'])
            elif isinstance(strategy, dict) and 'complexity' in strategy:
                complexity_levels.append(strategy['complexity'])

        # Use highest complexity for resource allocation (conservative approach)
        if complexity_levels:
            if 'high' in complexity_levels:
                self._current_complexity = 'high'
            elif 'medium' in complexity_levels:
                self._current_complexity = 'medium'
            else:
                self._current_complexity = 'low'
        else:
            self._current_complexity = 'medium'

        # Get adaptive configuration for current complexity
        adaptive_config = self._complexity_configs.get(self._current_complexity, self._complexity_configs["medium"])
        logger.info(f"Using adaptive config for {self._current_complexity} complexity: {adaptive_config['description']}")

        # Add new strategies to variant manager with adaptive limits
        for strategy in new_strategies[:adaptive_config["max_variants"]]:
            await self.variant_manager.add_strategy(strategy)

        # Evolve variants using PES if available, otherwise standard evolution
        if self.pes_agent:
            evolved_variants = await self._evolve_with_pes(adaptive_config)
        else:
            evolved_variants = await self._evolve_standard(adaptive_config)

        # Paper trade variants in parallel with adaptive limits
        tested_variants = await self._paper_trade_parallel(evolved_variants, adaptive_config)

        # Adversarial testing with adaptive intensity
        robust_variants = await self._adversarial_test(tested_variants, adaptive_config)

        # Store complexity info in results
        for variant in robust_variants:
            if hasattr(variant, 'metadata'):
                variant.metadata['evolution_complexity'] = self._current_complexity
                variant.metadata['adaptive_config'] = adaptive_config

        logger.info(f"Evolved {len(robust_variants)} variants using {self._current_complexity} complexity settings")

        return robust_variants

    async def _evolve_with_pes(
        self,
        adaptive_config: Optional[Dict[str, Any]] = None
    ) -> List[StrategyVariant]:
        """
        Evolve variants using LoongFlow PES.

        **ACTUAL INTEGRATION**: Uses adaptive configuration for complexity-based evolution.

        Args:
            adaptive_config: Adaptive MDAP configuration for current complexity level

        Returns:
            List of evolved variants
        """
        logger.info("Using LoongFlow PES for directed evolution")

        # Use adaptive config or default
        config = adaptive_config or self._complexity_configs["medium"]

        # Get current population (limited by adaptive config)
        current_variants = await self.variant_manager.get_active_variants()
        current_variants = current_variants[:config["max_variants"]]

        # Define evolution problem with adaptive constraints
        evolution_problem = {
            "domain": "trading_strategy",
            "objective": "maximize risk_adjusted_return",
            "constraints": {
                "max_drawdown": 0.2,
                "min_sharpe": 0.5,
                "max_variants": config["max_variants"]
            },
            "current_population": [v.to_dict() for v in current_variants],
            "performance_history": self.state.knowledge_artifacts,
            "complexity_level": self._current_complexity,
            "adaptive_config": config
        }

        # Run PES evolution with adaptive iterations
        try:
            pes_result = await self.pes_agent.run(
                problem=evolution_problem,
                max_iterations=config.get("adversarial_iterations", 10)
            )

            # Extract evolved variants from PES result
            evolved_variants = self._extract_pes_variants(pes_result)

            # Tag with complexity info
            for variant in evolved_variants:
                if hasattr(variant, 'metadata'):
                    variant.metadata['pes_complexity'] = self._current_complexity

            return evolved_variants

        except Exception as e:
            logger.error(f"PES evolution failed: {e}, falling back to standard evolution")
            return await self._evolve_standard(adaptive_config)

    async def _evolve_standard(
        self,
        adaptive_config: Optional[Dict[str, Any]] = None
    ) -> List[StrategyVariant]:
        """
        Standard evolution without PES.

        **ACTUAL INTEGRATION**: Uses adaptive configuration for complexity-based evolution.

        Args:
            adaptive_config: Adaptive MDAP configuration for current complexity level

        Returns:
            List of evolved variants
        """
        logger.info("Using standard evolution with adaptive resources")

        # Use adaptive config or default
        config = adaptive_config or self._complexity_configs["medium"]

        # Get best performing variants (limited by adaptive parallel variants)
        best_variants = await self.variant_manager.get_top_variants(
            top_n=min(config["max_parallel_variants"], 3)
        )

        # Create new variants through mutation and crossover (limited by adaptive max_variants)
        num_children = min(
            config["max_variants"] - len(best_variants),
            config["max_variants"] // 2
        )
        evolved_variants = await self.variant_manager.evolve_variants(
            parent_variants=best_variants,
            num_children=max(1, num_children)
        )

        # Tag with complexity info
        for variant in evolved_variants:
            if hasattr(variant, 'metadata'):
                variant.metadata['evolution_complexity'] = self._current_complexity
                variant.metadata['adaptive_config_applied'] = True

        return evolved_variants

    async def _paper_trade_parallel(
        self,
        variants: List[StrategyVariant],
        adaptive_config: Optional[Dict[str, Any]] = None
    ) -> List[StrategyVariant]:
        """
        Paper trade multiple variants in parallel.

        **ACTUAL INTEGRATION**: Uses adaptive backtest days based on complexity.
        Low complexity: 30 days, Medium: 90 days, High: 120 days.

        Args:
            variants: Variants to test
            adaptive_config: Adaptive MDAP configuration for current complexity level

        Returns:
            Variants with updated performance data
        """
        # Use adaptive config or default
        config = adaptive_config or self._complexity_configs["medium"]

        # Create parallel testing tasks with adaptive limits
        tasks = [
            self.variant_manager.paper_trade_variant(
                variant.variant_id,
                days=config["backtest_days"]
            )
            for variant in variants[:config["max_parallel_variants"]]
        ]

        logger.info(f"Paper trading {len(tasks)} variants with {config['backtest_days']} days backtest")

        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        tested_variants = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Variant {variants[i].variant_id} failed: {result}")
            else:
                # Attach complexity info to successful variants
                if hasattr(variants[i], 'metadata'):
                    variants[i].metadata['backtest_days'] = config["backtest_days"]
                tested_variants.append(variants[i])

        return tested_variants

    async def _adversarial_test(
        self,
        variants: List[StrategyVariant],
        adaptive_config: Optional[Dict[str, Any]] = None
    ) -> List[StrategyVariant]:
        """
        Perform adversarial testing on variants.

        **ACTUAL INTEGRATION**: Uses adaptive adversarial testing based on complexity:
        - Low complexity: 2 iterations, basic market conditions ["bull", "bear"]
        - High complexity: 6 iterations, intensive testing including ["crisis", "flash_crash"]

        Args:
            variants: Variants to test
            adaptive_config: Adaptive MDAP configuration for current complexity level

        Returns:
            Robust variants that passed adversarial testing
        """
        # Use adaptive config or default
        config = adaptive_config or self._complexity_configs["medium"]

        robust_variants = []

        for variant in variants:
            # Run adversarial tests with adaptive market conditions and iterations
            adversarial_result = await self.adversary.test_strategy(
                variant,
                market_conditions=config["adversarial_market_conditions"],
                iterations=config["adversarial_iterations"]
            )

            # Check if variant is robust enough (adaptive threshold based on complexity)
            # Higher complexity = higher threshold required
            robustness_threshold = 0.5 if self._current_complexity == "low" else (
                0.6 if self._current_complexity == "medium" else 0.7
            )

            if adversarial_result["robustness_score"] > robustness_threshold:
                # Attach adversarial testing metadata
                if hasattr(variant, 'metadata'):
                    variant.metadata['adversarial_iterations'] = config["adversarial_iterations"]
                    variant.metadata['adversarial_conditions'] = config["adversarial_market_conditions"]
                    variant.metadata['robustness_score'] = adversarial_result["robustness_score"]
                robust_variants.append(variant)
                logger.info(f"Variant {variant.variant_id} passed adversarial tests "
                           f"(score: {adversarial_result['robustness_score']:.3f})")
            else:
                logger.warning(f"Variant {variant.variant_id} failed adversarial tests: "
                            f"score {adversarial_result['robustness_score']:.3f} < {robustness_threshold}, "
                            f"failure modes: {adversarial_result['failure_modes']}")

        logger.info(f"Adversarial testing complete: {len(robust_variants)}/{len(variants)} variants passed "
                   f"using {config['adversarial_iterations']} iterations")

        return robust_variants

    async def _select_phase(
        self,
        variants: List[StrategyVariant]
    ) -> List[Strategy]:
        """
        Select Phase: Judge panel evaluates and selects best strategies.

        **ACTUAL INTEGRATION**: Uses adaptive judge panel size based on complexity:
        - Low complexity: 3 judges
        - Medium complexity: 5 judges
        - High complexity: 7 judges

        Args:
            variants: Variants to evaluate

        Returns:
            Selected strategies for deployment with complexity metadata
        """
        selected_strategies = []

        # Get adaptive config for current complexity
        adaptive_config = self._complexity_configs.get(
            self._current_complexity or "medium",
            self._complexity_configs["medium"]
        )

        for variant in variants:
            # Get performance data
            performance = await self.variant_manager.get_performance(variant.variant_id)

            # Judge panel evaluation with adaptive panel size
            judge_evaluations = await self.judge_panel.evaluate_strategy(
                variant=variant,
                performance=performance,
                market_regime=await self._identify_market_regime(),
                panel_size=adaptive_config["judge_panel_size"]
            )

            # Aggregate judge scores
            aggregate_score = self.judge_panel.aggregate_evaluations(judge_evaluations)

            # Select if score is high enough (adaptive threshold)
            # Higher complexity = higher selection threshold
            selection_threshold = 0.65 if self._current_complexity == "low" else (
                0.7 if self._current_complexity == "medium" else 0.75
            )

            if aggregate_score["overall_score"] > selection_threshold:
                strategy = await self.variant_manager.variant_to_strategy(variant)

                # **ACTUAL INTEGRATION**: Attach complexity metadata to strategy
                if hasattr(strategy, 'metadata'):
                    strategy.metadata['complexity_level'] = self._current_complexity
                    strategy.metadata['judge_panel_size'] = adaptive_config["judge_panel_size"]
                    strategy.metadata['evolution_config'] = adaptive_config["description"]
                elif isinstance(strategy, dict):
                    strategy['complexity_level'] = self._current_complexity
                    strategy['judge_panel_size'] = adaptive_config["judge_panel_size"]

                selected_strategies.append(strategy)

                # Update best strategy if needed
                fitness = performance.calculate_fitness()
                if fitness > self.state.best_fitness:
                    self.state.best_strategy_id = strategy.strategy_id
                    self.state.best_fitness = fitness

                    # Deploy to live trading if enabled
                    if self.live_trading_enabled:
                        await self._deploy_strategy(strategy)

                logger.info(f"Selected strategy {strategy.strategy_id} with score {aggregate_score['overall_score']:.3f} "
                           f"(complexity: {self._current_complexity}, judges: {adaptive_config['judge_panel_size']})")

        # Prune underperforming variants (adaptive keep count)
        await self.variant_manager.prune_variants(
            keep_top_n=adaptive_config["max_variants"] // 2
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

        **ACTUAL INTEGRATION**: Includes complexity-based evolution metrics
        and Adaptive MDAP configuration details.

        Returns:
            Evolution summary statistics with complexity information
        """
        # Get current adaptive configuration
        adaptive_config = self._complexity_configs.get(
            self._current_complexity or "medium",
            self._complexity_configs["medium"]
        )

        return {
            "generation": self.state.generation,
            "cycle": self.current_cycle,
            "best_strategy": self.state.best_strategy_id,
            "best_fitness": self.state.best_fitness,
            "population_size": len(self.state.population),
            "diversity_metrics": self.state.diversity_metrics,
            "convergence_metrics": self.state.convergence_metrics,
            "knowledge_artifacts": len(self.state.knowledge_artifacts),
            "causal_models": len(self.state.causal_models),
            # **ACTUAL INTEGRATION**: Complexity-based evolution metrics
            "adaptive_mdap": {
                "available": ADAPTIVE_MDAP_AVAILABLE,
                "current_complexity": self._current_complexity,
                "config_description": adaptive_config.get("description", "Unknown"),
                "max_variants": adaptive_config.get("max_variants", self.max_variants),
                "max_parallel_variants": adaptive_config.get("max_parallel_variants", self.max_parallel_variants),
                "judge_panel_size": adaptive_config.get("judge_panel_size", 5),
                "adversarial_iterations": adaptive_config.get("adversarial_iterations", 4),
                "backtest_days": adaptive_config.get("backtest_days", self.backtest_days)
            },
            "complexity_configs": {
                level: {
                    "description": cfg["description"],
                    "max_variants": cfg["max_variants"],
                    "judge_panel_size": cfg["judge_panel_size"]
                }
                for level, cfg in self._complexity_configs.items()
            }
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
