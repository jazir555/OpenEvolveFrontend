#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variant Manager

Manages multiple trading strategy variants in parallel.
Handles paper trading, performance tracking, pruning, and hybridization.

Key capabilities:
- Maintain multiple strategy variants
- Parallel paper trading
- Performance tracking and comparison
- Variant pruning (eliminate underperformers)
- Variant hybridization (combine successful features)

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import logging
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np
from collections import defaultdict

from openevolve.agents.trading.schemas import (
    Strategy,
    StrategyVariant,
    StrategyPerformance,
    MarketData,
    TradeSignal,
    SignalType
)


logger = logging.getLogger(__name__)


class VariantManager:
    """
    Manages trading strategy variants.

    Handles the lifecycle of strategy variants:
    1. Creation from base strategies
    2. Paper trading and evaluation
    3. Performance tracking
    4. Pruning underperformers
    5. Hybridization of top performers

    Usage:
        manager = VariantManager(max_variants=10, backtest_days=90)

        await manager.add_strategy(strategy)
        variants = await manager.get_active_variants()
        performance = await manager.get_performance(variant_id)
    """

    def __init__(
        self,
        max_variants: int = 10,
        backtest_days: int = 90,
        min_trades: int = 20,
        pruning_threshold: float = 0.5,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Variant Manager.

        Args:
            max_variants: Maximum number of variants to maintain
            backtest_days: Days of historical data for paper trading
            min_trades: Minimum trades required for valid evaluation
            pruning_threshold: Performance percentile for pruning (0-1)
            config: Additional configuration
        """
        self.max_variants = max_variants
        self.backtest_days = backtest_days
        self.min_trades = min_trades
        self.pruning_threshold = pruning_threshold
        self.config = config or {}

        # Variant storage
        self.variants: Dict[str, StrategyVariant] = {}
        self.variant_performances: Dict[str, StrategyPerformance] = {}
        self.variant_history: Dict[str, List[Dict]] = defaultdict(list)

        # Strategy to variants mapping
        self.strategy_variants: Dict[str, List[str]] = defaultdict(list)

        logger.info(f"VariantManager initialized with max_variants={max_variants}")

    async def add_strategy(self, strategy: Strategy) -> StrategyVariant:
        """
        Add a strategy and create initial variant.

        Args:
            strategy: Strategy to add

        Returns:
            Created variant
        """
        # Check if we have room for new variant
        if len(self.variants) >= self.max_variants:
            # Prune worst performing variant
            await self.prune_variants(keep_top_n=self.max_variants - 1)

        # Create initial variant
        variant = StrategyVariant(
            variant_id=str(uuid.uuid4()),
            parent_strategy_id=strategy.strategy_id,
            name=f"{strategy.name}_v0",
            parameters=strategy.parameters.copy(),
            generation=0,
            status="initialized"
        )

        # Store variant
        self.variants[variant.variant_id] = variant
        self.strategy_variants[strategy.strategy_id].append(variant.variant_id)

        logger.info(f"Added strategy {strategy.strategy_id} as variant {variant.variant_id}")

        return variant

    async def paper_trade_variant(
        self,
        variant_id: str,
        days: Optional[int] = None
    ) -> StrategyPerformance:
        """
        Paper trade a variant on historical data.

        Args:
            variant_id: Variant to test
            days: Days of historical data (uses default if None)

        Returns:
            Performance metrics
        """
        variant = self.variants.get(variant_id)
        if not variant:
            raise ValueError(f"Variant {variant_id} not found")

        days = days or self.backtest_days

        logger.info(f"Paper trading variant {variant_id} for {days} days")

        # Simulate paper trading
        # In real implementation, this would:
        # 1. Fetch historical market data
        # 2. Generate signals using variant's strategy
        # 3. Simulate trades
        # 4. Calculate performance metrics

        # Placeholder implementation
        performance = await self._simulate_paper_trading(variant, days)

        # Store performance
        self.variant_performances[variant_id] = performance
        self.variant_history[variant_id].append({
            "timestamp": datetime.now(UTC).isoformat(),
            "performance": performance.to_dict()
        })

        # Update variant status
        variant.status = "tested"

        logger.info(f"Variant {variant_id} paper trading complete. "
                   f"Sharpe: {performance.sharpe_ratio:.3f}, "
                   f"Return: {performance.total_return:.3f}")

        return performance

    async def _simulate_paper_trading(
        self,
        variant: StrategyVariant,
        days: int
    ) -> StrategyPerformance:
        """
        Simulate paper trading (placeholder).

        In real implementation, this would execute actual backtesting.
        """
        # Simulate performance with some randomness
        # Better parameters lead to better performance
        np.random.seed(hash(variant.variant_id) % (2**32))

        # Base performance
        base_return = np.random.normal(0.1, 0.2)  # 10% average return, 20% std
        base_sharpe = np.random.normal(1.0, 0.5)

        # Adjust based on parameters
        param_quality = sum(1.0 for v in variant.parameters.values() if v > 0)
        quality_multiplier = 1.0 + (param_quality * 0.1)

        total_return = base_return * quality_multiplier
        sharpe_ratio = base_sharpe * min(quality_multiplier, 2.0)
        max_drawdown = abs(np.random.normal(0.15, 0.05))

        # Generate other metrics
        win_rate = np.random.beta(5, 3)  # Biased toward winning
        profit_factor = 1.5 + np.random.exponential(0.5)

        # Calculate derived metrics
        sortino_ratio = sharpe_ratio * 1.2
        volatility = abs(total_return / sharpe_ratio) if sharpe_ratio > 0 else 0.2
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Create performance object
        performance = StrategyPerformance(
            strategy_id=variant.variant_id,
            period=f"{days}_days",
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=0.02 + np.random.exponential(0.01),
            avg_loss=-0.015 - np.random.exponential(0.005),
            trades=np.random.randint(20, 200),
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            information_ratio=np.random.normal(0.5, 0.3)
        )

        return performance

    async def get_active_variants(self) -> List[StrategyVariant]:
        """Get all active variants."""
        return [v for v in self.variants.values() if v.status != "pruned"]

    async def get_top_variants(self, top_n: int = 5) -> List[StrategyVariant]:
        """
        Get top performing variants.

        Args:
            top_n: Number of top variants to return

        Returns:
            List of top variants sorted by fitness
        """
        # Get all tested variants
        tested_variants = [
            variant for variant in self.variants.values()
            if variant.status == "tested" and variant.variant_id in self.variant_performances
        ]

        # Sort by fitness
        sorted_variants = sorted(
            tested_variants,
            key=lambda v: self.variant_performances[v.variant_id].calculate_fitness(),
            reverse=True
        )

        return sorted_variants[:top_n]

    async def get_performance(self, variant_id: str) -> StrategyPerformance:
        """
        Get performance for a variant.

        Args:
            variant_id: Variant ID

        Returns:
            Performance metrics
        """
        if variant_id not in self.variant_performances:
            raise ValueError(f"No performance data for variant {variant_id}")

        return self.variant_performances[variant_id]

    async def get_performance_history(
        self,
        variant_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for a variant.

        Args:
            variant_id: Variant ID

        Returns:
            List of historical performance snapshots
        """
        return self.variant_history.get(variant_id, [])

    async def evolve_variants(
        self,
        parent_variants: List[StrategyVariant],
        num_children: int
    ) -> List[StrategyVariant]:
        """
        Evolve new variants from parents.

        Uses mutation and crossover to create new variants.

        Args:
            parent_variants: Parent variants
            num_children: Number of children to create

        Returns:
            List of new child variants
        """
        logger.info(f"Evolving {num_children} variants from {len(parent_variants)} parents")

        children = []

        for i in range(num_children):
            # Select parents
            if len(parent_variants) >= 2 and i % 2 == 0:
                # Crossover
                parent1 = parent_variants[i % len(parent_variants)]
                parent2 = parent_variants[(i + 1) % len(parent_variants)]
                child = await self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = parent_variants[i % len(parent_variants)]
                child = await self._mutate(parent)

            children.append(child)

        # Add children to manager
        for child in children:
            if len(self.variants) < self.max_variants:
                self.variants[child.variant_id] = child
                self.strategy_variants[child.parent_strategy_id].append(child.variant_id)

        logger.info(f"Created {len(children)} child variants")

        return children

    async def _mutate(self, parent: StrategyVariant) -> StrategyVariant:
        """
        Create mutated variant.

        Args:
            parent: Parent variant

        Returns:
            Mutated child variant
        """
        # Mutate parameters
        mutated_params = parent.parameters.copy()

        for param_name, param_value in mutated_params.items():
            # Add random mutation
            mutation_rate = np.random.normal(0, 0.1)
            mutated_params[param_name] = param_value * (1 + mutation_rate)

            # Ensure positive
            if mutated_params[param_name] < 0:
                mutated_params[param_name] = param_value * 0.5

        # Create child variant
        child = StrategyVariant(
            variant_id=str(uuid.uuid4()),
            parent_strategy_id=parent.parent_strategy_id,
            name=f"{parent.name}_mut",
            parameters=mutated_params,
            generation=parent.generation + 1,
            status="initialized",
            created_at=datetime.now(UTC)
        )

        return child

    async def _crossover(
        self,
        parent1: StrategyVariant,
        parent2: StrategyVariant
    ) -> StrategyVariant:
        """
        Create child variant via crossover.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Child variant
        """
        # Crossover parameters
        child_params = {}

        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())

        for param_name in all_params:
            val1 = parent1.parameters.get(param_name, 0)
            val2 = parent2.parameters.get(param_name, 0)

            # Average of parents
            child_params[param_name] = (val1 + val2) / 2

        # Create child variant
        child = StrategyVariant(
            variant_id=str(uuid.uuid4()),
            parent_strategy_id=parent1.parent_strategy_id,
            name=f"{parent1.name}_x_{parent2.name}",
            parameters=child_params,
            generation=max(parent1.generation, parent2.generation) + 1,
            status="initialized",
            created_at=datetime.now(UTC)
        )

        return child

    async def prune_variants(self, keep_top_n: int):
        """
        Prune underperforming variants.

        Args:
            keep_top_n: Number of top variants to keep
        """
        if len(self.variants) <= keep_top_n:
            return

        logger.info(f"Pruning variants, keeping top {keep_top_n}")

        # Get top variants
        top_variants = await self.get_top_variants(top_n=keep_top_n)
        top_ids = set(v.variant_id for v in top_variants)

        # Mark others as pruned
        pruned_count = 0
        for variant_id, variant in self.variants.items():
            if variant_id not in top_ids and variant.status != "pruned":
                variant.status = "pruned"
                pruned_count += 1
                logger.info(f"Pruned variant {variant_id}")

        logger.info(f"Pruned {pruned_count} variants")

    async def hybridize_variants(
        self,
        parent_variants: List[StrategyVariant]
    ) -> StrategyVariant:
        """
        Hybridize multiple variants into one.

        Combines the best features from multiple variants.

        Args:
            parent_variants: Variants to hybridize

        Returns:
            Hybrid variant
        """
        if len(parent_variants) < 2:
            raise ValueError("Need at least 2 variants to hybridize")

        logger.info(f"Hybridizing {len(parent_variants)} variants")

        # Calculate weights based on fitness
        performances = [
            self.variant_performances[v.variant_id]
            for v in parent_variants
            if v.variant_id in self.variant_performances
        ]

        if not performances:
            raise ValueError("No performance data for parent variants")

        weights = np.array([p.calculate_fitness() for p in performances])
        weights = weights / weights.sum()  # Normalize

        # Combine parameters
        hybrid_params = {}
        for param in parent_variants[0].parameters.keys():
            hybrid_value = sum(
                weight * parent.parameters.get(param, 0)
                for weight, parent in zip(weights, parent_variants)
            )
            hybrid_params[param] = hybrid_value

        # Create hybrid variant
        hybrid = StrategyVariant(
            variant_id=str(uuid.uuid4()),
            parent_strategy_id=parent_variants[0].parent_strategy_id,
            name=f"hybrid_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            parameters=hybrid_params,
            generation=max(v.generation for v in parent_variants) + 1,
            status="initialized",
            created_at=datetime.now(UTC)
        )

        # Add to manager
        self.variants[hybrid.variant_id] = hybrid

        logger.info(f"Created hybrid variant {hybrid.variant_id}")

        return hybrid

    async def variant_to_strategy(self, variant: StrategyVariant) -> Strategy:
        """
        Convert variant back to strategy.

        Args:
            variant: Variant to convert

        Returns:
            Strategy object
        """
        # Get parent strategy info if available
        # For now, create basic strategy from variant

        strategy = Strategy(
            strategy_id=variant.variant_id,  # Use variant ID as strategy ID
            name=variant.name,
            description=f"Evolved strategy from variant {variant.variant_id}",
            strategy_type=StrategyType.MOMENTUM,  # Default type
            parameters=variant.parameters,
            metadata={
                "variant_id": variant.variant_id,
                "generation": variant.generation,
                "parent_strategy_id": variant.parent_strategy_id
            }
        )

        return strategy

    async def get_diversity_metrics(self) -> Dict[str, float]:
        """
        Calculate diversity metrics of current population.

        Returns:
            Diversity metrics
        """
        if len(self.variants) < 2:
            return {"diversity": 0.0}

        # Calculate parameter diversity
        param_values = defaultdict(list)
        for variant in self.variants.values():
            for param_name, param_value in variant.parameters.items():
                param_values[param_name].append(param_value)

        # Average coefficient of variation across parameters
        cvs = []
        for param_name, values in param_values.items():
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values)
                if mean > 0:
                    cv = std / mean
                    cvs.append(cv)

        diversity = np.mean(cvs) if cvs else 0.0

        return {
            "diversity": diversity,
            "num_variants": len(self.variants),
            "parameter_diversity": diversity
        }

    async def get_convergence_metrics(self) -> Dict[str, float]:
        """
        Calculate convergence metrics.

        Returns:
            Convergence metrics
        """
        if len(self.variant_history) < 2:
            return {"convergence": 0.0}

        # Calculate fitness improvement over time
        fitness_history = []

        for variant_id, history in self.variant_history.items():
            for snapshot in history:
                perf = StrategyPerformance(**snapshot["performance"])
                fitness = perf.calculate_fitness()
                fitness_history.append(fitness)

        if len(fitness_history) < 2:
            return {"convergence": 0.0}

        # Calculate improvement rate
        recent_fitness = fitness_history[-10:]
        early_fitness = fitness_history[:10]

        recent_avg = np.mean(recent_fitness)
        early_avg = np.mean(early_fitness)

        improvement = (recent_avg - early_avg) / abs(early_avg) if early_avg != 0 else 0

        return {
            "convergence": improvement,
            "recent_fitness": recent_avg,
            "early_fitness": early_avg
        }
