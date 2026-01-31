#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Judge Panel Evaluator

Multiple evaluation perspectives for trading strategies.
Provides risk-adjusted return analysis, drawdown evaluation,
market condition robustness, and consensus/conflict detection.

Judge perspectives:
1. Risk Manager: Focus on drawdown and risk controls
2. Return Optimizer: Focus on profitability and Sharpe ratio
3. Robustness Expert: Focus on market condition stability
4. Sustainability Judge: Focus on long-term viability
5. Implementation Specialist: Focus on practical execution

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from openevolve.agents.trading.schemas import (
    Strategy,
    StrategyVariant,
    StrategyPerformance,
    JudgeEvaluation
)


logger = logging.getLogger(__name__)


class JudgePanel:
    """
    Multi-perspective evaluation panel for trading strategies.

    Each judge evaluates the strategy from their perspective,
    then scores are aggregated with consensus/conflict detection.

    Usage:
        panel = JudgePanel(knowledge_engine=ke)

        evaluations = await panel.evaluate_strategy(
            variant=variant,
            performance=performance,
            market_regime=regime
        )

        aggregate = panel.aggregate_evaluations(evaluations)
    """

    def __init__(self, knowledge_engine=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Judge Panel.

        Args:
            knowledge_engine: Optional knowledge engine for learning
            config: Additional configuration
        """
        self.knowledge_engine = knowledge_engine
        self.config = config or {}

        # Define judges
        self.judges = {
            "risk_manager": RiskManagerJudge(),
            "return_optimizer": ReturnOptimizerJudge(),
            "robustness_expert": RobustnessExpertJudge(),
            "sustainability_judge": SustainabilityJudge(),
            "implementation_specialist": ImplementationSpecialistJudge()
        }

        logger.info(f"JudgePanel initialized with {len(self.judges)} judges")

    async def evaluate_strategy(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> List[JudgeEvaluation]:
        """
        Evaluate strategy from all judge perspectives.

        Args:
            variant: Strategy variant to evaluate
            performance: Performance metrics
            market_regime: Current market regime

        Returns:
            List of evaluations from all judges
        """
        logger.info(f"Evaluating variant {variant.variant_id} with judge panel")

        evaluations = []

        # Run all judges in parallel
        judge_tasks = [
            judge.evaluate(variant, performance, market_regime)
            for judge in self.judges.values()
        ]

        results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Filter successful evaluations
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Judge {list(self.judges.keys())[i]} failed: {result}")
            else:
                evaluations.append(result)

        logger.info(f"Received {len(evaluations)} judge evaluations")

        return evaluations

    def aggregate_evaluations(
        self,
        evaluations: List[JudgeEvaluation]
    ) -> Dict[str, Any]:
        """
        Aggregate judge evaluations.

        Calculates:
        - Overall score (weighted average)
        - Consensus level
        - Conflicts and concerns
        - Final recommendation

        Args:
            evaluations: List of judge evaluations

        Returns:
            Aggregated evaluation results
        """
        if not evaluations:
            return {
                "overall_score": 0.0,
                "consensus": 0.0,
                "recommendation": "reject",
                "reason": "No evaluations received"
            }

        # Calculate weighted scores
        judge_weights = {
            "risk_manager": 0.25,
            "return_optimizer": 0.25,
            "robustness_expert": 0.2,
            "sustainability_judge": 0.15,
            "implementation_specialist": 0.15
        }

        weighted_score = 0.0
        total_weight = 0.0

        for eval in evaluations:
            weight = judge_weights.get(eval.judge_id, 0.2)
            weighted_score += eval.score * weight
            total_weight += weight

        overall_score = weighted_score / total_weight if total_weight > 0 else 0

        # Calculate consensus
        scores = [e.score for e in evaluations]
        consensus = 1.0 - np.std(scores)  # Lower std = higher consensus

        # Collect all concerns
        all_concerns = []
        for eval in evaluations:
            all_concerns.extend(eval.concerns)

        # Collect all recommendations
        all_recommendations = []
        for eval in evaluations:
            all_recommendations.extend(eval.recommendations)

        # Determine final recommendation
        if overall_score >= 0.8 and consensus > 0.7:
            recommendation = "approve"
        elif overall_score >= 0.6:
            recommendation = "conditional"
        else:
            recommendation = "reject"

        aggregate = {
            "overall_score": overall_score,
            "consensus": consensus,
            "recommendation": recommendation,
            "judge_scores": {e.judge_id: e.score for e in evaluations},
            "concerns": all_concerns,
            "recommendations": all_recommendations,
            "num_evaluations": len(evaluations)
        }

        logger.info(f"Aggregated score: {overall_score:.3f}, "
                   f"consensus: {consensus:.3f}, "
                   f"recommendation: {recommendation}")

        return aggregate

    def detect_conflicts(
        self,
        evaluations: List[JudgeEvaluation]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between judge evaluations.

        Args:
            evaluations: Judge evaluations

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for score discrepancies
        scores = [e.score for e in evaluations]
        if np.std(scores) > 0.3:
            conflicts.append({
                "type": "score_discrepancy",
                "severity": "high",
                "description": f"High variance in judge scores: {np.std(scores):.3f}"
            })

        # Check for conflicting concerns
        concern_keywords = {}
        for eval in evaluations:
            for concern in eval.concerns:
                for word in concern.lower().split():
                    concern_keywords[word] = concern_keywords.get(word, 0) + 1

        # Find overlapping concerns
        overlapping = {k: v for k, v in concern_keywords.items() if v > 1}
        if overlapping:
            conflicts.append({
                "type": "shared_concerns",
                "severity": "medium",
                "concerns": overlapping
            })

        return conflicts


# ============================================================================
# Individual Judge Implementations
# ============================================================================

class BaseJudge:
    """Base class for all judges."""

    def __init__(self, judge_id: str, perspective: str):
        self.judge_id = judge_id
        self.perspective = perspective

    async def evaluate(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate strategy from this judge's perspective."""
        raise NotImplementedError


class RiskManagerJudge(BaseJudge):
    """Judge focused on risk management and drawdowns."""

    def __init__(self):
        super().__init__("risk_manager", "risk_management")

    async def evaluate(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate from risk perspective."""
        score = 0.0
        concerns = []
        recommendations = []

        # Evaluate drawdown
        if performance.max_drawdown < 0.1:
            score += 0.4
        elif performance.max_drawdown < 0.2:
            score += 0.3
        elif performance.max_drawdown < 0.3:
            score += 0.1
            concerns.append("Maximum drawdown exceeds 20%")
        else:
            concerns.append("Maximum drawdown dangerously high")

        # Evaluate volatility
        if performance.volatility < 0.15:
            score += 0.3
        elif performance.volatility < 0.25:
            score += 0.2
        else:
            concerns.append("High volatility")

        # Evaluate win rate
        if performance.win_rate > 0.55:
            score += 0.3
        elif performance.win_rate > 0.45:
            score += 0.15
        else:
            concerns.append("Low win rate")

        # Check risk rules
        risk_rules = variant.parameters.get("risk_rules", {})
        if not risk_rules.get("stop_loss_pct"):
            concerns.append("No stop loss defined")
            recommendations.append("Implement stop loss")
        else:
            score += 0.1

        if not risk_rules.get("max_position_size"):
            concerns.append("No position size limit")
            recommendations.append("Set maximum position size")

        reasoning = f"Risk evaluation: drawdown={performance.max_drawdown:.2%}, " \
                   f"volatility={performance.volatility:.2%}, win_rate={performance.win_rate:.2%}"

        return JudgeEvaluation(
            judge_id=self.judge_id,
            perspective=self.perspective,
            score=min(score, 1.0),
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations
        )


class ReturnOptimizerJudge(BaseJudge):
    """Judge focused on returns and profitability."""

    def __init__(self):
        super().__init__("return_optimizer", "return_optimization")

    async def evaluate(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate from return perspective."""
        score = 0.0
        concerns = []
        recommendations = []

        # Evaluate total return
        if performance.total_return > 0.3:
            score += 0.3
        elif performance.total_return > 0.15:
            score += 0.2
        elif performance.total_return > 0.05:
            score += 0.1
        else:
            concerns.append("Low total return")

        # Evaluate Sharpe ratio
        if performance.sharpe_ratio > 2.0:
            score += 0.3
        elif performance.sharpe_ratio > 1.5:
            score += 0.2
        elif performance.sharpe_ratio > 1.0:
            score += 0.1
        else:
            concerns.append("Sharpe ratio below 1.0")

        # Evaluate Sortino ratio
        if performance.sortino_ratio > 2.0:
            score += 0.2
        elif performance.sortino_ratio > 1.5:
            score += 0.1

        # Evaluate profit factor
        if performance.profit_factor > 2.0:
            score += 0.2
        elif performance.profit_factor > 1.5:
            score += 0.1
        else:
            concerns.append("Profit factor below 1.5")

        reasoning = f"Return evaluation: return={performance.total_return:.2%}, " \
                   f"sharpe={performance.sharpe_ratio:.2f}, profit_factor={performance.profit_factor:.2f}"

        return JudgeEvaluation(
            judge_id=self.judge_id,
            perspective=self.perspective,
            score=min(score, 1.0),
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations
        )


class RobustnessExpertJudge(BaseJudge):
    """Judge focused on robustness across market conditions."""

    def __init__(self):
        super().__init__("robustness_expert", "robustness")

    async def evaluate(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate from robustness perspective."""
        score = 0.0
        concerns = []
        recommendations = []

        # Check if strategy has performed across regimes
        # This would require historical performance across different regimes
        # For now, use proxies

        # Evaluate consistency (use calmar ratio as proxy)
        if performance.calmar_ratio > 1.0:
            score += 0.4
        elif performance.calmar_ratio > 0.5:
            score += 0.2
        else:
            concerns.append("Poor risk-adjusted consistency")

        # Check market regime suitability
        current_regime = market_regime.get("regime", "unknown")
        strategy_type = variant.parameters.get("strategy_type", "unknown")

        regime_match = self._check_regime_suitability(strategy_type, current_regime)
        if regime_match:
            score += 0.3
        else:
            concerns.append(f"Strategy may not suit current {current_regime} regime")

        # Check number of trades (more trades = more robust)
        if performance.trades > 100:
            score += 0.3
        elif performance.trades > 50:
            score += 0.2
        elif performance.trades > 20:
            score += 0.1
        else:
            concerns.append("Insufficient trade history for robustness assessment")
            recommendations.append("Accumulate more trade data")

        reasoning = f"Robustness evaluation: calmar={performance.calmar_ratio:.2f}, " \
                   f"trades={performance.trades}, regime_match={regime_match}"

        return JudgeEvaluation(
            judge_id=self.judge_id,
            perspective=self.perspective,
            score=min(score, 1.0),
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations
        )

    def _check_regime_suitability(self, strategy_type: str, regime: str) -> bool:
        """Check if strategy suits current regime."""
        suitable_combinations = {
            "momentum": ["bull", "high_volatility"],
            "mean_reversion": ["sideways", "low_volatility"],
            "trend_following": ["bull", "bear"],
            "statistical_arbitrage": ["sideways"]
        }

        return regime in suitable_combinations.get(strategy_type, [])


class SustainabilityJudge(BaseJudge):
    """Judge focused on long-term sustainability."""

    def __init__(self):
        super().__init__("sustainability_judge", "sustainability")

    async def evaluate(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate from sustainability perspective."""
        score = 0.0
        concerns = []
        recommendations = []

        # Evaluate information ratio (consistency of outperformance)
        if performance.information_ratio > 0.5:
            score += 0.4
        elif performance.information_ratio > 0.3:
            score += 0.2
        else:
            concerns.append("Low information ratio")

        # Check strategy complexity (simpler = more sustainable)
        param_count = len(variant.parameters)
        if param_count < 10:
            score += 0.3
        elif param_count < 20:
            score += 0.2
        else:
            concerns.append("High parameter complexity may lead to overfitting")
            recommendations.append("Consider parameter reduction")

        # Check generation (newer strategies less proven)
        if variant.generation > 5:
            score += 0.3
        elif variant.generation > 2:
            score += 0.2
        else:
            concerns.append("Strategy is relatively new and unproven")

        reasoning = f"Sustainability evaluation: info_ratio={performance.information_ratio:.2f}, " \
                   f"complexity={param_count} params, generation={variant.generation}"

        return JudgeEvaluation(
            judge_id=self.judge_id,
            perspective=self.perspective,
            score=min(score, 1.0),
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations
        )


class ImplementationSpecialistJudge(BaseJudge):
    """Judge focused on practical implementation."""

    def __init__(self):
        super().__init__("implementation_specialist", "implementation")

    async def evaluate(
        self,
        variant: StrategyVariant,
        performance: StrategyPerformance,
        market_regime: Dict[str, Any]
    ) -> JudgeEvaluation:
        """Evaluate from implementation perspective."""
        score = 0.0
        concerns = []
        recommendations = []

        # Check if strategy has clear entry/exit conditions
        if variant.parameters.get("entry_conditions"):
            score += 0.3
        else:
            concerns.append("Unclear entry conditions")
            recommendations.append("Define clear entry rules")

        if variant.parameters.get("exit_conditions"):
            score += 0.3
        else:
            concerns.append("Unclear exit conditions")
            recommendations.append("Define clear exit rules")

        # Check if strategy requires exotic data
        data_requirements = variant.parameters.get("data_requirements", [])
        if not data_requirements or all(d in ["price", "volume", "ohlc"] for d in data_requirements):
            score += 0.2
        else:
            concerns.append("Strategy may require exotic or expensive data")

        # Check execution frequency
        avg_holding_period = variant.parameters.get("avg_holding_period", 1)
        if avg_holding_period >= 1:  # At least 1 day
            score += 0.2
        else:
            concerns.append("High-frequency trading may have execution challenges")

        reasoning = f"Implementation evaluation: entry_clear={bool(variant.parameters.get('entry_conditions'))}, " \
                   f"exit_clear={bool(variant.parameters.get('exit_conditions'))}, " \
                   f"data_complexity={len(data_requirements)}"

        return JudgeEvaluation(
            judge_id=self.judge_id,
            perspective=self.perspective,
            score=min(score, 1.0),
            reasoning=reasoning,
            concerns=concerns,
            recommendations=recommendations
        )
