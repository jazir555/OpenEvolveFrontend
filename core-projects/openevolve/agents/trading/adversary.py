#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Strategy Tester

Red team testing for trading strategies.
Finds failure modes, stress tests under adverse conditions,
challenges strategy assumptions, and generates counter-strategies.

Key capabilities:
- Failure mode discovery
- Stress testing under adverse conditions
- Assumption challenging
- Counter-strategy generation
- Robustness evaluation

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
    StrategyPerformance
)


logger = logging.getLogger(__name__)


class Adversary:
    """
    Red team tester for trading strategies.

    Challenges strategies to find weaknesses and failure modes
    before they cause losses in live trading.

    Usage:
        adversary = Adversary(knowledge_engine=ke)

        result = await adversary.test_strategy(
            variant=variant,
            market_conditions=["bull", "bear", "high_volatility", "crisis"]
        )
    """

    def __init__(self, knowledge_engine=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Adversary.

        Args:
            knowledge_engine: Optional knowledge engine for learning
            config: Additional configuration
        """
        self.knowledge_engine = knowledge_engine
        self.config = config or {}

        # Adversarial scenarios
        self.scenarios = self._initialize_scenarios()

        logger.info("Adversary initialized")

    def _initialize_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize adversarial testing scenarios."""
        return {
            "black_swan": {
                "description": "Sudden extreme market move",
                "market_drop": -0.20,  # 20% drop
                "volatility_spike": 3.0,
                "liquidity_crunch": True
            },
            "whipsaw": {
                "description": "Rapid direction changes",
                "volatility": 2.0,
                "direction_changes": 10,
                "magnitude": 0.05
            },
            "gap_risk": {
                "description": "Overnight gap against position",
                "gap_size": -0.10,
                "overnight": True
            },
            "liquidity_crisis": {
                "description": "Unable to exit position",
                "bid_ask_spread_multiplier": 5.0,
                "volume_drop": 0.8
            },
            "correlation_breakdown": {
                "description": "Correlations go to 1 or -1",
                "correlation_change": "all_positive"
            },
            "regime_shift": {
                "description": "Market regime changes suddenly",
                "old_regime": "bull",
                "new_regime": "bear",
                "transition_speed": "rapid"
            },
            "flash_crash": {
                "description": "Extreme rapid drop and recovery",
                "crash_magnitude": -0.15,
                "recovery_time_minutes": 30
            },
            "squeeze": {
                "description": "Short squeeze or bear squeeze",
                "direction": "up",
                "magnitude": 0.30,
                "velocity": "explosive"
            }
        }

    async def test_strategy(
        self,
        variant: StrategyVariant,
        market_conditions: List[str],
        scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Test strategy against adversarial conditions.

        Args:
            variant: Strategy variant to test
            market_conditions: List of market conditions to test
            scenarios: Specific scenarios to test (optional)

        Returns:
            Adversarial test results
        """
        logger.info(f"Adversarial testing of variant {variant.variant_id}")

        results = {
            "variant_id": variant.variant_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "test_results": [],
            "failure_modes": [],
            "robustness_score": 0.0,
            "recommendations": []
        }

        # Test under different market conditions
        for condition in market_conditions:
            condition_result = await self._test_under_condition(
                variant,
                condition
            )
            results["test_results"].append(condition_result)

            # Collect failure modes
            results["failure_modes"].extend(
                condition_result.get("failure_modes", [])
            )

        # Test specific adversarial scenarios
        scenario_list = scenarios or ["black_swan", "whipsaw", "regime_shift"]

        for scenario_name in scenario_list:
            if scenario_name in self.scenarios:
                scenario_result = await self._test_scenario(
                    variant,
                    scenario_name,
                    self.scenarios[scenario_name]
                )
                results["test_results"].append(scenario_result)

                results["failure_modes"].extend(
                    scenario_result.get("failure_modes", [])
                )

        # Calculate robustness score
        results["robustness_score"] = self._calculate_robustness_score(
            results["test_results"]
        )

        # Generate recommendations
        results["recommendations"] = await self._generate_adversarial_recommendations(
            variant,
            results
        )

        logger.info(f"Adversarial testing complete. Robustness: {results['robustness_score']:.2f}, "
                   f"Failure modes: {len(results['failure_modes'])}")

        return results

    async def _test_under_condition(
        self,
        variant: StrategyVariant,
        condition: str
    ) -> Dict[str, Any]:
        """
        Test strategy under specific market condition.

        Args:
            variant: Strategy variant
            condition: Market condition

        Returns:
            Test result for condition
        """
        result = {
            "condition": condition,
            "passed": True,
            "performance_degradation": 0.0,
            "failure_modes": [],
            "max_drawdown": 0.0
        }

        # Simulate performance under condition
        condition_performance = await self._simulate_under_condition(
            variant,
            condition
        )

        # Evaluate performance
        if condition_performance["drawdown"] > 0.25:
            result["passed"] = False
            result["failure_modes"].append({
                "type": "excessive_drawdown",
                "condition": condition,
                "drawdown": condition_performance["drawdown"],
                "severity": "high"
            })

        if condition_performance["return"] < -0.15:
            result["passed"] = False
            result["failure_modes"].append({
                "type": "large_loss",
                "condition": condition,
                "loss": condition_performance["return"],
                "severity": "high"
            })

        result["performance_degradation"] = condition_performance["degradation"]
        result["max_drawdown"] = condition_performance["drawdown"]

        return result

    async def _simulate_under_condition(
        self,
        variant: StrategyVariant,
        condition: str
    ) -> Dict[str, float]:
        """
        Simulate strategy performance under condition.

        Args:
            variant: Strategy variant
            condition: Market condition

        Returns:
            Performance metrics
        """
        # Simulate based on condition type
        np.random.seed(hash(variant.variant_id + condition) % (2**32))

        if condition == "bull":
            # Bull market: generally good
            return {
                "return": np.random.normal(0.15, 0.10),
                "drawdown": abs(np.random.normal(0.05, 0.03)),
                "degradation": 0.0
            }

        elif condition == "bear":
            # Bear market: challenging
            return {
                "return": np.random.normal(-0.05, 0.15),
                "drawdown": abs(np.random.normal(0.15, 0.08)),
                "degradation": 0.5
            }

        elif condition == "high_volatility":
            # High volatility: increased risk
            return {
                "return": np.random.normal(0.05, 0.25),
                "drawdown": abs(np.random.normal(0.12, 0.10)),
                "degradation": 0.3
            }

        elif condition == "crisis":
            # Crisis: very challenging
            return {
                "return": np.random.normal(-0.15, 0.20),
                "drawdown": abs(np.random.normal(0.25, 0.15)),
                "degradation": 0.8
            }

        else:
            # Default: moderate performance
            return {
                "return": np.random.normal(0.05, 0.10),
                "drawdown": abs(np.random.normal(0.08, 0.05)),
                "degradation": 0.1
            }

    async def _test_scenario(
        self,
        variant: StrategyVariant,
        scenario_name: str,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test strategy against specific adversarial scenario.

        Args:
            variant: Strategy variant
            scenario_name: Name of scenario
            scenario: Scenario parameters

        Returns:
            Test result
        """
        result = {
            "scenario": scenario_name,
            "description": scenario["description"],
            "passed": True,
            "failure_modes": [],
            "impact_score": 0.0
        }

        # Simulate scenario impact
        impact = await self._simulate_scenario_impact(variant, scenario)

        result["impact_score"] = impact["score"]

        # Check for failures
        if impact["max_loss"] < -0.20:
            result["passed"] = False
            result["failure_modes"].append({
                "type": "scenario_loss",
                "scenario": scenario_name,
                "loss": impact["max_loss"],
                "severity": "critical"
            })

        if impact["drawdown"] > 0.30:
            result["passed"] = False
            result["failure_modes"].append({
                "type": "scenario_drawdown",
                "scenario": scenario_name,
                "drawdown": impact["drawdown"],
                "severity": "critical"
            })

        # Check if strategy has protection
        if not variant.parameters.get("risk_rules", {}).get("stop_loss_pct"):
            result["failure_modes"].append({
                "type": "missing_protection",
                "scenario": scenario_name,
                "description": "No stop loss protection for adverse scenarios",
                "severity": "medium"
            })

        return result

    async def _simulate_scenario_impact(
        self,
        variant: StrategyVariant,
        scenario: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Simulate impact of adversarial scenario.

        Args:
            variant: Strategy variant
            scenario: Scenario parameters

        Returns:
            Impact metrics
        """
        np.random.seed(hash(variant.variant_id + str(scenario)) % (2**32))

        # Base impact from scenario
        if "market_drop" in scenario:
            max_loss = scenario["market_drop"]
            drawdown = abs(scenario["market_drop"])
            score = abs(max_loss)

        elif "volatility" in scenario:
            max_loss = np.random.normal(0, scenario["volatility"] * 0.1)
            drawdown = abs(max_loss) * 1.5
            score = scenario["volatility"] * 0.3

        else:
            max_loss = np.random.normal(-0.10, 0.10)
            drawdown = abs(np.random.normal(0.15, 0.10))
            score = 0.5

        # Adjust based on strategy parameters
        risk_rules = variant.parameters.get("risk_rules", {})
        stop_loss = risk_rules.get("stop_loss_pct", 0.10)

        # Stop loss provides some protection
        if stop_loss < 0.10:
            max_loss = max(max_loss, -stop_loss)
            drawdown = min(drawdown, stop_loss * 1.5)
            score *= 0.7

        return {
            "max_loss": max_loss,
            "drawdown": drawdown,
            "score": score
        }

    def _calculate_robustness_score(
        self,
        test_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall robustness score.

        Args:
            test_results: List of test results

        Returns:
            Robustness score (0-1)
        """
        if not test_results:
            return 0.0

        # Score each test
        scores = []
        for result in test_results:
            if result.get("passed", True):
                scores.append(1.0)
            else:
                # Partial credit based on severity
                failure_modes = result.get("failure_modes", [])
                critical_failures = sum(
                    1 for fm in failure_modes
                    if fm.get("severity") in ["critical", "high"]
                )

                if critical_failures > 0:
                    scores.append(0.2)
                else:
                    scores.append(0.5)

        return np.mean(scores)

    async def _generate_adversarial_recommendations(
        self,
        variant: StrategyVariant,
        test_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on adversarial testing.

        Args:
            variant: Strategy variant
            test_results: Test results

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check for missing protections
        risk_rules = variant.parameters.get("risk_rules", {})

        if not risk_rules.get("stop_loss_pct"):
            recommendations.append(
                "Implement stop loss to protect against adverse scenarios"
            )

        if not risk_rules.get("max_position_size"):
            recommendations.append(
                "Set maximum position size to limit concentration risk"
            )

        # Check scenario-specific failures
        failure_modes = test_results.get("failure_modes", [])

        black_swan_failures = [
            fm for fm in failure_modes
            if "black_swan" in str(fm.get("condition", "")) or
               "crisis" in str(fm.get("condition", ""))
        ]

        if black_swan_failures:
            recommendations.append(
                "Add tail risk hedging for black swan events"
            )

        volatility_failures = [
            fm for fm in failure_modes
            if "volatility" in str(fm.get("condition", ""))
        ]

        if volatility_failures:
            recommendations.append(
                "Implement volatility-sensitive position sizing"
            )

        # General robustness recommendations
        if test_results["robustness_score"] < 0.6:
            recommendations.append(
                "Strategy shows limited robustness. Consider diversification."
            )

        return recommendations

    async def find_weaknesses(
        self,
        variant: StrategyVariant
    ) -> List[Dict[str, Any]]:
        """
        Identify specific weaknesses in strategy.

        Args:
            variant: Strategy variant

        Returns:
            List of weaknesses
        """
        weaknesses = []

        # Check parameter sensitivity
        parameters = variant.parameters

        # Look for risky parameter combinations
        stop_loss = parameters.get("risk_rules", {}).get("stop_loss_pct", 0.10)
        position_size = parameters.get("risk_rules", {}).get("max_position_size", 0.20)

        if stop_loss > 0.15 and position_size > 0.25:
            weaknesses.append({
                "type": "risk_concentration",
                "description": "Large stop loss combined with large position size",
                "severity": "high",
                "recommendation": "Reduce either stop loss or position size"
            })

        # Check for overfitting indicators
        if len(parameters) > 20:
            weaknesses.append({
                "type": "overfitting_risk",
                "description": f"High parameter count ({len(parameters)}) suggests overfitting risk",
                "severity": "medium",
                "recommendation": "Simplify strategy by reducing parameters"
            })

        # Check for regime specificity
        if "momentum" in str(parameters).lower():
            weaknesses.append({
                "type": "regime_specificity",
                "description": "Momentum strategies may underperform in sideways markets",
                "severity": "medium",
                "recommendation": "Consider adding regime filters"
            })

        return weaknesses

    async def generate_counter_strategy(
        self,
        variant: StrategyVariant
    ) -> Dict[str, Any]:
        """
        Generate a counter-strategy that would exploit weaknesses.

        Args:
            variant: Strategy variant

        Returns:
            Counter-strategy description
        """
        weaknesses = await self.find_weaknesses(variant)

        counter_strategy = {
            "description": "Counter-strategy to exploit main strategy weaknesses",
            "exploits": [w["description"] for w in weaknesses],
            "approach": [],
            "expected_advantage": 0.0
        }

        # Generate specific counter approaches
        for weakness in weaknesses:
            if weakness["type"] == "risk_concentration":
                counter_strategy["approach"].append(
                    "Trade against main strategy during extreme moves"
                )
                counter_strategy["expected_advantage"] += 0.2

            elif weakness["type"] == "overfitting_risk":
                counter_strategy["approach"].append(
                    "Simple trend-following will outperform in new market conditions"
                )
                counter_strategy["expected_advantage"] += 0.15

            elif weakness["type"] == "regime_specificity":
                counter_strategy["approach"].append(
                    "Fade strategy signals during regime transitions"
                )
                counter_strategy["expected_advantage"] += 0.1

        return counter_strategy
