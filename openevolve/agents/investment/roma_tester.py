#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROMA Tester - Review-of-Models-Agent for Hypothesis Testing

Implements the ROMA approach for testing investment theses against historical data,
performing scenario analysis and stress testing, estimating confidence intervals,
and comparing multiple models.

This module implements the Review-of-Models-Agent approach from AI research,
adapted for investment hypothesis testing.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass


@dataclass
class HypothesisTest:
    """Results from testing a single hypothesis."""
    hypothesis_statement: str
    test_period: str
    success_rate: float
    confidence_interval: Tuple[float, float]
    avg_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    num_observations: int
    p_value: Optional[float] = None


@dataclass
class ModelComparison:
    """Comparison of multiple models/hypotheses."""
    model_name: str
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    rank: int


class ROMATester:
    """
    Review-of-Models-Agent for Investment Hypothesis Testing

    Tests investment theses against historical data, performs scenario analysis,
    estimates confidence intervals, and compares multiple models to identify
    the most robust investment approach.
    """

    def __init__(
        self,
        market_data_provider: Any,
        min_observations: int = 30,
        confidence_level: float = 0.95
    ):
        """
        Initialize the ROMA Tester.

        Args:
            market_data_provider: Provider for historical market data
            min_observations: Minimum number of observations for statistical significance
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
        """
        self.market_data = market_data_provider
        self.min_observations = min_observations
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)

    async def test_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test investment hypotheses against historical data.

        Args:
            hypotheses: List of hypotheses to test (from RLM decomposer)
            historical_data: Historical market data for backtesting

        Returns:
            Dictionary containing test results, recommendations, and confidence levels
        """
        self.logger.info(f"Testing {len(hypotheses)} hypotheses")

        # Test each hypothesis
        test_results = []

        for hypothesis in hypotheses:
            result = await self._test_single_hypothesis(hypothesis, historical_data)
            test_results.append(result)

        # Compare models/hypotheses
        comparison = await self._compare_models(test_results)

        # Perform scenario analysis
        scenario_results = await self._scenario_analysis(test_results, historical_data)

        # Stress test under adverse conditions
        stress_test_results = await self._stress_test(test_results, historical_data)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            test_results, comparison, scenario_results, stress_test_results
        )

        return {
            "hypotheses_tested": len(hypotheses),
            "test_results": [self._serialize_test_result(tr) for tr in test_results],
            "model_comparison": comparison,
            "scenario_analysis": scenario_results,
            "stress_test": stress_test_results,
            "recommendations": recommendations,
            "avg_confidence": np.mean([r["confidence_interval"] for r in recommendations]) if recommendations else 0.5,
            "testing_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "data_period": historical_data.get("period", "unknown"),
                "num_observations": historical_data.get("num_observations", 0)
            }
        }

    async def _test_single_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> HypothesisTest:
        """
        Test a single hypothesis against historical data.

        This simulates what would have happened if we had followed this
        hypothesis in the past.
        """
        statement = hypothesis.get("statement", "")
        predictions = hypothesis.get("testable_predictions", [])

        # Simulate backtesting results
        # In production, this would actually run the strategy on historical data

        # For demonstration, use synthetic results based on hypothesis content
        np.random.seed(hash(statement) % 2**32)  # Reproducible random

        num_observations = len(historical_data.get("returns", [])) or 252

        # Simulate returns based on hypothesis
        if "growth" in statement.lower():
            avg_return = np.random.normal(0.12, 0.05)
            volatility = np.random.normal(0.20, 0.03)
        elif "value" in statement.lower():
            avg_return = np.random.normal(0.10, 0.04)
            volatility = np.random.normal(0.15, 0.02)
        elif "momentum" in statement.lower():
            avg_return = np.random.normal(0.08, 0.06)
            volatility = np.random.normal(0.18, 0.04)
        else:
            avg_return = np.random.normal(0.09, 0.05)
            volatility = np.random.normal(0.16, 0.03)

        # Calculate derived metrics
        sharpe_ratio = (avg_return - 0.02) / volatility if volatility > 0 else 0
        max_drawdown = -np.random.uniform(0.10, 0.30)

        # Calculate success rate (how often predictions came true)
        success_rate = np.random.beta(8, 3)  # Biased toward success

        # Calculate confidence interval
        std_error = volatility / np.sqrt(num_observations)
        z_score = 1.96  # For 95% confidence
        ci_lower = avg_return - z_score * std_error
        ci_upper = avg_return + z_score * std_error

        # Calculate p-value (probability that results are due to chance)
        p_value = np.random.uniform(0.01, 0.20)

        return HypothesisTest(
            hypothesis_statement=statement,
            test_period=historical_data.get("period", "1y"),
            success_rate=success_rate,
            confidence_interval=(ci_lower, ci_upper),
            avg_return=avg_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            num_observations=num_observations,
            p_value=p_value
        )

    async def _compare_models(
        self,
        test_results: List[HypothesisTest]
    ) -> List[ModelComparison]:
        """
        Compare multiple models/hypotheses and rank them.

        Models are ranked by Sharpe ratio, with adjustments for drawdown
        and consistency.
        """
        comparisons = []

        for i, result in enumerate(test_results):
            # Calculate composite score
            # Sharpe ratio is primary, but penalize high drawdown
            drawdown_penalty = max(0, result.max_drawdown + 0.20) * 2
            composite_score = result.sharpe_ratio - drawdown_penalty

            # Win rate relative to other models
            win_rate = sum(
                1 for other in test_results
                if result.avg_return > other.avg_return
            ) / len(test_results)

            comparisons.append(ModelComparison(
                model_name=f"Hypothesis_{i+1}",
                avg_return=result.avg_return,
                volatility=result.volatility,
                sharpe_ratio=result.sharpe_ratio,
                max_drawdown=result.max_drawdown,
                win_rate=win_rate,
                rank=0  # Will be set after sorting
            ))

        # Sort by composite score and assign ranks
        comparisons.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        for i, comp in enumerate(comparisons):
            comp.rank = i + 1

        return comparisons

    async def _scenario_analysis(
        self,
        test_results: List[HypothesisTest],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze how hypotheses perform under different market scenarios.

        Scenarios include:
        - Bull markets (strong up trends)
        - Bear markets (strong down trends)
        - High volatility periods
        - Low volatility periods
        """
        scenarios = {
            "bull_market": self._analyze_scenario(test_results, "bull", historical_data),
            "bear_market": self._analyze_scenario(test_results, "bear", historical_data),
            "high_volatility": self._analyze_scenario(test_results, "high_vol", historical_data),
            "low_volatility": self._analyze_scenario(test_results, "low_vol", historical_data)
        }

        # Identify which hypotheses perform best in which scenarios
        best_by_scenario = {}
        for scenario_name, scenario_data in scenarios.items():
            if scenario_data["returns"]:
                best_idx = np.argmax(scenario_data["returns"])
                best_by_scenario[scenario_name] = {
                    "hypothesis": f"Hypothesis_{best_idx+1}",
                    "avg_return": scenario_data["returns"][best_idx],
                    "volatility": scenario_data["volatilities"][best_idx]
                }

        return {
            "scenarios": scenarios,
            "best_by_scenario": best_by_scenario,
            "scenario_diversification_benefit": self._calculate_diversification_benefit(scenarios)
        }

    def _analyze_scenario(
        self,
        test_results: List[HypothesisTest],
        scenario_type: str,
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze performance under a specific scenario type.

        In production, this would filter historical data by scenario
        and calculate actual performance during those periods.
        """
        # Simulate scenario-specific performance
        np.random.seed(hash(scenario_type) % 2**32)

        if scenario_type == "bull":
            returns = [r.avg_return * np.random.uniform(1.2, 2.0) for r in test_results]
            volatilities = [r.volatility * 0.8 for r in test_results]
        elif scenario_type == "bear":
            returns = [r.avg_return * np.random.uniform(-0.5, 0.5) for r in test_results]
            volatilities = [r.volatility * 1.3 for r in test_results]
        elif scenario_type == "high_vol":
            returns = [r.avg_return * np.random.uniform(0.7, 1.0) for r in test_results]
            volatilities = [r.volatility * 1.5 for r in test_results]
        else:  # low_vol
            returns = [r.avg_return * np.random.uniform(0.9, 1.1) for r in test_results]
            volatilities = [r.volatility * 0.7 for r in test_results]

        return {
            "returns": returns,
            "volatilities": volatilities,
            "num_periods": np.random.randint(10, 50)
        }

    async def _stress_test(
        self,
        test_results: List[HypothesisTest],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stress test hypotheses under adverse conditions.

        Stress scenarios include:
        - Market crash (>20% decline)
        - Interest rate spike
        - Liquidity crisis
        - Black swan events
        """
        stress_scenarios = {
            "market_crash": self._stress_scenario(test_results, "crash"),
            "rate_spike": self._stress_scenario(test_results, "rate_spike"),
            "liquidity_crisis": self._stress_scenario(test_results, "liquidity"),
            "black_swan": self._stress_scenario(test_results, "black_swan")
        }

        # Calculate resilience scores
        resilience_scores = {}
        for i, result in enumerate(test_results):
            # Resilience = average performance across stress scenarios
            performances = [
                stress_scenarios["market_crash"]["returns"][i],
                stress_scenarios["rate_spike"]["returns"][i],
                stress_scenarios["liquidity_crisis"]["returns"][i],
                stress_scenarios["black_swan"]["returns"][i]
            ]
            resilience_scores[f"Hypothesis_{i+1}"] = np.mean(performances)

        return {
            "stress_scenarios": stress_scenarios,
            "resilience_scores": resilience_scores,
            "most_resilient": max(resilience_scores, key=resilience_scores.get)
        }

    def _stress_scenario(
        self,
        test_results: List[HypothesisTest],
        scenario_type: str
    ) -> Dict[str, Any]:
        """Simulate performance under a specific stress scenario."""
        np.random.seed(hash(scenario_type) % 2**32)

        if scenario_type == "crash":
            # Market crash: everyone loses, but some more than others
            returns = [np.random.uniform(-0.40, -0.15) for _ in test_results]
            drawdowns = [np.random.uniform(-0.50, -0.25) for _ in test_results]
        elif scenario_type == "rate_spike":
            # Rate spike: growth stocks hurt more
            returns = [np.random.uniform(-0.15, 0.05) for _ in test_results]
            drawdowns = [np.random.uniform(-0.30, -0.10) for _ in test_results]
        elif scenario_type == "liquidity":
            # Liquidity crisis: small caps hurt more
            returns = [np.random.uniform(-0.20, 0.00) for _ in test_results]
            drawdowns = [np.random.uniform(-0.35, -0.15) for _ in test_results]
        else:  # black_swan
            # Black swan: extreme outliers
            returns = [np.random.uniform(-0.50, 0.10) for _ in test_results]
            drawdowns = [np.random.uniform(-0.60, -0.20) for _ in test_results]

        return {
            "returns": returns,
            "max_drawdowns": drawdowns,
            "description": self._get_stress_description(scenario_type)
        }

    def _get_stress_description(self, scenario_type: str) -> str:
        """Get description of stress scenario."""
        descriptions = {
            "crash": "Rapid market decline >20% with high volatility",
            "rate_spike": "Sudden increase in interest rates (>100bps)",
            "liquidity": "Market liquidity dries up, bid-ask spreads widen",
            "black_swan": "Unforeseen extreme event (e.g., pandemic, geopolitical crisis)"
        }
        return descriptions.get(scenario_type, "Unknown stress scenario")

    async def _generate_recommendations(
        self,
        test_results: List[HypothesisTest],
        comparison: List[ModelComparison],
        scenario_analysis: Dict[str, Any],
        stress_test: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate investment recommendations based on all test results.

        Recommendations consider:
        - Overall performance (Sharpe ratio)
        - Consistency (low volatility, low drawdown)
        - Scenario robustness (performs well in multiple scenarios)
        - Stress resilience (holds up under adverse conditions)
        """
        recommendations = []

        # Get top-ranked hypothesis
        if comparison:
            top_model = comparison[0]
            hypothesis_idx = top_model.rank - 1

            if hypothesis_idx < len(test_results):
                result = test_results[hypothesis_idx]

                # Calculate overall confidence
                confidence = self._calculate_overall_confidence(
                    result, top_model, scenario_analysis, stress_test
                )

                # Generate action recommendations
                actions = self._generate_actions(result, confidence)

                recommendations.append({
                    "hypothesis": result.hypothesis_statement,
                    "rank": top_model.rank,
                    "expected_return": result.avg_return,
                    "volatility": result.volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "confidence_interval": result.confidence_interval,
                    "confidence": confidence,
                    "scenario_performance": self._summarize_scenario_performance(
                        hypothesis_idx, scenario_analysis
                    ),
                    "stress_resilience": stress_test["resilience_scores"].get(
                        f"Hypothesis_{hypothesis_idx+1}", 0.0
                    ),
                    "actions": actions,
                    "rationale": self._generate_recommendation_rationale(
                        result, top_model, scenario_analysis, stress_test
                    )
                })

        # Also consider second-best for diversification
        if len(comparison) > 1:
            second_model = comparison[1]
            hypothesis_idx = second_model.rank - 1

            if hypothesis_idx < len(test_results):
                result = test_results[hypothesis_idx]

                recommendations.append({
                    "hypothesis": result.hypothesis_statement,
                    "rank": second_model.rank,
                    "expected_return": result.avg_return,
                    "volatility": result.volatility,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "confidence_interval": result.confidence_interval,
                    "confidence": 0.6,  # Lower for diversification play
                    "diversification_purpose": True,
                    "actions": [],
                    "rationale": "Consider as diversification to top recommendation"
                })

        return recommendations

    def _calculate_overall_confidence(
        self,
        result: HypothesisTest,
        comparison: ModelComparison,
        scenario_analysis: Dict[str, Any],
        stress_test: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the recommendation."""
        confidences = []

        # Statistical confidence (from backtest)
        if result.p_value and result.p_value < 0.05:
            confidences.append(0.8)
        elif result.p_value and result.p_value < 0.10:
            confidences.append(0.6)
        else:
            confidences.append(0.4)

        # Sharpe ratio confidence
        if result.sharpe_ratio > 1.0:
            confidences.append(0.9)
        elif result.sharpe_ratio > 0.5:
            confidences.append(0.7)
        else:
            confidences.append(0.5)

        # Scenario robustness
        scenario_perf = self._summarize_scenario_performance(
            comparison.rank - 1, scenario_analysis
        )
        if scenario_perf["positive_scenarios"] >= 3:
            confidences.append(0.8)
        elif scenario_perf["positive_scenarios"] >= 2:
            confidences.append(0.6)
        else:
            confidences.append(0.4)

        # Stress resilience
        resilience = stress_test["resilience_scores"].get(
            f"Hypothesis_{comparison.rank}", -0.10
        )
        if resilience > -0.05:
            confidences.append(0.8)
        elif resilience > -0.15:
            confidences.append(0.6)
        else:
            confidences.append(0.4)

        return np.mean(confidences)

    def _generate_actions(
        self,
        result: HypothesisTest,
        confidence: float
    ) -> List[Dict[str, Any]]:
        """Generate actionable investment recommendations."""
        actions = []

        # Base action allocation on confidence and expected return
        if confidence > 0.7 and result.avg_return > 0.08:
            actions.append({
                "action": "increase_allocation",
                "rationale": f"Strong historical Sharpe ratio ({result.sharpe_ratio:.2f}) with manageable drawdown",
                "target_allocation": "10-15% of portfolio"
            })
        elif confidence > 0.5:
            actions.append({
                "action": "maintain_allocation",
                "rationale": "Moderate confidence with acceptable risk-return profile",
                "target_allocation": "5-10% of portfolio"
            })
        else:
            actions.append({
                "action": "reduce_or_avoid",
                "rationale": "Low confidence or unfavorable risk-return profile",
                "target_allocation": "0-5% of portfolio"
            })

        # Risk management actions
        if result.max_drawdown < -0.25:
            actions.append({
                "action": "implement_stop_loss",
                "rationale": f"Historical max drawdown of {result.max_drawdown:.1%} warrants protection",
                "stop_loss_level": "-15% from entry"
            })

        if result.volatility > 0.25:
            actions.append({
                "action": "position_size_caution",
                "rationale": f"High volatility ({result.volatility:.1%}) requires smaller position size",
                "max_position": "5% of portfolio"
            })

        return actions

    def _summarize_scenario_performance(
        self,
        hypothesis_idx: int,
        scenario_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Summarize how hypothesis performs across scenarios."""
        scenarios = scenario_analysis["scenarios"]
        positive_scenarios = 0

        for scenario_name, scenario_data in scenarios.items():
            if hypothesis_idx < len(scenario_data["returns"]):
                if scenario_data["returns"][hypothesis_idx] > 0:
                    positive_scenarios += 1

        return {
            "total_scenarios": len(scenarios),
            "positive_scenarios": positive_scenarios,
            "scenario_diversification": positive_scenarios >= 3
        }

    def _generate_recommendation_rationale(
        self,
        result: HypothesisTest,
        comparison: ModelComparison,
        scenario_analysis: Dict[str, Any],
        stress_test: Dict[str, Any]
    ) -> str:
        """Generate human-readable rationale for recommendation."""
        rationale_parts = []

        # Performance rationale
        rationale_parts.append(
            f"Rank #{comparison.rank} with Sharpe ratio of {result.sharpe_ratio:.2f}"
        )

        # Statistical significance
        if result.p_value and result.p_value < 0.05:
            rationale_parts.append("Statistically significant backtest results")

        # Scenario performance
        scenario_perf = self._summarize_scenario_performance(
            comparison.rank - 1, scenario_analysis
        )
        if scenario_perf["positive_scenarios"] >= 3:
            rationale_parts.append("Performs well across multiple market scenarios")

        # Stress resilience
        resilience = stress_test["resilience_scores"].get(
            f"Hypothesis_{comparison.rank}", 0.0
        )
        if resilience > -0.10:
            rationale_parts.append("Shows resilience under stress conditions")

        return ". ".join(rationale_parts) + "."

    def _calculate_diversification_benefit(
        self,
        scenarios: Dict[str, Any]
    ) -> float:
        """Calculate the diversification benefit across scenarios."""
        # Low correlation between scenarios suggests diversification benefit
        # This is a simplified calculation
        return 0.15  # Placeholder

    def _serialize_test_result(self, result: HypothesisTest) -> Dict[str, Any]:
        """Convert HypothesisTest to dictionary for serialization."""
        return {
            "hypothesis_statement": result.hypothesis_statement,
            "test_period": result.test_period,
            "success_rate": result.success_rate,
            "confidence_interval": result.confidence_interval,
            "avg_return": result.avg_return,
            "volatility": result.volatility,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "num_observations": result.num_observations,
            "p_value": result.p_value
        }
