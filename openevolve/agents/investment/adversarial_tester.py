#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Tester - Red Team for Investment Recommendations

Challenges assumptions and biases in investment recommendations,
generates counter-arguments, identifies failure modes, and stress tests
under adverse conditions.

This module implements adversarial testing to improve the robustness of
investment decisions by actively trying to find weaknesses.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum


class BiasType(Enum):
    """Types of cognitive biases to check for."""
    CONFIRMATION_BIAS = "confirmation_bias"
    OVERCONFIDENCE = "overconfidence"
    ANCHORING = "anchoring"
    RECENCY_BIAS = "recency_bias"
    HERDING = "herding"
    LOSS_AVERSION = "loss_aversion"
    SURVIVORSHIP_BIAS = "survivorship_bias"


@dataclass
class AdversarialChallenge:
    """A specific challenge to an investment recommendation."""
    challenge_type: str
    description: str
    severity: float  # 0.0 to 1.0
    counter_argument: str
    potential_failure_mode: str
    mitigation: str


@dataclass
class BiasDetection:
    """Detection of a potential cognitive bias."""
    bias_type: BiasType
    description: str
    evidence: List[str]
    severity: float  # 0.0 to 1.0
    correction: str


class AdversarialTester:
    """
    Adversarial Tester for Investment Recommendations

    Acts as a "red team" to challenge investment recommendations,
    identify assumptions, detect biases, and find failure modes.
    """

    def __init__(
        self,
        severity_threshold: float = 0.6,
        max_challenges_per_recommendation: int = 5
    ):
        """
        Initialize the Adversarial Tester.

        Args:
            severity_threshold: Minimum severity to flag a concern
            max_challenges_per_recommendation: Maximum challenges to generate
        """
        self.severity_threshold = severity_threshold
        self.max_challenges = max_challenges_per_recommendation
        self.logger = logging.getLogger(__name__)

    async def challenge_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        portfolio_state: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Challenge investment recommendations from multiple angles.

        Args:
            recommendations: Investment recommendations to challenge
            portfolio_state: Current portfolio allocations

        Returns:
            Dictionary containing challenges, biases detected, and concerns
        """
        self.logger.info(f"Adversarial testing of {len(recommendations)} recommendations")

        all_challenges = []
        all_biases = []
        all_failure_modes = []

        for rec in recommendations:
            # Generate specific challenges to this recommendation
            challenges = await self._generate_challenges(rec, portfolio_state)
            all_challenges.extend(challenges)

            # Check for cognitive biases
            biases = await self._detect_biases(rec, recommendations)
            all_biases.extend(biases)

            # Identify potential failure modes
            failure_modes = await self._identify_failure_modes(rec)
            all_failure_modes.extend(failure_modes)

        # Calculate overall severity score
        high_severity_challenges = [
            c for c in all_challenges
            if c.severity >= self.severity_threshold
        ]

        high_severity_biases = [
            b for b in all_biases
            if b.severity >= self.severity_threshold
        ]

        # Generate overall concerns
        concerns = await self._synthesize_concerns(
            high_severity_challenges, high_severity_biases, all_failure_modes
        )

        return {
            "challenges": [self._serialize_challenge(c) for c in all_challenges],
            "biases": [self._serialize_bias(b) for b in all_biases],
            "failure_modes": all_failure_modes,
            "concerns": concerns,
            "severity_score": self._calculate_severity_score(
                all_challenges, all_biases
            ),
            "recommendation": self._generate_adversarial_recommendation(
                all_challenges, all_biases, all_failure_modes
            ),
            "testing_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "recommendations_tested": len(recommendations),
                "challenges_generated": len(all_challenges),
                "biases_detected": len(all_biases),
                "failure_modes_identified": len(all_failure_modes)
            }
        }

    async def _generate_challenges(
        self,
        recommendation: Dict[str, Any],
        portfolio_state: Dict[str, float]
    ) -> List[AdversarialChallenge]:
        """Generate specific challenges to a recommendation."""
        challenges = []
        hypothesis = recommendation.get("hypothesis", "")

        # Challenge 1: Data quality and representativeness
        if "historical" in hypothesis.lower() or "backtest" in hypothesis.lower():
            challenges.append(AdversarialChallenge(
                challenge_type="data_quality",
                description=f"Historical data may not represent future market conditions",
                severity=0.7,
                counter_argument="Past performance does not guarantee future results",
                potential_failure_mode="Regime change makes historical patterns invalid",
                mitigation="Use multiple time periods, stress test, combine with forward-looking indicators"
            ))

        # Challenge 2: Sample size and statistical significance
        if recommendation.get("num_observations", 0) < 100:
            challenges.append(AdversarialChallenge(
                challenge_type="sample_size",
                description="Sample size may be too small for statistical significance",
                severity=0.6,
                counter_argument="Results may be due to random chance",
                potential_failure_mode="Overfitting to limited data leads to poor out-of-sample performance",
                mitigation="Increase sample size, use cross-validation, apply statistical tests"
            ))

        # Challenge 3: Assumption stability
        if "growth" in hypothesis.lower():
            challenges.append(AdversarialChallenge(
                challenge_type="assumption_stability",
                description="Assumes growth stock dominance will continue",
                severity=0.65,
                counter_argument="Market regimes rotate; value may outperform in next cycle",
                potential_failure_mode="Value stocks outperform, leading to underperformance",
                mitigation="Diversify across styles, use dynamic allocation, monitor regime indicators"
            ))

        # Challenge 4: Crowded trade risk
        if "momentum" in hypothesis.lower() or "trend" in hypothesis.lower():
            challenges.append(AdversarialChallenge(
                challenge_type="crowded_trade",
                description="Momentum strategies can become crowded trades",
                severity=0.75,
                counter_argument="When everyone bets on the same trend, reversals can be violent",
                potential_failure_mode="Sudden trend reversal causes large losses",
                mitigation="Monitor positioning, use stop losses, limit position size"
            ))

        # Challenge 5: Liquidity risk
        expected_return = recommendation.get("expected_return", 0)
        if expected_return > 0.15:  # High returns often imply illiquidity
            challenges.append(AdversarialChallenge(
                challenge_type="liquidity_risk",
                description="High expected returns may come from illiquid positions",
                severity=0.6,
                counter_argument="In stressed markets, liquidity may disappear",
                potential_failure_mode="Cannot exit position at fair price during stress",
                mitigation="Limit illiquid allocations, stress test liquidity, maintain cash buffer"
            ))

        # Challenge 6: Model risk
        sharpe_ratio = recommendation.get("sharpe_ratio", 0)
        if sharpe_ratio > 1.5:
            challenges.append(AdversarialChallenge(
                challenge_type="model_risk",
                description=f"Very high Sharpe ratio ({sharpe_ratio:.2f}) may indicate data mining",
                severity=0.7,
                counter_argument="Exceptionally good backtest results are often too good to be true",
                potential_failure_mode="Real-world performance much worse than backtest",
                mitigation="Out-of-sample testing, paper trading, conservative expectations"
            ))

        # Challenge 7: Correlation breakdown
        actions = recommendation.get("actions", [])
        for action in actions:
            if action.get("action") == "maintain_allocation":
                challenges.append(AdversarialChallenge(
                    challenge_type="correlation_breakdown",
                    description="Assumes historical correlations will persist",
                    severity=0.55,
                    counter_argument="Correlations can break down in crisis (all assets fall together)",
                    potential_failure_mode="Diversification fails when most needed",
                    mitigation="Stress test correlations, use tail-risk hedges, hold some cash"
                ))

        # Limit to max challenges
        return challenges[:self.max_challenges]

    async def _detect_biases(
        self,
        recommendation: Dict[str, Any],
        all_recommendations: List[Dict[str, Any]]
    ) -> List[BiasDetection]:
        """Detect cognitive biases in recommendations."""
        biases = []

        # Check for confirmation bias
        if "evidence" in recommendation and "counter_evidence" in recommendation:
            evidence_count = len(recommendation["evidence"])
            counter_count = len(recommendation["counter_evidence"])

            if evidence_count > counter_count * 2:
                biases.append(BiasDetection(
                    bias_type=BiasType.CONFIRMATION_BIAS,
                    description="Recommendation may be overweighting supporting evidence",
                    evidence=[f"Found {evidence_count} pieces of evidence vs {counter_count} counter-evidence"],
                    severity=0.6,
                    correction="Actively seek disconfirming evidence, consider alternative views"
                ))

        # Check for overconfidence
        confidence = recommendation.get("confidence", 0.5)
        if confidence > 0.85:
            biases.append(BiasDetection(
                bias_type=BiasType.OVERCONFIDENCE,
                description=f"Very high confidence ({confidence:.1%}) may be unwarranted",
                evidence=["Confidence exceeds typical prediction accuracy"],
                severity=0.7,
                correction="Calibrate confidence using historical accuracy, build in margin of safety"
            ))

        # Check for recency bias
        if "recent" in str(recommendation).lower() or "latest" in str(recommendation).lower():
            biases.append(BiasDetection(
                bias_type=BiasType.RECENCY_BIAS,
                description="May be overweighting recent events/data",
                evidence=["Recommendation emphasizes recent information"],
                severity=0.55,
                correction="Use longer historical windows, test across different time periods"
            ))

        # Check for herding (similar to other recommendations)
        similar_count = sum(
            1 for rec in all_recommendations
            if rec.get("hypothesis", "") == recommendation.get("hypothesis", "")
        )

        if similar_count > 1:
            biases.append(BiasDetection(
                bias_type=BiasType.HERDING,
                description=f"Multiple recommendations make similar bets ({similar_count} total)",
                evidence=[f"Found {similar_count} similar recommendations"],
                severity=0.65,
                correction="Ensure diversification, consider contrarian views, check correlation"
            ))

        # Check for loss aversion
        max_drawdown = recommendation.get("max_drawdown", 0)
        if max_drawdown > -0.15:
            actions = recommendation.get("actions", [])
            for action in actions:
                if "stop_loss" in str(action).lower():
                    biases.append(BiasDetection(
                        bias_type=BiasType.LOSS_AVERSION,
                        description="May be overly focused on avoiding losses",
                        evidence=["Stop loss recommended despite moderate drawdown"],
                        severity=0.45,
                        correction="Consider total portfolio risk, not just individual positions"
                    ))

        # Check for survivorship bias
        if "backtest" in str(recommendation).lower():
            biases.append(BiasDetection(
                bias_type=BiasType.SURVIVORSHIP_BIAS,
                description="Backtest may only include surviving companies/assets",
                evidence=["Historical data may exclude delisted/bankrupt entities"],
                severity=0.6,
                correction="Adjust for survivorship bias, test on survivorship-free data"
            ))

        return biases

    async def _identify_failure_modes(
        self,
        recommendation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential ways the recommendation could fail."""
        failure_modes = []

        # Failure mode 1: Market regime change
        failure_modes.append({
            "mode": "Regime Change",
            "description": "Market enters new regime with different dynamics",
            "probability": "Medium (30-40%)",
            "impact": "High (-20% to -40%)",
            "early_warning_signals": [
                "Sustained deviation from historical relationships",
                "Change in correlation structure",
                "Shift in market leadership"
            ],
            "mitigation": "Monitor regime indicators, be ready to adapt quickly"
        })

        # Failure mode 2: Black swan event
        failure_modes.append({
            "mode": "Black Swan",
            "description": "Unforeseen extreme event",
            "probability": "Low (<5%)",
            "impact": "Very High (-40% to -80%)",
            "early_warning_signals": [
                "Hard to predict by definition",
                "Increase in market fragility measures",
                "Rising systemic risk indicators"
            ],
            "mitigation": "Maintain cash/liquid reserves, use tail-risk hedges"
        })

        # Failure mode 3: Liquidity crisis
        failure_modes.append({
            "mode": "Liquidity Crisis",
            "description": "Market liquidity dries up",
            "probability": "Low (5-10%)",
            "impact": "High (-15% to -30%)",
            "early_warning_signals": [
                "Widening bid-ask spreads",
                "Decreasing market depth",
                "Increasing price impact"
            ],
            "mitigation": "Limit illiquid positions, maintain cash buffer"
        })

        # Failure mode 4: Policy error
        failure_modes.append({
            "mode": "Policy Error",
            "description": "Central bank or government makes policy mistake",
            "probability": "Medium (20-30%)",
            "impact": "Medium (-10% to -25%)",
            "early_warning_signals": [
                "Policy uncertainty increasing",
                "Divergence from historical policy frameworks",
                "Political pressure on independent institutions"
            ],
            "mitigation": "Diversify across regions, consider policy-neutral strategies"
        })

        # Failure mode 5: Execution failure
        actions = recommendation.get("actions", [])
        if any(a.get("action") in ["buy", "sell"] for a in actions):
            failure_modes.append({
                "mode": "Execution Failure",
                "description": "Trades cannot be executed as planned",
                "probability": "Low (5-15%)",
                "impact": "Low to Medium (-5% to -15%)",
                "early_warning_signals": [
                    "Decreasing liquidity",
                    "Increasing volatility",
                    "Market stress"
                ],
                "mitigation": "Use limit orders, scale into positions, allow execution flexibility"
            })

        return failure_modes

    async def _synthesize_concerns(
        self,
        challenges: List[AdversarialChallenge],
        biases: List[BiasDetection],
        failure_modes: List[Dict[str, Any]]
    ) -> List[str]:
        """Synthesize key concerns from all adversarial testing."""
        concerns = []

        # High-severity challenges
        severe_challenges = [c for c in challenges if c.severity >= 0.7]
        if severe_challenges:
            concerns.append(
                f"Found {len(severe_challenges)} high-severity challenge(s) to recommendation"
            )

        # High-severity biases
        severe_biases = [b for b in biases if b.severity >= 0.7]
        if severe_biases:
            bias_names = [b.bias_type.value for b in severe_biases]
            concerns.append(
                f"Detected potential cognitive biases: {', '.join(bias_names)}"
            )

        # High-impact failure modes
        high_impact_failures = [
            f for f in failure_modes
            if "High" in f.get("impact", "") or "Very High" in f.get("impact", "")
        ]
        if high_impact_failures:
            failure_names = [f["mode"] for f in high_impact_failures]
            concerns.append(
                f"High-impact failure modes identified: {', '.join(failure_names)}"
            )

        # Model risk warning
        model_risk_challenges = [c for c in challenges if c.challenge_type == "model_risk"]
        if model_risk_challenges:
            concerns.append(
                "Backtest results may be optimistic; real-world performance likely worse"
            )

        # Liquidity warning
        liquidity_challenges = [c for c in challenges if c.challenge_type == "liquidity_risk"]
        if liquidity_challenges:
            concerns.append(
                "Recommendation may have liquidity risk in stressed markets"
            )

        return concerns

    def _calculate_severity_score(
        self,
        challenges: List[AdversarialChallenge],
        biases: List[BiasDetection]
    ) -> float:
        """Calculate overall severity score (0.0 to 1.0)."""
        if not challenges and not biases:
            return 0.0

        # Weight challenges and biases equally
        challenge_severity = np.mean([c.severity for c in challenges]) if challenges else 0.0
        bias_severity = np.mean([b.severity for b in biases]) if biases else 0.0

        # Weight by presence
        challenge_weight = 0.6 if challenges else 0.0
        bias_weight = 0.4 if biases else 0.0

        total_weight = challenge_weight + bias_weight
        if total_weight == 0:
            return 0.0

        return (
            challenge_severity * challenge_weight +
            bias_severity * bias_weight
        ) / total_weight

    def _generate_adversarial_recommendation(
        self,
        challenges: List[AdversarialChallenge],
        biases: List[BiasDetection],
        failure_modes: List[Dict[str, Any]]
    ) -> str:
        """Generate final recommendation based on adversarial testing."""
        severity = self._calculate_severity_score(challenges, biases)

        if severity < 0.4:
            return "PROCEED with recommendation. Adversarial testing found no major concerns."

        elif severity < 0.6:
            return (
                "PROCEED WITH CAUTION. Adversarial testing found moderate concerns. "
                "Implement recommended mitigations and monitor closely."
            )

        elif severity < 0.8:
            return (
                "RECONSIDER recommendation. Adversarial testing found significant concerns. "
                "Address high-severity issues before proceeding."
            )

        else:
            return (
                "REJECT recommendation. Adversarial testing found severe concerns. "
                "Too many risks and biases; safer alternatives should be sought."
            )

    def _serialize_challenge(self, challenge: AdversarialChallenge) -> Dict[str, Any]:
        """Convert challenge to dictionary."""
        return {
            "challenge_type": challenge.challenge_type,
            "description": challenge.description,
            "severity": challenge.severity,
            "counter_argument": challenge.counter_argument,
            "potential_failure_mode": challenge.potential_failure_mode,
            "mitigation": challenge.mitigation
        }

    def _serialize_bias(self, bias: BiasDetection) -> Dict[str, Any]:
        """Convert bias to dictionary."""
        return {
            "bias_type": bias.bias_type.value,
            "description": bias.description,
            "evidence": bias.evidence,
            "severity": bias.severity,
            "correction": bias.correction
        }
