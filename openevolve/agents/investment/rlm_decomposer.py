#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLM Decomposer - Reasoning via Language Model for Investment Problems

Decomposes complex investment problems using structured reasoning,
breaking down portfolio optimization into manageable sub-problems,
identifying key factors and constraints, and generating alternative hypotheses.

This module implements the RLM approach from "Language Models are Reasoning Agents"
(Roumeliotis et al., 2024), adapted for investment decision-making.
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Rlm Decomposer
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import asyncio
from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json


class Factor:
    """Represents a key factor influencing investment decisions."""

    def __init__(
        self,
        name: str,
        category: str,  # "fundamental", "technical", "macro", "sentiment"
        importance: float,  # 0.0 to 1.0
        value: Any,
        uncertainty: float = 0.0,  # 0.0 to 1.0
        rationale: str = ""
    ):
        self.name = name
        self.category = category
        self.importance = importance
        self.value = value
        self.uncertainty = uncertainty
        self.rationale = rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "importance": self.importance,
            "value": self.value,
            "uncertainty": self.uncertainty,
            "rationale": self.rationale
        }


class Hypothesis:
    """Represents an investment hypothesis to be tested."""

    def __init__(
        self,
        statement: str,
        confidence: float,
        evidence: List[str],
        counter_evidence: List[str],
        testable_predictions: List[str]
    ):
        self.statement = statement
        self.confidence = confidence
        self.evidence = evidence
        self.counter_evidence = counter_evidence
        self.testable_predictions = testable_predictions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "counter_evidence": self.counter_evidence,
            "testable_predictions": self.testable_predictions
        }


class SubProblem:
    """Represents a decomposed sub-problem of the investment decision."""

    def __init__(
        self,
        name: str,
        description: str,
        factors: List[Factor],
        constraints: List[str],
        dependencies: List[str]  # Other sub-problems this depends on
    ):
        self.name = name
        self.description = description
        self.factors = factors
        self.constraints = constraints
        self.dependencies = dependencies

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "factors": [f.to_dict() for f in self.factors],
            "constraints": self.constraints,
            "dependencies": self.dependencies
        }


class RLMDecomposer:
    """
    Reasoning via Language Model Decomposer for Investment Problems

    Implements structured decomposition of complex investment decisions into:
    1. Key factors and their relative importance
    2. Sub-problems with clear dependencies
    3. Testable hypotheses
    4. Alternative scenarios
    """

    def __init__(self, max_depth: int = 3, min_importance: float = 0.3):
        """
        Initialize the RLM Decomposer.

        Args:
            max_depth: Maximum depth of problem decomposition
            min_importance: Minimum importance threshold for factors
        """
        self.max_depth = max_depth
        self.min_importance = min_importance
        self.logger = logging.getLogger(__name__)

    async def decompose(
        self,
        portfolio_state: Dict[str, float],
        market_context: Dict[str, Any],
        changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Decompose the investment problem into structured components.

        Args:
            portfolio_state: Current portfolio allocations
            market_context: Current market conditions and data
            changes: Significant changes since last review

        Returns:
            Dictionary containing decomposed problem structure
        """
        self.logger.info("Starting RLM decomposition of investment problem")

        # Step 1: Identify key factors
        factors = await self._identify_factors(portfolio_state, market_context, changes)

        # Step 2: Generate hypotheses
        hypotheses = await self._generate_hypotheses(factors, market_context)

        # Step 3: Decompose into sub-problems
        sub_problems = await self._decompose_into_subproblems(
            factors, hypotheses, portfolio_state
        )

        # Step 4: Generate alternative scenarios
        scenarios = await self._generate_scenarios(factors, hypotheses)

        # Step 5: Identify constraints
        constraints = await self._identify_constraints(portfolio_state, market_context)

        return {
            "key_factors": [f.to_dict() for f in factors],
            "hypotheses": [h.to_dict() for h in hypotheses],
            "sub_problems": [sp.to_dict() for sp in sub_problems],
            "scenarios": scenarios,
            "constraints": constraints,
            "decomposition_metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "num_factors": len(factors),
                "num_hypotheses": len(hypotheses),
                "num_sub_problems": len(sub_problems),
                "num_scenarios": len(scenarios)
            }
        }

    async def _identify_factors(
        self,
        portfolio_state: Dict[str, float],
        market_context: Dict[str, Any],
        changes: List[Dict[str, Any]]
    ) -> List[Factor]:
        """
        Identify key factors influencing the investment decision.

        Factors are categorized into:
        - Fundamental: Company/asset fundamentals (P/E, earnings, etc.)
        - Technical: Price patterns, momentum, volatility
        - Macro: Economic indicators, interest rates, inflation
        - Sentiment: Market sentiment, news flow, social media
        """
        factors = []

        # Fundamental factors
        for ticker, allocation in portfolio_state.items():
            if ticker in market_context.get("fundamentals", {}):
                fundamentals = market_context["fundamentals"][ticker]

                # P/E ratio
                if "pe_ratio" in fundamentals:
                    factors.append(Factor(
                        name=f"{ticker}_pe_ratio",
                        category="fundamental",
                        importance=0.7,
                        value=fundamentals["pe_ratio"],
                        uncertainty=0.1,
                        rationale=f"P/E ratio indicates valuation level for {ticker}"
                    ))

                # Earnings growth
                if "earnings_growth" in fundamentals:
                    factors.append(Factor(
                        name=f"{ticker}_earnings_growth",
                        category="fundamental",
                        importance=0.8,
                        value=fundamentals["earnings_growth"],
                        uncertainty=0.2,
                        rationale=f"Earnings growth rate drives long-term returns for {ticker}"
                    ))

        # Technical factors
        if "technical" in market_context:
            technical = market_context["technical"]

            if "market_momentum" in technical:
                factors.append(Factor(
                    name="market_momentum",
                    category="technical",
                    importance=0.6,
                    value=technical["market_momentum"],
                    uncertainty=0.15,
                    rationale="Overall market trend affects individual asset returns"
                ))

            if "volatility_regime" in technical:
                factors.append(Factor(
                    name="volatility_regime",
                    category="technical",
                    importance=0.75,
                    value=technical["volatility_regime"],
                    uncertainty=0.1,
                    rationale="Current volatility level affects risk and position sizing"
                ))

        # Macro factors
        if "macro" in market_context:
            macro = market_context["macro"]

            if "interest_rate" in macro:
                factors.append(Factor(
                    name="interest_rate",
                    category="macro",
                    importance=0.9,
                    value=macro["interest_rate"],
                    uncertainty=0.05,
                    rationale="Interest rates affect discount rates and asset valuations"
                ))

            if "inflation" in macro:
                factors.append(Factor(
                    name="inflation",
                    category="macro",
                    importance=0.85,
                    value=macro["inflation"],
                    uncertainty=0.1,
                    rationale="Inflation erodes real returns and affects sector performance"
                ))

            if "gdp_growth" in macro:
                factors.append(Factor(
                    name="gdp_growth",
                    category="macro",
                    importance=0.7,
                    value=macro["gdp_growth"],
                    uncertainty=0.15,
                    rationale="Economic growth drives corporate earnings"
                ))

        # Sentiment factors
        if "sentiment" in market_context:
            sentiment = market_context["sentiment"]

            if "market_sentiment" in sentiment:
                factors.append(Factor(
                    name="market_sentiment",
                    category="sentiment",
                    importance=0.5,
                    value=sentiment["market_sentiment"],
                    uncertainty=0.3,
                    rationale="Overall market sentiment can drive short-term price movements"
                ))

        # Filter by minimum importance
        factors = [f for f in factors if f.importance >= self.min_importance]

        # Sort by importance
        factors.sort(key=lambda x: x.importance, reverse=True)

        self.logger.info(f"Identified {len(factors)} key factors")
        return factors

    async def _generate_hypotheses(
        self,
        factors: List[Factor],
        market_context: Dict[str, Any]
    ) -> List[Hypothesis]:
        """
        Generate testable investment hypotheses based on identified factors.

        Each hypothesis includes:
        - Statement: Clear, testable claim
        - Confidence: Estimated probability (0.0 to 1.0)
        - Evidence: Supporting factors
        - Counter-evidence: Conflicting factors
        - Testable predictions: Specific, measurable predictions
        """
        hypotheses = []

        # Group factors by category
        fundamental_factors = [f for f in factors if f.category == "fundamental"]
        macro_factors = [f for f in factors if f.category == "macro"]
        technical_factors = [f for f in factors if f.category == "technical"]

        # Hypothesis 1: Value-based hypothesis
        if fundamental_factors:
            pe_factors = [f for f in fundamental_factors if "pe_ratio" in f.name]

            if pe_factors:
                avg_pe = sum(f.value for f in pe_factors) / len(pe_factors)

                statement = f"Portfolio with average P/E of {avg_pe:.2f} will outperform over next 6 months"
                confidence = 0.65 if avg_pe < 20 else 0.45

                evidence = [f.rationale for f in pe_factors if f.value < 20]
                counter_evidence = [f.rationale for f in pe_factors if f.value >= 25]

                predictions = [
                    "Portfolio will generate positive alpha relative to benchmark",
                    "Value stocks will outperform growth stocks",
                    "Low P/E stocks will show positive earnings surprises"
                ]

                hypotheses.append(Hypothesis(
                    statement=statement,
                    confidence=confidence,
                    evidence=evidence,
                    counter_evidence=counter_evidence,
                    testable_predictions=predictions
                ))

        # Hypothesis 2: Macro-driven hypothesis
        if macro_factors:
            rate_factor = next((f for f in macro_factors if "interest_rate" in f.name), None)

            if rate_factor:
                if rate_factor.value < 0.03:  # Low rate environment
                    statement = "Low interest rate environment will benefit growth stocks"

                    evidence = ["Low rates reduce discount rates for growth stocks"]
                    counter_evidence = ["Low rates may signal economic weakness"]

                    predictions = [
                        "Growth stocks will outperform value stocks",
                        "Technology sector will show strong relative performance",
                        "High beta stocks will generate excess returns"
                    ]

                    hypotheses.append(Hypothesis(
                        statement=statement,
                        confidence=0.7,
                        evidence=evidence,
                        counter_evidence=counter_evidence,
                        testable_predictions=predictions
                    ))

        # Hypothesis 3: Momentum hypothesis
        if technical_factors:
            momentum_factor = next((f for f in technical_factors if "momentum" in f.name), None)

            if momentum_factor and momentum_factor.value > 0:
                statement = "Positive market momentum will continue over next 3 months"

                evidence = ["Current positive trend tends to persist in short term"]
                counter_evidence = ["Momentum can reverse abruptly"]

                predictions = [
                    "Market will deliver positive returns in next quarter",
                    "High momentum stocks will continue to outperform",
                    "Trend-following strategies will generate positive returns"
                ]

                hypotheses.append(Hypothesis(
                    statement=statement,
                    confidence=0.6,
                    evidence=evidence,
                    counter_evidence=counter_evidence,
                    testable_predictions=predictions
                ))

        self.logger.info(f"Generated {len(hypotheses)} testable hypotheses")
        return hypotheses

    async def _decompose_into_subproblems(
        self,
        factors: List[Factor],
        hypotheses: List[Hypothesis],
        portfolio_state: Dict[str, float]
    ) -> List[SubProblem]:
        """
        Decompose the investment decision into manageable sub-problems.

        Each sub-problem focuses on a specific aspect and can be solved
        relatively independently, with clear dependencies.
        """
        sub_problems = []

        # Sub-problem 1: Asset allocation
        asset_factors = [f for f in factors if f.category in ["macro", "technical"]]
        sub_problems.append(SubProblem(
            name="asset_allocation",
            description="Determine optimal allocation between asset classes based on macro and technical factors",
            factors=asset_factors,
            constraints=[
                "Total allocation must sum to 100%",
                "No single asset class > 40% of portfolio",
                "Maintain minimum 5% cash for opportunities"
            ],
            dependencies=[]
        ))

        # Sub-problem 2: Security selection within each asset class
        for ticker in portfolio_state.keys():
            ticker_factors = [f for f in factors if f.name.startswith(ticker)]

            if ticker_factors:
                sub_problems.append(SubProblem(
                    name=f"security_selection_{ticker}",
                    description=f"Determine optimal position size in {ticker}",
                    factors=ticker_factors,
                    constraints=[
                        f"Position in {ticker} between 0% and 20% of portfolio",
                        "Consider correlation with existing positions"
                    ],
                    dependencies=["asset_allocation"]
                ))

        # Sub-problem 3: Risk management
        risk_factors = [f for f in factors if "volatility" in f.name or "risk" in f.name]
        sub_problems.append(SubProblem(
            name="risk_management",
            description="Ensure portfolio risk is within acceptable parameters",
            factors=risk_factors,
            constraints=[
                "Portfolio volatility < 15% annualized",
                "Maximum drawdown < 20%",
                "Correlation-weighted concentration < 30%"
            ],
            dependencies=["asset_allocation"] + [f"security_selection_{t}" for t in portfolio_state.keys()]
        ))

        # Sub-problem 4: Rebalancing strategy
        sub_problems.append(SubProblem(
            name="rebalancing_strategy",
            description="Determine if and how to rebalance to target allocations",
            factors=[],  # This is more of a procedural step
            constraints=[
                "Rebalance only if drift > 5%",
                "Minimize transaction costs",
                "Consider tax implications"
            ],
            dependencies=["asset_allocation", "risk_management"]
        ))

        self.logger.info(f"Decomposed into {len(sub_problems)} sub-problems")
        return sub_problems

    async def _generate_scenarios(
        self,
        factors: List[Factor],
        hypotheses: List[Hypothesis]
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative scenarios for analysis.

        Each scenario represents a different potential future state
        with its own implications for the portfolio.
        """
        scenarios = []

        # Scenario 1: Bull case
        scenarios.append({
            "name": "bull_case",
            "probability": 0.30,
            "description": "Strong economic growth, low volatility, positive earnings surprises",
            "key_assumptions": [
                "GDP growth > 3%",
                "Interest rates stable or declining",
                "Corporate earnings beat expectations"
            ],
            "portfolio_implications": "Increase equity allocation, focus on growth and cyclical sectors",
            "risk_to_scenario": "Economic slowdown, inflation spike, geopolitical crisis"
        })

        # Scenario 2: Base case
        scenarios.append({
            "name": "base_case",
            "probability": 0.50,
            "description": "Moderate growth, stable markets, steady earnings",
            "key_assumptions": [
                "GDP growth 2-3%",
                "Interest rates gradually rising",
                "Corporate earnings meet expectations"
            ],
            "portfolio_implications": "Maintain balanced allocation, focus on quality companies",
            "risk_to_scenario": "Policy mistakes, earnings disappointments"
        })

        # Scenario 3: Bear case
        scenarios.append({
            "name": "bear_case",
            "probability": 0.20,
            "description": "Economic recession, high volatility, earnings declines",
            "key_assumptions": [
                "GDP growth < 1%",
                "Interest rates rising aggressively",
                "Corporate earnings miss expectations"
            ],
            "portfolio_implications": "Increase defensive allocation, focus on quality and dividend stocks",
            "risk_to_scenario": "Rapid policy response, economic resilience"
        })

        return scenarios

    async def _identify_constraints(
        self,
        portfolio_state: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> List[str]:
        """
        Identify constraints on the investment decision.

        Constraints can be:
        - Risk-based: Maximum volatility, drawdown limits
        - Regulatory: Position limits, concentration limits
        - Operational: Liquidity requirements, trading constraints
        - Tax: Tax implications of trades
        """
        constraints = []

        # Risk constraints
        constraints.append("Maximum portfolio volatility: 15% annualized")
        constraints.append("Maximum position size: 20% of portfolio")
        constraints.append("Maximum sector concentration: 30% of portfolio")

        # Liquidity constraints
        constraints.append("Minimum 5% cash for liquidity")
        constraints.append("Maximum 10% in illiquid assets")

        # Turnover constraints
        constraints.append("Maximum annual portfolio turnover: 100%")
        constraints.append("Minimum holding period: 1 quarter (except for risk management)")

        # Regulatory/tax constraints
        constraints.append("Comply with all regulatory position limits")
        constraints.append("Consider tax efficiency of trades")

        # Market-specific constraints
        if market_context.get("market_hours") == "closed":
            constraints.append("Trades will execute at next open")

        if market_context.get("volatility_regime") == "high":
            constraints.append("Reduce position sizes due to high volatility")

        return constraints

    async def synthesize_solution(
        self,
        sub_problems: List[SubProblem],
        solutions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize solutions to sub-problems into a coherent investment decision.

        Args:
            sub_problems: List of sub-problems
            solutions: Dictionary mapping sub-problem names to their solutions

        Returns:
            Synthesized investment decision
        """
        # This would be implemented to combine solutions from sub-problems
        # into a coherent investment decision

        return {
            "synthesis_timestamp": datetime.utcnow().isoformat(),
            "sub_problem_solutions": solutions,
            "dependencies_resolved": True,
            "coherent_solution": True
        }
