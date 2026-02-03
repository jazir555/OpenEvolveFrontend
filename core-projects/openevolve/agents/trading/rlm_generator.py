#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLM Strategy Generator

Uses Reasoning via Language Model for trading strategy ideation.
Decomposes trading problems, generates strategy variations, identifies
key parameters, and creates strategy hypotheses.

Key capabilities:
- Multi-step reasoning about strategy design
- Decomposition of complex trading problems
- Parameter space exploration
- Hypothesis generation
- Market regime adaptation

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import json
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional
import uuid

from openevolve.agents.trading.schemas import (
    Strategy,
    EvolutionState,
    StrategyType
)


logger = logging.getLogger(__name__)


class RLMGenerator:
    """
    Strategy Generator using Reasoning via Language Model.

    Applies systematic reasoning to generate high-quality trading strategies:
    1. Analyze market conditions
    2. Identify opportunities
    3. Generate strategy hypotheses
    4. Define parameter spaces
    5. Create concrete strategy definitions

    Usage:
        generator = RLMGenerator(knowledge_engine=ke)

        strategies = await generator.generate_strategies(
            market_regime={"regime": "bull", "volatility": "low"},
            num_ideas=5,
            current_state=state
        )
    """

    def __init__(self, knowledge_engine=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RLM Generator.

        Args:
            knowledge_engine: Optional knowledge engine for learning
            config: Additional configuration
        """
        self.knowledge_engine = knowledge_engine
        self.config = config or {}

        # Strategy templates
        self.strategy_templates = self._initialize_templates()

        # Reasoning prompts
        self.prompts = self._initialize_prompts()

        logger.info("RLM Generator initialized")

    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize strategy templates for different types."""
        return {
            "momentum": {
                "description": "Exploits persistence of price trends",
                "key_parameters": ["lookback_period", "threshold", "position_sizing"],
                "entry_logic": [
                    "price > moving_average(lookback)",
                    "momentum > threshold",
                    "volume confirmation"
                ],
                "exit_logic": [
                    "momentum < -threshold",
                    "price < moving_average(lookback * 0.5)",
                    "stop_loss hit"
                ],
                "risk_rules": {
                    "max_position_size": 0.1,
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15
                }
            },
            "mean_reversion": {
                "description": "Exploits tendency of prices to return to mean",
                "key_parameters": ["lookback_period", "entry_threshold", "exit_threshold"],
                "entry_logic": [
                    "price < bollinger_lower * entry_threshold",
                    "rsi < 30",
                    "price deviation from mean > 2 * std"
                ],
                "exit_logic": [
                    "price > moving_average",
                    "price > bollinger_middle",
                    "target_profit reached"
                ],
                "risk_rules": {
                    "max_position_size": 0.15,
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.08
                }
            },
            "statistical_arbitrage": {
                "description": "Exploits statistical mispricings between related assets",
                "key_parameters": ["correlation_window", "zscore_threshold", "holding_period"],
                "entry_logic": [
                    "zscore(spread) > threshold",
                    "cointegration test passes",
                    "correlation > minimum"
                ],
                "exit_logic": [
                    "zscore(spread) < 0",
                    "holding period exceeded",
                    "cointegration breaks"
                ],
                "risk_rules": {
                    "max_position_size": 0.2,
                    "pairs_limit": 5,
                    "stop_loss_pct": 0.04
                }
            },
            "trend_following": {
                "description": "Identifies and follows major market trends",
                "key_parameters": ["fast_ma", "slow_ma", "trend_strength"],
                "entry_logic": [
                    "fast_ma > slow_ma",
                    "ADX > trend_strength",
                    "no recent drawdown"
                ],
                "exit_logic": [
                    "fast_ma < slow_ma",
                    "trend reversal signal",
                    "trailing stop hit"
                ],
                "risk_rules": {
                    "max_position_size": 0.2,
                    "stop_loss_pct": 0.08,
                    "trailing_stop_pct": 0.05
                }
            },
            "pairs_trading": {
                "description": "Trades pairs of correlated assets",
                "key_parameters": ["pair_selection", "entry_zscore", "exit_zscore"],
                "entry_logic": [
                    "spread zscore > entry_threshold",
                    "hedge ratio calculated",
                    "both assets liquid"
                ],
                "exit_logic": [
                    "spread zscore < exit_threshold",
                    "max holding period reached",
                    "correlation breaks down"
                ],
                "risk_rules": {
                    "max_position_size": 0.25,
                    "max_pairs": 3,
                    "stop_loss_pct": 0.06
                }
            }
        }

    def _initialize_prompts(self) -> Dict[str, str]:
        """Initialize reasoning prompts."""
        return {
            "analyze_market": """
Analyze the current market conditions and identify trading opportunities.

Market Regime: {regime}
Volatility: {volatility}
Trend: {trend}

Consider:
1. What market inefficiencies exist?
2. What types of strategies would perform well?
3. What are the key risks?
4. What data sources would be valuable?

Provide structured analysis with reasoning.
""",

            "generate_strategy": """
Generate a trading strategy hypothesis based on analysis.

Market Analysis: {analysis}
Strategy Type: {strategy_type}

Provide:
1. Clear hypothesis about market behavior
2. Entry and exit logic
3. Key parameters and ranges
4. Risk management approach
5. Expected market conditions
6. Potential failure modes

Be specific and actionable.
""",

            "refine_parameters": """
Refine strategy parameters based on performance.

Strategy: {strategy}
Performance: {performance}
Market Regime: {regime}

Identify:
1. Which parameters are most important?
2. Optimal ranges for each parameter?
3. Parameter interactions?
4. Suggested variations to test?

Provide data-driven recommendations.
""",

            "combine_strategies": """
Combine multiple strategies into a hybrid approach.

Strategies: {strategies}
Performance Data: {performance}

Design:
1. How to combine signals from each strategy?
2. When to use which strategy?
3. Portfolio allocation approach?
4. Risk management across strategies?
5. Expected benefits of combination?

Provide specific implementation guidance.
"""
        }

    async def generate_strategies(
        self,
        market_regime: Dict[str, Any],
        num_ideas: int = 5,
        current_state: Optional[EvolutionState] = None
    ) -> List[Strategy]:
        """
        Generate new trading strategy ideas.

        Args:
            market_regime: Current market regime classification
            num_ideas: Number of strategies to generate
            current_state: Current evolution state

        Returns:
            List of generated strategies
        """
        logger.info(f"Generating {num_ideas} strategy ideas for regime {market_regime}")

        strategies = []

        # Step 1: Analyze market conditions
        market_analysis = await self._analyze_market_conditions(market_regime)

        # Step 2: Identify suitable strategy types
        suitable_types = await self._identify_strategy_types(
            market_regime,
            market_analysis
        )

        # Step 3: Generate strategies for each type
        for strategy_type in suitable_types[:num_ideas]:
            strategy = await self._generate_strategy_of_type(
                strategy_type,
                market_regime,
                market_analysis,
                current_state
            )

            if strategy:
                strategies.append(strategy)

        logger.info(f"Generated {len(strategies)} strategies")

        return strategies

    async def _analyze_market_conditions(
        self,
        market_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze market conditions using reasoning.

        Args:
            market_regime: Market regime data

        Returns:
            Market analysis with opportunities and risks
        """
        # Use reasoning to analyze market
        analysis_prompt = self.prompts["analyze_market"].format(
            regime=market_regime.get("regime", "unknown"),
            volatility=market_regime.get("volatility", "unknown"),
            trend=market_regime.get("trend", "unknown")
        )

        # In a real implementation, this would call an LLM
        # For now, use rule-based reasoning
        analysis = await self._rule_based_market_analysis(market_regime)

        logger.info(f"Market analysis: {len(analysis.get('opportunities', []))} opportunities")

        return analysis

    async def _rule_based_market_analysis(
        self,
        market_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Rule-based market analysis (placeholder for LLM)."""
        regime = market_regime.get("regime", "sideways")
        volatility = market_regime.get("volatility", "medium")

        opportunities = []
        risks = []

        if regime == "bull":
            opportunities.extend([
                "Momentum strategies likely to work",
                "Trend following potential",
                "Breakout strategies"
            ])
            risks.extend([
                "Sudden reversals",
                "Overextended valuations"
            ])

        elif regime == "bear":
            opportunities.extend([
                "Short selling opportunities",
                "Volatility strategies",
                "Safe haven assets"
            ])
            risks.extend([
                "Short squeezes",
                "Policy interventions"
            ])

        elif regime == "sideways":
            opportunities.extend([
                "Mean reversion strategies",
                "Range trading",
                "Options income strategies"
            ])
            risks.extend([
                "False breakouts",
                "Low volatility premiums"
            ])

        if volatility == "high":
            opportunities.append("Statistical arbitrage opportunities")
            risks.append("Increased transaction costs")

        return {
            "opportunities": opportunities,
            "risks": risks,
            "recommended_approach": self._get_recommended_approach(regime, volatility),
            "data_requirements": self._get_data_requirements(regime)
        }

    def _get_recommended_approach(self, regime: str, volatility: str) -> str:
        """Get recommended trading approach."""
        if regime == "bull":
            return "trend_following"
        elif regime == "bear":
            return "momentum" if volatility == "high" else "short_selling"
        else:
            return "mean_reversion"

    def _get_data_requirements(self, regime: str) -> List[str]:
        """Get data requirements for regime."""
        base_requirements = ["price", "volume"]

        if regime in ["bull", "bear"]:
            base_requirements.extend(["momentum_indicators", "trend_indicators"])
        else:
            base_requirements.extend(["volatility_indicators", "mean_reversion_indicators"])

        return base_requirements

    async def _identify_strategy_types(
        self,
        market_regime: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Identify suitable strategy types for current conditions.

        Args:
            market_regime: Market regime
            market_analysis: Market analysis

        Returns:
            List of suitable strategy types
        """
        regime = market_regime.get("regime", "sideways")
        volatility = market_regime.get("volatility", "medium")

        # Map regimes to strategy types
        regime_strategies = {
            "bull": ["momentum", "trend_following", "breakout"],
            "bear": ["momentum", "short_selling", "volatility"],
            "sideways": ["mean_reversion", "pairs_trading", "market_making"],
            "high_volatility": ["statistical_arbitrage", "options_strategy"],
            "low_volatility": ["momentum", "trend_following"]
        }

        suitable_types = regime_strategies.get(regime, ["momentum", "mean_reversion"])

        if volatility == "high":
            suitable_types.extend(["statistical_arbitrage", "options_strategy"])

        logger.info(f"Suitable strategy types: {suitable_types}")

        return suitable_types

    async def _generate_strategy_of_type(
        self,
        strategy_type: str,
        market_regime: Dict[str, Any],
        market_analysis: Dict[str, Any],
        current_state: Optional[EvolutionState]
    ) -> Optional[Strategy]:
        """
        Generate a specific strategy of given type.

        Args:
            strategy_type: Type of strategy to generate
            market_regime: Current market regime
            market_analysis: Market analysis
            current_state: Evolution state

        Returns:
            Generated strategy or None
        """
        try:
            # Get template
            template = self.strategy_templates.get(strategy_type)
            if not template:
                logger.warning(f"No template for strategy type: {strategy_type}")
                return None

            # Generate parameters
            parameters = await self._generate_parameters(
                strategy_type,
                template["key_parameters"],
                market_regime
            )

            # Create strategy
            strategy = Strategy(
                strategy_id=str(uuid.uuid4()),
                name=f"{strategy_type}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
                description=template["description"],
                strategy_type=StrategyType(strategy_type),
                parameters=parameters,
                entry_conditions=template["entry_logic"],
                exit_conditions=template["exit_logic"],
                risk_rules=template["risk_rules"],
                metadata={
                    "market_regime": market_regime,
                    "market_analysis": market_analysis,
                    "generation_method": "rlm",
                    "generated_at": datetime.now(UTC).isoformat()
                }
            )

            logger.info(f"Generated strategy {strategy.strategy_id} of type {strategy_type}")

            return strategy

        except Exception as e:
            logger.error(f"Error generating strategy of type {strategy_type}: {e}")
            return None

    async def _generate_parameters(
        self,
        strategy_type: str,
        key_parameters: List[str],
        market_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate strategy parameters with reasonable ranges.

        Args:
            strategy_type: Strategy type
            key_parameters: Key parameter names
            market_regime: Market regime

        Returns:
            Parameter dictionary
        """
        parameters = {}

        # Define parameter ranges for different types
        parameter_ranges = {
            "lookback_period": (5, 50),
            "threshold": (0.5, 3.0),
            "position_sizing": (0.01, 0.3),
            "entry_threshold": (1.5, 3.0),
            "exit_threshold": (0.0, 1.0),
            "fast_ma": (5, 20),
            "slow_ma": (20, 100),
            "trend_strength": (20, 50),
            "correlation_window": (20, 60),
            "zscore_threshold": (1.5, 2.5),
            "holding_period": (1, 10)
        }

        # Generate parameters with regime-specific adjustments
        for param in key_parameters:
            if param in parameter_ranges:
                min_val, max_val = parameter_ranges[param]

                # Adjust for regime
                if market_regime.get("volatility") == "high":
                    min_val *= 0.8
                    max_val *= 1.2

                # Use reasoning to pick good starting point
                value = await self._reason_parameter_value(param, min_val, max_val)
                parameters[param] = value
            else:
                # Default value
                parameters[param] = 1.0

        return parameters

    async def _reason_parameter_value(
        self,
        param_name: str,
        min_val: float,
        max_val: float
    ) -> float:
        """
        Use reasoning to select good parameter value.

        Args:
            param_name: Parameter name
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Selected parameter value
        """
        # Simple heuristic: use midpoint with some variation
        # In real implementation, would use LLM reasoning
        midpoint = (min_val + max_val) / 2

        # Add some variation based on parameter type
        if "period" in param_name or "window" in param_name:
            # Prefer round numbers for periods
            return round(midpoint)
        elif "threshold" in param_name:
            # Prefer slightly conservative thresholds
            return midpoint * 0.9
        else:
            return midpoint

    async def refine_strategy(
        self,
        strategy: Strategy,
        performance_feedback: Dict[str, Any],
        market_regime: Dict[str, Any]
    ) -> Strategy:
        """
        Refine a strategy based on performance feedback.

        Args:
            strategy: Strategy to refine
            performance_feedback: Performance metrics and feedback
            market_regime: Current market regime

        Returns:
            Refined strategy
        """
        logger.info(f"Refining strategy {strategy.strategy_id}")

        # Analyze what worked and what didn't
        analysis = await self._analyze_performance_feedback(
            strategy,
            performance_feedback
        )

        # Adjust parameters
        refined_parameters = await self._adjust_parameters(
            strategy.parameters,
            analysis,
            market_regime
        )

        # Create refined strategy
        refined_strategy = Strategy(
            strategy_id=str(uuid.uuid4()),
            name=f"{strategy.name}_refined",
            description=f"Refined version of {strategy.description}",
            strategy_type=strategy.strategy_type,
            parameters=refined_parameters,
            entry_conditions=strategy.entry_conditions,
            exit_conditions=strategy.exit_conditions,
            risk_rules=strategy.risk_rules,
            metadata={
                "parent_strategy": strategy.strategy_id,
                "refinement_reasoning": analysis,
                "performance_feedback": performance_feedback,
                "refined_at": datetime.now(UTC).isoformat()
            }
        )

        logger.info(f"Created refined strategy {refined_strategy.strategy_id}")

        return refined_strategy

    async def _analyze_performance_feedback(
        self,
        strategy: Strategy,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance feedback."""
        return {
            "strengths": feedback.get("strengths", []),
            "weaknesses": feedback.get("weaknesses", []),
            "parameter_sensitivity": feedback.get("parameter_sensitivity", {}),
            "market_conditions": feedback.get("market_conditions", {})
        }

    async def _adjust_parameters(
        self,
        current_parameters: Dict[str, float],
        analysis: Dict[str, Any],
        market_regime: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adjust parameters based on analysis."""
        adjusted = current_parameters.copy()

        # Adjust based on sensitivity analysis
        sensitivity = analysis.get("parameter_sensitivity", {})
        for param, sensitivity_value in sensitivity.items():
            if param in adjusted:
                if sensitivity_value > 0.7:  # High sensitivity
                    # Reduce parameter value to be more conservative
                    adjusted[param] *= 0.9
                elif sensitivity_value < 0.3:  # Low sensitivity
                    # Increase parameter value to explore more
                    adjusted[param] *= 1.1

        return adjusted

    async def combine_strategies(
        self,
        strategies: List[Strategy],
        performance_data: List[Dict[str, Any]]
    ) -> Strategy:
        """
        Combine multiple strategies into a hybrid.

        Args:
            strategies: Strategies to combine
            performance_data: Performance data for each strategy

        Returns:
            Hybrid strategy
        """
        logger.info(f"Combining {len(strategies)} strategies into hybrid")

        # Calculate weights based on performance
        total_performance = sum(p.get("sharpe_ratio", 0) for p in performance_data)
        weights = [
            p.get("sharpe_ratio", 0) / total_performance if total_performance > 0 else 1.0 / len(strategies)
            for p in performance_data
        ]

        # Combine parameters
        combined_parameters = {}
        for param in strategies[0].parameters.keys():
            combined_value = sum(
                weight * s.parameters.get(param, 0)
                for weight, s in zip(weights, strategies)
            )
            combined_parameters[param] = combined_value

        # Combine entry conditions
        combined_entry = []
        for strategy, weight in zip(strategies, weights):
            if weight > 0.3:  # Only include significant contributors
                combined_entry.extend(strategy.entry_conditions)

        # Create hybrid strategy
        hybrid_strategy = Strategy(
            strategy_id=str(uuid.uuid4()),
            name=f"hybrid_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            description=f"Hybrid of {len(strategies)} strategies",
            strategy_type=StrategyType.HYBRID,
            parameters=combined_parameters,
            entry_conditions=list(set(combined_entry)),  # Remove duplicates
            exit_conditions=strategies[0].exit_conditions,  # Use first strategy's exits
            risk_rules=strategies[0].risk_rules,  # Use first strategy's risk rules
            metadata={
                "component_strategies": [s.strategy_id for s in strategies],
                "weights": weights,
                "combination_method": "weighted_average",
                "created_at": datetime.now(UTC).isoformat()
            }
        )

        logger.info(f"Created hybrid strategy {hybrid_strategy.strategy_id}")

        return hybrid_strategy
