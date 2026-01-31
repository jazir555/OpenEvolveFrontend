#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Signal Evolver - Discover Trading Signals That Survive Crises

Evolves alpha signals that avoid overfitting to delisted microcaps and
survive multiple crisis periods (Dotcom, GFC, COVID, 2022 Inflation).

This module implements the core alpha discovery logic combining:
1. LoongFlow planning for feature engineering hypotheses
2. Evolutionary algorithms for signal optimization
3. Survivorship-bias-free backtesting
4. Crisis period validation

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

# Optional LoongFlow integration
try:
    from loongflow.framework.pes.pes_agent import PESAgent
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False

from openevolve.finance.verticals.hedge_fund.schemas import (
    AlphaSignal,
    AlphaDiscoveryResult,
    BacktestResult,
    CrisisPerformance,
    CrisisPeriod,
    AlphaSource,
    FeatureSet,
    FeatureHypothesis,
    SignalConstraints,
    SignalEvolutionResult,
    EnsembleSignal
)
from openevolve.finance.verticals.hedge_fund.feature_importance import FeatureImportanceTracker
from openevolve.finance.verticals.hedge_fund.signal_combiner import SignalCombiner


logger = logging.getLogger(__name__)


class MockBacktester:
    """Mock backtester for demonstration (replace with real implementation)."""

    async def run_parallel(
        self,
        strategies: List[Dict[str, Any]],
        universe: str,
        period: str,
        include_delisted: bool = True
    ) -> List["BacktestResult"]:
        """
        Run backtests on multiple strategies in parallel.

        Args:
            strategies: List of strategy definitions
            universe: Stock universe (e.g., "russell_3000")
            period: Time period for backtest (e.g., "2000-2026")
            include_delisted: Whether to include delisted stocks

        Returns:
            List of backtest results
        """
        # TODO: Implement actual backtesting logic
        # For now, return mock results
        results = []
        for i, strategy in enumerate(strategies):
            # Generate plausible mock results
            results.append(BacktestResult(
                signal_id=strategy.get("signal_id", f"signal_{i}"),
                universe=universe,
                start_date=datetime(2000, 1, 1),
                end_date=datetime(2026, 1, 1),
                include_delisted=include_delisted,
                total_return=np.random.uniform(0.5, 2.0),
                annual_return=np.random.uniform(0.05, 0.15),
                sharpe_ratio=np.random.uniform(0.8, 2.0),
                sortino_ratio=np.random.uniform(1.0, 2.5),
                information_ratio=np.random.uniform(0.3, 1.5),
                alpha=np.random.uniform(0.02, 0.08),
                beta=np.random.uniform(0.8, 1.2),
                tracking_error=np.random.uniform(0.05, 0.15),
                max_drawdown=np.random.uniform(-0.40, -0.10),
                volatility=np.random.uniform(0.10, 0.25),
                var_95=np.random.uniform(-0.03, -0.01),
                avg_market_cap=np.random.uniform(500_000_000, 10_000_000_000),
                turnover=np.random.uniform(0.3, 0.8),
                trading_costs=np.random.uniform(0.001, 0.005),
                delisting_rate=np.random.uniform(0.0, 0.02),
                survivorship_bias=np.random.uniform(-0.05, 0.05),
                crisis_performance=self._generate_mock_crisis_performance(),
                returns_by_period=self._generate_mock_period_returns()
            ))
        return results

    def _generate_mock_crisis_performance(self) -> Dict[str, CrisisPerformance]:
        """Generate mock crisis performance."""
        crises = {
            "dotcom": CrisisPerformance(
                crisis=CrisisPeriod.DOTCOM,
                start_date=datetime(2000, 3, 1),
                end_date=datetime(2002, 10, 1),
                return_pct=np.random.uniform(-0.10, 0.15),
                alpha_pct=np.random.uniform(-0.05, 0.10),
                max_drawdown_pct=np.random.uniform(-0.30, -0.10),
                volatility_pct=np.random.uniform(0.15, 0.30),
                survived=np.random.choice([True, False], p=[0.7, 0.3]),
                notes="Mock crisis performance"
            ),
            "gfc": CrisisPerformance(
                crisis=CrisisPeriod.GFC,
                start_date=datetime(2008, 9, 1),
                end_date=datetime(2009, 3, 1),
                return_pct=np.random.uniform(-0.15, 0.10),
                alpha_pct=np.random.uniform(-0.05, 0.08),
                max_drawdown_pct=np.random.uniform(-0.35, -0.15),
                volatility_pct=np.random.uniform(0.20, 0.40),
                survived=np.random.choice([True, False], p=[0.6, 0.4]),
                notes="Mock crisis performance"
            ),
            "covid": CrisisPerformance(
                crisis=CrisisPeriod.COVID,
                start_date=datetime(2020, 2, 1),
                end_date=datetime(2020, 12, 1),
                return_pct=np.random.uniform(-0.10, 0.20),
                alpha_pct=np.random.uniform(-0.03, 0.12),
                max_drawdown_pct=np.random.uniform(-0.25, -0.10),
                volatility_pct=np.random.uniform(0.25, 0.45),
                survived=np.random.choice([True, False], p=[0.8, 0.2]),
                notes="Mock crisis performance"
            ),
            "inflation_2022": CrisisPerformance(
                crisis=CrisisPeriod.INFLATION_2022,
                start_date=datetime(2022, 1, 1),
                end_date=datetime(2022, 12, 31),
                return_pct=np.random.uniform(-0.10, 0.15),
                alpha_pct=np.random.uniform(-0.05, 0.10),
                max_drawdown_pct=np.random.uniform(-0.20, -0.08),
                volatility_pct=np.random.uniform(0.15, 0.30),
                survived=np.random.choice([True, False], p=[0.7, 0.3]),
                notes="Mock crisis performance"
            )
        }
        return crises

    def _generate_mock_period_returns(self) -> Dict[str, float]:
        """Generate mock period-by-period returns."""
        periods = [
            "2000-2002", "2003-2007", "2008-2009", "2010-2019",
            "2020", "2021", "2022", "2023-2026"
        ]
        return {p: np.random.uniform(-0.15, 0.20) for p in periods}


class AlphaSignalEvolver:
    """
    Evolve alpha signals that survive crises and delistings.

    Key Features:
    - Survivorship-bias-free backtesting (includes delisted stocks)
    - Crisis period validation (2000, 2008, 2020, 2022)
    - Feature importance learning (what actually drives alpha)
    - Multi-signal combination (ensemble of surviving signals)

    Usage:
        evolver = AlphaSignalEvolver(config={
            "data_source": "CRSP_API",
            "include_delisted": True
        })

        result = await evolver.discover_alpha_signals(
            universe="russell_3000",
            constraints=SignalConstraints(
                min_market_cap=500_000_000,
                max_turnover=50,
                crisis_periods=["dotcom", "gfc", "covid"]
            )
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Alpha Signal Evolver.

        Args:
            config: Configuration dictionary
                - data_source: Data source for backtesting (e.g., "CRSP_API")
                - include_delisted: Whether to include delisted stocks
                - loongflow_config: Optional LoongFlow PES configuration
        """
        self.config = config or {}
        self.data_source = self.config.get("data_source", "MOCK")
        self.include_delisted = self.config.get("include_delisted", True)

        # Initialize components
        self.feature_importance = FeatureImportanceTracker()
        self.signal_combiner = SignalCombiner()
        self.backtester = MockBacktester()

        # Optional LoongFlow integration
        self.loongflow = None
        if LOONGFLOW_AVAILABLE:
            lf_config = self.config.get("loongflow_config", {})
            if lf_config:
                try:
                    self.loongflow = PESAgent(lf_config)
                    logger.info("LoongFlow integration enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize LoongFlow: {e}")

        self.logger = logging.getLogger(__name__)

    async def discover_alpha_signals(
        self,
        universe: str,
        constraints: SignalConstraints,
        num_hypotheses: int = 10
    ) -> AlphaDiscoveryResult:
        """
        Discover alpha signals that survive crises and delistings.

        Args:
            universe: Stock universe (e.g., "russell_3000", "sp_500")
            constraints: Signal constraints
            num_hypotheses: Number of feature hypotheses to test

        Returns:
            Surviving signals with performance metrics
        """
        self.logger.info(f"Starting alpha discovery for {universe}")

        # === PLAN PHASE: Generate feature hypotheses ===
        feature_hypotheses = await self._plan_feature_engineering(
            universe=universe,
            constraints=constraints,
            num_hypotheses=num_hypotheses
        )

        self.logger.info(f"Generated {len(feature_hypotheses)} feature hypotheses")

        # === EXECUTE PHASE: Evolve signals ===
        evolved_signals = []
        total_variants = 0

        for i, hypothesis in enumerate(feature_hypotheses):
            self.logger.info(f"Testing hypothesis {i+1}/{len(feature_hypotheses)}: {hypothesis.feature_name}")

            result = await self._evolve_signal(
                hypothesis=hypothesis,
                universe=universe,
                constraints=constraints
            )
            evolved_signals.append(result)
            total_variants += len(result.all_valid_signals)

        # === SUMMARIZE PHASE: Select survivors, extract insights ===
        survivors = await self._select_survivors(
            signals=evolved_signals,
            constraints=constraints
        )

        self.logger.info(f"Found {len(survivors)} surviving signals")

        # Extract feature importance
        importance = await self.feature_importance.extract_importance(survivors)

        # Calculate crisis performance
        crisis_perf = self._calculate_crisis_performance(survivors)

        # Combine surviving signals
        ensemble = await self.signal_combiner.combine(survivors)

        return AlphaDiscoveryResult(
            surviving_signals=survivors,
            ensemble_signal=ensemble,
            feature_importance=importance,
            crisis_performance=crisis_perf,
            total_hypotheses_tested=len(feature_hypotheses),
            total_variants_tested=total_variants,
            overall_survival_rate=len(survivors) / total_variants if total_variants > 0 else 0.0
        )

    async def _plan_feature_engineering(
        self,
        universe: str,
        constraints: SignalConstraints,
        num_hypotheses: int
    ) -> List[FeatureHypothesis]:
        """
        Generate feature engineering hypotheses.

        Uses LoongFlow if available, otherwise uses predefined hypotheses.
        """
        if self.loongflow:
            return await self._plan_with_loongflow(universe, constraints, num_hypotheses)
        else:
            return self._get_predefined_hypotheses(universe, constraints)

    async def _plan_with_loongflow(
        self,
        universe: str,
        constraints: SignalConstraints,
        num_hypotheses: int
    ) -> List[FeatureHypothesis]:
        """Use LoongFlow to generate feature hypotheses."""
        prompt = f"""
        You are generating alpha signal hypotheses for {universe}.

        Constraints:
        - Min market cap: ${constraints.min_market_cap:,}
        - Max turnover: {constraints.max_turnover}%
        - Must survive: {', '.join(constraints.crisis_periods)}

        Generate {num_hypotheses} feature hypotheses based on:
        1. Academic literature (momentum, value, quality, low vol)
        2. Alternative data (credit spreads, options flow, sentiment)
        3. Combinations that survived past crises

        For each hypothesis, specify:
        - Feature definition (formula)
        - Expected alpha source (behavioral, risk-based, etc.)
        - Why it might survive crises
        - Potential failure modes

        Focus on features that:
        - Have fundamental rationale (not data mining)
        - Avoid delisted microcaps
        - Have worked across market regimes
        """

        # Call LoongFlow
        loongflow_result = await self.loongflow.plan(
            task="generate_alpha_features",
            prompt=prompt
        )

        # Parse LoongFlow result into hypotheses
        # TODO: Implement actual parsing logic
        return self._get_predefined_hypotheses(universe, constraints)

    def _get_predefined_hypotheses(
        self,
        universe: str,
        constraints: SignalConstraints
    ) -> List[FeatureHypothesis]:
        """Get predefined feature hypotheses."""
        hypotheses = [
            FeatureHypothesis(
                hypothesis_id="momentum_12m",
                feature_name="12-Month Momentum",
                feature_definition="price_return[-252:-21]",  # Skip last month
                alpha_source=AlphaSource.BEHAVIORAL,
                rationale="Investors underreact to information, leading to price trends",
                crisis_resistance="Momentum can crash in crises but recovers quickly",
                failure_modes=["Momentum crashes", "High turnover", "Crowded trade"],
                expected_correlation=0.15,
                confidence=0.7
            ),
            FeatureHypothesis(
                hypothesis_id="value_fcf_yield",
                feature_name="Free Cash Flow Yield",
                feature_definition="fcf / enterprise_value",
                alpha_source=AlphaSource.RISK_BASED,
                rationale="Value stocks have higher expected returns (risk premium)",
                crisis_resistance="Value tends to outperform after crises",
                failure_modes=["Value traps", "Prolonged underperformance", "Accounting issues"],
                expected_correlation=0.12,
                confidence=0.75
            ),
            FeatureHypothesis(
                hypothesis_id="quality_roic",
                feature_name="Return on Invested Capital",
                feature_definition="nopat / (debt + equity - cash)",
                alpha_source=AlphaSource.QUALITY,
                rationale="High-quality companies have sustainable competitive advantages",
                crisis_resistance="Quality companies have stronger balance sheets",
                failure_modes=["Competitive disruption", "Mean reversion", "High valuations"],
                expected_correlation=0.10,
                confidence=0.65
            ),
            FeatureHypothesis(
                hypothesis_id="low_volatility",
                feature_name="Low Volatility Anomaly",
                feature_definition="1 / (id_volatility * sqrt(252))",
                alpha_source=AlphaSource.BEHAVIORAL,
                rationale="Investors prefer high-volatility stocks (lottery tickets), creating mispricing",
                crisis_resistance="Low vol stocks tend to fall less in crises",
                failure_modes=["Volatility clustering", "Interest rate sensitivity"],
                expected_correlation=0.08,
                confidence=0.6
            ),
            FeatureHypothesis(
                hypothesis_id="earnings_momentum",
                feature_name="Earnings Momentum",
                feature_definition="SUE = (eps_actual - eps_expected) / std_dev",
                alpha_source=AlphaSource.BEHAVIORAL,
                rationale="Analyst estimates are slow to update, creating predictable patterns",
                crisis_resistance="Earnings surprises can persist in downturns",
                failure_modes=["Guidance gaming", "Mean reversion", "Quarterly noise"],
                expected_correlation=0.11,
                confidence=0.7
            ),
            FeatureHypothesis(
                hypothesis_id="credit_spread_trend",
                feature_name="Credit Spread Trend",
                feature_definition="(credit_spread_t-20 - credit_spread_t-0) / credit_spread_t-20",
                alpha_source=AlphaSource.INFORMATION,
                rationale="Credit markets lead equity markets (debt holders more sophisticated)",
                crisis_resistance="Credit spreads are early warning indicators",
                failure_modes=["Liquidity premium", "Spread noise", "Data availability"],
                expected_correlation=0.13,
                confidence=0.6
            ),
            FeatureHypothesis(
                hypothesis_id="share_repurchase",
                feature_name="Share Repurchase Intensity",
                feature_definition="-shares_outstanding_change__pct",
                alpha_source=AlphaSource.STRUCTURAL,
                rationale="Insiders buy back when stock is undervalued (signaling)",
                crisis_resistance="Repurchases signal confidence to market",
                failure_modes=["Debt-funded buybacks", "Poor timing", "Accounting effects"],
                expected_correlation=0.09,
                confidence=0.55
            ),
            FeatureHypothesis(
                hypothesis_id="analyst_revision",
                feature_name="Analyst Revision Momentum",
                feature_definition="sum(eps_estimate_revision[-60:-0])",
                alpha_source=AlphaSource.INFORMATION,
                rationale="Analyst revisions contain new information",
                crisis_resistance="Analyst cuts can be leading indicators",
                failure_modes=["Herding", "Delayed revisions", "Conflicts of interest"],
                expected_correlation=0.10,
                confidence=0.5
            ),
            FeatureHypothesis(
                hypothesis_id="dividend_growth",
                feature_name="Dividend Growth",
                feature_definition="dividend_per_share_change_1y",
                alpha_source=AlphaSource.QUALITY,
                rationale="Dividend increases signal management confidence",
                crisis_resistance="Dividend payers tend to be more stable",
                failure_modes=["Payout ratio constraints", "Sector bias", "Tax sensitivity"],
                expected_correlation=0.07,
                confidence=0.6
            ),
            FeatureHypothesis(
                hypothesis_id="multi_signal_momentum_value",
                feature_name="Momentum + Value Combo",
                feature_definition="rank(momentum_12m) * rank(value_fcf_yield)",
                alpha_source=AlphaSource.COMBINATION,
                rationale="Combines two independent alpha sources with low correlation",
                crisis_resistance="Diversified across factors reduces crash risk",
                failure_modes=["Factor timing", "Correlation breakdown", "Complexity"],
                expected_correlation=0.18,
                confidence=0.8
            )
        ]

        return hypotheses

    async def _evolve_signal(
        self,
        hypothesis: FeatureHypothesis,
        universe: str,
        constraints: SignalConstraints
    ) -> SignalEvolutionResult:
        """
        Evolve signal variants for a hypothesis.

        Generates multiple variants of the signal, backtests them,
        and filters by constraints.
        """
        # Generate signal variants
        variants = self._generate_signal_variants(
            hypothesis=hypothesis,
            n_variants=20
        )

        # Backtest on survivorship-free data
        backtest_results = await self.backtester.run_parallel(
            strategies=variants,
            universe=universe,
            period="2000-2026",
            include_delisted=self.include_delisted  # Critical!
        )

        # Filter by constraints
        valid_signals = []

        for variant, result in zip(variants, backtest_results):
            # Check market cap constraint
            if result.avg_market_cap < constraints.min_market_cap:
                continue  # Too much microcap exposure

            # Check turnover constraint
            if result.turnover > constraints.max_turnover / 100:
                continue  # Too expensive to trade

            # Check delisting rate
            if result.delisting_rate > constraints.max_delisting_rate:
                continue  # Too much delisting risk

            # Check crisis survival
            crisis_survival = self._check_crisis_survival(
                result.crisis_performance,
                constraints.crisis_periods
            )

            if not crisis_survival["all_survived"]:
                continue  # Failed in crisis

            # Check Sharpe ratio
            if result.sharpe_ratio < constraints.min_sharpe:
                continue  # Not enough alpha

            # Check max drawdown
            if result.max_drawdown < constraints.max_drawdown:
                continue  # Too much risk

            # Calculate Information Ratio
            ir = result.information_ratio

            # Create AlphaSignal
            signal = AlphaSignal(
                signal_id=variant.get("signal_id", f"{hypothesis.hypothesis_id}_{len(valid_signals)}"),
                name=f"{hypothesis.feature_name} - Variant {len(valid_signals)+1}",
                description=f"Variant of {hypothesis.feature_name}",
                formula=hypothesis.feature_definition,
                features=FeatureSet(
                    features={hypothesis.feature_name: hypothesis.feature_definition},
                    feature_correlations={hypothesis.feature_name: hypothesis.expected_correlation}
                ),
                alpha_source=hypothesis.alpha_source,
                rationale=hypothesis.rationale,
                backtest=result,
                sharpe_ratio=result.sharpe_ratio,
                information_ratio=ir,
                crisis_performance=result.crisis_performance,
                feature_correlation={hypothesis.feature_name: hypothesis.expected_correlation}
            )

            valid_signals.append({
                "signal": signal,
                "backtest": result,
                "ir": ir,
                "crisis_performance": crisis_survival
            })

        # Rank by Information Ratio
        valid_signals.sort(key=lambda x: x["ir"], reverse=True)

        return SignalEvolutionResult(
            hypothesis=hypothesis,
            best_signal=valid_signals[0]["signal"] if valid_signals else None,
            all_valid_signals=valid_signals,
            survival_rate=len(valid_signals) / len(variants) if variants else 0.0
        )

    def _generate_signal_variants(
        self,
        hypothesis: FeatureHypothesis,
        n_variants: int
    ) -> List[Dict[str, Any]]:
        """
        Generate variants of a signal hypothesis.

        Creates variations by:
        - Adjusting lookback periods
        - Changing normalization methods
        - Adding filters
        """
        variants = []

        for i in range(n_variants):
            variant = {
                "signal_id": f"{hypothesis.hypothesis_id}_v{i}",
                "hypothesis_id": hypothesis.hypothesis_id,
                "feature_name": hypothesis.feature_name,
                "feature_definition": hypothesis.feature_definition,
                # Add variant-specific parameters
                "lookback_period": np.random.choice([20, 60, 126, 252]),
                "normalization": np.random.choice(["zscore", "rank", "percentile"]),
                "filter_market_cap": np.random.choice([True, False]),
                "filter_sector": np.random.choice([True, False])
            }
            variants.append(variant)

        return variants

    def _check_crisis_survival(
        self,
        crisis_performance: Dict[str, CrisisPerformance],
        required_crises: List[str]
    ) -> Dict[str, Any]:
        """
        Check if signal survived all required crises.

        Returns:
            Dictionary with survival status for each crisis
        """
        survival_status = {
            "all_survived": True,
            "crises": {}
        }

        for crisis_name in required_crises:
            if crisis_name not in crisis_performance:
                survival_status["all_survived"] = False
                survival_status["crises"][crisis_name] = {
                    "survived": False,
                    "reason": "No data"
                }
            else:
                cp = crisis_performance[crisis_name]
                survived = cp.survived and cp.alpha_pct > -0.10  # Must have positive alpha or small loss

                survival_status["crises"][crisis_name] = {
                    "survived": survived,
                    "return": cp.return_pct,
                    "alpha": cp.alpha_pct,
                    "max_drawdown": cp.max_drawdown_pct
                }

                if not survived:
                    survival_status["all_survived"] = False

        return survival_status

    async def _select_survivors(
        self,
        signals: List[SignalEvolutionResult],
        constraints: SignalConstraints
    ) -> List[AlphaSignal]:
        """
        Select surviving signals from all evolution results.

        Ranks by Information Ratio and filters by constraints.
        """
        all_signals = []

        for result in signals:
            if result.best_signal:
                all_signals.append(result.best_signal)

        # Sort by Information Ratio
        all_signals.sort(key=lambda s: s.information_ratio, reverse=True)

        # Return top signals (you might want to keep more)
        return all_signals[:20]

    def _calculate_crisis_performance(
        self,
        signals: List[AlphaSignal]
    ) -> Dict[str, Dict[str, CrisisPerformance]]:
        """
        Calculate aggregate crisis performance across signals.
        """
        crisis_perf = {}

        for signal in signals:
            for crisis_name, cp in signal.crisis_performance.items():
                if crisis_name not in crisis_perf:
                    crisis_perf[crisis_name] = []
                crisis_perf[crisis_name].append(cp)

        return crisis_perf
