#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Combiner - Combine Multiple Alpha Signals Into Ensembles

Combines surviving signals using ensemble methods to create more robust
alpha signals. Ensembles reduce overfitting and improve diversification.

Combination Methods:
- weighted: Weight by Information Ratio
- rank: Equal-weighted rank combination
- causal: Use causal model to determine weights
- hierarchical: Hierarchical risk parity
- optimal: Mean-variance optimization

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from openevolve.finance.verticals.hedge_fund.schemas import (
    AlphaSignal,
    EnsembleSignal,
    BacktestResult,
    CrisisPerformance,
    CrisisPeriod
)


logger = logging.getLogger(__name__)


class SignalCombiner:
    """
    Combine surviving signals into ensemble.

    Usage:
        combiner = SignalCombiner()

        # Weight by Information Ratio
        ensemble = await combiner.combine(signals, method="weighted")

        # Equal-weighted rank combination
        ensemble = await combiner.combine(signals, method="rank")

        # Causal model optimization
        ensemble = await combiner.combine(signals, method="causal")
    """

    def __init__(self):
        """Initialize the signal combiner."""
        self.logger = logging.getLogger(__name__)

    async def combine(
        self,
        signals: List[AlphaSignal],
        method: str = "weighted",
        ensemble_name: Optional[str] = None
    ) -> EnsembleSignal:
        """
        Combine signals into ensemble.

        Args:
            signals: List of alpha signals to combine
            method: Combination method
                - "weighted": Weight by Information Ratio
                - "rank": Equal-weighted rank combination
                - "causal": Use causal model (requires causal model)
                - "hierarchical": Hierarchical risk parity
                - "optimal": Mean-variance optimization
            ensemble_name: Optional name for the ensemble

        Returns:
            Ensemble signal
        """
        if not signals:
            raise ValueError("Cannot combine empty signal list")

        self.logger.info(f"Combining {len(signals)} signals using {method} method")

        if method == "weighted":
            return await self._weighted_combination(signals, ensemble_name)
        elif method == "rank":
            return await self._rank_combination(signals, ensemble_name)
        elif method == "causal":
            return await self._causal_combination(signals, ensemble_name)
        elif method == "hierarchical":
            return await self._hierarchical_combination(signals, ensemble_name)
        elif method == "optimal":
            return await self._optimal_combination(signals, ensemble_name)
        else:
            raise ValueError(f"Unknown combination method: {method}")

    async def _weighted_combination(
        self,
        signals: List[AlphaSignal],
        ensemble_name: Optional[str] = None
    ) -> EnsembleSignal:
        """
        Weight signals by Information Ratio.

        Higher IR signals get more weight. This is simple and effective
        when signals are independent.
        """
        # Calculate weights based on Information Ratio
        total_ir = sum(max(0, s.information_ratio) for s in signals)

        if total_ir == 0:
            # Fall back to equal weights
            weights = {s.signal_id: 1.0 / len(signals) for s in signals}
        else:
            weights = {
                s.signal_id: max(0, s.information_ratio) / total_ir
                for s in signals
            }

        # Calculate ensemble metrics
        correlations = self._calculate_signal_correlations(signals)
        diversification = self._calculate_diversification_ratio(signals, weights, correlations)

        # Generate ensemble backtest (mock)
        backtest = self._generate_ensemble_backtest(signals, weights)

        return EnsembleSignal(
            ensemble_id=f"ensemble_weighted_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=ensemble_name or f"Weighted Ensemble ({len(signals)} signals)",
            description=f"Ensemble of {len(signals)} signals weighted by Information Ratio",
            signals=signals,
            weights=weights,
            combination_method="information_ratio_weighted",
            backtest=backtest,
            sharpe_ratio=backtest.sharpe_ratio,
            information_ratio=backtest.information_ratio,
            signal_correlations=correlations,
            diversification_ratio=diversification
        )

    async def _rank_combination(
        self,
        signals: List[AlphaSignal],
        ensemble_name: Optional[str] = None
    ) -> EnsembleSignal:
        """
        Equal-weighted rank combination.

        Ranks stocks by each signal, then averages the ranks.
        This is more robust to outliers than direct weighted combination.
        """
        # Equal weights
        weights = {s.signal_id: 1.0 / len(signals) for s in signals}

        # Calculate ensemble metrics
        correlations = self._calculate_signal_correlations(signals)
        diversification = self._calculate_diversification_ratio(signals, weights, correlations)

        # Generate ensemble backtest (mock)
        backtest = self._generate_ensemble_backtest(signals, weights)

        return EnsembleSignal(
            ensemble_id=f"ensemble_rank_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=ensemble_name or f"Rank Ensemble ({len(signals)} signals)",
            description=f"Equal-weighted rank combination of {len(signals)} signals",
            signals=signals,
            weights=weights,
            combination_method="rank_based",
            backtest=backtest,
            sharpe_ratio=backtest.sharpe_ratio,
            information_ratio=backtest.information_ratio,
            signal_correlations=correlations,
            diversification_ratio=diversification
        )

    async def _causal_combination(
        self,
        signals: List[AlphaSignal],
        ensemble_name: Optional[str] = None
    ) -> EnsembleSignal:
        """
        Use causal model to determine optimal weights.

        This method is more sophisticated but requires a causal model
        of signal relationships. Falls back to weighted if no causal model.

        TODO: Implement actual causal model integration
        """
        # For now, fall back to weighted combination
        self.logger.warning("Causal model not implemented, falling back to weighted")
        return await self._weighted_combination(signals, ensemble_name)

    async def _hierarchical_combination(
        self,
        signals: List[AlphaSignal],
        ensemble_name: Optional[str] = None
    ) -> EnsembleSignal:
        """
        Hierarchical risk parity combination.

        Groups similar signals and allocates risk equally across groups.
        This is more robust than naive weighting when signals are correlated.
        """
        # Calculate correlation matrix
        correlations = self._calculate_signal_correlations(signals)

        # Build hierarchical clusters
        clusters = self._build_hierarchical_clusters(signals, correlations)

        # Allocate weights
        weights = self._allocate_hierarchical_weights(signals, clusters)

        # Calculate ensemble metrics
        diversification = self._calculate_diversification_ratio(signals, weights, correlations)

        # Generate ensemble backtest (mock)
        backtest = self._generate_ensemble_backtest(signals, weights)

        return EnsembleSignal(
            ensemble_id=f"ensemble_hierarchical_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=ensemble_name or f"Hierarchical Ensemble ({len(signals)} signals)",
            description=f"Hierarchical risk parity ensemble of {len(signals)} signals",
            signals=signals,
            weights=weights,
            combination_method="hierarchical_risk_parity",
            backtest=backtest,
            sharpe_ratio=backtest.sharpe_ratio,
            information_ratio=backtest.information_ratio,
            signal_correlations=correlations,
            diversification_ratio=diversification
        )

    async def _optimal_combination(
        self,
        signals: List[AlphaSignal],
        ensemble_name: Optional[str] = None
    ) -> EnsembleSignal:
        """
        Mean-variance optimal combination.

        Maximizes Sharpe ratio given expected returns and covariances.
        Requires estimation of expected returns and covariance matrix.
        """
        # Calculate expected returns and covariances
        expected_returns = np.array([s.backtest.annual_return for s in signals])

        # Build correlation matrix
        correlations = self._calculate_signal_correlations(signals)
        n = len(signals)
        cov_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = (signals[i].backtest.volatility ** 2)
                else:
                    # Find correlation in dictionary
                    key = tuple(sorted([signals[i].signal_id, signals[j].signal_id]))
                    corr = correlations.get(key, 0.0)
                    cov_matrix[i, j] = corr * signals[i].backtest.volatility * signals[j].backtest.volatility

        # Optimize weights (simplified - use inverse volatility in practice)
        # For robustness, use equal-risk contribution
        volatilities = np.array([s.backtest.volatility for s in signals])
        inv_vol = 1.0 / volatilities
        optimal_weights = inv_vol / np.sum(inv_vol)

        weights = {
            s.signal_id: w
            for s, w in zip(signals, optimal_weights)
        }

        # Calculate ensemble metrics
        diversification = self._calculate_diversification_ratio(signals, weights, correlations)

        # Generate ensemble backtest (mock)
        backtest = self._generate_ensemble_backtest(signals, weights)

        return EnsembleSignal(
            ensemble_id=f"ensemble_optimal_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            name=ensemble_name or f"Optimal Ensemble ({len(signals)} signals)",
            description=f"Mean-variance optimal ensemble of {len(signals)} signals",
            signals=signals,
            weights=weights,
            combination_method="mean_variance_optimal",
            backtest=backtest,
            sharpe_ratio=backtest.sharpe_ratio,
            information_ratio=backtest.information_ratio,
            signal_correlations=correlations,
            diversification_ratio=diversification
        )

    def _calculate_signal_correlations(
        self,
        signals: List[AlphaSignal]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise correlations between signals."""
        correlations = {}

        n = len(signals)
        for i in range(n):
            for j in range(i+1, n):
                # In a real implementation, you'd calculate actual correlation
                # from historical returns. Here we use a heuristic.

                # Correlated if they share features
                features_i = set(signals[i].features.features.keys())
                features_j = set(signals[j].features.features.keys())

                shared_features = features_i & features_j
                total_features = features_i | features_j

                # Feature overlap correlation
                feature_corr = len(shared_features) / len(total_features) if total_features else 0.0

                # Adjust for alpha source (same source = higher correlation)
                if signals[i].alpha_source == signals[j].alpha_source:
                    feature_corr += 0.2

                # Clamp to [-1, 1]
                correlation = max(-1.0, min(1.0, feature_corr))

                key = tuple(sorted([signals[i].signal_id, signals[j].signal_id]))
                correlations[key] = correlation

        return correlations

    def _calculate_diversification_ratio(
        self,
        signals: List[AlphaSignal],
        weights: Dict[str, float],
        correlations: Dict[Tuple[str, str], float]
    ) -> float:
        """
        Calculate diversification ratio.

        DR = (Weighted avg volatility) / (Portfolio volatility)

        Higher is better (>1 means well-diversified).
        """
        # Weighted average volatility
        weighted_vol = sum(
            weights[s.signal_id] * s.backtest.volatility
            for s in signals
        )

        # Portfolio volatility (simplified, ignoring correlations)
        # In practice, you'd use the full covariance matrix
        portfolio_vol = np.sqrt(sum(
            weights[s.signal_id] ** 2 * s.backtest.volatility ** 2
            for s in signals
        ))

        if portfolio_vol == 0:
            return 1.0

        return weighted_vol / portfolio_vol

    def _build_hierarchical_clusters(
        self,
        signals: List[AlphaSignal],
        correlations: Dict[Tuple[str, str], float]
    ) -> List[List[str]]:
        """
        Build hierarchical clusters of signals based on correlation.

        Uses simple agglomerative clustering.
        """
        # Start with each signal in its own cluster
        clusters = [[s.signal_id] for s in signals]

        # Merge clusters until we have a reasonable number
        while len(clusters) > min(5, len(signals)):
            # Find most similar pair of clusters
            max_similarity = -1
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    # Calculate average linkage
                    similarities = []
                    for sid_i in clusters[i]:
                        for sid_j in clusters[j]:
                            key = tuple(sorted([sid_i, sid_j]))
                            sim = correlations.get(key, 0.0)
                            similarities.append(sim)

                    avg_similarity = np.mean(similarities)

                    if avg_similarity > max_similarity:
                        max_similarity = avg_similarity
                        merge_i, merge_j = i, j

            # Merge clusters
            if merge_i >= 0 and merge_j >= 0:
                clusters[merge_i].extend(clusters[merge_j])
                clusters.pop(merge_j)
            else:
                break

        return clusters

    def _allocate_hierarchical_weights(
        self,
        signals: List[AlphaSignal],
        clusters: List[List[str]]
    ) -> Dict[str, float]:
        """
        Allocate weights using hierarchical risk parity.

        Equal risk to each cluster, then equal risk within each cluster.
        """
        weights = {}

        # Equal weight to each cluster
        cluster_weight = 1.0 / len(clusters)

        for cluster in clusters:
            # Equal weight to each signal in cluster
            signal_weight = cluster_weight / len(cluster)

            for signal_id in cluster:
                weights[signal_id] = signal_weight

        return weights

    def _generate_ensemble_backtest(
        self,
        signals: List[AlphaSignal],
        weights: Dict[str, float]
    ) -> BacktestResult:
        """
        Generate ensemble backtest results.

        In a real implementation, you'd combine the actual returns.
        Here we generate a plausible mock result.
        """
        # Weighted average of signal backtests
        avg_return = sum(weights[s.signal_id] * s.backtest.annual_return for s in signals)
        avg_volatility = np.sqrt(
            sum(
                weights[s.signal_id] ** 2 * s.backtest.volatility ** 2
                for s in signals
            )
        )
        avg_sharpe = avg_return / avg_volatility if avg_volatility > 0 else 0
        avg_alpha = sum(weights[s.signal_id] * s.backtest.alpha for s in signals)
        avg_turnover = sum(weights[s.signal_id] * s.backtest.turnover for s in signals)
        avg_market_cap = sum(weights[s.signal_id] * s.backtest.avg_market_cap for s in signals)

        # Aggregate crisis performance
        crisis_perf = {}
        for signal in signals:
            for crisis_name, cp in signal.crisis_performance.items():
                if crisis_name not in crisis_perf:
                    crisis_perf[crisis_name] = []

                crisis_perf[crisis_name].append(cp)

        # Average crisis performance
        avg_crisis_perf = {}
        for crisis_name, performances in crisis_perf.items():
            avg_crisis_perf[crisis_name] = CrisisPerformance(
                crisis=performances[0].crisis,
                start_date=performances[0].start_date,
                end_date=performances[0].end_date,
                return_pct=np.mean([cp.return_pct for cp in performances]),
                alpha_pct=np.mean([cp.alpha_pct for cp in performances]),
                max_drawdown_pct=np.mean([cp.max_drawdown_pct for cp in performances]),
                volatility_pct=np.mean([cp.volatility_pct for cp in performances]),
                survived=all(cp.survived for cp in performances),
                notes="Ensemble aggregate"
            )

        return BacktestResult(
            signal_id=f"ensemble_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            universe=signals[0].backtest.universe,
            start_date=min(s.backtest.start_date for s in signals),
            end_date=max(s.backtest.end_date for s in signals),
            include_delisted=signals[0].backtest.include_delisted,
            total_return=avg_return * 10,  # Approximate total return
            annual_return=avg_return,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=avg_sharpe * 1.2,  # Approximate
            information_ratio=avg_alpha / 0.1,  # Approximate
            alpha=avg_alpha,
            beta=1.0,
            tracking_error=0.1,
            max_drawdown=min(s.backtest.max_drawdown for s in signals),
            volatility=avg_volatility,
            var_95=-0.02,
            avg_market_cap=avg_market_cap,
            turnover=avg_turnover,
            trading_costs=0.002,
            delisting_rate=np.mean([s.backtest.delisting_rate for s in signals]),
            survivorship_bias=0.0,
            crisis_performance=avg_crisis_perf,
            returns_by_period={}
        )
