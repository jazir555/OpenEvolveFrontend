#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Importance Tracker - Track Which Features Actually Drive Alpha

Monitors feature importance across multiple evolutions to identify:
- Which features consistently appear in surviving signals
- How strongly features correlate with returns
- Which features work during crisis periods
- Feature redundancy and correlation

This information is used to:
1. Prioritize feature engineering efforts
2. Avoid redundant features
3. Build more robust signal combinations
4. Understand regime-dependent feature performance

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

from openevolve.finance.verticals.hedge_fund.schemas import (
    AlphaSignal,
    CrisisPeriod
)


logger = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    feature_name: str
    frequency: int  # How often it appears in survivors
    total_count: int  # Total opportunities to appear

    # Correlation statistics
    correlations: List[float] = field(default_factory=list)

    # Crisis performance
    crisis_worked: int = 0
    crisis_failed: int = 0

    # Aggregated metrics
    avg_correlation: float = 0.0
    crisis_success_rate: float = 0.0
    combined_score: float = 0.0

    # Metadata
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    def calculate_aggregates(self):
        """Calculate aggregated metrics."""
        if self.correlations:
            self.avg_correlation = np.mean(self.correlations)

        total_crisis = self.crisis_worked + self.crisis_failed
        if total_crisis > 0:
            self.crisis_success_rate = self.crisis_worked / total_crisis

        # Combined score: 60% correlation, 40% crisis robustness
        self.combined_score = (
            self.avg_correlation * 0.6 +
            self.crisis_success_rate * 0.4
        )


@dataclass
class FeatureCorrelation:
    """Correlation between two features."""
    feature_a: str
    feature_b: str
    correlation: float
    sample_size: int
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "correlation": self.correlation,
            "sample_size": self.sample_size,
            "last_updated": self.last_updated.isoformat()
        }


class FeatureImportanceTracker:
    """
    Track which features actually drive alpha.

    Maintains statistics across multiple evolutions to identify
    consistently predictive features.

    Usage:
        tracker = FeatureImportanceTracker()

        # After each evolution cycle
        importance = await tracker.extract_importance(signals)

        # Get feature ranking
        ranked_features = tracker.get_ranked_features()

        # Check for redundancy
        redundant = tracker.find_redundant_features(threshold=0.8)
    """

    def __init__(self):
        """Initialize the feature importance tracker."""
        self.feature_scores: Dict[str, FeatureStats] = {}
        self.crisis_importance: Dict[str, Dict[str, float]] = {}  # crisis -> feature -> score
        self.feature_correlations: Dict[Tuple[str, str], FeatureCorrelation] = {}
        self.evolution_count = 0

        self.logger = logging.getLogger(__name__)

    async def extract_importance(
        self,
        signals: List[AlphaSignal]
    ) -> Dict[str, float]:
        """
        Extract feature importance from surviving signals.

        Uses:
        1. Frequency in survivors (how often feature appears)
        2. Correlation with returns (how strongly it predicts)
        3. Crisis robustness (did it work in crises?)

        Args:
            signals: List of surviving alpha signals

        Returns:
            Dictionary mapping feature names to importance scores
        """
        self.evolution_count += 1
        self.logger.info(f"Extracting importance from {len(signals)} signals (evolution #{self.evolution_count})")

        importance = {}

        # Extract features from each signal
        for signal in signals:
            for feature_name in signal.features.features.keys():
                if feature_name not in importance:
                    importance[feature_name] = {
                        "frequency": 0,
                        "correlation": [],
                        "crisis_worked": 0,
                        "crisis_failed": 0,
                        "signals": []
                    }

                importance[feature_name]["frequency"] += 1

                # Get correlation for this feature
                corr = signal.feature_correlation.get(feature_name, 0.0)
                importance[feature_name]["correlation"].append(corr)

                # Track which signal used this feature
                importance[feature_name]["signals"].append(signal.signal_id)

                # Crisis performance
                for crisis_name, cp in signal.crisis_performance.items():
                    if cp.survived:
                        importance[feature_name]["crisis_worked"] += 1
                    else:
                        importance[feature_name]["crisis_failed"] += 1

        # Update global feature statistics
        for feature_name, stats in importance.items():
            if feature_name not in self.feature_scores:
                self.feature_scores[feature_name] = FeatureStats(
                    feature_name=feature_name,
                    frequency=0,
                    total_count=0
                )

            fs = self.feature_scores[feature_name]
            fs.frequency += stats["frequency"]
            fs.total_count += len(signals)
            fs.correlations.extend(stats["correlation"])
            fs.crisis_worked += stats["crisis_worked"]
            fs.crisis_failed += stats["crisis_failed"]
            fs.last_seen = datetime.utcnow()
            fs.calculate_aggregates()

        # Update crisis-specific importance
        self._update_crisis_importance(signals)

        # Update feature correlations
        self._update_feature_correlations(signals)

        # Calculate aggregate scores
        scores = {}
        for feature_name, stats in importance.items():
            # Average correlation
            avg_corr = np.mean(stats["correlation"]) if stats["correlation"] else 0.0

            # Crisis success rate
            total_crisis = stats["crisis_worked"] + stats["crisis_failed"]
            crisis_rate = stats["crisis_worked"] / total_crisis if total_crisis > 0 else 0.0

            # Frequency (normalized by number of signals)
            frequency = stats["frequency"] / len(signals)

            # Combined score (you can adjust weights)
            score = avg_corr * 0.5 + crisis_rate * 0.3 + frequency * 0.2
            scores[feature_name] = score

        return scores

    def _update_crisis_importance(self, signals: List[AlphaSignal]):
        """Update crisis-specific feature importance."""
        for signal in signals:
            for crisis_name, cp in signal.crisis_performance.items():
                if crisis_name not in self.crisis_importance:
                    self.crisis_importance[crisis_name] = {}

                for feature_name in signal.features.features.keys():
                    if feature_name not in self.crisis_importance[crisis_name]:
                        self.crisis_importance[crisis_name][feature_name] = {
                            "worked": 0,
                            "failed": 0,
                            "alpha": []
                        }

                    if cp.survived and cp.alpha_pct > 0:
                        self.crisis_importance[crisis_name][feature_name]["worked"] += 1
                        self.crisis_importance[crisis_name][feature_name]["alpha"].append(cp.alpha_pct)
                    else:
                        self.crisis_importance[crisis_name][feature_name]["failed"] += 1

    def _update_feature_correlations(self, signals: List[AlphaSignal]):
        """Update feature correlation matrix."""
        # For each signal, calculate correlation between its features
        for signal in signals:
            feature_names = list(signal.features.features.keys())

            # Calculate pairwise correlations
            for i, feat_a in enumerate(feature_names):
                for feat_b in feature_names[i+1:]:
                    # Create sorted tuple for dictionary key
                    key = tuple(sorted([feat_a, feat_b]))

                    # In a real implementation, you'd calculate actual correlation
                    # from historical data. Here we use a placeholder.
                    correlation = signal.feature_correlation.get(feat_a, 0.0) * \
                                  signal.feature_correlation.get(feat_b, 0.0)

                    if key not in self.feature_correlations:
                        self.feature_correlations[key] = FeatureCorrelation(
                            feature_a=feat_a,
                            feature_b=feat_b,
                            correlation=correlation,
                            sample_size=1
                        )
                    else:
                        # Update with new data point
                        fc = self.feature_correlations[key]
                        # Simple average (in production, use rolling correlation)
                        fc.correlation = (fc.correlation * fc.sample_size + correlation) / (fc.sample_size + 1)
                        fc.sample_size += 1
                        fc.last_updated = datetime.utcnow()

    def get_ranked_features(
        self,
        top_n: Optional[int] = None,
        min_frequency: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get features ranked by importance score.

        Args:
            top_n: Return only top N features (None for all)
            min_frequency: Minimum frequency threshold

        Returns:
            List of (feature_name, score) tuples, sorted by score
        """
        # Filter by frequency
        filtered = {
            name: stats
            for name, stats in self.feature_scores.items()
            if stats.frequency >= min_frequency
        }

        # Sort by combined score
        ranked = sorted(
            filtered.items(),
            key=lambda x: x[1].combined_score,
            reverse=True
        )

        if top_n:
            ranked = ranked[:top_n]

        return [(name, stats.combined_score) for name, stats in ranked]

    def get_crisis_robust_features(
        self,
        crisis: str,
        min_success_rate: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Get features that work well during a specific crisis.

        Args:
            crisis: Crisis period name (e.g., "gfc", "covid")
            min_success_rate: Minimum success rate threshold

        Returns:
            List of (feature_name, success_rate) tuples
        """
        if crisis not in self.crisis_importance:
            return []

        robust_features = []

        for feature_name, stats in self.crisis_importance[crisis].items():
            total = stats["worked"] + stats["failed"]
            if total == 0:
                continue

            success_rate = stats["worked"] / total
            if success_rate >= min_success_rate:
                avg_alpha = np.mean(stats["alpha"]) if stats["alpha"] else 0.0
                robust_features.append((feature_name, success_rate, avg_alpha))

        # Sort by success rate
        robust_features.sort(key=lambda x: x[1], reverse=True)

        return [(f[0], f[1]) for f in robust_features]

    def find_redundant_features(
        self,
        threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Find features that are highly correlated (redundant).

        Args:
            threshold: Correlation threshold for redundancy

        Returns:
            List of (feature_a, feature_b, correlation) tuples
        """
        redundant = []

        for (feat_a, feat_b), fc in self.feature_correlations.items():
            if abs(fc.correlation) >= threshold:
                redundant.append((feat_a, feat_b, fc.correlation))

        # Sort by correlation (highest first)
        redundant.sort(key=lambda x: abs(x[2]), reverse=True)

        return redundant

    def get_feature_report(self, feature_name: str) -> Dict[str, Any]:
        """
        Get detailed report for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with feature statistics
        """
        if feature_name not in self.feature_scores:
            return {"error": f"Feature {feature_name} not found"}

        stats = self.feature_scores[feature_name]

        report = {
            "feature_name": feature_name,
            "frequency": stats.frequency,
            "total_count": stats.total_count,
            "frequency_rate": stats.frequency / stats.total_count if stats.total_count > 0 else 0.0,
            "avg_correlation": stats.avg_correlation,
            "correlation_std": np.std(stats.correlations) if stats.correlations else 0.0,
            "crisis_success_rate": stats.crisis_success_rate,
            "combined_score": stats.combined_score,
            "first_seen": stats.first_seen.isoformat(),
            "last_seen": stats.last_seen.isoformat(),
            "crisis_performance": {}
        }

        # Add crisis-specific performance
        for crisis, crisis_stats in self.crisis_importance.items():
            if feature_name in crisis_stats:
                cs = crisis_stats[feature_name]
                total = cs["worked"] + cs["failed"]
                report["crisis_performance"][crisis] = {
                    "success_rate": cs["worked"] / total if total > 0 else 0.0,
                    "avg_alpha": np.mean(cs["alpha"]) if cs["alpha"] else 0.0,
                    "total_tests": total
                }

        return report

    def get_feature_correlations(
        self,
        feature_name: str
    ) -> List[Tuple[str, float]]:
        """
        Get correlations for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            List of (other_feature, correlation) tuples
        """
        correlations = []

        for (feat_a, feat_b), fc in self.feature_correlations.items():
            if feat_a == feature_name:
                correlations.append((feat_b, fc.correlation))
            elif feat_b == feature_name:
                correlations.append((feat_a, fc.correlation))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        return correlations

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker state to dictionary."""
        return {
            "evolution_count": self.evolution_count,
            "num_features": len(self.feature_scores),
            "ranked_features": self.get_ranked_features(top_n=20),
            "crisis_robust_features": {
                crisis: self.get_crisis_robust_features(crisis)
                for crisis in self.crisis_importance.keys()
            },
            "redundant_features": self.find_redundant_features(threshold=0.8),
            "feature_details": {
                name: self.get_feature_report(name)
                for name in list(self.feature_scores.keys())[:10]  # Top 10
            }
        }
