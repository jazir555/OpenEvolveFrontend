#!/usr/bin/env python3
"""
Bias Metrics Tracking Module for Φ₂: Metacognitive Reflection

This module provides comprehensive bias tracking metrics for the RESE Phase I debiasing system.
It implements the Confirmation Bias Index (CBI) calculation and tracking across epochs.

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Idempotency: Safe to run multiple times
- Structured Logging: JSON with correlation_id
- Law of UTC: All timestamps in UTC ISO-8601

Technical Manual Reference:
- Section 3.2: Confirmation Bias Index (CBI) tracking
- Section 3.2: Bias reduction measurement across epochs
"""

import os
import sys
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import time

# Add glue lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

from phase1_executor import StructuredLogger


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class BiasTrend(Enum):
    """Trend direction for bias over time"""
    IMPROVING = "improving"  # Bias decreasing
    STABLE = "stable"  # Bias constant
    WORSENING = "worsening"  # Bias increasing
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class BiasMeasurement:
    """Single bias measurement at a point in time"""
    epoch: int
    timestamp: str  # UTC ISO-8601
    confirmation_bias_index: float
    initial_cbi: float
    bias_reduction: float
    hypotheses_count: int
    correlation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasMeasurement':
        return cls(**data)


@dataclass
class BiasMetricsSummary:
    """Summary of bias metrics across epochs"""
    total_epochs: int
    current_cbi: float
    average_cbi: float
    min_cbi: float
    max_cbi: float
    cbi_trend: BiasTrend
    total_bias_reduction: float
    average_bias_reduction: float
    best_epoch: int
    worst_epoch: int
    measurements: List[BiasMeasurement]
    timestamp: str  # UTC ISO-8601

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['cbi_trend'] = self.cbi_trend.value
        data['measurements'] = [m.to_dict() for m in self.measurements]
        return data


@dataclass
class BiasThresholdConfig:
    """Configuration for bias threshold validation"""
    WARNING_THRESHOLD: float  # CBI level that triggers warning
    CRITICAL_THRESHOLD: float  # CBI level that requires action
    TARGET_THRESHOLD: float  # Target CBI to achieve
    MIN_IMPROVEMENT_RATE: float  # Minimum improvement rate per epoch

    @classmethod
    def from_env(cls) -> 'BiasThresholdConfig':
        """Load configuration from environment"""
        return cls(
            WARNING_THRESHOLD=float(os.getenv('BIAS_WARNING_THRESHOLD', '0.5')),
            CRITICAL_THRESHOLD=float(os.getenv('BIAS_CRITICAL_THRESHOLD', '0.7')),
            TARGET_THRESHOLD=float(os.getenv('BIAS_TARGET_THRESHOLD', '0.3')),
            MIN_IMPROVEMENT_RATE=float(os.getenv('BIAS_MIN_IMPROVEMENT_RATE', '0.05')),
        )


# ============================================================================
# MAIN BIAS METRICS TRACKER CLASS
# ============================================================================

class BiasMetricsTracker:
    """
    Bias Metrics Tracker for Φ₂: Metacognitive Reflection

    Tracks Confirmation Bias Index (CBI) across epochs and calculates trends.
    Provides metrics for bias reduction effectiveness.

    From RESE Manual §3.2:
    "CBI tracking across epochs enables measurement of debiasing effectiveness"
    """

    def __init__(self, logger: Optional[StructuredLogger] = None):
        """Initialize Bias Metrics Tracker

        Args:
            logger: Structured logger instance (created if None)
        """
        self.logger = logger or StructuredLogger('BiasMetricsTracker')
        self.measurements: List[BiasMeasurement] = []
        self.config = BiasThresholdConfig.from_env()

        self.logger.info("BiasMetricsTracker initialized",
            warning_threshold=self.config.WARNING_THRESHOLD,
            critical_threshold=self.config.CRITICAL_THRESHOLD,
            target_threshold=self.config.TARGET_THRESHOLD,
        )

    def record_measurement(
        self,
        epoch: int,
        confirmation_bias_index: float,
        initial_cbi: float,
        bias_reduction: float,
        hypotheses_count: int,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BiasMeasurement:
        """
        Record a bias measurement

        Law of Idempotency: Safe to record multiple times

        Args:
            epoch: Current epoch number
            confirmation_bias_index: Final CBI after debiasing
            initial_cbi: Initial CBI before debiasing
            bias_reduction: Percentage reduction in bias
            hypotheses_count: Number of hypotheses debiased
            correlation_id: Correlation ID for tracing
            metadata: Additional metadata

        Returns:
            BiasMeasurement object
        """
        measurement = BiasMeasurement(
            epoch=epoch,
            timestamp=datetime.now(timezone.utc).isoformat(),  # Law of UTC
            confirmation_bias_index=confirmation_bias_index,
            initial_cbi=initial_cbi,
            bias_reduction=bias_reduction,
            hypotheses_count=hypotheses_count,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        self.measurements.append(measurement)

        self.logger.info("Bias measurement recorded",
            epoch=epoch,
            cbi=confirmation_bias_index,
            initial_cbi=initial_cbi,
            bias_reduction=bias_reduction,
            hypotheses_count=hypotheses_count,
        )

        return measurement

    def calculate_summary(self) -> BiasMetricsSummary:
        """
        Calculate comprehensive summary of bias metrics

        Returns:
            BiasMetricsSummary with trends and statistics
        """
        if not self.measurements:
            # Return empty summary
            return BiasMetricsSummary(
                total_epochs=0,
                current_cbi=0.0,
                average_cbi=0.0,
                min_cbi=0.0,
                max_cbi=0.0,
                cbi_trend=BiasTrend.UNKNOWN,
                total_bias_reduction=0.0,
                average_bias_reduction=0.0,
                best_epoch=0,
                worst_epoch=0,
                measurements=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Calculate statistics
        cbis = [m.confirmation_bias_index for m in self.measurements]
        bias_reductions = [m.bias_reduction for m in self.measurements]

        total_epochs = len(self.measurements)
        current_cbi = cbis[-1]
        average_cbi = sum(cbis) / total_epochs
        min_cbi = min(cbis)
        max_cbi = max(cbis)

        # Find best and worst epochs
        best_idx = cbis.index(min_cbi)
        worst_idx = cbis.index(max_cbi)
        best_epoch = self.measurements[best_idx].epoch
        worst_epoch = self.measurements[worst_idx].epoch

        # Calculate total and average bias reduction
        total_bias_reduction = sum(bias_reductions)
        average_bias_reduction = total_bias_reduction / total_epochs

        # Determine trend
        cbi_trend = self._calculate_trend(cbis)

        summary = BiasMetricsSummary(
            total_epochs=total_epochs,
            current_cbi=current_cbi,
            average_cbi=average_cbi,
            min_cbi=min_cbi,
            max_cbi=max_cbi,
            cbi_trend=cbi_trend,
            total_bias_reduction=total_bias_reduction,
            average_bias_reduction=average_bias_reduction,
            best_epoch=best_epoch,
            worst_epoch=worst_epoch,
            measurements=self.measurements.copy(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.logger.debug("Bias metrics summary calculated",
            total_epochs=total_epochs,
            current_cbi=current_cbi,
            average_cbi=average_cbi,
            trend=cbi_trend.value,
        )

        return summary

    def _calculate_trend(self, cbis: List[float]) -> BiasTrend:
        """
        Calculate bias trend over time

        Args:
            cbis: List of CBI values in chronological order

        Returns:
            BiasTrend enum value
        """
        if len(cbis) < 2:
            return BiasTrend.UNKNOWN

        # Calculate simple linear regression slope
        n = len(cbis)
        x = list(range(n))
        y = cbis

        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        # Slope = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
        denominator = (n * sum_x2 - sum_x ** 2)
        if denominator == 0:
            return BiasTrend.STABLE

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Determine trend based on slope
        # Negative slope = improving (bias decreasing)
        # Positive slope = worsening (bias increasing)
        if slope < -self.config.MIN_IMPROVEMENT_RATE:
            return BiasTrend.IMPROVING
        elif slope > self.config.MIN_IMPROVEMENT_RATE:
            return BiasTrend.WORSENING
        else:
            return BiasTrend.STABLE

    def check_thresholds(self, cbi: float) -> Dict[str, Any]:
        """
        Check if CBI exceeds warning or critical thresholds

        Args:
            cbi: Confirmation Bias Index to check

        Returns:
            Dict with threshold check results
        """
        status = "ok"
        alerts = []

        if cbi >= self.config.CRITICAL_THRESHOLD:
            status = "critical"
            alerts.append(f"CBI ({cbi:.4f}) exceeds critical threshold ({self.config.CRITICAL_THRESHOLD})")
        elif cbi >= self.config.WARNING_THRESHOLD:
            status = "warning"
            alerts.append(f"CBI ({cbi:.4f}) exceeds warning threshold ({self.config.WARNING_THRESHOLD})")
        elif cbi <= self.config.TARGET_THRESHOLD:
            status = "target"
            alerts.append(f"CBI ({cbi:.4f}) meets target threshold ({self.config.TARGET_THRESHOLD})")

        result = {
            'status': status,
            'cbi': cbi,
            'warning_threshold': self.config.WARNING_THRESHOLD,
            'critical_threshold': self.config.CRITICAL_THRESHOLD,
            'target_threshold': self.config.TARGET_THRESHOLD,
            'alerts': alerts,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        if alerts:
            self.logger.warn("Bias threshold check", status=status, alerts=alerts)

        return result

    def get_improvement_rate(self, window_size: int = 5) -> float:
        """
        Calculate average improvement rate over recent epochs

        Args:
            window_size: Number of recent epochs to consider

        Returns:
            Average improvement rate (negative = bias decreasing)
        """
        if len(self.measurements) < 2:
            return 0.0

        # Get recent measurements
        recent = self.measurements[-window_size:]

        # Calculate average improvement
        improvements = []
        for i in range(1, len(recent)):
            improvement = recent[i].confirmation_bias_index - recent[i-1].confirmation_bias_index
            improvements.append(improvement)

        if not improvements:
            return 0.0

        return sum(improvements) / len(improvements)

    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics in canonical format

        Returns:
            Dict with all metrics data
        """
        summary = self.calculate_summary()

        return {
            'summary': summary.to_dict(),
            'config': {
                'warning_threshold': self.config.WARNING_THRESHOLD,
                'critical_threshold': self.config.CRITICAL_THRESHOLD,
                'target_threshold': self.config.TARGET_THRESHOLD,
                'min_improvement_rate': self.config.MIN_IMPROVEMENT_RATE,
            },
            'current_threshold_check': self.check_thresholds(summary.current_cbi),
            'improvement_rate': self.get_improvement_rate(),
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
        }

    def clear_history(self):
        """Clear all measurement history

        Law of Idempotency: Safe operation
        """
        count = len(self.measurements)
        self.measurements.clear()

        self.logger.info("Bias measurement history cleared",
            measurements_cleared=count,
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_cbi(
    hypothesis_confidence: float,
    antithetical_confidences: List[float],
) -> float:
    """
    Calculate Confirmation Bias Index (CBI)

    From RESE Manual §3.2:
    CBI = |P(H|E) - P(H̄|E)|

    Where:
    - P(H|E) = Probability of hypothesis given evidence
    - P(H̄|E) = Probability of opposite hypothesis

    Args:
        hypothesis_confidence: Confidence in original hypothesis (P(H|E))
        antithetical_confidences: Confidences in antithetical outcomes

    Returns:
        CBI value (0.0 = unbiased, 1.0 = fully biased)
    """
    if not antithetical_confidences:
        return 1.0  # Maximum bias if no alternatives

    # Average confidence of antithetical outcomes
    p_h_bar_given_e = sum(antithetical_confidences) / len(antithetical_confidences)

    # Calculate CBI
    cbi = abs(hypothesis_confidence - p_h_bar_given_e)

    return cbi


def calculate_bias_reduction(
    initial_cbi: float,
    final_cbi: float,
) -> float:
    """
    Calculate bias reduction percentage

    Args:
        initial_cbi: CBI before debiasing
        final_cbi: CBI after debiasing

    Returns:
        Percentage reduction (0-100)
    """
    if initial_cbi == 0:
        return 0.0

    reduction = ((initial_cbi - final_cbi) / initial_cbi) * 100
    return max(0.0, reduction)  # Ensure non-negative


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Bias Metrics Tracker')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to simulate')
    parser.add_argument('--export', help='Export metrics to JSON file')
    args = parser.parse_args()

    # Create tracker
    tracker = BiasMetricsTracker()

    # Simulate measurements
    print(f"Simulating {args.epochs} epochs of bias measurements...")
    for epoch in range(1, args.epochs + 1):
        # Simulate decreasing CBI (improving bias)
        initial_cbi = 0.8 - (epoch * 0.05)
        final_cbi = initial_cbi * (0.7 + (epoch * 0.02))
        bias_reduction = calculate_bias_reduction(initial_cbi, final_cbi)

        tracker.record_measurement(
            epoch=epoch,
            confirmation_bias_index=final_cbi,
            initial_cbi=initial_cbi,
            bias_reduction=bias_reduction,
            hypotheses_count=5,
            correlation_id=str(uuid.uuid4()),
        )

    # Calculate summary
    summary = tracker.calculate_summary()

    print("\n" + "=" * 60)
    print("BIAS METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Epochs: {summary.total_epochs}")
    print(f"Current CBI: {summary.current_cbi:.4f}")
    print(f"Average CBI: {summary.average_cbi:.4f}")
    print(f"Min CBI: {summary.min_cbi:.4f} (Epoch {summary.best_epoch})")
    print(f"Max CBI: {summary.max_cbi:.4f} (Epoch {summary.worst_epoch})")
    print(f"Trend: {summary.cbi_trend.value}")
    print(f"Total Bias Reduction: {summary.total_bias_reduction:.2f}%")
    print(f"Average Bias Reduction: {summary.average_bias_reduction:.2f}%")
    print("=" * 60)

    # Threshold check
    threshold_check = tracker.check_thresholds(summary.current_cbi)
    print(f"\nThreshold Check: {threshold_check['status'].upper()}")
    if threshold_check['alerts']:
        for alert in threshold_check['alerts']:
            print(f"  - {alert}")

    # Export if requested
    if args.export:
        metrics = tracker.export_metrics()
        with open(args.export, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics exported to: {args.export}")


if __name__ == '__main__':
    main()
