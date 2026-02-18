"""
A/B Testing Framework for OpenEvolve Knowledge Engine.

Provides experimentation capabilities for testing different strategies,
configurations, and algorithms.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import logging
import random
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ExperimentStatus(Enum):
    """Status of an A/B test"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Types of test variants"""
    CONTROL = "control"       # Baseline
    TREATMENT = "treatment"   # New variant being tested


@dataclass
class TestVariant:
    """
    A variant in an A/B test.

    Attributes:
        id: Unique identifier
        name: Variant name
        type: Variant type (control or treatment)
        config: Configuration for this variant
        traffic_allocation: Percentage of traffic (0-100)
    """
    id: str
    name: str
    type: VariantType
    config: Dict[str, Any] = field(default_factory=dict)
    traffic_allocation: int = 50  # Default to 50/50 split


@dataclass
class ExperimentResult:
    """
    Result from a single experiment execution.

    Attributes:
        variant_id: Which variant was used
        success: Whether the operation succeeded
        metrics: Dictionary of metric values
        timestamp: When result was recorded
        metadata: Additional metadata
    """
    variant_id: str
    success: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentStats:
    """
    Statistical summary for a variant.

    Attributes:
        variant_id: Variant identifier
        total_exposures: How many times this variant was shown
        total_conversions: How many times it succeeded
        conversion_rate: Success rate
        average_metrics: Average of each metric
        confidence_interval: 95% confidence interval (if calculable)
    """
    variant_id: str
    total_exposures: int
    total_conversions: int
    conversion_rate: float
    average_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_interval: Optional[tuple[float, float]] = None


@dataclass
class ExperimentSummary:
    """
    Summary of an A/B test experiment.

    Attributes:
        experiment_id: Experiment identifier
        status: Current status
        variants: Variants being tested
        results: All results collected
        stats: Statistics per variant
        winner: Winning variant (if determined)
        significance: Statistical significance (if calculable)
        started_at: When experiment started
        completed_at: When experiment completed
    """
    experiment_id: str
    status: ExperimentStatus
    variants: List[TestVariant]
    results: List[ExperimentResult] = field(default_factory=list)
    stats: Dict[str, ExperimentStats] = field(default_factory=dict)
    winner: Optional[str] = None
    significance: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================================================
# Main Framework
# ============================================================================

class ABTestFramework:
    """
    A/B Testing Framework for Knowledge Engine experiments.

    Features:
    - Multi-variant testing (A/B/n)
    - Traffic allocation
    - Result collection
    - Statistical analysis
    - Winner determination
    """

    def __init__(self):
        """Initialize the A/B testing framework."""
        self.experiments: Dict[str, ExperimentSummary] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {experiment_id: variant_id}

    def create_experiment(
        self,
        experiment_id: str,
        variants: List[TestVariant],
        auto_start: bool = True
    ) -> ExperimentSummary:
        """
        Create a new A/B test experiment.

        Args:
            experiment_id: Unique identifier
            variants: Variants to test
            auto_start: Whether to start the experiment immediately

        Returns:
            ExperimentSummary
        """
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")

        # Validate traffic allocation sums to 100
        total_allocation = sum(v.traffic_allocation for v in variants)
        if total_allocation != 100:
            raise ValueError(f"Traffic allocation must sum to 100, got {total_allocation}")

        summary = ExperimentSummary(
            experiment_id=experiment_id,
            status=ExperimentStatus.RUNNING if auto_start else ExperimentStatus.PENDING,
            variants=variants,
            started_at=datetime.now(timezone.utc) if auto_start else None
        )

        self.experiments[experiment_id] = summary

        logger.info({
            "msg": "Experiment created",
            "experiment_id": experiment_id,
            "variants": len(variants),
            "status": summary.status.value
        })

        return summary

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[TestVariant]:
        """
        Assign a variant to a user for an experiment.

        Uses consistent hashing to ensure the same user always gets
        the same variant (unless explicitly reassigned).

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier

        Returns:
            Assigned variant, or None if experiment not found
        """
        if experiment_id not in self.experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return None

        experiment = self.experiments[experiment_id]

        if experiment.status != ExperimentStatus.RUNNING:
            logger.warning(f"Experiment {experiment_id} is not running")
            return None

        # Check if user already assigned
        if experiment_id in self.user_assignments.get(user_id, {}):
            variant_id = self.user_assignments[user_id][experiment_id]
            for variant in experiment.variants:
                if variant.id == variant_id:
                    return variant

        # Assign new variant based on traffic allocation
        variant = self._select_variant(experiment, user_id)

        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant.id

        logger.debug({
            "msg": "Variant assigned",
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant_id": variant.id
        })

        return variant

    def _select_variant(
        self,
        experiment: ExperimentSummary,
        user_id: str
    ) -> TestVariant:
        """
        Select a variant based on traffic allocation.

        Uses consistent hashing for stable assignment.

        Args:
            experiment: Experiment summary
            user_id: User identifier

        Returns:
            Selected variant
        """
        # Create hash from user_id + experiment_id for stable assignment
        hash_input = f"{user_id}:{experiment.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random_value = (hash_value % 100) + 1  # 1-100

        # Select variant based on traffic allocation
        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_allocation
            if random_value <= cumulative:
                return variant

        # Fallback to last variant
        return experiment.variants[-1]

    def record_result(
        self,
        experiment_id: str,
        user_id: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record a result for an experiment.

        Args:
            experiment_id: Experiment identifier
            user_id: User who performed the action
            success: Whether the action succeeded
            metrics: Optional metrics (e.g., latency, score)
            metadata: Optional metadata

        Returns:
            True if recorded successfully
        """
        if experiment_id not in self.experiments:
            return False

        experiment = self.experiments[experiment_id]

        # Get user's assigned variant
        variant_id = self.user_assignments.get(user_id, {}).get(experiment_id)
        if not variant_id:
            logger.warning(f"No variant assigned for user {user_id} in experiment {experiment_id}")
            return False

        result = ExperimentResult(
            variant_id=variant_id,
            success=success,
            metrics=metrics or {},
            metadata=metadata or {}
        )

        experiment.results.append(result)

        logger.debug({
            "msg": "Result recorded",
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "success": success
        })

        return True

    def analyze_experiment(
        self,
        experiment_id: str
    ) -> ExperimentSummary:
        """
        Analyze an experiment and calculate statistics.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Updated experiment summary with statistics
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        # Calculate statistics for each variant
        for variant in experiment.variants:
            variant_results = [r for r in experiment.results if r.variant_id == variant.id]

            total = len(variant_results)
            conversions = sum(1 for r in variant_results if r.success)
            conversion_rate = conversions / total if total > 0 else 0.0

            # Calculate average metrics
            avg_metrics = {}
            if variant_results:
                for metric in variant_results[0].metrics.keys():
                    values = [r.metrics.get(metric, 0) for r in variant_results]
                    avg_metrics[metric] = sum(values) / len(values) if values else 0.0

            experiment.stats[variant.id] = ExperimentStats(
                variant_id=variant.id,
                total_exposures=total,
                total_conversions=conversions,
                conversion_rate=conversion_rate,
                average_metrics=avg_metrics
            )

        # Determine winner (highest conversion rate)
        if experiment.stats:
            winner_stats = max(experiment.stats.values(), key=lambda s: s.conversion_rate)
            experiment.winner = winner_stats.variant_id

        logger.info({
            "msg": "Experiment analyzed",
            "experiment_id": experiment_id,
            "winner": experiment.winner,
            "total_results": len(experiment.results)
        })

        return experiment

    def complete_experiment(
        self,
        experiment_id: str
    ) -> ExperimentSummary:
        """
        Mark an experiment as completed.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Final experiment summary
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)

        # Run final analysis
        return self.analyze_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Get an experiment by ID."""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[ExperimentSummary]:
        """
        List experiments, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ab_test(
    experiment_id: str,
    control_config: Dict[str, Any],
    treatment_config: Dict[str, Any],
    traffic_split: int = 50
) -> ExperimentSummary:
    """
    Convenience function to create a simple A/B test.

    Args:
        experiment_id: Unique identifier
        control_config: Control variant configuration
        treatment_config: Treatment variant configuration
        traffic_split: Traffic split percentage (50 = 50/50)

    Returns:
        ExperimentSummary
    """
    framework = ABTestFramework()

    variants = [
        TestVariant(
            id="control",
            name="Control",
            type=VariantType.CONTROL,
            config=control_config,
            traffic_allocation=traffic_split
        ),
        TestVariant(
            id="treatment",
            name="Treatment",
            type=VariantType.TREATMENT,
            config=treatment_config,
            traffic_allocation=100 - traffic_split
        )
    ]

    return framework.create_experiment(experiment_id, variants)


# Export all components
__all__ = [
    'ExperimentStatus',
    'VariantType',
    'TestVariant',
    'ExperimentResult',
    'ExperimentStats',
    'ExperimentSummary',
    'ABTestFramework',
    'create_ab_test'
]
