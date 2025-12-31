"""
Evaluation Metrics Collector

Collects and stores multi-dimensional evaluation metrics for
workflow artifacts and solutions.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of evaluation metrics"""
    QUALITY = "quality"              # Solution quality metrics
    PERFORMANCE = "performance"      # Performance metrics
    RELIABILITY = "reliability"      # Reliability metrics
    SECURITY = "security"            # Security metrics
    COMPLETENESS = "completeness"    # Completeness metrics
    EFFICIENCY = "efficiency"        # Efficiency metrics
    MAINTAINABILITY = "maintainability"  # Maintainability metrics
    SCALABILITY = "scalability"      # Scalability metrics


class MetricType(Enum):
    """Types of metrics"""
    # Quality metrics
    REQUIREMENTS_COVERAGE = "requirements_coverage"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION_QUALITY = "documentation_quality"

    # Performance metrics
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"

    # Reliability metrics
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    FAULT_TOLERANCE = "fault_tolerance"

    # Security metrics
    VULNERABILITY_COUNT = "vulnerability_count"
    SECURITY_SCORE = "security_score"
    COMPLIANCE_SCORE = "compliance_score"

    # Completeness metrics
    FEATURE_COVERAGE = "feature_coverage"
    EDGE_CASE_HANDLING = "edge_case_handling"
    TEST_COVERAGE = "test_coverage"

    # Efficiency metrics
    TIME_COMPLEXITY = "time_complexity"
    SPACE_COMPLEXITY = "space_complexity"
    OPTIMIZATION_SCORE = "optimization_score"

    # Maintainability metrics
    CODE_READABILITY = "code_readability"
    MODULARITY = "modularity"
    COUPLING = "coupling"

    # Scalability metrics
    HORIZONTAL_SCALABILITY = "horizontal_scalability"
    VERTICAL_SCALABILITY = "vertical_scalability"
    LOAD_HANDLING = "load_handling"


@dataclass
class MetricValue:
    """A single metric value with metadata"""
    metric_type: MetricType
    value: Union[float, int, str]
    category: MetricCategory
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)
    unit: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_type": self.metric_type.value,
            "value": self.value,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "unit": self.unit,
            "min_value": self.min_value,
            "max_value": self.max_value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricValue":
        """Create from dictionary"""
        return cls(
            metric_type=MetricType(data["metric_type"]),
            value=data["value"],
            category=MetricCategory(data["category"]),
            timestamp=data.get("timestamp", datetime.utcnow().timestamp()),
            metadata=data.get("metadata", {}),
            unit=data.get("unit"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value")
        )


@dataclass
class MetricSet:
    """A set of metrics for a specific artifact"""
    artifact_id: str
    artifact_type: str
    sub_problem_id: Optional[str] = None
    workflow_stage: Optional[str] = None
    metrics: Dict[MetricType, MetricValue] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: MetricValue):
        """Add a metric to this set"""
        self.metrics[metric.metric_type] = metric

    def get_metric(self, metric_type: MetricType) -> Optional[MetricValue]:
        """Get a specific metric"""
        return self.metrics.get(metric_type)

    def get_metrics_by_category(self, category: MetricCategory) -> List[MetricValue]:
        """Get all metrics in a category"""
        return [
            m for m in self.metrics.values()
            if m.category == category
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "sub_problem_id": self.sub_problem_id,
            "workflow_stage": self.workflow_stage,
            "metrics": {
                mt.value: m.to_dict()
                for mt, m in self.metrics.items()
            },
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricSet":
        """Create from dictionary"""
        metric_set = cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            sub_problem_id=data.get("sub_problem_id"),
            workflow_stage=data.get("workflow_stage"),
            timestamp=data.get("timestamp", datetime.utcnow().timestamp()),
            metadata=data.get("metadata", {})
        )

        for mt_str, m_data in data.get("metrics", {}).items():
            metric = MetricValue.from_dict(m_data)
            metric_set.metrics[MetricType(mt_str)] = metric

        return metric_set


class EvaluationMetricsCollector:
    """
    Collects and manages evaluation metrics for workflow artifacts.

    This integrates with RAGBits to store metrics in the vector store
    for semantic search and historical comparison.

    Usage:
        collector = EvaluationMetricsCollector(storage_manager)

        # Create metric set for an artifact
        metrics = MetricSet(artifact_id="art_123", artifact_type="solution")

        # Add metrics
        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=0.85,
            category=MetricCategory.QUALITY,
            metadata={"requirements_met": 17, "total_requirements": 20}
        ))

        # Store metrics
        await collector.collect_metrics(metrics)

        # Retrieve metrics
        retrieved = await collector.get_metrics("art_123")
    """

    def __init__(self, storage_manager=None):
        """
        Initialize metrics collector.

        Args:
            storage_manager: Optional storage manager for persistence
        """
        self.storage_manager = storage_manager

        # In-memory metric storage
        self.metrics_store: Dict[str, MetricSet] = {}

        # Metric history
        self.metric_history: List[Dict[str, Any]] = []

        logger.info("EvaluationMetricsCollector initialized")

    async def collect_metrics(
        self,
        metric_set: MetricSet,
        persist: bool = True
    ) -> bool:
        """
        Collect and store metrics for an artifact.

        Args:
            metric_set: Metric set to collect
            persist: Whether to persist to storage

        Returns:
            True if successful
        """
        try:
            # Store in memory
            self.metrics_store[metric_set.artifact_id] = metric_set

            # Add to history
            self.metric_history.append({
                "artifact_id": metric_set.artifact_id,
                "timestamp": metric_set.timestamp,
                "metric_count": len(metric_set.metrics)
            })

            # Persist to storage if enabled
            if persist and self.storage_manager:
                await self._persist_metrics(metric_set)

            logger.info(
                f"Collected {len(metric_set.metrics)} metrics "
                f"for artifact {metric_set.artifact_id}"
            )

            return True

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return False

    async def get_metrics(
        self,
        artifact_id: str
    ) -> Optional[MetricSet]:
        """
        Get metrics for an artifact.

        Args:
            artifact_id: Artifact ID

        Returns:
            Metric set if found, None otherwise
        """
        return self.metrics_store.get(artifact_id)

    async def get_metrics_by_subproblem(
        self,
        sub_problem_id: str
    ) -> List[MetricSet]:
        """
        Get all metrics for a sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            List of metric sets
        """
        return [
            ms for ms in self.metrics_store.values()
            if ms.sub_problem_id == sub_problem_id
        ]

    async def get_metrics_by_stage(
        self,
        workflow_stage: str
    ) -> List[MetricSet]:
        """
        Get all metrics from a workflow stage.

        Args:
            workflow_stage: Stage identifier

        Returns:
            List of metric sets
        """
        return [
            ms for ms in self.metrics_store.values()
            if ms.workflow_stage == workflow_stage
        ]

    async def get_historical_metrics(
        self,
        artifact_type: str,
        limit: int = 100
    ) -> List[MetricSet]:
        """
        Get historical metrics for an artifact type.

        Args:
            artifact_type: Type of artifact
            limit: Maximum number to return

        Returns:
            List of metric sets
        """
        metrics_list = [
            ms for ms in self.metrics_store.values()
            if ms.artifact_type == artifact_type
        ]

        # Sort by timestamp, most recent first
        metrics_list.sort(key=lambda x: x.timestamp, reverse=True)

        return metrics_list[:limit]

    async def aggregate_metrics_by_category(
        self,
        metric_sets: List[MetricSet],
        category: MetricCategory
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple sets by category.

        Args:
            metric_sets: List of metric sets
            category: Category to aggregate

        Returns:
            Aggregated statistics
        """
        all_metrics = []

        for ms in metric_sets:
            all_metrics.extend(ms.get_metrics_by_category(category))

        if not all_metrics:
            return {"category": category.value, "count": 0}

        # Calculate statistics
        values = [m.value for m in all_metrics if isinstance(m.value, (int, float))]

        if not values:
            return {"category": category.value, "count": len(all_metrics)}

        return {
            "category": category.value,
            "count": len(all_metrics),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "metric_types": list(set(m.metric_type.value for m in all_metrics))
        }

    async def compare_metrics(
        self,
        artifact_id_1: str,
        artifact_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare metrics between two artifacts.

        Args:
            artifact_id_1: First artifact ID
            artifact_id_2: Second artifact ID

        Returns:
            Comparison results
        """
        ms1 = self.metrics_store.get(artifact_id_1)
        ms2 = self.metrics_store.get(artifact_id_2)

        if not ms1 or not ms2:
            return {"error": "One or both artifacts not found"}

        comparison = {
            "artifact_1": artifact_id_1,
            "artifact_2": artifact_id_2,
            "differences": []
        }

        # Compare common metrics
        common_types = set(ms1.metrics.keys()) & set(ms2.metrics.keys())

        for metric_type in common_types:
            m1 = ms1.metrics[metric_type]
            m2 = ms2.metrics[metric_type]

            if isinstance(m1.value, (int, float)) and isinstance(m2.value, (int, float)):
                diff = {
                    "metric_type": metric_type.value,
                    "category": m1.category.value,
                    "artifact_1_value": m1.value,
                    "artifact_2_value": m2.value,
                    "difference": m2.value - m1.value,
                    "percent_change": ((m2.value - m1.value) / m1.value * 100)
                                   if m1.value != 0 else 0
                }
                comparison["differences"].append(diff)

        return comparison

    async def get_metric_trends(
        self,
        metric_type: MetricType,
        artifact_type: Optional[str] = None,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Get trends for a specific metric type over time.

        Args:
            metric_type: Type of metric to analyze
            artifact_type: Filter by artifact type
            window_size: Number of recent data points

        Returns:
            Trend analysis
        """
        # Get relevant metric sets
        if artifact_type:
            metric_sets = await self.get_historical_metrics(artifact_type, limit=window_size)
        else:
            metric_sets = list(self.metrics_store.values())[-window_size:]

        # Extract values
        values = []
        timestamps = []

        for ms in sorted(metric_sets, key=lambda x: x.timestamp):
            metric = ms.get_metric(metric_type)
            if metric and isinstance(metric.value, (int, float)):
                values.append(metric.value)
                timestamps.append(metric.timestamp)

        if not values:
            return {"metric_type": metric_type.value, "trend": "no_data"}

        # Calculate trend
        if len(values) >= 2:
            change = values[-1] - values[0]
            trend = "increasing" if change > 0 else "decreasing" if change < 0 else "stable"
        else:
            trend = "insufficient_data"

        return {
            "metric_type": metric_type.value,
            "trend": trend,
            "values": values,
            "timestamps": timestamps,
            "latest": values[-1] if values else None,
            "change": values[-1] - values[0] if len(values) >= 2 else 0
        }

    async def _persist_metrics(self, metric_set: MetricSet):
        """Persist metrics to storage"""
        if not self.storage_manager:
            return

        # Store as artifact metadata
        await self.storage_manager.store_artifact(
            artifact_type=f"metrics_{metric_set.artifact_type}",
            content=json.dumps(metric_set.to_dict()),
            metadata={
                "metrics_category": "evaluation_metrics",
                "metric_count": len(metric_set.metrics),
                "sub_problem_id": metric_set.sub_problem_id,
                "workflow_stage": metric_set.workflow_stage
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics"""
        return {
            "total_artifacts": len(self.metrics_store),
            "total_metrics_collected": sum(
                len(ms.metrics) for ms in self.metrics_store.values()
            ),
            "history_entries": len(self.metric_history),
            "artifacts_by_type": {
                art_type: sum(1 for ms in self.metrics_store.values()
                            if ms.artifact_type == art_type)
                for art_type in set(ms.artifact_type for ms in self.metrics_store.values())
            }
        }
