"""
RAGBits Evaluation Framework Integration

Phase 3: Enhanced evaluation framework with multi-dimensional metrics,
historical comparison, and gauntlet validation.
"""

from ragbits_integration.evaluation.metrics.evaluation_metrics import (
    EvaluationMetricsCollector,
    MetricType,
    MetricCategory
)
from ragbits_integration.evaluation.metrics.metrics_analyzer import (
    MetricsAnalyzer,
    AnalysisReport
)
from ragbits_integration.evaluation.gauntlets.enhanced_gauntlet import (
    EnhancedGauntletValidator,
    GauntletTestResult,
    MultiDimensionalScore
)
from ragbits_integration.evaluation.comparison.historical_comparator import (
    HistoricalComparator,
    ComparisonReport
)
from ragbits_integration.evaluation.dashboard.evaluation_dashboard import (
    EvaluationDashboard,
    DashboardReport
)

__all__ = [
    # Metrics
    "EvaluationMetricsCollector",
    "MetricType",
    "MetricCategory",
    "MetricsAnalyzer",
    "AnalysisReport",

    # Gauntlets
    "EnhancedGauntletValidator",
    "GauntletTestResult",
    "MultiDimensionalScore",

    # Comparison
    "HistoricalComparator",
    "ComparisonReport",

    # Dashboard
    "EvaluationDashboard",
    "DashboardReport"
]
