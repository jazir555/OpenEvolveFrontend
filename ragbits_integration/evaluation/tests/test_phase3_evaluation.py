"""
Phase 3 Comprehensive Tests

Tests for evaluation framework components including:
- Metrics collector and analyzer
- Enhanced gauntlet validator
- Historical comparator
- Evaluation dashboard
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ragbits_integration.evaluation.metrics.evaluation_metrics import (
    EvaluationMetricsCollector,
    MetricSet,
    MetricValue,
    MetricCategory,
    MetricType
)
from ragbits_integration.evaluation.metrics.metrics_analyzer import (
    MetricsAnalyzer,
    AnalysisReport
)
from ragbits_integration.evaluation.gauntlets.enhanced_gauntlet import (
    EnhancedGauntletValidator,
    MultiDimensionalScore,
    GauntletTestResult
)
from ragbits_integration.evaluation.comparison.historical_comparator import (
    HistoricalComparator,
    ComparisonReport
)
from ragbits_integration.evaluation.dashboard.evaluation_dashboard import (
    EvaluationDashboard
)


# Metrics Collector Tests

@pytest.mark.asyncio
async def test_metrics_collector_initialization():
    """Test metrics collector initialization"""
    collector = EvaluationMetricsCollector()

    assert collector is not None
    assert len(collector.metrics_store) == 0
    assert len(collector.metric_history) == 0


@pytest.mark.asyncio
async def test_collect_and_retrieve_metrics():
    """Test collecting and retrieving metrics"""
    collector = EvaluationMetricsCollector()

    # Create metric set
    metrics = MetricSet(
        artifact_id="test_artifact",
        artifact_type="solution"
    )

    metrics.add_metric(MetricValue(
        metric_type=MetricType.REQUIREMENTS_COVERAGE,
        value=0.85,
        category=MetricCategory.QUALITY
    ))

    # Collect metrics
    result = await collector.collect_metrics(metrics)

    assert result is True
    assert "test_artifact" in collector.metrics_store

    # Retrieve metrics
    retrieved = await collector.get_metrics("test_artifact")

    assert retrieved is not None
    assert retrieved.artifact_id == "test_artifact"
    assert len(retrieved.metrics) == 1


@pytest.mark.asyncio
async def test_metrics_aggregation_by_category():
    """Test aggregating metrics by category"""
    collector = EvaluationMetricsCollector()

    # Create multiple metric sets
    for i in range(3):
        metrics = MetricSet(
            artifact_id=f"artifact_{i}",
            artifact_type="solution"
        )

        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=0.7 + i * 0.1,
            category=MetricCategory.QUALITY
        ))

        await collector.collect_metrics(metrics)

    # Get all metric sets
    all_sets = list(collector.metrics_store.values())

    # Aggregate by category
    agg = await collector.aggregate_metrics_by_category(
        all_sets,
        MetricCategory.QUALITY
    )

    assert agg["category"] == "quality"
    assert agg["count"] == 3
    assert "mean" in agg


@pytest.mark.asyncio
async def test_metrics_comparison():
    """Test comparing metrics between artifacts"""
    collector = EvaluationMetricsCollector()

    # Create two metric sets
    for i, value in enumerate([0.7, 0.9]):
        metrics = MetricSet(
            artifact_id=f"artifact_{i}",
            artifact_type="solution"
        )

        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=value,
            category=MetricCategory.QUALITY
        ))

        await collector.collect_metrics(metrics)

    # Compare
    comparison = await collector.compare_metrics("artifact_0", "artifact_1")

    assert "artifact_1" in comparison
    assert "artifact_2" in comparison
    assert "differences" in comparison
    assert len(comparison["differences"]) > 0


# Metrics Analyzer Tests

@pytest.mark.asyncio
async def test_analyzer_initialization():
    """Test metrics analyzer initialization"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)

    assert analyzer is not None
    assert analyzer.metrics_collector == collector


@pytest.mark.asyncio
async def test_analyze_artifact():
    """Test analyzing artifact metrics"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)

    # Create and collect metrics
    metrics = MetricSet(
        artifact_id="test_artifact",
        artifact_type="solution"
    )

    metrics.add_metric(MetricValue(
        metric_type=MetricType.REQUIREMENTS_COVERAGE,
        value=0.85,
        category=MetricCategory.QUALITY
    ))

    await collector.collect_metrics(metrics)

    # Analyze
    report = await analyzer.analyze_artifact("test_artifact")

    assert report is not None
    assert report.artifact_id == "test_artifact"
    assert report.overall_score >= 0.0
    assert len(report.category_scores) > 0


@pytest.mark.asyncio
async def test_analyzer_recommendations():
    """Test analyzer generates recommendations"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)

    # Create low-quality metrics
    metrics = MetricSet(
        artifact_id="poor_artifact",
        artifact_type="solution"
    )

    metrics.add_metric(MetricValue(
        metric_type=MetricType.REQUIREMENTS_COVERAGE,
        value=0.3,
        category=MetricCategory.QUALITY
    ))

    await collector.collect_metrics(metrics)

    # Analyze
    report = await analyzer.analyze_artifact("poor_artifact")

    assert report is not None
    # Low score should generate recommendations
    if report.overall_score < 0.6:
        assert len(report.recommendations) > 0 or len(report.critical_issues) > 0


@pytest.mark.asyncio
async def test_compare_artifacts():
    """Test comparing two artifacts"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)

    # Create two artifacts with different scores
    for i, value in enumerate([0.7, 0.9]):
        metrics = MetricSet(
            artifact_id=f"artifact_{i}",
            artifact_type="solution"
        )

        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=value,
            category=MetricCategory.QUALITY
        ))

        await collector.collect_metrics(metrics)

    # Compare
    comparison = await analyzer.compare_artifacts("artifact_0", "artifact_1")

    assert comparison is not None
    assert "overall_score_diff" in comparison
    assert "category_comparison" in comparison


# Gauntlet Validator Tests

@pytest.mark.asyncio
async def test_gauntlet_validator_initialization():
    """Test gauntlet validator initialization"""
    collector = EvaluationMetricsCollector()
    validator = EnhancedGauntletValidator(collector)

    assert validator is not None
    assert validator.metrics_collector == collector


@pytest.mark.asyncio
async def test_validate_solution():
    """Test solution validation"""
    collector = EvaluationMetricsCollector()
    validator = EnhancedGauntletValidator(collector)

    solution_text = """
    Implement user authentication with JWT tokens and bcrypt password hashing.
    Include input validation, error handling, and rate limiting for security.
    """

    result = await validator.validate_solution(
        artifact_id="test_solution",
        solution_text=solution_text,
        test_types=[
            validator.test_registry["functional_requirements_coverage"].__self__._get_test_type_for_name("functional"),
            validator.test_registry["security_input_validation"].__self__._get_test_type_for_name("security")
        ],
        requirements=["JWT authentication", "bcrypt hashing", "rate limiting"]
    )

    assert result is not None
    assert result.artifact_id == "test_solution"
    assert result.multi_dimensional_score is not None
    assert len(result.test_results) > 0


@pytest.mark.asyncio
async def test_multi_dimensional_score():
    """Test multi-dimensional scoring"""
    score = MultiDimensionalScore(
        functionality=8.0,
        performance=7.0,
        security=9.0,
        reliability=8.0,
        tests_passed=10,
        tests_failed=2,
        tests_total=12
    )

    assert score.overall_score > 0
    assert score.functionality == 8.0
    assert score.security == 9.0
    assert score.tests_passed == 10
    assert score.get_verdict() in ["EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"]


# Historical Comparator Tests

@pytest.mark.asyncio
async def test_historical_comparator_initialization():
    """Test historical comparator initialization"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    comparator = HistoricalComparator(collector, analyzer)

    assert comparator is not None
    assert comparator.metrics_collector == collector
    assert comparator.metrics_analyzer == analyzer


@pytest.mark.asyncio
async def test_compare_with_historical():
    """Test comparing with historical data"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    comparator = HistoricalComparator(collector, analyzer)

    # Create historical artifacts
    for i in range(5):
        metrics = MetricSet(
            artifact_id=f"hist_artifact_{i}",
            artifact_type="solution"
        )

        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=0.6 + i * 0.05,
            category=MetricCategory.QUALITY
        ))

        await collector.collect_metrics(metrics)

    # Create current artifact
    current_metrics = MetricSet(
        artifact_id="current_artifact",
        artifact_type="solution"
    )

    current_metrics.add_metric(MetricValue(
        metric_type=MetricType.REQUIREMENTS_COVERAGE,
        value=0.8,
        category=MetricCategory.QUALITY
    ))

    await collector.collect_metrics(current_metrics)

    # Compare
    report = await comparator.compare_with_historical(
        artifact_id="current_artifact",
        artifact_type="solution",
        limit=10
    )

    assert report is not None
    assert report.artifact_id == "current_artifact"
    assert len(report.historical_scores) > 0
    assert report.current_score >= 0


@pytest.mark.asyncio
async def test_analyze_trends():
    """Test trend analysis"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    comparator = HistoricalComparator(collector, analyzer)

    # Create time series of artifacts
    for i in range(10):
        metrics = MetricSet(
            artifact_id=f"trend_artifact_{i}",
            artifact_type="solution"
        )

        # Gradually improving scores
        import time
        metrics.timestamp = time.time() - (10 - i) * 24 * 3600  # Spaced over days

        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=0.5 + i * 0.03,
            category=MetricCategory.QUALITY
        ))

        await collector.collect_metrics(metrics)

    # Analyze trends
    trend_analysis = await comparator.analyze_trends(
        artifact_type="solution",
        window_size=20
    )

    assert trend_analysis is not None
    assert "trend" in trend_analysis
    assert "data_points" in trend_analysis


# Dashboard Tests

@pytest.mark.asyncio
async def test_dashboard_initialization():
    """Test dashboard initialization"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    validator = EnhancedGauntletValidator(collector)
    comparator = HistoricalComparator(collector, analyzer)
    dashboard = EvaluationDashboard(collector, analyzer, validator, comparator)

    assert dashboard is not None


@pytest.mark.asyncio
async def test_generate_workflow_dashboard():
    """Test workflow dashboard generation"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    validator = EnhancedGauntletValidator(collector)
    comparator = HistoricalComparator(collector, analyzer)
    dashboard = EvaluationDashboard(collector, analyzer, validator, comparator)

    # Create test artifacts
    artifact_ids = []

    for i in range(3):
        metrics = MetricSet(
            artifact_id=f"dashboard_artifact_{i}",
            artifact_type="solution"
        )

        metrics.add_metric(MetricValue(
            metric_type=MetricType.REQUIREMENTS_COVERAGE,
            value=0.6 + i * 0.1,
            category=MetricCategory.QUALITY
        ))

        await collector.collect_metrics(metrics)
        artifact_ids.append(metrics.artifact_id)

    # Generate dashboard
    report = await dashboard.generate_workflow_dashboard(
        workflow_id="test_workflow",
        artifact_ids=artifact_ids
    )

    assert report is not None
    assert report.title == "Workflow Dashboard: test_workflow"
    assert len(report.metric_cards) > 0
    assert len(report.charts) > 0 or len(report.tables) > 0
    assert report.summary is not None


@pytest.mark.asyncio
async def test_dashboard_to_html():
    """Test dashboard HTML generation"""
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    validator = EnhancedGauntletValidator(collector)
    comparator = HistoricalComparator(collector, analyzer)
    dashboard = EvaluationDashboard(collector, analyzer, validator, comparator)

    # Create test artifact
    metrics = MetricSet(
        artifact_id="html_test_artifact",
        artifact_type="solution"
    )

    metrics.add_metric(MetricValue(
        metric_type=MetricType.REQUIREMENTS_COVERAGE,
        value=0.8,
        category=MetricCategory.QUALITY
    ))

    await collector.collect_metrics(metrics)

    # Generate dashboard
    report = await dashboard.generate_workflow_dashboard(
        workflow_id="html_test_workflow",
        artifact_ids=["html_test_artifact"]
    )

    # Generate HTML
    html = report.to_html()

    assert html is not None
    assert "<html>" in html
    assert report.title in html
    assert report.summary in html


# Integration Tests

@pytest.mark.asyncio
async def test_full_evaluation_pipeline():
    """Test complete evaluation pipeline"""
    # Initialize components
    collector = EvaluationMetricsCollector()
    analyzer = MetricsAnalyzer(collector)
    validator = EnhancedGauntletValidator(collector)
    comparator = HistoricalComparator(collector, analyzer)
    dashboard = EvaluationDashboard(collector, analyzer, validator, comparator)

    # Step 1: Validate solution
    solution_text = """
    Implement REST API with authentication, rate limiting, and error handling.
    Use JWT for tokens and bcrypt for password hashing.
    Include comprehensive input validation.
    """

    validation_result = await validator.validate_solution(
        artifact_id="pipeline_solution",
        solution_text=solution_text,
        requirements=["REST API", "JWT authentication", "rate limiting"]
    )

    assert validation_result is not None

    # Step 2: Analyze metrics (automatically stored by validator)
    analysis_report = await analyzer.analyze_artifact("pipeline_solution")

    assert analysis_report is not None

    # Step 3: Generate dashboard
    dashboard_report = await dashboard.generate_workflow_dashboard(
        workflow_id="pipeline_workflow",
        artifact_ids=["pipeline_solution"]
    )

    assert dashboard_report is not None
    assert len(dashboard_report.metric_cards) > 0

    # Step 4: Generate HTML report
    html = dashboard_report.to_html()

    assert html is not None
    assert "<html>" in html


if __name__ == "__main__":
    # Run tests manually
    import sys

    async def run_tests():
        print("Running Phase 3 Evaluation Tests...\n")

        tests = [
            ("Metrics Collector Initialization", test_metrics_collector_initialization),
            ("Collect and Retrieve Metrics", test_collect_and_retrieve_metrics),
            ("Metrics Aggregation", test_metrics_aggregation_by_category),
            ("Metrics Comparison", test_metrics_comparison),
            ("Analyzer Initialization", test_analyzer_initialization),
            ("Analyze Artifact", test_analyze_artifact),
            ("Gauntlet Validator", test_validate_solution),
            ("Multi-Dimensional Score", test_multi_dimensional_score),
            ("Historical Comparison", test_compare_with_historical),
            ("Trend Analysis", test_analyze_trends),
            ("Workflow Dashboard", test_generate_workflow_dashboard),
            ("Dashboard HTML", test_dashboard_to_html),
            ("Full Pipeline", test_full_evaluation_pipeline),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                await test_func()
                passed += 1
                print(f"[OK] PASSED: {name}")
            except Exception as e:
                failed += 1
                print(f"[FAIL] FAILED: {name}")
                print(f"   Error: {e}")

        print(f"\n{'='*70}")
        print(f"Passed: {passed}/{passed + failed}")
        print('='*70)

        if failed > 0:
            sys.exit(1)

    asyncio.run(run_tests())
