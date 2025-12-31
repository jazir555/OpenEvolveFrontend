"""
Evaluation Dashboard

Generates comprehensive dashboards for evaluation metrics,
gauntlet results, and historical comparisons.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

from ragbits_integration.evaluation.metrics.evaluation_metrics import (
    EvaluationMetricsCollector,
    MetricCategory
)
from ragbits_integration.evaluation.metrics.metrics_analyzer import (
    MetricsAnalyzer,
    AnalysisReport
)
from ragbits_integration.evaluation.gauntlets.enhanced_gauntlet import (
    EnhancedGauntletValidator,
    GauntletValidationResult
)
from ragbits_integration.evaluation.comparison.historical_comparator import (
    HistoricalComparator,
    ComparisonReport
)

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetricCard:
    """A metric card for the dashboard"""
    title: str
    value: str
    subtitle: Optional[str] = None
    trend: Optional[str] = None  # "up", "down", "stable"
    trend_value: Optional[float] = None
    status: str = "neutral"  # "success", "warning", "error", "neutral"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "value": self.value,
            "subtitle": self.subtitle,
            "trend": self.trend,
            "trend_value": self.trend_value,
            "status": self.status
        }


@dataclass
class DashboardChart:
    """A chart for the dashboard"""
    chart_type: str  # "line", "bar", "pie", "gauge"
    title: str
    data: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "data": self.data,
            "options": self.options or {}
        }


@dataclass
class DashboardTable:
    """A table for the dashboard"""
    title: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    sortable: bool = True
    filterable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "columns": self.columns,
            "rows": self.rows,
            "sortable": self.sortable,
            "filterable": self.filterable
        }


@dataclass
class DashboardReport:
    """Complete dashboard report"""
    report_id: str
    title: str
    timestamp: float
    metric_cards: List[DashboardMetricCard]
    charts: List[DashboardChart]
    tables: List[DashboardTable]
    summary: str
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "timestamp": self.timestamp,
            "metric_cards": [mc.to_dict() for mc in self.metric_cards],
            "charts": [c.to_dict() for c in self.charts],
            "tables": [t.to_dict() for t in self.tables],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }

    def to_html(self) -> str:
        """Generate HTML representation"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric-cards {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{
            flex: 1;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-card.success {{ border-left: 5px solid #28a745; }}
        .metric-card.warning {{ border-left: 5px solid #ffc107; }}
        .metric-card.error {{ border-left: 5px solid #dc3545; }}
        .metric-title {{ font-size: 14px; color: #666; }}
        .metric-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .metric-subtitle {{ font-size: 12px; color: #999; }}
        .section {{ margin: 30px 0; }}
        .section-title {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        .recommendations {{ background: #e7f3ff; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.title}</h1>
        <p>Generated: {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>{self.summary}</p>
    </div>

    <div class="metric-cards">
"""

        for card in self.metric_cards:
            html += f"""
        <div class="metric-card {card.status}">
            <div class="metric-title">{card.title}</div>
            <div class="metric-value">{card.value}</div>
            <div class="metric-subtitle">{card.subtitle or ''}</div>
        </div>
"""

        html += """
    </div>
"""

        # Add charts section (placeholder for actual chart rendering)
        if self.charts:
            html += """
    <div class="section">
        <div class="section-title">Charts</div>
        <p><i>Charts would be rendered here with a visualization library</i></p>
    </div>
"""

        # Add tables
        for table in self.tables:
            html += f"""
    <div class="section">
        <div class="section-title">{table.title}</div>
        <table>
            <thead>
                <tr>
"""
            for col in table.columns:
                html += f"                    <th>{col}</th>\n"

            html += "                </tr>\n            </thead>\n            <tbody>\n"

            for row in table.rows[:20]:  # Limit to 20 rows for HTML
                html += "                <tr>\n"
                for col in table.columns:
                    html += f"                    <td>{row.get(col, '')}</td>\n"
                html += "                </tr>\n"

            html += "            </tbody>\n        </table>\n    </div>\n"

        # Add recommendations
        if self.recommendations:
            html += """
    <div class="recommendations">
        <div class="section-title">Recommendations</div>
        <ul>
"""
            for rec in self.recommendations:
                html += f"            <li>{rec}</li>\n"

            html += "        </ul>\n    </div>\n"

        html += """
</body>
</html>
"""
        return html


class EvaluationDashboard:
    """
    Generates evaluation dashboards.

    Usage:
        dashboard = EvaluationDashboard(
            metrics_collector,
            metrics_analyzer,
            gauntlet_validator,
            historical_comparator
        )

        # Generate workflow dashboard
        report = await dashboard.generate_workflow_dashboard(
            workflow_id="workflow_123"
        )
    """

    def __init__(
        self,
        metrics_collector: EvaluationMetricsCollector,
        metrics_analyzer: MetricsAnalyzer,
        gauntlet_validator: EnhancedGauntletValidator,
        historical_comparator: HistoricalComparator
    ):
        """
        Initialize evaluation dashboard.

        Args:
            metrics_collector: Metrics collector
            metrics_analyzer: Metrics analyzer
            gauntlet_validator: Gauntlet validator
            historical_comparator: Historical comparator
        """
        self.metrics_collector = metrics_collector
        self.metrics_analyzer = metrics_analyzer
        self.gauntlet_validator = gauntlet_validator
        self.historical_comparator = historical_comparator

        logger.info("EvaluationDashboard initialized")

    async def generate_workflow_dashboard(
        self,
        workflow_id: str,
        artifact_ids: Optional[List[str]] = None
    ) -> DashboardReport:
        """
        Generate dashboard for a workflow.

        Args:
            workflow_id: Workflow ID
            artifact_ids: Optional list of artifact IDs to include

        Returns:
            Dashboard report
        """
        import uuid

        logger.info(f"Generating workflow dashboard for {workflow_id}")

        # Get all metrics if no specific artifacts provided
        if artifact_ids is None:
            all_metrics = self.metrics_collector.metrics_store.values()
            artifact_ids = [am.artifact_id for am in all_metrics]

        # Analyze artifacts
        artifact_reports = []

        for artifact_id in artifact_ids:
            report = await self.metrics_analyzer.analyze_artifact(artifact_id)
            if report:
                artifact_reports.append(report)

        if not artifact_reports:
            return self._create_empty_dashboard(workflow_id)

        # Generate metric cards
        metric_cards = self._generate_workflow_metric_cards(artifact_reports)

        # Generate charts
        charts = self._generate_workflow_charts(artifact_reports)

        # Generate tables
        tables = self._generate_workflow_tables(artifact_reports)

        # Generate summary and recommendations
        summary = self._generate_workflow_summary(artifact_reports)
        recommendations = self._generate_workflow_recommendations(artifact_reports)

        return DashboardReport(
            report_id=str(uuid.uuid4()),
            title=f"Workflow Dashboard: {workflow_id}",
            timestamp=datetime.utcnow().timestamp(),
            metric_cards=metric_cards,
            charts=charts,
            tables=tables,
            summary=summary,
            recommendations=recommendations,
            metadata={
                "workflow_id": workflow_id,
                "artifact_count": len(artifact_reports)
            }
        )

    async def generate_subproblem_dashboard(
        self,
        sub_problem_id: str
    ) -> DashboardReport:
        """
        Generate dashboard for a sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            Dashboard report
        """
        import uuid

        # Get artifacts for sub-problem
        metric_sets = await self.metrics_collector.get_metrics_by_subproblem(
            sub_problem_id
        )

        if not metric_sets:
            return self._create_empty_dashboard(f"Sub-problem: {sub_problem_id}")

        # Analyze artifacts
        artifact_reports = []

        for ms in metric_sets:
            report = await self.metrics_analyzer.analyze_artifact(ms.artifact_id)
            if report:
                artifact_reports.append(report)

        # Generate metric cards
        metric_cards = self._generate_subproblem_metric_cards(
            sub_problem_id,
            artifact_reports
        )

        # Generate charts
        charts = self._generate_subproblem_charts(artifact_reports)

        # Generate tables
        tables = self._generate_subproblem_tables(artifact_reports)

        # Generate summary
        summary = self._generate_subproblem_summary(sub_problem_id, artifact_reports)

        return DashboardReport(
            report_id=str(uuid.uuid4()),
            title=f"Sub-problem Dashboard: {sub_problem_id}",
            timestamp=datetime.utcnow().timestamp(),
            metric_cards=metric_cards,
            charts=charts,
            tables=tables,
            summary=summary,
            metadata={
                "sub_problem_id": sub_problem_id,
                "artifact_count": len(artifact_reports)
            }
        )

    async def generate_trend_dashboard(
        self,
        artifact_type: str,
        days: int = 30
    ) -> DashboardReport:
        """
        Generate trend dashboard for an artifact type.

        Args:
            artifact_type: Type of artifact
            days: Number of days to analyze

        Returns:
            Dashboard report
        """
        import uuid

        # Analyze trends
        trend_analysis = await self.historical_comparator.analyze_trends(
            artifact_type=artifact_type,
            window_size=50
        )

        # Generate metric cards
        metric_cards = self._generate_trend_metric_cards(artifact_type, trend_analysis)

        # Generate charts
        charts = self._generate_trend_charts(trend_analysis)

        return DashboardReport(
            report_id=str(uuid.uuid4()),
            title=f"Trend Dashboard: {artifact_type}",
            timestamp=datetime.utcnow().timestamp(),
            metric_cards=metric_cards,
            charts=charts,
            tables=[],
            summary=trend_analysis.get("trend", {}).get("direction", "No data"),
            metadata={
                "artifact_type": artifact_type,
                "days": days
            }
        )

    def _create_empty_dashboard(self, title: str) -> DashboardReport:
        """Create empty dashboard when no data available"""
        import uuid

        return DashboardReport(
            report_id=str(uuid.uuid4()),
            title=f"Dashboard: {title}",
            timestamp=datetime.utcnow().timestamp(),
            metric_cards=[
                DashboardMetricCard(
                    title="Status",
                    value="No Data",
                    subtitle="No evaluation data available"
                )
            ],
            charts=[],
            tables=[],
            summary="No evaluation data available for this dashboard.",
            metadata={}
        )

    def _generate_workflow_metric_cards(
        self,
        reports: List[AnalysisReport]
    ) -> List[DashboardMetricCard]:
        """Generate metric cards for workflow dashboard"""
        cards = []

        if not reports:
            return cards

        avg_score = sum(r.overall_score for r in reports) / len(reports)

        # Overall score card
        status = "success" if avg_score >= 0.7 else "warning" if avg_score >= 0.5 else "error"

        cards.append(DashboardMetricCard(
            title="Overall Score",
            value=f"{avg_score:.2f}",
            subtitle=f"Average across {len(reports)} artifacts",
            status=status
        ))

        # Best artifact
        best = max(reports, key=lambda r: r.overall_score)

        cards.append(DashboardMetricCard(
            title="Best Artifact",
            value=f"{best.overall_score:.2f}",
            subtitle=best.artifact_id[:20],
            status="success"
        ))

        # Pass rate
        pass_count = sum(1 for r in reports if r.overall_score >= 0.6)
        pass_rate = pass_count / len(reports) * 100

        cards.append(DashboardMetricCard(
            title="Pass Rate",
            value=f"{pass_rate:.0f}%",
            subtitle=f"{pass_count}/{len(reports)} artifacts passing",
            status="success" if pass_rate >= 80 else "warning" if pass_rate >= 50 else "error"
        ))

        # Total issues
        total_issues = sum(len(r.critical_issues) for r in reports)

        cards.append(DashboardMetricCard(
            title="Critical Issues",
            value=str(total_issues),
            subtitle="Across all artifacts",
            status="error" if total_issues > 0 else "success"
        ))

        return cards

    def _generate_workflow_charts(
        self,
        reports: List[AnalysisReport]
    ) -> List[DashboardChart]:
        """Generate charts for workflow dashboard"""
        charts = []

        if not reports:
            return charts

        # Score distribution chart
        scores = [r.overall_score for r in reports]

        charts.append(DashboardChart(
            chart_type="bar",
            title="Score Distribution",
            data={
                "labels": [r.artifact_id[:15] for r in reports],
                "datasets": [{
                    "label": "Overall Score",
                    "data": scores,
                    "backgroundColor": [
                        "#28a745" if s >= 0.7 else "#ffc107" if s >= 0.5 else "#dc3545"
                        for s in scores
                    ]
                }]
            }
        ))

        # Category comparison chart
        if reports:
            categories = {}
            for report in reports:
                for cs in report.category_scores:
                    if cs.category.value not in categories:
                        categories[cs.category.value] = []
                    categories[cs.category.value].append(cs.score)

            avg_categories = {
                cat: sum(scores) / len(scores)
                for cat, scores in categories.items()
            }

            charts.append(DashboardChart(
                chart_type="bar",
                title="Average Scores by Category",
                data={
                    "labels": list(avg_categories.keys()),
                    "datasets": [{
                        "label": "Average Score",
                        "data": list(avg_categories.values())
                    }]
                }
            ))

        return charts

    def _generate_workflow_tables(
        self,
        reports: List[AnalysisReport]
    ) -> List[DashboardTable]:
        """Generate tables for workflow dashboard"""
        tables = []

        # Artifacts table
        rows = []

        for report in sorted(reports, key=lambda r: r.overall_score, reverse=True):
            rows.append({
                "Artifact ID": report.artifact_id[:30],
                "Overall Score": f"{report.overall_score:.2f}",
                "Critical Issues": len(report.critical_issues),
                "Type": report.metadata.get("artifact_type", "unknown"),
                "Stage": report.metadata.get("workflow_stage", "unknown")
            })

        tables.append(DashboardTable(
            title="Artifacts Summary",
            columns=["Artifact ID", "Overall Score", "Critical Issues", "Type", "Stage"],
            rows=rows
        ))

        return tables

    def _generate_workflow_summary(
        self,
        reports: List[AnalysisReport]
    ) -> str:
        """Generate workflow summary"""
        if not reports:
            return "No artifacts to summarize"

        avg_score = sum(r.overall_score for r in reports) / len(reports)
        pass_count = sum(1 for r in reports if r.overall_score >= 0.6)

        return (
            f"Workflow contains {len(reports)} artifacts with an average score of {avg_score:.2f}. "
            f"{pass_count} artifacts ({pass_count/len(reports)*100:.0f}%) meet quality standards."
        )

    def _generate_workflow_recommendations(
        self,
        reports: List[AnalysisReport]
    ) -> List[str]:
        """Generate workflow recommendations"""
        recommendations = []

        # Find common issues
        issue_counts = {}

        for report in reports:
            for issue in report.critical_issues:
                # Extract category from issue
                for cat in ["quality", "performance", "security", "reliability"]:
                    if cat in issue.lower():
                        if cat not in issue_counts:
                            issue_counts[cat] = 0
                        issue_counts[cat] += 1

        # Generate recommendations for common issues
        for cat, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= len(reports) / 2:
                recommendations.append(
                    f"Multiple artifacts have {cat} issues - consider "
                    f"establishing {cat} guidelines and review processes"
                )

        # Add general recommendations
        avg_score = sum(r.overall_score for r in reports) / len(reports)

        if avg_score < 0.6:
            recommendations.append(
                "Overall workflow quality is below threshold - review "
                "all artifacts and establish quality gates"
            )

        if not recommendations:
            recommendations.append("All artifacts meet quality standards")

        return recommendations

    def _generate_subproblem_metric_cards(
        self,
        sub_problem_id: str,
        reports: List[AnalysisReport]
    ) -> List[DashboardMetricCard]:
        """Generate metric cards for sub-problem dashboard"""
        return self._generate_workflow_metric_cards(reports)

    def _generate_subproblem_charts(
        self,
        reports: List[AnalysisReport]
    ) -> List[DashboardChart]:
        """Generate charts for sub-problem dashboard"""
        return self._generate_workflow_charts(reports)

    def _generate_subproblem_tables(
        self,
        reports: List[AnalysisReport]
    ) -> List[DashboardTable]:
        """Generate tables for sub-problem dashboard"""
        return self._generate_workflow_tables(reports)

    def _generate_subproblem_summary(
        self,
        sub_problem_id: str,
        reports: List[AnalysisReport]
    ) -> str:
        """Generate sub-problem summary"""
        return self._generate_workflow_summary(reports)

    def _generate_trend_metric_cards(
        self,
        artifact_type: str,
        trend_analysis: Dict[str, Any]
    ) -> List[DashboardMetricCard]:
        """Generate metric cards for trend dashboard"""
        cards = []

        if "error" in trend_analysis:
            return [
                DashboardMetricCard(
                    title="Status",
                    value="No Data",
                    subtitle="No trend data available"
                )
            ]

        trend = trend_analysis.get("trend", {})
        direction = trend.get("direction", "stable")

        status_map = {
            "improving": "success",
            "stable": "neutral",
            "declining": "error"
        }

        cards.append(DashboardMetricCard(
            title="Trend Direction",
            value=direction.capitalize(),
            status=status_map.get(direction, "neutral")
        ))

        if "change" in trend:
            change = trend["change"]
            cards.append(DashboardMetricCard(
                title="Score Change",
                value=f"{change:+.2f}",
                trend="up" if change > 0 else "down" if change < 0 else "stable",
                trend_value=change,
                status="success" if change > 0 else "error" if change < 0 else "neutral"
            ))

        return cards

    def _generate_trend_charts(
        self,
        trend_analysis: Dict[str, Any]
    ) -> List[DashboardChart]:
        """Generate charts for trend dashboard"""
        charts = []

        if "scores" not in trend_analysis:
            return charts

        # Line chart of scores over time
        charts.append(DashboardChart(
            chart_type="line",
            title="Scores Over Time",
            data={
                "labels": list(range(len(trend_analysis["scores"]))),
                "datasets": [{
                    "label": "Score",
                    "data": trend_analysis["scores"],
                    "borderColor": "#007bff",
                    "fill": False
                }]
            },
            options={
                "scales": {
                    "yAxes": [{
                        "ticks": {"min": 0, "max": 10}
                    }]
                }
            }
        ))

        return charts
