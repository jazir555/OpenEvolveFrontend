"""
Evaluation Reporter Module for OpenEvolve

This module provides comprehensive reporting capabilities for the Evaluator Team,
including multi-format export (JSON, CSV, HTML, PDF), trend analysis, comparison
reports, and visual report generation.

Author: OpenEvolve
Date: 2025-01-04
"""
from __future__ import annotations


from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import csv
from io import StringIO
from pathlib import Path
import statistics
import numpy as np

from evaluator_analytics import (
    EvaluatorAnalytics,
    EvaluatorMetrics,
    BiasProfile,
    EvaluationRecord,
    EvaluationStage
)


class ReportFormat(Enum):
    """Report output formats"""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"


class ReportType(Enum):
    """Types of reports"""
    INDIVIDUAL_PERFORMANCE = "individual_performance"
    TEAM_OVERVIEW = "team_overview"
    BIAS_ANALYSIS = "bias_analysis"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    QUALITY_GATE = "quality_gate"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    report_type: ReportType
    format: ReportFormat
    evaluator_ids: Optional[List[str]] = None
    stage: Optional[EvaluationStage] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_charts: bool = True
    include_recommendations: bool = True
    include_biases: bool = True
    include_trends: bool = True
    comparison_baseline: Optional[str] = None
    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                "accuracy", "consistency_score", "reliability_score",
                "average_score", "average_confidence", "evaluation_frequency"
            ]


@dataclass
class ReportSection:
    """A section of a report"""
    title: str
    content: Any
    metadata: Dict[str, Any] = None


class EvaluationReporter:
    """
    Comprehensive reporter for evaluator analytics and metrics.
    Generates reports in multiple formats with visualizations.
    """

    def __init__(self, analytics: EvaluatorAnalytics):
        """
        Initialize evaluation reporter

        Args:
            analytics: EvaluatorAnalytics instance
        """
        self.analytics = analytics
        self.report_cache: Dict[str, Dict[str, Any]] = {}

    def generate_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """
        Generate a report based on configuration

        Args:
            config: Report configuration

        Returns:
            Report data dictionary
        """
        report_id = self._generate_report_id(config)

        # Check cache
        if report_id in self.report_cache:
            cached_report = self.report_cache[report_id]
            # Check if cache is still valid (1 hour)
            if (datetime.now() - datetime.fromisoformat(cached_report["generated_at"])).seconds < 3600:
                return cached_report

        # Generate report based on type
        if config.report_type == ReportType.INDIVIDUAL_PERFORMANCE:
            report = self._generate_individual_report(config)
        elif config.report_type == ReportType.TEAM_OVERVIEW:
            report = self._generate_team_report(config)
        elif config.report_type == ReportType.BIAS_ANALYSIS:
            report = self._generate_bias_report(config)
        elif config.report_type == ReportType.TREND_ANALYSIS:
            report = self._generate_trend_report(config)
        elif config.report_type == ReportType.COMPARISON:
            report = self._generate_comparison_report(config)
        elif config.report_type == ReportType.QUALITY_GATE:
            report = self._generate_quality_gate_report(config)
        else:
            report = self._generate_custom_report(config)

        report["report_id"] = report_id
        report["generated_at"] = datetime.now().isoformat()
        report["config"] = self._config_to_dict(config)

        # Cache report
        self.report_cache[report_id] = report

        return report

    def export_report(
        self,
        report: Dict[str, Any],
        format: ReportFormat,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export report in specified format

        Args:
            report: Report data dictionary
            format: Export format
            output_path: Optional output file path

        Returns:
            Exported report as string or file path
        """
        if format == ReportFormat.JSON:
            return self._export_json(report, output_path)
        elif format == ReportFormat.CSV:
            return self._export_csv(report, output_path)
        elif format == ReportFormat.HTML:
            return self._export_html(report, output_path)
        elif format == ReportFormat.MARKDOWN:
            return self._export_markdown(report, output_path)
        elif format == ReportFormat.PDF:
            return self._export_pdf(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_individual_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate individual performance report"""
        if not config.evaluator_ids or len(config.evaluator_ids) != 1:
            raise ValueError("Individual report requires exactly one evaluator_id")

        evaluator_id = config.evaluator_ids[0]
        quality_report = self.analytics.generate_quality_report(evaluator_id)
        metrics = self.analytics.get_evaluator_metrics(evaluator_id)

        sections = []

        # Overview section
        sections.append(ReportSection(
            title="Overview",
            content={
                "evaluator_id": evaluator_id,
                "total_evaluations": metrics.total_evaluations if metrics else 0,
                "overall_quality_score": quality_report.get("overall_quality_score", 0),
                "last_evaluation": metrics.last_evaluation.isoformat() if metrics and metrics.last_evaluation else None
            }
        ))

        # Performance metrics section
        if metrics:
            sections.append(ReportSection(
                title="Performance Metrics",
                content={
                    "average_score": metrics.average_score,
                    "average_confidence": metrics.average_confidence,
                    "average_time": metrics.average_time,
                    "accuracy": metrics.accuracy,
                    "consistency_score": metrics.consistency_score,
                    "reliability_score": metrics.reliability_score,
                    "evaluation_frequency": metrics.evaluation_frequency
                }
            ))

        # Stage performance section
        if metrics and metrics.stage_performance:
            sections.append(ReportSection(
                title="Stage Performance",
                content=metrics.stage_performance
            ))

        # Bias analysis section
        if config.include_biases:
            biases = self.analytics.detect_biases(evaluator_id)
            if biases:
                sections.append(ReportSection(
                    title="Bias Analysis",
                    content={
                        "biases_detected": len(biases),
                        "bias_details": [b.to_dict() for b in biases]
                    }
                ))

        # Trends section
        if config.include_trends:
            trends = self.analytics.analyze_performance_trends(evaluator_id)
            sections.append(ReportSection(
                title="Performance Trends",
                content=trends
            ))

        # Recommendations section
        if config.include_recommendations:
            recommendations = self._generate_recommendations(evaluator_id)
            sections.append(ReportSection(
                title="Recommendations",
                content={"recommendations": recommendations}
            ))

        return {
            "report_type": ReportType.INDIVIDUAL_PERFORMANCE.value,
            "title": f"Individual Performance Report - {evaluator_id}",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _generate_team_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate team overview report"""
        team_metrics = self.analytics.get_team_metrics()
        top_performers = {}

        for metric in config.metrics:
            top_performers[metric] = self.analytics.get_top_performers(metric, 5)

        sections = []

        # Team overview
        sections.append(ReportSection(
            title="Team Overview",
            content=team_metrics
        ))

        # Top performers
        sections.append(ReportSection(
            title="Top Performers",
            content=top_performers
        ))

        # Stage breakdown
        stage_performance = {}
        for stage in EvaluationStage:
            stage_perf = self.analytics.get_stage_performance(stage)
            if "error" not in stage_perf:
                stage_performance[stage.value] = stage_perf

        if stage_performance:
            sections.append(ReportSection(
                title="Stage Performance",
                content=stage_performance
            ))

        # Individual summaries
        if config.evaluator_ids:
            individual_summaries = []
            for eval_id in config.evaluator_ids:
                metrics = self.analytics.get_evaluator_metrics(eval_id)
                if metrics:
                    summary = {
                        "evaluator_id": eval_id,
                        "average_score": metrics.average_score,
                        "accuracy": metrics.accuracy,
                        "total_evaluations": metrics.total_evaluations
                    }

                    # Add ensemble metrics if available
                    if hasattr(metrics, 'ensemble_selection_count'):
                        summary["ensemble_selection_count"] = metrics.ensemble_selection_count
                        summary["ensemble_weight"] = metrics.ensemble_weight
                        summary["ensemble_utilization"] = metrics.ensemble_utilization

                    individual_summaries.append(summary)

            sections.append(ReportSection(
                title="Individual Summaries",
                content={"evaluators": individual_summaries}
            ))

        # Ensemble performance section (if ensemble data available)
        ensemble_metrics = self._get_ensemble_metrics(config.evaluator_ids)
        if ensemble_metrics:
            sections.append(ReportSection(
                title="Ensemble Performance",
                content=ensemble_metrics
            ))

        return {
            "report_type": ReportType.TEAM_OVERVIEW.value,
            "title": "Team Overview Report",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _generate_bias_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate bias analysis report"""
        evaluator_ids = config.evaluator_ids or list(self.analytics.evaluator_metrics.keys())

        all_biases = []
        bias_summary = {
            "total_evaluators_analyzed": 0,
            "evaluators_with_biases": 0,
            "bias_types": {}
        }

        for eval_id in evaluator_ids:
            biases = self.analytics.detect_biases(eval_id)
            if biases:
                bias_summary["evaluators_with_biases"] += 1
                for bias in biases:
                    bias_type = bias.bias_type.value
                    if bias_type not in bias_summary["bias_types"]:
                        bias_summary["bias_types"][bias_type] = {
                            "count": 0,
                            "average_severity": []
                        }
                    bias_summary["bias_types"][bias_type]["count"] += 1
                    bias_summary["bias_types"][bias_type]["average_severity"].append(bias.severity)

                all_biases.extend([b.to_dict() for b in biases])

            bias_summary["total_evaluators_analyzed"] += 1

        # Calculate average severities
        for bias_type, data in bias_summary["bias_types"].items():
            if data["average_severity"]:
                data["average_severity"] = float(np.mean(data["average_severity"]))
            else:
                data["average_severity"] = 0.0

        sections = []

        sections.append(ReportSection(
            title="Bias Summary",
            content=bias_summary
        ))

        if all_biases:
            sections.append(ReportSection(
                title="Detailed Biases",
                content={"biases": all_biases}
            ))

        # Generate recommendations
        recommendations = self._generate_bias_recommendations(all_biases)
        sections.append(ReportSection(
            title="Bias Mitigation Recommendations",
            content={"recommendations": recommendations}
        ))

        return {
            "report_type": ReportType.BIAS_ANALYSIS.value,
            "title": "Bias Analysis Report",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _generate_trend_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate trend analysis report"""
        evaluator_ids = config.evaluator_ids or list(self.analytics.evaluator_metrics.keys())

        trends = {}
        for eval_id in evaluator_ids:
            trend_data = self.analytics.analyze_performance_trends(eval_id)
            trends[eval_id] = trend_data

        sections = []

        sections.append(ReportSection(
            title="Score Trends",
            content={
                eval_id: {
                    "trend": data.get("score_trend"),
                    "slope": data.get("score_slope")
                }
                for eval_id, data in trends.items()
                if "error" not in data
            }
        ))

        sections.append(ReportSection(
            title="Time Efficiency Trends",
            content={
                eval_id: {
                    "trend": data.get("time_trend"),
                    "slope": data.get("time_slope")
                }
                for eval_id, data in trends.items()
                if "error" not in data
            }
        ))

        # Calculate team-wide trends
        all_metrics = list(self.analytics.evaluator_metrics.values())
        if all_metrics:
            team_trends = self._calculate_team_trends(all_metrics)
            sections.append(ReportSection(
                title="Team-Wide Trends",
                content=team_trends
            ))

        return {
            "report_type": ReportType.TREND_ANALYSIS.value,
            "title": "Trend Analysis Report",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _generate_comparison_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate comparison report"""
        if not config.evaluator_ids or len(config.evaluator_ids) < 2:
            raise ValueError("Comparison report requires at least two evaluator_ids")

        comparison = self.analytics.compare_evaluators(config.evaluator_ids)

        sections = []

        # Direct comparison
        sections.append(ReportSection(
            title="Metrics Comparison",
            content=comparison["metrics_comparison"]
        ))

        # Rankings
        sections.append(ReportSection(
            title="Rankings by Metric",
            content=comparison["ranking"]
        ))

        # Statistical analysis
        stats_analysis = self._generate_comparison_stats(config.evaluator_ids)
        sections.append(ReportSection(
            title="Statistical Analysis",
            content=stats_analysis
        ))

        return {
            "report_type": ReportType.COMPARISON.value,
            "title": "Evaluator Comparison Report",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _generate_quality_gate_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate quality gate report"""
        evaluator_ids = config.evaluator_ids or list(self.analytics.evaluator_metrics.keys())

        quality_data = {}
        for eval_id in evaluator_ids:
            quality_report = self.analytics.generate_quality_report(eval_id)
            quality_data[eval_id] = quality_report

        # Define quality thresholds
        thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "acceptable": 0.6,
            "needs_improvement": 0.0
        }

        # Categorize evaluators
        categorized = {
            "excellent": [],
            "good": [],
            "acceptable": [],
            "needs_improvement": []
        }

        for eval_id, data in quality_data.items():
            quality_score = data.get("overall_quality_score", 0)

            if quality_score >= thresholds["excellent"]:
                categorized["excellent"].append(eval_id)
            elif quality_score >= thresholds["good"]:
                categorized["good"].append(eval_id)
            elif quality_score >= thresholds["acceptable"]:
                categorized["acceptable"].append(eval_id)
            else:
                categorized["needs_improvement"].append(eval_id)

        sections = []

        sections.append(ReportSection(
            title="Quality Gate Results",
            content={
                "thresholds": thresholds,
                "categorized_evaluators": categorized,
                "total_evaluators": len(evaluator_ids)
            }
        ))

        sections.append(ReportSection(
            title="Quality Scores",
            content={
                eval_id: data.get("overall_quality_score", 0)
                for eval_id, data in quality_data.items()
            }
        ))

        # Action items
        action_items = self._generate_quality_action_items(categorized, quality_data)
        sections.append(ReportSection(
            title="Action Items",
            content={"action_items": action_items}
        ))

        return {
            "report_type": ReportType.QUALITY_GATE.value,
            "title": "Quality Gate Report",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _generate_custom_report(
        self,
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate custom report based on configuration"""
        sections = []

        if config.evaluator_ids:
            for eval_id in config.evaluator_ids:
                metrics = self.analytics.get_evaluator_metrics(eval_id)
                if metrics:
                    filtered_metrics = {
                        k: v for k, v in metrics.to_dict().items()
                        if not config.metrics or k in config.metrics
                    }
                    sections.append(ReportSection(
                        title=f"Evaluator: {eval_id}",
                        content=filtered_metrics
                    ))

        return {
            "report_type": ReportType.CUSTOM.value,
            "title": "Custom Report",
            "sections": [self._section_to_dict(s) for s in sections]
        }

    def _export_json(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Export report as JSON"""
        json_str = json.dumps(report, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            return output_path

        return json_str

    def _export_csv(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Export report as CSV"""
        output = StringIO()
        writer = csv.writer(output)

        # Write basic info
        writer.writerow(["Report Type", report.get("report_type", "")])
        writer.writerow(["Title", report.get("title", "")])
        writer.writerow(["Generated At", report.get("generated_at", "")])
        writer.writerow([])

        # Write sections
        for section in report.get("sections", []):
            writer.writerow([f"Section: {section.get('title', '')}"])
            writer.writerow([])

            content = section.get("content", {})
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    writer.writerow([key, value])
            writer.writerow([])

        csv_str = output.getvalue()

        if output_path:
            with open(output_path, 'w') as f:
                f.write(csv_str)
            return output_path

        return csv_str

    def _export_html(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Export report as HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report.get('title', 'Report')}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section-title {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            color: #2c3e50;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .good {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report.get('title', 'Report')}</h1>
        <p>Generated: {report.get('generated_at', '')}</p>
        <p>Type: {report.get('report_type', '')}</p>
    </div>
"""

        # Render sections
        for section in report.get("sections", []):
            html += f"""
    <div class="section">
        <h2 class="section-title">{section.get('title', '')}</h2>
        {self._render_html_content(section.get('content', {}))}
    </div>
"""

        html += """
</body>
</html>
"""

        if output_path:
            with open(output_path, 'w') as f:
                f.write(html)
            return output_path

        return html

    def _render_html_content(self, content: Any) -> str:
        """Render content as HTML"""
        if isinstance(content, dict):
            html = ""
            for key, value in content.items():
                if isinstance(value, (int, float)):
                    color_class = "good" if value >= 0.75 else "warning" if value >= 0.5 else "error"
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    html += f"""
                    <div class="metric">
                        <div class="metric-label">{key}</div>
                        <div class="metric-value {color_class}">{formatted_value}</div>
                    </div>
                    """
                elif isinstance(value, list):
                    html += f"<h3>{key}</h3><ul>"
                    for item in value[:10]:  # Limit to 10 items
                        if isinstance(item, dict):
                            html += f"<li>{json.dumps(item, indent=2)}</li>"
                        else:
                            html += f"<li>{item}</li>"
                    html += "</ul>"
                elif isinstance(value, dict):
                    html += f"<h3>{key}</h3>"
                    html += self._render_html_content(value)
            return html
        elif isinstance(content, list):
            html = "<ul>"
            for item in content:
                html += f"<li>{item}</li>"
            html += "</ul>"
            return html
        else:
            return str(content)

    def _export_markdown(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Export report as Markdown"""
        md = f"# {report.get('title', 'Report')}\n\n"
        md += f"**Generated:** {report.get('generated_at', '')}\n\n"
        md += f"**Type:** {report.get('report_type', '')}\n\n"
        md += "---\n\n"

        for section in report.get("sections", []):
            md += f"## {section.get('title', '')}\n\n"
            md += self._render_markdown_content(section.get('content', {}))
            md += "\n\n"

        if output_path:
            with open(output_path, 'w') as f:
                f.write(md)
            return output_path

        return md

    def _render_markdown_content(self, content: Any, indent: int = 0) -> str:
        """Render content as Markdown"""
        prefix = "  " * indent

        if isinstance(content, dict):
            md = ""
            for key, value in content.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    md += f"{prefix}- **{key}:** {formatted_value}\n"
                elif isinstance(value, list):
                    md += f"{prefix}- **{key}:**\n"
                    for item in value[:5]:
                        if isinstance(item, dict):
                            md += f"{prefix}  - {json.dumps(item)}\n"
                        else:
                            md += f"{prefix}  - {item}\n"
                elif isinstance(value, dict):
                    md += f"{prefix}- **{key}:**\n"
                    md += self._render_markdown_content(value, indent + 1)
            return md
        elif isinstance(content, list):
            md = ""
            for item in content:
                md += f"{prefix}- {item}\n"
            return md
        else:
            return f"{prefix}{content}\n"

    def _export_pdf(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Export report as PDF (requires additional dependencies)"""
        # For now, export as HTML and note PDF conversion
        html_path = self._export_html(report, output_path)

        if output_path:
            # In production, you would use a library like weasyprint or pdfkit
            # For now, return HTML with instructions
            return f"PDF export requires additional dependencies. HTML saved to: {html_path}"

        return html_path

    def _generate_recommendations(self, evaluator_id: str) -> List[str]:
        """Generate recommendations for an evaluator"""
        recommendations = []
        metrics = self.analytics.get_evaluator_metrics(evaluator_id)

        if not metrics:
            return ["No data available for recommendations"]

        # Accuracy recommendations
        if metrics.accuracy < 0.7:
            recommendations.append(
                "Consider additional training on evaluation criteria to improve accuracy"
            )

        # Consistency recommendations
        if metrics.consistency_score < 0.7:
            recommendations.append(
                "Review evaluation guidelines to improve consistency across evaluations"
            )

        # Time recommendations
        if metrics.average_time > 300:  # 5 minutes
            recommendations.append(
                "Evaluation time is high; consider streamlining the evaluation process"
            )

        # Confidence recommendations
        if metrics.average_confidence < 0.6:
            recommendations.append(
                "Low confidence scores suggest uncertainty; consider additional domain training"
            )

        # Bias recommendations
        biases = self.analytics.detect_biases(evaluator_id)
        if biases:
            recommendations.append(
                f"Address {len(biases)} detected bias(es) through calibration exercises"
            )

        if not recommendations:
            recommendations.append("Performance is satisfactory; continue current practices")

        return recommendations

    def _generate_bias_recommendations(self, all_biases: List[Dict]) -> List[str]:
        """Generate team-wide bias mitigation recommendations"""
        recommendations = []

        bias_types = {}
        for bias in all_biases:
            bias_type = bias.get("bias_type", "unknown")
            if bias_type not in bias_types:
                bias_types[bias_type] = []
            bias_types[bias_type].append(bias.get("severity", 0))

        # Generate recommendations for each bias type
        if "leniency" in bias_types:
            recommendations.append(
                "Conduct team-wide calibration sessions to address leniency bias"
            )

        if "severity" in bias_types:
            recommendations.append(
                "Review scoring criteria to address severity bias"
            )

        if "central_tendency" in bias_types:
            recommendations.append(
                "Encourage use of full scoring range to reduce central tendency bias"
            )

        if "temporal" in bias_types:
            recommendations.append(
                "Implement evaluation breaks to reduce temporal bias"
            )

        if not recommendations:
            recommendations.append("No significant biases detected; maintain current practices")

        return recommendations

    def _generate_comparison_stats(
        self,
        evaluator_ids: List[str]
    ) -> Dict[str, Any]:
        """Generate statistical comparison data"""
        metrics_data = []
        for eval_id in evaluator_ids:
            metrics = self.analytics.get_evaluator_metrics(eval_id)
            if metrics:
                metrics_data.append(metrics)

        if not metrics_data:
            return {}

        stats_results = {}

        for metric in ["average_score", "accuracy", "consistency_score", "reliability_score"]:
            values = [getattr(m, metric, 0) for m in metrics_data]
            if values:
                stats_results[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values))
                }

        return stats_results

    def _generate_quality_action_items(
        self,
        categorized: Dict[str, List[str]],
        quality_data: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """Generate action items based on quality gate results"""
        action_items = []

        # Needs improvement
        for eval_id in categorized.get("needs_improvement", []):
            data = quality_data.get(eval_id, {})
            action_items.append({
                "evaluator_id": eval_id,
                "priority": "high",
                "action": "Immediate intervention required",
                "details": f"Quality score: {data.get('overall_quality_score', 0):.2f}"
            })

        # Acceptable
        for eval_id in categorized.get("acceptable", []):
            action_items.append({
                "evaluator_id": eval_id,
                "priority": "medium",
                "action": "Performance improvement plan recommended",
                "details": "Target: Improve to 'good' category"
            })

        return action_items

    def _calculate_team_trends(
        self,
        all_metrics: List[EvaluatorMetrics]
    ) -> Dict[str, Any]:
        """Calculate team-wide trends"""
        if not all_metrics:
            return {}

        trends = {}

        # Calculate average metrics over time
        for metric in ["average_score", "accuracy", "consistency_score"]:
            values = [getattr(m, metric, 0) for m in all_metrics]
            if values:
                trends[metric] = {
                    "current": float(np.mean(values)),
                    "trend": "stable"  # Would need historical data for real trend
                }

        return trends

    def _generate_report_id(self, config: ReportConfig) -> str:
        """Generate unique report ID"""
        config_str = json.dumps(self._config_to_dict(config), sort_keys=True)
        import hashlib
        return hashlib.md5(config_str.encode()).hexdigest()

    def _config_to_dict(self, config: ReportConfig) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "report_type": config.report_type.value,
            "format": config.format.value,
            "evaluator_ids": config.evaluator_ids,
            "stage": config.stage.value if config.stage else None,
            "start_date": config.start_date.isoformat() if config.start_date else None,
            "end_date": config.end_date.isoformat() if config.end_date else None,
            "include_charts": config.include_charts,
            "include_recommendations": config.include_recommendations,
            "include_biases": config.include_biases,
            "include_trends": config.include_trends,
            "comparison_baseline": config.comparison_baseline,
            "metrics": config.metrics
        }

    def _section_to_dict(self, section: ReportSection) -> Dict[str, Any]:
        """Convert section to dictionary"""
        return {
            "title": section.title,
            "content": section.content,
            "metadata": section.metadata or {}
        }

    def get_report_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of a report

        Args:
            report: Report dictionary

        Returns:
            Summary dictionary
        """
        summary = {
            "title": report.get("title", ""),
            "type": report.get("report_type", ""),
            "generated_at": report.get("generated_at", ""),
            "section_count": len(report.get("sections", [])),
            "sections": [s.get("title", "") for s in report.get("sections", [])]
        }

        return summary

    def schedule_report(
        self,
        config: ReportConfig,
        schedule: str  # cron expression or timedelta
    ) -> str:
        """
        Schedule a report to be generated periodically

        Args:
            config: Report configuration
            schedule: Schedule expression or timedelta

        Returns:
            Schedule ID
        """
        # In production, this would integrate with a task scheduler
        schedule_id = f"scheduled_{self._generate_report_id(config)}"

        # For now, just return the ID
        # In production, store schedule and implement execution logic
        return schedule_id

    def clear_cache(self) -> None:
        """Clear report cache"""
        self.report_cache.clear()

    def _get_ensemble_metrics(self, evaluator_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Extract ensemble metrics from evaluator metrics.

        Args:
            evaluator_ids: Optional list of evaluator IDs to filter

        Returns:
            Dictionary with ensemble metrics or None if no ensemble data
        """
        evaluator_ids = evaluator_ids or list(self.analytics.evaluator_metrics.keys())

        ensemble_data = []
        has_ensemble_data = False

        for eval_id in evaluator_ids:
            metrics = self.analytics.get_evaluator_metrics(eval_id)
            if metrics and hasattr(metrics, 'ensemble_selection_count'):
                has_ensemble_data = True
                ensemble_data.append({
                    "evaluator_id": eval_id,
                    "ensemble_selection_count": metrics.ensemble_selection_count,
                    "ensemble_weight": metrics.ensemble_weight,
                    "ensemble_utilization": metrics.ensemble_utilization,
                    "total_evaluations": metrics.total_evaluations,
                    "utilization_percentage": (
                        (metrics.ensemble_selection_count / metrics.total_evaluations * 100)
                        if metrics.total_evaluations > 0 else 0.0
                    )
                })

        if not has_ensemble_data:
            return None

        # Calculate aggregate ensemble metrics
        total_selections = sum(e["ensemble_selection_count"] for e in ensemble_data)
        total_evaluations = sum(e["total_evaluations"] for e in ensemble_data)

        return {
            "ensemble_mode_active": True,
            "total_evaluators": len(ensemble_data),
            "total_selections": total_selections,
            "total_evaluations": total_evaluations,
            "ensemble_utilization_rate": (
                total_selections / total_evaluations if total_evaluations > 0 else 0.0
            ),
            "evaluator_details": ensemble_data,
            "top_utilized_evaluators": sorted(
                ensemble_data,
                key=lambda x: x["ensemble_selection_count"],
                reverse=True
            )[:5],
            "weight_distribution": {
                e["evaluator_id"]: e["ensemble_weight"] for e in ensemble_data
            }
        }
