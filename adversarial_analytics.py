"""
Advanced Analytics and Visualization for Adversarial Testing

This module provides comprehensive analytics, visualization, and reporting
capabilities for the enhanced adversarial testing system.

Features:
1. Real-time metrics tracking
2. Interactive visualizations
3. Trend analysis
4. Comparative analysis
5. Performance benchmarking
6. Custom report generation
7. Data export capabilities
8. Statistical analysis

Author: OpenEvolve Analytics Team
Created: 2025-01-07
Version: 1.0.0
"""

import json
import logging
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class MetricPoint:
    """A single metric data point"""
    timestamp: str
    iteration: int
    metric_name: str
    metric_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Benchmark comparison result"""
    test_name: str
    timestamp: str
    old_system_metrics: Dict[str, float]
    new_system_metrics: Dict[str, float]
    improvement_percentage: Dict[str, float]
    winner: str


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    trend: str  # "increasing", "decreasing", "stable", "volatile"
    slope: float
    confidence: float
    forecast: List[float]
    seasonality: Optional[List[float]] = None


# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class AdversarialAnalyticsEngine:
    """
    Comprehensive analytics engine for adversarial testing

    Tracks metrics, generates insights, creates visualizations,
    and produces reports.
    """

    def __init__(self, storage_path: str = "./adversarial_analytics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Metric storage
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.benchmarks: List[BenchmarkResult] = []
        self.trends: Dict[str, TrendAnalysis] = {}

        # Aggregated metrics
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        logger.info("Analytics engine initialized")

    def record_metric(
        self,
        test_id: str,
        iteration: int,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric data point"""
        point = MetricPoint(
            timestamp=datetime.utcnow().isoformat(),
            iteration=iteration,
            metric_name=metric_name,
            metric_value=metric_value,
            metadata=metadata or {}
        )

        self.metrics[test_id].append(point)
        self.aggregated_metrics[test_id][metric_name] = metric_value

        # Persist to disk
        self._save_metric(test_id, point)

    def record_test_result(self, test_id: str, result: Dict[str, Any]):
        """Record all metrics from a test result"""
        # Extract key metrics
        if "final_robustness" in result:
            self.record_metric(
                test_id,
                result.get("iterations_completed", 0),
                "robustness",
                result["final_robustness"]
            )

        if "duration" in result:
            self.record_metric(
                test_id,
                result.get("iterations_completed", 0),
                "duration",
                result["duration"]
            )

        # Record attack metrics
        metrics_data = result.get("metrics", {})
        if isinstance(metrics_data, dict):
            for metric_name, metric_value in metrics_data.items():
                if isinstance(metric_value, (int, float)):
                    self.record_metric(
                        test_id,
                        result.get("iterations_completed", 0),
                        metric_name,
                        float(metric_value)
                    )

        # Save full result
        self._save_test_result(test_id, result)

    def get_metric_history(
        self,
        test_id: str,
        metric_name: str,
        last_n: Optional[int] = None
    ) -> List[MetricPoint]:
        """Get historical values for a metric"""
        points = [p for p in self.metrics.get(test_id, []) if p.metric_name == metric_name]

        if last_n:
            points = points[-last_n:]

        return points

    def calculate_statistics(
        self,
        test_id: str,
        metric_name: str
    ) -> Dict[str, float]:
        """Calculate statistical measures for a metric"""
        points = self.get_metric_history(test_id, metric_name)

        if not points:
            return {}

        values = [p.metric_value for p in points]

        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "mode": statistics.mode(values) if len(values) > 1 else values[0],
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "variance": statistics.variance(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "sum": sum(values),
        }

        # Percentiles
        if len(values) >= 4:
            stats["percentile_25"] = np.percentile(values, 25)
            stats["percentile_50"] = np.percentile(values, 50)
            stats["percentile_75"] = np.percentile(values, 75)
            stats["percentile_90"] = np.percentile(values, 90)
            stats["percentile_95"] = np.percentile(values, 95)
            stats["percentile_99"] = np.percentile(values, 99)

        return stats

    def analyze_trend(
        self,
        test_id: str,
        metric_name: str,
        window_size: int = 10
    ) -> TrendAnalysis:
        """Analyze trend for a metric"""
        points = self.get_metric_history(test_id, metric_name, last_n=window_size)

        if len(points) < 3:
            return TrendAnalysis(
                metric_name=metric_name,
                trend="insufficient_data",
                slope=0.0,
                confidence=0.0,
                forecast=[]
            )

        # Extract values and iterations
        iterations = np.array([p.iteration for p in points])
        values = np.array([p.metric_value for p in points])

        # Linear regression
        slope, intercept = np.polyfit(iterations, values, 1)

        # Calculate R-squared (confidence)
        predictions = slope * iterations + intercept
        ss_res = np.sum((values - predictions) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        # Simple forecast (next 5 points)
        next_iterations = np.arange(max(iterations) + 1, max(iterations) + 6)
        forecast = list(slope * next_iterations + intercept)

        analysis = TrendAnalysis(
            metric_name=metric_name,
            trend=trend,
            slope=float(slope),
            confidence=float(r_squared),
            forecast=forecast
        )

        self.trends[f"{test_id}_{metric_name}"] = analysis
        return analysis

    def compare_tests(
        self,
        test_ids: List[str],
        metric_name: str
    ) -> Dict[str, Dict[str, float]]:
        """Compare a metric across multiple tests"""
        comparison = {}

        for test_id in test_ids:
            stats = self.calculate_statistics(test_id, metric_name)
            comparison[test_id] = stats

        return comparison

    def generate_insights(
        self,
        test_id: str,
        max_insights: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate actionable insights from metrics"""
        insights = []

        # Get all metric names for this test
        metric_names = set(p.metric_name for p in self.metrics.get(test_id, []))

        for metric_name in metric_names:
            stats = self.calculate_statistics(test_id, metric_name)
            trend = self.analyze_trend(test_id, metric_name)

            # Generate insight based on trend and statistics
            insight = {
                "metric": metric_name,
                "current_value": stats.get("mean", 0),
                "trend": trend.trend,
                "trend_strength": abs(trend.slope),
                "confidence": trend.confidence,
                "insight": self._generate_insight_text(metric_name, stats, trend)
            }

            insights.append(insight)

        # Sort by confidence and trend strength
        insights.sort(key=lambda x: x["confidence"] * x["trend_strength"], reverse=True)

        return insights[:max_insights]

    def _generate_insight_text(
        self,
        metric_name: str,
        stats: Dict[str, float],
        trend: TrendAnalysis
    ) -> str:
        """Generate human-readable insight text"""
        if trend.trend == "increasing":
            if metric_name in ["robustness", "defense_effectiveness", "successful_defenses"]:
                return f"✓ {metric_name} is improving (+{trend.slope:.4f} per iteration)"
            else:
                return f"⚠️ {metric_name} is increasing (+{trend.slope:.4f} per iteration) - may need attention"
        elif trend.trend == "decreasing":
            if metric_name in ["robustness", "defense_effectiveness", "successful_defenses"]:
                return f"⚠️ {metric_name} is declining ({trend.slope:.4f} per iteration) - investigate"
            else:
                return f"✓ {metric_name} is decreasing ({trend.slope:.4f} per iteration)"
        else:
            return f"→ {metric_name} is stable (mean: {stats.get('mean', 0):.2f})"

    def generate_report(
        self,
        test_id: str,
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive analytics report"""
        report = {
            "test_id": test_id,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self._generate_summary(test_id),
            "metrics": {},
            "trends": {},
            "insights": self.generate_insights(test_id),
            "recommendations": self._generate_recommendations(test_id)
        }

        # Add detailed metrics
        metric_names = set(p.metric_name for p in self.metrics.get(test_id, []))
        for metric_name in metric_names:
            report["metrics"][metric_name] = self.calculate_statistics(test_id, metric_name)
            report["trends"][metric_name] = asdict(self.analyze_trend(test_id, metric_name))

        # Generate report
        if format == "json":
            report_str = json.dumps(report, indent=2)
        elif format == "html":
            report_str = self._generate_html_report(report)
        elif format == "markdown":
            report_str = self._generate_markdown_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {output_path}")

        return report_str

    def _generate_summary(self, test_id: str) -> Dict[str, Any]:
        """Generate test summary"""
        points = self.metrics.get(test_id, [])

        if not points:
            return {"status": "no_data"}

        return {
            "total_metrics": len(points),
            "iterations": max(p.iteration for p in points),
            "metric_types": len(set(p.metric_name for p in points)),
            "time_span": {
                "start": min(p.timestamp for p in points),
                "end": max(p.timestamp for p in points)
            }
        }

    def _generate_recommendations(self, test_id: str) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Get insights
        insights = self.generate_insights(test_id)

        for insight in insights:
            metric = insight["metric"]
            trend = insight["trend"]

            if metric == "robustness" and trend == "decreasing":
                recommendations.append(
                    "Robustness is declining. Consider: "
                    "1) Increasing iterations, "
                    "2) Enabling ensemble attacks, "
                    "3) Reviewing defense strategies"
                )

            if metric == "duration" and trend == "increasing":
                recommendations.append(
                    "Test duration is increasing. Consider: "
                    "1) Reducing ensemble size, "
                    "2) Using caching, "
                    "3) Parallel evaluation"
                )

            if metric == "attack_success_rate" and insight["current_value"] > 0.5:
                recommendations.append(
                    "High attack success rate detected. Strengthen defenses by: "
                    "1) Enabling formal verification, "
                    "2) Using adaptive defense, "
                    "3) Reviewing code logic"
                )

        return recommendations if recommendations else ["System performing well, no specific recommendations"]

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Adversarial Testing Report - {report['test_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .metric {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007cba; }}
        .insight {{ background: #e8f4f8; padding: 10px; margin: 5px 0; border-radius: 5px; }}
        .recommendation {{ background: #fff3cd; padding: 10px; margin: 5px 0; border-left: 4px solid #ffc107; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .neutral {{ color: #6c757d; }}
    </style>
</head>
<body>
    <h1>Adversarial Testing Report</h1>
    <p><strong>Test ID:</strong> {report['test_id']}</p>
    <p><strong>Generated:</strong> {report['generated_at']}</p>

    <h2>Summary</h2>
    <div class="metric">
        <p>Total Metrics: {report['summary']['total_metrics']}</p>
        <p>Iterations: {report['summary'].get('iterations', 'N/A')}</p>
        <p>Metric Types: {report['summary']['metric_types']}</p>
    </div>

    <h2>Insights</h2>
"""

        for insight in report['insights']:
            trend_class = "positive" if insight['trend'] == "increasing" else "negative" if insight['trend'] == "decreasing" else "neutral"
            html += f"""
    <div class="insight">
        <p><strong>{insight['metric']}</strong> (<span class="{trend_class}">{insight['trend']}</span>)</p>
        <p>{insight['insight']}</p>
        <p><small>Confidence: {insight['confidence']:.2%}</small></p>
    </div>
"""

        html += """
    <h2>Recommendations</h2>
"""

        for rec in report['recommendations']:
            html += f"""
    <div class="recommendation">{rec}</div>
"""

        html += """
</body>
</html>
"""
        return html

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        md = f"""# Adversarial Testing Report

**Test ID**: {report['test_id']}
**Generated**: {report['generated_at']}

## Summary

- **Total Metrics**: {report['summary']['total_metrics']}
- **Iterations**: {report['summary'].get('iterations', 'N/A')}
- **Metric Types**: {report['summary']['metric_types']}

## Insights

"""

        for insight in report['insights']:
            md += f"### {insight['metric']} ({insight['trend']})\n\n"
            md += f"{insight['insight']}\n\n"
            md += f"*Confidence: {insight['confidence']:.2%}*\n\n"

        md += "## Recommendations\n\n"

        for rec in report['recommendations']:
            md += f"- {rec}\n"

        return md

    def export_metrics(
        self,
        test_id: str,
        format: str = "csv",
        output_path: Optional[str] = None
    ) -> str:
        """Export metrics to file"""
        points = self.metrics.get(test_id, [])

        if not points:
            return ""

        if format == "csv":
            lines = ["timestamp,iteration,metric_name,value,metadata"]
            for point in points:
                metadata_str = json.dumps(point.metadata).replace('"', "'")
                lines.append(f"{point.timestamp},{point.iteration},{point.metric_name},{point.metric_value},{metadata_str}")

            output = "\n".join(lines)

        elif format == "json":
            output = json.dumps([asdict(p) for p in points], indent=2)

        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, 'w') as f:
                f.write(output)
            logger.info(f"Metrics exported to {output_path}")

        return output

    def benchmark_comparison(
        self,
        old_result: Dict[str, Any],
        new_result: Dict[str, Any],
        test_name: str = "comparison"
    ) -> BenchmarkResult:
        """Benchmark old vs new system"""
        old_metrics = {
            "robustness": old_result.get("final_robustness", 0),
            "duration": old_result.get("duration", 0),
            "attacks_successful": old_result.get("metrics", {}).get("successful_attacks", 0),
        }

        new_metrics = {
            "robustness": new_result.get("final_robustness", 0),
            "duration": new_result.get("duration", 0),
            "attacks_successful": new_result.get("metrics", {}).get("successful_attacks", 0),
        }

        improvement = {}
        for key in old_metrics:
            if old_metrics[key] > 0:
                improvement[key] = ((new_metrics[key] - old_metrics[key]) / old_metrics[key]) * 100
            else:
                improvement[key] = 0.0

        # Determine winner (new is better if robustness improved or duration reduced)
        new_robustness_better = improvement.get("robustness", 0) > 0
        new_duration_better = improvement.get("duration", 0) < 0

        winner = "new" if (new_robustness_better or new_duration_better) else "old"

        result = BenchmarkResult(
            test_name=test_name,
            timestamp=datetime.utcnow().isoformat(),
            old_system_metrics=old_metrics,
            new_system_metrics=new_metrics,
            improvement_percentage=improvement,
            winner=winner
        )

        self.benchmarks.append(result)
        return result

    def _save_metric(self, test_id: str, point: MetricPoint):
        """Save metric to disk"""
        try:
            metric_file = self.storage_path / f"{test_id}_metrics.jsonl"

            with open(metric_file, 'a') as f:
                f.write(json.dumps(asdict(point)) + "\n")
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to save metric: {e}")

    def _save_test_result(self, test_id: str, result: Dict[str, Any]):
        """Save full test result to disk"""
        try:
            result_file = self.storage_path / f"{test_id}_result.json"

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to save test result: {e}")

    def load_test_result(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Load test result from disk"""
        try:
            result_file = self.storage_path / f"{test_id}_result.json"

            if result_file.exists():
                with open(result_file, 'r') as f:
                    return json.load(f)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            logger.error(f"Failed to load test result: {e}")

        return None


# =============================================================================
# VISUALIZATION ENGINE
# =============================================================================

class AdversarialVisualizer:
    """
    Create visualizations for adversarial testing metrics

    Supports multiple chart types and export formats
    """

    def __init__(self, analytics_engine: AdversarialAnalyticsEngine):
        self.analytics = analytics_engine

    def create_metric_timeline(
        self,
        test_id: str,
        metric_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """Create timeline visualization for a metric"""
        points = self.analytics.get_metric_history(test_id, metric_name)

        if not points:
            return "No data available"

        # Create ASCII chart
        values = [p.metric_value for p in points]
        min_val = min(values)
        max_val = max(values)

        chart = []
        chart.append(f"\n{metric_name} Timeline")
        chart.append("=" * 60)

        for i, point in enumerate(points):
            # Normalize value to 0-50 range
            normalized = int(((point.metric_value - min_val) / (max_val - min_val)) * 50) if max_val > min_val else 25
            bar = "█" * normalized
            chart.append(f"I{point.iteration:3d} | {bar} {point.metric_value:.4f}")

        chart.append("=" * 60)

        result = "\n".join(chart)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(result)

        return result

    def create_comparison_chart(
        self,
        test_ids: List[str],
        metric_name: str
    ) -> str:
        """Create comparison chart across tests"""
        chart = []
        chart.append(f"\n{metric_name} Comparison")
        chart.append("=" * 80)

        for test_id in test_ids:
            stats = self.analytics.calculate_statistics(test_id, metric_name)
            if stats:
                mean = stats.get("mean", 0)
                std_dev = stats.get("std_dev", 0)
                min_val = stats.get("min", 0)
                max_val = stats.get("max", 0)

                chart.append(f"\n{test_id}:")
                chart.append(f"  Mean: {mean:.4f}")
                chart.append(f"  Std Dev: {std_dev:.4f}")
                chart.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")

        chart.append("\n" + "=" * 80)

        return "\n".join(chart)

    def create_trend_visualization(
        self,
        test_id: str,
        metric_name: str
    ) -> str:
        """Create trend visualization"""
        trend = self.analytics.analyze_trend(test_id, metric_name)

        viz = []
        viz.append(f"\nTrend Analysis: {metric_name}")
        viz.append("=" * 60)
        viz.append(f"Trend: {trend.trend}")
        viz.append(f"Slope: {trend.slope:.6f} per iteration")
        viz.append(f"Confidence (R²): {trend.confidence:.4f}")

        if trend.forecast:
            viz.append(f"\nForecast (next 5 iterations):")
            for i, val in enumerate(trend.forecast, 1):
                viz.append(f"  {i}. {val:.4f}")

        viz.append("=" * 60)

        return "\n".join(viz)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_analytics_dashboard(test_id: str, storage_path: str = "./adversarial_analytics") -> str:
    """Create comprehensive analytics dashboard"""
    analytics = AdversarialAnalyticsEngine(storage_path)
    visualizer = AdversarialVisualizer(analytics)

    dashboard = []
    dashboard.append("=" * 80)
    dashboard.append("  ADVERSARIAL TESTING ANALYTICS DASHBOARD")
    dashboard.append("=" * 80)
    dashboard.append(f"Test ID: {test_id}")
    dashboard.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    dashboard.append("")

    # Summary
    summary = analytics._generate_summary(test_id)
    dashboard.append("## SUMMARY")
    for key, value in summary.items():
        if isinstance(value, dict):
            dashboard.append(f"{key}:")
            for k, v in value.items():
                dashboard.append(f"  {k}: {v}")
        else:
            dashboard.append(f"{key}: {value}")
    dashboard.append("")

    # Insights
    insights = analytics.generate_insights(test_id)
    dashboard.append("## KEY INSIGHTS")
    for insight in insights[:5]:
        dashboard.append(f"- {insight['insight']}")
    dashboard.append("")

    # Metric timeline for key metrics
    key_metrics = ["robustness", "duration", "attack_success_rate"]
    for metric in key_metrics:
        if any(p.metric_name == metric for p in analytics.metrics.get(test_id, [])):
            timeline = visualizer.create_metric_timeline(test_id, metric)
            dashboard.append(timeline)
            dashboard.append("")

    # Recommendations
    recommendations = analytics._generate_recommendations(test_id)
    dashboard.append("## RECOMMENDATIONS")
    for rec in recommendations:
        dashboard.append(f"• {rec}")
    dashboard.append("")

    return "\n".join(dashboard)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Adversarial Analytics Engine")
    print("=" * 60)

    # Create analytics engine
    analytics = AdversarialAnalyticsEngine()

    # Record some sample metrics
    for i in range(10):
        analytics.record_metric(
            test_id="test_001",
            iteration=i,
            metric_name="robustness",
            metric_value=0.7 + (i * 0.02) + (random.random() * 0.1 - 0.05)
        )

    # Generate insights
    insights = analytics.generate_insights("test_001")
    print("\nInsights:")
    for insight in insights[:3]:
        print(f"  - {insight['insight']}")

    # Create visualization
    visualizer = AdversarialVisualizer(analytics)
    timeline = visualizer.create_metric_timeline("test_001", "robustness")
    print(timeline)

    # Generate report
    report = analytics.generate_report("test_001", format="markdown")
    print("\nReport:")
    print(report[:500] + "...")
