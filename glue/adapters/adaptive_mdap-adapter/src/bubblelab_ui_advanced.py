"""
Advanced BubbleLab UI Components for Adaptive MDAP/MAKER Adapter

This module provides advanced UI components for BubbleLab including:
- Interactive complexity radar charts
- Real-time MAKER voting progress display
- Workflow execution timeline visualization
- ICR pattern insights dashboard
- Adapter health metrics dashboard with alerts
- Export functionality for reports

All components return data structures compatible with common charting libraries
(Chart.js, Plotly, ECharts) and can be rendered in Streamlit, React, or vanilla HTML/JS.
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from bubblelab_ui_integration import (
    get_bubblelab_ui_integration,
    ComplexityAnalysisResult,
    MAKERVotingDisplay
)

from openevolve_advanced import get_advanced_openevolve_integration

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types."""
    RADAR = "radar"
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    TIMELINE = "timeline"
    HEATMAP = "heatmap"
    GAUGE = "gauge"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ChartData:
    """Generic chart data structure."""
    chart_type: ChartType
    title: str
    data: Dict[str, Any]
    options: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineEvent:
    """Event for timeline visualization."""
    event_id: str
    timestamp: str
    stage: str
    status: str
    duration_ms: float
    details: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedBubbleLabUI:
    """
    Advanced UI components for BubbleLab integration.

    All methods return data structures ready for visualization with
    common charting libraries.
    """

    def __init__(self):
        """Initialize advanced UI components."""
        self.base_ui = get_bubblelab_ui_integration()
        self.advanced_openevolve = get_advanced_openevolve_integration()

        # Alert tracking
        self.alerts: List[Alert] = []
        self.alert_id_counter = 0

        # Timeline tracking
        self.timeline_events: Dict[str, List[TimelineEvent]] = defaultdict(list)

        logger.info("Advanced BubbleLab UI initialized")

    def create_complexity_radar_chart(
        self,
        analysis_id: str,
        include_recommendations: bool = True
    ) -> Optional[ChartData]:
        """
        Create interactive radar chart for complexity breakdown.

        Args:
            analysis_id: Complexity analysis ID
            include_recommendations: Whether to include strategy recommendations

        Returns:
            ChartData ready for Chart.js/Plotly
        """
        # Get analysis result
        analysis_result = self.base_ui.active_analyses.get(analysis_id)
        if not analysis_result:
            return None

        # Create radar chart data
        chart_data = ChartData(
            chart_type=ChartType.RADAR,
            title=f"Complexity Analysis - {analysis_result.problem_id}",
            data={
                "labels": [
                    "Text Length",
                    "Dependencies",
                    "Depth",
                    "Domain Knowledge",
                    "Resource Requirements"
                ],
                "datasets": [{
                    "label": "Complexity Scores",
                    "data": [
                        analysis_result.text_length_score,
                        analysis_result.dependency_score,
                        analysis_result.depth_score,
                        min(1.0, analysis_result.overall_complexity * 1.2),  # Simulated domain knowledge
                        min(1.0, analysis_result.overall_complexity * 1.5)   # Simulated resources
                    ],
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2,
                    "pointBackgroundColor": "rgba(54, 162, 235, 1)",
                    "pointBorderColor": "#fff",
                    "pointHoverBackgroundColor": "#fff",
                    "pointHoverBorderColor": "rgba(54, 162, 235, 1)"
                }]
            },
            options={
                "responsive": True,
                "scales": {
                    "r": {
                        "beginAtZero": True,
                        "max": 1.0,
                        "ticks": {
                            "stepSize": 0.2
                        }
                    }
                },
                "plugins": {
                    "legend": {
                        "position": "top"
                    },
                    "title": {
                        "display": True,
                        "text": f"Overall Complexity: {analysis_result.overall_complexity:.3f}"
                    }
                }
            },
            metadata={
                "analysis_id": analysis_id,
                "overall_complexity": analysis_result.overall_complexity,
                "recommended_strategy": analysis_result.recommended_strategy if include_recommendations else None,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
        )

        return chart_data

    def create_maker_voting_chart(
        self,
        workflow_id: str,
        decision_point: str
    ) -> Optional[ChartData]:
        """
        Create bar chart showing MAKER voting progress.

        Args:
            workflow_id: OpenEvolve workflow ID
            decision_point: Decision point description

        Returns:
            ChartData for voting visualization
        """
        # Get voting data (simulated for now, would come from active_votings)
        voting_data = self.base_ui.active_votings.get(f"{workflow_id}_{decision_point}")

        if not voting_data:
            # Create sample data
            chart_data = ChartData(
                chart_type=ChartType.BAR,
                title=f"MAKER Voting - {decision_point}",
                data={
                    "labels": ["Option A", "Option B", "Option C", "Option D", "Option E"],
                    "datasets": [{
                        "label": "Votes",
                        "data": [3, 5, 2, 1, 0],
                        "backgroundColor": [
                            "rgba(255, 99, 132, 0.6)",
                            "rgba(54, 162, 235, 0.6)",
                            "rgba(255, 206, 86, 0.6)",
                            "rgba(75, 192, 192, 0.6)",
                            "rgba(153, 102, 255, 0.6)"
                        ]
                    }]
                },
                options={
                    "responsive": True,
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 10
                        }
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Voting Progress (Simulated)"
                        }
                    }
                },
                metadata={"simulated": True}
            )
            return chart_data

        # Real voting data chart
        chart_data = ChartData(
            chart_type=ChartType.BAR,
            title=f"MAKER Voting - {voting_data.decision_point}",
            data={
                "labels": ["Consensus", "Dissenting"],
                "datasets": [{
                    "label": "Votes",
                    "data": [
                        voting_data.consensus_score * voting_data.votes_collected,
                        (1 - voting_data.consensus_score) * voting_data.votes_collected
                    ],
                    "backgroundColor": [
                        "rgba(75, 192, 192, 0.6)",
                        "rgba(255, 99, 132, 0.6)"
                    ]
                }]
            },
            options={
                "responsive": True,
                "plugins": {
                    "title": {
                        "display": True,
                        "text": f"Consensus: {voting_data.consensus_reached} ({voting_data.consensus_score:.3f})"
                    }
                }
            },
            metadata={
                "voting_id": voting_data.voting_id,
                "timestamp": voting_data.timestamp
            }
        )

        return chart_data

    def create_workflow_timeline(
        self,
        workflow_id: str
    ) -> ChartData:
        """
        Create timeline visualization of workflow execution.

        Args:
            workflow_id: OpenEvolve workflow ID

        Returns:
            ChartData for timeline visualization
        """
        # Get or create timeline events
        events = self.timeline_events.get(workflow_id, [])

        if not events:
            # Create sample timeline
            now = time.time() * 1000
            events = [
                TimelineEvent(
                    event_id=f"evt_{workflow_id}_1",
                    timestamp=datetime.fromtimestamp(now / 1000, tz=timezone.utc).isoformat(),
                    stage="Content Input",
                    status="completed",
                    duration_ms=500,
                    details={"progress": 100},
                    metadata={"icon": "input"}
                ),
                TimelineEvent(
                    event_id=f"evt_{workflow_id}_2",
                    timestamp=datetime.fromtimestamp((now + 500) / 1000, tz=timezone.utc).isoformat(),
                    stage="Analysis",
                    status="completed",
                    duration_ms=1500,
                    details={"progress": 100},
                    metadata={"icon": "analysis"}
                ),
                TimelineEvent(
                    event_id=f"evt_{workflow_id}_3",
                    timestamp=datetime.fromtimestamp((now + 2000) / 1000, tz=timezone.utc).isoformat(),
                    stage="Planning",
                    status="in_progress",
                    duration_ms=2000,
                    details={"progress": 60},
                    metadata={"icon": "planning"}
                )
            ]
            self.timeline_events[workflow_id] = events

        # Create Gantt-style timeline data
        chart_data = ChartData(
            chart_type=ChartType.TIMELINE,
            title=f"Workflow Execution Timeline - {workflow_id}",
            data={
                "events": [
                    {
                        "event_id": evt.event_id,
                        "stage": evt.stage,
                        "start": evt.timestamp,
                        "duration_ms": evt.duration_ms,
                        "status": evt.status,
                        "progress": evt.details.get("progress", 0)
                    }
                    for evt in events
                ]
            },
            options={
                "responsive": True,
                "height": 300,
                "scales": {
                    "x": {
                        "type": "time",
                        "time": {
                            "unit": "millisecond"
                        }
                    }
                }
            },
            metadata={
                "workflow_id": workflow_id,
                "total_events": len(events),
                "total_duration_ms": sum(evt.duration_ms for evt in events)
            }
        )

        return chart_data

    def create_icr_insights_dashboard(self) -> ChartData:
        """
        Create dashboard showing ICR pattern learning insights.

        Returns:
            ChartData for ICR insights visualization
        """
        # Get ICR insights
        icr_insights = self.base_ui.get_icr_insights()

        if not icr_insights.get("available"):
            return ChartData(
                chart_type=ChartType.BAR,
                title="ICR Pattern Learning (Not Available)",
                data={"labels": [], "datasets": []},
                options={},
                metadata={"available": False}
            )

        patterns = icr_insights.get("patterns", {})

        # Create insights dashboard
        chart_data = ChartData(
            chart_type=ChartType.BAR,
            title="ICR Pattern Learning Insights",
            data={
                "labels": list(patterns.keys()),
                "datasets": [
                    {
                        "label": "Pattern Count",
                        "data": [p.get("count", 0) for p in patterns.values()],
                        "backgroundColor": "rgba(54, 162, 235, 0.6)",
                        "borderColor": "rgba(54, 162, 235, 1)",
                        "yAxisID": "y"
                    },
                    {
                        "label": "Pass Rate",
                        "data": [p.get("pass_rate", 0) * 100 for p in patterns.values()],
                        "type": "line",
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "backgroundColor": "rgba(255, 99, 132, 0.2)",
                        "yAxisID": "y1"
                    }
                ]
            },
            options={
                "responsive": True,
                "scales": {
                    "y": {
                        "type": "linear",
                        "display": True,
                        "position": "left"
                    },
                    "y1": {
                        "type": "linear",
                        "display": True,
                        "position": "right",
                        "grid": {
                            "drawOnChartArea": False
                        },
                        "min": 0,
                        "max": 100
                    }
                },
                "plugins": {
                    "title": {
                        "display": True,
                        "text": "Pattern Learning Performance"
                    }
                }
            },
            metadata={
                "total_patterns": sum(p.get("count", 0) for p in patterns.values()),
                "average_confidence": sum(p.get("confidence", 0) for p in patterns.values()) / len(patterns) if patterns else 0
            }
        )

        return chart_data

    def create_adapter_health_dashboard(self) -> Dict[str, Any]:
        """
        Create comprehensive adapter health dashboard with alerts.

        Returns:
            Dashboard data with health metrics and alerts
        """
        health = self.base_ui.get_adapter_health_status()

        # Generate alerts based on health status
        new_alerts = []

        # Check MDAP adapter
        mdap_status = health.get("mdap_adapter", {}).get("status", "unknown")
        if mdap_status == "error":
            new_alerts.append(Alert(
                alert_id=f"alert_mdap_{int(time.time() * 1000)}",
                severity=AlertSeverity.CRITICAL,
                component="mdap_adapter",
                message="MDAP adapter is in error state",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"status": mdap_status}
            ))
        elif mdap_status == "degraded":
            new_alerts.append(Alert(
                alert_id=f"alert_mdap_{int(time.time() * 1000)}",
                severity=AlertSeverity.WARNING,
                component="mdap_adapter",
                message="MDAP adapter is degraded",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"status": mdap_status}
            ))

        # Check MAKER adapter
        maker_status = health.get("maker_adapter", {}).get("status", "unknown")
        if maker_status == "error":
            new_alerts.append(Alert(
                alert_id=f"alert_maker_{int(time.time() * 1000)}",
                severity=AlertSeverity.CRITICAL,
                component="maker_adapter",
                message="MAKER adapter is in error state",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"status": maker_status}
            ))
        elif maker_status == "degraded":
            new_alerts.append(Alert(
                alert_id=f"alert_maker_{int(time.time() * 1000)}",
                severity=AlertSeverity.WARNING,
                component="maker_adapter",
                message="MAKER adapter is degraded",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"status": maker_status}
            ))

        # Add new alerts
        self.alerts.extend(new_alerts)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        # Create dashboard data
        dashboard = {
            "health": health,
            "alerts": [
                {
                    "severity": alert.severity.value,
                    "component": alert.component,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in self.alerts[-20:]  # Last 20 alerts
            ],
            "metrics": {
                "mdap_requests_total": health.get("mdap_adapter", {}).get("metrics", {}).get("requests_total", 0),
                "mdap_requests_success": health.get("mdap_adapter", {}).get("metrics", {}).get("requests_success", 0),
                "mdap_success_rate": 0.0,
                "maker_requests_total": health.get("maker_adapter", {}).get("metrics", {}).get("requests_total", 0),
                "maker_requests_success": health.get("maker_adapter", {}).get("metrics", {}).get("requests_success", 0),
                "maker_success_rate": 0.0
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Calculate success rates
        mdap_total = dashboard["metrics"]["mdap_requests_total"]
        mdap_success = dashboard["metrics"]["mdap_requests_success"]
        if mdap_total > 0:
            dashboard["metrics"]["mdap_success_rate"] = mdap_success / mdap_total

        maker_total = dashboard["metrics"]["maker_requests_total"]
        maker_success = dashboard["metrics"]["maker_requests_success"]
        if maker_total > 0:
            dashboard["metrics"]["maker_success_rate"] = maker_success / maker_total

        return dashboard

    def export_report(
        self,
        workflow_id: str,
        format: str = "json",
        include_charts: bool = True
    ) -> str:
        """
        Export comprehensive report for a workflow.

        Args:
            workflow_id: OpenEvolve workflow ID
            format: Export format ("json" or "markdown")
            include_charts: Whether to include chart data

        Returns:
            Exported report as string
        """
        # Gather all data
        health = self.base_ui.get_adapter_health_status()
        timeline = self.create_workflow_timeline(workflow_id)
        icr_insights = self.base_ui.get_icr_insights()
        dashboard = self.create_adapter_health_dashboard()

        report = {
            "workflow_id": workflow_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "health": health,
            "timeline": timeline.data if include_charts else None,
            "icr_insights": icr_insights,
            "dashboard": dashboard,
            "alerts": dashboard["alerts"][-10:]  # Last 10 alerts
        }

        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "markdown":
            return self._format_markdown_report(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format report as markdown."""
        md = []
        md.append(f"# Workflow Report: {report['workflow_id']}")
        md.append(f"\nGenerated: {report['generated_at']}\n")

        # Health Status
        md.append("## Health Status")
        md.append(f"- MDAP Adapter: {report['health']['mdap_adapter']['status']}")
        md.append(f"- MAKER Adapter: {report['health']['maker_adapter']['status']}")
        md.append("")

        # Alerts
        if report['alerts']:
            md.append("## Recent Alerts")
            for alert in report['alerts']:
                severity_icon = {
                    "critical": "🔴",
                    "error": "❌",
                    "warning": "⚠️",
                    "info": "ℹ️"
                }.get(alert['severity'], "•")
                md.append(f"- {severity_icon} **{alert['component']}**: {alert['message']}")
            md.append("")

        # Timeline
        md.append("## Workflow Timeline")
        events = report.get('timeline', {}).get('events', [])
        for event in events:
            status_icon = "✅" if event['status'] == 'completed' else "🔄"
            md.append(f"- {status_icon} **{event['stage']}** ({event['duration_ms']}ms)")
        md.append("")

        # ICR Insights
        md.append("## ICR Pattern Learning")
        icr = report.get('icr_insights', {})
        if icr.get('available'):
            patterns = icr.get('patterns', {})
            for pattern_type, stats in patterns.items():
                md.append(f"- **{pattern_type}**:")
                md.append(f"  - Count: {stats.get('count', 0)}")
                md.append(f"  - Pass Rate: {stats.get('pass_rate', 0):.1%}")
                md.append(f"  - Confidence: {stats.get('confidence', 0):.1%}")
        else:
            md.append("ICR integration not available")

        return "\n".join(md)


# Global instance
_advanced_ui: Optional[AdvancedBubbleLabUI] = None


def get_advanced_bubblelab_ui() -> AdvancedBubbleLabUI:
    """Get or create global advanced UI instance."""
    global _advanced_ui
    if _advanced_ui is None:
        _advanced_ui = AdvancedBubbleLabUI()
    return _advanced_ui


__all__ = [
    "ChartType",
    "AlertSeverity",
    "ChartData",
    "Alert",
    "TimelineEvent",
    "AdvancedBubbleLabUI",
    "get_advanced_bubblelab_ui"
]
