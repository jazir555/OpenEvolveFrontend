"""
Decomposition Dashboard Module

Provides a comprehensive web-based dashboard for monitoring decomposition progress
with real-time updates, interactive visualizations, and rich configuration panels.

Features:
- Real-time progress monitoring
- Interactive dependency graphs
- Quality metrics visualization
- Team assignment views
- Strategy performance tracking
- Export to PNG/SVG/PDF
- Responsive design
- Dark/light theme support
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO
import asyncio
from dataclasses import asdict

logger = logging.getLogger(__name__)

class DashboardConfig:
    """Configuration for dashboard rendering."""

    def __init__(
        self,
        theme: str = "light",
        refresh_interval: int = 30,
        auto_refresh: bool = True,
        show_metrics: bool = True,
        show_timeline: bool = True,
        show_quality: bool = True,
        show_teams: bool = True,
        export_formats: List[str] = None
    ):
        """
        Initialize dashboard configuration.

        Args:
            theme: Theme mode ("light" or "dark")
            refresh_interval: Auto-refresh interval in seconds
            auto_refresh: Enable auto-refresh
            show_metrics: Show metrics section
            show_timeline: Show timeline view
            show_quality: Show quality charts
            show_teams: Show team assignments
            export_formats: Available export formats
        """
        self.theme = theme
        self.refresh_interval = refresh_interval
        self.auto_refresh = auto_refresh
        self.show_metrics = show_metrics
        self.show_timeline = show_timeline
        self.show_quality = show_quality
        self.show_teams = show_teams
        self.export_formats = export_formats or ["png", "svg", "pdf", "html"]


class DecompositionDashboard:
    """
    Comprehensive web dashboard for decomposition monitoring.

    Features:
    - Real-time progress updates
    - Interactive visualizations
    - Export capabilities
    - Theme support
    - Responsive layout
    """

    def __init__(self, config: DashboardConfig = None):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        logger.info(f"DecompositionDashboard initialized with theme: {self.config.theme}")

    def generate_dashboard(
        self,
        workflow_id: str,
        state,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive dashboard HTML.

        Args:
            workflow_id: Workflow identifier
            state: WorkflowState object
            output_path: Optional path to save dashboard

        Returns:
            Path to generated dashboard HTML
        """
        logger.info(f"Generating dashboard for workflow {workflow_id}")

        if output_path is None:
            output_path = f"dashboard_{workflow_id}.html"

        # Extract data from state
        plan = getattr(state, 'plan', None)
        progress = self._calculate_progress(state)

        # Generate dashboard sections
        html_sections = []

        # Header
        html_sections.append(self._generate_header(workflow_id, state))

        # Navigation tabs
        html_sections.append(self._generate_navigation())

        # Overview section
        html_sections.append(self._generate_overview_section(workflow_id, state, progress))

        # Progress section
        if self.config.show_metrics:
            html_sections.append(self._generate_progress_section(state, progress))

        # Dependency graph section
        if plan:
            html_sections.append(self._generate_dependency_section(plan))

        # Timeline section
        if self.config.show_timeline and plan:
            html_sections.append(self._generate_timeline_section(plan))

        # Quality metrics section
        if self.config.show_quality:
            html_sections.append(self._generate_quality_section(state))

        # Team assignments section
        if self.config.show_teams and plan:
            html_sections.append(self._generate_team_section(plan, state))

        # Logs section
        html_sections.append(self._generate_logs_section(state))

        # Export controls
        html_sections.append(self._generate_export_controls())

        # Combine sections
        html = self._wrap_html(
            f"Decomposition Dashboard - {workflow_id}",
            "\n".join(html_sections)
        )

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Dashboard generated at {output_path}")
        return output_path

    def _calculate_progress(self, state) -> Dict[str, Any]:
        """Calculate progress metrics from state."""
        plan = getattr(state, 'plan', None)

        if not plan:
            return {
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "blocked": 0,
                "failed": 0,
                "percentage": 0.0,
                "stages": []
            }

        total = len(plan.sub_problems)
        completed = len([sp for sp in plan.sub_problems if sp.status.value == "solved"])
        in_progress = len([sp for sp in plan.sub_problems if sp.status.value == "in_progress"])
        blocked = len([sp for sp in plan.sub_problems if sp.status.value == "blocked"])
        failed = len([sp for sp in plan.sub_problems if sp.status.value == "failed"])

        percentage = (completed / total * 100) if total > 0 else 0.0

        # Calculate stage progress
        stages = []
        stage_names = ["analysis", "decomposition", "validation", "execution"]
        for idx, stage_name in enumerate(stage_names):
            stage_progress = min(100, (percentage / (idx + 1)) * 100)
            stages.append({
                "name": stage_name.title(),
                "progress": stage_progress,
                "status": "completed" if stage_progress >= 100 else "active" if stage_progress > 0 else "pending"
            })

        return {
            "total": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "failed": failed,
            "percentage": percentage,
            "stages": stages
        }

    def _generate_header(self, workflow_id: str, state) -> str:
        """Generate dashboard header."""
        status = getattr(state, 'status', 'unknown')
        stage = getattr(state, 'current_stage', 'unknown')
        updated = getattr(state, 'updated_at', datetime.now())

        return f"""
        <header class="dashboard-header">
            <div class="header-content">
                <div class="header-title">
                    <h1>Decomposition Dashboard</h1>
                    <p class="workflow-id">Workflow: {workflow_id}</p>
                </div>
                <div class="header-status">
                    <div class="status-badge status-{status}">
                        <span class="status-dot"></span>
                        {status.replace('_', ' ').title()}
                    </div>
                    <div class="stage-info">
                        Stage: {stage.replace('_', ' ').title()}
                    </div>
                    <div class="last-updated">
                        Updated: {updated.strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                </div>
            </div>
        </header>
        """

    def _generate_navigation(self) -> str:
        """Generate navigation tabs."""
        return """
        <nav class="dashboard-nav">
            <button class="nav-tab active" data-tab="overview">
                <i class="icon-overview"></i>
                Overview
            </button>
            <button class="nav-tab" data-tab="progress">
                <i class="icon-progress"></i>
                Progress
            </button>
            <button class="nav-tab" data-tab="dependencies">
                <i class="icon-dependencies"></i>
                Dependencies
            </button>
            <button class="nav-tab" data-tab="timeline">
                <i class="icon-timeline"></i>
                Timeline
            </button>
            <button class="nav-tab" data-tab="quality">
                <i class="icon-quality"></i>
                Quality
            </button>
            <button class="nav-tab" data-tab="teams">
                <i class="icon-teams"></i>
                Teams
            </button>
            <button class="nav-tab" data-tab="logs">
                <i class="icon-logs"></i>
                Logs
            </button>
        </nav>
        """

    def _generate_overview_section(self, workflow_id: str, state, progress: Dict) -> str:
        """Generate overview section with key metrics."""
        return f"""
        <section id="overview" class="tab-content active">
            <h2>Overview</h2>

            <div class="metrics-grid">
                <div class="metric-card primary">
                    <div class="metric-icon">📊</div>
                    <div class="metric-value">{progress['percentage']:.1f}%</div>
                    <div class="metric-label">Overall Progress</div>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: {progress['percentage']}%"></div>
                    </div>
                </div>

                <div class="metric-card success">
                    <div class="metric-icon">✅</div>
                    <div class="metric-value">{progress['completed']}</div>
                    <div class="metric-label">Completed</div>
                </div>

                <div class="metric-card info">
                    <div class="metric-icon">🔄</div>
                    <div class="metric-value">{progress['in_progress']}</div>
                    <div class="metric-label">In Progress</div>
                </div>

                <div class="metric-card warning">
                    <div class="metric-icon">⚠️</div>
                    <div class="metric-value">{progress['blocked']}</div>
                    <div class="metric-label">Blocked</div>
                </div>

                <div class="metric-card danger">
                    <div class="metric-icon">❌</div>
                    <div class="metric-value">{progress['failed']}</div>
                    <div class="metric-label">Failed</div>
                </div>

                <div class="metric-card secondary">
                    <div class="metric-icon">📋</div>
                    <div class="metric-value">{progress['total']}</div>
                    <div class="metric-label">Total Sub-Problems</div>
                </div>
            </div>

            <div class="progress-stages">
                <h3>Decomposition Stages</h3>
                <div class="stages-container">
        """

        # Add stage progress bars
        for stage in progress['stages']:
            stage_class = f"stage-{stage['status']}"
            html = f"""
                    <div class="stage-item {stage_class}">
                        <div class="stage-label">{stage['name']}</div>
                        <div class="stage-progress-bar">
                            <div class="stage-progress-fill" style="width: {stage['progress']}%"></div>
                        </div>
                        <div class="stage-percentage">{stage['progress']:.0f}%</div>
                    </div>
            """
            overview_section = overview_section + html if 'overview_section' in locals() else html

        overview_section = overview_section + """
                </div>
            </div>

            <div class="quick-stats">
                <h3>Quick Statistics</h3>
                <div class="stats-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Success Rate</td>
                                <td>{:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Remaining Tasks</td>
                                <td>{}</td>
                            </tr>
                            <tr>
                                <td>Issues Detected</td>
                                <td>{}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
        """.format(
            (progress['completed'] / progress['total'] * 100) if progress['total'] > 0 else 0,
            progress['total'] - progress['completed'] - progress['in_progress'],
            progress['blocked'] + progress['failed']
        ) if 'overview_section' in locals() else ""

        return locals().get('overview_section', f'<section id="overview" class="tab-content active"><h2>Overview</h2><p>No data available</p></section>')

    def _generate_progress_section(self, state, progress: Dict) -> str:
        """Generate detailed progress section."""
        return f"""
        <section id="progress" class="tab-content">
            <h2>Detailed Progress</h2>

            <div class="progress-overview">
                <div class="progress-chart">
                    <canvas id="progressChart"></canvas>
                </div>
                <div class="progress-details">
                    <h3>Status Breakdown</h3>

                    <div class="status-breakdown">
                        <div class="status-item">
                            <div class="status-label">
                                <span class="status-indicator completed"></span>
                                Completed
                            </div>
                            <div class="status-bar">
                                <div class="status-fill completed" style="width: {(progress['completed']/progress['total']*100) if progress['total'] > 0 else 0}%"></div>
                            </div>
                            <div class="status-count">{progress['completed']} / {progress['total']}</div>
                        </div>

                        <div class="status-item">
                            <div class="status-label">
                                <span class="status-indicator in-progress"></span>
                                In Progress
                            </div>
                            <div class="status-bar">
                                <div class="status-fill in-progress" style="width: {(progress['in_progress']/progress['total']*100) if progress['total'] > 0 else 0}%"></div>
                            </div>
                            <div class="status-count">{progress['in_progress']} / {progress['total']}</div>
                        </div>

                        <div class="status-item">
                            <div class="status-label">
                                <span class="status-indicator blocked"></span>
                                Blocked
                            </div>
                            <div class="status-bar">
                                <div class="status-fill blocked" style="width: {(progress['blocked']/progress['total']*100) if progress['total'] > 0 else 0}%"></div>
                            </div>
                            <div class="status-count">{progress['blocked']} / {progress['total']}</div>
                        </div>

                        <div class="status-item">
                            <div class="status-label">
                                <span class="status-indicator failed"></span>
                                Failed
                            </div>
                            <div class="status-bar">
                                <div class="status-fill failed" style="width: {(progress['failed']/progress['total']*100) if progress['total'] > 0 else 0}%"></div>
                            </div>
                            <div class="status-count">{progress['failed']} / {progress['total']}</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _generate_dependency_section(self, plan) -> str:
        """Generate dependency graph section."""
        # Extract nodes and edges
        nodes = []
        edges = []

        for sp in plan.sub_problems:
            nodes.append({
                "id": sp.id,
                "label": sp.title,
                "title": f"{sp.title}\nPriority: {sp.priority}\nStatus: {sp.status.value}",
                "status": sp.status.value,
                "priority": sp.priority
            })

            for dep_id in sp.dependencies:
                edges.append({
                    "from": dep_id,
                    "to": sp.id
                })

        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)

        return f"""
        <section id="dependencies" class="tab-content">
            <h2>Dependency Graph</h2>

            <div class="graph-controls">
                <button id="refreshGraph" class="btn btn-secondary">
                    <i class="icon-refresh"></i> Refresh
                </button>
                <button id="exportGraph" class="btn btn-secondary">
                    <i class="icon-export"></i> Export
                </button>
                <button id="fitGraph" class="btn btn-secondary">
                    <i class="icon-fit"></i> Fit to Screen
                </button>
            </div>

            <div id="dependencyGraph" class="graph-container"></div>

            <div class="graph-legend">
                <h3>Legend</h3>
                <div class="legend-items">
                    <div class="legend-item">
                        <span class="legend-color solved"></span>
                        <span>Solved</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color in-progress"></span>
                        <span>In Progress</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color pending"></span>
                        <span>Pending</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color blocked"></span>
                        <span>Blocked</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-color failed"></span>
                        <span>Failed</span>
                    </div>
                </div>
            </div>

            <script type="text/javascript">
                // Dependency graph data
                const graphNodes = {nodes_json};
                const graphEdges = {edges_json};

                // Initialize graph (will be loaded by vis.js)
                document.addEventListener('DOMContentLoaded', function() {{
                    initDependencyGraph(graphNodes, graphEdges);
                }});
            </script>
        </section>
        """

    def _generate_timeline_section(self, plan) -> str:
        """Generate timeline visualization section."""
        # Calculate timeline data
        timeline_items = []
        for sp in plan.sub_problems:
            effort_hours = getattr(sp, 'estimated_resources', {}).get('time_hours', 8.0)
            timeline_items.append({
                "id": sp.id,
                "title": sp.title,
                "status": sp.status.value,
                "priority": sp.priority,
                "dependencies": len(sp.dependencies),
                "estimated_hours": effort_hours
            })

        timeline_json = json.dumps(timeline_items)

        return f"""
        <section id="timeline" class="tab-content">
            <h2>Execution Timeline</h2>

            <div class="timeline-controls">
                <button id="timelineDayView" class="btn btn-secondary">Day View</button>
                <button id="timelineWeekView" class="btn btn-secondary">Week View</button>
                <button id="timelineMonthView" class="btn btn-secondary">Month View</button>
            </div>

            <div id="timelineChart" class="timeline-container"></div>

            <div class="timeline-details">
                <h3>Sub-Problem Timeline</h3>
                <div class="timeline-list">
            """

        for item in timeline_items[:20]:  # Limit to 20 items
            section = section + f"""
                    <div class="timeline-item status-{item['status']}">
                        <div class="timeline-marker"></div>
                        <div class="timeline-content">
                            <h4>{item['title']}</h4>
                            <p>
                                <strong>Status:</strong> {item['status'].replace('_', ' ').title()} |
                                <strong>Priority:</strong> {item['priority']}/10 |
                                <strong>Dependencies:</strong> {item['dependencies']} |
                                <strong>Est. Time:</strong> {item['estimated_hours']}h
                            </p>
                        </div>
                    </div>
            """ if 'section' in locals() else f"""
                    <div class="timeline-item status-{item['status']}">
                        <div class="timeline-marker"></div>
                        <div class="timeline-content">
                            <h4>{item['title']}</h4>
                            <p>
                                <strong>Status:</strong> {item['status'].replace('_', ' ').title()} |
                                <strong>Priority:</strong> {item['priority']}/10 |
                                <strong>Dependencies:</strong> {item['dependencies']} |
                                <strong>Est. Time:</strong> {item['estimated_hours']}h
                            </p>
                        </div>
                    </div>
            """

        section = section + f"""
                </div>
            </div>

            <script type="text/javascript">
                const timelineData = {timeline_json};
                document.addEventListener('DOMContentLoaded', function() {{
                    initTimeline(timelineData);
                }});
            </script>
        </section>
        """ if 'section' in locals() else ""

        return locals().get('section', '<section id="timeline" class="tab-content"><h2>Timeline</h2><p>No timeline data available</p></section>')

    def _generate_quality_section(self, state) -> str:
        """Generate quality metrics section."""
        # Extract quality metrics if available
        quality_metrics = getattr(state, 'quality_metrics', {
            'completeness': 0.85,
            'correctness': 0.90,
            'quality': 0.82,
            'performance': 0.78,
            'security': 0.88
        })

        metrics_json = json.dumps(quality_metrics)

        return f"""
        <section id="quality" class="tab-content">
            <h2>Quality Metrics</h2>

            <div class="quality-overview">
                <div class="quality-radar">
                    <canvas id="qualityRadar"></canvas>
                </div>
                <div class="quality-details">
                    <h3>Quality Dimensions</h3>
            """

        for dim, value in quality_metrics.items():
            quality_section = quality_section + f"""
                    <div class="quality-item">
                        <div class="quality-label">
                            <span>{dim.replace('_', ' ').title()}</span>
                            <span class="quality-score">{value * 100:.1f}%</span>
                        </div>
                        <div class="quality-bar">
                            <div class="quality-fill" style="width: {value * 100}%"></div>
                        </div>
                    </div>
            """ if 'quality_section' in locals() else f"""
                    <div class="quality-item">
                        <div class="quality-label">
                            <span>{dim.replace('_', ' ').title()}</span>
                            <span class="quality-score">{value * 100:.1f}%</span>
                        </div>
                        <div class="quality-bar">
                            <div class="quality-fill" style="width: {value * 100}%"></div>
                        </div>
                    </div>
            """

        quality_section = quality_section + f"""
                </div>
            </div>

            <script type="text/javascript">
                const qualityData = {metrics_json};
                document.addEventListener('DOMContentLoaded', function() {{
                    initQualityChart(qualityData);
                }});
            </script>
        </section>
        """ if 'quality_section' in locals() else ""

        return locals().get('quality_section', '<section id="quality" class="tab-content"><h2>Quality Metrics</h2><p>No quality data available</p></section>')

    def _generate_team_section(self, plan, state) -> str:
        """Generate team assignments section."""
        # Analyze team assignments
        team_data = {
            'solver': {'count': 0, 'items': []},
            'patcher': {'count': 0, 'items': []},
            'red_team': {'count': 0, 'items': []},
            'gold_team': {'count': 0, 'items': []}
        }

        for sp in plan.sub_problems:
            assignments = getattr(sp, 'team_assignments', {})
            for team in team_data.keys():
                if assignments.get(team):
                    team_data[team]['count'] += 1
                    team_data[team]['items'].append(sp.title)

        return f"""
        <section id="teams" class="tab-content">
            <h2>Team Assignments</h2>

            <div class="team-overview">
                <div class="team-grid">
                    <div class="team-card solver">
                        <div class="team-header">
                            <h3>Solver Team</h3>
                            <div class="team-badge">{team_data['solver']['count']}</div>
                        </div>
                        <div class="team-list">
            """

        for title in team_data['solver']['items'][:10]:
            team_section = team_section + f"                            <div class='team-item'>{title}</div>\n" if 'team_section' in locals() else f"                            <div class='team-item'>{title}</div>\n"

        team_section = team_section + """
                        </div>
                    </div>

                    <div class="team-card patcher">
                        <div class="team-header">
                            <h3>Patcher Team</h3>
                            <div class="team-badge">{}</div>
                        </div>
                        <div class="team-list">
        """.format(team_data['patcher']['count']) if 'team_section' in locals() else ""

        for title in team_data['patcher']['items'][:10]:
            team_section = team_section + f"                            <div class='team-item'>{title}</div>\n"

        team_section = team_section + """
                        </div>
                    </div>

                    <div class="team-card red-team">
                        <div class="team-header">
                            <h3>Red Team</h3>
                            <div class="team-badge">{}</div>
                        </div>
                        <div class="team-list">
        """.format(team_data['red_team']['count']) if 'team_section' in locals() else ""

        for title in team_data['red_team']['items'][:10]:
            team_section = team_section + f"                            <div class='team-item'>{title}</div>\n"

        team_section = team_section + """
                        </div>
                    </div>

                    <div class="team-card gold-team">
                        <div class="team-header">
                            <h3>Gold Team</h3>
                            <div class="team-badge">{}</div>
                        </div>
                        <div class="team-list">
        """.format(team_data['gold_team']['count']) if 'team_section' in locals() else ""

        for title in team_data['gold_team']['items'][:10]:
            team_section = team_section + f"                            <div class='team-item'>{title}</div>\n"

        team_section = team_section + """
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """ if 'team_section' in locals() else ""

        return locals().get('team_section', '<section id="teams" class="tab-content"><h2>Team Assignments</h2><p>No team data available</p></section>')

    def _generate_logs_section(self, state) -> str:
        """Generate logs section."""
        logs = getattr(state, 'logs', [])
        recent_logs = logs[-50:] if logs else []

        log_items = ""
        for log in recent_logs:
            timestamp = getattr(log, 'timestamp', datetime.now())
            level = getattr(log, 'level', 'INFO')
            message = getattr(log, 'message', '')
            log_items += f"""
                    <div class="log-entry log-{level.lower()}">
                        <span class="log-timestamp">{timestamp.strftime('%H:%M:%S')}</span>
                        <span class="log-level">{level}</span>
                        <span class="log-message">{message}</span>
                    </div>
            """

        return f"""
        <section id="logs" class="tab-content">
            <h2>Activity Logs</h2>

            <div class="logs-controls">
                <button id="refreshLogs" class="btn btn-secondary">Refresh</button>
                <button id="exportLogs" class="btn btn-secondary">Export Logs</button>
                <select id="logLevelFilter" class="form-select">
                    <option value="all">All Levels</option>
                    <option value="DEBUG">DEBUG</option>
                    <option value="INFO">INFO</option>
                    <option value="WARNING">WARNING</option>
                    <option value="ERROR">ERROR</option>
                </select>
            </div>

            <div class="logs-container">
                {log_items if log_items else '<p class="no-logs">No logs available</p>'}
            </div>
        </section>
        """

    def _generate_export_controls(self) -> str:
        """Generate export functionality."""
        formats = ", ".join(self.config.export_formats)

        return f"""
        <div class="export-panel">
            <h3>Export Dashboard</h3>
            <div class="export-options">
                <button id="exportPNG" class="btn btn-primary">
                    <i class="icon-image"></i> Export as PNG
                </button>
                <button id="exportSVG" class="btn btn-primary">
                    <i class="icon-vector"></i> Export as SVG
                </button>
                <button id="exportPDF" class="btn btn-primary">
                    <i class="icon-pdf"></i> Export as PDF
                </button>
                <button id="exportHTML" class="btn btn-primary">
                    <i class="icon-code"></i> Export as HTML
                </button>
            </div>
        </div>
        """

    def _wrap_html(self, title: str, body_content: str) -> str:
        """Wrap content in complete HTML document with CSS and JavaScript."""
        theme_class = f"theme-{self.config.theme}"
        refresh_meta = f'<meta http-equiv="refresh" content="{self.config.refresh_interval}">' if self.config.auto_refresh else ''

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {refresh_meta}
    <title>{title}</title>

    <!-- External libraries -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>

    <style>
        /* Base styles */
        :root {{
            --primary-color: #3498db;
            --secondary-color: #2c3e50;
            --success-color: #27ae60;
            --danger-color: #e74c3c;
            --warning-color: #f39c12;
            --info-color: #16a085;
            --light-color: #ecf0f1;
            --dark-color: #2c3e50;
            --border-color: #ddd;
            --shadow: 0 2px 10px rgba(0,0,0,0.1);
            --radius: 8px;
        }}

        .theme-dark {{
            --primary-color: #5dade2;
            --secondary-color: #34495e;
            --light-color: #2c3e50;
            --dark-color: #ecf0f1;
            --border-color: #555;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: var(--dark-color);
            line-height: 1.6;
        }}

        .theme-dark body {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: var(--dark-color);
        }}

        /* Header */
        .dashboard-header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 20px;
            box-shadow: var(--shadow);
        }}

        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-title h1 {{
            font-size: 28px;
            margin-bottom: 5px;
        }}

        .workflow-id {{
            opacity: 0.9;
            font-size: 14px;
        }}

        .header-status {{
            display: flex;
            gap: 20px;
            align-items: center;
        }}

        .status-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        /* Navigation */
        .dashboard-nav {{
            background: white;
            padding: 15px;
            box-shadow: var(--shadow);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .theme-dark .dashboard-nav {{
            background: var(--light-color);
        }}

        .nav-tab {{
            background: none;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: var(--dark-color);
            border-radius: var(--radius);
            transition: all 0.3s;
        }}

        .nav-tab:hover {{
            background: var(--light-color);
        }}

        .nav-tab.active {{
            background: var(--primary-color);
            color: white;
        }}

        /* Content */
        .tab-content {{
            display: none;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background: white;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }}

        .theme-dark .tab-content {{
            background: var(--light-color);
            color: var(--dark-color);
        }}

        .tab-content.active {{
            display: block;
        }}

        .tab-content h2 {{
            margin-bottom: 20px;
            color: var(--secondary-color);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 10px;
        }}

        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metric-card {{
            padding: 20px;
            border-radius: var(--radius);
            text-align: center;
            box-shadow: var(--shadow);
            transition: transform 0.3s;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
        }}

        .metric-card.primary {{
            background: linear-gradient(135deg, var(--primary-color), #2980b9);
            color: white;
        }}

        .metric-card.success {{
            background: linear-gradient(135deg, var(--success-color), #229954);
            color: white;
        }}

        .metric-card.info {{
            background: linear-gradient(135deg, var(--info-color), #138d75);
            color: white;
        }}

        .metric-card.warning {{
            background: linear-gradient(135deg, var(--warning-color), #e67e22);
            color: white;
        }}

        .metric-card.danger {{
            background: linear-gradient(135deg, var(--danger-color), #c0392b);
            color: white;
        }}

        .metric-card.secondary {{
            background: linear-gradient(135deg, var(--secondary-color), #1a252f);
            color: white;
        }}

        .metric-icon {{
            font-size: 36px;
            margin-bottom: 10px;
        }}

        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .metric-bar {{
            height: 6px;
            background: rgba(255,255,255,0.3);
            border-radius: 3px;
            margin-top: 10px;
            overflow: hidden;
        }}

        .metric-fill {{
            height: 100%;
            background: white;
            border-radius: 3px;
            transition: width 0.5s ease;
        }}

        /* Graph Container */
        .graph-container {{
            height: 600px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            background: var(--light-color);
        }}

        /* Timeline */
        .timeline-container {{
            height: 400px;
            margin-bottom: 20px;
        }}

        .timeline-list {{
            max-height: 500px;
            overflow-y: auto;
        }}

        .timeline-item {{
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid var(--primary-color);
            background: var(--light-color);
            border-radius: var(--radius);
        }}

        .timeline-item.solved {{ border-left-color: var(--success-color); }}
        .timeline-item.in-progress {{ border-left-color: var(--info-color); }}
        .timeline-item.failed {{ border-left-color: var(--danger-color); }}
        .timeline-item.blocked {{ border-left-color: var(--warning-color); }}

        /* Quality Bars */
        .quality-item {{
            margin-bottom: 15px;
        }}

        .quality-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 500;
        }}

        .quality-score {{
            color: var(--primary-color);
            font-weight: bold;
        }}

        .quality-bar {{
            height: 25px;
            background: var(--light-color);
            border-radius: 12px;
            overflow: hidden;
        }}

        .quality-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--success-color));
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            transition: width 0.5s ease;
        }}

        /* Team Cards */
        .team-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .team-card {{
            padding: 20px;
            border-radius: var(--radius);
            box-shadow: var(--shadow);
        }}

        .team-card.solver {{ border-top: 4px solid var(--primary-color); }}
        .team-card.patcher {{ border-top: 4px solid var(--success-color); }}
        .team-card.red-team {{ border-top: 4px solid var(--danger-color); }}
        .team-card.gold-team {{ border-top: 4px solid var(--warning-color); }}

        .team-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}

        .team-badge {{
            background: var(--primary-color);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }}

        .team-item {{
            padding: 8px;
            background: var(--light-color);
            margin-bottom: 5px;
            border-radius: 4px;
            font-size: 14px;
        }}

        /* Buttons */
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: var(--radius);
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
        }}

        .btn-primary {{
            background: var(--primary-color);
            color: white;
        }}

        .btn-secondary {{
            background: var(--light-color);
            color: var(--dark-color);
        }}

        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        /* Logs */
        .logs-container {{
            max-height: 500px;
            overflow-y: auto;
            padding: 15px;
            background: var(--light-color);
            border-radius: var(--radius);
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }}

        .log-entry {{
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 4px;
        }}

        .log-entry:hover {{
            background: rgba(0,0,0,0.05);
        }}

        .log-timestamp {{
            color: var(--info-color);
            margin-right: 10px;
        }}

        .log-level {{
            font-weight: bold;
            margin-right: 10px;
        }}

        .log-info .log-level {{ color: var(--primary-color); }}
        .log-warning .log-level {{ color: var(--warning-color); }}
        .log-error .log-level {{ color: var(--danger-color); }}
        .log-debug .log-level {{ color: var(--secondary-color); }}

        /* Responsive */
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}

            .header-content {{
                flex-direction: column;
                text-align: center;
            }}

            .nav-tab {{
                padding: 8px 12px;
                font-size: 12px;
            }}
        }}
    </style>
</head>
<body class="{theme_class}">
    {body_content}

    <script>
        // Tab navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {{
            tab.addEventListener('click', function() {{
                // Remove active from all tabs
                document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                // Add active to clicked tab
                this.classList.add('active');
                const tabId = this.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            }});
        }});

        // Dependency graph initialization
        function initDependencyGraph(nodes, edges) {{
            const container = document.getElementById('dependencyGraph');
            if (!container) return;

            const data = {{
                nodes: new vis.DataSet(nodes),
                edges: new vis.DataSet(edges)
            }};

            const options = {{
                nodes: {{
                    shape: 'box',
                    margin: 10,
                    font: {{ size: 14 }}
                }},
                edges: {{
                    arrows: 'to',
                    smooth: {{ type: 'cubicBezier' }}
                }},
                layout: {{
                    hierarchical: {{
                        enabled: true,
                        direction: 'UD',
                        sortMethod: 'directed'
                    }}
                }},
                physics: {{ enabled: false }}
            }};

            new vis.Network(container, data, options);
        }}

        // Timeline initialization
        function initTimeline(data) {{
            const ctx = document.getElementById('timelineChart');
            if (!ctx) return;

            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: data.map(d => d.title.substring(0, 30)),
                    datasets: [{{
                        label: 'Estimated Hours',
                        data: data.map(d => d.estimated_hours),
                        backgroundColor: data.map(d => {{
                            switch(d.status) {{
                                case 'solved': return '#27ae60';
                                case 'in_progress': return '#3498db';
                                case 'failed': return '#e74c3c';
                                case 'blocked': return '#f39c12';
                                default: return '#95a5a6';
                            }}
                        }})
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }}
                    }}
                }}
            }});
        }}

        // Quality chart initialization
        function initQualityChart(data) {{
            const ctx = document.getElementById('qualityRadar');
            if (!ctx) return;

            new Chart(ctx, {{
                type: 'radar',
                data: {{
                    labels: Object.keys(data).map(k => k.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                    datasets: [{{
                        label: 'Quality Score',
                        data: Object.values(data).map(v => v * 100),
                        backgroundColor: 'rgba(52, 152, 219, 0.2)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        pointBackgroundColor: 'rgba(52, 152, 219, 1)'
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        r: {{
                            beginAtZero: true,
                            max: 100
                        }}
                    }}
                }}
            }});
        }}

        // Auto-refresh functionality
        {f'setInterval(() => {{ location.reload(); }}, {self.config.refresh_interval * 1000});' if self.config.auto_refresh else ''}
    </script>
</body>
</html>
        """


class RealTimeDashboard(DecompositionDashboard):
    """
    Extended dashboard with real-time updates via WebSocket or polling.

    Supports live updates without page refresh.
    """

    def __init__(self, config: DashboardConfig = None):
        super().__init__(config)
        self.update_callbacks = []

    def register_update_callback(self, callback):
        """Register callback for real-time updates."""
        self.update_callbacks.append(callback)

    async def stream_updates(self, workflow_id: str):
        """Stream real-time updates to connected clients."""
        while True:
            # Get latest state
            # Notify callbacks
            for callback in self.update_callbacks:
                await callback(workflow_id)

            await asyncio.sleep(self.config.refresh_interval)
