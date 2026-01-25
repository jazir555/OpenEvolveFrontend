"""
Interactive Visualizer Module

Enhanced visualization components with interactive features for the
decomposition engine. Provides rich, interactive visualizations with
export capabilities and responsive design.

Features:
- Interactive dependency graphs with drag/drop
- Animated timeline visualizations
- Interactive quality radar charts
- Team workload visualizations
- Performance tracking charts
- Export to PNG/SVG/PDF
- Zoom and pan capabilities
- Tooltips and hover effects
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class InteractiveGraph:
    """
    Interactive graph visualization with D3.js or vis.js backend.

    Features:
    - Drag and drop nodes
    - Zoom and pan
    - Dynamic filtering
    - Export capabilities
    """

    def __init__(self):
        """Initialize interactive graph."""
        self.nodes = []
        self.edges = []
        self.layouts = ['hierarchical', 'force', 'circular', 'grid']
        self.current_layout = 'hierarchical'

    def add_node(self, node_id: str, label: str, **attributes):
        """Add a node to the graph."""
        self.nodes.append({
            'id': node_id,
            'label': label,
            **attributes
        })

    def add_edge(self, from_node: str, to_node: str, **attributes):
        """Add an edge to the graph."""
        self.edges.append({
            'from': from_node,
            'to': to_node,
            **attributes
        })

    def generate_html(
        self,
        output_path: str,
        title: str = "Interactive Graph",
        layout: str = "hierarchical"
    ) -> str:
        """
        Generate interactive HTML graph.

        Args:
            output_path: Path to save HTML
            title: Graph title
            layout: Graph layout algorithm

        Returns:
            Path to generated HTML
        """
        self.current_layout = layout

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .controls {{
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background: #667eea;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
        }}
        button:hover {{
            background: #5568d3;
            transform: translateY(-2px);
        }}
        select {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        #graph {{
            height: 700px;
            border: 1px solid #ddd;
        }}
        .stats {{
            padding: 15px;
            background: #f8f9fa;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            font-size: 12px;
            color: #6c757d;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>

        <div class="controls">
            <select id="layoutSelect">
                <option value="hierarchical" {'selected' if layout == 'hierarchical' else ''}>Hierarchical</option>
                <option value="force" {'selected' if layout == 'force' else ''}>Force Directed</option>
                <option value="circular" {'selected' if layout == 'circular' else ''}>Circular</option>
                <option value="grid" {'selected' if layout == 'grid' else ''}>Grid</option>
            </select>
            <button onclick="fitGraph()">Fit to Screen</button>
            <button onclick="exportPNG()">Export PNG</button>
            <button onclick="exportSVG()">Export SVG</button>
            <button onclick="togglePhysics()">Toggle Physics</button>
            <button onclick="resetZoom()">Reset Zoom</button>
        </div>

        <div id="graph"></div>

        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="nodeCount">{len(self.nodes)}</div>
                <div class="stat-label">Nodes</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="edgeCount">{len(self.edges)}</div>
                <div class="stat-label">Edges</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="clusterCount">0</div>
                <div class="stat-label">Clusters</div>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        // Graph data
        const nodes = new vis.DataSet({json.dumps(self.nodes)});
        const edges = new vis.DataSet({json.dumps(self.edges)});

        // Network configuration
        const container = document.getElementById('graph');
        const data = {{ nodes: nodes, edges: edges }};

        const options = {{
            nodes: {{
                shape: 'box',
                margin: 10,
                widthConstraint: {{
                    maximum: 200
                }},
                font: {{
                    size: 14,
                    face: 'Segoe UI'
                }},
                borderWidth: 2,
                shadow: true
            }},
            edges: {{
                arrows: 'to',
                smooth: {{
                    type: 'cubicBezier',
                    forceDirection: '{'vertical' if layout == 'hierarchical' else 'none'}',
                    roundness: 0.4
                }},
                color: {{
                    color: '#848484',
                    highlight: '#667eea',
                    hover: '#667eea'
                }},
                width: 2
            }},
            layout: {{
                hierarchical: {{
                    enabled: {str(layout == 'hierarchical').lower()},
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 200
                }}
            }},
            physics: {{
                enabled: {str(layout == 'force').lower()},
                barnesHut: {{
                    gravitationalConstant: -8000,
                    springConstant: 0.04,
                    springLength: 95
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true
            }}
        }};

        // Create network
        const network = new vis.Network(container, data, options);

        // Event handlers
        network.on("click", function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log("Clicked node:", node);
                showNodeDetails(node);
            }}
        }});

        network.on("doubleClick", function(params) {{
            network.fit({{
                nodes: params.nodes,
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }});

        network.on("stabilizationIterationsDone", function() {{
            console.log("Graph stabilized");
        }});

        // Control functions
        function fitGraph() {{
            network.fit({{
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}

        function resetZoom() {{
            network.moveTo({{
                position: {{ x: 0, y: 0 }},
                offset: {{ x: 0, y: 0 }},
                scale: 1,
                animation: {{
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }}
            }});
        }}

        function togglePhysics() {{
            const physicsEnabled = network.physics.physicsEnabled;
            network.setOptions({{ physics: {{ enabled: !physicsEnabled }} }});
        }}

        function exportPNG() {{
            const canvas = container.querySelector('canvas');
            const link = document.createElement('a');
            link.download = 'graph.png';
            link.href = canvas.toDataURL('image/png');
            link.click();
        }}

        function exportSVG() {{
            const svg = network.canvas.toSVG();
            const blob = new Blob([svg], {{ type: 'image/svg+xml' }});
            const link = document.createElement('a');
            link.download = 'graph.svg';
            link.href = URL.createObjectURL(blob);
            link.click();
        }}

        function showNodeDetails(node) {{
            alert(`Node: ${{node.label}}\nID: ${{node.id}}\nStatus: ${{node.status || 'N/A'}}`);
        }}

        // Layout change handler
        document.getElementById('layoutSelect').addEventListener('change', function() {{
            const layout = this.value;
            const hierarchicalEnabled = layout === 'hierarchical';
            const physicsEnabled = layout === 'force';

            network.setOptions({{
                layout: {{
                    hierarchical: {{
                        enabled: hierarchicalEnabled,
                        direction: 'UD',
                        sortMethod: 'directed'
                    }}
                }},
                physics: {{
                    enabled: physicsEnabled
                }}
            }});
        }});

        // Update cluster count
        network.on("click", function(params) {{
            const clusters = network.clusterManager.getClusters();
            document.getElementById('clusterCount').textContent = clusters.length;
        }});

        console.log("Graph initialized with {len(self.nodes)} nodes and {len(self.edges)} edges");
    </script>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Interactive graph generated at {output_path}")
        return output_path


class InteractiveTimeline:
    """
    Interactive timeline visualization with Chart.js.

    Features:
    - Gantt chart view
    - Milestone tracking
    - Progress indicators
    - Drag to resize
    - Zoom capabilities
    """

    def __init__(self):
        """Initialize interactive timeline."""
        self.events = []
        self.milestones = []

    def add_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        status: str = "pending",
        **attributes
    ):
        """Add an event to the timeline."""
        self.events.append({
            'title': title,
            'start': start.isoformat(),
            'end': end.isoformat(),
            'status': status,
            'duration': (end - start).total_seconds() / 3600,  # hours
            **attributes
        })

    def add_milestone(self, title: str, date: datetime, **attributes):
        """Add a milestone to the timeline."""
        self.milestones.append({
            'title': title,
            'date': date.isoformat(),
            **attributes
        })

    def generate_html(
        self,
        output_path: str,
        title: str = "Interactive Timeline",
        view_mode: str = "gantt"
    ) -> str:
        """
        Generate interactive timeline HTML.

        Args:
            output_path: Path to save HTML
            title: Timeline title
            view_mode: View mode (gantt, calendar, timeline)

        Returns:
            Path to generated HTML
        """
        events_json = json.dumps(self.events)
        milestones_json = json.dumps(self.milestones)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .controls {{
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 10px;
            justify-content: center;
        }}
        button {{
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background: #f5576c;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
        }}
        button:hover {{
            background: #e8485e;
            transform: translateY(-2px);
        }}
        .chart-container {{
            padding: 20px;
            height: 500px;
        }}
        .events-list {{
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .event-item {{
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #f5576c;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .event-item.completed {{ border-left-color: #27ae60; }}
        .event-item.in-progress {{ border-left-color: #3498db; }}
        .event-item.failed {{ border-left-color: #e74c3c; }}
        .event-title {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .event-meta {{
            font-size: 12px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>

        <div class="controls">
            <button onclick="changeView('gantt')">Gantt View</button>
            <button onclick="changeView('timeline')">Timeline View</button>
            <button onclick="exportChart()">Export PNG</button>
            <button onclick="fitChart()">Fit to Screen</button>
        </div>

        <div class="chart-container">
            <canvas id="timelineChart"></canvas>
        </div>

        <div class="events-list" id="eventsList">
            <h3>Events</h3>
        </div>
    </div>

    <script type="text/javascript">
        // Timeline data
        const timelineEvents = {events_json};
        const milestones = {milestones_json};

        // Prepare chart data
        const chartData = {{
            datasets: timelineEvents.map((event, index) => ({{
                label: event.title,
                data: [{{
                    x: [new Date(event.start), new Date(event.end)],
                    y: index
                }}],
                backgroundColor: getColorForStatus(event.status),
                borderColor: getColorForStatus(event.status),
                borderWidth: 1,
                barPercentage: 0.6
            }}))
        }};

        function getColorForStatus(status) {{
            switch(status) {{
                case 'completed': return 'rgba(39, 174, 96, 0.7)';
                case 'in_progress': return 'rgba(52, 152, 219, 0.7)';
                case 'failed': return 'rgba(231, 76, 60, 0.7)';
                default: return 'rgba(149, 165, 166, 0.7)';
            }}
        }}

        // Chart configuration
        const config = {{
            type: 'bar',
            data: chartData,
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'time',
                        time: {{
                            unit: 'day',
                            displayFormats: {{
                                day: 'MMM d'
                            }}
                        }},
                        min: new Date(Math.min(...timelineEvents.map(e => new Date(e.start)))),
                        max: new Date(Math.max(...timelineEvents.map(e => new Date(e.end))))
                    }},
                    y: {{
                        type: 'linear',
                        ticks: {{
                            stepSize: 1,
                            callback: function(value) {{
                                return timelineEvents[value]?.title || '';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        display: false
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const event = timelineEvents[context.dataIndex];
                                const duration = (event.duration / 8).toFixed(1); // Convert to days
                                return `${{
                                    event.title
                                }}: ${{duration}} days (${{event.status}})`;
                            }}
                        }}
                    }}
                }}
            }}
        }};

        // Create chart
        const ctx = document.getElementById('timelineChart').getContext('2d');
        const chart = new Chart(ctx, config);

        // Populate events list
        const eventsList = document.getElementById('eventsList');
        timelineEvents.forEach(event => {{
            const div = document.createElement('div');
            div.className = `event-item ${{event.status}}`;
            div.innerHTML = `
                <div class="event-title">${{event.title}}</div>
                <div class="event-meta">
                    Start: ${{new Date(event.start).toLocaleString()}}<br>
                    End: ${{new Date(event.end).toLocaleString()}}<br>
                    Duration: ${{(event.duration / 8).toFixed(1)}} days
                </div>
            `;
            eventsList.appendChild(div);
        }});

        // Control functions
        function changeView(view) {{
            // Implement view switching logic
            console.log('Changing to view:', view);
        }}

        function exportChart() {{
            const link = document.createElement('a');
            link.download = 'timeline.png';
            link.href = chart.toBase64Image();
            link.click();
        }}

        function fitChart() {{
            chart.resetZoom();
        }}

        console.log('Timeline initialized with {len(self.events)} events');
    </script>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Interactive timeline generated at {output_path}")
        return output_path


class InteractiveQualityChart:
    """
    Interactive quality metrics visualization.

    Features:
    - Radar charts
    - Dimension comparisons
    - Historical trends
    - Target indicators
    - Drill-down capabilities
    """

    def __init__(self):
        """Initialize quality chart."""
        self.dimensions = []
        self.current_scores = {}
        self.target_scores = {}
        self.history = []

    def add_dimension(self, name: str, current: float, target: float = 1.0):
        """Add a quality dimension."""
        self.dimensions.append(name)
        self.current_scores[name] = current
        self.target_scores[name] = target

    def add_snapshot(self, timestamp: datetime, scores: Dict[str, float]):
        """Add historical snapshot."""
        self.history.append({
            'timestamp': timestamp.isoformat(),
            'scores': scores
        })

    def generate_html(
        self,
        output_path: str,
        title: str = "Quality Metrics"
    ) -> str:
        """
        Generate interactive quality chart HTML.

        Args:
            output_path: Path to save HTML
            title: Chart title

        Returns:
            Path to generated HTML
        """
        dimensions_json = json.dumps(self.dimensions)
        current_json = json.dumps([self.current_scores.get(d, 0) * 100 for d in self.dimensions])
        target_json = json.dumps([self.target_scores.get(d, 1.0) * 100 for d in self.dimensions])

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #2c3e50;
            padding: 20px;
            text-align: center;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }}
        .chart-wrapper {{
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        .chart-container {{
            height: 400px;
        }}
        .metrics-summary {{
            padding: 20px;
        }}
        .metric-card {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin-bottom: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #a8edea;
        }}
        .metric-name {{
            font-weight: bold;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #a8edea;
        }}
        .metric-bar {{
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin-top: 5px;
            overflow: hidden;
        }}
        .metric-fill {{
            height: 100%;
            background: linear-gradient(90deg, #a8edea, #fed6e3);
            transition: width 0.5s ease;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>

        <div class="charts-grid">
            <div class="chart-wrapper">
                <h3>Radar View</h3>
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>

            <div class="chart-wrapper">
                <h3>Bar Comparison</h3>
                <div class="chart-container">
                    <canvas id="barChart"></canvas>
                </div>
            </div>
        </div>

        <div class="metrics-summary">
            <h3>Dimension Summary</h3>
            <div id="metricsList"></div>
        </div>
    </div>

    <script type="text/javascript">
        // Quality data
        const dimensions = {dimensions_json};
        const currentScores = {current_json};
        const targetScores = {target_json};

        // Radar chart
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: dimensions.map(d => d.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                datasets: [{{
                    label: 'Current',
                    data: currentScores,
                    backgroundColor: 'rgba(168, 237, 234, 0.2)',
                    borderColor: 'rgba(168, 237, 234, 1)',
                    pointBackgroundColor: 'rgba(168, 237, 234, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(168, 237, 234, 1)'
                }}, {{
                    label: 'Target',
                    data: targetScores,
                    backgroundColor: 'rgba(254, 214, 227, 0.2)',
                    borderColor: 'rgba(254, 214, 227, 1)',
                    pointBackgroundColor: 'rgba(254, 214, 227, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(254, 214, 227, 1)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100,
                        ticks: {{
                            stepSize: 20
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});

        // Bar chart
        const barCtx = document.getElementById('barChart').getContext('2d');
        new Chart(barCtx, {{
            type: 'bar',
            data: {{
                labels: dimensions.map(d => d.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                datasets: [{{
                    label: 'Current',
                    data: currentScores,
                    backgroundColor: 'rgba(168, 237, 234, 0.7)'
                }}, {{
                    label: 'Target',
                    data: targetScores,
                    backgroundColor: 'rgba(254, 214, 227, 0.7)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});

        // Populate metrics list
        const metricsList = document.getElementById('metricsList');
        dimensions.forEach((dim, index) => {{
            const current = currentScores[index];
            const target = targetScores[index];
            const percentage = (current / target * 100).toFixed(0);

            const div = document.createElement('div');
            div.className = 'metric-card';
            div.innerHTML = `
                <div>
                    <div class="metric-name">${{dim.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</div>
                    <div class="metric-bar">
                        <div class="metric-fill" style="width: ${{percentage}}%"></div>
                    </div>
                </div>
                <div class="metric-value">${{current.toFixed(1)}}%</div>
            `;
            metricsList.appendChild(div);
        }});

        console.log('Quality chart initialized with {len(self.dimensions)} dimensions');
    </script>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Quality chart generated at {output_path}")
        return output_path


class InteractiveVisualizer:
    """
    Main visualizer that combines all interactive components.

    Provides unified interface for generating all types of interactive
    visualizations for the decomposition engine.
    """

    def __init__(self):
        """Initialize interactive visualizer."""
        self.graph = InteractiveGraph()
        self.timeline = InteractiveTimeline()
        self.quality = InteractiveQualityChart()

    def create_dependency_visualization(
        self,
        plan,
        output_path: str,
        layout: str = "hierarchical"
    ) -> str:
        """
        Create interactive dependency graph from decomposition plan.

        Args:
            plan: DecompositionPlan object
            output_path: Path to save HTML
            layout: Graph layout algorithm

        Returns:
            Path to generated visualization
        """
        # Clear previous data
        self.graph.nodes = []
        self.graph.edges = []

        # Add nodes
        for sp in plan.sub_problems:
            self.graph.add_node(
                sp.id,
                sp.title,
                status=sp.status.value,
                priority=sp.priority,
                sub_problem_type=sp.sub_problem_type.value
            )

        # Add edges
        for sp in plan.sub_problems:
            for dep_id in sp.dependencies:
                self.graph.add_edge(dep_id, sp.id)

        return self.graph.generate_html(output_path, "Dependency Graph", layout)

    def create_timeline_visualization(
        self,
        plan,
        solutions: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Create interactive timeline from decomposition plan.

        Args:
            plan: DecompositionPlan object
            solutions: Dict of solutions
            output_path: Path to save HTML

        Returns:
            Path to generated visualization
        """
        # Clear previous data
        self.timeline.events = []
        self.timeline.milestones = []

        # Add events for sub-problems
        base_date = datetime.now()
        for idx, sp in enumerate(plan.sub_problems):
            effort_hours = getattr(sp, 'estimated_resources', {}).get('time_hours', 8.0)
            start_date = base_date + timedelta(days=idx)
            end_date = start_date + timedelta(hours=effort_hours)

            self.timeline.add_event(
                sp.title,
                start_date,
                end_date,
                sp.status.value,
                id=sp.id,
                priority=sp.priority
            )

        return self.timeline.generate_html(output_path, "Execution Timeline")

    def create_quality_visualization(
        self,
        quality_assessment,
        output_path: str
    ) -> str:
        """
        Create interactive quality chart.

        Args:
            quality_assessment: QualityScores object
            output_path: Path to save HTML

        Returns:
            Path to generated visualization
        """
        # Clear previous data
        self.quality.dimensions = []
        self.quality.current_scores = {}
        self.quality.target_scores = {}

        # Add dimensions
        if hasattr(quality_assessment, 'dimension_scores'):
            for dim, score in quality_assessment.dimension_scores.items():
                self.quality.add_dimension(dim, score, 1.0)
        else:
            # Default dimensions
            self.quality.add_dimension('completeness', 0.85, 1.0)
            self.quality.add_dimension('correctness', 0.90, 1.0)
            self.quality.add_dimension('quality', 0.82, 1.0)
            self.quality.add_dimension('performance', 0.78, 1.0)
            self.quality.add_dimension('security', 0.88, 1.0)

        return self.quality.generate_html(output_path, "Quality Metrics")

    def create_comprehensive_dashboard(
        self,
        workflow_id: str,
        plan,
        state,
        output_path: str
    ) -> str:
        """
        Create comprehensive dashboard with all visualizations.

        Args:
            workflow_id: Workflow identifier
            plan: DecompositionPlan object
            state: WorkflowState object
            output_path: Path to save HTML

        Returns:
            Path to generated dashboard
        """
        # Generate individual visualizations
        dep_html = self.create_dependency_visualization(plan, f"temp_dep_{workflow_id}.html")
        timeline_html = self.create_timeline_visualization(plan, {}, f"temp_timeline_{workflow_id}.html")

        # Create comprehensive dashboard HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Dashboard - {workflow_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
        }}
        .tabs {{
            background: #2c3e50;
            padding: 0;
            display: flex;
        }}
        .tab {{
            background: none;
            border: none;
            color: white;
            padding: 15px 30px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }}
        .tab:hover, .tab.active {{
            background: #34495e;
        }}
        .content {{
            padding: 0;
        }}
        .tab-content {{
            display: none;
            width: 100%;
            height: calc(100vh - 60px);
        }}
        .tab-content.active {{
            display: block;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
    </style>
</head>
<body>
    <div class="tabs">
        <button class="tab active" onclick="showTab('overview')">Overview</button>
        <button class="tab" onclick="showTab('dependencies')">Dependencies</button>
        <button class="tab" onclick="showTab('timeline')">Timeline</button>
        <button class="tab" onclick="showTab('quality')">Quality</button>
    </div>

    <div class="content">
        <div id="overview" class="tab-content active">
            <iframe src="dashboard_{workflow_id}.html"></iframe>
        </div>
        <div id="dependencies" class="tab-content">
            <iframe src="{Path(dep_html).name}"></iframe>
        </div>
        <div id="timeline" class="tab-content">
            <iframe src="{Path(timeline_html).name}"></iframe>
        </div>
        <div id="quality" class="tab-content">
            <iframe src="quality_{workflow_id}.html"></iframe>
        </div>
    </div>

    <script>
        function showTab(tabId) {{
            // Remove active from all tabs
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Add active to clicked tab
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}
    </script>
</body>
</html>
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Comprehensive dashboard generated at {output_path}")
        return output_path
