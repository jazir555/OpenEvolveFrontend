"""
Visualization Module for OpenEvolve Knowledge Engine

This module provides visualization capabilities for knowledge graphs,
including interactive graph explorer, temporal visualizer, and community visualizer.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path

# Import temporal visualization components
from .temporal_viz import TemporalVisualizer, TimeRange, TemporalVisualizationOptions

logger = logging.getLogger(__name__)

__all__ = [
    'VisualizationOptions',
    'VisualizationResult',
    'ExportHandler',
    'GraphExplorer',
    'TemporalVisualizer',
    'TimeRange',
    'TemporalVisualizationOptions',
    'CommunityVisualizer',
]




@dataclass
class VisualizationOptions:
    """Options for visualization generation."""
    width: int = 1200
    height: int = 800
    layout: str = "force_directed"
    include_labels: bool = True
    include_communities: bool = True
    node_size_attr: str = "degree"
    edge_thickness_attr: str = "weight"


@dataclass
class VisualizationResult:
    """Result of a visualization operation."""
    output_path: str
    node_count: int
    edge_count: int
    community_count: int
    processing_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class ExportHandler:
    """
    Handler for exporting visualizations in various formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the export handler.
        
        Args:
            config: Configuration for export operations
        """
        self.config = config or {
            "output_dir": "./visualizations",
            "supported_formats": ["html", "json", "svg", "png"],
            "default_format": "html"
        }
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.config.get("output_dir", "./visualizations"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info({
            "msg": "ExportHandler initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def export_to_file(
        self,
        visualization_data: str,
        output_path: str,
        format: Optional[str] = None
    ) -> bool:
        """
        Export visualization data to a file.
        
        Args:
            visualization_data: Visualization data to export
            output_path: Path for output file
            format: Format to export as (auto-detected from extension if None)
            
        Returns:
            True if export successful
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting visualization export",
            "output_path": output_path,
            "format": format,
            "data_size": len(visualization_data),
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Determine format from extension if not provided
            if not format:
                format = Path(output_path).suffix.lower().lstrip('.')
            
            # Validate format
            supported_formats = self.config.get("supported_formats", ["html", "json", "svg", "png"])
            if format not in supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported: {supported_formats}")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(visualization_data)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Visualization export completed",
                "output_path": output_path,
                "format": format,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return True
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Visualization export failed",
                "output_path": output_path,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return False


class GraphExplorer:
    """
    Interactive graph explorer for knowledge graphs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the graph explorer.
        
        Args:
            config: Configuration for graph explorer
        """
        self.config = config or {
            "default_width": 1200,
            "default_height": 800,
            "default_layout": "force_directed",
            "node_radius_min": 5,
            "node_radius_max": 20
        }
        
        logger.info({
            "msg": "GraphExplorer initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def visualize(
        self,
        triples: List[Tuple[str, str, str]],
        correlation_id: Optional[str] = None,
        options: Optional[VisualizationOptions] = None
    ) -> str:
        """
        Generate interactive graph visualization.
        
        Args:
            triples: List of (subject, predicate, object) triples
            correlation_id: Correlation ID for tracking
            options: Visualization options
            
        Returns:
            HTML string with interactive visualization
        """
        correlation_id = correlation_id or f"graph_explore_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting graph explorer visualization",
            "triple_count": len(triples),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Create visualization data structure
            nodes_set = set()
            links = []
            
            for subj, pred, obj in triples:
                nodes_set.add(subj)
                nodes_set.add(obj)
                links.append({
                    "source": subj,
                    "target": obj,
                    "relationship": pred
                })
            
            nodes = []
            for i, node in enumerate(nodes_set):
                nodes.append({
                    "id": node,
                    "label": node,
                    "group": hash(node) % 10,  # Simple grouping
                    "degree": sum(1 for link in links if link["source"] == node or link["target"] == node)
                })
            
            # Use default options if none provided
            if not options:
                options = VisualizationOptions()
            
            # Generate HTML with embedded D3.js visualization
            html_content = self._generate_d3_visualization_html(
                nodes=nodes,
                links=links,
                options=options
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Graph explorer visualization completed",
                "correlation_id": correlation_id,
                "node_count": len(nodes),
                "edge_count": len(links),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return html_content
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Graph explorer visualization failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            raise
    
    def _generate_d3_visualization_html(
        self,
        nodes: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        options: VisualizationOptions
    ) -> str:
        """
        Generate HTML with D3.js visualization code.
        
        Args:
            nodes: List of node dictionaries
            links: List of link dictionaries
            options: Visualization options
            
        Returns:
            HTML string with visualization
        """
        # Create the visualization data
        viz_data = {
            "nodes": nodes,
            "links": links,
            "options": {
                "width": options.width,
                "height": options.height,
                "layout": options.layout,
                "include_labels": options.include_labels,
                "include_communities": options.include_communities
            }
        }
        
        # Generate HTML template with embedded D3.js
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Knowledge Graph Explorer</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 10px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        #graph-container {{
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .node-label {{
            font-size: 12px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h2>Knowledge Graph Explorer</h2>
    <div id="graph-container"></div>
    
    <script>
        const vizData = {json.dumps(viz_data)};
        
        // Set up dimensions
        const width = vizData.options.width || 800;
        const height = vizData.options.height || 600;
        
        // Create SVG
        const svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create simulation
        const simulation = d3.forceSimulation(vizData.nodes)
            .force("link", d3.forceLink(vizData.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        // Add links
        const link = svg.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(vizData.links)
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("stroke-width", 1);
        
        // Add nodes
        const node = svg.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(vizData.nodes)
            .enter()
            .append("circle")
            .attr("class", "node")
            .attr("r", 8)
            .attr("fill", d => d.group ? ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][d.group % 10] : '#1f77b4')
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended)
            );
        
        // Add labels if requested
        if (vizData.options.include_labels) {{
            const label = svg.append("g")
                .attr("class", "labels")
                .selectAll("text")
                .data(vizData.nodes)
                .enter()
                .append("text")
                .attr("class", "node-label")
                .text(d => d.label)
                .attr("dx", 12)
                .attr("dy", ".35em");
        }}
        
        // Update positions on tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            if (vizData.options.include_labels) {{
                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }}
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
        """
        
        return html_template


class TemporalVisualizer:
    """
    Temporal visualizer for knowledge graphs showing evolution over time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the temporal visualizer.
        
        Args:
            config: Configuration for temporal visualizer
        """
        self.config = config or {
            "default_width": 1200,
            "default_height": 800,
            "time_resolution": "day",  # day, week, month, year
            "show_timeline": True
        }
        
        logger.info({
            "msg": "TemporalVisualizer initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def visualize(
        self,
        temporal_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        options: Optional[VisualizationOptions] = None
    ) -> str:
        """
        Generate temporal visualization of knowledge graph evolution.
        
        Args:
            temporal_data: Temporal data with time-indexed knowledge
            correlation_id: Correlation ID for tracking
            options: Visualization options
            
        Returns:
            HTML string with temporal visualization
        """
        correlation_id = correlation_id or f"temporal_viz_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting temporal visualization",
            "data_points": len(temporal_data.get("timestamps", [])) if isinstance(temporal_data, dict) else 0,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Use default options if none provided
            if not options:
                options = VisualizationOptions()
            
            # Generate HTML with embedded D3.js visualization
            html_content = self._generate_temporal_visualization_html(
                temporal_data=temporal_data,
                options=options
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Temporal visualization completed",
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return html_content
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Temporal visualization failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            raise
    
    def _generate_temporal_visualization_html(
        self,
        temporal_data: Dict[str, Any],
        options: VisualizationOptions
    ) -> str:
        """
        Generate HTML with temporal D3.js visualization code.
        
        Args:
            temporal_data: Temporal data to visualize
            options: Visualization options
            
        Returns:
            HTML string with visualization
        """
        # Generate HTML template with embedded D3.js for temporal visualization
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Temporal Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 10px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        #graph-container, #timeline-container {{
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
            margin-bottom: 10px;
        }}
        #timeline-container {{
            height: 150px;
        }}
        .axis {{
            font: 12px sans-serif;
        }}
        .axis path,
        .axis line {{
            fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        }}
    </style>
</head>
<body>
    <h2>Temporal Knowledge Graph Visualization</h2>
    <div id="timeline-container"></div>
    <div id="graph-container"></div>
    
    <script>
        // Sample temporal data - in real implementation, this would come from temporal_data
        const timePoints = [
            {{"time": "2023-01-01", "nodes": 10, "edges": 15}},
            {{"time": "2023-04-01", "nodes": 15, "edges": 25}},
            {{"time": "2023-07-01", "nodes": 20, "edges": 35}},
            {{"time": "2023-10-01", "nodes": 25, "edges": 45}},
            {{"time": "2024-01-01", "nodes": 30, "edges": 55}}
        ];
        
        // Set up dimensions
        const width = {options.width};
        const height = {options.height};
        const timelineHeight = 100;
        
        // Parse dates
        const parseDate = d3.timeParse("%Y-%m-%d");
        timePoints.forEach(d => d.date = parseDate(d.time));
        
        // Timeline SVG
        const timelineSvg = d3.select("#timeline-container")
            .append("svg")
            .attr("width", width)
            .attr("height", timelineHeight);
        
        // Set up scales for timeline
        const xScaleTimeline = d3.scaleTime()
            .domain(d3.extent(timePoints, d => d.date))
            .range([50, width - 50]);
        
        const yScaleTimeline = d3.scaleLinear()
            .domain([0, d3.max(timePoints, d => Math.max(d.nodes, d.edges))])
            .range([timelineHeight - 30, 10]);
        
        // Add timeline axis
        timelineSvg.append("g")
            .attr("class", "axis")
            .attr("transform", "translate(0," + (timelineHeight - 20) + ")")
            .call(d3.axisBottom(xScaleTimeline));
        
        // Add timeline line for nodes
        const lineNodes = d3.line()
            .x(d => xScaleTimeline(d.date))
            .y(d => yScaleTimeline(d.nodes));
        
        timelineSvg.append("path")
            .datum(timePoints)
            .attr("class", "line-nodes")
            .attr("fill", "none")
            .attr("stroke", "steelblue")
            .attr("stroke-width", 2)
            .attr("d", lineNodes);
        
        // Add timeline line for edges
        const lineEdges = d3.line()
            .x(d => xScaleTimeline(d.date))
            .y(d => yScaleTimeline(d.edges));
        
        timelineSvg.append("path")
            .datum(timePoints)
            .attr("class", "line-edges")
            .attr("fill", "none")
            .attr("stroke", "red")
            .attr("stroke-width", 2)
            .attr("d", lineEdges);
        
        // Main graph SVG
        const svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // For demonstration, create a simple static graph
        // In real implementation, this would show the graph at a selected time point
        const nodes = [
            {{"id": "A", "x": width * 0.2, "y": height * 0.3}},
            {{"id": "B", "x": width * 0.5, "y": height * 0.3}},
            {{"id": "C", "x": width * 0.8, "y": height * 0.3}},
            {{"id": "D", "x": width * 0.35, "y": height * 0.7}},
            {{"id": "E", "x": width * 0.65, "y": height * 0.7}}
        ];
        
        const links = [
            {{"source": "A", "target": "B"}},
            {{"source": "B", "target": "C"}},
            {{"source": "A", "target": "D"}},
            {{"source": "B", "target": "D"}},
            {{"source": "C", "target": "E"}},
            {{"source": "D", "target": "E"}}
        ];
        
        // Add links
        svg.selectAll(".link")
            .data(links)
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("x1", d => nodes.find(n => n.id === d.source).x)
            .attr("y1", d => nodes.find(n => n.id === d.source).y)
            .attr("x2", d => nodes.find(n => n.id === d.target).x)
            .attr("y2", d => nodes.find(n => n.id === d.target).y)
            .attr("stroke", "#999")
            .attr("stroke-width", 2);
        
        // Add nodes
        svg.selectAll(".node")
            .data(nodes)
            .enter()
            .append("circle")
            .attr("class", "node")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", 10)
            .attr("fill", "steelblue");
        
        // Add node labels
        svg.selectAll(".node-label")
            .data(nodes)
            .enter()
            .append("text")
            .attr("class", "node-label")
            .attr("x", d => d.x + 12)
            .attr("y", d => d.y + 4)
            .text(d => d.id)
            .attr("font-size", "12px");
    </script>
</body>
</html>
        """
        
        return html_template


class CommunityVisualizer:
    """
    Community visualizer for detecting and visualizing communities in knowledge graphs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the community visualizer.
        
        Args:
            config: Configuration for community visualizer
        """
        self.config = config or {
            "default_width": 1200,
            "default_height": 800,
            "community_detection_algorithm": "louvain",  # louvain, label_propagation, etc.
            "min_community_size": 3
        }
        
        logger.info({
            "msg": "CommunityVisualizer initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def visualize(
        self,
        triples: List[Tuple[str, str, str]],
        correlation_id: Optional[str] = None,
        options: Optional[VisualizationOptions] = None
    ) -> str:
        """
        Generate community visualization of knowledge graph.
        
        Args:
            triples: List of (subject, predicate, object) triples
            correlation_id: Correlation ID for tracking
            options: Visualization options
            
        Returns:
            HTML string with community visualization
        """
        correlation_id = correlation_id or f"community_viz_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting community visualization",
            "triple_count": len(triples),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Use default options if none provided
            if not options:
                options = VisualizationOptions()
            
            # Generate HTML with embedded D3.js visualization
            html_content = self._generate_community_visualization_html(
                triples=triples,
                options=options
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Community visualization completed",
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return html_content
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Community visualization failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            raise
    
    def _generate_community_visualization_html(
        self,
        triples: List[Tuple[str, str, str]],
        options: VisualizationOptions
    ) -> str:
        """
        Generate HTML with community D3.js visualization code.
        
        Args:
            triples: List of (subject, predicate, object) triples
            options: Visualization options
            
        Returns:
            HTML string with visualization
        """
        # Create visualization data structure with simulated communities
        nodes_set = set()
        links = []
        
        for subj, pred, obj in triples:
            nodes_set.add(subj)
            nodes_set.add(obj)
            links.append({
                "source": subj,
                "target": obj,
                "relationship": pred
            })
        
        # Simulate community detection by assigning random communities
        # In a real implementation, this would use actual community detection algorithms
        nodes = []
        for i, node in enumerate(nodes_set):
            nodes.append({
                "id": node,
                "label": node,
                "community": i % 5,  # Assign to one of 5 communities
                "degree": sum(1 for link in links if link["source"] == node or link["target"] == node)
            })
        
        # Create visualization data
        viz_data = {
            "nodes": nodes,
            "links": links,
            "options": {
                "width": options.width,
                "height": options.height,
                "layout": options.layout,
                "include_labels": options.include_labels,
                "include_communities": options.include_communities
            }
        }
        
        # Generate HTML template with embedded D3.js
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Community Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 10px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        #graph-container {{
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: white;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .node-label {{
            font-size: 12px;
            pointer-events: none;
        }}
        .community-label {{
            font-weight: bold;
            font-size: 14px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h2>Community Knowledge Graph Visualization</h2>
    <div id="graph-container"></div>
    
    <script>
        const vizData = {json.dumps(viz_data)};
        
        // Set up dimensions
        const width = vizData.options.width || 800;
        const height = vizData.options.height || 600;
        
        // Create SVG
        const svg = d3.select("#graph-container")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create simulation
        const simulation = d3.forceSimulation(vizData.nodes)
            .force("link", d3.forceLink(vizData.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // Add links
        const link = svg.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(vizData.links)
            .enter()
            .append("line")
            .attr("class", "link")
            .attr("stroke-width", 1);
        
        // Add nodes
        const node = svg.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(vizData.nodes)
            .enter()
            .append("circle")
            .attr("class", "node")
            .attr("r", 10)
            .attr("fill", d => {{
                // Color by community
                const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'];
                return colors[d.community % colors.length];
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended)
            );
        
        // Add labels if requested
        if (vizData.options.include_labels) {{
            const label = svg.append("g")
                .attr("class", "labels")
                .selectAll("text")
                .data(vizData.nodes)
                .enter()
                .append("text")
                .attr("class", "node-label")
                .text(d => d.label)
                .attr("dx", 15)
                .attr("dy", ".35em");
        }}
        
        // Update positions on tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            if (vizData.options.include_labels) {{
                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            }}
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
        """
        
        return html_template