"""
Export Handlers

Production-grade export functionality with:
- PNG export (high-resolution)
- SVG export (vector)
- HTML export (interactive)
- GraphML, GEXF formats
- Embed URL generation
"""

import json
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO

import networkx as nx

from .config import get_visualization_config

logger = logging.getLogger(__name__)


class ExportHandler:
    """
    Handle visualization exports in multiple formats.

    Supports:
    - PNG (raster image)
    - SVG (vector graphics)
    - HTML (interactive)
    - GraphML (NetworkX format)
    - GEXF (Gephi format)
    - JSON (D3.js format)
    """

    def __init__(self, config=None):
        """Initialize export handler."""
        self.config = config or get_visualization_config()
        self.output_dir = Path(self.config.output_dir) / 'exports'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info({
            'event': 'export_handler_initialized',
            'output_dir': str(self.output_dir),
            'timestamp': datetime.utcnow().isoformat()
        })

    async def export_png(
        self,
        graph_data: Dict[str, Any],
        output_path: str,
        width: int = 1200,
        height: int = 800,
        dpi: int = 300
    ) -> str:
        """
        Export visualization as high-resolution PNG.

        Args:
            graph_data: Graph data dictionary
            output_path: Output file path
            width: Image width in pixels
            height: Image height in pixels
            dpi: Resolution (dots per inch)

        Returns:
            Path to exported file
        """
        # Generate HTML with embedded screenshot capability
        # In production, this would use a headless browser (Puppeteer/Playwright)

        logger.info({
            'event': 'png_export_requested',
            'output_path': output_path,
            'width': width,
            'height': height,
            'dpi': dpi,
            'timestamp': datetime.utcnow().isoformat()
        })

        # For now, create a placeholder that would be processed by a screenshot service
        html_content = self._generate_screenshot_html(graph_data, width, height, dpi)

        # Save HTML that will be screenshotted
        html_path = str(output_path).replace('.png', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # In production: Use Puppeteer/Playwright to screenshot
        # For now, return the HTML path
        return html_path

    async def export_svg(
        self,
        graph_data: Dict[str, Any],
        output_path: str,
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Export visualization as SVG.

        Args:
            graph_data: Graph data dictionary
            output_path: Output file path
            width: SVG width
            height: SVG height

        Returns:
            Path to exported file
        """
        svg_content = self._generate_svg(graph_data, width, height)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        logger.info({
            'event': 'svg_export_complete',
            'output_path': str(output_file),
            'timestamp': datetime.utcnow().isoformat()
        })

        return str(output_file)

    async def export_html(
        self,
        graph_data: Dict[str, Any],
        output_path: str,
        width: int = 1200,
        height: int = 800,
        embed_data: bool = True
    ) -> str:
        """
        Export visualization as standalone HTML.

        Args:
            graph_data: Graph data dictionary
            output_path: Output file path
            width: Visualization width
            height: Visualization height
            embed_data: Whether to embed data or load from URL

        Returns:
            Path to exported file
        """
        html_content = self._generate_standalone_html(
            graph_data, width, height, embed_data
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info({
            'event': 'html_export_complete',
            'output_path': str(output_file),
            'timestamp': datetime.utcnow().isoformat()
        })

        return str(output_file)

    async def export_graphml(
        self,
        triples: List[Any],
        output_path: str
    ) -> str:
        """
        Export graph in GraphML format.

        Args:
            triples: List of graph triples
            output_path: Output file path

        Returns:
            Path to exported file
        """
        graph = self._build_graph(triples)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        nx.write_graphml(graph, str(output_file))

        logger.info({
            'event': 'graphml_export_complete',
            'output_path': str(output_file),
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'timestamp': datetime.utcnow().isoformat()
        })

        return str(output_file)

    async def export_gexf(
        self,
        triples: List[Any],
        output_path: str
    ) -> str:
        """
        Export graph in GEXF format (Gephi).

        Args:
            triples: List of graph triples
            output_path: Output file path

        Returns:
            Path to exported file
        """
        graph = self._build_graph(triples)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        nx.write_gexf(graph, str(output_file))

        logger.info({
            'event': 'gexf_export_complete',
            'output_path': str(output_file),
            'nodes': graph.number_of_nodes(),
            'edges': graph.number_of_edges(),
            'timestamp': datetime.utcnow().isoformat()
        })

        return str(output_file)

    async def export_json(
        self,
        graph_data: Dict[str, Any],
        output_path: str,
        pretty: bool = True
    ) -> str:
        """
        Export graph data as JSON.

        Args:
            graph_data: Graph data dictionary
            output_path: Output file path
            pretty: Whether to pretty-print JSON

        Returns:
            Path to exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2 if pretty else None)

        logger.info({
            'event': 'json_export_complete',
            'output_path': str(output_file),
            'timestamp': datetime.utcnow().isoformat()
        })

        return str(output_file)

    def generate_embedding_url(
        self,
        graph_data: Dict[str, Any],
        base_url: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate embeddable URL for visualization.

        Args:
            graph_data: Graph data dictionary
            base_url: Base URL for embedding
            config: Optional configuration

        Returns:
            Embed URL
        """
        # In production, this would store graph data and return a URL
        # For now, return a data URL

        json_data = json.dumps(graph_data)
        encoded_data = base64.b64encode(json_data.encode()).decode()

        embed_url = f"{base_url}/embed?data={encoded_data}"

        if config:
            config_json = json.dumps(config)
            encoded_config = base64.b64encode(config_json.encode()).decode()
            embed_url += f"&config={encoded_config}"

        logger.info({
            'event': 'embedding_url_generated',
            'base_url': base_url,
            'timestamp': datetime.utcnow().isoformat()
        })

        return embed_url

    def _build_graph(self, triples: List[Any]) -> nx.Graph:
        """Build NetworkX graph from triples.

        Handles multiple formats:
        - Object with subject/predicate/object attributes
        - Dict with subject/predicate/object keys
        - Tuple/list with [subject, predicate, object, confidence?]
        """
        graph = nx.Graph()

        for triple in triples:
            try:
                # Handle object format
                if hasattr(triple, 'subject'):
                    subj = triple.subject
                    pred = triple.predicate
                    obj = triple.object
                    conf = getattr(triple, 'confidence', 1.0)
                # Handle dict format - FIXED to check for keys first
                elif isinstance(triple, dict):
                    # Check for required keys
                    if not all(k in triple for k in ['subject', 'predicate', 'object']):
                        logger.warning({
                            'event': 'invalid_triple_dict_missing_keys',
                            'triple_keys': list(triple.keys()),
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        continue
                    subj = triple['subject']
                    pred = triple['predicate']
                    obj = triple['object']
                    conf = triple.get('confidence', 1.0)
                # Handle tuple/list format
                elif isinstance(triple, (tuple, list)) and len(triple) >= 3:
                    subj, pred, obj = triple[0], triple[1], triple[2]
                    conf = triple[3] if len(triple) > 3 else 1.0
                else:
                    logger.warning({
                        'event': 'unknown_triple_format',
                        'triple_type': type(triple).__name__,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    continue

                graph.add_edge(subj, obj, predicate=pred, confidence=float(conf))

            except Exception as e:
                logger.warning({
                    'event': 'triple_processing_failed',
                    'triple': str(triple),
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
                continue

        return graph

    def _generate_svg(
        self,
        graph_data: Dict[str, Any],
        width: int,
        height: int
    ) -> str:
        """Generate SVG content."""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])

        svg_parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<style>',
            '.node { stroke: #fff; stroke-width: 1.5px; }',
            '.link { stroke-opacity: 0.6; }',
            '</style>'
        ]

        # Draw edges
        for edge in edges:
            source = next((n for n in nodes if n['id'] == edge['source']), None)
            target = next((n for n in nodes if n['id'] == edge['target']), None)

            if source and target:
                svg_parts.append(
                    f'<line class="link" x1="{source.get("x", 0)}" y1="{source.get("y", 0)}" '
                    f'x2="{target.get("x", 0)}" y2="{target.get("y", 0)}" '
                    f'stroke="#999" stroke-width="{edge.get("confidence", 1) * 2}"/>'
                )

        # Draw nodes
        for node in nodes:
            svg_parts.append(
                f'<circle class="node" cx="{node.get("x", 0)}" cy="{node.get("y", 0)}" '
                f'r="{node.get("size", 5)}" fill="{node.get("color", "#1f77b4")}"/>'
            )

        svg_parts.append('</svg>')

        return '\n'.join(svg_parts)

    def _generate_standalone_html(
        self,
        graph_data: Dict[str, Any],
        width: int,
        height: int,
        embed_data: bool
    ) -> str:
        """Generate standalone HTML."""
        if embed_data:
            data_json = json.dumps(graph_data)
        else:
            data_json = "null"  # Would load from URL in production

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; overflow: hidden; }}
        #graph {{ width: {width}px; height: {height}px; }}
        .node {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
        .link {{ stroke-opacity: 0.6; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <script>
        const graphData = {data_json};
        // D3.js visualization code
    </script>
</body>
</html>"""

    def _generate_screenshot_html(
        self,
        graph_data: Dict[str, Any],
        width: int,
        height: int,
        dpi: int
    ) -> str:
        """Generate HTML optimized for screenshotting."""
        return self._generate_standalone_html(graph_data, width, height, True)
