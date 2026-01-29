"""
Visualization API Endpoints

Production-grade FastAPI endpoints for visualization system.
Following CLAUDE.md principles with structured logging and error handling.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field

from .graph_explorer import (
    GraphExplorer, NodeFilter, EdgeFilter, VisualizationOptions
)
from .temporal_viz import (
    TemporalVisualizer, TimeRange, TemporalVisualizationOptions
)
from .community_viz import (
    CommunityVisualizer, CommunityVisualizationOptions
)
from .export_handlers import ExportHandler
from .config import get_visualization_config

logger = logging.getLogger(__name__)

# Initialize components
config = get_visualization_config()
graph_explorer = GraphExplorer(config)
temporal_visualizer = TemporalVisualizer(config)
community_visualizer = CommunityVisualizer(config)
export_handler = ExportHandler(config)


# Request/Response Models

class GraphVisualizationRequest(BaseModel):
    """Request for graph visualization."""
    triples: List[Dict[str, Any]] = Field(..., description="List of graph triples")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="List of entities")
    node_filter: Optional[Dict[str, Any]] = Field(None, description="Node filtering criteria")
    edge_filter: Optional[Dict[str, Any]] = Field(None, description="Edge filtering criteria")
    options: Optional[Dict[str, Any]] = Field(None, description="Visualization options")
    output_filename: Optional[str] = Field(None, description="Output filename")


class TemporalVisualizationRequest(BaseModel):
    """Request for temporal visualization."""
    triples: List[Dict[str, Any]] = Field(..., description="List of triples with timestamps")
    timestamps: List[str] = Field(..., description="ISO format timestamps")
    time_window_start: Optional[str] = Field(None, description="Time window start (ISO)")
    time_window_end: Optional[str] = Field(None, description="Time window end (ISO)")
    options: Optional[Dict[str, Any]] = Field(None, description="Visualization options")


class CommunityVisualizationRequest(BaseModel):
    """Request for community visualization."""
    triples: List[Dict[str, Any]] = Field(..., description="List of graph triples")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="List of entities")
    options: Optional[Dict[str, Any]] = Field(None, description="Visualization options")


class ExportRequest(BaseModel):
    """Request for export."""
    format: str = Field(..., description="Export format: png, svg, html, graphml, gexf, json")
    triples: List[Dict[str, Any]] = Field(..., description="List of graph triples")
    graph_data: Optional[Dict[str, Any]] = Field(None, description="Graph data for PNG/SVG/HTML")
    width: int = Field(1200, description="Output width")
    height: int = Field(800, description="Output height")
    dpi: int = Field(300, description="DPI for PNG")


class SubgraphExtractionRequest(BaseModel):
    """Request for subgraph extraction."""
    triples: List[Dict[str, Any]] = Field(..., description="List of graph triples")
    center_node: str = Field(..., description="Center node for subgraph")
    radius: int = Field(1, description="Radius for subgraph (hops)")
    min_degree: Optional[int] = Field(None, description="Minimum degree filter")


class VisualizationResponse(BaseModel):
    """Response for visualization requests."""
    visualization_id: str = Field(..., description="Unique visualization ID")
    output_path: str = Field(..., description="Path to generated visualization")
    url: str = Field(..., description="URL to access visualization")
    node_count: int = Field(..., description="Number of nodes")
    edge_count: int = Field(..., description="Number of edges")
    statistics: Dict[str, Any] = Field(..., description="Graph statistics")
    timestamp: str = Field(..., description="Generation timestamp (UTC)")


class StatisticsResponse(BaseModel):
    """Response for graph statistics."""
    visualization_id: str
    node_count: int
    edge_count: int
    communities: int
    density: float
    is_connected: bool
    avg_clustering: float
    diameter: Optional[float]
    centrality_scores: Dict[str, float]
    timestamp: str


async def create_graph_visualization(request: GraphVisualizationRequest) -> VisualizationResponse:
    """
    Create interactive graph visualization.

    Args:
        request: Visualization request

    Returns:
        VisualizationResponse with metadata

    Raises:
        HTTPException: If visualization fails
    """
    visualization_id = str(uuid.uuid4())
    start_time = datetime.utcnow()

    logger.info({
        'event': 'graph_visualization_requested',
        'visualization_id': visualization_id,
        'num_triples': len(request.triples),
        'timestamp': start_time.isoformat()
    })

    try:
        # Convert triples to appropriate format
        # (In production, would properly deserialize from dict)

        # Apply filters
        node_filter = None
        if request.node_filter:
            node_filter = NodeFilter(**request.node_filter)

        edge_filter = None
        if request.edge_filter:
            edge_filter = EdgeFilter(**request.edge_filter)

        options = None
        if request.options:
            options = VisualizationOptions(**request.options)

        # Generate visualization
        result = await graph_explorer.visualize(
            triples=request.triples,
            entities=request.entities,
            node_filter=node_filter,
            edge_filter=edge_filter,
            options=options
        )

        # Generate URL
        url = f"/visualizations/graph/{Path(result.output_path).name}"

        logger.info({
            'event': 'graph_visualization_complete',
            'visualization_id': visualization_id,
            'output_path': result.output_path,
            'generation_time': result.generation_time,
            'timestamp': datetime.utcnow().isoformat()
        })

        return VisualizationResponse(
            visualization_id=visualization_id,
            output_path=result.output_path,
            url=url,
            node_count=result.node_count,
            edge_count=result.edge_count,
            statistics=result.statistics,
            timestamp=result.timestamp
        )

    except Exception as e:
        logger.error({
            'event': 'graph_visualization_failed',
            'visualization_id': visualization_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")


async def create_temporal_visualization(request: TemporalVisualizationRequest) -> VisualizationResponse:
    """Create temporal graph visualization."""
    visualization_id = str(uuid.uuid4())

    logger.info({
        'event': 'temporal_visualization_requested',
        'visualization_id': visualization_id,
        'num_triples': len(request.triples),
        'timestamp': datetime.utcnow().isoformat()
    })

    try:
        # Parse timestamps
        timestamps = [
            datetime.fromisoformat(ts.replace('Z', '+00:00'))
            for ts in request.timestamps
        ]

        # Create time window
        time_window = None
        if request.time_window_start or request.time_window_end:
            time_window = TimeRange(
                start=datetime.fromisoformat(request.time_window_start) if request.time_window_start else None,
                end=datetime.fromisoformat(request.time_window_end) if request.time_window_end else None
            )

        options = None
        if request.options:
            options = TemporalVisualizationOptions(**request.options)

        # Generate visualization
        result = await temporal_visualizer.visualize_temporal(
            triples=request.triples,
            timestamps=timestamps,
            time_window=time_window,
            options=options
        )

        url = f"/visualizations/temporal/{Path(result['output_path']).name}"

        return VisualizationResponse(
            visualization_id=visualization_id,
            output_path=result['output_path'],
            url=url,
            node_count=result['statistics'].get('final_nodes', 0),
            edge_count=result['statistics'].get('final_edges', 0),
            statistics=result['statistics'],
            timestamp=result['timestamp']
        )

    except Exception as e:
        logger.error({
            'event': 'temporal_visualization_failed',
            'visualization_id': visualization_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Temporal visualization failed: {str(e)}")


async def create_community_visualization(request: CommunityVisualizationRequest) -> VisualizationResponse:
    """Create community-based visualization."""
    visualization_id = str(uuid.uuid4())

    logger.info({
        'event': 'community_visualization_requested',
        'visualization_id': visualization_id,
        'num_triples': len(request.triples),
        'timestamp': datetime.utcnow().isoformat()
    })

    try:
        options = None
        if request.options:
            options = CommunityVisualizationOptions(**request.options)

        # Generate visualization
        result = await community_visualizer.visualize_communities(
            triples=request.triples,
            entities=request.entities,
            options=options
        )

        url = f"/visualizations/community/{Path(result['output_path']).name}"

        return VisualizationResponse(
            visualization_id=visualization_id,
            output_path=result['output_path'],
            url=url,
            node_count=0,  # Would be filled in by actual implementation
            edge_count=0,
            statistics={'num_communities': result['num_communities']},
            timestamp=result['timestamp']
        )

    except Exception as e:
        logger.error({
            'event': 'community_visualization_failed',
            'visualization_id': visualization_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Community visualization failed: {str(e)}")


async def export_visualization(request: ExportRequest) -> Dict[str, str]:
    """Export visualization in specified format."""
    export_id = str(uuid.uuid4())

    logger.info({
        'event': 'export_requested',
        'export_id': export_id,
        'format': request.format,
        'timestamp': datetime.utcnow().isoformat()
    })

    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_path = f"data/visualizations/exports/{request.format}_{timestamp}.{request.format}"

        if request.format == 'png':
            result = await export_handler.export_png(
                graph_data=request.graph_data or {},
                output_path=output_path,
                width=request.width,
                height=request.height,
                dpi=request.dpi
            )
        elif request.format == 'svg':
            result = await export_handler.export_svg(
                graph_data=request.graph_data or {},
                output_path=output_path,
                width=request.width,
                height=request.height
            )
        elif request.format == 'html':
            result = await export_handler.export_html(
                graph_data=request.graph_data or {},
                output_path=output_path,
                width=request.width,
                height=request.height
            )
        elif request.format == 'graphml':
            result = await export_handler.export_graphml(
                triples=request.triples,
                output_path=output_path
            )
        elif request.format == 'gexf':
            result = await export_handler.export_gexf(
                triples=request.triples,
                output_path=output_path
            )
        elif request.format == 'json':
            result = await export_handler.export_json(
                graph_data=request.graph_data or {},
                output_path=output_path
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

        return {
            'export_id': export_id,
            'output_path': result,
            'format': request.format,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error({
            'event': 'export_failed',
            'export_id': export_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


async def extract_subgraph(request: SubgraphExtractionRequest) -> Dict[str, Any]:
    """Extract subgraph around a node."""
    extraction_id = str(uuid.uuid4())

    logger.info({
        'event': 'subgraph_extraction_requested',
        'extraction_id': extraction_id,
        'center_node': request.center_node,
        'radius': request.radius,
        'timestamp': datetime.utcnow().isoformat()
    })

    try:
        import networkx as nx

        # Build graph
        graph = nx.Graph()
        for triple in request.triples:
            if len(triple) >= 3:
                subj, pred, obj = triple[0], triple[1], triple[2]
                graph.add_edge(subj, obj, predicate=pred)

        # Extract ego graph
        if request.center_node not in graph.nodes():
            raise HTTPException(status_code=404, detail=f"Node {request.center_node} not found")

        subgraph = nx.ego_graph(graph, request.center_node, radius=request.radius)

        # Apply degree filter
        if request.min_degree:
            nodes_to_remove = [
                n for n in subgraph.nodes()
                if subgraph.degree(n) < request.min_degree
            ]
            subgraph.remove_nodes_from(nodes_to_remove)

        # Convert to triple format
        subgraph_triples = [
            {'subject': u, 'predicate': d['predicate'], 'object': v}
            for u, v, d in subgraph.edges(data=True)
        ]

        return {
            'extraction_id': extraction_id,
            'center_node': request.center_node,
            'radius': request.radius,
            'node_count': subgraph.number_of_nodes(),
            'edge_count': subgraph.number_of_edges(),
            'triples': subgraph_triples,
            'timestamp': datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error({
            'event': 'subgraph_extraction_failed',
            'extraction_id': extraction_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


async def get_graph_statistics(triples: List[Dict[str, Any]]) -> StatisticsResponse:
    """Compute comprehensive graph statistics."""
    stats_id = str(uuid.uuid4())

    try:
        import networkx as nx

        # Build graph
        graph = nx.Graph()
        for triple in triples:
            if len(triple) >= 3:
                subj, pred, obj = triple[0], triple[1], triple[2]
                conf = triple[3] if len(triple) > 3 else 1.0
                graph.add_edge(subj, obj, predicate=pred, confidence=conf)

        # Compute statistics
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        # Detect communities
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(graph)
            num_communities = len(communities)
        except:
            num_communities = 0

        # Centrality
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph)
        eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=100)

        centrality_scores = {}
        for node in graph.nodes():
            centrality_scores[node] = {
                'degree': degree_centrality.get(node, 0),
                'betweenness': betweenness_centrality.get(node, 0),
                'eigenvector': eigenvector_centrality.get(node, 0)
            }

        # Other stats
        density = nx.density(graph)
        is_connected = nx.is_connected(graph)
        avg_clustering = nx.average_clustering(graph)

        diameter = None
        if is_connected:
            try:
                diameter = nx.diameter(graph)
            except:
                pass

        return StatisticsResponse(
            visualization_id=stats_id,
            node_count=node_count,
            edge_count=edge_count,
            communities=num_communities,
            density=density,
            is_connected=is_connected,
            avg_clustering=avg_clustering,
            diameter=diameter,
            centrality_scores=centrality_scores,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error({
            'event': 'statistics_computation_failed',
            'stats_id': stats_id,
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })
        raise HTTPException(status_code=500, detail=f"Statistics computation failed: {str(e)}")
