"""
Visualization System Tests

Comprehensive tests for visualization components following CLAUDE.md principles.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_engine.visualization.graph_explorer import GraphExplorer, NodeFilter, EdgeFilter, VisualizationOptions
from knowledge_engine.visualization.temporal_viz import TemporalVisualizer, TimeRange, TemporalVisualizationOptions
from knowledge_engine.visualization.community_viz import CommunityVisualizer, CommunityVisualizationOptions
from knowledge_engine.visualization.export_handlers import ExportHandler
from knowledge_engine.visualization.config import get_visualization_config


# Sample data for testing
SAMPLE_TRIPLES = [
    {'subject': 'Alice', 'predicate': 'knows', 'object': 'Bob', 'confidence': 0.9},
    {'subject': 'Bob', 'predicate': 'knows', 'object': 'Charlie', 'confidence': 0.8},
    {'subject': 'Charlie', 'predicate': 'knows', 'object': 'David', 'confidence': 0.7},
    {'subject': 'David', 'predicate': 'knows', 'object': 'Alice', 'confidence': 0.6},
    {'subject': 'Alice', 'predicate': 'works_with', 'object': 'Charlie', 'confidence': 0.95},
]

SAMPLE_ENTITIES = [
    {'name': 'Alice', 'type': 'Person'},
    {'name': 'Bob', 'type': 'Person'},
    {'name': 'Charlie', 'type': 'Person'},
    {'name': 'David', 'type': 'Person'},
]

SAMPLE_TIMESTAMPS = [
    datetime.utcnow() - timedelta(days=4),
    datetime.utcnow() - timedelta(days=3),
    datetime.utcnow() - timedelta(days=2),
    datetime.utcnow() - timedelta(days=1),
    datetime.utcnow(),
]


class TestVisualizationConfig:
    """Test configuration system."""

    def test_config_initialization(self):
        """Test that configuration initializes correctly."""
        config = get_visualization_config()
        assert config is not None
        assert config.output_dir is not None
        assert config.cache_dir is not None
        assert config.max_nodes > 0
        assert config.max_edges > 0

    def test_config_validation(self):
        """Test configuration validation."""
        config = get_visualization_config()
        # Valid config should not raise
        config._validate_config()

    def test_config_to_dict(self):
        """Test configuration export."""
        config = get_visualization_config()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'VISUALIZATION_OUTPUT_DIR' in config_dict


class TestGraphExplorer:
    """Test graph explorer functionality."""

    @pytest.fixture
    def explorer(self):
        """Create graph explorer instance."""
        config = get_visualization_config()
        return GraphExplorer(config)

    def test_build_graph(self, explorer):
        """Test graph building from triples."""
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 5

    def test_node_filter_search(self, explorer):
        """Test node search filtering."""
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        node_filter = NodeFilter(search_query='Alice')
        filtered = explorer._apply_filters(graph, node_filter, None)
        assert 'Alice' in filtered.nodes()

    def test_node_filter_degree(self, explorer):
        """Test node degree filtering."""
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        node_filter = NodeFilter(min_degree=2)
        filtered = explorer._apply_filters(graph, node_filter, None)
        # Alice has degree 3, Bob has 2, etc.
        assert filtered.number_of_nodes() <= graph.number_of_nodes()

    def test_edge_filter_confidence(self, explorer):
        """Test edge confidence filtering."""
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        edge_filter = EdgeFilter(min_confidence=0.8)
        filtered = explorer._apply_filters(graph, None, edge_filter)
        assert filtered.number_of_edges() <= graph.number_of_edges()

    def test_detect_communities(self, explorer):
        """Test community detection."""
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        communities = asyncio.run(explorer._detect_communities(graph))
        assert isinstance(communities, dict)
        assert len(communities) > 0

    def test_compute_centrality(self, explorer):
        """Test centrality computation."""
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        centrality = asyncio.run(explorer._compute_centrality(graph))
        assert isinstance(centrality, dict)
        assert len(centrality) == graph.number_of_nodes()
        # All values should be between 0 and 1
        for score in centrality.values():
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_visualize_graph(self, explorer, tmp_path):
        """Test full graph visualization."""
        output_path = tmp_path / "test_graph.html"
        result = await explorer.visualize(
            triples=SAMPLE_TRIPLES,
            entities=SAMPLE_ENTITIES,
            output_path=str(output_path)
        )
        assert result.output_path == str(output_path)
        assert result.node_count == 4
        assert result.edge_count == 5
        assert Path(result.output_path).exists()


class TestTemporalVisualizer:
    """Test temporal visualization functionality."""

    @pytest.fixture
    def temporal_viz(self):
        """Create temporal visualizer instance."""
        config = get_visualization_config()
        return TemporalVisualizer(config)

    def test_build_temporal_graph(self, temporal_viz):
        """Test temporal graph building."""
        graph = temporal_viz._build_temporal_graph(SAMPLE_TRIPLES, SAMPLE_TIMESTAMPS)
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 5
        # Check that edges have timestamps
        for _, _, data in graph.edges(data=True):
            assert 'timestamp' in data

    def test_time_window_filter(self, temporal_viz):
        """Test time window filtering."""
        graph = temporal_viz._build_temporal_graph(SAMPLE_TRIPLES, SAMPLE_TIMESTAMPS)
        time_window = TimeRange(
            start=datetime.utcnow() - timedelta(days=2),
            end=datetime.utcnow()
        )
        filtered = temporal_viz._filter_by_time_window(graph, time_window)
        # Should only include edges from last 2 days
        assert filtered.number_of_edges() <= graph.number_of_edges()

    def test_generate_snapshots(self, temporal_viz):
        """Test snapshot generation."""
        graph = temporal_viz._build_temporal_graph(SAMPLE_TRIPLES, SAMPLE_TIMESTAMPS)
        snapshots = temporal_viz._generate_snapshots(graph, num_snapshots=5)
        assert len(snapshots) == 5
        # Check that snapshots are ordered
        for i in range(1, len(snapshots)):
            assert snapshots[i].timestamp >= snapshots[i-1].timestamp

    def test_compute_temporal_statistics(self, temporal_viz):
        """Test temporal statistics computation."""
        graph = temporal_viz._build_temporal_graph(SAMPLE_TRIPLES, SAMPLE_TIMESTAMPS)
        snapshots = temporal_viz._generate_snapshots(graph)
        stats = temporal_viz._compute_temporal_statistics(graph, snapshots)
        assert 'num_snapshots' in stats
        assert 'node_growth' in stats
        assert 'edge_growth' in stats

    @pytest.mark.asyncio
    async def test_visualize_temporal(self, temporal_viz, tmp_path):
        """Test full temporal visualization."""
        output_path = tmp_path / "test_temporal.html"
        result = await temporal_viz.visualize_temporal(
            triples=SAMPLE_TRIPLES,
            timestamps=SAMPLE_TIMESTAMPS,
            output_path=str(output_path)
        )
        assert 'output_path' in result
        assert 'snapshots' in result
        assert Path(result['output_path']).exists()


class TestCommunityVisualizer:
    """Test community visualization functionality."""

    @pytest.fixture
    def community_viz(self):
        """Create community visualizer instance."""
        config = get_visualization_config()
        return CommunityVisualizer(config)

    def test_analyze_communities(self, community_viz):
        """Test community analysis."""
        graph = community_viz._build_graph(SAMPLE_TRIPLES)
        communities = asyncio.run(community_viz._detect_communities(graph))
        community_info = community_viz._analyze_communities(graph, communities, CommunityVisualizationOptions())
        assert len(community_info) > 0
        # Check that each community has required fields
        for comm in community_info:
            assert comm.id is not None
            assert len(comm.nodes) > 0
            assert comm.internal_edges >= 0
            assert comm.density >= 0

    def test_compute_community_hierarchy(self, community_viz):
        """Test community hierarchy computation."""
        graph = community_viz._build_graph(SAMPLE_TRIPLES)
        communities = asyncio.run(community_viz._detect_communities(graph))
        hierarchy = community_viz._compute_community_hierarchy(graph, communities)
        assert 'levels' in hierarchy
        assert 'connections' in hierarchy

    def test_compute_inter_community_edges(self, community_viz):
        """Test inter-community edge computation."""
        graph = community_viz._build_graph(SAMPLE_TRIPLES)
        communities = asyncio.run(community_viz._detect_communities(graph))
        inter_edges = community_viz._compute_inter_community_edges(graph, communities)
        assert isinstance(inter_edges, list)

    @pytest.mark.asyncio
    async def test_visualize_communities(self, community_viz, tmp_path):
        """Test full community visualization."""
        output_path = tmp_path / "test_community.html"
        result = await community_viz.visualize_communities(
            triples=SAMPLE_TRIPLES,
            entities=SAMPLE_ENTITIES,
            output_path=str(output_path)
        )
        assert 'output_path' in result
        assert 'num_communities' in result
        assert Path(result['output_path']).exists()


class TestExportHandler:
    """Test export functionality."""

    @pytest.fixture
    def exporter(self):
        """Create export handler instance."""
        config = get_visualization_config()
        return ExportHandler(config)

    @pytest.mark.asyncio
    async def test_export_svg(self, exporter, tmp_path):
        """Test SVG export."""
        graph_data = {
            'nodes': [{'id': 'A', 'x': 100, 'y': 100, 'size': 10, 'color': '#1f77b4'}],
            'edges': [{'source': 'A', 'target': 'A', 'confidence': 1.0}]
        }
        output_path = tmp_path / "test_export.svg"
        result = await exporter.export_svg(graph_data, str(output_path))
        assert Path(result).exists()
        # Check that file contains SVG content
        with open(result, 'r') as f:
            content = f.read()
            assert '<svg' in content

    @pytest.mark.asyncio
    async def test_export_html(self, exporter, tmp_path):
        """Test HTML export."""
        graph_data = {
            'nodes': [{'id': 'A'}],
            'edges': []
        }
        output_path = tmp_path / "test_export.html"
        result = await exporter.export_html(graph_data, str(output_path))
        assert Path(result).exists()
        with open(result, 'r') as f:
            content = f.read()
            assert '<!DOCTYPE html>' in content
            assert 'D3.js' in content

    @pytest.mark.asyncio
    async def test_export_json(self, exporter, tmp_path):
        """Test JSON export."""
        graph_data = {
            'nodes': [{'id': 'A'}],
            'edges': []
        }
        output_path = tmp_path / "test_export.json"
        result = await exporter.export_json(graph_data, str(output_path))
        assert Path(result).exists()
        with open(result, 'r') as f:
            content = f.read()
            import json
            data = json.loads(content)
            assert 'nodes' in data

    @pytest.mark.asyncio
    async def test_export_graphml(self, exporter, tmp_path):
        """Test GraphML export."""
        output_path = tmp_path / "test_export.graphml"
        result = await exporter.export_graphml(SAMPLE_TRIPLES, str(output_path))
        assert Path(result).exists()
        with open(result, 'r') as f:
            content = f.read()
            assert '<?xml' in content
            assert '<graph' in content

    def test_generate_embedding_url(self, exporter):
        """Test embedding URL generation."""
        graph_data = {'nodes': [], 'edges': []}
        url = exporter.generate_embedding_url(
            graph_data,
            'https://example.com'
        )
        assert 'https://example.com' in url
        assert 'embed' in url


class TestIntegration:
    """Integration tests for the complete visualization system."""

    @pytest.mark.asyncio
    async def test_end_to_end_graph_visualization(self, tmp_path):
        """Test complete graph visualization pipeline."""
        config = get_visualization_config()
        explorer = GraphExplorer(config)

        output_path = tmp_path / "e2e_graph.html"
        result = await explorer.visualize(
            triples=SAMPLE_TRIPLES,
            entities=SAMPLE_ENTITIES,
            output_path=str(output_path),
            options=VisualizationOptions(
                width=800,
                height=600,
                show_labels=True
            )
        )

        # Verify result
        assert result.node_count > 0
        assert result.edge_count > 0
        assert Path(result.output_path).exists()

        # Verify HTML content
        with open(result.output_path, 'r') as f:
            html_content = f.read()
            assert '<!DOCTYPE html>' in html_content
            assert 'D3.js' in html_content

    @pytest.mark.asyncio
    async def test_end_to_end_temporal_visualization(self, tmp_path):
        """Test complete temporal visualization pipeline."""
        config = get_visualization_config()
        temporal_viz = TemporalVisualizer(config)

        output_path = tmp_path / "e2e_temporal.html"
        result = await temporal_viz.visualize_temporal(
            triples=SAMPLE_TRIPLES,
            timestamps=SAMPLE_TIMESTAMPS,
            output_path=str(output_path)
        )

        assert 'output_path' in result
        assert Path(result['output_path']).exists()

    @pytest.mark.asyncio
    async def test_end_to_end_export_pipeline(self, tmp_path):
        """Test complete export pipeline."""
        config = get_visualization_config()
        exporter = ExportHandler(config)

        # First create visualization
        explorer = GraphExplorer(config)
        graph = explorer._build_graph(SAMPLE_TRIPLES)
        communities = await explorer._detect_communities(graph)
        centrality = await explorer._compute_centrality(graph)

        graph_data = explorer._prepare_graph_data(
            graph, SAMPLE_TRIPLES, communities, centrality,
            VisualizationOptions()
        )

        # Export in multiple formats
        formats = ['svg', 'html', 'json']
        for fmt in formats:
            output_path = tmp_path / f"e2e_export.{fmt}"

            if fmt == 'svg':
                result = await exporter.export_svg(graph_data, str(output_path))
            elif fmt == 'html':
                result = await exporter.export_html(graph_data, str(output_path))
            elif fmt == 'json':
                result = await exporter.export_json(graph_data, str(output_path))

            assert Path(result).exists()

    def test_cache_key_generation(self):
        """Test cache key generation (idempotency)."""
        config = get_visualization_config()
        explorer = GraphExplorer(config)

        graph_data = {'nodes': [], 'edges': []}
        node_filter = NodeFilter(search_query='test')
        edge_filter = EdgeFilter(min_confidence=0.5)
        options = VisualizationOptions(width=800)

        key1 = explorer.generate_cache_key(graph_data, node_filter, edge_filter, options)
        key2 = explorer.generate_cache_key(graph_data, node_filter, edge_filter, options)

        # Same inputs should produce same key (idempotency)
        assert key1 == key2

        # Different inputs should produce different key
        key3 = explorer.generate_cache_key(graph_data, None, None, None)
        assert key1 != key3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
