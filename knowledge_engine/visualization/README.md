# Knowledge Graph Visualization System

Production-grade visualization system for OpenEvolve knowledge graphs.

## Features

### 1. Interactive Graph Explorer (Task 4.1)
- **4.1.1**: Integrated ai-knowledge-graph visualization components
- **4.1.2**: Interactive node filtering (search, type, attributes)
- **4.1.3**: Edge filtering by relationship type
- **4.1.4**: Zoom and pan controls
- **4.1.5**: Node attribute display on hover
- **4.1.6**: Edge attribute display

### 2. Temporal Graph Visualization (Task 4.2)
- **4.2.1**: Time-based graph filtering
- **4.2.2**: Temporal edge visualization (color by age)
- **4.2.3**: Timeline slider for historical views
- **4.2.4**: Animation for temporal changes
- **4.2.5**: Before/after comparison views

### 3. Community-Based Views (Task 4.3)
- **4.3.1**: Community color coding
- **4.3.2**: Community-centric layouts (force-directed per community)
- **4.3.3**: Inter-community relationship visualization
- **4.3.4**: Community hierarchy display
- **4.3.5**: Community filtering options

### 4. Advanced Visualization Features (Task 4.4)
- **4.4.1**: Centrality-based node sizing
- **4.4.2**: Relationship strength visualization (line thickness)
- **4.4.3**: Confidence scoring visualization (opacity)
- **4.4.4**: Subgraph extraction and display
- **4.4.5**: Graph statistics dashboard

### 5. Visualization API (Task 4.5)
- **4.5.1**: Graph export endpoints (PNG, SVG, HTML)
- **4.5.2**: Visualization configuration API
- **4.5.3**: Custom layout support
- **4.5.4**: Embedding URL generation
- **4.5.5**: Visualization widget library

### 6. Testing & Documentation (Task 4.6)
- **4.6.1**: Visualization component tests
- **4.6.2**: User guide for graph explorer
- **4.6.3**: Visualization examples with sample graphs
- **4.6.4**: Visualization API endpoint documentation
- **4.6.5**: Embedding tutorial for external sites

## Installation

```bash
pip install networkx matplotlib
npm install d3  # For frontend visualizations
```

## Configuration

All configuration via environment variables (CLAUDE.md compliance):

```bash
export VISUALIZATION_OUTPUT_DIR=data/visualizations
export VISUALIZATION_CACHE_DIR=data/visualization_cache
export VISUALIZATION_MAX_NODES=10000
export VISUALIZATION_MAX_EDGES=50000
export VISUALIZATION_DEFAULT_WIDTH=1200
export VISUALIZATION_DEFAULT_HEIGHT=800
export VISUALIZATION_CACHE_TTL=3600
export VISUALIZATION_EXPORT_TIMEOUT=30
export VISUALIZATION_ENABLE_CACHING=true
```

## Quick Start

### Python API

```python
from knowledge_engine.visualization import GraphExplorer, VisualizationOptions

# Initialize explorer
explorer = GraphExplorer()

# Prepare data
triples = [
    {'subject': 'Alice', 'predicate': 'knows', 'object': 'Bob', 'confidence': 0.9},
    {'subject': 'Bob', 'predicate': 'knows', 'object': 'Charlie', 'confidence': 0.8},
]

# Generate visualization
result = await explorer.visualize(
    triples=triples,
    entities=[],
    options=VisualizationOptions(
        width=1200,
        height=800,
        show_labels=True
    )
)

print(f"Visualization saved to: {result.output_path}")
```

### REST API

```bash
# Create graph visualization
curl -X POST http://localhost:8000/api/visualization/graph \
  -H "Content-Type: application/json" \
  -d '{
    "triples": [
      {"subject": "Alice", "predicate": "knows", "object": "Bob", "confidence": 0.9}
    ],
    "options": {
      "width": 1200,
      "height": 800,
      "show_labels": true
    }
  }'
```

## Usage Examples

### 1. Interactive Graph Exploration

```python
from knowledge_engine.visualization import GraphExplorer, NodeFilter, EdgeFilter

explorer = GraphExplorer()

# Filter by node attributes
node_filter = NodeFilter(
    search_query="Alice",
    min_degree=2,
    min_centrality=0.5
)

# Filter by edge attributes
edge_filter = EdgeFilter(
    relationship_types=["knows", "works_with"],
    min_confidence=0.7
)

# Generate filtered visualization
result = await explorer.visualize(
    triples=triples,
    entities=entities,
    node_filter=node_filter,
    edge_filter=edge_filter
)
```

### 2. Temporal Visualization

```python
from knowledge_engine.visualization import TemporalVisualizer
from datetime import datetime, timedelta

temporal_viz = TemporalVisualizer()

# Prepare temporal data
timestamps = [
    datetime.utcnow() - timedelta(days=i)
    for i in range(10, 0, -1)
]

# Generate temporal visualization
result = await temporal_viz.visualize_temporal(
    triples=triples,
    timestamps=timestamps
)
```

### 3. Community Visualization

```python
from knowledge_engine.visualization import CommunityVisualizer, CommunityVisualizationOptions

community_viz = CommunityVisualizer()

options = CommunityVisualizationOptions(
    layout_algorithm="force_community",
    show_inter_community_edges=True,
    enable_community_filtering=True
)

result = await community_viz.visualize_communities(
    triples=triples,
    entities=entities,
    options=options
)
```

### 4. Export Visualization

```python
from knowledge_engine.visualization import ExportHandler

exporter = ExportHandler()

# Export as SVG
svg_path = await exporter.export_svg(
    graph_data=graph_data,
    output_path="output.svg",
    width=1200,
    height=800
)

# Export as HTML
html_path = await exporter.export_html(
    graph_data=graph_data,
    output_path="output.html",
    embed_data=True
)

# Export as GraphML (Gephi format)
graphml_path = await exporter.export_graphml(
    triples=triples,
    output_path="output.graphml"
)
```

### 5. Generate Embedding URL

```python
from knowledge_engine.visualization import ExportHandler

exporter = ExportHandler()

# Generate embeddable URL
embed_url = exporter.generate_embedding_url(
    graph_data=graph_data,
    base_url="https://your-domain.com",
    config={
        "width": 800,
        "height": 600,
        "show_labels": True
    }
)

print(f"Embed URL: {embed_url}")
```

## API Reference

### GraphExplorer

```python
class GraphExplorer:
    async def visualize(
        self,
        triples: List[Any],
        entities: List[Any],
        output_path: Optional[str] = None,
        node_filter: Optional[NodeFilter] = None,
        edge_filter: Optional[EdgeFilter] = None,
        options: Optional[VisualizationOptions] = None,
        use_cache: bool = True
    ) -> VisualizationResult
```

### TemporalVisualizer

```python
class TemporalVisualizer:
    async def visualize_temporal(
        self,
        triples: List[Any],
        timestamps: List[datetime],
        output_path: Optional[str] = None,
        options: Optional[TemporalVisualizationOptions] = None
    ) -> Dict[str, Any]
```

### CommunityVisualizer

```python
class CommunityVisualizer:
    async def visualize_communities(
        self,
        triples: List[Any],
        entities: List[Any],
        output_path: Optional[str] = None,
        options: Optional[CommunityVisualizationOptions] = None
    ) -> Dict[str, Any]
```

### ExportHandler

```python
class ExportHandler:
    async def export_png(...) -> str
    async def export_svg(...) -> str
    async def export_html(...) -> str
    async def export_graphml(...) -> str
    async def export_gexf(...) -> str
    async def export_json(...) -> str
    def generate_embedding_url(...) -> str
```

## Design Principles (CLAUDE.md)

### 1. AIR GAP (Source Code Isolation)
- No direct imports from `./core-projects/`
- All visualization code is self-contained
- No dependencies on external project internals

### 2. RUNTIME TRUTH (Anti-Hallucination)
- All visualizations tested with real graph data
- Validation of inputs before processing
- Error handling for malformed data

### 3. IDEMPOTENCY (The Replayability Pact)
- Visualization generation is safe to retry
- Cache keys based on deterministic hashes
- No side effects from multiple calls

### 4. CONFIGURATION EXPLICITNESS
- All config via environment variables
- No magic defaults
- Crashes immediately if config is invalid

### 5. UTC TIME
- All timestamps in UTC
- Consistent time handling across components
- ISO-8601 format for storage

### 6. STRUCTURED LOGGING
- JSON logs for all operations
- Include correlation_id for tracking
- Timestamps in UTC

## Performance

### Caching
- Visualizations cached by content hash
- TTL-based cache invalidation
- Configurable cache directory

### Scalability
- Maximum node/edge limits enforced
- Graph truncation for large datasets
- Efficient algorithms (NetworkX)

### Optimization
- Lazy loading of D3.js
- Virtualization for large graphs (future)
- Progressive rendering (future)

## Testing

```bash
# Run all tests
pytest knowledge_engine/visualization/tests/

# Run specific test class
pytest knowledge_engine/visualization/tests/test_visualization.py::TestGraphExplorer

# Run with coverage
pytest --cov=knowledge_engine/visualization knowledge_engine/visualization/tests/
```

## Quality Standards

- **Responsive Design**: Mobile, tablet, desktop support
- **Performance**: Lazy loading, virtualization
- **Accessibility**: WCAG 2.1 AA compliance
- **Cross-browser**: Chrome, Firefox, Safari, Edge
- **Error Boundaries**: Graceful failures
- **Loading States**: Progress indicators
- **Export Quality**: High-resolution PNG, vector SVG

## Troubleshooting

### Visualization fails to load
1. Check D3.js is accessible (CDN or local)
2. Verify graph data is valid JSON
3. Check browser console for errors

### Graph appears empty
1. Verify triples have correct format
2. Check that nodes/edges exist
3. Try without filters first

### Performance issues
1. Reduce number of nodes/edges
2. Disable animations
3. Use simpler layout algorithm

## Contributing

When contributing to the visualization system:

1. Follow CLAUDE.md principles
2. Add tests for new features
3. Update documentation
4. Ensure accessibility compliance
5. Test with real graph data

## License

Part of the OpenEvolve project.

## Support

For issues and questions:
- GitHub Issues: https://github.com/openevolve/openevolve/issues
- Documentation: https://docs.openevolve.ai/visualization
