# Visualization System - Quick Reference

## Quick Start

```python
# 1. Basic Graph Visualization
from knowledge_engine.visualization import GraphExplorer

explorer = GraphExplorer()
result = await explorer.visualize(triples, entities)
print(f"Visualization: {result.output_path}")

# 2. Temporal Visualization
from knowledge_engine.visualization import TemporalVisualizer

temporal = TemporalVisualizer()
result = await temporal.visualize_temporal(triples, timestamps)

# 3. Community Visualization
from knowledge_engine.visualization import CommunityVisualizer

community = CommunityVisualizer()
result = await community.visualize_communities(triples, entities)

# 4. Export
from knowledge_engine.visualization import ExportHandler

exporter = ExportHandler()
path = await exporter.export_svg(graph_data, "output.svg")
```

## API Endpoints

```bash
# Create visualization
POST /api/visualization/graph
{"triples": [...], "options": {"width": 1200}}

# Export
POST /api/visualization/export
{"format": "svg", "graph_data": {...}}

# Statistics
POST /api/visualization/statistics
{"triples": [...]}
```

## Environment Variables

```bash
VISUALIZATION_OUTPUT_DIR=data/visualizations
VISUALIZATION_MAX_NODES=10000
VISUALIZATION_MAX_EDGES=50000
VISUALIZATION_CACHE_TTL=3600
VISUALIZATION_ENABLE_CACHING=true
```

## Common Tasks

### Filter Nodes
```python
from knowledge_engine.visualization import NodeFilter

filter = NodeFilter(
    search_query="Alice",
    min_degree=2,
    min_centrality=0.5
)
```

### Filter Edges
```python
from knowledge_engine.visualization import EdgeFilter

filter = EdgeFilter(
    relationship_types=["knows"],
    min_confidence=0.8
)
```

### Export Formats
- PNG: Raster image (high-res)
- SVG: Vector graphics (scalable)
- HTML: Interactive (standalone)
- GraphML: Gephi import
- JSON: Raw data

## Testing

```bash
# Run all tests
pytest knowledge_engine/visualization/tests/

# Run specific test
pytest knowledge_engine/visualization/tests/test_visualization.py::TestGraphExplorer

# With coverage
pytest --cov=knowledge_engine/visualization
```

## Documentation

- **README.md**: Full API documentation
- **USER_GUIDE.md**: User manual
- **examples/example_usage.py**: 9 working examples
- **SPRINT_4_COMPLETION_REPORT.md**: Implementation details

## File Structure

```
knowledge_engine/visualization/
├── config.py              # Configuration
├── graph_explorer.py      # Interactive explorer
├── temporal_viz.py        # Temporal visualization
├── community_viz.py       # Community visualization
├── export_handlers.py     # Export functionality
├── api.py                 # REST API
├── tests/                 # Test suite
├── examples/              # Usage examples
├── README.md              # Technical docs
└── USER_GUIDE.md         # User guide
```

## Support

- GitHub: https://github.com/openevolve/openevolve
- Docs: https://docs.openevolve.ai/visualization
- Issues: https://github.com/openevolve/openevolve/issues
