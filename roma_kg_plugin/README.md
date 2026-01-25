# ROMA Knowledge Graph Plugin

A plugin for ROMA that adds knowledge graph visualization, analytics, and exploration capabilities without modifying ROMA core files.

## Overview

This plugin extends ROMA's terminal user interface (TUI) with powerful knowledge graph features:

- **Interactive Visualization**: ASCII-based graph rendering for terminal displays
- **Analytics Dashboard**: Comprehensive graph metrics and statistics
- **Exploration Tools**: Graph traversal, neighborhood exploration, path finding
- **Command Interface**: 8 custom commands for all graph operations
- **Integration**: Seamless connection to knowledge engine

## Architecture

The plugin follows the **Air Gap principle** - zero coupling to ROMA core files. All dependencies are injected through the plugin system.

```
┌─────────────────────────────────────────────────────┐
│                   ROMA TUI                          │
│                                                       │
│  ┌──────────────────────────────────────────────┐  │
│  │         Plugin Loader (ROMA Core)            │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                                │
│  ┌──────────────────▼───────────────────────────┐  │
│  │     ROMA Knowledge Graph Plugin             │  │
│  │                                              │  │
│  │  ┌─────────┐  ┌──────────┐  ┌────────────┐  │  │
│  │  │ Panels  │  │ Commands │  │   Menus    │  │  │
│  │  └─────────┘  └──────────┘  └────────────┘  │  │
│  │                                              │  │
│  │  ┌────────────────────────────────────────┐ │  │
│  │  │   Knowledge Engine Integration         │ │  │
│  │  └────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Features

### 1. Knowledge Graph Panel

Visualize knowledge graphs directly in the terminal:

```
┌─ Knowledge Graph Explorer ──────────────────────────┐
├──────────────┬──────────────────────────────────────┤
│ Graph Stats  │ Interactive Graph                   │
│              │                                      │
│ - Nodes: 123 │     [Node A]───→[Node B]            │
│ - Edges: 456 │        │         │                   │
│ - Comm: 12   │        ↓         ↓                   │
│              │     [Node C]   [Node D]              │
├──────────────┴──────────────────────────────────────┤
│ Search: [______________] [Filter] [Export]           │
│ Selected: Node A - Related to B, C                   │
└──────────────────────────────────────────────────────┘
```

**Features:**
- Interactive node selection
- Community browsing
- Real-time search and filter
- Export to multiple formats (JSON, GEXF, CSV)

### 2. Analytics Dashboard

Comprehensive graph analytics and visualizations:

```
┌─ Analytics Dashboard ────────────────────────────────┐
├──────────────┬──────────────────────────────────────┤
│ Metrics      │ Visualizations                       │
│              │                                      │
│ Density: 0.23│ ▂▄▆█▆▄▂ (Node Degree Dist)            │
│ Clustering:  │ ████▇▆▅▄▂ (Centrality)               │
│   0.45       │                                      │
│              │ ▅▆▇████▅▆ (Community Size)           │
│ Components:5 │                                      │
├──────────────┴──────────────────────────────────────┤
│ [Refresh] [Export] [Detailed Analysis]              │
└──────────────────────────────────────────────────────┘
```

**Features:**
- Graph metrics (density, clustering, components)
- Centrality rankings (degree, betweenness, closeness)
- Community statistics and modularity
- Temporal evolution charts
- Performance metrics

### 3. Command Interface

Eight powerful commands for knowledge graph operations:

```bash
# Search the knowledge graph
/kg search python

# Explore neighborhood around an entity
/kg explore "Python Programming" 2

# Find shortest path between entities
/kg path "Python" "Machine Learning"

# List communities
/kg communities 10

# Show graph statistics
/kg stats

# Export graph
/kg export json output.json

# Show temporal timeline
/kg timeline "Python Programming"

# Run analysis
/kg analyze centrality
```

### 4. Interactive Exploration

Advanced graph traversal and exploration:

- **Neighborhood Exploration**: Explore 1-hop, 2-hop, or n-hop neighborhoods
- **Path Finding**: Find shortest paths between any two entities
- **Graph Traversal**: BFS/DFS traversal with filtering
- **Node Expansion**: Expand/collapse nodes to manage complexity

### 5. Knowledge Engine Integration

Seamless integration with the knowledge engine:

- **Query Interface**: Direct access to knowledge graph queries
- **Real-time Updates**: Live updates as knowledge evolves
- **Caching**: Intelligent caching for performance
- **Export**: Multiple export formats for external analysis

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

Quick start:

```bash
# Copy plugin to ROMA
cp -r roma_kg_plugin ROMA/plugins/

# Register in ROMA/config/plugins.yaml
# ROMA will auto-load the plugin
```

## Usage

### Basic Usage

```bash
# Start ROMA
python -m roma_dspy

# Open knowledge graph panel
/panel knowledge_graph

# Run commands
/kg search "python"
/kg stats
```

### Advanced Usage

```bash
# Explore neighborhood with depth
/kg explore "Python" 3

# Find path with visualization
/kg path "Python" "Data Science" --visualize

# Export with specific format and options
/kg export gexf --include-communities --compress

# Run multiple analyses
/kg analyze centrality --top-k 50
/kg analyze community --algorithm louvain
```

## Configuration

Plugin configuration in `ROMA/config/plugins.yaml`:

```yaml
plugins:
  - name: roma_kg_plugin
    enabled: true
    priority: 10
    config:
      knowledge_engine_path: "../knowledge_engine"

      # Visualization settings
      visualization:
        max_nodes: 1000
        layout_algorithm: "spring"

      # Analytics settings
      analytics:
        enable_charts: true
        cache_metrics: true

      # Export settings
      export:
        formats: ["json", "gexf", "csv"]
```

## Architecture Details

### Directory Structure

```
roma_kg_plugin/
├── __init__.py              # Plugin entry point
├── plugin.py                # Main plugin class
├── panels/                  # TUI panels
│   ├── __init__.py
│   ├── knowledge_graph_panel.py
│   └── analytics_dashboard.py
├── interactive/             # Interactive exploration
│   ├── __init__.py
│   └── exploration.py
├── visualization/           # ASCII rendering
│   ├── __init__.py
│   └── ascii_graph.py
├── commands/                # Command handlers
│   ├── __init__.py
│   └── kg_commands.py
├── menus/                   # Menu definitions
│   ├── __init__.py
│   └── kg_menu.py
├── integration/             # Knowledge engine integration
│   ├── __init__.py
│   └── knowledge_integration.py
├── tests/                   # Plugin tests
│   ├── __init__.py
│   └── test_plugin.py
├── examples/                # Usage examples
│   ├── __init__.py
│   └── plugin_demo.py
├── README.md                # This file
└── INSTALL.md               # Installation guide
```

### Key Principles

1. **Air Gap Compliance**: No direct imports from ROMA core
2. **Dependency Injection**: All dependencies injected through constructors
3. **Plugin Architecture**: Uses ROMA's plugin system for registration
4. **Isolation**: Plugin can be installed/removed without affecting ROMA

## Development

### Adding New Features

1. **New Panel**: Create in `panels/`, register in `plugin.py`
2. **New Command**: Add to `commands/kg_commands.py`
3. **New Menu**: Create in `menus/`, register in `plugin.py`
4. **New Integration**: Add to `integration/`

### Testing

```bash
# Run plugin tests
cd roma_kg_plugin
python -m pytest tests/

# Run with ROMA integration
cd ROMA
python -m pytest plugins/roma_kg_plugin/tests/
```

### Contributing

Contributions should:
- Follow the Air Gap principle
- Use dependency injection
- Include tests
- Update documentation

## Examples

See `examples/plugin_demo.py` for complete usage examples.

## Troubleshooting

See [INSTALL.md](INSTALL.md#troubleshooting) for troubleshooting guide.

## License

Same license as ROMA.

## Compliance

This plugin follows CLAUDE.md principles:

- ✅ **Air Gap**: No modifications to ROMA core
- ✅ **Runtime Truth**: Validates at startup
- ✅ **Idempotency**: Safe install/uninstall
- ✅ **Configuration Explicitness**: All config via YAML
- ✅ **UTC**: All timestamps in UTC ISO-8601

## Support

For issues or questions:
1. Check [INSTALL.md](INSTALL.md#troubleshooting)
2. Enable debug logging: `ROMA_KG_LOG_LEVEL=DEBUG`
3. Check ROMA logs: `ROMA/logs/roma.log`
