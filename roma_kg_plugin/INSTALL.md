# ROMA Knowledge Graph Plugin - Installation Guide

## Overview

This plugin adds knowledge graph visualization, analytics, and exploration capabilities to ROMA without modifying ROMA core files. It follows the **Air Gap principle** from CLAUDE.md - zero coupling to ROMA internals.

## Features

- **Knowledge Graph Panel**: Interactive graph visualization in ROMA TUI
- **Analytics Dashboard**: Graph metrics, centrality rankings, community statistics
- **Interactive Exploration**: Graph traversal, neighborhood exploration, path finding
- **8 Custom Commands**: Full command-line interface for graph operations
- **Hierarchical Menu**: Integrated TUI menu system
- **ASCII Graph Rendering**: Text-based graph visualization for terminals

## Prerequisites

- ROMA TUI installed and working
- Python 3.9+
- Required dependencies:
  ```bash
  pip install networkx loguru prompt-toolkit
  pip install asciichartpy  # Optional: for charts
  ```

## Installation

### Step 1: Copy Plugin to ROMA Plugins Directory

```bash
# Create ROMA plugins directory if it doesn't exist
mkdir -p ROMA/plugins

# Copy plugin
cp -r frontend/roma_kg_plugin ROMA/plugins/roma_kg_plugin
```

### Step 2: Register Plugin in ROMA Config

Create or edit `ROMA/config/plugins.yaml`:

```yaml
plugins:
  - name: roma_kg_plugin
    enabled: true
    priority: 10
    config:
      # Path to knowledge engine (relative to ROMA root)
      knowledge_engine_path: "../knowledge_engine"

      # Graph rendering options
      rendering:
        default_width: 80
        default_height: 20
        show_labels: true

      # Analytics options
      analytics:
        enable_charts: true
        cache_metrics: true
        cache_ttl: 300  # seconds

      # Export options
      export:
        default_format: json
        output_directory: "./exports"
```

### Step 3: ROMA Will Auto-Load Plugin

ROMA's plugin system will automatically:
1. Load the plugin at startup
2. Call `create_plugin()` factory function
3. Initialize the plugin with ROMA client
4. Register all panels, commands, and menus
5. Make them available in TUI

## Usage

Once installed, the plugin adds:

### New Panels

**Knowledge Graph Panel** (`/panel knowledge_graph`)
- Interactive graph visualization
- Node/edge details display
- Community browser
- Search and filter functionality
- Export capabilities

**Analytics Dashboard** (`/panel analytics`)
- Graph metrics (density, clustering, components)
- Centrality rankings (degree, betweenness, closeness)
- Community statistics
- Temporal evolution charts
- Performance metrics

### New Commands (prefix: `/kg`)

```bash
# Search knowledge graph
/kg search <query>

# Explore neighborhood around a node
/kg explore <entity_id> [depth]

# Find shortest path between nodes
/kg path <from_entity> <to_entity>

# List and browse communities
/kg communities [limit]

# Show comprehensive statistics
/kg stats

# Export graph data
/kg export <format> [output_path]
# Formats: json, gexf, csv

# Show temporal timeline for entity
/kg timeline <entity_id>

# Run graph analysis
/kg analyze <type>
# Types: centrality, community, connectivity, components
```

### New Menu

**Knowledge Graph Menu** - Access via TUI menu system:
- Visualization
- Analytics
- Exploration
- Export
- Settings

## Configuration

### Plugin Config

Edit `ROMA/config/plugins.yaml`:

```yaml
plugins:
  - name: roma_kg_plugin
    enabled: true
    priority: 10
    config:
      # Knowledge engine integration
      knowledge_engine_path: "../knowledge_engine"

      # Graph visualization settings
      visualization:
        max_nodes: 1000
        max_edges: 5000
        layout_algorithm: "spring"  # spring, circular, random
        node_size: "centrality"     # centrality, degree, uniform

      # Analytics settings
      analytics:
        enable_temporal: true
        cache_size: 100
        update_interval: 60  # seconds

      # Export settings
      export:
        formats: ["json", "gexf", "csv"]
        compression: false
        include_metadata: true
```

### Environment Variables

Optional environment variables:

```bash
# Knowledge engine location
export ROMA_KG_ENGINE_PATH="/path/to/knowledge/engine"

# Plugin log level
export ROMA_KG_LOG_LEVEL="INFO"

# Graph cache directory
export ROMA_KG_CACHE_DIR="/path/to/cache"
```

## Verification

Test plugin installation:

```bash
# Start ROMA
cd ROMA
python -m roma_dspy

# In ROMA TUI, check plugin status
/plugin status

# Try knowledge graph commands
/kg stats
/kg search test

# Open panels
/panel knowledge_graph
/panel analytics
```

## Troubleshooting

### Plugin Not Loading

Check ROMA logs:
```bash
tail -f ROMA/logs/roma.log | grep roma_kg_plugin
```

Common issues:
1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Wrong path**: Check `knowledge_engine_path` in config
3. **Permission denied**: Ensure ROMA has read access to plugin directory

### Commands Not Registered

Verify plugin initialization:
```python
# In ROMA Python REPL
from roma_kg_plugin import create_plugin
plugin = create_plugin()
print(plugin.get_info())
```

### Panel Display Issues

Check terminal compatibility:
- Minimum terminal size: 80x24
- Requires UTF-8 support
- Color support recommended

## Uninstallation

```bash
# Remove plugin directory
rm -rf ROMA/plugins/roma_kg_plugin

# Remove plugin entry from ROMA/config/plugins.yaml
# Edit the file and remove the roma_kg_plugin section

# Clear plugin cache (if any)
rm -rf ROMA/.cache/plugins/roma_kg_plugin
```

## Upgrading

```bash
# Backup existing config
cp ROMA/config/plugins.yaml ROMA/config/plugins.yaml.bak

# Remove old plugin
rm -rf ROMA/plugins/roma_kg_plugin

# Copy new version
cp -r frontend/roma_kg_plugin ROMA/plugins/roma_kg_plugin

# Restore config if needed
cp ROMA/config/plugins.yaml.bak ROMA/config/plugins.yaml
```

## Compliance with CLAUDE.md

This plugin follows all 6 Immutable Laws:

✅ **Air Gap Principle**: No imports from ROMA core, zero coupling
✅ **Runtime Truth**: Validates configuration at startup
✅ **Untouchable DB**: Read-only access to knowledge graph
✅ **Idempotency**: Safe to install/uninstall multiple times
✅ **Configuration Explicitness**: All settings via YAML/ENV
✅ **UTC**: All timestamps in UTC ISO-8601

## Support

For issues or questions:
1. Check ROMA logs: `ROMA/logs/roma.log`
2. Enable debug logging: Set `ROMA_KG_LOG_LEVEL=DEBUG`
3. Run plugin self-test: `/kg diagnose` (if available)

## Architecture

```
roma_kg_plugin/
├── __init__.py              # Plugin entry point
├── plugin.py                # Main plugin class
├── panels/                  # TUI panels
│   ├── knowledge_graph_panel.py
│   └── analytics_dashboard.py
├── interactive/             # Interactive exploration
│   └── exploration.py
├── visualization/           # ASCII rendering
│   └── ascii_graph.py
├── commands/                # Command handlers
│   └── kg_commands.py
├── menus/                   # Menu definitions
│   └── kg_menu.py
├── integration/             # Knowledge engine integration
│   └── knowledge_integration.py
├── tests/                   # Plugin tests
│   └── test_plugin.py
└── examples/                # Usage examples
    └── plugin_demo.py
```

All components use dependency injection - no direct ROMA core imports.
