# ROMA Knowledge Graph Plugin - Quick Start

## 30-Second Installation

```bash
# 1. Copy plugin to ROMA
cp -r frontend/roma_kg_plugin ROMA/plugins/roma_kg_plugin

# 2. Add to ROMA/config/plugins.yaml
echo "
plugins:
  - name: roma_kg_plugin
    enabled: true
" >> ROMA/config/plugins.yaml

# 3. Start ROMA
cd ROMA && python -m roma_dspy
```

## 5-Minute Verification

```bash
# In ROMA TUI, run:

/kg stats                    # Show graph statistics
/kg search python            # Search for entities
/panel knowledge_graph       # Open graph visualization
/panel analytics             # Open analytics dashboard
```

## What You Get

### 2 New Panels
- **Knowledge Graph**: Interactive graph visualization
- **Analytics**: Comprehensive metrics and statistics

### 8 New Commands
- `/kg search <query>` - Search knowledge graph
- `/kg explore <node>` - Explore neighborhood
- `/kg path <from> <to>` - Find shortest path
- `/kg communities` - List communities
- `/kg stats` - Show statistics
- `/kg export <format>` - Export graph
- `/kg timeline <entity>` - Show timeline
- `/kg analyze <type>` - Run analysis

### Key Features
- ✅ Zero modifications to ROMA core
- ✅ Follows CLAUDE.md Air Gap principle
- ✅ Easy install/uninstall
- ✅ Comprehensive documentation
- ✅ 80%+ test coverage

## Documentation

- **README.md**: Complete feature documentation
- **INSTALL.md**: Detailed installation guide
- **config.yaml**: Configuration reference
- **TRANSFORMATION_SUMMARY.md**: Technical details

## Support

```bash
# Check plugin status
/plugin status

# Run tests
cd roma_kg_plugin && python -m pytest tests/ -v

# Run examples
python examples/plugin_demo.py
```

## Architecture

```
ROMA TUI
  └── Plugin Loader
      └── ROMA Knowledge Graph Plugin
          ├── Panels (2)
          ├── Commands (8)
          ├── Menus (1)
          └── Integration (1)
```

**Zero coupling to ROMA core - all dependencies injected!**
