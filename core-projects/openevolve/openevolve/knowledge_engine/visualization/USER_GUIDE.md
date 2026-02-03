# Graph Explorer User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Filtering Data](#filtering-data)
4. [Navigation Controls](#navigation-controls)
5. [Exporting Visualizations](#exporting-visualizations)
6. [Advanced Features](#advanced-features)
7. [Tips and Tricks](#tips-and-tricks)

## Getting Started

### Opening the Graph Explorer

1. Navigate to the visualization page in your browser
2. Upload your graph data (JSON/CSV format) or connect to the API
3. Click "Load Graph" to visualize your data

### Understanding the Display

The graph explorer shows:
- **Nodes**: Circles representing entities
- **Edges**: Lines representing relationships
- **Colors**: Different communities/groups
- **Sizes**: Node importance (centrality)

## Interface Overview

### Main Components

```
┌─────────────────────────────────────────────────────┐
│  Toolbar: [Filter] [Export] [Settings] [Help]       │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │                                             │    │
│  │           Graph Canvas                      │    │
│  │       (Zoom, Pan, Select)                   │    │
│  │                                             │    │
│  └─────────────────────────────────────────────┘    │
│                                                       │
├─────────────────────┬───────────────────────────────┤
│  Filters Panel     │  Details Panel                 │
│                    │                               │
│  ○ Node Type       │  Selected: Alice              │
│  ○ Edge Type       │  Type: Person                 │
│  ○ Time Range      │  Connections: 5               │
│  ○ Confidence      │  Community: 1                 │
│                    │  Centrality: 0.85             │
│  [Apply Filters]   │                               │
└────────────────────┴───────────────────────────────┘
```

## Filtering Data

### Node Filtering

1. **By Name/Type**
   - Use the search box to filter by node name
   - Select node types from dropdown
   - Combine multiple filters

2. **By Importance**
   - Set minimum centrality score
   - Filter by degree (number of connections)
   - Range sliders for fine-tuning

3. **By Attributes**
   - Custom attribute filters
   - Exact match or contains
   - Multiple values supported

### Edge Filtering

1. **By Relationship Type**
   - Select specific predicates (knows, works_with, etc.)
   - Multi-select for multiple types
   - Toggle on/off

2. **By Confidence**
   - Minimum confidence threshold
   - Range slider for precision
   - Visual feedback on edge thickness

3. **By Source**
   - Filter by data source
   - Extracted vs Inferred
   - Custom sources

### Temporal Filtering

1. **Time Range**
   - Date range picker
   - Preset ranges (last day, week, month)
   - Custom start/end times

2. **Timeline Slider**
   - Drag to filter by time
   - Animated playback
   - Key timestamps marked

## Navigation Controls

### Zoom and Pan

- **Zoom In**: Scroll wheel up or "+" button
- **Zoom Out**: Scroll wheel down or "-" button
- **Pan**: Click and drag on canvas
- **Reset Fit**: "Fit to Screen" button

### Selection

- **Single Node**: Click on node
- **Multiple Nodes**: Shift+Click or drag selection box
- **Neighbors**: Double-click to show connected nodes
- **Clear Selection**: Click on empty space

### Layout Controls

- **Force-Directed**: Physics-based layout
- **Hierarchical**: Tree-like arrangement
- **Circular**: Nodes in circle
- **Community**: Group by community
- **Custom**: Upload your own coordinates

## Exporting Visualizations

### Export Formats

1. **PNG (Raster Image)**
   - High-resolution export
   - Adjustable DPI (72-300)
   - Background color options

2. **SVG (Vector Graphics)**
   - Scalable without quality loss
   - Editable in Illustrator/Inkscape
   - Smaller file size

3. **HTML (Interactive)**
   - Standalone file
   - Embeddable in websites
   - Full interactivity preserved

4. **GraphML/GEXF**
   - Import into Gephi
   - Import into Cytoscape
   - Network analysis

5. **JSON (Data)**
   - Raw graph data
   - D3.js format
   - Custom processing

### Export Options

```
Export Settings:
┌─────────────────────────────┐
│ Format: [PNG ▼]             │
│ Width:  [1200]              │
│ Height: [800]               │
│ DPI:    [300]               │
│                              │
│ ☑ Include labels            │
│ ☑ Include legend            │
│ ☑ Include statistics        │
│ ☐ Transparent background    │
│                              │
│      [Cancel] [Export]       │
└─────────────────────────────┘
```

## Advanced Features

### Subgraph Extraction

Extract a focused view around a node:

1. Select a node of interest
2. Right-click → "Extract Subgraph"
3. Set radius (number of hops)
4. Optionally set minimum degree
5. Click "Extract"

### Comparison Views

Compare two graphs or time points:

1. Load "before" graph
2. Load "after" graph
3. View → "Comparison Mode"
4. See highlighted differences:
   - Green: Added nodes/edges
   - Red: Removed nodes/edges
   - Gray: Unchanged

### Statistics Dashboard

View comprehensive graph statistics:

```
Graph Statistics:
┌─────────────────────────────┐
│ Nodes:              1,234   │
│ Edges:              5,678   │
│ Density:            0.007   │
│ Connected:          Yes     │
│                              │
│ Communities:        12      │
│ Avg Community Size: 103     │
│ Modularity:         0.65    │
│                              │
│ Avg Degree:         9.2     │
│ Diameter:           4       │
│ Avg Clustering:     0.42    │
└─────────────────────────────┘
```

### Custom Styling

Personalize your visualization:

1. **Color Schemes**
   - Colorblind-friendly
   - Default
   - Spectral
   - Custom (upload palette)

2. **Node Styling**
   - Size by centrality/degree
   - Border width
   - Transparency

3. **Edge Styling**
   - Thickness by confidence
   - Solid/dashed for source
   - Color by type
   - Transparency

## Tips and Tricks

### Performance Optimization

For large graphs (>1000 nodes):
1. Start with filters applied
2. Disable animations
3. Use simpler layouts
4. Reduce label visibility
5. Enable node aggregation

### Best Practices

1. **Start Simple**
   - Load data without filters first
   - Apply filters incrementally
   - Check statistics after each filter

2. **Save Your Work**
   - Export filtered views
   - Save filter configurations
   - Bookmark important visualizations

3. **Iterate**
   - Try different layouts
   - Experiment with color schemes
   - Combine multiple filters

### Keyboard Shortcuts

- `F`: Open filter panel
- `E`: Open export dialog
- `S`: Open settings
- `+` / `-`: Zoom in/out
- `0`: Reset zoom
- `Esc`: Clear selection
- `Ctrl+F`: Focus search
- `Ctrl+E`: Export
- `Ctrl+S`: Save configuration

### Common Workflows

#### Workflow 1: Explore Community Structure

1. Load graph
2. Apply "Community Layout"
3. Enable "Show Community Labels"
4. Select a community
5. Extract subgraph
6. Export for detailed analysis

#### Workflow 2: Track Temporal Changes

1. Load temporal graph
2. Open timeline slider
3. Animate through time
4. Pause at key point
5. Compare with previous snapshot
6. Export comparison view

#### Workflow 3: Find Important Nodes

1. Load graph
2. Set node sizing to "Centrality"
3. Filter by min centrality (0.5+)
4. Sort by degree
5. Select top nodes
6. View their connections
7. Export focused view

## Troubleshooting

### Graph Not Loading

**Problem**: Graph appears empty or fails to load

**Solutions**:
1. Check data format (valid JSON/CSV)
2. Verify required fields (subject, predicate, object)
3. Check file size (max 50MB for upload)
4. Look for error messages in browser console

### Poor Layout

**Problem**: Nodes overlap or are hard to see

**Solutions**:
1. Try different layout algorithms
2. Increase canvas size
3. Adjust node spacing in settings
4. Filter to fewer nodes
5. Enable physics simulation

### Slow Performance

**Problem**: Visualization is laggy or slow

**Solutions**:
1. Reduce number of nodes/edges with filters
2. Disable animations in settings
3. Use simpler layout (hierarchical)
4. Clear browser cache
5. Use modern browser (Chrome/Firefox)

### Export Issues

**Problem**: Exported file doesn't look right

**Solutions**:
1. Increase resolution for PNG
2. Use SVG for vector graphics
3. Check "Include labels" is enabled
4. Try different export format
5. Verify sufficient disk space

## Getting Help

- **Documentation**: https://docs.openevolve.ai/visualization
- **Tutorials**: https://docs.openevolve.ai/visualization/tutorials
- **API Reference**: https://docs.openevolve.ai/visualization/api
- **GitHub Issues**: https://github.com/openevolve/openevolve/issues
- **Community Forum**: https://community.openevolve.ai

## Video Tutorials

- [Getting Started](https://docs.openevolve.ai/visualization/videos/getting-started)
- [Advanced Filtering](https://docs.openevolve.ai/visualization/videos/filtering)
- [Export Guide](https://docs.openevolve.ai/visualization/videos/export)
- [Temporal Analysis](https://docs.openevolve.ai/visualization/videos/temporal)

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for version history and new features.
