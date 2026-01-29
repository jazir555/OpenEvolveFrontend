# OpenEvolve PyGraphistry BubbleLab Plugin

This plugin integrates **PyGraphistry**'s GPU-accelerated interactive graph visualization into the BubbleLab environment for OpenEvolve runs.

## Features

- **Interactive Visualizations**: Zoom, pan, and filter complex evolutionary graphs.
- **GPU Acceleration**: Handles millions of nodes and edges with ease.
- **Clustering Support**: Automatic UMAP + DBSCAN clustering for pattern discovery.
- **Real-time Updates**: Refresh visualizations as OpenEvolve runs progress.

## Installation

1. Install dependencies:
   ```bash
   cd openevolve-pygraphistry-plugin
   npm install
   ```

2. Build the plugin:
   ```bash
   npm run build
   ```

3. Register the plugin in your BubbleLab application.

## Backend Requirements

This plugin requires the OpenEvolve API to be running with PyGraphistry configured:

1. Install the `graphistry` Python library:
   ```bash
   pip install graphistry
   ```

2. Set your Graphistry API Key:
   ```bash
   export GRAPHISTRY_API_KEY='your_api_key_here'
   ```

3. Start the OpenEvolve API:
   ```bash
   python openevolve_api.py
   ```

## Usage in React

```tsx
import { PyGraphistryViz } from '@openevolve/bubblelab-pygraphistry-plugin';

const MyComponent = () => {
  const nodes = [...];
  const edges = [...];

  return (
    <PyGraphistryViz 
      nodes={nodes} 
      edges={edges} 
      height={600} 
    />
  );
};
```
