# Knowledge Engine Analytics Integration

## Overview

This document describes how the advanced analytics tools (Karate Club, PAMI, NeuralKG, Causal-Learn, Lagrange-Mapper) are integrated into the OpenEvolve Knowledge Engine.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Engine Core                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Knowledge   │  │  Knowledge   │  │  Knowledge   │          │
│  │  Extractor   │  │    Graph     │  │   Storage    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                 │                                      │
│         └────────┬────────┘                                      │
│                  │                                               │
│  ┌───────────────▼────────────────┐                             │
│  │  KnowledgeEngineIntegration    │                             │
│  │         Connector              │                             │
│  └───────────────┬────────────────┘                             │
└──────────────────┼──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              Advanced Analytics Engine                           │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │   Karate     │     PAMI     │   NeuralKG   │  Causal-Learn│  │
│  │    Club      │              │              │              │  │
│  │  (Graph      │   (Pattern   │  (KG Embeds) │    (Causal   │  │
│  │ Communities) │    Mining)   │              │  Discovery)  │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Lagrange-Mapper                          │  │
│  │              (Topological Analysis)                       │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Knowledge Extractor → PAMI (Pattern Mining)

**Location**: `knowledge_engine/advanced_analytics_engine.py::mine_artifact_patterns()`

**Purpose**: Mine patterns from extracted knowledge artifacts

**Usage**:
```python
from knowledge_engine import KnowledgeEngineIntegrationConnector

connector = KnowledgeEngineIntegrationConnector()
artifacts = [...]  # KnowledgeArtifact objects

result = connector.analyze_artifacts_with_pattern_mining(
    artifacts=artifacts,
    pattern_type='frequent'  # or 'sequential', 'association'
)
```

**Features**:
- Frequent pattern mining from artifact types/sources
- Sequential pattern discovery
- Association rule generation

### 2. Knowledge Graph → Karate Club (Community Detection)

**Location**: `knowledge_engine/advanced_analytics_engine.py::analyze_knowledge_graph_communities()`

**Purpose**: Detect communities in knowledge graphs

**Usage**:
```python
graph_data = {
    'nodes': [{'id': 'A', 'type': 'Person'}, ...],
    'edges': [{'source': 'A', 'target': 'B', 'type': 'knows'}, ...]
}

result = connector.detect_graph_communities(graph_data)
```

**Features**:
- Louvain community detection
- Leiden algorithm
- Label propagation
- Overlapping community detection (BigClam, CFinder)
- Node embeddings (Node2Vec, DeepWalk, GraphSAGE)
- Graph embeddings (Graph2Vec, SF)

### 3. Knowledge Graph → NeuralKG (Embeddings)

**Location**: `knowledge_engine/advanced_analytics_engine.py::generate_knowledge_embeddings()`

**Purpose**: Generate knowledge graph embeddings

**Usage**:
```python
triples = [
    ('Alice', 'knows', 'Bob'),
    ('Bob', 'works_for', 'AcmeCorp')
]

result = connector.generate_graph_embeddings(
    triples=triples,
    model='transe'  # or 'rotate', 'complex', 'distmult'
)
```

**Features**:
- Multiple embedding models (TransE, RotatE, ComplEx, DistMult)
- Link prediction
- Entity similarity search
- Ensemble embeddings

### 4. Knowledge Metrics → Causal-Learn (Causal Discovery)

**Location**: `knowledge_engine/advanced_analytics_engine.py::discover_causal_relationships()`

**Purpose**: Discover causal relationships in knowledge metrics

**Usage**:
```python
import numpy as np

# Example: Analyze how node properties relate causally
data = np.array([
    [5, 0.5, 0.3],  # degree, betweenness, clustering
    [3, 0.2, 0.5],
    [4, 0.3, 0.4]
])

result = connector.discover_causal_structure(
    data=data,
    variable_names=['degree', 'betweenness', 'clustering'],
    algorithm='pc'  # or 'fci', 'ges', 'lingam'
)
```

**Features**:
- PC algorithm (constraint-based)
- FCI (handles latent confounders)
- GES (score-based)
- LiNGAM (non-Gaussian causal models)
- Confounder identification

### 5. Knowledge Embeddings → Lagrange-Mapper (Topological Analysis)

**Location**: `knowledge_engine/advanced_analytics_engine.py::analyze_embedding_landscape()`

**Purpose**: Analyze topological structure of knowledge embeddings

**Usage**:
```python
import numpy as np

embeddings = np.random.randn(100, 50)  # 100 entities, 50 dimensions

result = connector.analyze_knowledge_landscape(
    embeddings=embeddings,
    labels=['entity_1', 'entity_2', ...]
)
```

**Features**:
- Attractor landscape mapping
- Cluster identification
- Knowledge graph topology analysis
- Attractor basin computation
- Landscape transition detection

## Complete Integration Example

```python
from knowledge_engine import KnowledgeEngineIntegrationConnector
from knowledge_engine.knowledge_extractor import KnowledgeExtractor

# Initialize connector
connector = KnowledgeEngineIntegrationConnector()

# Extract knowledge from workflow
extractor = KnowledgeExtractor()
artifacts = extractor.extract_from_workflow(workflow_data)

# 1. Mine patterns from artifacts
pattern_result = connector.analyze_artifacts_with_pattern_mining(
    artifacts=artifacts,
    pattern_type='frequent'
)

# 2. Build knowledge graph from artifacts
graph_data = {
    'nodes': [...],
    'edges': [...]
}

# 3. Comprehensive graph analysis
analysis = connector.analyze_graph_comprehensive(graph_data)

# 4. Generate embeddings
triples = [(e['source'], e['type'], e['target']) for e in graph_data['edges']]
emb_result = connector.generate_graph_embeddings(triples)

# 5. Analyze embedding landscape
if emb_result['status'] == 'success':
    embeddings = list(emb_result['embeddings']['entities'].values())
    landscape = connector.analyze_knowledge_landscape(embeddings)
```

## File Structure

```
knowledge_engine/
├── integrations/                          # Integration modules
│   ├── karateclub_integration.py         # Karate Club integration
│   ├── pami_integration.py               # PAMI integration
│   ├── neuralkg_integration.py           # NeuralKG integration
│   ├── causal_learn_integration.py       # Causal-Learn integration
│   ├── lagrange_mapper_integration.py    # Lagrange-Mapper integration
│   └── __init__.py                       # Exports all integrations
├── advanced_analytics_engine.py          # Analytics engine
├── KNOWLEDGE_ENGINE_INTEGRATION_         # Connector module
│   CONNECTOR.py
└── ANALYTICS_INTEGRATION_README.md       # This file
```

## Configuration

```python
config = {
    'karateclub': {'enabled': True},
    'pami': {'enabled': True, 'min_support': 0.1},
    'neuralkg': {'enabled': True, 'embedding_dim': 100},
    'causal_learn': {'enabled': True, 'alpha': 0.05},
    'lagrange_mapper': {'enabled': True, 'n_clusters': 8}
}

connector = KnowledgeEngineIntegrationConnector(config)
```

## Dependencies

Each integration has its own dependencies:

- **Karate Club**: `networkx`, `numpy`
- **PAMI**: Pure Python fallback (or `PAMI` library)
- **NeuralKG**: `torch`, `numpy`
- **Causal-Learn**: `numpy`, `networkx`
- **Lagrange-Mapper**: `numpy`, `scikit-learn` (optional)

## Error Handling

All methods return standardized results:

```python
{
    'status': 'success' | 'error',
    'data': {...},           # Analysis results
    'metadata': {...},       # Additional info
    'errors': [...]          # Error messages if any
}
```

## Testing

```python
# Test all integrations
from knowledge_engine.tests.test_new_integrations import run_tests
run_tests()

# Or use pytest
pytest knowledge_engine/tests/test_new_integrations.py
```

## Usage in IntegratedKnowledgeEngine

The `AdvancedAnalyticsEngine` can be used within the `IntegratedKnowledgeEngine`:

```python
from knowledge_engine.integrated_engine import IntegratedKnowledgeEngine

engine = IntegratedKnowledgeEngine()

# Access analytics through the engine
if hasattr(engine, 'analytics'):
    result = engine.analytics.analyze_knowledge_graph_communities(graph_data)
```

## Future Enhancements

1. **Caching**: Cache embedding results for reuse
2. **Distributed Processing**: Parallelize analytics across clusters
3. **Visualization**: Add visualization for all analytics
4. **Real-time Analysis**: Stream processing for live knowledge graphs
5. **Auto-tuning**: Automatic parameter tuning for algorithms
