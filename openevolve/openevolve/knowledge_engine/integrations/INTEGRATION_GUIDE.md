# Knowledge Engine Integration Guide

This guide documents the integration of advanced AI/ML libraries into the OpenEvolve Knowledge Engine.

## Overview

The knowledge engine now integrates the following projects:

| Project | Purpose | Integration Module |
|---------|---------|-------------------|
| **Karate Club** | Graph analysis & community detection | `karateclub_integration.py` |
| **PAMI** | Pattern mining (frequent patterns, sequences) | `pami_integration.py` |
| **NeuralKG** | Knowledge graph embeddings | `neuralkg_integration.py` |
| **Causal-Learn** | Causal discovery & inference | `causal_learn_integration.py` |
| **Lagrange-Mapper** | Topological data analysis | `lagrange_mapper_integration.py` |
| **Generic Knowledge Extraction Tool** | Unified extraction interface | `unified_knowledge_extraction.py` |

## Quick Start

### Basic Usage

```python
from knowledge_engine.integrations import AIKnowledgeGraphIntegrator

# Initialize the integrator
integrator = AIKnowledgeGraphIntegrator()

# Check available integrations
status = integrator.get_integration_status()
print(status)
```

### Using Individual Modules

```python
from knowledge_engine.integrations import (
    PAMIPatternMiner,
    NeuralKGEmbedder,
    CausalDiscoveryEngine,
    LagrangeAttractorAnalyzer
)

# Pattern Mining with PAMI
miner = PAMIPatternMiner()
if miner.is_available():
    result = miner.mine_frequent_patterns(
        transactions=[['a', 'b'], ['b', 'c'], ['a', 'c']],
        min_support=0.5
    )

# Knowledge Graph Embeddings with NeuralKG
embedder = NeuralKGEmbedder()
if embedder.is_available():
    embeddings = embedder.generate_embeddings(
        triples=[('Alice', 'knows', 'Bob'), ('Bob', 'knows', 'Charlie')],
        model_name='transe'
    )

# Causal Discovery with Causal-Learn
engine = CausalDiscoveryEngine()
if engine.is_available():
    import numpy as np
    data = np.random.randn(100, 3)
    causal_graph = engine.discover_causal_structure(
        data=data,
        algorithm='pc'
    )

# Topological Analysis with Lagrange-Mapper
analyzer = LagrangeAttractorAnalyzer()
if analyzer.is_available():
    embeddings = np.random.randn(50, 10)
    landscape = analyzer.analyze_embedding_landscape(embeddings)
```

## Detailed Module Documentation

### 1. PAMI Integration (`pami_integration.py`)

**Purpose**: Pattern mining for frequent patterns, sequences, and graph patterns.

**Key Classes**:
- `PAMIPatternMiner`: Main pattern mining interface

**Methods**:

#### `mine_frequent_patterns()`
```python
result = miner.mine_frequent_patterns(
    transactions=[
        ['bread', 'milk'],
        ['bread', 'butter', 'milk'],
        ['eggs', 'milk']
    ],
    min_support=0.2,
    algorithm='fpgrowth',
    max_pattern_length=5
)
```

#### `mine_sequences()`
```python
result = miner.mine_sequences(
    sequences=[
        [['a'], ['b'], ['c']],
        [['a'], ['b'], ['d']]
    ],
    min_support=0.5,
    max_gap=2
)
```

#### `discover_association_rules()`
```python
result = miner.discover_association_rules(
    transactions=transactions,
    min_support=0.1,
    min_confidence=0.5
)
```

#### `analyze_knowledge_graph_patterns()`
```python
result = miner.analyze_knowledge_graph_patterns(
    graph_data={
        'nodes': [{'id': 'A', 'type': 'Person'}],
        'edges': [{'source': 'A', 'target': 'B', 'type': 'knows'}]
    },
    min_support=0.1
)
```

### 2. NeuralKG Integration (`neuralkg_integration.py`)

**Purpose**: Generate knowledge graph embeddings using various models.

**Supported Models**:
- `transe`: Translating Embeddings
- `rotate`: Rotation in Complex Space
- `complex`: Complex Embeddings
- `distmult`: DistMult
- `rgcn`: Relational GCN
- `compgcn`: Composition-based GCN

**Key Classes**:
- `NeuralKGEmbedder`: Main embedding interface

**Methods**:

#### `generate_embeddings()`
```python
result = embedder.generate_embeddings(
    triples=[
        ('Alice', 'knows', 'Bob'),
        ('Bob', 'works_for', 'AcmeCorp')
    ],
    model_name='transe',
    embedding_dim=100,
    epochs=100
)
```

#### `predict_links()`
```python
predictions = embedder.predict_links(
    head='Alice',
    relation='knows',
    candidate_tails=['Bob', 'Charlie', 'Dave'],
    embeddings=embeddings,
    top_k=5
)
```

#### `find_similar_entities()`
```python
similar = embedder.find_similar_entities(
    entity='Alice',
    embeddings=embeddings,
    top_k=10
)
```

#### `ensemble_embeddings()`
```python
ensemble = embedder.ensemble_embeddings(
    triples=triples,
    models=['transe', 'complex'],
    embedding_dim=100
)
```

### 3. Causal-Learn Integration (`causal_learn_integration.py`)

**Purpose**: Discover causal relationships from data.

**Supported Algorithms**:
- `pc`: Peter-Clark algorithm
- `fci`: Fast Causal Inference (handles latent variables)
- `ges`: Greedy Equivalence Search
- `lingam`: Linear Non-Gaussian Acyclic Model
- `direct_lingam`: Direct LiNGAM
- `granger`: Granger causality (time series)

**Key Classes**:
- `CausalDiscoveryEngine`: Main causal discovery interface

**Methods**:

#### `discover_causal_structure()`
```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.randn(500)
Y = 2 * X + np.random.randn(500) * 0.1
Z = 1.5 * Y + np.random.randn(500) * 0.1
data = np.column_stack([X, Y, Z])

result = engine.discover_causal_structure(
    data=data,
    variable_names=['X', 'Y', 'Z'],
    algorithm='pc',
    alpha=0.05,
    independence_test='fisherz'
)
```

#### `identify_confounders()`
```python
result = engine.identify_confounders(
    graph_data=causal_graph,
    target_x='Treatment',
    target_y='Outcome'
)
```

#### `analyze_causal_graph()`
```python
analysis = engine.analyze_causal_graph(graph_data)
```

### 4. Lagrange-Mapper Integration (`lagrange_mapper_integration.py`)

**Purpose**: Topological data analysis and attractor landscape mapping.

**Key Classes**:
- `LagrangeAttractorAnalyzer`: Main analysis interface

**Methods**:

#### `analyze_embedding_landscape()`
```python
import numpy as np

# Sample embeddings
embeddings = np.random.randn(100, 50)

result = analyzer.analyze_embedding_landscape(
    embeddings=embeddings,
    labels=[f'item_{i}' for i in range(100)],
    n_clusters=8,
    reduction_method='pca',
    reduction_dims=2
)
```

#### `analyze_knowledge_topology()`
```python
result = analyzer.analyze_knowledge_topology(
    graph_data=graph,
    embedding_dim=50
)
```

#### `find_attractor_basins()`
```python
result = analyzer.find_attractor_basins(
    embeddings=embeddings,
    attractor_centers=cluster_centers,
    resolution=50
)
```

#### `detect_landscape_transitions()`
```python
result = analyzer.detect_landscape_transitions(
    embeddings_t1=embeddings_v1,
    embeddings_t2=embeddings_v2
)
```

### 5. Unified Knowledge Extractor (`unified_knowledge_extraction.py`)

**Purpose**: Single interface for all extraction capabilities.

**Key Classes**:
- `UnifiedKnowledgeExtractor`: Main unified interface
- `ExtractionResult`: Standardized result container

**Methods**:

#### `run_extraction_pipeline()`
```python
extractor = UnifiedKnowledgeExtractor()

input_data = {
    'text': 'Sample text for extraction',
    'graph': {
        'nodes': [...],
        'edges': [...]
    },
    'transactions': [...],
    'triples': [...]
}

result = extractor.run_extraction_pipeline(
    input_data=input_data,
    pipeline_config={
        'extract_text': True,
        'analyze_graph': True,
        'mine_patterns': True,
        'generate_embeddings': True
    }
)
```

#### Convenience Function
```python
from knowledge_engine.integrations.unified_knowledge_extraction import extract_knowledge

result = extract_knowledge(
    data={'text': 'Sample text'},
    operations=['text', 'graph', 'patterns']
)
```

## Integration Examples

### Example 1: Complete Knowledge Extraction Pipeline

```python
from knowledge_engine.integrations import AIKnowledgeGraphIntegrator
import numpy as np

integrator = AIKnowledgeGraphIntegrator()

# 1. Extract knowledge from text
text = """
Alice works at AcmeCorp as a software engineer. 
She knows Bob who works at TechInc. 
Charlie also works at TechInc and knows both Alice and Bob.
"""

extraction_result = integrator.extract_knowledge_with_deepke(text)

# 2. Convert to graph and analyze
if extraction_result['status'] == 'success':
    artifacts = extraction_result.get('knowledge_artifacts', [])
    
    # Build graph
    graph_data = {
        'nodes': [],
        'edges': []
    }
    # ... populate graph from artifacts
    
    # Analyze graph
    analysis = integrator.analyze_graph_with_karateclub(graph_data)
    
    # 3. Mine patterns
    transactions = [[a['type'], a.get('predicate', '')] for a in artifacts]
    patterns = integrator.mine_patterns_with_pami(
        {'transactions': transactions},
        config={'mining_type': 'frequent_patterns', 'min_support': 0.1}
    )
    
    # 4. Generate embeddings
    triples = [(a['subject'], a['predicate'], a['object']) 
               for a in artifacts if a.get('knowledge_type') == 'triple']
    embeddings = integrator.embed_knowledge_graph_with_neuralkg(triples)
    
    # 5. Analyze topology
    if integrator.lagrange_analyzer:
        entity_embeddings = list(embeddings.get('embeddings', {}).get('entities', {}).values())
        if entity_embeddings:
            topology = integrator.analyze_attractor_landscape(
                np.array(entity_embeddings)
            )
```

### Example 2: Causal Analysis of Knowledge Graph

```python
from knowledge_engine.integrations import CausalDiscoveryEngine

engine = CausalDiscoveryEngine()

# Convert graph metrics to data matrix
# For example, analyze how node properties causally influence each other
node_metrics = {
    'degree': [5, 3, 4, 2, 6],
    'betweenness': [0.5, 0.2, 0.3, 0.1, 0.6],
    'clustering': [0.3, 0.5, 0.4, 0.2, 0.3],
    'pagerank': [0.2, 0.15, 0.18, 0.1, 0.25]
}

data = np.column_stack([
    node_metrics['degree'],
    node_metrics['betweenness'],
    node_metrics['clustering'],
    node_metrics['pagerank']
])

result = engine.discover_causal_structure(
    data=data,
    variable_names=['degree', 'betweenness', 'clustering', 'pagerank'],
    algorithm='pc',
    alpha=0.05
)

if result['status'] == 'success':
    graph = result['graph']
    print(f"Discovered {len(graph['edges'])} causal relationships")
```

### Example 3: Pattern Mining for Knowledge Discovery

```python
from knowledge_engine.integrations import PAMIPatternMiner

miner = PAMIPatternMiner()

# Mine frequent entity-relationship patterns
knowledge_base = [
    ['Person', 'works_for', 'Organization'],
    ['Person', 'knows', 'Person'],
    ['Person', 'works_for', 'Organization'],
    ['Person', 'located_in', 'City'],
    ['Organization', 'located_in', 'City'],
    ['Person', 'knows', 'Person'],
    ['Person', 'works_for', 'Organization']
]

result = miner.mine_frequent_patterns(
    transactions=knowledge_base,
    min_support=0.3,
    algorithm='fpgrowth'
)

if result['status'] == 'success':
    print(f"Found {result['statistics']['total_patterns']} patterns")
    for pattern in result['patterns'][:5]:
        print(f"Pattern: {pattern['pattern']}, Support: {pattern['support']}")
```

## Error Handling

All integration modules follow consistent error handling:

```python
result = module.some_method(...)

if result['status'] == 'success':
    # Process successful result
    data = result['data']
elif result['status'] == 'error':
    # Handle error
    error_message = result['message']
    print(f"Error: {error_message}")
```

## Testing

Run the comprehensive test suite:

```bash
# From project root
python -m knowledge_engine.tests.test_new_integrations

# Or using pytest
pytest knowledge_engine/tests/test_new_integrations.py -v
```

## Dependencies

Each integration has its own dependencies:

- **Karate Club**: `networkx`, `numpy`, `scipy`
- **PAMI**: (Optional) `PAMI` library or fallback implementation
- **NeuralKG**: `torch`, `pytorch_lightning`, `numpy`
- **Causal-Learn**: `numpy`, `networkx`, `scipy`
- **Lagrange-Mapper**: `numpy`, `scikit-learn` (optional)

Install all dependencies:
```bash
pip install networkx numpy scipy scikit-learn torch pytorch_lightning
```

## Architecture

```
knowledge_engine/
├── integrations/
│   ├── __init__.py                    # Main integrator (AIKnowledgeGraphIntegrator)
│   ├── karateclub_integration.py      # Graph analysis
│   ├── pami_integration.py            # Pattern mining
│   ├── neuralkg_integration.py        # KG embeddings
│   ├── causal_learn_integration.py    # Causal discovery
│   ├── lagrange_mapper_integration.py # Topological analysis
│   └── unified_knowledge_extraction.py # Unified interface
└── tests/
    └── test_new_integrations.py       # Test suite
```

## Contributing

When adding new integrations:

1. Create a new `{project}_integration.py` file
2. Implement availability checking via `is_available()`
3. Follow the error handling pattern (return dict with 'status' key)
4. Add tests to `test_new_integrations.py`
5. Update this documentation

## License

Each integrated project maintains its own license. This integration code follows the OpenEvolve project license.
