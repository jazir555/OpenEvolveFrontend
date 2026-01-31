# Knowledge Engine Integrations - Quick Reference

## One-Liners

```python
from knowledge_engine.integrations import AIKnowledgeGraphIntegrator

# Initialize
ke = AIKnowledgeGraphIntegrator()

# Check what's available
ke.get_integration_status()
```

## Pattern Mining (PAMI)

```python
from knowledge_engine.integrations import PAMIPatternMiner

m = PAMIPatternMiner()

# Frequent patterns
m.mine_frequent_patterns(transactions=[['a','b'], ['b','c']], min_support=0.5)

# Sequential patterns
m.mine_sequences(sequences=[[['a'],['b']], [['a'],['c']]], min_support=0.5)

# Association rules
m.discover_association_rules(transactions, min_support=0.1, min_confidence=0.5)

# Graph patterns
m.analyze_knowledge_graph_patterns({'nodes':[...], 'edges':[...]}, min_support=0.1)
```

## KG Embeddings (NeuralKG)

```python
from knowledge_engine.integrations import NeuralKGEmbedder

e = NeuralKGEmbedder()

# Generate embeddings
e.generate_embeddings(triples=[('A','knows','B')], model_name='transe', embedding_dim=100)

# Link prediction
e.predict_links(head='A', relation='knows', candidate_tails=['B','C'], embeddings=emb, top_k=5)

# Find similar entities
e.find_similar_entities(entity='A', embeddings=emb, top_k=10)

# Ensemble
e.ensemble_embeddings(triples, models=['transe','complex'], embedding_dim=100)
```

## Causal Discovery (Causal-Learn)

```python
from knowledge_engine.integrations import CausalDiscoveryEngine
import numpy as np

c = CausalDiscoveryEngine()

# PC algorithm
c.discover_causal_structure(data=np.random.randn(100,3), algorithm='pc', alpha=0.05)

# FCI (with latent variables)
c.discover_causal_structure(data, algorithm='fci', alpha=0.05)

# LiNGAM
c.discover_causal_structure(data, algorithm='lingam')

# Find confounders
c.identify_confounders(graph_data, target_x='X', target_y='Y')

# Analyze graph
c.analyze_causal_graph(graph_data)
```

## Topological Analysis (Lagrange-Mapper)

```python
from knowledge_engine.integrations import LagrangeAttractorAnalyzer
import numpy as np

l = LagrangeAttractorAnalyzer()

# Analyze embedding landscape
l.analyze_embedding_landscape(embeddings=np.random.randn(100,50), n_clusters=8)

# Knowledge graph topology
l.analyze_knowledge_topology({'nodes':[...], 'edges':[...]}, embedding_dim=50)

# Attractor basins
l.find_attractor_basins(embeddings, attractor_centers=centers, resolution=50)

# Detect transitions
l.detect_landscape_transitions(embeddings_v1, embeddings_v2)
```

## Unified Extractor

```python
from knowledge_engine.integrations.unified_knowledge_extraction import (
    UnifiedKnowledgeExtractor, extract_knowledge
)

u = UnifiedKnowledgeExtractor()

# Run full pipeline
u.run_extraction_pipeline(
    input_data={'text': '...', 'graph': {...}, 'triples': [...]},
    pipeline_config={'extract_text': True, 'analyze_graph': True}
)

# Quick extraction
extract_knowledge(data={'text': '...'}, operations=['text'])
```

## Common Patterns

### Check Availability
```python
if module.is_available():
    result = module.method()
```

### Handle Results
```python
result = module.method(...)
if result['status'] == 'success':
    data = result['data']  # or result['patterns'], result['embeddings'], etc.
else:
    print(f"Error: {result['message']}")
```

### Complete Pipeline
```python
ke = AIKnowledgeGraphIntegrator()

# 1. Extract from text
ext = ke.extract_knowledge_with_deepke(text)

# 2. Analyze graph
graph = ke.analyze_graph_with_karateclub(graph_data)

# 3. Mine patterns
patterns = ke.mine_patterns_with_pami(data, config={'mining_type': 'frequent_patterns'})

# 4. Generate embeddings
emb = ke.embed_knowledge_graph_with_neuralkg(triples, model='transe')

# 5. Causal discovery
causal = ke.discover_causal_structure(data_matrix, algorithm='pc')

# 6. Analyze topology
topo = ke.analyze_attractor_landscape(embeddings)
```

## Module Availability

| Module | Method | Fallback if Unavailable |
|--------|--------|------------------------|
| PAMI | `is_available()` | Pure Python implementation |
| NeuralKG | `is_available()` | Simplified embeddings |
| Causal-Learn | `is_available()` | Basic correlation analysis |
| Lagrange-Mapper | `is_available()` | Basic clustering |

## Data Formats

### Knowledge Graph
```python
{
    'nodes': [{'id': 'A', 'type': 'Person'}, ...],
    'edges': [{'source': 'A', 'target': 'B', 'type': 'knows'}, ...]
}
```

### Triples
```python
[('Alice', 'knows', 'Bob'), ('Bob', 'works_for', 'AcmeCorp')]
```

### Transactions
```python
[['bread', 'milk'], ['bread', 'butter'], ['milk', 'eggs']]
```

### Sequences
```python
[[['a'], ['b'], ['c']], [['a'], ['c'], ['d']]]
```

## Error Messages

| Error | Solution |
|-------|----------|
| "Module not available" | Install dependencies or use fallback |
| "Algorithm not available" | Check `get_available_algorithms()` |
| "Data format error" | Verify input matches expected format |
| "Insufficient data" | Provide more samples/transactions |

## Performance Tips

1. **Batch operations**: Process multiple items at once
2. **Cache embeddings**: Reuse embeddings for multiple operations
3. **Reduce dimensions**: Use PCA before topological analysis
4. **Sample large datasets**: Use representative samples for speed
5. **Parallel processing**: Use multiprocessing for independent tasks

## Help & Documentation

```python
# Get module status
module.get_status()

# Get available methods
module.get_available_algorithms()  # or models, etc.

# Get detailed info
module.get_algorithm_info('pc')
module.get_model_info('transe')
```
