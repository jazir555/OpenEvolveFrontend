# AI-Knowledge-Graph Integration Documentation

## Overview

This document provides comprehensive documentation for the AI-Knowledge-Graph (AIKG) advanced features integration into the OpenEvolve Knowledge Engine.

## Features

### 1. Entity Standardization
- Multi-level entity deduplication
- Text normalization (Unicode, lowercase, stopword removal)
- Frequency-based grouping
- Root word analysis (4-char prefix matching)
- LLM-assisted resolution (optional)
- Self-reference filtering
- Variant tracking and mapping

### 2. Relationship Inference
- Transitive inference (A→B, B→C → A→C)
- LLM-based inter-community inference
- Within-community inference
- Lexical similarity inference
- Confidence scoring
- Deduplication of inferred relationships

### 3. Visualization
- D3.js interactive graph visualizations
- Community detection (Louvain algorithm)
- Centrality-based node sizing
- Color-coded communities
- Edge type differentiation (solid/dashed)
- Zoom and pan capabilities
- Export to multiple formats (JSON, CSV, GEXF, GraphML)

## Architecture

```
knowledge_engine/
├── integrations/
│   ├── aikg_standardization.py    # Entity standardization
│   ├── aikg_inference.py          # Relationship inference
│   ├── aikg_visualization.py      # D3.js visualization
│   ├── aikg_integration.py        # Main integration orchestration
│   ├── test_aikg.py               # Test suite
│   └── example_aikg.py            # Usage examples
└── config/
    └── aikg_integration.yaml      # Configuration
```

## Installation

The AIKG integration is included in the Knowledge Engine. No additional installation required beyond the base Knowledge Engine setup.

## Configuration

Edit `knowledge_engine/config/aikg_integration.yaml`:

```yaml
standardization:
  enabled: true
  use_llm_for_entities: false
  stopword_removal: true
  root_word_analysis: true
  self_reference_filtering: true

inference:
  enabled: true
  apply_transitive: true
  use_llm_for_inference: false
  similarity_threshold: 0.7
  max_inference_depth: 3

visualization:
  enabled: true
  output_dir: "data/visualizations"
  community_algorithm: "louvain"
  node_sizing: "centrality"
  edge_differentiation: true
  color_scheme: "colorblind"
```

## Usage

### Quick Start

```python
from knowledge_engine.engine import KnowledgeEngine

# Initialize engine
engine = KnowledgeEngine()

# Process text with complete pipeline
result = await engine.process_with_aikg(
    text="Python is used for web development with Django...",
    enable_standardization=True,
    enable_inference=True,
    generate_visualization=True
)

print(f"Original triples: {result.original_triple_count}")
print(f"Inferred triples: {result.inferred_triple_count}")
print(f"Visualization: {result.visualization_path}")
```

### Entity Standardization

```python
from knowledge_engine.integrations.aikg_standardization import Entity, Triple

# Create entities with duplicates
entities = [
    Entity("Python"),
    Entity("python"),
    Entity("PYTHON")
]

# Create triples
triples = [
    Triple("Python", "used_for", "Web Development")
]

# Standardize
result = await engine.standardize_entities_with_aikg(entities, triples)

print(f"Canonical entities: {len(result.canonical_entities)}")
print(f"Variant mappings: {result.variant_mappings}")
```

### Relationship Inference

```python
# Infer new relationships
result = await engine.infer_relationships_with_aikg(triples, entities)

print(f"Inferred {len(result.inferred_triples)} relationships")
for triple in result.inferred_triples:
    print(f"  {triple.subject} -> {triple.predicate} -> {triple.object}")
    print(f"    Confidence: {triple.confidence:.2f}")
```

### Visualization

```python
# Generate D3.js visualization
result = await engine.visualize_knowledge_graph(
    triples=triples,
    entities=entities,
    output_path="graph.html",
    width=1200,
    height=800
)

print(f"Visualization: {result.output_path}")
print(f"Communities: {result.community_count}")
```

### Export Data

```python
# Export to JSON
json_data = await engine.export_knowledge_graph(triples, format="json")

# Export to CSV
csv_data = await engine.export_knowledge_graph(triples, format="csv")

# Export to Gephi format
gephi_data = await engine.export_knowledge_graph(triples, format="gexf")
```

## API Reference

### KnowledgeEngine Methods

#### `process_with_aikg()`
Complete pipeline processing.

```python
async def process_with_aikg(
    text: str,
    enable_standardization: bool = True,
    enable_inference: bool = True,
    generate_visualization: bool = True,
    output_path: Optional[str] = None
) -> AIKGResult
```

**Returns:** `AIKGResult` object with:
- `original_triples`: List of original triples
- `standardized_entities`: List of canonical entities
- `all_triples`: Original + inferred triples
- `visualization_path`: Path to D3.js HTML file
- `get_summary()`: Get processing statistics

#### `standardize_entities_with_aikg()`
Entity standardization only.

```python
async def standardize_entities_with_aikg(
    entities: List[Entity],
    triples: List[Triple]
) -> StandardizationResult
```

**Returns:** `StandardizationResult` with:
- `canonical_entities`: Deduplicated entities
- `variant_mappings`: Canonical -> variants mapping
- `statistics`: Processing statistics

#### `infer_relationships_with_aikg()`
Relationship inference only.

```python
async def infer_relationships_with_aikg(
    triples: List[Triple],
    entities: List[Entity]
) -> InferenceResult
```

**Returns:** `InferenceResult` with:
- `original_triples`: Input triples
- `inferred_triples`: Newly inferred triples
- `all_triples`: Combined list
- `confidence_scores`: Triple -> confidence mapping
- `inference_sources`: Triple -> source method mapping

#### `visualize_knowledge_graph()`
Generate D3.js visualization.

```python
async def visualize_knowledge_graph(
    triples: List[Triple],
    entities: List[Entity],
    output_path: str,
    width: int = 1200,
    height: int = 800
) -> VisualizationResult
```

**Returns:** `VisualizationResult` with:
- `output_path`: Path to HTML file
- `node_count`: Number of nodes
- `edge_count`: Number of edges
- `community_count`: Number of communities
- `statistics`: Graph statistics

#### `export_knowledge_graph()`
Export graph data.

```python
async def export_knowledge_graph(
    triples: List[Triple],
    format: str = "json"
) -> str
```

**Formats:** `json`, `csv`, `gexf`, `graphml`

## Data Models

### Entity
```python
class Entity:
    name: str                          # Entity name
    entity_type: Optional[str]         # Optional type
    attributes: Dict[str, Any]         # Optional attributes
    variants: List[str]                # Variant names
    canonical: Optional[str]           # Canonical form
```

### Triple
```python
class Triple:
    subject: str                       # Subject entity
    predicate: str                     # Relationship type
    object: str                        # Object entity
    confidence: float = 1.0            # Confidence score
    source: str = "extracted"          # "extracted" or "inferred"
```

### AIKGResult
```python
class AIKGResult:
    original_triples: List[Triple]
    original_entities: List[Entity]
    standardized_entities: List[Entity]
    all_triples: List[Triple]
    visualization_path: Optional[str]

    # Computed properties
    @property
    def original_triple_count(self) -> int

    @property
    def inferred_triple_count(self) -> int

    @property
    def total_triple_count(self) -> int

    @property
    def entity_reduction_rate(self) -> float

    def get_summary(self) -> Dict[str, Any]
```

## Examples

See `knowledge_engine/integrations/example_aikg.py` for comprehensive examples:

1. Complete pipeline processing
2. Entity standardization
3. Relationship inference
4. Visualization generation
5. Direct integration usage
6. Data export
7. Variant mappings

Run examples:
```bash
cd knowledge_engine/integrations
python example_aikg.py
```

## Testing

Run the test suite:

```bash
cd knowledge_engine/integrations
pytest test_aikg.py -v
```

Test coverage:
- Entity standardization tests
- Relationship inference tests
- Visualization generation tests
- Complete pipeline tests

## Performance Considerations

### Entity Standardization
- **Complexity:** O(n²) for grouping, where n = number of entities
- **Optimization:** Use frequency-based grouping before root word analysis
- **Recommended max:** 10,000 entities

### Relationship Inference
- **Transitive inference:** O(n × d) where n = nodes, d = max depth
- **Lexical similarity:** O(n²) where n = entities
- **Recommended max:** 5,000 triples for inference

### Visualization
- **Community detection:** O(n²) for Louvain algorithm
- **Rendering:** O(n + e) where n = nodes, e = edges
- **Recommended max:** 500 nodes for smooth D3.js interaction

## Troubleshooting

### Issue: "AIKG integration not initialized"
**Solution:** Check that `knowledge_engine/config/aikg_integration.yaml` exists and is valid.

### Issue: Low entity reduction rate
**Solution:** Enable `root_word_analysis` and adjust `similarity_threshold` in config.

### Issue: Too many inferred relationships
**Solution:** Reduce `max_inference_depth` or increase `min_inference_confidence` in config.

### Issue: Visualization performance
**Solution:** Reduce number of nodes (<500 for optimal performance) or disable `enable_physics`.

### Issue: NetworkX Louvain algorithm not available
**Solution:** Install `python-louvain`: `pip install python-louvain`

## Configuration Options

### Standardization
- `use_llm_for_entities`: Enable LLM for entity resolution (requires API)
- `stopword_removal`: Remove common stopwords during normalization
- `root_word_analysis`: Use 4-char prefix matching
- `self_reference_filtering`: Remove triples where subject == object

### Inference
- `apply_transitive`: Enable transitive inference
- `use_llm_for_inference`: Enable LLM for inter-community inference
- `similarity_threshold`: Minimum similarity for lexical inference (0.0-1.0)
- `max_inference_depth`: Maximum depth for transitive inference

### Visualization
- `output_dir`: Directory for output files
- `community_algorithm`: Community detection algorithm
- `node_sizing`: Strategy for node sizing (centrality/degree/uniform)
- `edge_differentiation`: Differentiate edge types visually
- `color_scheme`: Color palette (colorblind/default/spectral)

## Advanced Features

### Custom Stopwords

Add custom stopwords in config:

```yaml
standardization:
  custom_stopwords:
    - "custom_word1"
    - "custom_word2"
```

### LLM Integration

For advanced entity resolution and inference, configure LLM:

```yaml
llm:
  provider: "anthropic"
  entity_resolution_model: "claude-sonnet-4-20250514"
  inference_model: "claude-sonnet-4-20250514"
  temperature: 0.3
```

### Custom Centrality Weights

Adjust centrality calculation for node sizing:

```yaml
visualization:
  centrality_weights:
    degree: 0.6      # 60% weight
    betweenness: 0.3  # 30% weight
    eigenvector: 0.1  # 10% weight
```

## Best Practices

1. **Start Simple:** Begin with default configuration, adjust as needed
2. **Monitor Performance:** Track processing time for large datasets
3. **Validate Results:** Review standardization and inference quality
4. **Export Often:** Save intermediate results for analysis
5. **Tune Thresholds:** Adjust similarity thresholds based on domain
6. **Use Variants:** Leverage variant mappings for entity lookup

## Contributing

When contributing to AIKG integration:

1. Add tests for new features
2. Update documentation
3. Follow existing code style
4. Add examples for new functionality
5. Test with various data sizes

## License

Part of the OpenEvolve Knowledge Engine project.

## Support

For issues or questions:
- Check examples: `example_aikg.py`
- Run tests: `pytest test_aikg.py`
- Review configuration: `aikg_integration.yaml`
- Check logs for error messages
