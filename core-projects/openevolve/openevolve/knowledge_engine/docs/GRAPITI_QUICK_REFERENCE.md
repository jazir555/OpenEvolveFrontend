# Graphiti Temporal Integration - Quick Reference

## Quick Start

```python
from datetime import datetime, timedelta
from knowledge_engine.core.temporal_knowledge_engine import TemporalKnowledgeEngine

# Initialize
engine = TemporalKnowledgeEngine(
    enable_temporal=True,
    enable_hybrid_search=True,
)

# Add temporal knowledge
await engine.add_knowledge_temporal(
    content="Python 3.11 improved async performance",
    artifact_type="solution_pattern",
    valid_at=datetime.utcnow(),
)

# Query
results = await engine.search_with_graphiti("async optimization")
```

## Common Operations

### Adding Knowledge

```python
# Simple temporal knowledge
await engine.add_knowledge_temporal(
    content="Use async for I/O operations",
    artifact_type="solution_pattern",
    valid_at=now,
)

# Knowledge with expiration
await engine.add_knowledge_temporal(
    content="Temporary workaround",
    artifact_type="workflow",
    valid_at=now,
    invalid_at=now + timedelta(days=30),
)

# Knowledge with metadata
await engine.add_knowledge_temporal(
    content="Docker for containerization",
    artifact_type="solution_pattern",
    valid_at=now,
    metadata={"technology": "docker", "category": "containers"},
)
```

### Querying Knowledge

```python
# Point-in-time query
results = await engine.query_at_time(
    query="python async",
    timestamp=datetime(2024, 6, 1),
)

# Hybrid search
results = await engine.search_with_graphiti(
    query="database optimization",
    use_hybrid=True,
    rerank_method="rrf",
    max_results=10,
)

# Local search (no Graphiti)
results = await engine._local_search("docker", max_results=10)
```

### Temporal Operations

```python
# Get valid knowledge at time
valid = await engine.get_valid_knowledge(timestamp=now)

# Invalidate knowledge
await engine.invalidate_knowledge(
    artifact_id="artifact_123",
    invalid_at=now + timedelta(hours=24),
)

# Get timeline
timeline = await engine.get_timeline(
    entity="Python",
    start_time=datetime(2024, 1, 1),
    end_time=datetime.utcnow(),
)
```

### Contradiction Detection

```python
# Detect all contradictions
result = await engine.detect_contradictions()

if result.has_contradictions:
    print(f"Found {len(result.contradictions)} contradictions")

# Check specific artifact
result = await engine.detect_contradictions(knowledge_id="artifact_123")
```

## Configuration

### Temporal Settings

```yaml
temporal_reasoning:
  enabled: true
  default_time_range_hours: 24
  track_validity: true
  point_in_time_queries: true
```

### Hybrid Search Settings

```yaml
hybrid_search:
  enabled: true
  rerank_method: rrf  # rrf, cross_encoder, weighted, none
  components:
    use_bm25: true
    use_vector: true
    use_graph_traversal: true
```

### Entity Mappings

```yaml
entity_mappings:
  solution_pattern: Procedure
  workflow: Document
  agent: Organization
  tool: Technology
```

## KnowledgeArtifact

### Attributes

```python
artifact = KnowledgeArtifact(
    id="unique_id",
    content="Knowledge content",
    artifact_type="solution_pattern",
    valid_at=datetime.utcnow(),
    invalid_at=None,  # Optional
    created_at=datetime.utcnow(),
    source="openevolve",
    metadata={},
    entities=[],
    relationships=[],
    confidence=1.0,
    group_id=None,
)
```

### Methods

```python
# Check validity
is_valid = artifact.is_valid_at(timestamp=now)

# Serialize
data = artifact.to_dict()

# Deserialize
artifact = KnowledgeArtifact.from_dict(data)
```

## Reranking Methods

### RRF (Reciprocal Rank Fusion)
- **Best for:** General purpose
- **Speed:** Fast
- **Quality:** Good
- **Use when:** You need balanced performance

```python
results = await engine.search_with_graphiti(
    query="...",
    rerank_method="rrf",
)
```

### Cross Encoder
- **Best for:** High accuracy
- **Speed:** Slower
- **Quality:** Best
- **Use when:** Accuracy is critical

```python
results = await engine.search_with_graphiti(
    query="...",
    rerank_method="cross_encoder",
)
```

### Weighted
- **Best for:** Custom ranking
- **Speed:** Fast
- **Quality:** Configurable
- **Use when:** You have specific weights

```python
# Configure weights in config.yaml
weights:
  bm25: 0.3
  vector: 0.5
  graph_traversal: 0.2
```

## Temporal Filters

### CURRENT
```python
# Only currently valid knowledge
temporal_filters = {"filter_type": "current"}
```

### TIME_RANGE
```python
# Knowledge within time range
temporal_filters = {
    "filter_type": "time_range",
    "start_time": datetime(2024, 1, 1),
    "end_time": datetime(2024, 12, 31),
}
```

### HISTORICAL
```python
# All historical knowledge
temporal_filters = {"filter_type": "historical"}
```

## Best Practices

### 1. Temporal Tracking

```python
# DO: Set appropriate valid_at times
await engine.add_knowledge_temporal(
    content="Current best practice",
    artifact_type="solution_pattern",
    valid_at=datetime.utcnow(),  # Now
)

# DON'T: Use default times for historical data
await engine.add_knowledge_temporal(
    content="Historical fact",
    artifact_type="solution_pattern",
    valid_at=datetime(2020, 1, 1),  # Be explicit
)
```

### 2. Knowledge Expiration

```python
# DO: Set expiration for temporary knowledge
await engine.add_knowledge_temporal(
    content="Temporary workaround",
    artifact_type="workflow",
    valid_at=now,
    invalid_at=now + timedelta(days=30),  # Expires in 30 days
)

# DON'T: Leave temporary knowledge indefinitely valid
```

### 3. Hybrid Search

```python
# DO: Use appropriate reranking for your use case
# Fast results
rerank_method="rrf"

# Best quality
rerank_method="cross_encoder"

# Custom balance
rerank_method="weighted"

# DON'T: Use cross_encoder for real-time queries (too slow)
```

### 4. Contradiction Detection

```python
# DO: Run contradiction detection periodically
result = await engine.detect_contradictions()
if result.has_contradictions:
    # Handle contradictions
    pass

# DON'T: Run on every knowledge addition (too expensive)
```

## Error Handling

```python
try:
    results = await engine.query_at_time(
        query="python async",
        timestamp=now,
    )
except Exception as e:
    logger.error(f"Query failed: {e}")
    # Fallback to local search
    results = await engine._local_search("python async")
```

## Performance Tips

1. **Use caching** (enabled by default)
2. **Limit result size** (max_results=10-20)
3. **Use RRF reranking** (fastest)
4. **Narrow time ranges** for temporal queries
5. **Batch operations** when adding multiple artifacts

## Testing

```python
# Test temporal knowledge
@pytest.mark.asyncio
async def test_temporal_knowledge():
    engine = TemporalKnowledgeEngine()
    artifact = await engine.add_knowledge_temporal(
        content="Test",
        artifact_type="solution_pattern",
        valid_at=datetime.utcnow(),
    )
    assert artifact is not None
```

## Troubleshooting

### No Results from Query
- Check valid_at timestamp
- Verify temporal filters
- Ensure knowledge was added

### Slow Performance
- Reduce max_results
- Use 'rrf' reranking
- Check Neo4j performance
- Enable caching

### Graphiti Not Available
- Check installation: `pip install graphiti-core`
- Verify Neo4j is running
- Check connection settings in config.yaml

## Examples

See: `knowledge_engine/examples/temporal_graphiti_example.py`

## Documentation

Full guide: `knowledge_engine/docs/GRAPITI_TEMPORAL_INTEGRATION.md`
