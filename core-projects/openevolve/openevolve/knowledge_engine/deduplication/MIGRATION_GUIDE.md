# Unified Deduplication System - Migration Guide

## Overview

The Unified Deduplication Manager combines four powerful strategies:
- **SEMHASH**: Fast rule-based deduplication (from kg-gen)
- **LM Cluster**: ML-based clustering (from kg-gen)
- **Standardization**: Entity normalization (from ai-knowledge-graph)
- **Semantic**: LLM-based semantic matching (from Graphiti)

## Migration from Basic Deduplication

### Before (Old Approach)

```python
# Old basic duplicate detection in enhanced_storage.py
duplicates = []
for i, artifact1 in enumerate(artifacts):
    for artifact2 in artifacts[i + 1:]:
        if artifact1.name.lower() == artifact2.name.lower():
            duplicates.append((artifact1, artifact2))
```

### After (New Unified Manager)

```python
from knowledge_engine.core.deduplication import UnifiedDeduplicationManager, Entity

# Convert artifacts to entities
entities = [
    Entity(
        id=artifact.id,
        name=artifact.name,
        entity_type=artifact.type,
        description=artifact.description,
        properties=artifact.metadata
    )
    for artifact in artifacts
]

# Deduplicate with automatic strategy selection
dedup_manager = UnifiedDeduplicationManager()
result = await dedup_manager.deduplicate(entities, strategy='auto')

# Get canonical entities
canonical_artifacts = [
    artifact for artifact in artifacts
    if artifact.id in [e.id for e in result.canonical_entities]
]
```

## Strategy Selection Guide

### Automatic Selection (Recommended)

```python
result = await manager.deduplicate(entities, strategy='auto')
```

The system will automatically choose:
- **< 100 entities**: `semhash` (fastest)
- **100-1000 entities**: `standardization`
- **> 1000 entities**: `lm_cluster` (most accurate)
- **Ambiguous entities**: `semantic` (LLM-based)

### Manual Selection

Choose a specific strategy based on your needs:

```python
# Fast, exact/near-exact duplicates
result = await manager.deduplicate(entities, strategy='semhash')

# Entity normalization and standardization
result = await manager.deduplicate(entities, strategy='standardization')

# Large datasets with semantic similarity
result = await manager.deduplicate(entities, strategy='lm_cluster')

# Ambiguous entities requiring LLM understanding
result = await manager.deduplicate(entities, strategy='semantic')
```

## Configuration

### 1. Basic Configuration

```python
from knowledge_engine.core.deduplication import UnifiedDeduplicationManager

# Use default config
manager = UnifiedDeduplicationManager()

# Use custom config
manager = UnifiedDeduplicationManager(
    config_path='/path/to/custom/deduplication.yaml'
)
```

### 2. YAML Configuration

Edit `knowledge_engine/config/deduplication.yaml`:

```yaml
# Default strategy
default_strategy: auto

# Cache settings
cache_enabled: true
cache_ttl: 3600

# Strategy-specific settings
strategies:
  semhash:
    enabled: true
    similarity_threshold: 0.95

  lm_cluster:
    enabled: true
    cluster_size: 128

  standardization:
    enabled: true
    stem_length: 4

  semantic:
    enabled: true
    confidence_threshold: 0.8
```

## Integration Examples

### Example 1: Knowledge Engine Integration

```python
from knowledge_engine.enhanced_storage import EnhancedKnowledgeStorage
from knowledge_engine.core.deduplication import UnifiedDeduplicationManager, Entity

class DeduplicatedKnowledgeStorage(EnhancedKnowledgeStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dedup_manager = UnifiedDeduplicationManager()

    async def store_artifacts(self, artifacts):
        # Convert to entities
        entities = [
            Entity(
                id=a.id,
                name=a.name,
                entity_type=a.type,
                description=a.description,
                properties={'source': a.source}
            )
            for a in artifacts
        ]

        # Deduplicate
        result = await self.dedup_manager.deduplicate(entities)

        # Store only canonical entities
        canonical_ids = {e.id for e in result.canonical_entities}
        canonical_artifacts = [a for a in artifacts if a.id in canonical_ids]

        return await super().store_artifacts(canonical_artifacts)
```

### Example 2: Real-time Deduplication

```python
async def process_streaming_entities(entity_stream):
    """Process entities in real-time with caching."""
    manager = UnifiedDeduplicationManager()
    batch = []

    async for entity in entity_stream:
        batch.append(entity)

        # Process batch when full
        if len(batch) >= 100:
            result = await manager.deduplicate(batch, use_cache=True)
            batch = []

            # Yield canonical entities
            for entity in result.canonical_entities:
                yield entity

    # Process remaining
    if batch:
        result = await manager.deduplicate(batch)
        for entity in result.canonical_entities:
            yield entity
```

### Example 3: Tracking Canonical Mappings

```python
manager = UnifiedDeduplicationManager()

# Deduplicate
result = await manager.deduplicate(entities)

# Get canonical mappings
mappings = manager.get_canonical_mapping()

# mappings = {
#     'canonical_id_1': ['variant_1', 'variant_2'],
#     'canonical_id_2': ['variant_3']
# }

# Use mappings for updates
async def update_entity(canonical_id, new_data):
    # Update canonical and all variants
    variant_ids = mappings.get(canonical_id, [])
    all_ids = [canonical_id] + variant_ids

    for entity_id in all_ids:
        await db.update(entity_id, new_data)
```

## Performance Considerations

### Caching

Enable caching for repeated deduplication:

```python
# First call - computes and caches
result1 = await manager.deduplicate(entities, use_cache=True)

# Second call - returns cached result (much faster)
result2 = await manager.deduplicate(entities, use_cache=True)

# Clear cache when needed
manager.clear_cache()
```

### Batch Processing

For large datasets, batch processing is automatic:

```python
# SEMHASH and LM Cluster automatically batch
large_entities = [...]  # 10,000+ entities

# Processes in batches automatically
result = await manager.deduplicate(large_entities, strategy='lm_cluster')
```

### Parallel Processing

Configure parallel workers:

```python
# In deduplication.yaml
performance:
  enable_parallel: true
  max_parallel_workers: 8
```

## Testing

### Unit Tests

```python
import pytest
from knowledge_engine.core.deduplication import UnifiedDeduplicationManager, Entity

@pytest.mark.asyncio
async def test_deduplication():
    manager = UnifiedDeduplicationManager()

    entities = [
        Entity(id="e1", name="Test", entity_type="concept"),
        Entity(id="e2", name="test", entity_type="concept"),  # Duplicate
    ]

    result = await manager.deduplicate(entities)

    assert len(result.canonical_entities) == 1
    assert len(result.duplicate_groups) == 1
```

### Benchmarking

```python
import time

async def benchmark_deduplication():
    manager = UnifiedDeduplicationManager()

    # Create test data
    entities = create_test_entities(n=1000)

    # Benchmark each strategy
    for strategy in ['semhash', 'standardization', 'lm_cluster', 'semantic']:
        start = time.time()
        result = await manager.deduplicate(entities, strategy=strategy)
        elapsed = time.time() - start

        print(f"{strategy}: {elapsed:.2f}s, "
              f"{len(entities)} -> {len(result.canonical_entities)} entities")
```

## Troubleshooting

### Issue: Import Errors

```python
# Missing dependencies
# Install: pip install sentence-transformers scikit-learn
```

### Issue: Slow Performance

```python
# 1. Enable caching
result = await manager.deduplicate(entities, use_cache=True)

# 2. Use faster strategy for small datasets
result = await manager.deduplicate(entities, strategy='semhash')

# 3. Adjust batch size
# In deduplication.yaml:
# performance:
#   batch_size: 200
```

### Issue: Too Many False Positives

```python
# Increase threshold
# In deduplication.yaml:
strategies:
  semhash:
    similarity_threshold: 0.98  # Higher threshold
```

## Best Practices

1. **Use 'auto' strategy** for automatic optimization
2. **Enable caching** for repeated operations
3. **Monitor stats** to understand performance
4. **Test with sample data** before large runs
5. **Clear cache periodically** for long-running systems

## API Reference

### UnifiedDeduplicationManager

```python
class UnifiedDeduplicationManager:
    def __init__(self, config_path: Optional[str] = None)

    async def deduplicate(
        self,
        entities: List[Entity],
        strategy: str = 'auto',
        use_cache: bool = True
    ) -> DeduplicationResult

    async def merge_entities(
        self,
        entity_group: List[Entity]
    ) -> Entity

    def get_canonical_mapping(self) -> Dict[str, List[str]]

    def clear_cache(self)

    def get_stats(self) -> Dict[str, Any]
```

### Entity

```python
@dataclass
class Entity:
    id: str
    name: str
    entity_type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### DeduplicationResult

```python
@dataclass
class DeduplicationResult:
    canonical_entities: List[Entity]
    duplicate_groups: List[List[Entity]]
    stats: Dict[str, Any]
    strategy_used: str
    processing_time_ms: float
```

## Support

For issues or questions:
1. Check test suite: `knowledge_engine/core/deduplication/test_deduplication.py`
2. Review configuration: `knowledge_engine/config/deduplication.yaml`
3. Check logs for detailed error messages
