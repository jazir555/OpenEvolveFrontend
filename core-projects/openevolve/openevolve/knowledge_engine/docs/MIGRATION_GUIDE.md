# Migration Guide: From Basic to Temporal Knowledge Engine

## Overview

This guide helps you migrate from the basic KnowledgeEngine to the new TemporalKnowledgeEngine with Graphiti integration.

## Why Migrate?

### Before (Basic KnowledgeEngine)
- No temporal tracking
- Simple keyword search
- No contradiction detection
- Limited query capabilities

### After (TemporalKnowledgeEngine)
- ✅ Temporal knowledge tracking
- ✅ Hybrid search (BM25 + Vector + Graph)
- ✅ Contradiction detection
- ✅ Point-in-time queries
- ✅ Timeline reconstruction
- ✅ Knowledge expiration

## Migration Path

### Step 1: Update Imports

**Before:**
```python
from knowledge_engine.engine import KnowledgeEngine
```

**After:**
```python
from knowledge_engine.core.temporal_knowledge_engine import TemporalKnowledgeEngine
```

### Step 2: Update Initialization

**Before:**
```python
engine = KnowledgeEngine(
    indexer_config_path="knowledge_engine/indexer_config.yaml",
    api_secrets_path="mcp_agent.secrets.yaml",
    persistence_path="knowledge_graph.json",
)
```

**After:**
```python
engine = TemporalKnowledgeEngine(
    indexer_config_path="knowledge_engine/indexer_config.yaml",
    api_secrets_path="mcp_agent.secrets.yaml",
    persistence_path="knowledge_graph.json",
    graphiti_config_path="integrations/graphiti/config.yaml",
    enable_temporal=True,           # NEW: Enable temporal tracking
    enable_hybrid_search=True,      # NEW: Enable hybrid search
    default_rerank_method=RerankMethod.RRF,  # NEW: Reranking strategy
)
```

### Step 3: Update Knowledge Addition

**Before:**
```python
# Add document (no temporal info)
text = await engine.add_document(path="doc.pdf")
```

**After:**
```python
# Add with temporal metadata
from datetime import datetime

now = datetime.utcnow()

artifact = await engine.add_knowledge_temporal(
    content="Document content",
    artifact_type="solution_pattern",
    valid_at=now,
    metadata={"source": "doc.pdf"},
)
```

### Step 4: Update Search

**Before:**
```python
# Simple keyword search
results = engine.query_index_by_keyword(index_data, "python")
```

**After:**
```python
# Hybrid search with temporal filtering
results = await engine.search_with_graphiti(
    query="python async",
    use_hybrid=True,
    rerank_method="rrf",
    max_results=10,
)
```

## Common Migration Patterns

### Pattern 1: Adding Temporal Context

**Before:**
```python
# Store knowledge without time context
knowledge = {
    "content": "Use async for I/O",
    "type": "solution",
}
```

**After:**
```python
# Store with temporal context
artifact = await engine.add_knowledge_temporal(
    content="Use async for I/O",
    artifact_type="solution_pattern",
    valid_at=datetime.utcnow(),  # Now
    invalid_at=None,  # Still valid
    metadata={"language": "python"},
)
```

### Pattern 2: Querying with Time

**Before:**
```python
# Get current knowledge
results = search_engine.search("current practices")
```

**After:**
```python
# Get knowledge as of a specific time
past_time = datetime(2024, 1, 1)
results = await engine.query_at_time(
    query="current practices",
    timestamp=past_time,
)
```

### Pattern 3: Handling Outdated Knowledge

**Before:**
```python
# Manual deletion or marking
def delete_outdated(knowledge_id):
    # Manual cleanup
    pass
```

**After:**
```python
# Automatic expiration
await engine.add_knowledge_temporal(
    content="Temporary workaround",
    artifact_type="workflow",
    valid_at=now,
    invalid_at=now + timedelta(days=30),  # Auto-expires
)
```

## Backward Compatibility

### Existing Code Still Works

```python
# Basic KnowledgeEngine methods still available
await engine.add_document(path="doc.pdf")
await engine.generate_knowledge(context, query)
await engine.query_bedrock_knowledge_base(kb_id, query)
```

### Graceful Degradation

```python
# If Graphiti unavailable, falls back to local
results = await engine.search_with_graphiti(
    query="test",
    use_hybrid=True,
)
# Falls back to _local_search() automatically
```

## API Compatibility Table

| Old Method | New Method | Notes |
|------------|------------|-------|
| `add_document()` | `add_knowledge_temporal()` | Add temporal metadata |
| `query_index_by_keyword()` | `search_with_graphiti()` | Hybrid search |
| N/A | `query_at_time()` | Point-in-time queries |
| N/A | `detect_contradictions()` | New feature |
| N/A | `get_timeline()` | New feature |
| N/A | `invalidate_knowledge()` | New feature |

## Data Migration

### Migrate Existing Knowledge Graph

```python
async def migrate_knowledge_graph(
    old_engine: KnowledgeEngine,
    new_engine: TemporalKnowledgeEngine,
):
    """Migrate knowledge from old to new engine."""

    # Load old knowledge
    await old_engine.load_graph()

    # Migrate entities
    for entity_name, attrs in old_engine.entity_graph.get_entities().items():
        await new_engine.add_knowledge_temporal(
            content=f"Entity: {entity_name}",
            artifact_type="entity",
            valid_at=datetime.utcnow(),
            metadata=attrs,
        )

    # Migrate relationships
    for rel in old_engine.entity_graph.relationships:
        await new_engine.add_knowledge_temporal(
            content=f"{rel['source']} {rel['relation']} {rel['target']}",
            artifact_type="relationship",
            valid_at=datetime.utcnow(),
            metadata=rel,
        )
```

## Configuration Migration

### Update Config Files

**Old Config** (indexer_config.yaml):
```yaml
llm:
  anthropic_default_model: claude-sonnet-4-20250514
  openai_default_model: o3-mini
```

**New Config** (integrations/graphiti/config.yaml):
```yaml
features:
  temporal_tracking: true
  hybrid_search:
    enabled: true
    rerank_method: rrf
  contradiction_detection:
    enabled: true
```

## Testing Migration

### Test Checklist

- [ ] Can initialize TemporalKnowledgeEngine
- [ ] Can add temporal knowledge
- [ ] Can query with temporal filters
- [ ] Can use hybrid search
- [ ] Can detect contradictions
- [ ] Can get timelines
- [ ] Existing code still works
- [ ] Performance is acceptable

### Example Test

```python
@pytest.mark.asyncio
async def test_migration():
    # Old engine
    old_engine = KnowledgeEngine()

    # New engine
    new_engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
    )

    # Test that old methods still work
    text = await new_engine.add_document("test.pdf")
    assert text is not None

    # Test new methods
    artifact = await new_engine.add_knowledge_temporal(
        content="Test",
        artifact_type="solution_pattern",
        valid_at=datetime.utcnow(),
    )
    assert artifact is not None
```

## Performance Considerations

### Expected Performance Changes

| Operation | Old Engine | New Engine | Change |
|-----------|------------|------------|--------|
| Add knowledge | Fast | Fast | Same |
| Simple search | Fast | Medium | Slightly slower (better quality) |
| Temporal query | N/A | Fast | New feature |
| Contradiction detection | N/A | Medium | New feature |
| Hybrid search | N/A | Medium | New feature |

### Optimization Tips

1. **Use caching** (enabled by default)
2. **Limit result size** (max_results=10-20)
3. **Use RRF reranking** (fastest hybrid search)
4. **Narrow time ranges** for temporal queries
5. **Batch operations** when adding knowledge

## Common Issues

### Issue 1: Import Errors

**Problem:**
```python
ImportError: cannot import name 'TemporalKnowledgeEngine'
```

**Solution:**
```python
# Use correct import path
from knowledge_engine.core.temporal_knowledge_engine import TemporalKnowledgeEngine
```

### Issue 2: Graphiti Not Available

**Problem:**
```
Graphiti not available - bridge in degraded mode
```

**Solution:**
```python
# This is expected - falls back to local search
# Performance will be slightly reduced
# To enable Graphiti:
# 1. Install: pip install graphiti-core
# 2. Start Neo4j: bolt://localhost:7687
# 3. Set password: export NEO4J_PASSWORD=your_password
```

### Issue 3: Timezone Issues

**Problem:**
```python
Knowledge artifacts have inconsistent timestamps
```

**Solution:**
```python
# Always use UTC
from datetime import datetime, timezone

now = datetime.now(timezone.utc)  # ✅ Correct
# NOT: datetime.now()  # ❌ Uses local time
```

## Rollback Plan

### If Issues Occur

1. **Revert to old engine:**
```python
# Just use old import and initialization
from knowledge_engine.engine import KnowledgeEngine
engine = KnowledgeEngine()  # Works as before
```

2. **Keep both engines:**
```python
from knowledge_engine.engine import KnowledgeEngine
from knowledge_engine.core.temporal_knowledge_engine import TemporalKnowledgeEngine

# Use old engine for critical paths
old_engine = KnowledgeEngine()

# Use new engine for experimental features
new_engine = TemporalKnowledgeEngine(enable_temporal=True)
```

## Best Practices After Migration

### 1. Always Use Temporal Metadata

```python
# DO: Add temporal context
await engine.add_knowledge_temporal(
    content="Best practice",
    artifact_type="solution_pattern",
    valid_at=datetime.utcnow(),
)

# DON'T: Add without time context
```

### 2. Use Appropriate Artifact Types

```python
# Use standard types
artifact_type = "solution_pattern"  # ✅
artifact_type = "workflow"          # ✅
artifact_type = "problem"           # ✅

# NOT: Custom types without mapping
artifact_type = "my_custom_type"    # ❌
```

### 3. Set Expiration for Temporary Knowledge

```python
# DO: Set expiration
await engine.add_knowledge_temporal(
    content="Temporary fix",
    artifact_type="workflow",
    valid_at=now,
    invalid_at=now + timedelta(days=30),
)

# DON'T: Leave temporary knowledge valid indefinitely
```

### 4. Run Contradiction Detection Periodically

```python
# DO: Check for contradictions
result = await engine.detect_contradictions()
if result.has_contradictions:
    # Handle them
    pass

# DON'T: Check on every operation (too expensive)
```

## Support

### Documentation
- [Complete Guide](./GRAPITI_TEMPORAL_INTEGRATION.md)
- [Quick Reference](./GRAPITI_QUICK_REFERENCE.md)
- [Examples](../examples/temporal_graphiti_example.py)

### Getting Help
1. Check documentation
2. Review examples
3. Run test suite
4. Check logs for errors

## Summary

Migrating to TemporalKnowledgeEngine provides:
- ✅ Temporal reasoning capabilities
- ✅ Better search quality (hybrid)
- ✅ Contradiction detection
- ✅ Backward compatibility
- ✅ Graceful degradation

The migration is straightforward and most existing code will continue to work with minimal changes.
