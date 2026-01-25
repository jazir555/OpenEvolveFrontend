# KG-Gen Sprint 2 Quick Reference

## Quick Import

```python
# Import all components
from knowledge_engine.integrations.kggen import (
    # Extraction
    ExtractionPipeline,
    ExtractionResult,
    PipelineConfig,

    # Deduplication
    DeduplicationEngine,
    DeduplicationResult,
    DeduplicationMethod,

    # MCP Server
    KGGenMCPServer,
    MemoryType,
    MemoryQuery,

    # Conversation Analysis
    ConversationAnalyzer,

    # Graph Aggregation
    GraphAggregator,
)
```

## Common Patterns

### Basic Extraction
```python
pipeline = ExtractionPipeline()
result = await pipeline.extract(text=your_text)
await pipeline.close()
```

### With Deduplication
```python
pipeline = ExtractionPipeline()
dedup = DeduplicationEngine()

# Extract
result = await pipeline.extract(text=your_text)

# Deduplicate
dedup_result = await dedup.deduplicate(
    entities=result.entities,
    method=DeduplicationMethod.FULL
)

# Use dedup_result.unique_entities

await pipeline.close()
await dedup.close()
```

### Memory Operations
```python
server = KGGenMCPServer()

# Add memories
await server.add_memories(
    memories=[{"content": "fact", "memory_type": "fact"}],
    session_id="session1"
)

# Retrieve memories
result = await server.retrieve_relevant_memories(
    query_text="search terms",
    session_id="session1",
    max_results=10
)

await server.close()
```

### Conversation Analysis
```python
analyzer = ConversationAnalyzer()

messages = [
    {"role": "user", "content": "question", "speaker_id": "user1"},
    {"role": "assistant", "content": "answer", "speaker_id": "bot"}
]

result = await analyzer.analyze(messages=messages)

# Access result.entities, result.relationships, result.summary

await analyzer.close()
```

### Graph Aggregation
```python
aggregator = GraphAggregator()

graphs = [
    {"entities": ["A", "B"], "relationships": []},
    {"entities": ["B", "C"], "relationships": []}
]

result = await aggregator.aggregate(graphs=graphs)

# Access result.aggregated_graph

await aggregator.close()
```

## Deduplication Methods

| Method | Use Case | Speed | Accuracy |
|--------|----------|-------|----------|
| `DeduplicationMethod.SEMHASH` | Exact/near-exact duplicates | Fast | Medium |
| `DeduplicationMethod.LM_CLUSTER` | Semantic duplicates | Slow | High |
| `DeduplicationMethod.FULL` | Comprehensive | Slowest | Highest |

## Environment Variables (Quick Set)

```bash
# Essential
export KGGEN_ENTITY_MODEL="gpt-4o"
export KGGEN_RELATION_MODEL="gpt-4o"
export KGGEN_CHUNK_SIZE="5000"
export KGGEN_PARALLEL_WORKERS="4"

# Deduplication
export KGGEN_SEMHASH_THRESHOLD="0.95"
export KGGEN_LM_SIMILARITY_THRESHOLD="0.85"

# Memory
export KGGEN_MEMORY_PERSISTENCE="true"
export KGGEN_MEMORY_STORAGE_PATH="./data/kggen_memories"
```

## Testing Commands

```bash
# Run all tests
pytest knowledge_engine/integrations/kggen/test_sprint2.py -v

# Run specific test class
pytest knowledge_engine/integrations/kggen/test_sprint2.py::TestExtractionPipeline -v

# Run with coverage
pytest knowledge_engine/integrations/kggen/test_sprint2.py --cov=knowledge_engine.integrations.kggen
```

## Probe Commands

```bash
# Verify extraction
bash knowledge_engine/integrations/kggen/probes/check_extraction_pipeline.sh

# Verify deduplication
bash knowledge_engine/integrations/kggen/probes/check_deduplication_engine.sh

# Verify MCP server
bash knowledge_engine/integrations/kggen/probes/check_mcp_server.sh
```

## Result Objects

### ExtractionResult
```python
result.entities          # List[str]
result.relationships     # List[Dict[str, str]]
result.entity_count       # int
result.relationship_count # int
result.correlation_id     # str
result.processing_time_seconds # float
```

### DeduplicationResult
```python
result.unique_entities   # List[str]
result.entity_clusters   # List[EntityCluster]
result.original_count     # int
result.final_count        # int
result.duplicates_removed # int
result.reduction_rate     # float
```

### ConversationResult
```python
result.entities              # List[str]
result.relationships         # List[Dict[str, str]]
result.speaker_entities      # List[SpeakerEntity]
result.summary               # ConversationSummary
result.total_speakers        # int
result.total_entities        # int
```

### AggregationResult
```python
result.aggregated_graph   # GraphVersion
result.total_entities     # int
result.total_relationships # int
result.conflicts_resolved # int
```

## Common Issues

### Timeout
```python
# Increase timeout
config = PipelineConfig(
    entity_timeout=600.0,  # 10 minutes
    relation_timeout=1200.0
)
pipeline = ExtractionPipeline(config)
```

### Memory Issues
```python
# Reduce parallelism
config = PipelineConfig(
    parallel_workers=2,  # Reduce from 4
    chunk_size=3000      # Smaller chunks
)
```

### Import Errors
```python
# Add to PYTHONPATH
import sys
sys.path.append("/path/to/OpenEvolve/Frontend")
```

## Best Practices

1. **Always close resources**: `await pipeline.close()`
2. **Use correlation IDs**: For tracking and debugging
3. **Check results**: Validate reduction rates and cluster sizes
4. **Handle errors**: Try/except with proper logging
5. **Use progress callbacks**: For long-running operations

## CLAUDE.md Principles

- ✅ **AIR GAP**: Adapter pattern, no direct imports from kg-gen
- ✅ **RUNTIME TRUTH**: Probe scripts verify functionality
- ✅ **IDEMPOTENCY**: All operations safe to retry
- ✅ **CONFIGURATION EXPLICITNESS**: All config via env vars
- ✅ **UTC TIME**: All timestamps in UTC
- ✅ **STRUCTURED LOGGING**: JSON logs with correlation IDs

## Documentation

- **Integration Guide**: `SPRINT2_INTEGRATION_GUIDE.md`
- **Usage Examples**: `PIPELINE_USAGE_EXAMPLES.md`
- **Deduplication Tutorial**: `DEDUPLICATION_TUTORIAL.md`
- **Completion Report**: `SPRINT2_COMPLETION_REPORT.md`

## File Structure

```
knowledge_engine/integrations/kggen/
├── __init__.py
├── extraction_pipeline.py    # 3-stage extraction
├── deduplication_engine.py   # Advanced deduplication
├── mcp_server.py             # Memory MCP server
├── conversation_analyzer.py  # Conversation analysis
├── graph_aggregator.py       # Graph aggregation
├── test_sprint2.py           # Test suite
├── probes/                   # Runtime verification
└── *.md                      # Documentation
```

## Status

✅ **ALL 28 TASKS COMPLETE**
✅ **PRODUCTION-GRADE CODE**
✅ **FULLY TESTED**
✅ **COMPREHENSIVELY DOCUMENTED**
