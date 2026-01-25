# IntegratedKnowledgeEngine - Quick Reference Card

## Quick Start

```python
from knowledge_engine import IntegratedKnowledgeEngine

async with IntegratedKnowledgeEngine() as engine:
    # Use the engine
    result = await engine.process_document("doc.pdf")
```

## Common Operations

### 1. Document Processing

```python
# Single document
result = await engine.process_document("path/to/document.pdf")

# With options
from knowledge_engine.integrated_engine import ProcessingOptions
options = ProcessingOptions(extract_temporal=True, timeout_ms=60000)
result = await engine.process_document("doc.pdf", options)
```

### 2. Batch Processing

```python
# Batch with progress tracking
files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

def progress(msg, pct, meta):
    print(f"{msg}: {pct:.1f}%")

result = await engine.batch_process_documents(
    files,
    progress_callback=progress,
    max_concurrent=5
)

print(f"Success: {result.successful}/{result.total_items}")
```

### 3. Search Knowledge

```python
# Hybrid search
result = await engine.search_knowledge("machine learning", limit=10)

# Keyword search
result = await engine.search_knowledge("algorithm", query_type="keyword")

# With filters
result = await engine.search_knowledge(
    "query",
    filters={"type": "document"},
    limit=5
)
```

### 4. Code Analysis

```python
result = await engine.analyze_code("/path/to/repo")
print(f"Indexed: {result['indexed_files']} files")
print(f"Patterns: {result['patterns_found']}")
```

### 5. Temporal Queries

```python
from datetime import datetime, timezone

# Query knowledge at specific time
timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
result = await engine.query_temporal("AI", timestamp=timestamp)
```

### 6. Health & Statistics

```python
# Health check
health = await engine.health_check()
print(f"Status: {health['overall']}")

# Statistics
stats = await engine.get_statistics()
print(f"Entities: {stats['knowledge']['entities']}")
```

## Processing Options

```python
options = ProcessingOptions(
    extract_temporal=True,      # Use temporal extraction
    extract_bilingual=False,    # Use bilingual extraction
    use_embeddings=True,        # Generate embeddings
    validate_results=True,      # Validate results
    cache_results=True,         # Cache results
    timeout_ms=30000,          # Timeout (ms)
    max_retries=3,             # Max retries
    correlation_id="custom_id"  # Tracking ID
)
```

## Sprint Types

- `TEMPORAL_GRAPHITI` - Temporal knowledge tracking
- `BILINGUAL_ONEKE` - Multilingual extraction
- `GENERIC_KGGEN` - General document processing
- `HYBRID_AUTO` - Automatic selection

## Result Structures

### ProcessDocument Result
```python
{
    "success": True,
    "correlation_id": "doc_20250108_...",
    "document_path": "path/to/doc.pdf",
    "sprint_used": "temporal_graphiti",
    "entities": [...],
    "relations": [...],
    "artifacts_stored": 5,
    "processing_time_ms": 1234.56
}
```

### BatchResult
```python
{
    "total_items": 10,
    "successful": 8,
    "failed": 2,
    "success_rate": 0.8,
    "results": [...],
    "errors": [...],
    "total_time_ms": 15000.0
}
```

### SearchResult
```python
{
    "success": True,
    "query": "machine learning",
    "results": [...],
    "count": 10,
    "execution_time_ms": 123.45,
    "correlation_id": "search_..."
}
```

## Environment Variables

### Required
- `GRAPHITI_PASSWORD` - Neo4j password
- `OPENAI_API_KEY` - OpenAI API key (for KG-Gen)

### Common Optional
- `GRAPHITI_URI` - Neo4j URI (default: bolt://localhost:7687)
- `ELASTICSEARCH_HOSTS` - Elasticsearch (default: http://localhost:9200)
- `DEFAULT_TIMEOUT_MS` - Timeout (default: 30000)
- `MAX_RETRIES` - Retry count (default: 3)

## Error Handling

```python
result = await engine.process_document("doc.pdf")

if not result['success']:
    print(f"Error: {result['error']}")
    print(f"Correlation ID: {result['correlation_id']}")
    # Handle error
else:
    # Process result
    pass
```

## Best Practices

1. **Use context managers**
   ```python
   async with IntegratedKnowledgeEngine() as engine:
       # Auto cleanup
   ```

2. **Check results**
   ```python
   if result['success']:
       # Continue
   ```

3. **Use correlation IDs**
   ```python
   options = ProcessingOptions(correlation_id="batch_123")
   ```

4. **Set timeouts**
   ```python
   options = ProcessingOptions(timeout_ms=60000)
   ```

5. **Monitor progress**
   ```python
   result = await engine.batch_process_documents(
       files,
       progress_callback=lambda msg, pct, meta: print(f"{pct}%")
   )
   ```

## Troubleshooting

### Initialization fails
- Check environment variables
- Verify services are running (Neo4j, MongoDB, etc.)
- Check network connectivity

### Processing fails
- Verify file format (PDF, TXT, MD)
- Increase timeout
- Check file permissions

### No search results
- Verify documents processed
- Try different query_type
- Check filters

### Performance issues
- Reduce `max_concurrent`
- Enable caching
- Check system resources

## File Paths

- Implementation: `knowledge_engine/integrated_engine.py`
- Tests: `knowledge_engine/tests/test_integrated_engine.py`
- Guide: `knowledge_engine/INTEGRATED_ENGINE_GUIDE.md`
- Summary: `INTEGRATED_ENGINE_IMPLEMENTATION_SUMMARY.md`

## Get Help

- Full guide: `INTEGRATED_ENGINE_GUIDE.md`
- API docs: Docstrings in source code
- Examples: See guide for real-world examples
- Tests: See test file for usage examples

---
**Version**: 2.0.0 | **Status**: Production Ready
