# RAGBits Document Processor - Quick Reference

## Import

```python
from knowledge_engine.ragbits_document_processor import (
    RAGBitsDocumentProcessor,
    RAGBitsProcessorConfig
)
```

## Basic Usage

```python
# Create and initialize
processor = RAGBitsDocumentProcessor()
await processor.initialize()

# Ingest document
result = await processor.ingest_text("Your text here")
print(f"Document ID: {result.document_id}")

# Search
results = await processor.search("search query", top_k=5)
for r in results:
    print(f"{r['score']:.3f}: {r['content']}")
```

## Common Operations

### Ingest from Text
```python
await processor.ingest_text(
    text="Document content...",
    metadata={"title": "My Doc", "category": "AI"},
    source="manual"
)
```

### Ingest from File
```python
await processor.ingest_file(
    "path/to/document.txt",
    metadata={"author": "John"}
)
```

### Ingest Directory
```python
results = await processor.ingest_directory(
    directory="./docs",
    pattern="*.md",
    metadata={"project": "MyProject"},
    max_files=50
)
```

### Search (Basic)
```python
results = await processor.search("query text", top_k=10)
```

### Search (With Filters)
```python
results = await processor.search(
    "query",
    top_k=5,
    filters={"category": "AI"},
    min_score=0.7
)
```

### Get Statistics
```python
stats = await processor.get_statistics()
print(f"Documents: {stats['ingested_documents']}")
print(f"Store: {stats['vector_store_type']}")
```

### Clear All Documents
```python
await processor.clear()
```

## Configuration

### Environment Variables
```bash
export RAGBITS_EMBEDDING_MODEL="text-embedding-3-small"
export RAGBITS_VECTOR_STORE="memory"  # or "qdrant"
export RAGBITS_QDRANT_URL="http://localhost:6333"
export RAGBITS_QDRANT_COLLECTION="knowledge_engine"
export RAGBITS_CHUNK_SIZE="1000"
export RAGBITS_CHUNK_OVERLAP="200"
```

### Configuration Object
```python
config = RAGBitsProcessorConfig(
    embedding_model="text-embedding-3-small",
    vector_store_type="qdrant",  # or "memory"
    qdrant_url="http://localhost:6333",
    qdrant_collection="my_collection",
    chunk_size=1000,
    chunk_overlap=200
)

processor = RAGBitsDocumentProcessor(config)
```

## Return Values

### DocumentProcessingResult
```python
{
    "success": True,
    "document_id": "doc_abc123_def456",
    "chunks_ingested": 1,
    "processing_time": 0.15,
    "error": None,
    "metadata": {...}
}
```

### Search Results
```python
[
    {
        "content": "Document text...",
        "score": 0.95,
        "metadata": {"title": "My Doc", "category": "AI"}
    },
    ...
]
```

## Error Handling

```python
# Check if initialized
if not await processor.initialize():
    print("RAGBits not available, using fallback")

# Check processing result
result = await processor.ingest_text("text")
if not result.success:
    print(f"Error: {result.error}")

# Handle empty search results
results = await processor.search("query")
if not results:
    print("No results found")
```

## Idempotency

```python
# Safe to run multiple times
doc_id = "unique_source"

# First ingestion
result1 = await processor.ingest_text("text", source=doc_id)
# result1.chunks_ingested = 1

# Second ingestion (same content)
result2 = await processor.ingest_text("text", source=doc_id)
# result2.chunks_ingested = 0 (skipped)
```

## Storage Options

### In-Memory (Default)
```python
# No setup required
processor = RAGBitsDocumentProcessor()
```

### Qdrant (Persistent)
```python
# Requires Qdrant running
config = RAGBitsProcessorConfig(
    vector_store_type="qdrant",
    qdrant_url="http://localhost:6333",
    qdrant_collection="my_docs"
)
processor = RAGBitsDocumentProcessor(config)
```

## Common Patterns

### Build Knowledge Base
```python
# Ingest all docs
await processor.ingest_directory("./docs", "*.md")

# Search
results = await processor.search("deployment guide")
```

### Filter by Category
```python
# Ingest with categories
await processor.ingest_text(
    "Text about ML",
    metadata={"category": "AI", "topic": "ML"}
)

# Search with filter
results = await processor.search(
    "machine learning",
    filters={"category": "AI"}
)
```

### Minimum Score
```python
# Only high-relevance results
results = await processor.search(
    "query",
    min_score=0.8  # 80% similarity threshold
)
```

## Quick Test

```python
import asyncio
from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor

async def test():
    p = RAGBitsDocumentProcessor()
    if await p.initialize():
        await p.ingest_text("Test document about Python programming")
        results = await p.search("Python")
        print(f"Found {len(results)} results")

asyncio.run(test())
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "RAGBits not available" | `pip install ragbits-document-search ragbits-core` |
| "Qdrant connection failed" | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| Slow ingestion | Increase chunk size or process in batches |
| Poor search results | Check document quality, add more context |

## Tips

✅ Use rich metadata for better filtering
✅ Adjust chunk size based on document type
✅ Use Qdrant for production (persistent storage)
✅ Use in-memory for testing (no setup)
✅ Safe to re-ingest (idempotent)
✅ Add correlation IDs for tracking

⚠️ Don't make chunks too small (< 100)
⚠️ Don't make chunks too large (> 2000)
⚠️ Don't skip metadata (essential for filtering)

---

**See Also**: [Complete Guide](RAGBITS_DOCUMENT_PROCESSOR_GUIDE.md)
