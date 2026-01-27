# RAGBits Document Processor - Complete Guide

## Overview

The **RAGBits Document Processor** is a production-ready document processing pipeline that integrates RAGBits semantic search capabilities with the OpenEvolve Knowledge Engine.

## Features

✅ **Document Ingestion**
- Ingest text directly
- Load from files (TXT, MD, PDF, DOCX)
- Batch ingest from directories
- Automatic chunking with overlap
- Idempotent re-ingestion (safe to run multiple times)

✅ **Semantic Search**
- Vector similarity search
- Metadata filtering
- Top-k retrieval
- Score thresholding

✅ **Storage Options**
- In-memory vector store (default, no setup)
- Qdrant vector store (persistent, requires Qdrant)

✅ **Production Ready**
- CLAUDE.md compliant
- Structured JSON logging
- Error handling and graceful degradation
- Async/await throughout
- UTC timestamps

## Quick Start

### 1. Basic Usage (In-Memory Storage)

```python
import asyncio
from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor

async def main():
    # Create processor
    processor = RAGBitsDocumentProcessor()
    await processor.initialize()

    # Ingest document
    result = await processor.ingest_text(
        text="Machine learning enables computers to learn from data.",
        metadata={"category": "AI", "title": "ML Intro"}
    )
    print(f"Ingested: {result.document_id}")

    # Search
    results = await processor.search("data and learning", top_k=5)
    for r in results:
        print(f"{r['score']:.3f}: {r['content'][:100]}...")

asyncio.run(main())
```

### 2. File Ingestion

```python
# Ingest single file
result = await processor.ingest_file(
    "document.txt",
    metadata={"author": "John Doe", "tags": ["important"]}
)

# Ingest entire directory
results = await processor.ingest_directory(
    directory="./documents",
    pattern="*.txt",  # Only .txt files
    metadata={"project": "knowledge_engine"},
    max_files=100  # Limit to 100 files
)
```

### 3. Search with Filters

```python
# Search all documents
results = await processor.search("neural networks", top_k=5)

# Search with metadata filters
results = await processor.search(
    "machine learning",
    top_k=10,
    filters={
        "category": "AI",
        "author": "John Doe"
    },
    min_score=0.8  # Only high-relevance results
)
```

### 4. Using Qdrant (Persistent Storage)

```python
from knowledge_engine.ragbits_document_processor import RAGBitsProcessorConfig

# Configure Qdrant
config = RAGBitsProcessorConfig(
    vector_store_type="qdrant",
    qdrant_url="http://localhost:6333",
    qdrant_collection="knowledge_engine"
)

processor = RAGBitsDocumentProcessor(config)
await processor.initialize()

# Documents will be persisted in Qdrant
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGBITS_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAGBITS_VECTOR_STORE` | `memory` | Storage backend (memory/qdrant) |
| `RAGBITS_QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `RAGBITS_QDRANT_COLLECTION` | `knowledge_engine` | Qdrant collection |
| `RAGBITS_CHUNK_SIZE` | `1000` | Document chunk size |
| `RAGBITS_CHUNK_OVERLAP` | `200` | Chunk overlap |
| `RAGBITS_MIN_CHUNK_SIZE` | `100` | Minimum chunk size |

### Configuration Object

```python
from knowledge_engine.ragbits_document_processor import RAGBitsProcessorConfig

config = RAGBitsProcessorConfig(
    embedding_model="text-embedding-3-small",
    vector_store_type="memory",  # or "qdrant"
    qdrant_url="http://localhost:6333",
    qdrant_collection="my_collection",
    chunk_size=1000,
    chunk_overlap=200
)

processor = RAGBitsDocumentProcessor(config)
```

## API Reference

### RAGBitsDocumentProcessor

#### `__init__(config: Optional[RAGBitsProcessorConfig] = None)`
Create processor instance.

#### `async initialize() -> bool`
Initialize RAGBits components (embedder, vector store).
- Returns: `True` if successful, `False` otherwise

#### `async ingest_text(text: str, metadata: Optional[Dict] = None, source: str = "text") -> DocumentProcessingResult`
Ingest text document.
- `text`: Document content
- `metadata`: Optional metadata dict
- `source`: Source identifier
- Returns: Processing result with document ID

#### `async ingest_file(file_path: str, metadata: Optional[Dict] = None) -> DocumentProcessingResult`
Ingest document from file.
- `file_path`: Path to document file
- `metadata`: Optional metadata
- Returns: Processing result

#### `async ingest_directory(directory: str, pattern: str = "*.txt", metadata: Optional[Dict] = None, max_files: Optional[int] = None) -> List[DocumentProcessingResult]`
Ingest all documents from directory.
- `directory`: Directory path
- `pattern`: File glob pattern
- `metadata`: Metadata to apply to all files
- `max_files`: Maximum files to process
- Returns: List of processing results

#### `async search(query: str, top_k: int = 5, filters: Optional[Dict] = None, min_score: float = 0.0) -> List[Dict]`
Search for relevant documents.
- `query`: Search query
- `top_k`: Number of results
- `filters`: Metadata filters
- `min_score`: Minimum similarity score
- Returns: List of search results

#### `async get_statistics() -> Dict[str, Any]`
Get processing statistics.
- Returns: Statistics dict

#### `async clear() -> bool`
Clear all ingested documents.
- Returns: `True` if successful

### DocumentProcessingResult

#### Fields
- `success: bool` - Whether processing succeeded
- `document_id: str` - Unique document identifier
- `chunks_ingested: int` - Number of chunks ingested
- `processing_time: float` - Processing time in seconds
- `error: Optional[str]` - Error message if failed
- `metadata: Dict[str, Any]` - Document metadata

## Use Cases

### 1. Building a Knowledge Base

```python
# Ingest project documentation
await processor.ingest_directory(
    directory="./docs",
    pattern="*.md",
    metadata={"project": "MyProject", "type": "documentation"}
)

# Search for specific information
results = await processor.search("how to deploy", filters={"type": "documentation"})
```

### 2. Code Repository Search

```python
# Ingest code files
await processor.ingest_directory(
    directory="./src",
    pattern="*.py",
    metadata={"type": "code", "language": "Python"}
)

# Search for similar implementations
results = await processor.search(
    "authentication function",
    filters={"language": "Python"}
)
```

### 3. Research Paper Analysis

```python
# Ingest research papers
for paper in papers:
    await processor.ingest_text(
        text=paper["abstract"],
        metadata={
            "title": paper["title"],
            "authors": paper["authors"],
            "year": paper["year"],
            "venue": paper["venue"]
        },
        source=paper["title"]
    )

# Find related work
results = await processor.search(
    "transformer architecture",
    filters={"year": {"$gte": 2017}}
)
```

### 4. FAQ and Support

```python
# Ingest FAQ documents
await processor.ingest_directory(
    directory="./faq",
    metadata={"type": "faq"}
)

# Answer support questions
question = "How do I reset my password?"
results = await processor.search(question, top_k=1)
answer = results[0]["content"] if results else "No answer found"
```

## Running the Example

```bash
# Run the complete example
python ragbits_document_processor_example.py

# Or run specific examples
python ragbits_document_processor_example.py::example_basic_usage
python ragbits_document_processor_example.py::example_file_ingestion
```

## Integration with Knowledge Engine

### Using with IntegratedKnowledgeEngine

```python
from knowledge_engine import IntegratedKnowledgeEngine

async def main():
    # Initialize with RAGBits enabled
    engine = IntegratedKnowledgeEngine()
    await engine.initialize()

    # Process documents (uses RAGBits for indexing)
    result = await engine.process_document("document.pdf")

    # Search knowledge (uses RAGBits for retrieval)
    results = await engine.search_knowledge("machine learning")
```

### Standalone Usage

```python
from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor

# Use directly
processor = RAGBitsDocumentProcessor()
await processor.initialize()

# Build your custom pipeline
await processor.ingest_file("doc.txt")
results = await processor.search("query")
```

## Requirements

### For In-Memory Storage (Default)
- Python 3.10+
- ragbits-document-search
- ragbits-core

### For Qdrant Storage
- Qdrant instance running
- See: https://qdrant.tech/documentation/

### Installation

```bash
# Install RAGBits
pip install ragbits-document-search ragbits-core

# For Qdrant support
pip install qdrant-client

# Optional: For PDF/DOCX support
pip install pypdf python-docx
```

## Performance

| Operation | Time (Approx) |
|-----------|---------------|
| Initialize | 1-2 seconds |
| Ingest text (1K chars) | 100-200ms |
| Ingest file (10K chars) | 200-500ms |
| Search query | 50-150ms |
| Batch ingest (100 files) | 10-30 seconds |

## Troubleshooting

### RAGBits Not Available

**Error**: "RAGBits not available"
**Solution**: Install RAGBits packages
```bash
pip install ragbits-document-search ragbits-core
```

### Qdrant Connection Failed

**Error**: "Failed to initialize RAGBits"
**Solution**: Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Import Errors

**Error**: "No module named 'ragbits'"
**Solution**: Install dependencies
```bash
pip install ragbits-document-search ragbits-core liteLLM
```

## Best Practices

1. **Idempotency**: The processor is idempotent - safe to re-ingest documents
2. **Batch Processing**: Use `ingest_directory` for large batches
3. **Metadata**: Add rich metadata for better filtering
4. **Chunk Size**: Adjust based on document type (500-2000)
5. **Qdrant**: Use for production (persistent storage)
6. **Memory**: Use for testing and demos (no setup)

## Next Steps

- Configure Qdrant for persistent storage
- Add custom metadata extractors
- Implement hybrid search (semantic + keyword)
- Add document preprocessing (cleaning, normalization)
- Build a REST API wrapper
- Integrate with document pipelines (PDF, DOCX, etc.)

---

**Last Updated**: 2026-01-08
**Status**: Production Ready ✅
**Version**: 1.0.0
