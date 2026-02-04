# Ragbits Integration Guide

## Overview

The Ragbits integration provides advanced retrieval-augmented generation (RAG) capabilities to the Knowledge Engine. Ragbits enables efficient document search, semantic retrieval, and knowledge augmentation for language models.

### Key Features
- Document ingestion and indexing
- Semantic search with embeddings
- Vector store integration
- Query rephrasing and expansion
- Result reranking
- Hybrid search (keyword + semantic)
- Multiple vector store backends

### Use Cases
- Building RAG applications
- Document search systems
- Knowledge base queries
- Context-augmented generation
- Semantic similarity search
- Question-answering systems

## Installation

```bash
# Core installation
pip install ragbits

# With specific vector store
pip install ragbits[qdrant]  # For Qdrant
pip install ragbits[chroma]  # For ChromaDB
pip install ragbits[pinecone]  # For Pinecone

# With Knowledge Engine
pip install knowledge-engine[ragbits]
```

### Configuration

Set up environment variables:

```bash
export RAGBITS_VECTOR_STORE="qdrant"
export RAGBITS_COLLECTION_NAME="knowledge_docs"
export RAGBITS_EMBEDDING_MODEL="text-embedding-3-small"
export QDRANT_URL="http://localhost:6333"
```

## Quick Start

### Basic Usage

```python
from knowledge_engine.integrations import RagbitsIntegration

# Initialize integration
integration = RagbitsIntegration()

# Ingest documents
documents = [
    {"content": "Document 1 text...", "metadata": {"source": "doc1.pdf"}},
    {"content": "Document 2 text...", "metadata": {"source": "doc2.pdf"}},
]

await integration.ingest_documents(documents)

# Search
results = await integration.search(
    query="What is machine learning?",
    top_k=5
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
    print(f"Metadata: {result.metadata}")
```

### Advanced Search

```python
# Search with filters
results = await integration.search(
    query="Neural networks",
    top_k=10,
    filters={
        "source": "research_papers",
        "year": {"gte": 2020}
    }
)

# Hybrid search
results = await integration.hybrid_search(
    query="deep learning",
    keyword_weight=0.3,  # 30% keyword search
    semantic_weight=0.7,  # 70% semantic search
    top_k=5
)
```

## Configuration Options

### Full Configuration Schema

```python
config = {
    # Vector Store Configuration
    "vector_store": {
        "type": "qdrant",  # qdrant, chroma, pinecone, weaviate
        "config": {
            "location": ":memory:",  # :memory:, http://..., or path
            "url": None,  # For remote vector stores
            "api_key": None,  # API key for cloud services
            "collection_name": "knowledge_artifacts",
            "vector_size": 1536,  # Embedding dimension
            "distance": "Cosine"  # Cosine, Euclidean, Dot
        }
    },

    # Embedding Configuration
    "embeddings": {
        "model": "text-embedding-3-small",
        "dimension": 1536,
        "batch_size": 100,
        "normalize": True,
        "show_progress": True
    },

    # Search Configuration
    "search": {
        "default_top_k": 10,
        "similarity_threshold": 0.7,
        "reranker": {
            "type": "noop",  # noop, cohere, colbert, mmr
            "top_k": 5
        },
        "query_rephraser": {
            "type": "noop",  # noop, llm, hyde
            "model": "gpt-4o"
        }
    },

    # Ingestion Configuration
    "ingestion": {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "strategy": "recursive",  # recursive, fixed, semantic
        "max_workers": 4
    },

    # Indexing Configuration
    "indexing": {
        "create_index": True,
        "index_type": "HNSW",  # HNSW, IVF, FLAT
        "index_params": {
            "m": 16,  # HNSW parameter
            "ef_construction": 100
        }
    }
}
```

## API Reference

### Core Methods

#### `ingest_documents(documents, options)`

Ingest documents into the vector store.

**Parameters:**
- `documents` (List[dict]): List of documents
  - Each document has:
    - `content` (str): Document text
    - `metadata` (dict, optional): Document metadata
- `options` (dict, optional): Ingestion options

**Returns:** Ingestion result with:
- `success` (bool): Success status
- `documents_ingested` (int): Number of documents
- `chunks_created` (int): Number of chunks
- `processing_time_ms` (float): Processing time

**Example:**
```python
documents = [
    {
        "content": "This is a document...",
        "metadata": {
            "title": "Doc 1",
            "author": "John Doe",
            "date": "2025-01-01"
        }
    }
]

result = await integration.ingest_documents(documents)
```

#### `search(query, top_k, filters, options)`

Search for similar documents.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results
- `filters` (dict, optional): Metadata filters
- `options` (dict, optional): Search options

**Returns:** Search results:
```python
[
    {
        "content": str,
        "metadata": dict,
        "score": float,
        "distance": float
    },
    ...
]
```

**Example:**
```python
results = await integration.search(
    query="Machine learning algorithms",
    top_k=5,
    filters={"category": "AI"}
)
```

#### `hybrid_search(query, keyword_weight, semantic_weight, options)`

Perform hybrid keyword + semantic search.

**Parameters:**
- `query` (str): Search query
- `keyword_weight` (float): Weight for keyword search (0-1)
- `semantic_weight` (float): Weight for semantic search (0-1)
- `options` (dict, optional): Search options

**Returns:** Combined and ranked results

**Example:**
```python
results = await integration.hybrid_search(
    query="neural network architecture",
    keyword_weight=0.3,
    semantic_weight=0.7,
    top_k=10
)
```

#### `delete_documents(filter_dict)`

Delete documents matching filters.

**Parameters:**
- `filter_dict` (dict): Filter criteria

**Returns:** Deletion result

**Example:**
```python
await integration.delete_documents(
    filters={"source": "outdated_docs"}
)
```

#### `update_document(document_id, content, metadata)`

Update an existing document.

**Parameters:**
- `document_id` (str): Document ID
- `content` (str, optional): New content
- `metadata` (dict, optional): New metadata

**Returns:** Update result

## Advanced Usage

### Custom Chunking Strategies

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Custom chunking
config = {
    "ingestion": {
        "strategy": "custom",
        "chunker": RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    }
}
integration = RagbitsIntegration(config=config)
```

### Query Rephrasing

```python
config = {
    "search": {
        "query_rephraser": {
            "type": "llm",
            "model": "gpt-4o",
            "num_rephrasings": 3
        }
    }
}
integration = RagbitsIntegration(config=config)

# Query will be rephrased for better results
results = await integration.search(
    query="How do neural networks learn?"
)
# Internally searches:
# - "How do neural networks learn?"
# - "Neural network training process"
# - "Learning algorithms in neural networks"
```

### Result Reranking

```python
config = {
    "search": {
        "reranker": {
            "type": "cohere",  # Requires Cohere API key
            "model": "rerank-english-v2.0",
            "top_k": 5
        }
    }
}
integration = RagbitsIntegration(config=config)

results = await integration.search(query)
# Results are reranked by Cohere for better relevance
```

### MMR (Maximal Marginal Relevance)

```python
config = {
    "search": {
        "reranker": {
            "type": "mmr",
            "diversity": 0.3,  # 0 = no diversity, 1 = maximum diversity
            "top_k": 10
        }
    }
}
integration = RagbitsIntegration(config=config)

results = await integration.search(query)
# Returns diverse results, not just similar ones
```

### Multiple Vector Stores

```python
from ragbits.core.vector_stores import QdrantVectorStore, ChromaVectorStore

# Primary store
primary = QdrantVectorStore(url="http://localhost:6333")

# Fallback store
fallback = ChromaVectorStore(persist_directory="./chroma_db")

integration = RagbitsIntegration(config={
    "vector_store": {
        "primary": primary,
        "fallback": fallback
    }
})
```

## Integration with Knowledge Engine

### Using with DSPy (RAG Pattern)

```python
from knowledge_engine.integrations import RagbitsIntegration, DSPyIntegration

ragbits = RagbitsIntegration()
dspy = DSPyIntegration()

async def rag_pipeline(question: str):
    # Retrieve context
    context = await ragbits.search(
        query=question,
        top_k=5
    )

    # Generate answer with context
    answer = await dspy.chain_of_thought(
        query=question,
        context={
            "documents": [doc["content"] for doc in context]
        }
    )

    return answer
```

### Using with ROMA-Ragbits Integration

```python
from knowledge_engine.integrations import ROMARagbitsIntegration

# Use integrated ROMA-Ragbits
roma_ragbits = ROMARagbitsIntegration()

result = await roma_ragbits.solve_with_retrieval(
    problem="Question requiring external knowledge",
    retrieval_config={
        "top_k": 10,
        "similarity_threshold": 0.75
    }
)
```

### Using with LeanAIDE-Ragbits Integration

```python
from knowledge_engine.integrations import LeanAideragbitsIntegration

# Literature-assisted proving
leanaide_ragbits = LeanAideragbitsIntegration()

result = await leanaide_ragbits.prove_with_literature(
    theorem="Statement to prove",
    retrieval_config={
        "domain": "mathematics",
        "top_k": 20
    }
)
```

## Performance Considerations

### Batch Ingestion

```python
# Ingest large collections efficiently
documents = load_large_corpus()  # 10,000+ documents

result = await integration.ingest_documents(
    documents,
    batch_size=100,  # Process 100 at a time
    max_workers=4  # Parallel processing
)
```

### Index Optimization

```python
config = {
    "indexing": {
        "index_type": "HNSW",
        "index_params": {
            "m": 16,  # Higher m = better recall, more memory
            "ef_construction": 100  # Higher = better quality, slower build
        }
    }
}
```

### Caching

```python
from functools import lru_cache

class CachedRagbits:
    def __init__(self, integration):
        self.integration = integration

    @lru_cache(maxsize=1000)
    async def cached_search(self, query: str, top_k: int):
        return await self.integration.search(query, top_k)
```

## Vector Store Options

### Qdrant (Recommended)

```python
config = {
    "vector_store": {
        "type": "qdrant",
        "config": {
            "location": "http://localhost:6333",
            "collection_name": "documents",
            "vector_size": 1536
        }
    }
}
```

### ChromaDB (Easy Setup)

```python
config = {
    "vector_store": {
        "type": "chroma",
        "config": {
            "persist_directory": "./chroma_db",
            "collection_name": "documents"
        }
    }
}
```

### Pinecone (Managed)

```python
config = {
    "vector_store": {
        "type": "pinecone",
        "config": {
            "api_key": os.getenv("PINECONE_API_KEY"),
            "environment": "us-east-1-aws",
            "index_name": "documents"
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Connection Error**
   ```python
   # Check vector store is running
   # For Qdrant:
   curl http://localhost:6333/health
   ```

2. **Low Search Quality**
   ```python
   # Adjust similarity threshold
   config = {"search": {"similarity_threshold": 0.6}}
   ```

3. **Slow Ingestion**
   ```python
   # Increase batch size and workers
   config = {
       "ingestion": {
           "batch_size": 200,
           "max_workers": 8
       }
   }
   ```

## Examples

See `examples/ragbits/`:
- `basic_search.py` - Basic search
- `rag_pipeline.py` - RAG with generation
- `hybrid_search.py` - Hybrid keyword + semantic
- `advanced_reranking.py` - Result reranking
- `multi_vector_store.py` - Multiple backends

## References

- [Ragbits Documentation](https://github.com/deepset-ai/ragbits)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

**Last Updated**: 2025-02-03
**Integration Version**: 1.0.0
