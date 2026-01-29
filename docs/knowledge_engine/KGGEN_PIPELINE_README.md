# KG-Gen Graph Generation Pipeline

## Overview

The KG-Gen Pipeline Integration provides advanced knowledge graph extraction capabilities using a 3-stage pipeline:

1. **Entity Extraction** - Extract entities with DSPy
2. **Relation Extraction** - Extract SPO (Subject-Predicate-Object) triples
3. **Deduplication** - SEMHASH + LM clustering for entity consolidation

Features:
- Intelligent document chunking with sentence boundary preservation
- Parallel chunk processing for large documents
- Neo4j auto-upload with batch operations
- Configurable pipeline stages
- Progress tracking and error handling
- Multiple export formats (JSON, CSV, GraphML)

## Installation

### Prerequisites

```bash
# Install Python dependencies
pip install neo4j pyyaml nltk

# Download NLTK data (for sentence tokenization)
python -m nltk.downloader punkt
```

### Optional: Neo4j Setup

```bash
# Using Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### Configuration

Set environment variables for Neo4j:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
```

## Quick Start

### Basic Usage

```python
from knowledge_engine.engine import KnowledgeEngine

# Initialize the knowledge engine
engine = KnowledgeEngine()

# Extract knowledge graph from text
text = """
Python is a high-level programming language created by Guido van Rossum.
Python is widely used for web development, data science, and machine learning.
"""

graph = await engine.extract_knowledge_graph(
    text=text,
    context="Programming languages",
    upload_to_neo4j=False
)

# Access results
print(f"Entities: {len(graph.entities)}")
print(f"Relationships: {len(graph.relationships)}")
```

### Large Document Processing

```python
# Process large documents with parallel chunking
graph = await engine.extract_from_document(
    document_path="large_document.txt",
    chunk_size=5000
)

print(f"Extracted {len(graph.entities)} entities")
print(f"Extracted {len(graph.relationships)} relationships")
```

### Neo4j Upload

```python
# Extract and upload to Neo4j
graph = await engine.extract_knowledge_graph(
    text=text,
    upload_to_neo4j=True
)

# Query Neo4j statistics
stats = await engine.get_neo4j_statistics()
print(f"Neo4j entities: {stats['entity_count']}")
print(f"Neo4j relationships: {stats['relationship_count']}")
```

## Pipeline Stages

### Stage 1: Entity Extraction

Extracts named entities from text using DSPy or fallback NER:

```python
entities = await pipeline._extract_entities(
    text="Python was created by Guido van Rossum.",
    context="Programming languages"
)
# Returns: ["Python", "Guido van Rossum"]
```

### Stage 2: Relation Extraction

Extracts relationships between entities:

```python
relationships = await pipeline._extract_relations(
    text="Python is a programming language.",
    entities=["Python", "programming language"],
    context="Technology"
)
# Returns: [("Python", "is_a", "programming language")]
```

### Stage 3: Deduplication

Removes duplicate entities using semantic hashing and LM clustering:

```python
deduped_graph = await pipeline._deduplicate_graph(
    graph,
    method='full'  # 'semhash', 'lm_cluster', or 'full'
)
```

## Advanced Features

### Custom Chunking Strategies

```python
from knowledge_engine.integrations.kggen_chunking import DocumentChunker

# Sentence-based chunking
chunker = DocumentChunker(chunk_size=5000, overlap=200)
chunks = chunker.chunk_with_preservation(text, preserve_sentences=True)

# Paragraph-based chunking
chunks = chunker.chunk_by_paragraphs(text, max_paragraphs_per_chunk=10)

# Semantic unit chunking
chunks = chunker.chunk_by_semantic_units(text)
```

### Parallel Processing

```python
from knowledge_engine.integrations.kggen_parallel import ParallelChunkProcessor

processor = ParallelChunkProcessor(max_workers=4)

# With progress tracking
def progress_callback(progress):
    print(f"{progress.completion_percentage:.1f}% complete")

results = await processor.process_with_progress(
    chunks,
    processor_func=lambda chunk: extract_kg(chunk.text),
    progress_callback=progress_callback
)
```

### Batch Processing

```python
# Process multiple texts in batch
texts = [
    "Text 1 about topic A.",
    "Text 2 about topic B.",
    "Text 3 about topic C."
]

graphs = await engine.extract_batch_knowledge_graphs(texts)
```

### Neo4j Operations

```python
# Query specific entity
entity_data = await engine.query_neo4j_entity("Python")
print(entity_data)

# Export graph
json_export = await engine.export_neo4j_graph(format='json')
csv_export = await engine.export_neo4j_graph(format='csv')
graphml_export = await engine.export_neo4j_graph(format='graphml')
```

## Configuration

Edit `knowledge_engine/config/kggen_pipeline.yaml`:

```yaml
pipeline:
  enabled: true
  default_chunk_size: 5000
  default_overlap: 200
  parallel_workers: 4

stages:
  entity_extraction:
    model: "openai/gpt-4o"
    temperature: 0.0
    max_tokens: 4000

  relation_extraction:
    model: "openai/gpt-4o"
    temperature: 0.0
    max_tokens: 8000

  deduplication:
    method: full
    semhash_threshold: 0.95
    lm_cluster_size: 128

neo4j_upload:
  enabled: true
  batch_size: 100
  create_indices: true
```

## Performance Optimization

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
async def extract_cached(text_hash: str):
    return await pipeline.extract_knowledge_graph(text)
```

### Streaming Upload

```python
async def upload_streaming(graph: KnowledgeGraph):
    async for batch in graph.batches(batch_size=100):
        await neo4j.upload_batch(batch)
```

### Memory Management

```yaml
# In kggen_pipeline.yaml
performance:
  enable_cache: true
  enable_batching: true
  enable_streaming: true
  memory_limit: 2048  # MB
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest knowledge_engine/integrations/test_kggen_pipeline.py -v

# Run specific test class
pytest knowledge_engine/integrations/test_kggen_pipeline.py::TestKnowledgeGraph -v

# Run performance tests
pytest knowledge_engine/integrations/test_kggen_pipeline.py::TestPerformance -v
```

## Examples

Run example scripts:

```bash
# Run all examples
python knowledge_engine/examples/kggen_pipeline_example.py

# Run specific example
python knowledge_engine/examples/kggen_pipeline_example.py 1
```

Available examples:
1. Simple Extraction
2. Large Document Processing
3. Batch Processing
4. Custom Context
5. Neo4j Integration
6. Export Graph
7. Query Entity
8. Advanced Chunking
9. Progress Tracking
10. Complete Workflow

## API Reference

### KnowledgeEngine

```python
# Extract knowledge graph
await engine.extract_knowledge_graph(
    text: str,
    context: str = "",
    upload_to_neo4j: bool = True
) -> KnowledgeGraph

# Extract from document
await engine.extract_from_document(
    document_path: str,
    chunk_size: int = 5000
) -> KnowledgeGraph

# Batch extraction
await engine.extract_batch_knowledge_graphs(
    texts: List[str]
) -> List[KnowledgeGraph]

# Query Neo4j
await engine.query_neo4j_entity(
    entity_name: str
) -> Optional[Dict[str, Any]]

# Get statistics
await engine.get_neo4j_statistics() -> Dict[str, Any]

# Export graph
await engine.export_neo4j_graph(
    format: str = 'json'
) -> str

# Cleanup
await engine.cleanup_kggen_pipeline()
```

### KnowledgeGraph

```python
# Create graph
graph = KnowledgeGraph(
    entities: List[str],
    relationships: List[Tuple[str, str, str]],
    metadata: Optional[Dict[str, Any]]
)

# Add entities
graph.add_entity(entity: str)

# Add relationships
graph.add_relationship(subject: str, predicate: str, obj: str)

# Merge graphs
graph.merge(other: KnowledgeGraph)

# Convert to dict
data = graph.to_dict()
```

### DocumentChunker

```python
chunker = DocumentChunker(chunk_size: int = 5000, overlap: int = 200)

# Chunk document
chunks = chunker.chunk_document(text: str) -> List[Chunk]

# With preservation
chunks = chunker.chunk_with_preservation(
    text: str,
    preserve_sentences: bool = True
) -> List[Chunk]

# By paragraphs
chunks = chunker.chunk_by_paragraphs(
    text: str,
    max_paragraphs_per_chunk: int = 10
) -> List[Chunk]

# By semantic units
chunks = chunker.chunk_by_semantic_units(
    text: str,
    unit_markers: Optional[List[str]] = None
) -> List[Chunk]

# Get statistics
stats = chunker.get_chunk_statistics(chunks: List[Chunk]) -> Dict
```

### ParallelChunkProcessor

```python
processor = ParallelChunkProcessor(max_workers: int = 4)

# Process parallel
results = await processor.process_chunks_parallel(
    chunks: List[Chunk],
    processor_func: Callable,
    timeout: Optional[float] = None
) -> List[Any]

# With progress
results = await processor.process_with_progress(
    chunks: List[Chunk],
    processor_func: Callable,
    progress_callback: Optional[Callable] = None,
    log_interval: float = 10.0
) -> List[Any]

# Batches
results = await processor.process_batches(
    chunks: List[Chunk],
    processor_func: Callable,
    batch_size: int = 10
) -> List[Any]

# With retry
results = await processor.process_with_retry(
    chunks: List[Chunk],
    processor_func: Callable,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> List[Any]
```

## Troubleshooting

### Neo4j Connection Issues

```python
# Check Neo4j backend
if not engine.neo4j_backend:
    print("Neo4j backend not initialized")
    print("Check NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables")
```

### Memory Issues with Large Documents

```python
# Reduce chunk size or batch processing
graph = await engine.extract_from_document(
    document_path="huge.txt",
    chunk_size=2000  # Smaller chunks
)
```

### Slow Processing

```python
# Increase parallel workers
# In kggen_pipeline.yaml:
pipeline:
  parallel_workers: 8  # Increase from 4

# Or in code:
processor = ParallelChunkProcessor(max_workers=8)
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional entity extraction methods
- More sophisticated relation extraction
- Enhanced deduplication algorithms
- Additional export formats
- Performance optimizations

## License

This integration is part of OpenEvolve Knowledge Engine.

## References

- [kg-gen](https://github.com/yourusername/kg-gen) - Knowledge graph generation
- [Neo4j](https://neo4j.com/) - Graph database
- [DSPy](https://github.com/stanfordnlp/dspy) - Declarative self-improving language programs
