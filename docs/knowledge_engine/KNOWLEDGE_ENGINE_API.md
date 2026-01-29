# KnowledgeEngine API Documentation

## Overview

The `KnowledgeEngine` class provides a unified, production-grade orchestration layer for the OpenEvolve Knowledge System. It integrates all knowledge engine capabilities into a single, easy-to-use interface.

**Version:** 1.0.0
**Author:** OpenEvolve Distinguished Engineer
**Following:** CLAUDE.md principles (Zero Trust, Runtime Truth, Idempotency)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)
6. [Error Handling](#error-handling)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install openai anthropic graphiti-core elasticsearch

# Set environment variables
export GRAPHITI_URI="bolt://localhost:7687"
export GRAPHITI_USER="neo4j"
export GRAPHITI_PASSWORD="your_password"
export OPENAI_API_KEY="your_openai_key"
```

### Basic Usage

```python
from knowledge_engine import create_knowledge_engine

# Create and initialize engine
async with await create_knowledge_engine() as engine:
    # Process a document
    result = await engine.process_document("research_paper.pdf")

    # Query temporal knowledge
    results = await engine.query_temporal("machine learning algorithms")

    # Detect contradictions
    contradictions = await engine.detect_contradictions("AI")

    # Generate visualization
    viz = await engine.visualize_graph("explorer", data={"triples": result.triples})
```

---

## Architecture

### Component Structure

```
KnowledgeEngine (Orchestration Layer)
├── Graphiti (Temporal Knowledge Graph)
│   ├── Temporal Bridge
│   ├── Agent Memory
│   └── Contradiction Detector
├── KG-Gen (Knowledge Extraction)
│   └── Extraction Pipeline
├── OneKE (Bilingual Extraction)
│   └── Model Adapter
├── Visualization
│   ├── Graph Explorer
│   ├── Temporal Visualizer
│   └── Community Visualizer
├── Elasticsearch (Full-text Search)
│   └── Search Engine
└── Code Indexer
    └── Code Indexer
```

### Design Principles

Following **CLAUDE.md** principles:

1. **CONFIGURATION EXPLICITNESS**: All config via environment variables
2. **UTC TIME**: All timestamps in UTC
3. **STRUCTURED LOGGING**: JSON logs with correlation IDs
4. **RUNTIME TRUTH**: Verify components before use
5. **IDEMPOTENCY**: All operations safe to run multiple times
6. **FAIL FAST**: Crash immediately if misconfigured

---

## Configuration

### Environment Variables

#### Required

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `GRAPHITI_PASSWORD` | Neo4j password | `secret123` | *None* |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` | *None* |

#### Optional

| Variable | Description | Example | Default |
|----------|-------------|---------|---------|
| `GRAPHITI_URI` | Neo4j connection URI | `bolt://localhost:7687` | `bolt://localhost:7687` |
| `GRAPHITI_USER` | Neo4j username | `neo4j` | `neo4j` |
| `KGGEN_ENTITY_MODEL` | Extraction model | `gpt-4o` | `gpt-4o` |
| `KGGEN_CHUNK_SIZE` | Text chunk size | `5000` | `5000` |
| `KGGEN_TIMEOUT_MS` | Extraction timeout | `30000` | `30000` |
| `ONEKE_MODEL_NAME` | OneKE model | `oneke/OneKE-13B` | `oneke/OneKE-13B` |
| `ONEKE_DEVICE` | Inference device | `cuda` | `cuda` |
| `ONEKE_TIMEOUT_MS` | OneKE timeout | `60000` | `60000` |
| `VIS_CACHE_TTL` | Viz cache TTL | `3600` | `3600` |
| `VIS_MAX_NODES` | Max graph nodes | `10000` | `10000` |
| `VIS_EXPORT_DIR` | Viz export directory | `./viz` | `./visualizations` |
| `ELASTICSEARCH_HOSTS` | ES hosts | `http://localhost:9200` | `http://localhost:9200` |
| `ELASTICSEARCH_API_KEY` | ES API key | `key` | `""` |
| `ELASTICSEARCH_INDEX_PREFIX` | ES index prefix | `openevolve` | `openevolve` |
| `LLM_TEMPERATURE` | LLM temperature | `0.1` | `0.1` |
| `LLM_MAX_TOKENS` | LLM max tokens | `2000` | `2000` |

### Configuration Validation

The engine validates configuration at startup and fails fast if required variables are missing:

```python
# This will raise RuntimeError if GRAPHITI_PASSWORD not set
engine = KnowledgeEngine()
await engine.initialize()
```

---

## API Reference

### Classes

#### `KnowledgeEngine`

Main orchestration class for the Knowledge Engine.

```python
class KnowledgeEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    async def initialize(self)
    async def close(self)
```

**Constructor Arguments:**

- `config` (Optional[Dict[str, Any]]): Configuration dictionary. If None, loads from environment variables.

**Example:**

```python
# Use environment variables
engine = KnowledgeEngine()

# Or provide explicit config
engine = KnowledgeEngine(config={
    "graphiti_uri": "bolt://localhost:7687",
    "graphiti_user": "neo4j",
    "graphiti_password": "secret",
    "openai_api_key": "sk-..."
})
```

---

#### `ProcessingResult`

Result from document/knowledge processing operations.

```python
@dataclass
class ProcessingResult:
    success: bool
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    triples: List[Tuple[str, str, str]]
    visualization: Optional[str]
    error: Optional[str]
    correlation_id: Optional[str]
    processing_time_ms: float
    metadata: Dict[str, Any]
```

**Methods:**

- `to_dict() -> Dict[str, Any]`: Convert to dictionary

---

#### `QueryResult`

Result from knowledge queries.

```python
@dataclass
class QueryResult:
    query: str
    results: List[Dict[str, Any]]
    count: int
    execution_time_ms: float
    correlation_id: str
    timestamp: str
    metadata: Dict[str, Any]
```

**Methods:**

- `to_dict() -> Dict[str, Any]`: Convert to dictionary

---

### Methods

#### `initialize()`

Initialize all components asynchronously.

```python
async def initialize(self)
```

**Raises:**
- `Exception`: If component initialization fails

**Example:**

```python
engine = KnowledgeEngine()
await engine.initialize()
```

---

#### `process_document()`

Process a document through the complete pipeline.

```python
async def process_document(
    self,
    document_path: str,
    extract_temporal: bool = True,
    extract_bilingual: bool = False,
    correlation_id: Optional[str] = None
) -> ProcessingResult
```

**Arguments:**

- `document_path` (str): Path to document
- `extract_temporal` (bool): Extract temporal knowledge (default: True)
- `extract_bilingual` (bool): Use bilingual extraction (default: False)
- `correlation_id` (Optional[str]): Correlation ID for tracking

**Returns:**

- `ProcessingResult`: Processing result with entities, relations, visualization

**Raises:**

- `FileNotFoundError`: If document not found
- `RuntimeError`: If no extraction engine available

**Example:**

```python
result = await engine.process_document(
    document_path="research_paper.pdf",
    extract_temporal=True,
    extract_bilingual=False,
    correlation_id="doc_001"
)

print(f"Extracted {len(result.entities)} entities")
print(f"Extracted {len(result.relations)} relations")
print(f"Processing time: {result.processing_time_ms}ms")
```

---

#### `query_temporal()`

Query knowledge at specific point in time.

```python
async def query_temporal(
    self,
    query: str,
    timestamp: Optional[datetime] = None,
    correlation_id: Optional[str] = None
) -> QueryResult
```

**Arguments:**

- `query` (str): Search query
- `timestamp` (Optional[datetime]): Point in time for query (defaults to now)
- `correlation_id` (Optional[str]): Correlation ID for tracking

**Returns:**

- `QueryResult`: Query results with metadata

**Raises:**

- `RuntimeError`: If Graphiti not available

**Example:**

```python
from datetime import datetime, timezone

# Query current knowledge
results = await engine.query_temporal(
    query="machine learning algorithms"
)

# Query knowledge at specific point in time
past_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
results = await engine.query_temporal(
    query="AI models",
    timestamp=past_time
)

print(f"Found {results.count} results")
print(f"Query time: {results.execution_time_ms}ms")
```

---

#### `detect_contradictions()`

Detect contradictions for an entity across time.

```python
async def detect_contradictions(
    self,
    entity_name: str,
    correlation_id: Optional[str] = None
) -> List[Dict[str, Any]]
```

**Arguments:**

- `entity_name` (str): Name of entity to check
- `correlation_id` (Optional[str]): Correlation ID for tracking

**Returns:**

- `List[Dict[str, Any]]`: List of detected contradictions

**Raises:**

- `RuntimeError`: If Graphiti not available

**Example:**

```python
contradictions = await engine.detect_contradictions(
    entity_name="Artificial Intelligence"
)

for contradiction in contradictions:
    print(f"Contradiction: {contradiction['description']}")
    print(f"Severity: {contradiction['severity']}")
```

---

#### `visualize_graph()`

Generate knowledge graph visualization.

```python
async def visualize_graph(
    self,
    graph_type: str = "explorer",
    data: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> str
```

**Arguments:**

- `graph_type` (str): Type of visualization ("explorer", "temporal", "community")
- `data` (Optional[Dict[str, Any]]): Data to visualize
- `options` (Optional[Dict[str, Any]]): Visualization options
- `correlation_id` (Optional[str]): Correlation ID for tracking

**Returns:**

- `str`: Visualization data (JSON or file path)

**Raises:**

- `RuntimeError`: If visualization not available
- `ValueError`: If unknown graph type

**Example:**

```python
# Generate explorer visualization
viz = await engine.visualize_graph(
    graph_type="explorer",
    data={"triples": result.triples}
)

# Generate temporal visualization
viz = await engine.visualize_graph(
    graph_type="temporal",
    data={"temporal_data": temporal_data}
)

# Generate community visualization
viz = await engine.visualize_graph(
    graph_type="community",
    data={"triples": result.triples}
)
```

---

#### `search_knowledge()`

Search the knowledge base.

```python
async def search_knowledge(
    self,
    query: str,
    query_type: str = "hybrid",
    limit: int = 10,
    correlation_id: Optional[str] = None
) -> QueryResult
```

**Arguments:**

- `query` (str): Search query
- `query_type` (str): Type of search ("keyword", "semantic", "hybrid")
- `limit` (int): Maximum results (default: 10)
- `correlation_id` (Optional[str]): Correlation ID for tracking

**Returns:**

- `QueryResult`: Search results with metadata

**Raises:**

- `RuntimeError`: If Elasticsearch not available

**Example:**

```python
# Hybrid search
results = await engine.search_knowledge(
    query="neural network architectures",
    query_type="hybrid",
    limit=20
)

# Keyword search
results = await engine.search_knowledge(
    query="deep learning",
    query_type="keyword"
)

print(f"Found {results.count} results in {results.execution_time_ms}ms")
```

---

#### `get_statistics()`

Get statistics about the knowledge engine.

```python
async def get_statistics(self) -> Dict[str, Any]
```

**Returns:**

- `Dict[str, Any]`: Statistics dictionary

**Example:**

```python
stats = await engine.get_statistics()

print(f"Components: {stats['components']}")
print(f"Entities: {stats['knowledge']['entities']}")
print(f"Relationships: {stats['knowledge']['relationships']}")
print(f"Graphiti stats: {stats.get('graphiti', {})}")
```

---

#### `health_check()`

Check health of all components.

```python
async def health_check(self) -> Dict[str, Any]
```

**Returns:**

- `Dict[str, Any]`: Health status dictionary

**Example:**

```python
health = await engine.health_check()

print(f"Overall: {health['overall']}")
print(f"Components: {health['components']}")

# Check if healthy
if health['overall'] == 'healthy':
    print("All systems operational")
```

---

#### `close()`

Close all components and cleanup resources.

```python
async def close(self)
```

**Example:**

```python
engine = KnowledgeEngine()
await engine.initialize()

# ... use engine ...

await engine.close()
```

---

### Convenience Functions

#### `create_knowledge_engine()`

Create and initialize a KnowledgeEngine instance.

```python
async def create_knowledge_engine(
    config: Optional[Dict[str, Any]] = None
) -> KnowledgeEngine
```

**Arguments:**

- `config` (Optional[Dict[str, Any]]): Optional configuration dictionary

**Returns:**

- `KnowledgeEngine`: Initialized KnowledgeEngine ready to use

**Example:**

```python
# One-line initialization
engine = await create_knowledge_engine()

# With custom config
engine = await create_knowledge_engine(config={
    "graphiti_uri": "bolt://localhost:7687",
    "graphiti_password": "secret"
})

# Use engine
result = await engine.process_document("doc.pdf")

# Cleanup
await engine.close()
```

---

## Usage Examples

### Example 1: Basic Document Processing

```python
from knowledge_engine import create_knowledge_engine

async def main():
    # Initialize engine
    engine = await create_knowledge_engine()

    # Process document
    result = await engine.process_document("research_paper.pdf")

    if result.success:
        print(f"✓ Processed successfully")
        print(f"  Entities: {len(result.entities)}")
        print(f"  Relations: {len(result.relations)}")
        print(f"  Time: {result.processing_time_ms}ms")
    else:
        print(f"✗ Processing failed: {result.error}")

    # Cleanup
    await engine.close()

asyncio.run(main())
```

---

### Example 2: Temporal Knowledge Queries

```python
from datetime import datetime, timezone
from knowledge_engine import create_knowledge_engine

async def main():
    async with await create_knowledge_engine() as engine:
        # Process documents at different times
        await engine.process_document("doc_v1.pdf")
        # ... time passes ...
        await engine.process_document("doc_v2.pdf")

        # Query current knowledge
        current = await engine.query_temporal("AI models")

        # Query knowledge at specific point in time
        past = datetime(2024, 1, 1, tzinfo=timezone.utc)
        historical = await engine.query_temporal("AI models", timestamp=past)

        # Compare
        print(f"Current knowledge: {current.count} facts")
        print(f"Historical knowledge: {historical.count} facts")

asyncio.run(main())
```

---

### Example 3: Contradiction Detection

```python
from knowledge_engine import create_knowledge_engine

async def main():
    engine = await create_knowledge_engine()

    # Process documents with conflicting information
    await engine.process_document("source_a.pdf")
    await engine.process_document("source_b.pdf")

    # Detect contradictions
    contradictions = await engine.detect_contradictions("Climate Change")

    print(f"Found {len(contradictions)} contradictions:")
    for c in contradictions:
        print(f"  - {c['description']}")
        print(f"    Severity: {c['severity']}")

    await engine.close()

asyncio.run(main())
```

---

### Example 4: Batch Processing

```python
import asyncio
from pathlib import Path
from knowledge_engine import create_knowledge_engine

async def main():
    engine = await create_knowledge_engine()

    # Process multiple documents concurrently
    documents = Path("./docs").glob("*.pdf")

    tasks = [
        engine.process_document(str(doc))
        for doc in documents
    ]

    results = await asyncio.gather(*tasks)

    # Summary
    successful = sum(1 for r in results if r.success)
    total_time = sum(r.processing_time_ms for r in results)

    print(f"Processed {successful}/{len(results)} documents")
    print(f"Total time: {total_time}ms")

    await engine.close()

asyncio.run(main())
```

---

### Example 5: Visualization Pipeline

```python
from knowledge_engine import create_knowledge_engine

async def main():
    async with await create_knowledge_engine() as engine:
        # Process document
        result = await engine.process_document("research.pdf")

        # Generate different visualizations
        explorer_viz = await engine.visualize_graph(
            "explorer",
            data={"triples": result.triples}
        )

        temporal_viz = await engine.visualize_graph(
            "temporal",
            data={"temporal_data": result.entities}
        )

        community_viz = await engine.visualize_graph(
            "community",
            data={"triples": result.triples}
        )

        # Export visualizations
        from knowledge_engine.visualization import ExportHandler
        export_handler = ExportHandler()

        await export_handler.export_to_file(
            explorer_viz,
            output_path="./viz/explorer.json"
        )

asyncio.run(main())
```

---

### Example 6: Health Monitoring

```python
import asyncio
from knowledge_engine import create_knowledge_engine

async def monitor_health():
    engine = await create_knowledge_engine()

    while True:
        health = await engine.health_check()

        if health['overall'] == 'healthy':
            print("✓ All systems operational")
        else:
            print(f"⚠ System degraded: {health['components']}")

        # Check every 60 seconds
        await asyncio.sleep(60)

asyncio.run(monitor_health())
```

---

## Error Handling

### Principles

Following **CLAUDE.md**: Handle failure gracefully

1. **Return Result Objects**: Operations return `ProcessingResult` or `QueryResult` objects
2. **Never Crash on User Data**: Invalid data returns result with `error` field
3. **Fail Fast on Config**: Misconfigured components raise `RuntimeError` at startup
4. **Log Everything**: All errors logged with correlation IDs

### Error Handling Patterns

#### 1. Document Processing Errors

```python
result = await engine.process_document("document.pdf")

if result.success:
    # Process result
    print(f"Extracted {len(result.entities)} entities")
else:
    # Handle error
    print(f"Processing failed: {result.error}")

    # Check error type
    if "not found" in result.error.lower():
        # File not found
        pass
    elif "unsupported" in result.error.lower():
        # Unsupported file type
        pass
```

#### 2. Component Unavailable

```python
try:
    results = await engine.query_temporal("test query")
except RuntimeError as e:
    if "Graphiti.*not available" in str(e):
        print("Temporal knowledge not available")
        # Fallback to alternative
        results = await engine.search_knowledge("test query")
```

#### 3. Configuration Errors

```python
try:
    engine = KnowledgeEngine()
    await engine.initialize()
except RuntimeError as e:
    if "Missing required" in str(e):
        print(f"Configuration error: {e}")
        print("Please set required environment variables")
        sys.exit(1)
```

---

## Performance Considerations

### 1. Concurrent Processing

```python
import asyncio

# Process multiple documents concurrently
tasks = [
    engine.process_document(doc)
    for doc in document_list
]

results = await asyncio.gather(*tasks)
```

### 2. Caching

Visualizations are cached based on TTL:

```python
# Set cache TTL
os.environ["VIS_CACHE_TTL"] = "3600"  # 1 hour
```

### 3. Batch Size

For large document sets, control batch size:

```python
from itertools import islice

def chunks(iterable, size):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk

for doc_batch in chunks(document_list, 10):
    tasks = [engine.process_document(doc) for doc in doc_batch]
    await asyncio.gather(*tasks)
```

### 4. Timeout Configuration

Set appropriate timeouts:

```python
os.environ["KGGEN_TIMEOUT_MS"] = "30000"  # 30 seconds
os.environ["ONEKE_TIMEOUT_MS"] = "60000"  # 60 seconds
```

---

## Best Practices

### 1. Use Async Context Managers

```python
# GOOD: Automatic cleanup
async with await create_knowledge_engine() as engine:
    result = await engine.process_document("doc.pdf")

# BAD: Manual cleanup
engine = await create_knowledge_engine()
result = await engine.process_document("doc.pdf")
await engine.close()
```

### 2. Use Correlation IDs

```python
# GOOD: Track operations
result = await engine.process_document(
    "doc.pdf",
    correlation_id="user_123_doc_456"
)

# BAD: No tracking
result = await engine.process_document("doc.pdf")
```

### 3. Handle Errors Gracefully

```python
# GOOD: Check results
result = await engine.process_document("doc.pdf")
if not result.success:
    logger.error(f"Processing failed: {result.error}")
    return

# BAD: Assume success
result = await engine.process_document("doc.pdf")
entities = result.entities  # May be empty on error
```

### 4. Monitor Health

```python
# GOOD: Check health before operations
health = await engine.health_check()
if health['overall'] != 'healthy':
    logger.warning(f"System degraded: {health['components']}")

# BAD: Assume health
result = await engine.process_document("doc.pdf")
```

### 5. Use UTC Timestamps

```python
# GOOD: UTC timestamps
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)

# BAD: Local timestamps
timestamp = datetime.now()
```

### 6. Set Explicit Timeouts

```python
# GOOD: Configured timeouts
engine = KnowledgeEngine(config={
    "kggen_timeout_ms": 30000,
    "oneke_timeout_ms": 60000
})

# BAD: Default timeouts
engine = KnowledgeEngine()
```

---

## Testing

### Unit Tests

```python
import pytest
from knowledge_engine import create_knowledge_engine

@pytest.mark.asyncio
async def test_document_processing():
    engine = await create_knowledge_engine()
    result = await engine.process_document("test.pdf")
    assert result.success
    await engine.close()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    async with await create_knowledge_engine() as engine:
        # Process
        result = await engine.process_document("test.pdf")

        # Query
        results = await engine.query_temporal("test")

        # Visualize
        viz = await engine.visualize_graph("explorer", data={"triples": result.triples})

        assert result.success
        assert results.count > 0
        assert viz is not None
```

---

## Troubleshooting

### Issue: "Missing required environment variables"

**Solution:** Set required environment variables before starting:

```bash
export GRAPHITI_PASSWORD="your_password"
export OPENAI_API_KEY="your_key"
```

### Issue: "Graphiti temporal knowledge not available"

**Solution:** Ensure Neo4j is running and credentials are correct:

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Test connection
cypher-shell -u neo4j -p your_password
```

### Issue: "Elasticsearch search not available"

**Solution:** Ensure Elasticsearch is running:

```bash
# Check Elasticsearch is running
curl http://localhost:9200/_cluster/health
```

### Issue: "Visualization components not available"

**Solution:** Install visualization dependencies:

```bash
pip install networkx matplotlib plotly
```

---

## Changelog

### Version 1.0.0 (2025-01-08)

- Initial release
- Unified orchestration layer
- Integration with Graphiti, KG-Gen, OneKE, Visualization
- Full-text search via Elasticsearch
- Comprehensive test suite
- Production-ready error handling

---

## Support

For issues, questions, or contributions, see:
- GitHub: [OpenEvolve/Frontend](https://github.com/OpenEvolve/Frontend)
- Documentation: `knowledge_engine/KNOWLEDGE_ENGINE_API.md`
- Tests: `knowledge_engine/tests/test_knowledge_engine.py`

---

**End of API Documentation**
