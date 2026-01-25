# IntegratedKnowledgeEngine - Comprehensive Usage Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Core Concepts](#core-concepts)
6. [API Reference](#api-reference)
7. [Advanced Usage](#advanced-usage)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

---

## Introduction

The `IntegratedKnowledgeEngine` is a comprehensive knowledge management system that combines multiple knowledge extraction sprints, storage backends, and search capabilities into a unified interface.

### Key Features

- **Multi-Sprint Processing**: Automatically selects the best extraction method (Graphiti, OneKE, KG-Gen)
- **Temporal Knowledge Tracking**: Track knowledge evolution over time
- **Bilingual Extraction**: Support for multilingual documents
- **Batch Processing**: Process multiple documents concurrently with progress tracking
- **Code Analysis**: Extract knowledge from code repositories
- **Intelligent Search**: Hybrid keyword and semantic search
- **Graceful Degradation**: Continues working even if some components fail
- **Production Ready**: Follows CLAUDE.md principles (UTC, idempotency, structured logging)

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│          IntegratedKnowledgeEngine                       │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Sprint Selection                      │  │
│  │  (Auto-select based on content type)               │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                             │
│                          ▼                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐      │
│  │Graphiti  │  │ OneKE   │  │     KG-Gen       │      │
│  │(Temporal)│  │(Bilingual)│  │   (Generic)      │      │
│  └─────┬────┘  └────┬─────┘  └────────┬─────────┘      │
│        │            │                  │               │
│        └────────────┴──────────────────┘               │
│                          │                             │
│                          ▼                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Knowledge Storage Layer                    │  │
│  │  (Qdrant, MongoDB, Neo4j, Redis)                  │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                             │
│                          ▼                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Knowledge Retrieval Layer                  │  │
│  │  (Hybrid Search, Recommendations, Trends)          │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

- Python 3.10+
- AsyncIO support
- (Optional) Neo4j for temporal knowledge
- (Optional) Qdrant for vector search
- (Optional) MongoDB for document storage
- (Optional) Redis for caching
- (Optional) Elasticsearch for full-text search

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/your-org/openevolve.git
cd openevolve

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GRAPHITI_PASSWORD="your_password"
export OPENAI_API_KEY="your_key"  # For KG-Gen
```

---

## Quick Start

### Minimal Example

```python
import asyncio
from knowledge_engine import IntegratedKnowledgeEngine

async def main():
    # Create engine with default configuration
    engine = IntegratedKnowledgeEngine()
    await engine.initialize()

    # Process a document
    result = await engine.process_document("document.pdf")
    print(f"Processed: {result['success']}")
    print(f"Entities found: {len(result.get('entities', []))}")

    # Search knowledge
    search_results = await engine.search_knowledge("machine learning")
    print(f"Found {search_results['count']} results")

    # Cleanup
    await engine.close()

asyncio.run(main())
```

### Using Context Manager (Recommended)

```python
async def main():
    config = {
        "graphiti_uri": "bolt://localhost:7687",
        "graphiti_user": "neo4j",
        "graphiti_password": "your_password"
    }

    async with IntegratedKnowledgeEngine(config) as engine:
        # Engine is automatically initialized
        result = await engine.process_document("doc.pdf")
        # Engine is automatically closed
    pass

asyncio.run(main())
```

### Convenience Function

```python
from knowledge_engine import create_integrated_knowledge_engine

async def main():
    # One-line creation and initialization
    engine = await create_integrated_knowledge_engine()

    result = await engine.process_document("doc.pdf")

    await engine.close()

asyncio.run(main())
```

---

## Configuration

### Environment Variables

All configuration should be done via environment variables (following CLAUDE.md principles):

```bash
# Graphiti (Temporal Knowledge Graph)
export GRAPHITI_URI="bolt://localhost:7687"
export GRAPHITI_USER="neo4j"
export GRAPHITI_PASSWORD="your_password"

# KG-Gen (Knowledge Generation)
export KGGEN_MODEL="gpt-4o"
export KGGEN_TIMEOUT_MS="30000"
export OPENAI_API_KEY="your_openai_key"

# OneKE (Bilingual Extraction)
export ONEKE_MODEL="oneke/OneKE-13B"
export ONEKE_DEVICE="cuda"
export ONEKE_TIMEOUT_MS="60000"

# Storage
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export MONGO_URI="mongodb://localhost:27017"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export REDIS_HOST="localhost"
export REDIS_PORT="6379"

# Elasticsearch
export ELASTICSEARCH_HOSTS="http://localhost:9200"
export ELASTICSEARCH_API_KEY="your_api_key"
export ELASTICSEARCH_INDEX_PREFIX="openevolve"

# Code Indexer
export INDEXER_CONFIG_PATH="knowledge_engine/indexer_config.yaml"

# Processing
export DEFAULT_TIMEOUT_MS="30000"
export MAX_RETRIES="3"
export CACHE_TTL="300"

# LLM
export ANTHROPIC_API_KEY="your_anthropic_key"
export LLM_TEMPERATURE="0.1"
export LLM_MAX_TOKENS="2000"
```

### Configuration Dictionary

Alternatively, pass configuration directly:

```python
config = {
    "graphiti_uri": "bolt://localhost:7687",
    "graphiti_user": "neo4j",
    "graphiti_password": "your_password",
    "elasticsearch_hosts": ["http://localhost:9200"],
    "default_timeout_ms": 30000,
    "max_retries": 3,
}

engine = IntegratedKnowledgeEngine(config)
```

---

## Core Concepts

### Sprint Types

The engine automatically selects the appropriate extraction sprint:

- **TEMPORAL_GRAPHITI**: For temporal knowledge and evolution tracking
- **BILINGUAL_ONEKE**: For multilingual documents
- **GENERIC_KGGEN**: For general document processing
- **HYBRID_AUTO**: Automatically tries available methods

### Processing Options

Customize how documents are processed:

```python
from knowledge_engine.integrated_engine import ProcessingOptions

options = ProcessingOptions(
    extract_temporal=True,      # Use temporal extraction
    extract_bilingual=False,    # Use bilingual extraction
    use_embeddings=True,        # Generate embeddings
    validate_results=True,      # Validate extraction results
    cache_results=True,         # Cache results
    timeout_ms=30000,          # Processing timeout
    max_retries=3,             # Maximum retry attempts
    correlation_id="custom_id"  # Custom correlation ID
)
```

### Batch Processing

Process multiple documents with progress tracking:

```python
async def batch_process():
    files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]

    # Progress callback
    def on_progress(message, percentage, metadata):
        print(f"{message}: {percentage:.1f}%")

    result = await engine.batch_process_documents(
        files,
        progress_callback=on_progress,
        max_concurrent=5
    )

    print(f"Processed {result.successful} of {result.total_items}")
    print(f"Failed: {result.failed}")
    print(f"Time: {result.total_time_ms}ms")
```

---

## API Reference

### Initialization

#### `IntegratedKnowledgeEngine(config=None)`

Create a new engine instance.

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Configuration dictionary

**Raises:**
- `RuntimeError`: If required configuration is missing

#### `async initialize()`

Initialize all components asynchronously.

**Raises:**
- `Exception`: If component initialization fails

### Document Processing

#### `async process_document(document_path, options=None)`

Process a single document.

**Parameters:**
- `document_path` (str): Path to document
- `options` (Optional[ProcessingOptions]): Processing options

**Returns:**
- Dict[str, Any]: Processing result with keys:
  - `success` (bool): Whether processing succeeded
  - `correlation_id` (str): Tracking ID
  - `document_path` (str): Path to document
  - `sprint_used` (str): Which sprint was used
  - `entities` (List[Dict]): Extracted entities
  - `relations` (List[Dict]): Extracted relations
  - `artifacts_stored` (int): Number of artifacts stored
  - `processing_time_ms` (float): Processing time
  - `error` (str, optional): Error message if failed

**Example:**
```python
result = await engine.process_document(
    "document.pdf",
    options=ProcessingOptions(extract_temporal=True)
)

if result['success']:
    print(f"Found {len(result['entities'])} entities")
else:
    print(f"Error: {result['error']}")
```

#### `async batch_process_documents(document_paths, options=None, progress_callback=None, max_concurrent=5)`

Process multiple documents in batch.

**Parameters:**
- `document_paths` (List[str]): List of document paths
- `options` (Optional[ProcessingOptions]): Processing options
- `progress_callback` (Optional[Callable]): Progress callback function
- `max_concurrent` (int): Maximum concurrent processing

**Returns:**
- BatchResult: Batch processing result with:
  - `total_items` (int): Total items processed
  - `successful` (int): Successful items
  - `failed` (int): Failed items
  - `success_rate` (float): Success rate (0-1)
  - `results` (List[Dict]): Individual results
  - `errors` (List[Dict]): Error details
  - `total_time_ms` (float): Total processing time

### Knowledge Search

#### `async search_knowledge(query, query_type="hybrid", filters=None, limit=10, correlation_id=None)`

Search the knowledge base.

**Parameters:**
- `query` (str): Search query
- `query_type` (str): Type of search ("hybrid", "keyword", "semantic")
- `filters` (Optional[Dict[str, Any]]): Filters to apply
- `limit` (int): Maximum results
- `correlation_id` (Optional[str]): Tracking ID

**Returns:**
- Dict[str, Any]: Search result with:
  - `success` (bool): Whether search succeeded
  - `query` (str): Search query
  - `results` (List[Dict]): Search results
  - `count` (int): Number of results
  - `execution_time_ms` (float): Execution time

**Example:**
```python
result = await engine.search_knowledge(
    "machine learning",
    query_type="hybrid",
    limit=5
)

for item in result['results']:
    print(f"Content: {item.get('content', '')[:100]}...")
```

### Code Analysis

#### `async analyze_code(repo_path, options=None, correlation_id=None)`

Analyze a code repository.

**Parameters:**
- `repo_path` (str): Path to repository
- `options` (Optional[ProcessingOptions]): Processing options
- `correlation_id` (Optional[str]): Tracking ID

**Returns:**
- Dict[str, Any]: Analysis result with:
  - `success` (bool): Whether analysis succeeded
  - `indexed_files` (int): Number of files indexed
  - `patterns_found` (int): Number of patterns found
  - `artifacts_extracted` (int): Number of artifacts extracted

### Temporal Queries

#### `async query_temporal(query, timestamp=None, correlation_id=None)`

Query knowledge at a specific point in time.

**Parameters:**
- `query` (str): Search query
- `timestamp` (Optional[datetime]): Point in time (defaults to now)
- `correlation_id` (Optional[str]): Tracking ID

**Returns:**
- Dict[str, Any]: Query result with:
  - `results` (List[Dict]): Temporal results
  - `reference_time` (str): Reference time ISO format

**Example:**
```python
from datetime import datetime, timezone

# Query knowledge as it was on January 1, 2024
timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
result = await engine.query_temporal(
    "machine learning",
    timestamp=timestamp
)
```

### Contradiction Detection

#### `async detect_contradictions(entity_name, correlation_id=None)`

Detect contradictions for an entity.

**Parameters:**
- `entity_name` (str): Entity to check
- `correlation_id` (Optional[str]): Tracking ID

**Returns:**
- Dict[str, Any]: Contradictions found

### Statistics and Health

#### `async get_statistics()`

Get engine statistics.

**Returns:**
- Dict[str, Any]: Statistics including:
  - `components`: Component availability
  - `knowledge`: Entity and relationship counts
  - `storage`: Storage statistics

#### `async health_check()`

Check health of all components.

**Returns:**
- Dict[str, Any]: Health status with:
  - `overall` (str): "healthy", "degraded", or "unhealthy"
  - `components`: Individual component health

### Cleanup

#### `async close()`

Close all components and cleanup resources.

**Note:** Use async context manager for automatic cleanup.

---

## Advanced Usage

### Custom Sprint Selection

Force a specific sprint:

```python
from knowledge_engine.integrated_engine import SprintType

# Process with specific sprint
options = ProcessingOptions(
    extract_temporal=True,  # Will prefer temporal sprint
    extract_bilingual=False
)

result = await engine.process_document("doc.pdf", options)
print(f"Sprint used: {result['sprint_used']}")
```

### Fallback Chains

The engine automatically falls back through sprints:

```
TEMPORAL_GRAPHITI → GENERIC_KGGEN → HYBRID_AUTO
BILINGUAL_ONEKE → GENERIC_KGGEN → TEMPORAL_GRAPHITI
GENERIC_KGGEN → TEMPORAL_GRAPHITI
HYBRID_AUTO → TEMPORAL_GRAPHITI → GENERIC_KGGEN → BILINGUAL_ONEKE
```

### Progress Tracking

Detailed progress tracking with callbacks:

```python
def progress_handler(message, percentage, metadata):
    print(f"[{percentage:.1f}%] {message}")
    if 'index' in metadata:
        print(f"  Processing item {metadata['index'] + 1} of {metadata['total']}")

result = await engine.batch_process_documents(
    files,
    progress_callback=progress_handler
)
```

### Error Handling

Graceful error handling:

```python
result = await engine.process_document("document.pdf")

if not result['success']:
    print(f"Processing failed: {result['error']}")
    print(f"Correlation ID: {result['correlation_id']}")

    # Retry with different options
    options = ProcessingOptions(timeout_ms=60000, max_retries=5)
    result = await engine.process_document("document.pdf", options)
```

### Workflow Integration

Integrate into your workflow:

```python
async def analyze_project_documents():
    # Initialize
    async with IntegratedKnowledgeEngine() as engine:
        # Health check
        health = await engine.health_check()
        if health['overall'] == 'unhealthy':
            print("Warning: Some components unhealthy")

        # Process all documents
        docs = list(Path("./docs").glob("*.pdf"))
        result = await engine.batch_process_documents(
            [str(d) for d in docs]
        )

        # Extract insights
        for doc_result in result.results:
            if doc_result['success']:
                print(f"{doc_result['document_path']}: {doc_result['artifacts_stored']} artifacts")

        # Search for patterns
        search = await engine.search_knowledge("architecture", limit=10)

        return {
            'processed': result.successful,
            'insights': search['results']
        }
```

---

## Best Practices

### 1. Use Context Managers

Always use async context managers for automatic cleanup:

```python
# Good
async with engine as e:
    result = await e.process_document("doc.pdf")

# Avoid
engine = IntegratedKnowledgeEngine()
await engine.initialize()
result = await engine.process_document("doc.pdf")
await engine.close()  # Easy to forget!
```

### 2. Check Results

Always check operation results:

```python
result = await engine.process_document("doc.pdf")

if not result['success']:
    logger.error(f"Failed: {result['error']}")
    return

# Continue with success case
```

### 3. Use Correlation IDs

Track operations with correlation IDs:

```python
import uuid

correlation_id = f"batch_{uuid.uuid4().hex}"
result = await engine.process_document(
    "doc.pdf",
    options=ProcessingOptions(correlation_id=correlation_id)
)

logger.info(f"Processing {correlation_id}: {result['success']}")
```

### 4. Handle Timeouts

Set appropriate timeouts:

```python
options = ProcessingOptions(
    timeout_ms=60000,  # 60 seconds
    max_retries=3
)
```

### 5. Batch Processing

Process multiple documents concurrently:

```python
# Process up to 10 documents concurrently
result = await engine.batch_process_documents(
    file_list,
    max_concurrent=10
)
```

### 6. Monitor Progress

Use progress callbacks for long operations:

```python
def progress_handler(msg, pct, meta):
    # Log to monitoring system
    monitoring.log_progress(pct, msg)
    # Update UI
    ui.update_progress(pct)

result = await engine.batch_process_documents(
    large_file_list,
    progress_callback=progress_handler
)
```

### 7. Health Checks

Check health before critical operations:

```python
async def critical_operation():
    health = await engine.health_check()
    if health['overall'] != 'healthy':
        raise RuntimeError("Engine not healthy")

    # Proceed with operation
    result = await engine.process_document("critical.pdf")
```

---

## Troubleshooting

### Initialization Fails

**Problem:** Engine fails to initialize

**Solutions:**
1. Check environment variables are set
2. Verify required services are running (Neo4j, MongoDB, etc.)
3. Check network connectivity
4. Review logs for specific errors

```bash
# Check Neo4j
curl http://localhost:7474

# Check MongoDB
mongosh --eval "db.adminCommand('ping')"

# Check Elasticsearch
curl http://localhost:9200/_cluster/health
```

### Document Processing Fails

**Problem:** Documents fail to process

**Solutions:**
1. Verify file format is supported (PDF, TXT, MD)
2. Check file permissions
3. Increase timeout
4. Check available memory

```python
options = ProcessingOptions(
    timeout_ms=60000,  # Increase timeout
    max_retries=5      # More retries
)
result = await engine.process_document("large_file.pdf", options)
```

### Search Returns No Results

**Problem:** Search returns empty results

**Solutions:**
1. Verify documents have been processed
2. Check Elasticsearch index
3. Try different query_type
4. Verify filters are correct

```python
# Try different search types
result = await engine.search_knowledge("query", query_type="keyword")
result = await engine.search_knowledge("query", query_type="hybrid")
result = await engine.search_knowledge("query", query_type="semantic")
```

### Sprint Selection Issues

**Problem:** Wrong sprint selected

**Solutions:**
1. Check document content type
2. Manually specify sprint
3. Review sprint selection logic

```python
# Force specific sprint
options = ProcessingOptions(
    extract_temporal=True,  # Force temporal
    extract_bilingual=False
)
```

### Performance Issues

**Problem:** Slow processing

**Solutions:**
1. Reduce concurrent processing
2. Increase timeout
3. Check system resources
4. Enable caching

```python
options = ProcessingOptions(
    cache_results=True,  # Enable caching
    timeout_ms=60000
)

result = await engine.batch_process_documents(
    files,
    options=options,
    max_concurrent=3  # Reduce concurrency
)
```

---

## Examples

### Example 1: Process Research Papers

```python
import asyncio
from pathlib import Path
from knowledge_engine import create_integrated_knowledge_engine

async def process_papers():
    engine = await create_integrated_knowledge_engine()

    # Get all PDFs
    papers = list(Path("./papers").glob("*.pdf"))

    # Process with progress tracking
    def progress(msg, pct, meta):
        print(f"[{pct:.1f}%] {msg}")

    result = await engine.batch_process_documents(
        [str(p) for p in papers],
        progress_callback=progress,
        max_concurrent=3
    )

    print(f"Successfully processed {result.successful}/{result.total_items} papers")

    # Search for key concepts
    for concept in ["neural networks", "deep learning", "transformers"]:
        search_result = await engine.search_knowledge(concept, limit=5)
        print(f"\n{concept}: {search_result['count']} results")

    await engine.close()

asyncio.run(process_papers())
```

### Example 2: Code Repository Analysis

```python
async def analyze_codebase():
    async with IntegratedKnowledgeEngine() as engine:
        # Analyze repository
        result = await engine.analyze_code("./my_project")

        if result['success']:
            print(f"Indexed {result['indexed_files']} files")
            print(f"Found {result['patterns_found']} patterns")
            print(f"Extracted {result['artifacts_extracted']} artifacts")

        # Search for specific patterns
        search = await engine.search_knowledge(
            "authentication",
            filters={"type": "code_pattern"},
            limit=10
        )

        print("\nAuthentication patterns found:")
        for item in search['results']:
            print(f"  - {item.get('file', 'unknown')}: {item.get('pattern', '')}")
```

### Example 3: Temporal Knowledge Tracking

```python
from datetime import datetime, timezone, timedelta

async def track_knowledge_evolution():
    engine = await create_integrated_knowledge_engine()

    # Process document at current time
    await engine.process_document("v1_document.pdf")

    # Simulate time passing and processing new version
    await asyncio.sleep(1)
    await engine.process_document("v2_document.pdf")

    # Query knowledge at different points in time
    now = datetime.now(timezone.utc)
    past = now - timedelta(minutes=1)

    current_knowledge = await engine.query_temporal("concept", timestamp=now)
    past_knowledge = await engine.query_temporal("concept", timestamp=past)

    print(f"Current: {current_knowledge['count']} results")
    print(f"Past: {past_knowledge['count']} results")

    # Check for contradictions
    contradictions = await engine.detect_contradictions("concept")
    print(f"Contradictions: {contradictions['count']}")

    await engine.close()
```

### Example 4: Multilingual Document Processing

```python
async def process_multilingual():
    config = {
        "oneke_model": "oneke/OneKE-13B",
        "oneke_device": "cuda"
    }

    async with IntegratedKnowledgeEngine(config) as engine:
        options = ProcessingOptions(
            extract_bilingual=True,  # Enable bilingual extraction
            extract_temporal=False
        )

        # Process Chinese document
        result = await engine.process_document(
            "chinese_document.pdf",
            options=options
        )

        if result['success']:
            print(f"Processed with sprint: {result['sprint_used']}")
            print(f"Entities: {len(result['entities'])}")

            # Search across languages
            search = await engine.search_knowledge("机器学习")  # "machine learning" in Chinese
            print(f"Cross-lingual results: {search['count']}")
```

### Example 5: Complete Workflow

```python
async def complete_workflow():
    """Complete knowledge extraction and analysis workflow."""

    async with IntegratedKnowledgeEngine() as engine:
        # 1. Health check
        health = await engine.health_check()
        print(f"Engine health: {health['overall']}")

        # 2. Get initial statistics
        stats = await engine.get_statistics()
        print(f"Initial entities: {stats['knowledge']['entities']}")

        # 3. Process documents
        docs = ["doc1.pdf", "doc2.pdf", "doc3.txt"]

        result = await engine.batch_process_documents(
            docs,
            max_concurrent=3
        )

        print(f"Processed: {result.successful}/{result.total_items}")

        # 4. Analyze code
        code_result = await engine.analyze_code("./src")
        if code_result['success']:
            print(f"Code patterns: {code_result['patterns_found']}")

        # 5. Search knowledge
        searches = ["architecture", "performance", "security"]
        for query in searches:
            search_result = await engine.search_knowledge(query, limit=5)
            print(f"{query}: {search_result['count']} results")

        # 6. Check for contradictions
        contradictions = await engine.detect_contradictions("architecture")
        if contradictions['count'] > 0:
            print(f"Warning: {contradictions['count']} contradictions found")

        # 7. Get final statistics
        final_stats = await engine.get_statistics()
        print(f"Final entities: {final_stats['knowledge']['entities']}")
        print(f"New entities: {final_stats['knowledge']['entities'] - stats['knowledge']['entities']}")

asyncio.run(complete_workflow())
```

---

## Additional Resources

- **API Documentation**: See docstrings in source code
- **Architecture**: See `ARCHITECTURE.md`
- **CLAUDE.md**: Engineering principles and best practices
- **Examples**: See `examples/` directory
- **Tests**: See `knowledge_engine/tests/`

---

## Support

For issues, questions, or contributions:

1. Check existing documentation
2. Review test cases for examples
3. Check GitHub issues
4. Create new issue with:
   - Clear description
   - Code example
   - Error messages
   - Environment details

---

**Version**: 2.0.0
**Last Updated**: 2025-01-08
**Author**: OpenEvolve Distinguished Engineer
