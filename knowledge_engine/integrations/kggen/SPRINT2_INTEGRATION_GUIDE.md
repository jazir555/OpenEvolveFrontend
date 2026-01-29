# KG-Gen Sprint 2 Integration Guide

Task 2.6.3: Create KG-Gen integration guide

## Overview

This guide provides comprehensive documentation for integrating KG-Gen's advanced knowledge graph generation pipeline with OpenEvolve. The integration follows CLAUDE.md principles ensuring production-grade quality, idempotency, and explicit configuration.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

## Installation

### Requirements

- Python 3.9+
- OpenEvolve Frontend
- Required dependencies (see requirements.txt)

### Environment Variables

All configuration uses environment variables (LAW OF CONFIGURATION EXPLICITNESS):

```bash
# Entity Extraction
export KGGEN_ENTITY_MODEL="gpt-4o"
export KGGEN_ENTITY_TEMPERATURE="0.0"
export KGGEN_ENTITY_MAX_TOKENS="4000"
export KGGEN_ENTITY_TIMEOUT="300.0"

# Relation Extraction
export KGGEN_RELATION_MODEL="gpt-4o"
export KGGEN_RELATION_TEMPERATURE="0.0"
export KGGEN_RELATION_MAX_TOKENS="8000"
export KGGEN_RELATION_TIMEOUT="600.0"

# Processing
export KGGEN_CHUNK_SIZE="5000"
export KGGEN_CHUNK_OVERLAP="200"
export KGGEN_PARALLEL_WORKERS="4"

# Deduplication
export KGGEN_SEMHASH_THRESHOLD="0.95"
export KGGEN_LM_CLUSTER_SIZE="128"
export KGGEN_LM_SIMILARITY_THRESHOLD="0.85"

# Memory
export KGGEN_MEMORY_PERSISTENCE="true"
export KGGEN_MEMORY_STORAGE_PATH="./data/kggen_memories"

# Aggregation
export KGGEN_MERGE_STRATEGY="union"
export KGGEN_CONFLICT_RESOLUTION="keep_both"
```

## Quick Start

### Basic Extraction

```python
import asyncio
from knowledge_engine.integrations.kggen import ExtractionPipeline

async def main():
    # Initialize pipeline
    pipeline = ExtractionPipeline()

    # Extract knowledge graph
    text = """
    Apple is a technology company founded by Steve Jobs.
    Google was founded by Larry Page and Sergey Brin.
    """

    result = await pipeline.extract(text=text)

    print(f"Entities: {result.entities}")
    print(f"Relationships: {result.relationships}")
    print(f"Processing time: {result.processing_time_seconds}s")

    await pipeline.close()

asyncio.run(main())
```

### With Deduplication

```python
from knowledge_engine.integrations.kggen import (
    ExtractionPipeline,
    DeduplicationEngine,
    DeduplicationMethod
)

async def main():
    # Extract
    pipeline = ExtractionPipeline()
    result = await pipeline.extract(text=text)

    # Deduplicate
    dedup = DeduplicationEngine()
    dedup_result = await dedup.deduplicate(
        entities=result.entities,
        method=DeduplicationMethod.FULL
    )

    print(f"Original: {dedup_result.original_count}")
    print(f"After dedup: {dedup_result.final_count}")
    print(f"Reduction: {dedup_result.reduction_rate:.1%}")

    await pipeline.close()
    await dedup.close()

asyncio.run(main())
```

## Components

### 1. Extraction Pipeline (Task 2.1)

**3-Stage Pipeline:**
1. Entity Extraction - Extract entities from text
2. Relation Extraction - Extract SPO triples
3. Validation - Quality check results

**Features:**
- Parallel chunk processing
- Progress tracking
- Status monitoring
- Configurable timeouts

**Classes:**
- `ExtractionPipeline` - Main pipeline
- `ExtractionResult` - Result object
- `PipelineConfig` - Configuration
- `PipelineStatus` - Real-time status

### 2. Deduplication Engine (Task 2.2)

**Methods:**
- `SEMHASH` - Semantic hash-based deduplication
- `LM_CLUSTER` - Language model clustering
- `FULL` - Combined approach

**Features:**
- Cross-document resolution
- Temporal tracking
- Quality metrics

**Classes:**
- `DeduplicationEngine` - Main engine
- `SEMHASHStrategy` - Hash-based dedup
- `LMClusterStrategy` - Clustering dedup
- `CrossDocumentResolver` - Cross-document resolution

### 3. MCP Server (Task 2.3)

**Memory Tools:**
- `add_memories` - Add multiple memories
- `retrieve_relevant_memories` - Semantic search
- `visualize_memories` - Memory statistics

**Features:**
- Persistent storage
- Backup/restore
- Session aggregation
- Embedding-based retrieval

**Classes:**
- `KGGenMCPServer` - MCP server
- `MemoryManager` - Memory management
- `Memory` - Memory object
- `MemoryQuery` - Query object

### 4. Conversation Analyzer (Task 2.4)

**Features:**
- Speaker entity extraction
- Concept relationships
- Conversation summarization
- Knowledge graph conversion

**Classes:**
- `ConversationAnalyzer` - Main analyzer
- `SpeakerEntityExtractor` - Entity extraction
- `ConversationResult` - Analysis result

### 5. Graph Aggregator (Task 2.5)

**Features:**
- Multi-source aggregation
- Conflict resolution
- Graph versioning
- Differential comparison

**Classes:**
- `GraphAggregator` - Main aggregator
- `GraphVersion` - Versioned graph
- `GraphDiff` - Version comparison
- `ConflictResolver` - Conflict handling

## Configuration

### Pipeline Configuration

```python
from knowledge_engine.integrations.kggen import PipelineConfig

config = PipelineConfig(
    entity_model="gpt-4o",
    entity_temperature=0.0,
    chunk_size=5000,
    parallel_workers=4,
    entity_timeout=300.0
)
```

### Deduplication Configuration

```python
from knowledge_engine.integrations.kggen import DeduplicationConfig

config = DeduplicationConfig(
    semhash_threshold=0.95,
    lm_cluster_size=128,
    enable_temporal=True
)
```

### Memory Configuration

```python
from knowledge_engine.integrations.kggen import MemoryStoreConfig

config = MemoryStoreConfig(
    persistence_enabled=True,
    storage_path="./data/memories",
    backup_enabled=True
)
```

## Usage Examples

### Example 1: Extract and Deduplicate

```python
import asyncio
from knowledge_engine.integrations.kggen import (
    ExtractionPipeline,
    DeduplicationEngine,
    DeduplicationMethod
)

async def extract_and_deduplicate():
    # Initialize
    pipeline = ExtractionPipeline()
    dedup = DeduplicationEngine()

    # Extract
    text = "Apple and Google are major tech companies..."
    result = await pipeline.extract(text=text)

    # Deduplicate
    dedup_result = await dedup.deduplicate(
        entities=result.entities,
        method=DeduplicationMethod.FULL
    )

    # Use results
    print(f"Unique entities: {dedup_result.unique_entities}")
    print(f"Clusters: {len(dedup_result.entity_clusters)}")

    # Cleanup
    await pipeline.close()
    await dedup.close()

asyncio.run(extract_and_deduplicate())
```

### Example 2: Conversation Analysis

```python
from knowledge_engine.integrations.kggen import ConversationAnalyzer

async def analyze_conversation():
    analyzer = ConversationAnalyzer()

    messages = [
        {"role": "user", "content": "Tell me about Apple", "speaker_id": "user1"},
        {"role": "assistant", "content": "Apple is a tech company", "speaker_id": "bot"},
    ]

    result = await analyzer.analyze(messages)

    print(f"Speakers: {result.total_speakers}")
    print(f"Entities: {result.entities}")
    print(f"Summary: {result.summary}")

    await analyzer.close()

asyncio.run(analyze_conversation())
```

### Example 3: Memory Operations

```python
from knowledge_engine.integrations.kggen import (
    KGGenMCPServer,
    MemoryType
)

async def memory_operations():
    server = KGGenMCPServer()

    # Add memories
    memories_data = [
        {"content": "Apple is a tech company", "memory_type": "fact"},
        {"content": "Google owns Android", "memory_type": "fact"},
    ]

    result = await server.add_memories(
        memories=memories_data,
        session_id="session1"
    )

    # Retrieve
    result = await server.retrieve_relevant_memories(
        query_text="tech companies",
        session_id="session1",
        max_results=10
    )

    print(f"Retrieved {result['count']} memories")

    await server.close()

asyncio.run(memory_operations())
```

### Example 4: Graph Aggregation

```python
from knowledge_engine.integrations.kggen import GraphAggregator

async def aggregate_graphs():
    aggregator = GraphAggregator()

    graphs = [
        {
            "entities": ["Apple", "Google"],
            "relationships": [
                {"subject": "Apple", "predicate": "competes_with", "object": "Google"}
            ]
        },
        {
            "entities": ["Apple", "Microsoft"],
            "relationships": [
                {"subject": "Apple", "predicate": "competes_with", "object": "Microsoft"}
            ]
        }
    ]

    result = await aggregator.aggregate(graphs)

    print(f"Aggregated {result.source_versions_count} graphs")
    print(f"Total entities: {result.total_entities}")
    print(f"Conflicts resolved: {result.conflicts_resolved}")

    await aggregator.close()

asyncio.run(aggregate_graphs())
```

### Example 5: Full Pipeline

```python
import asyncio
from knowledge_engine.integrations.kggen import (
    ExtractionPipeline,
    DeduplicationEngine,
    DeduplicationMethod,
    GraphAggregator
)

async def full_pipeline():
    # 1. Extract
    pipeline = ExtractionPipeline()
    extraction_result = await pipeline.extract(text=document_text)

    # 2. Deduplicate
    dedup = DeduplicationEngine()
    dedup_result = await dedup.deduplicate(
        entities=extraction_result.entities,
        method=DeduplicationMethod.FULL
    )

    # 3. Aggregate into versioned graph
    aggregator = GraphAggregator()

    graph = {
        "entities": dedup_result.unique_entities,
        "relationships": extraction_result.relationships
    }

    agg_result = await aggregator.aggregate([graph])

    print(f"Final graph: {agg_result.total_entities} entities, "
          f"{agg_result.total_relationships} relationships")

    # Cleanup
    await pipeline.close()
    await dedup.close()
    await aggregator.close()

asyncio.run(full_pipeline())
```

## API Reference

### ExtractionPipeline

**Methods:**
- `extract(text, context, correlation_id, progress_callback)` - Extract knowledge graph
- `extract_batch(texts, context)` - Batch extraction
- `get_status(correlation_id)` - Get pipeline status
- `close()` - Cleanup resources

**Returns:**
- `ExtractionResult` with entities, relationships, metrics

### DeduplicationEngine

**Methods:**
- `deduplicate(entities, method, correlation_id, document_id)` - Deduplicate entities
- `deduplicate_relationships(relationships, correlation_id)` - Deduplicate relationships
- `get_entity_history(entity)` - Get entity temporal history
- `close()` - Cleanup resources

**Returns:**
- `DeduplicationResult` with unique entities, clusters, metrics

### KGGenMCPServer

**MCP Tools:**
- `add_memories(memories, session_id)` - Add memories
- `retrieve_relevant_memories(query_text, session_id, max_results)` - Retrieve memories
- `visualize_memories(session_id)` - Get statistics

**Returns:**
- Dict with success status, count, memories

### ConversationAnalyzer

**Methods:**
- `analyze(messages, conversation_id, correlation_id)` - Analyze conversation
- `close()` - Cleanup resources

**Returns:**
- `ConversationResult` with entities, relationships, summary

### GraphAggregator

**Methods:**
- `aggregate(graphs, correlation_id, create_version)` - Aggregate graphs
- `get_version(version_id)` - Get specific version
- `get_latest_version()` - Get latest version
- `compare_versions(version_id1, version_id2)` - Compare versions
- `list_versions(limit)` - List recent versions
- `close()` - Cleanup resources

**Returns:**
- `AggregationResult` with aggregated graph, metrics

## Testing

### Run Tests

```bash
# Run all tests
pytest knowledge_engine/integrations/kggen/test_sprint2.py -v

# Run specific test class
pytest knowledge_engine/integrations/kggen/test_sprint2.py::TestExtractionPipeline -v

# Run with coverage
pytest knowledge_engine/integrations/kggen/test_sprint2.py --cov=knowledge_engine.integrations.kggen
```

### Run Probes

```bash
# Verify extraction pipeline
bash knowledge_engine/integrations/kggen/probes/check_extraction_pipeline.sh

# Verify deduplication engine
bash knowledge_engine/integrations/kggen/probes/check_deduplication_engine.sh

# Verify MCP server
bash knowledge_engine/integrations/kggen/probes/check_mcp_server.sh
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```python
# Ensure PYTHONPATH includes project root
import sys
sys.path.append("/path/to/OpenEvolve/Frontend")
```

**2. Configuration Validation Errors**
```python
# All config values must be valid ranges
# Check environment variables are set correctly
config = PipelineConfig()
config.validate()  # Will raise ValueError if invalid
```

**3. Timeout Errors**
```python
# Increase timeout for large documents
config = PipelineConfig(
    entity_timeout=600.0,  # 10 minutes
    relation_timeout=1200.0  # 20 minutes
)
```

**4. Memory Issues**
```python
# Reduce parallel workers for large documents
config = PipelineConfig(
    parallel_workers=2  # Reduce from default 4
)
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tuning

**For large documents:**
- Reduce `chunk_size` (default 5000)
- Increase `parallel_workers` (default 4)
- Use `FULL` deduplication only when needed

**For real-time processing:**
- Reduce `chunk_size` to 2000-3000
- Set `parallel_workers` to CPU count
- Use `SEMHASH` for faster deduplication

## CLAUDE.md Compliance

This integration follows all CLAUDE.md principles:

1. **AIR GAP**: Adapter pattern, no direct imports from kg-gen source
2. **RUNTIME TRUTH**: Probe scripts verify all functionality
3. **IDEMPOTENCY**: All operations safe to retry
4. **CONFIGURATION EXPLICITNESS**: All config via environment variables
5. **UTC TIME**: All timestamps in UTC
6. **STRUCTURED LOGGING**: JSON logs with correlation IDs

## Additional Resources

- Pipeline usage examples: See `examples/` directory
- Test suite: `test_sprint2.py`
- Probe scripts: `probes/` directory
- API documentation: Inline docstrings in all modules

## Support

For issues or questions:
1. Check troubleshooting section
2. Run probe scripts to verify setup
3. Review test examples
4. Check logs with correlation IDs
