# Graphiti Temporal Integration - Complete Implementation Guide

## Overview

This document describes the Phase 2.1 implementation of Graphiti Core Integration, which extends the OpenEvolve Knowledge Engine with temporal reasoning, hybrid search, and contradiction detection capabilities using Graphiti's episodic knowledge graph.

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [Features](#features)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Performance Considerations](#performance-considerations)

---

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    TemporalKnowledgeEngine                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            KnowledgeEngine (Base)                          │  │
│  │  - Document indexing                                      │  │
│  │  - LLM integration                                        │  │
│  │  - Knowledge graph management                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         Temporal Layer (New)                               │  │
│  │  - KnowledgeArtifact model                                 │  │
│  │  - Temporal metadata tracking (valid_at, invalid_at)       │  │
│  │  - Point-in-time queries                                   │  │
│  │  - Contradiction detection                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         GraphitiTemporalBridge                             │  │
│  │  - Artifact to Episode conversion                          │  │
│  │  - Entity type mapping                                     │  │
│  │  - Search result transformation                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         GraphitiBridge                                     │  │
│  │  - Configuration management                                │  │
│  │  - Caching layer                                           │  │
│  │  - Graceful degradation                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │         GraphitiAdapter (Extended)                         │  │
│  │  - Hybrid search (BM25 + Vector + BFS)                     │  │
│  │  - Temporal filtering                                      │  │
│  │  - Contradiction detection                                 │  │
│  │  - Timeline reconstruction                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Graphiti Knowledge Graph                      │
│  - Neo4j / FalkorDB backend                                     │
│  - Episodic knowledge storage                                   │
│  - Entity and relationship extraction                           │
│  - Temporal metadata                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. TemporalKnowledgeEngine

**Location:** `knowledge_engine/core/temporal_knowledge_engine.py`

**Purpose:** Extended KnowledgeEngine with temporal reasoning capabilities.

**Key Features:**
- Temporal knowledge tracking (valid_at, invalid_at timestamps)
- Point-in-time queries
- Hybrid search integration
- Contradiction detection
- Timeline reconstruction

**Key Classes:**
- `TemporalKnowledgeEngine` - Main engine class
- `KnowledgeArtifact` - Canonical knowledge representation
- `RerankMethod` - Reranking strategy enum
- `ContradictionDetection` - Contradiction detection result

### 2. GraphitiAdapter (Extended)

**Location:** `integrations/graphiti/adapter.py`

**New Methods:**
- `search_hybrid()` - Hybrid search with configurable components
- `get_episode_at_time()` - Get episode state at specific time
- `find_contradictions()` - Detect contradictions
- `get_entity_timeline()` - Get entity timeline

### 3. GraphitiTemporalBridge

**Location:** `knowledge_engine/integrations/graphiti_temporal_bridge.py`

**Purpose:** High-level bridge between KnowledgeEngine and Graphiti.

**Key Features:**
- KnowledgeArtifact to Episode conversion
- Search result transformation
- Entity type mapping
- Temporal query translation

### 4. Configuration

**Location:** `integrations/graphiti/config.yaml`

**New Sections:**
- `features.temporal_reasoning` - Temporal reasoning settings
- `features.hybrid_search` - Hybrid search configuration
- `features.contradiction_detection` - Contradiction detection settings
- `features.entity_mappings` - Entity type mappings

---

## Features

### 1. Temporal Knowledge Tracking

Knowledge artifacts track when they are valid and invalid:

```python
artifact = KnowledgeArtifact(
    id="sol_001",
    content="Use async/await for I/O operations",
    artifact_type="solution_pattern",
    valid_at=datetime(2024, 1, 1),
    invalid_at=datetime(2024, 6, 1),  # Deprecated after this date
)
```

**Benefits:**
- Track knowledge evolution over time
- Query knowledge as it was at specific points
- Automatic expiration of outdated knowledge
- Historical analysis capabilities

### 2. Point-in-Time Queries

Query the knowledge base as it existed at a specific time:

```python
# What did we know about async in February?
results = await engine.query_at_time(
    query="async programming",
    timestamp=datetime(2024, 2, 15),
)
```

**Use Cases:**
- Understanding historical decisions
- Analyzing knowledge evolution
- Debugging past issues
- Compliance and auditing

### 3. Hybrid Search

Combine multiple search strategies for optimal results:

```python
results = await engine.search_with_graphiti(
    query="database optimization",
    use_hybrid=True,
    rerank_method="rrf",  # Reciprocal Rank Fusion
    max_results=10,
)
```

**Search Components:**
- **BM25** - Keyword search (TF-IDF based)
- **Vector** - Semantic search (embeddings)
- **Graph Traversal** - BFS for related entities

**Reranking Methods:**
- `rrf` - Reciprocal Rank Fusion (default)
- `cross_encoder` - Cross-encoder reranking
- `weighted` - Weighted combination
- `none` - No reranking

### 4. Contradiction Detection

Automatically detect contradictory knowledge:

```python
result = await engine.detect_contradictions()

if result.has_contradictions:
    for contradiction in result.contradictions:
        print(f"Conflict: {contradiction['reason']}")
```

**Detection Patterns:**
- "not" vs. "always"
- "cannot" vs. "can"
- "never" vs. "will"
- Temporal overlap check

### 5. Timeline Reconstruction

Reconstruct the history of an entity:

```python
timeline = await engine.get_timeline(
    entity="Python",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31),
)

for event in timeline:
    print(f"{event['timestamp']}: {event['description']}")
```

**Use Cases:**
- Project evolution tracking
- Knowledge change analysis
- Pattern recognition
- Trend identification

---

## Configuration

### Basic Configuration

```yaml
# integrations/graphiti/config.yaml

features:
  # Enable temporal tracking
  temporal_tracking: true

  # Temporal reasoning settings
  temporal_reasoning:
    enabled: true
    default_time_range_hours: 24
    track_validity: true
    point_in_time_queries: true

  # Hybrid search configuration
  hybrid_search:
    enabled: true
    components:
      use_bm25: true
      use_vector: true
      use_graph_traversal: true
    rerank_method: rrf
    reranking:
      rrf_k: 60
      weights:
        bm25: 0.3
        vector: 0.5
        graph_traversal: 0.2

  # Contradiction detection
  contradiction_detection:
    enabled: true
    patterns:
      - "not "
      - "never "
      - "cannot "
    confidence_threshold: 0.5

  # Entity type mappings
  entity_mappings:
    solution_pattern: Procedure
    workflow: Document
    agent: Organization
```

### Environment Setup

```bash
# Set Neo4j password
export NEO4J_PASSWORD=your_password

# Optional: Set custom Graphiti path
export GRAPHITI_PATH=/path/to/graphiti
```

---

## Usage Examples

### Example 1: Adding Temporal Knowledge

```python
from datetime import datetime, timedelta
from knowledge_engine.core.temporal_knowledge_engine import (
    TemporalKnowledgeEngine,
)

# Create engine
engine = TemporalKnowledgeEngine(
    enable_temporal=True,
    enable_hybrid_search=True,
)

# Add knowledge with temporal metadata
now = datetime.utcnow()

await engine.add_knowledge_temporal(
    content="Python 3.11 introduced significant performance improvements",
    artifact_type="solution_pattern",
    valid_at=now,
    metadata={"language": "python", "version": "3.11"},
)
```

### Example 2: Point-in-Time Query

```python
# Query knowledge as it was last month
last_month = datetime.utcnow() - timedelta(days=30)

results = await engine.query_at_time(
    query="python async performance",
    timestamp=last_month,
)

for artifact in results:
    print(f"{artifact.content}")
    print(f"Valid at: {artifact.valid_at}")
```

### Example 3: Hybrid Search

```python
# Search with hybrid approach
results = await engine.search_with_graphiti(
    query="semantic search optimization",
    use_hybrid=True,
    rerank_method="rrf",
    max_results=5,
)

for artifact in results:
    print(f"{artifact.content}")
```

### Example 4: Contradiction Detection

```python
# Detect contradictions
result = await engine.detect_contradictions()

if result.has_contradictions:
    print(f"Found {len(result.contradictions)} contradictions")

    for c in result.contradictions:
        print(f"  - {c['reason']}")
        print(f"    Severity: {c['severity']}")
```

### Example 5: Timeline Reconstruction

```python
# Get entity timeline
timeline = await engine.get_timeline(
    entity="Docker",
    start_time=datetime(2024, 1, 1),
    end_time=datetime.utcnow(),
)

for event in timeline:
    print(f"[{event['timestamp']}] {event['description']}")
```

---

## API Reference

### TemporalKnowledgeEngine

#### Methods

**`add_knowledge_temporal()`**
```python
async def add_knowledge_temporal(
    content: str,
    artifact_type: str,
    valid_at: datetime,
    invalid_at: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
    source: str = "openevolve",
    group_id: Optional[str] = None,
) -> Optional[KnowledgeArtifact]
```
Add knowledge with temporal metadata.

**`query_at_time()`**
```python
async def query_at_time(
    query: str,
    timestamp: datetime,
    max_results: int = 10,
    group_ids: Optional[List[str]] = None,
) -> List[KnowledgeArtifact]
```
Query knowledge as it was at a specific time.

**`search_with_graphiti()`**
```python
async def search_with_graphiti(
    query: str,
    use_hybrid: bool = True,
    rerank_method: str = "rrf",
    temporal_filters: Optional[Dict[str, Any]] = None,
    max_results: int = 10,
    group_ids: Optional[List[str]] = None,
) -> List[KnowledgeArtifact]
```
Search using Graphiti's hybrid search.

**`detect_contradictions()`**
```python
async def detect_contradictions(
    knowledge_id: Optional[str] = None,
    group_ids: Optional[List[str]] = None,
) -> ContradictionDetection
```
Detect contradictions in the knowledge base.

**`get_timeline()`**
```python
async def get_timeline(
    entity: str,
    start_time: datetime,
    end_time: datetime,
    group_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]
```
Get timeline of events for an entity.

### KnowledgeArtifact

#### Attributes

- `id` - Unique identifier
- `content` - Knowledge content
- `artifact_type` - Type of artifact
- `valid_at` - When knowledge becomes valid
- `invalid_at` - When knowledge becomes invalid (optional)
- `created_at` - Creation timestamp
- `source` - Source identifier
- `metadata` - Additional metadata
- `entities` - Related entities
- `relationships` - Relationships
- `confidence` - Confidence score (0-1)
- `group_id` - Optional group ID

#### Methods

**`is_valid_at(timestamp)`**
```python
def is_valid_at(self, timestamp: datetime) -> bool
```
Check if artifact is valid at a given time.

---

## Testing

### Running Tests

```bash
# Run all temporal tests
pytest knowledge_engine/tests/test_temporal_graphiti.py -v

# Run specific test
pytest knowledge_engine/tests/test_temporal_graphiti.py::TestTemporalKnowledgeEngine -v

# Run with coverage
pytest knowledge_engine/tests/test_temporal_graphiti.py --cov=knowledge_engine.core.temporal_knowledge_engine
```

### Test Coverage

The test suite covers:
- KnowledgeArtifact creation and serialization
- Temporal validity checking
- Point-in-time queries
- Hybrid search with different reranking methods
- Contradiction detection
- Timeline reconstruction
- Entity type mapping
- Artifact to Episode conversion

### Example Test

```python
@pytest.mark.asyncio
async def test_temporal_knowledge_tracking():
    engine = TemporalKnowledgeEngine(enable_temporal=True)

    now = datetime.utcnow()

    # Add knowledge
    artifact = await engine.add_knowledge_temporal(
        content="Test knowledge",
        artifact_type="solution_pattern",
        valid_at=now,
    )

    # Query at time
    results = await engine.query_at_time("test", now)

    assert len(results) > 0
    assert any(a.id == artifact.id for a in results)
```

---

## Performance Considerations

### Temporal Query Optimization

1. **Indexing**
   - Ensure Graphiti indices are built: `await graphiti.build_indices_and_constraints()`
   - Use appropriate time ranges for queries

2. **Caching**
   - Bridge caching is enabled by default (TTL: 3600s)
   - Adjust cache TTL in config.yaml

3. **Batch Operations**
   - Use `batch_size` in config.yaml for bulk operations
   - Recommended batch size: 100

### Hybrid Search Performance

**BM25 (Keyword Search)**
- Fast: O(n) where n is document count
- Best for: Exact term matching
- Memory: Low

**Vector (Semantic Search)**
- Medium: O(n) with optimizations
- Best for: Conceptual similarity
- Memory: Medium (embeddings)

**Graph Traversal (BFS)**
- Variable: O(V + E) where V=vertices, E=edges
- Best for: Contextual relationships
- Memory: Low

**Recommended Configuration:**
```yaml
hybrid_search:
  components:
    use_bm25: true
    use_vector: true
    use_graph_traversal: true
  rerank_method: rrf  # Best balance of speed and quality
```

### Scalability

**Small Scale** (< 10K artifacts)
- All search components enabled
- Neo4j Community Edition
- Single instance

**Medium Scale** (10K - 100K artifacts)
- Prioritize vector search
- Neo4j Enterprise (optional)
- Horizontal scaling

**Large Scale** (> 100K artifacts)
- Use filters to narrow scope
- Neo4j Enterprise with clustering
- Distributed architecture

---

## Troubleshooting

### Common Issues

**Issue: "Graphiti not available"**
```
Solution:
1. Ensure Graphiti is installed: pip install graphiti-core
2. Check the Graphiti path in adapter.py
3. Verify Neo4j/FalkorDB is running
```

**Issue: "Temporal queries return no results"**
```
Solution:
1. Check valid_at timestamps
2. Verify temporal filters are correct
3. Ensure knowledge has been added
```

**Issue: "Hybrid search is slow"**
```
Solution:
1. Reduce max_results
2. Disable graph_traversal if not needed
3. Use 'rrf' instead of 'cross_encoder'
4. Check Neo4j performance metrics
```

---

## Future Enhancements

1. **Advanced Temporal Reasoning**
   - Temporal logic queries (before, after, during)
   - Causal relationship detection
   - Event sequence prediction

2. **Enhanced Contradiction Detection**
   - ML-based contradiction detection
   - Context-aware contradiction resolution
   - Automatic knowledge reconciliation

3. **Performance Optimizations**
   - Parallel temporal queries
   - Incremental index updates
   - Query result caching

4. **Additional Features**
   - Knowledge versioning
   - Change tracking and alerts
   - Temporal knowledge visualization

---

## Conclusion

The Graphiti Temporal Integration provides powerful capabilities for temporal knowledge management, enabling:

- **Temporal Reasoning** - Query knowledge as it was at any point in time
- **Hybrid Search** - Combine keyword, semantic, and graph-based search
- **Contradiction Detection** - Automatically identify conflicting knowledge
- **Timeline Reconstruction** - Track knowledge evolution over time

This implementation maintains backward compatibility with existing Graphiti integration while adding sophisticated temporal capabilities to the OpenEvolve Knowledge Engine.
