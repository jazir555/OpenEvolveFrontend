# Graphiti Integration Guide

**Document Version:** 1.0
**Date:** 2026-01-02
**Project:** OpenEvolve Frontend - Temporal Knowledge Graph
**Integration Status:** ✅ **COMPLETE** - PRODUCTION READY

---

## Table of Contents

1. [Overview](#1-overview)
2. [Purpose and GAP Analysis](#2-purpose-and-gap-analysis)
3. [Technical Implementation](#3-technical-implementation)
4. [Architecture](#4-architecture)
5. [Integration Points](#5-integration-points)
6. [Configuration](#6-configuration)
7. [Usage Examples](#7-usage-examples)
8. [API Reference](#8-api-reference)
9. [Testing](#9-testing)
10. [Troubleshooting](#10-troubleshooting)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Overview

### 1.1 What is Graphiti?

**Graphiti** is a temporally-aware knowledge graph library developed by Zep that enables:

- **Temporal Knowledge Tracking**: All knowledge is timestamped and tracked over time
- **Hybrid Search**: Combines semantic search, BM25 keyword matching, and graph traversal
- **Multi-Backend Support**: Works with Neo4j and FalkorDB
- **Community Detection**: Automatically identifies related knowledge clusters
- **Episode-Based Learning**: Organizes knowledge into discrete episodes

**Repository**: https://github.com/getzep/graphiti

### 1.2 Why Integrate Graphiti?

Graphiti fills critical gaps in the OpenEvolve knowledge ecosystem:

| Feature | Benefit | Impact |
|---------|---------|--------|
| **Temporal Metadata** | Track when knowledge was valid | Understand evolution over time |
| **Hybrid Search** | Semantic + keyword + graph | Better relevance and recall |
| **Community Detection** | Auto-discover related concepts | Better knowledge organization |
| **Episode Model** | Natural knowledge units | Easier knowledge management |

### 1.3 Integration Status

```
┌─────────────────────────────────────────────────────────────┐
│  GRAPHITI INTEGRATION STATUS: 100% COMPLETE                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Adapter Implementation       [████████████████████████] 100% │
│  Bridge Integration          [████████████████████████] 100% │
│  Configuration Management    [████████████████████████] 100% │
│  Documentation               [████████████████████████] 100% │
│  Testing Suite               [████████████████████████] 100% │
│  Knowledge Engine Updates    [████████████████████████] 100% │
│                                                             │
│  OVERALL COMPLETION: 100% ✅                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Purpose and GAP Analysis

### 2.1 GAP-14: Temporal Knowledge

**Problem**: OpenEvolve lacked temporal awareness in knowledge storage and retrieval.

**Solution**: Graphiti provides native temporal metadata tracking for all knowledge.

**Impact**:
- ✅ Knowledge can be queried with temporal filters
- ✅ Historical knowledge is preserved
- ✅ Knowledge evolution can be tracked
- ✅ Temporal reasoning is enabled

### 2.2 GAP-10: Knowledge Extraction

**Problem**: Limited knowledge extraction capabilities without temporal context.

**Solution**: Graphiti's episode-based model and hybrid search enhance extraction.

**Impact**:
- ✅ Better extraction through community detection
- ✅ Hybrid search improves recall
- ✅ Episode structure organizes extracted knowledge
- ✅ Graph traversal discovers hidden relationships

### 2.3 Use Cases

1. **Workflow Learning**: Track workflow evolution over time
2. **Decision History**: Understand why decisions were made
3. **Knowledge Evolution**: See how concepts change
4. **Contextual Search**: Find knowledge relevant to specific time periods
5. **Community Discovery**: Auto-organize related knowledge

---

## 3. Technical Implementation

### 3.1 Decoupled Adapter Pattern

The integration uses a **decoupled adapter pattern** to ensure zero modifications to Graphiti source code:

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTER PATTERN                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  OpenEvolve Knowledge Engine                               │
│           ↓                                                 │
│  KnowledgeGraphInterface (Abstract)                        │
│           ↓                                                 │
│  GraphitiAdapter (Implementation)                          │
│           ↓                                                 │
│  Graphiti Library (No Modifications)                       │
│           ↓                                                 │
│  Neo4j / FalkorDB                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Benefits**:
- ✅ Zero modifications to Graphiti source
- ✅ Easy to update Graphiti independently
- ✅ Consistent interface across knowledge backends
- ✅ Graceful degradation when Graphiti unavailable

### 3.2 Component Architecture

```
integrations/graphiti/
├── __init__.py              # Package exports
├── adapter.py               # Implements KnowledgeGraphInterface
├── bridge.py                # Connects to OpenEvolve knowledge_engine
└── config.yaml              # Configuration template

docs/integrations/
└── GRAPHITI_INTEGRATION_GUIDE.md  # This document

tests/integrations/
└── test_graphiti_integration.py   # Test suite

knowledge_engine/
└── bedrock_kb.py           # Updated to use Graphiti adapter
```

### 3.3 Key Design Decisions

1. **Singleton Bridge**: Single bridge instance for all connections
2. **Graceful Degradation**: Falls back gracefully when Graphiti unavailable
3. **Caching Layer**: Optional caching for improved performance
4. **Config-Based**: YAML configuration for easy management
5. **Environment Variables**: Secure password handling via env vars

---

## 4. Architecture

### 4.1 System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    OPENEVOLVE KNOWLEDGE ENGINE                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────┐         ┌──────────────────┐                 │
│  │  Workflows     │────────▶│  Knowledge       │                 │
│  │  (Producers)   │         │  Engine          │                 │
│  └────────────────┘         └────────┬─────────┘                 │
│                                      │                            │
│                         ┌────────────┼────────────┐              │
│                         │            │            │              │
│                         ▼            ▼            ▼              │
│              ┌─────────────┐ ┌──────────┐ ┌─────────────┐       │
│              │  Bedrock KB │ │ Graphiti │ │  Other KBs  │       │
│              │  (Traditional)│(Temporal)│ │             │       │
│              └─────────────┘ └────┬─────┘ └─────────────┘       │
│                                    │                            │
│                         ┌──────────┼──────────┐                │
│                         │          │          │                │
│                         ▼          ▼          ▼                │
│              ┌─────────────┐ ┌──────┐ ┌───────────┐           │
│              │  Elasticsearch│ Neo4j │ │ FalkorDB  │           │
│              └─────────────┘ └──────┘ └───────────┘           │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

**Adding Knowledge**:
```
Workflow Episode
    ↓
GraphitiBridge.add_episode()
    ↓
GraphitiAdapter.add_episode()
    ↓
Graphiti Library
    ↓
Neo4j/FalkorDB (with temporal metadata)
```

**Searching Knowledge**:
```
Search Query
    ↓
GraphitiBridge.search()
    ↓
Check Cache
    ↓ (cache miss)
GraphitiAdapter.search()
    ↓
Graphiti Hybrid Search
    ↓ (semantic + BM25 + graph)
Results + Temporal Context
```

### 4.3 Integration Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: INTERFACE                       │
│  KnowledgeGraphInterface - Abstract contract                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 2: ADAPTER                         │
│  GraphitiAdapter - Implements interface for Graphiti        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 3: BRIDGE                          │
│  GraphitiBridge - Manages singleton, config, caching        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 4: ENGINE                          │
│  OpenEvolve Knowledge Engine - Orchestrates all KBs         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Integration Points

### 5.1 Knowledge Engine Integration

The Graphiti bridge integrates into the OpenEvolve knowledge engine at multiple points:

**File**: `knowledge_engine/bedrock_kb.py`

```python
# Before: Only Bedrock KB
class BedrockKnowledgeBaseClient:
    async def query_knowledge_base(self, kb_id, query):
        # Query Bedrock only
        pass

# After: Bedrock KB + Graphiti
class BedrockKnowledgeBaseClient:
    def __init__(self, use_graphiti: bool = False):
        self.use_graphiti = use_graphiti
        self.graphiti_bridge = None

    async def query_knowledge_base(self, kb_id, query):
        # Query Bedrock
        bedrock_results = await self._query_bedrock(kb_id, query)

        # Optionally query Graphiti for temporal context
        if self.use_graphiti and self.graphiti_bridge:
            graphiti_results = await self.graphiti_bridge.search(query)
            return self._merge_results(bedrock_results, graphiti_results)

        return bedrock_results
```

### 5.2 Workflow Integration

Workflows can add episodes to Graphiti:

```python
from integrations.graphiti import get_bridge
from datetime import datetime

async def workflow_completion_handler(workflow_result):
    """Add workflow episode to Graphiti."""
    bridge = await get_bridge()
    if not bridge.is_initialized:
        return

    await bridge.add_episode(
        name=f"workflow_{workflow_result.id}",
        body=workflow_result.summary,
        reference_time=datetime.now(),
        metadata={
            "workflow_type": workflow_result.type,
            "status": workflow_result.status,
            "duration": workflow_result.duration,
        },
        source="workflow",
        group_id=workflow_result.project_id
    )
```

### 5.3 MCP Integration (Optional)

Graphiti can be exposed via MCP server for isolated operation:

```yaml
# config.yaml
features:
  mcp_server: true  # Enable MCP server

# This allows Graphiti to run as a separate service
# Communicating via MCP protocol
```

---

## 6. Configuration

### 6.1 Configuration File

**Location**: `integrations/graphiti/config.yaml`

```yaml
project:
  name: Graphiti
  version: 0.1.0
  enabled: true

connection:
  backend: neo4j  # or falkordb
  uri: bolt://localhost:7687
  user: neo4j
  password: ${NEO4J_PASSWORD}  # Environment variable

features:
  temporal_tracking: true
  hybrid_search: true
  mcp_server: false

integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true

performance:
  max_workers: 4
  timeout: 30
  batch_size: 100
```

### 6.2 Environment Variables

```bash
# Neo4j Configuration
export NEO4J_PASSWORD=your_secure_password

# FalkorDB Configuration (if using FalkorDB)
export FALKORDB_URI=redis://localhost:6379
export FALKORDB_PASSWORD=your_secure_password

# Optional: Custom Graphiti location
export GRAPHITI_PATH=/path/to/graphiti
```

### 6.3 Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `project.enabled` | boolean | true | Enable/disable Graphiti |
| `connection.backend` | string | neo4j | Backend: neo4j or falkordb |
| `connection.uri` | string | bolt://localhost:7687 | Database connection URI |
| `connection.user` | string | neo4j | Database username (Neo4j) |
| `connection.password` | string | - | Database password (use env var) |
| `features.temporal_tracking` | boolean | true | Enable temporal metadata |
| `features.hybrid_search` | boolean | true | Enable hybrid search |
| `features.mcp_server` | boolean | false | Enable MCP server |
| `integration.auto_start` | boolean | true | Auto-start on init |
| `integration.cache_enabled` | boolean | true | Enable caching |
| `integration.cache_ttl` | int | 3600 | Cache TTL in seconds |
| `integration.fallback_on_error` | boolean | true | Graceful fallback |
| `performance.max_workers` | int | 4 | Max concurrent workers |
| `performance.timeout` | int | 30 | Operation timeout (seconds) |
| `performance.batch_size` | int | 100 | Batch operation size |

### 6.4 Backend Configuration

**Neo4j**:
```yaml
connection:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password: ${NEO4J_PASSWORD}
```

**FalkorDB**:
```yaml
connection:
  backend: falkordb
  uri: redis://localhost:6379
  password: ${FALKORDB_PASSWORD}
```

---

## 7. Usage Examples

### 7.1 Basic Usage

```python
from integrations.graphiti import get_bridge
from datetime import datetime

# Get bridge instance
bridge = await get_bridge("integrations/graphiti/config.yaml")

# Initialize
await bridge.initialize()

# Add an episode
await bridge.add_episode(
    name="project_kickoff",
    body="Project X kicked off with team of 5 developers",
    reference_time=datetime.now(),
    metadata={"project": "X", "team_size": 5},
    source="openevolve"
)

# Search
results = await bridge.search("Project X team")
print(f"Found {len(results['nodes'])} nodes, {len(results['edges'])} edges")
```

### 7.2 Temporal Search

```python
from integrations.base.knowledge_interface import TemporalFilter
from datetime import datetime, timedelta

# Search with temporal filter
results = await bridge.search(
    query="Project decisions",
    temporal_filters={
        "filter_type": TemporalFilter.TIME_RANGE,
        "start_time": datetime.now() - timedelta(days=30),
        "end_time": datetime.now()
    },
    num_results=20
)
```

### 7.3 Community Detection

```python
# Detect communities in knowledge graph
communities = await bridge.get_community_detections()

print(f"Found {len(communities['communities'])} communities")
for community in communities['communities']:
    print(f"  - {community['name']}: {community['summary']}")
```

### 7.4 Workflow Integration

```python
async def workflow_with_graphiti(workflow_func, *args, **kwargs):
    """Execute workflow and store results in Graphiti."""
    # Execute workflow
    result = await workflow_func(*args, **kwargs)

    # Store in Graphiti
    bridge = await get_bridge()
    if bridge.is_initialized:
        await bridge.add_episode(
            name=result.name,
            body=result.summary,
            reference_time=datetime.now(),
            metadata=result.metadata,
            source="workflow"
        )

    return result
```

### 7.5 Hybrid Search Example

```python
# Graphiti performs hybrid search:
# 1. Semantic search (vector embeddings)
# 2. BM25 keyword search
# 3. Graph traversal (relationship following)

results = await bridge.search(
    query="machine learning algorithms for time series",
    num_results=10
)

# Results include:
# - Nodes: Relevant entities (concepts, people, documents)
# - Edges: Relationships between entities
# - Context: Episodic context for results
```

### 7.6 Direct Adapter Usage

```python
from integrations.graphiti import GraphitiAdapter

# Create adapter
adapter = GraphitiAdapter()

# Initialize with custom config
await adapter.initialize({
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password',
    'backend': 'neo4j'
})

# Use adapter
await adapter.add_episode(...)
results = await adapter.search(...)
```

---

## 8. API Reference

### 8.1 GraphitiBridge

#### `get_bridge(config_path: Optional[str] = None) -> GraphitiBridge`

Get the singleton bridge instance.

**Parameters**:
- `config_path`: Optional path to config.yaml

**Returns**: `GraphitiBridge` instance

**Example**:
```python
bridge = await get_bridge("integrations/graphiti/config.yaml")
```

---

#### `async initialize(config: Optional[Dict[str, Any]] = None) -> bool`

Initialize the Graphiti adapter.

**Parameters**:
- `config`: Optional configuration dict

**Returns**: `True` if successful

**Example**:
```python
success = await bridge.initialize()
```

---

#### `async add_episode(name, body, reference_time, metadata, source, group_id) -> Dict[str, Any]`

Add an episode to Graphiti.

**Parameters**:
- `name` (str): Episode name
- `body` (str): Episode content
- `reference_time` (datetime): When the episode occurred
- `metadata` (Optional[Dict[str, Any]]): Additional metadata
- `source` (str): Source identifier (default: "openevolve")
- `group_id` (Optional[str]): Group/partition identifier

**Returns**: Dictionary with episode results

**Example**:
```python
result = await bridge.add_episode(
    name="meeting_1",
    body="Discussed project roadmap",
    reference_time=datetime.now(),
    metadata={"attendees": 5}
)
```

---

#### `async search(query, temporal_filters, num_results, group_ids) -> Dict[str, Any]`

Search Graphiti with optional temporal filtering.

**Parameters**:
- `query` (str): Search query
- `temporal_filters` (Optional[Dict[str, Any]]): Temporal filters
- `num_results` (int): Max results (default: 10)
- `group_ids` (Optional[List[str]]): Group IDs to search

**Returns**: Search results dictionary

**Example**:
```python
results = await bridge.search(
    query="project decisions",
    num_results=20
)
```

---

#### `async get_community_detections(group_ids) -> Dict[str, Any]`

Get or compute community detections.

**Parameters**:
- `group_ids` (Optional[List[str]]): Group IDs to analyze

**Returns**: Community information

**Example**:
```python
communities = await bridge.get_community_detections()
```

---

#### `async validate() -> Dict[str, Any]`

Validate the bridge and adapter.

**Returns**: Validation results

**Example**:
```python
validation = await bridge.validate()
print(f"Valid: {validation['is_valid']}")
```

---

#### `async shutdown() -> bool`

Shutdown the bridge and adapter.

**Returns**: `True` if successful

**Example**:
```python
await bridge.shutdown()
```

---

### 8.2 GraphitiAdapter

The adapter implements `KnowledgeGraphInterface` with all methods:

- `async initialize(config) -> bool`
- `async add_episode(name, body, reference_time, metadata, source, group_id) -> Dict[str, Any]`
- `async search(query, temporal_filters, num_results, group_ids) -> Dict[str, Any]`
- `async get_community_detections(group_ids) -> Dict[str, Any]`
- `async validate() -> Dict[str, Any]`
- `async shutdown() -> bool`
- `async get_episodes(reference_time, last_n, group_ids) -> List[Dict[str, Any]]`
- `async add_triplet(source_entity, relationship, target_entity) -> Dict[str, Any]`
- `async remove_episode(episode_uuid) -> bool`

---

## 9. Testing

### 9.1 Running Tests

```bash
# Run all Graphiti integration tests
pytest tests/integrations/test_graphiti_integration.py -v

# Run specific test
pytest tests/integrations/test_graphiti_integration.py::test_adapter_initialization -v

# Run with coverage
pytest tests/integrations/test_graphiti_integration.py --cov=integrations/graphiti
```

### 9.2 Test Coverage

The test suite covers:

1. **Adapter Tests**:
   - Initialization with Neo4j
   - Initialization with FalkorDB
   - Add episode
   - Search
   - Community detection
   - Validation
   - Shutdown

2. **Bridge Tests**:
   - Singleton pattern
   - Config loading
   - Caching
   - Graceful degradation
   - Fallback behavior

3. **Integration Tests**:
   - Knowledge engine integration
   - Workflow integration
   - Concurrent operations

### 9.3 Mock Testing

For tests without Neo4j:

```python
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_adapter_without_graphiti():
    """Test graceful degradation when Graphiti unavailable."""
    with patch('integrations.graphiti.adapter.GRAPHITI_AVAILABLE', False):
        adapter = GraphitiAdapter()
        with pytest.raises(ConfigurationError):
            await adapter.initialize({})
```

### 9.4 Integration Test Example

```python
@pytest.mark.asyncio
async def test_full_workflow():
    """Test complete workflow with Graphiti."""
    # Initialize
    bridge = await get_bridge("test_config.yaml")
    await bridge.initialize()

    # Add episodes
    await bridge.add_episode(
        name="episode_1",
        body="Test episode content",
        reference_time=datetime.now()
    )

    # Search
    results = await bridge.search("test")

    # Verify
    assert len(results['nodes']) > 0
    assert len(results['edges']) >= 0

    # Cleanup
    await bridge.shutdown()
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Issue: "Graphiti not available"

**Symptoms**:
```
ConfigurationError: Graphiti is not available. Please ensure it is installed.
```

**Solutions**:
1. Install Graphiti:
   ```bash
   cd "projects to analyze/graphiti"
   pip install -e .
   ```

2. Check Graphiti path in `adapter.py`

3. Verify imports:
   ```python
   from graphiti_core import Graphiti
   ```

---

#### Issue: "Connection refused to Neo4j"

**Symptoms**:
```
ConnectionError: Failed to connect to Graphiti backend: Connection refused
```

**Solutions**:
1. Verify Neo4j is running:
   ```bash
   # Check Neo4j status
   systemctl status neo4j

   # Or check process
   ps aux | grep neo4j
   ```

2. Verify connection settings in `config.yaml`:
   ```yaml
   connection:
     uri: bolt://localhost:7687
     user: neo4j
     password: ${NEO4J_PASSWORD}
   ```

3. Test connection manually:
   ```bash
   cypher-shell -u neo4j -p your_password
   ```

---

#### Issue: "Password not set"

**Symptoms**:
```
ConfigurationError: Database password is required
```

**Solutions**:
1. Set environment variable:
   ```bash
   export NEO4J_PASSWORD=your_secure_password
   ```

2. Or hardcode in config (not recommended for production):
   ```yaml
   connection:
     password: "your_password"
   ```

---

#### Issue: "Search returns empty results"

**Symptoms**:
```python
results = await bridge.search("query")
assert results['nodes'] == []  # Empty!
```

**Solutions**:
1. Verify episodes have been added:
   ```python
   episodes = await bridge.adapter.get_episodes(datetime.now())
   print(f"Episodes: {len(episodes)}")
   ```

2. Check Graphiti indices are built:
   ```python
   # Should be automatic, but verify
   await bridge.adapter.graphiti.build_indices_and_constraints()
   ```

3. Try broader search terms

4. Check if backend has data:
   ```cypher
   # In Neo4j console
   MATCH (n) RETURN count(n);
   ```

---

#### Issue: "Community detection fails"

**Symptoms**:
```
AnalysisError: Failed to detect communities
```

**Solutions**:
1. Ensure sufficient data (need at least 10-20 episodes)

2. Check graph connectivity:
   ```cypher
   // In Neo4j console
   MATCH (n)-[r]->(m) RETURN count(r);
   ```

3. Manually trigger community building:
   ```python
   await bridge.adapter.graphiti.build_communities()
   ```

---

### 10.2 Performance Issues

#### Slow Search Performance

**Solutions**:
1. Enable caching:
   ```yaml
   integration:
     cache_enabled: true
     cache_ttl: 3600
   ```

2. Increase worker count:
   ```yaml
   performance:
     max_workers: 8  # Increase from 4
   ```

3. Optimize Neo4j:
   ```yaml
   # In neo4j.conf
   dbms.memory.heap.initial_size=2g
   dbms.memory.heap.max_size=4g
   ```

4. Use batch operations for bulk inserts:
   ```yaml
   performance:
     batch_size: 100  # Increase batch size
   ```

---

### 10.3 Debug Mode

Enable debug logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('integrations.graphiti').setLevel(logging.DEBUG)
logging.getLogger('graphiti_core').setLevel(logging.DEBUG)
```

Or in config:

```yaml
logging:
  level: DEBUG
```

---

### 10.4 Health Check

```python
async def health_check():
    """Check Graphiti integration health."""
    bridge = await get_bridge()

    # Check availability
    print(f"Available: {bridge.is_available}")

    # Check initialization
    print(f"Initialized: {bridge.is_initialized}")

    # Validate
    validation = await bridge.validate()
    print(f"Valid: {validation['is_valid']}")
    print(f"Issues: {validation['issues']}")
    print(f"Metrics: {validation['metrics']}")

    return bridge.is_initialized and validation['is_valid']
```

---

## 11. Future Enhancements

### 11.1 Planned Features

1. **Advanced Temporal Queries**
   - Time-series analysis of knowledge evolution
   - Temporal reasoning (before/after relations)
   - Historical trend detection

2. **Enhanced Community Detection**
   - Hierarchical community structures
   - Dynamic community evolution tracking
   - Community-based summarization

3. **Multi-Modal Knowledge**
   - Support for images in episodes
   - Video transcript integration
   - Audio extraction

4. **Advanced Search**
   - Natural language query parsing
   - Query suggestion/autocompletion
   - Result explanation/why this result

5. **Performance Optimizations**
   - Async batch operations
   - Connection pooling
   - Query optimization

6. **Monitoring & Observability**
   - Prometheus metrics
   - OpenTelemetry tracing
   - Performance dashboards

### 11.2 Integration Opportunities

1. **With BubbleLabs**:
   - Graph-based workflow knowledge
   - Community-aware workflow generation

2. **With Hephaestus**:
   - Task delegation knowledge tracking
   - Delegation effectiveness analysis

3. **With ROMA/MDAP/Maker**:
   - Decision evolution tracking
   - Multi-agent interaction history

4. **With LeanAide**:
   - Mathematical knowledge temporal tracking
   - Proof evolution over time

### 11.3 Research Directions

1. **Temporal Knowledge Graphs**: Advanced temporal reasoning
2. **Causal Discovery**: Extract causal relationships from episodes
3. **Knowledge Fusion**: Merge knowledge from multiple sources
4. **Explainable AI**: Explain why knowledge was retrieved

---

## Appendix A: Quick Reference

### A.1 Installation

```bash
# Install Graphiti
cd "projects to analyze/graphiti"
pip install -e .

# Set environment variables
export NEO4J_PASSWORD=your_password
```

### A.2 Basic Usage

```python
from integrations.graphiti import get_bridge
from datetime import datetime

# Initialize
bridge = await get_bridge("integrations/graphiti/config.yaml")
await bridge.initialize()

# Add episode
await bridge.add_episode(
    name="episode_1",
    body="Content here",
    reference_time=datetime.now()
)

# Search
results = await bridge.search("query")
```

### A.3 Configuration

```yaml
connection:
  backend: neo4j
  uri: bolt://localhost:7687
  user: neo4j
  password: ${NEO4J_PASSWORD}

integration:
  auto_start: true
  cache_enabled: true
```

### A.4 Validation

```python
validation = await bridge.validate()
assert validation['is_valid']
```

---

## Appendix B: Resources

- **Graphiti GitHub**: https://github.com/getzep/graphiti
- **Graphiti Docs**: https://docs.getzep.com/graphiti
- **Neo4j Docs**: https://neo4j.com/docs/
- **FalkorDB Docs**: https://www.falkordb.com/docs/
- **OpenEvolve Docs**: `/docs`

---

**Document End**

For questions or issues, refer to:
- Graphiti GitHub Issues: https://github.com/getzep/graphiti/issues
- OpenEvolve Documentation: `/docs`
- Testing Guide: `/docs/integrations/INTEGRATION_TESTING_GUIDE.md`
