# Unified Knowledge Graph Manager - API Design Document

## Overview

The Unified Knowledge Graph Manager provides a consistent API interface across multiple knowledge graph backends (Neo4j, Qdrant, MongoDB, KarateClub, Memory). This document describes the complete API design.

## Table of Contents

1. [Architecture](#architecture)
2. [REST API Endpoints](#rest-api-endpoints)
3. [Python API](#python-api)
4. [Data Models](#data-models)
5. [Error Handling](#error-handling)
6. [Configuration](#configuration)
7. [Examples](#examples)

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│           Unified Knowledge Graph Manager                    │
│  - Automatic backend selection                               │
│  - Fallback chain management                                 │
│  - Unified result formatting                                 │
│  - Health monitoring & circuit breaking                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Neo4j     │    │   Qdrant    │    │  MongoDB    │
│  Backend    │    │   Backend   │    │  Backend    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Graph       │    │ Vector      │    │ Document    │
│ Storage     │    │ Search      │    │ Storage     │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Backend Selection Strategy

1. **Operation-based routing**: Different operations use different backends
2. **Health checking**: Only healthy backends are selected
3. **Fallback chain**: If primary backend fails, try fallbacks
4. **Performance tracking**: Monitor backend performance for optimization

---

## REST API Endpoints

### Base URL
```
/api/kg
```

### 1. Add Knowledge

**Endpoint:** `POST /api/kg/knowledge`

**Description:** Add a new knowledge entry to the graph.

**Request Body:**
```json
{
  "source": "string",           // Source identifier
  "content": "string",          // Knowledge content
  "metadata": {                 // Optional metadata
    "tags": ["tag1", "tag2"],
    "author": "user@example.com",
    "priority": "high"
  },
  "use_graph": true             // Use graph backend (default: true)
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "entry_id": "uuid-string",
  "backend_used": "neo4j",
  "timestamp": "2025-01-07T12:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input data
- `503 Service Unavailable`: All backends unavailable

---

### 2. Search Knowledge

**Endpoint:** `GET /api/kg/search`

**Description:** Search knowledge in the graph.

**Query Parameters:**
- `query` (string, required): Search query
- `source` (string, optional): Filter by source
- `tags` (array, optional): Filter by tags
- `date_after` (string, optional): ISO date filter
- `use_graph` (boolean, optional): Use graph search (default: true)
- `limit` (integer, optional): Max results (default: 10)
- `offset` (integer, optional): Result offset (default: 0)

**Example Request:**
```
GET /api/kg/search?query=graph+database&source=documentation&limit=5
```

**Response:** `200 OK`
```json
{
  "query": "graph database",
  "results": [
    {
      "id": "uuid-1",
      "source": "documentation",
      "content": "Neo4j is a graph database...",
      "metadata": {
        "tags": ["neo4j", "graph"]
      },
      "score": 0.95,
      "timestamp": "2025-01-07T12:00:00Z"
    }
  ],
  "total_count": 42,
  "backend_used": "qdrant",
  "search_time_ms": 45.2,
  "filters": {
    "source": "documentation"
  }
}
```

---

### 3. Batch Add Knowledge

**Endpoint:** `POST /api/kg/knowledge/batch`

**Description:** Add multiple knowledge entries efficiently.

**Request Body:**
```json
{
  "entries": [
    {
      "source": "doc1",
      "content": "Content 1",
      "metadata": {"key": "value1"}
    },
    {
      "source": "doc2",
      "content": "Content 2",
      "metadata": {"key": "value2"}
    }
  ]
}
```

**Response:** `200 OK`
```json
{
  "success": true,
  "entry_ids": ["uuid-1", "uuid-2"],
  "backend_used": "mongodb",
  "count": 2
}
```

---

### 4. Analyze Graph

**Endpoint:** `POST /api/kg/analyze`

**Description:** Perform analysis on the knowledge graph.

**Request Body:**
```json
{
  "analysis_type": "string",    // Analysis type (see below)
  "target": "string"            // Optional target entity/graph
}
```

**Supported Analysis Types:**

- `connected_components`: Find connected components
- `entity_connections`: Find most connected entities
- `knowledge_by_source`: Analyze knowledge distribution by source
- `community_detection`: Detect communities (KarateClub)
- `node_embedding`: Generate node embeddings
- `centrality`: Calculate centrality measures
- `role_detection`: Detect structural roles
- `graph_statistics`: General graph statistics
- `source_distribution`: Distribution by source
- `tag_distribution`: Distribution by tags
- `temporal_analysis`: Knowledge over time

**Response:** `200 OK`
```json
{
  "analysis_type": "entity_connections",
  "target": "graph",
  "results": {
    "top_entities": [
      {"entity": "Neo4j", "connections": 42},
      {"entity": "Qdrant", "connections": 31}
    ]
  },
  "backend_used": "neo4j",
  "analysis_time_ms": 123.4
}
```

---

### 5. Visualize Graph

**Endpoint:** `GET /api/kg/visualize`

**Query Parameters:**
- `format` (string): Output format - `html`, `json`, `dot` (default: `html`)
- `options` (object, optional): Visualization parameters

**Example Request:**
```
GET /api/kg/visualize?format=html
```

**Response:** `200 OK`

For `format=html`:
```html
<!DOCTYPE html>
<html>
...
</html>
```

For `format=json`:
```json
{
  "nodes": [...],
  "edges": [...]
}
```

---

### 6. Get Statistics

**Endpoint:** `GET /api/kg/stats`

**Description:** Get comprehensive statistics across all backends.

**Response:** `200 OK`
```json
{
  "timestamp": "2025-01-07T12:00:00Z",
  "backends": {
    "neo4j": {
      "node_count": 1523,
      "edge_count": 3421,
      "metadata": {
        "knowledge_nodes": 1200,
        "entity_nodes": 323,
        "mention_edges": 3421
      }
    },
    "qdrant": {
      "node_count": 1200,
      "edge_count": 0,
      "metadata": {
        "collection": "knowledge_graph",
        "vector_size": 1536
      }
    }
  },
  "performance": {
    "neo4j": {
      "avg_time_ms": 45.2,
      "min_time_ms": 12.1,
      "max_time_ms": 123.4,
      "total_operations": 234
    }
  }
}
```

---

### 7. Health Check

**Endpoint:** `GET /api/kg/health`

**Description:** Check health of all backends.

**Response:** `200 OK`
```json
{
  "healthy": true,
  "backends": {
    "neo4j": true,
    "qdrant": true,
    "mongodb": false,
    "karateclub": true
  },
  "timestamp": "2025-01-07T12:00:00Z"
}
```

---

### 8. Delete Knowledge

**Endpoint:** `DELETE /api/kg/knowledge/{entry_id}`

**Description:** Delete a knowledge entry by ID.

**Response:** `200 OK`
```json
{
  "success": true,
  "deleted": true,
  "backend_used": "neo4j"
}
```

---

### 9. Export Knowledge

**Endpoint:** `GET /api/kg/export/{backend_name}`

**Description:** Export all knowledge from a specific backend.

**Query Parameters:**
- `output_path` (string): Path to save export file

**Response:** `200 OK`
```json
{
  "success": true,
  "backend": "neo4j",
  "output_path": "/exports/neo4j_export_20250107.json",
  "entries_count": 1523
}
```

---

## Python API

### Basic Usage

```python
from knowledge_engine.core import UnifiedKnowledgeGraph

# Initialize manager
kg = UnifiedKnowledgeGraph(config_path="config.yaml")

# Connect to backends
await kg.connect_all()

try:
    # Add knowledge
    entry_id = await kg.add_knowledge(
        source="documentation",
        content="Neo4j provides efficient graph traversal.",
        metadata={"tags": ["neo4j", "graph"], "priority": "high"}
    )

    # Search
    results = await kg.search(
        query="graph traversal",
        filters={"source": "documentation"},
        limit=10
    )

    # Analyze
    analysis = await kg.analyze(
        analysis_type="entity_connections"
    )

    # Visualize
    visualization = await kg.visualize(output_format="html")

    # Get statistics
    stats = await kg.get_graph_stats()

finally:
    await kg.disconnect_all()
```

### Using Async Context Manager

```python
async with UnifiedKnowledgeGraph() as kg:
    await kg.add_knowledge(
        source="example",
        content="Automatic resource cleanup"
    )

    results = await kg.search("automatic cleanup")
    # Resources automatically cleaned up on exit
```

### Batch Operations

```python
entries = [
    {"source": "doc1", "content": "Content 1", "metadata": {"key": "1"}},
    {"source": "doc2", "content": "Content 2", "metadata": {"key": "2"}},
    {"source": "doc3", "content": "Content 3", "metadata": {"key": "3"}},
]

ids = await kg.batch_add_knowledge(entries)
print(f"Added {len(ids)} entries")
```

---

## Data Models

### KnowledgeEntry

```python
@dataclass
class KnowledgeEntry:
    source: str                          # Source identifier
    content: str                         # Knowledge content
    metadata: Optional[Dict[str, Any]]   # Optional metadata
    embedding: Optional[List[float]]     # Optional vector embedding
    timestamp: Optional[str]             # ISO timestamp
    id: Optional[str]                    # Entry ID
```

### SearchResults

```python
@dataclass
class SearchResults:
    query: str                           # Search query
    results: List[Dict[str, Any]]        # Search results
    total_count: int                     # Total matching results
    backend_used: str                    # Backend used for search
    search_time_ms: float                # Search duration in ms
    metadata: Optional[Dict[str, Any]]   # Additional metadata
```

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    analysis_type: str                   # Type of analysis performed
    target: str                          # Analysis target
    results: Dict[str, Any]              # Analysis results
    backend_used: str                    # Backend used
    analysis_time_ms: float              # Analysis duration in ms
    metadata: Optional[Dict[str, Any]]   # Additional metadata
```

### GraphStatistics

```python
@dataclass
class GraphStatistics:
    node_count: int                      # Total nodes
    edge_count: int                      # Total edges
    backend: str                         # Backend name
    metadata: Dict[str, Any]             # Backend-specific metadata
    timestamp: str                       # ISO timestamp
```

---

## Error Handling

### Exception Hierarchy

```
KnowledgeGraphError (base)
├── BackendUnavailableError
├── ConnectionError
├── ValueError
└── NotImplementedError
```

### Error Response Format

```json
{
  "error": {
    "type": "BackendUnavailableError",
    "message": "No backend available for operation",
    "details": {
      "operation": "add_knowledge",
      "attempted_backends": ["neo4j", "qdrant"],
      "reason": "All backends unhealthy"
    }
  }
}
```

---

## Configuration

### YAML Configuration Example

```yaml
# knowledge_graph_config.yaml
backends:
  neo4j:
    enabled: true
    uri: bolt://localhost:7687
    user: neo4j
    password: ${NEO4J_PASSWORD}  # Environment variable
    database: neo4j

  qdrant:
    enabled: true
    host: localhost
    port: 6333
    collection: knowledge_graph
    api_key: ${QDRANT_API_KEY}  # Optional
    vector_size: 1536

  mongodb:
    enabled: true
    uri: mongodb://localhost:27017
    database: knowledge_graph
    collection: knowledge

  karateclub:
    enabled: true
    embedding_dim: 128
    random_state: 42

  memory:
    enabled: true  # Always available as fallback

# Fallback chain (tried in order)
fallback_chain:
  - neo4j
  - qdrant
  - mongodb
  - memory

# Operation-specific backend preferences
operations:
  add_knowledge: [neo4j, mongodb]
  search: [qdrant, neo4j, mongodb]
  analyze: [karateclub, neo4j]
  visualize: [neo4j, karateclub]

# Performance settings
performance:
  connection_timeout_ms: 5000
  request_timeout_ms: 30000
  max_retries: 3
  retry_delay_ms: 1000

# Logging
logging:
  level: INFO
  format: json
```

### Environment Variables

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=optional_api_key

# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=knowledge_graph
MONGODB_COLLECTION=knowledge
```

---

## Examples

### Complete Workflow Example

```python
import asyncio
from knowledge_engine.core import UnifiedKnowledgeGraph

async def main():
    # Initialize
    kg = UnifiedKnowledgeGraph("config.yaml")
    await kg.connect_all()

    try:
        # 1. Add knowledge from multiple sources
        docs = [
            ("doc1", "Neo4j is a graph database", {"type": "technical"}),
            ("doc2", "Qdrant provides vector search", {"type": "technical"}),
            ("doc3", "MongoDB stores documents", {"type": "technical"}),
        ]

        for source, content, metadata in docs:
            await kg.add_knowledge(source, content, metadata)

        # 2. Search for relevant knowledge
        results = await kg.search("database")
        print(f"Found {results.total_count} results")

        # 3. Analyze knowledge distribution
        analysis = await kg.analyze("source_distribution")
        print(f"Sources: {analysis.results}")

        # 4. Generate visualization
        html = await kg.visualize("html")
        with open("graph.html", "w") as f:
            f.write(html)

        # 5. Get comprehensive statistics
        stats = await kg.get_graph_stats()
        print(f"Statistics: {stats}")

    finally:
        await kg.disconnect_all()

asyncio.run(main())
```

### FastAPI Integration Example

```python
from fastapi import FastAPI, HTTPException
from knowledge_engine.core import UnifiedKnowledgeGraph

app = FastAPI()
kg = UnifiedKnowledgeGraph()

@app.on_event("startup")
async def startup():
    await kg.connect_all()

@app.on_event("shutdown")
async def shutdown():
    await kg.disconnect_all()

@app.post("/api/kg/knowledge")
async def add_knowledge(request: dict):
    try:
        entry_id = await kg.add_knowledge(
            source=request["source"],
            content=request["content"],
            metadata=request.get("metadata")
        )
        return {"success": True, "entry_id": entry_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kg/search")
async def search(query: str, limit: int = 10):
    try:
        results = await kg.search(query, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Performance Considerations

1. **Connection Pooling**: All backends maintain connection pools
2. **Batch Operations**: Use batch methods for bulk operations
3. **Caching**: Consider caching frequently accessed knowledge
4. **Monitoring**: Track performance metrics for optimization
5. **Circuit Breaking**: Unhealthy backends are automatically skipped

---

## Security Considerations

1. **Authentication**: Configure authentication for all backends
2. **Authorization**: Implement access control at API layer
3. **Input Validation**: Validate all inputs before processing
4. **SQL/NoSQL Injection**: Use parameterized queries
5. **Rate Limiting**: Implement rate limiting on API endpoints

---

## Monitoring & Observability

### Metrics to Track

- Request latency per backend
- Request success/failure rates
- Knowledge growth over time
- Backend health status
- Cache hit/miss ratios

### Logging Format

```json
{
  "timestamp": "2025-01-07T12:00:00Z",
  "level": "INFO",
  "correlation_id": "uuid",
  "operation": "add_knowledge",
  "backend_used": "neo4j",
  "duration_ms": 45.2,
  "success": true
}
```

---

## Testing

### Unit Tests

```bash
pytest knowledge_engine/core/test_unified_kg.py -v
```

### Integration Tests

```bash
pytest knowledge_engine/core/test_integrations.py -v --integration
```

### Load Tests

```bash
locust -f knowledge_engine/core/load_tests.py
```

---

## Future Enhancements

1. **GraphQL API**: Alternative to REST
2. **Real-time Updates**: WebSocket support
3. **Advanced Analytics**: More analysis types
4. **Multi-tenancy**: Tenant isolation
5. **Export/Import**: Backup and restore functionality
6. **Versioning**: Track knowledge changes over time
7. **Access Control**: Fine-grained permissions
8. **Audit Logging**: Track all operations
9. **Caching Layer**: Redis integration
10. **Distributed Processing**: Horizontally scalable architecture
