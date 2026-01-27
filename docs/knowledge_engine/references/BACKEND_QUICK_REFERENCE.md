# Knowledge Engine Backends - Quick Reference

## Quick Start

```python
from knowledge_engine.core.backends.memory_backend import MemoryBackend
from knowledge_engine.core.backends.base import KnowledgeEntry

# Create backend
backend = MemoryBackend(config={})
await backend.connect()

# Add knowledge
entry = KnowledgeEntry(
    source="doc_1",
    content="AI is transforming healthcare",
    metadata={"category": "AI"}
)
entry_id = await backend.add_knowledge(entry)

# Search
results = await backend.search(query="healthcare", limit=10)

# Cleanup
await backend.disconnect()
```

---

## Backend Selection

| Backend | Best For | Setup Complexity | Performance |
|---------|----------|------------------|-------------|
| **Memory** | Testing, Dev | ⭐ | ⚡⚡⚡ |
| **Neo4j** | Knowledge Graphs | ⭐⭐⭐ | ⚡⚡ |
| **Qdrant** | Semantic Search | ⭐⭐ | ⚡⚡⚡ |
| **MongoDB** | Document Storage | ⭐⭐ | ⚡⚡ |
| **KarateClub** | Graph Analytics | ⭐ | ⚡⚡ |

---

## Configuration Examples

### Memory Backend
```python
backend = MemoryBackend(config={})
```

### Neo4j Backend
```python
backend = Neo4jBackend(config={
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'password',
    'database': 'neo4j'
})
```

### Qdrant Backend
```python
backend = QdrantBackend(config={
    'host': 'localhost',
    'port': 6333,
    'collection': 'knowledge_graph',
    'vector_size': 1536
})
```

### MongoDB Backend
```python
backend = MongoDBBackend(config={
    'uri': 'mongodb://localhost:27017',
    'database': 'knowledge_graph',
    'collection': 'knowledge'
})
```

### KarateClub Backend
```python
backend = KarateClubBackend(config={
    'embedding_dim': 128,
    'random_state': 42
})
```

---

## Common Operations

### Add Knowledge
```python
entry = KnowledgeEntry(
    source="document_id",
    content="Knowledge content here",
    metadata={"key": "value"},
    timestamp=datetime.utcnow().isoformat()
)
entry_id = await backend.add_knowledge(entry)
```

### Search
```python
results = await backend.search(
    query="search terms",
    filters={"source": "specific_source"},
    limit=10,
    offset=0
)
```

### Batch Add
```python
entries = [entry1, entry2, entry3]
ids = await backend.batch_add_knowledge(entries)
```

### Update
```python
success = await backend.update_knowledge(
    entry_id,
    {"content": "Updated content"}
)
```

### Delete
```python
success = await backend.delete_knowledge(entry_id)
```

### Get Statistics
```python
stats = await backend.get_statistics()
print(f"Nodes: {stats.node_count}, Edges: {stats.edge_count}")
```

### Analyze
```python
# Backend-specific analysis
result = await backend.analyze(
    analysis_type="connected_components",  # Varies by backend
    target="graph"
)
```

### Visualize
```python
# JSON export
json_data = await backend.visualize(output_format='json')

# HTML visualization
html = await backend.visualize(output_format='html')
```

---

## Backend-Specific Features

### Neo4j Features
```python
# Entity connections
analysis = await backend.analyze(analysis_type="entity_connections")

# Connected components
analysis = await backend.analyze(analysis_type="connected_components")

# Knowledge by source
analysis = await backend.analyze(analysis_type="knowledge_by_source")

# Point-in-time query
cypher = """
MATCH (k:Knowledge)
WHERE k.valid_from <= datetime('2024-01-01')
AND k.valid_to >= datetime('2024-01-01')
RETURN k
"""
```

### Qdrant Features
```python
# Semantic search (auto-embedding)
results = await backend.search(query="machine learning")

# With custom embedding
entry = KnowledgeEntry(
    source="doc_1",
    content="...",
    embedding=[0.1, 0.2, ...]  # Your vector
)

# Hybrid search
results = await backend.search(
    query="AI",
    filters={"source": "doc_1"}
)
```

### MongoDB Features
```python
# Source distribution
analysis = await backend.analyze(analysis_type="source_distribution")

# Tag distribution
analysis = await backend.analyze(analysis_type="tag_distribution")

# Temporal analysis
analysis = await backend.analyze(analysis_type="temporal_analysis")

# Content statistics
analysis = await backend.analyze(analysis_type="content_statistics")
```

### KarateClub Features
```python
# Community detection
analysis = await backend.analyze(analysis_type="community_detection")

# Node embeddings
analysis = await backend.analyze(analysis_type="node_embedding")

# Centrality measures
analysis = await backend.analyze(analysis_type="centrality")

# Role detection
analysis = await backend.analyze(analysis_type="role_detection")

# Graph statistics
analysis = await backend.analyze(analysis_type="graph_statistics")
```

---

## Context Manager Usage

```python
# Auto connect/disconnect
async with backend:
    await backend.add_knowledge(entry)
    results = await backend.search(query="test")

# Manual management
await backend.connect()
try:
    await backend.add_knowledge(entry)
finally:
    await backend.disconnect()
```

---

## Error Handling

```python
# Connection error
try:
    await backend.connect()
except ConnectionError as e:
    logger.error(f"Cannot connect: {e}")
    # Fallback to memory backend
    backend = MemoryBackend(config={})

# Operation error
try:
    await backend.add_knowledge(entry)
except ValueError as e:
    logger.error(f"Invalid entry: {e}")
except ConnectionError as e:
    logger.error(f"Backend unavailable: {e}")
    # Retry or fallback
```

---

## Health Checks

```python
# Check backend health
is_healthy = await backend.health_check()

# Periodic monitoring
async def monitor(backend):
    while True:
        if not await backend.health_check():
            logger.warning("Backend unhealthy!")
            # Attempt reconnect
        await asyncio.sleep(30)
```

---

## Multi-Backend Usage

```python
# Use multiple backends
neo4j = Neo4jBackend(config={...})
qdrant = QdrantBackend(config={...})

await neo4j.connect()
await qdrant.connect()

# Add to both
await neo4j.add_knowledge(entry)
await qdrant.add_knowledge(entry)

# Hybrid search
graph_results = await neo4j.search(query="AI")
vector_results = await qdrant.search(query="machine learning")

# Cleanup
await neo4j.disconnect()
await qdrant.disconnect()
```

---

## Backend Factory

```python
def get_backend(backend_type: str):
    """Factory function for backends."""

    configs = {
        "neo4j": {
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        },
        "qdrant": {
            'host': 'localhost',
            'port': 6333,
            'collection': 'knowledge_graph'
        },
        "mongodb": {
            'uri': 'mongodb://localhost:27017',
            'database': 'knowledge_graph'
        },
        "karateclub": {
            'embedding_dim': 128
        },
        "memory": {}
    }

    backends = {
        "neo4j": Neo4jBackend,
        "qdrant": QdrantBackend,
        "mongodb": MongoDBBackend,
        "karateclub": KarateClubBackend,
        "memory": MemoryBackend
    }

    backend_class = backends.get(backend_type, MemoryBackend)
    return backend_class(config=configs.get(backend_type, {}))

# Usage
backend = get_backend("neo4j")
await backend.connect()
```

---

## Testing

```python
# Memory backend (no dependencies)
backend = MemoryBackend(config={})
await backend.connect()

# Add test data
entry = KnowledgeEntry(source="test", content="test content")
entry_id = await backend.add_knowledge(entry)

# Verify
results = await backend.search(query="test")
assert results.total_count > 0

# Cleanup
await backend.clear_all()
await backend.disconnect()
```

---

## Performance Tips

1. **Use batch operations** for multiple entries
2. **Create indexes** in MongoDB for faster queries
3. **Use filters** to reduce result sets
4. **Limit results** to avoid large transfers
5. **Use context managers** for automatic cleanup
6. **Monitor health** with periodic checks
7. **Pool connections** in production
8. **Cache frequently accessed data**

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused | Check service is running, verify URI/port |
| Import error | Install required package (pip install ...) |
| Slow queries | Create indexes, use filters, limit results |
| Out of memory | Reduce batch size, increase system RAM |
| Collection doesn't exist | Backend auto-creates, or create manually |

---

## Environment Variables

```bash
# Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# Qdrant
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export QDRANT_COLLECTION="knowledge_graph"

# MongoDB
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="knowledge_graph"
export MONGODB_COLLECTION="knowledge"

# Backend Selection
export KNOWLEDGE_BACKEND="neo4j"
```

---

## Docker Compose

```yaml
services:
  neo4j:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7687:7687"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
```

---

## Further Reading

- Full Guide: `BACKEND_GUIDE.md`
- API Reference: `knowledge_engine/core/backends/`
- Tests: `knowledge_engine/tests/test_backends.py`
