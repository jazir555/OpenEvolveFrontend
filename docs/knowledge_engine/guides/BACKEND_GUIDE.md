# Knowledge Engine Storage Backends - Complete Guide

## Overview

The Knowledge Engine supports multiple storage backends, each optimized for different use cases. All backends implement a unified interface defined in `base.py`, following CLAUDE.md principles:

- **Runtime Truth**: Verify connections before use
- **Configuration Explicitness**: All settings via config/environment variables
- **UTC**: All timestamps in UTC
- **Idempotency**: Operations are safe to replay
- **Circuit Breakers**: Graceful failure handling
- **Structured Logging**: JSON logs with correlation IDs

---

## Backend Types

### 1. Memory Backend

**Purpose**: Fast, in-memory storage for testing and development

**Pros**:
- Fastest performance
- No external dependencies
- Perfect for unit tests
- Instant startup

**Cons**:
- Data lost on restart
- Limited by RAM
- Not production-ready for persistent storage

**Configuration**:
```python
from knowledge_engine.core.backends.memory_backend import MemoryBackend

backend = MemoryBackend(config={})  # No config required
await backend.connect()
```

**Best For**:
- Testing
- Development
- Caching
- Temporary data processing

---

### 2. Neo4j Backend

**Purpose**: Graph database for complex relationship queries

**Pros**:
- Native graph storage
- Cypher query language
- Temporal knowledge tracking
- Bi-temporal queries (valid time + transaction time)
- Point-in-time graph reconstruction
- Rich relationship types

**Cons**:
- Requires Neo4j server
- More complex setup
- Higher resource usage

**Configuration**:
```python
from knowledge_engine.core.backends.neo4j_backend import Neo4jBackend

backend = Neo4jBackend(config={
    'uri': 'bolt://localhost:7687',
    'user': 'neo4j',
    'password': 'your_password',
    'database': 'neo4j'  # Optional, defaults to 'neo4j'
})
await backend.connect()
```

**Environment Variables**:
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export NEO4J_DATABASE="neo4j"
```

**Features**:
- Automatic entity extraction
- Relationship tracking (MENTIONS, RELATED_TO, etc.)
- Graph analytics (centrality, communities, paths)
- Point-in-time queries
- Temporal evolution tracking

**Best For**:
- Knowledge graphs with complex relationships
- Temporal knowledge tracking
- Graph analytics
- Relationship-heavy data
- Bi-temporal queries

**Example Queries**:
```python
# Add knowledge
entry = KnowledgeEntry(
    source="research_paper_1",
    content="Deep learning uses neural networks for pattern recognition.",
    metadata={"year": 2023, "authors": ["Smith", "Jones"]}
)
entry_id = await backend.add_knowledge(entry)

# Search
results = await backend.search(
    query="neural networks",
    filters={"source": "research_paper_1"},
    limit=10
)

# Analyze connections
analysis = await backend.analyze(analysis_type="entity_connections")
print(analysis.results["top_entities"])

# Get statistics
stats = await backend.get_statistics()
print(f"Nodes: {stats.node_count}, Edges: {stats.edge_count}")
```

---

### 3. Qdrant Backend

**Purpose**: Vector similarity search for semantic queries

**Pros**:
- Fast vector similarity search
- Semantic understanding
- Scalable to millions of vectors
- Built-in filtering
- Real-time updates

**Cons**:
- Requires Qdrant server
- Needs embeddings (OpenAI, SentenceTransformers, etc.)
- Less suited for exact matches

**Configuration**:
```python
from knowledge_engine.core.backends.qdrant_backend import QdrantBackend

backend = QdrantBackend(config={
    'host': 'localhost',
    'port': 6333,
    'collection': 'knowledge_graph',
    'vector_size': 1536,  # OpenAI default
    'api_key': None  # Optional, for authenticated Qdrant
})
await backend.connect()
```

**Environment Variables**:
```bash
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
export QDRANT_COLLECTION="knowledge_graph"
export QDRANT_API_KEY=""  # Optional
```

**Features**:
- Automatic embedding generation (or use your own)
- Cosine similarity search
- Hybrid search (semantic + filters)
- Batch operations
- Collection management

**Best For**:
- Semantic search
- Document similarity
- Recommendation systems
- Duplicate detection
- Content-based retrieval

**Example Usage**:
```python
# With embedding
entry = KnowledgeEntry(
    source="doc_1",
    content="Machine learning algorithms learn patterns from data.",
    embedding=[0.1, 0.2, ...]  # Your embedding vector
)
entry_id = await backend.add_knowledge(entry)

# Semantic search (embedding auto-generated if not provided)
results = await backend.search(
    query="pattern recognition",
    limit=10
)

# Hybrid search
results = await backend.search(
    query="AI",
    filters={"source": "doc_1"},
    limit=5
)

# Batch add
entries = [entry1, entry2, entry3]
ids = await backend.batch_add_knowledge(entries)
```

---

### 4. MongoDB Backend

**Purpose**: Flexible document storage with powerful aggregations

**Pros**:
- Flexible schema
- Rich aggregation pipeline
- Full-text search
- Easy horizontal scaling
- ACID transactions

**Cons**:
- Requires MongoDB server
- Less optimized for graph queries
- Manual relationship management

**Configuration**:
```python
from knowledge_engine.core.backends.mongodb_backend import MongoDBBackend

backend = MongoDBBackend(config={
    'uri': 'mongodb://localhost:27017',
    'database': 'knowledge_graph',
    'collection': 'knowledge'
})
await backend.connect()
```

**Environment Variables**:
```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DATABASE="knowledge_graph"
export MONGODB_COLLECTION="knowledge"
```

**Features**:
- Automatic indexing (text, fields)
- Aggregation pipelines
- Temporal analysis
- Source distribution
- Tag-based filtering
- Content statistics

**Best For**:
- Document storage
- Flexible schemas
- Aggregation queries
- Full-text search
- Analytics and reporting

**Example Usage**:
```python
# Add document
entry = KnowledgeEntry(
    source="web_page_1",
    content="AI is transforming industries worldwide.",
    metadata={
        "tags": ["AI", "technology"],
        "category": "tech",
        "language": "en"
    }
)
entry_id = await backend.add_knowledge(entry)

# Full-text search
results = await backend.search(
    query="transforming industries",
    filters={"tags": ["AI"]},
    limit=10
)

# Aggregation analysis
analysis = await backend.analyze(analysis_type="source_distribution")
print(analysis.results["by_source"])

# Temporal analysis
analysis = await backend.analyze(analysis_type="temporal_analysis")
print(analysis.results["timeline"])

# Update knowledge
await backend.update_knowledge(
    entry_id,
    {"content": "Updated content"}
)

# Delete
await backend.delete_knowledge(entry_id)
```

---

### 5. KarateClub Backend

**Purpose**: Graph ML and analytics using NetworkX + KarateClub

**Pros**:
- Advanced graph algorithms
- Community detection
- Node embeddings
- Centrality measures
- Role detection
- No external database required

**Cons**:
- In-memory only
- Not for persistent storage
- Computationally intensive for large graphs

**Configuration**:
```python
from knowledge_engine.core.backends.karateclub_backend import KarateClubBackend

backend = KarateClubBackend(config={
    'embedding_dim': 128,
    'random_state': 42
})
await backend.connect()
```

**Features**:
- Community detection (Label Propagation, Louvain)
- Node embeddings (DeepWalk, Node2Vec)
- Centrality measures (PageRank, Betweenness, Degree)
- Role detection (Role2Vec)
- Graph statistics
- D3.js visualizations

**Best For**:
- Graph analytics
- Community detection
- Feature engineering for ML
- Social network analysis
- Recommendation systems
- Research and prototyping

**Example Usage**:
```python
# Build graph from knowledge
for entry in knowledge_entries:
    await backend.add_knowledge(entry)

# Community detection
communities = await backend.analyze(analysis_type="community_detection")
print(f"Found {communities.results['num_communities']} communities")

# Node embeddings
embeddings = await backend.analyze(analysis_type="node_embedding")
print(f"Generated {embeddings.results['num_nodes']} embeddings")

# Centrality analysis
centrality = await backend.analyze(analysis_type="centrality")
print("Top nodes by PageRank:")
for node in centrality.results["top_pagerank"][:5]:
    print(f"  {node['node']}: {node['score']:.4f}")

# Visualize
html = await backend.visualize(output_format='html')
with open("graph_viz.html", "w") as f:
    f.write(html)
```

---

## Unified Backend Interface

All backends implement these methods:

### Connection Management

```python
# Connect to backend
await backend.connect()

# Check health
is_healthy = await backend.health_check()

# Disconnect
await backend.disconnect()

# Context manager
async with backend:
    # Backend automatically connects/disconnects
    await backend.add_knowledge(entry)
```

### Knowledge Operations

```python
# Add single entry
entry_id = await backend.add_knowledge(entry)

# Batch add (more efficient)
ids = await backend.batch_add_knowledge([entry1, entry2, entry3])

# Update entry
success = await backend.update_knowledge(entry_id, {"content": "New content"})

# Delete entry
success = await backend.delete_knowledge(entry_id)

# Clear all (destructive!)
count = await backend.clear_all()
```

### Search Operations

```python
# Basic search
results = await backend.search(
    query="machine learning",
    limit=10,
    offset=0
)

# Search with filters
results = await backend.search(
    query="AI",
    filters={
        "source": "research_paper_1",
        "date_after": "2023-01-01",
        "tags": ["technology"]
    },
    limit=20
)

# Batch search
results_list = await backend.batch_search(
    queries=["AI", "machine learning", "neural networks"],
    limit=10
)
```

### Analysis Operations

```python
# Get statistics
stats = await backend.get_statistics()
print(f"Nodes: {stats.node_count}, Edges: {stats.edge_count}")

# Analyze (backend-specific)
analysis = await backend.analyze(
    analysis_type="connected_components",  # Varies by backend
    target="graph"
)
```

### Visualization

```python
# Export as JSON
json_data = await backend.visualize(output_format='json')

# Generate HTML visualization
html = await backend.visualize(output_format='html')

# Other formats (backend-specific)
dot_data = await backend.visualize(output_format='dot')
```

---

## Backend Selection Guide

### Use Case Matrix

| Use Case | Recommended Backend | Alternative |
|----------|-------------------|-------------|
| **Complex Relationships** | Neo4j | KarateClub |
| **Semantic Search** | Qdrant | MongoDB |
| **Document Storage** | MongoDB | Neo4j |
| **Graph Analytics** | KarateClub | Neo4j |
| **Testing/Dev** | Memory | Any |
| **Production Knowledge Graph** | Neo4j | MongoDB |
| **Real-time Vector Search** | Qdrant | - |
| **Temporal Tracking** | Neo4j | MongoDB |
| **Flexible Schema** | MongoDB | - |
| **Social Network Analysis** | KarateClub | Neo4j |

### Performance Characteristics

| Backend | Write Speed | Read Speed | Memory Usage | Disk Usage |
|---------|-------------|------------|--------------|------------|
| Memory | ⚡⚡⚡ | ⚡⚡⚡ | High | None |
| Neo4j | ⚡⚡ | ⚡⚡ | Medium | Medium |
| Qdrant | ⚡⚡ | ⚡⚡⚡ | Medium | Low |
| MongoDB | ⚡⚡ | ⚡⚡ | Medium | Medium |
| KarateClub | ⚡⚡ | ⚡ | High | None |

---

## Multi-Backend Architecture

### Using Multiple Backends Simultaneously

```python
from knowledge_engine.core.backends.neo4j_backend import Neo4jBackend
from knowledge_engine.core.backends.qdrant_backend import QdrantBackend

# Initialize multiple backends
neo4j = Neo4jBackend(config={...})
qdrant = QdrantBackend(config={...})

await neo4j.connect()
await qdrant.connect()

# Use Neo4j for relationships
await neo4j.add_knowledge(entry)

# Use Qdrant for semantic search
await qdrant.add_knowledge(entry)

# Hybrid search
graph_results = await neo4j.search(query="AI")
semantic_results = await qdrant.search(query="machine learning")

# Combine results
combined = merge_results(graph_results, semantic_results)

await neo4j.disconnect()
await qdrant.disconnect()
```

### Backend Switching

```python
def get_backend(backend_type: str):
    """Factory function to create backend by type."""

    if backend_type == "neo4j":
        return Neo4jBackend(config={...})
    elif backend_type == "qdrant":
        return QdrantBackend(config={...})
    elif backend_type == "mongodb":
        return MongoDBBackend(config={...})
    elif backend_type == "karateclub":
        return KarateClubBackend(config={...})
    else:  # Default to memory
        return MemoryBackend(config={})

# Use based on configuration
backend_type = os.getenv("KNOWLEDGE_BACKEND", "memory")
backend = get_backend(backend_type)
```

### Graceful Fallback

```python
async def get_backend_with_fallback(preference: str):
    """Try preferred backend, fallback to alternatives."""

    backends_to_try = [preference, "memory"]

    for backend_type in backends_to_try:
        try:
            backend = get_backend(backend_type)
            await backend.connect()

            if await backend.health_check():
                logger.info(f"Using {backend_type} backend")
                return backend

        except Exception as e:
            logger.warning(f"{backend_type} failed: {e}, trying next...")
            continue

    raise RuntimeError("No backend available")
```

---

## Testing Backends

### Unit Tests

```bash
# Test all backends (requires services running)
pytest knowledge_engine/tests/test_backends.py -v

# Test only memory backend (no services needed)
pytest knowledge_engine/tests/test_backends.py::TestMemoryBackend -v

# Test specific backend
pytest knowledge_engine/tests/test_backends.py::TestNeo4jBackend -v
```

### Integration Tests

```bash
# Start required services
docker-compose up -d neo4j qdrant mongodb

# Run integration tests
pytest knowledge_engine/tests/test_backends.py -v -m integration

# Cleanup
docker-compose down
```

### Performance Tests

```bash
# Run performance benchmarks
pytest knowledge_engine/tests/test_backends.py::TestBackendPerformance -v
```

---

## Configuration Examples

### Docker Compose Setup

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  neo4j_data:
  qdrant_data:
  mongodb_data:
```

### Environment Configuration

```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=knowledge_graph

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=knowledge_graph
MONGODB_COLLECTION=knowledge

# Backend selection
KNOWLEDGE_BACKEND=neo4j  # or qdrant, mongodb, karateclub, memory
```

---

## Best Practices

### 1. Connection Management

```python
# GOOD: Use context managers
async with backend:
    await backend.add_knowledge(entry)

# GOOD: Explicit connect/disconnect
await backend.connect()
try:
    await backend.add_knowledge(entry)
finally:
    await backend.disconnect()

# BAD: Forgetting to disconnect
await backend.connect()
await backend.add_knowledge(entry)
# Missing disconnect!
```

### 2. Error Handling

```python
# GOOD: Handle connection errors
try:
    await backend.connect()
except ConnectionError as e:
    logger.error(f"Backend unavailable: {e}")
    # Fallback to alternative backend
    backend = MemoryBackend(config={})
    await backend.connect()

# GOOD: Handle operation errors
try:
    await backend.add_knowledge(entry)
except ValueError as e:
    logger.error(f"Invalid entry: {e}")
except ConnectionError as e:
    logger.error(f"Backend disconnected: {e}")
    # Retry or fallback
```

### 3. Batch Operations

```python
# GOOD: Use batch for multiple entries
ids = await backend.batch_add_knowledge(entries)

# BAD: Sequential adds
for entry in entries:
    await backend.add_knowledge(entry)  # Slower!
```

### 4. Index Management

```python
# MongoDB: Create indexes for better query performance
await backend.collection.create_index([("content", "text")])
await backend.collection.create_index([("timestamp", -1)])

# Neo4j: Create constraints for uniqueness
await backend.create_constraint("Knowledge", "id")
```

### 5. Health Checks

```python
# Periodic health checks
async def monitor_backend(backend):
    while True:
        is_healthy = await backend.health_check()
        if not is_healthy:
            logger.warning("Backend unhealthy, attempting reconnect...")
            try:
                await backend.connect()
            except Exception as e:
                logger.error(f"Reconnect failed: {e}")
        await asyncio.sleep(30)
```

---

## Troubleshooting

### Neo4j Issues

**Problem**: Connection refused
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Check logs
docker logs neo4j

# Test connection
cypher-shell -a bolt://localhost:7687 -u neo4j -p password
```

**Problem**: Memory issues
```python
# Increase memory in neo4j.conf
dbms.memory.heap.initial_size=2g
dbms.memory.heap.max_size=4g
```

### Qdrant Issues

**Problem**: Collection doesn't exist
```python
# Backend auto-creates collection, but you can manually create
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name="knowledge_graph",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

### MongoDB Issues

**Problem**: Slow queries
```python
# Check query execution plan
cursor = db.collection.find({"content": {"$regex": "AI"}}).explain()
print(cursor)

# Create appropriate indexes
await backend.collection.create_index([("content", "text")])
```

---

## Production Deployment

### Recommendations

1. **Neo4j Production**:
   - Use clustered setup for HA
   - Enable backup/restore
   - Monitor memory usage
   - Use connection pooling

2. **Qdrant Production**:
   - Enable replication
   - Use SSD storage
   - Monitor vector count
   - Set up snapshots

3. **MongoDB Production**:
   - Use replica sets
   - Enable sharding for large datasets
   - Create appropriate indexes
   - Monitor disk usage

4. **Monitoring**:
   - Track backend health
   - Monitor query latency
   - Set up alerts for failures
   - Log performance metrics

### Security

```python
# Use environment variables for credentials
import os

config = {
    'uri': os.getenv('NEO4J_URI'),
    'user': os.getenv('NEO4J_USER'),
    'password': os.getenv('NEO4J_PASSWORD')
}

# Enable SSL/TLS
config['uri'] = 'bolt+ssc://localhost:7687'  # Secure connection

# Use API keys for Qdrant
config['api_key'] = os.getenv('QDRANT_API_KEY')
```

---

## API Reference

See individual backend files for detailed API documentation:
- `base.py` - Base interface
- `neo4j_backend.py` - Neo4j implementation
- `qdrant_backend.py` - Qdrant implementation
- `mongodb_backend.py` - MongoDB implementation
- `karateclub_backend.py` - KarateClub implementation
- `memory_backend.py` - Memory implementation

---

## Contributing

When adding a new backend:

1. Inherit from `KnowledgeGraphBackend`
2. Implement all required methods
3. Follow CLAUDE.md principles
4. Add comprehensive tests
5. Update this documentation
6. Use structured logging
7. Implement health checks
8. Handle errors gracefully

---

## License

See LICENSE file for details.
