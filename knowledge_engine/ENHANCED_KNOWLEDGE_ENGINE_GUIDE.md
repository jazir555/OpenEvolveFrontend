# Enhanced Knowledge Engine Guide

## Overview

The Enhanced Knowledge Engine is a comprehensive upgrade to the existing knowledge management system, providing advanced capabilities for knowledge storage, retrieval, analysis, and continuous improvement.

## Key Features

### 1. Multi-Modal Knowledge Storage
- **Text**: Natural language documents, notes, articles
- **Code**: Source code snippets, functions, classes
- **Structured**: JSON, YAML, database records
- **Embeddings**: Vector representations for semantic search
- **Extensible**: Easy to add new knowledge types

### 2. Semantic Search with Embeddings
- **Vector Similarity**: Find semantically similar content
- **Hybrid Search**: Combine keyword and semantic search
- **Embedding Service**: Pluggable embedding generation
- **Similarity Metrics**: Cosine similarity, Euclidean distance
- **Approximate Nearest Neighbors**: Fast similarity search

### 3. Knowledge Graph Navigation
- **Graph Structure**: Nodes (items) and edges (relations)
- **Relation Types**: Similar, depends_on, part_of, derived_from, etc.
- **Graph Traversal**: BFS, DFS, path finding
- **Centrality Analysis**: Identify important nodes
- **Connected Components**: Cluster analysis

### 4. Smart Caching
- **LRU Eviction**: Least recently used items removed first
- **TTL Support**: Time-based expiration
- **Predictive Prefetching**: Anticipate user needs
- **Cache Statistics**: Hit rates, miss rates, evictions
- **Async Operations**: Non-blocking cache access

### 5. Active Learning
- **Feedback Collection**: User ratings and comments
- **Quality Scoring**: Aggregate feedback into quality metrics
- **Trend Analysis**: Track quality over time
- **Improvement Recommendations**: Suggest areas for enhancement
- **Automatic Updates**: Refine knowledge based on feedback

### 6. Advanced Analytics
- **Trend Analysis**: Identify patterns in knowledge usage
- **Quality Metrics**: Completeness, consistency, freshness
- **Usage Analytics**: Popular items, search patterns
- **Anomaly Detection**: Identify outliers and issues
- **Predictive Insights**: Forecast future trends

### 7. Real-Time Event Streaming
- **Event Types**: Created, updated, deleted
- **Event Handlers**: Custom callbacks for events
- **Async Processing**: Non-blocking event handling
- **Event Queue**: Reliable event delivery

### 8. Persistence & Recovery
- **JSON Storage**: Human-readable backup format
- **Automatic Saving**: Periodic persistence
- **Recovery**: Load from backup on startup
- **Version Migration**: Handle format changes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Enhanced Knowledge Engine                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Semantic   │  │   Knowledge  │  │    Smart     │      │
│  │    Search    │  │    Graph     │  │    Cache     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Active    │  │   Analytics  │  │   Embedding  │      │
│  │   Learning   │  │    Engine    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                      Core Storage                            │
│         (Items, Relations, Metadata, Embeddings)            │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install required dependencies
pip install numpy asyncio

# Optional: For advanced embedding models
pip install sentence-transformers

# Optional: For vector database
pip install qdrant-client
```

## Quick Start

### Basic Usage

```python
import asyncio
from knowledge_engine.enhanced_knowledge_engine import create_knowledge_engine
from knowledge_engine.enhanced_knowledge_core import KnowledgeType

async def main():
    # Create and initialize engine
    engine = await create_knowledge_engine(
        storage_path="./knowledge_storage",
        embedding_model="default",
        cache_size=10000
    )
    
    # Add knowledge
    item = await engine.add_knowledge(
        content="Python is a programming language",
        knowledge_type=KnowledgeType.TEXT,
        tags={"python", "programming"},
        source="documentation"
    )
    
    print(f"Created item: {item.id}")
    
    # Search
    results = await engine.search(
        query="programming language",
        search_mode="hybrid",
        max_results=5
    )
    
    for result in results:
        print(f"{result.relevance_score:.2f}: {result.item.content}")
    
    # Shutdown
    await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Knowledge Graph Operations

```python
# Create items
parent = await engine.add_knowledge(
    content="Machine Learning",
    knowledge_type=KnowledgeType.TEXT
)

child = await engine.add_knowledge(
    content="Neural Networks",
    knowledge_type=KnowledgeType.TEXT
)

# Create relation
relation = await engine.create_relation(
    parent.id,
    child.id,
    RelationType.PART_OF,
    weight=0.9
)

# Find related items
related = await engine.find_related(parent.id)
for item, rel in related:
    print(f"Related: {item.content} ({rel.relation_type.value})")

# Find paths
paths = await engine.find_path(parent.id, child.id)
for path in paths:
    print(" -> ".join(item.content for item in path))
```

### Active Learning & Feedback

```python
# Add knowledge
item = await engine.add_knowledge(
    content="Tutorial on async programming",
    knowledge_type=KnowledgeType.TEXT
)

# Record user feedback
await engine.record_feedback(
    item_id=item.id,
    feedback_type="positive",
    feedback_score=0.9,
    user_id="user-123"
)

# Get quality metrics
quality = await engine.get_item_quality(item.id)
print(f"Average score: {quality['average_score']:.2f}")
print(f"Feedback count: {quality['feedback_count']}")

# Get improvement recommendations
recommendations = await engine.get_learning_recommendations()
for rec in recommendations:
    print(f"{rec['type']}: {rec['message']}")
```

### Analytics & Reporting

```python
from knowledge_engine.knowledge_analytics import KnowledgeAnalyticsEngine

# Create analytics engine
analytics = KnowledgeAnalyticsEngine(knowledge_engine=engine)

# Record metrics
analytics.record_metric("daily_searches", 150)
analytics.record_metric("new_items", 25)

# Generate comprehensive report
items = list(engine._items.values())
report = analytics.generate_comprehensive_report(items)

# Print report sections
print("Quality Summary:")
print(f"  Average score: {report['quality']['average_overall_score']:.2f}")
print(f"  Distribution: {report['quality']['quality_distribution']}")

print("\nTrends:")
for metric, trend in report['trends'].items():
    print(f"  {metric}: {trend.direction} ({trend.change_percent:+.1f}%)")

print("\nInsights:")
for insight in report['insights']:
    print(f"  [{insight['severity']}] {insight['message']}")

# Dashboard data
dashboard = analytics.get_dashboard_data(items)
print("\nDashboard:")
print(f"  Total items: {dashboard['summary']['total_items']}")
print(f"  Open issues: {dashboard['summary']['open_issues']}")
```

## Advanced Features

### Custom Embedding Models

```python
from knowledge_engine.enhanced_knowledge_core import EmbeddingService

class CustomEmbeddingService(EmbeddingService):
    async def _embed_text(self, text: str):
        # Use your own embedding model
        # e.g., OpenAI, HuggingFace, custom model
        return your_embedding_model.encode(text)

# Use custom service
engine = EnhancedKnowledgeEngine()
engine.embedding_service = CustomEmbeddingService()
```

### Event Handling

```python
# Define event handler
def on_knowledge_created(event):
    print(f"New knowledge created: {event.item_id}")
    # Send notification, update index, etc.

# Register handler
engine.add_event_handler(on_knowledge_created)

# Async handler
async def async_handler(event):
    await send_webhook(event)

engine.add_event_handler(async_handler)
```

### Batch Operations

```python
# Add multiple items
items_to_add = [
    {"content": "Item 1", "type": KnowledgeType.TEXT},
    {"content": "Item 2", "type": KnowledgeType.TEXT},
    {"content": "Item 3", "type": KnowledgeType.CODE},
]

results = await asyncio.gather(*[
    engine.add_knowledge(**item) for item in items_to_add
])
```

### Custom Search Filters

```python
results = await engine.search(
    query="machine learning",
    filters={
        "source": "documentation",
        "min_confidence": 0.8
    },
    knowledge_types=["text", "code"],
    tags={"tutorial", "advanced"},
    max_results=20
)
```

## Configuration Options

### Engine Configuration

```python
engine = EnhancedKnowledgeEngine(
    storage_path="./storage",          # Persistent storage directory
    embedding_model="default",          # Embedding model name
    embedding_dimensions=1536,          # Embedding vector size
    cache_size=10000,                   # Maximum cache entries
    enable_graph=True,                  # Enable knowledge graph
    enable_learning=True                # Enable active learning
)
```

### Cache Configuration

```python
cache = SmartCacheManager(
    max_size=10000,                     # Maximum entries
    default_ttl=3600                    # Default TTL in seconds
)

# Set with custom TTL
await cache.set("key", value, ttl=300)  # 5 minutes
```

## Performance Tuning

### 1. Embedding Caching
- Embeddings are automatically cached
- Adjust `EmbeddingService._cache_size_limit` as needed
- Clear cache with `embedding_service.clear_cache()`

### 2. Search Index Optimization
- Use approximate nearest neighbor libraries (e.g., FAISS, Annoy)
- Index items in batches for large datasets
- Consider partitioning by knowledge type

### 3. Cache Tuning
- Monitor hit rates with `cache.get_stats()`
- Adjust TTL based on data freshness requirements
- Use larger cache for read-heavy workloads

### 4. Async Operations
- All I/O operations are async
- Use `asyncio.gather()` for parallel operations
- Consider connection pooling for database backends

## Best Practices

### 1. Knowledge Organization
- Use consistent tagging schemes
- Set appropriate knowledge types
- Include source information
- Set expiration dates for time-sensitive content

### 2. Search Optimization
- Use hybrid search for best results
- Set appropriate confidence thresholds
- Filter by knowledge type when possible
- Cache frequent queries

### 3. Graph Design
- Use appropriate relation types
- Set meaningful weights
- Avoid overly dense connections
- Document graph schema

### 4. Quality Assurance
- Monitor quality metrics regularly
- Act on improvement recommendations
- Collect user feedback
- Review and update stale content

### 5. Monitoring
- Track cache hit rates
- Monitor search performance
- Review analytics regularly
- Set up alerts for anomalies

## API Reference

### EnhancedKnowledgeEngine

#### Methods
- `add_knowledge()` - Add new knowledge item
- `get_knowledge()` - Retrieve item by ID
- `update_knowledge()` - Update existing item
- `delete_knowledge()` - Remove item
- `search()` - Search with multiple modes
- `semantic_search()` - Semantic similarity search
- `create_relation()` - Create graph relation
- `find_related()` - Find related items
- `record_feedback()` - Record user feedback
- `get_stats()` - Get engine statistics
- `get_health_check()` - Health status

### KnowledgeItem

#### Properties
- `id` - Unique identifier
- `content` - Item content
- `knowledge_type` - Type enum
- `embedding` - Vector embedding
- `metadata` - Custom metadata
- `tags` - Tag set
- `confidence` - Confidence score
- `version` - Version number

#### Methods
- `update_content()` - Update and version
- `add_tag()` / `remove_tag()` - Tag management
- `is_expired()` - Check expiration

### SearchQuery

#### Properties
- `text` - Query text
- `embedding` - Query embedding
- `filters` - Filter dict
- `knowledge_types` - Type filter
- `tags` - Tag filter
- `search_mode` - Search strategy

## Troubleshooting

### Common Issues

1. **Low Search Quality**
   - Check embedding generation
   - Verify knowledge types are correct
   - Increase result limit
   - Try different search modes

2. **Memory Issues**
   - Reduce cache size
   - Clear embedding cache
   - Use persistent storage
   - Implement pagination

3. **Slow Performance**
   - Enable caching
   - Use approximate nearest neighbors
   - Optimize graph queries
   - Profile async operations

4. **Data Loss**
   - Enable persistence
   - Set up regular backups
   - Verify storage permissions
   - Check disk space

## Migration Guide

### From Legacy Knowledge Base

```python
# Load legacy data
from knowledge_base import KnowledgeBase
legacy_kb = KnowledgeBase("knowledge_base.json")

# Create enhanced engine
engine = await create_knowledge_engine()

# Migrate artifacts
for artifact in legacy_kb.artifacts:
    await engine.add_knowledge(
        content=artifact.description,
        knowledge_type=KnowledgeType.TEXT,
        metadata=artifact.metadata,
        tags=set(artifact.tags)
    )

# Migrate relations
for experience in legacy_kb.experiences:
    # Create relations based on experience data
    pass

# Shutdown
await engine.shutdown()
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repo-url>
cd knowledge_engine

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest test_enhanced_knowledge_engine.py -v
```

### Adding New Knowledge Types

1. Add type to `KnowledgeType` enum
2. Update `EmbeddingService` for new type
3. Add validation in `KnowledgeValidator`
4. Update tests

### Adding New Relation Types

1. Add type to `RelationType` enum
2. Update graph visualization if applicable
3. Add traversal logic if needed
4. Document semantics

## License

Apache 2.0

## Support

For issues and feature requests, please use the GitHub issue tracker.
