# Knowledge Artifacts - Usage Guide

Complete guide for using KnowledgeExtractor, KnowledgeArtifact, KnowledgeStorage, and KnowledgeRetriever in the OpenEvolve Knowledge Engine.

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

---

## Overview

The Knowledge Artifacts system provides a comprehensive solution for:

- **Extracting** knowledge from workflow executions
- **Storing** artifacts in multiple database backends
- **Retrieving** knowledge through various search strategies
- **Analyzing** quality metrics and trends

### Key Features

- **Multi-modal Extraction**: Extract patterns, critiques, team performance, and gauntlet effectiveness
- **Quality Scoring**: Automatic quality assessment based on multiple factors
- **Pattern Recognition**: Identify and categorize solution patterns
- **Semantic Search**: Vector-based similarity search
- **Idempotent Operations**: Safe to run multiple times (CLAUDE.md compliant)
- **UTC Timestamps**: All times in UTC (CLAUDE.md compliant)
- **Structured Logging**: JSON-structured logs with correlation IDs

---

## Components

### 1. KnowledgeArtifact

Dataclass representing a piece of knowledge with rich metadata.

**Key Attributes:**
- `id`: Unique identifier (UUID/hash)
- `artifact_type`: Type of artifact (solution_pattern, critique_insight, etc.)
- `content`: Artifact content (dict)
- `confidence_score`: Quality confidence (0-1)
- `effectiveness_score`: Effectiveness measure (0-1)
- `validation_status`: Validation state (validated/invalid/unvalidated)
- `metadata`: Additional metadata dict

### 2. KnowledgeExtractor

Extracts knowledge artifacts from workflow execution data.

**Extraction Types:**
- Solution patterns
- Critique insights
- Team performance metrics
- Gauntlet effectiveness
- Cross-cutting patterns

**Features:**
- Pattern recognition
- Quality assessment
- Relationship mapping
- Entity extraction

### 3. KnowledgeStorage

Stores artifacts across multiple database backends.

**Supported Backends:**
- MongoDB (document storage)
- Qdrant (vector search)
- Neo4j (graph relationships)
- Redis (caching)

**Features:**
- Automatic caching
- Backup/restore
- Statistics tracking
- Idempotent operations

### 4. KnowledgeRetriever

Retrieves knowledge artifacts with advanced search capabilities.

**Search Types:**
- Keyword search
- Vector similarity search
- Hybrid search (combined)
- Context-aware recommendations

**Features:**
- Query caching
- Advanced filtering
- Trend analysis
- Quality metrics

---

## Installation

```bash
# Required dependencies
pip install pymongo qdrant-client redis neo4j

# Optional for embeddings
pip install sentence-transformers
```

---

## Quick Start

### Basic Extraction and Storage

```python
from datetime import datetime, timezone
from knowledge_engine.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.knowledge_storage import KnowledgeStorage
from knowledge_engine.knowledge_retriever import KnowledgeRetriever

# Sample workflow data
workflow_data = {
    'workflow_id': 'workflow_001',
    'domain': 'optimization',
    'complexity': 'high',
    'execution_time': 1800,
    'success': True,
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'solutions': [
        {
            'id': 'sol_001',
            'problem_type': 'optimization',
            'domain': 'mathematics',
            'approach': 'gradient descent',
            'implementation': 'iterative approach',
            'success_rate': 0.92,
            'complexity': 7,
            'code': 'def optimize(): pass',
            'documentation': 'Standard gradient descent',
            'performance': {'iterations': 100}
        }
    ],
    'critiques': [
        {
            'id': 'crit_001',
            'issue_type': 'convergence',
            'root_cause': 'learning rate too high',
            'prevention_strategy': 'adaptive learning rate',
            'severity': 'medium',
            'affected_components': ['optimizer']
        }
    ],
    'teams': [
        {
            'name': 'blue_team',
            'role': 'Blue',
            'domain': 'optimization',
            'success_rate': 0.90,
            'avg_response_time': 1.5,
            'completion_rate': 0.93,
            'quality_score': 0.88,
            'performance_trends': [0.85, 0.87, 0.88, 0.89, 0.90]
        }
    ],
    'gauntlets': [
        {
            'name': 'quality_gauntlet',
            'type': 'Gold',
            'domain': 'validation',
            'problem_type': 'quality',
            'detection_rate': 0.88,
            'false_positive_rate': 0.05,
            'true_positive_rate': 0.85,
            'average_score': 0.87,
            'performance_trends': [0.83, 0.85, 0.86, 0.87, 0.88]
        }
    ]
}

# Initialize components
extractor = KnowledgeExtractor()
storage = KnowledgeStorage()
retriever = KnowledgeRetriever(storage=storage)

# Extract knowledge
artifacts = extractor.extract_from_workflow(workflow_data)
print(f"Extracted {len(artifacts)} artifacts")

# Store artifacts
for artifact in artifacts:
    artifact_dict = artifact.to_dict()
    artifact_dict['type'] = artifact.artifact_type
    artifact_dict['source'] = artifact.source_workflow_id
    artifact_dict['content'] = json.dumps(artifact.content)

    artifact_id = storage.store_knowledge_artifact(artifact_dict)
    print(f"Stored artifact: {artifact_id}")

# Search knowledge
results = retriever.search_knowledge(
    query='optimization',
    query_type='hybrid',
    limit=5
)
print(f"Found {len(results)} results")
```

---

## Detailed Usage

### KnowledgeArtifact

#### Creating Artifacts

```python
from knowledge_engine.knowledge_extractor import KnowledgeArtifact
from datetime import datetime, timezone

artifact = KnowledgeArtifact(
    id='artifact_001',
    artifact_type='solution_pattern',
    content={
        'problem_type': 'decomposition',
        'solution_approach': 'hierarchical analysis',
        'success_rate': 0.95
    },
    source_workflow_id='workflow_123',
    extraction_timestamp=datetime.now(timezone.utc).timestamp(),
    domain='optimization',
    confidence_score=0.9,
    effectiveness_score=0.88
)

# Calculate quality score
quality = artifact.calculate_quality_score()
print(f"Quality score: {quality:.2f}")
```

#### Validation

```python
# Validate artifact
artifact.validate_artifact(
    validation_result=True,
    validator="automated_validator"
)

# After validation
assert artifact.validation_status == 'validated'
assert artifact.confidence_score == 0.95
```

#### Metadata Updates

```python
# Update metadata (automatically increments version)
artifact.update_metadata({
    'reviewed_by': 'expert_1',
    'tags': ['optimization', 'decomposition']
})

# Version and timestamp updated
print(f"Version: {artifact.version}")
print(f"Last updated: {artifact.last_updated}")
```

#### Serialization

```python
# Convert to dict
artifact_dict = artifact.to_dict()

# Convert back from dict
restored = KnowledgeArtifact.from_dict(artifact_dict)
```

### KnowledgeExtractor

#### Basic Extraction

```python
from knowledge_engine.knowledge_extractor import KnowledgeExtractor

extractor = KnowledgeExtractor({
    'quality_thresholds': {
        'high': 0.85,
        'medium': 0.65,
        'low': 0.40
    }
})

artifacts = extractor.extract_from_workflow(workflow_data)
```

#### Pattern Recognition

```python
# Extract solution patterns with pattern recognition
solution_artifacts = [
    a for a in artifacts
    if a.artifact_type == 'solution_pattern'
]

for solution in solution_artifacts:
    pattern_info = solution.metadata.get('pattern_recognition', {})
    print(f"Pattern: {pattern_info.get('pattern_type')}")
    print(f"Match score: {pattern_info.get('match_score'):.2f}")
```

#### Quality Filtering

```python
# Get extraction statistics
stats = extractor.get_extraction_stats()

print(f"Total extractions: {stats['total_extractions']}")
print(f"Success rate: {stats['success_rate']:.2f}")
print(f"Quality distribution: {stats['quality_distribution']}")

# High quality percentage
print(f"High quality: {stats['high_quality_percentage']:.1f}%")
```

#### Entity Relationships

```python
# Get discovered entity relationships
entity_relationships = extractor.get_entity_relationships()

for entity, artifact_types in list(entity_relationships.items())[:5]:
    print(f"{entity}: {list(artifact_types)}")
```

### KnowledgeStorage

#### Configuration

```python
from knowledge_engine.knowledge_storage import KnowledgeStorage

storage = KnowledgeStorage({
    'qdrant_host': 'localhost',
    'qdrant_port': 6333,
    'mongo_uri': 'mongodb://localhost:27017',
    'neo4j_uri': 'bolt://localhost:7687',
    'neo4j_user': 'neo4j',
    'neo4j_password': 'password',
    'redis_host': 'localhost',
    'redis_port': 6379
})
```

#### Storing Artifacts

```python
# Store with automatic ID generation
artifact_id = storage.store_knowledge_artifact({
    'type': 'solution_pattern',
    'source': 'workflow_001',
    'content': 'Hierarchical decomposition approach',
    'context': {
        'problem_type': 'decomposition',
        'complexity': 'high'
    },
    'embeddings': [0.1] * 768,  # 768-dim vector
    'related_entities': ['decomposition', 'hierarchy']
})

# Store with specific ID (idempotent)
artifact_id = storage.store_knowledge_artifact({
    '_id': 'specific_id',
    'type': 'solution_pattern',
    'source': 'workflow_001',
    'content': 'Content here'
})
```

#### Retrieving Artifacts

```python
# Get by ID
artifact = storage.get_artifact_by_id(artifact_id)

# Retrieve with filters
artifacts = storage.retrieve_knowledge_artifacts({
    'type': 'solution_pattern',
    'context.problem_type': 'decomposition'
}, limit=10)

# Vector similarity search
similar = storage.search_similar_artifacts(
    query_embedding=[0.1] * 768,
    artifact_type='solution_pattern',
    limit=5
)
```

#### Updating and Deleting

```python
# Update artifact
storage.update_artifact(artifact_id, {
    'content': 'Updated content',
    'metadata': {'updated': True}
})

# Delete artifact
storage.delete_artifact(artifact_id)
```

#### Statistics and Maintenance

```python
# Get statistics
stats = storage.get_statistics()
print(f"Total artifacts: {stats['total_artifacts']}")
print(f"Artifact types: {stats['artifact_types']}")
print(f"Storage size: {stats['storage_size']} bytes")

# Backup
storage.backup_knowledge_base('backup.json')

# Restore
storage.restore_knowledge_base('backup.json')
```

### KnowledgeRetriever

#### Basic Search

```python
from knowledge_engine.knowledge_retriever import KnowledgeRetriever

retriever = KnowledgeRetriever(storage=storage)

# Hybrid search (vector + keyword)
results = retriever.search_knowledge(
    query='decomposition optimization',
    query_type='hybrid',
    limit=10
)

# Keyword-only search
results = retriever.search_knowledge(
    query='optimization',
    query_type='keyword',
    limit=5
)

# Vector-only search
results = retriever.search_knowledge(
    query='pattern recognition',
    query_type='vector',
    limit=5
)
```

#### Context-Aware Recommendations

```python
# Get recommendations based on context
recommendations = retriever.get_recommendations(
    context={
        'problem_type': 'decomposition',
        'complexity': 'high',
        'domain': 'optimization'
    },
    recommendation_type='solution_pattern',
    limit=5
)

for rec in recommendations:
    print(f"Recommended: {rec['content']}")
```

#### Related Knowledge

```python
# Find related artifacts
related = retriever.get_related_knowledge(
    artifact_id='artifact_001',
    relationship_type='related',
    limit=5
)
```

#### Advanced Search

```python
# Advanced search with filtering and pagination
results = retriever.advanced_search({
    'query': 'solution',
    'filters': {'type': 'solution_pattern'},
    'sort_by': 'timestamp',
    'sort_order': 'desc',
    'facets': ['source', 'context.problem_type'],
    'page': 1,
    'page_size': 20
})

print(f"Total results: {results['total_results']}")
print(f"Page {results['page']} of {results['total_pages']}")

# Faceted results
for facet, counts in results['facets'].items():
    print(f"\n{facet}:")
    for item in counts[:5]:
        print(f"  {item['value']}: {item['count']}")
```

#### Trend Analysis

```python
# Analyze knowledge trends over time
trends = retriever.get_knowledge_trends(
    time_range='30d',
    artifact_type='solution_pattern'
)

print(f"Trend: {trends['trend_analysis']['trend']}")
print(f"Change: {trends['trend_analysis']['change_percentage']:.1f}%")

# Daily trends
for day in trends['daily_trends'][-7:]:
    print(f"{day['date']}: {day['count']} artifacts")
```

#### Quality Metrics

```python
# Get comprehensive quality metrics
metrics = retriever.get_knowledge_quality_metrics()

quality = metrics['quality_metrics']
print(f"Completeness: {quality['completeness']:.2f}")
print(f"Consistency: {quality['consistency']:.2f}")
print(f"Relevance: {quality['relevance']:.2f}")
print(f"Timeliness: {quality['timeliness']:.2f}")
print(f"Diversity: {quality['diversity']:.2f}")
print(f"\nOverall: {metrics['overall_quality_score']:.2f}")
```

---

## API Reference

### KnowledgeArtifact

```python
class KnowledgeArtifact:
    id: str
    artifact_type: str
    content: Dict[str, Any]
    source_workflow_id: str
    extraction_timestamp: float
    domain: Optional[str] = None
    problem_type: Optional[str] = None
    confidence_score: float = 0.8
    effectiveness_score: float = 0.0
    validation_status: str = "unvalidated"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeArtifact'
    def calculate_quality_score(self) -> float
    def validate_artifact(self, validation_result: bool, validator: str)
    def update_metadata(self, updates: Dict[str, Any])
```

### KnowledgeExtractor

```python
class KnowledgeExtractor:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def extract_from_workflow(self, workflow_data: Dict[str, Any]) -> List[KnowledgeArtifact]
    def get_extraction_stats(self) -> Dict[str, Any]
    def reset_stats(self)
    def get_entity_relationships(self) -> Dict[str, Set[str]]
    def get_domain_patterns(self) -> Dict[str, List[str]]
```

### KnowledgeStorage

```python
class KnowledgeStorage:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def store_knowledge_artifact(self, artifact: Dict[str, Any]) -> str
    def retrieve_knowledge_artifacts(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]
    def get_artifact_by_id(self, artifact_id: str) -> Optional[Dict[str, Any]]
    def update_artifact(self, artifact_id: str, updates: Dict[str, Any]) -> bool
    def delete_artifact(self, artifact_id: str) -> bool
    def search_similar_artifacts(self, query_embedding: List[float], artifact_type: str = None, limit: int = 5) -> List[Dict[str, Any]]
    def get_statistics(self) -> Dict[str, Any]
    def backup_knowledge_base(self, backup_path: str) -> bool
    def restore_knowledge_base(self, backup_path: str) -> bool
```

### KnowledgeRetriever

```python
class KnowledgeRetriever:
    def __init__(self, storage: KnowledgeStorage = None, config: Optional[Dict[str, Any]] = None)
    def search_knowledge(self, query: str, query_type: str = 'hybrid', filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]
    def get_recommendations(self, context: Dict[str, Any], recommendation_type: str = 'solution_pattern', limit: int = 5) -> List[Dict[str, Any]]
    def get_related_knowledge(self, artifact_id: str, relationship_type: str = 'related', limit: int = 5) -> List[Dict[str, Any]]
    def advanced_search(self, search_params: Dict[str, Any]) -> Dict[str, Any]
    def get_knowledge_trends(self, time_range: str = '30d', artifact_type: str = None) -> Dict[str, Any]
    def get_knowledge_quality_metrics(self) -> Dict[str, Any]
```

---

## Best Practices

### 1. CLAUDE.md Compliance

**Idempotency**
```python
# Store operations are idempotent
artifact_id = storage.store_knowledge_artifact(artifact)
artifact_id = storage.store_knowledge_artifact(artifact)  # Same ID
```

**UTC Timestamps**
```python
from datetime import datetime, timezone

# Always use UTC
timestamp = datetime.now(timezone.utc).timestamp()

# Store with UTC timestamp
artifact['timestamp'] = datetime.now(timezone.utc).isoformat()
```

**Structured Logging**
```python
import logging
import json

logger = logging.getLogger(__name__)

# Structured logging
logger.info(json.dumps({
    'msg': 'Artifact stored',
    'artifact_id': artifact_id,
    'artifact_type': artifact_type
}))
```

### 2. Quality Assessment

```python
# Set appropriate quality thresholds
extractor = KnowledgeExtractor({
    'quality_thresholds': {
        'high': 0.85,
        'medium': 0.65,
        'low': 0.40
    }
})

# Filter by quality
artifacts = extractor.extract_from_workflow(workflow_data)
high_quality = [a for a in artifacts if a.calculate_quality_score() >= 0.85]
```

### 3. Batch Operations

```python
# Store multiple artifacts efficiently
artifact_ids = []
for artifact in artifacts:
    artifact_dict = artifact.to_dict()
    artifact_id = storage.store_knowledge_artifact(artifact_dict)
    artifact_ids.append(artifact_id)
```

### 4. Caching

```python
# Retriever uses query caching automatically
retriever = KnowledgeRetriever(storage=storage, config={
    'cache_ttl': 300  # 5 minutes
})

# Subsequent queries with same parameters use cache
results1 = retriever.search_knowledge('optimization')
results2 = retriever.search_knowledge('optimization')  # From cache
```

### 5. Error Handling

```python
# Always handle exceptions gracefully
try:
    artifacts = extractor.extract_from_workflow(workflow_data)
    stats = extractor.get_extraction_stats()
except Exception as e:
    logger.error(f"Extraction failed: {str(e)}")
    # Fallback or recovery logic
```

---

## Examples

### Example 1: Complete Knowledge Pipeline

```python
from knowledge_engine.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.knowledge_storage import KnowledgeStorage
from knowledge_engine.knowledge_retriever import KnowledgeRetriever
import json

# Initialize
extractor = KnowledgeExtractor()
storage = KnowledgeStorage()
retriever = KnowledgeRetriever(storage=storage)

# Extract
artifacts = extractor.extract_from_workflow(workflow_data)

# Store
for artifact in artifacts:
    artifact_dict = artifact.to_dict()
    artifact_dict['type'] = artifact.artifact_type
    artifact_dict['source'] = artifact.source_workflow_id
    artifact_dict['content'] = json.dumps(artifact.content)
    storage.store_knowledge_artifact(artifact_dict)

# Retrieve and Search
results = retriever.search_knowledge('optimization', limit=10)
recommendations = retriever.get_recommendations({
    'problem_type': 'decomposition'
})

# Analytics
stats = extractor.get_extraction_stats()
trends = retriever.get_knowledge_trends('30d')
quality = retriever.get_knowledge_quality_metrics()
```

### Example 2: Pattern Recognition

```python
# Extract with pattern recognition
artifacts = extractor.extract_from_workflow(workflow_data)

# Find hierarchical patterns
hierarchical_solutions = [
    a for a in artifacts
    if a.artifact_type == 'solution_pattern' and
    a.metadata.get('pattern_recognition', {}).get('pattern_type') == 'hierarchical_decomposition'
]

print(f"Found {len(hierarchical_solutions)} hierarchical solutions")
```

### Example 3: Quality-Based Filtering

```python
# Extract artifacts
artifacts = extractor.extract_from_workflow(workflow_data)

# Filter by quality
high_quality = [a for a in artifacts if a.calculate_quality_score() >= 0.85]
medium_quality = [a for a in artifacts if 0.65 <= a.calculate_quality_score() < 0.85]

print(f"High quality: {len(high_quality)}")
print(f"Medium quality: {len(medium_quality)}")

# Store only high quality
for artifact in high_quality:
    artifact_dict = artifact.to_dict()
    storage.store_knowledge_artifact(artifact_dict)
```

### Example 4: Trend Analysis

```python
# Analyze trends
trends = retriever.get_knowledge_trends('30d', 'solution_pattern')

print(f"Trend: {trends['trend_analysis']['trend']}")
print(f"Change: {trends['trend_analysis']['change_percentage']:.1f}%")
print(f"Average daily: {trends['trend_analysis']['average_daily']:.1f}")

# Plot daily trends (requires matplotlib)
import matplotlib.pyplot as plt

days = [t['date'] for t in trends['daily_trends']]
counts = [t['count'] for t in trends['daily_trends']]

plt.figure(figsize=(12, 6))
plt.plot(days, counts)
plt.xlabel('Date')
plt.ylabel('Artifacts Created')
plt.title('Knowledge Creation Trends')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest knowledge_engine/tests/test_knowledge_artifacts.py -v

# Run specific test class
pytest knowledge_engine/tests/test_knowledge_artifacts.py::TestKnowledgeExtractor -v

# Run specific test
pytest knowledge_engine/tests/test_knowledge_artifacts.py::TestKnowledgeExtractor::test_extract_from_workflow -v

# Run with coverage
pytest knowledge_engine/tests/test_knowledge_artifacts.py --cov=knowledge_engine.knowledge_extractor --cov=knowledge_engine.knowledge_storage --cov=knowledge_engine.knowledge_retriever
```

---

## Performance Considerations

### Extraction Performance

- **Large workflows**: Can handle 100+ solutions efficiently
- **Quality filtering**: Automatic, prevents low-quality artifacts
- **Pattern matching**: Optimized pattern library

### Storage Performance

- **Caching**: Redis caching for frequently accessed artifacts
- **Batch operations**: Store multiple artifacts in batches
- **Idempotency**: Safe to retry operations

### Retrieval Performance

- **Query caching**: Automatic caching of search results
- **Vector search**: Optimized similarity search
- **Pagination**: Efficient pagination for large result sets

---

## Troubleshooting

### Issue: Low quality scores

**Solution**: Adjust quality thresholds or improve workflow data quality

```python
extractor = KnowledgeExtractor({
    'quality_thresholds': {
        'high': 0.75,  # Lower threshold
        'medium': 0.55,
        'low': 0.35
    }
})
```

### Issue: Poor search results

**Solution**: Use hybrid search or provide better embeddings

```python
results = retriever.search_knowledge(
    query='specific query',
    query_type='hybrid',  # Combine vector and keyword
    filters={'type': 'solution_pattern'}  # Add filters
)
```

### Issue: Storage connection errors

**Solution**: Check configuration and ensure databases are running

```python
storage = KnowledgeStorage({
    'mongo_uri': 'mongodb://localhost:27017',
    'qdrant_host': 'localhost',
    'qdrant_port': 6333
    # Verify these are correct
})
```

---

## Additional Resources

- **CLAUDE.md**: Project constitution and principles
- **Test Suite**: `knowledge_engine/tests/test_knowledge_artifacts.py`
- **Example Scripts**: See `__main__` sections in each module

---

## License

See project LICENSE file.
