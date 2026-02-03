# Temporal Bridge API Reference

Complete API reference for the Graphiti Temporal Bridge integration.

## Classes

### `GraphitiTemporalBridge`

High-level bridge for Graphiti temporal knowledge graph operations.

#### Constructor

```python
GraphitiTemporalBridge(
    graphiti_bridge: Optional[GraphitiBridge] = None,
    config_path: Optional[str] = None
)
```

**Parameters**:
- `graphiti_bridge`: Optional existing GraphitiBridge instance
- `config_path`: Path to Graphiti configuration file (default: "integrations/graphiti/config.yaml")

#### Methods

##### `initialize() -> bool`

Initialize the temporal bridge.

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
bridge = GraphitiTemporalBridge()
success = await bridge.initialize()
if success:
    print("Bridge initialized")
```

##### `add_artifact(artifact: KnowledgeArtifact) -> Dict[str, Any]`

Add a KnowledgeArtifact to Graphiti.

**Parameters**:
- `artifact`: KnowledgeArtifact to add

**Returns**: Result dictionary
```python
{
    "success": bool,
    "result": Any,  # Graphiti result
    "error": Optional[str]
}
```

**Example**:
```python
artifact = KnowledgeArtifact(
    id="artifact_001",
    content="Knowledge content",
    artifact_type="solution_pattern",
    valid_at=datetime.now()
)

result = await bridge.add_artifact(artifact)
if result["success"]:
    print("Artifact added")
```

##### `search_with_temporal_filters(...) -> List[KnowledgeArtifact]`

Search with temporal filtering.

**Signature**:
```python
async def search_with_temporal_filters(
    self,
    query: str,
    filter_type: TemporalFilter = TemporalFilter.CURRENT,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    max_results: int = 10,
    group_ids: Optional[List[str]] = None,
    use_hybrid: bool = True,
    rerank_method: RerankMethod = RerankMethod.RRF
) -> List[KnowledgeArtifact]
```

**Parameters**:
- `query` (str): Search query
- `filter_type` (TemporalFilter): Filter type
  - `TemporalFilter.CURRENT`: Currently valid knowledge
  - `TemporalFilter.ALL`: All historical knowledge
  - `TemporalFilter.TIME_RANGE`: Knowledge within time range
- `start_time` (datetime, optional): Start of time range
- `end_time` (datetime, optional): End of time range
- `max_results` (int): Maximum results to return (default: 10)
- `group_ids` (List[str], optional): Group IDs to scope search
- `use_hybrid` (bool): Use hybrid search (default: True)
- `rerank_method` (RerankMethod): Reranking method
  - `RerankMethod.RRF`: Reciprocal Rank Fusion
  - `RerankMethod.CROSS_ENCODER`: Cross-encoder reranking
  - `RerankMethod.WEIGHTED`: Weighted combination
  - `RerankMethod.NONE`: No reranking

**Returns**: List of KnowledgeArtifacts

**Example**:
```python
# Get current knowledge
results = await bridge.search_with_temporal_filters(
    query="authentication",
    filter_type=TemporalFilter.CURRENT,
    max_results=10
)

# Get knowledge from time range
results = await bridge.search_with_temporal_filters(
    query="authentication",
    filter_type=TemporalFilter.TIME_RANGE,
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 6, 1),
    max_results=20
)
```

##### `query_at_point_in_time(...) -> List[KnowledgeArtifact]`

Query knowledge at a specific point in time.

**Signature**:
```python
async def query_at_point_in_time(
    self,
    query: str,
    timestamp: datetime,
    max_results: int = 10,
    group_ids: Optional[List[str]] = None
) -> List[KnowledgeArtifact]
```

**Parameters**:
- `query` (str): Search query
- `timestamp` (datetime): Point in time
- `max_results` (int): Maximum results (default: 10)
- `group_ids` (List[str], optional): Group IDs

**Returns**: List of KnowledgeArtifacts valid at the given time

**Example**:
```python
# Query as of March 2024
results = await bridge.query_at_point_in_time(
    query="API endpoints",
    timestamp=datetime(2024, 3, 15),
    max_results=10
)
```

##### `detect_contradictions(...) -> ContradictionDetection`

Detect contradictions in knowledge.

**Signature**:
```python
async def detect_contradictions(
    self,
    entity_name: str,
    time_range: Optional[tuple[datetime, datetime]] = None
) -> ContradictionDetection
```

**Parameters**:
- `entity_name` (str): Entity to check
- `time_range` (tuple, optional): Time range as (start, end)

**Returns**: ContradictionDetection object
```python
{
    "has_contradictions": bool,
    "contradictions": List[Dict[str, Any]],
    "timestamp": datetime,
    "confidence": float
}
```

**Example**:
```python
# Check for contradictions
result = await bridge.detect_contradictions(
    entity_name="authentication_api"
)

if result.has_contradictions:
    print(f"Found {len(result.contradictions)} contradictions")
    for c in result.contradictions:
        print(f"  - {c['reason']}")
```

##### `get_entity_timeline(...) -> List[Dict[str, Any]]`

Get timeline of events for an entity.

**Signature**:
```python
async def get_entity_timeline(
    self,
    entity_name: str,
    start_time: datetime,
    end_time: datetime
) -> List[Dict[str, Any]]
```

**Parameters**:
- `entity_name` (str): Entity name
- `start_time` (datetime): Start of timeline
- `end_time` (datetime): End of timeline

**Returns**: List of temporal events
```python
[
    {
        "timestamp": datetime,
        "event_type": str,
        "description": str,
        "artifact_id": str,
        "source": str
    },
    ...
]
```

**Example**:
```python
# Get entity timeline for 2024
timeline = await bridge.get_entity_timeline(
    entity_name="user_service",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31)
)

for event in timeline:
    print(f"[{event['timestamp']}] {event['description']}")
```

##### `get_valid_knowledge_at_time(...) -> List[KnowledgeArtifact]`

Get all valid knowledge at a specific time.

**Signature**:
```python
async def get_valid_knowledge_at_time(
    self,
    timestamp: datetime,
    max_results: int = 100,
    group_ids: Optional[List[str]] = None
) -> List[KnowledgeArtifact]
```

**Parameters**:
- `timestamp` (datetime): Point in time
- `max_results` (int): Maximum results (default: 100)
- `group_ids` (List[str], optional): Group IDs

**Returns**: List of valid KnowledgeArtifacts

**Example**:
```python
# Get all knowledge valid on June 1st
knowledge = await bridge.get_valid_knowledge_at_time(
    timestamp=datetime(2024, 6, 1),
    max_results=100
)
```

## Data Classes

### `KnowledgeArtifact`

Represents a knowledge artifact with temporal metadata.

```python
@dataclass
class KnowledgeArtifact:
    id: str                              # Unique identifier
    content: str                          # Artifact content
    artifact_type: str                    # Type (solution_pattern, fact, etc.)
    valid_at: datetime                    # When knowledge becomes valid
    invalid_at: Optional[datetime] = None # When knowledge becomes invalid
    created_at: Optional[datetime] = None # When artifact was created
    source: str = "openevolve"            # Source identifier
    metadata: Dict[str, Any]              # Additional metadata
    entities: List[str]                   # Entities mentioned
    relationships: List[Dict[str, Any]]   # Relationships
    confidence: float = 1.0               # Confidence score (0-1)
    group_id: Optional[str] = None        # Group ID
```

#### Methods

##### `is_valid_at(timestamp: datetime) -> bool`

Check if artifact is valid at a given time.

**Parameters**:
- `timestamp` (datetime): Time to check

**Returns**: `True` if valid, `False` otherwise

**Example**:
```python
artifact = KnowledgeArtifact(
    id="001",
    content="API v1 uses basic auth",
    valid_at=datetime(2024, 1, 1),
    invalid_at=datetime(2024, 6, 1)
)

# Check validity
print(artifact.is_valid_at(datetime(2024, 3, 1)))  # True
print(artifact.is_valid_at(datetime(2024, 7, 1)))  # False
```

##### `to_dict() -> Dict[str, Any]`

Convert to dictionary.

**Returns**: Dictionary representation

##### `from_dict(data: Dict[str, Any]) -> KnowledgeArtifact`

Create from dictionary (class method).

**Parameters**:
- `data` (Dict): Dictionary data

**Returns**: KnowledgeArtifact instance

### `ContradictionDetection`

Result of contradiction detection.

```python
@dataclass
class ContradictionDetection:
    has_contradictions: bool              # Whether contradictions found
    contradictions: List[Dict[str, Any]]  # List of contradictions
    timestamp: datetime                   # Detection timestamp
    confidence: float                     # Confidence score (0-1)
```

## Enums

### `RerankMethod`

Reranking methods for hybrid search.

```python
class RerankMethod(Enum):
    RRF = "rrf"                    # Reciprocal Rank Fusion
    CROSS_ENCODER = "cross_encoder" # Cross-encoder reranking
    WEIGHTED = "weighted"          # Weighted combination
    NONE = "none"                  # No reranking
```

### `TemporalFilter`

Temporal filter types.

```python
class TemporalFilter(Enum):
    CURRENT = "current"           # Currently valid
    ALL = "all"                   # All historical
    TIME_RANGE = "time_range"     # Within time range
```

## Helper Functions

### `get_temporal_bridge(...)`

Get or create temporal bridge singleton.

```python
async def get_temporal_bridge(
    config_path: Optional[str] = None
) -> GraphitiTemporalBridge
```

**Parameters**:
- `config_path` (str, optional): Path to config

**Returns**: Initialized GraphitiTemporalBridge

**Example**:
```python
bridge = await get_temporal_bridge(
    config_path="custom/config.yaml"
)
```

## Error Handling

All async methods handle errors gracefully:

```python
# Method returns None or empty result on error
results = await bridge.search_with_temporal_filters(...)
if not results:
    print("Search failed or no results")

# Check success for operations that return dict
result = await bridge.add_artifact(artifact)
if not result["success"]:
    print(f"Error: {result['error']}")
```

## Type Hints

Full type hints are provided. Use with mypy:

```bash
mypy knowledge_engine/integrations/graphiti_temporal_bridge.py
```

## See Also

- [Temporal Knowledge Integration Guide](../temporal_kg_integration_guide.md)
- [Temporal Queries Tutorial](../tutorials/temporal_queries_tutorial.md)
