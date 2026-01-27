# Core Graph Modules - Quick Reference

**Version:** 2.0.0
**Status:** Production Ready ✅

---

## EntityKnowledgeGraph

### Import
```python
from knowledge_engine.core import EntityKnowledgeGraph, Entity, Relationship
```

### Create
```python
graph = EntityKnowledgeGraph(correlation_id="optional-id")
```

### Core Operations

#### Add Entity
```python
# Sync
graph.add_entity(name, entity_type, attributes={})
graph.add_entity("Alice", "Person", {"age": 30})

# Async
await graph.add_entity_async("Alice", "Person", {"age": 30})
```

#### Add Relationship
```python
# Sync
graph.add_relationship(source, target, relation_type, attributes={})
graph.add_relationship("Alice", "Bob", "KNOWS", {"since": "2020"})

# Async
await graph.add_relationship_async("Alice", "Bob", "KNOWS")
```

#### Get Entity
```python
# Sync
entity = graph.get_entity("Alice")

# Async
entity = await graph.get_entity_async("Alice")
```

#### Find Entities
```python
# By type
persons = graph.find_entities(entity_type="Person")

# By attributes
results = graph.find_entities(
    entity_type="Person",
    attributes={"role": "engineer", "level": 5}
)

# Async
results = await graph.find_entities_async(entity_type="Person")
```

#### Search Entities
```python
results = graph.search_entities("query string", limit=100)
# Async: await graph.search_entities_async("query")
```

#### Get Relationships
```python
relationships = graph.get_relationships("Alice")
# Async: await graph.get_relationships_async("Alice")
```

### Serialization

#### To JSON
```python
json_str = graph.to_json()
# Async: await graph.to_json_async()
```

#### From JSON
```python
graph = EntityKnowledgeGraph()
graph.from_json(json_str)
# Async: await graph.from_json_async(json_str)
```

### Statistics
```python
stats = graph.get_statistics()
# Returns: {
#   "entity_count": 10,
#   "entity_types": {"Person": 5, "Organization": 5},
#   "relationship_count": 15,
#   "correlation_id": "...",
#   "timestamp": "..."
# }
```

### Clear
```python
graph.clear()
# Async: await graph.clear_async()
```

---

## KnowledgeState

### Import
```python
from knowledge_engine.core import KnowledgeState, KnowledgeTriple, StateSnapshot
```

### Create
```python
state = KnowledgeState(query="Your research question", correlation_id="optional-id")
```

### Core Operations

#### Add Knowledge
```python
# Sync
triples = [
    ("AI", "is", "technology"),
    ("AI", "includes", "ML")
]
state.add_knowledge(triples, timestamp=None, source="optional")

# Async
await state.add_knowledge_async(triples, timestamp=None, source="optional")
```

#### Add Fact
```python
# Sync
state.add_fact("AI is important")
# Returns: True if added, False if duplicate

# Async
await state.add_fact_async("AI is important")
```

#### Add Uncertainty
```python
# Sync
state.add_uncertainty("AGI timeline unknown")

# Async
await state.add_uncertainty_async("AGI timeline unknown")
```

### State Queries

#### Get Current State
```python
# Sync
current = state.get_current_state()
# Returns: {
#   "query": "...",
#   "triples": [...],
#   "facts": [...],
#   "uncertainties": [...],
#   "version": 1,
#   "timestamp": "...",
#   "correlation_id": "..."
# }

# Async
current = await state.get_current_state_async()
```

#### Get State at Time
```python
# Sync
state_at_time = state.get_state_at_time("2024-01-01T12:00:00Z")
# Returns: state snapshot or None

# Async
state_at_time = await state.get_state_at_time_async("2024-01-01T12:00:00Z")
```

#### Get History
```python
# Sync
history = state.get_history()
# Returns: list of StateSnapshot dicts

# Async
history = await state.get_history_async()
```

#### Search Triples
```python
# Sync
results = state.search_triples(subject="AI", predicate="is", obj="technology")
# All parameters optional

# Async
results = await state.search_triples_async(subject="AI")
```

### Serialization

#### To JSON
```python
json_str = state.to_json()
# Async: await state.to_json_async()
```

#### From JSON
```python
state = KnowledgeState.from_json(json_str)
# Async: await KnowledgeState.from_json_async(json_str)
```

### Statistics
```python
stats = state.get_statistics()
# Returns: {
#   "query": "...",
#   "triple_count": 10,
#   "fact_count": 5,
#   "uncertainty_count": 2,
#   "snapshot_count": 3,
#   "version": 5,
#   "correlation_id": "...",
#   "timestamp": "..."
# }
```

### Clear
```python
state.clear()
# Async: await state.clear_async()
```

---

## Common Patterns

### Pattern 1: Build Knowledge Graph from Text
```python
graph = EntityKnowledgeGraph()
state = KnowledgeState(query="Extract knowledge about Python")

# Add entities
graph.add_entity("Python", "Language", {"paradigm": "multi"})
graph.add_entity("data_science", "Field")

# Add relationships
graph.add_relationship("Python", "data_science", "USED_IN")

# Add triples to state
state.add_knowledge([
    ("Python", "is", "programming language"),
    ("Python", "used_in", "data science")
])
```

### Pattern 2: Temporal Knowledge Tracking
```python
state = KnowledgeState(query="AI research timeline")

# Add knowledge at different times
state.add_knowledge([("AI", "started", "1950s")], timestamp="2024-01-01T10:00:00Z")
state.add_knowledge([("AI", "progressed", "2020s")], timestamp="2024-01-02T10:00:00Z")

# Query evolution
early_state = state.get_state_at_time("2024-01-01T12:00:00Z")
current_state = state.get_current_state()
history = state.get_history()
```

### Pattern 3: Async Pipeline
```python
import asyncio

async def process_knowledge():
    graph = EntityKnowledgeGraph()
    state = KnowledgeState(query="Async processing")

    # Concurrent operations
    await asyncio.gather(
        graph.add_entity_async("Entity1", "Type"),
        graph.add_entity_async("Entity2", "Type"),
        state.add_knowledge_async([("E1", "rel", "E2")])
    )

    # Query results
    entities = await graph.find_entities_async(entity_type="Type")
    current_state = await state.get_current_state_async()

    return entities, current_state

# Run
entities, state = asyncio.run(process_knowledge())
```

### Pattern 4: Serialization Roundtrip
```python
# Save
graph = EntityKnowledgeGraph()
graph.add_entity("Alice", "Person")
graph_json = graph.to_json()

# Load
graph2 = EntityKnowledgeGraph()
graph2.from_json(graph_json)
assert graph2.get_entity("Alice") is not None
```

---

## Error Handling

All methods return `False` on error and log structured error messages:

```python
result = graph.add_entity("", "Person")
# Returns: False
# Logs: {"msg": "Failed to add entity: ", "error": "Entity name must be non-empty"}
```

---

## Testing

Run comprehensive test suite:

```bash
cd /path/to/OpenEvolve/Frontend
python test_core_simple.py
```

Expected: All 32 tests pass ✅

---

## CLAUDE.md Principles

✅ **ZERO TRUST** - All inputs validated
✅ **RUNTIME TRUTH** - Operations verified
✅ **IDEMPOTENCY** - Safe to retry
✅ **CONFIGURATION EXPLICIT** - No magic defaults
✅ **UTC TIME** - All timestamps in UTC
✅ **STRUCTURED LOGGING** - JSON logs with correlation IDs

---

## Data Classes

### Entity
```python
@dataclass
class Entity:
    name: str
    entity_type: str
    attributes: Dict[str, Any]
    created_at: str
    updated_at: str
```

### Relationship
```python
@dataclass
class Relationship:
    source: str
    target: str
    relation_type: str
    attributes: Dict[str, Any]
    created_at: str
    id: str
```

### KnowledgeTriple
```python
@dataclass
class KnowledgeTriple:
    subject: str
    predicate: str
    obj: str
    confidence: float
    timestamp: str
    source: Optional[str]
    metadata: Dict[str, Any]
```

### StateSnapshot
```python
@dataclass
class StateSnapshot:
    timestamp: str
    triples: List[KnowledgeTriple]
    facts: List[str]
    uncertainties: List[str]
    version: int
    metadata: Dict[str, Any]
```

---

## Performance Notes

- **Thread-Safe:** Uses locks for concurrent access
- **Memory Efficient:** In-memory storage
- **Fast Lookups:** O(1) entity lookups by name
- **Idempotent:** Safe to retry operations
- **No External Dependencies:** Pure Python standard library

---

## Support

For issues or questions:
- Check test suite: `test_core_simple.py`
- See implementation: `knowledge_engine/core/entity_knowledge_graph.py`
- See implementation: `knowledge_engine/core/knowledge_state.py`
- Full docs: `knowledge_engine/CORE_GRAPH_MODULES_COMPLETE.md`
