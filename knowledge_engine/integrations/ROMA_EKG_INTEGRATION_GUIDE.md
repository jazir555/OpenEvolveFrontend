# ROMA-Entity Knowledge Graph Integration Guide

## Overview

The ROMA-Entity Knowledge Graph (EKG) Integration provides comprehensive bi-directional data flow between ROMA (Recursive Optimized Multi-Agent) decomposition system and the Entity Knowledge Graph. This integration enables:

1. **Knowledge Extraction**: Extract entities and relationships from ROMA decompositions and solutions
2. **Knowledge Storage**: Store ROMA artifacts in EKG for future reference
3. **Knowledge Retrieval**: Query EKG for similar past decompositions to enhance new problem solving
4. **Dependency Tracing**: Trace dependencies across ROMA problems and solutions
5. **Knowledge-Aware Operations**: Enhance ROMA operations using graph context

## Architecture

### Components

#### 1. ROMAEntityExtractor
Extracts knowledge entities from ROMA data:
- Problems and sub-problems
- Solutions and solution approaches
- Decomposition metadata
- Relationships (DECOMPOSED_FROM, DEPENDS_ON, etc.)

#### 2. ROMAKnowledgeWriter
Writes ROMA entities to EKG with:
- Idempotent writes (checks before creating)
- Batch operations for efficiency
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling

#### 3. ROMAKnowledgeReader
Queries EKG for ROMA entities:
- Find similar decompositions
- Retrieve solution artifacts
- Trace dependencies
- Build decomposition trees

### Entity Schema

```python
# ROMA Entity Types
ROMAEntityType.PROBLEM          # Main problem entity
ROMAEntityType.SUB_PROBLEM      # Decomposed sub-problem
ROMAEntityType.SOLUTION         # Solution entity
ROMAEntityType.DEPENDENCY       # Dependency entity
ROMAEntityType.DECOMPOSITION    # Decomposition metadata
ROMAEntityType.AGGREGATION      # Aggregated solution
```

### Relationship Schema

```python
# ROMA Relationship Types
ROMARelationshipType.DECOMPOSED_FROM   # Child -> Parent decomposition
ROMARelationshipType.SOLVES            # Solution -> Problem
ROMARelationshipType.DEPENDS_ON        # Problem -> Dependency
ROMARelationshipType.AGGREGATED_FROM   # Aggregated -> Component
ROMARelationshipType.SIMILAR_TO        # Similar problems
ROMARelationshipType.REUSES            # Solution reuse
ROMARelationshipType.VALIDATED_BY      # Validation relationship
```

## Usage Examples

### Example 1: Extract Entities from ROMA Decomposition

```python
import asyncio
from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph
from knowledge_engine.integrations.roma_integration import ROMAIntegration
from knowledge_engine.integrations.roma_entity_kg_integration import (
    ROMAEntityExtractor,
    ROMAKnowledgeWriter,
    create_roma_ekg_integration
)

async def extract_and_store():
    # Initialize components
    kg = EntityKnowledgeGraph()
    roma = ROMAIntegration()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    # Decompose a problem using ROMA
    result = await roma.decompose_problem(
        "Design a scalable RESTful API architecture"
    )

    if result.success and result.decomposition:
        # Extract entities from decomposition
        entities = await extractor.extract_from_decomposition(
            decomposition=result.decomposition.__dict__
        )

        print(f"Extracted {len(entities)} entities")
        for entity in entities:
            print(f"  - {entity.entity_type.value}: {entity.name}")

        # Store entities in knowledge graph
        entity_ids = await writer.store_entities(entities)
        print(f"Stored {len(entity_ids)} entities in EKG")

asyncio.run(extract_and_store())
```

### Example 2: Extract and Store Solutions

```python
async def extract_and_store_solutions():
    kg = EntityKnowledgeGraph()
    roma = ROMAIntegration()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    # Solve an atomic problem
    atomic_problem = {
        "decomposition_id": "problem_123",
        "problem": "Implement JWT authentication",
        "is_atomic": True,
        "depth": 0
    }

    result = await roma.solve_atomic(atomic_problem)

    if result.success and result.solutions:
        solution = result.solutions[0]

        # Extract solution entities
        entities = await extractor.extract_from_solution(
            solution=solution.__dict__
        )

        # Store as knowledge artifact
        artifact_id = await writer.store_artifact(
            solution=solution.__dict__
        )

        print(f"Stored solution artifact: {artifact_id}")

asyncio.run(extract_and_store_solutions())
```

### Example 3: Store Complete Decomposition Graph

```python
async def store_decomposition_graph():
    kg = EntityKnowledgeGraph()
    roma = ROMAIntegration()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    # Decompose problem
    result = await roma.decompose_problem(
        "Build a microservices e-commerce platform",
        max_depth=3
    )

    if result.success and result.decomposition:
        decomp = result.decomposition.__dict__

        # Extract entities
        entities = await extractor.extract_from_decomposition(decomp)

        # Extract relationships
        relationships = await extractor.extract_relationships(
            decomposition=decomp,
            entities=entities
        )

        # Store complete graph
        graph_id = await writer.store_decomposition_graph(
            decomposition=decomp,
            entities=entities,
            relationships=relationships
        )

        print(f"Stored decomposition graph: {graph_id}")
        print(f"  Entities: {len(entities)}")
        print(f"  Relationships: {len(relationships)}")

asyncio.run(store_decomposition_graph())
```

### Example 4: Find Similar Decompositions

```python
async def find_similar():
    kg = EntityKnowledgeGraph()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    # Search for similar decompositions
    similar = await reader.find_similar_decompositions(
        problem="Design API gateway with rate limiting",
        top_k=5
    )

    print(f"Found {len(similar)} similar decompositions:")
    for i, decomp in enumerate(similar, 1):
        print(f"\n{i}. {decomp.problem}")
        print(f"   Similarity: {decomp.similarity_score:.2f}")
        print(f"   Sub-problems: {decomp.sub_problems}")
        print(f"   Solution: {decomp.solution_summary[:100]}...")

asyncio.run(find_similar())
```

### Example 5: Trace Dependencies

```python
async def trace_dependencies():
    kg = EntityKnowledgeGraph()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    # Trace dependencies for a problem
    dependencies = await reader.trace_dependencies(
        problem_id="roma_problem_123"
    )

    print(f"Dependencies for roma_problem_123:")
    for dep in dependencies:
        print(f"  -> {dep['target_name']}")
        print(f"     Type: {dep['properties'].get('dependency_type', 'unknown')}")

asyncio.run(trace_dependencies())
```

### Example 6: Get Decomposition Tree

```python
async def get_tree():
    kg = EntityKnowledgeGraph()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    # Get complete decomposition tree
    tree = await reader.get_decomposition_tree(
        decomposition_id="main_problem_456",
        max_depth=5
    )

    def print_tree(node, indent=0):
        print("  " * indent + f"- {node['name']}")
        for child in node.get('sub_problems', []):
            print_tree(child, indent + 1)

    print_tree(tree)

asyncio.run(get_tree())
```

### Example 7: Knowledge-Aware ROMA Solving

```python
async def knowledge_aware_solving():
    kg = EntityKnowledgeGraph()
    roma = ROMAIntegration()
    extractor, writer, reader = create_roma_ekg_integration(kg)

    problem = "Implement distributed caching system"

    # Step 1: Find similar past problems
    similar = await reader.find_similar_decompositions(
        problem=problem,
        top_k=3
    )

    # Step 2: Extract knowledge from similar problems
    context = []
    for decomp in similar:
        artifacts = await reader.get_solution_artifacts(
            entity_id=decomp.decomposition_id
        )
        context.extend([a['solution'] for a in artifacts])

    # Step 3: Enhance problem with knowledge context
    enhanced_problem = f"""
Problem: {problem}

Relevant Context from Similar Problems:
{chr(10).join([f"- {c[:200]}" for c in context[:3]])}
"""

    # Step 4: Solve enhanced problem
    result = await roma.decompose_problem(enhanced_problem)

    # Step 5: Store new solution in knowledge graph
    if result.success:
        entities = await extractor.extract_from_decomposition(
            decomposition=result.decomposition.__dict__
        )
        await writer.store_entities(entities)

        print("Solved with knowledge awareness!")

asyncio.run(knowledge_aware_solving())
```

## Configuration

### Default Configuration

```python
config = {
    # Entity Extraction
    "extract_properties": True,
    "extract_metadata": True,
    "compute_embeddings": False,
    "min_confidence": 0.5,
    "max_sub_problems": 1000,

    # Knowledge Writer
    "auto_extract": True,
    "auto_store": True,
    "batch_size": 100,
    "timeout_seconds": 30,
    "idempotent": True,
    "retry_attempts": 3,
    "retry_backoff_ms": 1000,

    # Knowledge Reader
    "default_top_k": 5,
    "similarity_threshold": 0.7,
    "max_results": 100,
    "include_metadata": True
}

# Use custom config
extractor, writer, reader = create_roma_ekg_integration(
    knowledge_graph=kg,
    config=config
)
```

## Error Handling

All components follow CLAUDE.md principles with comprehensive error handling:

```python
# Extractor returns empty list on failure
entities = await extractor.extract_from_decomposition(decomposition)
if not entities:
    logger.error("Extraction failed")

# Writer returns partial results on failure
entity_ids = await writer.store_entities(entities)
if len(entity_ids) < len(entities):
    logger.warning(f"Only stored {len(entity_ids)}/{len(entities)} entities")

# Reader returns empty list on failure
similar = await reader.find_similar_decompositions(problem)
if not similar:
    logger.info("No similar decompositions found")
```

## Circuit Breaker Pattern

The ROMAKnowledgeWriter implements a circuit breaker pattern for fault tolerance:

```python
# Circuit breaker opens after N failures
config = {
    "failure_threshold": 5,           # Open circuit after 5 failures
    "recovery_timeout_seconds": 60    # Attempt recovery after 60s
}

# When circuit is open, writes are skipped
entity_ids = await writer.store_entities(entities)
if not entity_ids:
    if writer._is_circuit_breaker_open():
        logger.error("Circuit breaker is open, writes paused")
```

## Logging

All operations use structured logging with correlation IDs:

```python
# Enable debug logging
import logging
logging.getLogger("knowledge_engine.integrations.roma_entity_kg_integration").setLevel(logging.DEBUG)

# Logs include:
# - correlation_id for request tracking
# - timestamp in UTC
# - operation details
# - processing times
# - error details
```

## Best Practices

1. **Always check success status**: All operations return success/error status
2. **Use correlation IDs**: Track requests across components
3. **Batch operations**: Store entities in batches for efficiency
4. **Idempotent writes**: Configure idempotent=True for production
5. **Circuit breaker**: Monitor circuit breaker state in production
6. **Async operations**: Always use async/await for I/O operations
7. **Error recovery**: Handle partial failures gracefully

## Performance Considerations

- **Batch Size**: Default 100, adjust based on entity size
- **Timeout**: Default 30 seconds, increase for large graphs
- **Similarity Search**: Uses keyword overlap, can be enhanced with embeddings
- **Tree Traversal**: Max depth limits recursion for safety

## Integration with Existing Code

### With ROMA Integration

```python
from knowledge_engine.integrations.roma_integration import ROMAIntegration
from knowledge_engine.integrations.roma_entity_kg_integration import create_roma_ekg_integration

# Initialize both
roma = ROMAIntegration()
kg = EntityKnowledgeGraph()
extractor, writer, reader = create_roma_ekg_integration(kg)

# Use ROMA as usual
result = await roma.decompose_problem(problem)

# Extract and store knowledge
entities = await extractor.extract_from_decomposition(result.decomposition.__dict__)
await writer.store_entities(entities)
```

### With Master Engine

```python
from knowledge_engine.master_engine import MasterKnowledgeEngine

# Master engine includes EKG integration
master = MasterKnowledgeEngine()

# ROMA integration is auto-connected
roma_result = await master.roma.decompose_problem(problem)

# Knowledge is automatically extracted and stored
kg_result = await master.knowledge_graph.find_similar_entities(problem)
```

## Testing

```python
import pytest
from knowledge_engine.integrations.roma_entity_kg_integration import (
    ROMAEntityExtractor,
    ROMAKnowledgeWriter,
    ROMAKnowledgeReader
)

@pytest.mark.asyncio
async def test_extraction():
    extractor = ROMAEntityExtractor()

    decomposition = {
        "decomposition_id": "test_123",
        "problem": "Test problem",
        "sub_problems": [],
        "is_atomic": True,
        "depth": 0
    }

    entities = await extractor.extract_from_decomposition(decomposition)
    assert len(entities) > 0
    assert entities[0].entity_type == ROMAEntityType.PROBLEM

@pytest.mark.asyncio
async def test_storage():
    kg = EntityKnowledgeGraph()
    writer = ROMAKnowledgeWriter(kg)

    entity = ROMAEntity(
        entity_id="test_entity",
        entity_type=ROMAEntityType.PROBLEM,
        name="Test",
        description="Test description"
    )

    entity_ids = await writer.store_entities([entity])
    assert len(entity_ids) == 1
```

## Troubleshooting

### No entities extracted
- Check decomposition structure matches expected format
- Verify entity type enum values are correct
- Enable debug logging for detailed extraction info

### Circuit breaker open
- Check knowledge graph connectivity
- Verify entity data is valid
- Reset circuit breaker: `writer._reset_circuit_breaker()`

### Similar decompositions not found
- Verify entities are stored in knowledge graph
- Check entity_type values match query
- Adjust similarity_threshold in config

### Performance issues
- Reduce batch_size for smaller writes
- Increase timeout_seconds for large graphs
- Limit max_sub_problems for complex decompositions
