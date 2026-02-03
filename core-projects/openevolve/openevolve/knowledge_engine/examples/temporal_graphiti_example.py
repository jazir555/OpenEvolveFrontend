"""
Example: Using Temporal Knowledge Engine with Graphiti

This example demonstrates how to use the TemporalKnowledgeEngine
with Graphiti integration for temporal reasoning, hybrid search,
and contradiction detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from knowledge_engine.core.temporal_knowledge_engine import (
    TemporalKnowledgeEngine,
    KnowledgeArtifact,
    RerankMethod,
)
from knowledge_engine.integrations.graphiti_temporal_bridge import (
    GraphitiTemporalBridge,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_temporal_knowledge():
    """Example 1: Basic temporal knowledge tracking."""
    print("\n" + "="*80)
    print("Example 1: Basic Temporal Knowledge Tracking")
    print("="*80 + "\n")

    # Create temporal knowledge engine
    engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
    )

    # Current time
    now = datetime.utcnow()

    # Add knowledge that evolves over time
    print("Adding evolving knowledge...")

    # Initial understanding
    await engine.add_knowledge_temporal(
        content="Python's async/await was introduced in Python 3.5",
        artifact_type="solution_pattern",
        valid_at=now - timedelta(days=30),
        metadata={"language": "python", "topic": "async"},
    )

    # Updated understanding
    await engine.add_knowledge_temporal(
        content="Python 3.11 improved async performance significantly with task groups",
        artifact_type="solution_pattern",
        valid_at=now - timedelta(days=7),
        metadata={"language": "python", "topic": "async"},
    )

    # Latest understanding
    await engine.add_knowledge_temporal(
        content="Python 3.12 introduced enhanced error messages for async code",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"language": "python", "topic": "async"},
    )

    print("Knowledge added successfully!\n")

    # Query at different points in time
    print("Querying knowledge at different time points...")

    past_time = now - timedelta(days=20)
    recent_time = now - timedelta(days=3)
    current_time = now

    past_results = await engine.query_at_time("python async", past_time)
    recent_results = await engine.query_at_time("python async", recent_time)
    current_results = await engine.query_at_time("python async", current_time)

    print(f"\nResults at {past_time.date()}: {len(past_results)} artifacts")
    for artifact in past_results:
        print(f"  - {artifact.content[:80]}...")

    print(f"\nResults at {recent_time.date()}: {len(recent_results)} artifacts")
    for artifact in recent_results:
        print(f"  - {artifact.content[:80]}...")

    print(f"\nResults at {current_time.date()}: {len(current_results)} artifacts")
    for artifact in current_results:
        print(f"  - {artifact.content[:80]}...")


async def example_hybrid_search():
    """Example 2: Hybrid search with different reranking methods."""
    print("\n" + "="*80)
    print("Example 2: Hybrid Search with Different Reranking Methods")
    print("="*80 + "\n")

    engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
        default_rerank_method=RerankMethod.RRF,
    )

    now = datetime.utcnow()

    # Add diverse knowledge
    print("Adding diverse knowledge artifacts...")

    await engine.add_knowledge_temporal(
        content="Vector databases use embeddings for semantic search",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"domain": "databases", "search_type": "semantic"},
    )

    await engine.add_knowledge_temporal(
        content="BM25 algorithm improves keyword search with term frequency",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"domain": "search", "search_type": "keyword"},
    )

    await engine.add_knowledge_temporal(
        content="Graph databases traverse relationships for contextual search",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"domain": "databases", "search_type": "graph"},
    )

    await engine.add_knowledge_temporal(
        content="Hybrid search combines vector, keyword, and graph traversal",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"domain": "search", "search_type": "hybrid"},
    )

    print("Knowledge added!\n")

    # Try different reranking methods
    query = "semantic search optimization"

    print(f"Searching for: '{query}'\n")

    for rerank_method in [RerankMethod.RRF, RerankMethod.CROSS_ENCODER]:
        print(f"Using {rerank_method.value} reranking...")

        results = await engine.search_with_graphiti(
            query=query,
            use_hybrid=True,
            rerank_method=rerank_method.value,
            max_results=3,
        )

        print(f"  Found {len(results)} results:")
        for i, artifact in enumerate(results, 1):
            print(f"    {i}. {artifact.content[:70]}...")
        print()


async def example_contradaction_detection():
    """Example 3: Contradiction detection."""
    print("\n" + "="*80)
    print("Example 3: Contradiction Detection")
    print("="*80 + "\n")

    engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
    )

    now = datetime.utcnow()

    # Add potentially contradictory knowledge
    print("Adding potentially contradictory knowledge...")

    await engine.add_knowledge_temporal(
        content="PostgreSQL cannot handle more than 1000 concurrent connections",
        artifact_type="problem",
        valid_at=now,
        metadata={"database": "postgresql", "topic": "scalability"},
    )

    await engine.add_knowledge_temporal(
        content="PostgreSQL can handle unlimited concurrent connections with proper pooling",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"database": "postgresql", "topic": "scalability"},
    )

    await engine.add_knowledge_temporal(
        content="Redis is not suitable for persistent storage",
        artifact_type="problem",
        valid_at=now,
        metadata={"database": "redis", "topic": "persistence"},
    )

    await engine.add_knowledge_temporal(
        content="Redis provides persistence options including RDB and AOF",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"database": "redis", "topic": "persistence"},
    )

    print("Knowledge added!\n")

    # Detect contradictions
    print("Detecting contradictions...")

    result = await engine.detect_contradictions()

    print(f"\nContradictions found: {result.has_contradictions}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Number of potential issues: {len(result.contradictions)}")

    if result.contradictions:
        print("\nPotential contradictions:")
        for i, contradiction in enumerate(result.contradictions, 1):
            print(f"\n  {i}. Type: {contradiction.get('type', 'unknown')}")
            print(f"     Severity: {contradiction.get('severity', 'unknown')}")
            print(f"     Reason: {contradiction.get('reason', 'N/A')}")


async def example_timeline_reconstruction():
    """Example 4: Timeline reconstruction for an entity."""
    print("\n" + "="*80)
    print("Example 4: Timeline Reconstruction")
    print("="*80 + "\n")

    engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
    )

    # Create timeline
    now = datetime.utcnow()
    timeline_start = now - timedelta(days=30)

    print(f"Reconstructing timeline for entity: 'Python'")

    # Add knowledge over time
    events = [
        (timeline_start, "Python 3.10 released with pattern matching"),
        (timeline_start + timedelta(days=7), "Adopters report improved code readability"),
        (timeline_start + timedelta(days=14), "Some compatibility issues discovered with older libraries"),
        (timeline_start + timedelta(days=21), "Library maintainers release compatibility updates"),
        (now, "Python 3.10 adoption reaches 40% of projects"),
    ]

    for timestamp, content in events:
        await engine.add_knowledge_temporal(
            content=content,
            artifact_type="workflow",
            valid_at=timestamp,
            metadata={"entity": "Python", "version": "3.10"},
            entities=["Python"],
        )

    print(f"Added {len(events)} timeline events\n")

    # Get timeline
    timeline = await engine.get_timeline(
        entity="Python",
        start_time=timeline_start,
        end_time=now,
    )

    print(f"Timeline events: {len(timeline)}\n")
    for event in timeline:
        print(f"  [{event['timestamp'].date()}] {event['description']}")
        print(f"    Type: {event['event_type']}, Source: {event['source']}")


async def example_knowledge_invalidation():
    """Example 5: Knowledge invalidation over time."""
    print("\n" + "="*80)
    print("Example 5: Knowledge Invalidation")
    print("="*80 + "\n")

    engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
    )

    now = datetime.utcnow()

    # Add knowledge that becomes outdated
    print("Adding time-sensitive knowledge...")

    artifact = await engine.add_knowledge_temporal(
        content="Always use requests.get() for synchronous HTTP calls",
        artifact_type="solution_pattern",
        valid_at=now - timedelta(days=10),
        metadata={"topic": "http", "language": "python"},
    )

    print(f"Knowledge added at {artifact.valid_at}")
    print(f"Content: {artifact.content}")

    # Later, invalidate this knowledge
    invalidation_time = now

    print(f"\nInvalidating knowledge at {invalidation_time}...")

    success = await engine.invalidate_knowledge(
        artifact_id=artifact.id,
        invalid_at=invalidation_time,
    )

    if success:
        updated = await engine.get_artifact(artifact.id)
        print(f"Invalidation successful!")
        print(f"  Valid until: {updated.invalid_at}")

        # Check validity at different times
        before = invalidation_time - timedelta(hours=1)
        after = invalidation_time + timedelta(hours=1)

        print(f"\nChecking validity:")
        print(f"  Before invalidation: {updated.is_valid_at(before)}")
        print(f"  After invalidation: {updated.is_valid_at(after)}")


async def example_bridge_integration():
    """Example 6: Using GraphitiTemporalBridge directly."""
    print("\n" + "="*80)
    print("Example 6: GraphitiTemporalBridge Integration")
    print("="*80 + "\n")

    # Create bridge (without initializing Graphiti for this example)
    bridge = GraphitiTemporalBridge()

    # Create a knowledge artifact
    now = datetime.utcnow()
    artifact = KnowledgeArtifact(
        id="bridge_example_001",
        content="Docker containers provide isolated runtime environments",
        artifact_type="solution_pattern",
        valid_at=now,
        metadata={"topic": "containers", "technology": "docker"},
        entities=["Docker", "containers"],
        confidence=0.95,
    )

    print("Created KnowledgeArtifact:")
    print(f"  ID: {artifact.id}")
    print(f"  Type: {artifact.artifact_type}")
    print(f"  Content: {artifact.content}")
    print(f"  Entities: {', '.join(artifact.entities)}")

    # Convert to episode
    episode = await bridge.artifact_to_episode(artifact)

    print("\nConverted to Graphiti Episode:")
    print(f"  Name: {episode['name']}")
    print(f"  Body: {episode['body']}")
    print(f"  Graphiti Type: {episode['metadata']['graphiti_type']}")
    print(f"  Artifact ID: {episode['metadata']['artifact_id']}")

    # Show entity mappings
    print("\nEntity Type Mappings:")
    mappings = bridge.get_entity_type_mappings()
    for mapping in mappings[:5]:  # Show first 5
        print(f"  {mapping.ke_type:20} -> {mapping.graphiti_type}")


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("TEMPORAL KNOWLEDGE ENGINE - GRAPHITI INTEGRATION EXAMPLES")
    print("="*80)

    try:
        await example_basic_temporal_knowledge()
        await example_hybrid_search()
        await example_contradaction_detection()
        await example_timeline_reconstruction()
        await example_knowledge_invalidation()
        await example_bridge_integration()

        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
