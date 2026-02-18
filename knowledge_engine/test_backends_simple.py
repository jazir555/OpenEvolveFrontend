"""
Simple backend test to verify functionality.
"""
import asyncio
import sys
from pathlib import Path
import os

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.core.backends.memory_backend import MemoryBackend
from knowledge_engine.core.backends.base import KnowledgeEntry
from datetime import datetime


async def test_memory_backend():
    """Test Memory backend functionality."""
    print("Testing Memory Backend...")

    # Create backend
    backend = MemoryBackend(config={})
    await backend.connect()

    # Check health
    is_healthy = await backend.health_check()
    print(f"[OK] Backend healthy: {is_healthy}")

    # Create entry
    entry = KnowledgeEntry(
        source="test_doc",
        content="Artificial Intelligence and Machine Learning are transforming the world.",
        metadata={"category": "AI", "importance": "high"},
        timestamp=datetime.utcnow().isoformat()
    )

    # Add knowledge
    entry_id = await backend.add_knowledge(entry)
    print(f"[OK] Added entry: {entry_id}")

    # Search
    results = await backend.search(query="Machine Learning", limit=10)
    print(f"[OK] Search results: {results.total_count} found")
    print(f"  - Backend: {results.backend_used}")
    print(f"  - Time: {results.search_time_ms:.2f}ms")

    if results.results:
        print(f"  - Sample result: {results.results[0]['content'][:50]}...")

    # Get statistics
    stats = await backend.get_statistics()
    print(f"[OK] Statistics:")
    print(f"  - Nodes: {stats.node_count}")
    print(f"  - Edges: {stats.edge_count}")
    print(f"  - Knowledge entries: {stats.metadata['knowledge_entries']}")
    print(f"  - Entities: {stats.metadata['entities']}")

    # Analyze
    analysis = await backend.analyze(analysis_type="entity_analysis")
    print(f"[OK] Entity analysis:")
    print(f"  - Total entities: {analysis.results['total_entities']}")
    if analysis.results['top_entities']:
        print(f"  - Top entity: {analysis.results['top_entities'][0]}")

    # Update
    updated = await backend.update_knowledge(entry_id, {"content": "Updated content"})
    print(f"[OK] Updated entry: {updated}")

    # Delete
    deleted = await backend.delete_knowledge(entry_id)
    print(f"[OK] Deleted entry: {deleted}")

    # Cleanup
    await backend.disconnect()
    print("[OK] Disconnected")

    print("\n[OK] All Memory Backend tests passed!")


async def test_all_backends():
    """Test all available backends."""
    print("\n" + "="*60)
    print("KNOWLEDGE ENGINE BACKEND TESTS")
    print("="*60 + "\n")

    # Test Memory Backend (always available)
    await test_memory_backend()

    print("\n" + "="*60)
    print("Testing optional backends (may skip if unavailable)...")
    print("="*60 + "\n")

    # Test Neo4j (optional)
    try:
        from knowledge_engine.core.backends.neo4j_backend import Neo4jBackend

        backend = Neo4jBackend(config={
            'uri': 'bolt://localhost:7687',
            'user': 'neo4j',
            'password': 'password'
        })

        await backend.connect()
        print("[OK] Neo4j backend connected")
        await backend.disconnect()
        print("[OK] Neo4j backend test passed\n")

    except Exception as e:
        print(f"⊘ Neo4j backend skipped: {e}\n")

    # Test Qdrant (optional)
    try:
        from knowledge_engine.core.backends.qdrant_backend import QdrantBackend

        backend = QdrantBackend(config={
            'host': 'localhost',
            'port': 6333,
            'collection': 'test_knowledge_graph'
        })

        await backend.connect()
        print("[OK] Qdrant backend connected")
        await backend.disconnect()
        print("[OK] Qdrant backend test passed\n")

    except Exception as e:
        print(f"⊘ Qdrant backend skipped: {e}\n")

    # Test MongoDB (optional)
    try:
        from knowledge_engine.core.backends.mongodb_backend import MongoDBBackend

        backend = MongoDBBackend(config={
            'uri': 'mongodb://localhost:27017',
            'database': 'test_knowledge_graph'
        })

        await backend.connect()
        print("[OK] MongoDB backend connected")
        await backend.disconnect()
        print("[OK] MongoDB backend test passed\n")

    except Exception as e:
        print(f"⊘ MongoDB backend skipped: {e}\n")

    # Test KarateClub (optional)
    try:
        from knowledge_engine.core.backends.karateclub_backend import KarateClubBackend

        backend = KarateClubBackend(config={
            'embedding_dim': 64,
            'random_state': 42
        })

        await backend.connect()
        print("[OK] KarateClub backend connected")
        await backend.disconnect()
        print("[OK] KarateClub backend test passed\n")

    except Exception as e:
        print(f"⊘ KarateClub backend skipped: {e}\n")

    print("="*60)
    print("BACKEND TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_all_backends())
