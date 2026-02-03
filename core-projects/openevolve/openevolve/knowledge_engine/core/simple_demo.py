"""
Simple demonstration of Unified Knowledge Graph Manager.
Run this to verify the implementation works correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_engine.core.unified_knowledge_graph import UnifiedKnowledgeGraph


async def demo():
    """Simple demonstration"""
    print("=" * 80)
    print("Unified Knowledge Graph Manager - Simple Demo")
    print("=" * 80)

    # Create manager with memory backend (no external dependencies)
    kg = UnifiedKnowledgeGraph()

    print("\n1. Connecting to backends...")
    await kg.connect_all()
    health = await kg.health_check()
    print(f"   ✓ Connected to {len(health)} backend(s): {list(health.keys())}")

    try:
        print("\n2. Adding knowledge...")
        entry_id = await kg.add_knowledge(
            source="demo",
            content="The Unified Knowledge Graph Manager provides a consistent interface across multiple backends.",
            metadata={"tags": ["demo", "introduction"], "priority": "high"}
        )
        print(f"   ✓ Added entry: {entry_id}")

        # Add more knowledge
        await kg.add_knowledge(
            source="neo4j",
            content="Neo4j is a native graph database with efficient relationship traversal.",
            metadata={"type": "technical", "tags": ["graph", "database"]}
        )

        await kg.add_knowledge(
            source="qdrant",
            content="Qdrant provides high-performance vector similarity search.",
            metadata={"type": "technical", "tags": ["vector", "search"]}
        )

        await kg.add_knowledge(
            source="mongodb",
            content="MongoDB offers flexible document storage with rich querying capabilities.",
            metadata={"type": "technical", "tags": ["document", "nosql"]}
        )

        print(f"   ✓ Added 4 knowledge entries")

        print("\n3. Searching knowledge...")
        results = await kg.search("graph database")
        print(f"   ✓ Found {results.total_count} results in {results.search_time_ms:.2f}ms")
        for i, result in enumerate(results.results[:3], 1):
            print(f"     {i}. [{result['source']}] {result['content'][:60]}...")

        print("\n4. Analyzing graph...")
        analysis = await kg.analyze("source_distribution")
        print(f"   ✓ Analysis type: {analysis.analysis_type}")
        print(f"   ✓ Backend used: {analysis.backend_used}")
        print(f"   ✓ Analysis time: {analysis.analysis_time_ms:.2f}ms")

        print("\n5. Getting statistics...")
        stats = await kg.get_graph_stats()
        for backend_name, backend_stats in stats["backends"].items():
            print(f"   ✓ {backend_name}:")
            print(f"     - Nodes: {backend_stats['node_count']}")
            print(f"     - Edges: {backend_stats['edge_count']}")

        print("\n6. Generating visualization...")
        viz = await kg.visualize("json")
        print(f"   ✓ Generated JSON visualization ({len(viz)} characters)")

        print("\n7. Batch operations...")
        entries = [
            {"source": "batch1", "content": "Batch entry 1"},
            {"source": "batch2", "content": "Batch entry 2"},
            {"source": "batch3", "content": "Batch entry 3"},
        ]
        ids = await kg.batch_add_knowledge(entries)
        print(f"   ✓ Batch added {len(ids)} entries")

        print("\n" + "=" * 80)
        print("✓ Demo completed successfully!")
        print("=" * 80)

        print("\nKey Features Demonstrated:")
        print("  ✓ Backend initialization and connection")
        print("  ✓ Adding knowledge with metadata")
        print("  ✓ Searching with relevance ranking")
        print("  ✓ Graph analysis and statistics")
        print("  ✓ Visualization generation")
        print("  ✓ Batch operations")
        print("  ✓ Health monitoring")
        print("\nThe Unified Knowledge Graph Manager is working correctly!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n8. Cleaning up...")
        await kg.disconnect_all()
        print("   ✓ Disconnected from all backends")


if __name__ == "__main__":
    asyncio.run(demo())
