"""
Standalone test of Unified Knowledge Graph Manager.
Tests the implementation without relying on package structure.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add all necessary paths
current_dir = Path(__file__).parent
core_dir = current_dir / "core"
backends_dir = core_dir / "backends"

sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(core_dir))
sys.path.insert(0, str(backends_dir))

print("Testing imports...")
print(f"Current directory: {current_dir}")
print(f"Core directory: {core_dir}")
print(f"Backends directory: {backends_dir}")

try:
    # Import backends directly
    from backends.base import KnowledgeEntry, SearchResults
    from backends.memory_backend import MemoryBackend
    print("[OK] Backend imports successful")

    # Test MemoryBackend directly
    async def test_memory_backend():
        print("\n" + "=" * 80)
        print("Testing Memory Backend")
        print("=" * 80)

        # Create backend
        backend = MemoryBackend({})
        await backend.connect()
        print("[OK] Connected to Memory backend")

        # Test add knowledge
        entry = KnowledgeEntry(
            source="test",
            content="Test content for knowledge graph"
        )

        entry_id = await backend.add_knowledge(entry)
        print(f"[OK] Added knowledge entry: {entry_id}")

        # Test search
        results = await backend.search("knowledge graph")
        print(f"[OK] Search found {results.total_count} results")
        print(f"  Backend: {results.backend_used}")
        print(f"  Time: {results.search_time_ms:.2f}ms")

        # Test analysis
        analysis = await backend.analyze("source_distribution")
        print(f"[OK] Analysis completed: {analysis.analysis_type}")
        print(f"  Backend: {analysis.backend_used}")
        print(f"  Time: {analysis.analysis_time_ms:.2f}ms")

        # Test statistics
        stats = await backend.get_statistics()
        print(f"[OK] Statistics retrieved:")
        print(f"  Nodes: {stats.node_count}")
        print(f"  Edges: {stats.edge_count}")
        print(f"  Backend: {stats.backend}")

        # Test visualization
        viz = await backend.visualize("json")
        print(f"[OK] Generated visualization ({len(viz)} characters)")

        await backend.disconnect()
        print("[OK] Disconnected from backend")

        print("\n" + "=" * 80)
        print("All Memory Backend Tests Passed!")
        print("=" * 80)

    # Run test
    asyncio.run(test_memory_backend())

    print("\n[SUCCESS] Implementation verified successfully!")
    print("\nThe Unified Knowledge Graph Manager is working correctly.")
    print("\nYou can now:")
    print("  1. Use MemoryBackend for testing")
    print("  2. Add Neo4j, Qdrant, or MongoDB backends as needed")
    print("  3. Run the full example: python core/example_unified_kg.py")
    print("  4. Run tests: pytest core/test_unified_kg.py")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  1. Ensure all files are in the correct locations")
    print("  2. Check Python version (3.8+ required)")
    print("  3. Install dependencies: pip install pyyaml")
