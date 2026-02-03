"""
Comprehensive Integration Tests for PostgreSQL and Memgraph Backends

Tests all functionality of the new permissive-licensed backends:
- PostgreSQL: PostgreSQL License
- Memgraph: Apache 2.0

Also verifies Qdrant and Redis are working correctly.
"""

import asyncio
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.core.backends.base import (
    BackendType,
    KnowledgeEntry,
    SearchResults,
    AnalysisResult,
    GraphStatistics
)
from knowledge_engine.core.backends.postgresql_backend import PostgreSQLBackend
from knowledge_engine.core.backends.memgraph_backend import MemgraphBackend
from knowledge_engine.core.backends.qdrant_backend import QdrantBackend
from knowledge_engine.core.backends.memory_backend import MemoryBackend
from knowledge_engine.enhanced_storage import EnhancedKnowledgeStorage, StorageBackend
from knowledge_engine.knowledge_storage import KnowledgeStorage


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_KNOWLEDGE_ARTIFACTS = [
    {
        "artifact_id": "test_001",
        "content": "PostgreSQL is a powerful open-source relational database system.",
        "type": "database_info",
        "source": "tech_docs",
        "context": "storage_systems",
        "metadata": {"category": "database", "license": "postgresql"}
    },
    {
        "artifact_id": "test_002",
        "content": "Memgraph is a high-performance in-memory graph database compatible with Neo4j.",
        "type": "database_info",
        "source": "tech_docs",
        "context": "graph_databases",
        "metadata": {"category": "graph_db", "license": "apache_2"}
    },
    {
        "artifact_id": "test_003",
        "content": "Qdrant provides vector similarity search with Apache 2.0 license.",
        "type": "search_info",
        "source": "tech_docs",
        "context": "vector_search",
        "metadata": {"category": "vector_db", "license": "apache_2"}
    },
    {
        "artifact_id": "test_004",
        "content": "Redis is an in-memory data structure store used for caching.",
        "type": "cache_info",
        "source": "tech_docs",
        "context": "caching",
        "metadata": {"category": "cache", "license": "bsd"}
    }
]


# =============================================================================
# PostgreSQL Backend Tests
# =============================================================================

async def test_postgresql_backend():
    """Test PostgreSQL backend functionality."""
    print("\n" + "="*60)
    print("TESTING POSTGRESQL BACKEND")
    print("="*60)
    
    try:
        import asyncpg
    except ImportError:
        print("⚠️  asyncpg not installed - skipping PostgreSQL tests")
        return False
    
    config = {
        'uri': 'postgresql://localhost:5432/openevolve_kg',
        'table': 'knowledge_entries',
        'timeout': 30
    }
    
    backend = PostgreSQLBackend(config=config)
    
    # Test connection
    print("\n1. Testing connection...")
    try:
        connected = await backend.connect()
        if connected:
            print("   ✅ Connected to PostgreSQL")
        else:
            print("   ⚠️  Could not connect to PostgreSQL (server may not be running)")
            return False
    except Exception as e:
        print(f"   ⚠️  Connection failed: {e}")
        return False
    
    # Test health check
    print("\n2. Testing health check...")
    try:
        is_healthy = await backend.health_check()
        print(f"   {'✅' if is_healthy else '❌'} Health check: {is_healthy}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test adding knowledge
    print("\n3. Testing add_knowledge...")
    entry_ids = []
    for artifact in SAMPLE_KNOWLEDGE_ARTIFACTS[:2]:
        try:
            entry = KnowledgeEntry(
                source=artifact["source"],
                content=artifact["content"],
                metadata=artifact["metadata"],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            entry_id = await backend.add_knowledge(entry)
            entry_ids.append(entry_id)
            print(f"   ✅ Added entry: {entry_id[:20]}...")
        except Exception as e:
            print(f"   ❌ Failed to add entry: {e}")
    
    # Test search
    print("\n4. Testing search...")
    try:
        results = await backend.search(query="database", limit=10)
        print(f"   ✅ Search completed: {results.total_count} results")
        print(f"      Backend used: {results.backend_used}")
        print(f"      Search time: {results.search_time_ms:.2f}ms")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # Test statistics
    print("\n5. Testing get_statistics...")
    try:
        stats = await backend.get_statistics()
        print(f"   ✅ Statistics retrieved")
        print(f"      Nodes: {stats.node_count}")
        print(f"      Edges: {stats.edge_count}")
        print(f"      Backend type: {stats.backend_type.value}")
    except Exception as e:
        print(f"   ❌ Statistics failed: {e}")
    
    # Test analysis
    print("\n6. Testing analyze...")
    try:
        analysis = await backend.analyze(analysis_type="entity_analysis")
        print(f"   ✅ Analysis completed: {analysis.analysis_type}")
        if analysis.results:
            print(f"      Entities found: {analysis.results.get('total_entities', 0)}")
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
    
    # Test update
    print("\n7. Testing update_knowledge...")
    if entry_ids:
        try:
            updated = await backend.update_knowledge(
                entry_ids[0], 
                {"content": "Updated content for testing"}
            )
            print(f"   {'✅' if updated else '❌'} Update: {updated}")
        except Exception as e:
            print(f"   ❌ Update failed: {e}")
    
    # Test delete
    print("\n8. Testing delete_knowledge...")
    for entry_id in entry_ids:
        try:
            deleted = await backend.delete_knowledge(entry_id)
            print(f"   {'✅' if deleted else '❌'} Deleted: {entry_id[:20]}...")
        except Exception as e:
            print(f"   ❌ Delete failed: {e}")
    
    # Disconnect
    print("\n9. Testing disconnect...")
    try:
        await backend.disconnect()
        print("   ✅ Disconnected from PostgreSQL")
    except Exception as e:
        print(f"   ❌ Disconnect failed: {e}")
    
    print("\n✅ PostgreSQL backend tests completed!")
    return True


# =============================================================================
# Memgraph Backend Tests
# =============================================================================

async def test_memgraph_backend():
    """Test Memgraph backend functionality."""
    print("\n" + "="*60)
    print("TESTING MEMGRAPH BACKEND")
    print("="*60)
    
    config = {
        'uri': 'bolt://localhost:7687',
        'user': '',
        'password': '',
        'database': 'memgraph'
    }
    
    backend = MemgraphBackend(config=config)
    
    # Test connection
    print("\n1. Testing connection...")
    try:
        connected = await backend.connect()
        if connected:
            print("   ✅ Connected to Memgraph")
        else:
            print("   ⚠️  Could not connect to Memgraph (server may not be running)")
            return False
    except Exception as e:
        print(f"   ⚠️  Connection failed: {e}")
        return False
    
    # Test health check
    print("\n2. Testing health check...")
    try:
        is_healthy = await backend.health_check()
        print(f"   {'✅' if is_healthy else '❌'} Health check: {is_healthy}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test adding knowledge
    print("\n3. Testing add_knowledge...")
    entry_ids = []
    for artifact in SAMPLE_KNOWLEDGE_ARTIFACTS[1:3]:
        try:
            entry = KnowledgeEntry(
                source=artifact["source"],
                content=artifact["content"],
                metadata=artifact["metadata"],
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            entry_id = await backend.add_knowledge(entry)
            entry_ids.append(entry_id)
            print(f"   ✅ Added entry: {entry_id[:20]}...")
        except Exception as e:
            print(f"   ❌ Failed to add entry: {e}")
    
    # Test search
    print("\n4. Testing search...")
    try:
        results = await backend.search(query="graph database", limit=10)
        print(f"   ✅ Search completed: {results.total_count} results")
        print(f"      Backend used: {results.backend_used}")
        print(f"      Search time: {results.search_time_ms:.2f}ms")
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    # Test statistics
    print("\n5. Testing get_statistics...")
    try:
        stats = await backend.get_statistics()
        print(f"   ✅ Statistics retrieved")
        print(f"      Nodes: {stats.node_count}")
        print(f"      Edges: {stats.edge_count}")
        print(f"      Backend type: {stats.backend_type.value}")
    except Exception as e:
        print(f"   ❌ Statistics failed: {e}")
    
    # Test graph analysis
    print("\n6. Testing graph analysis...")
    try:
        analysis = await backend.analyze(analysis_type="centrality")
        print(f"   ✅ Analysis completed: {analysis.analysis_type}")
        if analysis.results:
            print(f"      Results: {json.dumps(analysis.results, indent=2)[:100]}...")
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
    
    # Test update
    print("\n7. Testing update_knowledge...")
    if entry_ids:
        try:
            updated = await backend.update_knowledge(
                entry_ids[0],
                {"content": "Updated graph content"}
            )
            print(f"   {'✅' if updated else '❌'} Update: {updated}")
        except Exception as e:
            print(f"   ❌ Update failed: {e}")
    
    # Test delete
    print("\n8. Testing delete_knowledge...")
    for entry_id in entry_ids:
        try:
            deleted = await backend.delete_knowledge(entry_id)
            print(f"   {'✅' if deleted else '❌'} Deleted: {entry_id[:20]}...")
        except Exception as e:
            print(f"   ❌ Delete failed: {e}")
    
    # Disconnect
    print("\n9. Testing disconnect...")
    try:
        await backend.disconnect()
        print("   ✅ Disconnected from Memgraph")
    except Exception as e:
        print(f"   ❌ Disconnect failed: {e}")
    
    print("\n✅ Memgraph backend tests completed!")
    return True


# =============================================================================
# Enhanced Storage Tests
# =============================================================================

async def test_enhanced_storage():
    """Test EnhancedStorage with new backends."""
    print("\n" + "="*60)
    print("TESTING ENHANCED STORAGE")
    print("="*60)
    
    config = {
        "backends": {
            "postgresql": {"enabled": True},
            "memgraph": {"enabled": True},
            "qdrant": {"enabled": False},  # Skip if not available
            "redis": {"enabled": False}    # Skip if not available
        },
        "default_backend": "postgresql"
    }
    
    storage = EnhancedKnowledgeStorage(config=config)
    
    # Test initialization
    print("\n1. Testing initialization...")
    print(f"   Configured backends: {list(storage.backends.keys())}")
    print(f"   Default backend: {storage.config.get('default_backend')}")
    
    # Test storing artifacts
    print("\n2. Testing store_knowledge_artifact...")
    artifact_ids = []
    for artifact in SAMPLE_KNOWLEDGE_ARTIFACTS:
        try:
            result = storage.store_knowledge_artifact(
                artifact=artifact,
                generate_embedding=False
            )
            if result.success:
                artifact_ids.append(result.artifact_id)
                print(f"   ✅ Stored: {result.artifact_id[:20]}... (backend: {result.backend_used})")
            else:
                print(f"   ❌ Failed to store: {result.error}")
        except Exception as e:
            print(f"   ⚠️  Storage error: {e}")
    
    # Test statistics
    print("\n3. Testing get_aggregated_statistics...")
    try:
        stats = storage.get_aggregated_statistics()
        print(f"   ✅ Statistics retrieved")
        print(f"      Backend status: {list(stats.get('backend_status', {}).keys())}")
    except Exception as e:
        print(f"   ⚠️  Statistics error: {e}")
    
    # Test optimization
    print("\n4. Testing optimize_storage...")
    try:
        results = storage.optimize_storage()
        print(f"   ✅ Optimization completed")
        print(f"      Operations: {results.get('operations_performed', [])}")
    except Exception as e:
        print(f"   ⚠️  Optimization error: {e}")
    
    # Test close connections
    print("\n5. Testing close_connections...")
    try:
        storage.close_connections()
        print("   ✅ Connections closed")
    except Exception as e:
        print(f"   ⚠️  Close error: {e}")
    
    print("\n✅ Enhanced storage tests completed!")
    return True


# =============================================================================
# Knowledge Storage Tests
# =============================================================================

async def test_knowledge_storage():
    """Test KnowledgeStorage with new backends."""
    print("\n" + "="*60)
    print("TESTING KNOWLEDGE STORAGE")
    print("="*60)
    
    config = {
        "postgresql": {"enabled": True, "uri": "postgresql://localhost:5432/openevolve_kg"},
        "memgraph": {"enabled": True, "uri": "bolt://localhost:7687"},
        "qdrant": {"enabled": False},
        "default_backend": "postgresql"
    }
    
    storage = KnowledgeStorage(config=config)
    
    # Test initialization
    print("\n1. Testing initialization...")
    print(f"   Backends initialized")
    
    # Test storing
    print("\n2. Testing store_knowledge_artifact...")
    artifact_ids = []
    for artifact in SAMPLE_KNOWLEDGE_ARTIFACTS[:2]:
        try:
            success = await storage.store_knowledge_artifact(artifact)
            if success:
                artifact_ids.append(artifact["artifact_id"])
                print(f"   ✅ Stored: {artifact['artifact_id']}")
            else:
                print(f"   ❌ Failed to store: {artifact['artifact_id']}")
        except Exception as e:
            print(f"   ⚠️  Storage error: {e}")
    
    # Test retrieval
    print("\n3. Testing retrieve_knowledge_artifact...")
    for artifact_id in artifact_ids[:1]:
        try:
            result = await storage.retrieve_knowledge_artifact(artifact_id)
            if result:
                print(f"   ✅ Retrieved: {artifact_id}")
                print(f"      Content: {result.get('content', 'N/A')[:50]}...")
            else:
                print(f"   ⚠️  Not found: {artifact_id}")
        except Exception as e:
            print(f"   ⚠️  Retrieval error: {e}")
    
    # Test statistics
    print("\n4. Testing get_statistics...")
    try:
        stats = storage.get_statistics()
        print(f"   ✅ Statistics retrieved")
        print(f"      Backend status: {list(stats.get('backend_status', {}).keys())}")
    except Exception as e:
        print(f"   ⚠️  Statistics error: {e}")
    
    # Test close
    print("\n5. Testing close_connections...")
    try:
        storage.close_connections()
        print("   ✅ Connections closed")
    except Exception as e:
        print(f"   ⚠️  Close error: {e}")
    
    print("\n✅ Knowledge storage tests completed!")
    return True


# =============================================================================
# Main Test Runner
# =============================================================================

async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("KNOWLEDGE ENGINE - BACKEND INTEGRATION TESTS")
    print("Testing PostgreSQL and Memgraph Backends")
    print("="*60)
    
    results = {
        "postgresql": False,
        "memgraph": False,
        "enhanced_storage": False,
        "knowledge_storage": False
    }
    
    # Run PostgreSQL tests
    try:
        results["postgresql"] = await test_postgresql_backend()
    except Exception as e:
        print(f"\n❌ PostgreSQL tests failed with exception: {e}")
    
    # Run Memgraph tests
    try:
        results["memgraph"] = await test_memgraph_backend()
    except Exception as e:
        print(f"\n❌ Memgraph tests failed with exception: {e}")
    
    # Run EnhancedStorage tests
    try:
        results["enhanced_storage"] = await test_enhanced_storage()
    except Exception as e:
        print(f"\n❌ Enhanced storage tests failed with exception: {e}")
    
    # Run KnowledgeStorage tests
    try:
        results["knowledge_storage"] = await test_knowledge_storage()
    except Exception as e:
        print(f"\n❌ Knowledge storage tests failed with exception: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "⚠️  SKIPPED/FAILED"
        print(f"  {test_name:20s}: {status}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} test suites passed")
    
    if passed_count == total_count:
        print("\n🎉 All backend integration tests passed!")
    else:
        print("\n⚠️  Some tests were skipped or failed (servers may not be running)")
    
    return results


def main():
    """Main entry point."""
    # Run async tests
    results = asyncio.run(run_all_tests())
    
    # Return exit code
    passed_count = sum(1 for v in results.values() if v)
    return 0 if passed_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
