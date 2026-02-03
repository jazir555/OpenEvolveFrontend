"""
Backward Compatibility Test for Knowledge Engine

Verifies that external projects depending on the knowledge engine
will not break after the MongoDB/Neo4j removal.

This test simulates what external projects might do and ensures
backward compatibility is maintained.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_mongodb_backend():
    """Test that external projects can still import MongoDBBackend."""
    print("\n" + "="*60)
    print("TEST 1: Import MongoDBBackend (backward compatibility)")
    print("="*60)
    
    tests = [
        ("Direct import", "from knowledge_engine.core.backends.mongodb_backend import MongoDBBackend"),
        ("From backends module", "from knowledge_engine.core.backends import MongoDBBackend"),
    ]
    
    passed = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  [OK] {name}: {import_stmt}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
    
    # Test instantiation
    try:
        from knowledge_engine.core.backends.mongodb_backend import MongoDBBackend
        backend = MongoDBBackend(config={"uri": "mongodb://localhost:27017"})
        print(f"  [OK] MongoDBBackend instantiated: {backend.backend_type}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] MongoDBBackend instantiation: {e}")
    
    print(f"\n  Result: {passed}/{len(tests)+1} tests passed")
    return passed == len(tests) + 1


def test_import_neo4j_backend():
    """Test that external projects can still import Neo4jBackend."""
    print("\n" + "="*60)
    print("TEST 2: Import Neo4jBackend (backward compatibility)")
    print("="*60)
    
    tests = [
        ("Direct import", "from knowledge_engine.core.backends.neo4j_backend import Neo4jBackend"),
        ("From backends module", "from knowledge_engine.core.backends import Neo4jBackend"),
    ]
    
    passed = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  [OK] {name}: {import_stmt}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
    
    # Test instantiation
    try:
        from knowledge_engine.core.backends.neo4j_backend import Neo4jBackend
        backend = Neo4jBackend(config={"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"})
        print(f"  [OK] Neo4jBackend instantiated: {backend.backend_type}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] Neo4jBackend instantiation: {e}")
    
    print(f"\n  Result: {passed}/{len(tests)+1} tests passed")
    return passed == len(tests) + 1


def test_backend_type_enum():
    """Test that BackendType enum has all expected values."""
    print("\n" + "="*60)
    print("TEST 3: BackendType enum backward compatibility")
    print("="*60)
    
    from knowledge_engine.core.backends.base import BackendType
    
    # Test active backends
    active_backends = [
        "POSTGRESQL", "MEMGRAPH", "QDRANT", "REDIS", "KARATECLUB", "MEMORY"
    ]
    
    # Test orphaned backends (for backward compatibility)
    orphaned_backends = [
        "NEO4J", "MONGODB"
    ]
    
    print("\n  Active backends:")
    passed = 0
    for backend in active_backends:
        if hasattr(BackendType, backend):
            value = getattr(BackendType, backend).value
            print(f"    [OK] BackendType.{backend} = '{value}'")
            passed += 1
        else:
            print(f"    [FAIL] BackendType.{backend} NOT FOUND")
    
    print("\n  Orphaned backends (backward compatibility):")
    for backend in orphaned_backends:
        if hasattr(BackendType, backend):
            value = getattr(BackendType, backend).value
            print(f"    [OK] BackendType.{backend} = '{value}' (deprecated/orphaned)")
            passed += 1
        else:
            print(f"    [FAIL] BackendType.{backend} NOT FOUND (breaking change!)")
    
    total = len(active_backends) + len(orphaned_backends)
    print(f"\n  Result: {passed}/{total} enum values present")
    return passed == total


def test_backend_all_exports():
    """Test that __all__ includes orphaned backends."""
    print("\n" + "="*60)
    print("TEST 4: backends.__all__ exports")
    print("="*60)
    
    from knowledge_engine.core.backends import __all__
    
    expected_active = [
        "PostgreSQLBackend", "MemgraphBackend", "QdrantBackend",
        "KarateClubBackend", "MemoryBackend"
    ]
    
    expected_orphaned = [
        "Neo4jBackend", "MongoDBBackend"
    ]
    
    print(f"\n  __all__ contents: {__all__}")
    
    print("\n  Active backends in __all__:")
    passed = 0
    for backend in expected_active:
        if backend in __all__:
            print(f"    [OK] {backend} in __all__")
            passed += 1
        else:
            print(f"    [FAIL] {backend} NOT in __all__")
    
    print("\n  Orphaned backends in __all__ (backward compatibility):")
    for backend in expected_orphaned:
        if backend in __all__:
            print(f"    [OK] {backend} in __all__ (deprecated/orphaned)")
            passed += 1
        else:
            print(f"    [FAIL] {backend} NOT in __all__ (breaking change!)")
    
    total = len(expected_active) + len(expected_orphaned)
    print(f"\n  Result: {passed}/{total} backends exported")
    return passed == total


def test_external_project_scenario():
    """Test a typical external project usage scenario."""
    print("\n" + "="*60)
    print("TEST 5: External Project Usage Scenario")
    print("="*60)
    
    print("\n  Simulating external project imports...")
    
    # Scenario 1: Project uses multiple backends
    try:
        from knowledge_engine.core.backends import (
            BackendType,
            PostgreSQLBackend,
            MemgraphBackend,
            MongoDBBackend,  # For legacy compatibility
            Neo4jBackend,    # For legacy compatibility
        )
        print("    [OK] Mixed import of active + orphaned backends")
    except Exception as e:
        print(f"    [FAIL] Mixed import failed: {e}")
        return False
    
    # Scenario 2: Project checks backend type
    try:
        if BackendType.MONGODB.value == "mongodb":
            print("    [OK] BackendType.MONGODB.value accessible")
    except Exception as e:
        print(f"    [FAIL] BackendType.MONGODB.value failed: {e}")
        return False
    
    try:
        if BackendType.NEO4J.value == "neo4j":
            print("    [OK] BackendType.NEO4J.value accessible")
    except Exception as e:
        print(f"    [FAIL] BackendType.NEO4J.value failed: {e}")
        return False
    
    # Scenario 3: Project creates backend instances
    try:
        pg = PostgreSQLBackend(config={"uri": "postgresql://localhost/db"})
        mg = MemgraphBackend(config={"uri": "bolt://localhost:7687"})
        mongo = MongoDBBackend(config={"uri": "mongodb://localhost:27017"})
        neo4j = Neo4jBackend(config={"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"})
        print("    [OK] All backend instances created")
    except Exception as e:
        print(f"    [FAIL] Backend instantiation failed: {e}")
        return False
    
    # Scenario 4: Project switches on backend type
    try:
        def get_backend_config(backend_type):
            configs = {
                BackendType.POSTGRESQL: {"uri": "postgresql://localhost/db"},
                BackendType.MEMGRAPH: {"uri": "bolt://localhost:7687"},
                BackendType.MONGODB: {"uri": "mongodb://localhost:27017"},
                BackendType.NEO4J: {"uri": "bolt://localhost:7687"},
            }
            return configs.get(backend_type)
        
        pg_config = get_backend_config(BackendType.POSTGRESQL)
        mongo_config = get_backend_config(BackendType.MONGODB)
        
        if pg_config and mongo_config:
            print("    [OK] Backend type switching works")
        else:
            print("    [FAIL] Backend type switching failed")
            return False
    except Exception as e:
        print(f"    [FAIL] Backend type switching failed: {e}")
        return False
    
    print("\n  [OK] All external project scenarios passed")
    return True


def test_storage_modules_no_breaking_changes():
    """Test that storage modules don't have breaking changes for external use."""
    print("\n" + "="*60)
    print("TEST 6: Storage Module Compatibility")
    print("="*60)
    
    # Test EnhancedKnowledgeStorage
    try:
        from knowledge_engine.enhanced_storage import EnhancedKnowledgeStorage, StorageBackend
        storage = EnhancedKnowledgeStorage(config={
            "backends": {
                "postgresql": {"enabled": True},
                "memgraph": {"enabled": True},
            },
            "default_backend": "postgresql"
        })
        print("    [OK] EnhancedKnowledgeStorage created successfully")
        
        # Verify StorageBackend enum has expected values
        for sb in ["POSTGRESQL", "MEMGRAPH", "QDRANT", "REDIS"]:
            if hasattr(StorageBackend, sb):
                print(f"    [OK] StorageBackend.{sb} available")
            else:
                print(f"    [FAIL] StorageBackend.{sb} missing")
                return False
    except Exception as e:
        print(f"    [FAIL] EnhancedKnowledgeStorage failed: {e}")
        return False
    
    # Test KnowledgeStorage
    try:
        from knowledge_engine.knowledge_storage import KnowledgeStorage
        storage = KnowledgeStorage(config={
            "postgresql": {"enabled": True},
            "memgraph": {"enabled": True},
            "default_backend": "postgresql"
        })
        print("    [OK] KnowledgeStorage created successfully")
    except Exception as e:
        print(f"    [FAIL] KnowledgeStorage failed: {e}")
        return False
    
    return True


def main():
    """Run all backward compatibility tests."""
    print("\n" + "="*60)
    print("KNOWLEDGE ENGINE - BACKWARD COMPATIBILITY TESTS")
    print("Verifying external projects won't break")
    print("="*60)
    
    results = {
        "MongoDBBackend Import": test_import_mongodb_backend(),
        "Neo4jBackend Import": test_import_neo4j_backend(),
        "BackendType Enum": test_backend_type_enum(),
        "__all__ Exports": test_backend_all_exports(),
        "External Project Scenario": test_external_project_scenario(),
        "Storage Module Compatibility": test_storage_modules_no_breaking_changes(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print(f"\n  Total: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n" + "="*60)
        print("ALL BACKWARD COMPATIBILITY TESTS PASSED!")
        print("="*60)
        print("\nExternal projects using the knowledge engine will NOT break.")
        print("\nKey compatibility maintained:")
        print("  - MongoDBBackend can still be imported and used")
        print("  - Neo4jBackend can still be imported and used")
        print("  - BackendType.MONGODB and BackendType.NEO4J still exist")
        print("  - All backends are exported in __all__")
        print("\nHowever, these orphaned backends are not used by active code.")
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
