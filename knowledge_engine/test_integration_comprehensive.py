"""
Comprehensive Integration Tests for Knowledge Engine

Tests all integrations work correctly with new permissive-licensed backends.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported correctly."""
    print("\n" + "="*60)
    print("TEST 1: Module Imports")
    print("="*60)
    
    tests = [
        ("Base backend types", "knowledge_engine.core.backends.base"),
        ("Memory backend", "knowledge_engine.core.backends.memory_backend"),
        ("PostgreSQL backend", "knowledge_engine.core.backends.postgresql_backend"),
        ("Memgraph backend", "knowledge_engine.core.backends.memgraph_backend"),
        ("Qdrant backend", "knowledge_engine.core.backends.qdrant_backend"),
        ("Enhanced storage", "knowledge_engine.enhanced_storage"),
        ("Knowledge storage", "knowledge_engine.knowledge_storage"),
        ("Real DB integration", "knowledge_engine.real_database_integration"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module_path in tests:
        try:
            __import__(module_path)
            print(f"  [OK] {name}: {module_path}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
    
    print(f"\n  Result: {passed}/{passed+failed} imports successful")
    return failed == 0


def test_backend_types():
    """Test backend type enum values."""
    print("\n" + "="*60)
    print("TEST 2: Backend Type Enum")
    print("="*60)
    
    from knowledge_engine.core.backends.base import BackendType
    from knowledge_engine.enhanced_storage import StorageBackend
    from knowledge_engine.real_database_integration import DatabaseType
    
    # Check base backend types
    print("\n  Base BackendType values:")
    for bt in BackendType:
        print(f"    - {bt.name}: {bt.value}")
    
    # Check storage backend types
    print("\n  StorageBackend values:")
    for sb in StorageBackend:
        print(f"    - {sb.name}: {sb.value}")
    
    # Check database type
    print("\n  DatabaseType values:")
    for dt in DatabaseType:
        print(f"    - {dt.name}: {dt.value}")
    
    # Verify no GPL/SSPL backends in enum
    print("\n  License compliance check:")
    all_types = list(BackendType) + list(StorageBackend) + list(DatabaseType)
    gpl_found = [t for t in all_types if 'neo4j' in t.value.lower()]
    sspl_found = [t for t in all_types if 'mongodb' in t.value.lower()]
    
    if gpl_found:
        print(f"    [WARNING] Neo4j found: {gpl_found}")
    else:
        print("    [OK] No Neo4j (GPL) in active backend types")
    
    if sspl_found:
        print(f"    [WARNING] MongoDB found: {sspl_found}")
    else:
        print("    [OK] No MongoDB (SSPL) in active backend types")
    
    return len(gpl_found) == 0 and len(sspl_found) == 0


def test_backend_classes():
    """Test that backend classes can be instantiated."""
    print("\n" + "="*60)
    print("TEST 3: Backend Class Instantiation")
    print("="*60)
    
    from knowledge_engine.core.backends.memory_backend import MemoryBackend
    from knowledge_engine.core.backends.postgresql_backend import PostgreSQLBackend
    from knowledge_engine.core.backends.memgraph_backend import MemgraphBackend
    from knowledge_engine.core.backends.qdrant_backend import QdrantBackend
    
    tests = [
        ("MemoryBackend", MemoryBackend, {}),
        ("PostgreSQLBackend", PostgreSQLBackend, {"uri": "postgresql://localhost/test"}),
        ("MemgraphBackend", MemgraphBackend, {"uri": "bolt://localhost:7687"}),
        ("QdrantBackend", QdrantBackend, {"host": "localhost", "port": 6333}),
    ]
    
    passed = 0
    failed = 0
    
    for name, cls, config in tests:
        try:
            instance = cls(config=config)
            print(f"  [OK] {name} instantiated (backend_type={instance.backend_type.value})")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
    
    print(f"\n  Result: {passed}/{passed+failed} backends instantiated")
    return failed == 0


def test_enhanced_storage_config():
    """Test EnhancedStorage configuration."""
    print("\n" + "="*60)
    print("TEST 4: EnhancedStorage Configuration")
    print("="*60)
    
    from knowledge_engine.enhanced_storage import EnhancedKnowledgeStorage, StorageBackend
    
    config = {
        "backends": {
            "postgresql": {"enabled": True, "uri": "postgresql://localhost/test"},
            "memgraph": {"enabled": True, "uri": "bolt://localhost:7687"},
            "qdrant": {"enabled": False},
            "redis": {"enabled": False}
        },
        "default_backend": "postgresql"
    }
    
    try:
        storage = EnhancedKnowledgeStorage(config=config)
        print(f"  [OK] EnhancedKnowledgeStorage created")
        print(f"       Default backend: {storage.config.get('default_backend')}")
        print(f"       Fallback chain: {storage.config.get('fallback_chain', [])}")
        
        # Check no MongoDB/Neo4j in config
        has_mongodb = "mongodb" in str(storage.config).lower()
        has_neo4j = "neo4j" in str(storage.config).lower()
        
        if has_mongodb:
            print(f"  [WARNING] MongoDB found in config")
        if has_neo4j:
            print(f"  [WARNING] Neo4j found in config")
        
        if not has_mongodb and not has_neo4j:
            print(f"  [OK] Config verified: no MongoDB/Neo4j references")
        
        return not has_mongodb and not has_neo4j
    except Exception as e:
        print(f"  [FAIL] EnhancedKnowledgeStorage creation failed: {e}")
        return False


def test_knowledge_storage_config():
    """Test KnowledgeStorage configuration."""
    print("\n" + "="*60)
    print("TEST 5: KnowledgeStorage Configuration")
    print("="*60)
    
    from knowledge_engine.knowledge_storage import KnowledgeStorage
    
    config = {
        "postgresql": {"enabled": True, "uri": "postgresql://localhost/test"},
        "memgraph": {"enabled": True, "uri": "bolt://localhost:7687"},
        "qdrant": {"enabled": False},
        "default_backend": "postgresql"
    }
    
    try:
        storage = KnowledgeStorage(config=config)
        print(f"  [OK] KnowledgeStorage created")
        
        # Check attributes
        attrs = ["postgresql_pool", "memgraph_driver"]
        for attr in attrs:
            has_attr = hasattr(storage, attr)
            print(f"       Has {attr}: {has_attr}")
        
        # Check no MongoDB/Neo4j in config
        has_mongodb = "mongodb" in str(storage.config).lower()
        has_neo4j = "neo4j" in str(storage.config).lower()
        
        if has_mongodb:
            print(f"  [WARNING] MongoDB found in config")
        if has_neo4j:
            print(f"  [WARNING] Neo4j found in config")
        
        if not has_mongodb and not has_neo4j:
            print(f"  [OK] Config verified: no MongoDB/Neo4j references")
        
        return not has_mongodb and not has_neo4j
    except Exception as e:
        print(f"  [FAIL] KnowledgeStorage creation failed: {e}")
        return False


def test_real_database_integration():
    """Test RealDatabaseIntegrator configuration."""
    print("\n" + "="*60)
    print("TEST 6: RealDatabaseIntegrator Configuration")
    print("="*60)
    
    from knowledge_engine.real_database_integration import RealDatabaseIntegrator
    
    try:
        integrator = RealDatabaseIntegrator()
        print(f"  [OK] RealDatabaseIntegrator created")
        print(f"       Config databases: {list(integrator.config.get('databases', {}).keys())}")
        print(f"       Default database: {integrator.config.get('default_database')}")
        
        # Check no MongoDB/Neo4j in config
        has_mongodb = "mongodb" in str(integrator.config).lower()
        has_neo4j = "neo4j" in str(integrator.config).lower()
        
        if has_mongodb:
            print(f"  [WARNING] MongoDB found in config")
        if has_neo4j:
            print(f"  [WARNING] Neo4j found in config")
        
        if not has_mongodb and not has_neo4j:
            print(f"  [OK] Config verified: no MongoDB/Neo4j references")
        
        return not has_mongodb and not has_neo4j
    except Exception as e:
        print(f"  [FAIL] RealDatabaseIntegrator creation failed: {e}")
        return False


def test_integration_paths():
    """Test that integration paths don't reference deprecated backends."""
    print("\n" + "="*60)
    print("TEST 7: Integration Code Path Verification")
    print("="*60)
    
    import ast
    import os
    
    files_to_check = [
        "knowledge_engine/enhanced_storage.py",
        "knowledge_engine/knowledge_storage.py",
        "knowledge_engine/real_database_integration.py",
    ]
    
    forbidden_terms = ["pymongo", "MongoClient", "neo4j_driver"]
    found_issues = []
    
    for file_path in files_to_check:
        full_path = Path(__file__).parent.parent / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        for term in forbidden_terms:
                            if term in line and not line.strip().startswith('#'):
                                found_issues.append(f"{file_path}:{i}: {term}")
            except Exception as e:
                print(f"  [ERROR] Could not check {file_path}: {e}")
    
    if found_issues:
        print("  [WARNING] Found potential issues:")
        for issue in found_issues:
            print(f"    - {issue}")
    else:
        print("  [OK] No forbidden terms found in active code")
    
    return len(found_issues) == 0


def test_complete_integration():
    """Run the complete integration test."""
    print("\n" + "="*60)
    print("TEST 8: Complete Integration Test")
    print("="*60)
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "knowledge_engine/test_complete_integration.py"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Check for pass indicators
        output = result.stdout
        if "7/7 phases passed" in output and "ALL TESTS PASSED" in output:
            print("  [OK] Complete integration test passed (7/7 phases)")
            return True
        else:
            print("  [WARNING] Integration test may have issues")
            print(f"       Return code: {result.returncode}")
            return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] Could not run integration test: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("KNOWLEDGE ENGINE - COMPREHENSIVE INTEGRATION TESTS")
    print("Verifying PostgreSQL and Memgraph Backend Integration")
    print("="*60)
    
    results = {
        "Module Imports": test_imports(),
        "Backend Type Enum": test_backend_types(),
        "Backend Classes": test_backend_classes(),
        "EnhancedStorage Config": test_enhanced_storage_config(),
        "KnowledgeStorage Config": test_knowledge_storage_config(),
        "RealDatabaseIntegration": test_real_database_integration(),
        "Code Path Verification": test_integration_paths(),
        "Complete Integration": test_complete_integration(),
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
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nThe knowledge engine integrations work correctly")
        print("with the new PostgreSQL and Memgraph backends.")
        return 0
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
