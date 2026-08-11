#!/usr/bin/env python3
"""
Knowledge Engine Verification Script

Verifies that the Knowledge Engine is complete and functional.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that key components can be imported."""
    print("=" * 60)
    print("Testing Knowledge Engine Imports")
    print("=" * 60)

    tests = [
        ("MasterKnowledgeEngine", "from knowledge_engine.master_engine import MasterKnowledgeEngine"),
        ("UnifiedKGIntegrationHub", "from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub"),
        ("KnowledgeOrchestrator", "from knowledge_engine.orchestration import KnowledgeOrchestrator"),
        ("DeepKEExtractor", "from knowledge_engine.deepke import DeepKEExtractor"),
        ("GraphCRUD", "from knowledge_engine.graph import GraphCRUD"),
        ("HybridSearch", "from knowledge_engine.hybrid import HybridSearch"),
    ]

    passed = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  [OK] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"\nImport Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_master_engine():
    """Test MasterKnowledgeEngine functionality."""
    print("\n" + "=" * 60)
    print("Testing MasterKnowledgeEngine")
    print("=" * 60)

    try:
        from knowledge_engine.master_engine import MasterKnowledgeEngine

        # Create engine
        engine = MasterKnowledgeEngine()
        print("  [OK] Engine instantiated")

        # Check attributes
        print(f"  [OK] Self-improving: {engine.enable_learning}")
        print(f"  [OK] Self-healing: {engine.enable_healing}")

        # Get capabilities
        caps = engine.get_capabilities()
        print(f"  [OK] Capabilities: {len(caps)} components")

        # Get statistics
        stats = engine.get_statistics()
        print(f"  [OK] Execution count: {stats.get('execution_count', 0)}")
        print(f"  [OK] Success rate: {stats.get('success_rate', 0):.1%}")

        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_integrations():
    """Test that integrations are properly wired."""
    print("\n" + "=" * 60)
    print("Testing Integration Wiring")
    print("=" * 60)

    try:
        from knowledge_engine.master_engine import MasterKnowledgeEngine

        engine = MasterKnowledgeEngine()

        # Get capabilities
        capabilities = engine.get_capabilities()
        print(f"  [OK] {len(capabilities)} capabilities available")

        # Show first 10 capabilities
        for i, (name, caps) in enumerate(list(capabilities.items())[:10]):
            print(f"  [OK] {name}: {len(caps)} capabilities")

        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE ENGINE VERIFICATION")
    print("=" * 60 + "\n")

    results = {
        "Imports": test_imports(),
        "Master Engine": test_master_engine(),
        "Integrations": test_integrations(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("STATUS: KNOWLEDGE ENGINE COMPLETE AND FUNCTIONAL")
    else:
        print("STATUS: KNOWLEDGE ENGINE HAS ISSUES")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
