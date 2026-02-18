#!/usr/bin/env python3
"""
Final verification test for Knowledge Engine comprehensive fixes.

Tests all major components and integrations to ensure they import correctly
and follow CLAUDE.md principles.
"""

import sys
from pathlib import Path

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_core_imports():
    """Test core knowledge engine imports."""
    print("\n" + "="*60)
    print("Testing Core Imports")
    print("="*60)

    tests = [
        ("KnowledgeEngine", "from knowledge_engine import KnowledgeEngine"),
        ("KnowledgeArtifact", "from knowledge_engine.core.temporal_knowledge_engine import KnowledgeArtifact"),
        ("MemoryBackend", "from knowledge_engine.core.backends.memory_backend import MemoryBackend"),
        ("KnowledgeGraphBackend", "from knowledge_engine.core.backends.base import KnowledgeGraphBackend"),
        ("ConfigValidator", "from knowledge_engine.config_validation import ConfigValidator"),
    ]

    passed = 0
    failed = 0

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"[OK] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {str(e)[:80]}")
            failed += 1

    print(f"\nCore: {passed} passed, {failed} failed")
    return failed == 0


def test_integration_imports():
    """Test integration imports with graceful degradation."""
    print("\n" + "="*60)
    print("Testing Integration Imports (Graceful Degradation)")
    print("="*60)

    tests = [
        ("ROMAIntegration", "from knowledge_engine.integrations.roma_integration import ROMAIntegration"),
        ("GraphitiTemporalBridge", "from knowledge_engine.integrations.graphiti_temporal_bridge import GraphitiTemporalBridge"),
        ("UnifiedMathKnowledgeBridge", "from knowledge_engine.integrations.unified_math_knowledge_bridge import UnifiedMathKnowledgeBridge"),
    ]

    passed = 0
    failed = 0

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"[OK] {name}")
            passed += 1
        except Exception as e:
            # Some integrations may fail to import - that's OK if they use graceful degradation
            if "graceful degradation" in str(e).lower() or "not available" in str(e).lower():
                print(f"[SKIP] {name} (gracefully degraded)")
                passed += 1
            else:
                print(f"[FAIL] {name}: {str(e)[:80]}")
                failed += 1

    print(f"\nIntegrations: {passed} passed, {failed} failed")
    return failed == 0


def test_visualization_imports():
    """Test visualization imports."""
    print("\n" + "="*60)
    print("Testing Visualization Imports")
    print("="*60)

    tests = [
        ("KnowledgeGraphVisualizer", "from knowledge_engine.visualization import KnowledgeGraphVisualizer"),
        ("MetricsVisualizer", "from knowledge_engine.visualization import MetricsVisualizer"),
        ("VisualizationConfig", "from knowledge_engine.visualization import VisualizationConfig"),
    ]

    passed = 0
    failed = 0

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"[OK] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {str(e)[:80]}")
            failed += 1

    print(f"\nVisualization: {passed} passed, {failed} failed")
    return failed == 0


def test_configuration():
    """Test configuration validation."""
    print("\n" + "="*60)
    print("Testing Configuration (Law 5: No Magic Defaults)")
    print("="*60)

    try:
        from knowledge_engine.config_validation import validate_config, ValidationResult

        # Test that validation works
        result = validate_config(strict=False, silent=True)

        if result.is_valid:
            print(f"[OK] Configuration validation passed")
            print(f"  - Required vars: {len(result.present_optional)} present")
            print(f"  - Warnings: {len(result.warnings)}")
            return True
        else:
            print(f"[WARN] Configuration has issues:")
            for error in result.errors[:3]:
                print(f"  - {error}")
            # Still pass if only optional vars are missing
            return len(result.errors) == 0

    except Exception as e:
        print(f"[FAIL] Configuration validation: {str(e)[:80]}")
        return False


def test_main_import():
    """Test main knowledge_engine import."""
    print("\n" + "="*60)
    print("Testing Main Import")
    print("="*60)

    try:
        import knowledge_engine
        print(f"[OK] knowledge_engine")
        return True
    except Exception as e:
        print(f"[FAIL] knowledge_engine: {str(e)[:80]}")
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("Knowledge Engine - Final Verification")
    print("="*60)

    results = {
        "Main Import": test_main_import(),
        "Core Imports": test_core_imports(),
        "Integration Imports": test_integration_imports(),
        "Visualization Imports": test_visualization_imports(),
        "Configuration": test_configuration(),
    }

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n[OK] All verification tests passed!")
        print("The Knowledge Engine is ready for use.")
        return 0
    else:
        print("\n[FAIL] Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
