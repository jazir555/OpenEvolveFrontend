#!/usr/bin/env python
"""
OpenEvolve Integration Verification Script

Run this script to verify that all OpenEvolve components are properly integrated.

Usage:
    python tests/integration/verify_integration.py

Expected output: All tests should PASS (4/4)
"""

import sys
import os

# Ensure UTF-8 output for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_gauntlets():
    """Test gauntlet imports"""
    print("\n" + "="*80)
    print("TEST: Gauntlets Integration")
    print("="*80)

    try:
        from openevolve.gauntlets import (
            LoongFlowGauntletEvaluator,
            ThreeRoundGauntletOrchestrator,
            MultiRoundGauntletOrchestrator,
        )
        print("[OK] LoongFlowGauntletEvaluator")
        print("[OK] ThreeRoundGauntletOrchestrator")
        print("[OK] MultiRoundGauntletOrchestrator")
        print("\nSTATUS: PASSED [OK]")
        return True
    except Exception as e:
        print(f"\nSTATUS: FAILED [FAIL]\nError: {e}")
        return False


def test_domain_optimizers():
    """Test domain optimizer imports and instantiation"""
    print("\n" + "="*80)
    print("TEST: Domain Optimizers Integration")
    print("="*80)

    try:
        from openevolve.domain import FinanceOptimizer

        # Test instantiation
        finance = FinanceOptimizer()
        print("[OK] FinanceOptimizer instantiated")

        # Test config retrieval
        config = finance.get_default_config()
        print(f"[OK] Config retrieved: {type(config).__name__}")

        print("\nSTATUS: PASSED [OK]")
        return True
    except Exception as e:
        print(f"\nSTATUS: FAILED [FAIL]\nError: {e}")
        return False


def test_unified_config():
    """Test unified config system"""
    print("\n" + "="*80)
    print("TEST: Unified Config System")
    print("="*80)

    try:
        from openevolve.unified.config import UnifiedEvolutionConfig

        config = UnifiedEvolutionConfig()
        print(f"[OK] UnifiedEvolutionConfig created")
        print(f"  - Type: {type(config).__name__}")

        print("\nSTATUS: PASSED [OK]")
        return True
    except Exception as e:
        print(f"\nSTATUS: FAILED [FAIL]\nError: {e}")
        return False


def test_knowledge_engine():
    """Test knowledge engine integration"""
    print("\n" + "="*80)
    print("TEST: Knowledge Engine Integration")
    print("="*80)

    try:
        from openevolve.knowledge_engine.integrations import LoongFlowKnowledgeExtractor
        print("[OK] LoongFlowKnowledgeExtractor imported")

        print("\nSTATUS: PASSED [OK]")
        return True
    except Exception as e:
        print(f"\nSTATUS: FAILED [FAIL]\nError: {e}")
        return False


def main():
    """Run all verification tests"""
    print("\n" + "="*80)
    print("OPENEVOLVE INTEGRATION VERIFICATION")
    print("="*80)
    print("\nTesting core components integration...")

    tests = [
        ("Gauntlets", test_gauntlets),
        ("Domain Optimizers", test_domain_optimizers),
        ("Unified Config", test_unified_config),
        ("Knowledge Engine", test_knowledge_engine),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*80)

    if passed == total:
        print("\n[OK] ALL TESTS PASSED - Integration is working correctly!")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} test(s) failed - Please check the errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
