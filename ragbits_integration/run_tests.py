#!/usr/bin/env python
"""
Test runner for RAGBits integration tests

Run with: python ragbits_integration/run_tests.py
"""

import sys
import asyncio


def run_integration_tests():
    """Run integration tests"""
    from ragbits_integration.tests import test_integration

    print("=" * 70)
    print("RAGBits Integration - Integration Tests")
    print("=" * 70)
    print()

    tests = [
        ("End-to-End Workflow Simulation", test_integration.test_end_to_end_workflow_simulation),
        ("Cross-Stage Context Flow", test_integration.test_cross_stage_context_flow),
        ("Lifecycle State Transitions", test_integration.test_lifecycle_state_transitions),
        ("Cache Functionality", test_integration.test_cache_functionality),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n{'─' * 70}")
        print(f"Running: {name}")
        print(f"{'─' * 70}")

        try:
            asyncio.run(test_func())
            passed += 1
            print(f"✅ PASSED: {name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}")

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_integration_tests()
