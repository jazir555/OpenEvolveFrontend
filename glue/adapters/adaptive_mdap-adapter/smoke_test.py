#!/usr/bin/env python3
"""
Smoke Test: Quick Validation of Adaptive MDAP/MAKER Adapter Integration

This script provides a fast validation that all components can be imported
and basic operations work. Run after deployment to verify success.

Usage:
    python smoke_test.py

Exit Codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import sys
import os
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set required environment variables
os.environ.setdefault("ADAPTIVE_MDAP_TIMEOUT_MS", "5000")
os.environ.setdefault("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY", "sk-test"))

# Track test results
TESTS_PASSED = []
TESTS_FAILED = []


def test(name: str):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            try:
                result = func()
                if result:
                    TESTS_PASSED.append(name)
                    print(f"  [PASS] {name}")
                else:
                    TESTS_FAILED.append(name)
                    print(f"  [FAIL] {name}")
                return result
            except Exception as e:
                TESTS_FAILED.append(name)
                print(f"  [FAIL] {name}: {e}")
                return False
        return wrapper
    return decorator


@test("Core MDAP adapter imports")
def test_1():
    """Test core MDAP adapter imports."""
    try:
        from src import get_adapter, CanonicalSubProblem
        adapter = get_adapter()
        return adapter is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("MAKER adapter imports")
def test_2():
    """Test MAKER adapter imports."""
    try:
        from src import get_maker_adapter
        adapter = get_maker_adapter()
        return adapter is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Advanced OpenEvolve integration imports")
def test_3():
    """Test advanced OpenEvolve imports."""
    try:
        from src import get_advanced_openevolve_integration
        integration = get_advanced_openevolve_integration()
        return integration is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Advanced BubbleLab UI imports")
def test_4():
    """Test advanced BubbleLab UI imports."""
    try:
        from src import get_advanced_bubblelab_ui
        ui = get_advanced_bubblelab_ui()
        return ui is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Advanced Gauntlet integration imports")
def test_5():
    """Test advanced gauntlet imports."""
    try:
        from src import get_advanced_gauntlet_integration
        gauntlet = get_advanced_gauntlet_integration()
        return gauntlet is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Advanced ICR integration imports")
def test_6():
    """Test advanced ICR imports."""
    try:
        from src import get_advanced_icr_integration
        icr = get_advanced_icr_integration()
        return icr is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Async adapter imports")
def test_7():
    """Test async adapter imports."""
    try:
        from src import get_async_adapter
        adapter = get_async_adapter()
        return adapter is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Performance monitor imports")
def test_8():
    """Test performance monitor imports."""
    try:
        from src import get_performance_monitor
        monitor = get_performance_monitor()
        return monitor is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Unified system monitor imports")
def test_9():
    """Test unified system monitor imports."""
    try:
        from src import get_unified_system_monitor
        monitor = get_unified_system_monitor()
        return monitor is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Integration manager imports")
def test_10():
    """Test integration manager imports."""
    try:
        from src import get_integration_manager
        manager = get_integration_manager()
        return manager is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("MDAP health check")
def test_11():
    """Test MDAP adapter health check."""
    try:
        from src import get_adapter
        adapter = get_adapter()
        health = adapter.health_check()
        return health.get("status") == "healthy"
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("MAKER health check")
def test_12():
    """Test MAKER adapter health check."""
    try:
        from src import get_maker_adapter
        adapter = get_maker_adapter()
        health = adapter.health_check()
        return health.get("status") == "healthy"
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Integration manager health check")
def test_13():
    """Test integration manager health check."""
    try:
        from src import get_integration_manager
        manager = get_integration_manager()
        health = manager.get_health_status()
        return health is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Unified entry point imports")
def test_14():
    """Test unified entry point imports."""
    try:
        from unified_entry import UnifiedAdapterInterface
        interface = UnifiedAdapterInterface()
        return interface is not None
    except Exception as e:
        print(f"    Error: {e}")
        return False


@test("Canonical schema definitions")
def test_15():
    """Test canonical schema classes exist."""
    try:
        from src import (
            CanonicalSubProblem,
            CanonicalComplexityScore,
            CanonicalStrategy,
            CanonicalResponse
        )
        return all([CanonicalSubProblem, CanonicalComplexityScore,
                   CanonicalStrategy, CanonicalResponse])
    except Exception as e:
        print(f"    Error: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 70)
    print("  ADAPTIVE MDAP/MAKER ADAPTER - SMOKE TEST")
    print("=" * 70)
    print(f"\nStart Time: {datetime.now(timezone.utc).isoformat()}\n")
    print("Running Tests...\n")

    # Run all tests
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    test_9()
    test_10()
    test_11()
    test_12()
    test_13()
    test_14()
    test_15()

    # Print summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)

    total = len(TESTS_PASSED) + len(TESTS_FAILED)
    passed = len(TESTS_PASSED)
    failed = len(TESTS_FAILED)
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Pass Rate: {pass_rate:.1f}%")

    if TESTS_FAILED:
        print(f"\nFailed Tests:")
        for test_name in TESTS_FAILED:
            print(f"  - {test_name}")

    print("\n" + "=" * 70)

    if failed == 0:
        print("\n  SUCCESS: All smoke tests passed!")
        print("  Integration is operational.\n")
        return 0
    else:
        print(f"\n  FAILURE: {failed} test(s) failed")
        print("  Integration has issues that must be resolved.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
