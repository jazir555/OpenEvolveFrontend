"""
Quick verification script for LoongFlow fallback implementation.
"""

import sys
import os

# Add openevolve to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'openevolve'))

def check_imports():
    """Verify all components can be imported."""
    print("Checking imports...")

    try:
        from openevolve.integrations import (
            LoongFlowAdapter,
            LoongFlowChecker,
            OpenEvolveFallbackAdapter
        )
        print("  [OK] All integrations imported successfully")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def check_checker():
    """Verify LoongFlowChecker works."""
    print("\nChecking LoongFlowChecker...")

    try:
        from openevolve.integrations import LoongFlowChecker

        installed = LoongFlowChecker.is_installed()
        print(f"  [OK] LoongFlow installed: {installed}")

        version = LoongFlowChecker.get_version()
        print(f"  [OK] LoongFlow version: {version or 'N/A'}")

        available = LoongFlowChecker.is_available()
        print(f"  [OK] LoongFlow available: {available}")

        diagnostics = LoongFlowChecker.get_diagnostics()
        print(f"  [OK] Diagnostics retrieved: {len(diagnostics)} keys")

        return True
    except Exception as e:
        print(f"  [FAIL] LoongFlowChecker failed: {e}")
        return False


def check_fallback_adapter():
    """Verify OpenEvolveFallbackAdapter works."""
    print("\nChecking OpenEvolveFallbackAdapter...")

    try:
        from openevolve.integrations import OpenEvolveFallbackAdapter

        config = {"mode": "standard", "max_iterations": 10}
        adapter = OpenEvolveFallbackAdapter(config)
        print(f"  [OK] Fallback adapter created: {adapter}")

        capabilities = adapter.get_capabilities()
        print(f"  [OK] Capabilities retrieved: {capabilities['system']}")

        return True
    except Exception as e:
        print(f"  [FAIL] OpenEvolveFallbackAdapter failed: {e}")
        return False


def check_loongflow_adapter():
    """Verify LoongFlowAdapter with fallback works."""
    print("\nChecking LoongFlowAdapter...")

    try:
        from openevolve.integrations import LoongFlowAdapter

        # Test with LoongFlow disabled (should use fallback)
        config = {
            "enable_loongflow": False,
            "mode": "standard",
            "show_messages": False
        }
        adapter = LoongFlowAdapter(config)
        print(f"  [OK] Adapter created (LoongFlow disabled)")

        status = adapter.get_status()
        print(f"  [OK] Status: {status['mode']}")
        print(f"  [OK] Using LoongFlow: {status['using_loongflow']}")
        print(f"  [OK] System: {status['capabilities']['system']}")

        capabilities = adapter.get_capabilities()
        print(f"  [OK] Capabilities: {capabilities['available']}")

        return True
    except Exception as e:
        print(f"  [FAIL] LoongFlowAdapter failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_messages():
    """Verify LoongFlowMessages works."""
    print("\nChecking LoongFlowMessages...")

    try:
        from openevolve.utils.messages import LoongFlowMessages

        msg1 = LoongFlowMessages.disabled_message()
        print(f"  [OK] Disabled message generated ({len(msg1)} chars)")

        msg2 = LoongFlowMessages.using_openevolve_message("standard")
        print(f"  [OK] OpenEvolve message generated ({len(msg2)} chars)")

        return True
    except Exception as e:
        print(f"  [FAIL] LoongFlowMessages failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("LoongFlow Fallback Implementation Verification")
    print("=" * 70)

    checks = [
        check_imports,
        check_checker,
        check_fallback_adapter,
        check_loongflow_adapter,
        check_messages,
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"\n  [FAIL] Check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\nChecks passed: {passed}/{total}")

    if passed == total:
        print("\n[OK] All checks passed! Fallback system is operational.")
        return 0
    else:
        print(f"\n[WARN]  {total - passed} check(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
