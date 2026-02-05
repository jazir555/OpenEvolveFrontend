#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Graceful Degradation in Knowledge Engine

This script tests that all integrations gracefully handle missing dependencies
and that the system continues to function with reduced capabilities.
"""

import logging
import sys
from pathlib import Path

# Setup encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_optional_imports_module():
    """Test the optional_imports module functionality."""
    print("\n" + "="*80)
    print("Testing Optional Imports Module")
    print("="*80)

    try:
        from knowledge_engine.optional_imports import (
            import_optional,
            require_dependency,
            is_available,
            check_all_optional_dependencies,
            OptionalDependencyError,
            create_failing_mock
        )

        print("[OK] Optional imports module loaded successfully")

        # Test is_available
        print("\nTesting availability checks:")
        for module in ['os', 'sys', 'nonexistent_module_test']:
            available = is_available(module)
            status = "[OK]" if available == (module in ['os', 'sys']) else "[FAIL]"
            print(f"  {status} is_available('{module}'): {available}")

        # Test import_optional with fail_silently=True
        print("\nTesting silent import failures:")
        result = import_optional(
            'nonexistent_test_module',
            'nonexistent-test',
            'testing',
            'pip install nonexistent-test',
            fail_silently=True
        )
        print(f"  [OK] Silent import returned None: {result is None}")

        # Test failing mock
        print("\nTesting failing mock creation:")
        MockClass = create_failing_mock(
            'test-package',
            'test feature',
            'pip install test-package'
        )
        try:
            mock = MockClass()
            print("  [FAIL] Mock should have raised error")
        except OptionalDependencyError as e:
            print(f"  [OK] Mock raised error as expected: {str(e)[:80]}...")

        # Check all optional dependencies
        print("\nChecking all optional dependencies:")
        results = check_all_optional_dependencies()
        print(f"\n  Checked {len(results)} dependencies")

        return True

    except Exception as e:
        logger.error(f"Failed to test optional_imports module: {e}")
        return False


def test_main_init_graceful_degradation():
    """Test that __init__.py degrades gracefully."""
    print("\n" + "="*80)
    print("Testing Main __init__.py Graceful Degradation")
    print("="*80)

    try:
        # This import should succeed even if some integrations are unavailable
        import knowledge_engine

        print("[OK] knowledge_engine imported successfully")

        # Check that key components are available
        key_components = [
            'OpenEvolveKnowledgeEngine',
            'KnowledgeEngineOutput',
            'get_knowledge_engine',
        ]

        print("\nChecking key components:")
        for component in key_components:
            if hasattr(knowledge_engine, component):
                print(f"  [OK] {component} is available")
            else:
                print(f"  [FAIL] {component} is NOT available")

        # Check for availability flags
        print("\nChecking integration availability flags:")
        integrations_module = getattr(knowledge_engine, 'integrations', None)
        if integrations_module:
            flags = [
                'Z3_INTEGRATION_AVAILABLE',
                'LEANAIDE_KE_AVAILABLE',
                'ROMA_INTEGRATION_AVAILABLE',
                'CAUSAL_LEARN_AVAILABLE',
            ]
            for flag in flags:
                if hasattr(integrations_module, flag):
                    value = getattr(integrations_module, flag)
                    print(f"  [OK] {flag}: {value}")
                else:
                    print(f"  [FAIL] {flag}: not defined")

        return True

    except Exception as e:
        logger.error(f"Failed to test main __init__.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_graceful_degradation():
    """Test that individual integrations degrade gracefully."""
    print("\n" + "="*80)
    print("Testing Integration Graceful Degradation")
    print("="*80)

    integrations_to_test = [
        'deepke_integration',
        'dspy_integration',
        'ragbits_integration',
        'agentic_context_integration',
        'agentjson_integration',
        'research_quest_integration',
        'mcp_gateway_integration',
        'roma_integration',
        'openevolve_integration_library',
    ]

    results = {}

    for integration_name in integrations_to_test:
        print(f"\n--- Testing {integration_name} ---")
        try:
            module_path = f'knowledge_engine.integrations.{integration_name}'
            module = __import__(module_path, fromlist=[''])

            # Look for availability flag
            flag_name = None
            for attr in dir(module):
                if 'AVAILABLE' in attr:
                    flag_name = attr
                    break

            if flag_name:
                available = getattr(module, flag_name)
                print(f"  Availability flag {flag_name}: {available}")
                results[integration_name] = {
                    'loaded': True,
                    'has_flag': True,
                    'available': available,
                    'flag_name': flag_name
                }
            else:
                print(f"  No availability flag found")
                results[integration_name] = {
                    'loaded': True,
                    'has_flag': False,
                }

        except ImportError as e:
            print(f"  Import failed: {e}")
            results[integration_name] = {
                'loaded': False,
                'error': str(e)
            }
        except Exception as e:
            print(f"  Unexpected error: {e}")
            results[integration_name] = {
                'loaded': False,
                'error': str(e)
            }

    # Summary
    print("\n" + "-"*80)
    print("Summary:")
    print("-"*80)
    loaded = sum(1 for r in results.values() if r.get('loaded', False))
    has_flags = sum(1 for r in results.values() if r.get('has_flag', False))
    available = sum(1 for r in results.values() if r.get('available', False))

    print(f"  Integrations tested: {len(integrations_to_test)}")
    print(f"  Successfully loaded: {loaded}")
    print(f"  Have availability flags: {has_flags}")
    print(f"  Marked as available: {available}")

    return loaded > 0


def test_capability_reporting():
    """Test that integrations report their capabilities."""
    print("\n" + "="*80)
    print("Testing Capability Reporting")
    print("="*80)

    try:
        from knowledge_engine import get_capabilities

        print("\nFetching system capabilities:")
        capabilities = get_capabilities()

        print("\n  Available Capabilities:")
        if 'available' in capabilities:
            for cap in capabilities['available']:
                print(f"    [OK] {cap}")

        print("\n  Unavailable Capabilities:")
        if 'unavailable' in capabilities:
            for cap in capabilities['unavailable']:
                reason = cap.get('reason', 'unknown')
                print(f"    [FAIL] {cap['name']}: {reason}")

        return True

    except ImportError:
        print("  get_capabilities function not found - needs to be implemented")
        return False
    except Exception as e:
        logger.error(f"Failed to test capability reporting: {e}")
        return False


def main():
    """Run all graceful degradation tests."""
    print("\n" + "="*80)
    print("KNOWLEDGE ENGINE GRACEFUL DEGRADATION TEST SUITE")
    print("="*80)

    tests = [
        ("Optional Imports Module", test_optional_imports_module),
        ("Main __init__ Degradation", test_main_init_graceful_degradation),
        ("Integration Degradation", test_integration_graceful_degradation),
        ("Capability Reporting", test_capability_reporting),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "[OK] PASSED" if result else "[FAIL] FAILED"
        print(f"  {status}: {test_name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All graceful degradation tests passed! [OK]")
        return 0
    else:
        print(f"\n  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
