#!/usr/bin/env python3
"""
Test script to verify MDAP Enhanced Red Flagging Integration

This script verifies that the MDAP reliability adapter correctly integrates
the enhanced red flagging system.
"""

import sys
import os

# Add reliability plugin to path
plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "reliability-plugin"))
if plugin_path not in sys.path:
    sys.path.insert(0, plugin_path)

def test_imports():
    """Test that all imports work correctly"""
    print("=" * 60)
    print("Testing Imports")
    print("=" * 60)

    try:
        from reliability_plugin.adapters.mdap.mdap_reliability_adapter import (
            MDAPReliabilityAdapter,
            create_mdap_adapter,
            solve_with_guardrails,
            solve_with_redflagging
        )
        print("✓ MDAP adapter imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_enhanced_redflagging_imports():
    """Test enhanced red flagging imports"""
    print("\n" + "=" * 60)
    print("Testing Enhanced Red Flagging Imports")
    print("=" * 60)

    try:
        from reliability_plugin.adapters.mdap.mdap_reliability_adapter import (
            ENHANCED_REDFLAGGING_AVAILABLE,
            EnhancedRedFlagger,
            EnhancedRedFlagRules,
            RedFlag,
            RedFlagSeverity
        )

        if ENHANCED_REDFLAGGING_AVAILABLE:
            print("✓ Enhanced red flagging is available")
            print(f"  - EnhancedRedFlagger: {EnhancedRedFlagger}")
            print(f"  - EnhancedRedFlagRules: {EnhancedRedFlagRules}")
            print(f"  - RedFlag: {RedFlag}")
            print(f"  - RedFlagSeverity: {RedFlagSeverity}")
        else:
            print("○ Enhanced red flagging not available (optional)")

        return True
    except ImportError as e:
        print(f"✗ Enhanced red flagging import failed: {e}")
        return False

def test_adapter_creation():
    """Test adapter creation with enhanced red flagging"""
    print("\n" + "=" * 60)
    print("Testing Adapter Creation")
    print("=" * 60)

    try:
        from reliability_plugin.adapters.mdap.mdap_reliability_adapter import create_mdap_adapter

        adapter = create_mdap_adapter()

        print("✓ Adapter created successfully")
        print(f"  - Type: {type(adapter).__name__}")
        print(f"  - Enhanced red flagging enabled: {adapter.enhanced_redflagging_enabled}")
        print(f"  - Enhanced red flagger: {adapter.enhanced_redflagger is not None}")

        return adapter
    except (ImportError, AttributeError, TypeError) as e:
        print(f"✗ Adapter creation failed: {e}")
        return None

def test_adapter_status(adapter):
    """Test adapter status reporting"""
    print("\n" + "=" * 60)
    print("Testing Adapter Status")
    print("=" * 60)

    if not adapter:
        print("✗ No adapter to test")
        return False

    try:
        status = adapter.get_status()

        print("✓ Status retrieved successfully")
        print(f"  - MDAP Core Available: {status.get('mdap_core_available')}")
        print(f"  - MDAP MCP Available: {status.get('mdap_mcp_available')}")
        print(f"  - Guardrails Available: {status.get('guardrails_available')}")
        print(f"  - Enhanced Red Flagging Available: {status.get('enhanced_redflagging_available')}")
        print(f"  - LMQL Available: {status.get('lmql_available')}")

        # Check layers
        layers = status.get('layers', {})
        if 'enhanced_redflagging' in layers:
            print("\nEnhanced Red Flagging Layer:")
            for key, value in layers['enhanced_redflagging'].items():
                print(f"  - {key}: {value}")

        return True
    except (AttributeError, KeyError, TypeError) as e:
        print(f"✗ Status retrieval failed: {e}")
        return False

def test_statistics(adapter):
    """Test statistics tracking"""
    print("\n" + "=" * 60)
    print("Testing Statistics")
    print("=" * 60)

    if not adapter:
        print("✗ No adapter to test")
        return False

    try:
        stats = adapter.get_statistics()

        print("✓ Statistics retrieved successfully")
        print("  Statistics:")
        for key, value in stats.items():
            print(f"    - {key}: {value}")

        # Check for new statistics
        if 'enhanced_redflagging_used' in stats:
            print("\n✓ Enhanced red flagging statistics tracked")
        if 'red_flags_detected' in stats:
            print("✓ Red flag detection statistics tracked")

        return True
    except (AttributeError, KeyError, TypeError) as e:
        print(f"✗ Statistics retrieval failed: {e}")
        return False

def test_method_signatures():
    """Test that new methods exist"""
    print("\n" + "=" * 60)
    print("Testing Method Signatures")
    print("=" * 60)

    try:
        from reliability_plugin.adapters.mdap.mdap_reliability_adapter import MDAPReliabilityAdapter

        # Check for new methods
        methods = [
            'solve_with_enhanced_redflagging',
            '_create_enhanced_redflagger',
            '_create_default_redflag_rules',
            '_solve_with_core_redflagging',
            '_convert_to_dict_result',
            '_extract_statistics'
        ]

        all_exist = True
        for method in methods:
            if hasattr(MDAPReliabilityAdapter, method):
                print(f"✓ Method exists: {method}")
            else:
                print(f"✗ Method missing: {method}")
                all_exist = False

        return all_exist
    except (ImportError, AttributeError) as e:
        print(f"✗ Method signature check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("MDAP Enhanced Red Flagging Integration Test")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Enhanced Red Flagging Imports", test_enhanced_redflagging_imports()))
    results.append(("Method Signatures", test_method_signatures()))

    adapter = test_adapter_creation()
    results.append(("Adapter Creation", adapter is not None))

    if adapter:
        results.append(("Adapter Status", test_adapter_status(adapter)))
        results.append(("Statistics", test_statistics(adapter)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Integration successful.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
