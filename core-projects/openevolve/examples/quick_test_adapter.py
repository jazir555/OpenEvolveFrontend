#!/usr/bin/env python3
"""
Quick test script for LoongFlow adapter

This script verifies that the adapter can be imported and used correctly.
"""

import asyncio
from openevolve.integrations import LoongFlowAdapter


def test_import():
    """Test 1: Import the adapter"""
    print("[Test 1] Importing LoongFlowAdapter...")
    try:
        from openevolve.integrations import LoongFlowAdapter
        print("[PASS] Import successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_initialization():
    """Test 2: Initialize the adapter"""
    print("\n[Test 2] Initializing adapter...")
    try:
        adapter = LoongFlowAdapter({"max_iterations": 10})
        print(f"[PASS] Adapter initialized: {adapter}")
        return True
    except Exception as e:
        print(f"[FAIL] Initialization failed: {e}")
        return False


def test_availability_check():
    """Test 3: Check availability"""
    print("\n[Test 3] Checking availability...")
    try:
        adapter = LoongFlowAdapter({})
        available = adapter.is_available()
        print(f"[PASS] Availability check: {available}")
        return True
    except Exception as e:
        print(f"[FAIL] Availability check failed: {e}")
        return False


def test_capabilities():
    """Test 4: Get capabilities"""
    print("\n[Test 4] Getting capabilities...")
    try:
        adapter = LoongFlowAdapter({})
        capabilities = adapter.get_capabilities()
        print(f"[PASS] Capabilities: {capabilities}")
        return True
    except Exception as e:
        print(f"[FAIL] Get capabilities failed: {e}")
        return False


async def test_evolve_fallback():
    """Test 5: Test evolve method (fallback mode)"""
    print("\n[Test 5] Testing evolve method...")
    try:
        adapter = LoongFlowAdapter({})
        result = await adapter.evolve(
            problem="Test problem",
            domain="general"
        )

        # Verify result structure
        required_keys = [
            "best_solution",
            "best_fitness",
            "total_evaluations",
            "improvement_rate",
            "iterations_performed",
            "strategy_used",
            "source"
        ]

        missing_keys = [k for k in required_keys if k not in result]
        if missing_keys:
            print(f"[FAIL] Missing keys in result: {missing_keys}")
            return False

        print(f"[PASS] Evolve method works")
        print(f"  - Best fitness: {result['best_fitness']}")
        print(f"  - Strategy: {result['strategy_used']}")
        print(f"  - Source: {result['source']}")
        return True
    except Exception as e:
        print(f"[FAIL] Evolve failed: {e}")
        return False


def test_config_mapping():
    """Test 6: Test configuration mapping"""
    print("\n[Test 6] Testing config mapping...")
    try:
        config = {
            "max_iterations": 50,
            "population_size": 10,
            "timeout": 120,
            "enable_planning": True,
            "enable_memory": False,
            "llm_config": {"model": "gpt-4"}
        }

        adapter = LoongFlowAdapter(config)
        mapped = adapter._map_config(config)

        # Verify mapping
        assert mapped["max_iterations"] == 50
        assert mapped["population_size"] == 10
        assert mapped["timeout"] == 120
        assert mapped["enable_planning"] is True
        assert mapped["enable_memory"] is False
        assert mapped["llm_config"]["model"] == "gpt-4"

        print("[PASS] Config mapping works correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Config mapping failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("="*60)
    print("LoongFlow Adapter - Quick Test Suite")
    print("="*60)

    tests = [
        test_import,
        test_initialization,
        test_availability_check,
        test_capabilities,
        test_config_mapping,
    ]

    # Run sync tests
    results = []
    for test in tests:
        results.append(test())

    # Run async test
    results.append(await test_evolve_fallback())

    # Summary
    print("\n" + "="*60)
    print(f"Test Summary: {sum(results)}/{len(results)} passed")
    print("="*60)

    if all(results):
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
