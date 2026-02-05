"""
Unified Evolution API - Verification Script
===========================================

Quick verification that the unified evolution API works correctly.
Tests basic functionality, integration points, and all 7 domains.

Author: Unified Evolution Team
Date: 2026-01-30
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openevolve.unified.unified_evolution_api import (
    UnifiedEvolutionAPI,
    evolve,
    quick_evolve,
    evolve_no_gauntlet,
    evolve_batch,
    EvolutionResult,
    SystemMode,
    ProgressUpdate
)


# ============================================================================
# VERIFICATION TESTS
# ============================================================================

async def verify_imports():
    """Verify all imports work"""
    print("\n" + "="*80)
    print("VERIFY 1: Imports")
    print("="*80)

    try:
        from openevolve.unified.unified_evolution_api import (
            UnifiedEvolutionAPI,
            evolve,
            quick_evolve,
            evolve_no_gauntlet,
            evolve_batch
        )
        print("[OK] All main imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


async def verify_basic_evolution():
    """Verify basic evolution works"""
    print("\n" + "="*80)
    print("VERIFY 2: Basic Evolution")
    print("="*80)

    try:
        result = await evolve(
            problem="Optimize function: f(x) = x^2",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )

        assert isinstance(result, EvolutionResult), "Result is EvolutionResult"
        assert result.best_solution is not None, "Has solution"
        assert len(result.best_solution) > 0, "Solution not empty"
        assert result.final_score >= 0.0, "Valid score"
        assert result.total_time >= 0.0, "Valid time"

        print(f"[OK] Basic evolution successful")
        print(f"   Solution: {result.best_solution[:50]}...")
        print(f"   Score: {result.final_score:.3f}")
        print(f"   Strategy: {result.strategy_used.system}/{result.strategy_used.mode}")
        return True

    except Exception as e:
        print(f"[FAIL] Basic evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def verify_progress_callback():
    """Verify progress callback works"""
    print("\n" + "="*80)
    print("VERIFY 3: Progress Callback")
    print("="*80)

    try:
        updates = []

        def callback(update):
            updates.append(update)

        result = await evolve(
            problem="Test problem",
            domain="general",
            callback=callback,
            run_gauntlet=False,
            store_knowledge=False
        )

        assert len(updates) > 0, "Callback was called"
        assert updates[-1].percent_complete == 100, "Final update at 100%"

        print(f"[OK] Progress callback successful")
        print(f"   Updates received: {len(updates)}")
        print(f"   Stages: {set(u.stage for u in updates)}")
        return True

    except Exception as e:
        print(f"[FAIL] Progress callback failed: {e}")
        return False


async def verify_convenience_functions():
    """Verify convenience functions work"""
    print("\n" + "="*80)
    print("VERIFY 4: Convenience Functions")
    print("="*80)

    try:
        # Test quick_evolve
        solution = await quick_evolve(
            problem="Simple optimization",
            domain="general"
        )
        assert isinstance(solution, str), "Returns string"
        assert len(solution) > 0, "Solution not empty"
        print("[OK] quick_evolve() works")

        # Test evolve_no_gauntlet
        result = await evolve_no_gauntlet(
            problem="Test problem",
            domain="general"
        )
        assert result.gauntlet_result is None, "No gauntlet result"
        print("[OK] evolve_no_gauntlet() works")

        # Test evolve_batch
        problems = ["Problem 1", "Problem 2"]
        results = await evolve_batch(
            problems=problems,
            domain="general",
            max_concurrent=2
        )
        assert len(results) == len(problems), "All results present"
        print("[OK] evolve_batch() works")

        return True

    except Exception as e:
        print(f"[FAIL] Convenience functions failed: {e}")
        return False


async def verify_result_serialization():
    """Verify result save/load works"""
    print("\n" + "="*80)
    print("VERIFY 5: Result Serialization")
    print("="*80)

    try:
        import tempfile
        import os

        # Run evolution
        result = await evolve(
            problem="Test problem",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )

        # Save result
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            result.save(filepath)
            print(f"[OK] Result saved to {filepath}")

            # Load result
            loaded = EvolutionResult.load(filepath)
            print(f"[OK] Result loaded from {filepath}")

            # Verify
            assert loaded.best_solution == result.best_solution, "Solution matches"
            assert loaded.final_score == result.final_score, "Score matches"

            print("[OK] Serialization successful")
            return True

        finally:
            # Cleanup
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        print(f"[FAIL] Serialization failed: {e}")
        return False


async def verify_all_domains():
    """Verify all domains work"""
    print("\n" + "="*80)
    print("VERIFY 6: All Domains")
    print("="*80)

    domains = {
        'finance': "Maximize portfolio Sharpe ratio",
        'trading': "Develop momentum trading strategy",
        'science': "Optimize experimental design",
        'engineering': "Minimize structural weight",
        'pharma': "Optimize drug binding affinity",
        'web': "Maximize Lighthouse performance",
        'general': "Solve traveling salesman problem"
    }

    results = {}

    for domain, problem in domains.items():
        try:
            result = await evolve(
                problem=problem,
                domain=domain,
                run_gauntlet=False,
                store_knowledge=False
            )
            results[domain] = result
            print(f"[OK] {domain:12s} - Score: {result.final_score:.3f}, Strategy: {result.strategy_used.mode}")

        except Exception as e:
            print(f"[FAIL] {domain:12s} - Failed: {e}")
            results[domain] = None

    success_count = sum(1 for r in results.values() if r is not None)
    print(f"\n[OK] Domains successful: {success_count}/{len(domains)}")

    return success_count == len(domains)


async def verify_api_class():
    """Verify UnifiedEvolutionAPI class works"""
    print("\n" + "="*80)
    print("VERIFY 7: API Class")
    print("="*80)

    try:
        api = UnifiedEvolutionAPI(
            enable_gauntlets=False,
            enable_knowledge_extraction=False
        )

        result = await api.evolve(
            problem="Test problem",
            domain="general"
        )

        assert result is not None, "Result exists"
        assert isinstance(result, EvolutionResult), "Correct type"

        print("[OK] API class works")
        print(f"   Result score: {result.final_score:.3f}")
        return True

    except Exception as e:
        print(f"[FAIL] API class failed: {e}")
        return False


async def verify_error_handling():
    """Verify graceful error handling"""
    print("\n" + "="*80)
    print("VERIFY 8: Error Handling")
    print("="*80)

    try:
        # This should handle errors gracefully
        api = UnifiedEvolutionAPI(
            enable_gauntlets=False,
            enable_knowledge_extraction=False
        )

        # Mock executor to fail
        import unittest.mock as mock
        with mock.patch.object(api, '_execute_openevolve', side_effect=Exception("Test error")):
            result = await api.evolve(
                problem="Test problem",
                domain="general",
                run_gauntlet=False,
                store_knowledge=False
            )

            assert result.error is not None, "Error captured"
            assert "Test error" in result.error, "Error message present"
            assert result.final_score == 0.0, "Zero score on error"

        print("[OK] Error handling works correctly")
        print(f"   Error captured: {result.error}")
        return True

    except Exception as e:
        print(f"[FAIL] Error handling verification failed: {e}")
        return False


# ============================================================================
# MAIN VERIFICATION
# ============================================================================

async def run_verification():
    """Run all verification tests"""
    print("\n" + "="*80)
    print("UNIFIED EVOLUTION API - VERIFICATION")
    print("="*80)
    print(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    tests = [
        ("Imports", verify_imports),
        ("Basic Evolution", verify_basic_evolution),
        ("Progress Callback", verify_progress_callback),
        ("Convenience Functions", verify_convenience_functions),
        ("Result Serialization", verify_result_serialization),
        ("All Domains", verify_all_domains),
        ("API Class", verify_api_class),
        ("Error Handling", verify_error_handling)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n[FAIL] {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for test_name, success in results.items():
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"{status:8s} | {test_name}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print("="*80)
    print(f"Total: {passed}/{total} tests passed")
    print("="*80)

    if passed == total:
        print("\n🎉 ALL VERIFICATION TESTS PASSED!")
        print("The Unified Evolution API is working correctly.")
        return 0
    else:
        print(f"\n[WARN]  {total - passed} test(s) failed. Please review the errors above.")
        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    exit_code = asyncio.run(run_verification())
    sys.exit(exit_code)
