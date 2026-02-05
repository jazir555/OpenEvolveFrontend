"""
Test Optional LoongFlow Integration
====================================

Test script to verify that the unified evolution API works correctly
with or without LoongFlow available.

Author: Unified Evolution Team
Date: 2026-01-30
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openevolve.unified.unified_evolution_api import (
    UnifiedEvolutionAPI,
    evolve,
    evolve_openevolve_only,
    evolve_with_loongflow,
    EvolutionResult
)
from openevolve.unified.config import UnifiedEvolutionConfig
from openevolve.integrations.loongflow_checker import (
    LoongFlowChecker,
    is_loongflow_available
)


def test_loongflow_checker():
    """Test LoongFlow availability checker"""
    print("\n" + "=" * 60)
    print("Testing LoongFlow Checker")
    print("=" * 60)

    # Check availability
    available = is_loongflow_available()
    print(f"[OK] LoongFlow Available: {available}")

    # Get detailed info
    info = LoongFlowChecker.get_availability_info()
    print(f"   Version: {info.get('version', 'N/A')}")
    print(f"   Path: {info.get('path', 'N/A')}")
    if info.get('error'):
        print(f"   Error: {info['error']}")

    return available


async def test_evolve_default():
    """Test evolve() with default settings"""
    print("\n" + "=" * 60)
    print("Test 1: evolve() with default settings")
    print("=" * 60)

    try:
        result = await evolve(
            problem="Optimize a simple function",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )

        print(f"[OK] Evolution complete")
        print(f"   System used: {result.system_used}")
        print(f"   Mode used: {result.mode_used}")
        print(f"   Final score: {result.final_score:.3f}")
        print(f"   LoongFlow was available: {result.metadata.get('loongflow_was_available')}")
        print(f"   LoongFlow was used: {result.metadata.get('loongflow_was_used')}")

        return result
    except Exception as e:
        print(f"[FAIL] Evolution failed: {e}")
        return None


async def test_evolve_openevolve_only():
    """Test evolve_openevolve_only()"""
    print("\n" + "=" * 60)
    print("Test 2: evolve_openevolve_only()")
    print("=" * 60)

    try:
        result = await evolve_openevolve_only(
            problem="Optimize a simple function",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )

        print(f"[OK] Evolution complete")
        print(f"   System used: {result.system_used}")
        print(f"   Mode used: {result.mode_used}")
        print(f"   Final score: {result.final_score:.3f}")

        # Verify OpenEvolve was used
        assert result.system_used == "openevolve", "Should use OpenEvolve"
        print(f"   [OK] Correctly used OpenEvolve")

        return result
    except Exception as e:
        print(f"[FAIL] Evolution failed: {e}")
        return None


async def test_evolve_with_loongflow():
    """Test evolve_with_loongflow()"""
    print("\n" + "=" * 60)
    print("Test 3: evolve_with_loongflow()")
    print("=" * 60)

    try:
        result = await evolve_with_loongflow(
            problem="Optimize a simple function",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )

        print(f"[OK] Evolution complete")
        print(f"   System used: {result.system_used}")
        print(f"   Mode used: {result.mode_used}")
        print(f"   Final score: {result.final_score:.3f}")

        # Verify LoongFlow or fallback was used
        if result.system_used == "loongflow":
            print(f"   [OK] Correctly used LoongFlow")
        else:
            print(f"   ℹ️  LoongFlow not available, used OpenEvolve fallback")

        return result
    except Exception as e:
        print(f"[FAIL] Evolution failed: {e}")
        return None


async def test_evolve_with_override():
    """Test evolve() with use_loongflow parameter"""
    print("\n" + "=" * 60)
    print("Test 4: evolve() with use_loongflow=False override")
    print("=" * 60)

    try:
        result = await evolve(
            problem="Optimize a simple function",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False,
            use_loongflow=False  # Force OpenEvolve-only
        )

        print(f"[OK] Evolution complete")
        print(f"   System used: {result.system_used}")
        print(f"   Mode used: {result.mode_used}")
        print(f"   Final score: {result.final_score:.3f}")

        # Verify OpenEvolve was used despite LoongFlow availability
        assert result.system_used == "openevolve", "Should use OpenEvolve (forced)"
        print(f"   [OK] Correctly used OpenEvolve (override worked)")

        return result
    except Exception as e:
        print(f"[FAIL] Evolution failed: {e}")
        return None


async def test_config_with_loongflow_disabled():
    """Test config with LoongFlow disabled"""
    print("\n" + "=" * 60)
    print("Test 5: Config with LoongFlow disabled")
    print("=" * 60)

    try:
        # Create config with LoongFlow disabled
        config = UnifiedEvolutionConfig.openevolve_only()

        result = await evolve(
            problem="Optimize a simple function",
            domain="general",
            config=config,
            run_gauntlet=False,
            store_knowledge=False
        )

        print(f"[OK] Evolution complete")
        print(f"   System used: {result.system_used}")
        print(f"   Mode used: {result.mode_used}")
        print(f"   Final score: {result.final_score:.3f}")

        # Verify OpenEvolve was used
        assert result.system_used == "openevolve", "Should use OpenEvolve (config)"
        print(f"   [OK] Correctly used OpenEvolve (config worked)")

        return result
    except Exception as e:
        print(f"[FAIL] Evolution failed: {e}")
        return None


async def test_require_loongflow():
    """Test require_loongflow=True (should fail if LoongFlow not available)"""
    print("\n" + "=" * 60)
    print("Test 6: require_loongflow=True")
    print("=" * 60)

    # Check if LoongFlow is available
    loongflow_available = is_loongflow_available()

    if not loongflow_available:
        print("ℹ️  LoongFlow not available, testing error handling...")

        try:
            # Create config that requires LoongFlow
            config = UnifiedEvolutionConfig(
                enable_loongflow=True,
                require_loongflow=True
            )

            result = await evolve(
                problem="Optimize a simple function",
                domain="general",
                config=config,
                run_gauntlet=False,
                store_knowledge=False
            )

            # Should have failed
            if result.error:
                print(f"[OK] Correctly returned error: {result.error}")
            else:
                print(f"[FAIL] Should have failed but didn't")

        except Exception as e:
            print(f"[OK] Correctly raised exception: {e}")
    else:
        print("ℹ️  LoongFlow available, testing with require_loongflow=True...")

        try:
            config = UnifiedEvolutionConfig(
                enable_loongflow=True,
                require_loongflow=True
            )

            result = await evolve(
                problem="Optimize a simple function",
                domain="general",
                config=config,
                run_gauntlet=False,
                store_knowledge=False
            )

            print(f"[OK] Evolution complete")
            print(f"   System used: {result.system_used}")
            print(f"   Mode used: {result.mode_used}")

        except Exception as e:
            print(f"[FAIL] Unexpected failure: {e}")


async def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("OPTIONAL LOONGFLOW INTEGRATION TEST SUITE")
    print("=" * 80)

    # Test 1: Check availability
    loongflow_available = test_loongflow_checker()

    # Test 2: Default evolve
    await test_evolve_default()

    # Test 3: OpenEvolve only
    await test_evolve_openevolve_only()

    # Test 4: With LoongFlow (if available, or fallback)
    await test_evolve_with_loongflow()

    # Test 5: Override parameter
    await test_evolve_with_override()

    # Test 6: Config with LoongFlow disabled
    await test_config_with_loongflow_disabled()

    # Test 7: Require LoongFlow
    await test_require_loongflow()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
