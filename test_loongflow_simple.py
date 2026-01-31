"""
Simple test for optional LoongFlow integration
"""
import asyncio
import sys
sys.path.insert(0, '.')

from openevolve.unified.unified_evolution_api import (
    evolve,
    evolve_openevolve_only,
    evolve_with_loongflow,
)
from openevolve.integrations.loongflow_checker import is_loongflow_available

async def main():
    print("\n" + "=" * 60)
    print("OPTIONAL LOONGFLOW INTEGRATION TEST")
    print("=" * 60)

    loongflow_available = is_loongflow_available()
    print(f"\nLoongFlow Available: {loongflow_available}")

    # Test 1: Default evolve (should use LoongFlow if available)
    print("\n--- Test 1: evolve() with defaults ---")
    result1 = await evolve(
        problem="Optimize a simple function",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )
    print(f"System used: {result1.system_used}")
    print(f"Mode used: {result1.mode_used}")
    print(f"Final score: {result1.final_score:.3f}")
    print(f"LoongFlow was used: {result1.metadata.get('loongflow_was_used')}")

    # Test 2: OpenEvolve only (forced)
    print("\n--- Test 2: evolve_openevolve_only() ---")
    result2 = await evolve_openevolve_only(
        problem="Optimize a simple function",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False
    )
    print(f"System used: {result2.system_used}")
    print(f"Mode used: {result2.mode_used}")
    print(f"Final score: {result2.final_score:.3f}")
    assert result2.system_used == "openevolve", "Should use OpenEvolve"
    print("PASS: Correctly used OpenEvolve")

    # Test 3: With LoongFlow override
    print("\n--- Test 3: evolve(use_loongflow=False) ---")
    result3 = await evolve(
        problem="Optimize a simple function",
        domain="general",
        run_gauntlet=False,
        store_knowledge=False,
        use_loongflow=False  # Force OpenEvolve
    )
    print(f"System used: {result3.system_used}")
    print(f"Mode used: {result3.mode_used}")
    print(f"Final score: {result3.final_score:.3f}")
    assert result3.system_used == "openevolve", "Should use OpenEvolve (forced)"
    print("PASS: Correctly used OpenEvolve (override)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
