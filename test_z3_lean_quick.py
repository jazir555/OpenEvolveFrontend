"""
Quick test for Z3-to-Lean improvements
"""

import sys

print("=" * 80)
print("Z3-TO-LEAN IMPROVEMENTS TEST")
print("=" * 80)
print()

# Test 1: Import availability flags
print("[TEST 1] Availability Flags")
print("-" * 80)

try:
    from enhanced_z3_to_lean_integration import ENHANCED_INTEGRATION_AVAILABLE
    from z3_to_lean_integration import BASE_INTEGRATION_AVAILABLE
    print(f"[PASS] ENHANCED_INTEGRATION_AVAILABLE = {ENHANCED_INTEGRATION_AVAILABLE}")
    print(f"[PASS] BASE_INTEGRATION_AVAILABLE = {BASE_INTEGRATION_AVAILABLE}")
except Exception as e:
    print(f"[FAIL] {e}")
    sys.exit(1)

print()

# Test 2: NL to Z3 constraint conversion
print("[TEST 2] NL to Z3 Constraint Conversion")
print("-" * 80)

try:
    from z3_to_lean_invention_integration import Z3LeanInventionIntegration

    integration = Z3LeanInventionIntegration(quality_threshold=0.7)

    # Test various patterns
    test_cases = [
        ("Temperature > 100", "chemistry"),
        ("Pressure <= 50", "physics"),
        ("Concentration = moles / volume", "chemistry"),
        ("Yield greater than 90%", "chemistry"),
        ("Rate proportional to temperature", "physics"),
    ]

    for text, domain in test_cases:
        constraint = integration._nl_to_z3_constraint(text, domain)
        if constraint:
            print(f"[PASS] '{text}' -> {constraint[:50]}...")
        else:
            print(f"[FAIL] '{text}' -> None")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Basic formalization
print("[TEST 3] Basic Formalization")
print("-" * 80)

try:
    import asyncio

    async def test_formalization():
        from z3_to_lean_invention_integration import Z3LeanInventionIntegration, InventionGoal

        integration = Z3LeanInventionIntegration(quality_threshold=0.7)

        goal = InventionGoal(
            goal_type="test",
            target="Test goal",
            domain="chemistry",
            key_requirements=[],
            constraints=[],
            success_definition="",
            complexity_score=0.5
        )

        result = await integration._formalize_basic("Temperature > 100", "chemistry", goal)

        if result:
            print(f"[PASS] Basic formalization created:")
            print(f"  Description: {result.description}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Level: {result.formalization_level.value}")
            if result.lean_theorem:
                lines = result.lean_theorem.split('\n')
                print(f"  Theorem: {lines[0][:50]}...")
            print(f"  Passes threshold: {result.confidence >= 0.7}")
        else:
            print(f"[FAIL] No formalization created")

    asyncio.run(test_formalization())

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Statistics
print("[TEST 4] Integration Statistics")
print("-" * 80)

try:
    stats = integration.get_statistics()
    print("[PASS] Statistics retrieved:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"[FAIL] {e}")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
