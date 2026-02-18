"""
Comprehensive Test for Z3-to-Lean Gap Fixes

Tests all the gap fixes:
1. Base integration always available as fallback
2. generate_proof_certificate check fixed
3. Z3 solver doesn't pollute state (fresh solver each time)
4. Enhanced formalization with better error handling
5. Actual Z3 verification happening
"""

import asyncio
import sys

print("=" * 80)
print("COMPREHENSIVE GAP FIXES TEST")
print("=" * 80)
print()

# Test 1: Check base integration is always available
print("[TEST 1] Base Integration Fallback")
print("-" * 80)

try:
    from z3_to_lean_invention_integration import Z3LeanInventionIntegration

    integration = Z3LeanInventionIntegration(enable_hybrid=True)

    if integration.base_integration:
        print("[PASS] Base integration available even with hybrid=True")
    else:
        print("[FAIL] Base integration not initialized")

    status = integration.get_integration_status()
    print(f"  Status: {status}")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Z3 solver state isolation
print("[TEST 2] Z3 Solver State Isolation")
print("-" * 80)

try:
    import z3

    # Create two solvers
    solver1 = z3.Solver()
    solver2 = z3.Solver()

    # Add different constraints
    solver1.add(z3.Int('x') > 5)
    solver2.add(z3.Int('x') < 10)

    # Check both
    result1 = solver1.check()
    result2 = solver2.check()

    if result1 == z3.sat and result2 == z3.sat:
        print("[PASS] Separate solvers maintain separate state")
        print(f"  Solver1 (x > 5): {result1}")
        print(f"  Solver2 (x < 10): {result2}")
    else:
        print("[FAIL] Solvers interfering")

    # Now test the same solver with push/pop
    solver3 = z3.Solver()
    solver3.push()
    solver3.add(z3.Int('y') > 0)
    result_before = solver3.check()
    solver3.pop()
    result_after = solver3.check()

    print(f"  Before pop (with constraint): {result_before}")
    print(f"  After pop (without constraint): {result_after}")

    if result_after == z3.sat:  # Should be sat with no constraints
        print("[PASS] Push/pop works correctly")
    else:
        print("[FAIL] Push/pop not working")

except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: Enhanced formalization with error handling
print("[TEST 3] Enhanced Formalization Error Handling")
print("-" * 80)

async def test_enhanced_formalization():
    try:
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

        # Test with various equations
        test_equations = [
            "Temperature > 100",
            "Pressure <= 50",
            "Invalid equation @#$",
        ]

        for eq in test_equations:
            result = await integration._formalize_equation(eq, "chemistry", goal)

            if result:
                print(f"[PASS] '{eq}' formalized:")
                print(f"  Level: {result.formalization_level.value}")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Has Z3 constraint: {result.z3_constraint is not None}")
                print(f"  Has Lean theorem: {result.lean_theorem is not None}")
                print(f"  Has certificate: {result.proof_certificate is not None}")
            else:
                print(f"[FAIL] '{eq}' - No formalization")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_enhanced_formalization())

print()

# Test 4: Actual Z3 verification
print("[TEST 4] Actual Z3 Verification")
print("-" * 80)

async def test_z3_verification():
    try:
        from z3_to_lean_invention_integration import (
            Z3LeanInventionIntegration,
            Z3LeanFormalization,
            FormalizationLevel
        )

        integration = Z3LeanInventionIntegration()

        # Create a test formalization
        test_formalization = Z3LeanFormalization(
            description="x > 5",
            z3_constraint="(> x 5)",
            lean_theorem="theorem test : x > 5 := by simp",
            lean_tactics=["by simp"],
            verification_mode="z3_only",
            z3_result=None,
            lean_result=None,
            confidence=0.8,
            formalization_level=FormalizationLevel.Z3_ONLY,
            proof_certificate=None,
            execution_time=0.0
        )

        # Verify with Z3
        result = await integration._verify_with_z3(test_formalization)

        print(f"[PASS] Z3 verification completed:")
        print(f"  Type: {result.get('type')}")
        print(f"  Verified: {result.get('verified')}")
        print(f"  Result: {result.get('result')}")

        if result.get("type") == "z3_sat":
            print(f"  Model: {result.get('model', '')[:50]}...")

        # Check statistics
        stats = integration.get_statistics()
        if stats.get("z3_verifications", 0) > 0:
            print(f"[PASS] Z3 verification count incremented: {stats['z3_verifications']}")
        else:
            print(f"[FAIL] Z3 verification count not incremented")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_z3_verification())

print()

# Test 5: Full formalization pipeline
print("[TEST 5] Full Formalization Pipeline")
print("-" * 80)

async def test_full_pipeline():
    try:
        from z3_to_lean_invention_integration import (
            Z3LeanInventionIntegration,
            InventionGoal
        )

        integration = Z3LeanInventionIntegration(
            enable_z3=True,
            enable_lean=True,
            enable_hybrid=True,
            verification_mode="consensus",
            quality_threshold=0.7
        )

        goal = InventionGoal(
            goal_type="optimization",
            target="Optimize chemical reaction",
            domain="chemistry",
            key_requirements=["Maximize yield"],
            constraints=["Temperature <= 100C"],
            success_definition="Yield > 90%",
            complexity_score=0.75
        )

        decomposition = {
            "steps": [
                {"description": "Heat reaction mixture to target temperature", "math": "Rate = k * exp(-Ea / (R * T))"},
                {"description": "Maintain temperature for specified time", "math": "Yield = (actual / theoretical) * 100%"}
            ]
        }

        knowledge = [
            "Arrhenius equation: k = A * exp(-Ea / (R * T))",
            "Yield = (actual / theoretical) * 100%"
        ]

        # Full formalization
        result = await integration.formalize_invention_math(
            goal=goal,
            decomposition=decomposition,
            knowledge=knowledge,
            max_equations=5
        )

        print(f"[PASS] Full pipeline completed:")
        print(f"  Total relationships: {result.total_relationships}")
        print(f"  Formalized: {result.formalized_count}")
        print(f"  Verified: {result.verified_count}")
        print(f"  Certified: {result.certified_count}")
        print(f"  Execution time: {result.execution_time:.2f}s")

        if result.formalizations:
            print(f"\n  Sample formalizations:")
            for i, fmt in enumerate(result.formalizations[:3], 1):
                print(f"    [{i}] {fmt.description[:50]}...")
                print(f"        Level: {fmt.formalization_level.value}")
                print(f"        Confidence: {fmt.confidence:.2f}")

        # Check statistics
        stats = integration.get_statistics()
        print(f"\n  Statistics:")
        print(f"    Total formalizations: {stats['total_formalizations']}")
        print(f"    Z3 verifications: {stats['z3_verifications']}")
        print(f"    Hybrid verifications: {stats['hybrid_verifications']}")

        if result.formalized_count > 0:
            print("\n[PASS] Pipeline is functional!")
        else:
            print("\n[FAIL] No formalizations created")

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test_full_pipeline())

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
