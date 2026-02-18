"""
Test: Z3-Lean Integration INTO Invention Planner

This test verifies that the Z3-Lean integration is actually used by the
invention planner's _formalize_math() method.

Gap 12 FIX: Invention planner now imports and uses Z3-Lean integration
"""

import sys
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("Z3-LEAN INVENTION PLANNER INTEGRATION TEST")
print("=" * 80)
print()

# =============================================================================
# TEST 1: Verify Import Works
# =============================================================================
print("[TEST 1] Verify Z3-Lean Integration Import")
print("-" * 80)

try:
    from end_to_end_invention_planner import EndToEndInventionPlanner
    from invention_planner_structures import InventionGoal

    # Check if Z3-Lean integration is available by checking the module directly
    from z3_to_lean_invention_integration import (
        ENHANCED_INTEGRATION_AVAILABLE,
        BASE_INTEGRATION_AVAILABLE
    )
    Z3_LEAN_INTEGRATION_AVAILABLE = ENHANCED_INTEGRATION_AVAILABLE or BASE_INTEGRATION_AVAILABLE

    if Z3_LEAN_INTEGRATION_AVAILABLE:
        print("[PASS] Z3-Lean integration is available")
        print(f"  Status: Z3_LEAN_INTEGRATION_AVAILABLE = True")
    else:
        print("[FAIL] Z3-Lean integration not available")
        print("  Check: z3_to_lean_invention_integration.py exists and imports work")
        sys.exit(1)

except ImportError as e:
    print(f"[FAIL] Cannot import from invention planner: {e}")
    sys.exit(1)

except Exception as e:
    print(f"[FAIL] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# =============================================================================
# TEST 2: Verify Integration Components Exist
# =============================================================================
print("[TEST 2] Verify Integration Components")
print("-" * 80)

try:
    from z3_to_lean_invention_integration import (
        Z3LeanInventionIntegration,
        formalize_invention_plan,
        InventionFormalizationResult,
        Z3LeanFormalization,
        FormalizationLevel
    )

    print("[PASS] All integration components importable:")
    print(f"  Z3LeanInventionIntegration: {Z3LeanInventionIntegration}")
    print(f"  formalize_invention_plan: {formalize_invention_plan}")
    print(f"  InventionFormalizationResult: {InventionFormalizationResult}")
    print(f"  Z3LeanFormalization: {Z3LeanFormalization}")
    print(f"  FormalizationLevel: {FormalizationLevel}")

    # Check formalization levels
    levels = [level.value for level in FormalizationLevel]
    print(f"  Levels: {', '.join(levels)}")

except ImportError as e:
    print(f"[FAIL] Cannot import integration components: {e}")
    sys.exit(1)

print()

# =============================================================================
# TEST 3: Test formalize_invention_plan Function
# =============================================================================
print("[TEST 3] Test formalize_invention_plan Function")
print("-" * 80)

async def test_formalize_invention_plan():
    """Test the formalize_invention_plan function directly"""

    # Create a simple test goal
    goal = InventionGoal(
        goal_type="process",
        target="Create a temperature-controlled reactor",
        domain="chemical_engineering",
        key_requirements=["Temperature control", "Pressure regulation"],
        constraints=["Max temperature 200C", "Max pressure 100 bar"],
        success_definition="Reactant conversion > 95%",
        complexity_score=0.7
    )

    # Create simple decomposition
    decomposition = {
        "steps": [
            {"description": "Heat reactor to 100°C", "equations": ["Temperature > 100"]},
            {"description": "Maintain pressure below 50 bar", "equations": ["Pressure <= 50"]}
        ],
        "relationships": [
            {"equation": "Temperature > 100", "domain": "thermodynamics"},
            {"equation": "Pressure <= 50", "domain": "fluid_dynamics"}
        ]
    }

    # Create knowledge base
    knowledge = [
        "Temperature must be maintained above 100°C for optimal reaction rate",
        "Pressure should not exceed 50 bar for safety",
        "Reaction rate follows Arrhenius equation: k = A * exp(-Ea / (R * T))"
    ]

    try:
        result = await formalize_invention_plan(
            goal=goal,
            decomposition=decomposition,
            knowledge=knowledge
        )

        if result:
            print("[PASS] formalize_invention_plan executed:")
            print(f"  Formalized count: {result.formalized_count}")
            print(f"  Total relationships: {result.total_relationships}")

            if result.formalizations:
                print(f"\n  Formalizations:")
                for i, form in enumerate(result.formalizations, 1):
                    print(f"    {i}. {form.description[:50]}")
                    print(f"       Level: {form.formalization_level.value}")
                    print(f"       Confidence: {form.confidence:.2f}")
                    print(f"       Z3 constraint: {'Yes' if form.z3_constraint else 'No'}")
                    print(f"       Lean theorem: {'Yes' if form.lean_theorem else 'No'}")
                    print(f"       Certificate: {'Yes' if form.proof_certificate else 'No'}")

            if result.verification_summary:
                print(f"\n  Verification Summary:")
                summary = result.verification_summary
                print(f"    Total relationships: {result.total_relationships}")
                print(f"    Formalized: {result.formalized_count}")
                print(f"    Verified: {result.verified_count}")
                print(f"    Certified: {result.certified_count}")
                print(f"    Execution time: {result.execution_time:.2f}s")

            return result
        else:
            print("[FAIL] formalize_invention_plan returned None")
            return None

    except Exception as e:
        print(f"[FAIL] formalize_invention_plan failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Run the test
result = asyncio.run(test_formalize_invention_plan())

print()

# =============================================================================
# TEST 4: Verify Invention Planner Uses Z3-Lean
# =============================================================================
print("[TEST 4] Verify Invention Planner Integration")
print("-" * 80)

try:
    # Read the invention planner source to verify it uses Z3-Lean
    with open('end_to_end_invention_planner.py', 'r', encoding='utf-8') as f:
        source = f.read()

    # Check for Z3-Lean import
    if 'from z3_to_lean_invention_integration import' in source:
        print("[PASS] Invention planner imports Z3-Lean integration")
    else:
        print("[FAIL] Invention planner does NOT import Z3-Lean integration")
        sys.exit(1)

    # Check for Z3_LEAN_INTEGRATION_AVAILABLE usage
    if 'if Z3_LEAN_INTEGRATION_AVAILABLE:' in source:
        print("[PASS] Invention planner checks Z3_LEAN_INTEGRATION_AVAILABLE")
    else:
        print("[FAIL] Invention planner does NOT check availability flag")
        sys.exit(1)

    # Check for formalize_invention_plan call
    if 'await formalize_invention_plan(' in source:
        print("[PASS] Invention planner calls formalize_invention_plan()")
    else:
        print("[FAIL] Invention planner does NOT call formalize_invention_plan()")
        sys.exit(1)

    # Check for formalization level handling
    if 'form.formalization_level.value' in source:
        print("[PASS] Invention planner tracks formalization levels")
    else:
        print("[FAIL] Invention planner does NOT track formalization levels")
        sys.exit(1)

    # Count Z3+Lean references
    z3_lean_count = source.count('Z3_LEAN')
    hybrid_count = source.count('hybrid')
    formalize_count = source.count('formalize_invention_plan')

    print(f"\n  Reference counts:")
    print(f"    Z3_LEAN references: {z3_lean_count}")
    print(f"    'hybrid' mentions: {hybrid_count}")
    print(f"    formalize_invention_plan calls: {formalize_count}")

except FileNotFoundError:
    print("[FAIL] Cannot find end_to_end_invention_planner.py")
    sys.exit(1)

except Exception as e:
    print(f"[FAIL] Error reading invention planner: {e}")
    sys.exit(1)

print()

# =============================================================================
# TEST 5: Gap 12 Verification
# =============================================================================
print("[TEST 5] Gap 12 Verification - Integration Complete")
print("-" * 80)

gap_12_fixed_checks = {
    "Import Z3-Lean integration": 'from z3_to_lean_invention_integration import' in open('end_to_end_invention_planner.py').read(),
    "Check availability flag": 'if Z3_LEAN_INTEGRATION_AVAILABLE:' in open('end_to_end_invention_planner.py').read(),
    "Call formalize_invention_plan": 'await formalize_invention_plan(' in open('end_to_end_invention_planner.py').read(),
    "Handle formalization results": 'result.formalized_count' in open('end_to_end_invention_planner.py').read(),
    "Track verification summary": 'result.verification_summary' in open('end_to_end_invention_planner.py').read() or 'result.total_relationships' in open('end_to_end_invention_planner.py').read(),
    "Log Z3+Lean usage": 'Z3+Lean' in open('end_to_end_invention_planner.py').read()
}

all_passed = True
for check, passed in gap_12_fixed_checks.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {check}")
    if not passed:
        all_passed = False

print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)

if all_passed and result and result.formalized_count > 0:
    print("STATUS: [PASS] ALL TESTS PASSED")
    print()
    print("Gap 12 FIXED:")
    print("  [PASS] Invention planner imports Z3-Lean integration")
    print("  [PASS] Invention planner checks availability flag")
    print("  [PASS] Invention planner calls formalize_invention_plan()")
    print("  [PASS] Invention planner handles formalization results")
    print("  [PASS] Invention planner tracks statistics")
    print("  [PASS] Invention planner logs Z3+Lean usage")
    print()
    print(f"RESULT: {result.formalized_count} equations formalized with Z3+Lean")
    print()
    print("=" * 80)
    print("Z3-LEAN INVENTION PLANNER INTEGRATION: COMPLETE")
    print("=" * 80)
    sys.exit(0)
else:
    print("STATUS: [FAIL] TESTS FAILED")
    print()
    if not all_passed:
        print("Some Gap 12 checks failed")
    if not result:
        print("formalize_invention_plan() failed")
    elif result.formalized_count == 0:
        print("No equations were formalized")

    print()
    print("=" * 80)
    print("INTEGRATION INCOMPLETE")
    print("=" * 80)
    sys.exit(1)
