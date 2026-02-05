#!/usr/bin/env python3
"""
End-to-End Integration Test for Z3 Constraint Hardening

This test demonstrates the complete Z3 integration workflow:
1. Parse natural language constraints
2. Encode to Z3 formulas
3. Invert constraints using formal logic
4. Verify satisfiability
5. Return hardened constraints

Following CLAUDE.md Law of Runtime Truth: Test actual behavior, not assumptions.
"""

import sys
import os
import json
import uuid
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from phase1_executor import (
    ConstraintHardener,
    Phase1Config,
    StructuredLogger
)


def test_e2e_z3_constraint_hardening():
    """End-to-end test of Z3 constraint hardening"""

    print("=" * 70)
    print("Z3 CONSTRAINT HARDENING - END-TO-END TEST")
    print("=" * 70)

    # Setup
    config = Phase1Config(
        TIMEOUT_MS=15000,
        CONSTRAINT_HARDENING_TIMEOUT_MS=5000,
        ASSUMPTION_MINING_TIMEOUT_MS=5000,
        CONTRADICTION_DETECTION_TIMEOUT_MS=10000,
        FALSIFICATION_TIMEOUT_MS=5000,
        MAX_ASSUMPTIONS=100,
        MAX_CONSTRAINTS=1000,
        MAX_CONTRADICTIONS=100,
        MAX_FALSIFICATION_ATTEMPTS=50,
        CIRCUIT_BREAKER_THRESHOLD=5,
        CIRCUIT_BREAKER_TIMEOUT_MS=60000,
        MIN_ASSUMPTION_CONFIDENCE=0.3,
        MIN_ROBUSTNESS_SCORE=0.5,
        ENABLE_TACIT_MINING=True,
        ENABLE_LEAN4_INTEGRATION=False,
        ENABLE_RED_TEAM_PROTOCOL=True,
        ENABLE_Z3_CONSTRAINT_HARDENING=True,
    )

    logger = StructuredLogger('E2ETest')
    hardener = ConstraintHardener(config, logger)

    correlation_id = str(uuid.uuid4())

    # Test Problem
    problem_description = """
    The system cannot process more than 1000 items per second.
    The temperature is impossible to exceed 500 degrees Celsius.
    Pressure must remain below 200 psi at all times.
    Data transfer rate is limited to 10 Gbps.
    """

    print(f"\n[PROBLEM DESCRIPTION]")
    print(problem_description)

    # Execute
    print(f"\n[EXECUTING CONSTRAINT HARDENING]")
    print(f"Z3 Enabled: {hardener.z3_enabled}")
    print(f"Correlation ID: {correlation_id}")

    constraints = hardener.harden_constraints(
        problem_description=problem_description,
        correlation_id=correlation_id
    )

    # Results
    print(f"\n[RESULTS]")
    print(f"Constraints Extracted: {len(constraints)}")

    print("\n" + "=" * 70)
    print("DETAILED CONSTRAINT ANALYSIS")
    print("=" * 70)

    for i, constraint in enumerate(constraints, 1):
        print(f"\n[CONSTRAINT {i}]")
        print(f"ID: {constraint['constraint_id']}")
        print(f"Category: {constraint['category']}")
        print(f"\nOriginal:")
        print(f"  {constraint['description']}")
        print(f"\nInverted:")
        print(f"  {constraint['inverted_description']}")
        print(f"\nFormalization:")
        print(f"  Formalized: {constraint['formalized']}")
        print(f"  Z3 Encoded: {constraint['z3_encoded']}")

        if constraint.get('z3_encoded'):
            print(f"\nZ3 Formula Details:")
            if 'fol_structure' in constraint:
                fol = constraint['fol_structure']
                print(f"  Variables: {fol['variables']}")
                print(f"  Quantifiers: {fol['quantifiers']}")
                print(f"  Predicates: {fol['predicates']}")
            if 'z3_formula' in constraint:
                print(f"  Formula: {constraint['z3_formula']}")
            if 'simplified_formula' in constraint:
                print(f"  Simplified: {constraint['simplified_formula']}")
            if 'inverted_formula' in constraint:
                print(f"  Inverted: {constraint['inverted_formula']}")

        print(f"\nSatisfiability:")
        satisfiable = constraint.get('satisfiable')
        if satisfiable is True:
            print(f"  Status: SATISFIABLE")
            if 'model' in constraint and constraint['model']:
                print(f"  Model: {constraint['model']}")
        elif satisfiable is False:
            print(f"  Status: UNSATISFIABLE")
            if 'unsat_reason' in constraint:
                print(f"  Reason: {constraint['unsat_reason']}")
        else:
            print(f"  Status: NOT CHECKED")

        if 'z3_error' in constraint:
            print(f"\nZ3 Error:")
            print(f"  {constraint['z3_error']}")

    # Summary Statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    total = len(constraints)
    formalized = sum(1 for c in constraints if c['formalized'])
    z3_encoded = sum(1 for c in constraints if c['z3_encoded'])
    satisfiable = sum(1 for c in constraints if c.get('satisfiable') is True)
    unsatisfiable = sum(1 for c in constraints if c.get('satisfiable') is False)

    print(f"Total Constraints: {total}")
    print(f"Formalized: {formalized} ({formalized/total*100:.1f}%)")
    print(f"Z3 Encoded: {z3_encoded} ({z3_encoded/total*100:.1f}%)")
    print(f"Satisfiable: {satisfiable} ({satisfiable/total*100:.1f}%)")
    print(f"Unsatisfiable: {unsatisfiable} ({unsatisfiable/total*100:.1f}%)")

    # Assertions
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    assert total > 0, "No constraints extracted"
    assert z3_encoded > 0, "No constraints Z3 encoded"
    assert formalized == z3_encoded, "All Z3 encoded should be formalized"

    print("[PASS] All assertions passed")

    # Export JSON
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)

    output_file = "z3_constraint_hardening_results.json"
    results = {
        'test_type': 'end_to_end_z3_integration',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'correlation_id': correlation_id,
        'problem_description': problem_description,
        'constraints': constraints,
        'statistics': {
            'total_constraints': total,
            'formalized_count': formalized,
            'z3_encoded_count': z3_encoded,
            'satisfiable_count': satisfiable,
            'unsatisfiable_count': unsatisfiable,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results exported to: {output_file}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return results


def test_idempotency():
    """Test that constraint hardening is idempotent"""

    print("\n" + "=" * 70)
    print("IDEMPOTENCY TEST")
    print("=" * 70)

    config = Phase1Config.from_env()
    logger = StructuredLogger('IdempotencyTest')
    hardener = ConstraintHardener(config, logger)

    problem = "The system cannot process more than 1000 items"

    # Run twice
    constraints1 = hardener.harden_constraints(problem, "test-1")
    constraints2 = hardener.harden_constraints(problem, "test-2")

    # Compare
    assert len(constraints1) == len(constraints2), "Different number of constraints"

    for c1, c2 in zip(constraints1, constraints2):
        assert c1['description'] == c2['description'], "Descriptions differ"
        assert c1['inverted_description'] == c2['inverted_description'], "Inversions differ"

    print("[PASS] Idempotency verified - same input produces same output")

    return True


def test_fallback_to_text_based():
    """Test graceful fallback when Z3 is disabled"""

    print("\n" + "=" * 70)
    print("FALLBACK TEST (Z3 DISABLED)")
    print("=" * 70)

    config = Phase1Config.from_env()
    config.ENABLE_Z3_CONSTRAINT_HARDENING = False

    logger = StructuredLogger('FallbackTest')
    hardener = ConstraintHardener(config, logger)

    problem = "The system cannot process more than 1000 items"

    constraints = hardener.harden_constraints(problem, "test-fallback")

    assert len(constraints) > 0, "No constraints with text-based fallback"
    assert not constraints[0]['z3_encoded'], "Should not be Z3 encoded"
    assert constraints[0]['inverted_description'], "Should have text-based inversion"

    print(f"[PASS] Text-based fallback working")
    print(f"  Inverted: {constraints[0]['inverted_description']}")

    return constraints


def main():
    """Run all integration tests"""

    print("\n" + "=" * 70)
    print("Z3 INTEGRATION - END-TO-END TEST SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    try:
        # Test 1: End-to-end
        print("\n[TEST 1] End-to-End Z3 Constraint Hardening")
        results = test_e2e_z3_constraint_hardening()

        # Test 2: Idempotency
        print("\n[TEST 2] Idempotency")
        test_idempotency()

        # Test 3: Fallback
        print("\n[TEST 3] Text-Based Fallback")
        test_fallback_to_text_based()

        print("\n" + "=" * 70)
        print("ALL INTEGRATION TESTS PASSED")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
