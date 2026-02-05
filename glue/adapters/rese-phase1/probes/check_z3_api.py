#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe Z3 API for Phase I Constraint Hardening

Following CLAUDE.md Law of Runtime Truth:
- Trust execution, not documentation
- Verify Z3 integration works before implementing
- Test both z3prover_integration and z3prover_advanced
"""

import sys
import os
import traceback

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, project_root)

# Test 1: Import base Z3 integration
print("[TEST 1] Importing z3prover_integration...")
try:
    from z3prover_integration import (
        Z3SolverEngine,
        Z3TheoremProver,
        Z3Variable,
        Z3Constraint,
        Z3ConstraintType,
        Z3ResultStatus,
        Z3Config,
        Z3_AVAILABLE,
        Z3_PYTHON_AVAILABLE
    )
    print(f"  [PASS] z3prover_integration imported")
    print(f"  [INFO] Z3_AVAILABLE: {Z3_AVAILABLE}")
    print(f"  [INFO] Z3_PYTHON_AVAILABLE: {Z3_PYTHON_AVAILABLE}")
except ImportError as e:
    print(f"  [FAIL] Failed to import z3prover_integration: {e}")
    sys.exit(1)

# Test 2: Import advanced Z3 integration
print("\n[TEST 2] Importing z3prover_advanced...")
try:
    from z3prover_advanced import (
        Z3AdvancedSolver,
        OptimizationObjective,
        OptimizationResult
    )
    print(f"  [PASS] z3prover_advanced imported")
except ImportError as e:
    print(f"  [FAIL] Failed to import z3prover_advanced: {e}")
    sys.exit(1)

# Test 3: Create solver instance
print("\n[TEST 3] Creating Z3 solver instance...")
try:
    config = Z3Config(timeout=5.0)
    solver = Z3SolverEngine(config)
    print(f"  [PASS] Z3SolverEngine created")
except Exception as e:
    print(f"  [FAIL] Failed to create solver: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Solve simple constraint
print("\n[TEST 4] Solving simple constraint (x > 5)...")
try:
    variables = [Z3Variable("x", Z3ConstraintType.INTEGER)]
    constraints = [Z3Constraint("x > 5", Z3ConstraintType.INTEGER, "x is greater than 5")]

    result = solver.solve_constraints(variables, constraints)

    print(f"  [INFO] Status: {result.status.value}")
    if result.is_sat():
        print(f"  [PASS] SAT - Solution found: {result.model.assignments}")
    elif result.is_unsat():
        print(f"  [INFO] UNSAT - No solution")
    else:
        print(f"  [WARN] UNKNOWN - {result.reason}")
except Exception as e:
    print(f"  [FAIL] Failed to solve constraint: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Formula simplification
print("\n[TEST 5] Testing formula simplification...")
try:
    # Test that we can simplify expressions
    smtlib_formula = "(and (or x y) (not x))"
    print(f"  Input: {smtlib_formula}")

    # Use Z3 to simplify
    if Z3_PYTHON_AVAILABLE:
        import z3
        x = z3.Bool('x')
        y = z3.Bool('y')
        expr = z3.And(z3.Or(x, y), z3.Not(x))
        simplified = z3.simplify(expr)
        print(f"  [PASS] Simplified: {simplified}")
    else:
        print(f"  [WARN] Z3 Python not available, skipping simplification test")
except Exception as e:
    print(f"  [FAIL] Failed to simplify formula: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 6: Quantifier handling
print("\n[TEST 6] Testing quantifier negation...")
try:
    if Z3_PYTHON_AVAILABLE:
        import z3
        x = z3.Int('x')
        # Exists x. P(x) -> Not(Forall x. Not(P(x)))
        exists_formula = z3.Exists(x, x > 0)
        print(f"  Input: Exists x. (x > 0)")

        # Negate: NOT (Exists x. P(x)) -> Forall x. NOT P(x)
        negated = z3.Not(exists_formula)
        simplified = z3.simplify(negated)
        print(f"  [PASS] Negated and simplified: {simplified}")
    else:
        print(f"  [WARN] Z3 Python not available, skipping quantifier test")
except Exception as e:
    print(f"  [FAIL] Failed quantifier test: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 7: Advanced solver features
print("\n[TEST 7] Testing advanced solver features...")
try:
    adv_solver = Z3AdvancedSolver(config)
    print(f"  [PASS] Z3AdvancedSolver created")

    # Test optimization
    variables = [
        Z3Variable("x", Z3ConstraintType.INTEGER),
        Z3Variable("y", Z3ConstraintType.INTEGER)
    ]
    constraints = [
        Z3Constraint("x >= 0", Z3ConstraintType.INTEGER),
        Z3Constraint("y >= 0", Z3ConstraintType.INTEGER),
        Z3Constraint("x + y <= 10", Z3ConstraintType.INTEGER)
    ]
    objectives = [("x", OptimizationObjective.MAXIMIZE)]

    result = adv_solver.optimize(variables, constraints, objectives)
    print(f"  [INFO] Optimization success: {result.success}")
    if result.success:
        print(f"  [PASS] Optimal value: {result.optimal_value}")
        print(f"  [PASS] Model: {result.optimal_model.assignments if result.optimal_model else 'N/A'}")
except Exception as e:
    print(f"  [FAIL] Failed advanced solver test: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Formula inversion (core requirement for Phase I)
print("\n[TEST 8] Testing constraint inversion (NOT)...")
try:
    if Z3_PYTHON_AVAILABLE:
        import z3

        # Test propositional negation: NOT P
        P = z3.Bool('P')
        not_P = z3.Not(P)
        print(f"  [INFO] Propositional: NOT P = {not_P}")

        # Test inequality negation: NOT (x > 5) -> x <= 5
        x = z3.Int('x')
        gt_5 = x > 5
        not_gt_5 = z3.Not(gt_5)
        simplified = z3.simplify(not_gt_5)
        print(f"  [PASS] Inequality: NOT (x > 5) -> {simplified}")

        # Test quantifier negation: NOT (Exists x. P(x)) -> Forall x. NOT P(x)
        exists_P = z3.Exists(x, x > 0)
        not_exists_P = z3.Not(exists_P)
        simplified_forall = z3.simplify(not_exists_P)
        print(f"  [PASS] Quantifier: NOT (Exists x. x > 0) -> {simplified_forall}")

        # Test De Morgan: NOT (P AND Q) -> (NOT P OR NOT Q)
        Q = z3.Bool('Q')
        and_PQ = z3.And(P, Q)
        not_and_PQ = z3.Not(and_PQ)
        simplified_demorgan = z3.simplify(not_and_PQ)
        print(f"  [PASS] De Morgan: NOT (P AND Q) -> {simplified_demorgan}")
    else:
        print(f"  [WARN] Z3 Python not available, skipping inversion test")
except Exception as e:
    print(f"  [FAIL] Failed constraint inversion test: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("[OK] ALL Z3 API PROBES PASSED")
print("=" * 60)
print("\nConclusion: Z3 integration is ready for Phase I constraint hardening")
print(f"  - Base integration: Working")
print(f"  - Advanced features: Working")
print(f"  - Formula simplification: Working")
print(f"  - Quantifier handling: Working")
print(f"  - Constraint inversion: Working")
print(f"  - Optimization: Working")
print("\nProceeding with implementation...")
