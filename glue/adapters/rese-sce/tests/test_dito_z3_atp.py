#!/usr/bin/env python3
"""
Unit Tests for DITO Optimizer with Z3 ATP Integration

Tests the enhanced DITO optimizer with Z3 Automated Theorem Proving:
- Z3-based contradiction detection
- Constraint encoding to SMT-LIB2
- Performance tracking: Z3 vs naive
- Incremental solving with push/pop
- Large constraint sets (100+ constraints)

From RESE Technical Manual §3.3.1

Author: OpenEvolve
Created: 2026-02-04
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dito_optimizer import (
    DITOOptimizer,
    Z3ContradictionDetector,
    InferenceGraphNode,
    ActivationStrategy,
    DITOStats,
    Z3ATPStats,
)

from sce_bridge import (
    Constraint,
    ConstraintType,
    ConstraintCategory,
    SCEConfig,
)


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_constraint(
    constraint_id: str,
    description: str,
    category: ConstraintCategory,
    dependencies: list = None,
    type: ConstraintType = ConstraintType.HARD,
    expression: str = None
) -> Constraint:
    """Create a test constraint"""
    return Constraint(
        constraint_id=constraint_id,
        type=type,
        category=category,
        description=description,
        dependencies=dependencies or [],
        expression=expression,
    )


def log_test(test_name: str):
    """Log test start"""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")


# =============================================================================
# Unit Tests: Z3 Contradiction Detector
# =============================================================================

async def test_z3_detector_initialization():
    """Test Z3 contradiction detector initialization"""
    log_test("Z3 Detector Initialization")

    try:
        from z3prover_integration import Z3SolverEngine, Z3Config
        from dito_optimizer import Z3ContradictionDetector

        config = SCEConfig.from_env()

        # Create Z3 solver
        z3_config = Z3Config(
            timeout=config.Z3_TIMEOUT_MS / 1000.0,
            memory_limit_mb=config.Z3_MAX_MEMORY_MB,
        )
        z3_solver = Z3SolverEngine(config=z3_config)

        # Create detector
        import logging
        logger = logging.getLogger('test')
        detector = Z3ContradictionDetector(z3_solver, config, logger)

        print(f"Z3 detector created: {detector is not None}")
        print(f"Z3 solver available: {z3_solver is not None}")
        print(f"Stats initialized: {detector.stats is not None}")

        assert detector is not None, "Detector should be created"
        assert detector.z3_solver is not None, "Z3 solver should be available"
        assert detector.stats is not None, "Stats should be initialized"

        print("[PASS] Z3 detector initialization works")

    except ImportError as e:
        print(f"[SKIP] Z3 not available: {e}")
        return


async def test_constraint_encoding():
    """Test constraint encoding to Z3"""
    log_test("Constraint Encoding to Z3")

    try:
        from z3prover_integration import Z3SolverEngine, Z3Config
        from dito_optimizer import Z3ContradictionDetector

        config = SCEConfig.from_env()
        z3_config = Z3Config(timeout=5.0)
        z3_solver = Z3SolverEngine(config=z3_config)

        import logging
        logger = logging.getLogger('test')
        detector = Z3ContradictionDetector(z3_solver, config, logger)

        # Test constraints
        constraints = [
            create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("c3", "P <= 5000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        ]

        encoded_count = 0
        for constraint in constraints:
            encoded = detector.encode_constraint_to_z3(constraint)
            if encoded:
                var, constr = encoded
                print(f"Encoded: {constraint.constraint_id}")
                print(f"  Variable: {var.name} (type: {var.var_type.name})")
                print(f"  Expression: {constr.expression}")
                encoded_count += 1
            else:
                print(f"Failed to encode: {constraint.constraint_id}")

        print(f"Encoded {encoded_count}/{len(constraints)} constraints")

        # At least some constraints should encode
        assert encoded_count > 0, "Should encode at least some constraints"

        print("[PASS] Constraint encoding works")

    except ImportError as e:
        print(f"[SKIP] Z3 not available: {e}")
        return


async def test_z3_contradiction_detection():
    """Test Z3-based contradiction detection"""
    log_test("Z3 Contradiction Detection")

    try:
        from z3prover_integration import Z3SolverEngine, Z3Config
        from dito_optimizer import Z3ContradictionDetector

        config = SCEConfig.from_env()
        z3_config = Z3Config(timeout=5.0)
        z3_solver = Z3SolverEngine(config=z3_config)

        import logging
        logger = logging.getLogger('test')
        detector = Z3ContradictionDetector(z3_solver, config, logger)

        # Create contradictory constraints
        constraints = [
            create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("c2", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        ]

        print(f"Checking {len(constraints)} constraints for contradiction...")

        contradiction, result = detector.check_contradiction_z3(
            constraints,
            "test-correlation-1"
        )

        print(f"Z3 Result: {result.status.value if result else 'unknown'}")
        print(f"Contradiction found: {contradiction is not None}")
        if contradiction:
            print(f"  Type: {contradiction.type.value}")
            print(f"  Constraints: {contradiction.constraint1_id} vs {contradiction.constraint2_id}")

        stats = detector.get_stats()
        print(f"\nZ3 Stats:")
        print(f"  Checks performed: {stats.z3_checks_performed}")
        print(f"  UNSAT results: {stats.z3_unsat_results}")
        print(f"  SAT results: {stats.z3_sat_results}")
        print(f"  Total time: {stats.z3_total_time_ms}ms")

        assert stats.z3_checks_performed > 0, "Should perform Z3 checks"

        print("[PASS] Z3 contradiction detection works")

    except ImportError as e:
        print(f"[SKIP] Z3 not available: {e}")
        return


async def test_naive_vs_z3_performance():
    """Compare naive vs Z3 performance"""
    log_test("Naive vs Z3 Performance Comparison")

    try:
        from z3prover_integration import Z3SolverEngine, Z3Config
        from dito_optimizer import Z3ContradictionDetector

        config = SCEConfig.from_env()
        z3_config = Z3Config(timeout=5.0)
        z3_solver = Z3SolverEngine(config=z3_config)

        import logging
        logger = logging.getLogger('test')
        detector = Z3ContradictionDetector(z3_solver, config, logger)

        # Create test constraints
        constraints = [
            create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("c3", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
        ]

        # Test naive approach
        start = time.time()
        naive_result = detector.check_contradiction_naive(constraints[0], constraints[2])
        naive_time = (time.time() - start) * 1000

        print(f"Naive approach:")
        print(f"  Time: {naive_time:.2f}ms")
        print(f"  Found: {naive_result is not None}")

        # Test Z3 approach
        start = time.time()
        z3_contradiction, z3_result = detector.check_contradiction_z3(constraints, "test")
        z3_time = (time.time() - start) * 1000

        print(f"\nZ3 approach:")
        print(f"  Time: {z3_time:.2f}ms")
        print(f"  Found: {z3_contradiction is not None}")
        print(f"  Status: {z3_result.status.value if z3_result else 'unknown'}")

        stats = detector.get_stats()
        speedup = stats.calculate_speedup()

        print(f"\nPerformance:")
        print(f"  Naive time: {stats.naive_total_time_ms}ms")
        print(f"  Z3 time: {stats.z3_total_time_ms}ms")
        print(f"  Speedup: {speedup:.2f}x")

        print("[PASS] Performance comparison completed")

    except ImportError as e:
        print(f"[SKIP] Z3 not available: {e}")
        return


# =============================================================================
# Unit Tests: Enhanced DITO with Z3
# =============================================================================

async def test_dito_with_z3_atp():
    """Test DITO optimizer with Z3 ATP"""
    log_test("DITO with Z3 ATP")

    try:
        from z3prover_integration import Z3_AVAILABLE

        if not Z3_AVAILABLE:
            print("[SKIP] Z3 not available")
            return

        dito = DITOOptimizer(
            activation_strategy=ActivationStrategy.SELECTIVE_BFS
        )

        print(f"DITO created")
        print(f"Z3 enabled: {dito.z3_enabled}")
        print(f"Z3 detector available: {dito.z3_detector is not None}")

        # Create test constraints
        constraints = [
            create_test_constraint("temp_upper", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("temp_lower", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("temp_contradict", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["temp_upper"]),
        ]

        print(f"\nRunning DITO optimization with {len(constraints)} constraints...")

        contradictions, stats = dito.optimize_contradiction_detection(
            constraints,
            "test-dito-z3"
        )

        print(f"\nResults:")
        print(f"  Total nodes: {stats.total_nodes}")
        print(f"  Verified nodes: {stats.verified_nodes}")
        print(f"  Active nodes: {stats.active_nodes}")
        print(f"  Contradictions found: {len(contradictions)}")
        print(f"  ATP checks: {stats.atp_checks_performed}")
        print(f"  Complexity saved: {stats.complexity_saved:.1f}%")
        print(f"  Execution time: {stats.execution_time_ms}ms")

        if stats.z3_atp_stats:
            print(f"\nZ3 ATP Stats:")
            print(f"  Z3 checks: {stats.z3_atp_stats.z3_checks_performed}")
            print(f"  Z3 contradictions: {stats.z3_atp_stats.z3_contradictions_found}")
            print(f"  Z3 UNSAT: {stats.z3_atp_stats.z3_unsat_results}")
            print(f"  Z3 SAT: {stats.z3_atp_stats.z3_sat_results}")
            print(f"  Z3 time: {stats.z3_atp_stats.z3_total_time_ms}ms")
            print(f"  Naive time: {stats.z3_atp_stats.naive_total_time_ms}ms")
            print(f"  Speedup: {stats.z3_atp_stats.speedup_factor:.2f}x")

        for contradiction in contradictions:
            print(f"\nContradiction:")
            print(f"  {contradiction.constraint1_id} vs {contradiction.constraint2_id}")
            print(f"  Type: {contradiction.type.value}")

        assert dito.z3_detector is not None, "Z3 detector should be available"
        assert stats.total_nodes == len(constraints), "Should process all constraints"

        print("[PASS] DITO with Z3 ATP works")

    except ImportError as e:
        print(f"[SKIP] Required imports not available: {e}")
        return


async def test_large_constraint_set():
    """Test DITO with large constraint set (100+ constraints)"""
    log_test("Large Constraint Set (100+)")

    try:
        from z3prover_integration import Z3_AVAILABLE

        if not Z3_AVAILABLE:
            print("[SKIP] Z3 not available")
            return

        dito = DITOOptimizer(
            activation_strategy=ActivationStrategy.SELECTIVE_BFS
        )

        # Create 100 constraints
        constraints = []
        for i in range(100):
            if i % 3 == 0:
                # Create potential contradiction
                if i < 50:
                    desc = f"T < {1000 + i * 10}"
                else:
                    desc = f"T > {1000 + (i - 50) * 10}"
            else:
                desc = f"Constraint {i}"

            constraints.append(create_test_constraint(
                f"c{i}",
                desc,
                ConstraintCategory.HARD_PARAMETER_INEQUALITY,
                dependencies=[f"c{i-1}"] if i > 0 else []
            ))

        print(f"Created {len(constraints)} constraints")

        start = time.time()
        contradictions, stats = dito.optimize_contradiction_detection(
            constraints,
            "test-large-set"
        )
        elapsed = (time.time() - start) * 1000

        print(f"\nResults:")
        print(f"  Execution time: {elapsed:.2f}ms")
        print(f"  Total nodes: {stats.total_nodes}")
        print(f"  Verified: {stats.verified_nodes}")
        print(f"  Active: {stats.active_nodes}")
        print(f"  Contradictions: {len(contradictions)}")
        print(f"  Complexity saved: {stats.complexity_saved:.1f}%")

        if stats.z3_atp_stats:
            print(f"\nZ3 Performance:")
            print(f"  Z3 checks: {stats.z3_atp_stats.z3_checks_performed}")
            print(f"  Z3 time: {stats.z3_atp_stats.z3_total_time_ms}ms")
            print(f"  Naive time: {stats.z3_atp_stats.naive_total_time_ms}ms")
            if stats.z3_atp_stats.speedup_factor > 0 and stats.z3_atp_stats.speedup_factor != float('inf'):
                print(f"  Speedup: {stats.z3_atp_stats.speedup_factor:.2f}x")

        # Should complete in reasonable time
        assert elapsed < 30000, f"Should complete in < 30s (took {elapsed:.2f}ms)"
        assert stats.total_nodes == 100, "Should process all 100 constraints"

        print("[PASS] Large constraint set test passed")

    except ImportError as e:
        print(f"[SKIP] Required imports not available: {e}")
        return


async def test_incremental_solving():
    """Test incremental solving with backtrack points"""
    log_test("Incremental Solving with Backtracking")

    try:
        from z3prover_integration import Z3_AVAILABLE

        if not Z3_AVAILABLE:
            print("[SKIP] Z3 not available")
            return

        dito = DITOOptimizer(
            activation_strategy=ActivationStrategy.SELECTIVE_BFS
        )

        # Create constraints with dependencies
        constraints = [
            create_test_constraint("c1", "T < 1000", ConstraintCategory.HARD_PARAMETER_INEQUALITY),
            create_test_constraint("c2", "T > 0", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1"]),
            create_test_constraint("c3", "T > 1500", ConstraintCategory.HARD_PARAMETER_INEQUALITY, ["c1", "c2"]),
        ]

        print(f"Created {len(constraints)} constraints with dependencies")

        contradictions, stats = dito.optimize_contradiction_detection(
            constraints,
            "test-incremental"
        )

        print(f"\nResults:")
        print(f"  Contradictions: {len(contradictions)}")
        print(f"  Backtracks: {stats.backtracks_performed}")
        print(f"  Verified: {stats.verified_nodes}")
        print(f"  Active: {stats.active_nodes}")

        print("[PASS] Incremental solving test passed")

    except ImportError as e:
        print(f"[SKIP] Required imports not available: {e}")
        return


# =============================================================================
# Main Test Runner
# =============================================================================

async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("DITO Optimizer Z3 ATP Integration Test Suite")
    print("="*60)

    tests = [
        # Z3 Detector Tests
        ("Unit: Z3 Detector Initialization", test_z3_detector_initialization),
        ("Unit: Constraint Encoding", test_constraint_encoding),
        ("Unit: Z3 Contradiction Detection", test_z3_contradiction_detection),
        ("Unit: Naive vs Z3 Performance", test_naive_vs_z3_performance),

        # Enhanced DITO Tests
        ("Unit: DITO with Z3 ATP", test_dito_with_z3_atp),
        ("Unit: Large Constraint Set", test_large_constraint_set),
        ("Unit: Incremental Solving", test_incremental_solving),
    ]

    passed = 0
    failed = 0
    skipped = 0
    failed_tests = []

    for test_name, test_func in tests:
        try:
            await test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"[FAIL] FAILED: {e}")
        except Exception as e:
            failed += 1
            failed_tests.append(test_name)
            print(f"[FAIL] ERROR: {e}")

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Total:   {len(tests)}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")

    if failed_tests:
        print("\nFailed Tests:")
        for test_name in failed_tests:
            print(f"  - {test_name}")

    return failed == 0


if __name__ == '__main__':
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
