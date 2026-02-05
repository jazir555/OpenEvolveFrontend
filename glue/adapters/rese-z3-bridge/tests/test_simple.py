"""
Simple Tests for RESE-Z3 Bridge Adapter

Basic tests to verify the bridge structure and core functionality.

Author: RESE Team
Created: 2026-02-04
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    # Test schema import
    try:
        from rese_z3_schema import (
            CanonicalVariable,
            CanonicalConstraint,
            ConstraintType,
            ProblemType,
            Z3ResultStatus,
            validate_solver_request,
            canonical_to_smtlib,
        )
        print("[OK] Schema imports successful")
    except Exception as e:
        print(f"[FAIL] Schema imports failed: {e}")
        return False

    # Test client import
    try:
        from rese_z3_client import (
            Z3Client,
            Z3ClientConfig,
            CircuitBreakerConfig,
            CircuitBreakerState,
        )
        print("[OK] Client imports successful")
    except Exception as e:
        print(f"[FAIL] Client imports failed: {e}")
        return False

    # Test bridge import
    try:
        from rese_z3_bridge import (
            RESEZ3Bridge,
            RESEZ3BridgeConfig,
        )
        print("[OK] Bridge imports successful")
    except Exception as e:
        print(f"[FAIL] Bridge imports failed: {e}")
        return False

    return True


def test_schema_validation():
    """Test canonical schema validation"""
    print("\nTesting schema validation...")

    from rese_z3_schema import (
        validate_solver_request,
        CanonicalVariable,
        CanonicalConstraint,
        ConstraintType,
        ProblemType,
    )

    # Valid request
    valid_data = {
        "problem": "(declare-const x Int) (assert (> x 10)) (check-sat)",
        "problem_type": "constraint_sat",
        "timeout_ms": 30000,
    }

    is_valid, error = validate_solver_request(valid_data)
    if is_valid:
        print("[OK] Valid request accepted")
    else:
        print(f"[FAIL] Valid request rejected: {error}")
        return False

    # Invalid request (missing problem)
    invalid_data = {
        "problem_type": "constraint_sat",
        "timeout_ms": 30000,
    }

    is_valid, error = validate_solver_request(invalid_data)
    if not is_valid:
        print("[OK] Invalid request rejected correctly")
    else:
        print("[FAIL] Invalid request not rejected")
        return False

    return True


def test_smtlib_generation():
    """Test SMT-LIB generation"""
    print("\nTesting SMT-LIB generation...")

    from rese_z3_schema import (
        CanonicalSolverRequest,
        CanonicalVariable,
        CanonicalConstraint,
        ConstraintType,
        ProblemType,
        canonical_to_smtlib,
    )

    variables = [
        CanonicalVariable("x", ConstraintType.INTEGER),
        CanonicalVariable("y", ConstraintType.REAL),
    ]

    constraints = [
        CanonicalConstraint("(> x 10)", ConstraintType.INTEGER, "x > 10"),
        CanonicalConstraint("(< y 20.5)", ConstraintType.REAL, "y < 20.5"),
    ]

    request = CanonicalSolverRequest(
        problem="",
        problem_type=ProblemType.CONSTRAINT_SAT,
        variables=variables,
        constraints=constraints,
        timeout_ms=30000,
    )

    smtlib = canonical_to_smtlib(request)

    # Check SMT-LIB content
    if "(set-logic ALL)" in smtlib:
        print("[OK] SMT-LIB has logic declaration")
    else:
        print("[FAIL] SMT-LIB missing logic declaration")
        return False

    if "(declare-fun x () Int)" in smtlib:
        print("[OK] SMT-LIB has variable declarations")
    else:
        print("[FAIL] SMT-LIB missing variable declarations")
        return False

    if "(assert (> x 10))" in smtlib:
        print("[OK] SMT-LIB has constraint assertions")
    else:
        print("[FAIL] SMT-LIB missing constraint assertions")
        return False

    return True


def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("\nTesting circuit breaker...")

    from rese_z3_client import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
    import logging

    logger = logging.getLogger("test")
    config = CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        timeout_ms=1000,
    )

    cb = CircuitBreaker(config, logger)

    # Initial state should be CLOSED
    stats = cb.get_stats()
    if stats["state"] == "closed":
        print("[OK] Circuit breaker starts in CLOSED state")
    else:
        print(f"[FAIL] Circuit breaker in wrong state: {stats['state']}")
        return False

    # Record failures to open circuit
    cb.record_failure()
    cb.record_failure()

    stats = cb.get_stats()
    if stats["state"] == "open":
        print("[OK] Circuit breaker opens after failures")
    else:
        print(f"[FAIL] Circuit breaker didn't open: {stats['state']}")
        return False

    # Wait for timeout to HALF_OPEN
    import time
    time.sleep(1.1)

    if cb.can_execute():
        print("[OK] Circuit breaker transitions to HALF_OPEN after timeout")
    else:
        print("[FAIL] Circuit breaker didn't transition to HALF_OPEN")
        return False

    # Record successes to close
    cb.record_success()
    cb.record_success()

    stats = cb.get_stats()
    if stats["state"] == "closed":
        print("[OK] Circuit breaker closes after successes")
    else:
        print(f"[FAIL] Circuit breaker didn't close: {stats['state']}")
        return False

    return True


def test_bridge_structure():
    """Test bridge structure and configuration"""
    print("\nTesting bridge structure...")

    from rese_z3_bridge import RESEZ3BridgeConfig

    # Test default config
    config = RESEZ3BridgeConfig()
    if config.z3_timeout_ms > 0:
        print("[OK] Bridge has default timeout")
    else:
        print("[FAIL] Bridge missing default timeout")
        return False

    # Test from_env config
    config = RESEZ3BridgeConfig.from_env()
    if config.z3_base_url:
        print("[OK] Bridge can load config from environment")
    else:
        print("[FAIL] Bridge failed to load config from environment")
        return False

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("RESE-Z3 Bridge Simple Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_schema_validation,
        test_smtlib_generation,
        test_circuit_breaker,
        test_bridge_structure,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[FAIL] Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
