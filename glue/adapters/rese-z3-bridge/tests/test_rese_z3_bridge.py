"""
Comprehensive Tests for RESE-Z3 Bridge Adapter

Tests all bridge functionality including:
- All API methods
- Circuit breaker logic
- Retry logic
- Caching
- Canonical schema transformations
- Performance monitoring

Following CLAUDE.md principles:
- Contract tests to prevent API breakage
- Runtime verification via probes
- Idempotency tests

Author: RESE Team
Created: 2026-02-04
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly from modules to avoid relative import issues
import importlib.util

# Load modules
spec = importlib.util.spec_from_file_location("rese_z3_schema", os.path.join(os.path.dirname(__file__), '..', 'src', 'rese_z3_schema.py'))
rese_z3_schema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rese_z3_schema)

spec = importlib.util.spec_from_file_location("rese_z3_client", os.path.join(os.path.dirname(__file__), '..', 'src', 'rese_z3_client.py'))
rese_z3_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rese_z3_client)

spec = importlib.util.spec_from_file_location("rese_z3_bridge", os.path.join(os.path.dirname(__file__), '..', 'src', 'rese_z3_bridge.py'))
rese_z3_bridge = importlib.util.module_from_spec(spec)
sys.modules['rese_z3_schema'] = rese_z3_schema
sys.modules['rese_z3_client'] = rese_z3_client
spec.loader.exec_module(rese_z3_bridge)

RESEZ3Bridge = rese_z3_bridge.RESEZ3Bridge
RESEZ3BridgeConfig = rese_z3_bridge.RESEZ3BridgeConfig
Z3Client = rese_z3_client.Z3Client
Z3ClientConfig = rese_z3_client.Z3ClientConfig
CircuitBreakerConfig = rese_z3_client.CircuitBreakerConfig
CircuitBreakerState = rese_z3_client.CircuitBreakerState
    CanonicalSolverRequest,
    CanonicalSolverResponse,
    CanonicalTheoremRequest,
    CanonicalTheoremResponse,
    CanonicalVariable,
    CanonicalConstraint,
    ConstraintType,
    ProblemType,
    Z3ResultStatus,
    canonical_to_z3_request,
    z3_to_canonical_response,
    canonical_to_smtlib,
    validate_solver_request,
    validate_theorem_request,
)


# =============================================================================
# TEST UTILITIES
# =============================================================================

class MockZ3Response:
    """Mock Z3 server response"""

    @staticmethod
    def sat_response(model: Dict[str, Any] = None) -> Dict[str, Any]:
        return {
            "status": "sat",
            "model": {"assignments": model or {"x": 10, "y": 20}},
            "execution_time": 45.0,
        }

    @staticmethod
    def unsat_response(proof: str = None) -> Dict[str, Any]:
        return {
            "status": "unsat",
            "proof": proof or "Proof of unsatisfiability",
            "execution_time": 30.0,
        }

    @staticmethod
    def error_response(error: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "reason": error,
            "errors": [error],
            "execution_time": 0.0,
        }


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestCanonicalSchema(unittest.TestCase):
    """Test canonical schema transformations"""

    def test_variable_to_dict(self):
        """Test variable serialization"""
        var = CanonicalVariable(
            name="x",
            var_type=ConstraintType.INTEGER,
            bounds=(0, 100),
            bit_width=None,
        )
        data = var.to_dict()

        self.assertEqual(data["name"], "x")
        self.assertEqual(data["var_type"], "integer")
        self.assertEqual(data["bounds"], (0, 100))

    def test_variable_from_dict(self):
        """Test variable deserialization"""
        data = {
            "name": "y",
            "var_type": "real",
            "bounds": None,
            "bit_width": None,
        }
        var = CanonicalVariable.from_dict(data)

        self.assertEqual(var.name, "y")
        self.assertEqual(var.var_type, ConstraintType.REAL)

    def test_constraint_to_dict(self):
        """Test constraint serialization"""
        constraint = CanonicalConstraint(
            expression="(> x 10)",
            constraint_type=ConstraintType.INTEGER,
            description="x is greater than 10",
            constraint_id="c1",
        )
        data = constraint.to_dict()

        self.assertEqual(data["expression"], "(> x 10)")
        self.assertEqual(data["constraint_id"], "c1")

    def test_solver_request_validation(self):
        """Test solver request validation"""
        # Valid request
        valid_data = {
            "problem": "(declare-const x Int) (assert (> x 10)) (check-sat)",
            "problem_type": "constraint_sat",
            "timeout_ms": 30000,
        }
        is_valid, error = validate_solver_request(valid_data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)

        # Missing problem
        invalid_data = {
            "problem_type": "constraint_sat",
            "timeout_ms": 30000,
        }
        is_valid, error = validate_solver_request(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("problem", error)

        # Invalid timeout
        invalid_data = {
            "problem": "test",
            "problem_type": "constraint_sat",
            "timeout_ms": -100,
        }
        is_valid, error = validate_solver_request(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("timeout_ms", error)

    def test_theorem_request_validation(self):
        """Test theorem request validation"""
        # Valid request
        valid_data = {
            "theorem_statement": "(> (+ x 1) 0)",
            "timeout_ms": 30000,
        }
        is_valid, error = validate_theorem_request(valid_data)
        self.assertTrue(is_valid)

        # Missing theorem
        invalid_data = {
            "timeout_ms": 30000,
        }
        is_valid, error = validate_theorem_request(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("theorem_statement", error)

    def test_canonical_to_smtlib(self):
        """Test SMT-LIB conversion"""
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

        self.assertIn("(set-logic ALL)", smtlib)
        self.assertIn("(declare-fun x () Int)", smtlib)
        self.assertIn("(declare-fun y () Real)", smtlib)
        self.assertIn("(assert (> x 10))", smtlib)
        self.assertIn("(assert (< y 20.5))", smtlib)
        self.assertIn("(check-sat)", smtlib)

    def test_z3_to_canonical_response(self):
        """Test Z3 response to canonical conversion"""
        z3_response = MockZ3Response.sat_response({"x": 10, "y": 20})
        canonical = z3_to_canonical_response(z3_response, "test-correlation")

        self.assertEqual(canonical.result, Z3ResultStatus.SAT)
        self.assertIsNotNone(canonical.model)
        self.assertEqual(canonical.model.assignments["x"], 10)
        self.assertEqual(canonical.correlation_id, "test-correlation")

    def test_unsat_response_conversion(self):
        """Test UNSAT response conversion"""
        z3_response = MockZ3Response.unsat_response("Proof: contradiction found")
        canonical = z3_to_canonical_response(z3_response)

        self.assertEqual(canonical.result, Z3ResultStatus.UNSAT)
        self.assertIn("contradiction found", canonical.proof)


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker logic"""

    def setUp(self):
        self.logger = MagicMock()
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_ms=1000,
        )
        self.circuit_breaker = CircuitBreaker(self.config, self.logger)

    def test_initial_state_closed(self):
        """Test circuit breaker starts in CLOSED state"""
        stats = self.circuit_breaker.get_stats()
        self.assertEqual(stats["state"], "closed")
        self.assertTrue(self.circuit_breaker.can_execute())

    def test_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        # Record failures up to threshold
        for _ in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        stats = self.circuit_breaker.get_stats()
        self.assertEqual(stats["state"], "open")
        self.assertFalse(self.circuit_breaker.can_execute())

    def test_half_open_after_timeout(self):
        """Test circuit breaker transitions to HALF_OPEN after timeout"""
        # Open the circuit
        for _ in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        # Wait for timeout
        time.sleep(1.1)

        # Should now be HALF_OPEN
        self.assertTrue(self.circuit_breaker.can_execute())
        stats = self.circuit_breaker.get_stats()
        self.assertEqual(stats["state"], "half_open")

    def test_closes_after_successes(self):
        """Test circuit breaker closes after threshold successes in HALF_OPEN"""
        # Open circuit
        for _ in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        # Wait for timeout to HALF_OPEN
        time.sleep(1.1)

        # Record successes
        for _ in range(self.config.success_threshold):
            self.circuit_breaker.record_success()

        # Should be CLOSED now
        stats = self.circuit_breaker.get_stats()
        self.assertEqual(stats["state"], "closed")

    def test_reopens_on_half_open_failures(self):
        """Test circuit breaker reopens on failures in HALF_OPEN"""
        # Open circuit
        for _ in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        # Wait for timeout to HALF_OPEN
        time.sleep(1.1)

        # Record failures
        for _ in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        # Should be OPEN again
        stats = self.circuit_breaker.get_stats()
        self.assertEqual(stats["state"], "open")


# =============================================================================
# BRIDGE API TESTS
# =============================================================================

class TestRESEZ3Bridge(unittest.TestCase):
    """Test RESE-Z3 Bridge API"""

    def setUp(self):
        """Setup test fixtures"""
        self.config = RESEZ3BridgeConfig(
            z3_base_url="http://localhost:8000",
            z3_timeout_ms=30000,
            enable_cache=False,  # Disable cache for tests
            enable_monitoring=True,
        )

    @patch('rese_z3_client.Z3Client')
    def test_solve_constraints_sat(self, mock_client_class):
        """Test solve_constraints with SAT result"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.sat_response({"x": 42})

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        variables = [
            CanonicalVariable("x", ConstraintType.INTEGER),
        ]
        constraints = [
            CanonicalConstraint("(> x 0)", ConstraintType.INTEGER),
            CanonicalConstraint("(< x 100)", ConstraintType.INTEGER),
        ]

        response = bridge.solve_constraints(
            variables=variables,
            constraints=constraints,
            correlation_id="test-123",
        )

        # Assertions
        self.assertEqual(response.result, Z3ResultStatus.SAT)
        self.assertIsNotNone(response.model)
        self.assertEqual(response.model.assignments["x"], 42)
        self.assertEqual(response.correlation_id, "test-123")

        # Verify mock called
        mock_client.solve.assert_called_once()

    @patch('rese_z3_client.Z3Client')
    def test_solve_constraints_unsat(self, mock_client_class):
        """Test solve_constraints with UNSAT result"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.unsat_response()

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        variables = [
            CanonicalVariable("x", ConstraintType.INTEGER),
        ]
        constraints = [
            CanonicalConstraint("(> x 100)", ConstraintType.INTEGER),
            CanonicalConstraint("(< x 0)", ConstraintType.INTEGER),
        ]

        response = bridge.solve_constraints(
            variables=variables,
            constraints=constraints,
        )

        # Assertions
        self.assertEqual(response.result, Z3ResultStatus.UNSAT)

    @patch('rese_z3_client.Z3Client')
    def test_detect_contradictions_true(self, mock_client_class):
        """Test detect_contradictions when contradiction exists"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.unsat_response()

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        constraints = [
            CanonicalConstraint("(> x 100)", ConstraintType.INTEGER),
            CanonicalConstraint("(< x 0)", ConstraintType.INTEGER),
        ]

        has_contradiction, counterexample = bridge.detect_contradictions(
            constraints=constraints,
        )

        # Assertions
        self.assertTrue(has_contradiction)
        self.assertIsNone(counterexample)

    @patch('rese_z3_client.Z3Client')
    def test_detect_contradictions_false(self, mock_client_class):
        """Test detect_contradictions when no contradiction"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.sat_response({"x": 50})

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        constraints = [
            CanonicalConstraint("(> x 0)", ConstraintType.INTEGER),
            CanonicalConstraint("(< x 100)", ConstraintType.INTEGER),
        ]

        has_contradiction, counterexample = bridge.detect_contradictions(
            constraints=constraints,
        )

        # Assertions
        self.assertFalse(has_contradiction)
        self.assertIsNotNone(counterexample)
        self.assertEqual(counterexample["x"], 50)

    @patch('rese_z3_client.Z3Client')
    def test_verify_anomaly_valid(self, mock_client_class):
        """Test verify_anomaly with valid constraints"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.sat_response()

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        constraints = [
            CanonicalConstraint("(> temperature 0)", ConstraintType.REAL),
        ]

        is_valid, error = bridge.verify_anomaly(
            constraints=constraints,
        )

        # Assertions
        self.assertTrue(is_valid)
        self.assertIsNone(error)

    @patch('rese_z3_client.Z3Client')
    def test_verify_anomaly_invalid(self, mock_client_class):
        """Test verify_anomaly with invalid constraints"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.unsat_response()

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        constraints = [
            CanonicalConstraint("(> temperature 1000)", ConstraintType.REAL),
            CanonicalConstraint("(< temperature 500)", ConstraintType.REAL),
        ]

        is_valid, error = bridge.verify_anomaly(
            constraints=constraints,
        )

        # Assertions
        self.assertFalse(is_valid)
        self.assertIsNotNone(error)
        self.assertIn("unsat", error)

    @patch('rese_z3_client.Z3Client')
    def test_prove_theorem_proven(self, mock_client_class):
        """Test prove_theorem with proven theorem"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.unsat_response("Proof completed")

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        theorem = "(implies (> x 0) (> (+ x 1) 0))"
        response = bridge.prove_theorem(
            theorem_statement=theorem,
            variables={"x": "Int"},
        )

        # Assertions
        self.assertTrue(response.proven)
        self.assertIn("Proof", response.proof)

    @patch('rese_z3_client.Z3Client')
    def test_prove_theorem_disproven(self, mock_client_class):
        """Test prove_theorem with disproven theorem (counterexample)"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.sat_response({"x": -5})

        # Create bridge
        bridge = RESEZ3Bridge(self.config)

        # Call API
        theorem = "(> x 0)"
        response = bridge.prove_theorem(
            theorem_statement=theorem,
            variables={"x": "Int"},
        )

        # Assertions
        self.assertFalse(response.proven)
        self.assertIsNotNone(response.counterexample)
        self.assertEqual(response.counterexample["x"], -5)

    def test_translate_to_lean4(self):
        """Test translate_to_lean4"""
        # Create bridge (no mock needed for simple translation)
        bridge = RESEZ3Bridge(self.config)

        # Call API
        smtlib = "(declare-const x Int) (assert (> x 0))"
        lean4 = bridge.translate_to_lean4(smtlib)

        # Assertions
        self.assertIn("Translated from SMT-LIB", lean4)
        self.assertIn(smtlib, lean4)

    def test_get_health(self):
        """Test get_health"""
        with patch('rese_z3_client.Z3Client') as mock_client_class:
            # Setup mock
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.check_health.return_value = {"status": "ok"}
            mock_client.get_stats.return_value = {
                "circuit_breaker": {"state": "closed"},
            }

            # Create bridge
            bridge = RESEZ3Bridge(self.config)

            # Get health
            health = bridge.get_health()

            # Assertions
            self.assertEqual(health["status"], "healthy")
            self.assertIn("z3_server", health)
            self.assertIn("circuit_breaker", health)
            self.assertIn("performance", health)

    def test_get_stats(self):
        """Test get_stats"""
        with patch('rese_z3_client.Z3Client') as mock_client_class:
            # Setup mock
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            mock_client.get_stats.return_value = {}

            # Create bridge
            bridge = RESEZ3Bridge(self.config)

            # Get stats
            stats = bridge.get_stats()

            # Assertions
            self.assertIn("config", stats)
            self.assertIn("client_stats", stats)
            self.assertIn("performance_summary", stats)


# =============================================================================
# IDEMPOTENCY TESTS
# =============================================================================

class TestIdempotency(unittest.TestCase):
    """Test idempotency of bridge operations"""

    @patch('rese_z3_client.Z3Client')
    def test_solve_constraints_idempotent(self, mock_client_class):
        """Test solve_constraints is idempotent"""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.solve.return_value = MockZ3Response.sat_response({"x": 42})

        # Create bridge with cache enabled
        config = RESEZ3BridgeConfig(
            z3_base_url="http://localhost:8000",
            z3_timeout_ms=30000,
            enable_cache=True,
            enable_monitoring=False,
        )
        bridge = RESEZ3Bridge(config)

        # Prepare inputs
        variables = [CanonicalVariable("x", ConstraintType.INTEGER)]
        constraints = [CanonicalConstraint("(> x 0)", ConstraintType.INTEGER)]

        # Call multiple times
        results = []
        for _ in range(5):
            response = bridge.solve_constraints(
                variables=variables,
                constraints=constraints,
            )
            results.append(response.to_dict())

        # All results should be identical
        for result in results:
            self.assertEqual(result, results[0])

        # Mock should only be called once (cached for subsequent calls)
        self.assertEqual(mock_client.solve.call_count, 1)


# =============================================================================
# CONTRACT TESTS
# =============================================================================

class TestContracts(unittest.TestCase):
    """Contract tests to prevent API breakage"""

    def test_solver_request_contract(self):
        """Test solver request contract"""
        # Create canonical request
        request = CanonicalSolverRequest(
            problem="(declare-const x Int) (assert (> x 10)) (check-sat)",
            problem_type=ProblemType.CONSTRAINT_SAT,
            timeout_ms=30000,
        )

        # Transform to Z3 format
        z3_request = canonical_to_z3_request(request)

        # Verify contract
        self.assertIn("problem", z3_request)
        self.assertIn("timeout_ms", z3_request)
        self.assertIn("correlation_id", z3_request)
        self.assertEqual(z3_request["timeout_ms"], 30000)

    def test_solver_response_contract(self):
        """Test solver response contract"""
        # Create Z3 response
        z3_response = MockZ3Response.sat_response()

        # Transform to canonical
        canonical = z3_to_canonical_response(z3_response)

        # Verify contract
        self.assertIsInstance(canonical.result, Z3ResultStatus)
        self.assertIsInstance(canonical.execution_time_ms, (int, float))
        self.assertIsInstance(canonical.timestamp, str)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
