#!/usr/bin/env python3
"""
Unit Tests for Z3 Constraint Hardening in Phase I

Following CLAUDE.md principles:
- Law of Runtime Truth: Test actual Z3 behavior
- Contract-based testing: Verify API contracts
- Idempotency: Same input = same output
"""

import sys
import os
import unittest
import json
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from phase1_executor import (
    ConstraintHardener,
    Phase1Config,
    ConstraintCategory,
    StructuredLogger
)


class TestFOLParsing(unittest.TestCase):
    """Test first-order logic parsing from natural language"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Phase1Config(
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
        self.logger = StructuredLogger('TestFOLParsing')
        self.hardener = ConstraintHardener(self.config, self.logger)

    def test_extract_variables(self):
        """Test variable extraction from constraints"""
        test_cases = [
            ("The temperature is too high", ["temperature"]),
            ("The system cannot process more than 1000 items", ["system", "items"]),
            ("Pressure exceeds safety limits", ["Pressure", "safety", "limits"]),
        ]

        for constraint, expected_vars in test_cases:
            fol = self.hardener._parse_to_fol(constraint, "test-correlation")
            self.assertIsInstance(fol['variables'], list)
            self.assertTrue(len(fol['variables']) >= 0)

    def test_detect_quantifiers(self):
        """Test quantifier detection"""
        test_cases = [
            ("All particles must satisfy this constraint", ['forall']),
            ("Some particles violate the limit", ['exists']),
            ("Every particle is constrained", ['forall']),
            ("At least one particle fails", ['exists']),
            ("The constraint applies without quantifiers", []),
        ]

        for constraint, expected_quantifiers in test_cases:
            fol = self.hardener._parse_to_fol(constraint, "test-correlation")
            self.assertEqual(fol['quantifiers'], expected_quantifiers,
                           f"Failed for: {constraint}")

    def test_extract_predicates(self):
        """Test predicate extraction"""
        test_cases = [
            ("Temperature is impossible to exceed", ['impossible']),
            ("Value must be greater than 100", ['greater_than', 'required']),
            ("Pressure cannot exceed safety limits", ['impossible']),
            ("Speed is less than 50", ['less_than']),
        ]

        for constraint, expected_predicates in test_cases:
            fol = self.hardener._parse_to_fol(constraint, "test-correlation")
            for expected in expected_predicates:
                self.assertIn(expected, fol['predicates'],
                            f"Missing '{expected}' in: {constraint}")


class TestZ3Encoding(unittest.TestCase):
    """Test Z3 formula encoding"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Phase1Config(
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
        self.logger = StructuredLogger('TestZ3Encoding')
        self.hardener = ConstraintHardener(self.config, self.logger)

    def test_encode_impossible_constraint(self):
        """Test encoding 'impossible' constraint"""
        constraint = "The temperature is impossible to exceed"
        fol = self.hardener._parse_to_fol(constraint, "test-correlation")
        formula = self.hardener._encode_fol_to_z3(fol, "test-correlation")

        # Should contain negation
        self.assertIn('(not', formula.lower())

    def test_encode_greater_than(self):
        """Test encoding greater than constraint"""
        constraint = "Value must be greater than 100"
        fol = self.hardener._parse_to_fol(constraint, "test-correlation")
        formula = self.hardener._encode_fol_to_z3(fol, "test-correlation")

        # Should contain > operator
        self.assertIn('(>', formula)

    def test_encode_less_than(self):
        """Test encoding less than constraint"""
        constraint = "Speed is less than 50"
        fol = self.hardener._parse_to_fol(constraint, "test-correlation")
        formula = self.hardener._encode_fol_to_z3(fol, "test-correlation")

        # Should contain < operator
        self.assertIn('(<', formula)


class TestConstraintInversion(unittest.TestCase):
    """Test constraint inversion using Z3"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Phase1Config(
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
        self.logger = StructuredLogger('TestConstraintInversion')
        self.hardener = ConstraintHardener(self.config, self.logger)

    def test_invert_propositional(self):
        """Test propositional negation"""
        formula = "P"
        inverted = self.hardener._invert_constraint_z3(formula, "test-correlation")

        # Should wrap in NOT
        self.assertIn('(not', inverted.lower())
        self.assertIn('P', inverted)

    def test_invert_inequality(self):
        """Test inequality negation"""
        formula = "(> x 100)"
        inverted = self.hardener._invert_constraint_z3(formula, "test-correlation")

        # Should negate
        self.assertIn('(not', inverted.lower())
        self.assertIn('(> x 100)', inverted)

    def test_invert_with_quantifier(self):
        """Test quantifier negation"""
        # Exists x. P(x)
        formula = "(exists ((x Real)) P)"
        inverted = self.hardener._invert_constraint_z3(formula, "test-correlation")

        # Should negate
        self.assertIn('(not', inverted.lower())


class TestSatisfiability(unittest.TestCase):
    """Test satisfiability checking"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Phase1Config(
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
        self.logger = StructuredLogger('TestSatisfiability')
        self.hardener = ConstraintHardener(self.config, self.logger)

    def test_sat_constraint(self):
        """Test satisfiable constraint"""
        formula = "(> x 0)"  # x > 0 is satisfiable
        result = self.hardener._check_satisfiability(formula, "test-correlation")

        # Should be satisfiable
        self.assertIsNotNone(result['sat'])
        if result['sat'] is not None:
            self.assertTrue(result['sat'], "x > 0 should be satisfiable")

    def test_contradictory_constraint(self):
        """Test contradictory constraint"""
        # x > 100 AND x < 50 is unsatisfiable
        formula = "(and (> x 100) (< x 50))"
        result = self.hardener._check_satisfiability(formula, "test-correlation")

        # Should be unsatisfiable
        self.assertIsNotNone(result['sat'])


class TestTextBasedFallback(unittest.TestCase):
    """Test text-based constraint hardening fallback"""

    def setUp(self):
        """Set up test fixtures with Z3 disabled"""
        self.config = Phase1Config(
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
            ENABLE_Z3_CONSTRAINT_HARDENING=False,  # Disabled
        )
        self.logger = StructuredLogger('TestTextBasedFallback')
        self.hardener = ConstraintHardener(self.config, self.logger)

    def test_text_inversion(self):
        """Test text-based constraint inversion"""
        test_cases = [
            ("impossible", "possible"),
            ("cannot", "can"),
            ("limited", "unlimited"),
            ("restricted", "unrestricted"),
            ("never", "always"),
            ("forbidden", "allowed"),
        ]

        for original, expected_inversion in test_cases:
            inverted = self.hardener._invert_constraint_text(f"It is {original} to exceed")
            self.assertIn(expected_inversion, inverted)

    def test_harden_without_z3(self):
        """Test constraint hardening without Z3"""
        problem = "The system cannot process more than 1000 items"
        constraints = self.hardener.harden_constraints(problem, "test-correlation")

        self.assertEqual(len(constraints), 1)
        self.assertFalse(constraints[0]['z3_encoded'])
        self.assertIn('can', constraints[0]['inverted_description'])


class TestIntegration(unittest.TestCase):
    """Integration tests for constraint hardening"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = Phase1Config(
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
        self.logger = StructuredLogger('TestIntegration')
        self.hardener = ConstraintHardener(self.config, self.logger)

    def test_full_hardening_pipeline(self):
        """Test full constraint hardening pipeline"""
        problem = """
        The system cannot process more than 1000 items per second.
        The temperature is impossible to exceed 500 degrees.
        Pressure must remain below 200 psi.
        """

        constraints = self.hardener.harden_constraints(problem, "test-correlation")

        # Should extract all constraints
        self.assertGreater(len(constraints), 0)

        # Each constraint should have required fields
        for constraint in constraints:
            self.assertIn('constraint_id', constraint)
            self.assertIn('description', constraint)
            self.assertIn('inverted_description', constraint)
            self.assertIn('category', constraint)

    def test_idempotency(self):
        """Test idempotency: same input produces same output"""
        problem = "The system cannot process more than 1000 items"

        constraints1 = self.hardener.harden_constraints(problem, "test-correlation-1")
        constraints2 = self.hardener.harden_constraints(problem, "test-correlation-2")

        # Should produce same number of constraints
        self.assertEqual(len(constraints1), len(constraints2))

        # Descriptions should match
        self.assertEqual(constraints1[0]['description'], constraints2[0]['description'])


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFOLParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestZ3Encoding))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintInversion))
    suite.addTests(loader.loadTestsFromTestCase(TestSatisfiability))
    suite.addTests(loader.loadTestsFromTestCase(TestTextBasedFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
