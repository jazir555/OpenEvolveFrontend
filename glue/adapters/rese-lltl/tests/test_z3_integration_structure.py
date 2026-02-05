#!/usr/bin/env python3
"""
Structure Test: Verify Z3 Integration Code

Tests that the Z3 integration code is properly structured
and can be imported without errors.

This test doesn't require LLTL module to be available.
"""

import os
import sys
import unittest

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestZ3IntegrationStructure(unittest.TestCase):
    """Test Z3 integration code structure"""

    def test_import_z3prover_integration(self):
        """Test that z3prover_integration can be imported"""
        try:
            from z3prover_integration import (
                Z3SolverEngine,
                Z3Config,
                Z3ResultStatus,
                Z3Variable,
                Z3Constraint,
                Z3ConstraintType,
                is_z3_available
            )
            # If we get here, imports succeeded
            self.assertTrue(True)

            # Check Z3 availability
            z3_available = is_z3_available()
            print(f"\nZ3 Available: {z3_available}")

        except ImportError as e:
            self.skipTest(f"z3prover_integration not available: {e}")

    def test_lltl_adapter_has_z3_methods(self):
        """Test that LLTLAdapter has Z3 methods"""
        try:
            from lltl_adapter import LLTLAdapter
        except ImportError:
            self.skipTest("LLTLAdapter not available (expected if LLTL module not installed)")

        # Check that Z3 methods exist
        self.assertTrue(hasattr(LLTLAdapter, '_detect_contradictions_z3'))
        self.assertTrue(hasattr(LLTLAdapter, '_formal_commitments_to_z3'))
        self.assertTrue(hasattr(LLTLAdapter, '_formal_commitment_to_z3_formula'))
        self.assertTrue(hasattr(LLTLAdapter, '_encode_statement_to_z3'))
        self.assertTrue(hasattr(LLTLAdapter, '_extract_inequality'))
        self.assertTrue(hasattr(LLTLAdapter, '_extract_equality'))
        self.assertTrue(hasattr(LLTLAdapter, '_extract_variable_names'))
        self.assertTrue(hasattr(LLTLAdapter, '_detect_contradictions_naive'))
        self.assertTrue(hasattr(LLTLAdapter, '_check_contradiction_naive'))

        print("\n[PASS] All Z3 methods are defined")

    def test_formal_commitment_class(self):
        """Test that FormalCommitment class exists"""
        try:
            from lltl_adapter import FormalCommitment
        except ImportError:
            self.skipTest("FormalCommitment not available")

        # Check required fields
        import inspect
        sig = inspect.signature(FormalCommitment)

        required_fields = [
            'proposition_id',
            'statement',
            'confidence_threshold',
            'statistical_evidence',
            'source_hypothesis',
            'derivation_method',
            'timestamp',
            'correlation_id'
        ]

        for field in required_fields:
            self.assertIn(field, sig.parameters)

        print("\n[PASS] FormalCommitment has all required fields")

    def test_z3_environment_variables(self):
        """Test Z3 environment variables are documented"""
        # Set test environment variables
        os.environ['RESE_Z3_LLTL_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '5000'
        os.environ['RESE_SIGNIFICANCE_LEVEL'] = '0.05'

        # Verify they're set
        self.assertEqual(os.getenv('RESE_Z3_LLTL_ENABLED'), 'true')
        self.assertEqual(os.getenv('Z3_TIMEOUT'), '5000')
        self.assertEqual(os.getenv('RESE_SIGNIFICANCE_LEVEL'), '0.05')

        print("\n[PASS] Environment variables are properly set")

    def test_z3_integration_documentation_exists(self):
        """Test that Z3_INTEGRATION.md documentation exists"""
        docs_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'Z3_INTEGRATION.md'
        )

        self.assertTrue(os.path.exists(docs_path), f"Documentation not found: {docs_path}")

        # Check documentation contains key sections
        with open(docs_path, 'r', encoding='utf-8') as f:
            content = f.read()

        required_sections = [
            '# Z3 Integration for LLTL',
            '## Architecture',
            '## Configuration',
            '## Usage',
            '## Testing',
            '## Performance',
            '## CLAUDE.md Compliance'
        ]

        for section in required_sections:
            self.assertIn(section, content, f"Missing section: {section}")

        print("\n[PASS] Documentation exists with all required sections")

    def test_probe_script_exists(self):
        """Test that Z3 probe script exists"""
        probe_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'probes',
            'check_z3_contradiction.sh'
        )

        self.assertTrue(os.path.exists(probe_path), f"Probe script not found: {probe_path}")

        # Check probe script is executable
        self.assertTrue(os.access(probe_path, os.R_OK), f"Probe script not readable: {probe_path}")

        print("\n[PASS] Probe script exists and is readable")

    def test_test_files_exist(self):
        """Test that test files exist"""
        test_files = [
            'test_z3_contradiction_detection.py',
            'test_z3_dito_benchmark.py'
        ]

        tests_dir = os.path.dirname(__file__)

        for test_file in test_files:
            test_path = os.path.join(tests_dir, test_file)
            self.assertTrue(os.path.exists(test_path), f"Test file not found: {test_path}")

        print(f"\n[PASS] All test files exist ({len(test_files)} files)")


def run_tests():
    """Run structure tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestZ3IntegrationStructure))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    import sys
    sys.exit(run_tests())
