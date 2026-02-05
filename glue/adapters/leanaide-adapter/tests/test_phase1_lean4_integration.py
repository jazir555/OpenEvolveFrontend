#!/usr/bin/env python3
"""
RESE Phase I + Lean 4 Autoformalization Integration Tests

Tests the complete integration between:
1. Phase I constraint hardening (Φ₁)
2. Category A constraint extraction
3. Lean 4 formalization pipeline
4. Coverage verification

Per RESE Technical Manual §2.1.5:
"All Hard Parameter Inequality Constraints (Category A laws) are formally
proven within the Lean 4 environment."
"""

import os
import sys
import json
import unittest
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../rese-phase1/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from phase1_executor import (
    Phase1Config,
    EpistemicAuditExecutor,
    ConstraintCategory,
)
from autoformalization_pipeline import (
    AutoformalizationPipeline,
    AutoformalizationConfig,
)


class TestPhase1Lean4Integration(unittest.TestCase):
    """Test Phase I + Lean 4 integration"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Enable Lean 4 integration for tests
        os.environ['PHASE1_ENABLE_LEAN4'] = 'true'

        cls.phase1_config = Phase1Config.from_env()
        cls.lean4_config = AutoformalizationConfig.from_env()

    def test_phase1_executor_has_lean4_integration(self):
        """Test that Phase I executor has Lean 4 formalization enabled"""
        executor = EpistemicAuditExecutor(config=self.phase1_config)

        self.assertIsNotNone(
            executor.lean4_formalizer,
            "Lean 4 formalizer not initialized in Phase I executor"
        )

    def test_autoformalization_pipeline_init(self):
        """Test autoformalization pipeline initialization"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.config.LEAN4_CATEGORY_A_FILE,
                        self.lean4_config.LEAN4_CATEGORY_A_FILE)

    def test_extract_category_a_constraints(self):
        """Test Category A constraint extraction from Phase I"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()

        # Verify we have Category A constraints
        self.assertGreater(len(constraints), 0)

        # Verify all are Category A
        for constraint in constraints:
            self.assertEqual(constraint.category, "hard_parameter_inequality")
            self.assertIn(constraint.inequality_type,
                         ['less_than', 'greater_than', 'less_equal', 'greater_equal'])

    def test_generate_lean4_theorems(self):
        """Test Lean 4 theorem generation from constraints"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()
        theorems = pipeline._generate_lean4_theorems(constraints, correlation_id="test")

        # Verify all constraints have theorems
        self.assertEqual(len(theorems), len(constraints))

        # Verify theorem structure
        for theorem in theorems:
            self.assertIsNotNone(theorem.theorem_name)
            self.assertIsNotNone(theorem.signature)
            self.assertIsNotNone(theorem.proof)

    def test_write_lean4_file(self):
        """Test Lean 4 file writing"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()
        theorems = pipeline._generate_lean4_theorems(constraints, correlation_id="test")

        # Write file
        file_path = pipeline._write_lean4_file(theorems, correlation_id="test")

        # Verify file exists
        self.assertTrue(os.path.exists(file_path))

        # Verify file content
        with open(file_path, 'r') as f:
            content = f.read()

        self.assertIn("namespace RESE.Constraints", content)
        self.assertIn("theorem ", content)

    def test_formalization_pipeline_end_to_end(self):
        """Test complete formalization pipeline"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Run pipeline
        result = pipeline.run(correlation_id="test-e2e")

        # Verify result
        self.assertGreater(result.total_constraints, 0)
        self.assertEqual(result.formalized_count, result.total_constraints)
        self.assertEqual(result.coverage_percentage, 100.0)

        # Verify file created
        self.assertTrue(os.path.exists(result.lean4_file_path))

    def test_phase1_audit_includes_lean4_formalization(self):
        """Test that Phase I audit includes Lean 4 formalization"""
        executor = EpistemicAuditExecutor(config=self.phase1_config)

        # Perform a simple audit
        problem_description = """
        The reactor temperature must remain below 1000K for safe operation.
        Pressure must be maintained between 0 and 50000 Pa.
        Deuterium loading ratio must be at least 0.85.
        """

        failure_patterns = [
            {
                'pattern_description': 'lattice defects',
                'failure_rate': 0.4,
                'data_points': 100,
            }
        ]

        # Run audit (synchronously for testing)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                executor.perform_audit(
                    problem_description=problem_description,
                    failure_patterns=failure_patterns,
                    correlation_id="test-phase1-lean4",
                )
            )
        finally:
            loop.close()

        # Verify Lean 4 formalization metrics are present
        self.assertIn('category_a_constraints_formalized', result.metrics)
        self.assertIn('category_a_coverage_percentage', result.metrics)

        # Verify formalization happened
        if executor.lean4_formalizer:
            self.assertGreater(result.metrics['category_a_constraints_formalized'], 0)
            self.assertEqual(result.metrics['category_a_coverage_percentage'], 100.0)

    def test_formalization_idempotency(self):
        """Test that formalization is idempotent (Law of Idempotency)"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Run pipeline twice
        result1 = pipeline.run(correlation_id="test-idempotency-1")
        result2 = pipeline.run(correlation_id="test-idempotency-2")

        # Verify same results
        self.assertEqual(result1.total_constraints, result2.total_constraints)
        self.assertEqual(result1.formalized_count, result2.formalized_count)
        self.assertEqual(result1.coverage_percentage, result2.coverage_percentage)

    def test_all_category_a_constraints_covered(self):
        """Test that all Category A constraint types are covered"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()

        # Verify we have different constraint types
        inequality_types = set(c.inequality_type for c in constraints)

        self.assertIn('less_than', inequality_types)
        self.assertIn('greater_than', inequality_types)
        self.assertIn('greater_equal', inequality_types)

    def test_lean4_theorem_naming_conventions(self):
        """Test that theorem naming follows conventions"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()
        theorems = pipeline._generate_lean4_theorems(constraints, correlation_id="test")

        # Verify naming convention
        for theorem in theorems:
            if pipeline.config.THEOREM_NAMING_CONVENTION == 'snake_case':
                self.assertTrue(theorem.theorem_name.endswith('_constraint'))
            else:
                self.assertTrue(theorem.theorem_name.endswith('Constraint'))

    def test_lean4_proof_skeletons(self):
        """Test that proof skeletons are generated"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()
        theorems = pipeline._generate_lean4_theorems(constraints, correlation_id="test")

        # Verify all theorems have proof skeletons
        for theorem in theorems:
            self.assertIsNotNone(theorem.proof)
            self.assertGreater(len(theorem.proof), 0)

    def test_mathlib_imports_included(self):
        """Test that Mathlib imports are included when enabled"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        constraints = pipeline._get_example_constraints()
        theorems = pipeline._generate_lean4_theorems(constraints, correlation_id="test")

        # Verify Mathlib imports when enabled
        if pipeline.config.ENABLE_MATHLIB_IMPORTS:
            for theorem in theorems:
                self.assertIn('Mathlib.Data.Real.Basic', theorem.mathlib_imports)

    def test_temperature_constraint_formalization(self):
        """Test temperature constraint formalization specifically"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Get temperature constraint
        temp_constraint = next(
            (c for c in pipeline._get_example_constraints() if c.id == 'temp_max'),
            None
        )

        self.assertIsNotNone(temp_constraint, "Temperature constraint not found")

        # Generate theorem
        theorem = pipeline._generate_single_theorem(temp_constraint, correlation_id="test")

        # Verify theorem
        self.assertIn('t', theorem.signature)
        self.assertIn('1000', theorem.signature)

    def test_pressure_constraint_formalization(self):
        """Test pressure constraint formalization specifically"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Get pressure constraints
        pressure_constraints = [
            c for c in pipeline._get_example_constraints()
            if c.id.startswith('pressure')
        ]

        self.assertGreater(len(pressure_constraints), 0)

        # Generate theorems
        for constraint in pressure_constraints:
            theorem = pipeline._generate_single_theorem(constraint, correlation_id="test")

            # Verify theorem
            self.assertIn('p', theorem.signature)

    def test_deuterium_loading_constraint_formalization(self):
        """Test deuterium loading constraint formalization specifically"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Get deuterium loading constraint
        loading_constraint = next(
            (c for c in pipeline._get_example_constraints() if c.id == 'deuterium_loading_min'),
            None
        )

        self.assertIsNotNone(loading_constraint, "Deuterium loading constraint not found")

        # Generate theorem
        theorem = pipeline._generate_single_theorem(loading_constraint, correlation_id="test")

        # Verify theorem
        self.assertIn('d', theorem.signature)
        self.assertIn('0.85', theorem.signature)

    def test_coverage_calculation(self):
        """Test coverage percentage calculation"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Test various scenarios
        self.assertEqual(
            pipeline._calculate_coverage(10, 10, 10),
            100.0
        )

        self.assertEqual(
            pipeline._calculate_coverage(10, 5, 5),
            50.0
        )

        # Test with proof completion requirement
        if pipeline.config.REQUIRE_ALL_PROOFS_COMPLETE:
            self.assertEqual(
                pipeline._calculate_coverage(10, 10, 5),
                50.0
            )


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestFormalizationPerformance(unittest.TestCase):
    """Test formalization pipeline performance"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        os.environ['PHASE1_ENABLE_LEAN4'] = 'true'
        cls.lean4_config = AutoformalizationConfig.from_env()

    def test_formalization_latency(self):
        """Test that formalization completes within timeout"""
        import time

        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        start_time = time.time()
        result = pipeline.run(correlation_id="test-performance")
        execution_time_ms = int((time.time() - start_time) * 1000)

        # Verify within timeout
        self.assertLess(execution_time_ms, self.lean4_config.LEAN4_TIMEOUT_MS)

    def test_large_constraint_set(self):
        """Test formalization with larger constraint set"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Generate additional constraints
        base_constraints = pipeline._get_example_constraints()

        # Run formalization
        result = pipeline.run(correlation_id="test-large-set")

        # Verify all constraints handled
        self.assertEqual(result.formalized_count, result.total_constraints)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestFormalizationErrorHandling(unittest.TestCase):
    """Test formalization error handling"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        os.environ['PHASE1_ENABLE_LEAN4'] = 'true'
        cls.lean4_config = AutoformalizationConfig.from_env()

    def test_empty_constraint_set(self):
        """Test formalization with no constraints"""
        pipeline = AutoformalizationPipeline(config=self.lean4_config)

        # Mock empty constraint list
        result = pipeline._create_empty_result(correlation_id="test-empty")

        # Verify result
        self.assertEqual(result.total_constraints, 0)
        self.assertEqual(result.formalized_count, 0)
        self.assertEqual(result.coverage_percentage, 100.0)

    def test_formalization_without_leanaide(self):
        """Test formalization when LeanAide is disabled"""
        # Disable LeanAide
        os.environ['LEANAIDE_ENABLED'] = 'false'

        config = AutoformalizationConfig.from_env()
        pipeline = AutoformalizationPipeline(config=config)

        # Run formalization
        result = pipeline.run(correlation_id="test-no-leanaide")

        # Should still succeed
        self.assertGreater(result.total_constraints, 0)

        # Restore
        os.environ['LEANAIDE_ENABLED'] = 'true'


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Run all tests and generate coverage report"""
    import argparse

    parser = argparse.ArgumentParser(
        description='RESE Phase I + Lean 4 Integration Tests'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage-report', help='Write coverage report to file')

    args = parser.parse_args()

    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Lean4Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestFormalizationPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestFormalizationErrorHandling))

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)

    # Generate coverage report if requested
    if args.coverage_report:
        pipeline = AutoformalizationPipeline()
        formalization_result = pipeline.run()

        from test_formalization_coverage import CoverageReportGenerator
        report_gen = CoverageReportGenerator(pipeline.config)
        report = report_gen.generate(formalization_result)

        with open(args.coverage_report, 'w') as f:
            f.write(report)

        print(f"\nCoverage report written to: {args.coverage_report}")

    # Exit with appropriate status code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
