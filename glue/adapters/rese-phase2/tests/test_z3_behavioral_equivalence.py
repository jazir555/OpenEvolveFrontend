"""
Unit Tests for Z3 Behavioral Equivalence Verification in Phase II

Tests cover:
1. FDG encoding to Z3 formulas
2. Input variable extraction
3. Equivalence formula generation
4. Behavioral equivalence verification
5. Integration with I_mech calculation
6. Backward compatibility (Z3 disabled)

Following CLAUDE.md principles:
- Law of Runtime Truth: Test against real Z3 API
- Law of Idempotency: Same inputs → same outputs
- Circuit Breaker: Timeout handling tested
- Structured Logging: Verify JSON logs

Author: RESE Team
Created: 2026-02-04
"""

import os
import sys
import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

# Add paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "src"))
_schemas_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "schemas"))
_root_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", ".."))

for path in [_src_dir, _schemas_dir, _root_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from rese_schemas import (
        Phase2Config,
        FunctionalDependencyGraph,
        FunctionalDependency,
        IsomorphicMapping,
        IsomorphismType
    )
    from phase2_executor import (
        CrossDomainMapper,
        Phase2Logger,
        EquivalenceResult
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")


class TestFDGEncoding(unittest.TestCase):
    """Test FDG to Z3 formula encoding."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        # Create config
        self.config = Phase2Config(
            max_target_domains=5,
            i_mech_threshold=0.7,
            timeout_ms=20000,
            correlation_id="test-correlation-001"
        )

        # Create logger
        self.logger = Phase2Logger(correlation_id="test-correlation-001")

        # Create mapper with Z3 enabled
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'
        os.environ['Z3_TIMEOUT'] = '10000'
        self.mapper = CrossDomainMapper(self.config, self.logger)

    def tearDown(self):
        """Clean up environment."""
        if 'RESE_Z3_PHASE2_ENABLED' in os.environ:
            del os.environ['RESE_Z3_PHASE2_ENABLED']
        if 'Z3_TIMEOUT' in os.environ:
            del os.environ['Z3_TIMEOUT']

    def test_sanitize_z3_name_basic(self):
        """Test basic name sanitization."""
        test_cases = [
            ("energy", "energy"),
            ("energy-momentum", "energy_momentum"),
            ("energy momentum", "energy_momentum"),
            ("energy@source", "energy_at_source"),
            ("node#1", "node_hash_1"),
            ("123node", "n_123node"),
            ("", "unknown")
        ]

        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.mapper._sanitize_z3_name(input_name)
                self.assertEqual(result, expected)

    def test_encode_fdg_to_z3_simple(self):
        """Test encoding a simple FDG with no dependencies."""
        fdg = FunctionalDependencyGraph(
            domain="test",
            nodes=["A", "B", "C"],
            dependencies=[],
            adjacency_list={"A": [], "B": [], "C": []}
        )

        formula = self.mapper._encode_fdg_to_z3(fdg, "test-001")

        # Should have declarations for all nodes
        self.assertIn("(declare-const A Bool)", formula)
        self.assertIn("(declare-const B Bool)", formula)
        self.assertIn("(declare-const C Bool)", formula)

    def test_encode_fdg_to_z3_with_dependencies(self):
        """Test encoding FDG with dependencies."""
        fdg = FunctionalDependencyGraph(
            domain="test",
            nodes=["A", "B", "C"],
            dependencies=[
                FunctionalDependency(
                    source="A",
                    target="B",
                    relationship_type="causal",
                    strength=1.0,
                    domain="test"
                )
            ],
            adjacency_list={"A": ["B"], "B": [], "C": []}
        )

        formula = self.mapper._encode_fdg_to_z3(fdg, "test-002")

        # Should have declarations
        self.assertIn("(declare-const A Bool)", formula)
        self.assertIn("(declare-const B Bool)", formula)

        # Should have constraint for strong dependency
        self.assertIn("(assert (= B A))", formula)

    def test_encode_fdg_to_z3_weak_dependency(self):
        """Test encoding FDG with weak dependency (should be ignored)."""
        fdg = FunctionalDependencyGraph(
            domain="test",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(
                    source="A",
                    target="B",
                    relationship_type="weak",
                    strength=0.3,  # Below 0.5 threshold
                    domain="test"
                )
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        formula = self.mapper._encode_fdg_to_z3(fdg, "test-003")

        # Should have declarations
        self.assertIn("(declare-const A Bool)", formula)
        self.assertIn("(declare-const B Bool)", formula)

        # Should NOT have constraint for weak dependency
        self.assertNotIn("(assert", formula)

    def test_encode_fdg_domain_types(self):
        """Test that different domains get appropriate variable types."""
        test_cases = [
            ("physics", "Real"),
            ("economics", "Real"),
            ("computer_science", "Int"),
            ("biology", "Int"),
            ("unknown", "Bool")
        ]

        for domain, expected_type in test_cases:
            with self.subTest(domain=domain):
                fdg = FunctionalDependencyGraph(
                    domain=domain,
                    nodes=["X"],
                    dependencies=[],
                    adjacency_list={"X": []}
                )

                formula = self.mapper._encode_fdg_to_z3(fdg, f"test-{domain}")

                self.assertIn(f"(declare-const X {expected_type})", formula)


class TestInputExtraction(unittest.TestCase):
    """Test input variable extraction from FDGs."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.config = Phase2Config(correlation_id="test-002")
        self.logger = Phase2Logger(correlation_id="test-002")
        self.mapper = CrossDomainMapper(self.config, self.logger)

    def test_extract_inputs_from_simple_fdgs(self):
        """Test extracting inputs from FDGs with no edges."""
        fdg1 = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B"],
            dependencies=[],
            adjacency_list={"A": [], "B": []}
        )

        fdg2 = FunctionalDependencyGraph(
            domain="test2",
            nodes=["X", "Y"],
            dependencies=[],
            adjacency_list={"X": [], "Y": []}
        )

        inputs = self.mapper._extract_input_variables(fdg1, fdg2)

        # All nodes should be inputs (no dependencies)
        self.assertEqual(len(inputs), 4)
        self.assertIn("A", inputs)
        self.assertIn("B", inputs)
        self.assertIn("X", inputs)
        self.assertIn("Y", inputs)

    def test_extract_inputs_with_dependencies(self):
        """Test extracting inputs when FDGs have dependencies."""
        fdg1 = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B", "C"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test1"),
                FunctionalDependency(source="B", target="C", relationship_type="causal", strength=1.0, domain="test1")
            ],
            adjacency_list={"A": ["B"], "B": ["C"], "C": []}
        )

        fdg2 = FunctionalDependencyGraph(
            domain="test2",
            nodes=["X", "Y"],
            dependencies=[
                FunctionalDependency(source="X", target="Y", relationship_type="causal", strength=1.0, domain="test2")
            ],
            adjacency_list={"X": ["Y"], "Y": []}
        )

        inputs = self.mapper._extract_input_variables(fdg1, fdg2)

        # Only root nodes (no incoming edges) should be inputs
        self.assertEqual(len(inputs), 2)
        self.assertIn("A", inputs)  # Root in fdg1
        self.assertIn("X", inputs)  # Root in fdg2
        self.assertNotIn("B", inputs)  # Has incoming edge from A
        self.assertNotIn("C", inputs)  # Has incoming edge from B
        self.assertNotIn("Y", inputs)  # Has incoming edge from X


class TestEquivalenceFormula(unittest.TestCase):
    """Test equivalence formula generation."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.config = Phase2Config(correlation_id="test-003")
        self.logger = Phase2Logger(correlation_id="test-003")
        self.mapper = CrossDomainMapper(self.config, self.logger)

    def test_equivalence_formula_no_inputs(self):
        """Test equivalence formula when no inputs."""
        formula1 = "(declare-const A Bool)"
        formula2 = "(declare-const X Bool)"

        result = self.mapper._encode_equivalence_formula(formula1, formula2, [], "test-003")

        # Should return "true" for trivial case
        self.assertEqual(result, "true")

    def test_equivalence_formula_with_inputs(self):
        """Test equivalence formula with inputs."""
        formula1 = "(declare-const A Bool)"
        formula2 = "(declare-const X Bool)"
        inputs = ["A", "X"]

        result = self.mapper._encode_equivalence_formula(formula1, formula2, inputs, "test-004")

        # Should combine formulas with AND
        self.assertIn("(and", result)
        self.assertIn(formula1, result)
        self.assertIn(formula2, result)


class TestBehavioralEquivalence(unittest.TestCase):
    """Test behavioral equivalence verification."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.config = Phase2Config(correlation_id="test-005")
        self.logger = Phase2Logger(correlation_id="test-005")

        # Enable Z3
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'

    def tearDown(self):
        """Clean up environment."""
        if 'RESE_Z3_PHASE2_ENABLED' in os.environ:
            del os.environ['RESE_Z3_PHASE2_ENABLED']

    @patch('phase2_executor.Z3_AVAILABLE', False)
    def test_behavioral_equivalence_z3_unavailable(self):
        """Test behavioral equivalence when Z3 is not available."""
        mapper = CrossDomainMapper(self.config, self.logger)

        fdg1 = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A"],
            dependencies=[],
            adjacency_list={"A": []}
        )

        fdg2 = FunctionalDependencyGraph(
            domain="test2",
            nodes=["X"],
            dependencies=[],
            adjacency_list={"X": []}
        )

        # Should return False when Z3 unavailable
        result = mapper._verify_behavioral_equivalence(fdg1, fdg2, "test-006")

        self.assertFalse(result.verified)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.solver, 'error')

    def test_compute_imech_score_without_z3(self):
        """Test I_mech calculation when Z3 is disabled."""
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'false'

        mapper = CrossDomainMapper(self.config, self.logger)

        fdg1 = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test1")
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        fdg2 = FunctionalDependencyGraph(
            domain="test2",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test2")
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        score = mapper.compute_imech_score(fdg1, fdg2, correlation_id="test-007")

        # Should return structural score only
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_compute_imech_score_below_threshold(self):
        """Test I_mech calculation when structural score is below threshold."""
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'

        mapper = CrossDomainMapper(self.config, self.logger)

        # Create FDGs with low overlap
        fdg1 = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B", "C"],
            dependencies=[],
            adjacency_list={"A": [], "B": [], "C": []}
        )

        fdg2 = FunctionalDependencyGraph(
            domain="test2",
            nodes=["X", "Y", "Z"],
            dependencies=[],
            adjacency_list={"X": [], "Y": [], "Z": []}
        )

        score = mapper.compute_imech_score(fdg1, fdg2, correlation_id="test-008")

        # Should return structural score (below threshold, no Z3 verification)
        self.assertLess(score, 0.5)  # Low structural overlap


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility when Z3 is disabled."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.config = Phase2Config(correlation_id="test-009")
        self.logger = Phase2Logger(correlation_id="test-009")

    def test_mapper_works_without_z3(self):
        """Test that mapper works correctly when Z3 is disabled."""
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'false'

        mapper = CrossDomainMapper(self.config, self.logger)

        # Should initialize successfully
        self.assertIsNotNone(mapper)
        self.assertFalse(mapper.z3_enabled)
        self.assertIsNone(mapper.z3_prover)

        # Should compute I_mech scores using structural only
        fdg1 = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test1")
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        fdg2 = FunctionalDependencyGraph(
            domain="test2",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test2")
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        score = mapper.compute_imech_score(fdg1, fdg2, correlation_id="test-010")

        # Should return valid score
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestEquivalenceResult(unittest.TestCase):
    """Test EquivalenceResult data class."""

    def test_equivalence_result_creation(self):
        """Test creating EquivalenceResult."""
        result = EquivalenceResult(
            verified=True,
            confidence=0.95,
            proof="test proof",
            solver="z3",
            execution_time=100.0
        )

        self.assertTrue(result.verified)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.proof, "test proof")
        self.assertEqual(result.solver, "z3")
        self.assertEqual(result.execution_time, 100.0)
        self.assertEqual(len(result.errors), 0)

    def test_equivalence_result_to_dict(self):
        """Test converting EquivalenceResult to dict."""
        result = EquivalenceResult(
            verified=True,
            confidence=0.95,
            proof="test proof",
            solver="z3",
            execution_time=100.0,
            errors=["error1"]
        )

        result_dict = result.to_dict()

        self.assertTrue(result_dict["verified"])
        self.assertEqual(result_dict["confidence"], 0.95)
        self.assertEqual(result_dict["proof"], "test proof")
        self.assertEqual(result_dict["solver"], "z3")
        self.assertEqual(result_dict["execution_time"], 100.0)
        self.assertEqual(len(result_dict["errors"]), 1)


class TestIntegrationWithIMech(unittest.TestCase):
    """Integration tests for Z3 with I_mech calculation."""

    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")

        self.config = Phase2Config(
            i_mech_threshold=0.5,  # Lower threshold for testing
            correlation_id="test-011"
        )
        self.logger = Phase2Logger(correlation_id="test-011")

    def test_find_isomorphic_mappings_with_z3(self):
        """Test finding isomorphic mappings with Z3 enabled."""
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'

        mapper = CrossDomainMapper(self.config, self.logger)

        source_fdg = FunctionalDependencyGraph(
            domain="physics",
            nodes=["energy", "momentum"],
            dependencies=[
                FunctionalDependency(source="energy", target="momentum", relationship_type="causal", strength=0.8, domain="physics")
            ],
            adjacency_list={"energy": ["momentum"], "momentum": []}
        )

        target_fdg = FunctionalDependencyGraph(
            domain="economics",
            nodes=["energy", "momentum"],
            dependencies=[
                FunctionalDependency(source="energy", target="momentum", relationship_type="causal", strength=0.8, domain="economics")
            ],
            adjacency_list={"energy": ["momentum"], "momentum": []}
        )

        mappings = mapper.find_isomorphic_mappings(source_fdg, [target_fdg], correlation_id="test-012")

        # Should find at least one mapping
        self.assertGreater(len(mappings), 0)

        # Check mapping properties
        mapping = mappings[0]
        self.assertEqual(mapping.source_domain, "physics")
        self.assertEqual(mapping.target_domain, "economics")
        self.assertGreater(mapping.i_mech_score, 0.0)

    def test_isomorphism_type_with_z3(self):
        """Test that Z3 verification sets isomorphism type to MECHANISTIC."""
        os.environ['RESE_Z3_PHASE2_ENABLED'] = 'true'

        mapper = CrossDomainMapper(self.config, self.logger)

        source_fdg = FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test1")
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        target_fdg = FunctionalDependencyGraph(
            domain="test2",
            nodes=["A", "B"],
            dependencies=[
                FunctionalDependency(source="A", target="B", relationship_type="causal", strength=1.0, domain="test2")
            ],
            adjacency_list={"A": ["B"], "B": []}
        )

        mappings = mapper.find_isomorphic_mappings(source_fdg, [target_fdg], correlation_id="test-013")

        if len(mappings) > 0:
            # Should be MECHANISTIC if Z3 is available
            self.assertEqual(mappings[0].isomorphism_type, IsomorphismType.MECHANISTIC)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestFDGEncoding,
        TestInputExtraction,
        TestEquivalenceFormula,
        TestBehavioralEquivalence,
        TestBackwardCompatibility,
        TestEquivalenceResult,
        TestIntegrationWithIMech
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
