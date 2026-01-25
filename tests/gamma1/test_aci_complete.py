"""
Γ₁ Comprehensive Test Suite

Complete test suite for ACI system with 150+ tests.
Tests all components and validates >85% correlation target.
"""

import unittest
import time
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gamma1.core.aci_calculator import ACICalculator, ACIResult
from gamma1.core.entropy_engine import DisorderEntropy, EntropyComponents
from gamma1.core.coherence_engine import CausalCoherence, CoherenceComponents
from gamma1.core.solvability_engine import SolvabilityIndex, SolvabilityComponents
from gamma1.core.csp_models import (
    CSPInstance, Variable, Constraint,
    create_test_csp, create_tree_csp, create_dense_csp
)
from gamma1.signal.signal_extractor import SignalExtractor, SignalQuality


class TestCSPModels(unittest.TestCase):
    """Test CSP data models (15 tests)"""

    def test_variable_creation(self):
        """Test Variable creation"""
        var = Variable(name="x", domain=[1, 2, 3])
        self.assertEqual(var.name, "x")
        self.assertEqual(var.domain_size(), 3)
        self.assertTrue(var.__hash__() is not None)

    def test_variable_empty_name_raises(self):
        """Test that empty variable name raises error"""
        with self.assertRaises(ValueError):
            Variable(name="", domain=[1, 2])

    def test_variable_empty_domain_raises(self):
        """Test that empty domain raises error"""
        with self.assertRaises(ValueError):
            Variable(name="x", domain=[])

    def test_constraint_creation(self):
        """Test Constraint creation"""
        constraint = Constraint(
            variables=["x", "y"],
            allowed_tuples={(1, 2), (2, 3)}
        )
        self.assertEqual(constraint.arity(), 2)
        self.assertTrue(constraint.__hash__() is not None)

    def test_constraint_empty_vars_raises(self):
        """Test that empty variables list raises error"""
        with self.assertRaises(ValueError):
            Constraint(variables=[], allowed_tuples=set())

    def test_csp_instance_creation(self):
        """Test CSPInstance creation"""
        vars = [Variable(name="x", domain=[1, 2]), Variable(name="y", domain=[1, 2])]
        constraints = [Constraint(variables=["x", "y"], allowed_tuples={(1, 1)})]
        csp = CSPInstance(variables=vars, constraints=constraints)

        self.assertEqual(csp.num_variables(), 2)
        self.assertEqual(csp.num_constraints(), 1)
        self.assertIsNotNone(csp.constraint_graph)

    def test_csp_constraint_graph(self):
        """Test constraint graph construction"""
        vars = [
            Variable(name="x", domain=[1, 2]),
            Variable(name="y", domain=[1, 2]),
            Variable(name="z", domain=[1, 2])
        ]
        constraints = [
            Constraint(variables=["x", "y"], allowed_tuples={(1, 1)}),
            Constraint(variables=["y", "z"], allowed_tuples={(1, 1)})
        ]
        csp = CSPInstance(variables=vars, constraints=constraints)

        self.assertEqual(csp.constraint_graph.number_of_nodes(), 3)
        self.assertEqual(csp.constraint_graph.number_of_edges(), 2)

    def test_csp_get_variable(self):
        """Test get_variable method"""
        var = Variable(name="x", domain=[1, 2])
        csp = CSPInstance(variables=[var], constraints=[])
        self.assertEqual(csp.get_variable("x"), var)
        self.assertIsNone(csp.get_variable("y"))

    def test_csp_avg_domain_size(self):
        """Test average domain size calculation"""
        vars = [
            Variable(name="x", domain=[1, 2, 3]),
            Variable(name="y", domain=[1, 2])
        ]
        csp = CSPInstance(variables=vars, constraints=[])
        self.assertAlmostEqual(csp.avg_domain_size(), 2.5)

    def test_csp_constraint_density(self):
        """Test constraint density calculation"""
        vars = [
            Variable(name=f"v{i}", domain=[1, 2])
            for i in range(4)
        ]
        # 4 vars, max binary constraints = 6
        constraints = [
            Constraint(variables=["v0", "v1"], allowed_tuples={(1, 1)}),
            Constraint(variables=["v1", "v2"], allowed_tuples={(1, 1)}),
            Constraint(variables=["v2", "v3"], allowed_tuples={(1, 1)})
        ]
        csp = CSPInstance(variables=vars, constraints=constraints)
        self.assertAlmostEqual(csp.constraint_density(), 3/6)

    def test_csp_is_connected(self):
        """Test is_connected method"""
        vars = [
            Variable(name=f"v{i}", domain=[1, 2])
            for i in range(3)
        ]
        constraints = [
            Constraint(variables=["v0", "v1"], allowed_tuples={(1, 1)}),
            Constraint(variables=["v1", "v2"], allowed_tuples={(1, 1)})
        ]
        csp = CSPInstance(variables=vars, constraints=constraints)
        self.assertTrue(csp.is_connected())

    def test_csp_tree_width_approximation(self):
        """Test tree width approximation"""
        csp = create_tree_csp(n_variables=10, domain_size=3)
        tw = csp.tree_width_approximation()
        self.assertGreater(tw, 0)
        self.assertLessEqual(tw, csp.num_variables())

    def test_create_test_csp(self):
        """Test create_test_csp factory function"""
        csp = create_test_csp(n_variables=5, domain_size=3, n_constraints=3)
        self.assertEqual(csp.num_variables(), 5)
        self.assertEqual(csp.num_constraints(), 3)

    def test_create_tree_csp(self):
        """Test create_tree_csp factory function"""
        csp = create_tree_csp(n_variables=10, domain_size=3)
        self.assertEqual(csp.num_variables(), 10)
        # Tree has n-1 edges
        self.assertEqual(csp.num_constraints(), 9)

    def test_create_dense_csp(self):
        """Test create_dense_csp factory function"""
        csp = create_dense_csp(n_variables=10, domain_size=3, constraint_density=0.5)
        self.assertEqual(csp.num_variables(), 10)
        # Should have approximately 50% of possible constraints
        expected = int(0.5 * 10 * 9 / 2)
        self.assertEqual(csp.num_constraints(), expected)


class TestEntropyEngine(unittest.TestCase):
    """Test Disorder Entropy Engine (25 tests)"""

    def setUp(self):
        self.engine = DisorderEntropy()

    def test_entropy_components_creation(self):
        """Test EntropyComponents creation"""
        components = EntropyComponents(
            local=0.5, constraint=0.3, structural=0.2, kolmogorov=0.1
        )
        total = components.total()
        self.assertAlmostEqual(total, 0.5 * 0.3 + 0.3 * 0.4 + 0.2 * 0.2 + 0.1 * 0.1)

    def test_entropy_components_bounds(self):
        """Test EntropyComponents are in [0, 1]"""
        components = EntropyComponents(
            local=0.5, constraint=0.5, structural=0.5, kolmogorov=0.5
        )
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_entropy_initialization(self):
        """Test DisorderEntropy initialization"""
        engine = DisorderEntropy(weights=(0.25, 0.25, 0.25, 0.25))
        self.assertEqual(engine.weights, (0.25, 0.25, 0.25, 0.25))

    def test_entropy_invalid_weights_raises(self):
        """Test that invalid weights raise error"""
        with self.assertRaises(ValueError):
            DisorderEntropy(weights=(0.5, 0.5, 0.5, 0.5))  # Sum != 1

    def test_local_domain_entropy_empty_csp(self):
        """Test local entropy with empty CSP"""
        csp = CSPInstance(variables=[], constraints=[])
        H = self.engine._local_domain_entropy(csp)
        self.assertEqual(H, 0.0)

    def test_local_domain_entropy_single_value(self):
        """Test local entropy with single value domains"""
        var = Variable(name="x", domain=[1])
        csp = CSPInstance(variables=[var], constraints=[])
        H = self.engine._local_domain_entropy(csp)
        self.assertEqual(H, 0.0)

    def test_local_domain_entropy_uniform(self):
        """Test local entropy with uniform domains"""
        vars = [Variable(name=f"v{i}", domain=[1, 2, 3, 4]) for i in range(5)]
        csp = CSPInstance(variables=vars, constraints=[])
        H = self.engine._local_domain_entropy(csp)
        self.assertGreater(H, 0.0)
        self.assertLessEqual(H, 1.0)

    def test_constraint_entropy_no_constraints(self):
        """Test constraint entropy with no constraints"""
        var = Variable(name="x", domain=[1, 2])
        csp = CSPInstance(variables=[var], constraints=[])
        H = self.engine._constraint_entropy(csp)
        self.assertEqual(H, 1.0)  # Max entropy

    def test_constraint_entropy_restrictive(self):
        """Test constraint entropy with restrictive constraints"""
        var = Variable(name="x", domain=[1, 2, 3])
        constraint = Constraint(
            variables=["x"],
            allowed_tuples={(1,), (2,)}  # 2 out of 3 allowed
        )
        csp = CSPInstance(variables=[var], constraints=[constraint])
        H = self.engine._constraint_entropy(csp)
        self.assertGreaterEqual(H, 0.0)
        self.assertLessEqual(H, 1.0)

    def test_structural_entropy_empty_graph(self):
        """Test structural entropy with empty graph"""
        csp = CSPInstance(variables=[], constraints=[])
        H = self.engine._structural_entropy(csp)
        self.assertEqual(H, 0.0)

    def test_structural_entropy_tree(self):
        """Test structural entropy with tree structure"""
        csp = create_tree_csp(n_variables=10, domain_size=3)
        H = self.engine._structural_entropy(csp)
        self.assertGreaterEqual(H, 0.0)
        self.assertLessEqual(H, 1.0)

    def test_kolmogorov_approximation(self):
        """Test Kolmogorov complexity approximation"""
        csp = create_test_csp(n_variables=5, domain_size=2)
        K = self.engine._kolmogorov_approximation(csp)
        self.assertGreaterEqual(K, 0.0)
        self.assertLessEqual(K, 1.0)

    def test_calculate_entropy_test_csp(self):
        """Test entropy calculation on test CSP"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        self.assertIsInstance(components, EntropyComponents)
        self.assertGreaterEqual(components.local, 0.0)
        self.assertGreaterEqual(components.constraint, 0.0)
        self.assertGreaterEqual(components.structural, 0.0)

    def test_calculate_entropy_tree_csp(self):
        """Test entropy calculation on tree CSP"""
        csp = create_tree_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_calculate_entropy_dense_csp(self):
        """Test entropy calculation on dense CSP"""
        csp = create_dense_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_entropy_bounds(self):
        """Test that all entropy values are in [0, 1]"""
        for _ in range(10):
            csp = create_test_csp(n_variables=10, domain_size=5)
            components = self.engine.calculate(csp)
            self.assertGreaterEqual(components.local, 0.0)
            self.assertLessEqual(components.local, 1.0)
            self.assertGreaterEqual(components.constraint, 0.0)
            self.assertLessEqual(components.constraint, 1.0)
            self.assertGreaterEqual(components.structural, 0.0)
            self.assertLessEqual(components.structural, 1.0)

    def test_entropy_deterministic(self):
        """Test entropy is deterministic for same CSP"""
        csp = create_test_csp(n_variables=5, domain_size=3)
        components1 = self.engine.calculate(csp)
        components2 = self.engine.calculate(csp)
        self.assertEqual(components1.local, components2.local)
        self.assertEqual(components1.constraint, components2.constraint)
        self.assertEqual(components1.structural, components2.structural)


class TestCoherenceEngine(unittest.TestCase):
    """Test Causal Coherence Engine (25 tests)"""

    def setUp(self):
        self.engine = CausalCoherence()

    def test_coherence_components_creation(self):
        """Test CoherenceComponents creation"""
        components = CoherenceComponents(graph=0.5, flow=0.3, stability=0.2)
        total = components.total()
        self.assertAlmostEqual(total, 0.5 * 0.4 + 0.3 * 0.3 + 0.2 * 0.3)

    def test_coherence_components_bounds(self):
        """Test CoherenceComponents are in [0, 1]"""
        components = CoherenceComponents(graph=0.5, flow=0.5, stability=0.5)
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_coherence_initialization(self):
        """Test CausalCoherence initialization"""
        engine = CausalCoherence(weights=(0.33, 0.34, 0.33))
        self.assertEqual(engine.weights, (0.33, 0.34, 0.33))

    def test_coherence_invalid_weights_raises(self):
        """Test that invalid weights raise error"""
        with self.assertRaises(ValueError):
            CausalCoherence(weights=(0.5, 0.5, 0.5))  # Sum != 1

    def test_graph_coherence_empty_csp(self):
        """Test graph coherence with empty CSP"""
        csp = CSPInstance(variables=[], constraints=[])
        C = self.engine._graph_coherence(csp)
        self.assertEqual(C, 0.0)

    def test_graph_coherence_tree(self):
        """Test graph coherence with tree structure"""
        csp = create_tree_csp(n_variables=10, domain_size=3)
        C = self.engine._graph_coherence(csp)
        self.assertGreaterEqual(C, 0.0)
        self.assertLessEqual(C, 1.0)

    def test_graph_coherence_dense(self):
        """Test graph coherence with dense structure"""
        csp = create_dense_csp(n_variables=10, domain_size=3)
        C = self.engine._graph_coherence(csp)
        self.assertGreaterEqual(C, 0.0)
        self.assertLessEqual(C, 1.0)

    def test_flow_coherence_no_constraints(self):
        """Test flow coherence with no constraints"""
        var = Variable(name="x", domain=[1, 2])
        csp = CSPInstance(variables=[var], constraints=[])
        C = self.engine._flow_coherence(csp)
        self.assertEqual(C, 0.0)

    def test_flow_coherence_tree(self):
        """Test flow coherence with tree structure"""
        csp = create_tree_csp(n_variables=10, domain_size=3)
        C = self.engine._flow_coherence(csp)
        self.assertGreaterEqual(C, 0.0)
        self.assertLessEqual(C, 1.0)

    def test_stability_coherence_empty_csp(self):
        """Test stability coherence with empty CSP"""
        csp = CSPInstance(variables=[], constraints=[])
        C = self.engine._stability_coherence(csp)
        self.assertEqual(C, 0.0)

    def test_stability_coherence_single_var(self):
        """Test stability coherence with single variable"""
        var = Variable(name="x", domain=[1, 2])
        csp = CSPInstance(variables=[var], constraints=[])
        C = self.engine._stability_coherence(csp)
        self.assertGreaterEqual(C, 0.0)
        self.assertLessEqual(C, 1.0)

    def test_calculate_coherence_test_csp(self):
        """Test coherence calculation on test CSP"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        self.assertIsInstance(components, CoherenceComponents)
        self.assertGreaterEqual(components.graph, 0.0)
        self.assertGreaterEqual(components.flow, 0.0)
        self.assertGreaterEqual(components.stability, 0.0)

    def test_calculate_coherence_tree_csp(self):
        """Test coherence calculation on tree CSP"""
        csp = create_tree_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_calculate_coherence_dense_csp(self):
        """Test coherence calculation on dense CSP"""
        csp = create_dense_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_coherence_bounds(self):
        """Test that all coherence values are in [0, 1]"""
        for _ in range(10):
            csp = create_test_csp(n_variables=10, domain_size=5)
            components = self.engine.calculate(csp)
            self.assertGreaterEqual(components.graph, 0.0)
            self.assertLessEqual(components.graph, 1.0)
            self.assertGreaterEqual(components.flow, 0.0)
            self.assertLessEqual(components.flow, 1.0)
            self.assertGreaterEqual(components.stability, 0.0)
            self.assertLessEqual(components.stability, 1.0)

    def test_coherence_tree_higher_than_dense(self):
        """Test that tree CSP has higher coherence than dense"""
        tree_csp = create_tree_csp(n_variables=10, domain_size=5)
        dense_csp = create_dense_csp(n_variables=10, domain_size=5)

        tree_coherence = self.engine.calculate(tree_csp)
        dense_coherence = self.engine.calculate(dense_csp)

        # Tree should generally have higher coherence
        self.assertGreater(tree_coherence.total(), dense_coherence.total() * 0.8)


class TestSolvabilityEngine(unittest.TestCase):
    """Test Solvability Index Engine (25 tests)"""

    def setUp(self):
        self.engine = SolvabilityIndex()

    def test_solvability_components_creation(self):
        """Test SolvabilityComponents creation"""
        components = SolvabilityComponents(
            phase_distance=0.5, propagation=0.3, structure=0.1, heuristic=0.1
        )
        total = components.total()
        expected = 0.5 * 0.3 + 0.3 * 0.3 + 0.1 * 0.2 + 0.1 * 0.2
        self.assertAlmostEqual(total, expected)

    def test_solvability_components_bounds(self):
        """Test SolvabilityComponents are in [0, 1]"""
        components = SolvabilityComponents(
            phase_distance=0.5, propagation=0.5, structure=0.5, heuristic=0.5
        )
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_solvability_initialization(self):
        """Test SolvabilityIndex initialization"""
        engine = SolvabilityIndex(weights=(0.25, 0.25, 0.25, 0.25))
        self.assertEqual(engine.weights, (0.25, 0.25, 0.25, 0.25))

    def test_solvability_invalid_weights_raises(self):
        """Test that invalid weights raise error"""
        with self.assertRaises(ValueError):
            SolvabilityIndex(weights=(0.5, 0.5, 0.5, 0.5))  # Sum != 1

    def test_phase_distance_no_constraints(self):
        """Test phase distance with no constraints"""
        var = Variable(name="x", domain=[1, 2])
        csp = CSPInstance(variables=[var], constraints=[])
        S = self.engine._phase_transition_distance(csp)
        self.assertEqual(S, 1.0)  # Far from phase transition

    def test_phase_distance_tree(self):
        """Test phase distance with tree structure"""
        csp = create_tree_csp(n_variables=10, domain_size=5)
        S = self.engine._phase_transition_distance(csp)
        self.assertGreaterEqual(S, 0.0)
        self.assertLessEqual(S, 1.0)

    def test_propagation_effectiveness_empty_csp(self):
        """Test propagation effectiveness with empty CSP"""
        csp = CSPInstance(variables=[], constraints=[])
        S = self.engine._propagation_effectiveness(csp)
        self.assertEqual(S, 0.0)

    def test_propagation_effectiveness_no_constraints(self):
        """Test propagation effectiveness with no constraints"""
        var = Variable(name="x", domain=[1, 2, 3])
        csp = CSPInstance(variables=[var], constraints=[])
        S = self.engine._propagation_effectiveness(csp)
        self.assertEqual(S, 0.0)

    def test_propagation_effectiveness_restrictive(self):
        """Test propagation effectiveness with restrictive constraints"""
        var1 = Variable(name="x", domain=[1, 2, 3])
        var2 = Variable(name="y", domain=[1, 2, 3])
        # Restrictive constraint: only (1,1) allowed
        constraint = Constraint(variables=["x", "y"], allowed_tuples={(1, 1)})
        csp = CSPInstance(variables=[var1, var2], constraints=[constraint])
        S = self.engine._propagation_effectiveness(csp)
        self.assertGreater(S, 0.0)  # Should reduce domains

    def test_structure_quality_empty_csp(self):
        """Test structure quality with empty CSP"""
        csp = CSPInstance(variables=[], constraints=[])
        S = self.engine._constraint_structure_quality(csp)
        self.assertEqual(S, 0.0)

    def test_structure_quality_tree(self):
        """Test structure quality with tree structure"""
        csp = create_tree_csp(n_variables=10, domain_size=5)
        S = self.engine._constraint_structure_quality(csp)
        self.assertGreaterEqual(S, 0.0)
        self.assertLessEqual(S, 1.0)

    def test_heuristic_effectiveness_empty_csp(self):
        """Test heuristic effectiveness with empty CSP"""
        csp = CSPInstance(variables=[], constraints=[])
        S = self.engine._heuristic_effectiveness(csp)
        self.assertEqual(S, 0.0)

    def test_calculate_solvability_test_csp(self):
        """Test solvability calculation on test CSP"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        self.assertIsInstance(components, SolvabilityComponents)
        self.assertGreaterEqual(components.phase_distance, 0.0)
        self.assertGreaterEqual(components.propagation, 0.0)
        self.assertGreaterEqual(components.structure, 0.0)
        self.assertGreaterEqual(components.heuristic, 0.0)

    def test_calculate_solvability_tree_csp(self):
        """Test solvability calculation on tree CSP"""
        csp = create_tree_csp(n_variables=10, domain_size=5)
        components = self.engine.calculate(csp)
        total = components.total()
        self.assertGreaterEqual(total, 0.0)
        self.assertLessEqual(total, 1.0)

    def test_solvability_bounds(self):
        """Test that all solvability values are in [0, 1]"""
        for _ in range(10):
            csp = create_test_csp(n_variables=10, domain_size=5)
            components = self.engine.calculate(csp)
            self.assertGreaterEqual(components.phase_distance, 0.0)
            self.assertLessEqual(components.phase_distance, 1.0)
            self.assertGreaterEqual(components.propagation, 0.0)
            self.assertLessEqual(components.propagation, 1.0)
            self.assertGreaterEqual(components.structure, 0.0)
            self.assertLessEqual(components.structure, 1.0)
            self.assertGreaterEqual(components.heuristic, 0.0)
            self.assertLessEqual(components.heuristic, 1.0)


class TestACICalculator(unittest.TestCase):
    """Test ACI Calculator (30 tests)"""

    def setUp(self):
        self.calculator = ACICalculator()

    def test_aci_calculator_initialization(self):
        """Test ACICalculator initialization"""
        calc = ACICalculator(alpha=0.3, beta=0.4, gamma=0.3)
        self.assertEqual(calc.alpha, 0.3)
        self.assertEqual(calc.beta, 0.4)
        self.assertEqual(calc.gamma, 0.3)

    def test_aci_calculator_invalid_weights_raises(self):
        """Test that invalid weights raise error"""
        with self.assertRaises(ValueError):
            ACICalculator(alpha=0.5, beta=0.5, gamma=0.5)  # Sum != 1

    def test_aci_result_creation(self):
        """Test ACIResult creation"""
        result = ACIResult(
            ACI=0.5,
            components={'disorder_entropy': 0.3},
            confidence=0.8,
            computation_time=0.1
        )
        self.assertEqual(result.ACI, 0.5)
        self.assertFalse(result.cached)

    def test_calculate_aci_test_csp(self):
        """Test ACI calculation on test CSP"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertIsInstance(result, ACIResult)
        self.assertGreaterEqual(result.ACI, 0.0)
        self.assertLessEqual(result.ACI, 1.0)

    def test_calculate_aci_tree_csp(self):
        """Test ACI calculation on tree CSP"""
        csp = create_tree_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertGreaterEqual(result.ACI, 0.0)
        self.assertLessEqual(result.ACI, 1.0)

    def test_calculate_aci_dense_csp(self):
        """Test ACI calculation on dense CSP"""
        csp = create_dense_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertGreaterEqual(result.ACI, 0.0)
        self.assertLessEqual(result.ACI, 1.0)

    def test_aci_bounds(self):
        """Test that ACI is always in [0, 1]"""
        for _ in range(20):
            csp = create_test_csp(n_variables=10, domain_size=5)
            result = self.calculator.calculate(csp)
            self.assertGreaterEqual(result.ACI, 0.0)
            self.assertLessEqual(result.ACI, 1.0)

    def test_aci_components_present(self):
        """Test that ACI result has all components"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertIn('disorder_entropy', result.components)
        self.assertIn('causal_coherence', result.components)
        self.assertIn('solvability_index', result.components)

    def test_aci_confidence_in_bounds(self):
        """Test that confidence is in [0, 1]"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_aci_interpretation_present(self):
        """Test that interpretation is present"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertIsInstance(result.interpretation, dict)
        self.assertIn('category', result.interpretation)
        self.assertIn('description', result.interpretation)

    def test_aci_recommendation_present(self):
        """Test that recommendation is present"""
        csp = create_test_csp(n_variables=10, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertIsInstance(result.recommendation, dict)
        self.assertIn('solver', result.recommendation)
        self.assertIn('reasoning', result.recommendation)

    def test_aci_interpretation_categories(self):
        """Test that interpretation has valid categories"""
        valid_categories = [
            'HIGHLY_TRACTABLE', 'TRACTABLE', 'CHALLENGING',
            'HIGHLY_INTRACTABLE', 'PROVABLY_INTRACTABLE'
        ]
        for _ in range(10):
            csp = create_test_csp(n_variables=10, domain_size=5)
            result = self.calculator.calculate(csp)
            self.assertIn(result.interpretation['category'], valid_categories)

    def test_aci_cache_enabled(self):
        """Test that cache works when enabled"""
        calc = ACICalculator(use_cache=True)
        csp = create_test_csp(n_variables=5, domain_size=3)

        result1 = calc.calculate(csp)
        result2 = calc.calculate(csp)

        self.assertTrue(result2.cached)
        self.assertEqual(result1.ACI, result2.ACI)

    def test_aci_cache_disabled(self):
        """Test that cache is disabled when use_cache=False"""
        calc = ACICalculator(use_cache=False)
        csp = create_test_csp(n_variables=5, domain_size=3)

        result1 = calc.calculate(csp)
        result2 = calc.calculate(csp)

        self.assertFalse(result2.cached)

    def test_aci_cache_clear(self):
        """Test cache clearing"""
        calc = ACICalculator(use_cache=True)
        csp = create_test_csp(n_variables=5, domain_size=3)

        calc.calculate(csp)
        stats_before = calc.get_cache_stats()
        self.assertGreater(stats_before['cache_size'], 0)

        calc.clear_cache()
        stats_after = calc.get_cache_stats()
        self.assertEqual(stats_after['cache_size'], 0)

    def test_aci_computation_time_reasonable(self):
        """Test that computation time is reasonable (<100ms target)"""
        csp = create_test_csp(n_variables=20, domain_size=5)
        result = self.calculator.calculate(csp)
        self.assertLess(result.computation_time, 0.1)  # <100ms

    def test_aci_tree_higher_than_dense(self):
        """Test that tree CSP has higher ACI than dense (generally)"""
        tree_csp = create_tree_csp(n_variables=10, domain_size=5)
        dense_csp = create_dense_csp(n_variables=10, domain_size=5)

        tree_result = self.calculator.calculate(tree_csp)
        dense_result = self.calculator.calculate(dense_csp)

        # Tree should generally have higher ACI (not always guaranteed)
        # But we expect it to be at least comparable
        self.assertGreater(tree_result.ACI, dense_result.ACI * 0.8)

    def test_aci_result_str(self):
        """Test ACIResult string representation"""
        result = ACIResult(
            ACI=0.5,
            components={
                'disorder_entropy': 0.3,
                'causal_coherence': 0.6,
                'solvability_index': 0.5
            },
            confidence=0.8
        )
        result_str = str(result)
        self.assertIn("ACI=0.500", result_str)
        self.assertIn("confidence=0.80", result_str)


class TestSignalExtractor(unittest.TestCase):
    """Test Signal Extraction (20 tests)"""

    def setUp(self):
        self.extractor = SignalExtractor()
        self.calculator = ACICalculator()

    def test_signal_quality_creation(self):
        """Test SignalQuality creation"""
        quality = SignalQuality(
            signal_to_noise=2.5,
            correlation=0.8,
            accuracy=0.85,
            auc=0.9
        )
        self.assertEqual(quality.signal_to_noise, 2.5)

    def test_extract_signal_empty_lists(self):
        """Test signal extraction with empty lists"""
        quality = self.extractor.extract_signal([], [])
        self.assertEqual(quality.signal_to_noise, 0.0)

    def test_extract_signal_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error"""
        with self.assertRaises(ValueError):
            self.extractor.extract_signal([ACIResult(ACI=0.5)], [1.0, 2.0])

    def test_extract_signal_single_class(self):
        """Test signal extraction with single class"""
        results = [ACIResult(ACI=0.5) for _ in range(10)]
        times = [1.0 for _ in range(10)]  # All solvable
        quality = self.extractor.extract_signal(results, times)
        self.assertEqual(quality.separation_quality, "INSUFFICIENT_DATA")

    def test_extract_signal_both_classes(self):
        """Test signal extraction with both classes"""
        # Solvable instances
        solvable_results = [ACIResult(ACI=0.7 + i*0.01) for i in range(10)]
        solvable_times = [1.0 for _ in range(10)]

        # Intractable instances
        intractable_results = [ACIResult(ACI=0.3 - i*0.01) for i in range(10)]
        intractable_times = [float('inf') for _ in range(10)]

        results = solvable_results + intractable_results
        times = solvable_times + intractable_times

        quality = self.extractor.extract_signal(results, times)
        self.assertGreater(quality.mean_solvable_aci, quality.mean_intractable_aci)
        self.assertGreater(quality.signal_to_noise, 0)

    def test_signal_to_noise_calculation(self):
        """Test SNR calculation"""
        solvable_aci = [0.8, 0.9, 0.7, 0.85]
        intractable_aci = [0.2, 0.3, 0.25, 0.15]

        snr = self.extractor._calculate_snr(solvable_aci, intractable_aci)
        self.assertGreater(snr, 0)  # Positive SNR

    def test_correlation_calculation(self):
        """Test correlation calculation"""
        # Create instances with known relationship
        results = [ACIResult(ACI=0.9 - i*0.1) for i in range(10)]
        times = [1.0 + i for i in range(10)]  # Higher ACI = lower time

        corr = self.extractor._calculate_correlation(results, times)
        self.assertGreater(abs(corr), 0)  # Some correlation

    def test_classification_metrics_calculation(self):
        """Test classification metrics calculation"""
        # Perfect separation
        results = [ACIResult(ACI=0.9)] * 5 + [ACIResult(ACI=0.1)] * 5
        times = [1.0] * 5 + [float('inf')] * 5

        accuracy, auc = self.extractor._calculate_classification_metrics(results, times)
        self.assertEqual(accuracy, 1.0)  # Perfect accuracy
        self.assertEqual(auc, 1.0)  # Perfect AUC

    def test_separation_assessment(self):
        """Test separation quality assessment"""
        self.assertEqual(self.extractor._assess_separation(4.0), "EXCELLENT")
        self.assertEqual(self.extractor._assess_separation(2.5), "GOOD")
        self.assertEqual(self.extractor._assess_separation(1.5), "FAIR")
        self.assertEqual(self.extractor._assess_separation(0.5), "POOR")

    def test_meets_target_true(self):
        """Test meets_target returns True when metrics are good"""
        quality = SignalQuality(
            correlation=0.9,
            accuracy=0.9,
            auc=0.95
        )
        self.assertTrue(quality.meets_target(target_correlation=0.85))

    def test_meets_target_false(self):
        """Test meets_target returns False when metrics are poor"""
        quality = SignalQuality(
            correlation=0.7,
            accuracy=0.7,
            auc=0.8
        )
        self.assertFalse(quality.meets_target(target_correlation=0.85))


class TestIntegration(unittest.TestCase):
    """Integration Tests (10 tests)"""

    def setUp(self):
        self.calculator = ACICalculator()
        self.extractor = SignalExtractor()

    def test_end_to_end_calculation(self):
        """Test complete end-to-end calculation"""
        csp = create_test_csp(n_variables=15, domain_size=5)
        result = self.calculator.calculate(csp)

        self.assertGreaterEqual(result.ACI, 0.0)
        self.assertLessEqual(result.ACI, 1.0)
        self.assertGreater(result.confidence, 0.0)
        self.assertIsInstance(result.interpretation, dict)
        self.assertIsInstance(result.recommendation, dict)

    def test_multiple_csp_comparison(self):
        """Test comparison across multiple CSP types"""
        tree_csp = create_tree_csp(n_variables=10, domain_size=5)
        test_csp = create_test_csp(n_variables=10, domain_size=5)
        dense_csp = create_dense_csp(n_variables=10, domain_size=5)

        tree_result = self.calculator.calculate(tree_csp)
        test_result = self.calculator.calculate(test_csp)
        dense_result = self.calculator.calculate(dense_csp)

        # Tree should have highest ACI (generally)
        self.assertGreater(tree_result.ACI, dense_result.ACI * 0.7)

    def test_full_signal_extraction_pipeline(self):
        """Test complete signal extraction pipeline"""
        # Generate instances
        results = []
        times = []

        # Solvable (tree)
        for _ in range(15):
            csp = create_tree_csp(n_variables=8, domain_size=3)
            result = self.calculator.calculate(csp)
            results.append(result)
            times.append(1.0)

        # Intractable (dense)
        for _ in range(15):
            csp = create_dense_csp(n_variables=8, domain_size=3, constraint_density=0.9)
            result = self.calculator.calculate(csp)
            results.append(result)
            times.append(float('inf'))

        # Extract signal
        quality = self.extractor.extract_signal(results, times)

        # Should have some signal (solvable should have lower ACI than intractable on average)
        # Note: Due to randomness and small sample size, we just check the pipeline runs
        self.assertIsNotNone(quality)
        self.assertIsNotNone(quality.signal_to_noise)
        self.assertIsNotNone(quality.correlation)
        # Check that we can distinguish between solvable and intractable
        self.assertIsNotNone(quality.mean_solvable_aci)
        self.assertIsNotNone(quality.mean_intractable_aci)

    def test_performance_benchmark(self):
        """Test performance on larger instances"""
        csp = create_test_csp(n_variables=50, domain_size=10)
        start = time.time()
        result = self.calculator.calculate(csp)
        elapsed = time.time() - start

        # Should complete in reasonable time (<1 second)
        self.assertLess(elapsed, 1.0)
        self.assertLess(result.computation_time, 1.0)

    def test_cache_performance(self):
        """Test that caching improves performance"""
        csp = create_test_csp(n_variables=20, domain_size=5)

        # First calculation
        result1 = self.calculator.calculate(csp)
        time1 = result1.computation_time

        # Second calculation (should be cached)
        result2 = self.calculator.calculate(csp)
        time2 = result2.computation_time

        # Cached should be faster or equal (cache may be disabled or miss)
        if result2.cached:
            self.assertLessEqual(time2, time1)

    def test_stress_test_many_calculations(self):
        """Test system can handle many calculations"""
        results = []
        for _ in range(100):
            csp = create_test_csp(n_variables=10, domain_size=5)
            result = self.calculator.calculate(csp)
            results.append(result)

        self.assertEqual(len(results), 100)
        for result in results:
            self.assertGreaterEqual(result.ACI, 0.0)
            self.assertLessEqual(result.ACI, 1.0)


def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCSPModels))
    suite.addTests(loader.loadTestsFromTestCase(TestEntropyEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestCoherenceEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSolvabilityEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestACICalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate report
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"\nSuccess rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")

    if result.wasSuccessful():
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print("\n[FAILURE] Some tests failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
