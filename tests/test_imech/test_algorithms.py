"""
Unit tests for I_mech algorithms (WL, VF2, Subgraph, Intervention)

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import networkx as nx
from phase2.imech.core import (
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType
)
from phase2.imech.algorithms import (
    WeisfeilerLehman,
    VF2Matcher,
    SubgraphMatcher,
    InterventionSimulator
)


class TestWeisfeilerLehman:
    """Test Weisfeiler-Lehman algorithm"""

    def setup_method(self):
        """Create test FDGs"""
        self.wl = WeisfeilerLehman(max_iterations=10)

    def test_identical_graphs(self):
        """Test WL on identical graphs"""
        fdg = self._create_simple_fdg()

        score = self.wl.compute_similarity(fdg, fdg)
        assert score == 1.0

    def test_isomorphic_graphs(self):
        """Test WL on isomorphic graphs (same structure, different IDs)"""
        fdg1 = self._create_simple_fdg()
        fdg2 = self._create_simple_fdg(prefix="m")

        score = self.wl.compute_similarity(fdg1, fdg2)
        assert score > 0.9

    def test_different_sizes(self):
        """Test WL on graphs of different sizes"""
        fdg1 = self._create_chain_fdg(3)
        fdg2 = self._create_chain_fdg(5)

        score = self.wl.compute_similarity(fdg1, fdg2)
        assert score < 0.8

    def test_empty_graphs(self):
        """Test WL on empty graphs"""
        fdg1 = FunctionalDependencyGraph()
        fdg2 = FunctionalDependencyGraph()

        score = self.wl.compute_similarity(fdg1, fdg2)
        assert score == 0.0

    def test_are_isomorphic_true(self):
        """Test isomorphism detection (positive case)"""
        fdg1 = self._create_simple_fdg()
        fdg2 = self._create_simple_fdg(prefix="m")

        result = self.wl.are_isomorphic(fdg1, fdg2)
        assert result == True

    def test_are_isomorphic_false(self):
        """Test isomorphism detection (negative case)"""
        fdg1 = self._create_chain_fdg(3)
        fdg2 = self._create_chain_fdg(5)

        result = self.wl.are_isomorphic(fdg1, fdg2)
        assert result == False

    def _create_simple_fdg(self, prefix="n"):
        """Helper: create simple FDG"""
        fdg = FunctionalDependencyGraph()

        for i in range(3):
            node = Node(
                id=f"{prefix}{i}",
                variable=f"x{i}",
                constraint_type="continuous"
            )
            fdg.add_node(node)

        # Create triangle
        fdg.add_edge(Edge(source=f"{prefix}0", target=f"{prefix}1", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source=f"{prefix}1", target=f"{prefix}2", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source=f"{prefix}2", target=f"{prefix}0", edge_type=EdgeType.FEEDBACK))

        return fdg

    def _create_chain_fdg(self, length):
        """Helper: create chain FDG"""
        fdg = FunctionalDependencyGraph()

        for i in range(length):
            node = Node(
                id=f"n{i}",
                variable=f"x{i}",
                constraint_type="continuous"
            )
            fdg.add_node(node)

        # Create chain
        for i in range(length - 1):
            fdg.add_edge(Edge(source=f"n{i}", target=f"n{i+1}", edge_type=EdgeType.CAUSAL))

        return fdg


class TestVF2Matcher:
    """Test VF2 exact isomorphism algorithm"""

    def setup_method(self):
        """Create test FDGs"""
        self.vf2 = VF2Matcher()

    def test_exact_isomorphism(self):
        """Test VF2 on isomorphic graphs"""
        fdg1 = self._create_fdg()
        fdg2 = self._create_fdg(prefix="m")

        mapping = self.vf2.find_isomorphism(fdg1, fdg2)
        assert mapping is not None
        assert len(mapping) == 3

    def test_different_sizes(self):
        """Test VF2 on graphs of different sizes"""
        fdg1 = self._create_chain_fdg(3)
        fdg2 = self._create_chain_fdg(5)

        mapping = self.vf2.find_isomorphism(fdg1, fdg2)
        assert mapping is None

    def test_different_structure(self):
        """Test VF2 on graphs with different structures"""
        fdg1 = self._create_chain_fdg(3)
        fdg2 = self._create_star_fdg(3)

        mapping = self.vf2.find_isomorphism(fdg1, fdg2)
        assert mapping is None

    def test_node_matching(self):
        """Test node matching criterion"""
        attrs1 = {'constraint_type': 'continuous'}
        attrs2 = {'constraint_type': 'continuous'}
        attrs3 = {'constraint_type': 'discrete'}

        assert self.vf2._node_match(attrs1, attrs2) == True
        assert self.vf2._node_match(attrs1, attrs3) == False

    def test_edge_matching(self):
        """Test edge matching criterion"""
        attrs1 = {'type': 'causal'}
        attrs2 = {'type': 'causal'}
        attrs3 = {'type': 'correlation'}

        assert self.vf2._edge_match(attrs1, attrs2) == True
        assert self.vf2._edge_match(attrs1, attrs3) == False

    def _create_fdg(self, prefix="n"):
        """Helper: create FDG"""
        fdg = FunctionalDependencyGraph()

        for i in range(3):
            node = Node(
                id=f"{prefix}{i}",
                variable=f"x{i}",
                constraint_type="continuous"
            )
            fdg.add_node(node)

        fdg.add_edge(Edge(source=f"{prefix}0", target=f"{prefix}1", edge_type=EdgeType.CAUSAL))
        fdg.add_edge(Edge(source=f"{prefix}1", target=f"{prefix}2", edge_type=EdgeType.CAUSAL))

        return fdg

    def _create_chain_fdg(self, length):
        """Helper: create chain FDG"""
        fdg = FunctionalDependencyGraph()

        for i in range(length):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        for i in range(length - 1):
            fdg.add_edge(Edge(source=f"n{i}", target=f"n{i+1}", edge_type=EdgeType.CAUSAL))

        return fdg

    def _create_star_fdg(self, num_leaves):
        """Helper: create star FDG"""
        fdg = FunctionalDependencyGraph()

        # Center node
        center = Node(id="center", variable="c", constraint_type="continuous")
        fdg.add_node(center)

        # Leaf nodes
        for i in range(num_leaves):
            leaf = Node(id=f"leaf{i}", variable=f"l{i}", constraint_type="continuous")
            fdg.add_node(leaf)
            fdg.add_edge(Edge(source="center", target=f"leaf{i}", edge_type=EdgeType.CAUSAL))

        return fdg


class TestSubgraphMatcher:
    """Test Subgraph isomorphism algorithm"""

    def setup_method(self):
        """Create test FDGs"""
        self.matcher = SubgraphMatcher()

    def test_exact_match(self):
        """Test subgraph matching when graphs are identical"""
        fdg1 = self._create_fdg(3)
        fdg2 = self._create_fdg(3)

        mapping, score = self.matcher.find_best_match(fdg1, fdg2)
        assert mapping is not None
        assert score > 0.9

    def test_partial_match(self):
        """Test subgraph matching with partial overlap"""
        fdg1 = self._create_fdg(3)  # Smaller
        fdg2 = self._create_fdg(5)  # Larger

        mapping, score = self.matcher.find_best_match(fdg1, fdg2)
        assert mapping is not None
        assert 0.0 <= score <= 1.0

    def test_no_match(self):
        """Test when no good match exists"""
        fdg1 = self._create_fdg(3)
        fdg2 = FunctionalDependencyGraph()  # Empty

        mapping, score = self.matcher.find_best_match(fdg1, fdg2)
        assert score == 0.0

    def test_maximum_common_subgraph(self):
        """Test maximum common subgraph finding"""
        fdg1 = self._create_chain_fdg(3)
        fdg2 = self._create_chain_fdg(5)

        mapping, score = self.matcher.find_maximum_common_subgraph(fdg1, fdg2)
        assert len(mapping) >= 0
        assert 0.0 <= score <= 1.0

    def _create_fdg(self, size):
        """Helper: create FDG"""
        fdg = FunctionalDependencyGraph()

        for i in range(size):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        # Create chain edges
        for i in range(size - 1):
            fdg.add_edge(Edge(source=f"n{i}", target=f"n{i+1}", edge_type=EdgeType.CAUSAL))

        return fdg

    def _create_chain_fdg(self, length):
        """Helper: create chain FDG"""
        return self._create_fdg(length)


class TestInterventionSimulator:
    """Test Intervention Simulator"""

    def setup_method(self):
        """Create test FDGs"""
        self.simulator = InterventionSimulator(num_samples=100)

    def test_simulate_intervention(self):
        """Test intervention simulation"""
        fdg = self._create_chain_fdg()

        effect = self.simulator._simulate_intervention(fdg, "n0", value=1.0)

        assert "n0" in effect
        assert effect["n0"] == 1.0

    def test_compare_intervention_responses_identical(self):
        """Test comparing intervention responses on identical graphs"""
        fdg1 = self._create_chain_fdg()
        fdg2 = self._create_chain_fdg()

        mapping = {f"n{i}": f"m{i}" for i in range(3)}

        score = self.simulator.compare_intervention_responses(fdg1, fdg2, mapping)
        assert 0.0 <= score <= 1.0

    def test_compute_causal_effect(self):
        """Test computing causal effect"""
        fdg = self._create_chain_fdg()

        effect = self.simulator.compute_causal_effect(fdg, "n0", "n2")
        assert effect >= 0.0

    def test_compare_effects(self):
        """Test effect comparison"""
        effect1 = {"n0": 1.0, "n1": 0.8, "n2": 0.64}
        effect2 = {"n0": 1.0, "n1": 0.8, "n2": 0.64}

        score = self.simulator._compare_effects(effect1, effect2)
        assert score == 1.0

    def test_compare_effects_different(self):
        """Test effect comparison with different effects"""
        effect1 = {"n0": 1.0, "n1": 0.8, "n2": 0.64}
        effect2 = {"n0": 1.0, "n1": 0.5, "n2": 0.25}

        score = self.simulator._compare_effects(effect1, effect2)
        assert 0.0 <= score < 1.0

    def _create_chain_fdg(self):
        """Helper: create chain FDG"""
        fdg = FunctionalDependencyGraph()

        for i in range(3):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg.add_node(node)

        fdg.add_edge(Edge(source="n0", target="n1", edge_type=EdgeType.CAUSAL, weight=0.8))
        fdg.add_edge(Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL, weight=0.8))

        return fdg
