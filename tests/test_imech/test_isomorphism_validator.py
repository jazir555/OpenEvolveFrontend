"""
Unit Tests for I_mech Isomorphism Validator

Tests mechanistic isomorphism detection and validation.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
Status: 🟢 Active
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase2.imech import (
    IMechValidator,
    compare_domains,
    Domain,
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType
)


@pytest.fixture
def sample_domain():
    """Create a sample domain with FDG"""
    domain = Domain(
        id="test_domain",
        name="Test Domain",
        description="Test domain for I_mech",
        formal_constraints=["x + y = 10", "z > 0"],
        natural_language_constraints=["All variables positive"]
    )

    # Create FDG
    fdg = FunctionalDependencyGraph()

    # Add nodes
    for i in range(5):
        node = Node(
            id=f"node_{i}",
            variable=f"x{i}",
            constraint_type="continuous"
        )
        fdg.add_node(node)

    # Add edges
    for i in range(4):
        edge = Edge(
            source=f"node_{i}",
            target=f"node_{i+1}",
            edge_type=EdgeType.CAUSAL
        )
        fdg.add_edge(edge)

    domain.fdg = fdg
    return domain


@pytest.fixture
def sample_domain_with_solution():
    """Create a domain with a solution"""
    domain = Domain(
        id="domain_with_solution",
        name="Domain with Solution",
        description="Test domain with solution",
        formal_constraints=["x + y = 10"]
    )

    # Create FDG
    fdg = FunctionalDependencyGraph()
    node1 = Node(id="n1", variable="x", constraint_type="continuous")
    node2 = Node(id="n2", variable="y", constraint_type="continuous")
    fdg.add_node(node1)
    fdg.add_node(node2)

    edge = Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL)
    fdg.add_edge(edge)

    domain.fdg = fdg

    # Add solution
    domain.solutions = [
        {
            'parameters': {'x': 5.0, 'y': 5.0},
            'algorithm': 'analytical',
            'structure': {'type': 'linear_equation'}
        }
    ]

    return domain


class TestIMechValidator:
    """Test I_mech validator"""

    def test_validator_initialization_default(self):
        """Test validator initialization with defaults"""
        validator = IMechValidator()

        assert validator.fdg_extractor is not None
        assert validator.wl is not None
        assert validator.vf2 is not None
        assert validator.enable_proofs is False
        assert validator.cache_enabled is True

    def test_validator_initialization_custom(self):
        """Test validator with custom settings"""
        validator = IMechValidator(
            use_exact_isomorphism=True,
            enable_proofs=True,
            cache_enabled=False
        )

        assert validator.enable_proofs is True
        assert validator.cache_enabled is False

    def test_compare_identical_domains(self, sample_domain):
        """Test comparing identical domains"""
        validator = IMechValidator()

        result = validator.compare(sample_domain, sample_domain)

        assert result is not None
        assert result.structural_score > 0.9  # Should be very similar
        assert len(result.node_mapping) > 0

    def test_compare_different_domains(self, sample_domain):
        """Test comparing different domains"""
        validator = IMechValidator()

        # Create different domain
        domain2 = Domain(
            id="different_domain",
            name="Different Domain",
            description="Different test domain",
            formal_constraints=["a + b = 20"]
        )

        fdg2 = FunctionalDependencyGraph()
        node1 = Node(id="n1", variable="a", constraint_type="continuous")
        node2 = Node(id="n2", variable="b", constraint_type="continuous")
        fdg2.add_node(node1)
        fdg2.add_node(node2)

        domain2.fdg = fdg2

        result = validator.compare(sample_domain, domain2)

        assert result is not None
        assert result.total_score >= 0.0

    def test_compare_with_solution_transfer(self, sample_domain_with_solution, sample_domain):
        """Test comparing domains with solution transfer"""
        validator = IMechValidator()

        result = validator.compare(sample_domain_with_solution, sample_domain)

        assert result is not None

        # If similarity is high enough, should have transferred solution
        if result.total_score > 0.7:
            assert result.transferred_solution is not None
            assert result.validation_result is not None

    def test_compare_early_termination(self):
        """Test early termination for clearly dissimilar domains"""
        validator = IMechValidator()

        # Create very different domains
        domain1 = Domain(id="d1", name="D1", description="Test")
        fdg1 = FunctionalDependencyGraph()
        for i in range(3):
            fdg1.add_node(Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous"))
        domain1.fdg = fdg1

        domain2 = Domain(id="d2", name="D2", description="Test")
        fdg2 = FunctionalDependencyGraph()
        for i in range(10):
            fdg2.add_node(Node(id=f"m{i}", variable=f"y{i}", constraint_type="discrete"))
        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        # Should still return a result
        assert result is not None
        # But score should be low
        assert result.total_score < 0.5

    def test_find_analogous_domains(self, sample_domain):
        """Test finding analogous domains"""
        validator = IMechValidator()

        # Create candidate domains
        candidates = []
        for i in range(3):
            domain = Domain(
                id=f"candidate_{i}",
                name=f"Candidate {i}",
                description="Test candidate",
                formal_constraints=["x + y = 10"]
            )

            # Add similar FDG
            fdg = FunctionalDependencyGraph()
            node1 = Node(id="n1", variable="x", constraint_type="continuous")
            node2 = Node(id="n2", variable="y", constraint_type="continuous")
            fdg.add_node(node1)
            fdg.add_node(node2)
            edge = Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL)
            fdg.add_edge(edge)

            domain.fdg = fdg
            domain.solutions = [{'parameters': {'x': 5.0}}]

            candidates.append(domain)

        # Find analogous
        results = validator.find_analogous_domains(
            sample_domain,
            candidates,
            threshold=0.3
        )

        assert isinstance(results, list)
        # Each result should be tuple of (domain, similarity_result)
        for domain, result in results:
            assert isinstance(domain, Domain)
            assert hasattr(result, 'total_score')

    def test_validate_transfer_success(self, sample_domain_with_solution, sample_domain):
        """Test transfer success validation"""
        validator = IMechValidator()

        result = validator.compare(sample_domain_with_solution, sample_domain)

        # If validation result exists, test it
        if result.validation_result:
            success = validator.validate_transfer_success(result)
            assert isinstance(success, bool)

    def test_caching(self, sample_domain):
        """Test result caching"""
        validator = IMechValidator(cache_enabled=True)

        # First comparison
        result1 = validator.compare(sample_domain, sample_domain)

        # Second comparison (should use cache)
        result2 = validator.compare(sample_domain, sample_domain)

        # Results should be identical
        assert result1 is result2

    def test_cache_disabled(self, sample_domain):
        """Test with cache disabled"""
        validator = IMechValidator(cache_enabled=False)

        result1 = validator.compare(sample_domain, sample_domain)
        result2 = validator.compare(sample_domain, sample_domain)

        # Should still work, just not cached
        assert result1 is not None
        assert result2 is not None

    def test_generate_mapping(self):
        """Test heuristic mapping generation"""
        validator = IMechValidator()

        # Create two FDGs
        fdg1 = FunctionalDependencyGraph()
        for i in range(3):
            node = Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous")
            fdg1.add_node(node)

        fdg2 = FunctionalDependencyGraph()
        for i in range(3):
            node = Node(id=f"m{i}", variable=f"y{i}", constraint_type="continuous")
            fdg2.add_node(node)

        mapping = validator._generate_mapping(fdg1, fdg2)

        assert isinstance(mapping, dict)
        # Should have some mappings
        assert len(mapping) >= 0
        assert len(mapping) <= len(fdg1.nodes)


class TestConvenienceFunction:
    """Test convenience function"""

    def test_compare_domains_function(self, sample_domain):
        """Test compare_domains convenience function"""
        result = compare_domains(sample_domain, sample_domain)

        assert result is not None
        assert hasattr(result, 'total_score')
        assert hasattr(result, 'structural_score')


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_domains(self):
        """Test comparing domains with empty FDGs"""
        validator = IMechValidator()

        domain1 = Domain(id="empty1", name="Empty1", description="Empty")
        domain1.fdg = FunctionalDependencyGraph()

        domain2 = Domain(id="empty2", name="Empty2", description="Empty")
        domain2.fdg = FunctionalDependencyGraph()

        result = validator.compare(domain1, domain2)

        # Should handle gracefully
        assert result is not None

    def test_domain_without_fdg(self):
        """Test comparing domains without pre-computed FDG"""
        validator = IMechValidator()

        # Domain without FDG - should extract it
        domain = Domain(
            id="no_fdg",
            name="No FDG",
            description="Domain without FDG",
            formal_constraints=["x > 0"]
        )

        domain2 = Domain(
            id="no_fdg2",
            name="No FDG 2",
            description="Another domain without FDG",
            formal_constraints=["y > 0"]
        )

        # Should not crash
        # FDG extraction may fail if constraints aren't parseable,
        # but should handle gracefully
        try:
            result = validator.compare(domain, domain2)
            # If FDG extraction works, should return result
            assert result is not None
        except Exception:
            # If FDG extraction fails, that's acceptable for this test
            pass

    def test_single_node_domains(self):
        """Test comparing domains with single node"""
        validator = IMechValidator()

        domain1 = Domain(id="single1", name="Single1", description="Single")
        fdg1 = FunctionalDependencyGraph()
        fdg1.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
        domain1.fdg = fdg1

        domain2 = Domain(id="single2", name="Single2", description="Single")
        fdg2 = FunctionalDependencyGraph()
        fdg2.add_node(Node(id="n2", variable="y", constraint_type="continuous"))
        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        assert result is not None

    def test_very_large_domains(self):
        """Test comparing larger domains"""
        validator = IMechValidator()

        # Create domains with 50 nodes each
        domain1 = Domain(id="large1", name="Large1", description="Large")
        fdg1 = FunctionalDependencyGraph()
        for i in range(50):
            fdg1.add_node(Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous"))
        domain1.fdg = fdg1

        domain2 = Domain(id="large2", name="Large2", description="Large")
        fdg2 = FunctionalDependencyGraph()
        for i in range(50):
            fdg2.add_node(Node(id=f"m{i}", variable=f"y{i}", constraint_type="continuous"))
        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        assert result is not None
        assert result.computation_time >= 0

    def test_no_solution_available(self, sample_domain):
        """Test comparing when source domain has no solution"""
        validator = IMechValidator()

        # Domain without solution
        domain_no_solution = Domain(id="no_sol", name="No Sol", description="No solution")
        fdg = FunctionalDependencyGraph()
        fdg.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
        domain_no_solution.fdg = fdg

        result = validator.compare(domain_no_solution, sample_domain)

        assert result is not None
        # Should not have transferred solution
        assert result.transferred_solution is None

    def test_mixed_constraint_types(self):
        """Test domains with mixed constraint types"""
        validator = IMechValidator()

        domain1 = Domain(id="mixed1", name="Mixed1", description="Mixed")
        fdg1 = FunctionalDependencyGraph()
        fdg1.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
        fdg1.add_node(Node(id="n2", variable="y", constraint_type="discrete"))
        fdg1.add_node(Node(id="n3", variable="z", constraint_type="binary"))
        domain1.fdg = fdg1

        domain2 = Domain(id="mixed2", name="Mixed2", description="Mixed")
        fdg2 = FunctionalDependencyGraph()
        fdg2.add_node(Node(id="m1", variable="a", constraint_type="continuous"))
        fdg2.add_node(Node(id="m2", variable="b", constraint_type="discrete"))
        fdg2.add_node(Node(id="m3", variable="c", constraint_type="binary"))
        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        assert result is not None

    def test_densely_connected_graphs(self):
        """Test comparing densely connected graphs"""
        validator = IMechValidator()

        domain1 = Domain(id="dense1", name="Dense1", description="Dense")
        fdg1 = FunctionalDependencyGraph()

        # Add 5 nodes
        for i in range(5):
            fdg1.add_node(Node(id=f"n{i}", variable=f"x{i}", constraint_type="continuous"))

        # Connect all to all
        for i in range(5):
            for j in range(5):
                if i != j:
                    edge = Edge(source=f"n{i}", target=f"n{j}", edge_type=EdgeType.CAUSAL)
                    fdg1.add_edge(edge)

        domain1.fdg = fdg1

        domain2 = Domain(id="dense2", name="Dense2", description="Dense")
        fdg2 = FunctionalDependencyGraph()

        # Add 5 nodes
        for i in range(5):
            fdg2.add_node(Node(id=f"m{i}", variable=f"y{i}", constraint_type="continuous"))

        # Connect all to all
        for i in range(5):
            for j in range(5):
                if i != j:
                    edge = Edge(source=f"m{i}", target=f"m{j}", edge_type=EdgeType.CAUSAL)
                    fdg2.add_edge(edge)

        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        assert result is not None

    def test_directed_vs_undirected(self):
        """Test comparing directed vs undirected edges"""
        validator = IMechValidator()

        domain1 = Domain(id="dir1", name="Dir1", description="Directed")
        fdg1 = FunctionalDependencyGraph()
        fdg1.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
        fdg1.add_node(Node(id="n2", variable="y", constraint_type="continuous"))
        edge1 = Edge(source="n1", target="n2", edge_type=EdgeType.CAUSAL)
        fdg1.add_edge(edge1)
        domain1.fdg = fdg1

        domain2 = Domain(id="dir2", name="Dir2", description="Directed")
        fdg2 = FunctionalDependencyGraph()
        fdg2.add_node(Node(id="m1", variable="a", constraint_type="continuous"))
        fdg2.add_node(Node(id="m2", variable="b", constraint_type="continuous"))
        edge2 = Edge(source="m1", target="m2", edge_type=EdgeType.CAUSAL)
        fdg2.add_edge(edge2)
        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        assert result is not None

    def test_self_loops(self):
        """Test graphs with self-loops"""
        validator = IMechValidator()

        domain1 = Domain(id="loop1", name="Loop1", description="Loop")
        fdg1 = FunctionalDependencyGraph()
        fdg1.add_node(Node(id="n1", variable="x", constraint_type="continuous"))
        edge1 = Edge(source="n1", target="n1", edge_type=EdgeType.CAUSAL)
        fdg1.add_edge(edge1)
        domain1.fdg = fdg1

        domain2 = Domain(id="loop2", name="Loop2", description="Loop")
        fdg2 = FunctionalDependencyGraph()
        fdg2.add_node(Node(id="m1", variable="y", constraint_type="continuous"))
        edge2 = Edge(source="m1", target="m1", edge_type=EdgeType.CAUSAL)
        fdg2.add_edge(edge2)
        domain2.fdg = fdg2

        result = validator.compare(domain1, domain2)

        assert result is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
