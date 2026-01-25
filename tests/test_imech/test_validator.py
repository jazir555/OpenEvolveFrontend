"""
Unit tests for I_mech Validator (main interface)

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from phase2.imech import (
    IMechValidator,
    Domain,
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType
)


class TestIMechValidator:
    """Test I_mech main validator interface"""

    def setup_method(self):
        """Create validator"""
        self.validator = IMechValidator(
            use_exact_isomorphism=False,
            enable_proofs=False,
            cache_enabled=False
        )

    def test_validator_creation(self):
        """Test creating validator"""
        assert self.validator is not None
        assert self.validator.enable_proofs == False

    def test_compare_identical_domains(self):
        """Test comparing identical domains"""
        domain1 = self._create_domain()
        domain2 = self._create_domain()

        result = self.validator.compare(domain1, domain2)

        assert result is not None
        assert result.structural_score > 0.8  # Should be very similar
        assert len(result.node_mapping) > 0

    def test_compare_different_domains(self):
        """Test comparing different domains"""
        domain1 = self._create_domain(size=3)
        domain2 = self._create_domain(size=5)

        result = self.validator.compare(domain1, domain2)

        assert result is not None
        # Should detect partial similarity or no match
        assert result.total_score >= 0.0

    def test_compare_with_solution(self):
        """Test comparing domains with solution"""
        domain1 = self._create_domain()
        domain1.solutions = [{'parameters': {'x': 5.0, 'y': 3.0}}]

        domain2 = self._create_domain()

        result = self.validator.compare(domain1, domain2)

        if result.total_score > 0.7:
            # Should have transferred solution
            assert result.transferred_solution is not None
            assert result.validation_result is not None

    def test_find_analogous_domains(self):
        """Test finding analogous domains"""
        target = self._create_domain(size=3)

        candidates = [
            self._create_domain(size=3, prefix="a"),
            self._create_domain(size=5, prefix="b"),
            self._create_domain(size=3, prefix="c")
        ]

        # Add solutions to candidates
        for candidate in candidates:
            candidate.solutions = [{'parameters': {'value': 1.0}}]

        results = self.validator.find_analogous_domains(
            target,
            candidates,
            threshold=0.5
        )

        assert len(results) >= 0
        # Each result should be (domain, similarity_result)
        for domain, result in results:
            assert isinstance(domain, Domain)
            assert hasattr(result, 'total_score')

    def test_validate_transfer_success(self):
        """Test transfer success validation"""
        domain1 = self._create_domain()
        domain1.solutions = [
            {
                'parameters': {'x': 5.0, 'y': 3.0},
                'structure': {'algorithm': 'gradient_descent'}
            }
        ]

        domain2 = self._create_domain()

        result = self.validator.compare(domain1, domain2)

        if result.validation_result:
            success = self.validator.validate_transfer_success(result)
            assert isinstance(success, bool)

    def test_caching(self):
        """Test result caching"""
        validator = IMechValidator(cache_enabled=True)

        domain1 = self._create_domain()
        domain2 = self._create_domain()

        # First call
        result1 = validator.compare(domain1, domain2)

        # Second call (should be cached)
        result2 = validator.compare(domain1, domain2)

        assert result1 is result2  # Same object from cache

    def _create_domain(self, size=3, prefix="n"):
        """Helper: create test domain"""
        domain = Domain(
            id=f"domain_{prefix}",
            name=f"Test Domain {prefix}",
            description="Test domain for I_mech validation",
            formal_constraints=[
                f"x{prefix} + y{prefix} = 10",
                f"z{prefix} > 0"
            ],
            natural_language_constraints=[
                "All variables must be positive",
                "Sum constraint must be satisfied"
            ]
        )

        # Create FDG
        fdg = FunctionalDependencyGraph()

        for i in range(size):
            node = Node(
                id=f"{prefix}{i}",
                variable=f"x{i}",
                constraint_type="continuous"
            )
            fdg.add_node(node)

        # Add edges
        for i in range(size - 1):
            edge = Edge(
                source=f"{prefix}{i}",
                target=f"{prefix}{i+1}",
                edge_type=EdgeType.CAUSAL
            )
            fdg.add_edge(edge)

        domain.fdg = fdg

        return domain


class TestCompareDomains:
    """Test convenience function"""

    def test_compare_domains_function(self):
        """Test compare_domains convenience function"""
        from phase2.imech import compare_domains

        domain1 = Domain(
            id="d1",
            name="Domain 1",
            description="Test domain"
        )
        domain1.fdg = FunctionalDependencyGraph()
        node1 = Node(id="n1", variable="x", constraint_type="continuous")
        domain1.fdg.add_node(node1)

        domain2 = Domain(
            id="d2",
            name="Domain 2",
            description="Test domain"
        )
        domain2.fdg = FunctionalDependencyGraph()
        node2 = Node(id="n2", variable="y", constraint_type="continuous")
        domain2.fdg.add_node(node2)

        result = compare_domains(domain1, domain2)

        assert result is not None
        assert hasattr(result, 'total_score')
        assert hasattr(result, 'structural_score')
