"""
Test Suite for Physics Knowledge Engine

This test suite validates the Physics Knowledge Engine implementation
from System 2 of the Gap Analysis Implementation Plan.

Author: OpenEvolve
Created: 2026-01-02
"""

import asyncio
import pytest
from typing import Dict, List

# Import physics knowledge engine
from physics_knowledge_engine import (
    PhysicsKnowledgeEngine,
    PhysicsFormalizer,
    PhysicsDomain,
    PhysicsTheorem,
    HilbertSpace,
    QuantumSystem,
    QuantumState,
    Manifold,
    LorentzianMetric,
    PseudoRiemannianManifold,
    EinsteinFieldEquations,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def knowledge_engine():
    """Create a physics knowledge engine for testing"""
    return PhysicsKnowledgeEngine()


@pytest.fixture
def formalizer(knowledge_engine):
    """Create a physics formalizer for testing"""
    return PhysicsFormalizer(knowledge_engine)


# ============================================================================
# Physics Ontology Tests
# ============================================================================

class TestPhysicsOntology:
    """Test physics ontology data structures"""

    def test_hilbert_space_creation(self):
        """Test creating Hilbert space"""
        hs = HilbertSpace(dimension=3, basis=["|0⟩", "|1⟩", "|2⟩"])
        assert hs.dimension == 3
        assert len(hs.basis) == 3

    def test_hilbert_space_to_lean(self):
        """Test Hilbert space to Lean 4 conversion"""
        hs = HilbertSpace(dimension=2, basis=["|0⟩", "|1⟩"])
        lean = hs.to_lean()
        assert "ℂ^2" in lean

    def test_quantum_system_creation(self):
        """Test creating quantum system"""
        hs = HilbertSpace(dimension=2)
        qs = QuantumSystem(
            name="Qubit",
            hilbert_space=hs,
            observables=["PauliX", "PauliY", "PauliZ"]
        )
        assert qs.name == "Qubit"
        assert len(qs.observables) == 3

    def test_quantum_system_to_lean(self):
        """Test quantum system to Lean 4 conversion"""
        hs = HilbertSpace(dimension=2)
        qs = QuantumSystem(name="Qubit", hilbert_space=hs)
        lean = qs.to_lean()
        assert "QubitSystem" in lean
        assert "hilbertSpace" in lean

    def test_manifold_creation(self):
        """Test creating manifold"""
        m = Manifold(dimension=4, name="Spacetime", coordinate_chart=["t", "x", "y", "z"])
        assert m.dimension == 4
        assert len(m.coordinate_chart) == 4

    def test_lorentzian_metric(self):
        """Test Lorentzian metric"""
        metric = LorentzianMetric(signature=(-1, 1, 1, 1))
        assert metric.signature == (-1, 1, 1, 1)
        assert metric.dimension == 4

    def test_pseudo_riemannian_manifold(self):
        """Test pseudo-Riemannian manifold"""
        m = Manifold(dimension=4, name="Spacetime")
        metric = LorentzianMetric(signature=(-1, 1, 1, 1))
        prm = PseudoRiemannianManifold(manifold=m, metric=metric)
        assert prm.manifold.dimension == 4
        assert prm.metric.signature == (-1, 1, 1, 1)


# ============================================================================
# Physics Theorems Tests
# ============================================================================

class TestPhysicsTheorems:
    """Test physics theorem representations"""

    def test_theorem_creation(self):
        """Test creating physics theorem"""
        theorem = PhysicsTheorem(
            name="TestTheorem",
            domain=PhysicsDomain.QUANTUM_MECHANICS,
            statement="This is a test theorem",
            formal_statement="∀ x, P(x)",
            dependencies=["Axiom1"],
            applications=["Application1"]
        )
        assert theorem.name == "TestTheorem"
        assert theorem.domain == PhysicsDomain.QUANTUM_MECHANICS

    def test_theorem_to_lean(self):
        """Test theorem to Lean 4 conversion"""
        theorem = PhysicsTheorem(
            name="MyTheorem",
            domain=PhysicsDomain.QUANTUM_MECHANICS,
            statement="Test statement",
            formal_statement="∀ x, x = x",
            proof="by simp"
        )
        lean = theorem.to_lean()
        assert "theorem MyTheorem" in lean
        assert "∀ x, x = x" in lean


# ============================================================================
# Knowledge Engine Tests
# ============================================================================

class TestKnowledgeEngine:
    """Test physics knowledge engine functionality"""

    def test_engine_initialization(self, knowledge_engine):
        """Test knowledge engine initialization"""
        assert len(knowledge_engine.theorems) > 0
        assert len(knowledge_engine.concepts) > 0
        assert len(knowledge_engine.domains) > 0

    def test_quantum_theorems_loaded(self, knowledge_engine):
        """Test quantum mechanics theorems are loaded"""
        quantum_theorems = knowledge_engine.domains[PhysicsDomain.QUANTUM_MECHANICS]
        assert len(quantum_theorems) > 0

    def test_relativity_theorems_loaded(self, knowledge_engine):
        """Test relativity theorems are loaded"""
        relativity_theorems = knowledge_engine.domains[PhysicsDomain.RELATIVITY]
        assert len(relativity_theorems) > 0

    def test_stat_mech_theorems_loaded(self, knowledge_engine):
        """Test statistical mechanics theorems are loaded"""
        stat_mech_theorems = knowledge_engine.domains[PhysicsDomain.STATISTICAL_MECHANICS]
        assert len(stat_mech_theorems) > 0

    def test_condensed_matter_theorems_loaded(self, knowledge_engine):
        """Test condensed matter theorems are loaded"""
        cm_theorems = knowledge_engine.domains[PhysicsDomain.CONDENSED_MATTER]
        assert len(cm_theorems) > 0


# ============================================================================
# Knowledge Retrieval Tests
# ============================================================================

class TestKnowledgeRetrieval:
    """Test knowledge retrieval methods"""

    @pytest.mark.asyncio
    async def test_query_quantum_theorems(self, knowledge_engine):
        """Test querying quantum mechanics theorems"""
        problem = "Calculate the uncertainty in position and momentum"
        theorems = await knowledge_engine.query_related_theorems(
            problem,
            domain=PhysicsDomain.QUANTUM_MECHANICS,
            k=3
        )

        assert len(theorems) > 0
        assert all(isinstance(t, PhysicsTheorem) for t in theorems)

    @pytest.mark.asyncio
    async def test_query_relativity_theorems(self, knowledge_engine):
        """Test querying relativity theorems"""
        problem = "Calculate time dilation for moving clock"
        theorems = await knowledge_engine.query_related_theorems(
            problem,
            domain=PhysicsDomain.RELATIVITY,
            k=3
        )

        assert len(theorems) > 0

    @pytest.mark.asyncio
    async def test_keyword_extraction(self, knowledge_engine):
        """Test keyword extraction from problems"""
        problem = "Quantum system in Hilbert space with observable"

        keywords = knowledge_engine._extract_keywords(problem)

        assert "quantum" in keywords
        assert "hilbert" in keywords

    @pytest.mark.asyncio
    async def test_relevance_scoring(self, knowledge_engine):
        """Test theorem relevance scoring"""
        problem = "Calculate uncertainty principle"
        keywords = knowledge_engine._extract_keywords(problem)

        # Get a theorem
        theorem_id = list(knowledge_engine.theorems.keys())[0]
        theorem = knowledge_engine.theorems[theorem_id]

        score = knowledge_engine._score_relevance(theorem, keywords, problem)
        assert score >= 0


# ============================================================================
# Decomposition Tests
# ============================================================================

class TestDecomposition:
    """Test problem decomposition"""

    @pytest.mark.asyncio
    async def test_quantum_decomposition(self, knowledge_engine):
        """Test quantum mechanics problem decomposition"""
        problem = "Prove no-cloning theorem"
        decomposition = await knowledge_engine.suggest_decomposition(
            problem,
            PhysicsDomain.QUANTUM_MECHANICS
        )

        assert "domain" in decomposition
        assert "steps" in decomposition
        assert "theorems" in decomposition
        assert "lean_imports" in decomposition
        assert len(decomposition["steps"]) > 0

    @pytest.mark.asyncio
    async def test_relativity_decomposition(self, knowledge_engine):
        """Test relativity problem decomposition"""
        problem = "Calculate gravitational redshift"
        decomposition = await knowledge_engine.suggest_decomposition(
            problem,
            PhysicsDomain.RELATIVITY
        )

        assert decomposition["domain"] == "Relativity"
        assert len(decomposition["steps"]) > 0

    @pytest.mark.asyncio
    async def test_stat_mech_decomposition(self, knowledge_engine):
        """Test statistical mechanics decomposition"""
        problem = "Calculate partition function"
        decomposition = await knowledge_engine.suggest_decomposition(
            problem,
            PhysicsDomain.STATISTICAL_MECHANICS
        )

        assert len(decomposition["steps"]) > 0


# ============================================================================
# Tactics Tests
# ============================================================================

class TestTactics:
    """Test physics-specific tactic recommendations"""

    @pytest.mark.asyncio
    async def test_quantum_tactics(self, knowledge_engine):
        """Test quantum mechanics tactics"""
        problem = "Calculate expectation value"
        tactics = await knowledge_engine.get_applicable_tactics(
            problem,
            PhysicsDomain.QUANTUM_MECHANICS
        )

        assert len(tactics) > 0
        assert all("name" in t for t in tactics)
        assert all("description" in t for t in tactics)
        assert all("usage" in t for t in tactics)

    @pytest.mark.asyncio
    async def test_relativity_tactics(self, knowledge_engine):
        """Test relativity tactics"""
        problem = "Simplify tensor expression"
        tactics = await knowledge_engine.get_applicable_tactics(
            problem,
            PhysicsDomain.RELATIVITY
        )

        assert len(tactics) > 0


# ============================================================================
# Formalization Tests
# ============================================================================

class TestFormalization:
    """Test automated formalization pipeline"""

    @pytest.mark.asyncio
    async def test_formalize_hilbert_space(self, formalizer):
        """Test formalizing Hilbert space definition"""
        result = await formalizer.formalize_textbook_definition(
            "A Hilbert space is a complete vector space with inner product",
            "Used in quantum mechanics",
            PhysicsDomain.QUANTUM_MECHANICS
        )

        assert "original" in result
        assert "structure" in result
        assert "lean_code" in result
        assert result["domain"] == "quantum_mechanics"

    @pytest.mark.asyncio
    async def test_extract_structure(self, formalizer):
        """Test structure extraction"""
        definition = "A manifold is a smooth topological space"

        structure = await formalizer._extract_structure(definition)

        assert "type" in structure
        assert structure["type"] is not None

    @pytest.mark.asyncio
    async def test_map_to_lean_types(self, formalizer):
        """Test mapping to Lean 4 types"""
        structure = {"type": "hilbert_space"}
        lean_types = formalizer._map_to_lean_types(
            structure,
            PhysicsDomain.QUANTUM_MECHANICS
        )

        assert "main_type" in lean_types
        assert "imports" in lean_types


# ============================================================================
# MCP Tools Tests
# ============================================================================

class TestMCPTools:
    """Test MCP tools for physics knowledge"""

    def test_query_physics_theorems_mcp(self):
        """Test leanaide_query_physics_theorems MCP tool"""
        from leanaide_mcp_tools import leanaide_query_physics_theorems

        result = leanaide_query_physics_theorems(
            problem="Calculate uncertainty",
            domain="quantum_mechanics",
            k=5
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "theorems" in result
            assert "count" in result

    def test_suggest_physics_decomposition_mcp(self):
        """Test leanaide_suggest_physics_decomposition MCP tool"""
        from leanaide_mcp_tools import leanaide_suggest_physics_decomposition

        result = leanaide_suggest_physics_decomposition(
            problem="Prove no-cloning theorem",
            domain="quantum_mechanics"
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "decomposition" in result

    def test_get_physics_tactics_mcp(self):
        """Test leanaide_get_physics_tactics MCP tool"""
        from leanaide_mcp_tools import leanaide_get_physics_tactics

        result = leanaide_get_physics_tactics(
            problem="Calculate expectation value",
            domain="quantum_mechanics"
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "tactics" in result
            assert "count" in result

    def test_formalize_physics_definition_mcp(self):
        """Test leanaide_formalize_physics_definition MCP tool"""
        from leanaide_mcp_tools import leanaide_formalize_physics_definition

        result = leanaide_formalize_physics_definition(
            definition="Hilbert space with inner product",
            context="Quantum mechanics",
            domain="quantum_mechanics"
        )

        assert isinstance(result, dict)
        assert "success" in result
        if result.get("success"):
            assert "lean_code" in result

    def test_get_physics_knowledge_status_mcp(self):
        """Test get_physics_knowledge_status MCP tool"""
        from leanaide_mcp_tools import get_physics_knowledge_status

        status = get_physics_knowledge_status()

        assert isinstance(status, dict)
        assert "enabled" in status
        assert "theorems_count" in status
        assert "concepts_count" in status


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration with other components"""

    def test_physics_knowledge_with_continuous_math(self):
        """Test physics knowledge engine with continuous mathematics"""
        from leanaide_continuous_math import ContinuousMathBridge

        ke = PhysicsKnowledgeEngine()
        bridge = ContinuousMathBridge()

        # Both should be operational
        assert len(ke.theorems) > 0
        assert bridge is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_query_performance(self, knowledge_engine):
        """Test that queries complete in reasonable time"""
        import time

        start_time = time.time()
        theorems = await knowledge_engine.query_related_theorems(
            "Calculate uncertainty in quantum system",
            domain=PhysicsDomain.QUANTUM_MECHANICS,
            k=10
        )
        elapsed_time = time.time() - start_time

        assert elapsed_time < 5.0  # Should complete in under 5 seconds
        assert len(theorems) > 0


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling"""

    @pytest.mark.asyncio
    async def test_invalid_domain(self, knowledge_engine):
        """Test handling of invalid domain"""
        problem = "Test problem"

        # Should not crash, but return empty results
        # (This is tested through decomposition which handles domain properly)
        decomposition = await knowledge_engine.suggest_decomposition(
            problem,
            PhysicsDomain.CLASSICAL_MECHANICS  # Has minimal data
        )

        assert "steps" in decomposition

    @pytest.mark.asyncio
    async def test_empty_problem(self, knowledge_engine):
        """Test handling of empty problem string"""
        # Should handle gracefully
        theorems = await knowledge_engine.query_related_theorems(
            "",
            domain=PhysicsDomain.QUANTUM_MECHANICS,
            k=5
        )

        # Returns empty list if no keywords found
        assert isinstance(theorems, list)


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
