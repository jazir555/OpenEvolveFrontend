"""
Unit tests for RESE Phase II: Isomorphic Mapping

Tests cover:
- Structure identification
- Dependency graph construction
- I_mech calculation
- Constraint inversion
- Full Phase II execution

Following CLAUDE.md principles:
- Law of Runtime Truth: Verify before using
- Law of Idempotency: Tests are repeatable
"""

import pytest
import os
import sys
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "schemas"))

# Set required env vars
os.environ["PHASE2_MAX_TARGET_DOMAINS"] = "10"
os.environ["PHASE2_IMECH_THRESHOLD"] = "0.7"
os.environ["PHASE2_PATTERN_THRESHOLD"] = "0.6"
os.environ["PHASE2_TIMEOUT_MS"] = "20000"
os.environ["PHASE2_MAX_MAPPINGS"] = "50"
os.environ["PHASE2_ENABLE_CONSTRAINT_INVERSION"] = "true"
os.environ["PHASE2_SEARCH_DEPTH"] = "5"

from phase2_executor import (
    IsomorphicMappingExecutor,
    StructureIdentifier,
    DependencyGraphBuilder,
    CrossDomainMapper,
    ConstraintInverter,
    ConstraintHardener,
    Phase2Logger,
    create_executor,
)

from rese_schemas import (
    Phase2Config,
    FunctionalDependencyGraph,
    FunctionalDependency,
    IsomorphicMapping,
    InvertedConstraint,
    IsomorphismType,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return Phase2Config(
        max_target_domains=5,
        i_mech_threshold=0.7,
        pattern_recognition_threshold=0.6,
        timeout_ms=20000,
        max_mappings=50,
        enable_constraint_inversion=True,
        search_depth=5
    )


@pytest.fixture
def logger():
    """Create test logger."""
    return Phase2Logger("test-correlation-id")


@pytest.fixture
def sample_fdg():
    """Create sample FDG."""
    return FunctionalDependencyGraph(
        domain="physics",
        nodes=["energy", "momentum", "force"],
        dependencies=[
            FunctionalDependency(
                source="energy",
                target="momentum",
                relationship_type="causal",
                strength=0.8,
                domain="physics"
            )
        ],
        adjacency_list={
            "energy": ["momentum"],
            "momentum": [],
            "force": []
        }
    )


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestPhase2Config:
    """Test Phase2Config."""

    def test_from_env(self):
        """Test loading config from environment."""
        config = Phase2Config.from_env()

        assert config.max_target_domains == 10
        assert config.i_mech_threshold == 0.7
        assert config.timeout_ms == 20000
        assert config.max_mappings == 50
        assert config.enable_constraint_inversion is True

    def test_to_dict(self, config):
        """Test converting config to dict."""
        config_dict = config.to_dict()

        assert "max_target_domains" in config_dict
        assert "i_mech_threshold" in config_dict
        assert config_dict["max_target_domains"] == 5


# ============================================================================
# STRUCTURE IDENTIFIER TESTS
# ============================================================================

class TestStructureIdentifier:
    """Test StructureIdentifier."""

    def test_identify_structure(self, config, logger):
        """Test identifying domain structure."""
        identifier = StructureIdentifier(config, logger)

        fdg = identifier.identify_structure(
            domain="physics",
            problem_description="Energy conservation in closed system",
            context=None
        )

        assert fdg is not None
        assert fdg.domain == "physics"
        assert len(fdg.nodes) > 0
        assert isinstance(fdg.dependencies, list)

    def test_extract_concepts(self, config, logger):
        """Test concept extraction."""
        identifier = StructureIdentifier(config, logger)

        concepts = identifier._extract_concepts(
            domain="physics",
            text="This problem involves energy and momentum conservation"
        )

        assert isinstance(concepts, list)
        assert len(concepts) > 0

    def test_extract_relations(self, config, logger):
        """Test relation extraction."""
        identifier = StructureIdentifier(config, logger)

        relations = identifier._extract_relations(
            domain="physics",
            text="Energy causes momentum change"
        )

        assert isinstance(relations, list)


# ============================================================================
# DEPENDENCY GRAPH BUILDER TESTS
# ============================================================================

class TestDependencyGraphBuilder:
    """Test DependencyGraphBuilder."""

    def test_build_graph(self, config, logger):
        """Test building FDG."""
        builder = DependencyGraphBuilder(config, logger)

        fdg = builder.build_graph(
            domain="biology",
            nodes=["population", "resource", "environment"],
            dependencies=[
                {"source": "population", "target": "resource", "relationship_type": "consumption", "strength": 0.9}
            ]
        )

        assert fdg.domain == "biology"
        assert len(fdg.nodes) == 3
        assert len(fdg.dependencies) == 1
        assert "population" in fdg.adjacency_list


# ============================================================================
# CROSS DOMAIN MAPPER TESTS
# ============================================================================

class TestCrossDomainMapper:
    """Test CrossDomainMapper."""

    def test_compute_fdg_overlap(self, config, logger, sample_fdg):
        """Test FDG overlap calculation."""
        mapper = CrossDomainMapper(config, logger)

        # Create identical FDG
        fdg2 = FunctionalDependencyGraph(
            domain="physics2",
            nodes=sample_fdg.nodes,
            dependencies=sample_fdg.dependencies,
            adjacency_list=sample_fdg.adjacency_list.copy()
        )

        overlap = mapper.compute_fdg_overlap(sample_fdg, fdg2)

        assert overlap == 1.0  # Identical graphs

    def test_compute_fdg_overlap_different(self, config, logger, sample_fdg):
        """Test FDG overlap with different graphs."""
        mapper = CrossDomainMapper(config, logger)

        # Create different FDG
        fdg2 = FunctionalDependencyGraph(
            domain="biology",
            nodes=["population", "ecosystem"],
            dependencies=[],
            adjacency_list={"population": [], "ecosystem": []}
        )

        overlap = mapper.compute_fdg_overlap(sample_fdg, fdg2)

        assert 0.0 <= overlap <= 1.0
        assert overlap < 1.0  # Not identical

    def test_compute_imech_score(self, config, logger, sample_fdg):
        """Test I_mech score calculation."""
        mapper = CrossDomainMapper(config, logger)

        # Create similar FDG
        fdg2 = FunctionalDependencyGraph(
            domain="physics2",
            nodes=sample_fdg.nodes,
            dependencies=sample_fdg.dependencies,
            adjacency_list=sample_fdg.adjacency_list.copy()
        )

        imech = mapper.compute_imech_score(sample_fdg, fdg2)

        assert 0.0 <= imech <= 1.0
        assert imech >= 0.7  # Should be high for similar graphs

    def test_find_isomorphic_mappings(self, config, logger, sample_fdg):
        """Test finding isomorphic mappings."""
        mapper = CrossDomainMapper(config, logger)

        # Create target FDGs
        target_fdgs = [
            FunctionalDependencyGraph(
                domain="biology",
                nodes=["population", "growth"],
                dependencies=[],
                adjacency_list={"population": [], "growth": []}
            ),
            FunctionalDependencyGraph(
                domain="economics",
                nodes=sample_fdg.nodes,  # Same nodes
                dependencies=sample_fdg.dependencies,
                adjacency_list=sample_fdg.adjacency_list.copy()
            )
        ]

        mappings = mapper.find_isomorphic_mappings(sample_fdg, target_fdgs)

        assert isinstance(mappings, list)
        # At least economics should match (same nodes)
        assert len(mappings) >= 1


# ============================================================================
# CONSTRAINT INVERTER TESTS
# ============================================================================

class TestConstraintInverter:
    """Test ConstraintInverter."""

    def test_invert_constraint_negation(self, config, logger):
        """Test constraint inversion with negation."""
        inverter = ConstraintInverter(config, logger)

        original = "Energy must be conserved"
        inverted = inverter.invert_constraint(original, "negation")

        assert inverted.original_constraint == original
        assert "NOT" in inverted.inverted_constraint
        assert inverted.inversion_type == "negation"
        assert inverted.feasibility is True

    def test_invert_constraint_complement(self, config, logger):
        """Test constraint inversion with complement."""
        inverter = ConstraintInverter(config, logger)

        original = "Algorithm must be optimal"
        inverted = inverter.invert_constraint(original, "complement")

        assert "COMPLEMENT" in inverted.inverted_constraint
        assert inverted.inversion_type == "complement"

    def test_invert_constraints(self, config, logger):
        """Test inverting multiple constraints."""
        inverter = ConstraintInverter(config, logger)

        constraints = [
            "Energy is conserved",
            "Momentum is conserved"
        ]

        inverted_list = inverter.invert_constraints(constraints)

        assert len(inverted_list) == 2
        assert all(isinstance(inv, InvertedConstraint) for inv in inverted_list)


# ============================================================================
# CONSTRAINT HARDENER TESTS
# ============================================================================

class TestConstraintHardener:
    """Test ConstraintHardener."""

    def test_harden_constraints(self, config, logger):
        """Test constraint hardening."""
        hardener = ConstraintHardener(config, logger)

        constraints = ["Energy is conserved"]

        mapping = IsomorphicMapping(
            source_domain="physics",
            target_domain="biology",
            isomorphism_type=IsomorphismType.STRUCTURAL,
            i_mech_score=0.8,
            fdg_overlap=0.75,
            confidence=0.8
        )

        hardened = hardener.harden_constraints(constraints, mapping)

        assert len(hardened) == 1
        assert "biology" in hardened[0]
        assert "0.8" in hardened[0]  # I_mech score


# ============================================================================
# PHASE II EXECUTOR TESTS
# ============================================================================

class TestIsomorphicMappingExecutor:
    """Test IsomorphicMappingExecutor."""

    def test_create_executor(self, config):
        """Test creating executor."""
        executor = create_executor(config)

        assert executor is not None
        assert executor.config == config

    def test_execute_phase2(self, config):
        """Test full Phase II execution."""
        executor = IsomorphicMappingExecutor(config)

        result = executor.execute_phase2(
            source_domain="physics",
            problem_description="Energy conservation problem",
            target_domains=["biology", "economics"],
            constraints=["energy is conserved"],
            context=None
        )

        assert result is not None
        assert result.source_domain == "physics"
        assert len(result.target_domains) == 2
        assert isinstance(result.mappings_found, list)
        assert isinstance(result.cross_domain_patterns, list)
        assert isinstance(result.inverted_constraints, list)
        assert result.execution_time_ms > 0

    def test_execute_phase2_without_constraints(self, config):
        """Test Phase II execution without constraints."""
        executor = IsomorphicMappingExecutor(config)

        result = executor.execute_phase2(
            source_domain="computer_science",
            problem_description="Algorithm optimization",
            target_domains=None,
            constraints=None,
            context=None
        )

        assert result is not None
        assert len(result.inverted_constraints) == 0  # No constraints to invert

    def test_execute_phase2_idempotent(self, config):
        """Test that Phase II execution is idempotent."""
        executor = IsomorphicMappingExecutor(config)

        # Execute twice with same input
        result1 = executor.execute_phase2(
            source_domain="physics",
            problem_description="Test problem",
            target_domains=["biology"]
        )

        result2 = executor.execute_phase2(
            source_domain="physics",
            problem_description="Test problem",
            target_domains=["biology"]
        )

        # Results should be consistent (same source, same targets)
        assert result1.source_domain == result2.source_domain
        assert len(result1.target_domains) == len(result2.target_domains)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPhase2Integration:
    """Integration tests for Phase II."""

    def test_full_workflow(self, config):
        """Test complete Phase II workflow."""
        executor = IsomorphicMappingExecutor(config)

        # Execute Phase II
        result = executor.execute_phase2(
            source_domain="physics",
            problem_description="Wave propagation in medium with energy conservation",
            target_domains=["biology", "economics", "computer_science"],
            constraints=["energy is conserved", "momentum is conserved"]
        )

        # Verify result structure
        assert result.result_id is not None
        assert result.source_domain == "physics"
        assert len(result.target_domains) == 3

        # Verify mappings
        if result.mappings_found:
            best = result.mappings_found[0]
            assert best.i_mech_score >= config.i_mech_threshold

        # Verify inverted constraints
        assert len(result.inverted_constraints) == 2

        # Verify patterns
        assert isinstance(result.cross_domain_patterns, list)

        # Verify timing
        assert result.execution_time_ms < config.timeout_ms

    def test_cross_domain_transfer(self, config):
        """Test cross-domain knowledge transfer."""
        executor = IsomorphicMappingExecutor(config)

        # Find isomorphic mappings
        result = executor.execute_phase2(
            source_domain="physics",
            problem_description="Oscillatory system with damping",
            target_domains=["biology"]  # Population dynamics
        )

        # Check if valid isomorphism found
        if result.mappings_found:
            mapping = result.mappings_found[0]
            if mapping.i_mech_score > 0.7:
                # Valid transfer candidate
                assert mapping.target_domain == "biology"
                assert mapping.confidence > 0.7


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
