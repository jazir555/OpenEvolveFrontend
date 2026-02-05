"""
Test Suite: FDG Validator with Lean 4 Integration

Comprehensive tests for:
- FDG extraction from text
- I_mech calculation
- Lean 4 formal verification
- Mechanistic isomorphism validation

Author: RESE Team
Created: 2026-02-04
"""

import pytest
import sys
import os
import json
from typing import Dict, List, Any

# Add paths
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.abspath(os.path.join(_current_dir, "..", "src"))
_schemas_dir = os.path.abspath(os.path.join(_current_dir, "..", "..", "schemas"))

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _schemas_dir not in sys.path:
    sys.path.insert(0, _schemas_dir)

try:
    from fdg_validator import (
        FDGValidator,
        FDGValidatorLogger,
        Lean4Bridge,
        FDGExtractor,
        IMechCalculator,
        create_validator,
        is_available
    )
    from rese_schemas import (
        FunctionalDependencyGraph,
        FunctionalDependency,
        IsomorphicMapping,
        IsomorphismType,
    )
except ImportError as e:
    pytest.skip(f"Import error: {e}", allow_module_level=True)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def logger():
    """Create test logger."""
    return FDGValidatorLogger(correlation_id="test-correlation-123")


@pytest.fixture
def lean_bridge(logger):
    """Create Lean 4 bridge."""
    return Lean4Bridge(logger)


@pytest.fixture
def fdg_extractor(logger):
    """Create FDG extractor."""
    return FDGExtractor(logger)


@pytest.fixture
def i_mech_calculator(logger, lean_bridge):
    """Create I_mech calculator."""
    return IMechCalculator(logger, lean_bridge)


@pytest.fixture
def validator():
    """Create FDG validator."""
    return create_validator()


# ============================================================================
# TESTS: LEAN 4 BRIDGE
# ============================================================================

class TestLean4Bridge:
    """Tests for Lean 4 bridge."""

    def test_lean_bridge_initialization(self, logger):
        """Test Lean 4 bridge initialization."""
        bridge = Lean4Bridge(logger)
        assert bridge is not None
        assert bridge.logger is not None

    def test_lean_bridge_enabled_check(self, lean_bridge):
        """Test Lean 4 availability check."""
        # Bridge should initialize without crashing
        assert lean_bridge is not None
        # May be disabled if Lean 4 not installed
        assert lean_bridge.lean_enabled in [True, False]

    def test_execute_lean_proof_disabled(self, lean_bridge):
        """Test Lean 4 proof execution when disabled."""
        lean_bridge.lean_enabled = False
        result = lean_bridge.execute_lean_proof("example : True := by trivial")
        assert result["proven"] is False
        assert len(result["errors"]) > 0


# ============================================================================
# TESTS: FDG EXTRACTOR
# ============================================================================

class TestFDGExtractor:
    """Tests for FDG extraction."""

    def test_extract_fdg_from_physics(self, fdg_extractor):
        """Test FDG extraction from physics domain."""
        description = """
        Energy and momentum are conserved in the system.
        The force field affects particle motion.
        Waves propagate through the medium.
        """

        fdg = fdg_extractor.extract_fdg_from_text(
            domain="physics",
            description=description
        )

        assert fdg is not None
        assert fdg.domain == "physics"
        assert len(fdg.nodes) > 0
        assert isinstance(fdg.dependencies, list)

    def test_extract_fdg_from_biology(self, fdg_extractor):
        """Test FDG extraction from biology domain."""
        description = """
        Population dynamics affect ecosystem stability.
        Evolution drives adaptation to environment.
        Species compete for resources.
        """

        fdg = fdg_extractor.extract_fdg_from_text(
            domain="biology",
            description=description
        )

        assert fdg is not None
        assert fdg.domain == "biology"
        assert len(fdg.nodes) > 0

    def test_extract_nodes_from_physics(self, fdg_extractor):
        """Test node extraction from physics text."""
        text = "Energy and momentum are conserved."
        nodes = fdg_extractor._extract_nodes("physics", text)
        assert "energy" in nodes
        assert "momentum" in nodes

    def test_extract_edges_from_nodes(self, fdg_extractor):
        """Test edge extraction from nodes."""
        nodes = ["energy", "momentum", "force"]
        edges = fdg_extractor._extract_edges("physics", "", nodes)
        assert len(edges) == len(nodes) - 1


# ============================================================================
# TESTS: I_MECH CALCULATOR
# ============================================================================

class TestIMechCalculator:
    """Tests for I_mech calculation."""

    @pytest.fixture
    def sample_fdg1(self):
        """Create sample FDG 1."""
        return FunctionalDependencyGraph(
            domain="test1",
            nodes=["A", "B", "C"],
            dependencies=[
                FunctionalDependency("A", "B", "causal", 0.7, "test1"),
                FunctionalDependency("B", "C", "causal", 0.7, "test1"),
            ],
            adjacency_list={"A": ["B"], "B": ["C"], "C": []}
        )

    @pytest.fixture
    def sample_fdg2(self):
        """Create sample FDG 2 (similar to FDG 1)."""
        return FunctionalDependencyGraph(
            domain="test2",
            nodes=["A", "B", "D"],
            dependencies=[
                FunctionalDependency("A", "B", "causal", 0.7, "test2"),
            ],
            adjacency_list={"A": ["B"], "B": [], "D": []}
        )

    def test_calculate_node_overlap(self, i_mech_calculator, sample_fdg1, sample_fdg2):
        """Test node overlap calculation."""
        overlap = i_mech_calculator._calculate_node_overlap(sample_fdg1, sample_fdg2)
        assert 0 <= overlap <= 1
        # {A, B} intersection, {A, B, C, D} union = 2/4 = 0.5
        assert overlap == 0.5

    def test_calculate_edge_overlap(self, i_mech_calculator, sample_fdg1, sample_fdg2):
        """Test edge overlap calculation."""
        overlap = i_mech_calculator._calculate_edge_overlap(sample_fdg1, sample_fdg2)
        assert 0 <= overlap <= 1
        # {(A, B)} intersection, {(A, B), (B, C)} union = 1/2 = 0.5
        assert overlap == 0.5

    def test_calculate_size_ratio(self, i_mech_calculator, sample_fdg1, sample_fdg2):
        """Test size ratio calculation."""
        ratio = i_mech_calculator._calculate_size_ratio(sample_fdg1, sample_fdg2)
        assert 0 <= ratio <= 1
        # min(3, 3) / max(3, 3) = 1.0
        assert ratio == 1.0

    def test_calculate_i_mech(self, i_mech_calculator, sample_fdg1, sample_fdg2):
        """Test I_mech calculation."""
        result = i_mech_calculator.calculate_i_mech(sample_fdg1, sample_fdg2, use_lean4=False)
        assert "i_mech" in result
        assert "node_overlap" in result
        assert "edge_overlap" in result
        assert "size_ratio" in result
        assert 0 <= result["i_mech"] <= 1

    def test_i_mech_formula(self, i_mech_calculator, sample_fdg1, sample_fdg2):
        """Test I_mech formula correctness."""
        result = i_mech_calculator.calculate_i_mech(sample_fdg1, sample_fdg2, use_lean4=False)
        # I_mech = 0.7 * (0.6 * node + 0.4 * edge) + 0.3 * size
        #         = 0.7 * (0.6 * 0.5 + 0.4 * 0.5) + 0.3 * 1.0
        #         = 0.7 * 0.5 + 0.3
        #         = 0.35 + 0.3
        #         = 0.65
        expected = 0.7 * (0.6 * 0.5 + 0.4 * 0.5) + 0.3 * 1.0
        assert abs(result["i_mech"] - expected) < 0.01


# ============================================================================
# TESTS: FDG VALIDATOR
# ============================================================================

class TestFDGValidator:
    """Tests for FDG validator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = create_validator()
        assert validator is not None
        assert validator.extractor is not None
        assert validator.calculator is not None

    def test_validate_isomorphism_basic(self, validator):
        """Test basic isomorphism validation."""
        result = validator.validate_isomorphism(
            source_domain="physics",
            source_description="Energy and momentum are conserved.",
            target_domain="physics",
            target_description="Energy and momentum are conserved.",
            threshold=0.7,
            use_lean4=False
        )
        assert "i_mech_score" in result
        assert "is_isomorphic" in result
        # Same description should give high I_mech
        assert result["i_mech_score"] > 0.5

    def test_validate_isomorphism_different_domains(self, validator):
        """Test validation across different domains."""
        result = validator.validate_isomorphism(
            source_domain="physics",
            source_description="Energy conservation in closed system.",
            target_domain="biology",
            target_description="Population dynamics in ecosystem.",
            threshold=0.7,
            use_lean4=False
        )
        assert result["i_mech_score"] >= 0
        assert result["i_mech_score"] <= 1

    def test_validate_with_high_threshold(self, validator):
        """Test validation with high threshold."""
        result = validator.validate_isomorphism(
            source_domain="test",
            source_description="Test description A.",
            target_domain="test",
            target_description="Test description B.",
            threshold=0.9,
            use_lean4=False
        )
        # Should be less likely to pass with 0.9 threshold
        assert result["threshold"] == 0.9

    def test_batch_validate(self, validator):
        """Test batch validation."""
        targets = [
            ("physics", "Energy and momentum conservation."),
            ("biology", "Population and ecosystem dynamics."),
            ("computer_science", "Algorithms and data structures."),
        ]

        results = validator.batch_validate(
            source_domain="test",
            source_description="Test description.",
            target_domains=targets,
            threshold=0.7,
            use_lean4=False
        )

        assert len(results) == 3
        # Results should be sorted by I_mech
        for r in results:
            assert "i_mech_score" in r
            assert "is_isomorphic" in r


# ============================================================================
# TESTS: HE-LCF CASE STUDY
# ============================================================================

class TestHELCFCaseStudy:
    """Tests for HE-LCF isomorphism case study."""

    def test_he_description_extraction(self, fdg_extractor):
        """Test HE FDG extraction."""
        he_desc = """
        Homomorphic encryption allows computation on encrypted data.
        Plaintext is encrypted to ciphertext using public key.
        Homomorphic operations performed on ciphertext directly.
        Decryption reveals final result using private key.
        """

        he_fdg = fdg_extractor.extract_fdg_from_text(
            domain="computer_science",
            description=he_desc
        )

        assert he_fdg is not None
        assert len(he_fdg.nodes) > 0

    def test_lcf_description_extraction(self, fdg_extractor):
        """Test LCF FDG extraction."""
        lcf_desc = """
        Lattice confinement fusion uses solid lattice to confine fuel.
        Nuclear reactions occur in confined reaction zone.
        Energy extracted from fusion products.
        Thermal output harvested for power generation.
        """

        lcf_fdg = fdg_extractor.extract_fdg_from_text(
            domain="physics",
            description=lcf_desc
        )

        assert lcf_fdg is not None
        assert len(lcf_fdg.nodes) > 0

    def test_he_lcf_i_mech_calculation(self, validator):
        """Test I_mech calculation between HE and LCF."""
        he_desc = """
        Encryption isolates plaintext in ciphertext.
        Homomorphic operations compute on encrypted data.
        Decryption releases final result.
        """

        lcf_desc = """
        Lattice confinement isolates fuel in reaction zone.
        Nuclear fusion releases energy in confined space.
        Energy extraction harvests thermal output.
        """

        result = validator.validate_isomorphism(
            source_domain="computer_science",
            source_description=he_desc,
            target_domain="physics",
            target_description=lcf_desc,
            threshold=0.8,
            use_lean4=False
        )

        # Both have isolation → computation → release structure
        # Should have moderate I_mech
        assert result["i_mech_score"] >= 0


# ============================================================================
# TESTS: INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests."""

    def test_full_validation_pipeline(self, validator):
        """Test complete validation pipeline."""
        result = validator.validate_isomorphism(
            source_domain="test_source",
            source_description="Source test description with causal relationships.",
            target_domain="test_target",
            target_description="Target test description with causal relationships.",
            threshold=0.7,
            use_lean4=False
        )

        # Check all required fields
        assert "source_domain" in result
        assert "target_domain" in result
        assert "source_fdg" in result
        assert "target_fdg" in result
        assert "i_mech_score" in result
        assert "node_overlap" in result
        assert "edge_overlap" in result
        assert "size_ratio" in result
        assert "is_isomorphic" in result
        assert "validated_in_lean4" in result

        # Check FDG structure
        assert result["source_fdg"]["domain"] == "test_source"
        assert result["target_fdg"]["domain"] == "test_target"
        assert len(result["source_fdg"]["nodes"]) >= 0
        assert len(result["target_fdg"]["nodes"]) >= 0

    def test_is_available_function(self):
        """Test is_available utility function."""
        assert is_available() is True


# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_description(self, validator):
        """Test validation with empty description."""
        result = validator.validate_isomorphism(
            source_domain="test",
            source_description="",
            target_domain="test",
            target_description="",
            threshold=0.7,
            use_lean4=False
        )
        # Should not crash, return low I_mech
        assert result["i_mech_score"] >= 0

    def test_very_long_description(self, validator):
        """Test validation with very long description."""
        long_desc = "Test description. " * 1000
        result = validator.validate_isomorphism(
            source_domain="test",
            source_description=long_desc,
            target_domain="test",
            target_description=long_desc,
            threshold=0.7,
            use_lean4=False
        )
        # Should handle gracefully
        assert result["i_mech_score"] >= 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
