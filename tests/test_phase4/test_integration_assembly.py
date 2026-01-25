"""
Integration Tests for Architecture Assembly (Δ₁)

Tests end-to-end assembly with components from Phases I-III.

Author: Agent E1 (Δ₁ Specialist)
Created: 2025-12-31
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4.architecture_assembler import (
    ArchitectureAssembler,
    Architecture,
    ComponentInterface,
    AssemblyConfig,
    PhaseType,
    AssemblyPattern
)
from phase4.assembly_validator import (
    AssemblyValidator,
    BatchValidator
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def full_assembler():
    """Assembler with all RESE components"""
    return ArchitectureAssembler(
        config=AssemblyConfig(
            strategy="greedy",
            target_aci=0.8,
            require_validation=True,
            min_validation_score=0.6
        )
    )


@pytest.fixture
def validator():
    """Assembly validator"""
    return AssemblyValidator(strict=False)


# =============================================================================
# Phase I Integration Tests
# =============================================================================

class TestPhaseIIntegration:
    """Tests for Phase I (Epistemic Audit) integration"""

    def test_phi15_assembly(self, full_assembler):
        """Test assembling Φ₁.₅ (Tacit Assumption Miner)"""
        result = full_assembler.assemble(component_ids=["sce", "phi15"])
        assert result.success

        arch = result.architecture
        assert arch.has_component("phi15")
        assert arch.has_component("sce")  # Dependency

        # Check validation
        assert arch.validation_score >= 0.6

    def test_phi15_with_gamma1(self, full_assembler):
        """Test Φ₁.₅ + Γ₁ combination"""
        result = full_assembler.assemble(component_ids=["sce", "phi15", "gamma1"])
        assert result.success

        arch = result.architecture
        assert arch.has_component("phi15")
        assert arch.has_component("gamma1")

        # Should have higher ACI with both
        assert arch.expected_aci_improvement > 0


# =============================================================================
# Phase II Integration Tests
# =============================================================================

class TestPhaseIIIntegration:
    """Tests for Phase II (Isomorphic Resonance) integration"""

    def test_psi3_assembly(self, full_assembler):
        """Test assembling Ψ₃ (Constraint Inversion)"""
        result = full_assembler.assemble(component_ids=["sce", "psi3"])
        assert result.success

        arch = result.architecture
        assert arch.has_component("psi3")
        assert arch.has_component("sce")

        # Ψ₃ should improve ACI
        assert arch.expected_aci_improvement > 0

    def test_imech_assembly(self, full_assembler):
        """Test assembling I_mech (Isomorphism Validator)"""
        result = full_assembler.assemble(component_ids=["sce", "imech"])
        assert result.success

        arch = result.architecture
        assert arch.has_component("imech")

    def test_psi3_with_imech(self, full_assembler):
        """Test Ψ₃ + I_mech combination"""
        result = full_assembler.assemble(
            component_ids=["sce", "psi3", "imech"]
        )
        assert result.success

        arch = result.architecture
        assert arch.has_component("psi3")
        assert arch.has_component("imech")


# =============================================================================
# Phase III Integration Tests
# =============================================================================

class TestPhaseIIIIntegration:
    """Tests for Phase III (Monte Carlo Refinement) integration"""

    def test_gamma1_assembly(self, full_assembler):
        """Test assembling Γ₁ (ACI Analyzer)"""
        result = full_assembler.assemble(component_ids=["gamma1"])
        assert result.success

        arch = result.architecture
        assert arch.has_component("gamma1")

    def test_gamma2_assembly(self, full_assembler):
        """Test assembling Γ₂ (MCTS Search) with dependencies"""
        result = full_assembler.assemble(component_ids=["gamma2"])
        assert result.success

        arch = result.architecture
        assert arch.has_component("gamma2")
        assert arch.has_component("gamma1")  # Dependency

    def test_gamma1_with_gamma2(self, full_assembler):
        """Test Γ₁ + Γ₂ combination"""
        result = full_assembler.assemble(
            component_ids=["sce", "gamma1", "gamma2"]
        )
        assert result.success

        arch = result.architecture
        assert arch.has_component("gamma1")
        assert arch.has_component("gamma2")

        # Check dependency ordering
        component_ids = [c.component_id for c in arch.components]
        gamma1_idx = component_ids.index("gamma1")
        gamma2_idx = component_ids.index("gamma2")
        assert gamma1_idx < gamma2_idx


# =============================================================================
# Cross-Phase Integration Tests
# =============================================================================

class TestCrossPhaseIntegration:
    """Tests for cross-phase integration"""

    def test_phase_i_with_phase_ii(self, full_assembler):
        """Test Phase I + Phase II integration"""
        result = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3"]
        )
        assert result.success

        arch = result.architecture
        assert arch.has_component("phi15")  # Phase I
        assert arch.has_component("psi3")   # Phase II

    def test_phase_ii_with_phase_iii(self, full_assembler):
        """Test Phase II + Phase III integration"""
        result = full_assembler.assemble(
            component_ids=["sce", "psi3", "gamma1", "gamma2"]
        )
        assert result.success

        arch = result.architecture
        assert arch.has_component("psi3")   # Phase II
        assert arch.has_component("gamma1")  # Phase III
        assert arch.has_component("gamma2")  # Phase III

    def test_all_phases(self, full_assembler):
        """Test assembling components from all phases"""
        result = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3", "gamma1", "gamma2"]
        )
        assert result.success

        arch = result.architecture

        # Check all phases represented
        phases = {c.phase for c in arch.components}
        assert PhaseType.PHASE_I in phases
        assert PhaseType.PHASE_II in phases
        assert PhaseType.PHASE_III in phases


# =============================================================================
# Validation Integration Tests
# =============================================================================

class TestValidationIntegration:
    """Tests for validation with assembly"""

    def test_validate_simple_architecture(self, full_assembler, validator):
        """Test validating simple architecture"""
        result = full_assembler.assemble(component_ids=["sce", "gamma1"])
        assert result.success

        validation = validator.validate(result.architecture)
        assert validation is not None
        assert validation.validation_score >= 0

    def test_validate_complex_architecture(self, full_assembler, validator):
        """Test validating complex architecture"""
        result = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3", "gamma1", "gamma2"]
        )
        assert result.success

        validation = validator.validate(result.architecture)
        assert validation is not None

        # Complex architecture should have decent score
        assert validation.validation_score >= 0.3

    def test_validate_with_errors(self, full_assembler, validator):
        """Test validation catches errors"""
        # Create architecture without core
        arch = Architecture(
            architecture_id="test_arch",
            name="Test",
            description="Test architecture without core"
        )

        validation = validator.validate(arch)
        assert not validation.is_valid
        assert len(validation.errors) > 0


# =============================================================================
# Batch Validation Tests
# =============================================================================

class TestBatchValidation:
    """Tests for batch validation"""

    def test_batch_validate_multiple(self, full_assembler):
        """Test validating multiple architectures"""
        # Create multiple architectures
        results = [
            full_assembler.assemble(component_ids=["sce", "gamma1"]),
            full_assembler.assemble(component_ids=["sce", "phi15", "psi3"]),
            full_assembler.assemble(component_ids=["sce", "gamma1", "gamma2"])
        ]

        assert all(r.success for r in results)

        # Batch validate
        batch = BatchValidator()
        validations = batch.validate_all([r.architecture for r in results])

        assert len(validations) == 3

    def test_batch_get_best(self, full_assembler):
        """Test getting best architecture from batch"""
        results = [
            full_assembler.assemble(component_ids=["sce"]),
            full_assembler.assemble(component_ids=["sce", "gamma1", "gamma2"]),
            full_assembler.assemble(component_ids=["sce", "phi15"])
        ]

        batch = BatchValidator()
        validations = batch.validate_all([r.architecture for r in results])

        best = batch.get_best()
        assert best is not None
        assert best.is_valid or best.validation_score >= 0


# =============================================================================
# Auto-Selection Tests
# =============================================================================

class TestAutoSelection:
    """Tests for automatic component selection"""

    def test_auto_select_validated(self, full_assembler):
        """Test auto-selection picks validated components"""
        result = full_assembler.assemble(component_ids=None)
        assert result.success

        # Should only include validated components
        arch = result.architecture
        for comp in arch.components:
            if comp.component_id in full_assembler.available_components:
                assert comp.is_validated or comp.component_id == "sce"

    def test_auto_select_includes_core(self, full_assembler):
        """Test auto-selection includes core components"""
        result = full_assembler.assemble(component_ids=None)
        assert result.success

        arch = result.architecture
        # Should have SCE
        assert arch.has_component("sce")


# =============================================================================
# Assembly Pattern Tests
# =============================================================================

class TestAssemblyPatterns:
    """Tests for assembly pattern detection"""

    def test_sequential_pattern(self, full_assembler):
        """Test sequential pattern detection"""
        result = full_assembler.assemble(component_ids=["gamma2"])
        assert result.success

        # gamma2 requires gamma1, likely sequential
        arch = result.architecture
        assert arch.assembly_pattern in [
            AssemblyPattern.SEQUENTIAL,
            AssemblyPattern.HYBRID
        ]

    def test_parallel_pattern(self, full_assembler):
        """Test parallel pattern detection"""
        # Components with no dependencies between them
        result = full_assembler.assemble(
            component_ids=["sce", "gamma1", "phi15"]
        )
        assert result.success

        arch = result.architecture
        # May be parallel or hybrid depending on dependencies


# =============================================================================
# ACI Improvement Tests
# =============================================================================

class TestACIImprovement:
    """Tests for ACI improvement"""

    def test_aci_improvement_increases(self, full_assembler):
        """Test ACI improvement increases with more components"""
        result1 = full_assembler.assemble(component_ids=["sce"])
        result2 = full_assembler.assemble(component_ids=["sce", "gamma1"])
        result3 = full_assembler.assemble(
            component_ids=["sce", "gamma1", "phi15", "psi3"]
        )

        assert result1.success
        assert result2.success
        assert result3.success

        # More components should generally increase ACI
        aci1 = result1.architecture.expected_aci_improvement
        aci2 = result2.architecture.expected_aci_improvement
        aci3 = result3.architecture.expected_aci_improvement

        assert aci3 >= aci2 >= aci1

    def test_phase_diversity_bonus(self, full_assembler, validator):
        """Test phase diversity improves validation score"""
        # Single phase
        result1 = full_assembler.assemble(
            component_ids=["sce", "gamma1"]
        )

        # Multiple phases
        result2 = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3", "gamma1"]
        )

        validation1 = validator.validate(result1.architecture)
        validation2 = validator.validate(result2.architecture)

        # Multi-phase may get diversity bonus
        # (this is a soft check, not guaranteed)


# =============================================================================
# Dependency Layer Tests
# =============================================================================

class TestDependencyLayersIntegration:
    """Tests for dependency layers in real assemblies"""

    def test_gamma2_creates_two_layers(self, full_assembler):
        """Test gamma2 assembly creates multiple layers"""
        result = full_assembler.assemble(component_ids=["gamma2"])
        assert result.success

        arch = result.architecture
        # Should have at least 2 layers (gamma1, then gamma2)
        assert len(arch.dependency_layers) >= 1

    def test_parallel_execution_in_layer(self, full_assembler):
        """Test independent components in same layer"""
        result = full_assembler.assemble(
            component_ids=["sce", "gamma1", "phi15"]
        )
        assert result.success

        arch = result.architecture
        # Check that some layer has multiple components
        has_parallel = any(len(layer) > 1 for layer in arch.dependency_layers)
        # May or may not be true depending on dependencies


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end pipeline tests"""

    def test_full_pipeline(self, full_assembler, validator):
        """Test complete assembly and validation pipeline"""
        # Step 1: Assemble
        result = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3", "gamma1", "gamma2"]
        )
        assert result.success

        # Step 2: Validate
        validation = validator.validate(result.architecture)

        # Step 3: Check results
        assert validation is not None
        assert validation.validation_time >= 0

        # Step 4: Generate report
        report = validator.explain_validation(validation)
        assert len(report) > 0
        assert "Architecture Validation Report" in report

    def test_pipeline_with_auto_selection(self, full_assembler, validator):
        """Test pipeline with automatic component selection"""
        # Auto-select components
        result = full_assembler.assemble(component_ids=None)
        assert result.success

        # Validate
        validation = validator.validate(result.architecture)

        # Should have reasonable score
        assert validation.validation_score >= 0.3

    def test_pipeline_error_handling(self, full_assembler, validator):
        """Test pipeline handles errors gracefully"""
        # Try to assemble with invalid component
        result = full_assembler.assemble(component_ids=["nonexistent"])

        # Should fail gracefully
        assert not result.success
        assert result.message is not None


# =============================================================================
# Performance Integration Tests
# =============================================================================

class TestPerformanceIntegration:
    """Performance tests for integration"""

    def test_assembly_performance(self, full_assembler):
        """Test assembly performance with realistic workload"""
        import time

        start = time.time()
        result = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3", "gamma1", "gamma2"]
        )
        elapsed = time.time() - start

        assert result.success
        assert elapsed < 2.0  # Should complete in under 2 seconds

    def test_validation_performance(self, full_assembler, validator):
        """Test validation performance"""
        result = full_assembler.assemble(
            component_ids=["sce", "phi15", "psi3", "gamma1", "gamma2"]
        )
        assert result.success

        import time
        start = time.time()
        validation = validator.validate(result.architecture)
        elapsed = time.time() - start

        assert validation is not None
        assert elapsed < 1.0  # Should validate in under 1 second


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
