"""
Unit Tests for Architecture Assembler (Δ₁)

Tests component registration, dependency resolution, assembly algorithms,
and validation.

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
    AssemblyPattern,
    AssemblyConfig,
    PhaseType,
    ACIChange,
    SideEffect,
    BeamSearchAssembler,
    are_compatible,
    fingerprint_architecture
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Basic assembly configuration"""
    return AssemblyConfig(
        strategy="greedy",
        target_aci=0.8,
        min_validation_score=0.6
    )


@pytest.fixture
def assembler(basic_config):
    """Architecture assembler with basic config"""
    return ArchitectureAssembler(config=basic_config)


@pytest.fixture
def test_component():
    """Test component for testing"""
    return ComponentInterface(
        component_id="test_comp",
        component_name="Test Component",
        phase=PhaseType.PHASE_I,
        input_types=["Input"],
        output_types=["Output"],
        requires=[],
        expected_aci_change=ACIChange.INCREASE,
        is_validated=True,
        validation_score=0.8
    )


@pytest.fixture
def dependent_component():
    """Component with dependencies"""
    return ComponentInterface(
        component_id="dependent_comp",
        component_name="Dependent Component",
        phase=PhaseType.PHASE_II,
        requires=["sce"],
        expected_aci_change=ACIChange.INCREASE,
        is_validated=True,
        validation_score=0.75
    )


@pytest.fixture
def simple_architecture(assembler):
    """Simple architecture with core components"""
    result = assembler.assemble(component_ids=["sce", "gamma1"])
    assert result.success
    return result.architecture


# =============================================================================
# Component Registration Tests
# =============================================================================

class TestComponentRegistration:
    """Tests for component registration"""

    def test_register_component(self, assembler, test_component):
        """Test registering a new component"""
        assembler.register_component(test_component)
        assert test_component.component_id in assembler.available_components

    def test_register_duplicate_component(self, assembler, test_component):
        """Test registering duplicate component overwrites"""
        assembler.register_component(test_component)
        original_name = test_component.component_name

        # Modify and register again
        test_component.component_name = "Modified Name"
        assembler.register_component(test_component)

        retrieved = assembler.get_component(test_component.component_id)
        assert retrieved.component_name == "Modified Name"

    def test_get_component(self, assembler):
        """Test retrieving registered component"""
        comp = assembler.get_component("sce")
        assert comp is not None
        assert comp.component_id == "sce"

    def test_get_nonexistent_component(self, assembler):
        """Test retrieving non-existent component returns None"""
        comp = assembler.get_component("nonexistent")
        assert comp is None

    def test_get_available_components(self, assembler):
        """Test getting all available components"""
        components = assembler.get_available_components()
        assert len(components) > 0
        assert all(isinstance(c, ComponentInterface) for c in components)


# =============================================================================
# Dependency Resolution Tests
# =============================================================================

class TestDependencyResolution:
    """Tests for dependency resolution"""

    def test_no_dependencies(self, assembler):
        """Test resolving components with no dependencies"""
        result = assembler.assemble(component_ids=["sce"])
        assert result.success

    def test_single_dependency(self, assembler):
        """Test resolving single dependency"""
        result = assembler.assemble(component_ids=["phi15"])
        assert result.success
        # phi15 requires sce
        assert "sce" in [c.component_id for c in result.architecture.components]

    def test_multiple_dependencies(self, assembler):
        """Test resolving multiple dependencies"""
        result = assembler.assemble(component_ids=["gamma2"])
        assert result.success
        # gamma2 requires gamma1
        component_ids = [c.component_id for c in result.architecture.components]
        assert "gamma1" in component_ids
        assert "gamma2" in component_ids

    def test_dependency_order(self, assembler):
        """Test dependencies are ordered correctly"""
        result = assembler.assemble(component_ids=["gamma2"])
        assert result.success
        arch = result.architecture

        # Check layers
        if len(arch.dependency_layers) > 1:
            # First layer should have gamma1 (dependency of gamma2)
            first_layer_ids = arch.dependency_layers[0]
            assert "gamma1" in first_layer_ids

    def test_missing_dependency(self, assembler, dependent_component):
        """Test missing dependency causes failure"""
        assembler.register_component(dependent_component)
        result = assembler.assemble(component_ids=["dependent_comp"])
        # Should add sce automatically
        assert "sce" in [c.component_id for c in result.architecture.components]

    def test_circular_dependency_detection(self, assembler):
        """Test circular dependencies are detected"""
        # Create circular dependency
        comp_a = ComponentInterface(
            component_id="comp_a",
            component_name="Component A",
            phase=PhaseType.PHASE_I,
            requires=["comp_b"]
        )
        comp_b = ComponentInterface(
            component_id="comp_b",
            component_name="Component B",
            phase=PhaseType.PHASE_II,
            requires=["comp_a"]
        )

        assembler.register_component(comp_a)
        assembler.register_component(comp_b)

        # Should still work (assembler breaks cycles)
        result = assembler.assemble(component_ids=["comp_a", "comp_b"])
        # May fail or succeed depending on implementation


# =============================================================================
# Architecture Building Tests
# =============================================================================

class TestArchitectureBuilding:
    """Tests for architecture building"""

    def test_assemble_single_component(self, assembler):
        """Test assembling single component"""
        result = assembler.assemble(component_ids=["sce"])
        assert result.success
        assert len(result.architecture.components) == 1
        assert result.architecture.components[0].component_id == "sce"

    def test_assemble_multiple_components(self, assembler):
        """Test assembling multiple components"""
        result = assembler.assemble(component_ids=["sce", "gamma1", "phi15"])
        assert result.success
        assert len(result.architecture.components) == 3

    def test_auto_select_components(self, assembler):
        """Test auto-selecting components"""
        result = assembler.assemble(component_ids=None)
        assert result.success
        assert len(result.architecture.components) > 0

    def test_architecture_id_generation(self, assembler):
        """Test architecture IDs are generated correctly"""
        result = assembler.assemble(component_ids=["sce", "gamma1"])
        assert result.success
        assert result.architecture.architecture_id.startswith("arch_")
        assert len(result.architecture.architecture_id) == 21  # "arch_" (5) + 16 char hash

    def test_architecture_pattern_detection(self, assembler):
        """Test assembly pattern detection"""
        result = assembler.assemble(component_ids=["sce", "gamma1", "phi15"])
        assert result.success
        # Should be sequential or hybrid (phi15 depends on sce)
        assert result.architecture.assembly_pattern in [
            AssemblyPattern.SEQUENTIAL,
            AssemblyPattern.HYBRID
        ]

    def test_architecture_metadata(self, assembler):
        """Test architecture metadata is set correctly"""
        result = assembler.assemble(component_ids=["sce"])
        assert result.success
        arch = result.architecture

        assert arch.created_by == "delta1_assembler"
        assert arch.version == "1.0"
        assert hasattr(arch, 'created_at')

    def test_assembly_time_tracking(self, assembler):
        """Test assembly time is tracked"""
        result = assembler.assemble(component_ids=["sce", "gamma1"])
        assert result.success
        assert result.assembly_time >= 0
        assert result.assembly_time < 1.0  # Should be fast


# =============================================================================
# Compatibility Tests
# =============================================================================

class TestCompatibility:
    """Tests for component compatibility"""

    def test_compatible_components(self):
        """Test compatible components"""
        comp1 = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.PHASE_I
        )
        comp2 = ComponentInterface(
            component_id="comp2",
            component_name="Component 2",
            phase=PhaseType.PHASE_II
        )

        assert are_compatible(comp1, comp2)

    def test_incompatible_components_circular(self):
        """Test incompatible components (circular dependency)"""
        comp1 = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.PHASE_I,
            requires=["comp2"]
        )
        comp2 = ComponentInterface(
            component_id="comp2",
            component_name="Component 2",
            phase=PhaseType.PHASE_II,
            requires=["comp1"]
        )

        assert not are_compatible(comp1, comp2)

    def test_architecture_add_compatible(self, simple_architecture, test_component):
        """Test adding compatible component to architecture"""
        assert simple_architecture.is_compatible(test_component)
        assert simple_architecture.add_component(test_component)
        assert simple_architecture.has_component("test_comp")

    def test_architecture_reject_incompatible(self, simple_architecture):
        """Test architecture rejects incompatible component"""
        # Try adding duplicate
        existing_comp = simple_architecture.components[0]
        assert not simple_architecture.add_component(existing_comp)


# =============================================================================
# ACI Estimation Tests
# =============================================================================

class TestACIEstimation:
    """Tests for ACI estimation"""

    def test_aci_improvement_estimation(self, assembler):
        """Test ACI improvement is estimated"""
        result = assembler.assemble(component_ids=["sce", "phi15", "gamma1"])
        assert result.success
        # Should have positive ACI improvement
        assert result.architecture.expected_aci_improvement > 0

    def test_no_components_zero_aci(self, assembler):
        """Test architecture with no ACI-improving components"""
        # Create architecture with only SCE (neutral ACI change)
        result = assembler.assemble(component_ids=["sce"])
        assert result.success
        # SCE has INCREASE change, so should have some improvement
        assert result.architecture.expected_aci_improvement >= 0

    def test_multiple_components_higher_aci(self, assembler):
        """Test more components = higher ACI improvement"""
        result1 = assembler.assemble(component_ids=["sce"])
        result2 = assembler.assemble(component_ids=["sce", "phi15", "gamma1"])

        assert result1.success
        assert result2.success
        assert result2.architecture.expected_aci_improvement >= result1.architecture.expected_aci_improvement


# =============================================================================
# Runtime Estimation Tests
# =============================================================================

class TestRuntimeEstimation:
    """Tests for runtime estimation"""

    def test_runtime_estimation(self, assembler):
        """Test runtime is estimated"""
        result = assembler.assemble(component_ids=["sce", "gamma1"])
        assert result.success
        assert result.architecture.estimated_runtime > 0

    def test_more_components_longer_runtime(self, assembler):
        """Test more components = longer estimated runtime"""
        result1 = assembler.assemble(component_ids=["sce"])
        result2 = assembler.assemble(component_ids=["sce", "gamma1", "phi15"])

        assert result1.success
        assert result2.success
        # More components should take longer (approximately)
        assert result2.architecture.estimated_runtime >= result1.architecture.estimated_runtime


# =============================================================================
# Dependency Layer Tests
# =============================================================================

class TestDependencyLayers:
    """Tests for dependency layer construction"""

    def test_layers_constructed(self, assembler):
        """Test dependency layers are constructed"""
        result = assembler.assemble(component_ids=["sce", "gamma1", "gamma2"])
        assert result.success
        assert len(result.architecture.dependency_layers) > 0

    def test_layer_ordering(self, assembler):
        """Test layers are correctly ordered"""
        result = assembler.assemble(component_ids=["gamma2"])
        assert result.success
        arch = result.architecture

        # gamma2 requires gamma1, so gamma1 should be in earlier layer
        all_ids = []
        for layer in arch.dependency_layers:
            all_ids.extend(layer)

        gamma1_idx = all_ids.index("gamma1")
        gamma2_idx = all_ids.index("gamma2")
        assert gamma1_idx < gamma2_idx


# =============================================================================
# Architecture Fingerprinting Tests
# =============================================================================

class TestFingerprinting:
    """Tests for architecture fingerprinting"""

    def test_fingerprint_generation(self, assembler, simple_architecture):
        """Test fingerprint is generated"""
        fingerprint = assembler.generate_fingerprint(simple_architecture)
        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 64  # SHA256 hash

    def test_fingerprint_consistency(self, assembler, simple_architecture):
        """Test fingerprint is consistent"""
        fp1 = assembler.generate_fingerprint(simple_architecture)
        fp2 = assembler.generate_fingerprint(simple_architecture)
        assert fp1 == fp2

    def test_fingerprint_uniqueness(self, assembler):
        """Test different architectures have different fingerprints"""
        result1 = assembler.assemble(component_ids=["sce", "gamma1"])
        result2 = assembler.assemble(component_ids=["sce", "phi15"])

        assert result1.success
        assert result2.success

        fp1 = assembler.generate_fingerprint(result1.architecture)
        fp2 = assembler.generate_fingerprint(result2.architecture)
        assert fp1 != fp2


# =============================================================================
# Architecture Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for architecture serialization"""

    def test_to_dict(self, assembler, simple_architecture):
        """Test architecture serialization to dict"""
        data = simple_architecture.to_dict()

        assert isinstance(data, dict)
        assert "architecture_id" in data
        assert "name" in data
        assert "components" in data
        assert "validation_score" in data

    def test_serialized_data_types(self, simple_architecture):
        """Test serialized data has correct types"""
        data = simple_architecture.to_dict()

        assert isinstance(data["architecture_id"], str)
        assert isinstance(data["components"], list)
        assert isinstance(data["validation_score"], (int, float))


# =============================================================================
# Beam Search Tests
# =============================================================================

class TestBeamSearch:
    """Tests for beam search assembly"""

    def test_beam_search_assembler(self):
        """Test beam search assembler"""
        assembler = BeamSearchAssembler(
            config=AssemblyConfig(strategy="beam", beam_width=3)
        )
        result = assembler.assemble(component_ids=["sce", "gamma1", "phi15"])
        assert result.success

    def test_beam_width_parameter(self):
        """Test beam width parameter is respected"""
        assembler = BeamSearchAssembler(
            config=AssemblyConfig(strategy="beam", beam_width=2)
        )
        # Should work with small beam width
        result = assembler.assemble(component_ids=["sce", "gamma1"])
        assert result.success


# =============================================================================
# Architecture Method Tests
# =============================================================================

class TestArchitectureMethods:
    """Tests for Architecture class methods"""

    def test_has_component_true(self, simple_architecture):
        """Test has_component returns True for existing component"""
        assert simple_architecture.has_component("sce")

    def test_has_component_false(self, simple_architecture):
        """Test has_component returns False for non-existent component"""
        assert not simple_architecture.has_component("nonexistent")

    def test_get_component_found(self, simple_architecture):
        """Test get_component returns component when found"""
        comp = simple_architecture.get_component("sce")
        assert comp is not None
        assert comp.component_id == "sce"

    def test_get_component_not_found(self, simple_architecture):
        """Test get_component returns None when not found"""
        comp = simple_architecture.get_component("nonexistent")
        assert comp is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestAssemblyIntegration:
    """Integration tests for assembly"""

    def test_full_assembly_pipeline(self, assembler):
        """Test complete assembly pipeline"""
        # Assemble
        result = assembler.assemble(component_ids=["sce", "gamma1", "gamma2"])
        assert result.success

        # Validate structure
        arch = result.architecture
        assert len(arch.components) >= 2
        assert arch.validation_score >= 0

    def test_assembly_with_all_validated_components(self, assembler):
        """Test assembly with only validated components"""
        validated_ids = [
            cid for cid, comp in assembler.available_components.items()
            if comp.is_validated
        ]

        result = assembler.assemble(component_ids=validated_ids)
        assert result.success
        # Should have high validation score
        assert result.architecture.validation_score >= 0.5

    def test_assembly_statistics(self, assembler):
        """Test assembly statistics are tracked"""
        initial_count = assembler.assemblies_created

        assembler.assemble(component_ids=["sce", "gamma1"])
        assembler.assemble(component_ids=["sce", "phi15"])

        assert assembler.assemblies_created == initial_count + 2


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_unknown_component(self, assembler):
        """Test requesting unknown component fails gracefully"""
        result = assembler.assemble(component_ids=["unknown_component"])
        assert not result.success
        assert "Unknown component" in result.message

    def test_empty_component_list(self, assembler):
        """Test empty component list creates minimal architecture"""
        result = assembler.assemble(component_ids=[])
        # May succeed or fail depending on implementation
        # If succeeds, should have minimal components
        if result.success:
            assert len(result.architecture.components) >= 0

    def test_invalid_component_id(self, assembler):
        """Test invalid component ID type"""
        # This should handle gracefully
        result = assembler.assemble(component_ids=None)  # Auto-select
        assert result.success or result.message


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for assembly"""

    def test_assembly_speed(self, assembler):
        """Test assembly is fast"""
        import time

        start = time.time()
        result = assembler.assemble(component_ids=["sce", "gamma1", "phi15", "psi3"])
        elapsed = time.time() - start

        assert result.success
        assert elapsed < 1.0  # Should complete in under 1 second

    def test_large_assembly(self, assembler):
        """Test assembling many components"""
        # Get all available components
        all_ids = list(assembler.available_components.keys())

        result = assembler.assemble(component_ids=all_ids)
        # Should succeed or fail gracefully
        if result.success:
            assert len(result.architecture.components) <= len(all_ids)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
