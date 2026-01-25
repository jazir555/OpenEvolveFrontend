"""
Comprehensive unit tests for Architecture Assembler (Δ₁)

Tests architecture assembly, component management, dependency resolution,
and pattern detection.

Author: Agent E1 (Δ₁ Specialist)
Created: 2025-12-31
"""

import pytest
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock

# Try to import architecture assembler
try:
    from rese.phase4.architecture_assembler import (
        ArchitectureAssembler,
        Architecture,
        ComponentInterface,
        AssemblyResult,
        AssemblyConfig,
        AssemblyPattern,
        ACIChange,
        PhaseType,
        SideEffect,
        BeamSearchAssembler,
        are_compatible,
        fingerprint_architecture
    )
except ImportError:
    pytest.skip("Architecture assembler module not available", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def basic_config():
    """Create basic assembly config"""
    return AssemblyConfig(
        strategy="greedy",
        require_validation=True,
        min_validation_score=0.6,
        verbose=False
    )


@pytest.fixture
def mock_aci_calculator():
    """Create mock ACI calculator"""
    calc = Mock()
    calc.calculate.return_value = Mock(ACI=0.7)
    return calc


@pytest.fixture
def sample_components():
    """Create sample components for testing"""
    return [
        ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE,
            requires=[],
            provides=["capability1"],
            is_validated=True,
            validation_score=0.9
        ),
        ComponentInterface(
            component_id="comp2",
            component_name="Component 2",
            phase=PhaseType.PHASE_I,
            requires=["comp1"],
            provides=["capability2"],
            is_validated=True,
            validation_score=0.8
        ),
        ComponentInterface(
            component_id="comp3",
            component_name="Component 3",
            phase=PhaseType.PHASE_II,
            requires=["comp1"],
            provides=["capability3"],
            is_validated=True,
            validation_score=0.85
        )
    ]


# =============================================================================
# ComponentInterface Tests
# =============================================================================

class TestComponentInterface:
    """Test ComponentInterface functionality"""

    def test_initialization(self):
        """Test component interface initialization"""
        comp = ComponentInterface(
            component_id="test_comp",
            component_name="Test Component",
            phase=PhaseType.PHASE_I,
            input_types=["Problem"],
            output_types=["Solution"],
            preconditions=["problem not None"],
            postconditions=["solution valid"],
            side_effects=[SideEffect.READ_ONLY],
            requires=["comp1"],
            provides=["test_capability"],
            min_input_aci=0.2,
            max_input_aci=0.8,
            expected_aci_change=ACIChange.INCREASE,
            time_complexity="O(n)",
            space_complexity="O(n)",
            is_validated=True,
            validation_score=0.85
        )

        assert comp.component_id == "test_comp"
        assert comp.component_name == "Test Component"
        assert comp.phase == PhaseType.PHASE_I
        assert comp.expected_aci_change == ACIChange.INCREASE
        assert comp.is_validated
        assert comp.validation_score == 0.85

    def test_default_values(self):
        """Test default values"""
        comp = ComponentInterface(
            component_id="test",
            component_name="Test",
            phase=PhaseType.CORE
        )

        assert len(comp.input_types) == 0
        assert len(comp.output_types) == 0
        assert len(comp.preconditions) == 0
        assert len(comp.postconditions) == 0
        assert len(comp.side_effects) == 0
        assert len(comp.requires) == 0
        assert len(comp.provides) == 0
        assert comp.min_input_aci == 0.0
        assert comp.max_input_aci == 1.0
        assert comp.expected_aci_change == ACIChange.NEUTRAL
        assert not comp.is_validated
        assert comp.validation_score == 0.0


# =============================================================================
# Architecture Tests
# =============================================================================

class TestArchitecture:
    """Test Architecture functionality"""

    def test_initialization(self):
        """Test architecture initialization"""
        arch = Architecture(
            architecture_id="test_arch",
            name="Test Architecture",
            description="Test description"
        )

        assert arch.architecture_id == "test_arch"
        assert arch.name == "Test Architecture"
        assert arch.description == "Test description"
        assert len(arch.components) == 0
        assert arch.assembly_pattern == AssemblyPattern.SEQUENTIAL
        assert arch.validation_score == 0.0

    def test_add_component(self):
        """Test adding component"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE,
            is_validated=True,
            validation_score=0.9
        )

        result = arch.add_component(comp)

        assert result
        assert len(arch.components) == 1
        assert "comp1" in arch.component_validations
        assert arch.component_validations["comp1"] == 0.9

    def test_add_incompatible_component(self):
        """Test adding incompatible component"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        comp1 = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )

        comp2 = ComponentInterface(
            component_id="comp1",  # Duplicate ID
            component_name="Component 2",
            phase=PhaseType.PHASE_I
        )

        arch.add_component(comp1)
        result = arch.add_component(comp2)

        assert not result
        assert len(arch.components) == 1

    def test_is_compatible(self):
        """Test compatibility check"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        # Add first component
        comp1 = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE,
            requires=[]
        )
        arch.add_component(comp1)

        # Compatible component (no extra dependencies)
        comp2 = ComponentInterface(
            component_id="comp2",
            component_name="Component 2",
            phase=PhaseType.PHASE_I,
            requires=[]
        )

        assert arch.is_compatible(comp2)

        # Incompatible component (missing dependencies)
        comp3 = ComponentInterface(
            component_id="comp3",
            component_name="Component 3",
            phase=PhaseType.PHASE_II,
            requires=["nonexistent"]
        )

        assert not arch.is_compatible(comp3)

    def test_has_component(self):
        """Test has_component method"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )

        assert not arch.has_component("comp1")

        arch.add_component(comp)
        assert arch.has_component("comp1")

    def test_get_component(self):
        """Test get_component method"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )

        arch.add_component(comp)

        retrieved = arch.get_component("comp1")
        assert retrieved is not None
        assert retrieved.component_id == "comp1"

        not_found = arch.get_component("comp2")
        assert not_found is None

    def test_to_dict(self):
        """Test to_dict serialization"""
        arch = Architecture(
            architecture_id="test_arch",
            name="Test Architecture",
            description="Test Description"
        )

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )
        arch.add_component(comp)

        data = arch.to_dict()

        assert data['architecture_id'] == "test_arch"
        assert data['name'] == "Test Architecture"
        assert data['description'] == "Test Description"
        assert 'comp1' in data['components']


# =============================================================================
# ArchitectureAssembler Tests
# =============================================================================

class TestArchitectureAssembler:
    """Test ArchitectureAssembler functionality"""

    def test_initialization(self, basic_config):
        """Test assembler initialization"""
        assembler = ArchitectureAssembler(config=basic_config)

        assert assembler.config == basic_config
        assert len(assembler.available_components) > 0  # Default components
        assert assembler.assemblies_created == 0

    def test_initialization_with_aci(self, basic_config, mock_aci_calculator):
        """Test initialization with ACI calculator"""
        assembler = ArchitectureAssembler(
            config=basic_config,
            aci_calculator=mock_aci_calculator
        )

        assert assembler.aci_calculator == mock_aci_calculator

    def test_register_component(self, basic_config):
        """Test component registration"""
        assembler = ArchitectureAssembler(config=basic_config)

        comp = ComponentInterface(
            component_id="custom_comp",
            component_name="Custom Component",
            phase=PhaseType.PHASE_I,
            is_validated=True,
            validation_score=0.9
        )

        assembler.register_component(comp)

        assert "custom_comp" in assembler.available_components
        retrieved = assembler.get_component("custom_comp")
        assert retrieved.component_id == "custom_comp"

    def test_assemble_specific_components(self, basic_config):
        """Test assembling specific components"""
        assembler = ArchitectureAssembler(config=basic_config)

        # Assemble with specific component IDs
        result = assembler.assemble(
            component_ids=["sce", "gamma1"],
            strategy="greedy"
        )

        assert result.success
        assert result.architecture is not None
        assert len(result.architecture.components) == 2

    def test_assemble_unknown_component(self, basic_config):
        """Test assembling with unknown component"""
        assembler = ArchitectureAssembler(config=basic_config)

        result = assembler.assemble(
            component_ids=["unknown_component"]
        )

        assert not result.success
        assert "Unknown component" in result.message

    def test_assemble_auto_select(self, basic_config):
        """Test auto-selecting components"""
        assembler = ArchitectureAssembler(config=basic_config)

        result = assembler.assemble(
            component_ids=None,  # Auto-select
            strategy="greedy"
        )

        assert result.success
        assert result.architecture is not None
        assert len(result.architecture.components) > 0

    def test_select_components(self, basic_config):
        """Test component selection"""
        assembler = ArchitectureAssembler(config=basic_config)

        # Without problem (select validated components)
        selected = assembler._select_components(problem=None)

        assert len(selected) > 0
        for cid in selected:
            comp = assembler.get_component(cid)
            assert comp.is_validated
            assert comp.validation_score >= 0.7

    def test_resolve_dependencies(self, basic_config):
        """Test dependency resolution"""
        assembler = ArchitectureAssembler(config=basic_config)

        # Components with clear dependencies
        component_ids = ["gamma2", "gamma1", "sce"]

        resolved = assembler._resolve_dependencies(component_ids)

        # Should be topologically sorted
        assert "sce" in resolved
        assert "gamma1" in resolved
        assert "gamma2" in resolved

        # gamma2 should come after gamma1
        assert resolved.index("gamma1") < resolved.index("gamma2")

    def test_resolve_cyclic_dependencies(self, basic_config):
        """Test cyclic dependency detection"""
        assembler = ArchitectureAssembler(config=basic_config)

        # Create cyclic dependency
        comp1 = ComponentInterface(
            component_id="cycle1",
            component_name="Cycle 1",
            phase=PhaseType.PHASE_I,
            requires=["cycle2"]
        )

        comp2 = ComponentInterface(
            component_id="cycle2",
            component_name="Cycle 2",
            phase=PhaseType.PHASE_I,
            requires=["cycle1"]
        )

        assembler.register_component(comp1)
        assembler.register_component(comp2)

        with pytest.raises(ValueError, match="Cyclic dependency"):
            assembler._resolve_dependencies(["cycle1", "cycle2"])

    def test_determine_pattern(self, basic_config):
        """Test assembly pattern determination"""
        assembler = ArchitectureAssembler(config=basic_config)

        # Sequential pattern
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        comp1 = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE,
            requires=[]
        )

        comp2 = ComponentInterface(
            component_id="comp2",
            component_name="Component 2",
            phase=PhaseType.PHASE_I,
            requires=["comp1"]
        )

        arch.add_component(comp1)
        arch.add_component(comp2)

        pattern = assembler._determine_pattern(arch)
        assert pattern == AssemblyPattern.SEQUENTIAL

    def test_estimate_aci_improvement(self, basic_config):
        """Test ACI improvement estimation"""
        assembler = ArchitectureAssembler(config=basic_config)

        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        # Add components that increase ACI
        for i in range(3):
            comp = ComponentInterface(
                component_id=f"comp{i}",
                component_name=f"Component {i}",
                phase=PhaseType.PHASE_I,
                expected_aci_change=ACIChange.INCREASE,
                is_validated=True,
                validation_score=0.8
            )
            arch.add_component(comp)

        improvement = assembler._estimate_aci_improvement(arch)

        assert improvement > 0
        assert improvement <= 1.0

    def test_estimate_runtime(self, basic_config):
        """Test runtime estimation"""
        assembler = ArchitectureAssembler(config=basic_config)

        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        # Add components
        for i in range(3):
            comp = ComponentInterface(
                component_id=f"comp{i}",
                component_name=f"Component {i}",
                phase=PhaseType.CORE
            )
            arch.add_component(comp)

        runtime = assembler._estimate_runtime(arch)

        assert runtime > 0

    def test_validate_architecture(self, basic_config):
        """Test architecture validation"""
        assembler = ArchitectureAssembler(config=basic_config)

        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        # Add SCE (required)
        sce = ComponentInterface(
            component_id="sce",
            component_name="Symbolic Constraint Engine",
            phase=PhaseType.CORE,
            is_validated=True,
            validation_score=1.0
        )
        arch.add_component(sce)

        score = assembler._validate_architecture(arch)

        assert score > 0

    def test_validate_architecture_without_sce(self, basic_config):
        """Test architecture without SCE fails validation"""
        assembler = ArchitectureAssembler(config=basic_config)

        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        # Add non-SCE component
        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.PHASE_I
        )
        arch.add_component(comp)

        score = assembler._validate_architecture(arch)

        # Should fail (no SCE)
        assert score == 0.0

    def test_generate_fingerprint(self, basic_config):
        """Test fingerprint generation"""
        assembler = ArchitectureAssembler(config=basic_config)

        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )
        arch.add_component(comp)

        fingerprint = assembler.generate_fingerprint(arch)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16  # First 16 chars of hash

    def test_get_available_components(self, basic_config):
        """Test getting available components"""
        assembler = ArchitectureAssembler(config=basic_config)

        components = assembler.get_available_components()

        assert len(components) > 0
        assert all(isinstance(c, ComponentInterface) for c in components)

    def test_get_component(self, basic_config):
        """Test getting specific component"""
        assembler = ArchitectureAssembler(config=basic_config)

        sce = assembler.get_component("sce")

        assert sce is not None
        assert sce.component_id == "sce"

        not_found = assembler.get_component("nonexistent")
        assert not_found is None


# =============================================================================
# BeamSearchAssembler Tests
# =============================================================================

class TestBeamSearchAssembler:
    """Test BeamSearchAssembler functionality"""

    def test_beam_search_basic(self):
        """Test basic beam search assembly"""
        config = AssemblyConfig(
            strategy="beam",
            beam_width=3,
            verbose=False
        )

        assembler = BeamSearchAssembler(config=config)

        result = assembler.assemble(
            component_ids=["sce", "gamma1", "psi3"],
            strategy="beam"
        )

        # Should complete
        assert isinstance(result, AssemblyResult)

    def test_create_empty_arch(self):
        """Test creating empty architecture"""
        assembler = BeamSearchAssembler()

        arch = assembler._create_empty_arch()

        assert arch.architecture_id == "empty"
        assert arch.name == "Empty"

    def test_copy_architecture(self):
        """Test copying architecture"""
        assembler = BeamSearchAssembler()

        arch1 = Architecture(
            architecture_id="original",
            name="Original",
            description="Test"
        )

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )
        arch1.add_component(comp)

        arch2 = assembler._copy_architecture(arch1)

        assert arch2.architecture_id == "original_copy"
        assert len(arch2.components) == len(arch1.components)

    def test_score_architecture(self):
        """Test architecture scoring"""
        assembler = BeamSearchAssembler()

        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )
        arch.validation_score = 0.8
        arch.expected_aci_improvement = 0.3

        score = assembler._score_architecture(arch)

        assert 0 <= score <= 1.0


# =============================================================================
# Utility Functions Tests
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions"""

    def test_are_compatible(self):
        """Test component compatibility check"""
        comp1 = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE,
            requires=["comp2"]
        )

        comp2 = ComponentInterface(
            component_id="comp2",
            component_name="Component 2",
            phase=PhaseType.PHASE_I,
            requires=["comp1"]
        )

        # Circular dependency - not compatible
        assert not are_compatible(comp1, comp2)

        # Independent components - compatible
        comp3 = ComponentInterface(
            component_id="comp3",
            component_name="Component 3",
            phase=PhaseType.PHASE_II,
            requires=[]
        )

        assert are_compatible(comp1, comp3)

    def test_fingerprint_architecture(self):
        """Test architecture fingerprinting"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )
        arch.assembly_pattern = AssemblyPattern.SEQUENTIAL
        arch.expected_aci_improvement = 0.5

        comp = ComponentInterface(
            component_id="comp1",
            component_name="Component 1",
            phase=PhaseType.CORE
        )
        arch.add_component(comp)

        fingerprint = fingerprint_architecture(arch)

        assert isinstance(fingerprint, str)
        assert len(fingerprint) == 16


# =============================================================================
# AssemblyResult Tests
# =============================================================================

class TestAssemblyResult:
    """Test AssemblyResult functionality"""

    def test_successful_result(self):
        """Test successful assembly result"""
        arch = Architecture(
            architecture_id="test",
            name="Test",
            description="Test"
        )

        result = AssemblyResult(
            architecture=arch,
            success=True,
            message="Assembly successful",
            assembly_time=1.5,
            components_considered=3,
            components_added=3,
            is_validated=True,
            validation_score=0.85
        )

        assert result.success
        assert result.architecture == arch
        assert result.assembly_time == 1.5
        assert result.components_considered == 3
        assert result.components_added == 3
        assert result.is_validated
        assert result.validation_score == 0.85

    def test_failed_result(self):
        """Test failed assembly result"""
        result = AssemblyResult(
            architecture=None,
            success=False,
            message="Validation failed",
            assembly_time=0.5
        )

        assert not result.success
        assert result.architecture is None


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_component_list(self, basic_config):
        """Test with empty component list"""
        assembler = ArchitectureAssembler(config=basic_config)

        result = assembler.assemble(component_ids=[])

        # Should still complete (empty architecture)
        assert result is not None

    def test_very_large_architecture(self, basic_config):
        """Test with many components"""
        assembler = ArchitectureAssembler(config=basic_config)

        # Register many components
        for i in range(20):
            comp = ComponentInterface(
                component_id=f"comp_{i}",
                component_name=f"Component {i}",
                phase=PhaseType.PHASE_I,
                is_validated=True,
                validation_score=0.8
            )
            assembler.register_component(comp)

        # Assemble with limit
        config = AssemblyConfig(
            max_components=10,
            verbose=False
        )

        assembler_with_limit = ArchitectureAssembler(config=config)

        # Should handle gracefully
        result = assembler_with_limit.assemble()
        assert isinstance(result, AssemblyResult)

    def test_duplicate_component_registration(self, basic_config):
        """Test registering duplicate component"""
        assembler = ArchitectureAssembler(config=basic_config)

        comp = ComponentInterface(
            component_id="duplicate",
            component_name="Duplicate",
            phase=PhaseType.CORE
        )

        assembler.register_component(comp)

        # Register again (should overwrite)
        assembler.register_component(comp)

        assert "duplicate" in assembler.available_components
