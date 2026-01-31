"""
Unified Configuration Tests
Comprehensive test suite for configuration system

Author: AI Architecture Team
Date: 2026-01-30
"""

import pytest
from typing import List, Dict, Any
from openevolve.unified import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    QDConfig,
    MOConfig,
    AdversarialConfig,
    LLMModelConfig,
    ConfigValidator,
    ConfigMapper,
    ValidationError,
    validate_config,
    is_valid_config
)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def minimal_config():
    """Minimal valid configuration"""
    return UnifiedEvolutionConfig()


@pytest.fixture
def pes_config():
    """PES mode configuration"""
    return UnifiedEvolutionConfig(
        evolution_mode=EvolutionMode.PES,
        pes=PESConfig(enabled=True),
        llm={
            "models": [LLMModelConfig(name="gpt-4")]
        },
        database={
            "enable_memory": True
        }
    )


@pytest.fixture
def qd_config():
    """QD mode configuration"""
    return UnifiedEvolutionConfig(
        evolution_mode=EvolutionMode.QD,
        qd=QDConfig(enabled=True),
        database={
            "feature_dimensions": ["complexity", "diversity"]
        }
    )


@pytest.fixture
def mo_config():
    """Multi-objective configuration"""
    return UnifiedEvolutionConfig(
        evolution_mode=EvolutionMode.MO,
        mo=MOConfig(
            enabled=True,
            objectives=["return", "risk", "liquidity"]
        )
    )


@pytest.fixture
def adversarial_config():
    """Adversarial mode configuration"""
    return UnifiedEvolutionConfig(
        evolution_mode=EvolutionMode.ADVERSARIAL,
        adversarial=AdversarialConfig(
            enabled=True,
            red_team_models=["gpt-4"]
        )
    )


# ============================================================================
# TEST: Configuration Creation
# ============================================================================

class TestConfigurationCreation:
    """Test configuration creation and defaults"""

    def test_minimal_config(self, minimal_config):
        """Test minimal configuration creation"""
        assert minimal_config.max_iterations == 10000
        assert minimal_config.evolution_mode == EvolutionMode.AUTO
        assert minimal_config.domain == DomainType.GENERAL

    def test_pes_config_creation(self, pes_config):
        """Test PES configuration"""
        assert pes_config.evolution_mode == EvolutionMode.PES
        assert pes_config.pes.enabled is True
        assert pes_config.pes.enable_planning is True

    def test_qd_config_creation(self, qd_config):
        """Test QD configuration"""
        assert qd_config.evolution_mode == EvolutionMode.QD
        assert qd_config.qd.enabled is True
        assert qd_config.qd.grid_resolution == 10

    def test_mo_config_creation(self, mo_config):
        """Test MO configuration"""
        assert mo_config.evolution_mode == EvolutionMode.MO
        assert mo_config.mo.enabled is True
        assert len(mo_config.mo.objectives) == 3

    def test_adversarial_config_creation(self, adversarial_config):
        """Test adversarial configuration"""
        assert adversarial_config.evolution_mode == EvolutionMode.ADVERSARIAL
        assert adversarial_config.adversarial.enabled is True


# ============================================================================
# TEST: Auto Mode Detection
# ============================================================================

class TestAutoModeDetection:
    """Test automatic mode detection"""

    def test_auto_pes_detection(self):
        """Test auto-detect PES mode"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.AUTO,
            pes=PESConfig(enabled=True)
        )
        # Auto-detection should set to PES
        assert config.evolution_mode == EvolutionMode.PES

    def test_auto_qd_detection(self):
        """Test auto-detect QD mode"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.AUTO,
            qd=QDConfig(enabled=True)
        )
        assert config.evolution_mode == EvolutionMode.QD

    def test_auto_mo_detection(self):
        """Test auto-detect MO mode"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.AUTO,
            mo=MOConfig(enabled=True, objectives=["a", "b"])
        )
        assert config.evolution_mode == EvolutionMode.MO

    def test_auto_standard_fallback(self):
        """Test fallback to standard mode"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.AUTO
        )
        assert config.evolution_mode == EvolutionMode.STANDARD


# ============================================================================
# TEST: Configuration Validation
# ============================================================================

class TestConfigurationValidation:
    """Test configuration validation"""

    def test_valid_minimal_config(self, minimal_config):
        """Test minimal config is valid"""
        validator = ConfigValidator(minimal_config)
        errors, warnings = validator.validate()
        assert len(errors) == 0

    def test_valid_pes_config(self, pes_config):
        """Test PES config is valid"""
        validator = ConfigValidator(pes_config)
        errors, warnings = validator.validate()
        assert len(errors) == 0

    def test_valid_qd_config(self, qd_config):
        """Test QD config is valid"""
        validator = ConfigValidator(qd_config)
        errors, warnings = validator.validate()
        assert len(errors) == 0

    def test_valid_mo_config(self, mo_config):
        """Test MO config is valid"""
        validator = ConfigValidator(mo_config)
        errors, warnings = validator.validate()
        assert len(errors) == 0

    def test_invalid_mo_no_objectives(self):
        """Test MO config fails without objectives"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.MO,
            mo=MOConfig(enabled=True)
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert len(errors) > 0
        assert any("objectives" in e.message.lower() for e in errors)

    def test_invalid_qd_no_dimensions(self):
        """Test QD config fails without feature dimensions"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(enabled=True),
            database={"feature_dimensions": []}
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert len(errors) > 0

    def test_invalid_multiple_modes(self):
        """Test validation catches multiple enabled modes"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(enabled=True),
            qd=QDConfig(enabled=True),
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert len(errors) > 0
        assert any("multiple" in e.message.lower() for e in errors)

    def test_warning_no_llm_models(self):
        """Test warning when no LLM models configured"""
        config = UnifiedEvolutionConfig(
            llm={"models": []}
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert len(errors) > 0
        assert any("llm" in e.category.lower() for e in errors)

    def test_population_exceeds_islands(self):
        """Test validation catches population < islands"""
        config = UnifiedEvolutionConfig(
            database={
                "num_islands": 100,
                "population_size": 10
            }
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert len(errors) > 0


# ============================================================================
# TEST: Config Mapper
# ============================================================================

class TestConfigMapper:
    """Test configuration mapping between formats"""

    def test_to_pes_config(self, pes_config):
        """Test conversion to PES format"""
        pes_dict = ConfigMapper.to_pes_config(pes_config)
        assert "task" in pes_dict
        assert "evolve" in pes_dict
        assert "llm" in pes_dict
        assert "database" in pes_dict
        assert pes_dict["evolve"]["enable_planning"] is True

    def test_to_openevolve_config(self, qd_config):
        """Test conversion to OpenEvolve format"""
        oe_dict = ConfigMapper.to_openevolve_config(qd_config)
        assert "max_iterations" in oe_dict
        assert "database" in oe_dict
        assert "llm" in oe_dict
        assert "evaluator" in oe_dict
        assert oe_dict["evolution_mode"] == "qd"

    def test_to_qd_config(self, qd_config):
        """Test conversion to QD format"""
        qd_dict = ConfigMapper.to_qd_config(qd_config)
        assert qd_dict["evolution_mode"] == "qd"
        assert "grid_resolution" in qd_dict

    def test_to_mo_config(self, mo_config):
        """Test conversion to MO format"""
        mo_dict = ConfigMapper.to_mo_config(mo_config)
        assert mo_dict["evolution_mode"] == "mo"
        assert "objectives" in mo_dict
        assert len(mo_dict["objectives"]) == 3

    def test_from_openevolve_dict(self):
        """Test conversion from OpenEvolve dict"""
        oe_dict = {
            "max_iterations": 500,
            "database": {
                "population_size": 200,
                "num_islands": 3,
            },
            "llm": {
                "temperature": 0.8,
            }
        }
        unified = ConfigMapper.from_openevolve_dict(oe_dict)
        assert unified.max_iterations == 500
        assert unified.database.population_size == 200
        assert unified.llm.temperature == 0.8

    def test_from_pes_dict(self):
        """Test conversion from PES dict"""
        pes_dict = {
            "task": {
                "max_iterations": 100,
            },
            "evolve": {
                "enable_planning": True,
            },
            "database": {
                "num_islands": 5,
            }
        }
        unified = ConfigMapper.from_pes_dict(pes_dict)
        assert unified.evolution_mode == EvolutionMode.PES
        assert unified.max_iterations == 100
        assert unified.pes.enable_planning is True


# ============================================================================
# TEST: Domain-Specific Validation
# ============================================================================

class TestDomainValidation:
    """Test domain-specific validation"""

    def test_finance_domain_recommends_mo(self):
        """Test finance domain suggests MO"""
        config = UnifiedEvolutionConfig(
            domain=DomainType.FINANCE,
            mo=MOConfig(enabled=False)
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert any("multi-objective" in w.message.lower() for w in warnings)

    def test_science_domain_recommends_pes(self):
        """Test science domain suggests PES"""
        config = UnifiedEvolutionConfig(
            domain=DomainType.SCIENCE,
            evolution_mode=EvolutionMode.QD
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert any("pes" in w.message.lower() for w in warnings)

    def test_math_domain_warns_qd(self):
        """Test math domain warns against QD"""
        config = UnifiedEvolutionConfig(
            domain=DomainType.MATH,
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(enabled=True)
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert any("qd" in w.message.lower() or "single objective" in w.message.lower() for w in warnings)


# ============================================================================
# TEST: Parameter Constraints
# ============================================================================

class TestParameterConstraints:
    """Test parameter constraint validation"""

    def test_temperature_range(self):
        """Test temperature must be in [0, 2]"""
        with pytest.raises(Exception):  # Pydantic validation error
            UnifiedEvolutionConfig(
                llm={"temperature": 3.0}
            )

    def test_negative_iterations_invalid(self):
        """Test negative iterations are invalid"""
        with pytest.raises(Exception):
            UnifiedEvolutionConfig(
                max_iterations=-1
            )

    def test_selection_ratios_sum_warning(self):
        """Test warning when selection ratios don't sum to 1.0"""
        config = UnifiedEvolutionConfig(
            database={
                "elite_selection_ratio": 0.5,
                "exploration_ratio": 0.5,
                "exploitation_ratio": 0.5,  # Sum = 1.5
            }
        )
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        assert len(warnings) > 0
        assert any("selection ratios" in w.message.lower() for w in warnings)


# ============================================================================
# TEST: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_validate_config_function(self, minimal_config):
        """Test validate_config convenience function"""
        errors, warnings = validate_config(minimal_config)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)

    def test_is_valid_config_function(self, minimal_config):
        """Test is_valid_config convenience function"""
        assert is_valid_config(minimal_config) is True

    def test_is_invalid_config(self):
        """Test is_valid_config with invalid config"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.MO,
            mo=MOConfig(enabled=True)  # No objectives!
        )
        assert is_valid_config(config) is False


# ============================================================================
# TEST: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full workflows"""

    def test_full_pes_workflow(self):
        """Test complete PES workflow"""
        # Create config
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(enabled=True),
            llm={"models": [LLMModelConfig(name="gpt-4")]},
            database={"enable_memory": True},
        )

        # Validate
        assert is_valid_config(config)

        # Convert to PES format
        pes_dict = ConfigMapper.to_pes_config(config)

        # Verify conversion
        assert pes_dict["evolve"]["enable_planning"] is True
        assert pes_dict["database"]["enable_memory"] is True

    def test_full_qd_workflow(self):
        """Test complete QD workflow"""
        # Create config
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(
                enabled=True,
                grid_resolution=15,
                feature_dimensions=["performance", "complexity"]
            ),
            database={
                "population_size": 800,
                "num_islands": 7,
            },
        )

        # Validate
        assert is_valid_config(config)

        # Convert to OpenEvolve format
        oe_dict = ConfigMapper.to_openevolve_config(config)

        # Verify conversion
        assert oe_dict["evolution_mode"] == "qd"
        assert oe_dict["qd"]["grid_resolution"] == 15
        assert oe_dict["database"]["num_islands"] == 7

    def test_round_trip_conversion(self):
        """Test round-trip conversion preserves data"""
        # Start with OpenEvolve dict
        original = {
            "max_iterations": 777,
            "database": {
                "population_size": 333,
                "num_islands": 9,
            },
        }

        # Convert to unified
        unified = ConfigMapper.from_openevolve_dict(original)

        # Convert back to OpenEvolve
        converted = ConfigMapper.to_openevolve_config(unified)

        # Verify key fields preserved
        assert converted["max_iterations"] == 777
        assert converted["database"]["population_size"] == 333
        assert converted["database"]["num_islands"] == 9


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
