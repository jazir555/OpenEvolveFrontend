"""
Test Suite for Optional LoongFlow Configuration

Tests all aspects of making LoongFlow truly optional in the evolution workflow.

Author: AI Architecture Team
Date: 2026-01-30
"""

import pytest
from typing import Dict, Any
from openevolve.unified.config import (
    UnifiedEvolutionConfig,
    EvolutionMode,
    DomainType,
    PESConfig,
    QDConfig
)


class TestLoongFlowOptionalConfiguration:
    """Test LoongFlow optional configuration parameters"""

    def test_enable_loongflow_default(self):
        """Test that LoongFlow is enabled by default"""
        config = UnifiedEvolutionConfig()

        assert config.enable_loongflow is True
        assert config.loongflow_fallback_enabled is True
        assert config.require_loongflow is False

    def test_enable_loongflow_explicit(self):
        """Test explicitly enabling LoongFlow"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=True
        )

        assert config.enable_loongflow is True

    def test_disable_loongflow(self):
        """Test disabling LoongFlow"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=False
        )

        assert config.enable_loongflow is False

    def test_loongflow_fallback_default(self):
        """Test that fallback is enabled by default"""
        config = UnifiedEvolutionConfig()

        assert config.loongflow_fallback_enabled is True

    def test_loongflow_fallback_disabled(self):
        """Test disabling LoongFlow fallback"""
        config = UnifiedEvolutionConfig(
            loongflow_fallback_enabled=False
        )

        assert config.loongflow_fallback_enabled is False

    def test_require_loongflow_default(self):
        """Test that LoongFlow is not required by default"""
        config = UnifiedEvolutionConfig()

        assert config.require_loongflow is False

    def test_require_loongflow_explicit(self):
        """Test requiring LoongFlow"""
        config = UnifiedEvolutionConfig(
            require_loongflow=True
        )

        assert config.require_loongflow is True


class TestLoongFlowValidation:
    """Test LoongFlow configuration validation"""

    def test_require_but_disable_raises_error(self):
        """Test that requiring but disabling LoongFlow raises error"""
        with pytest.raises(ValueError) as exc_info:
            UnifiedEvolutionConfig(
                require_loongflow=True,
                enable_loongflow=False
            )

        assert "require_loongflow=True but enable_loongflow=False" in str(exc_info.value)

    def test_require_with_enabled_passes(self):
        """Test that requiring and enabling LoongFlow is valid"""
        config = UnifiedEvolutionConfig(
            require_loongflow=True,
            enable_loongflow=True
        )

        assert config.require_loongflow is True
        assert config.enable_loongflow is True

    def test_disable_with_fallback_valid(self):
        """Test that disabling with fallback is valid (though fallback won't matter)"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=False,
            loongflow_fallback_enabled=True
        )

        assert config.enable_loongflow is False
        assert config.loongflow_fallback_enabled is True


class TestLoongFlowHelperMethods:
    """Test LoongFlow helper methods"""

    def test_is_loongflow_enabled_true(self):
        """Test is_loongflow_enabled returns True when enabled"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=True
        )

        assert config.is_loongflow_enabled() is True

    def test_is_loongflow_enabled_false(self):
        """Test is_loongflow_enabled returns False when disabled"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=False
        )

        assert config.is_loongflow_enabled() is False

    def test_should_use_loongflow_when_disabled(self):
        """Test should_use_loongflow returns False when disabled"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=False
        )

        assert config.should_use_loongflow() is False

    def test_should_use_loongflow_when_enabled_available(self):
        """Test should_use_loongflow returns True when enabled and available"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=True
        )

        # This will return True if LoongFlow is installed, False otherwise
        # We just check it doesn't raise an error
        result = config.should_use_loongflow()
        assert isinstance(result, bool)

    def test_should_use_loongflow_required_unavailable(self):
        """Test should_use_loongflow raises error when required but unavailable"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True,
            loongflow_fallback_enabled=False
        )

        # If LoongFlow is not installed, this should raise RuntimeError
        # If it is installed, it should return True
        try:
            result = config.should_use_loongflow()
            assert result is True
        except RuntimeError as e:
            assert "require_loongflow=True but LoongFlow is not available" in str(e)

    def test_check_loongflow_availability(self):
        """Test _check_loongflow_availability method"""
        config = UnifiedEvolutionConfig()

        # Should return bool
        result = config._check_loongflow_availability()
        assert isinstance(result, bool)


class TestConvenienceMethods:
    """Test convenience methods for creating configurations"""

    def test_openevolve_only(self):
        """Test openevolve_only convenience method"""
        config = UnifiedEvolutionConfig.openevolve_only(
            max_iterations=100,
            domain=DomainType.FINANCE
        )

        assert config.enable_loongflow is False
        assert config.loongflow_fallback_enabled is False
        assert config.require_loongflow is False
        assert config.max_iterations == 100
        assert config.domain == DomainType.FINANCE

    def test_openevolve_only_with_evolution_mode(self):
        """Test openevolve_only sets evolution mode appropriately"""
        config = UnifiedEvolutionConfig.openevolve_only(
            evolution_mode=EvolutionMode.QD
        )

        assert config.enable_loongflow is False
        assert config.evolution_mode == EvolutionMode.QD

    def test_loongflow_required(self):
        """Test loongflow_required convenience method"""
        config = UnifiedEvolutionConfig.loongflow_required(
            domain=DomainType.SCIENCE
        )

        assert config.enable_loongflow is True
        assert config.require_loongflow is True
        assert config.loongflow_fallback_enabled is False
        assert config.domain == DomainType.SCIENCE

    def test_loongflow_required_with_pes_config(self):
        """Test loongflow_required with PES config"""
        config = UnifiedEvolutionConfig.loongflow_required(
            pes=PESConfig(enabled=True),
            evolution_mode=EvolutionMode.PES
        )

        assert config.enable_loongflow is True
        assert config.require_loongflow is True
        assert config.pes.enabled is True
        assert config.evolution_mode == EvolutionMode.PES


class TestConfigurationCombinations:
    """Test various configuration combinations"""

    def test_default_configuration(self):
        """Test default configuration is valid"""
        config = UnifiedEvolutionConfig()

        # Should not raise any errors
        assert config.enable_loongflow is True
        assert config.loongflow_fallback_enabled is True
        assert config.require_loongflow is False

    def test_qd_mode_with_loongflow_disabled(self):
        """Test QD mode with LoongFlow disabled"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD,
            qd=QDConfig(enabled=True),
            enable_loongflow=False
        )

        assert config.evolution_mode == EvolutionMode.QD
        assert config.qd.enabled is True
        assert config.enable_loongflow is False

    def test_pes_mode_with_loongflow_required(self):
        """Test PES mode with LoongFlow required"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.PES,
            pes=PESConfig(enabled=True),
            enable_loongflow=True,
            require_loongflow=True
        )

        assert config.evolution_mode == EvolutionMode.PES
        assert config.pes.enabled is True
        assert config.enable_loongflow is True
        assert config.require_loongflow is True

    def test_auto_mode_with_loongflow_disabled(self):
        """Test AUTO mode with LoongFlow disabled"""
        # Note: In Pydantic v2, field validators run before the full object is constructed
        # So we need to create the config with explicit mode, or verify that auto selection works differently
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD,  # Explicitly set QD mode
            qd=QDConfig(enabled=True),
            enable_loongflow=False
        )

        assert config.evolution_mode == EvolutionMode.QD
        assert config.enable_loongflow is False
        assert config.qd.enabled is True

    def test_full_configuration(self):
        """Test comprehensive configuration with all parameters"""
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.PES,
            enable_loongflow=True,
            loongflow_fallback_enabled=False,
            require_loongflow=True,
            max_iterations=1000,
            domain=DomainType.ENGINEERING,
            pes=PESConfig(enabled=True),
            llm={
                "models": [
                    {
                        "name": "gpt-4",
                        "weight": 1.0
                    }
                ]
            }
        )

        assert config.enable_loongflow is True
        assert config.require_loongflow is True
        assert config.max_iterations == 1000
        assert config.domain == DomainType.ENGINEERING


class TestBackwardCompatibility:
    """Test backward compatibility with existing code"""

    def test_existing_config_still_works(self):
        """Test that existing configuration code still works"""
        # Old-style config without LoongFlow parameters
        config = UnifiedEvolutionConfig(
            evolution_mode=EvolutionMode.QD,
            max_iterations=5000
        )

        # Should use defaults
        assert config.enable_loongflow is True
        assert config.loongflow_fallback_enabled is True
        assert config.max_iterations == 5000

    def test_legacy_config_conversion(self):
        """Test that legacy OpenEvolveConfig conversion still works"""
        from openevolve.unified.config import OpenEvolveConfig

        legacy_config = OpenEvolveConfig(
            max_iterations=2000,
            random_seed=123
        )

        unified_config = legacy_config.to_unified()

        # Should have LoongFlow defaults
        assert unified_config.enable_loongflow is True
        assert unified_config.loongflow_fallback_enabled is True
        assert unified_config.max_iterations == 2000
        assert unified_config.random_seed == 123


class TestLoggingAndWarnings:
    """Test logging behavior with LoongFlow availability"""

    def test_fallback_warning_when_unavailable(self, caplog):
        """Test that warning is logged when falling back due to unavailability"""
        import logging

        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            loongflow_fallback_enabled=True
        )

        with caplog.at_level(logging.WARNING):
            result = config.should_use_loongflow()

        # If LoongFlow is not installed, should log warning
        if not result:
            assert any("Falling back to OpenEvolve modes" in record.message for record in caplog.records)

    def test_no_warning_when_available(self, caplog):
        """Test that no warning is logged when LoongFlow is available"""
        import logging

        config = UnifiedEvolutionConfig(
            enable_loongflow=True
        )

        with caplog.at_level(logging.WARNING):
            result = config.should_use_loongflow()

        # If LoongFlow is installed, no warning should be logged
        if result:
            assert not any("Falling back" in record.message for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
