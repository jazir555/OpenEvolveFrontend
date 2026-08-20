"""
Tests for configuration presets.

Tests all preset categories:
- Performance presets (4 tests)
- Domain presets (18 tests)
- Use case presets (5 tests)
- System mode presets (4 tests)
- Problem type presets (5 tests)
- Preset manager tests (10+ tests)

Total: 46+ tests
"""

import pytest
from pathlib import Path

# The presets subsystem depends on `knowledge_engine` (external package) whose
# pydantic v1/v2 handling can raise NameError at import time (e.g. `Field` is
# not defined). We skip these tests when that dependency can't be used, rather
# than masking the failure as a hard error. See presets/base.py import chain.
try:
    from openevolve.unified.presets import (
    # Performance
    FastPreset,
    BalancedPreset,
    ThoroughPreset,
    BudgetPreset,
    # Domains - Finance
    FinanceGeneralPreset,
    FinancePortfolioPreset,
    FinanceRiskPreset,
    # Domains - Trading
    TradingGeneralPreset,
    TradingSignalPreset,
    TradingParameterPreset,
    # Domains - Science
    ScienceGeneralPreset,
    ScienceOptimizationPreset,
    ScienceDiscoveryPreset,
    # Domains - Engineering
    EngineeringGeneralPreset,
    EngineeringDesignPreset,
    EngineeringControlPreset,
    # Domains - Pharma
    PharmaGeneralPreset,
    PharmaDrugDiscoveryPreset,
    PharmaClinicalPreset,
    # Domains - Web Design
    WebDesignGeneralPreset,
    WebDesignUxPreset,
    WebDesignPerformancePreset,
    # Use Cases
    QuickPrototypePreset,
    ProductionPreset,
    ResearchPreset,
    ResourceConstrainedPreset,
    QualityCriticalPreset,
    # Systems
    PureOpenEvolvePreset,
    PureLoongFlowPreset,
    HybridAutoPreset,
    CustomPreset,
    # Problem Types
    SingleObjectivePreset,
    MultiObjectivePreset,
    ExpensiveEvaluationPreset,
    FastEvaluationPreset,
    SafetyCriticalPreset,
    # Manager
    PresetManager,
    get_preset_manager,
)
    from openevolve.unified.config import UnifiedEvolutionConfig
except Exception as _exc:  # pragma: no cover - dependency / env issue
    pytest.skip(
        f"openevolve.unified.presets unavailable (dependency issue): {_exc}",
        allow_module_level=True,
    )


# ============================================================================
# PERFORMANCE PRESET TESTS (4 tests)
# ============================================================================

class TestPerformancePresets:
    """Tests for performance-oriented presets."""

    def test_fast_preset_creation(self):
        """Test FastPreset can be created."""
        preset = FastPreset()
        assert preset.name == "fast"
        assert preset.category == "performance"
        assert preset.max_iterations == 20
        assert preset.population_size == 100
        assert preset.log_level == "WARNING"

    def test_balanced_preset_creation(self):
        """Test BalancedPreset can be created."""
        preset = BalancedPreset()
        assert preset.name == "balanced"
        assert preset.category == "performance"
        assert preset.max_iterations == 100
        assert preset.population_size == 500

    def test_thorough_preset_creation(self):
        """Test ThoroughPreset can be created."""
        preset = ThoroughPreset()
        assert preset.name == "thorough"
        assert preset.category == "performance"
        assert preset.max_iterations == 500
        assert preset.population_size == 2000

    def test_budget_preset_creation(self):
        """Test BudgetPreset can be created."""
        preset = BudgetPreset()
        assert preset.name == "budget"
        assert preset.category == "performance"
        assert preset.max_iterations == 10
        assert preset.population_size == 50


# ============================================================================
# DOMAIN PRESET TESTS (18 tests)
# ============================================================================

class TestDomainPresets:
    """Tests for domain-specific presets."""

    # Finance (3 tests)
    def test_finance_general_preset(self):
        """Test FinanceGeneralPreset."""
        preset = FinanceGeneralPreset()
        assert preset.name == "finance_general"
        assert preset.category == "domain"
        assert preset.evolution_mode == "pes"

    def test_finance_portfolio_preset(self):
        """Test FinancePortfolioPreset."""
        preset = FinancePortfolioPreset()
        assert preset.name == "finance_portfolio"
        assert preset.evolution_mode == "mo"

    def test_finance_risk_preset(self):
        """Test FinanceRiskPreset."""
        preset = FinanceRiskPreset()
        assert preset.name == "finance_risk"
        assert preset.evolution_mode == "qd"

    # Trading (3 tests)
    def test_trading_general_preset(self):
        """Test TradingGeneralPreset."""
        preset = TradingGeneralPreset()
        assert preset.name == "trading_general"
        assert preset.category == "domain"
        assert preset.evolution_mode == "adversarial"

    def test_trading_signal_preset(self):
        """Test TradingSignalPreset."""
        preset = TradingSignalPreset()
        assert preset.name == "trading_signal"
        assert preset.evolution_mode == "qd"

    def test_trading_parameter_preset(self):
        """Test TradingParameterPreset."""
        preset = TradingParameterPreset()
        assert preset.name == "trading_parameter"
        assert preset.evolution_mode == "pes"

    # Science (3 tests)
    def test_science_general_preset(self):
        """Test ScienceGeneralPreset."""
        preset = ScienceGeneralPreset()
        assert preset.name == "science_general"
        assert preset.category == "domain"

    def test_science_optimization_preset(self):
        """Test ScienceOptimizationPreset."""
        preset = ScienceOptimizationPreset()
        assert preset.name == "science_optimization"
        assert preset.evolution_mode == "qd"

    def test_science_discovery_preset(self):
        """Test ScienceDiscoveryPreset."""
        preset = ScienceDiscoveryPreset()
        assert preset.name == "science_discovery"
        assert preset.evolution_mode == "qd"

    # Engineering (3 tests)
    def test_engineering_general_preset(self):
        """Test EngineeringGeneralPreset."""
        preset = EngineeringGeneralPreset()
        assert preset.name == "engineering_general"
        assert preset.category == "domain"

    def test_engineering_design_preset(self):
        """Test EngineeringDesignPreset."""
        preset = EngineeringDesignPreset()
        assert preset.name == "engineering_design"
        assert preset.evolution_mode == "mo"

    def test_engineering_control_preset(self):
        """Test EngineeringControlPreset."""
        preset = EngineeringControlPreset()
        assert preset.name == "engineering_control"
        assert preset.category == "domain"

    # Pharma (3 tests)
    def test_pharma_general_preset(self):
        """Test PharmaGeneralPreset."""
        preset = PharmaGeneralPreset()
        assert preset.name == "pharma_general"
        assert preset.category == "domain"
        assert preset.evolution_mode == "qd"

    def test_pharma_drug_discovery_preset(self):
        """Test PharmaDrugDiscoveryPreset."""
        preset = PharmaDrugDiscoveryPreset()
        assert preset.name == "pharma_drug_discovery"
        assert preset.evolution_mode == "qd"

    def test_pharma_clinical_preset(self):
        """Test PharmaClinicalPreset."""
        preset = PharmaClinicalPreset()
        assert preset.name == "pharma_clinical"
        assert preset.evolution_mode == "pes"

    # Web Design (3 tests)
    def test_web_design_general_preset(self):
        """Test WebDesignGeneralPreset."""
        preset = WebDesignGeneralPreset()
        assert preset.name == "web_design_general"
        assert preset.category == "domain"

    def test_web_design_ux_preset(self):
        """Test WebDesignUxPreset."""
        preset = WebDesignUxPreset()
        assert preset.name == "web_design_ux"
        assert preset.evolution_mode == "mo"

    def test_web_design_performance_preset(self):
        """Test WebDesignPerformancePreset."""
        preset = WebDesignPerformancePreset()
        assert preset.name == "web_design_performance"
        assert preset.category == "domain"


# ============================================================================
# USE CASE PRESET TESTS (5 tests)
# ============================================================================

class TestUseCasePresets:
    """Tests for use case presets."""

    def test_quick_prototype_preset(self):
        """Test QuickPrototypePreset."""
        preset = QuickPrototypePreset()
        assert preset.name == "quick_prototype"
        assert preset.category == "use_case"
        assert preset.max_iterations == 10
        assert preset.log_level == "ERROR"

    def test_production_preset(self):
        """Test ProductionPreset."""
        preset = ProductionPreset()
        assert preset.name == "production"
        assert preset.category == "use_case"
        assert preset.max_iterations == 200

    def test_research_preset(self):
        """Test ResearchPreset."""
        preset = ResearchPreset()
        assert preset.name == "research"
        assert preset.category == "use_case"
        assert preset.population_size == 1000

    def test_resource_constrained_preset(self):
        """Test ResourceConstrainedPreset."""
        preset = ResourceConstrainedPreset()
        assert preset.name == "resource_constrained"
        assert preset.category == "use_case"
        assert preset.concurrency == 1

    def test_quality_critical_preset(self):
        """Test QualityCriticalPreset."""
        preset = QualityCriticalPreset()
        assert preset.name == "quality_critical"
        assert preset.category == "use_case"
        assert preset.max_iterations == 300


# ============================================================================
# SYSTEM MODE PRESET TESTS (4 tests)
# ============================================================================

class TestSystemPresets:
    """Tests for system mode presets."""

    def test_pure_openevolve_preset(self):
        """Test PureOpenEvolvePreset."""
        preset = PureOpenEvolvePreset()
        assert preset.name == "pure_openevolve"
        assert preset.category == "system"
        assert preset.evolution_mode == "openevolve"

    def test_pure_loongflow_preset(self):
        """Test PureLoongFlowPreset."""
        preset = PureLoongFlowPreset()
        assert preset.name == "pure_loongflow"
        assert preset.category == "system"
        assert preset.evolution_mode == "pes"

    def test_hybrid_auto_preset(self):
        """Test HybridAutoPreset."""
        preset = HybridAutoPreset()
        assert preset.name == "hybrid_auto"
        assert preset.category == "system"
        assert preset.evolution_mode == "hybrid"

    def test_custom_preset(self):
        """Test CustomPreset."""
        preset = CustomPreset()
        assert preset.name == "custom"
        assert preset.category == "system"


# ============================================================================
# PROBLEM TYPE PRESET TESTS (5 tests)
# ============================================================================

class TestProblemTypePresets:
    """Tests for problem type presets."""

    def test_single_objective_preset(self):
        """Test SingleObjectivePreset."""
        preset = SingleObjectivePreset()
        assert preset.name == "single_objective"
        assert preset.category == "problem_type"
        assert preset.evolution_mode == "openevolve"

    def test_multi_objective_preset(self):
        """Test MultiObjectivePreset."""
        preset = MultiObjectivePreset()
        assert preset.name == "multi_objective"
        assert preset.category == "problem_type"
        assert preset.evolution_mode == "mo"

    def test_expensive_evaluation_preset(self):
        """Test ExpensiveEvaluationPreset."""
        preset = ExpensiveEvaluationPreset()
        assert preset.name == "expensive_evaluation"
        assert preset.category == "problem_type"
        assert preset.max_iterations == 20

    def test_fast_evaluation_preset(self):
        """Test FastEvaluationPreset."""
        preset = FastEvaluationPreset()
        assert preset.name == "fast_evaluation"
        assert preset.category == "problem_type"
        assert preset.max_iterations == 500

    def test_safety_critical_preset(self):
        """Test SafetyCriticalPreset."""
        preset = SafetyCriticalPreset()
        assert preset.name == "safety_critical"
        assert preset.category == "problem_type"
        assert preset.evolution_mode == "adversarial"


# ============================================================================
# PRESET MANAGER TESTS (10+ tests)
# ============================================================================

class TestPresetManager:
    """Tests for PresetManager."""

    def test_manager_initialization(self):
        """Test manager loads all presets."""
        manager = PresetManager()
        assert len(manager.presets) >= 36  # At least 36 presets

    def test_list_all_presets(self):
        """Test listing all presets."""
        manager = get_preset_manager()
        presets = manager.list_presets()
        assert len(presets) >= 36
        assert "fast" in presets
        assert "balanced" in presets
        assert "finance_general" in presets

    def test_list_presets_by_category(self):
        """Test filtering presets by category."""
        manager = get_preset_manager()
        performance = manager.list_presets(category="performance")
        assert len(performance) == 4
        assert "fast" in performance

    def test_get_preset(self):
        """Test getting a specific preset."""
        manager = get_preset_manager()
        preset = manager.get_preset("fast")
        assert preset.name == "fast"
        assert preset.max_iterations == 20

    def test_get_preset_not_found(self):
        """Test getting non-existent preset raises error."""
        manager = get_preset_manager()
        with pytest.raises(ValueError, match="not found"):
            manager.get_preset("nonexistent_preset")

    def test_get_preset_info(self):
        """Test getting preset information."""
        manager = get_preset_manager()
        info = manager.get_preset_info("fast")
        assert info.name == "fast"
        assert info.category == "performance"
        assert len(info.trade_offs) > 0

    def test_apply_preset(self):
        """Test applying preset to configuration."""
        manager = get_preset_manager()
        config = manager.apply_preset("fast")
        assert isinstance(config, UnifiedEvolutionConfig)
        assert config.common.max_iterations == 20

    def test_validate_preset(self):
        """Test preset validation."""
        manager = get_preset_manager()
        result = manager.validate_preset("fast")
        assert result.is_valid is True

    def test_compare_presets(self):
        """Test comparing two presets."""
        manager = get_preset_manager()
        comparison = manager.compare_presets("fast", "thorough")
        assert comparison.preset1 == "fast"
        assert comparison.preset2 == "thorough"
        assert len(comparison.differences) > 0

    def test_search_presets(self):
        """Test searching presets."""
        manager = get_preset_manager()
        results = manager.search_presets("finance")
        assert "finance_general" in results

    def test_save_and_load_preset(self, tmp_path):
        """Test saving and loading preset."""
        manager = get_preset_manager()
        preset = manager.get_preset("fast")

        # Save preset
        save_path = tmp_path / "fast_preset.yaml"
        manager.save_preset(preset, str(save_path))

        # Load preset
        loaded = manager.load_preset(str(save_path))
        assert loaded.name == "fast"
        assert loaded.max_iterations == 20


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPresetIntegration:
    """Integration tests for presets."""

    def test_preset_to_config_conversion(self):
        """Test converting preset to unified config."""
        preset = FastPreset()
        config_dict = preset.to_unified_config()
        assert "evolution_mode" in config_dict
        assert "common" in config_dict
        assert config_dict["common"]["max_iterations"] == 20

    def test_preset_validation(self):
        """Test preset validation."""
        preset = FastPreset()
        result = preset.validate()
        assert result.is_valid is True

    def test_all_presets_can_be_instantiated(self):
        """Test all presets can be instantiated."""
        manager = get_preset_manager()
        for name in manager.list_presets():
            preset = manager.get_preset(name)
            assert preset is not None
            assert preset.name == name

    def test_all_presets_have_info(self):
        """Test all presets provide information."""
        manager = get_preset_manager()
        for name in manager.list_presets():
            info = manager.get_preset_info(name)
            assert info.name == name
            assert info.category is not None
            assert len(info.description) > 0
            assert len(info.when_to_use) > 0
            assert len(info.trade_offs) > 0

    def test_all_presets_convert_to_config(self):
        """Test all presets convert to unified config."""
        manager = get_preset_manager()
        for name in manager.list_presets():
            preset = manager.get_preset(name)
            config_dict = preset.to_unified_config()
            assert "evolution_mode" in config_dict
            assert "common" in config_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
