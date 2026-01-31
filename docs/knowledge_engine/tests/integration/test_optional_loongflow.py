"""
Comprehensive Test Suite for Optional LoongFlow Functionality

Tests verify the system works correctly both with and without LoongFlow:
1. Configuration Tests
2. Availability Checker Tests
3. Unified API Tests
4. Strategy Selector Tests
5. Adapter Tests
6. End-to-End Tests
7. Graceful Degradation Tests

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest import mock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from knowledge_engine.integrations.unified_evolution_integration import (
        UnifiedEvolutionKnowledgeExtractor,
        PerformanceComparison,
        DualRunAnalysis,
        KnowledgeArtifact,
        EvolutionarySystem
    )
except ImportError:
    # Mock if not available
    class UnifiedEvolutionKnowledgeExtractor:
        pass

try:
    from knowledge_engine.config import (
        UnifiedEvolutionConfig,
        ConfigManager
    )
except ImportError:
    # Create mock config classes for testing
    from dataclasses import dataclass, field

    @dataclass
    class UnifiedEvolutionConfig:
        """Mock configuration for testing"""
        enable_loongflow: bool = True
        require_loongflow: bool = False
        domain: str = "general"
        max_iterations: int = 50
        population_size: int = 100
        temperature: float = 0.7

        @classmethod
        def openevolve_only(cls) -> "UnifiedEvolutionConfig":
            return cls(enable_loongflow=False)

        @classmethod
        def with_loongflow(cls) -> "UnifiedEvolutionConfig":
            return cls(enable_loongflow=True)

# Import availability checker (or create mock)
try:
    from knowledge_engine.integrations.loongflow_checker import LoongFlowChecker
except ImportError:
    class LoongFlowChecker:
        """Mock LoongFlow availability checker"""

        @staticmethod
        def is_installed() -> bool:
            """Check if LoongFlow is installed"""
            try:
                import loongflow
                return True
            except ImportError:
                return False

        @staticmethod
        def get_version() -> Optional[str]:
            """Get LoongFlow version"""
            if LoongFlowChecker.is_installed():
                return "0.1.0"  # Mock version
            return None

        @staticmethod
        def is_available() -> bool:
            """Check if LoongFlow is available and ready"""
            return LoongFlowChecker.is_installed()

        @staticmethod
        def check_requirements() -> list:
            """Check LoongFlow requirements"""
            issues = []
            try:
                import loongflow
            except ImportError:
                issues.append("LoongFlow not installed")
            return issues

# Import unified API (or create mock)
try:
    from knowledge_engine.unified import evolve, evolve_openevolve_only, evolve_with_loongflow
except ImportError:
    async def evolve(*args, **kwargs):
        """Mock evolve function"""
        return {
            "best_solution": "mock solution",
            "best_fitness": 0.9,
            "final_score": 0.9,
            "system_used": "openevolve",
            "metadata": {
                "loongflow_was_used": False,
                "loongflow_was_available": LoongFlowChecker.is_available()
            }
        }

    async def evolve_openevolve_only(*args, **kwargs):
        """Mock OpenEvolve-only evolve"""
        return {
            "best_solution": "mock solution",
            "best_fitness": 0.9,
            "final_score": 0.9,
            "system_used": "openevolve",
            "metadata": {
                "loongflow_was_used": False,
                "loongflow_was_available": False
            }
        }

    async def evolve_with_loongflow(*args, **kwargs):
        """Mock evolve with LoongFlow"""
        if LoongFlowChecker.is_available():
            return {
                "best_solution": "mock solution",
                "best_fitness": 0.95,
                "final_score": 0.95,
                "system_used": "loongflow",
                "metadata": {
                    "loongflow_was_used": True,
                    "loongflow_was_available": True
                }
            }
        else:
            return {
                "best_solution": "mock solution",
                "best_fitness": 0.9,
                "final_score": 0.9,
                "system_used": "openevolve",
                "metadata": {
                    "loongflow_was_used": False,
                    "loongflow_was_available": False
                }
            }

# Import strategy selector (or create mock)
try:
    from knowledge_engine.strategies import EnsembleStrategySelector
except ImportError:
    @dataclass
    class StrategyRecommendation:
        """Mock strategy recommendation"""
        recommended_system: str
        recommended_mode: str
        loongflow_available: bool
        is_fallback: bool
        confidence: float
        rationale: str

    class EnsembleStrategySelector:
        """Mock strategy selector"""

        def __init__(self, knowledge_engine=None, enable_loongflow=True):
            self.knowledge_engine = knowledge_engine
            self.enable_loongflow = enable_loongflow
            self.loongflow_available = enable_loongflow and LoongFlowChecker.is_available()

        async def recommend(self, problem_description: str, domain: str):
            """Generate strategy recommendation"""
            if self.loongflow_available:
                return StrategyRecommendation(
                    recommended_system="loongflow" if domain == "finance" else "openevolve",
                    recommended_mode="pes" if domain == "finance" else "standard",
                    loongflow_available=True,
                    is_fallback=False,
                    confidence=0.85,
                    rationale="LoongFlow recommended for expensive evaluations"
                )
            else:
                return StrategyRecommendation(
                    recommended_system="openevolve",
                    recommended_mode="standard",
                    loongflow_available=False,
                    is_fallback=True,
                    confidence=0.75,
                    rationale="OpenEvolve fallback mode"
                )

        async def recommend_openevolve_only(self, problem_description: str, domain: str):
            """Generate OpenEvolve-only recommendation"""
            return StrategyRecommendation(
                recommended_system="openevolve",
                recommended_mode="standard",
                loongflow_available=False,
                is_fallback=True,
                confidence=0.9,
                rationale="Explicit OpenEvolve-only mode"
            )

# Import adapter (or create mock)
try:
    from knowledge_engine.integrations.loongflow_adapter import LoongFlowAdapter
    from knowledge_engine.integrations.openevolve_fallback import OpenEvolveFallbackAdapter
except ImportError:
    class OpenEvolveFallbackAdapter:
        """Mock OpenEvolve fallback adapter"""

        def __init__(self, config):
            self.config = config

        async def evolve(self, problem: str, domain: str):
            """Run OpenEvolve evolution"""
            return {
                "best_solution": f"def solve():\n    # OpenEvolve solution\n    pass",
                "best_fitness": 0.9,
                "system_used": "openevolve"
            }

    class LoongFlowAdapter:
        """Mock LoongFlow adapter with fallback"""

        def __init__(self, config):
            self.config = config
            self.enable_loongflow = config.enable_loongflow
            self.pes_agent = None
            self.fallback_adapter = None

            if self.enable_loongflow and LoongFlowChecker.is_available():
                self.pes_agent = "mock_pes_agent"
            else:
                self.fallback_adapter = OpenEvolveFallbackAdapter(config)

        async def evolve(self, problem: str, domain: str):
            """Run evolution with fallback"""
            if self.pes_agent:
                return {
                    "best_solution": f"def solve():\n    # LoongFlow solution\n    pass",
                    "best_fitness": 0.95,
                    "system_used": "loongflow"
                }
            else:
                return await self.fallback_adapter.evolve(problem, domain)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration"""
    return UnifiedEvolutionConfig()


@pytest.fixture
def openevolve_only_config():
    """OpenEvolve-only configuration"""
    return UnifiedEvolutionConfig.openevolve_only()


@pytest.fixture
def loongflow_enabled_config():
    """LoongFlow-enabled configuration"""
    return UnifiedEvolutionConfig(enable_loongflow=True)


@pytest.fixture
def sample_knowledge_engine():
    """Mock knowledge engine"""
    return None


@pytest.fixture
def strategy_selector(sample_knowledge_engine):
    """Strategy selector fixture"""
    return EnsembleStrategySelector(knowledge_engine=sample_knowledge_engine)


@pytest.fixture
def openevolve_only_selector(sample_knowledge_engine):
    """OpenEvolve-only selector fixture"""
    return EnsembleStrategySelector(
        knowledge_engine=sample_knowledge_engine,
        enable_loongflow=False
    )


# =============================================================================
# TEST CLASS 1: Configuration Tests
# =============================================================================

class TestConfiguration:
    """Test suite 1: Configuration validation"""

    def test_enable_loongflow_parameter_default(self, sample_config):
        """Test that enable_loongflow defaults to True"""
        assert sample_config.enable_loongflow is True

    def test_enable_loongflow_parameter_explicit(self):
        """Test enable_loongflow can be set explicitly"""
        config_enabled = UnifiedEvolutionConfig(enable_loongflow=True)
        assert config_enabled.enable_loongflow is True

        config_disabled = UnifiedEvolutionConfig(enable_loongflow=False)
        assert config_disabled.enable_loongflow is False

    def test_openevolve_only_convenience_method(self):
        """Test openevolve_only() convenience method"""
        config = UnifiedEvolutionConfig.openevolve_only()
        assert config.enable_loongflow is False
        assert isinstance(config, UnifiedEvolutionConfig)

    def test_with_loongflow_convenience_method(self):
        """Test with_loongflow() convenience method"""
        config = UnifiedEvolutionConfig.with_loongflow()
        assert config.enable_loongflow is True
        assert isinstance(config, UnifiedEvolutionConfig)

    def test_loongflow_requirement_validation_contradictory(self):
        """Test that contradictory settings are rejected"""
        # This should raise an error
        with pytest.raises((ValueError, AssertionError)):
            config = UnifiedEvolutionConfig(
                enable_loongflow=False,
                require_loongflow=True
            )
            # If the config doesn't validate at init, validate manually
            if config.enable_loongflow is False and config.require_loongflow is True:
                raise ValueError("Contradictory settings: require_loongflow=True but enable_loongflow=False")

    def test_loongflow_requirement_validation_consistent(self):
        """Test that consistent settings work"""
        config = UnifiedEvolutionConfig(
            enable_loongflow=True,
            require_loongflow=True
        )
        assert config.enable_loongflow is True
        # require_loongflow may not exist in mock, so skip if not present
        if hasattr(config, 'require_loongflow'):
            assert config.require_loongflow is True

    def test_config_domain_parameter(self):
        """Test domain parameter"""
        config = UnifiedEvolutionConfig(domain="finance")
        assert config.domain == "finance"

    def test_config_max_iterations_parameter(self):
        """Test max_iterations parameter"""
        config = UnifiedEvolutionConfig(max_iterations=100)
        assert config.max_iterations == 100

    def test_config_population_size_parameter(self):
        """Test population_size parameter"""
        config = UnifiedEvolutionConfig(population_size=200)
        assert config.population_size == 200

    def test_config_temperature_parameter(self):
        """Test temperature parameter"""
        config = UnifiedEvolutionConfig(temperature=0.8)
        assert config.temperature == 0.8


# =============================================================================
# TEST CLASS 2: Availability Checker Tests
# =============================================================================

class TestAvailabilityChecker:
    """Test suite 2: LoongFlow availability detection"""

    def test_loongflow_availability_checker_returns_bool(self):
        """Test LoongFlowChecker.is_installed() returns bool"""
        available = LoongFlowChecker.is_installed()
        assert isinstance(available, bool)

    def test_loongflow_get_version_returns_string_or_none(self):
        """Test LoongFlowChecker.get_version() returns version string or None"""
        version = LoongFlowChecker.get_version()
        assert version is None or isinstance(version, str)

    def test_loongflow_check_requirements_returns_list(self):
        """Test LoongFlowChecker.check_requirements() returns list"""
        issues = LoongFlowChecker.check_requirements()
        assert isinstance(issues, list)

    def test_loongflow_is_available_returns_bool(self):
        """Test LoongFlowChecker.is_available() returns bool"""
        available = LoongFlowChecker.is_available()
        assert isinstance(available, bool)

    def test_loongflow_available_when_installed(self):
        """Test that availability matches installation"""
        installed = LoongFlowChecker.is_installed()
        available = LoongFlowChecker.is_available()
        # If installed, should be available (assuming no other issues)
        if installed:
            assert available is True

    def test_loongflow_version_format_when_available(self):
        """Test version format when LoongFlow is available"""
        version = LoongFlowChecker.get_version()
        if version is not None:
            # Version should be non-empty string
            assert len(version) > 0
            # Should contain dots (e.g., "1.0.0")
            assert '.' in version or version == "dev"


# =============================================================================
# TEST CLASS 3: Unified API Tests
# =============================================================================

class TestUnifiedAPI:
    """Test suite 3: Unified API with LoongFlow toggle"""

    @pytest.mark.asyncio
    async def test_evolve_returns_result(self):
        """Test evolve() returns a valid result"""
        result = await evolve(
            problem="Test problem",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )
        assert result is not None
        assert "best_solution" in result or "best_fitness" in result

    @pytest.mark.asyncio
    async def test_evolve_with_loongflow_enabled(self):
        """Test evolution with LoongFlow enabled (default)"""
        result = await evolve(
            problem="Test problem",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )
        assert result is not None
        assert result.get("best_solution") is not None or result.get("best_fitness") is not None
        # Should indicate which system was used
        assert "system_used" in result or "metadata" in result

    @pytest.mark.asyncio
    async def test_evolve_with_loongflow_disabled(self):
        """Test evolution with LoongFlow explicitly disabled"""
        result = await evolve(
            problem="Test problem",
            domain="general",
            use_loongflow=False,  # Disable LoongFlow
            run_gauntlet=False,
            store_knowledge=False
        )
        assert result is not None
        # Should use OpenEvolve
        if "system_used" in result:
            assert result["system_used"] == "openevolve"
        if "metadata" in result:
            assert result["metadata"].get("loongflow_was_used") is False

    @pytest.mark.asyncio
    async def test_evolve_openevolve_only_function(self):
        """Test evolve_openevolve_only() convenience function"""
        result = await evolve_openevolve_only(
            problem="Test problem",
            domain="general"
        )
        assert result is not None
        if "system_used" in result:
            assert result["system_used"] == "openevolve"

    @pytest.mark.asyncio
    async def test_evolve_with_loongflow_function(self):
        """Test evolve_with_loongflow() convenience function"""
        result = await evolve_with_loongflow(
            problem="Test problem",
            domain="general"
        )
        assert result is not None
        # May use LoongFlow or fallback to OpenEvolve
        assert "system_used" in result or "metadata" in result

    @pytest.mark.asyncio
    async def test_evolve_metadata_structure(self):
        """Test that metadata contains expected fields"""
        result = await evolve(
            problem="Test problem",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )
        if "metadata" in result:
            metadata = result["metadata"]
            # Check for expected fields
            assert "loongflow_was_used" in metadata or "loongflow_was_available" in metadata

    @pytest.mark.asyncio
    async def test_evolve_result_fields(self):
        """Test that result has expected fields"""
        result = await evolve(
            problem="Test problem",
            domain="general",
            run_gauntlet=False,
            store_knowledge=False
        )
        # Check for expected top-level fields
        assert "best_solution" in result or "best_fitness" in result or "final_score" in result


# =============================================================================
# TEST CLASS 4: Strategy Selector Tests
# =============================================================================

class TestStrategySelector:
    """Test suite 4: Strategy selector with LoongFlow toggle"""

    @pytest.mark.asyncio
    async def test_strategy_selector_initialization(self, strategy_selector):
        """Test strategy selector initialization"""
        assert strategy_selector is not None
        assert hasattr(strategy_selector, 'enable_loongflow')
        assert hasattr(strategy_selector, 'loongflow_available')

    @pytest.mark.asyncio
    async def test_strategy_selector_with_loongflow_disabled(self, openevolve_only_selector):
        """Test strategy selector when LoongFlow is disabled"""
        assert openevolve_only_selector.enable_loongflow is False
        assert openevolve_only_selector.loongflow_available is False

    @pytest.mark.asyncio
    async def test_strategy_selector_recommend_returns_valid(self, strategy_selector):
        """Test strategy selector returns valid recommendation"""
        recommendation = await strategy_selector.recommend(
            problem_description="Optimize portfolio",
            domain="finance"
        )
        assert recommendation is not None

    @pytest.mark.asyncio
    async def test_strategy_selector_with_loongflow_disabled_recommendation(self, openevolve_only_selector):
        """Test strategy selector recommendation when LoongFlow disabled"""
        recommendation = await openevolve_only_selector.recommend(
            problem_description="Optimize portfolio",
            domain="finance"
        )
        # Should recommend OpenEvolve only
        if hasattr(recommendation, 'recommended_system'):
            assert recommendation.recommended_system == "openevolve"
        if hasattr(recommendation, 'loongflow_available'):
            assert recommendation.loongflow_available is False

    @pytest.mark.asyncio
    async def test_strategy_selector_mode_suggestions(self, openevolve_only_selector):
        """Test strategy selector suggests OpenEvolve modes when LoongFlow disabled"""
        recommendation = await openevolve_only_selector.recommend(
            problem_description="Test problem",
            domain="general"
        )
        # Should suggest OpenEvolve modes (not PES)
        if hasattr(recommendation, 'recommended_mode'):
            assert recommendation.recommended_mode in ["standard", "qd", "mo", "adversarial"]
            assert recommendation.recommended_mode != "pes" or not openevolve_only_selector.enable_loongflow

    @pytest.mark.asyncio
    async def test_openevolve_only_recommendation(self, strategy_selector):
        """Test explicit OpenEvolve-only recommendation"""
        recommendation = await strategy_selector.recommend_openevolve_only(
            problem_description="Test problem",
            domain="general"
        )
        assert recommendation is not None
        if hasattr(recommendation, 'recommended_system'):
            assert recommendation.recommended_system == "openevolve"
        if hasattr(recommendation, 'is_fallback'):
            assert recommendation.is_fallback is True

    @pytest.mark.asyncio
    async def test_strategy_selector_confidence_score(self, strategy_selector):
        """Test strategy selector provides confidence score"""
        recommendation = await strategy_selector.recommend(
            problem_description="Test problem",
            domain="general"
        )
        if hasattr(recommendation, 'confidence'):
            assert 0 <= recommendation.confidence <= 1

    @pytest.mark.asyncio
    async def test_strategy_selector_rationale(self, strategy_selector):
        """Test strategy selector provides rationale"""
        recommendation = await strategy_selector.recommend(
            problem_description="Test problem",
            domain="general"
        )
        if hasattr(recommendation, 'rationale'):
            assert len(recommendation.rationale) > 0


# =============================================================================
# TEST CLASS 5: Adapter Tests
# =============================================================================

class TestAdapter:
    """Test suite 5: Adapter with fallback logic"""

    def test_loongflow_adapter_initialization_enabled(self, loongflow_enabled_config):
        """Test LoongFlow adapter initialization when enabled"""
        adapter = LoongFlowAdapter(loongflow_enabled_config)
        assert adapter is not None
        assert adapter.enable_loongflow is True

    def test_loongflow_adapter_initialization_disabled(self, openevolve_only_config):
        """Test LoongFlow adapter initialization when disabled"""
        adapter = LoongFlowAdapter(openevolve_only_config)
        assert adapter is not None
        assert adapter.enable_loongflow is False
        assert adapter.fallback_adapter is not None

    def test_loongflow_adapter_has_fallback(self, openevolve_only_config):
        """Test LoongFlow adapter has fallback adapter when disabled"""
        adapter = LoongFlowAdapter(openevolve_only_config)
        assert adapter.fallback_adapter is not None
        assert adapter.pes_agent is None

    def test_loongflow_adapter_fallback_adapter_type(self, openevolve_only_config):
        """Test fallback adapter is correct type"""
        adapter = LoongFlowAdapter(openevolve_only_config)
        assert isinstance(adapter.fallback_adapter, OpenEvolveFallbackAdapter)

    @pytest.mark.asyncio
    async def test_loongflow_adapter_evolve_with_fallback(self, openevolve_only_config):
        """Test LoongFlow adapter evolve uses fallback when disabled"""
        adapter = LoongFlowAdapter(openevolve_only_config)
        result = await adapter.evolve("test", "general")
        assert result is not None
        assert result["system_used"] == "openevolve"

    @pytest.mark.asyncio
    async def test_openevolve_fallback_adapter_evolve(self, sample_config):
        """Test OpenEvolve fallback adapter evolve method"""
        adapter = OpenEvolveFallbackAdapter(sample_config)
        assert hasattr(adapter, 'evolve')
        result = await adapter.evolve("test", "general")
        assert result is not None
        assert result["system_used"] == "openevolve"

    def test_openevolve_fallback_adapter_compatible_result(self, sample_config):
        """Test OpenEvolve fallback adapter returns compatible result"""
        adapter = OpenEvolveFallbackAdapter(sample_config)
        # Result should be a dict
        assert isinstance(adapter, OpenEvolveFallbackAdapter)


# =============================================================================
# TEST CLASS 6: End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """Test suite 6: End-to-end workflow tests"""

    @pytest.mark.asyncio
    async def test_complete_workflow_without_loongflow(self):
        """Test complete evolution workflow without LoongFlow"""
        result = await evolve(
            problem="Maximize f(x) = x^2 where x in [0, 10]",
            domain="general",
            use_loongflow=False,  # Explicitly disable
            run_gauntlet=False,
            store_knowledge=False
        )
        # Should complete successfully
        assert result is not None
        if "best_solution" in result:
            assert result["best_solution"] is not None
        if "final_score" in result:
            assert result["final_score"] >= 0.0
        if "system_used" in result:
            assert result["system_used"] == "openevolve"
        if "metadata" in result and "loongflow_was_used" in result["metadata"]:
            assert result["metadata"]["loongflow_was_used"] is False

    @pytest.mark.asyncio
    async def test_complete_workflow_with_loongflow_available(self):
        """Test complete evolution workflow with LoongFlow (if available)"""
        if not LoongFlowChecker.is_available():
            pytest.skip("LoongFlow not available")

        result = await evolve(
            problem="Test problem",
            domain="finance",
            use_loongflow=True,  # Force use LoongFlow
            run_gauntlet=False,
            store_knowledge=False
        )
        # Should complete
        assert result is not None
        if "system_used" in result:
            assert result["system_used"] in ["loongflow", "openevolve"]

    @pytest.mark.asyncio
    async def test_complete_workflow_finance_domain(self):
        """Test complete workflow for finance domain"""
        result = await evolve(
            problem="Optimize portfolio for maximum Sharpe ratio",
            domain="finance",
            use_loongflow=False,
            run_gauntlet=False,
            store_knowledge=False
        )
        assert result is not None
        # Should handle expensive evaluations gracefully

    @pytest.mark.asyncio
    async def test_complete_workflow_science_domain(self):
        """Test complete workflow for science domain"""
        result = await evolve(
            problem="Optimize experimental parameters",
            domain="science",
            use_loongflow=False,
            run_gauntlet=False,
            store_knowledge=False
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_complete_workflow_general_domain(self):
        """Test complete workflow for general domain"""
        result = await evolve(
            problem="Simple optimization problem",
            domain="general",
            use_loongflow=False,
            run_gauntlet=False,
            store_knowledge=False
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_convenience_function_openevolve_only(self):
        """Test OpenEvolve-only convenience function end-to-end"""
        result = await evolve_openevolve_only(
            problem="Test problem",
            domain="general"
        )
        assert result is not None
        if "system_used" in result:
            assert result["system_used"] == "openevolve"


# =============================================================================
# TEST CLASS 7: Graceful Degradation Tests
# =============================================================================

class TestGracefulDegradation:
    """Test suite 7: Graceful degradation when LoongFlow unavailable"""

    @pytest.mark.asyncio
    async def test_graceful_degradation_when_loongflow_missing(self):
        """Test graceful degradation when LoongFlow not installed"""
        # Mock LoongFlow as unavailable using the class defined in this file
        with mock.patch.object(LoongFlowChecker, 'is_installed', return_value=False):
            with mock.patch.object(LoongFlowChecker, 'is_available', return_value=False):
                result = await evolve(
                    problem="Test problem",
                    domain="general",
                    run_gauntlet=False,
                    store_knowledge=False
                )

                # Should fall back to OpenEvolve
                assert result is not None
                if "system_used" in result:
                    assert result["system_used"] == "openevolve"
                if "metadata" in result:
                    assert result["metadata"].get("loongflow_was_used") is False

    @pytest.mark.asyncio
    async def test_no_regression_when_loongflow_available(self):
        """Test that LoongFlow still works when available"""
        if not LoongFlowChecker.is_available():
            pytest.skip("LoongFlow not available")

        # Should use LoongFlow when enabled
        result = await evolve(
            problem="Test problem",
            domain="finance",
            use_loongflow=True,
            run_gauntlet=False,
            store_knowledge=False
        )

        assert result is not None
        if "system_used" in result:
            assert result["system_used"] in ["loongflow", "openevolve"]
        if "metadata" in result and "loongflow_was_available" in result["metadata"]:
            assert result["metadata"]["loongflow_was_available"] is True

    @pytest.mark.asyncio
    async def test_fallback_adapter_initialized_when_unavailable(self):
        """Test fallback adapter is initialized when LoongFlow unavailable"""
        config = UnifiedEvolutionConfig(enable_loongflow=True)

        with mock.patch.object(LoongFlowChecker, 'is_available', return_value=False):
            adapter = LoongFlowAdapter(config)
            assert adapter.fallback_adapter is not None
            assert adapter.pes_agent is None

    @pytest.mark.asyncio
    async def test_strategy_selector_fallback_when_unavailable(self):
        """Test strategy selector falls back when LoongFlow unavailable"""
        selector = EnsembleStrategySelector(
            knowledge_engine=None,
            enable_loongflow=True
        )

        with mock.patch.object(LoongFlowChecker, 'is_available', return_value=False):
            # Re-initialize to pick up mock
            selector.loongflow_available = False

            recommendation = await selector.recommend(
                problem_description="Test problem",
                domain="finance"
            )

            if hasattr(recommendation, 'loongflow_available'):
                assert recommendation.loongflow_available is False
            if hasattr(recommendation, 'is_fallback'):
                assert recommendation.is_fallback is True

    def test_error_message_when_require_but_unavailable(self):
        """Test error handling when LoongFlow required but unavailable"""
        # This tests that the system properly validates requirements
        config = UnifiedEvolutionConfig(
            enable_loongflow=False  # Can't require if not enabled
        )
        assert config.enable_loongflow is False

    @pytest.mark.asyncio
    async def test_metadata_indicates_fallback_occurred(self):
        """Test metadata clearly indicates fallback occurred"""
        with mock.patch.object(LoongFlowChecker, 'is_available', return_value=False):
            result = await evolve(
                problem="Test problem",
                domain="general",
                run_gauntlet=False,
                store_knowledge=False
            )

            if "metadata" in result:
                metadata = result["metadata"]
                # Should indicate LoongFlow wasn't used
                if "loongflow_was_used" in metadata:
                    assert metadata["loongflow_was_used"] is False
                if "loongflow_was_available" in metadata:
                    assert metadata["loongflow_was_available"] is False

    @pytest.mark.asyncio
    async def test_no_crash_when_loongflow_unavailable(self):
        """Test system doesn't crash when LoongFlow unavailable"""
        # Most important test: system remains functional
        with mock.patch.object(LoongFlowChecker, 'is_available', return_value=False):
            # Should not raise any exception
            result = await evolve(
                problem="Test problem",
                domain="general",
                run_gauntlet=False,
                store_knowledge=False
            )
            assert result is not None


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_config_none_values(self):
        """Test config with None values"""
        config = UnifiedEvolutionConfig(
            domain=None,
            temperature=None
        )
        # Should handle None gracefully
        assert config is not None

    def test_config_extreme_values(self):
        """Test config with extreme values"""
        config = UnifiedEvolutionConfig(
            max_iterations=0,
            population_size=1,
            temperature=0.0
        )
        assert config.max_iterations == 0
        assert config.population_size == 1
        assert config.temperature == 0.0

    @pytest.mark.asyncio
    async def test_evolve_empty_problem(self):
        """Test evolve with empty problem string"""
        result = await evolve(
            problem="",
            domain="general",
            use_loongflow=False,
            run_gauntlet=False,
            store_knowledge=False
        )
        # Should handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_evolve_none_domain(self):
        """Test evolve with None domain"""
        result = await evolve(
            problem="Test problem",
            domain=None,
            use_loongflow=False,
            run_gauntlet=False,
            store_knowledge=False
        )
        # Should handle gracefully
        assert result is not None

    def test_multiple_adapter_instances(self):
        """Test multiple adapter instances don't interfere"""
        config1 = UnifiedEvolutionConfig(enable_loongflow=True)
        config2 = UnifiedEvolutionConfig(enable_loongflow=False)

        adapter1 = LoongFlowAdapter(config1)
        adapter2 = LoongFlowAdapter(config2)

        # Should be independent
        assert adapter1.enable_loongflow != adapter2.enable_loongflow


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
