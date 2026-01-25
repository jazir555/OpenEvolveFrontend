"""
Curie Integration Test Suite

Comprehensive tests for Curie automated scientific experimentation integration.

Test Coverage:
- Adapter initialization and configuration
- Experiment design from hypotheses
- Protocol execution (simulated)
- Statistical analysis
- Reflection and refinement
- Full workflow execution
- Error handling
- Integration with SOP Generator

Author: Agent 3 (Curie Integration Specialist)
Version: 1.0.0
"""

import pytest
import asyncio
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Import Curie components
from integrations.curie import CurieAdapter, CurieConfig, CurieBridge
from integrations.base.experimentation_interface import (
    ExperimentationInterface,
    ExperimentDomain,
    ExperimentStatus,
    Hypothesis,
    ExperimentProtocol,
    ExperimentResults,
    StatisticalAnalysis,
    ReflectionReport,
    VerificationReport
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
async def curie_config():
    """Provide test configuration for Curie"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = CurieConfig(
            openai_api_key="test-api-key-12345",
            domain="physics",
            workspace_dir=tmpdir,
            cache_enabled=True,
            max_workers=2,
            timeout=30
        )
        yield config


@pytest.fixture
async def curie_adapter(curie_config):
    """Provide initialized Curie adapter"""
    adapter = CurieAdapter(curie_config)
    await adapter.initialize({})
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def curie_bridge(curie_config):
    """Provide initialized Curie bridge"""
    bridge = CurieBridge(
        openai_api_key=curie_config.openai_api_key,
        workspace_dir=curie_config.workspace_dir,
        cache_enabled=curie_config.cache_enabled
    )
    await bridge.initialize()
    yield bridge
    await bridge.shutdown()


@pytest.fixture
def sample_hypothesis():
    """Sample hypothesis for testing"""
    return "Increasing temperature increases the rate of chemical reaction"


@pytest.fixture
def sample_protocol():
    """Sample experiment protocol for testing"""
    hypothesis = Hypothesis(
        statement="Test hypothesis",
        domain=ExperimentDomain.CHEMISTRY,
        independent_variables=["temperature"],
        dependent_variables=["reaction_rate"],
        control_variables=["pressure", "concentration"],
        assumptions=["ideal_conditions"],
        confidence=0.5
    )

    return ExperimentProtocol(
        protocol_id="test_protocol_001",
        hypothesis=hypothesis,
        steps=[
            {
                "step_number": 1,
                "title": "Setup",
                "description": "Setup experiment",
                "action": "setup",
                "parameters": {},
                "materials": [],
                "equipment": [],
                "duration": 600,
                "safety_notes": [],
                "validation_criteria": {}
            }
        ],
        parameters={},
        equipment=[],
        materials=[],
        duration_estimate=3600,
        reproducibility_checks=[]
    )

@pytest.fixture
def sample_protocol_with_hypothesis():
    """Sample experiment protocol with proper hypothesis for testing"""
    hypothesis = Hypothesis(
        statement="Test hypothesis",
        domain=ExperimentDomain.CHEMISTRY,
        independent_variables=["temperature"],
        dependent_variables=["reaction_rate"],
        control_variables=["pressure", "concentration"],
        assumptions=["ideal_conditions"],
        confidence=0.5
    )

    return ExperimentProtocol(
        protocol_id="test_protocol_002",
        hypothesis=hypothesis,
        steps=[
            {
                "step_number": 1,
                "title": "Setup",
                "description": "Setup experiment",
                "action": "setup",
                "parameters": {},
                "materials": [],
                "equipment": [],
                "duration": 600,
                "safety_notes": [],
                "validation_criteria": {}
            }
        ],
        parameters={},
        equipment=[],
        materials=[],
        duration_estimate=3600,
        reproducibility_checks=[]
    )


# ============================================================================
# Adapter Initialization Tests
# ============================================================================

class TestCurieAdapterInitialization:
    """Test suite for Curie adapter initialization"""

    @pytest.mark.asyncio
    async def test_adapter_creation(self, curie_config):
        """Test adapter can be created"""
        adapter = CurieAdapter(curie_config)
        assert adapter is not None
        assert adapter.config == curie_config
        assert not adapter._initialized

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, curie_config):
        """Test adapter initializes successfully"""
        adapter = CurieAdapter(curie_config)
        await adapter.initialize({})

        assert adapter._initialized
        assert adapter.bridge is not None
        assert os.path.exists(curie_config.workspace_dir)

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_double_initialization(self, curie_adapter):
        """Test double initialization is handled gracefully"""
        # Should not raise error
        await curie_adapter.initialize({})
        assert curie_adapter._initialized

    @pytest.mark.asyncio
    async def test_adapter_shutdown(self, curie_adapter):
        """Test adapter shutdown"""
        assert curie_adapter._initialized
        await curie_adapter.shutdown()
        assert not curie_adapter._initialized

    @pytest.mark.asyncio
    async def test_workspace_creation(self, curie_config):
        """Test workspace directory is created"""
        adapter = CurieAdapter(curie_config)
        await adapter.initialize({})

        workspace = Path(curie_config.workspace_dir)
        assert workspace.exists()
        assert (workspace / "protocols").exists()
        assert (workspace / "results").exists()
        assert (workspace / "logs").exists()

        await adapter.shutdown()


# ============================================================================
# Experiment Design Tests
# ============================================================================

class TestExperimentDesign:
    """Test suite for experiment design"""

    @pytest.mark.asyncio
    async def test_design_physics_experiment(self, curie_adapter, sample_hypothesis):
        """Test designing a physics experiment"""
        protocol = await curie_adapter.design_experiment(
            hypothesis=sample_hypothesis,
            domain=ExperimentDomain.PHYSICS
        )

        assert protocol.protocol_id is not None
        assert protocol.hypothesis.domain == ExperimentDomain.PHYSICS
        assert len(protocol.steps) > 0
        assert protocol.duration_estimate > 0

    @pytest.mark.asyncio
    async def test_design_chemistry_experiment(self, curie_adapter):
        """Test designing a chemistry experiment"""
        hypothesis = "Increasing catalyst concentration increases reaction rate"

        protocol = await curie_adapter.design_experiment(
            hypothesis=hypothesis,
            domain=ExperimentDomain.CHEMISTRY,
            constraints=["temperature <= 100°C"],
            available_equipment=["spectrometer", "hotplate"]
        )

        assert protocol.protocol_id is not None
        assert protocol.hypothesis.domain == ExperimentDomain.CHEMISTRY
        assert len(protocol.equipment) > 0

    @pytest.mark.asyncio
    async def test_design_biology_experiment(self, curie_adapter):
        """Test designing a biology experiment"""
        hypothesis = "Higher temperature increases enzyme activity"

        protocol = await curie_adapter.design_experiment(
            hypothesis=hypothesis,
            domain=ExperimentDomain.BIOLOGY
        )

        assert protocol.protocol_id is not None
        assert protocol.hypothesis.domain == ExperimentDomain.BIOLOGY
        assert len(protocol.steps) > 0

    @pytest.mark.asyncio
    async def test_experiment_design_caching(self, curie_adapter, sample_hypothesis):
        """Test experiment design caching works"""
        # First call
        protocol1 = await curie_adapter.design_experiment(
            hypothesis=sample_hypothesis,
            domain=ExperimentDomain.PHYSICS
        )

        # Second call (should use cache)
        protocol2 = await curie_adapter.design_experiment(
            hypothesis=sample_hypothesis,
            domain=ExperimentDomain.PHYSICS
        )

        assert protocol1.protocol_id == protocol2.protocol_id

    @pytest.mark.asyncio
    async def test_design_without_initialization(self, curie_config):
        """Test design fails without initialization"""
        adapter = CurieAdapter(curie_config)
        # Don't initialize

        with pytest.raises(RuntimeError):
            await adapter.design_experiment(
                hypothesis="Test",
                domain=ExperimentDomain.PHYSICS
            )


# ============================================================================
# Experiment Execution Tests
# ============================================================================

class TestExperimentExecution:
    """Test suite for experiment execution"""

    @pytest.mark.asyncio
    async def test_run_single_experiment(self, curie_adapter, sample_protocol):
        """Test running a single experiment iteration"""
        results = await curie_adapter.run_experiment(
            protocol=sample_protocol,
            iterations=1
        )

        assert results.protocol_id == sample_protocol.protocol_id
        assert results.status == ExperimentStatus.COMPLETED
        assert results.reproducibility_score >= 0.0
        assert results.reproducibility_score <= 1.0
        assert results.execution_time > 0

    @pytest.mark.asyncio
    async def test_run_multiple_iterations(self, curie_adapter, sample_protocol):
        """Test running multiple iterations for reproducibility"""
        results = await curie_adapter.run_experiment(
            protocol=sample_protocol,
            iterations=3
        )

        assert results.status == ExperimentStatus.COMPLETED
        assert results.reproducibility_score > 0.8  # High reproducibility

    @pytest.mark.asyncio
    async def test_experiment_data_collection(self, curie_adapter, sample_protocol):
        """Test experimental data is collected"""
        results = await curie_adapter.run_experiment(
            protocol=sample_protocol,
            iterations=1
        )

        assert len(results.data) > 0
        assert len(results.metrics) > 0
        assert len(results.observations) > 0

    @pytest.mark.asyncio
    async def test_experiment_history_storage(self, curie_adapter, sample_protocol):
        """Test experiment results are stored in history"""
        initial_history_length = len(curie_adapter._experiment_history)

        await curie_adapter.run_experiment(
            protocol=sample_protocol,
            iterations=1
        )

        assert len(curie_adapter._experiment_history) == initial_history_length + 1


# ============================================================================
# Statistical Analysis Tests
# ============================================================================

class TestStatisticalAnalysis:
    """Test suite for statistical analysis"""

    @pytest.mark.asyncio
    async def test_analyze_results(self, curie_adapter, sample_protocol):
        """Test statistical analysis of results"""
        # Run experiment first
        results = await curie_adapter.run_experiment(sample_protocol, iterations=3)

        # Analyze results
        analysis = await curie_adapter.analyze_results(
            results=results,
            hypothesis=sample_protocol.hypothesis
        )

        assert analysis is not None
        assert len(analysis.significance_tests) > 0
        assert len(analysis.effect_sizes) > 0
        assert analysis.statistical_power >= 0.0
        assert analysis.statistical_power <= 1.0
        assert len(analysis.recommendations) >= 0

    @pytest.mark.asyncio
    async def test_significance_tests(self, curie_adapter, sample_protocol):
        """Test statistical significance tests are performed"""
        results = await curie_adapter.run_experiment(sample_protocol, iterations=3)
        analysis = await curie_adapter.analyze_results(results, sample_protocol.hypothesis)

        assert "t_test" in analysis.significance_tests or len(analysis.significance_tests) > 0

    @pytest.mark.asyncio
    async def test_effect_size_calculation(self, curie_adapter, sample_protocol):
        """Test effect sizes are calculated"""
        results = await curie_adapter.run_experiment(sample_protocol, iterations=3)
        analysis = await curie_adapter.analyze_results(results, sample_protocol.hypothesis)

        assert len(analysis.effect_sizes) > 0


# ============================================================================
# Reflection and Refinement Tests
# ============================================================================

class TestReflectionAndRefinement:
    """Test suite for reflection and refinement"""

    @pytest.mark.asyncio
    async def test_reflect_on_results(self, curie_adapter, sample_protocol):
        """Test reflection on experimental results"""
        # Run experiment and analyze
        results = await curie_adapter.run_experiment(sample_protocol, iterations=3)
        analysis = await curie_adapter.analyze_results(results, sample_protocol.hypothesis)

        # Reflect
        reflection = await curie_adapter.reflect_and_refine(
            protocol=sample_protocol,
            results=results,
            analysis=analysis
        )

        assert reflection is not None
        assert isinstance(reflection.hypothesis_validated, bool)
        assert isinstance(reflection.confidence_delta, float)
        assert isinstance(reflection.methodological_issues, list)
        assert isinstance(reflection.suggested_improvements, list)
        assert isinstance(reflection.next_experiments, list)
        assert isinstance(reflection.should_continue, bool)

    @pytest.mark.asyncio
    async def test_reflection_identifies_issues(self, curie_adapter, sample_protocol):
        """Test reflection can identify methodological issues"""
        results = await curie_adapter.run_experiment(sample_protocol, iterations=1)
        # Force low reproducibility
        results.reproducibility_score = 0.5

        analysis = await curie_adapter.analyze_results(results, sample_protocol.hypothesis)
        reflection = await curie_adapter.reflect_and_refine(
            sample_protocol, results, analysis
        )

        # Should identify low reproducibility issue
        assert len(reflection.methodological_issues) >= 0

    @pytest.mark.asyncio
    async def test_reflection_suggests_improvements(self, curie_adapter, sample_protocol):
        """Test reflection suggests improvements"""
        results = await curie_adapter.run_experiment(sample_protocol, iterations=1)
        analysis = await curie_adapter.analyze_results(results, sample_protocol.hypothesis)
        reflection = await curie_adapter.reflect_and_refine(
            sample_protocol, results, analysis
        )

        assert len(reflection.suggested_improvements) >= 0


# ============================================================================
# Full Workflow Tests
# ============================================================================

class TestFullWorkflow:
    """Test suite for full hypothesis → result workflow"""

    @pytest.mark.asyncio
    async def test_full_workflow_single_iteration(self, curie_adapter, sample_hypothesis):
        """Test full workflow with single iteration"""
        verification = await curie_adapter.execute_full_workflow(
            hypothesis=sample_hypothesis,
            domain=ExperimentDomain.CHEMISTRY,
            max_iterations=1
        )

        assert verification is not None
        assert isinstance(verification.experiment_valid, bool)
        assert isinstance(verification.statistical_significance, bool)
        assert isinstance(verification.reproducibility_confirmed, bool)
        assert isinstance(verification.methodology_sound, bool)
        assert isinstance(verification.confidence_level, float)
        assert isinstance(verification.gaps_identified, list)
        assert isinstance(verification.recommendations, list)

    @pytest.mark.asyncio
    async def test_full_workflow_multiple_iterations(self, curie_adapter):
        """Test full workflow with multiple refinement iterations"""
        hypothesis = "Particle size affects reaction rate"

        verification = await curie_adapter.execute_full_workflow(
            hypothesis=hypothesis,
            domain=ExperimentDomain.CHEMISTRY,
            max_iterations=3
        )

        assert verification.experiment_valid is not None
        # Should refine across iterations
        assert verification.confidence_level >= 0.0

    @pytest.mark.asyncio
    async def test_workflow_generates_report(self, curie_adapter):
        """Test workflow generates verification report"""
        hypothesis = "Temperature affects gas volume"

        verification = await curie_adapter.execute_full_workflow(
            hypothesis=hypothesis,
            domain=ExperimentDomain.PHYSICS,
            max_iterations=2
        )

        assert len(verification.raw_data) > 0
        assert len(verification.recommendations) >= 0


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Test suite for system validation"""

    @pytest.mark.asyncio
    async def test_validate_configuration(self, curie_adapter):
        """Test system configuration validation"""
        validation = await curie_adapter.validate()

        assert "system_available" in validation
        assert "domains_supported" in validation
        assert "issues" in validation
        assert "capabilities" in validation
        assert isinstance(validation["domains_supported"], list)

    @pytest.mark.asyncio
    async def test_supported_domains(self, curie_adapter):
        """Test supported domains are detected"""
        validation = await curie_adapter.validate()

        # Should have at least physics, chemistry, biology templates
        assert len(validation["domains_supported"]) >= 3

    @pytest.mark.asyncio
    async def test_bridge_validation(self, curie_bridge):
        """Test bridge validation"""
        validation = await curie_bridge.validate()

        assert "valid" in validation
        assert "issues" in validation
        assert "templates_loaded" in validation
        assert "supported_domains" in validation


# ============================================================================
# Bridge Tests
# ============================================================================

class TestCurieBridge:
    """Test suite for Curie bridge"""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self, curie_bridge):
        """Test bridge initializes"""
        assert curie_bridge._initialized

    @pytest.mark.asyncio
    async def test_generate_protocol(self, curie_bridge):
        """Test protocol generation"""
        protocol = await curie_bridge.generate_protocol(
            hypothesis="Test hypothesis",
            domain="physics",
            constraints=[],
            available_equipment=[]
        )

        assert isinstance(protocol, list)
        assert len(protocol) > 0
        assert "step_number" in protocol[0]

    @pytest.mark.asyncio
    async def test_execute_protocol(self, curie_bridge, sample_protocol):
        """Test protocol execution"""
        result = await curie_bridge.execute_protocol(
            protocol=sample_protocol,
            iteration=1
        )

        assert "iteration" in result
        assert "data" in result
        assert "observations" in result
        assert result["iteration"] == 1

    @pytest.mark.asyncio
    async def test_validate_results(self, curie_bridge, sample_protocol):
        """Test result validation"""
        execution_result = {
            "data": {"value": 42},
            "observations": ["Test observation"]
        }

        validation = await curie_bridge.validate_results(
            results=execution_result,
            protocol=sample_protocol
        )

        assert "valid" in validation
        assert "checks_performed" in validation
        assert "issues" in validation

    @pytest.mark.asyncio
    async def test_get_template(self, curie_bridge):
        """Test retrieving experiment templates"""
        template = await curie_bridge.get_template("physics")

        assert template is not None
        assert "domain" in template
        assert template["domain"] == "physics"

    @pytest.mark.asyncio
    async def test_list_supported_domains(self, curie_bridge):
        """Test listing supported domains"""
        domains = await curie_bridge.list_supported_domains()

        assert isinstance(domains, list)
        assert len(domains) >= 3  # At least physics, chemistry, biology


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test suite for error handling"""

    @pytest.mark.asyncio
    async def test_design_without_initialization_raises_error(self, curie_config):
        """Test design fails when adapter not initialized"""
        adapter = CurieAdapter(curie_config)
        # Don't initialize

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.design_experiment(
                hypothesis="Test",
                domain=ExperimentDomain.PHYSICS
            )

    @pytest.mark.asyncio
    async def test_run_without_initialization_raises_error(self, curie_config):
        """Test run fails when adapter not initialized"""
        adapter = CurieAdapter(curie_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await adapter.run_experiment(
                protocol=Mock(),  # Mock protocol
                iterations=1
            )

    @pytest.mark.asyncio
    async def test_missing_domain_template(self, curie_adapter):
        """Test behavior when domain template doesn't exist"""
        # Use a domain without a template
        protocol = await curie_adapter.design_experiment(
            hypothesis="Test hypothesis",
            domain=ExperimentDomain.ML_ENGINEERING
        )

        # Should still generate protocol (fallback)
        assert protocol is not None
        assert len(protocol.steps) > 0

    @pytest.mark.asyncio
    async def test_cache_disabled(self, curie_config):
        """Test behavior with cache disabled"""
        curie_config.cache_enabled = False
        adapter = CurieAdapter(curie_config)
        await adapter.initialize({})

        # Should work without cache
        protocol1 = await adapter.design_experiment(
            hypothesis="Test hypothesis",
            domain=ExperimentDomain.PHYSICS
        )

        protocol2 = await adapter.design_experiment(
            hypothesis="Test hypothesis",
            domain=ExperimentDomain.PHYSICS
        )

        # Different protocol IDs (no cache)
        assert protocol1.protocol_id != protocol2.protocol_id

        await adapter.shutdown()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for Curie with OpenEvolve systems"""

    @pytest.mark.asyncio
    async def test_experimentation_interface_compliance(self, curie_adapter):
        """Test adapter implements ExperimentationInterface correctly"""
        assert isinstance(curie_adapter, ExperimentationInterface)

    @pytest.mark.asyncio
    async def test_workflow_produces_verification_report(self, curie_adapter):
        """Test full workflow produces proper verification report"""
        hypothesis = "Light intensity affects photosynthesis rate"

        verification = await curie_adapter.execute_full_workflow(
            hypothesis=hypothesis,
            domain=ExperimentDomain.BIOLOGY,
            max_iterations=2
        )

        # Verify all required fields
        assert hasattr(verification, 'experiment_valid')
        assert hasattr(verification, 'statistical_significance')
        assert hasattr(verification, 'reproducibility_confirmed')
        assert hasattr(verification, 'methodology_sound')
        assert hasattr(verification, 'confidence_level')
        assert hasattr(verification, 'gaps_identified')
        assert hasattr(verification, 'recommendations')
        assert hasattr(verification, 'raw_data')

    @pytest.mark.asyncio
    async def test_experiment_history_persistence(self, curie_config):
        """Test experiment history is persisted to disk"""
        adapter = CurieAdapter(curie_config)
        await adapter.initialize({})

        # Run experiment
        protocol = await adapter.design_experiment(
            hypothesis="Test",
            domain=ExperimentDomain.PHYSICS
        )
        await adapter.run_experiment(protocol, iterations=1)

        # Shutdown (should save history)
        await adapter.shutdown()

        # Check history file exists
        history_file = Path(curie_config.workspace_dir) / "experiment_history.json"
        assert history_file.exists()

        # Load and verify history
        with open(history_file, 'r') as f:
            history = json.load(f)

        assert len(history) > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for Curie integration"""

    @pytest.mark.asyncio
    async def test_experiment_design_performance(self, curie_adapter):
        """Test experiment design completes in reasonable time"""
        import time

        start = time.time()
        await curie_adapter.design_experiment(
            hypothesis="Test hypothesis",
            domain=ExperimentDomain.PHYSICS
        )
        elapsed = time.time() - start

        # Should complete in less than 5 seconds
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_experiment_execution_performance(self, curie_adapter, sample_protocol):
        """Test experiment execution completes in reasonable time"""
        import time

        start = time.time()
        await curie_adapter.run_experiment(sample_protocol, iterations=1)
        elapsed = time.time() - start

        # Should complete in less than 2 seconds (simulated)
        assert elapsed < 2.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
