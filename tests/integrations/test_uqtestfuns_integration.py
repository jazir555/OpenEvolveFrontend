"""
uqtestfuns Integration Tests

Comprehensive test suite for uqtestfuns integration with OpenEvolve.
Tests the adapter, bridge, and validation pipeline components.
"""

import pytest
import asyncio
import numpy as np
from typing import List
import sys
import os

# Add integrations directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from integrations.uqtestfuns import UQTestFunsAdapter, UQTestFunsBridge, validate_with_uq
from integrations.base.uq_interface import (
    ProbabilisticInput,
    UQResult,
    SamplingMethod,
    SensitivityMethod,
    ImportError,
    ValidationError,
    SamplingError,
    EvaluationError,
    AnalysisError,
    PipelineError
)


# Test fixtures

@pytest.fixture
async def adapter():
    """Create and initialize an adapter instance."""
    adapter = UQTestFunsAdapter()
    success = await adapter.initialize({'enabled': True, 'cache_enabled': True})
    if not success:
        pytest.skip("uqtestfuns library not available")
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def bridge(adapter):
    """Create and initialize a bridge instance."""
    bridge = UQTestFunsBridge(adapter=adapter)
    success = await bridge.initialize({'enabled': True})
    if not success:
        pytest.skip("Bridge initialization failed")
    yield bridge
    await bridge.shutdown()


@pytest.fixture
def sample_inputs() -> List[ProbabilisticInput]:
    """Create sample probabilistic inputs."""
    return [
        ProbabilisticInput(
            name='x1',
            distribution='uniform',
            parameters=[-3.14159, 3.14159]
        ),
        ProbabilisticInput(
            name='x2',
            distribution='uniform',
            parameters=[-3.14159, 3.14159]
        ),
        ProbabilisticInput(
            name='x3',
            distribution='uniform',
            parameters=[-3.14159, 3.14159]
        )
    ]


# Adapter Tests

class TestUQTestFunsAdapter:
    """Test suite for UQTestFunsAdapter."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Test adapter initialization."""
        adapter = UQTestFunsAdapter()

        # Test successful initialization
        success = await adapter.initialize({'enabled': True})
        if not success:
            pytest.skip("uqtestfuns library not available")

        assert adapter._initialized is True
        assert adapter._config is not None

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_disabled_initialization(self):
        """Test adapter initialization when disabled."""
        adapter = UQTestFunsAdapter()
        success = await adapter.initialize({'enabled': False})
        assert success is False

    @pytest.mark.asyncio
    async def test_list_available_functions(self, adapter):
        """Test listing available test functions."""
        functions = await adapter.list_available_functions()

        assert isinstance(functions, list)
        assert len(functions) > 0
        assert 'ishigami' in functions
        assert 'ackley' in functions
        assert 'rosenbrock' in functions

    @pytest.mark.asyncio
    async def test_get_function_info(self, adapter):
        """Test getting information about a test function."""
        info = await adapter.get_function_info('ishigami')

        assert 'name' in info
        assert info['name'] == 'ishigami'
        assert 'description' in info
        assert 'input_dimension' in info

    @pytest.mark.asyncio
    async def test_get_function_info_invalid(self, adapter):
        """Test getting info for invalid function."""
        with pytest.raises(ValidationError):
            await adapter.get_function_info('invalid_function_name')

    @pytest.mark.asyncio
    async def test_define_probabilistic_inputs(self, adapter, sample_inputs):
        """Test defining probabilistic inputs."""
        result = await adapter.define_probabilistic_inputs(sample_inputs)

        assert 'input_spec' in result
        assert result['input_spec']['n_inputs'] == 3
        assert result['metadata']['validated'] is True

    @pytest.mark.asyncio
    async def test_define_probabilistic_inputs_empty(self, adapter):
        """Test defining empty probabilistic inputs."""
        with pytest.raises(ValidationError):
            await adapter.define_probabilistic_inputs([])

    @pytest.mark.asyncio
    async def test_define_probabilistic_inputs_invalid_distribution(self, adapter):
        """Test defining inputs with invalid distribution."""
        invalid_inputs = [
            ProbabilisticInput(
                name='x1',
                distribution='invalid_distribution',
                parameters=[0, 1]
            )
        ]

        with pytest.raises(ValidationError):
            await adapter.define_probabilistic_inputs(invalid_inputs)

    @pytest.mark.asyncio
    async def test_sample_inputs_monte_carlo(self, adapter, sample_inputs):
        """Test Monte Carlo sampling."""
        samples = await adapter.sample_inputs(
            sample_inputs,
            n_samples=100,
            method=SamplingMethod.MONTE_CARLO,
            seed=42
        )

        assert samples.shape == (100, 3)
        assert np.all(np.isfinite(samples))

    @pytest.mark.asyncio
    async def test_sample_inputs_invalid_n_samples(self, adapter, sample_inputs):
        """Test sampling with invalid n_samples."""
        with pytest.raises(ValidationError):
            await adapter.sample_inputs(sample_inputs, n_samples=0)

    @pytest.mark.asyncio
    async def test_sample_inputs_normal_distribution(self, adapter):
        """Test sampling from normal distribution."""
        inputs = [
            ProbabilisticInput(
                name='x1',
                distribution='normal',
                parameters=[0, 1],  # mean, std
            )
        ]

        samples = await adapter.sample_inputs(
            inputs,
            n_samples=100,
            method=SamplingMethod.MONTE_CARLO,
            seed=42
        )

        assert samples.shape == (100, 1)
        # Check mean is approximately 0 (within 3 std errors)
        assert abs(np.mean(samples) - 0) < 0.3

    @pytest.mark.asyncio
    async def test_evaluate_test_function(self, adapter, sample_inputs):
        """Test evaluating test function."""
        # Create input samples
        input_samples = await adapter.sample_inputs(
            sample_inputs,
            n_samples=10,
            seed=42
        )

        # Evaluate function
        outputs = await adapter.evaluate_test_function(
            'ishigami',
            input_samples
        )

        assert outputs.shape == (10,)
        assert np.all(np.isfinite(outputs))

    @pytest.mark.asyncio
    async def test_evaluate_test_function_invalid_inputs(self, adapter):
        """Test evaluating function with invalid inputs."""
        with pytest.raises(EvaluationError):
            await adapter.evaluate_test_function(
                'ishigami',
                np.array([1, 2, 3])  # Wrong shape
            )

    @pytest.mark.asyncio
    async def test_compute_statistics(self, adapter):
        """Test computing statistics on outputs."""
        outputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = await adapter.compute_statistics(outputs)

        assert 'mean' in stats
        assert abs(stats['mean'] - 3.0) < 1e-10
        assert 'variance' in stats
        assert 'std' in stats
        assert 'percentiles' in stats
        assert 'min' in stats
        assert 'max' in stats

    @pytest.mark.asyncio
    async def test_compute_sensitivity(self, adapter, sample_inputs):
        """Test sensitivity analysis."""
        sensitivity = await adapter.compute_sensitivity(
            'ishigami',
            sample_inputs,
            n_samples=100,
            method=SensitivityMethod.SOBOL,
            seed=42
        )

        assert 'method' in sensitivity
        assert 'first_order' in sensitivity
        assert 'total_order' in sensitivity
        assert len(sensitivity['first_order']) == 3

    @pytest.mark.asyncio
    async def test_run_uq_pipeline(self, adapter, sample_inputs):
        """Test complete UQ pipeline."""
        result = await adapter.run_uq_pipeline(
            function_name='ishigami',
            inputs=sample_inputs,
            n_samples=100,
            sampling_method=SamplingMethod.MONTE_CARLO,
            compute_sensitivity=True,
            seed=42
        )

        assert isinstance(result, UQResult)
        assert result.function_name == 'ishigami'
        assert result.input_samples.shape == (100, 3)
        assert result.output_samples.shape == (100,)
        assert 'mean' in result.statistics
        assert result.sensitivity is not None

    @pytest.mark.asyncio
    async def test_run_uq_pipeline_without_sensitivity(self, adapter, sample_inputs):
        """Test UQ pipeline without sensitivity analysis."""
        result = await adapter.run_uq_pipeline(
            function_name='ishigami',
            inputs=sample_inputs,
            n_samples=100,
            compute_sensitivity=False,
            seed=42
        )

        assert result.sensitivity is None
        assert result.statistics is not None

    @pytest.mark.asyncio
    async def test_pipeline_caching(self, adapter, sample_inputs):
        """Test pipeline result caching."""
        # Run pipeline twice with same parameters
        result1 = await adapter.run_uq_pipeline(
            function_name='ishigami',
            inputs=sample_inputs,
            n_samples=100,
            seed=42
        )

        result2 = await adapter.run_uq_pipeline(
            function_name='ishigami',
            inputs=sample_inputs,
            n_samples=100,
            seed=42
        )

        # Results should be identical (cached)
        np.testing.assert_array_equal(result1.input_samples, result2.input_samples)
        np.testing.assert_array_equal(result1.output_samples, result2.output_samples)

    @pytest.mark.asyncio
    async def test_validate(self, adapter):
        """Test system validation."""
        validation = await adapter.validate()

        assert 'is_valid' in validation
        assert 'checks' in validation
        assert 'dependencies' in validation
        assert 'issues' in validation

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter):
        """Test adapter shutdown."""
        success = await adapter.shutdown()
        assert success is True
        assert adapter._initialized is False


# Bridge Tests

class TestUQTestFunsBridge:
    """Test suite for UQTestFunsBridge."""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge._initialized is True

    @pytest.mark.asyncio
    async def test_validate_model_with_uncertainty(self, bridge, sample_inputs):
        """Test model validation with uncertainty."""
        predictions = list(np.random.randn(100))

        result = await bridge.validate_model_with_uncertainty(
            model_predictions=predictions,
            test_function_name='ishigami',
            probabilistic_inputs=sample_inputs,
            confidence_level=0.95
        )

        assert 'is_valid' in result
        assert 'uncertainty_bounds' in result
        assert 'error_metrics' in result
        assert 'sensitivity_analysis' in result
        assert 'recommendation' in result

    @pytest.mark.asyncio
    async def test_analyze_experiment_uncertainty(self, bridge):
        """Test experiment uncertainty analysis."""
        experiment_results = {
            'type': 'optimization',
            'objective_value': 15.7
        }

        input_parameters = {
            'x1': {
                'value': 0.5,
                'uncertainty': {
                    'distribution': 'normal',
                    'parameters': [0.5, 0.1]
                }
            },
            'x2': {
                'value': -0.3,
                'uncertainty': {
                    'distribution': 'normal',
                    'parameters': [-0.3, 0.1]
                }
            }
        }

        result = await bridge.analyze_experiment_uncertainty(
            experiment_results=experiment_results,
            input_parameters=input_parameters,
            n_samples=100
        )

        assert 'propagated_uncertainty' in result
        assert 'sensitivity_to_inputs' in result
        assert 'confidence_intervals' in result
        assert 'recommendations' in result

    @pytest.mark.asyncio
    async def test_enhance_test_verification(self, bridge):
        """Test test verification enhancement."""
        test_results = {
            'status': 'PASS',
            'n_inputs': 3,
            'n_samples': 50
        }

        result = await bridge.enhance_test_verification(
            test_results=test_results,
            test_function_name='ishigami',
            significance_level=0.05
        )

        assert 'enhanced_status' in result
        assert 'statistical_significance' in result
        assert 'uncertainty_metrics' in result
        assert 'recommendations' in result

    @pytest.mark.asyncio
    async def test_get_validation_report(self, bridge):
        """Test validation report generation."""
        report = await bridge.get_validation_report()

        assert 'system_status' in report
        assert 'validation_checks' in report
        assert 'available_functions' in report
        assert 'n_available_functions' in report
        assert isinstance(report['available_functions'], list)


# Convenience Function Tests

class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    @pytest.mark.asyncio
    async def test_validate_with_uq(self):
        """Test convenience validation function."""
        predictions = list(np.random.randn(100))

        result = await validate_with_uq(
            model_predictions=predictions,
            test_function='ishigami',
            n_samples=100
        )

        assert 'is_valid' in result
        assert 'recommendation' in result


# Error Handling Tests

class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_uninitialized_adapter_error(self):
        """Test error when using uninitialized adapter."""
        adapter = UQTestFunsAdapter()

        with pytest.raises(RuntimeError):
            await adapter.run_uq_pipeline(
                function_name='ishigami',
                inputs=[],
                n_samples=10
            )

    @pytest.mark.asyncio
    async def test_invalid_function_name(self, adapter):
        """Test error with invalid function name."""
        inputs = [
            ProbabilisticInput(
                name='x1',
                distribution='uniform',
                parameters=[0, 1]
            )
        ]

        with pytest.raises((EvaluationError, PipelineError)):
            await adapter.run_uq_pipeline(
                function_name='nonexistent_function',
                inputs=inputs,
                n_samples=10
            )


# Performance Tests

class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_large_sample_size(self, adapter, sample_inputs):
        """Test with large sample size."""
        result = await adapter.run_uq_pipeline(
            function_name='ishigami',
            inputs=sample_inputs,
            n_samples=10000,  # Large sample size
            compute_sensitivity=False  # Skip sensitivity for speed
        )

        assert result.input_samples.shape == (10000, 3)
        assert result.output_samples.shape == (10000,)

    @pytest.mark.asyncio
    async def test_repeated_execution(self, adapter, sample_inputs):
        """Test repeated execution for stability."""
        results = []
        n_runs = 5

        for _ in range(n_runs):
            result = await adapter.run_uq_pipeline(
                function_name='ishigami',
                inputs=sample_inputs,
                n_samples=100,
                seed=None  # Different seeds
            )
            results.append(result)

        # All results should be valid
        assert len(results) == n_runs
        for result in results:
            assert isinstance(result, UQResult)


# Integration Tests

class TestIntegration:
    """Integration tests with full workflows."""

    @pytest.mark.asyncio
    async def test_full_validation_workflow(self, bridge):
        """Test complete validation workflow."""
        # Step 1: Generate synthetic model predictions
        np.random.seed(42)
        predictions = list(np.random.randn(50))

        # Step 2: Define inputs
        from integrations.base.uq_interface import ProbabilisticInput
        inputs = [
            ProbabilisticInput(name='x1', distribution='uniform', parameters=[-3.14, 3.14]),
            ProbabilitalInput(name='x2', distribution='uniform', parameters=[-3.14, 3.14]),
            ProbabilisticInput(name='x3', distribution='uniform', parameters=[-3.14, 3.14])
        ]

        # Step 3: Validate
        result = await bridge.validate_model_with_uncertainty(
            model_predictions=predictions,
            test_function_name='ishigami',
            probabilistic_inputs=inputs
        )

        # Step 4: Verify results
        assert 'is_valid' in result
        assert 'uncertainty_bounds' in result
        assert 'sensitivity_analysis' in result

        # Step 5: Get report
        report = await bridge.get_validation_report()
        assert report['system_status'] in ['HEALTHY', 'ISSUES_DETECTED']

    @pytest.mark.asyncio
    async def test_experiment_analysis_workflow(self, bridge):
        """Test complete experiment analysis workflow."""
        # Define experiment
        experiment = {
            'type': 'sensitivity',
            'objective_value': 10.5
        }

        # Define parameters with uncertainties
        parameters = {
            'param1': {
                'value': 1.0,
                'uncertainty': {
                    'distribution': 'normal',
                    'parameters': [1.0, 0.1]
                }
            }
        }

        # Analyze
        result = await bridge.analyze_experiment_uncertainty(
            experiment_results=experiment,
            input_parameters=parameters,
            n_samples=500
        )

        # Verify
        assert 'propagated_uncertainty' in result
        assert 'recommendations' in result
        assert isinstance(result['recommendations'], list)


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
