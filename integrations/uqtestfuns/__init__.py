"""
uqtestfuns Integration for OpenEvolve

This package provides a decoupled adapter pattern integration for the uqtestfuns library,
enabling uncertainty quantification in OpenEvolve's validation and testing workflows.

Components:
- adapter: UQTestFunsAdapter implementing UncertaintyQuantificationInterface
- bridge: UQTestFunsBridge for integration with validation systems
- config: Configuration for UQ analysis

Usage:
    from integrations.uqtestfuns import UQTestFunsAdapter, UQTestFunsBridge

    # Initialize adapter
    adapter = UQTestFunsAdapter()
    await adapter.initialize({'enabled': True})

    # Run UQ analysis
    result = await adapter.run_uq_pipeline(
        function_name='ishigami',
        inputs=probabilistic_inputs,
        n_samples=1000
    )

    # Or use bridge for validation
    bridge = UQTestFunsBridge()
    await bridge.initialize({'enabled': True})

    validation_result = await bridge.validate_model_with_uncertainty(
        model_predictions=predictions,
        test_function_name='ishigami',
        probabilistic_inputs=inputs
    )
"""

from .adapter import UQTestFunsAdapter
from .bridge import UQTestFunsBridge, validate_with_uq

__version__ = '0.1.0'
__all__ = [
    'UQTestFunsAdapter',
    'UQTestFunsBridge',
    'validate_with_uq'
]

# Package metadata
PROJECT_NAME = 'uqtestfuns'
REPOSITORY = 'https://github.com/damar-wicaksono/uqtestfuns'
GAP_FILLED = 'GAP-15 (Uncertainty Quantification)'
INTEGRATION_VALUE = 'P3 (MEDIUM VALUE)'
