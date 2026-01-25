# uqtestfuns Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [Purpose](#purpose)
3. [Technical Implementation](#technical-implementation)
4. [Architecture](#architecture)
5. [Integration Points](#integration-points)
6. [Configuration](#configuration)
7. [Test Functions](#test-functions)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)
12. [Future Enhancements](#future-enhancements)

---

## Overview

**uqtestfuns** is a Python library providing a comprehensive collection of test functions for uncertainty quantification (UQ). It has been integrated into OpenEvolve using a decoupled adapter pattern to fill **GAP-15 (Uncertainty Quantification)**.

**Repository**: https://github.com/damar-wicaksono/uqtestfuns

### What is uqtestfuns?

uqtestfuns is a lightweight, well-documented library that provides:
- **Test Function Library**: 20+ benchmark test functions for UQ
- **Probabilistic Input Specifications**: Define uncertain inputs with various distributions
- **Lightweight Dependencies**: Only requires NumPy and SciPy
- **Validation Pipeline**: End-to-end uncertainty propagation and sensitivity analysis

### Key Capabilities

1. **Test Function Integration**
   - Classic UQ test functions (Ishigami, Ackley, Rosenbrock, etc.)
   - Engineering test functions (OTL circuit, wing weight, piston)
   - Optimization benchmark functions (sphere, branin, styblinski-tang)

2. **Probabilistic Input Specifications**
   - Multiple distribution types (uniform, normal, beta, gamma, etc.)
   - Customizable distribution parameters
   - Support for correlated inputs

3. **Sampling Methods**
   - Monte Carlo sampling
   - Latin Hypercube sampling
   - Quasi-random sequences (Sobol, Halton)

4. **Sensitivity Analysis**
   - Variance-based methods (Sobol)
   - Screening methods (Morris)
   - Fourier amplitude sensitivity test (FAST)

---

## Purpose

### GAP-15: Uncertainty Quantification

uqtestfuns fills a critical gap in OpenEvolve's validation and verification capabilities:

**Before Integration**:
- Limited uncertainty quantification in model validation
- No standardized test functions for UQ benchmarking
- Missing sensitivity analysis capabilities
- Incomplete uncertainty propagation in experimental workflows

**After Integration**:
- Comprehensive UQ test function library
- Standardized uncertainty quantification workflows
- Sensitivity analysis for identifying critical parameters
- Enhanced validation with statistical rigor

### Integration Value

**Priority**: P3 (MEDIUM VALUE)

**Impact Areas**:
1. **Model Validation**: Quantify uncertainty in model predictions
2. **Experimentation**: Analyze uncertainty propagation in experiments
3. **Testing**: Enhance test verification with statistical methods
4. **Verification**: Add sensitivity analysis to verification workflows

---

## Technical Implementation

### Adapter Pattern

uqtestfuns is integrated using a **decoupled adapter pattern** that provides:

1. **Interface Abstraction**: `UncertaintyQuantificationInterface` defines the contract
2. **Adapter Implementation**: `UQTestFunsAdapter` implements the interface
3. **Bridge Integration**: `UQTestFunsBridge` connects to validation systems
4. **Zero Modifications**: No changes to uqtestfuns source code

### Component Structure

```
integrations/
├── base/
│   └── uq_interface.py              # Abstract UQ interface
├── uqtestfuns/
│   ├── __init__.py                  # Package initialization
│   ├── adapter.py                   # uqtestfuns adapter
│   ├── bridge.py                    # Validation system bridge
│   └── config.yaml                  # Configuration
```

### Key Design Principles

1. **Decoupling**: Adapter pattern isolates uqtestfuns from OpenEvolve internals
2. **Extensibility**: Interface allows adding other UQ libraries (SALib, Chaospy)
3. **Performance**: Async/await for concurrent operations
4. **Caching**: Optional result caching for improved performance
5. **Error Handling**: Comprehensive exception hierarchy

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenEvolve Systems                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Model       │  │Experimentation│  │    Test      │      │
│  │  Validation  │  │   Results    │  │  Verification│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │  UQTestFuns     │                        │
│                   │  Bridge         │                        │
│                   └────────┬────────┘                        │
│                            │                                 │
└────────────────────────────┼─────────────────────────────────┘
                             │
                   ┌────────▼────────┐
                   │  UQTestFuns     │
                   │  Adapter        │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  uqtestfuns     │
                   │  Library        │
                   └─────────────────┘
```

### UQ Validation Pipeline

```
┌──────────────────────────────────────────────────────────┐
│              UQ Validation Pipeline                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. Define Probabilistic Inputs                         │
│     └─> Specify distributions (normal, uniform, etc.)   │
│                                                          │
│  2. Sample Input Points                                  │
│     └─> Monte Carlo, Latin Hypercube, etc.              │
│                                                          │
│  3. Evaluate Test Function                              │
│     └─> Ishigami, Ackley, Rosenbrock, etc.              │
│                                                          │
│  4. Compute Statistics                                  │
│     └─> Mean, variance, percentiles, etc.               │
│                                                          │
│  5. Sensitivity Analysis (Optional)                     │
│     └─> Sobol, Morris, FAST methods                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Model Validation

**Integration**: `UQTestFunsBridge.validate_model_with_uncertainty()`

Enhances model validation by comparing predictions against UQ test function results:
- Quantifies prediction uncertainty
- Computes confidence intervals
- Performs sensitivity analysis
- Generates validation recommendations

**Example**:
```python
result = await bridge.validate_model_with_uncertainty(
    model_predictions=predictions,
    test_function_name='ishigami',
    probabilistic_inputs=inputs,
    confidence_level=0.95
)
```

### 2. Experimentation Results

**Integration**: `UQTestFunsBridge.analyze_experiment_uncertainty()`

Analyzes uncertainty propagation in experimental results:
- Propagates input uncertainties to outputs
- Identifies critical parameters via sensitivity analysis
- Computes confidence intervals for results
- Generates uncertainty reduction recommendations

**Example**:
```python
result = await bridge.analyze_experiment_uncertainty(
    experiment_results=results,
    input_parameters=parameters_with_uncertainties,
    n_samples=1000
)
```

### 3. Test Verification

**Integration**: `UQTestFunsBridge.enhance_test_verification()`

Enhances test verification with statistical rigor:
- Quantifies uncertainty in test results
- Performs statistical significance testing
- Adds sensitivity analysis to test verification
- Provides uncertainty-aware pass/fail criteria

**Example**:
```python
result = await bridge.enhance_test_verification(
    test_results=test_data,
    test_function_name='ishigami',
    significance_level=0.05
)
```

---

## Configuration

### Configuration File: `config.yaml`

```yaml
project:
  name: uqtestfuns
  version: 0.1.0
  enabled: true

features:
  test_functions: true
  probabilistic_inputs: true
  uncertainty_propagation: true
  sensitivity_analysis: true

integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true

performance:
  max_workers: 4
  timeout: 30
  batch_size: 100

sampling:
  default_method: monte_carlo
  available_methods:
    - monte_carlo
    - latin_hypercube
    - sobol
    - halton
    - grid

sensitivity:
  default_method: sobol
  available_methods:
    - sobol
    - morris
    - fast
    - delta
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable/disable UQ integration |
| `cache_enabled` | boolean | `true` | Cache results for performance |
| `max_workers` | integer | `4` | Maximum parallel workers |
| `timeout` | integer | `30` | Operation timeout (seconds) |
| `auto_start` | boolean | `true` | Auto-load test functions on init |

---

## Test Functions

### Available Test Functions

uqtestfuns provides 20+ benchmark test functions:

#### Classic UQ Functions

| Function | Dimensions | Description | Use Case |
|----------|------------|-------------|----------|
| **Ishigami** | 3 | Highly nonlinear | Sensitivity analysis benchmark |
| **Sobol-G** | variable | Sum of transformed inputs | Variance-based sensitivity |
| **Friedman** | variable | Polynomial with interaction | Feature importance analysis |

#### Optimization Benchmarks

| Function | Dimensions | Properties | Use Case |
|----------|------------|------------|----------|
| **Ackley** | variable | Multimodal | Global optimization |
| **Rosenbrock** | variable | Valley-shaped | Local optimization |
| **Branin** | 2 | Multimodal (2D) | Global optimization |
| **Sphere** | variable | Unimodal, convex | Baseline comparison |

#### Engineering Test Functions

| Function | Domain | Description |
|----------|--------|-------------|
| **OTL Circuit** | Electrical | Analog circuit design |
| **Wing Weight** | Aerospace | Aircraft wing design |
| **Piston** | Mechanical | Mechanical design optimization |

### Function Selection Guide

**For Sensitivity Analysis**:
- Use `ishigami` for benchmarking
- Use `sobol-g` for variable-dimensional analysis

**For Optimization**:
- Use `ackley` for global optimization
- Use `rosenbrock` for local optimization

**For Engineering Applications**:
- Use `otlcircuit` for electrical systems
- Use `wing-weight` for aerospace applications

---

## Usage Examples

### Example 1: Basic UQ Pipeline

```python
import asyncio
from integrations.uqtestfuns import UQTestFunsAdapter
from integrations.base.uq_interface import (
    ProbabilisticInput,
    SamplingMethod
)

async def basic_uq_example():
    # Initialize adapter
    adapter = UQTestFunsAdapter()
    await adapter.initialize({'enabled': True, 'cache_enabled': True})

    # Define probabilistic inputs
    inputs = [
        ProbabilisticInput(
            name='x1',
            distribution='uniform',
            parameters=[-3.14159, 3.14159]  # [low, high]
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

    # Run UQ pipeline
    result = await adapter.run_uq_pipeline(
        function_name='ishigami',
        inputs=inputs,
        n_samples=1000,
        sampling_method=SamplingMethod.MONTE_CARLO,
        compute_sensitivity=True,
        seed=42
    )

    # Print results
    print(f"Mean: {result.statistics['mean']:.4f}")
    print(f"Std Dev: {result.statistics['std']:.4f}")
    print(f"Sensitivity: {result.sensitivity['first_order']}")

    # Shutdown
    await adapter.shutdown()

# Run example
asyncio.run(basic_uq_example())
```

### Example 2: Model Validation

```python
import asyncio
from integrations.uqtestfuns import UQTestFunsBridge
from integrations.base.uq_interface import ProbabilisticInput

async def model_validation_example():
    # Initialize bridge
    bridge = UQTestFunsBridge()
    await bridge.initialize({'enabled': True})

    # Model predictions to validate
    model_predictions = [1.5, 2.3, -0.8, 0.5, 1.2] * 20  # 100 predictions

    # Define inputs
    inputs = [
        ProbabilisticInput(name='x1', distribution='uniform', parameters=[-3.14, 3.14]),
        ProbabilisticInput(name='x2', distribution='uniform', parameters=[-3.14, 3.14]),
        ProbabilisticInput(name='x3', distribution='uniform', parameters=[-3.14, 3.14])
    ]

    # Validate with uncertainty quantification
    result = await bridge.validate_model_with_uncertainty(
        model_predictions=model_predictions,
        test_function_name='ishigami',
        probabilistic_inputs=inputs,
        confidence_level=0.95
    )

    # Print validation results
    print(f"Valid: {result['is_valid']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Uncertainty Bounds: {result['uncertainty_bounds']}")

    # Shutdown
    await bridge.shutdown()

asyncio.run(model_validation_example())
```

### Example 3: Experiment Uncertainty Analysis

```python
import asyncio
from integrations.uqtestfuns import UQTestFunsBridge

async def experiment_uncertainty_example():
    # Initialize bridge
    bridge = UQTestFunsBridge()
    await bridge.initialize({'enabled': True})

    # Experiment results
    experiment_results = {
        'type': 'optimization',
        'objective_value': 15.7,
        'parameters': {'x1': 0.5, 'x2': -0.3, 'x3': 0.8}
    }

    # Input parameters with uncertainties
    input_parameters = {
        'x1': {
            'value': 0.5,
            'uncertainty': {
                'distribution': 'normal',
                'parameters': [0.5, 0.1],  # [mean, std]
                'bounds': [0, 1]
            }
        },
        'x2': {
            'value': -0.3,
            'uncertainty': {
                'distribution': 'normal',
                'parameters': [-0.3, 0.1]
            }
        },
        'x3': {
            'value': 0.8,
            'uncertainty': {
                'distribution': 'normal',
                'parameters': [0.8, 0.15]
            }
        }
    }

    # Analyze uncertainty
    result = await bridge.analyze_experiment_uncertainty(
        experiment_results=experiment_results,
        input_parameters=input_parameters,
        n_samples=1000
    )

    # Print results
    print(f"Uncertainty Level: {result['propagated_uncertainty']['uncertainty_level']}")
    print(f"CoV: {result['propagated_uncertainty']['coefficient_of_variation']:.4f}")
    print(f"Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")

    # Shutdown
    await bridge.shutdown()

asyncio.run(experiment_uncertainty_example())
```

### Example 4: Convenience Function

```python
import asyncio
from integrations.uqtestfuns import validate_with_uq

async def quick_validation_example():
    # Quick validation with convenience function
    predictions = [1.2, 2.5, -0.3, 0.8, 1.7] * 20

    result = await validate_with_uq(
        model_predictions=predictions,
        test_function='ishigami',
        n_samples=100
    )

    print(f"Validation Result: {result['recommendation']}")

asyncio.run(quick_validation_example())
```

---

## API Reference

### UncertaintyQuantificationInterface

Abstract base interface for UQ implementations.

#### Methods

##### `async initialize(config: Dict[str, Any]) -> bool`
Initialize the UQ system with configuration.

##### `async list_available_functions() -> List[str]`
List all available test functions.

##### `async get_function_info(function_name: str) -> Dict[str, Any]`
Get detailed information about a test function.

##### `async define_probabilistic_inputs(inputs: List[ProbabilisticInput]) -> Dict[str, Any]`
Define and validate probabilistic input specifications.

##### `async sample_inputs(inputs, n_samples, method, seed) -> np.ndarray`
Sample input points from probabilistic specifications.

##### `async evaluate_test_function(function_name, input_samples) -> np.ndarray`
Evaluate test function on sampled inputs.

##### `async compute_statistics(output_samples) -> Dict[str, Any]`
Compute statistical summaries of output samples.

##### `async compute_sensitivity(function_name, inputs, n_samples, method, seed) -> Dict[str, Any]`
Perform sensitivity analysis on test function.

##### `async run_uq_pipeline(...) -> UQResult`
Run complete UQ validation pipeline.

### UQTestFunsAdapter

Implementation of `UncertaintyQuantificationInterface` using uqtestfuns.

#### Constructor
```python
UQTestFunsAdapter()
```

#### Example
```python
adapter = UQTestFunsAdapter()
await adapter.initialize({'enabled': True, 'cache_enabled': True})
```

### UQTestFunsBridge

Bridge for integrating uqtestfuns with validation systems.

#### Constructor
```python
UQTestFunsBridge(adapter: Optional[UQTestFunsAdapter] = None)
```

#### Methods

##### `async validate_model_with_uncertainty(...) -> Dict[str, Any]`
Validate model predictions using UQ test functions.

##### `async analyze_experiment_uncertainty(...) -> Dict[str, Any]`
Analyze uncertainty propagation in experimental results.

##### `async enhance_test_verification(...) -> Dict[str, Any]`
Enhance test verification with uncertainty quantification.

##### `async get_validation_report() -> Dict[str, Any]`
Get comprehensive validation report.

### Data Classes

#### ProbabilisticInput
```python
@dataclass
class ProbabilisticInput:
    name: str                          # Input parameter name
    distribution: str                  # Distribution type
    parameters: List[float]            # Distribution parameters
    bounds: Optional[tuple] = None     # Optional bounds
```

#### UQResult
```python
@dataclass
class UQResult:
    function_name: str                 # Function name
    input_samples: np.ndarray          # Input samples
    output_samples: np.ndarray         # Output samples
    statistics: Dict[str, Any]         # Statistical summaries
    sensitivity: Optional[Dict] = None # Sensitivity results
    metadata: Optional[Dict] = None    # Additional metadata
```

### Enums

#### SamplingMethod
```python
class SamplingMethod(Enum):
    MONTE_CARLO = "monte_carlo"
    LATIN_HYPERCUBE = "latin_hypercube"
    SOBOL = "sobol"
    HALTON = "halton"
    GRID = "grid"
```

#### SensitivityMethod
```python
class SensitivityMethod(Enum):
    SOBOL = "sobol"
    MORRIS = "morris"
    FAST = "fast"
    DELTA = "delta"
```

---

## Testing

### Running Tests

```bash
# Run all uqtestfuns integration tests
pytest tests/integrations/test_uqtestfuns_integration.py -v

# Run specific test
pytest tests/integrations/test_uqtestfuns_integration.py::test_adapter_initialization -v

# Run with coverage
pytest tests/integrations/test_uqtestfuns_integration.py --cov=integrations/uqtestfuns
```

### Test Coverage

The test suite covers:
- Adapter initialization and configuration
- Test function listing and information retrieval
- Probabilistic input definition and validation
- Sampling methods (Monte Carlo, Latin Hypercube)
- Test function evaluation
- Statistical computation
- Sensitivity analysis
- Complete UQ pipeline execution
- Bridge integration methods
- Error handling and edge cases

### Example Test

```python
import pytest
import asyncio
from integrations.uqtestfuns import UQTestFunsAdapter
from integrations.base.uq_interface import ProbabilisticInput, SamplingMethod

@pytest.mark.asyncio
async def test_uq_pipeline():
    """Test complete UQ pipeline execution."""
    adapter = UQTestFunsAdapter()
    await adapter.initialize({'enabled': True})

    inputs = [
        ProbabilisticInput(name='x1', distribution='uniform', parameters=[-3.14, 3.14]),
        ProbabilisticInput(name='x2', distribution='uniform', parameters=[-3.14, 3.14]),
        ProbabilisticInput(name='x3', distribution='uniform', parameters=[-3.14, 3.14])
    ]

    result = await adapter.run_uq_pipeline(
        function_name='ishigami',
        inputs=inputs,
        n_samples=100,
        compute_sensitivity=True
    )

    assert result.function_name == 'ishigami'
    assert result.input_samples.shape == (100, 3)
    assert result.output_samples.shape == (100,)
    assert 'mean' in result.statistics
    assert result.sensitivity is not None

    await adapter.shutdown()
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "uqtestfuns library not available"

**Cause**: uqtestfuns is not installed

**Solution**:
```bash
pip install uqtestfuns
```

#### Issue 2: "SciPy not available - limited sampling capabilities"

**Cause**: SciPy is not installed (optional dependency)

**Solution**:
```bash
pip install scipy
```

**Impact**: Without SciPy, only basic Monte Carlo sampling is available.

#### Issue 3: Function evaluation fails

**Cause**: Invalid function name or input dimensions

**Solution**:
- Check function name with `list_available_functions()`
- Verify input dimensions match function requirements
- Use `get_function_info()` to check function specifications

#### Issue 4: High memory usage with large samples

**Cause**: Large `n_samples` values (e.g., > 100,000)

**Solution**:
- Reduce sample size
- Enable caching to reuse results
- Use incremental sampling approach

#### Issue 5: Slow sensitivity analysis

**Cause**: Sobol analysis requires many samples

**Solution**:
- Use `n_samples=1000` for faster (less accurate) results
- Consider Morris screening for faster analysis
- Use parallel execution (increase `max_workers`)

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### Validation Check

Run system validation:

```python
bridge = UQTestFunsBridge()
await bridge.initialize({'enabled': True})

report = await bridge.get_validation_report()
print(report)
```

---

## Future Enhancements

### Planned Improvements

1. **Enhanced Sensitivity Methods**
   - Full Sobol' implementation with proper A/B sampling
   - Moment-independent methods (delta, Borgonovo)
   - Regional sensitivity analysis

2. **Multi-output Support**
   - Support for test functions with multiple outputs
   - Multi-output sensitivity indices
   - Correlation analysis between outputs

3. **Advanced Sampling**
   - Adaptive sampling strategies
   - Importance sampling for rare events
   - Quasi-Monte Carlo sequences (full integration)

4. **Visualization**
   - Uncertainty visualization utilities
   - Sensitivity index plots
   - Convergence diagnostics

5. **Performance**
   - GPU acceleration for large-scale sampling
   - Distributed computing support
   - Incremental/online analysis

6. **Integration Extensions**
   - Integration with workflow orchestration
   - Automated UQ report generation
   - Real-time uncertainty monitoring

### Contribution

To contribute enhancements:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## References

### uqtestfuns Documentation
- **Repository**: https://github.com/damar-wicaksono/uqtestfuns
- **Documentation**: https://uqtestfuns.readthedocs.io/

### UQ Methods
- **Sobol' Indices**: Sobol, I. (2001). "Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates"
- **Morris Method**: Morris, M.D. (1991). "Factorial sampling plans for preliminary computational experiments"
- **LHS**: McKay, M.D. et al. (1979). "A Comparison of Three Methods for Selecting Values of Input Variables"

### OpenEvolve Integration
- **Gap Analysis**: See `PROJECT_GAP_ANALYSIS_AND_RECOMMENDATIONS.md`
- **Integration Tasks**: See `MULTI_AGENT_INTEGRATION_TASK.md`

---

## Support

For questions or issues:
1. Check this guide's Troubleshooting section
2. Review uqtestfuns documentation
3. Check OpenEvolve integration documentation
4. Submit issue on GitHub repository

---

**Version**: 0.1.0
**Last Updated**: 2025-01-02
**Agent**: Agent 6 (uqtestfuns Integration Specialist)
