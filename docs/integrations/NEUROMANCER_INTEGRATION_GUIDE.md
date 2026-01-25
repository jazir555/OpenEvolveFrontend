# NeuroMANCER Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [Purpose and GAP Analysis](#purpose-and-gap-analysis)
3. [Technical Implementation](#technical-implementation)
4. [Architecture](#architecture)
5. [Integration Points](#integration-points)
6. [Configuration](#configuration)
7. [Problem Templates](#problem-templates)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)
12. [Future Enhancements](#future-enhancements)

---

## Overview

**NeuroMANCER** (Neural Modular Architecture for Neuro-Control and Estimation) is a physics-informed machine learning library developed by PNNL (Pacific Northwest National Laboratory). It provides:

- **Physics-informed system identification**: Learn dynamics from data while respecting physical constraints
- **Constrained optimization**: Solve optimization problems with equality/inequality constraints
- **Differentiable programming**: Automatic differentiation through complex computations
- **Neural ODE/PDE solvers**: Learn and solve differential equations

### Why Integrate NeuroMANCER?

NeuroMANCER fills **GAP-3 (Numerical Computation)** and enhances **GAP-1 (Continuous Math)** in OpenEvolve:

| Gap | Description | NeuroMANCER Contribution |
|-----|-------------|-------------------------|
| GAP-3 | Limited numerical optimization capabilities | Physics-informed optimization, constrained solvers |
| GAP-1 | Basic continuous math support | Advanced ODE/PDE solving, differentiable programming |

### Key Capabilities

1. **Hybrid Symbolic-Numerical Solving**: Combines LeanAide (symbolic reasoning) with NeuroMANCER (numerical optimization)
2. **Physics-Informed Constraints**: Enforces conservation laws, physical principles
3. **Scalable Numerical Computing**: Handles large-scale optimization problems
4. **System Identification**: Learn dynamics from data with physics constraints

---

## Purpose and GAP Analysis

### Problem Statement

OpenEvolve lacked:
- Robust numerical optimization capabilities
- Physics-informed problem solving
- System identification from data
- Hybrid symbolic-numerical approaches

### Solution: NeuroMANCER Integration

The integration provides:

1. **Numerical Computation Engine**
   - Constrained/unconstrained optimization
   - ODE/PDE solving
   - System identification

2. **Physics-Informed Solving**
   - Enforces physical constraints
   - Conservation laws
   - Real-world problem modeling

3. **Hybrid Approach**
   - LeanAide: Symbolic analysis, theorem proving, formal verification
   - NeuroMANCER: Numerical optimization, differentiable programming
   - Combined: Rigorous symbolic reasoning + scalable numerical methods

### Value Proposition

- **Medium Priority (P3)**: Fills important numerical gaps
- **2-3 Week Integration**: Decoupled adapter pattern minimizes effort
- **Zero Modifications**: No changes to NeuroMANCER source required
- **PyTorch Isolation**: Runs in separate conda environment

---

## Technical Implementation

### Decoupled Adapter Pattern

The integration uses a **decoupled adapter pattern** to maintain separation between OpenEvolve and NeuroMANCER:

```
OpenEvolve              Adapter Layer              NeuroMANCER
-----------             --------------             -----------
LeanAide Client  ←→  OptimizationInterface   ←→  (Isolated)
OpenEvolve Core  ←→  NeuroMANCERAdapter      ←→  PyTorch Env
                  ←→  HybridSolver           ←→  (Subprocess)
```

### Key Design Principles

1. **Interface-Based Design**: `OptimizationInterface` defines contract
2. **Environment Isolation**: PyTorch runs in separate conda environment
3. **Subprocess Communication**: No direct dependency on NeuroMANCER code
4. **Serialization-Based**: Problems serialized to JSON, results read back

### Component Architecture

```
integrations/
├── base/
│   └── optimization_interface.py    # Abstract interface
├── neuromancer/
│   ├── adapter.py                   # NeuroMANCER implementation
│   ├── bridge.py                    # LeanAide integration
│   ├── config.yaml                  # Configuration
│   ├── templates/
│   │   ├── ode.yaml                 # ODE problem templates
│   │   ├── pde.yaml                 # PDE problem templates
│   │   └── optimization.yaml        # Optimization templates
│   └── __init__.py                  # Package exports
└── tests/
    └── test_neuromancer_integration.py
```

---

## Architecture

### Hybrid Solver Architecture

The **HybridSolver** combines LeanAide and NeuroMANCER for enhanced problem solving:

```
┌─────────────────────────────────────────────────────────┐
│                    HybridSolver                         │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │   LeanAide       │      │  NeuroMANCER     │        │
│  │  (Symbolic)      │      │  (Numerical)     │        │
│  │                  │      │                  │        │
│  │ • Analysis       │      │ • Optimization   │        │
│  │ • Simplification │      │ • ODE/PDE        │        │
│  │ • Verification   │      │ • System ID      │        │
│  └──────────────────┘      └──────────────────┘        │
│           │                          │                  │
│           └──────────┬───────────────┘                  │
│                      ↓                                  │
│              ┌───────────────┐                          │
│              │   Integration │                          │
│              │    Layer      │                          │
│              └───────────────┘                          │
└─────────────────────────────────────────────────────────┘
                      ↓
              ┌───────────────┐
              │    Result     │
              │  Verification │
              │  & Refinement │
              └───────────────┘
```

### Workflow

1. **Symbolic Analysis** (LeanAide)
   - Simplify constraints
   - Identify redundant constraints
   - Reformulate problem structure

2. **Numerical Optimization** (NeuroMANCER)
   - Solve optimization problem
   - Handle constraints numerically
   - Use gradient-based methods

3. **Verification & Refinement** (Hybrid)
   - Symbolically verify solution
   - Iteratively refine if needed
   - Combine symbolic + numerical insights

---

## Integration Points

### 1. LeanAide Integration

**File**: `integrations/neuromancer/bridge.py`

The bridge connects LeanAide and NeuroMANCER:

```python
from integrations.neuromancer import HybridSolver
from leanaide_client import LeanAideClient

# Create hybrid solver
leanaide = LeanAideClient()
hybrid = HybridSolver(leanaide_client=leanaide)

# Initialize
await hybrid.initialize({
    "leanaide_config": {...},
    "neuromancer_config": {...},
    "hybrid_mode": "sequential"
})

# Solve problem
result = await hybrid.solve_optimization_problem(problem)
```

### 2. MCP Tools Integration

**File**: `leanaide_mcp_tools.py`

Add NeuroMANCER optimization to MCP tools:

```python
@mcp_tool()
async def optimize_with_neuromancer(
    objective: str,
    constraints: List[str],
    variables: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """Solve optimization problem using NeuroMANCER."""
    from integrations.neuromancer import LeanAideNeuroMANCERBridge

    bridge = LeanAideNeuroMANCERBridge()
    await bridge.initialize(config)

    return await bridge.optimize(objective, constraints, variables)
```

### 3. Workflow Integration

**File**: `leanaide_client.py`

Enhance LeanAide client with numerical optimization:

```python
class LeanAideClient:
    def __init__(self):
        self.neuromancer_bridge = None

    async def initialize(self):
        # Initialize NeuroMANCER bridge
        self.neuromancer_bridge = LeanAideNeuroMANCERBridge()
        await self.neuromancer_bridge.initialize(config)

    async def solve_numerical_optimization(self, problem):
        """Delegate numerical optimization to NeuroMANCER."""
        return await self.neuromancer_bridge.optimize(...)
```

---

## Configuration

### Configuration File

**Location**: `integrations/neuromancer/config.yaml`

### Key Configuration Sections

#### 1. Connection Settings

```yaml
connection:
  pytorch_env: neuromancer_env      # Conda environment name
  device: cpu                        # cuda or cpu
  neuromancer_path: null             # Optional path to NeuroMANCER
  max_workers: 4                     # Parallel workers
  timeout: 30                        # Timeout in seconds
```

#### 2. Feature Flags

```yaml
features:
  system_identification: true
  constrained_optimization: true
  differentiable_programming: true
  physics_informed: true
  hybrid_solver: true
```

#### 3. Hybrid Solver Settings

```yaml
hybrid_solver:
  mode: sequential                   # sequential, parallel, adaptive
  max_iterations: 3                  # Refinement iterations
  convergence_tolerance: 1.0e-6
  verify_solutions: true
```

#### 4. Solver Defaults

```yaml
solvers:
  ode:
    method: automatic
    time_step: 0.01
    atol: 1.0e-8
    rtol: 1.0e-6

  pde:
    num_collocation_points: 1000
    hidden_layers: [64, 64, 64]
    activation: tanh

  optimization:
    method: adam
    penalty_parameter: 1000.0
```

### Loading Configuration

```python
import yaml

with open("integrations/neuromancer/config.yaml") as f:
    config = yaml.safe_load(f)

# Use with adapter
adapter = NeuroMANCERAdapter()
await adapter.initialize(config["connection"])
```

---

## Problem Templates

### Template System

Problem templates provide pre-configured examples for common problems:

- **ODE Problems**: `templates/ode.yaml`
- **PDE Problems**: `templates/pde.yaml`
- **Optimization Problems**: `templates/optimization.yaml`

### Using Templates

```python
from integrations.neuromancer import NeuroMANCERAdapter

adapter = NeuroMANCERAdapter()
await adapter.initialize(config)

# Get template
template = await adapter.get_template("harmonic_oscillator")

# Use template as starting point
problem = OptimizationProblem(
    problem_type=ProblemType.ODE,
    **template
)

# Customize and solve
result = await adapter.solve_ode(
    ode_definition=template["ode_definition"],
    initial_conditions=template["initial_conditions"],
    time_span=(0, 20)
)
```

### Available Templates

#### ODE Templates

- `exponential_decay`: First-order linear ODE
- `harmonic_oscillator`: Second-order linear ODE
- `damped_oscillator`: Damped harmonic motion
- `van_der_pol`: Nonlinear oscillator
- `lorenz_system`: Chaotic system
- `sir_model`: Epidemic model

#### PDE Templates

- `heat_equation`: Parabolic diffusion
- `wave_equation`: Hyperbolic wave propagation
- `laplace_equation`: Elliptic steady-state
- `burgers_equation`: Nonlinear shock waves
- `navier_stokes`: Fluid flow
- `reaction_diffusion`: Chemical reactions
- `schrodinger_equation`: Quantum mechanics

#### Optimization Templates

- `quadratic_minimization`: Unconstrained quadratic
- `linear_programming`: Linear programming
- `quadratic_programming`: Quadratic programming
- `rosenbrock`: Non-convex optimization
- `portfolio_optimization`: Finance optimization
- `optimal_control`: Control problems

---

## Usage Examples

### Example 1: Unconstrained Optimization

```python
from integrations.neuromancer import NeuroMANCERAdapter
from integrations.base.optimization_interface import OptimizationProblem, ProblemType

# Initialize adapter
adapter = NeuroMANCERAdapter()
await adapter.initialize({
    "pytorch_env": "neuromancer_env",
    "device": "cpu"
})

# Define problem (Rosenbrock function)
problem = OptimizationProblem(
    problem_type=ProblemType.OPTIMIZATION,
    variables={
        "x": {"initial_value": -1.5, "bounds": (-5, 5)},
        "y": {"initial_value": 2.0, "bounds": (-5, 5)}
    },
    parameters={
        "a": 1.0,
        "b": 100.0
    }
)

# Solve
result = await adapter.solve(problem)

print(f"Optimal value: {result.optimal_value}")
print(f"Optimal variables: {result.optimal_variables}")
print(f"Iterations: {result.iterations}")
```

### Example 2: Constrained Optimization

```python
from integrations.neuromancer import LeanAideNeuroMANCERBridge

# Create bridge
bridge = LeanAideNeuroMANCERBridge()
await bridge.initialize(config)

# Solve constrained problem
result = await bridge.optimize(
    objective="minimize x^2 + y^2",
    constraints=["x + y >= 1", "x >= 0", "y >= 0"],
    variables={
        "x": (0, 10),
        "y": (0, 10)
    },
    use_hybrid=True
)

print(f"Solution: {result}")
```

### Example 3: ODE Solving

```python
# Solve harmonic oscillator
result = await adapter.solve_ode(
    ode_definition={
        "equations": [
            "dx/dt = v",
            "dv/dt = -omega^2 * x"
        ],
        "variables": ["x", "v"],
        "parameters": {"omega": 2.0}
    },
    initial_conditions={
        "x": 1.0,
        "v": 0.0
    },
    time_span=(0, 20),
    method="neural_ode"
)

# Access solution
solution = result["solution"]
time_points = result["time_points"]
```

### Example 4: System Identification

```python
# Identify system from data
input_data = [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]
output_data = [[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]

result = await adapter.identify_system(
    data={
        "inputs": input_data,
        "outputs": output_data
    },
    physics_constraints={
        "conservation_of_mass": True,
        "positivity": True
    }
)

model = result["model"]
metrics = result["metrics"]
print(f"Model parameters: {model}")
print(f"Fit quality: {metrics['r2']}")
```

### Example 5: Hybrid Solver

```python
from integrations.neuromancer import HybridSolver

# Create hybrid solver
hybrid = HybridSolver(leanaide_client=leanaide)
await hybrid.initialize({
    "hybrid_mode": "adaptive",
    "max_iterations": 5
})

# Solve with symbolic analysis + numerical optimization
result = await hybrid.solve_optimization_problem(
    problem=problem,
    symbolic_analysis=True
)

# Check if solution was symbolically verified
if result.metadata.get("symbolically_verified"):
    print("Solution verified by LeanAide!")
```

### Example 6: PDE Solving (Heat Equation)

```python
# Solve heat equation
result = await adapter.solve_pde(
    pde_definition={
        "equation": "∂u/∂t = α∇²u",
        "variables": ["u"],
        "parameters": {"alpha": 0.01}
    },
    boundary_conditions={
        "type": "dirichlet",
        "conditions": {
            "x=0": "u(0,t) = 0",
            "x=1": "u(1,t) = 0"
        }
    },
    initial_conditions={
        "u(x,0)": "sin(πx)"
    },
    domain={
        "type": "interval",
        "bounds": [0, 1]
    }
)

solution = result["solution"]
grid = result["grid"]
```

---

## API Reference

### NeuroMANCERAdapter

Main adapter class for NeuroMANCER optimization.

#### Methods

##### `async initialize(config: Dict[str, Any]) -> bool`

Initialize the adapter with configuration.

**Parameters**:
- `config`: Configuration dictionary
  - `pytorch_env` (str): Conda environment name
  - `device` (str): Device to use ('cpu' or 'cuda')
  - `max_workers` (int): Number of parallel workers
  - `timeout` (int): Timeout in seconds

**Returns**: `True` if successful

**Raises**: `ConfigurationError`, `ConnectionError`

##### `async solve(problem: OptimizationProblem, optimization_type: OptimizationType, solver_params: Dict) -> OptimizationResult`

Solve an optimization problem.

**Parameters**:
- `problem`: Problem definition
- `optimization_type`: Type of optimization
- `solver_params`: Optional solver parameters

**Returns**: `OptimizationResult` object

**Raises**: `ValidationError`, `SolverError`, `TimeoutError`

##### `async solve_ode(ode_definition: Dict, initial_conditions: Dict, time_span: Tuple, method: str) -> Dict`

Solve an ordinary differential equation.

**Parameters**:
- `ode_definition`: ODE system definition
- `initial_conditions`: Initial values
- `time_span`: (t_start, t_end) tuple
- `method`: Solver method

**Returns**: Dictionary with solution, time_points, success

##### `async solve_pde(pde_definition: Dict, boundary_conditions: Dict, initial_conditions: Dict, domain: Dict) -> Dict`

Solve a partial differential equation.

**Parameters**:
- `pde_definition`: PDE definition
- `boundary_conditions`: Boundary conditions
- `initial_conditions`: Initial conditions (optional)
- `domain`: Spatial domain

**Returns**: Dictionary with solution, grid, success

##### `async identify_system(data: Dict, model_structure: Dict, physics_constraints: Dict) -> Dict`

Perform physics-informed system identification.

**Parameters**:
- `data`: Input/output data
- `model_structure`: Optional model structure
- `physics_constraints`: Optional physics constraints

**Returns**: Dictionary with model, metrics, predictions

##### `async validate() -> Dict`

Validate adapter state.

**Returns**: Validation status and metrics

##### `async shutdown() -> bool`

Shutdown adapter and release resources.

### HybridSolver

Combines LeanAide and NeuroMANCER.

#### Methods

##### `async initialize(config: Dict) -> bool`

Initialize hybrid solver.

**Parameters**:
- `config`: Configuration
  - `leanaide_config`: LeanAide config
  - `neuromancer_config`: NeuroMANCER config
  - `hybrid_mode`: 'sequential', 'parallel', or 'adaptive'
  - `max_iterations`: Maximum refinement iterations

##### `async solve_optimization_problem(problem: OptimizationProblem, symbolic_analysis: bool) -> OptimizationResult`

Solve using hybrid approach.

##### `async solve_physics_informed_problem(problem_def: Dict, symbolic_formulation: bool) -> Dict`

Solve physics-informed problem (ODE/PDE/system ID).

##### `async hybrid_optimization_with_constraints(objective: Dict, constraints: List, variables: Dict, verify_solution: bool) -> OptimizationResult`

Constrained optimization with symbolic verification.

### LeanAideNeuroMANCERBridge

High-level convenience interface.

#### Methods

##### `async optimize(objective: str, constraints: List[str], variables: Dict, use_hybrid: bool) -> Dict`

High-level optimization interface.

##### `async solve_differential_equation(equation: str, equation_type: str, conditions: Dict, domain: Dict) -> Dict`

High-level differential equation solver.

##### `async identify_system(input_data: List, output_data: List, physics_constraints: Dict) -> Dict`

High-level system identification.

---

## Testing

### Running Tests

```bash
# Run all NeuroMANCER tests
pytest tests/integrations/test_neuromancer_integration.py -v

# Run specific test
pytest tests/integrations/test_neuromancer_integration.py::test_adapter_init -v

# Run with coverage
pytest tests/integrations/test_neuromancer_integration.py --cov=integrations/neuromancer
```

### Test Structure

```python
import pytest
from integrations.neuromancer import NeuroMANCERAdapter, HybridSolver

@pytest.mark.asyncio
async def test_adapter_init():
    """Test adapter initialization."""
    adapter = NeuroMANCERAdapter()
    result = await adapter.initialize(config)
    assert result is True

@pytest.mark.asyncio
async def test_optimization():
    """Test unconstrained optimization."""
    adapter = NeuroMANCERAdapter()
    await adapter.initialize(config)

    problem = OptimizationProblem(...)
    result = await adapter.solve(problem)

    assert result.success is True
    assert result.optimal_value < initial_value

@pytest.mark.asyncio
async def test_hybrid_solver():
    """Test hybrid solver."""
    hybrid = HybridSolver()
    await hybrid.initialize(config)

    result = await hybrid.solve_optimization_problem(problem)
    assert result.success is True
```

### Test Coverage

The test suite should cover:
- Adapter initialization and validation
- Unconstrained optimization
- Constrained optimization
- ODE solving
- PDE solving
- System identification
- Hybrid solver functionality
- Error handling
- Timeout handling

---

## Troubleshooting

### Common Issues

#### 1. PyTorch Environment Not Found

**Error**: `ConfigurationError: PyTorch environment 'neuromancer_env' not found`

**Solution**:
```bash
# Create conda environment
conda create -n neuromancer_env python=3.9 -y
conda activate neuromancer_env

# Install PyTorch
pip install torch torchvision torchaudio

# Install NeuroMANCER
pip install neuromancer
```

#### 2. CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
- Reduce batch size in config
- Switch to CPU: `device: cpu`
- Use gradient checkpointing

#### 3. Solver Timeout

**Error**: `TimeoutError: Solver exceeded timeout`

**Solution**:
```yaml
# Increase timeout in config
connection:
  timeout: 120  # Increase from 30 to 120 seconds

# Or simplify problem
# - Reduce problem dimensionality
# - Use simpler solver method
# - Relax convergence tolerance
```

#### 4. Import Errors

**Error**: `ModuleNotFoundError: No module named 'neuromancer'`

**Solution**:
- Ensure conda environment is activated
- Install NeuroMANCER: `pip install neuromancer`
- Check environment name in config matches actual environment

#### 5. Process Communication Errors

**Error**: `Solver process failed`

**Solution**:
- Check NeuroMANCER installation
- Verify Python version compatibility
- Check file permissions for temporary directories
- Enable verbose logging:
```yaml
logging:
  level: DEBUG
  verbose: true
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Or in config
logging:
  level: DEBUG
  file: neuromancer_debug.log
  verbose: true
```

### Getting Help

1. Check logs: `neuromancer_debug.log`
2. Validate configuration: `await adapter.validate()`
3. Check NeuroMANCER documentation: https://github.com/pnnl/neuromancer
4. Open issue on OpenEvolve repository

---

## Future Enhancements

### Planned Improvements

1. **Enhanced LeanAide Integration**
   - Symbolic gradient computation
   - Automatic problem reformulation
   - Theorem proving for optimality

2. **Advanced Solvers**
   - Support for more PDE types
   - Adaptive mesh refinement
   - Multi-scale optimization

3. **Performance Optimization**
   - GPU acceleration for PINNs
   - Distributed optimization
   - Caching and memoization

4. **User Experience**
   - Interactive problem builder
   - Visualization tools
   - Real-time monitoring

5. **Extended Templates**
   - More physics problems
   - Industry-specific templates
   - Custom template editor

### Contribution Guidelines

To contribute:

1. Follow the adapter pattern for new integrations
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

---

## References

- **NeuroMANCER Repository**: https://github.com/pnnl/neuromancer
- **NeuroMANCER Documentation**: https://neuromancer.readthedocs.io
- **PNNL**: https://www.pnnl.gov
- **OpenEvolve Documentation**: See main README

---

## Appendix

### A. Configuration Reference

Full configuration options reference (see `config.yaml`)

### B. Template Reference

All available templates and their parameters

### C. Error Codes

Complete list of error codes and meanings

### D. Performance Benchmarks

Expected performance for different problem types

---

**Last Updated**: 2025-01-02
**Version**: 0.1.0
**Maintainer**: OpenEvolve Team
