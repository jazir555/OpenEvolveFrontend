# Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration

A comprehensive integration that connects Microsoft Z3 SMT Solver with LeanAIDE formal verification, OpenEvolve workflow engine, and BubbleLabs visualization platform.

## Overview

This integration provides a unified framework for:
- **Constraint Solving**: Using Z3 for optimization and constraint satisfaction
- **Theorem Proving**: Using both Z3 and LeanAIDE for formal verification
- **Adaptive Problem Solving**: Automatically selecting the best solver for each problem
- **Cross-Validation**: Verifying results with multiple solvers
- **Workflow Integration**: Seamless integration with OpenEvolve's evolutionary workflows
- **Visualization**: Rich UI components in BubbleLabs for tracking and control

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OpenEvolve Workflow                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Z3LeanAideOpenEvolveIntegration                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │  Problem     │  │   Adaptive   │  │   Cross      │   │  │
│  │  │  Classifier  │──│   Solver     │──│ Verification │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │   Z3 Solver      │              │   LeanAIDE       │        │
│  │   (SMT/Constraints)│◄──────────►│   (Theorem Prover)│       │
│  └──────────────────┘              └──────────────────┘        │
│           ▲                                    ▲                │
│           │      Z3LeanAideBridge              │                │
│           │    (Translation, Cross-Validation) │                │
│           ▼                                    ▼                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Z3BubbleLabsUIManager                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │  │
│  │  │Classifier│ │  Z3      │ │  Cross   │ │  Lean    │   │  │
│  │  │   Node   │ │  Solver  │ │  Verify  │ │  Prover  │   │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Z3 Prover Integration (`z3prover_integration.py`)

Core Z3 solver interface providing:
- **Constraint Solving**: SAT/SMT solving for constraint satisfaction problems
- **Theorem Proving**: Proof generation and verification
- **Optimization**: Linear and non-linear optimization
- **SMT-LIB Support**: Full SMT-LIB2 format compatibility

**Key Classes:**
- `Z3SolverEngine`: Main constraint solver
- `Z3TheoremProver`: Theorem proving capabilities
- `Z3ProblemDetector`: Detects Z3-suitable problems

**Example:**
```python
from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType, get_z3_solver_engine

engine = get_z3_solver_engine()

variables = [
    Z3Variable("x", Z3ConstraintType.INTEGER),
    Z3Variable("y", Z3ConstraintType.INTEGER)
]

constraints = [
    Z3Constraint("(> x 0)", Z3ConstraintType.INTEGER),
    Z3Constraint("(< x 10)", Z3ConstraintType.INTEGER),
    Z3Constraint("(= y (+ x 5))", Z3ConstraintType.INTEGER)
]

result = engine.solve_constraints(variables, constraints)
if result.is_sat():
    print(f"Solution: {result.model.assignments}")
```

### 2. Z3-LeanAIDE Bridge (`z3_leanaide_bridge.py`)

Bidirectional integration between Z3 and LeanAIDE:
- **Translation**: SMT-LIB ↔ Lean 4 code conversion
- **Cross-Verification**: Run both solvers and compare results
- **Strategy Selection**: Adaptive solver selection

**Key Classes:**
- `Z3LeanAideBridge`: Main bridge class
- `SMTtoLeanTranslator`: SMT-LIB to Lean 4 translation
- `LeantoSMTTranslator`: Lean 4 to SMT-LIB translation

**Verification Strategies:**
- `Z3_FIRST`: Try Z3, fall back to LeanAIDE
- `LEAN_FIRST`: Try LeanAIDE, fall back to Z3
- `PARALLEL`: Run both concurrently
- `CONSENSUS`: Both must agree
- `ADAPTIVE`: Select based on problem type

**Example:**
```python
from z3_leanaide_bridge import get_z3_leanaide_bridge_sync, VerificationStrategy

bridge = get_z3_leanaide_bridge_sync()

# Translate SMT to Lean
result = await bridge.translate_smt_to_lean(smtlib_content)
print(result.translation)

# Cross-verify
result = await bridge.verify_with_both(problem, VerificationStrategy.PARALLEL)
print(f"Agreement: {result.agreement}, Confidence: {result.confidence_score}")
```

### 3. OpenEvolve Workflow Integration (`z3_leanaide_openevolve_integration.py`)

Full workflow integration:
- **Problem Classification**: Automatic categorization (constraint/theorem/standard)
- **Adaptive Solving**: Route to appropriate solver
- **Enhanced Verification**: Cross-validation for critical problems
- **BubbleLabs Integration**: Workflow visualization

**Key Classes:**
- `Z3LeanAideOpenEvolveIntegration`: Main integration class
- `IntegratedProblemClassifier`: Problem categorization
- `IntegratedSolution`: Unified solution representation

**Problem Categories:**
- `CONSTRAINT_SOLVING`: Use Z3
- `OPTIMIZATION`: Use Z3 optimization
- `THEOREM_PROVING`: Use LeanAIDE
- `SMT_VERIFICATION`: Use Z3 SMT
- `HYBRID`: Use combined approach
- `STANDARD`: Use standard OpenEvolve

**Example:**
```python
from z3_leanaide_openevolve_integration import solve_with_z3_leanaide

result = await solve_with_z3_leanaide("""
    Find x and y where:
    - x > 0 and x < 10
    - y = x + 5
""")

print(f"Classification: {result['classification']['category']}")
print(f"Solution: {result['solution']['content']}")
print(f"Verified: {result['solution']['verification_status']}")
```

### 4. BubbleLabs UI Integration (`z3_leanaide_bubblelabs_ui.py`)

Visual workflow components:
- **Workflow Nodes**: Drag-and-drop solver nodes
- **Real-time Visualization**: Live progress tracking
- **Result Display**: Formatted output display
- **Comparison Views**: Side-by-side Z3/Lean results

**Available Node Types:**
- `z3_problem_classifier`: Classify problems
- `z3_constraint_solver`: Solve constraints
- `z3_theorem_prover`: Prove theorems
- `z3_smt_solver`: SMT-LIB solver
- `z3_leanaide_cross_verify`: Cross-verification

**Example:**
```python
from z3_leanaide_bubblelabs_ui import get_z3_bubblelabs_ui, register_z3_leanaide_bubblelabs_tools

# Register with BubbleLabs
register_z3_leanaide_bubblelabs_tools()

# Use UI manager
ui = get_z3_bubblelabs_ui()
state = await ui.create_classification_node(problem_text)
print(f"Classification: {state.classification}")
```

## Installation

### Prerequisites

1. **Z3 Solver**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install z3

   # macOS
   brew install z3

   # Python bindings
   pip install z3-solver
   ```

2. **LeanAIDE**: See LeanAIDE documentation for setup

3. **OpenEvolve**: Already integrated in your environment

### Setup

The integration modules are automatically available in the OpenEvolve codebase:

```python
# Core Z3 integration
from z3prover_integration import get_z3_solver_engine

# Z3-LeanAIDE bridge
from z3_leanaide_bridge import get_z3_leanaide_bridge_sync

# OpenEvolve integration
from z3_leanaide_openevolve_integration import get_z3_leanaide_openevolve_integration

# BubbleLabs UI
from z3_leanaide_bubblelabs_ui import get_z3_bubblelabs_ui
```

## Usage Examples

### Example 1: Simple Constraint Solving

```python
import asyncio
from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType, get_z3_solver_engine

async def solve_scheduling():
    engine = get_z3_solver_engine()
    
    # Define scheduling problem
    variables = [
        Z3Variable("shift_a", Z3ConstraintType.INTEGER),
        Z3Variable("shift_b", Z3ConstraintType.INTEGER)
    ]
    
    constraints = [
        Z3Constraint("(>= shift_a 1)", Z3ConstraintType.INTEGER),
        Z3Constraint("(<= shift_a 5)", Z3ConstraintType.INTEGER),
        Z3Constraint("(>= shift_b 1)", Z3ConstraintType.INTEGER),
        Z3Constraint("(<= shift_b 5)", Z3ConstraintType.INTEGER),
        Z3Constraint("(>= (+ shift_a shift_b) 6)", Z3ConstraintType.INTEGER)
    ]
    
    result = engine.solve_constraints(variables, constraints)
    
    if result.is_sat():
        print(f"Solution: {result.model.assignments}")
    else:
        print("No solution exists")

asyncio.run(solve_scheduling())
```

### Example 2: Theorem Proving with Cross-Verification

```python
import asyncio
from z3_leanaide_bridge import get_z3_leanaide_bridge_sync, VerificationStrategy

async def prove_and_verify():
    bridge = get_z3_leanaide_bridge_sync()
    
    theorem = """
    (set-logic LIA)
    (declare-fun x () Int)
    (assert (> x 0))
    (assert (not (> (+ x 1) 0)))
    (check-sat)
    """
    
    # Cross-verify with both solvers
    result = await bridge.verify_with_both(
        theorem, 
        VerificationStrategy.CONSENSUS
    )
    
    print(f"Proven: {result.success}")
    print(f"Z3 result: {result.z3_result.status if result.z3_result else 'N/A'}")
    print(f"Agreement: {result.agreement}")
    print(f"Confidence: {result.confidence_score}")

asyncio.run(prove_and_verify())
```

### Example 3: Complete Workflow

```python
import asyncio
from z3_leanaide_openevolve_integration import solve_with_z3_leanaide

async def full_workflow():
    problem = """
    A farmer has 100 acres and wants to plant wheat ($200/acre) and corn ($300/acre).
    Wheat needs 2 hours/acre, corn needs 4 hours/acre. 240 hours available.
    Find optimal allocation.
    """
    
    result = await solve_with_z3_leanaide(problem)
    
    print(f"Category: {result['classification']['category']}")
    print(f"Recommended: {result['classification']['recommended_solver']}")
    print(f"Solution: {result['solution']['content']}")
    print(f"Confidence: {result['solution']['confidence_score']}")

asyncio.run(full_workflow())
```

### Example 4: SMT-LIB to Lean Translation

```python
import asyncio
from z3_leanaide_bridge import get_z3_leanaide_bridge_sync

async def translate_problem():
    bridge = get_z3_leanaide_bridge_sync()
    
    smtlib = """
    (set-logic LIA)
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (> x 0))
    (assert (= y (+ x 5)))
    (check-sat)
    """
    
    # Translate to Lean
    result = await bridge.translate_smt_to_lean(smtlib)
    
    if result.success:
        print("Generated Lean 4 code:")
        print(result.translation)
    
    # Translate back to verify
    smt_result = await bridge.translate_lean_to_smt(result.translation)
    print(f"\nRound-trip successful: {smt_result.success}")

asyncio.run(translate_problem())
```

## Running the Demo

```bash
# Run the comprehensive demo
python demo_z3_leanaide_integration.py
```

This will demonstrate:
1. Z3 constraint solving
2. Z3 theorem proving
3. SMT to Lean translation
4. Combined verification
5. Problem classification
6. Complete integrated workflow
7. BubbleLabs UI nodes

## Running Tests

```bash
# Run all tests
pytest test_z3_leanaide_integration.py -v

# Run specific test category
pytest test_z3_leanaide_integration.py::TestZ3Integration -v
pytest test_z3_leanaide_integration.py::TestZ3LeanAideBridge -v
pytest test_z3_leanaide_integration.py::TestOpenEvolveIntegration -v

# Run integration tests
pytest test_z3_leanaide_integration.py -m integration -v

# Run performance tests
pytest test_z3_leanaide_integration.py -m performance -v
```

## Integration Points

### With OpenEvolve Workflow

The integration hooks into OpenEvolve's workflow stages:

```python
# Stage 1: Problem Classification
classification = integration.classifier.classify(problem)

# Stage 2: Adaptive Solving
solution = await integration._solve_problem(problem, classification, id)

# Stage 3: Verification
verification = await integration._verify_solution(problem, solution, classification)
```

### With BubbleLabs

UI components are registered as BubbleLabs workflow nodes:

```python
# Register nodes
register_z3_leanaide_bubblelabs_tools()

# Available in BubbleLabs UI:
# - z3_problem_classifier
# - z3_constraint_solver
# - z3_theorem_prover
# - z3_leanaide_cross_verify
```

### With LeanAIDE

The bridge provides bidirectional communication:

```python
# Translate and verify
lean_code = await bridge.translate_smt_to_lean(smtlib)
result = await bridge.verify_with_lean(lean_code)

# Cross-validation
combined = await bridge.cross_validate(smtlib)
```

## Configuration

### Z3 Configuration

```python
from z3prover_integration import Z3Config

config = Z3Config(
    timeout=30.0,
    memory_limit_mb=4096,
    num_threads=4,
    proof_generation=True
)

engine = get_z3_solver_engine(config)
```

### Bridge Configuration

```python
from z3_leanaide_bridge import Z3LeanAideConfig

config = Z3LeanAideConfig(
    z3_timeout=30.0,
    leanaide_timeout=300.0,
    default_strategy=VerificationStrategy.ADAPTIVE,
    enable_cross_validation=True
)

bridge = Z3LeanAideBridge(config)
```

### Workflow Configuration

```python
from z3_leanaide_openevolve_integration import WorkflowIntegrationConfig

config = WorkflowIntegrationConfig(
    z3_preference_threshold=0.6,
    lean_preference_threshold=0.6,
    enable_cross_validation=True,
    enable_bubblelabs_visualization=True
)

integration = Z3LeanAideOpenEvolveIntegration(config)
```

## Troubleshooting

### Z3 Not Available

If Z3 is not detected:
```bash
# Check Z3 installation
z3 --version

# Install Python bindings
pip install z3-solver
```

### LeanAIDE Connection Issues

```python
# Check LeanAIDE status
status = bridge.get_status()
print(f"LeanAIDE available: {status['leanaide_available']}")

# Test connection
initialized = await lean_integrator.initialize()
```

### Performance Issues

```python
# Increase timeout
config = Z3Config(timeout=60.0)

# Use simpler strategy
result = await bridge.verify_with_both(problem, VerificationStrategy.Z3_FIRST)
```

## API Reference

See the module docstrings for detailed API documentation:

- `z3prover_integration.py`: Core Z3 interface
- `z3_leanaide_bridge.py`: Bridge functionality
- `z3_leanaide_openevolve_integration.py`: Workflow integration
- `z3_leanaide_bubblelabs_ui.py`: UI components

## License

This integration is part of the OpenEvolve project and follows the same license terms.

## Contributing

To extend this integration:

1. Add new solver types to `Z3ConstraintType`
2. Extend `SMTtoLeanTranslator` for new constructs
3. Add node types to `Z3BubbleLabsUIManager`
4. Update tests in `test_z3_leanaide_integration.py`

## Support

For issues or questions:
- Check the demo: `python demo_z3_leanaide_integration.py`
- Run tests: `pytest test_z3_leanaide_integration.py -v`
- Review logs for detailed error messages
