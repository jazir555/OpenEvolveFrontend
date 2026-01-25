# RESE Developer Guide

**Recursive Epistemic Solvability Engine**
**Version:** 1.0.0
**Last Updated:** 2025-12-31

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Documentation](#module-documentation)
3. [Contribution Guide](#contribution-guide)
4. [Testing](#testing)
5. [Code Style](#code-style)
6. [Performance Optimization](#performance-optimization)
7. [Extending RESE](#extending-rese)

---

## Architecture Overview

### System Architecture

RESE is organized as a **four-phase pipeline** with modular components:

```
┌─────────────────────────────────────────────────────────┐
│                     RESE Pipeline                        │
│  (rese_pipeline.py: RESEPipeline)                       │
└───────────┬───────────────┬───────────────┬────────────┘
            │               │               │
    ┌───────▼─────┐  ┌─────▼──────┐  ┌───▼────────┐
    │  Phase I    │  │  Phase II  │  │ Phase III  │
    │  Epistemic  │→│  Isomorphic│→│  Monte     │→
    │  Audit      │  │  Resonance │  │  Carlo     │
    └───────┬─────┘  └─────┬──────┘  └───┬────────┘
            │               │              │
            └───────────────┴──────────────┴──→
                        ┌───▼───────┐
                        │ Phase IV  │
                        │ Architect.│
                        └───────────┘
```

### Directory Structure

```
rese/
├── core/                      # Phase 0: Core Infrastructure
│   ├── symbolic_constraint_engine.py    # Φ₁: SCE (Agent A1)
│   ├── logic_to_loss_translation.py     # LLTL (Agent A2)
│   ├── dito_optimizer.py                # DITO (Agent A3)
│   └── constraint_*.py                  # Integration modules
│
├── phase1/                   # Phase I: Epistemic Audit
│   ├── tacit_assumption_miner.py        # Φ₁.₅ (Agent B1)
│   ├── cognitive_biases.py              # Φ₂ (Agent B2)
│   └── phi2_integration.py              # Φ₂ Integration
│
├── phase2/                   # Phase II: Isomorphic Resonance
│   ├── imech/                          # I_mech (Agent G3)
│   │   ├── core/                       # Core domain models
│   │   ├── algorithms/                 # Isomorphism algorithms
│   │   └── transfer/                   # Knowledge transfer
│   └── psi3/                           # Ψ₃ (Agent G1)
│
├── phase3/                   # Phase III: Monte Carlo Refinement
│   ├── mcts_search.py                  # Γ₂ (Agent D2)
│   ├── statistical_validator.py        # Γ₃ (Agent D3)
│   └── convergence_controller.py       # N_max (Agent D3)
│
├── phase4/                   # Phase IV: Architectural Synthesis
│   ├── aci_reduction_validator.py      # Δ₃ (Agent E3)
│   └── statistical_tests.py            # Validation tests
│
├── gamma1/                   # Γ₁: ACI Analysis (Agent D1)
│   ├── core/
│   │   ├── aci_calculator.py
│   │   ├── entropy_engine.py
│   │   ├── coherence_engine.py
│   │   └── solvability_engine.py
│   └── signal/                        # Signal extraction
│
├── lean4/                    # Lean 4 formalizations
│   └── scripts/
│
├── tests/                    # Test suites
│   ├── test_core/
│   ├── test_phase1/
│   ├── test_phase2/
│   ├── test_phase3/
│   └── test_gamma1/
│
├── config.py                 # Configuration system
├── rese_pipeline.py          # Main pipeline orchestrator
└── docs/                     # Documentation
```

### Data Flow

```
User Input
    ↓
ProblemInput (description, constraints, variables)
    ↓
┌─────────────────────────────────────────┐
│ Phase I: Epistemic Audit                 │
│  - SCE adds/validates constraints        │
│  - Φ₁.₅ mines tacit assumptions          │
│  - Φ₂ detects cognitive biases           │
│  - Φ₃ resolves contradictions            │
└───────────────┬─────────────────────────┘
                ↓
Validated Constraints + Assumptions
    ↓
┌─────────────────────────────────────────┐
│ Phase II: Isomorphic Resonance           │
│  - Ψ₁ inverts constraints                │
│  - Ψ₂ maps ontologies                    │
│  - Ψ₃/I_mech validates isomorphism       │
└───────────────┬─────────────────────────┘
                ↓
Inverted Constraints + Isomorphisms
    ↓
┌─────────────────────────────────────────┐
│ Phase III: Monte Carlo Refinement        │
│  - Γ₁ calculates ACI                     │
│  - Γ₂ runs MCTS search                   │
│  - Γ₃ validates statistically            │
│  - N_max controls convergence            │
└───────────────┬─────────────────────────┘
                ↓
Best Solutions + ACI History
    ↓
┌─────────────────────────────────────────┐
│ Phase IV: Architectural Synthesis        │
│  - Δ₁ assembles architecture             │
│  - Δ₂ generates predictions              │
│  - Δ₃ validates ACI reduction            │
└───────────────┬─────────────────────────┘
                ↓
Validated Solution
```

---

## Module Documentation

### Core Modules

#### Symbolic Constraint Engine (SCE)

**File:** `core/symbolic_constraint_engine.py`

**Purpose:** Foundation for all RESE phases - manages constraints and dependencies.

**Key Classes:**
- `Constraint`: Represents a formal constraint
- `SymbolicConstraintEngine`: Manages constraint collection

**Example:**
```python
from core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)

# Create engine
sce = SymbolicConstraintEngine()

# Add constraint
constraint = Constraint(
    id="c1",
    type=ConstraintType.HARD,
    description="All variables must be positive",
    formalization="∀ x ∈ variables: x > 0",
    source="user"
)
sce.add_constraint(constraint)

# Detect conflicts
conflicts = sce.detect_conflicts()

# Get execution order
order = sce.get_execution_order()
```

**Design Patterns:**
- **Repository Pattern:** Centralized constraint storage
- **Observer Pattern:** Dependency tracking via NetworkX
- **Cache Pattern:** Conflict detection caching

---

#### Logic to Loss Translation (LLTL)

**File:** `core/logic_to_loss_translation.py`

**Purpose:** Translates formal logic constraints to differentiable loss functions.

**Key Classes:**
- `LogicToLossTranslator`: Converts logic to loss
- `LossFunction`: Generated loss function

**Example:**
```python
from core.logic_to_loss_translation import LogicToLossTranslator

# Create translator
translator = LogicToLossTranslator()

# Translate constraint to loss
loss_fn = translator.translate(
    logic="∀ x: x > 0",
    variables={"x": torch.tensor([1, 2, 3])}
)

# Compute loss
loss_value = loss_fn()
loss_value.backward()  # Gradient for optimization
```

**Algorithm:**
1. Parse formal logic (Lean 4 syntax)
2. Identify logical connectives (∀, ∃, ∧, ∨, ¬)
3. Map to differentiable operators:
   - `∀` → min over domain
   - `∃` → max over domain
   - `∧` → multiplication
   - `∨` → addition
   - `¬` → negation
4. Add relaxation for strict inequalities
5. Return PyTorch-compatible loss function

---

#### DITO Optimizer

**File:** `core/dito_optimizer.py`

**Purpose:** O(n log n) contradiction detection in large constraint sets.

**Key Classes:**
- `DITOOptimizer`: Main optimizer
- `KnowledgeGraph`: Constraint dependency graph

**Example:**
```python
from core.dito_optimizer import DITOOptimizer

# Create optimizer
optimizer = DITOOptimizer()

# Add constraints (can handle 10K+)
for constraint in large_constraint_set:
    optimizer.add_constraint(constraint)

# Detect contradictions in O(n log n)
contradictions = optimizer.detect_contradictions()

# Get statistics
stats = optimizer.get_statistics()
print(f"Checked {stats['num_constraints']} in {stats['time']:.2f}s")
```

**Complexity:**
- Building graph: O(n log n)
- Contradiction detection: O(n log n)
- Space: O(n)

**Algorithm:**
1. Build knowledge graph with constraint dependencies
2. Topological sort for execution order
3. Detect cycles (circular dependencies)
4. Check for logical contradictions at each node
5. Use memoization for repeated checks

---

### Phase I Modules

#### Tacit Assumption Miner (Φ₁.₅)

**File:** `phase1/tacit_assumption_miner.py`

**Purpose:** Discover hidden constraints from null results.

**Key Classes:**
- `TacitAssumptionMiner`: Main miner
- `Assumption`: Discovered assumption

**Example:**
```python
from phase1.tacit_assumption_miner import TacitAssumptionMiner

miner = TacitAssumptionMiner()

# Mine from failure cases
assumptions = miner.mine(
    failure_cases=known_failures,
    constraints=existing_constraints,
    num_assumptions=10
)

for assumption in assumptions:
    print(f"{assumption.description} (confidence: {assumption.confidence:.2f})")
```

**Algorithm:**
1. Collect failure cases (unsolvable instances)
2. Extract common patterns in failures
3. Identify "near misses" (almost solved)
4. Infer hidden constraints from patterns
5. Validate with statistical tests
6. Rank by confidence

---

#### Cognitive Bias Detector (Φ₂)

**File:** `phase1/cognitive_biases.py`

**Purpose:** Detect and mitigate cognitive biases in problem formulation.

**Key Classes:**
- `CognitiveBiasDetector`: Detects biases
- `Debiaser`: Applies debiasing interventions

**Example:**
```python
from phase1.cognitive_biases import CognitiveBiasDetector

detector = CognitiveBiasDetector()

# Analyze constraints
report = detector.analyze_constraints(constraints)

print(f"Bias score: {report.overall_bias_score:.2f}")
for detection in report.detections:
    print(f"{detection.bias_type.value}: {detection.description}")
```

**Bias Types Detected:**
- Confirmation bias
- Anchoring bias
- Availability bias
- Sunk cost fallacy
- Framing effect
- Overconfidence effect
- Dunning-Kruger effect
- Authority bias
- Clustering illusion
- Texas sharpshooter fallacy

---

### Phase II Modules

#### I_mech Isomorphism Validator

**File:** `phase2/imech/`

**Purpose:** Validate mechanistic similarity between domains for knowledge transfer.

**Key Classes:**
- `IMechValidator`: Main validator
- `Domain`: Problem domain representation
- `MechanisticStructure`: Causal structure
- `IsomorphismResult`: Comparison result

**Example:**
```python
from phase2.imech import IMechValidator, Domain

# Define domains
source = Domain(
    id="tsp",
    name="Traveling Salesman",
    variables=...,
    constraints=...
)

target = Domain(
    id="vrp",
    name="Vehicle Routing",
    variables=...,
    constraints=...
)

# Compare
validator = IMechValidator()
result = validator.compare_domains(source, target)

print(f"Similarity: {result.score:.2f}")
print(f"Confidence: {result.confidence:.2f}")

# Transfer if similar
if result.score > 0.7:
    transferred = validator.transfer_knowledge(source, target)
```

**Algorithms:**
1. **Weisfeiler-Lehman:** Graph isomorphism testing
2. **VF2:** Subgraph isomorphism
3. **Interventional Testing:** Causal structure validation

---

#### Constraint Inverter (Ψ₃)

**File:** `phase2/psi3/`

**Purpose:** Invert constraints to reduce search complexity (2^n → 2^(n/10)).

**Key Classes:**
- `ConstraintInverter`: Main inverter
- `InvertedConstraint`: Inverted constraint

**Example:**
```python
from phase2.psi3 import ConstraintInverter

inverter = ConstraintInverter()

# Invert constraints
inverted = inverter.invert(constraints)

print(f"Original complexity: 2^{len(constraints)}")
print(f"Inverted complexity: 2^{len(inverted)}")
```

**Algorithm:**
1. Identify constraint dependencies
2. Find "bottleneck" constraints (high dependency count)
3. Invert: replace "what must be true" with "what cannot be false"
4. Merge redundant constraints
5. Result: 10x fewer independent constraints

---

### Phase III Modules

#### ACI Calculator (Γ₁)

**File:** `gamma1/core/aci_calculator.py`

**Purpose:** Calculate Algorithmic Complexity Index for CSP instances.

**Key Classes:**
- `ACICalculator`: Main calculator
- `ACIResult`: Calculation result

**Example:**
```python
from gamma1.core.aci_calculator import ACICalculator
from gamma1.core.csp_models import CSPInstance

# Create CSP instance
csp = CSPInstance(
    variables=...,
    domains=...,
    constraints=...
)

# Calculate ACI
aci_calc = ACICalculator()
result = aci_calc.calculate(csp)

print(f"ACI = {result.ACI:.3f}")
print(f"Components: {result.components}")
print(f"Confidence: {result.confidence:.2f}")
```

**Formula:**
```
ACI = α·(1-H) + β·C + γ·S

Where:
- H = Disorder Entropy (measured by entropy_engine.py)
- C = Causal Coherence (measured by coherence_engine.py)
- S = Solvability Index (measured by solvability_engine.py)
```

---

#### MCTS Search (Γ₂)

**File:** `phase3/mcts_search.py`

**Purpose:** ACI-guided Monte Carlo Tree Search.

**Key Classes:**
- `MCTSSearch`: Main search
- `MCTSNode`: Search tree node
- `MCTSResult`: Search result

**Example:**
```python
from phase3.mcts_search import MCTSSearch
from gamma1.core.aci_calculator import ACICalculator

# Create ACI calculator for guidance
aci_calc = ACICalculator()

# Create search
search = MCTSSearch(
    aci_calculator=aci_calc,
    iterations=1000,
    exploration_constant=1.41,
    parallel_agents=4
)

# Search
result = search.search(initial_state)

print(f"Best value: {result.best_value}")
print(f"ACI history: {result.aci_history}")
```

**Algorithm:**
1. **Selection:** Use ACI-guided UCB to select node
2. **Expansion:** Add child node if not terminal
3. **Simulation:** Run random playout to termination
4. **Backpropagation:** Update statistics up the tree
5. **Repeat** for N iterations
6. **Return** best node found

---

#### Convergence Controller (N_max)

**File:** `phase3/convergence_controller.py`

**Purpose:** Control MCTS convergence with early stopping.

**Key Classes:**
- `ConvergenceController`: Main controller
- `ConvergenceCriterion`: Stopping criterion

**Example:**
```python
from phase3.convergence_controller import ConvergenceController

controller = ConvergenceController(
    patience=50,
    min_delta=0.001,
    max_iterations=10000
)

# Check convergence during MCTS
for iteration in range(10000):
    # ... run MCTS iteration ...

    if controller.should_stop(current_value):
        print(f"Converged at iteration {iteration}")
        break
```

**Stopping Criteria:**
1. **Plateau:** No improvement for `patience` iterations
2. **Min Delta:** Improvement < `min_delta`
3. **Max Iterations:** Reach `max_iterations`
4. **ACI Saturation:** ACI stops improving

---

### Phase IV Modules

#### ACI Reduction Validator (Δ₃)

**File:** `phase4/aci_reduction_validator.py`

**Purpose:** Validate solution by ACI reduction (non-circular validation).

**Key Classes:**
- `Delta3Validator`: Main validator
- `ValidationResult`: Validation result

**Example:**
```python
from phase4.aci_reduction_validator import Delta3Validator

validator = Delta3Validator(
    validation_threshold=0.7,
    min_aci_reduction=0.2
)

# Validate solution
result = validator.validate(problem, solution)

print(f"Valid: {result.is_valid}")
print(f"Score: {result.validation_score:.2f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"ACI reduction: {result.aci_reduction:.2f}")
```

**Validation Strategy:**
1. Split data into train/validation/test sets
2. Measure ACI on training set (before solution)
3. Apply solution
4. Measure ACI on validation set (after solution)
5. Require: ACI_validation < ACI_train - min_reduction
6. Test on holdout set for final validation

**Non-Circular:**
- Doesn't use same data for training and validation
- Holdout set prevents overfitting
- Statistical significance testing

---

## Contribution Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/rese.git
cd rese

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Review Process

1. **Fork** the repository
2. **Create branch** for your feature
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make changes** with tests
4. **Run tests**
   ```bash
   pytest tests/
   ```
5. **Format code**
   ```bash
   black rese/
   isort rese/
   ```
6. **Commit** with clear message
   ```bash
   git commit -m "Add: Feature for X"
   ```
7. **Push** to your fork
   ```bash
   git push origin feature/my-feature
   ```
8. **Create Pull Request** on GitHub

### Commit Message Convention

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `Add:` New feature
- `Fix:` Bug fix
- `Refactor:` Code restructuring
- `Docs:` Documentation update
- `Test:` Test addition/modification
- `Perf:` Performance improvement

**Example:**
```
Add: ACI-guided MCTS selection policy

Implements ACI-based node selection in MCTS to focus
search on high-solvability regions.

Closes #123
```

---

## Testing

### Test Structure

```
tests/
├── test_core/
│   ├── test_symbolic_constraint_engine.py
│   ├── test_logic_to_loss_translation.py
│   └── test_dito_optimizer.py
├── test_phase1/
│   ├── test_tacit_assumption_miner.py
│   └── test_cognitive_biases.py
├── test_phase2/
│   ├── test_imech.py
│   └── test_psi3.py
├── test_phase3/
│   ├── test_aci_calculator.py
│   ├── test_mcts_search.py
│   └── test_convergence_controller.py
└── test_phase4/
    └── test_aci_reduction_validator.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core/test_symbolic_constraint_engine.py

# Run with coverage
pytest --cov=rese --cov-report=html tests/

# Run specific test
pytest tests/test_core/test_sce.py::test_add_constraint
```

### Writing Tests

```python
import pytest
from core.symbolic_constraint_engine import (
    SymbolicConstraintEngine,
    Constraint,
    ConstraintType
)

def test_add_constraint():
    """Test adding constraint to engine"""
    sce = SymbolicConstraintEngine()
    constraint = Constraint(
        id="test_c1",
        type=ConstraintType.HARD,
        description="Test constraint",
        formalization="x > 0",
        source="test"
    )

    sce.add_constraint(constraint)

    assert "test_c1" in sce.constraints
    assert sce.get_constraint("test_c1") == constraint

def test_conflict_detection():
    """Test conflict detection"""
    sce = SymbolicConstraintEngine()

    # Add conflicting constraints
    c1 = Constraint("c1", ConstraintType.HARD, "x > 0", "x > 0", "test")
    c2 = Constraint("c2", ConstraintType.HARD, "x < 0", "x < 0", "test")

    sce.add_constraint(c1)
    sce.add_constraint(c2)

    conflicts = sce.detect_conflicts()

    assert len(conflicts) == 1
    assert conflicts[0] == ("c1", "c2")
```

### Test Coverage

**Target:** >80% coverage

**Current coverage:** Run with `--cov` flag

**Check coverage:**
```bash
pytest --cov=rese --cov-report=term-missing
```

---

## Code Style

### Python Style Guide

RESE follows **PEP 8** with modifications:

1. **Line Length:** 100 characters (not 79)
2. **Imports:** Group by stdlib, third-party, local
3. **Docstrings:** Google style
4. **Type Hints:** Required for all public functions

### Formatting

**Use Black:**
```bash
black rese/ --line-length=100
```

**Use isort:**
```bash
isort rese/ --profile black
```

### Linting

**Use pylint:**
```bash
pylint rese/ --max-line-length=100
```

**Use flake8:**
```bash
flake8 rese/ --max-line-length=100 --extend-ignore=E203
```

### Docstring Style

```python
def calculate_aci(
    self,
    csp_instance: CSPInstance,
    use_cache: bool = True
) -> ACIResult:
    """
    Calculate Algorithmic Complexity Index for CSP instance.

    ACI = α·(1-H) + β·C + γ·S

    Args:
        csp_instance: CSP instance to analyze
        use_cache: Whether to use cached results

    Returns:
        ACIResult with ACI score and components

    Raises:
        ValueError: If CSP instance is invalid

    Example:
        >>> calc = ACICalculator()
        >>> result = calc.calculate(my_csp)
        >>> print(f"ACI = {result.ACI:.3f}")
    """
```

---

## Performance Optimization

### Profiling

```python
import cProfile
import pstats

# Profile code
profiler = cProfile.Profile()
profiler.enable()

# ... run code ...

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)  # Top 10 functions
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile function
python -m memory_profiler rese/core/symbolic_constraint_engine.py
```

### Optimization Tips

**1. Use Caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(x, y):
    # Expensive computation
    return result
```

**2. Vectorize Operations:**
```python
import numpy as np

# Slow: Python loop
result = [x**2 for x in data]

# Fast: NumPy vectorization
result = np.square(data)
```

**3. Use Generators:**
```python
# Memory-intensive: List
def get_all_items():
    return [item for item in large_dataset]

# Memory-efficient: Generator
def get_all_items():
    for item in large_dataset:
        yield item
```

**4. Parallel Processing:**
```python
from multiprocessing import Pool

def process_item(item):
    # Process item
    return result

with Pool(processes=4) as pool:
    results = pool.map(process_item, items)
```

### Performance Benchmarks

**Target Performance:**

| Module | Operation | Target |
|--------|-----------|--------|
| SCE | Add constraint | <1ms |
| SCE | Detect conflicts | <100ms for 10K constraints |
| DITO | Detect contradictions | <1s for 10K constraints |
| Φ₁.₅ | Mine assumptions | <10s for 100 failures |
| I_mech | Compare domains | <5s for 1K variables |
| Γ₁ | Calculate ACI | <500ms per instance |
| Γ₂ | MCTS iteration | <1ms per iteration |
| Δ₃ | Validate solution | <5s |

---

## Extending RESE

### Adding a New Phase Component

```python
# 1. Create new component file
# rese/phaseX/my_component.py

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MyComponentResult:
    """Result from my component"""
    value: float
    metadata: Dict[str, Any]

class MyComponent:
    """My custom component"""

    def __init__(self, param1: float, param2: str):
        self.param1 = param1
        self.param2 = param2

    def execute(self, input_data: Any) -> MyComponentResult:
        """Execute component logic"""
        # Implementation here
        return MyComponentResult(
            value=0.5,
            metadata={}
        )

# 2. Add to phase executor
# rese/rese_pipeline.py

class PhaseXExecutor(PhaseExecutor):
    """Phase X: My Custom Phase"""

    def execute(self, input_data: Any) -> PhaseResult:
        from phaseX.my_component import MyComponent

        component = MyComponent(
            param1=self.config.phaseX.param1,
            param2=self.config.phaseX.param2
        )

        result = component.execute(input_data)

        return PhaseResult(
            phase_name="phaseX",
            status=PhaseStatus.COMPLETED,
            output=result
        )

# 3. Add configuration
# rese/config.py

@dataclass
class PhaseXConfig:
    """Configuration for Phase X"""
    param1: float = 0.5
    param2: str = "default"

# 4. Add tests
# tests/test_phaseX/test_my_component.py
```

### Adding Custom Bias Detectors

```python
# 1. Define bias type
# rese/phase1/cognitive_biases.py

class BiasType(Enum):
    # ... existing types ...
    MY_CUSTOM_BIAS = "my_custom_bias"

# 2. Create detector
class MyCustomBiasDetector:
    """Detector for my custom bias"""

    def detect(self, constraints: List[Constraint]) -> List[BiasDetection]:
        detections = []

        for constraint in constraints:
            if self._has_bias(constraint):
                detections.append(BiasDetection(
                    bias_type=BiasType.MY_CUSTOM_BIAS,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    description="Detected custom bias",
                    suggestion="Remove bias"
                ))

        return detections

    def _has_bias(self, constraint: Constraint) -> bool:
        # Detection logic
        return False

# 3. Register detector
# In CognitiveBiasDetector.__init__
self.detectors.append(MyCustomBiasDetector())
```

### Adding Custom ACI Components

```python
# 1. Create component
# rese/gamma1/core/my_component.py

class MyACIComponent:
    """Custom ACI component"""

    def calculate(self, csp_instance: CSPInstance) -> float:
        """
        Calculate custom ACI component.

        Returns:
            Value in [0, 1] where higher = better
        """
        # Implementation
        return 0.5

# 2. Integrate into ACI calculator
# rese/gamma1/core/aci_calculator.py

class ACICalculator:
    def __init__(self, ..., use_my_component: bool = True):
        # ...
        if use_my_component:
            self.my_component = MyACIComponent()

    def calculate(self, csp_instance: CSPInstance) -> ACIResult:
        # ... existing components ...

        if hasattr(self, 'my_component'):
            my_value = self.my_component.calculate(csp_instance)

        # Update ACI formula
        ACI = (alpha * (1-H) + beta * C +
               gamma * S + delta * my_value)
```

### Creating Custom Integration Points

```python
# rese/core/custom_integration.py

class CustomIntegration:
    """Custom integration between phases"""

    def __init__(self, config: RESEConfig):
        self.config = config

    def integrate(
        self,
        phase1_output: Any,
        phase2_output: Any
    ) -> Any:
        """
        Integrate outputs from multiple phases.

        Args:
            phase1_output: Output from Phase I
            phase2_output: Output from Phase II

        Returns:
            Integrated result
        """
        # Integration logic
        return integrated_result
```

---

## Resources

### Internal Documentation
- [User Guide](user_guide.md)
- [API Reference](api_reference.md)
- [Integration Guide](e2e_integration.md)
- [Troubleshooting](troubleshooting.md)

### External References
- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

### Research Papers
- RESE Theoretical Foundation: See project README
- ACI Paper: `rese/docs/gamma1_aci_research.md`
- I_mech Paper: `rese/docs/imech_isomorphism_research.md`

---

**Happy Coding! 🚀**
