<<<<<<< HEAD
# RESE Developer Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Component Interaction](#component-interaction)
4. [Extension Points](#extension-points)
5. [Contribution Guidelines](#contribution-guidelines)
6. [Testing Guidelines](#testing-guidelines)
7. [Code Style Guide](#code-style-guide)
8. [Performance Optimization](#performance-optimization)
9. [Debugging](#debugging)
10. [Release Process](#release-process)

---

## Introduction

### Purpose of This Guide

This guide is for developers who want to:
- Contribute to RESE core codebase
- Extend RESE with custom components
- Understand RESE architecture in depth
- Optimize RESE performance
- Debug RESE issues

### Developer Prerequisites

**Required:**
- Python 3.9+
- Strong understanding of data structures and algorithms
- Familiarity with graph theory (for I_mech)
- Knowledge of statistical methods (for Phase III)
- Experience with formal logic (for Phase IV)

**Recommended:**
- Familiarity with Lean 4 or similar proof assistants
- Experience with MCTS algorithms
- Knowledge of constraint satisfaction problems
- Understanding of functional dependency graphs

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RESE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API LAYER                                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│  │  │ REST API │  │ WebSocket│  │ Python   │           │   │
│  │  └──────────┘  └──────────┘  │ API      │           │   │
│  │                              └──────────┘           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PIPELINE ORCHESTRATION                   │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │  RESEPipeline                                 │     │   │
│  │  │  - Phase orchestration                       │     │   │
│  │  │  - Progress tracking                         │     │   │
│  │  │  - Caching                                    │     │   │
│  │  │  - Error handling                            │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PHASE EXECUTORS                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │ Phase I │  │ Phase II│  │ Phase III│ │ Phase IV│ │   │
│  │  │ Executor│  │ Executor│  │ Executor│ │ Executor│ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CORE ALGORITHMS                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ SCE (Φ₁) │ │ Φ₁.₅     │ │ I_mech   │             │   │
│  │  └──────────┘ └──────────┘ └──────────┘             │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ MCTS (Γ₂)│ │ ACI (Γ₁) │ │ Δ₃       │             │   │
│  │  └──────────┘ └──────────┘ └──────────┘             │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              INFRASTRUCTURE                           │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ Config   │ │ Monitoring│ │ Cache    │             │   │
│  │  │ System   │ │ System   │ │ Manager  │             │   │
│  │  └──────────┘ └──────────┘ └──────────┘             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
rese/
├── __init__.py
├── rese_pipeline.py          # Main pipeline orchestrator
├── config.py                 # Configuration system
├── api.py                    # REST API
├── monitoring.py             # Monitoring system
├── quickstart.py             # Quick start script
│
├── core/                     # Core algorithms (Phase I)
│   ├── symbolic_constraint_engine.py    # Φ₁: SCE
│   ├── constraint_stage1_integration.py
│   ├── constraint_lean4_bridge.py
│   ├── dito_graphs.py
│   ├── constraint_optimizer.py
│   └── constraint_lltl_handoff.py
│
├── phase1/                   # Phase I: Epistemic Audit
│   ├── cognitive_biases.py   # Φ₂: Bias detection
│   ├── phi2_integration.py   # Φ₁-SCE integration
│   ├── tacit_assumption_miner.py  # Φ₁.₅: Assumption mining
│   ├── failure_database.py   # Historical failure patterns
│   └── tests/
│       ├── test_cognitive_biases.py
│       ├── test_phi2_integration.py
│       └── test_tacit_assumption_miner.py
│
├── phase2/                   # Phase II: Isomorphic Resonance
│   ├── imech/                # I_mech: Mechanistic Isomorphism
│   │   ├── __init__.py
│   │   ├── isomorphism_validator.py
│   │   ├── core/
│   │   │   ├── fdg.py        # Functional Dependency Graphs
│   │   │   ├── domain.py     # Domain representation
│   │   │   ├── result.py     # Similarity results
│   │   │   ├── causality.py  # Causal structure
│   │   │   └── scoring.py    # Scoring algorithms
│   │   ├── algorithms/
│   │   │   ├── weisfeiler_lehman.py
│   │   │   ├── vf2.py
│   │   │   ├── subgraph.py
│   │   │   └── intervention.py
│   │   ├── transfer/
│   │   │   ├── mapper.py     # Solution mapping
│   │   │   ├── validator.py  # Transfer validation
│   │   │   └── repair.py     # Solution repair
│   │   ├── lean4/
│   │   │   └── proof_generator.py
│   │   └── tests/
│   │
│   ├── psi3/                 # Ψ₃: Isomorphism Validation
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── constraint.py
│   │   │   │   ├── constraint_inverter.py
│   │   │   │   └── expression.py
│   │   │   ├── algorithms/
│   │   │   │   ├── preprocessing.py
│   │   │   │   └── dependency_analyzer.py
│   │   │   └── solvers/
│   │   │       └── sat_wrapper.py
│   │   ├── tests/
│   │   └── examples/
│   │
│   ├── ontology_components/  # Ψ₂: Ontology Mapping
│   │   ├── semantic_matcher.py
│   │   ├── lexical_matcher.py
│   │   ├── graph_embedder.py
│   │   ├── kg_validator.py
│   │   └── __init__.py
│   │
│   ├── ontology_imech_integration.py
│   └── test_phase2_debug.py
│
├── phase3/                   # Phase III: Monte Carlo Refinement
│   ├── stage3_integration.py # MCTS integration
│   ├── mcts_search.py        # Γ₂: MCTS implementation
│   ├── convergence_controller.py  # N_max: Convergence
│   ├── aci_analyzer/         # Γ₁: ACI Analysis
│   │   └── aci_analyzer.py
│   ├── statistical_validator.py   # Γ₃: Statistical validation
│   ├── tests/
│   │   ├── test_mcts_search.py
│   │   ├── test_convergence_controller.py
│   │   └── test_statistical_validator.py
│   │
│   └── gamma1/               # ACI Calculator
│       ├── core/
│       │   ├── aci_calculator.py
│       │   └── csp_models.py
│       └── signal/
│           └── validator.py
│
├── phase4/                   # Phase IV: Architectural Synthesis
│   ├── architecture_assembler.py   # Δ₁: Assembly
│   ├── assembly_validator.py
│   ├── predictive_model_generator.py  # Δ₂: Prediction
│   ├── aci_reduction_validator.py    # Δ₃: Validation
│   ├── phase_transition.py
│   ├── independence_checker.py
│   ├── tests/
│   │   ├── test_architecture_assembler.py
│   │   ├── test_predictive_model_generator.py
│   │   └── test_aci_reduction_validator.py
│   │
│   └── stage8_integration.py
│
├── performance/              # Performance optimization
│   ├── optimizer.py
│   ├── sce_optimizer.py
│   ├── cache_manager.py
│   └── sce_optimizer.py
│
├── security/                 # Security and validation
│   ├── input_validator.py
│   ├── error_handler.py
│   ├── resource_limiter.py
│   ├── security_audit.py
│   └── security_tests.py
│
├── integrations/             # E2E Stage integrations
│   ├── stage1.py             # Stage 1 integration
│   ├── stage2.py             # Stage 2 integration
│   ├── stage3.py             # Stage 3 integration
│   ├── stage5.py             # Stage 5 integration
│   ├── stage6.py             # Stage 6 integration
│   ├── stage7.py             # Stage 7 integration
│   ├── stage8.py             # Stage 8 integration
│   ├── stage9.py             # Stage 9 integration
│   ├── validate_integrations.py
│   └── test_e2e_pipeline.py
│
├── tests/                    # Integration tests
│   ├── test_integration/
│   ├── test_performance/
│   ├── test_validation/
│   └── phase1/
│
└── lean4/                    # Lean 4 integration
    └── scripts/
        └── export_proofs.py
```

---

## Component Interaction

### Phase Execution Flow

```
User Request
    │
    ▼
┌──────────────────┐
│ API Layer        │ → Validates request, creates ProblemInput
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ RESEPipeline     │ → Creates phase executors, manages state
└──────────────────┘
    │
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
┌──────────────────┐                          ┌──────────────────┐
│ Phase1Executor   │                          │ CacheManager     │
└──────────────────┘                          └──────────────────┘
    │                                                 │
    ├──────────────┬──────────────┬──────────────┐    │
    ▼              ▼              ▼              ▼    ▼
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐ Check cache
│  SCE   │   │ Φ₁.₅   │   │  Φ₂    │   │  Φ₃    │────────────┐
└────────┘   └────────┘   └────────┘   └────────┘            │
    │              │              │              │            │
    └──────────────┴──────────────┴──────────────┘            │
                           │                                 │
                           ▼                                 ▼
                    PhaseResult                        Return cached
                           │                                 result
                           ▼
                    ┌──────────────┐
                    │ Phase2Executor│
                    └──────────────┘
                           │
                    (Similar structure for all phases)
                           │
                           ▼
                    PipelineResult
```

### Data Structures

#### ProblemInput
```python
@dataclass
class ProblemInput:
    id: str
    description: str
    constraints: List[Dict[str, Any]]
    variables: Dict[str, Any]
    objective: Optional[str]
    domain: str
    metadata: Dict[str, Any]
```

#### PhaseResult
```python
@dataclass
class PhaseResult:
    phase_name: str
    status: PhaseStatus
    output: Any
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    elapsed_seconds: float
```

#### PipelineResult
```python
@dataclass
class PipelineResult:
    pipeline_id: str
    problem_id: str
    status: PipelineStatus
    phase_results: Dict[str, PhaseResult]
    final_solution: Optional[Dict[str, Any]]
    aci_history: List[float]
    validation_score: float
    confidence: float
    start_time: datetime
    end_time: Optional[datetime]
    elapsed_seconds: float
```

---

## Extension Points

### 1. Custom Phase Executors

Extend `PhaseExecutor` base class:

```python
from rese.rese_pipeline import PhaseExecutor, PhaseResult

class MyCustomExecutor(PhaseExecutor):
    def execute(self, input_data: Any) -> PhaseResult:
        start_time = datetime.now()
        result = PhaseResult(
            phase_name="my_custom_phase",
            status=PhaseStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Your custom logic here
            output = self._custom_processing(input_data)

            result.output = output
            result.metrics = {'custom_metric': 42}
            result.status = PhaseStatus.COMPLETED

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now()
        result.elapsed_seconds = (result.end_time - start_time).total_seconds()

        return result

    def _custom_processing(self, input_data):
        # Implement your custom phase logic
        return {"result": "processed"}

# Register custom phase
pipeline = RESEPipeline()
pipeline.phase_executors['custom'] = MyCustomExecutor("custom", config)
```

### 2. Custom ACI Calculators

```python
from gamma1.core.aci_calculator import ACICalculator

class MyCustomACICalculator(ACICalculator):
    def calculate_solution(
        self,
        constraints: List[Constraint],
        solution_variables: Dict[str, Any],
        domain: str
    ) -> float:
        # Base ACI calculation
        base_aci = super().calculate_solution(
            constraints, solution_variables, domain
        )

        # Add custom factors
        custom_factor = self._calculate_custom_factor(
            constraints, solution_variables
        )

        return base_aci * custom_factor

    def _calculate_custom_factor(self, constraints, solution_variables):
        # Your custom logic
        return 1.0

# Use custom calculator
config.phase3.gamma2_aci_calculator = MyCustomACICalculator()
```

### 3. Custom Isomorphism Algorithms

```python
from phase2.imech.algorithms import WeisfeilerLehman

class MyCustomAlgorithm(WeisfeilerLehman):
    def compare(
        self,
        graph1: FunctionalDependencyGraph,
        graph2: FunctionalDependencyGraph
    ) -> float:
        # Custom comparison logic
        similarity = super().compare(graph1, graph2)

        # Add domain-specific adjustments
        adjusted_similarity = self._adjust_for_domain(
            similarity, graph1, graph2
        )

        return adjusted_similarity

    def _adjust_for_domain(self, similarity, graph1, graph2):
        # Your custom domain adjustment
        return similarity

# Register custom algorithm
from phase2.imech import IMechValidator
IMechValidator.register_algorithm('custom', MyCustomAlgorithm)
```

### 4. Custom MCTS Policies

```python
from phase3.mcts_search import MCTSSearch

class MyCustomMCTS(MCTSSearch):
    def _selection_policy(self, node):
        # Custom selection policy
        # Default: UCB = Q/N + C * sqrt(ln(N_parent)/N)

        # Your custom policy
        custom_score = self._custom_selection_score(node)
        return custom_score

    def _custom_selection_score(self, node):
        # Your logic
        return node.ucb_score

# Use custom MCTS
mcts = MyCustomMCTS(
    exploration_constant=1.41,
    max_iterations=1000,
    aci_guided=True
)
```

---

## Contribution Guidelines

### Code Review Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write Code**
   - Follow code style guide
   - Add tests
   - Update documentation

3. **Test**
   ```bash
   pytest rese/tests/
   pytest rese/tests/phase1/
   pytest rese/tests/phase2/imech/
   ```

4. **Submit Pull Request**
   - Describe changes
   - Link to issue
   - Request review

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(imech): Add custom isomorphism algorithm support

- Add IMechValidator.register_algorithm() method
- Implement custom algorithm base class
- Add tests for custom algorithms
- Update documentation with examples

Closes #123
```

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes (or documented if breaking)
- [ ] Code follows style guide
- [ ] Commit messages follow format
- [ ] PR description clearly explains changes

---

## Testing Guidelines

### Unit Tests

```python
import unittest
from rese.phase1.cognitive_biases import CognitiveBiasDetector

class TestCognitiveBiasDetector(unittest.TestCase):
    def setUp(self):
        self.detector = CognitiveBiasDetector()

    def test_detect_confirmation_bias(self):
        constraints = [
            # Test constraints
        ]
        report = self.detector.analyze_constraints(constraints)

        self.assertGreater(report.overall_bias_score, 0)
        self.assertIn('confirmation_bias', [d['bias_type'] for d in report.detections])

    def test_no_biases(self):
        # Test with unbiased constraints
        pass
```

### Integration Tests

```python
import unittest
from rese.rese_pipeline import RESEPipeline, ProblemInput

class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        problem = ProblemInput(
            id="test_problem",
            description="Test problem",
            constraints=[...],
            variables={...}
        )

        pipeline = RESEPipeline()
        result = pipeline.run(problem, phases=['phase1'])

        self.assertEqual(result.status.value, 'completed')
        self.assertIn('phase1', result.phase_results)
```

### Performance Tests

```python
import time
from rese.phase2.imech import IMechValidator

def test_isomorphism_performance():
    validator = IMechValidator()

    start = time.time()
    result = validator.compare_domains(large_graph1, large_graph2)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Too slow: {elapsed}s"
```

### Test Coverage

```bash
# Run coverage analysis
pytest --cov=rese --cov-report=html rese/tests/

# View report
open htmlcov/index.html
```

**Target Coverage:**
- Core algorithms: 90%+
- Phase executors: 85%+
- API layer: 80%+
- Overall: 85%+

---

## Code Style Guide

### Python Style (PEP 8)

**Indentation:** 4 spaces

**Line Length:** Max 100 characters (soft limit 120)

**Imports:**
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from fastapi import FastAPI

# Local
from rese.config import RESEConfig
from rese.rese_pipeline import RESEPipeline
```

**Naming Conventions:**
```python
# Classes: PascalCase
class SymbolicConstraintEngine:
    pass

# Functions/variables: snake_case
def calculate_aci():
    aci_value = 0.5

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 1000
DEFAULT_THRESHOLD = 0.7

# Private: _leading_underscore
def _internal_function():
    pass
```

**Docstrings:**
```python
def calculate_aci(
    constraints: List[Constraint],
    solution_variables: Dict[str, Any],
    domain: str
) -> float:
    """
    Calculate Algorithmic Complexity Index for a solution.

    Args:
        constraints: List of constraints
        solution_variables: Variable assignments
        domain: Problem domain

    Returns:
        ACI value (0-1)

    Raises:
        ValueError: If constraints are invalid

    Example:
        >>> aci = calculate_aci(constraints, {'x': 42}, 'optimization')
        >>> print(f"ACI: {aci:.3f}")
    """
```

### Type Hints

**Always use type hints:**
```python
from typing import List, Dict, Optional, Any

def process_data(
    data: List[Dict[str, Any]],
    config: Optional[Config] = None
) -> Dict[str, Any]:
    # Implementation
    pass
```

---

## Performance Optimization

### Profiling

```python
import cProfile

def profile_function():
    pr = cProfile.Profile()
    pr.enable()

    # Your code here
    result = pipeline.run(problem)

    pr.disable()
    pr.print_stats(sort='cumulative')
```

### Optimization Strategies

**1. Use Numba for numerical code:**
```python
from numba import jit

@jit(nopython=True)
def calculate_aci_fast(constraints_array):
    # Fast numerical calculation
    pass
```

**2. Use caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(param1, param2):
    # Cached result
    pass
```

**3. Parallel processing:**
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_item, items))
```

**4. Use generators:**
```python
# Instead of:
def get_all_items():
    return [item for item in large_collection]

# Use:
def get_all_items():
    for item in large_collection:
        yield item
```

---

## Debugging

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

### pdb Debugging

```python
import pdb

def my_function():
    # Set breakpoint
    pdb.set_trace()

    # Code to debug
    result = complex_calculation()
```

### Common Issues

**Issue:** Import errors
```python
# Add to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

**Issue:** Memory leaks
```python
# Use weak references
from weakref import WeakValueDictionary

cache = WeakValueDictionary()  # Auto-cleanup
```

---

## Release Process

### Versioning

**Semantic Versioning:** MAJOR.MINOR.PATCH

- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `config.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create git tag: `git tag v1.0.0`
6. Push tag: `git push origin v1.0.0`

---

**Developer Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team
=======
# RESE Developer Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Component Interaction](#component-interaction)
4. [Extension Points](#extension-points)
5. [Contribution Guidelines](#contribution-guidelines)
6. [Testing Guidelines](#testing-guidelines)
7. [Code Style Guide](#code-style-guide)
8. [Performance Optimization](#performance-optimization)
9. [Debugging](#debugging)
10. [Release Process](#release-process)

---

## Introduction

### Purpose of This Guide

This guide is for developers who want to:
- Contribute to RESE core codebase
- Extend RESE with custom components
- Understand RESE architecture in depth
- Optimize RESE performance
- Debug RESE issues

### Developer Prerequisites

**Required:**
- Python 3.9+
- Strong understanding of data structures and algorithms
- Familiarity with graph theory (for I_mech)
- Knowledge of statistical methods (for Phase III)
- Experience with formal logic (for Phase IV)

**Recommended:**
- Familiarity with Lean 4 or similar proof assistants
- Experience with MCTS algorithms
- Knowledge of constraint satisfaction problems
- Understanding of functional dependency graphs

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RESE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API LAYER                                │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │   │
│  │  │ REST API │  │ WebSocket│  │ Python   │           │   │
│  │  └──────────┘  └──────────┘  │ API      │           │   │
│  │                              └──────────┘           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PIPELINE ORCHESTRATION                   │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │  RESEPipeline                                 │     │   │
│  │  │  - Phase orchestration                       │     │   │
│  │  │  - Progress tracking                         │     │   │
│  │  │  - Caching                                    │     │   │
│  │  │  - Error handling                            │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              PHASE EXECUTORS                          │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │
│  │  │ Phase I │  │ Phase II│  │ Phase III│ │ Phase IV│ │   │
│  │  │ Executor│  │ Executor│  │ Executor│ │ Executor│ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              CORE ALGORITHMS                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ SCE (Φ₁) │ │ Φ₁.₅     │ │ I_mech   │             │   │
│  │  └──────────┘ └──────────┘ └──────────┘             │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ MCTS (Γ₂)│ │ ACI (Γ₁) │ │ Δ₃       │             │   │
│  │  └──────────┘ └──────────┘ └──────────┘             │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              INFRASTRUCTURE                           │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ Config   │ │ Monitoring│ │ Cache    │             │   │
│  │  │ System   │ │ System   │ │ Manager  │             │   │
│  │  └──────────┘ └──────────┘ └──────────┘             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
rese/
├── __init__.py
├── rese_pipeline.py          # Main pipeline orchestrator
├── config.py                 # Configuration system
├── api.py                    # REST API
├── monitoring.py             # Monitoring system
├── quickstart.py             # Quick start script
│
├── core/                     # Core algorithms (Phase I)
│   ├── symbolic_constraint_engine.py    # Φ₁: SCE
│   ├── constraint_stage1_integration.py
│   ├── constraint_lean4_bridge.py
│   ├── dito_graphs.py
│   ├── constraint_optimizer.py
│   └── constraint_lltl_handoff.py
│
├── phase1/                   # Phase I: Epistemic Audit
│   ├── cognitive_biases.py   # Φ₂: Bias detection
│   ├── phi2_integration.py   # Φ₁-SCE integration
│   ├── tacit_assumption_miner.py  # Φ₁.₅: Assumption mining
│   ├── failure_database.py   # Historical failure patterns
│   └── tests/
│       ├── test_cognitive_biases.py
│       ├── test_phi2_integration.py
│       └── test_tacit_assumption_miner.py
│
├── phase2/                   # Phase II: Isomorphic Resonance
│   ├── imech/                # I_mech: Mechanistic Isomorphism
│   │   ├── __init__.py
│   │   ├── isomorphism_validator.py
│   │   ├── core/
│   │   │   ├── fdg.py        # Functional Dependency Graphs
│   │   │   ├── domain.py     # Domain representation
│   │   │   ├── result.py     # Similarity results
│   │   │   ├── causality.py  # Causal structure
│   │   │   └── scoring.py    # Scoring algorithms
│   │   ├── algorithms/
│   │   │   ├── weisfeiler_lehman.py
│   │   │   ├── vf2.py
│   │   │   ├── subgraph.py
│   │   │   └── intervention.py
│   │   ├── transfer/
│   │   │   ├── mapper.py     # Solution mapping
│   │   │   ├── validator.py  # Transfer validation
│   │   │   └── repair.py     # Solution repair
│   │   ├── lean4/
│   │   │   └── proof_generator.py
│   │   └── tests/
│   │
│   ├── psi3/                 # Ψ₃: Isomorphism Validation
│   │   ├── src/
│   │   │   ├── core/
│   │   │   │   ├── constraint.py
│   │   │   │   ├── constraint_inverter.py
│   │   │   │   └── expression.py
│   │   │   ├── algorithms/
│   │   │   │   ├── preprocessing.py
│   │   │   │   └── dependency_analyzer.py
│   │   │   └── solvers/
│   │   │       └── sat_wrapper.py
│   │   ├── tests/
│   │   └── examples/
│   │
│   ├── ontology_components/  # Ψ₂: Ontology Mapping
│   │   ├── semantic_matcher.py
│   │   ├── lexical_matcher.py
│   │   ├── graph_embedder.py
│   │   ├── kg_validator.py
│   │   └── __init__.py
│   │
│   ├── ontology_imech_integration.py
│   └── test_phase2_debug.py
│
├── phase3/                   # Phase III: Monte Carlo Refinement
│   ├── stage3_integration.py # MCTS integration
│   ├── mcts_search.py        # Γ₂: MCTS implementation
│   ├── convergence_controller.py  # N_max: Convergence
│   ├── aci_analyzer/         # Γ₁: ACI Analysis
│   │   └── aci_analyzer.py
│   ├── statistical_validator.py   # Γ₃: Statistical validation
│   ├── tests/
│   │   ├── test_mcts_search.py
│   │   ├── test_convergence_controller.py
│   │   └── test_statistical_validator.py
│   │
│   └── gamma1/               # ACI Calculator
│       ├── core/
│       │   ├── aci_calculator.py
│       │   └── csp_models.py
│       └── signal/
│           └── validator.py
│
├── phase4/                   # Phase IV: Architectural Synthesis
│   ├── architecture_assembler.py   # Δ₁: Assembly
│   ├── assembly_validator.py
│   ├── predictive_model_generator.py  # Δ₂: Prediction
│   ├── aci_reduction_validator.py    # Δ₃: Validation
│   ├── phase_transition.py
│   ├── independence_checker.py
│   ├── tests/
│   │   ├── test_architecture_assembler.py
│   │   ├── test_predictive_model_generator.py
│   │   └── test_aci_reduction_validator.py
│   │
│   └── stage8_integration.py
│
├── performance/              # Performance optimization
│   ├── optimizer.py
│   ├── sce_optimizer.py
│   ├── cache_manager.py
│   └── sce_optimizer.py
│
├── security/                 # Security and validation
│   ├── input_validator.py
│   ├── error_handler.py
│   ├── resource_limiter.py
│   ├── security_audit.py
│   └── security_tests.py
│
├── integrations/             # E2E Stage integrations
│   ├── stage1.py             # Stage 1 integration
│   ├── stage2.py             # Stage 2 integration
│   ├── stage3.py             # Stage 3 integration
│   ├── stage5.py             # Stage 5 integration
│   ├── stage6.py             # Stage 6 integration
│   ├── stage7.py             # Stage 7 integration
│   ├── stage8.py             # Stage 8 integration
│   ├── stage9.py             # Stage 9 integration
│   ├── validate_integrations.py
│   └── test_e2e_pipeline.py
│
├── tests/                    # Integration tests
│   ├── test_integration/
│   ├── test_performance/
│   ├── test_validation/
│   └── phase1/
│
└── lean4/                    # Lean 4 integration
    └── scripts/
        └── export_proofs.py
```

---

## Component Interaction

### Phase Execution Flow

```
User Request
    │
    ▼
┌──────────────────┐
│ API Layer        │ → Validates request, creates ProblemInput
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ RESEPipeline     │ → Creates phase executors, manages state
└──────────────────┘
    │
    ├─────────────────────────────────────────────────┐
    │                                                 │
    ▼                                                 ▼
┌──────────────────┐                          ┌──────────────────┐
│ Phase1Executor   │                          │ CacheManager     │
└──────────────────┘                          └──────────────────┘
    │                                                 │
    ├──────────────┬──────────────┬──────────────┐    │
    ▼              ▼              ▼              ▼    ▼
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐ Check cache
│  SCE   │   │ Φ₁.₅   │   │  Φ₂    │   │  Φ₃    │────────────┐
└────────┘   └────────┘   └────────┘   └────────┘            │
    │              │              │              │            │
    └──────────────┴──────────────┴──────────────┘            │
                           │                                 │
                           ▼                                 ▼
                    PhaseResult                        Return cached
                           │                                 result
                           ▼
                    ┌──────────────┐
                    │ Phase2Executor│
                    └──────────────┘
                           │
                    (Similar structure for all phases)
                           │
                           ▼
                    PipelineResult
```

### Data Structures

#### ProblemInput
```python
@dataclass
class ProblemInput:
    id: str
    description: str
    constraints: List[Dict[str, Any]]
    variables: Dict[str, Any]
    objective: Optional[str]
    domain: str
    metadata: Dict[str, Any]
```

#### PhaseResult
```python
@dataclass
class PhaseResult:
    phase_name: str
    status: PhaseStatus
    output: Any
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    start_time: datetime
    end_time: Optional[datetime]
    elapsed_seconds: float
```

#### PipelineResult
```python
@dataclass
class PipelineResult:
    pipeline_id: str
    problem_id: str
    status: PipelineStatus
    phase_results: Dict[str, PhaseResult]
    final_solution: Optional[Dict[str, Any]]
    aci_history: List[float]
    validation_score: float
    confidence: float
    start_time: datetime
    end_time: Optional[datetime]
    elapsed_seconds: float
```

---

## Extension Points

### 1. Custom Phase Executors

Extend `PhaseExecutor` base class:

```python
from rese.rese_pipeline import PhaseExecutor, PhaseResult

class MyCustomExecutor(PhaseExecutor):
    def execute(self, input_data: Any) -> PhaseResult:
        start_time = datetime.now()
        result = PhaseResult(
            phase_name="my_custom_phase",
            status=PhaseStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Your custom logic here
            output = self._custom_processing(input_data)

            result.output = output
            result.metrics = {'custom_metric': 42}
            result.status = PhaseStatus.COMPLETED

        except Exception as e:
            result.status = PhaseStatus.FAILED
            result.errors.append(str(e))

        result.end_time = datetime.now()
        result.elapsed_seconds = (result.end_time - start_time).total_seconds()

        return result

    def _custom_processing(self, input_data):
        # Implement your custom phase logic
        return {"result": "processed"}

# Register custom phase
pipeline = RESEPipeline()
pipeline.phase_executors['custom'] = MyCustomExecutor("custom", config)
```

### 2. Custom ACI Calculators

```python
from gamma1.core.aci_calculator import ACICalculator

class MyCustomACICalculator(ACICalculator):
    def calculate_solution(
        self,
        constraints: List[Constraint],
        solution_variables: Dict[str, Any],
        domain: str
    ) -> float:
        # Base ACI calculation
        base_aci = super().calculate_solution(
            constraints, solution_variables, domain
        )

        # Add custom factors
        custom_factor = self._calculate_custom_factor(
            constraints, solution_variables
        )

        return base_aci * custom_factor

    def _calculate_custom_factor(self, constraints, solution_variables):
        # Your custom logic
        return 1.0

# Use custom calculator
config.phase3.gamma2_aci_calculator = MyCustomACICalculator()
```

### 3. Custom Isomorphism Algorithms

```python
from phase2.imech.algorithms import WeisfeilerLehman

class MyCustomAlgorithm(WeisfeilerLehman):
    def compare(
        self,
        graph1: FunctionalDependencyGraph,
        graph2: FunctionalDependencyGraph
    ) -> float:
        # Custom comparison logic
        similarity = super().compare(graph1, graph2)

        # Add domain-specific adjustments
        adjusted_similarity = self._adjust_for_domain(
            similarity, graph1, graph2
        )

        return adjusted_similarity

    def _adjust_for_domain(self, similarity, graph1, graph2):
        # Your custom domain adjustment
        return similarity

# Register custom algorithm
from phase2.imech import IMechValidator
IMechValidator.register_algorithm('custom', MyCustomAlgorithm)
```

### 4. Custom MCTS Policies

```python
from phase3.mcts_search import MCTSSearch

class MyCustomMCTS(MCTSSearch):
    def _selection_policy(self, node):
        # Custom selection policy
        # Default: UCB = Q/N + C * sqrt(ln(N_parent)/N)

        # Your custom policy
        custom_score = self._custom_selection_score(node)
        return custom_score

    def _custom_selection_score(self, node):
        # Your logic
        return node.ucb_score

# Use custom MCTS
mcts = MyCustomMCTS(
    exploration_constant=1.41,
    max_iterations=1000,
    aci_guided=True
)
```

---

## Contribution Guidelines

### Code Review Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Write Code**
   - Follow code style guide
   - Add tests
   - Update documentation

3. **Test**
   ```bash
   pytest rese/tests/
   pytest rese/tests/phase1/
   pytest rese/tests/phase2/imech/
   ```

4. **Submit Pull Request**
   - Describe changes
   - Link to issue
   - Request review

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(imech): Add custom isomorphism algorithm support

- Add IMechValidator.register_algorithm() method
- Implement custom algorithm base class
- Add tests for custom algorithms
- Update documentation with examples

Closes #123
```

### Pull Request Checklist

- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes (or documented if breaking)
- [ ] Code follows style guide
- [ ] Commit messages follow format
- [ ] PR description clearly explains changes

---

## Testing Guidelines

### Unit Tests

```python
import unittest
from rese.phase1.cognitive_biases import CognitiveBiasDetector

class TestCognitiveBiasDetector(unittest.TestCase):
    def setUp(self):
        self.detector = CognitiveBiasDetector()

    def test_detect_confirmation_bias(self):
        constraints = [
            # Test constraints
        ]
        report = self.detector.analyze_constraints(constraints)

        self.assertGreater(report.overall_bias_score, 0)
        self.assertIn('confirmation_bias', [d['bias_type'] for d in report.detections])

    def test_no_biases(self):
        # Test with unbiased constraints
        pass
```

### Integration Tests

```python
import unittest
from rese.rese_pipeline import RESEPipeline, ProblemInput

class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        problem = ProblemInput(
            id="test_problem",
            description="Test problem",
            constraints=[...],
            variables={...}
        )

        pipeline = RESEPipeline()
        result = pipeline.run(problem, phases=['phase1'])

        self.assertEqual(result.status.value, 'completed')
        self.assertIn('phase1', result.phase_results)
```

### Performance Tests

```python
import time
from rese.phase2.imech import IMechValidator

def test_isomorphism_performance():
    validator = IMechValidator()

    start = time.time()
    result = validator.compare_domains(large_graph1, large_graph2)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"Too slow: {elapsed}s"
```

### Test Coverage

```bash
# Run coverage analysis
pytest --cov=rese --cov-report=html rese/tests/

# View report
open htmlcov/index.html
```

**Target Coverage:**
- Core algorithms: 90%+
- Phase executors: 85%+
- API layer: 80%+
- Overall: 85%+

---

## Code Style Guide

### Python Style (PEP 8)

**Indentation:** 4 spaces

**Line Length:** Max 100 characters (soft limit 120)

**Imports:**
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
from fastapi import FastAPI

# Local
from rese.config import RESEConfig
from rese.rese_pipeline import RESEPipeline
```

**Naming Conventions:**
```python
# Classes: PascalCase
class SymbolicConstraintEngine:
    pass

# Functions/variables: snake_case
def calculate_aci():
    aci_value = 0.5

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 1000
DEFAULT_THRESHOLD = 0.7

# Private: _leading_underscore
def _internal_function():
    pass
```

**Docstrings:**
```python
def calculate_aci(
    constraints: List[Constraint],
    solution_variables: Dict[str, Any],
    domain: str
) -> float:
    """
    Calculate Algorithmic Complexity Index for a solution.

    Args:
        constraints: List of constraints
        solution_variables: Variable assignments
        domain: Problem domain

    Returns:
        ACI value (0-1)

    Raises:
        ValueError: If constraints are invalid

    Example:
        >>> aci = calculate_aci(constraints, {'x': 42}, 'optimization')
        >>> print(f"ACI: {aci:.3f}")
    """
```

### Type Hints

**Always use type hints:**
```python
from typing import List, Dict, Optional, Any

def process_data(
    data: List[Dict[str, Any]],
    config: Optional[Config] = None
) -> Dict[str, Any]:
    # Implementation
    pass
```

---

## Performance Optimization

### Profiling

```python
import cProfile

def profile_function():
    pr = cProfile.Profile()
    pr.enable()

    # Your code here
    result = pipeline.run(problem)

    pr.disable()
    pr.print_stats(sort='cumulative')
```

### Optimization Strategies

**1. Use Numba for numerical code:**
```python
from numba import jit

@jit(nopython=True)
def calculate_aci_fast(constraints_array):
    # Fast numerical calculation
    pass
```

**2. Use caching:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(param1, param2):
    # Cached result
    pass
```

**3. Parallel processing:**
```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_item, items))
```

**4. Use generators:**
```python
# Instead of:
def get_all_items():
    return [item for item in large_collection]

# Use:
def get_all_items():
    for item in large_collection:
        yield item
```

---

## Debugging

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
```

### pdb Debugging

```python
import pdb

def my_function():
    # Set breakpoint
    pdb.set_trace()

    # Code to debug
    result = complex_calculation()
```

### Common Issues

**Issue:** Import errors
```python
# Add to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

**Issue:** Memory leaks
```python
# Use weak references
from weakref import WeakValueDictionary

cache = WeakValueDictionary()  # Auto-cleanup
```

---

## Release Process

### Versioning

**Semantic Versioning:** MAJOR.MINOR.PATCH

- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Checklist

1. Update version in `config.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create git tag: `git tag v1.0.0`
6. Push tag: `git push origin v1.0.0`

---

**Developer Guide Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team
>>>>>>> 1cb9c5e35 (update)
