# Z3 Prover Integration Modules

This directory contains comprehensive Z3 SMT solver integration modules for the OpenEvolve platform.

## Overview

The Z3 integration provides formal verification capabilities across multiple subsystems:

- **Reliability Checking** - Verify component and system reliability constraints
- **Decomposition Validation** - Validate problem decomposition correctness
- **Quality Gate Verification** - Formal verification of SOPs and quality constraints
- **Workflow Stages** - Native Z3 solving as workflow primitives
- **Blue Team Validation** - Security property verification
- **Analytics** - Z3 solving metrics and performance tracking
- **Evolution Fitness** - Constraint-based evolutionary fitness evaluation
- **Knowledge Graph** - Store and reuse proofs
- **Chronicle Memory** - Retrieve similar past problems

## Modules

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `z3_reliability_checker.py` | ROMA/MDAP reliability verification | `Z3ReliabilityChecker`, `ComponentReliabilityModel` |
| `decomposition_z3_validator.py` | Decomposition correctness | `Z3DecompositionValidator`, `SubProblemModel` |
| `quality_gate_z3_verifier.py` | Quality gate formal verification | `Z3QualityGateVerifier` |
| `workflow_stage_z3.py` | Workflow engine integration | `Z3WorkflowStage`, `Z3StageRegistry` |
| `blue_team_z3_validator.py` | Security validation | `BlueTeamZ3Validator` |
| `analytics_z3_connector.py` | Analytics integration | `AnalyticsZ3Connector` |
| `evolution_z3_fitness.py` | Evolution fitness evaluation | `Z3FitnessEvaluator` |
| `knowledge_graph_z3_connector.py` | Knowledge graph storage | `KnowledgeGraphZ3Connector` |
| `chronicle_memory_z3_integration.py` | Chronicle memory integration | `ChronicleMemoryZ3Integration` |
| `n8n_z3_nodes.py` | n8n automation nodes | `Z3SolveNode`, `Z3OptimizeNode` |

## Usage Examples

### 1. Reliability Verification

```python
from z3_reliability_checker import (
    get_z3_reliability_checker,
    ComponentReliabilityModel,
    ReliabilityConstraint,
    ReliabilityProperty
)

# Create checker
checker = get_z3_reliability_checker()

# Define component model
component = ComponentReliabilityModel(
    component_id="auth_service",
    availability=0.995,
    mtbf_hours=4380,
    mttr_hours=0.5
)

# Define requirements
requirements = [
    ReliabilityConstraint(
        property_type=ReliabilityProperty.AVAILABILITY,
        threshold=0.999
    )
]

# Verify
result = checker.verify_component_reliability(component, requirements)
print(f"Verified: {result.verified}")
print(f"Recommendations: {result.recommendations}")
```

### 2. Decomposition Validation

```python
from decomposition_z3_validator import (
    get_z3_decomposition_validator,
    SubProblemModel,
    EntanglementSpecification
)

# Create validator
validator = get_z3_decomposition_validator()

# Define sub-problems
subproblems = [
    SubProblemModel(subproblem_id="sp1", complexity_score=2.0),
    SubProblemModel(subproblem_id="sp2", complexity_score=1.5)
]

# Define entanglements
entanglements = [
    EntanglementSpecification(
        entanglement_id="ent1",
        source_subproblem="sp1",
        target_subproblem="sp2",
        shared_variables=["x"]
    )
]

# Validate
result = validator.validate_decomposition(
    original_problem="(set-logic LIA)...",
    subproblems=subproblems,
    entanglements=entanglements
)
print(f"Valid: {result.valid}")
print(f"Properties verified: {result.properties_verified}")
```

### 3. Quality Gate Verification

```python
from quality_gate_z3_verifier import get_z3_quality_gate_verifier

# Create verifier
verifier = get_z3_quality_gate_verifier()

# Verify SOP safety
result = verifier.verify_sop_safety(
    sop_steps=["Step 1: Authenticate user", "Step 2: Process request"],
    safety_invariants=["(assert (> security_level 0))"]
)
print(f"Safety verified: {result.status}")

# Verify performance guarantees
result = verifier.verify_performance_guarantee(
    constraint_specs=[
        {"expression": "(< latency_ms 100)"},
        {"expression": "(< cpu_percent 80)"}
    ]
)
print(f"Performance feasible: {result.verified}")
```

### 4. Workflow Stage

```python
from workflow_stage_z3 import (
    get_z3_stage_registry,
    Z3StageConfig,
    Z3StageType
)

# Create registry
registry = get_z3_stage_registry()

# Configure solve stage
config = Z3StageConfig(
    stage_type=Z3StageType.SOLVE,
    timeout_seconds=60.0,
    variables=[{"name": "x", "type": "INTEGER"}],
    constraints=["(> x 0)", "(< x 10)"]
)

# Create and execute stage
stage = registry.create_stage(config)
result = stage.execute(context={})
print(f"Status: {result.status}")
print(f"Model: {result.model}")
```

### 5. Evolution Fitness

```python
from evolution_z3_fitness import get_z3_fitness_evaluator, FitnessConstraint

# Create evaluator
evaluator = get_z3_fitness_evaluator()

# Define individual
individual = {"id": "ind1", "x": 5, "y": 10}

# Define constraints
constraints = [
    FitnessConstraint("c1", "(> x 0)", weight=1.0, is_hard=True),
    FitnessConstraint("c2", "(< y 20)", weight=1.0, is_hard=True)
]

# Evaluate
result = evaluator.evaluate_fitness(individual, constraints)
print(f"Fitness: {result.fitness_score}")
print(f"Feasible: {result.is_feasible}")
```

## Integration Points

### Quality Gate Engine

The `quality_gate_engine.py` now includes Z3 verification:

```python
from quality_gate_engine import QualityGateEngine

engine = QualityGateEngine()

# Run formal verification
result = engine.verify_with_z3(
    verification_type="sop_safety",
    config={
        "steps": [...],
        "invariants": [...]
    }
)
```

### Decomposition Engine

The `decomposition_engine.py` now includes Z3 validation:

```python
from decomposition_engine import DecompositionEngine

engine = DecompositionEngine()

# Create decomposition plan
plan = engine.decompose(problem)

# Validate with Z3
validation = engine.validate_with_z3(problem, plan)
```

### Evolution

The `evolution.py` now includes Z3 fitness evaluation:

```python
from evolution import evaluate_fitness_with_z3

result = evaluate_fitness_with_z3(
    individual={"x": 5},
    constraints=[{"expression": "(> x 0)"}]
)
```

## Configuration

All Z3 modules support configuration through `Z3Config`:

```python
from z3prover_integration import Z3Config

config = Z3Config(
    timeout=60.0,
    memory_limit_mb=4096,
    proof_generation=True,
    num_threads=4
)
```

## Testing

Run unit tests:

```bash
python -m pytest test_z3_reliability_checker.py -v
python -m pytest test_decomposition_z3_validator.py -v
```

## Dependencies

- `z3prover_integration.py` - Base Z3 integration (required)
- `z3prover_advanced.py` - Advanced Z3 features (optional)
- `z3_leanaide_bridge.py` - Lean translation (optional)

## Error Handling

All modules gracefully handle missing Z3:

```python
try:
    from z3prover_integration import Z3SolverEngine
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
```

When Z3 is not available, modules return appropriate fallback responses.

## Performance Considerations

- **Timeouts**: Default 60s for most operations, 120s for decomposition
- **Caching**: Results cached to avoid repeated solving
- **Thread Safety**: All modules use thread-safe designs
- **Memory**: Memory limits configurable through Z3Config

## License

OpenEvolve - Proprietary
