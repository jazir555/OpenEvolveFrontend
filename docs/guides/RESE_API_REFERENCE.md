<<<<<<< HEAD
# RESE API Reference

## Table of Contents

1. [Overview](#overview)
2. [Pipeline API](#pipeline-api)
3. [Phase I APIs](#phase-i-apis)
4. [Phase II APIs](#phase-ii-apis)
5. [Phase III APIs](#phase-iii-apis)
6. [Phase IV APIs](#phase-iv-apis)
7. [REST API Endpoints](#rest-api-endpoints)
8. [WebSocket API](#websocket-api)
9. [Configuration API](#configuration-api)
10. [Error Handling](#error-handling)
11. [Type Definitions](#type-definitions)

---

## Overview

### API Version

**Current Version:** v1.0.0
**Base Path:** `/api/v1`
**Content-Type:** `application/json`

### Authentication

Most endpoints require API key authentication:

```http
X-API-Key: your-api-key-here
```

Set via environment variable:
```bash
export RESE_API_KEYS="key1,key2,key3"
```

### Rate Limiting

- **Default:** 60 requests per minute
- **Header:** `X-RateLimit-Remaining: 45`
- **Error:** HTTP 429 when exceeded

---

## Pipeline API

### RESEPipeline

Main pipeline orchestrator for running RESE analysis.

#### Constructor

```python
RESEPipeline(config: Optional[RESEConfig] = None) -> RESEPipeline
```

**Parameters:**
- `config` (Optional[RESEConfig]): Configuration object. Uses default if None.

**Returns:**
- `RESEPipeline`: Pipeline instance

**Example:**
```python
from rese.rese_pipeline import RESEPipeline
from rese.config import get_config

# Use default config
pipeline = RESEPipeline()

# Use custom config
config = get_config()
config.pipeline.enable_caching = True
pipeline = RESEPipeline(config)
```

---

#### run()

Execute complete RESE pipeline on a problem.

```python
run(
    problem: ProblemInput,
    phases: Optional[List[str]] = None,
    use_cache: bool = True
) -> PipelineResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem` | ProblemInput | Yes | - | Input problem definition |
| `phases` | Optional[List[str]] | No | `['phase1', 'phase2', 'phase3', 'phase4']` | Phases to execute |
| `use_cache` | bool | No | `True` | Enable caching of intermediate results |

**Valid Phase Values:**
- `'phase1'`: Epistemic Audit (Φ₁, Φ₁.₅, Φ₂, Φ₃)
- `'phase2'`: Isomorphic Resonance (Ψ₁, Ψ₂, Ψ₃, I_mech)
- `'phase3'`: Monte Carlo Refinement (Γ₁, Γ₂, Γ₃, N_max)
- `'phase4'`: Architectural Synthesis (Δ₁, Δ₂, Δ₃)

**Returns:**
- `PipelineResult`: Complete execution results

**Raises:**
- `PhaseExecutionError`: If a phase fails fatally
- `ValidationError`: If input validation fails
- `CachingError`: If cache operation fails

**Example:**
```python
# Run all phases
result = pipeline.run(problem)

# Run only Phase I and III
result = pipeline.run(problem, phases=['phase1', 'phase3'])

# Run without cache
result = pipeline.run(problem, use_cache=False)
```

---

#### add_progress_callback()

Add callback function for progress updates.

```python
add_progress_callback(callback: Callable[[PipelineResult], None]) -> None
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `callback` | Callable[[PipelineResult], None] | Yes | Function to call with updates |

**Callback Signature:**
```python
def my_callback(result: PipelineResult) -> None:
    print(f"Status: {result.status.value}")
    print(f"Progress: {len(result.phase_results)}/4 phases")
```

**Example:**
```python
def track_progress(result):
    for phase_name, phase_result in result.phase_results.items():
        if phase_result.status == PhaseStatus.COMPLETED:
            print(f"{phase_name} completed: {phase_result.elapsed_seconds:.2f}s")

pipeline.add_progress_callback(track_progress)
result = pipeline.run(problem)
```

---

#### cancel()

Cancel currently running pipeline execution.

```python
cancel() -> None
```

**Example:**
```python
# Run in background
import threading

thread = threading.Thread(target=pipeline.run, args=(problem,))
thread.start()

# Cancel if needed
pipeline.cancel()
thread.join()
```

---

#### get_status()

Get current pipeline status.

```python
get_status() -> PipelineStatus
```

**Returns:**
- `PipelineStatus`: Current status enum

**Possible Values:**
- `PipelineStatus.IDLE`: Not running
- `PipelineStatus.RUNNING`: Currently executing
- `PipelineStatus.PAUSED`: Paused (future feature)
- `PipelineStatus.COMPLETED`: Finished successfully
- `PipelineStatus.FAILED`: Failed with error
- `PipelineStatus.CANCELLED`: Cancelled by user

**Example:**
```python
status = pipeline.get_status()
if status == PipelineStatus.RUNNING:
    print("Pipeline is running...")
```

---

#### get_progress()

Get detailed progress information.

```python
get_progress() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Progress information

**Structure:**
```python
{
    'pipeline_id': str,
    'status': str,
    'elapsed_seconds': float,
    'phases': {
        'phase1': {
            'status': str,
            'elapsed': float,
            'metrics': Dict[str, Any]
        },
        # ... other phases
    }
}
```

**Example:**
```python
progress = pipeline.get_progress()
print(f"Pipeline: {progress['pipeline_id']}")
print(f"Status: {progress['status']}")

for phase_name, phase_info in progress['phases'].items():
    print(f"{phase_name}: {phase_info['status']}")
```

---

### Convenience Functions

#### run_rese()

Quick function to run RESE without explicit pipeline creation.

```python
run_rese(
    problem_description: str,
    constraints: List[Dict[str, Any]],
    variables: Dict[str, Any],
    config: Optional[RESEConfig] = None
) -> PipelineResult
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `problem_description` | str | Yes | Natural language problem description |
| `constraints` | List[Dict[str, Any]] | Yes | List of constraint dictionaries |
| `variables` | Dict[str, Any] | Yes | Problem variables |
| `config` | Optional[RESEConfig] | No | Configuration object |

**Returns:**
- `PipelineResult`: Execution results

**Example:**
```python
from rese.rese_pipeline import run_rese

result = run_rese(
    problem_description="Optimize production schedule",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'description': 'cost < 10000'},
        {'id': 'c2', 'type': 'soft', 'description': 'maximize throughput'}
    ],
    variables={'cost': 'float', 'throughput': 'float'}
)
```

---

## Phase I APIs

### SymbolicConstraintEngine

Formalizes and manages constraints.

#### Constructor

```python
SymbolicConstraintEngine(max_constraints: int = 10000) -> SymbolicConstraintEngine
```

**Parameters:**
- `max_constraints` (int): Maximum number of constraints (default: 10000)

---

#### add_constraint()

Add a constraint to the engine.

```python
add_constraint(constraint: Constraint) -> None
```

**Parameters:**
- `constraint` (Constraint): Constraint object

**Example:**
```python
from core.symbolic_constraint_engine import SymbolicConstraintEngine, Constraint, ConstraintType

sce = SymbolicConstraintEngine()

constraint = Constraint(
    id='c1',
    type=ConstraintType.HARD,
    description='Cost must be below 1000',
    formalization='cost < 1000',
    source='user'
)
sce.add_constraint(constraint)
```

---

#### detect_conflicts()

Detect conflicting constraints.

```python
detect_conflicts() -> List[Conflict]
```

**Returns:**
- `List[Conflict]`: List of detected conflicts

**Conflict Structure:**
```python
{
    'constraint_ids': List[str],  # Conflicting constraint IDs
    'type': str,                   # Conflict type
    'description': str,            # Human-readable description
    'severity': str                # 'error', 'warning'
}
```

**Example:**
```python
conflicts = sce.detect_conflicts()
for conflict in conflicts:
    print(f"Conflict: {conflict['constraint_ids']}")
    print(f"Description: {conflict['description']}")
```

---

#### get_all_constraints()

Retrieve all constraints.

```python
get_all_constraints() -> List[Constraint]
```

**Returns:**
- `List[Constraint]`: All constraints

---

### CognitiveBiasDetector

Detects cognitive biases in constraints.

#### analyze_constraints()

Analyze constraints for biases.

```python
analyze_constraints(
    constraints: List[Constraint],
    threshold: float = 0.5
) -> BiasReport
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `constraints` | List[Constraint] | Yes | - | Constraints to analyze |
| `threshold` | float | No | 0.5 | Bias detection threshold |

**Returns:**
- `BiasReport`: Bias analysis report

**BiasReport Structure:**
```python
{
    'overall_bias_score': float,      # 0-1, higher = more biased
    'total_detections': int,           # Number of biases found
    'detections': [
        {
            'bias_type': str,          # Type of bias
            'constraint_id': str,      # Affected constraint
            'severity': str,           # 'low', 'medium', 'high'
            'description': str,        # Description
            'recommendation': str      # How to fix
        }
    ]
}
```

**Bias Types:**
- `confirmation_bias`: Seeking confirming evidence only
- `anchoring_bias`: Over-relying on initial information
- `availability_bias`: Overweighting easily recalled examples
- `sunk_cost_bias`: Continuing failing approaches

**Example:**
```python
from phase1.cognitive_biases import CognitiveBiasDetector

detector = CognitiveBiasDetector()
report = detector.analyze_constraints(constraints)

print(f"Bias Score: {report.overall_bias_score:.2f}")
print(f"Detections: {report.total_detections}")

for detection in report.detections:
    print(f"{detection['bias_type']}: {detection['description']}")
    print(f"Recommendation: {detection['recommendation']}")
```

---

### TacitAssumptionMiner (Φ₁.₅)

Mines hidden assumptions from constraint set.

#### mine_assumptions()

Extract tacit assumptions.

```python
mine_assumptions(
    constraints: List[Constraint],
    domain: str,
    max_assumptions: int = 100,
    threshold: float = 0.6
) -> List[Assumption]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `constraints` | List[Constraint] | Yes | - | Constraints to analyze |
| `domain` | str | Yes | - | Problem domain |
| `max_assumptions` | int | No | 100 | Maximum assumptions to return |
| `threshold` | float | No | 0.6 | Confidence threshold |

**Returns:**
- `List[Assumption]`: List of assumptions

**Assumption Structure:**
```python
{
    'id': str,
    'description': str,           # Assumption description
    'type': str,                  # Category of assumption
    'confidence': float,          # 0-1 confidence score
    'source': str,                # 'failure_db', 'domain_kb', 'inference'
    'justification': str          # Why this assumption was made
}
```

**Example:**
```python
from phase1.tacit_assumption_miner import TacitAssumptionMiner

miner = TacitAssumptionMiner()
assumptions = miner.mine_assumptions(
    constraints=constraints,
    domain='bridge_design',
    threshold=0.7
)

print(f"Found {len(assumptions)} tacit assumptions:")
for assumption in assumptions:
    print(f"- {assumption['description']} (confidence: {assumption['confidence']:.2f})")
```

---

## Phase II APIs

### IMechValidator

Validates mechanistic isomorphisms between domains.

#### compare_domains()

Compare two domains for isomorphism.

```python
compare_domains(
    source_domain: Domain,
    target_domain: Domain,
    algorithm: str = 'weisfeiler_lehman'
) -> SimilarityResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_domain` | Domain | Yes | - | Source domain |
| `target_domain` | Domain | Yes | - | Target domain |
| `algorithm` | str | No | `'weisfeiler_lehman'` | Isomorphism algorithm |

**Valid Algorithms:**
- `'weisfeiler_lehman'`: Fast, scalable graph isomorphism
- `'vf2'`: Exact graph isomorphism (slower)
- `'subgraph'`: Partial isomorphism detection

**Returns:**
- `SimilarityResult`: Similarity analysis result

**SimilarityResult Structure:**
```python
{
    'score': float,                     # 0-1 overall similarity
    'structural_similarity': float,      # Graph topology similarity
    'causal_similarity': float,          # Causal structure similarity
    'interventional_similarity': float,  # Intervention response similarity
    'is_isomorphic': bool,               # True if similarity > threshold
    'confidence': float,                 # Statistical confidence
    'mapping': Dict[str, str]           # Variable mapping
}
```

**Example:**
```python
from phase2.imech import IMechValidator, Domain

validator = IMechValidator()

# Define domains
source = Domain(
    id='chemical_reactor',
    name='Chemical Reactor',
    variables={'temperature', 'pressure', 'yield'},
    constraints=['yield = f(temp, pressure)']
)

target = Domain(
    id='electrical_circuit',
    name='Circuit',
    variables={'voltage', 'current', 'power'},
    constraints=['power = g(voltage, current)']
)

# Compare
similarity = validator.compare_domains(source, target)

print(f"Isomorphism Score: {similarity.score:.2f}")
print(f"Structural: {similarity.structural_similarity:.2f}")
print(f"Causal: {similarity.causal_similarity:.2f}")
print(f"Is Isomorphic: {similarity.is_isomorphic}")
```

---

#### generate_isomorphism_proof()

Generate Lean 4 proof for isomorphism.

```python
generate_isomorphism_proof(
    source_domain: Domain,
    target_domain: Domain
) -> Proof
```

**Parameters:**
- `source_domain` (Domain): Source domain
- `target_domain` (Domain): Target domain

**Returns:**
- `Proof`: Lean 4 proof object

**Proof Structure:**
```python
{
    'is_valid': bool,
    'lean4_code': str,           # Lean 4 proof code
    'verification_status': str,   # 'verified', 'pending', 'failed'
    'theorem': str,              # Proven theorem
    'timestamp': str
}
```

**Example:**
```python
proof = validator.generate_isomorphism_proof(source, target)

if proof['is_valid']:
    print(f"Theorem: {proof['theorem']}")
    print(f"Lean 4 Code:\n{proof['lean4_code']}")
```

---

### SemanticMatcher

Maps ontologies using semantic similarity.

#### find_similar_domains()

Find semantically similar domains.

```python
find_similar_domains(
    problem_description: str,
    similarity_threshold: float = 0.7,
    max_results: int = 10
) -> List[DomainMatch]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem_description` | str | Yes | - | Problem description |
| `similarity_threshold` | float | No | 0.7 | Minimum similarity |
| `max_results` | int | No | 10 | Maximum results |

**Returns:**
- `List[DomainMatch]`: List of matching domains

**DomainMatch Structure:**
```python
{
    'domain_id': str,
    'domain_name': str,
    'similarity': float,           # 0-1 semantic similarity
    'description': str,
    'key_concepts': List[str]      # Matching concepts
}
```

**Example:**
```python
from phase2.ontology_components.semantic_matcher import SemanticMatcher

matcher = SemanticMatcher()
matches = matcher.find_similar_domains(
    problem_description="Optimize neural network architecture for image classification",
    similarity_threshold=0.75
)

for match in matches:
    print(f"{match['domain_name']}: {match['similarity']:.2f}")
    print(f"  Concepts: {', '.join(match['key_concepts'])}")
```

---

## Phase III APIs

### MCTSSearch (Γ₂)

Monte Carlo Tree Search with ACI guidance.

#### Constructor

```python
MCTSSearch(
    exploration_constant: float = 1.41,
    max_iterations: int = 1000,
    playout_depth: int = 100,
    aci_guided: bool = True,
    parallel_agents: int = 4
) -> MCTSSearch
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `exploration_constant` | float | No | 1.41 | UCB exploration constant (C) |
| `max_iterations` | int | No | 1000 | Maximum MCTS iterations |
| `playout_depth` | int | No | 100 | Max playout depth |
| `aci_guided` | bool | No | True | Use ACI for exploration bonus |
| `parallel_agents` | int | No | 4 | Parallel search agents |

---

#### search()

Execute MCTS search for optimal solution.

```python
search(
    problem: ProblemInput,
    constraints: List[Constraint]
) -> Solution
```

**Parameters:**
- `problem` (ProblemInput): Problem definition
- `constraints` (List[Constraint]): Constraints

**Returns:**
- `Solution`: Best solution found

**Solution Structure:**
```python
{
    'variables': Dict[str, Any],   # Variable assignments
    'aci': float,                  # Final ACI
    'confidence': float,           # Statistical confidence
    'iterations': int,             # Iterations used
    'converged': bool,             # Convergence status
    'value': float                 # Objective value
}
```

**Example:**
```python
from phase3.mcts_search import MCTSSearch

mcts = MCTSSearch(
    max_iterations=5000,
    aci_guided=True
)

solution = mcts.search(problem, constraints)

print(f"Best Solution: {solution['variables']}")
print(f"ACI: {solution['aci']:.3f}")
print(f"Confidence: {solution['confidence']:.2f}")
print(f"Converged: {solution['converged']}")
```

---

### ACICalculator (Γ₁)

Calculates Algorithmic Complexity Index.

#### calculate_solution()

Calculate ACI for a solution.

```python
calculate_solution(
    constraints: List[Constraint],
    solution_variables: Dict[str, Any],
    domain: str
) -> float
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `constraints` | List[Constraint] | Yes | Constraints |
| `solution_variables` | Dict[str, Any] | Yes | Solution variable assignments |
| `domain` | str | Yes | Problem domain |

**Returns:**
- `float`: ACI value (0-1)

**Example:**
```python
from gamma1.core.aci_calculator import ACICalculator

calculator = ACICalculator()

aci = calculator.calculate_solution(
    constraints=constraints,
    solution_variables={'x': 42, 'y': 17},
    domain='optimization'
)

print(f"ACI: {aci:.3f}")
```

---

### StatisticalValidator (Γ₃)

Validates solutions with statistical tests.

#### validate()

Validate solution statistically.

```python
validate(
    solution: Solution,
    confidence_level: float = 0.95,
    n_bootstrap_samples: int = 1000
) -> ValidationResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `solution` | Solution | Yes | - | Solution to validate |
| `confidence_level` | float | No | 0.95 | Confidence level |
| `n_bootstrap_samples` | int | No | 1000 | Bootstrap iterations |

**Returns:**
- `ValidationResult`: Validation result

**ValidationResult Structure:**
```python
{
    'is_valid': bool,
    'p_value': float,                    # Statistical significance
    'confidence_interval': Tuple[float, float],
    'effect_size': float,
    'power': float,
    'recommendations': List[str]
}
```

**Example:**
```python
from phase3.statistical_validator import StatisticalValidator

validator = StatisticalValidator(confidence_level=0.95)

validation = validator.validate(solution)

print(f"Valid: {validation['is_valid']}")
print(f"P-value: {validation['p_value']:.4f}")
print(f"95% CI: {validation['confidence_interval']}")
```

---

## Phase IV APIs

### Delta3Validator (Δ₃)

Validates ACI reduction.

#### validate()

Validate final solution.

```python
validate(
    problem: Problem,
    solution: RESESolution
) -> ValidationResult
```

**Parameters:**
- `problem` (Problem): Original problem
- `solution` (RESESolution): RESE solution

**Returns:**
- `ValidationResult`: Validation result

**ValidationResult Structure:**
```python
{
    'is_valid': bool,
    'validation_score': float,          # 0-1
    'confidence': float,                # 0-1
    'aci_reduction': float,             # 0-1 (target: ≥0.2)
    'p_value': float,
    'significance': str,                # 'significant', 'marginal', 'not_significant'
    'recommendations': List[str]
}
```

**Example:**
```python
from phase4.aci_reduction_validator import Delta3Validator, Problem, RESESolution

validator = Delta3Validator(
    validation_threshold=0.7,
    min_aci_reduction=0.2
)

problem = Problem(
    id='problem_1',
    description='Optimize system',
    constraints=constraints,
    variables=variables
)

solution = RESESolution(
    problem_id='problem_1',
    solution={'x': 42, 'y': 17},
    aci_history=[0.85, 0.72, 0.55, 0.28, 0.15],
    stage_results={...}
)

validation = validator.validate(problem, solution)

print(f"Valid: {validation.is_valid}")
print(f"Score: {validation.validation_score:.2f}")
print(f"ACI Reduction: {validation.aci_reduction * 100:.1f}%")
```

---

### ArchitectureAssembler (Δ₁)

Assembles final solution architecture.

#### assemble()

Assemble architecture from phase outputs.

```python
assemble(
    phase1_output: Any,
    phase2_output: Any,
    phase3_output: Any
) -> Architecture
```

**Parameters:**
- `phase1_output` (Any): Phase I output
- `phase2_output` (Any): Phase II output
- `phase3_output` (Any): Phase III output

**Returns:**
- `Architecture`: Assembled architecture

**Architecture Structure:**
```python
{
    'components': List[Component],
    'integration_strategy': str,        # 'hierarchical', 'flat', 'hybrid'
    'validation_score': float,
    'metadata': Dict[str, Any]
}
```

---

## REST API Endpoints

### Base URL

```
http://localhost:8000/api/v1
```

### Health Endpoints

#### GET /health

Check API health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-31T12:00:00Z",
  "uptime_seconds": 3600.5
}
```

---

### Pipeline Endpoints

#### POST /pipeline/run

Submit problem to RESE pipeline.

**Request:**
```http
POST /api/v1/pipeline/run
Content-Type: application/json
X-API-Key: your-api-key

{
  "description": "Optimize production schedule",
  "constraints": [
    {
      "id": "c1",
      "type": "hard",
      "description": "cost < 10000",
      "formalization": "cost < 10000"
    }
  ],
  "variables": {
    "cost": "float",
    "throughput": "float"
  },
  "phases": ["phase1", "phase2", "phase3", "phase4"],
  "use_cache": true
}
```

**Response:**
```json
{
  "pipeline_id": "rese_abc123",
  "problem_id": "problem_def456",
  "status": "completed",
  "final_solution": {
    "cost": 8500.0,
    "throughput": 1250.0
  },
  "aci_history": [0.85, 0.72, 0.55, 0.28, 0.15],
  "validation_score": 0.87,
  "confidence": 0.85,
  "elapsed_seconds": 45.3,
  "phase_results": { ... }
}
```

---

#### GET /pipeline/{pipeline_id}/status

Get pipeline status.

**Response:**
```json
{
  "pipeline_id": "rese_abc123",
  "problem_id": "problem_def456",
  "status": "running",
  "elapsed_seconds": 23.5,
  "phases": {
    "phase1": {
      "status": "completed",
      "elapsed": 5.2,
      "metrics": { ... }
    },
    "phase2": {
      "status": "running",
      "elapsed": 18.3,
      "metrics": { ... }
    }
  }
}
```

---

#### GET /pipeline/{pipeline_id}/result

Get complete pipeline result.

**Response:** Same as POST /pipeline/run

---

#### DELETE /pipeline/{pipeline_id}

Cancel running pipeline.

**Response:**
```json
{
  "message": "Pipeline rese_abc123 cancelled",
  "pipeline_id": "rese_abc123"
}
```

---

### Admin Endpoints

#### GET /admin/stats

Get system statistics.

**Response:**
```json
{
  "active_pipelines": 3,
  "stored_results": 127,
  "websocket_connections": 5,
  "uptime_seconds": 86400.0
}
```

---

#### POST /admin/cache/clear

Clear pipeline cache.

**Response:**
```json
{
  "message": "Cache cleared"
}
```

---

## WebSocket API

### Connection

Connect to WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/pipeline/{pipeline_id}');
```

### Subscribe to Pipeline Updates

**Send:**
```json
{
  "type": "subscribe",
  "pipeline_id": "rese_abc123"
}
```

**Receive:**
```json
{
  "type": "subscribed",
  "pipeline_id": "rese_abc123",
  "client_id": "client_xyz789",
  "timestamp": "2025-12-31T12:00:00Z"
}
```

### Real-Time Updates

**Receive:**
```json
{
  "type": "pipeline_update",
  "pipeline_id": "rese_abc123",
  "status": "running",
  "progress": {
    "phase_results": { ... },
    "aci_history": [0.85, 0.72]
  }
}
```

### Ping/Pong

**Send:**
```json
{
  "type": "ping"
}
```

**Receive:**
```json
{
  "type": "pong",
  "timestamp": "2025-12-31T12:00:00Z"
}
```

### Unsubscribe

**Send:**
```json
{
  "type": "unsubscribe",
  "pipeline_id": "rese_abc123"
}
```

---

## Configuration API

### RESEConfig

Master configuration object.

#### Constructor

```python
RESEConfig(
    environment: str = "development",
    version: str = "1.0.0",
    phase1: Phase1Config = default,
    phase2: Phase2Config = default,
    phase3: Phase3Config = default,
    phase4: Phase4Config = default,
    pipeline: PipelineConfig = default,
    api: APIConfig = default,
    monitoring: MonitoringConfig = default
) -> RESEConfig
```

---

#### to_dict()

Export configuration to dictionary.

```python
to_dict() -> Dict[str, Any]
```

---

#### save()

Save configuration to file.

```python
save(config_path: Optional[Path] = None) -> None
```

---

#### for_environment()

Create configuration for specific environment.

```python
for_environment(environment: Environment) -> RESEConfig
```

**Example:**
```python
from config import RESEConfig, Environment

config = RESEConfig()
production_config = config.for_environment(Environment.PRODUCTION)
```

---

### get_config()

Get current configuration (singleton).

```python
get_config() -> RESEConfig
```

**Example:**
```python
from config import get_config

config = get_config()
config.pipeline.enable_caching = True
```

---

## Error Handling

### Exception Hierarchy

```
PipelineError
├── PhaseExecutionError
├── ValidationError
└── CachingError
```

### Error Response Format

```json
{
  "error": "PhaseExecutionError",
  "detail": "Phase I failed: Constraint conflict detected",
  "timestamp": "2025-12-31T12:00:00Z",
  "pipeline_id": "rese_abc123",
  "phase": "phase1"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

### Error Handling Best Practices

```python
from rese.rese_pipeline import RESEPipeline, PhaseExecutionError

pipeline = RESEPipeline()

try:
    result = pipeline.run(problem)
except PhaseExecutionError as e:
    print(f"Phase failed: {e}")
    # Handle partial results
    if pipeline.current_result:
        print(f"Completed phases: {len(pipeline.current_result.phase_results)}")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Fix input and retry
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and report
```

---

## Type Definitions

### ProblemInput

```python
@dataclass
class ProblemInput:
    id: str                                          # Unique identifier
    description: str                                 # Problem description
    constraints: List[Dict[str, Any]]                # Constraints
    variables: Dict[str, Any]                        # Variables
    objective: Optional[str] = None                  # Objective function
    domain: str = "general"                          # Domain
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### PhaseResult

```python
@dataclass
class PhaseResult:
    phase_name: str                                  # Phase identifier
    status: PhaseStatus                              # Execution status
    output: Any = None                               # Phase output
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    pipeline_id: str                                 # Pipeline identifier
    problem_id: str                                  # Problem identifier
    status: PipelineStatus                           # Execution status
    phase_results: Dict[str, PhaseResult] = field(default_factory=dict)
    final_solution: Optional[Dict[str, Any]] = None  # Final solution
    aci_history: List[float] = field(default_factory=list)
    validation_score: float = 0.0                    # Validation score
    confidence: float = 0.0                          # Confidence
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
```

### Constraint

```python
@dataclass
class Constraint:
    id: str                                          # Constraint ID
    type: ConstraintType                             # HARD or SOFT
    description: str                                 # Human-readable description
    formalization: str                               # Formal representation
    source: str = "user"                             # Source
```

### ConstraintType

```python
class ConstraintType(Enum):
    HARD = "hard"                                    # Must satisfy
    SOFT = "soft"                                    # Prefer to satisfy
    PREFERENCE = "preference"                        # Nice to have
```

---

**API Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team
=======
# RESE API Reference

## Table of Contents

1. [Overview](#overview)
2. [Pipeline API](#pipeline-api)
3. [Phase I APIs](#phase-i-apis)
4. [Phase II APIs](#phase-ii-apis)
5. [Phase III APIs](#phase-iii-apis)
6. [Phase IV APIs](#phase-iv-apis)
7. [REST API Endpoints](#rest-api-endpoints)
8. [WebSocket API](#websocket-api)
9. [Configuration API](#configuration-api)
10. [Error Handling](#error-handling)
11. [Type Definitions](#type-definitions)

---

## Overview

### API Version

**Current Version:** v1.0.0
**Base Path:** `/api/v1`
**Content-Type:** `application/json`

### Authentication

Most endpoints require API key authentication:

```http
X-API-Key: your-api-key-here
```

Set via environment variable:
```bash
export RESE_API_KEYS="key1,key2,key3"
```

### Rate Limiting

- **Default:** 60 requests per minute
- **Header:** `X-RateLimit-Remaining: 45`
- **Error:** HTTP 429 when exceeded

---

## Pipeline API

### RESEPipeline

Main pipeline orchestrator for running RESE analysis.

#### Constructor

```python
RESEPipeline(config: Optional[RESEConfig] = None) -> RESEPipeline
```

**Parameters:**
- `config` (Optional[RESEConfig]): Configuration object. Uses default if None.

**Returns:**
- `RESEPipeline`: Pipeline instance

**Example:**
```python
from rese.rese_pipeline import RESEPipeline
from rese.config import get_config

# Use default config
pipeline = RESEPipeline()

# Use custom config
config = get_config()
config.pipeline.enable_caching = True
pipeline = RESEPipeline(config)
```

---

#### run()

Execute complete RESE pipeline on a problem.

```python
run(
    problem: ProblemInput,
    phases: Optional[List[str]] = None,
    use_cache: bool = True
) -> PipelineResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem` | ProblemInput | Yes | - | Input problem definition |
| `phases` | Optional[List[str]] | No | `['phase1', 'phase2', 'phase3', 'phase4']` | Phases to execute |
| `use_cache` | bool | No | `True` | Enable caching of intermediate results |

**Valid Phase Values:**
- `'phase1'`: Epistemic Audit (Φ₁, Φ₁.₅, Φ₂, Φ₃)
- `'phase2'`: Isomorphic Resonance (Ψ₁, Ψ₂, Ψ₃, I_mech)
- `'phase3'`: Monte Carlo Refinement (Γ₁, Γ₂, Γ₃, N_max)
- `'phase4'`: Architectural Synthesis (Δ₁, Δ₂, Δ₃)

**Returns:**
- `PipelineResult`: Complete execution results

**Raises:**
- `PhaseExecutionError`: If a phase fails fatally
- `ValidationError`: If input validation fails
- `CachingError`: If cache operation fails

**Example:**
```python
# Run all phases
result = pipeline.run(problem)

# Run only Phase I and III
result = pipeline.run(problem, phases=['phase1', 'phase3'])

# Run without cache
result = pipeline.run(problem, use_cache=False)
```

---

#### add_progress_callback()

Add callback function for progress updates.

```python
add_progress_callback(callback: Callable[[PipelineResult], None]) -> None
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `callback` | Callable[[PipelineResult], None] | Yes | Function to call with updates |

**Callback Signature:**
```python
def my_callback(result: PipelineResult) -> None:
    print(f"Status: {result.status.value}")
    print(f"Progress: {len(result.phase_results)}/4 phases")
```

**Example:**
```python
def track_progress(result):
    for phase_name, phase_result in result.phase_results.items():
        if phase_result.status == PhaseStatus.COMPLETED:
            print(f"{phase_name} completed: {phase_result.elapsed_seconds:.2f}s")

pipeline.add_progress_callback(track_progress)
result = pipeline.run(problem)
```

---

#### cancel()

Cancel currently running pipeline execution.

```python
cancel() -> None
```

**Example:**
```python
# Run in background
import threading

thread = threading.Thread(target=pipeline.run, args=(problem,))
thread.start()

# Cancel if needed
pipeline.cancel()
thread.join()
```

---

#### get_status()

Get current pipeline status.

```python
get_status() -> PipelineStatus
```

**Returns:**
- `PipelineStatus`: Current status enum

**Possible Values:**
- `PipelineStatus.IDLE`: Not running
- `PipelineStatus.RUNNING`: Currently executing
- `PipelineStatus.PAUSED`: Paused (future feature)
- `PipelineStatus.COMPLETED`: Finished successfully
- `PipelineStatus.FAILED`: Failed with error
- `PipelineStatus.CANCELLED`: Cancelled by user

**Example:**
```python
status = pipeline.get_status()
if status == PipelineStatus.RUNNING:
    print("Pipeline is running...")
```

---

#### get_progress()

Get detailed progress information.

```python
get_progress() -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Progress information

**Structure:**
```python
{
    'pipeline_id': str,
    'status': str,
    'elapsed_seconds': float,
    'phases': {
        'phase1': {
            'status': str,
            'elapsed': float,
            'metrics': Dict[str, Any]
        },
        # ... other phases
    }
}
```

**Example:**
```python
progress = pipeline.get_progress()
print(f"Pipeline: {progress['pipeline_id']}")
print(f"Status: {progress['status']}")

for phase_name, phase_info in progress['phases'].items():
    print(f"{phase_name}: {phase_info['status']}")
```

---

### Convenience Functions

#### run_rese()

Quick function to run RESE without explicit pipeline creation.

```python
run_rese(
    problem_description: str,
    constraints: List[Dict[str, Any]],
    variables: Dict[str, Any],
    config: Optional[RESEConfig] = None
) -> PipelineResult
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `problem_description` | str | Yes | Natural language problem description |
| `constraints` | List[Dict[str, Any]] | Yes | List of constraint dictionaries |
| `variables` | Dict[str, Any] | Yes | Problem variables |
| `config` | Optional[RESEConfig] | No | Configuration object |

**Returns:**
- `PipelineResult`: Execution results

**Example:**
```python
from rese.rese_pipeline import run_rese

result = run_rese(
    problem_description="Optimize production schedule",
    constraints=[
        {'id': 'c1', 'type': 'hard', 'description': 'cost < 10000'},
        {'id': 'c2', 'type': 'soft', 'description': 'maximize throughput'}
    ],
    variables={'cost': 'float', 'throughput': 'float'}
)
```

---

## Phase I APIs

### SymbolicConstraintEngine

Formalizes and manages constraints.

#### Constructor

```python
SymbolicConstraintEngine(max_constraints: int = 10000) -> SymbolicConstraintEngine
```

**Parameters:**
- `max_constraints` (int): Maximum number of constraints (default: 10000)

---

#### add_constraint()

Add a constraint to the engine.

```python
add_constraint(constraint: Constraint) -> None
```

**Parameters:**
- `constraint` (Constraint): Constraint object

**Example:**
```python
from core.symbolic_constraint_engine import SymbolicConstraintEngine, Constraint, ConstraintType

sce = SymbolicConstraintEngine()

constraint = Constraint(
    id='c1',
    type=ConstraintType.HARD,
    description='Cost must be below 1000',
    formalization='cost < 1000',
    source='user'
)
sce.add_constraint(constraint)
```

---

#### detect_conflicts()

Detect conflicting constraints.

```python
detect_conflicts() -> List[Conflict]
```

**Returns:**
- `List[Conflict]`: List of detected conflicts

**Conflict Structure:**
```python
{
    'constraint_ids': List[str],  # Conflicting constraint IDs
    'type': str,                   # Conflict type
    'description': str,            # Human-readable description
    'severity': str                # 'error', 'warning'
}
```

**Example:**
```python
conflicts = sce.detect_conflicts()
for conflict in conflicts:
    print(f"Conflict: {conflict['constraint_ids']}")
    print(f"Description: {conflict['description']}")
```

---

#### get_all_constraints()

Retrieve all constraints.

```python
get_all_constraints() -> List[Constraint]
```

**Returns:**
- `List[Constraint]`: All constraints

---

### CognitiveBiasDetector

Detects cognitive biases in constraints.

#### analyze_constraints()

Analyze constraints for biases.

```python
analyze_constraints(
    constraints: List[Constraint],
    threshold: float = 0.5
) -> BiasReport
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `constraints` | List[Constraint] | Yes | - | Constraints to analyze |
| `threshold` | float | No | 0.5 | Bias detection threshold |

**Returns:**
- `BiasReport`: Bias analysis report

**BiasReport Structure:**
```python
{
    'overall_bias_score': float,      # 0-1, higher = more biased
    'total_detections': int,           # Number of biases found
    'detections': [
        {
            'bias_type': str,          # Type of bias
            'constraint_id': str,      # Affected constraint
            'severity': str,           # 'low', 'medium', 'high'
            'description': str,        # Description
            'recommendation': str      # How to fix
        }
    ]
}
```

**Bias Types:**
- `confirmation_bias`: Seeking confirming evidence only
- `anchoring_bias`: Over-relying on initial information
- `availability_bias`: Overweighting easily recalled examples
- `sunk_cost_bias`: Continuing failing approaches

**Example:**
```python
from phase1.cognitive_biases import CognitiveBiasDetector

detector = CognitiveBiasDetector()
report = detector.analyze_constraints(constraints)

print(f"Bias Score: {report.overall_bias_score:.2f}")
print(f"Detections: {report.total_detections}")

for detection in report.detections:
    print(f"{detection['bias_type']}: {detection['description']}")
    print(f"Recommendation: {detection['recommendation']}")
```

---

### TacitAssumptionMiner (Φ₁.₅)

Mines hidden assumptions from constraint set.

#### mine_assumptions()

Extract tacit assumptions.

```python
mine_assumptions(
    constraints: List[Constraint],
    domain: str,
    max_assumptions: int = 100,
    threshold: float = 0.6
) -> List[Assumption]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `constraints` | List[Constraint] | Yes | - | Constraints to analyze |
| `domain` | str | Yes | - | Problem domain |
| `max_assumptions` | int | No | 100 | Maximum assumptions to return |
| `threshold` | float | No | 0.6 | Confidence threshold |

**Returns:**
- `List[Assumption]`: List of assumptions

**Assumption Structure:**
```python
{
    'id': str,
    'description': str,           # Assumption description
    'type': str,                  # Category of assumption
    'confidence': float,          # 0-1 confidence score
    'source': str,                # 'failure_db', 'domain_kb', 'inference'
    'justification': str          # Why this assumption was made
}
```

**Example:**
```python
from phase1.tacit_assumption_miner import TacitAssumptionMiner

miner = TacitAssumptionMiner()
assumptions = miner.mine_assumptions(
    constraints=constraints,
    domain='bridge_design',
    threshold=0.7
)

print(f"Found {len(assumptions)} tacit assumptions:")
for assumption in assumptions:
    print(f"- {assumption['description']} (confidence: {assumption['confidence']:.2f})")
```

---

## Phase II APIs

### IMechValidator

Validates mechanistic isomorphisms between domains.

#### compare_domains()

Compare two domains for isomorphism.

```python
compare_domains(
    source_domain: Domain,
    target_domain: Domain,
    algorithm: str = 'weisfeiler_lehman'
) -> SimilarityResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_domain` | Domain | Yes | - | Source domain |
| `target_domain` | Domain | Yes | - | Target domain |
| `algorithm` | str | No | `'weisfeiler_lehman'` | Isomorphism algorithm |

**Valid Algorithms:**
- `'weisfeiler_lehman'`: Fast, scalable graph isomorphism
- `'vf2'`: Exact graph isomorphism (slower)
- `'subgraph'`: Partial isomorphism detection

**Returns:**
- `SimilarityResult`: Similarity analysis result

**SimilarityResult Structure:**
```python
{
    'score': float,                     # 0-1 overall similarity
    'structural_similarity': float,      # Graph topology similarity
    'causal_similarity': float,          # Causal structure similarity
    'interventional_similarity': float,  # Intervention response similarity
    'is_isomorphic': bool,               # True if similarity > threshold
    'confidence': float,                 # Statistical confidence
    'mapping': Dict[str, str]           # Variable mapping
}
```

**Example:**
```python
from phase2.imech import IMechValidator, Domain

validator = IMechValidator()

# Define domains
source = Domain(
    id='chemical_reactor',
    name='Chemical Reactor',
    variables={'temperature', 'pressure', 'yield'},
    constraints=['yield = f(temp, pressure)']
)

target = Domain(
    id='electrical_circuit',
    name='Circuit',
    variables={'voltage', 'current', 'power'},
    constraints=['power = g(voltage, current)']
)

# Compare
similarity = validator.compare_domains(source, target)

print(f"Isomorphism Score: {similarity.score:.2f}")
print(f"Structural: {similarity.structural_similarity:.2f}")
print(f"Causal: {similarity.causal_similarity:.2f}")
print(f"Is Isomorphic: {similarity.is_isomorphic}")
```

---

#### generate_isomorphism_proof()

Generate Lean 4 proof for isomorphism.

```python
generate_isomorphism_proof(
    source_domain: Domain,
    target_domain: Domain
) -> Proof
```

**Parameters:**
- `source_domain` (Domain): Source domain
- `target_domain` (Domain): Target domain

**Returns:**
- `Proof`: Lean 4 proof object

**Proof Structure:**
```python
{
    'is_valid': bool,
    'lean4_code': str,           # Lean 4 proof code
    'verification_status': str,   # 'verified', 'pending', 'failed'
    'theorem': str,              # Proven theorem
    'timestamp': str
}
```

**Example:**
```python
proof = validator.generate_isomorphism_proof(source, target)

if proof['is_valid']:
    print(f"Theorem: {proof['theorem']}")
    print(f"Lean 4 Code:\n{proof['lean4_code']}")
```

---

### SemanticMatcher

Maps ontologies using semantic similarity.

#### find_similar_domains()

Find semantically similar domains.

```python
find_similar_domains(
    problem_description: str,
    similarity_threshold: float = 0.7,
    max_results: int = 10
) -> List[DomainMatch]
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `problem_description` | str | Yes | - | Problem description |
| `similarity_threshold` | float | No | 0.7 | Minimum similarity |
| `max_results` | int | No | 10 | Maximum results |

**Returns:**
- `List[DomainMatch]`: List of matching domains

**DomainMatch Structure:**
```python
{
    'domain_id': str,
    'domain_name': str,
    'similarity': float,           # 0-1 semantic similarity
    'description': str,
    'key_concepts': List[str]      # Matching concepts
}
```

**Example:**
```python
from phase2.ontology_components.semantic_matcher import SemanticMatcher

matcher = SemanticMatcher()
matches = matcher.find_similar_domains(
    problem_description="Optimize neural network architecture for image classification",
    similarity_threshold=0.75
)

for match in matches:
    print(f"{match['domain_name']}: {match['similarity']:.2f}")
    print(f"  Concepts: {', '.join(match['key_concepts'])}")
```

---

## Phase III APIs

### MCTSSearch (Γ₂)

Monte Carlo Tree Search with ACI guidance.

#### Constructor

```python
MCTSSearch(
    exploration_constant: float = 1.41,
    max_iterations: int = 1000,
    playout_depth: int = 100,
    aci_guided: bool = True,
    parallel_agents: int = 4
) -> MCTSSearch
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `exploration_constant` | float | No | 1.41 | UCB exploration constant (C) |
| `max_iterations` | int | No | 1000 | Maximum MCTS iterations |
| `playout_depth` | int | No | 100 | Max playout depth |
| `aci_guided` | bool | No | True | Use ACI for exploration bonus |
| `parallel_agents` | int | No | 4 | Parallel search agents |

---

#### search()

Execute MCTS search for optimal solution.

```python
search(
    problem: ProblemInput,
    constraints: List[Constraint]
) -> Solution
```

**Parameters:**
- `problem` (ProblemInput): Problem definition
- `constraints` (List[Constraint]): Constraints

**Returns:**
- `Solution`: Best solution found

**Solution Structure:**
```python
{
    'variables': Dict[str, Any],   # Variable assignments
    'aci': float,                  # Final ACI
    'confidence': float,           # Statistical confidence
    'iterations': int,             # Iterations used
    'converged': bool,             # Convergence status
    'value': float                 # Objective value
}
```

**Example:**
```python
from phase3.mcts_search import MCTSSearch

mcts = MCTSSearch(
    max_iterations=5000,
    aci_guided=True
)

solution = mcts.search(problem, constraints)

print(f"Best Solution: {solution['variables']}")
print(f"ACI: {solution['aci']:.3f}")
print(f"Confidence: {solution['confidence']:.2f}")
print(f"Converged: {solution['converged']}")
```

---

### ACICalculator (Γ₁)

Calculates Algorithmic Complexity Index.

#### calculate_solution()

Calculate ACI for a solution.

```python
calculate_solution(
    constraints: List[Constraint],
    solution_variables: Dict[str, Any],
    domain: str
) -> float
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `constraints` | List[Constraint] | Yes | Constraints |
| `solution_variables` | Dict[str, Any] | Yes | Solution variable assignments |
| `domain` | str | Yes | Problem domain |

**Returns:**
- `float`: ACI value (0-1)

**Example:**
```python
from gamma1.core.aci_calculator import ACICalculator

calculator = ACICalculator()

aci = calculator.calculate_solution(
    constraints=constraints,
    solution_variables={'x': 42, 'y': 17},
    domain='optimization'
)

print(f"ACI: {aci:.3f}")
```

---

### StatisticalValidator (Γ₃)

Validates solutions with statistical tests.

#### validate()

Validate solution statistically.

```python
validate(
    solution: Solution,
    confidence_level: float = 0.95,
    n_bootstrap_samples: int = 1000
) -> ValidationResult
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `solution` | Solution | Yes | - | Solution to validate |
| `confidence_level` | float | No | 0.95 | Confidence level |
| `n_bootstrap_samples` | int | No | 1000 | Bootstrap iterations |

**Returns:**
- `ValidationResult`: Validation result

**ValidationResult Structure:**
```python
{
    'is_valid': bool,
    'p_value': float,                    # Statistical significance
    'confidence_interval': Tuple[float, float],
    'effect_size': float,
    'power': float,
    'recommendations': List[str]
}
```

**Example:**
```python
from phase3.statistical_validator import StatisticalValidator

validator = StatisticalValidator(confidence_level=0.95)

validation = validator.validate(solution)

print(f"Valid: {validation['is_valid']}")
print(f"P-value: {validation['p_value']:.4f}")
print(f"95% CI: {validation['confidence_interval']}")
```

---

## Phase IV APIs

### Delta3Validator (Δ₃)

Validates ACI reduction.

#### validate()

Validate final solution.

```python
validate(
    problem: Problem,
    solution: RESESolution
) -> ValidationResult
```

**Parameters:**
- `problem` (Problem): Original problem
- `solution` (RESESolution): RESE solution

**Returns:**
- `ValidationResult`: Validation result

**ValidationResult Structure:**
```python
{
    'is_valid': bool,
    'validation_score': float,          # 0-1
    'confidence': float,                # 0-1
    'aci_reduction': float,             # 0-1 (target: ≥0.2)
    'p_value': float,
    'significance': str,                # 'significant', 'marginal', 'not_significant'
    'recommendations': List[str]
}
```

**Example:**
```python
from phase4.aci_reduction_validator import Delta3Validator, Problem, RESESolution

validator = Delta3Validator(
    validation_threshold=0.7,
    min_aci_reduction=0.2
)

problem = Problem(
    id='problem_1',
    description='Optimize system',
    constraints=constraints,
    variables=variables
)

solution = RESESolution(
    problem_id='problem_1',
    solution={'x': 42, 'y': 17},
    aci_history=[0.85, 0.72, 0.55, 0.28, 0.15],
    stage_results={...}
)

validation = validator.validate(problem, solution)

print(f"Valid: {validation.is_valid}")
print(f"Score: {validation.validation_score:.2f}")
print(f"ACI Reduction: {validation.aci_reduction * 100:.1f}%")
```

---

### ArchitectureAssembler (Δ₁)

Assembles final solution architecture.

#### assemble()

Assemble architecture from phase outputs.

```python
assemble(
    phase1_output: Any,
    phase2_output: Any,
    phase3_output: Any
) -> Architecture
```

**Parameters:**
- `phase1_output` (Any): Phase I output
- `phase2_output` (Any): Phase II output
- `phase3_output` (Any): Phase III output

**Returns:**
- `Architecture`: Assembled architecture

**Architecture Structure:**
```python
{
    'components': List[Component],
    'integration_strategy': str,        # 'hierarchical', 'flat', 'hybrid'
    'validation_score': float,
    'metadata': Dict[str, Any]
}
```

---

## REST API Endpoints

### Base URL

```
http://localhost:8000/api/v1
```

### Health Endpoints

#### GET /health

Check API health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-31T12:00:00Z",
  "uptime_seconds": 3600.5
}
```

---

### Pipeline Endpoints

#### POST /pipeline/run

Submit problem to RESE pipeline.

**Request:**
```http
POST /api/v1/pipeline/run
Content-Type: application/json
X-API-Key: your-api-key

{
  "description": "Optimize production schedule",
  "constraints": [
    {
      "id": "c1",
      "type": "hard",
      "description": "cost < 10000",
      "formalization": "cost < 10000"
    }
  ],
  "variables": {
    "cost": "float",
    "throughput": "float"
  },
  "phases": ["phase1", "phase2", "phase3", "phase4"],
  "use_cache": true
}
```

**Response:**
```json
{
  "pipeline_id": "rese_abc123",
  "problem_id": "problem_def456",
  "status": "completed",
  "final_solution": {
    "cost": 8500.0,
    "throughput": 1250.0
  },
  "aci_history": [0.85, 0.72, 0.55, 0.28, 0.15],
  "validation_score": 0.87,
  "confidence": 0.85,
  "elapsed_seconds": 45.3,
  "phase_results": { ... }
}
```

---

#### GET /pipeline/{pipeline_id}/status

Get pipeline status.

**Response:**
```json
{
  "pipeline_id": "rese_abc123",
  "problem_id": "problem_def456",
  "status": "running",
  "elapsed_seconds": 23.5,
  "phases": {
    "phase1": {
      "status": "completed",
      "elapsed": 5.2,
      "metrics": { ... }
    },
    "phase2": {
      "status": "running",
      "elapsed": 18.3,
      "metrics": { ... }
    }
  }
}
```

---

#### GET /pipeline/{pipeline_id}/result

Get complete pipeline result.

**Response:** Same as POST /pipeline/run

---

#### DELETE /pipeline/{pipeline_id}

Cancel running pipeline.

**Response:**
```json
{
  "message": "Pipeline rese_abc123 cancelled",
  "pipeline_id": "rese_abc123"
}
```

---

### Admin Endpoints

#### GET /admin/stats

Get system statistics.

**Response:**
```json
{
  "active_pipelines": 3,
  "stored_results": 127,
  "websocket_connections": 5,
  "uptime_seconds": 86400.0
}
```

---

#### POST /admin/cache/clear

Clear pipeline cache.

**Response:**
```json
{
  "message": "Cache cleared"
}
```

---

## WebSocket API

### Connection

Connect to WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/pipeline/{pipeline_id}');
```

### Subscribe to Pipeline Updates

**Send:**
```json
{
  "type": "subscribe",
  "pipeline_id": "rese_abc123"
}
```

**Receive:**
```json
{
  "type": "subscribed",
  "pipeline_id": "rese_abc123",
  "client_id": "client_xyz789",
  "timestamp": "2025-12-31T12:00:00Z"
}
```

### Real-Time Updates

**Receive:**
```json
{
  "type": "pipeline_update",
  "pipeline_id": "rese_abc123",
  "status": "running",
  "progress": {
    "phase_results": { ... },
    "aci_history": [0.85, 0.72]
  }
}
```

### Ping/Pong

**Send:**
```json
{
  "type": "ping"
}
```

**Receive:**
```json
{
  "type": "pong",
  "timestamp": "2025-12-31T12:00:00Z"
}
```

### Unsubscribe

**Send:**
```json
{
  "type": "unsubscribe",
  "pipeline_id": "rese_abc123"
}
```

---

## Configuration API

### RESEConfig

Master configuration object.

#### Constructor

```python
RESEConfig(
    environment: str = "development",
    version: str = "1.0.0",
    phase1: Phase1Config = default,
    phase2: Phase2Config = default,
    phase3: Phase3Config = default,
    phase4: Phase4Config = default,
    pipeline: PipelineConfig = default,
    api: APIConfig = default,
    monitoring: MonitoringConfig = default
) -> RESEConfig
```

---

#### to_dict()

Export configuration to dictionary.

```python
to_dict() -> Dict[str, Any]
```

---

#### save()

Save configuration to file.

```python
save(config_path: Optional[Path] = None) -> None
```

---

#### for_environment()

Create configuration for specific environment.

```python
for_environment(environment: Environment) -> RESEConfig
```

**Example:**
```python
from config import RESEConfig, Environment

config = RESEConfig()
production_config = config.for_environment(Environment.PRODUCTION)
```

---

### get_config()

Get current configuration (singleton).

```python
get_config() -> RESEConfig
```

**Example:**
```python
from config import get_config

config = get_config()
config.pipeline.enable_caching = True
```

---

## Error Handling

### Exception Hierarchy

```
PipelineError
├── PhaseExecutionError
├── ValidationError
└── CachingError
```

### Error Response Format

```json
{
  "error": "PhaseExecutionError",
  "detail": "Phase I failed: Constraint conflict detected",
  "timestamp": "2025-12-31T12:00:00Z",
  "pipeline_id": "rese_abc123",
  "phase": "phase1"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

### Error Handling Best Practices

```python
from rese.rese_pipeline import RESEPipeline, PhaseExecutionError

pipeline = RESEPipeline()

try:
    result = pipeline.run(problem)
except PhaseExecutionError as e:
    print(f"Phase failed: {e}")
    # Handle partial results
    if pipeline.current_result:
        print(f"Completed phases: {len(pipeline.current_result.phase_results)}")
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Fix input and retry
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and report
```

---

## Type Definitions

### ProblemInput

```python
@dataclass
class ProblemInput:
    id: str                                          # Unique identifier
    description: str                                 # Problem description
    constraints: List[Dict[str, Any]]                # Constraints
    variables: Dict[str, Any]                        # Variables
    objective: Optional[str] = None                  # Objective function
    domain: str = "general"                          # Domain
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### PhaseResult

```python
@dataclass
class PhaseResult:
    phase_name: str                                  # Phase identifier
    status: PhaseStatus                              # Execution status
    output: Any = None                               # Phase output
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
```

### PipelineResult

```python
@dataclass
class PipelineResult:
    pipeline_id: str                                 # Pipeline identifier
    problem_id: str                                  # Problem identifier
    status: PipelineStatus                           # Execution status
    phase_results: Dict[str, PhaseResult] = field(default_factory=dict)
    final_solution: Optional[Dict[str, Any]] = None  # Final solution
    aci_history: List[float] = field(default_factory=list)
    validation_score: float = 0.0                    # Validation score
    confidence: float = 0.0                          # Confidence
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
```

### Constraint

```python
@dataclass
class Constraint:
    id: str                                          # Constraint ID
    type: ConstraintType                             # HARD or SOFT
    description: str                                 # Human-readable description
    formalization: str                               # Formal representation
    source: str = "user"                             # Source
```

### ConstraintType

```python
class ConstraintType(Enum):
    HARD = "hard"                                    # Must satisfy
    SOFT = "soft"                                    # Prefer to satisfy
    PREFERENCE = "preference"                        # Nice to have
```

---

**API Version:** 1.0.0
**Last Updated:** 2025-12-31
**Authors:** RESE Development Team
>>>>>>> 1cb9c5e35 (update)
