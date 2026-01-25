# RESE API Reference

**Recursive Epistemic Solvability Engine**
**Version:** 1.0.0
**Last Updated:** 2025-12-31

---

## Table of Contents

1. [Pipeline API](#pipeline-api)
2. [Core Module API](#core-module-api)
3. [Phase I API](#phase-i-api)
4. [Phase II API](#phase-ii-api)
5. [Phase III API](#phase-iii-api)
6. [Phase IV API](#phase-iv-api)
7. [Configuration API](#configuration-api)
8. [Data Structures](#data-structures)

---

## Pipeline API

### RESEPipeline

**Module:** `rese.rese_pipeline`

**Main pipeline orchestrator for all 4 RESE phases.**

---

#### `__init__(config: Optional[RESEConfig] = None)`

Initialize RESE pipeline.

**Parameters:**
- `config` (Optional[RESEConfig]): Configuration object (uses default if None)

**Returns:**
- RESEPipeline instance

**Example:**
```python
from rese.rese_pipeline import RESEPipeline
from rese.config import RESEConfig

# With default config
pipeline = RESEPipeline()

# With custom config
config = RESEConfig(environment="production")
pipeline = RESEPipeline(config)
```

---

#### `run(problem: ProblemInput, phases: Optional[List[str]] = None, use_cache: bool = True) -> PipelineResult`

Run complete RESE pipeline.

**Parameters:**
- `problem` (ProblemInput): Input problem definition
- `phases` (Optional[List[str]]): List of phases to run (default: all phases)
  - Valid values: `["phase1", "phase2", "phase3", "phase4"]`
- `use_cache` (bool): Whether to use cached intermediate results

**Returns:**
- PipelineResult: Complete execution result

**Raises:**
- PipelineError: If pipeline execution fails
- ValidationError: If input validation fails

**Example:**
```python
from rese.rese_pipeline import RESEPipeline, ProblemInput

pipeline = RESEPipeline()

problem = ProblemInput(
    id="tsp_50",
    description="Traveling Salesman Problem with 50 cities",
    constraints=[...],
    variables={"num_cities": 50}
)

result = pipeline.run(problem)
print(f"Status: {result.status}")
print(f"Confidence: {result.confidence}")
```

---

#### `add_progress_callback(callback: Callable[[PipelineResult], None]) -> None`

Add progress callback function.

**Parameters:**
- `callback` (Callable): Function called with pipeline updates

**Example:**
```python
def progress_handler(result):
    print(f"Progress: {result.status}")

pipeline.add_progress_callback(progress_handler)
```

---

#### `cancel() -> None`

Cancel current pipeline execution.

**Example:**
```python
# In another thread
pipeline.cancel()
```

---

#### `get_status() -> PipelineStatus`

Get current pipeline status.

**Returns:**
- PipelineStatus enum: Current status

**Example:**
```python
status = pipeline.get_status()
print(status)  # PipelineStatus.RUNNING
```

---

#### `get_progress() -> Dict[str, Any]`

Get current progress information.

**Returns:**
- Dict with keys:
  - `pipeline_id` (str): Pipeline identifier
  - `status` (str): Current status
  - `elapsed_seconds` (float): Time elapsed
  - `phases` (Dict[str, Dict]): Phase-specific progress

**Example:**
```python
progress = pipeline.get_progress()
for phase_name, phase_info in progress['phases'].items():
    print(f"{phase_name}: {phase_info['status']}")
```

---

### Convenience Functions

#### `run_rese(problem_description: str, constraints: List[Dict], variables: Dict, config: Optional[RESEConfig] = None) -> PipelineResult`

Convenience function to run RESE pipeline.

**Parameters:**
- `problem_description` (str): Human-readable problem description
- `constraints` (List[Dict]): List of constraint dictionaries
- `variables` (Dict): Problem variables
- `config` (Optional[RESEConfig]): Configuration

**Returns:**
- PipelineResult

**Example:**
```python
from rese.rese_pipeline import run_rese

result = run_rese(
    problem_description="Optimize delivery routes",
    constraints=[...],
    variables={"num_locations": 50}
)
```

---

## Core Module API

### SymbolicConstraintEngine

**Module:** `rese.core.symbolic_constraint_engine`

**Manages constraints and their dependencies.**

---

#### `__init__()`

Initialize constraint engine.

**Example:**
```python
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine

sce = SymbolicConstraintEngine()
```

---

#### `add_constraint(constraint: Constraint) -> None`

Add constraint to engine.

**Parameters:**
- `constraint` (Constraint): Constraint to add

**Raises:**
- ValueError: If constraint validation fails

**Example:**
```python
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

constraint = Constraint(
    id="c1",
    type=ConstraintType.HARD,
    description="All variables must be positive",
    formalization="∀ x ∈ variables: x > 0",
    source="user"
)
sce.add_constraint(constraint)
```

---

#### `get_constraint(constraint_id: str) -> Optional[Constraint]`

Get constraint by ID.

**Parameters:**
- `constraint_id` (str): Constraint identifier

**Returns:**
- Constraint if found, None otherwise

---

#### `get_all_constraints() -> List[Constraint]`

Get all constraints.

**Returns:**
- List[Constraint]: All constraints in engine

---

#### `detect_conflicts() -> List[Tuple[str, str]]`

Detect conflicting constraints.

**Returns:**
- List[Tuple[str, str]]: List of (id1, id2) conflict pairs

**Example:**
```python
conflicts = sce.detect_conflicts()
for c1, c2 in conflicts:
    print(f"Conflict: {c1} vs {c2}")
```

---

#### `get_execution_order() -> List[str]`

Get topological sort of constraints based on dependencies.

**Returns:**
- List[str]: Constraint IDs in execution order

---

#### `validate() -> Dict[str, Any]`

Validate all constraints.

**Returns:**
- Dict with keys:
  - `is_valid` (bool): Overall validity
  - `errors` (List[str]): Validation errors
  - `warnings` (List[str]): Validation warnings

---

### Constraint

**Dataclass representing a formal constraint.**

**Fields:**
- `id` (str): Unique identifier
- `type` (ConstraintType): HARD, SOFT, or PREFERENCE
- `description` (str): Human-readable description
- `formalization` (str): Lean 4 representation
- `source` (str): Origin of constraint
- `dependencies` (List[str]): IDs of dependent constraints
- `verified` (bool): Whether verified in Lean 4
- `lean_theorem` (Optional[str]): Lean 4 theorem proof

**Methods:**
- `is_hard() -> bool`: Check if hard constraint
- `is_verified() -> bool`: Check if verified in Lean 4

---

### LogicToLossTranslator

**Module:** `rese.core.logic_to_loss_translation`

**Translates formal logic to differentiable loss functions.**

---

#### `translate(logic: str, variables: Dict[str, Tensor]) -> Callable[[], Tensor]`

Translate logic constraint to loss function.

**Parameters:**
- `logic` (str): Formal logic statement (Lean 4 syntax)
- `variables` (Dict[str, Tensor]): Variable tensors

**Returns:**
- Callable function that returns loss tensor

**Example:**
```python
from rese.core.logic_to_loss_translation import LogicToLossTranslator
import torch

translator = LogicToLossTranslator()

loss_fn = translator.translate(
    logic="∀ x: x > 0",
    variables={"x": torch.tensor([1, 2, 3])}
)

loss = loss_fn()
loss.backward()
```

---

### DITOOptimizer

**Module:** `rese.core.dito_optimizer`

**O(n log n) contradiction detection in large constraint sets.**

---

#### `add_constraint(constraint: Constraint) -> None`

Add constraint to optimizer.

**Parameters:**
- `constraint` (Constraint): Constraint to add

---

#### `detect_contradictions() -> List[Tuple[str, str]]`

Detect contradictions in O(n log n).

**Returns:**
- List[Tuple[str, str]]: Contradictory constraint pairs

**Example:**
```python
from rese.core.dito_optimizer import DITOOptimizer

optimizer = DITOOptimizer()
for constraint in large_constraint_set:
    optimizer.add_constraint(constraint)

contradictions = optimizer.detect_contradictions()
print(f"Found {len(contradictions)} contradictions")
```

---

#### `get_statistics() -> Dict[str, Any]`

Get optimizer statistics.

**Returns:**
- Dict with keys:
  - `num_constraints` (int): Total constraints
  - `num_contradictions` (int): Contradictions found
  - `time` (float): Computation time
  - `graph_size` (int): Knowledge graph size

---

## Phase I API

### TacitAssumptionMiner

**Module:** `rese.phase1.tacit_assumption_miner`

**Discovers hidden constraints from null results.**

---

#### `mine(failure_cases: List[Dict], constraints: List[Constraint], num_assumptions: int = 10) -> List[Assumption]`

Mine tacit assumptions from failures.

**Parameters:**
- `failure_cases` (List[Dict]): Known failure cases
- `constraints` (List[Constraint]): Existing constraints
- `num_assumptions` (int): Maximum assumptions to return

**Returns:**
- List[Assumption]: Discovered assumptions

**Example:**
```python
from rese.phase1.tacit_assumption_miner import TacitAssumptionMiner

miner = TacitAssumptionMiner()
assumptions = miner.mine(
    failure_cases=known_failures,
    constraints=existing_constraints,
    num_assumptions=10
)

for assumption in assumptions:
    print(f"{assumption.description} (confidence: {assumption.confidence:.2f})")
```

---

### CognitiveBiasDetector

**Module:** `rese.phase1.cognitive_biases`

**Detects and mitigates cognitive biases.**

---

#### `analyze_constraints(constraints: List[Constraint]) -> BiasReport`

Analyze constraints for cognitive biases.

**Parameters:**
- `constraints` (List[Constraint]): Constraints to analyze

**Returns:**
- BiasReport: Bias detection report

**BiasReport Fields:**
- `overall_bias_score` (float): Overall bias score [0, 1]
- `total_detections` (int): Total biases detected
- `by_severity` (Dict[Severity, int]): Count by severity
- `detections` (List[BiasDetection]): All detections

**Example:**
```python
from rese.phase1.cognitive_biases import CognitiveBiasDetector

detector = CognitiveBiasDetector()
report = detector.analyze_constraints(constraints)

print(f"Bias score: {report.overall_bias_score:.2f}")
print(f"Total detections: {report.total_detections}")

for detection in report.detections:
    print(f"{detection.bias_type.value}: {detection.description}")
    print(f"Severity: {detection.severity.name}")
```

---

#### `debias(constraints: List[Constraint], report: BiasReport) -> List[Constraint]`

Debias constraints based on report.

**Parameters:**
- `constraints` (List[Constraint]): Original constraints
- `report` (BiasReport): Bias detection report

**Returns:**
- List[Constraint]: Debiased constraints

---

## Phase II API

### IMechValidator

**Module:** `rese.phase2.imech`

**Validates mechanistic similarity between domains.**

---

#### `compare_domains(source: Domain, target: Domain) -> IsomorphismResult`

Compare two domains for mechanistic similarity.

**Parameters:**
- `source` (Domain): Source domain
- `target` (Domain): Target domain

**Returns:**
- IsomorphismResult: Comparison result

**IsomorphismResult Fields:**
- `score` (float): Similarity score [0, 1]
- `confidence` (float): Confidence in score [0, 1]
- `shared_structure` (List[str]): Shared structural elements
- `differences` (List[str]): Key differences
- `transfer_recommended` (bool): Whether knowledge transfer recommended

**Example:**
```python
from rese.phase2.imech import IMechValidator, Domain

validator = IMechValidator()

source = Domain(id="tsp", name="Traveling Salesman", ...)
target = Domain(id="vrp", name="Vehicle Routing", ...)

result = validator.compare_domains(source, target)

print(f"Similarity: {result.score:.2f}")
print(f"Confidence: {result.confidence:.2f}")

if result.transfer_recommended:
    transferred = validator.transfer_knowledge(source, target)
```

---

#### `transfer_knowledge(source: Domain, target: Domain) -> List[Constraint]`

Transfer knowledge from source to target domain.

**Parameters:**
- `source` (Domain): Source domain
- `target` (Domain): Target domain

**Returns:**
- List[Constraint]: Transferred constraints

---

### ConstraintInverter

**Module:** `rese.phase2.psi3`

**Inverts constraints to reduce search complexity.**

---

#### `invert(constraints: List[Constraint]) -> List[Constraint]`

Invert constraints for complexity reduction.

**Parameters:**
- `constraints` (List[Constraint]): Original constraints

**Returns:**
- List[Constraint]: Inverted constraints

**Example:**
```python
from rese.phase2.psi3 import ConstraintInverter

inverter = ConstraintInverter()
inverted = inverter.invert(constraints)

print(f"Original: {len(constraints)} constraints")
print(f"Inverted: {len(inverted)} constraints")
print(f"Complexity reduction: 2^{len(constraints)} → 2^{len(inverted)}")
```

---

## Phase III API

### ACICalculator

**Module:** `rese.gamma1.core.aci_calculator`

**Calculates Algorithmic Complexity Index.**

---

#### `__init__(alpha: float = 0.35, beta: float = 0.35, gamma: float = 0.30, use_cache: bool = True)`

Initialize ACI calculator.

**Parameters:**
- `alpha` (float): Weight for (1-H) component
- `beta` (float): Weight for C component
- `gamma` (float): Weight for S component
- `use_cache` (bool): Enable result caching

**Note:** alpha + beta + gamma should equal 1.0

---

#### `calculate(csp_instance: CSPInstance) -> ACIResult`

Calculate ACI for CSP instance.

**Parameters:**
- `csp_instance` (CSPInstance): CSP instance to analyze

**Returns:**
- ACIResult: ACI calculation result

**ACIResult Fields:**
- `ACI` (float): Final ACI score [0, 1]
- `components` (Dict[str, float]): Component breakdown
  - `disorder_entropy` (float): H value [0, 1]
  - `causal_coherence` (float): C value [0, 1]
  - `solvability_index` (float): S value [0, 1]
- `confidence` (float): Confidence in score [0, 1]
- `interpretation` (Dict): Human-readable interpretation
- `recommendation` (Dict): Search strategy recommendation
- `computation_time` (float): Time in seconds
- `cached` (bool): Whether from cache

**Example:**
```python
from rese.gamma1.core.aci_calculator import ACICalculator
from rese.gamma1.core.csp_models import CSPInstance

aci_calc = ACICalculator(alpha=0.35, beta=0.35, gamma=0.30)
result = aci_calc.calculate(csp_instance)

print(f"ACI = {result.ACI:.3f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"H={result.components['disorder_entropy']:.3f}, "
      f"C={result.components['causal_coherence']:.3f}, "
      f"S={result.components['solvability_index']:.3f}")
```

---

### MCTSSearch

**Module:** `rese.phase3.mcts_search`

**ACI-guided Monte Carlo Tree Search.**

---

#### `__init__(aci_calculator: ACICalculator, iterations: int = 1000, exploration_constant: float = 1.41, parallel_agents: int = 4)`

Initialize MCTS search.

**Parameters:**
- `aci_calculator` (ACICalculator): ACI calculator for guidance
- `iterations` (int): Number of MCTS iterations
- `exploration_constant` (float): UCB exploration constant (default: sqrt(2))
- `parallel_agents` (int): Number of parallel search agents

---

#### `search(initial_state: Any) -> MCTSResult`

Run MCTS search from initial state.

**Parameters:**
- `initial_state` (Any): Initial problem state

**Returns:**
- MCTSResult: Search result

**MCTSResult Fields:**
- `best_value` (float): Best value found
- `best_path` (List[Any]): Path to best solution
- `aci_history` (List[float]): ACI progression
- `iterations` (int): Iterations performed
- `converged` (bool): Whether converged
- `computation_time` (float): Time in seconds

**Example:**
```python
from rese.phase3.mcts_search import MCTSSearch
from rese.gamma1.core.aci_calculator import ACICalculator

aci_calc = ACICalculator()
search = MCTSSearch(
    aci_calculator=aci_calc,
    iterations=1000
)

result = search.search(initial_state)

print(f"Best value: {result.best_value:.2f}")
print(f"ACI progression: {result.aci_history}")
```

---

### ConvergenceController

**Module:** `rese.phase3.convergence_controller`

**Controls MCTS convergence with early stopping.**

---

#### `__init__(patience: int = 50, min_delta: float = 0.001, max_iterations: int = 10000)`

Initialize convergence controller.

**Parameters:**
- `patience` (int): Iterations without improvement before stopping
- `min_delta` (float): Minimum improvement to reset patience
- `max_iterations` (int): Maximum iterations

---

#### `should_stop(current_value: float) -> bool`

Check if should stop iteration.

**Parameters:**
- `current_value` (float): Current iteration value

**Returns:**
- bool: True if should stop

**Example:**
```python
from rese.phase3.convergence_controller import ConvergenceController

controller = ConvergenceController(patience=50, min_delta=0.001)

for i in range(10000):
    value = run_iteration()

    if controller.should_stop(value):
        print(f"Converged at iteration {i}")
        break
```

---

#### `reset() -> None`

Reset convergence state.

---

## Phase IV API

### Delta3Validator

**Module:** `rese.phase4.aci_reduction_validator`

**Validates solution by ACI reduction (non-circular).**

---

#### `__init__(validation_threshold: float = 0.7, min_aci_reduction: float = 0.2, holdout_ratio: float = 0.2)`

Initialize validator.

**Parameters:**
- `validation_threshold` (float): Minimum validation score
- `min_aci_reduction` (float): Minimum ACI reduction required
- `holdout_ratio` (float): Ratio of data for holdout set

---

#### `validate(problem: Problem, solution: RESESolution) -> ValidationResult`

Validate solution.

**Parameters:**
- `problem` (Problem): Original problem
- `solution` (RESESolution): Proposed solution

**Returns:**
- ValidationResult: Validation result

**ValidationResult Fields:**
- `is_valid` (bool): Whether solution is valid
- `validation_score` (float): Overall validation score [0, 1]
- `confidence` (float): Confidence in validation [0, 1]
- `aci_reduction` (float): ACI reduction achieved
- `statistical_significance` (float): P-value of improvement
- `errors` (List[str]): Validation errors

**Example:**
```python
from rese.phase4.aci_reduction_validator import Delta3Validator

validator = Delta3Validator()
result = validator.validate(problem, solution)

print(f"Valid: {result.is_valid}")
print(f"Score: {result.validation_score:.2f}")
print(f"ACI reduction: {result.aci_reduction:.2f}")
print(f"P-value: {result.statistical_significance:.4f}")
```

---

## Configuration API

### RESEConfig

**Module:** `rese.config`

**Master configuration for RESE system.**

---

#### `__init__(environment: str = "development", ...)`

Initialize configuration.

**Parameters:**
- `environment` (str): Environment name
- `phase1` (Phase1Config): Phase I configuration
- `phase2` (Phase2Config): Phase II configuration
- `phase3` (Phase3Config): Phase III configuration
- `phase4` (Phase4Config): Phase IV configuration
- `pipeline` (PipelineConfig): Pipeline configuration
- `api` (APIConfig): API configuration
- `monitoring` (MonitoringConfig): Monitoring configuration

---

#### `save(config_path: Optional[Path] = None) -> None`

Save configuration to file.

**Parameters:**
- `config_path` (Optional[Path]): Path to save (default: data_path/config.json)

---

#### `to_dict() -> Dict[str, Any]`

Convert configuration to dictionary.

**Returns:**
- Dict: Configuration dictionary

---

#### `for_environment(environment: Environment) -> RESEConfig`

Create configuration for specific environment.

**Parameters:**
- `environment` (Environment): Environment enum

**Returns:**
- RESEConfig: Environment-specific configuration

**Example:**
```python
from rese.config import RESEConfig, Environment

dev_config = RESEConfig().for_environment(Environment.DEVELOPMENT)
prod_config = RESEConfig().for_environment(Environment.PRODUCTION)
```

---

### Convenience Functions

#### `get_config() -> RESEConfig`

Get current configuration (singleton).

**Returns:**
- RESEConfig: Current configuration

**Example:**
```python
from rese.config import get_config

config = get_config()
print(config.environment)
```

---

#### `load_config(config_path: Optional[Path] = None) -> RESEConfig`

Load configuration from file.

**Parameters:**
- `config_path` (Optional[Path]): Path to config file

**Returns:**
- RESEConfig: Loaded configuration

**Example:**
```python
from rese.config import load_config
from pathlib import Path

config = load_config(Path("my_config.json"))
```

---

## Data Structures

### ProblemInput

**Input problem for RESE pipeline.**

**Fields:**
- `id` (str): Problem identifier
- `description` (str): Problem description
- `constraints` (List[Dict[str, Any]]): Constraint definitions
- `variables` (Dict[str, Any]): Problem variables
- `objective` (Optional[str]): Objective function
- `domain` (str): Problem domain (default: "general")
- `metadata` (Dict[str, Any]): Additional metadata

---

### PhaseResult

**Result from a single phase.**

**Fields:**
- `phase_name` (str): Phase identifier
- `status` (PhaseStatus): Execution status
- `output` (Any): Phase output data
- `metrics` (Dict[str, Any]): Phase metrics
- `errors` (List[str]): Error messages
- `warnings` (List[str]): Warning messages
- `start_time` (datetime): Start time
- `end_time` (Optional[datetime]): End time
- `elapsed_seconds` (float): Elapsed time

---

### PipelineResult

**Result from complete pipeline execution.**

**Fields:**
- `pipeline_id` (str): Pipeline identifier
- `problem_id` (str): Problem identifier
- `status` (PipelineStatus): Execution status
- `phase_results` (Dict[str, PhaseResult]): Phase-wise results
- `final_solution` (Optional[Dict[str, Any]]): Final solution
- `aci_history` (List[float]): ACI progression
- `validation_score` (float): Final validation score
- `confidence` (float): Confidence in solution
- `start_time` (datetime): Start time
- `end_time` (Optional[datetime]): End time
- `elapsed_seconds` (float): Total elapsed time
- `metadata` (Dict[str, Any]): Additional metadata
- `errors` (List[str]): Error messages

---

### Enums

#### PipelineStatus

Pipeline execution status.

**Values:**
- `IDLE`: Not started
- `RUNNING`: Currently executing
- `PAUSED`: Paused by user
- `COMPLETED`: Successfully completed
- `FAILED`: Failed with errors
- `CANCELLED`: Cancelled by user

---

#### PhaseStatus

Phase execution status.

**Values:**
- `PENDING`: Not yet started
- `RUNNING`: Currently executing
- `COMPLETED`: Successfully completed
- `FAILED`: Failed with errors
- `SKIPPED`: Skipped (not run)

---

#### ConstraintType

Constraint type.

**Values:**
- `HARD`: Must satisfy (blocking)
- `SOFT`: Prefer to satisfy (optimization)
- `PREFERENCE`: Nice to have (guidance)

---

#### BiasType

Cognitive bias type.

**Values:**
- `CONFIRMATION`: Confirmation bias
- `AVAILABILITY`: Availability bias
- `ANCHORING`: Anchoring bias
- `SUNK_COST`: Sunk cost fallacy
- `FRAMING`: Framing effect
- `OVERCONFIDENCE`: Overconfidence effect
- `DUNNING_KRUGER`: Dunning-Kruger effect
- `AUTHORITY`: Authority bias
- `CLUSTERING`: Clustering illusion
- `TEXAS_SHARPSHOOTER`: Texas sharpshooter fallacy
- `CAUSAL_OVERSIMPLIFICATION`: Causal oversimplification
- `ILLUSION_OF_CONTROL`: Illusion of control

---

#### Severity

Bias severity level.

**Values:**
- `LOW` (1): Minor bias
- `MEDIUM` (2): Moderate bias
- `HIGH` (3): Severe bias
- `CRITICAL` (4): Extreme bias

---

#### Environment

Deployment environment.

**Values:**
- `DEVELOPMENT`: Development environment
- `TESTING`: Testing environment
- `STAGING`: Staging environment
- `PRODUCTION`: Production environment

---

## Exceptions

### PipelineError

Base pipeline exception.

---

### PhaseExecutionError

Exception during phase execution.

**Raised when:** Phase executor fails

---

### ValidationError

Exception during validation.

**Raised when:** Input validation fails

---

### CachingError

Exception during caching operations.

**Raised when:** Cache read/write fails

---

## Type Hints

RESE uses Python type hints extensively. Key types:

```python
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass

# Generic types
Dict[str, Any]          # Dictionary with string keys
List[Constraint]        # List of constraints
Optional[str]           # String or None
Tuple[str, str]         # Tuple of two strings
Callable[[int], bool]   # Function taking int, returning bool

# Result types
PhaseResult             # Phase execution result
PipelineResult          # Pipeline execution result
ACIResult               # ACI calculation result
ValidationResult        # Validation result
```

---

## End of API Reference

For more details, see:
- [User Guide](user_guide.md)
- [Developer Guide](developer_guide.md)
- [Integration Guide](e2e_integration.md)
