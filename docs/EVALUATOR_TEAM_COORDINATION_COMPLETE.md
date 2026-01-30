# Evaluator Team Coordination System - Complete Implementation Guide

## Overview

The Evaluator Team Coordination System provides enhanced orchestration for coordinating multiple Evaluator Team members working in parallel to validate sub-problem solutions. This system integrates seamlessly with the Decomposition Engine to serve as the final quality gate before solution assembly.

## Architecture

```
DecompositionEngine → EvaluatorTeamCoordinator → Multiple EvaluatorMember (parallel)
                         ↓
                    Evaluation Task Queue Management
                         ↓
                    Consensus Building
                         ↓
                    Quality-Gated Solutions
```

## Key Components

### 1. EvaluatorTeamCoordinator

The main orchestration class that coordinates multiple evaluators.

**Features:**
- Parallel evaluation execution across multiple evaluators
- Intelligent task distribution based on expertise
- Consensus building algorithms (6 methods)
- Bias detection and mitigation
- Integration with DecompositionEngine
- State persistence and recovery
- Quality gate enforcement
- Performance tracking and analytics

### 2. Data Models

#### EvaluationTask
Represents a single evaluation task to be coordinated.

```python
@dataclass
class EvaluationTask:
    task_id: str
    sub_problem_id: str
    sub_problem_description: str
    solution_content: str
    original_content: Optional[str] = None
    criteria: Optional[List[EvaluationCriterion]] = None
    content_type: str = "general"
    priority: EvaluationTaskPriority = EvaluationTaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    assigned_evaluators: List[str] = field(default_factory=list)
    status: EvaluationTaskStatus = EvaluationTaskStatus.PENDING
    consensus_score: float = 0.0
    consensus_reached: bool = False
    quality_gate_passed: bool = False
    threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL
```

#### EvaluationSession
Represents a complete evaluation session for multiple tasks.

```python
@dataclass
class EvaluationSession:
    session_id: str
    problem_statement: str
    sub_problems: List[Dict[str, Any]]
    solutions: Dict[str, str]
    tasks: List[EvaluationTask] = field(default_factory=list)
    status: EvaluationTaskStatus = EvaluationTaskStatus.PENDING
    total_tasks: int = 0
    completed_tasks: int = 0
    quality_gate_passed_tasks: int = 0
    consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE
```

#### EvaluatorMetrics
Tracks performance metrics for individual evaluators.

```python
@dataclass
class EvaluatorMetrics:
    evaluator_id: str
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    total_time_spent: float = 0.0
    average_evaluation_time: float = 0.0
    current_load: int = 0
    reliability_score: float = 1.0
    consensus_agreement_rate: float = 0.0
    bias_profile: Dict[str, float] = field(default_factory=dict)
```

### 3. Consensus Building Methods

The system supports 6 different consensus building algorithms:

#### 1. Weighted Average
Weights evaluator contributions based on reliability and expertise.

```python
consensus_method = ConsensusMethod.WEIGHTED_AVERAGE
```

#### 2. Majority Vote
Uses majority voting on verdicts (APPROVED/NEEDS_WORK/REJECTED).

```python
consensus_method = ConsensusMethod.MAJORITY_VOTE
```

#### 3. Median
Uses median score (robust to outliers).

```python
consensus_method = ConsensusMethod.MEDIAN
```

#### 4. Batesian
Weights by historical reliability and success rate.

```python
consensus_method = ConsensusMethod.BATESIAN
```

#### 5. Dempster-Shafer
Uses evidence theory to combine beliefs.

```python
consensus_method = ConsensusMethod.DEMPSTER_SHAFER
```

#### 6. Delphi
Iterative refinement until convergence.

```python
consensus_method = ConsensusMethod.DELPHI
```

### 4. Load Balancing Strategies

Five strategies for distributing evaluation tasks:

#### Round Robin
```python
load_balancing_strategy = LoadBalancingStrategy.ROUND_ROBIN
```

#### Least Loaded
```python
load_balancing_strategy = LoadBalancingStrategy.LEAST_LOADED
```

#### Specialization Based
Matches evaluator specializations to content type.

```python
load_balancing_strategy = LoadBalancingStrategy.SPECIALIZATION_BASED
```

#### Random
```python
load_balancing_strategy = LoadBalancingStrategy.RANDOM
```

#### Expertise Matched
Matches expertise level to task complexity.

```python
load_balancing_strategy = LoadBalancingStrategy.EXPERTISE_MATCHED
```

## Usage Examples

### Basic Usage

```python
from evaluator_team_coordinator import (
    EvaluatorTeamCoordinator,
    EvaluationThreshold,
    ConsensusMethod
)

# Initialize coordinator
coordinator = EvaluatorTeamCoordinator(
    max_concurrent_evaluations=5,
    consensus_method=ConsensusMethod.WEIGHTED_AVERAGE,
    bias_detection_enabled=True
)

# Define problem and solutions
problem_statement = "Design a secure authentication system"
sub_problems = [
    {
        "id": "sp_001",
        "description": "Implement password hashing",
        "priority": 8
    },
    {
        "id": "sp_002",
        "description": "Design session management",
        "priority": 7
    }
]

solutions = {
    "sp_001": """
def hash_password(password: str) -> str:
    import bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()
""",
    "sp_002": """
class SessionManager:
    def __init__(self):
        self.sessions = {}
"""
}

# Coordinate evaluations
session = coordinator.coordinate_solution_evaluations(
    problem_statement=problem_statement,
    sub_problems=sub_problems,
    solutions=solutions,
    content_types={"sp_001": "code", "sp_002": "code"},
    threshold=EvaluationThreshold.HIGH_QUALITY
)

# Check results
print(f"Session {session.session_id} completed:")
print(f"  Total tasks: {session.total_tasks}")
print(f"  Passed quality gate: {session.quality_gate_passed_tasks}")

for task in session.tasks:
    print(f"\nSub-problem {task.sub_problem_id}:")
    print(f"  Consensus score: {task.consensus_score:.2f}")
    print(f"  Quality gate passed: {task.quality_gate_passed}")
    print(f"  Final verdict: {task.integrated_evaluation.final_verdict}")

# Shutdown
coordinator.shutdown()
```

### Integration with Decomposition Engine

```python
from evaluator_team_coordinator import DecompositionEvaluationBridge

# Initialize bridge
bridge = DecompositionEvaluationBridge(
    auto_validate=True,
    quality_threshold=EvaluationThreshold.STANDARD_APPROVAL
)

# Validate all solutions from decomposition
session = bridge.validate_all_solutions(
    problem_statement=problem_statement,
    sub_problems=sub_problems,
    solutions=solutions,
    content_types={"sp_001": "code", "sp_002": "code"}
)

# Generate validation report
report = bridge.get_validation_report(session)
print(f"Validation rate: {report['validation_rate']:.2%}")
print(f"Passed: {report['passed_validations']}/{report['total_sub_problems']}")
```

### Single Solution Validation

```python
# Validate one solution at a time
result = bridge.validate_solution(
    sub_problem_id="sp_001",
    sub_problem_description="Implement password hashing",
    solution=password_hashing_code,
    content_type="code"
)

if result['validation_passed']:
    print(f"Solution {result['sub_problem_id']} validated!")
    print(f"Consensus score: {result['consensus_score']:.2f}")
else:
    print(f"Solution {result['sub_problem_id']} needs revision")
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
```

### Custom Evaluation Criteria

```python
from evaluator_team import EvaluationCriterion, EvaluationMetric

# Define custom criteria
custom_criteria = [
    EvaluationCriterion(
        metric=EvaluationMetric.SECURITY,
        weight=0.25,
        importance="critical",
        threshold=85.0
    ),
    EvaluationCriterion(
        metric=EvaluationMetric.ROBUSTNESS,
        weight=0.20,
        importance="important",
        threshold=75.0
    )
]

# Use custom criteria in evaluation
session = coordinator.coordinate_solution_evaluations(
    problem_statement=problem_statement,
    sub_problems=sub_problems,
    solutions=solutions,
    criteria={"sp_001": custom_criteria}
)
```

## Configuration Options

### EvaluatorTeamCoordinator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `evaluator_team` | EvaluatorTeam | None | Pre-configured evaluator team |
| `max_concurrent_evaluations` | int | 5 | Max parallel evaluations |
| `load_balancing_strategy` | LoadBalancingStrategy | SPECIALIZATION_BASED | Task distribution strategy |
| `consensus_method` | ConsensusMethod | WEIGHTED_AVERAGE | Consensus algorithm |
| `task_timeout` | int | 300 | Task timeout in seconds |
| `enable_persistence` | bool | True | Enable state persistence |
| `persistence_path` | str | "./evaluator_coordinator_state.pkl" | State file path |
| `bias_detection_enabled` | bool | True | Enable bias detection |
| `quality_gate_threshold` | EvaluationThreshold | STANDARD_APPROVAL | Quality gate level |
| `min_evaluators_per_task` | int | 3 | Minimum evaluators per task |
| `max_evaluators_per_task` | int | 5 | Maximum evaluators per task |

### DecompositionEvaluationBridge Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coordinator` | EvaluatorTeamCoordinator | None | Custom coordinator |
| `auto_validate` | bool | True | Auto-validate solutions |
| `quality_threshold` | EvaluationThreshold | STANDARD_APPROVAL | Quality threshold |

## Quality Gates

The system enforces quality gates based on three criteria:

### 1. Consensus Reached
Evaluators must reach consensus (low variance in scores).

### 2. Score Threshold
Consensus score must meet the required threshold.

**Threshold Levels:**
- `MINIMAL_ACCEPTANCE`: 60.0
- `STANDARD_APPROVAL`: 75.0
- `HIGH_QUALITY`: 85.0
- `EXCEPTIONAL`: 95.0

### 3. Variance Check
Variance must be acceptable (not too much disagreement).

## Bias Detection and Mitigation

The system automatically detects and mitigates evaluator bias:

### Detection Methods

1. **Outlier Detection**: Identifies evaluators consistently deviating from consensus
2. **Philosophy Analysis**: Tracks strictness vs leniency bias
3. **Specialization Bias**: Detects harshness on specialized topics

### Mitigation Strategies

1. **Reliability Adjustment**: Reduces weight of biased evaluators
2. **Bias History Tracking**: Maintains historical bias profile
3. **Dynamic Weighting**: Adjusts consensus weights based on bias

## Performance Metrics

### Evaluator-Level Metrics

```python
performance = coordinator.get_evaluator_performance()

for evaluator_id, metrics in performance.items():
    print(f"{evaluator_id}:")
    print(f"  Evaluations: {metrics['evaluations_completed']}")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Avg time: {metrics['average_time']:.2f}s")
    print(f"  Reliability: {metrics['reliability_score']:.2f}")
```

### Coordinator-Level Metrics

```python
status = coordinator.get_coordinator_status()

print(f"Evaluators: {status['evaluators']}")
print(f"Active tasks: {status['active_tasks']}")
print(f"Completed: {status['completed_tasks']}")
print(f"Quality gate pass rate: {status['quality_gate_pass_rate']:.2%}")
```

## State Persistence

The coordinator automatically saves state to disk for recovery:

```python
# Enable persistence (default)
coordinator = EvaluatorTeamCoordinator(
    enable_persistence=True,
    persistence_path="./my_coordinator_state.pkl"
)

# State is automatically saved:
# - After each session
# - On shutdown
# - Periodically during long sessions

# Manual save
coordinator._save_state()

# Manual load (automatic on init)
coordinator._load_state()
```

## Integration Workflow

### Complete Decomposition Workflow

```python
from decomposition_mcp_tools import (
    decompose_problem_into_sub_problems,
    solve_sub_problem_with_team
)

# 1. Decompose problem
decomposition = decompose_problem_into_sub_problems(
    problem_statement=problem,
    max_sub_problems=10
)

# 2. Solve sub-problems (with Blue Team)
solutions = {}
for sp in decomposition['sub_problems']:
    result = solve_sub_problem_with_team(
        sub_problem_id=sp['id'],
        sub_problem_description=sp['description'],
        team_name="BlueTeam1"
    )
    solutions[sp['id']] = result['solution']

# 3. Validate solutions (with Evaluator Team)
bridge = DecompositionEvaluationBridge()
session = bridge.validate_all_solutions(
    problem_statement=problem,
    sub_problems=decomposition['sub_problems'],
    solutions=solutions
)

# 4. Generate report
report = bridge.get_validation_report(session)

# 5. Handle failed validations
if report['validation_rate'] < 1.0:
    print(f"Some solutions failed validation:")
    for sp_id in report['failed_sub_problems']:
        print(f"  - {sp_id}: needs revision")
        # Re-solve or apply fixes
```

## Testing

The system includes a comprehensive test suite with 30+ tests:

```bash
# Run all tests
python test_evaluator_team_coordinator.py

# Run with pytest
pytest test_evaluator_team_coordinator.py -v

# Run specific test class
pytest test_evaluator_team_coordinator.py::TestConsensusBuilding -v

# Run with coverage
pytest test_evaluator_team_coordinator.py --cov=evaluator_team_coordinator --cov-report=html
```

### Test Coverage

- **Initialization Tests**: Configuration and setup
- **Task Management Tests**: Creation, distribution, execution
- **Consensus Building Tests**: All 6 consensus methods
- **Quality Gate Tests**: Gate enforcement logic
- **Bias Detection Tests**: Outlier detection and mitigation
- **State Persistence Tests**: Save/load functionality
- **Integration Tests**: Decomposition engine integration
- **Performance Tests**: Metrics and tracking
- **Workflow Tests**: End-to-end workflows
- **Recommendation Tests**: Recommendation generation

## Advanced Features

### Custom Load Balancing

```python
# Custom load balancing strategy
def custom_assigner(task, evaluators, num_evaluators):
    # Your custom logic here
    return selected_evaluators

# Override method
coordinator._assign_evaluators = custom_assigner
```

### Custom Consensus Method

```python
# Define custom consensus algorithm
def custom_consensus(assessments, content, content_type, threshold):
    # Your custom consensus logic here
    return integrated_evaluation

# Use in session
session.consensus_method = custom_consensus
```

### Progress Callbacks

```python
def progress_handler(event_type, data):
    if event_type == "session_started":
        print(f"Session {data.session_id} started")
    elif event_type == "task_completed":
        print(f"Task {data.task_id} completed")
    elif event_type == "session_completed":
        print(f"Session completed: {data.completed_tasks}/{data.total_tasks}")

session = coordinator.coordinate_solution_evaluations(
    problem_statement=problem,
    sub_problems=sub_problems,
    solutions=solutions,
    progress_callback=progress_handler
)
```

## Best Practices

### 1. Choose the Right Consensus Method

- **Weighted Average**: Best for general use, balances expertise
- **Majority Vote**: Good for binary decisions (pass/fail)
- **Median**: Robust to outliers, use with diverse evaluators
- **Batesian**: When historical reliability is important
- **Dempster-Shafer**: For uncertain situations with conflicting evidence
- **Delphi**: When you want iterative refinement

### 2. Configure Evaluators per Task

```python
# For simple tasks
coordinator = EvaluatorTeamCoordinator(
    min_evaluators_per_task=2,
    max_evaluators_per_task=3
)

# For critical tasks
coordinator = EvaluatorTeamCoordinator(
    min_evaluators_per_task=5,
    max_evaluators_per_task=7
)
```

### 3. Set Appropriate Quality Gates

```python
# For prototyping
threshold=EvaluationThreshold.MINIMAL_ACCEPTANCE

# For production code
threshold=EvaluationThreshold.HIGH_QUALITY

# For security-critical code
threshold=EvaluationThreshold.EXCEPTIONAL
```

### 4. Enable Persistence for Long Sessions

```python
coordinator = EvaluatorTeamCoordinator(
    enable_persistence=True,
    persistence_path="./production_evaluator_state.pkl"
)
```

### 5. Monitor Performance

```python
# Check evaluator performance regularly
performance = coordinator.get_evaluator_performance()

# Identify underperforming evaluators
for evaluator_id, metrics in performance.items():
    if metrics['success_rate'] < 0.7:
        print(f"Evaluator {evaluator_id} needs attention")
```

## Troubleshooting

### Low Consensus Rate

**Symptom**: Evaluators rarely reach consensus

**Solutions**:
1. Check evaluator diversity (avoid similar philosophies)
2. Adjust consensus method (try MEDIAN for robustness)
3. Increase number of evaluators per task
4. Review evaluation criteria clarity

### High Failure Rate

**Symptom**: Many solutions failing quality gate

**Solutions**:
1. Lower quality gate threshold
2. Review and adjust evaluation criteria
3. Check if evaluators are too strict
4. Provide better guidance to Blue Team

### Bias Detection Triggering

**Symptom**: Evaluators flagged as biased

**Solutions**:
1. Review bias history for patterns
2. Adjust evaluator philosophy
3. Provide calibration training
4. Consider removing consistently biased evaluators

## Performance Optimization

### Increase Throughput

```python
coordinator = EvaluatorTeamCoordinator(
    max_concurrent_evaluations=10,  # Increase concurrency
    task_timeout=180  # Reduce timeout for faster failures
)
```

### Reduce Memory Usage

```python
# Limit session history
coordinator.session_history = coordinator.session_history[-10:]

# Disable persistence if not needed
coordinator = EvaluatorTeamCoordinator(
    enable_persistence=False
)
```

### Optimize Consensus Building

```python
# Use faster consensus method
coordinator = EvaluatorTeamCoordinator(
    consensus_method=ConsensusMethod.MEDIAN  # Faster than weighted
)
```

## API Reference

### EvaluatorTeamCoordinator

#### Methods

- `coordinate_solution_evaluations()`: Main entry point for evaluation
- `get_coordinator_status()`: Get current status
- `get_evaluator_performance()`: Get performance metrics
- `shutdown()`: Cleanup and save state

#### Internal Methods

- `_assign_evaluators()`: Assign evaluators to task
- `_build_consensus()`: Build consensus from assessments
- `_apply_quality_gate()`: Apply quality gate
- `_detect_and_mitigate_bias()`: Detect and correct bias

### DecompositionEvaluationBridge

#### Methods

- `validate_solution()`: Validate single solution
- `validate_all_solutions()`: Validate all solutions
- `get_validation_report()`: Generate validation report

## Summary

The Evaluator Team Coordination System provides:

1. **Parallel Evaluation**: Multiple evaluators work simultaneously
2. **Intelligent Consensus**: 6 consensus building algorithms
3. **Quality Enforcement**: Configurable quality gates
4. **Bias Mitigation**: Automatic bias detection and correction
5. **Decomposition Integration**: Seamless workflow integration
6. **State Persistence**: Recoverable operation
7. **Performance Tracking**: Comprehensive metrics
8. **Flexible Configuration**: Highly customizable

This system serves as the final quality gate in the decomposition workflow, ensuring that only validated solutions proceed to assembly.
