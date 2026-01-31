# Multi-Round Gauntlet Orchestration

## Overview

The Multi-Round Gauntlet Orchestrator provides sophisticated state management, decision logic, and artifact fusion for the 3-round gauntlet evaluation system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Round Gauntlet Orchestrator              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Round 1    │ -> │   Round 2    │ -> │   Round 3    │  │
│  │  LoongFlow   │    │   Red Team   │    │  Gold Team   │  │
│  │    AI Eval   │    │   Adversary  │    │   Consensus  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │         │
│         v                    v                    v         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              State Management                       │   │
│  │  - Progress Tracking                                │   │
│  │  - Score Normalization                              │   │
│  │  - Decision Recording                               │   │
│  │  - Artifact Collection                               │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                                                    │
│         v                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Artifact Fusion                          │   │
│  │  - Consensus Detection                                │   │
│  │  - Conflict Identification                            │   │
│  │  - Improvement Prioritization                         │   │
│  │  - Recommendation Generation                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. GauntletState

Tracks complete state across all rounds:

```python
@dataclass
class GauntletState:
    solution: str
    problem: str
    domain: str

    # Progress
    current_round: int  # 0=not started, 1=R1, 2=R2, 3=R3, 4=complete
    rounds_completed: List[int]

    # Results
    round1_result: Optional[Round1Result]
    round2_result: Optional[Round2Result]
    round3_result: Optional[Round3Result]

    # Normalized Scores (0-1 scale)
    round1_normalized_score: Optional[float]
    round2_normalized_score: Optional[float]
    round3_normalized_score: Optional[float]

    # Decisions
    round1_decision: Optional[str]  # "continue" or "terminate"
    round2_decision: Optional[str]
    round3_decision: Optional[str]

    # Artifacts
    collected_artifacts: List[Any]

    # Performance
    total_evaluation_time: float
    round_times: Dict[int, float]

    # Metadata
    started_at: datetime
    completed_at: Optional[datetime]
    status: str  # in_progress, completed, terminated, error
```

### 2. Decision Logic

Each round makes intelligent continue/terminate decisions:

**Round 1 (LoongFlow AI Eval):**
- Score >= threshold (default 0.7)
- Confidence >= minimum (default 0.6)
- Weaknesses < maximum (default 5)

**Round 2 (Red Team Attack):**
- Score >= threshold (default 0.6)
- Successful attacks < maximum (default 3)
- Robustness >= minimum (default 0.5)

**Round 3 (Gold Team Verify):**
- Score >= threshold (default 0.85)
- Consensus >= minimum (default 0.75)
- Formal verification passed (if required)

### 3. Score Normalization

Different rounds use different scales:

- **Round 1**: 0-1 (already normalized)
- **Round 2**: 0-100 → divide by 100
- **Round 3**: 0-10 → divide by 10

All scores normalized to 0-1 for comparison.

### 4. Artifact Fusion

Combines insights from all rounds:

```python
@dataclass
class FusedArtifacts:
    # All artifacts
    all_scores: Dict[str, float]
    all_feedback: List[str]
    all_strengths: List[str]
    all_weaknesses: List[str]

    # Consensus (mentioned by 2+ rounds)
    consensus_strengths: List[str]
    consensus_weaknesses: List[str]

    # Conflicts (strength in one, weakness in another)
    conflicting_feedback: List[Tuple[str, str]]

    # Trends
    robustness_trend: List[float]
    confidence_trend: List[float]
    quality_trend: List[float]

    # Recommendations
    overall_recommendation: str
    improvement_priority: List[str]
```

### 5. Performance Metrics

Tracks execution quality and efficiency:

```python
@dataclass
class PerformanceMetrics:
    # Time
    total_time: float
    round_times: Dict[int, float]

    # Quality
    average_score: float
    score_variance: float
    trend: str  # "improving", "declining", "stable"

    # Efficiency
    evaluations_per_round: Dict[int, int]
    total_evaluations: int
    cost_estimate: float

    # Decisions
    termination_round: Optional[int]
    termination_reason: Optional[str]
    false_positive_risk: float
    false_negative_risk: float
```

## Usage Examples

### Basic Usage

```python
from openevolve.gauntlets import MultiRoundGauntletOrchestrator, MultiRoundConfig

# Create orchestrator
config = MultiRoundConfig(
    round1_threshold=0.7,
    round2_threshold=0.6,
    round3_threshold=0.85
)
orchestrator = MultiRoundGauntletOrchestrator(config)

# Execute full gauntlet
state = await orchestrator.execute_full_gauntlet(
    solution=my_solution,
    problem="Optimize trading strategy",
    domain="finance"
)

# Get results
report = orchestrator.generate_progress_report(state)
print(report)

# Get fused artifacts
fused = orchestrator.fuse_artifacts(state)
print(f"Consensus strengths: {fused.consensus_strengths}")
print(f"Consensus weaknesses: {fused.consensus_weaknesses}")

# Get performance metrics
metrics = orchestrator.get_performance_metrics(state)
print(f"Average score: {metrics.average_score:.2%}")
print(f"Trend: {metrics.trend}")
```

### Custom Configuration

```python
# Strict configuration
strict_config = MultiRoundConfig(
    round1_threshold=0.85,
    round2_threshold=0.75,
    round3_threshold=0.95,
    max_weaknesses=2,
    max_vulnerabilities=1,
    require_formal_verification=True
)

# Lenient configuration
lenient_config = MultiRoundConfig(
    round1_threshold=0.5,
    round2_threshold=0.4,
    round3_threshold=0.7,
    max_weaknesses=10,
    max_vulnerabilities=5
)
```

### Step-by-Step Execution

```python
# Initialize
state = await orchestrator.initialize_gauntlet(
    solution=solution,
    problem=problem,
    domain=domain
)

# Execute Round 1
state = await orchestrator.execute_round(1, state)
print(f"R1 Decision: {state.round1_decision}")

# Check if should continue
if state.round1_decision == "continue":
    # Execute Round 2
    state = await orchestrator.execute_round(2, state)
    print(f"R2 Decision: {state.round2_decision}")

    if state.round2_decision == "continue":
        # Execute Round 3
        state = await orchestrator.execute_round(3, state)
        print(f"R3 Decision: {state.round3_decision}")
```

### Progress Reporting

```python
# Generate progress report
report = orchestrator.generate_progress_report(state)

# Report includes:
# - Current status and rounds completed
# - Score breakdown by round
# - Strengths and weaknesses
# - Decision rationale
# - Final recommendation
# - Performance metrics

print(report)
```

### Performance Analysis

```python
# Get performance metrics
metrics = orchestrator.get_performance_metrics(state)

# Analyze
print(f"Total time: {metrics.total_time:.1f}s")
print(f"Average score: {metrics.average_score:.2%}")
print(f"Score variance: {metrics.score_variance:.3f}")
print(f"Trend: {metrics.trend}")
print(f"Total evaluations: {metrics.total_evaluations}")
print(f"Estimated cost: ${metrics.cost_estimate:.2f}")

# Risk analysis
print(f"False positive risk: {metrics.false_positive_risk:.2%}")
print(f"False negative risk: {metrics.false_negative_risk:.2%}")
```

## Configuration Options

### Thresholds

```python
config = MultiRoundConfig(
    # Round 1 thresholds
    round1_threshold=0.7,        # Min score to continue
    min_confidence=0.6,            # Min confidence
    max_weaknesses=5,              # Max weaknesses allowed

    # Round 2 thresholds
    round2_threshold=0.6,         # Min normalized score
    max_vulnerabilities=3,         # Max critical vulnerabilities
    min_robustness=0.5,            # Min robustness score

    # Round 3 thresholds
    round3_threshold=0.85,        # Min normalized score
    min_consensus=0.75,            # Min consensus score
    require_formal_verification=False  # Lean 4 required?
)
```

### Score Weights

```python
config = MultiRoundConfig(
    round1_weight=0.2,  # LoongFlow weight
    round2_weight=0.3,  # Red Team weight
    round3_weight=0.5   # Gold Team weight
)
```

### Execution Options

```python
config = MultiRoundConfig(
    enable_parallel_execution=True,   # Parallel where possible
    max_parallel_evaluations=5,       # Max parallel tasks
    timeout_per_round=300,            # Timeout (seconds)

    enable_early_termination=True,    # Stop on failure
    fail_fast=True                    # Immediate termination
)
```

### Artifact Fusion

```python
config = MultiRoundConfig(
    consensus_threshold=2,      # Min rounds for consensus
    conflict_detection=True     # Detect conflicting feedback
)
```

## State Management

### State Lifecycle

```
not_started -> in_progress -> [completed | terminated | error]
                 |
                 v
        [Round 1 -> Round 2 -> Round 3]
                 |
                 v
            Artifact Fusion
                 |
                 v
            Performance Metrics
```

### State Persistence

```python
# Serialize state
state_dict = state.to_dict()
import json
with open('gauntlet_state.json', 'w') as f:
    json.dump(state_dict, f)

# Deserialize state (future feature)
# state = GauntletState.from_dict(state_dict)
```

## Decision Points

### Round 1 Decision Factors

1. **Score Quality**: Overall solution score (0-1)
2. **Confidence**: Evaluation confidence (0-1)
3. **Flaw Count**: Number of weaknesses identified

**Continue if:**
- Score >= threshold AND
- Confidence >= minimum AND
- Weaknesses < maximum

### Round 2 Decision Factors

1. **Survivability**: Score after attacks (0-1)
2. **Vulnerability Count**: Successful attacks
3. **Robustness**: Stability under stress

**Continue if:**
- Score >= threshold AND
- Successful attacks < maximum AND
- Robustness >= minimum

### Round 3 Decision Factors

1. **Quality Score**: Consensus evaluation (0-1)
2. **Agreement**: Judge consensus level
3. **Formal Verification**: Lean 4 proof (if required)

**Final Approval if:**
- Score >= threshold AND
- Consensus >= minimum AND
- Formal verification passed (if required)

## Artifact Fusion

### Consensus Detection

Items mentioned by multiple rounds:

```python
# Example:
# Round 1: "Clear code structure"
# Round 2: "Clear architecture"
# Round 3: "Well-organized code"
#
# Consensus: "clear" mentioned by multiple rounds
```

### Conflict Detection

Items identified as strength in one round, weakness in another:

```python
# Example:
# Round 1: "Good performance"
# Round 2: "Poor performance under load"
#
# Conflict: ("Good performance", "Poor performance under load")
```

### Improvement Prioritization

Priority levels:
1. **HIGH**: Security/safety critical
2. **MEDIUM**: Consensus weaknesses
3. **LOW**: Other issues

## Performance Tracking

### Time Metrics

- Total execution time
- Per-round execution time
- Evaluation vs. overhead ratio

### Quality Metrics

- Average score across rounds
- Score variance (consistency)
- Trend analysis (improving/declining/stable)

### Efficiency Metrics

- Total evaluations performed
- Evaluations per round
- Estimated API costs

### Risk Metrics

- **False Positive Risk**: Passing bad solutions
- **False Negative Risk**: Failing good solutions

## Parallel Execution

Round 3 (Gold Team) supports parallel evaluation:

```python
config = MultiRoundConfig(
    enable_parallel_execution=True,
    max_parallel_evaluations=5
)

# Judges evaluate in parallel
results = await asyncio.gather(*[
    evaluate_with_judge(model)
    for model in judge_models
])

# Results aggregated
state = aggregate_results(state, results)
```

## Progress Reporting

### Real-Time Progress

```python
# After each round
report = orchestrator.generate_progress_report(state)

# Shows current status
# - Which rounds completed
# - Scores so far
# - Decisions made
```

### Final Report

```python
# After all rounds (or termination)
final_report = orchestrator.generate_progress_report(state)

# Includes:
# - Complete score breakdown
# - Consensus strengths/weaknesses
# - Improvement priorities
# - Overall recommendation
# - Performance metrics
```

## Best Practices

### 1. Configure Thresholds Appropriately

```python
# For safety-critical systems
strict_config = MultiRoundConfig(
    round1_threshold=0.9,
    round3_threshold=0.95,
    require_formal_verification=True
)

# For experimentation
exploratory_config = MultiRoundConfig(
    round1_threshold=0.5,
    round3_threshold=0.7,
    enable_early_termination=False
)
```

### 2. Use Progress Reports

```python
# Monitor progress
for round_num in [1, 2, 3]:
    state = await orchestrator.execute_round(round_num, state)
    report = orchestrator.generate_progress_report(state)
    logger.info(report)

    # Early exit if terminated
    if state.status == "terminated":
        break
```

### 3. Analyze Performance

```python
# Track metrics over time
metrics = orchestrator.get_performance_metrics(state)

# Log for analysis
logger.info(f"Trend: {metrics.trend}")
logger.info(f"Cost: ${metrics.cost_estimate:.2f}")

# Optimize based on metrics
if metrics.total_evaluations > budget:
    config.max_evaluations = budget * 0.8
```

### 4. Handle Errors Gracefully

```python
try:
    state = await orchestrator.execute_full_gauntlet(...)
except Exception as e:
    logger.error(f"Gauntlet failed: {e}")

    # Check partial results
    if state.rounds_completed:
        report = orchestrator.generate_progress_report(state)
        logger.info(f"Partial results:\n{report}")
```

## Troubleshooting

### Issue: Early Termination

**Symptoms**: Gauntlet stops after Round 1

**Solutions**:
- Lower thresholds
- Check solution quality
- Review Round 1 feedback

### Issue: High False Negative Rate

**Symptoms**: Good solutions failing

**Solutions**:
- Lower thresholds
- Increase consensus_threshold
- Review conflict detection

### Issue: Slow Execution

**Symptoms**: Long evaluation times

**Solutions**:
- Enable parallel execution
- Reduce max_evaluations
- Adjust timeout_per_round

## API Reference

See [`multi_round_orchestrator.py`](../../openevolve/gauntlets/multi_round_orchestrator.py) for complete API documentation.

### Classes

- `MultiRoundGauntletOrchestrator`: Main orchestrator
- `GauntletState`: State management
- `FusedArtifacts`: Artifact fusion
- `PerformanceMetrics`: Performance tracking
- `MultiRoundConfig`: Configuration

### Methods

- `initialize_gauntlet()`: Initialize gauntlet
- `execute_round()`: Execute single round
- `execute_full_gauntlet()`: Execute all rounds
- `make_decision()`: Make continue/terminate decision
- `normalize_scores()`: Normalize scores to 0-1
- `fuse_artifacts()`: Combine artifacts from all rounds
- `calculate_final_score()`: Calculate weighted final score
- `generate_progress_report()`: Generate progress report
- `get_performance_metrics()`: Get performance metrics

## Future Enhancements

- [ ] State persistence and recovery
- [ ] Distributed execution across machines
- [ ] Adaptive thresholds based on historical data
- [ ] Multi-objective optimization
- [ ] Real-time progress streaming
- [ ] Custom decision functions
- [ ] Integration with knowledge engine
- [ ] Automated threshold tuning
