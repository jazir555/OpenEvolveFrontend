# Architecture Decision Record: Gauntlet Adapter

**Status:** Accepted
**Date:** 2026-02-12
**Context:** OpenEvolve Federation - Gauntlet Integration

---

## Context

The **Gauntlet** is an intelligent execution orchestration system that validates solutions through multi-round testing (Longflow, Red Team, Gold Team). The Gauntlet Adapter provides AI-powered orchestration with multi-objective optimization (accuracy, speed, cost).

### Key Challenges

1. **Multi-Round Execution** - Three distinct validation rounds with different strategies
2. **Resource Allocation** - Optimal distribution of compute resources across rounds
3. **Strategy Selection** - Choosing optimal execution strategy (sequential, parallel, adaptive, hierarchical)
4. **Stopping Conditions** - Early termination when solution quality is clear
5. **Fallback Plans** - Graceful degradation when primary executor fails
6. **Adaptation** - Dynamic strategy adjustment during execution

### Gauntlet Architecture

The Gauntlet uses a **Three-Round Validation Process**:

1. **Round 1 (Longflow)** - Extensive automated testing (50+ evaluations)
2. **Round 2 (Red Team)** - Adversarial attack simulation (10+ attacks)
3. **Round 3 (Gold Team)** - Expert human evaluation (3+ evaluators)

**Final Score**: Weighted average of all rounds (threshold: 0.6 to pass)

---

## Decision

### Architecture Pattern: Intelligent Orchestration Sidecar

We chose an **Intelligent Orchestration Sidecar Pattern** with the following characteristics:

```
┌─────────────────────────────────────────────────────────────┐
│                  Gauntlet Adapter                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Intelligent Orchestrator                  │  │
│  │  • Multi-objective optimization                  │  │
│  │  • Strategy selection (sequential/parallel/adaptive)  │  │
│  │  • Resource allocation                               │  │
│  │  • Stopping conditions                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────┬──────────────────┬──────────────┐│
│  │   ML Optimizer  │  Adaptive Learner │  Predictive   ││
│  │                 │                  │  Executor    ││
│  └──────────────────┴──────────────────┴──────────────┘│
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Gauntlet Execution Engine                     │
│  • Round execution (Longflow, Red Team, Gold Team)    │
│  • Score aggregation                                      │
│  • Pass/fail determination                               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Adapter Location**: `/glue/adapters/gauntlet-adapter/`
   - Isolated from core-projects (Law of Air Gap)
   - Rewritten gauntlet utilities in adapter layer
   - Canonical schema at `/glue/schemas/gauntlet-canonical.json`

2. **Multi-Objective Optimization**: Balance competing objectives
   - **MAXIMIZE_ACCURACY** - Prioritize solution quality
   - **MINIMIZE_TIME** - Fastest execution
   - **MINIMIZE_COST** - Lowest computational cost
   - **MAXIMIZE_THROUGHPUT** - Maximum parallelism
   - **BALANCED** - Equal weighting (default)

3. **Orchestration Strategies**: Adaptive strategy selection
   - **SEQUENTIAL** - Execute rounds one after another (reliable)
   - **PARALLEL** - Execute rounds in parallel where possible (fast)
   - **ADAPTIVE** - Adjust strategy based on intermediate results (smart)
   - **HIERARCHICAL** - Multi-level decision tree (efficient)

4. **Resource Allocation**: Dynamic allocation based on complexity
   - High complexity → More evaluations, longer timeout
   - Low complexity → Fewer evaluations, faster execution

5. **Fallback Plans**: Graceful degradation
   - Primary executor failed → Use mock executor
   - Round timeout → Reduce complexity
   - Circuit breaker open → Skip gauntlet (risky)

---

## Alternatives Considered

### Alternative 1: Simple Sequential Execution
**Rejected**: No optimization for speed/cost, poor resource utilization

### Alternative 2: Always Parallel Execution
**Rejected**: Red Team rounds must be sequential, can't parallelize all rounds

### Alternative 3: Fixed Resource Allocation
**Rejected**: Wastes resources on simple problems, fails on complex problems

### Alternative 4: No Adaptive Strategy
**Rejected**: Can't adjust to intermediate results, misses early termination opportunities

---

## Consequences

### Positive Benefits

1. **Multi-Objective Optimization** - Balance accuracy, speed, cost
2. **Adaptive Strategy** - Adjusts to problem characteristics
3. **Resource Efficiency** - Optimal allocation based on complexity
4. **Early Termination** - Stop execution when result is clear
5. **Graceful Degradation** - Fallback plans prevent total failure
6. **Intelligent Scheduling** - Learn optimal execution patterns

### Negative Tradeoffs

1. **Planning Overhead** - Strategy analysis adds 100-500ms latency
2. **Complexity** - Multiple strategies increase code complexity
3. **Prediction Accuracy** - ML optimizer may mispredict optimal strategy
4. **Fallback Risk** - Mock executor may accept invalid solutions
5. **Adaptation Cost** - Adaptive strategy requires more execution time

### Known Limitations

1. **Strategy Selection Heuristic** - Based on code analysis, not actual execution
2. **Time Estimation** - Estimates can be off by 50-100%
3. **Resource Prediction** - Doesn't account for system load
4. **Fallback Quality** - Mock executor provides minimal validation
5. **Adaptation Lag** - Adaptive strategy requires 1-2 rounds to adapt

---

## Implementation Details

### Core Components

#### 1. IntelligentGauntletOrchestrator
```python
class IntelligentGauntletOrchestrator:
    def create_orchestration_plan(
        solution: str,
        problem: str,
        domain: str,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationPlan

    async def execute_orchestration(
        solution: str,
        problem: str,
        domain: str,
        plan: Optional[OrchestrationPlan] = None
    ) -> OrchestrationResult
```

**Capabilities**:
- Analyze problem/solution characteristics
- Select optimal orchestration strategy
- Allocate resources per round
- Create stopping conditions
- Generate fallback plans

**Example**:
```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED
)

# Create optimal plan
plan = orchestrator.create_orchestration_plan(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance"
)

# Execute with orchestration
result = await orchestrator.execute_orchestration(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance",
    plan=plan
)

# Result includes:
# - passed: True/False
# - final_score: 0.75
# - rounds_completed: 3
# - execution_time: 45.2
# - adaptations_made: ["Adjusted thresholds after low round1 score"]
# - recommendations: ["Excellent solution quality"]
```

#### 2. OrchestrationPlan
```python
@dataclass
class OrchestrationPlan:
    strategy: OrchestrationStrategy  # SEQUENTIAL, PARALLEL, ADAPTIVE, HIERARCHICAL
    execution_order: List[str]         # Order of round execution
    resource_allocation: Dict[str, Dict[str, Any]]  # Resources per round
    stopping_conditions: List[Dict[str, Any]]  # Early termination
    fallback_plans: List[Dict[str, Any]]  # Backup plans
    estimated_time: float              # Estimated total time (seconds)
    estimated_cost: float              # Estimated computational cost
```

#### 3. OrchestrationResult
```python
@dataclass
class OrchestrationResult:
    passed: bool                     # Passed gauntlet
    final_score: float               # Weighted average score
    rounds_completed: int            # Number of rounds completed
    execution_time: float            # Actual execution time (seconds)
    actual_cost: float               # Actual computational cost
    resource_utilization: Dict[str, float]  # Resource usage
    adaptations_made: List[str]     # Adaptations during execution
    recommendations: List[str]        # Improvement suggestions
```

### API Endpoints

| Endpoint | Purpose | Timeout | Async |
|----------|---------|---------|--------|
| `create_orchestration_plan` | Analyze and create plan | 5s | No |
| `execute_orchestration` | Execute gauntlet with plan | 300s | Yes |
| `get_orchestration_stats` | Get execution statistics | 1s | No |
| `set_ml_optimizer` | Set ML optimizer for learning | 1s | No |
| `set_predictive_executor` | Set predictive executor | 1s | No |

### Orchestration Strategies

#### SEQUENTIAL (Default)
**Use case**: General purpose, reliable execution

```
Round 1 (Longflow)
    ↓
Round 2 (Red Team)
    ↓
Round 3 (Gold Team)
    ↓
Aggregate Scores
```

**Pros**: Reliable, well-tested
**Cons**: Slower than parallel

#### PARALLEL
**Use case**: Simple problems, speed-critical

```
Round 1 (Longflow) ──┐
                      ├─→ Aggregate
Round 2 (Red Team) ──┘    (only if independent)
Round 3 (Gold Team)
```

**Pros**: Fastest execution
**Cons**: Limited parallelism (Red Team sequential)

#### ADAPTIVE
**Use case**: High complexity, uncertain solution quality

```
Round 1 (Longflow)
    ↓ (analyze result)
If score > 0.8:
    → Skip to Round 3 (Gold Team)
Else:
    → Continue to Round 2 (Red Team)
```

**Pros**: Adapts to solution quality
**Cons**: Requires 1-2 rounds to adapt

#### HIERARCHICAL
**Use case**: Time-critical, high-confidence solutions

```
Round 1 (Longflow)
    ↓
If score > 0.8:
    → Round 3 (Gold Team) [Skip Round 2]
Else:
    → Sequential execution
```

**Pros**: Fast for good solutions
**Cons**: Risky (may skip important validation)

### Complexity Analysis

```python
def _calculate_complexity(solution: str) -> float:
    """Calculate solution complexity (0.0-1.0)"""
    complexity = 0.5  # baseline
    lines = len(solution.splitlines())

    # Length penalty
    if lines > 100:
        complexity += 0.2
    elif lines > 50:
        complexity += 0.1
    elif lines < 10:
        complexity -= 0.2

    # Function/class count
    func_count = solution.count("def ")
    complexity += min(0.15, func_count * 0.03)

    class_count = solution.count("class ")
    complexity += min(0.1, class_count * 0.05)

    return max(0.0, min(1.0, complexity))
```

**Complexity Ranges**:
- 0.0-0.3: Simple (few functions, short)
- 0.4-0.6: Medium (moderate complexity)
- 0.7-1.0: Complex (many functions, long)

### Configuration Requirements

#### Environment Variables
```bash
# Orchestrator Configuration
GAUNTLET_OBJECTIVE=balanced           # optimization objective
GAUNTLET_MAX_PARALLELISM=4           # max parallel executions
GAUNTLET_ENABLE_PREDICTION=true      # enable predictive executor
GAUNTLET_ENABLE_OPTIMIZATION=true    # enable ML optimizer

# Resource Limits
GAUNTLET_MAX_ROUND1_EVALUATIONS=50   # Round 1 max evaluations
GAUNTLET_MAX_ROUND2_ATTACKS=10       # Round 2 max attacks
GAUNTLET_MAX_ROUND3_EVALUATORS=3     # Round 3 max evaluators

# Timeouts
GAUNTLET_ROUND1_TIMEOUT=30           # Round 1 timeout (seconds)
GAUNTLET_ROUND2_TIMEOUT=60           # Round 2 timeout (seconds)
GAUNTLET_ROUND3_TIMEOUT=90           # Round 3 timeout (seconds)

# Adapter Configuration
GAUNTLET_ADAPTER_HOST=gauntlet-adapter  # Service name
GAUNTLET_ADAPTER_PORT=8002                # HTTP port
GAUNTLET_LOG_LEVEL=INFO                   # Logging level
```

#### Python Configuration
```python
config = {
    "objective": OptimizationObjective.BALANCED,
    "max_parallelism": 4,
    "enable_prediction": True,
    "enable_optimization": True,
    "timeouts": {
        "round1": 30,
        "round2": 60,
        "round3": 90
    },
    "resource_limits": {
        "round1_max_evaluations": 50,
        "round2_max_attacks": 10,
        "round3_max_evaluators": 3
    }
}
```

---

## Gotchas

### API Quirks Discovered

1. **Strategy Selection Latency**:
   - Complexity analysis adds 100-500ms before execution
   - **Gotcha**: First request is slower than subsequent requests
   - **Solution**: Cache complexity analysis for identical solutions

2. **Adaptive Strategy Warmup**:
   - Adaptive strategy requires 1-2 rounds to adapt
   - **Gotcha**: First execution uses sequential strategy
   - **Solution**: Pre-warm with similar historical problems

3. **Time Estimation Inaccuracy**:
   - Estimates can be off by 50-100%
   - **Gotcha**: Users misled about execution time
   - **Solution**: Show confidence intervals, update estimate during execution

4. **Fallback Plan Execution**:
   - Mock executor may accept invalid solutions
   - **Gotcha**: False positive pass
   - **Solution**: Log fallback usage, flag for manual review

5. **Parallel Execution Limitations**:
   - Red Team rounds must be sequential (stateful attacks)
   - **Gotcha**: Limited parallelism despite PARALLEL strategy
   - **Solution**: Only parallelize independent rounds (Round 1, Round 3)

### Version Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|---------------------|-------|
| Python | 3.10 | 3.11+ | 3.11 improves asyncio |
| NumPy | 1.20 | 1.24+ | Faster numerical operations |

### Non-Obvious Behaviors

1. **Strategy State Machine**:
   - Strategy changes during execution (ADAPTIVE)
   - **Gotcha**: Strategy in plan != strategy actually used
   - **Solution**: Track actual strategy in result

2. **Resource Contention**:
   - Multiple parallel executions compete for resources
   - **Gotcha**: Actual execution time >> estimated time
   - **Solution**: Dynamic resource allocation, queue system

3. **Stopping Condition Evaluation**:
   - Stopping conditions checked after each round
   - **Gotcha**: Mid-round stopping not supported
   - **Solution**: Shorter rounds for faster feedback

4. **Fallback Plan Cascades**:
   - Primary fallback may also fail
   - **Gotcha**: No fallback for fallback
   - **Solution**: Multiple fallback plans with priority

---

## Testing Strategy

### 1. Probes (Before Implementation)

```bash
# Verify gauntlet executor API
python probes/check_gauntlet_api.sh

# Verify orchestration strategies
python probes/check_strategies.sh

# Verify resource allocation
python probes/check_resources.sh
```

### 2. Contract Tests (On Every Deploy)

```bash
npm run test:contract
```

Tests validate:
- Orchestration plan creation
- All four strategies (sequential, parallel, adaptive, hierarchical)
- Resource allocation
- Stopping conditions
- Fallback plans
- Error handling

### 3. Integration Tests

```python
from gauntlet_adapter import IntelligentGauntletOrchestrator, OptimizationObjective

# Test plan creation
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED
)
plan = orchestrator.create_orchestration_plan(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance"
)

assert plan.strategy in [OrchestrationStrategy.SEQUENTIAL, OrchestrationStrategy.ADAPTIVE]
assert len(plan.execution_order) == 3
assert plan.estimated_time > 0

# Test execution
result = await orchestrator.execute_orchestration(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance",
    plan=plan
)

assert result.passed in [True, False]
assert 0.0 <= result.final_score <= 1.0
assert result.execution_time > 0
```

---

## Federation Constitution Compliance Checklist

- ✅ **Law of Air Gap**: No imports from `core-projects/`
- ✅ **Law of Runtime Truth**: Probes verify executor API before use
- ✅ **Law of Untouchable DB**: No database writes (stateless orchestration)
- ✅ **Law of Idempotency**: Orchestrator creates new plans each time (stateless)
- ✅ **Law of Configuration Explicitness**: All required env vars validated
- ✅ **Law of UTC**: All timestamps in UTC ISO-8601

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy adapter code
COPY src/ /app/src/

# Expose port
EXPOSE 8002

# Run adapter
CMD ["python", "-m", "gauntlet_adapter"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gauntlet-adapter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gauntlet-adapter
  template:
    metadata:
      labels:
        app: gauntlet-adapter
    spec:
      containers:
      - name: orchestrator
        image: gauntlet-adapter:latest
        ports:
        - containerPort: 8002
        env:
        - name: GAUNTLET_OBJECTIVE
          value: "balanced"
        - name: GAUNTLET_MAX_PARALLELISM
          value: "4"
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Monitoring

### Key Metrics

1. **Orchestration Strategy Distribution** - SEQUENTIAL vs PARALLEL vs ADAPTIVE vs HIERARCHICAL
2. **Planning Time** - Time to create orchestration plan (target: <500ms)
3. **Execution Time Accuracy** | Estimated vs Actual (target: ±20%)
4. **Fallback Usage** - Frequency of fallback plan execution (target: <5%)
5. **Adaptation Frequency** - How often adaptive strategy changes (target: 10-30%)
6. **Pass Rate** - Percentage of solutions passing gauntlet

### Logging Format

```json
{
  "level": "info",
  "msg": "Orchestration complete",
  "timestamp": "2026-02-12T10:30:00.000Z",
  "correlation_id": "a1b2c3d4-...",
  "source_service": "gauntlet-adapter",
  "target_service": "gauntlet-executor",
  "domain": "finance",
  "strategy": "adaptive",
  "passed": true,
  "final_score": 0.75,
  "execution_time": 45.2,
  "adaptations_made": 1
}
```

---

## Future Improvements

1. **ML-Based Strategy Selection** - Train model on historical executions
2. **Real-time Strategy Switching** - Change strategy mid-execution
3. **Resource Prediction** - ML-based time and cost estimation
4. **Multi-Objective Pareto Front** - Show tradeoffs between objectives
5. **Automatic Complexity Detection** - Learn complexity from code features
6. **Distributed Orchestration** - Coordinate across multiple gauntlet instances

---

## References

- [Federation Constitution](../../../../CLAUDE.md)
- [Gauntlet Core](../../../../core-projects/Gauntlet/README.md)
- [Intelligent Orchestrator](./src/intelligent_orchestrator.py)
- [Adaptive Learner](./src/adaptive_learner.py)
- [ML Optimizer](./src/ml_optimizer.py)
- [Predictive Executor](./src/predictive_gauntlet_executor.py)

---

**Created**: 2026-02-12
**Author**: OpenEvolve Architecture Team
**Status**: Accepted, Implemented
**Last Updated**: 2026-02-12
