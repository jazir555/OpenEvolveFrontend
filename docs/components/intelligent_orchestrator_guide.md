# Intelligent Gauntlet Orchestrator - User Guide

Complete guide for using the Intelligent Gauntlet Orchestrator component.

## Overview

The Intelligent Gauntlet Orchestrator provides AI-powered orchestration of complex gauntlet workflows with multi-objective optimization, automated composition, and dynamic adaptation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Orchestration Strategies](#orchestration-strategies)
3. [Objectives](#objectives)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Best Practices](#best-practices)

---

## Quick Start

### Basic Orchestration

```python
from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
    IntelligentGauntletOrchestrator,
    OptimizationObjective
)

# Create orchestrator
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED,
    max_parallelism=4
)

# Execute with intelligent orchestration
result = await orchestrator.execute_orchestration(
    solution="def solve(): return optimal_solution",
    problem="Optimize the packing problem",
    domain="math"
)

print(f"Passed: {result.passed}")
print(f"Score: {result.final_score:.3f}")
print(f"Adaptations: {result.adaptations_made}")
```

---

## Orchestration Strategies

### SEQUENTIAL

Execute rounds one after another in order.

**Best for:**
- Resource-constrained environments
- When order matters
- Maximum accuracy needed

```python
result = await orchestrator.execute_orchestration(
    solution=solution,
    problem=problem,
    domain="domain"
)
# Automatically uses sequential if not parallelizable
```

### PARALLEL

Execute rounds concurrently where possible.

**Best for:**
- Independent rounds
- Time-sensitive evaluations
- When you have sufficient resources

```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MINIMIZE_TIME,
    max_parallelism=4
)
```

### ADAPTIVE

Adjust strategy based on intermediate results.

**Best for:**
- Uncertain solution quality
- When you want to optimize resource usage
- Complex, multi-stage problems

```python
result = await orchestrator.execute_orchestration(
    solution=solution,
    problem=problem,
    domain="domain"
)
# Automatically adapts based on results
```

### HIERARCHICAL

Multi-level decision tree with early termination.

**Best for:**
- Complex solutions with clear structure
- When you can make early decisions
- Time-constrained evaluations

```python
# High score on round 1 → skip to round 3
# Low score on round 1 → full sequential execution
```

---

## Objectives

### MAXIMIZE_ACCURACY

Prioritize correctness over speed or cost.

```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MAXIMIZE_ACCURACY
)
```

**Results in:**
- More evaluation rounds
- Higher thresholds
- More thorough testing
- Longer execution times

### MINIMIZE_TIME

Prioritize fast execution.

```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MINIMIZE_TIME
)
```

**Results in:**
- Parallel execution
- Lower thresholds
- Early termination
- Optimized scheduling

### MINIMIZE_COST

Prioritize computational efficiency.

```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MINIMIZE_COST
)
```

**Results in:**
- Fewer evaluations
- Resource-efficient allocation
- Cost-aware scheduling

### BALANCED (Default)

Balance all objectives.

```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED
)
```

---

## API Reference

### IntelligentGauntletOrchestrator

#### Constructor

```python
IntelligentGauntletOrchestrator(
    objective: OptimizationObjective = OptimizationObjective.BALANCED,
    max_parallelism: int = 4,
    enable_prediction: bool = True,
    enable_optimization: bool = True
)
```

**Parameters:**
- `objective`: Primary optimization goal
- `max_parallelism`: Maximum concurrent executions
- `enable_prediction`: Use predictive executor
- `enable_optimization`: Use ML optimizer

#### Methods

##### create_orchestration_plan()

```python
def create_orchestration_plan(
    self,
    solution: str,
    problem: str,
    domain: str,
    context: Optional[Dict[str, Any]] = None
) -> OrchestrationPlan
```

Create optimal orchestration plan.

**Returns:**
- `strategy`: Chosen orchestration strategy
- `execution_order`: Order of round execution
- `resource_allocation`: Resources per round
- `estimated_time`: Expected execution time
- `estimated_cost`: Expected computational cost

##### execute_orchestration()

```python
async def execute_orchestration(
    self,
    solution: str,
    problem: str,
    domain: str,
    plan: Optional[OrchestrationPlan] = None,
    gauntlet_executor: Optional[Any] = None
) -> OrchestrationResult
```

Execute gauntlet with intelligent orchestration.

**Returns:**
- `passed`: Whether solution passed
- `final_score`: Final aggregated score
- `rounds_completed`: Number of rounds completed
- `execution_time`: Total time taken
- `adaptations_made`: List of adaptations

---

## Examples

### Example 1: Domain-Specific Optimization

```python
# Different domains benefit from different strategies
domains = {
    "finance": OptimizationObjective.MAXIMIZE_ACCURACY,
    "web": OptimizationObjective.MINIMIZE_TIME,
    "general": OptimizationObjective.BALANCED
}

for domain, objective in domains.items():
    orchestrator = IntelligentGauntletOrchestrator(
        objective=objective
    )

    result = await orchestrator.execute_orchestration(
        solution=solution,
        problem=problem,
        domain=domain
    )

    print(f"{domain}: {result.passed}, {result.final_score:.3f}")
```

### Example 2: Batch Processing

```python
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED,
    max_parallelism=8
)

solutions = [sol1, sol2, sol3, sol4, sol5]

# Process in parallel
tasks = [
    orchestrator.execute_orchestration(
        solution=sol,
        problem=problem,
        domain="code"
    )
    for sol in solutions
]

results = await asyncio.gather(*tasks)

for i, result in enumerate(results):
    print(f"Solution {i+1}: {result.passed}")
```

### Example 3: Progressive Refinement

```python
# First pass: Quick evaluation
quick_orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MINIMIZE_TIME
)

quick_result = await quick_orchestrator.execute_orchestration(
    solution=solution,
    problem=problem,
    domain="code"
)

if quick_result.passed:
    print("Quick pass - solution is good")
else:
    print("Needs thorough evaluation")

    # Second pass: Thorough evaluation
    thorough_orchestrator = IntelligentGauntletOrchestrator(
        objective=OptimizationObjective.MAXIMIZE_ACCURACY
    )

    thorough_result = await thorough_orchestrator.execute_orchestration(
        solution=solution,
        problem=problem,
        domain="code"
    )
```

### Example 4: Resource-Constrained Execution

```python
# Limited resources
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MINIMIZE_COST,
    max_parallelism=2  # Limited parallelism
)

plan = orchestrator.create_orchestration_plan(
    solution=solution,
    problem=problem,
    domain="code"
)

print(f"Estimated cost: {plan.estimated_cost:.1f}")
print(f"Estimated time: {plan.estimated_time:.1f}s")

result = await orchestrator.execute_orchestration(
    solution=solution,
    problem=problem,
    domain="code",
    plan=plan
)

print(f"Actual cost: {result.actual_cost:.1f}")
```

---

## Best Practices

### 1. Choose the Right Objective

```python
# High-stakes domains
finance_orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MAXIMIZE_ACCURACY
)

# Low-risk domains
web_orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.MINIMIZE_TIME
)

# General purpose
general_orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED
)
```

### 2. Set Parallelism Appropriately

```python
# Match parallelism to available resources
import os

cpu_count = os.cpu_count()

# Use 50-75% of available cores
max_parallelism = max(1, int(cpu_count * 0.75))

orchestrator = IntelligentGauntletOrchestrator(
    max_parallelism=max_parallelism
)
```

### 3. Monitor Adaptations

```python
result = await orchestrator.execute_orchestration(
    solution=solution,
    problem=problem,
    domain="code"
)

# Review what adaptations were made
for adaptation in result.adaptations_made:
    print(f"Adaptation: {adaptation}")

# Adjust based on patterns
if len(result.adaptations_made) > 5:
    print("Many adaptations - consider reviewing solution quality")
```

### 4. Review Recommendations

```python
# Get recommendations for improvement
for recommendation in result.recommendations:
    print(f"Recommendation: {recommendation}")

# Get orchestration statistics over time
stats = orchestrator.get_orchestration_stats()
print(f"Pass rate: {stats['pass_rate']:.2%}")
print(f"Average score: {stats['average_score']:.3f}")
print(f"Average time: {stats['average_time']:.1f}s")
```

### 5. Plan Before Execution

```python
# Always create and review plan first
plan = orchestrator.create_orchestration_plan(
    solution=solution,
    problem=problem,
    domain=domain
)

# Review plan before executing
print(f"Strategy: {plan.strategy.value}")
print(f"Execution order: {plan.execution_order}")
print(f"Est. time: {plan.estimated_time:.1f}s")
print(f"Est. cost: {plan.estimated_cost:.1f}")

# Approve if acceptable
if plan.estimated_cost < budget:
    result = await orchestrator.execute_orchestration(
        solution=solution,
        problem=problem,
        domain=domain,
        plan=plan
    )
```

---

## Advanced Usage

### Custom Resource Allocation

```python
class CustomOrchestrator(IntelligentGauntletOrchestrator):
    def _allocate_resources(self, characteristics, execution_order):
        # Custom resource allocation logic
        allocation = {}

        for round_name in execution_order:
            if "round1" in round_name:
                allocation[round_name] = {
                    "max_evaluations": 100,
                    "timeout": 60,
                    "parallel": True
                }
            elif "round2" in round_name:
                allocation[round_name] = {
                    "max_attacks": 20,
                    "timeout": 120,
                    "parallel": False
                }

        return allocation
```

### Integration with ML Components

```python
# Enable all ML features
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED,
    enable_prediction=True,
    enable_optimization=True
)

# Set up integrations
from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import PredictiveGauntletExecutor
from glue.adapters.gauntlet_adapter.src.ml_optimizer import MLBasedGauntletOptimizer

orchestrator.set_predictive_executor(PredictiveGauntletExecutor())
orchestrator.set_ml_optimizer(MLBasedGauntletOptimizer())

# Execute with full AI optimization
result = await orchestrator.execute_orchestration(
    solution=solution,
    problem=problem,
    domain="domain"
)
```

---

## Troubleshooting

### Execution Timeout

**Issue**: Gauntlet execution times out.

**Solutions**:
1. Reduce `max_parallelism` to avoid resource contention
2. Use `OptimizationObjective.MINIMIZE_TIME`
3. Increase timeouts in plan
4. Check system resources

### Low Pass Rate

**Issue**: Most solutions are failing.

**Solutions**:
1. Use `OptimizationObjective.MINIMIZE_ACCURACY` to lower thresholds
2. Review solution quality before execution
3. Enable prediction to skip low-probability solutions
4. Adjust domain-specific difficulty settings

### High Cost

**Issue**: Execution is too expensive.

**Solutions**:
1. Use `OptimizationObjective.MINIMIZE_COST`
2. Reduce `max_parallelism`
3. Enable prediction to skip expensive evaluations
4. Use hierarchical strategy to skip unnecessary rounds

---

## Support

For issues or questions:
- GitHub: https://github.com/openevolve/intelligent-orchestrator/issues
- Documentation: https://docs.openevolve.org/intelligent-orchestrator
