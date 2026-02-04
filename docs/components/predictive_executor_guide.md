# Predictive Gauntlet Executor - User Guide

Complete guide for using the Predictive Gauntlet Executor component.

## Overview

The Predictive Gauntlet Executor uses machine learning to predict gauntlet outcomes before execution, enabling intelligent resource allocation and dynamic difficulty adjustment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prediction API](#prediction-api)
3. [Execution Planning](#execution-planning)
4. [Adaptive Execution](#adaptive-execution)
5. [Best Practices](#best-practices)

---

## Quick Start

### Basic Prediction

```python
from glue.adapters.gauntlet_adapter.src.predictive_gauntlet_executor import (
    PredictiveGauntletExecutor
)

# Create executor
executor = PredictiveGauntletExecutor(
    success_threshold=0.3,
    confidence_threshold=0.6
)

# Predict success
prediction = executor.predict_success(
    solution="def solve(): return 42",
    problem="Return the answer to life",
    domain="code"
)

print(f"Success Probability: {prediction.success_probability:.2%}")
print(f"Confidence: {prediction.confidence:.2%}")
print(f"Risk Factors: {prediction.risk_factors}")
```

### Execution with Prediction

```python
# Get execution plan
plan = executor.create_execution_plan(prediction)

if plan.decision == ExecutionDecision.PROCEED:
    # Execute gauntlet
    result = executor.execute_with_prediction(
        solution="def solve(): return 42",
        problem="Return the answer to life",
        domain="code",
        prediction=prediction
    )

    print(f"Passed: {result.actual_outcome['passed']}")
    print(f"Prediction Accuracy: {result.prediction_accuracy:.2%}")
else:
    print(f"Skipped: {plan.reasoning}")
```

---

## Prediction API

### Predict Success

```python
def predict_success(
    self,
    solution: str,
    problem: str,
    domain: str,
    context: Optional[Dict[str, Any]] = None
) -> PredictionResult
```

Predict the success probability for a given solution.

**Returns:**
- `success_probability`: 0.0 to 1.0
- `confidence`: 0.0 to 1.0
- `risk_factors`: List of identified risks
- `recommended_difficulty`: "easy", "medium", or "hard"
- `estimated_time`: Estimated execution time in seconds
- `estimated_cost`: Computational cost units

**Example:**

```python
prediction = executor.predict_success(
    solution=complex_solution,
    problem=hard_problem,
    domain="math"
)

if prediction.success_probability > 0.7:
    print("High chance of passing - proceed")
elif prediction.success_probability > 0.4:
    print("Moderate chance - consider adjustments")
else:
    print("Low chance - skip or improve solution")
```

---

## Execution Planning

### Create Execution Plan

```python
def create_execution_plan(
    self,
    prediction: PredictionResult,
    base_config: Optional[Dict[str, Any]] = None
) -> ExecutionPlan
```

Create an execution plan based on prediction.

**Possible Decisions:**

1. **PROCEED**: Execute gauntlet with standard or adjusted config
2. **SKIP_LOW_PROBABILITY**: Skip due to low success probability
3. **SKIP_HIGH_COST**: Skip due to high estimated cost
4. **ADJUST_DIFFICULTY**: Proceed with adjusted difficulty

**Example:**

```python
prediction = executor.predict_success(
    solution=solution,
    problem=problem,
    domain="finance"
)

plan = executor.create_execution_plan(prediction)

if plan.decision == ExecutionDecision.PROCEED:
    print(f"Proceed with config: {plan.adjusted_config}")
elif plan.decision == ExecutionDecision.ADJUST_DIFFICULTY:
    print(f"Adjusting difficulty: {plan.reasoning}")
    print(f"New config: {plan.adjusted_config}")
else:
    print(f"Skipping: {plan.reasoning}")
```

---

## Adaptive Execution

### Execute with Prediction

```python
def execute_with_prediction(
    self,
    solution: str,
    problem: str,
    domain: str,
    prediction: Optional[PredictionResult] = None,
    config: Optional[Dict[str, Any]] = None,
    gauntlet_executor: Optional[Any] = None
) -> ExecutionResult
```

Execute gauntlet with prediction guidance.

**Returns:**
- `prediction`: Original prediction
- `actual_outcome`: Actual execution result
- `prediction_accuracy`: How accurate prediction was
- `execution_time`: Time taken
- `cost_savings`: Resources saved by skipping low-probability executions

**Example:**

```python
result = executor.execute_with_prediction(
    solution=solution,
    problem=problem,
    domain="code"
)

print(f"Prediction: {result.prediction.success_probability:.2%}")
print(f"Actual: {result.actual_outcome['passed']}")
print(f"Accuracy: {result.prediction_accuracy:.2%}")

if result.cost_savings > 0:
    print(f"Saved ${result.cost_savings:.2f} by skipping")
```

---

## Best Practices

### 1. Set Appropriate Thresholds

```python
# Lenient: Allow more executions
executor = PredictiveGauntletExecutor(
    success_threshold=0.2,  # Low threshold
    confidence_threshold=0.5
)

# Strict: Only high-probability executions
executor = PredictiveGauntletExecutor(
    success_threshold=0.5,  # High threshold
    confidence_threshold=0.8
)

# Balanced: Middle ground
executor = PredictiveGauntletExecutor(
    success_threshold=0.3,  # Default
    confidence_threshold=0.6  # Default
)
```

### 2. Use Context Information

```python
# Provide context for better predictions
context = {
    "author": "experienced_developer",
    "previous_success_rate": 0.85,
    "time_spent": 3600  # seconds
}

prediction = executor.predict_success(
    solution=solution,
    problem=problem,
    domain="code",
    context=context
)
```

### 3. Track Prediction Accuracy

```python
# Execute multiple predictions
for i in range(10):
    result = executor.execute_with_prediction(
        solution=solutions[i],
        problem=problems[i],
        domain=domains[i]
    )

# Get accuracy statistics
stats = executor.get_prediction_accuracy_stats()
print(f"Mean Accuracy: {stats['mean_accuracy']:.2%}")
print(f"Total Predictions: {stats['total_predictions']}")
```

### 4. Continuous Learning

```python
# Each execution improves future predictions
result = executor.execute_with_prediction(
    solution=solution,
    problem=problem,
    domain="domain"
)

# Learning data is automatically stored
learning_data = result.learning_data
```

---

## Advanced Usage

### Batch Prediction

```python
solutions = [sol1, sol2, sol3, sol4, sol5]
problems = [prob1, prob2, prob3, prob4, prob5]

predictions = []
for sol, prob in zip(solutions, problems):
    pred = executor.predict_success(
        solution=sol,
        problem=prob,
        domain="code"
    )
    predictions.append(pred)

# Filter high-probability solutions
good_solutions = [
    (sol, pred) for sol, pred in zip(solutions, predictions)
    if pred.success_probability > 0.7
]

print(f"High-probability solutions: {len(good_solutions)}/5")
```

### Domain-Specific Thresholds

```python
domain_thresholds = {
    "math": {"success": 0.4, "confidence": 0.7},
    "code": {"success": 0.3, "confidence": 0.6},
    "finance": {"success": 0.5, "confidence": 0.8},
    "general": {"success": 0.3, "confidence": 0.5}
}

def get_executor(domain):
    thresholds = domain_thresholds.get(domain, {})
    return PredictiveGauntletExecutor(**thresholds)

# Use domain-specific executor
math_executor = get_executor("math")
code_executor = get_executor("code")
```

### Cost Optimization

```python
# Set low cost threshold to save resources
executor = PredictiveGauntletExecutor(
    success_threshold=0.3,
    confidence_threshold=0.6,
    cost_threshold=50.0  # Lower threshold
)

# Will skip expensive executions
result = executor.execute_with_prediction(
    solution=complex_solution,
    problem=hard_problem,
    domain="math"
)

if result.cost_savings > 0:
    print(f"Saved {result.cost_savings:.1f} cost units")
```

---

## Troubleshooting

### Low Prediction Accuracy

**Issue**: Predictions are not accurate.

**Solutions**:
1. Provide more historical data
2. Include more context information
3. Use domain-specific configurations
4. Collect more execution examples

### Too Many Skips

**Issue**: Most executions are being skipped.

**Solutions**:
1. Lower `success_threshold`
2. Lower `confidence_threshold`
3. Lower `cost_threshold`
4. Review prediction logic for domain-specific issues

### High False Negative Rate

**Issue**: Good solutions are being predicted as failures.

**Solutions**:
1. Lower `success_threshold`
2. Adjust risk factor weights
3. Collect more positive training examples
4. Review feature extraction logic

---

## Support

For issues or questions:
- GitHub: https://github.com/openevolve/predictive-executor/issues
- Documentation: https://docs.openevolve.org/predictive-executor
