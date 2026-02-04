# Gauntlet System API Documentation

Complete API documentation for the OpenEvolve Gauntlet System.

## Overview

The Gauntlet System provides a comprehensive, multi-stage quality assurance framework for validating solutions through adversarial testing and consensus verification.

## Table of Contents

1. [REST API](#rest-api)
2. [WebSocket API](#websocket-api)
3. [Python API](#python-api)
4. [Authentication](#authentication)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## REST API

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Execute Gauntlet

Run a complete gauntlet evaluation against a solution.

**Endpoint:** `POST /gauntlets/execute`

**Request Body:**
```json
{
  "solution": "def solve(): return 42",
  "problem": "Create a function that returns the answer to life",
  "domain": "code",
  "config": {
    "round1_threshold": 0.5,
    "round2_threshold": 0.6,
    "round3_threshold": 0.7,
    "execution_order": "sequential"
  }
}
```

**Response (200 OK):**
```json
{
  "execution_id": "exec_abc123",
  "passed": true,
  "final_score": 0.85,
  "rounds_completed": 3,
  "rounds": [
    {
      "round": "round1_loongflow",
      "passed": true,
      "score": 0.9,
      "feedback": "Excellent solution",
      "execution_time": 5.2
    },
    {
      "round": "round2_red_team",
      "passed": true,
      "score": 0.8,
      "feedback": "Robust against adversarial attacks",
      "execution_time": 15.3
    },
    {
      "round": "round3_gold_team",
      "passed": true,
      "score": 0.85,
      "feedback": "Consensus achieved",
      "execution_time": 20.1
    }
  ],
  "total_time": 40.6,
  "timestamp": "2026-02-03T12:00:00Z"
}
```

#### 2. Create Gauntlet Definition

Create a custom gauntlet definition.

**Endpoint:** `POST /gauntlets`

**Request Body:**
```json
{
  "name": "security_validation",
  "description": "Security-focused gauntlet with penetration testing",
  "rounds": [
    {
      "rule_id": "automated_security_scan",
      "rule_type": "automated",
      "description": "Automated security vulnerability scan",
      "min_score": 0.85,
      "max_attempts": 3
    },
    {
      "rule_id": "red_team_penetration",
      "rule_type": "red_team",
      "description": "Red team penetration testing",
      "min_score": 0.75,
      "max_attempts": 3
    }
  ],
  "execution_order": "sequential",
  "stop_on_first_failure": false
}
```

**Response (201 Created):**
```json
{
  "gauntlet_id": "gauntlet_xyz789",
  "name": "security_validation",
  "created_at": "2026-02-03T12:00:00Z"
}
```

#### 3. Get Gauntlet Status

Get the status of a running gauntlet execution.

**Endpoint:** `GET /gauntlets/{execution_id}/status`

**Response (200 OK):**
```json
{
  "execution_id": "exec_abc123",
  "status": "in_progress",
  "current_round": "round2_red_team",
  "progress": 0.6,
  "estimated_time_remaining": 15.0
}
```

#### 4. List Gauntlets

List all available gauntlet definitions.

**Endpoint:** `GET /gauntlets`

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20)
- `type`: Filter by type (optional)

**Response (200 OK):**
```json
{
  "gauntlets": [
    {
      "gauntlet_id": "standard_validation",
      "name": "Standard Validation Gauntlet",
      "description": "3-round validation with automated tests, red team review, and gold team verification",
      "rounds_count": 3
    }
  ],
  "total": 25,
  "page": 1,
  "limit": 20
}
```

---

## WebSocket API

### Connection

Connect to the WebSocket server for real-time updates.

**URL:** `ws://localhost:8765`

### Event Types

#### Connection Events

**Client → Server:**
```json
{
  "event_type": "ping",
  "data": {},
  "timestamp": 1675435200.0
}
```

**Server → Client:**
```json
{
  "event_type": "connection_ack",
  "data": {
    "connection_id": "conn_xyz789"
  },
  "timestamp": 1675435200.0
}
```

#### Execution Events

**Subscribe to Execution:**
```json
{
  "event_type": "execution_started",
  "data": {
    "execution_id": "exec_abc123"
  },
  "execution_id": "exec_abc123"
}
```

**Progress Update:**
```json
{
  "event_type": "progress_update",
  "data": {
    "round_number": 2,
    "progress": 0.6,
    "status": "Running red team evaluation"
  },
  "execution_id": "exec_abc123",
  "timestamp": 1675435260.0
}
```

**Round Completed:**
```json
{
  "event_type": "round_completed",
  "data": {
    "round_number": 1,
    "passed": true,
    "score": 0.9,
    "feedback": "Excellent solution quality"
  },
  "execution_id": "exec_abc123",
  "timestamp": 1675435280.0
}
```

**Execution Completed:**
```json
{
  "event_type": "execution_completed",
  "data": {
    "passed": true,
    "final_score": 0.85,
    "rounds_completed": 3,
    "total_time": 40.6
  },
  "execution_id": "exec_abc123",
  "timestamp": 1675435300.0
}
```

---

## Python API

### ML-Based Gauntlet Optimizer

```python
from glue.adapters.gauntlet_adapter.src.ml_optimizer import (
    MLBasedGauntletOptimizer,
    OptimizationStrategy,
    OptimizationObjective,
    create_optimizer
)

# Create optimizer
optimizer = create_optimizer(
    strategy="q_learning",
    learning_rate=0.1,
    max_iterations=100
)

# Optimize configuration
result = optimizer.optimize(
    domain="code",
    objective=Objective.BALANCED
)

print(f"Best configuration: {result.best_state.to_dict()}")
print(f"Improvement: {result.improvement_percent:.1f}%")
```

### Predictive Gauntlet Executor

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
    solution="def solve(): return optimal",
    problem="Optimize portfolio allocation",
    domain="finance"
)

print(f"Success probability: {prediction.success_probability:.2%}")
print(f"Confidence: {prediction.confidence:.2%}")
print(f"Risk factors: {prediction.risk_factors}")

# Get execution plan
plan = executor.create_execution_plan(prediction)

if plan.decision == ExecutionDecision.PROCEED:
    # Execute gauntlet
    result = executor.execute_with_prediction(
        solution="def solve(): return optimal",
        problem="Optimize portfolio allocation",
        domain="finance",
        prediction=prediction
    )

    print(f"Passed: {result.actual_outcome['passed']}")
    print(f"Prediction accuracy: {result.prediction_accuracy:.2%}")
```

### Advanced Adaptive Learner

```python
from glue.adapters.gauntlet_adapter.src.adaptive_learner import (
    AdvancedAdaptiveLearner,
    LearningAlgorithm,
    create_learner
)

# Create learner
learner = create_learner(
    algorithm="dqn",
    state_size=8,
    action_size=10
)

# Train from historical data
metrics = learner.train_from_history(
    history=execution_history,
    episodes=100
)

# Get adaptive strategy
state = np.array([0.5, 0.6, 0.7, 0.5, 0.5, 3.0, 0.5, 0.5], dtype=np.float32)
strategy = learner.get_adaptive_strategy(state)

print(f"Recommended strategy: {strategy}")
```

### Intelligent Gauntlet Orchestrator

```python
from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
    IntelligentGauntletOrchestrator,
    OptimizationObjective,
    OrchestrationStrategy
)

# Create orchestrator
orchestrator = IntelligentGauntletOrchestrator(
    objective=OptimizationObjective.BALANCED,
    max_parallelism=4
)

# Create orchestration plan
plan = orchestrator.create_orchestration_plan(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance"
)

print(f"Strategy: {plan.strategy.value}")
print(f"Estimated time: {plan.estimated_time:.1f}s")

# Execute with orchestration
result = await orchestrator.execute_orchestration(
    solution="def solve(): return optimal",
    problem="Optimize portfolio",
    domain="finance",
    plan=plan
)

print(f"Passed: {result.passed}")
print(f"Score: {result.final_score:.3f}")
print(f"Adaptations: {result.adaptations_made}")
```

### Three-Round Gauntlet Orchestrator

```python
from core_projects.openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundGauntletOrchestrator,
    ThreeRoundConfig,
    create_balanced_config
)

# Create orchestrator with configuration
config = create_balanced_config()
orchestrator = ThreeRoundGauntletOrchestrator(config=config)

# Run full gauntlet
result = await orchestrator.run_full_gauntlet(
    solution="def solve(): return optimal_solution",
    problem="Optimize the packing problem",
    domain="math"
)

print(f"Passed: {result.passed}")
print(f"Final Score: {result.final_score:.3f}")
print(f"Rounds Completed: {result.rounds_completed}")
print(f"Total Time: {result.total_time:.2f}s")
```

---

## Authentication

### API Key Authentication

Include your API key in the request header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/api/v1/gauntlets/execute
```

### WebSocket Authentication

Include authentication token in the initial connection message:

```json
{
  "event_type": "authenticate",
  "data": {
    "token": "YOUR_AUTH_TOKEN"
  }
}
```

---

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid solution format",
    "details": {
      "field": "solution",
      "issue": "Solution cannot be empty"
    }
  },
  "timestamp": "2026-02-03T12:00:00Z"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Invalid input parameters |
| `NOT_FOUND` | Resource not found |
| `AUTHENTICATION_ERROR` | Invalid or missing credentials |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Server error |

---

## Rate Limiting

### Limits

- **Default**: 100 requests per minute
- **WebSocket**: 10 messages per second per connection

### Rate Limit Headers

Response includes rate limit information:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1675435260
```

### Handling Rate Limits

When rate limited, the API returns `429 Too Many Requests`:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "retry_after": 60
    }
  }
}
```

Implement exponential backoff for retries:

```python
import time

def make_request_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        response = requests.post(url)

        if response.status_code == 429:
            retry_after = response.json()['error']['details']['retry_after']
            time.sleep(retry_after * (2 ** attempt))
        else:
            return response

    raise Exception("Max retries exceeded")
```

---

## Quick Start Example

```python
import asyncio
from glue.adapters.gauntlet_adapter.src.intelligent_orchestrator import (
    IntelligentGauntletOrchestrator,
    OptimizationObjective
)

async def main():
    # Create orchestrator
    orchestrator = IntelligentGauntletOrchestrator(
        objective=OptimizationObjective.BALANCED
    )

    # Execute gauntlet with intelligent orchestration
    result = await orchestrator.execute_orchestration(
        solution="""
def optimize_portfolio(returns, risk_tolerance=0.1):
    n = len(returns)
    weights = np.ones(n) / n

    for _ in range(1000):
        gradient = np.dot(returns, weights)
        weights -= 0.01 * gradient
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

    return weights
        """,
        problem="Optimize portfolio allocation for maximum return with given risk tolerance",
        domain="finance"
    )

    print(f"Result: {result.to_dict()}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Support

For issues or questions:
- GitHub: https://github.com/openevolve/gauntlet-system
- Documentation: https://docs.openevolve.org/gauntlets
- Email: support@openevolve.org
