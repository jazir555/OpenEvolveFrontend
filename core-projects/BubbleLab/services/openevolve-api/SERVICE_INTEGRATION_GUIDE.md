# BubbleLab Service Integration Guide

Complete guide for integrating OpenEvolve FastAPI service with BubbleLab services.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Service Adapters](#service-adapters)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Testing Integration](#testing-integration)
7. [Deployment](#deployment)

---

## Overview

The OpenEvolve FastAPI service integrates with three key BubbleLab services:

| Service | Purpose | Endpoint |
|---------|---------|----------|
| **Judge** | Code evaluation using visual LLM | `/api/evolution-judge` |
| **Mutate** | Code mutation for evolution | `/api/evolution-mutate` |
| **LeanAide** | Lean 4 theorem proving | `/api/leanaide` |
| **Z3** | SMT solving | (port 7655) |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│       OpenEvolve FastAPI Service            │
│                                             │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │ Evolution    │──────▶│ Judge Adapter  │ │
│  │ Engine       │      └────────┬─────────┘ │
│  └──────────────┘               │          │
│                                  ▼          │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │ Adversarial  │──────▶│ Mutate Adapter │─┼─▶ BubbleLab API
│  │ Engine       │      └────────┬─────────┘ │    (port 3001)
│  └──────────────┘               │          │
│                                  ▼          │
│  ┌──────────────┐      ┌─────────────────┐ │
│  │ Sovereign    │──────▶│ LeanAide       │ │
│  │ Engine       │      │ Adapter        │─┘
│  └──────────────┘      └─────────────────┘
```

---

## Service Adapters

### Judge Adapter

Evaluates generated code using visual LLM judge.

```python
from services.adapters import get_judge_adapter

# Get adapter instance
judge = get_judge_adapter()

# Evaluate single code
result = await judge.evaluate(
    code="def add(a, b): return a + b",
    problem_statement="Create a function to add two numbers",
    weights={
        "correctness": 0.4,
        "efficiency": 0.3,
        "style": 0.2,
        "documentation": 0.1,
    }
)

print(result["overall_score"])
print(result["criteria"])

# Evaluate batch
results = await judge.evaluate_batch(
    codes=["code1", "code2", "code3"],
    problem_statement="Problem statement here",
)

# Health check
is_healthy = await judge.health_check()
```

**Response Format**:
```json
{
  "overall_score": 0.85,
  "criteria": [
    {"name": "correctness", "score": 0.9, "feedback": "..."},
    {"name": "efficiency", "score": 0.8, "feedback": "..."},
    {"name": "style", "score": 0.85, "feedback": "..."},
    {"name": "documentation", "score": 0.75, "feedback": "..."}
  ]
}
```

### Mutate Adapter

Performs code mutations for evolutionary algorithms.

```python
from services.adapters import get_mutate_adapter

# Get adapter instance
mutate = get_mutate_adapter()

# Single mutation
result = await mutate.mutate(
    code="def add(a, b): return a + b",
    mutation_type="point",
    mutation_rate=0.1,
)

print(result["mutated_code"])
print(result["mutations_count"])

# Batch mutation
results = await mutate.mutate_batch(
    codes=["code1", "code2", "code3"],
    mutation_type="point",
    mutation_rate=0.1,
)

# Crossover
child = await mutate.crossover(
    code1="def add(a, b): return a + b",
    code2="def sum(x, y): return x + y",
    num_points=1,
)

# Health check
is_healthy = await mutate.health_check()
```

**Response Format**:
```json
{
  "mutated_code": "def add(a, b): return a + b  # mutated",
  "mutations_count": 2,
  "mutation_locations": [10, 25]
}
```

### LeanAide Adapter

Provides Lean 4 theorem proving capabilities.

```python
from services.adapters import get_leanaide_adapter

# Get adapter instance
leanaide = get_leanaide_adapter()

# Generate proof
result = await leanaide.generate_proof(
    proposition="∀ n : Nat, n + 0 = n",
    tactic="induction",
)

print(result["proof"])
print(result["formalized"])

# Verify proof
verification = await leanaide.verify_proof(
    proof=result["proof"],
    proposition="∀ n : Nat, n + 0 = n",
)

print(verification["is_valid"])
print(verification["errors"])

# Get available models
models = await leanaide.get_models()

# Run benchmark
benchmark = await leanaide.run_benchmark(
    benchmark_name="arith_basic",
    timeout=300.0,
)

# Health check
is_healthy = await leanaide.health_check()
```

**Response Format**:
```json
{
  "proof": "theorem add_zero (n : Nat) : n + 0 = n := by...",
  "formalized": "∀ (n : Nat), n + 0 = n",
  "is_valid": true,
  "errors": []
}
```

---

## Configuration

### Environment Variables

Create a `.env` file in the service directory:

```bash
# BubbleLab Service URLs
BUBBLELAB_API_URL=http://localhost:3001
JUDGE_API_URL=http://localhost:3001/api/evolution-judge
MUTATE_API_URL=http://localhost:3001/api/evolution-mutate
LEANAIDE_API_URL=http://localhost:3001/api/leanaide
Z3_API_URL=http://localhost:7655

# Timeouts (seconds)
JUDGE_TIMEOUT=60.0
MUTATE_TIMEOUT=60.0
LEANAIDE_TIMEOUT=120.0
Z3_TIMEOUT=60.0

# Execution
MAX_WORKERS=5
TASK_TIMEOUT=600.0

# CORS
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Python Configuration

```python
from services.config import settings

# Access settings
print(settings.JUDGE_API_URL)
print(settings.MAX_WORKERS)
print(settings.LOG_LEVEL)

# Settings are loaded from environment variables
# with sensible defaults
```

---

## Usage Examples

### Evolution Engine with Judge Integration

```python
from core.evolution import EvolutionEngine
from services.adapters import get_judge_adapter

async def evolution_workflow():
    # Initialize evolution engine
    engine = EvolutionEngine(
        max_iterations=10,
        population_size=5,
    )

    # Get judge adapter
    judge = get_judge_adapter()

    # Evaluate code
    code = "def add(a, b): return a + b"
    problem = "Create a function to add two numbers"

    evaluation = await judge.evaluate(code, problem)

    # Use evaluation score as fitness
    fitness = evaluation["overall_score"]

    print(f"Fitness: {fitness}")

    # Evolve code using mutations
    from services.adapters import get_mutate_adapter
    mutate = get_mutate_adapter()

    mutated = await mutate.mutate(code, mutation_rate=0.1)
    new_code = mutated["mutated_code"]

    # Evaluate mutated code
    new_evaluation = await judge.evaluate(new_code, problem)
    new_fitness = new_evaluation["overall_score"]

    print(f"New fitness: {new_fitness}")
```

### Adversarial Engine with Mutate Integration

```python
from core.adversarial import AdversarialEngine
from services.adapters import get_mutate_adapter

async def adversarial_workflow():
    # Initialize adversarial engine
    engine = AdversarialEngine(
        rounds=3,
        attack_types=["fuzzing", "mutation"],
    )

    # Get mutate adapter
    mutate = get_mutate_adapter()

    # Generate adversarial examples
    original_code = "def add(a, b): return a + b"

    # Apply mutations
    mutations = await mutate.mutate_batch(
        codes=[original_code] * 5,
        mutation_type="point",
        mutation_rate=0.2,
    )

    # Test robustness against mutations
    for mutation in mutations:
        mutated_code = mutation["mutated_code"]
        # Test if code still works
        print(f"Mutated: {mutated_code}")
```

### Sovereign Engine with LeanAide Integration

```python
from core.sovereign import SovereignEngine
from services.adapters import get_leanaide_adapter

async def sovereign_workflow():
    # Initialize sovereign engine
    engine = SovereignEngine(
        decomposition_depth=3,
        parallel_subproblems=5,
    )

    # Get LeanAide adapter
    leanaide = get_leanaide_adapter()

    # Decompose and prove theorem
    proposition = "∀ n : Nat, n + 0 = n"

    # Generate proof
    result = await leanaide.generate_proof(
        proposition=proposition,
        tactic="induction",
    )

    proof = result["proof"]

    # Verify proof
    verification = await leanaide.verify_proof(
        proof=proof,
        proposition=proposition,
    )

    if verification["is_valid"]:
        print("✅ Proof is valid")
    else:
        print(f"❌ Proof errors: {verification['errors']}")
```

---

## Testing Integration

### Health Checks

```python
import asyncio
from services.adapters import (
    get_judge_adapter,
    get_mutate_adapter,
    get_leanaide_adapter,
)

async def check_all_services():
    """Check health of all integrated services"""

    adapters = {
        "Judge": get_judge_adapter(),
        "Mutate": get_mutate_adapter(),
        "LeanAide": get_leanaide_adapter(),
    }

    results = {}

    for name, adapter in adapters.items():
        is_healthy = await adapter.health_check()
        results[name] = is_healthy
        status = "✅" if is_healthy else "❌"
        print(f"{status} {name} service")

    all_healthy = all(results.values())

    if all_healthy:
        print("\n🎉 All services healthy!")
    else:
        print("\n⚠️  Some services are unhealthy")

    return results

# Run health checks
asyncio.run(check_all_services())
```

### Integration Test

```python
import pytest
from services.adapters import (
    get_judge_adapter,
    get_mutate_adapter,
)

@pytest.mark.asyncio
async def test_evolution_workflow():
    """Test complete evolution workflow"""

    judge = get_judge_adapter()
    mutate = get_mutate_adapter()

    # Start with initial code
    code = "def add(a, b): return a + b"
    problem = "Create a function to add two numbers"

    # Evaluate fitness
    evaluation = await judge.evaluate(code, problem)
    initial_fitness = evaluation["overall_score"]

    # Mutate code
    mutation = await mutate.mutate(code, mutation_rate=0.1)
    mutated_code = mutation["mutated_code"]

    # Evaluate mutated code
    new_evaluation = await judge.evaluate(mutated_code, problem)
    new_fitness = new_evaluation["overall_score"]

    # Assert fitness improved or stayed same
    assert new_fitness >= initial_fitness

    print(f"Initial fitness: {initial_fitness}")
    print(f"New fitness: {new_fitness}")
```

---

## Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  # BubbleLab API
  bubblelab-api:
    build: ../../apps/bubblelab-api
    ports:
      - "3001:3001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}

  # OpenEvolve API
  openevolve-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - BUBBLELAB_API_URL=http://bubblelab-api:3001
      - JUDGE_API_URL=http://bubblelab-api:3001/api/evolution-judge
      - MUTATE_API_URL=http://bubblelab-api:3001/api/evolution-mutate
      - LEANAIDE_API_URL=http://bubblelab-api:3001/api/leanaide
    depends_on:
      - bubblelab-api

  # LeanAide Server (if standalone)
  leanaide:
    image: leanaide:latest
    ports:
      - "7654:7654"

  # Z3 Prover
  z3:
    image: z3prover:latest
    ports:
      - "7655:7655"
```

### Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: openevolve-config
data:
  BUBBLELAB_API_URL: "http://bubblelab-api:3001"
  JUDGE_API_URL: "http://bubblelab-api:3001/api/evolution-judge"
  MUTATE_API_URL: "http://bubblelab-api:3001/api/evolution-mutate"
  LEANAIDE_API_URL: "http://bubblelab-api:3001/api/leanaide"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openevolve-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openevolve-api
  template:
    metadata:
      labels:
        app: openevolve-api
    spec:
      containers:
      - name: openevolve-api
        image: openevolve-api:latest
        ports:
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: openevolve-config
```

---

## Troubleshooting

### Service Unavailable

**Error**: `Connection refused` when calling adapter

**Solution**:
```bash
# Check if BubbleLab API is running
curl http://localhost:3001/health

# Check specific service
curl http://localhost:3001/api/evolution-judge/health

# Start BubbleLab API
cd BubbleLab/apps/bubblelab-api
npm start
```

### Timeout Errors

**Error**: `Timeout during request`

**Solution**:
```python
# Increase timeout in settings
from services.config import settings

settings.JUDGE_TIMEOUT = 120.0  # Increase from 60.0
settings.LEANAIDE_TIMEOUT = 300.0  # Increase from 120.0
```

### Authentication Errors

**Error**: `401 Unauthorized` or `Missing API keys`

**Solution**:
```bash
# Set API keys in BubbleLab API
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Or in .env file
echo "OPENAI_API_KEY=your-key" >> BubbleLab/apps/bubblelab-api/.env
```

---

## Best Practices

1. **Always check health before using adapters**
```python
if not await judge.health_check():
    raise Exception("Judge service unavailable")
```

2. **Use batch operations when possible**
```python
# Better
results = await judge.evaluate_batch(codes, problem)

# Worse (slower)
for code in codes:
    result = await judge.evaluate(code, problem)
```

3. **Handle errors gracefully**
```python
try:
    result = await judge.evaluate(code, problem)
except httpx.TimeoutError:
    logger.error("Judge timeout")
    # Fallback logic
except Exception as e:
    logger.error("Judge failed", error=str(e))
    # Fallback logic
```

4. **Clean up resources**
```python
# Close adapters when done
await judge.close()
await mutate.close()
await leanaide.close()
```

---

**Last Updated**: 2026-01-27
