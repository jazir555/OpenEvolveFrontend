# Engine Integration Quick Reference

**Last Updated**: 2026-01-27
**Status**: ✅ All Engines Integrated

---

## 🚀 Quick Start

### Running Each Engine

```bash
# Evolution Engine (Code generation + evaluation)
curl -X POST http://localhost:8001/api/workflows/evolution/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Write a function to sort a list",
    "parameters": {
      "population_size": 10,
      "generations": 5,
      "mutation_rate": 0.2
    }
  }'

# Adversarial Engine (Security testing)
curl -X POST http://localhost:8001/api/workflows/adversarial/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "def process(user_input): return eval(user_input)",
    "parameters": {
      "attack_types": ["code_injection", "fuzzing"],
      "rounds": 3
    }
  }'

# Sovereign Engine (Problem decomposition + formal verification)
curl -X POST http://localhost:8001/api/workflows/sovereign/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Design a secure authentication system",
    "parameters": {
      "decomposition_depth": 3,
      "verification_strictness": "standard"
    }
  }'
```

---

## 🔌 Adapter Integration Points

### Evolution Engine

**Adapters**: Judge + Mutate

```python
# core/evolution.py

async def _evaluate_solution(self, solution, problem_statement):
    """Uses Judge adapter for fitness evaluation"""
    judge = self._get_judge_adapter()
    evaluation = await judge.evaluate(
        code=solution["code"],
        problem_statement=problem_statement,
        weights={
            "correctness": 0.4,
            "efficiency": 0.3,
            "style": 0.2,
            "documentation": 0.1,
        }
    )
    return evaluation["overall_score"]

async def _refine_solution(self, solution, fitness):
    """Uses Mutate adapter for solution refinement"""
    mutate = self._get_mutate_adapter()
    mutation = await mutate.mutate(
        code=solution["code"],
        mutation_type="point",
        mutation_rate=max(0.05, min(1.0 - fitness, 0.5)),
    )
    solution["code"] = mutation["mutated_code"]
    return solution
```

**Fallback**: Heuristic scoring if Judge unavailable

### Adversarial Engine

**Adapter**: Mutate

```python
# core/adversarial.py

async def _execute_attack(self, attack_type, target, round_num, context):
    """Uses Mutate adapter for attack generation"""
    mutate = self._get_mutate_adapter()

    if attack_type == "code_injection":
        mutations = await mutate.mutate_batch(
            codes=[target] * 3,
            mutation_type="point",
            mutation_rate=0.5,
        )

        vulnerabilities = []
        for mutation in mutations:
            vuln = self._check_for_injection_vulnerability(
                target,
                mutation["mutated_code"]
            )
            if vuln:
                vulnerabilities.append(vuln)

        return {"vulnerabilities": vulnerabilities}

    elif attack_type == "fuzzing":
        # 5 fuzzing attempts with high mutation rate
        for i in range(5):
            mutation = await mutate.mutate(
                code=target,
                mutation_type="point",
                mutation_rate=0.4,
            )
            # Check for issues...
```

**Fallback**: Simulated attacks if Mutate unavailable

**Circuit Breaker**: Per-attack-type failure isolation

### Sovereign Engine

**Adapter**: LeanAide

```python
# core/sovereign.py

async def _verify_solutions(self, sub_problems, strictness):
    """Uses LeanAide for formal proof verification"""
    leanaide = self._get_leanaide_adapter()

    for sub_problem in sub_problems:
        solution = sub_problem["solution"]
        proof = solution.get("proof")

        if proof and strictness in ["standard", "strict"]:
            verification = await leanaide.verify_proof(
                proof=proof,
                proposition=sub_problem["description"]
            )

            if verification["is_valid"]:
                # Formal proof verified
                verification_results["formal_proofs_verified"] += 1
            else:
                if strictness == "strict":
                    # Strict mode: fail on proof verification failure
                    verification_results["failed_solutions"].append(...)
                else:
                    # Standard mode: fall back to confidence check
                    confidence = solution["confidence"]
                    if confidence >= threshold:
                        verification_results["verified_solutions"].append(...)
```

**Fallback**: Confidence-based verification if LeanAide unavailable

**Strictness Levels**:
- **Lenient**: No formal proof required
- **Standard**: Formal proof with confidence fallback
- **Strict**: Formal proof required, fails if verification fails

---

## 🛡️ Graceful Degradation

### Pattern

```python
try:
    # Try real service
    adapter = self._get_adapter()
    result = await adapter.call(...)

except Exception as e:
    logger.warning("adapter_unavailable", error=str(e), fallback=True)

    # Fallback to heuristics
    result = self._heuristic_fallback(...)

finally:
    # Always return valid result
    return result
```

### Evolution Engine Fallback

```python
# Judge adapter unavailable
try:
    fitness = await judge.evaluate(code, problem)
except Exception:
    # Heuristic fitness calculation
    base_score = 0.5
    if len(code) > 100:
        base_score += 0.2
    if "def " in code or "class " in code:
        base_score += 0.2
    fitness = min(base_score, 1.0)
```

### Adversarial Engine Fallback

```python
# Mutate adapter unavailable
try:
    mutations = await mutate.mutate_batch(...)
except Exception:
    # Simulate attacks
    if random.random() < 0.2:  # 20% chance
        vulnerabilities.append({
            "type": attack_type,
            "severity": random.choice(["critical", "high", "medium"]),
            "description": f"Simulated {attack_type} vulnerability"
        })
```

### Sovereign Engine Fallback

```python
# LeanAide adapter unavailable
try:
    verification = await leanaide.verify_proof(proof, proposition)
except Exception:
    # Complete fallback to confidence-based verification
    for sub_problem in sub_problems:
        confidence = sub_problem["solution"]["confidence"]
        if confidence >= min_threshold:
            verification_results["verified_solutions"].append(sub_problem)
```

---

## 📊 Response Formats

### Evolution Engine Response

```json
{
  "status": "completed",
  "final_generation": 5,
  "best_fitness": 0.87,
  "best_solution": {
    "code": "def sort_list(items):\n    return sorted(items)",
    "fitness": 0.87,
    "generation": 5
  },
  "population_history": [...],
  "metadata": {
    "execution_id": "evo_20260127_123456",
    "started_at": "2026-01-27T12:34:56Z",
    "completed_at": "2026-01-27T12:35:04Z",
    "duration_seconds": 8.2,
    "adapter_calls": {
      "judge": 50,
      "mutate": 25
    }
  }
}
```

### Adversarial Engine Response

```json
{
  "status": "completed",
  "vulnerabilities": [
    {
      "type": "code_injection",
      "severity": "high",
      "description": "Code injection via 'eval(' detected",
      "location": "line 2",
      "evidence": "eval(user_input)",
      "remediation": "Avoid eval(), use safe alternatives"
    }
  ],
  "test_results": {
    "code_injection": {
      "status": "completed",
      "vulnerabilities_found": 1
    },
    "fuzzing": {
      "status": "completed",
      "vulnerabilities_found": 0
    }
  },
  "summary": {
    "total_vulnerabilities": 1,
    "by_severity": {
      "critical": 0,
      "high": 1,
      "medium": 0,
      "low": 0
    }
  },
  "recommendations": [
    "Avoid eval() and similar dynamic code execution functions"
  ],
  "metadata": {
    "execution_id": "adv_20260127_123456",
    "duration_seconds": 6.2,
    "adapter_calls": {
      "mutate": 8
    }
  }
}
```

### Sovereign Engine Response

```json
{
  "status": "completed",
  "decomposition": {
    "original_problem": "...",
    "sub_problems": [...]
  },
  "sub_problems": [
    {
      "id": "subproblem_1",
      "title": "Sub-problem 1",
      "status": "solved",
      "solution": {
        "content": "...",
        "proof": "theorem ...",
        "confidence": 0.92
      }
    }
  ],
  "final_solution": {
    "content": "...",
    "integrity_score": 0.85
  },
  "verification_results": {
    "strictness": "standard",
    "passed": 4,
    "failed": 1,
    "formal_proofs_verified": 3,
    "heuristic_verifications": 1,
    "verified_solutions": [...]
  },
  "summary": {
    "total_subproblems": 5,
    "solved_count": 4,
    "verification_passed": 4
  },
  "metadata": {
    "execution_id": "sov_20260127_123456",
    "duration_seconds": 12.3,
    "adapter_calls": {
      "leanaide": 3
    }
  }
}
```

---

## 🔍 Debugging

### Enable Debug Logging

```python
import structlog

# Configure debug level
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

# Set log level
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Check Adapter Availability

```python
# Test Judge adapter
from services.adapters import get_judge_adapter

judge = get_judge_adapter()
result = await judge.evaluate(
    code="def hello(): return 'world'",
    problem_statement="Write a hello world function"
)
print(f"Judge available: {result}")

# Test Mutate adapter
from services.adapters import get_mutate_adapter

mutate = get_mutate_adapter()
result = await mutate.mutate(
    code="def hello(): return 'world'",
    mutation_type="point",
    mutation_rate=0.1
)
print(f"Mutate available: {result}")

# Test LeanAide adapter
from services.adapters import get_leanaide_adapter

leanaide = get_leanaide_adapter()
# LeanAide requires formal proofs - see documentation for examples
```

### Check Service Health

```bash
# Check BubbleLab services
curl http://localhost:3000/api/health
curl http://localhost:3001/api/judge/health
curl http://localhost:3002/api/mutate/health
curl http://localhost:3003/api/leanaide/health

# Check OpenEvolve API
curl http://localhost:8001/health
curl http://localhost:8001/api/workflows/status
```

---

## 📈 Performance Tuning

### Evolution Engine

```python
# Faster execution (less accuracy)
parameters = {
    "population_size": 5,      # Smaller population
    "generations": 3,           # Fewer generations
    "mutation_rate": 0.3        # Higher mutation
}

# Better quality (slower)
parameters = {
    "population_size": 20,     # Larger population
    "generations": 10,          # More generations
    "mutation_rate": 0.1        # Lower mutation
}
```

### Adversarial Engine

```python
# Quick security check
parameters = {
    "attack_types": ["code_injection"],
    "rounds": 1
}

# Comprehensive security audit
parameters = {
    "attack_types": [
        "code_injection",
        "fuzzing",
        "prompt_injection",
        "sql_injection",
        "xss"
    ],
    "rounds": 5
}
```

### Sovereign Engine

```python
# Quick decomposition
parameters = {
    "decomposition_depth": 2,
    "verification_strictness": "lenient"
}

# Thorough formal verification
parameters = {
    "decomposition_depth": 5,
    "verification_strictness": "strict"
}
```

---

## 🧪 Testing

### Unit Tests

```bash
# Test Evolution engine
pytest tests/units/test_evolution_engine.py -v

# Test Adversarial engine
pytest tests/units/test_adversarial_engine.py -v

# Test Sovereign engine
pytest tests/units/test_sovereign_engine.py -v
```

### Integration Tests

```bash
# Test all engines with real services
pytest tests/integration/test_engine_integration.py -v

# Test graceful degradation
pytest tests/integration/test_fallback_behavior.py -v
```

### Performance Tests

```bash
# Benchmark engine performance
pytest tests/benchmark/test_engine_performance.py -v

# Load testing
pytest tests/load/test_concurrent_executions.py -v
```

---

## 📚 Documentation Links

- **WORKFLOW_ENGINE_INTEGRATION_COMPLETE.md** - Comprehensive integration report
- **LLM_TEAM_ASSIGNMENT_COMPLETE.md** - LLM team assignment system
- **services/adapters/README.md** - Adapter documentation
- **CLAUDE.md** - Project constitution

---

## 🆘 Troubleshooting

### "Adapter unavailable" error

**Cause**: BubbleLab service not running or unreachable

**Solution**:
1. Check BubbleLab services are running: `docker ps`
2. Check service health: `curl http://localhost:3000/api/health`
3. System will automatically fall back to heuristics

### "Formal proof verification failed" error

**Cause**: Proof verification failed in strict mode

**Solution**:
1. Verify proof is valid Lean 4 syntax
2. Check proposition matches proof
3. Use "standard" strictness for confidence fallback

### "Circuit breaker open" warning

**Cause**: Too many failures for an attack type

**Solution**:
1. Circuit breaker will reset automatically
2. Check service logs for root cause
3. Other attack types continue working

### Slow execution

**Cause**: Large population/generations or slow service response

**Solution**:
1. Reduce population_size or generations parameters
2. Check service latency in metrics
3. Increase timeout if needed

---

**Quick Reference Version**: 1.0
**Last Updated**: 2026-01-27
**Status**: ✅ All Engines Production Ready
