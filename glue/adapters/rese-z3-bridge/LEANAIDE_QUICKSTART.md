# LeanAide Integration - Quick Start Guide

Get started with LeanAide integration in 5 minutes.

---

## Prerequisites

1. **LeanAide Server Running**
   ```bash
   # Check if LeanAide is available
   curl http://localhost:7654/
   ```

2. **Python 3.8+**
   ```bash
   python --version
   ```

3. **Verify Integration**
   ```bash
   cd glue/adapters/rese-z3-bridge
   bash probes/check_leanaide.sh
   ```

---

## Installation

```bash
cd glue/adapters/rese-z3-bridge
pip install -r requirements.txt
```

---

## 5-Minute Examples

### Example 1: Autoformalize (2 minutes)

Convert natural language to Lean 4:

```python
from rese_z3_bridge import RESEZ3Bridge

# Create bridge
bridge = RESEZ3Bridge()

# Autoformalize
response = bridge.autoformalize(
    natural_language="There are infinitely many prime numbers",
    theorem_name="infinitely_many_primes",
)

# Check result
if response.success:
    print("✓ Autoformalization successful!")
    print(f"\nGenerated Lean 4 code:\n{response.lean_code}")
else:
    print(f"✗ Failed: {response.error}")

# Cleanup
bridge.close()
```

**Expected Output:**
```
✓ Autoformalization successful!

Generated Lean 4 code:
theorem infinitely_many_primes :
  Infinite {p : Nat | Prime p} := by
  sorry
```

---

### Example 2: AI-Powered Proving (2 minutes)

Generate proofs automatically:

```python
from rese_z3_bridge import RESEZ3Bridge

bridge = RESEZ3Bridge()

# Prove theorem
response = bridge.prove_with_ai(
    theorem_text="For all natural numbers n, n + 0 = n",
)

if response.success:
    print("✓ Proof generated!")
    print(f"\nProof:\n{response.proof}")
    print(f"\nTactics used: {response.tactics_used}")
else:
    print(f"✗ Failed: {response.error}")

bridge.close()
```

---

### Example 3: Z3 to Lean Translation (1 minute)

Bridge Z3 constraints to Lean 4:

```python
from rese_z3_bridge import RESEZ3Bridge, ConstraintType

bridge = RESEZ3Bridge()

# Z3 constraint
smtlib = """
(declare-fun x () Real)
(declare-fun y () Real)
(assert (> x 0.0))
(assert (> y 0.0))
(assert (> (+ x y) 0.0))
"""

# Translate to Lean
response = bridge.translate_z3_to_lean(
    smtlib_content=smtlib,
    constraint_type=ConstraintType.REAL,
)

if response.success:
    print("✓ Translation successful!")
    print(f"\nLean 4 code:\n{response.lean_code}")
    print(f"\nVariables: {response.variables}")

bridge.close()
```

---

## Common Use Cases

### Use Case 1: Formalize Mathematical Theorems

```python
theorems = [
    "There are infinitely many primes",
    "The square root of 2 is irrational",
    "Every natural number has a unique prime factorization",
]

bridge = RESEZ3Bridge()

for theorem in theorems:
    response = bridge.autoformalize(natural_language=theorem)
    if response.success:
        print(f"✓ {theorem}")
        print(f"  {response.lean_code[:80]}...")

bridge.close()
```

### Use Case 2: Get Proof Help

```python
bridge = RESEZ3Bridge()

# Stuck on a proof? Get tactic suggestions
response = bridge.suggest_tactics(
    goal_state="⊢ x + y = y + x",
    num_suggestions=3,
)

for suggestion in response.suggestions:
    print(f"{suggestion.tactic}: {suggestion.description}")
    print(f"Confidence: {suggestion.confidence:.2f}\n")

bridge.close()
```

### Use Case 3: Verify Z3 Results in Lean

```python
bridge = RESEZ3Bridge()

# Step 1: Check with Z3
z3_result = bridge.solve_constraints(
    variables=[],
    constraints=[...],
)

# Step 2: Translate to Lean for formal verification
if z3_result.result.value == "unsat":
    lean_response = bridge.translate_z3_to_lean(
        smtlib_content=smtlib,
    )
    # Now you have a formal Lean 4 theorem to verify

bridge.close()
```

---

## Environment Configuration

### Minimal Setup

```bash
# Only required if defaults don't work
export LEANAIDE_BASE_URL=http://localhost:7654
export LEANAIDE_TIMEOUT_MS=60000
export LEANAIDE_ENABLE=true
```

### Full Configuration

```bash
# LeanAide
export LEANAIDE_BASE_URL=http://localhost:7654
export LEANAIDE_TIMEOUT_MS=60000
export LEANAIDE_ENABLE=true

# Z3
export Z3_BASE_URL=http://localhost:8000
export Z3_TIMEOUT_MS=30000

# Resilience
export Z3_CIRCUIT_BREAKER_THRESHOLD=5
export Z3_MAX_RETRIES=3
export Z3_ENABLE_CACHE=true
export Z3_CACHE_TTL_MS=300000
```

---

## Troubleshooting

### Problem: "Connection refused"

**Solution:** Start LeanAide server
```bash
# Check server status
curl http://localhost:7654/

# If not running, start LeanAide
cd /path/to/leanaide
python -m leanaide.server --port 7654
```

### Problem: "Timeout"

**Solution:** Increase timeout
```python
response = bridge.autoformalize(
    natural_language=complex_theorem,
    timeout_ms=120000,  # 2 minutes
)
```

### Problem: "Circuit breaker open"

**Solution:** Wait 60 seconds or fix LeanAide server
```bash
# Check server health
curl http://localhost:7654/

# Check circuit breaker status
python -c "
from rese_z3_bridge import RESEZ3Bridge
bridge = RESEZ3Bridge()
print(bridge.get_stats()['client_stats']['circuit_breaker'])
"
```

---

## Next Steps

1. **Read Full Documentation**
   ```bash
   cat docs/LEANAIDE_INTEGRATION.md
   ```

2. **Run Tests**
   ```bash
   pytest tests/test_leanaide_integration.py -v
   ```

3. **Check Integration**
   ```bash
   bash probes/check_leanaide.sh
   ```

4. **Explore Examples**
   - See `docs/LEANAIDE_INTEGRATION.md` for advanced examples
   - See `README.md` for API reference

---

## Quick Reference

| Method | Purpose | Timeout |
|--------|---------|---------|
| `autoformalize()` | NL → Lean 4 | 60s |
| `prove_with_ai()` | Generate proofs | 60s |
| `translate_z3_to_lean()` | Z3 → Lean 4 | 30s |
| `suggest_tactics()` | Get tactics | 15s |

---

## Support

- **Documentation:** `docs/LEANAIDE_INTEGRATION.md`
- **Tests:** `tests/test_leanaide_integration.py`
- **Probes:** `probes/check_leanaide.sh`
- **Issues:** Check logs with correlation IDs

---

**Happy Proving! 🎯**
