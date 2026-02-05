# Gauntlet Selection Guide

> **Quick Reference**: Choose the right gauntlet type for your validation needs

## Selection Flowchart

```
Start
  │
  ├─► Need security/robustness testing? ──► Adversarial Gauntlet
  │
  ├─► Need formal proofs/safety verification? ──► Formal Verification Gauntlet
  │
  ├─► Need statistical validation? ──► Statistical Gauntlet
  │
  ├─► Domain-specific problem?
  │   ├─► Physics ──► Domain (Physics) Gauntlet
  │   ├─► Finance ──► Domain (Finance) Gauntlet
  │   ├─► Chemistry ──► Domain (Chemistry) Gauntlet
  │   └─► Engineering ──► Domain (Engineering) Gauntlet
  │
  ├─► Multiple competing objectives? ──► Multi-Objective Gauntlet
  │
  ├─► Need fitness-based evaluation? ──► Evolutionary Gauntlet
  │
  ├─► Time-series/stability testing? ──► Temporal Gauntlet
  │
  └─► Need robustness/generalization? ──► Cross-Validation Gauntlet
```

---

## Gauntlet Selection Matrix

| Validation Need | Recommended Gauntlet | Secondary Options |
|----------------|---------------------|-------------------|
| Security flaws | **Adversarial** | Statistical |
| Formal correctness | **Formal Verification** | Statistical |
| Edge cases | **Adversarial** | Cross-Validation |
| Performance bounds | **Statistical** | Temporal |
| Physics simulation | **Domain (Physics)** | Formal Verification |
| Financial model | **Domain (Finance)** | Statistical |
| Chemical process | **Domain (Chemistry)** | Statistical |
| Structural design | **Domain (Engineering)** | Formal Verification |
| Trade-off analysis | **Multi-Objective** | Statistical |
| Algorithm quality | **Evolutionary** | Cross-Validation |
| Stability over time | **Temporal** | Statistical |
| Generalization | **Cross-Validation** | Statistical |
| Resource usage | **Multi-Objective** | Evolutionary |

---

## Detailed Selection Criteria

### 1. Adversarial Gauntlet

**Choose when:**
- ✅ Security validation is critical
- ✅ You need to find edge cases
- ✅ Robustness against attacks matters
- ✅ Input validation needs testing

**Don't choose when:**
- ❌ You need mathematical proofs
- ❌ The solution has no security aspects
- ❌ Performance is the main concern

**Attack Modes:**
| Mode | Description | Use Case |
|-----|-------------|----------|
| `systematic` | Methodical coverage | General robustness |
| `focused_attack` | Target specific areas | Known vulnerabilities |
| `deep_dive` | Intensive single-area | Critical components |
| `adversarial` | ML-style adversarial | Neural networks |
| `poka_yoke` | Error-proofing | Safety-critical systems |

**Example:**
```python
gauntlet = AdversarialGauntlet("security_test", {
    "attack_modes": ["systematic", "focused_attack"],
    "use_blue_team": True
})
```

---

### 2. Formal Verification Gauntlet

**Choose when:**
- ✅ Safety-critical systems
- ✅ Mathematical correctness required
- ✅ No failures can be tolerated
- ✅ Properties can be formally defined

**Don't choose when:**
- ❌ Properties are unclear
- ❌ Z3 solver not available
- ❌ Time constraints are tight (>60s timeout)

**Property Types:**
| Property | Description | Example |
|---------|-------------|---------|
| `null_safety` | No null pointer exceptions | `x is not None` checks |
| `bounds_check` | Array/index within bounds | `0 <= i < len(arr)` |
| `type_safety` | Type correctness | No type confusion |
| `invariant` | System invariants | Balance >= 0 |

**Example:**
```python
gauntlet = FormalVerificationGauntlet("safety_check", {"timeout": 30})
result = gauntlet.execute(solution, {
    "properties": [
        {"name": "null_safety"},
        {"name": "bounds_check"}
    ]
})
```

---

### 3. Statistical Gauntlet

**Choose when:**
- ✅ Probabilistic behavior needs validation
- ✅ Large sample data available
- ✅ Distribution fitting required
- ✅ A/B testing scenarios

**Don't choose when:**
- ❌ Deterministic validation needed
- ❌ Sample size is small (<30)
- ❌ Data distribution unknown

**Test Types:**
| Test | Purpose | Confidence |
|-----|---------|------------|
| `mean` | Central tendency | 95% |
| `variance` | Spread/consistency | 95% |
| `distribution` | Overall fit | 90% |

**Example:**
```python
gauntlet = StatisticalGauntlet("stats_test", {
    "num_samples": 1000,
    "tests": ["mean", "variance"]
})
```

---

### 4. Domain-Specific Gauntlets

#### Physics Gauntlet
**Choose for:**
- Physical simulations
- Engineering calculations
- Scientific computing

**Checks:** Unit consistency, dimensional analysis, conservation laws, physical constraints

#### Finance Gauntlet
**Choose for:**
- Trading algorithms
- Risk models
- Portfolio optimization

**Checks:** Arbitrage detection, risk bounds, regulatory compliance, portfolio constraints

#### Chemistry Gauntlet
**Choose for:**
- Reaction simulations
- Molecular modeling
- Process optimization

**Checks:** Stoichiometry, reaction validity, safety constraints, thermodynamic feasibility

#### Engineering Gauntlet
**Choose for:**
- Structural design
- Mechanical systems
- Manufacturing processes

**Checks:** Safety factors, stress analysis, material constraints, manufacturability

---

### 5. Multi-Objective Gauntlet

**Choose when:**
- ✅ Multiple competing objectives exist
- ✅ Trade-off analysis needed
- ✅ Pareto optimality matters
- ✅ Resource allocation decisions

**Objectives:**
| Objective | Minimize/Maximize | Weight Typical |
|-----------|------------------|----------------|
| Cost | Minimize | 0.3 |
| Performance | Maximize | 0.4 |
| Reliability | Maximize | 0.2 |
| Time | Minimize | 0.1 |

**Example:**
```python
gauntlet = MultiObjectiveGauntlet("mo_test", {
    "objectives": ["cost", "performance", "reliability"],
    "weights": [0.3, 0.5, 0.2]
})
```

---

### 6. Evolutionary Gauntlet

**Choose when:**
- ✅ Fitness landscape exploration needed
- ✅ Competitive algorithm comparison
- ✅ Generational improvement tracking
- ✅ Population-based validation

**Don't choose when:**
- ❌ Single-shot evaluation sufficient
- ❌ Computation time limited

**Example:**
```python
gauntlet = EvolutionaryGauntlet("evo_test", {
    "population_size": 50,
    "generations": 10
})
```

---

### 7. Temporal Gauntlet

**Choose when:**
- ✅ Dynamic system validation
- ✅ Stability over time matters
- ✅ Convergence verification
- ✅ Time-series analysis

**Metrics:**
| Metric | Good Value | Bad Value |
|--------|------------|-----------|
| Stability | CV < 0.1 | CV > 0.3 |
| Convergence | Variance < 0.01 | Variance > 0.1 |
| Trend | Stable/Improving | Degrading |

---

### 8. Cross-Validation Gauntlet

**Choose when:**
- ✅ Generalization assessment needed
- ✅ Overfitting detection
- ✅ Model validation
- ✅ Dataset robustness

**K-Fold Selection:**
| Dataset Size | Recommended K |
|-------------|---------------|
| < 100 | 3-5 |
| 100-1000 | 5-10 |
| > 1000 | 10 |

---

## Orchestration Selection

### Sequential
**Use when:**
- Gauntlets have dependencies
- Early termination desired
- Resource conservation important

### Parallel
**Use when:**
- Gauntlets are independent
- Time is critical
- Resources available

### Hierarchical
**Use when:**
- Multi-level validation needed
- Different validation phases
- Progressive refinement

### Adaptive
**Use when:**
- Validation depth should vary
- Performance-based selection
- Resource optimization

### Chain
**Use when:**
- Output feeds to next gauntlet
- Progressive improvement
- Iterative refinement

---

## Quick Decision Tree

```
Q1: Is security the primary concern?
    YES → Adversarial Gauntlet
    NO → Continue

Q2: Do you need mathematical proof?
    YES → Formal Verification Gauntlet
    NO → Continue

Q3: Is this a domain-specific problem?
    YES → Domain-Specific Gauntlet
    NO → Continue

Q4: Are there multiple competing objectives?
    YES → Multi-Objective Gauntlet
    NO → Continue

Q5: Is the system dynamic/time-varying?
    YES → Temporal Gauntlet
    NO → Continue

Q6: Do you have statistical data?
    YES → Statistical Gauntlet
    NO → Continue

Q7: Is generalization important?
    YES → Cross-Validation Gauntlet
    NO → Evolutionary Gauntlet
```

---

## Performance Guidelines

| Scenario | Recommended Setup |
|----------|-------------------|
| Quick validation | Single Statistical gauntlet |
| Standard validation | Sequential: Domain + Statistical |
| Thorough validation | Hierarchical: 3 levels, parallel per level |
| Security audit | Adversarial + Formal (sequential) |
| Production release | All applicable types (hierarchical) |
| Research prototype | Adaptive with 3-4 types |

---

## Examples by Use Case

### Web Application Security
```python
gauntlets = [
    AdversarialGauntlet("web_security", {
        "attack_modes": ["systematic", "focused_attack"]
    }),
    FormalVerificationGauntlet("input_validation", {
        "timeout": 30
    })
]
```

### Financial Model
```python
gauntlets = [
    DomainSpecificGauntlet("finance", "finance_check"),
    StatisticalGauntlet("model_validation", {
        "num_samples": 10000
    }),
    CrossValidationGauntlet("generalization", {"k_folds": 10})
]
```

### Engineering Design
```python
gauntlets = [
    DomainSpecificGauntlet("engineering", "design_check"),
    MultiObjectiveGauntlet("tradeoffs", {
        "objectives": ["cost", "safety", "performance"]
    }),
    FormalVerificationGauntlet("safety_bounds")
]
```

### Machine Learning Model
```python
gauntlets = [
    CrossValidationGauntlet("generalization", {"k_folds": 5}),
    StatisticalGauntlet("performance_dist"),
    TemporalGauntlet("prediction_stability")
]
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Gauntlet too slow | Use Parallel orchestration or reduce samples |
| False positives | Increase pass thresholds or use Chain mode |
| Missed issues | Add Adversarial gauntlet or increase depth |
| Inconsistent results | Use Cross-Validation or Statistical |
| Memory issues | Reduce population_size or num_samples |
| Timeout errors | Increase timeout or use Sequential mode |

---

**Version**: 1.0  
**Last Updated**: February 4, 2026
