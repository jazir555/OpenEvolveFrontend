# Strategy Selection Quick Reference

## 🚀 Quick Start

```python
from decomposition_engine import DecompositionEngine
from sovereign_data_models import ProblemDefinition, ...

# Create engine (intelligent selection enabled by default)
engine = DecompositionEngine()

# Decompose problem (automatically selects best strategy)
plan = engine.decompose(problem)

# Check which strategy was used
print(f"Strategy: {plan.strategy}")
```

---

## 📊 Weight-Based Strategy Selection

The system analyzes 5 dimensions and selects the best decomposition strategy:

| Dimension | What It Detects | Maps To |
|-----------|----------------|---------|
| **Functional** | Components, modules, interfaces | Semantic |
| **Temporal** | Phases, stages, timelines | Semantic |
| **Risk** | Criticality, security, complexity | Complexity |
| **Value** | Business value, priorities, stakeholders | Semantic |
| **Technical** | Dependencies, infrastructure, APIs | Dependency |

---

## 🎯 Selection Logic

```
1. Calculate weights for all 5 dimensions (0.0 - 1.0)
2. Find dimension with highest weight
3. IF highest weight > 0.6:
      Use that single strategy
   ELSE:
      Use hybrid approach combining top strategies
```

---

## 💡 Example Scenarios

### Scenario 1: Modular Component System
```python
problem = ProblemDefinition(
    title="Build modular component system",
    description="Create independent modules with clear interfaces",
    ...
)

# Weights calculated:
# functional: 0.75 (high - components, modules, interfaces)
# technical: 0.62 (medium - interfaces)
# value_based: 0.30 (low)
# risk_based: 0.15 (low)
# temporal: 0.00 (none)

# Result: semantic strategy (weight 0.75 > 0.6)
```

### Scenario 2: Phased Project
```python
problem = ProblemDefinition(
    title="Multi-phase project",
    description="Phase 1: Foundation, Phase 2: Features, Phase 3: Testing",
    ...
)

# Weights calculated:
# technical: 0.59 (medium)
# functional: 0.55 (medium)
# value_based: 0.30 (low)
# risk_based: 0.15 (low)
# temporal: 0.12 (low - but phases detected)

# Result: hybrid strategy (no clear winner > 0.6)
```

### Scenario 3: Security System
```python
problem = ProblemDefinition(
    title="Security System",
    description="Implement secure authentication with risk mitigation",
    constraints=[
        Constraint(type="security", severity="hard", ...)
    ],
    complexity_score=ComplexityScore(overall_complexity=9.0, ...)
)

# Weights calculated:
# risk_based: 0.60 (high - security keywords + high complexity)
# functional: 0.35 (medium)
# technical: 0.34 (medium)
# value_based: 0.30 (low)
# temporal: 0.00 (none)

# Result: complexity strategy (weight 0.60 > 0.6)
```

---

## ⚙️ Configuration

### Enable/Disable Intelligent Selection

```python
# Default: enabled
engine = DecompositionEngine(use_intelligent_selection=True)

# Disable (use LLM-based instead)
engine = DecompositionEngine(use_intelligent_selection=False)
```

### Force LLM-Based Selection

```python
# Even with intelligent enabled, can force LLM
strategy = engine.select_strategy(problem, use_llm=True)
```

### View Selection Reasoning

```python
import logging
logging.basicConfig(level=logging.INFO)

# This will log:
# - All 5 weights
# - Which strategy was selected
# - Why (threshold logic)

strategy = select_decomposition_strategy_v2(problem)
```

---

## 📈 Performance

| Method | Time | Cost | Deterministic |
|--------|------|------|--------------|
| **Intelligent v2** | < 0.01s | Free | ✅ Yes |
| **LLM-based** | 2-5s | Tokens | ❌ No |

**Speedup: 500x faster with intelligent selection!**

---

## 🔧 API Reference

### Functions

```python
def select_decomposition_strategy_v2(
    problem: ProblemDefinition,
    analyzed_context=None
) -> str:
    """Select best decomposition strategy using weight-based algorithm."""
```

### DecompositionEngine Methods

```python
def select_strategy_intelligent(
    problem: ProblemDefinition
) -> str:
    """Select strategy using intelligent algorithm (always)."""

def select_strategy(
    problem: ProblemDefinition,
    use_llm: bool = False
) -> str:
    """Select strategy (intelligent by default, LLM if use_llm=True)."""
```

---

## 🎓 Best Practices

1. **Use Intelligent Selection by Default**
   - Faster, cheaper, deterministic
   - Works well for most problems

2. **Enable LLM Selection When**
   - Problem has nuanced characteristics
   - You need LLM reasoning
   - Time/cost not a concern

3. **Check Weights for Complex Problems**
   - Enable INFO logging to see all weights
   - Understand why a strategy was selected
   - Verify it matches expectations

4. **Trust the Hybrid**
   - When no clear winner (> 0.6)
   - Hybrid approach combines best aspects
   - More robust than arbitrary choice

---

## 🐛 Troubleshooting

### "Wrong strategy selected"

**Solution:** Check the weights with logging enabled:
```python
import logging
logging.basicConfig(level=logging.INFO)
strategy = select_decomposition_strategy_v2(problem)
# Review all 5 weights in logs
```

### "Always selects hybrid"

**Cause:** No single dimension has strong indicators (> 0.6)

**Solution:** This is correct behavior! The problem genuinely has mixed characteristics. Hybrid is the safest choice.

### "Want LLM selection back"

**Solution:**
```python
# Method 1: Disable intelligent selection
engine = DecompositionEngine(use_intelligent_selection=False)

# Method 2: Force LLM for specific call
strategy = engine.select_strategy(problem, use_llm=True)
```

---

## 📚 Related Documentation

- **Implementation Details:** See `STRATEGY_SELECTION_COMPLETE.md`
- **Specification:** See `Decomposition_Workflow.md` lines 1760-1782
- **Tests:** See `test_strategy_selection_simple.py`

---

**Quick Reference Version:** 1.0
**Last Updated:** 2026-01-03
**Status:** Production Ready ✅
