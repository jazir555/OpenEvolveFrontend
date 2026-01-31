# OpenEvolve-Only Mode Quick Reference

## Quick Start

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

# Create selector (auto-detects LoongFlow)
selector = EnsembleStrategySelector(knowledge_engine=my_ke)

# Get recommendation (auto-fallback to OpenEvolve-only if needed)
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize portfolio allocation",
    domain="finance",
    constraints={"objectives": ["maximize_returns", "minimize_risk"]}
)

print(f"System: {prediction.strategy[0]}")
print(f"Mode: {prediction.strategy[1]}")
```

## Three Ways to Use

### 1. Automatic Fallback (Default)

```python
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=True  # Auto-disabled if unavailable
)

# Automatically uses OpenEvolve-only if LoongFlow unavailable
prediction = await selector.recommend_with_ensemble(...)
```

### 2. Force OpenEvolve-Only (Per-Instance)

```python
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=False  # Force OpenEvolve-only
)

# Always uses OpenEvolve-only
prediction = await selector.recommend_with_ensemble(...)
```

### 3. Force OpenEvolve-Only (Per-Request)

```python
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=True
)

# This specific request uses OpenEvolve-only
prediction = await selector.recommend_with_ensemble(
    problem_description="...",
    domain="...",
    constraints={},
    enable_loongflow=False  # Override for this call
)
```

## Convenience Method

```python
# Explicit OpenEvolve-only recommendation
prediction = await selector.recommend_openevolve_only(
    problem_description="Design robust bridge",
    domain="engineering",
    constraints={"safety_critical": True}
)
```

## Check Status

```python
# Check if LoongFlow is available
if selector.is_loongflow_available():
    print("Full ensemble available")
else:
    print("OpenEvolve-only mode")

# Get available modes
modes = selector.get_available_modes()
print(f"Available modes: {modes}")
# Full: ['pes', 'qd', 'mo', 'adversarial', 'standard']
# OpenEvolve-only: ['qd', 'mo', 'adversarial', 'standard']
```

## OpenEvolve Modes

| Mode         | Use For                        | Example Domains        |
|--------------|--------------------------------|------------------------|
| **Standard** | General optimization           | Web, general           |
| **QD**       | Diversity exploration          | Science, pharma        |
| **MO**       | Multi-objective optimization   | Finance, engineering   |
| **Adversarial** | Robustness testing        | Engineering, trading   |

## Decision Logic

```
Multiple objectives?  → MO mode
Requires diversity?   → QD mode
Requires robustness?  → Adversarial mode
Otherwise            → Standard mode
```

## Example Scenarios

### Finance Portfolio Optimization

```python
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize portfolio for risk-adjusted returns",
    domain="finance",
    constraints={
        "objectives": ["maximize_returns", "minimize_risk"],
        "constraints": ["budget_limit"]
    }
)

# Result: OpenEvolve MO mode (multi-objective)
```

### Scientific Exploration

```python
prediction = await selector.recommend_with_ensemble(
    problem_description="Find diverse protein folding structures",
    domain="science",
    constraints={"requires_diversity": True}
)

# Result: OpenEvolve QD mode (diversity)
```

### Safety-Critical Engineering

```python
prediction = await selector.recommend_with_ensemble(
    problem_description="Design bridge to withstand extreme conditions",
    domain="engineering",
    constraints={
        "objectives": ["minimize_weight", "maximize_strength"],
        "safety_critical": True
    }
)

# Result: OpenEvolve Adversarial mode (robustness)
```

### Web Optimization

```python
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize web page load time",
    domain="web",
    constraints={"time_limit_seconds": 1}
)

# Result: OpenEvolve Standard mode (default)
```

## Reading Predictions

```python
prediction = await selector.recommend_with_ensemble(...)

# Extract strategy
(system, mode), agreement = prediction.strategy

# Extract performance
point_estimate = prediction.point_estimate  # Expected performance
lower_bound, upper_bound = prediction.confidence_interval  # 95% CI
confidence_level = prediction.confidence_level  # 0.95

# Extract reasoning
reasoning = prediction.reasoning  # Human-readable explanation
methods_used = prediction.prediction_methods  # Methods that agreed

# Print summary
print(f"Recommended: {system}/{mode}")
print(f"Expected performance: {point_estimate:.2%}")
print(f"95% CI: [{lower_bound:.2%}, {upper_bound:.2%}]")
print(f"Agreement: {agreement:.1%}")
print(f"Reasoning: {reasoning}")
```

## Cold Start Handling

```python
# When no historical data available
prediction = await selector.handle_cold_start(
    problem_chars=problem_chars,
    domain="new_domain",
    enable_loongflow=False  # OpenEvolve-only cold start
)

# Lower confidence but still valid
assert prediction.confidence_level < 1.0
assert prediction.strategy[0].value == "openevolve"
```

## Testing

```bash
# Run OpenEvolve-only tests
cd knowledge_engine
python tests/test_openevolve_simple.py

# Expected output:
# ALL TESTS PASSED!
# OpenEvolve-only mode is working correctly!
```

## Troubleshooting

### Problem: Still recommends LoongFlow when disabled

**Solution:**
```python
# Use explicit override
prediction = await selector.recommend_with_ensemble(
    ...,
    enable_loongflow=False  # Explicit override
)

# Or use convenience method
prediction = await selector.recommend_openevolve_only(...)
```

### Problem: Low confidence in OpenEvolve-only mode

**Solution:**
```python
# This is expected when:
# 1. Cold start (no historical data)
# 2. LoongFlow unavailable (fewer options)
# 3. Complex problem (harder to predict)

# Address by:
# - Adding historical data
# - Enabling learning (default: enabled)
# - Providing more problem context
```

### Problem: Wrong mode selected

**Solution:**
```python
# Check problem characteristics
problem_chars = await selector.analyze_problem_characteristics(
    problem="...",
    domain="...",
    constraints={}
)

print(f"Multi-objective: {problem_chars.has_multiple_objectives}")
print(f"Requires diversity: {problem_chars.requires_diversity}")
print(f"Requires robustness: {problem_chars.requires_robustness}")

# If wrong, provide better constraints
constraints = {
    "has_multiple_objectives": True,  # Force MO
    "requires_diversity": True,  # Force QD
    "safety_critical": True  # Force Adversarial
}
```

## Key Takeaways

1. **Automatic**: Selector auto-detects LoongFlow availability
2. **Seamless**: Same API works with or without LoongFlow
3. **Flexible**: Can force OpenEvolve-only if needed
4. **Complete**: All ensemble methods work in OpenEvolve-only
5. **Tested**: Comprehensive test suite included

## Full Documentation

See:
- [OpenEvolve-Only Mode Documentation](./OPENEVOLVE_ONLY_MODE.md)
- [Implementation Summary](./STRATEGY_SELECTOR_UPDATE_SUMMARY.md)
- [Strategy Recommender Documentation](./STRATEGY_RECOMMENDER.md)
