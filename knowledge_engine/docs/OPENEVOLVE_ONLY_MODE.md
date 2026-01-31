# OpenEvolve-Only Mode

## Overview

The Strategy Selector now supports OpenEvolve-only operation when LoongFlow is unavailable or disabled. This ensures the knowledge engine can provide intelligent evolutionary strategy recommendations even without LoongFlow's PES (Plan-Execute-Summarize) capabilities.

## Features

### 1. Automatic LoongFlow Detection

The `LoongFlowChecker` class automatically detects if LoongFlow is available:

```python
from knowledge_engine.core.strategy_recommender import LoongFlowChecker

# Check availability
is_available = LoongFlowChecker.is_available()
print(f"LoongFlow available: {is_available}")
```

### 2. OpenEvolve-Only Recommendations

The selector automatically falls back to OpenEvolve-only mode when LoongFlow is unavailable:

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

# Initialize (will auto-detect LoongFlow)
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=True  # Will be disabled if unavailable
)

# Get recommendation (automatically uses OpenEvolve-only if needed)
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize portfolio allocation",
    domain="finance",
    constraints={"objectives": ["maximize_returns", "minimize_risk"]}
)
```

### 3. Explicit OpenEvolve-Only Mode

You can explicitly request OpenEvolve-only mode:

```python
# Method 1: Disable at initialization
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=False  # Force OpenEvolve-only
)

# Method 2: Override at recommendation time
prediction = await selector.recommend_with_ensemble(
    problem_description="...",
    domain="finance",
    constraints={},
    enable_loongflow=False  # Force OpenEvolve-only for this call
)

# Method 3: Use convenience method
prediction = await selector.recommend_openevolve_only(
    problem_description="...",
    domain="finance",
    constraints={}
)
```

## OpenEvolve Modes

When operating in OpenEvolve-only mode, the selector can recommend these modes:

### 1. Standard Mode
Traditional evolutionary algorithm with selection, mutation, and crossover.

**Best for:**
- Simple optimization problems
- Well-understood search spaces
- Fast evaluations

### 2. QD (Quality-Diversity) Mode
MAP-Elites algorithm for behavioral diversity exploration.

**Best for:**
- Finding diverse solutions
- Exploring behavioral space
- Innovation and novelty discovery

**Domains:** Science, pharma, finance

### 3. MO (Multi-Objective) Mode
Pareto optimization for multiple conflicting objectives.

**Best for:**
- Trade-off analysis
- Multiple competing objectives
- Decision support

**Domains:** Finance, engineering, pharma

### 4. Adversarial Mode
Red team / Blue team co-evolution for robustness testing.

**Best for:**
- Safety-critical systems
- Robustness validation
- Adversarial environments

**Domains:** Engineering, pharma, trading

## Decision Logic

### Rule-Based Selection (OpenEvolve-Only)

```
IF has_multiple_objectives THEN
    → RECOMMEND: MO mode
    → Confidence: 0.90

ELSE IF requires_diversity THEN
    → RECOMMEND: QD mode
    → Confidence: 0.80

ELSE IF requires_robustness THEN
    → RECOMMEND: Adversarial mode
    → Confidence: 0.85

ELSE
    → RECOMMEND: Standard mode
    → Confidence: 0.75
```

### Domain-Specific Defaults

When no clear rule applies, domain defaults are used:

| Domain      | Default Mode    | Rationale                          |
|-------------|-----------------|------------------------------------|
| Finance     | Standard        | Can still work well                |
| Trading     | Adversarial     | OpenEvolve adversarial is good     |
| Science     | QD              | QD for exploration                 |
| Engineering | Standard        | Reliable baseline                  |
| Pharma      | QD              | QD for chemical space              |
| Web Design  | Standard        | Fast, adequate                     |
| General     | Standard        | Safe default                       |

## Ensemble Methods

All four ensemble prediction methods work in OpenEvolve-only mode:

### 1. Rule-Based
Deterministic rules based on problem characteristics. Always available.

### 2. Similarity-Based
Finds similar historical OpenEvolve runs and uses their best strategies.

### 3. Trend-Based
Analyzes recent performance trends across OpenEvolve modes.

### 4. ML-Based
Trains a Random Forest classifier on OpenEvolve historical data.

## API Reference

### EnsembleStrategySelector

#### `__init__(knowledge_engine, llm_client, use_ai_analysis, learning_enabled, enable_ml, enable_loongflow)`

**Parameters:**
- `enable_loongflow` (bool): Enable LoongFlow recommendations (auto-disabled if unavailable)

#### `recommend_with_ensemble(problem_description, domain, constraints, confidence_level, enable_loongflow)`

**Parameters:**
- `enable_loongflow` (Optional[bool]): Override to force OpenEvolve-only or LoongFlow-only

**Returns:**
- `EnsemblePrediction` with recommended strategy

#### `recommend_openevolve_only(problem_description, domain, constraints, confidence_level)`

Convenience method for OpenEvolve-only recommendations.

**Returns:**
- `EnsemblePrediction` with OpenEvolve-only strategy

#### `is_loongflow_available() -> bool`

Check if LoongFlow is available for recommendations.

#### `get_available_modes() -> List[str]`

Get list of available evolutionary modes.

**Returns:**
- Full ensemble: `["pes", "qd", "mo", "adversarial", "standard"]`
- OpenEvolve-only: `["qd", "mo", "adversarial", "standard"]`

### LoongFlowChecker

#### `is_available() -> bool`

Check if LoongFlow can be imported and used.

#### `reset()`

Reset cached availability check (for testing).

## Examples

### Example 1: Automatic Fallback

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

# Create selector (will auto-detect LoongFlow)
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke
)

# Check availability
if selector.is_loongflow_available():
    print("Full ensemble (including LoongFlow PES) available")
else:
    print("OpenEvolve-only mode (LoongFlow unavailable)")

# Get recommendation (automatically adapts)
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize neural network architecture",
    domain="science",
    constraints={
        "objectives": ["maximize_accuracy", "minimize_latency"],
        "time_limit_seconds": 600
    }
)

print(f"Recommended: {prediction.strategy}")
print(f"Reasoning: {prediction.reasoning}")
```

### Example 2: Explicit OpenEvolve-Only

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

# Force OpenEvolve-only mode
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=False
)

# Get recommendation (guaranteed OpenEvolve-only)
prediction = await selector.recommend_with_ensemble(
    problem_description="Design robust bridge structure",
    domain="engineering",
    constraints={
        "objectives": ["minimize_weight", "maximize_strength"],
        "safety_critical": True
    }
)

system, mode = prediction.strategy
assert system == "openevolve"  # Guaranteed
assert mode == "adversarial"   # For safety-critical

print(f"OpenEvolve mode: {mode}")
print(f"Confidence: {prediction.point_estimate:.2%}")
```

### Example 3: Per-Request Override

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

# Create selector with LoongFlow enabled
selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=True
)

# Recommendation 1: Full ensemble (if LoongFlow available)
pred1 = await selector.recommend_with_ensemble(
    problem_description="Trading strategy optimization",
    domain="trading",
    constraints={}
)

# Recommendation 2: Force OpenEvolve-only for this call
pred2 = await selector.recommend_with_ensemble(
    problem_description="Trading strategy optimization",
    domain="trading",
    constraints={},
    enable_loongflow=False  # Override: OpenEvolve-only
)
```

### Example 4: Checking Available Modes

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

selector = EnsembleStrategySelector(knowledge_engine=my_ke)

# Get available modes
modes = selector.get_available_modes()

if "pes" in modes:
    print("Full ensemble available (includes LoongFlow PES)")
    print(f"Available modes: {modes}")
else:
    print("OpenEvolve-only mode")
    print(f"Available modes: {modes}")
    # Output: ["qd", "mo", "adversarial", "standard"]
```

### Example 5: Cold Start with OpenEvolve-Only

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

selector = EnsembleStrategySelector(
    knowledge_engine=my_ke,
    enable_loongflow=False
)

# Analyze problem
problem_chars = await selector.analyze_problem_characteristics(
    problem="Find diverse protein folding structures",
    domain="pharma",
    constraints={"requires_diversity": True}
)

# Handle cold start (no historical data)
prediction = await selector.handle_cold_start(
    problem_chars=problem_chars,
    domain="pharma",
    enable_loongflow=False
)

print(f"Cold start recommendation: {prediction.strategy}")
print(f"Confidence: {prediction.confidence_level:.1%}")
# Lower confidence due to cold start, but still valid
```

## Migration Guide

### From LoongFlow-Dependent Code

**Before (required LoongFlow):**

```python
# Would fail if LoongFlow unavailable
from loongflow.agents.math_agent import MathEvolveAgent

prediction = await selector.recommend_with_ensemble(...)
# Error: ImportError if LoongFlow not installed
```

**After (graceful fallback):**

```python
# No import needed, auto-detects availability
prediction = await selector.recommend_with_ensemble(...)
# Works with or without LoongFlow
```

### Updating Configuration Files

**Before:**

```yaml
# config.yaml
loongflow:
  enabled: true
  path: /path/to/loongflow
```

**After:**

```yaml
# config.yaml
loongflow:
  enabled: true  # Will be auto-disabled if unavailable
  # Optional: path config (auto-detected if not specified)
```

## Performance Considerations

### OpenEvolve-Only vs Full Ensemble

**OpenEvolve-Only:**
- Pros: Always available, no external dependencies
- Cons: Cannot leverage LoongFlow's 60% efficiency gain on expensive evaluations

**Full Ensemble (with LoongFlow):**
- Pros: Optimal for expensive evaluations, broader mode selection
- Cons: Requires LoongFlow installation and availability

### When to Use OpenEvolve-Only

Use OpenEvolve-only mode when:

1. **LoongFlow not installed** - Automatic fallback
2. **Cheap evaluations** - OpenEvolve works fine
3. **No PES requirement** - Standard evolutionary algorithms sufficient
4. **Deployment constraints** - Cannot install LoongFlow

### When to Use Full Ensemble

Use full ensemble (with LoongFlow) when:

1. **Expensive evaluations** - LoongFlow PES saves 60% of evaluations
2. **Need optimal performance** - Best of both systems
3. **Planning complexity** - LoongFlow's planning phase helps
4. **LoongFlow available** - No reason not to use it

## Testing

### Unit Tests

```bash
# Run OpenEvolve-only tests
pytest knowledge_engine/tests/test_strategy_selector_openevolve_only.py -v
```

### Integration Tests

```bash
# Run full test suite
pytest knowledge_engine/tests/ -k "openevolve_only" -v
```

### Manual Testing

```python
# Test OpenEvolve-only mode
import asyncio
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

async def test_openevolve_only():
    selector = EnsembleStrategySelector(
        knowledge_engine=None,
        enable_loongflow=False
    )

    prediction = await selector.recommend_with_ensemble(
        problem_description="Optimize portfolio",
        domain="finance",
        constraints={}
    )

    print(f"System: {prediction.strategy[0]}")
    print(f"Mode: {prediction.strategy[1]}")
    print(f"Confidence: {prediction.point_estimate:.2%}")

    assert prediction.strategy[0] == "openevolve"

asyncio.run(test_openevolve_only())
```

## Troubleshooting

### Issue: Recommendations use LoongFlow despite disabling

**Solution:**
```python
# Check if actually disabled
selector = EnsembleStrategySelector(enable_loongflow=False)
print(f"LoongFlow available: {selector.is_loongflow_available()}")

# Force override at call time
prediction = await selector.recommend_with_ensemble(
    ...,
    enable_loongflow=False
)
```

### Issue: Low confidence in OpenEvolve-only mode

**Solution:**
```python
# Lower confidence is expected when:
# 1. Cold start (no historical data)
# 2. LoongFlow unavailable (fewer options)
# 3. Complex problem (harder to predict)

# Address by:
# 1. Adding historical data
# 2. Enabling learning (default: enabled)
# 3. Providing more problem context
```

### Issue: Wrong mode selected

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

# If characteristics are wrong, provide better constraints
```

## Future Enhancements

Planned improvements for OpenEvolve-only mode:

1. **Enhanced ML models** - Train specialized models for OpenEvolve-only predictions
2. **Cross-system learning** - Learn from LoongFlow runs when available
3. **Performance baselines** - Establish OpenEvolve performance benchmarks
4. **Auto-tuning** - Adjust ensemble weights based on performance
5. **Mode switching** - Dynamic switching between modes during runs

## References

- [Strategy Selector Documentation](./STRATEGY_RECOMMENDER.md)
- [LoongFlow Integration](./LOONGFLOW_INTEGRATION.md)
- [Ensemble Methods](./ENSEMBLE_METHODS.md)
- [OpenEvolve Documentation](../../README.md)
