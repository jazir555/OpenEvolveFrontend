# Gauntlet-ICR-MDAP Integration Guide

**Version:** 1.0  
**Date:** 2026-02-01  
**Scope:** Integration guide for GauntletSystem ICR and AdaptiveMDAPAllocator ICR features

---

## Overview

This guide documents the integration between:

1. **GauntletSystem** - Quality control and validation system
2. **ICR (Iterative Contextual Refinements)** - Continuous improvement framework
3. **AdaptiveMDAPAllocator** - Strategy selection for MDAP/MAKER

The integration enables:
- Automatic refinement triggers based on gauntlet results
- Learning from gauntlet patterns to improve MDAP strategy selection
- Adaptive threshold adjustment based on execution outcomes

---

## Gauntlet-ICR Integration

### Enhanced GauntletSystem

The `GauntletSystem` class now supports ICR integration:

```python
from sovereign_gauntlets import GauntletSystem, RefinementCoordinator

# Create refinement coordinator (optional)
refinement_coordinator = RefinementCoordinator(...)

# Create gauntlet system with ICR
gauntlet_system = GauntletSystem(
    openevolve_client=openevolve_client,
    refinement_coordinator=refinement_coordinator,  # Enable ICR
    track_patterns=True  # Enable pattern learning
)
```

### New Methods

#### run_with_icr_refinement()

Run gauntlets with automatic ICR refinement trigger:

```python
result = gauntlet_system.run_with_icr_refinement(
    plan=decomposition_plan,
    max_refinement_cycles=5,
    refinement_threshold=0.7,
    convergence_threshold=0.01
)

# Result contains:
# {
#     'plan_id': '...',
#     'final_plan_id': '...',
#     'total_cycles': 3,
#     'converged': True,
#     'final_quality': 0.85,
#     'final_results': {...},
#     'refinement_history': [...]
# }
```

#### get_gauntlet_effectiveness()

Get effectiveness metrics for each gauntlet:

```python
effectiveness = gauntlet_system.get_gauntlet_effectiveness()
# {
#     'coherence': {
#         'total_runs': 50,
#         'pass_rate': 0.85,
#         'avg_score': 0.78,
#         'fail_rate': 0.15
#     },
#     ...
# }
```

#### get_failure_patterns()

Get learned failure patterns:

```python
patterns = gauntlet_system.get_failure_patterns()
# {
#     ('coherence', 'completeness'): [
#         {'plan_id': '...', 'overall_quality': 0.5, ...},
#         ...
#     ]
# }
```

#### suggest_optimal_gauntlets()

Get gauntlet recommendations based on plan complexity:

```python
suggestions = gauntlet_system.suggest_optimal_gauntlets(
    plan_type="analysis",
    complexity=0.7  # 0.0 - 1.0
)
# Returns: ['coherence', 'completeness', 'feasibility', 'dependency', 'adaptive', 'hierarchical']
```

#### adapt_gauntlet_config()

Adapt gauntlet configuration based on historical performance:

```python
config = gauntlet_system.adapt_gauntlet_config(
    gauntlet_name='coherence',
    plan_context={'complexity': 0.6}
)
# Returns: {'min_score': 0.75}
```

---

## MDAP-ICR Integration

### Enhanced AdaptiveMDAPAllocator

The `AdaptiveMDAPAllocator` now supports ICR pattern learning:

```python
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator

allocator = AdaptiveMDAPAllocator(
    enable_learning=True,  # Enable ICR pattern learning
    enable_context_aware=True
)
```

### New Methods

#### detect_strategy_patterns()

Detect patterns in strategy effectiveness:

```python
patterns = allocator.detect_strategy_patterns()
# {
#     'has_enough_data': True,
#     'total_samples': 50,
#     'complexity_ranges': {
#         'low': {...},
#         'medium': {...},
#         ...
#     },
#     'strategy_effectiveness': {
#         'DIRECT': {'total_uses': 20, 'success_rate': 0.9, ...},
#         ...
#     },
#     'underperforming': [...],
#     'recommendations': [...]
# }
```

#### adapt_thresholds_from_patterns()

Adapt allocation thresholds based on learned patterns:

```python
new_thresholds, changes = allocator.adapt_thresholds_from_patterns(
    patterns=None,  # Use detected patterns
    target_success_rate=0.85
)
# Returns: ([0.15, 0.35, 0.55, 0.80], ['Lowered t1 threshold...'])
```

#### get_strategy_for_context()

Get comprehensive strategy recommendation with ICR insights:

```python
result = allocator.get_strategy_for_context(
    complexity_score=0.5,
    context=None,
    use_icr_patterns=True
)
# {
#     'complexity_score': 0.5,
#     'recommended_strategy': 'MDAP_MEDIUM',
#     'n_agents': 5,
#     'k_ahead': 1,
#     'reasoning': ['ICR: medium complexity - DIRECT has higher success rate...'],
#     'icr_insights': {
#         'complexity_band': 'medium',
#         'strategy_success_rate': 0.75,
#         'sample_count': 20,
#         'alternative_strategies': [...]
#     }
# }
```

#### record_gauntlet_feedback()

Record gauntlet feedback for MDAP strategy learning:

```python
gauntlet_results = {
    'coherence': {'score': 0.8, 'passed': True},
    'completeness': {'score': 0.6, 'passed': True},
    'feasibility': {'score': 0.4, 'passed': False}
}

allocator.record_gauntlet_feedback(
    complexity_score=0.5,
    strategy=SolveStrategy.MDAP_MEDIUM,
    gauntlet_results=gauntlet_results,
    refinement_applied=True
)
```

---

## Integration Workflow

### End-to-End Example

```python
from sovereign_gauntlets import GauntletSystem, RefinementCoordinator
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator
from sovereign_data_models import DecompositionPlan

# 1. Set up components
refinement_coordinator = RefinementCoordinator(...)
gauntlet_system = GauntletSystem(
    openevolve_client=client,
    refinement_coordinator=refinement_coordinator,
    track_patterns=True
)

mdap_allocator = AdaptiveMDAPAllocator(
    enable_learning=True,
    enable_context_aware=True
)

# 2. Run gauntlets with ICR refinement
plan = DecompositionPlan(...)
result = gauntlet_system.run_with_icr_refinement(
    plan=plan,
    max_refinement_cycles=5,
    refinement_threshold=0.7
)

# 3. Record feedback to MDAP allocator
if result['refinement_history']:
    for cycle in result['refinement_history']:
        if 'gauntlet_results' in cycle:
            mdap_allocator.record_gauntlet_feedback(
                complexity_score=plan.complexity_score,
                strategy=current_strategy,
                gauntlet_results=cycle['gauntlet_results'],
                refinement_applied=cycle.get('refinement_applied', False)
            )

# 4. Adapt thresholds based on learned patterns
if len(mdap_allocator._learning_data) >= 10:
    patterns = mdap_allocator.detect_strategy_patterns()
    new_thresholds, changes = mdap_allocator.adapt_thresholds_from_patterns(patterns)
    print("Threshold changes:", changes)

# 5. Get optimized strategy for future problems
strategy_rec = mdap_allocator.get_strategy_for_context(
    complexity_score=0.6,
    use_icr_patterns=True
)
print(f"Recommended strategy: {strategy_rec['recommended_strategy']}")
```

---

## Pattern Storage

### GauntletSystem Patterns

Patterns are stored by failed gauntlet combination:

```python
# Internal storage
self._gauntlet_patterns: Dict[Tuple[str, ...], List[Dict]]
self._gauntlet_metrics: Dict[str, Dict[str, float]]

# Example:
# _gauntlet_patterns[('coherence', 'completeness')] = [
#     {'plan_id': '...', 'overall_quality': 0.5, 'timestamp': '...'},
#     ...
# ]
```

### MDAP Allocator Patterns

Learning data is stored as a list of outcomes:

```python
self._learning_data: List[Dict]
# Example entry:
# {
#     'complexity_score': 0.5,
#     'strategy': 'MDAP_MEDIUM',
#     'success': True,
#     'cost': 5.0,
#     'quality': 0.85,
#     'timestamp': 1699999999.0
# }
```

---

## Configuration

### GauntletSystem

```yaml
# In your config
gauntlet_system:
  track_patterns: true  # Enable pattern learning
  refinement_threshold: 0.7  # Quality below which to refine
  convergence_threshold: 0.01  # Min improvement to continue
  max_refinement_cycles: 5
```

### AdaptiveMDAPAllocator

```yaml
# In your config
mdap_allocator:
  enable_learning: true  # Enable ICR pattern learning
  enable_context_aware: true  # Use context for allocation
  thresholds: [0.2, 0.4, 0.6, 0.8]  # Default thresholds
```

---

## Clearing Patterns

To reset learned patterns:

```python
# Clear GauntletSystem patterns
gauntlet_system.clear_patterns()

# Clear MDAP allocator learning data (requires new instance)
mdap_allocator = AdaptiveMDAPAllocator(enable_learning=True)
# Or manually clear: mdap_allocator._learning_data.clear()
```

---

## Best Practices

1. **Enable Learning Early**: Start with `enable_learning=True` to collect data from the beginning

2. **Monitor Effectiveness**: Regularly check `get_gauntlet_effectiveness()` to identify problematic gauntlets

3. **Adapt Thresholds**: Run `adapt_thresholds_from_patterns()` periodically (e.g., daily) to optimize allocations

4. **Use RefinementCoordinator**: Connect GauntletSystem to RefinementCoordinator for automatic refinement

5. **Clear Patterns When Needed**: Reset patterns when the problem domain changes significantly

---

## Expected Benefits

| Metric | Expected Improvement |
|--------|---------------------|
| Decomposition Quality | +15-25% |
| Gauntlet Pass Rate | +20-30% |
| MDAP Strategy Accuracy | +25-35% |
| False Positive Rate | -30-40% |

---

## Files Reference

| File | Purpose |
|------|---------|
| `sovereign_gauntlets.py` | GauntletSystem with ICR methods |
| `adaptive_mdap/allocators/resource_allocator.py` | AdaptiveMDAPAllocator with ICR methods |
| `test_icr_gauntlet_mdap_integration.py` | Integration tests |
| `docs/todos/ICR_INTEGRATION_STATUS_REPORT.md` | Status report |

---

**Last Updated:** 2026-02-01  
**Next Review:** 2026-02-08
