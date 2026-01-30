# Evaluator Team Ensemble Integration

## Overview

This document describes the integration of OpenEvolve's ensemble functionality with the Evaluator Team coordination system. The refactoring replaces custom coordination logic with OpenEvolve's proven ensemble approach for parallel execution, load balancing, and result aggregation.

## Architecture

### Before (Custom Coordination)
```
EvaluatorTeamCoordinator
    ↓
ThreadPoolExecutor
    ↓
Custom Load Balancing
    ↓
Custom Parallel Execution
    ↓
Consensus Building (6 algorithms)
    ↓
Quality Gate
```

### After (Ensemble-Based)
```
EvaluatorTeamCoordinator
    ↓
LLMEnsemble (OpenEvolve)
    ├── Weighted Model Selection
    ├── Parallel Generation
    ├── Built-in Retry Logic
    └── Error Handling
    ↓
Consensus Building (6 algorithms preserved)
    ↓
Quality Gate
```

## Key Changes

### 1. Evaluator Team Coordinator (`evaluator_team_coordinator.py`)

#### New Parameters
- `use_ensemble: bool = True` - Enable ensemble-based coordination
- `ensemble_config: Optional[List[LLMModelConfig]]` - Custom ensemble configuration

#### Ensemble Integration
```python
# Initialize ensemble from evaluators
def _initialize_ensemble(self, ensemble_config):
    if not ensemble_config and self.evaluators:
        # Map evaluators to LLM model configs
        ensemble_config = []
        for evaluator in self.evaluators:
            weight = evaluator.expertise_level / 10.0
            model_cfg = LLMModelConfig(
                name=evaluator.evaluator_id,
                weight=weight,
                model_id=evaluator.evaluator_id
            )
            ensemble_config.append(model_cfg)

    if ensemble_config:
        self.ensemble = LLMEnsemble(ensemble_config)
```

#### Parallel Execution
```python
def _execute_evaluation_tasks_parallel(self, session):
    if self.use_ensemble:
        self._execute_tasks_with_ensemble(session)
    else:
        # Fallback to ThreadPoolExecutor
        self._execute_tasks_with_executor(session)
```

### 2. Evaluator Analytics (`evaluator_analytics.py`)

#### New Metrics
Added ensemble-specific tracking to `EvaluatorMetrics`:
- `ensemble_selection_count: int` - Times selected by ensemble
- `ensemble_weight: float` - Current weight in ensemble
- `ensemble_utilization: float` - Utilization rate when available

### 3. Consensus Algorithms (Preserved)

All 6 consensus algorithms remain intact and work with ensemble results:

1. **MAJORITY_VOTE** - Simple majority decision
2. **WEIGHTED_AVERAGE** - Weighted by reliability and expertise
3. **MEDIAN** - Robust to outliers
4. **BATESIAN** - Weighted by historical reliability
5. **DEMPSTER_SHAFER** - Evidence theory aggregation
6. **DELPHI** - Iterative refinement

### 4. Bias Detection (Preserved)

All 7 bias detection types remain functional:
- Leniency bias
- Severity bias
- Central tendency bias
- Halo effect
- Recency bias
- Confirmation bias
- Temporal bias
- Subject matter bias

### 5. Quality Gate (Preserved)

4-stage validation remains unchanged:
- Pre-evaluation checks
- Comprehensive evaluation
- Post-evaluation verification
- Appeal process

## Benefits of Ensemble Integration

### 1. Improved Load Balancing
- **Before**: Custom load balancing strategies
- **After**: Ensemble's weighted sampling automatically balances load based on model weights

### 2. Better Parallel Execution
- **Before**: Manual ThreadPoolExecutor management
- **After**: Ensemble's async parallel execution with built-in error handling

### 3. Enhanced Reliability
- **Before**: Custom retry logic
- **After**: Ensemble's proven retry mechanisms

### 4. Simplified Code
- **Before**: ~500 lines of coordination logic
- **After**: ~50 lines delegating to ensemble

### 5. Better Scalability
- Ensemble designed for high-throughput parallel execution
- Automatic resource management
- Proven in production with OpenEvolve

## Migration Guide

### For Existing Code

```python
# Old way (still works)
coordinator = EvaluatorTeamCoordinator(
    max_concurrent_evaluations=5,
    load_balancing_strategy=LoadBalancingStrategy.SPECIALIZATION_BASED
)

# New way (recommended)
coordinator = EvaluatorTeamCoordinator(
    use_ensemble=True,  # Enable ensemble
    ensemble_config=None,  # Auto-generate from evaluators
    # Other parameters unchanged
    max_concurrent_evaluations=5
)
```

### Custom Ensemble Configuration

```python
from openevolve.config import LLMModelConfig

# Create custom ensemble config
ensemble_config = [
    LLMModelConfig(
        name="gpt-4",
        weight=0.5,
        model_id="gpt-4",
        temperature=0.7,
        max_tokens=4096,
        api_key="sk-...",
        api_base="https://api.openai.com/v1"
    ),
    LLMModelConfig(
        name="claude-3",
        weight=0.3,
        model_id="claude-3-opus",
        temperature=0.7,
        max_tokens=4096
    ),
    LLMModelConfig(
        name="evaluator_1",
        weight=0.2,
        model_id="custom-evaluator-1"
    )
]

coordinator = EvaluatorTeamCoordinator(
    use_ensemble=True,
    ensemble_config=ensemble_config
)
```

## Fallback Behavior

If ensemble is not available or disabled, the coordinator automatically falls back to the original ThreadPoolExecutor implementation:

```python
# Automatically detected
if ENSEMBLE_AVAILABLE:
    use_ensemble = True
else:
    logger.warning("Ensemble not available, using fallback")
    use_ensemble = False
```

## Performance Considerations

### Ensemble Advantages
- **Async execution**: Non-blocking parallel evaluation
- **Weighted sampling**: Efficient model selection
- **Built-in caching**: Reduced redundant computations
- **Resource pooling**: Better resource utilization

### When to Use Ensemble
- High-volume evaluation scenarios (100+ tasks)
- Multiple evaluators with varying expertise
- Need for async/non-blocking execution
- Complex dependency graphs

### When to Use Fallback
- Simple scenarios (1-10 tasks)
- Single evaluator
- Synchronous execution required
- Testing and development

## Testing

### Unit Tests
```python
def test_ensemble_initialization():
    coordinator = EvaluatorTeamCoordinator(use_ensemble=True)
    assert coordinator.use_ensemble == True
    assert hasattr(coordinator, 'ensemble')

def test_fallback_when_no_ensemble():
    coordinator = EvaluatorTeamCoordinator(use_ensemble=False)
    assert coordinator.use_ensemble == False
    assert hasattr(coordinator, 'executor')

def test_ensemble_execution():
    coordinator = EvaluatorTeamCoordinator(use_ensemble=True)
    session = coordinator.coordinate_solution_evaluations(...)
    assert session.completed_tasks > 0
```

### Integration Tests
```python
def test_end_to_end_with_ensemble():
    # Create coordinator with ensemble
    coordinator = EvaluatorTeamCoordinator(use_ensemble=True)

    # Evaluate solutions
    session = coordinator.coordinate_solution_evaluations(
        problem_statement="Test problem",
        sub_problems=[...],
        solutions={...}
    )

    # Verify results
    assert session.completed_tasks == len(sub_problems)
    assert all(t.consensus_reached for t in session.tasks)
```

## Monitoring and Analytics

### New Metrics
The ensemble integration adds new metrics to track:

```python
# In analytics
metrics.ensemble_selection_count  # How often evaluator is selected
metrics.ensemble_weight  # Current weight in ensemble
metrics.ensemble_utilization  # Utilization rate
```

### Reporting
Reports automatically include ensemble metrics:

```python
report = analytics.generate_quality_report(evaluator_id)
print(report['ensemble_selection_count'])
print(report['ensemble_utilization'])
```

## Troubleshooting

### Issue: Ensemble not available
```
"OpenEvolve Ensemble not available - using fallback coordination"
```
**Solution**: Install openevolve package or disable ensemble
```python
coordinator = EvaluatorTeamCoordinator(use_ensemble=False)
```

### Issue: Weighted sampling not working
**Solution**: Ensure ensemble configs have valid weights
```python
# Weights must sum to 1.0 (automatically normalized)
ensemble_config = [
    LLMModelConfig(name="model1", weight=0.5),
    LLMModelConfig(name="model2", weight=0.5)
]
```

### Issue: Async execution errors
**Solution**: Check event loop initialization
```python
# Should be automatic, but can manually create loop
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
```

## Best Practices

1. **Start with ensemble enabled** - It's the recommended default
2. **Use custom weights** - Set weights based on evaluator expertise
3. **Monitor utilization** - Track ensemble_selection_count to balance load
4. **Test with fallback** - Ensure code works with use_ensemble=False
5. **Profile performance** - Compare ensemble vs fallback for your workload

## Future Enhancements

Potential improvements to ensemble integration:

1. **Dynamic weight adjustment** - Auto-adjust weights based on performance
2. **Ensemble aggregation** - Use ensemble's aggregation for consensus
3. **Multi-ensemble support** - Different ensembles for different tasks
4. **A/B testing** - Compare ensemble strategies
5. **Metrics dashboard** - Real-time ensemble performance visualization

## References

- OpenEvolve Ensemble: `openevolve/llm/ensemble.py`
- Evaluator Team: `evaluator_team.py`
- Coordinator: `evaluator_team_coordinator.py`
- Analytics: `evaluator_analytics.py`
- Quality Gate: `quality_gate_engine.py`

## Changelog

### Version 2.0.0 (Current)
- **Added**: Ensemble-based coordination
- **Added**: Ensemble metrics tracking
- **Improved**: Parallel execution performance
- **Maintained**: All 6 consensus algorithms
- **Maintained**: All bias detection types
- **Maintained**: Quality gate functionality

### Version 1.0.0 (Previous)
- Custom ThreadPoolExecutor coordination
- Manual load balancing
- Synchronous execution

## Support

For questions or issues:
1. Check this documentation
2. Review test files for examples
3. Consult OpenEvolve documentation
4. Create issue in repository
