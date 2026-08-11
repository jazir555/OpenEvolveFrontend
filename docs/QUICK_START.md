# Quick Start: LoongFlow Gauntlet Integration

## 30-Second Setup

```python
# 1. Import
from enhanced_gauntlet_manager import create_enhanced_gauntlet_system

# 2. Configure
llm_config = {'model': 'claude-3-5-sonnet-20241022', 'api_key': 'sk-...', 'url': 'http://localhost:8001'}
system = create_enhanced_gauntlet_system(llm_config=llm_config)

# 3. Create gauntlet
gauntlet = system.create_enhanced_gauntlet(problem_type="engineering", strictness="standard")

# 4. Execute
execution = await system.execute_gauntlet(
    gauntlet=gauntlet,
    solution=my_solution,
    context={'problem': 'Design a bridge'}
)

# 5. Check result
print(f"Passed: {execution.overall_passed}, Score: {execution.final_score:.2f}")
```

## What You Get

### 3-Round Validation
- **Round 1**: AI quality check (30s) - Fast screening
- **Round 2**: Adversarial testing (2-3min) - Find flaws
- **Round 3**: Consensus approval (3-4min) - Final check

### Strictness Levels
- `strictness="lenient"` - Lower thresholds (0.6, 0.6, 0.75)
- `strictness="standard"` - Medium thresholds (0.7, 0.7, 0.85)
- `strictness="strict"` - Higher thresholds (0.8, 0.75, 0.9)

### Domains Supported
- `"trading"` - Financial algorithms
- `"engineering"` - Design problems
- `"security"` - Security systems
- `"scientific"` - Research validation
- `"finance"` - Financial analysis
- `"general"` - Any problem type

## Files Created

```
Frontend/
├── evaluators/
│   ├── __init__.py
│   └── loongflow_adapter.py           # Main adapter
├── enhanced_gauntlet_manager.py       # Gauntlet system
├── tests/
│   └── test_loongflow_adapter.py      # Test suite
├── examples/
│   └── enhanced_gauntlet_example.py   # Usage examples
└── docs/
    └── loongflow_gauntlet_integration.md  # Full docs
```

## Testing

```bash
# Quick test
python test_integration.py

# Full test suite
pytest tests/test_loongflow_adapter.py -v

# Run examples
python examples/enhanced_gauntlet_example.py
```

## Common Use Cases

### Evaluate Single Solution
```python
from evaluators.loongflow_adapter import create_loongflow_evaluator

adapter = create_loongflow_evaluator(llm_config=llm_config)
result = await adapter.evaluate_round(solution, round_rule, context)
print(f"Score: {result.score:.2f}, Passed: {result.passed}")
```

### Batch Evaluation
```python
results = await adapter.batch_evaluate(
    solutions=[sol1, sol2, sol3],
    round_rule=round_rule,
    context=context
)
for i, r in enumerate(results):
    print(f"Solution {i+1}: {r.score:.2f}")
```

### Custom Gauntlet
```python
gauntlet = system.create_enhanced_gauntlet(
    problem_type="security",
    strictness="strict"
)
# Uses security-specific attack modes: injection, bypass, flood, exploit
```

## Troubleshooting

**Problem**: Import errors
```python
# Solution: Add to path
import sys
sys.path.insert(0, '/path/to/Frontend')
```

**Problem**: All solutions fail
```python
# Solution: Use lenient strictness
gauntlet = system.create_enhanced_gauntlet(
    problem_type="engineering",
    strictness="lenient"  # Lower thresholds
)
```

**Problem**: LoongFlow not available
```python
# Solution: Adapter auto-falls back to basic evaluation
# Check result.details for "evaluation_type": "fallback"
```

## Next Steps

1. Read full docs: `docs/loongflow_gauntlet_integration.md`
2. Run examples: `python examples/enhanced_gauntlet_example.py`
3. Write tests: Follow pattern in `tests/test_loongflow_adapter.py`
4. Integrate real red/gold teams: Replace mock evaluators

## Support

- Full documentation: `docs/loongflow_gauntlet_integration.md`
- Complete report: `LOONGFLOW_INTEGRATION_REPORT.md`
- Test suite: `tests/test_loongflow_adapter.py`
- Examples: `examples/enhanced_gauntlet_example.py`
