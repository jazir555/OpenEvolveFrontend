# LoongFlow Gauntlet Integration

## Overview

This integration adds LoongFlow's AI evaluation capabilities to the OpenEvolve gauntlet system, enabling fast, automated quality screening as Round 1 of the 3-round validation process.

## Architecture

The enhanced gauntlet system implements a 3-round validation pipeline:

### Round 1: LoongFlow AI Evaluation (Quick Screen)
- **Purpose**: Fast, automated quality assessment
- **Duration**: 10-30 seconds
- **Evaluator**: LoongFlow `GeneralEvaluator` with AI agent
- **Output**: Score (0-1+), feedback, suggestions
- **Fallback**: Keyword/pattern-based evaluation if LoongFlow unavailable

### Round 2: Red Team Attack (Adversarial)
- **Purpose**: Find flaws and edge cases
- **Duration**: 1-3 minutes
- **Evaluator**: Red team agents with adversarial prompts
- **Output**: Identified vulnerabilities, severity scores
- **Attack Modes**: Domain-specific (market_crash, injection, etc.)

### Round 3: Gold Team Verification (Consensus)
- **Purpose**: Final quality assurance
- **Duration**: 2-4 minutes
- **Evaluator**: Gold team agents with voting
- **Output**: Consensus approval, detailed metrics
- **Voting**: First-to-ahead-by-k or majority consensus

## Installation

### Prerequisites

1. **LoongFlow** (optional but recommended):
   ```bash
   # Clone LoongFlow
   git clone https://github.com/your-org/LoongFlow.git
   ```

2. **Dependencies**:
   ```bash
   pip install openevolve-structures
   pip install asyncio
   ```

### Setup

1. Place files in the OpenEvolve frontend:
   ```
   Frontend/
   ├── evaluators/
   │   ├── __init__.py
   │   └── loongflow_adapter.py
   ├── enhanced_gauntlet_manager.py
   ├── tests/
   │   └── test_loongflow_adapter.py
   └── examples/
       └── enhanced_gauntlet_example.py
   ```

2. Configure environment:
   ```bash
   export LLM_API_KEY="your-api-key"
   export LLM_API_URL="http://localhost:8001"
   export LLM_MODEL="claude-3-5-sonnet-20241022"
   ```

## Usage

### Basic Evaluation

```python
from evaluators.loongflow_adapter import create_loongflow_evaluator

# Configure adapter
llm_config = {
    'model': 'claude-3-5-sonnet-20241022',
    'api_key': 'sk-...',
    'url': 'http://localhost:8001'
}

adapter = create_loongflow_evaluator(
    llm_config=llm_config,
    timeout=60
)

# Evaluate solution
result = await adapter.evaluate_round(
    solution=solution_attempt,
    round_rule=gauntlet_round,
    context={
        'problem': 'Solve X',
        'criteria': ['correctness', 'clarity']
    }
)

print(f"Score: {result.score}")
print(f"Passed: {result.passed}")
print(f"Feedback: {result.feedback}")
```

### Complete Gauntlet

```python
from enhanced_gauntlet_manager import create_enhanced_gauntlet_system

# Create system
system = create_enhanced_gauntlet_system(
    llm_config=llm_config,
    enable_loongflow=True
)

# Create gauntlet
gauntlet = system.create_enhanced_gauntlet(
    problem_type="engineering",
    strictness="standard"
)

# Execute gauntlet
execution = await system.execute_gauntlet(
    gauntlet=gauntlet,
    solution=my_solution,
    context={'problem': 'Design a bridge'}
)

print(f"Overall Passed: {execution.overall_passed}")
print(f"Final Score: {execution.final_score}")
```

## Configuration

### Strictness Levels

Three strictness levels control score thresholds:

| Level  | Round 1 | Round 2 | Round 3 |
|--------|---------|---------|---------|
| Lenient| 0.60    | 0.60    | 0.75    |
| Standard| 0.70   | 0.70    | 0.85    |
| Strict | 0.80    | 0.75    | 0.90    |

### Problem Domains

Supported domains with specialized attack modes:

- **trading**: market_crash, regime_change, black_swan
- **engineering**: overload, fatigue, extreme_conditions
- **security**: injection, bypass, flood, exploit
- **scientific**: outlier, noise, confounding, bias
- **finance**: volatility_spike, correlation_breakdown, tail_risk

### LLM Configuration

```python
llm_config = {
    'model': 'claude-3-5-sonnet-20241022',  # Model ID
    'api_key': 'sk-...',                     # API key
    'url': 'http://localhost:8001',          # API endpoint
    'temperature': 0.3,                      # Temperature
    'max_tokens': 4096,                      # Max tokens
    'timeout': 60,                           # Request timeout
    'agent_config': {                        # Agent-specific config
        'max_turns': 10,
        'skills': ['python', 'testing']
    }
}
```

## API Reference

### LoongFlowEvaluatorAdapter

Main adapter class for LoongFlow evaluation.

#### Methods

**`__init__(llm_config, timeout=60, enable_loongflow=True)`**
- Initialize adapter
- `llm_config`: LLM configuration dict
- `timeout`: Evaluation timeout in seconds
- `enable_loongflow`: If False, use fallback only

**`async evaluate_round(solution, round_rule, context) -> GauntletRoundResult`**
- Evaluate single solution
- Returns: GauntletRoundResult with score, feedback, passed/failed

**`async batch_evaluate(solutions, round_rule, context) -> List[GauntletRoundResult]`**
- Evaluate multiple solutions in parallel
- Returns: List of results (same order as input)

### GauntletRoundResult

Result from single round evaluation.

#### Attributes

- `rule_id`: ID of the round rule
- `passed`: Whether solution passed (bool)
- `score`: Score achieved (0.0-1.0+)
- `feedback`: Human-readable feedback (str)
- `details`: Additional evaluation details (dict)
- `execution_time`: Time in seconds (float)
- `timestamp`: Evaluation timestamp (float)

### EnhancedGauntletSystem

Enhanced gauntlet system with LoongFlow integration.

#### Methods

**`__init__(llm_config, enable_loongflow=True, red_team_evaluator=None, gold_team_evaluator=None)`**
- Initialize system
- Optionally provide custom red/gold team evaluators

**`create_enhanced_gauntlet(problem_type, strictness='standard') -> GauntletDefinition`**
- Create 3-round gauntlet
- `problem_type`: Domain (trading, engineering, etc.)
- `strictness`: lenient, standard, or strict

**`async execute_gauntlet(gauntlet, solution, context) -> GauntletExecution`**
- Execute complete gauntlet
- Returns: GauntletExecution with all round results

### GauntletExecution

Complete gauntlet execution result.

#### Attributes

- `gauntlet_id`: ID of gauntlet
- `solution_id`: ID of solution
- `rounds_results`: List of GauntletRoundResult
- `rounds_passed`: List of passed round IDs
- `rounds_failed`: List of failed round IDs
- `final_score`: Final average score (float)
- `overall_passed`: Whether gauntlet passed (bool)
- `execution_time`: Total time in seconds (float)

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/test_loongflow_adapter.py -v

# Run specific test
pytest tests/test_loongflow_adapter.py::TestLoongFlowAdapter::test_fallback_evaluation -v

# Run with coverage
pytest tests/test_loongflow_adapter.py --cov=evaluators --cov=enhanced_gauntlet_manager
```

### Test Coverage

Tests cover:
- Adapter initialization
- Solution content extraction
- Fallback evaluation mode
- Batch evaluation
- Parallel execution
- Gauntlet creation
- Gauntlet execution
- Strictness levels
- Domain-specific attack modes

## Examples

See `examples/enhanced_gauntlet_example.py` for complete examples:

1. **Example 1**: Basic LoongFlow evaluation
2. **Example 2**: Batch evaluation
3. **Example 3**: Complete 3-round gauntlet
4. **Example 4**: Strictness level comparison
5. **Example 5**: Domain-specific gauntlets

Run examples:
```bash
python examples/enhanced_gauntlet_example.py
```

## Performance

### Typical Execution Times

- Round 1 (LoongFlow): 10-30 seconds
- Round 2 (Red Team): 60-180 seconds
- Round 3 (Gold Team): 120-240 seconds
- **Total**: 3-8 minutes for complete gauntlet

### Benefits

1. **Fast Failure**: Round 1 screens out low-quality solutions quickly
2. **Resource Efficiency**: Skip expensive rounds if Round 1 fails
3. **Consistent Scoring**: AI-based evaluation is more consistent than human-only
4. **Scalability**: Batch evaluation supports parallel processing
5. **Fallback**: Works even without LoongFlow installed

## Troubleshooting

### LoongFlow Not Available

If LoongFlow is not installed, the adapter automatically falls back to simple keyword-based evaluation.

**Check**: Look for `"evaluation_type": "fallback"` in result details.

**Solution**: Install LoongFlow or configure `enable_loongflow=False`.

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'loongflow'`

**Solution**: Add LoongFlow to path or install:
```python
import sys
sys.path.insert(0, '/path/to/LoongFlow')
```

### Timeout Errors

**Error**: `Evaluation timed out after 60s`

**Solution**: Increase timeout:
```python
adapter = create_loongflow_evaluator(
    llm_config=llm_config,
    timeout=120  # Increase to 2 minutes
)
```

### Low Scores

**Issue**: All solutions score low

**Solution**: Adjust thresholds:
```python
gauntlet = system.create_enhanced_gauntlet(
    problem_type="engineering",
    strictness="lenient"  # Lower thresholds
)
```

## Contributing

To extend the integration:

1. Add new evaluators in `evaluators/` directory
2. Update `enhanced_gauntlet_manager.py` to route to new evaluators
3. Add tests in `tests/test_loongflow_adapter.py`
4. Update examples and documentation

## License

This integration follows the same license as OpenEvolve.

## References

- [LoongFlow Documentation](https://github.com/your-org/LoongFlow)
- [OpenEvolve Documentation](https://github.com/your-org/OpenEvolve)
- [Gauntlet System Design](./gauntlet_system_design.md)
