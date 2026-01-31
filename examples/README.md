# OpenEvolve Examples

This directory contains working examples demonstrating how to use OpenEvolve for evolutionary code optimization.

## Quick Start

1. **Start here**: Read [QUICKSTART.md](QUICKSTART.md) for a complete guide
2. **Run examples**: Each example is self-contained and ready to run
3. **Learn patterns**: Study the evaluators to understand how to design your own

## Examples Overview

### Example 1: Basic Evolution
**Files**: `01_basic_evolution.py`, `01_basic_evolution_evaluator.py`

**What it does**: Maximizes a simple mathematical function (f(x) = x²)

**Time**: 5 minutes

**Run it**:
```bash
openevolve 01_basic_evolution.py 01_basic_evolution_evaluator.py -i 10
```

**Learn**:
- Basic OpenEvolve workflow
- How to write an evaluator
- Evolution markers (`EVOLVE-BLOCK-START/END`)

---

### Example 2: Function Evolution
**Files**: `02_function_evolution.py`, `02_function_evolution_evaluator.py`

**What it does**: Evolves a slow bubble sort into a faster sorting algorithm

**Time**: 10 minutes

**Run it**:
```bash
openevolve 02_function_evolution.py 02_function_evolution_evaluator.py -i 20
```

**Learn**:
- Evolving existing functions
- Testing correctness and performance
- Evolution can discover entirely different approaches

---

### Example 3: Configuration File
**Files**: `03_config_file.py`, `03_optimize_evaluator.py`, `config_example.yaml`

**What it does**: 2D optimization using YAML configuration

**Time**: 15 minutes

**Run it**:
```bash
openevolve 03_config_file.py 03_optimize_evaluator.py --config config_example.yaml
```

**Learn**:
- Using YAML configuration files
- Reproducible experiments
- All configuration options

---

### Example 4: Python API
**Files**: `04_python_api.py`, `04_string_evaluator.py`

**What it does**: Demonstrates programmatic usage of OpenEvolve

**Time**: 20 minutes

**Run it**:
```bash
python 04_python_api.py
```

**Learn**:
- Using OpenEvolve as a library
- `run_evolution()`, `evolve_function()`, `evolve_code()`
- Custom configuration in Python
- Accessing detailed results

---

### Example 5: CLI Usage
**Files**: `05_cli_usage.py`, `05_algo_evaluator.py`

**What it does**: Algorithm parameter optimization

**Time**: 15 minutes

**Run it**:
```bash
openevolve 05_cli_usage.py 05_algo_evaluator.py --iterations 50 --output results
```

**Learn**:
- Command-line interface
- All CLI options
- Checkpointing and resumption
- Output structure

---

### Example 6: Advanced Features
**Files**: `06_advanced_features.py`, `06_multi_evaluator.py`

**What it does**: Multi-objective optimization with advanced features

**Time**: 30 minutes

**Run it**:
```bash
openevolve 06_advanced_features.py 06_multi_evaluator.py
```

**Learn**:
- Checkpoints and resumption
- Early stopping
- Multi-objective optimization
- Island-based evolution
- Evolution tracing
- Custom feature dimensions

---

## Configuration

All examples can use the example configuration file:

```bash
openevolve <program>.py <evaluator>.py --config config_example.yaml
```

See `config_example.yaml` for all available options.

## Common Patterns

### Pattern 1: Simple Optimization

```python
# Program
# EVOLVE-BLOCK-START
def solve():
    return optimize_something()
# EVOLVE-BLOCK-END

# Evaluator
def evaluate(program_path):
    module = load_program(program_path)
    result = module.solve()
    return {"combined_score": result}
```

### Pattern 2: Test-Driven Evolution

```python
# Evaluator
def evaluate(program_path):
    module = load_program(program_path)

    passed = 0
    for test_case in test_cases:
        if module.function(test_case.input) == test_case.expected:
            passed += 1

    score = passed / len(test_cases)
    return {"combined_score": score}
```

### Pattern 3: Performance Optimization

```python
# Evaluator
def evaluate(program_path):
    module = load_program(program_path)

    start = time.time()
    result = module.function(large_input)
    duration = time.time() - start

    correct = (result == expected)
    score = correct * (1.0 / (1.0 + duration))

    return {"combined_score": score, "speed": duration}
```

### Pattern 4: Multi-Objective

```python
# Evaluator
def evaluate(program_path):
    module = load_program(program_path)

    accuracy = test_accuracy(module)
    speed = test_speed(module)
    complexity = measure_complexity(program_path)

    combined = (
        accuracy * 0.5 +
        speed * 0.3 +
        complexity * 0.2
    )

    return {
        "combined_score": combined,
        "accuracy": accuracy,
        "speed": speed,
        "complexity": complexity
    }
```

## File Structure

```
examples/
├── QUICKSTART.md              # Complete getting started guide
├── README.md                  # This file
├── config_example.yaml        # Example configuration
│
├── 01_basic_evolution.py              # Example 1: Basic
├── 01_basic_evolution_evaluator.py
│
├── 02_function_evolution.py           # Example 2: Function
├── 02_function_evolution_evaluator.py
│
├── 03_config_file.py                  # Example 3: Config
├── 03_optimize_evaluator.py
│
├── 04_python_api.py                   # Example 4: Python API
├── 04_string_evaluator.py
│
├── 05_cli_usage.py                    # Example 5: CLI
├── 05_algo_evaluator.py
│
├── 06_advanced_features.py            # Example 6: Advanced
└── 06_multi_evaluator.py
```

## Running Examples

### Prerequisites

1. Install OpenEvolve:
   ```bash
   pip install -e ../openevolve
   ```

2. Set API key:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

### CLI Method

```bash
openevolve <program>.py <evaluator>.py [OPTIONS]
```

Options:
- `--config, -c`: Configuration file
- `--iterations, -i`: Max iterations
- `--output, -o`: Output directory
- `--target-score, -t`: Target score for early stopping
- `--log-level, -l`: Logging level
- `--checkpoint`: Resume from checkpoint
- `--api-base`: LLM API base URL
- `--primary-model`: Primary LLM model
- `--secondary-model`: Secondary LLM model

### Python API Method

```python
from openevolve import run_evolution

result = run_evolution(
    'program.py',
    'evaluator.py',
    iterations=10,
    config='config.yaml'
)

print(f"Best score: {result.best_score}")
print(f"Best code:\n{result.best_code}")
```

## Expected Output

Each evolution run creates an output directory:

```
openevolve_output/
├── best/
│   ├── best_program.py          # Best evolved code
│   └── best_program_info.json   # Metrics and metadata
├── checkpoints/
│   ├── checkpoint_10/
│   ├── checkpoint_20/
│   └── ...
└── logs/
    └── openevolve_20260130_120000.log
```

## Tips for Success

### 1. Start Simple

Begin with Example 1 to understand the basics before moving to more complex examples.

### 2. Design Good Evaluators

The evaluator is critical - it guides evolution toward good solutions.

**Good evaluator**:
- Tests what matters (correctness, speed, etc.)
- Returns `combined_score` (0-1 range preferred)
- Handles errors gracefully
- Provides additional metrics for analysis

### 3. Use Appropriate Iterations

- **Learning**: 10-20 iterations
- **Quick tests**: 20-50 iterations
- **Production**: 100+ iterations

Enable early stopping to save time:
```python
config.early_stopping_patience = 20
```

### 4. Leverage Configuration

Use config files for:
- Reproducible experiments
- Team collaboration
- Version control
- Environment-specific settings

### 5. Monitor Progress

Check logs to see evolution progress:
```bash
tail -f openevolve_output/logs/openevolve_*.log
```

## Troubleshooting

### "No LLM models configured"

Set up config with models or use environment variable:
```bash
export OPENAI_API_KEY="your-key"
```

### Evolution is slow

- Reduce `max_iterations`
- Enable `early_stopping`
- Use smaller `population_size`
- Enable `parallel_evaluations`

### Not finding better solutions

- Check evaluator returns proper scores
- Increase `population_size` for more diversity
- Increase `num_islands` for better exploration
- Adjust `temperature` for more variation

### Program crashes

Add error handling in evaluator:
```python
try:
    result = module.function()
    return {"combined_score": result}
except Exception as e:
    return {"combined_score": 0.0, "error": str(e)}
```

## Next Steps

1. ✅ Complete Example 1 (Basic)
2. ✅ Complete Example 2 (Function evolution)
3. ✅ Try Example 3-6 for advanced features
4. ✅ Read [QUICKSTART.md](QUICKSTART.md) for details
5. ✅ Design your own evolution problem

## Additional Resources

- **Documentation**: `../docs/`
- **API Reference**: `../openevolve/api.py`
- **Config Options**: `../openevolve/config.py`
- **Advanced Examples**: `../openevolve/examples/`

## Support

- Check [QUICKSTART.md](QUICKSTART.md) for detailed guide
- Review examples for common patterns
- Read documentation for advanced features
- Check existing issues or ask questions

Happy evolving! 🚀
