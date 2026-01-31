# OpenEvolve Quickstart Guide

Get started with OpenEvolve in 5 minutes! This guide will walk you through the basics and have you evolving code in no time.

## Table of Contents

1. [Installation](#installation)
2. [5-Minute Quickstart](#5-minute-quickstart)
3. [Example Walkthroughs](#example-walkthroughs)
4. [Next Steps](#next-steps)
5. [Common Patterns](#common-patterns)

---

## Installation

### Option 1: Install from source

```bash
cd openevolve
pip install -e .
```

### Option 2: Install with pip

```bash
pip install openevolve
```

### Requirements

- Python 3.8+
- OpenAI API key (or compatible API)

### Set up API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## 5-Minute Quickstart

### Step 1: Create your first program

Create a file called `my_program.py`:

```python
# EVOLVE-BLOCK-START
def solve():
    """Find the best solution"""
    x = 5  # Starting guess
    return x ** 2
# EVOLVE-BLOCK-END
```

### Step 2: Create an evaluator

Create a file called `my_evaluator.py`:

```python
import importlib.util

def evaluate(program_path):
    """Test how good the program is"""
    spec = importlib.util.spec_from_file_location("prog", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.solve()
    score = result / 100.0  # Normalize (max is 100)

    return {"combined_score": score}
```

### Step 3: Run evolution!

```bash
openevolve my_program.py my_evaluator.py -i 10
```

Or using Python:

```python
from openevolve import run_evolution

result = run_evolution('my_program.py', 'my_evaluator.py', iterations=10)
print(f"Best score: {result.best_score}")
print(f"Best code:\n{result.best_code}")
```

That's it! You've just evolved your first program. 🎉

---

## Example Walkthroughs

### Example 1: Basic Evolution (5 min)

**File**: `01_basic_evolution.py`

Simplest way to use OpenEvolve. Maximize a mathematical function.

```bash
# Run with CLI
openevolve 01_basic_evolution.py 01_basic_evolution_evaluator.py -i 10

# Or Python API
python -c "from openevolve import run_evolution; result = run_evolution('01_basic_evolution.py', '01_basic_evolution_evaluator.py', iterations=10); print(f'Score: {result.best_score:.4f}')"
```

**Expected Output**:
- Best score: ~1.0 (or 100.0 depending on normalization)
- Best solution: x = 10
- Time: ~2-3 minutes

**Key Concepts**:
- `EVOLVE-BLOCK-START/END` markers define what to evolve
- Evaluator must return `combined_score` metric
- Evolution explores the solution space automatically

---

### Example 2: Function Evolution (10 min)

**File**: `02_function_evolution.py`

Evolve a slow bubble sort into a faster algorithm.

```bash
openevolve 02_function_evolution.py 02_function_evolution_evaluator.py -i 20
```

**What happens**:
1. Start with bubble sort (O(n²))
2. Evolution tries variations
3. Discovers faster algorithms (quick sort, built-in sorted)
4. Tests both correctness and speed

**Expected Output**:
- Correctness: 100% (6/6 tests pass)
- Speed: Significant improvement
- Best code: Uses built-in `sorted()` or implements quick sort

**Key Concepts**:
- Test multiple aspects (correctness + performance)
- Evolution can discover entirely different approaches
- Trade-offs between simplicity and speed

---

### Example 3: Configuration File (15 min)

**File**: `03_config_file.py`

Use YAML config for reproducible experiments.

**Create config.yaml**:
```yaml
llm:
  api_base: "https://api.openai.com/v1"
  models:
    - name: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.7

max_iterations: 50
database:
  population_size: 100
  num_islands: 3
```

**Run with config**:
```bash
openevolve 03_config_file.py 03_optimize_evaluator.py --config config.yaml
```

**Benefits**:
- Reproducible experiments
- Easy version control
- Share configurations with team
- Environment-specific settings

---

### Example 4: Python API (20 min)

**File**: `04_python_api.py`

Programmatic control over evolution.

```python
from openevolve import run_evolution, evolve_function
from openevolve.config import Config, LLMModelConfig

# Example 1: Simple API call
result = run_evolution('program.py', 'evaluator.py', iterations=10)

# Example 2: Custom config
config = Config()
config.max_iterations = 50
config.llm.models = [
    LLMModelConfig(name='gpt-4', api_key='...')
]
result = run_evolution('program.py', 'evaluator.py', config=config)

# Example 3: Function helper
def my_func(arr):
    return sum(arr)

result = evolve_function(
    my_func,
    test_cases=[([1,2,3], 6), ([4,5], 9)],
    iterations=10
)
```

**Key Concepts**:
- Multiple ways to use OpenEvolve
- Choose based on your use case
- Full control vs convenience

---

### Example 5: CLI Usage (15 min)

**File**: `05_cli_usage.py`

Command-line interface for quick experiments.

```bash
# Basic
openevolve 05_cli_usage.py 05_algo_evaluator.py

# With options
openevolve 05_cli_usage.py 05_algo_evaluator.py \\
    --iterations 100 \\
    --output results \\
    --log-level INFO \\
    --target-score 0.90

# Resume from checkpoint
openevolve program.py evaluator.py \\
    --checkpoint results/checkpoints/checkpoint_50
```

**Output Structure**:
```
results/
├── best/
│   ├── best_program.py
│   └── best_program_info.json
├── checkpoints/
│   ├── checkpoint_10/
│   └── checkpoint_20/
└── logs/
    └── openevolve_20260130_120000.log
```

---

### Example 6: Advanced Features (30 min)

**File**: `06_advanced_features.py`

Checkpoints, early stopping, multi-objective optimization.

```python
from openevolve import run_evolution
from openevolve.config import Config

config = Config()

# Enable checkpoints
config.checkpoint_interval = 10

# Early stopping
config.early_stopping_patience = 15
config.convergence_threshold = 0.001

# Island-based evolution
config.database.num_islands = 5
config.database.migration_interval = 25

# Multi-objective
config.database.feature_dimensions = [
    "complexity", "diversity", "score1", "score2"
]

# Evolution tracing
config.evolution_trace.enabled = True
config.evolution_trace.format = "jsonl"

result = run_evolution(
    '06_advanced_features.py',
    '06_multi_evaluator.py',
    config=config
)
```

**Advanced Features**:
- **Checkpoints**: Save and resume evolution
- **Early Stopping**: Stop when converged
- **Multi-Objective**: Optimize competing goals
- **Island Evolution**: Better exploration
- **Tracing**: Analyze evolution process

---

## Next Steps

### Learn More

1. **Read the docs**: `docs/`
2. **Study examples**: `examples/`
3. **Explore config**: All options in `openevolve/config.py`

### Common Use Cases

#### 1. Algorithm Optimization

```python
# Evolve faster algorithms
result = run_evolution('sort.py', 'sort_evaluator.py')
```

#### 2. Hyperparameter Tuning

```python
# Find optimal ML parameters
result = run_evolution('ml_params.py', 'ml_evaluator.py')
```

#### 3. Code Refactoring

```python
# Improve code quality
result = run_evolution('legacy_code.py', 'quality_evaluator.py')
```

#### 4. Test Generation

```python
# Generate better tests
result = run_evolution('test_gen.py', 'coverage_evaluator.py')
```

### Best Practices

#### 1. Evaluator Design

```python
def evaluate(program_path):
    # Load program
    module = load_program(program_path)

    # Test functionality
    correctness = test_correctness(module)

    # Test performance
    speed = test_speed(module)

    # Test edge cases
    robustness = test_edge_cases(module)

    # Combine metrics
    combined = (
        correctness * 0.5 +
        speed * 0.3 +
        robustness * 0.2
    )

    return {
        "combined_score": combined,  # Required
        "correctness": correctness,
        "speed": speed,
        "robustness": robustness
    }
```

#### 2. Evolution Markers

```python
# EVOLVE-BLOCK-START
def function_to_evolve():
    # This code will be evolved
    pass
# EVOLVE-BLOCK-END

def helper_function():
    # This won't be evolved
    pass
```

#### 3. Config Management

```yaml
# config.yaml
llm:
  models:
    - name: "gpt-4"
      temperature: 0.7

max_iterations: 100
early_stopping_patience: 20

database:
  population_size: 100
  num_islands: 5
```

---

## Common Patterns

### Pattern 1: Test-Driven Evolution

```python
def evaluate(program_path):
    module = load_program(program_path)

    # Run test suite
    passed = run_tests(module)

    # Score based on pass rate
    return {"combined_score": passed / total_tests}
```

### Pattern 2: Performance Optimization

```python
def evaluate(program_path):
    module = load_program(program_path)

    # Measure runtime
    start = time.time()
    result = module.function(large_input)
    duration = time.time() - start

    # Check correctness
    correct = (result == expected)

    # Score: correctness + speed
    score = correct * (1.0 / (1.0 + duration))

    return {"combined_score": score}
```

### Pattern 3: Multi-Objective Optimization

```python
def evaluate(program_path):
    module = load_program(program_path)

    # Multiple objectives
    accuracy = test_accuracy(module)
    speed = test_speed(module)
    memory = test_memory(module)
    simplicity = test_complexity(module)

    # Weighted combination
    combined = (
        accuracy * 0.4 +
        speed * 0.3 +
        memory * 0.2 +
        simplicity * 0.1
    )

    return {
        "combined_score": combined,
        "accuracy": accuracy,
        "speed": speed,
        "memory": memory,
        "simplicity": simplicity
    }
```

---

## Troubleshooting

### Issue: "No LLM models configured"

**Solution**: Set up config with LLM models

```python
from openevolve.config import Config, LLMModelConfig

config = Config()
config.llm.models = [
    LLMModelConfig(
        name='gpt-4',
        api_key='your-api-key'
    )
]

result = run_evolution('program.py', 'evaluator.py', config=config)
```

### Issue: Evolution is slow

**Solutions**:
1. Reduce `max_iterations`
2. Enable `early_stopping`
3. Use smaller `population_size`
4. Enable `parallel_evaluations`

```python
config = Config()
config.max_iterations = 20
config.early_stopping_patience = 10
config.database.population_size = 50
config.evaluator.parallel_evaluations = 4
```

### Issue: Not finding better solutions

**Solutions**:
1. Check evaluator returns proper scores
2. Increase `population_size` for more diversity
3. Increase `num_islands` for better exploration
4. Adjust `temperature` for more variation

```python
config = Config()
config.database.population_size = 200
config.database.num_islands = 5
config.llm.models[0].temperature = 0.9  # More variation
```

### Issue: Program crashes during evolution

**Solutions**:
1. Add error handling in evaluator
2. Use `timeout` in evaluator config
3. Return low score instead of raising exception

```python
def evaluate(program_path):
    try:
        module = load_program(program_path)
        result = module.function()
        return {"combined_score": result}
    except Exception as e:
        # Return low score instead of crashing
        return {"combined_score": 0.0, "error": str(e)}
```

---

## Resources

- **Documentation**: `docs/`
- **Examples**: `examples/`
- **API Reference**: `openevolve/api.py`
- **Config Options**: `openevolve/config.py`
- **GitHub**: [Your repo URL]

---

## Getting Help

1. Check the examples in this directory
2. Read the documentation
3. Search existing issues
4. Ask in discussions/forums

Happy evolving! 🚀
