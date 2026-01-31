# OpenEvolve Examples Index

> **Quick Links**: [QUICKSTART.md](QUICKSTART.md) | [README.md](README.md) | [Test Examples](test_examples.py)

---

## 📚 Complete Example Library

Welcome to the OpenEvolve examples! This collection provides everything you need to go from zero to evolutionary optimization expert.

## 🚀 Quick Start (5 Minutes)

**New to OpenEvolve?** Start here:

1. Read [QUICKSTART.md](QUICKSTART.md) - Complete getting started guide
2. Run Example 1 below - Your first evolution
3. Explore other examples - Learn advanced features

## 📖 Examples

### 🔰 Example 1: Basic Evolution
**Time**: 5 minutes | **Difficulty**: Beginner

**Files**:
- [`01_basic_evolution.py`](01_basic_evolution.py) - Maximize f(x) = x²
- [`01_basic_evolution_evaluator.py`](01_basic_evolution_evaluator.py) - Simple evaluator

**What you'll learn**:
- Basic OpenEvolve workflow
- Evolution markers (`EVOLVE-BLOCK-START/END`)
- Writing simple evaluators
- Understanding `combined_score`

**Run it**:
```bash
openevolve 01_basic_evolution.py 01_basic_evolution_evaluator.py -i 10
```

---

### 🔄 Example 2: Function Evolution
**Time**: 10 minutes | **Difficulty**: Beginner

**Files**:
- [`02_function_evolution.py`](02_function_evolution.py) - Slow bubble sort
- [`02_function_evolution_evaluator.py`](02_function_evolution_evaluator.py) - Test correctness + speed

**What you'll learn**:
- Evolving existing functions
- Testing multiple aspects (correctness + performance)
- Evolution can discover entirely different approaches
- Performance optimization

**Run it**:
```bash
openevolve 02_function_evolution.py 02_function_evolution_evaluator.py -i 20
```

**Expected**: Evolution discovers faster sorting algorithms

---

### ⚙️ Example 3: Configuration File
**Time**: 15 minutes | **Difficulty**: Intermediate

**Files**:
- [`03_config_file.py`](03_config_file.py) - 2D optimization
- [`03_optimize_evaluator.py`](03_optimize_evaluator.py) - Multi-point evaluation
- [`config_example.yaml`](config_example.yaml) - Full config reference

**What you'll learn**:
- Using YAML configuration files
- Reproducible experiments
- All configuration options
- Version control for settings

**Run it**:
```bash
openevolve 03_config_file.py 03_optimize_evaluator.py --config config_example.yaml
```

---

### 🐍 Example 4: Python API
**Time**: 20 minutes | **Difficulty**: Intermediate

**Files**:
- [`04_python_api.py`](04_python_api.py) - Multiple API examples
- [`04_string_evaluator.py`](04_string_evaluator.py) - String processing tests

**What you'll learn**:
- Programmatic usage with `run_evolution()`
- Using `evolve_function()` helper
- Custom configuration in Python
- Accessing detailed results
- Multiple usage patterns

**Run it**:
```bash
python 04_python_api.py
```

**Contains**: 5 different API usage examples

---

### 💻 Example 5: CLI Usage
**Time**: 15 minutes | **Difficulty**: Intermediate

**Files**:
- [`05_cli_usage.py`](05_cli_usage.py) - Algorithm parameters
- [`05_algo_evaluator.py`](05_algo_evaluator.py) - Parameter optimization

**What you'll learn**:
- Command-line interface
- All CLI options and flags
- Checkpointing and resumption
- Output directory structure
- Log files

**Run it**:
```bash
openevolve 05_cli_usage.py 05_algo_evaluator.py --iterations 50 --output results
```

**Features**: Checkpointing, logging, output structure

---

### 🚀 Example 6: Advanced Features
**Time**: 30 minutes | **Difficulty**: Advanced

**Files**:
- [`06_advanced_features.py`](06_advanced_features.py) - Multi-objective optimization
- [`06_multi_evaluator.py`](06_multi_evaluator.py) - Competing objectives

**What you'll learn**:
- Checkpoints and resumption
- Early stopping based on convergence
- Multi-objective optimization
- Island-based evolution
- Evolution tracing and analysis
- Custom feature dimensions

**Run it**:
```bash
openevolve 06_advanced_features.py 06_multi_evaluator.py
```

**Advanced**: 8 different advanced features demonstrated

---

## 📚 Documentation

### [README.md](README.md)
**Quick reference guide**
- Examples overview
- Common patterns
- File structure
- Troubleshooting

### [QUICKSTART.md](QUICKSTART.md)
**Complete getting started guide**
- 5-minute tutorial
- Step-by-step walkthroughs
- Best practices
- Common patterns
- Troubleshooting

### [config_example.yaml](config_example.yaml)
**Full configuration reference**
- All options documented
- Comments explaining each setting
- Usage examples
- Default values

### [EXAMPLES_SUMMARY.md](EXAMPLES_SUMMARY.md)
**Project summary**
- What was created
- File listing
- Success criteria
- Testing results

---

## 🧪 Testing

### [test_examples.py](test_examples.py)
Validate all examples:

```bash
python test_examples.py
```

**Tests**:
- ✅ Syntax validation
- ✅ Structure checks
- ✅ Evolution markers
- ✅ Evaluator requirements

**Result**: All 15 tests pass ✓

---

## 🎯 Learning Path

### Level 1: Beginner (30 min)
1. Read QUICKSTART.md (5 min)
2. Run Example 1 (5 min)
3. Run Example 2 (10 min)
4. Review code (10 min)

**Skills**: Basic workflow, simple evaluators

### Level 2: Intermediate (45 min)
1. Run Example 3 (15 min)
2. Run Example 4 (20 min)
3. Experiment with config (10 min)

**Skills**: YAML config, Python API, custom settings

### Level 3: Advanced (60 min)
1. Run Example 5 (15 min)
2. Run Example 6 (30 min)
3. Explore features (15 min)

**Skills**: CLI, advanced features, multi-objective

---

## 🛠️ Quick Reference

### Run an example
```bash
openevolve <program>.py <evaluator>.py -i 10
```

### Use config file
```bash
openevolve <program>.py <evaluator>.py --config config.yaml
```

### Python API
```python
from openevolve import run_evolution
result = run_evolution('program.py', 'evaluator.py', iterations=10)
print(f"Score: {result.best_score}")
```

### Resume from checkpoint
```bash
openevolve program.py evaluator.py --checkpoint output/checkpoints/checkpoint_50
```

---

## 📊 Example Comparison

| Example | Time | Difficulty | Focus | Key Features |
|---------|------|------------|-------|--------------|
| 1. Basic | 5 min | Beginner | Simplest case | Basic workflow |
| 2. Function | 10 min | Beginner | Algorithm evolution | Correctness + speed |
| 3. Config | 15 min | Intermediate | YAML configuration | Reproducible experiments |
| 4. Python API | 20 min | Intermediate | Programmatic usage | Multiple API methods |
| 5. CLI | 15 min | Intermediate | Command-line | All CLI options |
| 6. Advanced | 30 min | Advanced | Multi-objective | 8 advanced features |

---

## ✅ Success Checklist

- [ ] Read QUICKSTART.md
- [ ] Run Example 1 successfully
- [ ] Understand evolution markers
- [ ] Write your own evaluator
- [ ] Try Examples 2-4
- [ ] Experiment with configuration
- [ ] Explore advanced features
- [ ] Design custom evolution problem

---

## 🎓 Common Patterns

### Pattern 1: Simple Optimization
```python
def evaluate(program_path):
    module = load_program(program_path)
    result = module.function()
    return {"combined_score": result}
```

### Pattern 2: Test-Driven Evolution
```python
def evaluate(program_path):
    module = load_program(program_path)
    passed = sum(1 for t in tests if module.function(t.input) == t.expected)
    return {"combined_score": passed / len(tests)}
```

### Pattern 3: Performance Optimization
```python
def evaluate(program_path):
    module = load_program(program_path)
    start = time.time()
    result = module.function(large_input)
    duration = time.time() - start
    correct = (result == expected)
    score = correct * (1.0 / (1.0 + duration))
    return {"combined_score": score}
```

---

## 🤝 Support

- **Quick help**: Check QUICKSTART.md
- **Details**: See README.md
- **Config**: Reference config_example.yaml
- **Validation**: Run test_examples.py
- **Issues**: Check existing issues or ask

---

## 📈 Next Steps

1. ✅ **Start**: Run Example 1 (5 minutes)
2. ✅ **Learn**: Read QUICKSTART.md
3. ✅ **Explore**: Try Examples 2-4
4. ✅ **Master**: Use advanced features
5. ✅ **Create**: Build your own evolution

---

## 🎉 Summary

**Total Examples**: 6
**Total Files**: 16 (6 programs + 6 evaluators + 4 docs)
**All Tested**: ✓ 15/15 tests pass
**Documentation**: Complete (QUICKSTART, README, config)
**Ready to use**: Yes!

**From zero to evolution in 5 minutes!** 🚀

---

**Start here**: [QUICKSTART.md](QUICKSTART.md)
