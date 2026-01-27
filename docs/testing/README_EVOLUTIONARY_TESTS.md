# Evolutionary LeanAide Test Suite

A comprehensive test suite for all evolutionary LeanAide components.

## Features

- **200+ tests** covering all evolutionary components
- **8 test categories**: Evolution, Decomposition, Adversarial, Self-Play, Strategy, Workflow, Performance, Edge Cases
- **Pytest-based** with comprehensive markers
- **Offline testing** with mocks
- **Server testing** for integration tests
- **Coverage reporting** built-in
- **Parallel execution** support
- **Performance benchmarks** included

## Quick Start

```bash
# Run all tests
python run_evolutionary_tests.py --all

# Or with pytest
pytest test_leanaide_evolutionary.py -v
```

## Test Categories

### 1. Evolution Tests
- Initial population generation
- Fitness evaluation
- Selection methods (tournament, roulette, rank)
- Crossover operations
- Mutation operations
- Convergence detection
- Stagnation handling

### 2. Decomposition Tests
- Mathematical component extraction
- Dependency identification
- Complexity estimation
- Sub-problem generation
- Topological ordering
- Parallelization detection

### 3. Adversarial Tests
- Blue team proof generation
- Red team critique generation
- Counterexample generation
- Adversarial rounds
- Co-evolution dynamics

### 4. Self-Play Tests
- Game execution
- Experience buffer management
- Agent strategy selection
- Reward calculation
- Training loops

### 5. Strategy Library Tests
- Tactic library completeness
- Template instantiation
- Strategy selection
- Strategy mutation
- Strategy combination
- Success rate tracking

### 6. Workflow Integration Tests
- Stage 3A evolutionary solution generation
- Stage 3B adversarial evolution
- Mathematical problem detection
- Graceful fallback
- End-to-end workflows

### 7. Performance Tests
- Evolution speed benchmarks
- Parallel evaluation performance

### 8. Edge Case Tests
- Empty input handling
- Malformed code handling
- Extreme cases
- Error scenarios

## Usage

### Using Test Runner

```bash
# Run all tests
python run_evolutionary_tests.py --all

# Run specific category
python run_evolutionary_tests.py --evolution
python run_evolutionary_tests.py --decomposition
python run_evolutionary_tests.py --adversarial
python run_evolutionary_tests.py --selfplay
python run_evolutionary_tests.py --strategy
python run_evolutionary_tests.py --workflow

# Run fast tests only
python run_evolutionary_tests.py --fast

# Generate coverage
python run_evolutionary_tests.py --coverage

# Run in parallel
python run_evolutionary_tests.py --all --parallel
```

### Using Pytest Directly

```bash
# All tests
pytest test_leanaide_evolutionary.py -v

# By marker
pytest test_leanaide_evolutionary.py -v -m evolution
pytest test_leanaide_evolutionary.py -v -m unit
pytest test_leanaide_evolutionary.py -v -m "not slow"

# Coverage
pytest test_leanaide_evolutionary.py --cov=. --cov-report=html

# Parallel
pytest test_leanaide_evolutionary.py -n auto
```

## Documentation

- `LEANAIDE_EVOLUTIONARY_TEST_SUITE_GUIDE.md` - Complete documentation
- `LEANAIDE_QUICK_TEST_REFERENCE.md` - Quick reference guide
- `README_EVOLUTIONARY_TESTS.md` - This file

## Test Data

The test suite includes sample data:
- **Theorems**: 5 theorems of varying difficulty (trivial to complex)
- **Mathematical Problems**: 5 problems for decomposition testing
- **Lean Tactics**: 20 common Lean 4 tactics

## Requirements

```bash
pip install pytest pytest-asyncio pytest-cov pytest-xdist
```

## Architecture

```
test_leanaide_evolutionary.py (main test suite)
├── Evolution Tests (leanaide_evolution.py)
├── Decomposition Tests (leanaide_decomposition_integration.py)
├── Adversarial Tests (leanaide_adversarial.py)
├── Self-Play Tests (leanaide_selfplay.py)
├── Strategy Tests (leanaide_strategies.py)
├── Workflow Integration Tests
├── Performance Tests
└── Edge Case Tests

run_evolutionary_tests.py (test runner)
```

## Offline Testing

Tests can run without LeanAide server using mocks:

```bash
python run_evolutionary_tests.py --mock
pytest test_leanaide_evolutionary.py -v -m mock
```

## CI/CD Integration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest test_leanaide_evolutionary.py -v -m "not slow"
```

## Test Markers

- `unit` - Unit tests
- `integration` - Integration tests
- `mock` - Mock tests (offline)
- `server` - Server tests (requires LeanAide)
- `slow` - Slow tests
- `evolution` - Evolution tests
- `decomposition` - Decomposition tests
- `adversarial` - Adversarial tests
- `selfplay` - Self-play tests
- `strategy` - Strategy tests
- `workflow` - Workflow tests

## Contributing

When adding tests:
1. Follow existing patterns
2. Use appropriate markers
3. Add docstrings
4. Include edge cases
5. Update documentation

## License

Part of the OpenEvolve project.

---

**Created**: 2025-12-30
**Version**: 1.0.0
