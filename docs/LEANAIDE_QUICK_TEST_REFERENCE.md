# Evolutionary LeanAide Test Suite - Quick Reference

## Quick Start

```bash
# Run all tests
python run_evolutionary_tests.py --all

# Or directly with pytest
pytest test_leanaide_evolutionary.py -v
```

## Common Commands

### Run Specific Test Categories

```bash
# Evolution tests
python run_evolutionary_tests.py --evolution
pytest test_leanaide_evolutionary.py -v -m evolution

# Decomposition tests
python run_evolutionary_tests.py --decomposition
pytest test_leanaide_evolutionary.py -v -m decomposition

# Adversarial tests
python run_evolutionary_tests.py --adversarial
pytest test_leanaide_evolutionary.py -v -m adversarial

# Self-play tests
python run_evolutionary_tests.py --selfplay
pytest test_leanaide_evolutionary.py -v -m selfplay

# Strategy tests
python run_evolutionary_tests.py --strategy
pytest test_leanaide_evolutionary.py -v -m strategy

# Workflow tests
python run_evolutionary_tests.py --workflow
pytest test_leanaide_evolutionary.py -v -m workflow
```

### Run by Test Type

```bash
# Unit tests only (fast)
python run_evolutionary_tests.py --unit
pytest test_leanaide_evolutionary.py -v -m unit

# Integration tests
python run_evolutionary_tests.py --integration
pytest test_leanaide_evolutionary.py -v -m integration

# Fast tests only (skip slow ones)
python run_evolutionary_tests.py --fast
pytest test_leanaide_evolutionary.py -v -m "not slow"
```

### Server vs Offline Tests

```bash
# Offline tests (no server required)
python run_evolutionary_tests.py --mock
pytest test_leanaide_evolutionary.py -v -m mock

# Server tests (requires LeanAide server)
python run_evolutionary_tests.py --server
pytest test_leanaide_evolutionary.py -v -m server
```

### Coverage and Performance

```bash
# Generate coverage report
python run_evolutionary_tests.py --coverage
pytest test_leanaide_evolutionary.py --cov=. --cov-report=html

# Run performance benchmarks
python run_evolutionary_tests.py --benchmark
pytest test_leanaide_evolutionary.py -v -m slow
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
python run_evolutionary_tests.py --all --parallel
pytest test_leanaide_evolutionary.py -n auto
```

## Run Specific Tests

```bash
# Run specific test class
pytest test_leanaide_evolutionary.py -v TestLeanProofStrategy

# Run specific test method
pytest test_leanaide_evolutionary.py -v TestLeanProofStrategy::test_strategy_creation

# Run multiple specific tests
pytest test_leanaide_evolutionary.py -v TestLeanProofStrategy::test_strategy_creation TestLeanProofPopulation::test_population_size
```

## Test Output Options

```bash
# Verbose output
pytest test_leanaide_evolutionary.py -v

# Very verbose
pytest test_leanaide_evolutionary.py -vv

# Show print statements
pytest test_leanaide_evolutionary.py -v -s

# Quiet output
pytest test_leanaide_evolutionary.py -q

# Stop on first failure
pytest test_leanaide_evolutionary.py -x

# Stop on N failures
pytest test_leanaide_evolutionary.py --maxfail=3
```

## Test Files

- `test_leanaide_evolutionary.py` - Main test suite
- `run_evolutionary_tests.py` - Test runner script
- `LEANAIDE_EVOLUTIONARY_TEST_SUITE_GUIDE.md` - Full documentation
- `LEANAIDE_QUICK_TEST_REFERENCE.md` - This file

## Pytest Markers

Use markers to filter tests:

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

## Examples

```bash
# Run evolution and adversarial tests
pytest test_leanaide_evolutionary.py -v -m "evolution or adversarial"

# Run unit tests but not slow ones
pytest test_leanaide_evolutionary.py -v -m "unit and not slow"

# Run mock tests only
pytest test_leanaide_evolutionary.py -v -m mock

# Run all tests except server tests
pytest test_leanaide_evolutionary.py -v -m "not server"
```

## Test Runner Options

```bash
python run_evolutionary_tests.py [OPTIONS]

Options:
  --all              Run all tests
  --evolution        Run evolution tests
  --decomposition    Run decomposition tests
  --adversarial      Run adversarial tests
  --selfplay         Run self-play tests
  --strategy         Run strategy tests
  --workflow         Run workflow tests
  --unit             Run unit tests
  --integration      Run integration tests
  --fast             Run fast tests
  --server           Run server tests
  --mock             Run mock tests
  --coverage         Generate coverage report
  --benchmark        Run performance benchmarks
  --quiet            Less verbose output
  --parallel         Run tests in parallel
  --save             Save test results to file
  --help             Show help message
```

## Troubleshooting

### Import errors
```bash
# Check if modules are available
python -c "import leanaide_evolution; print('OK')"
```

### Server tests failing
```bash
# Make sure LeanAide server is running
cd LeanAide
python leanaide_server.py

# Or skip server tests
pytest test_leanaide_evolutionary.py -v -m "not server"
```

### Tests too slow
```bash
# Run fast tests only
pytest test_leanaide_evolutionary.py -v -m "not slow"
```

## CI/CD Integration

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest test_leanaide_evolutionary.py -v -m "not slow"

- name: Generate coverage
  run: pytest test_leanaide_evolutionary.py --cov=. --cov-report=xml
```

## Quick Reference Table

| What | Command |
|------|---------|
| All tests | `python run_evolutionary_tests.py --all` |
| Evolution only | `python run_evolutionary_tests.py --evolution` |
| Unit tests | `python run_evolutionary_tests.py --unit` |
| Fast tests | `python run_evolutionary_tests.py --fast` |
| With coverage | `python run_evolutionary_tests.py --coverage` |
| Parallel | `python run_evolutionary_tests.py --all --parallel` |
| Offline only | `python run_evolutionary_tests.py --mock` |

For more details, see `LEANAIDE_EVOLUTIONARY_TEST_SUITE_GUIDE.md`
