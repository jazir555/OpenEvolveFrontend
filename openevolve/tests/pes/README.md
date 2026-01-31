# PES Test Suite

Comprehensive test suite for OpenEvolve's Prompt Evolution Strategy (PES) system.

## Overview

This test suite provides thorough testing of the OpenEvolve evolutionary system, including:
- Unit tests for all core components
- Integration tests for end-to-end evolution
- Performance benchmarks
- Mock infrastructure for testing without API calls

## Directory Structure

```
tests/pes/
├── __init__.py                    # Package initialization
├── fixtures.py                    # Reusable test fixtures
├── test_controller.py             # Controller unit tests
├── test_evaluator.py              # Evaluator unit tests
├── test_database.py               # Database unit tests
├── integration/
│   ├── __init__.py
│   └── test_pes_optimization.py   # Integration tests
├── run_tests.py                   # Test runner script
└── README.md                      # This file
```

## Quick Start

### Run All Tests

```bash
cd tests/pes
python run_tests.py
```

### Run with Coverage

```bash
python run_tests.py --coverage
```

### Run Only Unit Tests

```bash
python run_tests.py --unit
```

### Run Only Integration Tests

```bash
python run_tests.py --integration
```

### Run Specific Test Pattern

```bash
python run_tests.py -k "test_database"
```

### Run with Verbose Output

```bash
python run_tests.py --verbose
```

## Test Categories

### Unit Tests

#### `test_controller.py`
Tests the main `OpenEvolve` controller class:
- Initialization with various configurations
- File loading and validation
- Component initialization (database, evaluator)
- Error handling
- Configuration management

#### `test_evaluator.py`
Tests the program evaluation system:
- Program execution and evaluation
- Timeout handling
- Error recovery
- Multiple evaluations for stability
- Performance measurement
- Edge case handling (invalid output, NaN, empty output)

#### `test_database.py`
Tests the program database:
- Program storage and retrieval
- Evolutionary tree tracking
- Parent-child relationships
- Ancestry queries
- Generation statistics
- Persistence and loading
- Deletion and cleanup

### Integration Tests

#### `integration/test_pes_optimization.py`
End-to-end evolution tests:
- Simple optimization problems
- Convergence detection
- Multi-objective optimization
- Evolution over generations
- Error recovery
- Performance benchmarks
- Various configuration scenarios

## Test Fixtures

The `fixtures.py` file provides reusable test fixtures:

- `temp_dir`: Temporary directory for test files
- `sample_config`: Sample configuration dictionary
- `simple_optimization_problem`: Basic optimization problem definition
- `sample_program_code`: Sample Python code
- `sample_evaluation_script`: Sample evaluator script
- `sample_initial_program`: Sample initial program
- `mock_llm_client`: Mock LLM for testing without API calls
- `mock_llm_ensemble`: Mock ensemble of LLMs
- `evolutionary_test_data`: Sample evolutionary data
- `sample_trace_data`: Sample evolution trace

## Mock LLM Client

Located in `openevolve/llm/mocks/mock_client.py`, the mock LLM client provides:

### Features

- **No API Calls**: Test without expensive LLM API calls
- **Realistic Responses**: Returns sensible mock responses based on prompt content
- **Configurable**: Supports different response qualities ("good", "medium", "poor")
- **Error Simulation**: Can simulate random errors for testing error handling
- **Latency Simulation**: Can simulate API latency

### Usage

```python
from openevolve.llm.mocks import MockLLMClient

# Create mock client
client = MockLLMClient(
    model_name="mock-model",
    latency_ms=0,
    error_rate=0.0,
    response_quality="good"
)

# Generate response
response = await client.generate("Improve this function")
print(response.content)
```

## Coverage Goals

Target test coverage:
- **Overall**: 80%+
- **Core modules**: 90%+
- **Integration coverage**: 70%+

Current coverage can be checked by running:
```bash
python run_tests.py --coverage
```

## Writing New Tests

### 1. Create Test File

```bash
touch tests/pes/test_your_component.py
```

### 2. Import Dependencies

```python
import pytest
from pathlib import Path
from openevolve.your_module import YourClass
```

### 3. Use Fixtures

```python
def test_something(your_fixture):
    """Test description"""
    assert your_fixture is not None
```

### 4. Run Tests

```bash
python run_tests.py -k "test_something"
```

## Test Best Practices

1. **Use Fixtures**: Leverage existing fixtures for common setup
2. **Isolate Tests**: Each test should be independent
3. **Mock External Dependencies**: Use mocks for LLM, file system, etc.
4. **Test Edge Cases**: Include boundary conditions and error cases
5. **Use Descriptive Names**: Test names should clearly describe what they test
6. **Add Docstrings**: Document what each test verifies
7. **Clean Up**: Use fixtures with automatic cleanup

## CI/CD Integration

The test suite is designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run PES Tests
  run: |
    cd tests/pes
    python run_tests.py --coverage --unit

- name: Upload Coverage
  uses: codecov/codecov-action@v2
  with:
    directory: ./htmlcov
```

## Troubleshooting

### Tests Fail to Import

**Issue**: `ImportError: No module named 'openevolve'`

**Solution**:
```bash
# Install openevolve in development mode
cd /path/to/openevolve
pip install -e .
```

### Tests Timeout

**Issue**: Tests take too long or timeout

**Solution**:
- Reduce `population_size` and `max_generations` in test configs
- Use mock LLM client instead of real API
- Increase timeout in individual tests

### Coverage Below Target

**Issue**: Coverage below 80% target

**Solution**:
- Add tests for uncovered code paths
- Use `pytest --cov-report=html` to see specific uncovered lines
- Focus on testing edge cases and error conditions

## Performance Benchmarks

The test suite includes performance benchmarks to ensure:
- Initialization time < 5 seconds
- Database queries < 1 second for 100 programs
- Evaluator completes within timeout

Run benchmarks:
```bash
python run_tests.py -k "performance"
```

## Contributing

When adding new features to openevolve:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Update fixtures if needed
5. Document new tests

## Support

For issues or questions:
1. Check existing tests for examples
2. Review test fixtures for utilities
3. Consult main openevolve documentation
4. Open an issue on GitHub

## License

Same as openevolve project.
