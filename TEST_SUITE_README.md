# Test Suite Framework

A comprehensive, production-ready test suite management system for OpenEvolve Frontend that supports multiple test frameworks, parallel execution, coverage reporting, and advanced features like test filtering and retry logic.

## Features

- **Multiple Framework Support**: Works with pytest, unittest, and generic Python test functions
- **Parallel Execution**: Run tests concurrently for faster feedback
- **Test Discovery**: Automatically discover tests in your project
- **Test Filtering**: Filter tests by type, priority, tags, or patterns
- **Coverage Reporting**: Integrated code coverage tracking
- **HTML/JSON Reports**: Generate detailed test reports
- **Test History**: Track test trends over time
- **Flaky Test Detection**: Identify and retry unstable tests
- **CI/CD Integration**: Designed for continuous integration pipelines
- **Comprehensive Logging**: Structured logging for debugging

## Installation

The test suite framework is included in the OpenEvolve Frontend project. No additional installation required.

## Quick Start

### Basic Usage

```python
from test_suite import create_test_suite

# Create a test suite
suite = create_test_suite(
    name="my_tests",
    test_dirs=["tests"],
)

# Run all tests
result = suite.run()

# Print results
print(f"Passed: {result.passed}/{result.total_tests}")
print(f"Success rate: {result.success_rate:.1f}%")
```

### Quick Test Execution

```python
from test_suite import run_tests, TestType

# Run all tests
result = run_tests(verbose=True)

# Run only unit tests
result = run_tests(test_type=TestType.UNIT)

# Run tests in parallel
result = run_tests(parallel=True)
```

## Configuration

### Basic Configuration

```python
from test_suite import TestSuiteConfig, TestSuite

config = TestSuiteConfig(
    test_dirs=["tests"],
    parallel_workers=4,
    enable_coverage=True,
    coverage_threshold=70.0,
    verbose=True,
)

suite = TestSuite(config=config)
```

### Advanced Configuration

```python
config = TestSuiteConfig(
    # Test discovery
    test_dirs=[Path("tests/unit"), Path("tests/integration")],
    test_patterns=["test_*.py", "*_test.py"],
    exclude_patterns=["*/test_*.py"],

    # Execution
    parallel_workers=8,
    timeout=300,
    max_retries=3,
    retry_flaky_tests=True,
    stop_on_first_failure=False,

    # Coverage
    enable_coverage=True,
    coverage_threshold=80.0,

    # Reporting
    verbose=True,
    json_report=Path("reports/test_results.json"),
    html_report=Path("reports/test_results.html"),
    history_file=Path("reports/test_history.json"),
    log_file=Path("logs/test_execution.log"),
)
```

## Test Types

The framework supports the following test types:

- **UNIT**: Unit tests (fast, isolated)
- **INTEGRATION**: Integration tests (multiple components)
- **E2E**: End-to-end tests (full workflows)
- **PERFORMANCE**: Performance/benchmark tests
- **SECURITY**: Security vulnerability tests
- **API**: API endpoint tests
- **DATABASE**: Database integration tests
- **UI**: User interface tests

## Test Filtering

### Filter by Type

```python
suite = create_test_suite()

# Get only unit tests
unit_tests = suite.filter_tests(test_type=TestType.UNIT)
result = suite.run(test_names=unit_tests)
```

### Filter by Priority

```python
from test_suite import Priority

# Get critical and high priority tests
important_tests = suite.filter_tests(priority=Priority.HIGH)
result = suite.run(test_names=important_tests)
```

### Filter by Tags

```python
# Get tests with specific tags
api_tests = suite.filter_tests(tags={"api", "fast"})
result = suite.run(test_names=api_tests)
```

### Filter by Pattern

```python
# Get tests matching a regex pattern
user_tests = suite.filter_tests(pattern=r"user.*")
result = suite.run(test_names=user_tests)
```

## Test Decorators

### Flaky Tests

Mark tests that may fail intermittently:

```python
from test_suite import flaky

@flaky(max_runs=3, min_passes=1)
def test_network_call():
    response = requests.get("https://api.example.com")
    assert response.status_code == 200
```

### Slow Tests

Mark slow-running tests:

```python
from test_suite import slow

@slow
def test_large_dataset():
    data = process_large_dataset()
    assert len(data) > 0
```

### Network Tests

Mark tests requiring network access:

```python
from test_suite import requires_network

@requires_network
def test_external_api():
    response = requests.get("https://api.example.com")
    assert response.status_code == 200
```

## Pytest Markers

The framework integrates with pytest markers:

```python
import pytest

@pytest.mark.integration
def test_database_connection():
    assert db.connect() is True

@pytest.mark.slow
def test_large_operation():
    assert process_large_data() is True

@pytest.mark.requires_network
def test_api_call():
    assert make_request() is successful

@pytest.mark.critical
def test_core_functionality():
    assert core_feature() is working
```

## Test History and Trends

Track test performance over time:

```python
suite = create_test_suite(
    history_file="test_history.json"
)

# Run tests to populate history
result = suite.run()

# Identify flaky tests (pass rate < 80%)
flaky_tests = suite.get_flaky_tests(threshold=0.8)
for test_name in flaky_tests:
    history = suite.get_trends(test_name)
    print(f"{test_name}: {history.pass_rate:.1f}% pass rate")

# Identify slow tests (avg duration > 5s)
slow_tests = suite.get_slow_tests(threshold=5.0)
for test_name in slow_tests:
    history = suite.get_trends(test_name)
    print(f"{test_name}: {history.avg_duration:.2f}s average")
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          python -c "
          from test_suite import create_test_suite, TestSuiteConfig
          from pathlib import Path

          config = TestSuiteConfig(
              test_dirs=[Path('tests')],
              parallel_workers=4,
              enable_coverage=True,
              coverage_threshold=70.0,
              json_report=Path('test_results.json'),
              html_report=Path('test_results.html'),
          )

          suite = create_test_suite(config=config)
          result = suite.run()

          print(f'Success rate: {result.success_rate:.1f}%')
          exit(0 if result.was_successful else 1)
          "
```

### Command Line Usage

```bash
# Run all tests
python test_suite.py

# Run only unit tests
python test_suite.py --type unit

# Run tests matching pattern
python test_suite.py --pattern "api.*"

# Run tests in parallel
python test_suite.py --parallel

# Generate reports
python test_suite.py --json-report results.json --html-report results.html

# Verbose output
python test_suite.py --verbose
```

## Test Discovery

The framework automatically discovers tests in the following patterns:

- `test_*.py`: Files starting with `test_`
- `*_test.py`: Files ending with `_test.py`

Within these files, it looks for:

- Functions starting with `test_`
- Classes inheriting from `unittest.TestCase`
- Pytest test functions

## Report Formats

### JSON Report

```json
{
  "suite_name": "my_tests",
  "framework": "pytest",
  "timestamp": "2024-01-15T10:30:00",
  "summary": {
    "total": 100,
    "passed": 95,
    "failed": 3,
    "skipped": 2,
    "errors": 0,
    "duration": 45.2,
    "success_rate": 95.0
  },
  "tests": [
    {
      "name": "test_example",
      "status": "passed",
      "duration": 0.123,
      "error": null
    }
  ]
}
```

### HTML Report

The HTML report provides a visual summary of test results with:

- Overall statistics
- Individual test results
- Color-coded status indicators
- Duration information
- Error messages (if any)

## Best Practices

### 1. Organize Tests by Type

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component integration tests
├── e2e/           # End-to-end tests
└── performance/   # Performance benchmarks
```

### 2. Use Descriptive Test Names

```python
# Good
def test_user_login_with_invalid_credentials_returns_error():
    pass

# Bad
def test_login():
    pass
```

### 3. Mark Tests Appropriately

```python
@pytest.mark.slow
@pytest.mark.requires_database
def test_complex_query():
    pass
```

### 4. Keep Tests Independent

Each test should be able to run in isolation without depending on other tests.

### 5. Use Fixtures for Setup

```python
import pytest

@pytest.fixture
def database():
    db = create_test_database()
    yield db
    db.cleanup()

def test_query(database):
    result = database.query("SELECT * FROM users")
    assert len(result) > 0
```

## API Reference

### TestSuite

Main test suite class.

**Methods:**

- `discover_tests()`: Discover all tests
- `filter_tests(test_type, priority, tags, pattern)`: Filter tests
- `run(test_names)`: Run tests
- `get_trends(test_name)`: Get test history
- `get_flaky_tests(threshold)`: Identify flaky tests
- `get_slow_tests(threshold)`: Identify slow tests

### TestSuiteConfig

Configuration class for test suites.

**Attributes:**

- `project_root`: Project root directory
- `test_dirs`: Directories to search for tests
- `test_patterns`: File patterns to match
- `parallel_workers`: Number of parallel workers
- `enable_coverage`: Enable code coverage
- `coverage_threshold`: Minimum coverage percentage
- `timeout`: Test timeout in seconds
- `max_retries`: Maximum retry attempts
- `retry_flaky_tests`: Enable retry for flaky tests
- `verbose`: Enable verbose output
- `json_report`: Path to JSON report
- `html_report`: Path to HTML report
- `history_file`: Path to history file

### SuiteResult

Test suite execution result.

**Attributes:**

- `total_tests`: Total number of tests
- `passed`: Number of passed tests
- `failed`: Number of failed tests
- `skipped`: Number of skipped tests
- `errors`: Number of errors
- `duration`: Execution duration in seconds
- `success_rate`: Success rate percentage
- `was_successful`: Whether suite was successful

## Troubleshooting

### Tests Not Discovered

If tests are not being discovered:

1. Check that test files match the patterns (`test_*.py` or `*_test.py`)
2. Verify test functions start with `test_`
3. Ensure test directories are specified in configuration

### Parallel Execution Issues

If parallel execution causes problems:

1. Reduce number of workers: `parallel_workers=2`
2. Disable parallel execution: `parallel_workers=1`
3. Check for shared resources that need locks

### Coverage Reporting Issues

If coverage reporting fails:

1. Ensure `pytest-cov` is installed
2. Disable coverage: `enable_coverage=False`
3. Check that tests are in the same package as code

## Examples

See `test_suite_examples.py` for comprehensive usage examples including:

1. Basic test suite creation
2. Custom configuration
3. Test filtering
4. Parallel execution
5. Using decorators
6. CI/CD integration
7. Test history and trends
8. Quick execution
9. Selective execution
10. Custom discovery
11. Report generation
12. Environment-specific testing

## Contributing

When adding new features:

1. Add unit tests to `test_test_suite.py`
2. Update this README with usage examples
3. Ensure backward compatibility
4. Add type hints to all functions

## License

MIT License - See LICENSE file for details
