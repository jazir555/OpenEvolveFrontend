# RESE Testing Guide

Comprehensive testing guide for RESE (Reasoning Engine for Symbolic Enhancement) components.

**Author:** Agent Z2 (Testing/QA Specialist)
**Created:** 2025-12-31
**Status:** 🟢 Active

---

## Table of Contents

1. [Overview](#overview)
2. [Test Infrastructure](#test-infrastructure)
3. [Running Tests](#running-tests)
4. [Test Organization](#test-organization)
5. [Writing Tests](#writing-tests)
6. [Performance Testing](#performance-testing)
7. [Validation Testing](#validation-testing)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Coverage Requirements](#coverage-requirements)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The RESE testing infrastructure provides comprehensive test coverage for all components:

- **Phase I (Φ₁.₅):** Tacit Assumption Miner
- **Phase II (I_mech, Ψ₃):** Isomorphic Mechanism Transfer & Constraint Inverter
- **Phase III (Γ₁):** MCTS-Guided Multi-Objective Search
- **Phase IV (Δ₃, DITO):** Statistical Validator & Optimizer
- **Core:** Symbolic Constraint Engine, Lean4 Bridge, Logic-to-Loss Translation

### Test Types

1. **Unit Tests:** Test individual components in isolation
2. **Integration Tests:** Test component interactions
3. **Performance Tests:** Load testing, stress testing, benchmarks
4. **Validation Tests:** Verify KEY INNOVATIONS meet thresholds

---

## Test Infrastructure

### Directory Structure

```
rese/
├── tests/
│   ├── conftest.py                 # Pytest fixtures and configuration
│   ├── test_utils.py               # Test utilities and helpers
│   ├── test_integration/           # Integration tests
│   │   ├── test_phase1_integration.py
│   │   ├── test_phase2_integration.py
│   │   ├── test_phase3_integration.py
│   │   ├── test_phase4_integration.py
│   │   └── test_full_pipeline.py
│   ├── test_performance/           # Performance tests
│   │   ├── test_load_testing.py
│   │   └── test_benchmarks.py
│   ├── test_validation/            # Validation tests
│   │   └── test_key_innovations.py
│   ├── test_core/                  # Core component tests
│   ├── test_imech/                 # I_mech tests
│   ├── gamma1/                     # Γ₁ tests
│   ├── phase1/                     # Phase I tests
│   ├── phase2/                     # Phase II tests
│   ├── phase3/                     # Phase III tests
│   └── phase4/                     # Phase IV tests
└── .github/workflows/              # CI/CD workflows
    └── test_pipeline.yml
```

### Fixtures

Key fixtures available in `conftest.py`:

```python
# Path fixtures
rese_root()              # RESE root directory
test_data_dir()          # Test data directory
test_db_dir()            # Test database directory
temp_dir()               # Temporary directory

# Phase I fixtures
sample_null_result()     # Single null result
sample_null_results()    # Multiple null results
phi15_engine()           # Φ₁.₅ engine

# Phase II fixtures
sample_fdg()             # Fundamental Dependency Graph
sample_source_domain()   # Source domain
sample_target_domain()   # Target domain

# Performance fixtures
performance_thresholds() # Performance thresholds
benchmark_results()      # Benchmark collector

# Validation fixtures
innovation_validators()  # Validators for KEY INNOVATIONS
sample_validation_data() # Validation test data
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_phi15.py -v

# Run specific test
pytest tests/test_phi15.py::TestPhi15Engine::test_engine_initialization -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run in parallel
pytest tests/ -n auto
```

### Run by Type

```bash
# Unit tests only
pytest tests/ -m unit -v

# Integration tests only
pytest tests/ -m integration -v

# Performance tests only
pytest tests/ -m performance -v

# Validation tests only
pytest tests/ -m validation -v

# Phase-specific tests
pytest tests/ -m phase1 -v
pytest tests/ -m phase2 -v
pytest tests/ -m phase3 -v
pytest tests/ -m phase4 -v
```

### Run with Options

```bash
# Verbose output
pytest tests/ -v

# Show print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Fail on warnings
pytest tests/ -W error

# Run slow tests
pytest tests/ --runslow

# Run performance tests
pytest tests/ --performance

# Run validation tests
pytest tests/ --validation
```

---

## Test Organization

### Test Markers

Tests are marked with pytest markers:

- `unit`: Unit tests
- `integration`: Integration tests
- `performance`: Performance tests
- `validation`: Validation tests
- `slow`: Slow-running tests
- `phase1`: Phase I tests
- `phase2`: Phase II tests
- `phase3`: Phase III tests
- `phase4`: Phase IV tests
- `core`: Core component tests

### Test Naming Conventions

```
test_<component>.py              # Component test file
test_<feature>_<aspect>.py       # Feature test file
Test<Class>                      # Test class
test_<method>_<aspect>           # Test method
test_integration_<phase>.py      # Integration test
test_performance_<type>.py       # Performance test
```

---

## Writing Tests

### Basic Test Structure

```python
import pytest
from test_utils import PerformanceTimer, ValidationHelpers

pytestmark = pytest.mark.unit

class TestMyComponent:
    """Test my component"""

    @pytest.fixture
    def component(self):
        """Get component instance"""
        return MyComponent()

    def test_basic_functionality(self, component):
        """Test basic functionality"""
        result = component.do_something()
        assert result is not None

    def test_edge_case(self, component):
        """Test edge case"""
        with pytest.raises(ValueError):
            component.do_something_invalid()
```

### Using Fixtures

```python
def test_with_fixtures(sample_null_result, phi15_engine):
    """Test using fixtures"""
    assumptions, _ = phi15_engine.process_null_results([sample_null_result])

    assert isinstance(assumptions, list)
```

### Performance Testing

```python
def test_performance(self):
    """Test performance"""
    with PerformanceTimer("my_test") as timer:
        # Do work
        result = expensive_operation()

    elapsed = timer.get_elapsed()
    assert elapsed < 1.0, "Should complete in < 1 second"
```

### Validation Testing

```python
def test_validation(self):
    """Test validation"""
    passed, metric = ValidationHelpers.validate_phi15_accuracy(
        predictions, ground_truth, min_accuracy=0.70
    )

    assert passed, f"Accuracy {metric:.2%} below threshold"
```

---

## Performance Testing

### Load Testing

```bash
# Run load tests
pytest tests/test_performance/test_load_testing.py -v -s

# With specific parameters
pytest tests/test_performance/ -k "load_1000" -v
```

### Stress Testing

```bash
# Run stress tests
pytest tests/test_performance/ -k "stress" -v

# Memory stress tests
pytest tests/test_performance/ -k "memory" -v
```

### Benchmarking

```bash
# Run benchmarks
pytest tests/test_performance/ --benchmark-only

# Save benchmark results
pytest tests/test_performance/ --benchmark-only --benchmark-json=benchmark.json
```

### Performance Thresholds

| Component | Metric | Threshold |
|-----------|--------|-----------|
| Φ₁.₅ | Throughput | > 5 failures/second |
| SCE | Throughput | > 30 constraints/second |
| DITO | Speedup | > 3000x |
| Memory | Peak usage | < 500 MB |

---

## Validation Testing

### KEY INNOVATIONS Validation

Run all validation tests:

```bash
pytest tests/test_validation/ -v -s
```

### Individual Innovation Validation

```bash
# Φ₁.₅ accuracy validation
pytest tests/test_validation/test_key_innovations.py::TestPhi15Validation -v

# I_mech transfer validation
pytest tests/test_validation/test_key_innovations.py::TestImechValidation -v

# Γ₁ correlation validation
pytest tests/test_validation/test_key_innovations.py::TestGamma1Validation -v

# Δ₃ correlation validation
pytest tests/test_validation/test_key_innovations.py::TestDelta3Validation -v

# Ψ₃ reduction validation
pytest tests/test_validation/test_key_innovations.py::TestPsi3Validation -v

# DITO speedup validation
pytest tests/test_validation/test_key_innovations.py::TestDitoValidation -v
```

### Validation Thresholds

| Innovation | Metric | Threshold |
|------------|--------|-----------|
| Φ₁.₅ | Accuracy | > 70% |
| I_mech | Transfer Rate | > 80% |
| Γ₁ | Correlation | > 85% |
| Δ₃ | Correlation | > 85% |
| Ψ₃ | Reduction | > 10x |
| DITO | Speedup | > 3000x |

---

## CI/CD Pipeline

### GitHub Actions Workflow

The RESE testing pipeline uses GitHub Actions:

**File:** `.github/workflows/test_pipeline.yml`

**Jobs:**
1. `unit-tests`: Run unit tests with coverage
2. `integration-tests`: Run integration tests (matrix: phase1-4)
3. `performance-tests`: Run performance tests
4. `validation-tests`: Run validation tests
5. `coverage-report`: Generate coverage report
6. `test-summary`: Summarize all results
7. `performance-regression-check`: Check for performance regression

### Triggering CI/CD

```bash
# Push to trigger tests
git push origin main

# Pull request to trigger tests
gh pr create

# Manual trigger
gh workflow run test_pipeline.yml
```

### Viewing Results

```bash
# Check workflow status
gh workflow list

# View recent runs
gh run list --workflow=test_pipeline.yml

# View specific run
gh run view <run-id>
```

---

## Coverage Requirements

### Target Coverage

**Overall:** > 80%

**By Component:**
- Phase I (Φ₁.₅): > 80%
- Phase II (I_mech, Ψ₃): > 80%
- Phase III (Γ₁): > 80%
- Phase IV (Δ₃, DITO): > 80%
- Core: > 85%

### Generating Coverage Reports

```bash
# HTML report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest tests/ --cov=. --cov-report=xml

# Terminal report
pytest tests/ --cov=. --cov-report=term-missing

# JSON report
pytest tests/ --cov=. --cov-report=json
```

### Interpreting Coverage

```bash
# Check coverage threshold
coverage report --fail-under=80

# View missing lines
coverage report -m

# Filter by file
coverage report --include="rese/phase1/*"
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'rese'`

**Solution:**
```bash
# Add RESE root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Fixture Not Found

**Problem:** `fixture 'sample_null_result' not found`

**Solution:**
```bash
# Ensure conftest.py is in tests/ directory
ls tests/conftest.py

# Check fixture is defined
grep -n "def sample_null_result" tests/conftest.py
```

#### 3. Performance Tests Timeout

**Problem:** Performance tests exceed time limit

**Solution:**
```bash
# Run performance tests separately
pytest tests/ -m performance --timeout=300

# Skip slow tests
pytest tests/ -m "not slow"
```

#### 4. Database Lock Errors

**Problem:** `sqlite3.OperationalError: database is locked`

**Solution:**
```bash
# Clean up test databases
rm -rf tests/test_databases/*.db

# Use unique database per test (handled by fixtures)
```

### Debug Mode

```bash
# Enable debug logging
pytest tests/ -v --log-cli-level=DEBUG

# Drop into debugger on failure
pytest tests/ -v --pdb

# Show local variables on failure
pytest tests/ -v -l
```

### Test Isolation

```bash
# Run tests in isolation (one per process)
pytest tests/ -n 0

# Run tests in random order (detect dependencies)
pytest tests/ --random-order

# Run tests repeatedly (detect flaky tests)
pytest tests/ --count=10
```

---

## Best Practices

### 1. Test Independence

Tests should be independent and order-independent:

```python
# Good: Each test sets up its own data
def test_feature_x():
    data = create_test_data()
    result = process(data)
    assert result is not None

# Bad: Tests depend on execution order
def test_feature_1():
    global.data = create_data()

def test_feature_2():
    result = process(global.data)  # Depends on test_feature_1
```

### 2. Descriptive Names

Use descriptive test names:

```python
# Good
def test_phi15_accuracy_exceeds_threshold():
    ...

def test_imech_transfer_rate_with_high_quality_mappings():
    ...

# Bad
def test_1():
    ...

def test_it_works():
    ...
```

### 3. Fixtures for Shared Data

Use fixtures for shared test data:

```python
# Good: Use fixtures
@pytest.fixture
def sample_constraints():
    return generate_constraints(count=10)

def test_processing(sample_constraints):
    result = process(sample_constraints)
    ...

# Bad: Hardcode data in each test
def test_processing():
    constraints = [c1, c2, c3, ...]  # Duplicated
    result = process(constraints)
    ...
```

### 4. Assertions

Use specific assertions:

```python
# Good: Specific assertion
assert result == expected, f"Expected {expected}, got {result}"
assert len(constraints) > 0, "Should have constraints"

# Bad: Generic assertion
assert result  # What exactly should be true?
```

### 5. Test Documentation

Document complex tests:

```python
def test_complex_workflow():
    """
    Test complete workflow from constraint generation to optimization.

    Steps:
    1. Generate 100 constraints
    2. Apply Ψ₃ reduction
    3. Optimize with DITO
    4. Validate results

    Expected: At least 10x reduction, < 1 second processing
    """
    # Test implementation
    ...
```

---

## Quick Reference

### Essential Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific phase
pytest tests/ -m phase1 -v

# Run performance tests
pytest tests/ -m performance -v

# Run validation tests
pytest tests/ -m validation -v

# Run in parallel
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x -v
```

### Test Structure Template

```python
"""
Tests for <Component>

Author: Your Name
Created: Date
"""

import pytest
from test_utils import PerformanceTimer, ValidationHelpers

pytestmark = pytest.mark.unit  # or integration, performance, validation

class TestComponent:
    """Test <Component>"""

    @pytest.fixture
    def component(self):
        """Get component instance"""
        return Component()

    def test_basic_functionality(self, component):
        """Test basic functionality"""
        result = component.method()
        assert result is not None

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (3, 6),
        (5, 10),
    ])
    def test_parameterized(self, component, input, expected):
        """Test with parameters"""
        result = component.process(input)
        assert result == expected
```

---

## Contact & Support

For testing-related questions or issues:

- **Testing Lead:** Agent Z2 (Testing/QA Specialist)
- **Documentation:** `TESTING_GUIDE.md`
- **Issues:** GitHub Issues

---

**Last Updated:** 2025-12-31
**Version:** 1.0.0
