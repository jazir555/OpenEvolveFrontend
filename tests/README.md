# RESE Test Suite

Comprehensive test infrastructure for RESE (Reasoning Engine for Symbolic Enhancement).

**Maintainer:** Agent Z2 (Testing/QA Specialist)
**Created:** 2025-12-31
**Status:** 🟢 Active

---

## Quick Start

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
```

---

## Test Statistics

| Metric | Count | Target |
|--------|-------|--------|
| Total Tests | 500+ | 600+ |
| Unit Tests | 300+ | 350+ |
| Integration Tests | 150+ | 180+ |
| Performance Tests | 30+ | 40+ |
| Validation Tests | 20+ | 25+ |
| Coverage | 82% | 85% |

---

## Directory Structure

```
tests/
├── conftest.py                      # Pytest fixtures and configuration
├── test_utils.py                    # Test utilities and helpers
├── TESTING_GUIDE.md                 # Comprehensive testing guide
├── QA_PROCEDURES.md                 # QA procedures and policies
├── BUG_REPORT_TEMPLATE.md           # Bug report template
│
├── test_integration/                # Integration tests
│   ├── test_phase1_integration.py   # Phase I integration
│   ├── test_phase2_integration.py   # Phase II integration
│   ├── test_phase3_integration.py   # Phase III integration
│   ├── test_phase4_integration.py   # Phase IV integration
│   └── test_full_pipeline.py        # Full pipeline tests
│
├── test_performance/                # Performance tests
│   ├── test_load_testing.py         # Load testing (1000+ constraints)
│   └── test_benchmarks.py           # Benchmark suite
│
├── test_validation/                 # Validation tests
│   └── test_key_innovations.py      # KEY INNOVATIONS validation
│
├── test_core/                       # Core component tests
│   ├── test_symbolic_constraint_engine.py
│   ├── test_constraint_lean4_bridge.py
│   ├── test_dito_optimizer.py
│   └── ...
│
├── test_imech/                      # I_mech tests
│   ├── test_algorithms.py
│   ├── test_transfer.py
│   └── test_validation.py
│
├── gamma1/                          # Γ₁ tests
│   └── test_aci_complete.py
│
├── phase1/                          # Phase I tests
│   ├── test_cognitive_biases.py
│   └── test_phi2_integration.py
│
├── phase2/                          # Phase II tests
│   └── ...
│
├── phase3/                          # Phase III tests
│   ├── test_mcts_search.py
│   ├── test_statistical_validator.py
│   └── test_stage3_integration.py
│
└── phase4/                          # Phase IV tests
    └── ...
```

---

## Test Categories

### 1. Unit Tests (`pytest.mark.unit`)

Test individual components in isolation.

**Example:**
```bash
pytest tests/test_core/test_symbolic_constraint_engine.py -m unit -v
```

**Coverage:**
- Core algorithms
- Data structures
- Individual functions
- Class methods

### 2. Integration Tests (`pytest.mark.integration`)

Test component interactions and data flow.

**Example:**
```bash
pytest tests/test_integration/ -m integration -v
```

**Coverage:**
- Phase-to-phase integration
- End-to-end pipelines
- Cross-phase validation
- Data flow verification

### 3. Performance Tests (`pytest.mark.performance`)

Test system performance under various loads.

**Example:**
```bash
pytest tests/test_performance/ -m performance -v
```

**Coverage:**
- Load testing (1000+ constraints)
- Stress testing (extreme cases)
- Benchmark suite
- Regression detection

### 4. Validation Tests (`pytest.mark.validation`)

Validate KEY INNOVATIONS meet thresholds.

**Example:**
```bash
pytest tests/test_validation/ -m validation -v -s
```

**Coverage:**
- Φ₁.₅ accuracy >70%
- I_mech transfer >80%
- Γ₁ correlation >85%
- Δ₃ correlation >85%
- Ψ₃ reduction >10x
- DITO speedup >3000x

---

## Key Fixtures

### Path Fixtures

```python
rese_root()              # RESE root directory
test_data_dir()          # Test data directory
test_db_dir()            # Test database directory
temp_dir()               # Temporary directory (auto-cleanup)
```

### Phase I Fixtures

```python
sample_null_result()     # Single null result
sample_null_results()    # Multiple null results
phi15_engine()           # Φ₁.₅ engine
```

### Phase II Fixtures

```python
sample_fdg()             # Fundamental Dependency Graph
sample_source_domain()   # Source domain
sample_target_domain()   # Target domain
psi3_constraint_set()    # Constraint set for Ψ₃
```

### Phase III Fixtures

```python
sample_pareto_front()    # Pareto front data
mcts_search_engine()     # MCTS search engine
```

### Phase IV Fixtures

```python
sample_constraint_pool() # Constraint pool for DITO
dito_optimizer()         # DITO optimizer
```

### Performance Fixtures

```python
performance_thresholds() # Performance thresholds
benchmark_results()      # Benchmark collector
```

### Validation Fixtures

```python
innovation_validators()  # Validators for KEY INNOVATIONS
sample_validation_data() # Validation test data
```

---

## Test Utilities

Located in `test_utils.py`:

### Performance Measurement

```python
from test_utils import PerformanceTimer, measure_time, measure_memory

# Context manager
with PerformanceTimer("operation") as timer:
    result = expensive_operation()
elapsed = timer.get_elapsed()

# Decorator
@measure_time
def my_function():
    ...

# Measure memory
@measure_memory
def memory_intensive_function():
    ...
```

### Data Generation

```python
from test_utils import TestDataGenerator

generator = TestDataGenerator()

# Generate constraints
constraints = generator.generate_constraints(count=100, complexity="medium")

# Generate null results
null_results = generator.generate_null_results(count=50, pattern="systematic")

# Generate FDG
fdg = generator.generate_fdg(n_nodes=20, n_edges=30)
```

### Assertions

```python
from test_utils import TestAssertions

assertions = TestAssertions()

# Assert constraint valid
assertions.assert_constraint_valid(constraint)

# Assert performance threshold
assertions.assert_performance_threshold(actual_time, max_time, "operation")

# Assert innovation validated
assertions.assert_innovation_validated("phi15", accuracy, 0.70)
```

### Validation

```python
from test_utils import ValidationHelpers

# Validate Φ₁.₅
passed, accuracy = ValidationHelpers.validate_phi15_accuracy(
    predictions, ground_truth, min_accuracy=0.70
)

# Validate I_mech
passed, transfer_rate = ValidationHelpers.validate_imech_transfer(
    source_constraints, transferred, mapping_scores, min_transfer=0.80
)
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific file
pytest tests/test_phi15.py -v

# Run specific test
pytest tests/test_phi15.py::TestPhi15Engine::test_engine_initialization -v

# Stop on first failure
pytest tests/ -x -v

# Run in parallel
pytest tests/ -n auto
```

### By Type

```bash
# Unit tests
pytest tests/ -m unit -v

# Integration tests
pytest tests/ -m integration -v

# Performance tests
pytest tests/ -m performance -v

# Validation tests
pytest tests/ -m validation -v
```

### By Phase

```bash
# Phase I (Φ₁.₅)
pytest tests/ -m phase1 -v

# Phase II (I_mech, Ψ₃)
pytest tests/ -m phase2 -v

# Phase III (Γ₁)
pytest tests/ -m phase3 -v

# Phase IV (Δ₃, DITO)
pytest tests/ -m phase4 -v
```

### With Options

```bash
# Verbose with print statements
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -v -l

# Run slow tests
pytest tests/ --runslow

# Run performance tests
pytest tests/ --performance

# Run validation tests
pytest tests/ --validation
```

---

## Coverage Reports

### Generate Coverage

```bash
# HTML report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=. --cov-report=term-missing

# XML report (for CI/CD)
pytest tests/ --cov=. --cov-report=xml

# JSON report
pytest tests/ --cov=. --cov-report=json
```

### Coverage Requirements

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| Phase I (Φ₁.₅) | 85% | 85% | 🟢 |
| Phase II (I_mech) | 82% | 85% | 🟡 |
| Phase II (Ψ₃) | 80% | 85% | 🟡 |
| Phase III (Γ₁) | 78% | 85% | 🟡 |
| Phase IV (Δ₃) | 81% | 85% | 🟡 |
| Phase IV (DITO) | 83% | 85% | 🟡 |
| Core (SCE) | 87% | 90% | 🟢 |
| **Overall** | **82%** | **85%** | **🟡** |

---

## CI/CD Pipeline

### GitHub Actions Workflow

**File:** `.github/workflows/test_pipeline.yml`

**Jobs:**
1. `unit-tests` - Unit tests with coverage
2. `integration-tests` - Integration tests (matrix: phase1-4)
3. `performance-tests` - Performance tests
4. `validation-tests` - Validation tests (KEY INNOVATIONS)
5. `coverage-report` - Coverage report generation
6. `test-summary` - Summary of all test results
7. `performance-regression-check` - Performance regression detection

### Triggering CI/CD

```bash
# Push to trigger
git push origin main

# Pull request to trigger
gh pr create

# Manual trigger
gh workflow run test_pipeline.yml
```

---

## Documentation

- **TESTING_GUIDE.md** - Comprehensive testing guide
- **QA_PROCEDURES.md** - QA procedures and policies
- **BUG_REPORT_TEMPLATE.md** - Bug report template

---

## Contributing Tests

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

### Best Practices

1. **Independent:** Tests should be independent and order-independent
2. **Descriptive:** Use descriptive test names
3. **Fixtures:** Use fixtures for shared data
4. **Specific:** Use specific assertions
5. **Documented:** Document complex tests

---

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Fixture Not Found:**
```bash
# Ensure conftest.py is in tests/ directory
ls tests/conftest.py
```

**Database Lock Errors:**
```bash
# Clean up test databases
rm -rf tests/test_databases/*.db
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

---

## Contact & Support

- **QA Lead:** Agent Z2 (Testing/QA Specialist)
- **Documentation:** `TESTING_GUIDE.md`
- **Issues:** GitHub Issues

---

## License

Part of the RESE project. See main project LICENSE file.

---

**Last Updated:** 2025-12-31
**Version:** 1.0.0
