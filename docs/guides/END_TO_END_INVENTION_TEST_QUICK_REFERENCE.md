# End-to-End Invention Planner - Test Suite Quick Reference

Quick guide for running and using the test suite and demonstration scripts.

---

## Quick Start

### Run All Tests
```bash
pytest test_end_to_end_invention_comprehensive.py -v
```

### Run All Demos
```bash
python demo_end_to_end_invention.py
```

---

## Test Categories

### Markers
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# End-to-end tests only
pytest -m end_to_end

# Real invention tests
pytest -m real_invention

# Validation tests
pytest -m validation

# Performance tests
pytest -m performance

# Skip slow tests
pytest -m "not slow"
```

---

## Specific Test Examples

### Real Invention Tests

#### Magnetic Nanoparticles (Chemistry)
```bash
pytest test_end_to_end_invention_comprehensive.py::TestRealInventions::test_magnetic_nanoparticles -v
```

#### High-Temperature Superconductor (Physics)
```bash
pytest test_end_to_end_invention_comprehensive.py::TestRealInventions::test_high_temperature_superconductor -v
```

#### Novel Alloy (Materials Science)
```bash
pytest test_end_to_end_invention_comprehensive.py::TestRealInventions::test_novel_alloy -v
```

#### Biological Assay (Biology)
```bash
pytest test_end_to_end_invention_comprehensive.py::TestRealInventions::test_biological_assay -v
```

### Validation Tests

#### Known Invention (Penicillin)
```bash
pytest test_end_to_end_invention_comprehensive.py::TestValidationInventions::test_known_invention_penicillin -v
```

#### Impossible Invention (Perpetual Motion)
```bash
pytest test_end_to_end_invention_comprehensive.py::TestValidationInventions::test_impossible_invention_perpetual_motion -v
```

---

## Coverage Report

### Generate Coverage Report
```bash
pytest test_end_to_end_invention_comprehensive.py --cov=end_to_end_invention_planner --cov-report=html
```

View report:
```bash
# On Windows
start htmlcov/index.html

# On Linux/Mac
open htmlcov/index.html
```

---

## Running Individual Demos

### Using Python
```python
import asyncio
from demo_end_to_end_invention import demo_2_simple_invention

# Run single demo
asyncio.run(demo_2_simple_invention())
```

### Available Demos

1. `demo_1_capabilities()` - System capabilities (fast)
2. `demo_2_simple_invention()` - Magnetic nanoparticles (requires execution)
3. `demo_3_complex_invention()` - High-temperature superconductor (optional)
4. `demo_4_material_science()` - Novel alloy (optional)
5. `demo_5_export_document()` - Export executable document (requires execution)
6. `demo_6_binary_validation()` - Binary criteria explanation (fast)
7. `demo_7_complete_workflow()` - Workflow visualization (fast)
8. `demo_8_comparison_with_without_integrations()` - Integration impact (fast)
9. `demo_9_real_world_example()` - Real-world comparison (fast)

---

## Test Configuration

### Pytest Configuration File
Create `pytest.ini`:
```ini
[pytest]
testpaths = .
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests for individual components
    integration: Integration tests for subsystems
    end_to_end: End-to-end pipeline tests
    real_invention: Tests with real scientific inventions
    validation: Validation tests (known/impossible inventions)
    performance: Performance and stress tests
    slow: Tests that take longer to run
    async: Async tests
asyncio_mode = auto
```

---

## Expected Test Results

### Current Implementation Status

The current `end_to_end_invention_planner.py` is a skeleton implementation. Expected pass rates:

- Unit Tests: ~90%
- Integration Tests: ~70%
- End-to-End Tests: ~60%
- Real Invention Tests: ~50%
- Validation Tests: ~80%
- Performance Tests: ~100%

### Full Implementation

After completing all tasks in `END_TO_END_INVENTION_AGENT_TASKS.md`:
- All Tests: ~95%+
- Real Invention Tests: ~90%+
- End-to-End Tests: ~95%+

---

## Troubleshooting

### Import Errors

```bash
# Ensure end_to_end_invention_planner.py is in the current directory
ls end_to_end_invention_planner.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing Dependencies

```bash
# Install required packages
pip install pytest pytest-asyncio pytest-cov
```

### Tests Hanging

```bash
# Add timeout to tests
pytest test_end_to_end_invention_comprehensive.py -v --timeout=300
```

### Slow Tests

```bash
# Skip slow tests
pytest test_end_to_end_invention_comprehensive.py -m "not slow" -v
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test End-to-End Invention Planner

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install pytest pytest-asyncio pytest-cov
      - run: pytest test_end_to_end_invention_comprehensive.py -v --cov
      - uses: codecov/codecov-action@v2
```

---

## Performance Benchmarks

### Expected Timing

| Test Category | Expected Time |
|--------------|---------------|
| Unit Tests | < 1 minute |
| Integration Tests | < 2 minutes |
| End-to-End Tests | < 5 minutes |
| Real Invention Tests | < 15 minutes |
| Validation Tests | < 5 minutes |
| Performance Tests | < 10 minutes |
| **Total** | **< 40 minutes** |

---

## Test Output Examples

### Successful Test Output

```
test_end_to_end_invention_comprehensive.py::TestInventionGoal::test_invention_goal_creation PASSED
test_end_to_end_invention_comprehensive.py::TestRealInventions::test_magnetic_nanoparticles PASSED
test_end_to_end_invention_comprehensive.py::TestValidationInventions::test_known_invention_penicillin PASSED

=== 52 passed, 3 skipped, 1 warning in 15.3s ===
```

### Failed Test Output

```
test_end_to_end_invention_comprehensive.py::TestRealInventions::test_magnetic_nanoparticles FAILED

    def test_magnetic_nanoparticles(self, invention_planner):
        bulletproof = await invention_planner.plan_invention(...)
>       assert "nanoparticle" in bulletproof.invention_goal.target.lower()
E       AssertionError: assert 'nanoparticle' in 'iron oxide particles'
```

---

## Next Steps

1. Run unit tests to verify basic functionality:
   ```bash
   pytest test_end_to_end_invention_comprehensive.py -m unit -v
   ```

2. Run a simple real invention test:
   ```bash
   pytest test_end_to_end_invention_comprehensive.py::TestRealInventions::test_magnetic_nanoparticles -v -s
   ```

3. Run the demonstration script:
   ```bash
   python demo_end_to_end_invention.py
   ```

4. Generate coverage report:
   ```bash
   pytest test_end_to_end_invention_comprehensive.py --cov=end_to_end_invention_planner --cov-report=html
   ```

---

For more details, see:
- `END_TO_END_INVENTION_TEST_SUITE_REPORT.md` - Comprehensive test documentation
- `END_TO_END_INVENTION_AGENT_TASKS.md` - Implementation tasks
- `test_end_to_end_invention_comprehensive.py` - Test suite
- `demo_end_to_end_invention.py` - Demonstration scripts
