# RESE Framework Testing & Validation Suite

**Comprehensive Debugging and Validation for Phase 1 & Phase 2**

**Generated**: 2025-12-31
**Version**: 1.0.0
**Status**: 🟢 Ready for Testing

---

## Overview

This testing suite provides comprehensive validation of the RESE (Recursive Synthesis Engine) framework Phase 1 and Phase 2 modules, including:

### Phase 1 Modules
- **Φ₁.₅ Tacit Assumption Miner**: Mining hidden assumptions from failure patterns
- **Φ₂ Metacognitive Debiasing**: Detecting and mitigating cognitive biases

### Phase 2 Modules
- **Ψ₃ Constraint Inversion**: Reducing constraint redundancy (6.6x target)
- **Ψ₂ Ontology Mapping**: Cross-domain knowledge graph alignment
- **I_mech Isomorphism Validator**: Transferring solutions across similar domains

---

## Quick Start

### 1. Run All Tests (One Command)

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python run_rese_tests.py --phase all --verbose
```

### 2. Run Specific Phase

```bash
# Phase 1 only
python run_rese_tests.py --phase 1 --verbose

# Phase 2 only
python run_rese_tests.py --phase 2 --verbose
```

### 3. Run with Pytest Directly

```bash
# All Phase 1 tests
pytest rese/tests/phase1/ -v

# All I_mech tests
pytest rese/tests/test_imech/ -v

# Integration tests
pytest rese/tests/test_integration/ -v
```

### 4. Check Results

Results saved to: `rese_test_results.json`

---

## Documentation Files

### 1. RESE_PHASE_DEBUG_REPORT.md
**Comprehensive testing documentation** (11,000+ words)

Contents:
- Executive summary with status dashboard
- Complete test structure for all modules
- Detailed test coverage breakdown
- Performance benchmarks and targets
- KEY INNOVATIONS validation criteria
- Test data examples and fixtures
- Common issues and solutions
- Debug report template

**Use when**: You need detailed information about test structure, coverage, and validation criteria.

### 2. run_rese_tests.py
**Automated test runner script**

Features:
- Run all tests with one command
- Colored output and progress tracking
- Automatic result parsing
- JSON report generation
- Phase-specific testing
- Verbose/debug modes

**Use when**: You want to run tests programmatically or automate testing.

### 3. RESE_BUG_TRACKING_TEMPLATE.md
**Bug tracking and validation template** (8,000+ words)

Contents:
- Bug tracking log for all modules
- Individual bug report template
- Test validation checklists
- KEY INNOVATIONS validation steps
- Performance benchmark tables
- Test execution log template
- Regression testing guide
- Final validation checklist
- Sign-off template

**Use when**: You need to track bugs, validate fixes, or document testing progress.

### 4. RESE_QUICK_START_DEBUG.md
**Quick debugging reference** (5,000+ words)

Contents:
- 5-minute quick start
- Module-specific debugging
- Common issues and fixes
- Debug commands
- Validation examples
- Integration testing guide
- Common debugging patterns
- Quick fixes

**Use when**: You need quick answers to common problems or debugging steps.

---

## Test Structure

### Phase 1 Tests (150+ tests)

#### Φ₁.₅ Tacit Assumption Miner
**File**: `rese/tests/test_phi15.py` (618 lines)

**Test Classes**:
- `TestDataStructures`: Data structures & serialization
- `TestFailurePreprocessor`: Feature extraction (7 tests)
- `TestAnomalyDetector`: Anomaly detection (3 tests)
- `TestFailureClusterer`: DBSCAN clustering (3 tests)
- `TestAssumptionGenerator`: Assumption generation (2 tests)
- `TestConfidenceScorer`: Confidence scoring (2 tests)
- `TestParadigmShiftDetector`: Crisis detection (3 tests)
- `TestPhi15Engine`: Main engine (4 tests)
- `TestIntegration`: End-to-end pipeline (3 tests)

**Performance Targets**:
- 1K failures: <10s
- Assumption accuracy: >70%

#### Φ₂ Metacognitive Debiasing
**File**: `rese/tests/phase1/test_cognitive_biases.py` (501 lines)

**Test Classes**:
- `TestBiasDetection`: 8+ bias types (12 tests)
- `TestDebiasingStrategies`: 4 strategies (5 tests)
- `TestBiasDetectionAccuracy`: Accuracy validation (5 tests)
- `TestEdgeCases`: Edge case handling (6 tests)
- `TestRecommendationGeneration`: Recommendations (2 tests)

**Bias Types**:
- Confirmation, Overconfidence, Anchoring, Sunk Cost, Availability, Authority, Illusion of Control, Framing

#### Phase 1 Integration
**File**: `rese/tests/test_integration/test_phase1_integration.py` (619 lines)

**Test Classes**:
- `TestPhi15EndToEnd`: Systematic/diverse patterns (6 tests)
- `TestPhi15ComponentIntegration`: Component flow (6 tests)
- `TestPhi15DataFlow`: Data transformations (3 tests)
- `TestPhi15Performance`: Large datasets (2 tests)
- `TestPhi15Validation`: Accuracy validation (2 tests)

### Phase 2 Tests (150+ tests)

#### I_mech Isomorphism Validator
**File**: `rese/tests/test_imech/test_validator.py` (201 lines)

**Test Classes**:
- `TestIMechValidator`: Core validation (8 tests)
- `TestCompareDomains`: Convenience functions (2 tests)

**Performance Targets**:
- 10-node graphs: <5s
- 50-node graphs: <30s
- Isomorphism detection: >0.7 score

#### I_mech Integration
**File**: `rese/tests/test_imech/test_integration.py` (326 lines)

**Test Classes**:
- `TestFullPipeline`: Complete pipeline (5 tests)
- `TestHistoricalAnalogies`: Historical cases (4 tests)
- `TestPerformance`: Benchmarks (3 tests)

**Historical Analogies**:
- Candle → Light bulb
- Steam engine → Internal combustion
- Telegraph → Telephone

#### Ψ₃ Constraint Inversion
**File**: `rese/phase2/psi3/tests/unit/test_constraint_inverter.py` (583 lines)

**Test Classes**:
- `TestConstraint`: Data structures (6 tests)
- `TestExpression`: Expression AST (15 tests)
- `TestSyntacticPreprocessing`: Stage 1 (8 tests)
- `TestDependencyAnalysis`: Stage 2 (6 tests)
- `TestMinimalCover`: Stage 3 (3 tests)
- `TestIntegration`: End-to-end (4 tests)

**Performance Targets**:
- Hierarchical constraints: 10x+ reduction
- Average reduction: 6.6x
- Independent constraints: 1x (no reduction)

#### Ψ₂ Ontology Mapping
**Files**: `rese/tests/test_ontology_mapper/`

**Test Coverage**:
- Lexical matching
- Semantic matching
- Graph embedding
- KG validation
- Cross-domain mapping

**Performance Targets**:
- Mapping precision: >80%

---

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest rese/tests/ -v

# Run Phase 1 only
pytest rese/tests/phase1/ -v

# Run Phase 2 only
pytest rese/tests/test_imech/ -v

# Run integration tests
pytest rese/tests/test_integration/ -v

# Run with coverage
pytest rese/tests/ --cov=rese --cov-report=html
```

### Using Test Runner

```bash
# Run all phases
python run_rese_tests.py --phase all

# Run specific phase
python run_rese_tests.py --phase 1

# Verbose mode
python run_rese_tests.py --phase all --verbose

# With coverage
python run_rese_tests.py --phase all --coverage

# Debug mode (stop on first failure)
python run_rese_tests.py --phase all --debug
```

### Pytest Markers

```bash
# Run only unit tests
pytest rese/tests/ -m unit -v

# Run only integration tests
pytest rese/tests/ -m integration -v

# Run only performance tests
pytest rese/tests/ -m performance -v

# Run only Phase 1 tests
pytest rese/tests/ -m phase1 -v

# Run only Phase 2 tests
pytest rese/tests/ -m phase2 -v

# Skip slow tests
pytest rese/tests/ -m "not slow" -v
```

### Specific Test Files

```bash
# Φ₁.₅ tests
pytest rese/tests/test_phi15.py -v

# Φ₂ tests
pytest rese/tests/phase1/test_cognitive_biases.py -v

# I_mech validator
pytest rese/tests/test_imech/test_validator.py -v

# I_mech integration
pytest rese/tests/test_imech/test_integration.py -v

# Ψ₃ constraint inverter
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py -v

# Phase 1 integration
pytest rese/tests/test_integration/test_phase1_integration.py -v
```

---

## Debugging

### Enable Debug Output

```bash
# Verbose output
pytest rese/tests/test_phi15.py -v -s

# Long tracebacks
pytest rese/tests/test_phi15.py -v --tb=long

# Debug logging
pytest rese/tests/test_phi15.py -v --log-cli-level=DEBUG

# Drop into debugger on failure
pytest rese/tests/test_phi15.py --pdb
```

### Run Individual Tests

```bash
# Specific test method
pytest rese/tests/test_phi15.py::TestPhi15Engine::test_process_null_results -v

# Specific test class
pytest rese/tests/test_phi15.py::TestPhi15Engine -v

# Stop on first failure
pytest -x

# Stop on first error in file
pytest --maxfail=1
```

### Common Debugging Scenarios

#### Scenario 1: Test Fails with Import Error

```bash
# Check import paths
python -c "import sys; print('\n'.join(sys.path))"

# Add project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Scenario 2: Test Fails with Data Type Error

```bash
# Run with debugger
pytest test_file.py::test_method --pdb

# In debugger, inspect types
(pdb) p type(variable)
(pdb) p variable.dtype  # for numpy arrays
```

#### Scenario 3: Performance Test Fails

```bash
# Run with timing
pytest test_file.py::TestPerformance::test_method -v -s

# Profile slow code
python -m cProfile -o profile.stats your_test.py
python -m pstats profile.stats
# > sort cumulative
# > stats 20
```

---

## Performance Validation

### Run All Performance Tests

```bash
# Using test runner
python run_rese_tests.py --phase all --verbose

# Using pytest markers
pytest rese/tests/ -m performance -v

# Individual performance tests
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Performance -v
pytest rese/tests/test_imech/test_integration.py::TestPerformance -v
```

### Performance Targets

| Module | Operation | Target | Test Method |
|--------|-----------|--------|-------------|
| Φ₁.₅ | 1K failures | <10s | `test_large_dataset_performance` |
| Φ₁.₅ | Assumption accuracy | >70% | `test_phi15_accuracy_validation` |
| Ψ₃ | Hierarchical reduction | 10x+ | `test_full_pipeline_hierarchical` |
| Ψ₃ | Average reduction | 6.6x | `test_full_pipeline_*` |
| I_mech | 10-node graphs | <5s | `test_small_graphs_performance` |
| I_mech | 50-node graphs | <30s | `test_medium_graphs_performance` |
| I_mech | Transfer success | >80% | `test_full_pipeline_isomorphic_domains` |
| Ψ₂ | Mapping precision | >80% | Entity mapping tests |

---

## KEY INNOVATIONS Validation

### Innovation 1: Φ₁.₅ Tacit Assumption Mining

**Objective**: Mine hidden assumptions from failure patterns
**Target**: >70% accuracy on labeled data

**Validation**:
```bash
# Run accuracy validation
pytest rese/tests/test_integration/test_phase1_integration.py::TestPhi15Validation::test_phi15_accuracy_validation -v
```

### Innovation 2: Φ₂ Metacognitive Debiasing

**Objective**: Detect and mitigate cognitive biases
**Target**: Detect all 8+ bias types with >50% confidence

**Validation**:
```bash
# Test all bias types
pytest rese/tests/phase1/test_cognitive_biases.py::TestBiasDetection -v

# Test specific biases
pytest rese/tests/phase1/test_cognitive_biases.py::TestBiasDetection::test_confirmation_bias_detection -v
pytest rese/tests/phase1/test_cognitive_biases.py::TestBiasDetection::test_overconfidence_detection -v
```

### Innovation 3: Ψ₃ Constraint Inversion

**Objective**: Reduce constraint redundancy
**Target**: 6.6x average reduction

**Validation**:
```bash
# Test hierarchical constraints (should achieve 10x+ reduction)
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py::TestIntegration::test_full_pipeline_hierarchical -v

# Test independent constraints (should have 1x, no reduction)
pytest rese/phase2/psi3/tests/unit/test_constraint_inverter.py::TestIntegration::test_full_pipeline_independent -v
```

### Innovation 4: Ψ₂ Ontology Mapping

**Objective**: Align knowledge graphs across domains
**Target**: >80% precision on entity mapping

**Validation**:
```bash
# Test ontology mapping
pytest rese/tests/test_ontology_mapper/test_ontology_mapper.py -v

# Test integration
pytest rese/tests/test_ontology_mapper/test_integration.py -v
```

### Innovation 5: I_mech Isomorphism Validator

**Objective**: Transfer solutions across similar domains
**Target**: >80% successful transfer on isomorphic domains

**Validation**:
```bash
# Test isomorphic domains
pytest rese/tests/test_imech/test_integration.py::TestFullPipeline::test_full_pipeline_isomorphic_domains -v

# Test solution transfer
pytest rese/tests/test_imech/test_integration.py::TestFullPipeline::test_full_pipeline_with_solution_transfer -v

# Test historical analogies
pytest rese/tests/test_imech/test_integration.py::TestHistoricalAnalogies -v
```

---

## Test Reports

### Generate Test Report

```bash
# Run tests with JSON output
pytest rese/tests/ --json-report --json-report-file=test_report.json

# Or use test runner (generates rese_test_results.json)
python run_rese_tests.py --phase all
```

### Generate Coverage Report

```bash
# HTML coverage report
pytest rese/tests/ --cov=rese --cov-report=html

# Terminal coverage report
pytest rese/tests/ --cov=rese --cov-report=term-missing

# XML coverage (for CI)
pytest rese/tests/ --cov=rese --cov-report=xml
```

### Generate Performance Report

```bash
# Run performance tests
pytest rese/tests/ -m performance -v

# Save to file
pytest rese/tests/ -m performance -v > performance_report.txt
```

---

## Common Issues

### Issue 1: Module Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'phase1.tacit_assumption_miner'`

**Fix**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue 2: Missing Test Fixtures

**Symptom**: `Fixture 'sample_null_result' not found`

**Fix**: Ensure `conftest.py` is in the tests directory

### Issue 3: Z3 Solver Not Available

**Symptom**: Tests skip with "Z3 not available"

**Fix**:
```bash
pip install z3-solver
```

### Issue 4: NumPy Version Incompatibility

**Symptom**: `AttributeError: module 'numpy' has no attribute 'float'`

**Fix**:
```bash
pip install numpy==1.24.0
```

---

## Best Practices

### 1. Run Tests Before Committing

```bash
# Run full test suite
python run_rese_tests.py --phase all

# Ensure all tests pass before committing
```

### 2. Use Descriptive Test Names

```python
def test_phi15_should_detect_assumptions_from_systematic_failures():
    """Test that Φ₁.₅ detects assumptions from systematic failure patterns"""
    pass
```

### 3. Add Tests for New Features

```python
def test_new_feature():
    """Test new feature does what it's supposed to do"""
    # Arrange
    input_data = create_test_data()

    # Act
    result = process_data(input_data)

    # Assert
    assert result is not None
    assert result.is_valid()
```

### 4. Use Test Fixtures Effectively

```python
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return {
        "test": True,
        "value": 42
    }

def test_with_fixture(sample_data):
    """Test using fixture"""
    assert sample_data["test"] is True
```

### 5. Mock External Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked external service"""
    mock_service = Mock()
    mock_service.get_data.return_value = {"test": True}

    with patch('module.external_service', mock_service):
        result = function_using_service()
        assert result is True
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
name: RESE Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run Phase 1 tests
        run: pytest rese/tests/phase1/ -v
      - name: Run Phase 2 tests
        run: pytest rese/tests/test_imech/ -v
      - name: Generate coverage
        run: pytest rese/tests/ --cov=rese --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

---

## Contributing

### Adding New Tests

1. Create test file in appropriate directory
2. Import necessary modules
3. Create test class inheriting from unittest.TestCase or using pytest
4. Add test methods with `test_` prefix
5. Use fixtures from `conftest.py`
6. Run tests to verify they work
7. Update this README with new test coverage

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_is_being_tested>`

Example:
```python
# test_tacit_assumption_miner.py
class TestPhi15Engine:
    def test_process_null_results(self):
        pass
    def test_get_top_assumptions(self):
        pass
```

---

## Support

### Getting Help

1. Check documentation files:
   - `RESE_PHASE_DEBUG_REPORT.md` - Detailed testing info
   - `RESE_QUICK_START_DEBUG.md` - Quick debugging guide
   - `RESE_BUG_TRACKING_TEMPLATE.md` - Bug tracking template

2. Check test logs:
   ```bash
   pytest rese/tests/ -v -s 2>&1 | tee test_output.txt
   ```

3. Check module documentation:
   - Individual module README files
   - Docstrings in source code
   - Code comments

### Reporting Issues

Use bug tracking template in `RESE_BUG_TRACKING_TEMPLATE.md`:

1. Document bug with template
2. Include reproduction steps
3. Add error messages and stack traces
4. Attach test output
5. Note environment details

---

## Summary

### Test Suite Statistics

- **Total Tests**: 350+
- **Phase 1 Tests**: 150+
- **Phase 2 Tests**: 150+
- **Integration Tests**: 50+
- **Performance Tests**: 20+
- **Coverage**: TBD (run `--cov` to find out)

### Documentation

- **Main Report**: `RESE_PHASE_DEBUG_REPORT.md` (11,000+ words)
- **Bug Tracking**: `RESE_BUG_TRACKING_TEMPLATE.md` (8,000+ words)
- **Quick Start**: `RESE_QUICK_START_DEBUG.md` (5,000+ words)
- **Test Runner**: `run_rese_tests.py` (500+ lines)

### Status

**Status**: 🟢 Ready for Testing
**Version**: 1.0.0
**Last Updated**: 2025-12-31

---

## Start Testing Now!

```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
python run_rese_tests.py --phase all --verbose
```

Good luck and happy testing! 🚀

---

**Files Included**:
1. `RESE_TESTING_README.md` (this file)
2. `RESE_PHASE_DEBUG_REPORT.md` (comprehensive report)
3. `run_rese_tests.py` (test runner script)
4. `RESE_BUG_TRACKING_TEMPLATE.md` (bug tracking)
5. `RESE_QUICK_START_DEBUG.md` (quick start)

**Total Documentation**: 25,000+ words
**Total Tests**: 350+
**Status**: Ready for comprehensive testing
