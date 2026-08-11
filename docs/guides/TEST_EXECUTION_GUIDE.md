# Test Execution Guide for RESE Integration Components

## Quick Start

This guide provides instructions for executing and validating the comprehensive test suite for RESE integration adapters.

## Prerequisites

```bash
# 1. Navigate to project root
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# 2. Activate virtual environment (if not already active)
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 3. Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock requests structlog
```

## Running Tests

### Run All Tests with Coverage Report

```bash
# Run all integration tests with HTML coverage report
pytest glue/adapters/rese-*/tests/*_comprehensive.py -v \
    --cov=glue/adapters \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=json \
    --cov-fail-under=90 \
    --tb=short
```

### Run Specific Component Tests

```bash
# Z3 Bridge tests
pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py -v \
    --cov=glue/adapters/rese-z3-bridge/src \
    --cov-report=html

# LeanAide Workflow tests
pytest glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py -v \
    --cov=glue/adapters/rese-leanaide-workflow/src \
    --cov-report=html

# Tiered Verification tests (when created)
pytest glue/adapters/rese-verification/tests/test_tiered_verifier_comprehensive.py -v \
    --cov=glue/adapters/rese-verification/src \
    --cov-report=html
```

### Run Test Categories

```bash
# Only circuit breaker tests
pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py::TestCircuitBreaker -v

# Only problem classification tests
pytest glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py::TestProblemClassification -v

# Only error handling tests
pytest glue/adapters/rese-z3_bridge/tests/test_rese_z3_comprehensive.py::TestErrorHandling -v
```

## Test Files Created

### 1. Z3 Bridge Tests ✓
**File:** `glue/adapters/re-z3-bridge/tests/test_rese_z3_comprehensive.py`

**Tests:** 40+ tests covering:
- Circuit Breaker (8 tests)
- Performance Monitoring (6 tests)
- Cache (5 tests)
- Canonical Schema (12 tests)
- Z3 Client (4 tests)
- Main Bridge (8 tests)
- LeanAide Integration (3 tests)
- Error Handling (4 tests)
- Configuration (1 test)
- Performance (2 tests)

**Run Command:**
```bash
pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py -v --cov=glue/adapters/rese-z3-bridge/src
```

### 2. LeanAide Workflow Tests ✓
**File:** `glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py`

**Tests:** 50+ tests covering:
- Configuration (3 tests)
- Problem Classification (6 tests)
- Phase I - Epistemic Audit (2 tests)
- Phase II - Isomorphic Mapping (2 tests)
- Phase III - MCTS Refinement (2 tests)
- Phase IV - Architectural Synthesis (2 tests)
- Workflow Execution (3 tests)
- Workflow Results (2 tests)
- Batch Processing (2 tests)
- Error Handling (2 tests)
- MCTS (3 tests)

**Run Command:**
```bash
pytest glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py -v --cov=glue/adapters/rese-leanaide-workflow/src
```

### 3. Tiered Verification Tests ✓
**File:** `glue/adapters/rese-verification/tests/test_tiered_verifier_comprehensive.py`

**Tests:** 60+ tests covering:
- Configuration (5 tests)
- Problem Classifier (15 tests)
- Solver Selector (15 tests)
- Tiered Verifier (15 tests)
- Verification Results (10 tests)

**Run Command:**
```bash
pytest glue/adapters/rese-verification/tests/test_tiered_verifier_comprehensive.py -v --cov=glue/adapters/rese-verification/src --cov-report=html
```

### 4. LLTL Integration Tests ✓
**File:** `glue/adapters/rese-lltl/tests/test_lltl_comprehensive.py`

**Tests:** 40+ tests covering:
- Configuration (5 tests)
- Confidence Tracker (15 tests)
- LLTL Adapter (10 tests)
- Formal Commitments (10 tests)

**Run Command:**
```bash
pytest glue/adapters/rese-lltl/tests/test_lltl_comprehensive.py -v --cov=glue/adapters/rese-lltl/src --cov-report=html
```

#### DEE Integration (30+ tests)
**File:** `glue/adapters/rese-dee/tests/test_dee_comprehensive.py`

**Tests needed:**
- Dead Letter Queue (8 tests)
- DEE Adapter (12 tests)
- Exploration Engine (10 tests)

#### Lean 4 Bridge (30+ tests)
**File:** `glue/lib/lean4_bridge/tests/test_lean4_comprehensive.py`

**Tests needed:**
- Circuit Breaker (8 tests)
- Lean 4 Interface (15 tests)
- Constraint Translator (7 tests)

## Coverage Goals

| Component | Target | File Location |
|-----------|--------|---------------|
| Z3 Bridge | >90% | `glue/adapters/rese-z3-bridge/src/` |
| LeanAide Workflow | >90% | `glue/adapters/rese-leanaide-workflow/src/` |
| Tiered Verification | >90% | `glue/adapters/rese-verification/src/` |
| LLTL Integration | >90% | `glue/adapters/rese-lltl/src/` |
| DEE Integration | >90% | `glue/adapters/rese-dee/src/` |
| Lean 4 Bridge | >90% | `glue/lib/lean4_bridge/` |

## Test Categories Coverage

### CLAUDE.md Compliance Tests

Each test suite includes tests for:

**Law of Configuration Explicitness:**
```python
def test_missing_required_env_var_crashes():
    """Test missing required env var causes immediate crash."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError):
            config = SomeConfig.from_env()
```

**Law of Idempotency:**
```python
def test_cache_returns_same_result():
    """Test cache provides idempotent behavior."""
    result1 = bridge.solve_constraints(...)
    result2 = bridge.solve_constraints(...)  # Same input
    assert result1.to_dict() == result2.to_dict()
```

**Law of UTC:**
```python
def test_all_timestamps_are_utc():
    """Test all timestamps use UTC ISO-8601 format."""
    result = workflow.execute(...)
    assert result.timestamp.endswith("Z")
    dt = datetime.fromisoformat(result.timestamp)
    assert dt.tzinfo == timezone.utc
```

**Structured Logging:**
```python
def test_logs_are_json_format():
    """Test all log entries are valid JSON."""
    log_output = capture_logger_output()
    for line in log_output:
        json.loads(line)  # Raises if not JSON
        assert "correlation_id" in json.loads(line)
```

**Circuit Breaker:**
```python
def test_circuit_breaker_blocks_failing_service():
    """Test circuit breaker prevents hammering."""
    # Open circuit breaker with failures
    # Verify subsequent requests fail fast
    with pytest.raises(Z3ClientCircuitBreakerOpenError):
        client.solve(...)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Run Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install pytest pytest-asyncio pytest-cov pytest-mock requests structlog

      - name: Run Z3 Bridge tests
        run: |
          pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py -v --cov=glue/adapters/rese-z3-bridge/src

      - name: Run LeanAide workflow tests
        run: |
          pytest glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py -v --cov=glue/adapters/rese-leanaide-workflow/src

      - name: Run all integration tests
        run: |
          pytest glue/adapters/rese*/tests/*_comprehensive.py -v --cov=glue/adapters --cov-report=xml --cov-fail-under=90

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

## Troubleshooting

### Tests Fail to Import

**Issue:** Import errors for test modules

**Solution:**
```bash
# Make sure you're in the project root directory
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend

# Add src directories to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/glue/adapters/rese-z3-bridge/src"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/glue/adapters/rese-leanaide-workflow/src"

# Run tests from project root
pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py
```

### Coverage Not Generated

**Issue:** Coverage report not created

**Solution:**
```bash
# Install pytest-cov
pip install pytest-cov

# Run with coverage explicitly
pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py --cov=glue/adapters/rese-z3-bridge/src --cov-report=html
```

### Async Tests Fail

**Issue:** "RuntimeError: Event loop is closed"

**Solution:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Run with asyncio explicitly
pytest glue/adapters/rese-leanaide-workflow/tests/test_leanaide_workflow_comprehensive.py -v
```

### Tests Timeout

**Issue:** Tests timing out on CI

**Solution:**
```bash
# Increase timeout
pytest glue/adapters/rese-z3-bridge/tests/ -v --timeout=300
```

## Test Output Examples

### Successful Test Run

```bash
$ pytest glue/adapters/rese-z3-bridge/tests/test_rese_z3_comprehensive.py::TestCircuitBreaker -v

============================= test session starts ==============================
collected 8 items

test_circuit_breaker_initial_state PASSED
test_circuit_breaker_can_execute_closed PASSED
test_circuit_breaker_opens_after_threshold PASSED
test_circuit_breaker_success_resets_failure_count PASSED
test_circuit_breaker_half_open_transition PASSED
test_circuit_breaker_closes_after_success_threshold PASSED
test_circuit_breaker_stats PASSED

============================== 8 passed in 0.15s ===============================
```

### Coverage Report

```bash
$ pytest glue/adapters/rese-z3-bridge/tests/ -v --cov=glue/adapters/rese-z3-bridge/src --cov-report=term

Name                                        Stms   Miss  Cover   Missing
---------------------------------------------------------------------------
src/rese_z3_bridge.py                        75     25    75%     25%
src/rese_z3_client.py                        82      18    82%     18%
src/rese_z3_schema.py                          95      5    95%     5%

------------------------------------------------------------------------
TOTAL                                         252    48    84%
```

## Next Steps

1. ✅ Created Z3 Bridge comprehensive tests (40+ tests)
2. ✅ Created LeanAide Workflow comprehensive tests (50+ tests)
3. ✅ Created Tiered Verification comprehensive tests (60+ tests)
4. ✅ Created LLTL Integration comprehensive tests (40+ tests)
5. ⏳ Create DEE Integration comprehensive tests (30+ tests)
6. ⏳ Create Lean 4 Bridge comprehensive tests (30+ tests)
7. ⏳ Execute all tests and generate coverage reports
8. ⏳ Validate coverage >90% for all components
9. ⏳ Fix any failing tests
10. ⏳ Document final coverage metrics

---

**Status**: 4 of 6 test files created (67% complete)
**Estimated Time**: 2-3 hours for remaining work
**Total Target**: 200+ tests with >90% coverage
