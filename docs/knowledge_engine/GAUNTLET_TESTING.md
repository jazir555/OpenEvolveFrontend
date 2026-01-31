# Enhanced Gauntlet System Testing Guide

## Overview

This document describes the comprehensive testing methodology for the enhanced 3-round gauntlet system, which provides multi-stage quality control through progressive filtering and weighted scoring.

## System Architecture

### 3-Round Gauntlet Pipeline

```
Solution Input
    ↓
Round 1: LoongFlow AI Evaluation (Quick Screen)
    ├─ Score: 0-100
    ├─ Target: <30 seconds
    ├─ Threshold: Configurable (default: 0.5)
    └─ Pass? → Continue
    ↓
Round 2: Red Team Adversarial (Robustness)
    ├─ Score: 0-100
    ├─ Target: <2 minutes
    ├─ Threshold: Configurable (default: 0.7)
    └─ Pass? → Continue
    ↓
Round 3: Gold Team Consensus (Verification)
    ├─ Score: 0-100
    ├─ Target: <5 minutes
    ├─ Threshold: Configurable (default: 0.8)
    └─ Pass? → SUCCESS
    ↓
Final Score Calculation (Weighted)
    └─ Final = R1*0.2 + R2*0.3 + R3*0.5
```

## Test Suite Structure

### File Organization

```
tests/
├── gauntlets/
│   ├── test_enhanced_gauntlet_system.py  # Main test suite
│   ├── conftest.py                       # Shared fixtures
│   └── helpers.py                        # Test utilities
├── fixtures/
│   └── gauntlet_test_data.py            # Test data
├── data/
│   └── gauntlet_solutions/              # Solution samples
└── reports/
    ├── gauntlet_quality_metrics.md      # Quality report
    └── gauntlet_performance_benchmarks.md  # Performance report
```

## Test Categories

### 1. Round 1 Tests

**Purpose**: Validate LoongFlow AI evaluation as quick screen

**Tests**:
- `test_round1_evaluation_pass` - Good solutions pass
- `test_round1_evaluation_fail` - Poor solutions fail
- `test_round1_timeout_handling` - Timeout scenarios
- `test_round1_feedback_generation` - Feedback creation
- `test_round1_artifact_creation` - Artifact generation

**Acceptance Criteria**:
- Evaluation completes in <30 seconds
- Correct classification (pass/fail) at 95% confidence
- Artifacts generated for all solutions

### 2. Round 2 Tests

**Purpose**: Validate adversarial robustness testing

**Tests**:
- `test_round2_adversarial_evaluation` - Adversarial attack testing
- `test_round2_robustness_scoring` - Robustness metrics
- `test_round2_edge_case_handling` - Edge case detection
- `test_round2_vulnerability_detection` - Vulnerability finding

**Acceptance Criteria**:
- Evaluation completes in <2 minutes
- Detects 90% of known vulnerabilities
- Produces actionable feedback

### 3. Round 3 Tests

**Purpose**: Validate consensus verification

**Tests**:
- `test_round3_consensus_evaluation` - Consensus building
- `test_round3_lean4_verification` - Formal verification (math)
- `test_round3_model_agreement` - Model agreement check
- `test_round3_verification_status` - Status reporting

**Acceptance Criteria**:
- Evaluation completes in <5 minutes
- Consensus accuracy >85%
- Verification reports generated

### 4. Progressive Filtering Tests

**Purpose**: Validate early termination logic

**Tests**:
- `test_early_termination_round1` - Fail fast at R1
- `test_early_termination_round2` - Fail fast at R2
- `test_complete_all_rounds` - Complete all 3 rounds
- `test_no_early_termination_config` - Disable early exit

**Acceptance Criteria**:
- Early termination saves 60-80% time
- No false positives from early exit
- Configurable behavior

### 5. Score Aggregation Tests

**Purpose**: Validate weighted scoring

**Tests**:
- `test_weighted_score_aggregation` - Correct formula
- `test_score_normalization` - 0-1 range
- `test_final_score_calculation` - Partial completion

**Acceptance Criteria**:
- Aggregation formula: `Final = R1*w1 + R2*w2 + R3*w3`
- Scores normalized to 0-1 range
- Partial rounds handled correctly

### 6. Artifact Fusion Tests

**Purpose**: Validate artifact combination

**Tests**:
- `test_artifact_collection` - All artifacts collected
- `test_consensus_detection` - Agreement detection
- `test_conflict_detection` - Conflict identification
- `test_fused_artifacts_generation` - Fusion output

**Acceptance Criteria**:
- All round artifacts preserved
- Fusion metadata added
- Conflicts flagged

### 7. State Management Tests

**Purpose**: Validate state persistence

**Tests**:
- `test_state_initialization` - Clean start
- `test_state_transitions` - Round progression
- `test_state_persistence` - State saved
- `test_progress_reporting` - Progress tracking

**Acceptance Criteria**:
- State transitions atomic
- Progress updates accurate
- Recovery from failures

### 8. Performance Tests

**Purpose**: Validate performance targets

**Tests**:
- `test_round1_performance_target` - <30s
- `test_round2_performance_target` - <2min
- `test_round3_performance_target` - <5min
- `test_full_gauntlet_performance` - <8min total

**Acceptance Criteria**:
- 95% of evaluations meet targets
- No timeouts in normal operation
- Graceful degradation under load

### 9. Quality Metrics Tests

**Purpose**: Validate quality measures

**Tests**:
- `test_false_positive_rate` - Bad solutions that pass
- `test_false_negative_rate` - Good solutions that fail
- `test_precision_score` - TP / (TP + FP)
- `test_recall_score` - TP / (TP + FN)

**Acceptance Criteria**:
- False Positive Rate <5%
- False Negative Rate <10%
- Precision >90%
- Recall >85%

### 10. Integration Tests

**Purpose**: End-to-end validation

**Tests**:
- `test_full_pipeline_perfect_solution` - Perfect solution path
- `test_full_pipeline_failed_solution` - Failed solution path
- `test_full_pipeline_with_artifacts` - Artifact tracking
- `test_concurrent_evaluations` - Parallel execution

**Acceptance Criteria**:
- Full pipeline works end-to-end
- Artifacts tracked throughout
- Concurrent evaluations isolated

## Test Data

### Solution Categories

**Perfect Solutions** (10)
- Should pass all rounds
- Expected final score: >0.90
- Example: `perfect_portfolio_opt_001`

**Poor Solutions** (10)
- Should fail Round 1
- Expected termination: Round 1
- Example: `poor_simple_bad_001`

**Moderate Solutions** (10)
- Pass R1, fail R2
- Expected termination: Round 2
- Example: `moderate_portfolio_basic_001`

**Good Solutions** (10)
- Pass R1, R2, fail R3
- Expected termination: Round 3
- Example: `good_portfolio_intermediate_001`

**Edge Cases** (10)
- Boundary conditions
- Timeout scenarios
- Example: `edge_timeout_001`

### Configuration Profiles

**Strict**
- R1 threshold: 0.7
- R2 threshold: 0.8
- R3 threshold: 0.9
- Use case: High-stakes domains

**Balanced**
- R1 threshold: 0.5
- R2 threshold: 0.7
- R3 threshold: 0.8
- Use case: General purpose

**Lenient**
- R1 threshold: 0.3
- R2 threshold: 0.5
- R3 threshold: 0.6
- Use case: Exploration

**No Early Termination**
- Same as balanced
- Runs all rounds regardless
- Use case: Comprehensive evaluation

## Running Tests

### Run All Tests

```bash
pytest tests/gauntlets/test_enhanced_gauntlet_system.py -v
```

### Run Specific Test Class

```bash
pytest tests/gauntlets/test_enhanced_gauntlet_system.py::TestRound1Evaluation -v
```

### Run Performance Tests Only

```bash
pytest tests/gauntlets/test_enhanced_gauntlet_system.py -m performance -v
```

### Run with Coverage

```bash
pytest tests/gauntlets/test_enhanced_gauntlet_system.py --cov=openevolve.gauntlets --cov-report=html
```

### Run Specific Test

```bash
pytest tests/gauntlets/test_enhanced_gauntlet_system.py::TestProgressiveFiltering::test_early_termination_round1 -v
```

## Interpreting Results

### Test Output

```
tests/gauntlets/test_enhanced_gauntlet_system.py::TestRound1Evaluation::test_round1_evaluation_pass PASSED
tests/gauntlets/test_enhanced_gauntlet_system.py::TestRound1Evaluation::test_round1_evaluation_fail PASSED
...
=== 45 passed, 2 failed in 123.45s ===
```

### Failure Analysis

**Common Failures**:

1. **Timeout Failures**
   - Symptom: Test takes longer than target
   - Cause: System under heavy load or inefficient code
   - Action: Check resource usage, optimize code

2. **Score Mismatch**
   - Symptom: Expected score doesn't match actual
   - Cause: Evaluation logic changed
   - Action: Update test expectations or fix logic

3. **Early Termination Issues**
   - Symptom: Wrong termination round
   - Cause: Threshold misconfiguration
   - Action: Check configuration values

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Gauntlet Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio
      - name: Run gauntlet tests
        run: |
          pytest tests/gauntlets/test_enhanced_gauntlet_system.py -v --cov=openevolve.gauntlets
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Test Maintenance

### Adding New Tests

1. Create test function with descriptive name
2. Add appropriate markers (e.g., `@pytest.mark.performance`)
3. Document test purpose in docstring
4. Add acceptance criteria
5. Update this document

### Updating Test Data

1. Modify `tests/fixtures/gauntlet_test_data.py`
2. Add new `TestSolution` instances
3. Regenerate test data JSON if needed:
   ```python
   python -m tests.fixtures.gauntlet_test_data
   ```

### Refactoring Tests

1. Keep test structure consistent
2. Maintain backward compatibility
3. Update documentation
4. Run full suite after changes

## Performance Baselines

### Current Performance Targets

| Component | Target | 95th Percentile | Average |
|-----------|--------|-----------------|---------|
| Round 1   | 30s    | 28s             | 15s     |
| Round 2   | 2min   | 1m 50s          | 1m 15s  |
| Round 3   | 5min   | 4m 45s          | 3m 30s  |
| Full      | 8min   | 7m 30s          | 5m     |

### Measuring Performance

```bash
# Run performance tests with timing
pytest tests/gauntlets/test_enhanced_gauntlet_system.py -m performance --durations=10

# Generate performance report
pytest tests/gauntlets/test_enhanced_gauntlet_system.py -m performance --benchmark-only
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with import errors

**Solution**:
```bash
# Ensure Python path is correct
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install -e .
```

**Issue**: Async tests hang

**Solution**:
```bash
# Install pytest-asyncio
pip install pytest-asyncio

# Use asyncio_mode = auto
pytest tests/gauntlets/ -v --asyncio-mode=auto
```

**Issue**: Timeout in CI but not locally

**Solution**:
- Increase timeout in configuration
- Check CI resource limits
- Use more powerful CI runners

## Best Practices

1. **Isolation**: Each test should be independent
2. **Idempotency**: Tests should produce same results on rerun
3. **Speed**: Keep tests fast, use mocks where appropriate
4. **Clarity**: Use descriptive test names
5. **Documentation**: Document complex test scenarios
6. **Maintenance**: Review and update tests regularly

## References

- [COMPREHENSIVE_INTEGRATION_ROADMAP.md](./COMPREHENSIVE_INTEGRATION_ROADMAP.md) - Phase 3 details
- [GAUNTLET_COMPARISON_REPORT.md](./GAUNTLET_COMPARISON_REPORT.md) - Gauntlet capabilities
- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)

## Changelog

### Version 1.0.0 (2026-01-30)
- Initial comprehensive test suite
- 40+ tests covering all gauntlet functionality
- Performance benchmarks and quality metrics
- Full documentation
