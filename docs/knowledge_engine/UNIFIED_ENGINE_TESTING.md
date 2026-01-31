# Unified Evolution Engine Testing Guide

**Version:** 1.0
**Date:** January 30, 2026
**Status:** Final - Phase 4 Deliverable

---

## Table of Contents

1. [Overview](#overview)
2. [Test Architecture](#test-architecture)
3. [Test Categories](#test-categories)
4. [Running Tests](#running-tests)
5. [Test Data & Fixtures](#test-data--fixtures)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Continuous Integration](#continuous-integration)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)

---

## Overview

### Purpose

This guide documents the comprehensive integration test suite for the **Unified Evolution Engine**, which validates the complete pipeline integrating:

- **OpenEvolve** - Quality Diversity, Multi-Objective, Adversarial evolution
- **LoongFlow PES** - Plan-Execute-Summarize reasoning-guided evolution
- **Knowledge Engine** - Temporal knowledge graph with cross-run learning
- **Gauntlet System** - 3-round quality evaluation (AI → Red Team → Gold Team)
- **Domain Optimizers** - Specialized configurations for 6 domains

### Test Scope

The integration tests validate:

✅ **40+ comprehensive tests** covering the complete pipeline
✅ **All 6 domains** (finance, trading, science, engineering, pharma, web_design)
✅ **All evolutionary modes** (PES, QD, MO, Adversarial, Standard)
✅ **Complete workflow** (Strategy → Evolve → Knowledge → Gauntlet → Learn)
✅ **Cross-domain knowledge transfer**
✅ **Error handling & recovery**
✅ **Performance benchmarks**

### Success Criteria

- **Success Rate:** ≥95% (38+ of 40 tests passing)
- **Performance:** 70-80% improvement over baseline
- **Coverage:** All 6 domains, all 5 modes tested
- **Reliability:** <5% flaky test rate

---

## Test Architecture

### Directory Structure

```
tests/
├── integration/
│   ├── __init__.py
│   ├── test_unified_evolution_engine.py  # Main test suite (40+ tests)
│   ├── test_knowledge_engine_evolution_integration.py
│   ├── test_loongflow_adapter.py
│   └── README.md
├── fixtures/
│   ├── __init__.py
│   └── unified_engine_fixtures.py  # Reusable fixtures
├── data/
│   ├── __init__.py
│   └── unified_engine_test_data.py  # Sample data
├── reports/
│   └── unified_engine_performance.md  # Performance report template
└── gauntlets/
    └── (gauntlet test solutions)
```

### Test Components

#### 1. Main Test Suite (`test_unified_evolution_engine.py`)

**40+ integration tests organized into 10 categories:**

1. **Strategy Selection** (5 tests)
   - Expensive evaluations → PES
   - Multi-objective → MO
   - Diversity needed → QD
   - Safety-critical → Adversarial
   - Default fallback

2. **Evolution Execution** (6 tests)
   - PES mode execution
   - QD mode execution
   - MO mode execution
   - Adversarial mode execution
   - Standard mode execution
   - Mode comparison

3. **Knowledge Extraction** (5 tests)
   - PES artifact extraction
   - QD artifact extraction
   - Memory fusion (OpenEvolve + LoongFlow)
   - Cross-domain pattern matching
   - Strategy recommendations

4. **Gauntlet Integration** (4 tests)
   - All rounds passed
   - Early termination
   - Partial pass
   - Score aggregation

5. **Cross-Domain Transfer** (4 tests)
   - Finance → Trading
   - Engineering → Pharma
   - Similarity detection
   - Pattern validation

6. **Learning Loops** (3 tests)
   - Multiple runs
   - Strategy selector learning
   - Adaptive parameter tuning

7. **All 6 Domains** (6 tests)
   - Finance, Trading, Science
   - Engineering, Pharma, Web Design

8. **Error Handling** (4 tests)
   - Invalid problem
   - Evolution failure
   - Gauntlet timeout
   - Knowledge engine unavailable

9. **Performance Benchmarks** (4 tests)
   - General domain
   - Finance domain
   - Sample efficiency
   - Full pipeline

10. **End-to-End Workflows** (4 tests)
    - Complete workflow success
    - Batch evolution
    - Concurrent evolution
    - Iterative improvement

#### 2. Fixtures (`unified_engine_fixtures.py`)

**Reusable test fixtures for:**

- Domain configurations (6 domains with pre-configured settings)
- Problem templates (standardized problem descriptions)
- Strategy selector test cases (all decision paths)
- Mock evolution results (realistic data for all modes)
- Gauntlet round configs (3-round evaluation)
- Gauntlet test scenarios (4 quality levels)
- Knowledge artifacts (PES, QD, MO, cross-domain)
- Strategy recommendations (AI-powered)
- Performance benchmarks (target metrics)
- Mock factories (strategy selector, evolution engine)
- Test helpers (Pareto verification, efficiency calculation)

#### 3. Test Data (`unified_engine_test_data.py`)

**Comprehensive test data including:**

- Sample problems (all domains)
- Expected results (all modes)
- Cross-domain examples
- Strategy selection rules
- Performance targets
- Gauntlet criteria
- Knowledge patterns
- Learning iterations
- Error scenarios
- Batch test cases
- Performance regression tests

#### 4. Performance Report (`unified_engine_performance.md`)

**Template for performance reporting:**

- Executive summary
- Performance by domain
- Evolution mode comparison
- Knowledge engine metrics
- Gauntlet system performance
- Cross-domain transfer results
- Learning loop progress
- Error handling metrics
- Regression detection
- Recommendations

---

## Test Categories

### Category 1: Strategy Selection (5 tests)

**Purpose:** Validate AI-powered strategy selection logic.

**Tests:**
1. `test_strategy_selection_expensive_evaluations` - PES for expensive problems
2. `test_strategy_selection_multi_objective` - MO for multiple objectives
3. `test_strategy_selection_diversity_needed` - QD for exploration
4. `test_strategy_selection_safety_critical` - Adversarial for safety
5. `test_strategy_selection_default` - Default fallback to PES

**Validation:**
- Correct mode selected
- Confidence score ≥ threshold
- Reason is justified

### Category 2: Evolution Execution (6 tests)

**Purpose:** Validate all evolutionary modes execute correctly.

**Tests:**
1. `test_pes_evolution_execution` - PES with planning & early stopping
2. `test_qd_evolution_execution` - QD with archive & diversity
3. `test_mo_evolution_execution` - MO with Pareto front
4. `test_adversarial_evolution_execution` - Adversarial with robustness
5. `test_standard_evolution_execution` - Standard baseline
6. `test_evolution_modes_have_different_characteristics` - Mode comparison

**Validation:**
- Fitness ≥ threshold
- Evaluations within range
- Artifacts contain mode-specific data
- Distinct characteristics per mode

### Category 3: Knowledge Extraction (5 tests)

**Purpose:** Validate knowledge extraction and storage.

**Tests:**
1. `test_knowledge_extraction_from_pes_run` - PES patterns extracted
2. `test_knowledge_extraction_from_qd_run` - QD archive metrics extracted
3. `test_memory_fusion_openevolve_loongflow` - Both systems stored
4. `test_cross_domain_pattern_matching` - Similar patterns found
5. `test_strategy_recommendation_from_knowledge` - Recommendations generated

**Validation:**
- Artifacts stored successfully
- Correct metadata captured
- Patterns retrieved accurately
- Recommendations reasonable

### Category 4: Gauntlet Integration (4 tests)

**Purpose:** Validate 3-round gauntlet evaluation.

**Tests:**
1. `test_gauntlet_all_rounds_passed` - Excellent solution passes all
2. `test_gauntlet_early_termination` - Poor solution fails early
3. `test_gauntlet_partial_pass` - Moderate solution fails round 3
4. `test_gauntlet_score_aggregation` - Weighted scoring correct

**Validation:**
- Correct pass/fail determination
- Accurate scoring (20%, 30%, 50% weights)
- Early termination works
- Round results captured

### Category 5: Cross-Domain Transfer (4 tests)

**Purpose:** Validate knowledge transfers between domains.

**Tests:**
1. `test_knowledge_transfer_finance_to_trading` - Finance patterns help trading
2. `test_knowledge_transfer_engineering_to_pharma` - Engineering robustness → Pharma
3. `test_cross_domain_similarity_detection` - Similar problems identified
4. `test_cross_domain_pattern_validation` - Patterns validated before use

**Validation:**
- Knowledge queried across domains
- Similarity scores calculated
- Applicability validated
- Transfer improves results

### Category 6: Learning Loops (3 tests)

**Purpose:** Validate continuous learning across runs.

**Tests:**
1. `test_learning_loop_multiple_runs` - 3 runs accumulate knowledge
2. `test_strategy_selector_learning` - Strategy improves with data
3. `test_adaptive_parameter_tuning` - Parameters adapt based on performance

**Validation:**
- Knowledge accumulates
- Strategy selector uses history
- Parameters adapt
- Performance improves or maintains

### Category 7: All 6 Domains (6 tests)

**Purpose:** Validate domain-specific optimizers.

**Tests:**
1. `test_finance_domain` - Portfolio optimization
2. `test_trading_domain` - Trading strategy
3. `test_science_domain` - Experimental design
4. `test_engineering_domain` - Structural optimization
5. `test_pharma_domain` - Molecular optimization
6. `test_web_design_domain` - Conversion optimization

**Validation:**
- Correct strategy selected
- Domain-specific constraints respected
- Performance meets targets
- Artifacts captured

### Category 8: Error Handling (4 tests)

**Purpose:** Validate graceful error handling.

**Tests:**
1. `test_invalid_problem_handling` - Empty problem raises error
2. `test_evolution_failure_recovery` - Evolution crash handled
3. `test_gauntlet_timeout_handling` - Timeout managed
4. `test_knowledge_engine_unavailable` - Continues without knowledge

**Validation:**
- Errors detected
- Graceful degradation
- No data loss
- Meaningful error messages

### Category 9: Performance Benchmarks (4 tests)

**Purpose:** Validate performance targets.

**Tests:**
1. `test_general_domain_performance` - General targets met
2. `test_finance_domain_performance` - Finance targets met
3. `test_sample_efficiency_comparison` - PES better than standard
4. `test_full_pipeline_performance` - Complete pipeline fast enough

**Validation:**
- Time < target
- Evaluations < target
- Fitness ≥ target
- Efficiency ≥ target

### Category 10: End-to-End Workflows (4 tests)

**Purpose:** Validate complete real-world workflows.

**Tests:**
1. `test_complete_workflow_success` - Full pipeline succeeds
2. `test_batch_evolution_workflow` - Multiple problems processed
3. `test_concurrent_evolution_workflow` - Parallel execution works
4. `test_iterative_improvement_workflow` - Iterations improve results

**Validation:**
- All steps execute in order
- Knowledge flows between steps
- Results improve over iterations
- No data corruption

---

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Install OpenEvolve and dependencies
pip install -e openevolve/

# Optional: Install LoongFlow for full integration
git submodule update --init --recursive
pip install -e LoongFlow/
```

### Quick Start

```bash
# Run all integration tests
pytest tests/integration/test_unified_evolution_engine.py -v

# Run with coverage
pytest tests/integration/test_unified_evolution_engine.py \
    --cov=openevolve \
    --cov-report=html

# Run specific test category
pytest tests/integration/test_unified_evolution_engine.py \
    -k "strategy_selection" -v

# Run performance tests
pytest tests/integration/test_unified_evolution_engine.py \
    -m "performance" -v
```

### Advanced Options

```bash
# Run with parallel execution
pytest tests/integration/test_unified_evolution_engine.py \
    -n auto  # Requires pytest-xdist

# Run with verbose output
pytest tests/integration/test_unified_evolution_engine.py \
    -vv -s

# Run and generate report
pytest tests/integration/test_unified_evolution_engine.py \
    --html=report.html \
    --self-contained-html

# Run with markers
pytest tests/integration/test_unified_evolution_engine.py \
    -m "not slow"  # Skip slow tests

# Debug failed tests
pytest tests/integration/test_unified_evolution_engine.py \
    --pdb  # Drop into debugger on failure
```

### Environment Variables

```bash
# Set test environment variables
export OPENEVOLVE_TEST_MODE=true
export OPENEVOLVE_MOCK_LOONGFLOW=false  # Set to true if LoongFlow unavailable
export OPENEVOLVE_MOCK_KNOWLEDGE=false  # Set to true if knowledge engine unavailable
export OPENEVOLVE_VERBOSE_LOGGING=true

# Run with environment
pytest tests/integration/test_unified_evolution_engine.py -v
```

---

## Test Data & Fixtures

### Using Fixtures

```python
import pytest
from tests.fixtures.unified_engine_fixtures import (
    domain_configurations,
    problem_templates,
    mock_evolution_results
)

def test_example(domain_configurations):
    """Test using domain configuration fixture."""
    finance_config = domain_configurations["finance"]

    assert finance_config["default_mode"] == "pes"
    assert finance_config["max_evaluations"] == 50
    assert finance_config["success_threshold"] == 0.7
```

### Creating Custom Fixtures

```python
@pytest.fixture
def custom_evolution_result():
    """Custom evolution result for specific test."""
    return MockEvolutionResult(
        best_solution="def custom(): return optimal",
        fitness=0.90,
        evaluations=40,
        total_time=50.0,
        strategy_used=MockStrategyResult.pes_strategy()
    )
```

### Using Test Data

```python
from tests.data.unified_engine_test_data import (
    SAMPLE_PROBLEMS,
    EXPECTED_RESULTS,
    PERFORMANCE_TARGETS
)

def test_finance_problem():
    """Test with sample finance problem."""
    problem = SAMPLE_PROBLEMS["finance"]["portfolio_optimization"]

    assert "portfolio" in problem.lower()
    assert "sharpe" in problem.lower()
```

---

## Performance Benchmarks

### Target Metrics by Domain

| Domain | Time | Evaluations | Fitness | Efficiency |
|--------|------|-------------|---------|------------|
| General | <60s | <100 | ≥0.70 | ≥40% |
| Finance | <600s | <50 | ≥0.70 | ≥60% |
| Trading | <300s | <80 | ≥0.75 | ≥50% |
| Science | <900s | <20 | ≥0.80 | ≥70% |
| Engineering | <600s | <100 | ≥0.80 | ≥30% |
| Pharma | <600s | <100 | ≥0.85 | ≥30% |
| Web Design | <120s | <200 | ≥0.75 | ≥30% |

### Measuring Performance

```python
@pytest.mark.performance
def test_performance_benchmark():
    """Test with performance tracking."""
    import time

    start_time = time.time()

    result = await engine.evolve(
        problem="Test problem",
        domain="finance"
    )

    elapsed = time.time() - start_time

    # Assert against target
    assert elapsed < PERFORMANCE_TARGETS["finance"]["target_time"]
    assert result.evaluations < PERFORMANCE_TARGETS["finance"]["target_evals"]
    assert result.fitness >= PERFORMANCE_TARGETS["finance"]["min_fitness"]
```

### Generating Performance Reports

```bash
# Run tests with performance tracking
pytest tests/integration/test_unified_evolution_engine.py \
    -m "performance" \
    --benchmark-only \
    --benchmark-json=benchmark.json

# Generate report from JSON
python scripts/generate_performance_report.py benchmark.json
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Unified Engine Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install -e openevolve/
        pip install pytest pytest-asyncio pytest-cov

    - name: Run integration tests
      run: |
        pytest tests/integration/test_unified_evolution_engine.py \
          --cov=openevolve \
          --cov-report=xml \
          --junitxml=test-results.xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: unified-engine-tests
        name: Run unified engine tests
        entry: pytest tests/integration/test_unified_evolution_engine.py
        language: system
        pass_filenames: false
EOF

# Install hooks
pre-commit install
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'openevolve'`

**Solution:**
```bash
# Install OpenEvolve in development mode
pip install -e openevolve/

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Issue 2: LoongFlow Not Available

**Problem:** `ImportError: LoongFlow not available`

**Solution:**
```bash
# Set environment variable to use mocks
export OPENEVOLVE_MOCK_LOONGFLOW=true

# Or install LoongFlow
git submodule update --init --recursive
pip install -e LoongFlow/
```

#### Issue 3: Knowledge Engine Not Running

**Problem:** `ConnectionError: Knowledge engine unavailable`

**Solution:**
```bash
# Start knowledge engine services
docker-compose up -d neo4j qdrant mongodb

# Or use mock mode
export OPENEVOLVE_MOCK_KNOWLEDGE=true
```

#### Issue 4: Tests Timing Out

**Problem:** Tests take too long or hang

**Solution:**
```bash
# Run with timeout
pytest tests/integration/test_unified_evolution_engine.py \
    --timeout=300  # 5 minutes

# Skip slow tests
pytest tests/integration/test_unified_evolution_engine.py \
    -m "not slow"

# Use mocks instead of real execution
export OPENEVOLVE_FAST_MODE=true
```

### Debug Mode

```bash
# Enable verbose logging
export OPENEVOLVE_VERBOSE_LOGGING=true
export OPENEVOLVE_LOG_LEVEL=DEBUG

# Run with pytest debugging
pytest tests/integration/test_unified_evolution_engine.py \
    -vv \
    --log-cli-level=DEBUG \
    --pdb
```

---

## Contributing

### Adding New Tests

1. **Choose appropriate category** (1-10)
2. **Name test descriptively:** `test_<what>_<condition>_expectation`
3. **Use fixtures:** Reuse existing fixtures when possible
4. **Add docstring:** Explain what and why
5. **Include assertions:** Validate behavior
6. **Mock external dependencies:** Use mocks, not real services

Example:

```python
@pytest.mark.asyncio
async def test_new_feature(self, mock_evolution_engine):
    """
    Test that new feature works correctly.

    Validates:
    - Feature executes without errors
    - Returns expected result
    - Performance meets target
    """
    result = await mock_evolution_engine.run_evolution(
        problem="Test new feature",
        config={},
        mode="pes"
    )

    assert result.fitness >= 0.8
    assert "new_feature_data" in result.metadata
```

### Code Style

- Follow PEP 8
- Use type hints
- Keep tests focused (one assertion per test ideal)
- Use descriptive variable names
- Add comments for complex logic

### Test Review Checklist

- [ ] Test has clear docstring
- [ ] Test uses appropriate fixtures
- [ ] Test has meaningful assertions
- [ ] Test is independent (no dependencies on other tests)
- [ ] Test is fast (<5 seconds if possible)
- [ ] Test handles errors gracefully
- [ ] Test has been reviewed by at least one other person

---

## Appendix

### A. Test Metrics Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total Tests | 40+ | 40+ | ✅ |
| Test Categories | 10 | 10 | ✅ |
| Domains Covered | 6 | 6 | ✅ |
| Evolution Modes | 5 | 5 | ✅ |
| Average Test Time | <5s | <10s | ✅ |
| Success Rate | ≥95% | ≥95% | ✅ |

### B. Quick Reference

```bash
# Run all tests
pytest tests/integration/test_unified_evolution_engine.py -v

# Run specific category
pytest tests/integration/test_unified_evolution_engine.py -k "strategy" -v

# Run with coverage
pytest tests/integration/test_unified_evolution_engine.py --cov=openevolve

# Run performance tests
pytest tests/integration/test_unified_evolution_engine.py -m "performance"

# Debug mode
pytest tests/integration/test_unified_evolution_engine.py -vv -s --pdb
```

### C. Resources

- **Main Test Suite:** `tests/integration/test_unified_evolution_engine.py`
- **Fixtures:** `tests/fixtures/unified_engine_fixtures.py`
- **Test Data:** `tests/data/unified_engine_test_data.py`
- **Performance Report:** `tests/reports/unified_engine_performance.md`
- **Comprehensive Roadmap:** `docs/knowledge_engine/COMPREHENSIVE_INTEGRATION_ROADMAP.md`

---

**Document Version:** 1.0
**Last Updated:** January 30, 2026
**Maintained By:** OpenEvolve Development Team
**For Questions:** Create issue in GitHub repository
