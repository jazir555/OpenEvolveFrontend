# Integration Tests - Quick Start Guide

**Version:** 2.0
**Last Updated:** January 30, 2026

---

## Overview

This directory contains comprehensive integration test suites for:

1. **Knowledge Engine Integration** (`test_knowledge_engine_evolution_integration.py`)
   - 45 tests validating knowledge extraction, storage, retrieval, and learning

2. **Unified Evolution Engine** (`test_unified_evolution_engine.py`)
   - 40+ tests validating the complete unified evolutionary optimization pipeline
   - Tests all 6 domains, all 5 evolution modes, gauntlet integration, and learning loops

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-mock pytest-html

# Install OpenEvolve (if not already installed)
pip install -e openevolve/
```

### 2. Run All Tests

```bash
# Run all knowledge engine integration tests
pytest tests/integration/test_knowledge_engine_evolution_integration.py -v

# Run all unified evolution engine tests
pytest tests/integration/test_unified_evolution_engine.py -v

# Run all integration tests
pytest tests/integration/ -v
```

### 3. View Results

You should see output like:
```
======================== test session starts =========================
collected 45 items

test_knowledge_engine_evolution_integration.py::TestLoongFlowKnowledgeExtraction::test_extract_complete_pes_run PASSED
test_knowledge_engine_evolution_integration.py::TestLoongFlowKnowledgeExtraction::test_planning_strategy_extraction PASSED
...
======================== 45 passed in 2.34s =========================
```

---

## Running Specific Tests

### By Test Class

```bash
# Knowledge Engine Tests - LoongFlow extraction
pytest tests/integration/test_knowledge_engine_evolution_integration.py::TestLoongFlowKnowledgeExtraction -v

# Knowledge Engine Tests - Performance
pytest tests/integration/test_knowledge_engine_evolution_integration.py::TestKnowledgeEnginePerformance -v

# Unified Engine Tests - Strategy Selection
pytest tests/integration/test_unified_evolution_engine.py::TestUnifiedEvolutionEngine::test_strategy_selection_expensive_evaluations -v

# Unified Engine Tests - All Domains
pytest tests/integration/test_unified_evolution_engine.py -k "test_all_domains" -v

# Unified Engine Tests - Performance
pytest tests/integration/test_unified_evolution_engine.py -m "performance" -v
```

### By Test Method

```bash
# Run single test
pytest tests/integration/test_knowledge_engine_evolution_integration.py::TestLoongFlowKnowledgeExtraction::test_extract_complete_pes_run -v
```

### By Pattern

```bash
# Run all tests with "performance" in the name
pytest tests/integration/test_knowledge_engine_evolution_integration.py -k "performance" -v

# Run all tests with "extraction" in the name
pytest tests/integration/test_knowledge_engine_evolution_integration.py -k "extraction" -v
```

---

## Generating Reports

### Coverage Report

```bash
# Generate HTML coverage report
pytest tests/integration/test_knowledge_engine_evolution_integration.py \
    --cov=knowledge_engine/integrations/loongflow_integration \
    --cov-report=html \
    --cov-report=term-missing

# View in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### HTML Test Report

```bash
# Generate detailed HTML report
pytest tests/integration/test_knowledge_engine_evolution_integration.py \
    --html=report.html \
    --self-contained-html

# View in browser
open report.html
```

### JSON Report

```bash
# Generate JSON report for CI/CD
pytest tests/integration/test_knowledge_engine_evolution_integration.py \
    --json-report=report.json \
    --json-report-summary
```

---

## Common Workflows

### Development Workflow

```bash
# 1. Run tests quickly (parallel)
pytest tests/integration/test_knowledge_engine_evolution_integration.py -n auto -q

# 2. If tests fail, run with verbose output
pytest tests/integration/test_knowledge_engine_evolution_integration.py -vv

# 3. Run specific failing test
pytest tests/integration/test_knowledge_engine_evolution_integration.py::TestClass::test_method -vv

# 4. Run with coverage
pytest tests/integration/test_knowledge_engine_evolution_integration.py --cov
```

### Pre-commit Workflow

```bash
# Run all tests with coverage
pytest tests/integration/test_knowledge_engine_evolution_integration.py \
    --cov=knowledge_engine \
    --cov-report=term-missing \
    -v

# Check for coverage >80%
# If coverage is too low, add more tests
```

### CI/CD Workflow

```bash
# Run tests with all reports
pytest tests/integration/test_knowledge_engine_evolution_integration.py \
    --cov=knowledge_engine \
    --cov-report=xml \
    --cov-report=term-missing \
    --html=report.html \
    --junitxml=report.xml \
    -v

# Check exit code
echo $?  # 0 = all tests passed
```

---

## Troubleshooting

### Tests Won't Run

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
# Install the package in development mode
pip install -e .
```

### Async Tests Fail

**Problem:** `RuntimeError: Event loop is closed`

**Solution:**
```bash
# Install pytest-asyncio
pip install pytest-asyncio
```

### Slow Tests

**Problem:** Tests take too long

**Solution:**
```bash
# Run in parallel
pip install pytest-xdist
pytest tests/integration/test_knowledge_engine_evolution_integration.py -n auto
```

---

## Test Structure

### Knowledge Engine Integration Tests (45 tests)

1. **LoongFlow Knowledge Extraction** (10 tests)
   - Complete PES extraction
   - Individual artifact types
   - Storage integration
   - Error handling

2. **Knowledge Storage & Retrieval** (4 tests)
   - Artifact storage
   - Query functionality
   - Search capabilities

3. **Dual-Run Analysis** (4 tests)
   - OpenEvolve vs LoongFlow
   - Performance comparison
   - Winner identification

4. **Strategy Recommendation** (4 tests)
   - AI-powered recommendations
   - Historical learning
   - Confidence scoring

5. **Learning Loop** (4 tests)
   - Single run learning
   - Multi-run accumulation
   - Recommendation improvement

6. **Cross-Domain Transfer** (4 tests)
   - Knowledge reuse
   - Domain adaptation
   - Relevance scoring

7. **Temporal Evolution** (4 tests)
   - Time tracking
   - Historical queries
   - Obsolescence detection

8. **Performance** (4 tests)
   - Query speed (<100ms)
   - Storage speed (<200ms)
   - Scalability (1000+ artifacts)

9. **Error Handling** (7 tests)
   - Invalid inputs
   - Edge cases
   - Boundary conditions

10. **Full Pipeline** (4 tests)
    - End-to-end integration
    - Multiple runs
    - Error recovery

### Unified Evolution Engine Tests (40+ tests)

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

3. **Knowledge Extraction & Memory Fusion** (5 tests)
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

5. **Cross-Domain Knowledge Transfer** (4 tests)
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

8. **Error Handling & Recovery** (4 tests)
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

---

## Unified Evolution Engine Test Details

### Test File

**Location:** `tests/integration/test_unified_evolution_engine.py`

**Purpose:** Comprehensive integration tests for the unified evolutionary optimization pipeline.

### What Gets Validated

1. **LoongFlow Knowledge Extraction** (10 tests)
   - Complete PES extraction
   - Individual artifact types
   - Storage integration
   - Error handling

2. **Knowledge Storage & Retrieval** (4 tests)
   - Artifact storage
   - Query functionality
   - Search capabilities

3. **Dual-Run Analysis** (4 tests)
   - OpenEvolve vs LoongFlow
   - Performance comparison
   - Winner identification

4. **Strategy Recommendation** (4 tests)
   - AI-powered recommendations
   - Historical learning
   - Confidence scoring

5. **Learning Loop** (4 tests)
   - Single run learning
   - Multi-run accumulation
   - Recommendation improvement

6. **Cross-Domain Transfer** (4 tests)
   - Knowledge reuse
   - Domain adaptation
   - Relevance scoring

7. **Temporal Evolution** (4 tests)
   - Time tracking
   - Historical queries
   - Obsolescence detection

8. **Performance** (4 tests)
   - Query speed (<100ms)
   - Storage speed (<200ms)
   - Scalability (1000+ artifacts)

9. **Error Handling** (7 tests)
   - Invalid inputs
   - Edge cases
   - Boundary conditions

10. **Full Pipeline** (4 tests)
    - End-to-end integration
    - Multiple runs
    - Error recovery

---

## Understanding Test Results

### Pass/Fail

- ✅ **PASSED** - Test succeeded
- ❌ **FAILED** - Test failed with assertion or error
- ⚠️ **SKIPPED** - Test skipped (missing dependencies, etc.)
- 🔄 **XFAILED** - Expected to fail (known issue)
- ❌ **XPASS** - Expected to fail but passed

### Coverage

- **Lines:** Percentage of code lines executed
- **Branches:** Percentage of if/else branches taken
- **Target:** >80% coverage

### Performance

- **Query:** <100ms for 1000 artifacts
- **Storage:** <200ms per artifact batch
- **Total:** All tests should complete in <5 seconds

---

## Next Steps

### For Developers

1. Read the full testing guide: `docs/knowledge_engine/KNOWLEDGE_ENGINE_TESTING.md`
2. Explore test fixtures: `tests/fixtures/evolution_test_data.py`
3. Write new tests for your features

### For DevOps

1. Set up CI/CD pipeline using provided GitHub Actions workflow
2. Configure coverage reporting (Codecov, Coveralls)
3. Set up test result notifications

### For QA

1. Run full test suite before releases
2. Review coverage reports
3. Add domain-specific test cases

---

## Getting Help

### Documentation

- **Full Guide:** `docs/knowledge_engine/KNOWLEDGE_ENGINE_TESTING.md`
- **Roadmap:** `docs/knowledge_engine/COMPREHENSIVE_INTEGRATION_ROADMAP.md`
- **Architecture:** `docs/knowledge_engine/comprehensive_documentation.md`

### Commands

```bash
# See all pytest options
pytest --help

# See available fixtures
pytest --fixtures

# See available markers
pytest --markers
```

### Issues

If tests are failing unexpectedly:

1. Check you have the latest dependencies: `pip install -e .`
2. Clean test cache: `pytest --cache-clear`
3. Run with verbose output: `-vv`
4. Check for CI/CD issues in GitHub Actions

---

## Summary

### Knowledge Engine Integration Tests (45 tests)

This test suite validates:

✅ Knowledge extraction from LoongFlow and OpenEvolve
✅ Storage in graph, vector, and document databases
✅ Retrieval via semantic search and metadata filtering
✅ Dual-run performance comparison
✅ AI-powered strategy recommendations
✅ Continuous learning and improvement
✅ Cross-domain knowledge transfer
✅ Temporal knowledge evolution
✅ System performance and scalability
✅ Robust error handling

### Unified Evolution Engine Tests (40+ tests)

This test suite validates:

✅ Strategy selection across all 5 modes (PES, QD, MO, Adversarial, Standard)
✅ Evolution execution for all 6 domains
✅ Knowledge extraction and memory fusion
✅ 3-round gauntlet integration (AI → Red Team → Gold Team)
✅ Cross-domain knowledge transfer
✅ Learning loops across multiple runs
✅ Error handling and graceful recovery
✅ Performance benchmarks (70-80% improvement target)
✅ End-to-end workflows (batch, concurrent, iterative)

**Total:** 85+ comprehensive integration tests

---

## Quick Start Commands

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run knowledge engine tests
pytest tests/integration/test_knowledge_engine_evolution_integration.py -v

# Run unified evolution engine tests
pytest tests/integration/test_unified_evolution_engine.py -v

# Run specific test category
pytest tests/integration/test_unified_evolution_engine.py -k "strategy_selection" -v

# Run performance tests
pytest tests/integration/test_unified_evolution_engine.py -m "performance" -v

# Run with coverage
pytest tests/integration/ --cov=openevolve --cov-report=html
```

---

**Ready to test?** Choose your test suite and run now!
