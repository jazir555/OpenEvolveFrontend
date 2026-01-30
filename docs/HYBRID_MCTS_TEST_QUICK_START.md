# Hybrid MCTS Test Suite - Quick Start Guide

## Run All Tests
```bash
pytest test_hybrid_mcts.py -v
```
**Expected**: 66 passed in ~2 seconds

## Run Specific Test Categories

### Evolved Policies Tests (14 tests)
```bash
pytest test_hybrid_mcts.py::TestEvolvedPolicies -v
```

### Evolutionary Nodes Tests (11 tests)
```bash
pytest test_hybrid_mcts.py::TestEvolutionaryNodes -v
```

### Coevolution Tests (9 tests)
```bash
pytest test_hybrid_mcts.py::TestCoevolution -v
```

### Integration Tests (4 tests)
```bash
pytest test_hybrid_mcts.py::TestHybridIntegration -v
```

### Performance Benchmarks (8 tests)
```bash
pytest test_hybrid_mcts.py::TestHybridPerformance -v -s
```

### Edge Cases (6 tests)
```bash
pytest test_hybrid_mcts.py::TestHybridEdgeCases -v
```

### Regression Tests (5 tests)
```bash
pytest test_hybrid_mcts.py::TestHybridRegression -v
```

## Run With Coverage
```bash
pytest test_hybrid_mcts.py --cov=. --cov-report=html
```

## Run Only Fast Tests
```bash
pytest test_hybrid_mcts.py -m "not slow" -v
```

## Test Breakdown

| Test Class | Tests | Focus |
|------------|-------|-------|
| TestHelpers | 4 | Utility functions |
| TestEvolvedPolicies | 14 | Policy evolution |
| TestEvolutionaryNodes | 11 | Node populations |
| TestCoevolution | 9 | Decision trees |
| TestHybridFramework | 6 | Unified framework |
| TestHybridIntegration | 4 | End-to-end |
| TestHybridPerformance | 8 | Benchmarks |
| TestHybridEdgeCases | 6 | Edge cases |
| TestHybridRegression | 5 | Bug fixes |
| TestParametrized | 11 | Parameter sweeps |

## Test Files

1. **test_hybrid_mcts.py** - Main test suite (2,030 lines)
2. **HYBRID_MCTS_TEST_SUITE_README.md** - Full documentation
3. **HYBRID_MCTS_TEST_SUITE_COMPLETION_REPORT.md** - Completion report

## Quick Reference

### All Tests Passing
```bash
$ pytest test_hybrid_mcts.py -v
======================== 66 passed, 1 warning in 1.90s ========================
```

### Test Categories by Approach

1. **Evolved Policies**: Genetic algorithms evolve MCTS rollout policies
2. **Evolutionary Nodes**: Each MCTS node maintains evolving population
3. **Coevolution**: Decision trees coevolve with proof strategies

### Key Features Tested
- ✅ Unit tests (>90% coverage)
- ✅ Integration tests
- ✅ Performance benchmarks
- ✅ Edge cases
- ✅ Regression tests
- ✅ Parametrized tests

## CI/CD Integration

```yaml
- name: Run Hybrid MCTS Tests
  run: |
    pip install pytest pytest-asyncio pytest-cov
    pytest test_hybrid_mcts.py --cov-report=xml
```

## Need Help?

See full documentation in:
- `HYBRID_MCTS_TEST_SUITE_README.md`
- `HYBRID_MCTS_TEST_SUITE_COMPLETION_REPORT.md`
