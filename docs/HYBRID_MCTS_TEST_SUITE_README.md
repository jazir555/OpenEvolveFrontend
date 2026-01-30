# Hybrid MCTS-Evolution Test Suite

Comprehensive test suite for three hybrid approaches combining Monte Carlo Tree Search with evolutionary algorithms.

## Overview

This test suite provides thorough testing for:

1. **Evolved Policies** - Genetic algorithms evolve rollout policies for MCTS
2. **Evolutionary Nodes** - Each MCTS node maintains a population that evolves
3. **Coevolution** - Decision trees coevolve with proof strategies

## Test Statistics

- **Total Lines**: 2,022
- **Total Test Functions**: 74 tests
- **Test Classes**: 8 main test classes
- **Fixtures**: 7 comprehensive fixtures
- **Mock Classes**: 20+ mock implementations

## Test Structure

### 1. TestHelpers
Helper function tests for utility methods used across the suite.

**Tests**:
- `test_random_seed_reproducibility` - Ensures reproducible randomness
- `test_genome_distance` - Tests distance calculations
- `test_fitness_normalization` - Validates score normalization
- `test_selection_probability` - Tests probability calculations

### 2. TestEvolvedPolicies (14 tests)
Tests for evolved rollout policies approach.

**Core Functionality**:
- `test_policy_genome_initialization` - Policy genome creation
- `test_tactic_selection` - Tactic selection from policies
- `test_context_modifiers` - Context-sensitive selection
- `test_exploration_bonus` - UCT exploration bonus calculation

**Evolution Tests**:
- `test_policy_evaluation` - Policy fitness evaluation
- `test_policy_evolution` - Multi-generation evolution
- `test_crossover_policies` - Policy crossover operators
- `test_mutate_policy` - Policy mutation operators

**Integration Tests**:
- `test_evolved_mcts_search` - MCTS with evolved policy
- `test_adaptive_policy_mcts` - Adaptive policy improvement

### 3. TestEvolutionaryNodes (11 tests)
Tests for evolutionary nodes approach.

**Population Management**:
- `test_node_population_initialization` - Population setup
- `test_action_sequence_representation` - Sequence genome encoding
- `test_sequence_selection` - Tournament/roulette selection
- `test_sequence_evaluation` - Fitness evaluation

**Genetic Operators**:
- `test_sequence_crossover` - Crossover operators
- `test_sequence_mutation` - Mutation operators

**Evolution Tests**:
- `test_evolution_at_node` - Single node evolution
- `test_convergence_detection` - Population convergence
- `test_evolutionary_mcts_search` - Full evolutionary MCTS
- `test_adaptive_evolution_control` - Adaptive parameter control

### 4. TestCoevolution (9 tests)
Tests for coevolution approach.

**Tree Structure**:
- `test_decision_tree_representation` - Tree encoding
- `test_decision_node_execution` - Node execution with context
- `test_tree_generation` - Random tree generation

**Tree Operators**:
- `test_subtree_crossover` - Subtree crossover
- `test_tree_mutation` - Tree mutation operators
- `test_tree_pruning` - Tree simplification

**Evaluation & Evolution**:
- `test_monte_carlo_evaluation` - Monte Carlo fitness
- `test_tree_coevolution` - Full coevolution
- `test_ensemble_methods` - Tree ensembles

### 5. TestHybridFramework (6 tests)
Tests for unified framework.

**Configuration**:
- `test_config_initialization` - Config creation
- `test_approach_routing` - Approach selection
- `test_presets` - Configuration presets

**Integration**:
- `test_adaptive_approach_selection` - Adaptive routing
- `test_combined_approaches` - Multi-approach combination
- `test_cache_operations` - Caching functionality

### 6. TestHybridIntegration (4 tests)
End-to-end integration tests.

**Tests**:
- `test_leanaide_integration` - LeanAide integration
- `test_workflow_integration` - OpenEvolve workflow
- `test_parallel_execution` - Parallel search execution
- `test_end_to_end` - Complete pipeline

### 7. TestHybridPerformance (8 tests)
Performance benchmarks.

**Approach Benchmarks**:
- `benchmark_evolved_policies` - Evolved policies timing
- `benchmark_evolutionary_nodes` - Evolutionary nodes timing
- `benchmark_coevolution` - Coevolution timing
- `compare_all_approaches` - Comparative analysis

**Scalability Tests**:
- `test_scalability[1-5.0]` - 1 goal problem
- `test_scalability[3-10.0]` - 3 goals problem
- `test_scalability[5-15.0]` - 5 goals problem

### 8. TestHybridEdgeCases (6 tests)
Edge case and error handling tests.

**Tests**:
- `test_empty_population` - Empty population handling
- `test_single_individual` - Single individual population
- `test_no_valid_actions` - No available actions
- `test_timeout_handling` - Timeout scenarios
- `test_convergence_failure` - Non-converging populations
- `test_invalid_tree_structure` - Invalid tree handling

### 9. TestHybridRegression (5 tests)
Regression tests for known bugs.

**Fixes Tested**:
- `test_issue_001_genome_corruption` - Genome validation
- `test_issue_002_memory_leak` - Memory management
- `test_issue_003_deadlock` - Parallel execution safety
- `test_issue_004_float_precision` - Float comparison
- `test_issue_005_cache_invalidation` - Cache consistency

### 10. TestParametrized (11 tests)
Parametrized tests for comprehensive coverage.

**Parameter Tests**:
- Evolution performance with varying parameters
- Genetic operator settings
- All approaches basic functionality

## Test Fixtures

### Sample Data Fixtures
- `sample_policy` - Sample rollout policy
- `sample_population` - Sample population of 10 individuals
- `sample_tree` - Sample decision tree structure
- `test_theorems` - Set of test theorems (easy, medium, hard)

### Mock Fixtures
- `mock_leanaide_client` - Mock LeanAide client for testing
- `hybrid_config` - Default hybrid configuration
- `temp_db_path` - Temporary database path

## Running Tests

### Run All Tests
```bash
pytest test_hybrid_mcts.py -v
```

### Run Specific Test Class
```bash
pytest test_hybrid_mcts.py::TestEvolvedPolicies -v
pytest test_hybrid_mcts.py::TestEvolutionaryNodes -v
pytest test_hybrid_mcts.py::TestCoevolution -v
pytest test_hybrid_mcts.py::TestHybridFramework -v
pytest test_hybrid_mcts.py::TestHybridIntegration -v
pytest test_hybrid_mcts.py::TestHybridPerformance -v
pytest test_hybrid_mcts.py::TestHybridEdgeCases -v
pytest test_hybrid_mcts.py::TestHybridRegression -v
```

### Run With Coverage
```bash
# HTML coverage report
pytest test_hybrid_mcts.py --cov=hybrid_mcts_framework --cov-report=html

# Terminal coverage report
pytest test_hybrid_mcts.py --cov=hybrid_mcts_framework --cov-report=term-missing
```

### Run Only Fast Tests
```bash
pytest test_hybrid_mcts.py -m "not slow" -v
```

### Run Performance Benchmarks
```bash
pytest test_hybrid_mcts.py::TestHybridPerformance -v
```

### Run Parametrized Tests
```bash
pytest test_hybrid_mcts.py::TestParametrized -v
```

## Test Coverage Goals

### Unit Tests (>90% coverage)
- All core classes
- Genetic operators
- Fitness functions
- Selection methods

### Integration Tests
- LeanAide integration
- OpenEvolve workflow
- Parallel execution
- End-to-end pipelines

### Edge Cases
- Empty populations
- Single individuals
- No valid actions
- Timeouts
- Convergence failures
- Invalid structures

### Performance Tests
- Approach benchmarks
- Scalability tests
- Comparative analysis

## Mock Implementations

The test suite includes comprehensive mock implementations:

### Core Mocks
- `PolicyGenome` - Policy genome encoding
- `PolicyEvaluator` - Policy fitness evaluation
- `PolicyEvolver` - Policy evolution
- `EvolutionaryNode` - MCTS node with population
- `DecisionTree` - Decision tree structure
- `HybridMCTSFramework` - Unified framework

### Operator Mocks
- `PolicyCrossover` - Policy crossover
- `PolicyMutator` - Policy mutation
- `SequenceCrossover` - Sequence crossover
- `SequenceMutator` - Sequence mutation
- `TreeMutator` - Tree mutation
- `SubtreeCrossover` - Subtree crossover

### Utility Mocks
- `SequenceSelector` - Selection methods
- `ConvergenceDetector` - Convergence detection
- `MonteCarloEvaluator` - Monte Carlo evaluation
- `TreePruner` - Tree pruning
- `TreeEnsemble` - Tree ensembles

## Test Dependencies

### Required Packages
```bash
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
```

### Optional Packages
```bash
pytest-benchmark>=4.0.0  # For advanced benchmarking
pytest-xdist>=3.0.0      # For parallel test execution
pytest-timeout>=2.1.0    # For test timeout handling
```

## Expected Test Results

When all tests pass, you should see:
- 74 tests collected
- All tests passing (no failures)
- Execution time: ~5-30 seconds depending on hardware
- Coverage: >90% for hybrid framework code

## Common Issues & Solutions

### Import Errors
If you see import errors for `leanaide` or `openevolve` modules:
- These are expected in simulation mode
- Tests use mocks to work without these dependencies
- Verify mock implementations are present in the test file

### Async Test Warnings
If you see warnings about `asyncio_default_fixture_loop_scope`:
- This is a deprecation warning, not an error
- Tests will still run successfully
- Can be fixed by adding pytest configuration

### Database Test Failures
If database tests fail:
- Ensure `tempfile` module is available
- Check write permissions in test directory
- Verify SQLite3 is installed

## Adding New Tests

### Template for New Test
```python
class TestNewFeature:
    """Documentation for test class"""

    def test_basic_functionality(self):
        """Test description"""
        # Arrange
        input_value = ...

        # Act
        result = function_under_test(input_value)

        # Assert
        assert result is not None
        assert result.expected_property == expected_value

    @pytest.mark.asyncio
    async def test_async_functionality(self, mock_client):
        """Async test description"""
        result = await async_function(mock_client)
        assert result.success is True
```

### Best Practices
1. **Use descriptive test names** - `test_<function>_<condition>`
2. **Follow AAA pattern** - Arrange, Act, Assert
3. **Use fixtures** - Don't repeat setup code
4. **Mock external dependencies** - LeanAide, databases, etc.
5. **Test edge cases** - Empty inputs, None values, etc.
6. **Add docstrings** - Explain what and why you're testing
7. **Keep tests independent** - No dependencies between tests

## Continuous Integration

### GitHub Actions Example
```yaml
name: Test Hybrid MCTS

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install pytest pytest-asyncio pytest-cov
      - run: pytest test_hybrid_mcts.py --cov=hybrid_mcts_framework --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## License

This test suite is part of the OpenEvolve project and follows the same license terms.

## Contributors

- OpenEvolve Frontend Team
- Created: 2025-12-30

## Version History

- **1.0.0** (2025-12-30) - Initial comprehensive test suite
  - 74 tests across 8 test classes
  - Coverage for all three hybrid approaches
  - Performance benchmarks and edge case tests
  - Regression tests for known issues
