# Evolutionary LeanAide Test Suite Documentation

## Overview

This comprehensive test suite provides complete coverage for all evolutionary LeanAide components, including evolution, decomposition, adversarial training, self-play, and strategy libraries.

## Test Suite Structure

### Main Test File: `test_leanaide_evolutionary.py`

The main test file contains all test classes organized by functionality:

1. **Evolution Tests** (`TestLeanProofStrategy`, `TestLeanProofPopulation`, `TestLeanProofMutator`, `TestLeanProofCrossover`)
   - Test initial population generation
   - Test fitness evaluation
   - Test selection methods (tournament, roulette, rank)
   - Test crossover operations (uniform, single-point, two-point, ordered)
   - Test mutation operations (tactic substitution, step insertion/deletion, goal restructuring)
   - Test convergence detection
   - Test stagnation handling

2. **Decomposition Tests** (`TestMathematicalComponentExtraction`, `TestDependencyIdentification`, `TestComplexityEstimation`, `TestSubProblemGeneration`, `TestTopologicalOrdering`, `TestParallelizationDetection`)
   - Test mathematical component extraction
   - Test dependency identification
   - Test complexity estimation
   - Test sub-problem generation
   - Test topological ordering
   - Test parallelization detection

3. **Adversarial Tests** (`TestBlueTeamAgent`, `TestRedTeamAgent`, `TestAdversarialRound`, `TestCoevolutionDynamics`)
   - Test blue team proof generation
   - Test red team critique generation
   - Test counterexample generation
   - Test adversarial rounds
   - Test co-evolution dynamics
   - Test convergence to robust proofs

4. **Self-Play Tests** (`TestSelfPlayAgent`, `TestExperienceBuffer`, `TestSelfPlayGame`, `TestSelfPlayTraining`)
   - Test self-play game execution
   - Test experience buffer storage/retrieval
   - Test agent strategy selection
   - Test reward calculation
   - Test training from buffer
   - Test improvement over iterations

5. **Strategy Library Tests** (`TestTacticLibrary`, `TestStrategySelection`, `TestStrategyMutation`, `TestStrategyCombination`, `TestSuccessTracker`)
   - Test tactic library completeness
   - Test template instantiation
   - Test strategy selection
   - Test strategy mutation
   - Test strategy combination
   - Test success rate tracking

6. **Workflow Integration Tests** (`TestStage3AEvolutionarySolution`, `TestStage3BAdversarialEvolution`, `TestMathematicalProblemDetection`, `TestGracefulFallback`, `TestEndToEndEvolutionaryWorkflow`)
   - Test Stage 3A evolutionary solution generation
   - Test Stage 3B adversarial evolution
   - Test Stage 3C evolutionary verification
   - Test Stage 5 final evolutionary verification
   - Test mathematical problem detection
   - Test graceful fallback
   - Test end-to-end evolutionary workflow

7. **Performance Tests** (`TestEvolutionPerformance`)
   - Test evolution speed
   - Test parallel evaluation performance
   - Benchmark various population sizes

8. **Edge Case Tests** (`TestEdgeCases`)
   - Test empty input handling
   - Test malformed Lean code handling
   - Test extremely long proofs
   - Test zero population size
   - Test server unavailable scenarios

## Test Runner: `run_evolutionary_tests.py`

A convenient script for running tests with various options:

### Usage

```bash
# Run all tests
python run_evolutionary_tests.py --all

# Run specific test categories
python run_evolutionary_tests.py --evolution
python run_evolutionary_tests.py --decomposition
python run_evolutionary_tests.py --adversarial
python run_evolutionary_tests.py --selfplay
python run_evolutionary_tests.py --strategy
python run_evolutionary_tests.py --workflow

# Run unit or integration tests
python run_evolutionary_tests.py --unit
python run_evolutionary_tests.py --integration

# Run fast tests only (exclude slow tests)
python run_evolutionary_tests.py --fast

# Generate coverage report
python run_evolutionary_tests.py --coverage

# Run tests in parallel
python run_evolutionary_tests.py --all --parallel

# Save results to file
python run_evolutionary_tests.py --all --save
```

### Direct pytest Usage

```bash
# Run all tests
pytest test_leanaide_evolutionary.py -v

# Run specific test categories
pytest test_leanaide_evolutionary.py -v -m evolution
pytest test_leanaide_evolutionary.py -v -m decomposition
pytest test_leanaide_evolutionary.py -v -m adversarial
pytest test_leanaide_evolutionary.py -v -m selfplay
pytest test_leanaide_evolutionary.py -v -m strategy
pytest test_leanaide_evolutionary.py -v -m workflow

# Run unit tests only
pytest test_leanaide_evolutionary.py -v -m unit

# Run integration tests only
pytest test_leanaide_evolutionary.py -v -m integration

# Run fast tests only
pytest test_leanaide_evolutionary.py -v -m "not slow"

# Run with coverage
pytest test_leanaide_evolutionary.py --cov=. --cov-report=html

# Run specific test class
pytest test_leanaide_evolutionary.py -v TestLeanProofStrategy::test_strategy_creation

# Run specific test method
pytest test_leanaide_evolutionary.py -v TestLeanProofStrategy::test_strategy_creation
```

## Pytest Markers

The test suite uses pytest markers to categorize tests:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for end-to-end workflows
- `mock`: Tests that use mocking (offline testing)
- `server`: Tests that require LeanAide server running
- `slow`: Tests that take longer to run
- `evolution`: Evolution-specific tests
- `decomposition`: Decomposition-specific tests
- `adversarial`: Adversarial-specific tests
- `selfplay`: Self-play-specific tests
- `strategy`: Strategy-specific tests
- `workflow`: Workflow integration tests

## Test Data

### Sample Theorems

The test suite includes sample theorems of varying difficulty:

```python
{
    "trivial": "True is true",
    "simple": "There are infinitely many prime numbers",
    "medium": "The square root of 2 is irrational",
    "complex": "Every natural number has a unique prime factorization",
    "algebraic": "The product of two even numbers is even"
}
```

### Sample Mathematical Problems

```python
{
    "single_step": "Prove that 2 + 2 = 4",
    "multi_step": "Prove that the sum of two even numbers is even",
    "complex": "Prove that every natural number greater than 1 has a prime divisor",
    "with_dependencies": "Prove that if n is composite, then n has a prime factor",
    "parallelizable": "Prove that for any integers a, b, c, d: if a divides b and c divides d, then ac divides bd"
}
```

## Test Fixtures

### Core Fixtures

- `test_data_dir`: Directory for test data
- `sample_theorems`: Sample theorems of varying difficulty
- `sample_mathematical_problems`: Sample mathematical problems for decomposition
- `sample_lean_tactics`: Sample Lean 4 tactics
- `temp_cache_dir`: Temporary directory for cache testing
- `mock_verification_result`: Mock verification result

## Test Categories in Detail

### 1. Evolution Tests

Test the core evolutionary proof generation system:

#### LeanProofStrategy Tests
- `test_strategy_creation`: Verify strategy object creation
- `test_tactics_sequence`: Test getting tactics as sequence
- `test_complexity_calculation`: Test complexity score calculation
- `test_elegance_calculation`: Test elegance score calculation
- `test_strategy_serialization`: Test converting strategy to dict

#### LeanProofPopulation Tests
- `test_population_size`: Verify population size
- `test_get_best_strategy`: Test getting best strategy
- `test_get_worst_strategy`: Test getting worst strategy
- `test_diversity_calculation`: Test population diversity calculation
- `test_tournament_selection`: Test tournament selection method
- `test_roulette_selection`: Test roulette wheel selection
- `test_rank_selection`: Test rank-based selection
- `test_elitism`: Test getting elite strategies
- `test_population_statistics`: Test population statistics calculation

#### LeanProofMutator Tests
- `test_tactic_substitution`: Test substituting tactics
- `test_step_insertion`: Test inserting proof steps
- `test_step_deletion`: Test deleting proof steps
- `test_goal_restructuring`: Test restructuring proof goals
- `test_lemma_introduction`: Test introducing helper lemmas
- `test_lemma_removal`: Test removing helper lemmas
- `test_reordering`: Test reordering proof steps
- `test_full_mutation`: Test full mutation process
- `test_custom_tactics`: Test mutator with custom tactics

#### LeanProofCrossover Tests
- `test_uniform_crossover`: Test uniform crossover
- `test_single_point_crossover`: Test single-point crossover
- `test_two_point_crossover`: Test two-point crossover
- `test_ordered_crossover`: Test ordered crossover
- `test_crossover_rate`: Test crossover rate affects result
- `test_crossover_with_different_lengths`: Test crossover with different length parents

### 2. Decomposition Tests

Test the mathematical problem decomposition system:

#### MathematicalComponentExtraction Tests
- `test_extract_simple_components`: Test extracting from simple problems
- `test_extract_complex_components`: Test extracting from complex problems
- `test_extract_with_dependencies`: Test extracting with dependencies

#### DependencyIdentification Tests
- `test_identify_simple_dependencies`: Test simple dependency identification
- `test_identify_complex_dependencies`: Test complex dependency identification
- `test_circular_dependency_detection`: Test circular dependency detection

#### ComplexityEstimation Tests
- `test_estimate_simple_complexity`: Test simple complexity estimation
- `test_estimate_medium_complexity`: Test medium complexity estimation
- `test_estimate_complex_complexity`: Test complex complexity estimation

#### SubProblemGeneration Tests
- `test_generate_simple_subproblems`: Test generating simple sub-problems
- `test_generate_complex_subproblems`: Test generating complex sub-problems
- `test_subproblem_dependencies`: Test sub-problem dependencies

#### TopologicalOrdering Tests
- `test_order_simple_subproblems`: Test ordering simple sub-problems
- `test_order_complex_subproblems`: Test ordering complex sub-problems

#### ParallelizationDetection Tests
- `test_detect_parallelizable_subproblems`: Test detecting parallelizable components
- `test_non_parallelizable`: Test non-parallelizable problem detection

### 3. Adversarial Tests

Test the adversarial proof improvement system:

#### BlueTeamAgent Tests
- `test_generate_initial_proof`: Test generating initial proofs
- `test_refine_proof`: Test refining proofs based on feedback
- `test_defend_proof`: Test defending proofs against attacks

#### RedTeamAgent Tests
- `test_generate_critique`: Test generating proof critiques
- `test_generate_counterexample`: Test generating counterexamples
- `test_attack_proof`: Test attacking proofs

#### AdversarialRound Tests
- `test_single_adversarial_round`: Test single adversarial round
- `test_multiple_adversarial_rounds`: Test multiple adversarial rounds
- `test_adversarial_convergence`: Test adversarial convergence

#### CoevolutionDynamics Tests
- `test_blue_red_coevolution`: Test blue-red team co-evolution
- `test_adaptive_difficulty`: Test adaptive difficulty adjustment

### 4. Self-Play Tests

Test the self-play learning system:

#### SelfPlayAgent Tests
- `test_agent_initialization`: Test agent initialization
- `test_select_strategy`: Test strategy selection
- `test_update_from_experience`: Test learning from experience

#### ExperienceBuffer Tests
- `test_store_experience`: Test storing experiences
- `test_retrieve_experience`: Test retrieving experiences
- `test_buffer_capacity`: Test buffer capacity limits

#### SelfPlayGame Tests
- `test_single_game`: Test single game execution
- `test_tournament`: Test tournament execution

#### SelfPlayTraining Tests
- `test_training_convergence`: Test training convergence
- `test_reward_calculation`: Test reward calculation

### 5. Strategy Library Tests

Test the strategy library and management:

#### TacticLibrary Tests
- `test_library_completeness`: Test essential tactics are present
- `test_tactic_metadata`: Test tactic metadata

#### StrategySelection Tests
- `test_select_strategy_simple`: Test selection for simple problems
- `test_select_strategy_complex`: Test selection for complex problems

#### StrategyMutation Tests
- `test_mutate_tactics`: Test mutating tactics
- `test_mutate_parameters`: Test mutating parameters

#### StrategyCombination Tests
- `test_combine_strategies`: Test combining multiple strategies

#### SuccessTracker Tests
- `test_track_success`: Test tracking successful strategies
- `test_track_failure`: Test tracking failed strategies
- `test_success_rate_calculation`: Test success rate calculation

### 6. Workflow Integration Tests

Test integration with OpenEvolve workflow:

#### Stage3AEvolutionarySolution Tests
- `test_evolutionary_solution_generation`: Test evolutionary solution generation

#### Stage3BAdversarialEvolution Tests
- `test_adversarial_evolution_integration`: Test adversarial evolution

#### MathematicalProblemDetection Tests
- `test_detect_mathematical_problems`: Test mathematical problem detection

#### GracefulFallback Tests
- `test_fallback_when_server_unavailable`: Test fallback without server

#### EndToEndEvolutionaryWorkflow Tests
- `test_full_evolutionary_pipeline`: Test complete pipeline

### 7. Performance Tests

Test performance characteristics:

#### EvolutionPerformance Tests
- `test_evolution_speed`: Test evolution completes in reasonable time
- `test_parallel_evaluation`: Test parallel evaluation performance

### 8. Edge Case Tests

Test error handling and edge cases:

#### EdgeCase Tests
- `test_empty_theorem`: Test empty theorem handling
- `test_malformed_lean_code`: Test malformed Lean code handling
- `test_extremely_long_proof`: Test very long proof handling
- `test_zero_population_size`: Test zero population handling

## Offline Testing

The test suite supports offline testing using mocks:

```bash
# Run mock tests (no server required)
pytest test_leanaide_evolutionary.py -v -m mock

# Or via test runner
python run_evolutionary_tests.py --mock
```

## Server Testing

For tests requiring the LeanAide server:

```bash
# Start LeanAide server first
# Then run server tests
pytest test_leanaide_evolutionary.py -v -m server

# Or via test runner
python run_evolutionary_tests.py --server
```

## Coverage Reporting

Generate coverage reports:

```bash
# HTML coverage report
pytest test_leanaide_evolutionary.py --cov=. --cov-report=html

# Terminal coverage report
pytest test_leanaide_evolutionary.py --cov=. --cov-report=term-missing

# Both
pytest test_leanaide_evolutionary.py --cov=. --cov-report=html --cov-report=term-missing

# Or via test runner
python run_evolutionary_tests.py --coverage
```

## Parallel Execution

Run tests in parallel for faster execution:

```bash
# Requires pytest-xdist
pip install pytest-xdist

# Run in parallel
pytest test_leanaide_evolutionary.py -n auto

# Or via test runner
python run_evolutionary_tests.py --all --parallel
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Evolutionary LeanAide Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov pytest-xdist
      - name: Run fast tests
        run: pytest test_leanaide_evolutionary.py -v -m "not slow"
      - name: Generate coverage
        run: pytest test_leanaide_evolutionary.py --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors

If you see import errors for evolutionary components:

```bash
# Ensure the LeanAide directory is in the path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or check if modules exist
python -c "import leanaide_evolution; print('OK')"
```

### Server Unavailable

If LeanAide server tests fail:

1. Start the LeanAide server:
   ```bash
   cd LeanAide
   python leanaide_server.py
   ```

2. Or run mock tests instead:
   ```bash
   pytest test_leanaide_evolutionary.py -v -m mock
   ```

### Slow Tests

Skip slow tests for faster development:

```bash
pytest test_leanaide_evolutionary.py -v -m "not slow"
```

## Best Practices

1. **Run fast tests during development**: Use `-m "not slow"` for quick feedback
2. **Run all tests before committing**: Use `--all` or no markers
3. **Use coverage reports**: Check coverage with `--cov`
4. **Parallel execution**: Use `-n auto` for faster test runs
5. **Mock tests for CI**: Use `--mock` to avoid server dependencies
6. **Server tests locally**: Use `--server` only when LeanAide is running

## Test Suite Architecture

```
test_leanaide_evolutionary.py
├── Evolution Tests (leanaide_evolution.py)
│   ├── LeanProofStrategy
│   ├── LeanProofPopulation
│   ├── LeanProofMutator
│   ├── LeanProofCrossover
│   └── LeanProofEvaluator
├── Decomposition Tests (leanaide_decomposition_integration.py)
│   ├── MathematicalComponentExtraction
│   ├── DependencyIdentification
│   ├── ComplexityEstimation
│   ├── SubProblemGeneration
│   ├── TopologicalOrdering
│   └── ParallelizationDetection
├── Adversarial Tests (leanaide_adversarial.py)
│   ├── BlueTeamAgent
│   ├── RedTeamAgent
│   ├── AdversarialRound
│   └── CoevolutionDynamics
├── Self-Play Tests (leanaide_selfplay.py)
│   ├── SelfPlayAgent
│   ├── ExperienceBuffer
│   ├── SelfPlayGame
│   └── SelfPlayTraining
├── Strategy Library Tests (leanaide_strategies.py)
│   ├── TacticLibrary
│   ├── StrategySelection
│   ├── StrategyMutation
│   ├── StrategyCombination
│   └── SuccessTracker
├── Workflow Integration Tests
│   ├── Stage3AEvolutionarySolution
│   ├── Stage3BAdversarialEvolution
│   ├── MathematicalProblemDetection
│   ├── GracefulFallback
│   └── EndToEndEvolutionaryWorkflow
├── Performance Tests
│   └── EvolutionPerformance
└── Edge Case Tests
    └── EdgeCases
```

## Contributing

When adding new tests:

1. Follow existing test patterns
2. Use appropriate pytest markers
3. Add docstrings explaining what is tested
4. Include both positive and negative test cases
5. Add edge cases and error handling tests
6. Update this documentation

## License

This test suite is part of the OpenEvolve project.

---

**Author**: OpenEvolve
**Created**: 2025-12-30
**Version**: 1.0.0
