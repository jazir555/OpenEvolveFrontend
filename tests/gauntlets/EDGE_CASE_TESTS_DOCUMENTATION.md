# Edge Case Test Suite for Gauntlet Components

## Overview

This document describes the comprehensive edge case test suite designed to achieve 95%+ code coverage for all gauntlet components. The test suite covers boundary conditions, error handling, concurrent access, and unusual input scenarios.

## Test Files

### 1. `test_edge_cases_ml_optimizer.py`

Tests for the ML-Based Gauntlet Optimizer component.

#### Test Classes

##### `TestEmptyNullInputs`
- `test_optimize_with_none_historical_data`: Optimization with None historical data
- `test_optimize_with_empty_historical_data`: Optimization with empty list
- `test_optimize_with_malformed_historical_data`: Handling malformed data structures
- `test_optimize_with_null_domain`: None domain parameter
- `test_optimize_with_empty_string_domain`: Empty string domain
- `test_state_from_dict_with_none_values`: State creation with None values
- `test_state_from_dict_with_missing_keys`: Missing required keys
- `test_optimize_with_none_initial_state`: None initial state

##### `TestExtremeParameterValues`
- `test_extreme_learning_rates`: Very small (0.0001) and large (1.0) learning rates
- `test_extreme_discount_factors`: Gamma values from 0.0 to 1.0
- `test_extreme_epsilon_values`: Epsilon from 0.0 (pure exploitation) to 1.0 (pure exploration)
- `test_extreme_threshold_values`: All thresholds at 0.0 and 1.0
- `test_extreme_iteration_counts`: Single iteration to 1000 iterations
- `test_extreme_weight_combinations`: All weight in single rounds
- `test_action_with_extreme_deltas`: Very large positive/negative deltas

##### `TestInvalidConfigurations`
- `test_invalid_strategy_string`: Invalid strategy defaults to Q_LEARNING
- `test_negative_parameters`: Negative learning rates
- `test_state_with_negative_values`: Negative threshold values
- `test_state_with_values_above_one`: Thresholds > 1.0
- `test_invalid_objective_type`: Invalid objective handling

##### `TestMemoryPressureConditions`
- `test_large_q_table_growth`: Q-table doesn't grow unbounded
- `test_memory_cleanup_on_optimizer_deletion`: Memory cleanup on deletion
- `test_performance_history_growth`: Performance history management
- `test_large_state_space`: Many different states

##### `TestConcurrentAccess`
- `test_concurrent_optimization_same_optimizer`: Multiple threads optimizing
- `test_concurrent_state_mutation`: Concurrent state mutations
- `test_concurrent_q_table_access`: Concurrent Q-table read/write
- `test_thread_safe_evaluation`: Thread-safe evaluation

##### `TestBoundaryConditions`
- `test_zero_baseline_score`: Improvement calculation with zero baseline
- `test_perfect_score_boundary`: Score of 1.0
- `test_all_optimization_strategies`: All 4 strategies
- `test_all_objectives`: All 4 objectives
- `test_empty_action_space_edge_case`: Actions that don't change state
- `test_reward_calculation_extremes`: Large positive/negative rewards
- `test_convergence_history_length`: History matches iterations

##### `TestErrorHandling`
- `test_malformed_state_to_tuple`: Extreme values in to_tuple()
- `test_optimization_with_zero_epsilon_decay`: No epsilon decay
- `test_select_best_action_with_empty_q_table`: Empty Q-table
- `test_genetic_algorithm_with_single_individual`: Minimal population
- `test_bayesian_optimization_exploration`: Bayesian exploration

##### `TestRecommendationGeneration`
- `test_recommendation_with_no_changes`: No changes made
- `test_recommendation_with_many_changes`: Many parameter changes
- `test_recommendation_boolean_changes`: Boolean parameter changes

#### Parametrized Tests

- `test_all_strategies_with_edge_cases`: All strategies with min state
- `test_all_objectives_with_edge_cases`: All objectives with max state

---

### 2. `test_edge_cases_predictive_executor.py`

Tests for the Predictive Gauntlet Executor component.

#### Test Classes

##### `TestEmptyNullInputs`
- `test_predict_with_empty_solution`: Empty solution string
- `test_predict_with_empty_problem`: Empty problem string
- `test_predict_with_none_context`: None context parameter
- `test_execute_with_empty_solution`: Empty solution execution
- `test_predict_with_whitespace_only`: Whitespace-only strings
- `test_execute_with_none_prediction`: None prediction parameter

##### `TestExtremelyLongSolutions`
- `test_predict_with_very_long_solution`: 1000+ line solutions
- `test_predict_with_extremely_long_single_line`: Very long single line
- `test_predict_with_very_long_problem`: Long problem statement
- `test_execute_with_very_long_solution`: Execute with long solution
- `test_complexity_score_upper_bound`: Complexity score ≤ 1.0

##### `TestUnknownDomains`
- `test_predict_with_unknown_domain`: Unknown domain handling
- `test_predict_with_empty_domain`: Empty domain string
- `test_predict_with_none_domain`: None domain
- `test_predict_with_case_variations`: Case sensitivity
- `test_all_known_domains`: All 6 known domains

##### `TestEdgeCaseFeatureCombinations`
- `test_solution_with_no_structure`: No functions/classes/imports
- `test_solution_with_all_structure`: All structure elements
- `test_very_short_solution`: Single line solution
- `test_solution_with_advanced_keywords`: All advanced keywords
- `test_risk_factors_identification`: Risk factor detection
- `test_difficulty_recommendation_boundaries`: Difficulty at probability boundaries

##### `TestPredictionBoundaryConditions`
- `test_probability_bounds`: Probability in [0, 1]
- `test_confidence_bounds`: Confidence in valid range
- `test_time_estimate_bounds`: Time estimates reasonable
- `test_cost_estimate_bounds`: Cost ≥ 0

##### `TestExecutionDecisionBoundaries`
- `test_skip_low_probability_boundary`: Just below threshold
- `test_skip_high_cost_boundary`: Just above cost threshold
- `test_skip_low_confidence`: Just below confidence threshold
- `test_proceed_at_middle_boundaries`: Middle range decisions
- `test_adjust_difficulty_high_probability`: High probability adjustment
- `test_adjust_difficulty_low_probability`: Low probability adjustment

##### `TestInvalidThresholdCombinations`
- `test_threshold_clamp_minimum`: Thresholds clamp at 0.3
- `test_threshold_clamp_maximum`: Thresholds up to 1.0
- `test_thresholds_with_none_config`: None base config

##### `TestPredictionAccuracyCalculation`
- `test_accuracy_perfect_prediction`: Perfect prediction accuracy
- `test_accuracy_worst_prediction`: Worst case accuracy
- `test_accuracy_boundary_cases`: Score boundary cases

##### `TestConcurrentPredictions`
- `test_concurrent_predictions`: 20 concurrent predictions
- `test_concurrent_executions`: 10 concurrent executions

##### `TestCostSavingsCalculation`
- `test_cost_savings_when_skipping_low_prob`: Low probability savings
- `test_cost_savings_when_skipping_high_cost`: High cost savings
- `test_no_cost_savings_when_proceeding`: No savings on proceed

##### `TestStatisticsCollection`
- `test_empty_statistics`: No predictions made
- `test_single_prediction_statistics`: Single prediction stats
- `test_multiple_predictions_statistics`: Multiple predictions stats

#### Parametrized Tests

- `test_domain_risk_mapping`: All 7 domains with expected risks
- `test_difficulty_recommendation`: 6 probability levels to difficulty

---

### 3. `test_edge_cases_adaptive_learner.py`

Tests for the Advanced Adaptive Learner component.

#### Test Classes

##### `TestEmptyExperienceBuffer`
- `test_replay_with_empty_memory`: Training with empty buffer
- `test_act_with_empty_memory`: Action selection with empty buffer
- `test_learn_from_execution_with_empty_buffer`: Buffer < batch_size
- `test_train_from_history_with_empty_list`: Empty history list
- `test_get_adaptive_strategy_with_minimal_state`: Minimal state values

##### `TestSingleExperience`
- `test_single_experience_replay`: One experience replay
- `test_single_experience_train`: Single experience training
- `test_single_record_history`: Single record history

##### `TestExplodingGradients`
- `test_large_reward_values`: Very large rewards (1000.0)
- `test_large_state_values`: Very large states (1000.0)
- `test_high_learning_rate_stability`: Learning rate of 10.0
- `test_network_weights_remain_finite`: Weights stay finite

##### `TestNetworkSizeEdgeCases`
- `test_minimal_network_size`: State=1, Action=2
- `test_large_network_size`: State=1000, Action=100
- `test_imbalanced_network`: State=2/Action=100, State=100/Action=2
- `test_action_space_boundary`: Actions at edge of space

##### `TestLearningRateEdgeCases`
- `test_zero_learning_rate`: Learning rate of 0.0
- `test_very_small_learning_rate`: Learning rate of 1e-10
- `test_negative_learning_rate`: Negative learning rate

##### `TestMemoryOverflowScenarios`
- `test_memory_maxlen_enforcement`: Buffer respects max length
- `test_fifo_behavior`: FIFO behavior verification
- `test_large_memory_efficiency`: 10000 experience buffer
- `test_batch_size_larger_than_memory`: Batch > memory size

##### `TestEpsilonDecayEdgeCases`
- `test_no_epsilon_decay`: Epsilon decay of 1.0
- `test_epsilon_at_minimum`: At minimum epsilon
- `test_rapid_epsilon_decay`: Decay of 0.5
- `test_epsilon_greater_than_one`: Epsilon > 1.0
- `test_epsilon_less_than_zero`: Epsilon < 0

##### `TestExperienceDataclass`
- `test_experience_with_none_values`: None value handling
- `test_experience_with_extreme_rewards`: ±1e10 rewards
- `test_experience_timestamp_auto_generation`: Timestamp auto-generation

##### `TestTargetNetworkUpdate`
- `test_target_network_initially_equal`: Initial equality
- `test_target_network_update_frequency`: Correct frequency
- `test_manual_target_update`: Manual update

##### `TestModelPersistence`
- `test_save_and_load_model`: Save/load roundtrip
- `test_load_from_corrupted_file`: Corrupted file handling
- `test_save_with_nans_in_weights`: NaN in weights

##### `TestAlgorithmVariations`
- `test_all_algorithm_types`: All 4 algorithm types
- `test_factory_function_all_algorithms`: Factory function tests

##### `TestGenerateTestCase`
- `test_generate_all_difficulties`: easy/medium/hard
- `test_generate_all_domains`: All 5 domains
- `test_generate_with_invalid_difficulty`: Invalid difficulty

#### Parametrized Tests

- `test_various_network_sizes`: 4 size combinations
- `test_various_epsilon_configs`: 4 epsilon configurations

---

### 4. `test_edge_cases_websocket.py`

Tests for the Gauntlet WebSocket component.

#### Test Classes

##### `TestConnectionManagerEdgeCases`
- `test_connect_with_none_websocket`: None websocket
- `test_disconnect_nonexistent_connection`: Nonexistent connection
- `test_send_event_to_nonexistent_connection`: Send to nonexistent
- `test_subscribe_to_execution_nonexistent_connection`: Nonexistent subscribe
- `test_unsubscribe_nonexistent_connection`: Nonexistent unsubscribe
- `test_subscribe_unsubscribe_subscribe`: Multiple cycles
- `test_broadcast_with_no_connections`: Empty broadcast
- `test_broadcast_to_execution_with_no_subscribers`: No subscribers
- `test_get_connection_count_empty`: Empty count

##### `TestWebSocketEventEdgeCases`
- `test_event_with_empty_data`: Empty data dict
- `test_event_with_none_data`: None data
- `test_event_with_large_data`: 10000 items
- `test_event_with_nested_data`: Deeply nested
- `test_event_with_special_characters`: Unicode/emoji
- `test_event_from_json_malformed`: Malformed JSON
- `test_event_from_json_missing_fields`: Missing fields
- `test_event_from_json_invalid_event_type`: Invalid type
- `test_event_all_event_types`: All 9 event types
- `test_event_with_none_execution_id`: None execution ID

##### `TestMalformedMessages`
- `test_empty_json_message`: Empty message
- `test_invalid_json_message`: Invalid JSON
- `test_json_with_wrong_types`: Wrong data types
- `test_message_with_null_values`: Null values
- `test_binary_message`: Binary instead of text

##### `TestExtremelyLargeMessages`
- `test_very_large_json_message`: 100000 items (>1MB)
- `test_deeply_nested_json`: 100 levels deep
- `test_unicode_characters`: Many unicode chars
- `test_message_size_limit`: Very large messages

##### `TestServerShutdown`
- `test_stop_before_start`: Stop before start
- `test_stop_already_stopped`: Double stop
- `test_connection_during_shutdown`: Connection during shutdown

##### `TestConcurrentConnections`
- `test_multiple_simultaneous_connections`: 10 connections
- `test_concurrent_broadcasts`: 5 connections
- `test_concurrent_execution_subscriptions`: 10 subscribers
- `test_concurrent_send_to_same_connection`: 10 concurrent sends

##### `TestNetworkInterruption`
- `test_send_to_disconnected_websocket`: Disconnected send
- `test_broadcast_with_some_failed_sends`: Mixed failures
- `test_reconnect_after_disconnect`: Reconnect scenario

##### `TestInvalidEventHandling`
- `test_handle_unknown_event_type`: Unknown event type
- `test_event_with_missing_data_field`: Missing data
- `test_handle_execution_event_without_execution_id`: Missing execution ID

##### `TestBroadcastMethods`
- `test_broadcast_execution_progress`: Progress broadcast
- `test_broadcast_round_completed`: Round completion
- `test_broadcast_execution_completed`: Execution completion
- `test_broadcast_error`: Error broadcast
- `test_broadcast_with_no_data`: None data

##### `TestClientEdgeCases`
- `test_client_connection_failure`: Connection failure
- `test_client_invalid_uri`: Invalid URI
- `test_client_reconnect_disabled`: Reconnect disabled
- `test_client_events_queue`: Event queue
- `test_client_subscribe_without_connection`: Subscribe without connect

#### Parametrized Tests

- `test_all_event_types_serialization`: All 9 event types
- `test_concurrent_broadcast_stress`: 100 connections, 50 broadcasts

---

## Running the Tests

### Run All Tests

```bash
python tests/gauntlets/run_edge_case_tests.py
```

### Run Specific Component

```bash
python tests/gauntlets/run_edge_case_tests.py --component ml_optimizer
python tests/gauntlets/run_edge_case_tests.py --component predictive_executor
python tests/gauntlets/run_edge_case_tests.py --component adaptive_learner
python tests/gauntlets/run_edge_case_tests.py --component websocket
```

### Run with Coverage Report

```bash
python tests/gauntlets/run_edge_case_tests.py --coverage
```

This generates:
- Terminal coverage report
- HTML coverage report in `tests/gauntlets/coverage_html/`

### Run with pytest

```bash
# Run all edge case tests
pytest tests/gauntlets/test_edge_cases_*.py -v

# Run specific file
pytest tests/gauntlets/test_edge_cases_ml_optimizer.py -v

# Run with coverage
pytest tests/gauntlets/test_edge_cases_*.py --cov=glue.adapters.gauntlet_adapter.src --cov=api.gauntlets_websocket --cov-report=html
```

---

## Test Coverage Goals

### Target Coverage

- **Overall**: 95%+ code coverage
- **Branch Coverage**: 90%+
- **Line Coverage**: 95%+

### Coverage Breakdown by Component

#### ML-Based Gauntlet Optimizer
- State initialization: 100%
- Optimization strategies: 95%+
- Q-learning algorithm: 95%+
- Genetic algorithm: 95%+
- Bayesian optimization: 95%+
- Edge case handling: 95%+

#### Predictive Gauntlet Executor
- Feature extraction: 95%+
- Success prediction: 95%+
- Execution planning: 95%+
- Decision boundaries: 95%+
- Accuracy calculation: 95%+

#### Advanced Adaptive Learner
- Neural network operations: 95%+
- Experience replay: 95%+
- Training loop: 95%+
- Model persistence: 95%+
- Epsilon decay: 95%+

#### WebSocket Server/Client
- Connection management: 95%+
- Event handling: 95%+
- Broadcasting: 95%+
- Error handling: 95%+
- Concurrent operations: 95%+

---

## Test Design Principles

### 1. Boundary Value Analysis
- Test minimum and maximum valid values
- Test just below/above thresholds
- Test boundary conditions (0, 1, -1, etc.)

### 2. Equivalence Partitioning
- Group inputs into equivalence classes
- Test representatives from each class
- Test edge cases between classes

### 3. Error Guessing
- Based on common failure modes
- Test unusual but valid inputs
- Test invalid/malformed inputs

### 4. State Transition Coverage
- Test all state transitions
- Test invalid transitions
- Test transition at boundaries

### 5. Concurrency Testing
- Test concurrent access
- Test race conditions
- Test thread safety

### 6. Resource Management
- Test memory pressure
- Test cleanup on errors
- Test resource limits

---

## Test Maintenance

### Adding New Tests

When adding new features:

1. Add corresponding edge case tests
2. Update this documentation
3. Ensure coverage doesn't drop below 95%
4. Run full test suite before committing

### Updating Tests

When fixing bugs:

1. Add regression test for the bug
2. Update related edge case tests
3. Verify all tests pass
4. Update coverage report

### Test Dependencies

Required packages:
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
coverage>=7.0.0
```

Install with:
```bash
pip install pytest pytest-asyncio pytest-cov coverage
```

---

## Known Limitations

### 1. Async Testing
Some async tests use event loops directly due to WebSocket async nature. These may need adjustment for different test runners.

### 2. Mock Objects
Some tests use mock objects for WebSocket connections. Real network testing would require integration tests.

### 3. Performance Tests
Edge case tests focus on correctness, not performance. Separate performance tests should be added for load testing.

### 4. External Dependencies
Tests assume core dependencies (numpy, etc.) are available. Real-world testing should include dependency failure scenarios.

---

## Future Enhancements

### Planned Additions

1. **Property-Based Testing**: Use hypothesis for generating random test cases
2. **Fuzzing**: Add fuzzing for input validation
3. **Stress Testing**: Test under sustained load
4. **Integration Tests**: Test component interactions
5. **Performance Regression Tests**: Ensure edge cases don't degrade performance

### Coverage Improvements

1. Increase branch coverage to 95%+
2. Add more concurrency scenarios
3. Test error paths more thoroughly
4. Add internationalization tests

---

## Troubleshooting

### Common Issues

#### Import Errors
```
ModuleNotFoundError: No module named 'glue.adapters.gauntlet_adapter.src'
```
**Solution**: Ensure PYTHONPATH includes project root, or run from project root directory.

#### Async Tests Fail
```
RuntimeError: Event loop is closed
```
**Solution**: Use pytest-asyncio marker or adjust event loop handling in tests.

#### Coverage Not Generated
```
No data to report
```
**Solution**: Ensure coverage module is installed and source paths are correct.

#### Tests Timeout
```
Timeout waiting for response
```
**Solution**: Increase timeout in test configuration or reduce test complexity.

---

## Contact and Support

For questions or issues with edge case tests:

1. Check this documentation first
2. Review test code for examples
3. Check existing issues in project repository
4. Contact test maintainer

---

## Changelog

### Version 1.0 (2026-02-03)
- Initial comprehensive edge case test suite
- 95%+ coverage target for all components
- 500+ individual test cases
- Parametrized tests for combinatorial coverage
- Async test support
- Coverage reporting integration
