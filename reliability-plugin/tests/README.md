# Reliability Plugin Test Suite

This directory contains comprehensive integration tests for the Reliability Plugin components. The test suite is designed to ensure the reliability, robustness, and correctness of all reliability layer implementations.

## Test Suite Structure

```
tests/
├── __init__.py                           # Package initialization
├── conftest.py                           # Shared pytest fixtures and utilities
├── test_lmql_adapter.py                 # LMQL adapter tests
├── test_guardrails_adapter.py           # Guardrails adapter tests
├── test_roma_adapter.py                 # ROMA adapter tests
├── test_mdap_adapter.py                 # MDAP adapter tests
├── test_unified_bridge.py               # Unified Bridge tests
└── README.md                            # This documentation
```

## Running Tests

### Prerequisites

- Python 3.8+
- pytest
- pytest-asyncio
- pytest-mock

Install dependencies:
```bash
pip install pytest pytest-asyncio pytest-mock
```

### Running All Tests

```bash
pytest reliability-plugin/tests/
```

### Running Specific Test Files

```bash
# Test only LMQL adapter
pytest reliability-plugin/tests/test_lmql_adapter.py

# Test only Guardrails adapter
pytest reliability-plugin/tests/test_guardrails_adapter.py

# Test only ROMA adapter
pytest reliability-plugin/tests/test_roma_adapter.py

# Test only MDAP adapter
pytest reliability-plugin/tests/test_mdap_adapter.py

# Test only Unified Bridge
pytest reliability-plugin/tests/test_unified_bridge.py
```

### Running Tests with Coverage

```bash
pytest reliability-plugin/tests/ --cov=reliability --cov-report=html
```

### Running Tests Verbosely

```bash
pytest reliability-plugin/tests/ -v
```

### Running Tests with Specific Markers

```bash
# Run only integration tests
pytest reliability-plugin/tests/ -m integration

# Run only unit tests
pytest reliability-plugin/tests/ -m "not integration"

# Run only performance tests
pytest reliability-plugin/tests/ -m performance
```

## Test Categories

### Unit Tests
- Test individual component initialization and configuration
- Test adapter-specific functionality
- Test error handling and edge cases
- Mock external dependencies for isolated testing

### Integration Tests
- Test coordination between reliability layers
- Test Unified Bridge orchestration
- Test batch generation capabilities
- Test real-world scenarios with actual dependencies

### Performance Tests
- Test generation coordination performance
- Test batch generation performance
- Test concurrent execution
- Test large-scale processing

### Parameterized Tests
- Test with different configurations
- Test with various retry counts
- Test with different batch sizes
- Test with different layer orders

## Test Coverage Areas

### LMQL Adapter Tests
- ✅ Initialization with default and custom configurations
- ✅ Constraint creation and validation
- ✅ Constrained generation with success and failure cases
- ✅ Structured generation with JSON validation
- ✅ Availability checks and graceful degradation
- ✅ Fallback mechanisms when LMQL is unavailable
- ✅ Error handling and retries
- ✅ Statistics tracking
- ✅ Integration with other layers
- ✅ Performance testing
- ✅ Edge cases (empty tasks, unicode, boundary values)

### Guardrails Adapter Tests
- ✅ Initialization with different validator configurations
- ✅ Validator registration and management
- ✅ Input validation with various failure scenarios
- ✅ Output validation with remediation strategies
- ✅ Batch validation for multiple outputs
- ✅ All remediation strategies: fix, reask, filter, refrain, exception
- ✅ Statistics tracking and health monitoring
- ✅ Integration with LMQL and other layers
- ✅ Performance testing with large batches
- ✅ Edge cases and error conditions

### ROMA Adapter Tests
- ✅ Dual-mode operation (core integration vs MCP fallback)
- ✅ Initialization with different layer combinations
- ✅ Core integration when ROMA is available
- ✅ MCP fallback when core is unavailable
- ✅ Solve with constraints for different task types
- ✅ Analyze with constraints for complexity assessment
- ✅ Verify with constraints for solution validation
- ✅ Critique with constraints for quality assessment
- ✅ Health checks and availability monitoring
- ✅ Statistics tracking and error handling
- ✅ Performance testing and parameterized configurations
- ✅ Integration with other reliability layers

### MDAP Adapter Tests
- ✅ Initialization with different layer combinations
- ✅ Dual-mode operation (core integration vs MCP fallback)
- ✅ Vote validation with various scenarios
- ✅ Core integration with LMQL constraints and Guardrails validation
- ✅ MCP fallback when core is unavailable
- ✅ Solve with validation for different thresholds
- ✅ Statistics tracking and persistence
- ✅ Health checks and availability monitoring
- ✅ Error handling and exception management
- ✅ Performance testing and concurrent validation
- ✅ Integration with other reliability layers
- ✅ Edge cases and boundary conditions

### Unified Bridge Tests
- ✅ Initialization with default and custom configurations
- ✅ Layer order validation and configuration
- ✅ Generation coordination across all layers
- ✅ Validation coordination across layers
- ✅ Batch generation with concurrent execution
- ✅ Graceful degradation when layers are unavailable
- ✅ Statistics tracking and health monitoring
- ✅ Error handling and retry logic
- ✅ Performance testing with large batches
- ✅ Integration with all reliability components
- ✅ Edge cases and corner scenarios
- ✅ Parameterized tests for different configurations

## Test Configuration

### Shared Fixtures (conftest.py)

The `conftest.py` file provides shared fixtures for all test modules:

- `mock_config`: Common configuration for all reliability components
- `mock_lmql_adapter`: Mock LMQL adapter with various responses
- `mock_guardrails_adapter`: Mock Guardrails adapter with validators
- `mock_roma_core`: Mock ROMA core with solvers and planners
- `mock_roma_mcp_tools`: Mock ROMA MCP tools
- `mock_mdap_core`: Mock MDAP core with solvers and validators
- `test_prompts`: Common test prompts for testing
- `test_constraints`: Common test constraints for testing
- `test_tasks`: Common test tasks for testing
- `sample_validation_failures`: Sample validation failures
- `sample_constraint_violations`: Sample constraint violations
- Parameterized fixtures for different configurations
- Context managers for environment patching

### Mocking Strategy

Tests use comprehensive mocking to ensure:

1. **Isolated Testing**: Each component is tested independently
2. **Deterministic Results**: Mocks provide consistent responses
3. **Edge Case Coverage**: Mocks simulate various failure scenarios
4. **Performance**: Fast test execution without external dependencies
5. **Environment Independence**: Tests work without real services

### Graceful Degradation

Tests verify graceful degradation when dependencies are unavailable:

- Tests skip when required modules are not installed
- Mock implementations provide fallback behavior
- Tests verify correct operation with partial component availability

## Test Design Principles

### 1. Comprehensive Coverage
- All public methods and classes are tested
- Both success and failure scenarios are covered
- Edge cases and boundary conditions are tested
- Integration points between components are tested

### 2. Clear Test Organization
- Tests are organized by component and functionality
- Test names are descriptive and follow conventions
- Each test has a clear purpose and assertion
- Related tests are grouped in logical classes

### 3. Maintainable Test Code
- Shared fixtures reduce code duplication
- Helper functions improve readability
- Clear error messages for debugging
- Modular design allows easy extension

### 4. Realistic Scenarios
- Tests use realistic prompts and constraints
- Batch processing scenarios are tested
- Error conditions mirror real-world usage
- Performance characteristics are verified

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'reliability'
   ```
   Solution: Ensure the reliability plugin is properly installed or in the Python path.

2. **Test Failures with Dependencies**
   ```
   pytest.mark.skipif not working as expected
   ```
   Solution: Check if the required dependencies are installed and available.

3. **Async Test Issues**
   ```
   RuntimeError: Event loop is closed
   ```
   Solution: Ensure pytest-asyncio is properly installed and configured.

### Debugging Tests

Run tests with debug output:
```bash
pytest reliability-plugin/tests/ -v --tb=short
```

Run tests with breakpoint debugging:
```bash
pytest reliability-plugin/tests/ --pdb
```

Run specific test with verbose output:
```bash
pytest reliability-plugin/tests/test_lmql_adapter.py::TestLMQLAdapter::test_initialization -v -s
```

## Contributing

### Adding New Tests

1. **Follow the naming convention**: `test_<component>_<functionality>`
2. **Use existing fixtures**: Leverage shared fixtures from `conftest.py`
3. **Add both success and failure cases**: Test positive and negative scenarios
4. **Include edge cases**: Test boundary conditions and error scenarios
5. **Document complex tests**: Add docstrings explaining test purpose
6. **Use parameterized tests**: For testing multiple configurations

### Test Best Practices

1. **Keep tests focused**: Each test should test one specific aspect
2. **Use meaningful assertions**: Verify expected behavior clearly
3. **Mock external dependencies**: Isolate tests from external services
4. **Test error handling**: Ensure graceful degradation and error recovery
5. **Maintain test independence**: Tests should not depend on execution order
6. **Update fixtures when needed**: Add new fixtures to `conftest.py` for shared functionality

## Continuous Integration

### GitHub Actions

The test suite is integrated with GitHub Actions for automated testing:

- **On Pull Request**: Run all tests on PR submission
- **On Push**: Run all tests on push to main branch
- **On Schedule**: Run tests nightly to catch regressions
- **On Tag**: Run tests before releases

### CI Configuration

```yaml
name: Reliability Plugin Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install pytest pytest-asyncio pytest-mock
      - name: Run tests
        run: pytest reliability-plugin/tests/
```

## Test Results and Reports

### HTML Coverage Reports

Generate coverage reports:
```bash
pytest reliability-plugin/tests/ --cov=reliability --cov-report=html
```

View the coverage report at `htmlcov/index.html`.

### JSON Reports

Generate JSON test reports:
```bash
pytest reliability-plugin/tests/ --json-report --json-report-file=test-report.json
```

### JUnit XML Reports

Generate JUnit XML reports for CI:
```bash
pytest reliability-plugin/tests/ --junitxml=test-results.xml
```

## Performance Benchmarks

### Test Execution Time

- **Unit Tests**: < 1 second for individual modules
- **Integration Tests**: < 5 seconds for coordinated tests
- **Performance Tests**: Variable based on batch size
- **Full Test Suite**: < 30 seconds total

### Memory Usage

Tests are designed to be memory-efficient:

- Mock objects are lightweight
- Test data is minimal but representative
- Batch tests simulate realistic loads without excessive memory

## Future Enhancements

### Planned Test Improvements

1. **Property-Based Testing**: Use hypothesis for property-based testing
2. **Chaos Engineering**: Test system resilience with simulated failures
3. **Load Testing**: Test with high concurrency and large datasets
4. **Contract Testing**: Verify compatibility with external dependencies
5. **Visual Testing**: Test UI components and visual outputs

### Additional Coverage Areas

1. **Security Testing**: Test security validation and sanitization
2. **Accessibility Testing**: Test accessibility compliance
3. **Localization Testing**: Test with different languages and locales
4. **Browser Testing**: Test web interface components

## Conclusion

This comprehensive test suite ensures the Reliability Plugin components are robust, reliable, and performant under various conditions. The tests cover all aspects of the plugin functionality from basic operations to complex integration scenarios.

By following this testing strategy, we maintain high code quality and ensure the plugin meets the reliability requirements for production use.