<<<<<<< HEAD
# BubbleLabs OpenEvolve Plugin - Test Guide

## Overview

This comprehensive test suite provides complete coverage for all BubbleLabs OpenEvolve Plugin integrations, including:

- **Plugin System**: Registration, lifecycle, event bus, hot-reloading, health checks
- **LeanAide Integration**: Translation, proof generation, verification, MCTS visualization
- **Evolution Integration**: Workflow creation, adversarial testing, progress tracking
- **Knowledge Engine Integration**: Graph queries, multi-source querying, visualization
- **Maker/CrewAI Integration**: Tool creation, delegation, repository management
- **UI Components**: Parameter rendering, visualization, export/import, security

## Test Statistics

- **Total Tests**: 100+
- **Test Classes**: 10
- **Coverage Target**: 70%+
- **Test Framework**: pytest with pytest-asyncio

## Installation

### Prerequisites

```bash
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-mock
```

### Optional (for full integration testing)

```bash
pip install requests-mock moto  # For API mocking
pip install pytest-benchmark     # For performance tests
```

## Running Tests

### Run All Tests

```bash
pytest test_bubblelabs_comprehensive.py
```

### Run Specific Test Class

```bash
pytest test_bubblelabs_comprehensive.py::TestPluginSystem
pytest test_bubblelabs_comprehensive.py::TestLeanAideIntegration
pytest test_bubblelabs_comprehensive.py::TestEvolutionIntegration
```

### Run Specific Test

```bash
pytest test_bubblelabs_comprehensive.py::TestPluginSystem::test_plugin_registration
```

### Run with Verbosity

```bash
pytest test_bubblelabs_comprehensive.py -v
```

### Run with Coverage Report

```bash
pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=html
```

### Run Only Fast Tests

```bash
pytest test_bubblelabs_comprehensive.py -m "not slow"
```

### Run Only Unit Tests

```bash
pytest test_bubblelabs_comprehensive.py -m unit
```

### Run in Parallel (Faster)

```bash
pytest test_bubblelabs_comprehensive.py -n auto
```

### Run and Stop on First Failure

```bash
pytest test_bubblelabs_comprehensive.py -x
```

### Run with Detailed Output

```bash
pytest test_bubblelabs_comprehensive.py -vv -s
```

## Test Organization

### Test Classes

1. **TestPluginSystem** (12 tests)
   - Plugin registration and lifecycle
   - Event bus functionality
   - Dependency management
   - Hot-reloading
   - Health checks
   - Configuration loading and validation

2. **TestLeanAideIntegration** (10 tests)
   - Translation tasks (success, timeout, with name)
   - Proof generation (success, with pre-translated code)
   - Code verification (success, with errors)
   - Math queries
   - MCTS visualization data
   - Lean4 proof tracking
   - Concurrent requests

3. **TestEvolutionIntegration** (6 tests)
   - Workflow creation
   - Adversarial testing integration
   - Progress tracking
   - Background task management
   - Checkpoint creation and restoration

4. **TestKnowledgeEngineIntegration** (4 tests)
   - Knowledge graph queries
   - Multi-source querying
   - Visualization data generation
   - Bedrock KB integration

5. **TestMakerCrewAIIntegration** (8 tests)
   - Tool creation workflow
   - CrewAI delegation
   - Tool repository management
   - Ticket creation and updates
   - MDAP task synchronization
   - MAKER run synchronization

6. **TestUIComponents** (8 tests)
   - Parameter rendering
   - Workflow visualization
   - Export/import functionality
   - XSS protection
   - SQL injection protection
   - Parameter validation
   - Component rendering

7. **TestFullIntegration** (5 tests)
   - End-to-end workflow
   - LeanAide to Evolution pipeline
   - Knowledge Engine to Maker pipeline
   - CrewAI ticket lifecycle
   - Async workflow execution

8. **TestPerformance** (3 tests)
   - Translation performance
   - Concurrent request performance
   - Memory usage

9. **TestSecurity** (5 tests)
   - Input sanitization
   - API key protection
   - Rate limiting
   - Authentication requirements
   - Authorization checks

10. **TestErrorHandling** (5 tests)
    - Connection errors
    - Timeout errors
    - Invalid inputs
    - Retry mechanisms

11. **TestThreadSafety** (2 tests)
    - Concurrent plugin registration
    - Concurrent workflow updates

## Test Markers

Tests are marked with categories for selective execution:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (may require external services)
- `@pytest.mark.e2e`: End-to-end tests (full workflow)
- `@pytest.mark.slow`: Slow tests (performance, stress tests)
- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.requires_api`: Tests requiring API keys
- `@pytest.mark.requires_crewai`: Tests requiring CrewAI
- `@pytest.mark.requires_leanaide`: Tests requiring LeanAide

## Fixtures

### Common Fixtures

- `mock_api_key`: Mock API key for testing
- `mock_base_url`: Mock base URL for API endpoints
- `mock_workflow_state`: Mock workflow state
- `mock_sub_problem`: Mock sub-problem
- `mock_leanaide_client`: Mock LeanAide client
- `mock_crewai_client`: Mock CrewAI client
- `sample_lean_code`: Sample Lean 4 code
- `sample_theorem_text`: Sample natural language theorem
- `event_loop`: Event loop for async tests

## Writing New Tests

### Template for Unit Tests

```python
class TestNewFeature:
    """Test suite for new feature"""

    def test_basic_functionality(self, mock_fixture):
        """Test basic functionality works"""
        result = function_under_test(mock_fixture)
        assert result["success"] is True

    def test_error_handling(self, mock_fixture):
        """Test errors are handled properly"""
        with pytest.raises(ExpectedException):
            function_under_test(invalid_input)
```

### Template for Integration Tests

```python
@pytest.mark.integration
def test_integration_with_external_service(mock_api_client):
    """Test integration with external service"""
    mock_api_client.call = Mock(return_value={"status": "ok"})

    result = function_under_test(mock_api_client)

    assert result["status"] == "ok"
```

### Template for Async Tests

```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality"""
    result = await async_function()
    assert result is not None
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test BubbleLabs Plugin

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio pytest-cov
      - run: pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_bubblelabs_comprehensive.py
```

### Async Tests Not Running

Ensure pytest-asyncio is installed:

```bash
pip install pytest-asyncio
```

### Coverage Report Issues

If coverage fails to generate:

```bash
pip install pytest-cov
pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=html --no-cov-on-fail
```

### Tests Hanging

If tests hang, check for:

1. Missing `await` keywords in async tests
2. Infinite loops in test code
3. Blocking operations in async tests

Use timeout to prevent hanging:

```bash
pytest test_bubblelabs_comprehensive.py --timeout=10
```

## Test Maintenance

### When to Update Tests

- When adding new features
- When changing API interfaces
- When fixing bugs
- When refactoring code

### Test Review Checklist

- [ ] Tests are readable and self-documenting
- [ ] Tests have clear assertions
- [ ] Tests are isolated (no dependencies between tests)
- [ ] Tests use fixtures appropriately
- [ ] Tests handle errors correctly
- [ ] Tests are fast (unless marked as slow)
- [ ] Tests are marked appropriately

## Best Practices

1. **Arrange, Act, Assert (AAA) Pattern**
   ```python
   def test_example():
       # Arrange: Set up test data
       input_data = {"key": "value"}

       # Act: Call function under test
       result = function(input_data)

       # Assert: Verify expected outcome
       assert result["success"] is True
   ```

2. **Use Descriptive Test Names**
   ```python
   def test_translation_returns_valid_lean_code():  # Good
   def test_translation():  # Less descriptive
   ```

3. **Test One Thing Per Test**
   ```python
   def test_translation_and_proof_generation():  # Bad
   def test_translation_succeeds():  # Good
   def test_proof_generation_succeeds():  # Good
   ```

4. **Use Fixtures for Common Setup**
   ```python
   @pytest.fixture
   def mock_client():
       return MockClient()

   def test_with_client(mock_client):
       result = mock_client.call()
       assert result is not None
   ```

5. **Mock External Dependencies**
   ```python
   @patch('module.external_api_call')
   def test_with_mock(mock_api):
       mock_api.return_value = {"status": "ok"}
       result = function_under_test()
       assert result["status"] == "ok"
   ```

## Coverage Goals

Current coverage targets:

- **Overall**: 70%+
- **Critical Paths**: 90%+
- **Error Handling**: 80%+
- **Integration Points**: 75%+

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## Support

For questions or issues with the test suite:

1. Check this guide first
2. Review pytest documentation
3. Check existing tests for examples
4. Contact the BubbleLabs development team

---

**Last Updated**: 2026-01-03
**Test Suite Version**: 1.0.0
**Maintained By**: BubbleLabs Development Team
=======
# BubbleLabs OpenEvolve Plugin - Test Guide

## Overview

This comprehensive test suite provides complete coverage for all BubbleLabs OpenEvolve Plugin integrations, including:

- **Plugin System**: Registration, lifecycle, event bus, hot-reloading, health checks
- **LeanAide Integration**: Translation, proof generation, verification, MCTS visualization
- **Evolution Integration**: Workflow creation, adversarial testing, progress tracking
- **Knowledge Engine Integration**: Graph queries, multi-source querying, visualization
- **Maker/CrewAI Integration**: Tool creation, delegation, repository management
- **UI Components**: Parameter rendering, visualization, export/import, security

## Test Statistics

- **Total Tests**: 100+
- **Test Classes**: 10
- **Coverage Target**: 70%+
- **Test Framework**: pytest with pytest-asyncio

## Installation

### Prerequisites

```bash
pip install pytest pytest-asyncio pytest-cov pytest-xdist pytest-mock
```

### Optional (for full integration testing)

```bash
pip install requests-mock moto  # For API mocking
pip install pytest-benchmark     # For performance tests
```

## Running Tests

### Run All Tests

```bash
pytest test_bubblelabs_comprehensive.py
```

### Run Specific Test Class

```bash
pytest test_bubblelabs_comprehensive.py::TestPluginSystem
pytest test_bubblelabs_comprehensive.py::TestLeanAideIntegration
pytest test_bubblelabs_comprehensive.py::TestEvolutionIntegration
```

### Run Specific Test

```bash
pytest test_bubblelabs_comprehensive.py::TestPluginSystem::test_plugin_registration
```

### Run with Verbosity

```bash
pytest test_bubblelabs_comprehensive.py -v
```

### Run with Coverage Report

```bash
pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=html
```

### Run Only Fast Tests

```bash
pytest test_bubblelabs_comprehensive.py -m "not slow"
```

### Run Only Unit Tests

```bash
pytest test_bubblelabs_comprehensive.py -m unit
```

### Run in Parallel (Faster)

```bash
pytest test_bubblelabs_comprehensive.py -n auto
```

### Run and Stop on First Failure

```bash
pytest test_bubblelabs_comprehensive.py -x
```

### Run with Detailed Output

```bash
pytest test_bubblelabs_comprehensive.py -vv -s
```

## Test Organization

### Test Classes

1. **TestPluginSystem** (12 tests)
   - Plugin registration and lifecycle
   - Event bus functionality
   - Dependency management
   - Hot-reloading
   - Health checks
   - Configuration loading and validation

2. **TestLeanAideIntegration** (10 tests)
   - Translation tasks (success, timeout, with name)
   - Proof generation (success, with pre-translated code)
   - Code verification (success, with errors)
   - Math queries
   - MCTS visualization data
   - Lean4 proof tracking
   - Concurrent requests

3. **TestEvolutionIntegration** (6 tests)
   - Workflow creation
   - Adversarial testing integration
   - Progress tracking
   - Background task management
   - Checkpoint creation and restoration

4. **TestKnowledgeEngineIntegration** (4 tests)
   - Knowledge graph queries
   - Multi-source querying
   - Visualization data generation
   - Bedrock KB integration

5. **TestMakerCrewAIIntegration** (8 tests)
   - Tool creation workflow
   - CrewAI delegation
   - Tool repository management
   - Ticket creation and updates
   - MDAP task synchronization
   - MAKER run synchronization

6. **TestUIComponents** (8 tests)
   - Parameter rendering
   - Workflow visualization
   - Export/import functionality
   - XSS protection
   - SQL injection protection
   - Parameter validation
   - Component rendering

7. **TestFullIntegration** (5 tests)
   - End-to-end workflow
   - LeanAide to Evolution pipeline
   - Knowledge Engine to Maker pipeline
   - CrewAI ticket lifecycle
   - Async workflow execution

8. **TestPerformance** (3 tests)
   - Translation performance
   - Concurrent request performance
   - Memory usage

9. **TestSecurity** (5 tests)
   - Input sanitization
   - API key protection
   - Rate limiting
   - Authentication requirements
   - Authorization checks

10. **TestErrorHandling** (5 tests)
    - Connection errors
    - Timeout errors
    - Invalid inputs
    - Retry mechanisms

11. **TestThreadSafety** (2 tests)
    - Concurrent plugin registration
    - Concurrent workflow updates

## Test Markers

Tests are marked with categories for selective execution:

- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (may require external services)
- `@pytest.mark.e2e`: End-to-end tests (full workflow)
- `@pytest.mark.slow`: Slow tests (performance, stress tests)
- `@pytest.mark.asyncio`: Async tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.requires_api`: Tests requiring API keys
- `@pytest.mark.requires_crewai`: Tests requiring CrewAI
- `@pytest.mark.requires_leanaide`: Tests requiring LeanAide

## Fixtures

### Common Fixtures

- `mock_api_key`: Mock API key for testing
- `mock_base_url`: Mock base URL for API endpoints
- `mock_workflow_state`: Mock workflow state
- `mock_sub_problem`: Mock sub-problem
- `mock_leanaide_client`: Mock LeanAide client
- `mock_crewai_client`: Mock CrewAI client
- `sample_lean_code`: Sample Lean 4 code
- `sample_theorem_text`: Sample natural language theorem
- `event_loop`: Event loop for async tests

## Writing New Tests

### Template for Unit Tests

```python
class TestNewFeature:
    """Test suite for new feature"""

    def test_basic_functionality(self, mock_fixture):
        """Test basic functionality works"""
        result = function_under_test(mock_fixture)
        assert result["success"] is True

    def test_error_handling(self, mock_fixture):
        """Test errors are handled properly"""
        with pytest.raises(ExpectedException):
            function_under_test(invalid_input)
```

### Template for Integration Tests

```python
@pytest.mark.integration
def test_integration_with_external_service(mock_api_client):
    """Test integration with external service"""
    mock_api_client.call = Mock(return_value={"status": "ok"})

    result = function_under_test(mock_api_client)

    assert result["status"] == "ok"
```

### Template for Async Tests

```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality"""
    result = await async_function()
    assert result is not None
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test BubbleLabs Plugin

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio pytest-cov
      - run: pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest test_bubblelabs_comprehensive.py
```

### Async Tests Not Running

Ensure pytest-asyncio is installed:

```bash
pip install pytest-asyncio
```

### Coverage Report Issues

If coverage fails to generate:

```bash
pip install pytest-cov
pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=html --no-cov-on-fail
```

### Tests Hanging

If tests hang, check for:

1. Missing `await` keywords in async tests
2. Infinite loops in test code
3. Blocking operations in async tests

Use timeout to prevent hanging:

```bash
pytest test_bubblelabs_comprehensive.py --timeout=10
```

## Test Maintenance

### When to Update Tests

- When adding new features
- When changing API interfaces
- When fixing bugs
- When refactoring code

### Test Review Checklist

- [ ] Tests are readable and self-documenting
- [ ] Tests have clear assertions
- [ ] Tests are isolated (no dependencies between tests)
- [ ] Tests use fixtures appropriately
- [ ] Tests handle errors correctly
- [ ] Tests are fast (unless marked as slow)
- [ ] Tests are marked appropriately

## Best Practices

1. **Arrange, Act, Assert (AAA) Pattern**
   ```python
   def test_example():
       # Arrange: Set up test data
       input_data = {"key": "value"}

       # Act: Call function under test
       result = function(input_data)

       # Assert: Verify expected outcome
       assert result["success"] is True
   ```

2. **Use Descriptive Test Names**
   ```python
   def test_translation_returns_valid_lean_code():  # Good
   def test_translation():  # Less descriptive
   ```

3. **Test One Thing Per Test**
   ```python
   def test_translation_and_proof_generation():  # Bad
   def test_translation_succeeds():  # Good
   def test_proof_generation_succeeds():  # Good
   ```

4. **Use Fixtures for Common Setup**
   ```python
   @pytest.fixture
   def mock_client():
       return MockClient()

   def test_with_client(mock_client):
       result = mock_client.call()
       assert result is not None
   ```

5. **Mock External Dependencies**
   ```python
   @patch('module.external_api_call')
   def test_with_mock(mock_api):
       mock_api.return_value = {"status": "ok"}
       result = function_under_test()
       assert result["status"] == "ok"
   ```

## Coverage Goals

Current coverage targets:

- **Overall**: 70%+
- **Critical Paths**: 90%+
- **Error Handling**: 80%+
- **Integration Points**: 75%+

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

## Support

For questions or issues with the test suite:

1. Check this guide first
2. Review pytest documentation
3. Check existing tests for examples
4. Contact the BubbleLabs development team

---

**Last Updated**: 2026-01-03
**Test Suite Version**: 1.0.0
**Maintained By**: BubbleLabs Development Team
>>>>>>> 1cb9c5e35 (update)
