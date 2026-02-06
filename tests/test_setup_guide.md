# Test Setup Guide for OpenEvolve Frontend

This guide explains how to properly configure and run tests, addressing common configuration and setup issues.

## Table of Contents
1. [Environment Configuration](#environment-configuration)
2. [Test Structure](#test-structure)
3. [Common Issues and Solutions](#common-issues-and-solutions)
4. [Running Tests](#running-tests)
5. [Test Fixtures](#test-fixtures)

---

## Environment Configuration

### Required Environment Variables

All tests require the following environment variables to be set. The root `conftest.py` will set defaults, but you can override them:

```bash
# Database Configuration
export DATABASE_URL="sqlite:///:memory:"
export TEST_DATABASE_URL="sqlite:///:memory:"

# API Configuration
export API_HOST="localhost"
export API_PORT="8000"
export API_TIMEOUT="5"

# Testing Configuration
export TESTING="true"
export TEST_LOG_LEVEL="WARNING"

# Timezone
export TZ="UTC"
```

### Optional Integration-Specific Variables

```bash
# ROMA Integration
export ROMA_ENABLED="false"
export ROMA_DECOMPOSITION_DEPTH="3"

# DSPy Integration
export DSPY_ENABLED="false"

# DeepKE Integration
export DEEPKE_ENABLED="false"

# Knowledge Engine
export KNOWLEDGE_GRAPH_ENABLED="false"
export ENTITY_EXTRACTION_ENABLED="false"
```

---

## Test Structure

### Directory Layout

```
tests/
├── conftest.py                 # Root pytest configuration (REQUIRED)
├── test_helpers.py             # Helper functions for tests
├── test_setup_guide.md         # This file
├── conftest.py                 # Project-specific fixtures
├── unit/                       # Unit tests (isolated, no external deps)
├── integration/                # Integration tests (may require services)
├── e2e/                        # End-to-end tests
├── gauntlet_monitoring/        # Gauntlet monitoring tests
│   └── conftest.py             # Monitoring-specific fixtures
└── gauntlets/                  # Gauntlet tests
    └── conftest.py             # Gauntlet-specific fixtures
```

### Test Categories

1. **Unit Tests** (`@pytest.mark.unit`): Isolated tests, no external dependencies
2. **Integration Tests** (`@pytest.mark.integration`): May require external services
3. **Slow Tests** (`@pytest.mark.slow`): Long-running tests (>10 seconds)
4. **Performance Tests** (`@pytest.mark.performance`): Benchmark tests

---

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:**
```
ImportError: No module named 'knowledge_engine.integrations.roma_integration'
```

**Solution:**
The root `conftest.py` automatically adds the project root to `sys.path`. Ensure you're running tests from the project root:

```bash
cd /path/to/OpenEvolve/Frontend
python -m pytest tests/
```

**Alternative:** Add this to the top of your test file:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Issue 2: Missing Environment Variables

**Problem:**
```
KeyError: 'TESTING'
```

**Solution:** The root `conftest.py` sets default values for all required environment variables. If you see this error, ensure you're using the root conftest:

```bash
# Verify conftest.py exists in tests/ directory
ls tests/conftest.py

# Run with verbose output to see fixture loading
python -m pytest tests/ -v --setup-show
```

### Issue 3: Database Initialization Failures

**Problem:**
```
OperationalError: unable to open database file
```

**Solution:** Use the `test_db_path` or `temp_directory` fixtures for test-specific databases:

```python
def test_with_database(test_db_path):
    # Test-specific database file
    # Automatically cleaned up after test
    pass
```

### Issue 4: Async Test Failures

**Problem:**
```
RuntimeError: This event loop is already closed
```

**Solution:** Use the `event_loop` fixture from root conftest:

```python
@pytest.mark.asyncio
async def test_async_operation(event_loop):
    # Your async test code here
    await some_async_function()
```

### Issue 5: Mock/Stub Initialization Issues

**Problem:**
```
AttributeError: Mock object has no attribute 'some_method'
```

**Solution:** Use the helper fixtures from root conftest:

```python
def test_with_mock_entity_kg(mock_entity_knowledge_graph):
    # Properly mocked knowledge graph
    result = mock_entity_knowledge_graph.add_entity(...)
    assert result == "entity-001"
```

### Issue 6: Integration Import Failures

**Problem:**
```
ImportError: cannot import name 'ROMAIntegration'
```

**Solution:** Check if integration is available before importing:

```python
# Use the integration_availability fixture
def test_roma_integration(integration_availability):
    if not integration_availability.get("roma"):
        pytest.skip("ROMA integration not available")

    # Your test code here
```

Or use the helper function:

```python
from test_helpers import skip_if_integration_missing

def test_roma_integration():
    if skip_reason := skip_if_integration_missing('roma'):
        pytest.skip(skip_reason)

    # Your test code here
```

---

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_security.py -v

# Run specific test function
python -m pytest tests/test_security.py::TestInputValidation::test_sql_injection_in_entity_name -v
```

### Test Selection by Marker

```bash
# Run only unit tests
python -m pytest tests/ -m unit

# Run only integration tests
python -m pytest tests/ -m integration

# Skip slow tests
python -m pytest tests/ -m "not slow"

# Skip performance tests
python -m pytest tests/ -m "not performance"
```

### Test Discovery and Collection

```bash
# List all tests without running
python -m pytest tests/ --collect-only

# Show which tests would be selected by markers
python -m pytest tests/ --collect-only -m unit
```

### Debugging Tests

```bash
# Show print output
python -m pytest tests/ -s

# Stop on first failure
python -m pytest tests/ -x

# Enter debugger on failure
python -m pytest tests/ --pdb

# Show local variables on failure
python -m pytest tests/ -l
```

### Coverage Reporting

```bash
# Run tests with coverage
python -m pytest tests/ --cov=knowledge_engine --cov-report=html

# Generate coverage report
python -m pytest tests/ --cov=knowledge_engine --cov-report=term-missing
```

---

## Test Fixtures

### Available Fixtures (from root conftest.py)

#### Database Fixtures
- `test_db_path`: Temporary database file path
- `temp_directory`: Temporary directory for test files

#### Mock Fixtures
- `mock_logger`: Mock logger object
- `mock_config`: Default test configuration dictionary
- `mock_async_response`: Factory for creating async HTTP responses
- `mock_entity_knowledge_graph`: Mock knowledge graph
- `mock_knowledge_artifact`: Mock knowledge artifact

#### ROMA Fixtures
- `mock_roma_integration`: Mock ROMA integration

#### Data Fixtures
- `sample_entity_data`: Sample entity data
- `sample_relationship_data`: Sample relationship data
- `sample_problem_data`: Sample problem data

#### Async Fixtures
- `event_loop`: Event loop for async tests
- `async_setup`: Generic async setup/teardown

#### Validation Fixtures
- `validation_success`: Successful validation result
- `validation_failure`: Failed validation result

#### Utility Fixtures
- `integration_availability`: Dictionary of available integrations
- `fixed_timestamp`: Fixed timestamp for testing
- `mock_time`: Mocked time functions

### Using Fixtures in Tests

```python
import pytest
from test_helpers import (
    create_mock_entity,
    create_mock_relationship,
    assert_valid_entity
)

class TestMyFeature:
    @pytest.fixture
    def custom_setup(self, mock_config):
        # Setup code using fixtures
        config = mock_config.copy()
        config["my_key"] = "my_value"
        yield config
        # Teardown happens automatically

    def test_with_fixtures(self, custom_setup, sample_entity_data):
        # Use the fixtures
        entity = create_mock_entity(**sample_entity_data)
        assert_valid_entity(entity)
```

---

## Best Practices

### 1. Use Fixtures for Setup/Teardown

```python
# GOOD: Use fixtures
@pytest.fixture
def my_entity():
    entity = create_entity()
    yield entity
    cleanup_entity(entity)

def test_something(my_entity):
    assert my_entity.is_valid()

# AVOID: Manual setup
def test_something():
    entity = create_entity()
    try:
        assert entity.is_valid()
    finally:
        cleanup_entity(entity)
```

### 2. Check Integration Availability

```python
# GOOD: Check availability
def test_roma_feature(integration_availability):
    if not integration_availability.get("roma"):
        pytest.skip("ROMA not available")
    # Test code here

# AVOID: Assume integration exists
def test_roma_feature():
    roma = ROMAIntegration()  # May fail if not installed
```

### 3. Use Helper Functions

```python
# GOOD: Use helpers from test_helpers.py
from test_helpers import assert_valid_entity, create_mock_entity

def test_entity():
    entity = create_mock_entity()
    assert_valid_entity(entity)

# AVOID: Duplicate validation logic
def test_entity():
    entity = Mock()
    assert hasattr(entity, 'entity_id')
    assert hasattr(entity, 'entity_type')
    # ... lots of validation code
```

### 4. Use Markers Appropriately

```python
# GOOD: Mark tests appropriately
@pytest.mark.slow
@pytest.mark.integration
def test_expensive_operation():
    time.sleep(60)

# AVOID: No markers
def test_expensive_operation():
    time.sleep(60)  # Will run even when running quick tests
```

### 5. Handle Async Correctly

```python
# GOOD: Use async fixtures and event_loop
@pytest.mark.asyncio
async def test_async_feature(event_loop):
    result = await async_function()
    assert result is not None

# AVOID: Run async code in sync tests
def test_async_feature():
    result = asyncio.run(async_function())  # May conflict with event loop
```

---

## Troubleshooting Commands

### Check if Tests are Configured Correctly

```bash
# Verify conftest.py is loaded
python -m pytest tests/ --collect-only | head -20

# Check fixtures are available
python -m pytest tests/ --fixtures

# Test specific fixture
python -m pytest tests/ -k "test_with_fixture_name" -v
```

### Debug Import Issues

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Test specific import
python -c "from knowledge_engine.integrations.roma_integration import ROMAIntegration; print('OK')"

# Run with trace
python -m pytest tests/ --tb=long
```

### Check Environment Variables

```bash
# List all test environment variables
python -c "import os; import sys; sys.path.insert(0, 'tests'); from conftest import set_test_environment_defaults; set_test_environment_defaults(); import pprint; pprint.pprint(dict(os.environ))"
```

---

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- Project CLAUDE.md: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\CLAUDE.md`
