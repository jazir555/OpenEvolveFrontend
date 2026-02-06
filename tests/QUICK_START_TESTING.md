# Quick Start Guide - Writing Tests

A quick reference for writing tests in the OpenEvolve Frontend project.

## Table of Contents
1. [Basic Test Template](#basic-test-template)
2. [Common Patterns](#common-patterns)
3. [Available Fixtures](#available-fixtures)
4. [Helper Functions](#helper-functions)
5. [Troubleshooting](#troubleshooting)

---

## Basic Test Template

```python
"""
Test module for MyFeature
"""
import pytest
from test_helpers import (
    create_mock_entity,
    assert_valid_entity,
    skip_if_integration_missing
)

class TestMyFeature:
    """Test suite for MyFeature"""

    def test_basic_functionality(self, mock_config):
        """Test basic functionality"""
        # Arrange
        config = mock_config

        # Act
        result = my_function(config)

        # Assert
        assert result is not None
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_async_functionality(self, event_loop):
        """Test async functionality"""
        # Act
        result = await my_async_function()

        # Assert
        assert result is not None
```

---

## Common Patterns

### Pattern 1: Testing with Mocked Knowledge Graph

```python
def test_entity_operations(mock_entity_knowledge_graph):
    """Test entity operations with mocked KG"""
    # Setup mock
    mock_entity_knowledge_graph.add_entity.return_value = "entity-001"

    # Test
    entity_id = mock_entity_knowledge_graph.add_entity(entity_data)

    # Assert
    assert entity_id == "entity-001"
    mock_entity_knowledge_graph.add_entity.assert_called_once()
```

### Pattern 2: Testing with Temp Database

```python
def test_with_database(test_db_path):
    """Test with temporary database"""
    # Use test_db_path for database file
    db = Database(f"sqlite:///{test_db_path}")

    # Test code here
    db.add_entity(...)

    # Cleanup happens automatically
```

### Pattern 3: Testing Optional Integration

```python
def test_roma_integration(integration_availability):
    """Test ROMA integration if available"""
    # Skip if not available
    if not integration_availability.get("roma"):
        pytest.skip("ROMA integration not available")

    # Test code here
    # Or use helper:
    # if skip_reason := skip_if_integration_missing('roma'):
    #     pytest.skip(skip_reason)
```

### Pattern 4: Testing with Custom Environment

```python
def test_with_custom_env():
    """Test with custom environment variables"""
    from test_helpers import with_env_vars

    with with_env_vars({"MY_VAR": "test_value"}):
        # Test code that uses MY_VAR
        assert os.environ["MY_VAR"] == "test_value"

    # Variable is automatically cleaned up
```

### Pattern 5: Async Test with Proper Setup

```python
@pytest.mark.asyncio
async def test_async_feature(event_loop):
    """Test async feature with proper event loop"""
    # Setup
    resource = await create_async_resource()

    # Test
    result = await resource.do_something()

    # Assert
    assert result is not None

    # Cleanup happens automatically
```

---

## Available Fixtures

### Database & Filesystem
```python
def test_example(test_db_path, temp_directory):
    # test_db_path: Temporary database file (auto-cleanup)
    # temp_directory: Temporary directory (auto-cleanup)
    pass
```

### Mocks
```python
def test_example(mock_config, mock_logger, mock_entity_knowledge_graph):
    # mock_config: Test configuration dictionary
    # mock_logger: Mock logger object
    # mock_entity_knowledge_graph: Mock knowledge graph
    pass
```

### Data
```python
def test_example(sample_entity_data, sample_relationship_data, sample_problem_data):
    # sample_entity_data: Sample entity data
    # sample_relationship_data: Sample relationship data
    # sample_problem_data: Sample problem data
    pass
```

### Async
```python
@pytest.mark.asyncio
async def test_example(event_loop, async_setup):
    # event_loop: Event loop for async tests
    # async_setup: Generic async setup/teardown
    pass
```

### ROMA
```python
def test_example(mock_roma_integration):
    # mock_roma_integration: Mock ROMA integration
    pass
```

### Validation
```python
def test_example(validation_success, validation_failure):
    # validation_success: Successful validation result
    # validation_failure: Failed validation result
    pass
```

---

## Helper Functions

### Import Testing
```python
from test_helpers import safe_import, check_integration_available

# Safe import
success, module, error = safe_import("knowledge_engine.integrations.roma_integration")

# Check integration
if check_integration_available('roma'):
    # Integration is available
    pass
```

### Mock Creation
```python
from test_helpers import (
    create_mock_entity,
    create_mock_relationship,
    create_mock_knowledge_artifact,
    create_mock_roma_decomposition,
)

# Create mock entity
entity = create_mock_entity(entity_id="test-001")

# Create mock relationship
rel = create_mock_relationship(source_id="e1", target_id="e2")

# Create mock artifact
artifact = create_mock_knowledge_artifact()
```

### Assertions
```python
from test_helpers import (
    assert_valid_entity,
    assert_valid_relationship,
    assert_valid_artifact,
    assert_valid_timestamp,
)

# Validate entity
assert_valid_entity(entity, entity_type="Person")

# Validate relationship
assert_valid_relationship(rel, rel_type="RELATED_TO")

# Validate artifact
assert_valid_artifact(artifact, artifact_type="document")

# Validate timestamp
assert_valid_timestamp("2026-02-06T12:00:00Z")
```

### Test Data Generation
```python
from test_helpers import generate_test_entities, generate_test_relationships

# Generate entities
entities = generate_test_entities(count=10)

# Generate relationships
relationships = generate_test_relationships(count=10)
```

### Configuration
```python
from test_helpers import build_test_config

# Build test config
config = build_test_config({
    "api": {"port": 9000}
})
```

### Environment
```python
from test_helpers import set_test_env_var, with_env_vars

# Set env var
set_test_env_var("MY_VAR", "value")

# Use context manager
with with_env_vars({"VAR1": "val1", "VAR2": "val2"}):
    # Test code here
    pass
```

---

## Troubleshooting

### Problem: Import Error
```bash
ImportError: No module named 'knowledge_engine'
```

**Solution:** Run tests from project root
```bash
cd /path/to/OpenEvolve/Frontend
python -m pytest tests/
```

### Problem: Missing Environment Variable
```bash
KeyError: 'TESTING'
```

**Solution:** Root conftest.py sets this automatically. Verify it's loaded:
```bash
python -m pytest tests/ --collect-only
```

### Problem: Async Test Fails
```bash
RuntimeError: This event loop is already closed
```

**Solution:** Use event_loop fixture:
```python
@pytest.mark.asyncio
async def test_my_feature(event_loop):
    # Your test here
    pass
```

### Problem: Integration Not Available
```bash
ImportError: cannot import name 'ROMAIntegration'
```

**Solution:** Check availability before using:
```python
def test_roma(integration_availability):
    if not integration_availability.get("roma"):
        pytest.skip("ROMA not available")
    # Test code here
```

### Problem: Mock Missing Attribute
```bash
AttributeError: Mock object has no attribute 'some_method'
```

**Solution:** Use helper functions instead of manual mocks:
```python
from test_helpers import create_mock_entity
entity = create_mock_entity()  # Has all required attributes
```

---

## Quick Commands

```bash
# Check test environment
python tests/init_test_env.py

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_security.py -v

# Run specific test
python -m pytest tests/test_security.py::TestInputValidation::test_sql_injection_in_entity_name -v

# Run only unit tests
python -m pytest tests/ -m unit

# Skip slow tests
python -m pytest tests/ -m "not slow"

# Show fixtures
python -m pytest tests/ --fixtures

# Collect tests without running
python -m pytest tests/ --collect-only

# Run with coverage
python -m pytest tests/ --cov=knowledge_engine
```

---

## Test Markers

```python
# Mark test as unit test
@pytest.mark.unit
def test_something():
    pass

# Mark test as integration test
@pytest.mark.integration
def test_something():
    pass

# Mark test as slow
@pytest.mark.slow
def test_something():
    pass

# Mark test as performance test
@pytest.mark.performance
def test_something():
    pass

# Mark test as async
@pytest.mark.asyncio
async def test_something():
    pass
```

---

## Best Practices

### ✅ DO
- Use fixtures from conftest.py
- Use helper functions from test_helpers.py
- Check integration availability
- Use proper markers
- Follow Arrange-Act-Assert pattern

### ❌ DON'T
- Create manual mocks (use helpers)
- Assume integrations are available
- Skip markers on slow tests
- Use hardcoded paths
- Ignore async test patterns

---

## Getting Help

1. **Read the full guide:** `tests/test_setup_guide.md`
2. **Check fixtures:** `python -m pytest tests/ --fixtures`
3. **Initialize environment:** `python tests/init_test_env.py`
4. **View summary:** `TEST_CONFIGURATION_FIXES_SUMMARY.md`

---

**Happy Testing!** 🚀
