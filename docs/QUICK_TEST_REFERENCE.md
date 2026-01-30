# Quick Test Reference Guide

## Common Commands

### Run Tests
```bash
# All tests
pytest test_bubblelabs_comprehensive.py

# With coverage
pytest test_bubblelabs_comprehensive.py --cov=. --cov-report=html

# Only fast tests
pytest test_bubblelabs_comprehensive.py -m "not slow"

# Verbose output
pytest test_bubblelabs_comprehensive.py -v

# Stop on first failure
pytest test_bubblelabs_comprehensive.py -x
```

### Using run_tests.py
```bash
python run_tests.py                    # All tests
python run_tests.py --unit              # Unit tests only
python run_tests.py --coverage          # With coverage
python run_tests.py --parallel          # In parallel
```

## Test Structure

11 Test Classes with 68+ test methods covering:
- Plugin System (12 tests)
- LeanAide Integration (10 tests)
- Evolution Integration (6 tests)
- Knowledge Engine (4 tests)
- Maker/Hephaestus (8 tests)
- UI Components (8 tests)
- Full Integration (5 tests)
- Performance (3 tests)
- Security (5 tests)
- Error Handling (5 tests)
- Thread Safety (2 tests)

## Quick Examples

### Unit Test
```python
def test_basic_functionality(mock_fixture):
    result = function_under_test(mock_fixture)
    assert result["success"] is True
```

### Integration Test
```python
def test_integration_workflow(mock_api_client, mock_workflow):
    mock_api_client.call = Mock(return_value={"status": "ok"})
    result = workflow_step(mock_api_client, mock_workflow)
    assert result["status"] == "ok"
```

### Async Test
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

## Common Fixtures

- `mock_api_key` - Mock API key
- `mock_workflow_state` - Mock workflow state
- `mock_leanaide_client` - Mock LeanAide client
- `mock_hephaestus_client` - Mock Hephaestus client
- `sample_lean_code` - Sample Lean 4 code
- `sample_theorem_text` - Sample theorem

## Running by Marker

```bash
pytest test_bubblelabs_comprehensive.py -m unit
pytest test_bubblelabs_comprehensive.py -m integration
pytest test_bubblelabs_comprehensive.py -m security
pytest test_bubblelabs_comprehensive.py -m performance
pytest test_bubblelabs_comprehensive.py -m "not slow"
```

---

**Last Updated**: 2026-01-03
