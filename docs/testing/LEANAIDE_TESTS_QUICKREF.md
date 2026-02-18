# LeanAide Test Suite - Quick Reference

## Files Created

```
Frontend/
├── test_leanaide_integration.py       # Main test suite (comprehensive)
├── run_leanaide_tests.py              # Test runner with options
├── pytest_leanaide.ini                # Pytest configuration
├── validate_leanaide_tests.py         # Validation script
├── LEANAIDE_TESTS_README.md           # Full documentation
├── LEANAIDE_TESTS_QUICKREF.md         # This file
└── test_leanaide_data/
    ├── sample_theorems.json           # Sample theorem data
    └── sample_lean_code.lean          # Sample Lean code
```

## Quick Commands

```bash
# Validate test setup
python validate_leanaide_tests.py

# Run all tests
python test_leanaide_integration.py
# OR
python run_leanaide_tests.py

# Run specific categories
python run_leanaide_tests.py --unit           # Unit tests only
python run_leanaide_tests.py --integration    # Integration tests only
python run_leanaide_tests.py --mock           # Offline (no server required)
python run_leanaide_tests.py --fast           # Skip slow tests

# With coverage
python run_leanaide_tests.py --coverage

# Verbose output
python run_leanaide_tests.py --verbose
```

## Test Categories

| Category | Marker | Description | Server Required |
|----------|--------|-------------|-----------------|
| Unit Tests | `@mark.unit` | Individual component tests | No |
| Integration | `@mark.integration` | End-to-end workflows | Optional |
| Mock Tests | `@mark.mock` | Offline with mocks | No |
| Server Tests | `@mark.server` | Requires running server | Yes |
| Slow Tests | `@mark.slow` | Long-running tests | Varies |

## MCP Tools Tested (8)

1. `leanaide_translate_theorem` - Natural language → Lean theorem
2. `leanaide_translate_definition` - Natural language → Lean definition
3. `leanaide_generate_proof` - Generate proof for theorem
4. `leanaide_verify_solution` - Verify Lean code correctness
5. `leanaide_math_query` - Answer math questions
6. `leanaide_generate_documentation` - Generate documentation
7. `leanaide_elaborate_code` - Elaborate and check errors
8. `get_leanaide_status` - Check server status

## crewai Bridge Phases (6)

1. **Phase 1: Analysis** - Detect/classify mathematical content
2. **Phase 2: Translate** - NL → Lean 4 translation
3. **Phase 3: Verify** - Solution verification
4. **Phase 4: Proof Check** - Proof validity/completeness
5. **Phase 5: Formal Verification** - Comprehensive verification
6. **Phase 6: Knowledge Extraction** - Extract verified theorems

## Key Fixtures

```python
@pytest.fixture
def sample_theorems():      # Mathematical theorems

@pytest.fixture
def sample_definitions():   # Mathematical definitions

@pytest.fixture
def sample_lean_code():     # Lean code examples

@pytest.fixture
async def mock_client():    # Mocked LeanAide client
```

## Test Structure

```python
@mark.unit
class TestComponentName:
    """Test component X."""

    def test_feature_success(self):
        """Test successful operation."""
        pass

    @patch('module.function')
    def test_feature_with_mock(self, mock_func):
        """Test with mocked dependency."""
        pass
```

## Common Patterns

### Async Tests

```python
@mark.unit
class TestAsyncFeature:
    @pytest.mark.asyncio
    async def test_async_operation(self):
        result = await some_async_function()
        assert result.success is True
```

### Mock Tests

```python
@patch('leanaide_mcp_tools.get_client')
def test_with_mock(self, mock_get_client):
    mock_client = MagicMock()
    mock_client.method.return_value = {...}
    mock_get_client.return_value = mock_client

    result = function_being_tested()
    assert result["success"] is True
```

### Parameterized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("simple", True),
    ("complex", False),
])
def test_parameterized(self, input, expected):
    result = process(input)
    assert result == expected
```

## Validation Output Example

```
LeanAide Test Suite Validator
======================================================================

✓ Python 3.10.0
✓ pytest is installed
✓ pytest_asyncio is installed
⚠ pytest_cov is NOT installed (Optional)
⚠ pytest_xdist is NOT installed (Optional)

Checking LeanAide Modules:
✓ Client module available
✓ MCP Tools module available
⚠ Bridge module NOT available (offline mode)

Checking Test Structure:
✓ Test class TestLeanAideClientInitialization present
✓ Test class TestMCPToolRegistry present
✓ Test class TestMCPTool1_TranslateTheorem present
...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"` |
| Server tests fail | Start LeanAide server: `cd LeanAide && python leanaide_server.py` |
| Async tests fail | Install pytest-asyncio: `pip install pytest-asyncio>=0.21.0` |
| Slow tests | Skip with: `pytest -m "not slow"` |

## Test Coverage Goals

- **Unit Tests**: 80%+ coverage of core functions
- **Integration Tests**: All main workflows covered
- **Error Handling**: All error paths tested
- **Edge Cases**: Empty inputs, timeouts, malformed data

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Run LeanAide Tests
  run: |
    python validate_leanaide_tests.py
    python run_leanaide_tests.py --unit --coverage
    python run_leanaide_tests.py --integration --fast
```

## Resources

- Full documentation: `LEANAIDE_TESTS_README.md`
- Pytest docs: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/

## Support

For issues with LeanAide integration, refer to:
- Main project documentation
- Test suite README for detailed examples
- LeanAide module docstrings
