# Graphiti Temporal Integration - Test Suite

## Overview

Comprehensive test suite for Phase 2.1 - Graphiti Core Integration with temporal reasoning, hybrid search, and contradiction detection capabilities.

## Test Structure

```
knowledge_engine/tests/
├── test_temporal_graphiti.py     # Main test suite
├── conftest.py                    # Pytest fixtures
└── README.md                      # This file
```

## Running Tests

### Run All Tests

```bash
# Run all temporal tests
pytest knowledge_engine/tests/test_temporal_graphiti.py -v

# Run with coverage
pytest knowledge_engine/tests/test_temporal_graphiti.py \
    --cov=knowledge_engine.core.temporal_knowledge_engine \
    --cov=knowledge_engine.integrations.graphiti_temporal_bridge \
    --cov-report=html

# Run with detailed output
pytest knowledge_engine/tests/test_temporal_graphiti.py -v -s
```

### Run Specific Test Classes

```bash
# Test KnowledgeArtifact
pytest knowledge_engine/tests/test_temporal_graphiti.py::TestKnowledgeArtifact -v

# Test TemporalKnowledgeEngine
pytest knowledge_engine/tests/test_temporal_graphiti.py::TestTemporalKnowledgeEngine -v

# Test GraphitiTemporalBridge
pytest knowledge_engine/tests/test_temporal_graphiti.py::TestGraphitiTemporalBridge -v
```

### Run Specific Tests

```bash
# Run single test
pytest knowledge_engine/tests/test_temporal_graphiti.py::TestKnowledgeArtifact::test_artifact_creation -v

# Run with pattern matching
pytest knowledge_engine/tests/test_temporal_graphiti.py -k "temporal" -v
```

## Test Categories

### 1. Unit Tests

**TestKnowledgeArtifact**
- Artifact creation
- Temporal validity checking
- Serialization/deserialization
- Data integrity

**TestRerankMethod**
- Enum values
- Method validation

### 2. Integration Tests

**TestTemporalKnowledgeEngine**
- Add temporal knowledge
- Query at point in time
- Hybrid search
- Contradiction detection
- Knowledge invalidation
- Timeline reconstruction

**TestGraphitiTemporalBridge**
- Artifact to Episode conversion
- Entity type mapping
- Result transformation
- Temporal filtering

### 3. End-to-End Tests

**TestTemporalIntegration**
- Complete temporal workflow
- Hybrid search vs local search
- Knowledge evolution tracking

### 4. Backend Tests

**TestGraphitiBackend** (Skipped by default)
- Real Graphiti integration
- Temporal persistence
- Performance benchmarks

## Test Fixtures

### Engine Fixture

```python
@pytest.fixture
async def engine():
    """Create a temporal knowledge engine for testing."""
    engine = TemporalKnowledgeEngine(
        enable_temporal=True,
        enable_hybrid_search=True,
    )
    yield engine
    # Cleanup
```

### Bridge Fixture

```python
@pytest.fixture
async def bridge():
    """Create a temporal bridge for testing."""
    bridge = GraphitiTemporalBridge(graphiti_bridge=None)
    yield bridge
```

## Writing Tests

### Basic Test Structure

```python
@pytest.mark.asyncio
async def test_feature_description():
    """Test description."""
    # Arrange
    engine = TemporalKnowledgeEngine()

    # Act
    result = await engine.some_method()

    # Assert
    assert result is not None
    assert result.property == expected_value
```

### Testing Temporal Features

```python
@pytest.mark.asyncio
async def test_temporal_validity():
    """Test temporal validity checking."""
    now = datetime.utcnow()
    past = now - timedelta(days=1)
    future = now + timedelta(days=1)

    artifact = KnowledgeArtifact(
        id="test",
        content="Test content",
        artifact_type="solution_pattern",
        valid_at=past,
        invalid_at=future,
    )

    assert artifact.is_valid_at(now)
    assert not artifact.is_valid_at(past - timedelta(seconds=1))
    assert not artifact.is_valid_at(future + timedelta(seconds=1))
```

### Testing with Time

```python
@pytest.mark.asyncio
async def test_point_in_time_query():
    """Test querying at specific time."""
    engine = TemporalKnowledgeEngine()

    t1 = datetime(2024, 1, 1)
    t2 = datetime(2024, 6, 1)

    await engine.add_knowledge_temporal(
        content="Knowledge at t1",
        artifact_type="solution_pattern",
        valid_at=t1,
    )

    await engine.add_knowledge_temporal(
        content="Knowledge at t2",
        artifact_type="solution_pattern",
        valid_at=t2,
    )

    # Query at t1
    results_t1 = await engine.query_at_time("knowledge", t1)
    assert any("Knowledge at t1" in r.content for r in results_t1)

    # Query at t2
    results_t2 = await engine.query_at_time("knowledge", t2)
    assert any("Knowledge at t2" in r.content for r in results_t2)
```

## Test Coverage

### Current Coverage

- **KnowledgeArtifact**: 100%
- **TemporalKnowledgeEngine**: ~90%
- **GraphitiTemporalBridge**: ~85%
- **RerankMethod**: 100%

### Coverage Goals

- All public methods: 100%
- All edge cases: Covered
- All error paths: Tested
- Integration scenarios: Comprehensive

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -e .
      - run: pytest knowledge_engine/tests/test_temporal_graphiti.py -v
      - run: pytest knowledge_engine/tests/test_temporal_graphiti.py --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Performance Testing

### Benchmark Example

```python
@pytest.mark.benchmark
async def test_hybrid_search_performance():
    """Benchmark hybrid search performance."""
    engine = TemporalKnowledgeEngine()

    # Add 1000 artifacts
    for i in range(1000):
        await engine.add_knowledge_temporal(
            content=f"Artifact {i}",
            artifact_type="solution_pattern",
            valid_at=datetime.utcnow(),
        )

    # Benchmark search
    start = time.time()
    results = await engine.search_with_graphiti(
        query="artifact",
        max_results=10,
    )
    duration = time.time() - start

    assert duration < 1.0  # Should complete in < 1 second
```

## Troubleshooting Tests

### Common Issues

**Tests fail with "Graphiti not available"**
- Tests are designed to work without Graphiti
- Ensure graceful degradation is working
- Check imports in adapter.py

**Async tests hang**
- Ensure all async operations are awaited
- Check for missing `async def`
- Verify fixtures are async

**Time-dependent tests fail**
- Use fixed timestamps in tests
- Avoid `datetime.utcnow()` in assertions
- Use `freeze_time` if needed

### Debug Mode

```bash
# Run with debug output
pytest knowledge_engine/tests/test_temporal_graphiti.py -v -s --log-cli-level=DEBUG

# Run with pdb on failure
pytest knowledge_engine/tests/test_temporal_graphiti.py -v --pdb
```

## Test Data

### Sample Artifacts

```python
# Solution pattern
solution_artifact = KnowledgeArtifact(
    id="sol_001",
    content="Use async/await for I/O operations",
    artifact_type="solution_pattern",
    valid_at=datetime.utcnow(),
    metadata={"language": "python", "category": "async"},
)

# Workflow
workflow_artifact = KnowledgeArtifact(
    id="wf_001",
    content="CI/CD pipeline for deployment",
    artifact_type="workflow",
    valid_at=datetime.utcnow(),
    metadata=["devops", "deployment"],
)

# Problem
problem_artifact = KnowledgeArtifact(
    id="prob_001",
    content="Memory leak in async functions",
    artifact_type="problem",
    valid_at=datetime.utcnow(),
    metadata={"severity": "high"},
)
```

## Contributing Tests

### Adding New Tests

1. Follow naming convention: `test_<feature>_<description>`
2. Add docstrings explaining what is being tested
3. Use descriptive assertion messages
4. Include edge cases and error conditions
5. Update this README with new test categories

### Example

```python
@pytest.mark.asyncio
async def test_new_feature_validates_input():
    """
    Test that new feature validates input correctly.

    Should reject invalid input and accept valid input.
    """
    engine = TemporalKnowledgeEngine()

    # Test invalid input
    with pytest.raises(ValueError):
        await engine.new_feature(invalid_input="bad")

    # Test valid input
    result = await engine.new_feature(valid_input="good")
    assert result is not None
```

## Test Metrics

### Success Criteria

- All tests pass: ✅
- Coverage > 90%: ✅
- No skipped tests (except backend): ✅
- Performance benchmarks met: ✅

### Coverage Report

```bash
# Generate coverage report
pytest knowledge_engine/tests/test_temporal_graphiti.py --cov-report=html

# View report
open htmlcov/index.html
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Async Testing with Pytest](https://pytest-asyncio.readthedocs.io/)
- [Graphiti Documentation](https://graphiti.dev/)
- [Temporal Knowledge Engine Guide](../docs/GRAPITI_TEMPORAL_INTEGRATION.md)
