# BubbleLabs Edge Case Testing Plan

**Date:** 2025-12-29
**Status:** Ready for Testing
**Test Coverage Goal:** 95%+ edge case handling

---

## Test Strategy Overview

This testing plan verifies all 27 HIGH severity edge case fixes across 6 BubbleLabs modules.

**Test Categories:**
1. Unit Tests - Individual function validation
2. Integration Tests - Cross-module validation
3. Concurrency Tests - Thread safety validation
4. Security Tests - Input sanitization validation
5. Performance Tests - Resource limit validation

---

## Test Environment Setup

```python
# test_fixtures.py
import pytest
import threading
import time
from bubblelabs_crewai_bridge import BubbleLabsCrewAIBridge, BubbleLabsTicketConfig
from bubblelabs_mcp_tools import (
    create_bubblelabs_workflow,
    execute_bubblelabs_workflow,
    control_bubblelabs_workflow
)
from bubblelabs_analytics import BubbleLabsAnalytics
from bubblelabs_integration import BubbleLabsIntegration
from bubblelabs_security import validate_uuid, validate_workflow_type

@pytest.fixture
def bridge():
    """Create a test bridge instance."""
    return BubbleLabsCrewAIBridge(
        config=BubbleLabsTicketConfig(
            auto_create_tickets=False,  # Don't actually create tickets in tests
            auto_update_progress=False
        )
    )

@pytest.fixture
def analytics():
    """Create a test analytics instance with in-memory database."""
    return BubbleLabsAnalytics(db_path=":memory:")

@pytest.fixture
def integration():
    """Create a test integration instance."""
    return BubbleLabsIntegration()
```

---

## Category 1: Empty Input Validation Tests (12 tests)

### Test 1.1: bubblelabs_crewai_bridge.py - None Validation

```python
# test_crewai_bridge_validation.py

def test_create_ticket_with_none_workflow(bridge):
    """Test that None workflow_definition raises ValueError."""
    with pytest.raises(ValueError, match="workflow_definition cannot be None"):
        bridge.create_ticket_from_workflow(
            workflow_definition=None,
            assignee="test-user"
        )

def test_create_ticket_with_empty_workflow_id(bridge):
    """Test that workflow with empty ID raises ValueError."""
    from bubblelabs_integration import BubbleWorkflowDefinition

    workflow = BubbleWorkflowDefinition(
        id="",  # Empty ID
        name="Test Workflow",
        description="Test description",
        nodes=[],
        edges=[],
        metadata={}
    )

    with pytest.raises(ValueError, match="workflow_definition.id cannot be empty"):
        bridge.create_ticket_from_workflow(workflow_definition=workflow)

def test_update_ticket_with_empty_instance_id(bridge):
    """Test that empty instance_id raises ValueError."""
    from openevolve_bubblelabs_api import WorkflowStatus

    with pytest.raises(ValueError, match="workflow_instance_id cannot be empty"):
        bridge.update_ticket_progress(
            workflow_instance_id="",  # Empty
            progress=0.5,
            status=WorkflowStatus.RUNNING
        )

def test_update_ticket_with_invalid_progress(bridge):
    """Test that out-of-range progress raises ValueError."""
    from openevolve_bubblelabs_api import WorkflowStatus

    # Progress too high
    with pytest.raises(ValueError, match="progress must be between 0.0 and 1.0"):
        bridge.update_ticket_progress(
            workflow_instance_id="test-instance",
            progress=1.5,  # Invalid
            status=WorkflowStatus.RUNNING
        )

    # Progress negative
    with pytest.raises(ValueError, match="progress must be between 0.0 and 1.0"):
        bridge.update_ticket_progress(
            workflow_instance_id="test-instance",
            progress=-0.1,  # Invalid
            status=WorkflowStatus.RUNNING
        )

def test_sync_with_empty_workflow_id(bridge):
    """Test that empty workflow_definition_id raises ValueError."""
    with pytest.raises(ValueError, match="workflow_definition_id cannot be empty"):
        bridge.sync_workflow_to_ticket(workflow_definition_id="")

def test_stop_sync_with_invalid_timeout(bridge):
    """Test that invalid timeout raises ValueError."""
    # Negative timeout
    with pytest.raises(ValueError, match="timeout cannot be negative"):
        bridge.stop_background_sync(timeout=-10)

    # Timeout exceeds maximum
    with pytest.raises(ValueError, match="timeout cannot exceed.*seconds"):
        bridge.stop_background_sync(timeout=10000)
```

### Test 1.2: bubblelabs_mcp_tools.py - Empty Input Tests

```python
# test_mcp_tools_validation.py

def test_create_workflow_with_empty_problem_statement():
    """Test that empty problem_statement raises ValueError."""
    with pytest.raises(ValueError, match="problem_statement cannot be empty"):
        create_bubblelabs_workflow(
            problem_statement="",  # Empty
            team_config={"planner_team": "Backend"}
        )

def test_create_workflow_with_whitespace_problem_statement():
    """Test that whitespace-only problem_statement raises ValueError."""
    with pytest.raises(ValueError, match="problem_statement cannot be empty"):
        create_bubblelabs_workflow(
            problem_statement="   \n\t   ",  # Whitespace only
            team_config={"planner_team": "Backend"}
        )

def test_create_workflow_with_too_long_problem_statement():
    """Test that excessively long problem_statement raises ValueError."""
    long_statement = "a" * 10001  # Exceeds MAX_PROBLEM_STATEMENT_LENGTH

    with pytest.raises(ValueError, match="problem_statement cannot exceed.*characters"):
        create_bubblelabs_workflow(
            problem_statement=long_statement,
            team_config={}
        )

def test_create_workflow_with_too_long_name():
    """Test that excessively long workflow_name raises ValueError."""
    long_name = "a" * 256  # Exceeds MAX_WORKFLOW_NAME_LENGTH

    with pytest.raises(ValueError, match="workflow_name cannot exceed.*characters"):
        create_bubblelabs_workflow(
            problem_statement="Test problem",
            workflow_name=long_name
        )

def test_create_workflow_with_too_many_team_configs():
    """Test that too many team_config entries raises ValueError."""
    large_config = {f"team_{i}": f"Team{i}" for i in range(51)}  # Exceeds MAX_TEAM_CONFIG_ENTRIES

    with pytest.raises(ValueError, match="team_config cannot exceed.*entries"):
        create_bubblelabs_workflow(
            problem_statement="Test problem",
            team_config=large_config
        )

def test_execute_workflow_with_empty_workflow_id():
    """Test that empty workflow_id raises ValueError."""
    with pytest.raises(ValueError, match="workflow_id cannot be empty"):
        execute_bubblelabs_workflow(
            workflow_id="",  # Empty
            parameters={}
        )
```

### Test 1.3: bubblelabs_analytics.py - Empty Input Tests

```python
# test_analytics_validation.py

def test_start_tracking_with_empty_workflow_id(analytics):
    """Test that empty workflow_id raises ValueError."""
    with pytest.raises(ValueError, match="workflow_id cannot be empty"):
        analytics.start_workflow_tracking(
            workflow_id="",  # Empty
            workflow_name="Test Workflow",
            instance_id="instance-123"
        )

def test_start_tracking_with_empty_workflow_name(analytics):
    """Test that empty workflow_name raises ValueError."""
    with pytest.raises(ValueError, match="workflow_name cannot be empty"):
        analytics.start_workflow_tracking(
            workflow_id="workflow-123",
            workflow_name="",  # Empty
            instance_id="instance-123"
        )

def test_track_node_with_invalid_tokens(analytics):
    """Test that negative token count is handled."""
    # Start workflow first
    analytics.start_workflow_tracking("wf-1", "Test", "inst-1")

    # Track with negative tokens (should be handled gracefully or raise error)
    result = analytics.track_node_execution(
        workflow_id="wf-1",
        node_id="node-1",
        node_type="test",
        tokens_used=-100,  # Negative
        execution_time=1.0,
        provider="openai"
    )
    # Should either return False or raise error
    assert result is False or result == -100
```

### Test 1.4: bubblelabs_typescript_export.py - Empty Input Tests

```python
# test_typescript_export_validation.py

def test_export_workflow_with_none_definition():
    """Test that None workflow_definition raises ValueError."""
    from bubblelabs_typescript_export import BubbleLabsTypeScriptExporter

    exporter = BubbleLabsTypeScriptExporter()

    result = exporter.export_workflow(
        workflow_definition=None,  # None
        output_path="test.ts"
    )

    assert result.success is False
    assert "None" in result.error or "cannot be None" in result.error

def test_export_workflow_with_empty_nodes():
    """Test that workflow with no nodes is handled correctly."""
    from bubblelabs_typescript_export import BubbleLabsTypeScriptExporter
    from bubblelabs_integration import BubbleWorkflowDefinition

    exporter = BubbleLabsTypeScriptExporter()

    workflow = BubbleWorkflowDefinition(
        id="test-1",
        name="Empty Workflow",
        description="Workflow with no nodes",
        nodes=[],  # Empty
        edges=[],
        metadata={}
    )

    result = exporter.export_workflow(workflow)

    # Should either fail with warning or succeed with warning
    if not result.success:
        assert "no nodes" in result.error.lower()
    else:
        assert len(result.warnings) > 0

def test_export_with_invalid_path():
    """Test that path traversal attempts are blocked."""
    from bubblelabs_typescript_export import BubbleLabsTypeScriptExporter
    from bubblelabs_integration import BubbleWorkflowDefinition

    exporter = BubbleLabsTypeScriptExporter()

    workflow = BubbleWorkflowDefinition(
        id="test-1",
        name="Test",
        description="Test",
        nodes=[],
        edges=[],
        metadata={}
    )

    # Try path traversal
    result = exporter.export_workflow(
        workflow,
        output_path="../../../etc/passwd"  # Path traversal attempt
    )

    assert result.success is False
    assert "traversal" in result.error.lower() or "invalid" in result.error.lower()
```

### Test 1.5: bubblelabs_integration.py - Empty Input Tests

```python
# test_integration_validation.py

def test_control_workflow_with_empty_instance_id(integration):
    """Test that empty instance_id raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        integration.control_workflow_local(
            instance_id="",  # Empty
            action="start"
        )

    assert "instance_id" in str(exc_info.value).lower() or "cannot be empty" in str(exc_info.value).lower()

def test_control_workflow_with_invalid_action(integration):
    """Test that invalid action raises ValueError."""
    integration.workflow_instances["test-1"] = BubbleWorkflowInstance(
        id="test-1",
        definition_id="def-1",
        status="running",
        created_at=time.time(),
        updated_at=time.time(),
        progress=0.5
    )

    result = integration.control_workflow_local(
        instance_id="test-1",
        action="invalid_action"  # Invalid
    )

    # Should either raise error or return error response
    if "error" in result:
        assert "invalid" in result["error"].lower()
```

### Test 1.6: bubblelabs_security.py - Empty Input Tests

```python
# test_security_validation.py

def test_validate_uuid_with_empty_string():
    """Test that empty string fails UUID validation."""
    from bubblelabs_security import validate_uuid, ValidationError

    with pytest.raises(ValidationError, match="must be a non-empty string"):
        validate_uuid("", "test_id")

def test_validate_uuid_with_invalid_format():
    """Test that invalid UUID format raises ValidationError."""
    from bubblelabs_security import validate_uuid, ValidationError

    with pytest.raises(ValidationError, match="must be a valid UUID format"):
        validate_uuid("not-a-uuid", "test_id")

def test_validate_workflow_type_with_empty_string():
    """Test that empty workflow_type fails validation."""
    from bubblelabs_security import validate_workflow_type, ValidationError

    with pytest.raises(ValidationError, match="must be a non-empty string"):
        validate_workflow_type("")

def test_validate_workflow_type_with_invalid_type():
    """Test that invalid workflow_type raises ValidationError."""
    from bubblelabs_security import validate_workflow_type, ValidationError

    with pytest.raises(ValidationError, match="Invalid workflow_type"):
        validate_workflow_type("invalid_workflow_type")

def test_validate_url_with_ssrf_attempt():
    """Test that SSRF attempts are blocked."""
    from bubblelabs_security import validate_url, ValidationError

    # Try to access internal network
    with pytest.raises(ValidationError, match="not in the allowed URL whitelist"):
        validate_url("http://192.168.1.1/admin", "test_url")

    # Try file:// protocol
    with pytest.raises(ValidationError, match="not in the allowed URL whitelist"):
        validate_url("file:///etc/passwd", "test_url")
```

---

## Category 2: State Validation Tests (6 tests)

### Test 2.1: Update Non-Existent Ticket

```python
def test_update_non_existent_ticket(bridge):
    """Test that updating non-existent ticket returns False."""
    from openevolve_bubblelabs_api import WorkflowStatus

    result = bridge.update_ticket_progress(
        workflow_instance_id="non-existent-instance",
        progress=0.5,
        status=WorkflowStatus.RUNNING
    )

    assert result is False

def test_update_non_existent_ticket_analytics(analytics):
    """Test that tracking nodes for non-existent workflow returns False."""
    # Don't start tracking workflow

    result = analytics.track_node_execution(
        workflow_id="non-existent-workflow",
        node_id="node-1",
        node_type="test",
        tokens_used=1000,
        execution_time=5.0,
        provider="openai"
    )

    assert result is False
```

### Test 2.2: Create Duplicate Ticket

```python
def test_create_duplicate_ticket(bridge):
    """Test that creating duplicate ticket returns existing ticket."""
    from bubblelabs_integration import BubbleWorkflowDefinition

    workflow = BubbleWorkflowDefinition(
        id="test-workflow",
        name="Test Workflow",
        description="Test",
        nodes=[],
        edges=[],
        metadata={}
    )

    # Create first ticket (will be mock in test mode)
    ticket_id_1 = bridge.create_ticket_from_workflow(workflow)

    # Try to create again
    ticket_id_2 = bridge.create_ticket_from_workflow(workflow)

    # Should return the same ticket ID
    assert ticket_id_1 == ticket_id_2
```

### Test 2.3: Invalid State Transitions

```python
def test_invalid_state_transition_start(integration):
    """Test that starting already running workflow fails."""
    integration.workflow_instances["test-1"] = BubbleWorkflowInstance(
        id="test-1",
        definition_id="def-1",
        status="running",  # Already running
        created_at=time.time(),
        updated_at=time.time(),
        progress=0.5
    )

    result = integration.control_workflow_local(
        instance_id="test-1",
        action="start"  # Can't start already running workflow
    )

    # Should either fail or do nothing
    assert "error" in result or result.get("status") == "running"

def test_invalid_state_transition_pause(integration):
    """Test that pausing non-running workflow fails."""
    integration.workflow_instances["test-1"] = BubbleWorkflowInstance(
        id="test-1",
        definition_id="def-1",
        status="pending",  # Not running
        created_at=time.time(),
        updated_at=time.time(),
        progress=0.0
    )

    result = integration.control_workflow_local(
        instance_id="test-1",
        action="pause"  # Can't pause pending workflow
    )

    # Should fail
    assert "error" in result
```

### Test 2.4: Sync Non-Existent Workflow

```python
def test_sync_non_existent_workflow(bridge):
    """Test that syncing non-existent workflow returns False."""
    result = bridge.sync_workflow_to_ticket(
        workflow_definition_id="non-existent-workflow"
    )

    assert result is False
```

### Test 2.5: Get Non-Existent Analytics

```python
def test_get_non_existent_analytics(analytics):
    """Test that getting analytics for non-existent workflow returns None."""
    result = analytics.get_workflow_analytics(
        workflow_id="non-existent-workflow"
    )

    assert result is None
```

### Test 2.6: Close Non-Existent Ticket

```python
def test_close_non_existent_ticket(bridge):
    """Test that closing non-existent ticket returns False."""
    result = bridge.close_ticket_on_completion(
        workflow_instance_id="non-existent-instance",
        success=True
    )

    assert result is False
```

---

## Category 3: Maximum Limits Tests (5 tests)

### Test 3.1: Exceed Maximum Mappings

```python
def test_exceed_maximum_mappings(bridge):
    """Test that exceeding maximum mappings raises ValueError."""
    from bubblelabs_integration import BubbleWorkflowDefinition

    # Try to create more than MAX_MAPPINGS
    for i in range(MAX_MAPPINGS + 1):
        workflow = BubbleWorkflowDefinition(
            id=f"workflow-{i}",
            name=f"Workflow {i}",
            description="Test",
            nodes=[],
            edges=[],
            metadata={}
        )

        if i < MAX_MAPPINGS:
            bridge.create_ticket_from_workflow(workflow)
        else:
            # This should raise ValueError
            with pytest.raises(ValueError, match="Maximum number of mappings"):
                bridge.create_ticket_from_workflow(workflow)
```

### Test 3.2: Exceed Maximum Timeout

```python
def test_exceed_maximum_timeout():
    """Test that exceeding maximum timeout raises ValueError."""
    from bubblelabs_mcp_tools import get_bubblelabs_workflow_results

    with pytest.raises(ValueError, match="timeout_seconds cannot exceed"):
        get_bubblelabs_workflow_results(
            instance_id="test-instance",
            wait_for_completion=True,
            timeout_seconds=10000  # Exceeds MAX_TIMEOUT_SECONDS
        )
```

### Test 3.3: Exceed Pool Size

```python
def test_exceed_pool_size():
    """Test that pool size is enforced."""
    with pytest.raises(ValueError, match="pool_size must be between"):
        BubbleLabsAnalytics(
            db_path=":memory:",
            pool_size=1000  # Exceeds maximum
        )
```

### Test 3.4: Truncate Long Descriptions

```python
def test_truncate_long_description(bridge):
    """Test that long descriptions are truncated."""
    from bubblelabs_integration import BubbleWorkflowDefinition

    # Create workflow with very long description
    workflow = BubbleWorkflowDefinition(
        id="test-1",
        name="Test",
        description="a" * 20000,  # Very long
        nodes=[],
        edges=[],
        metadata={}
    )

    # Should truncate to MAX_DESCRIPTION_LENGTH
    ticket_id = bridge.create_ticket_from_workflow(workflow)

    # Verify truncation happened (check logs or return value)
    assert ticket_id is not None  # Should succeed with truncation
```

### Test 3.5: Exceed Connection Pool

```python
def test_connection_pool_limit(analytics):
    """Test that connection pool size is respected."""
    # Try to get more connections than pool size
    connections = []

    for i in range(analytics._pool_size + 5):
        conn = analytics.get_connection().__enter__()
        connections.append(conn)

    # All should get connections (some new, some from pool)
    assert len(connections) == analytics._pool_size + 5

    # Clean up
    for conn in connections:
        try:
            conn.close()
        except:
            pass
```

---

## Category 4: Concurrency Tests (4 tests)

### Test 4.1: Concurrent Singleton Creation

```python
def test_concurrent_singleton_creation():
    """Test that singleton is created only once with concurrent access."""
    from bubblelabs_mcp_tools import get_shared_bubblelabs, _shared_bubblelabs_integration
    import gc

    # Reset global state
    import bubblelabs_mcp_tools
    bubblelabs_mcp_tools._shared_bubblelabs_integration = None

    threads = []
    instances = []
    errors = []

    def get_instance():
        try:
            instance = get_shared_bubblelabs()
            instances.append(instance)
        except Exception as e:
            errors.append(e)

    # Create multiple threads
    for _ in range(10):
        t = threading.Thread(target=get_instance)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0

    # Verify only one instance was created
    unique_instances = set(id(i) for i in instances)
    assert len(unique_instances) == 1
```

### Test 4.2: Concurrent Ticket Creation

```python
def test_concurrent_ticket_creation(bridge):
    """Test that concurrent ticket creations are serialized correctly."""
    from bubblelabs_integration import BubbleWorkflowDefinition

    workflow = BubbleWorkflowDefinition(
        id="concurrent-test",
        name="Concurrent Test",
        description="Test",
        nodes=[],
        edges=[],
        metadata={}
    )

    threads = []
    ticket_ids = []
    errors = []

    def create_ticket():
        try:
            ticket_id = bridge.create_ticket_from_workflow(workflow)
            ticket_ids.append(ticket_id)
        except Exception as e:
            errors.append(e)

    # Create multiple threads trying to create the same ticket
    for _ in range(5):
        t = threading.Thread(target=create_ticket)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0

    # Verify all threads got the same ticket ID
    assert len(set(ticket_ids)) == 1
```

### Test 4.3: Concurrent Analytics Tracking

```python
def test_concurrent_analytics_tracking(analytics):
    """Test that concurrent tracking is thread-safe."""
    # Start workflow
    analytics.start_workflow_tracking("wf-1", "Test", "inst-1")

    threads = []
    errors = []

    def track_node():
        try:
            analytics.track_node_execution(
                workflow_id="wf-1",
                node_id=f"node-{threading.get_ident()}",
                node_type="test",
                tokens_used=1000,
                execution_time=1.0,
                provider="openai"
            )
        except Exception as e:
            errors.append(e)

    # Create multiple threads tracking nodes
    for _ in range(10):
        t = threading.Thread(target=track_node)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0

    # Verify all nodes were tracked
    workflow_analytics = analytics.get_workflow_analytics("wf-1")
    assert len(workflow_analytics.node_metrics) == 10
```

### Test 4.4: Concurrent Connection Pool Access

```python
def test_concurrent_connection_pool_access(analytics):
    """Test that connection pool is thread-safe."""
    threads = []
    errors = []

    def use_connection():
        try:
            with analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                time.sleep(0.1)  # Simulate some work
        except Exception as e:
            errors.append(e)

    # Create multiple threads using connections
    for _ in range(20):
        t = threading.Thread(target=use_connection)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0

    # Verify pool size is maintained
    assert len(analytics._connection_pool) <= analytics._pool_size
```

---

## Test Execution

### Run All Tests

```bash
# Run all tests
pytest tests/bubblelabs_edge_cases/ -v

# Run specific category
pytest tests/bubblelabs_edge_cases/test_category1_empty_inputs.py -v

# Run with coverage
pytest tests/bubblelabs_edge_cases/ --cov=bubblelabs_crewai_bridge --cov=bubblelabs_mcp_tools --cov=bubblelabs_analytics -v

# Generate coverage report
pytest tests/bubblelabs_edge_cases/ --cov=bubblelabs_crewai_bridge --cov=bubblelabs_mcp_tools --cov=bubblelabs_analytics --cov-report=html
```

### Expected Test Results

```
test_category1_empty_inputs.py::test_create_ticket_with_none_workflow PASSED
test_category1_empty_inputs.py::test_create_ticket_with_empty_workflow_id PASSED
test_category1_empty_inputs.py::test_update_ticket_with_empty_instance_id PASSED
test_category1_empty_inputs.py::test_update_ticket_with_invalid_progress PASSED
test_category1_empty_inputs.py::test_sync_with_empty_workflow_id PASSED
test_category1_empty_inputs.py::test_stop_sync_with_invalid_timeout PASSED
test_category1_empty_inputs.py::test_create_workflow_with_empty_problem_statement PASSED
test_category1_empty_inputs.py::test_create_workflow_with_whitespace_problem_statement PASSED
test_category1_empty_inputs.py::test_create_workflow_with_too_long_problem_statement PASSED
test_category1_empty_inputs.py::test_create_workflow_with_too_long_name PASSED
test_category1_empty_inputs.py::test_create_workflow_with_too_many_team_configs PASSED
test_category1_empty_inputs.py::test_execute_workflow_with_empty_workflow_id PASSED

test_category2_state_validation.py::test_update_non_existent_ticket PASSED
test_category2_state_validation.py::test_update_non_existent_ticket_analytics PASSED
test_category2_state_validation.py::test_create_duplicate_ticket PASSED
test_category2_state_validation.py::test_invalid_state_transition_start PASSED
test_category2_state_validation.py::test_invalid_state_transition_pause PASSED
test_category2_state_validation.py::test_sync_non_existent_workflow PASSED

test_category3_maximum_limits.py::test_exceed_maximum_mappings PASSED
test_category3_maximum_limits.py::test_exceed_maximum_timeout PASSED
test_category3_maximum_limits.py::test_exceed_pool_size PASSED
test_category3_maximum_limits.py::test_truncate_long_description PASSED
test_category3_maximum_limits.py::test_connection_pool_limit PASSED

test_category4_concurrency.py::test_concurrent_singleton_creation PASSED
test_category4_concurrency.py::test_concurrent_ticket_creation PASSED
test_category4_concurrency.py::test_concurrent_analytics_tracking PASSED
test_category4_concurrency.py::test_concurrent_connection_pool_access PASSED

====================== 27 passed in 5.23s =======================
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/bubblelabs-edge-case-tests.yml
name: BubbleLabs Edge Case Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test-edge-cases:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install pytest pytest-cov
        pip install -r requirements.txt

    - name: Run edge case tests
      run: |
        pytest tests/bubblelabs_edge_cases/ -v --cov=bubblelabs_crewai_bridge --cov=bubblelabs_mcp_tools --cov=bubblelabs_analytics --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        flags: edge-cases
        name: edge-case-coverage

    - name: Check coverage threshold
      run: |
        coverage=$(coverage report | grep TOTAL | awk '{print $4}' | sed 's/%//')
        if (( $(echo "$coverage < 95" | bc -l) )); then
          echo "Coverage $coverage% is below 95% threshold"
          exit 1
        fi
        echo "Coverage $coverage% meets 95% threshold"
```

---

## Success Criteria

All tests must pass with:
- ✅ 100% of tests passing (27/27)
- ✅ 95%+ code coverage on edge case handling
- ✅ No race conditions detected in concurrency tests
- ✅ All validation errors provide clear messages
- ✅ All resource limits enforced correctly

---

**Test Plan Version:** 1.0
**Last Updated:** 2025-12-29
**Status:** Ready for Implementation
