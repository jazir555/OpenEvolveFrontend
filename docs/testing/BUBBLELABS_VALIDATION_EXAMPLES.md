# BubbleLabs Edge Case Validation - Code Examples

This document provides the key validation code snippets that were added to fix all 27 HIGH severity edge case issues.

---

## 1. Validation Helper Functions

These functions are used across all BubbleLabs modules:

```python
def validate_not_none(value: Any, param_name: str) -> Any:
    """Validate that a value is not None."""
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    return value


def validate_not_empty(value: str, param_name: str) -> str:
    """Validate that a string is not empty or just whitespace."""
    if not value or not value.strip():
        raise ValueError(f"{param_name} cannot be empty or whitespace")
    return value


def validate_string_length(value: str, max_length: int, param_name: str) -> str:
    """Validate string length."""
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    if len(value) > max_length:
        raise ValueError(f"{param_name} cannot exceed {max_length} characters")
    return value


def validate_dict_size(value: Dict[str, Any], max_size: int, param_name: str) -> Dict[str, Any]:
    """Validate dictionary size."""
    if value is None:
        return {}
    if len(value) > max_size:
        raise ValueError(f"{param_name} cannot exceed {max_size} entries")
    return value


def validate_range(value: int, min_value: int, max_value: int, param_name: str) -> int:
    """Validate numeric range."""
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    if value < min_value or value > max_value:
        raise ValueError(f"{param_name} must be between {min_value} and {max_value}")
    return value
```

---

## 2. Validation Constants

Each module defines its own validation constants:

```python
# bubblelabs_crewai_bridge.py
MAX_MAPPINGS = 1000
MAX_DESCRIPTION_LENGTH = 10000
MAX_SYNC_INTERVAL = 3600
MAX_BATCH_SIZE = 100

# bubblelabs_mcp_tools.py
MAX_PROBLEM_STATEMENT_LENGTH = 10000
MAX_WORKFLOW_NAME_LENGTH = 255
MAX_TEAM_CONFIG_ENTRIES = 50
MAX_TIMEOUT_SECONDS = 3600
MAX_PARAMETERS_COUNT = 100

# bubblelabs_analytics.py
MAX_WORKFLOWS_TRACKED = 10000
MAX_NODES_PER_WORKFLOW = 1000
MAX_WORKFLOW_EXECUTION_TIME = 86400
MAX_TOKEN_COUNT = 10**12

# bubblelabs_integration.py
MAX_CONCURRENT_INSTANCES = 100
MAX_WORKFLOW_INSTANCES_PER_DEFINITION = 1000
MAX_DEFINITION_HISTORY = 100

# bubblelabs_security.py
MAX_SESSIONS_PER_USER = 10
MAX_RATE_LIMIT_WINDOW = 3600
MAX_PERMISSION_ENTRIES_PER_USER = 100
MAX_CSRF_TOKENS_PER_SESSION = 100
```

---

## 3. Input Validation Examples

### 3.1 Validating Workflow Creation

```python
@mcp_tool("create_bubblelabs_workflow")
def create_bubblelabs_workflow(
    problem_statement: str,
    team_config: Optional[Dict[str, str]] = None,
    gauntlet_config: Optional[Dict[str, str]] = None,
    workflow_name: Optional[str] = None,
    workflow_type: str = "sovereign_decomposition",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a BubbleLabs workflow from a problem statement.

    Raises:
        ValueError: If problem_statement is empty or exceeds max length
    """
    # Input validation
    validate_not_empty(problem_statement, "problem_statement")
    validate_string_length(problem_statement, MAX_PROBLEM_STATEMENT_LENGTH, "problem_statement")

    if workflow_name:
        validate_string_length(workflow_name, MAX_WORKFLOW_NAME_LENGTH, "workflow_name")

    # Validate config sizes
    team_config = validate_dict_size(team_config or {}, MAX_TEAM_CONFIG_ENTRIES, "team_config")
    gauntlet_config = validate_dict_size(gauntlet_config or {}, MAX_TEAM_CONFIG_ENTRIES, "gauntlet_config")

    # Continue with workflow creation...
```

### 3.2 Validating Ticket Creation

```python
def create_ticket_from_workflow(
    self,
    workflow_definition: BubbleWorkflowDefinition,
    assignee: Optional[str] = None,
    additional_labels: Optional[List[str]] = None
) -> Optional[str]:
    """
    Create a CrewAI ticket from a BubbleLabs workflow definition.

    Raises:
        ValueError: If workflow_definition is None or empty
    """
    # Input validation
    validate_not_none(workflow_definition, "workflow_definition")
    validate_not_empty(workflow_definition.id, "workflow_definition.id")
    validate_not_empty(workflow_definition.name, "workflow_definition.name")

    # State validation: Check if workflow already has a ticket
    with self.lock:
        if workflow_definition.id in self.mappings:
            logger.warning(f"Workflow {workflow_definition.id} already has a ticket")
            return self.mappings[workflow_definition.id].ticket_id

        # Maximum limit validation
        if len(self.mappings) >= MAX_MAPPINGS:
            raise ValueError(f"Maximum number of mappings ({MAX_MAPPINGS}) reached")

    # Continue with ticket creation...
```

### 3.3 Validating Analytics Tracking

```python
def start_workflow_tracking(
    self,
    workflow_id: str,
    workflow_name: str,
    instance_id: str
) -> bool:
    """
    Start tracking a workflow execution.

    Raises:
        ValueError: If any parameter is empty
    """
    # Input validation
    if not workflow_id or not workflow_id.strip():
        raise ValueError("workflow_id cannot be empty")
    if not workflow_name or not workflow_name.strip():
        raise ValueError("workflow_name cannot be empty")
    if not instance_id or not instance_id.strip():
        raise ValueError("instance_id cannot be empty")

    try:
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO workflows
                    (workflow_id, workflow_name, instance_id, start_time, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (workflow_id, workflow_name, instance_id, time.time(), "running"))

                conn.commit()

        logger.info(f"Started tracking workflow: {workflow_id} (instance: {instance_id})")
        return True

    except Exception as e:
        logger.error(f"Error starting workflow tracking: {e}")
        return False
```

---

## 4. State Validation Examples

### 4.1 Checking State Before Updates

```python
def update_ticket_progress(
    self,
    workflow_instance_id: str,
    progress: float,
    status: WorkflowStatus,
    metrics: Optional[WorkflowMetrics] = None
) -> bool:
    """
    Update CrewAI ticket with workflow progress.

    Raises:
        ValueError: If workflow_instance_id is empty or progress is out of range
    """
    # Input validation
    validate_not_empty(workflow_instance_id, "workflow_instance_id")
    if progress < 0.0 or progress > 1.0:
        raise ValueError(f"progress must be between 0.0 and 1.0, got {progress}")
    validate_not_none(status, "status")

    # State validation: Check if instance exists before updating
    with self.lock:
        mapping = self._find_mapping_by_instance_id(workflow_instance_id)
        if not mapping or not mapping.ticket_id:
            logger.warning(f"No ticket found for workflow instance {workflow_instance_id}")
            return False

    # Continue with update...
```

### 4.2 Validating Workflow Exists Before Sync

```python
def sync_workflow_to_ticket(self, workflow_definition_id: str) -> bool:
    """
    Sync workflow definition to existing ticket.

    Raises:
        ValueError: If workflow_definition_id is empty
    """
    # Input validation
    validate_not_empty(workflow_definition_id, "workflow_definition_id")

    # State validation: Check if ticket exists before syncing
    with self.lock:
        mapping = self.mappings.get(workflow_definition_id)
        if not mapping or not mapping.ticket_id:
            logger.warning(f"No ticket found for workflow {workflow_definition_id}")
            return False

    # Continue with sync...
```

### 4.3 Checking Instance Status Before Control

```python
def control_workflow_local(self, instance_id: str, action: str) -> Dict[str, Any]:
    """
    Control a running workflow instance locally.

    Raises:
        ValueError: If action is not valid for current state
    """
    # Input validation
    validate_not_empty(instance_id, "instance_id")
    validate_not_empty(action, "action")

    # Validate action against whitelist
    allowed_actions = {"start", "pause", "resume", "cancel", "restart"}
    if action not in allowed_actions:
        raise ValueError(f"Invalid action: {action}. Valid actions: {allowed_actions}")

    # Thread-safe: atomic check-then-act with lock
    with self._instances_lock:
        if instance_id not in self.workflow_instances:
            return {"error": "Workflow instance not found"}

        instance = self.workflow_instances[instance_id]

        # State validation: Check if action is valid for current state
        if action == "start" and instance.status not in ["pending", "created"]:
            return {"error": f"Cannot start workflow in state: {instance.status}"}
        elif action == "pause" and instance.status != "running":
            return {"error": f"Cannot pause workflow in state: {instance.status}"}
        elif action == "resume" and instance.status != "paused":
            return {"error": f"Cannot resume workflow in state: {instance.status}"}

    # Continue with control operation...
```

---

## 5. Maximum Limits Enforcement Examples

### 5.1 Enforcing Mapping Limits

```python
def create_ticket_from_workflow(self, workflow_definition: BubbleWorkflowDefinition, ...) -> Optional[str]:
    # Maximum limit validation
    with self.lock:
        if len(self.mappings) >= MAX_MAPPINGS:
            raise ValueError(f"Maximum number of mappings ({MAX_MAPPINGS}) reached")

    # Description length limit
    description = self._build_ticket_description(workflow_definition)
    if len(description) > MAX_DESCRIPTION_LENGTH:
        logger.warning(f"Description exceeds {MAX_DESCRIPTION_LENGTH} chars, truncating")
        description = description[:MAX_DESCRIPTION_LENGTH]

    # Continue with creation...
```

### 5.2 Enforcing Timeout Limits

```python
def get_bubblelabs_workflow_results(
    instance_id: str,
    wait_for_completion: bool = False,
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """
    Get the results of a completed BubbleLabs workflow.

    Raises:
        ValueError: If timeout_seconds exceeds maximum
    """
    # Input validation
    validate_not_empty(instance_id, "instance_id")

    if timeout_seconds < 0:
        raise ValueError("timeout_seconds cannot be negative")
    if timeout_seconds > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds cannot exceed {MAX_TIMEOUT_SECONDS}")

    # Continue with execution...
```

### 5.3 Enforcing Pool Size Limits

```python
def __init__(self, db_path: Optional[str] = None, pool_size: int = 5):
    """
    Initialize analytics tracker.

    Raises:
        ValueError: If pool_size is out of valid range
    """
    # Input validation
    if pool_size is not None:
        validate_range(pool_size, 1, 50, "pool_size")

    self._pool_size = pool_size
    self._connection_pool: List[sqlite3.Connection] = []
    self._pool_lock = threading.Lock()

    # Continue with initialization...
```

---

## 6. Concurrency Safety Examples

### 6.1 Thread-Safe Singleton Pattern

```python
_shared_bubblelabs_integration = None
_singleton_lock = Lock()

def get_shared_bubblelabs() -> BubbleLabsIntegration:
    """
    Get or create the shared BubbleLabsIntegration instance.

    Thread-safe singleton with double-check locking pattern.
    """
    global _shared_bubblelabs_integration

    # First check (no lock) - fast path for already-initialized singleton
    if _shared_bubblelabs_integration is not None:
        return _shared_bubblelabs_integration

    # Lock for initialization
    with _singleton_lock:
        # Second check (with lock) - prevent race condition
        if _shared_bubblelabs_integration is None:
            _shared_bubblelabs_integration = BubbleLabsIntegration()
            logger.info("Created shared BubbleLabs integration instance (thread-safe)")

    return _shared_bubblelabs_integration
```

### 6.2 Serializing Critical Operations

```python
def create_ticket_from_workflow(self, workflow_definition: BubbleWorkflowDefinition, ...) -> Optional[str]:
    # Serialization: Check and create ticket atomically
    with self.lock:
        # Check if already exists
        if workflow_definition.id in self.mappings:
            return self.mappings[workflow_definition.id].ticket_id

        # Check limits
        if len(self.mappings) >= MAX_MAPPINGS:
            raise ValueError(f"Maximum number of mappings ({MAX_MAPPINGS}) reached")

    # Perform API call OUTSIDE of lock to avoid holding during I/O
    ticket_id = self.crewai.create_ticket(...)

    if ticket_id:
        # Update cache atomically
        with self.lock:
            mapping = WorkflowTicketMapping(workflow_definition.id)
            mapping.ticket_id = ticket_id
            self.mappings[workflow_definition.id] = mapping

    return ticket_id
```

### 6.3 Connection Pooling with Thread Safety

```python
@contextmanager
def get_connection(self):
    """
    Context manager for database connections with connection pooling.

    Thread-safe connection pooling to reuse connections.
    """
    conn = None
    try:
        # Try to get connection from pool (thread-safe)
        with self._pool_lock:
            if self._connection_pool:
                conn = self._connection_pool.pop()
                logger.debug(f"Reusing connection from pool (pool size: {len(self._connection_pool)})")

        # Create new connection if pool was empty
        if conn is None:
            conn = sqlite3.connect(self.db_path)

        yield conn

        # Return connection to pool on success (thread-safe)
        with self._pool_lock:
            if len(self._connection_pool) < self._pool_size:
                self._connection_pool.append(conn)
                conn = None  # Mark as returned to pool

    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        # Close connection if not returned to pool
        if conn is not None:
            conn.close()
```

---

## 7. Security Validation Examples

### 7.1 UUID Validation

```python
def validate_uuid(instance_id: str, param_name: str = "instance_id") -> str:
    """
    Validate that a string is a valid UUID.

    Raises:
        ValidationError: If validation fails
    """
    if not instance_id or not isinstance(instance_id, str):
        raise ValidationError(f"{param_name} must be a non-empty string")

    try:
        # Validate UUID format
        uuid.UUID(instance_id)
        return instance_id
    except ValueError:
        raise ValidationError(f"{param_name} must be a valid UUID format: {instance_id}")
```

### 7.2 Workflow Type Whitelist Validation

```python
ALLOWED_WORKFLOW_TYPES = {
    'evolution',
    'adversarial',
    'sovereign',
    'sovereign_decomposition',
    'bubblelabs_openevolve'
}

def validate_workflow_type(workflow_type: str) -> str:
    """
    Validate workflow_type against whitelist.

    Raises:
        ValidationError: If validation fails
    """
    if not workflow_type or not isinstance(workflow_type, str):
        raise ValidationError("workflow_type must be a non-empty string")

    workflow_type = workflow_type.strip().lower()

    if workflow_type not in ALLOWED_WORKFLOW_TYPES:
        raise ValidationError(
            f"Invalid workflow_type '{workflow_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_WORKFLOW_TYPES))}"
        )

    return workflow_type
```

### 7.3 Path Traversal Protection

```python
def validate_output_path(output_path: str, allowed_base_dir: Optional[str] = None) -> str:
    """
    Validate and sanitize the output path to prevent path traversal attacks.

    Raises:
        ValueError: If path is invalid or contains path traversal attempts
    """
    if not output_path:
        raise ValueError("Output path cannot be empty")

    # Convert to absolute path
    abs_path = os.path.abspath(output_path)

    # Check for path traversal attempts
    if ".." in output_path or output_path.startswith("~/"):
        raise ValueError(f"Path traversal detected in output path: {output_path}")

    # If base directory is specified, ensure the path is within it
    if allowed_base_dir:
        allowed_base = os.path.abspath(allowed_base_dir)
        if not abs_path.startswith(allowed_base):
            raise ValueError(f"Output path must be within {allowed_base_dir}")

    return abs_path
```

---

## 8. Error Handling Patterns

### 8.1 Consistent Error Returns

```python
def create_bubblelabs_workflow(...) -> Dict[str, Any]:
    """Create a workflow with consistent error handling."""
    try:
        # Input validation
        validate_not_empty(problem_statement, "problem_statement")
        validate_string_length(problem_statement, MAX_PROBLEM_STATEMENT_LENGTH, "problem_statement")

        # Business logic
        integration = get_shared_bubblelabs()
        definition = integration.create_workflow_definition_from_openevolve(...)

        return {
            "success": True,
            "workflow_id": definition.id,
            "message": f"Workflow '{definition.name}' created successfully"
        }

    except ValueError as e:
        # Input validation errors
        logger.warning(f"Validation error: {e}")
        return {
            "success": False,
            "error": "Invalid input",
            "message": str(e)
        }

    except Exception as e:
        # Unexpected errors
        logger.error(f"Error creating BubbleLabs workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create workflow: {str(e)}"
        }
```

---

## 9. Testing Examples

### 9.1 Testing Empty Input Validation

```python
def test_empty_validation():
    """Test that empty inputs are rejected."""
    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_not_empty("", "test_param")

    # Whitespace only
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_not_empty("   ", "test_param")

    # None value
    with pytest.raises(ValueError, match="cannot be None"):
        validate_not_none(None, "test_param")
```

### 9.2 Testing Maximum Limits

```python
def test_maximum_limits():
    """Test that maximum limits are enforced."""
    # String length limit
    with pytest.raises(ValueError, match="cannot exceed.*characters"):
        validate_string_length("a" * 1001, 1000, "test_param")

    # Dictionary size limit
    with pytest.raises(ValueError, match="cannot exceed.*entries"):
        validate_dict_size({f"k{i}": f"v{i}" for i in range(101)}, 100, "test_param")

    # Range limit
    with pytest.raises(ValueError, match="must be between"):
        validate_range(150, 1, 100, "test_param")
```

### 9.3 Testing State Validation

```python
def test_state_validation():
    """Test that state is validated before operations."""
    bridge = BubbleLabsCrewAIBridge()

    # Update non-existent ticket
    result = bridge.update_ticket_progress("non-existent-id", 0.5, WorkflowStatus.RUNNING)
    assert result is False

    # Create duplicate ticket
    workflow = BubbleWorkflowDefinition(...)
    ticket_id_1 = bridge.create_ticket_from_workflow(workflow)
    ticket_id_2 = bridge.create_ticket_from_workflow(workflow)
    assert ticket_id_1 == ticket_id_2  # Should return existing ticket
```

### 9.4 Testing Concurrency Safety

```python
def test_concurrent_singleton():
    """Test that singleton creation is thread-safe."""
    threads = []
    instances = []

    def get_instance():
        instance = get_shared_bubblelabs()
        instances.append(instance)

    # Create multiple threads
    for _ in range(10):
        t = Thread(target=get_instance)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    # Verify only one instance was created
    assert len(set(id(i) for i in instances)) == 1
```

---

## Summary

These validation patterns provide comprehensive edge case handling:

✅ **Input Validation** - All parameters validated for None, empty, and range checks
✅ **State Validation** - All operations check state before mutations
✅ **Resource Limits** - Maximum limits enforced on all resources
✅ **Concurrency Safety** - Thread-safe with proper locking and serialization
✅ **Security** - Path traversal protection, whitelists, and format validation
✅ **Error Handling** - Consistent error responses across all APIs

Use these patterns as templates for adding validation to any new BubbleLabs code.

---

**Document Version:** 1.0
**Last Updated:** 2025-12-29
**Author:** Claude (Anthropic)
