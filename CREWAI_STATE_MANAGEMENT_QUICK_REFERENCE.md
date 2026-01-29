# CrewAI State Management - Quick Reference

## Import Patterns

### Basic Import
```python
from crewai_state_management import (
    WorkflowState,
    SolutionAttempt,
    StateManager,
    StateTransitionGuard,
    create_workflow_state,
    create_state_manager
)
```

### SGD Workflow Orchestrator Import
```python
from crewai_state_management import SolutionAttempt
```

### Claudiomiro Bridge Import
```python
from crewai_state_management import (
    WorkflowState,
    SubProblem,
    DecompositionPlan,
    StateManager
)
```

## Common Operations

### Create State Manager
```python
# Basic
state_mgr = StateManager()

# With options
state_mgr = StateManager(
    storage_dir="./crewai_states",
    enable_compression=True,
    max_versions=10
)
```

### Create Workflow State
```python
# Using factory function
state = create_workflow_state(
    workflow_id="wf_001",
    problem_statement="Solve X",
    execution_method=ExecutionMethod.ROMA_MDAP_MAKER
)

# Direct instantiation
state = WorkflowState(
    workflow_id="wf_001",
    problem_statement="Solve X",
    phase=1,
    status=WorkflowStatus.PENDING,
    execution_method=ExecutionMethod.AUTO
)
```

### Create Solution Attempt (sgd_workflow_orchestrator.py compatible)
```python
import time

attempt = SolutionAttempt(
    id="att_001",
    sub_problem_id="sub_001",
    content="Solution content here",
    generated_by_model="gpt-4",
    timestamp=time.time(),
    status="PENDING"  # or "IN_PROGRESS", "COMPLETED", "FAILED", "ROLLED_BACK"
)

# Add to state
state.sub_solutions["sub_001"] = attempt
```

### Save/Load State
```python
# Basic save/load
state_mgr.save_state(state.workflow_id, state)
loaded_state = state_mgr.load_state(state.workflow_id)

# Check existence
exists = state_mgr.state_exists(workflow_id)

# Delete
state_mgr.delete_state(workflow_id)
```

### Versioning
```python
# Save with versioning
version_id = state_mgr.save_state_with_versioning(workflow_id, state)

# List versions
versions = state_mgr.get_state_versions(workflow_id)

# Load specific version
old_state = state_mgr.load_state_version(workflow_id, version_id)

# Rollback
success = state_mgr.rollback_to_version(workflow_id, version_id)
```

### Snapshots
```python
# Create snapshot
snapshot_id = state_mgr.create_snapshot(
    workflow_id,
    snapshot_name="checkpoint_1"
)

# List snapshots
snapshots = state_mgr.list_snapshots(workflow_id)

# Restore snapshot
success = state_mgr.restore_snapshot(workflow_id, snapshot_id)
```

### Export/Import
```python
# Export
state_mgr.export_state(workflow_id, "./backup.json")

# Import
imported_state = state_mgr.import_state(
    "./backup.json",
    workflow_id="wf_002"  # Optional new ID
)
```

### Query & Inspection
```python
# List all workflows
all_workflows = state_mgr.list_workflows()

# Filter by status
completed = state_mgr.list_workflows(status=WorkflowStatus.COMPLETED)

# Get summary (fast, doesn't load full state)
summary = state_mgr.get_state_summary(workflow_id)
```

### Maintenance
```python
# Cleanup old states
cleaned = state_mgr.cleanup_old_states(max_age_days=30)
```

## State Transitions

### Using StateTransitionGuard
```python
guard = StateTransitionGuard()

# Check if valid
is_valid = guard.validate_transition(
    current_status=WorkflowStatus.IN_PROGRESS,
    new_status=WorkflowStatus.COMPLETED
)

# Execute with validation
try:
    state = guard.guard_transition(state, WorkflowStatus.COMPLETED)
except ValueError as e:
    print(f"Invalid transition: {e}")
```

### Valid Transitions
```
PENDING → IN_PROGRESS, CANCELLED
IN_PROGRESS → SETUP_COMPLETE, FAILED, CANCELLED
SETUP_COMPLETE → SOLVING, FAILED, CANCELLED
SOLVING → CRITIQUE, FAILED, CANCELLED
CRITIQUE → VERIFYING, FAILED, CANCELLED
VERIFYING → REASSEMBLING, FAILED, CANCELLED
REASSEMBLING → FINAL_VALIDATION, FAILED, CANCELLED
FINAL_VALIDATION → COMPLETED, FAILED, CANCELLED
COMPLETED → (terminal)
FAILED → (terminal)
CANCELLED → (terminal)
```

## Enums

### ExecutionMethod
- `TRADITIONAL`
- `ROMA`
- `ROMA_MDAP_MAKER` (Zero-Error)
- `CLAUDIOMIRO`
- `DATAPIZZA`
- `HYBRID`
- `AUTO`

### WorkflowStatus
- `PENDING`
- `IN_PROGRESS`
- `SETUP_COMPLETE`
- `SOLVING`
- `CRITIQUE`
- `VERIFYING`
- `REASSEMBLING`
- `FINAL_VALIDATION`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

## Common Patterns

### Pattern 1: Workflow with Versioning
```python
state_mgr = StateManager(max_versions=5)
state = create_workflow_state("wf_001", "Problem")

# Initial save
version_1 = state_mgr.save_state_with_versioning("wf_001", state)

# Make changes
state.phase = 2
version_2 = state_mgr.save_state_with_versioning("wf_001", state)

# If something goes wrong, rollback
if error_occurred:
    state_mgr.rollback_to_version("wf_001", version_1)
```

### Pattern 2: Checkpoint Before Major Phase
```python
# Before starting validation phase
snapshot_id = state_mgr.create_snapshot(
    workflow_id,
    snapshot_name="before_validation"
)

# Run validation
try:
    results = run_validation(state)
    state.verification_results = results
    state_mgr.save_state(workflow_id, state)
except Exception as e:
    # Restore if validation fails
    state_mgr.restore_snapshot(workflow_id, snapshot_id)
```

### Pattern 3: Regular Backups
```python
from datetime import datetime

# Export with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
state_mgr.export_state(
    workflow_id,
    f"./backups/{workflow_id}_{timestamp}.json"
)
```

### Pattern 4: State Query Dashboard
```python
# Get overview of all workflows
summaries = []
for wf_id in state_mgr.list_workflows():
    summary = state_mgr.get_state_summary(wf_id)
    summaries.append(summary)

# Filter by status
running = [s for s in summaries if s['status'] == 'in_progress']
completed = [s for s in summaries if s['status'] == 'completed']

print(f"Running: {len(running)}, Completed: {len(completed)}")
```

## Error Handling

```python
try:
    state = state_mgr.load_state(workflow_id)
    if state is None:
        raise ValueError(f"Workflow {workflow_id} not found")
except Exception as e:
    logger.error(f"State management error: {e}")
    # Handle error appropriately
```

## Performance Tips

1. **Enable compression** for large states
   ```python
   state_mgr = StateManager(enable_compression=True)
   ```

2. **Limit versions** to prevent disk bloat
   ```python
   state_mgr = StateManager(max_versions=10)
   ```

3. **Use summary** instead of full load when possible
   ```python
   # Fast
   summary = state_mgr.get_state_summary(workflow_id)

   # Slow
   state = state_mgr.load_state(workflow_id)
   ```

4. **Regular cleanup** of old states
   ```python
   state_mgr.cleanup_old_states(max_age_days=30)
   ```

## Testing

```python
import pytest
from crewai_state_management import create_workflow_state, StateManager

def test_workflow_lifecycle():
    state_mgr = StateManager(storage_dir="./test_states")

    # Create
    state = create_workflow_state("test_001", "Test")

    # Save
    state_mgr.save_state("test_001", state)

    # Load
    loaded = state_mgr.load_state("test_001")
    assert loaded.workflow_id == "test_001"

    # Cleanup
    import shutil
    shutil.rmtree("./test_states")
```

## File Locations

- **State files**: `{storage_dir}/{workflow_id}.json` or `.json.gz`
- **Version files**: `{storage_dir}/{workflow_id}_v{timestamp}.json.gz`
- **Snapshot files**: `{storage_dir}/{workflow_id}_{snapshot_name}.json.gz`
- **Version registry**: `{storage_dir}/.versions.json`
