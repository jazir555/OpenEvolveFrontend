# CrewAI State Management Module

## Overview

Production-ready state management for CrewAI workflows with persistence, thread-safe operations, versioning, and multi-backend storage support.

## Features

### Core Capabilities

- **Type-Safe Models**: Pydantic-based models with validation
- **State Persistence**: JSON file storage with optional gzip compression
- **Versioning**: Automatic state versioning with rollback support
- **Snapshots**: Named snapshots for checkpoint/restore
- **Export/Import**: State serialization for debugging and migration
- **State Transition Validation**: Guards against invalid workflow state changes
- **Query & Inspection**: List workflows, get summaries, filter by status

### Storage Backends

- **In-Memory**: Default, fastest, no persistence
- **File-Based**: JSON files with optional compression
- **Future**: Database backend via sovereign_persistence (planned)

## Installation

The module is part of the OpenEvolve Frontend project. No additional dependencies beyond Pydantic.

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

## Quick Start

### Basic Usage

```python
from crewai_state_management import create_workflow_state, create_state_manager

# Create state manager
state_mgr = create_state_manager(storage_dir="./crewai_states")

# Create workflow state
state = create_workflow_state(
    workflow_id="workflow_001",
    problem_statement="Solve the optimization problem",
    execution_method=ExecutionMethod.ROMA_MDAP_MAKER
)

# Save state
state_mgr.save_state(state.workflow_id, state)

# Load state
loaded_state = state_mgr.load_state(state.workflow_id)
```

### Advanced Usage with Versioning

```python
# Save with automatic versioning
version_id = state_mgr.save_state_with_versioning(state.workflow_id, state)

# List available versions
versions = state_mgr.get_state_versions(state.workflow_id)

# Rollback to a previous version
state_mgr.rollback_to_version(state.workflow_id, versions[0])
```

### Snapshots

```python
# Create a named snapshot
snapshot_id = state_mgr.create_snapshot(
    state.workflow_id,
    snapshot_name="before_validation"
)

# List snapshots
snapshots = state_mgr.list_snapshots(state.workflow_id)

# Restore from snapshot
state_mgr.restore_snapshot(state.workflow_id, snapshot_id)
```

### Export/Import

```python
# Export state to JSON
state_mgr.export_state(state.workflow_id, "./workflow_backup.json")

# Import state from JSON
imported_state = state_mgr.import_state(
    "./workflow_backup.json",
    workflow_id="workflow_002"  # Optional: assign new ID
)
```

## Core Classes

### WorkflowState

Complete workflow state model.

**Key Fields:**
- `workflow_id`: Unique identifier
- `problem_statement`: Original problem
- `phase`: Current phase (1-6)
- `status`: WorkflowStatus enum
- `execution_method`: ExecutionMethod enum
- `decomposition_plan`: Phase 1 results
- `sub_solutions`: Dict of solution attempts
- `critique_reports`: List of critique reports
- `verification_results`: List of validation results
- `reassembly_result`: Final reassembled solution
- `final_validation`: Phase 6 validation
- `metadata`: Additional context

**Example:**

```python
state = WorkflowState(
    workflow_id="wf_001",
    problem_statement="Design distributed system",
    phase=1,
    status=WorkflowStatus.IN_PROGRESS,
    execution_method=ExecutionMethod.ROMA_MDAP_MAKER
)
```

### SolutionAttempt

Solution attempt model (compatible with sgd_workflow_orchestrator.py).

**Key Fields:**
- `id`: Unique attempt identifier
- `sub_problem_id`: Sub-problem being solved
- `content`: Solution content
- `generated_by_model`: Model/agent name
- `timestamp`: Unix timestamp
- `status`: Status string (PENDING, IN_PROGRESS, COMPLETED, FAILED, ROLLED_BACK)
- `confidence_score`: Quality estimate (0-1)
- `execution_time`: Time taken (seconds)
- `error_message`: Error details if failed

**Example:**

```python
attempt = SolutionAttempt(
    id="att_001",
    sub_problem_id="sub_001",
    content="Use Raft consensus protocol",
    generated_by_model="gpt-4",
    timestamp=time.time(),
    status="PENDING"
)

# Add to workflow state
state.sub_solutions["sub_001"] = attempt
```

### StateManager

Manages state persistence and lifecycle.

**Key Methods:**

#### Basic Operations
- `save_state(workflow_id, state)`: Save workflow state
- `load_state(workflow_id)`: Load workflow state
- `delete_state(workflow_id)`: Delete workflow state
- `state_exists(workflow_id)`: Check if state exists
- `list_workflows(status=None)`: List all workflows (optional status filter)

#### Versioning
- `save_state_with_versioning(workflow_id, state)`: Save with automatic versioning
- `get_state_versions(workflow_id)`: List available versions
- `load_state_version(workflow_id, version_id)`: Load specific version
- `rollback_to_version(workflow_id, version_id)`: Rollback to version

#### Snapshots
- `create_snapshot(workflow_id, snapshot_name=None)`: Create named snapshot
- `list_snapshots(workflow_id)`: List snapshots
- `restore_snapshot(workflow_id, snapshot_id)`: Restore from snapshot

#### Export/Import
- `export_state(workflow_id, export_path)`: Export to JSON file
- `import_state(import_path, workflow_id=None)`: Import from JSON file

#### Query & Inspection
- `get_state_summary(workflow_id)`: Get state summary without loading full state

#### Maintenance
- `cleanup_old_states(max_age_days=30)`: Delete old states

**Example:**

```python
# Initialize
state_mgr = StateManager(
    storage_dir="./crewai_states",
    enable_compression=True,
    max_versions=10
)

# Create workflow
state = create_workflow_state("wf_001", "My problem")

# Save with versioning
version_id = state_mgr.save_state_with_versioning("wf_001", state)

# Create snapshot
snapshot_id = state_mgr.create_snapshot("wf_001", "checkpoint_1")

# Get summary
summary = state_mgr.get_state_summary("wf_001")
print(summary)

# Cleanup
state_mgr.cleanup_old_states(max_age_days=30)
```

### StateTransitionGuard

Validates state transitions to prevent invalid workflow state changes.

**Methods:**
- `validate_transition(current_status, new_status)`: Check if transition is valid
- `guard_transition(state, new_status)`: Execute transition or raise error

**Example:**

```python
guard = StateTransitionGuard()

# Check if transition is valid
is_valid = guard.validate_transition(
    WorkflowStatus.IN_PROGRESS,
    WorkflowStatus.COMPLETED
)

# Execute transition with validation
try:
    state = guard.guard_transition(state, WorkflowStatus.COMPLETED)
except ValueError as e:
    print(f"Invalid transition: {e}")
```

## Enums

### ExecutionMethod

Available execution methods:
- `TRADITIONAL`: Traditional sequential execution
- `ROMA`: Roma-based execution
- `ROMA_MDAP_MAKER`: Roma + MDAP + MAKER (zero-error)
- `CLAUDIOMIRO`: Claudiomiro-based execution
- `DATAPIZZA`: Datapizza-based execution
- `HYBRID`: Hybrid approach
- `AUTO`: Automatic method selection

### WorkflowStatus

Workflow status values:
- `PENDING`: Workflow created, not started
- `IN_PROGRESS`: Workflow running
- `SETUP_COMPLETE`: Phase 1 (setup) complete
- `SOLVING`: Phase 2 (solving sub-problems)
- `CRITIQUE`: Phase 3 (critique solutions)
- `VERIFYING`: Phase 4 (verification)
- `REASSEMBLING`: Phase 5 (reassembly)
- `FINAL_VALIDATION`: Phase 6 (final validation)
- `COMPLETED`: Workflow complete
- `FAILED`: Workflow failed
- `CANCELLED`: Workflow cancelled

## Best Practices

### 1. Use Versioning for Critical Workflows

```python
# Always use versioning for important workflows
version_id = state_mgr.save_state_with_versioning(workflow_id, state)
```

### 2. Create Snapshots Before Major Changes

```python
# Snapshot before validation phase
snapshot_id = state_mgr.create_snapshot(
    workflow_id,
    snapshot_name="before_phase_4_validation"
)
```

### 3. Export States for Backup

```python
# Regular exports for backup
state_mgr.export_state(workflow_id, f"./backups/{workflow_id}_{date}.json")
```

### 4. Use State Transition Guards

```python
# Validate transitions to prevent invalid states
guard = StateTransitionGuard()
state = guard.guard_transition(state, new_status)
```

### 5. Cleanup Old States Regularly

```python
# Scheduled cleanup
cleaned = state_mgr.cleanup_old_states(max_age_days=30)
```

## Error Handling

```python
from crewai_state_management import StateManager

try:
    state = state_mgr.load_state(workflow_id)
    if not state:
        print(f"Workflow {workflow_id} not found")
except Exception as e:
    logger.error(f"Failed to load state: {e}")
    # Handle error appropriately
```

## Performance Considerations

### Compression

Enable gzip compression for large states:

```python
state_mgr = StateManager(
    storage_dir="./crewai_states",
    enable_compression=True  # Reduces disk usage by ~80%
)
```

### Version Limits

Limit versions to prevent disk bloat:

```python
state_mgr = StateManager(
    storage_dir="./crewai_states",
    max_versions=10  # Keep only 10 most recent versions
)
```

### State Summary

Use `get_state_summary()` instead of `load_state()` when you only need metadata:

```python
# Fast: Only loads summary
summary = state_mgr.get_state_summary(workflow_id)

# Slower: Loads full state
state = state_mgr.load_state(workflow_id)
```

## Integration with CrewAI

```python
from crewai import Crew, Process
from crewai_state_management import WorkflowState, StateManager

# Initialize state manager
state_mgr = StateManager()

# Create workflow state
state = create_workflow_state(
    workflow_id="crewai_wf_001",
    problem_statement="Optimize database queries"
)

# Execute CrewAI workflow
crew = Crew(
    agents=my_agents,
    tasks=my_tasks,
    process=Process.sequential
)

result = crew.kickoff()

# Update state with results
state.status = WorkflowStatus.COMPLETED
state.sub_solutions["final"] = SolutionAttempt(
    id="final",
    sub_problem_id="final",
    content=str(result),
    generated_by_model="crewai",
    timestamp=time.time(),
    status="COMPLETED"
)

# Save final state
state_mgr.save_state(state.workflow_id, state)
```

## Testing

```python
import pytest
from crewai_state_management import create_workflow_state, StateManager

def test_state_lifecycle():
    state_mgr = StateManager(storage_dir="./test_states")

    # Create
    state = create_workflow_state("test_001", "Test problem")

    # Save
    state_mgr.save_state("test_001", state)

    # Load
    loaded = state_mgr.load_state("test_001")
    assert loaded.workflow_id == "test_001"

    # Delete
    assert state_mgr.delete_state("test_001") == True
    assert state_mgr.state_exists("test_001") == False
```

## Troubleshooting

### Issue: "State file corrupted"

**Solution:** Use versioning to rollback:

```python
versions = state_mgr.get_state_versions(workflow_id)
if versions:
    state_mgr.rollback_to_version(workflow_id, versions[-2])
```

### Issue: "Disk space full"

**Solution:** Enable compression and cleanup:

```python
state_mgr = StateManager(enable_compression=True)
state_mgr.cleanup_old_states(max_age_days=7)
```

### Issue: "Performance slow"

**Solution:** Use in-memory backend for testing:

```python
# For testing only - no persistence
state_mgr = StateManager(backend=StorageBackend.MEMORY)
```

## API Reference

See the module docstrings for complete API documentation:

```bash
python -m pydoc crewai_state_management
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: [OpenEvolve Frontend](https://github.com/openevolve/frontend)
- Documentation: See ARCHITECTURE.md
