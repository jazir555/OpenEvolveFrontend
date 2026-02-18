# MDAP/MAKER-CrewAI Integration Guide

## Overview

This guide explains the integration between MDAP/MAKER systems and the CrewAI project management system in OpenEvolve. This integration enables you to track, monitor, and manage MDAP (Multi-step Debate and Aggregation Protocol) and MAKER workflows through CrewAI tickets.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [MDAP Integration](#mdap-integration)
5. [MAKER Integration](#maker-integration)
6. [Combined Workflows](#combined-workflows)
7. [API Reference](#api-reference)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is MDAP?

MDAP (Multi-step Debate and Aggregation Protocol) is a multi-agent voting system that uses:
- **Voting-based consensus** with first-to-ahead-by-k mechanism
- **Red-flagging** to discard unreliable responses
- **Retry logic** with configurable max attempts
- **Team-based agent selection** with specialization

### What is MAKER?

MAKER (Maximal Agentic decomposition, first-to-ahead-by-k Error correction, and Red-flagging) is a framework for solving complex tasks through:
- **Recursive decomposition** of complex tasks
- **First-to-ahead-by-k voting** for error correction
- **Red-flagging** for quality control
- **Atomic solving** for simple tasks

### What is CrewAI?

CrewAI is a project management system that provides:
- **Ticket tracking** for tasks and sub-tasks
- **Status management** (TODO, IN_PROGRESS, DONE, etc.)
- **Label-based organization** for filtering and search
- **Integration APIs** for external systems

---

## Architecture

### Integration Components

```
┌─────────────────────────────────────────────────────────┐
│         OpenEvolve Workflow System                       │
│  ┌──────────────┐          ┌──────────────┐           │
│  │ MDAP Engine  │          │ MAKER Engine │           │
│  └──────┬───────┘          └──────┬───────┘           │
│         │                         │                     │
│         └────────┬────────────────┘                    │
│                  │                                     │
│         ┌────────▼────────┐                           │
│         │  Integration    │                           │
│         │   Manager       │                           │
│         └────────┬────────┘                           │
└──────────────────┼────────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   CrewAI API   │
         │  (Ticket System)   │
         └────────────────────┘
```

### Ticket Hierarchy

```
Workflow Epic (OpenEvolve workflow)
│
├─── MDAP Task Ticket
│    │
│    ├─── MDAP Step 1 Ticket
│    ├─── MDAP Step 2 Ticket
│    └─── MDAP Step N Ticket
│
├─── MAKER Run Ticket
│    │
│    ├─── MAKER Step 1 Ticket
│    ├─── MAKER Step 2 Ticket
│    └─── MAKER Step N Ticket
│
└─── Sub-problem Tickets (standard workflow)
```

---

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- OpenEvolve system
- CrewAI instance running

### Dependencies

The integration requires these modules:
```python
# Core dependencies
- crewai_integration.py
- mdap_engine.py
- maker_engine.py
- mdap_maker_complete.py
- workflow_structures.py
```

### Setup

1. **Import the integration manager**:
```python
from crewai_integration import CrewAIIntegrationManager
from workflow_structures import WorkflowState
```

2. **Initialize the manager**:
```python
manager = CrewAIIntegrationManager(
    api_base="http://localhost:8000",
    api_key="your-api-key",
    project_id="your-project-id"
)
```

---

## MDAP Integration

### Creating an MDAP Task

```python
from mdap_engine import MDAPTask, MDAPStep, MDAPConfig

# Create MDAP steps
steps = [
    MDAPStep(
        step_id="analyze",
        prompt="Analyze the problem requirements",
        task_type="decomposition",
        priority=1
    ),
    MDAPStep(
        step_id="solve",
        prompt="Generate a solution",
        task_type="solve",
        priority=2
    )
]

# Create MDAP task
mdap_task = MDAPTask(
    task_id="task-001",
    description="Solve complex problem",
    steps=steps,
    max_retries=2,
    target_success_rate=0.95
)
```

### Syncing MDAP Task to CrewAI

```python
# Sync task to CrewAI
ticket_id = manager.sync_mdap_task(
    mdap_task=mdap_task,
    workflow_epic_id=workflow_epic_id
)

print(f"Created MDAP task ticket: {ticket_id}")
```

### Tracking MDAP Step Execution

```python
# After executing a step, sync results
vote_result = MDAPVoteResult(
    winner={"solution": "best solution"},
    votes={"{\"solution\": \"best solution\"}": 5,
           "{\"solution\": \"alternative\"}": 2},
    red_flags=0,
    confidence=0.71,
    attempts=7,
    duration_seconds=15.5
)

step_result = MDAPStepResult(
    step_id="analyze",
    vote_result=vote_result,
    status="success",
    retries=0
)

manager.sync_mdap_step_result(
    step_id="analyze",
    step_result=step_result,
    vote_result=vote_result
)
```

### Completing MDAP Task

```python
# After all steps complete
run_result = MDAPRunResult(
    task_id="task-001",
    step_results={
        "analyze": step_result_1,
        "solve": step_result_2
    },
    metrics={
        "steps_completed": 2,
        "steps_failed": 0,
        "votes_cast": 15,
        "red_flags": 0
    }
)

manager.sync_mdap_task_completion(
    task_id="task-001",
    run_result=run_result
)
```

---

## MAKER Integration

### Creating a MAKER Run

```python
from maker_engine import MakerConfig

# Configure MAKER
maker_config = MakerConfig(
    k_min=2,
    k_max=8,
    max_votes_per_step=50,
    max_steps=100,
    timeout_seconds=90
)

# Define initial state
initial_state = {
    "problem": "Solve Towers of Hanoi with 5 disks",
    "towers": {
        "A": [5, 4, 3, 2, 1],
        "B": [],
        "C": []
    }
}
```

### Syncing MAKER Run to CrewAI

```python
# Sync run to CrewAI
ticket_id = manager.sync_maker_run(
    run_id="maker-run-001",
    initial_state=initial_state,
    config=maker_config,
    workflow_epic_id=workflow_epic_id
)

print(f"Created MAKER run ticket: {ticket_id}")
```

### Tracking MAKER Step Execution

```python
from maker_engine import MakerState

# After each MAKER step
state = MakerState(
    step_index=1,
    current_state=current_state,
    history=[{"action": "move_disk"}],
    last_action={"from": "A", "to": "C", "disk": 1}
)

action = {"move": "A->C", "disk": 1}

manager.sync_maker_step(
    run_id="maker-run-001",
    step_index=1,
    state=state,
    action=action
)
```

### Completing MAKER Run

```python
# After MAKER completes
run_result = MakerRunResult(
    state=final_state,
    metrics={
        "steps": 31,
        "votes_cast": 155,
        "red_flags": 3,
        "escalations": 0,
        "errors": 0
    },
    terminated_reason="stop_condition_met"
)

manager.sync_maker_run_completion(
    run_id="maker-run-001",
    run_result=run_result
)
```

### Recursive MAKER Solve

```python
from mdap_maker_complete import MAKERRunMetrics

# After recursive solve completes
metrics = MAKERRunMetrics(
    total_steps=25,
    total_votes=100,
    red_flags=2,
    decompositions=5,
    atomic_solves=20,
    voting_rounds=25,
    total_time=45.2,
    avg_confidence=0.93
)

manager.sync_maker_recursive_solve(
    run_id="maker-recursive-001",
    solution=solution,
    metrics=metrics
)
```

---

## Combined Workflows

### Initializing Combined MDAP/MAKER Workflow

```python
# Create workflow state
workflow = WorkflowState(
    problem_statement="Complex problem requiring both MDAP and MAKER",
    workflow_id="workflow-001",
    start_time=time.time()
)

# Initialize combined workflow
ticket_ids = manager.initialize_mdap_maker_workflow(
    workflow_state=workflow,
    mdap_task=mdap_task,
    maker_run_id="maker-run-001",
    maker_config=maker_config,
    maker_initial_state=initial_state
)

print(f"Workflow Epic: {ticket_ids['workflow_epic']}")
print(f"MDAP Task: {ticket_ids['mdap_task']}")
print(f"MAKER Run: {ticket_ids['maker_run']}")
```

### Monitoring Combined Workflow

```python
# Get sync status
status = manager.get_mdap_maker_sync_status()

print(f"MDAP Available: {status['mdap_available']}")
print(f"MAKER Available: {status['maker_available']}")
print(f"MDAP Tasks Synced: {status['mdap_tasks_synced']}")
print(f"MAKER Runs Synced: {status['maker_runs_synced']}")
```

---

## API Reference

### CrewAIIntegrationManager

#### `__init__(api_base, api_key, project_id)`
Initialize the integration manager.

**Parameters:**
- `api_base` (str): Base URL for CrewAI API
- `api_key` (str): API key for authentication
- `project_id` (str): Project ID in CrewAI

#### MDAP Methods

##### `sync_mdap_task(mdap_task, workflow_epic_id=None)`
Sync an MDAP task to CrewAI.

**Returns:** Ticket ID (str) or None

##### `sync_mdap_step_result(step_id, step_result, vote_result)`
Sync MDAP step execution results.

**Returns:** bool indicating success

##### `sync_mdap_task_completion(task_id, run_result)`
Sync MDAP task completion.

**Returns:** bool indicating success

#### MAKER Methods

##### `sync_maker_run(run_id, initial_state, config, workflow_epic_id=None)`
Sync a MAKER run to CrewAI.

**Returns:** Ticket ID (str) or None

##### `sync_maker_step(run_id, step_index, state, action)`
Sync MAKER step execution.

**Returns:** bool indicating success

##### `sync_maker_run_completion(run_id, run_result)`
Sync MAKER run completion.

**Returns:** bool indicating success

##### `sync_maker_recursive_solve(run_id, solution, metrics)`
Sync MAKER recursive solve results.

**Returns:** bool indicating success

#### Combined Methods

##### `initialize_mdap_maker_workflow(...)`
Initialize a combined MDAP/MAKER workflow.

**Returns:** Dict with ticket IDs

##### `get_mdap_maker_sync_status()`
Get sync status for MDAP and MAKER.

**Returns:** Dict with status information

---

## Examples

### Example 1: Complete MDAP Workflow

```python
import time
from crewai_integration import CrewAIIntegrationManager
from mdap_engine import MDAPTask, MDAPStep, MDAPOrchestrator, MDAPConfig
from workflow_structures import Team, ModelConfig

# Setup
manager = CrewAIIntegrationManager(
    api_base="http://localhost:8000",
    api_key="your-key",
    project_id="your-project"
)

# Create team
team = Team(
    name="solver-team",
    members=[
        ModelConfig(
            model_id="gpt-4",
            api_key="your-openai-key",
            api_base="https://api.openai.com/v1"
        )
    ]
)

# Create MDAP task
steps = [
    MDAPStep(step_id="s1", prompt="Step 1", task_type="decomposition"),
    MDAPStep(step_id="s2", prompt="Step 2", task_type="solve")
]
task = MDAPTask(task_id="task-1", description="Test task", steps=steps)

# Sync to CrewAI
manager.sync_mdap_task(task)

# Execute MDAP
config = MDAPConfig()
orchestrator = MDAPOrchestrator(team, config)
result = orchestrator.execute_task(task)

# Sync results
for step_id, step_result in result.step_results.items():
    manager.sync_mdap_step_result(
        step_id,
        step_result,
        step_result.vote_result
    )

manager.sync_mdap_task_completion(task.task_id, result)
```

### Example 2: Complete MAKER Workflow

```python
from crewai_integration import CrewAIIntegrationManager
from maker_engine import MakerEngine, MakerConfig, MakerStep
from workflow_structures import Team, ModelConfig

# Setup
manager = CrewAIIntegrationManager(
    api_base="http://localhost:8000",
    api_key="your-key",
    project_id="your-project"
)

# Create team
team = Team(name="maker-team", members=[...])

# Create MAKER config
config = MakerConfig(max_steps=100)

# Initial state
state = {"counter": 0, "target": 10}

# Sync to CrewAI
manager.sync_maker_run("run-1", state, config)

# Execute MAKER
engine = MakerEngine(team, config)

def step_builder(current_state, history):
    return MakerStep(
        step_id=f"step-{current_state['counter']}",
        prompt_template="Increment counter: {state}",
        task_type="general"
    )

def apply_action(state, action):
    state["counter"] += 1
    return state

result = engine.solve(state, step_builder, apply_action)

# Sync completion
manager.sync_maker_run_completion("run-1", result)
```

### Example 3: Bidirectional Sync

```python
# Start sync loop (runs in background)
manager.start_sync_loop(workflow_state, interval=60)

# ... workflow executes ...

# Status updates from CrewAI are automatically synced to workflow_state

# Stop sync loop when done
manager.stop_sync_loop(workflow_state.workflow_id)
```

---

## Troubleshooting

### Common Issues

#### 1. MDAP/MAKER Not Available

**Problem:** `MDAP not available` or `MAKER not available` warnings

**Solution:**
```python
# Check availability
from crewai_integration import MDAP_AVAILABLE, MAKER_AVAILABLE

if not MDAP_AVAILABLE:
    print("MDAP libraries not installed")
    # Install: pip install -r requirements.txt
```

#### 2. Ticket Creation Fails

**Problem:** `Failed to create ticket` errors

**Solution:**
- Check CrewAI API is running
- Verify API credentials
- Check network connectivity
- Review CrewAI logs

```python
# Test connection
response = manager.client.session.get(f"{manager.client.api_base}/health")
print(response.status_code)  # Should be 200
```

#### 3. Sync Not Updating Tickets

**Problem:** Tickets not being updated

**Solution:**
```python
# Check ticket mappings
print(manager.mdap_sync.task_id_to_ticket_map)
print(manager.maker_sync.run_id_to_ticket_map)

# Verify sync availability
status = manager.get_mdap_maker_sync_status()
print(status)
```

#### 4. Missing Step Tickets

**Problem:** MDAP step tickets not created

**Solution:**
```python
# Manually create step tickets
mdap_task = ...  # your MDAP task
manager.mdap_sync._create_mdap_step_tickets(
    mdap_task,
    parent_task_id="task-ticket-id"
)
```

### Debug Logging

Enable detailed logging:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("crewai_integration")
logger.setLevel(logging.DEBUG)
```

### Testing the Integration

Run the test suite:

```bash
pytest test_mdap_maker_crewai_integration.py -v
```

---

## Best Practices

1. **Always check availability** before using MDAP/MAKER features
2. **Handle None returns** from sync methods gracefully
3. **Use workflow epics** to organize related tickets
4. **Monitor sync status** regularly
5. **Implement error handling** for API failures
6. **Use labels effectively** for ticket organization
7. **Set appropriate timeouts** for long-running tasks
8. **Clean up resources** by stopping sync loops

---

## Additional Resources

- [MDAP Documentation](./MDAP_DOCUMENTATION.md)
- [MAKER Documentation](./MAKER_DOCUMENTATION.md)
- [CrewAI API Reference](./HEPH_API_REFERENCE.md)
- [OpenEvolve Integration Guide](./OPENEVOLVE_INTEGRATION.md)

---

## Changelog

### Version 1.0.0 (2025-01-02)
- Initial release of MDAP/MAKER-CrewAI integration
- Support for MDAP task and step synchronization
- Support for MAKER run and step synchronization
- Combined workflow initialization
- Bidirectional sync support
- Comprehensive test suite
