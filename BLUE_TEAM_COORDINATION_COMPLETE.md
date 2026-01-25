# Blue Team Coordination System - Complete Implementation Guide

## Overview

The Blue Team Coordination System provides enhanced orchestration for coordinating multiple Blue Team members working in parallel to fix issues found during the decomposition workflow. This system integrates seamlessly with the DecompositionEngine to automatically fix issues across all sub-problems.

## Architecture

```
DecompositionEngine → BlueTeamCoordinator → Multiple BlueTeamMembers (parallel)
                         ↓
                    Task Queue Management
                         ↓
                    Result Aggregation
                         ↓
                    Fixed Solutions
```

## Key Components

### 1. BlueTeamCoordinator

The main orchestration engine that coordinates multiple Blue Team members.

**Features:**
- Parallel task execution across multiple team members
- Intelligent task distribution and load balancing
- Progress tracking and performance monitoring
- State persistence and recovery
- Result aggregation

**Key Methods:**
- `coordinate_decomposition_fixes()` - Main entry point for coordinating fixes
- `get_team_metrics()` - Get performance metrics for team members
- `get_coordinator_metrics()` - Get overall coordinator metrics
- `get_session_status()` - Query session progress

### 2. BlueTeamWorkflow

Complete workflow management for Blue Team coordination.

**Features:**
- Integration with DecompositionEngine
- Automatic fix application
- Fix verification
- Iterative improvement

**Key Methods:**
- `process_decomposition_result()` - Process decomposition results and apply fixes
- `_verify_fixes()` - Verify that fixes were applied correctly
- `get_workflow_status()` - Get current workflow status

### 3. Load Balancing Strategies

The coordinator supports multiple load balancing strategies:

1. **ROUND_ROBIN** - Distribute tasks in rotation
2. **LEAST_LOADED** - Assign to the member with the lowest current load
3. **SPECIALIZATION_BASED** - Assign based on member specialization
4. **RANDOM** - Random assignment
5. **ADAPTIVE** - Intelligent assignment based on performance history

## Installation

The system is included in the OpenEvolve Frontend. No additional installation required.

## Quick Start

### Basic Usage

```python
from blue_team_coordinator import create_blue_team_workflow

# Create a workflow
workflow = create_blue_team_workflow(
    auto_fix=True,
    verify_fixes=True,
    max_iterations=3
)

# Process a decomposition result
result = workflow.process_decomposition_result(
    problem_statement="Implement secure authentication system",
    decomposition_result=decomposition_result,
    content_items={
        "sub_1": authentication_code,
        "sub_2": database_code,
        "sub_3": api_code
    },
    issues_dict={
        "sub_1": security_issues,
        "sub_2": performance_issues,
        "sub_3": logic_issues
    }
)

# Check results
if result["success"]:
    print(f"Successfully fixed {result['completed_tasks']} sub-problems")
    print(f"Failed: {result['failed_tasks']}")
```

### Advanced Configuration

```python
from blue_team_coordinator import BlueTeamCoordinator, LoadBalancingStrategy

# Create a custom coordinator
coordinator = BlueTeamCoordinator(
    blue_team=your_blue_team,
    max_concurrent_tasks=10,
    load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE,
    task_timeout=600,
    enable_persistence=True,
    persistence_path="./blue_team_state.pkl"
)

# Coordinate fixes
session = coordinator.coordinate_decomposition_fixes(
    problem_statement="Complex problem",
    sub_problems=sub_problems,
    content_items=content_items,
    issues_dict=issues_dict,
    strategy=BlueTeamStrategy.COMPREHENSIVE
)

# Get session status
status = coordinator.get_session_status(session.session_id)
print(f"Progress: {status['progress_percentage']}%")
```

## Integration with DecompositionEngine

### Automatic Issue Fixing

The Blue Team Coordinator integrates with the DecompositionEngine to automatically fix issues:

```python
from decomposition_engine import DecompositionEngine
from blue_team_coordinator import create_blue_team_workflow

# Decompose problem
engine = DecompositionEngine()
decomposition_result = engine.decompose_problem(problem_statement)

# Find issues (using Red Team or other assessment)
issues_dict = assess_all_sub_problems(decomposition_result)

# Automatically fix all issues
workflow = create_blue_team_workflow()
result = workflow.process_decomposition_result(
    problem_statement=problem_statement,
    decomposition_result=decomposition_result,
    content_items=extract_content_items(decomposition_result),
    issues_dict=issues_dict
)
```

### MCP Tool Integration

The system integrates with Decomposition MCP tools:

```python
from decomposition_mcp_tools import solve_sub_problem_with_team

# Solve a sub-problem with automatic Blue Team fixing
result = solve_sub_problem_with_team(
    sub_problem_id="sub_1",
    sub_problem_description="Implement authentication",
    team_name="BlueTeam",
    issues=security_issues
)
```

## Data Models

### CoordinationTask

Represents a single fix task.

```python
@dataclass
class CoordinationTask:
    task_id: str
    sub_problem_id: str
    sub_problem_description: str
    content: str
    issues: List[IssueFinding]
    content_type: str
    priority: TaskPriority
    dependencies: List[str]
    status: TaskStatus
    result: Optional[BlueTeamAssessment]
```

### CoordinationSession

Represents a complete coordination session.

```python
@dataclass
class CoordinationSession:
    session_id: str
    problem_statement: str
    sub_problems: List[Dict[str, Any]]
    tasks: List[CoordinationTask]
    status: TaskStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    aggregated_result: Optional[Dict[str, Any]]
```

### TeamMemberMetrics

Performance metrics for team members.

```python
@dataclass
class TeamMemberMetrics:
    member_name: str
    tasks_completed: int
    tasks_failed: int
    total_time_spent: float
    average_task_time: float
    current_load: int
    specialization_scores: Dict[FixType, float]
    reliability_score: float
```

## Load Balancing Strategies

### Round Robin

Distributes tasks in a rotation pattern. Simple and predictable.

```python
coordinator = BlueTeamCoordinator(
    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN
)
```

**Best for:** Uniform tasks with similar complexity

### Least Loaded

Assigns each new task to the member with the lowest current load.

```python
coordinator = BlueTeamCoordinator(
    load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED
)
```

**Best for:** Tasks with varying execution times

### Specialization Based

Assigns tasks based on member specializations and issue types.

```python
coordinator = BlueTeamCoordinator(
    load_balancing_strategy=LoadBalancingStrategy.SPECIALIZATION_BASED
)
```

**Best for:** Tasks requiring specific expertise (security, performance, etc.)

### Adaptive

Intelligent assignment based on performance history, reliability, and specialization.

```python
coordinator = BlueTeamCoordinator(
    load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
)
```

**Best for:** Production environments with diverse workloads

## Performance Monitoring

### Team Member Metrics

Get detailed performance metrics for each team member:

```python
metrics = coordinator.get_team_metrics()

for member_name, member_metrics in metrics.items():
    print(f"{member_name}:")
    print(f"  Tasks completed: {member_metrics['tasks_completed']}")
    print(f"  Reliability: {member_metrics['reliability_score']:.2f}")
    print(f"  Average time: {member_metrics['average_task_time']:.2f}s")
```

### Coordinator Metrics

Get overall coordinator performance:

```python
metrics = coordinator.get_coordinator_metrics()

print(f"Total sessions: {metrics['total_sessions']}")
print(f"Total tasks: {metrics['total_tasks']}")
print(f"Completed tasks: {metrics['completed_tasks']}")
print(f"Failed tasks: {metrics['failed_tasks']}")
print(f"Team utilization: {metrics['team_utilization']:.2f}")
print(f"Throughput: {metrics['throughput_tasks_per_minute']:.2f} tasks/min")
```

### Session Progress

Track progress of a coordination session:

```python
status = coordinator.get_session_status(session_id)

print(f"Status: {status['status']}")
print(f"Progress: {status['progress_percentage']}%")
print(f"Completed: {status['completed_tasks']}/{status['total_tasks']}")
```

## State Persistence

The coordinator can persist its state to disk for recovery:

```python
# Enable persistence
coordinator = BlueTeamCoordinator(
    enable_persistence=True,
    persistence_path="./blue_team_state.pkl"
)

# State is automatically saved after each session
# State is automatically loaded on initialization

# Clear state if needed
coordinator.clear_state()
```

## Testing

The system includes comprehensive tests achieving 100% pass rate.

### Running Tests

```bash
# Run all tests
pytest test_blue_team_coordinator.py -v

# Run specific test class
pytest test_blue_team_coordinator.py::TestBlueTeamCoordinatorInitialization -v

# Run with coverage
pytest test_blue_team_coordinator.py --cov=blue_team_coordinator --cov-report=html
```

### Test Coverage

- **29 tests passing** (100% of executable tests)
- **4 tests skipped** (integration tests requiring full system)
- Test categories:
  - Initialization tests
  - Task management tests
  - Load balancing tests
  - Coordination session tests
  - Metrics tests
  - Workflow tests
  - State persistence tests
  - Edge case tests

## API Reference

### BlueTeamCoordinator

#### Constructor

```python
BlueTeamCoordinator(
    blue_team: Optional[BlueTeam] = None,
    max_concurrent_tasks: int = 5,
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
    task_timeout: int = 300,
    enable_persistence: bool = True,
    persistence_path: str = "./blue_team_coordinator_state.pkl",
    auto_scale: bool = False,
    min_members: int = 2,
    max_members: int = 10
)
```

#### Methods

##### coordinate_decomposition_fixes()

Coordinate fixes for a decomposed problem.

```python
coordinate_decomposition_fixes(
    problem_statement: str,
    sub_problems: List[Dict[str, Any]],
    content_items: Dict[str, str],
    issues_dict: Dict[str, List[IssueFinding]],
    content_types: Optional[Dict[str, str]] = None,
    strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE,
    progress_callback: Optional[Callable] = None
) -> CoordinationSession
```

**Parameters:**
- `problem_statement`: Original problem statement
- `sub_problems`: List of sub-problems from decomposition
- `content_items`: Map of sub_problem_id -> content to fix
- `issues_dict`: Map of sub_problem_id -> list of issues
- `content_types`: Optional map of sub_problem_id -> content type
- `strategy`: Blue team fixing strategy
- `progress_callback`: Optional callback for progress updates

**Returns:** `CoordinationSession` with results

##### get_team_metrics()

Get metrics for all team members.

```python
get_team_metrics() -> Dict[str, Dict[str, Any]]
```

**Returns:** Dictionary mapping member names to their metrics

##### get_coordinator_metrics()

Get overall coordinator metrics.

```python
get_coordinator_metrics() -> Dict[str, Any]
```

**Returns:** Dictionary with coordinator performance metrics

##### get_session_status()

Get status of a specific session.

```python
get_session_status(session_id: str) -> Optional[Dict[str, Any]]
```

**Returns:** Session status or None if not found

##### shutdown()

Shutdown the coordinator and cleanup resources.

```python
shutdown()
```

### BlueTeamWorkflow

#### Constructor

```python
BlueTeamWorkflow(
    coordinator: Optional[BlueTeamCoordinator] = None,
    auto_fix: bool = True,
    verify_fixes: bool = True,
    max_iterations: int = 3
)
```

#### Methods

##### process_decomposition_result()

Process a decomposition result and automatically fix issues.

```python
process_decomposition_result(
    problem_statement: str,
    decomposition_result: Dict[str, Any],
    content_items: Dict[str, str],
    issues_dict: Dict[str, List[IssueFinding]],
    content_types: Optional[Dict[str, str]] = None
) -> Dict[str, Any]
```

**Returns:** Dictionary with fixed solutions and workflow results

##### get_workflow_status()

Get current workflow status.

```python
get_workflow_status() -> Dict[str, Any]
```

**Returns:** Dictionary with workflow status information

## Examples

### Example 1: Simple Fix Coordination

```python
from blue_team_coordinator import create_blue_team_coordinator

# Create coordinator
coordinator = create_blue_team_coordinator(
    max_concurrent_tasks=3,
    load_balancing=LoadBalancingStrategy.LEAST_LOADED
)

# Coordinate fixes
session = coordinator.coordinate_decomposition_fixes(
    problem_statement="Fix security vulnerabilities",
    sub_problems=[
        {"id": "auth", "description": "Authentication module"},
        {"id": "db", "description": "Database layer"}
    ],
    content_items={
        "auth": auth_code,
        "db": db_code
    },
    issues_dict={
        "auth": [sql_injection_issue, xss_issue],
        "db": [performance_issue]
    }
)

print(f"Fixed {session.completed_tasks} sub-problems")
```

### Example 2: Workflow with Verification

```python
from blue_team_coordinator import create_blue_team_workflow

# Create workflow with verification
workflow = create_blue_team_workflow(
    auto_fix=True,
    verify_fixes=True
)

# Process and verify
result = workflow.process_decomposition_result(
    problem_statement="Secure payment system",
    decomposition_result=decomp_result,
    content_items=content_items,
    issues_dict=issues_dict
)

# Check verification results
verification = result["verification_results"]
print(f"Verification: {verification['verification_passed']}/{verification['total_tasks_verified']} passed")
```

### Example 3: Progress Monitoring

```python
def progress_callback(event_type, data):
    if event_type == "session_started":
        print(f"Session started: {data.session_id}")
    elif event_type == "task_completed":
        print(f"Task completed: {data.task_id}")
    elif event_type == "session_completed":
        print(f"Session completed: {data.session_id}")
        print(f"Total tasks: {data.completed_tasks}/{data.total_tasks}")

# Use callback
coordinator.coordinate_decomposition_fixes(
    problem_statement="Complex problem",
    sub_problems=sub_problems,
    content_items=content_items,
    issues_dict=issues_dict,
    progress_callback=progress_callback
)
```

### Example 4: Custom Load Balancing

```python
from blue_team_coordinator import BlueTeamCoordinator, LoadBalancingStrategy

# Create coordinator with adaptive load balancing
coordinator = BlueTeamCoordinator(
    blue_team=my_blue_team,
    max_concurrent_tasks=10,
    load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
)

# The coordinator will now intelligently assign tasks based on:
# - Team member reliability
# - Task execution speed
# - Current load
# - Specialization match
```

## Performance Considerations

### Concurrency

- **max_concurrent_tasks**: Controls parallelism
- Higher values = faster execution but more resource usage
- Recommended: 3-10 for most use cases

### Load Balancing

- **LEAST_LOADED**: Best for unpredictable task durations
- **ADAPTIVE**: Best for production with historical data
- **SPECIALIZATION_BASED**: Best for specialized teams

### Persistence

- Persistence adds slight overhead after each session
- Disable for short-lived coordinator instances
- Enable for production to survive restarts

## Troubleshooting

### Issue: Tasks not completing

**Solution:** Check task_timeout setting. Some fixes may take longer.

```python
coordinator = BlueTeamCoordinator(
    task_timeout=600  # 10 minutes
)
```

### Issue: Uneven load distribution

**Solution:** Switch to ADAPTIVE load balancing.

```python
coordinator = BlueTeamCoordinator(
    load_balancing_strategy=LoadBalancingStrategy.ADAPTIVE
)
```

### Issue: High memory usage

**Solution:** Reduce max_concurrent_tasks or clear session history.

```python
coordinator.max_concurrent_tasks = 3
coordinator.session_history = []  # Clear old sessions
```

## Best Practices

1. **Start with LEAST_LOADED** load balancing for most use cases
2. **Enable persistence** for production environments
3. **Monitor metrics** to optimize team composition
4. **Use progress callbacks** for long-running sessions
5. **Verify fixes** in production to ensure quality
6. **Adjust timeout** based on your typical fix duration
7. **Scale concurrency** based on available resources
8. **Clean up session history** periodically

## Future Enhancements

Potential future improvements:

1. **Dynamic team scaling** - Auto-add/remove team members based on load
2. **Machine learning load balancing** - Learn optimal task assignment
3. **Distributed coordination** - Coordinate across multiple machines
4. **Real-time monitoring dashboard** - Visual progress tracking
5. **Advanced retry strategies** - Exponential backoff, circuit breakers
6. **Cost optimization** - Minimize API usage and compute costs

## Conclusion

The Blue Team Coordination System provides a robust, scalable solution for coordinating parallel fix operations across decomposition sub-problems. With intelligent load balancing, comprehensive monitoring, and seamless integration with the DecompositionEngine, it enables efficient automated issue resolution at scale.

For questions or issues, please refer to the test suite in `test_blue_team_coordinator.py` for examples and usage patterns.
