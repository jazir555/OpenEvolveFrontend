# Checkpointing Guide for OpenEvolve Gauntlet System

This guide explains how to use the checkpointing system to create reliable,
resumable problem-solving pipelines.

## Table of Contents

1. [Checkpoint Lifecycle](#checkpoint-lifecycle)
2. [When Checkpoints Are Created](#when-checkpoints-are-created)
3. [How to Resume from Checkpoint](#how-to-resume-from-checkpoint)
4. [Best Practices](#best-practices)
5. [Common Issues](#common-issues)

---

## Checkpoint Lifecycle

### What is a Checkpoint?

A checkpoint is a snapshot of the entire pipeline state at a specific moment, including:
- The current problem being solved
- Execution context (team, stage, configuration)
- Solutions found so far
- Decomposition tree structure
- Execution status and metrics

### Lifecycle Stages

```
1. CREATE
   ├─ Problem decomposition completes
   ├─ Atomic problem solved
   ├─ Reassembly completes
   └─ Before final validation

2. STORE
   ├─ Serialize state to JSON/MessagePack
   ├─ Compress (optional)
   └─ Write to storage (file/database/Redis)

3. LOAD (when needed)
   ├─ Read from storage
   ├─ Decompress (if compressed)
   └─ Deserialize to PipelineState

4. RESUME
   ├─ Validate loaded state
   ├─ Restore execution context
   ├─ Continue from checkpoint
   └─ Cleanup old checkpoints

5. CLEANUP
   ├─ Automatic cleanup on success
   ├─ Manual cleanup as needed
   └─ Retention policy (keep last N)
```

---

## When Checkpoints Are Created

### Automatic Checkpoint Points

The system automatically creates checkpoints at these stages:

1. **Before Decomposition** (`stage: 'before_decomposition'`)
   - Captures initial problem state
   - Enables restart from beginning

2. **After Decomposition** (`stage: 'decomposition_complete'`)
   - Captures decomposed problem tree
   - Enables skipping decomposition on retry

3. **After Atomic Solution** (`stage: 'atomic_solve_complete'`)
   - Captures each atomic solution
   - Enables resuming at next atomic problem

4. **After Reassembly** (`stage: 'reassembly_complete'`)
   - Captures reassembled solution
   - Enables resume before validation

5. **Before Final Gauntlet** (`stage: 'before_validation'`)
   - Captures state before final validation
   - Enables re-running validation

### Manual Checkpoint Creation

You can also manually create checkpoints:

```python
from bubblelabs_nodes import create_checkpoint_manager

manager = create_checkpoint_manager()

# Manual checkpoint
await manager.create_checkpoint(
    problem=problem,
    context={'stage': 'custom_checkpoint'},
    solutions=partial_solutions,
    level=0,
    stage='manual'
)
```

---

## How to Resume from Checkpoint

### Automatic Resume

The system can automatically resume from the latest checkpoint:

```python
from bubblelabs_nodes import CheckpointedPipeline

pipeline = CheckpointedPipeline()

result = await pipeline.execute_with_checkpointing(
    problem=problem,
    solve_func=solve_function,
    resume_from_checkpoint=None  # Auto-find latest
)

# If crash occurred, automatically resumes from last checkpoint
```

### Manual Resume

Specify exact checkpoint to resume from:

```python
checkpoint_id = "problem_123_0_partial_solution_20250123_120000"

result = await pipeline.execute_with_checkpointing(
    problem=problem,
    solve_func=solve_function,
    resume_from_checkpoint=checkpoint_id
)
```

### Resume from Specific Problem

List checkpoints for a problem and choose one:

```python
# List available checkpoints
checkpoints = await manager.list_checkpoints('problem_123')

# Show checkpoints
for cp in checkpoints:
    print(f"{cp['checkpoint_id']}: {cp['stage']} @ {cp['timestamp']}")

# Resume from specific checkpoint
state = await manager.load_checkpoint(checkpoint_id)
if state:
    # Continue execution
    result = await continue_execution(state)
```

---

## Best Practices

### 1. Checkpoint Frequency

**DO:**
- Use 'major' frequency for long pipelines (default)
- Use 'minor' for critical operations
- Use 'all' only for debugging

**DON'T:**
- Checkpoint too frequently (performance overhead)
- Disable checkpointing for long operations

### 2. Checkpoint Size Management

**DO:**
- Enable compression for large states
- Set reasonable retention policy (keep last 5)
- Monitor checkpoint directory size

**DON'T:**
- Keep unlimited checkpoints
- Store large objects in context

### 3. Crash Recovery

**DO:**
- Design operations to be idempotent
- Validate restored state before continuing
- Test crash recovery regularly

**DON'T:**
- Assume checkpoint will always exist
- Skip validation of restored state

### 4. Production Deployment

**DO:**
- Use file-based storage for persistence
- Enable compression for production
- Monitor checkpoint creation success rate
- Set up automated cleanup

**DON'T:**
- Use in-memory storage in production
- Disable checkpointing for performance

---

## Common Issues

### Issue 1: Checkpoint Too Large

**Symptoms:**
- Slow checkpoint creation
- High disk usage
- Failed saves

**Solutions:**
1. Enable compression: `create_checkpoint_manager(compression=True)`
2. Reduce checkpoint frequency: `checkpoint_frequency='major'`
3. Clean context: Remove large objects before checkpointing
4. Implement selective checkpointing: Only essential data

### Issue 2: Corrupted Checkpoint

**Symptoms:**
- Resume fails
- Deserialization errors
- Missing data

**Solutions:**
1. **Verify backup**: Check if backup checkpoint exists
2. **Manual inspection**: Load checkpoint JSON and validate structure
3. **Fallback to previous**: Use earlier checkpoint
4. **Fix state**: Manually edit checkpoint JSON if needed

**Example: Manual checkpoint editing**

```python
import json

# Load checkpoint
with open('checkpoints/problem_123_0_xxx.json', 'r') as f:
    checkpoint_data = json.load(f)

# Fix corrupted data
checkpoint_data['state']['problem']['id'] = 'correct_id'

# Save fixed checkpoint
with open('checkpoints/problem_123_0_xxx.json', 'w') as f:
    json.dump(checkpoint_data, f, indent=2)
```

### Issue 3: Cannot Resume

**Symptoms:**
- Resume starts from beginning
- Checkpoint not found
- State doesn't match

**Solutions:**
1. **Check path**: Verify checkpoint file exists
2. **Verify ID**: Ensure checkpoint_id is correct
3. **Check permissions**: Ensure read access to checkpoint directory
4. **Validate storage**: Ensure storage backend is accessible

### Issue 4: Stale Checkpoints

**Symptoms:**
- Old checkpoints slowing down system
- Disk space full
- Wrong checkpoint being loaded

**Solutions:**
```python
# Cleanup old checkpoints
deleted = await manager.cleanup_checkpoints(
    problem_id='problem_123',
    keep_last_n=5
)

# List to verify
checkpoints = await manager.list_checkpoints('problem_123')
print(f"Remaining checkpoints: {len(checkpoints)}")
```

---

## Recovery Procedures

### Full Pipeline Recovery

If the entire pipeline crashes and needs recovery:

1. **Identify latest checkpoint**
```python
checkpoints = await manager.list_checkpoints(problem_id)
latest = max(checkpoints, key=lambda cp: cp['timestamp'])
checkpoint_id = latest['checkpoint_id']
```

2. **Load and validate checkpoint**
```python
state = await manager.load_checkpoint(checkpoint_id)

if not state:
    logger.error("Cannot recover - no valid checkpoint found")
    # Restart from beginning
    return await start_fresh(problem)
```

3. **Validate state integrity**
```python
def validate_state(state: PipelineState) -> bool:
    if not state.problem:
        return False
    if not state.context:
        return False
    # Add more validations as needed
    return True

if not validate_state(state):
    logger.warning("Checkpoint state invalid, trying previous checkpoint")
    # Try previous checkpoint
```

4. **Resume execution**
```python
result = await continue_from_checkpoint(state)
```

### Partial Recovery

If only a subproblem crashed:

1. **Find subproblem checkpoint**
```python
subproblem_id = f"{problem_id}_{subproblem_index}"
checkpoint = await manager.load_checkpoint(subproblem_id)
```

2. **Resume subproblem**
```python
result = await solve_subproblem_from_checkpoint(checkpoint)
```

3. **Integrate result**
```python
# Add result to parent solutions
parent_solutions[subproblem_id] = result
```

---

## Monitoring Checkpoints

### Checkpoint Statistics

```python
# List all checkpoints for a problem
checkpoints = await manager.list_checkpoints(problem_id)

# Calculate statistics
total_size = sum(cp['state_size'] for cp in checkpoints)
avg_age = sum(
    (datetime.now() - cp['timestamp']).total_seconds()
    for cp in checkpoints
) / len(checkpoints)

print(f"Total checkpoints: {len(checkpoints)}")
print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
print(f"Average age: {avg_age / 3600:.1f} hours")
```

### Checkpoint Health Monitoring

```python
# Check checkpoint health
def check_checkpoint_health(checkpoint_id: str):
    try:
        state = await manager.load_checkpoint(checkpoint_id)
        if state and validate_state(state):
            return "healthy"
        else:
            return "invalid"
    except Exception as e:
        return f"error: {str(e)}"

# Check all checkpoints
for cp in checkpoints:
    health = check_checkpoint_health(cp['checkpoint_id'])
    print(f"{cp['checkpoint_id']}: {health}")
```

---

## Advanced Usage

### Selective Checkpointing

Only checkpoint critical data:

```python
await manager.create_checkpoint(
    problem=problem,
    context={
        'stage': 'optimized',
        'critical_only': True,  # Flag for selective checkpointing
    },
    solutions=essential_solutions,  # Only save important solutions
    level=0,
    stage='selective'
)
```

### Incremental Checkpointing

Checkpoint only changes since last checkpoint:

```python
# Calculate delta
changes = calculate_delta(last_state, current_state)

# Save only changes
await manager.create_checkpoint(
    problem=problem,
    context={
        'stage': 'incremental',
        'is_delta': True,
        'base_checkpoint': last_checkpoint_id
    },
    solutions=changes,
    level=0,
    stage='incremental'
)
```

### Cross-Server Checkpointing

Share checkpoints across servers:

```python
# Save to shared storage
from bubblelabs_nodes import CheckpointRepository

repo = CheckpointRepository(
    storage_type='redis',
    storage_url='redis://shared-redis:6379'
)

await manager.create_checkpoint(
    problem=problem,
    context=context,
    solutions=solutions,
    level=0,
    stage='shared'
)

# Load from any server
state = await repo.load(checkpoint_id)
```

---

## Troubleshooting Guide

### Problem: Checkpoints Not Created

**Checklist:**
- [ ] Is checkpointing enabled? `checkpointing_enabled=True`
- [ ] Is checkpoint manager initialized?
- [ ] Are permissions correct on checkpoint directory?
- [ ] Is disk space available?

**Debug:**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Try manual checkpoint creation
try:
    cp_id = await manager.create_checkpoint(...)
    print(f"Created: {cp_id}")
except Exception as e:
    print(f"Failed: {e}")
```

### Problem: Resume Not Working

**Checklist:**
- [ ] Is checkpoint_id correct?
- [ ] Does checkpoint file exist?
- [ ] Is state serializable?
- [ ] Are all required fields present?

**Debug:**
```python
# Load and inspect checkpoint
state = await manager.load_checkpoint(checkpoint_id)

print("Problem:", state.problem)
print("Context:", state.context)
print("Solutions:", state.solutions)
print("Status:", state.execution_status)

# Check for missing data
assert state.problem, "Missing problem!"
assert state.context, "Missing context!"
```

---

## Summary

- **Checkpoints** provide reliability for long-running pipelines
- **Automatic creation** at key stages is default
- **Manual creation** is available for custom checkpoints
- **Resume** is automatic or manual
- **Cleanup** happens automatically on success
- **Compression** reduces size by 60-80%

For more information, see:
- `bubblelabs_nodes/checkpoint_manager.py` - Implementation
- `bubblelabs_nodes/gauntlet_pipeline_checkpointed.py` - Integration
- `PHASE1_COMPLETE.md` - Phase 1 documentation
