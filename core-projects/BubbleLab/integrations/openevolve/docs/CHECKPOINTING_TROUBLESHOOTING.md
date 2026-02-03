# Troubleshooting Guide for OpenEvolve Gauntlet Checkpointing

This guide helps you diagnose and resolve common issues with the checkpointing system.

## Table of Contents

1. [Common Issues](#common-issues)
2. [Corrupted Checkpoints](#corrupted-checkpoints)
3. [Manual Checkpoint Editing](#manual-checkpoint-editing)
4. [Recovery Procedures](#recovery-procedures)

---

## Common Issues

### Issue 1: Checkpoint Creation Fails

**Symptoms:**
```
Error: Failed to save checkpoint: Permission denied
Error: Failed to create checkpoint directory
```

**Diagnosis:**
```bash
# Check directory permissions
ls -la ./gauntlet_checkpoints/

# Check disk space
df -h

# Check write permissions
touch ./gauntlet_checkpoints/test.tmp
rm ./gauntlet_checkpointstest.tmp
```

**Solutions:**

1. **Fix permissions:**
```bash
chmod 755 ./gauntlet_checkpoints/
```

2. **Create directory:**
```bash
mkdir -p ./gauntlet_checkpoints/
```

3. **Free disk space:**
```bash
# Remove old files
rm -rf ./gauntlet_checkpoints/*.checkpoint

# Or increase quota
```

---

### Issue 2: Checkpoint Load Fails

**Symptoms:**
```
Error: Cannot load checkpoint: Invalid JSON
Error: Checkpoint not found: problem_123_xxx
Error: Deserialization failed
```

**Diagnosis:**
```python
# Check if checkpoint exists
from bubblelabs_nodes import create_checkpoint_manager
manager = create_checkpoint_manager()

checkpoints = await manager.list_checkpoints('problem_123')
print(f"Found {len(checkpoints)} checkpoints")

# Try loading specific checkpoint
if checkpoints:
    state = await manager.load_checkpoint(checkpoints[0]['checkpoint_id'])
    print("Loaded successfully")
```

**Solutions:**

1. **List available checkpoints:**
```python
checkpoints = await manager.list_checkpoints()
for cp in checkpoints:
    print(f"{cp['checkpoint_id']}: {cp['stage']} @ {cp['timestamp']}")
```

2. **Use previous checkpoint:**
```python
# Try second-latest checkpoint
if len(checkpoints) >= 2:
    state = await manager.load_checkpoint(checkpoints[1]['checkpoint_id'])
```

3. **Start fresh if no checkpoints:**
```python
if not checkpoints:
    logger.warning("No checkpoints found, starting fresh")
    result = await solve_problem_from_start(problem)
```

---

## Corrupted Checkpoints

### Detecting Corruption

**Symptoms:**
- JSON parse errors
- Missing required fields
- Invalid data types

**Detection Code:**
```python
import json

def validate_checkpoint_file(filepath: str) -> bool:
    """Validate checkpoint file structure"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check required fields
        required = ['problem_id', 'timestamp', 'state', 'metadata']
        for field in required:
            if field not in data:
                print(f"Missing required field: {field}")
                return False

        # Validate state
        state = data.get('state', {})
        if not state.get('problem'):
            print("Missing problem in state")
            return False

        return True

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"Validation error: {e}")
        return False

# Check all checkpoints
import glob
for cp_file in glob.glob('./gauntlet_checkpoints/*.meta'):
    valid = validate_checkpoint_file(cp_file)
    print(f"{cp_file}: {'✅' if valid else '❌'}")
```

---

### Manual Checkpoint Editing

When you need to fix a corrupted checkpoint:

**Step 1: Load checkpoint**
```python
import json

with open('./gauntlet_checkpoints/problem_123_0_xxx.meta', 'r') as f:
    metadata = json.load(f)

print(f"Checkpoint ID: {metadata['checkpoint_id']}")
print(f"Problem ID: {metadata['problem_id']}")
print(f"Stage: {metadata['stage']}")
```

**Step 2: Load checkpoint data**
```python
# Checkpoint data
with open('./gauntlet_checkpoints/problem_123_0_xxx.checkpoint', 'rb') as f:
    data = f.read()

# Check if compressed
try:
    import gzip
    data = gzip.decompress(data)
    print("Data is compressed")
except:
    print("Data is not compressed")

# Parse JSON
state = json.loads(data.decode('utf-8'))
```

**Step 3: Fix the issue**

```python
# Example: Fix missing problem ID
if not state.get('problem'):
    state['problem'] = {'id': 'problem_123'}

# Example: Fix corrupted solution
if not state.get('solutions'):
    state['solutions'] = {}

# Example: Fix execution status
if not state.get('execution_status'):
    state['execution_status'] = {'main': 'pending'}
```

**Step 4: Save fixed checkpoint**
```python
# Save metadata
with open('./gauntlet_checkpoints/problem_123_0_xxx.meta', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save checkpoint data
with open('./gauntlet_checkpoints/problem_123_0_xxx.checkpoint', 'wb') as f:
    f.write(data)
```

---

## Recovery Procedures

### Procedure 1: Recover from All Checkpoints Corrupted

**When:** All checkpoints for a problem are corrupted

**Steps:**

1. **Identify problem**
```python
problem_id = 'problem_123'
checkpoints = await manager.list_checkpoints(problem_id)

all_corrupted = True
for cp in checkpoints:
    try:
        state = await manager.load_checkpoint(cp['checkpoint_id'])
        if state and validate_state(state):
            all_corrupted = False
    except:
        continue

print(f"All checkpoints corrupted: {all_corrupted}")
```

2. **Restart from beginning**
```python
if all_corrupted:
    logger.warning(f"All checkpoints corrupted for {problem_id}, restarting")
    result = await solve_problem_from_start(problem)
```

3. **Create new checkpoint**
```python
# Create initial checkpoint
await manager.create_checkpoint(
    problem=problem,
    context={'stage': 'restart'},
    level=0,
    stage='initial'
)
```

---

### Procedure 2: Recover from Partial Checkpoint

**When:** Checkpoint is partial (missing some data)

**Steps:**

1. **Load partial checkpoint**
```python
state = await manager.load_checkpoint(checkpoint_id)

print(f"Has problem: {bool(state.problem)}")
print(f"Has context: {bool(state.context)}")
print(f"Has solutions: {bool(state.solutions)}")
```

2. **Fill missing data**
```python
# Fill missing problem
if not state.problem:
    logger.info("Recovering problem from original definition")
    state.problem = load_original_problem(problem_id)

# Fill missing context
if not state.context:
    logger.info("Recovering context from defaults")
    state.context = get_default_context()

# Fill missing solutions
if not state.solutions:
    logger.info("Recovering solutions from previous stage")
    state.solutions = recover_partial_solutions(state)
```

3. **Validate and continue**
```python
if validate_state(state):
    logger.info("Recovered checkpoint state, continuing")
    result = await continue_from_checkpoint(state)
else:
    logger.warning("Could not recover checkpoint state, restarting")
```

---

### Procedure 3: Recovery with State Migration

**When:** Checkpoint format has changed

**Steps:**

1. **Detect old format**
```python
def detect_checkpoint_version(filepath: str) -> str:
    """Detect checkpoint version"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Check for version 1 format
    if 'format_version' in data:
        return data['format_version']

    # Assume version 0 (original)
    return 'v0'
```

2. **Migrate to new format**
```python
def migrate_checkpoint_v0_to_v1(data: dict) -> dict:
    """Migrate checkpoint from v0 to v1"""
    # Add version field
    data['format_version'] = 'v1'

    # Restructure old fields
    if 'problem' in data and not isinstance(data['problem'], dict):
        data['problem'] = {'id': data['problem']}

    return data

# Load and migrate
with open(checkpoint_file, 'r') as f:
    data = json.load(f)

version = detect_checkpoint_version(checkpoint_file)
print(f"Checkpoint version: {version}")

if version == 'v0':
    print("Migrating to v1...")
    data = migrate_checkpoint_v0_to_v1(data)

    # Save migrated
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f, indent=2)
```

3. **Load migrated checkpoint**
```python
state = await manager.load_checkpoint(checkpoint_id)
```

---

## Emergency Recovery

### Emergency Reset

**WARNING:** This deletes all checkpoints!

```bash
# Delete all checkpoints
rm -rf ./gauntlet_checkpoints/*

# Or programmatically
await manager.clear()
```

### Emergency Backup Before Critical Operation

```python
# Backup existing checkpoints before risky operation
import shutil
from datetime import datetime

backup_dir = f"./gauntlet_checkpoints_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
shutil.copytree('./gauntlet_checkpoints', backup_dir)

print(f"Backed up to: {backup_dir}")

# Perform risky operation
try:
    result = await risky_operation()
except Exception as e:
    # Restore from backup
    shutil.rmtree('./gauntlet_checkpoints')
    shutil.copytree(backup_dir, './gauntlet_checkpoints')
    print("Restored from backup")
    raise
```

---

## Prevention Best Practices

### 1. Regular Validation

```python
# Validate checkpoint creation
async def create_safe_checkpoint(manager, problem, context, solutions):
    """Create checkpoint with validation"""
    try:
        cp_id = await manager.create_checkpoint(
            problem=problem,
            context=context,
            solutions=solutions,
            level=0,
            stage='validation'
        )

        # Validate by loading it back
        state = await manager.load_checkpoint(cp_id)
        if not state or not state.problem:
            raise ValueError("Checkpoint validation failed")

        logger.info(f"Validated checkpoint: {cp_id}")
        return cp_id

    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        # Rollback - delete invalid checkpoint
        await manager.delete_checkpoint(cp_id)
        return None
```

### 2. Checkpoint Verification Tests

```python
async def test_checkpoint_recovery():
    """Test checkpoint creation and recovery"""
    # Create test problem
    problem = {
        'id': 'test_problem',
        'statement': 'Test checkpoint recovery',
    }

    # Create checkpoint
    manager = create_checkpoint_manager()
    cp_id = await manager.create_checkpoint(
        problem=problem,
        context={'test': True},
        solutions={'test': 'result'},
        level=0,
        stage='test'
    )

    # Test recovery
    state = await manager.load_checkpoint(cp_id)
    assert state is not None, "Failed to load checkpoint"
    assert state.problem == problem, "Problem mismatch"

    # Cleanup
    await manager.delete_checkpoint(cp_id)
    print("✅ Checkpoint recovery test passed")
```

### 3. Monitoring Setup

```python
# Monitor checkpoint health
import asyncio

async def monitor_checkpoints():
    """Monitor checkpoint health periodically"""
    while True:
        problems = await list_all_problems()

        for problem_id in problems:
            checkpoints = await manager.list_checkpoints(problem_id)

            for cp in checkpoints:
                # Validate each checkpoint
                try:
                    state = await manager.load_checkpoint(cp['checkpoint_id'])
                    if not state:
                        logger.warning(f"Invalid checkpoint: {cp['checkpoint_id']}")
                except Exception as e:
                    logger.error(f"Checkpoint error: {e}")

        # Check every 5 minutes
        await asyncio.sleep(300)

# Start monitoring
asyncio.create_task(monitor_checkpoints())
```

---

## Summary

This troubleshooting guide covers:
- ✅ Common issues and solutions
- ✅ Corrupted checkpoint detection and repair
- ✅ Manual checkpoint editing procedures
- ✅ Recovery procedures for various scenarios
- ✅ Emergency recovery options

For additional help:
- See `CHECKPOINTING_GUIDE.md` for usage
- See implementation in `bubblelabs_nodes/checkpoint_manager.py`
- Check logs for detailed error messages
