# ACE Checkpoint System Documentation

## Overview

The checkpoint system in ACE (Agentic Context Engineering) provides automatic saving of skillbook state during offline training, enabling training resumption, comparison of skillbook evolution, and recovery from interruptions.

## Architecture

### Location
- **File Path**: `ace/adaptation.py` - OfflineACE.run() method (lines 604-721)
- **Checkpoint Logic**: Lines 679-694

### Components

```python
def run(
    self,
    samples: Sequence[Sample],
    environment: TaskEnvironment,
    epochs: int = 1,
    checkpoint_interval: Optional[int] = None,  # NEW
    checkpoint_dir: Optional[str] = None,       # NEW
    wait_for_learning: bool = True,
) -> List[ACEStepResult]:
```

## Checkpoint Format

### File Naming Convention

```
{checkpoint_dir}/
├── ace_checkpoint_10.json      # Numbered checkpoint (10 samples processed)
├── ace_checkpoint_20.json      # Numbered checkpoint (20 samples processed)
├── ace_checkpoint_30.json      # Numbered checkpoint (30 samples processed)
└── ace_latest.json             # Always contains most recent checkpoint
```

### File Content

Each checkpoint file contains a complete skillbook in JSON format:

```json
{
  "skills": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "section": "general",
      "content": "Always verify your calculations before presenting the final answer",
      "metadata": {
        "helpful": 5,
        "harmful": 0
      }
    }
  ]
}
```

### Metadata (Implicit)

While checkpoints don't include explicit metadata fields, the file naming encodes training progress:
- **Number**: Total samples processed across all epochs
- **Latest**: Most recent state (overwritten on each save)

## Behavior

### Saving Logic

1. **Checkpoint Interval**: Checkpoints are saved every N successful samples
2. **Accumulation**: Numbered checkpoints accumulate (not deleted)
3. **Latest Alias**: `ace_latest.json` is always overwritten with most recent checkpoint
4. **Failed Samples**: Skipped and don't count toward checkpoint interval
5. **Directory Creation**: Automatically creates checkpoint directory if missing

### Checkpoint Triggering

```python
# In adaptation.py, lines 679-694
if (
    checkpoint_interval
    and checkpoint_dir
    and len(results) % checkpoint_interval == 0
):
    checkpoint_path = Path(checkpoint_dir)
    numbered_checkpoint = (
        checkpoint_path / f"ace_checkpoint_{len(results)}.json"
    )
    latest_checkpoint = checkpoint_path / "ace_latest.json"

    self.skillbook.save_to_file(str(numbered_checkpoint))
    self.skillbook.save_to_file(str(latest_checkpoint))
    logger.info(
        f"Checkpoint saved: {len(results)} samples → {numbered_checkpoint.name}"
    )
```

### Example Timeline

```
Training Progress (25 samples, checkpoint_interval=10, epochs=1):
├── Sample 1-9:   No checkpoint
├── Sample 10:    ✓ Save ace_checkpoint_10.json, ace_latest.json
├── Sample 11-19: No checkpoint
├── Sample 20:    ✓ Save ace_checkpoint_20.json, ace_latest.json
└── Sample 21-25: No checkpoint (interval not reached)
```

## Usage Examples

### Basic Checkpointing

```python
from ace import OfflineACE, Agent, Reflector, SkillManager, Sample
from ace.llm_providers import LiteLLMClient

# Setup
client = LiteLLMClient(model="gpt-4")
agent = Agent(client)
reflector = Reflector(client)
skill_manager = SkillManager(client)

ace = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)

# Training with checkpoints
results = ace.run(
    samples=train_samples,
    environment=environment,
    epochs=3,
    checkpoint_interval=100,    # Save every 100 samples
    checkpoint_dir="./checkpoints"
)
```

### Resume from Checkpoint

```python
from ace import Skillbook, OfflineACE

# Load checkpoint
loaded_skillbook = Skillbook.load_from_file("./checkpoints/ace_latest.json")

# Create new ACE instance with loaded skillbook
ace_2 = OfflineACE(
    skillbook=loaded_skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager
)

# Continue training
results_2 = ace_2.run(
    samples=more_samples,
    environment=environment,
    checkpoint_interval=100,
    checkpoint_dir="./checkpoints"
)
```

### Compare Skillbook Evolution

```python
from ace import Skillbook

# Load different checkpoints
checkpoint_10 = Skillbook.load_from_file("./checkpoints/ace_checkpoint_10.json")
checkpoint_20 = Skillbook.load_from_file("./checkpoints/ace_checkpoint_20.json")
checkpoint_30 = Skillbook.load_from_file("./checkpoints/ace_checkpoint_30.json")

# Compare skill counts
print(f"Checkpoint 10: {len(checkpoint_10.skills())} skills")
print(f"Checkpoint 20: {len(checkpoint_20.skills())} skills")
print(f"Checkpoint 30: {len(checkpoint_30.skills())} skills")

# Analyze skill evolution
skills_10 = set(s.content for s in checkpoint_10.skills())
skills_20 = set(s.content for s in checkpoint_20.skills())

new_skills = skills_20 - skills_10
print(f"New skills learned between checkpoint 10-20: {len(new_skills)}")
```

### Multi-Epoch Training with Checkpoints

```python
# 2 epochs over 50 samples = 100 total samples
results = ace.run(
    samples=train_samples[:50],
    environment=environment,
    epochs=2,
    checkpoint_interval=25,    # Checkpoints at 25, 50, 75, 100
    checkpoint_dir="./checkpoints"
)

# Resulting files:
# - ace_checkpoint_25.json   (End of epoch 1, sample 25)
# - ace_checkpoint_50.json   (End of epoch 1)
# - ace_checkpoint_75.json   (Epoch 2, sample 25)
# - ace_checkpoint_100.json  (End of epoch 2)
# - ace_latest.json          (Same as ace_checkpoint_100.json)
```

## Validation and Error Handling

### Parameter Validation

```python
# Error: checkpoint_interval requires checkpoint_dir
ace.run(samples, environment, checkpoint_interval=10)
# Raises: ValueError("checkpoint_dir must be provided when checkpoint_interval is set")
```

### Directory Creation

```python
# Non-existent directories are created automatically
ace.run(
    samples,
    environment,
    checkpoint_interval=10,
    checkpoint_dir="./non/existent/path"  # ✓ Created automatically
)
```

### Edge Cases

1. **Interval > Total Samples**: No checkpoints saved
2. **Zero Samples**: No checkpoints created
3. **Interval = 1**: Checkpoint after every sample
4. **Failed Samples**: Skipped, don't affect checkpoint numbering

## Integration with Async Learning

The checkpoint system works seamlessly with async learning mode:

```python
ace = OfflineACE(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,
    max_reflector_workers=3
)

results = ace.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=100,
    checkpoint_dir="./checkpoints",
    wait_for_learning=True  # Wait for learning before checkpoints
)
```

**Important**: With async learning, checkpoints are saved after learning completes for those samples.

## Use Cases

### 1. Training Resumption

**Scenario**: Training interrupted at sample 750/1000

```python
# Resume from checkpoint 750
skillbook = Skillbook.load_from_file("./checkpoints/ace_checkpoint_750.json")
ace = OfflineACE(skillbook=skillbook, agent=agent, ...)

# Continue from sample 751
results = ace.run(
    samples[750:],
    environment,
    checkpoint_interval=250,
    checkpoint_dir="./checkpoints"
)
```

### 2. Early Stopping

**Scenario**: Monitor validation performance and stop at best checkpoint

```python
best_checkpoint = None
best_score = 0.0

for epoch in range(10):
    results = ace.run(
        train_samples,
        environment,
        epochs=1,
        checkpoint_interval=500,
        checkpoint_dir=f"./checkpoints/epoch_{epoch}"
    )

    # Evaluate on validation set
    val_score = evaluate(ace.skillbook, val_samples)

    if val_score > best_score:
        best_score = val_score
        best_checkpoint = f"./checkpoints/epoch_{epoch}/ace_latest.json"

# Load best checkpoint
best_skillbook = Skillbook.load_from_file(best_checkpoint)
```

### 3. Ablation Studies

**Scenario**: Compare different training configurations

```python
# Configuration 1
ace_1 = OfflineACE(..., checkpoint_interval=100, checkpoint_dir="./config1")
results_1 = ace_1.run(samples, environment, epochs=5)

# Configuration 2
ace_2 = OfflineACE(..., checkpoint_interval=100, checkpoint_dir="./config2")
results_2 = ace_2.run(samples, environment, epochs=5)

# Compare evolution
for i in range(1, 6):
    ckpt_1 = Skillbook.load_from_file(f"./config1/ace_checkpoint_{i*100}.json")
    ckpt_2 = Skillbook.load_from_file(f"./config2/ace_checkpoint_{i*100}.json")

    print(f"Epoch {i}: Config1={len(ckpt_1.skills())} skills, "
          f"Config2={len(ckpt_2.skills())} skills")
```

### 4. Skill Analysis

**Scenario**: Analyze which skills were learned when

```python
checkpoints = [10, 20, 30, 40, 50]
skill_evolution = {}

for i in range(len(checkpoints)):
    current = Skillbook.load_from_file(f"./ckpts/ace_checkpoint_{checkpoints[i]}.json")

    if i == 0:
        skill_evolution[checkpoints[i]] = set(s.content for s in current.skills())
    else:
        prev = Skillbook.load_from_file(f"./ckpts/ace_checkpoint_{checkpoints[i-1]}.json")
        prev_skills = set(s.content for s in prev.skills())
        curr_skills = set(s.content for s in current.skills())

        new_skills = curr_skills - prev_skills
        skill_evolution[checkpoints[i]] = new_skills
        print(f"Samples {checkpoints[i-1]}-{checkpoints[i]}: "
              f"Learned {len(new_skills)} new skills")
```

## Best Practices

### 1. Checkpoint Frequency

```python
# Good: Balance between storage space and granularity
checkpoint_interval = 100  # For large datasets (10K+ samples)
checkpoint_interval = 10   # For small datasets (<1K samples)
checkpoint_interval = 1    # For critical experiments (maximum granularity)
```

### 2. Checkpoint Organization

```python
# Organize by experiment
checkpoint_dir = "./experiments/exp_001/checkpoints"

# Organize by timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_dir = f"./checkpoints/{timestamp}"
```

### 3. Storage Management

```python
# Prune old checkpoints (keep every Nth)
import glob
import shutil

checkpoint_dir = "./checkpoints"
checkpoints = sorted(glob.glob(f"{checkpoint_dir}/ace_checkpoint_*.json"))

# Keep every 5th checkpoint
for ckpt in checkpoints[::5]:
    if ckpt != f"{checkpoint_dir}/ace_latest.json":
        os.remove(ckpt)
```

### 4. Validation Before Resume

```python
# Verify checkpoint integrity
try:
    skillbook = Skillbook.load_from_file(checkpoint_path)
    assert len(skillbook.skills()) > 0, "Empty skillbook"
    print("✓ Checkpoint valid")
except Exception as e:
    print(f"✗ Checkpoint corrupted: {e}")
    # Load previous checkpoint or start fresh
```

## Troubleshooting

### Issue: Checkpoints not being created

**Possible causes**:
1. `checkpoint_interval` not set
2. Training failing before reaching interval
3. Permission issues with checkpoint directory

**Solutions**:
```python
# Verify parameters
assert checkpoint_interval is not None
assert checkpoint_dir is not None

# Check file permissions
import os
os.makedirs(checkpoint_dir, exist_ok=True)
os.access(checkpoint_dir, os.W_OK)

# Check logs for training errors
# Logs show: "Failed to process sample X/Y: ..."
```

### Issue: Checkpoint file corrupted

**Possible causes**:
1. Disk full during write
2. Process killed during write
3. Concurrent writes

**Solutions**:
```python
# Validate checkpoint before loading
def load_checkpoint_safe(path: str) -> Skillbook:
    try:
        skillbook = Skillbook.load_from_file(path)
        return skillbook
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Checkpoint corrupted: {path}")
        # Try latest checkpoint
        latest = path.replace(f"_{n}.json", "_latest.json")
        return Skillbook.load_from_file(latest)
```

### Issue: Checkpoint numbering seems wrong

**Possible causes**:
1. Multiple epochs
2. Failed samples skipped
3. Resume from checkpoint

**Explanation**: Checkpoint number = total successful samples processed

```python
# Example: 50 samples, 2 epochs, checkpoint_interval=50
# Result: ace_checkpoint_50.json, ace_checkpoint_100.json
# (50 samples * 2 epochs = 100 total samples)
```

## Performance Considerations

### Checkpoint Overhead

- **I/O Time**: ~10-100ms per checkpoint (depends on skillbook size)
- **Storage Cost**: ~1-10MB per checkpoint (depends on skills)
- **Frequency Impact**: Negligible for intervals > 10

### Recommendations

| Dataset Size | Checkpoint Interval | Storage Cost | Recovery Granularity |
|--------------|---------------------|--------------|----------------------|
| < 1K         | 10-50              | < 50MB       | Excellent            |
| 1K-10K       | 100-500            | 50-500MB     | Good                 |
| > 10K        | 500-1000           | 500MB-2GB    | Fair                 |

## Testing

Comprehensive integration tests are available in:
- `tests/test_checkpoint_integration.py` (15 test cases)

### Test Coverage

- ✅ Checkpoint saving at correct intervals
- ✅ Checkpoint file format validation
- ✅ Checkpoint numbering accuracy
- ✅ Latest checkpoint management
- ✅ Resume from checkpoint functionality
- ✅ Multi-epoch checkpoint behavior
- ✅ Directory creation
- ✅ Existing skillbook preservation
- ✅ Parameter validation
- ✅ Edge cases (interval > samples, zero samples, interval=1)

### Running Tests

```bash
# Run all checkpoint tests
pytest tests/test_checkpoint_integration.py -v

# Run specific test
pytest tests/test_checkpoint_integration.py::TestCheckpointSavingDuringTraining::test_checkpoint_saving_during_training -v

# Run with coverage
pytest tests/test_checkpoint_integration.py --cov=ace.adaptation --cov-report=html
```

## Summary

The checkpoint system provides:

1. **Reliability**: Automatic saves prevent data loss
2. **Flexibility**: Resume training from any checkpoint
3. **Observability**: Track skillbook evolution over time
4. **Simplicity**: Easy to use with minimal configuration
5. **Robustness**: Handles edge cases and failures gracefully

**Key Files**:
- Implementation: `ace/adaptation.py` (lines 604-721, checkpoint logic: 679-694)
- Tests: `tests/test_checkpoint_integration.py`
- Documentation: `docs/CHECKPOINT_SYSTEM.md` (this file)

**Version**: ACE v0.5.0+
**Status**: Production-ready ✅
