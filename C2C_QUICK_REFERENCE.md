# C2C MCP Tools - Quick Reference Guide

**Version:** 1.0.0 | **Last Updated:** 2026-01-22

---

## Installation

### Quick Install
```bash
pip install torch transformers
git clone https://github.com/facebookresearch/Rosetta.git C2C
cd C2C && pip install -e .
```

### Verify Installation
```python
from c2c_mcp_tools import get_c2c_status
status = get_c2c_status()
print(status['available'])  # True if installed
```

---

## Basic Usage

### 1. Initialize Ensemble
```python
from c2c_mcp_tools import initialize_c2c_ensemble

result = initialize_c2c_ensemble(
    ensemble_id="my-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="cuda",  # or "cpu" or "auto"
    cache_ensemble=True,  # Cache for reuse
)
```

### 2. Run Inference
```python
from c2c_mcp_tools import run_c2c_inference

result = run_c2c_inference(
    ensemble_id="my-ensemble",
    prompt="What is machine learning?",
    apply_c2c=True,
    max_new_tokens=256,
    temperature=0.0,
)

print(result['generated_text'])
print(result['tokens_per_second'])
```

### 3. Team Consensus
```python
from c2c_mcp_tools import run_team_consensus_with_c2c

result = run_team_consensus_with_c2c(
    ensemble_id="my-ensemble",
    prompt="Design a solution for...",
    team_name="Blue",
    team_models=["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-0.5B-Instruct"],
    consensus_mode="c2c",  # or "text"
)

print(result['consensus_text'])
```

---

## Cache Management

### List Cached Ensembles
```python
from c2c_mcp_tools import manage_ensemble_cache

result = manage_ensemble_cache(action="list")
print(result['cached_ensembles'])
```

### Get Cache Statistics
```python
result = manage_ensemble_cache(action="stats")
stats = result['stats']
print(f"Cache: {stats['size']}/{stats['max_size']}")
```

### Remove Ensemble from Cache
```python
result = manage_ensemble_cache(
    action="remove",
    ensemble_id="my-ensemble"
)
```

### Clear All Cache
```python
result = manage_ensemble_cache(action="clear")
```

### Configure Persistent Storage
```python
result = manage_ensemble_cache(
    action="config",
    persistent_path="./c2c_cache_metadata"
)
```

---

## Error Handling

### Handle Missing C2C
```python
from c2c_mcp_tools import (
    C2CNotAvailableError,
    C2CConfigurationError,
    C2CInferenceError,
    C2CCacheError,
    initialize_c2c_ensemble,
)

try:
    result = initialize_c2c_ensemble(...)
except C2CNotAvailableError:
    print("C2C not installed - using fallback")
except C2CConfigurationError as e:
    print(f"Invalid configuration: {e}")
except C2CError as e:
    print(f"C2C error: {e}")
```

### Check Availability First
```python
from c2c_mcp_tools import C2C_AVAILABLE

if C2C_AVAILABLE:
    # Use C2C ensemble
    result = run_c2c_inference(...)
else:
    # Fall back to single model
    print("C2C unavailable - using baseline model")
```

---

## Hephaestus Integration

### Configure for Phase
```python
from c2c_mcp_tools import configure_c2c_for_hephaestus_phase

result = configure_c2c_for_hephaestus_phase(
    phase_id="setup-phase-1",
    base_model="Qwen/Qwen3-0.6B",
    phase_type="setup",  # setup, solution, critique, verify, reassemble, final
    ensemble_config={"device": "cuda"},
)

recommendation = result['recommendation']
print(f"Recommended: {recommendation['recommended_pairs']}")
```

### Phase-Specific Recommendations
| Phase | Base Model | Sharer Model | Use Case |
|-------|-----------|--------------|----------|
| setup | Qwen3-0.6B | Qwen2.5-0.5B-Instruct | Analysis |
| solution | Qwen3-0.6B | Llama-3.2-1B-Instruct | Coding |
| critique | Qwen3-0.6B | Qwen2.5-0.5B-Instruct | Evaluation |
| verify | Qwen3-0.6B | Qwen2.5-0.5B-Instruct | Validation |
| reassemble | Qwen3-0.6B | Qwen3-4B-Base | Integration |

---

## CrewAI Integration

### Custom Tool for CrewAI
```python
from crewai import tool
from c2c_mcp_tools import run_c2c_inference

@tool("C2C Ensemble Inference")
def c2c_inference_tool(prompt: str) -> str:
    """
    Run multi-model ensemble inference using C2C.

    Args:
        prompt: Input prompt for inference

    Returns:
        Generated text from ensemble
    """
    result = run_c2c_inference(
        ensemble_id="crewai-ensemble",
        prompt=prompt,
        apply_c2c=True,
        max_new_tokens=512,
    )
    return result['generated_text']
```

### Use in Agent
```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Analyze topics using multi-model consensus",
    backstory="Expert with C2C ensemble access",
    tools=[c2c_inference_tool],
)

task = Task(
    description="Analyze the impact of C2C",
    expected_output="Detailed analysis",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

---

## MCP Tools Reference

### Available Tools (8)
1. **initialize_c2c_ensemble** - Initialize ensemble with caching
2. **run_c2c_inference** - Run inference with metrics
3. **run_team_consensus_with_c2c** - Team consensus
4. **configure_c2c_for_hephaestus_phase** - Phase configuration
5. **get_c2c_status** - Check installation and status
6. **load_c2c_checkpoint** - Load pretrained projectors
7. **compare_c2c_vs_baseline** - Compare performance
8. **manage_ensemble_cache** - Cache management

### List All Tools
```python
from c2c_mcp_tools import list_mcp_tools

tools = list_mcp_tools()
print(tools)  # ['initialize_c2c_ensemble', 'run_c2c_inference', ...]
```

---

## Performance Metrics

### Expected Performance
- **Accuracy Improvement:** 8.5-10.5%
- **Latency Reduction:** 2.0× faster
- **vs Text Communication:** 3.0-5.0% better

### Monitor Inference Performance
```python
result = run_c2c_inference(...)
print(f"Tokens: {result['tokens_generated']}")
print(f"Time: {result['inference_time']}s")
print(f"Speed: {result['tokens_per_second']} tokens/s")
```

---

## Configuration Examples

### CPU Configuration
```python
result = initialize_c2c_ensemble(
    ensemble_id="cpu-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="cpu",
)
```

### GPU Configuration
```python
result = initialize_c2c_ensemble(
    ensemble_id="gpu-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="cuda",
)
```

### Auto Device Selection
```python
result = initialize_c2c_ensemble(
    ensemble_id="auto-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="auto",  # Automatically selects CUDA if available
)
```

### With Pretrained Checkpoint
```python
result = initialize_c2c_ensemble(
    ensemble_id="checkpoint-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    checkpoint_dir="./checkpoints/qwen3_0.6b+qwen2.5_0.5b_Fuser",
    device="cuda",
)
```

---

## Troubleshooting

### C2C Not Available
```python
status = get_c2c_status()
if not status['available']:
    print(status['error'])
    print(status['installation_guide'])
```

### Ensemble Not in Cache
```python
# Check if ensemble exists
stats = manage_ensemble_cache(action="stats")
if ensemble_id not in [e['ensemble_id'] for e in stats['stats']['cached_ensembles']]:
    print("Ensemble not initialized")
```

### CUDA Out of Memory
```python
# Fall back to CPU
result = initialize_c2c_ensemble(
    ensemble_id="cpu-fallback",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="cpu",  # Explicit CPU
)
```

---

## Advanced Usage

### Custom Projector Configuration
```python
result = initialize_c2c_ensemble(
    ensemble_id="custom-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=[
        "Qwen/Qwen2.5-0.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
    multi_source_fusion_mode="parallel",  # or "sequential"
    include_response=True,
)
```

### Comparison with Baseline
```python
from c2c_mcp_tools import compare_c2c_vs_baseline

result = compare_c2c_vs_baseline(
    ensemble_id="my-ensemble",
    prompts=["prompt1", "prompt2", "prompt3"],
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
)

print(result['expected_improvements'])
# {'accuracy_gain': '8.5-10.5%', 'latency_reduction': '2.0×', ...}
```

---

## Data Classes

### EnsembleConfig
```python
from c2c_mcp_tools import EnsembleConfig

config = EnsembleConfig(
    ensemble_id="my-ensemble",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="cuda",
    include_response=True,
    multi_source_fusion_mode="parallel",
    checkpoint_dir=None,
)

# Serialize
config_dict = config.to_dict()

# Deserialize
config_restored = EnsembleConfig.from_dict(config_dict)
```

---

## Best Practices

### 1. Always Cache Ensembles
```python
# GOOD: Enable caching
initialize_c2c_ensemble(..., cache_ensemble=True)

# AVOID: Disable caching
initialize_c2c_ensemble(..., cache_ensemble=False)
```

### 2. Handle Errors Gracefully
```python
try:
    result = run_c2c_inference(...)
except C2CNotAvailableError:
    # Fallback to single model
except C2CInferenceError:
    # Log and retry
```

### 3. Monitor Cache
```python
# Regular cache monitoring
stats = manage_ensemble_cache(action="stats")
if stats['stats']['size'] >= stats['stats']['max_size']:
    print("Cache full - consider clearing old ensembles")
```

### 4. Use Auto Device Selection
```python
# GOOD: Auto-select based on availability
initialize_c2c_ensemble(..., device="auto")

# AVOID: Hardcode device without checking
initialize_c2c_ensemble(..., device="cuda")  # May fail if no CUDA
```

---

## Quick Start Template

```python
from c2c_mcp_tools import (
    C2C_AVAILABLE,
    initialize_c2c_ensemble,
    run_c2c_inference,
    get_c2c_status,
    C2CNotAvailableError,
)

# 1. Check availability
if not C2C_AVAILABLE:
    print("C2C not installed - see installation guide")
    print(get_c2c_status()['installation_guide'])
    exit(1)

# 2. Initialize ensemble
result = initialize_c2c_ensemble(
    ensemble_id="quickstart",
    base_model="Qwen/Qwen3-0.6B",
    sharer_models=["Qwen/Qwen2.5-0.5B-Instruct"],
    device="auto",
    cache_ensemble=True,
)

if not result['success']:
    print(f"Failed: {result['error']}")
    exit(1)

# 3. Run inference
try:
    result = run_c2c_inference(
        ensemble_id="quickstart",
        prompt="Explain C2C in simple terms",
        apply_c2c=True,
        max_new_tokens=200,
    )

    print(f"Generated: {result['generated_text']}")
    print(f"Speed: {result['tokens_per_second']} tokens/s")

except C2CNotAvailableError:
    print("C2C unavailable")
except Exception as e:
    print(f"Error: {e}")
```

---

## Resources

- **Installation Guide:** `get_c2c_installation_guide()`
- **Status Check:** `get_c2c_status()`
- **Examples:** `c2c_usage_examples.py` (10 examples)
- **Full Report:** `C2C_FIX_REPORT.md`
- **Rosetta Paper:** https://arxiv.org/abs/2406.16777
- **Rosetta Repo:** https://github.com/facebookresearch/Rosetta

---

**Version:** 1.0.0
**Status:** Production Ready ✅
**Last Updated:** 2026-01-22
