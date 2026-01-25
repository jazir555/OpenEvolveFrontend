# Gauntlet Configuration System Guide

This guide explains how to configure and customize the OpenEvolve Gauntlet system for your specific needs.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Options](#configuration-options)
3. [Environment Variables](#environment-variables)
4. [Strategy Profiles](#strategy-profiles)
5. [Configuration Files](#configuration-files)
6. [Validation](#validation)
7. [Best Practices](#best-practices)

---

## Quick Start

### Default Configuration

```python
from bubblelabs_nodes import create_config

# Create configuration with defaults
config = create_config()

# Use the configuration
print(f"Max gauntlet rounds: {config.max_gauntlet_rounds}")
print(f"Pass threshold: {config.pass_threshold}")
```

### Environment-Based Configuration

```python
# Set environment variables
import os
os.environ['CACHE_ENABLED'] = 'true'
os.environ['MAX_GAUNTLET_ROUNDS'] = '5'

# Create configuration from environment
config = create_config(from_env=True)
```

### Profile-Based Configuration

```python
from bubblelabs_nodes import create_config, StrategyProfile

# Apply a strategy profile
config = create_config(profile=StrategyProfile.CONSERVATIVE)
```

---

## Configuration Options

### Cache Configuration

Controls solution caching behavior.

```python
from bubblelabs_nodes import CacheConfig, CacheType

cache_config = CacheConfig(
    enabled=True,
    cache_type=CacheType.MEMORY,  # or CacheType.REDIS, CacheType.NONE
    ttl_seconds=3600,  # 1 hour
    max_size=1000,
    redis_url="redis://localhost:6379"  # Required if using REDIS
)
```

**Options:**
- `enabled` (bool): Enable/disable caching (default: True)
- `cache_type` (CacheType): Cache backend - memory, redis, or none
- `ttl_seconds` (int): Time-to-live for cached entries (default: 3600)
- `max_size` (int): Maximum number of cached entries (default: 1000)
- `redis_url` (str): Redis connection URL (required for Redis backend)

### Checkpointing Configuration

Controls crash recovery and state persistence.

```python
from bubblelabs_nodes import CheckpointConfig, CheckpointFrequency

checkpoint_config = CheckpointConfig(
    enabled=True,
    storage_path="./gauntlet_checkpoints",
    compression=True,
    frequency=CheckpointFrequency.MAJOR,  # MAJOR, MINOR, or ALL
    retention_count=5,  # Keep last 5 checkpoints
    auto_cleanup=True
)
```

**Options:**
- `enabled` (bool): Enable/disable checkpointing (default: True)
- `storage_path` (str): Directory for checkpoint storage (default: ./gauntlet_checkpoints)
- `compression` (bool): Compress checkpoint data (default: True)
- `frequency` (CheckpointFrequency): How often to create checkpoints
  - `MAJOR`: Only at major milestones (decomposition, reassembly)
  - `MINOR`: At minor milestones (after each atomic problem)
  - `ALL`: After every operation
- `retention_count` (int): Number of checkpoints to retain (default: 5)
- `auto_cleanup` (bool): Automatically cleanup old checkpoints (default: True)

### Parallel Execution Configuration

Controls parallel problem solving behavior.

```python
from bubblelabs_nodes import ParallelExecutionConfig

parallel_config = ParallelExecutionConfig(
    enabled=True,
    max_parallelism=10,  # Maximum concurrent operations
    timeout_seconds=300,  # 5 minutes
    use_worker_pool=False,
    worker_pool_size=4
)
```

**Options:**
- `enabled` (bool): Enable/disable parallel execution (default: True)
- `max_parallelism` (int): Maximum concurrent operations (default: 10)
- `timeout_seconds` (int): Timeout for parallel operations (default: 300)
- `use_worker_pool` (bool): Use worker pool instead of asyncio (default: False)
- `worker_pool_size` (int): Number of worker processes (default: 4)

### Circuit Breaker Configuration

Controls fault isolation and recovery behavior.

```python
from bubblelabs_nodes import CircuitBreakerConfig, CircuitBreakerStrategy

circuit_breaker_config = CircuitBreakerConfig(
    enabled=True,
    strategy=CircuitBreakerStrategy.HIERARCHICAL,  # INDIVIDUAL, HIERARCHICAL, or GLOBAL
    failure_threshold=5,  # Open circuit after 5 failures
    recovery_timeout_seconds=60,  # Wait 60s before attempting recovery
    half_open_max_calls=3  # Try 3 calls in half-open state
)
```

**Options:**
- `enabled` (bool): Enable/disable circuit breakers (default: True)
- `strategy` (CircuitBreakerStrategy): Circuit breaker strategy
  - `INDIVIDUAL`: Per-problem breaker
  - `HIERARCHICAL`: Per-level breaker (recommended)
  - `GLOBAL`: Single breaker for all operations
- `failure_threshold` (int): Failures before opening circuit (default: 5)
- `recovery_timeout_seconds` (int): Cooldown before recovery attempts (default: 60)
- `half_open_max_calls` (int): Test calls in half-open state (default: 3)

### Fuzzing Configuration

Controls automated vulnerability testing.

```python
from bubblelabs_nodes import FuzzingConfig

fuzzing_config = FuzzingConfig(
    enabled=False,  # Disabled by default
    max_iterations=1000,
    timeout_seconds=30,
    crash_analysis_enabled=True,
    input_types=["auto", "random", "boundary"]
)
```

**Options:**
- `enabled` (bool): Enable/disable fuzzing (default: False)
- `max_iterations` (int): Maximum fuzzing iterations (default: 1000)
- `timeout_seconds` (int): Timeout per fuzzing test (default: 30)
- `crash_analysis_enabled` (bool): Analyze crashes (default: True)
- `input_types` (List[str]): Input generation strategies

### Dynamic Difficulty Configuration

Controls adaptive difficulty adjustment.

```python
from bubblelabs_nodes import DifficultyConfig, DifficultyLevel

difficulty_config = DifficultyConfig(
    enabled=True,
    initial_level=DifficultyLevel.MEDIUM,
    adjustment_window=10,  # Adjust after every 10 problems
    success_threshold=0.7,  # Increase difficulty if >70% success
    failure_threshold=0.3  # Decrease difficulty if <30% success
)
```

**Options:**
- `enabled` (bool): Enable/disable dynamic difficulty (default: True)
- `initial_level` (DifficultyLevel): Starting difficulty (1-5)
  - `TRIVIAL` (1): Very easy problems
  - `EASY` (2): Easy problems
  - `MEDIUM` (3): Medium difficulty (default)
  - `HARD` (4): Difficult problems
  - `EXPERT` (5): Expert-level problems
- `adjustment_window` (int): Problems between adjustments (default: 10)
- `success_threshold` (float): Success rate to increase difficulty (default: 0.7)
- `failure_threshold` (float): Success rate to decrease difficulty (default: 0.3)

### ML Decomposition Configuration

Controls ML-based decomposition depth prediction.

```python
from bubblelabs_nodes import MLDecompositionConfig

ml_config = MLDecompositionConfig(
    enabled=False,  # Disabled by default
    model_path=None,  # Path to trained model
    data_collection_path="./data/decomposition",
    auto_train=False,
    training_threshold=100  # Train after 100 examples
)
```

**Options:**
- `enabled` (bool): Enable/disable ML decomposition (default: False)
- `model_path` (str): Path to trained ML model
- `data_collection_path` (str): Training data storage (default: ./data/decomposition)
- `auto_train` (bool): Automatically retrain model (default: False)
- `training_threshold` (int): Minimum examples before training (default: 100)

### Plugin Configuration

Controls plugin system behavior.

```python
from bubblelabs_nodes import PluginConfig

plugin_config = PluginConfig(
    enabled=False,  # Disabled by default
    plugin_dir="./plugins",
    sandbox_enabled=True,
    sandbox_timeout=30.0,
    auto_load=False
)
```

**Options:**
- `enabled` (bool): Enable/disable plugin system (default: False)
- `plugin_dir` (str): Directory for plugins (default: ./plugins)
- `sandbox_enabled` (bool): Isolate plugin execution (default: True)
- `sandbox_timeout` (float): Plugin execution timeout (default: 30.0)
- `auto_load` (bool): Automatically load plugins on startup (default: False)

---

## Environment Variables

All configuration options can be set via environment variables.

### Naming Convention

Environment variables use UPPER_SNAKE_CASE:

```bash
# Cache
CACHE_ENABLED=true
CACHE_TYPE=redis
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=1000
CACHE_REDIS_URL=redis://localhost:6379

# Checkpointing
CHECKPOINTING_ENABLED=true
CHECKPOINTING_STORAGE_PATH=./checkpoints
CHECKPOINTING_COMPRESSION=true
CHECKPOINTING_FREQUENCY=major
CHECKPOINTING_RETENTION_COUNT=5

# Parallel Execution
PARALLEL_EXECUTION_ENABLED=true
PARALLEL_EXECUTION_MAX_PARALLELISM=10
PARALLEL_EXECUTION_TIMEOUT_SECONDS=300

# Circuit Breaker
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_STRATEGY=hierarchical
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=60

# Fuzzing
FUZZING_ENABLED=false
FUZZING_MAX_ITERATIONS=1000

# Difficulty
DIFFICULTY_ADJUSTMENT_ENABLED=true
DIFFICULTY_INITIAL_LEVEL=3

# ML Decomposition
ML_DECOMPOSITION_ENABLED=false
ML_DECOMPOSITION_MODEL_PATH=./models/decomposition.pkl

# Plugin System
PLUGIN_ENABLED=false
PLUGIN_DIR=./plugins
PLUGIN_SANDBOX_ENABLED=true

# General Settings
MAX_GAUNTLET_ROUNDS=3
PASS_THRESHOLD=0.75
MAX_DECOMPOSITION_DEPTH=3
LOG_LEVEL=INFO
STRATEGY_PROFILE=balanced
```

### Loading from Environment

```python
from bubblelabs_nodes import create_config

# Load all settings from environment
config = create_config(from_env=True)
```

---

## Strategy Profiles

Predefined configurations for common use cases.

### Conservative Profile

**Use case:** Production systems where correctness is critical

```python
from bubblelabs_nodes import create_config, StrategyProfile

config = create_config(profile=StrategyProfile.CONSERVATIVE)
```

**Settings:**
- Max gauntlet rounds: 5
- Pass threshold: 0.85
- Max decomposition depth: 5
- Initial difficulty: EASY
- Fuzzing: Disabled
- Circuit breakers: Enabled

### Balanced Profile (Default)

**Use case:** General-purpose problem solving

```python
config = create_config(profile=StrategyProfile.BALANCED)
```

**Settings:**
- Max gauntlet rounds: 3
- Pass threshold: 0.75
- Max decomposition depth: 3
- Initial difficulty: MEDIUM
- Fuzzing: Disabled
- Circuit breakers: Enabled

### Aggressive Profile

**Use case:** Fast iteration on non-critical problems

```python
config = create_config(profile=StrategyProfile.AGGRESSIVE)
```

**Settings:**
- Max gauntlet rounds: 2
- Pass threshold: 0.65
- Max decomposition depth: 2
- Initial difficulty: HARD
- Fuzzing: Disabled
- Circuit breakers: Enabled

### Fast Profile

**Use case:** Maximum performance on well-understood problems

```python
config = create_config(profile=StrategyProfile.FAST)
```

**Settings:**
- Max gauntlet rounds: 1
- Pass threshold: 0.60
- Max decomposition depth: 2
- Parallel execution: Enabled (max 20)
- Caching: Enabled
- Fuzzing: Disabled

### Thorough Profile

**Use case:** Maximum quality assurance on critical problems

```python
config = create_config(profile=StrategyProfile.THOROUGH)
```

**Settings:**
- Max gauntlet rounds: 5
- Pass threshold: 0.90
- Max decomposition depth: 5
- Fuzzing: Enabled
- Circuit breakers: Enabled
- Extensive validation

---

## Configuration Files

### JSON Configuration File

Save configuration to JSON for reproducibility.

```python
from bubblelabs_nodes import create_config, StrategyProfile

# Create and save configuration
config = create_config(profile=StrategyProfile.BALANCED)
config.save_to_file("./gauntlet_config.json")
```

**Example configuration file:**

```json
{
  "max_gauntlet_rounds": 3,
  "pass_threshold": 0.75,
  "max_decomposition_depth": 3,
  "log_level": "INFO",
  "strategy_profile": "balanced",
  "cache": {
    "enabled": true,
    "cache_type": "memory",
    "ttl_seconds": 3600,
    "max_size": 1000,
    "redis_url": null
  },
  "checkpointing": {
    "enabled": true,
    "storage_path": "./gauntlet_checkpoints",
    "compression": true,
    "frequency": "major",
    "retention_count": 5,
    "auto_cleanup": true
  },
  "parallel_execution": {
    "enabled": true,
    "max_parallelism": 10,
    "timeout_seconds": 300,
    "use_worker_pool": false,
    "worker_pool_size": 4
  },
  "circuit_breaker": {
    "enabled": true,
    "strategy": "hierarchical",
    "failure_threshold": 5,
    "recovery_timeout_seconds": 60,
    "half_open_max_calls": 3
  },
  "fuzzing": {
    "enabled": false,
    "max_iterations": 1000,
    "timeout_seconds": 30,
    "crash_analysis_enabled": true,
    "input_types": ["auto"]
  },
  "difficulty": {
    "enabled": true,
    "initial_level": 3,
    "adjustment_window": 10,
    "success_threshold": 0.7,
    "failure_threshold": 0.3
  },
  "ml_decomposition": {
    "enabled": false,
    "model_path": null,
    "data_collection_path": "./data/decomposition",
    "auto_train": false,
    "training_threshold": 100
  },
  "plugin": {
    "enabled": false,
    "plugin_dir": "./plugins",
    "sandbox_enabled": true,
    "sandbox_timeout": 30.0,
    "auto_load": false
  }
}
```

### Loading Configuration File

```python
from bubblelabs_nodes import create_config

# Load from file
config = create_config(config_file="./gauntlet_config.json")
```

### Combining File, Environment, and Profile

Configuration precedence (highest to lowest):
1. Configuration file
2. Environment variables
3. Strategy profile
4. Defaults

```python
config = create_config(
    config_file="./gauntlet_config.json",  # Base config
    profile=StrategyProfile.CONSERVATIVE,  # Override with profile
    from_env=True  # Override with environment
)
```

---

## Validation

### Automatic Validation

Configuration is automatically validated on creation:

```python
from bubblelabs_nodes import create_config

try:
    config = create_config()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

### Manual Validation

```python
from bubblelabs_nodes import GauntletConfig

config = GauntletConfig()
is_valid, errors = config.validate()

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

### Common Validation Errors

**Example Error 1: Invalid TTL**
```
Configuration validation failed:
  - cache.ttl_seconds must be non-negative
```

**Fix:**
```python
config.cache.ttl_seconds = 3600  # Valid value
```

**Example Error 2: Missing Redis URL**
```
Configuration validation failed:
  - cache.redis_url required when cache_type is REDIS
```

**Fix:**
```python
config.cache.redis_url = "redis://localhost:6379"
# or change cache type
config.cache.cache_type = CacheType.MEMORY
```

**Example Error 3: Invalid Thresholds**
```
Configuration validation failed:
  - difficulty.success_threshold must be greater than failure_threshold
```

**Fix:**
```python
config.difficulty.success_threshold = 0.7
config.difficulty.failure_threshold = 0.3  # Must be less than success_threshold
```

---

## Best Practices

### 1. Development vs Production

**Development:**
```python
# Fast iteration, lenient validation
config = create_config(profile=StrategyProfile.FAST)
config.log_level = "DEBUG"
config.checkpointing.frequency = CheckpointFrequency.MINOR
```

**Production:**
```python
# Maximum reliability, strict validation
config = create_config(profile=StrategyProfile.CONSERVATIVE)
config.log_level = "WARNING"
config.checkpointing.frequency = CheckpointFrequency.MAJOR
config.circuit_breaker.enabled = True
config.fuzzing.enabled = True
```

### 2. Resource Management

**High-Resource Environment:**
```python
# Use all available resources
config.parallel_execution.max_parallelism = 20
config.cache.max_size = 10000
config.checkpointing.retention_count = 10
```

**Low-Resource Environment:**
```python
# Conserve resources
config.parallel_execution.max_parallelism = 3
config.cache.max_size = 100
config.cache.cache_type = CacheType.NONE  # Disable caching
config.checkpointing.compression = True  # Save disk space
```

### 3. Testing

**Unit Tests:**
```python
# Deterministic, fast
config = create_config(profile=StrategyProfile.FAST)
config.cache.cache_type = CacheType.NONE  # No caching
config.checkpointing.enabled = False  # No checkpointing
config.fuzzing.enabled = False
```

**Integration Tests:**
```python
# Realistic configuration
config = create_config(profile=StrategyProfile.BALANCED)
config.checkpointing.storage_path = "./test_checkpoints"
config.cache.cache_type = CacheType.MEMORY
```

### 4. Monitoring

**Enable Metrics Collection:**
```python
from bubblelabs_nodes import get_metrics_collector, track_performance

# Collect metrics for monitoring
collector = get_metrics_collector()
collector.start_resource_monitoring(interval_seconds=5.0)

# Use performance tracking decorator
@track_performance("solve_problem")
async def solve_problem(problem):
    # Your implementation
    pass
```

### 5. Security

**Production Security:**
```python
# Enable all security features
config.fuzzing.enabled = True
config.fuzzing.crash_analysis_enabled = True
config.circuit_breaker.enabled = True
config.circuit_breaker.strategy = CircuitBreakerStrategy.HIERARCHICAL
config.plugin.sandbox_enabled = True
```

---

## Summary

The Gauntlet configuration system provides:
- ✅ Flexible configuration via code, environment, or files
- ✅ Predefined strategy profiles for common use cases
- ✅ Automatic validation with clear error messages
- ✅ Configuration layering (file > env > profile > defaults)
- ✅ Type-safe configuration with dataclasses

For more information:
- See `bubblelabs_nodes/gauntlet_config.py` for implementation
- See `CHECKPOINTING_GUIDE.md` for checkpointing details
- See `bubblelabs_nodes/gauntlet_metrics.py` for monitoring
