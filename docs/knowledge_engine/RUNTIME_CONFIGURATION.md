# Runtime Configuration System

## Overview

The Runtime Configuration System enables dynamic configuration changes during evolution runs without requiring restarts. This powerful feature allows for adaptive optimization, resource-aware scaling, and hot-reloading of configuration files.

## Features

### 1. Runtime Configuration Updates

Update any configuration parameter during execution:

```python
from openevolve.config.runtime_config import RuntimeConfigUpdater
from openevolve.unified.config import UnifiedEvolutionConfig

# Initialize
config = UnifiedEvolutionConfig()
updater = RuntimeConfigUpdater(config)

# Update single parameter
await updater.update_parameter(
    "max_iterations",
    200,
    scope="common"
)

# Update nested parameter with dot notation
await updater.update_parameter("llm.temperature", 0.9)

# Batch updates
updates = {
    "common.max_iterations": 200,
    "common.concurrency": 10,
    "database.population_size": 500
}
await updater.update_parameters(updates)

# Rollback if needed
await updater.rollback_updates(num_updates=1)
```

### 2. Configuration File Watcher (Hot-Reload)

Monitor and automatically reload configuration files:

```python
from openevolve.config.config_watcher import ConfigFileWatcher

# Create watcher
watcher = ConfigFileWatcher("config.yaml", poll_interval=1.0)

# Register callback
def on_config_reload(new_config):
    print(f"Config reloaded: {new_config}")

watcher.register_callback(on_config_reload)

# Start watching
watcher.start()

# Stop watching
watcher.stop()

# Get statistics
stats = watcher.get_stats()
print(f"Reloads: {stats['reload_count']}")
```

### 3. Dynamic Strategy Switching

Switch between evolutionary strategies mid-run:

```python
from openevolve.config.dynamic_strategy import DynamicStrategySwitcher, SystemMode

# Initialize
switcher = DynamicStrategySwitcher(SystemMode.OPENEVOLVE)

# Switch to Quality Diversity mode
target_config = UnifiedEvolutionConfig(evolution_mode="qd")

await switcher.switch_strategy(
    SystemMode.QD,
    target_config,
    preserve_state=True,
    reason="Need more diversity"
)

# Get switch history
history = switcher.get_switch_history()
```

### 4. Adaptive Configuration

Automatically tune parameters based on performance:

```python
from openevolve.config.adaptive_config import AdaptiveConfigurator

# Initialize
config = UnifiedEvolutionConfig()
adaptive = AdaptiveConfigurator(config)

# Feed performance metrics
performance = {
    "fitness": 0.85,
    "diversity": 0.3,
    "convergence_rate": 0.001,
    "improvement_rate": 0.0,
    "evaluation_time": 15.0,
    "success_rate": 0.95
}

suggestions = await adaptive.adapt_configuration(performance, iteration=100)

# Apply suggestions
await updater.update_parameters(suggestions)

# Get performance summary
summary = adaptive.get_performance_summary()
```

### 5. Resource-Aware Configuration

Auto-adjust based on available resources:

```python
from openevolve.config.resource_config import ResourceAwareConfigurator, ResourceLimits

# Initialize
resource_config = ResourceAwareConfigurator()

# Detect resources
resources = resource_config.detect_resources()
print(f"CPUs: {resources.cpu_count}")
print(f"Memory: {resources.memory_available_gb}GB")
print(f"GPUs: {resources.gpu_count}")

# Adjust configuration
base_config = UnifiedEvolutionConfig()
adjusted = resource_config.adjust_config_for_resources(
    base_config,
    limits=ResourceLimits(
        max_memory_mb=4096,
        max_cpu_cores=4.0
    )
)

# Get recommendations
recommendations = resource_config.get_resource_recommendations(adjusted)
```

### 6. Configuration Metrics

Track and analyze configuration performance:

```python
from openevolve.config.config_metrics import ConfigurationMetrics

# Initialize
metrics = ConfigurationMetrics()

# Track usage
metrics.track_config_usage(config, modified_params=["common.max_iterations"])

# Track performance
metrics.track_config_performance(
    config,
    performance=0.9,
    iteration=50
)

# Get most used parameters
most_used = metrics.get_most_used_parameters(limit=10)

# Analyze parameter trends
trends = metrics.analyze_parameter_trends("common.max_iterations")

# Get performance summary
summary = metrics.get_performance_summary()

# Export all metrics
all_metrics = metrics.export_metrics()
```

## Use Cases

### Use Case 1: Adaptive Learning Rate Adjustment

```python
# During evolution
if improvement_rate < 0.01:
    await updater.update_parameter("database.exploitation_ratio", 0.8)
    await updater.update_parameter("database.exploration_rate", 0.2)
```

### Use Case 2: Resource-Constrained Scaling

```python
# When running on limited resources
resource_config = ResourceAwareConfigurator()
resources = resource_config.detect_resources()

if resources.memory_available_gb < 4.0:
    adjusted = resource_config.adjust_config_for_resources(config)
    await updater.update_parameters(adjusted.to_dict())
```

### Use Case 3: Strategy Switch Based on Performance

```python
# If stuck in local optima, switch to QD
if is_stuck_in_local_optima():
    qd_config = UnifiedEvolutionConfig(evolution_mode="qd")
    await switcher.switch_strategy(
        SystemMode.QD,
        qd_config,
        preserve_state=True,
        reason="Escaping local optima"
    )
```

### Use Case 4: Configuration Hot-Reload in Production

```python
# Watch production config file
watcher = ConfigFileWatcher("/etc/evolution/config.yaml")

def on_reload(new_config):
    # Validate before applying
    updater.update_parameters(new_config.to_dict())

watcher.register_callback(on_reload)
watcher.start()
```

## API Reference

### RuntimeConfigUpdater

| Method | Description |
|--------|-------------|
| `update_parameter(name, value, validate=True, scope="common")` | Update single parameter |
| `update_parameters(updates, validate=True, rollback_on_error=True)` | Update multiple parameters |
| `get_update_history(limit=None, parameter_filter=None)` | Get update history |
| `rollback_to(timestamp)` | Rollback to specific timestamp |
| `rollback_updates(num_updates)` | Rollback last N updates |
| `create_snapshot(label=None)` | Create configuration snapshot |

### ConfigFileWatcher

| Method | Description |
|--------|-------------|
| `start()` | Start watching config file |
| `stop()` | Stop watching |
| `register_callback(callback)` | Register reload callback |
| `get_stats()` | Get watcher statistics |

### DynamicStrategySwitcher

| Method | Description |
|--------|-------------|
| `switch_strategy(new_strategy, config, preserve_state=True)` | Switch strategies |
| `get_switch_history()` | Get switch history |
| `get_current_strategy()` | Get current strategy |

### AdaptiveConfigurator

| Method | Description |
|--------|-------------|
| `adapt_configuration(metrics, iteration)` | Get adaptation suggestions |
| `get_performance_summary()` | Get performance summary |
| `get_adjustment_history(limit=None)` | Get adjustment history |

### ResourceAwareConfigurator

| Method | Description |
|--------|-------------|
| `detect_resources()` | Detect system resources |
| `adjust_config_for_resources(config, limits=None)` | Adjust configuration |
| `get_resource_recommendations(config)` | Get optimization recommendations |
| `monitor_resources()` | Get current resource usage |

### ConfigurationMetrics

| Method | Description |
|--------|-------------|
| `track_config_usage(config, modified_params)` | Track parameter usage |
| `track_config_performance(config, performance, iteration)` | Track performance |
| `get_parameter_usage_stats(parameter=None)` | Get usage statistics |
| `get_performance_summary()` | Get performance summary |
| `analyze_parameter_trends(parameter)` | Analyze parameter trends |
| `export_metrics()` | Export all metrics |

## Configuration Schema

All configuration parameters can be updated at runtime using dot notation:

### Common Parameters
- `common.max_iterations` - Maximum iterations
- `common.concurrency` - Parallel execution count
- `common.log_level` - Logging level

### Database Parameters
- `database.population_size` - Population size
- `database.elite_archive_size` - Archive size
- `database.exploration_rate` - Exploration probability
- `database.exploitation_ratio` - Exploitation ratio

### LLM Parameters
- `llm.default_temperature` - Sampling temperature
- `llm.default_api_base` - API endpoint
- `llm.models[i].name` - Model name
- `llm.models[i].weight` - Model weight

### Evaluator Parameters
- `evaluator.timeout` - Evaluation timeout
- `evaluator.parallel_evaluations` - Parallel count
- `evaluator.cascade_evaluation` - Enable cascade

## Performance Considerations

1. **Update Frequency**: Avoid updating parameters too frequently (recommended: every 10-50 iterations)

2. **Validation Overhead**: Set `validate=False` for batch updates to skip validation

3. **Snapshot Overhead**: Snapshots create deep copies - use sparingly

4. **Watcher Polling**: Default poll interval is 1 second - adjust based on needs

5. **Resource Detection**: Cache detected resources to avoid repeated detection

## Error Handling

```python
# Handle validation errors
try:
    success = await updater.update_parameter("invalid.param", value)
    if not success:
        # Handle failure
        pass
except Exception as e:
    # Handle exception
    pass

# Handle switch failures
success = await switcher.switch_strategy(
    SystemMode.QD,
    config,
    preserve_state=True
)

if not success:
    # Switch failed, current strategy preserved
    pass
```

## Best Practices

1. **Validate Changes**: Always validate parameter updates in production

2. **Snapshot Before Major Changes**: Create snapshots before batch updates

3. **Monitor Resource Usage**: Use resource monitoring to guide adjustments

4. **Track Performance**: Use metrics to learn optimal configurations

5. **Test Rollbacks**: Verify rollback mechanisms work correctly

6. **Use Hot-Reload Carefully**: Only in production with proper validation

## Troubleshooting

### Configuration Update Fails

```python
# Check validation result
if not success:
    errors = updater.validators.errors
    # Fix validation errors
```

### Strategy Switch Fails

```python
# Check compatibility
is_valid = switcher._validate_switch(new_strategy, new_config)
if not is_valid:
    # Fix configuration
```

### Resource Detection Fails

```python
# Install psutil for accurate detection
pip install psutil

# Or use minimal detection
resources = resource_config._minimal_resource_detection()
```

## Integration Examples

### With Evolution Loop

```python
async def evolution_loop():
    config = UnifiedEvolutionConfig()
    updater = RuntimeConfigUpdater(config)
    adaptive = AdaptiveConfigurator(config)

    for iteration in range(max_iterations):
        # Run evolution
        performance = run_evolution(config)

        # Adapt configuration
        if iteration % 10 == 0:
            metrics = measure_performance()
            suggestions = await adaptive.adapt_configuration(metrics, iteration)
            await updater.update_parameters(suggestions)

        # Hot-reload check
        if iteration % 50 == 0:
            # Check for config file changes
            pass
```

### With Monitoring

```python
import asyncio

async def monitor_and_adapt():
    resource_config = ResourceAwareConfigurator()
    resources = resource_config.detect_resources()

    while True:
        # Monitor resources
        usage = resource_config.monitor_resources()

        # Adjust if needed
        if usage["memory_percent"] > 90:
            # Reduce memory usage
            await updater.update_parameter(
                "database.population_size",
                config.database.population_size // 2
            )

        await asyncio.sleep(60)  # Check every minute
```

## Future Enhancements

1. Distributed configuration synchronization
2. Machine learning-based parameter optimization
3. Automatic A/B testing of configurations
4. Cloud resource detection and scaling
5. Configuration versioning and rollback
6. Real-time performance dashboard integration

## See Also

- [Enhanced Configuration System](./ENHANCED_CONFIGURATION.md)
- [Configuration Management](./CONFIGURATION_MANAGEMENT.md)
- [Configuration Presets](./CONFIGURATION_PRESETS.md)
