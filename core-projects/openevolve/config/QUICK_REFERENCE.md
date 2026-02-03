# Configuration System Quick Reference

## Import

```python
from openevolve.config import ConfigManager
```

## Basic Usage

```python
# Create manager
manager = ConfigManager()

# Load configuration
config = manager.load_config(
    profile='production',
    config_file='config.yaml',
    env_override=True,
    runtime_overrides={'max_iterations': 200}
)
```

## Profiles

```python
# Load profile
config = manager.load_config(profile='development')

# List profiles
profiles = manager.list_profiles()
# ['development', 'testing', 'production', 'benchmarking', 'quickstart']

# Create custom profile
custom = manager.create_profile(
    name='my_profile',
    base='development',
    overrides={'max_iterations': 15}
)
```

## Environment Variables

```bash
# Format: EVOLVE_PARAMETER_NAME=value
export EVOLVE_MAX_ITERATIONS=100
export EVOLVE_TEMPERATURE=0.8
export EVOLVE_ENABLE_PLANNING=true
export EVOLVE_MODEL_ID=gpt-4o
```

## Configuration Files

### YAML
```yaml
max_iterations: 50
temperature: 0.7
enable_planning: true
```

### JSON
```json
{
  "max_iterations": 50,
  "temperature": 0.7,
  "enable_planning": true
}
```

### Loading
```python
# Auto-detect format
config = manager.load_config(config_file='config.yaml')

# Specific format
config = manager.load_config(config_file='config.json')
```

## Saving Configuration

```python
# Save to YAML
manager.save_config(config, 'output.yaml', format='yaml')

# Save to JSON
manager.save_config(config, 'output.json', format='json')

# Save to TOML
manager.save_config(config, 'output.toml', format='toml')
```

## Validation

```python
# Validate config
result = manager.validate_config(config)

if not result.is_valid:
    print("Errors:", result.get_error_messages())
    print("Warnings:", result.get_warning_messages())
```

## Hot-Reload

```python
# Enable hot-reload
def on_change(event):
    print(f"Config changed: {event.changes}")

manager.enable_hot_reload('config.yaml', on_change)

# Disable hot-reload
manager.disable_hot_reload()
```

## Parameter Information

```python
# Get parameter info
info = manager.get_parameter_info('max_iterations')
# {'name': 'max_iterations', 'env_var': 'EVOLVE_MAX_ITERATIONS', ...}

# List all parameters
params = manager.list_all_parameters()

# Get env var name for parameter
env_var = manager.get_env_var_for_param('max_iterations')
# 'EVOLVE_MAX_ITERATIONS'
```

## Comparing Configurations

```python
# Compare two configs
diff = manager.compare_configs(config1, config2)
# {'only_in_first': {...}, 'only_in_second': {...}, 'different_values': {...}}
```

## Merging Configurations

```python
# Merge with different strategies
merged = manager.merge_configs(
    config1,
    config2,
    strategy='override'  # 'override', 'deep', 'if_missing', 'if_present'
)
```

## Configuration Priority

1. **Runtime overrides** (highest)
2. **Environment variables**
3. **Config file**
4. **Profile**
5. **User config** (~/.evolve/config.yaml)
6. **Global config** (/etc/evolve/config.yaml)
7. **Defaults** (lowest)

## Common Parameters

| Parameter | Environment Variable | Type | Default |
|-----------|---------------------|------|---------|
| `max_iterations` | `EVOLVE_MAX_ITERATIONS` | int | 50 |
| `population_size` | `EVOLVE_POPULATION_SIZE` | int | 20 |
| `temperature` | `EVOLVE_TEMPERATURE` | float | 0.7 |
| `enable_planning` | `EVOLVE_ENABLE_PLANNING` | bool | true |
| `enable_memory` | `EVOLVE_ENABLE_MEMORY` | bool | false |
| `qd_enabled` | `EVOLVE_QD_ENABLED` | bool | false |
| `enable_gauntlet` | `EVOLVE_ENABLE_GAUNTLET` | bool | true |
| `log_level` | `EVOLVE_LOG_LEVEL` | str | INFO |

## Profile Characteristics

### Development
- Fast iteration (20 iterations)
- DEBUG logging
- Gauntlet disabled
- Single worker

### Testing
- Minimal (5 iterations)
- Deterministic (seed=42)
- No expensive features

### Production
- Maximum quality (100 iterations)
- All features enabled
- Strict validation
- Parallel execution

### Benchmarking
- High iterations (200)
- Max parallel workers
- Reproducible (seed=42)

### QuickStart
- Balanced (30 iterations)
- Beginner-friendly
- Essential features

## Error Handling

```python
from openevolve.config import ConfigManager, ConfigValidationError

manager = ConfigManager()

try:
    config = manager.load_config(profile='production')
except ConfigValidationError as e:
    print(f"Validation failed: {e.errors}")
    for suggestion in e.suggestions:
        print(f"Suggestion: {suggestion}")
```

## Export Environment Variables

```python
# Export config as shell script
script = manager.export_env_vars(config, 'config.env')

# Output:
# export EVOLVE_MAX_ITERATIONS=50
# export EVOLVE_TEMPERATURE=0.7
```

## Finding Configuration Files

```python
# Find highest-priority config file
config_file = manager.find_config_file()

# List all config sources
sources = manager.get_config_sources()
for source in sources:
    print(f"{source['name']}: {source['path']} (exists: {source['exists']})")
```

## Complete Example

```python
from openevolve.config import ConfigManager

def main():
    # Create manager
    manager = ConfigManager()

    # Load configuration
    config = manager.load_config(
        profile='production',
        config_file='./config.yaml',
        env_override=True,
        runtime_overrides={'experiment_name': 'my_experiment'}
    )

    # Validate
    result = manager.validate_config(config)
    if not result.is_valid:
        print(f"Errors: {result.get_error_messages()}")
        return 1

    # Enable hot-reload
    def on_change(event):
        print(f"Config changed: {len(event.changes)} parameters")
        for param, change in event.changes.items():
            print(f"  {param}: {change['old']} -> {change['new']}")

    manager.enable_hot_reload('./config.yaml', on_change)

    # Use configuration
    print(f"Max iterations: {config['max_iterations']}")
    print(f"Temperature: {config['temperature']}")

    # Save for reproducibility
    manager.save_config(config, 'experiment_config.json')

    # Export environment variables
    manager.export_env_vars(config, 'experiment.env')

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
```

## Help

For detailed documentation, see:
- **CONFIGURATION_SYSTEM.md** - Complete reference
- **openevolve/config/README.md** - Quick start
- **openevolve/config/examples.py** - Usage examples

For API reference, see:
- ConfigManager class
- ConfigLoader class
- EnvConfigParser class
- ConfigValidator class
- ProfileManager class
