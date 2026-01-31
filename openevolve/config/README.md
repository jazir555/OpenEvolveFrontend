# OpenEvolve Configuration System

A comprehensive, highly-configurable system supporting environment variables, configuration files, profiles, hierarchical overrides, hot-reload, and validation.

## Features

- ✅ **102+ Configurable Parameters** - Every aspect of the system can be configured
- ✅ **Multiple File Formats** - YAML, JSON, TOML support
- ✅ **Environment Variables** - All parameters accessible via `EVOLVE_*` env vars
- ✅ **Configuration Profiles** - Built-in profiles (dev, test, prod, quickstart)
- ✅ **Hierarchical Overrides** - 7-level priority system
- ✅ **Hot-Reload** - Watch files and auto-reload on changes
- ✅ **Comprehensive Validation** - Type, range, dependency, and consistency checks
- ✅ **Easy to Use** - Simple, unified interface via ConfigManager

## Quick Start

```python
from openevolve.config import ConfigManager

# Create manager
manager = ConfigManager()

# Load configuration
config = manager.load_config(
    profile='production',
    config_file='config.yaml',
    env_override=True
)

# Use configuration
print(config['max_iterations'])
```

## Installation

The configuration system is part of OpenEvolve. No additional installation required.

Optional dependencies for enhanced functionality:

```bash
# YAML support
pip install pyyaml

# TOML support
pip install tomli  # Python < 3.11
# tomllib is built-in for Python 3.11+

# File watching (optional, uses polling by default)
pip install watchdog
```

## Usage Examples

### Using Profiles

```python
# Development - fast iteration
config = manager.load_config(profile='development')

# Production - maximum quality
config = manager.load_config(profile='production')

# Testing - minimal overhead
config = manager.load_config(profile='testing')
```

### Environment Variables

```bash
# Set environment variables
export EVOLVE_MAX_ITERATIONS=100
export EVOLVE_TEMPERATURE=0.8
export EVOLVE_ENABLE_PLANNING=true
```

### Configuration Files

```yaml
# config.yaml
max_iterations: 50
population_size: 20
temperature: 0.7
enable_planning: true

llm:
  model_id: gpt-4o
  max_tokens: 2048
```

### Hot Reload

```python
def on_config_change(event):
    print(f"Config changed: {event.changes}")
    # Apply new config...

manager.enable_hot_reload('config.yaml', on_config_change)
```

## Configuration Priority

1. Runtime overrides (highest)
2. Environment variables
3. Config file
4. Profile
5. User config (~/.evolve/config.yaml)
6. Global config (/etc/evolve/config.yaml)
7. Defaults (lowest)

## Documentation

See [CONFIGURATION_SYSTEM.md](../../../docs/knowledge_engine/CONFIGURATION_SYSTEM.md) for comprehensive documentation.

## Testing

Run the test suite:

```bash
pytest tests/config/test_config_system.py -v
```

## Files

- `config_loader.py` - File loading (YAML, JSON, TOML)
- `env_parser.py` - Environment variable parsing
- `env_mappings.py` - Parameter to env var mappings (102+ params)
- `validator.py` - Configuration validation
- `profiles.py` - Pre-defined profiles
- `hierarchy.py` - Hierarchical config resolution
- `hot_reload.py` - File watching and auto-reload
- `manager.py` - Unified interface

## License

MIT
