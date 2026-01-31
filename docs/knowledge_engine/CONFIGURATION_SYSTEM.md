# OpenEvolve Configuration System

## Overview

The OpenEvolve Configuration System is a comprehensive, highly-configurable system supporting multiple configuration sources, formats, and validation. All 102+ parameters can be configured through environment variables, configuration files, profiles, or runtime overrides.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration Sources](#configuration-sources)
3. [Configuration Files](#configuration-files)
4. [Environment Variables](#environment-variables)
5. [Profiles](#profiles)
6. [Configuration Manager](#configuration-manager)
7. [Hot Reload](#hot-reload)
8. [Validation](#validation)
9. [Advanced Usage](#advanced-usage)
10. [API Reference](#api-reference)

## Quick Start

### Basic Usage

```python
from openevolve.config import ConfigManager

# Create manager
manager = ConfigManager()

# Load configuration with profile
config = manager.load_config(profile='development')

# Use configuration
print(config['max_iterations'])  # 20
```

### Using Environment Variables

```bash
# Set environment variables
export EVOLVE_MAX_ITERATIONS=100
export EVOLVE_TEMPERATURE=0.8
export EVOLVE_ENABLE_PLANNING=true

# Load config (will automatically pick up env vars)
python your_script.py
```

### Using Configuration Files

```python
# Load from file
config = manager.load_config(config_file='config.yaml')

# Or let it find config.yaml automatically
config = manager.load_config()
```

## Configuration Sources

The configuration system supports 7 levels of priority (highest to lowest):

1. **Runtime Overrides** - Direct function arguments
2. **Environment Variables** - `EVOLVE_*` environment variables
3. **Config File** - Local `config.yaml`, `config.json`, etc.
4. **Profile** - Pre-defined profiles (dev, test, prod)
5. **User Config** - `~/.evolve/config.yaml`
6. **Global Config** - `/etc/evolve/config.yaml`
7. **Defaults** - Built-in default values

Example:

```python
config = manager.load_config(
    config_file='./my_config.yaml',  # Level 3
    profile='production',              # Level 4
    env_override=True,                 # Level 2
    runtime_overrides={                # Level 1 (highest)
        'max_iterations': 200
    }
)
```

## Configuration Files

### Supported Formats

- **YAML** (.yaml, .yml) - Recommended (requires PyYAML)
- **JSON** (.json) - Built-in support
- **TOML** (.toml) - Requires tomli/tomllib

### YAML Example

```yaml
# config.yaml
max_iterations: 50
population_size: 20
temperature: 0.7
enable_planning: true

llm:
  model_id: gpt-4o
  max_tokens: 2048

pes:
  enable_memory: true
  memory_type: episodic
```

### JSON Example

```json
{
  "max_iterations": 50,
  "population_size": 20,
  "temperature": 0.7,
  "enable_planning": true,
  "llm": {
    "model_id": "gpt-4o",
    "max_tokens": 2048
  }
}
```

### Loading Configuration Files

```python
# Auto-detect format
config = manager.load_config(config_file='config.yaml')

# Specific format
config = manager.load_config(config_file='config.json')
```

## Environment Variables

### Naming Convention

All environment variables use the `EVOLVE_` prefix:

```bash
# Format: EVOLVE_PARAMETER_NAME=value
export EVOLVE_MAX_ITERATIONS=100
export EVOLVE_TEMPERATURE=0.7
export EVOLVE_ENABLE_PLANNING=true
```

### Parameter Mapping

Configuration parameters map directly to environment variables:

| Config Parameter | Environment Variable | Type |
|-----------------|---------------------|------|
| `max_iterations` | `EVOLVE_MAX_ITERATIONS` | int |
| `temperature` | `EVOLVE_TEMPERATURE` | float |
| `enable_planning` | `EVOLVE_ENABLE_PLANNING` | bool |
| `model_id` | `EVOLVE_MODEL` | str |

See [env_mappings.py](../../openevolve/config/env_mappings.py) for complete list of 102+ parameters.

### Type Conversion

```bash
# Integer
export EVOLVE_MAX_ITERATIONS=100

# Float
export EVOLVE_TEMPERATURE=0.7

# Boolean (true/false, 1/0, yes/no, on/off)
export EVOLVE_ENABLE_PLANNING=true

# String
export EVOLVE_MODEL_ID=gpt-4o

# List (comma-separated)
export EVOLVE_TAGS=dev,test,experiment
```

### Disabling Environment Override

```python
# Load without environment variables
config = manager.load_config(env_override=False)
```

## Profiles

### Built-in Profiles

#### Development Profile

Fast iteration during development:

```python
config = manager.load_config(profile='development')
```

- Low iteration counts (20)
- Verbose logging (DEBUG)
- Gauntlet disabled
- Intermediate results saved
- Single worker

#### Testing Profile

Optimized for running tests:

```python
config = manager.load_config(profile='testing')
```

- Minimal iterations (5)
- Deterministic (seed=42)
- Fast models
- No expensive features
- Minimal logging

#### Production Profile

Maximum quality for production:

```python
config = manager.load_config(profile='production')
```

- High iteration counts (100)
- Full features enabled
- Best models
- Strict validation
- Comprehensive logging
- Parallel execution

#### QuickStart Profile

Quick start for new users:

```python
config = manager.load_config(profile='quickstart')
```

- Conservative defaults
- Moderate iteration counts (30)
- Good model balance
- Essential features only
- Clear logging

### Creating Custom Profiles

```python
# Create from base profile
custom_config = manager.create_profile(
    name='my_custom',
    base='development',
    overrides={
        'max_iterations': 15,
        'custom_param': 'value'
    }
)

# Use custom profile
config = manager.load_config(profile='my_custom')
```

### Listing Profiles

```python
# List all available profiles
profiles = manager.list_profiles()
print(profiles)  # ['development', 'testing', 'production', 'quickstart', ...]

# Get profile info
info = manager.get_profile_info('development')
print(info.description)
```

## Configuration Manager

The `ConfigManager` class provides a unified interface for all configuration operations.

### Loading Configuration

```python
from openevolve.config import ConfigManager

manager = ConfigManager()

# Load with various options
config = manager.load_config(
    config_file='config.yaml',
    profile='development',
    env_override=True,
    runtime_overrides={'max_iterations': 200}
)
```

### Saving Configuration

```python
# Save to YAML
manager.save_config(config, 'output.yaml', format='yaml')

# Save to JSON
manager.save_config(config, 'output.json', format='json', pretty=True)

# Save to TOML
manager.save_config(config, 'output.toml', format='toml')
```

### Validating Configuration

```python
# Validate config
result = manager.validate_config(config)

if result.is_valid:
    print("Configuration is valid!")
else:
    print("Errors:", result.get_error_messages())
    print("Warnings:", result.get_warning_messages())
```

### Comparing Configurations

```python
# Compare two configs
config1 = manager.load_config(profile='development')
config2 = manager.load_config(profile='production')

diff = manager.compare_configs(config1, config2)
print("Differences:", diff)
```

### Merging Configurations

```python
# Merge with different strategies
merged = manager.merge_configs(
    config1,
    config2,
    strategy='override'  # 'override', 'deep', 'if_missing', 'if_present'
)
```

## Hot Reload

Watch configuration files for changes and automatically reload.

### Basic Usage

```python
from openevolve.config import ConfigManager

manager = ConfigManager()

def on_config_change(event):
    print(f"Config changed: {event.changes}")
    # Apply new configuration...

# Enable hot-reload
manager.enable_hot_reload(
    config_file='config.yaml',
    callback=on_config_change,
    poll_interval=1.0  # Check every second
)

# ... application runs ...

# Disable when done
manager.disable_hot_reload()
```

### Change Event

```python
def on_config_change(event):
    # Event properties:
    print(f"File: {event.filepath}")
    print(f"Timestamp: {event.timestamp}")
    print(f"Changes: {event.changes}")

    # Changes dict format:
    # {
    #   'param_name': {
    #     'old': old_value,
    #     'new': new_value
    #   }
    # }
```

### Multi-File Watching

```python
from openevolve.config.hot_reload import MultiFileHotReload

def on_change(filepath, event):
    print(f"{filepath} changed: {event.changes}")

watcher = MultiFileHotReload(
    config_files=['config.yaml', 'secrets.yaml'],
    callback=on_change
)

watcher.start()
```

## Validation

### Automatic Validation

```python
# Validation is enabled by default
config = manager.load_config(profile='production')
```

### Manual Validation

```python
from openevolve.config import ConfigValidator

validator = ConfigValidator()
result = validator.validate(config)

if not result.is_valid:
    for error in result.errors:
        print(f"{error.parameter}: {error.message}")
        if error.suggestion:
            print(f"  Suggestion: {error.suggestion}")
```

### Validation Rules

The validator checks:

1. **Type constraints** - Parameters must be correct type
2. **Range constraints** - Numeric values in valid ranges
3. **Choice constraints** - Values must be from allowed set
4. **Dependencies** - Required parameters when features enabled
5. **Logical consistency** - Parameter relationships

Example dependency:

```python
# If enable_memory=True, memory_type is required
config = {
    'enable_memory': True
    # Missing: 'memory_type'
}

result = validator.validate(config)
# Error: memory_type is required when enable_memory=True
```

### Strict vs Lenient Validation

```python
from openevolve.config.validator import StrictConfigValidator, LenientConfigValidator

# Strict: rejects unknown parameters
strict = StrictConfigValidator()
result = strict.validate(config)

# Lenient: warnings only, never errors
lenient = LenientConfigValidator()
result = lenient.validate(config)
# Always returns is_valid=True
```

## Advanced Usage

### Exporting Environment Variables

```python
# Export config as shell script
script = manager.export_env_vars(config, output_file='config.env')

# Output:
# # OpenEvolve Configuration Environment Variables
#
# export EVOLVE_MAX_ITERATIONS=50
# export EVOLVE_TEMPERATURE=0.7
# export EVOLVE_ENABLE_PLANNING=True
```

### Getting Parameter Information

```python
# Get info about a parameter
info = manager.get_parameter_info('max_iterations')
print(info)
# {
#   'name': 'max_iterations',
#   'env_var': 'EVOLVE_MAX_ITERATIONS',
#   'type': 'int',
#   'range': [1, 10000]
# }

# List all parameters
params = manager.list_all_parameters()
print(f"Total parameters: {len(params)}")  # 102+
```

### Finding Configuration Files

```python
# Find highest-priority config file
config_file = manager.find_config_file()
print(f"Using: {config_file}")

# List all config sources
sources = manager.get_config_sources()
for source in sources:
    print(f"{source['name']}: {source['path']} (exists: {source['exists']})")
```

### Deep Merging

```python
from openevolve.config.hierarchy import ConfigMerge

# Deep merge nested dicts
base = {
    'llm': {'model_id': 'gpt-4o-mini', 'temperature': 0.7}
}
override = {
    'llm': {'temperature': 0.9}
}

merged = ConfigMerge.deep_merge(base, override)
# Result: {'llm': {'model_id': 'gpt-4o-mini', 'temperature': 0.9}}
```

### Configuration Snapshots

```python
from openevolve.config.hierarchy import ConfigSnapshot

# Create snapshot
snapshot = ConfigSnapshot(config)

# Later, restore or compare
restored = snapshot.restore()
diff = snapshot.diff(current_config)
```

## API Reference

### ConfigManager

```python
class ConfigManager:
    """Main configuration manager"""

    def load_config(
        self,
        config_file: Optional[str] = None,
        profile: Optional[str] = None,
        env_override: bool = True,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """Load configuration from multiple sources"""

    def save_config(
        self,
        config: Dict[str, Any],
        filepath: str,
        format: str = 'yaml'
    ) -> None:
        """Save configuration to file"""

    def validate_config(
        self,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate configuration"""

    def enable_hot_reload(
        self,
        config_file: str,
        callback: Callable[[ConfigChangeEvent], None]
    ) -> None:
        """Enable hot-reload for config file"""

    def list_profiles(self) -> List[str]:
        """List all available profiles"""

    def create_profile(
        self,
        name: str,
        base: str = 'quickstart',
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new profile"""
```

### ConfigLoader

```python
class ConfigLoader:
    """Load configuration from files"""

    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        """Load YAML file"""

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load JSON file"""

    def load_toml(self, filepath: str) -> Dict[str, Any]:
        """Load TOML file"""

    def load_auto(self, filepath: str) -> Dict[str, Any]:
        """Auto-detect format and load"""

    def save_yaml(self, config: Dict, filepath: str) -> None:
        """Save to YAML"""

    def save_json(self, config: Dict, filepath: str, pretty: bool = True) -> None:
        """Save to JSON"""
```

### EnvConfigParser

```python
class EnvConfigParser:
    """Parse environment variables"""

    ENV_PREFIX = "EVOLVE_"

    def parse_env(self) -> Dict[str, Any]:
        """Parse all EVOLVE_ environment variables"""

    def get_env_value(
        self,
        param_name: str,
        default: Any = None
    ) -> Any:
        """Get single environment variable value"""

    @staticmethod
    def param_to_env_name(param_name: str) -> str:
        """Convert param name to env var name"""
```

### ConfigValidator

```python
class ConfigValidator:
    """Validate configuration"""

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate complete configuration"""

    def validate_parameter(
        self,
        name: str,
        value: Any
    ) -> Tuple[bool, Optional[str]]:
        """Validate single parameter"""

    def check_dependencies(
        self,
        config: Dict[str, Any]
    ) -> List[ValidationError]:
        """Check parameter dependencies"""

    def suggest_fixes(
        self,
        errors: List[ValidationError]
    ) -> List[str]:
        """Generate suggestions for fixing errors"""
```

### ProfileManager

```python
class ProfileManager:
    """Manage configuration profiles"""

    PROFILES = {
        'development': DevelopmentProfile,
        'testing': TestingProfile,
        'production': ProductionProfile,
        'quickstart': QuickStartProfile,
    }

    def load_profile(self, profile_name: str) -> Dict[str, Any]:
        """Load profile by name"""

    def save_profile(
        self,
        profile_name: str,
        parameters: Dict[str, Any]
    ) -> None:
        """Save custom profile"""

    def create_profile(
        self,
        name: str,
        base: str = 'quickstart',
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create new profile from base"""

    def delete_profile(self, name: str) -> None:
        """Delete custom profile"""

    def list_profiles(self) -> List[str]:
        """List all profiles"""
```

### ConfigHotReload

```python
class ConfigHotReload:
    """Watch config file for changes"""

    def __init__(
        self,
        config_file: str,
        callback: Callable[[ConfigChangeEvent], None],
        poll_interval: float = 1.0
    ):
        """Initialize hot-reload watcher"""

    def start(self) -> None:
        """Start watching file"""

    def stop(self) -> None:
        """Stop watching file"""

    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get current configuration"""

    def get_stats(self) -> Dict[str, Any]:
        """Get hot-reload statistics"""
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Complete OpenEvolve configuration example
"""

from openevolve.config import ConfigManager

def main():
    # Create configuration manager
    manager = ConfigManager()

    # Load configuration with multiple sources
    print("Loading configuration...")
    config = manager.load_config(
        profile='production',              # Start with production profile
        config_file='./config.yaml',       # Override with local file
        env_override=True,                 # Apply environment variables
        runtime_overrides={                # Final runtime overrides
            'max_iterations': 150,
            'experiment_name': 'my_experiment'
        }
    )

    # Print configuration
    print(f"\nConfiguration loaded with {len(config)} parameters")
    print(f"Max iterations: {config['max_iterations']}")
    print(f"Temperature: {config['temperature']}")
    print(f"Planning enabled: {config['enable_planning']}")

    # Validate configuration
    print("\nValidating configuration...")
    result = manager.validate_config(config)

    if not result.is_valid:
        print("Validation errors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    if result.warnings:
        print("Validation warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print("Configuration is valid!")

    # Enable hot-reload (optional)
    def on_config_change(event):
        print(f"\nConfiguration changed!")
        print(f"Modified parameters: {len(event.changes)}")
        for param, change in event.changes.items():
            print(f"  {param}: {change['old']} -> {change['new']}")

    print("\nEnabling hot-reload...")
    manager.enable_hot_reload(
        config_file='./config.yaml',
        callback=on_config_change
    )

    # Save configuration for reproducibility
    print("\nSaving configuration...")
    manager.save_config(config, 'experiment_config.json', format='json')

    # Export environment variables
    env_script = manager.export_env_vars(config, 'experiment.env')
    print("Exported environment variables to experiment.env")

    # Main application logic here...
    print("\nApplication running...")
    print("Press Ctrl+C to stop")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")

    # Cleanup
    manager.disable_hot_reload()
    print("Done!")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
```

## Best Practices

1. **Use Profiles** - Start with built-in profiles, then customize
2. **Environment Variables** - Use for sensitive data (API keys) and deployment-specific settings
3. **Validation** - Always validate configuration before using
4. **Documentation** - Document custom profiles and important parameters
5. **Version Control** - Keep configuration files in version control (except secrets)
6. **Hot Reload** - Use in development for quick iteration
7. **Defaults** - Rely on defaults, only override what you need

## Troubleshooting

### Configuration Not Loading

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check what sources are being used
sources = manager.get_config_sources()
for source in sources:
    print(f"{source['name']}: {source['exists']}")
```

### Validation Errors

```python
# Get detailed error information
result = manager.validate_config(config)

for error in result.errors:
    print(f"Parameter: {error.parameter}")
    print(f"Value: {error.value}")
    print(f"Error: {error.message}")
    print(f"Suggestion: {error.suggestion}")
```

### Hot Reload Not Working

```python
# Check hot-reload stats
stats = manager.hot_reload.get_stats()
print(stats)
# {'running': True, 'reload_count': 5, 'error_count': 0}
```

## Further Reading

- [Environment Variable Mappings](../../openevolve/config/env_mappings.py) - Complete parameter list
- [Configuration Templates](../../openevolve/config/templates/) - Example configs
- [Test Suite](../../tests/config/test_config_system.py) - Usage examples

## License

MIT License - See LICENSE file for details
