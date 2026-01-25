# UnifiedConfiguration Integration Quick Reference

## Overview

The `UnifiedConfiguration` class serves as the single source of truth for all 272 OpenEvolve parameters. Both `EvolutionConfiguration` and `AdversarialConfiguration` now support bidirectional conversion with `UnifiedConfiguration`.

## Quick Start

### Creating Configurations

```python
# Method 1: Create module-specific config directly
from evolution import EvolutionConfiguration
config = EvolutionConfiguration()
config.max_iterations = 20

# Method 2: Create from UnifiedConfiguration
from unified_configuration import create_unified_config
unified = create_unified_config({'max_iterations': 20, 'temperature': 0.8})
config = EvolutionConfiguration.from_unified_config(unified)
```

### Converting Between Modules

```python
# Evolution → Adversarial
evo_config = EvolutionConfiguration()
evo_config.max_iterations = 15

# Step 1: Convert to UnifiedConfiguration
unified = evo_config.to_unified_config()

# Step 2: Convert to AdversarialConfiguration
from adversarial import AdversarialConfiguration
adv_config = AdversarialConfiguration.from_unified_config(unified)

# Now adv_config has the same parameters!
print(adv_config.max_iterations)  # 15
```

## API Reference

### EvolutionConfiguration Methods

#### `to_unified_config()`
Converts EvolutionConfiguration to UnifiedConfiguration.

```python
config = EvolutionConfiguration()
config.temperature = 0.9
unified = config.to_unified_config()
print(unified.temperature)  # 0.9
```

#### `to_dict()`
Exports configuration as dictionary (all 272 parameters).

```python
config = EvolutionConfiguration()
config_dict = config.to_dict()
print(len(config_dict))  # 272
```

#### `get_summary()`
Returns human-readable summary of key parameters.

```python
config = EvolutionConfiguration()
summary = config.get_summary()
# Returns: {
#   'evolution_mode': 'standard',
#   'max_iterations': 10,
#   'temperature': 0.7,
#   'key_params_count': 272,
#   'advanced_features': [...]
# }
```

#### `validate_with_manager(param_manager=None)`
Validates configuration using ParameterManager.

```python
config = EvolutionConfiguration()
result = config.validate_with_manager()
if not result.valid:
    print(f"Errors: {result.errors}")
```

### AdversarialConfiguration Methods

#### `to_unified_config()`
Converts AdversarialConfiguration to UnifiedConfiguration.

```python
config = AdversarialConfiguration()
config.attack_strength = 0.9
unified = config.to_unified_config()
print(unified.attack_strength)  # 0.9
```

#### `to_dict()`
Exports configuration as dictionary.

```python
config = AdversarialConfiguration()
config_dict = config.to_dict()
```

#### `get_summary()`
Returns human-readable summary of adversarial parameters.

```python
config = AdversarialConfiguration()
summary = config.get_summary()
# Returns: {
#   'adversarial_rounds': 5,
#   'attack_strength': 0.5,
#   'defense_strategy': 'reactive',
#   'advanced_features': [...]
# }
```

#### `validate_with_manager(param_manager=None)`
Validates configuration using ParameterManager.

```python
config = AdversarialConfiguration()
result = config.validate_with_manager()
```

### UnifiedConfiguration Methods

#### `to_evolution_config()`
Converts to EvolutionConfiguration.

```python
from unified_configuration import UnifiedConfiguration
unified = UnifiedConfiguration({'max_iterations': 20})
config = unified.to_evolution_config()
```

#### `to_adversarial_config()`
Converts to AdversarialConfiguration.

```python
unified = UnifiedConfiguration({'attack_strength': 0.8})
config = unified.to_adversarial_config()
```

#### Convenience Properties
```python
unified = UnifiedConfiguration({...})

# Evolution parameters
unified.evolution_mode    # 'standard'
unified.max_iterations    # 10
unified.temperature       # 0.7
unified.population_size   # 20

# Adversarial parameters
unified.adversarial_rounds  # 5
unified.attack_strength     # 0.5
unified.defense_strategy    # 'reactive'

# Model parameters
unified.api_key          # ''
unified.api_base         # 'https://api.openai.com/v1'
unified.model_id         # 'gpt-4'
unified.max_tokens       # 2048
```

## Common Patterns

### Pattern 1: Save and Load Configuration

```python
import json
from evolution import EvolutionConfiguration
from unified_configuration import UnifiedConfiguration

# Save
config = EvolutionConfiguration()
config.max_iterations = 25
with open('config.json', 'w') as f:
    json.dump(config.to_dict(), f)

# Load
with open('config.json') as f:
    data = json.load(f)
unified = UnifiedConfiguration(data, validate=False)
config = EvolutionConfiguration.from_unified_config(unified)
```

### Pattern 2: Merge Configurations

```python
from unified_configuration import UnifiedConfiguration

# Base config
base = UnifiedConfiguration({'max_iterations': 10, 'temperature': 0.7})

# Override config
overrides = {'temperature': 0.9, 'population_size': 50}

# Merge (overrides take precedence)
merged = base.merge(overrides, validate=False)
print(merged.temperature)        # 0.9 (from overrides)
print(merged.max_iterations)      # 10 (from base)
print(merged.population_size)     # 50 (from overrides)
```

### Pattern 3: Validate Before Use

```python
from evolution import EvolutionConfiguration

config = EvolutionConfiguration()
config.max_iterations = 1000  # Unusually high

# Validate
result = config.validate_with_manager()
if not result.valid:
    print(f"Configuration has {len(result.errors)} errors:")
    for error in result.errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

### Pattern 4: Inspect Configuration

```python
from evolution import EvolutionConfiguration

config = EvolutionConfiguration()
config.meta_learning = True
config.archive_size = 200

# Get summary
summary = config.get_summary()
print(json.dumps(summary, indent=2))

# Output:
# {
#   "evolution_mode": "standard",
#   "max_iterations": 10,
#   "population_size": 20,
#   "temperature": 0.7,
#   "max_tokens": 2048,
#   "model_id": "gpt-4",
#   "key_params_count": 272,
#   "archive_size": 200,
#   "advanced_features": ["meta_learning", "quality_diversity"]
# }
```

### Pattern 5: Dynamic Parameter Access

```python
from unified_configuration import UnifiedConfiguration

unified = UnifiedConfiguration({...})

# Access any parameter by name
value = unified.get('max_iterations', default=10)

# Set parameter
unified.set('max_iterations', 20, validate=True)

# Update multiple parameters
unified.update({
    'max_iterations': 20,
    'temperature': 0.8
}, validate=True)

# Check if parameter exists
if 'max_iterations' in unified:
    print(f"max_iterations = {unified['max_iterations']}")
```

## Category-Specific Parameters

### Get Parameters by Category

```python
from unified_configuration import UnifiedConfiguration

unified = UnifiedConfiguration({...})

# Get all core evolution parameters
evo_params = unified.get_category_params('core_evolution')

# Get all adversarial parameters
adv_params = unified.get_category_params('adversarial')

# Get all model config parameters
model_params = unified.get_category_params('model_config')
```

## Preset Configurations

### Standard Evolution Preset

```python
from unified_configuration import create_standard_evolution_config

config = create_standard_evolution_config(
    max_iterations=20,
    population_size=50,
    temperature=0.8
)
```

### Adversarial Testing Preset

```python
from unified_configuration import create_adversarial_testing_config

config = create_adversarial_testing_config(
    adversarial_rounds=10,
    attack_strength=0.9,
    defense_strategy='proactive'
)
```

### Quality Diversity Preset

```python
from unified_configuration import create_quality_diversity_config

config = create_quality_diversity_config(
    archive_size=200,
    feature_bins=15,
    diversity_weight=0.7
)
```

## Best Practices

### 1. Always Use UnifiedConfiguration for Cross-Module Work

```python
# GOOD
evo_config = EvolutionConfiguration()
unified = evo_config.to_unified_config()
adv_config = unified.to_adversarial_config()

# AVOID: Direct conversion between module configs
# (Not supported, loses information)
```

### 2. Validate Configuration Before Critical Operations

```python
config = EvolutionConfiguration()
result = config.validate_with_manager()
if not result.valid:
    # Handle errors
    pass
```

### 3. Use Summaries for Logging and Debugging

```python
config = EvolutionConfiguration()
logger.info(f"Configuration: {config.get_summary()}")
```

### 4. Export to Dict for Persistence

```python
# Save
config_dict = config.to_dict()
with open('config.json', 'w') as f:
    json.dump(config_dict, f)

# Load
with open('config.json') as f:
    unified = UnifiedConfiguration(json.load(f), validate=False)
    config = EvolutionConfiguration.from_unified_config(unified)
```

## Troubleshooting

### Issue: Parameter Not Preserved During Conversion

**Problem:** Parameter value lost when converting between configs.

**Solution:** Ensure parameter exists in both config classes. All 272 parameters are defined in ParameterManager schema.

```python
# Check if parameter exists in unified config
if 'my_param' in unified:
    print(f"Parameter exists: {unified['my_param']}")
```

### Issue: Validation Fails

**Problem:** Configuration validation returns errors.

**Solution:** Check validation result for specific errors.

```python
result = config.validate_with_manager()
if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
```

### Issue: Type Mismatch

**Problem:** Parameter has wrong type after conversion.

**Solution:** UnifiedConfiguration applies type conversion based on ParameterManager schema. Check schema for expected types.

```python
from parameter_manager import ParameterManager

manager = ParameterManager()
param_def = manager.schema.parameters.get('max_iterations')
print(f"Expected type: {param_def.param_type}")
```

## Complete Example

```python
from evolution import EvolutionConfiguration
from adversarial import AdversarialConfiguration
from unified_configuration import UnifiedConfiguration
import json

# 1. Create evolution config
evo_config = EvolutionConfiguration()
evo_config.max_iterations = 20
evo_config.temperature = 0.8
evo_config.meta_learning = True

# 2. Get summary
print("Evolution Config Summary:")
print(json.dumps(evo_config.get_summary(), indent=2))

# 3. Convert to UnifiedConfiguration
unified = evo_config.to_unified_config()
print(f"\nUnifiedConfiguration: {len(unified)} parameters")

# 4. Convert to AdversarialConfiguration
adv_config = unified.to_adversarial_config()
print(f"\nAdversarial Config: {adv_config.max_iterations} iterations")

# 5. Validate
validation = adv_config.validate_with_manager()
if validation.valid:
    print("\nConfiguration is valid!")
else:
    print(f"\nValidation errors: {validation.errors}")

# 6. Export to JSON
config_dict = adv_config.to_dict()
with open('adversarial_config.json', 'w') as f:
    json.dump(config_dict, f, indent=2)
print("\nConfiguration saved to adversarial_config.json")
```

## Reference

- **unified_configuration.py** - Main UnifiedConfiguration class
- **evolution.py** - EvolutionConfiguration class
- **adversarial.py** - AdversarialConfiguration class
- **parameter_manager.py** - ParameterManager and schema definitions
- **BATCH_3B_UNIFIED_CONFIG_INTEGRATION_COMPLETE.md** - Complete implementation details
