# Configuration Examples

This directory contains working configuration examples for the OpenEvolve Knowledge Engine.

## Directory Structure

```
examples/
├── README.md                 # This file
├── config/                   # Configuration file examples
│   ├── minimal.yaml          # Minimal 5-line config
│   ├── standard.yaml         # Standard production config
│   └── finance.yaml          # Finance-specific config
├── profiles/                 # Profile examples
│   └── custom_finance.yaml   # Custom finance profile
└── presets/                  # Preset examples
    ├── fast.yaml             # Fast execution preset
    ├── balanced.yaml         # Balanced preset (default)
    ├── thorough.yaml         # Thorough preset
    └── budget.yaml           # Budget-conscious preset
```

## Usage

### Using Config Files

```bash
# Use specific config file
evolve --config examples/config/standard.yaml problem="..."

# Copy to your project
cp examples/config/standard.yaml evolve.config.yaml
```

### Using Profiles

```bash
# List built-in profiles
evolve profile list

# Use profile
evolve --profile dev problem="..."

# Create custom profile from example
cp examples/profiles/custom_finance.yaml ~/.evolve/profiles/
evolve --profile custom_finance problem="..."
```

### Using Presets

```bash
# List built-in presets
evolve preset list

# Apply preset
evolve preset apply balanced -o evolve.config.yaml

# Use preset directly
evolve --preset fast problem="..."
```

## Examples

### Example 1: Minimal Configuration

```bash
# Copy minimal config
cp examples/config/minimal.yaml evolve.config.yaml

# Run evolution
evolve problem="Optimize portfolio allocation"
```

### Example 2: Finance Domain

```bash
# Use finance config
evolve --config examples/config/finance.yaml \
  problem="Optimize portfolio allocation"
```

### Example 3: Fast Execution

```bash
# Use fast preset
evolve --preset fast \
  problem="Quick optimization test"
```

### Example 4: Custom Profile

```bash
# Copy custom profile
cp examples/profiles/custom_finance.yaml ~/.evolve/profiles/

# Use profile
evolve --profile custom_finance problem="..."
```

## Documentation

For complete documentation:
- [Configuration Guide](../CONFIGURATION_GUIDE.md)
- [Configuration Reference](../CONFIGURATION_REFERENCE.md)
- [Profile Guide](../PROFILE_GUIDE.md)
- [Preset Catalog](../PRESET_CATALOG.md)
- [Configuration Examples](../CONFIGURATION_EXAMPLES.md)
- [Migration Guide](../CONFIGURATION_MIGRATION.md)

## Tips

1. **Start Simple:** Begin with `minimal.yaml` and add parameters as needed
2. **Use Presets:** Presets are tested starting points for common use cases
3. **Validate Always:** Run `evolve config validate` before using new configs
4. **Test Small:** Test with `--max-evaluations 5` before full runs
5. **Version Control:** Commit your config files to track changes
