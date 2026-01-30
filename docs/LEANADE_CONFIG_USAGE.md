# LeanAide Configuration Module - Usage Guide

## Quick Start

### Basic Usage

```python
from leanaide_config import load_leanaide_config, get_leanaide_config

# Load configuration (uses defaults)
config = load_leanaide_config()

# Access configuration
print(f"Server URL: {config.server.get_base_url()}")
print(f"Auto-verification: {config.verification.enable_auto}")
print(f"Cache enabled: {config.cache.enable}")
```

### Loading with Custom Settings

```python
# Override via Python API
config = load_leanaide_config(
    server__host="leanaide.example.com",
    server__port=9090,
    verification__complexity_threshold=75,
    cache__ttl=3600
)
```

### Using Environment Variables

```bash
# Set environment variables before running
export LEANAIDE_SERVER_HOST="leanaide.example.com"
export LEANAIDE_SERVER_PORT=9090
export LEANAIDE_VERIFICATION_ENABLE_AUTO=true
export LEANAIDE_CACHE_TTL=3600

python your_script.py
```

### Using YAML Configuration File

Create `leanaide_config.yaml`:

```yaml
server:
  host: leanaide.example.com
  port: 9090

verification:
  enable_auto: true
  complexity_threshold: 75

cache:
  enable: true
  ttl: 3600
```

Then in your code:

```python
from leanaide_config import load_leanaide_config

config = load_leanaide_config()
```

## Configuration Sections

### 1. Server Configuration

```python
config.server.host              # Server hostname
config.server.port              # Server port
config.server.timeout           # Request timeout
config.server.use_ssl           # Enable SSL
config.server.get_base_url()    # Get full base URL
```

### 2. Verification Configuration

```python
config.verification.enable_auto              # Auto-verification enabled
config.verification.complexity_threshold     # Complexity threshold (0-100)
config.verification.domains                  # Lean 4 domains
config.verification.parallel_verifications   # Parallel threads
config.verification.verification_strategy    # "quick", "thorough", or "adaptive"
```

### 3. Cache Configuration

```python
config.cache.enable                    # Caching enabled
config.cache.ttl                       # Time-to-live in seconds
config.cache.cache_dir                 # Cache directory
config.cache.max_cache_size_mb         # Max cache size
config.cache.persistent_cache          # Persist across restarts
```

### 4. Workflow Integration

```python
config.workflow.stage_3c_enabled       # Stage 3C verification
config.workflow.stage_5_enabled        # Stage 5 verification
config.workflow.async_verification     # Run async
config.workflow.failure_action         # "warn", "error", "continue", "fallback"
```

### 5. Lean 4 Environment

```python
config.lean4.lean_path         # Path to Lean executable
config.lean4.lake_path         # Path to Lake build tool
config.lean4.mathlib_path      # Path to MathLib
config.lean4.project_root      # Project root directory
```

### 6. Logging Configuration

```python
config.logging.level                    # Log level
config.logging.log_file                 # Log file path
config.logging.log_verification_details # Verbose logging
```

### 7. Security Configuration

```python
config.security.enable_sandboxing       # Sandboxing enabled
config.security.max_memory_mb           # Memory limit
config.security.trusted_domains         # Trusted import domains
```

### 8. Performance Configuration

```python
config.performance.worker_threads        # Number of workers
config.performance.optimization_level    # 0-3
config.performance.preload_mathlib       # Preload MathLib
```

## Configuration Precedence

Settings are loaded in this order (later overrides earlier):

1. **Python API overrides** (highest priority)
2. **Environment variables** (LEANAIDE_*)
3. **YAML files** (leanaide_config.yaml, config.yaml)
4. **Default values** (lowest priority)

## Examples

### Development Configuration

```python
config = load_leanaide_config(
    environment="development",
    server__host="localhost",
    server__port=8080,
    verification__strict_mode=False,
    logging__level="DEBUG"
)
```

### Production Configuration

```python
config = load_leanaide_config(
    environment="production",
    server__host="leanaide.prod.example.com",
    server__use_ssl=True,
    verification__strict_mode=True,
    logging__level="WARNING",
    security__enable_sandboxing=True
)
```

### Workflow Integration

```python
config = load_leanaide_config(
    workflow__stage_3c_enabled=True,
    workflow__stage_5_enabled=True,
    workflow__async_verification=True,
    workflow__failure_action="error",
    workflow__inject_proof_hints=True
)
```

### Custom Lean 4 Paths

```python
config = load_leanaide_config(
    lean4__lean_path="/usr/local/bin/lean",
    lean4__lake_path="/usr/local/bin/lake",
    lean4__mathlib_path="/opt/lean/mathlib",
    lean4__project_root="/opt/lean/projects"
)
```

## Validation

Configuration is automatically validated on load. If invalid, a `ValidationError` is raised:

```python
from leanaide_config import load_leanaide_config, ValidationError

try:
    config = load_leanaide_config(server__port=99999)
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Global Configuration Instance

```python
from leanaide_config import load_leanaide_config, get_leanaide_config, reload_leanaide_config

# Load configuration
config = load_leanaide_config()

# Get current configuration (same instance)
config = get_leanaide_config()

# Force reload
config = reload_leanaide_config()

# Get configuration summary (safe for logging)
summary = get_leanaide_config_summary()
print(summary)
```

## Environment Variables

All settings can be overridden via environment variables using these patterns:

- `LEANAIDE_<SECTION>_<KEY>` for nested settings
- `LEANAIDE_<KEY>` for top-level settings

Examples:

```bash
# Server settings
export LEANAIDE_SERVER_HOST="leanaide.example.com"
export LEANAIDE_SERVER_PORT=9090
export LEANAIDE_SERVER_USE_SSL=true

# Verification settings
export LEANAIDE_VERIFICATION_ENABLE_AUTO=true
export LEANAIDE_VERIFICATION_COMPLEXITY_THRESHOLD=75
export LEANAIDE_VERIFICATION_DOMAINS="mathlib,std,analysis"

# Cache settings
export LEANAIDE_CACHE_ENABLE=true
export LEANAIDE_CACHE_TTL=3600

# Workflow settings
export LEANAIDE_WORKFLOW_STAGE_3C_ENABLED=true
export LEANAIDE_WORKFLOW_FAILURE_ACTION=error

# Global settings
export LEANAIDE_ENABLED=true
export LEANAIDE_ENVIRONMENT=production
```

## Troubleshooting

### Configuration Not Loading

```python
# Enable debug logging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG)

config = load_leanaide_config()
```

### Check Current Configuration

```python
from leanaide_config import get_leanaide_config_summary

summary = get_leanaide_config_summary()
import json
print(json.dumps(summary, indent=2))
```

### Validate Configuration

```python
from leanaide_config import load_leanaide_config

config = load_leanaide_config()
errors = config.validate()
if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

## Migration

The configuration module supports automatic migration from older versions. If you have a configuration file from an older version, it will be automatically migrated to the current format.

## See Also

- `leanaide_config.example.yaml` - Example configuration file
- `leanaide_config.py` - Module with full documentation
- `test_leanaide_config.py` - Comprehensive test suite
