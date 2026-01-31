# LoongFlow Integration with Graceful Fallback

OpenEvolve provides seamless integration with LoongFlow's PES (Plan-Execute-Summarize) system, with automatic graceful fallback to OpenEvolve-native capabilities when LoongFlow is not available.

## Overview

The integration ensures that OpenEvolve works perfectly whether LoongFlow is installed or not. This provides several benefits:

- **Zero Breaking Changes**: Existing code continues to work without modification
- **Transparent Fallback**: Users don't need to worry about LoongFlow availability
- **Full Functionality**: All features remain available in OpenEvolve-native mode
- **Clear Communication**: User-friendly messages explain what's happening

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LoongFlowAdapter                         │
│  (Seamless Interface - Works with or without LoongFlow)      │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐        ┌──────────────────┐
│  LoongFlow   │        │  OpenEvolve      │
│  PES System  │        │  Fallback        │
│              │        │  Adapter         │
│ (if enabled  │        │                  │
│  & available)│        │ (when LoongFlow  │
│              │        │  unavailable)    │
└──────────────┘        └──────────────────┘
```

## Configuration Options

### Basic Configuration

```python
from openevolve.integrations import LoongFlowAdapter

# Default configuration - uses LoongFlow if available
config = {
    "max_iterations": 100,
    "population_size": 20,
}

adapter = LoongFlowAdapter(config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_loongflow` | bool | True | Enable/disable LoongFlow integration |
| `require_loongflow` | bool | False | If True, fail instead of falling back |
| `show_messages` | bool | True | Show user-friendly status messages |
| `mode` | str | "standard" | OpenEvolve mode for fallback |
| `max_iterations` | int | 100 | Maximum evolution iterations |
| `population_size` | int | 20 | Population size |
| `enable_planning` | bool | True | Enable planning phase (LoongFlow) |
| `enable_memory` | bool | True | Enable memory system (LoongFlow) |
| `llm_config` | dict | {} | LLM configuration |
| `timeout` | int | 300 | Timeout in seconds |

## Usage Examples

### Example 1: Default Configuration (Automatic Fallback)

```python
import asyncio
from openevolve.integrations import LoongFlowAdapter

async def evolve_problem():
    # Create adapter - automatically uses LoongFlow if available
    config = {
        "max_iterations": 50,
        "population_size": 10
    }
    adapter = LoongFlowAdapter(config)

    # Check which system is being used
    status = adapter.get_status()
    print(f"Using: {status['capabilities']['system']}")

    # Run evolution - works seamlessly regardless
    result = await adapter.evolve(
        problem="Optimize function: f(x) = x^2",
        domain="math"
    )

    print(f"Best fitness: {result['best_fitness']}")
    print(f"System used: {result['system_used']}")

asyncio.run(evolve_problem())
```

### Example 2: OpenEvolve-Only Mode

```python
# Explicitly use OpenEvolve, disable LoongFlow
config = {
    "enable_loongflow": False,
    "mode": "qd",  # Quality-Diversity mode
    "max_iterations": 100
}

adapter = LoongFlowAdapter(config)

# System will use OpenEvolve's QD mode
result = await adapter.evolve(
    problem="Find diverse sorting algorithms",
    domain="code"
)
```

### Example 3: Strict LoongFlow Requirement

```python
# Fail if LoongFlow is not available
config = {
    "enable_loongflow": True,
    "require_loongflow": True  # Don't fall back
}

try:
    adapter = LoongFlowAdapter(config)
    # If we get here, LoongFlow is available
    result = await adapter.evolve(problem="...", domain="math")
except RuntimeError as e:
    print(f"LoongFlow required but not available: {e}")
```

### Example 4: Production Configuration

```python
config = {
    # LoongFlow settings
    "enable_loongflow": True,
    "require_loongflow": False,

    # OpenEvolve fallback settings
    "mode": "standard",
    "max_iterations": 100,
    "population_size": 20,

    # Features
    "enable_planning": True,
    "enable_memory": True,

    # User experience
    "show_messages": True,

    # LLM configuration
    "llm_config": {
        "model": "gpt-4",
        "temperature": 0.7
    }
}

adapter = LoongFlowAdapter(config)
```

## OpenEvolve Modes

When using OpenEvolve fallback, you can choose from several evolution modes:

### Standard Mode
```python
config = {"mode": "standard"}  # Default
```
Basic evolutionary optimization with fitness-based selection.

### Quality-Diversity (QD) Mode
```python
config = {"mode": "qd"}
```
Explores behavioral space using MAP-Elites for diverse solutions.

### Multi-Objective (MO) Mode
```python
config = {"mode": "mo"}
```
Pareto optimization for multiple conflicting objectives.

### Adversarial Mode
```python
config = {"mode": "adversarial"}
```
Co-evolution for robustness testing and adversarial training.

## Checking LoongFlow Availability

```python
from openevolve.integrations import LoongFlowChecker

# Check if installed
installed = LoongFlowChecker.is_installed()

# Get version
version = LoongFlowChecker.get_version()

# Check availability (with deep requirement check)
available = LoongFlowChecker.is_available(requirement_check=True)

# Get comprehensive diagnostics
diagnostics = LoongFlowChecker.get_diagnostics()
print(f"Installed: {diagnostics['installed']}")
print(f"Version: {diagnostics['version']}")
print(f"Issues: {diagnostics['issues']}")

# Print human-readable diagnostics
LoongFlowChecker.print_diagnostics()
```

## Understanding Status Messages

### LoongFlow Disabled
```
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow Disabled                                          ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow PES has been disabled in the configuration.        ║
║  Evolution will proceed using OpenEvolve-only mode.           ║
...
```

### LoongFlow Not Available
```
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow Not Available                                    ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow package is not installed.                        ║
║  Automatically falling back to OpenEvolve-only mode.         ║
...
```

### Using OpenEvolve-Only Mode
```
╔═══════════════════════════════════════════════════════════════╗
║  Using OpenEvolve-Only Mode                                 ║
║  ────────────────────────────────────────────────────────────  ║
║  Evolution will proceed using OpenEvolve.                    ║
║  Mode: qd                                                     ║
...
```

### LoongFlow Initialized
```
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow PES Initialized                                  ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow Plan-Execute-Summarize system is active.          ║
...
```

## Result Format

Regardless of which system is used, results are returned in a consistent format:

```python
result = {
    "best_solution": "code or solution string",
    "best_fitness": 0.95,
    "iterations_performed": 100,
    "total_evaluations": 2000,
    "convergence_curve": [0.1, 0.3, 0.5, 0.7, 0.85, 0.95],

    # LoongFlow-specific fields (empty in OpenEvolve mode)
    "planning_strategies": [],
    "execution_patterns": [],
    "summaries": [],

    # System identification
    "system_used": "loongflow",  # or "openevolve"
    "mode_used": "pes",  # or "standard", "qd", "mo", "adversarial"
}
```

## Best Practices

### 1. Use Default Configuration for Production
```python
# Recommended: Let the system decide
config = {
    "max_iterations": 100,
    "enable_loongflow": True,
    "require_loongflow": False  # Allow fallback
}
```

### 2. Check Status Before Critical Operations
```python
adapter = LoongFlowAdapter(config)
status = adapter.get_status()

if status['using_loongflow']:
    print("Using advanced PES capabilities")
else:
    print("Using OpenEvolve-native mode")
```

### 3. Handle Both Result Types Uniformly
```python
result = await adapter.evolve(problem="...", domain="...")

# Works for both LoongFlow and OpenEvolve
print(f"System: {result['system_used']}")
print(f"Fitness: {result['best_fitness']}")

# Optional: Handle LoongFlow-specific features
if result['system_used'] == 'loongflow':
    print(f"Planning strategies: {result['planning_strategies']}")
```

### 4. Disable Messages in Automated Scripts
```python
config = {
    "show_messages": False,  # Suppress status messages
    # ... other config
}
```

## Troubleshooting

### LoongFlow Not Installing

1. **Check Installation:**
   ```python
   from openevolve.integrations import LoongFlowChecker
   LoongFlowChecker.print_diagnostics()
   ```

2. **Install LoongFlow:**
   ```bash
   pip install git+https://github.com/baidu-baige/LoongFlow.git
   ```

3. **Use OpenEvolve-Only Mode:**
   ```python
   config = {"enable_loongflow": False}
   ```

### Need LoongFlow Features

If you specifically need LoongFlow's PES capabilities:

```python
config = {
    "enable_loongflow": True,
    "require_loongflow": True  # Fail if not available
}

try:
    adapter = LoongFlowAdapter(config)
except RuntimeError:
    print("Please install LoongFlow to use this feature")
    sys.exit(1)
```

## API Reference

### LoongFlowAdapter

```python
class LoongFlowAdapter:
    def __init__(self, config: Dict[str, Any])
    async def evolve(self, problem: str, domain: str, **kwargs) -> Dict[str, Any]
    def is_available(self) -> bool
    def get_status(self) -> Dict[str, Any]
    def get_capabilities(self) -> Dict[str, Any]
    def print_status() -> None
```

### LoongFlowChecker

```python
class LoongFlowChecker:
    @staticmethod
    def is_installed() -> bool

    @staticmethod
    def get_version() -> Optional[str]

    @staticmethod
    def check_requirements() -> List[str]

    @staticmethod
    def is_available(requirement_check: bool = False) -> bool

    @staticmethod
    def get_diagnostics() -> dict

    @staticmethod
    def print_diagnostics() -> None
```

### OpenEvolveFallbackAdapter

```python
class OpenEvolveFallbackAdapter:
    def __init__(self, openevolve_config: Dict[str, Any])
    async def evolve(self, problem: str, domain: str, **kwargs) -> Dict[str, Any]
    def get_capabilities(self) -> Dict[str, Any]
```

## Summary

The LoongFlow integration with graceful fallback ensures that:

- ✅ **Zero Dependencies**: Works without LoongFlow installed
- ✅ **Full Functionality**: All features available in fallback mode
- ✅ **Transparent Operation**: No code changes needed
- ✅ **Clear Communication**: Users know what's happening
- ✅ **Production Ready**: Robust error handling and recovery
- ✅ **Flexible Configuration**: Choose your preferred mode

OpenEvolve is ready for production use, with or without LoongFlow!
