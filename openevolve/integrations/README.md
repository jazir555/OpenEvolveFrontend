# LoongFlow Integration Wrapper

This directory contains the integration adapter for connecting OpenEvolve with LoongFlow's PES (Plan-Execute-Summarize) system.

## Overview

The `LoongFlowAdapter` class provides a seamless interface between OpenEvolve's evolutionary framework and LoongFlow's PES agent system. It handles:

- Configuration mapping between OpenEvolve and LoongFlow formats
- Graceful fallback when LoongFlow is not installed
- Error handling and recovery
- Result format conversion

## Installation

The adapter requires LoongFlow to be installed:

```bash
pip install loongflow
```

If LoongFlow is not installed, the adapter will operate in fallback mode and return appropriate error messages.

## Quick Start

```python
import asyncio
from openevolve.integrations import LoongFlowAdapter

async def main():
    # Configure the adapter
    config = {
        "max_iterations": 50,
        "population_size": 10,
        "enable_planning": True,
        "enable_memory": True,
        "llm_config": {
            "model": "gpt-4",
            "temperature": 0.7
        }
    }

    # Initialize adapter
    adapter = LoongFlowAdapter(config)

    # Check availability
    if adapter.is_available():
        # Run evolution
        result = await adapter.evolve(
            problem="Optimize this function",
            domain="code"
        )

        print(f"Best fitness: {result['best_fitness']}")
        print(f"Evaluations: {result['total_evaluations']}")
    else:
        print("LoongFlow not available")

asyncio.run(main())
```

## Configuration

The adapter accepts the following configuration parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 100 | Maximum number of evolution iterations |
| `population_size` | int | 20 | Size of the population |
| `timeout` | int | 300 | Timeout in seconds |
| `enable_planning` | bool | True | Enable planning phase |
| `enable_memory` | bool | True | Enable memory system |
| `llm_config` | dict | {} | LLM configuration |

## API Reference

### `LoongFlowAdapter(config: Dict[str, Any])`

Creates a new LoongFlow adapter instance.

### `async evolve(problem: str, domain: str, **kwargs) -> Dict`

Run PES evolution using LoongFlow.

**Parameters:**
- `problem` (str): Problem description
- `domain` (str): Problem domain (general, math, code, ml)
- `initial_code` (str, optional): Initial code solution
- `evaluator` (Any, optional): Evaluator function/object
- `**kwargs`: Additional parameters

**Returns:**
```python
{
    "best_solution": str | None,
    "best_fitness": float,
    "total_evaluations": int,
    "improvement_rate": float,
    "iterations_performed": int,
    "strategy_used": "pes",
    "source": "loongflow_pes",
    "error": str | None
}
```

### `is_available() -> bool`

Check if LoongFlow is available and initialized.

### `get_capabilities() -> Dict[str, Any]`

Get adapter capabilities including supported domains and features.

## Testing

Run the integration tests:

```bash
pytest tests/integrations/test_loongflow_adapter.py -v
```

Run the usage examples:

```bash
python examples/loongflow_adapter_usage.py
```

## Architecture

```
OpenEvolve                    LoongFlow
    │                             │
    ├── Config ──────────────> Map Config
    │                             │
    ├── Adapter                   │
    │   │                         │
    │   ├── evolve()              │
    │   │   │                     │
    │   │   └── Convert ──────> PES Agent
    │   │                         │
    │   └── Result <───────── Result
    │                             │
```

## Error Handling

The adapter implements robust error handling:

1. **Import Errors**: If LoongFlow is not installed, the adapter sets `available=False`
2. **Evolution Errors**: If evolution fails, returns result with error message
3. **Fallback Mode**: Always returns a valid result structure, even when unavailable

## Examples

See `examples/loongflow_adapter_usage.py` for comprehensive examples including:

- Basic usage
- LLM configuration
- Math optimization
- Error handling

## License

Apache-2.0
