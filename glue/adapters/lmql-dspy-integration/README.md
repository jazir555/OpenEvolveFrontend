# LMQL-DSPy Integration

This integration enables LMQL (Language Model Query Language) to work seamlessly with DSPy (Declarative Structured Prompt Engineering) for enhanced constrained reasoning and programmatic prompting capabilities.

## Architecture

The integration follows the CLAUDE.md principles:

- **Zero Trust**: All inputs and outputs are validated
- **Anti-Hallucination**: Data integrity is verified through constraint validation
- **Read-Only State**: Underlying systems' data remains unmodified
- **Idempotency**: Operations are safe to repeat
- **Configuration Explicitness**: All parameters are configurable via environment variables
- **UTC**: All timestamps are stored in UTC

## Components

### 1. LMQLDSPyAdapter
The main adapter class that bridges LMQL with DSPy, providing:

- Constrained chain of thought reasoning
- Constrained program of thought execution
- Constrained multi-step reasoning
- Constrained signature-based solving
- Batch solving with constraints

### 2. Interface Functions
Provides a clean API for unified access to both systems:

- `constrained_cot`: Chain of thought with constraints
- `constrained_pot`: Program of thought with constraints
- `constrained_multi`: Multi-step reasoning with constraints
- `constrained_signature`: Signature-based solving with constraints
- `batch_constrained`: Batch solving with constraints

## Usage

### Direct Usage
```python
import asyncio
from lmql_dspy_adapter import LMQLDSPyAdapter, create_unified_interface
from lmql_adapter import create_list_constraint

async def main():
    # Initialize the adapter
    config = {
        'log_level': 'INFO',
        'dspy_config': {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4096
        },
        'lmql_config': {
            'fallback_on_error': True,
            'enable_metrics': True
        }
    }
    
    adapter = LMQLDSPyAdapter(config=config)
    
    # Create the unified interface
    unified_interface = create_unified_interface(adapter)
    
    # Example: Constrained chain of thought
    boolean_constraint = create_list_constraint("answer", ["yes", "no"])
    
    cot_result = await unified_interface(
        'constrained_cot',
        question="Is the Earth round?",
        constraints=[boolean_constraint]
    )
    print("Constrained chain of thought result:", cot_result)
    
    # Example: Constrained program of thought
    pot_result = await unified_interface(
        'constrained_pot',
        question="What is 15 * 23?",
        constraints=[create_datatype_constraint("answer", "int")]
    )
    print("Constrained program of thought result:", pot_result)

# Run the async function
asyncio.run(main())
```

### Integration with Existing Systems
The adapter can be integrated into existing workflows to enhance them with constrained reasoning:

```python
import asyncio
from lmql_dspy_adapter import LMQLDSPyAdapter, create_unified_interface
from lmql_adapter import create_range_constraint

async def enhanced_reasoning_workflow():
    # Initialize the adapter
    adapter = LMQLDSPyAdapter()
    unified_interface = create_unified_interface(adapter)
    
    # Use constrained reasoning to ensure outputs meet specific criteria
    question = "Calculate the probability of success for this project"
    
    # Apply a range constraint to ensure the answer is between 0 and 1
    probability_constraint = create_range_constraint("probability", 0.0, 1.0)
    
    result = await unified_interface(
        'constrained_signature',
        question=question,
        signature="question -> probability",
        constraints=[probability_constraint]
    )
    
    return result
```

## Configuration

The adapter supports the following configuration options:

- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `timeout_seconds`: Timeout for operations
- `dspy_config`: Configuration for the DSPy integration
- `lmql_config`: Configuration for the LMQL adapter
- `integration_enabled`: Whether the integration is enabled

## Testing

Run the test suite:
```bash
python test_integration.py
```

## Deployment

The integration can be deployed as a container using the provided Dockerfile:

```bash
docker build -t lmql-dspy-adapter .
docker run -d --name lmql-dspy lmql-dspy-adapter
```

## Security

This integration follows security best practices:

- No direct access to underlying systems' internal data structures
- Input validation for all queries
- Read-only access to data sources
- Proper error handling to prevent information disclosure
- Isolated execution environments for each system