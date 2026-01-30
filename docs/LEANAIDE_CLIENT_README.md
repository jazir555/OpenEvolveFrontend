# LeanAide Async Python Client

A production-ready asynchronous Python client for the LeanAide theorem prover server.

## Features

- **Async/Await Support**: Built on `aiohttp` for efficient async operations
- **Connection Pooling**: Configurable connection limits for high-throughput scenarios
- **Automatic Retries**: Exponential backoff retry logic for resilient operations
- **Comprehensive Error Handling**: Structured error types for different failure scenarios
- **Health Checks**: Built-in server health monitoring
- **Batch Operations**: Parallel execution of multiple tasks
- **Type Hints**: Full type annotations for better IDE support
- **Logging**: Optional request/response logging for debugging
- **Context Manager**: Proper resource cleanup with async context manager support

## Installation

### Requirements

```bash
pip install aiohttp
```

### Files

- `leanaide_client.py` - Main client library
- `test_leanaide_client.py` - Comprehensive test suite

## Quick Start

### Basic Usage

```python
import asyncio
from leanaide_client import LeanAideClient

async def main():
    # Create client with default settings (localhost:7654)
    async with LeanAideClient() as client:
        # Check if server is healthy
        is_healthy = await client.health_check()
        if not is_healthy:
            print("Server is not responding")
            return

        # Translate a theorem
        result = await client.translate_thm(
            "There are infinitely many prime numbers"
        )

        if result.success:
            print(f"Translation: {result.data}")
            print(f"Response time: {result.response_time:.2f}s")
        else:
            print(f"Error: {result.error}")

asyncio.run(main())
```

### Custom Configuration

```python
from leanaide_client import LeanAideClient, LeanAideConfig

# Create custom configuration
config = LeanAideConfig(
    host="remote-server.com",
    port=7654,
    timeout=300.0,  # 5 minutes
    max_retries=5,
    retry_delay=2.0,
    max_connections=50
)

async with LeanAideClient(config=config) as client:
    result = await client.translate_thm("Test theorem")
```

## Supported Tasks

### 1. Translate Theorem

Translate natural-language theorem to Lean:

```python
result = await client.translate_thm(
    "There are infinitely many prime numbers"
)
```

### 2. Translate Theorem (Detailed)

Translate with optional name:

```python
result = await client.translate_thm_detailed(
    "There are infinitely many prime numbers",
    theorem_name="infinitely_many_primes"
)
```

### 3. Translate Definition

Translate natural-language definition:

```python
result = await client.translate_def(
    "A number is cube-free if it is not divisible by the cube of any prime"
)
```

### 4. Generate Theorem Documentation

Generate docs for a theorem:

```python
result = await client.theorem_doc(
    theorem_name="infinitely_many_primes",
    theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
)
```

### 5. Generate Definition Documentation

Generate docs for a definition:

```python
result = await client.def_doc(
    definition_name="cube_free",
    definition_code="def cube_free (n : Nat) : Prop := ..."
)
```

### 6. Generate Theorem Name

Auto-generate theorem name:

```python
result = await client.theorem_name(
    "There are infinitely many prime numbers"
)
```

### 7. Prove for Formalization

Generate proof sketch:

```python
result = await client.prove_for_formalization(
    theorem_text="There are infinitely many prime numbers",
    theorem_code="Infinite {p : Nat | Prime p}",
    theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
)
```

### 8. JSON Structured

Convert document to structured JSON:

```python
result = await client.json_structured(
    "Theorem: There are infinitely many primes. Proof: ..."
)
```

### 9. Lean from JSON Structured

Generate Lean code from JSON:

```python
result = await client.lean_from_json_structured({
    "title": "Infinite Primes",
    "statements": [...],
    "proofs": [...]
})
```

### 10. Elaborate Lean Code

Elaborate and analyze Lean code:

```python
result = await client.elaborate("""
theorem infinitely_many_primes : Infinite {p : Nat | Prime p} := by
  sorry
""")
```

Result includes:
- `declarations`: List of declaration names
- `logs`: Elaboration logs
- `sorries`: Unproven obligations
- `sorriesAfterPurge`: Remaining obligations after simplification

### 11. Math Query

Ask math questions:

```python
result = await client.math_query(
    "What is the fundamental theorem of algebra?",
    n=3  # Generate 3 candidate answers
)
```

With conversation history:

```python
result = await client.math_query(
    "Can you explain more?",
    history=[
        {"role": "user", "content": "What is a group?"},
        {"role": "assistant", "content": "A group is..."}
    ],
    n=1
)
```

## Advanced Features

### Batch Operations

Translate multiple theorems in parallel:

```python
theorems = [
    "There are infinitely many primes",
    "The square root of 2 is irrational",
    "Every natural number has a unique prime factorization"
]

results = await client.batch_translate_theorems(theorems)

for i, result in enumerate(results):
    print(f"Theorem {i+1}: {result.success}")
    if result.data:
        print(f"  Translation: {result.data}")
```

Batch translate definitions:

```python
definitions = [
    "A prime number is...",
    "A perfect square is...",
    "A cube-free number is..."
]

results = await client.batch_translate_definitions(definitions)
```

### Parallel Task Execution

Execute different tasks in parallel:

```python
tasks = [
    {"task": "translate_thm", "theorem_text": "Test 1"},
    {"task": "translate_def", "definition_text": "Test 2"},
    {"task": "theorem_doc", "theorem_name": "test", "theorem_statement": "..."}
]

results = await client.execute_parallel_tasks(tasks)
```

### Chained Tasks

Execute tasks where output feeds into next:

```python
tasks = [
    {"task": "translate_thm", "theorem_text": "There are infinitely many primes"},
    {"task": "theorem_doc", "theorem_name": "infinitely_many_primes"}
]

results = await client.execute_tasks(tasks)
```

## Error Handling

The client provides structured error handling:

```python
from leanaide_client import (
    LeanAideClient,
    ConnectionError,
    TimeoutError,
    TaskExecutionError,
    ValidationError
)

async with LeanAideClient() as client:
    result = await client.translate_thm("test")

    if result.success:
        print("Success:", result.data)
    else:
        print("Error:", result.error)

        # Check error type
        if "timeout" in result.error.lower():
            print("Request timed out - try increasing timeout")
        elif "connection" in result.error.lower():
            print("Cannot connect to server")
        elif "validation" in result.error.lower():
            print("Invalid input parameters")
```

### Result Structure

All operations return a `LeanAideResult`:

```python
@dataclass
class LeanAideResult:
    success: bool              # True if operation succeeded
    task: str                  # Task name
    data: Optional[Dict]       # Response data (if success)
    error: Optional[str]       # Error message (if failed)
    logs: Optional[str]        # Server logs
    response_time: float       # Response time in seconds
    timestamp: str             # ISO timestamp
```

## Configuration Options

### LeanAideConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | `"localhost"` | Server hostname/IP |
| `port` | int | `7654` | Server port |
| `timeout` | float | `6000.0` | Request timeout (seconds) |
| `connect_timeout` | float | `30.0` | Connection timeout (seconds) |
| `max_retries` | int | `3` | Maximum retry attempts |
| `retry_delay` | float | `1.0` | Initial retry delay (seconds) |
| `max_connections` | int | `100` | Max connection pool size |
| `enable_logging` | bool | `True` | Enable request logging |
| `verify_ssl` | bool | `False` | Verify SSL certificates |

## Testing

### Run All Tests

```bash
pytest test_leanaide_client.py -v
```

### Run Specific Test

```bash
pytest test_leanaide_client.py::test_translate_thm -v
```

### Integration Tests (requires running server)

```bash
pytest test_leanaide_client.py -m integration
```

### Test Coverage

```bash
pytest test_leanaide_client.py --cov=leanaide_client --cov-report=html
```

## Examples

### Example 1: Complete Translation Workflow

```python
import asyncio
from leanaide_client import LeanAideClient

async def translate_and_document():
    async with LeanAideClient() as client:
        # Step 1: Translate theorem
        translation = await client.translate_thm(
            "There are infinitely many prime numbers"
        )

        if not translation.success:
            print(f"Translation failed: {translation.error}")
            return

        print(f"Translated: {translation.data}")

        # Step 2: Generate documentation
        if translation.data:
            doc = await client.theorem_doc(
                theorem_name="infinitely_many_primes",
                theorem_statement=translation.data.get("result", "")
            )

            if doc.success:
                print(f"Documentation: {doc.data}")

asyncio.run(translate_and_document())
```

### Example 2: Batch Processing with Error Handling

```python
async def batch_translate_with_retry():
    async with LeanAideClient(
        max_retries=5,
        timeout=300.0
    ) as client:
        theorems = [
            "There are infinitely many primes",
            "The square root of 2 is irrational",
            "Every natural number has a unique prime factorization"
        ]

        results = await client.batch_translate_theorems(theorems)

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        print(f"Successfully translated: {successful}/{len(results)}")

        for i, result in enumerate(results):
            if result.success:
                print(f"✓ Theorem {i+1}: {result.response_time:.2f}s")
            else:
                print(f"✗ Theorem {i+1}: {result.error}")

asyncio.run(batch_translate_with_retry())
```

### Example 3: Interactive Math Assistant

```python
async def math_assistant():
    async with LeanAideClient() as client:
        history = []

        while True:
            question = input("\nYour question (or 'quit'): ")

            if question.lower() == 'quit':
                break

            result = await client.math_query(
                question,
                history=history,
                n=1
            )

            if result.success and result.data:
                answer = result.data.get("result", ["No answer"])[0]
                print(f"\nAnswer: {answer}")

                # Add to conversation history
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})
            else:
                print(f"Error: {result.error}")

asyncio.run(math_assistant())
```

## Performance Tips

1. **Use Batch Operations**: Process multiple items in parallel for better throughput
2. **Adjust Connection Pool**: Increase `max_connections` for high-load scenarios
3. **Tune Timeouts**: Set appropriate timeouts based on task complexity
4. **Enable Retries**: Keep retries enabled for resilience (default: 3)
5. **Reuse Client**: Use context manager to reuse connections

## Troubleshooting

### Connection Refused

```
Error: Cannot connect to server
```

**Solutions:**
- Ensure LeanAide server is running: `python3 leanaide_server.py`
- Check host and port configuration
- Verify firewall settings

### Timeout Errors

```
Error: Request timed out after 6000.0s
```

**Solutions:**
- Increase timeout in config: `LeanAideConfig(timeout=12000.0)`
- Check server logs for bottlenecks
- Reduce task complexity

### Validation Errors

```
Error: Missing required field: theorem_text
```

**Solutions:**
- Check required parameters for task type
- Ensure JSON serialization is correct
- Review API documentation

## Server Setup

For information on setting up the LeanAide server, see:
[LeanAide/README.md](LeanAide/README.md)

### Quick Start

```bash
cd LeanAide
pip install -r server/requirements.txt
python3 leanaide_server.py --ui
```

## API Reference

See inline documentation in `leanaide_client.py` for detailed API reference.

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- New features include tests
- Code follows Python best practices
- Type hints are included

## License

Same as LeanAide project.

## Support

For issues specific to this client, open an issue in the repository.
For LeanAide server issues, see the main LeanAide project.
