# LeanAide Client - Quick Reference

## Installation

```bash
pip install -r leanaide_client_requirements.txt
```

## Import

```python
from leanaide_client import LeanAideClient, LeanAideConfig
```

## Basic Usage

```python
async with LeanAideClient() as client:
    result = await client.translate_thm("Theorem text")
    if result.success:
        print(result.data)
```

## All Task Methods

| Method | Description |
|--------|-------------|
| `translate_thm(theorem_text)` | Translate theorem to Lean |
| `translate_thm_detailed(text, name=None)` | Translate with naming |
| `translate_def(definition_text)` | Translate definition |
| `theorem_doc(name, statement)` | Generate theorem docs |
| `def_doc(name, code)` | Generate definition docs |
| `theorem_name(theorem_text)` | Generate theorem name |
| `prove_for_formalization(text, code, statement)` | Generate proof sketch |
| `json_structured(document_text)` | Convert to structured JSON |
| `lean_from_json_structured(json)` | Generate Lean from JSON |
| `elaborate(document_code)` | Elaborate Lean code |
| `math_query(query, history=None, n=3)` | Ask math question |

## Configuration

```python
config = LeanAideConfig(
    host="localhost",        # Server host
    port=7654,              # Server port
    timeout=6000.0,         # Request timeout (seconds)
    connect_timeout=30.0,   # Connection timeout
    max_retries=3,          # Retry attempts
    retry_delay=1.0,        # Initial retry delay
    max_connections=100,    # Connection pool size
    enable_logging=True     # Enable logging
)
```

## Result Structure

```python
LeanAideResult(
    success: bool,          # True if succeeded
    task: str,              # Task name
    data: Dict,             # Response data
    error: str,             # Error message
    logs: str,              # Server logs
    response_time: float,   # Response time (seconds)
    timestamp: str          # ISO timestamp
)
```

## Batch Operations

```python
# Batch translate theorems
results = await client.batch_translate_theorems([
    "Theorem 1",
    "Theorem 2"
])

# Batch translate definitions
results = await client.batch_translate_definitions([
    "Definition 1",
    "Definition 2"
])

# Parallel different tasks
results = await client.execute_parallel_tasks([
    {"task": "translate_thm", "theorem_text": "..."},
    {"task": "translate_def", "definition_text": "..."}
])
```

## Health Check

```python
is_healthy = await client.health_check()
```

## Error Handling

```python
result = await client.translate_thm("test")

if result.success:
    print(result.data)
else:
    print(f"Error: {result.error}")
    if "timeout" in result.error.lower():
        # Handle timeout
    elif "connection" in result.error.lower():
        # Handle connection error
```

## Example: Complete Workflow

```python
async def translate_workflow():
    async with LeanAideClient() as client:
        # Translate
        result = await client.translate_thm("Theorem text")

        if result.success:
            # Elaborate
            elaborated = await client.elaborate(result.data["result"])

            # Generate docs
            docs = await client.theorem_doc(
                "theorem_name",
                result.data["result"]
            )

            return {
                "translation": result.data,
                "elaboration": elaborated.data,
                "docs": docs.data
            }
```

## Testing

```bash
# Run all tests
pytest test_leanaide_client.py -v

# Run integration tests (requires server)
pytest test_leanaide_client.py -m integration

# Run with coverage
pytest test_leanaide_client.py --cov=leanaide_client
```

## Demo

```bash
# Run automated demos
python demo_leanaide_client.py

# Run interactive demo
python demo_leanaide_client.py --interactive
```

## Common Patterns

### Custom Config with Retries

```python
config = LeanAideConfig(
    timeout=300.0,
    max_retries=5,
    retry_delay=2.0
)
async with LeanAideClient(config=config) as client:
    result = await client.translate_thm("...")
```

### Parallel Processing

```python
tasks = [
    client.translate_thm(t) for t in theorems
]
results = await asyncio.gather(*tasks)
```

### Error Recovery

```python
async def safe_translate(client, text, max_attempts=3):
    for attempt in range(max_attempts):
        result = await client.translate_thm(text)
        if result.success:
            return result
        await asyncio.sleep(2 ** attempt)
    return result
```

## Server Setup

```bash
cd LeanAide
pip install -r server/requirements.txt
python3 leanaide_server.py --ui
```

Default: `http://localhost:7654`

## Files

- `leanaide_client.py` - Main client library
- `test_leanaide_client.py` - Test suite
- `demo_leanaide_client.py` - Demo scripts
- `LEANAIDE_CLIENT_README.md` - Full documentation
- `leanaide_client_requirements.txt` - Dependencies
