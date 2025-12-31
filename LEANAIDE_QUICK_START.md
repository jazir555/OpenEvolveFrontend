# LeanAide Quick Start Guide

Get started with LeanAide integration in OpenEvolve workflows in 5 minutes.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Common Operations](#common-operations)
4. [Examples](#examples)
5. [Next Steps](#next-steps)

---

## Installation

### 1. Install Dependencies

```bash
pip install aiohttp asyncio requests
```

### 2. Start LeanAide Server

```bash
# Navigate to LeanAide directory
cd LeanAide

# Start the server
python leanaide_server.py

# Server runs on http://localhost:7654
```

### 3. Verify Installation

```bash
# Check server health
curl http://localhost:7654/

# Or use Python
python -c "import requests; print(requests.get('http://localhost:7654/').status_code)"
```

---

## Basic Usage

### Translate a Theorem

```python
import asyncio
from leanaide_client import LeanAideClient

async def translate():
    client = LeanAideClient()
    
    result = await client.translate_thm(
        "There are infinitely many prime numbers"
    )
    
    if result.success:
        print(f"Lean code: {result.data.get('lean_code')}")
    
    await client.close()

asyncio.run(translate())
```

### Math Q&A

```python
async def math_qa():
    client = LeanAideClient()
    
    result = await client.math_query(
        "What is the fundamental theorem of algebra?",
        n=3
    )
    
    if result.success:
        for i, answer in enumerate(result.data.get('answers', []), 1):
            print(f"Answer {i}: {answer}")
    
    await client.close()

asyncio.run(math_qa())
```

### Verify Lean Code

```python
async def verify():
    client = LeanAideClient()
    
    lean_code = """
    theorem add_comm (a b : Nat) : a + b = b + a := by
      simp [Nat.add_comm]
    """
    
    result = await client.elaborate(lean_code)
    
    if result.success:
        print("Code verified successfully!")
    else:
        print("Verification failed")
    
    await client.close()

asyncio.run(verify())
```

---

## Common Operations

### Batch Translation

```python
async def batch_translate():
    client = LeanAideClient()
    
    theorems = [
        "There are infinitely many primes",
        "The square root of 2 is irrational",
        "Every natural number has a unique prime factorization"
    ]
    
    results = await client.batch_translate_theorems(theorems)
    
    for i, result in enumerate(results):
        if result.success:
            print(f"Theorem {i+1}: {result.data.get('lean_code')}")
    
    await client.close()

asyncio.run(batch_translate())
```

### Generate Documentation

```python
async def generate_docs():
    client = LeanAideClient()
    
    result = await client.theorem_doc(
        theorem_name="infinitely_many_primes",
        theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
    )
    
    if result.success:
        print(result.data.get('documentation'))
    
    await client.close()

asyncio.run(generate_docs())
```

---

## Examples

### Example 1: Full Workflow

```python
from leanaide_hephaestus_bridge import LeanAideHephaestusBridge
import asyncio

async def full_workflow():
    bridge = LeanAideHephaestusBridge()
    
    result = await bridge.execute_full_workflow(
        "Prove that there are infinitely many prime numbers"
    )
    
    if result['workflow_success']:
        print("Workflow completed successfully!")
        print(f"Domain: {result['phases']['phase_1']['metadata']['domain']}")
        print(f"Lean code: {result['phases']['phase_2']['lean_code']}")
    else:
        print(f"Workflow failed: {result.get('failure_phase')}")
    
    await bridge.cleanup()

asyncio.run(full_workflow())
```

### Example 2: Mathematical Content Detection

```python
from leanaide_hephaestus_bridge import MathematicalProblemDetector

detector = MathematicalProblemDetector()

# Check if text is mathematical
text = "Prove there are infinitely many prime numbers"
has_math = detector.detect_mathematical_content(text)
print(f"Is mathematical: {has_math}")

# Classify domain
domain = detector.classify_domain(text)
print(f"Domain: {domain.value}")  # NUMBER_THEORY
```

### Example 3: Using MCP Tools

```python
from leanaide_mcp_tools import leanaide_translate_theorem

# Translate theorem
result = leanaide_translate_theorem(
    theorem_text="There are infinitely many prime numbers"
)

print(result)
# Output: {'success': True, 'lean_code': '...', ...}
```

---

## Next Steps

1. **Read Full Documentation**: [LEANAIDE_INTEGRATION_GUIDE.md](LEANAIDE_INTEGRATION_GUIDE.md)
2. **Explore API Reference**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3. **Check Examples**: See `demo_leanaide_client.py` for more examples
4. **Run Tests**: Execute `test_leanaide_client.py` to verify installation

## Configuration

Set environment variables:

```bash
export LEANAIDE_HOST=localhost
export LEANAIDE_PORT=7654
export LEANAIDE_TIMEOUT=120
```

Or use Python configuration:

```python
from leanaide_client import LeanAideConfig

config = LeanAideConfig(
    host="localhost",
    port=7654,
    timeout=6000.0,
    max_connections=100
)
client = LeanAideClient(config)
```

## Troubleshooting

**Server not available?**
```bash
# Check if server is running
curl http://localhost:7654/

# Start server
cd LeanAide && python leanaide_server.py
```

**Timeout errors?**
```python
# Increase timeout
config = LeanAideConfig(timeout=600.0)  # 10 minutes
```

**Import errors?**
```bash
# Install dependencies
pip install aiohttp asyncio requests
```

## Support

- Documentation: [LEANAIDE_INTEGRATION_GUIDE.md](LEANAIDE_INTEGRATION_GUIDE.md)
- API Reference: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- Examples: `demo_leanaide_client.py`
- Tests: `test_leanaide_client.py`
