# LeanAide MCP Tools - Quick Reference Guide

## Overview

Comprehensive MCP (Model Context Protocol) tools for integrating LeanAide's AI-powered formal mathematics capabilities with CrewAI agents and workflows.

**File Location:** `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\leanaide_mcp_tools.py`

## Architecture

```
CrewAI Orchestrator
    ↓
LeanAide MCP Tools
    ↓
LeanAide Server (localhost:7654)
    ↓
Lean 4 Theorem Prover
```

## Available Tools (8 Total)

### 1. `leanaide_translate_theorem`
Translate natural language theorems to Lean code.

**Inputs:**
- `theorem_text` (str, required): Natural language theorem statement
- `theorem_name` (str, optional): Name for the theorem
- `host` (str, optional): LeanAide server host (default: localhost)
- `port` (int, optional): LeanAide server port (default: 7654)
- `timeout` (int, optional): Request timeout in seconds (default: 120)

**Returns:**
```python
{
    "success": True,
    "theorem_name": "infinitely_many_primes",
    "lean_code": "theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p} := by sorry",
    "elaborated_type": "...",
    "command_syntax": "...",
    "execution_time": 2.45,
    "message": "Theorem translated successfully in 2.45s"
}
```

**Example:**
```python
from leanaide_mcp_tools import leanaide_translate_theorem

result = leanaide_translate_theorem(
    theorem_text="There are infinitely many primes",
    theorem_name="infinitely_many_primes"
)
print(result['lean_code'])
```

---

### 2. `leanaide_translate_definition`
Translate natural language definitions to Lean code.

**Inputs:**
- `definition_text` (str, required): Natural language definition
- `host`, `port`, `timeout` (optional): Server configuration

**Returns:**
```python
{
    "success": True,
    "definition_text": "...",
    "lean_code": "def cube_free (n : Nat) := ...",
    "execution_time": 1.83,
    "message": "Definition translated successfully"
}
```

**Example:**
```python
result = leanaide_translate_definition(
    definition_text="A number is cube-free if it is not divisible by the cube of any prime"
)
print(result['lean_code'])
```

---

### 3. `leanaide_generate_proof`
Generate proofs for theorems.

**Inputs:**
- `theorem_text` (str, required): Natural language theorem
- `theorem_code` (str, optional): Pre-translated Lean code
- `host`, `port`, `timeout` (optional): Server configuration

**Returns:**
```python
{
    "success": True,
    "theorem_text": "...",
    "proof_document": "We will prove by contradiction...",
    "structured_proof": {...},
    "lean_proof": "...",
    "execution_time": 15.2
}
```

**Example:**
```python
result = leanaide_generate_proof(
    theorem_text="The square root of 2 is irrational"
)
print(result['proof_document'])
```

---

### 4. `leanaide_verify_solution`
Verify Lean code correctness by elaboration.

**Inputs:**
- `code` (str, required): Lean code to verify
- `host`, `port`, `timeout` (optional): Server configuration

**Returns:**
```python
{
    "success": True,
    "is_valid": True,
    "declarations": ["add_comm", "mul_assoc"],
    "logs": [...],
    "sorries": [],
    "sorries_after_purge": [],
    "unproven_count": 0,
    "execution_time": 3.1
}
```

**Example:**
```python
code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [add_comm]
"""

result = leanaide_verify_solution(code)
print(f"Valid: {result['is_valid']}")
print(f"Unproven: {result['unproven_count']}")
```

---

### 5. `leanaide_math_query`
Answer mathematical questions.

**Inputs:**
- `query` (str, required): Mathematical question
- `history` (list, optional): Conversation history
- `n` (int, optional): Number of answers (default: 3)
- `host`, `port`, `timeout` (optional): Server configuration

**Returns:**
```python
{
    "success": True,
    "query": "What is the fundamental theorem of calculus?",
    "answers": ["Answer 1", "Answer 2", "Answer 3"],
    "num_answers": 3,
    "execution_time": 4.2
}
```

**Example:**
```python
result = leanaide_math_query(
    query="What is the fundamental theorem of calculus?",
    n=3
)

for i, answer in enumerate(result['answers'], 1):
    print(f"Answer {i}: {answer}")
```

---

### 6. `leanaide_generate_documentation`
Generate documentation for Lean code.

**Inputs:**
- `name` (str, required): Name of theorem/definition
- `code` (str, required): Lean code
- `doc_type` (str, required): "theorem" or "definition"
- `host`, `port`, `timeout` (optional): Server configuration

**Returns:**
```python
{
    "success": True,
    "name": "infinitely_many_primes",
    "doc_type": "theorem",
    "documentation": "This theorem states that there are infinitely many prime numbers...",
    "execution_time": 2.1
}
```

**Example:**
```python
result = leanaide_generate_documentation(
    name="infinitely_many_primes",
    code="theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p}",
    doc_type="theorem"
)
print(result['documentation'])
```

---

### 7. `leanaide_elaborate_code`
Elaborate Lean code and report errors (alias for verify with error focus).

**Inputs:**
- `code` (str, required): Lean code to elaborate
- `host`, `port`, `timeout` (optional): Server configuration

**Returns:**
```python
{
    "success": True,
    "has_errors": False,
    "declarations": [...],
    "logs": [...],
    "errors": [],
    "warnings": [...],
    "unsolved_goals": [],
    "unsolved_goal_count": 0,
    "execution_time": 2.8
}
```

**Example:**
```python
result = leanaide_elaborate_code(code)
if result['has_errors']:
    print("Errors found:")
    for error in result['errors']:
        print(f"  - {error}")
else:
    print("Code elaborates successfully!")
```

---

### 8. `get_leanaide_status`
Check LeanAide server availability.

**Inputs:** None

**Returns:**
```python
{
    "available": True,
    "host": "localhost",
    "port": 7654,
    "timeout": 120,
    "message": "LeanAide server is reachable at localhost:7654"
}
```

**Example:**
```python
from leanaide_mcp_tools import get_leanaide_status

status = get_leanaide_status()
if status['available']:
    print("LeanAide is ready!")
else:
    print(f"Cannot connect: {status['message']}")
```

---

## Configuration

### Environment Variables

```bash
# LeanAide Server Configuration
export LEANAIDE_HOST=localhost      # Server hostname (default: localhost)
export LEANAIDE_PORT=7654           # Server port (default: 7654)
export LEANAIDE_TIMEOUT=120         # Request timeout in seconds (default: 120)
```

### Starting the LeanAide Server

```bash
# Navigate to LeanAide directory
cd LeanAide

# Start the server
python3 leanaide_server.py

# Or with UI
python3 leanaide_server.py --ui
```

---

## Async Versions

All main tools have async variants for asynchronous workflows:

```python
import asyncio
from leanaide_mcp_tools import leanaide_translate_theorem_async

async def main():
    result = await leanaide_translate_theorem_async(
        theorem_text="There are infinitely many primes"
    )
    print(result['lean_code'])

asyncio.run(main())
```

Available async functions:
- `leanaide_translate_theorem_async`
- `leanaide_translate_definition_async`
- `leanaide_generate_proof_async`
- `leanaide_verify_solution_async`

---

## Error Handling

All tools return error-safe dictionaries:

```python
result = leanaide_translate_theorem(theorem_text="...")

if result['success']:
    print(f"Success: {result['message']}")
    print(f"Lean code: {result['lean_code']}")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

Common errors:
- `LeanAideConnectionError`: Server not reachable
- `LeanAideTimeoutError`: Request timed out
- `LeanAideClientError`: Other server errors

---

## Integration with CrewAI

### Basic Usage

```python
from crewai_client import CrewAIClient
from leanaide_mcp_tools import leanaide_translate_theorem

# Delegate theorem translation to LeanAide
result = leanaide_translate_theorem(
    theorem_text="Every natural number has a prime factorization"
)

# Use in CrewAI workflow
if result['success']:
    lean_code = result['lean_code']
    # Continue with CrewAI workflow
```

### MCP Tool Registry

```python
from leanaide_mcp_tools import list_mcp_tools, get_mcp_tool

# List all available tools
tools = list_mcp_tools()
print(f"Available tools: {tools}")

# Get a specific tool
translate_tool = get_mcp_tool("leanaide_translate_theorem")
result = translate_tool(theorem_text="...")
```

---

## LeanAide JSON API Tasks

The MCP tools map to LeanAide's JSON API tasks:

| MCP Tool | LeanAide Task | Description |
|----------|---------------|-------------|
| `leanaide_translate_theorem` | `translate_thm`, `translate_thm_detailed` | Translate theorems |
| `leanaide_translate_definition` | `translate_def` | Translate definitions |
| `leanaide_generate_proof` | `prove_for_formalization` | Generate proofs |
| `leanaide_verify_solution` | `elaborate` | Elaborate code |
| `leanaide_elaborate_code` | `elaborate` | Elaborate code (error focus) |
| `leanaide_math_query` | `math_query` | Answer math questions |
| `leanaide_generate_documentation` | `theorem_doc`, `def_doc` | Generate docs |

---

## Example Workflow

```python
from leanaide_mcp_tools import (
    leanaide_translate_theorem,
    leanaide_generate_proof,
    leanaide_verify_solution,
    get_leanaide_status
)

# 1. Check server status
status = get_leanaide_status()
if not status['available']:
    print("LeanAide server not available!")
    exit(1)

# 2. Translate a theorem
theorem = "The sum of two even numbers is even"
result = leanaide_translate_theorem(
    theorem_text=theorem,
    theorem_name="sum_even_even"
)

if not result['success']:
    print(f"Translation failed: {result.get('error')}")
    exit(1)

lean_code = result['lean_code']
print(f"Translated:\n{lean_code}\n")

# 3. Generate a proof
proof_result = leanaide_generate_proof(
    theorem_text=theorem,
    theorem_code=lean_code
)

print(f"Proof sketch:\n{proof_result['proof_document']}\n")

# 4. Verify the code
verify_result = leanaide_verify_solution(lean_code)
print(f"Valid: {verify_result['is_valid']}")
print(f"Unproven goals: {verify_result['unproven_count']}")
```

---

## Security Features

- Input validation for all parameters
- Path traversal protection
- Command injection prevention
- Safe error messages (no sensitive data leakage)
- Thread-safe MCP tool registry
- Timeout protection for long-running requests

---

## Performance Considerations

- **Timeouts**: Default 120s, configurable via `LEANAIDE_TIMEOUT`
- **Caching**: The LeanAide server handles caching internally
- **Async support**: Use async versions for concurrent requests
- **Connection pooling**: Reuses global client instance

---

## Troubleshooting

### Server Not Responding

```bash
# Check if server is running
curl http://localhost:7654

# Start server
cd LeanAide
python3 leanaide_server.py
```

### Timeout Errors

```python
# Increase timeout for complex proofs
result = leanaide_generate_proof(
    theorem_text="...",
    timeout=300  # 5 minutes
)
```

### Import Errors

```bash
# Install dependencies
pip install pydantic

# Ensure ace_security_utils.py is in the same directory
# or adjust the import path
```

---

## File Structure

```
leanaide_mcp_tools.py (924 lines)
├── Imports & Configuration
├── Custom Exceptions
│   ├── LeanAideClientError
│   ├── LeanAideConnectionError
│   └── LeanAideTimeoutError
├── MCP Tool Registry (thread-safe)
├── LeanAideClient Class
│   ├── translate_theorem()
│   ├── translate_definition()
│   ├── generate_proof()
│   ├── elaborate_code()
│   ├── math_query()
│   └── generate_documentation()
├── 8 MCP Tool Functions
│   ├── leanaide_translate_theorem
│   ├── leanaide_translate_definition
│   ├── leanaide_generate_proof
│   ├── leanaide_verify_solution
│   ├── leanaide_math_query
│   ├── leanaide_generate_documentation
│   ├── leanaide_elaborate_code
│   └── get_leanaide_status
├── Async Wrappers (4 functions)
└── Module Initialization
```

---

## Quick Start

```python
# 1. Import the tools
from leanaide_mcp_tools import (
    leanaide_translate_theorem,
    get_leanaide_status
)

# 2. Check server
status = get_leanaide_status()
print(f"LeanAide available: {status['available']}")

# 3. Translate a theorem
result = leanaide_translate_theorem(
    theorem_text="There are infinitely many primes"
)

# 4. Use the result
if result['success']:
    print(f"Success: {result['lean_code']}")
else:
    print(f"Error: {result.get('error')}")
```

---

## Additional Resources

- **LeanAide README**: `LeanAide/README.md` - Complete LeanAide documentation
- **LeanAide Server**: `LeanAide/leanaide_server.py` - Server implementation
- **API Server**: `LeanAide/server/api_server.py` - HTTP API endpoint
- **ROMA MCP Tools**: `roma_mcp_tools.py` - Similar pattern for ROMA integration
- **ACE MCP Tools**: `ace_mcp_tools.py` - Similar pattern for ACE integration

---

## Version History

- **v1.0.0** (2025-12-30): Initial release with 8 MCP tools
  - Full LeanAide JSON API coverage
  - Async support
  - Comprehensive error handling
  - Thread-safe tool registry
  - Security hardening

---

## License

This module follows the same license as the OpenEvolve project.
