# Lean 4 Integration - Quick Reference

## Quick Start

```python
import asyncio
from lean4_integration import create_lean4_verification_engine, AutoformalizationEngine

async def main():
    # Create engine
    engine = create_lean4_verification_engine("http://localhost:7654")

    # Autoformalize
    auto = AutoformalizationEngine(engine.client, engine.cache)
    result = await auto.autoformalize("For all n, n + 0 = n", "theorem", "add_zero")
    print(result.lean_code)

    # Verify
    verification = await engine.verify_mathematical_solution(result.lean_code)
    print(f"Success: {verification.success}")

    await engine.close()

asyncio.run(main())
```

## Common Tasks

### 1. Autoformalize Theorem

```python
result = await autoformalization.autoformalize(
    natural_language="For all natural numbers n, n + 0 = n",
    statement_type="theorem",
    name="add_zero"
)
```

### 2. Verify Code

```python
result = await engine.verify_mathematical_solution(lean_code)
print(f"Success: {result.success}")
print(f"Time: {result.verification_time:.2f}s")
```

### 3. Search Similar Theorems

```python
from lean4_integration import ProofSearchEngine
proof_search = ProofSearchEngine(engine.client, engine.cache)

results = await proof_search.search_related_theorems(
    query="additive identity",
    num_results=5
)

for r in results:
    print(f"{r.name}: {r.distance:.4f}")
```

### 4. Batch Verification

```python
theorems = [code1, code2, code3]
results = await engine.batch_verify(theorems)

for i, r in enumerate(results):
    print(f"{i+1}. {'✓' if r.success else '✗'}")
```

### 5. Full Pipeline

```python
from lean4_integration import MathematicalProblemProcessor

processor = MathematicalProblemProcessor(
    verification_engine=engine,
    autoformalization_engine=auto,
    proof_search_engine=proof_search
)

result = await processor.process_mathematical_problem(
    problem_description="Prove that n + 0 = n for all natural numbers n",
    enable_proof_search=True,
    enable_dependency_analysis=True
)
```

## Configuration

```python
from lean4_integration import Lean4ServerConfig, Lean4VerificationConfig

server_config = Lean4ServerConfig(
    host="localhost",
    port=7654,
    timeout=600,
    enable_simulation_fallback=True
)

verification_config = Lean4VerificationConfig(
    enable_caching=True,
    cache_ttl_seconds=3600,
    cache_file=".leanaide_cache/verification_cache.db"
)
```

## Result Objects

### VerificationResult

```python
result.success              # bool
result.proof                # str
result.errors               # List[str]
result.verification_time    # float
result.server_available     # bool
result.used_fallback        # bool
result.lean_code            # str
```

### AutoformalizationResult

```python
result.success              # bool
result.lean_code            # str
result.theorem_name         # str
result.errors               # List[str]
result.warnings             # List[str]
result.elaborated           # bool
result.server_available     # bool
```

### SimilaritySearchResult

```python
result.name                 # str
result.type                 # str
result.doc_string           # str
result.distance             # float
result.module               # str
result.is_prop              # bool
```

## Error Handling

```python
from lean4_integration import (
    Lean4VerificationError,
    LeanAideServerError,
    LeanAideConnectionError
)

try:
    result = await engine.verify_mathematical_solution(code)
except LeanAideConnectionError:
    print("Server unavailable")
except LeanAideServerError as e:
    print(f"Server error: {e}")
except Lean4VerificationError as e:
    print(f"Verification failed: {e}")
```

## LeanAide Server

### Start Server

```bash
cd LeanAide
python leanaide_server.py
```

### Server Endpoints

- `POST /` - Main endpoint for translate tasks
- `POST /run-sim-search` - Similarity search
- `GET /` - Health check

### Server Tasks

```json
// translate_thm
{
  "task": "translate_thm",
  "theorem_text": "For all n, n + 0 = n"
}

// translate_thm_detailed
{
  "task": "translate_thm_detailed",
  "theorem_text": "For all n, n + 0 = n",
  "theorem_name": "add_zero"
}

// translate_def
{
  "task": "translate_def",
  "definition_text": "A group is..."
}

// similarity_search
{
  "query": "additive identity",
  "num": 10,
  "descField": "docString"
}
```

## Cache Operations

```python
# Check cache
cached = engine.cache.get_verification(lean_code)

# Store in cache
engine.cache.set_verification(lean_code, result)

# Cleanup expired
engine.cache.cleanup_expired()
```

## Testing

```bash
# Run all tests
python test_lean4_integration_enhanced.py

# Test with custom server
python test_lean4_integration_enhanced.py http://localhost:8080
```

## Dependencies

```
aiohttp>=3.8.0
sqlite3 (standard library)
asyncio (standard library)
```

## File Structure

```
lean4_integration.py              # Main module
test_lean4_integration_enhanced.py # Test suite
LEAN4_INTEGRATION_GUIDE.md        # Full guide
LEAN4_QUICK_REFERENCE.md          # This file
.leanaide_cache/                  # Cache directory
└── verification_cache.db         # SQLite cache
```

## Exported Classes

```python
from lean4_integration import (
    # Exceptions
    Lean4VerificationError,
    LeanAideServerError,
    LeanAideConnectionError,

    # Result Classes
    VerificationResult,
    SimilaritySearchResult,
    AutoformalizationResult,
    DependencyInfo,

    # Data Classes
    MathematicalComponent,
    Lean4ServerConfig,
    Lean4VerificationConfig,

    # Core Classes
    VerificationCache,
    LeanAideClient,
    Lean4VerificationEngine,
    AutoformalizationEngine,
    ProofSearchEngine,
    DependencyGraphAnalyzer,

    # Processing Classes
    MathematicalProblemDetector,
    MathematicalProblemProcessor,
    Lean4MathematicalKnowledge,

    # Helper Functions
    create_lean4_verification_engine,
    detect_and_verify_mathematical_problems,
    verify_mathematical_solution_async
)
```

## Common Patterns

### With Statement (Auto Cleanup)

```python
async with engine:
    result = await engine.verify_mathematical_solution(code)
    # Auto closes on exit
```

### Retry Logic

```python
from asyncio import sleep

max_retries = 3
for attempt in range(max_retries):
    try:
        result = await engine.verify_mathematical_solution(code)
        break
    except LeanAideConnectionError:
        if attempt < max_retries - 1:
            await sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

### Progress Tracking

```python
import time

start = time.time()
result = await engine.verify_mathematical_solution(code)
elapsed = time.time() - start

print(f"Verified in {elapsed:.2f}s")
print(f"Server: {'Yes' if result.server_available else 'No'}")
print(f"Fallback: {'Yes' if result.used_fallback else 'No'}")
```

## Performance Tips

1. **Enable caching** for repeated verifications
2. **Use batch operations** for multiple theorems
3. **Reuse engine instances** across operations
4. **Adjust timeout** based on complexity
5. **Monitor cache hit rate** with logging

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Server unavailable | Check `http://localhost:7654` |
| Timeout errors | Increase `timeout` in config |
| Import errors | Install dependencies: `pip install aiohttp` |
| Cache errors | Delete `.leanaide_cache/` directory |
| Slow verification | Check server resources and complexity |

## Resources

- Full Guide: `LEAN4_INTEGRATION_GUIDE.md`
- Tests: `test_lean4_integration_enhanced.py`
- LeanAide: https://github.com/yangky11/LeanAide
- Lean 4 Docs: https://leanprover.github.io/lean4/doc/
