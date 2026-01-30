# Enhanced Lean 4 Integration Guide

## Overview

The enhanced `lean4_integration.py` module provides comprehensive integration with [LeanAide](https://github.com/yangky11/LeanAide), a powerful tool for translating natural language mathematics to formal Lean 4 code and verifying proofs.

## Key Features

### 1. Real LeanAide Server Integration
- **No more simulation**: Direct integration with actual LeanAide server
- **Autoformalization**: Natural language → Lean 4 code translation
- **Proof verification**: Formal verification using Lean 4 theorem prover
- **Similarity search**: Find related theorems in Mathlib

### 2. Advanced Capabilities
- **Autoformalization Pipeline**: Convert natural language math to formal Lean code
- **Proof Search**: Retrieve similar theorems to aid in proof development
- **Batch Verification**: Verify multiple theorems concurrently
- **Dependency Analysis**: Analyze theorem dependencies and relationships
- **SQLite Caching**: Persistent cache for verified theorems and searches

### 3. Production-Ready Features
- **Fallback Mode**: Graceful degradation to simulation when server unavailable
- **Error Handling**: Comprehensive error handling for server failures
- **Connection Management**: Automatic health checks and connection pooling
- **Performance**: Caching layer for faster subsequent verifications

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│             lean4_integration.py                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ LeanAideClient   │  │VerificationCache │                │
│  │  - HTTP client   │  │  - SQLite cache  │                │
│  │  - Health check  │  │  - TTL support   │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
│           │                     │                            │
│  ┌────────▼─────────────────────▼──────────┐                │
│  │     Lean4VerificationEngine              │                │
│  │  - verify_mathematical_solution()       │                │
│  │  - batch_verify()                       │                │
│  └────────┬────────────────────────────────┘                │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────┐      │
│  │     AutoformalizationEngine                        │      │
│  │  - autoformalize()  (NL → Lean)                   │      │
│  └────────┬──────────────────────────────────────────┘      │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────┐      │
│  │     ProofSearchEngine                             │      │
│  │  - search_related_theorems()                      │      │
│  │  - find_proof_strategy()                          │      │
│  └────────┬──────────────────────────────────────────┘      │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────┐      │
│  │     MathematicalProblemProcessor                  │      │
│  │  - Full pipeline integration                      │      │
│  └────────┬──────────────────────────────────────────┘      │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────┐      │
│  │     Lean4MathematicalKnowledge                    │      │
│  │  - Knowledge base management                      │      │
│  └───────────────────────────────────────────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LeanAide Server (port 7654)                   │
│  - translate_thm (NL → Lean)                               │
│  - translate_def (definitions)                             │
│  - similarity search (Mathlib search)                      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

1. **Python Dependencies**:
```bash
pip install aiohttp sqlite3
```

2. **LeanAide Server**:
```bash
cd LeanAide
python leanaide_server.py
```

The server will start on `http://localhost:7654` by default.

### Configuration

```python
from lean4_integration import (
    Lean4ServerConfig,
    Lean4VerificationConfig,
    create_lean4_verification_engine
)

# Server configuration
server_config = Lean4ServerConfig(
    host="localhost",
    port=7654,
    timeout=600,
    enable_simulation_fallback=True  # Use simulation if server unavailable
)

# Verification configuration
verification_config = Lean4VerificationConfig(
    enable_caching=True,
    cache_ttl_seconds=3600,  # 1 hour
    cache_file=".leanaide_cache/verification_cache.db"
)

# Create engine
engine = create_lean4_verification_engine(
    server_url="http://localhost:7654",
    server_config=server_config,
    config=verification_config
)
```

## Usage Examples

### Example 1: Autoformalization

Convert natural language mathematics to Lean code:

```python
import asyncio
from lean4_integration import (
    Lean4VerificationEngine,
    AutoformalizationEngine,
    Lean4ServerConfig,
    Lean4VerificationConfig
)

async def autoformalize_example():
    # Setup
    server_config = Lean4ServerConfig()
    verification_config = Lean4VerificationConfig()
    engine = Lean4VerificationEngine(
        "http://localhost:7654",
        server_config,
        verification_config
    )

    # Create autoformalization engine
    autoformalization = AutoformalizationEngine(engine.client, engine.cache)

    # Convert natural language to Lean
    result = await autoformalization.autoformalize(
        natural_language="For all natural numbers n, n + 0 = n",
        statement_type="theorem",
        name="add_zero"
    )

    if result.success:
        print("Generated Lean Code:")
        print(result.lean_code)
    else:
        print("Errors:", result.errors)

    await engine.close()

asyncio.run(autoformalize_example())
```

**Output**:
```lean
theorem add_zero : ∀ (n : Nat), n + 0 = n := by
  sorry
```

### Example 2: Mathematical Verification

Verify a mathematical solution:

```python
async def verification_example():
    engine = create_lean4_verification_engine()

    lean_code = """
    theorem mul_one (n : Nat) : n * 1 = n := by
      sorry
    """

    result = await engine.verify_mathematical_solution(lean_code)

    print(f"Success: {result.success}")
    print(f"Verification Time: {result.verification_time:.2f}s")
    print(f"Server Available: {result.server_available}")
    print(f"Used Fallback: {result.used_fallback}")

    if result.errors:
        print("Errors:", result.errors)

    await engine.close()

asyncio.run(verification_example())
```

### Example 3: Similarity Search

Find related theorems in Mathlib:

```python
async def similarity_search_example():
    engine = create_lean4_verification_engine()
    proof_search = ProofSearchEngine(engine.client, engine.cache)

    # Search for related theorems
    results = await proof_search.search_related_theorems(
        query="additive identity",
        num_results=5,
        search_field="docString"
    )

    for result in results:
        print(f"Theorem: {result.name}")
        print(f"Type: {result.type}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Documentation: {result.doc_string[:100]}...")
        print()

    await engine.close()

asyncio.run(similarity_search_example())
```

### Example 4: Full Pipeline

Process a mathematical problem through the complete pipeline:

```python
async def full_pipeline_example():
    # Setup all components
    server_config = Lean4ServerConfig()
    verification_config = Lean4VerificationConfig()
    engine = Lean4VerificationEngine(
        "http://localhost:7654",
        server_config,
        verification_config
    )

    autoformalization = AutoformalizationEngine(engine.client, engine.cache)
    proof_search = ProofSearchEngine(engine.client, engine.cache)
    dependency_analyzer = DependencyGraphAnalyzer("./LeanAide")

    processor = MathematicalProblemProcessor(
        engine,
        autoformalization,
        proof_search,
        dependency_analyzer
    )

    # Process a problem
    problem = """
    Mathematical Problem: Additive Identity

    Theorem: For all natural numbers n, we have n + 0 = n.

    This theorem states that adding zero to any natural number
    returns the same natural number.
    """

    result = await processor.process_mathematical_problem(
        problem_description=problem,
        enable_proof_search=True,
        enable_dependency_analysis=True
    )

    print(f"Mathematical Content: {result['has_mathematical_content']}")
    print(f"Components Extracted: {result['components_extracted']}")
    print(f"Verification Success: {result['verification_result']['success']}")

    if result['proof_search_results']:
        print(f"Proof Search Confidence: {result['proof_search_results']['confidence']}")

    await engine.close()

asyncio.run(full_pipeline_example())
```

### Example 5: Batch Verification

Verify multiple theorems concurrently:

```python
async def batch_verification_example():
    engine = create_lean4_verification_engine()

    theorems = [
        "theorem thm1 (n : Nat) : n + 0 = n := by sorry",
        "theorem thm2 (n m : Nat) : n + m = m + n := by sorry",
        "theorem thm3 (n : Nat) : n * 1 = n := by sorry",
        "theorem thm4 (a b : Nat) : a ≤ b → a + 1 ≤ b + 1 := by sorry"
    ]

    results = await engine.batch_verify(theorems)

    for i, result in enumerate(results, 1):
        print(f"Theorem {i}: {'✓' if result.success else '✗'}")

    await engine.close()

asyncio.run(batch_verification_example())
```

## API Reference

### Classes

#### `LeanAideClient`

Client for communicating with LeanAide server.

**Methods**:
- `async check_server_health() -> bool`: Check if server is available
- `async translate_thm(theorem_text: str, theorem_name: Optional[str] = None) -> Dict`: Translate theorem
- `async translate_def(definition_text: str) -> Dict`: Translate definition
- `async similarity_search(query: str, num: int = 10, desc_field: str = "docString") -> List[SimilaritySearchResult]`: Search similar theorems

#### `Lean4VerificationEngine`

Handles verification requests.

**Methods**:
- `async verify_mathematical_solution(lean_code: str, timeout: Optional[int] = None) -> VerificationResult`: Verify solution
- `async batch_verify(lean_codes: List[str]) -> List[VerificationResult]`: Batch verification
- `async close()`: Clean up resources

#### `AutoformalizationEngine`

Autoformalization pipeline.

**Methods**:
- `async autoformalize(natural_language: str, statement_type: str = "theorem", name: Optional[str] = None) -> AutoformalizationResult`: Convert NL to Lean

#### `ProofSearchEngine`

Proof search and retrieval.

**Methods**:
- `async search_related_theorems(query: str, num_results: int = 10, search_field: str = "docString") -> List[SimilaritySearchResult]`: Search theorems
- `async find_proof_strategy(theorem_statement: str) -> Dict`: Find proof strategy

#### `MathematicalProblemProcessor`

Full pipeline integration.

**Methods**:
- `async process_mathematical_problem(problem_description: str, enable_proof_search: bool = True, enable_dependency_analysis: bool = True) -> Dict`: Process problem

### Data Classes

#### `VerificationResult`

Result of verification.

**Fields**:
- `success: bool`: Verification success
- `proof: str`: Verified proof
- `errors: List[str]`: Error messages
- `verification_time: float`: Time taken
- `proof_steps: List[str]`: Proof steps
- `complexity_score: float`: Complexity score
- `theorem_types: List[str]`: Theorem types
- `lean_code: str`: Lean code
- `elaborated_type: str`: Elaborated type
- `server_available: bool`: Server was available
- `used_fallback: bool`: Used fallback simulation

#### `AutoformalizationResult`

Result of autoformalization.

**Fields**:
- `success: bool`: Success status
- `lean_code: str`: Generated Lean code
- `theorem_name: str`: Theorem name
- `errors: List[str]`: Errors
- `warnings: List[str]`: Warnings
- `elaborated: bool`: Was elaborated
- `server_available: bool`: Server was available

#### `SimilaritySearchResult`

Result from similarity search.

**Fields**:
- `name: str`: Theorem name
- `type: str`: Theorem type
- `doc_string: str`: Documentation
- `distance: float`: Distance score
- `module: str`: Module name
- `is_prop: bool`: Is proposition

## LeanAide Server Tasks

The LeanAide server supports the following tasks:

### `translate_thm`
Translate natural language theorem to Lean.

**Request**:
```json
{
  "task": "translate_thm",
  "theorem_text": "For all natural numbers n, n + 0 = n"
}
```

**Response**:
```json
{
  "lean_code": "theorem ...",
  "type": "...",
  "errors": []
}
```

### `translate_thm_detailed`
Translate with theorem name.

**Request**:
```json
{
  "task": "translate_thm_detailed",
  "theorem_text": "For all natural numbers n, n + 0 = n",
  "theorem_name": "add_zero"
}
```

### `translate_def`
Translate definition.

**Request**:
```json
{
  "task": "translate_def",
  "definition_text": "A group is a set with an associative binary operation..."
}
```

### `/run-sim-search`
Similarity search endpoint.

**Request**:
```json
{
  "query": "additive identity",
  "num": 10,
  "descField": "docString"
}
```

## Caching

The module uses SQLite-based caching for:

1. **Verification Cache**: Stores verified theorems
2. **Similarity Cache**: Stores similarity search results
3. **Translation Cache**: Stores autoformalization results

Cache Configuration:
```python
verification_config = Lean4VerificationConfig(
    enable_caching=True,
    cache_ttl_seconds=3600,  # 1 hour TTL
    cache_file=".leanaide_cache/verification_cache.db"
)
```

Cache Cleanup:
```python
# Cleanup expired entries
engine.cache.cleanup_expired()
```

## Fallback Mode

When the LeanAide server is unavailable, the system can fall back to simulation mode:

```python
server_config = Lean4ServerConfig(
    enable_simulation_fallback=True  # Enable fallback
)
```

In fallback mode:
- Autoformalization generates basic templates
- Verification performs basic syntax checking
- Proof search returns empty results
- All results are marked with `used_fallback=True` and `server_available=False`

## Error Handling

The module provides comprehensive error handling:

```python
from lean4_integration import (
    Lean4VerificationError,
    LeanAideServerError,
    LeanAideConnectionError
)

try:
    result = await engine.verify_mathematical_solution(lean_code)
except LeanAideConnectionError:
    print("Cannot connect to LeanAide server")
except LeanAideServerError as e:
    print(f"Server error: {e}")
except Lean4VerificationError as e:
    print(f"Verification failed: {e}")
```

## Performance Tips

1. **Enable Caching**: Cache significantly improves performance for repeated verifications
2. **Batch Operations**: Use `batch_verify()` for multiple theorems
3. **Adjust Timeout**: Increase timeout for complex proofs
4. **Connection Pooling**: Reuse engine instances for multiple operations

## Testing

Run the comprehensive test suite:

```bash
python test_lean4_integration_enhanced.py
```

Tests include:
- Server connection check
- Autoformalization
- Similarity search
- Verification
- Batch verification
- Full pipeline
- Caching performance

## Troubleshooting

### Server Not Available

**Problem**: Cannot connect to LeanAide server

**Solutions**:
1. Check server is running: `ps aux | grep leanaide_server`
2. Verify port: `netstat -an | grep 7654`
3. Enable fallback mode in configuration

### Timeout Errors

**Problem**: Verification timeout

**Solutions**:
1. Increase timeout in `Lean4ServerConfig`
2. Simplify the theorem statement
3. Check server resources

### Import Errors

**Problem**: Cannot import lean4_integration

**Solutions**:
1. Ensure module is in Python path
2. Install dependencies: `pip install aiohttp`
3. Check Python version (3.7+ required)

## Integration with Workflow Engine

The enhanced Lean 4 integration is fully compatible with the Sovereign decomposition workflow:

```python
from lean4_integration import verify_mathematical_solution_async

# In a workflow stage
result = await verify_mathematical_solution_async(
    problem_statement="Prove that for all natural numbers n, n + 0 = n",
    solution_content="Proof: By definition of addition...",
    lean4_config=Lean4VerificationConfig()
)

if result.success:
    print("Solution verified!")
```

## Advanced Usage

### Custom Cache Implementation

```python
from lean4_integration import VerificationCache

class CustomCache(VerificationCache):
    def __init__(self, cache_file: str):
        super().__init__(cache_file)
        # Add custom initialization

    def get_verification(self, lean_code: str):
        # Custom cache retrieval logic
        result = super().get_verification(lean_code)
        # Add custom processing
        return result
```

### Custom Similarity Search

```python
from lean4_integration import ProofSearchEngine

class CustomProofSearch(ProofSearchEngine):
    async def search_related_theorems(self, query: str, **kwargs):
        results = await super().search_related_theorems(query, **kwargs)
        # Add custom filtering/ranking
        results = [r for r in results if r.distance < 0.5]
        return results
```

## Contributing

When contributing to the Lean 4 integration:

1. Maintain backward compatibility with existing interfaces
2. Add comprehensive error handling
3. Include caching for new operations
4. Update tests for new features
5. Document API changes

## License

This integration module is part of the OpenEvolve project.

## References

- [LeanAide GitHub](https://github.com/yangky11/LeanAide)
- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib](https://github.com/leanprover-community/mathlib4)
- [OpenEvolve Documentation](./README.md)

## Changelog

### Version 2.0 (Enhanced)
- Real LeanAide server integration
- Autoformalization pipeline
- Similarity search
- Batch verification
- Dependency analysis
- SQLite caching
- Fallback mode
- Comprehensive error handling

### Version 1.0 (Original)
- Basic simulation mode
- Simple verification
- Mathematical content detection
