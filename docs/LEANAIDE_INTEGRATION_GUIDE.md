# LeanAide Integration Guide for OpenEvolve Workflows

## Overview

This guide describes the integration of LeanAide formal verification system into the OpenEvolve Sovereign-Grade Decomposition Workflow. LeanAide provides formal mathematical verification capabilities using the Lean 4 theorem prover, enabling rigorous verification of mathematical problems and solutions.

## Table of Contents

1. [Introduction](#introduction)
2. [Integration Points](#integration-points)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)

---

## Introduction

LeanAide is an AI-powered formal verification system that uses Lean 4 to:
- Translate natural-language mathematical statements into formal Lean 4 code
- Generate and verify formal proofs
- Elaborate and check Lean 4 code for correctness
- Answer mathematical queries with formal verification

The integration allows OpenEvolve workflows to optionally use LeanAide for formal verification of mathematical problems at two key stages:
- **Stage 3C (Gold Team Gauntlet)**: Verify individual sub-problem solutions
- **Stage 5 (Final Verification)**: Verify the final integrated solution

### Key Features

- **Automatic mathematical problem detection** - Automatically detects if a problem is mathematical
- **Graceful fallback** - Falls back to standard verification for non-mathematical problems
- **Configurable confidence thresholds** - Control verification strictness
- **Parallel batch verification** - Verify multiple sub-problems concurrently
- **Comprehensive error handling** - Robust error handling with detailed logging

---

## Integration Points

### Stage 3C: Gold Team Gauntlet (Sub-Problem Verification)

The `verify_sub_problem_with_leanaide()` function in `workflow_stage_functions.py` provides LeanAide verification for individual sub-problem solutions.

**Function Signature:**
```python
def verify_sub_problem_with_leanaide(
    sub_problem: 'SubProblem',
    solution_attempt: 'SolutionAttempt',
    workflow_state: 'WorkflowState'
) -> 'VerificationReport'
```

**Usage in Workflow:**
```python
from workflow_stage_functions import verify_sub_problem_with_leanaide

# After generating a solution for a sub-problem
verification_report = verify_sub_problem_with_leanaide(
    sub_problem=sub_problem,
    solution_attempt=solution_attempt,
    workflow_state=workflow_state
)

# Check if verification passed
if verification_report.is_approved:
    # Solution passed formal verification
    pass
else:
    # Solution needs refinement
    pass
```

### Stage 5: Final Verification

The `verify_final_solution_with_leanaide()` function provides LeanAide verification for the final integrated solution.

**Function Signature:**
```python
def verify_final_solution_with_leanaide(
    integrated_solution: str,
    workflow_state: 'WorkflowState'
) -> 'VerificationReport'
```

**Usage in Workflow:**
```python
from workflow_stage_functions import verify_final_solution_with_leanaide

# After assembling the final solution
final_verification = verify_final_solution_with_leanaide(
    integrated_solution=final_solution,
    workflow_state=workflow_state
)

# Check final verification
if final_verification.is_approved:
    # Final solution approved
    pass
else:
    # Trigger self-healing loop
    pass
```

---

## Configuration

### WorkflowState Parameters

Configure LeanAide verification through `WorkflowState` parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `leanaide_enabled` | `bool` | `False` | Enable LeanAide formal verification |
| `leanaide_host` | `str` | `"localhost"` | LeanAide server hostname |
| `leanaide_port` | `int` | `7654` | LeanAide server port |
| `leanaide_confidence_threshold` | `float` | `0.7` | Minimum confidence for verification success (0.0-1.0) |
| `leanaide_auto_detect_math` | `bool` | `True` | Automatically detect mathematical problems |
| `leanaide_require_formal_proof` | `bool` | `False` | Require formal proof generation |
| `leanaide_store_proofs` | `bool` | `True` | Store generated proofs in results |
| `leanaide_verification_method` | `Literal` | `"standard_primary"` | Verification priority: `"leanaide_only"`, `"leanaide_primary"`, `"standard_primary"` |
| `leanaide_timeout` | `int` | `300` | Timeout for verification in seconds |

### Configuration Example

```python
# In workflow initialization
workflow_state = WorkflowState(
    workflow_id="workflow_001",
    workflow_type="sovereign_grade_decomposition",
    problem_statement="Prove the infinitude of prime numbers",
    current_stage="Stage 3",
    # LeanAide configuration
    leanaide_enabled=True,
    leanaide_host="localhost",
    leanaide_port=7654,
    leanaide_confidence_threshold=0.8,
    leanaide_auto_detect_math=True,
    leanaide_require_formal_proof=False,
    leanaide_store_proofs=True,
    leanaide_verification_method="leanaide_primary",
    leanaide_timeout=300
)
```

### Verification Method Priority

The `leanaide_verification_method` parameter controls how LeanAide verification is used:

- **`leanaide_only`**: Use only LeanAide verification (fails if LeanAide unavailable)
- **`leanaide_primary`**: Try LeanAide first, fall back to standard verification on error
- **`standard_primary`**: Use standard verification by default, LeanAide as enhancement (default)

---

## Usage Examples

### Example 1: Basic Sub-Problem Verification

```python
import asyncio
from workflow_stage_functions import verify_sub_problem_with_leanaide
from workflow_structures import SubProblem, SolutionAttempt, WorkflowState

# Create workflow state with LeanAide enabled
workflow_state = WorkflowState(
    workflow_id="example_001",
    workflow_type="decomposition",
    problem_statement="Prove that sqrt(2) is irrational",
    current_stage="Stage 3C",
    leanaide_enabled=True,
    leanaide_confidence_threshold=0.7
)

# Create sub-problem
sub_problem = SubProblem(
    id="sp_001",
    description="Prove irrationality of square root of 2",
    gold_team_gauntlet_name="leanaide_verification"
)

# Create solution attempt
solution = SolutionAttempt(
    sub_problem_id="sp_001",
    content="Assume for contradiction that √2 is rational...",
    generated_by_model="gpt-4",
    timestamp=time.time()
)

# Verify with LeanAide
verification_report = verify_sub_problem_with_leanaide(
    sub_problem=sub_problem,
    solution_attempt=solution,
    workflow_state=workflow_state
)

print(f"Verification passed: {verification_report.is_approved}")
print(f"Confidence score: {verification_report.average_score}")
print(f"Summary: {verification_report.summary}")
```

### Example 2: Final Solution Verification

```python
from workflow_stage_functions import verify_final_solution_with_leanaide

# After assembling final solution
final_solution = """
Theorem: There are infinitely many prime numbers.

Proof: Assume there are finitely many primes p1, p2, ..., pn.
Consider N = p1 * p2 * ... * pn + 1.
N is not divisible by any prime, so N must be prime itself,
contradicting the assumption that p1,...,pn are all primes.
Therefore, there are infinitely many primes.
"""

final_verification = verify_final_solution_with_leanaide(
    integrated_solution=final_solution,
    workflow_state=workflow_state
)

if final_verification.is_approved:
    print("Final solution verified with LeanAide!")
else:
    print("Verification failed - refinement needed")
    print(final_verification.summary)
```

### Example 3: Direct LeanAide Integration

```python
import asyncio
from leanaide_workflow_integration import (
    LeanAideWorkflowIntegrator,
    LeanAideWorkflowConfig,
    create_standard_leanaide_config
)

async def verify_solution():
    # Create configuration
    config = create_standard_leanaide_config(
        host="localhost",
        port=7654,
        confidence_threshold=0.7
    )

    # Create integrator
    integrator = LeanAideWorkflowIntegrator(config)

    try:
        # Initialize connection
        if not await integrator.initialize():
            print("Failed to connect to LeanAide server")
            return

        # Verify a sub-problem
        result = await integrator.verify_sub_problem_solution(
            sub_problem_id="sp_001",
            problem_statement="Prove that the square root of 2 is irrational",
            solution_content="Assume for contradiction that √2 = a/b where a,b are integers..."
        )

        print(f"Success: {result.success}")
        print(f"Is Mathematical: {result.is_mathematical}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Lean Code:\n{result.lean_code}")

        if result.formal_proof:
            print(f"Formal Proof:\n{result.formal_proof}")

    finally:
        await integrator.close()

# Run the async verification
asyncio.run(verify_solution())
```

### Example 4: Batch Verification

```python
import asyncio
from leanaide_workflow_integration import LeanAideWorkflowIntegrator, LeanAideWorkflowConfig

async def batch_verify():
    config = LeanAideWorkflowConfig(enabled=True, host="localhost", port=7654)
    integrator = LeanAideWorkflowIntegrator(config)

    try:
        await integrator.initialize()

        # Prepare sub-problems
        sub_problems = [
            {
                "id": "sp_001",
                "description": "Prove the intermediate value theorem",
                "solution": "Let f be continuous on [a,b]..."
            },
            {
                "id": "sp_002",
                "description": "Design sorting algorithm",
                "solution": "Implement quicksort with pivot selection..."
            }
        ]

        # Batch verify
        results = await integrator.batch_verify_sub_problems(sub_problems)

        # Process results
        for sp_id, result in results.items():
            print(f"{sp_id}: math={result.is_mathematical}, "
                  f"success={result.success}, "
                  f"confidence={result.confidence_score:.2f}")

    finally:
        await integrator.close()

asyncio.run(batch_verify())
```

---

## API Reference

### LeanAideWorkflowIntegrator

Main class for LeanAide workflow integration.

#### Methods

##### `__init__(config: Optional[LeanAideWorkflowConfig] = None)`

Initialize the integrator with optional configuration.

##### `async initialize() -> bool`

Initialize connection to LeanAide server. Returns `True` if successful.

##### `async verify_sub_problem_solution(...) -> LeanAideVerificationResult`

Verify a sub-problem solution.

**Parameters:**
- `sub_problem_id` (str): ID of the sub-problem
- `problem_statement` (str): Original problem statement
- `solution_content` (str): Solution to verify
- `verification_requirements` (Optional[Dict]): Additional requirements

**Returns:** `LeanAideVerificationResult`

##### `async verify_final_solution(...) -> LeanAideVerificationResult`

Verify the final integrated solution.

**Parameters:**
- `problem_statement` (str): Original problem statement
- `final_solution` (str): Final solution to verify
- `sub_problems` (List[Dict]): List of sub-problems with solutions
- `verification_requirements` (Optional[Dict]): Additional requirements

**Returns:** `LeanAideVerificationResult`

##### `async batch_verify_sub_problems(sub_problems: List[Dict]) -> Dict[str, LeanAideVerificationResult]`

Verify multiple sub-problems in parallel.

**Parameters:**
- `sub_problems`: List of sub-problem dictionaries

**Returns:** Dictionary mapping sub-problem IDs to results

##### `async close()`

Close the LeanAide client connection.

### LeanAideVerificationResult

Data class containing verification results.

**Attributes:**
- `success` (bool): Whether verification passed
- `is_mathematical` (bool): Whether the problem is mathematical
- `confidence_score` (float): Confidence score (0.0-1.0)
- `verification_method` (str): Method used
- `lean_code` (Optional[str]): Generated Lean 4 code
- `formal_proof` (Optional[str]): Generated formal proof
- `errors` (List[str]): List of errors
- `warnings` (List[str]): List of warnings
- `metadata` (Dict[str, Any]): Additional metadata
- `execution_time` (float): Execution time in seconds

### LeanAideWorkflowConfig

Configuration for LeanAide integration.

**Attributes:**
- `enabled` (bool): Enable LeanAide verification
- `host` (str): Server hostname
- `port` (int): Server port
- `timeout` (float): Request timeout
- `max_retries` (int): Maximum retry attempts
- `auto_detect_math` (bool): Auto-detect mathematical problems
- `fallback_to_standard` (bool): Fall back to standard verification
- `confidence_threshold` (float): Minimum confidence for success
- `require_formal_proof` (bool): Require formal proof generation
- `store_proofs` (bool): Store generated proofs

### MathematicalProblemDetector

Detects whether problems are mathematical.

#### Methods

##### `is_mathematical_problem(problem_statement: str, solution_content: Optional[str] = None) -> Tuple[bool, float]`

Detect if a problem is mathematical.

**Returns:** Tuple of (is_mathematical, confidence_score)

---

## Troubleshooting

### LeanAide Server Not Responding

**Problem:** Connection timeout or "server not available" errors.

**Solutions:**
1. Verify LeanAide server is running: `curl http://localhost:7654/`
2. Check server logs for errors
3. Verify correct host and port in configuration
4. Check firewall settings

### Import Errors

**Problem:** `ImportError: No module named 'leanaide_client'`

**Solutions:**
1. Ensure `leanaide_client.py` is in the Python path
2. Install dependencies: `pip install aiohttp`
3. Check file exists in correct location

### Verification Always Fails

**Problem:** All verifications return `success=False`

**Solutions:**
1. Lower `leanaide_confidence_threshold` (e.g., from 0.8 to 0.6)
2. Check problem is actually mathematical (check `is_mathematical` flag)
3. Review errors in `LeanAideVerificationResult.errors`
4. Enable debug logging for more details

### Non-Mathematical Problems Rejected

**Problem:** Non-mathematical problems marked as mathematical

**Solutions:**
1. Set `leanaide_auto_detect_math=False` to disable auto-detection
2. Adjust mathematical keywords in configuration
3. Set `leanaide_verification_method="standard_primary"` for mixed problems

### Performance Issues

**Problem:** Verification takes too long

**Solutions:**
1. Reduce `leanaide_timeout` to fail fast on unresponsive server
2. Use batch verification for multiple sub-problems
3. Disable formal proof generation: `leanaide_require_formal_proof=False`
4. Consider using standard verification for non-critical problems

---

## Best Practices

1. **Start with Standard Verification**: Begin with `leanaide_verification_method="standard_primary"` to establish baseline performance
2. **Gradual Rollout**: Enable LeanAide for specific mathematical sub-problems first
3. **Monitor Performance**: Track execution times and success rates
4. **Set Appropriate Thresholds**: Choose confidence thresholds based on problem criticality
5. **Handle Errors Gracefully**: Always check `is_mathematical` flag and `success` status
6. **Use Batch Verification**: Verify multiple sub-problems in parallel for efficiency
7. **Store Proofs**: Enable `leanaide_store_proofs=True` for audit trail
8. **Fallback Strategy**: Configure appropriate fallback behavior for production use

---

## Additional Resources

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [LeanAide Repository](https://github.com/yangky11/leanaide)
- [OpenEvolve Workflow Documentation](./Decomposition_Workflow.md)
- [Workflow Structures Reference](./workflow_structures.py)
- [Workflow Stage Functions](./workflow_stage_functions.py)

---

## New Integration Components (v1.1)

### Updated Client (leanaide_client.py)

Production-ready async client with connection pooling, retries, and comprehensive error handling.

```python
from leanaide_client import LeanAideClient, LeanAideConfig
import asyncio

async def example():
    config = LeanAideConfig(
        host="localhost",
        port=7654,
        timeout=6000.0,
        max_connections=100
    )

    async with LeanAideClient(config) as client:
        # Health check
        is_healthy = await client.health_check()

        # Translate theorem
        result = await client.translate_thm(
            "There are infinitely many prime numbers"
        )

        if result.success:
            print(f"Lean code: {result.data.get('lean_code')}")

        # Batch operations
        theorems = ["Theorem 1", "Theorem 2", "Theorem 3"]
        results = await client.batch_translate_theorems(theorems)

asyncio.run(example())
```

### CrewAI Bridge (leanaide_crewai_bridge.py)

Complete 6-phase workflow integration with ticket tracking.

```python
from leanaide_crewai_bridge import (
    LeanAideCrewAIBridge,
    LeanAideConfig,
    MathematicalDomain,
    ExecutionMode
)
import asyncio

async def full_workflow():
    config = LeanAideConfig(
        host="localhost",
        port=7654,
        enable_tickets=True,
        ticket_base_url="http://localhost:8000"
    )

    bridge = LeanAideCrewAIBridge(config)

    try:
        # Run complete 6-phase workflow
        result = await bridge.execute_full_workflow(
            "Prove that there are infinitely many prime numbers"
        )

        if result['workflow_success']:
            print("Workflow completed successfully!")

            # Access phase results
            phase1 = result['phases']['phase_1']
            print(f"Domain: {phase1['metadata']['domain']}")
            print(f"Components: {phase1['metadata']['num_components']}")
        else:
            print(f"Workflow failed: {result.get('failure_phase')}")

    finally:
        await bridge.cleanup()

asyncio.run(full_workflow())
```

### MCP Tools (leanaide_mcp_tools.py)

Model Context Protocol tools for agent integration.

Available tools:
- `leanaide_translate_theorem`
- `leanaide_prove_theorem`
- `leanaide_verify_code`
- `leanaide_math_query`
- `leanaide_generate_docs`
- `leanaide_extract_components`
- `leanaide_batch_translate`

```python
from leanaide_mcp_tools import (
    leanaide_translate_theorem,
    leanaide_math_query,
    leanaide_verify_code
)

# Translate theorem
result = leanaide_translate_theorem(
    theorem_text="There are infinitely many prime numbers"
)

# Math query
answer = leanaide_math_query(
    query="What is the fundamental theorem of algebra?",
    n=3
)

# Verify code
verification = leanaide_verify_code(
    lean_code="theorem add_comm (a b : Nat) : a + b = b + a := by simp"
)
```

### Mathematical Problem Detector

Automatic detection and classification of mathematical content.

```python
from leanaide_crewai_bridge import MathematicalProblemDetector

detector = MathematicalProblemDetector()

# Detect mathematical content
has_math = detector.detect_mathematical_content(
    "Prove there are infinitely many primes"
)
print(f"Has math: {has_math}")

# Classify domain
domain = detector.classify_domain(
    "Groups, rings, and fields are algebraic structures"
)
print(f"Domain: {domain.value}")  # ALGEBRA

# Extract components
components = detector.extract_components(text)
for component in components:
    print(f"{component.type}: {component.name}")
```

### Supported Mathematical Domains

- `ALGEBRA` - Groups, rings, fields, vector spaces
- `ANALYSIS` - Limits, derivatives, integrals
- `TOPOLOGY` - Topological spaces, metrics
- `NUMBER_THEORY` - Primes, divisibility, modular arithmetic
- `COMBINATORICS` - Permutations, combinations, graphs
- `GEOMETRY` - Triangles, circles, polygons
- `LOGIC` - Propositional logic, predicates
- `SET_THEORY` - Sets, cardinality, infinity
- `GENERAL` - Mixed mathematical content

---

## Evolutionary LeanAide Integration

### Overview

Evolutionary LeanAide extends the basic integration with advanced evolutionary algorithms for automated proof generation. Instead of relying on single proof attempts, evolutionary approaches use population-based search, adversarial competition, and self-play to systematically explore the proof space.

### Key Components

1. **Genetic Evolution** (`leanaide_evolution.py`) - Population-based genetic algorithm
2. **Adversarial Evolution** (`leanaide_adversarial.py`) - Red team vs blue team competition
3. **Self-Play** (`leanaide_selfplay.py`) - AlphaZero-style self-improvement
4. **Strategy Library** (`leanaide_strategies.py`) - Reusable proof patterns

### Integration with Workflow

Evolutionary LeanAide integrates with OpenEvolve workflows at additional stages:

**Stage 1 (Decomposition):** Generate diverse proof strategies
```python
from leanaide_evolution import evolve_proof

# Evolve proof strategies during decomposition
result = await evolve_proof(
    theorem=sub_problem.mathematical_statement,
    max_generations=30,
    population_size=20
)
```

**Stage 3A (Solution Generation):** Evolve proofs for mathematical sub-problems
```python
# Use genetic evolution for broad search
genetic_result = await evolve_proof(theorem, max_generations=40)

# Use adversarial evolution for robustness testing
from leanaide_adversarial import LeanAdversarialEvolution
evolution = LeanAdversarialEvolution()
proof, rounds, stats = await evolution.run_adversarial_evolution(theorem, rounds=10)
```

**Stage 3B (Evaluation Team):** Adversarial testing of proof robustness
```python
# Red team critique
red_team = LeanRedTeamAgent()
critiques = red_team.critique_proof(strategy, theorem, context)
```

**Stage 5 (Final Verification):** Self-play improvement of final proofs
```python
from leanaide_selfplay import LeanSelfPlayEngine

engine = LeanSelfPlayEngine()
final_proof = await engine.run_self_play(theorem, games=15)
```

**Stage 6 (Knowledge Extraction):** Extract successful strategies for learning
```python
# Extract learned strategies
strategies = engine.agent.known_strategies
for strategy in strategies:
    if strategy.success_rate > 0.8:
        # Store successful strategy in knowledge base
        knowledge_base.add_strategy(strategy)
```

### Configuration

**Enable Evolutionary Features:**
```python
# In workflow configuration
workflow_config = {
    "enable_evolutionary_leanaide": True,
    "evolutionary_approach": "genetic",  # or "adversarial", "selfplay", "hybrid"
    "genetic_params": {
        "population_size": 30,
        "max_generations": 50,
        "mutation_rate": 0.1
    },
    "adversarial_params": {
        "rounds": 10,
        "approaches": ["constructive", "classical", "computational"]
    },
    "selfplay_params": {
        "games": 15,
        "buffer_capacity": 5000
    }
}
```

### API Reference

**Genetic Evolution:**
```python
from leanaide_evolution import evolve_proof, LeanProofEvolutionEngine

# Convenience function
result = await evolve_proof(
    theorem="∀ n : Nat, n + 0 = n",
    max_generations=30,
    population_size=20
)

# Full engine
engine = LeanProofEvolutionEngine(
    theorem=theorem,
    population_size=50,
    max_generations=100,
    parallel_evaluation=True
)
result = await engine.evolve()
await engine.close()
```

**Adversarial Evolution:**
```python
from leanaide_adversarial import LeanAdversarialEvolution, evolve_lean_proof

# Convenience function
result = evolve_lean_proof(theorem, rounds=10)

# Full engine
evolution = LeanAdversarialEvolution(api_key="your-key")
proof, rounds, stats = await evolution.run_adversarial_evolution(theorem, rounds=12)
```

**Self-Play:**
```python
from leanaide_selfplay import LeanSelfPlayEngine

engine = LeanSelfPlayEngine(
    leanaide_url="http://localhost:7654",
    buffer_capacity=10000
)

# Single theorem
proof = await engine.run_self_play(theorem, games=20)

# Batch training
results = await engine.run_batch_self_play(theorems, games_per_theorem=15)
metrics = await engine.train_from_buffer(batch_size=32, iterations=100)

await engine.close()
```

### Performance Considerations

**Evolutionary approaches require more resources:**

| Approach | Time (parallel) | Verifications | Best For |
|----------|----------------|---------------|----------|
| Basic LeanAide | 1-5 min | 1 | Simple theorems |
| Genetic | 5-30 min | 500-5000 | Broad search |
| Adversarial | 10-40 min | 50-200 | Robustness testing |
| Self-Play | 30-120 min | 100-1000 | Continuous improvement |
| Hybrid | 30-90 min | 650-5700 | Maximum quality |

**Resource Optimization:**
```python
# For faster results
evolution_config = {
    "population_size": 20,
    "max_generations": 30,
    "parallel_evaluation": True,
    "cache_enabled": True
}

# For better results
evolution_config = {
    "population_size": 50,
    "max_generations": 100,
    "mutation_rate": 0.15,
    "verification_weight": 12.0
}
```

### When to Use Evolutionary Approaches

**Use Evolutionary When:**
- Theorems have multiple possible proof approaches
- Proof space is large and complex
- Need maximum robustness and verification
- Processing batch of related theorems
- Research-level or difficult theorems

**Use Basic LeanAide When:**
- Simple, straightforward theorems
- Time constraints
- Limited computational resources
- Known proof strategy

### Documentation

For detailed information on evolutionary LeanAide:
- **Usage Guide:** `LEANAIDE_EVOLUTIONARY_GUIDE.md`
- **API Reference:** `LEANAIDE_EVOLUTIONARY_API.md`
- **Examples:** `LEANAIDE_EVOLUTIONARY_EXAMPLES.md`

---

## Version History

### v1.1 (2025-12-30)
- Added production-ready async client (leanaide_client.py)
- Added complete CrewAI bridge (leanaide_crewai_bridge.py)
- Added MCP tools for agents (leanaide_mcp_tools.py)
- Added mathematical problem detector
- Added batch operation support
- Added comprehensive test coverage

### v1.0 (Initial Release)
- Basic LeanAide integration
- Sub-problem verification
- Final solution verification
- Mathematical problem detection

