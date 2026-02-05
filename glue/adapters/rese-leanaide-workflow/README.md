# LeanAide-RESE Workflow Adapter

AI-powered theorem proving integrated with RESE's 4-phase pipeline for formal verification and mathematical reasoning.

## Overview

This adapter integrates LeanAide's autoformalization and AI-guided proof search with RESE's Recursive Epistemic Solvability Engine, enabling:

- **Phase I - Epistemic Audit**: Autoformalize constraints and verify them
- **Phase II - Isomorphic Mapping**: Formalize and verify isomorphisms
- **Phase III - MCTS Refinement**: AI-guided hypothesis testing
- **Phase IV - Architectural Synthesis**: Prove efficacy claims formally

## Features

### Autoformalization
- Natural language to Lean 4 translation
- Automatic domain detection
- Phase-specific formalization strategies
- Batch processing support

### Proof Search
- MCTS-guided proof search
- Z3-LeanAide hybrid verification
- Auto tactics generation
- Counterexample detection

### Workflow Orchestration
- Problem classification
- Adaptive solver selection
- Stage-aware processing
- Comprehensive error handling

## Installation

### Prerequisites

- Python 3.11+
- LeanAide server (optional, simulation mode available)
- Z3 solver (optional, for hybrid verification)

### Setup

1. Clone the repository:
```bash
cd glue/adapters/rese-leanaide-workflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
export LEANAIDE_HOST=localhost
export LEANAIDE_PORT=7654
```

4. Verify installation:
```bash
chmod +x probes/check_leanaide_workflow.sh
./probes/check_leanaide_workflow.sh
```

## Usage

### Basic Usage

```python
import asyncio
from src.leanaide_rese_workflow import execute_workflow

async def main():
    result = await execute_workflow(
        problem_statement="Prove that for all natural numbers n, n + 0 = n"
    )

    print(f"Status: {result.overall_status}")
    print(f"Summary: {result.summary}")

asyncio.run(main())
```

### Phase I: Epistemic Audit

```python
from src.autoformalization_service import AutoformalizationService
from src.proof_search_service import ProofSearchService

# Create services
auto_service = AutoformalizationService()
proof_service = ProofSearchService()

# Autoformalize constraint
auto_result = await auto_service.autoformalize_phase_i(
    constraint_text="All prime numbers greater than 2 are odd",
    constraint_type="arithmetic"
)

# Search proof
proof_result = await proof_service.search_phase_i(
    lean_code=auto_result.lean_code
)

print(f"Lean code: {auto_result.lean_code}")
print(f"Proof found: {proof_result.proof_found}")
```

### Phase II: Isomorphic Mapping

```python
# Autoformalize isomorphism
auto_result = await auto_service.autoformalize_phase_ii(
    mapping_description="Structure-preserving bijection",
    source_domain="natural_numbers",
    target_domain="integers"
)

# Verify isomorphism
proof_result = await proof_service.search_phase_ii(
    lean_code=auto_result.lean_code
)
```

### Phase III: MCTS Refinement

```python
# Autoformalize hypothesis
auto_result = await auto_service.autoformalization_phase_iii(
    hypothesis_text="If x > 0 and y > 0, then x + y > 0",
    hypothesis_type="causal"
)

# MCTS proof search
proof_result = await proof_service.search_phase_iii(
    lean_code=auto_result.lean_code
)
```

### Phase IV: Architectural Synthesis

```python
# Autoformalize efficacy claim
auto_result = await auto_service.autoformalize_phase_iv(
    model_description="Linear regression with squared error loss",
    efficacy_claim="Model converges to true values with sufficient data"
)

# Verify efficacy
proof_result = await proof_service.search_phase_iv(
    lean_code=auto_result.lean_code
)
```

### Batch Processing

```python
# Batch autoformalization
items = [
    {"text": "Constraint 1", "type": "logical"},
    {"text": "Constraint 2", "type": "arithmetic"},
]

results = await auto_service.batch_autoformalize(
    items=items,
    phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT
)

for result in results:
    print(f"Success: {result.success}")
    print(f"Lean code: {result.lean_code}")
```

## Configuration

All configuration via environment variables:

```bash
# LeanAide Server
export LEANAIDE_HOST=localhost
export LEANAIDE_PORT=7654
export LEANAIDE_TIMEOUT_MS=30000

# Autoformalization
export LEANAIDE_CONFIDENCE_THRESHOLD=0.7
export LEANAIDE_MAX_ALTERNATIVES=3

# Proof Search
export PROOF_SEARCH_TIMEOUT_MS=60000
export PROOF_SEARCH_MAX_DEPTH=100
export PROOF_SEARCH_MCTS_ITERATIONS=1000
export PROOF_SEARCH_ENABLE_Z3=true
export PROOF_SEARCH_CONFIDENCE_THRESHOLD=0.8

# Workflow
export WORKFLOW_PHASE_I_TIMEOUT_MS=60000
export WORKFLOW_PHASE_II_TIMEOUT_MS=90000
export WORKFLOW_PHASE_III_TIMEOUT_MS=120000
export WORKFLOW_PHASE_IV_TIMEOUT_MS=90000
export WORKFLOW_TIMEOUT_MS=600000
export WORKFLOW_MAX_RETRIES=3
```

## Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_leanaide_rese_workflow.py::TestAutoformalizationService -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Run Probe

```bash
./probes/check_leanaide_workflow.sh
```

## Docker Deployment

### Build Image

```bash
docker build -t rese-leanaide-workflow:latest .
```

### Run Container

```bash
docker run -d \
  --name rese-leanaide-workflow \
  -p 7654:7654 \
  -e LEANAIDE_HOST=leanaide-server \
  -e LEANAIDE_PORT=7654 \
  rese-leanaide-workflow:latest
```

### Check Health

```bash
docker ps | grep rese-leanaide-workflow
```

## API Reference

### AutoformalizationService

```python
class AutoformalizationService:
    async def autoformalize_phase_i(
        self,
        constraint_text: str,
        constraint_type: str = "logical",
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult

    async def autoformalize_phase_ii(
        self,
        mapping_description: str,
        source_domain: str,
        target_domain: str,
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult

    async def autoformalize_phase_iii(
        self,
        hypothesis_text: str,
        hypothesis_type: str = "causal",
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult

    async def autoformalize_phase_iv(
        self,
        model_description: str,
        efficacy_claim: str,
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult

    async def batch_autoformalize(
        self,
        items: List[Dict[str, Any]],
        phase: AutoformalizationPhase,
        correlation_id: Optional[str] = None
    ) -> List[AutoformalizationResult]
```

### ProofSearchService

```python
class ProofSearchService:
    async def search_phase_i(
        self,
        lean_code: str,
        constraint_type: str = "logical",
        strategy: ProofStrategy = ProofStrategy.Z3_LEAN_HYBRID,
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult

    async def search_phase_ii(
        self,
        lean_code: str,
        isomorphism_type: str = "structural",
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult

    async def search_phase_iii(
        self,
        lean_code: str,
        hypothesis: Optional[Hypothesis] = None,
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult

    async def search_phase_iv(
        self,
        lean_code: str,
        efficacy_claim: str = "",
        correlation_id: Optional[str] = None
    ) -> ProofSearchResult
```

### LeanAideRESEWorkflow

```python
class LeanAideRESEWorkflow:
    async def execute(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> WorkflowResult
```

## Examples

### Example 1: Simple Theorem

```python
result = await execute_workflow(
    problem_statement="Prove that adding zero to any number returns the same number"
)
```

### Example 2: Isomorphism Detection

```python
result = await execute_workflow(
    problem_statement="Find isomorphic mapping between natural numbers and integers"
)
```

### Example 3: Hypothesis Testing

```python
result = await execute_workflow(
    problem_statement="Test hypothesis: sum of two positive numbers is positive"
)
```

### Example 4: Model Validation

```python
result = await execute_workflow(
    problem_statement="Validate that linear regression model converges to true values"
)
```

## Troubleshooting

### LeanAide Server Not Available

If LeanAide server is not available, the adapter will use simulation mode:
- Autoformalization will use template-based generation
- Proof search will use fallback tactics

To enable full functionality:
1. Start LeanAide server: `leanaide-server --port 7654`
2. Verify connectivity: `curl http://localhost:7654`
3. Check adapter health: `./probes/check_leanaide_workflow.sh`

### Timeout Errors

Increase timeout values:
```bash
export LEANAIDE_TIMEOUT_MS=60000
export PROOF_SEARCH_TIMEOUT_MS=120000
export WORKFLOW_TIMEOUT_MS=600000
```

### Import Errors

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Contributing

When contributing to this adapter:

1. Follow CLAUDE.md principles strictly
2. Ensure all tests pass: `pytest tests/ -v`
3. Run probe script: `./probes/check_leanaide_workflow.sh`
4. Update documentation
5. Use structured logging with correlation IDs

## License

See LICENSE file for details.

## Authors

OpenEvolve Team

## Version

1.0.0
