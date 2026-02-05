# RESE Tiered Verification System

A unified 3-tier verification system integrating Z3, LeanAide, and Lean 4 for scalable formal verification.

## Overview

The tiered verification system provides a unified API for constraint satisfaction, theorem proving, and formal verification across three tiers of increasing rigor:

- **Tier 1: Z3 Fast Verification** - Fast SMT solving (<1 second, 0-100 constraints)
- **Tier 2: LeanAide AI-Assisted Proving** - AI-guided theorem proving (<1 minute, 100-1000 constraints)
- **Tier 3: Lean 4 Formal Verification** - Machine-checkable proofs (any time, 1000+ constraints)

## Features

- **Adaptive Solver Selection** - Automatically selects appropriate tier based on problem complexity
- **Automatic Tier Escalation** - Escalates to higher tiers if lower tiers cannot solve
- **Unified API** - Single interface for all verification types
- **Performance Monitoring** - Tracks solver performance and effectiveness
- **Circuit Breaker Pattern** - Detects and handles solver failures gracefully
- **Problem Classification** - Classifies problems by type, domain, and complexity

## Installation

### Prerequisites

- Python 3.11+
- Z3 (optional, for Tier 1)
- LeanAide (optional, for Tier 2)
- Lean 4 (optional, for Tier 3)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Solvers

#### Z3 (Tier 1)

```bash
# Linux
curl -L https://github.com/Z3Prover/z3/releases/download/z3-4.12.6/z3-4.12.6-x64-glibc-2.35.zip -o z3.zip
unzip z3.zip
sudo mv z3-4.12.6-x64-glibc-2.35/bin/z3 /usr/local/bin/

# macOS
brew install z3
```

#### Lean 4 (Tier 3)

```bash
curl -L https://github.com/leanprover/lean4/releases/download/v4.6.0/lean-4.6.0-linux.tar.gz -o lean4.tar.gz
tar -xzf lean4.tar.gz
sudo mv lean-4.6.0-linux/lean /usr/local/bin/
```

### Verify Installation

```bash
./probes/check_tiered_verification.sh
```

## Quick Start

### Basic Usage

```python
from glue.adapters.rese_verification.src import TieredVerifier

# Create verifier
verifier = TieredVerifier()

# Verify a problem
result = verifier.verify("forall x, P(x) -> Q(x)")

# Check result
if result.is_successful():
    print(f"✓ {result.get_summary()}")
else:
    print(f"✗ Verification failed")
```

### With Constraints

```python
from glue.adapters.rese_verification.src import TieredVerifier

verifier = TieredVerifier()

# Verify with constraints
constraints = [
    "x > 0",
    "y > 0",
    "x + y > 0"
]

result = verifier.verify(
    "Find x, y such that constraints hold",
    constraints=constraints
)

print(result.get_summary())
```

### Specify Tier

```python
from glue.adapters.rese_verification.src import TieredVerifier, VerificationTier

verifier = TieredVerifier()

# Verify with specific tier
result = verifier.verify_with_tier(
    "Prove theorem",
    tier=VerificationTier.TIER3_LEAN4
)
```

## Configuration

All configuration is via environment variables (Law of Configuration Explicitness):

```bash
# Tier 1 (Z3)
export TIER1_TIMEOUT_MS=1000
export TIER1_MAX_CONSTRAINTS=100

# Tier 2 (LeanAide)
export TIER2_TIMEOUT_MS=60000
export TIER2_MAX_CONSTRAINTS=1000

# Tier 3 (Lean 4)
export TIER3_TIMEOUT_MS=300000

# Auto-escalation
export AUTO_ESCALATE=true
export MAX_TIER=3

# Selection strategy
export SELECTION_STRATEGY=adaptive  # fast_first, accurate_first, parallel, adaptive
export PREFER_FAST_SOLVER=true

# Performance
export MIN_CONFIDENCE_THRESHOLD=0.7
export ENABLE_MONITORING=true
```

## API Reference

### TieredVerifier

Main verification orchestrator.

#### Methods

- `verify(problem, constraints=None, variables=None, metadata=None, correlation_id=None)` - Main verification entry point
- `verify_with_tier(problem, tier, constraints=None, variables=None, correlation_id=None)` - Verify with specific tier
- `escalate_tier(current_result, problem, constraints=None, variables=None, correlation_id=None)` - Escalate to next tier
- `get_verification_status(correlation_id)` - Get verification status
- `combine_results(results, correlation_id=None)` - Combine results from multiple tiers

### UnifiedVerificationResult

Result object combining all tier results.

#### Attributes

- `correlation_id` - Correlation ID for tracing
- `problem_class` - Problem classification
- `problem_domain` - Problem domain
- `tier1_result` - Tier 1 result (Z3)
- `tier2_result` - Tier 2 result (LeanAide)
- `tier3_result` - Tier 3 result (Lean 4)
- `final_status` - Final verification status
- `successful_tier` - Tier that succeeded
- `confidence` - Confidence in result (0.0 to 1.0)
- `escalation_path` - Tiers tried in order
- `total_execution_time_ms` - Total execution time

#### Methods

- `is_successful()` - Check if verification was successful
- `get_successful_result()` - Get the successful tier result
- `get_summary()` - Get human-readable summary

## Docker Usage

### Build Image

```bash
docker build -t rese-verification:latest .
```

### Run Container

```bash
docker run -d \
  -e TIER1_TIMEOUT_MS=1000 \
  -e TIER2_TIMEOUT_MS=60000 \
  -e TIER3_TIMEOUT_MS=300000 \
  -p 8080:8080 \
  --name rese-verification \
  rese-verification:latest
```

### Check Health

```bash
docker ps | grep rese-verification
```

## Testing

### Run All Tests

```bash
python tests/test_tiered_verifier.py
```

### Run Specific Test

```bash
python tests/test_tiered_verifier.py TestProblemClassifier
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

## Architecture

See `ARCHITECTURE.md` for detailed architecture documentation.

## Performance Characteristics

| Tier | Solver | Time | Constraints | Confidence | Use Case |
|------|--------|------|-------------|------------|----------|
| 1    | Z3     | <1s  | 0-100       | 70%        | Quick checks |
| 2    | LeanAide | <1m | 100-1000    | 85%        | AI-assisted proofs |
| 3    | Lean 4 | Any  | 1000+       | 100%       | Formal verification |

## Decision Trees

### Solver Selection

```
Start
  |
  v
Has quantifiers?
  Yes -> Has deep nesting?
    Yes -> Tier 3 (Lean 4)
    No -> Tier 2 (LeanAide)
  No -> Is nonlinear?
    Yes -> Tier 2 (LeanAide)
    No -> Tier 1 (Z3)
```

### Escalation

```
Tier 1 (Z3)
  |
  v
Timeout or Unknown?
  Yes -> Tier 2 (LeanAide)
    |
    v
  Timeout or Failed?
    Yes -> Tier 3 (Lean 4)
      |
      v
    Success or Final
```

## Troubleshooting

### Z3 Not Found

```bash
# Check if Z3 is installed
z3 --version

# Install if missing
# See installation instructions above
```

### Lean 4 Not Found

```bash
# Check if Lean 4 is installed
lean --version

# Install if missing
# See installation instructions above
```

### Import Errors

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## Contributing

Follow CLAUDE.md principles:
- Law of Configuration Explicitness
- Law of Runtime Truth
- Law of Idempotency
- Circuit Breaker Pattern
- Structured Logging

## License

See LICENSE file.

## Authors

RESE Team

## See Also

- `ARCHITECTURE.md` - Architecture documentation
- `docs/TIERED_VERIFICATION.md` - Detailed tiered verification guide
- `CLAUDE.md` - Project constitution
