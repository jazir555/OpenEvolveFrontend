# LeanAide User Guide

Complete user guide for LeanAide - Lean 4 integration with LLM-powered autoformalization for OpenEvolve.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Autoformalization](#autoformalization)
6. [Proof Verification](#proof-verification)
7. [Proof Completion](#proof-completion)
8. [Continuous Mathematics](#continuous-mathematics)
9. [Z3 Integration](#z3-integration)
10. [API Reference](#api-reference)
11. [Examples](#examples)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

LeanAide is a comprehensive integration between:
- **Lean 4**: A powerful theorem prover and programming language
- **LLMs (GPT-4, Claude)**: Large language models for natural language understanding
- **OpenEvolve**: Evolutionary optimization platform

### Key Features

| Feature | Description |
|---------|-------------|
| Autoformalization | Convert natural language to Lean 4 code |
| Proof Verification | Verify Lean 4 proofs automatically |
| Proof Completion | Use LLM to complete incomplete proofs |
| Continuous Math | Real analysis, calculus, linear algebra |
| Z3 Bridge | Hybrid SMT/theorem proving |
| Batch Processing | Process multiple problems at once |

---

## Quick Start

### One-Command Setup

```bash
# Install everything automatically
python setup_lean4_enhanced.py --auto-install

# Verify installation
python setup_lean4_enhanced.py --verify
```

### First Autoformalization

```python
import asyncio
from lean4_integration_enhanced import create_lean4_service

async def main():
    service = create_lean4_service(openai_api_key="sk-...")
    
    result = await service.autoformalize(
        "For all natural numbers n, n + 0 = n",
        domain="arithmetic"
    )
    
    print(result.lean_code)

asyncio.run(main())
```

---

## Installation

### Prerequisites

- Python 3.10+
- Internet connection
- OpenAI or Anthropic API key (for LLM features)

### Automated Installation

```bash
# Check current status
python setup_lean4_enhanced.py --check-only

# Install Lean 4, mathlib4, and dependencies
python setup_lean4_enhanced.py --auto-install

# Verify everything works
python setup_lean4_enhanced.py --verify
```

### Manual Installation

If automatic installation fails:

**Linux/macOS:**
```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env
elan toolchain install stable
elan default stable
```

**Windows:**
```powershell
# In PowerShell
Invoke-RestMethod -Uri 'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.ps1' | Invoke-Expression
```

### Environment Variables

```bash
# Required for LLM features
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: Set preferred provider
export LLM_PROVIDER="openai"  # or "anthropic"

# Optional: Lean configuration
export LEAN_EXECUTABLE="lean"
export LAKE_EXECUTABLE="lake"
export LEAN_TIMEOUT="60"
```

---

## Basic Usage

### Creating a Service

```python
from lean4_integration_enhanced import (
    LeanAideServiceEnhanced,
    Lean4ServerConfig,
    LLMProvider
)

# Basic configuration
config = Lean4ServerConfig(
    enable_caching=True,
    timeout_seconds=60.0
)

# With LLM support
config_with_llm = Lean4ServerConfig(
    llm_provider=LLMProvider.OPENAI,
    openai_api_key="sk-...",
    openai_model="gpt-4",
    enable_caching=True
)

# Create service
service = LeanAideServiceEnhanced(config_with_llm)
```

### Checking Installation Status

```python
from setup_lean4_enhanced import Lean4EnhancedSetupManager

manager = Lean4EnhancedSetupManager()
status = manager.check_installation()

print(f"Lean available: {status.lean_available}")
print(f"Lake available: {status.lake_available}")
print(f"Mathlib4 available: {status.mathlib_available}")
```

---

## Autoformalization

### Natural Language to Lean 4

```python
result = await service.autoformalize(
    "The limit as x approaches 0 of sin(x)/x equals 1",
    domain="real_analysis"
)

if result.success:
    print(f"Generated code:\n{result.lean_code}")
    print(f"Confidence: {result.confidence}")
```

### Supported Domains

| Domain | Description | Example |
|--------|-------------|---------|
| `arithmetic` | Basic number theory | "n + 0 = n" |
| `algebra` | Algebraic structures | "x + y = y + x" |
| `real_analysis` | Limits, continuity | "lim(x→0) sin(x)/x = 1" |
| `calculus` | Derivatives, integrals | "d/dx(x²) = 2x" |
| `linear_algebra` | Matrices, vectors | "det(AB) = det(A)det(B)" |
| `number_theory` | Primes, divisibility | "√2 is irrational" |

### Batch Autoformalization

```python
problems = [
    {"text": "2 + 2 = 4", "domain": "arithmetic"},
    {"text": "x + y = y + x", "domain": "algebra"},
    {"text": "The derivative of x^2 is 2x", "domain": "calculus"},
]

results = []
for problem in problems:
    result = await service.autoformalize(
        problem["text"],
        domain=problem["domain"]
    )
    results.append(result)

# Summarize
successes = sum(1 for r in results if r.success)
print(f"Success rate: {successes}/{len(problems)}")
```

---

## Proof Verification

### Verifying Lean Code

```python
lean_code = """
theorem add_zero (n : Nat) : n + 0 = n := by
  rfl
"""

result = await service.verify(lean_code)

print(f"Status: {result.status}")
print(f"Success: {result.success}")
if result.errors:
    print(f"Errors: {result.errors}")
```

### Verification Status Codes

| Status | Meaning |
|--------|---------|
| `SUCCESS` | Proof is correct |
| `SYNTAX_ERROR` | Syntax error in code |
| `TYPE_ERROR` | Type checking failed |
| `PROOF_ERROR` | Proof is incomplete/incorrect |
| `TIMEOUT` | Verification timed out |
| `LEAN_NOT_INSTALLED` | Lean 4 not available |

### Batch Verification

```python
theorems = [
    "theorem t1 : 1 + 1 = 2 := by rfl",
    "theorem t2 (n : Nat) : n + 0 = n := by rfl",
    "theorem t3 (n : Nat) : n * 1 = n := by rfl",
]

results = await service.batch_verify(theorems)

for code, result in zip(theorems, results):
    status = "✓" if result.success else "✗"
    print(f"{status} {code[:40]}...")
```

---

## Proof Completion

### Completing Proofs with LLM

```python
incomplete = """
theorem square_nonneg (x : ℝ) : x^2 ≥ 0 := by
  -- complete this proof
"""

result = await service.complete_proof(incomplete)

if result.success:
    print(f"Completed proof:\n{result.completed_code}")
    print(f"Tactics used: {result.tactics_used}")
```

### Getting Proof Suggestions

```python
result = await service.suggest_proof_tactics(
    "theorem add_comm (n m : Nat) : n + m = m + n"
)

for suggestion in result.suggestions:
    print(f"Tactic: {suggestion.tactic}")
    print(f"Confidence: {suggestion.confidence}")
    print(f"Explanation: {suggestion.explanation}")
```

---

## Continuous Mathematics

### Using the Math Engine

```python
from leanaide_continuous_math import create_continuous_math_engine

engine = create_continuous_math_engine()

# Compute limit
limit_result = await engine.compute_limit(
    "sin(x)/x", 
    variable="x", 
    point=0.0
)
print(f"Limit: {limit_result.limit_value}")

# Compute derivative
derivative = await engine.compute_derivative(
    "x^3 + 2*x^2",
    variable="x",
    order=1
)
print(f"Derivative: {derivative.derivative}")

# Compute integral
integral = await engine.compute_integral(
    "x^2",
    variable="x",
    lower=0.0,
    upper=1.0
)
print(f"Integral: {integral.value}")
```

### Supported Operations

| Operation | Method | Example |
|-----------|--------|---------|
| Limit | `compute_limit()` | lim(x→0) sin(x)/x |
| Derivative | `compute_derivative()` | d/dx(x²) = 2x |
| Integral | `compute_integral()` | ∫₀¹ x² dx = 1/3 |
| Optimization | `optimize()` | min (x-2)² + (y-3)² |
| ODE | `solve_ode()` | dy/dt = -y |

---

## Z3 Integration

### Using the Z3 Bridge

```python
from z3_leanaide_bridge import create_z3_lean_bridge

bridge = create_z3_lean_bridge()

# Translate Z3 to Lean
from z3 import Real, And
x = Real('x')
y = Real('y')
z3_expr = And(x > 0, y > 0, x + y > 0)

lean_code = bridge.z3_to_lean4(z3_expr)
print(lean_code.lean_code)

# Hybrid verification
verification = await bridge.verify(lean_code.lean_code)
print(f"Z3 result: {verification.z3_result}")
print(f"Lean result: {verification.lean_result}")
print(f"Agreed: {verification.agreed}")
```

---

## API Reference

### LeanAideServiceEnhanced

| Method | Description | Returns |
|--------|-------------|---------|
| `verify(code)` | Verify Lean 4 code | `VerificationResult` |
| `autoformalize(text, domain)` | Convert NL to Lean | `AutoformalizationResult` |
| `complete_proof(code)` | Complete incomplete proof | `ProofCompletionResult` |
| `suggest_proof_tactics(code)` | Get tactic suggestions | `ProofSuggestionResult` |
| `batch_verify(codes)` | Verify multiple proofs | `List[VerificationResult]` |
| `close()` | Clean up resources | `None` |

### Configuration Options

```python
Lean4ServerConfig(
    lean_executable="lean",          # Path to lean
    lake_executable="lake",          # Path to lake
    timeout_seconds=60.0,            # Verification timeout
    max_memory_mb=4096,              # Memory limit
    enable_caching=True,             # Enable result caching
    cache_dir=".lean_cache",         # Cache directory
    
    # LLM Configuration
    llm_provider=LLMProvider.OPENAI,
    openai_api_key="...",
    openai_model="gpt-4",
    anthropic_api_key="...",
    anthropic_model="claude-3-opus",
    autoformalization_temperature=0.2,
    max_llm_retries=3,
)
```

---

## Examples

### Example 1: Basic Arithmetic

```python
result = await service.autoformalize(
    "For all natural numbers n, n plus zero equals n"
)

# Output:
# theorem add_zero (n : ℕ) : n + 0 = n := by
#   rfl
```

### Example 2: Calculus

```python
result = await service.autoformalize(
    "The derivative of x cubed is 3 times x squared",
    domain="calculus"
)

# Output:
# theorem derivative_x_cubed : 
#   deriv (λ x : ℝ => x^3) = λ x => 3 * x^2 := by
#   funext x
#   simp [deriv_pow]
```

### Example 3: Linear Algebra

```python
result = await service.autoformalize(
    "The determinant of a product of matrices equals the product of determinants",
    domain="linear_algebra"
)

# Output:
# theorem det_mul {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) :
#   (A * B).det = A.det * B.det := by
#   apply Matrix.det_mul
```

### Example 4: End-to-End Workflow

```python
async def complete_workflow():
    """Complete workflow: NL → Formal → Verify → Use"""
    
    # 1. Start with natural language
    problem = "The sum of first n natural numbers is n(n+1)/2"
    
    # 2. Autoformalize
    formal = await service.autoformalize(problem, domain="arithmetic")
    if not formal.success:
        print("Autoformalization failed")
        return
    
    print(f"Generated:\n{formal.lean_code}")
    
    # 3. Verify
    verify_result = await service.verify(formal.lean_code)
    if verify_result.success:
        print("✓ Verification passed")
    else:
        print(f"✗ Verification failed: {verify_result.errors}")
    
    # 4. Complete proof if needed
    if "sorry" in formal.lean_code:
        completed = await service.complete_proof(formal.lean_code)
        if completed.success:
            print(f"Completed:\n{completed.completed_code}")

asyncio.run(complete_workflow())
```

---

## Troubleshooting

### Issue: "lean: command not found"

**Solution:**
```bash
# Check if elan is installed
which elan || echo "elan not found"

# Add to PATH
export PATH="$HOME/.elan/bin:$PATH"

# Or run setup again
python setup_lean4_enhanced.py --auto-install
```

### Issue: "No LLM provider available"

**Solution:**
```bash
# Install OpenAI package
pip install openai

# Set API key
export OPENAI_API_KEY="sk-..."

# Verify
python -c "import os; print(os.environ.get('OPENAI_API_KEY'))"
```

### Issue: Mathlib4 build fails

**Solution:**
```bash
# Mathlib4 is large and takes time
# Increase timeout
export LEAN_SETUP_TIMEOUT=1800  # 30 minutes

# Or build manually
cd ~/lean_projects/mathlib_project
lake update
lake build
```

### Issue: Verification times out

**Solution:**
```python
# Increase timeout in config
config = Lean4ServerConfig(
    timeout_seconds=120.0  # Increase from default 60s
)
```

### Issue: API rate limits

**Solution:**
```python
# Enable caching to reduce API calls
config = Lean4ServerConfig(
    enable_caching=True,
    cache_ttl_seconds=3600
)
```

---

## Additional Resources

- [Lean 4 Documentation](https://lean-lang.org/lean4/doc/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)
- [LeanAide Setup Guide](LEANAIDE_SETUP.md)
- [OpenEvolve Documentation](docs/knowledge_engine/)

---

## Support

For issues or questions:
1. Check this guide first
2. Review [LEANAIDE_SETUP.md](LEANAIDE_SETUP.md)
3. Run diagnostics: `python setup_lean4_enhanced.py --check-only --json`
4. Check examples in `examples/lean/`

---

**Version**: 2.0.0  
**Last Updated**: February 2026  
**License**: Apache 2.0 / MIT
