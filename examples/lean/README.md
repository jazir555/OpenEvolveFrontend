# Lean 4 Proof Examples for LeanAide

This directory contains verified Lean 4 proof examples organized by mathematical domain.

## Quick Start

### Prerequisites
```bash
# Install Lean 4 (one command)
python setup_lean4_enhanced.py --auto-install

# Verify installation
python setup_lean4_enhanced.py --verify
```

### Checking Proofs

```bash
# Check individual files
lean basic_arithmetic.lean
lean calculus.lean
lean linear_algebra.lean

# Or use lake to build
lake build
```

## File Overview

| File | Topics | Theorems |
|------|--------|----------|
| `basic_arithmetic.lean` | ℕ, ℤ, divisibility, even/odd | 20+ |
| `calculus.lean` | Limits, derivatives, continuity | 25+ |
| `linear_algebra.lean` | Vector spaces, matrices, eigenvalues | 30+ |

## Example Proofs

### Arithmetic
```lean
theorem add_zero_right (n : ℕ) : n + 0 = n := by
  rfl
```

### Calculus
```lean
theorem derivative_of_sin :
  deriv Real.sin = Real.cos := by
  funext x
  exact Real.deriv_sin
```

### Linear Algebra
```lean
theorem cauchy_schwarz {n : ℕ} (u v : Fin n → ℝ) :
  |dotProduct u v| ≤ ‖u‖ * ‖v‖ := by
  apply abs_real_inner_le_norm
```

## Integration with LeanAide

These examples can be used with LeanAide's autoformalization:

```python
from lean4_integration_enhanced import LeanAideServiceEnhanced

service = LeanAideServiceEnhanced()

# Verify a proof from examples
with open('examples/lean/basic_arithmetic.lean') as f:
    code = f.read()
    
result = await service.verify(code)
print(f"Verification: {result.success}")
```

## Structure

Each file is organized into sections:
1. **Imports** - Mathlib and other dependencies
2. **Namespace** - Organized under `BasicArithmetic`, `Calculus`, etc.
3. **Sections** - Thematic groupings (e.g., `NaturalNumbers`, `Derivatives`)
4. **Theorems** - Formal statements with proofs

## Extending Examples

To add new proofs:
1. Create a new `.lean` file
2. Import required libraries
3. Define theorems with `theorem name : statement := by`
4. Provide proof tactics
5. Test with `lean your_file.lean`

## Resources

- [Lean 4 Documentation](https://lean-lang.org/lean4/doc/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)
