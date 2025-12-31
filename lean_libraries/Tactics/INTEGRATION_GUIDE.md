# Lean 4 Physics Tactics Library - Integration Guide

## Quick Start

### 1. Directory Structure

```
lean_libraries/Tactics/
├── Index.lean           # Main import file (imports all tactics)
├── Quantum.lean         # Quantum mechanics tactics
├── Relativity.lean      # General relativity tactics
├── StatMech.lean        # Statistical mechanics tactics
├── Analysis.lean        # Mathematical analysis tactics
├── Testing.lean         # Test suite for all tactics
├── README.md            # Detailed documentation
└── INTEGRATION_GUIDE.md # This file
```

### 2. Add to Lake Configuration

Add to your `lakefile.lean`:

```lean
lean_lib LeanLraries {
  -- add library configuration options
}
```

Or add individual modules:

```lean
lean_lib LeanLraries.Tactics {
  roots := #[`Index, `Quantum, `Relativity, `StatMech, `Analysis]
}
```

### 3. Import in Your Files

#### Option A: Import All Tactics

```lean
import LeanLraries.Tactics

-- All tactics are now available
```

#### Option B: Import Specific Modules

```lean
import LeanLraries.Tactics.Quantum
import LeanLibraries.Tactics.Relativity
-- etc.
```

## Usage Examples

### Quantum Mechanics

```lean
import LeanLraries.Tactics.Quantum

example {ℋ : Type*} [HilbertSpace ℋ] (ψ : ℋ) (U : Unitary ℋ) :
    ‖U ψ‖ = ‖ψ‖ := by
  apply_unitary U
  rfl
```

### General Relativity

```lean
import LeanLraries.Tactics.Relativity

example {M : Type*} [PseudoRiemannianManifold M I]
    (R : RiemannCurvature M) :
    R^α_{βγδ} = -R^α_{βδγ} := by
  curvature_identities [symmetry]
```

### Statistical Mechanics

```lean
import LeanLraries.Tactics.StatMech

example {Q : ℕ → ℝ} [Extensive Q] :
    lim_{N→∞} (Q(N)/N) = q := by
  thermodynamic_limit as N → ∞ of Q(N)
```

### Mathematical Analysis

```lean
import LeanLraries.Tactics.Analysis

example (x : ℝ) (h : x → 0) :
    sin x = x - x³/6 + O(x⁵) := by
  asymptotic_expand as x → 0 up to 5
```

## Available Tactics Reference

### Quantum.lean

| Tactic | Purpose | Syntax |
|--------|---------|--------|
| `quantum_normalize` | Normalize states | `quantum_normalize` |
| `apply_unitary` | Apply unitary operators | `apply_unitary U` |
| `compute_expectation` | Calculate expectation values | `compute_expectation A ψ` |
| `spectral_decompose` | Spectral decomposition | `spectral_decompose A` |
| `quantum_simp` | Combine all quantum tactics | `quantum_simp` |

### Relativity.lean

| Tactic | Purpose | Syntax |
|--------|---------|--------|
| `tensor_simplify` | Simplify tensors | `tensor_simplify using symmetry` |
| `covariant_derivative` | Apply ∇ rules | `covariant_derivative` |
| `raise_lower_indices` | Raise/lower indices | `raise_lower_indices ↑ α` |
| `curvature_identities` | Apply curvature identities | `curvature_identities [bianchi]` |
| `relativity_simp` | Combine all relativity tactics | `relativity_simp` |
| `einstein_simplify` | EFE specialized tactics | `einstein_simplify` |

### StatMech.lean

| Tactic | Purpose | Syntax |
|--------|---------|--------|
| `ensemble_average` | Compute averages | `ensemble_average A` |
| `thermodynamic_limit` | Take N→∞ limits | `thermodynamic_limit as N → ∞` |
| `maxwell_boltzmann` | Apply MB distribution | `maxwell_boltzmann velocity` |
| `canonical_transform` | Transform ensembles | `canonical_transform from microcanonical to canonical` |
| `statmech_simp` | Combine all statmech tactics | `statmech_simp` |
| `canonical_simplify` | Canonical ensemble tactics | `canonical_simplify` |

### Analysis.lean

| Tactic | Purpose | Syntax |
|--------|---------|--------|
| `asymptotic_expand` | Asymptotic expansions | `asymptotic_expand as x → 0 up to n` |
| `interval_arithmetic` | Interval computations | `interval_arithmetic with precision ε` |
| `perturbation_theory` | Perturbation expansions | `perturbation_theory with parameter ε to order n` |
| `analysis_simp` | Combine all analysis tactics | `analysis_simp` |
| `series_expand` | Series expansion tactics | `series_expand to order n` |
| `rigorous_bound` | Rigorous error bounds | `rigorous_bound with precision ε` |

## Integration with Existing Code

### Combining with Standard Tactics

```lean
example {ℋ : Type*} [HilbertSpace ℋ] (ψ : ℋ) (h : ‖ψ‖ = 1) :
    ⟪ψ, ψ⟫ = 1 := by
  quantum_normalize at h
  -- Now use standard tactics
  rw [← norm_sq_eq_inner] at h
  -- Continue proof
  sorry
```

### Creating Custom Tactic Combinations

```lean
macro "my_physics_simp" : Tactic => do
  `(tactic| (
    quantum_normalize
    simp [norm_sq_eq_inner]
    tensor_simplify using symmetry
  ))
```

## Testing

Run the test suite:

```bash
lake build test
```

Or test specific modules:

```bash
lake test LeanLraries.Tactics.Quantum
lake test LeanLraries.Tactics.Relativity
lake test LeanLraries.Tactics.StatMech
lake test LeanLraries.Tactics.Analysis
```

## Implementation Notes

### Placeholder Theorems

Many helper theorems use `sorry` placeholders. To make the library production-ready, you need to:

1. Replace `sorry` with formal proofs
2. Add necessary import dependencies
3. Ensure all theorems are well-formed

Example:

```lean
-- Before (placeholder)
theorem inner_prod_self_eq_one (h : ‖ψ‖ = 1) :
    ⟪ψ, ψ⟫ = 1 := by
  sorry

-- After (formal proof)
theorem inner_prod_self_eq_one (h : ‖ψ‖ = 1) :
    ⟪ψ, ψ⟫ = 1 := by
  rw [← norm_sq_eq_inner, h, pow_one]
```

### Dependencies

The tactics depend on these Mathlib modules:

- `Mathlib.Tactic`
- `Mathlib.Analysis.InnerProductSpace.Basic`
- `Mathlib.LinearAlgebra.UnitaryGroup`
- `Mathlib.LinearAlgebra.TensorProduct`
- `Mathlib.Analysis.MeanInequalities`
- `Mathlib.ProbabilityTheory`
- `Mathlib.Analysis.Asymptotics.Asymptotics`

Ensure your Mathlib is up to date:

```bash
lake update
lake build
```

## Customization

### Adding New Tactics

1. Create a new tactic in the appropriate module
2. Follow the existing pattern:
   ```lean
   elab (name := myTactic) "my_tactic" arg:term : Tactic => do
     -- Implementation
   ```
3. Add helper theorems in the `Theorems` section
4. Update the README with usage examples
5. Add tests to `Testing.lean`

### Modifying Existing Tactics

To modify an existing tactic:

1. Locate the tactic in its module file
2. Modify the elaboration code
3. Update helper theorems if needed
4. Update documentation
5. Add regression tests

## Troubleshooting

### Common Issues

**Issue**: Tactic not found after import
```lean
-- Solution: Make sure to import the correct module
import LeanLraries.Tactics.Quantum  -- Not just LeanLraries.Tactics
```

**Issue**: Type class failures
```lean
-- Solution: Add necessary type class instances
variable [HilbertSpace ℋ] [FiniteDimensional ℂ ℋ]
```

**Issue**: Tactic does nothing
```lean
-- Solution: Check that hypotheses are in the right form
-- Some tactics require specific structure
```

### Debugging Tactic Behavior

Use `show_term` to see what the tactic is doing:

```lean
example : True := by
  show_term
  quantum_simp  -- See what transformations are applied
  trivial
```

## Best Practices

1. **Start with combination tactics**: Use `quantum_simp`, `relativity_simp`, etc., before trying individual tactics
2. **Provide explicit parameters**: When in doubt, provide explicit parameters to tactics
3. **Check goal state**: Use `show_term` to understand what transformations are being applied
4. **Combine with standard tactics**: These tactics work best combined with `simp`, `rw`, etc.
5. **Report issues**: If you find bugs or unexpected behavior, note them for fixing

## Performance Considerations

- Some tactics (like `spectral_decompose`) can be slow on large expressions
- Use `only` variants to limit what tactics apply
- Consider `simp only [...]` to control simplification
- For performance-critical code, profile with `set_option profiler true`

## Future Enhancements

Planned improvements:

1. More specialized physics tactics (fluid dynamics, solid state, etc.)
2. Better automation with ML-based suggestion
3. Integration with computer algebra systems
4. Performance optimizations
5. More complete formal proofs for helper theorems

## Contributing

To contribute new tactics:

1. Follow the code style in existing files
2. Add comprehensive documentation
3. Include test cases
4. Update this integration guide
5. Ensure compatibility with existing tactics

## Contact and Support

For questions about integration:
- Check the README.md for detailed documentation
- See Testing.lean for usage examples
- Review the individual tactic files for implementation details

## License

Part of the OpenEvolve project.
