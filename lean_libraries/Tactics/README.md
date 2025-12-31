# Lean 4 Physics-Specific Tactics Library

This directory contains custom Lean 4 tactics specifically designed for physics proofs in quantum mechanics, general relativity, statistical mechanics, and mathematical analysis.

## Overview

The tactics library is organized into four main modules:

1. **Quantum.lean** - Quantum mechanics tactics
2. **Relativity.lean** - General relativity and differential geometry tactics
3. **StatMech.lean** - Statistical mechanics and thermodynamics tactics
4. **Analysis.lean** - Mathematical analysis and asymptotic methods tactics

## Usage

### Basic Import

```lean
import LeanLraries.Tactics
```

Or import individual modules:

```lean
import LeanLraries.Tactics.Quantum
import LeanLraries.Tactics.Relativity
import LeanLraries.Tactics.StatMech
import LeanLraries.Tactics.Analysis
```

## Quantum Tactics (`Quantum.lean`)

### Available Tactics

#### `quantum_normalize`
Normalizes quantum states in orthonormal basis.

```lean
example {ℋ : Type*} [HilbertSpace ℋ] (ψ : ℋ) (h : ‖ψ‖ = 1) :
    ⟪ψ, ψ⟫ = 1 := by
  quantum_normalize at h
  exact h
```

#### `apply_unitary`
Applies unitary operators to quantum states using U†U = I.

```lean
example {ℋ : Type*} [HilbertSpace ℋ] (ψ : ℋ) (U : Unitary ℋ) :
    ‖U ψ‖ = ‖ψ‖ := by
  apply_unitary U
  rfl
```

#### `compute_expectation`
Calculates expectation values of observables ⟨ψ|A|ψ⟩.

```lean
example {ℋ : Type*} [HilbertSpace ℋ] (ψ : ℋ) (A : ℋ →ₗ[ℂ] ℋ) [IsHermitian A] :
    ⟪ψ, A ψ⟫ = ⟪A ψ, ψ⟫ := by
  compute_expectation A ψ
```

#### `spectral_decompose`
Performs spectral decomposition of Hermitian operators.

```lean
example {ℋ : Type*} [HilbertSpace ℋ] [FiniteDimensional ℂ ℋ]
    (A : ℋ →ₗ[ℂ] ℋ) [IsHermitian A] [IsDiagonalizable ℂ A.toLinearMap] :
    True := by
  spectral_decompose A
  -- Proceed with spectral theorem proof
```

#### `quantum_simp`
Combines all quantum tactics in sequence.

```lean
example {ℋ : Type*} [HilbertSpace ℋ] [FiniteDimensional ℂ ℋ] (ψ : ℋ) :
    True := by
  quantum_simp
```

## Relativity Tactics (`Relativity.lean`)

### Available Tactics

#### `tensor_simplify`
Simplifies tensor expressions using symmetries and algebraic identities.

```lean
example {M : Type*} [PseudoRiemannianManifold M I] (T : Tensor M) :
    T α β = T β α := by
  tensor_simplify using symmetry
```

#### `covariant_derivative`
Applies covariant derivative rules (Leibniz, metric compatibility).

```lean
example {M : Type*} [PseudoRiemannianManifold M I]
    (f : M → ℝ) (X : TangentSpace M) :
    ∇ₓf = X f := by
  covariant_derivative
```

#### `raise_lower_indices`
Raises and lowers tensor indices using the metric.

```lean
example {M : Type*} [PseudoRiemannianManifold M I]
    (g : Metric I M) (T : Tensor M) (α : I.Index) :
    T^α = g^{αβ} T_β := by
  raise_lower_indices (g : Metric I M) ↑ α
```

#### `curvature_identities`
Applies Bianchi identities and curvature tensor symmetries.

```lean
example {M : Type*} [PseudoRiemannianManifold M I]
    (R : RiemannCurvature M) :
    R^α_{βγδ} = -R^α_{βδγ} := by
  curvature_identities [symmetry]
```

#### `relativity_simp`
Combines all relativity tactics.

```lean
example {M : Type*} [PseudoRiemannianManifold M I] :
    True := by
  relativity_simp
```

#### `einstein_simplify`
Specializes for Einstein field equations.

```lean
example {M : Type*} [PseudoRiemannianManifold M I] :
    True := by
  einstein_simplify
```

## Statistical Mechanics Tactics (`StatMech.lean`)

### Available Tactics

#### `ensemble_average`
Computes ensemble averages using ergodic hypothesis.

```lean
example {Ω : Type*} [MeasureSpace Ω] {A : Ω → ℝ}
    (T : ℝ) (μ : Measure Ω) :
    lim_{T→∞} (1/T) ∫₀ᵀ A(t) dt = ∫ A dμ := by
  ensemble_average using ergodic A
```

#### `thermodynamic_limit`
Takes N → ∞ limits for extensive/intensive quantities.

```lean
example {Q : ℕ → ℝ} [Extensive Q] (N : Nat) :
    lim_{N→∞} (Q(N)/N) = q := by
  thermodynamic_limit as N → ∞ of Q(N)
```

#### `maxwell_boltzmann`
Applies Maxwell-Boltzmann distribution.

```lean
example (v : ℝ³) (m T : ℝ) :
    f(v) = (m/(2πkT))^(3/2) * exp(-m|v|²/(2kT)) := by
  maxwell_boltzmann velocity
```

#### `canonical_transform`
Transforms between statistical ensembles.

```lean
example (β : ℝ) (Z : ℝ) (Ω_E : Set Ω) :
    Z(β) = ∫ e^{-βE} Ω(E) dE := by
  canonical_transform from microcanonical to canonical
```

#### `statmech_simp`
Combines all statistical mechanics tactics.

```lean
example : True := by
  statmech_simp
```

#### `canonical_simplify`
Specializes for canonical ensemble calculations.

```lean
example : True := by
  canonical_simplify
```

## Analysis Tactics (`Analysis.lean`)

### Available Tactics

#### `asymptotic_expand`
Generates asymptotic expansions with big-O notation.

```lean
example (x : ℝ) (h : x → 0) :
    sin x = x - x³/6 + O(x⁵) := by
  asymptotic_expand as x → 0 up to 5
```

#### `interval_arithmetic`
Performs rigorous interval computations.

```lean
example (x y : ℝ) (hx : x ∈ [0, 1]) (hy : y ∈ [2, 3]) :
    x + y ∈ [2, 4] := by
  interval_arithmetic using bounds
```

#### `perturbation_theory`
Applies perturbation theory expansions.

```lean
example (ε : ℝ) (hε : ε ≪ 1) :
    solve y' + ε y² = 0 for y := by
  perturbation_theory with parameter ε to order 2 regular
```

#### `analysis_simp`
Combines all analysis tactics.

```lean
example : True := by
  analysis_simp
```

#### `series_expand`
Specializes for series expansions.

```lean
example : True := by
  series_expand to order 5
```

#### `rigorous_bound`
Specializes for rigorous error bounds.

```lean
example (f : ℝ → ℝ) (x : ℝ) (h : 0 < x) (h' : x < 1) :
    f x ∈ [f 0.1, f 0.9] := by
  rigorous_bound with precision 0.001
```

## Implementation Details

### Tactic Elaboration

Each tactic is implemented using Lean 4's tactic elaboration framework:

```lean
elab (name := tacticName) "tactic_syntax" loc:(ppSpace)? args:term : Tactic => do
  -- Tactic implementation
```

### Helper Theorems

Each tactic module includes helper theorems that support the automation:

- **Quantum**: Inner product identities, unitary operator properties, spectral theorem
- **Relativity**: Tensor symmetries, covariant derivative rules, curvature identities
- **StatMech**: Ergodic hypothesis, thermodynamic limits, ensemble transformations
- **Analysis**: Taylor expansions, interval arithmetic, perturbation series

### Combination Tactics

Each module provides combination tactics that apply multiple tactics in sequence for common proof patterns.

## Integration with Mathlib

All tactics are designed to work seamlessly with Lean 4's Mathlib:

- Use standard Mathlib definitions where available
- Compatible with existing Mathlib tactics (`simp`, `rw`, etc.)
- Follow Mathlib naming conventions
- Include proper theorem statements (with `sorry` placeholders for full proofs)

## Development Status

**Current Status**: Alpha Release

The tactics library is under active development. Current features:

- ✅ Tactic syntax and elaboration implemented
- ✅ Helper theorems with structure defined
- ✅ Example usage provided
- ⚠️ Many helper theorems use `sorry` placeholders (need formal proofs)
- ⚠️ Integration testing needed
- ⚠️ Performance optimization pending

### TODO

1. Complete proofs for all helper theorems
2. Add more specialized tactics for each domain
3. Improve error messages and tactic failure handling
4. Add more comprehensive examples and documentation
5. Performance optimization for large proofs
6. Integration testing with actual physics proofs

## Contributing

When adding new tactics:

1. Follow the established structure in existing files
2. Include proper documentation strings
3. Add helper theorems as needed
4. Provide example usage
5. Update this README

## License

This tactics library is part of the OpenEvolve project.

## Contact

For questions or issues, please refer to the main OpenEvolve documentation.
