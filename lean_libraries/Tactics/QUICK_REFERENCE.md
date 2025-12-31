# Physics Tactics Quick Reference Card

## Quantum Mechanics Tactics

```
quantum_normalize          Normalize states in orthonormal basis
quantum_normalize at h     Apply to hypothesis h

apply_unitary              Apply unitary operator (auto-detect)
apply_unitary U            Apply specific unitary U
apply_unitary with U       Alternative syntax

compute_expectation A ψ    Calculate ⟨ψ|A|ψ⟩
compute_expectation        Auto-detect observable

spectral_decompose A       Decompose operator A = Σ λᵢPᵢ
spectral_decompose         Auto-detect operator

quantum_simp               Combine all quantum tactics
```

## General Relativity Tactics

```
tensor_simplify                           General simplification
tensor_simplify at h                      Apply to hypothesis
tensor_simplify using symmetry            Use symmetries
tensor_simplify using algebra             Use algebraic rules
tensor_simplify using metric              Use metric relations

covariant_derivative                      Apply ∇ rules
covariant_derivative ∇                    With connection

raise_lower_indices                       Auto-detect metric
raise_lower_indices (g : Metric)          Specify metric
raise_lower_indices ↑ α                   Raise index α
raise_lower_indices ↓ β                   Lower index β

curvature_identities                      Apply all identities
curvature_identities [bianchi]            First Bianchi only
curvature_identities [symmetry]           Symmetries only
curvature_identities [ricci]              Ricci decomposition

relativity_simp                           Combine all relativity tactics
einstein_simplify                         EFE specialized tactics
```

## Statistical Mechanics Tactics

```
ensemble_average A                    Compute ensemble average
ensemble_average using ergodic A      Ergodic hypothesis
ensemble_average using microcanonical Microcanonical ensemble
ensemble_average using canonical      Canonical ensemble
ensemble_average using grand_canonical Grand canonical

thermodynamic_limit                   Take N → ∞ limit
thermodynamic_limit as N → ∞          Specify limit variable
thermodynamic_limit of Q              Specify quantity

maxwell_boltzmann                     Apply MB distribution
maxwell_boltzmann velocity            Velocity distribution
maxwell_boltzmann energy              Energy distribution
maxwell_boltzmann moment              Calculate moments

canonical_transform                   Transform ensembles
canonical_transform to canonical      To canonical ensemble
canonical_transform from microcanonical to canonical  Specify transformation

statmech_simp                         Combine all statmech tactics
canonical_simplify                    Canonical ensemble tactics
```

## Analysis Tactics

```
asymptotic_expand                     Generate asymptotic expansion
asymptotic_expand as x → 0            Specify limit point
asymptotic_expand up to n              Specify order n
asymptotic_expand with O               Use big-O notation
asymptotic_expand with o               Use little-o notation

interval_arithmetic                   Interval computation
interval_arithmetic with precision ε  Set precision
interval_arithmetic using bounds      Basic bounds
interval_arithmetic using rounding    Directed rounding
interval_arithmetic using affine      Affine arithmetic

perturbation_theory                   Apply perturbation theory
perturbation_theory with parameter ε  Specify small parameter
perturbation_theory to order n         Specify order
perturbation_theory regular           Regular perturbation
perturbation_theory singular          Singular perturbation
perturbation_theory multiscale        Multi-scale analysis

analysis_simp                         Combine all analysis tactics
series_expand to order n              Series expansion tactics
rigorous_bound with precision ε       Rigorous error bounds
```

## Common Patterns

### Quantum State Normalization
```lean
example (ψ : ℋ) (h : ‖ψ‖ = 1) : ⟪ψ, ψ⟫ = 1 := by
  quantum_normalize at h
  exact h
```

### Unitary Operator Application
```lean
example (U : Unitary ℋ) : ‖U ψ‖ = ‖ψ‖ := by
  apply_unitary U
  rfl
```

### Tensor Simplification
```lean
example : T α β = T β α := by
  tensor_simplify using symmetry
```

### Index Manipulation
```lean
example : T^α = g^{αβ} T_β := by
  raise_lower_indices ↑ α
```

### Ensemble Average
```lean
example : lim_{T→∞} (1/T) ∫₀ᵀ A(t) dt = ∫ A dμ := by
  ensemble_average using ergodic A
```

### Thermodynamic Limit
```lean
example : lim_{N→∞} (Q(N)/N) = q := by
  thermodynamic_limit as N → ∞ of Q(N)
```

### Asymptotic Expansion
```lean
example : sin x = x - x³/6 + O(x⁵) := by
  asymptotic_expand as x → 0 up to 5
```

## Tactic Modifiers

### Location Specifiers
```
at h              Apply to hypothesis h
at *              Apply to all hypotheses
(no modifier)     Apply to goal
```

### Mode Specifiers
```
using symmetry    Use symmetry rules
using algebra     Use algebraic rules
using metric       Use metric relations
with precision ε  Set precision ε
to order n        Expand to order n
```

### Index Operations
```
↑ α              Raise index α
↓ β              Lower index β
```

## Combination Tactics Summary

```
quantum_simp         = quantum_normalize + apply_unitary +
                      compute_expectation + spectral_decompose

relativity_simp      = tensor_simplify + covariant_derivative +
                      raise_lower_indices + curvature_identities

einstein_simplify    = tensor_simplify (metric) + covariant_derivative +
                      curvature_identities (selected)

statmech_simp        = ensemble_average + thermodynamic_limit +
                      maxwell_boltzmann

canonical_simplify   = ensemble_average (canonical) + thermodynamic_limit +
                      canonical_transform

analysis_simp        = asymptotic_expand + interval_arithmetic +
                      perturbation_theory

series_expand        = asymptotic_expand + perturbation_theory

rigorous_bound       = interval_arithmetic (bounds)
```

## Debugging

```
show_term           Show what transformations are applied
set_option profiler true  Enable profiling
```

## Import Options

```lean
import LeanLraries.Tactics              -- All tactics
import LeanLraries.Tactics.Quantum      -- Quantum only
import LeanLraries.Tactics.Relativity   -- Relativity only
import LeanLraries.Tactics.StatMech     -- StatMech only
import LeanLraries.Tactics.Analysis     -- Analysis only
```

## Type Classes Needed

```
[HilbertSpace ℋ]                          Quantum
[FiniteDimensional ℂ ℋ]                   Quantum
[IsHermitian A]                           Quantum
[Unitary ℋ]                               Quantum
[PseudoRiemannianManifold M I]            Relativity
[SymmetricTensor T]                       Relativity
[AntisymmetricTensor T]                   Relativity
[MeasureSpace Ω]                          StatMech
[Extensive Q]                             StatMech
[Intensive Q]                             StatMech
```

## Common Errors

```
Error: Could not infer unitary operator
→ Use: apply_unitary U

Error: Could not find metric in context
→ Use: raise_lower_indices (g : Metric)

Error: Could not find observable
→ Use: ensemble_average A

Error: Could not find quantity
→ Use: thermodynamic_limit of Q
```

## Quick Tips

1. Start with combination tactics (`quantum_simp`, etc.)
2. Use `show_term` to debug
3. Provide explicit parameters when auto-detection fails
4. Combine with standard tactics (`simp`, `rw`, etc.)
5. Check type class instances are available
