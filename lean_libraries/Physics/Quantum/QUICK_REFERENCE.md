# Quantum Mechanics Library - Quick Reference

## Import Statements

```lean
import lean_libraries.Physics.Quantum.Foundations
import lean_libraries.Physics.Quantum.HilbertSpace
import lean_libraries.Physics.Quantum.Operators
import lean_libraries.Physics.Quantum.Entanglement
```

## Core Type Signatures

### Quantum Systems and States
```lean
variable {𝓗 : Type*} [Hilbert 𝓗] [FiniteDimensional ℂ 𝓗]

-- A quantum system
structure QuantumSystem where
  hilbertSpace : Type*
  [hilbert : Hilbert hilbertSpace]
  [finite : FiniteDimensional ℂ hilbertSpace]
  stateSpace : ProjectiveSpace ℂ hilbertSpace
  observables : Set (LinearMap.End ℂ hilbertSpace)
  dynamics : LinearMap.End ℂ hilbertSpace → ℝ → LinearMap.End ℂ hilbertSpace

-- A quantum state (pure or mixed)
inductive QuantumState (Q : QuantumSystem) where
  | pure (ψ : Q.hilbertSpace) (h_norm : ‖ψ‖ = 1)
  | mixed (ρ : LinearMap.End ℂ Q.hilbertSpace)
      (h_pos : ∀ ψ, 0 ≤ Re (conj ψ * ρ ψ))
      (h_trace : Complex.linearMap.trace ρ = 1)
```

### Operators
```lean
-- Self-adjoint operator (observable)
structure SelfAdjointOperator where
  op : LinearMap.End ℂ 𝓗
  isSelfAdjoint : op.isSelfAdjoint

-- Unitary operator (symmetry/time evolution)
structure UnitaryOperator where
  op : LinearMap.End ℂ 𝓗
  isUnitary : op.isAdjointUnitary

-- Projection operator (measurement)
structure ProjectionOperator where
  op : LinearMap.End ℂ 𝓗
  idempotent : op ∘ₗ op = op
  selfAdjoint : op.isSelfAdjoint

-- Commutator
def commutator (A B : LinearMap.End ℂ 𝓗) : LinearMap.End ℂ 𝓗 :=
  A ∘ₗ B - B ∘ₗ A

notation:100 "[" A ", " B "]" =>:commutator A B
```

### Entanglement
```lean
-- Composite system
structure CompositeSystem where
  systemA : Type
  systemB : Type
  [hilbertA : Hilbert systemA]
  [hilbertB : Hilbert systemB]
  compositeSpace : TensorProduct ℂ systemA systemB

-- Separable vs entangled
def IsSeparable (ψ : 𝓗₁ ⊗ₜ[ℂ] 𝓗₂) : Prop :=
  ∃ (ψ₁ : 𝓗₁) (ψ₂ : 𝓗₂), ‖ψ₁‖ = 1 ∧ ‖ψ₂‖ = 1 ∧ ψ = ψ₁ ⊗ₜ ψ₂

def IsEntangled (ψ : 𝓗₁ ⊗ₜ[ℂ] 𝓗₂) : Prop :=
  ¬IsSeparable ψ

-- Bell states
inductive BellState : (ℂ ⊗ₜ[ℂ] ℂ) → Type where
  | phi_plus | phi_minus | psi_plus | psi_minus
```

## Key Theorems

### Foundations
```lean
-- No-cloning theorem
theorem noCloningTheorem :
  ¬∃ (U : LinearMap.End ℂ (𝓗 ⊗ 𝓗')),
    IsUnitary U ∧
    ∀ (ψ : 𝓗) (h : ‖ψ‖ = 1),
      U (ψ ⊗ (1 : ℕ → ℂ)) = ψ ⊗ ψ

-- Uncertainty principle
theorem uncertaintyPrinciple
    (ψ : QuantumSystem) (A B : LinearMap.End ℂ ψ.hilbertSpace)
    (h_selfA : A.isSelfAdjoint) (h_selfB : B.isSelfAdjoint)
    (φ : ψ.hilbertSpace) (h_norm : ‖φ‖ = 1) :
    let ΔA := sqrt(Re(conj φ * (A ∘ₗ A φ)) - (Re(conj φ * A φ))²)
    let ΔB := sqrt(Re(conj φ * (B ∘ₗ B φ)) - (Re(conj φ * B φ))²)
    ΔA * ΔB ≥ |Re(trace((A ∘ₗ B - B ∘ₗ A) • (·⊗·) φ φ))| / 2

-- Schrödinger equation
theorem schrodingerEquation
    (Q : QuantumSystem) (H : LinearMap.End ℂ Q.hilbertSpace)
    (h_hamiltonian : H.isSelfAdjoint) (ψ₀ : Q.hilbertSpace) (t : ℝ) :
    ∃ ψ : ℝ → Q.hilbertSpace,
      ψ 0 = ψ₀ ∧
      ∀ t, HasDerivAt ψ (-(I : ℂ) / (ℏ : ℂ) • (H (ψ t))) t
```

### Hilbert Space Theory
```lean
-- Cauchy-Schwarz
theorem cauchySchwarz (x y : 𝓗) :
  |⟪x, y⟫| ≤ ‖x‖ * ‖y‖

-- Parseval's identity
theorem parsevalIdentity (B : OrthonormalBasis) (v : 𝓗)
    (coeffs : 𝓗 → ℂ) (h_exp : v = ∑ i, coeffs i • basis i) :
  ‖v‖² = ∑ i, |coeffs i|²

-- Spectral theorem
theorem spectralTheoremFiniteDim
    (A : LinearMap.End ℂ 𝓗) (h_self : A.isSelfAdjoint) :
    ∃ (λ : Fin n → ℝ) (P : Fin n → Subspace ℂ 𝓗),
      (∀ i v, v ∈ P i → A v = (λ i : ℂ) • v) ∧
      (∀ i ≠ j, ∀ v ∈ P i, ∀ w ∈ P j, ⟪v, w⟫ = 0) ∧
      A = ∑ i, (λ i : ℂ) • projection P i
```

### Operators
```lean
-- Real spectrum
theorem realSpectrum (A : SelfAdjointOperator) :
  ∀ λ ∈ Spectrum ℂ A.op, Im λ = 0

-- Eigenvector orthogonality
theorem eigenvectorsOrthogonal (A : SelfAdjointOperator)
    {λ₁ λ₂ : ℝ} (h_ne : λ₁ ≠ λ₂)
    (v₁ : 𝓗) (h₁ : A.op v₁ = (λ₁ : ℂ) • v₁)
    (v₂ : 𝓗) (h₂ : A.op v₂ = (λ₂ : ℂ) • v₂) :
  ⟪v₁, v₂⟫ = 0

-- Uncertainty from commutator
theorem uncertaintyFromCommutator
    (A B : SelfAdjointOperator) (ψ : 𝓗) (h_norm : ‖ψ‖ = 1) :
    let ΔA := A.uncertainty ψ h_norm
    let ΔB := B.uncertainty ψ h_norm
    let commExp := ⟪ψ, [A.toLinearMap, B.toLinearMap] ψ⟫
    (ΔA * ΔB)² ≥ |commExp|² / 4

-- Heisenberg equation
theorem heisenbergEquation
    (H A : SelfAdjointOperator) (t : ℝ) :
    dA/dt = (i/ℏ)[H, A] + ∂A/∂t

-- Simultaneous diagonalization
theorem simultaneousDiagonalization (A B : SelfAdjointOperator) :
    [A.toLinearMap, B.toLinearMap] = 0 ↔
    ∃ (basis : OrthonormalBasis 𝓗),
      ∀ v ∈ basis.vectors,
        ∃ λ₁ λ₂ : ℝ,
          A.toLinearMap v = (λ₁ : ℂ) • v ∧
          B.toLinearMap v = (λ₂ : ℂ) • v
```

### Entanglement
```lean
-- Schmidt decomposition
theorem schmidtDecomposition
    (ψ : 𝓗₁ ⊗ₜ[ℂ] 𝓗₂) (h_norm : ‖ψ‖ = 1) :
    ∃ (λ : Fin n → ℝ) (e₁ : Fin n → 𝓗₁) (e₂ : Fin n → 𝓗₂),
      (∀ i, 0 ≤ λ i) ∧ (∑ i, λ i) = 1 ∧
      (∀ i, ‖e₁ i‖ = 1) ∧ (∀ i, ‖e₂ i‖ = 1) ∧
      (∀ i ≠ j, ⟪e₁ i, e₁ j⟫ = 0) ∧
      (∀ i ≠ j, ⟪e₂ i, e₂ j⟫ = 0) ∧
      ψ = ∑ i, Real.sqrt (λ i) • (e₁ i ⊗ₜ e₂ i)

-- Schmidt rank criterion
theorem schmidtRankOne_iff_separable
    {ψ : 𝓗₁ ⊗ₜ[ℂ] 𝓗₂} (h_norm : ‖ψ‖ = 1) :
    schmidtRank ψ h_norm = 1 ↔ IsSeparable ψ

-- Bell's theorem
theorem bellsTheorem :
  ¬∃ (λ : Type) (ρ : λ → ℝ),
    ∀ (ψ : QuantumState (𝓗₁ ⊗ₜ[ℂ] 𝓗₂))
      (A : SelfAdjointOperator 𝓗₁)
      (B : SelfAdjointOperator 𝓗₂),
      ψ.expectation (A.toLinearMap ⊗ₜ B.toLinearMap) = ∫ ω, A ω * B ω

-- Monogamy theorem
theorem monogamyTheorem
    (ψ : 𝓗₁ ⊗ₜ[ℂ] 𝓗₂ ⊗ₜ[ℂ] ℂ) (h_norm : ‖ψ‖ = 1) :
    let τ_AB := entanglementEntropy (partialTrace (partialTrace ψ))
    let τ_AC := entanglementEntropy (partialTrace (partialTrace ψ))
    let τ_ABC := entanglementEntropy (partialTrace ψ)
    τ_ABC ≥ τ_AB + τ_AC
```

## Common Patterns

### Define a Quantum State
```lean
-- Pure state
def myState : QuantumState Q :=
  QuantumState.pure ψ (by simp [h_norm])

-- Mixed state
def myMixedState : QuantumState Q :=
  QuantumState.mixed ρ
    (by intro χ; constructor; sorry)
    (by simp [trace_property])
```

### Compute Expectation Values
```lean
example (A : SelfAdjointOperator) (ψ : 𝓗) (h_norm : ‖ψ‖ = 1) : ℝ :=
  A.expectation ψ h_norm
  -- = Re(⟪ψ, A.op ψ⟫)
```

### Time Evolution
```lean
example (H : SelfAdjointOperator) (t : ℝ) : UnitaryOperator :=
  UnitaryOperator.timeEvolution H t
  -- = exp(-iHt/ℏ)
```

### Check Entanglement
```lean
example (ψ : 𝓗₁ ⊗ₜ[ℂ] 𝓗₂) : Bool :=
  if Entanglement.IsEntangled ψ then
    "State is entangled"
  else
    "State is separable"
```

### Compute Uncertainty
```lean
example (A : SelfAdjointOperator) (ψ : 𝓗) (h_norm : ‖ψ‖ = 1) : ℝ :=
  A.uncertainty ψ h_norm
  -- = sqrt(Var(A)) = sqrt(⟨A²⟩ - ⟨A⟩²)
```

## Notation Guide

| Physics | Lean 4 | Description |
|---------|--------|-------------|
| `|ψ⟩` | `ψ : 𝓗` | Vector in Hilbert space |
| `⟨φ|` | `⟪φ, ·⟫` | Linear functional via inner product |
| `⟨φ|ψ⟩` | `⟪φ, ψ⟫` | Inner product |
| `‖ψ‖` | `‖ψ‖` | Norm |
| `A|ψ⟩` | `A ψ` or `A.op ψ` | Operator application |
| `⟨φ|A|ψ⟩` | `⟪φ, A ψ⟫` | Matrix element |
| `A†` | `A.adjoint` or `A†` | Adjoint operator |
| `[A,B]` | `[A, B]` | Commutator |
| `𝓗₁ ⊗ 𝓗₂` | `𝓗₁ ⊗ₜ[ℂ] 𝓗₂` | Tensor product |
| `exp(-iHt)` | `LinearMap.exp (-(I : ℂ) • H * t)` | Exponential map |
| `Tr(ρ)` | `Complex.linearMap.trace ρ` | Trace |

## Module Organization

```
lean_libraries/Physics/Quantum/
├── Foundations.lean      -- Core quantum mechanics
│   ├── QuantumSystem
│   ├── QuantumState
│   ├── noCloningTheorem
│   └── uncertaintyPrinciple
│
├── HilbertSpace.lean     -- Mathematical foundations
│   ├── ComplexHilbert
│   ├── OrthonormalBasis
│   ├── spectralTheorem
│   └── rieszRepresentation
│
├── Operators.lean        -- Operator algebra
│   ├── SelfAdjointOperator
│   ├── UnitaryOperator
│   ├── ProjectionOperator
│   ├── commutator
│   └── uncertaintyFromCommutator
│
└── Entanglement.lean     -- Quantum correlations
    ├── CompositeSystem
    ├── IsEntangled
    ├── BellState
    ├── schmidtDecomposition
    ├── bellsTheorem
    └── monogamyTheorem
```

## Integration Examples

### With Hephaestus Bridge
```lean
def quantumTask (problem : String) : IO Unit := do
  let system ← Hephaestus.delegate "quantum_solver" {
    method := "expectation_value"
    observable := "Hamiltonian"
    state := current_state
  }
  IO.println s!"Result: {system.result}"
```

### With Knowledge Engine
```lean
def addQuantumTheorem (thm : Name) : MetaM Unit := do
  KnowledgeEngine.addArtifact {
    name := thm
    domain := "quantum_mechanics"
    formalization := ← getProofTerm thm
    verification := "mathlib4_checked"
  }
```

### With ACE Analytics
```lean
def quantumLearning (data : List State) : IO Model := do
  let model ← ACE.train {
    algorithm := "quantum_inspired"
    features := data.map (·.toFeatureVector)
    kernel := "hilbert_space"
  }
  return model
```

## Testing

Run all quantum mechanics tests:
```bash
lake build lean_libraries/Physics/Quantum
```

Test specific file:
```bash
lake lean lean_libraries/Physics/Quantum/Foundations.lean
```

Check for errors:
```bash
lake exe lean getLib? lean_libraries/Physics/Quantum/Foundations.lean
```

## Performance Notes

- **Finite dimensions only** - All Hilbert spaces assume finite dimension
- **No numerics** - Library is purely symbolic; use numerical packages for computation
- **Typeclass inference** - Hilbert space instances are found automatically
- **Partial proofs** - Some theorems marked `sorry` need completion

## Further Reading

See `README.md` for:
- Detailed mathematics
- References to textbooks
- Future development plans
- Contribution guidelines

---

**File**: `lean_libraries/Physics/Quantum/QUICK_REFERENCE.md`
**Last Updated**: 2025-12-30
**Library Version**: 1.0.0
