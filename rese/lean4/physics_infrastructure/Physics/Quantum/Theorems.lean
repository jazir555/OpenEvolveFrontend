import Physics.Quantum.Foundations
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.Analysis.InnerProductSpace.TensorProduct

/-!
# Quantum Theorems

This file implements key theorems of quantum mechanics as specified in the
Physics Knowledge Engine (System 2).
-/

noncomputable section

open ComplexConjugate InnerProductSpace TensorProduct

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H]

-- Assume the tensor product space is complete (Hilbert tensor product)
-- This is automatically true for finite dimensions, but needs this assumption for the general case
-- when using the algebraic tensor product type as the base.
variable [CompleteSpace (H ⊗[ℂ] H)]

/--
The No-Cloning Theorem.
It is impossible to create an independent and identical copy of an arbitrary 
unknown quantum state.

Statement: There is no unitary operator U such that for all states |ψ⟩ and |φ⟩:
U(|ψ⟩ ⊗ |s⟩) = |ψ⟩ ⊗ |ψ⟩
-/
theorem no_cloning (s : H) (hs : ‖s‖ = 1) :
  ¬ ∃ (U : (H ⊗[ℂ] H) →L[ℂ] (H ⊗[ℂ] H)), 
    (U.comp U.adjoint = 1 ∧ U.adjoint.comp U = 1) ∧ 
    (∀ ψ : H, ‖ψ‖ = 1 → U (ψ ⊗ₜ s) = ψ ⊗ₜ ψ) := by
  -- Proof by contradiction:
  -- If such a U existed, it would have to preserve inner products.
  -- ⟨ψ|φ⟩ = ⟨ψ|ψ⟩⟨s|s⟩ = ⟨U(ψ⊗s)|U(φ⊗s)⟩ = ⟨ψ|φ⟩⟨ψ|φ⟩ = ⟨ψ|φ⟩²
  -- This only holds if ⟨ψ|φ⟩ is 0 or 1, which contradicts the "for all ψ" condition.
  sorry

/--
Superposition Principle.
Any two valid quantum states can be added together ("superposed") to form 
another valid quantum state.
-/
theorem superposition_principle (ψ₁ ψ₂ : PureState H) (a b : ℂ) :
  ‖a • ψ₁.vec + b • ψ₂.vec‖ = 1 → 
  ∃ (ψ₃ : PureState H), ψ₃.vec = a • ψ₁.vec + b • ψ₂.vec := by
  intro h
  exact ⟨⟨a • ψ₁.vec + b • ψ₂.vec, h⟩, rfl⟩
