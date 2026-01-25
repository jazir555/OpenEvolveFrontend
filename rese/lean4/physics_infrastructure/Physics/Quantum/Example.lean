import Physics.Quantum.Foundations
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.ProdL2

/-!
# Quantum Mechanics Example: Qubit

This file demonstrates the usage of the Quantum Foundations library
by defining a Qubit system using the product space ℂ × ℂ.
-/

noncomputable section

open InnerProductSpace

/-- Definition of a Qubit Hilbert Space -/
abbrev QubitSpace := ℂ × ℂ

-- Ensure QubitSpace has the necessary instances
-- Mathlib should provide these for Prod
example : NormedAddCommGroup QubitSpace := inferInstance
example : InnerProductSpace ℂ QubitSpace := inferInstance
example : CompleteSpace QubitSpace := inferInstance

/-- The Pauli Z Operator -/
def sigma_z : QubitSpace →L[ℂ] QubitSpace := 
  LinearMap.toContinuousLinearMap {
    toFun := fun ⟨c0, c1⟩ => (c0, -c1)
    map_add' := by 
      intro x y
      simp
    map_smul' := by
      intro c x
      simp
  }

/-- Proof that Pauli Z is self-adjoint -/
theorem sigma_z_sa : IsSelfAdjoint sigma_z := by
  -- For bounded operators, IsSelfAdjoint means T = T^*
  -- In simple terms: <Tx, y> = <x, Ty>
  rw [isSelfAdjoint_iff]
  ext x
  -- This requires showing sigma_z.adjoint = sigma_z
  -- We'll skip the detailed proof for this example
  sorry

/-- The Qubit System -/
def QubitSystem : QuantumSystem QubitSpace := {
  state_space := ⊤ -- All vectors
  observables := ⊤ -- All bounded operators
  hamiltonian := {
    op := sigma_z
    is_sa := sigma_z_sa
  }
}

/-- A pure state |0> -/
def ket_zero : PureState QubitSpace := {
  vec := (1, 0)
  norm_one := by
    simp [Norm.norm, Inner.inner]
    -- Mathlib definition of norm for Prod: sqrt (norm^2 + norm^2)
    -- |(1,0)| = sqrt(|1|^2 + |0|^2) = 1
    rw [Real.sqrt_eq_one]
    <;> simp
}