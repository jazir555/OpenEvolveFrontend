import Physics.Quantum.Foundations
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Normed.Operator.Basic

/-!
# Quantum Dynamics

This file implements the Schrödinger Equation and the laws of quantum evolution.
-/

noncomputable section

open Complex InnerProductSpace

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℂ H] [CompleteSpace H]

/--
The Time-Dependent Schrödinger Equation (TDSE).
iħ ∂/∂t |ψ(t)⟩ = H |ψ(t)⟩
We use units where ħ = 1.
-/
def satisfies_tdse (H_op : SelfAdjointOperator H) (ψ : ℝ → H) : Prop :=
  ∀ t, HasDerivAt ψ (I • (H_op.op (ψ t))) t

/--
Stationary States (Time-Independent Schrödinger Equation).
H |ψ⟩ = E |ψ⟩
-/
def is_stationary_state (H_op : SelfAdjointOperator H) (ψ : H) (E : ℝ) : Prop :=
  H_op.op ψ = E • ψ

/--
Unitary Propagator.
If H is time-independent, U(t) = exp(-iHt).
-/
def is_unitary_propagator (H_op : SelfAdjointOperator H) (U : ℝ → (H →L[ℂ] H)) : Prop :=
  ∀ (t : ℝ) (ψ₀ : H), satisfies_tdse H_op (fun s => U s ψ₀) ∧ U 0 = 1

/--
Conservation of Probability.
The norm of a state evolving under TDSE is constant in time.
-/
theorem probability_conservation (H_op : SelfAdjointOperator H) (ψ : ℝ → H) 
  (h : satisfies_tdse H_op ψ) :
  ∀ t₁ t₂, ‖ψ t₁‖ = ‖ψ t₂‖ := by
  -- Proof: show d/dt ⟨ψ|ψ⟩ = 0
  -- d/dt ⟨ψ|ψ⟩ = ⟨ψ'|ψ⟩ + ⟨ψ|ψ'⟩ = ⟨iHψ|ψ⟩ + ⟨ψ|iHψ⟩ = -i⟨Hψ|ψ⟩ + i⟨ψ|Hψ⟩
  -- Since H is self-adjoint, ⟨Hψ|ψ⟩ = ⟨ψ|Hψ⟩, so the sum is zero.
  sorry