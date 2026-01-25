import Mathlib
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Adjoint
import Mathlib.LinearAlgebra.SelfAdjoint

noncomputable section

universe u

open Complex
open InnerProductSpace

variable {ℋ : Type*} [NormedAddCommGroup ℋ] [InnerProductSpace ℂ ℋ] [CompleteSpace ℋ]

/-! ## Observable Structure -/

structure Observable where
  operator : ℋ →L[ℂ] ℋ
  self_adjoint : ∀ ψ φ : ℋ, ⟪operator ψ, φ⟫_ℂ = ⟪ψ, operator φ⟫_ℂ

/-! ## Pure State Structure -/

structure PureState where
  vector : ℋ
  normalized : ‖vector‖ = 1

/-! ## Commutator Definition -/

def commutator (A B : ℋ →L[ℂ] ℋ) : ℋ →L[ℂ] ℋ :=
  A ∘L B - B ∘L A

/-! ## Statistical Quantities -/

namespace Observable

def expectation (A : Observable) (ψ : PureState) : ℝ :=
  re ⟪ψ.vector, A.operator ψ.vector⟫_ℂ

def variance (A : Observable) (ψ : PureState) : ℝ :=
  ‖A.operator ψ.vector - (A.expectation ψ : ℂ) • ψ.vector‖ ^ 2

noncomputable def stdDev (A : Observable) (ψ : PureState) : ℝ :=
  Real.sqrt (A.variance ψ)

end Observable

/-! ## Helper Lemmas -/

lemma im_inner_self_adjoint (T : ℋ →L[ℂ] ℋ) (h_self : ∀ ψ φ, ⟪T ψ, φ⟫_ℂ = ⟪ψ, T φ⟫_ℂ) 
    (ψ : ℋ) : im ⟪ψ, T ψ⟫_ℂ = 0 := by
  have h : ⟪ψ, T ψ⟫_ℂ = conj ⟪ψ, T ψ⟫_ℂ := by
    rw [← inner_conj_sym, h_self ψ ψ]
  have : im (⟪ψ, T ψ⟫_ℂ) = im (conj ⟪ψ, T ψ⟫_ℂ) := by rw [h]
  rw [Complex.conj_im] at this
  linarith

lemma re_inner_anti_self_adjoint (T : ℋ →L[ℂ] ℋ) (h_anti : ∀ ψ φ, ⟪T ψ, φ⟫_ℂ = -⟪ψ, T φ⟫_ℂ) 
    (ψ : ℋ) : re ⟪ψ, T ψ⟫_ℂ = 0 := by
  have h : ⟪ψ, T ψ⟫_ℂ = -conj ⟪ψ, T ψ⟫_ℂ := by
    rw [← inner_conj_sym, h_anti ψ ψ, neg_neg]
  have : re (⟪ψ, T ψ⟫_ℂ) = re (-conj ⟪ψ, T ψ⟫_ℂ) := by rw [h]
  rw [Complex.neg_re, Complex.conj_re] at this
  linarith

lemma commutator_centered_eq (A B : Observable) (ψ : PureState) :
    commutator (A.operator - (A.expectation ψ : ℂ) • (1 : ℋ →L[ℂ] ℋ))
              (B.operator - (B.expectation ψ : ℂ) • (1 : ℋ →L[ℂ] ℋ)) = 
    commutator A.operator B.operator := by
  ext x
  simp [commutator, sub_smul, smul_sub, ContinuousLinearMap.one_apply, 
        ContinuousLinearMap.smul_apply, ContinuousLinearMap.sub_apply,
        ContinuousLinearMap.comp_apply]
  ring

/-! ## Main Theorem -/

theorem robertson_schrodinger_uncertainty (A B : Observable) (ψ : PureState) :
    A.stdDev ψ * B.stdDev ψ ≥ |im ⟪ψ.vector, commutator A.operator B.operator ψ.vector⟫_ℂ| / 2 := by
  -- Define centered operators
  set a := A.expectation ψ with ha
  set b := B.expectation ψ with hb
  set A' := A.operator - (a : ℂ) • (1 : ℋ →L[ℂ] ℋ) with hA'
  set B' := B.operator - (b : ℂ) • (1 : ℋ →L[ℂ] ℋ) with hB'
  
  -- Prove centered operators are self-adjoint
  have hA'_self : ∀ ψ φ, ⟪A' ψ, φ⟫_ℂ = ⟪ψ, A' φ⟫_ℂ := by
    intro ψ φ
    simp [A', inner_sub_left, inner_sub_right, inner_smul_left, inner_smul_right, 
          A.self_adjoint, ContinuousLinearMap.one_apply]
    ring
    
  have hB'_self : ∀ ψ φ, ⟪B' ψ, φ⟫_ℂ = ⟪ψ, B' φ⟫_ℂ := by
    intro ψ φ
    simp [B', inner_sub_left, inner_sub_right, inner_smul_left, inner_smul_right,
          B.self_adjoint, ContinuousLinearMap.one_apply]
    ring
  
  -- Relate standard deviations to norms of centered operators
  have h_stdA : A.stdDev ψ = ‖A' ψ.vector‖ := by
    rw [Observable.stdDev, Observable.variance]
    simp [A', a, ha, norm_sub_sq, ψ.normalized, ← norm_sq_eq_inner]
    ring
    
  have h_stdB : B.stdDev ψ = ‖B' ψ.vector‖ := by
    rw [Observable.stdDev, Observable.variance]
    simp [B', b, hb, norm_sub_sq, ψ.normalized, ← norm_sq_eq_inner]
    ring
  
  rw [h_stdA, h_stdB]
  
  -- Cauchy-Schwarz inequality
  have h_cs : ‖A' ψ.vector‖ * ‖B' ψ.vector‖ ≥ Complex.abs ⟪A' ψ.vector, B' ψ.vector⟫_ℂ := by
    calc
      ‖A' ψ.vector‖ * ‖B' ψ.vector‖ ≥ ‖⟪A' ψ.vector, B' ψ.vector⟫_ℂ‖ := by
        rw [← norm_inner_le_norm]
      _ = Complex.abs ⟪A' ψ.vector, B' ψ.vector⟫_ℂ := rfl
  
  set z := ⟪A' ψ.vector, B' ψ.vector⟫_ℂ with hz
  
  -- Bound on imaginary part
  have h_im_bound : Complex.abs z ≥ Complex.abs (im z) :=
    calc
      Complex.abs z ≥ Complex.abs (im z) := by
        rw [← Complex.abs_im_le_abs z]
      _ = |im z| := by rw [Complex.abs_ofReal]
  
  -- Compute imaginary part in terms of commutator
  have h_im_z : im z = (1/2 : ℂ) * im (⟪ψ.vector, commutator A.operator B.operator ψ.vector⟫_ℂ) := by
    let S := A' ∘L B' + B' ∘L A'
    let C := A' ∘L B' - B' ∘L A'
    
    have hS_self : ∀ ψ φ, ⟪S ψ, φ⟫_ℂ = ⟪ψ, S φ⟫_ℂ := by
      intro ψ φ
      simp [S, inner_add_left, inner_add_right, hA'_self, hB'_self]
      ring
      
    have hC_anti : ∀ ψ φ, ⟪C ψ, φ⟫_ℂ = -⟪ψ, C φ⟫_ℂ := by
      intro ψ φ
      simp [C, inner_sub_left, inner_sub_right, hA'_self, hB'_self]
      ring
    
    have h_decomp : A' ∘L B' = (1/2 : ℂ) • S + (1/2 : ℂ) • C := by
      ext ψ
      simp [S, C, smul_add, smul_sub]
      abel
    
    have hz' : z = ⟪ψ.vector, A' (B' ψ.vector)⟫_ℂ := by
      rw [hA'_self]
      
    rw [← ContinuousLinearMap.comp_apply, h_decomp] at hz'
    simp [hz', inner_add_right, inner_smul_right] at hz
    rw [hz]
    
    have hS_im : im (⟪ψ.vector, S ψ.vector⟫_ℂ) = 0 :=
      im_inner_self_adjoint S hS_self ψ.vector
    
    have hC_eq : C = commutator A.operator B.operator := by
      rw [← commutator_centered_eq A B ψ]
      
    simp [hS_im, hC_eq]
  
  -- Combine all inequalities
  rw [h_im_z] at h_im_bound
  have h_abs_im : Complex.abs ((1/2 : ℂ) * im (⟪ψ.vector, commutator A.operator B.operator ψ.vector⟫_ℂ)) = 
      (1/2 : ℝ) * |im (⟪ψ.vector, commutator A.operator B.operator ψ.vector⟫_ℂ)| := by
    rw [Complex.abs_mul, Complex.abs_ofReal, abs_of_pos (by norm_num : (0:ℝ) < 1/2)]
    
  rw [h_abs_im] at h_im_bound
  linarith [h_cs, h_im_bound]

/-! ## Special Case: Position-Momentum Uncertainty -/

theorem position_momentum_uncertainty (x p : Observable) (ψ : PureState) (hbar : ℝ) 
    (h_pos : 0 < hbar) (h_comm : commutator x.operator p.operator = (Complex.I * hbar) • (1 : ℋ →L[ℂ] ℋ)) :
    x.stdDev ψ * p.stdDev ψ ≥ hbar / 2 := by
  have h_main := robertson_schrodinger_uncertainty x p ψ
  have h_val : im ⟪ψ.vector, commutator x.operator p.operator ψ.vector⟫_ℂ = hbar := by
    rw [h_comm]
    simp [inner_smul_right, ψ.normalized, ← norm_sq_eq_inner]
  rw [h_val] at h_main
  have : |hbar| = hbar := abs_of_pos h_pos
  rw [this] at h_main
  exact h_main

end