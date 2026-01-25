import Mathlib
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.TensorProduct

noncomputable section

universe u

open Complex TensorProduct

variable {ℋ : Type*} [NormedAddCommGroup ℋ] [InnerProductSpace ℂ ℋ] [CompleteSpace ℋ]

/-! ## Helper Lemmas -/

/-- Inner product on tensor product: ⟨ψ₁⊗φ₁|ψ₂⊗φ₂⟩ = ⟨ψ₁|ψ₂⟩⟨φ₁|φ₂⟩ -/
lemma inner_product_tensor {ψ₁ ψ₂ φ₁ φ₂ : ℋ} :
    inner (ψ₁ ⊗ₜ φ₁) (ψ₂ ⊗ₜ φ₂) = inner ψ₁ ψ₂ * inner φ₁ φ₂ := by
  -- Mathlib defines the inner product on the tensor product of Hilbert spaces 
  -- such that this identity holds by definition of the induced metric.
  exact InnerProductSpace.Core.inner_tmul ψ₁ φ₁ ψ₂ φ₂

/-! ## Main Theorem: No-Cloning -/

/-- **No-Cloning Theorem**
There does not exist a unitary operator U that can clone arbitrary quantum states.
Specifically, if U clones two states ψ₁ and ψ₂, their inner product must be 0 or 1.
-/
theorem no_cloning_theorem
    {ψ₁ ψ₂ : ℋ}
    (h_distinct : ψ₁ ≠ ψ₂)
    (h_nonorth : inner ψ₁ ψ₂ ≠ 0)
    (h_normalized : ‖ψ₁‖ = 1 ∧ ‖ψ₂‖ = 1) :
    ¬ ∃ (U : (ℋ ⊗[ℂ] ℋ) →L[ℂ] (ℋ ⊗[ℂ] ℋ)),
      (∀ ψ₁' ψ₂' : ℋ ⊗[ℂ] ℋ, inner (U ψ₁') (U ψ₂') = inner ψ₁' ψ₂') ∧
      (U (ψ₁ ⊗ₜ ψ₁)) = (ψ₁ ⊗ₜ ψ₁) ∧ -- (Simplified: usually U(ψ⊗0) = ψ⊗ψ)
      (∀ ψ : ℋ, ‖ψ‖ = 1 → U (ψ ⊗ₜ ψ₁) = ψ ⊗ₜ ψ) := by
  -- Standard proof uses the preservation of inner products by unitary U.
  -- Let U be a cloning machine: U(ψ ⊗ s) = ψ ⊗ ψ
  -- For two states ψ₁ and ψ₂:
  -- ⟨U(ψ₁ ⊗ s) | U(ψ₂ ⊗ s)⟩ = ⟨ψ₁ ⊗ ψ₁ | ψ₂ ⊗ ψ₂⟩
  -- By unitarity: ⟨ψ₁ ⊗ s | ψ₂ ⊗ s⟩ = ⟨ψ₁ ⊗ ψ₁ | ψ₂ ⊗ ψ₂⟩
  -- ⟨ψ₁ | ψ₂⟩ ⟨s | s⟩ = ⟨ψ₁ | ψ₂⟩ ⟨ψ₁ | ψ₂⟩
  -- Since ⟨s | s⟩ = 1, we get: ⟨ψ₁ | ψ₂⟩ = (⟨ψ₁ | ψ₂⟩)²
  
  by_contra h_exists
  obtain ⟨U, h_unitary, h_clone⟩ := h_exists
  
  -- Let s be the initial state (here we used ψ₁ as the 'blank' or auxiliary state)
  let s := ψ₁
  have h_s_norm : ‖s‖ = 1 := h_normalized.1
  have h_s_inner : inner s s = (1 : ℂ) := by 
    rw [inner_self_eq_norm_sq_to_K, h_s_norm]
    simp
    
  -- Apply the unitary property to the cloning of ψ₁ and ψ₂
  let lhs := inner (U (ψ₁ ⊗ₜ s)) (U (ψ₂ ⊗ₜ s))
  let rhs := inner (ψ₁ ⊗ₜ s) (ψ₂ ⊗ₜ s)
  
  -- 1. By Unitarity
  have h1 : lhs = rhs := h_unitary _ _
  
  -- 2. By Cloning Definition
  -- U(ψ₁ ⊗ s) = ψ₁ ⊗ ψ₁
  -- U(ψ₂ ⊗ s) = ψ₂ ⊗ ψ₂
  have h_u1 : U (ψ₁ ⊗ₜ s) = ψ₁ ⊗ₜ ψ₁ := by apply h_clone; exact h_normalized.1
  have h_u2 : U (ψ₂ ⊗ₜ s) = ψ₂ ⊗ₜ ψ₂ := by apply h_clone; exact h_normalized.2
  
  have h2 : lhs = inner (ψ₁ ⊗ₜ ψ₁) (ψ₂ ⊗ₜ ψ₂) := by rw [h_u1, h_u2]
  
  -- 3. Expand inner products
  rw [inner_product_tensor] at h1
  rw [inner_product_tensor] at h2
  
  -- Combine
  rw [h1, h_s_inner, mul_one] at h2
  
  -- We have: ⟨ψ₁|ψ₂⟩ = ⟨ψ₁|ψ₂⟩ * ⟨ψ₁|ψ₂⟩
  -- Which is: z = z²  => z(1-z) = 0
  set z := inner ψ₁ ψ₂
  have h_eq : z = z * z := h2
  
  have h_final : z = 0 ∨ z = 1 := by
    have : z * (1 - z) = 0 := by 
      rw [mul_sub, mul_one, ← h_eq]
      simp
    exact mul_eq_zero.mp this
    
  cases h_final with
  | inl h_zero => exact h_nonorth h_zero
  | inr h_one => 
      -- If ⟨ψ₁|ψ₂⟩ = 1 and both are normalized, then ψ₁ = ψ₂
      have h_eq_states : ψ₁ = ψ₂ := by
        insert h_normalized
        rw [← inner_self_eq_norm_sq_to_K] at h_normalized
        -- Use the fact that equality in Cauchy-Schwarz for normalized vectors implies equality
        apply eq_of_inner_eq_norm_mul_norm
        rw [h_one, h_normalized.1, h_normalized.2, mul_one]
        simp
      exact h_distinct h_eq_states