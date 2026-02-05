/-
Linear Algebra in Lean 4
========================

This file contains fundamental linear algebra theorems and definitions:
- Vector spaces
- Linear transformations
- Matrices
- Eigenvalues and eigenvectors
- Inner products

Uses Mathlib's comprehensive linear algebra library.

Author: OpenEvolve LeanAide
Version: 1.0.0
-/

import Mathlib

open BigOperators
open Matrix
open FiniteDimensional

namespace LinearAlgebra

-- ============================================================================
-- Section 1: Vector Space Basics
-- ============================================================================

section VectorSpaces

-- ℝⁿ is a vector space over ℝ
theorem real_vector_space (n : ℕ) : Module ℝ (Fin n → ℝ) := by
  infer_instance

-- Zero vector is the additive identity
theorem zero_vector_identity {n : ℕ} (v : Fin n → ℝ) :
  (0 : Fin n → ℝ) + v = v := by
  simp

-- Vector addition is commutative
theorem vector_add_comm {n : ℕ} (v w : Fin n → ℝ) :
  v + w = w + v := by
  ext i
  simp [add_comm]

-- Vector addition is associative
theorem vector_add_assoc {n : ℕ} (u v w : Fin n → ℝ) :
  (u + v) + w = u + (v + w) := by
  ext i
  simp [add_assoc]

-- Scalar multiplication distributes over vector addition
theorem scalar_mul_distrib {n : ℕ} (c : ℝ) (v w : Fin n → ℝ) :
  c • (v + w) = c • v + c • w := by
  ext i
  simp [mul_add]

-- Scalar multiplication distributes over scalar addition
theorem scalar_add_distrib {n : ℕ} (c d : ℝ) (v : Fin n → ℝ) :
  (c + d) • v = c • v + d • v := by
  ext i
  simp [add_mul]

-- Scalar multiplication is associative
theorem scalar_mul_assoc {n : ℕ} (c d : ℝ) (v : Fin n → ℝ) :
  (c * d) • v = c • (d • v) := by
  ext i
  simp [mul_assoc]

-- 1 is the multiplicative identity for scalar multiplication
theorem scalar_one {n : ℕ} (v : Fin n → ℝ) :
  (1 : ℝ) • v = v := by
  ext i
  simp

end VectorSpaces

-- ============================================================================
-- Section 2: Linear Independence and Basis
-- ============================================================================

section LinearIndependence

-- Definition: A set of vectors is linearly independent if the only linear
-- combination that equals zero is the trivial combination
def LinearlyIndependent' {n m : ℕ} (vs : Fin m → Fin n → ℝ) : Prop :=
  ∀ (c : Fin m → ℝ), (∑ i, c i • vs i) = 0 → ∀ i, c i = 0

-- The zero vector is linearly dependent
theorem zero_vector_dependent {n : ℕ} :
  ¬ LinearlyIndependent' (λ (_ : Fin 1) => (0 : Fin n → ℝ)) := by
  intro h
  have h0 : (λ (_ : Fin 1) => (0 : Fin n → ℝ)) = (λ _ => 0) := by
    funext i
    rfl
  rw [h0] at h
  -- Use c = 1 which gives non-zero combination equal to 0
  specialize h (λ _ => 1)
  have h1 : (∑ i : Fin 1, (1 : ℝ) • (0 : Fin n → ℝ)) = 0 := by
    simp
  have h2 := h h1
  -- This should give 1 = 0, a contradiction
  have h3 := h2 0
  simp at h3

-- A single non-zero vector is linearly independent
theorem single_vector_independent {n : ℕ} (v : Fin n → ℝ) (hv : v ≠ 0) :
  LinearlyIndependent' (λ (_ : Fin 1) => v) := by
  intro c hsum
  have : (∑ i : Fin 1, c i • v) = c 0 • v := by
    simp [Finset.sum_singleton]
  rw [this] at hsum
  intro i
  fin_cases i
  -- If c 0 • v = 0 and v ≠ 0, then c 0 = 0
  have : c 0 = 0 := by
    by_contra h
    have : c 0 • v ≠ 0 := by
      apply smul_ne_zero
      · exact h
      · exact hv
    contradiction
  exact this

end LinearIndependence

-- ============================================================================
-- Section 3: Linear Transformations
-- ============================================================================

section LinearTransformations

-- Linear transformation definition
def IsLinear {n m : ℕ} (T : (Fin n → ℝ) → (Fin m → ℝ)) : Prop :=
  (∀ u v, T (u + v) = T u + T v) ∧ (∀ c v, T (c • v) = c • T v)

-- The zero transformation is linear
theorem zero_transformation_linear {n m : ℕ} :
  IsLinear (λ (_ : Fin n → ℝ) => (0 : Fin m → ℝ)) := by
  constructor
  · intro u v
    simp
  · intro c v
    simp

-- The identity transformation is linear
theorem identity_transformation_linear {n : ℕ} :
  IsLinear (λ v : Fin n → ℝ => v) := by
  constructor
  · intro u v
    rfl
  · intro c v
    rfl

-- Composition of linear transformations is linear
theorem composition_linear {n m p : ℕ} {T : (Fin m → ℝ) → (Fin p → ℝ)}
  {S : (Fin n → ℝ) → (Fin m → ℝ)}
  (hT : IsLinear T) (hS : IsLinear S) :
  IsLinear (λ v => T (S v)) := by
  constructor
  · intro u v
    rw [hS.1 u v]
    rw [hT.1 (S u) (S v)]
  · intro c v
    rw [hS.2 c v]
    rw [hT.2 c (S v)]

-- Matrix multiplication defines a linear transformation
theorem matrix_multiplication_linear {n m : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
  IsLinear (λ v => A.mulVec v) := by
  constructor
  · intro u v
    ext i
    simp [Matrix.mulVec, Matrix.dotProduct, add_mul]
  · intro c v
    ext i
    simp [Matrix.mulVec, Matrix.dotProduct, mul_assoc]

end LinearTransformations

-- ============================================================================
-- Section 4: Matrix Operations
-- ============================================================================

section Matrices

-- Matrix addition is commutative
theorem matrix_add_comm {n m : ℕ} (A B : Matrix (Fin m) (Fin n) ℝ) :
  A + B = B + A := by
  ext i j
  simp [add_comm]

-- Matrix addition is associative
theorem matrix_add_assoc {n m : ℕ} (A B C : Matrix (Fin m) (Fin n) ℝ) :
  (A + B) + C = A + (B + C) := by
  ext i j
  simp [add_assoc]

-- Zero matrix is the additive identity
theorem zero_matrix_identity {n m : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
  (0 : Matrix (Fin m) (Fin n) ℝ) + A = A := by
  ext i j
  simp

-- Matrix multiplication is associative (when dimensions match)
theorem matrix_mul_assoc {l m n p : ℕ} 
  (A : Matrix (Fin l) (Fin m) ℝ)
  (B : Matrix (Fin m) (Fin n) ℝ)
  (C : Matrix (Fin n) (Fin p) ℝ) :
  (A * B) * C = A * (B * C) := by
  apply Matrix.mul_assoc

-- Identity matrix is the multiplicative identity
theorem identity_matrix_mul {n m : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
  (1 : Matrix (Fin m) (Fin m) ℝ) * A = A := by
  apply Matrix.one_mul

-- Transpose of transpose is the original matrix
theorem transpose_transpose {n m : ℕ} (A : Matrix (Fin m) (Fin n) ℝ) :
  Aᵀᵀ = A := by
  apply Matrix.transpose_transpose

-- Transpose of a product reverses order
theorem transpose_mul {n m p : ℕ}
  (A : Matrix (Fin m) (Fin n) ℝ)
  (B : Matrix (Fin n) (Fin p) ℝ) :
  (A * B)ᵀ = Bᵀ * Aᵀ := by
  apply Matrix.transpose_mul

-- Trace of a square matrix
theorem trace_add {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) :
  (A + B).trace = A.trace + B.trace := by
  apply Matrix.trace_add

-- Trace is invariant under transpose
theorem trace_transpose {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) :
  Aᵀ.trace = A.trace := by
  apply Matrix.trace_transpose

-- Determinant of identity is 1
theorem det_identity {n : ℕ} :
  (1 : Matrix (Fin n) (Fin n) ℝ).det = 1 := by
  apply Matrix.det_one

-- Determinant of product is product of determinants
theorem det_mul {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℝ) :
  (A * B).det = A.det * B.det := by
  apply Matrix.det_mul

end Matrices

-- ============================================================================
-- Section 5: Inner Products and Norms
-- ============================================================================

section InnerProducts

-- Dot product is symmetric
theorem dot_product_sym {n : ℕ} (v w : Fin n → ℝ) :
  dotProduct v w = dotProduct w v := by
  simp [dotProduct, mul_comm]

-- Dot product is linear in the first argument
theorem dot_product_linear_left {n : ℕ} (c : ℝ) (u v w : Fin n → ℝ) :
  dotProduct (c • u + v) w = c * dotProduct u w + dotProduct v w := by
  simp [dotProduct, Finset.sum_add_distrib, Finset.mul_sum, mul_assoc]

-- Dot product with zero vector is zero
theorem dot_product_zero {n : ℕ} (v : Fin n → ℝ) :
  dotProduct v 0 = 0 := by
  simp [dotProduct]

-- Norm squared equals dot product with self
theorem norm_sq_eq_dot_self {n : ℕ} (v : Fin n → ℝ) :
  ‖v‖^2 = dotProduct v v := by
  simp [norm_eq_sqrt_real_inner, inner_self_eq_norm_sq]

-- Cauchy-Schwarz inequality
theorem cauchy_schwarz {n : ℕ} (u v : Fin n → ℝ) :
  |dotProduct u v| ≤ ‖u‖ * ‖v‖ := by
  have h := abs_real_inner_le_norm u v
  simp [inner_eq_sum_mul] at h
  exact h

-- Triangle inequality for vectors
theorem triangle_inequality {n : ℕ} (u v : Fin n → ℝ) :
  ‖u + v‖ ≤ ‖u‖ + ‖v‖ := by
  apply norm_add_le

-- Pythagorean theorem: if u ⟂ v then ‖u + v‖² = ‖u‖² + ‖v‖²
theorem pythagorean {n : ℕ} (u v : Fin n → ℝ)
  (h_ortho : dotProduct u v = 0) :
  ‖u + v‖^2 = ‖u‖^2 + ‖v‖^2 := by
  have h : ‖u + v‖^2 = dotProduct (u + v) (u + v) := by
    apply norm_sq_eq_dot_self
  rw [h]
  simp [dotProduct, Finset.sum_add_distrib, add_mul, mul_add]
  rw [h_ortho]
  ring

end InnerProducts

-- ============================================================================
-- Section 6: Eigenvalues and Eigenvectors
-- ============================================================================

section Eigenvalues

-- Definition: v is an eigenvector of A with eigenvalue λ if A*v = λ*v
def IsEigenvector {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (v : Fin n → ℝ) (λ_val : ℝ) : Prop :=
  v ≠ 0 ∧ A.mulVec v = λ_val • v

-- Eigenvalue equation: det(A - λI) = 0
def CharacteristicPolynomial {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (λ_val : ℝ) : ℝ :=
  (A - λ_val • (1 : Matrix (Fin n) (Fin n) ℝ)).det

-- Eigenvalues of identity matrix are all 1
theorem identity_eigenvalues {n : ℕ} (v : Fin n → ℝ) (hv : v ≠ 0) :
  IsEigenvector (1 : Matrix (Fin n) (Fin n) ℝ) v 1 := by
  constructor
  · exact hv
  · ext i
    simp [Matrix.mulVec, Matrix.dotProduct]

-- Eigenvalues of zero matrix are all 0
theorem zero_eigenvalues {n : ℕ} (v : Fin n → ℝ) (hv : v ≠ 0) :
  IsEigenvector (0 : Matrix (Fin n) (Fin n) ℝ) v 0 := by
  constructor
  · exact hv
  · ext i
    simp [Matrix.mulVec, Matrix.dotProduct]

-- If λ is eigenvalue of A, then λ² is eigenvalue of A²
theorem eigenvalue_power {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ)
  (v : Fin n → ℝ) (λ_val : ℝ)
  (h : IsEigenvector A v λ_val) :
  IsEigenvector (A ^ 2) v (λ_val ^ 2) := by
  constructor
  · exact h.1
  · have h1 : (A ^ 2).mulVec v = A.mulVec (A.mulVec v) := by
      simp [pow_two, Matrix.mulVec_mulVec]
    rw [h1, h.2]
    have h2 : A.mulVec (λ_val • v) = λ_val • (A.mulVec v) := by
      ext i
      simp [Matrix.mulVec, Matrix.dotProduct, mul_assoc]
    rw [h2, h.2]
    ext i
    simp [mul_assoc]

end Eigenvalues

-- ============================================================================
-- Section 7: Special Matrices
-- ============================================================================

section SpecialMatrices

-- Symmetric matrix: A = Aᵀ
def IsSymmetric {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  A = Aᵀ

-- Diagonal matrix multiplication is commutative
theorem diagonal_mul_comm {n : ℕ} (D₁ D₂ : Matrix (Fin n) (Fin n) ℝ)
  (hD₁ : ∀ i j, i ≠ j → D₁ i j = 0)
  (hD₂ : ∀ i j, i ≠ j → D₂ i j = 0) :
  D₁ * D₂ = D₂ * D₁ := by
  ext i j
  simp [Matrix.mul_apply]
  apply Finset.sum_congr
  · rfl
  · intro k hk
    -- Both off-diagonal elements are zero
    by_cases hik : i = k
    · by_cases hjk : j = k
      · simp [hik, hjk]
      · simp [hik, hD₂ k j (by intro h; apply hjk; linarith)]
    · simp [hD₁ i k hik]

-- Orthogonal matrix: QᵀQ = I
def IsOrthogonal {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Qᵀ * Q = 1

-- Orthogonal matrices preserve inner products
theorem orthogonal_preserves_inner {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ)
  (h_ortho : IsOrthogonal Q) (u v : Fin n → ℝ) :
  dotProduct (Q.mulVec u) (Q.mulVec v) = dotProduct u v := by
  have h : dotProduct (Q.mulVec u) (Q.mulVec v) = dotProduct u ((Qᵀ * Q).mulVec v) := by
    simp [dotProduct, Matrix.mulVec, Matrix.mulVec_mulVec, Matrix.dotProduct]
  rw [h, h_ortho]
  simp

-- Orthogonal matrices preserve norms
theorem orthogonal_preserves_norm {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ)
  (h_ortho : IsOrthogonal Q) (v : Fin n → ℝ) :
  ‖Q.mulVec v‖ = ‖v‖ := by
  have h : ‖Q.mulVec v‖^2 = ‖v‖^2 := by
    rw [norm_sq_eq_dot_self]
    rw [orthogonal_preserves_inner Q h_ortho v v]
    rw [← norm_sq_eq_dot_self]
  have h_nonneg : ‖Q.mulVec v‖ ≥ 0 := by apply norm_nonneg
  have h_nonneg2 : ‖v‖ ≥ 0 := by apply norm_nonneg
  nlinarith

end SpecialMatrices

end LinearAlgebra
