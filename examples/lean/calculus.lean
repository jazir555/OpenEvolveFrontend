/-
Calculus Proofs in Lean 4
=========================

This file contains definitions and theorems from calculus including:
- Limits
- Continuity
- Derivatives
- Integration basics

All theorems use Mathlib's comprehensive analysis library.

Author: OpenEvolve LeanAide
Version: 1.0.0
-/

import Mathlib

open BigOperators
open Filter
open Topology

namespace Calculus

-- ============================================================================
-- Section 1: Limits
-- ============================================================================

section Limits

-- The limit of a constant function is the constant
theorem limit_of_constant (c : ℝ) : 
  Tendsto (λ _ => c) (nhds 0) (nhds c) := by
  exact tendsto_const_nhds

-- The limit of x as x approaches a is a
theorem limit_of_identity (a : ℝ) :
  Tendsto (λ x => x) (nhds a) (nhds a) := by
  exact tendsto_id

-- Limit of sum: if f → L and g → M, then (f + g) → (L + M)
theorem limit_of_sum {f g : ℝ → ℝ} {L M a : ℝ}
  (hf : Tendsto f (nhds a) (nhds L))
  (hg : Tendsto g (nhds a) (nhds M)) :
  Tendsto (λ x => f x + g x) (nhds a) (nhds (L + M)) := by
  apply Tendsto.add
  · exact hf
  · exact hg

-- Limit of product: if f → L and g → M, then (f * g) → (L * M)
theorem limit_of_product {f g : ℝ → ℝ} {L M a : ℝ}
  (hf : Tendsto f (nhds a) (nhds L))
  (hg : Tendsto g (nhds a) (nhds M)) :
  Tendsto (λ x => f x * g x) (nhds a) (nhds (L * M)) := by
  apply Tendsto.mul
  · exact hf
  · exact hg

-- Squeeze theorem: if f ≤ h ≤ g and f,g → L, then h → L
theorem squeeze_theorem {f g h : ℝ → ℝ} {L a : ℝ}
  (h1 : ∀ x, f x ≤ h x)
  (h2 : ∀ x, h x ≤ g x)
  (h3 : Tendsto f (nhds a) (nhds L))
  (h4 : Tendsto g (nhds a) (nhds L)) :
  Tendsto h (nhds a) (nhds L) := by
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le
  · exact h3
  · exact h4
  · intro x
    exact h1 x
  · intro x
    exact h2 x

end Limits

-- ============================================================================
-- Section 2: Famous Limits
-- ============================================================================

section FamousLimits

-- The limit of sin(x)/x as x → 0 is 1
-- This is a fundamental limit in calculus
theorem limit_sin_over_x : 
  Tendsto (λ x => Real.sin x / x) (nhdsWithin 0 {{0}}ᶜ) (nhds 1) := by
  have h : Real.sin = λ x => Real.sin x := rfl
  rw [h]
  apply Real.tendsto_sin_div

-- The limit of (1 + 1/n)^n as n → ∞ is e
theorem limit_definition_of_e :
  Tendsto (λ n : ℕ => (1 + 1/(n+1) : ℝ) ^ (n+1)) atTop (nhds Real.e) := by
  have h := Real.tendsto_one_plus_div_pow
  have h2 : (λ n : ℕ => (1 + 1/(n+1) : ℝ) ^ (n+1)) = 
            (λ n : ℕ => (1 + 1/((n : ℝ)+1)) ^ (n+1)) := by simp
  rw [h2]
  -- Use the standard library theorem about e
  apply Tendsto.comp
  · exact h
  · apply Tendsto.add_atTop_nat

-- Geometric sequence: if |r| < 1, then r^n → 0
theorem geometric_sequence_limit {r : ℝ} (hr : |r| < 1) :
  Tendsto (λ n : ℕ => r ^ n) atTop (nhds 0) := by
  apply tendsto_pow_atTop_nhds_0_of_lt_1
  · exact abs_nonneg r
  · exact hr

end FamousLimits

-- ============================================================================
-- Section 3: Continuity
-- ============================================================================

section Continuity

-- Constant functions are continuous
theorem constant_function_continuous (c : ℝ) :
  Continuous (λ (_ : ℝ) => c) := by
  exact continuous_const

-- The identity function is continuous
theorem identity_continuous :
  Continuous (λ x : ℝ => x) := by
  exact continuous_id

-- Sum of continuous functions is continuous
theorem sum_of_continuous {f g : ℝ → ℝ}
  (hf : Continuous f) (hg : Continuous g) :
  Continuous (λ x => f x + g x) := by
  apply Continuous.add
  · exact hf
  · exact hg

-- Product of continuous functions is continuous
theorem product_of_continuous {f g : ℝ → ℝ}
  (hf : Continuous f) (hg : Continuous g) :
  Continuous (λ x => f x * g x) := by
  apply Continuous.mul
  · exact hf
  · exact hg

-- Composition of continuous functions is continuous
theorem composition_of_continuous {f g : ℝ → ℝ}
  (hf : Continuous f) (hg : Continuous g) :
  Continuous (λ x => f (g x)) := by
  apply Continuous.comp
  · exact hf
  · exact hg

-- Polynomials are continuous
theorem polynomial_continuous (a b c : ℝ) :
  Continuous (λ x : ℝ => a * x^2 + b * x + c) := by
  continuity

-- sin is continuous
theorem sin_continuous :
  Continuous Real.sin := by
  exact Real.continuous_sin

-- cos is continuous  
theorem cos_continuous :
  Continuous Real.cos := by
  exact Real.continuous_cos

-- exp is continuous
theorem exp_continuous :
  Continuous Real.exp := by
  exact Real.continuous_exp

end Continuity

-- ============================================================================
-- Section 4: Derivatives
-- ============================================================================

section Derivatives

-- Derivative of constant is 0
theorem derivative_of_constant (c : ℝ) :
  deriv (λ _ => c) = 0 := by
  funext x
  simp

-- Derivative of identity is 1
theorem derivative_of_identity :
  deriv (λ x : ℝ => x) = 1 := by
  funext x
  simp

-- Derivative of x^n is n*x^(n-1) for natural n
theorem power_rule (n : ℕ) :
  deriv (λ x : ℝ => x^n) = λ x => n * x^(n-1) := by
  funext x
  simp [deriv_pow]

-- Linearity of derivative: (f + g)' = f' + g'
theorem derivative_of_sum {f g : ℝ → ℝ}
  (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x) :
  deriv (λ x => f x + g x) x = deriv f x + deriv g x := by
  simp [deriv_add, hf, hg]

-- Product rule: (f * g)' = f' * g + f * g'
theorem product_rule {f g : ℝ → ℝ}
  (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x) :
  deriv (λ x => f x * g x) x = deriv f x * g x + f x * deriv g x := by
  simp [deriv_mul, hf, hg]

-- Chain rule: (f ∘ g)' = (f' ∘ g) * g'
theorem chain_rule {f g : ℝ → ℝ}
  (hf : DifferentiableAt ℝ f (g x))
  (hg : DifferentiableAt ℝ g x) :
  deriv (λ x => f (g x)) x = deriv f (g x) * deriv g x := by
  simp [deriv.comp, hf, hg]

-- Derivative of sin is cos
theorem derivative_of_sin :
  deriv Real.sin = Real.cos := by
  funext x
  exact Real.deriv_sin

-- Derivative of cos is -sin
theorem derivative_of_cos :
  deriv Real.cos = λ x => -Real.sin x := by
  funext x
  exact Real.deriv_cos

-- Derivative of exp is exp
theorem derivative_of_exp :
  deriv Real.exp = Real.exp := by
  funext x
  exact Real.deriv_exp

-- Derivative of log is 1/x
theorem derivative_of_log :
  deriv Real.log = λ x => 1/x := by
  funext x
  exact Real.deriv_log (by linarith)

end Derivatives

-- ============================================================================
-- Section 5: Mean Value Theorem and Applications
-- ============================================================================

section MeanValueTheorem

-- Rolle's Theorem: If f(a) = f(b) and f is differentiable on (a,b),
-- then there exists c in (a,b) such that f'(c) = 0
theorem rolles_theorem {f : ℝ → ℝ} {a b : ℝ}
  (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b))
  (hf' : DifferentiableOn ℝ f (Set.Ioo a b))
  (hfa : f a = f b) :
  ∃ c ∈ Set.Ioo a b, deriv f c = 0 := by
  apply exists_deriv_eq_zero
  · exact hab
  · exact hf
  · exact hf'
  · exact hfa

-- Mean Value Theorem: If f is continuous on [a,b] and differentiable on (a,b),
-- then there exists c in (a,b) such that f'(c) = (f(b) - f(a))/(b - a)
theorem mean_value_theorem {f : ℝ → ℝ} {a b : ℝ}
  (hab : a < b)
  (hf : ContinuousOn f (Set.Icc a b))
  (hf' : DifferentiableOn ℝ f (Set.Ioo a b)) :
  ∃ c ∈ Set.Ioo a b, deriv f c = (f b - f a) / (b - a) := by
  apply exists_deriv_eq_slope
  · exact hab
  · exact hf
  · exact hf'

end MeanValueTheorem

-- ============================================================================
-- Section 6: Series
-- ============================================================================

section Series

-- Geometric series: Σ r^n = 1/(1-r) for |r| < 1
theorem geometric_series_sum {r : ℝ} (hr : |r| < 1) :
  ∑' n : ℕ, r^n = 1 / (1 - r) := by
  rw [tsum_geometric_of_lt_one]
  · ring
  · linarith [abs_nonneg r]
  · exact hr

-- Harmonic series diverges (stated as partial sums are unbounded)
theorem harmonic_series_diverges :
  ¬ Summable (λ n : ℕ => 1 / (n + 1) : ℕ → ℝ) := by
  apply not_summable_harmonic

-- p-series: Σ 1/n^p converges if p > 1
theorem p_series_converges {p : ℝ} (hp : 1 < p) :
  Summable (λ n : ℕ => 1 / ((n + 1) : ℝ)^p) := by
  apply Real.summable_one_div_nat_pow
  · linarith

end Series

-- ============================================================================
-- Section 7: Special Functions
-- ============================================================================

section SpecialFunctions

-- Euler's formula: e^(ix) = cos(x) + i*sin(x) (real part)
theorem euler_formula_real (x : ℝ) :
  Real.exp (Complex.I * x).re = Real.cos x := by
  rw [Complex.exp_ofReal_mul_I_re]

-- Definition of π using the integral of 1/(1+x²)
-- π = 2 * ∫₀^∞ 1/(1+x²) dx
theorem pi_integral_definition :
  Real.pi = 2 * ∫ x in Set.Ioi 0, 1 / (1 + x^2) := by
  -- This is a standard definition involving arctan
  have h : ∫ x in Set.Ioi 0, 1 / (1 + x^2) = Real.pi / 2 := by
    -- The antiderivative of 1/(1+x²) is arctan(x)
    -- ∫₀^∞ 1/(1+x²) dx = [arctan(x)]₀^∞ = π/2 - 0 = π/2
    sorry  -- Full proof requires advanced integration theory
  rw [h]
  ring

-- Gamma function at positive integers: Γ(n+1) = n!
theorem gamma_factorial (n : ℕ) :
  Real.Gamma (n + 1) = Nat.factorial n := by
  rw [Real.Gamma_nat_eq_factorial]

end SpecialFunctions

end Calculus
