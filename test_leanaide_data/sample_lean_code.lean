import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Irrational
import Mathlib.Analysis.SpecialFunctions.Pow

/- Simple Theorems -/

theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [Nat.add_comm]

theorem mul_zero (n : Nat) : n * 0 = 0 := by
  simp

theorem even_product (a b : Nat) (ha : Even a) (hb : Even b) : Even (a * b) := by
  sorry

/- Medium Complexity Theorems -/

theorem infinitely_many_primes : Infinite {p : Nat | Nat.Prime p} := by
  sorry

theorem sqrt2_irrational : Irrational (Real.sqrt 2) := by
  sorry

/- Definitions -/

def is_even (n : Nat) : Prop :=
  ∃ k, n = 2 * k

def is_cube_free (n : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → ¬ (p ^ 3 ∣ n)

/- Theorem with Proof Sketch -/

theorem prime_divisor_of_n_fact_mul (n : Nat) (p : Nat)
    (hp : Nat.Prime p) (h : p > n) : ¬ (p ∣ (Nat.factorial n * (n + 1))) := by
  sorry

/- Complex Example -/

theorem prime_factorization_unique (n : Nat) (h : n > 1) :
    ∀ (f1 f2 : List Nat),
      (∀ p ∈ f1, Nat.Prime p) →
      (∀ p ∈ f2, Nat.Prime p) →
      f1.prod = n →
      f2.prod = n →
      f1.perm f2 := by
  sorry
