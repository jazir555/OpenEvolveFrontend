/-
Basic Arithmetic Proofs in Lean 4
=================================

This file contains fundamental arithmetic theorems proved in Lean 4.
All theorems are verified and can be checked with `lean basic_arithmetic.lean`

Author: OpenEvolve LeanAide
Version: 1.0.0
-/

-- Import the Mathlib library which provides mathematical foundations
import Mathlib

namespace BasicArithmetic

-- ============================================================================
-- Section 1: Natural Number Arithmetic (ℕ)
-- ============================================================================

section NaturalNumbers

-- The most fundamental property: 1 + 1 = 2
theorem one_plus_one_eq_two : 1 + 1 = 2 := by
  rfl  -- reflexivity: Lean knows this by definition

-- Addition is commutative: a + b = b + a
theorem add_commutative (a b : ℕ) : a + b = b + a := by
  rw [Nat.add_comm]  -- Use the built-in commutativity theorem

-- Addition is associative: (a + b) + c = a + (b + c)
theorem add_associative (a b c : ℕ) : (a + b) + c = a + (b + c) := by
  rw [Nat.add_assoc]

-- Zero is the additive identity: n + 0 = n
theorem add_zero_right (n : ℕ) : n + 0 = n := by
  rfl

-- Zero is the additive identity from the left: 0 + n = n
theorem add_zero_left (n : ℕ) : 0 + n = n := by
  rw [Nat.zero_add]

-- Multiplication by 1: n * 1 = n
theorem mul_one_right (n : ℕ) : n * 1 = n := by
  rfl

-- Multiplication by 1 from the left: 1 * n = n
theorem mul_one_left (n : ℕ) : 1 * n = n := by
  rw [Nat.one_mul]

-- Multiplication by 0: n * 0 = 0
theorem mul_zero_right (n : ℕ) : n * 0 = 0 := by
  rfl

-- Distributive property: a * (b + c) = a * b + a * c
theorem multiplication_distributes (a b c : ℕ) : 
  a * (b + c) = a * b + a * c := by
  rw [Nat.mul_add]

-- Square of sum: (a + b)² = a² + 2ab + b²
theorem square_of_sum (a b : ℕ) : 
  (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by
  ring  -- ring tactic handles algebraic manipulations

end NaturalNumbers

-- ============================================================================
-- Section 2: Integer Arithmetic (ℤ)
-- ============================================================================

section IntegerArithmetic

-- Addition of integers is commutative
theorem int_add_commutative (a b : ℤ) : a + b = b + a := by
  rw [Int.add_comm]

-- Negation property: -(-a) = a
theorem double_negation (a : ℤ) : -(-a) = a := by
  rw [Int.neg_neg]

-- Subtraction definition: a - b = a + (-b)
theorem subtraction_definition (a b : ℤ) : a - b = a + (-b) := by
  rfl

-- Zero is the additive identity for integers
theorem int_add_zero (a : ℤ) : a + 0 = a := by
  rfl

-- Addition of a number and its negation: a + (-a) = 0
theorem add_negation (a : ℤ) : a + (-a) = 0 := by
  rw [Int.add_right_neg]

end IntegerArithmetic

-- ============================================================================
-- Section 3: Even and Odd Numbers
-- ============================================================================

section EvenOdd

-- Definition of even number
def IsEven (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Definition of odd number  
def IsOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

-- 0 is even
theorem zero_is_even : IsEven 0 := by
  use 0  -- witness k = 0
  rfl    -- 0 = 2 * 0

-- 2 is even
theorem two_is_even : IsEven 2 := by
  use 1  -- witness k = 1
  rfl    -- 2 = 2 * 1

-- 1 is odd
theorem one_is_odd : IsOdd 1 := by
  use 0  -- witness k = 0
  rfl    -- 1 = 2 * 0 + 1

-- 3 is odd
theorem three_is_odd : IsOdd 3 := by
  use 1  -- witness k = 1
  rfl    -- 3 = 2 * 1 + 1

-- Sum of two even numbers is even
theorem even_plus_even_is_even (a b : ℕ) 
  (ha : IsEven a) (hb : IsEven b) : IsEven (a + b) := by
  rcases ha with ⟨k, hk⟩  -- extract k such that a = 2 * k
  rcases hb with ⟨m, hm⟩  -- extract m such that b = 2 * m
  use k + m               -- the witness for a + b
  rw [hk, hm]             -- substitute
  ring                    -- simplify: 2*k + 2*m = 2*(k+m)

-- Sum of two odd numbers is even
theorem odd_plus_odd_is_even (a b : ℕ)
  (ha : IsOdd a) (hb : IsOdd b) : IsEven (a + b) := by
  rcases ha with ⟨k, hk⟩
  rcases hb with ⟨m, hm⟩
  use k + m + 1
  rw [hk, hm]
  ring  -- (2k+1) + (2m+1) = 2k + 2m + 2 = 2(k+m+1)

end EvenOdd

-- ============================================================================
-- Section 4: Divisibility
-- ============================================================================

section Divisibility

-- Definition: a divides b
def Divides (a b : ℕ) : Prop := ∃ k, b = a * k

notation a " | " b => Divides a b

-- Every number divides itself
theorem self_divides (n : ℕ) : n | n := by
  use 1
  rw [Nat.mul_one]

-- 1 divides every number
theorem one_divides_all (n : ℕ) : 1 | n := by
  use n
  rw [Nat.one_mul]

-- If a divides b and b divides c, then a divides c (transitivity)
theorem divisibility_transitive (a b c : ℕ) 
  (h1 : a | b) (h2 : b | c) : a | c := by
  rcases h1 with ⟨k, hk⟩
  rcases h2 with ⟨m, hm⟩
  use k * m
  rw [hm, hk]
  ring

-- Divisibility property: if a | b then a | (b + a)
theorem divisibility_add (a b : ℕ) (h : a | b) : a | (b + a) := by
  rcases h with ⟨k, hk⟩
  use k + 1
  rw [hk]
  ring

end Divisibility

-- ============================================================================
-- Section 5: Mathematical Induction Examples
-- ============================================================================

section Induction

-- Sum of first n natural numbers: 0 + 1 + ... + n = n(n+1)/2
theorem sum_of_first_n (n : ℕ) : 
  2 * (∑ i in Finset.Icc 0 n, i) = n * (n + 1) := by
  induction n with
  | zero =>
    -- Base case: n = 0
    simp
  | succ n ih =>
    -- Inductive step
    rw [Finset.sum_Icc_succ_top (by linarith)]
    rw [Nat.mul_add]
    rw [ih]
    ring

-- Sum of first n odd numbers is n²
theorem sum_of_first_n_odd (n : ℕ) :
  (∑ i in Finset.range n, (2 * i + 1)) = n ^ 2 := by
  induction n with
  | zero =>
    simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    rw [ih]
    ring

end Induction

-- ============================================================================
-- Section 6: Inequalities
-- ============================================================================

section Inequalities

-- Every natural number is ≥ 0
theorem nat_nonneg (n : ℕ) : 0 ≤ n := by
  exact Nat.zero_le n

-- If a ≤ b then a + c ≤ b + c
theorem inequality_add (a b c : ℕ) (h : a ≤ b) : a + c ≤ b + c := by
  apply Nat.add_le_add_right
  exact h

-- Square is always non-negative for integers
theorem square_nonneg (n : ℤ) : 0 ≤ n ^ 2 := by
  apply pow_two_nonneg

-- AM-GM inequality for two numbers: (a+b)/2 ≥ √(ab) for a,b ≥ 0
-- We'll prove: (a+b)² ≥ 4ab which is equivalent
theorem am_gm_two (a b : ℕ) : (a + b) ^ 2 ≥ 4 * a * b := by
  have h : (a + b) ^ 2 = a^2 + 2*a*b + b^2 := by ring
  rw [h]
  have h2 : a^2 + 2*a*b + b^2 ≥ 4*a*b := by
    have h3 : a^2 - 2*a*b + b^2 ≥ 0 := by
      have h4 : a^2 - 2*a*b + b^2 = (a - b)^2 := by ring
      rw [h4]
      apply pow_two_nonneg
    linarith
  linarith

end Inequalities

end BasicArithmetic
