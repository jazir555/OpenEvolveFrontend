import Mathlib
import MathlibProject.Core.SSVTypes
import MathlibProject.Core.ArithmeticSafety

namespace MathlibProject

/-- Basic theorems and definitions for the SSV formal verification project.

  This file provides foundational results used throughout the project.
-/

/-- The SSV token total supply fits within Ethereum's 256-bit unsigned integer range -/
theorem ssv_supply_safe : SSV_MAX_SUPPLY < ETH_MAX := by
  exact ssv_supply_within_eth_bounds

/-- Addition is associative for natural numbers -/
example (a b c : Nat) : (a + b) + c = a + (b + c) := by
  ring

/-- Multiplication distributes over addition -/
example (a b c : Nat) : a * (b + c) = a * b + a * c := by
  ring

/-- Zero is the additive identity -/
example (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero]

/-- One is the multiplicative identity -/
example (a : Nat) : a * 1 = a := by
  rw [Nat.mul_one]

/-- Reflexivity of equality -/
example (a : Nat) : a = a := by
  rfl

/-- Symmetry of equality -/
example {a b : Nat} (h : a = b) : b = a := by
  symmetry
  exact h

/-- Transitivity of equality -/
example {a b c : Nat} (h1 : a = b) (h2 : b = c) : a = c := by
  transitivity b
  · exact h1
  · exact h2

/-- Basic inequality: if a < b and b < c then a < c -/
example {a b c : Nat} (h1 : a < b) (h2 : b < c) : a < c := by
  exact Nat.lt_trans h1 h2

/-- Multiplication by zero yields zero -/
example (a : Nat) : a * 0 = 0 := by
  rw [Nat.mul_zero]

/-- Zero is less than or equal to any natural number -/
example (a : Nat) : 0 ≤ a := by
  exact Nat.zero_le a

end MathlibProject
