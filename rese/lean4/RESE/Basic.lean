import Mathlib.Data.List.Dedup

/-!
# RESE.Basic

Basic definitions and utilities for RESE formalizations.

This module provides:
- Type aliases for common RESE types
- Utility functions for propositions and logic
- Basic lemmas about constraints and dependencies
-/

namespace RESE.Basic

/-!
## Section 1: Type Aliases and Basic Definitions
-/

/-- A constraint identifier is just a string -/
abbrev ConstraintId := String

/-- A proposition represents a logical statement -/
abbrev Proposition := Prop

/-- A constraint set is a list of constraint IDs -/
abbrev ConstraintSet := List ConstraintId

/-!
## Section 2: Dependency Graph Theory
-/

/-- A dependency is a pair of constraint IDs (A depends on B) -/
structure Dependency where
  dependent : ConstraintId
  depends_on : ConstraintId
deriving Repr, BEq, Hashable

/-- Create a dependency -/
def mkDependency (fromId : ConstraintId) (toId : ConstraintId) : Dependency :=
  ⟨toId, fromId⟩

/-!
## Section 3: Basic Lemmas
-/

/-- Empty list has no elements -/
theorem not_mem_nil {α : Type} (x : α) : x ∉ ([] : List α) :=
  by
    intro h
    cases h

/-- If x is in cons, it's either the head or in the tail -/
theorem mem_cons_or {α : Type} {x y : α} {l : List α} :
    x ∈ y :: l ↔ x = y ∨ x ∈ l :=
  List.mem_cons

/-- Append preserves membership -/
theorem mem_append {α : Type} {x : α} {l1 l2 : List α} :
    x ∈ l1 ++ l2 ↔ x ∈ l1 ∨ x ∈ l2 :=
  List.mem_append

/-!
## Section 4: Logical Utilities
-/

/-- Convert a boolean to a proposition -/
def boolToProp (b : Bool) : Prop :=
  if b then True else False

/-- Negation of a proposition -/
def Not (P : Prop) : Prop :=
  P → False

/-- Implication is transitive -/
theorem imp_transitive {P Q R : Prop} : (P → Q) → (Q → R) → (P → R) :=
  by
    intro hpq hqr p
    apply hqr
    apply hpq
    assumption

/-!
## Section 5: List Utilities for Constraint Management
-/

/-- Remove duplicates from a list using the standard library's dedup function.
    This uses DecidableEq instead of BEq/Hashable for better provability. -/
abbrev dedup {α : Type} [DecidableEq α] (l : List α) : List α :=
  List.dedup l

/-- Length of deduplicated list is ≤ original -/
theorem length_dedup_le {α : Type} [DecidableEq α] (l : List α) :
    (dedup l).length ≤ l.length :=
  by
    -- Direct induction proof using the definition of dedup
    -- Based on the fact that dedup removes elements (never adds them)
    induction l with
    | nil =>
      -- Base case: dedup [] = [], and 0 ≤ 0
      simp [dedup]
    | cons x xs ih =>
      -- Inductive case: dedup (x :: xs)
      -- Unfold dedup to use List.dedup
      unfold dedup
      -- Use the defining equation for dedup on cons
      rw [List.dedup_cons]
      -- Split on whether x is in xs
      if h : x ∈ xs then
        -- Case 1: x ∈ xs, so dedup (x::xs) = dedup xs
        -- Then |dedup (x::xs)| = |dedup xs| ≤ |xs| ≤ |x::xs|
        simp [h]
        -- We have |dedup xs| ≤ |xs| by IH
        -- And |xs| ≤ |x::xs| = |xs| + 1
        -- So by transitivity: |dedup (x::xs)| = |dedup xs| ≤ |xs| + 1 = |x::xs|
        apply Nat.le_trans ih
        apply Nat.le_add_right
      else
        -- Case 2: x ∉ xs, so dedup (x::xs) = x :: dedup xs
        -- Then |dedup (x::xs)| = 1 + |dedup xs| ≤ 1 + |xs| = |x::xs|
        simp [h]  -- simplifies the if-else using h
        -- Now the goal is: xs.dedup.length ≤ xs.length
        -- Which is exactly our induction hypothesis!
        exact ih

end RESE.Basic
