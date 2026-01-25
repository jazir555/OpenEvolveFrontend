import RESE.Basic
import RESE.Constraint

/-!
# RESE.Infrastructure

Infrastructure lemmas for RESE formal verification.

This module provides foundational lemmas about lists, indices, and topological sorts
that are needed to complete advanced proofs in the RESE system.

## Main Lemmas

1. **List Index Properties**: Lemmas about list indices and element access
2. **NoDup Properties**: Lemmas about lists without duplicates

## Authors

- Agent O1: Lean 4 Formalization Specialist

-/

open RESE.Basic RESE.Constraint

namespace RESE.Infrastructure

/-!
## Section 1: List Index Lemmas

Fundamental lemmas about list element access and indices.
-/

/-- If an element is at some index in a list, then it's a member of the list -/
theorem mem_of_getElem? {α : Type} [BEq α] {l : List α} {i : Nat} {x : α}
    (h : l[i]? = some x) : x ∈ l := by
  -- Use the Mathlib lemma: x ∈ l if and only if there exists i such that l[i]? = some x
  -- We have such an i (given) and proof h, so we can apply the lemma
  apply List.mem_iff_getElem?.mpr
  exact ⟨i, h⟩

/-- In a list without duplicates (Nodup), if the same element appears at two indices,
    those indices must be equal -/
theorem index_of_unique {α : Type} [BEq α] {l : List α} (h_nodup : l.Nodup)
    {x : α} {i j : Nat}
    (h_i : l[i]? = some x) (h_j : l[j]? = some x) : i = j := by
  -- Use the Mathlib lemma Nodup.getElem_inj_iff via automation
  -- This lemma states: for Nodup l with valid indices, l[i] = l[j] ↔ i = j
  -- We have l[i]? = some x and l[j]? = some x, which gives l[i] = x = l[j]
  aesop (add simp [List.Nodup.getElem_inj_iff, List.getElem?_eq_some_iff])

/-!
## Section 2: Topological Sort Properties

This section documents the properties that topological sorts should satisfy.
To avoid circular dependencies with Templates.lean, these are documented here
as axioms rather than formal theorems.

**Topological Sort Nodup Property**

A fundamental theorem in graph theory states:
"The topological ordering of a DAG contains each vertex exactly once."

In RESE, this means: if order is a valid topological sort of well-formed constraints,
then order.Nodup (no duplicate elements).

Proof Sketch: Assume order has duplicates, so exists x, i, j with i < j and
order[i]? = order[j]? = some x. By well-formedness, x = c.id for some constraint c.
By the topological sort property, all dependencies of c appear before c.
This creates a contradiction when the same element appears multiple times.

Usage: This property is used as an AXIOM in Templates.lean (line 408),
justified by standard graph theory results.
-/

/-!
## Section 3: Combined Infrastructure

Lemmas that combine the above infrastructure for practical use.
-/

/-- In a Nodup list, if the same element appears at two indices, those indices are equal.

    This combines the Nodup property with index uniqueness.

    Usage pattern in Templates.lean:
    1. Prove order.Nodup from topological sort property
    2. Apply index_of_unique to get index equality
    3. Derive contradiction with index ordering
-/
theorem nodup_index_unique {α : Type} [BEq α] {l : List α}
    (h_nodup : l.Nodup) {x : α} {i j : Nat}
    (h_i : l[i]? = some x) (h_j : l[j]? = some x) : i = j :=
  -- Direct application of index_of_unique
  index_of_unique h_nodup h_i h_j

/-!
## Section 4: Documentation of Proof Strategies

This section documents how the infrastructure lemmas are used in practice.

**Usage Example: Proving a contradiction from duplicate indices**

Given:
- h_nodup : order.Nodup (proved from topological sort property)
- h_i : order[i]? = some x
- h_j : order[j]? = some x
- h_lt : i < j

To derive a contradiction:
  have h_same := nodup_index_unique h_nodup h_i h_j
  rw [h_same] at h_lt
  -- Now h_lt : i < i, which contradicts Nat.lt_irrefl
  apply Nat.lt_irrefl i at h_lt
  contradiction

This pattern is used in acyclicity_by_topological_sort in Templates.lean.
-/

end RESE.Infrastructure
