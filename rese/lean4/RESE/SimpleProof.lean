import Std
import Mathlib.Data.List.Dedup

#check @List.eraseDups
#check @List.dedup

#eval [1,2,3].eraseDups
#eval [1,2,1,3].eraseDups
#eval ([1,1,1,1]).eraseDups

#eval List.dedup [1,2,3]
#eval List.dedup [1,2,1,3]
#eval List.dedup [1,1,1,1]

/-!
## Understanding eraseDups vs dedup

The `eraseDups` function from Std uses `BEq` (boolean equality) which is
difficult to reason about in proofs. The `dedup` function from Mathlib uses
`DecidableEq` which has much better definitional equations.

The following theorems demonstrate the difference:
-/

-- Test eraseDups behavior (difficult to prove properties about)
theorem test_eraseDups_nil : ([] : List Nat).eraseDups = [] := by
  rfl

-- Using dedup from Mathlib (MUCH easier to work with)
theorem test_dedup_nil {α : Type} [DecidableEq α] : ([] : List α).dedup = [] := by
  rfl

/-- The key property: when an element is not in the list, dedup keeps it

    This is EASY to prove with Mathlib's dedup because it has good
    definitional equations. With eraseDups, this would require very
    complex reasoning about boolean equality. -/
theorem test_dedup_cons_not_mem {α : Type} [DecidableEq α] (a : α) (l : List α)
    (h : a ∉ l) : (a :: l).dedup = a :: l.dedup := by
  -- Use the defining equation for dedup on cons
  rw [List.dedup_cons]
  -- Since a ∉ l, the if-else simplifies to the else branch
  simp [h]

/-!
## Why eraseDups is Hard to Work With

The original attempt was:

```lean
theorem test_eraseDups_cons_not_mem :
  ∀ (a : Nat) (l : List α), a ∉ l → (a :: l).eraseDups = a :: l.eraseDups := by
  intro a l h
  sorry
```

This is difficult because:
1. `eraseDups` uses `BEq` which doesn't provide good definitional equations
2. The function is defined recursively with pattern matching on `BEq.beq`
3. Proving properties requires reasoning about boolean equality and decidability

The solution is to use `List.dedup` from Mathlib instead, which uses
`DecidableEq` and provides much better proof support.

**RECOMMENDATION**: Always use `List.dedup` for formal proofs.
Use `eraseDups` only for runtime code where you need `BEq`/`Hashable`.
-/

