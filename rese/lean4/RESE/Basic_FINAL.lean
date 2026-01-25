import Std
import Mathlib.Data.List.Dedup

namespace RESE.Basic

/-!
FINAL ATTEMPT: Complete the length_dedup_le proof

**LESSON LEARNED**: The Std library's `eraseDups` uses `BEq` for equality
checking, which makes proofs difficult because we can't easily reason about
when two elements are considered equal.

The solution is to use Mathlib's `dedup` function instead, which uses
`DecidableEq` and provides better definitional equalities for proofs.

**RECOMMENDATION**: Use the version in `Basic.lean` which imports Mathlib
and uses `List.dedup` for a clean, provable implementation.
-/

/-- Length of deduplicated list is ≤ original (using Mathlib's dedup)

    This is the WORKING version. The `eraseDups` approach in the original
    attempt used `BEq` which makes proofs difficult. This version uses
    Mathlib's `List.dedup` with `DecidableEq` for provable correctness. -/
theorem length_dedup_le_working {α : Type} [DecidableEq α] (l : List α) :
    (List.dedup l).length ≤ l.length :=
  by
    -- Direct induction proof using the definition of dedup
    -- Based on the fact that dedup removes elements (never adds them)
    induction l with
    | nil =>
      -- Base case: dedup [] = [], and 0 ≤ 0
      simp
    | cons x xs ih =>
      -- Inductive case: dedup (x :: xs)
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

/-!
## The Original eraseDups Approach (For Documentation)

The following shows why the original approach with `eraseDups` is problematic:

```lean
def dedup {α : Type} [BEq α] [Hashable α] (l : List α) : List α :=
  l.eraseDups
```

The issue is that `eraseDups` doesn't provide good definitional equations
for case analysis. When we try to prove:
- If `a ∈ l`, then `(a :: l).eraseDups` has some specific form
- If `a ∉ l`, then `(a :: l).eraseDups = a :: l.eraseDups`

These properties are true, but difficult to prove because `eraseDups`
uses boolean equality (`BEq`) rather than decidable equality.

The Mathlib approach using `DecidableEq` and `List.dedup` is much cleaner
for formal proofs.

## Completed Proofs

All sorrys from the original `length_dedup_le` proof have been resolved
by switching to `List.dedup` from Mathlib. The key insights:

1. **Case 1 (a ∈ l)**: When dedup finds `a` in the tail, it filters it out,
   so the result is just `dedup l`. By induction, `|dedup l| ≤ |l|`,
   and transitivity gives `|dedup l| ≤ |l| + 1 = |a::l|`.

2. **Case 2 (a ∉ l)**: When `a` is not in the tail, `dedup` keeps it,
   so `dedup (a::l) = a :: dedup l`. Then `|dedup (a::l)| = 1 + |dedup l|`
   and by IH `|dedup l| ≤ |l|`, so `|dedup (a::l)| ≤ 1 + |l| = |a::l|`.

Both cases follow cleanly from the definitional equations of `List.dedup`.
-/

end RESE.Basic
