# TestCases.lean - Quick Reference Guide

**Companion to**: `TESTCASES_PROOF_ANALYSIS.md`
**Purpose**: Quick lookup for proof strategies and tactics

---

## Test Cases at a Glance

| # | Test Case | Line | Difficulty | Status | Strategy |
|---|-----------|------|------------|--------|----------|
| 1 | Non-contradictory constraints | 82 | EASY | Ready to prove | Show `¬¬(True ∧ True)` |
| 2 | Simple acyclic graph | 93 | MODERATE | Ready to prove | Check no self-loops |
| 3 | Cyclic graph detection | 99 | **BLOCKED** | Definition issue | `hasCycle` can't detect 2-cycles |
| 4 | Equivalent constraint sets | 121, 123 | MODERATE | Ready to prove | Both have `formalization = True` |
| 5 | Polynomial complexity bound | 139 | EASY | Ready to prove | 3 ≤ 9 by computation |
| 6 | Linear complexity chain | 151 | EASY | Ready to prove | 2 ≤ 3 by computation |
| 7 | Topological sort validation | 181 | MODERATE | Ready to prove | Check indices |
| 8 | Integrated system | 210 | MODERATE | Ready to prove | Check no self-loops |

---

## Proof Strategy Quick Lookup

### Test Case 1: Non-Contradictory Constraints
```lean
unfold contradict
intro h
have : True ∧ True := by constructor <;> trivial
contradiction
```

**Key insight**: Both constraints have `formalization = True`.

### Test Cases 5 & 6: Complexity Bounds
```lean
unfold countDependencies
simp [List.foldl]
linarith  -- or decide
```

**Key insight**: Direct computation shows the inequality.

### Test Case 7: Topological Sort
```lean
unfold isTopologicallySorted
intro c hc dep hdep
-- Case analysis on which constraint c is
-- Find indices using List.get?
-- Show idx_dep < idx_c using arithmetic
```

**Key insight**: Manual index verification for each constraint.

---

## Common Proof Patterns

### Working with `List.any`
```lean
-- To show no element satisfies a predicate
intro h
unfold List.any at h
-- h : ∃ x ∈ l, P x
cases h with
| intro elem h_elem =>
  -- elem : (α × β)
  -- h_elem : elem ∈ l ∧ P elem
  -- Derive contradiction
```

### Working with `List.foldl`
```lean
-- To compute foldl explicitly
simp [List.foldl]
-- Or prove by induction on the list
induction l with
| nil => simp
| cons head tail ih =>
  simp [List.foldl, ih]
```

### Working with `List.get?`
```lean
-- To show an element exists at an index
have h_mem : x ∈ l := by -- proof
simp [List.get?_eq_some] at h_mem
cases h_mem with
| intro i h_idx =>
  -- i : Nat
  -- h_idx : l[i]? = some x
```

---

## Critical Definitions Reference

### `contradict`
```lean
def contradict (c1 c2 : Constraint) : Prop :=
  ¬(c1.formalization ∧ c2.formalization)
```
**Usage**: Prove `¬(P ∧ Q)` or show one constraint implies `¬(other)`.

### `DependencyGraph.hasCycle`
```lean
def DependencyGraph.hasCycle (g : DependencyGraph) : Bool :=
  if g.nodes.length = 0 then false
  else g.edges.any (λ e => e.1 == e.2)
```
**⚠️ WARNING**: Only checks for self-loops! Does NOT detect longer cycles.

### `equivalentSets`
```lean
def equivalentSets (S1 S2 : List Constraint) : Prop :=
  ∀ P, (∀ c ∈ S1, satisfiedBy c P) ↔ (∀ c ∈ S2, satisfiedBy c P)
```
**Usage**: Prove both directions of implication for arbitrary P.

### `countDependencies`
```lean
def countDependencies (constraints : List Constraint) : Nat :=
  constraints.foldl (λ acc c => acc + c.dependencies.length) 0
```
**Usage**: Sum of all dependency list lengths.

### `isTopologicallySorted`
```lean
def isTopologicallySorted (constraints : List Constraint) (order : List ConstraintId) : Prop :=
  ∀ c ∈ constraints,
    ∀ dep ∈ c.dependencies,
      ∃ (idx_c idx_dep : Nat),
        order[idx_c]? = some c.id ∧
        order[idx_dep]? = some dep ∧
        idx_dep < idx_c
```
**Usage**: For each constraint and each dependency, find indices showing dependency comes first.

### `satisfiedBy`
```lean
def satisfiedBy (c : Constraint) (P : Prop) : Prop :=
  P → c.formalization
```
**Usage**: Show proposition P implies the constraint's formalization.

---

## Recommended Lean 4 Imports

```lean
import RESE.Basic
import RESE.Constraint
import RESE.Templates

-- From Mathlib4 (if needed)
import Std.Data.List
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
```

---

## Common Tactics by Category

### Logic
- `intro` / `intros` - Introduce hypotheses
- `apply` - Apply a theorem or hypothesis
- `exact` - Give exact proof term
- `contradiction` - Find contradiction in hypotheses
- `cases` - Destruct structures or hypotheses
- `constructor` - Construct a conjunction/exists/etc.

### Computation
- `simp` - Simplify using definitional equalities
- `rfl` - Prove equality by reflexivity
- `decide` - Prove by computation (for decidable propositions)
- `linarith` - Solve linear arithmetic goals

### Lists
- `unfold` - Unfold a definition
- `induction` - Prove by induction on list structure

---

## Debugging Tips

### When a proof fails
1. **Check types**: Use `infer_type` to ensure hypotheses match
2. **Unfold definitions**: Use `unfold` or `simp` to see what you're working with
3. **Check implicit arguments**: Add `@` before functions to see all arguments
4. **Use `have`**: Break complex goals into smaller lemmas

### Common pitfalls
1. **String equality**: Use `==` for boolean equality, `=` for propositional
2. **List indexing**: `l[i]?` returns an `Option`, need to handle `none` case
3. **Bool vs Prop**: `hasCycle` returns `Bool`, not `Prop`
4. `dependencies` is a `List ConstraintId`, not a constraint itself

---

## Recommended Proof Order

1. **Start with**: Test Case 1 (trivial, builds confidence)
2. **Then**: Test Cases 5, 6 (easy, similar pattern)
3. **Then**: Test Case 7 (moderate, tests list operations)
4. **Then**: Test Cases 2, 8 (moderate, same pattern)
5. **Finally**: Test Cases 3, 4 (require more thought)

---

## When You're Stuck

1. **Re-read the definition**: Make sure you understand what you're proving
2. **Check the type**: What is the exact goal type?
3. **Simplify**: Use `unfold` and `simp` to reduce the goal
4. **Extract a lemma**: If stuck on a sub-proof, make it a separate theorem
5. **Ask for help**: The templates in `Templates.lean` may help

---

## Quick Tactics for Each Test Case

### TC1: `unfold`, `have`, `contradiction`
### TC2: `unfold`, `simp`, `List.any`, cases on edges
### TC3: **BLOCKED** - needs definition fix
### TC4: `unfold`, constructor, cases on which constraint
### TC5: `unfold`, `simp`, `linarith`
### TC6: `unfold`, `simp`, `linarith`
### TC7: `unfold`, intro, cases on constraints, index arithmetic
### TC8: same as TC2

---

**For detailed analysis, see**: `TESTCASES_PROOF_ANALYSIS.md`
