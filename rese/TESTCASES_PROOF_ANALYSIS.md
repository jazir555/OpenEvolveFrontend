# TestCases.lean Proof Requirements Analysis

**Date**: 2026-01-01
**File**: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\rese\lean4\RESE\TestCases.lean`
**Analysis Type**: Proof Strategy and Requirements Assessment
**Total Test Cases**: 8 (with 10 sorry marks across different lines)

---

## Executive Summary

This document provides a detailed analysis of the 8 test case theorems in TestCases.lean that require proof completion. The test cases range from trivial logical manipulations to substantial graph theory and complexity proofs requiring significant auxiliary lemmas.

**Key Findings**:
- **Trivial cases (2)**: Can be completed with basic tactics
- **Moderate cases (3)**: Require some lemmas and careful reasoning
- **Substantial cases (3)**: Require significant auxiliary work, definitions, or potentially problem reformulation

---

## Test Case-by-Case Analysis

### **Test Case 1: Non-Contradictory Constraints**
**Location**: Line 82 (sorry mark)
**Theorem Name**: `non_contradictory_constraints`

#### Mathematical/Logical Meaning
Prove that two specific constraints do NOT contradict each other:
- `c1`: Temperature must be less than 1000°C (formalization = True)
- `c2`: Pressure should be below 10 bar (formalization = True)

Both constraints have `formalization = True`, so both are always satisfiable.

#### Proof Strategy
**TRIVIAL** - This is actually a proof about True ∧ True.

The key insight: both constraints have `formalization := True`, meaning:
- `c1.formalization = True`
- `c2.formalization = True`
- Therefore `c1.formalization ∧ c2.formalization = True ∧ True = True`
- So `¬(c1.formalization ∧ c2.formalization)` = `¬True` = `False`
- But we need to prove `¬contradict c1 c2`, i.e., `¬(¬(c1.formalization ∧ c2.formalization))`

#### Required Lean 4 Elements
- **Tactics**: `unfold`, `intro`, `contradiction` or `trivial`
- **Theorems**: Basic logic rules about True and ¬
- **Definitions**: `contradict` definition from Constraint.lean

#### Proof Sketch
```lean
theorem non_contradictory_constraints :
    let c1 : Constraint := ⟨"temp_limit", ConstraintType.hard,
      "Temperature must be less than 1000°C", True, [], "test"⟩
    let c2 : Constraint := ⟨"pressure_limit", ConstraintType.soft,
      "Pressure should be below 10 bar", True, [], "test"⟩
    ¬contradict c1 c2 := by
  intro c1 c2
  unfold contradict
  -- Need to show: ¬¬(True ∧ True)
  -- This is equivalent to True ∧ True, which is True
  intro h
  -- h : ¬(True ∧ True)
  -- But True ∧ True is True, so h : ¬True
  have : True ∧ True := by constructor <;> trivial
  contradiction
```

#### Difficulty
**EASY** - Can be completed in 5-10 lines with basic tactics.

#### Clarifications Needed
None. The statement is clear.

---

### **Test Case 2: Simple Acyclic Graph**
**Location**: Lines 89-93 (sorry at line 93)
**Theorem Name**: Unnamed `example`

#### Mathematical/Logical Meaning
Prove that the following graph has no cycles:
- Nodes: ["A", "B", "C"]
- Edges: [("A", "B"), ("B", "C")]

This is a simple linear chain A → B → C with no cycles.

#### Proof Strategy
**MODERATE** - Depends on the definition of `hasCycle`.

The current `DependencyGraph.hasCycle` definition (Constraint.lean:96-100):
```lean
def DependencyGraph.hasCycle (g : DependencyGraph) : Bool :=
  if g.nodes.length = 0 then false
  else
    g.edges.any (λ e => e.1 == e.2)
```

This is a **SIMPLIFIED** cycle detection that only checks for self-loops!

#### Required Lean 4 Elements
- **Tactics**: `unfold`, `intro`, `cases`, `List.any`, `List.mem`
- **Theorems**: List membership theorems from Mathlib4
- **Definitions**: `DependencyGraph.hasCycle`, `List.any`

#### Proof Sketch
```lean
example : ¬({ nodes := ["A", "B", "C"], edges := [("A", "B"), ("B", "C")] : DependencyGraph}).hasCycle := by
  intro hcycle
  unfold DependencyGraph.hasCycle at hcycle
  -- hcycle : (if nodes.length = 0 then false else edges.any (λ e => e.1 = e.2)) = true
  -- Since nodes.length = 3 ≠ 0, this simplifies to:
  -- edges.any (λ e => e.1 = e.2) = true
  -- Need to show no edge has e.1 = e.2
  simp [List.any] at hcycle
  -- Check each edge:
  -- ("A", "B"): A ≠ B
  -- ("B", "C"): B ≠ C
  -- Contradiction
```

#### Difficulty
**MODERATE** - Requires working with `List.any` and string inequality.

#### Clarifications Needed
1. The `hasCycle` definition is simplified and won't detect longer cycles (e.g., A→B→A)
2. For proper cycle detection, a full DFS or topological sort algorithm is needed
3. Consider whether this simplified definition is sufficient for the project's needs

#### Dependencies
Can use helper lemmas from `List.any` and `BEq` for strings.

---

### **Test Case 3: Cyclic Graph Detection**
**Location**: Lines 96-99 (sorry at line 99)
**Theorem Name**: Unnamed `example`

#### Mathematical/Logical Meaning
Prove that the following graph HAS a cycle:
- Nodes: ["A", "B"]
- Edges: [("A", "B"), ("B", "A")]

This graph has a cycle A → B → A.

#### Proof Strategy
**PROBLEMATIC** - The current `hasCycle` definition CANNOT detect this cycle!

The simplified definition only checks for self-loops (edges where `e.1 == e.2`), not cycles of length > 1.

#### Required Lean 4 Elements
- **Major Issue**: The definition is inadequate for this test case
- **Options**:
  1. **Refine `hasCycle`**: Implement proper cycle detection (DFS, path-based, etc.)
  2. **Change test case**: Use a graph with a self-loop instead
  3. **Add helper definition**: `hasCycleOfLength` that checks for cycles up to length N

#### Proof Sketch (if definition is fixed)
```lean
-- Option 1: With proper cycle detection
example : ({ nodes := ["A", "B"], edges := [("A", "B"), ("B", "A")] : DependencyGraph}).hasCycle := by
  unfold DependencyGraph.hasCycle
  -- Proof would show path A→B→A exists
  sorry

-- Option 2: Change test to use self-loop
example : ({ nodes := ["A"], edges := [("A", "A")] : DependencyGraph}).hasCycle := by
  unfold DependencyGraph.hasCycle
  simp
  -- Show ("A", "A") is in edges
```

#### Difficulty
**HARD (definition issue)** - The proof itself is trivial if the definition supports it, but the definition needs rework.

#### Clarifications Needed
**CRITICAL**: The current `hasCycle` definition is insufficient for this test case. Must decide:
1. Should `hasCycle` be redefined with proper cycle detection?
2. Should the test case be changed to use a self-loop?
3. Should a separate `hasCycleProper` definition be added?

#### Dependencies
Depends on fixing `DependencyGraph.hasCycle` definition in Constraint.lean.

---

### **Test Case 4: Equivalent Constraint Sets**
**Location**: Lines 106-123 (sorry at lines 121 and 123)
**Theorem Name**: `equivalent_sets_example`

#### Mathematical/Logical Meaning
Prove two constraint sets are equivalent:
- `S1`: [c1 (hard), c2 (soft, depends on c1)]
- `S2`: [c2 (soft, no deps), c1 (hard)]

The sets differ only in:
- Order of constraints (shouldn't matter)
- Dependency of c2 on c1 in S1 but not in S2

**Question**: Why should these be equivalent? The dependencies differ!

#### Proof Strategy
**NEEDS CLARIFICATION** - The test case may be flawed or demonstrating a subtle point.

`equivalentSets S1 S2` means:
```
∀ P, (∀ c ∈ S1, satisfiedBy c P) ↔ (∀ c ∈ S2, satisfiedBy c P)
```

For these to be equivalent, the dependency information in c2 must not affect satisfiability. Since both have `formalization = True`, this might hold.

#### Required Lean 4 Elements
- **Tactics**: `unfold`, `intro`, `constructor`, `simp`, List membership reasoning
- **Theorems**: `List.mem`, `satisfiedBy` properties
- **Definitions**: `equivalentSets`, `satisfiedBy`

#### Proof Sketch
```lean
theorem equivalent_sets_example :
    let S1 := [
      ⟨"c1", ConstraintType.hard, "Constraint 1", True, [], "test"⟩,
      ⟨"c2", ConstraintType.soft, "Constraint 2", True, ["c1"], "test"⟩
    ]
    let S2 := [
      ⟨"c2", ConstraintType.soft, "Constraint 2", True, [], "test"⟩,
      ⟨"c1", ConstraintType.hard, "Constraint 1", True, [], "test"⟩
    ]
    equivalentSets S1 S2 := by
  intro S1 S2
  unfold equivalentSets
  intro P
  constructor
  . -- Forward: S1 satisfied → S2 satisfied
    intro h1 c2 hc2
    -- c2 is in S2, need to show satisfiedBy c2 P
    -- c2.formalization = True, so P → True is always true
    unfold satisfiedBy
    intro _
    trivial
    -- Similar for c1
    sorry
  . -- Backward: S2 satisfied → S1 satisfied
    sorry
```

#### Difficulty
**MODERATE** - The key insight is that both constraints have `formalization = True`, making them trivially satisfiable.

#### Clarifications Needed
1. **Is this test case meaningful?** Since both have `formalization = True`, they're trivially equivalent.
2. **Should dependencies matter?** The definition of `equivalentSets` doesn't consider dependencies, only satisfiability.
3. **Better test case**: Use constraints with non-trivial formalizations to show genuine equivalence.

#### Dependencies
Can be proved independently. Uses lemmas about `satisfiedBy` and True.

---

### **Test Case 5: Polynomial Complexity Bound**
**Location**: Lines 130-139 (sorry at line 139)
**Theorem Name**: `complexity_polynomial_bound`

#### Mathematical/Logical Meaning
Prove that for 3 constraints with specific dependencies:
```lean
c1: deps = []
c2: deps = ["c1"]
c3: deps = ["c1", "c2"]
```

The total dependency count ≤ n² where n = 3.

`countDependencies` = 0 + 1 + 2 = 3
`constraints.length ^ 2` = 3² = 9
3 ≤ 9 ✓

#### Proof Strategy
**EASY TO MODERATE** - Direct computation.

This is a specific instance of a general theorem: for any list of n constraints, the maximum number of dependencies is O(n²).

#### Required Lean 4 Elements
- **Tactics**: `unfold`, `simp`, `linarith` or `decide`
- **Theorems**: Basic arithmetic, properties of `foldl`
- **Definitions**: `countDependencies`

#### Proof Sketch
```lean
theorem complexity_polynomial_bound :
    let constraints : List Constraint := [
      ⟨"c1", ConstraintType.hard, "Constraint 1", True, [], "test"⟩,
      ⟨"c2", ConstraintType.hard, "Constraint 2", True, ["c1"], "test"⟩,
      ⟨"c3", ConstraintType.soft, "Constraint 3", True, ["c1", "c2"], "test"⟩
    ]
    countDependencies constraints ≤ constraints.length ^ 2 := by
  intro constraints
  unfold countDependencies
  -- countDependencies = foldl (λ acc c => acc + c.dependencies.length) 0
  -- = 0 + 0 + 1 + 2 = 3
  simp [List.foldl]
  -- constraints.length ^ 2 = 3^2 = 9
  -- Show 3 ≤ 9
  linarith
```

#### Difficulty
**EASY** - Computational, can potentially use `decide` or `simp`.

#### Clarifications Needed
None. The statement is clear and true.

#### Dependencies
Independent. Could also prove the general lemma: `∀ (L : List Constraint), countDependencies L ≤ L.length ^ 2`.

---

### **Test Case 6: Linear Complexity for Chain Dependencies**
**Location**: Lines 142-151 (sorry at line 151)
**Theorem Name**: `complexity_linear_chain`

#### Mathematical/Logical Meaning
Prove that for a chain of dependencies:
```lean
c1: deps = []
c2: deps = ["c1"]
c3: deps = ["c2"]
```

The total dependency count ≤ n (linear).

`countDependencies` = 0 + 1 + 1 = 2
`constraints.length` = 3
2 ≤ 3 ✓

#### Proof Strategy
**EASY** - Similar to Test Case 5, but for linear chains.

This demonstrates that "nice" dependency structures (trees, chains) have linear complexity.

#### Required Lean 4 Elements
- **Tactics**: `unfold`, `simp`, `linarith` or `decide`
- **Theorems**: Basic arithmetic
- **Definitions**: `countDependencies`

#### Proof Sketch
```lean
theorem complexity_linear_chain :
    let constraints : List Constraint := [
      ⟨"c1", ConstraintType.hard, "Constraint 1", True, [], "test"⟩,
      ⟨"c2", ConstraintType.hard, "Constraint 2", True, ["c1"], "test"⟩,
      ⟨"c3", ConstraintType.hard, "Constraint 3", True, ["c2"], "test"⟩
    ]
    countDependencies constraints ≤ constraints.length := by
  intro constraints
  unfold countDependencies
  -- countDependencies = 0 + 0 + 1 + 1 = 2
  simp [List.foldl]
  -- constraints.length = 3
  -- Show 2 ≤ 3
  linarith
```

#### Difficulty
**EASY** - Direct computation.

#### Clarifications Needed
None. Clear and true.

#### Dependencies
Related to Test Case 5. Both illustrate specific instances of complexity bounds.

---

### **Test Case 7: Topological Sort Validation**
**Location**: Lines 171-181 (sorry at line 181)
**Theorem Name**: `topological_order_valid`

#### Mathematical/Logical Meaning
Prove that the order `["c1", "c2", "c3"]` is a valid topological sort for:
```lean
c1: deps = []
c2: deps = ["c1"]
c3: deps = ["c1", "c2"]
```

A valid topological sort means all dependencies appear before their dependents.

#### Proof Strategy
**MODERATE** - Requires checking all dependencies.

`isTopologicallySorted` definition (Constraint.lean:206-212):
```lean
def isTopologicallySorted (constraints : List Constraint) (order : List ConstraintId) : Prop :=
  ∀ c ∈ constraints,
    ∀ dep ∈ c.dependencies,
      ∃ (idx_c idx_dep : Nat),
        order[idx_c]? = some c.id ∧
        order[idx_dep]? = some dep ∧
        idx_dep < idx_c
```

Need to verify for each constraint:
- c1 (no deps): vacuously true
- c2 (dep: "c1"): "c1" is at index 0, "c2" at index 1, 0 < 1 ✓
- c3 (deps: "c1", "c2"): "c1" at 0, "c2" at 1, "c3" at 2, both < 2 ✓

#### Required Lean 4 Elements
- **Tactics**: `unfold`, `intro`, `simp`, List indexing, `List.get?`
- **Theorems**: `List.mem`, `List.get?`, arithmetic
- **Definitions**: `isTopologicallySorted`

#### Proof Sketch
```lean
theorem topological_order_valid :
    let constraints : List Constraint := [
      ⟨"c1", ConstraintType.hard, "Constraint 1", True, [], "test"⟩,
      ⟨"c2", ConstraintType.hard, "Constraint 2", True, ["c1"], "test"⟩,
      ⟨"c3", ConstraintType.hard, "Constraint 3", True, ["c1", "c2"], "test"⟩
    ]
    let order := ["c1", "c2", "c3"]
    isTopologicallySorted constraints order := by
  intro constraints order
  unfold isTopologicallySorted
  intro c hc dep hdep
  -- Need to find indices and show dependency comes before
  cases hc
  . -- c = c1
    -- c1.dependencies = [], so hdep is impossible
    contradiction
  . -- c = c2
    -- dep must be "c1"
    -- Show: "c1" at index 0, "c2" at index 1, 0 < 1
    sorry
  . -- c = c3
    -- dep is "c1" or "c2"
    -- Show both appear before index 2
    sorry
```

#### Difficulty
**MODERATE** - Requires case analysis and index calculations.

#### Clarifications Needed
None. The definition is clear.

#### Dependencies
Independent. Could use helper lemmas for index calculation.

---

### **Test Case 8: Integrated Constraint System Verification**
**Location**: Lines 188-210 (sorry at line 210)
**Theorem Name**: `integrated_constraint_system`

#### Mathematical/Logical Meaning
Verify that a multi-constraint system with 4 constraints and 2 edges is acyclic.

Graph structure:
- Nodes: ["temp_max", "temp_min", "pressure_max", "flow_rate"]
- Edges: [("temp_max", "temp_min"), ("pressure_max", "flow_rate")]

This creates two disconnected chains:
1. temp_max → temp_min
2. pressure_max → flow_rate

#### Proof Strategy
**MODERATE (with simplified hasCycle)** or **HARD (with proper hasCycle)**

With the current simplified definition, this is easy (just check no self-loops).
With proper cycle detection, need to show no path from any node back to itself.

#### Required Lean 4 Elements
- **Tactics**: Same as Test Case 2
- **Theorems**: Same as Test Case 2
- **Definitions**: `DependencyGraph.hasCycle`

#### Proof Sketch (with simplified definition)
```lean
theorem integrated_constraint_system :
    let constraints : List Constraint := [...]
    let graph : DependencyGraph := {
      nodes := ["temp_max", "temp_min", "pressure_max", "flow_rate"],
      edges := [
        ("temp_max", "temp_min"),
        ("pressure_max", "flow_rate")
      ]
    }
    ¬graph.hasCycle := by
  intro constraints graph
  unfold DependencyGraph.hasCycle
  -- No self-loops in edges
  -- Show: [("temp_max", "temp_min"), ("pressure_max", "flow_rate")] contains no (x, x)
  intro h
  -- h : edges.any (λ e => e.1 = e.2) = true
  -- Check each edge: none are self-loops
  sorry
```

#### Difficulty
**MODERATE** - Same issues as Test Case 2.

#### Clarifications Needed
Same as Test Cases 2 and 3: the simplified `hasCycle` definition may not be sufficient.

#### Dependencies
Directly related to Test Cases 2 and 3. Solving the `hasCycle` definition issue here resolves it for all three cases.

---

## Summary of Required Lean 4 Elements

### Tactics Needed
1. **Basic**: `unfold`, `intro`, `constructor`, `exact`, `apply`, `trivial`
2. **Logic**: `contradiction`, `cases`, `simp`
3. **Lists**: `List.mem`, `List.any`, `List.foldl`, `List.get?`
4. **Arithmetic**: `linarith`, `decide`, `Nat` arithmetic
5. **String**: `BEq` for String equality/inequality

### Theorems/Definitions from Mathlib4
1. **List**: `List.mem_cons`, `List.mem_append`, `List.any`, `List.foldl`, `List.get?_eq_some`
2. **Logic**: Basic propositional logic, `not_not`, `and_self`
3. **Arithmetic**: Nat comparison, inequality lemmas

### Definitions from RESE
1. `contradict` (Constraint.lean:143)
2. `DependencyGraph.hasCycle` (Constraint.lean:96) - **NEEDS REFINEMENT**
3. `equivalentSets` (Constraint.lean:177)
4. `countDependencies` (Constraint.lean:191)
5. `isTopologicallySorted` (Constraint.lean:206)
6. `satisfiedBy` (Constraint.lean:167)

---

## Difficulty Classification

### **Trivial (Can complete in <10 lines)**
1. **Test Case 1**: Non-contradictory constraints (True ∧ True reasoning)

### **Easy (Direct computation, <20 lines)**
2. **Test Case 5**: Polynomial complexity bound
3. **Test Case 6**: Linear complexity chain

### **Moderate (Requires case analysis, 20-40 lines)**
4. **Test Case 2**: Simple acyclic graph
5. **Test Case 4**: Equivalent constraint sets
6. **Test Case 7**: Topological sort validation
7. **Test Case 8**: Integrated system verification

### **Hard (Requires definition changes or substantial work)**
8. **Test Case 3**: Cyclic graph detection - **BLOCKED by inadequate definition**

---

## Critical Issues Requiring Resolution

### **Issue 1: hasCycle Definition is Inadequate**
**Location**: `Constraint.lean:96-100`

**Problem**: The current definition only checks for self-loops:
```lean
def DependencyGraph.hasCycle (g : DependencyGraph) : Bool :=
  if g.nodes.length = 0 then false
  else
    g.edges.any (λ e => e.1 == e.2)
```

This cannot detect cycles like A → B → A.

**Options**:
1. **Keep simplified**: Accept that hasCycle only detects self-loops
   - **Impact**: Test Case 3 fails, must be changed
   - **Workaround**: Add a comment explaining the limitation

2. **Implement proper cycle detection**:
   ```lean
   def hasCycle' (g : DependencyGraph) : Bool :=
     -- Implement DFS or Kahn's algorithm
     -- Check if back edges exist during DFS
     -- Or check if topological sort exists
   ```
   - **Impact**: More accurate, but requires significant work
   - **Dependencies**: Need path existence predicates, reachability

3. **Use path-based definition**:
   ```lean
   def hasCycle (g : DependencyGraph) : Prop :=
     ∃ (n : ConstraintId), transitiveDepends g n n
   ```
   - **Impact**: Already have `transitiveDepends` definition
   - **Note**: This changes `Bool` to `Prop`

**Recommendation**:
- **Short term**: Keep simplified definition, modify Test Case 3 to use self-loop
- **Long term**: Implement proper cycle detection using `transitiveDepends`

### **Issue 2: Test Case 4 May Be Trivial**
**Location**: `TestCases.lean:106-123`

**Problem**: Both constraints have `formalization = True`, making equivalence trivial.

**Options**:
1. **Keep as is**: Demonstrates the definition works
2. **Make more interesting**: Use constraints with non-trivial formalizations
3. **Add comment**: Explain that this tests the definition, not interesting properties

**Recommendation**: Keep but add explanatory comment.

---

## Dependencies Between Test Cases

### **Direct Dependencies**
None of the test cases directly depend on each other. They can be proved in any order.

### **Shared Infrastructure**
1. **Test Cases 2, 3, 8**: All depend on `hasCycle` definition
   - **Resolution approach**: Fix `hasCycle` once, all three benefit
   - **Priority**: HIGH

2. **Test Cases 5, 6**: Both use `countDependencies`
   - Could extract common lemmas about list fold operations

3. **Test Case 4 & 7**: Both use constraint satisfaction concepts
   - Could share helper lemmas about `satisfiedBy`

### **Recommended Order of Proofs**
1. **Test Case 1** (trivial, builds confidence)
2. **Test Cases 5, 6** (easy, build familiarity)
3. **Test Case 7** (moderate, tests list operations)
4. **Test Cases 2, 8** (moderate, same as 3)
5. **Test Case 3** (hard, may need definition change)
6. **Test Case 4** (moderate, but may need clarification)

---

## Recommended Auxiliary Lemmas

### For List Operations
```lean
-- List.any helpers
theorem List.any_not_self {α : Type} [BEq α] (l : List (α × α)) :
    (¬l.any (λ e => e.1 == e.2)) ↔ ∀ e ∈ l, e.1 ≠ e.2 := by
  sorry

-- Index calculation helpers
theorem List.index_of_mem {α : Type} [BEq α] (l : List α) (x : α) (h : x ∈ l) :
    ∃ i, l[i]? = some x := by
  sorry
```

### For Counting Dependencies
```lean
-- General complexity bound
theorem countDependencies_le_length_sq (L : List Constraint) :
    countDependencies L ≤ L.length ^ 2 := by
  sorry

-- Chain-specific bound
theorem countDependencies_chain {L : List Constraint}
    (h_chain : ∀ c ∈ L, c.dependencies.length ≤ 1) :
    countDependencies L ≤ L.length := by
  sorry
```

### For Topological Sort
```lean
-- Order preservation
theorem topological_order_index_lt
    (constraints : List Constraint) (order : List ConstraintId)
    (h_sorted : isTopologicallySorted constraints order)
    (c : Constraint) (hc : c ∈ constraints) (dep : ConstraintId) (hdep : dep ∈ c.dependencies) :
    ∃ (idx_c idx_dep : Nat),
      order[idx_c]? = some c.id ∧
      order[idx_dep]? = some dep ∧
      idx_dep < idx_c := by
  exact h_sorted c hc dep hdep
```

---

## Proof Completeness Checklist

For each test case, verify:

- [ ] All definitions are unfolded correctly
- [ ] All implicit arguments are handled
- [ ] All side conditions are discharged
- [ ] No `sorry` or `admit` remain
- [ ] Proof compiles with `lake build`
- [ ] Proof is readable and maintainable
- [ ] Comments explain non-trivial steps

---

## Conclusion

### Test Cases Summary
- **Total test cases**: 8
- **Trivial/Easy**: 4 (1, 5, 6, 4)
- **Moderate**: 3 (2, 7, 8)
- **Hard/Blocked**: 1 (3)

### Critical Path
1. **Resolve `hasCycle` definition** - Blocks Test Cases 2, 3, 8
2. **Prove easy cases first** - Build confidence and infrastructure
3. **Extract common lemmas** - Reduce code duplication
4. **Add comments** - Explain design decisions and limitations

### Estimated Effort
- **With current definitions**: 4-8 hours (assuming hasCycle issue is resolved)
- **With improved hasCycle**: +4-6 hours (to implement proper cycle detection)

### Next Steps
1. Decide on `hasCycle` definition approach
2. Implement/modify definition if needed
3. Prove Test Case 1 (quick win)
4. Prove Test Cases 5, 6 (build momentum)
5. Prove Test Cases 2, 7, 8 (moderate work)
6. Address Test Case 4 (may need clarification)
7. Tackle Test Case 3 (depends on definition)

---

**END OF ANALYSIS**
