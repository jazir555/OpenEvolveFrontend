import RESE.Basic

/-!
# RESE.Constraint

Formalization of constraint theory for RESE.

This module provides the Lean 4 formalization of constraints as defined in the
Python Symbolic Constraint Engine (SCE). It includes:

- Constraint types (hard, soft, preference)
- Constraint structure with dependencies
- Dependency graph theory
- Contradiction detection
- Constraint satisfaction
- Complexity measures

## Authors

- Agent O1: Lean 4 Formalization Specialist

-/

open RESE.Basic

namespace RESE.Constraint

/-!
## Section 1: Constraint Types

Formalization of the three constraint types from SCE.
-/

/-- The three types of constraints in RESE -/
inductive ConstraintType where
  | hard       -- Must satisfy (blocking constraint)
  | soft       -- Prefer to satisfy (optimization constraint)
  | preference -- Nice to have (guidance constraint)
deriving Repr, BEq, Hashable

/-!
## Section 2: Constraint Structure

Formalization of the Constraint dataclass from SCE.
-/

/-- A constraint in the RESE system -/
structure Constraint where
  id            : ConstraintId       -- Unique identifier
  type          : ConstraintType     -- Hard/soft/preference
  description   : String            -- Human-readable description
  formalization : Prop              -- Lean 4 proposition (the actual constraint)
  dependencies  : List ConstraintId -- Constraints this depends on
  source        : String := "unknown" -- Where this constraint came from

/-!
## Section 3: Basic Constraint Operations
-/

/-- Check if a constraint is hard -/
def Constraint.isHard (c : Constraint) : Bool :=
  c.type == ConstraintType.hard

/-- Check if a constraint is soft -/
def Constraint.isSoft (c : Constraint) : Bool :=
  c.type == ConstraintType.soft

/-- Check if a constraint is a preference -/
def Constraint.isPref (c : Constraint) : Bool :=
  c.type == ConstraintType.preference

/-!
## Section 4: Dependency Graph Theory

Formalization of dependency graphs and their properties.
-/

/-- A dependency graph is a directed graph with constraints as nodes -/
structure DependencyGraph where
  nodes : List ConstraintId
  edges : List (ConstraintId × ConstraintId)  -- (from, to): from depends on to

/-- Add a node to the graph -/
def DependencyGraph.addNode (g : DependencyGraph) (id : ConstraintId) : DependencyGraph :=
  if id ∈ g.nodes then g else { nodes := id :: g.nodes, edges := g.edges }

/-- Add an edge to the graph -/
def DependencyGraph.addEdge (g : DependencyGraph) (fromId toId : ConstraintId) : DependencyGraph :=
  { nodes := g.nodes, edges := (fromId, toId) :: g.edges }

/-- Get all dependencies of a node (edges where this node is the first element) -/
def DependencyGraph.getDeps (g : DependencyGraph) (id : ConstraintId) : List ConstraintId :=
  g.edges.filter (λ e => e.1 == id) |>.map (λ e => e.2)

/-- Check if the graph has a cycle (simplified version - detects self-loops only) -/
def DependencyGraph.hasCycle (g : DependencyGraph) : Bool :=
  if g.nodes.length = 0 then false
  else
    -- Simplified cycle detection: check for self-loops
    g.edges.any (λ e => e.1 == e.2)

/-- Dependencies are transitive: if A depends on B and B depends on C, then A transitively depends on C -/
def transitiveDepends (g : DependencyGraph) (a b : ConstraintId) : Prop :=
  ∃ path : List ConstraintId,
    path.length > 0 ∧
    path.head? = some a ∧
    path.getLast? = some b ∧
    (∀ i, i < path.length - 1 → (path.getD i "", path.getD (i + 1) "") ∈ g.edges)

/-- A graph has a proper cycle if there exists a path from any node back to itself -/
def DependencyGraph.hasProperCycle (g : DependencyGraph) : Prop :=
  ∃ (a : ConstraintId), transitiveDepends g a a

/-!
## Section 5: Lemmas About Dependencies

Fundamental theorems about constraint dependencies.
-/

/-- A constraint with no dependencies is independent -/
theorem independent_if_no_deps {c : Constraint} (h : c.dependencies = []) :
    ∀ (depId : ConstraintId), ¬depId ∈ c.dependencies := by
  intro depId hdep
  rw [h] at hdep
  contradiction

/-- A relation is irreflexive if no element relates to itself -/
def IsIrreflexive {α : Type} (R : α → α → Prop) : Prop :=
  ∀ a, ¬R a a

/-- Transitive dependencies form a partial order (for direct edges only)

    **LIMITED VERSION**: This theorem uses the Boolean `hasCycle` predicate which only
    detects self-loops (direct edges from a node to itself). For graphs with
    longer cycles (e.g., A → B → C → A), `hasCycle` returns `false` even though
    a cycle exists.

    Therefore, this theorem proves a weaker property: if hasCycle is false
    and nodes is non-empty, then no self-loop edges exist.

    **IMPORTANT**: This theorem requires `g.nodes.length ≠ 0` as an additional premise.
    When nodes is empty, hasCycle returns false even if self-loop edges exist,
    which is a quirk of the simplified implementation.

    For the complete and general theorem, use `transitive_deps_irreflexive_acyclic_proper`
    which uses the proper cycle detection predicate `hasProperCycle`. -/
theorem transitive_deps_partial_order {g : DependencyGraph}
    (h_nonempty : g.nodes.length ≠ 0)
    (hacyclic : ¬g.hasCycle) :
    ∀ (a : ConstraintId), ¬((a, a) ∈ g.edges) := by
  intro a h_self_edge
  -- If there's a direct edge (a,a), then hasCycle would be true
  -- This contradicts hacyclic
  --
  -- Proof strategy:
  -- 1. From (a,a) ∈ edges, prove that edges.any (λ e => e.1 == e.2) = true
  -- 2. From ¬g.hasCycle and h_nonempty, derive ¬(edges.any ... = true)
  -- 3. Contradiction!
  have h_any_true : g.edges.any (λ e => e.1 == e.2) = true := by
    apply List.any_eq_true.2
    use (a, a)
    constructor
    . exact h_self_edge
    . -- Show (a == a) = true
      simp only [Prod.fst, Prod.snd]
      apply beq_self_eq_true
  -- Now show that hacyclic implies ¬(edges.any ... = true)
  have h_not_any : ¬(g.edges.any (λ e => e.1 == e.2) = true) := by
    unfold DependencyGraph.hasCycle at hacyclic
    -- hacyclic: ¬(if nodes.length = 0 then false else edges.any ... = true)
    -- Since nodes is non-empty, this simplifies to: ¬(edges.any ... = true)
    --
    -- Proof: Assume edges.any ... = true, then hasCycle = true (since nodes non-empty)
    -- which contradicts hacyclic
    intro h_any
    -- This directly gives us: (if nodes.length = 0 then false else edges.any ...) = true
    -- Since nodes.length ≠ 0, the if-then-else returns edges.any ...
    -- So we have: edges.any ... = true
    -- But hacyclic says ¬(... = true), so:
    apply hacyclic
    -- Need to show: (if nodes.length = 0 then false else edges.any ...) = true
    -- Since nodes is non-empty and edges.any ... = true (by h_any), this holds
    simp [h_nonempty, h_any]
  -- Contradiction between h_any_true and h_not_any
  contradiction

/-- **Complete theorem using proper cycle detection:**

    This version uses `hasProperCycle` which correctly detects all cycles,
    not just self-loops. This theorem has no limitations and is fully general. -/
theorem transitive_deps_irreflexive_acyclic_proper {g : DependencyGraph}
    (hacyclic : ¬g.hasProperCycle) :
    IsIrreflexive (λ a b : ConstraintId => transitiveDepends g a b) := by
  intro a htrans
  -- Proof: If a transitively depends on itself, then by definition
  -- g.hasProperCycle holds (exists a such that transitiveDepends g a a)
  -- This directly contradicts hacyclic
  apply hacyclic
  -- Exhibit the cycle: the node a itself
  -- htrans is exactly the proof that transitiveDepends g a a
  constructor
  exact htrans

/-!
## Documentation on the hasCycle vs hasProperCycle distinction

The module defines two notions of cycles:

1. **hasCycle** (Bool): Only detects self-loops, i.e., edges where e.1 = e.2
   - Efficient to compute (O(n) scan of edge list)
   - Incomplete: misses cycles like A → B → C → A
   - Used in `transitive_deps_partial_order` (limited theorem)

2. **hasProperCycle** (Prop): Detects any cycle, defined using transitiveDepends
   - Complete: captures all cyclic dependencies
   - Propositional, not Boolean (requires proof search)
   - Used in `transitive_deps_irreflexive_acyclic_proper` (complete theorem)

**Recommendation**: Use `hasProperCycle` and `transitive_deps_irreflexive_acyclic_proper`
 for all formal reasoning about acyclic graphs. The `hasCycle` predicate is kept
 for computational purposes but is insufficient for mathematical proofs.
-/

/-!
## Section 6: Contradiction Detection

Formalization of constraint contradiction.
-/

/-- Two constraints contradict each other if they cannot both be satisfied -/
def contradicts (c1 c2 : Constraint) : Prop :=
  ¬(c1.formalization ∧ c2.formalization)

/-- A set of constraints is inconsistent if some subset contradicts itself -/
def isInconsistent (constraints : List Constraint) : Prop :=
  ∃ (subset : List Constraint),
    subset.length > 0 ∧
    subset ⊆ constraints ∧
    (∀ c ∈ subset, c.formalization) → False

/-!
## Section 7: Constraint Satisfaction

Formalization of what it means to satisfy constraints.
-/

/-- An assignment satisfies a constraint if the constraint's proposition holds -/
def satisfies (assignment : ConstraintId → Prop) (c : Constraint) : Prop :=
  -- In a full implementation, we would evaluate the constraint under the assignment
  -- For now, we just check if the constraint's proposition holds
  c.formalization

/-- An assignment satisfies all constraints in a list -/
def satisfiesAll (assignment : ConstraintId → Prop) (constraints : List Constraint) : Prop :=
  ∀ c ∈ constraints, satisfies assignment c

/-!
## Section 8: Complexity Measures

Metrics for constraint systems.
-/

/-- Count how many constraints are hard -/
def countHard (constraints : List Constraint) : Nat :=
  constraints.filter (λ c => c.isHard) |>.length

/-- Count total dependencies in a constraint set -/
def countDependencies (constraints : List Constraint) : Nat :=
  constraints.foldl (λ (acc : Nat) (c : Constraint) => acc + c.dependencies.length) 0

/-- Count how many constraints are soft -/
def countSoft (constraints : List Constraint) : Nat :=
  constraints.filter (λ c => c.isSoft) |>.length

/-- Count how many constraints are preferences -/
def countPref (constraints : List Constraint) : Nat :=
  constraints.filter (λ c => c.isPref) |>.length

/-- Calculate total complexity score (weighted sum) -/
def complexityScore (constraints : List Constraint) : Nat :=
  countHard constraints * 3 +
  countSoft constraints * 2 +
  countPref constraints * 1

end RESE.Constraint
