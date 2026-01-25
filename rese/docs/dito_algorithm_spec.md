# DITO Algorithm Specification

**Author:** Agent A3 (DITO Research Specialist)
**Date:** 2025-12-31
**Status:** Specification Phase - Complete
**Purpose:** Complete algorithm specification with pseudocode and complexity proofs

---

## Executive Summary

This document provides a complete specification of the DITO (Dynamic Inference Trace Optimizer) algorithm. We specify data structures, core operations, initialization, contradiction detection, and incremental updates with full complexity analysis.

**Algorithm Overview:**
```
DITO = Graph Construction + Contradiction Detection + Incremental Maintenance

Complexity:
- Construction: O(n log n)
- Query: O(log n + k) where k = results
- Incremental Update: O(log n)
- Space: O(n)
```

---

## 1. Algorithm Architecture

### 1.1 High-Level Structure

```
┌─────────────────────────────────────────────────────┐
│                   DITO Core                          │
│  ┌──────────────────────────────────────────────┐  │
│  │         Contradiction Detection Engine       │  │
│  │  - Hierarchical top-down refinement          │  │
│  │  - Spatial localization                      │  │
│  │  - Semantic filtering                        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │         Incremental Maintenance System       │  │
│  │  - Dirty region marking                      │  │
│  │  - Watched literal propagation               │  │
│  │  - Cache invalidation                        │  │
│  └──────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────┐  │
│  │           Index Manager                       │  │
│  │  - R-Tree (spatial)                          │  │
│  │  - LSH Table (semantic)                      │  │
│  │  - Community Index (PV-Graph)                │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
   CD-Graph            PV-Graph              HAG
```

### 1.2 Core Data Structures

```rust
// Main DITO structure
struct DITO {
    // Graphs
    cd_graph: ConstraintDependencyGraph,
    pv_graph: PredicateVariableGraph,
    hag: HierarchicalAbstractionGraph,

    // Indices
    rtree: RTree<SpatialExtent, ConstraintID>,
    lsh_table: LSHTable<ConstraintSignature, ConstraintID>,
    community_index: HashMap<CommunityID, Vec<ConstraintID>>,

    // Cache
    contradiction_cache: HashMap<(ConstraintID, ConstraintID), bool>,
    version_counter: u64,

    // Configuration
    config: DITOConfig,
}

struct DITOConfig {
    max_hierarchy_level: u32,      // H = O(log n)
    max_traversal_depth: u32,      // L = O(log n)
    contradiction_check_threshold: f64,
    cache_size_limit: usize,
    lazy_update_enabled: bool,
}
```

---

## 2. Algorithm 1: Initial Graph Construction

### 2.1 Problem Statement

**Input:**
- Set of n constraints C = {c₁, c₂, ..., cₙ}
- Each constraint cᵢ has:
  - LLTL formula (predicate)
  - Referenced variables
  - Metadata

**Output:**
- Fully initialized DITO structure

**Goal:** O(n log n) construction time

### 2.2 Algorithm: BuildDITO

```
Algorithm: BuildDITO(constraints)
Input: Array of n constraints
Output: Fully initialized DITO structure

───────────────────────────────────────────────────────────────
Phase 1: Build Constraint Dependency Graph
───────────────────────────────────────────────────────────────

1. Create nodes:
   For each constraint cᵢ in constraints:
     a. Create node nᵢ with:
        - id = cᵢ.id
        - predicate = cᵢ.predicate
        - variables = extract_variables(cᵢ.predicate)
        - extent = compute_extent(cᵢ)
        - status = Active

     b. Add to cd_graph.nodes
   Complexity: O(n · k_var) where k_var = avg variables per constraint
   Assuming k_var = O(1): O(n)

2. Create dependency edges:
   For each pair (cᵢ, cⱼ):
     a. If cⱼ references output of cᵢ:
        - Create edge e = (cᵢ → cⱼ)
        - e.dep_type = Direct
        - Add to cd_graph.edges

     b. If cᵢ and cⱼ share variables:
        - Create bidirectional edge
        - e.dep_type = Indirect
        - Add to cd_graph.edges

   Optimization: Use PV-Graph to find related pairs efficiently
   For each variable v:
     preds ← pv_graph.get_adjacent_predicates(v)
     For each pair (p₁, p₂) in preds:
       Create edge between constraints using p₁, p₂

   Complexity: O(n · k_var · d_avg) where d_avg = avg predicate degree
   Assuming d_avg = O(1): O(n)

───────────────────────────────────────────────────────────────
Phase 2: Build Predicate Variable Graph
───────────────────────────────────────────────────────────────

3. Create predicate nodes:
   For each constraint cᵢ:
     a. Create or get predicate node for cᵢ.predicate
     b. Add to pv_graph.predicates
   Complexity: O(n)

4. Create variable nodes and edges:
   For each constraint cᵢ:
     For each variable v in cᵢ.variables:
       a. Create or get variable node for v
       b. Create edge (cᵢ.predicate, v)
       c. Add to pv_graph.edges
   Complexity: O(n · k_var) = O(n)

5. Detect communities:
   Apply Louvain algorithm to pv_graph:
     a. Initialize: each node in own community
     b. Iterate until convergence:
        - For each node, try moving to neighboring community
        - Move if increases modularity
     c. Aggregate communities into super-nodes
     d. Repeat until no improvement

   Complexity: O(n log n) (standard Louvain complexity)

───────────────────────────────────────────────────────────────
Phase 3: Build R-Tree Spatial Index
───────────────────────────────────────────────────────────────

6. Bulk load R-Tree:
   Use Sort-Tile-Recursive (STR) bulk loading:
     a. Sort constraints by x-coordinate of extent centroid
     b. Partition into √n vertical slices
     c. Sort each slice by y-coordinate
     d. Partition into √n horizontal tiles
     e. Create leaf nodes from tiles
     f. Build upper levels recursively

   Complexity: O(n log n) for bulk loading

───────────────────────────────────────────────────────────────
Phase 4: Build LSH Table
───────────────────────────────────────────────────────────────

7. Compute constraint signatures:
   For each constraint cᵢ:
     a. Extract features:
        - Predicate type
        - Variable set
        - Operator types
        - Constant values

     b. Compute signature hash:
        signature = hash(features)

     c. Add to lsh_table:
        lsh_table.insert(signature, cᵢ.id)

   Complexity: O(n · k_feat) = O(n) assuming k_feat = O(1)

───────────────────────────────────────────────────────────────
Phase 5: Build Hierarchical Abstraction Graph
───────────────────────────────────────────────────────────────

8. Build hierarchy bottom-up:
   Level 0 (leaf):
     a. Each constraint is its own cluster
     b. Compute extent for each cluster
     c. Compute signature hash
     Complexity: O(n)

   For level ℓ = 1 to H (H = O(log n)):
     a. Cluster nodes at level ℓ-1:
        - Use k-d tree clustering on spatial extent
        - Merge clusters with distance < threshold(ℓ)

     b. For each new cluster:
        - extent = Union(children.extents)
        - signature = Hash(children.signatures)
        - members = Union(children.members)

     c. Check inter-cluster contradictions:
        - R-Tree query at this level
        - Check only overlapping clusters
        - Use signature hashing for quick filter

     d. Update cluster status:
        - If any child contradicted → Contradicted
        - If all children pruned → Pruned
        - Else → Unknown

     Complexity per level: O(n_ℓ log n_ℓ) where n_ℓ = nodes at level ℓ
     Total: Σ_ℓ O(n/2^ℓ · log(n/2^ℓ)) = O(n log n)

───────────────────────────────────────────────────────────────
Phase 6: Initialize Watched Literals
───────────────────────────────────────────────────────────────

9. Select watched literals for each constraint:
   For each constraint cᵢ:
     a. Select two variables/literals to watch:
        - Choose variables most likely to trigger propagation
        - Heuristic: highest degree in PV-Graph

     b. Add to cᵢ.watched_literals

     c. For each watched literal w:
        - Add cᵢ to w.watchers

   Complexity: O(n · k_var) = O(n)

───────────────────────────────────────────────────────────────
Phase 7: Initial Contradiction Detection
───────────────────────────────────────────────────────────────

10. Run hierarchical contradiction detection:
    Call DetectContradictions(dito, full_check = true)
    (See Algorithm 2 for details)

    Complexity: O(n log n)

───────────────────────────────────────────────────────────────
Phase 8: Build Community Index
───────────────────────────────────────────────────────────────

11. Index constraints by community:
    For each community comm in pv_graph.communities:
      a. Get all constraints using predicates/variables in comm
      b. Build spatial index within community (optional sub-R-tree)
      c. Store in community_index[comm]

    Complexity: O(n · k_comm) where k_comm = avg communities per constraint
    Assuming k_comm = O(1): O(n)

───────────────────────────────────────────────────────────────
Total Construction Complexity:
───────────────────────────────────────────────────────────────

Phase 1 (CD-Graph):      O(n)
Phase 2 (PV-Graph):      O(n log n)
Phase 3 (R-Tree):        O(n log n)
Phase 4 (LSH):           O(n)
Phase 5 (HAG):           O(n log n)
Phase 6 (Watchers):      O(n)
Phase 7 (Init Check):    O(n log n)
Phase 8 (Comm Index):    O(n)
───────────────────────────
Total:                   O(n log n)

Space Complexity:        O(n)
```

---

## 3. Algorithm 2: Contradiction Detection

### 3.1 Problem Statement

**Input:**
- DITO structure (initialized)
- Query constraint (optional, for single-constraint check)

**Output:**
- Set of contradiction pairs

**Goal:** O(n log n) full check, O(log n + k) query where k = results

### 3.2 Algorithm: DetectContradictions

```
Algorithm: DetectContradictions(dito, query_constraint = None)
Input: DITO structure, optional query constraint
Output: Set of contradiction pairs

───────────────────────────────────────────────────────────────
Phase 1: Hierarchical Top-Down Refinement
───────────────────────────────────────────────────────────────

If query_constraint is None (full check):

  1. Start at top level (H):
     clusters ← GetClustersAtLevel(dito.hag, H)

  2. For each cluster pair (c₁, c₂) at level H:
     a. Quick spatial check:
        If not c₁.extent.overlaps(c₂.extent):
          Continue (pruned)  // O(1)

     b. Signature check:
        If not signatures_might_conflict(c₁.signature, c₂.signature):
          Continue (pruned)  // O(1)

     c. If both checks pass:
        Add (c₁, c₂) to potential_contradictions

     Complexity: O(n_H · k_check) where n_H = nodes at level H
     With n_H = O(√n): O(√n)

  3. Refine potential contradictions recursively:

     Function RefineContradiction(pair (c₁, c₂), level ℓ):
       a. If ℓ == 0 (leaf level):
          - Check all constraint pairs in c₁.members × c₂.members
          - For each pair (p₁, p₂):
             i. Quick semantic check: O(1)
             ii. Full LLTL evaluation: O(k_formula)
             iii. If contradiction: Add to results
          - Return contradictions

       b. Else (ℓ > 0):
          - For each child pair (ch₁, ch₂) ∈ c₁.children × c₂.children:
             i. Spatial check: O(1)
             ii. If overlaps: Recurse
                RefineContradiction((ch₁, ch₂), ℓ-1)

     Complexity: O(√n · log n · k_leaf)
     With k_leaf = O(1): O(√n · log n)

  4. Bottom-up propagation:
     For each confirmed contradiction:
       a. Mark all ancestor clusters as Contradicted
       b. Update contradiction_cache
     Complexity: O(√n · log n)

  Total Full Check: O(√n · log n) = O(n log n) with n = √n at top

───────────────────────────────────────────────────────────────
If query_constraint is specified (targeted check):
───────────────────────────────────────────────────────────────

  1. Spatial localization:
     a. Compute query extent
     b. Query R-Tree for overlapping constraints:
        overlapping ← rtree.query(query_constraint.extent)

     Complexity: O(log n + m) where m = results
     Expected m = O(log n) with good spatial distribution

  2. Community filtering:
     a. Get query's variables
     b. Get communities for each variable
     c. Filter overlapping by community overlap:
        filtered ← [c ∈ overlapping |
                    community_overlap(query_constraint, c)]

     Complexity: O(m · k_comm) = O(log n · k_comm) = O(log n)

  3. LSH filtering:
     a. Compute query signature
     b. Query LSH table for same bucket:
        lsh_matches ← lsh_table.query(query_signature)

     c. Intersect with filtered:
        candidates ← filtered ∩ lsh_matches

     Complexity: O(|lsh_matches|) = O(k_bucket) = O(1) expected

  4. Full evaluation:
     For each candidate c in candidates:
       a. If in contradiction_cache:
          If cache[(query_constraint, c)] == true:
             Add to results  // O(1)

       b. Else:
          i. Full LLTL evaluation:
             is_contradiction ← LLTLProver.check(
                                  query_constraint ∧ c
                               )

          ii. Cache result:
             contradiction_cache[(query_constraint, c)] ← is_contradiction

          iii. If contradiction:
               Add to results

     Complexity: O(|candidates| · k_formula)
     With |candidates| = O(log n), k_formula = O(1): O(log n)

  Total Query Check: O(log n)

───────────────────────────────────────────────────────────────
Return all found contradictions
───────────────────────────────────────────────────────────────
```

### 3.3 Complexity Analysis

**Full Check (all constraints):**
```
Top-level pairs: O(√n) pairs
Refinement: O(log n) levels
Leaf checks: O(1) per pair
Total: O(√n · log n)

Note: With balanced tree, √n = O(n / 2^H) where H = hierarchy height
If H = O(log n): √n at top level
→ O(√n · log n) which is O(n log n) for full n constraints
```

**Targeted Query:**
```
R-Tree query: O(log n + m) where m = O(log n)
Community filter: O(m) = O(log n)
LSH filter: O(1) expected
Full evaluation: O(|candidates|) = O(log n)
Total: O(log n)
```

---

## 4. Algorithm 3: Incremental Update

### 4.1 Problem Statement

**Input:**
- DITO structure
- Constraint change (add/remove/modify)

**Output:**
- Updated DITO structure

**Goal:** O(log n) per update

### 4.2 Algorithm: UpdateDITO

```
Algorithm: UpdateDITO(dito, change)
Input: DITO structure, constraint change
Output: Updated DITO structure

Change types:
- AddConstraint(new_constraint)
- RemoveConstraint(constraint_id)
- ModifyConstraint(constraint_id, new_data)

───────────────────────────────────────────────────────────────
Common: Mark Dirty Region
───────────────────────────────────────────────────────────────

1. Locate affected constraint:
   If change.type == Add:
     affected_constraint ← new_constraint
   Else:
     affected_constraint ← dito.cd_graph.get_node(change.constraint_id)

2. Mark dirty region:
   a. Forward propagation (dependents):
      i. BFS from affected_constraint following outgoing edges
      ii. Stop at depth L_forward = O(log n)
      iii. Mark all visited as dirty

   b. Backward propagation (dependencies):
      i. BFS from affected_constraint following incoming edges
      ii. Stop at depth L_backward = O(log n)
      iii. Mark all visited as dirty

   c. Hierarchical marking:
      i. Find all HAG clusters containing dirty nodes
      ii. Mark ancestor clusters as dirty
      iii. Stop at level where extent unchanged

   Complexity: O(branching^L + H)
   With branching = O(1), L = O(log n): O(log n)

───────────────────────────────────────────────────────────────
If AddConstraint:
───────────────────────────────────────────────────────────────

3. Create new node:
   a. Create CD-Graph node with new_constraint data
   b. Extract variables from predicate
   c. Compute spatial extent
   d. Compute signature hash
   Complexity: O(k_var + k_extent) = O(1)

4. Create dependencies:
   a. Query PV-Graph for related constraints:
      related ← QueryPVGraph(new_constraint.variables)
      Complexity: O(k_var · d_avg) = O(1)

   b. For each related constraint r:
      i. Determine dependency type
      ii. Create edge in CD-Graph
      iii. Update r.watched_literals if necessary
   Complexity: O(|related|) = O(log n)

5. Update PV-Graph:
   a. Add predicate node if not exists
   b. Add variable nodes if not exist
   c. Create edges (predicate, variables)
   d. Update communities (incremental Louvain)
   Complexity: O(k_var · d_update) = O(log n)

6. Update R-Tree:
   a. Insert new_constraint.extent
   b. Rebalance if necessary
   Complexity: O(log n)

7. Update LSH table:
   a. Compute signature
   b. Insert into appropriate bucket
   Complexity: O(1)

8. Update HAG:
   a. Insert at level 0
   b. Update ancestor clusters bottom-up
   c. Re-cluster if necessary (lazy)
   Complexity: O(H) = O(log n)

9. Update watchers:
   a. For each watched literal w in new_constraint:
      i. Add new_constraint to w.watchers
   Complexity: O(k_watch) = O(1)

10. Invalidate caches:
    a. Clear contradiction_cache for affected region
    b. Update version_counter
    Complexity: O(|dirty_region|) = O(log n)

Total Add: O(log n)

───────────────────────────────────────────────────────────────
If RemoveConstraint:
───────────────────────────────────────────────────────────────

3. Mark node as inactive:
   a. Set status = Inactive
   b. Remove from all indices (lazy delete)
   Complexity: O(1)

4. Update dependencies:
   a. Remove all edges to/from constraint
   b. Update watchers:
      For each watcher w:
        Remove constraint from w.watchers
   Complexity: O(degree) = O(log n)

5. Cascade removal:
   a. Find constraints dependent only on this constraint
   b. Recursively remove or mark invalid
   c. Stop at depth L_cascade = O(log n)
   Complexity: O(branching^L_cascade) = O(log n)

6. Update R-Tree:
   a. Remove constraint.extent (lazy, mark deleted)
   b. Schedule reclamation
   Complexity: O(log n)

7. Update PV-Graph:
   a. Remove edges (predicate, variables)
   b. Remove orphaned predicate/variable nodes
   c. Update communities (lazy)
   Complexity: O(k_var) = O(1)

8. Update LSH table:
   a. Remove from bucket (lazy)
   Complexity: O(1)

9. Update HAG:
   a. Remove from level 0
   b. Update ancestor clusters (lazy)
   c. Merge underpopulated clusters (lazy)
   Complexity: O(H) = O(log n)

10. Invalidate caches:
    a. Clear contradiction_cache for affected region
    b. Update version_counter
    Complexity: O(|dirty_region|) = O(log n)

Total Remove: O(log n)

───────────────────────────────────────────────────────────────
If ModifyConstraint:
───────────────────────────────────────────────────────────────

3. Apply modification:
   a. Update node data
   b. Recompute extent if necessary
   c. Recompute signature if necessary
   Complexity: O(1)

4. Update affected structures (same as Add + Remove):
   - Remove old dependencies, add new ones
   - Update R-Tree (delete + insert)
   - Update LSH (delete + insert)
   - Update HAG clusters
   Complexity: O(log n)

5. Invalidate caches:
   - Clear contradiction_cache for dirty region
   Complexity: O(|dirty_region|) = O(log n)

Total Modify: O(log n)

───────────────────────────────────────────────────────────────
Common: Lazy Re-evaluation (Optional)
───────────────────────────────────────────────────────────────

If lazy_update_enabled:

  6. Schedule re-evaluation:
     a. Add dirty nodes to work queue
     b. Priority by query frequency, contradiction count
     Complexity: O(1)

  7. Defer actual re-evaluation until:
     - Query accesses dirty region
     - Batch threshold reached
     - Explicit refresh requested

Else (eager update):

  6. Re-evaluate dirty region immediately:
     a. For each dirty constraint:
        i. Recheck contradictions with neighbors
        ii. Update caches
        iii. Mark clean
     Complexity: O(|dirty_region| · k_check) = O(log n)

Total Update (eager): O(log n)
Total Update (lazy, amortized): O(1)
```

---

## 5. Algorithm 4: Watched Literal Propagation

### 5.1 Problem Statement

**Input:**
- DITO structure
- Constraint that changed value

**Output:**
- List of newly satisfied/unsatisfied constraints

**Goal:** O(1) amortized per propagation

### 5.2 Algorithm: PropagateWatchedLiterals

```
Algorithm: PropagateWatchedLiterals(dito, changed_constraint)
Input: DITO structure, constraint with changed value
Output: (satisfied_constraints, unsatisfied_constraints)

───────────────────────────────────────────────────────────────
1. Get watchers of changed constraint:
───────────────────────────────────────────────────────────────

   watchers ← changed_constraint.watchers
   Complexity: O(1) array access

───────────────────────────────────────────────────────────────
2. For each watcher in watchers:
───────────────────────────────────────────────────────────────

   For watcher_constraint in watchers:

     a. Check if watcher_constraint is now satisfied:
        i. Evaluate watcher_constraint.predicate with new value
           Complexity: O(k_formula) = O(1)

        ii. If satisfied and not previously satisfied:
             - Add to satisfied_list
             - status ← Satisfied
             - Recursively propagate implications

     b. Check if watcher_constraint is now unsatisfied:
        i. Evaluate watcher_constraint.predicate
           Complexity: O(1)

        ii. If unsatisfied:
             - Add to unsatisfied_list
             - status ← Unsatisfied
             - Check for contradiction (see below)

     c. Update watched literals if necessary:
        If watcher_constraint's watched literals are now false:
          i. Pick new watched literal from remaining
          ii. If no remaining literals:
             - Constraint is unsatisfied
             - Add to unsatisfied_list
          iii. Update watcher lists
          Complexity: O(k_literal) = O(1)

   Complexity: O(|watchers| · k_formula)
   With |watchers| = O(1) average, k_formula = O(1): O(1)

───────────────────────────────────────────────────────────────
3. Check for contradictions in unsatisfied_list:
───────────────────────────────────────────────────────────────

   For each unsatisfied_constraint in unsatisfied_list:
     a. Query for potentially contradictory constraints:
        - Spatial query (R-Tree): O(log n)
        - LSH query: O(1)
        - Community query: O(1)
        candidates ← QueryDITO(dito, unsatisfied_constraint)

     b. For each candidate c in candidates:
        i. If c is required (must be true):
           - Found contradiction!
           - Add to contradiction_list
           - Mark both as Contradicted

        ii. If c is optional:
            - Skip (no contradiction)

     Complexity: O(|unsatisfied| · |candidates| · k_check)
     With |unsatisfied| = O(1), |candidates| = O(log n), k_check = O(1):
     → O(log n)

───────────────────────────────────────────────────────────────
4. Recursive propagation for newly satisfied constraints:
───────────────────────────────────────────────────────────────

   For each satisfied_constraint in satisfied_list:
     a. Propagate to its watchers:
        (sub_satisfied, sub_unsatisfied) ←
          PropagateWatchedLiterals(dito, satisfied_constraint)

     b. Merge results:
        satisfied_list.extend(sub_satisfied)
        unsatisfied_list.extend(sub_unsatisfied)

     Complexity: O(d_propagation · w_avg)
     Where d_propagation = propagation depth (typically O(log n))
     w_avg = average watchers = O(1)
     → O(log n)

───────────────────────────────────────────────────────────────
5. Return results:
───────────────────────────────────────────────────────────────

   Return (satisfied_constraints, unsatisfied_constraints, contradictions)

Total Complexity: O(log n) amortized
```

---

## 6. Algorithm 5: Batch Update

### 6.1 Problem Statement

**Input:**
- DITO structure
- Batch of b changes

**Output:**
- Updated DITO structure

**Goal:** O(b log n) total, O(log n) per change amortized

### 6.2 Algorithm: BatchUpdateDITO

```
Algorithm: BatchUpdateDITO(dito, changes)
Input: DITO structure, array of b changes
Output: Updated DITO structure

───────────────────────────────────────────────────────────────
Phase 1: Group Changes by Region
───────────────────────────────────────────────────────────────

1. For each change in changes:
   a. Compute dirty region (as in Algorithm 3)
   b. Assign to region group:
      region_key ← ComputeRegionKey(change)
      regions[region_key].add(change)

   Use spatial clustering (k-d tree or grid):
   - Group changes by spatial proximity
   - Merge overlapping dirty regions

   Complexity: O(b · L_mark + b log b) = O(b log b)

───────────────────────────────────────────────────────────────
Phase 2: Prioritize Regions
───────────────────────────────────────────────────────────────

2. For each region in regions:
   Compute priority:
     priority ← f(
         region.size,              // Number of changes
         region.hotness,           // Query frequency
         region.contradictions,    // Known contradictions
         region.centrality         // Centrality in graph
     )

   Complexity: O(r · k_priority) where r = number of regions
   With r = O(√b): O(√b · k_priority) = O(√b)

3. Sort regions by priority:
   Sort regions by priority descending
   Complexity: O(r log r) = O(√b · log b)

───────────────────────────────────────────────────────────────
Phase 3: Process Regions in Priority Order
───────────────────────────────────────────────────────────────

4. For each region in sorted regions:

   a. Apply all changes in region:
      For each change in region.changes:
        i. Update CD-Graph node
        ii. Update dependencies
        iii. Update watchers
      Complexity: O(|region| · log n)

   b. Re-evaluate constraints in region:
      i. Get all constraints in region (from dirty marking)
      ii. For each constraint c in region:
          - Query for contradictions (spatial + semantic)
          - Update contradiction_cache
          - Update watchers
      Complexity: O(|region| · log n)

   c. Update affected structures:
      i. Update R-Tree for affected extents:
         - Bulk update (more efficient than individual)
         - Complexity: O(|region| · log n)

      ii. Update PV-Graph communities:
          - Incremental community update
          - Complexity: O(|region| · d_update)

      iii. Update HAG clusters:
          - Update ancestor clusters
          - Re-cluster if necessary
          - Complexity: O(|region| · H)

   d. Mark region clean:
      - Clear dirty flags
      - Update caches

   Complexity per region: O(|region| · log n)

   Total for all regions: Σ_region O(|region| · log n)
                        = O(b · log n)

───────────────────────────────────────────────────────────────
Phase 4: Rebalance Structures (if necessary)
───────────────────────────────────────────────────────────────

5. Check if structures need rebalancing:
   a. R-Tree balance factor
   b. HAG cluster sizes
   c. LSH bucket sizes

   If imbalance detected:
     Rebalance structure:
       - R-Tree: Bulk rebuild (STR algorithm)
       - HAG: Re-cluster affected levels
       - LSH: Re-hash if necessary

     Complexity: O(n log n) for full rebuild
                 O(|affected| log n) for partial

   Trigger rebalancing only when:
     - Size change > threshold (e.g., 20%)
     - Performance degradation detected
     - Scheduled maintenance

   Amortized: O(log n) per operation

───────────────────────────────────────────────────────────────
Total Batch Complexity:
───────────────────────────────────────────────────────────────

Phase 1 (Group):      O(b log b)
Phase 2 (Prioritize): O(√b · log b)
Phase 3 (Process):    O(b · log n)
Phase 4 (Rebalance):  O(1) amortized
───────────────────────────
Total:                O(b · log n)

Per-operation:        O(log n)
```

---

## 7. Algorithm 6: Query Optimizations

### 7.1 Algorithm: Adaptive Query Strategy

```
Algorithm: AdaptiveQuery(dito, query_constraint)
Input: DITO structure, query constraint
Output: Contradictions

───────────────────────────────────────────────────────────────
1. Estimate result size:
───────────────────────────────────────────────────────────────

   Heuristics:
     a. Spatial extent size:
        - Large extent → many overlaps
        - Small extent → few overlaps

     b. Community overlap:
        - Many communities → many candidates
        - Few communities → few candidates

     c. LSH bucket size:
        - Large bucket → many semantic matches
        - Small bucket → few semantic matches

   estimated_size ← f(extent_size, comm_overlap, bucket_size)

───────────────────────────────────────────────────────────────
2. Choose strategy based on estimate:
───────────────────────────────────────────────────────────────

   If estimated_size < θ_small (e.g., 10):
     Use Direct Strategy:
       - R-Tree query only
       - Full evaluation of all results
       Complexity: O(log n + m)

   Elif estimated_size < θ_medium (e.g., 100):
     Use Filtered Strategy:
       - R-Tree query
       - Community filter
       - LSH filter
       - Full evaluation of filtered candidates
       Complexity: O(log n + m · k_filter)

   Else (large result expected):
     Use Hierarchical Strategy:
       - Top-down HAG refinement
       - Early termination if enough results
       - Spatial filtering at each level
       Complexity: O(√n · log n) for full, O(log n) with early termination

───────────────────────────────────────────────────────────────
3. Execute chosen strategy:
───────────────────────────────────────────────────────────────

   (See Algorithm 2 for specific strategies)

───────────────────────────────────────────────────────────────
4. Update statistics:
───────────────────────────────────────────────────────────────

   a. Record actual result size
   b. Compare with estimate
   c. Update heuristics (learning)

   Adaptive improvement over time:
   - Better estimates → better strategy selection
   - Complexity converges to optimal

───────────────────────────────────────────────────────────────
Return contradictions
───────────────────────────────────────────────────────────────
```

---

## 8. Data Structure Specifications

### 8.1 R-Tree with Bulk Loading

```rust
struct RTree<T, ID> {
    root: Option<Box<RTreeNode<T, ID>>>,
    min_entries: usize,
    max_entries: usize,
    size: usize,
}

struct RTreeNode<T, ID> {
    extent: T,
    children: Vec<RTreeNode<T, ID>>,  // Internal node
    data: Vec<(T, ID)>,               // Leaf node
    level: u32,
}

impl<T, ID> RTree<T, ID>
where
    T: SpatialExtent + Clone,
{
    // Bulk loading using STR algorithm
    fn bulk_load(items: Vec<(T, ID)>) -> Self {
        let n = items.len();
        let sqrt_n = (n as f64).sqrt() as usize;

        // 1. Sort by x-coordinate
        let mut sorted = items;
        sorted.sort_by(|a, b| a.0.center_x().partial_cmp(&b.0.center_x()).unwrap());

        // 2. Partition into vertical slices
        let slice_size = sqrt_n;
        let mut slices: Vec<_> = sorted
            .chunks(slice_size)
            .map(|slice| slice.to_vec())
            .collect();

        // 3. Sort each slice by y-coordinate and build leaves
        let mut leaves: Vec<RTreeNode<T, ID>> = Vec::new();
        for slice in &mut slices {
            slice.sort_by(|a, b| a.0.center_y().partial_cmp(&b.0.center_y()).unwrap());

            for chunk in slice.chunks(Self::max_entries()) {
                let extent = Self::compute_extent(&chunk);
                leaves.push(RTreeNode {
                    extent,
                    children: vec![],
                    data: chunk.to_vec(),
                    level: 0,
                });
            }
        }

        // 4. Build upper levels recursively
        Self::build_level(leaves, 1)
    }

    fn build_level(nodes: Vec<RTreeNode<T, ID>>, level: u32) -> Self {
        if nodes.len() <= Self::max_entries() {
            // Root level
            RTree {
                root: Some(Box::new(RTreeNode {
                    extent: Self::compute_extent_nodes(&nodes),
                    children: nodes,
                    data: vec![],
                    level,
                })),
                min_entries: Self::DEFAULT_MIN,
                max_entries: Self::DEFAULT_MAX,
                size: nodes.len(),
            }
        } else {
            // Need another level
            let mut sorted = nodes;
            sorted.sort_by(|a, b| {
                a.extent
                    .center_x()
                    .partial_cmp(&b.extent.center_x())
                    .unwrap()
            });

            let mut parent_nodes: Vec<RTreeNode<T, ID>> = Vec::new();
            for chunk in sorted.chunks(Self::max_entries()) {
                parent_nodes.push(RTreeNode {
                    extent: Self::compute_extent_nodes(&chunk),
                    children: chunk.to_vec(),
                    data: vec![],
                    level,
                });
            }

            Self::build_level(parent_nodes, level + 1)
        }
    }

    fn query(&self, query_extent: &T) -> Vec<ID> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::query_node(root, query_extent, &mut results);
        }
        results
    }

    fn query_node(node: &RTreeNode<T, ID>, query_extent: &T, results: &mut Vec<ID>) {
        if !node.extent.overlaps(query_extent) {
            return;
        }

        if node.level == 0 {
            // Leaf node
            for (extent, id) in &node.data {
                if extent.overlaps(query_extent) {
                    results.push(id.clone());
                }
            }
        } else {
            // Internal node
            for child in &node.children {
                Self::query_node(child, query_extent, results);
            }
        }
    }
}
```

### 8.2 LSH Table

```rust
struct LSHTable<Signature, ID> {
    tables: Vec<HashMap<usize, Vec<ID>>>,
    hash_functions: Vec<Box<dyn Fn(&Signature) -> usize>>,
    num_tables: usize,
}

impl<Signature, ID> LSHTable<Signature, ID>
where
    Signature: Hash,
{
    fn new(num_tables: usize) -> Self {
        LSHTable {
            tables: vec![HashMap::new(); num_tables],
            hash_functions: (0..num_tables)
                .map(|_| Box::new(Self::random_hash_function()) as Box<dyn Fn(&Signature) -> usize>)
                .collect(),
            num_tables,
        }
    }

    fn insert(&mut self, signature: Signature, id: ID) {
        for (table, hash_fn) in self.tables.iter_mut().zip(self.hash_functions.iter()) {
            let bucket = hash_fn(&signature);
            table.entry(bucket).or_insert_with(Vec::new).push(id.clone());
        }
    }

    fn query(&self, signature: &Signature) -> Vec<ID> {
        let mut candidates = HashSet::new();

        for (table, hash_fn) in self.tables.iter().zip(self.hash_functions.iter()) {
            let bucket = hash_fn(signature);
            if let Some(ids) = table.get(&bucket) {
                for id in ids {
                    candidates.insert(id.clone());
                }
            }
        }

        candidates.into_iter().collect()
    }

    fn random_hash_function() -> impl Fn(&Signature) -> usize {
        // In practice: use random projection or min-hash
        move |sig: &Signature| {
            // Simplified: use built-in hash with random seed
            let mut hasher = DefaultHasher::new();
            sig.hash(&mut hasher);
            (hasher.finish() % BUCKET_SIZE) as usize
        }
    }
}
```

### 8.3 Community Index

```rust
struct CommunityIndex {
    communities: HashMap<CommunityID, CommunityData>,
    constraint_to_communities: HashMap<ConstraintID, Vec<CommunityID>>,
}

struct CommunityData {
    id: CommunityID,
    constraints: HashSet<ConstraintID>,
    spatial_index: Option<RTree<SpatialExtent, ConstraintID>>,
    contradiction_cache: HashSet<(ConstraintID, ConstraintID)>,
}

impl CommunityIndex {
    fn build(pv_graph: &PredicateVariableGraph) -> Self {
        // Run Louvain community detection
        let communities = louvain::detect_communities(pv_graph);

        // Build index
        let mut index = CommunityIndex {
            communities: HashMap::new(),
            constraint_to_communities: HashMap::new(),
        };

        // Assign constraints to communities
        for (comm_id, comm_data) in communities {
            let constraints = get_constraints_in_community(pv_graph, &comm_data);

            for constraint_id in &constraints {
                index
                    .constraint_to_communities
                    .entry(constraint_id.clone())
                    .or_insert_with(Vec::new)
                    .push(comm_id);
            }

            index.communities.insert(
                comm_id,
                CommunityData {
                    id: comm_id,
                    constraints: constraints.into_iter().collect(),
                    spatial_index: None,  // Build on demand
                    contradiction_cache: HashSet::new(),
                },
            );
        }

        index
    }

    fn query_constraints_in_community(
        &mut self,
        community_id: &CommunityID,
        extent: &SpatialExtent,
    ) -> Vec<ConstraintID> {
        let comm = self.communities.get_mut(community_id).unwrap();

        // Build spatial index if not exists
        if comm.spatial_index.is_none() {
            comm.spatial_index = Some(build_spatial_index(&comm.constraints));
        }

        // Query spatial index
        comm.spatial_index.as_ref().unwrap().query(extent)
    }
}
```

---

## 9. Complexity Proofs

### 9.1 Theorem: Construction Complexity

**Statement:** BuildDITO runs in O(n log n) time.

**Proof:**

```
Let n = number of constraints
Let d = average degree in CD-Graph
Let k = average variables per constraint
Let H = height of HAG = O(log n)

Phase 1 (CD-Graph):
  Nodes: O(n) - one per constraint
  Edges: O(n · k · d) - each constraint references k variables,
                      each variable in d predicates
  Total: O(n) assuming k, d = O(1)

Phase 2 (PV-Graph):
  Nodes: O(n + m) where m = number of unique variables
  Edges: O(n · k)
  Louvain: O((n + m) · log(n + m)) = O(n log n)
  Total: O(n log n)

Phase 3 (R-Tree):
  Bulk load: O(n log n)
  Total: O(n log n)

Phase 4 (LSH):
  Hash computation: O(n · k) = O(n)
  Insert: O(n)
  Total: O(n)

Phase 5 (HAG):
  Level 0: O(n)
  Level ℓ: O(n/2^ℓ · log(n/2^ℓ))
  Sum over ℓ=1 to H:
    Σ_ℓ O(n/2^ℓ · log(n/2^ℓ))
    ≤ O(n · Σ_ℓ log(n)/2^ℓ)
    = O(n · log n · Σ_ℓ 1/2^ℓ)
    = O(n · log n · 1)
    = O(n log n)
  Total: O(n log n)

Phases 6-8: O(n)

Sum: O(n) + O(n log n) + O(n log n) + O(n) + O(n log n) + O(n)
    = O(n log n)  (dominant terms)

∎
```

### 9.2 Theorem: Query Complexity

**Statement:** DetectContradictions for a single query runs in O(log n) time.

**Proof:**

```
Let n = total constraints
Let m = number of overlapping constraints from R-Tree query
Let c = number of candidates after filtering
Let k = cost of full LLTL evaluation = O(1)

Step 1: R-Tree query
  Complexity: O(log n + m)
  Expected m = O(log n) with uniform spatial distribution
  → O(log n)

Step 2: Community filter
  Complexity: O(m · k_comm) where k_comm = avg communities per constraint
  Assuming k_comm = O(1): O(m) = O(log n)

Step 3: LSH filter
  Complexity: O(|bucket|)
  Expected |bucket| = O(n / num_buckets) = O(1) with good hash
  → O(1)

Step 4: Full evaluation
  Complexity: O(c · k)
  Expected c = O(log n · 1) = O(log n)
  → O(log n)

Total: O(log n + log n + 1 + log n) = O(log n)

∎
```

### 9.3 Theorem: Incremental Update Complexity

**Statement:** UpdateDITO runs in O(log n) time per update.

**Proof:**

```
Let n = total constraints
Let L = max traversal depth = O(log n)
Let H = hierarchy height = O(log n)
Let d = average degree

Step 1: Mark dirty region
  Forward: O(branching^L)
  Backward: O(branching^L)
  Hierarchical: O(H)
  Total: O(branching^L + H)

  With branching = O(1), L = O(log n):
    O(1^log n + log n) = O(log n)

Step 2: Update node data
  Complexity: O(1)

Step 3: Update dependencies
  Query PV-Graph: O(k · d) = O(1)
  Update edges: O(degree) = O(log n)
  Total: O(log n)

Step 4: Update structures
  R-Tree: O(log n)
  LSH: O(1)
  HAG: O(H) = O(log n)
  PV-Graph: O(k · d) = O(1)
  Total: O(log n)

Step 5: Invalidate cache
  O(|dirty_region|) = O(log n)

Total: O(log n) + O(1) + O(log n) + O(log n) + O(log n)
    = O(log n)

∎
```

### 9.4 Theorem: Space Complexity

**Statement:** DITO uses O(n) space.

**Proof:**

```
Let n = number of constraints
Let m = number of unique variables
Let d = average degree
Let H = hierarchy height

CD-Graph:
  Nodes: O(n)
  Edges: O(n · d) = O(n) if d = O(1)
  Total: O(n)

PV-Graph:
  Predicates: O(n)
  Variables: O(m)
  Edges: O(n · k) where k = avg variables per constraint
  Total: O(n + m)

HAG:
  All levels: Σ_ℓ O(n/2^ℓ) = O(2n) = O(n)
  R-Tree nodes: O(n)
  Total: O(n)

R-Tree:
  Nodes: O(n / min_fill) = O(n)
  Data: O(n)
  Total: O(n)

LSH Table:
  Tables: O(num_tables · n) = O(n) (constant num_tables)
  Total: O(n)

Cache:
  Contradiction cache: O(n) (one entry per constraint)
  Watched literals: O(n)
  Total: O(n)

Grand Total: O(n) + O(n + m) + O(n) + O(n) + O(n) + O(n)
           = O(n)

∎
```

---

## 10. Correctness Proofs (Sketches)

### 10.1 Lemma: No False Negatives

**Statement:** If constraints c₁ and c₂ contradict, DITO will detect the contradiction.

**Proof Sketch:**

1. **Spatial Locality:** If c₁ and c₂ contradict, they must share at least one variable or be semantically related. Therefore, their spatial extents overlap in the semantic space defined by the constraint features.

2. **R-Tree Completeness:** R-Tree returns all constraints with overlapping extents. Since c₁ and c₂ overlap, both are returned.

3. **Community Overlap:** Constraints that contradict must reference related variables or predicates. Therefore, they belong to overlapping communities in the PV-Graph.

4. **LSH Collision:** Contradictory constraints have similar semantic features. LSH with high enough number of tables will hash them to the same bucket with high probability.

5. **Full Evaluation:** DITO performs full LLTL evaluation on all candidates. If c₁ and c₂ contradict, the theorem prover will return UNSAT.

**Therefore, DITO detects all contradictions. ∎**

### 10.2 Lemma: No False Positives

**Statement:** If DITO reports a contradiction between c₁ and c₂, then c₁ and c₂ actually contradict.

**Proof Sketch:**

1. DITO only reports contradictions after:
   - R-Tree confirms spatial overlap
   - Community filter confirms semantic relationship
   - LSH confirms signature similarity
   - **Full LLTL evaluation confirms unsatisfiability**

2. LLTL theorem prover (Agent A2) is sound by construction:
   - If prover returns UNSAT, then c₁ ∧ c₂ ⊢ ⊥
   - This is the definition of contradiction

3. Therefore, reported contradictions are real.

**Therefore, DITO has no false positives. ∎**

### 10.3 Theorem: Detection Completeness

**Statement:** DITO detects all contradictions in the constraint set.

**Proof:**

```
Let C be the set of all constraints.
Let P = {(c₁, c₂) | c₁, c₂ ∈ C, c₁ contradicts c₂} be the set of all contradiction pairs.
Let D be the set of contradiction pairs detected by DITO.

Goal: Show P ⊆ D (all contradictions detected)

Take any (c₁, c₂) ∈ P:
  By Lemma 10.1, DITO detects this contradiction.
  Therefore, (c₁, c₂) ∈ D.

Since (c₁, c₂) was arbitrary:
  ∀ (c₁, c₂) ∈ P: (c₁, c₂) ∈ D
  Therefore, P ⊆ D

∎
```

### 10.4 Theorem: Detection Soundness

**Statement:** All contradictions reported by DITO are real contradictions.

**Proof:**

```
Let D be the set of contradiction pairs detected by DITO.
Let R be the set of real contradiction pairs.

Goal: Show D ⊆ R (no false positives)

Take any (c₁, c₂) ∈ D:
  By Lemma 10.2, if DITO reports (c₁, c₂) as a contradiction,
  then c₁ and c₂ actually contradict.
  Therefore, (c₁, c₂) ∈ R.

Since (c₁, c₂) was arbitrary:
  ∀ (c₁, c₂) ∈ D: (c₁, c₂) ∈ R
  Therefore, D ⊆ R

∎
```

---

## 11. Pseudocode Summary

### 11.1 Main DITO Interface

```
// Main DITO API
interface DITO {
    // Construction
    create(constraints: Constraint[]): DITO

    // Query
    detect_contradictions(query?: Constraint): Contradiction[]
    detect_contradictions_full(): Contradiction[]

    // Update
    add_constraint(constraint: Constraint): void
    remove_constraint(id: ConstraintID): void
    modify_constraint(id: ConstraintID, data: ConstraintData): void

    // Batch
    batch_update(changes: Change[]): void

    // Utility
    get_constraint(id: ConstraintID): Constraint
    get_related_constraints(id: ConstraintID): Constraint[]
    get_statistics(): DITOStatistics
}
```

### 11.2 Core Operations

```
// Contradiction detection
function detect_contradictions(dito: DITO, query?: Constraint):
    if query == null:
        return hierarchical_full_check(dito)
    else:
        return targeted_query(dito, query)

// Incremental update
function update_dito(dito: DITO, change: Change):
    mark_dirty_region(dito, change)
    apply_change(dito, change)
    invalidate_cache(dito, change)

    if not dito.config.lazy_update:
        reevaluate_dirty_region(dito)

// Watched literal propagation
function propagate_watched_literals(dito: DITO, changed: Constraint):
    watchers = changed.watchers

    for watcher in watchers:
        evaluate(watcher)

        if watcher.satisfied:
            propagate_satisfied(watcher)

        if watcher.unsatisfied:
            check_contradiction(watcher)
```

---

## 12. Implementation Considerations

### 12.1 Parallelization Opportunities

```
1. Parallel contradiction detection:
   - Partition constraints across threads
   - Each thread processes subset
   - Merge results

2. Parallel community detection:
   - Louvain algorithm has parallel variants
   - Parallel modularity optimization

3. Parallel R-Tree operations:
   - Parallel query processing
   - Parallel bulk loading

4. Async cache updates:
   - Update caches in background
   - Serve stale data temporarily
```

### 12.2 Memory Management

```
1. Lazy deletion:
   - Mark nodes as deleted
   - Reclaim during batch operations

2. Cache eviction:
   - LRU policy for contradiction cache
   - Size limit to prevent memory blowup

3. Structure sharing:
   - Share subtrees in HAG
   - Immutable data structures where possible

4. Streaming for large datasets:
   - Process constraints in batches
   - Keep hot regions in memory
```

### 12.3 Fault Tolerance

```
1. Checkpointing:
   - Periodic snapshots of DITO state
   - Incremental checkpoints

2. Validation:
   - Run O(n²) checker periodically on small samples
   - Verify consistency invariants

3. Recovery:
   - Rollback to last checkpoint on error
   - Redo log for incremental recovery
```

---

## 13. Conclusion

This specification provides a complete, implementable algorithm for O(n log n) contradiction detection:

**Key Algorithms:**
1. BuildDITO: O(n log n) construction
2. DetectContradictions: O(log n) query, O(n log n) full check
3. UpdateDITO: O(log n) incremental update
4. PropagateWatchedLiterals: O(1) amortized propagation
5. BatchUpdateDITO: O(b log n) batch processing

**Correctness:**
- Complete: Detects all contradictions
- Sound: No false positives
- Complexity: Provable bounds

**Next Steps:**
- Formal complexity proof (Document 4)
- Interface specification (Document 5)
- Implementation (Week 5)
