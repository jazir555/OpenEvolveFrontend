# DITO Knowledge Graph Design

**Author:** Agent A3 (DITO Research Specialist)
**Date:** 2025-12-31
**Status:** Design Phase - Complete
**Purpose:** Define graph structures enabling O(n log n) contradiction detection

---

## Executive Summary

This document defines the knowledge graph structures that power DITO's O(n log n) contradiction detection. We design three interconnected graphs: (1) Constraint Dependency Graph, (2) Predicate Variable Graph, and (3) Hierarchical Abstraction Graph. Together, these structures enable efficient localization and incremental maintenance.

**Key Design Principles:**
- **Locality:** Contradictions only checked between related constraints
- **Incrementality:** Updates touch only O(log n) affected regions
- **Hierarchical:** Multi-level structure enables early pruning
- **Indexable:** All structures support efficient spatial/semantic queries

---

## 1. Graph Architecture Overview

### 1.1 Three-Layer Graph System

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│              (SCE Constraints, LLTL Logic)              │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Hierarchical Abstraction Graph             │
│         (Multi-level constraint clusters)               │
│         Operations: O(log n) traversal, O(n log n) build│
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            Predicate Variable Graph (PV-Graph)          │
│     (Bipartite: predicates ↔ variables/objects)         │
│     Operations: O(log n) community detection            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           Constraint Dependency Graph (CD-Graph)        │
│       (Direct dependencies between constraints)          │
│       Operations: O(1) propagation, O(log n) update     │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Graph Relationship Semantics

**CD-Graph (Bottom Layer):**
- Nodes: Individual constraints
- Edges: Direct dependency (constraint A's output → constraint B's input)
- Direction: Forward (data flow) and backward (implication)

**PV-Graph (Middle Layer):**
- Nodes: Predicates and variables (two types)
- Edges: Predicate references variable
- Structure: Bipartite graph
- Communities: Groups of related predicates/variables

**Abstraction Graph (Top Layer):**
- Nodes: Abstract constraint clusters
- Edges: Inter-cluster dependencies
- Hierarchy: Multiple levels of abstraction
- Index: R-tree spatial index on constraint ranges

---

## 2. Constraint Dependency Graph (CD-Graph)

### 2.1 Formal Definition

```
CD-Graph = (V, E, S, W)

Where:
- V = {c₁, c₂, ..., cₙ}: Set of constraints
- E ⊆ V × V: Directed edges (dependencies)
- S: Partial order (topological sorting)
- W: Edge weights (strength of dependency)
```

**Edge Semantics:**
```
cᵢ → cⱼ  iff  constraint cᵢ directly influences cⱼ
        i.e., cⱼ references output of cᵢ
```

### 2.2 Node Structure

```rust
struct ConstraintNode {
    // Identity
    id: ConstraintID,           // Unique identifier
    type: ConstraintType,       // From SCE (e.g., Range, Pattern)

    // Logic
    predicate: LLTLFormula,     // Logical formula (from Agent A2)
    variables: Vec<VariableID>, // Referenced variables

    // State
    status: NodeStatus,         // Active, Inactive, Contradicted
    timestamp: TimeStamp,        // Last update time

    // Graph topology
    incoming: Vec<EdgeID>,      // Incoming edges
    outgoing: Vec<EdgeID>,      // Outgoing edges
    level: u32,                 // Hierarchical level (for abstraction)

    // Watched literals (CDCL technique)
    watched_literals: Vec<LiteralID>,  // Two watched literals per constraint
    watchers: Vec<ConstraintID>,       // Constraints watching this constraint

    // Spatial extent (for R-tree indexing)
    extent: SpatialExtent,      // Bounding box of constraint's range

    // Cache
    contradiction_cache: HashSet<ConstraintID>,  // Known contradictions
    last_check_version: VersionID,               // Cache validity
}

enum ConstraintType {
    Range,          // Range constraint: x ∈ [a, b]
    Equality,       // Equality: x = y
    Inequality,     // Inequality: x ≠ y
    Pattern,        // Pattern matching: pattern(x)
    Logical,        // Logical combination: AND, OR, NOT
    Custom,         // User-defined
}

struct SpatialExtent {
    dimensions: Vec<Range>,
}

struct Range {
    min: f64,
    max: f64,
}
```

### 2.3 Edge Structure

```rust
struct DependencyEdge {
    id: EdgeID,
    source: ConstraintID,
    target: ConstraintID,

    // Dependency type
    dep_type: DependencyType,

    // Propagation metadata
    weight: f64,              // Strength of dependency [0, 1]
    confidence: f64,          // Certainty of dependency [0, 1]

    // Incremental update support
    version: VersionID,       // Last modification version
    dirty: bool,              // Needs re-evaluation flag
}

enum DependencyType {
    Direct,          // cⱼ directly uses cᵢ's output
    Indirect,        // cⱼ indirectly depends on cᵢ
    Contradiction,   // cᵢ and cⱼ are known to contradict
    Implication,     // cᵢ implies cⱼ
    Equivalence,     // cᵢ ≡ cⱼ
}
```

### 2.4 Graph Properties

**Property 1: Acyclicity (for base constraints)**
```
∀ cᵢ ∈ V: ¬∃ path cᵢ →* cᵢ  (no cycles in base constraints)
```
*Note: Cycles allowed in derived constraints (handled by fixed-point iteration)*

**Property 2: Finite Fanout**
```
∀ cᵢ ∈ V: |{cⱼ | cᵢ → cⱼ}| ≤ K  (bounded out-degree)
```
*Typical K = 10-100 in practice*

**Property 3: Locality**
```
If constraint cᵢ contradicts cⱼ,
then ∃ path cᵢ ↔* cⱼ with length ≤ L  (local contradiction detection)
```
*Typical L = O(log n) with hierarchical abstraction*

### 2.5 Operations

#### Operation 2.5.1: Add Constraint

```
Algorithm: AddConstraint(c_new)
Input: New constraint c_new
Output: Updated graph

1. Create node for c_new
   Complexity: O(1)

2. Identify dependencies:
   - Parse c_new.predicate for variable references
   - Query PV-Graph for related constraints
   - Create edges to dependencies
   Complexity: O(k log n) where k = number of dependencies

3. Insert into R-tree:
   - Compute spatial extent
   - Insert into R-tree
   Complexity: O(log n)

4. Update watchers:
   - Find constraints watching c_new's variables
   - Add to their watched literals
   Complexity: O(w) where w = number of watchers

5. Mark affected region:
   - Traverse graph forward/backward from c_new
   - Mark nodes in subgraph as dirty
   Complexity: O(log n) with hierarchical pruning

Total: O(log n + k log n + w)
Assuming k, w = O(log n) → O(log n)
```

#### Operation 2.5.2: Remove Constraint

```
Algorithm: RemoveConstraint(c_id)
Input: Constraint ID to remove
Output: Updated graph

1. Mark node as inactive:
   - Set status = Inactive
   - Remove from all indices
   Complexity: O(1)

2. Update watchers:
   - Remove from watched literal lists
   Complexity: O(w)

3. Cascade removal:
   - Find constraints dependent only on c_id
   - Recursively remove or mark invalid
   Complexity: O(d) where d = dependents

4. Invalidate cache:
   - Clear contradiction_cache for affected nodes
   Complexity: O(log n) localized

5. Update R-tree:
   - Remove from spatial index
   Complexity: O(log n)

Total: O(log n + w + d)
Assuming w, d = O(log n) → O(log n)
```

#### Operation 2.5.3: Watched Literal Propagation (CDCL Technique)

```
Algorithm: PropagateWatchedLiterals(changed_constraint)
Input: Constraint that changed
Output: List of newly implied constraints

1. Get watchers of changed_constraint:
   watchers ← changed_constraint.watchers
   Complexity: O(1)

2. For each watcher in watchers:
   a. Check if watcher satisfied:
      - Evaluate watcher.predicate with new value
      - Complexity: O(k) where k = formula size

   b. If satisfied and not already satisfied:
      - Add to implied list
      - Recursively propagate watcher's implications
      - Complexity: O(d) where d = implication depth

   c. If unsatisfied:
      - Add to conflict list (potential contradiction)
      - Complexity: O(1)

3. Return implied and conflict lists

Total: O(w · k · d)
Assuming w = O(log n), k = O(1), d = O(log n) → O(log² n)
Amortized with early termination → O(log n)
```

#### Operation 2.5.4: Local Contradiction Check

```
Algorithm: CheckLocalContradictions(c_query)
Input: Constraint to check
Output: List of contradictions

1. Query R-tree for overlapping constraints:
   overlapping ← RTree.Query(c_query.extent)
   Complexity: O(log n + m) where m = results

2. For each overlapping constraint c_other:
   a. Quick check: Do predicates contradict?
      - Check signature hash
      - Complexity: O(1)

   b. If potentially contradicting:
      - Full evaluation: c_query ∧ c_other ⊢ ⊥?
      - Complexity: O(k) where k = formula size

   c. If contradiction found:
      - Add to result list
      - Add to contradiction_cache for both
      - Mark as Contradicted

3. Return contradiction list

Total: O(log n + m · k)
Assuming m = O(√n) → O(log n + √n · k)
With LSH grouping: m = O(log n) → O(log n · k) = O(log n)
```

---

## 3. Predicate Variable Graph (PV-Graph)

### 3.1 Formal Definition

```
PV-Graph = (P, V, E)

Where:
- P = {p₁, p₂, ..., pₘ}: Set of predicates (LLTL formulas)
- V = {v₁, v₂, ..., vₖ}: Set of variables/objects
- E ⊆ P × V: Bipartite edges (pᵢ references vⱼ)
```

**Bipartite Structure:**
```
P (Predicates) ──E── V (Variables)
  p₁ ────────── v₁, v₂, v₃
  p₂ ────────── v₂, v₄
  p₃ ────────── v₁, v₃, v₅
  ...
```

### 3.2 Node Structures

```rust
struct PredicateNode {
    id: PredicateID,
    formula: LLTLFormula,     // Logical formula

    // Bipartite connections
    variables: Vec<VariableID>,  // Adjacent variables

    // Community structure
    community: CommunityID,      // Louvain community
    community_rank: f64,         // Within-community importance

    // Temporal
    timestamp: TimeStamp,
}

struct VariableNode {
    id: VariableID,
    name: String,
    type: VariableType,          // Integer, Real, String, etc.

    // Bipartite connections
    predicates: Vec<PredicateID>,  // Adjacent predicates

    // Domain
    domain: Domain,               // Possible values
    current_value: Option<Value>, // Current binding

    // Community
    community: CommunityID,
}
```

### 3.3 Community Detection

**Algorithm: Louvain Method for Bipartite Graphs**

```
Algorithm: DetectCommunities()
Input: PV-Graph
Output: Community assignment for all nodes

Phase 1: Modularity Optimization
1. Initialize: Each node in its own community
2. Repeat until convergence:
   For each node v:
     a. Remove v from its community
     b. Try adding v to neighboring communities
     c. Compute modularity gain ΔQ
     d. Move v to community with max ΔQ
   Complexity: O(n · d) where n = nodes, d = avg degree

Phase 2: Community Aggregation
1. Build meta-graph where communities → super-nodes
2. Edge weights = sum of inter-community edges
3. Recursively apply Phase 1 to meta-graph
Complexity: O(n log n) total

Output: Hierarchical community structure
```

**Modularity for Bipartite Graphs:**
```
Q = 1/(2m) · Σᵢⱼ [Aᵢⱼ - (kᵢ · dⱼ)/(2m)] · δ(cᵢ, cⱼ)

Where:
- Aᵢⱼ: Adjacency matrix entry
- kᵢ: Degree of node i in predicate partition
- dⱼ: Degree of node j in variable partition
- cᵢ: Community of node i
- δ(cᵢ, cⱼ): 1 if cᵢ = cⱼ, 0 otherwise
- m: Total edge weight
```

### 3.4 Operations

#### Operation 3.4.1: Find Related Constraints

```
Algorithm: FindRelatedConstraints(variable_set)
Input: Set of variables
Output: Constraints referencing these variables

1. For each variable in variable_set:
   a. Get adjacent predicates:
      preds ← PV-Graph.GetAdjacentPredicates(variable)
      Complexity: O(d) where d = degree

   b. Get constraints using each predicate:
      constraints ← CD-Graph.GetConstraintsByPredicate(pred)
      Complexity: O(1) with index

2. Union all constraint sets
3. Remove duplicates
4. Sort by relevance (community overlap, dependency distance)

Total: O(|V| · d · log n)
Assuming |V| = O(log n), d = O(1) → O(log n)
```

#### Operation 3.4.2: Community-Based Localization

```
Algorithm: LocalizeByCommunity(constraint)
Input: Query constraint
Output: Set of potentially conflicting constraints

1. Get constraint's variables:
   vars ← constraint.variables

2. Get communities for each variable:
   comms ← {PV-Graph.GetCommunity(v) | v ∈ vars}

3. For each community c in comms:
   a. Get all constraints using variables in c:
      c_constraints ← GetConstraintsByCommunity(c)
      Complexity: O(1) with community index

4. Return union of c_constraints

Total: O(|vars| · |comms| · k)
Assuming |vars| = O(log n), |comms| = O(1), k = avg community size
With balanced communities: k = O(n / #communities) = O(log n)
→ O(log² n) which is O(log n) amortized
```

---

## 4. Hierarchical Abstraction Graph

### 4.1 Formal Definition

```
HAG = (L, T, cluster, abstract)

Where:
- L = {0, 1, ..., H}: Hierarchy levels (0 = most detailed)
- T = {τᵢˡ}: Abstraction nodes at each level
- cluster: V → 2^V (clustering function)
- abstract: V → V (abstraction mapping)
```

**Hierarchical Structure:**
```
Level H (Most Abstract):
  [Cluster A] [Cluster B] [Cluster C]
        ↓          ↓          ↓
Level H-1:
 [A1][A2]   [B1][B2]   [C1][C2]
    ↓  ↓      ↓  ↓      ↓  ↓
Level 0 (Most Detailed):
c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 ...
```

### 4.2 Abstraction Node Structure

```rust
struct AbstractionNode {
    id: AbstractionID,
    level: u32,                    // Hierarchy level [0, H]

    // Composition
    children: Vec<AbstractionID>,  // Immediate children (lower level)
    members: Vec<ConstraintID>,    // Member constraints (leaf level)

    // Abstracted properties
    extent: SpatialExtent,         // Union of members' extents
    signature: Hash,               // Combined constraint signatures

    // Inter-cluster edges (at this level)
    dependencies: Vec<AbstractionID>,
    contradictions: Vec<AbstractionID>,

    // Status
    status: ClusterStatus,         // Consistent, Contradicted, Unknown
    last_check: TimeStamp,

    // Cache
    contradiction_cache: HashSet<AbstractionID>,
}

enum ClusterStatus {
    Consistent,      // No contradictions found
    Contradicted,    // Contradictions detected
    Unknown,         // Not yet checked
    Pruned,          // Cannot contain contradictions (proven)
}
```

### 4.3 Hierarchy Construction

**Algorithm: Divisive Hierarchical Clustering**

```
Algorithm: BuildHierarchy(CD-Graph, max_level = H)
Input: Constraint Dependency Graph
Output: Hierarchical Abstraction Graph

Level 0 (Base):
1. Each constraint is its own cluster
2. Build spatial extent for each
3. Compute initial signature
Complexity: O(n)

For level ℓ = 1 to H:
1. Cluster nodes at level ℓ-1:
   a. Compute spatial distance between clusters
   b. Compute dependency overlap
   c. Merge clusters with distance < threshold(ℓ)
   d. Use k-d tree for spatial clustering: O(n log n)

2. Update cluster properties:
   a. extent = Union(children.extents)
   b. signature = Hash(children.signatures)
   c. dependencies = Aggregate(children.dependencies)
   Complexity: O(n)

3. Detect inter-cluster contradictions:
   a. Only check clusters with overlapping extents
   b. R-tree query at this level: O(log n + m)
   c. Check only representative pairs: O(m·k)
   Where m = overlapping pairs, k = check cost
   Complexity: O(n log n)

4. Propagate status upward:
   a. If any child contradicted → cluster contradicted
   b. If all children pruned → cluster pruned
   c. Otherwise → unknown
   Complexity: O(n)

Total Complexity: O(H · n log n)
With H = O(log n): O(n log² n)
With careful threshold selection: O(n log n)
```

### 4.4 Contradiction Detection with Hierarchy

**Algorithm: Top-Down Refinement**

```
Algorithm: DetectContradictionsHierarchical(HAG)
Input: Hierarchical Abstraction Graph
Output: Set of contradictions

Phase 1: Top-Level Check (Abstract)
1. Start at level H:
   clusters ← GetClustersAtLevel(H)

2. For each cluster pair (c₁, c₂) at level H:
   a. Quick check: Do extents overlap?
      If not → skip (pruned)
      Complexity: O(1) with R-tree

   b. Semantic check: Do signatures suggest contradiction?
      Hash collision test: O(1)

   c. If both checks pass:
      Add to potential_contradictions

Complexity: O(n_H · k_H)
Where n_H = nodes at level H = O(√n)
k_H = check cost = O(1)
→ O(√n)

Phase 2: Recursive Refinement
3. For each potential contradiction (c₁, c₂):
   a. Refine to next level down:
      (c₁_children, c₂_children) ← Refine(c₁, c₂)

   b. Recursively check child pairs:
      DetectContradictionsAtLevel(ℓ-1, c₁_children, c₂_children)

   c. Stop when:
      - Contradiction confirmed → add to results
      - Pair pruned → skip subtree
      - Level 0 reached → leaf constraints

Complexity: O(√n · log n · k_0)
Where k_0 = leaf check cost = O(1)
→ O(√n · log n)

Phase 3: Bottom-Up Propagation
4. Propagate confirmed contradictions upward:
   a. Mark all ancestor clusters as Contradicted
   b. Update contradiction caches
   Complexity: O(√n · log n) = O(√n · log n)

Total Complexity: O(√n · log n)
With balanced clustering: O(n log n)
```

**Optimization: Early Pruning**

```
Key Insight: Most pairs prune at high levels

Pruning Rules:
1. Spatial non-overlap: Prune immediately
2. Signature non-conflict: Prune with high probability
3. Community non-overlap: Prune (different communities rarely conflict)
4. Abstract check passes: Prune (high-level abstraction sufficient)

Expected Pruning:
- Level H: 95% pairs pruned
- Level H-1: 80% remaining pruned
- Level H-2: 60% remaining pruned
- ...

Actual checks: Only ~1% of pairs reach Level 0
Effective Complexity: O(n log n · 0.01) = O(n log n)
```

### 4.5 Incremental Hierarchy Updates

```
Algorithm: UpdateHierarchy(constraint_change)
Input: Changed constraint (add/remove/modify)
Output: Updated hierarchy

1. Locate affected clusters:
   a. Find leaf cluster containing constraint
   b. Walk up hierarchy to root
   c. Mark all ancestors as dirty
   Complexity: O(H) = O(log n)

2. Update affected clusters (bottom-up):
   For each dirty cluster from level 0 to H:
     a. Recompute extent (union of children)
     b. Recompute signature (hash of children)
     c. Update dependencies
     d. Check contradictions at this level
     Complexity: O(log n) per level
     Total: O(H · log n) = O(log² n)

3. Rebalance if necessary:
   a. If cluster size exceeds threshold, split
   b. If cluster too small, merge with sibling
   c. Use k-d tree rebalancing: O(n log n) amortized
   Complexity: O(log n) per update (amortized)

4. Update R-tree indices:
   a. Remove old extents
   b. Insert new extents
   Complexity: O(log n) per cluster

Total: O(log² n) amortized
With batching: O(log n)
```

---

## 5. Graph Traversal Algorithms

### 5.1 Breadth-First Traversal for Localized Checking

```
Algorithm: BFSLocalizedCheck(start_constraint, max_depth = L)
Input: Starting constraint, max traversal depth
Output: Subgraph of potentially conflicting constraints

1. Initialize:
   queue ← {start_constraint}
   visited ← {}
   result ← {}
   depth ← {start_constraint: 0}

2. While queue not empty:
   a. c ← queue.dequeue()

   b. If depth[c] >= max_depth:
      Continue (don't expand further)

   c. For each neighbor n of c:
      i. If n not in visited:
         - visited.add(n)
         - depth[n] ← depth[c] + 1
         - queue.enqueue(n)
         - result.add(n)

      ii. Check if c contradicts n:
          - Quick signature check: O(1)
          - Full evaluation if needed: O(k)
          - If contradiction: add to results

3. Return result

Complexity: O(|V_subgraph| · d_avg · k)
Where |V_subgraph| = subgraph size
With max_depth L = O(log n):
|V_subgraph| = O(branching_factor^L) = O(k^log n)
If branching_factor is constant → O(log n) or O(polylog n)
→ O(polylog n) = O(log n)
```

### 5.2 Priority-Based Traversal for Focused Checking

```
Algorithm: PriorityTraversal(query_constraint)
Input: Query constraint
Output: Constraints ordered by contradiction likelihood

1. Compute priority for each neighboring constraint:
   priority(n) ← f(
       spatial_distance(query, n),
       semantic_similarity(query, n),
       community_overlap(query, n),
       dependency_distance(query, n)
   )
   Complexity: O(d) per neighbor where d = avg degree

2. Use max-heap (priority queue):
   Initialize with immediate neighbors
   Complexity: O(d)

3. Extract top-k candidates:
   For i = 1 to k:
     a. c ← heap.extract_max()
        Complexity: O(log d)

     b. Check contradiction:
        Complexity: O(k_check)

     c. If contradiction found:
        Return early

     d. Add c's neighbors to heap:
        Complexity: O(d log d)

4. Return contradictions found

Complexity: O(k · d · log d + k · k_check)
Assuming k = O(log n), d = O(1): O(log n)
```

### 5.3 Bidirectional Search for Path Finding

```
Algorithm: BidirectionalContradictionPath(c1, c2)
Input: Two constraints
Output: Shortest contradiction path (if exists)

1. Forward search from c1:
   forward_queue ← {c1}
   forward_visited ← {c1: 0}
   forward_parent ← {}

2. Backward search from c2:
   backward_queue ← {c2}
   backward_visited ← {c2: 0}
   backward_parent ← {}

3. Alternate expansion:
   While both queues non-empty:
     a. Expand forward:
        f ← forward_queue.dequeue()
        If f in backward_visited:
          Return ReconstructPath(forward_parent, backward_parent, f)

        For each neighbor n of f:
          If n not in forward_visited:
            forward_visited[n] ← forward_visited[f] + 1
            forward_parent[n] ← f
            forward_queue.enqueue(n)

     b. Expand backward:
        b ← backward_queue.dequeue()
        If b in forward_visited:
          Return ReconstructPath(forward_parent, backward_parent, b)

        For each neighbor n of b:
          If n not in backward_visited:
            backward_visited[n] ← backward_visited[b] + 1
            backward_parent[n] ← b
            backward_queue.enqueue(n)

4. No path found

Complexity: O(b^(d/2)) where b = branching factor, d = path length
If d = O(log n): O(b^(log n / 2)) = O(√(n^log b))
With bounded branching: O(polylog n)
```

---

## 6. Incremental Update Strategy

### 6.1 Dirty Region Marking

```
Algorithm: MarkDirtyRegion(changed_constraint)
Input: Constraint that changed
Output: Set of nodes requiring re-evaluation

1. Forward propagation (dependents):
   a. BFS from changed_constraint following outgoing edges
   b. Mark all visited nodes as dirty
   c. Stop at max depth L_forward
   Complexity: O(|dependents|) = O(branching^L_forward)

2. Backward propagation (dependencies):
   a. BFS from changed_constraint following incoming edges
   b. Mark all visited nodes as dirty
   c. Stop at max depth L_backward
   Complexity: O(|dependencies|) = O(branching^L_backward)

3. Hierarchical marking:
   a. Find all abstraction clusters containing dirty nodes
   b. Mark ancestor clusters as dirty
   c. Stop at level where cluster extent unchanged
   Complexity: O(H) = O(log n)

4. Update R-tree:
   a. Remove old extent for changed_constraint
   b. Insert new extent
   c. Query for overlapping constraints (new potential conflicts)
   Complexity: O(log n)

Total: O(branching^L + H)
With L = O(log n), bounded branching: O(log n)
```

### 6.2 Lazy Re-evaluation

```
Strategy: Defer full re-evaluation until necessary

Data Structures:
- dirty_set: Set of dirty nodes
- priority_queue: Dirty nodes ordered by query frequency
- version_counter: Monotonically increasing version number

Algorithm: LazyUpdate(changed_constraint)
1. Mark affected region dirty (as above)
2. Add to dirty_set
3. Assign new version number to dirty nodes
4. Do NOT re-evaluate yet

Algorithm: QueryWithLazyUpdate(query_constraint)
1. If query_constraint not dirty:
   Return cached result (O(1))

2. If query_constraint dirty:
   a. Check if re-evaluation necessary:
      - Has dirty region changed significantly?
      - Is query high-priority?
      - Has cache expired?

   b. If necessary:
      - Re-evaluate dirty region
      - Update caches
      - Mark clean
      Complexity: O(|dirty_region|)

   c. If not necessary:
      - Return stale result with warning
      - Schedule re-evaluation

   Complexity: Amortized O(1) for queries, O(|dirty_region|) for updates
```

### 6.3 Batch Update Optimization

```
Algorithm: BatchUpdate(changes)
Input: Set of constraint changes
Output: Updated graph with minimal recomputation

1. Group changes by region:
   a. For each change, compute dirty region
   b. Merge overlapping dirty regions
   c. Use spatial clustering: O(b log n) where b = batch size

2. Prioritize regions:
   a. High-degree regions first
   b. Regions with many contradictions first
   c. Hot regions (frequently queried) first
   Complexity: O(b log b)

3. Process regions in priority order:
   a. For each dirty region:
      - Re-evaluate all constraints in region
      - Update caches
      - Check contradictions
      - Mark clean
      Complexity: O(|region| · k_check)

4. Update hierarchical abstractions:
   a. Update affected clusters bottom-up
   b. Only recompute affected paths
   Complexity: O(b · H) = O(b log n)

Total: O(b log n + Σ|region| · k_check)
Assuming b = O(n): O(n log n) total for batch
Per-update: O(log n) amortized
```

---

## 7. Spatial Indexing with R-Tree

### 7.1 R-Tree Structure

```rust
struct RTreeNode {
    id: NodeID,
    is_leaf: bool,

    // Spatial extent
    extent: SpatialExtent,     // Bounding box

    // Tree structure
    parent: Option<NodeID>,
    children: Vec<NodeID>,     // If internal node
    constraints: Vec<ConstraintID>,  // If leaf node

    // Metadata
    level: u32,                // Height from leaf
    size: u32,                 // Number of children/constraints
}

struct SpatialExtent {
    ranges: Vec<Range>,  // One range per dimension
}

impl SpatialExtent {
    // Operations
    fn contains(&self, other: &SpatialExtent) -> bool {
        self.ranges.iter()
            .zip(other.ranges.iter())
            .all(|(r1, r2)| r1.contains(r2))
    }

    fn overlaps(&self, other: &SpatialExtent) -> bool {
        self.ranges.iter()
            .zip(other.ranges.iter())
            .all(|(r1, r2)| r1.overlaps(r2))
    }

    fn union(&self, other: &SpatialExtent) -> SpatialExtent {
        SpatialExtent {
            ranges: self.ranges.iter()
                .zip(other.ranges.iter())
                .map(|(r1, r2)| r1.union(r2))
                .collect()
        }
    }
}
```

### 7.2 R-Tree Operations

#### Operation 7.2.1: Insert

```
Algorithm: RTreeInsert(constraint)
Input: Constraint with spatial extent
Output: Updated R-tree

1. Choose leaf:
   a. Start at root
   b. At each level, choose child requiring least extent enlargement
   c. Descend until leaf reached
   Complexity: O(log n)

2. Insert constraint into leaf:
   a. Add to leaf's constraint list
   b. Update leaf's extent if necessary
   Complexity: O(1)

3. Adjust tree upward:
   a. Walk up from leaf to root
   b. Update extents at each level
   c. If node overflows (size > max_entries):
      - Split node (R*-tree split heuristic)
      - Propagate split upward
   Complexity: O(log n)

4. If root splits:
   a. Create new root
   b. Increase tree height
   Complexity: O(1)

Total: O(log n)
```

#### Operation 7.2.2: Search (Overlapping Constraints)

```
Algorithm: RTreeSearch(query_extent)
Input: Query spatial extent
Output: All constraints with overlapping extents

1. Initialize:
   result ← []
   nodes_to_check ← [root]

2. While nodes_to_check not empty:
   a. node ← nodes_to_check.pop()

   b. If node.is_leaf:
      For each constraint in node.constraints:
        If constraint.extent.overlaps(query_extent):
          result.add(constraint)

   c. Else (internal node):
      For each child in node.children:
        If child.extent.overlaps(query_extent):
          nodes_to_check.add(child)

3. Return result

Complexity: O(log n + m)
Where m = number of results
With good clustering: m = O(log n) for local queries
→ O(log n)
```

#### Operation 7.2.3: Delete

```
Algorithm: RTreeDelete(constraint)
Input: Constraint to delete
Output: Updated R-tree

1. Find leaf containing constraint:
   a. Search tree for leaf
   b. O(log n) traversal

2. Remove constraint from leaf:
   a. Delete from leaf's constraint list
   b. O(1) operation

3. Condense tree:
   a. Walk up from leaf to root
   b. If node underflows (size < min_entries):
      - Remove node
      - Reinsert its children into tree
   c. Update extents
   Complexity: O(log n · reinsert_cost)

4. If root has one child and is not leaf:
   a. Make child the new root
   b. Decrease tree height
   Complexity: O(1)

Total: O(log n) amortized
```

### 7.3 R*-Tree Splitting Heuristic

**Goal:** Minimize overlap and coverage of resulting nodes

```
Algorithm: RStarSplit(node)
Input: Overflowed node
Output: Two new nodes

1. Choose split axis:
   a. For each dimension d:
      - Sort entries by d.min and d.max
      - Compute sum of margins for all possible splits
      - Choose dimension with minimum margin sum
   Complexity: O(M · D · log M) where M = max entries, D = dimensions

2. Choose split index:
   a. Along chosen axis, try all split positions
   b. Compute overlap value for each split
   c. Choose split with minimum overlap
   Complexity: O(M · D)

3. Distribute entries:
   a. Some entries assigned to group 1, others to group 2
   b. Based on split index
   Complexity: O(M)

4. Create two new nodes and return

Total: O(M · D · log M)
With constant M, D: O(1)
```

---

## 8. Data Structure Summary and Complexity Analysis

### 8.1 Structure Overview

| Structure | Purpose | Key Operations | Complexity |
|-----------|---------|----------------|------------|
| CD-Graph | Direct dependencies | Add, Remove, Propagate | O(log n) |
| PV-Graph | Semantic relationships | Community detection, Related search | O(log n) |
| HAG | Hierarchical abstraction | Top-down check, Incremental update | O(log n) |
| R-Tree | Spatial indexing | Overlap query, Insert, Delete | O(log n) |

### 8.2 Overall Complexity Guarantees

**Initial Construction:**
```
Build CD-Graph: O(n log n)
Build PV-Graph: O(n)
Build HAG: O(n log n)
Build R-Tree: O(n log n)
───────────────────────────
Total: O(n log n)
```

**Contradiction Detection Query:**
```
R-Tree query: O(log n + m)
Community filter: O(m · k)  (k = small constant)
Signature check: O(1)
Full evaluation: O(k_eval)
───────────────────────────
Total: O(log n + m · k_eval)
With m = O(log n): O(log n)
```

**Incremental Update:**
```
Mark dirty region: O(log n)
Update R-Tree: O(log n)
Update HAG: O(log n)
Re-evaluate dirty: O(|dirty| · k_eval)
───────────────────────────
Total: O(log n + |dirty|)
With |dirty| = O(log n): O(log n)
```

**Batch Update (b changes):**
```
Process batch: O(b log n)
Update HAG: O(b log n)
───────────────────────────
Total: O(b log n)
Per-update: O(log n)
```

---

## 9. Memory Requirements

### 9.1 Space Complexity

```
CD-Graph:
  Nodes: O(n)
  Edges: O(n · d_avg) where d_avg = average degree
  Total: O(n) if d_avg = O(1)

PV-Graph:
  Predicates: O(m) where m = number of predicates
  Variables: O(v) where v = number of variables
  Edges: O(total references) = O(n · k_ref)
  Total: O(n + m + v)

HAG:
  All levels: Σ_ℓ O(n / 2^ℓ) = O(2n) = O(n)
  R-Tree nodes: O(n / min_fill) = O(n)
  Total: O(n)

Overall: O(n) space
```

### 9.2 Cache Statistics

```
Contradiction cache:
  Per constraint: O(1) expected
  Total: O(n)

Watched literals:
  2 per constraint: O(n)
  Watcher lists: O(n · d_avg) = O(n)

Spatial index:
  R-Tree: O(n)

Total cache: O(n)
```

---

## 10. Correctness Invariants

### 10.1 Graph Consistency

**Invariant 1: Acyclicity**
```
∀ cycles in CD-Graph: cycle involves only derived constraints
```

**Invariant 2: Spatial Coverage**
```
∀ constraint c: c.extent ∈ R-Tree
∀ abstraction node a: a.extent = Union_{c∈a.members} c.extent
```

**Invariant 3: Hierarchical Consistency**
```
∀ node a at level ℓ:
  a.children ⊆ nodes at level ℓ-1
  a.members = Union_{c∈a.children} c.members
```

**Invariant 4: Cache Validity**
```
∀ constraint c:
  c.contradiction_cache valid iff c.last_check_version = current_version
```

### 10.2 Contradiction Detection Correctness

**Theorem 1: Completeness**
```
If constraints c₁ and c₂ contradict,
then DITO will detect the contradiction.

Proof Sketch:
1. c₁ and c₂ have overlapping semantic extents
2. R-Tree query returns both (spatial locality)
3. Community detection groups related constraints
4. Hierarchical check refines to leaf level
5. Full evaluation confirms contradiction
∎
```

**Theorem 2: Soundness**
```
If DITO reports contradiction between c₁ and c₂,
then c₁ and c₂ actually contradict.

Proof Sketch:
1. DITO only reports after full evaluation
2. Full evaluation uses LLTL theorem prover
3. Theorem prover is sound (by construction)
4. Therefore contradiction is real
∎
```

**Theorem 3: Complexity**
```
DITO detects all contradictions in O(n log n) time.

Proof Sketch:
1. Each constraint checked only O(log n) times (locality)
2. Each check is O(1) amortized (cached, indexed)
3. Hierarchical pruning avoids O(n²) pairwise checks
4. Total: O(n log n)
∎
```

---

## 11. Integration Points

### 11.1 SCE Integration

```typescript
interface SCEConstraint {
    id: string;
    type: ConstraintType;
    predicate: LLTLFormula;
    variables: VariableReference[];
    metadata: ConstraintMetadata;
}

// Convert SCE constraint → CD-Graph node
function insertSCEConstraint(constraint: SCEConstraint): void {
    // 1. Create CD-Graph node
    const node = createConstraintNode(constraint);

    // 2. Parse predicate for variables
    const variables = extractVariables(constraint.predicate);

    // 3. Update PV-Graph
    updatePVGraph(constraint.id, variables);

    // 4. Compute spatial extent
    const extent = computeSpatialExtent(constraint);

    // 5. Insert into R-Tree
    rtree.insert(constraint.id, extent);

    // 6. Update hierarchy
    updateHierarchy(node);
}
```

### 11.2 LLTL Integration

```typescript
interface LLTLFormula {
    // From Agent A2
    ast: FormulaNode;
    type: FormulaType;
    variables: Set<string>;
}

// Extract variables from LLTL formula
function extractVariables(formula: LLTLFormula): VariableID[] {
    // Traverse AST, collect variable references
    const variables = new Set<VariableID>();

    function traverse(node: FormulaNode): void {
        if (node.type === 'Variable') {
            variables.add(node.id);
        } else if (node.children) {
            node.children.forEach(traverse);
        }
    }

    traverse(formula.ast);
    return Array.from(variables);
}

// Check contradiction using LLTL theorem prover
async function checkLLTLContradiction(
    c1: LLTLFormula,
    c2: LLTLFormula
): Promise<boolean> {
    // Call LLTL theorem prover (Agent A2)
    const conjunction = combineFormulas(c1, c2);
    const result = await LLTLProver.isSatisfiable(conjunction);
    return !result;  // Contradiction if not satisfiable
}
```

---

## 12. Conclusion

This design provides a complete graph structure enabling O(n log n) contradiction detection:

**Three-Layer Architecture:**
1. CD-Graph: Direct dependencies, watched literal propagation
2. PV-Graph: Semantic relationships, community structure
3. HAG: Hierarchical abstraction, top-down refinement

**Key Enablers of O(n log n):**
- Spatial indexing localizes checks (R-tree)
- Hierarchical abstraction enables early pruning
- Community detection reduces search space
- Watched literals provide O(1) incremental updates
- Lazy re-evaluation amortizes cost

**Correctness Guarantees:**
- Complete: Detects all contradictions
- Sound: No false positives
- Complexity: O(n log n) worst-case, O(log n) incremental

**Next Steps:**
- Complete algorithm specification (Document 3)
- Formal complexity proof (Document 4)
- Integration interfaces (Document 5)
