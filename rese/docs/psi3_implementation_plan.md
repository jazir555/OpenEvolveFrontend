# Ψ₃ Implementation Plan

**Module:** Ψ₃ Specialist (Constraint Inversion)
**Complexity Target:** 2^n → 2^(n/10) (10x reduction)
**Implementation Timeline:** 8 weeks
**Target Week:** 27

---

## Table of Contents
1. [Implementation Overview](#implementation-overview)
2. [Data Structure Specifications](#data-structure-specifications)
3. [Component Architecture](#component-architecture)
4. [Integration with OpenEvolve](#integration-with-openevolve)
5. [Implementation Phases](#implementation-phases)
6. [Testing Strategy](#testing-strategy)
7. [Performance Optimization](#performance-optimization)
8. [Risk Mitigation](#risk-mitigation)

---

## 1. Implementation Overview

### 1.1 Technology Stack

**Core Implementation Language**: Python 3.11+
- **Rationale**: Rapid prototyping, extensive library ecosystem
- **Production**: Rust rewrite for performance-critical paths

**Key Dependencies**:
```toml
[dependencies]
# SAT/SMT Solvers
z3-solver = "^4.12"           # SMT solver for implication checking
pysat = "^0.1.8"              # SAT solver interfaces

# Graph Algorithms
networkx = "^3.2"             # Dependency graph manipulation
graphillion = "^1.0"          # Advanced graph algorithms

# Formal Verification
lean4 = "^4.0"                # Proof assistant (via subprocess)

# Numerical Computing
numpy = "^1.24"               # Implication matrices
scipy = "^1.11"               # Sparse matrix operations

# Data Structures
attrs = "^23.1"               # Data classes
pydantic = "^2.0"             # Validation

# Testing
pytest = "^7.4"               # Unit testing
hypothesis = "^6.82"          # Property-based testing
```

**Development Tools**:
- **Language Server**: pyright (Python), rust-analyzer (Rust)
- **Testing**: pytest with coverage
- **Benchmarking**: pytest-benchmark
- **Profiling**: py-spy, cProfile

### 1.2 Directory Structure

```
rese/psi3/
├── src/
│   ├── psi3/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── constraint.py          # Constraint data structures
│   │   │   ├── expression.py          # AST definitions
│   │   │   └── metadata.py            # Metadata structures
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── preprocessing.py       # Stage 1: Syntactic
│   │   │   ├── dependency.py          # Stage 2: Dependency analysis
│   │   │   ├── minimal_cover.py       # Stage 3: Minimal cover
│   │   │   └── verification.py        # Stage 4: Verification
│   │   ├── structures/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py               # Dependency graph
│   │   │   ├── matrix.py              # Implication matrix
│   │   │   └── proof_tree.py          # Proof tree
│   │   ├── solvers/
│   │   │   ├── __init__.py
│   │   │   ├── sat_interface.py       # SAT solver interface
│   │   │   └── lean4_interface.py     # Lean 4 interface
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── interface.py           # Public API
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py             # Logging utilities
│   │       └── metrics.py             # Performance metrics
├── tests/
│   ├── unit/
│   │   ├── test_constraint.py
│   │   ├── test_preprocessing.py
│   │   ├── test_dependency.py
│   │   ├── test_minimal_cover.py
│   │   └── test_verification.py
│   ├── integration/
│   │   ├── test_psi3_pipeline.py
│   │   ├── test_stage2_integration.py
│   │   └── test_psi1_integration.py
│   ├── benchmarks/
│   │   ├── bench_reduction.py
│   │   ├── bench_verification.py
│   │   └── bench_real_world.py
│   └── fixtures/
│       ├── constraints/               # Sample constraint sets
│       └── proofs/                    # Expected proof outputs
├── lean4/
│   ├── PSI3/
│   │   ├── Basic.lean
│   │   ├── Constraint.lean
│   │   ├── Equivalence.lean
│   │   └── Theorems.lean
│   └── scripts/
│       ├── verify.py                  # Lean 4 verification script
│       └── export.py                  # Export proofs to Python
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── user_guide.md
├── examples/
│   ├── basic_reduction.py
│   ├── database_queries.py
│   └── type_constraints.py
├── pyproject.toml
├── setup.py
└── README.md
```

### 1.3 Milestones and Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | **Phase 1: Core Infrastructure** | Data structures, basic preprocessing |
| 3-4 | **Phase 2: Dependency Analysis** | SAT integration, dependency graph |
| 5-6 | **Phase 3: Minimal Cover** | Greedy algorithm, optimization |
| 7-8 | **Phase 4: Verification** | Lean 4 integration, testing |

---

## 2. Data Structure Specifications

### 2.1 Core Data Structures

**Constraint Structure**:
```python
from attrs import define, field
from typing import Set, Dict, Any, Optional
from enum import Enum

class ConstraintType(Enum):
    """Constraint type classification"""
    BOOL = "bool"           # Boolean expression
    ARITH = "arith"         # Arithmetic expression
    QUANT = "quant"         # Quantified expression
    TYPE = "type"           # Type constraint

@define(frozen=True)
class Constraint:
    """
    Immutable constraint representation

    Properties:
        - Hashable for use in sets
        - Cached computations for performance
        - Type-safe construction
    """
    id: int                           # Unique identifier
    expr: 'Expr'                      # Logical expression
    type: ConstraintType              # Constraint classification
    vars: frozenset[str]              # Free variables
    metadata: 'Metadata'              # Provenance information

    # Cached fields (computed on construction)
    hash: int = field(init=False)
    normalized: Optional['Expr'] = field(init=False, default=None)

    def __attrs_post_init__(self):
        """Compute cached fields"""
        object.__setattr__(self, 'hash', hash(self.expr))
        object.__setattr__(self, 'normalized', normalize_expr(self.expr))

    def subsumes(self, other: 'Constraint', solver: 'SATInterface') -> bool:
        """
        Check if self ⊨ other (self implies other)

        Uses SAT solver: UNSAT(¬(self → other)) means self ⊨ other
        """
        negation = And(self.expr, Not(other.expr))
        result = solver.solve(negation)
        return result == SatResult.UNSATISFIABLE

    def is_equivalent(self, other: 'Constraint', solver: 'SATInterface') -> bool:
        """Check if self ≡ other (mutual implication)"""
        return self.subsumes(other, solver) and other.subsumes(self, solver)

    def simplify(self) -> 'Constraint':
        """
        Simplify constraint expression
        """
        simplified_expr = simplify_expr(self.expr)
        return Constraint(
            id=self.id,
            expr=simplified_expr,
            type=self.type,
            vars=self.vars,
            metadata=self.metadata
        )
```

**Expression AST**:
```python
from abc import ABC, abstractmethod
from typing import List, Union

class Expr(ABC):
    """Base expression class"""

    @abstractmethod
    def __str__(self) -> str:
        """String representation"""

    @abstractmethod
    def __hash__(self) -> int:
        """Hash for caching"""

    @abstractmethod
    def equals(self, other: 'Expr') -> bool:
        """Structural equality"""

class BoolExpr(Expr):
    """Boolean expression"""

    def __init__(self, op: BoolOp, args: List[Expr]):
        self.op = op
        self.args = args

    def __str__(self) -> str:
        match self.op:
            case BoolOp.AND:
                return f"({' ∧ '.join(str(a) for a in self.args)})"
            case BoolOp.OR:
                return f"({' ∨ '.join(str(a) for a in self.args)})"
            case BoolOp.NOT:
                return f"¬{self.args[0]}"
            case BoolOp.IMPLIES:
                return f"({self.args[0]} → {self.args[1]})"
            case BoolOp.IFF:
                return f"({self.args[0]} ↔ {self.args[1]})"

    def __hash__(self) -> int:
        return hash((self.op, tuple(self.args)))

class BoolOp(Enum):
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()
    IFF = auto()

class ArithExpr(Expr):
    """Arithmetic expression"""

    def __init__(self, op: ArithOp, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def __str__(self) -> str:
        match self.op:
            case ArithOp.LT:
                return f"({self.left} < {self.right})"
            case ArithOp.LE:
                return f"({self.left} ≤ {self.right})"
            case ArithOp.GT:
                return f"({self.left} > {self.right})"
            case ArithOp.GE:
                return f"({self.left} ≥ {self.right})"
            case ArithOp.EQ:
                return f"({self.left} = {self.right})"
            case ArithOp.NE:
                return f"({self.left} ≠ {self.right})"

    def __hash__(self) -> int:
        return hash((self.op, self.left, self.right))

class ArithOp(Enum):
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    EQ = auto()
    NE = auto()

class QuantExpr(Expr):
    """Quantified expression"""

    def __init__(self, quant: Quantifier, vars: List[str], body: Expr):
        self.quant = quant
        self.vars = vars
        self.body = body

    def __str__(self) -> str:
        var_str = ', '.join(self.vars)
        match self.quant:
            case Quantifier.FORALL:
                return f"∀{var_str}. {self.body}"
            case Quantifier.EXISTS:
                return f"∃{var_str}. {self.body}"

    def __hash__(self) -> int:
        return hash((self.quant, tuple(self.vars), self.body))

class Quantifier(Enum):
    FORALL = auto()
    EXISTS = auto()
```

**Metadata Structure**:
```python
@define
class Metadata:
    """
    Constraint metadata
    """
    source: str                          # Origin (user, derived, etc.)
    priority: int = field(default=5)     # Importance (1-10)
    confidence: float = field(default=1.0)  # Trust level (0.0-1.0)
    dependencies: List[int] = field(factory=list)  # Implied constraints
    verified: bool = field(default=False)  # Formal verification status
    timestamp: datetime = field(factory=datetime.now)
    tags: Set[str] = field(factory=set)  # User-defined tags
```

### 2.2 Dependency Graph

```python
import networkx as nx
from typing import Dict, Set, List, Tuple

class DependencyGraph:
    """
    Dependency graph using NetworkX backend

    Properties:
        - DAG (directed acyclic graph) after SCC condensation
        - Efficient graph algorithms via NetworkX
        - Cachable transitive closure
    """

    def __init__(self, constraints: List[Constraint]):
        """
        Initialize graph from constraint list
        """
        self.constraints = {c.id: c for c in constraints}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from([c.id for c in constraints])

        # Cached computations
        self._transitive_closure: Optional[Dict[int, Set[int]]] = None
        self._sccs: Optional[List[Set[int]]] = None

    def add_implication(self, source_id: int, target_id: int):
        """
        Add implication edge: source ⊨ target
        """
        self.graph.add_edge(source_id, target_id)
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidate cached computations"""
        self._transitive_closure = None
        self._sccs = None

    def compute_transitive_closure(self) -> Dict[int, Set[int]]:
        """
        Compute transitive closure using NetworkX
        """
        if self._transitive_closure is not None:
            return self._transitive_closure

        # Use NetworkX transitive closure
        closure = nx.transitive_closure(self.graph)

        self._transitive_closure = {
            node: set(closure.successors(node))
            for node in self.graph.nodes()
        }

        return self._transitive_closure

    def find_strongly_connected_components(self) -> List[Set[int]]:
        """
        Find SCCs using NetworkX
        SCCs represent equivalence classes (mutual implication)
        """
        if self._sccs is not None:
            return self._sccs

        self._sccs = [
            set(comp)
            for comp in nx.strongly_connected_components(self.graph)
        ]

        return self._sccs

    def transitive_reduction(self) -> 'DependencyGraph':
        """
        Compute transitive reduction (remove redundant edges)

        If a → b → c exists, remove direct edge a → c
        """
        reduced_graph = nx.transitive_reduction(self.graph)

        # Create new DependencyGraph with reduced edges
        result = DependencyGraph(list(self.constraints.values()))
        result.graph = reduced_graph

        return result

    def get_predecessors(self, node_id: int) -> Set[int]:
        """Get immediate predecessors of node"""
        return set(self.graph.predecessors(node_id))

    def get_successors(self, node_id: int) -> Set[int]:
        """Get immediate successors of node"""
        return set(self.graph.successors(node_id))

    def get_reachable(self, node_id: int) -> Set[int]:
        """Get all nodes reachable from node (transitive)"""
        closure = self.compute_transitive_closure()
        return closure[node_id]

    def has_path(self, source_id: int, target_id: int) -> bool:
        """Check if there's a path from source to target"""
        return nx.has_path(self.graph, source_id, target_id)

    def topological_sort(self) -> List[int]:
        """
        Return topological ordering of nodes
        Only valid on DAG (use after SCC condensation)
        """
        return list(nx.topological_sort(self.graph))

    def find_longest_path(self, start_id: int) -> List[int]:
        """
        Find longest path from start node
        Uses DAG longest path algorithm
        """
        # Condense SCCs to single nodes
        condensed = nx.condensation(self.graph)

        # Find longest path in condensed DAG
        longest = nx.dag_longest_path(condensed)

        # Map back to original nodes
        path = []
        for component_node in longest:
            component = condensed.nodes[component_node]['members']
            path.extend(sorted(component))

        return path
```

### 2.3 Implication Matrix

```python
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Set

class ImplicationMatrix:
    """
    Dense/sparse matrix representation of implications

    Properties:
        - M[i,j] = 1 if constraint i implies constraint j
        - Supports both dense and sparse storage
        - Efficient matrix operations
    """

    def __init__(
        self,
        constraints: List[Constraint],
        sparse: bool = True
    ):
        """
        Initialize matrix

        Args:
            constraints: List of constraints
            sparse: Use sparse matrix if True (recommended for large k)
        """
        self.constraints = constraints
        self.k = len(constraints)
        self.sparse = sparse

        # Initialize matrix (zeros = no implication)
        if sparse:
            self.matrix = csr_matrix((self.k, self.k), dtype=bool)
        else:
            self.matrix = np.zeros((self.k, self.k), dtype=bool)

        # Index mapping
        self.id_to_idx = {c.id: i for i, c in enumerate(constraints)}
        self.idx_to_id = {i: c.id for i, c in enumerate(constraints)}

    def set_implication(self, source_id: int, target_id: int, value: bool = True):
        """
        Set implication matrix entry
        """
        i = self.id_to_idx[source_id]
        j = self.id_to_idx[target_id]

        if self.sparse:
            # Convert to coo for efficient single element update
            self.matrix[i, j] = value
        else:
            self.matrix[i, j] = value

    def get_implication(self, source_id: int, target_id: int) -> bool:
        """
        Get implication matrix entry
        """
        i = self.id_to_idx[source_id]
        j = self.id_to_idx[target_id]
        return bool(self.matrix[i, j])

    def compute_transitive_closure(self):
        """
        Compute transitive closure using matrix multiplication

        M* = I + M + M² + M³ + ... (until fixed point)
        In Boolean semiring
        """
        # Add identity (reflexive closure)
        if self.sparse:
            closure = self.matrix + csr_matrix(np.eye(self.k))
        else:
            closure = self.matrix | np.eye(self.k, dtype=bool)

        # Iterate until fixed point (at most k iterations)
        for _ in range(self.k):
            if self.sparse:
                new_closure = closure @ closure | closure
            else:
                # Boolean matrix multiplication
                new_closure = closure.dot(closure) | closure

            # Check convergence
            if self.sparse:
                if (new_closure != closure).nnz == 0:
                    break
            else:
                if np.array_equal(new_closure, closure):
                    break

            closure = new_closure

        self.matrix = closure

    def get_redundant_constraints(self) -> Set[int]:
        """
        Find constraints implied by others

        A constraint is redundant if some other constraint implies it
        """
        redundant = set()

        for j in range(self.k):
            # Check if any constraint i ≠ j implies j
            column = self.matrix[:, j]
            if self.sparse:
                implied_by = column.nonzero()[0]
            else:
                implied_by = np.where(column)[0]

            # Check if any i ≠ j implies j
            for i in implied_by:
                if i != j:
                    redundant.add(self.idx_to_id[j])
                    break

        return redundant

    def find_antichain(self) -> Set[int]:
        """
        Find maximal antichain (set of mutually incomparable constraints)

        Uses Dilworth's theorem
        """
        # Convert to DAG (remove self-loops)
        if self.sparse:
            adj = self.matrix.copy()
        else:
            adj = self.matrix.copy()

        # Find largest antichain using minimum path cover
        # Implementation via bipartite matching
        # (Omitted for brevity, see NetworkX implementation)

        # Placeholder: return nodes with no in/out edges
        isolated = set()
        for i in range(self.k):
            in_edges = self.matrix[:, i].sum()
            out_edges = self.matrix[i, :].sum()
            if in_edges == 0 and out_edges == 0:
                isolated.add(self.idx_to_id[i])

        return isolated
```

### 2.4 Proof Tree

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class ProofNode:
    """
    Single step in reduction proof tree

    Each node represents one constraint elimination step
    """
    operation: str                          # Type of operation
    constraint_removed: int                 # ID of removed constraint
    justification: str                      # Human-readable justification
    implied_by: Optional[int] = None        # ID of implying constraint
    timestamp: float = field(default_factory=time.time)

    # For proof reconstruction
    children: List['ProofNode'] = field(factory=list)

    def to_lean4(self, context: 'ProofContext') -> str:
        """
        Convert proof node to Lean 4 proof term
        """
        match self.operation:
            case "subsumption":
                return self._subsumption_to_lean4(context)
            case "implication":
                return self._implication_to_lean4(context)
            case "transitive_reduction":
                return self._transitive_to_lean4(context)
            case _:
                raise ValueError(f"Unknown operation: {self.operation}")

    def _subsumption_to_lean4(self, context: 'ProofContext') -> str:
        """
        Generate Lean 4 for subsumption proof

        Example: c₁: x ≥ 10, c₂: x > 5
        Proof: c₁ ⊨ c₂ by monotonicity of >
        """
        c_removed = context.get_constraint(self.constraint_removed)
        c_implied = context.get_constraint(self.implied_by)

        return f"""
        have h_subsum_{self.constraint_removed} :
            {expr_to_lean4(c_implied.expr)} →
            {expr_to_lean4(c_removed.expr)}
        := by
            apply subsumption_lemma
            -- Proof that stronger implies weaker
            ...
        """

    def _implication_to_lean4(self, context: 'ProofContext') -> str:
        """
        Generate Lean 4 for implication proof
        """
        return f"""
        have h_imp_{self.constraint_removed} :
            (∧ C_reduced) → {expr_to_lean4(context.get_constraint(self.constraint_removed).expr)}
        := by
            intro h_conj
            -- Extract relevant constraints from conjunction
            have h₁ := h_conj.1  -- First relevant constraint
            have h₂ := h_conj.2  -- Second relevant constraint
            -- Derive implied constraint
            ...
        """

@dataclass
class ProofTree:
    """
    Complete proof tree for constraint reduction

    Represents the entire reduction process with justification
    """
    root: ProofNode
    original_constraints: Set[int]
    final_constraints: Set[int]
    reduction_steps: int = 0
    reduction_time: float = 0.0

    def add_step(self, node: ProofNode):
        """
        Add proof step to tree
        """
        node.children = [self.root]
        self.root = node
        self.reduction_steps += 1

    def to_lean4(self) -> str:
        """
        Convert entire proof tree to Lean 4 theorem
        """
        proof_body = self._generate_proof_body()

        return f"""
        theorem constraint_reduction :
            (∧ original_constraints) ↔ (∧ minimal_constraints) :=
        by
          constructor
          · -- Soundness: minimal ⊨ original
            {self._soundness_proof()}
          · -- Completeness: original ⊨ minimal
            {self._completeness_proof()}
        """

    def _soundness_proof(self) -> str:
        """Generate soundness proof (minimal ⊨ original)"""
        # Traverse tree, collecting subsumption/implication proofs
        return self.root.to_lean4(ProofContext())

    def _completeness_proof(self) -> str:
        """Generate completeness proof (original ⊨ minimal)"""
        # Trivial: minimal ⊆ original
        return """
        intro h_orig
        intro c hc
        -- c is in minimal set, which is subset of original
        exact h_orig c (by simp [hc])
        """
```

---

## 3. Component Architecture

### 3.1 Core Components

```python
# src/psi3/api/interface.py

from typing import List, Set
from dataclasses import dataclass

@dataclass
class PSI3Config:
    """
    Ψ₃ configuration options
    """
    # Algorithm mode
    mode: str = "standard"  # "fast", "standard", "aggressive"

    # Verification options
    verify: bool = True
    verification_level: str = "standard"  # "fast", "standard", "full"

    # Performance options
    parallel: bool = True
    num_workers: int = 4

    # SAT solver options
    sat_solver: str = "z3"  # "z3", "kissat", "cadical"
    sat_timeout: float = 10.0  # seconds

    # Reduction thresholds
    min_reduction_threshold: float = 1.5  # minimum 1.5x reduction
    target_reduction: float = 10.0  # target 10x reduction

@dataclass
class PSI3Result:
    """
    Ψ₃ result containing reduced constraints and proof
    """
    minimal_constraints: Set[Constraint]
    proof_tree: ProofTree
    equivalence_certificate: Optional['EquivalenceCertificate']

    # Metrics
    original_size: int
    final_size: int
    reduction_ratio: float
    runtime_seconds: float

    # Breakdown by stage
    stage1_time: float
    stage2_time: float
    stage3_time: float
    stage4_time: float

class PSI3Interface:
    """
    Public API for Ψ₃ constraint inversion
    """

    def __init__(self, config: PSI3Config = PSI3Config()):
        self.config = config
        self.sat_solver = self._init_sat_solver()
        self.lean4_interface = Lean4Interface()

    def reduce_constraints(
        self,
        constraints: List[Constraint],
        timeout: float = 300.0
    ) -> PSI3Result:
        """
        Main entry point: Reduce constraint set

        Args:
            constraints: Input constraint set
            timeout: Maximum runtime in seconds

        Returns:
            PSI3Result with minimal constraints and proof
        """
        start_time = time.time()

        # Stage 1: Syntactic preprocessing
        stage1_start = time.time()
        c1 = self._syntactic_preprocessing(constraints)
        stage1_time = time.time() - stage1_start

        # Stage 2: Dependency analysis
        stage2_start = time.time()
        graph = self._dependency_analysis(c1)
        stage2_time = time.time() - stage2_start

        # Stage 3: Minimal cover generation
        stage3_start = time.time()
        c_min = self._minimal_cover_generation(c1, graph)
        stage3_time = time.time() - stage3_start

        # Stage 4: Equivalence verification
        stage4_start = time.time()
        proof = self._verify_equivalence(constraints, c_min)
        stage4_time = time.time() - stage4_start

        total_time = time.time() - start_time

        # Build result
        return PSI3Result(
            minimal_constraints=c_min,
            proof_tree=proof.proof_tree,
            equivalence_certificate=proof,
            original_size=len(constraints),
            final_size=len(c_min),
            reduction_ratio=len(constraints) / len(c_min),
            runtime_seconds=total_time,
            stage1_time=stage1_time,
            stage2_time=stage2_time,
            stage3_time=stage3_time,
            stage4_time=stage4_time
        )
```

### 3.2 SAT Solver Interface

```python
# src/psi3/solvers/sat_interface.py

from z3 import *

class SatResult(Enum):
    SATISFIABLE = 1
    UNSATISFIABLE = 0
    UNKNOWN = -1

class SATInterface:
    """
    Interface to SAT/SMT solvers

    Supports multiple backends: Z3, Kissat, CaDiCaL
    """

    def __init__(self, solver_type: str = "z3", timeout: float = 10.0):
        self.solver_type = solver_type
        self.timeout = timeout
        self.solver = self._init_solver()

    def _init_solver(self):
        """Initialize solver backend"""
        match self.solver_type:
            case "z3":
                solver = Solver()
                solver.set("timeout", int(self.timeout * 1000))
                return solver
            case _:
                raise ValueError(f"Unknown solver: {self.solver_type}")

    def check_implication(
        self,
        antecedent: Expr,
        consequent: Expr
    ) -> bool:
        """
        Check if antecedent ⊨ consequent

        Method: Check UNSAT(antecedent ∧ ¬consequent)
        """
        # Build formula: antecedent ∧ ¬consequent
        negation = And(
            self._expr_to_z3(antecedent),
            Not(self._expr_to_z3(consequent))
        )

        # Query solver
        result = self.solver.check(negation)

        return result == unsat

    def check_equivalence(
        self,
        expr1: Expr,
        expr2: Expr
    ) -> bool:
        """
        Check if expr1 ≡ expr2 (mutual implication)
        """
        return (self.check_implication(expr1, expr2) and
                self.check_implication(expr2, expr1))

    def find_model(self, constraints: List[Expr]) -> Optional[Dict[str, Any]]:
        """
        Find satisfying assignment for constraints

        Returns None if unsatisfiable
        """
        self.solver.push()

        # Add constraints
        for c in constraints:
            self.solver.add(self._expr_to_z3(c))

        # Check satisfiability
        result = self.solver.check()

        if result == sat:
            model = self.solver.model()
            assignment = self._extract_assignment(model)
            self.solver.pop()
            return assignment
        else:
            self.solver.pop()
            return None

    def _expr_to_z3(self, expr: Expr) -> z3.ExprRef:
        """
        Convert internal Expr to Z3 expression
        """
        match expr:
            case BoolExpr(op=BoolOp.AND, args=args):
                return And(*[self._expr_to_z3(a) for a in args])
            case BoolExpr(op=BoolOp.OR, args=args):
                return Or(*[self._expr_to_z3(a) for a in args])
            case BoolExpr(op=BoolOp.NOT, args=[arg]):
                return Not(self._expr_to_z3(arg))
            case ArithExpr(op=ArithOp.LT, left=left, right=right):
                return self._expr_to_z3(left) < self._expr_to_z3(right)
            case ArithExpr(op=ArithOp.GE, left=left, right=right):
                return self._expr_to_z3(left) >= self._expr_to_z3(right)
            # ... more cases
            case _:
                raise ValueError(f"Unsupported expression: {expr}")

    def _extract_assignment(self, model: ModelRef) -> Dict[str, Any]:
        """Extract variable assignment from Z3 model"""
        assignment = {}
        for decl in model:
            name = decl.name()
            value = model[decl]
            assignment[name] = value
        return assignment
```

### 3.3 Lean 4 Interface

```python
# src/psi3/solvers/lean4_interface.py

import subprocess
from typing import Optional, Dict

class Lean4Interface:
    """
    Interface to Lean 4 proof assistant

    Handles:
        - Proof verification
        - Proof generation
        - Export/import of proof objects
    """

    def __init__(self, lean_executable: str = "lake"):
        self.lean_executable = lean_executable
        self.lean_dir = Path(__file__).parent.parent.parent / "lean4" / "PSI3"

    def verify_proof(self, lean_code: str) -> bool:
        """
        Verify Lean 4 proof

        Args:
            lean_code: Lean 4 code to verify

        Returns:
            True if proof verified, False otherwise
        """
        # Write to temporary file
        temp_file = self.lean_dir / "TempProof.lean"
        with open(temp_file, 'w') as f:
            f.write(lean_code)

        # Run Lean 4
        result = subprocess.run(
            [self.lean_executable, "build", str(temp_file)],
            cwd=self.lean_dir,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Clean up
        temp_file.unlink()

        return result.returncode == 0

    def generate_equivalence_proof(
        self,
        constraints_orig: Set[Constraint],
        constraints_min: Set[Constraint],
        proof_tree: ProofTree
    ) -> str:
        """
        Generate Lean 4 equivalence proof

        Returns:
            Lean 4 proof term as string
        """
        # Import PSI3 theory library
        header = f"""
        import PSI3.Basic
        import PSI3.Constraint
        import PSI3.Equivalence

        namespace PSI3
        """

        # Translate constraints to Lean 4
        lean_constraints_orig = [
            self._constraint_to_lean4(c) for c in constraints_orig
        ]
        lean_constraints_min = [
            self._constraint_to_lean4(c) for c in constraints_min
        ]

        # Build theorem statement
        theorem_stmt = f"""
        theorem reduction_equivalence :
            (∧ {' '.join(lean_constraints_orig)}) ↔
            (∧ {' '.join(lean_constraints_min)}) :=
        """

        # Build proof
        proof_body = proof_tree.to_lean4(ProofContext(constraints_orig, constraints_min))

        # Combine
        lean_code = header + theorem_stmt + proof_body

        return lean_code

    def _constraint_to_lean4(self, c: Constraint) -> str:
        """Convert constraint to Lean 4 syntax"""
        # Translate expression to Lean 4
        return self._expr_to_lean4(c.expr)

    def _expr_to_lean4(self, expr: Expr) -> str:
        """Convert expression to Lean 4 syntax"""
        match expr:
            case BoolExpr(op=BoolOp.AND, args=args):
                inner = ' '.join([self._expr_to_lean4(a) for a in args])
                return f"(And {inner})"
            case ArithExpr(op=ArithOp.GE, left=left, right=right):
                l = self._expr_to_lean4(left)
                r = self._expr_to_lean4(right)
                return f"(Ge {l} {r})"
            # ... more cases
            case _:
                raise ValueError(f"Unsupported: {expr}")
```

---

## 4. Integration with OpenEvolve

### 4.1 Ψ₁ Integration

```python
# src/psi3/api/psi1_integration.py

from psi1 import FormalSpecification

class PSI1Adapter:
    """
    Adapter for Ψ₁ (Problem Formalization) output

    Converts formal specification to constraint set
    """

    @staticmethod
    def from_psi1_output(spec: FormalSpecification) -> List[Constraint]:
        """
        Convert Ψ₁ output to Ψ₃ input

        Args:
            spec: Formal specification from Ψ₁

        Returns:
            List of constraints
        """
        constraints = []

        # Extract type constraints
        for type_constraint in spec.type_constraints:
            c = Constraint(
                id=len(constraints),
                expr=type_constraint.to_expr(),
                type=ConstraintType.TYPE,
                vars=type_constraint.free_vars,
                metadata=Metadata(
                    source="psi1",
                    priority=type_constraint.priority
                )
            )
            constraints.append(c)

        # Extract logical constraints
        for logical_constraint in spec.constraints:
            c = Constraint(
                id=len(constraints),
                expr=logical_constraint.formula,
                type=ConstraintType.BOOL,
                vars=logical_constraint.variables,
                metadata=Metadata(
                    source="psi1",
                    priority=logical_constraint.priority
                )
            )
            constraints.append(c)

        return constraints
```

### 4.2 Stage 2 Integration

```python
# src/psi3/api/stage2_integration.py

from stage2 import IsomorphicMapper

class PSI3ToStage2Adapter:
    """
    Adapter for Stage 2 (Isomorphic Mapping)

    Passes minimal constraints to Stage 2 for canonical mapping
    """

    def __init__(self):
        self.mapper = IsomorphicMapper()

    def export_to_stage2(
        self,
        psi3_result: PSI3Result
    ) -> Stage2Input:
        """
        Export Ψ₃ result to Stage 2 input format

        Args:
            psi3_result: Result from Ψ₃ reduction

        Returns:
            Stage2Input for isomorphic mapping
        """
        # Build Stage 2 input
        stage2_input = Stage2Input(
            constraints=psi3_result.minimal_constraints,
            equivalence_proof=psi3_result.equivalence_certificate,
            complexity_reduction=psi3_result.reduction_ratio,
            metadata={
                "original_size": psi3_result.original_size,
                "reduction_time": psi3_result.runtime_seconds,
                "verification_status": "verified" if psi3_result.equivalence_certificate else "unverified"
            }
        )

        return stage2_input

    def verify_stage2_compatibility(
        self,
        psi3_result: PSI3Result
    ) -> bool:
        """
        Verify that Ψ₃ output is compatible with Stage 2

        Checks:
            - All constraints have valid Lean 4 types
            - Equivalence proof is verifiable
            - No unsupported constraint types
        """
        # Check constraints
        for c in psi3_result.minimal_constraints:
            if not self._is_supported_constraint(c):
                return False

        # Check proof
        if psi3_result.equivalence_certificate:
            if not psi3_result.equivalence_certificate.verify():
                return False

        return True
```

### 4.3 Ψ₄ Integration

```python
# src/psi3/api/psi4_integration.py

from psi4 import SynthesisEngine

class PSI3ToPSI4Adapter:
    """
    Adapter for Ψ₄ (Synthesis Engine)

    Passes minimal constraints to synthesis engine
    """

    @staticmethod
    def export_to_psi4(
        psi3_result: PSI3Result
    ) -> PSI4Input:
        """
        Export Ψ₃ result to Ψ₄ input

        Synthesis engine benefits from reduced constraint set
        """
        # Extract optimization hints from reduction
        hints = PSI3ToPSI4Adapter._extract_hints(psi3_result)

        return PSI4Input(
            constraints=psi3_result.minimal_constraints,
            equivalence_proof=psi3_result.equivalence_certificate,
            optimization_hints=hints,
            complexity_metrics={
                "original_size": psi3_result.original_size,
                "reduced_size": psi3_result.final_size,
                "reduction_factor": psi3_result.reduction_ratio
            }
        )

    @staticmethod
    def _extract_hints(psi3_result: PSI3Result) -> List[Hint]:
        """
        Extract optimization hints from reduction process

        Hints help synthesis engine exploit structure
        """
        hints = []

        # Hint 1: Independent constraints (parallel synthesis)
        independent = PSI3ToPSI4Adapter._find_independent(psi3_result)
        if independent:
            hints.append(Hint(
                type="parallel_synthesis",
                constraints=independent
            ))

        # Hint 2: Constraint hierarchy (top-down synthesis)
        hierarchy = PSI3ToPSI4Adapter._extract_hierarchy(psi3_result)
        if hierarchy:
            hints.append(Hint(
                type="hierarchical_synthesis",
                hierarchy=hierarchy
            ))

        return hints
```

---

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Objectives**:
- Implement core data structures
- Build basic preprocessing pipeline
- Set up testing framework

**Deliverables**:
```python
# src/psi3/core/constraint.py
class Constraint:
    """Immutable constraint representation"""
    pass

# src/psi3/core/expression.py
class Expr:
    """Expression AST"""
    pass

# src/psi3/algorithms/preprocessing.py
def syntactic_preprocessing(constraints: List[Constraint]) -> List[Constraint]:
    """Stage 1: Syntactic redundancy elimination"""
    pass

# tests/unit/test_preprocessing.py
def test_remove_duplicates():
    """Test duplicate removal"""
    pass

def test_subsumption_detection():
    """Test subsumption detection"""
    pass
```

**Acceptance Criteria**:
- [ ] All data structures implemented with type hints
- [ ] Unit tests achieve 80%+ coverage
- [ ] Syntactic preprocessing passes basic tests
- [ ] CI/CD pipeline configured

### Phase 2: Dependency Analysis (Week 3-4)

**Objectives**:
- Integrate SAT solver (Z3)
- Build dependency graph
- Implement implication detection

**Deliverables**:
```python
# src/psi3/solvers/sat_interface.py
class SATInterface:
    """SAT solver interface"""
    pass

# src/psi3/algorithms/dependency.py
def build_dependency_graph(constraints: List[Constraint]) -> DependencyGraph:
    """Build implication graph"""
    pass

# tests/unit/test_dependency.py
def test_implication_detection():
    """Test implication checking"""
    pass

def test_dependency_graph():
    """Test graph construction"""
    pass
```

**Acceptance Criteria**:
- [ ] Z3 integration functional
- [ ] Dependency graph correctly identifies implications
- [ ] Transitive closure computed correctly
- [ ] Performance acceptable on medium problems (100-500 constraints)

### Phase 3: Minimal Cover (Week 5-6)

**Objectives**:
- Implement minimal cover algorithm
- Add optimization strategies
- Integrate with Stage 2

**Deliverables**:
```python
# src/psi3/algorithms/minimal_cover.py
def generate_minimal_cover(
    constraints: List[Constraint],
    graph: DependencyGraph
) -> List[Constraint]:
    """Generate minimal equivalent set"""
    pass

# src/psi3/api/stage2_integration.py
class PSI3ToStage2Adapter:
    """Stage 2 integration"""
    pass

# tests/integration/test_stage2_integration.py
def test_end_to_end_reduction():
    """Test full pipeline"""
    pass
```

**Acceptance Criteria**:
- [ ] Minimal cover achieves 3-5x reduction on structured problems
- [ ] Stage 2 integration functional
- [ ] Performance acceptable on large problems (1000+ constraints)

### Phase 4: Verification (Week 7-8)

**Objectives**:
- Integrate Lean 4
- Implement equivalence verification
- Comprehensive testing and benchmarking

**Deliverables**:
```python
# src/psi3/solvers/lean4_interface.py
class Lean4Interface:
    """Lean 4 interface"""
    pass

# src/psi3/algorithms/verification.py
def verify_equivalence(
    constraints_orig: List[Constraint],
    constraints_min: List[Constraint]
) -> EquivalenceCertificate:
    """Verify equivalence"""
    pass

# tests/benchmarks/
def bench_real_world_problems():
    """Benchmark on real-world constraint sets"""
    pass
```

**Acceptance Criteria**:
- [ ] Lean 4 integration functional
- [ ] Equivalence proofs generated and verified
- [ ] Benchmarks show 5-10x reduction on suitable problems
- [ ] All integration tests passing

---

## 6. Testing Strategy

### 6.1 Unit Testing

**Test Coverage Target**: 80%+

**Test Categories**:

1. **Constraint Tests**:
```python
def test_constraint_equality():
    """Test constraint equality"""
    c1 = Constraint(1, expr1, ConstraintType.BOOL, ...)
    c2 = Constraint(1, expr1, ConstraintType.BOOL, ...)
    assert c1 == c2

def test_constraint_subsumption():
    """Test subsumption detection"""
    c1 = Constraint(1, parse_expr("x ≥ 10"), ...)
    c2 = Constraint(2, parse_expr("x > 5"), ...)
    assert c1.subsumes(c2, solver)
```

2. **Preprocessing Tests**:
```python
def test_remove_duplicates():
    """Test duplicate removal"""
    constraints = [c1, c1, c2]
    reduced = syntactic_preprocessing(constraints)
    assert len(reduced) == 2

def test_subsumption_removal():
    """Test subsumption elimination"""
    constraints = [
        Constraint(1, parse_expr("x > 0"), ...),
        Constraint(2, parse_expr("x > 5"), ...),
        Constraint(3, parse_expr("x > 10"), ...)
    ]
    reduced = syntactic_preprocessing(constraints)
    assert len(reduced) == 1  # Only x > 10 remains
```

3. **Dependency Tests**:
```python
def test_implication_detection():
    """Test implication checking"""
    c1 = Constraint(1, parse_expr("x ≥ 10"), ...)
    c2 = Constraint(2, parse_expr("x > 5"), ...)
    assert check_implication(c1, c2, solver)  # c1 ⊨ c2

def test_transitive_closure():
    """Test transitive closure computation"""
    graph = DependencyGraph([c1, c2, c3])
    graph.add_implication(c1.id, c2.id)
    graph.add_implication(c2.id, c3.id)
    closure = graph.compute_transitive_closure()
    assert c3.id in closure[c1.id]  # c1 →* c3
```

4. **Minimal Cover Tests**:
```python
def test_minimal_cover_total_order():
    """Test minimal cover on total order (best case)"""
    constraints = [
        Constraint(i, parse_expr(f"x > {i}"), ...)
        for i in range(10)
    ]
    minimal = generate_minimal_cover(constraints, graph)
    assert len(minimal) == 1  # Only strongest constraint

def test_minimal_cover_partial_order():
    """Test minimal cover on partial order"""
    # Hierarchical constraints
    constraints = [
        Constraint(1, parse_expr("x > 0"), ...),
        Constraint(2, parse_expr("x > 5"), ...),
        Constraint(3, parse_expr("y < 100"), ...)
    ]
    minimal = generate_minimal_cover(constraints, graph)
    assert len(minimal) == 2  # {x > 0, y < 100}
```

### 6.2 Integration Testing

**End-to-End Pipeline**:
```python
def test_full_pipeline():
    """Test complete Ψ₃ pipeline"""
    # Input: Database query constraints
    constraints = [
        Constraint(1, parse_expr("age > 18"), ...),
        Constraint(2, parse_expr("age > 21"), ...),
        Constraint(3, parse_expr("income ≥ 50000"), ...),
        Constraint(4, parse_expr("age > 21 ∧ income ≥ 50000"), ...)
    ]

    # Run Ψ₃
    result = psi3_interface.reduce_constraints(constraints)

    # Verify reduction
    assert len(result.minimal_constraints) == 2
    assert result.reduction_ratio >= 2.0

    # Verify equivalence
    assert result.equivalence_certificate.verify()
```

**Stage 2 Integration**:
```python
def test_stage2_integration():
    """Test integration with Stage 2"""
    # Run Ψ₃
    psi3_result = psi3_interface.reduce_constraints(constraints)

    # Export to Stage 2
    stage2_input = stage2_adapter.export_to_stage2(psi3_result)

    # Verify compatibility
    assert stage2_adapter.verify_stage2_compatibility(psi3_result)

    # Run Stage 2
    stage2_result = stage2_mapper.map_to_canonical(stage2_input)
    assert stage2_result is not None
```

### 6.3 Property-Based Testing

**Using Hypothesis**:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(min_value=0, max_value=100), min_size=0, max_size=20))
def test_reduction_preserves_satisfiability(bound_values):
    """
    Property: Reduction preserves satisfiability

    If original constraints satisfiable, minimal also satisfiable
    """
    # Generate constraints: x > bound for each bound
    constraints = [
        Constraint(i, parse_expr(f"x > {bound}"), ...)
        for i, bound in enumerate(bound_values)
    ]

    # Run Ψ₃
    result = psi3_interface.reduce_constraints(constraints)

    # Check satisfiability preserved
    orig_sat = check_satisfiability(constraints)
    min_sat = check_satisfiability(result.minimal_constraints)

    assert orig_sat == min_sat

@given(st.lists(st.integers(min_value=0, max_value=50), min_size=10, max_size=50))
def test_monotonic_complexity_reduction(bound_values):
    """
    Property: Complexity reduction monotonic

    More input constraints → at least proportional reduction
    """
    constraints = [
        Constraint(i, parse_expr(f"x > {bound}"), ...)
        for i, bound in enumerate(bound_values)
    ]

    result = psi3_interface.reduce_constraints(constraints)

    # Should achieve some reduction (unless antichain)
    if len(constraints) > 10:
        assert len(result.minimal_constraints) <= len(constraints)
```

### 6.4 Benchmarking

**Benchmark Suite**:
```python
# tests/benchmarks/bench_reduction.py

@pytest.mark.benchmark
def test_database_query_reduction(benchmark):
    """Benchmark: Database query constraint reduction"""
    # Typical SQL WHERE clause with 20 conditions
    constraints = generate_database_query_constraints(20)

    result = benchmark(psi3_interface.reduce_constraints, constraints)

    # Expected: 3-5x reduction
    assert result.reduction_ratio >= 3.0

@pytest.mark.benchmark
def test_type_constraint_reduction(benchmark):
    """Benchmark: Type constraint hierarchy reduction"""
    # Type hierarchy with 30 constraints
    constraints = generate_type_constraints(30)

    result = benchmark(psi3_interface.reduce_constraints, constraints)

    # Expected: 5-10x reduction
    assert result.reduction_ratio >= 5.0

@pytest.mark.benchmark
def test_real_world_config(benchmark):
    """Benchmark: Real-world configuration problem"""
    # Software feature model with 100 constraints
    constraints = load_feature_model("linux_kernel")

    result = benchmark(psi3_interface.reduce_constraints, constraints)

    # Expected: 8-15x reduction
    assert result.reduction_ratio >= 8.0
```

---

## 7. Performance Optimization

### 7.1 Parallelization Strategy

**Parallel Implication Checking**:
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_check_implications(
    constraints: List[Constraint],
    num_workers: int = 4
) -> ImplicationMatrix:
    """
    Check all implication pairs in parallel
    """
    k = len(constraints)
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            lambda pair: check_implication_pair(
                constraints[pair[0]],
                constraints[pair[1]]
            ),
            pairs
        )

    # Build matrix from results
    matrix = ImplicationMatrix(constraints)
    for (i, j), implies in zip(pairs, results):
        if implies:
            matrix.set_implication(i, j, True)

    return matrix
```

### 7.2 Caching Strategy

**Implication Cache**:
```python
from functools import lru_cache

class ImplicationCache:
    """
    LRU cache for implication checks
    """
    def __init__(self, max_size: int = 10000):
        self.cache = lru_cache(maxsize=max_size)(self._check_uncached)

    def check_implication(self, c1: Constraint, c2: Constraint) -> bool:
        """Check implication with caching"""
        key = (c1.id, c2.id)
        return self.cache(key)

    def _check_uncached(self, key: Tuple[int, int]) -> bool:
        """Actual implication check (uncached)"""
        # Implementation
        pass
```

### 7.3 Incremental Updates

**Dynamic Constraint Sets**:
```python
class IncrementalPSI3:
    """
    Incremental version for dynamic constraint sets
    """
    def __init__(self):
        self.current_set: Set[Constraint] = set()
        self.dependency_graph: Optional[DependencyGraph] = None
        self.implication_cache: ImplicationCache = ImplicationCache()

    def add_constraint(self, c: Constraint):
        """Add constraint incrementally"""
        # Check against existing constraints
        for existing in self.current_set:
            if self.implication_cache.check_implication(c, existing):
                # c implies existing, remove existing
                self.current_set.remove(existing)
                self._invalidate_graph()
            elif self.implication_cache.check_implication(existing, c):
                # existing implies c, don't add c
                return

        self.current_set.add(c)
        self._update_graph(c)

    def remove_constraint(self, c: Constraint):
        """Remove constraint incrementally"""
        if c in self.current_set:
            self.current_set.remove(c)
            self._invalidate_graph()

    def get_minimal_set(self) -> Set[Constraint]:
        """Get current minimal set"""
        return self.current_set.copy()
```

### 7.4 Adaptive Algorithm Selection

**Strategy Selection Based on Problem Characteristics**:
```python
def adaptive_psi3(constraints: List[Constraint]) -> PSI3Result:
    """
    Adaptively select algorithm based on problem structure
    """
    # Analyze constraint structure
    analysis = analyze_problem_structure(constraints)

    if analysis.is_total_order():
        # Best case: Use O(k) algorithm
        return reduce_total_order(constraints)

    elif analysis.treewidth < 5:
        # Low treewidth: Use tree decomposition
        return reduce_with_decomposition(constraints, analysis.treewidth)

    elif analysis.redundancy > 0.7:
        # High redundancy: Use aggressive reduction
        return aggressive_reduction(constraints)

    else:
        # Default: Standard algorithm
        return standard_psi3(constraints)

def analyze_problem_structure(constraints: List[Constraint]) -> ProblemAnalysis:
    """
    Analyze problem structure for adaptive selection
    """
    # Sample constraints to estimate structure
    sample_size = min(100, len(constraints))
    sample = constraints[:sample_size]

    # Estimate redundancy
    redundancy = estimate_redundancy(sample)

    # Estimate treewidth
    treewidth = estimate_treewidth(sample)

    # Check for total order
    is_total = check_total_order(sample)

    return ProblemAnalysis(
        redundancy=redundancy,
        treewidth=treewidth,
        is_total_order=is_total
    )
```

---

## 8. Risk Mitigation

### 8.1 Technical Risks

**Risk 1**: Minimal cover computation is NP-hard
**Mitigation**:
- Use polynomial-time approximation (greedy algorithm)
- Accept 1-10% suboptimality
- Fallback to heuristic methods on timeout

**Risk 2**: Equivalence verification expensive
**Mitigation**:
- Use random testing for fast validation
- Formal verification only on critical problems
- Cache verification results

**Risk 3**: No reduction on unstructured problems
**Mitigation**:
- Detect unstructured problems early (quick test)
- Skip Ψ₃ if redundancy < threshold
- Return original constraints with explanation

### 8.2 Integration Risks

**Risk 4**: Ψ₃ output incompatible with Stage 2
**Mitigation**:
- Define interface contract early
- Validate compatibility in CI/CD
- Provide adapter layer if needed

**Risk 5**: Performance overhead > benefit
**Mitigation**:
- Benchmark realistic workloads
- Adaptive activation (only when beneficial)
- Performance profiling and optimization

### 8.3 Validation Risks

**Risk 6**: Benchmark not representative
**Mitigation**:
- Use diverse test suite
- Include real-world problems
- Continuously add new test cases

---

## 9. Success Criteria

### 9.1 Functional Requirements

- [ ] Ψ₃ achieves ≥10x reduction on ≥60% of structured problems
- [ ] Ψ₃ achieves ≥5x reduction on ≥80% of structured problems
- [ ] Equivalence verified by Lean 4 on all reduced sets
- [ ] Integration with Stage 2 functional
- [ ] Integration with Ψ₁ and Ψ₄ functional

### 9.2 Performance Requirements

- [ ] Runtime overhead ≤10x on large problems (1000+ constraints)
- [ ] Memory usage ≤2x input size
- [ ] Stage 1 (preprocessing): <1 second for 1000 constraints
- [ ] Stage 2 (dependency): <10 seconds for 1000 constraints
- [ ] Stage 3 (minimal cover): <5 seconds for 1000 constraints
- [ ] Stage 4 (verification): <30 seconds for 1000 constraints

### 9.3 Quality Requirements

- [ ] Unit test coverage ≥80%
- [ ] All integration tests passing
- [ ] Benchmarks meet performance targets
- [ ] Code review approved
- [ ] Documentation complete

---

## 10. Next Steps

1. **Week 1**: Set up project structure, implement core data structures
2. **Week 2**: Implement syntactic preprocessing, unit tests
3. **Week 3-4**: Integrate SAT solver, implement dependency analysis
4. **Week 5-6**: Implement minimal cover, Stage 2 integration
5. **Week 7-8**: Integrate Lean 4, comprehensive testing, optimization

**Next Document**: `psi3_validation_strategy.md`
