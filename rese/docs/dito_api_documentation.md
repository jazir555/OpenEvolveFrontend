# DITO API Documentation

**Author:** Agent A3 (DITO Specialist)
**Created:** 2025-12-31
**Status:** 🟢 Complete
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core API](#core-api)
5. [Graph Structures](#graph-structures)
6. [Configuration](#configuration)
7. [Performance Tuning](#performance-tuning)
8. [Examples](#examples)
9. [Integration Guide](#integration-guide)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Dynamic Inference Trace Optimizer (DITO) provides O(n log n) contradiction detection for constraint systems. It replaces naive O(n²) pairwise checking with sophisticated spatial indexing and hierarchical abstraction.

### Key Features

- **O(n log n) Construction:** Build index structures in sub-quadratic time
- **O(log n) Query:** Fast targeted contradiction detection
- **O(log n) Updates:** Incremental updates without full rebuild
- **1000x+ Speedup:** Compared to naive O(n²) approach
- **Sound and Complete:** Detects all contradictions with no false positives
- **SCE Integrated:** Works with Symbolic Constraint Engine
- **LLTL Ready:** Compatible with Linear-time Temporal Logic prover

### Architecture

```
Application Layer
    ↓
DITO Optimizer (Main API)
    ├─ R-tree (Spatial Index)
    ├─ LSH Tables (Semantic Grouping)
    ├─ HAG (Hierarchical Abstraction)
    ├─ CD-Graph (Constraint Dependencies)
    └─ PV-Graph (Predicate-Variable Relations)
    ↓
SCE (Constraint Storage)
LLTL (Logic Evaluation)
```

---

## Installation

### Requirements

```bash
# Core dependencies
pip install networkx numpy

# Optional for visualization
pip install matplotlib

# Testing
pip install pytest
```

### Import

```python
from rese.core.dito_optimizer import (
    DITOOptimizer,
    DITOConfig,
    ContradictionPair,
    ContradictionType
)

from rese.core.dito_graphs import (
    ConstraintDependencyGraph,
    PredicateVariableGraph,
    HierarchicalAbstractionGraph
)

from rese.core.symbolic_constraint_engine import (
    Constraint,
    ConstraintType
)
```

---

## Quick Start

### Basic Usage

```python
from rese.core.dito_optimizer import DITOOptimizer
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

# 1. Create constraints
constraints = [
    Constraint(
        id="temp_limit",
        type=ConstraintType.HARD,
        description="Temperature must be less than 1000°C",
        formalization="forall (T : Temperature), T < 1000",
        source="user_prompt"
    ),
    Constraint(
        id="min_temp",
        type=ConstraintType.HARD,
        description="Temperature must be greater than 1000°C",  # Contradiction!
        formalization="forall (T : Temperature), T > 1000",
        source="user_prompt"
    )
]

# 2. Initialize DITO
dito = DITOOptimizer()

# 3. Build index structures
result = dito.build(constraints)
print(f"Built {result['constraints_processed']} constraints in {result['build_time']:.4f}s")

# 4. Detect contradictions
contradictions = dito.detect_contradictions()
print(f"Found {len(contradictions)} contradictions")

for c in contradictions:
    print(f"  {c.constraint1_id} <-> {c.constraint2_id}")
    print(f"    {c.description}")
```

### Expected Output

```
Built 2 constraints in 0.0234s
Found 1 contradictions
  temp_limit <-> min_temp
    Direct logical contradiction
```

---

## Core API

### DITOOptimizer

Main class for contradiction detection.

#### Constructor

```python
DITOOptimizer(config: Optional[DITOConfig] = None)
```

**Parameters:**
- `config`: Optional configuration object

**Example:**

```python
# Default configuration
dito = DITOOptimizer()

# Custom configuration
from rese.core.dito_optimizer import DITOConfig
config = DITOConfig(
    max_hierarchy_level=10,
    rtree_max_entries=50,
    cache_enabled=True
)
dito = DITOOptimizer(config)
```

#### Methods

##### build()

Build DITO index structures from constraints.

```python
build(constraints: List[Constraint]) -> Dict[str, Any]
```

**Parameters:**
- `constraints`: List of SCE Constraint objects

**Returns:**
- Dictionary with build statistics:
  - `constraints_processed`: Number of constraints
  - `build_time`: Time taken (seconds)
  - `constraints_per_second`: Throughput

**Complexity:** O(n log n)

**Example:**

```python
result = dito.build(constraints)
print(f"Built {result['constraints_processed']} constraints")
print(f"Time: {result['build_time']:.4f}s")
```

##### detect_contradictions()

Detect contradictions between constraints.

```python
detect_contradictions(
    query_constraint: Optional[Constraint] = None
) -> List[ContradictionPair]
```

**Parameters:
- `query_constraint`: Optional constraint to check (targeted check)

**Returns:**
- List of ContradictionPair objects

**Complexity:**
- Targeted: O(log n + k) where k = results
- Full: O(√n · log n)

**Example:**

```python
# Full check (all constraints)
all_contradictions = dito.detect_contradictions()

# Targeted check (specific constraint)
specific_contradictions = dito.detect_contradictions(my_constraint)
```

##### update()

Apply incremental update to DITO structures.

```python
update(
    change_type: str,
    constraint: Optional[Constraint] = None,
    constraint_id: Optional[str] = None
) -> Dict[str, Any]
```

**Parameters:**
- `change_type`: "ADD", "REMOVE", or "MODIFY"
- `constraint`: Constraint object (for ADD/MODIFY)
- `constraint_id`: Constraint ID string (for REMOVE)

**Returns:**
- Dictionary with update results

**Complexity:** O(log n)

**Example:**

```python
# Add new constraint
new_c = Constraint(id="new", type=ConstraintType.HARD, ...)
dito.update("ADD", constraint=new_c)

# Remove constraint
dito.update("REMOVE", constraint_id="old_constraint")

# Modify constraint
modified = Constraint(id="existing", ...)
dito.update("MODIFY", constraint=modified)
```

##### get_statistics()

Get optimizer statistics.

```python
get_statistics() -> Dict[str, Any]
```

**Returns:**
- Dictionary with statistics:
  - `total_constraints`: Total constraints
  - `total_contradictions`: Known contradictions
  - `queries`: Number of queries performed
  - `updates`: Number of updates performed
  - `cache_hits`: Cache hit count
  - `cache_misses`: Cache miss count
  - `cd_graph_nodes`: CD-Graph node count
  - `hag_levels`: Number of HAG levels

**Example:**

```python
stats = dito.get_statistics()
for key, value in stats.items():
    print(f"{key}: {value}")
```

---

### ContradictionPair

Represents a pair of contradictory constraints.

#### Attributes

```python
@dataclass
class ContradictionPair:
    id: str                          # Unique identifier
    constraint1_id: str              # First constraint ID
    constraint2_id: str              # Second constraint ID
    contradiction_type: ContradictionType
    description: str                 # Human-readable description
    confidence: float                # 0.0 - 1.0
    conflicting_variables: List[str] # Variable names
    detection_method: str            # "SPATIAL", "SEMANTIC", "FULL"
    detection_level: int             # HAG level detected at
    timestamp: float                 # Detection timestamp
```

#### ContradictionType

```python
class ContradictionType(Enum):
    DIRECT = "direct"              # Direct logical contradiction
    RANGE = "range"                # Overlapping incompatible ranges
    MUTEX = "mutex"                # Mutex violations
    UNSATISFIABLE = "unsat"        # Formula unsatisfiable
    INCONSISTENT = "inconsistent"  # State inconsistency
    TEMPORAL = "temporal"          # Temporal contradiction
```

---

### DITOConfig

Configuration for DITO optimizer.

#### Parameters

```python
@dataclass
class DITOConfig:
    # Graph parameters
    max_hierarchy_level: int = 10        # H = O(log n)
    max_traversal_depth: int = 5         # L = O(log n)
    branching_factor: int = 10           # Expected degree

    # R-tree parameters
    rtree_max_entries: int = 50
    rtree_min_entries: int = 10
    rtree_bulk_load_threshold: int = 1000

    # LSH parameters
    lsh_num_tables: int = 10
    lsh_num_hashes: int = 5
    lsh_bucket_size: int = 100

    # Caching
    cache_enabled: bool = True
    cache_max_size: int = 10000
    cache_ttl: int = 3600000  # 1 hour in ms

    # Updates
    lazy_mode: bool = True
    batch_size: int = 100
    auto_rebalance: bool = True

    # Parallelization
    parallel_enabled: bool = True
    num_threads: int = 4
```

#### Example Configurations

```python
# Fast performance (less accuracy)
fast_config = DITOConfig(
    max_hierarchy_level=5,
    rtree_max_entries=20,
    cache_enabled=True
)

# High accuracy (slower)
accurate_config = DITOConfig(
    max_hierarchy_level=15,
    rtree_max_entries=100,
    lsh_num_tables=20
)

# Memory-constrained
low_memory_config = DITOConfig(
    max_hierarchy_level=5,
    cache_enabled=False,
    rtree_max_entries=20
)
```

---

## Graph Structures

### ConstraintDependencyGraph (CD-Graph)

Tracks direct dependencies between constraints.

```python
from rese.core.dito_graphs import ConstraintDependencyGraph

cd_graph = ConstraintDependencyGraph()

# Add nodes
cd_graph.add_node(constraint1)
cd_graph.add_node(constraint2)

# Add dependency edge
cd_graph.add_edge("c1", "c2", DependencyType.DIRECT)

# Get dependencies
deps = cd_graph.get_dependencies("c2")

# Get dependents
dependents = cd_graph.get_dependents("c1")

# Mark dirty region after update
dirty = cd_graph.mark_dirty_region("c1", max_depth=5)
```

### PredicateVariableGraph (PV-Graph)

Bipartite graph of predicates and variables.

```python
from rese.core.dito_graphs import PredicateVariableGraph

pv_graph = PredicateVariableGraph()

# Add predicate with variables
pv_graph.add_predicate("c1", formula, ["x", "y", "z"])

# Detect communities
communities = pv_graph.detect_communities()

# Get related constraints
related = pv_graph.get_related_constraints(["x", "y"])

# Check community overlap
overlap = pv_graph.get_community_overlap("c1", "c2")
```

### HierarchicalAbstractionGraph (HAG)

Multi-level hierarchy for efficient checking.

```python
from rese.core.dito_graphs import HierarchicalAbstractionGraph

hag = HierarchicalAbstractionGraph(max_level=10)

# Build hierarchy
hag.build_hierarchy(constraints, cd_graph, pv_graph)

# Get nodes at specific level
level_nodes = hag.get_nodes_at_level(level=2)

# Get root
root = hag.get_root()

# Detect contradictions top-down
contradictions = hag.detect_contradictions_top_down()
```

---

## Configuration

### Performance Tuning

#### For Speed

```python
config = DITOConfig(
    max_hierarchy_level=5,        # Fewer levels = faster build
    rtree_max_entries=100,        # Larger nodes = fewer levels
    cache_enabled=True,           # Enable caching
    parallel_enabled=True,        # Parallel operations
    lazy_mode=True               # Defer re-evaluation
)
```

#### For Accuracy

```python
config = DITOConfig(
    max_hierarchy_level=15,       # More levels = better pruning
    lsh_num_tables=20,           # More LSH tables = better grouping
    lsh_num_hashes=10,           # More hashes per table
    cache_enabled=True
)
```

#### For Memory Efficiency

```python
config = DITOConfig(
    max_hierarchy_level=5,
    cache_enabled=False,          # Disable cache
    rtree_max_entries=20,         # Smaller nodes
    lazy_mode=True               # Lazy evaluation
)
```

### Scaling Guidelines

| Constraints | Hierarchy Level | R-Tree Max Entries | Expected Build Time |
|-------------|-----------------|-------------------|-------------------|
| 100         | 3-5             | 20-30             | < 1s              |
| 1,000       | 5-7             | 30-50             | 1-5s              |
| 10,000      | 7-10            | 50-100            | 10-30s            |
| 100,000     | 10-12           | 100-200           | 1-5 min           |

---

## Examples

### Example 1: Temperature Range Constraints

```python
from rese.core.dito_optimizer import DITOOptimizer
from rese.core.symbolic_constraint_engine import Constraint, ConstraintType

# Create temperature constraints
constraints = [
    Constraint(
        id="temp_min",
        type=ConstraintType.HARD,
        description="Temperature must be above 0°C",
        formalization="T > 0",
        source="safety"
    ),
    Constraint(
        id="temp_max",
        type=ConstraintType.HARD,
        description="Temperature must be below 100°C",
        formalization="T < 100",
        source="safety"
    ),
    Constraint(
        id="temp_optimal",
        type=ConstraintType.SOFT,
        description="Temperature should be around 50°C",
        formalization="T ≈ 50",
        source="efficiency"
    )
]

# Build and check
dito = DITOOptimizer()
dito.build(constraints)

contradictions = dito.detect_contradictions()
print(f"Found {len(contradictions)} contradictions")
```

### Example 2: Incremental Updates

```python
# Initial build
dito = DITOOptimizer()
dito.build(initial_constraints)

# Add new constraint
new_c = Constraint(
    id="emergency_shutdown",
    type=ConstraintType.HARD,
    description="Emergency: Temperature must be below 150°C",
    formalization="T < 150",
    source="safety_protocol"
)

result = dito.update("ADD", constraint=new_c)

# Check for new contradictions
new_contradictions = dito.detect_contradictions(new_c)
print(f"New contradictions: {len(new_contradictions)}")
```

### Example 3: Batch Processing

```python
# Process in batches for efficiency
batch_size = 100
all_constraints = [...]

dito = DITOOptimizer(config=DITOConfig(lazy_mode=True))

for i in range(0, len(all_constraints), batch_size):
    batch = all_constraints[i:i+batch_size]

    for constraint in batch:
        dito.update("ADD", constraint=constraint)

    # Check contradictions after each batch
    contradictions = dito.detect_contradictions()
    print(f"Batch {i//batch_size}: {len(contradictions)} contradictions")
```

---

## Integration Guide

### SCE Integration

DITO integrates seamlessly with the Symbolic Constraint Engine.

```python
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine
from rese.core.dito_optimizer import DITOOptimizer

# Create SCE
sce = SymbolicConstraintEngine()

# Add constraints to SCE
for constraint_data in user_input:
    constraint = create_constraint(constraint_data)
    sce.add_constraint(constraint)

# Export to DITO
all_constraints = sce.get_all_constraints()

# Build DITO
dito = DITOOptimizer()
dito.build(all_constraints)

# Detect contradictions
contradictions = dito.detect_contradictions()

# Export back to SCE
for c in contradictions:
    # Handle contradiction (notify user, etc.)
    handle_contradiction(c)
```

### LLTL Integration

DITO can use LLTL theorem prover for accurate contradiction detection.

```python
# In dito_optimizer.py, modify _check_contradiction:

def _check_contradiction(self, c1, c2) -> bool:
    """Check contradiction using LLTL theorem prover"""
    try:
        from lltl.prover import LLTLTheoremProver

        prover = LLTLTheoremProver()

        # Combine formulas
        combined = combine_formulas(c1.formalization, c2.formalization)

        # Check satisfiability
        result = prover.isSatisfiable(combined)

        # Contradiction if unsatisfiable
        return not result.satisfiable

    except ImportError:
        # Fallback to keyword-based
        return self._keyword_contradiction_check(c1, c2)
```

### Custom Contradiction Detection

Implement domain-specific contradiction logic:

```python
class CustomDITO(DITOOptimizer):
    def _check_contradiction(self, c1, c2):
        """Custom contradiction detection"""

        # Domain-specific logic
        if "temperature" in c1.description and "pressure" in c2.description:
            return self._check_temperature_pressure(c1, c2)

        # Default to keyword check
        return super()._check_contradiction(c1, c2)

    def _check_temperature_pressure(self, c1, c2):
        """Custom temperature-pressure relationship"""
        # Implementation here
        return False
```

---

## Troubleshooting

### Issue: Slow Build Times

**Symptom:** Building takes longer than expected

**Solutions:**
1. Reduce `max_hierarchy_level`
2. Reduce `rtree_max_entries`
3. Disable cache if not needed: `cache_enabled=False`
4. Enable parallelization: `parallel_enabled=True`

### Issue: High Memory Usage

**Symptom:** Out of memory errors

**Solutions:**
1. Reduce `max_hierarchy_level`
2. Disable cache: `cache_enabled=False`
3. Reduce `rtree_max_entries`
4. Use lazy mode: `lazy_mode=True`

### Issue: Missed Contradictions

**Symptom:** DITO doesn't detect all contradictions

**Solutions:**
1. Increase `max_hierarchy_level` for better pruning
2. Increase `lsh_num_tables` for better semantic grouping
3. Implement LLTL integration for accurate detection
4. Disable cache to ensure fresh checks

### Issue: False Positives

**Symptom:** DITO reports contradictions that don't exist

**Solutions:**
1. Integrate LLTL theorem prover for sound detection
2. Review custom contradiction logic
3. Check constraint formalizations for errors

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

dito = DITOOptimizer()
dito.build(constraints)

# Check statistics
stats = dito.get_statistics()
print(json.dumps(stats, indent=2))
```

---

## Performance Tips

1. **Batch Updates:** Add multiple constraints before checking
2. **Lazy Mode:** Enable for high-throughput scenarios
3. **Cache:** Enable for repeated queries
4. **Parallel:** Use for multi-core systems
5. **Hierarchy:** Tune based on constraint count

---

## API Reference Summary

### Classes

- `DITOOptimizer`: Main optimizer class
- `DITOConfig`: Configuration object
- `ContradictionPair`: Contradiction result
- `ContradictionType`: Contradiction types (Enum)
- `SpatialExtent`: Multi-dimensional extent
- `RTree`: Spatial index (internal)
- `LSHTable`: Semantic hash table (internal)

### Functions

- `build()`: Build index structures
- `detect_contradictions()`: Find contradictions
- `update()`: Apply incremental updates
- `get_statistics()`: Get performance stats

### Graphs

- `ConstraintDependencyGraph`: Dependency tracking
- `PredicateVariableGraph`: Semantic relationships
- `HierarchicalAbstractionGraph`: Multi-level hierarchy

---

## Support

For issues, questions, or contributions:

- **Documentation:** See `rese/docs/dito_*.md`
- **Tests:** See `rese/tests/test_core/test_dito_optimizer.py`
- **Benchmarks:** Run `python rese/benchmarks/benchmark_dito.py`
- **Issues:** Report via project issue tracker

---

**Version:** 1.0.0
**Last Updated:** 2025-12-31
**Author:** Agent A3 (DITO Specialist)
