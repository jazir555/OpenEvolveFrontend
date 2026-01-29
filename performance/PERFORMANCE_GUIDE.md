# RESE Performance Tuning Guide

## Overview

This guide provides comprehensive performance optimization recommendations for all RESE components to meet production performance targets.

## Performance Targets

| Component | Target | Input Scale |
|-----------|--------|-------------|
| SCE | <1s | 10K constraints |
| DITO | <10s | 100K constraints |
| Phi1.5 | <10s | 1K failures |
| I_mech | <30s | Domain pairs |
| Gamma1 | <5s | ACI calculation |
| MCTS | <60s | 1000 iterations |

## Component-Specific Optimizations

### 1. Symbolic Constraint Engine (SCE)

#### Bottlenecks Identified
- O(n²) contradiction detection
- Full cache invalidation on updates
- Sequential dependency graph operations

#### Optimizations Implemented
1. **Parallel Conflict Detection**
   - Multi-threaded pairwise checking
   - Chunked processing for load balancing
   - Target: 4-8x speedup on multi-core

2. **Incremental Cache Invalidation**
   - Only invalidate affected constraint pairs
   - LRU cache with automatic eviction
   - Cache hit rate target: >80%

3. **Memoized Dependency Queries**
   - Cache dependency lists
   - Batch graph operations
   - Lazy evaluation

#### Tuning Parameters
```python
from rese.performance.sce_optimizer import SCEOptimizer

# Configuration for optimal performance
sce = SCEOptimizer(
    use_parallel=True,      # Enable parallel processing
    num_threads=4,          # Match CPU cores
)

# Expected performance:
# - 10K constraints: <1s
# - 100K constraints: <10s
```

#### Monitoring
```python
stats = sce.get_statistics()
print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2%}")
print(f"Avg add time: {stats['avg_add_time']*1000:.2f}ms")
```

---

### 2. Dynamic Inference Trace Optimizer (DITO)

#### Bottlenecks Identified
- O(n²) spatial queries without indexing
- Expensive LSH recomputation
- Hierarchical rebuilding cost

#### Optimizations Implemented
1. **R-tree Spatial Indexing**
   - O(log n) overlap queries
   - Bulk loading for initial construction
   - R*-tree split heuristics

2. **LSH Semantic Grouping**
   - MinHash signatures for fast similarity
   - Multiple hash tables for recall
   - Configurable bucket size

3. **Lazy Hierarchy Updates**
   - Batch updates with rebalancing
   - Incremental tree maintenance
   - Deferred contradiction checks

#### Tuning Parameters
```python
from rese.core.dito_optimizer import DITOConfig, DITOOptimizer

config = DITOConfig(
    # R-tree parameters
    rtree_max_entries=50,
    rtree_min_entries=10,
    rtree_bulk_load_threshold=1000,

    # LSH parameters
    lsh_num_tables=10,
    lsh_num_hashes=5,
    lsh_bucket_size=100,

    # Parallelization
    parallel_enabled=True,
    num_threads=4,

    # Caching
    cache_enabled=True,
    cache_max_size=10000,
)

dito = DITOOptimizer(config)

# Expected performance:
# - Build 100K constraints: <5s
# - Detect contradictions: <5s
# - Total: <10s
```

---

### 3. Phi1.5 Tacit Assumption Miner

#### Bottlenecks Identified
- O(n²) clustering distance calculations
- Repeated feature extraction
- Expensive anomaly detection

#### Optimizations Implemented
1. **Optimized Clustering**
   - Use scikit-learn's optimized implementations
   - Precompute distance matrices
   - Mini-batch clustering for large datasets

2. **Feature Caching**
   - Cache extracted feature vectors
   - Reuse for multiple clusterings
   - Incremental updates

3. **Vectorized Operations**
   - NumPy vectorization for distance calculations
   - Batch processing for feature extraction
   - Parallel anomaly detection

#### Tuning Parameters
```python
from rese.phase1.tacit_assumption_miner import Phi15Engine

config = {
    'confidence_threshold': 0.6,
    'crisis_threshold': 0.7,
    'min_failures_for_clustering': 10,
    'anomaly_contamination': 0.1,
    'batch_size': 100,  # Process in batches
}

engine = Phi15Engine(config)

# Expected performance:
# - 1K failures: <10s
# - 10K failures: <60s
```

---

### 4. I_mech (Isomorphic Mechanism)

#### Bottlenecks Identified
- NP-complete subgraph isomorphism
- Repeated FDG traversal
- Expensive causal discovery

#### Optimizations Implemented
1. **Parallel Isomorphism Detection**
   - VF2 algorithm with pruning
   - WL coloring for early rejection
   - Parallel candidate evaluation

2. **Subgraph Caching**
   - Cache frequent patterns
   - Reuse isomorphism results
   - LRU eviction

3. **Incremental Causal Discovery**
   - PC algorithm with parallelization
   - Conditional independence caching
   - Adaptive significance testing

#### Tuning Parameters
```python
from rese.phase2.imech.algorithms.vf2 import VF2Isomorphism

vf2 = VF2Isomorphism(
    use_coloring=True,       # Enable WL coloring
    parallel=True,           # Parallel matching
    cache_subgraphs=True,    # Cache results
    max_cache_size=1000,
)

# Expected performance:
# - Domain pair comparison: <30s
```

---

### 5. Gamma1 (ACI Calculator)

#### Bottlenecks Identified
- Expensive entropy calculations
- Repeated coherence computations
- Solvability prediction overhead

#### Optimizations Implemented
1. **Memoized Component Calculations**
   - Cache entropy components
   - Cache coherence metrics
   - Lazy solvability evaluation

2. **Parallel Entropy Calculation**
   - Parallel local entropy computation
   - Vectorized constraint entropy
   - Batch structural entropy

3. **Incremental Updates**
   - Update only affected components
   - Delta calculations
   - Version tracking

#### Tuning Parameters
```python
from rese.gamma1.core.aci_calculator import ACICalculator

calculator = ACICalculator(
    alpha=0.35,
    beta=0.35,
    gamma=0.30,
    use_cache=True,  # Enable caching
)

# Expected performance:
# - 100 variables: <5s
# - 1000 variables: <30s
```

#### Cache Management
```python
# Get cache stats
stats = calculator.get_cache_stats()
print(f"Cache size: {stats['cache_size']}")
print(f"Cache enabled: {stats['cache_enabled']}")

# Clear cache if needed
calculator.clear_cache()
```

---

### 6. MCTS Search

#### Bottlenecks Identified
- Sequential tree traversal
- Expensive playouts
- No reuse across searches

#### Optimizations Implemented
1. **Parallel MCTS**
   - Root parallelization
   - Tree parallelization with virtual loss
   - Thread-safe updates

2. **Adaptive Playouts**
   - ACI-guided depth adjustment
   - Early termination
   - Causally-guided selection

3. **Progressive Widening**
   - Control expansion rate
   - Adaptive exploration
   - Bandit-based selection

#### Tuning Parameters
```python
from rese.phase3.mcts_search import MCTSConfig, MCTSSearch

config = MCTSConfig(
    # Parallelization
    num_workers=4,              # Number of parallel workers
    virtual_loss=True,          # Enable tree parallelization

    # Adaptive parameters
    adaptive_c=True,            # Adaptive exploration
    adaptive_depth=True,        # Adaptive playout depth

    # Progressive widening
    progressive_widening=True,
    widening_constant=0.5,

    # Early stopping
    early_stopping=True,
    convergence_window=20,

    # ACI guidance
    aci_guided=True,
)

mcts = MCTSSearch(config)

# Expected performance:
# - 1000 iterations: <60s
# - With 4 workers: <20s
```

---

## Caching Strategy

### Global Cache Manager

The `CacheManager` provides unified caching across all components:

```python
from rese.performance.cache_manager import CacheManager

# Initialize cache manager
cache = CacheManager(
    use_redis=False,        # Set True for production
    max_memory_size=10000,  # Max entries
)

# Use in components
cache.set("key", value, component="SCE", ttl=3600.0)
value = cache.get("key", component="SCE")

# Invalidate on updates
cache.invalidate_constraint("constraint_123")
```

### Cache Invalidation Strategy

1. **Time-based expiration**
   - Default TTL: 1 hour
   - Component-specific TTLs
   - Automatic cleanup

2. **Event-based invalidation**
   - On constraint add/remove/modify
   - On graph updates
   - On parameter changes

3. **LRU eviction**
   - Least recently used eviction
   - Configurable cache size
   - Automatic management

---

## Production Deployment

### Configuration

```python
# config.py
PERFORMANCE_CONFIG = {
    # Global settings
    "enable_parallel": True,
    "num_threads": 4,
    "enable_cache": True,
    "use_redis": True,
    "redis_host": "localhost",
    "redis_port": 6379,

    # Component settings
    "sce": {
        "parallel_threshold": 1000,
        "cache_size": 10000,
    },
    "dito": {
        "rtree_max_entries": 50,
        "lsh_num_tables": 10,
        "parallel": True,
    },
    "phi15": {
        "batch_size": 100,
        "cache_features": True,
    },
    "gamma1": {
        "cache_components": True,
        "parallel_entropy": True,
    },
    "mcts": {
        "num_workers": 4,
        "virtual_loss": True,
    },
}
```

### Monitoring

```python
from rese.performance.benchmarks import PerformanceBenchmark

# Run benchmarks
suite = PerformanceBenchmark()
suite.benchmark(...)

# Generate report
report = suite.generate_report("performance_report.txt")

# Save metrics
suite.save_results("metrics.json")
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile component
profiler = cProfile.Profile()
profiler.enable()

# Run code
result = component.process(data)

profiler.disable()

# Analyze
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Troubleshooting

### Common Issues

#### 1. SCE slow for large constraint sets
**Symptoms**: Adding constraints takes >1s for 10K

**Solutions**:
- Enable parallel processing: `use_parallel=True`
- Increase cache size: `cache_max_size=20000`
- Check for cache misses in stats

#### 2. DITO memory usage high
**Symptoms**: Memory grows with constraint count

**Solutions**:
- Reduce R-tree node size: `rtree_max_entries=25`
- Enable lazy evaluation: `lazy_mode=True`
- Clear cache periodically

#### 3. Phi1.5 clustering slow
**Symptoms**: Processing 1K failures takes >10s

**Solutions**:
- Reduce clustering attempts: `n_clusters_range=(2, 5)`
- Use mini-batch clustering
- Enable feature caching

#### 4. MCTS not converging
**Symptoms**: Search runs max iterations

**Solutions**:
- Adjust convergence threshold
- Enable early stopping
- Increase exploration parameter

---

## Performance Checklist

Before deploying to production:

- [ ] All components meet performance targets
- [ ] Caching enabled and configured
- [ ] Parallel processing enabled
- [ ] Redis/memcached configured (if using)
- [ ] Monitoring/logging in place
- [ ] Benchmark baseline established
- [ ] Load testing completed
- [ ] Memory usage acceptable
- [ ] Error handling robust
- [ ] Documentation updated

---

## Contact

For performance issues or questions:
- Agent M1 (Performance Optimization Specialist)
- Created: 2025-12-31
- Version: 1.0
