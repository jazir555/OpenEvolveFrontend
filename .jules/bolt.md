## 2026-02-15 - [Anti-pattern] O(n) LRU Cache Implementation
**Learning:** Multiple components (`llm_caching.py`, `performance_optimization.py`) implemented LRU caches using a simple list for access tracking, leading to $O(n)$ removals on every hit/set. In a performance-critical system with large caches, this becomes a significant bottleneck.
**Action:** Use `collections.OrderedDict` for LRU caches to ensure $O(1)$ operations. Always check existing cache implementations for this pattern when adding new caching layers.

## 2026-02-15 - [Architecture] Heavy Parameter Initialization
**Learning:** `ParameterManager` was re-initializing a schema with 211 parameters on every instantiation. In workflows where managers are created frequently (e.g., inside loops or per-request), this adds unnecessary CPU overhead.
**Action:** Use a singleton or class-level cache for heavy schema/configuration objects that don't change during runtime.

## 2026-04-08 - [Algorithm] Quadratic Deterministic Topological Sort
**Learning:** `DependencyDecomposition._topological_sort` was using `list.sort()` and `list.pop(0)` inside a loop over nodes, leading to $O(V^2 \log V)$ complexity. For large dependency graphs, this causes significant slowdown.
**Action:** Use `heapq` for deterministic topological sort to achieve $O(V \log V)$ while maintaining order consistency.

## 2026-04-08 - [Resource Management] ThreadPool Churn
**Learning:** `ParallelProcessor` was using a `with concurrent.futures.ThreadPoolExecutor(...)` block inside its processing method, causing a new pool of threads to be created and destroyed on every call. This adds massive overhead for short parallel tasks.
**Action:** Maintain a persistent `ThreadPoolExecutor` instance within the class and reuse it across calls.
