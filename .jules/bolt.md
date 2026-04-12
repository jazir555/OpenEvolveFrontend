## 2026-02-15 - [Anti-pattern] O(n) LRU Cache Implementation
**Learning:** Multiple components (`llm_caching.py`, `performance_optimization.py`) implemented LRU caches using a simple list for access tracking, leading to $O(n)$ removals on every hit/set. In a performance-critical system with large caches, this becomes a significant bottleneck.
**Action:** Use `collections.OrderedDict` for LRU caches to ensure $O(1)$ operations. Always check existing cache implementations for this pattern when adding new caching layers.

## 2026-02-15 - [Architecture] Heavy Parameter Initialization
**Learning:** `ParameterManager` was re-initializing a schema with 211 parameters on every instantiation. In workflows where managers are created frequently (e.g., inside loops or per-request), this adds unnecessary CPU overhead.
**Action:** Use a singleton or class-level cache for heavy schema/configuration objects that don't change during runtime.

## 2026-04-12 - [Database] SQLite REAL vs TEXT Timestamps
**Learning:** Storing timestamps as REAL (Unix) in SQLite is significantly more efficient for temporal comparisons than ISO TEXT strings.
**Action:** Always prefer REAL for timestamps in SQLite caches and implement migration logic to handle legacy ISO strings.

## 2026-04-12 - [Concurrency] Buffered Hit Count Updates
**Learning:** High-frequency DB writes for metrics like `hit_count` cause lock contention. Buffering updates in memory and flushing periodically improves throughput.
**Action:** Implement memory buffering for high-frequency write counters in database-backed caches.

## 2026-04-12 - [Algorithm] O(L²) to O(U²) Diversity Calculation
**Learning:** Evolutionary diversity calculations using pairwise string comparisons are O(L²). Grouping duplicate individuals using `Counter` reduces this to O(U²).
**Action:** Always group duplicates before performing expensive pairwise calculations in evolutionary algorithms.
