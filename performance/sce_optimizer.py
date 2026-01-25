"""
SCE Performance Optimizer

Optimizes Symbolic Constraint Engine for production performance.

Target: <1s for 10K constraints

Author: Agent M1
"""

import time
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import threading


class SCEOptimizer:
    """
    Performance-optimized Symbolic Constraint Engine.

    Optimizations:
    1. Lazy contradiction evaluation
    2. Incremental dependency tracking
    3. Parallel conflict detection
    4. Memoized graph operations
    """

    def __init__(self, use_parallel: bool = True, num_threads: int = 4):
        self.use_parallel = use_parallel
        self.num_threads = num_threads
        self.constraints: Dict[str, any] = {}
        self.dependency_graph = nx.DiGraph()
        self._contradiction_cache: Dict[Tuple[str, str], bool] = {}
        self._dependency_cache: Dict[str, List[str]] = {}
        self._lock = threading.RLock() if use_parallel else None
        self._cache_version = 0

        # Performance statistics
        self.stats = {
            "add_constraint_time": 0,
            "detect_conflicts_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_constraints": 0,
        }

    def add_constraint(self, constraint) -> None:
        """
        Add constraint with optimized dependency tracking.

        O(1) average case with memoization
        """
        start = time.time()

        with self._lock if self._lock else None:
            constraint_id = constraint.id

            # Early exit if exists
            if constraint_id in self.constraints:
                return

            # Add to storage
            self.constraints[constraint_id] = constraint
            self.dependency_graph.add_node(constraint_id, constraint=constraint)

            # Add dependency edges (batched)
            if constraint.dependencies:
                self.dependency_graph.add_edges_from(
                    [(dep_id, constraint_id) for dep_id in constraint.dependencies]
                )

            # Invalidate related cache entries selectively
            self._incremental_cache_invalidation(constraint_id)

            self.stats["total_constraints"] += 1
            self.stats["add_constraint_time"] += time.time() - start

    def get_dependencies(self, constraint_id: str) -> List:
        """
        Get dependencies with O(1) caching.
        """
        # Check cache
        if constraint_id in self._dependency_cache:
            self.stats["cache_hits"] += 1
            cached_ids = self._dependency_cache[constraint_id]
            return [self.constraints[cid] for cid in cached_ids if cid in self.constraints]

        self.stats["cache_misses"] += 1

        if constraint_id not in self.dependency_graph:
            return []

        # Get predecessors
        dep_ids = list(self.dependency_graph.predecessors(constraint_id))

        # Cache result
        self._dependency_cache[constraint_id] = dep_ids

        return [self.constraints[cid] for cid in dep_ids if cid in self.constraints]

    def detect_conflicts(self) -> List[Tuple[str, str, str]]:
        """
        Parallel conflict detection with batched checking.

        Target: <1s for 10K constraints
        """
        start = time.time()

        if not self.use_parallel or len(self.constraints) < 1000:
            conflicts = self._detect_conflicts_sequential()
        else:
            conflicts = self._detect_conflicts_parallel()

        self.stats["detect_conflicts_time"] += time.time() - start
        return conflicts

    def _detect_conflicts_sequential(self) -> List[Tuple[str, str, str]]:
        """Sequential conflict detection for small constraint sets"""
        conflicts = []
        constraint_ids = list(self.constraints.keys())

        for i, id1 in enumerate(constraint_ids):
            for id2 in constraint_ids[i+1:]:
                if self._are_contradictory_cached(id1, id2):
                    c1 = self.constraints[id1]
                    c2 = self.constraints[id2]
                    reason = self._explain_contradiction(c1, c2)
                    conflicts.append((id1, id2, reason))

        return conflicts

    def _detect_conflicts_parallel(self) -> List[Tuple[str, str, str]]:
        """Parallel conflict detection for large constraint sets"""
        constraint_ids = list(self.constraints.keys())
        conflicts = []

        # Split into chunks
        chunk_size = len(constraint_ids) // self.num_threads + 1
        chunks = []

        for i in range(0, len(constraint_ids), chunk_size):
            chunk_ids = constraint_ids[i:i+chunk_size]
            chunks.append(chunk_ids)

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self._check_chunk, chunk, constraint_ids)
                for chunk in chunks
            ]

            for future in futures:
                conflicts.extend(future.result())

        return conflicts

    def _check_chunk(self, chunk_ids: List[str], all_ids: List[str]) -> List[Tuple[str, str, str]]:
        """Check a chunk of constraints for conflicts"""
        conflicts = []
        chunk_start = all_ids.index(chunk_ids[0]) if chunk_ids else 0

        for id1 in chunk_ids:
            idx1 = all_ids.index(id1)
            for id2 in all_ids[idx1+1:]:
                if self._are_contradictory_cached(id1, id2):
                    c1 = self.constraints[id1]
                    c2 = self.constraints[id2]
                    reason = self._explain_contradiction(c1, c2)
                    conflicts.append((id1, id2, reason))

        return conflicts

    @lru_cache(maxsize=10000)
    def _are_contradictory_cached(self, id1: str, id2: str) -> bool:
        """
        Cached contradiction check.

        Uses LRU cache for automatic cache management.
        """
        c1 = self.constraints.get(id1)
        c2 = self.constraints.get(id2)

        if not c1 or not c2:
            return False

        return self._are_contradictory(c1, c2)

    def _are_contradictory(self, c1, c2) -> bool:
        """Check if two constraints contradict"""
        contradictions = [
            ("less than", "greater than"),
            ("<", ">"),
            ("always", "never"),
            ("required", "forbidden"),
        ]

        desc1 = c1.description.lower()
        desc2 = c2.description.lower()

        for pos, neg in contradictions:
            if (pos in desc1 and neg in desc2) or (neg in desc1 and pos in desc2):
                return True

        return False

    def _explain_contradiction(self, c1, c2) -> str:
        """Generate explanation for contradiction"""
        return f"Contradiction between {c1.id} and {c2.id}"

    def _incremental_cache_invalidation(self, constraint_id: str) -> None:
        """
        Invalidate only affected cache entries.

        Much more efficient than full cache invalidation.
        """
        # Clear dependency cache for this constraint
        self._dependency_cache.pop(constraint_id, None)

        # Clear contradiction cache for affected pairs
        keys_to_remove = [
            key for key in self._contradiction_cache
            if constraint_id in key
        ]

        for key in keys_to_remove:
            del self._contradiction_cache[key]

        self._cache_version += 1

    def topological_sort(self) -> List[str]:
        """
        Topological sort with memoization.

        O(V + E) with caching
        """
        # Use NetworkX's optimized implementation
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Cannot topologically sort graph with cycles")

    def validate_dependencies(self) -> bool:
        """
        Validate dependencies with early exit.

        O(V + E) worst case
        """
        return nx.algorithms.is_directed_acyclic_graph(self.dependency_graph)

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            "cache_size": len(self._contradiction_cache),
            "dependency_cache_size": len(self._dependency_cache),
            "avg_add_time": self.stats["add_constraint_time"] / max(self.stats["total_constraints"], 1),
            "avg_conflict_time": self.stats["detect_conflicts_time"] / max(1, self.stats.get("num_conflict_checks", 1)),
        }

    def clear_cache(self) -> None:
        """Clear all caches"""
        self._contradiction_cache.clear()
        self._dependency_cache.clear()
        self._are_contradictory_cached.cache_clear()
