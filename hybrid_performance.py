"""
Performance Optimization for Hybrid MAKER Integration System

This module provides performance enhancements:
- Population caching
- Proof memoization
- Parallel evaluation
- Resource pooling
- Performance monitoring

Author: OpenEvolve Hybrid Performance Team
Created: 2025-01-07
Version: 1.0.0
"""

import asyncio
import hashlib
import functools
import time
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE CACHING FOR HYBRID SYSTEM
# =============================================================================

class HybridProofCache:
    """Cache for proof evaluations and generations"""

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _generate_key(self, theorem: str, strategy: str, config: Any) -> str:
        """Generate cache key"""
        config_str = str(config) if config else ""
        key_str = f"{strategy}:{theorem}:{config_str}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, theorem: str, strategy: str, config: Any = None) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(theorem, strategy, config)

        if key in self.cache:
            self.hits += 1
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, theorem: str, strategy: str, result: Any, config: Any = None):
        """Set cache entry"""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]

        key = self._generate_key(theorem, strategy, config)
        self.cache[key] = result
        self.access_count[key] = 0

    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


# =============================================================================
# PARALLEL EVALUATION
# =============================================================================

class ParallelPopulationEvaluator:
    """Parallel evaluator for population fitness"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    async def evaluate_population(
        self,
        population: List[Any],
        fitness_func: Callable,
        theorem: str,
        context: Dict[str, Any]
    ) -> List[float]:
        """Evaluate population in parallel"""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def evaluate_with_limit(individual):
            async with semaphore:
                if asyncio.iscoroutinefunction(fitness_func):
                    return await fitness_func(individual, theorem, context)
                else:
                    return fitness_func(individual, theorem, context)

        tasks = [evaluate_with_limit(ind) for ind in population]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        fitnesses = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Evaluation error: {result}")
                fitnesses.append(0.0)
            else:
                fitnesses.append(result)

        return fitnesses


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

@dataclass
class HybridPerformanceMetrics:
    """Performance metrics for hybrid system"""
    theorem: str
    strategy: str
    start_time: float
    end_time: Optional[float] = None
    generations: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def fitness_per_second(self) -> float:
        """Get fitness improvement rate"""
        if self.duration > 0:
            return self.best_fitness / self.duration
        return 0.0


class HybridPerformanceMonitor:
    """Monitor hybrid system performance"""

    def __init__(self):
        self.metrics: List[HybridPerformanceMetrics] = []
        self.current: Optional[HybridPerformanceMetrics] = None

    def start_tracking(self, theorem: str, strategy: str, population_size: int = 0):
        """Start tracking performance"""
        self.current = HybridPerformanceMetrics(
            theorem=theorem,
            strategy=strategy,
            start_time=time.time(),
            population_size=population_size
        )

    def end_tracking(self, generations: int, best_fitness: float, cache_stats: Dict[str, Any] = None):
        """End tracking and record metrics"""
        if self.current:
            self.current.end_time = time.time()
            self.current.generations = generations
            self.current.best_fitness = best_fitness

            if cache_stats:
                self.current.cache_hits = cache_stats.get("hits", 0)
                self.current.cache_misses = cache_stats.get("misses", 0)

            self.metrics.append(self.current)
            self.current = None

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {}

        total_duration = sum(m.duration for m in self.metrics)
        avg_fitness = sum(m.best_fitness for m in self.metrics) / len(self.metrics)
        avg_generations = sum(m.generations for m in self.metrics) / len(self.metrics)

        return {
            "total_runs": len(self.metrics),
            "total_duration": total_duration,
            "avg_duration": total_duration / len(self.metrics),
            "avg_fitness": avg_fitness,
            "avg_generations": avg_generations,
            "total_cache_hits": sum(m.cache_hits for m in self.metrics),
            "total_cache_misses": sum(m.cache_misses for m in self.metrics),
        }


# =============================================================================
# DECORATORS
# =============================================================================

def cached_hybrid_proof(cache: HybridProofCache):
    """Decorator for caching proof generation"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(theorem: str, strategy: str, config: Any = None, **kwargs):
            # Try cache
            cached_result = cache.get(theorem, strategy, config)
            if cached_result is not None:
                return cached_result

            # Generate
            result = await func(theorem, strategy, config, **kwargs)

            # Cache
            cache.set(theorem, strategy, result, config)

            return result

        return wrapper
    return decorator


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    print("Hybrid MAKER Performance Optimization")
    print("=" * 60)

    # Create cache
    cache = HybridProofCache(max_size=100)

    # Test cache
    print("\n1. Proof Caching")
    print("-" * 40)

    theorem = "forall n m : nat, n + m = m + n"
    strategy = "MCTS_Then_MAKER"

    result = cache.get(theorem, strategy, None)
    print(f"First access (miss): {result}")

    cache.set(theorem, strategy, {"success": True, "fitness": 0.85})
    result = cache.get(theorem, strategy, None)
    print(f"Second access (hit): {result}")

    stats = cache.get_stats()
    print(f"Cache stats: {stats}")

    # Test parallel evaluation
    print("\n2. Parallel Evaluation")
    print("-" * 40)

    evaluator = ParallelPopulationEvaluator(max_workers=4)

    # Mock population
    class MockIndividual:
        def __init__(self, id):
            self.id = id
            self.genome = f"proof_{id}"

    population = [MockIndividual(i) for i in range(10)]

    async def mock_fitness(ind, theorem, ctx):
        await asyncio.sleep(0.1)
        return 0.5 + ind.id * 0.05

    start = time.time()
    fitnesses = asyncio.run(evaluator.evaluate_population(
        population,
        mock_fitness,
        theorem,
        {}
    ))
    duration = time.time() - start

    print(f"Evaluated {len(population)} individuals in {duration:.2f}s")
    print(f"Fitnesses: {fitnesses[:3]}...")

    # Test performance monitoring
    print("\n3. Performance Monitoring")
    print("-" * 40)

    monitor = HybridPerformanceMonitor()
    monitor.start_tracking(theorem, strategy, population_size=20)
    # Simulate work
    time.sleep(0.5)
    monitor.end_tracking(generations=10, best_fitness=0.85)

    summary = monitor.get_summary()
    print(f"Performance summary: {summary}")

    print("\n" + "=" * 60)
    print("Performance optimization demo complete!")
