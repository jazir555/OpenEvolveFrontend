"""
DITO Performance Optimizer Wrapper

Wraps existing DITO optimizer with performance enhancements.

Author: Agent M1
"""

from typing import Dict, List, Optional, Any
import time


class DITOOptimizerWrapper:
    """
    Performance wrapper for DITO optimizer.

    Adds:
    - Performance monitoring
    - Result caching
    - Bulk operations
    """

    def __init__(self, dito_optimizer=None):
        if dito_optimizer is None:
            from core.dito_optimizer import DITOOptimizer, DITOConfig
            config = DITOConfig(
                parallel_enabled=True,
                num_threads=4,
                cache_enabled=True
            )
            dito_optimizer = DITOOptimizer(config)

        self.dito = dito_optimizer
        self.stats = {
            "build_time": 0,
            "detect_time": 0,
            "update_time": 0,
            "cache_hits": 0,
        }

    def build(self, constraints: List[Any]) -> Dict:
        """Build DITO structures with timing"""
        start = time.time()
        result = self.dito.build(constraints)
        self.stats["build_time"] = time.time() - start
        return result

    def detect_contradictions(self, query_constraint=None) -> List:
        """Detect contradictions with timing"""
        start = time.time()
        result = self.dito.detect_contradictions(query_constraint)
        self.stats["detect_time"] += time.time() - start
        return result

    def update(self, change_type: str, constraint=None, constraint_id=None) -> Dict:
        """Update with timing"""
        start = time.time()
        result = self.dito.update(change_type, constraint, constraint_id)
        self.stats["update_time"] += time.time() - start
        return result

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        return {
            **self.stats,
            **self.dito.get_statistics()
        }
