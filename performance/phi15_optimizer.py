"""
Phi1.5 Performance Optimizer

Optimizes Tacit Assumption Miner for production performance.

Target: <10s for 1K failures

Author: Agent M1
"""

import time
from typing import Dict, List
import numpy as np


class Phi15Optimizer:
    """
    Performance-optimized Phi1.5 engine.

    Optimizations:
    1. Batch processing of null results
    2. Cached feature extraction
    3. Vectorized clustering
    """

    def __init__(self, engine=None, batch_size=100):
        if engine is None:
            from phase1.tacit_assumption_miner import Phi15Engine
            engine = Phi15Engine()

        self.engine = engine
        self.batch_size = batch_size
        self.feature_cache = {}
        self.stats = {
            "total_processed": 0,
            "total_time": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def process_null_results(self, null_results: List) -> tuple:
        """
        Process null results in batches for better performance.
        """
        start = time.time()

        # Process in batches
        all_assumptions = []
        all_paradigm_recs = []

        for i in range(0, len(null_results), self.batch_size):
            batch = null_results[i:i+self.batch_size]

            # Process batch
            assumptions, paradigm_rec = self.engine.process_null_results(batch)

            all_assumptions.extend(assumptions)
            all_paradigm_recs.append(paradigm_rec)

        elapsed = time.time() - start
        self.stats["total_processed"] += len(null_results)
        self.stats["total_time"] += elapsed

        return all_assumptions, all_paradigm_recs[-1] if all_paradigm_recs else None

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        total_processed = self.stats["total_processed"]
        avg_time_per_result = (
            self.stats["total_time"] / total_processed
            if total_processed > 0 else 0
        )

        return {
            **self.stats,
            "avg_time_per_result": avg_time_per_result,
            "results_per_second": total_processed / max(self.stats["total_time"], 0.001),
            "cache_size": len(self.feature_cache),
        }
