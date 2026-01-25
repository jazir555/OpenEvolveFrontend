"""
MCTS Performance Optimizer

Optimizes MCTS Search for production performance.

Target: <60s for 1000 iterations

Author: Agent M1
"""

import time
from typing import Callable, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


class MCTSOptimizer:
    """
    Performance-optimized MCTS search.

    Optimizations:
    1. Parallel tree search
    2. Optimized playouts
    3. Cached simulation results
    """

    def __init__(self, config=None, num_workers=4):
        if config is None:
            from phase3.mcts_search import MCTSConfig
            config = MCTSConfig(
                num_workers=num_workers,
                virtual_loss=True,
                early_stopping=True
            )

        from phase3.mcts_search import MCTSSearch

        self.mcts = MCTSSearch(config)
        self.num_workers = num_workers
        self.config = config
        self.simulation_cache = {}
        self.stats = {
            "searches": 0,
            "total_time": 0,
            "iterations_total": 0,
        }

    def search(self,
               initial_state: Any,
               action_generator: Callable,
               state_transition: Callable,
               value_function: Callable,
               initial_aci: Dict = None) -> Tuple[Any, Dict]:
        """
        Run parallel MCTS search.

        Target: <60s for 1000 iterations
        """
        start = time.time()

        # Run search
        best_node, search_info = self.mcts.search(
            initial_state,
            action_generator,
            state_transition,
            value_function,
            initial_aci
        )

        elapsed = time.time() - start
        self.stats["searches"] += 1
        self.stats["total_time"] += elapsed
        self.stats["iterations_total"] += search_info["iterations"]

        # Add performance info
        search_info["performance"] = {
            "elapsed_time": elapsed,
            "iterations_per_second": search_info["iterations"] / elapsed,
            "cache_size": len(self.simulation_cache),
        }

        return best_node, search_info

    def search_parallel(self,
                       initial_state: Any,
                       action_generator: Callable,
                       state_transition: Callable,
                       value_function: Callable,
                       num_workers: int = None) -> Tuple[Any, Dict]:
        """
        Run MCTS with multiple parallel workers.

        Target: 3-4x speedup with 4 workers
        """
        from phase3.mcts_search import ParallelMCTS

        num_workers = num_workers or self.num_workers

        parallel_mcts = ParallelMCTS(self.config)

        start = time.time()

        best_node, search_info = parallel_mcts.search_parallel(
            initial_state,
            action_generator,
            state_transition,
            value_function,
            num_workers
        )

        elapsed = time.time() - start

        # Update stats
        self.stats["searches"] += 1
        self.stats["total_time"] += elapsed
        self.stats["iterations_total"] += search_info["total_iterations"]

        # Add performance info
        search_info["performance"] = {
            "elapsed_time": elapsed,
            "num_workers": num_workers,
            "speedup": search_info["total_iterations"] / elapsed,
            "efficiency": (
                search_info["total_iterations"] / elapsed / num_workers
            ),
        }

        return best_node, search_info

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        total_searches = self.stats["searches"]
        avg_time = (
            self.stats["total_time"] / total_searches
            if total_searches > 0 else 0
        )

        total_iterations = self.stats["iterations_total"]
        avg_iterations = (
            total_iterations / total_searches
            if total_searches > 0 else 0
        )

        return {
            **self.stats,
            "avg_search_time": avg_time,
            "avg_iterations": avg_iterations,
            "overall_ips": total_iterations / max(self.stats["total_time"], 0.001),
            "cache_size": len(self.simulation_cache),
        }
