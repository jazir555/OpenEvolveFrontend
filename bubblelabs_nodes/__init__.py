"""
BubbleLabs Node Registry

Central registry for all OpenEvolve nodes integrated with BubbleLabs.

NOTE: Only Phase 1.1 (Parallel Execution) and 1.4 (Checkpointing) are currently
integrated and working. All other components have skeleton files but are NOT integrated.
"""

from typing import Dict, Type, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class NodeRegistry:
    """
    Registry for managing available node types.
    """

    _nodes: Dict[str, Type[BubbleLabsNode]] = {}

    @classmethod
    def register(cls, node_type: str, node_class: Type[BubbleLabsNode]):
        """Register a node type."""
        if not issubclass(node_class, BubbleLabsNode):
            raise TypeError(f"{node_class} must inherit from BubbleLabsNode")

        cls._nodes[node_type] = node_class

    @classmethod
    def get(cls, node_type: str, config: Optional[Dict] = None) -> BubbleLabsNode:
        """Create a node instance."""
        node_class = cls._nodes.get(node_type)
        if not node_class:
            available = ', '.join(cls._nodes.keys())
            raise ValueError(
                f"Unknown node type: {node_type}. "
                f"Available types: {available}"
            )

        try:
            return node_class(config)
        except Exception as e:
            raise ValueError(f"Failed to instantiate node {node_type}: {str(e)}")

    @classmethod
    def list_nodes(cls) -> Dict[str, Type[BubbleLabsNode]]:
        """Get all registered node types"""
        return cls._nodes.copy()


# Convenience functions
def register_node(node_type: str, node_class: Type[BubbleLabsNode]):
    """Register a node type (convenience function)"""
    try:
        NodeRegistry.register(node_type, node_class)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to register node {node_type}: {e}")
        raise


def get_node(node_type: str, config: Optional[Dict] = None) -> BubbleLabsNode:
    """Create a node instance (convenience function)"""
    try:
        return NodeRegistry.get(node_type, config)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to get node {node_type}: {e}")
        raise


def list_nodes() -> Dict[str, Type[BubbleLabsNode]]:
    """List all registered nodes (convenience function)"""
    try:
        return NodeRegistry.list_nodes()
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__).error(f"Failed to list nodes: {e}")
        return {}


__all__ = [
    'BubbleLabsNode',
    'NodeExecutionError',
    'NodeRegistry',
    'register_node',
    'get_node',
    'list_nodes',
]

# Phase 1 Gauntlet Enhancements - ONLY WHAT ACTUALLY WORKS
from .parallel_executor import (
    ParallelProblemExecutor,
    ProblemDependencyAnalyzer,
    ExecutionResult,
    ParallelExecutionSummary,
    get_parallel_executor,
)

from .worker_pool_executor import (
    WorkerPoolExecutor,
    WorkerTask,
    WorkerResult,
    PoolExecutionSummary,
    create_worker_pool_executor,
)

from .gauntlet_solver import (
    GauntletSolver,
    solveProblem,
)

from .solution_cache import (
    AtomicSolutionCache,
    ProblemHasher,
    InMemoryCache,
    CacheStatistics,
    create_solution_cache,
)

from .checkpoint_manager import (
    CheckpointManager,
    CheckpointRepository,
    StateSerializer,
    PipelineState,
    CheckpointMetadata,
    create_checkpoint_manager,
)

__all__ += [
    # Parallel Execution (WORKING)
    'ParallelProblemExecutor',
    'ProblemDependencyAnalyzer',
    'ExecutionResult',
    'ParallelExecutionSummary',
    'get_parallel_executor',

    # Worker Pool (WORKING)
    'WorkerPoolExecutor',
    'WorkerTask',
    'WorkerResult',
    'PoolExecutionSummary',
    'create_worker_pool_executor',

    # Enhanced Solver (WORKING with cache integration)
    'GauntletSolver',
    'solveProblem',

    # Solution Caching (WORKING)
    'AtomicSolutionCache',
    'ProblemHasher',
    'InMemoryCache',
    'CacheStatistics',
    'create_solution_cache',

    # Checkpointing (WORKING)
    'CheckpointManager',
    'CheckpointRepository',
    'StateSerializer',
    'PipelineState',
    'CheckpointMetadata',
    'create_checkpoint_manager',
]
