"""
RESE Performance Optimization Package

Provides performance-optimized versions of all RESE components.

Author: Agent M1 (Performance Optimization Specialist)
Created: 2025-12-31
"""

from .sce_optimizer import SCEOptimizer
from .dito_optimizer import DITOOptimizerWrapper
from .phi15_optimizer import Phi15Optimizer
from .imech_optimizer import IMechOptimizer
from .gamma1_optimizer import Gamma1Optimizer
from .mcts_optimizer import MCTSOptimizer
from .cache_manager import CacheManager, InMemoryCache
from .benchmarks import (
    PerformanceBenchmark,
    create_sce_benchmark,
    create_dito_benchmark,
    create_phi15_benchmark,
    create_gamma1_benchmark,
    create_mcts_benchmark,
)

__all__ = [
    # Optimizers
    'SCEOptimizer',
    'DITOOptimizerWrapper',
    'Phi15Optimizer',
    'IMechOptimizer',
    'Gamma1Optimizer',
    'MCTSOptimizer',

    # Caching
    'CacheManager',
    'InMemoryCache',

    # Benchmarks
    'PerformanceBenchmark',
    'create_sce_benchmark',
    'create_dito_benchmark',
    'create_phi15_benchmark',
    'create_gamma1_benchmark',
    'create_mcts_benchmark',
]

__version__ = '1.0.0'
__author__ = 'Agent M1'
__status__ = 'OPTIMIZATION_COMPLETE'
