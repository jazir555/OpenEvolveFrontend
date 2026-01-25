"""
Knowledge Graph Performance Benchmark Suite

Comprehensive performance testing framework for the OpenEvolve Knowledge Graph system.

This package provides:
- Performance benchmarking for all KG components
- Throughput, latency, memory, and scalability testing
- Automated report generation
- Visualization tools
- CI/CD integration support

Main Components:
- KnowledgeGraphPerformanceBenchmarks: Core benchmark suite
- BenchmarkRunner: Command-line benchmark execution
- BenchmarkVisualizer: Chart and graph generation

Usage:
    from tests.benchmarks import KnowledgeGraphPerformanceBenchmarks

    benchmarks = KnowledgeGraphPerformanceBenchmarks(kg_engine)
    result = await benchmarks.benchmark_knowledge_addition(
        num_artifacts=1000
    )

Author: OpenEvolve Framework
Date: 2025-01-07
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve Framework"

from .kg_performance_benchmarks import (
    KnowledgeGraphPerformanceBenchmarks,
    BenchmarkResult
)

__all__ = [
    "KnowledgeGraphPerformanceBenchmarks",
    "BenchmarkResult",
]
