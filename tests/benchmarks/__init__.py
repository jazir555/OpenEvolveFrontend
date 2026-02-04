"""
Gauntlet Performance Benchmarking Suite

This package provides comprehensive performance benchmarks for the OpenEvolve
Gauntlet System with baseline metrics comparison and CI/CD integration.

Main Components:
- GauntletBenchmarkSuite: Main benchmark runner
- BaselineMetrics: Performance baseline definitions
- PerformanceTargets: Pass/fail criteria

Example:
    >>> from tests.benchmarks import GauntletBenchmarkSuite
    >>> suite = GauntletBenchmarkSuite()
    >>> results = suite.run_all_benchmarks()
    >>> results.to_json("results.json")

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

from .gauntlet_benchmarks import (
    GauntletBenchmarkSuite,
    BaselineMetrics,
    PerformanceTargets,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkStatus
)

__all__ = [
    "GauntletBenchmarkSuite",
    "BaselineMetrics",
    "PerformanceTargets",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkStatus"
]

__version__ = "1.0.0"
