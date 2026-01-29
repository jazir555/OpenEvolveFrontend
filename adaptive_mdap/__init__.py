"""
Adaptive MDAP - Massively Decomposed Agentic Processes with Adaptive Resource Allocation

This package implements the Adaptive-MAKER integration, combining the MAKER framework
(MDAP - Massively Decomposed Agentic Processes) with adaptive resource allocation
to achieve 30-50% cost reduction while maintaining quality within ±1% of baseline.

Based on: "Solving a Million-Step LLM Task with Zero Errors" by Meyerson et al.
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve Integration Team"

from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import AdaptiveMDAPAllocator, SolveConfig, SolveStrategy
from adaptive_mdap.controllers.execution_controller import AdaptiveExecutionController

__all__ = [
    "TaskComplexityClassifier",
    "AdaptiveMDAPAllocator", 
    "SolveConfig",
    "SolveStrategy",
    "AdaptiveExecutionController",
]
