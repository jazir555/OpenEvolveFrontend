"""
Evaluators Package for OpenEvolve Gauntlets

This package contains various evaluator implementations that can be used
within the gauntlet system for validating solutions.
"""

from .loongflow_adapter import LoongFlowEvaluatorAdapter

__all__ = [
    'LoongFlowEvaluatorAdapter',
]
