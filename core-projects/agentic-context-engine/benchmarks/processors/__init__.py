"""
Benchmark domain processors for ACE evaluation framework.

This package provides domain-specific data processors for handling different
benchmark datasets and converting them into standardized evaluation samples.
"""

from .finance import FinanceDataProcessor, load_finance_data

__all__ = [
    "FinanceDataProcessor",
    "load_finance_data",
]
