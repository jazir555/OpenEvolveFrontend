"""
Load Testing Package for Knowledge Graph System

This package provides comprehensive load testing capabilities including:
- Asynchronous load test framework
- Locust integration for HTTP load testing
- Result analysis and reporting
- Performance monitoring and bottleneck detection

Main Components:
- KnowledgeGraphLoadTest: Core load testing framework
- LoadTestAnalyzer: Result analysis and reporting
- Locust files: HTTP-based load testing
"""

from .kg_load_tests import KnowledgeGraphLoadTest, LoadTestResult
from .analyze_results import LoadTestAnalyzer

__all__ = [
    "KnowledgeGraphLoadTest",
    "LoadTestResult",
    "LoadTestAnalyzer"
]

__version__ = "1.0.0"
