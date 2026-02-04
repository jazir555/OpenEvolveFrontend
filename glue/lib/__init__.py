"""
Glue Library - Shared Utilities for OpenEvolve Frontend

This library provides common utilities and adapters for integrating
core projects into the Mega-Structure.

Following CLAUDE.md principles:
- Law of Air Gap: No imports from core-projects/
- Law of Configuration Explicitness: All config via env vars
- Law of Idempotency: UPSERT logic
- Structured Logging: JSON with correlation_id
"""

__version__ = "1.0.0"

# Export RESE components
try:
    from .rese_dee import (
        DeepExplorationEngine,
        HypothesisGenerator,
        PatternRecognizer,
        MCTSExplainer,
        DEELogger,
        CircuitBreaker,
        CircuitBreakerOpenError,
        retry_with_backoff,
    )
    __all__ = [
        "DeepExplorationEngine",
        "HypothesisGenerator",
        "PatternRecognizer",
        "MCTSExplainer",
        "DEELogger",
        "CircuitBreaker",
        "CircuitBreakerOpenError",
        "retry_with_backoff",
    ]
except ImportError:
    # RESE components not available
    __all__ = []
