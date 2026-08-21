"""
Adaptive MDAP Module Bridge

Multi-Dimensional Adaptive Planning module for OpenEvolve.
This file bridges to the adaptive_mdap package for full functional implementation.
"""
from __future__ import annotations


"""
Adaptive MDAP Module Bridge

Multi-Dimensional Adaptive Planning module for OpenEvolve.
This file bridges to the adaptive_mdap package for full functional implementation.
"""

try:  # pragma: no cover - integration optional
    from adaptive_mdap_pes_integration import (  # type: ignore
        TaskComplexityClassifier,
        AdaptiveMDAPAllocator,
        AdaptiveExecutionController,
        get_health_checker,
        ComplexityScore,
    )
except (ImportError, AttributeError):
    class _AdaptiveFallback:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class TaskComplexityClassifier(_AdaptiveFallback):
        pass

    class AdaptiveMDAPAllocator(_AdaptiveFallback):
        pass

    class AdaptiveExecutionController(_AdaptiveFallback):
        pass

    class ComplexityScore(_AdaptiveFallback):
        pass

    def get_health_checker(*args, **kwargs):
        return None

try:  # pragma: no cover - integration optional
    from adaptive_mdap_pes_integration import (  # type: ignore
        AdaptiveWorkflowIntegration,
        AdaptiveWorkflowConfig,
        get_adaptive_workflow,
    )
except (ImportError, AttributeError):
    class AdaptiveWorkflowIntegration:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class AdaptiveWorkflowConfig:
        def __init__(self, *args, **kwargs):
            pass

    def get_adaptive_workflow(*args, **kwargs):
        return AdaptiveWorkflowIntegration()

# Export everything
__all__ = [
    'TaskComplexityClassifier',
    'AdaptiveMDAPAllocator',
    'AdaptiveExecutionController',
    'get_health_checker',
    'ComplexityScore',
    'AdaptiveWorkflowIntegration',
    'AdaptiveWorkflowConfig',
    'get_adaptive_workflow'
]
