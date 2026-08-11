"""
Adaptive MDAP Module Bridge

Multi-Dimensional Adaptive Planning module for OpenEvolve.
This file bridges to the adaptive_mdap package for full functional implementation.
"""

from adaptive_mdap import *
from adaptive_mdap.integrations.workflow_engine_integration import (
    AdaptiveWorkflowIntegration,
    AdaptiveWorkflowConfig,
    get_adaptive_workflow,
)

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
