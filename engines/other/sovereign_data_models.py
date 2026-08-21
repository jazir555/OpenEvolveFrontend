"""
Sovereign Data Models (Legacy/Facade)
This module now redirects to openevolve.kernel.schema.
Maintained for backward compatibility.
"""
from __future__ import annotations


from openevolve.kernel.schema import *

import sys as _sys

# Backward-compatibility fallbacks for symbols that are no longer present in the
# kernel schema. The original modules import these names directly; provide minimal
# data-model stand-ins so imports keep working until callers are migrated.
_MISSING_NAMES = [
    "AuditEvent", "AuditTrail", "AutomatedCheckResults", "CheckpointInfo",
    "ComplexityBreakdown", "ComplexityDimension", "DecompositionStrategy",
    "DependencyGraph", "EnhancedDomainContext", "EnhancedQualityScores",
    "Feedback", "GauntletAssignment", "GauntletExecution", "HealingResult",
    "HealthIssue", "Pattern", "ProblemCategory", "QualityScores",
    "RedTeamCritiqueReport", "ResourceEstimate", "SolutionValidationResults",
    "StrategyPerformanceMetrics", "TeamAssignment", "TeamPerformanceMetrics",
    "ValidationCheckpoint", "ValidationRequirements", "ValidationResult",
    "WorkflowProgress",
]


class _SovereignFallback:
    """Minimal fallback data model: accepts arbitrary fields as kwargs."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({fields})"


_self = _sys.modules[__name__]
for _name in _MISSING_NAMES:
    if not hasattr(_self, _name):
        setattr(_self, _name, type(_name, (_SovereignFallback,), {}))