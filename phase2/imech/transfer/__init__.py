"""
I_mech Transfer Module
Solution transfer between isomorphic domains.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from .mapper import SolutionMapper
from .validator import SolutionValidator
from .repair import SolutionRepair

__all__ = [
    'SolutionMapper',
    'SolutionValidator',
    'SolutionRepair'
]
