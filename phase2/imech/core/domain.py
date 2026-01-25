"""
Domain Representation for I_mech

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .fdg import FunctionalDependencyGraph


@dataclass
class Domain:
    """
    Problem domain representation
    """
    id: str
    name: str
    description: str

    # Constraints
    formal_constraints: List[Any] = field(default_factory=list)
    natural_language_constraints: List[str] = field(default_factory=list)

    # Historical data
    historical_data: Optional[Any] = None
    solutions: List[Any] = field(default_factory=list)

    # Extracted FDG
    fdg: Optional[FunctionalDependencyGraph] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Units and conversions
    units: Dict[str, str] = field(default_factory=dict)

    def has_solution(self) -> bool:
        """Check if domain has solutions"""
        return len(self.solutions) > 0

    def get_primary_solution(self) -> Optional[Any]:
        """Get the primary solution"""
        return self.solutions[0] if self.solutions else None

    def __repr__(self):
        return f"Domain(id='{self.id}', name='{self.name}', constraints={len(self.formal_constraints)})"
