"""
Lean4 Data Models for Formal Verification
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ProofObligation:
    """
    Represents a formal proof obligation in Lean4.

    Attributes:
        name: Unique identifier for the proof obligation
        statement: Formal statement to be proven
        property_type: Type of property (correctness, termination, etc.)
        function_name: Associated function name (if applicable)
        timestamp: When the obligation was created
        metadata: Additional metadata
    """
    name: str
    statement: str
    property_type: str = "correctness"
    function_name: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'statement': self.statement,
            'property_type': self.property_type,
            'function_name': self.function_name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }
