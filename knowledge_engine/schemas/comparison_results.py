"""Knowledge Engine Comparison Results Schema."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ComparisonResult:
    """Comparison result."""
    source_id: str = ""
    target_id: str = ""
    similarity: float = 0.0
    differences: List[str] = None
    common_features: List[str] = None
    
    def __post_init__(self):
        if self.differences is None:
            self.differences = []
        if self.common_features is None:
            self.common_features = []
