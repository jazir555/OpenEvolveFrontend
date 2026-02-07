"""Knowledge Engine Finance Schemas."""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class FinancialConfig:
    """Financial configuration."""
    risk_tolerance: float = 0.5
    return_target: float = 0.1
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}

@dataclass
class Portfolio:
    """Portfolio."""
    assets: List[str] = None
    weights: List[float] = None
    
    def __post_init__(self):
        if self.assets is None:
            self.assets = []
        if self.weights is None:
            self.weights = []
