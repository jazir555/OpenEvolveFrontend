"""
Reliability System Configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ReliabilityConfig:
    """Configuration for the reliability system."""
    enable_redflagger: bool = True
    enable_guardrails: bool = True
    enable_lmql: bool = True
    
    # Redflagger configuration
    redflagger_threshold: float = 0.7
    max_flags_per_solution: int = 10
    
    # Guardrails configuration
    guardrails_api_url: Optional[str] = os.environ.get("GUARDRAILS_API_URL")
    
    # LMQL configuration
    lmql_server_url: Optional[str] = os.environ.get("LMQL_SERVER_URL")
    
    # Adaptive configuration
    use_adaptive_thresholds: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_redflagger": self.enable_redflagger,
            "enable_guardrails": self.enable_guardrails,
            "enable_lmql": self.enable_lmql,
            "redflagger_threshold": self.redflagger_threshold,
            "max_flags_per_solution": self.max_flags_per_solution,
            "use_adaptive_thresholds": self.use_adaptive_thresholds
        }
