"""
ROMA Configuration
Configuration management for ROMA (Reliable Organic Multi-Agent system)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ROMAConfig:
    """Configuration for ROMA system components."""
    
    # Agent configuration
    num_agents: int = 5
    agent_timeout: int = 30
    max_retries: int = 3
    
    # Learning configuration
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    
    # Domain configuration
    domains: List[str] = field(default_factory=lambda: ["physics", "mathematics", "computer_science"])
    
    # MDAP integration
    mdap_enabled: bool = True
    mdap_endpoint: Optional[str] = None
    mdap_timeout: int = 60
    
    # Validation
    strict_validation: bool = False
    validation_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "num_agents": self.num_agents,
            "agent_timeout": self.agent_timeout,
            "max_retries": self.max_retries,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "domains": self.domains,
            "mdap_enabled": self.mdap_enabled,
            "mdap_endpoint": self.mdap_endpoint,
            "mdap_timeout": self.mdap_timeout,
            "strict_validation": self.strict_validation,
            "validation_interval": self.validation_interval
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROMAConfig":
        """Create configuration from dictionary."""
        return cls(**data)


# Default ROMA configuration
DEFAULT_ROMA_CONFIG = ROMAConfig()
