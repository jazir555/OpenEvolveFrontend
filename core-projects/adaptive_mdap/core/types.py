"""Core type definitions for Adaptive MDAP."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SolveStrategy(Enum):
    """Enumeration of solving strategies with granular tiers."""
    DIRECT = "direct"           # Single agent, no voting
    MDAP_LIGHT = "mdap_light"   # 3 agents, k=1
    MDAP_MEDIUM = "mdap_medium" # 5 agents, k=1
    MAKER_FULL = "maker_full"   # 5 agents, k=2
    MAKER_ULTRA = "maker_ultra" # 7+ agents, k=3


@dataclass
class ComplexityScore:
    """Complexity score with component breakdown."""
    overall_score: float  # 0.0 to 1.0
    text_length_score: float
    domain_rarity_score: float
    depth_score: float
    historical_error_score: float
    dependency_score: float
    feature_weights: Dict[str, float]
    keyword_score: float = 0.0
    constraint_score: float = 0.0
    
    def __post_init__(self):
        """Validate scores are in valid range."""
        for name, value in [
            ("overall_score", self.overall_score),
            ("text_length_score", self.text_length_score),
            ("domain_rarity_score", self.domain_rarity_score),
            ("depth_score", self.depth_score),
            ("historical_error_score", self.historical_error_score),
            ("dependency_score", self.dependency_score),
            ("keyword_score", self.keyword_score),
            ("constraint_score", self.constraint_score),
        ]:
            if not 0.0 <= value <= 1.0:
                # Handle small floating point errors
                if -0.0001 < value < 0:
                    setattr(self, name, 0.0)
                elif 1.0 < value < 1.0001:
                    setattr(self, name, 1.0)
                else:
                    raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


@dataclass
class SolveConfig:
    """Configuration for solving a sub-problem."""
    strategy: SolveStrategy
    n_agents: int
    k_ahead: int
    max_retries: int
    timeout_ms: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.n_agents <= 0:
            raise ValueError(f"n_agents must be > 0, got {self.n_agents}")
        if self.k_ahead < 0:
            raise ValueError(f"k_ahead must be >= 0, got {self.k_ahead}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")


@dataclass
class AllocationDecision:
    """Decision made by the allocator."""
    complexity_score: float
    allocated_strategy: SolveStrategy
    config: SolveConfig
    estimated_cost: float
    estimated_quality: float
    timestamp: float


@dataclass
class ExecutionResult:
    """Result of executing a sub-problem."""
    subproblem_id: str
    success: bool
    solution: Optional[Any]
    strategy_used: SolveStrategy
    actual_cost: float
    latency_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SubProblem:
    """Represents a sub-problem to be solved."""
    id: str
    description: str
    domain: str
    depth: int
    dependencies: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}
