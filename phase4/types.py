"""
RESE Phase IV: Shared Type Definitions

Common data types used across Phase IV modules to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ACIMeasurement:
    """
    Single ACI (Algorithmic Complexity Index) measurement.

    Attributes:
        timestamp: When the measurement was taken
        aci_value: The ACI value (0-1, lower is better)
        disorder_entropy: Disorder entropy component
        causal_coherence: Causal coherence component
        num_constraints: Number of constraints
        stage: Which stage this measurement represents
        metadata: Additional metadata about the measurement
    """
    timestamp: datetime
    aci_value: float
    disorder_entropy: float
    causal_coherence: float
    num_constraints: int
    stage: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ACI value is in valid range"""
        if not 0 <= self.aci_value <= 1:
            raise ValueError(f"ACI value must be between 0 and 1, got {self.aci_value}")


@dataclass
class Problem:
    """
    Problem to be solved

    Attributes:
        id: Unique identifier
        description: Problem description
        constraints: List of constraints from SCE
        variables: Dictionary of variables
        objective: Optional objective function
        domain: Problem domain
        metadata: Additional metadata
    """
    id: str
    description: str
    constraints: List[Any]  # From SCE
    variables: Dict[str, Any]
    objective: Optional[str] = None
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RESESolution:
    """
    Solution produced by RESE

    Attributes:
        problem_id: ID of the problem being solved
        solution: Solution dictionary
        aci_history: History of ACI values
        stage_results: Results from each stage
        metadata: Additional metadata
        timestamp: When solution was created
    """
    problem_id: str
    solution: Dict[str, Any]
    aci_history: List[float]
    stage_results: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ACIReduction:
    """
    Result of ACI reduction analysis.

    Attributes:
        baseline: Initial ACI measurement
        final: Final ACI measurement after intervention
        reduction: Absolute reduction in ACI
        relative_reduction: Relative reduction as percentage
        statistically_significant: Whether reduction is statistically significant
        confidence_interval: 95% confidence interval for reduction
        effect_size: Cohen's d effect size
    """
    baseline: ACIMeasurement
    final: ACIMeasurement
    reduction: float
    relative_reduction: float
    statistically_significant: bool
    confidence_interval: Optional[tuple] = None
    effect_size: Optional[float] = None
    p_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'baseline_aci': self.baseline.aci_value,
            'final_aci': self.final.aci_value,
            'reduction': self.reduction,
            'relative_reduction': self.relative_reduction,
            'statistically_significant': self.statistically_significant,
            'confidence_interval': self.confidence_interval,
            'effect_size': self.effect_size,
            'p_value': self.p_value
        }


__all__ = [
    'ACIMeasurement',
    'Problem',
    'RESESolution',
    'ACIReduction',
]
