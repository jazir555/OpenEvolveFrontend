"""
Data structures for End-to-End Invention Planning System.
Breaking circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

@dataclass
class InventionGoal:
    """Parsed invention goal from prompt"""
    goal_type: str  # "technology", "material", "device", "process", etc.
    target: str  # What is being invented
    domain: str  # "physics", "chemistry", "biology", "engineering", etc.
    key_requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    success_definition: str = "TBD"
    complexity_score: float = 0.5  # 0-1

@dataclass
class ValidatedMath:
    """Mathematical relationship formalized in Lean"""
    description: str
    lean_theorem: str
    lean_proof: str
    variables: Dict[str, str] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    verification_method: str = "unknown"
    confidence: float = 0.0

@dataclass
class ErrorSource:
    """Potential source of error"""
    error_type: str
    description: str
    probability: float  # Estimated probability
    impact: str  # "critical", "high", "medium", "low"
    mitigation_strategy: str
    verification_method: str
    acceptance_criteria: str

@dataclass
class PhysicsValidationReport:
    """Report from physics validation"""
    passed: bool
    confidence: float
    consistency_checks: Dict[str, bool]
    formal_verifications: List[Dict[str, Any]]
    error_sources: List[ErrorSource]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class InventionPlan:
    """Complete invention plan SOP"""
    invention_goal: InventionGoal
    decomposition: Dict[str, Any]
    formalized_math: List[ValidatedMath]
    physics_validation: PhysicsValidationReport
    error_analysis: List[ErrorSource]
    sop_document: str
    success_criteria: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
