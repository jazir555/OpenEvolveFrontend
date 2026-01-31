"""
Evolutionary Artifact Schemas

Data structures for knowledge artifacts extracted from evolutionary systems.
Defines canonical schemas for OpenEvolve and LoongFlow artifacts.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC


class ArtifactType(Enum):
    """Types of evolutionary knowledge artifacts"""
    SOLUTION_PATTERN = "solution_pattern"
    EVOLUTIONARY_TRAJECTORY = "evolutionary_trajectory"
    MAP_ELITES_ARCHIVE = "map_elites_archive"
    PES_PATTERNS = "pes_patterns"
    PARAMETER_EFFECTIVENESS = "parameter_effectiveness"
    SUMMARY_INSIGHTS = "summary_insights"
    PERFORMANCE_METRICS = "performance_metrics"
    EVOLUTIONARY_TREE = "evolutionary_tree"
    BEST_PRACTICE = "best_practice"
    SYNERGY_OPPORTUNITY = "synergy_opportunity"


class SystemType(Enum):
    """Evolutionary system types"""
    OPENEVOLVE = "openevolve"
    LOONGFLOW = "loongflow"
    HYBRID = "hybrid"


class DomainType(Enum):
    """Problem domains"""
    FINANCE = "finance"
    TRADING = "trading"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    PHARMA = "pharma"
    WEB = "web"
    GENERAL = "general"


@dataclass
class SolutionPatternArtifact:
    """
    Best solution pattern artifact

    Captures the structure and characteristics of high-quality solutions
    """
    solution_code: str
    fitness_score: float
    iteration_found: int
    system_type: SystemType
    domain: DomainType
    characteristics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.SOLUTION_PATTERN.value,
            "solution_code": self.solution_code,
            "fitness_score": self.fitness_score,
            "iteration_found": self.iteration_found,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "characteristics": self.characteristics,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EvolutionaryTrajectoryArtifact:
    """
    Evolutionary trajectory artifact

    Tracks the optimization path over time
    """
    history: List[Dict[str, Any]]
    improvement_rate: float
    convergence_point: Optional[int] = None
    system_type: SystemType = SystemType.OPENEVOLVE
    domain: DomainType = DomainType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.EVOLUTIONARY_TRAJECTORY.value,
            "history": self.history,
            "improvement_rate": self.improvement_rate,
            "convergence_point": self.convergence_point,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MAPElitesArchiveArtifact:
    """
    MAP-Elites archive artifact (OpenEvolve)

    Captures behavioral space coverage and diverse solutions
    """
    feature_dimensions: List[str]
    feature_bins: Dict[str, int]
    archive_coverage: float  # 0-1
    cell_occupancy: Dict[str, int]
    diverse_solutions: List[Dict[str, Any]]
    system_type: SystemType = SystemType.OPENEVOLVE
    domain: DomainType = DomainType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.MAP_ELITES_ARCHIVE.value,
            "feature_dimensions": self.feature_dimensions,
            "feature_bins": self.feature_bins,
            "archive_coverage": self.archive_coverage,
            "cell_occupancy": self.cell_occupancy,
            "diverse_solutions": self.diverse_solutions,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PESPatternsArtifact:
    """
    PES patterns artifact (LoongFlow)

    Captures Plan-Execute-Summarize patterns
    """
    num_generations: int
    planning_strategies: List[str]
    execution_patterns: List[str]
    summary_insights: List[str]
    system_type: SystemType = SystemType.LOONGFLOW
    domain: DomainType = DomainType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.PES_PATTERNS.value,
            "num_generations": self.num_generations,
            "planning_strategies": self.planning_strategies,
            "execution_patterns": self.execution_patterns,
            "summary_insights": self.summary_insights,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ParameterEffectivenessArtifact:
    """
    Parameter effectiveness artifact

    Identifies which parameters were most effective
    """
    config: Dict[str, Any]
    effective_parameters: Dict[str, Any]
    sensitivity_analysis: Optional[Dict[str, float]] = None
    system_type: SystemType = SystemType.OPENEVOLVE
    domain: DomainType = DomainType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.PARAMETER_EFFECTIVENESS.value,
            "config": self.config,
            "effective_parameters": self.effective_parameters,
            "sensitivity_analysis": self.sensitivity_analysis,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PerformanceMetricsArtifact:
    """
    Performance metrics artifact

    Comprehensive performance data from evolutionary run
    """
    total_evaluations: int
    best_fitness: float
    convergence_generation: Optional[int]
    sample_efficiency: float  # fitness per evaluation
    computational_cost: Dict[str, float]
    diversity_metrics: Optional[Dict[str, float]] = None
    system_type: SystemType = SystemType.LOONGFLOW
    domain: DomainType = DomainType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.PERFORMANCE_METRICS.value,
            "total_evaluations": self.total_evaluations,
            "best_fitness": self.best_fitness,
            "convergence_generation": self.convergence_generation,
            "sample_efficiency": self.sample_efficiency,
            "computational_cost": self.computational_cost,
            "diversity_metrics": self.diversity_metrics,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class EvolutionaryTreeArtifact:
    """
    Evolutionary tree artifact (LoongFlow)

    Captures the ancestry and branching patterns
    """
    root_id: str
    num_generations: int
    branching_factor: float
    best_path: List[str]
    all_solutions: List[Dict[str, Any]]
    system_type: SystemType = SystemType.LOONGFLOW
    domain: DomainType = DomainType.GENERAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": ArtifactType.EVOLUTIONARY_TREE.value,
            "root_id": self.root_id,
            "num_generations": self.num_generations,
            "branching_factor": self.branching_factor,
            "best_path": self.best_path,
            "all_solutions": self.all_solutions,
            "system_type": self.system_type.value,
            "domain": self.domain.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


# Factory function for creating artifacts from dictionaries
def create_artifact_from_dict(data: Dict[str, Any]) -> Union[
    SolutionPatternArtifact,
    EvolutionaryTrajectoryArtifact,
    MAPElitesArchiveArtifact,
    PESPatternsArtifact,
    ParameterEffectivenessArtifact,
    PerformanceMetricsArtifact,
    EvolutionaryTreeArtifact
]:
    """
    Create appropriate artifact from dictionary data

    Args:
        data: Dictionary containing artifact data

    Returns:
        Appropriate artifact instance

    Raises:
        ValueError: If artifact type is unknown
    """
    artifact_type = data.get("artifact_type")

    if artifact_type == ArtifactType.SOLUTION_PATTERN.value:
        return SolutionPatternArtifact(**data)
    elif artifact_type == ArtifactType.EVOLUTIONARY_TRAJECTORY.value:
        return EvolutionaryTrajectoryArtifact(**data)
    elif artifact_type == ArtifactType.MAP_ELITES_ARCHIVE.value:
        return MAPElitesArchiveArtifact(**data)
    elif artifact_type == ArtifactType.PES_PATTERNS.value:
        return PESPatternsArtifact(**data)
    elif artifact_type == ArtifactType.PARAMETER_EFFECTIVENESS.value:
        return ParameterEffectivenessArtifact(**data)
    elif artifact_type == ArtifactType.PERFORMANCE_METRICS.value:
        return PerformanceMetricsArtifact(**data)
    elif artifact_type == ArtifactType.EVOLUTIONARY_TREE.value:
        return EvolutionaryTreeArtifact(**data)
    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")
