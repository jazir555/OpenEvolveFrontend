"""
Comparison Results Schemas

Data structures for storing and analyzing comparison results between
OpenEvolve and LoongFlow systems.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, UTC


class ComparisonCategory(Enum):
    """Categories for system comparison"""
    CONVERGENCE_SPEED = "convergence_speed"
    SOLUTION_QUALITY = "solution_quality"
    EVALUATION_EFFICIENCY = "evaluation_efficiency"
    DIVERSITY = "diversity"
    COMPUTATIONAL_COST = "computational_cost"
    SCALABILITY = "scalability"


class WinnerType(Enum):
    """Possible comparison winners"""
    OPENEVOLVE = "openevolve"
    LOONGFLOW = "loongflow"
    TIE = "tie"


class SynergyType(Enum):
    """Types of synergy opportunities"""
    TECHNIQUE_TRANSFER = "technique_transfer"
    PARAMETER_TUNING = "parameter_tuning"
    HYBRID_ARCHITECTURE = "hybrid_architecture"
    EVALUATION_STRATEGY = "evaluation_strategy"


class ComplexityLevel(Enum):
    """Implementation complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CategoryComparison:
    """
    Comparison within a single category

    Attributes:
        category: The category being compared
        openevolve_value: OpenEvolve's metric value
        loongflow_value: LoongFlow's metric value
        ratio: OE/LF ratio (or LF/OE for time-based metrics)
        winner: Which system won this category
        confidence: Statistical confidence (0-1)
        significance: Whether difference is statistically significant
    """
    category: ComparisonCategory
    openevolve_value: float
    loongflow_value: float
    ratio: float
    winner: WinnerType
    confidence: float
    significance: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "openevolve_value": self.openevolve_value,
            "loongflow_value": self.loongflow_value,
            "ratio": self.ratio,
            "winner": self.winner.value,
            "confidence": self.confidence,
            "significance": self.significance,
            "metadata": self.metadata
        }


@dataclass
class DetailedPerformanceComparison:
    """
    Comprehensive performance comparison between systems

    Extends PerformanceComparison with additional metadata
    and analysis capabilities
    """
    convergence_speed: CategoryComparison
    solution_quality: CategoryComparison
    evaluation_efficiency: CategoryComparison
    diversity: Dict[str, CategoryComparison]
    computational_cost: Dict[str, CategoryComparison]
    scalability: Optional[CategoryComparison] = None

    overall_winner: WinnerType = WinnerType.TIE
    overall_confidence: float = 0.0
    category_winners: Dict[str, WinnerType] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    domain: str = "general"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of comparison results"""
        return {
            "overall_winner": self.overall_winner.value,
            "confidence": self.overall_confidence,
            "category_winners": {
                k: v.value for k, v in self.category_winners.items()
            },
            "key_findings": self._extract_key_findings(),
            "recommendation": self._generate_recommendation()
        }

    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from comparison"""
        findings = []

        # Check for significant differences
        if self.convergence_speed.significance:
            better = "OpenEvolve" if self.convergence_speed.winner == WinnerType.OPENEVOLVE else "LoongFlow"
            pct = abs(1 - self.convergence_speed.ratio) * 100
            findings.append(f"{better} converged {pct:.1f}% faster")

        if self.solution_quality.significance:
            better = "OpenEvolve" if self.solution_quality.winner == WinnerType.OPENEVOLVE else "LoongFlow"
            pct = abs(1 - self.solution_quality.ratio) * 100
            findings.append(f"{better} achieved {pct:.1f}% better solution quality")

        if self.evaluation_efficiency.significance:
            better = "OpenEvolve" if self.evaluation_efficiency.winner == WinnerType.OPENEVOLVE else "LoongFlow"
            pct = abs(1 - self.evaluation_efficiency.ratio) * 100
            findings.append(f"{better} had {pct:.1f}% better evaluation efficiency")

        return findings

    def _generate_recommendation(self) -> str:
        """Generate recommendation based on comparison"""
        if self.overall_winner == WinnerType.TIE:
            return "Both systems perform similarly. Consider hybrid approach."

        winner_name = "OpenEvolve" if self.overall_winner == WinnerType.OPENEVOLVE else "LoongFlow"

        if self.overall_confidence > 0.8:
            return f"{winner_name} is strongly recommended for this domain."
        elif self.overall_confidence > 0.6:
            return f"{winner_name} has moderate advantage for this domain."
        else:
            return f"{winner_name} shows slight advantage. Hybrid approach may be best."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "convergence_speed": self.convergence_speed.to_dict(),
            "solution_quality": self.solution_quality.to_dict(),
            "evaluation_efficiency": self.evaluation_efficiency.to_dict(),
            "diversity": {k: v.to_dict() for k, v in self.diversity.items()},
            "computational_cost": {k: v.to_dict() for k, v in self.computational_cost.items()},
            "scalability": self.scalability.to_dict() if self.scalability else None,
            "overall_winner": self.overall_winner.value,
            "overall_confidence": self.overall_confidence,
            "category_winners": {k: v.value for k, v in self.category_winners.items()},
            "statistical_tests": self.statistical_tests,
            "domain": self.domain,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.get_summary()
        }


@dataclass
class SynergyOpportunityDetailed:
    """
    Detailed synergy opportunity between systems

    Extends SynergyOpportunity with implementation guidance
    """
    opportunity_type: SynergyType
    source_system: WinnerType
    target_system: WinnerType
    title: str
    description: str
    expected_improvement: float  # 0-1
    confidence: float  # 0-1
    complexity: ComplexityLevel
    priority: float  # 0-100

    # Implementation details
    implementation_steps: List[str] = field(default_factory=list)
    required_changes: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    estimated_effort: Optional[str] = None  # e.g., "2-3 weeks"
    dependencies: List[str] = field(default_factory=list)

    # Validation details
    validation_criteria: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_implementation_plan(self) -> Dict[str, Any]:
        """Get detailed implementation plan"""
        return {
            "title": self.title,
            "complexity": self.complexity.value,
            "estimated_effort": self.estimated_effort,
            "steps": self.implementation_steps,
            "changes_required": self.required_changes,
            "risks": self.potential_risks,
            "dependencies": self.dependencies,
            "validation": {
                "criteria": self.validation_criteria,
                "success_metrics": self.success_metrics
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "opportunity_type": self.opportunity_type.value,
            "source_system": self.source_system.value,
            "target_system": self.target_system.value,
            "title": self.title,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "complexity": self.complexity.value,
            "priority": self.priority,
            "implementation_plan": self.get_implementation_plan(),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class BestPracticeDetailed:
    """
    Detailed best practice from analysis

    Extends BestPractice with evidence and implementation guidance
    """
    practice: str
    source_system: WinnerType
    domain: str
    title: str
    description: str
    evidence: Dict[str, Any]

    # Supporting data
    supporting_data: List[Dict[str, Any]] = field(default_factory=list)
    statistical_significance: Optional[float] = None

    # Applicability
    applicable_domains: List[str] = field(default_factory=list)
    applicable_systems: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    # Implementation
    implementation_guidance: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    # Anti-patterns to avoid
    anti_patterns: List[str] = field(default_factory=list)

    confidence: float = 0.0
    priority: float = 50.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "practice": self.practice,
            "source_system": self.source_system.value,
            "domain": self.domain,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "supporting_data": self.supporting_data,
            "statistical_significance": self.statistical_significance,
            "applicability": {
                "domains": self.applicable_domains,
                "systems": self.applicable_systems,
                "constraints": self.constraints
            },
            "implementation": {
                "guidance": self.implementation_guidance,
                "examples": self.examples,
                "anti_patterns": self.anti_patterns
            },
            "confidence": self.confidence,
            "priority": self.priority,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class HybridRecommendationDetailed:
    """
    Detailed hybrid strategy recommendation

    Extends HybridStrategyRecommendation with detailed configuration
    """
    recommended_mode: str
    confidence: float
    rationale: str
    title: str
    description: str
    configuration: Dict[str, Any]
    expected_improvement: float

    # Parameter values
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    architecture: Optional[str] = None

    # Expected outcomes
    expected_benefits: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

    # Implementation
    implementation_phases: List[Dict[str, Any]] = field(default_factory=list)
    required_components: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)

    # Validation
    success_criteria: List[str] = field(default_factory=list)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    rollback_plan: Optional[str] = None

    # Alternatives considered
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    why_not_alternatives: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_implementation_roadmap(self) -> Dict[str, Any]:
        """Get detailed implementation roadmap"""
        return {
            "title": self.title,
            "phases": self.implementation_phases,
            "components": self.required_components,
            "integration": self.integration_points,
            "validation": {
                "success_criteria": self.success_criteria,
                "benchmarks": self.performance_benchmarks,
                "rollback": self.rollback_plan
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "recommended_mode": self.recommended_mode,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "title": self.title,
            "description": self.description,
            "configuration": self.configuration,
            "parameter_values": self.parameter_values,
            "architecture": self.architecture,
            "expected_outcomes": {
                "improvement": self.expected_improvement,
                "benefits": self.expected_benefits,
                "risks": self.risk_factors,
                "mitigations": self.mitigation_strategies
            },
            "implementation": self.get_implementation_roadmap(),
            "alternatives": {
                "considered": self.alternatives,
                "reasoning": self.why_not_alternatives
            },
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DualRunAnalysisReport:
    """
    Complete dual-run analysis report

    Combines all comparison data into comprehensive report
    """
    run_id: str
    domain: str
    problem_description: str

    # Core analysis
    performance_comparison: DetailedPerformanceComparison
    best_practices: List[BestPracticeDetailed]
    synergy_opportunities: List[SynergyOpportunityDetailed]
    hybrid_recommendation: HybridRecommendationDetailed

    # Metadata
    openevolve_config: Optional[Dict[str, Any]] = None
    loongflow_config: Optional[Dict[str, Any]] = None
    run_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Analysis metadata
    analyst_version: str = "1.0.0"
    analysis_duration: Optional[float] = None  # seconds

    def get_executive_summary(self) -> Dict[str, Any]:
        """Get executive summary for decision makers"""
        winner = self.performance_comparison.overall_winner.value
        confidence = self.performance_comparison.overall_confidence

        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "winner": winner,
            "confidence": confidence,
            "key_recommendation": self.hybrid_recommendation.title,
            "expected_improvement": f"{self.hybrid_recommendation.expected_improvement * 100:.1f}%",
            "top_synergies": len([s for s in self.synergy_opportunities if s.priority > 70]),
            "implementation_complexity": self._get_overall_complexity()
        }

    def get_technical_details(self) -> Dict[str, Any]:
        """Get technical details for engineers"""
        return {
            "performance_comparison": self.performance_comparison.to_dict(),
            "best_practices": [bp.to_dict() for bp in self.best_practices],
            "synergy_opportunities": [so.to_dict() for so in self.synergy_opportunities],
            "hybrid_recommendation": self.hybrid_recommendation.to_dict(),
            "configurations": {
                "openevolve": self.openevolve_config,
                "loongflow": self.loongflow_config
            }
        }

    def _get_overall_complexity(self) -> str:
        """Get overall implementation complexity"""
        if not self.synergy_opportunities:
            return "low"

        avg_complexity = sum(
            1 if s.complexity == ComplexityLevel.LOW else
            2 if s.complexity == ComplexityLevel.MEDIUM else 3
            for s in self.synergy_opportunities
        ) / len(self.synergy_opportunities)

        if avg_complexity < 1.5:
            return "low"
        elif avg_complexity < 2.5:
            return "medium"
        else:
            return "high"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete dictionary"""
        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "problem_description": self.problem_description,
            "executive_summary": self.get_executive_summary(),
            "technical_details": self.get_technical_details(),
            "metadata": {
                "openevolve_config": self.openevolve_config,
                "loongflow_config": self.loongflow_config,
                "run_metadata": self.run_metadata,
                "analyst_version": self.analyst_version,
                "analysis_duration": self.analysis_duration,
                "timestamp": self.timestamp.isoformat()
            }
        }
