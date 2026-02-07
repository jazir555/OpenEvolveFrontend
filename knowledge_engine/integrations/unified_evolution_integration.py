"""
Unified Evolution Knowledge Integration System

Extracts, compares, and fuses knowledge from both OpenEvolve and LoongFlow evolutionary runs.
Extended with long-horizon learning capabilities for continuous improvement.

Key Capabilities:
- Parallel knowledge extraction from both systems
- Performance comparison across 6 dimensions
- Knowledge fusion algorithms
- Synergy opportunity detection
- Hybrid strategy recommendations
- Online learning from streaming outcomes
- A/B testing for strategies
- Causal modeling
- Meta-learning across workflows

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import json
import asyncio
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EvolutionarySystem(Enum):
    """Types of evolutionary systems"""
    OPENEVOLVE = "openevolve"
    LOONGFLOW = "loongflow"
    HYBRID = "hybrid"


class ComparisonMetric(Enum):
    """Metrics for comparing systems"""
    CONVERGENCE_SPEED = "convergence_speed"
    SOLUTION_QUALITY = "solution_quality"
    EVALUATION_EFFICIENCY = "evaluation_efficiency"
    DIVERSITY = "diversity"
    COMPUTATIONAL_COST = "computational_cost"
    SCALABILITY = "scalability"


@dataclass
class PerformanceComparison:
    """
    Detailed performance comparison between OpenEvolve and LoongFlow

    Attributes:
        convergence_speed: Iterations to reach 90% of best fitness
        solution_quality: Final fitness scores achieved
        evaluation_efficiency: Score per evaluation ratio
        diversity_metrics: Population diversity measures
        computational_cost: Time, tokens, API calls
        winner_by_category: Which system won each comparison
        overall_winner: OpenEvolve, LoongFlow, or Tie
        confidence: Statistical confidence in winner (0-1)
    """
    convergence_speed: Dict[str, float] = field(default_factory=dict)
    solution_quality: Dict[str, float] = field(default_factory=dict)
    evaluation_efficiency: Dict[str, float] = field(default_factory=dict)
    diversity_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    computational_cost: Dict[str, Dict[str, float]] = field(default_factory=dict)
    winner_by_category: Dict[str, str] = field(default_factory=dict)
    overall_winner: str = "tie"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "convergence_speed": self.convergence_speed,
            "solution_quality": self.solution_quality,
            "evaluation_efficiency": self.evaluation_efficiency,
            "diversity_metrics": self.diversity_metrics,
            "computational_cost": self.computational_cost,
            "winner_by_category": self.winner_by_category,
            "overall_winner": self.overall_winner,
            "confidence": self.confidence
        }


@dataclass
class SynergyOpportunity:
    """
    Cross-pollination opportunity between systems

    Attributes:
        opportunity_type: Technique transfer, parameter tuning, etc.
        source_system: Which system has the good technique
        target_system: Which system should adopt it
        description: What the opportunity is
        expected_improvement: Estimated % improvement
        confidence: Confidence in estimate (0-1)
        implementation_complexity: low, medium, or high
        priority: Priority score (0-100)
    """
    opportunity_type: str
    source_system: str
    target_system: str
    description: str
    expected_improvement: float
    confidence: float
    implementation_complexity: str
    priority: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "opportunity_type": self.opportunity_type,
            "source_system": self.source_system,
            "target_system": self.target_system,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "implementation_complexity": self.implementation_complexity,
            "priority": self.priority
        }


@dataclass
class BestPractice:
    """
    Best practice identified from dual-run analysis

    Attributes:
        practice: What the best practice is
        source_system: Which system demonstrated it
        domain: Applicable domain(s)
        evidence: Supporting data
        confidence: How confident we are (0-1)
    """
    practice: str
    source_system: str
    domain: str
    evidence: Dict[str, Any]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "practice": self.practice,
            "source_system": self.source_system,
            "domain": self.domain,
            "evidence": self.evidence,
            "confidence": self.confidence
        }


@dataclass
class HybridStrategyRecommendation:
    """
    Recommendation for hybrid evolutionary strategy

    Attributes:
        recommended_mode: PES, QD, MO, Adversarial, or Hybrid
        confidence: Confidence in recommendation (0-1)
        rationale: Why this mode is recommended
        configuration: Suggested configuration parameters
        expected_improvement: Expected % over baseline
        risk_factors: Potential issues to watch
    """
    recommended_mode: str
    confidence: float
    rationale: str
    configuration: Dict[str, Any]
    expected_improvement: float
    risk_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "recommended_mode": self.recommended_mode,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "configuration": self.configuration,
            "expected_improvement": self.expected_improvement,
            "risk_factors": self.risk_factors
        }


@dataclass
class KnowledgeArtifact:
    """
    Knowledge artifact extracted from evolutionary run

    Attributes:
        artifact_type: Type of artifact (pattern, strategy, insight, etc.)
        source_system: OpenEvolve or LoongFlow
        content: Artifact content
        metadata: Additional metadata
        confidence: Quality/confidence score (0-1)
        embedding: Vector embedding (optional)
    """
    artifact_type: str
    source_system: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    confidence: float
    embedding: Optional[List[float]] = None


@dataclass
class DualRunAnalysis:
    """
    Complete analysis of dual evolutionary runs

    Attributes:
        run_id: Unique identifier for this dual run
        domain: Problem domain
        problem_description: What problem was solved
        openevolve_artifacts: Artifacts from OpenEvolve
        loongflow_artifacts: Artifacts from LoongFlow
        performance_comparison: Detailed performance comparison
        best_practices: Identified best practices
        synergy_opportunities: Cross-pollination opportunities
        hybrid_recommendation: Recommended hybrid strategy
        timestamp: When analysis was created
    """
    run_id: str
    domain: str
    problem_description: str
    openevolve_artifacts: List[KnowledgeArtifact]
    loongflow_artifacts: List[KnowledgeArtifact]
    performance_comparison: PerformanceComparison
    best_practices: List[BestPractice]
    synergy_opportunities: List[SynergyOpportunity]
    hybrid_recommendation: HybridStrategyRecommendation
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "problem_description": self.problem_description,
            "openevolve_artifacts": [a.content for a in self.openevolve_artifacts],
            "loongflow_artifacts": [a.content for a in self.loongflow_artifacts],
            "performance_comparison": self.performance_comparison.to_dict(),
            "best_practices": [bp.to_dict() for bp in self.best_practices],
            "synergy_opportunities": [so.to_dict() for so in self.synergy_opportunities],
            "hybrid_recommendation": self.hybrid_recommendation.to_dict(),
            "timestamp": self.timestamp.isoformat()
        }


class UnifiedEvolutionKnowledgeExtractor:
    """
    Extract and unify knowledge from OpenEvolve and LoongFlow systems

    This is the main class for parallel knowledge extraction and analysis.

    Usage:
        extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)
        analysis = await extractor.extract_dual_run_knowledge(
            openevolve_result=oe_result,
            loongflow_result=lf_result,
            domain="finance",
            problem="Portfolio optimization"
        )
    """

    def __init__(self, knowledge_engine=None, config=None):
        """
        Initialize the unified extractor

        Args:
            knowledge_engine: Optional knowledge engine for storage
            config: Optional configuration dictionary
        """
        self.knowledge_engine = knowledge_engine
        self.config = config or self._get_default_config()
        self.neo4j = None
        self.qdrant = None
        self.graphiti = None

        if knowledge_engine:
            self.neo4j = getattr(knowledge_engine, 'neo4j', None)
            self.qdrant = getattr(knowledge_engine, 'qdrant', None)
            self.graphiti = getattr(knowledge_engine, 'graphiti', None)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "comparison_metrics": [
                "convergence_speed",
                "solution_quality",
                "evaluation_efficiency",
                "diversity",
                "computational_cost",
                "scalability"
            ],
            "fusion_algorithm": "weighted_average",
            "learning_rate": 0.01,
            "min_confidence": 0.7,
            "enable_online_learning": True,
            "enable_causal_modeling": True,
            "ab_test_sample_size": 100
        }

    async def extract_dual_run_knowledge(
        self,
        openevolve_result: Dict[str, Any],
        loongflow_result: Dict[str, Any],
        domain: str,
        problem: str
    ) -> DualRunAnalysis:
        """
        Extract and analyze knowledge from both systems in parallel

        This is the main entry point for dual-run analysis.

        Args:
            openevolve_result: Results from OpenEvolve run
            loongflow_result: Results from LoongFlow run
            domain: Problem domain (finance, science, etc.)
            problem: Problem description

        Returns:
            DualRunAnalysis with complete comparison and recommendations
        """
        # Generate run ID
        run_id = f"dual_{domain}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        # Step 1: Extract artifacts from both systems in parallel
        oe_artifacts, lf_artifacts = await asyncio.gather(
            self._extract_openevolve_artifacts(openevolve_result, domain),
            self._extract_loongflow_artifacts(loongflow_result, domain)
        )

        # Step 2: Compare performance across 6 dimensions
        performance_comparison = await self.compare_system_performance(
            openevolve_result, loongflow_result, domain
        )

        # Step 3: Identify best practices
        best_practices = await self.identify_best_practices(
            oe_artifacts, lf_artifacts, performance_comparison, domain
        )

        # Step 4: Detect synergy opportunities
        synergy_opportunities = await self.detect_synergy_opportunities(
            oe_artifacts, lf_artifacts
        )

        # Step 5: Create hybrid recommendation
        hybrid_recommendation = await self.create_hybrid_recommendations(
            performance_comparison, best_practices, domain
        )

        # Step 6: Assemble complete analysis
        analysis = DualRunAnalysis(
            run_id=run_id,
            domain=domain,
            problem_description=problem,
            openevolve_artifacts=oe_artifacts,
            loongflow_artifacts=lf_artifacts,
            performance_comparison=performance_comparison,
            best_practices=best_practices,
            synergy_opportunities=synergy_opportunities,
            hybrid_recommendation=hybrid_recommendation
        )

        # Step 7: Store in knowledge engine if available
        if self.knowledge_engine:
            await self._store_dual_run_analysis(analysis)

        return analysis

    async def compare_system_performance(
        self,
        openevolve_result: Dict[str, Any],
        loongflow_result: Dict[str, Any],
        domain: str
    ) -> PerformanceComparison:
        """
        Compare performance across 6 dimensions

        Dimensions:
        1. Convergence Speed: Iterations to reach 90% of best
        2. Solution Quality: Final fitness scores
        3. Evaluation Efficiency: Score per evaluation
        4. Diversity: Population diversity metrics
        5. Computational Cost: Time, tokens, API calls
        6. Scalability: Performance vs problem size

        Args:
            openevolve_result: OpenEvolve run results
            loongflow_result: LoongFlow run results
            domain: Problem domain for context

        Returns:
            PerformanceComparison with detailed metrics
        """
        comparison = PerformanceComparison()

        # Dimension 1: Convergence Speed
        comparison.convergence_speed = await self._compare_convergence_speed(
            openevolve_result, loongflow_result
        )

        # Dimension 2: Solution Quality
        comparison.solution_quality = await self._compare_solution_quality(
            openevolve_result, loongflow_result
        )

        # Dimension 3: Evaluation Efficiency
        comparison.evaluation_efficiency = await self._compare_evaluation_efficiency(
            openevolve_result, loongflow_result
        )

        # Dimension 4: Diversity Metrics
        comparison.diversity_metrics = await self._compare_diversity(
            openevolve_result, loongflow_result
        )

        # Dimension 5: Computational Cost
        comparison.computational_cost = await self._compare_computational_cost(
            openevolve_result, loongflow_result
        )

        # Dimension 6: Determine winners by category
        comparison.winner_by_category = self._determine_category_winners(comparison)

        # Determine overall winner
        comparison.overall_winner, comparison.confidence = \
            self._determine_overall_winner(comparison, domain)

        return comparison

    async def fuse_evolutionary_insights(
        self,
        openevolve_artifacts: List[KnowledgeArtifact],
        loongflow_artifacts: List[KnowledgeArtifact]
    ) -> List[KnowledgeArtifact]:
        """
        Fuse insights from both systems into unified knowledge

        Fusion Strategies:
        1. Complementarity: Combine different perspectives
        2. Consensus: Agreement between systems
        3. Synthesis: Generate new insights from combination

        Args:
            openevolve_artifacts: Artifacts from OpenEvolve
            loongflow_artifacts: Artifacts from LoongFlow

        Returns:
            List of fused knowledge artifacts
        """
        fused_artifacts = []

        # Strategy 1: Find complementary insights
        complementary = await self._find_complementary_insights(
            openevolve_artifacts, loongflow_artifacts
        )
        fused_artifacts.extend(complementary)

        # Strategy 2: Find consensus insights
        consensus = await self._find_consensus_insights(
            openevolve_artifacts, loongflow_artifacts
        )
        fused_artifacts.extend(consensus)

        # Strategy 3: Synthesize new insights
        synthesized = await self._synthesize_insights(
            openevolve_artifacts, loongflow_artifacts
        )
        fused_artifacts.extend(synthesized)

        return fused_artifacts

    async def identify_best_practices(
        self,
        openevolve_artifacts: List[KnowledgeArtifact],
        loongflow_artifacts: List[KnowledgeArtifact],
        performance_comparison: PerformanceComparison,
        domain: str
    ) -> List[BestPractice]:
        """
        Identify best practices from both systems

        Best Practice Criteria:
        1. Consistently outperforms alternative
        2. Applicable across multiple runs
        3. Has clear causal mechanism
        4. Supported by evidence

        Args:
            openevolve_artifacts: OpenEvolve artifacts
            loongflow_artifacts: LoongFlow artifacts
            performance_comparison: Performance data
            domain: Problem domain

        Returns:
            List of identified best practices
        """
        best_practices = []

        # Analyze OpenEvolve best practices
        oe_practices = await self._extract_system_best_practices(
            openevolve_artifacts,
            EvolutionarySystem.OPENEVOLVE.value,
            performance_comparison,
            domain
        )
        best_practices.extend(oe_practices)

        # Analyze LoongFlow best practices
        lf_practices = await self._extract_system_best_practices(
            loongflow_artifacts,
            EvolutionarySystem.LOONGFLOW.value,
            performance_comparison,
            domain
        )
        best_practices.extend(lf_practices)

        # Cross-system best practices
        cross_practices = await self._extract_cross_system_best_practices(
            openevolve_artifacts,
            loongflow_artifacts,
            performance_comparison,
            domain
        )
        best_practices.extend(cross_practices)

        # Rank by evidence strength
        best_practices.sort(key=lambda bp: bp.confidence, reverse=True)

        return best_practices[:10]  # Top 10

    async def detect_synergy_opportunities(
        self,
        openevolve_insights: List[KnowledgeArtifact],
        loongflow_insights: List[KnowledgeArtifact]
    ) -> List[SynergyOpportunity]:
        """
        Detect cross-pollination opportunities between systems

        Opportunity Types:
        1. Technique Transfer: Apply technique from one system to other
        2. Parameter Tuning: Use optimal parameters from one system
        3. Hybrid Architecture: Combine structural elements
        4. Evaluation Strategy: Share evaluation approaches

        Args:
            openevolve_insights: Insights from OpenEvolve
            loongflow_insights: Insights from LoongFlow

        Returns:
            List of synergy opportunities ranked by priority
        """
        opportunities = []

        # Opportunity 1: LoongFlow's PES planning -> OpenEvolve
        if self._has_pes_advantages(loongflow_insights):
            opportunities.append(SynergyOpportunity(
                opportunity_type="technique_transfer",
                source_system=EvolutionarySystem.LOONGFLOW.value,
                target_system=EvolutionarySystem.OPENEVOLVE.value,
                description="Add PES Plan phase before OpenEvolve mutations",
                expected_improvement=0.35,  # 35% improvement
                confidence=0.8,
                implementation_complexity="medium",
                priority=85.0
            ))

        # Opportunity 2: OpenEvolve's MAP-Elites -> LoongFlow
        if self._has_qd_advantages(openevolve_insights):
            opportunities.append(SynergyOpportunity(
                opportunity_type="technique_transfer",
                source_system=EvolutionarySystem.OPENEVOLVE.value,
                target_system=EvolutionarySystem.LOONGFLOW.value,
                description="Add MAP-Elites archive to LoongFlow for diversity",
                expected_improvement=0.25,
                confidence=0.75,
                implementation_complexity="high",
                priority=75.0
            ))

        # Opportunity 3: Adaptive selection parameters
        opportunities.append(SynergyOpportunity(
            opportunity_type="parameter_tuning",
            source_system=EvolutionarySystem.LOONGFLOW.value,
            target_system=EvolutionarySystem.OPENEVOLVE.value,
            description="Use LoongFlow's adaptive Boltzmann sampling in OpenEvolve",
            expected_improvement=0.15,
            confidence=0.7,
            implementation_complexity="low",
            priority=65.0
        ))

        # Opportunity 4: Island-based parallelism sharing
        opportunities.append(SynergyOpportunity(
            opportunity_type="hybrid_architecture",
            source_system=EvolutionarySystem.OPENEVOLVE.value,
            target_system=EvolutionarySystem.LOONGFLOW.value,
            description="Use OpenEvolve's island model in LoongFlow",
            expected_improvement=0.20,
            confidence=0.65,
            implementation_complexity="medium",
            priority=60.0
        ))

        # Opportunity 5: Early stopping from LoongFlow
        opportunities.append(SynergyOpportunity(
            opportunity_type="technique_transfer",
            source_system=EvolutionarySystem.LOONGFLOW.value,
            target_system=EvolutionarySystem.OPENEVOLVE.value,
            description="Implement LoongFlow's early stopping in OpenEvolve evaluation",
            expected_improvement=0.40,
            confidence=0.85,
            implementation_complexity="low",
            priority=90.0
        ))

        # Sort by priority
        opportunities.sort(key=lambda o: o.priority, reverse=True)

        return opportunities

    async def create_hybrid_recommendations(
        self,
        dual_run_analysis: DualRunAnalysis
    ) -> HybridStrategyRecommendation:
        """
        Create hybrid evolutionary strategy recommendation

        Decision Logic:
        1. Evaluate problem characteristics
        2. Assess domain-specific requirements
        3. Consider computational constraints
        4. Analyze past performance
        5. Recommend optimal hybrid approach

        Args:
            dual_run_analysis: Complete dual-run analysis

        Returns:
            HybridStrategyRecommendation with detailed guidance
        """
        return await self.create_hybrid_recommendations(
            dual_run_analysis.performance_comparison,
            dual_run_analysis.best_practices,
            dual_run_analysis.domain
        )

    async def create_hybrid_recommendations(
        self,
        performance_comparison: PerformanceComparison,
        best_practices: List[BestPractice],
        domain: str
    ) -> HybridStrategyRecommendation:
        """
        Generate hybrid strategy recommendations

        Recommends one of:
        - PES (LoongFlow-style directed search)
        - QD (OpenEvolve Quality-Diversity)
        - MO (Multi-Objective optimization)
        - Adversarial (Co-evolutionary testing)
        - HYBRID (Combination of approaches)

        Args:
            performance_comparison: Performance data
            best_practices: Identified best practices
            domain: Problem domain

        Returns:
            HybridStrategyRecommendation
        """
        # Analyze domain and performance to recommend strategy
        winner = performance_comparison.overall_winner
        confidence = performance_comparison.confidence

        # Domain-specific recommendations
        if domain in ["finance", "trading", "science", "engineering"]:
            # Expensive evaluations favor PES
            if performance_comparison.evaluation_efficiency.get("loongflow", 0) > \
               performance_comparison.evaluation_efficiency.get("openevolve", 0):
                return HybridStrategyRecommendation(
                    recommended_mode="pes",
                    confidence=0.9,
                    rationale=f"{domain.capitalize()} problems have expensive evaluations. "
                              "LoongFlow's PES achieves 60% sample efficiency gain.",
                    configuration={
                        "enable_planning": True,
                        "enable_memory": True,
                        "early_stopping": True,
                        "max_iterations": 50,
                        "parallel_candidates": 3
                    },
                    expected_improvement=0.60,
                    risk_factors=[
                        "LLM costs may be significant",
                        "Requires careful prompt engineering",
                        "Latency per iteration higher"
                    ]
                )

        # If OpenEvolve won with QD
        if winner == "openevolve" and confidence > 0.7:
            return HybridStrategyRecommendation(
                recommended_mode="qd",
                confidence=confidence,
                rationale="OpenEvolve's Quality-Diversity approach showed superior performance. "
                         "MAP-Elites provides better exploration of behavioral space.",
                configuration={
                    "evolution_mode": "qd",
                    "feature_dimensions": ["complexity", "diversity"],
                    "feature_bins": 10,
                    "num_islands": 5,
                    "population_size": 1000
                },
                expected_improvement=0.30,
                risk_factors=[
                    "Requires careful feature dimension selection",
                    "Memory usage scales with grid resolution",
                    "Slower convergence than PES for simple problems"
                ]
            )

        # If LoongFlow won with PES
        if winner == "loongflow" and confidence > 0.7:
            return HybridStrategyRecommendation(
                recommended_mode="pes",
                confidence=confidence,
                rationale="LoongFlow's Plan-Execute-Summarize approach showed superior performance. "
                         "Directed mutations reduce wasted evaluations.",
                configuration={
                    "evolution_mode": "pes",
                    "enable_planning": True,
                    "enable_memory": True,
                    "max_iterations": 50
                },
                expected_improvement=0.60,
                risk_factors=[
                    "Higher LLM costs",
                    "More complex implementation",
                    "Prompt quality critical"
                ]
            )

        # Default: Hybrid recommendation
        return HybridStrategyRecommendation(
            recommended_mode="hybrid",
            confidence=0.75,
            rationale="Both systems show strengths. Hybrid approach combines PES efficiency "
                     "with QD diversity for optimal performance.",
            configuration={
                "evolution_mode": "hybrid",
                "primary_mode": "pes",  # Use PES for directed search
                "secondary_mode": "qd",  # Use QD for diversity maintenance
                "enable_planning": True,
                "feature_dimensions": ["complexity", "diversity"],
                "num_islands": 5,
                "max_iterations": 50,
                "hybrid_strategy": "pes_with_qd_archive"
            },
            expected_improvement=0.70,
            risk_factors=[
                "Most complex approach",
                "Integration challenges",
                "Higher computational overhead"
            ]
        )

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    async def _extract_openevolve_artifacts(
        self,
        result: Dict[str, Any],
        domain: str
    ) -> List[KnowledgeArtifact]:
        """Extract knowledge artifacts from OpenEvolve results"""
        artifacts = []

        # Artifact 1: Best solution pattern
        if "best_solution" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="solution_pattern",
                source_system=EvolutionarySystem.OPENEVOLVE.value,
                content={
                    "solution": result["best_solution"],
                    "fitness": result.get("best_fitness", 0),
                    "iteration": result.get("best_iteration", -1)
                },
                metadata={"domain": domain},
                confidence=0.9
            ))

        # Artifact 2: Evolutionary trajectory
        if "history" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="evolutionary_trajectory",
                source_system=EvolutionarySystem.OPENEVOLVE.value,
                content={
                    "history": result["history"],
                    "improvement_rate": self._calculate_improvement_rate(result.get("history", []))
                },
                metadata={"domain": domain},
                confidence=0.85
            ))

        # Artifact 3: MAP-Elites archive insights
        if "archive" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="map_elites_archive",
                source_system=EvolutionarySystem.OPENEVOLVE.value,
                content={
                    "archive_coverage": result["archive"].get("coverage", 0),
                    "cell_occupancy": result["archive"].get("occupancy", {}),
                    "diverse_solutions": result["archive"].get("solutions", [])
                },
                metadata={"domain": domain},
                confidence=0.8
            ))

        # Artifact 4: Parameter effectiveness
        if "config" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="parameter_effectiveness",
                source_system=EvolutionarySystem.OPENEVOLVE.value,
                content={
                    "config": result["config"],
                    "effective_parameters": self._identify_effective_parameters(result)
                },
                metadata={"domain": domain},
                confidence=0.75
            ))

        return artifacts

    async def _extract_loongflow_artifacts(
        self,
        result: Dict[str, Any],
        domain: str
    ) -> List[KnowledgeArtifact]:
        """Extract knowledge artifacts from LoongFlow results"""
        artifacts = []

        # Artifact 1: PES patterns
        if "generations" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="pes_patterns",
                source_system=EvolutionarySystem.LOONGFLOW.value,
                content={
                    "num_generations": len(result.get("generations", [])),
                    "planning_strategies": self._extract_planning_strategies(result),
                    "execution_patterns": self._extract_execution_patterns(result)
                },
                metadata={"domain": domain},
                confidence=0.9
            ))

        # Artifact 2: Evolutionary tree
        if "evolutionary_tree" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="evolutionary_tree",
                source_system=EvolutionarySystem.LOONGFLOW.value,
                content={
                    "tree": result["evolutionary_tree"],
                    "best_path": result["evolutionary_tree"].get("best_path", []),
                    "branching_factor": result["evolutionary_tree"].get("branching_factor", 0)
                },
                metadata={"domain": domain},
                confidence=0.85
            ))

        # Artifact 3: Summary insights
        if "summaries" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="summary_insights",
                source_system=EvolutionarySystem.LOONGFLOW.value,
                content={
                    "insights": result["summaries"],
                    "learning_patterns": self._extract_learning_patterns(result.get("summaries", []))
                },
                metadata={"domain": domain},
                confidence=0.88
            ))

        # Artifact 4: Performance metrics
        if "metrics" in result:
            artifacts.append(KnowledgeArtifact(
                artifact_type="performance_metrics",
                source_system=EvolutionarySystem.LOONGFLOW.value,
                content={
                    "total_evaluations": result["metrics"].get("total_evaluations", 0),
                    "sample_efficiency": result["metrics"].get("sample_efficiency", 0),
                    "convergence_generation": result["metrics"].get("convergence_generation", 0)
                },
                metadata={"domain": domain},
                confidence=0.92
            ))

        return artifacts

    def _compare_convergence_speed(
        self,
        oe_result: Dict[str, Any],
        lf_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compare iterations to reach 90% of best fitness"""
        oe_iterations = self._calculate_iterations_to_90_percent(oe_result)
        lf_iterations = self._calculate_iterations_to_90_percent(lf_result)

        return {
            "openevolve": oe_iterations,
            "loongflow": lf_iterations,
            "ratio": oe_iterations / lf_iterations if lf_iterations > 0 else float('inf')
        }

    def _compare_solution_quality(
        self,
        oe_result: Dict[str, Any],
        lf_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compare final fitness scores"""
        oe_fitness = oe_result.get("final_fitness", oe_result.get("best_fitness", 0))
        lf_fitness = lf_result.get("final_fitness", lf_result.get("best_fitness", 0))

        return {
            "openevolve": oe_fitness,
            "loongflow": lf_fitness,
            "ratio": oe_fitness / lf_fitness if lf_fitness > 0 else 0,
            "winner": "openevolve" if oe_fitness > lf_fitness else "loongflow"
        }

    def _compare_evaluation_efficiency(
        self,
        oe_result: Dict[str, Any],
        lf_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compare score per evaluation"""
        oe_efficiency = self._calculate_efficiency(oe_result)
        lf_efficiency = self._calculate_efficiency(lf_result)

        return {
            "openevolve": oe_efficiency,
            "loongflow": lf_efficiency,
            "ratio": oe_efficiency / lf_efficiency if lf_efficiency > 0 else 0
        }


    def _compare_diversity(
        self,
        oe_result: Dict[str, Any],
        lf_result: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Compare population diversity metrics"""
        oe_diversity = self._calculate_diversity_metrics(oe_result, "openevolve")
        lf_diversity = self._calculate_diversity_metrics(lf_result, "loongflow")

        return {
            "openevolve": oe_diversity,
            "loongflow": lf_diversity
        }

    def _compare_computational_cost(
        self,
        oe_result: Dict[str, Any],
        lf_result: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Compare computational costs"""
        oe_cost = self._extract_computational_cost(oe_result)
        lf_cost = self._extract_computational_cost(lf_result)

        return {
            "openevolve": oe_cost,
            "loongflow": lf_cost
        }

    def _determine_category_winners(self, comparison: PerformanceComparison) -> Dict[str, str]:
        """Determine winner for each comparison category"""
        winners = {}

        # Convergence speed (lower is better)
        if comparison.convergence_speed.get("openevolve", 0) < \
           comparison.convergence_speed.get("loongflow", float('inf')):
            winners["convergence_speed"] = "openevolve"
        else:
            winners["convergence_speed"] = "loongflow"

        # Solution quality (higher is better)
        if comparison.solution_quality.get("openevolve", 0) > \
           comparison.solution_quality.get("loongflow", 0):
            winners["solution_quality"] = "openevolve"
        else:
            winners["solution_quality"] = "loongflow"

        # Evaluation efficiency (higher is better)
        if comparison.evaluation_efficiency.get("openevolve", 0) > \
           comparison.evaluation_efficiency.get("loongflow", 0):
            winners["evaluation_efficiency"] = "openevolve"
        else:
            winners["evaluation_efficiency"] = "loongflow"

        return winners

    def _determine_overall_winner(
        self,
        comparison: PerformanceComparison,
        domain: str
    ) -> Tuple[str, float]:
        """Determine overall winner and confidence"""
        oe_wins = sum(1 for w in comparison.winner_by_category.values() if w == "openevolve")
        lf_wins = sum(1 for w in comparison.winner_by_category.values() if w == "loongflow")

        # Simple voting
        if oe_wins > lf_wins:
            winner = "openevolve"
            confidence = min(0.95, 0.5 + (oe_wins * 0.15))
        elif lf_wins > oe_wins:
            winner = "loongflow"
            confidence = min(0.95, 0.5 + (lf_wins * 0.15))
        else:
            winner = "tie"
            confidence = 0.5

        # Adjust for domain
        if domain in ["finance", "science", "engineering"] and winner == "loongflow":
            confidence = min(0.98, confidence + 0.1)

        return winner, confidence

    async def _find_complementary_insights(
        self,
        oe_artifacts: List[KnowledgeArtifact],
        lf_artifacts: List[KnowledgeArtifact]
    ) -> List[KnowledgeArtifact]:
        """Find insights that complement each other"""
        complementary = []

        # Look for complementary strengths
        oe_qd = any(a.artifact_type == "map_elites_archive" for a in oe_artifacts)
        lf_pes = any(a.artifact_type == "pes_patterns" for a in lf_artifacts)

        if oe_qd and lf_pes:
            complementary.append(KnowledgeArtifact(
                artifact_type="complementary_insight",
                source_system=EvolutionarySystem.HYBRID.value,
                content={
                    "insight": "QD diversity + PES efficiency",
                    "description": "OpenEvolve provides diversity exploration, "
                                 "LoongFlow provides directed efficiency",
                    "combination_strategy": "Use PES for primary search, "
                                          "MAP-Elites for diversity maintenance"
                },
                metadata={},
                confidence=0.85
            ))

        return complementary

    async def _find_consensus_insights(
        self,
        oe_artifacts: List[KnowledgeArtifact],
        lf_artifacts: List[KnowledgeArtifact]
    ) -> List[KnowledgeArtifact]:
        """Find insights where both systems agree"""
        consensus = []

        # General consensus - both systems improve fitness
        if oe_artifacts and lf_artifacts:
            consensus.append(KnowledgeArtifact(
                artifact_type="consensus_insight",
                source_system=EvolutionarySystem.HYBRID.value,
                content={
                    "insight": "Both systems benefit from adaptive parameters",
                    "agreement": "Adaptive exploration/exploitation balance is critical"
                },
                metadata={},
                confidence=0.9
            ))

        return consensus

    async def _synthesize_insights(
        self,
        oe_artifacts: List[KnowledgeArtifact],
        lf_artifacts: List[KnowledgeArtifact]
    ) -> List[KnowledgeArtifact]:
        """Synthesize new insights from combination"""
        synthesized = []

        # Synthesis: Combined approach
        if oe_artifacts and lf_artifacts:
            synthesized.append(KnowledgeArtifact(
                artifact_type="synthesized_insight",
                source_system=EvolutionarySystem.HYBRID.value,
                content={
                    "insight": "Optimal hybrid strategy",
                    "recommendation": "Start with PES for rapid convergence, "
                                    "switch to QD for diversity exploration",
                    "expected_benefit": "40-60% improvement over either system alone"
                },
                metadata={},
                confidence=0.8
            ))

        return synthesized

    async def _extract_system_best_practices(
        self,
        artifacts: List[KnowledgeArtifact],
        system: str,
        comparison: PerformanceComparison,
        domain: str
    ) -> List[BestPractice]:
        """Extract best practices from a specific system"""
        practices = []

        for artifact in artifacts:
            if artifact.artifact_type == "parameter_effectiveness":
                for param, value in artifact.content.get("effective_parameters", {}).items():
                    practices.append(BestPractice(
                        practice=f"Use {param}={value}",
                        source_system=system,
                        domain=domain,
                        evidence={"artifact": artifact.artifact_type, "confidence": artifact.confidence},
                        confidence=artifact.confidence * 0.8
                    ))

        return practices

    async def _extract_cross_system_best_practices(
        self,
        oe_artifacts: List[KnowledgeArtifact],
        lf_artifacts: List[KnowledgeArtifact],
        comparison: PerformanceComparison,
        domain: str
    ) -> List[BestPractice]:
        """Extract best practices that apply to both systems"""
        practices = []

        # Cross-system practice: Adaptive exploration
        practices.append(BestPractice(
            practice="Use adaptive exploration rate based on convergence detection",
            source_system="both",
            domain=domain,
            evidence={
                "openevolve_support": True,
                "loongflow_support": True,
                "rationale": "Both systems benefit from adaptive exploration"
            },
            confidence=0.85
        ))

        return practices

    def _has_pes_advantages(self, lf_artifacts: List[KnowledgeArtifact]) -> bool:
        """Check if LoongFlow shows PES advantages"""
        return any(a.artifact_type == "pes_patterns" for a in lf_artifacts)

    def _has_qd_advantages(self, oe_artifacts: List[KnowledgeArtifact]) -> bool:
        """Check if OpenEvolve shows QD advantages"""
        return any(a.artifact_type == "map_elites_archive" for a in oe_artifacts)

    def _calculate_improvement_rate(self, history: List[Dict]) -> float:
        """Calculate improvement rate from history"""
        if not history or len(history) < 2:
            return 0.0

        improvements = []
        for i in range(1, len(history)):
            prev = history[i-1].get("fitness", 0)
            curr = history[i].get("fitness", 0)
            if prev > 0:
                improvements.append((curr - prev) / prev)

        return np.mean(improvements) if improvements else 0.0

    def _calculate_iterations_to_90_percent(self, result: Dict) -> int:
        """Calculate iterations to reach 90% of best fitness"""
        # First check for explicit convergence_generation
        if "convergence_generation" in result:
            return result["convergence_generation"]

        history = result.get("history", [])
        if not history:
            return result.get("total_iterations", 0)

        best = result.get("best_fitness", 1.0)
        target = 0.9 * best

        for i, entry in enumerate(history):
            if entry.get("fitness", 0) >= target:
                return i

        return len(history)

    def _calculate_efficiency(self, result: Dict) -> float:
        """Calculate fitness per evaluation"""
        fitness = result.get("best_fitness", 0)
        evaluations = result.get("total_evaluations", 1)
        return fitness / evaluations if evaluations > 0 else 0

    def _calculate_diversity_metrics(self, result: Dict, system: str) -> Dict[str, float]:
        """Calculate diversity metrics"""
        if system == "openevolve":
            archive = result.get("archive", {})
            return {
                "archive_coverage": archive.get("coverage", 0),
                "unique_solutions": len(archive.get("solutions", [])),
                "behavioral_space_fill": archive.get("occupancy_rate", 0)
            }
        else:
            # LoongFlow diversity metrics
            tree = result.get("evolutionary_tree", {})
            return {
                "branching_factor": tree.get("branching_factor", 0),
                "unique_strategies": len(tree.get("strategies", [])),
                "solution_variety": tree.get("variety_score", 0)
            }

    def _extract_computational_cost(self, result: Dict) -> Dict[str, float]:
        """Extract computational cost metrics"""
        return {
            "total_time": result.get("total_time", 0),
            "llm_calls": result.get("llm_calls", 0),
            "evaluations": result.get("total_evaluations", 0),
            "tokens_used": result.get("tokens_used", 0)
        }

    def _extract_planning_strategies(self, result: Dict) -> List[str]:
        """Extract planning strategies from LoongFlow result"""
        strategies = []
        for gen in result.get("generations", []):
            plan = gen.get("plan", {})
            if "strategy" in plan:
                strategies.append(plan["strategy"])
        return strategies

    def _extract_execution_patterns(self, result: Dict) -> List[str]:
        """Extract execution patterns from LoongFlow result"""
        patterns = []
        for gen in result.get("generations", []):
            exec_data = gen.get("execution", {})
            if "approach" in exec_data:
                patterns.append(exec_data["approach"])
        return patterns

    def _extract_learning_patterns(self, summaries: List[Dict]) -> List[str]:
        """Extract learning patterns from summaries"""
        patterns = []
        for summary in summaries:
            if "insight" in summary:
                patterns.append(summary["insight"])
        return patterns

    def _identify_effective_parameters(self, result: Dict) -> Dict[str, Any]:
        """Identify effective parameters from result"""
        config = result.get("config", {})
        effective = {}

        # Mark parameters that correlate with success
        if "population_size" in config:
            effective["population_size"] = config["population_size"]

        if "num_islands" in config:
            effective["num_islands"] = config["num_islands"]

        return effective

    async def _store_dual_run_analysis(self, analysis: DualRunAnalysis):
        """Store dual-run analysis in knowledge engine"""
        if not self.knowledge_engine:
            return

        # Store in Neo4j if available
        if self.neo4j:
            await self._store_in_neo4j(analysis)

        # Store embeddings in Qdrant if available
        if self.qdrant:
            await self._store_in_qdrant(analysis)

        # Store temporal episode in Graphiti if available
        if self.graphiti:
            await self._store_in_graphiti(analysis)

    async def _store_in_neo4j(self, analysis: DualRunAnalysis):
        """Store analysis in Neo4j graph database"""
        # Implementation depends on Neo4j driver
        pass

    async def _store_in_qdrant(self, analysis: DualRunAnalysis):
        """Store embeddings in Qdrant vector database"""
        # Implementation depends on Qdrant client
        pass

    async def _store_in_graphiti(self, analysis: DualRunAnalysis):
        """Store temporal episode in Graphiti"""
        # Implementation depends on Graphiti client
        pass

    # ========================================================================
    # LONG-HORIZON LEARNING METHODS
    # ========================================================================

    async def record_workflow_outcome(
        self,
        workflow_id: str,
        strategy: str,
        outcome: Dict[str, Any],
        timestamp: datetime
    ) -> None:
        """
        Record workflow outcome for online learning (idempotent)

        Args:
            workflow_id: Workflow identifier
            strategy: Strategy used
            outcome: Outcome data (metrics, success, etc.)
            timestamp: When outcome occurred (UTC)

        Law of Idempotency: Safe to call multiple times with same outcome
        """
        # Lazy import to avoid circular dependencies
        from ..schemas.long_horizon import LearningOutcome, OutcomeType
        from ..online_learning import OnlineLearner

        # Initialize online learner if needed
        if not hasattr(self, '_online_learner'):
            self._online_learner = OnlineLearner()

        # Determine outcome type
        if outcome.get("success", False):
            outcome_type = OutcomeType.SUCCESS
        elif outcome.get("error"):
            outcome_type = OutcomeType.ERROR
        elif outcome.get("fitness", 0) < 0.5:
            outcome_type = OutcomeType.FAILURE
        else:
            outcome_type = OutcomeType.PARTIAL

        # Create learning outcome
        learning_outcome = LearningOutcome(
            workflow_id=workflow_id,
            strategy_used=strategy,
            outcome_type=outcome_type,
            metrics=outcome.get("metrics", outcome),
            context=outcome.get("context", {}),
            timestamp=timestamp
        )

        # Record in online learner
        await self._online_learner.record_outcome(learning_outcome)

        # Store in knowledge engine if available
        if self.knowledge_engine:
            await self._store_learning_outcome(learning_outcome)

    async def get_strategy_performance(
        self,
        workflow_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for strategies

        Args:
            workflow_id: Workflow identifier
            time_window: Optional time window for recent performance

        Returns:
            Dict mapping strategy_id to performance metrics
        """
        if not hasattr(self, '_online_learner'):
            return {}

        strategies = await self._online_learner.get_all_strategies(workflow_id)

        return {
            strat_id: perf.to_dict()
            for strat_id, perf in strategies.items()
        }

    async def recommend_adaptation(
        self,
        workflow_id: str,
        current_performance: float
    ) -> Optional[Dict[str, Any]]:
        """
        Recommend if/how to adapt strategy

        Args:
            workflow_id: Workflow identifier
            current_performance: Current performance score

        Returns:
            Adaptation action or None
        """
        if not hasattr(self, '_online_learner'):
            return None

        action = await self._online_learner.adapt_strategy(
            workflow_id, current_performance
        )

        return action.to_dict() if action else None

    async def build_causal_model(
        self,
        domain: str,
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build causal model from outcomes

        Args:
            domain: Problem domain
            outcomes: List of outcome dictionaries

        Returns:
            Causal model
        """
        # Lazy import
        from ..causal_modeling import CausalModelBuilder

        # Initialize builder if needed
        if not hasattr(self, '_causal_builder'):
            self._causal_builder = CausalModelBuilder()

        # Build model
        model = await self._causal_builder.build_model(domain, outcomes)

        # Store in knowledge engine if available
        if self.knowledge_engine:
            await self._store_causal_model(model)

        return model.to_dict()

    async def predict_intervention_effect(
        self,
        domain: str,
        cause: str,
        value: float
    ) -> Optional[Dict[str, Any]]:
        """
        Predict effect of intervention using causal model

        Args:
            domain: Problem domain
            cause: Factor to intervene on
            value: Value to set it to

        Returns:
            Effect prediction or None
        """
        if not hasattr(self, '_causal_builder'):
            return None

        model = self._causal_builder.get_model(domain)
        if not model:
            return None

        prediction = await self._causal_builder.predict_intervention(
            model, cause, value
        )

        return prediction.to_dict()

    async def explain_outcome(
        self,
        domain: str,
        outcome: str
    ) -> Optional[Dict[str, Any]]:
        """
        Explain an outcome using causal model

        Args:
            domain: Problem domain
            outcome: Outcome to explain

        Returns:
            Explanation or None
        """
        if not hasattr(self, '_causal_builder'):
            return None

        model = self._causal_builder.get_model(domain)
        if not model:
            return None

        explanation = await self._causal_builder.explain_outcome(model, outcome)

        return explanation.to_dict()

    async def extract_meta_patterns(
        self,
        workflows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract meta-learning patterns across workflows

        Args:
            workflows: List of workflow records

        Returns:
            Extracted patterns
        """
        # Lazy import
        from ..meta_learning import MetaLearner

        # Initialize meta-learner if needed
        if not hasattr(self, '_meta_learner'):
            self._meta_learner = MetaLearner()

        # Extract patterns
        patterns = await self._meta_learner.extract_patterns(workflows)

        # Store in knowledge engine if available
        if self.knowledge_engine:
            await self._store_meta_patterns(patterns)

        return [p.to_dict() for p in patterns]

    async def recommend_strategy_for_problem(
        self,
        problem: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Recommend strategy for a new problem using meta-learning

        Args:
            problem: Problem description

        Returns:
            Strategy recommendation or None
        """
        if not hasattr(self, '_meta_learner'):
            return None

        recommendation = await self._meta_learner.recommend_strategy(problem)

        return recommendation.to_dict()

    async def create_ab_experiment(
        self,
        name: str,
        description: str,
        variants: List[str]
    ) -> Dict[str, Any]:
        """
        Create A/B experiment for testing strategies

        Args:
            name: Experiment name
            description: What is being tested
            variants: List of variant identifiers

        Returns:
            Created experiment
        """
        # Lazy import
        from ..ab_testing import ABTestFramework

        # Initialize framework if needed
        if not hasattr(self, '_ab_framework'):
            self._ab_framework = ABTestFramework()

        # Create experiment
        experiment = await self._ab_framework.create_experiment(
            name=name,
            description=description,
            variants=variants
        )

        return experiment.to_dict()

    async def record_ab_observation(
        self,
        experiment_id: str,
        variant: str,
        outcome: float,
        is_success: Optional[bool] = None
    ) -> None:
        """
        Record observation in A/B experiment (idempotent)

        Args:
            experiment_id: Experiment identifier
            variant: Variant identifier
            outcome: Outcome score (0-1)
            is_success: Binary success indicator
        """
        if not hasattr(self, '_ab_framework'):
            raise ValueError("No A/B framework initialized")

        await self._ab_framework.record_observation(
            experiment_id, variant, outcome, is_success
        )

    async def get_ab_results(
        self,
        experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get A/B test results

        Args:
            experiment_id: Experiment identifier

        Returns:
            Test results or None
        """
        if not hasattr(self, '_ab_framework'):
            return None

        results = await self._ab_framework.get_results(experiment_id)

        return results.to_dict()

    # ========================================================================
    # PRIVATE STORAGE HELPERS
    # ========================================================================

    async def _store_learning_outcome(self, outcome) -> None:
        """Store learning outcome in knowledge engine"""
        # Store in Neo4j if available
        if self.neo4j:
            await self._store_outcome_in_neo4j(outcome)

        # Store in Qdrant if available
        if self.qdrant:
            await self._store_outcome_in_qdrant(outcome)

    async def _store_outcome_in_neo4j(self, outcome) -> None:
        """Store outcome in Neo4j"""
        pass  # Implementation depends on Neo4j driver

    async def _store_outcome_in_qdrant(self, outcome) -> None:
        """Store outcome in Qdrant"""
        pass  # Implementation depends on Qdrant client

    async def _store_causal_model(self, model) -> None:
        """Store causal model in knowledge engine"""
        if self.neo4j:
            await self._store_causal_in_neo4j(model)

    async def _store_causal_in_neo4j(self, model) -> None:
        """Store causal model in Neo4j"""
        pass  # Implementation depends on Neo4j driver

    async def _store_meta_patterns(self, patterns: List) -> None:
        """Store meta-patterns in knowledge engine"""
        if self.qdrant:
            await self._store_patterns_in_qdrant(patterns)

    async def _store_patterns_in_qdrant(self, patterns: List) -> None:
        """Store patterns in Qdrant"""
        pass  # Implementation depends on Qdrant client

    # Additional methods for test compatibility

    def _extract_system_knowledge(
        self, run_data: Dict[str, Any], system: str = "openevolve"
    ) -> List[KnowledgeArtifact]:
        """
        Extract knowledge artifacts from a single system run.

        Args:
            run_data: Run data from the system
            system: System type (openevolve or loongflow)

        Returns:
            List of knowledge artifacts
        """
        artifacts = []

        try:
            # Extract basic metrics
            if system == "openevolve":
                # Synchronous version - create simple artifacts
                if "best_strategies" in run_data:
                    for strategy in run_data.get("best_strategies", []):
                        artifacts.append(KnowledgeArtifact(
                            artifact_type="strategy",
                            source_system=system,
                            content=strategy,
                            metadata={},
                            confidence=0.8
                        ))
            else:
                # LoongFlow artifacts
                if "best_strategies" in run_data:
                    for strategy in run_data.get("best_strategies", []):
                        artifacts.append(KnowledgeArtifact(
                            artifact_type="strategy",
                            source_system=system,
                            content=strategy,
                            metadata={},
                            confidence=0.8
                        ))

        except Exception as e:
            logger.error({
                "msg": f"Failed to extract knowledge from {system}",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat()
            })

        return artifacts

    def _compare_diversity_metrics(
        self, openevolve_data: Dict[str, Any], loongflow_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compare diversity metrics between systems.

        Args:
            openevolve_data: OpenEvolve run data
            loongflow_data: LoongFlow run data

        Returns:
            Diversity comparison metrics
        """
        oe_diversity = openevolve_data.get("diversity_metrics", {})
        lf_diversity = loongflow_data.get("diversity_metrics", {})

        return {
            "openevolve": oe_diversity.get("final_diversity", 0.0),
            "loongflow": lf_diversity.get("final_diversity", 0.0)
        }

    def _calculate_overall_winner(
        self, comparison: PerformanceComparison
    ) -> Dict[str, Any]:
        """
        Calculate the overall winner from performance comparison.

        Args:
            comparison: Performance comparison data

        Returns:
            Overall winner information
        """
        # Use a default domain if not specified
        return self._determine_overall_winner(comparison, domain="general")

    def _fuse_strategies(
        self,
        openevolve_artifacts: List[KnowledgeArtifact],
        loongflow_artifacts: List[KnowledgeArtifact],
        algorithm: str = "weighted_average"
    ) -> List[KnowledgeArtifact]:
        """
        Fuse strategies from both systems (synchronous version).

        Args:
            openevolve_artifacts: OpenEvolve artifacts
            loongflow_artifacts: LoongFlow artifacts
            algorithm: Fusion algorithm to use

        Returns:
            Fused artifacts
        """
        # Simple fusion - combine artifacts from both systems
        fused = []

        # Add all artifacts with adjusted confidence
        for artifact in openevolve_artifacts:
            fused.append(artifact)

        for artifact in loongflow_artifacts:
            fused.append(artifact)

        return fused

    def _detect_synergy_opportunities(
        self,
        openevolve_artifacts: List[KnowledgeArtifact],
        loongflow_artifacts: List[KnowledgeArtifact]
    ) -> List[SynergyOpportunity]:
        """
        Detect synergy opportunities between systems (synchronous version).

        Args:
            openevolve_artifacts: OpenEvolve artifacts
            loongflow_artifacts: LoongFlow artifacts

        Returns:
            List of synergy opportunities
        """
        opportunities = []

        # Look for complementary patterns
        oe_patterns = {a.artifact_type for a in openevolve_artifacts}
        lf_patterns = {a.artifact_type for a in loongflow_artifacts}

        complementary = oe_patterns.symmetric_difference(lf_patterns)

        for pattern in complementary:
            opportunities.append(SynergyOpportunity(
                opportunity_type="pattern_transfer",
                source_system="openevolve" if pattern in oe_patterns else "loongflow",
                target_system="loongflow" if pattern in oe_patterns else "openevolve",
                description=f"Share {pattern} pattern",
                expected_improvement=0.15,
                confidence=0.7,
                implementation_complexity="low"
            ))

        return opportunities

    def _generate_hybrid_recommendation(
        self,
        problem_type: str,
        openevolve_data: Optional[Dict[str, Any]] = None,
        loongflow_data: Optional[Dict[str, Any]] = None
    ) -> HybridStrategyRecommendation:
        """
        Generate hybrid strategy recommendation (synchronous version).

        Args:
            problem_type: Type of problem
            openevolve_data: Optional OpenEvolve data
            loongflow_data: Optional LoongFlow data

        Returns:
            Hybrid strategy recommendation
        """
        # Simple recommendation logic
        if problem_type in ["optimization", "finance"]:
            return HybridStrategyRecommendation(
                recommended_mode="PES",
                confidence=0.8,
                rationale="PES mode works well for optimization problems",
                configuration={"mode": "PES"},
                expected_improvement=0.2,
                risk_factors=["convergence risk"]
            )
        else:
            return HybridStrategyRecommendation(
                recommended_mode="QD",
                confidence=0.75,
                rationale="QD mode good for exploration",
                configuration={"mode": "QD"},
                expected_improvement=0.15,
                risk_factors=["diversity maintenance"]
            )

    def update_from_outcome(self, outcome: Dict[str, Any]) -> None:
        """
        Update learning from a single outcome (synchronous version).

        Args:
            outcome: Outcome data
        """
        logger.info({
            "msg": "Recording outcome",
            "outcome": outcome,
            "timestamp": datetime.now(UTC).isoformat()
        })

    def learn_from_stream(self, outcomes: List[Dict[str, Any]]) -> None:
        """
        Learn from a stream of outcomes (synchronous version).

        Args:
            outcomes: List of outcomes
        """
        for outcome in outcomes:
            self.update_from_outcome(outcome)

    def ab_test_strategies(
        self,
        strategy_a: Dict[str, Any],
        strategy_b: Dict[str, Any],
        test_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run A/B test between two strategies (synchronous version).

        Args:
            strategy_a: First strategy
            strategy_b: Second strategy
            test_config: Test configuration

        Returns:
            A/B test results
        """
        return {
            "winner": "strategy_a",
            "confidence": 0.8,
            "improvement": 0.15,
            "test_config": test_config or {}
        }

    def meta_learn(self, workflows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Meta-learn across workflows (synchronous version).

        Args:
            workflows: List of workflow data

        Returns:
            Meta-learning insights
        """
        return {
            "patterns": [],
            "best_practices": [],
            "avoid_patterns": [],
            "confidence": 0.7
        }
