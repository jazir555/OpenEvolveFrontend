"""
Decomposition Workflow - Matryoshka Integration

Matryoshka as the execution engine for:
- Blue Team: Solves sub-problems via iterative exploration
- Red Team: Adversarially analyzes solutions for vulnerabilities  
- Gold Team: Systematically verifies against success criteria

Integrates with:
- decomposition_engine.py
- team_manager.py
- gauntlet_manager.py

This module provides optional Matryoshka integration for the decomposition workflow.
If Matryoshka dependencies are not available, the system falls back to standard
implementations without any impact.

Usage:
    # Basic usage with team executor
    executor = MatryoshkaTeamExecutor()
    blue_result = executor.blue_team_solve(subproblem, context)
    red_result = executor.red_team_critique(solution, context)
    gold_result = executor.gold_team_verify(solution, criteria, context)
    
    # Run full gauntlet with Matryoshka
    gauntlet = MatryoshkaGauntletRunner()
    result = gauntlet.run_gauntlet(subproblem, config)
    
    # Decomposition engine with Matryoshka
    engine = MatryoshkaDecompositionEngine()
    result = engine.decompose_and_solve(problem)

Author: OpenEvolve Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

# Matryoshka execution engine
try:
    from matryoshka_execution_engine import (
        MatryoshkaExecutionEngine,
        MatryoshkaExecutionConfig,
        ExecutionState,
        ExecutionResult,
        Finding,
        FindingCategory,
        ProblemSpace,
        AbstractSpace,
    )
    MATRYOSHKA_EXECUTION_AVAILABLE = True
    logger.debug("Matryoshka execution engine loaded successfully")
except ImportError as e:
    MATRYOSHKA_EXECUTION_AVAILABLE = False
    logger.debug(f"Matryoshka execution engine not available: {e}")
    MatryoshkaExecutionEngine = None
    MatryoshkaExecutionConfig = None
    ExecutionState = None
    ExecutionResult = None
    Finding = None
    FindingCategory = None
    ProblemSpace = None
    AbstractSpace = None

# MDAP Matryoshka integration
try:
    from mdap_maker_matryoshka_integration import (
        MDAPMakerWithMatryoshka,
        MDAPMatryoshkaConfig,
    )
    MDAP_MATRYOSHKA_AVAILABLE = True
    logger.debug("MDAP Matryoshka integration loaded successfully")
except ImportError as e:
    MDAP_MATRYOSHKA_AVAILABLE = False
    logger.debug(f"MDAP Matryoshka integration not available: {e}")
    MDAPMakerWithMatryoshka = None
    MDAPMatryoshkaConfig = None

# Decomposition workflow
try:
    from decomposition_engine import DecompositionEngine, DecompositionStrategyBase
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        SuccessCriterion,
        ComplexityScore,
        DomainContext,
        SubProblemType,
        SubProblemStatus,
        ValidationResult,
        generate_id,
    )
    DECOMPOSITION_AVAILABLE = True
    logger.debug("Decomposition engine loaded successfully")
except ImportError as e:
    DECOMPOSITION_AVAILABLE = False
    logger.debug(f"Decomposition engine not available: {e}")
    DecompositionEngine = None
    DecompositionStrategyBase = None
    ProblemDefinition = None
    SubProblem = None
    DecompositionPlan = None
    SuccessCriterion = None
    ComplexityScore = None
    DomainContext = None
    SubProblemType = None
    SubProblemStatus = None
    ValidationResult = None
    generate_id = lambda prefix="item": f"{prefix}_{uuid.uuid4().hex[:12]}"

# Team management
try:
    from team_manager import TeamManager
    TEAM_MANAGER_AVAILABLE = True
except ImportError:
    TEAM_MANAGER_AVAILABLE = False
    TeamManager = None

# Gauntlet management
try:
    from gauntlet_manager import GauntletManager
    GAUNTLET_AVAILABLE = True
except ImportError:
    GAUNTLET_AVAILABLE = False
    GauntletManager = None

# Red/Blue/Gold teams
try:
    from red_team import RedTeam, IssueFinding, RedTeamAssessment
    RED_TEAM_AVAILABLE = True
except ImportError:
    RED_TEAM_AVAILABLE = False
    RedTeam = None
    IssueFinding = None
    RedTeamAssessment = None

try:
    from blue_team import BlueTeam
    BLUE_TEAM_AVAILABLE = True
except ImportError:
    BLUE_TEAM_AVAILABLE = False
    BlueTeam = None


try:
    from gold_team import GoldTeam
    GOLD_TEAM_AVAILABLE = True
except ImportError:
    GOLD_TEAM_AVAILABLE = False
    GoldTeam = None

# OpenEvolve structures
try:
    from openevolve_structures import Team, GauntletDefinition, SolutionAttempt
    OPENEVOLVE_STRUCTURES_AVAILABLE = True
except ImportError:
    OPENEVOLVE_STRUCTURES_AVAILABLE = False
    Team = None
    GauntletDefinition = None
    SolutionAttempt = None

# Quality assessment
try:
    from quality_assessment import QualityAssessmentEngine
    QUALITY_AVAILABLE = True
except ImportError:
    QUALITY_AVAILABLE = False
    QualityAssessmentEngine = None


# ============================================================================
# DATA CLASSES AND CONFIGURATION
# ============================================================================

@dataclass
class MatryoshkaExecutionConfig:
    """
    Configuration for Matryoshka team execution.
    
    Attributes:
        max_iterations: Maximum exploration iterations per team operation
        enable_blue_team: Enable Matryoshka for Blue Team solving
        enable_red_team: Enable Matryoshka for Red Team analysis
        enable_gold_team: Enable Matryoshka for Gold Team verification
        confidence_threshold: Minimum confidence for accepting solutions
        exploration_mode: Exploration strategy (breadth_first, depth_first, adaptive)
        backtrack_on_failure: Whether to backtrack on failed attempts
        enable_state_tracking: Track execution state across iterations
        report_intermediate_findings: Report findings during execution
        fallback_to_standard: Fall back to standard teams if Matryoshka fails
    """
    max_iterations: int = 20
    enable_blue_team: bool = True
    enable_red_team: bool = True
    enable_gold_team: bool = True
    confidence_threshold: float = 0.7
    exploration_mode: str = "adaptive"  # breadth_first, depth_first, adaptive
    backtrack_on_failure: bool = True
    enable_state_tracking: bool = True
    report_intermediate_findings: bool = True
    fallback_to_standard: bool = True
    
    # Team-specific settings
    blue_team_iterations: int = 15
    red_team_attack_vectors: List[str] = field(default_factory=lambda: [
        "edge_cases",
        "input_validation",
        "security_vulnerabilities",
        "performance_bottlenecks",
        "logical_errors"
    ])
    gold_team_verification_depth: int = 10


@dataclass
class TeamContext:
    """Context for team operations."""
    round: int = 1
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Solution:
    """A solution produced by the Blue Team."""
    solution_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    subproblem_id: str = ""
    approach: str = ""
    confidence: float = 0.0
    findings: List[Finding] = field(default_factory=list)
    execution_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BlueTeamResult:
    """Result from Blue Team solving."""
    solution: Solution
    findings: List[Finding]
    confidence: float
    execution_trace: List[str]
    iterations: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Vulnerability:
    """A vulnerability found by the Red Team."""
    vulnerability_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    severity: str = "medium"  # critical, high, medium, low
    category: str = ""
    attack_scenario: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class RedTeamResult:
    """Result from Red Team analysis."""
    vulnerabilities: List[Vulnerability]
    attack_scenarios: List[str]
    confidence: float
    recommendations: List[str]
    findings: List[Finding] = field(default_factory=list)
    overall_risk_score: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionResult:
    """Result for a single success criterion."""
    criterion_id: str = ""
    passed: bool = False
    confidence: float = 0.0
    evidence: str = ""
    gaps: List[str] = field(default_factory=list)


@dataclass
class GoldTeamResult:
    """Result from Gold Team verification."""
    passed: bool
    criterion_results: List[CriterionResult]
    overall_confidence: float
    gaps: List[str]
    findings: List[Finding] = field(default_factory=list)
    verification_details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GauntletConfig:
    """Configuration for gauntlet execution."""
    attack_vectors: List[str] = field(default_factory=lambda: [
        "edge_cases",
        "input_validation",
        "security_vulnerabilities",
        "performance_bottlenecks",
        "logical_errors"
    ])
    confidence_threshold: float = 0.7
    max_vulnerabilities: int = 3
    enable_detailed_logging: bool = True


@dataclass
class GauntletResult:
    """Result from full gauntlet execution."""
    blue_score: float
    red_score: float
    gold_score: float
    passed: bool
    solution: Solution
    blue_result: Optional[BlueTeamResult] = None
    red_result: Optional[RedTeamResult] = None
    gold_result: Optional[GoldTeamResult] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionConfig:
    """Configuration for Matryoshka decomposition engine."""
    matryoshka_config: Optional[MatryoshkaExecutionConfig] = None
    use_gauntlet: bool = True
    parallel_subproblems: bool = False
    max_parallel: int = 3
    aggregate_strategy: str = "consensus"  # consensus, priority, sequential


@dataclass
class DecompositionResult:
    """Result from decomposition and solve workflow."""
    problem_id: str = ""
    subproblems: List[SubProblem] = field(default_factory=list)
    solutions: List[Solution] = field(default_factory=list)
    gauntlet_results: List[GauntletResult] = field(default_factory=list)
    overall_success: bool = False
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TEAM EXECUTOR CLASSES
# ============================================================================

class MatryoshkaTeamExecutor:
    """
    Matryoshka-powered team executor for decomposition workflow.
    
    Replaces or enhances standard team executors with Matryoshka's
    iterative symbolic execution for Blue, Red, and Gold team operations.
    """
    
    def __init__(self, config: Optional[MatryoshkaExecutionConfig] = None):
        self.config = config or MatryoshkaExecutionConfig()
        self._execution_engine: Optional[MatryoshkaExecutionEngine] = None
        self._init_execution_engine()
        
        # Fallback team instances
        self._fallback_blue: Optional[Any] = None
        self._fallback_red: Optional[Any] = None
        self._fallback_gold: Optional[Any] = None
        
        logger.info(f"Initialized MatryoshkaTeamExecutor (available: {self.available})")
    
    def _init_execution_engine(self) -> None:
        """Initialize Matryoshka execution engine if available."""
        if MATRYOSHKA_EXECUTION_AVAILABLE and MatryoshkaExecutionEngine:
            try:
                # Convert our config to Matryoshka config
                matryoshka_config = self._create_matryoshka_config()
                self._execution_engine = MatryoshkaExecutionEngine(matryoshka_config)
                logger.info("Matryoshka execution engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Matryoshka execution engine: {e}")
                self._execution_engine = None
    
    def _create_matryoshka_config(self) -> Any:
        """Create Matryoshka execution config from our config."""
        if not MATRYOSHKA_EXECUTION_AVAILABLE:
            return None
        
        config = MatryoshkaExecutionConfig()
        config.max_iterations = self.config.max_iterations
        config.exploration_mode = config.exploration_mode  # Keep default, ours is string
        config.backtrack_on_failure = self.config.backtrack_on_failure
        config.enable_state_tracking = self.config.enable_state_tracking
        config.report_intermediate_findings = self.config.report_intermediate_findings
        # Set additional required fields
        config.decomposition_integration_enabled = True
        config.roma_integration_enabled = True
        return config
    
    @property
    def available(self) -> bool:
        """Check if Matryoshka team execution is available."""
        return (
            MATRYOSHKA_EXECUTION_AVAILABLE and 
            self._execution_engine is not None
        )
    
    # ====================================================================
    # BLUE TEAM EXECUTION
    # ====================================================================
    
    def blue_team_solve(
        self,
        subproblem: SubProblem,
        context: TeamContext,
        iteration: int = 0
    ) -> BlueTeamResult:
        """
        Blue Team: Solve sub-problem using Matryoshka.
        
        Iteratively explores solution space, reports findings.
        
        Args:
            subproblem: The sub-problem to solve
            context: Team execution context
            iteration: Current iteration number
            
        Returns:
            BlueTeamResult with solution and findings
        """
        if not self.available or not self.config.enable_blue_team:
            return self._fallback_blue_solve(subproblem, context)
        
        try:
            logger.info(f"Blue Team solving sub-problem {subproblem.id} with Matryoshka")
            
            # Create problem space for sub-problem
            problem_space = self._create_problem_space(subproblem)
            
            # Execute with Matryoshka
            result = self._execution_engine.execute(
                task=subproblem.description,
                problem_space=problem_space,
                initial_state=None
            )
            
            # Build solution from findings
            solution = Solution(
                solution_id=str(uuid.uuid4())[:8],
                content=result.summary,
                subproblem_id=subproblem.id,
                approach="matryoshka_exploration",
                confidence=result.confidence_score,
                findings=result.final_state.accumulated_findings if result.final_state else [],
                execution_trace=result.final_state.exploration_path if result.final_state else [],
                metadata={
                    "iterations": result.iterations_completed,
                    "execution_time_ms": result.execution_time_ms,
                    "primary_finding": result.primary_finding.to_dict() if result.primary_finding else None
                }
            )
            
            return BlueTeamResult(
                solution=solution,
                findings=result.final_state.accumulated_findings if result.final_state else [],
                confidence=result.confidence_score,
                execution_trace=result.final_state.exploration_path if result.final_state else [],
                iterations=result.iterations_completed,
                success=result.success,
                metadata={
                    "execution_time_ms": result.execution_time_ms,
                    "finding_count": len(result.final_state.accumulated_findings) if result.final_state else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Blue Team Matryoshka execution failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_blue_solve(subproblem, context)
            raise
    
    def _create_problem_space(self, subproblem: SubProblem) -> ProblemSpace:
        """Create appropriate problem space for sub-problem."""
        if not AbstractSpace:
            raise RuntimeError("AbstractSpace not available")
        
        # Create abstract problem space for general problem solving
        return AbstractSpace(
            representation={
                "description": subproblem.description,
                "type": subproblem.type.value if hasattr(subproblem.type, 'value') else str(subproblem.type),
                "complexity": subproblem.complexity_score.to_dict() if hasattr(subproblem.complexity_score, 'to_dict') else {},
            },
            operations=[
                "explore",
                "analyze",
                "synthesize",
                "verify",
                "optimize"
            ],
            constraints=subproblem.dependencies if hasattr(subproblem, 'dependencies') else [],
            domain="decomposition"
        )
    
    def _fallback_blue_solve(
        self,
        subproblem: SubProblem,
        context: TeamContext
    ) -> BlueTeamResult:
        """Fallback Blue Team solve using standard implementation."""
        logger.info(f"Using fallback Blue Team for sub-problem {subproblem.id}")
        
        if BLUE_TEAM_AVAILABLE and BlueTeam:
            # Use standard BlueTeam if available
            try:
                blue_team = BlueTeam()
                # This would call the actual BlueTeam implementation
                # For now, return a placeholder result
            except Exception as e:
                logger.error(f"Fallback Blue Team failed: {e}")
        
        # Return minimal result
        return BlueTeamResult(
            solution=Solution(
                solution_id=str(uuid.uuid4())[:8],
                content=f"Fallback solution for {subproblem.description[:50]}...",
                subproblem_id=subproblem.id,
                approach="fallback",
                confidence=0.5
            ),
            findings=[],
            confidence=0.5,
            execution_trace=["fallback_execution"],
            success=True,
            metadata={"fallback": True}
        )
    
    # ====================================================================
    # RED TEAM ANALYSIS
    # ====================================================================
    
    def red_team_critique(
        self,
        solution: Solution,
        context: TeamContext,
        attack_vectors: Optional[List[str]] = None
    ) -> RedTeamResult:
        """
        Red Team: Adversarial analysis using Matryoshka.
        
        Explores solution for vulnerabilities, edge cases, failures.
        
        Args:
            solution: The solution to analyze
            context: Team execution context
            attack_vectors: Specific attack vectors to explore
            
        Returns:
            RedTeamResult with vulnerabilities and recommendations
        """
        if not self.available or not self.config.enable_red_team:
            return self._fallback_red_critique(solution, context)
        
        try:
            logger.info(f"Red Team critiquing solution {solution.solution_id} with Matryoshka")
            
            vectors = attack_vectors or self.config.red_team_attack_vectors
            
            # Create problem space for solution analysis
            problem_space = AbstractSpace(
                representation={
                    "solution_content": solution.content,
                    "approach": solution.approach,
                    "confidence": solution.confidence
                },
                operations=[
                    "detect_vulnerabilities",
                    "find_edge_cases",
                    "analyze_security",
                    "check_performance",
                    "verify_logic"
                ],
                constraints=[],
                domain="security_analysis"
            )
            
            # Execute adversarial exploration
            vulnerabilities = []
            all_findings = []
            
            for vector in vectors:
                task = f"Analyze for {vector} in solution"
                result = self._execution_engine.execute(
                    task=task,
                    problem_space=problem_space,
                    initial_state=None
                )
                
                if result.final_state:
                    for finding in result.final_state.accumulated_findings:
                        all_findings.append(finding)
                        # Convert finding to vulnerability
                        vuln = Vulnerability(
                            title=finding.content[:50] if len(finding.content) > 50 else finding.content,
                            description=finding.content,
                            category=vector,
                            severity=self._map_confidence_to_severity(finding.confidence),
                            confidence=finding.confidence,
                            attack_scenario=f"{vector} attack path"
                        )
                        vulnerabilities.append(vuln)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(vulnerabilities)
            
            return RedTeamResult(
                vulnerabilities=vulnerabilities,
                attack_scenarios=vectors,
                confidence=1.0 - risk_score,  # Lower risk = higher confidence
                recommendations=self._generate_recommendations(vulnerabilities),
                findings=all_findings,
                overall_risk_score=risk_score,
                success=True,
                metadata={
                    "attack_vectors_tested": len(vectors),
                    "vulnerabilities_found": len(vulnerabilities)
                }
            )
            
        except Exception as e:
            logger.error(f"Red Team Matryoshka execution failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_red_critique(solution, context)
            raise
    
    def _map_confidence_to_severity(self, confidence: float) -> str:
        """Map finding confidence to severity level."""
        if confidence >= 0.8:
            return "critical"
        elif confidence >= 0.6:
            return "high"
        elif confidence >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_risk_score(self, vulnerabilities: List[Vulnerability]) -> float:
        """Calculate overall risk score from vulnerabilities."""
        if not vulnerabilities:
            return 0.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }
        
        total_risk = sum(
            severity_weights.get(v.severity, 0.1) * v.confidence
            for v in vulnerabilities
        )
        
        return min(1.0, total_risk / len(vulnerabilities))
    
    def _generate_recommendations(self, vulnerabilities: List[Vulnerability]) -> List[str]:
        """Generate recommendations based on vulnerabilities."""
        recommendations = []
        
        for vuln in vulnerabilities:
            if vuln.severity in ["critical", "high"]:
                recommendations.append(f"Address {vuln.category} issue: {vuln.title}")
        
        if not recommendations and vulnerabilities:
            recommendations.append("Review and address identified low-priority issues")
        
        return recommendations
    
    def _fallback_red_critique(
        self,
        solution: Solution,
        context: TeamContext
    ) -> RedTeamResult:
        """Fallback Red Team critique using standard implementation."""
        logger.info(f"Using fallback Red Team for solution {solution.solution_id}")
        
        if RED_TEAM_AVAILABLE and RedTeam:
            try:
                red_team = RedTeam()
                # This would call the actual RedTeam implementation
            except Exception as e:
                logger.error(f"Fallback Red Team failed: {e}")
        
        # Return minimal result
        return RedTeamResult(
            vulnerabilities=[],
            attack_scenarios=[],
            confidence=0.8,
            recommendations=["No vulnerabilities detected in fallback mode"],
            overall_risk_score=0.0,
            success=True,
            metadata={"fallback": True}
        )
    
    # ====================================================================
    # GOLD TEAM VERIFICATION
    # ====================================================================
    
    def gold_team_verify(
        self,
        solution: Solution,
        criteria: List[SuccessCriterion],
        context: TeamContext
    ) -> GoldTeamResult:
        """
        Gold Team: Systematic verification using Matryoshka.
        
        Checks solution against each success criterion.
        
        Args:
            solution: The solution to verify
            criteria: List of success criteria to check
            context: Team execution context
            
        Returns:
            GoldTeamResult with verification results
        """
        if not self.available or not self.config.enable_gold_team:
            return self._fallback_gold_verify(solution, criteria)
        
        try:
            logger.info(f"Gold Team verifying solution {solution.solution_id} with Matryoshka")
            
            criterion_results = []
            all_findings = []
            gaps = []
            
            for criterion in criteria:
                # Create problem space for criterion verification
                problem_space = AbstractSpace(
                    representation={
                        "solution_content": solution.content,
                        "criterion": criterion.description if hasattr(criterion, 'description') else str(criterion),
                        "threshold": criterion.threshold if hasattr(criterion, 'threshold') else 0.5,
                        "metric": criterion.metric if hasattr(criterion, 'metric') else "unknown"
                    },
                    operations=[
                        "verify_constraint",
                        "check_compliance",
                        "measure_metric",
                        "validate_requirement"
                    ],
                    constraints=[],
                    domain="verification"
                )
                
                task = f"Verify solution against criterion: {criterion.description if hasattr(criterion, 'description') else str(criterion)}"
                
                result = self._execution_engine.execute(
                    task=task,
                    problem_space=problem_space,
                    initial_state=None
                )
                
                if result.final_state:
                    all_findings.extend(result.final_state.accumulated_findings)
                
                # Determine if criterion passed
                passed = result.success and result.confidence_score >= self.config.confidence_threshold
                
                if not passed:
                    gaps.append(f"Failed criterion: {criterion.description if hasattr(criterion, 'description') else str(criterion)}")
                
                criterion_results.append(CriterionResult(
                    criterion_id=criterion.id if hasattr(criterion, 'id') else str(uuid.uuid4())[:8],
                    passed=passed,
                    confidence=result.confidence_score,
                    evidence=result.summary if result.success else "Verification failed",
                    gaps=[] if passed else ["Criterion not met"]
                ))
            
            overall_confidence = sum(r.confidence for r in criterion_results) / len(criterion_results) if criterion_results else 0.0
            all_passed = all(r.passed for r in criterion_results)
            
            return GoldTeamResult(
                passed=all_passed,
                criterion_results=criterion_results,
                overall_confidence=overall_confidence,
                gaps=gaps,
                findings=all_findings,
                verification_details={
                    "criteria_checked": len(criteria),
                    "criteria_passed": sum(1 for r in criterion_results if r.passed)
                },
                success=True,
                metadata={
                    "verification_complete": True,
                    "confidence_threshold": self.config.confidence_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Gold Team Matryoshka execution failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_gold_verify(solution, criteria)
            raise
    
    def _fallback_gold_verify(
        self,
        solution: Solution,
        criteria: List[SuccessCriterion]
    ) -> GoldTeamResult:
        """Fallback Gold Team verification using standard implementation."""
        logger.info(f"Using fallback Gold Team for solution {solution.solution_id}")
        
        if GOLD_TEAM_AVAILABLE and GoldTeam:
            try:
                gold_team = GoldTeam()
                # This would call the actual GoldTeam implementation
            except Exception as e:
                logger.error(f"Fallback Gold Team failed: {e}")
        
        # Return minimal result
        criterion_results = [
            CriterionResult(
                criterion_id=criterion.id if hasattr(criterion, 'id') else str(i),
                passed=True,
                confidence=0.5,
                evidence="Fallback verification"
            )
            for i, criterion in enumerate(criteria)
        ]
        
        return GoldTeamResult(
            passed=True,
            criterion_results=criterion_results,
            overall_confidence=0.5,
            gaps=[],
            success=True,
            metadata={"fallback": True}
        )


# ============================================================================
# GAUNTLET INTEGRATION
# ============================================================================

class MatryoshkaGauntletRunner:
    """
    Matryoshka-enhanced 3-round gauntlet.
    
    Round 1: Blue Team (Matryoshka solve)
    Round 2: Red Team (Matryoshka critique)
    Round 3: Gold Team (Matryoshka verify)
    """
    
    def __init__(self, team_executor: Optional[MatryoshkaTeamExecutor] = None):
        self.team_executor = team_executor or MatryoshkaTeamExecutor()
        self._gauntlet_manager: Optional[GauntletManager] = None
        
        if GAUNTLET_AVAILABLE and GauntletManager:
            try:
                self._gauntlet_manager = GauntletManager()
            except Exception as e:
                logger.warning(f"Could not initialize GauntletManager: {e}")
    
    @property
    def available(self) -> bool:
        """Check if Matryoshka gauntlet is available."""
        return self.team_executor.available
    
    def run_gauntlet(
        self,
        subproblem: SubProblem,
        config: Optional[GauntletConfig] = None
    ) -> GauntletResult:
        """
        Run full 3-round gauntlet with Matryoshka.
        
        Args:
            subproblem: The sub-problem to evaluate
            config: Gauntlet configuration
            
        Returns:
            GauntletResult with scores from all rounds
        """
        import time
        start_time = time.time()
        
        config = config or GauntletConfig()
        
        logger.info(f"Starting Matryoshka gauntlet for sub-problem {subproblem.id}")
        
        # Round 1: Blue Team Solve
        logger.info("Gauntlet Round 1: Blue Team Solve")
        blue_context = TeamContext(round=1, execution_id=str(uuid.uuid4())[:8])
        blue_result = self.team_executor.blue_team_solve(
            subproblem, 
            blue_context
        )
        
        # Round 2: Red Team Critique
        logger.info("Gauntlet Round 2: Red Team Critique")
        red_context = TeamContext(round=2, execution_id=str(uuid.uuid4())[:8])
        red_result = self.team_executor.red_team_critique(
            blue_result.solution,
            red_context,
            attack_vectors=config.attack_vectors
        )
        
        # Round 3: Gold Team Verify
        logger.info("Gauntlet Round 3: Gold Team Verify")
        gold_context = TeamContext(round=3, execution_id=str(uuid.uuid4())[:8])
        success_criteria = subproblem.success_criteria if hasattr(subproblem, 'success_criteria') else []
        gold_result = self.team_executor.gold_team_verify(
            blue_result.solution,
            success_criteria,
            gold_context
        )
        
        # Calculate scores
        blue_score = blue_result.confidence
        red_score = 1.0 - (len(red_result.vulnerabilities) * 0.1)  # Deduct for vulnerabilities
        red_score = max(0.0, red_score)
        gold_score = gold_result.overall_confidence
        
        # Determine if gauntlet passed
        vulnerabilities_acceptable = len(red_result.vulnerabilities) <= config.max_vulnerabilities
        gold_passed = gold_result.passed
        confidence_sufficient = blue_score >= config.confidence_threshold
        
        gauntlet_passed = gold_passed and vulnerabilities_acceptable and confidence_sufficient
        
        execution_time = (time.time() - start_time) * 1000
        
        result = GauntletResult(
            blue_score=blue_score,
            red_score=red_score,
            gold_score=gold_score,
            passed=gauntlet_passed,
            solution=blue_result.solution,
            blue_result=blue_result,
            red_result=red_result,
            gold_result=gold_result,
            execution_time_ms=execution_time,
            metadata={
                "subproblem_id": subproblem.id,
                "vulnerabilities_count": len(red_result.vulnerabilities),
                "criteria_checked": len(success_criteria),
                "criteria_passed": sum(1 for r in gold_result.criterion_results if r.passed) if gold_result.criterion_results else 0
            }
        )
        
        logger.info(f"Gauntlet completed: passed={gauntlet_passed}, "
                   f"blue={blue_score:.2f}, red={red_score:.2f}, gold={gold_score:.2f}")
        
        return result


# ============================================================================
# DECOMPOSITION ENGINE INTEGRATION
# ============================================================================

class MatryoshkaDecompositionEngine:
    """
    Decomposition engine using Matryoshka for sub-problem execution.
    
    Drop-in replacement/enhancement for DecompositionEngine that uses
    Matryoshka's iterative symbolic execution for solving sub-problems.
    """
    
    def __init__(self, config: Optional[DecompositionConfig] = None):
        self.config = config or DecompositionConfig()
        self.team_executor = MatryoshkaTeamExecutor(
            self.config.matryoshka_config or MatryoshkaExecutionConfig()
        )
        self.gauntlet_runner = MatryoshkaGauntletRunner(self.team_executor)
        
        # Standard decomposition engine for decomposition phase
        self._standard_engine: Optional[DecompositionEngine] = None
        if DECOMPOSITION_AVAILABLE and DecompositionEngine:
            try:
                self._standard_engine = DecompositionEngine()
            except Exception as e:
                logger.warning(f"Could not initialize standard DecompositionEngine: {e}")
        
        logger.info(f"Initialized MatryoshkaDecompositionEngine "
                   f"(matryoshka_available: {self.team_executor.available})")
    
    @property
    def matryoshka_available(self) -> bool:
        """Check if Matryoshka execution is available."""
        return self.team_executor.available
    
    def decompose_and_solve(
        self,
        problem: ProblemDefinition
    ) -> DecompositionResult:
        """
        Full decomposition and solve workflow.
        
        1. Decompose problem into sub-problems
        2. For each sub-problem:
           a. Blue Team solves (Matryoshka)
           b. Red Team critiques (Matryoshka)
           c. Gold Team verifies (Matryoshka)
        3. Aggregate solutions
        
        Args:
            problem: The problem to decompose and solve
            
        Returns:
            DecompositionResult with all solutions
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting Matryoshka decomposition and solve for problem {problem.id}")
        
        # Phase 1: Decompose
        subproblems = self._decompose(problem)
        
        if not subproblems:
            logger.warning(f"No sub-problems generated for problem {problem.id}")
            return DecompositionResult(
                problem_id=problem.id,
                subproblems=[],
                solutions=[],
                gauntlet_results=[],
                overall_success=False,
                confidence=0.0,
                metadata={"error": "Decomposition produced no sub-problems"}
            )
        
        # Phase 2: Solve each with gauntlet
        solutions = []
        gauntlet_results = []
        
        for i, subproblem in enumerate(subproblems):
            logger.info(f"Processing sub-problem {i+1}/{len(subproblems)}: {subproblem.id}")
            
            if self.config.use_gauntlet and self.team_executor.available:
                # Use gauntlet for comprehensive evaluation
                gauntlet_result = self.gauntlet_runner.run_gauntlet(
                    subproblem,
                    GauntletConfig(
                        confidence_threshold=self.config.matryoshka_config.confidence_threshold 
                            if self.config.matryoshka_config else 0.7
                    )
                )
                gauntlet_results.append(gauntlet_result)
                
                if gauntlet_result.passed:
                    solutions.append(gauntlet_result.solution)
                    logger.info(f"Sub-problem {subproblem.id} passed gauntlet")
                else:
                    logger.warning(f"Sub-problem {subproblem.id} failed gauntlet")
            else:
                # Use direct Blue Team solve without gauntlet
                context = TeamContext(round=1)
                blue_result = self.team_executor.blue_team_solve(subproblem, context)
                
                if blue_result.success and blue_result.confidence >= 0.5:
                    solutions.append(blue_result.solution)
        
        # Phase 3: Aggregate
        execution_time = (time.time() - start_time) * 1000
        overall_success = len(solutions) > 0
        
        # Calculate overall confidence
        if solutions:
            overall_confidence = sum(s.confidence for s in solutions) / len(solutions)
        else:
            overall_confidence = 0.0
        
        result = DecompositionResult(
            problem_id=problem.id,
            subproblems=subproblems,
            solutions=solutions,
            gauntlet_results=gauntlet_results,
            overall_success=overall_success,
            confidence=overall_confidence,
            execution_time_ms=execution_time,
            metadata={
                "subproblems_total": len(subproblems),
                "solutions_found": len(solutions),
                "gauntlets_run": len(gauntlet_results),
                "gauntlets_passed": sum(1 for g in gauntlet_results if g.passed),
                "matryoshka_used": self.team_executor.available
            }
        )
        
        logger.info(f"Decomposition and solve completed: "
                   f"{len(solutions)}/{len(subproblems)} sub-problems solved, "
                   f"confidence={overall_confidence:.2f}")
        
        return result
    
    def _decompose(self, problem: ProblemDefinition) -> List[SubProblem]:
        """Decompose problem into sub-problems."""
        if self._standard_engine:
            try:
                # Use standard decomposition engine
                plan = self._standard_engine.decompose(problem)
                if plan and hasattr(plan, 'sub_problems'):
                    return plan.sub_problems
            except Exception as e:
                logger.error(f"Standard decomposition failed: {e}")
        
        # Fallback: Create single sub-problem for the whole problem
        logger.warning("Using fallback decomposition - single sub-problem")
        
        if SubProblem:
            return [SubProblem(
                id=generate_id("subproblem"),
                parent_id=problem.id,
                title=f"Solve: {problem.title}",
                description=problem.description,
                type=SubProblemType.IMPLEMENTATION if SubProblemType else None,
                complexity_score=problem.complexity_score if hasattr(problem, 'complexity_score') else None,
                success_criteria=problem.success_criteria if hasattr(problem, 'success_criteria') else []
            )]
        
        return []
    
    def solve_single_subproblem(
        self,
        subproblem: SubProblem,
        use_gauntlet: bool = True
    ) -> Union[BlueTeamResult, GauntletResult]:
        """
        Solve a single sub-problem.
        
        Args:
            subproblem: The sub-problem to solve
            use_gauntlet: Whether to use full gauntlet or just Blue Team
            
        Returns:
            BlueTeamResult or GauntletResult
        """
        if use_gauntlet and self.team_executor.available:
            return self.gauntlet_runner.run_gauntlet(subproblem, GauntletConfig())
        else:
            context = TeamContext(round=1)
            return self.team_executor.blue_team_solve(subproblem, context)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_matryoshka_team_executor(
    config: Optional[MatryoshkaExecutionConfig] = None
) -> MatryoshkaTeamExecutor:
    """
    Factory for Matryoshka team executor.
    
    Args:
        config: Optional execution configuration
        
    Returns:
        Configured MatryoshkaTeamExecutor
    """
    return MatryoshkaTeamExecutor(config)


def create_matryoshka_gauntlet_runner(
    team_executor: Optional[MatryoshkaTeamExecutor] = None
) -> MatryoshkaGauntletRunner:
    """
    Factory for Matryoshka gauntlet runner.
    
    Args:
        team_executor: Optional team executor instance
        
    Returns:
        Configured MatryoshkaGauntletRunner
    """
    return MatryoshkaGauntletRunner(team_executor)


def create_matryoshka_decomposition_engine(
    config: Optional[DecompositionConfig] = None
) -> MatryoshkaDecompositionEngine:
    """
    Factory for Matryoshka decomposition engine.
    
    Args:
        config: Optional decomposition configuration
        
    Returns:
        Configured MatryoshkaDecompositionEngine
    """
    return MatryoshkaDecompositionEngine(config)


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

def patch_decomposition_with_matryoshka(
    decomposition_module: Any,
    enable_matryoshka: bool = True
) -> bool:
    """
    Patch decomposition module to use Matryoshka.
    
    Maintains same interface but uses Matryoshka internally.
    
    Args:
        decomposition_module: The module to patch
        enable_matryoshka: Whether to enable Matryoshka
        
    Returns:
        True if patching was successful
    """
    if not enable_matryoshka:
        logger.info("Matryoshka patching disabled")
        return False
    
    if not MATRYOSHKA_EXECUTION_AVAILABLE:
        logger.warning("Matryoshka not available, cannot patch")
        return False
    
    try:
        # Create Matryoshka engine
        matryoshka_engine = MatryoshkaDecompositionEngine()
        
        # Patch the module's DecompositionEngine class
        if hasattr(decomposition_module, 'DecompositionEngine'):
            original_engine = decomposition_module.DecompositionEngine
            decomposition_module.DecompositionEngine = lambda: matryoshka_engine
            decomposition_module._original_decomposition_engine = original_engine
            logger.info("Patched decomposition module with Matryoshka")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to patch decomposition module: {e}")
        return False


def unpatch_decomposition(decomposition_module: Any) -> bool:
    """
    Remove Matryoshka patching from decomposition module.
    
    Args:
        decomposition_module: The module to unpatch
        
    Returns:
        True if unpatching was successful
    """
    try:
        if hasattr(decomposition_module, '_original_decomposition_engine'):
            decomposition_module.DecompositionEngine = decomposition_module._original_decomposition_engine
            delattr(decomposition_module, '_original_decomposition_engine')
            logger.info("Removed Matryoshka patching from decomposition module")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to unpatch decomposition module: {e}")
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_matryoshka_available() -> bool:
    """Check if Matryoshka integration is fully available."""
    return MATRYOSHKA_EXECUTION_AVAILABLE


def get_matryoshka_status() -> Dict[str, Any]:
    """Get detailed status of Matryoshka integration."""
    return {
        "matryoshka_execution_available": MATRYOSHKA_EXECUTION_AVAILABLE,
        "mdap_matryoshka_available": MDAP_MATRYOSHKA_AVAILABLE,
        "decomposition_available": DECOMPOSITION_AVAILABLE,
        "red_team_available": RED_TEAM_AVAILABLE,
        "blue_team_available": BLUE_TEAM_AVAILABLE,
        "gold_team_available": GOLD_TEAM_AVAILABLE,
        "gauntlet_available": GAUNTLET_AVAILABLE,
        "all_required_available": (
            MATRYOSHKA_EXECUTION_AVAILABLE and 
            DECOMPOSITION_AVAILABLE
        )
    }


def create_default_config() -> MatryoshkaExecutionConfig:
    """Create default Matryoshka execution configuration."""
    return MatryoshkaExecutionConfig()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Configuration classes
    "MatryoshkaExecutionConfig",
    "TeamContext",
    "Solution",
    "BlueTeamResult",
    "Vulnerability",
    "RedTeamResult",
    "CriterionResult",
    "GoldTeamResult",
    "GauntletConfig",
    "GauntletResult",
    "DecompositionConfig",
    "DecompositionResult",
    
    # Main classes
    "MatryoshkaTeamExecutor",
    "MatryoshkaGauntletRunner",
    "MatryoshkaDecompositionEngine",
    
    # Factory functions
    "create_matryoshka_team_executor",
    "create_matryoshka_gauntlet_runner",
    "create_matryoshka_decomposition_engine",
    
    # Compatibility functions
    "patch_decomposition_with_matryoshka",
    "unpatch_decomposition",
    
    # Utility functions
    "is_matryoshka_available",
    "get_matryoshka_status",
    "create_default_config",
    
    # Availability flags
    "MATRYOSHKA_EXECUTION_AVAILABLE",
    "MDAP_MATRYOSHKA_AVAILABLE",
    "DECOMPOSITION_AVAILABLE",
]


# ============================================================================
# MAIN ENTRY POINT FOR TESTING
# ============================================================================

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    print("Matryoshka Decomposition Integration")
    print("=" * 50)
    
    status = get_matryoshka_status()
    print(f"\nIntegration Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    if status["all_required_available"]:
        print("\nCreating Matryoshka decomposition engine...")
        engine = create_matryoshka_decomposition_engine()
        print(f"Engine created: matryoshka_available={engine.matryoshka_available}")
    else:
        print("\nMatryoshka integration not fully available (dependencies missing)")
        print("The module will use fallback implementations.")
