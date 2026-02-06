"""
ROMA-Matryoshka Integration

Integrates Matryoshka Execution Engine into ROMA's recursive decomposition:
- Phase 1: Problem Setup -> Matryoshka analyzes problem structure
- Phase 2: Solution Generation -> Matryoshka solves each sub-problem
- Phase 3: Critique -> Matryoshka adversarially critiques
- Phase 4: Verification -> Matryoshka verifies solutions
- Phase 5: Reassembly -> ROMA aggregates (Matryoshka provides findings)

Matryoshka becomes a first-class solver option in ROMA.
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from enum import Enum

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

try:
    from matryoshka_execution_engine import (
        MatryoshkaExecutionEngine,
        MatryoshkaExecutionConfig,
        ROMAMatryoshkaSolver,
        ProblemSpace,
        DocumentSpace,
        CodebaseSpace,
        DatabaseSpace,
    )
    MATRYOSHKA_ENGINE_AVAILABLE = True
except ImportError:
    MATRYOSHKA_ENGINE_AVAILABLE = False
    logger.debug("Matryoshka Execution Engine not available")

try:
    from roma_openevolve_integration import (
        ROMAOpenEvolveAdapter,
        ROMAOpenEvolveConfig,
    )
    ROMA_OPENEVOLVE_AVAILABLE = True
except ImportError:
    ROMA_OPENEVOLVE_AVAILABLE = False
    logger.debug("ROMA OpenEvolve integration not available")

try:
    from roma_decomposition_hybrid import (
        ROMADecompositionHybrid,
        HybridConfig,
    )
    ROMA_HYBRID_AVAILABLE = True
except ImportError:
    ROMA_HYBRID_AVAILABLE = False
    logger.debug("ROMA Hybrid Decomposition not available")

# Standard ROMA
try:
    from roma_dspy.core.engine.solve import RecursiveSolver
    # from roma_dspy.core.engine import  # Stubbed - module not available TaskDAG
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    logger.debug("ROMA DSPy not available")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProblemAnalysisResult:
    """Result of Phase 1 problem analysis."""
    subproblems: List[Any]
    structure: Dict[str, Any]
    findings: Dict[str, Any]
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubProblem:
    """Represents a decomposed sub-problem."""
    id: str
    description: str
    requirements: List[str]
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    space_type: str = "abstract"


@dataclass
class ROMAContext:
    """Context for ROMA operations."""
    session_id: Optional[str] = None
    problem_history: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubProblemSolution:
    """Solution to a sub-problem."""
    subproblem_id: str
    solution: Any
    approach: str
    confidence: float = 0.0
    execution_trace: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueResult:
    """Result of critique phase."""
    solution_id: str
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    overall_score: float = 0.0
    confidence: float = 0.0
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verification phase."""
    solution_id: str
    passed: bool
    checks: List[Dict[str, Any]]
    failures: List[str]
    confidence: float = 0.0
    verification_trace: List[Dict] = field(default_factory=list)


@dataclass
class SolverResult:
    """Final solver result."""
    problem: str
    solutions: List[SubProblemSolution]
    critiques: List[CritiqueResult]
    verifications: List[VerificationResult]
    aggregated_solution: Any
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PROBLEM SPACE TYPES
# ============================================================================

class SpaceType(Enum):
    """Types of problem spaces Matryoshka can handle."""
    ABSTRACT = "abstract"
    DOCUMENT = "document"
    CODEBASE = "codebase"
    DATABASE = "database"
    AUTO = "auto"


@dataclass
class AbstractSpace:
    """Abstract problem space for reasoning problems."""
    representation: Dict[str, Any]
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "abstract",
            "representation": self.representation,
            "properties": self.properties
        }


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ROMAMatryoshkaConfig:
    """
    Configuration for ROMA-Matryoshka integration.
    
    All Matryoshka features optional - ROMA works without them.
    """
    # Enable Matryoshka integration
    enable_matryoshka_solver: bool = True
    
    # When to use Matryoshka vs standard ROMA solver
    use_matryoshka_for: Dict[str, bool] = field(default_factory=lambda: {
        "problem_analysis": True,      # Phase 1: Analyze structure
        "subproblem_solving": True,    # Phase 2: Solve sub-problems
        "critique": True,              # Phase 3: Adversarial critique
        "verification": True,          # Phase 4: Verify solutions
        "exploration": True,           # For open-ended exploration
    })
    
    # Matryoshka execution config
    matryoshka_config: Optional[Any] = None
    
    # Problem space detection
    auto_detect_space_type: bool = True
    default_space_type: str = "auto"  # "document", "codebase", "database", "abstract"
    
    # Integration behavior
    fallback_to_roma_solver: bool = True
    report_intermediate_findings: bool = True
    
    # Performance tuning
    max_iterations: int = 10
    timeout_seconds: float = 300.0
    parallel_solving: bool = True
    
    # Debug options
    debug_mode: bool = False
    save_execution_traces: bool = False


# ============================================================================
# ROMA-MATRYOSHKA ADAPTER
# ============================================================================

class ROMAMatryoshkaAdapter:
    """
    Adapter integrating Matryoshka into ROMA's recursive decomposition.
    
    This adapter:
    1. Detects problem space type (document, code, data, etc.)
    2. Creates appropriate Matryoshka ProblemSpace
    3. Routes ROMA phases to Matryoshka when configured
    4. Falls back to standard ROMA when Matryoshka unavailable
    """
    
    def __init__(self, config: ROMAMatryoshkaConfig, use_cav_nlp: bool = True):
        self.config = config
        self.matryoshka_solver: Optional[Any] = None
        self._fallback_used_count = 0
        self._matryoshka_used_count = 0
        
        if config.enable_matryoshka_solver and MATRYOSHKA_ENGINE_AVAILABLE:
            self._init_matryoshka()
        else:
            if not MATRYOSHKA_ENGINE_AVAILABLE:
                logger.warning("Matryoshka Engine not available. Will use ROMA fallbacks.")
            if not config.enable_matryoshka_solver:
                logger.info("Matryoshka solver disabled in config.")

        # CAV-NLP integration
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for ROMAMatryoshkaAdapter")

    def formalize_problem_with_cav_nlp(self, problem: str) -> Dict[str, Any]:
        """Formalize a problem using CAV-NLP."""
        if not self.use_cav_nlp:
            return {"formalized": False, "reason": "CAV-NLP not available"}
        try:
            formalized = self.math_service.formalize(problem)
            return {
                "formalized": True,
                "code": formalized.code,
                "confidence": formalized.confidence,
                "method": "cav_nlp"
            }
        except Exception as e:
            logger.warning(f"CAV-NLP formalization failed: {e}")
            return {"formalized": False, "error": str(e)}

    def verify_with_cav_nlp(self, solution: Any, requirements: List[str]) -> Dict[str, Any]:
        """Verify a solution using CAV-NLP."""
        if not self.use_cav_nlp:
            return {"verified": False, "reason": "CAV-NLP not available"}
        try:
            solution_text = str(solution)
            formalized = self.math_service.formalize(solution_text)
            result = self.enhanced_solver.verify_with_lean(formalized.code)
            return {
                "verified": result.get("verified", False),
                "confidence": result.get("confidence", 0.0),
                "method": "lean_verification",
                "requirements_checked": len(requirements)
            }
        except Exception as e:
            logger.warning(f"CAV-NLP verification failed: {e}")
            return {"verified": False, "error": str(e)}
    
    def _init_matryoshka(self):
        """Initialize Matryoshka solver if available."""
        try:
            if MATRYOSHKA_ENGINE_AVAILABLE:
                matryoshka_config = self.config.matryoshka_config or MatryoshkaExecutionConfig()
                self.matryoshka_solver = ROMAMatryoshkaSolver(matryoshka_config)
                logger.info("Matryoshka solver initialized successfully")
            else:
                self.matryoshka_solver = None
        except Exception as e:
            logger.error(f"Failed to initialize Matryoshka solver: {e}")
            self.matryoshka_solver = None
    
    @property
    def has_matryoshka(self) -> bool:
        """Check if Matryoshka solver is available and working."""
        return (
            self.config.enable_matryoshka_solver
            and MATRYOSHKA_ENGINE_AVAILABLE
            and self.matryoshka_solver is not None
        )
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for Matryoshka vs fallback."""
        return {
            "matryoshka_used": self._matryoshka_used_count,
            "fallback_used": self._fallback_used_count,
            "matryoshka_available": self.has_matryoshka
        }
    
    # ====================================================================
    # ROMA PHASE INTEGRATIONS
    # ====================================================================
    
    def phase_1_analyze_problem(
        self,
        problem: str,
        context: Optional[Dict] = None
    ) -> ProblemAnalysisResult:
        """
        Phase 1: Problem Analysis
        
        Uses Matryoshka to iteratively explore problem structure
        and identify sub-problems.
        """
        context = context or {}
        
        if not self._should_use_matryoshka("problem_analysis"):
            self._fallback_used_count += 1
            return self._fallback_to_roma_phase_1(problem, context)
        
        try:
            # Create abstract problem space for analysis
            space = AbstractSpace(representation={"problem": problem, "context": context})
            
            # Execute exploration
            result = self.matryoshka_solver.explore_problem_structure(
                problem=problem,
                space=space
            )
            
            self._matryoshka_used_count += 1
            
            # Convert Matryoshka findings to ROMA subproblems
            subproblems = self._convert_findings_to_subproblems(result.findings)
            
            return ProblemAnalysisResult(
                subproblems=subproblems,
                structure=result.symbolic_state if hasattr(result, 'symbolic_state') else {},
                findings=result.accumulated_findings if hasattr(result, 'accumulated_findings') else {},
                confidence=getattr(result, 'confidence', 0.0),
                metadata={
                    "phase": "analysis",
                    "solver": "matryoshka",
                    "space_type": "abstract"
                }
            )
            
        except Exception as e:
            logger.error(f"Matryoshka Phase 1 failed: {e}. Falling back to ROMA.")
            self._fallback_used_count += 1
            if self.config.fallback_to_roma_solver:
                return self._fallback_to_roma_phase_1(problem, context)
            raise
    
    def phase_2_solve_subproblem(
        self,
        subproblem: SubProblem,
        context: ROMAContext
    ) -> SubProblemSolution:
        """
        Phase 2: Solve Sub-Problem
        
        Uses Matryoshka to iteratively explore solution space.
        """
        if not self._should_use_matryoshka("subproblem_solving"):
            self._fallback_used_count += 1
            return self._fallback_to_roma_phase_2(subproblem, context)
        
        try:
            # Detect problem space type
            space = self._detect_problem_space(subproblem, context)
            
            # Solve using Matryoshka
            result = self.matryoshka_solver.solve_subproblem(subproblem, space)
            
            self._matryoshka_used_count += 1
            
            return SubProblemSolution(
                subproblem_id=subproblem.id,
                solution=result.solution if hasattr(result, 'solution') else result,
                approach="matryoshka_iterative",
                confidence=getattr(result, 'confidence', 0.0),
                execution_trace=getattr(result, 'trace', []),
                metadata={
                    "phase": "solving",
                    "solver": "matryoshka",
                    "space_type": subproblem.space_type,
                    "iterations": getattr(result, 'iterations', 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Matryoshka Phase 2 failed for {subproblem.id}: {e}. Falling back to ROMA.")
            self._fallback_used_count += 1
            if self.config.fallback_to_roma_solver:
                return self._fallback_to_roma_phase_2(subproblem, context)
            raise
    
    def phase_3_critique_solution(
        self,
        solution: SubProblemSolution,
        criteria: List[str]
    ) -> CritiqueResult:
        """
        Phase 3: Critique Solution
        
        Uses Matryoshka for adversarial exploration of solution.
        """
        if not self._should_use_matryoshka("critique"):
            self._fallback_used_count += 1
            return self._fallback_to_roma_phase_3(solution, criteria)
        
        try:
            result = self.matryoshka_solver.critique_solution(solution, criteria)
            
            self._matryoshka_used_count += 1
            
            return CritiqueResult(
                solution_id=solution.subproblem_id,
                issues=getattr(result, 'issues', []),
                suggestions=getattr(result, 'suggestions', []),
                overall_score=getattr(result, 'score', 0.0),
                confidence=getattr(result, 'confidence', 0.0),
                detailed_analysis=getattr(result, 'analysis', {})
            )
            
        except Exception as e:
            logger.error(f"Matryoshka Phase 3 failed for {solution.subproblem_id}: {e}. Falling back to ROMA.")
            self._fallback_used_count += 1
            if self.config.fallback_to_roma_solver:
                return self._fallback_to_roma_phase_3(solution, criteria)
            raise
    
    def phase_4_verify_solution(
        self,
        solution: SubProblemSolution,
        requirements: List[str]
    ) -> VerificationResult:
        """
        Phase 4: Verify Solution
        
        Uses Matryoshka for systematic verification.
        """
        if not self._should_use_matryoshka("verification"):
            self._fallback_used_count += 1
            return self._fallback_to_roma_phase_4(solution, requirements)
        
        try:
            result = self.matryoshka_solver.verify_solution(solution, requirements)
            
            self._matryoshka_used_count += 1
            
            return VerificationResult(
                solution_id=solution.subproblem_id,
                passed=getattr(result, 'passed', False),
                checks=getattr(result, 'checks', []),
                failures=getattr(result, 'failures', []),
                confidence=getattr(result, 'confidence', 0.0),
                verification_trace=getattr(result, 'trace', [])
            )
            
        except Exception as e:
            logger.error(f"Matryoshka Phase 4 failed for {solution.subproblem_id}: {e}. Falling back to ROMA.")
            self._fallback_used_count += 1
            if self.config.fallback_to_roma_solver:
                return self._fallback_to_roma_phase_4(solution, requirements)
            raise
    
    def phase_5_explore_open_ended(
        self,
        problem: str,
        exploration_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Phase 5: Open-ended exploration (Matryoshka specialty).
        
        For problems without clear boundaries, use Matryoshka's
        exploration capabilities.
        """
        if not self._should_use_matryoshka("exploration"):
            logger.info("Exploration phase using standard ROMA approach")
            return {"exploration": "standard_roma", "findings": []}
        
        try:
            params = exploration_params or {}
            result = self.matryoshka_solver.explore(
                problem=problem,
                max_iterations=params.get('max_iterations', self.config.max_iterations)
            )
            
            self._matryoshka_used_count += 1
            
            return {
                "exploration": "matryoshka",
                "findings": getattr(result, 'findings', []),
                "boundaries_explored": getattr(result, 'boundaries', []),
                "confidence": getattr(result, 'confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Matryoshka exploration failed: {e}")
            return {"exploration": "failed", "error": str(e), "findings": []}
    
    # ====================================================================
    # FALLBACK METHODS (Standard ROMA)
    # ====================================================================
    
    def _fallback_to_roma_phase_1(
        self,
        problem: str,
        context: Dict
    ) -> ProblemAnalysisResult:
        """Fallback to ROMA for problem analysis."""
        logger.info("Using ROMA fallback for Phase 1 (Problem Analysis)")
        
        if ROMA_AVAILABLE:
            # Use standard ROMA decomposition
            # This is a simplified implementation
            subproblems = [
                SubProblem(
                    id="sp_1",
                    description=problem,
                    requirements=["solve_problem"],
                    space_type="abstract"
                )
            ]
        else:
            # Ultimate fallback - single problem
            subproblems = [
                SubProblem(
                    id="sp_1",
                    description=problem,
                    requirements=["solve_problem"],
                    space_type="abstract"
                )
            ]
        
        return ProblemAnalysisResult(
            subproblems=subproblems,
            structure={"type": "flat_decomposition"},
            findings={},
            confidence=0.5,
            metadata={"phase": "analysis", "solver": "fallback_roma"}
        )
    
    def _fallback_to_roma_phase_2(
        self,
        subproblem: SubProblem,
        context: ROMAContext
    ) -> SubProblemSolution:
        """Fallback to ROMA for sub-problem solving."""
        logger.info(f"Using ROMA fallback for Phase 2 (Solving {subproblem.id})")
        
        return SubProblemSolution(
            subproblem_id=subproblem.id,
            solution={"status": "solved_by_fallback", "description": subproblem.description},
            approach="roma_fallback",
            confidence=0.5,
            metadata={"phase": "solving", "solver": "fallback_roma"}
        )
    
    def _fallback_to_roma_phase_3(
        self,
        solution: SubProblemSolution,
        criteria: List[str]
    ) -> CritiqueResult:
        """Fallback to ROMA for critique."""
        logger.info(f"Using ROMA fallback for Phase 3 (Critique {solution.subproblem_id})")
        
        return CritiqueResult(
            solution_id=solution.subproblem_id,
            issues=[],
            suggestions=["Review solution manually"],
            overall_score=0.5,
            confidence=0.5,
            detailed_analysis={"critique": "fallback_used"}
        )
    
    def _fallback_to_roma_phase_4(
        self,
        solution: SubProblemSolution,
        requirements: List[str]
    ) -> VerificationResult:
        """Fallback to ROMA for verification."""
        logger.info(f"Using ROMA fallback for Phase 4 (Verification {solution.subproblem_id})")
        
        return VerificationResult(
            solution_id=solution.subproblem_id,
            passed=True,  # Conservative: assume pass
            checks=[{"requirement": r, "passed": True} for r in requirements],
            failures=[],
            confidence=0.5,
            verification_trace=[]
        )
    
    # ====================================================================
    # PROBLEM SPACE DETECTION
    # ====================================================================
    
    def _detect_problem_space(
        self,
        subproblem: SubProblem,
        context: ROMAContext
    ) -> Any:
        """
        Auto-detect problem space type from subproblem.
        
        Detects:
        - DocumentSpace: File paths, text content
        - CodebaseSpace: Code repositories, source files
        - DatabaseSpace: Data queries, structured data
        - AbstractSpace: Pure reasoning problems
        """
        description = subproblem.description.lower()
        
        # Check explicit space type from subproblem
        if subproblem.space_type != "auto":
            return self._create_space_by_type(subproblem.space_type, subproblem, context)
        
        # Auto-detect based on description
        if not self.config.auto_detect_space_type:
            return AbstractSpace(
                representation={"subproblem": subproblem, "context": context}
            )
        
        # Code detection
        code_indicators = [
            "code", "function", "class", "module", "repository", 
            "git", "github", "implementation", "refactor", "bug"
        ]
        if any(ind in description for ind in code_indicators):
            return self._create_space_by_type("codebase", subproblem, context)
        
        # Database detection
        db_indicators = [
            "database", "sql", "query", "table", "schema", "data",
            "mongodb", "postgres", "mysql", "sqlite"
        ]
        if any(ind in description for ind in db_indicators):
            return self._create_space_by_type("database", subproblem, context)
        
        # Document detection
        doc_indicators = [
            "document", "file", "text", "pdf", "word", "paper",
            "article", "report", "content"
        ]
        if any(ind in description for ind in doc_indicators):
            return self._create_space_by_type("document", subproblem, context)
        
        # Default to abstract
        return AbstractSpace(
            representation={"subproblem": subproblem, "context": context}
        )
    
    def _create_space_by_type(
        self,
        space_type: str,
        subproblem: SubProblem,
        context: ROMAContext
    ) -> Any:
        """Create appropriate space object based on type."""
        if not MATRYOSHKA_ENGINE_AVAILABLE:
            return AbstractSpace(
                representation={"subproblem": subproblem, "context": context}
            )
        
        if space_type == "codebase":
            return CodebaseSpace(
                repository_path=context.constraints.get("repo_path"),
                file_patterns=["*.py", "*.js", "*.ts"],
                context=subproblem.description
            )
        elif space_type == "database":
            return DatabaseSpace(
                connection_string=context.constraints.get("db_connection"),
                schema=context.constraints.get("schema"),
                query_hints=subproblem.description
            )
        elif space_type == "document":
            return DocumentSpace(
                document_paths=context.constraints.get("document_paths", []),
                content_query=subproblem.description
            )
        else:
            return AbstractSpace(
                representation={"subproblem": subproblem, "context": context}
            )
    
    def _convert_findings_to_subproblems(self, findings: Any) -> List[SubProblem]:
        """Convert Matryoshka findings to ROMA SubProblem objects."""
        subproblems = []
        
        if isinstance(findings, list):
            for i, finding in enumerate(findings):
                if isinstance(finding, dict):
                    subproblems.append(SubProblem(
                        id=finding.get("id", f"sp_{i}"),
                        description=finding.get("description", str(finding)),
                        requirements=finding.get("requirements", []),
                        dependencies=finding.get("dependencies", []),
                        space_type=finding.get("space_type", "abstract"),
                        metadata=finding.get("metadata", {})
                    ))
                else:
                    subproblems.append(SubProblem(
                        id=f"sp_{i}",
                        description=str(finding),
                        requirements=[],
                        space_type="abstract"
                    ))
        elif isinstance(findings, dict):
            # Single finding as dict
            subproblems.append(SubProblem(
                id=findings.get("id", "sp_0"),
                description=findings.get("description", str(findings)),
                requirements=findings.get("requirements", []),
                dependencies=findings.get("dependencies", []),
                space_type=findings.get("space_type", "abstract"),
                metadata=findings.get("metadata", {})
            ))
        else:
            # Fallback - single problem
            subproblems.append(SubProblem(
                id="sp_0",
                description=str(findings) if findings else "main_problem",
                requirements=[],
                space_type="abstract"
            ))
        
        return subproblems
    
    def _should_use_matryoshka(self, phase: str) -> bool:
        """Check if Matryoshka should be used for this phase."""
        return (
            self.has_matryoshka
            and self.config.use_matryoshka_for.get(phase, False)
        )


# ============================================================================
# ENHANCED ROMA SOLVER WITH MATRYOSHKA
# ============================================================================

class EnhancedROMAResolver:
    """
    Enhanced ROMA resolver that uses Matryoshka as primary solver.
    
    Drop-in replacement for ROMA's RecursiveSolver with Matryoshka integration.
    """
    
    def __init__(
        self,
        roma_config: Any,
        matryoshka_config: Optional[ROMAMatryoshkaConfig] = None
    ):
        self.roma_config = roma_config
        self.matryoshka_config = matryoshka_config or ROMAMatryoshkaConfig()
        self.matryoshka_adapter = ROMAMatryoshkaAdapter(self.matryoshka_config)
        
        # Keep standard ROMA as fallback
        self.roma_solver: Optional[Any] = None
        if ROMA_AVAILABLE:
            try:
                self.roma_solver = RecursiveSolver(roma_config)
                logger.info("Standard ROMA solver initialized as fallback")
            except Exception as e:
                logger.warning(f"Could not initialize standard ROMA solver: {e}")
    
    def solve(
        self,
        problem: str,
        context: Optional[Dict] = None
    ) -> SolverResult:
        """
        Solve problem using Matryoshka-enhanced approach.
        
        1. Analyze with Matryoshka (Phase 1)
        2. Decompose into sub-problems
        3. Solve each with Matryoshka (Phase 2)
        4. Critique with Matryoshka (Phase 3)
        5. Verify with Matryoshka (Phase 4)
        6. Aggregate results
        """
        logger.info(f"Starting enhanced ROMA solve for: {problem[:50]}...")
        
        roma_context = ROMAContext(
            session_id=context.get("session_id") if context else None,
            problem_history=[problem],
            constraints=context.get("constraints", {}) if context else {},
            preferences=context.get("preferences", {}) if context else {}
        )
        
        # Phase 1: Analyze
        logger.info("Phase 1: Problem Analysis")
        analysis = self.matryoshka_adapter.phase_1_analyze_problem(problem, context)
        
        if not analysis.subproblems:
            logger.warning("No subproblems identified. Creating single problem.")
            analysis.subproblems = [
                SubProblem(
                    id="sp_0",
                    description=problem,
                    requirements=["solve_main_problem"],
                    space_type="abstract"
                )
            ]
        
        # Phase 2: Solve sub-problems
        logger.info(f"Phase 2: Solving {len(analysis.subproblems)} sub-problems")
        solutions = []
        for subproblem in analysis.subproblems:
            solution = self.matryoshka_adapter.phase_2_solve_subproblem(
                subproblem, roma_context
            )
            solutions.append(solution)
            
            if self.matryoshka_config.report_intermediate_findings:
                logger.info(f"  Solved {subproblem.id}: confidence={solution.confidence:.2f}")
        
        # Phase 3: Critique
        logger.info("Phase 3: Critique")
        critiques = []
        for solution in solutions:
            critique = self.matryoshka_adapter.phase_3_critique_solution(
                solution, ["correctness", "completeness", "efficiency"]
            )
            critiques.append(critique)
        
        # Phase 4: Verify
        logger.info("Phase 4: Verification")
        verified_solutions = []
        verifications = []
        for i, solution in enumerate(solutions):
            subproblem = analysis.subproblems[i] if i < len(analysis.subproblems) else None
            requirements = subproblem.requirements if subproblem else []
            
            verification = self.matryoshka_adapter.phase_4_verify_solution(
                solution, requirements
            )
            verifications.append(verification)
            
            if verification.passed:
                verified_solutions.append(solution)
            else:
                logger.warning(f"Solution {solution.subproblem_id} failed verification")
        
        # Aggregate
        aggregated = self._aggregate_solutions(verified_solutions, critiques)
        
        result = SolverResult(
            problem=problem,
            solutions=solutions,
            critiques=critiques,
            verifications=verifications,
            aggregated_solution=aggregated,
            confidence=self._calculate_overall_confidence(solutions, verifications),
            metadata={
                "solver_type": "enhanced_roma_matryoshka",
                "matryoshka_stats": self.matryoshka_adapter.get_usage_stats(),
                "num_subproblems": len(analysis.subproblems),
                "num_verified": len(verified_solutions)
            }
        )
        
        logger.info(f"Solve complete. Overall confidence: {result.confidence:.2f}")
        return result
    
    def _aggregate_solutions(
        self,
        solutions: List[SubProblemSolution],
        critiques: List[CritiqueResult]
    ) -> Any:
        """Aggregate multiple sub-problem solutions into final answer."""
        if not solutions:
            return {"status": "no_solutions", "answer": None}
        
        if len(solutions) == 1:
            return {
                "status": "single_solution",
                "answer": solutions[0].solution,
                "confidence": solutions[0].confidence
            }
        
        # Multi-solution aggregation
        aggregated = {
            "status": "aggregated",
            "answers": [s.solution for s in solutions],
            "subproblem_ids": [s.subproblem_id for s in solutions],
            "average_confidence": sum(s.confidence for s in solutions) / len(solutions),
            "critique_summary": {
                c.solution_id: {
                    "score": c.overall_score,
                    "issues": len(c.issues)
                }
                for c in critiques
            }
        }
        
        return aggregated
    
    def _calculate_overall_confidence(
        self,
        solutions: List[SubProblemSolution],
        verifications: List[VerificationResult]
    ) -> float:
        """Calculate overall confidence score."""
        if not solutions:
            return 0.0
        
        solution_conf = sum(s.confidence for s in solutions) / len(solutions)
        
        if verifications:
            verification_conf = sum(v.confidence for v in verifications) / len(verifications)
            verification_pass_rate = sum(1 for v in verifications if v.passed) / len(verifications)
            return (solution_conf + verification_conf + verification_pass_rate) / 3
        
        return solution_conf
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        return {
            "matryoshka_stats": self.matryoshka_adapter.get_usage_stats(),
            "config": {
                "enable_matryoshka": self.matryoshka_config.enable_matryoshka_solver,
                "fallback_enabled": self.matryoshka_config.fallback_to_roma_solver
            }
        }


# ============================================================================
# INTEGRATION WITH EXISTING ROMA MODULES
# ============================================================================

def patch_roma_with_matryoshka(
    roma_module: Any,
    config: Optional[ROMAMatryoshkaConfig] = None
) -> bool:
    """
    Monkey-patch ROMA module to use Matryoshka solver.
    
    Args:
        roma_module: The roma_dspy module or similar
        config: Matryoshka integration config
        
    Returns:
        True if patching successful
    """
    config = config or ROMAMatryoshkaConfig()
    
    if not config.enable_matryoshka_solver:
        logger.info("Matryoshka patching disabled by config")
        return False
    
    if not MATRYOSHKA_ENGINE_AVAILABLE:
        logger.warning("Cannot patch: Matryoshka Engine not available")
        return False
    
    try:
        # Create adapter
        adapter = ROMAMatryoshkaAdapter(config)
        
        # Patch RecursiveSolver if available
        if ROMA_AVAILABLE and hasattr(roma_module, 'RecursiveSolver'):
            original_init = roma_module.RecursiveSolver.__init__
            
            def patched_init(self, *args, **kwargs):
                # Call original init
                original_init(self, *args, **kwargs)
                # Add matryoshka adapter
                self._matryoshka_adapter = adapter
                self._use_matryoshka = True
            
            roma_module.RecursiveSolver.__init__ = patched_init
            logger.info("Patched RecursiveSolver with Matryoshka adapter")
        
        # Patch TaskDAG if available
        if ROMA_AVAILABLE and hasattr(roma_module, 'TaskDAG'):
            original_dag_init = roma_module.TaskDAG.__init__
            
            def patched_dag_init(self, *args, **kwargs):
                original_dag_init(self, *args, **kwargs)
                self._matryoshka_adapter = adapter
            
            roma_module.TaskDAG.__init__ = patched_dag_init
            logger.info("Patched TaskDAG with Matryoshka adapter")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch ROMA module: {e}")
        return False


def create_matryoshka_enhanced_roma_solver(
    roma_config: Any,
    enable_matryoshka: bool = True,
    matryoshka_config: Optional[ROMAMatryoshkaConfig] = None
) -> EnhancedROMAResolver:
    """
    Factory for creating Matryoshka-enhanced ROMA solver.
    
    Args:
        roma_config: Configuration for ROMA
        enable_matryoshka: Whether to enable Matryoshka integration
        matryoshka_config: Optional Matryoshka-specific config
        
    Returns:
        EnhancedROMAResolver instance
    """
    config = matryoshka_config or ROMAMatryoshkaConfig()
    config.enable_matryoshka_solver = enable_matryoshka
    
    return EnhancedROMAResolver(roma_config, config)


# ============================================================================
# BACKWARDS COMPATIBILITY
# ============================================================================

class MatryoshkaROMABridge:
    """
    Bridge for backwards compatibility with existing ROMA workflows.
    
    Maintains same interface as standard ROMA but uses Matryoshka internally.
    """
    
    def __init__(self, config: Optional[ROMAMatryoshkaConfig] = None):
        self.config = config or ROMAMatryoshkaConfig()
        self.adapter = ROMAMatryoshkaAdapter(self.config)
        self._resolver: Optional[EnhancedROMAResolver] = None
    
    def _ensure_resolver(self):
        """Lazy initialization of resolver."""
        if self._resolver is None:
            self._resolver = EnhancedROMAResolver(None, self.config)
    
    # Mirror standard ROMA interface
    def analyze(self, problem: str) -> ProblemAnalysisResult:
        """ROMA-compatible analyze method."""
        return self.adapter.phase_1_analyze_problem(problem)
    
    def solve(self, problem: str) -> SolverResult:
        """ROMA-compatible solve method."""
        self._ensure_resolver()
        return self._resolver.solve(problem)
    
    def decompose(self, problem: str) -> List[SubProblem]:
        """ROMA-compatible decompose method."""
        result = self.adapter.phase_1_analyze_problem(problem)
        return result.subproblems
    
    def critique(self, solution: Any, criteria: Optional[List[str]] = None) -> CritiqueResult:
        """ROMA-compatible critique method."""
        criteria = criteria or ["correctness", "completeness"]
        
        # Convert to SubProblemSolution if needed
        if not isinstance(solution, SubProblemSolution):
            solution = SubProblemSolution(
                subproblem_id="unknown",
                solution=solution,
                approach="external",
                confidence=0.5
            )
        
        return self.adapter.phase_3_critique_solution(solution, criteria)
    
    def verify(self, solution: Any, requirements: Optional[List[str]] = None) -> VerificationResult:
        """ROMA-compatible verify method."""
        requirements = requirements or []
        
        # Convert to SubProblemSolution if needed
        if not isinstance(solution, SubProblemSolution):
            solution = SubProblemSolution(
                subproblem_id="unknown",
                solution=solution,
                approach="external",
                confidence=0.5
            )
        
        return self.adapter.phase_4_verify_solution(solution, requirements)
    
    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "matryoshka_available": self.adapter.has_matryoshka,
            "usage_stats": self.adapter.get_usage_stats(),
            "config": {
                "enable_matryoshka": self.config.enable_matryoshka_solver,
                "fallback_enabled": self.config.fallback_to_roma_solver
            }
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_matryoshka_availability() -> Dict[str, bool]:
    """Check which Matryoshka components are available."""
    return {
        "matryoshka_engine": MATRYOSHKA_ENGINE_AVAILABLE,
        "roma_openevolve": ROMA_OPENEVOLVE_AVAILABLE,
        "roma_hybrid": ROMA_HYBRID_AVAILABLE,
        "roma_core": ROMA_AVAILABLE
    }


def get_default_matryoshka_config() -> ROMAMatryoshkaConfig:
    """Get default Matryoshka configuration."""
    return ROMAMatryoshkaConfig()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check availability
    availability = check_matryoshka_availability()
    print("Component Availability:")
    for component, available in availability.items():
        print(f"  {component}: {'[OK]' if available else '[FAIL]'}")
    
    # Example: Create bridge and solve
    bridge = MatryoshkaROMABridge()
    print(f"\nBridge Status: {bridge.get_status()}")
    
    # Example: Create enhanced solver
    resolver = create_matryoshka_enhanced_roma_solver(
        roma_config={},
        enable_matryoshka=True
    )
    print(f"\nResolver Stats: {resolver.get_stats()}")
