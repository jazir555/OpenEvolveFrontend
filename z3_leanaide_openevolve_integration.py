"""
Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration

This module provides comprehensive integration between:
- Z3 SMT Solver (constraint solving, theorem proving)
- LeanAIDE (formal verification, proof generation)
- OpenEvolve (evolutionary workflow engine)
- BubbleLabs (visualization and UI)

This integration enables:
- Automatic problem classification (constraint vs theorem vs standard)
- Adaptive solver selection (Z3 for constraints, Lean for theorems)
- Enhanced verification workflows
- Visual workflow tracking in BubbleLabs
- Combined proof generation and verification

Architecture:
    OpenEvolve Workflow
        └── Z3LeanAideOpenEvolveIntegration
            ├── Z3 Solver (constraints, optimization)
            ├── LeanAIDE (theorem proving)
            ├── Combined Verification
            └── BubbleLabs Visualization

Workflow Stages Integration:
- Stage 1: Problem Classification (Z3 vs Lean vs Standard)
- Stage 2: Decomposition (with formal structure awareness)
- Stage 3: Solution Generation (adaptive solver selection)
- Stage 4: Verification (combined Z3 + LeanAIDE)
- Stage 5: Final Proof Assembly

Author: OpenEvolve
Created: 2026-01-31
"""


import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 Integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3SolverResult, Z3TheoremResult,
        Z3Variable, Z3Constraint, Z3ConstraintType, Z3ResultStatus,
        Z3Config, Z3ProblemDetector
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 integration not available")

# Import Z3-LeanAIDE Bridge
try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge, Z3LeanAideConfig, CombinedVerificationResult,
        TranslationResult, VerificationStrategy,
        get_z3_leanaide_bridge_sync
    )
    Z3_LEANAIDE_BRIDGE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_BRIDGE_AVAILABLE = False
    # Define a fallback VerificationStrategy enum for type hints
    class VerificationStrategy:
        Z3_FIRST = "z3_first"
        LEAN_FIRST = "lean_first"
        PARALLEL = "parallel"
        ADAPTIVE = "adaptive"
    Z3LeanAideBridge = None
    Z3LeanAideConfig = None
    CombinedVerificationResult = None
    TranslationResult = None
    get_z3_leanaide_bridge_sync = None
    logger.warning("Z3-LeanAIDE bridge not available")

# Import LeanAIDE Integration
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator, LeanAideVerificationResult,
        MathematicalProblemDetector, create_standard_leanaide_config
    )
    LEANAIDE_WORKFLOW_AVAILABLE = True
except ImportError:
    LEANAIDE_WORKFLOW_AVAILABLE = False
    logger.warning("LeanAIDE workflow integration not available")

# Import OpenEvolve Workflow
try:
    from workflow_engine import WorkflowEngine
    from workflow_structures import (
        WorkflowState, SubProblem, SolutionAttempt, VerificationReport,
        DecompositionPlan
    )
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    logger.warning("OpenEvolve workflow not available")

# Import BubbleLabs Integration
try:
    from bubblelabs_integration import (
        BubbleLabsIntegration, BubbleWorkflowDefinition,
        BubbleWorkflowInstance
    )
    BUBBLELABS_AVAILABLE = True
except ImportError:
    BUBBLELABS_AVAILABLE = False
    logger.warning("BubbleLabs integration not available")

# Import CAV-NLP integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available")


# =============================================================================
# Data Classes and Enums
# =============================================================================

class ProblemCategory(Enum):
    """Categories of problems for solver selection."""
    CONSTRAINT_SOLVING = "constraint_solving"      # Use Z3
    THEOREM_PROVING = "theorem_proving"            # Use LeanAIDE
    OPTIMIZATION = "optimization"                  # Use Z3 optimization
    SMT_VERIFICATION = "smt_verification"          # Use Z3 SMT
    HYBRID = "hybrid"                              # Use combined
    STANDARD = "standard"                          # Use standard OpenEvolve


class WorkflowIntegrationStage(Enum):
    """Stages of integrated workflow."""
    PROBLEM_CLASSIFICATION = "problem_classification"
    FORMAL_TRANSLATION = "formal_translation"
    ADAPTIVE_SOLVING = "adaptive_solving"
    CROSS_VERIFICATION = "cross_verification"
    PROOF_ASSEMBLY = "proof_assembly"


@dataclass
class ProblemClassification:
    """Classification result for a problem."""
    category: ProblemCategory
    confidence: float
    recommended_solver: str
    alternative_solver: Optional[str] = None
    reasoning: str = ""
    suggested_strategy: VerificationStrategy = VerificationStrategy.ADAPTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "recommended_solver": self.recommended_solver,
            "alternative_solver": self.alternative_solver,
            "reasoning": self.reasoning,
            "suggested_strategy": self.suggested_strategy.value
        }


@dataclass
class IntegratedSolution:
    """Solution with integrated verification."""
    solution_id: str
    problem_category: ProblemCategory
    content: str
    formal_representation: Optional[str] = None
    z3_result: Optional[Z3SolverResult] = None
    lean_result: Optional[Any] = None
    combined_result: Optional[CombinedVerificationResult] = None
    confidence_score: float = 0.0
    verification_status: str = "pending"
    proof_steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_id": self.solution_id,
            "problem_category": self.problem_category.value,
            "content": self.content,
            "formal_representation": self.formal_representation,
            "confidence_score": self.confidence_score,
            "verification_status": self.verification_status,
            "proof_steps": self.proof_steps,
            "metadata": self.metadata
        }


@dataclass
class WorkflowIntegrationConfig:
    """Configuration for workflow integration."""
    # Problem classification thresholds
    z3_preference_threshold: float = 0.6
    lean_preference_threshold: float = 0.6
    hybrid_threshold: float = 0.4
    
    # Solver configuration
    z3_config: Z3Config = field(default_factory=Z3Config)
    leanaide_config: Optional[Any] = None
    
    # Strategy defaults
    default_strategy: VerificationStrategy = VerificationStrategy.ADAPTIVE
    enable_cross_validation: bool = True
    enable_proof_generation: bool = True
    
    # BubbleLabs integration
    enable_bubblelabs_visualization: bool = True
    bubblelabs_auto_register: bool = True
    
    # Execution
    max_concurrent_solvers: int = 2
    solver_timeout: float = 60.0
    
    def __post_init__(self):
        if self.leanaide_config is None and LEANAIDE_WORKFLOW_AVAILABLE:
            self.leanaide_config = create_standard_leanaide_config()


@dataclass
class IntegrationStatus:
    """Status of the integration."""
    z3_available: bool = False
    leanaide_available: bool = False
    z3_leanaide_bridge_available: bool = False
    openevolve_available: bool = False
    bubblelabs_available: bool = False
    cav_nlp_available: bool = False
    ready: bool = False
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Problem Classifier
# =============================================================================

class IntegratedProblemClassifier:
    """
    Classifies problems for appropriate solver selection.
    
    Combines Z3 problem detection with LeanAIDE mathematical detection
    to determine the best approach for a given problem.
    """
    
    def __init__(self, config: WorkflowIntegrationConfig):
        self.config = config
        self.z3_detector = Z3ProblemDetector() if Z3_AVAILABLE else None
        self.lean_detector = None
        if LEANAIDE_WORKFLOW_AVAILABLE:
            try:
                self.lean_detector = MathematicalProblemDetector(self.config.leanaide_config)
            except TypeError:
                self.lean_detector = MathematicalProblemDetector()
            except Exception as exc:
                logger.warning("Lean mathematical detector unavailable: %s", exc)
        
        # Keywords for classification
        self.constraint_keywords = [
            'solve', 'constraint', 'satisfy', 'system', 'equation',
            'inequality', 'optimize', 'minimize', 'maximize', 'allocation',
            'scheduling', 'assignment', 'satisfiability'
        ]
        
        self.theorem_keywords = [
            'prove', 'theorem', 'lemma', 'proof', 'verify', 'formal',
            'correctness', 'invariant', 'property', 'forall', 'exists'
        ]
        
        self.smt_keywords = [
            'smt', 'smt-lib', 'bitvector', 'array', 'quantifier',
            'decision procedure', 'model checking'
        ]

        self.web3_keywords = [
            "web3", "defi", "smart contract", "solidity", "evm", "onchain",
            "reentrancy", "flash loan", "oracle", "amm", "vault", "bridge",
            "slither", "foundry", "forge", "bug bounty", "exploit",
            "invariant", "symbolic execution",
        ]
    
    def classify(self, problem_statement: str) -> ProblemClassification:
        """
        Classify a problem for solver selection.
        
        Args:
            problem_statement: The problem to classify
            
        Returns:
            ProblemClassification with recommendation
        """
        text = problem_statement.lower()
        
        # Get scores from both detectors
        z3_type, z3_confidence = self._detect_z3_type(problem_statement) if self.z3_detector else ("unknown", 0.0)
        
        if self.lean_detector:
            is_math, math_confidence = self.lean_detector.is_mathematical_problem(problem_statement)
        else:
            is_math, math_confidence = False, 0.0
        
        # Check for explicit SMT-LIB
        is_smt = '(assert' in problem_statement or '(declare-fun' in problem_statement
        if is_smt:
            return ProblemClassification(
                category=ProblemCategory.SMT_VERIFICATION,
                confidence=0.95,
                recommended_solver="z3",
                reasoning="Explicit SMT-LIB format detected"
            )

        web3_confidence = self._detect_web3_audit_confidence(problem_statement)
        if web3_confidence >= 0.45:
            return ProblemClassification(
                category=ProblemCategory.HYBRID,
                confidence=web3_confidence,
                recommended_solver="combined",
                alternative_solver="z3",
                reasoning=(
                    "Web3 smart-contract audit detected; route to combined Z3/Lean "
                    "verification with invariant and exploit-witness analysis"
                ),
                suggested_strategy=VerificationStrategy.PARALLEL,
            )
        
        # Determine category based on scores
        if z3_type in ["constraint", "optimization"] and z3_confidence >= self.config.z3_preference_threshold:
            if z3_type == "optimization":
                return ProblemClassification(
                    category=ProblemCategory.OPTIMIZATION,
                    confidence=z3_confidence,
                    recommended_solver="z3",
                    alternative_solver="combined" if math_confidence > 0.5 else None,
                    reasoning=f"Z3 optimization problem detected (confidence: {z3_confidence:.2f})",
                    suggested_strategy=VerificationStrategy.Z3_FIRST
                )
            else:
                return ProblemClassification(
                    category=ProblemCategory.CONSTRAINT_SOLVING,
                    confidence=z3_confidence,
                    recommended_solver="z3",
                    alternative_solver="combined" if math_confidence > 0.5 else None,
                    reasoning=f"Constraint solving problem detected (confidence: {z3_confidence:.2f})",
                    suggested_strategy=VerificationStrategy.Z3_FIRST
                )
        
        if z3_type == "theorem" and z3_confidence >= self.config.lean_preference_threshold:
            return ProblemClassification(
                category=ProblemCategory.THEOREM_PROVING,
                confidence=z3_confidence,
                recommended_solver="leanaide",
                alternative_solver="z3" if z3_confidence > 0.5 else None,
                reasoning=f"Theorem proving problem detected (confidence: {z3_confidence:.2f})",
                suggested_strategy=VerificationStrategy.LEAN_FIRST
            )
        
        # Check for hybrid (both mathematical and constraint-based)
        if math_confidence > self.config.hybrid_threshold and z3_confidence > self.config.hybrid_threshold:
            return ProblemClassification(
                category=ProblemCategory.HYBRID,
                confidence=(math_confidence + z3_confidence) / 2,
                recommended_solver="combined",
                alternative_solver=None,
                reasoning=f"Hybrid problem - both mathematical ({math_confidence:.2f}) and constraint ({z3_confidence:.2f}) aspects",
                suggested_strategy=VerificationStrategy.PARALLEL
            )
        
        # Default to standard OpenEvolve
        if math_confidence > 0.5:
            return ProblemClassification(
                category=ProblemCategory.THEOREM_PROVING,
                confidence=math_confidence,
                recommended_solver="leanaide",
                reasoning=f"Mathematical problem detected (confidence: {math_confidence:.2f})"
            )
        
        return ProblemClassification(
            category=ProblemCategory.STANDARD,
            confidence=1.0 - max(z3_confidence, math_confidence),
            recommended_solver="standard",
            reasoning="Standard problem - no special solver needed"
        )
    
    def _detect_z3_type(self, problem: str) -> Tuple[str, float]:
        """Detect problem type using Z3 detector."""
        if not self.z3_detector:
            return "unknown", 0.0
        return self.z3_detector.detect_problem_type(problem)

    def _detect_web3_audit_confidence(self, problem: str) -> float:
        """Score whether a prompt is a Web3 smart-contract audit workflow."""
        text = (problem or "").lower()
        if not text:
            return 0.0
        hits = sum(1 for kw in self.web3_keywords if kw in text)
        if "smart contract" in text and ("exploit" in text or "audit" in text):
            hits += 2
        return min(1.0, hits / 6.0)


# =============================================================================
# Z3-LeanAIDE-OpenEvolve Integration
# =============================================================================

class Z3LeanAideOpenEvolveIntegration:
    """
    Main integration class connecting Z3, LeanAIDE, OpenEvolve, and BubbleLabs.
    
    This class orchestrates the entire workflow:
    1. Problem classification
    2. Adaptive solver selection
    3. Solution generation
    4. Verification (potentially cross-validated)
    5. Proof assembly
    6. Visualization in BubbleLabs
    """
    
    def __init__(self, config: Optional[WorkflowIntegrationConfig] = None):
        self.config = config or WorkflowIntegrationConfig()
        self.classifier = IntegratedProblemClassifier(self.config)
        
        # Initialize solvers
        self.z3_solver = None
        self.z3_prover = None
        self.z3_bridge = None
        self.lean_integrator = None
        self.bubblelabs = None
        
        # Initialize CAV-NLP components
        self.use_cav_nlp = config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE if hasattr(config, 'get') else CAV_NLP_AVAILABLE
        self.enhanced_solver = None
        self.math_service = None
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP enhancement enabled for Z3LeanAideOpenEvolveIntegration")
        
        self._initialize_components()
        
        # Thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_solvers)
        
        # Status tracking
        self._integration_status = IntegrationStatus()
        self._update_status()
        
        # Update status with CAV-NLP availability
        self._integration_status.cav_nlp_available = CAV_NLP_AVAILABLE
    
    def _initialize_components(self):
        """Initialize all integration components."""
        # Initialize Z3
        if Z3_AVAILABLE:
            from z3prover_integration import get_z3_solver_engine, get_z3_theorem_prover
            self.z3_solver = get_z3_solver_engine(self.config.z3_config)
            self.z3_prover = get_z3_theorem_prover(self.config.z3_config)
            logger.info("Z3 components initialized")
        
        # Initialize Z3-LeanAIDE bridge
        if Z3_LEANAIDE_BRIDGE_AVAILABLE:
            bridge_config = Z3LeanAideConfig(
                z3_timeout=self.config.solver_timeout,
                default_strategy=self.config.default_strategy,
                enable_cross_validation=self.config.enable_cross_validation
            )
            self.z3_bridge = get_z3_leanaide_bridge_sync(bridge_config)
            logger.info("Z3-LeanAIDE bridge initialized")
        
        # Initialize LeanAIDE
        if LEANAIDE_WORKFLOW_AVAILABLE:
            self.lean_integrator = LeanAideWorkflowIntegrator(self.config.leanaide_config)
            logger.info("LeanAIDE integrator initialized")
        
        # Initialize BubbleLabs
        if BUBBLELABS_AVAILABLE and self.config.enable_bubblelabs_visualization:
            self.bubblelabs = BubbleLabsIntegration()
            logger.info("BubbleLabs integration initialized")
    
    def _update_status(self):
        """Update integration status."""
        self._integration_status = IntegrationStatus(
            z3_available=Z3_AVAILABLE and self.z3_solver is not None,
            leanaide_available=LEANAIDE_WORKFLOW_AVAILABLE and self.lean_integrator is not None,
            z3_leanaide_bridge_available=Z3_LEANAIDE_BRIDGE_AVAILABLE and self.z3_bridge is not None,
            openevolve_available=OPENEVOLVE_AVAILABLE,
            bubblelabs_available=BUBBLELABS_AVAILABLE and self.bubblelabs is not None,
            ready=self._check_ready(),
            message=self._get_status_message()
        )
    
    def _check_ready(self) -> bool:
        """Check if integration is ready to use."""
        return (
            (Z3_AVAILABLE or LEANAIDE_WORKFLOW_AVAILABLE) and
            OPENEVOLVE_AVAILABLE
        )
    
    def _get_status_message(self) -> str:
        """Get human-readable status message."""
        if not self._check_ready():
            return "Integration not ready - missing required components"
        
        available = []
        if Z3_AVAILABLE:
            available.append("Z3")
        if LEANAIDE_WORKFLOW_AVAILABLE:
            available.append("LeanAIDE")
        if BUBBLELABS_AVAILABLE:
            available.append("BubbleLabs")
        
        return f"Integration ready with: {', '.join(available)}"
    
    def get_status(self) -> IntegrationStatus:
        """Get current integration status."""
        self._update_status()
        return self._integration_status
    
    # =========================================================================
    # Core Workflow Integration Methods
    # =========================================================================
    
    async def process_problem(
        self,
        problem_statement: str,
        workflow_id: Optional[str] = None,
        entanglement_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a problem through the integrated workflow.
        
        Args:
            problem_statement: The problem to solve
            workflow_id: Optional workflow ID
            
        Returns:
            Complete workflow result
        """
        start_time = time.time()
        workflow_id = workflow_id or f"z3_lean_oe_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting integrated workflow {workflow_id}")
        
        try:
            # Stage 1: Problem Classification
            classification = self.classifier.classify(problem_statement)
            logger.info(f"Problem classified as: {classification.category.value}")
            
            # Stage 2: Create workflow in BubbleLabs if available
            bubble_instance = None
            if self.bubblelabs:
                bubble_instance = self._create_bubblelabs_workflow(
                    workflow_id, problem_statement, classification, entanglement_context=entanglement_context
                )
            
            # Stage 3: Solve based on classification
            solution = await self._solve_problem(
                problem_statement, classification, workflow_id, entanglement_context=entanglement_context
            )
            
            # Stage 4: Verify
            verification = await self._verify_solution(
                problem_statement, solution, classification
            )
            
            # Stage 5: Assemble result
            result = {
                "workflow_id": workflow_id,
                "status": "completed",
                "classification": classification.to_dict(),
                "solution": solution.to_dict(),
                "verification": verification.to_dict() if hasattr(verification, 'to_dict') else verification,
                "execution_time": time.time() - start_time,
                "entanglement_context": entanglement_context or {}
            }
            
            # Update BubbleLabs
            if bubble_instance:
                self._update_bubblelabs_completion(bubble_instance.id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    async def _solve_problem(
        self,
        problem_statement: str,
        classification: ProblemClassification,
        workflow_id: str,
        entanglement_context: Optional[Dict[str, Any]] = None
    ) -> IntegratedSolution:
        """Solve problem based on classification."""
        solution_id = f"sol_{workflow_id}"

        if classification.category == ProblemCategory.CONSTRAINT_SOLVING:
            solution = await self._solve_constraint(problem_statement, solution_id)
        elif classification.category == ProblemCategory.OPTIMIZATION:
            solution = await self._solve_optimization(problem_statement, solution_id)
        elif classification.category == ProblemCategory.THEOREM_PROVING:
            solution = await self._solve_theorem(problem_statement, solution_id)
        elif classification.category == ProblemCategory.SMT_VERIFICATION:
            solution = await self._solve_smt(problem_statement, solution_id, entanglement_context=entanglement_context)
        elif classification.category == ProblemCategory.HYBRID:
            solution = await self._solve_hybrid(problem_statement, solution_id, entanglement_context=entanglement_context)
        else:
            # Standard problem - use OpenEvolve
            solution = await self._solve_standard(problem_statement, solution_id)

        if entanglement_context:
            solution.metadata.setdefault("entanglement_context", entanglement_context)

        return solution
    
    async def _solve_constraint(
        self,
        problem: str,
        solution_id: str
    ) -> IntegratedSolution:
        """Solve constraint problem with Z3."""
        if not self.z3_solver:
            return IntegratedSolution(
                solution_id=solution_id,
                problem_category=ProblemCategory.CONSTRAINT_SOLVING,
                content="Z3 not available",
                verification_status="failed"
            )
        
        # Extract constraints from problem (simplified)
        # In practice, this would use NLP or structured parsing
        variables, constraints = self._extract_constraints_from_text(problem)
        
        # Solve
        result = self.z3_solver.solve_constraints(variables, constraints)
        
        content = "Solution found" if result.is_sat() else "No solution exists"
        if result.model:
            content += f": {result.model.assignments}"
        
        return IntegratedSolution(
            solution_id=solution_id,
            problem_category=ProblemCategory.CONSTRAINT_SOLVING,
            content=content,
            z3_result=result,
            confidence_score=0.9 if result.is_sat() else 0.5,
            verification_status="verified" if result.is_sat() else "unsatisfiable"
        )
    
    async def _solve_optimization(
        self,
        problem: str,
        solution_id: str
    ) -> IntegratedSolution:
        """Solve optimization problem with Z3."""
        # Similar to constraint solving but with objective
        solution = await self._solve_constraint(problem, solution_id)
        solution.problem_category = ProblemCategory.OPTIMIZATION
        return solution
    
    async def _solve_theorem(
        self,
        problem: str,
        solution_id: str
    ) -> IntegratedSolution:
        """Solve theorem with LeanAIDE."""
        if not self.lean_integrator:
            return IntegratedSolution(
                solution_id=solution_id,
                problem_category=ProblemCategory.THEOREM_PROVING,
                content="LeanAIDE not available",
                verification_status="failed"
            )
        
        # Initialize LeanAIDE
        if not self.lean_integrator.client:
            initialized = await self.lean_integrator.initialize()
            if not initialized:
                return IntegratedSolution(
                    solution_id=solution_id,
                    problem_category=ProblemCategory.THEOREM_PROVING,
                    content="Failed to initialize LeanAIDE",
                    verification_status="failed"
                )
        
        # Verify with LeanAIDE
        result = await self.lean_integrator.verify_sub_problem_solution(
            sub_problem_id=solution_id,
            problem_statement=problem,
            solution_content=""
        )
        
        success = result.success if hasattr(result, 'success') else result.get('success', False)
        
        return IntegratedSolution(
            solution_id=solution_id,
            problem_category=ProblemCategory.THEOREM_PROVING,
            content="Theorem proven" if success else "Proof failed",
            lean_result=result,
            confidence_score=result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
            verification_status="verified" if success else "failed"
        )
    
    async def _solve_smt(
        self,
        problem: str,
        solution_id: str,
        entanglement_context: Optional[Dict[str, Any]] = None
    ) -> IntegratedSolution:
        """Solve SMT-LIB problem."""
        if not self.z3_solver:
            return IntegratedSolution(
                solution_id=solution_id,
                problem_category=ProblemCategory.SMT_VERIFICATION,
                content="Z3 not available",
                verification_status="failed"
            )
        
        problem = self._apply_entanglement_to_smt(problem, entanglement_context)
        result = self.z3_solver.solve_smtlib(problem)
        
        content = f"SMT result: {result.status.value}"
        if result.model:
            content += f" with model {result.model.assignments}"
        
        return IntegratedSolution(
            solution_id=solution_id,
            problem_category=ProblemCategory.SMT_VERIFICATION,
            content=content,
            z3_result=result,
            confidence_score=0.95 if result.is_sat() else 0.5,
            verification_status="verified" if result.is_sat() else "unsatisfiable"
        )
    
    async def _solve_hybrid(
        self,
        problem: str,
        solution_id: str,
        entanglement_context: Optional[Dict[str, Any]] = None
    ) -> IntegratedSolution:
        """Solve hybrid problem using combined approach."""
        if not self.z3_bridge:
            return IntegratedSolution(
                solution_id=solution_id,
                problem_category=ProblemCategory.HYBRID,
                content="Z3-LeanAIDE bridge not available",
                verification_status="failed"
            )
        
        # Use combined verification
        result = await self.z3_bridge.verify_with_both(
            problem,
            VerificationStrategy.PARALLEL,
            entanglement_context=entanglement_context
        )
        
        return IntegratedSolution(
            solution_id=solution_id,
            problem_category=ProblemCategory.HYBRID,
            content=result.recommendation,
            combined_result=result,
            confidence_score=result.confidence_score,
            verification_status="verified" if result.success else "failed"
        )
    
    async def _solve_standard(
        self,
        problem: str,
        solution_id: str
    ) -> IntegratedSolution:
        """Solve standard problem with OpenEvolve."""
        # This would integrate with standard OpenEvolve workflow
        return IntegratedSolution(
            solution_id=solution_id,
            problem_category=ProblemCategory.STANDARD,
            content="Standard OpenEvolve solution",
            verification_status="pending"
        )
    
    async def _verify_solution(
        self,
        problem: str,
        solution: IntegratedSolution,
        classification: ProblemClassification
    ) -> Optional[CombinedVerificationResult]:
        """Verify solution with appropriate method."""
        if not self.config.enable_cross_validation:
            return None
        
        if self.z3_bridge and classification.category in [ProblemCategory.HYBRID, ProblemCategory.THEOREM_PROVING]:
            entanglement_context = None
            if hasattr(solution, "metadata") and isinstance(solution.metadata, dict):
                entanglement_context = solution.metadata.get("entanglement_context")
            return await self.z3_bridge.verify_with_both(
                problem,
                entanglement_context=entanglement_context
            )
        
        return None
    
    def _extract_constraints_from_text(
        self,
        text: str
    ) -> Tuple[List[Z3Variable], List[Z3Constraint]]:
        """Extract Z3 constraints from natural language text."""
        variables: List[Z3Variable] = []
        constraints: List[Z3Constraint] = []

        if not text or not text.strip():
            return variables, constraints

        try:
            import workflow_engine
            from llm_utils import _compose_messages, _request_openai_compatible_chat
        except Exception as exc:
            logger.warning("Constraint extraction unavailable (LLM import failure): %s", exc)
            return variables, constraints

        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or os.getenv("OPENAI_API_TOKEN")
        )
        if not api_key:
            logger.warning("Constraint extraction skipped: OPENAI_API_KEY not set")
            return variables, constraints

        base_url = (
            os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = (
            os.getenv("OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL_ID")
            or "gpt-4o-mini"
        )

        system_prompt = (
            "You extract SMT constraints from natural language. "
            "Return ONLY a JSON object that matches the required schema."
        )
        user_prompt = (
            "Extract variables and constraints from the text below.\n\n"
            "Return JSON with this exact schema:\n"
            "{\n"
            "  \"variables\": [\n"
            "    {\"name\": \"x\", \"type\": \"integer|real|boolean\", "
            "\"bounds\": [\"min_or_null\", \"max_or_null\"], \"bit_width\": 32}\n"
            "  ],\n"
            "  \"constraints\": [\"SMT-LIB boolean expressions without (assert)\"]\n"
            "}\n\n"
            "Rules:\n"
            "- Use lowercase type strings: integer, real, boolean.\n"
            "- If bounds are unknown, use nulls.\n"
            "- Constraints must be SMT-LIB boolean expressions (no (assert)).\n"
            "- If nothing is found, return empty arrays.\n\n"
            f"Text:\n{text}"
        )

        messages = _compose_messages(system_prompt, user_prompt)
        try:
            response = _request_openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                temperature=0.0,
                top_p=1.0,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
        except Exception as exc:
            logger.warning("Constraint extraction failed: %s", exc)
            return variables, constraints

        if not response:
            return variables, constraints

        raw = response.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Constraint extraction JSON parse failed: %s", exc)
            return variables, constraints

        if not isinstance(parsed, dict):
            return variables, constraints

        type_map = {
            "integer": Z3ConstraintType.INTEGER,
            "int": Z3ConstraintType.INTEGER,
            "real": Z3ConstraintType.REAL,
            "float": Z3ConstraintType.REAL,
            "boolean": Z3ConstraintType.BOOLEAN,
            "bool": Z3ConstraintType.BOOLEAN
        }

        seen_names = set()
        for entry in parsed.get("variables") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name or name in seen_names:
                continue
            var_type_str = str(entry.get("type", "integer")).strip().lower()
            var_type = type_map.get(var_type_str, Z3ConstraintType.INTEGER)
            bounds = entry.get("bounds")
            bit_width = entry.get("bit_width")
            if isinstance(bounds, list) and len(bounds) == 2:
                min_val = None if bounds[0] in (None, "null") else bounds[0]
                max_val = None if bounds[1] in (None, "null") else bounds[1]
                bounds_tuple = (min_val, max_val)
            else:
                bounds_tuple = None
            if isinstance(bit_width, (int, float)):
                bit_width = int(bit_width)
            else:
                bit_width = None

            variables.append(Z3Variable(
                name=name,
                var_type=var_type,
                bounds=bounds_tuple,
                bit_width=bit_width
            ))
            seen_names.add(name)

        for constraint in parsed.get("constraints") or []:
            if not isinstance(constraint, str):
                continue
            text_constraint = constraint.strip()
            if not text_constraint:
                continue
            constraints.append(Z3Constraint(
                expression=text_constraint,
                constraint_type=Z3ConstraintType.BOOLEAN
            ))

        return variables, constraints

    @staticmethod
    def _merge_smtlib_constraints(smtlib: str, constraints: List[str]) -> str:
        """Inject constraints into SMT-LIB text using Z3 parsing."""
        if not constraints:
            return smtlib

        smtlib = smtlib or ""
        cleaned = []
        for constraint in constraints:
            if constraint is None:
                continue
            text = str(constraint).strip()
            if text:
                cleaned.append(text)
        if not cleaned:
            return smtlib

        def _fallback_merge() -> str:
            assert_lines = []
            for text in cleaned:
                if text.startswith("(assert"):
                    assert_lines.append(text)
                else:
                    assert_lines.append(f"(assert {text})")
            if not assert_lines:
                return smtlib
            insertion = "\n".join(assert_lines) + "\n"
            lower = smtlib.lower()
            idx = lower.rfind("(check-sat")
            if idx != -1:
                return smtlib[:idx] + insertion + smtlib[idx:]
            if smtlib and not smtlib.endswith("\n"):
                return smtlib + "\n" + insertion
            return smtlib + insertion

        try:
            from z3 import Solver, parse_smt2_string, Z3Exception
            from z3.z3util import get_vars
        except Exception as exc:
            logger.warning("Z3 not available for SMT merge: %s", exc)
            return _fallback_merge()

        try:
            solver = Solver()
            if smtlib.strip():
                solver.from_string(smtlib)

            decls: Dict[str, Any] = {}
            try:
                for assertion in solver.assertions():
                    for var in get_vars(assertion):
                        decls.setdefault(var.decl().name(), var)
            except Exception:
                decls = {}

            for text in cleaned:
                if "(declare" in text or "(define" in text or "(set-logic" in text:
                    solver.from_string(text)
                    continue

                candidate = text
                if not candidate.startswith("(assert"):
                    candidate = f"(assert {candidate})"

                try:
                    parsed = parse_smt2_string(candidate, decls=decls)
                    if parsed:
                        solver.add(*parsed)
                        for expr in parsed:
                            for var in get_vars(expr):
                                decls.setdefault(var.decl().name(), var)
                except Z3Exception:
                    try:
                        parsed = parse_smt2_string(text, decls=decls)
                        if parsed:
                            solver.add(*parsed)
                            for expr in parsed:
                                for var in get_vars(expr):
                                    decls.setdefault(var.decl().name(), var)
                        else:
                            solver.from_string(text)
                    except Z3Exception as exc:
                        logger.warning("SMT merge failed for constraint '%s': %s", text, exc)
                        return _fallback_merge()

            return solver.to_smt2()
        except Exception as exc:
            logger.warning("Failed to merge SMT-LIB via Z3: %s", exc)
            return _fallback_merge()

    @staticmethod
    def _resolve_entangled_constraints(entanglement_context: Optional[Dict[str, Any]]) -> List[str]:
        if not entanglement_context:
            return []

        entangled_constraints = entanglement_context.get("entangled_constraints")
        if entangled_constraints:
            return list(entangled_constraints)

        entanglement_constraints = entanglement_context.get("entanglement_constraints", {}) or {}
        entangled_with = entanglement_context.get("entangled_with", []) or []

        constraints: List[str] = []
        if isinstance(entanglement_constraints, dict):
            for ent_id in entangled_with:
                constraints.extend(entanglement_constraints.get(ent_id, []) or [])

        return constraints

    def _apply_entanglement_to_smt(
        self,
        problem: str,
        entanglement_context: Optional[Dict[str, Any]]
    ) -> str:
        problem_text = self._normalize_problem_input(problem)
        constraints = self._resolve_entangled_constraints(entanglement_context)
        if not constraints:
            return problem_text
        return self._merge_smtlib_constraints(problem_text, constraints)

    @staticmethod
    def _normalize_problem_input(problem: Any) -> str:
        """Normalize problem input to a string."""
        if isinstance(problem, str):
            return problem
        if isinstance(problem, bytes):
            try:
                return problem.decode("utf-8")
            except UnicodeDecodeError:
                return problem.decode("utf-8", errors="replace")
        if isinstance(problem, dict):
            for key in ("smtlib", "problem", "statement", "content"):
                value = problem.get(key)
                if isinstance(value, str):
                    return value
        return str(problem)
    
    # =========================================================================
    # BubbleLabs Integration
    # =========================================================================
    
    def _create_bubblelabs_workflow(
        self,
        workflow_id: str,
        problem: str,
        classification: ProblemClassification,
        entanglement_context: Optional[Dict[str, Any]] = None
    ) -> Optional[BubbleWorkflowInstance]:
        """Create workflow visualization in BubbleLabs."""
        if not self.bubblelabs:
            return None
        
        try:
            # Create workflow definition
            is_web3_workflow = (
                self.classifier._detect_web3_audit_confidence(problem) >= 0.45
            )
            web3_payload: Dict[str, Any] = {}
            if is_web3_workflow:
                web3_payload = {"enabled": True}
                if isinstance(entanglement_context, dict):
                    web3_ctx = entanglement_context.get("web3", {})
                    if isinstance(web3_ctx, dict):
                        for key in (
                            "project_path",
                            "run_fuzzing",
                            "slither_timeout_seconds",
                            "forge_timeout_seconds",
                        ):
                            if key in web3_ctx:
                                web3_payload[key] = web3_ctx[key]

            team_config = {
                "classifier_team": "z3_lean_classifier",
                "solver_team": classification.recommended_solver,
                "verifier_team": "cross_validator"
            }
            
            gauntlet_config = {
                "problem_gauntlet": "z3_lean_gauntlet",
                "verification_gauntlet": "formal_verification"
            }
            
            definition = self.bubblelabs.create_workflow_definition_from_openevolve(
                problem_statement=problem,
                team_config=team_config,
                gauntlet_config=gauntlet_config,
                workflow_type="web3" if is_web3_workflow else "sovereign_decomposition",
                web3_config=web3_payload if is_web3_workflow else None,
            )
            
            # Create instance
            instance = BubbleWorkflowInstance(
                id=workflow_id,
                definition_id=definition.id,
                status="running",
                created_at=time.time(),
                updated_at=time.time(),
                progress=0.0,
                data={
                    "classification": classification.to_dict(),
                    "current_stage": "problem_classification",
                    "workflow_type": "web3" if is_web3_workflow else "sovereign_decomposition",
                    "web3": web3_payload if is_web3_workflow else {},
                    "entanglement_context": entanglement_context or {}
                }
            )
            
            self.bubblelabs.workflow_instances[workflow_id] = instance
            
            logger.info(f"Created BubbleLabs workflow: {workflow_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create BubbleLabs workflow: {e}")
            return None
    
    def _update_bubblelabs_completion(
        self,
        instance_id: str,
        result: Dict[str, Any]
    ):
        """Update BubbleLabs with completion status."""
        if not self.bubblelabs:
            return
        
        try:
            instance = self.bubblelabs.workflow_instances.get(instance_id)
            if instance:
                instance.status = "completed"
                instance.progress = 1.0
                instance.updated_at = time.time()
                instance.data["result"] = result
                instance.data["current_stage"] = "completed"
        except Exception as e:
            logger.error(f"Failed to update BubbleLabs: {e}")
    
    # =========================================================================
    # OpenEvolve Workflow Stage Integration
    # =========================================================================
    
    async def enhanced_decompose_problem(
        self,
        workflow_state: Any,
        problem_statement: str
    ) -> Any:
        """
        Enhanced decomposition with formal awareness.
        
        Classifies problem and sets up appropriate solver for sub-problems.
        """
        # Classify problem
        classification = self.classifier.classify(problem_statement)
        
        # Store classification in workflow state
        if hasattr(workflow_state, 'metadata'):
            workflow_state.metadata['problem_classification'] = classification.to_dict()
        
        # Continue with standard decomposition
        # (Would integrate with actual decomposition engine)
        logger.info(f"Enhanced decomposition with classification: {classification.category.value}")
        
        return workflow_state
    
    async def enhanced_generate_solution(
        self,
        workflow_state: Any,
        subproblem: Any
    ) -> IntegratedSolution:
        """
        Enhanced solution generation with adaptive solver selection.
        """
        problem_text = subproblem.description if hasattr(subproblem, 'description') else str(subproblem)
        
        classification = self.classifier.classify(problem_text)
        solution_id = f"sol_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        
        return await self._solve_problem(problem_text, classification, solution_id)
    
    async def enhanced_verify_solution(
        self,
        workflow_state: Any,
        solution: IntegratedSolution,
        subproblem: Any
    ) -> Dict[str, Any]:
        """
        Enhanced verification with cross-validation.
        """
        problem_text = subproblem.description if hasattr(subproblem, 'description') else str(subproblem)
        
        verification = await self._verify_solution(
            problem_text, solution, 
            solution.problem_category
        )
        
        return verification.to_dict() if hasattr(verification, 'to_dict') else verification
    
    async def hybrid_verify(self, constraint) -> "VerificationResult":
        """Verify using hybrid Z3 + Lean via CAV-NLP.
        
        This method leverages the CAV-NLP unified math service to perform
        hybrid verification, combining Z3's constraint solving with Lean's
        theorem proving capabilities.
        
        Args:
            constraint: The constraint to verify (can be Z3 expression or string)
            
        Returns:
            VerificationResult from the hybrid verification
        """
        if not self.use_cav_nlp or not self.math_service:
            logger.warning("CAV-NLP not available for hybrid verification")
            # Return a basic result indicating CAV-NLP is unavailable
            return {"success": False, "error": "CAV-NLP not available"}
        
        try:
            return await self.math_service.verify(constraint)
        except Exception as e:
            logger.error(f"Hybrid verification failed: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# Global Instance
# =============================================================================

_z3_lean_oe_integration: Optional[Z3LeanAideOpenEvolveIntegration] = None
_integration_lock = threading.Lock()


def get_z3_leanaide_openevolve_integration(
    config: Optional[WorkflowIntegrationConfig] = None
) -> Z3LeanAideOpenEvolveIntegration:
    """Get global integration instance."""
    global _z3_lean_oe_integration
    if _z3_lean_oe_integration is None:
        with _integration_lock:
            if _z3_lean_oe_integration is None:
                _z3_lean_oe_integration = Z3LeanAideOpenEvolveIntegration(config)
    return _z3_lean_oe_integration


# =============================================================================
# Convenience Functions
# =============================================================================

async def solve_with_z3_leanaide(
    problem: str,
    workflow_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to solve a problem with integrated system.
    
    Args:
        problem: Problem statement
        workflow_id: Optional workflow ID
        
    Returns:
        Complete result with classification, solution, and verification
    """
    integration = get_z3_leanaide_openevolve_integration()
    return await integration.process_problem(problem, workflow_id)


def get_integration_status() -> Dict[str, Any]:
    """Get integration status."""
    integration = get_z3_leanaide_openevolve_integration()
    status = integration.get_status()
    return status.to_dict()


# =============================================================================
# Example Usage
# =============================================================================

async def example_constraint_problem():
    """Example: Solve a constraint problem."""
    problem = """
    Find values for x and y where:
    - x is an integer between 1 and 10
    - y is an integer
    - y equals x plus 5
    """
    
    result = await solve_with_z3_leanaide(problem)
    
    print("=" * 60)
    print("Constraint Problem Example")
    print("=" * 60)
    print(f"Classification: {result['classification']['category']}")
    print(f"Solution: {result['solution']['content']}")
    print(f"Status: {result['status']}")
    
    return result


async def example_theorem_problem():
    """Example: Solve a theorem problem."""
    problem = """
    Prove that for all integers x, if x > 0 then x + 1 > 0.
    """
    
    result = await solve_with_z3_leanaide(problem)
    
    print("\n" + "=" * 60)
    print("Theorem Problem Example")
    print("=" * 60)
    print(f"Classification: {result['classification']['category']}")
    print(f"Solution: {result['solution']['content']}")
    print(f"Status: {result['status']}")
    
    return result


async def example_smt_problem():
    """Example: Solve an SMT-LIB problem."""
    problem = """
    (set-logic LIA)
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (> x 0))
    (assert (< x 100))
    (assert (= y (* x 2)))
    (check-sat)
    """
    
    result = await solve_with_z3_leanaide(problem)
    
    print("\n" + "=" * 60)
    print("SMT-LIB Problem Example")
    print("=" * 60)
    print(f"Classification: {result['classification']['category']}")
    print(f"Solution: {result['solution']['content']}")
    print(f"Status: {result['status']}")
    
    return result


async def main():
    """Run all examples."""
    print("Z3-LeanAIDE-OpenEvolve Integration")
    print("=" * 60)
    
    # Show status
    status = get_integration_status()
    print(f"\nIntegration Status:")
    print(f"  Ready: {status['ready']}")
    print(f"  Message: {status['message']}")
    print(f"  Z3: {status['z3_available']}")
    print(f"  LeanAIDE: {status['leanaide_available']}")
    print(f"  OpenEvolve: {status['openevolve_available']}")
    print(f"  BubbleLabs: {status['bubblelabs_available']}")
    
    # Run examples
    if status['ready']:
        await example_constraint_problem()
        await example_theorem_problem()
        await example_smt_problem()
    else:
        print("\nIntegration not ready - skipping examples")


if __name__ == "__main__":
    asyncio.run(main())
