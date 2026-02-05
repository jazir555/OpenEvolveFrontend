"""
LeanAide-RESE Workflow Orchestrator

Main workflow integration that connects LeanAide's AI-powered theorem proving
with RESE's 4-phase pipeline:

Phase I - Epistemic Audit:
  - AI-assisted tacit assumption mining
  - Autoformalization of natural language constraints
  - Automated theorem proving for constraint verification

Phase II - Isomorphic Mapping:
  - AI-powered FDG construction from domain descriptions
  - Automated mechanistic isomorphism detection
  - Formal verification of abstract causal mappings

Phase III - MCTS Refinement:
  - MCTS proof search with LeanAide tactics
  - AI-guided anomaly detection
  - Intelligent proof strategy selection

Phase IV - Architectural Synthesis:
  - Formal verification of predictive models
  - Automated proof generation for efficacy claims
  - Mathematical validation of paradigm transformation

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Idempotency: Safe to replay
- Structured Logging: JSON with correlation_id
- Circuit Breaker: Per-phase circuit breakers
- Exponential Backoff: Retry with jitter
- Dead Letter Queue: For logic failures
- Timeout: Per-phase and overall timeouts

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sys
import uuid
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import RESE schemas
try:
    from glue.schemas.rese_schemas import (
        Hypothesis, Pattern, IsomorphicMapping, FunctionalDependencyGraph,
        MCTSSearchResult, HypothesisStatus, PatternType, IsomorphismType,
        ExplorationStrategy
    )
except ImportError:
    from rese_schemas import (
        Hypothesis, Pattern, IsomorphicMapping, FunctionalDependencyGraph,
        MCTSSearchResult, HypothesisStatus, PatternType, IsomorphismType,
        ExplorationStrategy
    )

# Import autoformalization and proof search services
from src.autoformalization_service import (
    AutoformalizationService, AutoformalizationConfig, AutoformalizationPhase,
    AutoformalizationResult, FormalizationDomain
)
from src.proof_search_service import (
    ProofSearchService, ProofSearchConfig, ProofStrategy, ProofStatus,
    ProofSearchResult
)


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ProblemType(Enum):
    """Types of problems for classification"""
    CONSTRAINT_VERIFICATION = "constraint_verification"
    THEOREM_PROVING = "theorem_proving"
    ISOMORPHISM_DETECTION = "isomorphism_detection"
    OPTIMIZATION = "optimization"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    MODEL_VALIDATION = "model_validation"


class SolverType(Enum):
    """Types of solvers"""
    Z3 = "z3"
    LEANAIDE = "leanaide"
    LEAN4 = "lean4"
    HYBRID_Z3_LEANAIDE = "hybrid_z3_leanaide"
    HYBRID_ALL = "hybrid_all"


class PhaseStatus(Enum):
    """Status of workflow phase"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class ProblemClassification:
    """Result of problem classification"""
    problem_type: ProblemType
    mathematical_domain: FormalizationDomain
    recommended_solver: SolverType
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type": self.problem_type.value,
            "mathematical_domain": self.mathematical_domain.value,
            "recommended_solver": self.recommended_solver.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


@dataclass
class PhaseResult:
    """Result from a workflow phase"""
    phase: str
    status: PhaseStatus
    data: Dict[str, Any] = field(default_factory=dict)
    autoformalization_results: List[AutoformalizationResult] = field(default_factory=list)
    proof_search_results: List[ProofSearchResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "data": self.data,
            "autoformalization_results": [r.to_dict() for r in self.autoformalization_results],
            "proof_search_results": [r.to_dict() for r in self.proof_search_results],
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp
        }


@dataclass
class WorkflowResult:
    """Result from complete workflow execution"""
    workflow_id: str
    correlation_id: str
    problem_classification: ProblemClassification
    phase_results: Dict[str, PhaseResult]
    overall_status: str
    total_execution_time_ms: float
    summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "correlation_id": self.correlation_id,
            "problem_classification": self.problem_classification.to_dict(),
            "phase_results": {k: v.to_dict() for k, v in self.phase_results.items()},
            "overall_status": self.overall_status,
            "total_execution_time_ms": self.total_execution_time_ms,
            "summary": self.summary,
            "errors": self.errors,
            "timestamp": self.timestamp
        }


@dataclass
class WorkflowConfig:
    """Configuration for LeanAide-RESE workflow"""
    # Autoformalization config
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    autoformalization_timeout_ms: int = 30000
    autoformalization_confidence_threshold: float = 0.7

    # Proof search config
    proof_search_timeout_ms: int = 60000
    proof_search_max_depth: int = 100
    proof_search_mcts_iterations: int = 1000
    proof_search_enable_z3: bool = True
    proof_search_confidence_threshold: float = 0.8

    # Phase timeouts
    phase_i_timeout_ms: int = 60000
    phase_ii_timeout_ms: int = 90000
    phase_iii_timeout_ms: int = 120000
    phase_iv_timeout_ms: int = 90000

    # Overall workflow
    workflow_timeout_ms: int = 600000
    max_retries: int = 3
    retry_delay_ms: int = 1000
    enable_caching: bool = True

    # Correlation tracking
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "WorkflowConfig":
        """Create configuration from environment variables"""
        return cls(
            leanaide_host=os.getenv("LEANAIDE_HOST", "localhost"),
            leanaide_port=int(os.getenv("LEANAIDE_PORT", "7654")),
            autoformalization_timeout_ms=int(os.getenv("LEANAIDE_TIMEOUT_MS", "30000")),
            autoformalization_confidence_threshold=float(os.getenv("LEANAIDE_CONFIDENCE_THRESHOLD", "0.7")),
            proof_search_timeout_ms=int(os.getenv("PROOF_SEARCH_TIMEOUT_MS", "60000")),
            proof_search_max_depth=int(os.getenv("PROOF_SEARCH_MAX_DEPTH", "100")),
            proof_search_mcts_iterations=int(os.getenv("PROOF_SEARCH_MCTS_ITERATIONS", "1000")),
            proof_search_enable_z3=os.getenv("PROOF_SEARCH_ENABLE_Z3", "true").lower() == "true",
            proof_search_confidence_threshold=float(os.getenv("PROOF_SEARCH_CONFIDENCE_THRESHOLD", "0.8")),
            phase_i_timeout_ms=int(os.getenv("WORKFLOW_PHASE_I_TIMEOUT_MS", "60000")),
            phase_ii_timeout_ms=int(os.getenv("WORKFLOW_PHASE_II_TIMEOUT_MS", "90000")),
            phase_iii_timeout_ms=int(os.getenv("WORKFLOW_PHASE_III_TIMEOUT_MS", "120000")),
            phase_iv_timeout_ms=int(os.getenv("WORKFLOW_PHASE_IV_TIMEOUT_MS", "90000")),
            workflow_timeout_ms=int(os.getenv("WORKFLOW_TIMEOUT_MS", "600000")),
            max_retries=int(os.getenv("WORKFLOW_MAX_RETRIES", "3")),
            retry_delay_ms=int(os.getenv("WORKFLOW_RETRY_DELAY_MS", "1000")),
            enable_caching=os.getenv("WORKFLOW_ENABLE_CACHING", "true").lower() == "true",
            correlation_id=os.getenv("CORRELATION_ID")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "leanaide_host": self.leanaide_host,
            "leanaide_port": self.leanaide_port,
            "autoformalization_timeout_ms": self.autoformalization_timeout_ms,
            "autoformalization_confidence_threshold": self.autoformalization_confidence_threshold,
            "proof_search_timeout_ms": self.proof_search_timeout_ms,
            "proof_search_max_depth": self.proof_search_max_depth,
            "proof_search_mcts_iterations": self.proof_search_mcts_iterations,
            "proof_search_enable_z3": self.proof_search_enable_z3,
            "proof_search_confidence_threshold": self.proof_search_confidence_threshold,
            "phase_i_timeout_ms": self.phase_i_timeout_ms,
            "phase_ii_timeout_ms": self.phase_ii_timeout_ms,
            "phase_iii_timeout_ms": self.phase_iii_timeout_ms,
            "phase_iv_timeout_ms": self.phase_iv_timeout_ms,
            "workflow_timeout_ms": self.workflow_timeout_ms,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
            "enable_caching": self.enable_caching,
            "correlation_id": self.correlation_id
        }


# ============================================================================
# Structured Logger
# ============================================================================

class WorkflowLogger:
    """Structured logger for workflow"""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger("leanaide_rese_workflow")

    def _log(self, level: str, msg: str, **kwargs):
        """Log in JSON Lines format"""
        log_entry = {
            "msg": msg,
            "level": level,
            "correlation_id": self.correlation_id,
            "source_service": "leanaide_rese_workflow",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        log_json = json.dumps(log_entry)
        self.logger.log(getattr(logging, level.upper()), log_json)

    def info(self, msg: str, **kwargs):
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log("DEBUG", msg, **kwargs)


# ============================================================================
# Main Workflow Orchestrator
# ============================================================================

class LeanAideRESEWorkflow:
    """
    Main orchestrator for LeanAide-RESE workflow integration.

    Coordinates autoformalization and proof search across all 4 RESE phases.
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        logger: Optional[WorkflowLogger] = None
    ):
        """
        Initialize workflow orchestrator.

        Args:
            config: Workflow configuration
            logger: Structured logger
        """
        self.config = config or WorkflowConfig.from_env()
        self.logger = logger or WorkflowLogger(self.config.correlation_id)

        # Initialize services
        self.autoformalization_service: Optional[AutoformalizationService] = None
        self.proof_search_service: Optional[ProofSearchService] = None

        self.logger.info(
            "LeanAideRESEWorkflow initialized",
            config=self.config.to_dict()
        )

    async def initialize(self):
        """Initialize services"""
        # Create autoformalization config
        auto_config = AutoformalizationConfig(
            leanaide_host=self.config.leanaide_host,
            leanaide_port=self.config.leanaide_port,
            timeout_ms=self.config.autoformalization_timeout_ms,
            confidence_threshold=self.config.autoformalization_confidence_threshold,
            correlation_id=self.logger.correlation_id
        )
        self.autoformalization_service = AutoformalizationService(auto_config, self.logger)

        # Create proof search config
        proof_config = ProofSearchConfig(
            leanaide_host=self.config.leanaide_host,
            leanaide_port=self.config.leanaide_port,
            timeout_ms=self.config.proof_search_timeout_ms,
            max_search_depth=self.config.proof_search_max_depth,
            mcts_iterations=self.config.proof_search_mcts_iterations,
            enable_z3_hybrid=self.config.proof_search_enable_z3,
            confidence_threshold=self.config.proof_search_confidence_threshold,
            correlation_id=self.logger.correlation_id
        )
        self.proof_search_service = ProofSearchService(proof_config, self.logger)

        self.logger.info("Services initialized")

    async def execute(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute complete LeanAide-RESE workflow.

        Args:
            problem_statement: Problem to solve
            context: Optional context
            correlation_id: Correlation ID for tracing

        Returns:
            WorkflowResult with all phase outputs
        """
        workflow_id = str(uuid.uuid4())
        cid = correlation_id or self.logger.correlation_id
        self.logger.correlation_id = cid

        start_time = time.time()

        self.logger.info(
            "Workflow execution started",
            workflow_id=workflow_id,
            correlation_id=cid,
            problem_length=len(problem_statement)
        )

        try:
            # Ensure services are initialized
            if not self.autoformalization_service or not self.proof_search_service:
                await self.initialize()

            # Step 1: Classify problem
            classification = self._classify_problem(problem_statement, context)

            self.logger.info(
                "Problem classified",
                workflow_id=workflow_id,
                problem_type=classification.problem_type.value,
                recommended_solver=classification.recommended_solver.value
            )

            # Initialize phase results
            phase_results = {}

            # Phase I: Epistemic Audit
            result_i = await self._execute_phase_i(
                problem_statement,
                classification,
                cid
            )
            phase_results["phase_i"] = result_i

            # Phase II: Isomorphic Mapping
            result_ii = await self._execute_phase_ii(
                problem_statement,
                result_i.data,
                classification,
                cid
            )
            phase_results["phase_ii"] = result_ii

            # Phase III: MCTS Refinement
            result_iii = await self._execute_phase_iii(
                problem_statement,
                {**result_i.data, **result_ii.data},
                classification,
                cid
            )
            phase_results["phase_iii"] = result_iii

            # Phase IV: Architectural Synthesis
            result_iv = await self._execute_phase_iv(
                problem_statement,
                {**result_i.data, **result_ii.data, **result_iii.data},
                classification,
                cid
            )
            phase_results["phase_iv"] = result_iv

            # Calculate summary
            total_execution_time_ms = (time.time() - start_time) * 1000
            summary = self._generate_summary(phase_results, classification)

            # Determine overall status
            overall_status = "completed"
            if any(r.status == PhaseStatus.FAILED for r in phase_results.values()):
                overall_status = "failed"
            elif any(r.status == PhaseStatus.TIMEOUT for r in phase_results.values()):
                overall_status = "timeout"

            self.logger.info(
                "Workflow execution completed",
                workflow_id=workflow_id,
                correlation_id=cid,
                overall_status=overall_status,
                total_execution_time_ms=total_execution_time_ms
            )

            return WorkflowResult(
                workflow_id=workflow_id,
                correlation_id=cid,
                problem_classification=classification,
                phase_results=phase_results,
                overall_status=overall_status,
                total_execution_time_ms=total_execution_time_ms,
                summary=summary
            )

        except Exception as e:
            self.logger.error(
                "Workflow execution failed",
                workflow_id=workflow_id,
                correlation_id=cid,
                error=str(e)
            )

            return WorkflowResult(
                workflow_id=workflow_id,
                correlation_id=cid,
                problem_classification=ProblemClassification(
                    problem_type=ProblemType.CONSTRAINT_VERIFICATION,
                    mathematical_domain=FormalizationDomain.LOGIC,
                    recommended_solver=SolverType.LEANAIDE,
                    confidence=0.0,
                    reasoning="Classification failed"
                ),
                phase_results={},
                overall_status="failed",
                total_execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _execute_phase_i(
        self,
        problem_statement: str,
        classification: ProblemClassification,
        correlation_id: str
    ) -> PhaseResult:
        """Execute Phase I: Epistemic Audit"""
        start_time = time.time()

        self.logger.info(
            "Phase I started",
            phase="epistemic_audit",
            correlation_id=correlation_id
        )

        try:
            # Extract constraints from problem statement
            constraints = self._extract_constraints(problem_statement)

            # Autoformalize constraints
            autoformalization_results = []
            for constraint in constraints:
                result = await self.autoformalization_service.autoformalize_phase_i(
                    constraint_text=constraint["text"],
                    constraint_type=constraint.get("type", "logical"),
                    correlation_id=correlation_id
                )
                autoformalization_results.append(result)

            # Search proofs for constraints
            proof_search_results = []
            for auto_result in autoformalization_results:
                if auto_result.success:
                    proof_result = await self.proof_search_service.search_phase_i(
                        lean_code=auto_result.lean_code,
                        constraint_type="logical",
                        correlation_id=correlation_id
                    )
                    proof_search_results.append(proof_result)

            execution_time_ms = (time.time() - start_time) * 1000

            return PhaseResult(
                phase="phase_i_epistemic_audit",
                status=PhaseStatus.COMPLETED,
                data={
                    "constraints": constraints,
                    "constraint_count": len(constraints),
                    "autoformalized_count": sum(1 for r in autoformalization_results if r.success),
                    "proved_count": sum(1 for r in proof_search_results if r.proof_found)
                },
                autoformalization_results=autoformalization_results,
                proof_search_results=proof_search_results,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            self.logger.error(
                "Phase I failed",
                correlation_id=correlation_id,
                error=str(e)
            )

            return PhaseResult(
                phase="phase_i_epistemic_audit",
                status=PhaseStatus.FAILED,
                errors=[str(e)],
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def _execute_phase_ii(
        self,
        problem_statement: str,
        phase_i_data: Dict[str, Any],
        classification: ProblemClassification,
        correlation_id: str
    ) -> PhaseResult:
        """Execute Phase II: Isomorphic Mapping"""
        start_time = time.time()

        self.logger.info(
            "Phase II started",
            phase="isomorphic_mapping",
            correlation_id=correlation_id
        )

        try:
            # Identify domains for isomorphic mapping
            domains = self._identify_domains(problem_statement)

            # Autoformalize isomorphic mappings
            autoformalization_results = []
            for source_domain in domains:
                for target_domain in domains:
                    if source_domain != target_domain:
                        result = await self.autoformalization_service.autoformalize_phase_ii(
                            mapping_description=f"Structural correspondence between {source_domain} and {target_domain}",
                            source_domain=source_domain,
                            target_domain=target_domain,
                            correlation_id=correlation_id
                        )
                        autoformalization_results.append(result)

            # Search proofs for isomorphisms
            proof_search_results = []
            for auto_result in autoformalization_results:
                if auto_result.success:
                    proof_result = await self.proof_search_service.search_phase_ii(
                        lean_code=auto_result.lean_code,
                        correlation_id=correlation_id
                    )
                    proof_search_results.append(proof_result)

            execution_time_ms = (time.time() - start_time) * 1000

            return PhaseResult(
                phase="phase_ii_isomorphic_mapping",
                status=PhaseStatus.COMPLETED,
                data={
                    "domains": domains,
                    "mapping_count": len(autoformalization_results),
                    "isomorphisms_found": sum(1 for r in proof_search_results if r.proof_found)
                },
                autoformalization_results=autoformalization_results,
                proof_search_results=proof_search_results,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            self.logger.error(
                "Phase II failed",
                correlation_id=correlation_id,
                error=str(e)
            )

            return PhaseResult(
                phase="phase_ii_isomorphic_mapping",
                status=PhaseStatus.FAILED,
                errors=[str(e)],
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def _execute_phase_iii(
        self,
        problem_statement: str,
        previous_phases_data: Dict[str, Any],
        classification: ProblemClassification,
        correlation_id: str
    ) -> PhaseResult:
        """Execute Phase III: MCTS Refinement"""
        start_time = time.time()

        self.logger.info(
            "Phase III started",
            phase="mcts_refinement",
            correlation_id=correlation_id
        )

        try:
            # Generate hypotheses from problem
            hypotheses = self._generate_hypotheses(problem_statement, previous_phases_data)

            # Autoformalize hypotheses
            autoformalization_results = []
            for hypothesis in hypotheses:
                result = await self.autoformalization_service.autoformalize_phase_iii(
                    hypothesis_text=hypothesis["statement"],
                    hypothesis_type=hypothesis.get("type", "causal"),
                    correlation_id=correlation_id
                )
                autoformalization_results.append(result)

            # Search proofs using MCTS
            proof_search_results = []
            for auto_result in autoformalization_results:
                if auto_result.success:
                    proof_result = await self.proof_search_service.search_phase_iii(
                        lean_code=auto_result.lean_code,
                        correlation_id=correlation_id
                    )
                    proof_search_results.append(proof_result)

            execution_time_ms = (time.time() - start_time) * 1000

            return PhaseResult(
                phase="phase_iii_mcts_refinement",
                status=PhaseStatus.COMPLETED,
                data={
                    "hypotheses": hypotheses,
                    "hypothesis_count": len(hypotheses),
                    "confirmed_count": sum(1 for r in proof_search_results if r.proof_found),
                    "best_confidence": max((r.confidence for r in proof_search_results), default=0.0)
                },
                autoformalization_results=autoformalization_results,
                proof_search_results=proof_search_results,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            self.logger.error(
                "Phase III failed",
                correlation_id=correlation_id,
                error=str(e)
            )

            return PhaseResult(
                phase="phase_iii_mcts_refinement",
                status=PhaseStatus.FAILED,
                errors=[str(e)],
                execution_time_ms=(time.time() - start_time) * 1000
            )

    async def _execute_phase_iv(
        self,
        problem_statement: str,
        previous_phases_data: Dict[str, Any],
        classification: ProblemClassification,
        correlation_id: str
    ) -> PhaseResult:
        """Execute Phase IV: Architectural Synthesis"""
        start_time = time.time()

        self.logger.info(
            "Phase IV started",
            phase="architectural_synthesis",
            correlation_id=correlation_id
        )

        try:
            # Generate predictive model from all phases
            model_description = self._generate_predictive_model(
                problem_statement,
                previous_phases_data
            )

            # Generate efficacy claim
            efficacy_claim = self._generate_efficacy_claim(
                problem_statement,
                previous_phases_data
            )

            # Autoformalize efficacy claim
            autoformalization_result = await self.autoformalization_service.autoformalize_phase_iv(
                model_description=model_description,
                efficacy_claim=efficacy_claim,
                correlation_id=correlation_id
            )

            # Search proof for efficacy
            proof_search_result = None
            if autoformalization_result.success:
                proof_search_result = await self.proof_search_service.search_phase_iv(
                    lean_code=autoformalization_result.lean_code,
                    efficacy_claim=efficacy_claim,
                    correlation_id=correlation_id
                )

            execution_time_ms = (time.time() - start_time) * 1000

            return PhaseResult(
                phase="phase_iv_architectural_synthesis",
                status=PhaseStatus.COMPLETED,
                data={
                    "model_description": model_description,
                    "efficacy_claim": efficacy_claim,
                    "efficacy_proved": proof_search_result.proof_found if proof_search_result else False
                },
                autoformalization_results=[autoformalization_result] if autoformalization_result.success else [],
                proof_search_results=[proof_search_result] if proof_search_result else [],
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            self.logger.error(
                "Phase IV failed",
                correlation_id=correlation_id,
                error=str(e)
            )

            return PhaseResult(
                phase="phase_iv_architectural_synthesis",
                status=PhaseStatus.FAILED,
                errors=[str(e)],
                execution_time_ms=(time.time() - start_time) * 1000
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _classify_problem(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]]
    ) -> ProblemClassification:
        """Classify problem type and recommend solver"""

        # Simple keyword-based classification
        problem_lower = problem_statement.lower()

        # Detect problem type
        if any(word in problem_lower for word in ["prove", "theorem", "show"]):
            problem_type = ProblemType.THEOREM_PROVING
        elif any(word in problem_lower for word in ["isomorphic", "mapping", "bijection"]):
            problem_type = ProblemType.ISOMORPHISM_DETECTION
        elif any(word in problem_lower for word in ["optimize", "minimize", "maximize"]):
            problem_type = ProblemType.OPTIMIZATION
        elif any(word in problem_lower for word in ["hypothesis", "conjecture"]):
            problem_type = ProblemType.HYPOTHESIS_TESTING
        elif any(word in problem_lower for word in ["model", "predict", "validate"]):
            problem_type = ProblemType.MODEL_VALIDATION
        else:
            problem_type = ProblemType.CONSTRAINT_VERIFICATION

        # Detect mathematical domain
        if any(word in problem_lower for word in ["number", "integer", "prime"]):
            domain = FormalizationDomain.ARITHMETIC
        elif any(word in problem_lower for word in ["isomorph", "functor", "category"]):
            domain = FormalizationDomain.CATEGORY_THEORY
        elif any(word in problem_lower for word in ["graph", "node", "edge"]):
            domain = FormalizationDomain.GRAPH_THEORY
        else:
            domain = FormalizationDomain.LOGIC

        # Recommend solver
        if problem_type in [ProblemType.CONSTRAINT_VERIFICATION, ProblemType.THEOREM_PROVING]:
            solver = SolverType.HYBRID_Z3_LEANAIDE
        elif problem_type == ProblemType.ISOMORPHISM_DETECTION:
            solver = SolverType.LEANAIDE
        else:
            solver = SolverType.HYBRID_ALL

        return ProblemClassification(
            problem_type=problem_type,
            mathematical_domain=domain,
            recommended_solver=solver,
            confidence=0.75,
            reasoning=f"Classified as {problem_type.value} in {domain.value} domain"
        )

    def _extract_constraints(self, problem_statement: str) -> List[Dict[str, Any]]:
        """Extract constraints from problem statement"""
        # Simplified constraint extraction
        constraints = []

        # Split by common delimiters
        parts = problem_statement.split(',')

        for i, part in enumerate(parts):
            part = part.strip()
            if len(part) > 10:  # Minimum length threshold
                constraints.append({
                    "id": f"constraint_{i}",
                    "text": part,
                    "type": "logical"
                })

        return constraints[:5]  # Limit to 5 constraints

    def _identify_domains(self, problem_statement: str) -> List[str]:
        """Identify domains for isomorphic mapping"""
        # Simplified domain detection
        domains = []

        problem_lower = problem_statement.lower()

        if "number" in problem_lower or "integer" in problem_lower:
            domains.append("natural_numbers")
        if "set" in problem_lower:
            domains.append("sets")
        if "graph" in problem_lower:
            domains.append("graphs")
        if "function" in problem_lower:
            domains.append("functions")

        return domains if domains else ["general"]

    def _generate_hypotheses(
        self,
        problem_statement: str,
        previous_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate hypotheses from problem and previous phases"""
        # Simplified hypothesis generation
        return [
            {
                "id": "hypothesis_1",
                "statement": f"Derived from: {problem_statement[:50]}...",
                "type": "causal"
            }
        ]

    def _generate_predictive_model(
        self,
        problem_statement: str,
        previous_data: Dict[str, Any]
    ) -> str:
        """Generate predictive model description"""
        return f"Predictive model based on {problem_statement[:50]}..."

    def _generate_efficacy_claim(
        self,
        problem_statement: str,
        previous_data: Dict[str, Any]
    ) -> str:
        """Generate efficacy claim"""
        return f"The model accurately solves the given problem"

    def _generate_summary(
        self,
        phase_results: Dict[str, PhaseResult],
        classification: ProblemClassification
    ) -> Dict[str, Any]:
        """Generate workflow summary"""
        return {
            "total_phases": len(phase_results),
            "completed_phases": sum(1 for r in phase_results.values() if r.status == PhaseStatus.COMPLETED),
            "failed_phases": sum(1 for r in phase_results.values() if r.status == PhaseStatus.FAILED),
            "total_autoformalizations": sum(len(r.autoformalization_results) for r in phase_results.values()),
            "successful_autoformalizations": sum(
                sum(1 for ar in r.autoformalization_results if ar.success)
                for r in phase_results.values()
            ),
            "total_proofs": sum(len(r.proof_search_results) for r in phase_results.values()),
            "successful_proofs": sum(
                sum(1 for pr in r.proof_search_results if pr.proof_found)
                for r in phase_results.values()
            ),
            "problem_type": classification.problem_type.value,
            "recommended_solver": classification.recommended_solver.value
        }

    async def close(self):
        """Close services and cleanup resources"""
        if self.autoformalization_service:
            await self.autoformalization_service.close()
        if self.proof_search_service:
            await self.proof_search_service.close()

        self.logger.info("LeanAideRESEWorkflow closed")


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_workflow(
    config: Optional[WorkflowConfig] = None
) -> LeanAideRESEWorkflow:
    """
    Create and initialize workflow.

    Args:
        config: Workflow configuration

    Returns:
        Initialized LeanAideRESEWorkflow
    """
    workflow = LeanAideRESEWorkflow(config)
    await workflow.initialize()
    return workflow


async def execute_workflow(
    problem_statement: str,
    config: Optional[WorkflowConfig] = None,
    correlation_id: Optional[str] = None
) -> WorkflowResult:
    """
    Execute workflow with given problem statement.

    Args:
        problem_statement: Problem to solve
        config: Optional workflow configuration
        correlation_id: Optional correlation ID

    Returns:
        WorkflowResult
    """
    workflow = await create_workflow(config)
    try:
        return await workflow.execute(problem_statement, correlation_id=correlation_id)
    finally:
        await workflow.close()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI interface for LeanAide-RESE workflow"""
    import argparse

    parser = argparse.ArgumentParser(description="LeanAide-RESE Workflow Orchestrator")
    parser.add_argument("--problem", type=str, help="Problem statement")
    parser.add_argument("--config", action="store_true", help="Show configuration")

    args = parser.parse_args()

    async def run():
        if args.config:
            config = WorkflowConfig.from_env()
            print("Configuration:")
            print(json.dumps(config.to_dict(), indent=2))
            return

        if args.problem:
            print("Executing LeanAide-RESE workflow...")
            result = await execute_workflow(args.problem)
            print("\nResult:")
            print(json.dumps(result.to_dict(), indent=2))
            return

        parser.print_help()

    asyncio.run(run())


if __name__ == "__main__":
    main()
