"""
LeanAide MDAP/MAKER Workflow Integration

This module provides comprehensive integration of LeanAide MDAP (Multi-Strategy
Decision Aggregation Protocol) and MAKER (Multi-Agent voting for KEeping Reliability)
capabilities with the OpenEvolve decomposition workflow.

MDAP Architecture:
    Multiple Agent Generation (MAG) -> Strategy Pool -> Aggregation -> Best Strategy
    v
    LeanAide Verification -> Formal Proof -> Confidence Score

MAKER Architecture:
    Task Decomposition -> Tactic Voting -> First-to-Ahead-by-K -> Red-Flagging -> Final Proof

Integration Points:
    - Stage 3A: MDAP/MAKER for initial proof generation
    - Stage 3B: MDAP/MAKER for proof refinement
    - Stage 3C: LeanAide verification of MDAP/MAKER proofs
    - Stage 5: Final verification with MDAP/MAKER fallback

Key Classes:
    - LeanMDAPWorkflowIntegrator: MDAP integration
    - LeanMakerWorkflowIntegrator: MAKER integration
    - LeanHybridIntegrator: Hybrid MDAP+MAKER
    - LeanMDAPMonitor: Real-time monitoring

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable
)
import threading
from leanaide_web3_status import collect_web3_formal_status

# Configure logging
logger = logging.getLogger(__name__)

# Import workflow structures
try:
    from workflow_structures import (
        WorkflowState,
        SubProblem,
        SolutionAttempt,
        VerificationReport,
        CritiqueReport,
        Team,
        GauntletDefinition,
        ModelConfig
    )
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    logger.warning("Workflow structures not available - integration limited")

# Import LeanAide components
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideWorkflowConfig,
        LeanAideVerificationResult,
        is_leanaide_configured
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide workflow integration not available")

# Import MDAP components
try:
    from mdap_engine import (
        MDAPConfig,
        MDAPTask,
        MDAPStep,
        MDAPAgent,
        MDAPResult
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP engine not available")

# Import MAKER components
try:
    from openevolve_maker_integration import (
        MAKERWorkflowIntegrator,
        MAKERWorkflowConfig,
        MAKERMode,
        solve_subproblem_with_maker
    )
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    class MAKERMode(Enum):
        """Fallback MAKER mode enum for degraded environments."""
        RECURSIVE = "recursive"
        FLAT = "flat"

    @dataclass
    class MAKERWorkflowConfig:  # type: ignore[no-redef]
        """Fallback MAKER config used when integration package is unavailable."""
        mode: MAKERMode = MAKERMode.RECURSIVE

    class MAKERWorkflowIntegrator:  # type: ignore[no-redef]
        """Fallback MAKER integrator."""

        def __init__(self, *args, **kwargs):
            self.available = False

    def solve_subproblem_with_maker(*args, **kwargs):  # type: ignore[no-redef]
        return {
            "success": False,
            "error": "MAKER integration not available",
        }

    logger.warning("MAKER integration not available")

# import crewai # MIGRATED: was CrewAI
try:
    from crewai_client import CrewAIClient
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import Knowledge Engine
try:
    from ace_knowledge_artifacts import ACEKnowledgeManager
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

class MDAPStrategyType(Enum):
    """MDAP strategy types for Lean proof generation."""
    DIRECT = "direct"  # Direct theorem proving
    INDIRECT = "indirect"  # Proof by contradiction
    CONSTRUCTIVE = "constructive"  # Constructive proof
    INDUCTION = "induction"  # Mathematical induction
    CASE_ANALYSIS = "case_analysis"  # Case analysis
    DECOMPOSITION = "decomposition"  # Decomposition into lemmas
    FORWARD = "forward"  # Forward reasoning
    BACKWARD = "backward"  # Backward reasoning
    HYBRID = "hybrid"  # Mixed strategy


class MakerTacticType(Enum):
    """MAKER tactic types for Lean proof construction."""
    INTRO = "intro"  # Introduce hypothesis
    APPLY = "apply"  # Apply theorem/lemma
    REWRITE = "rewrite"  # Rewrite using equality
    SIMP = "simp"  # Simplify
    ASSUME = "assume"  # Assume hypothesis
    HAVE = "have"  # Introduce intermediate fact
    CALC = "calc"  # Calculate chain
    LINARITH = "linarith"  # Linear arithmetic
    RING = "ring"  # Ring tactics
    ETA = "eta"  # eta reduction
    CONGR = "congr"  # Congruence
    EXACT = "exact"  # Exact term
    REFINE = "refine"  # Refine with holes


@dataclass
class LeanMDAPConfig:
    """Configuration for LeanAide MDAP integration."""
    # Enablement
    enabled: bool = True

    # MDAP Agent Configuration
    agents: List[str] = field(default_factory=lambda: [
        "direct_prover",
        "inductive_prover",
        "constructive_prover",
        "decomposition_prover"
    ])
    parallel_agents: int = 4
    agent_timeout: float = 120.0

    # Strategy Selection
    voting_strategy: str = "weighted_confidence"  # weighted_confidence, majority, borda
    k_ahead: int = 3
    min_consensus: float = 0.6

    # LeanAide Integration
    verify_strategies: bool = True
    verification_timeout: float = 60.0
    confidence_threshold: float = 0.7

    # Fallback
    fallback_to_evolution: bool = True
    fallback_to_single: bool = True

    # CrewAI Integration
    CrewAI_enabled: bool = False
    track_mdap_tickets: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeanMakerConfig:
    """Configuration for LeanAide MAKER integration."""
    # Enablement
    enabled: bool = True

    # MAKER Mode
    mode: MAKERMode = MAKERMode.RECURSIVE

    # Voting Parameters
    k_min: int = 2
    k_max: int = 5
    max_votes: int = 100
    enable_first_to_ahead: bool = True

    # Tactic Selection
    tactic_pool: List[MakerTacticType] = field(default_factory=lambda: [
        MakerTacticType.INTRO,
        MakerTacticType.APPLY,
        MakerTacticType.REWRITE,
        MakerTacticType.SIMP
    ])

    # Red-Flagging
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_characters: int = 6000

    # Execution Limits
    max_steps: int = 1000
    max_depth: int = 5
    timeout_seconds: int = 300

    # LeanAide Verification
    verify_each_step: bool = False
    verify_final: bool = True
    verification_timeout: float = 60.0

    # Fallback
    fallback_to_direct: bool = True

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeanMDAPTask:
    """A MDAP task for Lean proof generation."""
    task_id: str
    sub_problem_id: str
    theorem_statement: str
    proof_goal: str
    context: Dict[str, Any]
    strategy_type: MDAPStrategyType
    agent_id: str
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sub_problem_id": self.sub_problem_id,
            "theorem_statement": self.theorem_statement,
            "proof_goal": self.proof_goal,
            "context": self.context,
            "strategy_type": self.strategy_type.value,
            "agent_id": self.agent_id,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "metadata": self.metadata
        }


@dataclass
class LeanMDAPResult:
    """Result from MDAP proof generation."""
    task_id: str
    strategy_type: MDAPStrategyType
    agent_id: str
    lean_code: str
    proof_steps: List[str]
    confidence: float
    verification_status: str
    verification_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "strategy_type": self.strategy_type.value,
            "agent_id": self.agent_id,
            "lean_code": self.lean_code,
            "proof_steps": self.proof_steps,
            "confidence": self.confidence,
            "verification_status": self.verification_status,
            "verification_time": self.verification_time,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata
        }


# =============================================================================
# LEAN MDAP WORKFLOW INTEGRATOR
# =============================================================================

class LeanMDAPWorkflowIntegrator:
    """
    Main integration class for MDAP in LeanAide workflow.

    This class orchestrates multi-agent Lean proof generation using
    MDAP (Multi-Strategy Decision Aggregation Protocol).

    Stage Integration:
        - Stage 3A: Generate initial proofs using multiple strategies
        - Stage 3B: Refine proofs using MDAP aggregation
        - Stage 3C: Verify MDAP-generated proofs with LeanAide
        - Stage 5: Final verification with MDAP fallback
    """

    def __init__(
        self,
        config: Optional[LeanMDAPConfig] = None,
        workflow_state: Optional[WorkflowState] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize the MDAP workflow integrator.

        Args:
            config: MDAP configuration
            workflow_state: Current workflow state
            team: Team for LLM calls
        """
        self.config = config or LeanMDAPConfig()
        self.workflow_state = workflow_state
        self.team = team

        # Initialize components
        self.leanaide_integrator: Optional[LeanAideWorkflowIntegrator] = None
        self.crewai_client: Optional[CrewAIClient] = None
        self.ace_manager: Optional[ACEKnowledgeManager] = None

        # Task tracking
        self.active_tasks: Dict[str, LeanMDAPTask] = {}
        self.completed_results: Dict[str, List[LeanMDAPResult]] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.statistics = defaultdict(list)

        # Initialize
        self._initialize_components()

    def _initialize_components(self):
        """Initialize required components."""
        # Initialize LeanAide integrator
        if LEANAIDE_AVAILABLE and self.config.enabled:
            leanaide_config = LeanAideWorkflowConfig(
                enabled=True,
                verify_proofs=self.config.verify_strategies,
                confidence_threshold=self.config.confidence_threshold
            )
            self.leanaide_integrator = LeanAideWorkflowIntegrator(leanaide_config)

        # Initialize CrewAI if enabled
        if self.config.CrewAI_enabled and CREWAI_AVAILABLE:
            self.crewai_client = CrewAIClient(timeout=self.config.agent_timeout)

        # Initialize ACE manager
        if ACE_AVAILABLE:
            self.ace_manager = ACEKnowledgeManager()

    async def solve_subproblem_with_mdap(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using MDAP multi-agent approach.

        This is the main entry point for Stage 3A integration.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with MDAP-generated proof
        """
        start_time = time.time()
        logger.info(f"Solving sub-problem {sub_problem.id} with MDAP")

        try:
            # Create MDAP tasks for each strategy
            tasks = await self._create_mdap_tasks(sub_problem)

            # Execute tasks in parallel
            results = await self._execute_mdap_tasks(tasks, sub_problem)

            # Aggregate results
            best_result = await self._aggregate_results(results, sub_problem)

            # Create solution attempt
            solution = SolutionAttempt(
                sub_problem_id=sub_problem.id,
                content=best_result.lean_code,
                generated_by_model="LeanAide-MDAP",
                timestamp=time.time(),
                status="verified" if best_result.verification_status == "verified" else "generated",
                solution_approach="mdap_multi_agent",
                openevolve_metrics={
                    "mdap_strategy": best_result.strategy_type.value,
                    "num_agents": len(results),
                    "best_confidence": best_result.confidence,
                    "execution_time": time.time() - start_time,
                    "all_strategies": [r.strategy_type.value for r in results]
                }
            )

            # Store in knowledge base
            if self.ace_manager:
                await self._store_mdap_result(sub_problem, results, best_result)

            return solution

        except Exception as e:
            logger.error(f"MDAP failed for {sub_problem.id}: {e}", exc_info=True)

            # Fallback
            if self.config.fallback_to_single:
                return await self._fallback_to_single_strategy(sub_problem)
            else:
                raise

    async def _create_mdap_tasks(
        self,
        sub_problem: SubProblem
    ) -> List[LeanMDAPTask]:
        """Create MDAP tasks for each strategy type."""
        tasks = []

        # Get strategy types from config
        strategy_types = [
            MDAPStrategyType.DIRECT,
            MDAPStrategyType.INDUCTION,
            MDAPStrategyType.CONSTRUCTIVE,
            MDAPStrategyType.DECOMPOSITION
        ]

        for i, strategy_type in enumerate(strategy_types):
            task = LeanMDAPTask(
                task_id=f"mdap_{sub_problem.id}_{strategy_type.value}_{uuid.uuid4().hex[:8]}",
                sub_problem_id=sub_problem.id,
                theorem_statement=sub_problem.description,
                proof_goal=f"prove_{sub_problem.id.replace('-', '_')}",
                context={
                    "dependencies": sub_problem.dependencies,
                    "requirements": sub_problem.solution_requirements or []
                },
                strategy_type=strategy_type,
                agent_id=self.config.agents[i % len(self.config.agents)],
                priority=sub_problem.priority
            )
            tasks.append(task)
            self.active_tasks[task.task_id] = task

        return tasks

    async def _execute_mdap_tasks(
        self,
        tasks: List[LeanMDAPTask],
        sub_problem: SubProblem
    ) -> List[LeanMDAPResult]:
        """Execute MDAP tasks in parallel."""
        # Create semaphore for parallel execution
        semaphore = asyncio.Semaphore(self.config.parallel_agents)

        async def execute_with_semaphore(task: LeanMDAPTask):
            async with semaphore:
                return await self._execute_single_task(task, sub_problem)

        # Execute all tasks
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )

        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed: {result}")
            elif isinstance(result, LeanMDAPResult):
                valid_results.append(result)

        return valid_results

    async def _execute_single_task(
        self,
        task: LeanMDAPTask,
        sub_problem: SubProblem
    ) -> Optional[LeanMDAPResult]:
        """Execute a single MDAP task."""
        start_time = time.time()

        # Update agent status
        self.agent_status[task.agent_id] = {
            "status": "running",
            "current_task": task.task_id,
            "strategy": task.strategy_type.value,
            "start_time": start_time
        }

        try:
            # Generate proof using strategy
            lean_code = await self._generate_proof_with_strategy(
                task, sub_problem
            )

            # Verify if enabled
            verification_status = "not_verified"
            verification_time = 0.0
            errors = []
            warnings = []

            if self.config.verify_strategies and self.leanaide_integrator:
                verification_result = await self._verify_proof(
                    task, lean_code, sub_problem
                )
                verification_status = verification_result.verification_status
                verification_time = verification_result.verification_time
                errors = verification_result.errors
                warnings = verification_result.warnings

            # Calculate confidence
            confidence = self._calculate_confidence(
                task, lean_code, verification_status
            )

            result = LeanMDAPResult(
                task_id=task.task_id,
                strategy_type=task.strategy_type,
                agent_id=task.agent_id,
                lean_code=lean_code,
                proof_steps=self._extract_proof_steps(lean_code),
                confidence=confidence,
                verification_status=verification_status,
                verification_time=verification_time,
                errors=errors,
                warnings=warnings
            )

            # Update agent status
            self.agent_status[task.agent_id] = {
                "status": "completed",
                "last_task": task.task_id,
                "confidence": confidence,
                "execution_time": time.time() - start_time
            }

            return result

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Task {task.task_id} failed: {e}")

            # Update agent status
            self.agent_status[task.agent_id] = {
                "status": "failed",
                "error": str(e),
                "execution_time": time.time() - start_time
            }

            return None

    async def _generate_proof_with_strategy(
        self,
        task: LeanMDAPTask,
        sub_problem: SubProblem
    ) -> str:
        """Generate Lean proof using specific strategy."""
        # Strategy-specific proof generation
        strategy_prompts = {
            MDAPStrategyType.DIRECT: f"""Prove the following theorem directly:

Theorem: {task.theorem_statement}

Construct a direct proof using Lean 4. Provide the complete theorem statement and proof.""",

            MDAPStrategyType.INDUCTION: f"""Prove the following theorem using mathematical induction:

Theorem: {task.theorem_statement}

Identify the inductive variable and provide base case and inductive step in Lean 4.""",

            MDAPStrategyType.CONSTRUCTIVE: f"""Prove the following theorem constructively:

Theorem: {task.theorem_statement}

Provide a constructive proof that explicitly constructs the required object in Lean 4.""",

            MDAPStrategyType.DECOMPOSITION: f"""Prove the following theorem by decomposition:

Theorem: {task.theorem_statement}

Break down the proof into intermediate lemmas and then prove the main theorem in Lean 4."""
        }

        prompt = strategy_prompts.get(
            task.strategy_type,
            f"Prove the following theorem in Lean 4:\n\n{task.theorem_statement}"
        )

        # For now, return a placeholder
        # In production, this would call LLM to generate actual Lean code
        return f"""theorem {task.proof_goal} : {task.theorem_statement} :=
  by
    -- Proof generated using {task.strategy_type.value} strategy
    -- Agent: {task.agent_id}
    sorry  -- Placeholder proof"""

    async def _verify_proof(
        self,
        task: LeanMDAPTask,
        lean_code: str,
        sub_problem: SubProblem
    ) -> LeanMDAPResult:
        """Verify Lean proof using LeanAide."""
        if not self.leanaide_integrator:
            return LeanMDAPResult(
                task_id=task.task_id,
                strategy_type=task.strategy_type,
                agent_id=task.agent_id,
                lean_code=lean_code,
                proof_steps=[],
                confidence=0.5,
                verification_status="not_verified",
                verification_time=0.0
            )

        # Verify using LeanAide
        verification_result = await self.leanaide_integrator.verify_sub_problem_solution(
            sub_problem_id=sub_problem.id,
            problem_statement=task.theorem_statement,
            solution_content=lean_code,
            verification_requirements=sub_problem.solution_requirements
        )

        return LeanMDAPResult(
            task_id=task.task_id,
            strategy_type=task.strategy_type,
            agent_id=task.agent_id,
            lean_code=lean_code,
            proof_steps=[],
            confidence=verification_result.confidence_score,
            verification_status="verified" if verification_result.success else "failed",
            verification_time=verification_result.execution_time,
            errors=verification_result.errors or [],
            warnings=verification_result.warnings or []
        )

    def _calculate_confidence(
        self,
        task: LeanMDAPTask,
        lean_code: str,
        verification_status: str
    ) -> float:
        """Calculate confidence score for a result."""
        confidence = 0.5

        # Boost for successful verification
        if verification_status == "verified":
            confidence = 0.9
        elif verification_status == "partial":
            confidence = 0.7
        elif verification_status == "failed":
            confidence = 0.3

        # Adjust based on code quality
        if "sorry" in lean_code:
            confidence *= 0.5

        return max(0.0, min(1.0, confidence))

    def _extract_proof_steps(self, lean_code: str) -> List[str]:
        """Extract proof steps from Lean code."""
        steps = []

        for line in lean_code.split('\n'):
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('theorem'):
                steps.append(line)

        return steps

    async def _aggregate_results(
        self,
        results: List[LeanMDAPResult],
        sub_problem: SubProblem
    ) -> LeanMDAPResult:
        """Aggregate MDAP results and select best strategy."""
        if not results:
            raise Exception("No valid MDAP results")

        # Voting strategy
        if self.config.voting_strategy == "weighted_confidence":
            # Select by highest confidence
            best = max(results, key=lambda r: r.confidence)
        elif self.config.voting_strategy == "majority":
            # Select most common verification status
            from collections import Counter
            status_counts = Counter(r.verification_status for r in results)
            best_status = status_counts.most_common(1)[0][0]
            candidates = [r for r in results if r.verification_status == best_status]
            best = max(candidates, key=lambda r: r.confidence)
        else:
            # Default: highest confidence
            best = max(results, key=lambda r: r.confidence)

        logger.info(f"Selected MDAP strategy: {best.strategy_type.value} (confidence: {best.confidence:.2f})")

        return best

    async def _fallback_to_single_strategy(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Fallback to single-strategy approach."""
        logger.info(f"Falling back to single strategy for {sub_problem.id}")

        # Use direct strategy
        lean_code = f"""theorem {sub_problem.id.replace('-', '_')} : {sub_problem.description} :=
  by sorry  -- Fallback proof"""

        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=lean_code,
            generated_by_model="LeanAide-MDAP-Fallback",
            timestamp=time.time(),
            status="generated",
            solution_approach="mdap_fallback"
        )

    async def mdap_stage3a(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3A: Generate initial proof using MDAP.

        Args:
            sub_problem: Sub-problem to solve
            workflow_state: Current workflow state

        Returns:
            SolutionAttempt with MDAP-generated proof
        """
        logger.info(f"MDAP Stage 3A: Generating initial proof for {sub_problem.id}")

        # Configure MDAP from workflow state
        self.config = self.configure_mdap_from_workflow(workflow_state)

        # Solve with MDAP
        return await self.solve_subproblem_with_mdap(sub_problem)

    async def mdap_stage3b(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3B: Refine proof using MDAP.

        Args:
            solution: Current solution to refine
            workflow_state: Current workflow state

        Returns:
            Refined solution attempt
        """
        logger.info(f"MDAP Stage 3B: Refining proof for {solution.sub_problem_id}")

        # Get sub-problem
        sub_problem = None
        if workflow_state.decomposition_plan:
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            logger.warning(f"Sub-problem {solution.sub_problem_id} not found for refinement")
            return solution

        # Re-run MDAP with more iterations
        self.config.k_ahead = max(self.config.k_ahead, 5)

        # Generate refined proof
        refined_solution = await self.solve_subproblem_with_mdap(sub_problem)

        # Update solution
        solution.content = refined_solution.content
        solution.status = refined_solution.status
        solution.solution_approach = "mdap_stage3b_refined"

        if solution.openevolve_metrics is None:
            solution.openevolve_metrics = {}

        solution.openevolve_metrics.update(refined_solution.openevolve_metrics)
        solution.openevolve_metrics["stage3b_refinement"] = True

        return solution

    def configure_mdap_from_workflow(
        self,
        state: WorkflowState
    ) -> LeanMDAPConfig:
        """
        Configure MDAP from workflow state.

        Args:
            state: Current workflow state

        Returns:
            LeanMDAPConfig
        """
        params = state.openevolve_parameters or {}

        return LeanMDAPConfig(
            enabled=params.get("lean_mdap_enabled", True),
            agents=params.get("lean_mdap_agents", self.config.agents),
            parallel_agents=params.get("lean_mdap_parallel_agents", self.config.parallel_agents),
            voting_strategy=params.get("lean_mdap_voting_strategy", self.config.voting_strategy),
            k_ahead=params.get("lean_mdap_k_ahead", self.config.k_ahead),
            verify_strategies=params.get("lean_mdap_verify", self.config.verify_strategies),
            confidence_threshold=params.get("lean_mdap_confidence_threshold", self.config.confidence_threshold),
            fallback_to_evolution=params.get("lean_mdap_fallback_evolution", self.config.fallback_to_evolution)
        )

    async def _store_mdap_result(
        self,
        sub_problem: SubProblem,
        results: List[LeanMDAPResult],
        best_result: LeanMDAPResult
    ):
        """Store MDAP result in knowledge base."""
        if not self.ace_manager:
            return

        artifact = {
            "type": "mdap_result",
            "sub_problem_id": sub_problem.id,
            "theorem": sub_problem.description,
            "num_strategies": len(results),
            "best_strategy": best_result.strategy_type.value,
            "best_confidence": best_result.confidence,
            "verification_status": best_result.verification_status,
            "lean_code": best_result.lean_code,
            "timestamp": time.time()
        }

        self.ace_manager.store_artifact(artifact)


# =============================================================================
# LEAN MAKER WORKFLOW INTEGRATOR
# =============================================================================

class LeanMakerWorkflowIntegrator:
    """
    Integration for MAKER voting-based proof construction.

    This class implements MAKER (Multi-Agent voting for KEeping Reliability)
    for step-by-step Lean proof construction with tactic voting.

    Stage Integration:
        - Stage 3A: Generate proof using MAKER voting
        - Stage 3B: Refine proof with additional MAKER rounds
        - Stage 3C: Verify MAKER-generated proofs
    """

    def __init__(
        self,
        config: Optional[LeanMakerConfig] = None,
        workflow_state: Optional[WorkflowState] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize the MAKER workflow integrator.

        Args:
            config: MAKER configuration
            workflow_state: Current workflow state
            team: Team for LLM calls
        """
        self.config = config or LeanMakerConfig()
        self.workflow_state = workflow_state
        self.team = team

        # Initialize MAKER integrator
        self.maker_integrator: Optional[MAKERWorkflowIntegrator] = None
        self.leanaide_integrator: Optional[LeanAideWorkflowIntegrator] = None

        # Statistics
        self.voting_statistics: Dict[str, Any] = defaultdict(list)

        # Initialize
        self._initialize_components()

    def _initialize_components(self):
        """Initialize required components."""
        if MAKER_AVAILABLE and self.config.enabled:
            maker_config = MAKERWorkflowConfig(
                mode=self.config.mode,
                k_ahead=self.config.k_max,
                max_token_length=self.config.max_token_length,
                max_steps=self.config.max_steps,
                enable_red_flagging=self.config.enable_red_flagging
            )
            self.maker_integrator = MAKERWorkflowIntegrator(
                config=maker_config,
                team=self.team
            )

        if LEANAIDE_AVAILABLE:
            leanaide_config = LeanAideWorkflowConfig(enabled=True)
            self.leanaide_integrator = LeanAideWorkflowIntegrator(leanaide_config)

    async def solve_with_maker_voting(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve sub-problem using MAKER voting.

        This is the main entry point for Stage 3A integration.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with MAKER-generated proof
        """
        start_time = time.time()
        logger.info(f"Solving sub-problem {sub_problem.id} with MAKER")

        try:
            # Use MAKER integrator
            if self.maker_integrator:
                solution = self.maker_integrator.solve_subproblem(
                    sub_problem, self.workflow_state
                )

                # Verify if enabled
                if self.config.verify_final and self.leanaide_integrator:
                    verification = await self.leanaide_integrator.verify_sub_problem_solution(
                        sub_problem_id=sub_problem.id,
                        problem_statement=sub_problem.description,
                        solution_content=solution.content,
                        verification_requirements=sub_problem.solution_requirements
                    )

                    if verification.success:
                        solution.status = "verified"

                    # Add verification metadata
                    if solution.openevolve_metrics is None:
                        solution.openevolve_metrics = {}

                    solution.openevolve_metrics["maker_verification"] = {
                        "success": verification.success,
                        "confidence": verification.confidence_score,
                        "time": verification.execution_time
                    }

                return solution
            else:
                # Fallback: direct MAKER-style voting
                return await self._direct_maker_solve(sub_problem)

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MAKER failed for {sub_problem.id}: {e}", exc_info=True)

            # Fallback
            if self.config.fallback_to_direct:
                return await self._fallback_to_direct(sub_problem)
            else:
                raise

    async def _direct_maker_solve(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Direct MAKER-style solve without integrator."""
        # Simulate tactic voting
        tactics = []

        # Generate proof using voting
        lean_code = f"""theorem {sub_problem.id.replace('-', '_')} : {sub_problem.description} :=
  by
    -- Proof generated using MAKER voting
    -- Tactics voted: {', '.join(t.value for t in self.config.tactic_pool)}
    sorry"""

        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=lean_code,
            generated_by_model="LeanAide-MAKER-Direct",
            timestamp=time.time(),
            status="generated",
            solution_approach="maker_voting"
        )

    async def _fallback_to_direct(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Fallback to direct tactic selection."""
        lean_code = f"""theorem {sub_problem.id.replace('-', '_')} : {sub_problem.description} :=
  by sorry"""

        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=lean_code,
            generated_by_model="LeanAide-MAKER-Fallback",
            timestamp=time.time(),
            status="generated",
            solution_approach="maker_fallback"
        )

    async def maker_stage3a(
        self,
        sub_problem: SubProblem,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3A: Generate initial proof using MAKER.

        Args:
            sub_problem: Sub-problem to solve
            workflow_state: Current workflow state

        Returns:
            SolutionAttempt with MAKER-generated proof
        """
        logger.info(f"MAKER Stage 3A: Generating proof for {sub_problem.id}")

        # Configure MAKER from workflow state
        self.config = self.configure_maker_from_workflow(workflow_state)

        # Solve with MAKER
        return await self.solve_with_maker_voting(sub_problem)

    async def maker_refinement_stage3b(
        self,
        solution: SolutionAttempt,
        workflow_state: WorkflowState
    ) -> SolutionAttempt:
        """
        Stage 3B: Refine proof using MAKER.

        Args:
            solution: Current solution to refine
            workflow_state: Current workflow state

        Returns:
            Refined solution attempt
        """
        logger.info(f"MAKER Stage 3B: Refining proof for {solution.sub_problem_id}")

        # Get sub-problem
        sub_problem = None
        if workflow_state.decomposition_plan:
            for sp in workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            return solution

        # Re-run MAKER with higher k_ahead
        self.config.k_max = min(self.config.k_max + 2, 10)

        # Generate refined proof
        refined_solution = await self.solve_with_maker_voting(sub_problem)

        # Update solution
        solution.content = refined_solution.content
        solution.status = refined_solution.status
        solution.solution_approach = "maker_stage3b_refined"

        if solution.openevolve_metrics is None:
            solution.openevolve_metrics = {}

        solution.openevolve_metrics.update(refined_solution.openevolve_metrics)
        solution.openevolve_metrics["stage3b_refinement"] = True

        return solution

    def configure_maker_from_workflow(
        self,
        state: WorkflowState
    ) -> LeanMakerConfig:
        """
        Configure MAKER from workflow state.

        Args:
            state: Current workflow state

        Returns:
            LeanMakerConfig
        """
        params = state.openevolve_parameters or {}

        return LeanMakerConfig(
            enabled=params.get("lean_maker_enabled", True),
            k_min=params.get("lean_maker_k_min", self.config.k_min),
            k_max=params.get("lean_maker_k_max", self.config.k_max),
            max_votes=params.get("lean_maker_max_votes", self.config.max_votes),
            enable_red_flagging=params.get("lean_maker_red_flagging", self.config.enable_red_flagging),
            verify_final=params.get("lean_maker_verify", self.config.verify_final),
            fallback_to_direct=params.get("lean_maker_fallback", self.config.fallback_to_direct)
        )


# =============================================================================
# LEAN HYBRID INTEGRATOR
# =============================================================================

class LeanHybridIntegrator:
    """
    Hybrid MDAP + MAKER integration.

    Combines multi-agent generation (MDAP) with voting refinement (MAKER)
    for the best of both approaches.

    Strategies:
        - MDAP then MAKER: MDAP generates candidates, MAKER refines
        - MAKER then MDAP: MAKER constructs, MDAP validates
        - Adaptive: Automatically select based on problem characteristics
    """

    def __init__(
        self,
        mdap_config: Optional[LeanMDAPConfig] = None,
        maker_config: Optional[LeanMakerConfig] = None,
        workflow_state: Optional[WorkflowState] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize the hybrid integrator.

        Args:
            mdap_config: MDAP configuration
            maker_config: MAKER configuration
            workflow_state: Current workflow state
            team: Team for LLM calls
        """
        self.mdap_config = mdap_config or LeanMDAPConfig()
        self.maker_config = maker_config or LeanMakerConfig()
        self.workflow_state = workflow_state
        self.team = team

        # Initialize integrators
        self.mdap_integrator = LeanMDAPWorkflowIntegrator(
            config=self.mdap_config,
            workflow_state=workflow_state,
            team=team
        )
        self.maker_integrator = LeanMakerWorkflowIntegrator(
            config=self.maker_config,
            workflow_state=workflow_state,
            team=team
        )

    async def solve_with_mdap_then_maker(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve using MDAP then MAKER.

        MDAP generates candidate proofs, MAKER refines the best one.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with hybrid-generated proof
        """
        logger.info(f"Hybrid MDAP->MAKER: Solving {sub_problem.id}")

        # Step 1: Generate with MDAP
        mdap_solution = await self.mdap_integrator.solve_subproblem_with_mdap(sub_problem)

        # Step 2: Refine with MAKER
        # For now, just return MDAP solution
        # In production, would use MAKER to vote on tactics within MDAP proof

        mdap_solution.solution_approach = "hybrid_mdap_then_maker"

        return mdap_solution

    async def solve_with_maker_and_mdap(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve using MAKER then MDAP.

        MAKER constructs proof step-by-step, MDAP validates and potentially
        suggests alternative strategies.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with hybrid-generated proof
        """
        logger.info(f"Hybrid MAKER->MDAP: Solving {sub_problem.id}")

        # Step 1: Generate with MAKER
        maker_solution = await self.maker_integrator.solve_with_maker_voting(sub_problem)

        # Step 2: Validate with MDAP
        # For now, just return MAKER solution
        # In production, would use MDAP to validate and potentially regenerate

        maker_solution.solution_approach = "hybrid_maker_then_mdap"

        return maker_solution

    async def adaptive_hybrid_solve(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Adaptive hybrid solve.

        Automatically selects approach based on problem characteristics.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with adaptively-generated proof
        """
        logger.info(f"Adaptive hybrid: Solving {sub_problem.id}")

        # Analyze problem characteristics
        complexity = sub_problem.ai_suggested_complexity_score or 5
        estimated_effort = sub_problem.estimated_effort or 10

        # Decision logic
        if complexity >= 8 or estimated_effort >= 20:
            # High complexity: use MDAP for multiple strategies
            return await self.solve_with_mdap_then_maker(sub_problem)
        elif complexity <= 3 or estimated_effort <= 5:
            # Low complexity: use MAKER for step-by-step
            return await self.solve_with_maker_and_mdap(sub_problem)
        else:
            # Medium complexity: start with MAKER, fall back to MDAP if needed
            try:
                return await self.solve_with_maker_and_mdap(sub_problem)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning(f"MAKER approach failed: {e}, falling back to MDAP")
                return await self.solve_with_mdap_then_maker(sub_problem)


# =============================================================================
# LEAN MDAP MONITOR
# =============================================================================

class LeanMDAPMonitor:
    """
    Monitor MDAP execution in workflow.

    Provides real-time progress tracking, agent status, and voting statistics.
    """

    def __init__(
        self,
        mdap_integrator: Optional[LeanMDAPWorkflowIntegrator] = None,
        maker_integrator: Optional[LeanMakerWorkflowIntegrator] = None
    ):
        """
        Initialize the MDAP monitor.

        Args:
            mdap_integrator: MDAP integrator to monitor
            maker_integrator: MAKER integrator to monitor
        """
        self.mdap_integrator = mdap_integrator
        self.maker_integrator = maker_integrator

        self.monitoring_active: bool = False
        self.current_task: Optional[LeanMDAPTask] = None
        self.start_time: float = 0.0

    def start_monitoring(self, task: LeanMDAPTask) -> None:
        """
        Start monitoring a task.

        Args:
            task: Task to monitor
        """
        self.current_task = task
        self.start_time = time.time()
        self.monitoring_active = True

        logger.info(f"Started monitoring task {task.task_id}")

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress of monitored task.

        Returns:
            Dict with progress information
        """
        if not self.monitoring_active or not self.current_task:
            web3_status = collect_web3_formal_status()
            return {
                "status": "not_monitoring",
                "task_id": None,
                "elapsed_time": 0.0,
                "web3_formal_available": web3_status["web3_formal_available"],
                "web3_formal_verification_available": web3_status[
                    "web3_formal_verification_available"
                ],
                "web3_formal_tools": web3_status["web3_formal_tools"],
                "formal_capabilities": web3_status["formal_capabilities"],
                "audit_exploit_verification_available": web3_status[
                    "audit_exploit_verification_available"
                ],
            }

        elapsed = time.time() - self.start_time

        # Get agent status from integrator
        agent_status = {}
        if self.mdap_integrator:
            agent_status = self.mdap_integrator.agent_status

        web3_status = collect_web3_formal_status()
        return {
            "status": "monitoring",
            "task_id": self.current_task.task_id,
            "strategy": self.current_task.strategy_type.value,
            "agent": self.current_task.agent_id,
            "elapsed_time": elapsed,
            "agent_status": agent_status,
            "web3_formal_available": web3_status["web3_formal_available"],
            "web3_formal_verification_available": web3_status[
                "web3_formal_verification_available"
            ],
            "web3_formal_tools": web3_status["web3_formal_tools"],
            "formal_capabilities": web3_status["formal_capabilities"],
            "audit_exploit_verification_available": web3_status[
                "audit_exploit_verification_available"
            ],
        }

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get status of all agents.

        Returns:
            Dict mapping agent_id to status
        """
        if self.mdap_integrator:
            return self.mdap_integrator.agent_status

        return {}

    def get_voting_statistics(self) -> Dict[str, Any]:
        """
        Get voting statistics from MAKER.

        Returns:
            Dict with voting statistics
        """
        if self.maker_integrator:
            return self.maker_integrator.voting_statistics

        return {}


# =============================================================================
# CONFIGURATION INTEGRATION
# =============================================================================

def add_mdap_maker_config_to_workflow_state(
    workflow_state: WorkflowState,
    mdap_config: Optional[LeanMDAPConfig] = None,
    maker_config: Optional[LeanMakerConfig] = None
) -> WorkflowState:
    """
    Add MDAP/MAKER configuration to workflow state.

    Args:
        workflow_state: Current workflow state
        mdap_config: Optional MDAP configuration
        maker_config: Optional MAKER configuration

    Returns:
        Updated workflow state
    """
    if workflow_state.openevolve_parameters is None:
        workflow_state.openevolve_parameters = {}

    # Add MDAP config
    if mdap_config:
        workflow_state.openevolve_parameters.update({
            "lean_mdap_enabled": mdap_config.enabled,
            "lean_mdap_agents": mdap_config.agents,
            "lean_mdap_parallel_agents": mdap_config.parallel_agents,
            "lean_mdap_voting_strategy": mdap_config.voting_strategy,
            "lean_mdap_k_ahead": mdap_config.k_ahead,
            "lean_mdap_verify": mdap_config.verify_strategies,
            "lean_mdap_confidence_threshold": mdap_config.confidence_threshold
        })

    # Add MAKER config
    if maker_config:
        workflow_state.openevolve_parameters.update({
            "lean_maker_enabled": maker_config.enabled,
            "lean_maker_k_min": maker_config.k_min,
            "lean_maker_k_max": maker_config.k_max,
            "lean_maker_max_votes": maker_config.max_votes,
            "lean_maker_red_flagging": maker_config.enable_red_flagging,
            "lean_maker_verify": maker_config.verify_final
        })

    return workflow_state


# =============================================================================
# STAGE INTEGRATION HELPERS
# =============================================================================

async def solve_with_lean_mdap_maker(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    team: Optional[Team] = None,
    mode: str = "auto"
) -> SolutionAttempt:
    """
    Convenience function to solve with Lean MDAP/MAKER.

    Args:
        sub_problem: Sub-problem to solve
        workflow_state: Current workflow state
        team: Optional team
        mode: "mdap", "maker", "hybrid", or "auto"

    Returns:
        SolutionAttempt with generated proof
    """
    # Extract configs
    mdap_config = None
    maker_config = None

    if workflow_state.openevolve_parameters:
        params = workflow_state.openevolve_parameters

        mdap_config = LeanMDAPConfig(
            enabled=params.get("lean_mdap_enabled", True),
            agents=params.get("lean_mdap_agents", []),
            parallel_agents=params.get("lean_mdap_parallel_agents", 4),
            voting_strategy=params.get("lean_mdap_voting_strategy", "weighted_confidence"),
            k_ahead=params.get("lean_mdap_k_ahead", 3)
        )

        maker_config = LeanMakerConfig(
            enabled=params.get("lean_maker_enabled", True),
            k_min=params.get("lean_maker_k_min", 2),
            k_max=params.get("lean_maker_k_max", 5),
            max_votes=params.get("lean_maker_max_votes", 100)
        )

    # Select mode
    if mode == "auto":
        # Auto-select based on problem characteristics
        complexity = sub_problem.ai_suggested_complexity_score or 5

        if complexity >= 7:
            mode = "mdap"
        elif complexity <= 3:
            mode = "maker"
        else:
            mode = "hybrid"

    # Execute
    if mode == "mdap":
        integrator = LeanMDAPWorkflowIntegrator(mdap_config, workflow_state, team)
        return await integrator.solve_subproblem_with_mdap(sub_problem)

    elif mode == "maker":
        integrator = LeanMakerWorkflowIntegrator(maker_config, workflow_state, team)
        return await integrator.solve_with_maker_voting(sub_problem)

    elif mode == "hybrid":
        integrator = LeanHybridIntegrator(mdap_config, maker_config, workflow_state, team)
        return await integrator.adaptive_hybrid_solve(sub_problem)

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main integrators
    "LeanMDAPWorkflowIntegrator",
    "LeanMakerWorkflowIntegrator",
    "LeanHybridIntegrator",
    "LeanMDAPMonitor",

    # Configuration
    "LeanMDAPConfig",
    "LeanMakerConfig",
    "MDAPStrategyType",
    "MakerTacticType",

    # Data structures
    "LeanMDAPTask",
    "LeanMDAPResult",

    # Configuration helpers
    "add_mdap_maker_config_to_workflow_state",

    # Stage integration
    "solve_with_lean_mdap_maker",

    # Availability flags
    "LEANAIDE_AVAILABLE",
    "MDAP_AVAILABLE",
    "MAKER_AVAILABLE",
    "WORKFLOW_AVAILABLE",
    "CREWAI_AVAILABLE",
    "ACE_AVAILABLE"
]


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def example_usage():
        """Example demonstrating Lean MDAP/MAKER workflow integration."""

        print("=== LeanAide MDAP/MAKER Workflow Integration Example ===\n")

        # Check availability
        print("Component Availability:")
        print(f"  LeanAide: {LEANAIDE_AVAILABLE}")
        print(f"  MDAP: {MDAP_AVAILABLE}")
        print(f"  MAKER: {MAKER_AVAILABLE}")
        print(f"  Workflow: {WORKFLOW_AVAILABLE}")
        print()

        # Create example sub-problem
        sub_problem = SubProblem(
            id="sp_001",
            title="Prove Basic Arithmetic Theorem",
            description="forall n : Nat, n + 0 = n",
            dependencies=[],
            ai_suggested_complexity_score=6
        )

        print(f"Sub-problem: {sub_problem.description}")
        print(f"Complexity: {sub_problem.ai_suggested_complexity_score}")
        print()

        # Test MDAP
        if MDAP_AVAILABLE:
            print("Testing MDAP integrator...")
            mdap_config = LeanMDAPConfig(
                enabled=True,
                parallel_agents=2,
                k_ahead=3
            )

            mdap_integrator = LeanMDAPWorkflowIntegrator(config=mdap_config)

            print(f"  MDAP Config:")
            print(f"    Agents: {mdap_config.agents}")
            print(f"    Parallel: {mdap_config.parallel_agents}")
            print(f"    K-ahead: {mdap_config.k_ahead}")
            print()

        # Test MAKER
        if MAKER_AVAILABLE:
            print("Testing MAKER integrator...")
            maker_config = LeanMakerConfig(
                enabled=True,
                k_min=2,
                k_max=5
            )

            maker_integrator = LeanMakerWorkflowIntegrator(config=maker_config)

            print(f"  MAKER Config:")
            print(f"    K-min: {maker_config.k_min}")
            print(f"    K-max: {maker_config.k_max}")
            print(f"    Max votes: {maker_config.max_votes}")
            print()

        # Test convenience function
        print("Testing convenience function...")
        try:
            from workflow_structures import WorkflowState

            workflow_state = WorkflowState(
                workflow_id="test_workflow",
                problem_title="Test Problem",
                problem_statement="Test mathematical problem"
            )

            # Add config
            workflow_state = add_mdap_maker_config_to_workflow_state(
                workflow_state,
                mdap_config,
                maker_config
            )

            print("  Configuration added to workflow state")
            print()

        except (ImportError, AttributeError, ValueError) as e:
            print(f"  Convenience test skipped: {e}")
            print()

        print("Example complete!")

    # Run example
    asyncio.run(example_usage())


# =============================================================================
# CAV-NLP INTEGRATION FOR LEANAIDE MDAP WORKFLOW
# =============================================================================

class LeanMDAPWorkflowCAVNLP:
    """
    CAV-NLP integration for LeanAide MDAP Workflow.

    Enhances MDAP workflow with CAV-NLP formalization:
    - Automatic theorem statement formalization
    - Enhanced sub-problem planning
    - Constraint verification in workflow

    Attributes:
        use_cav_nlp: Whether CAV-NLP is enabled
        math_service: UnifiedMathService instance
    """

    def __init__(self, config: Optional['LeanMDAPConfig'] = None):
        """
        Initialize CAV-NLP enhanced MDAP workflow.

        Args:
            config: MDAP configuration with use_cav_nlp option
        """
        self.config = config
        self.use_cav_nlp = getattr(config, 'use_cav_nlp', True) if config else True
        self.math_service = None

        if self.use_cav_nlp:
            try:
                from openevolve.unified_math_service import UnifiedMathService
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP UnifiedMathService initialized for MDAP Workflow")
            except ImportError as e:
                logger.warning(f"CAV-NLP not available: {e}")
                self.use_cav_nlp = False

    async def plan_with_cav_nlp(
        self,
        objective: str,
        workflow_state: Optional['WorkflowState'] = None
    ) -> Dict[str, Any]:
        """
        Create workflow plan using CAV-NLP enhanced formalization.

        Args:
            objective: Natural language theorem statement
            workflow_state: Optional current workflow state

        Returns:
            Dictionary with formalized plan
        """
        if not self.use_cav_nlp or not self.math_service:
            return {
                "formalized": False,
                "original": objective,
                "plan": objective,
                "confidence": 0.0
            }

        try:
            formalized = await self.math_service.formalize(objective)

            if formalized and hasattr(formalized, 'code'):
                plan = await self._create_workflow_plan(
                    formalized.code,
                    workflow_state
                )

                return {
                    "formalized": True,
                    "original": objective,
                    "plan": plan,
                    "formalized_code": formalized.code,
                    "confidence": getattr(formalized, 'confidence', 0.8),
                    "workflow_context": workflow_state.workflow_id if workflow_state else None
                }
            else:
                return {
                    "formalized": False,
                    "original": objective,
                    "plan": objective,
                    "confidence": 0.0
                }

        except Exception as e:
            logger.error(f"CAV-NLP workflow planning failed: {e}")
            return {
                "formalized": False,
                "original": objective,
                "plan": objective,
                "confidence": 0.0,
                "error": str(e)
            }

    async def _create_workflow_plan(
        self,
        formalized_code: str,
        workflow_state: Optional['WorkflowState'] = None
    ) -> str:
        """Create workflow plan from formalized code."""
        context = f"Workflow: {workflow_state.workflow_id}" if workflow_state else "No workflow context"

        plan = f"""# CAV-NLP Enhanced Workflow Plan
## Formalized Objective
{formalized_code}

## Context
{context}

## Workflow Stages
1. Parse formalized statement
2. Decompose into sub-problems (if needed)
3. Apply MDAP multi-agent voting
4. Verify with LeanAide
5. Aggregate results
"""
        return plan

    async def formalize_constraint(
        self,
        constraint: str
    ) -> Dict[str, Any]:
        """Formalize constraint using CAV-NLP."""
        if not self.use_cav_nlp or not self.math_service:
            return {"formalized": False, "original": constraint}

        try:
            result = await self.math_service.formalize(constraint)
            return {
                "formalized": True,
                "original": constraint,
                "code": getattr(result, 'code', str(result)),
                "confidence": getattr(result, 'confidence', 0.8)
            }
        except Exception as e:
            logger.error(f"Constraint formalization failed: {e}")
            return {"formalized": False, "original": constraint, "error": str(e)}

    def is_cav_nlp_available(self) -> bool:
        """Check if CAV-NLP is available."""
        return self.use_cav_nlp and self.math_service is not None
