"""
LeanAide MCTS Workflow Integration

This module provides comprehensive integration of Monte Carlo Tree Search (MCTS)
capabilities with the OpenEvolve decomposition workflow. It seamlessly adds
MCTS-based proof search, refinement, and verification to existing workflow stages.

Key Features:
- MCTSWorkflowIntegrator: Main integration class for MCTS in workflow
- MCTSSubProblemSolver: Specialized solver using MCTS for mathematical sub-problems
- MCTSProofRefiner: Refines existing proofs using MCTS
- MCTSWorkflowMonitor: Real-time monitoring of MCTS progress
- Stage 3A/B/C integration: MCTS proof generation, refinement, and verification
- Stage 5 integration: Final MCTS verification for hard cases
- Hybrid strategies: MCTS + Evolution, MCTS + Adversarial, MCTS + Self-Play
- Configuration integration with WorkflowState
- Graceful fallback to non-MCTS approaches
- Comprehensive error handling and logging

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Set
)
import threading
import math

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
        LeanProof,
        LeanProofStatus,
        MathematicalDomain
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
        MathematicalProblemDetector,
        is_leanaide_configured
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide workflow integration not available")

# Try to import MCTS implementation
try:
    from leanaide_mcts import (
        MCTS,
        MCTSConfig,
        MCTSNode,
        MCTSResult,
        MCTSRolloutPolicy,
        MCTSSelectionPolicy,
        MCTSBackpropagation
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logger.warning("LeanAide MCTS not available - using stub implementation")

# Import evolutionary components for hybrid strategies
try:
    from leanaide_evolutionary_workflow import (
        LeanEvolutionaryWorkflowStage,
        EvolutionaryConfig,
        EvolutionStrategy
    )
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False
    logger.warning("LeanAide evolutionary workflow not available")

# import crewai # MIGRATED: was CrewAI integration
try:
    from crewai_client import CrewAIClient
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import ACE knowledge storage
try:
    from ace_knowledge_artifacts import ACEKnowledgeManager
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


class MCTSStrategy(Enum):
    """Available MCTS strategies for proof search."""
    STANDARD = "standard"  # Basic MCTS
    UCT = "uct"  # UCB1 for Trees
    THOMPSON_SAMPLING = "thompson"  # Thompson sampling
    HYBRID_EVOLUTION = "hybrid_evolution"  # MCTS + Genetic Algorithm
    HYBRID_ADVERSARIAL = "hybrid_adversarial"  # MCTS + Adversarial
    ADAPTIVE = "adaptive"  # Adaptive strategy selection


class MCTSRolloutType(Enum):
    """Rollout policies for MCTS simulation phase."""
    RANDOM = "random"  # Random tactic selection
    HEURISTIC = "heuristic"  # Heuristic-guided selection
    NEURAL = "neural"  # Neural network policy
    LEANAIDE = "leanaide"  # LeanAide-guided rollout
    SIMULATION = "simulation"  # Fast simulation without verification


@dataclass
class MCTSWorkflowConfig:
    """Configuration for MCTS workflow integration."""
    # MCTS enablement
    lean_mcts_enabled: bool = True
    lean_mcts_strategy: MCTSStrategy = MCTSStrategy.ADAPTIVE

    # MCTS core parameters
    lean_mcts_iterations: int = 1000
    lean_mcts_time_budget: float = 300.0  # seconds
    lean_mcts_c_param: float = 1.414  # UCT exploration parameter
    lean_mcts_rollout_policy: MCTSRolloutType = MCTSRolloutType.LEANAIDE
    lean_mcts_parallel_simulations: int = 4

    # MCTS tree parameters
    lean_mcts_max_depth: int = 100
    lean_mcts_min_visits: int = 5
    lean_mcts_expansion_threshold: int = 10

    # Proof refinement parameters
    lean_mcts_refinement_iterations: int = 100
    lean_mcts_refinement_depth: int = 20

    # Hybrid strategy parameters
    lean_mcts_evolution_generations: int = 20
    lean_mcts_evolution_population: int = 15
    lean_mcts_adversarial_rounds: int = 5

    # Verification parameters
    lean_mcts_verification_confidence: float = 0.8
    lean_mcts_timeout: float = 600.0

    # Fallback behavior
    lean_mcts_fallback_to_evolution: bool = True
    lean_mcts_fallback_to_standard: bool = True
    lean_mcts_partial_result_on_timeout: bool = True

    # Integration settings
    lean_mcts_auto_detect_applicable: bool = True
    lean_mcts_store_patterns: bool = True
    lean_mcts_track_statistics: bool = True

    # CrewAI integration
    CrewAI_enabled: bool = False
    CrewAI_timeout: float = 600.0

    # ACE integration
    ace_learning_enabled: bool = True
    ace_store_mcts_patterns: bool = True


@dataclass
class MCTSProgress:
    """Progress tracking for MCTS runs."""
    sub_problem_id: str
    strategy: MCTSStrategy
    iterations: int = 0
    best_score: float = 0.0
    current_best_proof: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    status: str = "in_progress"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    tree_size: int = 0
    nodes_explored: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_problem_id": self.sub_problem_id,
            "strategy": self.strategy.value,
            "iterations": self.iterations,
            "best_score": self.best_score,
            "current_best_proof": self.current_best_proof,
            "elapsed_time": self.elapsed_time,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "score_history": self.score_history,
            "tree_size": self.tree_size,
            "nodes_explored": self.nodes_explored
        }


@dataclass
class MCTSSearchSpace:
    """Characteristics of the search space for MCTS applicability."""
    branching_factor: int = 0
    estimated_depth: int = 0
    has_heuristics: bool = False
    tactic_diversity: float = 0.0
    complexity_score: float = 0.0
    is_applicable: bool = False

    def calculate_applicability_score(self) -> float:
        """Calculate if MCTS is applicable (0.0 to 1.0)."""
        score = 0.0

        # Large branching factor favors MCTS
        if self.branching_factor > 10:
            score += 0.3
        elif self.branching_factor > 5:
            score += 0.15

        # Deep proofs favor MCTS
        if self.estimated_depth > 20:
            score += 0.3
        elif self.estimated_depth > 10:
            score += 0.15

        # Heuristics improve MCTS effectiveness
        if self.has_heuristics:
            score += 0.2

        # High tactic diversity favors MCTS
        score += min(self.tactic_diversity * 0.2, 0.2)

        return min(score, 1.0)


# =============================================================================
# MCTS Workflow Monitor
# =============================================================================

class MCTSWorkflowMonitor:
    """
    Monitors MCTS progress during workflow execution.

    Provides real-time statistics, progress updates, and early termination logic.
    """

    def __init__(self, config: MCTSWorkflowConfig):
        """
        Initialize the MCTS workflow monitor.

        Args:
            config: MCTS workflow configuration
        """
        self.config = config
        self.active_mcts: Dict[str, MCTS] = {}
        self.progress: Dict[str, MCTSProgress] = {}
        self.statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self.lock = threading.Lock()

    def start_monitoring(self, mcts: MCTS, sub_problem_id: str) -> None:
        """
        Start monitoring an MCTS instance.

        Args:
            mcts: The MCTS instance to monitor
            sub_problem_id: Associated sub-problem ID
        """
        with self.lock:
            self.active_mcts[sub_problem_id] = mcts
            self.progress[sub_problem_id] = MCTSProgress(
                sub_problem_id=sub_problem_id,
                strategy=MCTSStrategy.STANDARD  # Will be updated
            )

    def update_progress(
        self,
        sub_problem_id: str,
        iteration: int,
        best_score: float,
        current_best_proof: Optional[str],
        tree_size: int,
        nodes_explored: int
    ) -> None:
        """
        Update progress for a monitored MCTS.

        Args:
            sub_problem_id: Sub-problem ID
            iteration: Current iteration
            best_score: Best score so far
            current_best_proof: Current best proof
            tree_size: Current tree size
            nodes_explored: Number of nodes explored
        """
        with self.lock:
            if sub_problem_id in self.progress:
                progress = self.progress[sub_problem_id]
                progress.iterations = iteration
                progress.best_score = best_score
                progress.current_best_proof = current_best_proof
                progress.elapsed_time = time.time() - progress.start_time
                progress.tree_size = tree_size
                progress.nodes_explored = nodes_explored
                progress.score_history.append(best_score)

                # Store statistics
                self.statistics[sub_problem_id]["iterations"].append(iteration)
                self.statistics[sub_problem_id]["scores"].append(best_score)
                self.statistics[sub_problem_id]["tree_sizes"].append(tree_size)

    def get_progress(self, sub_problem_id: str) -> Optional[Dict[str, Any]]:
        """
        Get progress for a specific sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            Progress dictionary or None
        """
        with self.lock:
            if sub_problem_id in self.progress:
                return self.progress[sub_problem_id].to_dict()
            return None

    def get_statistics(self, sub_problem_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            Statistics dictionary
        """
        with self.lock:
            if sub_problem_id in self.statistics:
                stats = dict(self.statistics[sub_problem_id])
                if stats.get("scores"):
                    stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
                    stats["max_score"] = max(stats["scores"])
                    stats["score_variance"] = (
                        sum((s - stats["avg_score"]) ** 2 for s in stats["scores"]) /
                        len(stats["scores"])
                    )
                return stats
            return {}

    def should_early_terminate(self, sub_problem_id: str) -> bool:
        """
        Check if MCTS should terminate early.

        Checks for:
        - Convergence (score plateau)
        - Timeout
        - Maximum iterations
        - Satisfactory solution found

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            True if should terminate early
        """
        with self.lock:
            if sub_problem_id not in self.progress:
                return False

            progress = self.progress[sub_problem_id]

            # Check timeout
            if progress.elapsed_time > self.config.lean_mcts_time_budget:
                return True

            # Check max iterations
            if progress.iterations >= self.config.lean_mcts_iterations:
                return True

            # Check for convergence (score plateau)
            if len(progress.score_history) > 20:
                recent_scores = progress.score_history[-20:]
                variance = sum((s - sum(recent_scores) / len(recent_scores)) ** 2
                             for s in recent_scores) / len(recent_scores)
                if variance < 0.001:  # Very low variance = converged
                    return True

            # Check for satisfactory solution
            if progress.best_score >= self.config.lean_mcts_verification_confidence:
                return True

            return False

    def stop_monitoring(self, sub_problem_id: str) -> None:
        """
        Stop monitoring a sub-problem.

        Args:
            sub_problem_id: Sub-problem ID
        """
        with self.lock:
            self.active_mcts.pop(sub_problem_id, None)
            if sub_problem_id in self.progress:
                self.progress[sub_problem_id].status = "completed"


# =============================================================================
# MCTS Workflow Integrator
# =============================================================================

class MCTSWorkflowIntegrator:
    """
    Main integration class for MCTS in workflow stages.

    This class wraps MCTS capabilities and provides seamless integration
    with workflow stages 3A, 3B, 3C, and 5.
    """

    def __init__(
        self,
        config: Optional[MCTSWorkflowConfig] = None,
        workflow_state: Optional[WorkflowState] = None
    ):
        """
        Initialize the MCTS workflow integrator.

        Args:
            config: MCTS workflow configuration
            workflow_state: Current workflow state
        """
        self.config = config or MCTSWorkflowConfig()
        self.workflow_state = workflow_state

        # Initialize integrators
        self.leanaide_integrator: Optional[LeanAideWorkflowIntegrator] = None
        self.crewai_client: Optional[CrewAIClient] = None
        self.ace_manager: Optional[ACEKnowledgeManager] = None
        self.evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None

        # Monitoring
        self.monitor = MCTSWorkflowMonitor(self.config)

        # Progress tracking
        self.mcts_progress: Dict[str, MCTSProgress] = {}
        self.statistics: Dict[str, Any] = defaultdict(list)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all required components."""
        # Initialize LeanAide integrator
        if LEANAIDE_AVAILABLE:
            leanaide_config = LeanAideWorkflowConfig(
                enabled=self.config.lean_mcts_enabled,
                auto_detect_math=self.config.lean_mcts_auto_detect_applicable,
                fallback_to_standard=self.config.lean_mcts_fallback_to_standard,
                confidence_threshold=self.config.lean_mcts_verification_confidence,
                store_proofs=self.config.lean_mcts_store_patterns
            )
            self.leanaide_integrator = LeanAideWorkflowIntegrator(leanaide_config)

        # Initialize CrewAI client if enabled
        if self.config.CrewAI_enabled and CREWAI_AVAILABLE:
            self.crewai_client = CrewAIClient(timeout=self.config.CrewAI_timeout)

        # Initialize ACE manager if enabled
        if self.config.ace_learning_enabled and ACE_AVAILABLE:
            self.ace_manager = ACEKnowledgeManager()

        # Initialize evolutionary stage for hybrid strategies
        if EVOLUTIONARY_AVAILABLE:
            evolutionary_config = EvolutionaryConfig(
                lean_evolution_enabled=True,
                lean_evolution_strategy=EvolutionStrategy.EVOLUTION,
                lean_evolution_generations=self.config.lean_mcts_evolution_generations,
                lean_evolution_population_size=self.config.lean_mcts_evolution_population
            )
            self.evolutionary_stage = LeanEvolutionaryWorkflowStage(
                config=evolutionary_config,
                workflow_state=self.workflow_state
            )

    def analyze_search_space(
        self,
        sub_problem: SubProblem
    ) -> MCTSSearchSpace:
        """
        Analyze the search space to determine if MCTS is applicable.

        Args:
            sub_problem: The sub-problem to analyze

        Returns:
            MCTSSearchSpace with characteristics
        """
        search_space = MCTSSearchSpace()

        # Estimate branching factor (number of applicable tactics)
        search_space.branching_factor = self._estimate_branching_factor(sub_problem)

        # Estimate proof depth
        search_space.estimated_depth = self._estimate_proof_depth(sub_problem)

        # Check for heuristics
        search_space.has_heuristics = self._has_applicable_heuristics(sub_problem)

        # Calculate tactic diversity
        search_space.tactic_diversity = self._calculate_tactic_diversity(sub_problem)

        # Calculate complexity
        search_space.complexity_score = sub_problem.ai_suggested_complexity_score

        # Determine if applicable
        search_space.is_applicable = (
            search_space.calculate_applicability_score() >= 0.5
        )

        return search_space

    def _estimate_branching_factor(self, sub_problem: SubProblem) -> int:
        """Estimate the branching factor for a sub-problem."""
        # Base estimate from complexity
        base_factor = min(sub_problem.ai_suggested_complexity_score * 2, 50)

        # Look for keywords that indicate many tactics
        description = sub_problem.description.lower()

        if any(kw in description for kw in ["induction", "recursive", "structure"]):
            base_factor *= 1.5
        if any(kw in description for kw in ["algebra", "calculation", "simplification"]):
            base_factor *= 1.2

        return int(base_factor)

    def _estimate_proof_depth(self, sub_problem: SubProblem) -> int:
        """Estimate the proof depth for a sub-problem."""
        # Base estimate from complexity
        base_depth = sub_problem.ai_suggested_complexity_score * 3

        # Look for depth indicators
        description = sub_problem.description.lower()

        if any(kw in description for kw in ["multiple", "several", "nested"]):
            base_depth *= 1.5
        if any(kw in description for kw in "elementary basic simple".split()):
            base_depth *= 0.7

        return int(base_depth)

    def _has_applicable_heuristics(self, sub_problem: SubProblem) -> bool:
        """Check if there are applicable heuristics for the sub-problem."""
        description = sub_problem.description.lower()

        # Domain-specific heuristics
        heuristic_domains = [
            "algebra", "geometry", "number theory", "combinatorics",
            "calculus", "analysis", "topology"
        ]

        return any(domain in description for domain in heuristic_domains)

    def _calculate_tactic_diversity(self, sub_problem: SubProblem) -> float:
        """Calculate tactic diversity (0.0 to 1.0)."""
        description = sub_problem.description.lower()

        # Tactic categories
        tactic_keywords = {
            "logic": ["prove", "show", "implies", "iff", "contradiction"],
            "algebra": ["simplify", "rewrite", "calculate", "linear"],
            "induction": ["induction", "recursive", "base case"],
            "geometry": ["congruent", "similar", "angle", "triangle"],
            "analysis": ["limit", "continuity", "derivative", "integral"]
        }

        categories_found = sum(
            1 for keywords in tactic_keywords.values()
            if any(kw in description for kw in keywords)
        )

        return min(categories_found / len(tactic_keywords), 1.0)

    async def solve_with_mcts(
        self,
        sub_problem: SubProblem,
        config: Optional[MCTSWorkflowConfig] = None
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using MCTS.

        This is the main integration point for Stage 3A (Solution Generation)
        with MCTS capabilities.

        Args:
            sub_problem: The sub-problem to solve
            config: Optional configuration override

        Returns:
            SolutionAttempt with MCTS-generated solution
        """
        start_time = time.time()
        sub_problem_id = sub_problem.id
        config = config or self.config

        # Create progress tracker
        progress = MCTSProgress(
            sub_problem_id=sub_problem_id,
            strategy=config.lean_mcts_strategy
        )
        self.mcts_progress[sub_problem_id] = progress

        try:
            # Analyze search space
            search_space = self.analyze_search_space(sub_problem)

            if not search_space.is_applicable and not config.lean_mcts_enabled:
                logger.info(f"MCTS not applicable for {sub_problem_id}, using standard approach")
                return await self._fallback_to_evolution(sub_problem, config)

            logger.info(f"Solving {sub_problem_id} with MCTS (applicability: {search_space.calculate_applicability_score():.2f})")

            # Select strategy
            if config.lean_mcts_strategy == MCTSStrategy.HYBRID_EVOLUTION:
                solution = await self._solve_with_mcts_evolution_hybrid(
                    sub_problem, search_space, config, progress
                )
            elif config.lean_mcts_strategy == MCTSStrategy.HYBRID_ADVERSARIAL:
                solution = await self._solve_with_mcts_adversarial_hybrid(
                    sub_problem, search_space, config, progress
                )
            elif config.lean_mcts_strategy == MCTSStrategy.ADAPTIVE:
                solution = await self._solve_with_adaptive_mcts(
                    sub_problem, search_space, config, progress
                )
            else:
                solution = await self._solve_with_pure_mcts(
                    sub_problem, search_space, config, progress
                )

            # Update progress
            progress.status = "completed"
            progress.elapsed_time = time.time() - start_time

            # Store statistics
            self.statistics[sub_problem_id].append({
                "strategy": config.lean_mcts_strategy.value,
                "elapsed_time": progress.elapsed_time,
                "iterations": progress.iterations,
                "best_score": progress.best_score,
                "applicability_score": search_space.calculate_applicability_score()
            })

            # Store in knowledge base if enabled
            if config.lean_mcts_store_patterns and self.ace_manager:
                await self._store_mcts_pattern(sub_problem, solution, progress, search_space)

            return solution

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MCTS solution failed for {sub_problem_id}: {e}")
            progress.errors.append(str(e))
            progress.status = "failed"

            if config.lean_mcts_fallback_to_evolution:
                return await self._fallback_to_evolution(sub_problem, config)
            elif config.lean_mcts_fallback_to_standard:
                return await self._fallback_to_standard_solution(sub_problem)
            else:
                raise

    async def _solve_with_pure_mcts(
        self,
        sub_problem: SubProblem,
        search_space: MCTSSearchSpace,
        config: MCTSWorkflowConfig,
        progress: MCTSProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using pure MCTS."""
        if not MCTS_AVAILABLE:
            logger.warning("MCTS not available, falling back to evolution")
            return await self._fallback_to_evolution(sub_problem, config)

        logger.info(f"Using pure MCTS for {sub_problem.id}")

        # Create MCTS configuration
        mcts_config = MCTSConfig(
            iterations=config.lean_mcts_iterations,
            time_budget=config.lean_mcts_time_budget,
            c_param=config.lean_mcts_c_param,
            rollout_policy=self._map_rollout_type(config.lean_mcts_rollout_policy),
            max_depth=config.lean_mcts_max_depth,
            parallel_simulations=config.lean_mcts_parallel_simulations
        )

        # Create and run MCTS
        mcts = MCTS(config=mcts_config)

        # Start monitoring
        self.monitor.start_monitoring(mcts, sub_problem.id)

        # Run MCTS search
        result = await mcts.search(
            theorem=sub_problem.description,
            theorem_name=f"theorem_{sub_problem.id.replace('-', '_')}"
        )

        # Stop monitoring
        self.monitor.stop_monitoring(sub_problem.id)

        # Update progress
        progress.iterations = result.iterations
        progress.best_score = result.best_score
        progress.current_best_proof = result.best_proof
        progress.tree_size = result.tree_size
        progress.nodes_explored = result.nodes_explored

        # Create solution attempt
        solution = SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=result.best_proof or "",
            generated_by_model="LeanAide-MCTS",
            timestamp=time.time(),
            status="verified" if result.best_score >= config.lean_mcts_verification_confidence else "generated",
            solution_approach="mcts_pure",
            openevolve_metrics={
                "mcts_strategy": "pure",
                "iterations": result.iterations,
                "tree_size": result.tree_size,
                "nodes_explored": result.nodes_explored,
                "best_score": result.best_score,
                "search_time": result.search_time,
                "branching_factor": search_space.branching_factor
            }
        )

        return solution

    async def _solve_with_mcts_evolution_hybrid(
        self,
        sub_problem: SubProblem,
        search_space: MCTSSearchSpace,
        config: MCTSWorkflowConfig,
        progress: MCTSProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using MCTS + Evolution hybrid."""
        logger.info(f"Using MCTS + Evolution hybrid for {sub_problem.id}")

        # First, use MCTS to explore and find promising regions
        mcts_solution = await self._solve_with_pure_mcts(
            sub_problem, search_space, config, progress
        )

        # Then use evolutionary approach to refine
        if self.evolutionary_stage and EVOLUTIONARY_AVAILABLE:
            logger.info(f"Refining {sub_problem.id} with evolution after MCTS")

            evolved_solution = await self.evolutionary_stage.evolve_solution_stage3a(
                mcts_solution,
                self.workflow_state or self._create_default_workflow_state()
            )

            # Combine metrics
            evolved_solution.openevolve_metrics = evolved_solution.openevolve_metrics or {}
            evolved_solution.openevolve_metrics.update({
                "hybrid_strategy": "mcts_evolution",
                "mcts_iterations": progress.iterations,
                "mcts_score": progress.best_score
            })

            evolved_solution.solution_approach = "mcts_evolution_hybrid"

            return evolved_solution

        return mcts_solution

    async def _solve_with_mcts_adversarial_hybrid(
        self,
        sub_problem: SubProblem,
        search_space: MCTSSearchSpace,
        config: MCTSWorkflowConfig,
        progress: MCTSProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using MCTS + Adversarial hybrid."""
        logger.info(f"Using MCTS + Adversarial hybrid for {sub_problem.id}")

        # Get initial MCTS solution
        mcts_solution = await self._solve_with_pure_mcts(
            sub_problem, search_space, config, progress
        )

        # Then use adversarial evolution for robustness
        if self.evolutionary_stage and EVOLUTIONARY_AVAILABLE:
            logger.info(f"Refining {sub_problem.id} with adversarial after MCTS")

            adversarial_solution = await self.evolutionary_stage.adversarial_evolution_stage3b(
                mcts_solution,
                self.workflow_state or self._create_default_workflow_state()
            )

            # Combine metrics
            adversarial_solution.openevolve_metrics = adversarial_solution.openevolve_metrics or {}
            adversarial_solution.openevolve_metrics.update({
                "hybrid_strategy": "mcts_adversarial",
                "mcts_iterations": progress.iterations,
                "mcts_score": progress.best_score
            })

            adversarial_solution.solution_approach = "mcts_adversarial_hybrid"

            return adversarial_solution

        return mcts_solution

    async def _solve_with_adaptive_mcts(
        self,
        sub_problem: SubProblem,
        search_space: MCTSSearchSpace,
        config: MCTSWorkflowConfig,
        progress: MCTSProgress
    ) -> SolutionAttempt:
        """Solve sub-problem using adaptive strategy selection."""
        logger.info(f"Using adaptive MCTS for {sub_problem.id}")

        applicability_score = search_space.calculate_applicability_score()

        # If highly applicable, use pure MCTS
        if applicability_score >= 0.8:
            logger.info(f"High applicability ({applicability_score:.2f}), using pure MCTS")
            return await self._solve_with_pure_mcts(sub_problem, search_space, config, progress)

        # If moderately applicable, use MCTS + Evolution hybrid
        elif applicability_score >= 0.5:
            logger.info(f"Medium applicability ({applicability_score:.2f}), using MCTS + Evolution")
            return await self._solve_with_mcts_evolution_hybrid(
                sub_problem, search_space, config, progress
            )

        # If low applicability, fall back to evolution
        else:
            logger.info(f"Low applicability ({applicability_score:.2f}), falling back to evolution")
            return await self._fallback_to_evolution(sub_problem, config)

    async def _fallback_to_evolution(
        self,
        sub_problem: SubProblem,
        config: MCTSWorkflowConfig
    ) -> SolutionAttempt:
        """Fallback to evolutionary approach."""
        if not EVOLUTIONARY_AVAILABLE or not self.evolutionary_stage:
            return await self._fallback_to_standard_solution(sub_problem)

        logger.info(f"Falling back to evolutionary approach for {sub_problem.id}")

        return await self.evolutionary_stage.solve_subproblem_evolutionary(
            sub_problem,
            self.workflow_state or self._create_default_workflow_state()
        )

    async def _fallback_to_standard_solution(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Fallback to standard non-MCTS solution."""
        logger.info(f"Using standard approach for {sub_problem.id}")

        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=f"# Standard solution for {sub_problem.description}",
            generated_by_model="Standard",
            timestamp=time.time(),
            status="generated",
            solution_approach="standard"
        )

    def _map_rollout_type(self, rollout_type: MCTSRolloutType) -> str:
        """Map rollout type to MCTS implementation."""
        mapping = {
            MCTSRolloutType.RANDOM: "random",
            MCTSRolloutType.HEURISTIC: "heuristic",
            MCTSRolloutType.NEURAL: "neural",
            MCTSRolloutType.LEANAIDE: "leanaide",
            MCTSRolloutType.SIMULATION: "simulation"
        }
        return mapping.get(rollout_type, "heuristic")

    def _create_default_workflow_state(self) -> WorkflowState:
        """Create a default workflow state if none exists."""
        if WORKFLOW_AVAILABLE:
            return WorkflowState(
                workflow_id=str(uuid.uuid4()),
                problem_statement="MCTS Proof Search",
                analyzed_context={}
            )
        return None

    async def mcts_stage3a(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Stage 3A integration: Solve sub-problem using MCTS.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with MCTS-generated solution
        """
        logger.info(f"Stage 3A: Solving {sub_problem.id} with MCTS")

        return await self.solve_with_mcts(sub_problem)

    async def mcts_stage3b(
        self,
        solution: SolutionAttempt
    ) -> SolutionAttempt:
        """
        Stage 3B integration: Refine solution using MCTS.

        Uses the existing proof as prior knowledge to guide MCTS search.

        Args:
            solution: Current solution attempt to refine

        Returns:
            Refined solution attempt
        """
        logger.info(f"Stage 3B: Refining {solution.sub_problem_id} with MCTS")

        if not MCTS_AVAILABLE:
            return solution

        # Get sub-problem
        sub_problem = None
        if self.workflow_state and self.workflow_state.decomposition_plan:
            for sp in self.workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            logger.warning(f"Sub-problem {solution.sub_problem_id} not found")
            return solution

        # Create refiner
        refiner = MCTSProofRefiner(self.config)

        # Create LeanProof from solution
        proof = LeanProof(
            proof_id=f"proof_{solution.sub_problem_id}",
            theorem_name=f"theorem_{solution.sub_problem_id}",
            lean_code=solution.content,
            natural_language_statement=sub_problem.description
        )

        # Refine proof
        refined_proof = await refiner.refine_proof(
            proof,
            iterations=self.config.lean_mcts_refinement_iterations
        )

        # Update solution
        if refined_proof.lean_code != solution.content:
            solution.content = refined_proof.lean_code
            solution.status = "verified" if refined_proof.proof_status == LeanProofStatus.VERIFIED else "refined"
            solution.solution_approach = "mcts_refinement_stage3b"

            if solution.openevolve_metrics is None:
                solution.openevolve_metrics = {}

            solution.openevolve_metrics.update({
                "stage3b_mcts_refinement": True,
                "refinement_iterations": self.config.lean_mcts_refinement_iterations,
                "refinement_score": refined_proof.complexity_score
            })

        return solution

    async def _store_mcts_pattern(
        self,
        sub_problem: SubProblem,
        solution: SolutionAttempt,
        progress: MCTSProgress,
        search_space: MCTSSearchSpace
    ):
        """Store MCTS pattern in knowledge base."""
        if not self.ace_manager:
            return

        try:
            artifact = {
                "type": "mcts_pattern",
                "sub_problem_id": sub_problem.id,
                "strategy": progress.strategy.value,
                "iterations": progress.iterations,
                "best_score": progress.best_score,
                "applicability_score": search_space.calculate_applicability_score(),
                "branching_factor": search_space.branching_factor,
                "estimated_depth": search_space.estimated_depth,
                "proof": solution.content,
                "timestamp": time.time()
            }

            self.ace_manager.store_artifact(artifact)
            logger.info(f"Stored MCTS pattern for {sub_problem.id} in knowledge base")
        except (IOError, AttributeError, KeyError, ValueError) as e:
            logger.error(f"Failed to store MCTS pattern: {e}")

    def configure_mcts_from_workflow(self, state: WorkflowState) -> MCTSWorkflowConfig:
        """
        Configure MCTS from workflow state.

        Args:
            state: Current workflow state

        Returns:
            MCTSWorkflowConfig
        """
        params = state.openevolve_parameters or {}

        return MCTSWorkflowConfig(
            lean_mcts_enabled=params.get("lean_mcts_enabled", True),
            lean_mcts_strategy=MCTSStrategy(
                params.get("lean_mcts_strategy", "adaptive")
            ),
            lean_mcts_iterations=params.get("lean_mcts_iterations", 1000),
            lean_mcts_time_budget=params.get("lean_mcts_time_budget", 300.0),
            lean_mcts_c_param=params.get("lean_mcts_c_param", 1.414),
            lean_mcts_parallel_simulations=params.get("lean_mcts_parallel_simulations", 4)
        )


# =============================================================================
# MCTS Sub-Problem Solver
# =============================================================================

class MCTSSubProblemSolver:
    """
    Solves mathematical sub-problems using MCTS.

    Detects when MCTS is appropriate based on search space characteristics
    and integrates seamlessly with SubProblem structure.
    """

    def __init__(
        self,
        integrator: MCTSWorkflowIntegrator,
        config: Optional[MCTSWorkflowConfig] = None
    ):
        """
        Initialize the MCTS sub-problem solver.

        Args:
            integrator: MCTS workflow integrator
            config: Optional configuration override
        """
        self.integrator = integrator
        self.config = config or integrator.config
        self.solved_problems: Dict[str, Dict[str, Any]] = {}

    async def can_solve_with_mcts(self, sub_problem: SubProblem) -> bool:
        """
        Determine if MCTS is appropriate for this sub-problem.

        Args:
            sub_problem: The sub-problem to evaluate

        Returns:
            True if MCTS is appropriate
        """
        search_space = self.integrator.analyze_search_space(sub_problem)

        # Check applicability score
        if not search_space.is_applicable:
            return False

        # Additional checks
        # Must have sufficient branching factor
        if search_space.branching_factor < 3:
            return False

        # Must have reasonable depth
        if search_space.estimated_depth < 5:
            return False

        return True

    async def solve_with_mcts(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using MCTS.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with MCTS-generated solution
        """
        # Check if already solved
        if sub_problem.id in self.solved_problems:
            logger.info(f"Sub-problem {sub_problem.id} already solved, returning cached solution")
            return self.solved_problems[sub_problem.id]["solution"]

        # Check if MCTS is applicable
        can_use_mcts = await self.can_solve_with_mcts(sub_problem)

        if not can_use_mcts:
            logger.info(f"MCTS not applicable for {sub_problem.id}, using alternative")
            return await self._solve_alternative(sub_problem)

        # Solve using MCTS
        logger.info(f"Solving {sub_problem.id} with MCTS")

        solution = await self.integrator.solve_with_mcts(sub_problem)

        # Add metadata
        solution.metadata = {
            "mcts": True,
            "solved_by": "MCTSSubProblemSolver",
            "search_space_analyzed": True
        }

        # Cache solution
        self.solved_problems[sub_problem.id] = {
            "solution": solution,
            "timestamp": time.time()
        }

        return solution

    def extract_proof_from_mcts(self, result: MCTSResult) -> LeanProof:
        """
        Extract a LeanProof from MCTS result.

        Args:
            result: MCTS search result

        Returns:
            LeanProof object
        """
        return LeanProof(
            proof_id=f"mcts_proof_{uuid.uuid4().hex[:8]}",
            theorem_name=result.theorem_name,
            lean_code=result.best_proof or "",
            natural_language_statement=result.theorem_statement,
            proof_status=LeanProofStatus.VERIFIED if result.best_score >= 0.8 else LeanProofStatus.PARTIAL,
            complexity_score=int(result.best_score * 10),
            proof_steps=result.proof_steps if hasattr(result, 'proof_steps') else [],
            metadata={
                "mcts_iterations": result.iterations,
                "mcts_score": result.best_score,
                "mcts_tree_size": result.tree_size
            }
        )

    def create_solution_attempt(
        self,
        proof: LeanProof,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Create a SolutionAttempt from a LeanProof.

        Args:
            proof: The Lean proof
            sub_problem: The associated sub-problem

        Returns:
            SolutionAttempt object
        """
        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=proof.lean_code,
            generated_by_model="LeanAide-MCTS",
            timestamp=time.time(),
            status="verified" if proof.proof_status == LeanProofStatus.VERIFIED else "generated",
            solution_approach="mcts",
            metadata={
                "proof_id": proof.proof_id,
                "complexity_score": proof.complexity_score,
                "mcts": True
            }
        )

    async def _solve_alternative(self, sub_problem: SubProblem) -> SolutionAttempt:
        """Solve using alternative approach."""
        # Try evolution first
        if self.config.lean_mcts_fallback_to_evolution and EVOLUTIONARY_AVAILABLE:
            return await self.integrator._fallback_to_evolution(
                sub_problem, self.config
            )

        # Fall back to standard
        return await self.integrator._fallback_to_standard_solution(sub_problem)


# =============================================================================
# MCTS Proof Refiner
# =============================================================================

class MCTSProofRefiner:
    """
    Refines existing proofs using MCTS.

    Uses existing proof as prior knowledge to initialize MCTS tree,
    then searches for improvements and expansions.
    """

    def __init__(self, config: MCTSWorkflowConfig):
        """
        Initialize the MCTS proof refiner.

        Args:
            config: MCTS workflow configuration
        """
        self.config = config

    async def refine_proof(
        self,
        proof: LeanProof,
        iterations: Optional[int] = None
    ) -> LeanProof:
        """
        Refine a proof using MCTS.

        Args:
            proof: The proof to refine
            iterations: Number of MCTS iterations (overrides config)

        Returns:
            Refined LeanProof
        """
        if not MCTS_AVAILABLE:
            logger.warning("MCTS not available for refinement")
            return proof

        logger.info(f"Refining proof {proof.proof_id} with MCTS")

        # Initialize MCTS tree from proof
        mcts_config = MCTSConfig(
            iterations=iterations or self.config.lean_mcts_refinement_iterations,
            time_budget=self.config.lean_mcts_time_budget,
            c_param=self.config.lean_mcts_c_param,
            rollout_policy=self._map_rollout_type(self.config.lean_mcts_rollout_policy),
            max_depth=self.config.lean_mcts_refinement_depth
        )

        # Create MCTS instance
        mcts = MCTS(config=mcts_config)

        # Initialize tree from existing proof
        tree = await self.initialize_tree_from_proof(proof)

        # Run MCTS refinement
        result = await mcts.refine_proof(
            initial_tree=tree,
            theorem=proof.natural_language_statement,
            theorem_name=proof.theorem_name
        )

        # Update proof
        if result.best_proof and result.best_proof != proof.lean_code:
            proof.lean_code = result.best_proof
            proof.proof_status = LeanProofStatus.VERIFIED if result.best_score >= 0.8 else LeanProofStatus.PARTIAL
            proof.complexity_score = int(result.best_score * 10)
            proof.metadata["refined_with_mcts"] = True
            proof.metadata["refinement_iterations"] = result.iterations
            proof.metadata["refinement_score"] = result.best_score

        return proof

    async def expand_partial_proof(self, proof: LeanProof) -> LeanProof:
        """
        Expand a partial proof using MCTS.

        Args:
            proof: Partial proof to expand

        Returns:
            Expanded LeanProof
        """
        if not MCTS_AVAILABLE:
            return proof

        logger.info(f"Expanding partial proof {proof.proof_id} with MCTS")

        # Increase iterations for expansion
        expanded_proof = await self.refine_proof(
            proof,
            iterations=self.config.lean_mcts_refinement_iterations * 2
        )

        return expanded_proof

    async def initialize_tree_from_proof(self, proof: LeanProof) -> Dict[str, Any]:
        """
        Initialize MCTS tree from existing proof.

        Args:
            proof: Existing proof to initialize from

        Returns:
            Initial MCTS tree structure
        """
        # Parse proof to extract proof steps
        proof_steps = self._extract_proof_steps(proof.lean_code)

        # Build tree structure
        tree = {
            "root": {
                "state": proof.lean_code,
                "visits": 1,
                "value": proof.complexity_score / 10.0,
                "children": []
            },
            "proof_steps": proof_steps,
            "initialized_from": proof.proof_id
        }

        # Add children for each proof step
        current = tree["root"]
        for step in proof_steps:
            child = {
                "state": step,
                "visits": 1,
                "value": 0.5,
                "children": []
            }
            current["children"].append(child)
            current = child

        return tree

    def _extract_proof_steps(self, lean_code: str) -> List[str]:
        """Extract proof steps from Lean code."""
        import re

        # Split by common tactic indicators
        tactics = [
            r'by\s+', r'\.\s+', r'rw\s+', r'apply\s+', r'have\s+',
            r'calc\s+', r'simp\s+', r'linarith\s+', r'ring\s+'
        ]

        steps = [lean_code]

        for tactic in tactics:
            new_steps = []
            for step in steps:
                parts = re.split(tactic, step, maxsplit=1)
                if len(parts) > 1:
                    new_steps.extend(parts)
                else:
                    new_steps.append(step)
            steps = new_steps

        # Clean and filter
        cleaned_steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 3]

        return cleaned_steps[:20]  # Limit to 20 steps

    def _map_rollout_type(self, rollout_type: MCTSRolloutType) -> str:
        """Map rollout type to MCTS implementation."""
        mapping = {
            MCTSRolloutType.RANDOM: "random",
            MCTSRolloutType.HEURISTIC: "heuristic",
            MCTSRolloutType.NEURAL: "neural",
            MCTSRolloutType.LEANAIDE: "leanaide",
            MCTSRolloutType.SIMULATION: "simulation"
        }
        return mapping.get(rollout_type, "heuristic")


# =============================================================================
# Workflow Integration Functions
# =============================================================================

def add_mcts_config_to_workflow_state(
    workflow_state: WorkflowState,
    config: MCTSWorkflowConfig
) -> WorkflowState:
    """
    Add MCTS configuration to workflow state.

    Args:
        workflow_state: Current workflow state
        config: MCTS configuration

    Returns:
        Updated workflow state
    """
    if workflow_state.openevolve_parameters is None:
        workflow_state.openevolve_parameters = {}

    workflow_state.openevolve_parameters.update({
        "lean_mcts_enabled": config.lean_mcts_enabled,
        "lean_mcts_strategy": config.lean_mcts_strategy.value,
        "lean_mcts_iterations": config.lean_mcts_iterations,
        "lean_mcts_time_budget": config.lean_mcts_time_budget,
        "lean_mcts_c_param": config.lean_mcts_c_param,
        "lean_mcts_rollout_policy": config.lean_mcts_rollout_policy.value,
        "lean_mcts_parallel_simulations": config.lean_mcts_parallel_simulations,
        "lean_mcts_refinement_iterations": config.lean_mcts_refinement_iterations,
        "lean_mcts_verification_confidence": config.lean_mcts_verification_confidence,
        "lean_mcts_auto_detect_applicable": config.lean_mcts_auto_detect_applicable,
        "lean_mcts_store_patterns": config.lean_mcts_store_patterns
    })

    return workflow_state


def extract_mcts_config_from_workflow_state(
    workflow_state: WorkflowState
) -> MCTSWorkflowConfig:
    """
    Extract MCTS configuration from workflow state.

    Args:
        workflow_state: Current workflow state

    Returns:
        MCTSWorkflowConfig
    """
    params = workflow_state.openevolve_parameters or {}

    return MCTSWorkflowConfig(
        lean_mcts_enabled=params.get("lean_mcts_enabled", True),
        lean_mcts_strategy=MCTSStrategy(
            params.get("lean_mcts_strategy", "adaptive")
        ),
        lean_mcts_iterations=params.get("lean_mcts_iterations", 1000),
        lean_mcts_time_budget=params.get("lean_mcts_time_budget", 300.0),
        lean_mcts_c_param=params.get("lean_mcts_c_param", 1.414),
        lean_mcts_rollout_policy=MCTSRolloutType(
            params.get("lean_mcts_rollout_policy", "leanaide")
        ),
        lean_mcts_parallel_simulations=params.get("lean_mcts_parallel_simulations", 4),
        lean_mcts_refinement_iterations=params.get(
            "lean_mcts_refinement_iterations", 100
        ),
        lean_mcts_verification_confidence=params.get(
            "lean_mcts_verification_confidence", 0.8
        ),
        lean_mcts_auto_detect_applicable=params.get(
            "lean_mcts_auto_detect_applicable", True
        ),
        lean_mcts_store_patterns=params.get("lean_mcts_store_patterns", True)
    )


async def solve_with_mcts_approach(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    config: Optional[MCTSWorkflowConfig] = None
) -> SolutionAttempt:
    """
    Convenience function to solve a sub-problem with MCTS.

    Args:
        sub_problem: The sub-problem to solve
        workflow_state: Current workflow state
        config: Optional MCTS configuration

    Returns:
        SolutionAttempt with MCTS-generated solution
    """
    # Create integrator
    integrator = MCTSWorkflowIntegrator(
        config=config,
        workflow_state=workflow_state
    )

    # Solve
    return await integrator.solve_with_mcts(sub_problem)


# =============================================================================
# Stub MCTS Implementation (when MCTS not available)
# =============================================================================

if not MCTS_AVAILABLE:

    @dataclass
    class MCTSConfig:
        """Stub MCTS configuration."""
        iterations: int = 1000
        time_budget: float = 300.0
        c_param: float = 1.414
        rollout_policy: str = "heuristic"
        max_depth: int = 100
        parallel_simulations: int = 4

    @dataclass
    class MCTSResult:
        """Stub MCTS result."""
        iterations: int
        best_score: float
        best_proof: Optional[str]
        theorem_name: str
        theorem_statement: str
        tree_size: int
        nodes_explored: int
        search_time: float

    class MCTS:
        """Stub MCTS implementation."""

        def __init__(self, config: MCTSConfig):
            self.config = config

        async def search(self, theorem: str, theorem_name: str) -> MCTSResult:
            """Stub search method."""
            return MCTSResult(
                iterations=self.config.iterations,
                best_score=0.7,
                best_proof=f"-- Proof generated by stub MCTS\ntheorem {theorem_name} : Prop := by\n  sorry",
                theorem_name=theorem_name,
                theorem_statement=theorem,
                tree_size=100,
                nodes_explored=50,
                search_time=10.0
            )

        async def refine_proof(self, initial_tree: Dict[str, Any], theorem: str, theorem_name: str) -> MCTSResult:
            """Stub refine method."""
            return await self.search(theorem, theorem_name)


# =============================================================================
# Stage Integration Functions for workflow_stage_functions.py
# =============================================================================

async def verify_sub_problem_with_leanaide_mcts(
    sub_problem: SubProblem,
    solution_attempt: SolutionAttempt,
    workflow_state: WorkflowState
) -> VerificationReport:
    """
    Verify a sub-problem solution using LeanAide MCTS verification.

    Integrates with workflow_stage_functions.py for Stage 3C.

    Args:
        sub_problem: The sub-problem being verified
        solution_attempt: The solution attempt to verify
        workflow_state: Current workflow state

    Returns:
        VerificationReport with MCTS-enhanced verification results
    """
    import time

    # Extract MCTS config
    config = extract_mcts_config_from_workflow_state(workflow_state)
    integrator = MCTSWorkflowIntegrator(config=config, workflow_state=workflow_state)

    # Check if MCTS was used
    used_mcts = solution_attempt.openevolve_metrics.get("mcts_strategy") if solution_attempt.openevolve_metrics else None

    # Get standard verification
    if LEANAIDE_AVAILABLE and integrator.leanaide_integrator:
        result = await integrator.leanaide_integrator.verify_sub_problem_solution(
            sub_problem_id=sub_problem.id,
            problem_statement=sub_problem.description,
            solution_content=solution_attempt.content,
            verification_requirements=sub_problem.solution_requirements
        )

        dimension_scores = {
            "mathematical_correctness": result.confidence_score,
            "formal_verification": 1.0 if result.success else 0.0,
            "mcts_enhanced": 1.0 if used_mcts else 0.0
        }

        criteria_met = []
        criteria_not_met = []

        if result.is_mathematical:
            if result.success:
                criteria_met.append("Formal verification passed")
            else:
                criteria_not_met.append(f"Formal verification failed (confidence: {result.confidence_score:.2f})")

        summary = f"""MCTS-Enhanced Verification Results:

Mathematical Problem: {result.is_mathematical}
Verification Success: {result.success}
Confidence Score: {result.confidence_score:.2f}
MCTS Used: {used_mcts or "No"}
"""

        return VerificationReport(
            solution_attempt_id=solution_attempt.sub_problem_id,
            gauntlet_name="mcts_enhanced_verification",
            is_approved=result.success or not result.is_mathematical,
            reports_by_judge=[{
                "method": "MCTS-Enhanced Verification",
                "mcts_used": used_mcts,
                "result": result.to_dict()
            }],
            average_score=result.confidence_score if result.is_mathematical else 0.8,
            summary=summary,
            verification_timestamp=time.time(),
            dimension_scores=dimension_scores,
            criteria_met=criteria_met,
            criteria_not_met=criteria_not_met
        )
    else:
        return VerificationReport(
            solution_attempt_id=solution_attempt.sub_problem_id,
            gauntlet_name="standard_verification",
            is_approved=True,
            reports_by_judge=[],
            average_score=0.7,
            summary="LeanAide not available, using standard verification"
        )


# Export main classes and functions
__all__ = [
    # Main classes
    'MCTSWorkflowIntegrator',
    'MCTSSubProblemSolver',
    'MCTSProofRefiner',
    'MCTSWorkflowMonitor',

    # Configuration and data classes
    'MCTSWorkflowConfig',
    'MCTSProgress',
    'MCTSSearchSpace',
    'MCTSStrategy',
    'MCTSRolloutType',

    # Workflow integration functions
    'add_mcts_config_to_workflow_state',
    'extract_mcts_config_from_workflow_state',
    'solve_with_mcts_approach',
    'verify_sub_problem_with_leanaide_mcts',

    # Availability flags
    'MCTS_AVAILABLE',
    'LEANAIDE_AVAILABLE',
    'EVOLUTIONARY_AVAILABLE',
    'WORKFLOW_AVAILABLE'
]


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def example_usage():
        """Example demonstrating the MCTS workflow integration."""

        print("=== LeanAide MCTS Workflow Integration Example ===\n")

        # Create configuration
        config = MCTSWorkflowConfig(
            lean_mcts_enabled=True,
            lean_mcts_strategy=MCTSStrategy.ADAPTIVE,
            lean_mcts_iterations=500,
            lean_mcts_time_budget=120.0,
            lean_mcts_c_param=1.414
        )

        print(f"Configuration:")
        print(f"  Strategy: {config.lean_mcts_strategy.value}")
        print(f"  Iterations: {config.lean_mcts_iterations}")
        print(f"  Time Budget: {config.lean_mcts_time_budget}s")
        print(f"  C Param: {config.lean_mcts_c_param}")
        print()

        # Create integrator
        integrator = MCTSWorkflowIntegrator(config=config)

        # Create example sub-problem
        if WORKFLOW_AVAILABLE:
            sub_problem = SubProblem(
                id="sp_mcts_001",
                description="Prove that for all natural numbers n, n + 0 = n",
                dependencies=[],
                ai_suggested_complexity_score=5
            )

            print(f"Sub-problem: {sub_problem.description}")
            print()

            # Analyze search space
            search_space = integrator.analyze_search_space(sub_problem)
            applicability = search_space.calculate_applicability_score()

            print(f"Search Space Analysis:")
            print(f"  Branching Factor: {search_space.branching_factor}")
            print(f"  Estimated Depth: {search_space.estimated_depth}")
            print(f"  Has Heuristics: {search_space.has_heuristics}")
            print(f"  Tactic Diversity: {search_space.tactic_diversity:.2f}")
            print(f"  Applicability Score: {applicability:.2f}")
            print(f"  Is Applicable: {search_space.is_applicable}")
            print()

            # Check if can solve with MCTS
            solver = MCTSSubProblemSolver(integrator)
            can_solve = await solver.can_solve_with_mcts(sub_problem)
            print(f"Can solve with MCTS: {can_solve}")
            print()

        # Test availability
        print("Component Availability:")
        print(f"  MCTS: {MCTS_AVAILABLE}")
        print(f"  LeanAide: {LEANAIDE_AVAILABLE}")
        print(f"  Evolutionary: {EVOLUTIONARY_AVAILABLE}")
        print(f"  Workflow: {WORKFLOW_AVAILABLE}")
        print()

        print("Example complete!")

    # Run example
    asyncio.run(example_usage())
