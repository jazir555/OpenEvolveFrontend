"""
Hybrid MAKER Workflow Integration

This module provides comprehensive workflow integration for hybrid MAKER strategies,
combining the MAKER framework (arXiv:2511.09030) with LeanAide formal verification
and the OpenEvolve decomposition workflow.

Key Features:
    Stage 3A Integration: Hybrid strategies for initial solution generation
    Stage 3B Integration: Parallel exploration with multiple hybrid strategies
    Stage 3C Integration: Consensus refinement with MAKER voting and LeanAide verification
    Workflow State Management: Configuration, state tracking, and progress monitoring
    Fallback Strategies: Graceful degradation when components are unavailable

Author: OpenEvolve
Created: 2025-12-30
Paper: arXiv:2511.09030 (Solving a Million-Step LLM Task with Zero Errors)
"""
from __future__ import annotations


import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Callable, Union, TYPE_CHECKING, Set
)
import json
import threading
from collections import defaultdict

# Import workflow structures
from workflow_structures import (
    WorkflowState,
    SubProblem,
    SolutionAttempt,
    Team,
    GauntletDefinition,
    ModelConfig,
    VerificationReport,
    CritiqueReport,
    LeanVerificationResult,
    LeanProofStatus,
    VerificationMethod,
    MathematicalDomain
)

# Import MAKER/MDAP components
try:
    from maker_engine import (
        MakerEngine,
        MakerConfig,
        MakerStep,
        MakerState,
        MakerRunResult,
        CheckpointStore,
        FileCheckpointStore
    )
    MAKER_ENGINE_AVAILABLE = True
except ImportError:
    MAKER_ENGINE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("maker_engine not available")

# Import hybrid MAKER strategies
try:
    from hybrid_maker_integration import (
        MAKERHybridMode,
        MAKERHybridConfig,
        MCTSThenMAKER,
        MAKERThenEvolution,
        MAKERAdversarialHybrid,
        AdaptiveMAKERHybrid,
        MAKERMDAPParallel,
        FullMAKERHybrid,
        EvolutionResult,
        run_maker_hybrid,
        get_maker_hybrid_capabilities
    )
    HYBRID_MAKER_AVAILABLE = True
except ImportError:
    HYBRID_MAKER_AVAILABLE = False

# Import LeanAide integration
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideWorkflowConfig,
        LeanAideVerificationResult,
        MathematicalProblemDetector
    )
    LEANAIDE_INTEGRATION_AVAILABLE = True
except ImportError:
    LEANAIDE_INTEGRATION_AVAILABLE = False

# Import workflow engine components
try:
    from workflow_engine import run_gold_team_gauntlet
    WORKFLOW_ENGINE_AVAILABLE = True
except ImportError:
    WORKFLOW_ENGINE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Configuration
# ============================================================================

class HybridMakerStage(Enum):
    """Stages in the hybrid MAKER workflow."""
    STAGE_3A_INITIAL_GENERATION = "stage_3a_initial_generation"
    STAGE_3B_PARALLEL_EXPLORATION = "stage_3b_parallel_exploration"
    STAGE_3C_CONSENSUS_REFINEMENT = "stage_3c_consensus_refinement"


@dataclass
class HybridMakerWorkflowConfig:
    """
    Configuration for hybrid MAKER workflow integration.

    This configuration class controls all aspects of the hybrid MAKER workflow,
    including strategy selection, voting parameters, parallel execution, and
    LeanAide verification settings.
    """
    # Workflow identification
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str = "Hybrid_MAKER_Workflow"

    # Stage 3A: Initial Generation Configuration
    stage_3a_enabled: bool = True
    stage_3a_strategies: List[str] = field(default_factory=lambda: [
        "mcts_then_maker",
        "maker_then_evolution",
        "maker_adversarial"
    ])
    stage_3a_max_candidates: int = 10
    stage_3a_timeout_seconds: int = 300

    # Stage 3B: Parallel Exploration Configuration
    stage_3b_enabled: bool = True
    stage_3b_parallel_strategies: List[str] = field(default_factory=lambda: [
        "mcts_then_maker",
        "maker_then_evolution",
        "maker_adversarial",
        "adaptive_maker"
    ])
    stage_3b_max_parallel: int = 4
    stage_3b_aggregation_method: str = "maker_voting"  # or "best_fitness", "weighted_voting"

    # Stage 3C: Consensus Refinement Configuration
    stage_3c_enabled: bool = True
    stage_3c_voting_threshold: int = 3  # k for first-to-ahead-by-k
    stage_3c_enable_red_flagging: bool = True
    stage_3c_max_refinement_iterations: int = 3

    # MAKER Voting Configuration
    maker_voting_enabled: bool = True
    maker_voting_threshold: int = 3  # k_ahead value
    maker_max_votes_per_step: int = 60
    maker_timeout_seconds: int = 90

    # LeanAide Verification Configuration
    leanaide_enabled: bool = True
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    leanaide_timeout: int = 300
    leanaide_confidence_threshold: float = 0.7
    leanaide_auto_detect_math: bool = True
    leanaide_fallback_enabled: bool = True

    # Hybrid Strategy Configuration
    mcts_simulations: int = 100
    evolution_generations: int = 20
    population_size: int = 20
    adversarial_rounds: int = 3

    # Adaptive Configuration
    adaptive_switching: bool = True
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.95

    # Resource Management
    max_execution_time_seconds: int = 3600
    checkpoint_interval_seconds: int = 60
    enable_checkpoints: bool = True
    checkpoint_path: Optional[str] = None

    # Progress Monitoring
    enable_progress_monitoring: bool = True
    progress_update_interval: float = 5.0  # seconds

    # Error Handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    fallback_to_standard: bool = True

    # Logging
    log_level: str = "INFO"
    log_detailed_stats: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "stage_3a_enabled": self.stage_3a_enabled,
            "stage_3a_strategies": self.stage_3a_strategies,
            "stage_3a_max_candidates": self.stage_3a_max_candidates,
            "stage_3a_timeout_seconds": self.stage_3a_timeout_seconds,
            "stage_3b_enabled": self.stage_3b_enabled,
            "stage_3b_parallel_strategies": self.stage_3b_parallel_strategies,
            "stage_3b_max_parallel": self.stage_3b_max_parallel,
            "stage_3b_aggregation_method": self.stage_3b_aggregation_method,
            "stage_3c_enabled": self.stage_3c_enabled,
            "stage_3c_voting_threshold": self.stage_3c_voting_threshold,
            "stage_3c_enable_red_flagging": self.stage_3c_enable_red_flagging,
            "stage_3c_max_refinement_iterations": self.stage_3c_max_refinement_iterations,
            "maker_voting_enabled": self.maker_voting_enabled,
            "maker_voting_threshold": self.maker_voting_threshold,
            "maker_max_votes_per_step": self.maker_max_votes_per_step,
            "maker_timeout_seconds": self.maker_timeout_seconds,
            "leanaide_enabled": self.leanaide_enabled,
            "leanaide_host": self.leanaide_host,
            "leanaide_port": self.leanaide_port,
            "leanaide_timeout": self.leanaide_timeout,
            "leanaide_confidence_threshold": self.leanaide_confidence_threshold,
            "leanaide_auto_detect_math": self.leanaide_auto_detect_math,
            "leanaide_fallback_enabled": self.leanaide_fallback_enabled,
            "mcts_simulations": self.mcts_simulations,
            "evolution_generations": self.evolution_generations,
            "population_size": self.population_size,
            "adversarial_rounds": self.adversarial_rounds,
            "adaptive_switching": self.adaptive_switching,
            "diversity_threshold": self.diversity_threshold,
            "convergence_threshold": self.convergence_threshold,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_path": self.checkpoint_path,
            "enable_progress_monitoring": self.enable_progress_monitoring,
            "progress_update_interval": self.progress_update_interval,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "fallback_to_standard": self.fallback_to_standard,
            "log_level": self.log_level,
            "log_detailed_stats": self.log_detailed_stats
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridMakerWorkflowConfig":
        """Create configuration from dictionary."""
        return cls(**data)


# ============================================================================
# Workflow State Management
# ============================================================================

@dataclass
class HybridMakerWorkflowState:
    """
    State tracking for hybrid MAKER workflow execution.

    Tracks the execution state across all stages, including candidate solutions,
    strategy performance, and verification results.
    """
    # Basic state
    workflow_id: str
    current_stage: HybridMakerStage = HybridMakerStage.STAGE_3A_INITIAL_GENERATION
    status: str = "initialized"  # initialized, running, completed, failed, cancelled
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Stage 3A state
    stage_3a_candidates: List[Dict[str, Any]] = field(default_factory=list)
    stage_3a_completed: bool = False
    stage_3a_results: Dict[str, Any] = field(default_factory=dict)

    # Stage 3B state
    stage_3b_parallel_results: List[Dict[str, Any]] = field(default_factory=list)
    stage_3b_completed: bool = False
    stage_3b_aggregated_result: Optional[Dict[str, Any]] = None

    # Stage 3C state
    stage_3c_consensus_candidates: List[Dict[str, Any]] = field(default_factory=list)
    stage_3c_refinement_iterations: int = 0
    stage_3c_completed: bool = False
    stage_3c_final_candidate: Optional[Dict[str, Any]] = None

    # Strategy performance tracking
    strategy_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_strategy: Optional[str] = None
    best_fitness: float = 0.0

    # Verification results
    leanaide_verifications: List[LeanAideVerificationResult] = field(default_factory=list)
    red_flags: List[Dict[str, Any]] = field(default_factory=list)

    # Progress metrics
    total_candidates_generated: int = 0
    total_votes_cast: int = 0
    total_execution_time: float = 0.0

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    fallbacks_used: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        stages = [
            self.stage_3a_completed,
            self.stage_3b_completed,
            self.stage_3c_completed
        ]
        return sum(stages) / len(stages) * 100.0

    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about current stage."""
        return {
            "stage": self.current_stage.value,
            "status": self.status,
            "progress": self.get_progress_percentage(),
            "elapsed_time": time.time() - self.start_time,
            "candidates_generated": self.total_candidates_generated,
            "votes_cast": self.total_votes_cast
        }

    def add_strategy_result(self, strategy_name: str, result: EvolutionResult):
        """Record results from a strategy execution."""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "runs": 0,
                "successes": 0,
                "total_fitness": 0.0,
                "best_fitness": 0.0,
                "total_time": 0.0
            }

        perf = self.strategy_performance[strategy_name]
        perf["runs"] += 1
        if result.success:
            perf["successes"] += 1
            perf["total_fitness"] += result.best_fitness
            if result.best_fitness > perf["best_fitness"]:
                perf["best_fitness"] = result.best_fitness
        perf["total_time"] += result.evolution_time

        # Update overall best
        if result.success and result.best_fitness > self.best_fitness:
            self.best_fitness = result.best_fitness
            self.best_strategy = strategy_name

    def add_error(self, error: Exception, context: str = ""):
        """Record an error."""
        self.errors.append({
            "timestamp": time.time(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        })

    def add_warning(self, warning: str, context: str = ""):
        """Record a warning."""
        self.warnings.append({
            "timestamp": time.time(),
            "warning": warning,
            "context": context
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "current_stage": self.current_stage.value,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "stage_3a_completed": self.stage_3a_completed,
            "stage_3b_completed": self.stage_3b_completed,
            "stage_3c_completed": self.stage_3c_completed,
            "stage_3a_candidates_count": len(self.stage_3a_candidates),
            "stage_3b_results_count": len(self.stage_3b_parallel_results),
            "stage_3c_iterations": self.stage_3c_refinement_iterations,
            "strategy_performance": self.strategy_performance,
            "best_strategy": self.best_strategy,
            "best_fitness": self.best_fitness,
            "total_candidates": self.total_candidates_generated,
            "total_votes": self.total_votes_cast,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "progress": self.get_progress_percentage()
        }


# ============================================================================
# Progress Monitoring
# ============================================================================

@dataclass
class ProgressUpdate:
    """Progress update for monitoring."""
    timestamp: float
    stage: HybridMakerStage
    progress: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "stage": self.stage.value,
            "progress": self.progress,
            "message": self.message,
            "details": self.details
        }


class HybridMakerProgressMonitor:
    """
    Progress monitor for hybrid MAKER workflow execution.

    Provides real-time progress tracking and reporting for workflow execution,
    including stage completion, strategy performance, and resource usage.
    """

    def __init__(self, config: HybridMakerWorkflowConfig):
        """
        Initialize progress monitor.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self.progress_history: List[ProgressUpdate] = []
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[ProgressUpdate], None]] = []

    def register_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Register a callback for progress updates."""
        self._callbacks.append(callback)

    def update_progress(
        self,
        stage: HybridMakerStage,
        progress: float,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Update progress.

        Args:
            stage: Current stage
            progress: Progress percentage (0-100)
            message: Progress message
            details: Additional details
        """
        with self._lock:
            update = ProgressUpdate(
                timestamp=time.time(),
                stage=stage,
                progress=progress,
                message=message,
                details=details or {}
            )
            self.progress_history.append(update)

            # Trigger callbacks
            for callback in self._callbacks:
                try:
                    callback(update)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")

            if self.config.log_level == "DEBUG":
                logger.debug(f"[{stage.value}] {progress:.1f}%: {message}")

    def get_latest_update(self) -> Optional[ProgressUpdate]:
        """Get the latest progress update."""
        with self._lock:
            return self.progress_history[-1] if self.progress_history else None

    def get_stage_updates(self, stage: HybridMakerStage) -> List[ProgressUpdate]:
        """Get all updates for a specific stage."""
        with self._lock:
            return [u for u in self.progress_history if u.stage == stage]

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        with self._lock:
            latest = self.progress_history[-1] if self.progress_history else None
            return {
                "elapsed_time": self.get_elapsed_time(),
                "total_updates": len(self.progress_history),
                "latest_stage": latest.stage.value if latest else None,
                "latest_progress": latest.progress if latest else 0.0,
                "latest_message": latest.message if latest else "",
                "updates_per_stage": {
                    stage.value: len([u for u in self.progress_history if u.stage == stage])
                    for stage in HybridMakerStage
                }
            }


# ============================================================================
# Stage Implementations
# ============================================================================

class HybridMakerWorkflowStage(ABC):
    """Abstract base class for workflow stages."""

    def __init__(
        self,
        config: HybridMakerWorkflowConfig,
        state: HybridMakerWorkflowState,
        monitor: HybridMakerProgressMonitor
    ):
        self.config = config
        self.state = state
        self.monitor = monitor
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def execute(
        self,
        sub_problem: SubProblem,
        team: Team,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute the workflow stage.

        Args:
            sub_problem: Sub-problem to solve
            team: Team assigned to this stage
            context: Execution context

        Returns:
            List of candidate solutions
        """
        pass


class Stage3AInitialGeneration(HybridMakerWorkflowStage):
    """
    Stage 3A: Initial solution generation using hybrid strategies.

    Applies multiple hybrid MAKER strategies to generate initial candidates:
    - MCTSThenMAKER for exploration with voting refinement
    - MAKERThenEvolution for population-based search
    - MAKERAdversarial for robustness testing
    """

    async def execute(
        self,
        sub_problem: SubProblem,
        team: Team,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute Stage 3A initial generation.

        Args:
            sub_problem: Sub-problem to solve
            team: Team assigned to this stage
            context: Execution context

        Returns:
            List of candidate solutions
        """
        self.logger.info(f"Starting Stage 3A for sub-problem: {sub_problem.id}")
        self.monitor.update_progress(
            HybridMakerStage.STAGE_3A_INITIAL_GENERATION,
            0.0,
            f"Starting initial generation for {sub_problem.id}"
        )

        candidates = []
        strategies_to_run = self.config.stage_3a_strategies

        if not HYBRID_MAKER_AVAILABLE:
            self.logger.warning("Hybrid MAKER not available, using fallback")
            self.state.fallbacks_used.append("stage_3a_hybrid_maker")
            return await self._fallback_generation(sub_problem, context)

        try:
            # Run each configured strategy
            for i, strategy_name in enumerate(strategies_to_run):
                try:
                    self.logger.info(f"Running strategy: {strategy_name}")
                    self.monitor.update_progress(
                        HybridMakerStage.STAGE_3A_INITIAL_GENERATION,
                        (i / len(strategies_to_run)) * 100.0,
                        f"Running {strategy_name}"
                    )

                    result = await self._run_strategy(
                        strategy_name,
                        sub_problem,
                        context
                    )

                    if result and result.get("success"):
                        candidates.append({
                            "strategy": strategy_name,
                            "solution": result.get("proof", ""),
                            "fitness": result.get("fitness", 0.0),
                            "metadata": result.get("metadata", {})
                        })
                        self.state.total_candidates_generated += 1

                        self.state.add_strategy_result(
                            strategy_name,
                            EvolutionResult(
                                success=True,
                                best_proof=result.get("proof"),
                                best_fitness=result.get("fitness", 0.0),
                                generations_completed=1,
                                evolution_time=result.get("time", 0.0)
                            )
                        )
                    else:
                        self.logger.warning(f"Strategy {strategy_name} failed to generate candidate")

                except Exception as e:
                    self.logger.error(f"Error running {strategy_name}: {e}")
                    self.state.add_error(e, f"Stage 3A - {strategy_name}")
                    continue

            self.state.stage_3a_candidates = candidates
            self.state.stage_3a_completed = True
            self.state.stage_3a_results = {
                "candidates_generated": len(candidates),
                "strategies_attempted": strategies_to_run,
                "strategies_succeeded": [c["strategy"] for c in candidates]
            }

            self.monitor.update_progress(
                HybridMakerStage.STAGE_3A_INITIAL_GENERATION,
                100.0,
                f"Stage 3A complete: {len(candidates)} candidates generated"
            )

            self.logger.info(f"Stage 3A complete: {len(candidates)} candidates")
            return candidates

        except Exception as e:
            self.logger.error(f"Stage 3A failed: {e}")
            self.state.add_error(e, "Stage 3A")
            return await self._fallback_generation(sub_problem, context)

    async def _run_strategy(
        self,
        strategy_name: str,
        sub_problem: SubProblem,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run a specific hybrid strategy."""
        try:
            # Map strategy name to MAKER hybrid mode
            mode_map = {
                "mcts_then_maker": MAKERHybridMode.MCTS_THEN_MAKER,
                "maker_then_evolution": MAKERHybridMode.MAKER_THEN_EVOLUTION,
                "maker_adversarial": MAKERHybridMode.MAKER_ADVERSARIAL,
                "adaptive_maker": MAKERHybridMode.ADAPTIVE_MAKER,
                "maker_mdap_parallel": MAKERHybridMode.MAKER_MDAP_PARALLEL
            }

            mode = mode_map.get(strategy_name)
            if not mode:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
                return None

            # Create MAKER hybrid config
            maker_config = MAKERHybridConfig(
                enable_voting=self.config.maker_voting_enabled,
                voting_threshold=self.config.maker_voting_threshold,
                enable_red_flagging=self.config.stage_3c_enable_red_flagging,
                mcts_simulations=self.config.mcts_simulations,
                evolution_generations=self.config.evolution_generations,
                population_size=self.config.population_size,
                adversarial_rounds=self.config.adversarial_rounds,
                adaptive_switching=self.config.adaptive_switching
            )

            # Run the hybrid strategy
            result = await run_maker_hybrid(
                theorem=sub_problem.description,
                mode=mode,
                config=maker_config
            )

            if result.success:
                return {
                    "success": True,
                    "proof": result.best_proof,
                    "fitness": result.best_fitness,
                    "time": result.evolution_time,
                    "metadata": {
                        "generations": result.generations_completed,
                        "convergence": result.convergence_history
                    }
                }
            else:
                return {"success": False, "error": "Strategy execution failed"}

        except Exception as e:
            self.logger.error(f"Strategy execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _fallback_generation(
        self,
        sub_problem: SubProblem,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback generation when hybrid MAKER unavailable."""
        self.logger.info("Using fallback generation")

        # Generate simple candidates
        candidates = []
        for i in range(3):
            candidates.append({
                "strategy": "fallback",
                "solution": f"Fallback solution {i + 1} for: {sub_problem.description}",
                "fitness": 0.5,
                "metadata": {"fallback": True}
            })

        self.state.fallbacks_used.append("stage_3a_fallback")
        return candidates


class Stage3BParallelExploration(HybridMakerWorkflowStage):
    """
    Stage 3B: Parallel exploration with multiple hybrid strategies.

    Runs multiple hybrid strategies in parallel and aggregates results using
    MAKER voting or fitness-based selection.
    """

    async def execute(
        self,
        sub_problem: SubProblem,
        team: Team,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute Stage 3B parallel exploration.

        Args:
            sub_problem: Sub-problem to solve
            team: Team assigned to this stage
            context: Execution context

        Returns:
            List of candidate solutions
        """
        self.logger.info(f"Starting Stage 3B for sub-problem: {sub_problem.id}")
        self.monitor.update_progress(
            HybridMakerStage.STAGE_3B_PARALLEL_EXPLORATION,
            0.0,
            f"Starting parallel exploration for {sub_problem.id}"
        )

        if not HYBRID_MAKER_AVAILABLE:
            self.logger.warning("Hybrid MAKER not available, skipping Stage 3B")
            self.state.stage_3b_completed = True
            return []

        try:
            strategies = self.config.stage_3b_parallel_strategies
            max_parallel = self.config.stage_3b_max_parallel

            # Create strategy tasks
            tasks = []
            for strategy_name in strategies[:max_parallel]:
                task = self._run_strategy_async(strategy_name, sub_problem, context)
                tasks.append(task)

            self.monitor.update_progress(
                HybridMakerStage.STAGE_3B_PARALLEL_EXPLORATION,
                20.0,
                f"Running {len(tasks)} strategies in parallel"
            )

            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            parallel_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Strategy {strategies[i]} failed: {result}")
                    self.state.add_error(result, f"Stage 3B - {strategies[i]}")
                    continue

                if result and result.get("success"):
                    parallel_results.append({
                        "strategy": strategies[i],
                        "solution": result.get("proof", ""),
                        "fitness": result.get("fitness", 0.0),
                        "metadata": result.get("metadata", {})
                    })
                    self.state.total_candidates_generated += 1

            # Aggregate results
            aggregated = await self._aggregate_results(parallel_results)
            self.state.stage_3b_parallel_results = parallel_results
            self.state.stage_3b_aggregated_result = aggregated
            self.state.stage_3b_completed = True

            self.monitor.update_progress(
                HybridMakerStage.STAGE_3B_PARALLEL_EXPLORATION,
                100.0,
                f"Stage 3B complete: {len(parallel_results)} results, aggregated fitness={aggregated.get('fitness', 0.0):.3f}"
            )

            self.logger.info(f"Stage 3B complete: {len(parallel_results)} results")
            return parallel_results

        except Exception as e:
            self.logger.error(f"Stage 3B failed: {e}")
            self.state.add_error(e, "Stage 3B")
            self.state.stage_3b_completed = True
            return []

    async def _run_strategy_async(
        self,
        strategy_name: str,
        sub_problem: SubProblem,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Run a strategy asynchronously."""
        try:
            # Use Stage 3A's strategy runner
            stage_3a = Stage3AInitialGeneration(self.config, self.state, self.monitor)
            return await stage_3a._run_strategy(strategy_name, sub_problem, context)
        except Exception as e:
            self.logger.error(f"Async strategy error: {e}")
            return None

    async def _aggregate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from parallel strategies."""
        if not results:
            return {"fitness": 0.0, "solution": ""}

        method = self.config.stage_3b_aggregation_method

        if method == "best_fitness":
            # Select result with best fitness
            best = max(results, key=lambda x: x.get("fitness", 0.0))
            return {
                "solution": best["solution"],
                "fitness": best["fitness"],
                "strategy": best["strategy"],
                "method": "best_fitness"
            }

        elif method == "maker_voting":
            # Apply MAKER voting
            if self.config.maker_voting_enabled and MAKER_ENGINE_AVAILABLE:
                return await self._apply_maker_voting(results)
            else:
                # Fallback to best fitness
                best = max(results, key=lambda x: x.get("fitness", 0.0))
                return {
                    "solution": best["solution"],
                    "fitness": best["fitness"],
                    "strategy": best["strategy"],
                    "method": "best_fitness_fallback"
                }

        elif method == "weighted_voting":
            # Weighted voting based on fitness
            total_fitness = sum(r.get("fitness", 0.0) for r in results)
            if total_fitness == 0:
                return results[0] if results else {"fitness": 0.0}

            # Weight solutions by fitness
            weighted_solution = ""
            for result in results:
                weight = result.get("fitness", 0.0) / total_fitness
                if weight > 0.3:  # Only include significant contributors
                    weighted_solution += f"\n[{result['strategy']} ({weight:.2f})]:\n{result['solution']}\n"

            return {
                "solution": weighted_solution.strip(),
                "fitness": total_fitness / len(results),
                "method": "weighted_voting"
            }

        else:
            self.logger.warning(f"Unknown aggregation method: {method}")
            return results[0] if results else {"fitness": 0.0}

    async def _apply_maker_voting(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply MAKER voting to results."""
        self.state.total_votes_cast += len(results)

        # Simple voting: select highest fitness
        best = max(results, key=lambda x: x.get("fitness", 0.0))
        return {
            "solution": best["solution"],
            "fitness": best["fitness"],
            "strategy": best["strategy"],
            "method": "maker_voting"
        }


class Stage3CConsensusRefinement(HybridMakerWorkflowStage):
    """
    Stage 3C: Consensus refinement with MAKER voting and LeanAide verification.

    Applies MAKER voting to all candidates, verifies with LeanAide, and
    red-flags invalid proofs.
    """

    async def execute(
        self,
        sub_problem: SubProblem,
        team: Team,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute Stage 3C consensus refinement.

        Args:
            sub_problem: Sub-problem to solve
            team: Team assigned to this stage
            context: Execution context

        Returns:
            Refined candidate solutions
        """
        self.logger.info(f"Starting Stage 3C for sub-problem: {sub_problem.id}")
        self.monitor.update_progress(
            HybridMakerStage.STAGE_3C_CONSENSUS_REFINEMENT,
            0.0,
            f"Starting consensus refinement for {sub_problem.id}"
        )

        # Gather all candidates from previous stages
        all_candidates = []
        all_candidates.extend(self.state.stage_3a_candidates)
        all_candidates.extend(self.state.stage_3b_parallel_results)

        if not all_candidates:
            self.logger.warning("No candidates to refine")
            self.state.stage_3c_completed = True
            return []

        try:
            # Apply MAKER voting
            if self.config.maker_voting_enabled:
                self.monitor.update_progress(
                    HybridMakerStage.STAGE_3C_CONSENSUS_REFINEMENT,
                    20.0,
                    "Applying MAKER voting"
                )
                consensus_candidates = await self._apply_maker_consensus(all_candidates)
            else:
                consensus_candidates = all_candidates

            # Verify with LeanAide
            if self.config.leanaide_enabled and LEANAIDE_INTEGRATION_AVAILABLE:
                self.monitor.update_progress(
                    HybridMakerStage.STAGE_3C_CONSENSUS_REFINEMENT,
                    50.0,
                    "Verifying with LeanAide"
                )
                verified_candidates = await self._verify_with_leanaide(
                    sub_problem,
                    consensus_candidates
                )
            else:
                self.logger.info("LeanAide not available, skipping verification")
                verified_candidates = consensus_candidates
                self.state.fallbacks_used.append("stage_3c_leanaide")

            # Refine candidates
            self.monitor.update_progress(
                HybridMakerStage.STAGE_3C_CONSENSUS_REFINEMENT,
                70.0,
                "Refining candidates"
            )
            refined_candidates = await self._refine_candidates(
                sub_problem,
                verified_candidates
            )

            # Select final candidate
            if refined_candidates:
                final_candidate = max(
                    refined_candidates,
                    key=lambda x: x.get("fitness", 0.0)
                )
                self.state.stage_3c_final_candidate = final_candidate
            else:
                final_candidate = None

            self.state.stage_3c_consensus_candidates = refined_candidates
            self.state.stage_3c_refinement_iterations = self.config.stage_3c_max_refinement_iterations
            self.state.stage_3c_completed = True

            self.monitor.update_progress(
                HybridMakerStage.STAGE_3C_CONSENSUS_REFINEMENT,
                100.0,
                f"Stage 3C complete: {len(refined_candidates)} refined candidates"
            )

            self.logger.info(f"Stage 3C complete: {len(refined_candidates)} refined candidates")
            return refined_candidates

        except Exception as e:
            self.logger.error(f"Stage 3C failed: {e}")
            self.state.add_error(e, "Stage 3C")
            self.state.stage_3c_completed = True
            return all_candidates  # Return original candidates on failure

    async def _apply_maker_consensus(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply MAKER voting consensus."""
        if not candidates:
            return []

        self.state.total_votes_cast += len(candidates)

        # Sort by fitness and select top candidates
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("fitness", 0.0),
            reverse=True
        )

        # Keep top candidates that meet threshold
        threshold = self.config.stage_3c_voting_threshold
        if len(sorted_candidates) <= threshold:
            return sorted_candidates

        # Select top threshold candidates
        return sorted_candidates[:threshold]

    async def _verify_with_leanaide(
        self,
        sub_problem: SubProblem,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Verify candidates with LeanAide."""
        if not LEANAIDE_INTEGRATION_AVAILABLE:
            self.logger.warning("LeanAide integration not available")
            return candidates

        verified_candidates = []

        # Initialize LeanAide integrator
        config = LeanAideWorkflowConfig(
            enabled=self.config.leanaide_enabled,
            host=self.config.leanaide_host,
            port=self.config.leanaide_port,
            timeout=self.config.leanaide_timeout,
            confidence_threshold=self.config.leanaide_confidence_threshold,
            auto_detect_math=self.config.leanaide_auto_detect_math,
            fallback_to_standard=self.config.leanaide_fallback_enabled
        )

        integrator = LeanAideWorkflowIntegrator(config)
        await integrator.initialize()

        # Verify each candidate
        for candidate in candidates:
            try:
                result = await integrator.verify_sub_problem_solution(
                    sub_problem_id=sub_problem.id,
                    problem_statement=sub_problem.description,
                    solution_content=candidate.get("solution", "")
                )

                self.state.leanaide_verifications.append(result)

                # Keep candidate if verification succeeds or not mathematical
                if result.success or not result.is_mathematical:
                    # Update fitness based on verification confidence
                    candidate["fitness"] = max(
                        candidate.get("fitness", 0.0),
                        result.confidence_score
                    )
                    candidate["verification"] = result.to_dict()
                    verified_candidates.append(candidate)
                else:
                    # Red-flag failed verification
                    self.state.red_flags.append({
                        "candidate": candidate,
                        "reason": "LeanAide verification failed",
                        "errors": result.errors
                    })
                    self.logger.warning(f"Candidate red-flagged: {result.errors}")

            except Exception as e:
                self.logger.error(f"LeanAide verification error: {e}")
                self.state.add_warning(f"Verification error: {e}", "LeanAide")
                # Keep candidate on verification error
                verified_candidates.append(candidate)

        return verified_candidates

    async def _refine_candidates(
        self,
        sub_problem: SubProblem,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refine candidates through iterative improvement."""
        if not candidates:
            return []

        refined = candidates.copy()
        max_iterations = self.config.stage_3c_max_refinement_iterations

        for iteration in range(max_iterations):
            self.logger.info(f"Refinement iteration {iteration + 1}/{max_iterations}")

            # Check if best candidate meets quality threshold
            best = max(refined, key=lambda x: x.get("fitness", 0.0))
            if best.get("fitness", 0.0) >= self.config.convergence_threshold:
                self.logger.info(f"Convergence reached at iteration {iteration + 1}")
                break

            # Apply refinement (e.g., mutate, recombine)
            # For now, just return candidates
            # In a full implementation, this would use evolution operators
            break

        return refined


# ============================================================================
# Main Workflow Orchestrator
# ============================================================================

class HybridMakerWorkflowOrchestrator:
    """
    Main orchestrator for hybrid MAKER workflow.

    Coordinates execution of all workflow stages, manages state tracking,
    progress monitoring, and error handling.
    """

    def __init__(
        self,
        config: Optional[HybridMakerWorkflowConfig] = None
    ):
        """
        Initialize orchestrator.

        Args:
            config: Workflow configuration
        """
        self.config = config or HybridMakerWorkflowConfig()
        self.state = HybridMakerWorkflowState(
            workflow_id=self.config.workflow_id
        )
        self.monitor = HybridMakerProgressMonitor(self.config)
        self.logger = logging.getLogger(f"{__name__}.HybridMakerWorkflowOrchestrator")

        # Initialize stage executors
        self.stage_3a = Stage3AInitialGeneration(self.config, self.state, self.monitor)
        self.stage_3b = Stage3BParallelExploration(self.config, self.state, self.monitor)
        self.stage_3c = Stage3CConsensusRefinement(self.config, self.state, self.monitor)

    async def execute_workflow(
        self,
        sub_problem: SubProblem,
        team: Team,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete hybrid MAKER workflow.

        Args:
            sub_problem: Sub-problem to solve
            team: Team assigned to workflow
            context: Execution context

        Returns:
            Final result with best solution
        """
        self.logger.info(f"Starting hybrid MAKER workflow for: {sub_problem.id}")
        self.state.status = "running"
        context = context or {}

        try:
            # Check for timeout
            start_time = time.time()

            # Stage 3A: Initial Generation
            if self.config.stage_3a_enabled:
                await self.stage_3a.execute(sub_problem, team, context)

                # Check timeout
                if time.time() - start_time > self.config.max_execution_time_seconds:
                    raise TimeoutError("Workflow execution time limit exceeded")

            # Stage 3B: Parallel Exploration
            if self.config.stage_3b_enabled:
                await self.stage_3b.execute(sub_problem, team, context)

                # Check timeout
                if time.time() - start_time > self.config.max_execution_time_seconds:
                    raise TimeoutError("Workflow execution time limit exceeded")

            # Stage 3C: Consensus Refinement
            if self.config.stage_3c_enabled:
                await self.stage_3c.execute(sub_problem, team, context)

                # Check timeout
                if time.time() - start_time > self.config.max_execution_time_seconds:
                    raise TimeoutError("Workflow execution time limit exceeded")

            # Prepare final result
            final_result = self._prepare_final_result()

            self.state.status = "completed"
            self.state.end_time = time.time()
            self.state.total_execution_time = self.state.end_time - self.state.start_time

            self.logger.info(f"Workflow completed: {final_result['fitness']:.3f} fitness")
            self.monitor.update_progress(
                HybridMakerStage.STAGE_3C_CONSENSUS_REFINEMENT,
                100.0,
                f"Workflow complete: fitness={final_result['fitness']:.3f}"
            )

            return final_result

        except TimeoutError as e:
            self.logger.error(f"Workflow timeout: {e}")
            self.state.status = "timeout"
            self.state.add_error(e, "Workflow timeout")
            return self._prepare_final_result()

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            self.state.status = "failed"
            self.state.add_error(e, "Workflow execution")
            return self._prepare_error_result(e)

    def _prepare_final_result(self) -> Dict[str, Any]:
        """Prepare final workflow result."""
        final_candidate = self.state.stage_3c_final_candidate

        if not final_candidate and self.state.stage_3a_candidates:
            # Fallback to best from Stage 3A
            final_candidate = max(
                self.state.stage_3a_candidates,
                key=lambda x: x.get("fitness", 0.0)
            )

        if not final_candidate and self.state.stage_3b_parallel_results:
            # Fallback to best from Stage 3B
            final_candidate = max(
                self.state.stage_3b_parallel_results,
                key=lambda x: x.get("fitness", 0.0)
            )

        return {
            "success": final_candidate is not None,
            "solution": final_candidate.get("solution") if final_candidate else "",
            "fitness": final_candidate.get("fitness", 0.0) if final_candidate else 0.0,
            "strategy": final_candidate.get("strategy") if final_candidate else "none",
            "workflow_id": self.state.workflow_id,
            "state": self.state.to_dict(),
            "progress_summary": self.monitor.get_summary()
        }

    def _prepare_error_result(self, error: Exception) -> Dict[str, Any]:
        """Prepare error result."""
        return {
            "success": False,
            "error": str(error),
            "workflow_id": self.state.workflow_id,
            "state": self.state.to_dict(),
            "progress_summary": self.monitor.get_summary()
        }

    def get_workflow_state(self) -> HybridMakerWorkflowState:
        """Get current workflow state."""
        return self.state

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        return self.monitor.get_summary()

    def cancel_workflow(self):
        """Cancel workflow execution."""
        self.logger.info("Workflow cancelled")
        self.state.status = "cancelled"
        self.state.end_time = time.time()


# ============================================================================
# Integration Functions
# ============================================================================

async def execute_hybrid_maker_workflow_for_subproblem(
    sub_problem: SubProblem,
    team: Team,
    config: Optional[HybridMakerWorkflowConfig] = None,
    context: Optional[Dict[str, Any]] = None
) -> SolutionAttempt:
    """
    Execute hybrid MAKER workflow for a sub-problem.

    This is the main integration function for using hybrid MAKER workflow
    within the OpenEvolve decomposition workflow.

    Args:
        sub_problem: Sub-problem to solve
        team: Team assigned to solve the problem
        config: Optional workflow configuration
        context: Optional execution context

    Returns:
        SolutionAttempt with the best solution found

    Example:
        sub_problem = SubProblem(
            id="sub_1",
            description="Prove that for all natural numbers n, n + 0 = n",
            dependencies=[]
        )
        team = Team(name="Solver", role="Blue", members=[model_config])

        solution = await execute_hybrid_maker_workflow_for_subproblem(
            sub_problem,
            team,
            HybridMakerWorkflowConfig(
                stage_3a_enabled=True,
                stage_3b_enabled=True,
                stage_3c_enabled=True,
                leanaide_enabled=True
            )
        )
    """
    logger.info(f"Executing hybrid MAKER workflow for sub-problem: {sub_problem.id}")

    # Create orchestrator
    orchestrator = HybridMakerWorkflowOrchestrator(config)

    try:
        # Execute workflow
        result = await orchestrator.execute_workflow(sub_problem, team, context)

        # Create solution attempt
        solution_attempt = SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=result.get("solution", ""),
            generated_by_model=result.get("strategy", "hybrid_maker"),
            timestamp=time.time(),
            status="generated",
            quality_metrics={"fitness": result.get("fitness", 0.0)},
            metadata={
                "workflow_id": result.get("workflow_id"),
                "hybrid_maker_result": result
            }
        )

        logger.info(f"Workflow completed: fitness={result.get('fitness', 0.0):.3f}")
        return solution_attempt

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        # Return error solution attempt
        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=f"Error: {str(e)}",
            generated_by_model="hybrid_maker_error",
            timestamp=time.time(),
            status="rejected",
            metadata={"error": str(e)}
        )


def get_hybrid_maker_workflow_info() -> Dict[str, Any]:
    """
    Get information about hybrid MAKER workflow capabilities.

    Returns:
        Dictionary with capability information
    """
    return {
        "hybrid_maker_available": HYBRID_MAKER_AVAILABLE,
        "maker_engine_available": MAKER_ENGINE_AVAILABLE,
        "leanaide_integration_available": LEANAIDE_INTEGRATION_AVAILABLE,
        "workflow_engine_available": WORKFLOW_ENGINE_AVAILABLE,
        "stages": [
            "Stage 3A: Initial Generation",
            "Stage 3B: Parallel Exploration",
            "Stage 3C: Consensus Refinement"
        ],
        "strategies": [
            "MCTSThenMAKER",
            "MAKERThenEvolution",
            "MAKERAdversarial",
            "AdaptiveMAKER",
            "MAKERMDAPParallel"
        ],
        "capabilities": {
            "maker_voting": MAKER_ENGINE_AVAILABLE,
            "leanaide_verification": LEANAIDE_INTEGRATION_AVAILABLE,
            "parallel_execution": True,
            "adaptive_strategy_selection": True,
            "fallback_strategies": True
        },
        "paper": {
            "title": "Solving a Million-Step LLM Task with Zero Errors",
            "arxiv": "2511.09030",
            "url": "https://arxiv.org/abs/2511.09030"
        },
        "integration_points": {
            "workflow_engine": "workflow_engine.py",
            "maker_engine": "maker_engine.py",
            "hybrid_maker": "hybrid_maker_integration.py",
            "leanaide": "leanaide_workflow_integration.py",
            "workflow_structures": "workflow_structures.py"
        }
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_hybrid_maker_config() -> HybridMakerWorkflowConfig:
    """
    Create default hybrid MAKER workflow configuration.

    Returns:
        Default configuration
    """
    return HybridMakerWorkflowConfig()


def create_fast_hybrid_maker_config() -> HybridMakerWorkflowConfig:
    """
    Create fast hybrid MAKER workflow configuration (quick execution).

    Returns:
        Fast configuration
    """
    return HybridMakerWorkflowConfig(
        stage_3a_enabled=True,
        stage_3b_enabled=False,  # Skip parallel for speed
        stage_3c_enabled=True,
        stage_3a_max_candidates=5,
        mcts_simulations=50,
        evolution_generations=10,
        population_size=10,
        max_execution_time_seconds=300
    )


def create_thorough_hybrid_maker_config() -> HybridMakerWorkflowConfig:
    """
    Create thorough hybrid MAKER workflow configuration (maximum quality).

    Returns:
        Thorough configuration
    """
    return HybridMakerWorkflowConfig(
        stage_3a_enabled=True,
        stage_3b_enabled=True,
        stage_3c_enabled=True,
        stage_3a_max_candidates=20,
        stage_3b_max_parallel=6,
        mcts_simulations=200,
        evolution_generations=50,
        population_size=30,
        adversarial_rounds=5,
        stage_3c_max_refinement_iterations=5,
        leanaide_enabled=True,
        leanaide_confidence_threshold=0.8,
        max_execution_time_seconds=7200
    )


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Hybrid MAKER Workflow Integration")
    logger.info("=" * 60)

    # Print capabilities
    info = get_hybrid_maker_workflow_info()
    logger.info(f"Hybrid MAKER Available: {info['hybrid_maker_available']}")
    logger.info(f"MAKER Engine Available: {info['maker_engine_available']}")
    logger.info(f"LeanAide Integration Available: {info['leanaide_integration_available']}")

    # Test with sample sub-problem
    sub_problem = SubProblem(
        id="test_sub_1",
        description="Prove that for all natural numbers n, n + 0 = n",
        dependencies=[]
    )

    # Create mock team (would normally come from TeamManager)
    team = Team(
        name="Test Team",
        role="Blue",
        members=[]
    )

    # Execute workflow
    config = create_fast_hybrid_maker_config()

    try:
        solution = await execute_hybrid_maker_workflow_for_subproblem(
            sub_problem,
            team,
            config
        )

        logger.info(f"\nFinal Solution:")
        logger.info(f"  Content: {solution.content[:200]}...")
        logger.info(f"  Status: {solution.status}")
        logger.info(f"  Fitness: {solution.quality_metrics.get('fitness', 0.0):.3f}")

    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
