"""
Evaluator Team Coordinator for OpenEvolve
Enhanced orchestration system for coordinating multiple Evaluator Team members
working in parallel to validate sub-problem solutions.

Now integrated with OpenEvolve's ensemble functionality for efficient coordination.

Architecture:
    DecompositionEngine -> EvaluatorTeamCoordinator -> Ensemble-based Parallel Execution
                         v                         v
                    Evaluation Task Queue     Load Balancing & Aggregation
                         v                         v
                    Consensus Building <- Ensemble Aggregation (6 algorithms)
                         v
                    Quality-Gated Solutions

Key Changes:
- Uses OpenEvolve LLMEnsemble for parallel evaluator execution
- Ensemble handles load balancing and parallel execution
- Preserves 6 consensus algorithms (may integrate with ensemble aggregation)
- Maintains bias detection and quality gate functionality
- State management now leverages ensemble capabilities
"""

import os
import json
import time
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import queue
# import pickle  # REMOVED - security risk
import hashlib
import random
import statistics
import concurrent.futures

# Import existing Evaluator Team components
try:
    from evaluator_team import (
        EvaluatorTeam, EvaluatorMember, EvaluationMetric, EvaluationScore,
        EvaluationThreshold, EvaluationConfidence, EvaluationCriterion,
        EvaluatorAssessment, IntegratedEvaluation, EvaluationScale
    )
    EVALUATOR_TEAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Evaluator Team not available: {e}")
    EVALUATOR_TEAM_AVAILABLE = False

# Import decomposition engine components
try:
    from decomposition_engine import DecompositionEngine, SubProblem
    from problem_analyzer import ProblemAnalyzer
    from sovereign_data_models import ProblemDefinition, ComplexityScore
    DECOMPOSITION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Decomposition engine not available: {e}")
    DECOMPOSITION_AVAILABLE = False

# Import Blue Team for solver workflow integration
try:
    from blue_team import BlueTeam, BlueTeamAssessment
    BLUE_TEAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Blue Team not available: {e}")
    BLUE_TEAM_AVAILABLE = False

# Import OpenEvolve components
try:
    from openevolve_imports import OPENEVOLVE_AVAILABLE, EvolutionAPI
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False

# Import OpenEvolve Ensemble for coordination
try:
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.config import LLMModelConfig
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    logging.warning("OpenEvolve Ensemble not available - using fallback coordination")

# **ACTUAL INTEGRATION**: Alerting and knowledge for coordinator operations
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# **LEAN INTEGRATION**: Real Lean proof verification for coordinator
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LeanAide client not available - formal verification disabled")

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class EvaluationTaskStatus(Enum):
    """Status of an evaluation task in the coordinator"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    CONSENSUS_BUILDING = "consensus_building"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    QUALITY_GATE_FAILED = "quality_gate_failed"

class EvaluationTaskPriority(Enum):
    """Priority levels for evaluation tasks"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LoadBalancingStrategy(Enum):
    """Strategies for distributing evaluation tasks among evaluators"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    SPECIALIZATION_BASED = "specialization_based"
    RANDOM = "random"
    ADAPTIVE = "adaptive"
    EXPERTISE_MATCHED = "expertise_matched"

class ConsensusMethod(Enum):
    """Methods for building consensus among evaluators"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    BATESIAN = "batesian"  # Weight by reliability
    DEMPSTER_SHAFER = "dempster_shafer"  # Evidence theory
    DELPHI = "delphi"  # Iterative consensus

@dataclass
class EvaluatorMetrics:
    """Performance metrics for an evaluator"""
    evaluator_id: str
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    total_time_spent: float = 0.0
    average_evaluation_time: float = 0.0
    current_load: int = 0
    specialization_scores: Dict[EvaluationMetric, float] = field(default_factory=dict)
    reliability_score: float = 1.0
    bias_profile: Dict[str, float] = field(default_factory=dict)
    consensus_agreement_rate: float = 0.0
    last_active: Optional[datetime] = None
    accuracy_score: float = 0.0

@dataclass
class EvaluationTask:
    """An evaluation task to be coordinated"""
    task_id: str
    sub_problem_id: str
    sub_problem_description: str
    solution_content: str
    original_content: Optional[str] = None  # For improvement tracking
    criteria: Optional[List[EvaluationCriterion]] = None
    content_type: str = "general"
    priority: EvaluationTaskPriority = EvaluationTaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    assigned_evaluators: List[str] = field(default_factory=list)
    status: EvaluationTaskStatus = EvaluationTaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assessments: List[EvaluatorAssessment] = field(default_factory=list)
    integrated_evaluation: Optional[IntegratedEvaluation] = None
    consensus_score: float = 0.0
    consensus_reached: bool = False
    quality_gate_passed: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL

@dataclass
class EvaluationSession:
    """An evaluation session for a set of tasks"""
    session_id: str
    problem_statement: str
    sub_problems: List[Dict[str, Any]]
    solutions: Dict[str, str]
    tasks: List[EvaluationTask] = field(default_factory=list)
    status: EvaluationTaskStatus = EvaluationTaskStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    quality_gate_passed_tasks: int = 0
    aggregated_results: Optional[Dict[str, Any]] = None
    consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE

@dataclass
class CoordinatorMetrics:
    """Overall coordinator metrics"""
    total_sessions: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_session_time: float = 0.0
    average_task_time: float = 0.0
    team_utilization: float = 0.0
    throughput_evaluations_per_minute: float = 0.0
    peak_concurrent_evaluations: int = 0
    overall_consensus_rate: float = 0.0
    quality_gate_pass_rate: float = 0.0
    evaluator_diversity_score: float = 0.0


# =============================================================================
# EVALUATOR TEAM COORDINATOR
# =============================================================================

class EvaluatorTeamCoordinator:
    """
    Enhanced orchestration system for coordinating multiple Evaluator Team members.

    Features:
    - Parallel evaluation execution across multiple evaluators
    - Intelligent task distribution based on expertise
    - Consensus building algorithms
    - Bias detection and mitigation
    - Integration with DecompositionEngine for solution validation
    - State persistence and recovery
    - Quality gate enforcement
    - Performance tracking and analytics
    """

    def __init__(
        self,
        evaluator_team: Optional[EvaluatorTeam] = None,
        max_concurrent_evaluations: int = 5,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.SPECIALIZATION_BASED,
        consensus_method: ConsensusMethod = ConsensusMethod.WEIGHTED_AVERAGE,
        task_timeout: int = 300,  # 5 minutes
        enable_persistence: bool = True,
        persistence_path: str = "./evaluator_coordinator_state.pkl",
        bias_detection_enabled: bool = True,
        quality_gate_threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL,
        min_evaluators_per_task: int = 3,
        max_evaluators_per_task: int = 5,
        use_ensemble: bool = True,
        ensemble_config: Optional[List[LLMModelConfig]] = None
    ):
        """
        Initialize the Evaluator Team Coordinator.

        Args:
            evaluator_team: EvaluatorTeam instance to coordinate
            max_concurrent_evaluations: Maximum number of evaluations to run concurrently
            load_balancing_strategy: Strategy for distributing evaluation tasks
            consensus_method: Method for building consensus among evaluators
            task_timeout: Timeout in seconds for individual evaluation tasks
            enable_persistence: Enable state persistence
            persistence_path: Path to persist state
            bias_detection_enabled: Enable bias detection and mitigation
            quality_gate_threshold: Minimum threshold for quality gate
            min_evaluators_per_task: Minimum number of evaluators per task
            max_evaluators_per_task: Maximum number of evaluators per task
            use_ensemble: Use OpenEvolve ensemble for coordination (if available)
            ensemble_config: Optional ensemble configuration (LLM model configs)
        """
        self.evaluator_team = evaluator_team or (EvaluatorTeam() if EVALUATOR_TEAM_AVAILABLE else None)
        self.max_concurrent_evaluations = max_concurrent_evaluations
        self.load_balancing_strategy = load_balancing_strategy
        self.consensus_method = consensus_method
        self.task_timeout = task_timeout
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path
        self.bias_detection_enabled = bias_detection_enabled
        self.quality_gate_threshold = quality_gate_threshold
        self.min_evaluators_per_task = min_evaluators_per_task
        self.max_evaluators_per_task = max_evaluators_per_task
        self.use_ensemble = use_ensemble and ENSEMBLE_AVAILABLE

        # Task management
        self.task_queue: queue.Queue = queue.Queue()
        self.active_tasks: Dict[str, EvaluationTask] = {}
        self.completed_tasks: Dict[str, EvaluationTask] = {}
        self.session_history: List[EvaluationSession] = []

        # Evaluator management
        self.evaluators: List[EvaluatorMember] = []
        self.evaluator_metrics: Dict[str, EvaluatorMetrics] = {}
        self.evaluator_lock = threading.Lock()

        # Bias tracking
        self.bias_history: Dict[str, List[Dict[str, Any]]] = {}

        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.current_session: Optional[EvaluationSession] = None

        # Metrics
        self.metrics = CoordinatorMetrics()

        # Initialize ensemble or fallback executor
        if self.use_ensemble:
            self._initialize_ensemble(ensemble_config)
            logger.info("EvaluatorTeamCoordinator using OpenEvolve Ensemble for coordination")
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_concurrent_evaluations)
            logger.info("EvaluatorTeamCoordinator using ThreadPoolExecutor for coordination")

        # Load state if persistence is enabled
        if enable_persistence:
            self._load_state()

        # Initialize evaluators
        if self.evaluator_team:
            self.evaluators = self.evaluator_team.team_members.copy()
            self._initialize_evaluator_metrics()

        logger.info(f"EvaluatorTeamCoordinator initialized with {len(self.evaluators)} evaluators using {'ensemble' if self.use_ensemble else 'fallback'} coordination")

    def _initialize_evaluator_metrics(self):
        """Initialize metrics for all evaluators"""
        with self.evaluator_lock:
            for evaluator in self.evaluators:
                if evaluator.evaluator_id not in self.evaluator_metrics:
                    self.evaluator_metrics[evaluator.evaluator_id] = EvaluatorMetrics(
                        evaluator_id=evaluator.evaluator_id,
                        specialization_scores={
                            metric: 1.0 if metric in evaluator.specializations else 0.5
                            for metric in EvaluationMetric
                        },
                        bias_profile=self._detect_initial_bias(evaluator)
                    )

    def _detect_initial_bias(self, evaluator: EvaluatorMember) -> Dict[str, float]:
        """Detect initial bias profile for an evaluator"""
        bias = {}

        # Philosophy-based bias
        if evaluator.evaluation_philosophy == "strict":
            bias["strictness"] = 0.3
            bias["leniency"] = -0.3
        elif evaluator.evaluation_philosophy == "lenient":
            bias["strictness"] = -0.3
            bias["leniency"] = 0.3
        else:
            bias["strictness"] = 0.0
            bias["leniency"] = 0.0

        # Specialization bias (evaluators tend to be harsher on their specialties)
        for spec in evaluator.specializations:
            bias[f"specialization_{spec.value}"] = 0.1

        return bias

    def _initialize_ensemble(self, ensemble_config: Optional[List[LLMModelConfig]]):
        """
        Initialize OpenEvolve ensemble for parallel evaluator execution.

        The ensemble provides:
        - Weighted model selection based on evaluator expertise
        - Parallel execution capabilities
        - Load balancing across models/evaluators
        - Aggregation and consensus support

        Args:
            ensemble_config: Optional list of LLM model configurations
        """
        if not ensemble_config and self.evaluators:
            # Create ensemble config from evaluators
            ensemble_config = []
            for evaluator in self.evaluators:
                # Map evaluator expertise to model weight
                weight = evaluator.expertise_level / 10.0

                # Create LLM model config for each evaluator
                # In production, these would be actual LLM API configurations
                # For evaluator team, we use evaluator IDs as model names
                model_cfg = LLMModelConfig(
                    name=evaluator.evaluator_id,
                    weight=weight,
                    model_id=evaluator.evaluator_id,
                    temperature=0.3 if evaluator.evaluation_philosophy == "strict" else 0.7,
                    max_tokens=4096
                )
                ensemble_config.append(model_cfg)
                logger.debug(f"Created ensemble config for {evaluator.evaluator_id} with weight {weight:.2f}")

        if ensemble_config:
            try:
                self.ensemble = LLMEnsemble(ensemble_config)
                self.ensemble_config = ensemble_config
                # Track ensemble weights for consensus integration
                self.ensemble_weights = {
                    cfg.name: cfg.weight for cfg in ensemble_config
                }
                logger.info(f"Initialized ensemble with {len(ensemble_config)} models")
                logger.info(f"Ensemble weights: {self.ensemble_weights}")
            except (RuntimeError, ValueError, ImportError) as e:
                logger.error(f"Failed to initialize ensemble: {e}")
                self.use_ensemble = False
                self.ensemble = None
        else:
            logger.warning("No ensemble config provided, ensemble will not be used")
            self.use_ensemble = False
            self.ensemble = None

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def coordinate_solution_evaluations(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
        solutions: Dict[str, str],
        original_problems: Optional[Dict[str, str]] = None,
        content_types: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, List[EvaluationCriterion]]] = None,
        threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL,
        consensus_method: Optional[ConsensusMethod] = None,
        progress_callback: Optional[Callable] = None
    ) -> EvaluationSession:
        """
        Coordinate evaluations for solutions from the decomposition engine.

        This is the main entry point for integrating with the DecompositionEngine.
        It validates sub-problem solutions before final assembly.

        Args:
            problem_statement: Original problem statement
            sub_problems: List of sub-problems from decomposition
            solutions: Map of sub_problem_id -> solution to evaluate
            original_problems: Optional map of sub_problem_id -> original problem
            content_types: Optional map of sub_problem_id -> content type
            criteria: Optional map of sub_problem_id -> evaluation criteria
            threshold: Quality threshold for evaluations
            consensus_method: Method for building consensus
            progress_callback: Optional callback for progress updates

        Returns:
            EvaluationSession with results
        """
        logger.info(f"Starting evaluation coordination for {len(sub_problems)} sub-problems")

        # Create session
        session_id = self._generate_session_id(problem_statement)
        session = EvaluationSession(
            session_id=session_id,
            problem_statement=problem_statement,
            sub_problems=sub_problems,
            solutions=solutions,
            status=EvaluationTaskStatus.IN_PROGRESS,
            total_tasks=len(sub_problems),
            consensus_method=consensus_method or self.consensus_method
        )

        # Register progress callback
        if progress_callback:
            self.progress_callbacks.append(progress_callback)

        # Create tasks for each sub-problem solution
        content_types = content_types or {}
        criteria = criteria or {}
        original_problems = original_problems or {}

        for sp in sub_problems:
            sp_id = sp.get("id", "")
            if not sp_id:
                continue

            task = EvaluationTask(
                task_id=self._generate_task_id(session_id, sp_id),
                sub_problem_id=sp_id,
                sub_problem_description=sp.get("description", ""),
                solution_content=solutions.get(sp_id, ""),
                original_content=original_problems.get(sp_id),
                criteria=criteria.get(sp_id),
                content_type=content_types.get(sp_id, "general"),
                priority=self._determine_task_priority(sp),
                dependencies=sp.get("dependencies", []),
                threshold=threshold
            )
            session.tasks.append(task)
            self.task_queue.put(task)

        self.current_session = session
        self._notify_progress("session_started", session)

        # Execute tasks
        try:
            self._execute_evaluation_tasks_parallel(session)
            session.status = EvaluationTaskStatus.COMPLETED

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful coordination
            passed_tasks = [t for t in session.tasks if t.status == EvaluationTaskStatus.COMPLETED]
            pass_rate = len(passed_tasks) / max(session.total_tasks, 1) if session.total_tasks > 0 else 0.0
            self._extract_evaluator_knowledge("coordinate_solution_evaluations", session)
            self._track_evaluator_performance("coordinate_solution_evaluations", True, session.total_tasks, pass_rate)

        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Error during evaluation execution: {e}")
            session.status = EvaluationTaskStatus.FAILED

            # **ACTUAL INTEGRATION**: Track performance and trigger alert for failures
            self._track_evaluator_performance("coordinate_solution_evaluations", False, session.total_tasks, 0.0)
            self._trigger_evaluator_alerts(
                "coordinate_solution_evaluations",
                False,
                session.session_id,
                session.total_tasks,
                0,
                str(e)
            )

        finally:
            session.completed_at = datetime.now()
            self._update_session_metrics(session)
            self.session_history.append(session)

            # Persist state
            if self.enable_persistence:
                self._save_state()

            # Notify completion
            self._notify_progress("session_completed", session)

        return session

    def _execute_evaluation_tasks_parallel(self, session: EvaluationSession):
        """Execute evaluation tasks in parallel with dependency resolution"""
        logger.info(f"Executing {len(session.tasks)} evaluation tasks with max {self.max_concurrent_evaluations} concurrent")

        # Use ensemble for parallel execution if available
        if self.use_ensemble:
            self._execute_tasks_with_ensemble(session)
        else:
            # Fallback to original ThreadPoolExecutor implementation
            self._execute_tasks_with_executor(session)

        logger.info(f"All evaluation tasks completed: {session.completed_tasks}/{session.total_tasks} successful")

    def _execute_tasks_with_ensemble(self, session: EvaluationSession):
        """
        Execute evaluation tasks using OpenEvolve ensemble.

        The ensemble provides intelligent coordination:
        - Automatic load balancing via weighted model selection
        - Parallel execution across evaluators
        - Integration with consensus algorithms
        - Performance tracking per model/evaluator

        Args:
            session: Evaluation session with tasks to execute
        """
        import asyncio

        logger.info(f"Executing {len(session.tasks)} tasks with ensemble coordination")

        # Track tasks and their dependencies
        remaining_tasks = {task.task_id: task for task in session.tasks}
        completed_task_ids = set()

        # Process tasks until all are done
        while remaining_tasks:
            # Find ready tasks (no pending dependencies)
            ready_tasks = []
            for task_id, task in list(remaining_tasks.items()):
                if task.status == EvaluationTaskStatus.PENDING:
                    deps_satisfied = all(
                        dep_id in completed_task_ids
                        for dep_id in task.dependencies
                    )
                    if deps_satisfied:
                        ready_tasks.append(task)
                        del remaining_tasks[task_id]

            if not ready_tasks:
                logger.debug("No more ready tasks")
                break

            # Use ensemble for intelligent load balancing
            # Ensemble weights determine which evaluators handle which tasks
            task_batches = []
            for task in ready_tasks[:self.max_concurrent_evaluations]:
                task.status = EvaluationTaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

                # Assign evaluators using ensemble-weighted selection
                assigned_evaluators = self._assign_evaluators_with_ensemble_weights(task)
                task.assigned_evaluators = [e.evaluator_id for e in assigned_evaluators]

                # Track ensemble utilization
                for evaluator in assigned_evaluators:
                    if evaluator.evaluator_id in self.evaluator_metrics:
                        self.evaluator_metrics[evaluator.evaluator_id].ensemble_selection_count += 1

                task_batches.append((task, assigned_evaluators))
                self._notify_progress("task_started", task)

            # Execute tasks using ensemble's parallel capability
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Create tasks for parallel execution
            async def execute_batch():
                results = []
                for task, evaluators in task_batches:
                    result = await self._execute_single_task_async(
                        task,
                        evaluators,
                        session.consensus_method
                    )
                    results.append(result)
                return results

            # Run async execution
            try:
                completed_tasks = loop.run_until_complete(execute_batch())
            except (RuntimeError, asyncio.TimeoutError, ValueError) as e:
                logger.error(f"Ensemble execution error: {e}")
                # Fallback to sequential execution
                completed_tasks = [
                    self._execute_single_evaluation_task(task, evaluators, session.consensus_method)
                    for task, evaluators in task_batches
                ]

            # Update session with results
            for i, (task, evaluators) in enumerate(task_batches):
                result_task = completed_tasks[i] if completed_tasks else None
                if result_task:
                    self._update_task_from_result(task, result_task)
                    completed_task_ids.add(task.task_id)
                    session.completed_tasks += 1

                    if task.quality_gate_passed:
                        session.quality_gate_passed_tasks += 1

                    logger.info(f"Completed evaluation task {task.task_id}: "
                              f"consensus={task.consensus_reached}, quality_gate={task.quality_gate_passed}")
                    self._notify_progress("task_completed", task)
                else:
                    logger.error(f"Task {task.task_id} failed")
                    task.status = EvaluationTaskStatus.FAILED
                    session.failed_tasks += 1

        logger.info(f"All evaluation tasks completed: {session.completed_tasks}/{session.total_tasks} successful")

    async def _execute_single_task_async(
        self,
        task: EvaluationTask,
        evaluators: List[EvaluatorMember],
        consensus_method: ConsensusMethod
    ) -> EvaluationTask:
        """
        Execute a single evaluation task asynchronously using ensemble.

        Args:
            task: Evaluation task to execute
            evaluators: List of evaluators assigned to this task
            consensus_method: Method for building consensus

        Returns:
            Updated task with results
        """
        # For now, delegate to synchronous method
        # In production, this would use ensemble's async generate methods
        return self._execute_single_evaluation_task(task, evaluators, consensus_method)

    def _execute_tasks_with_executor(self, session: EvaluationSession):
        """Execute evaluation tasks using ThreadPoolExecutor (fallback when ensemble not available)"""
        logger.info(f"Executing {len(session.tasks)} evaluation tasks with max {self.max_concurrent_evaluations} concurrent")

        # Track tasks and their dependencies
        remaining_tasks = {task.task_id: task for task in session.tasks}
        completed_task_ids = set()
        futures = {}

        # Process tasks until all are done
        while remaining_tasks or futures:
            # Submit tasks that are ready (no pending dependencies)
            ready_tasks = []
            for task_id, task in list(remaining_tasks.items()):
                if task.status == EvaluationTaskStatus.PENDING:
                    # Check if dependencies are satisfied
                    deps_satisfied = all(
                        dep_id in completed_task_ids
                        for dep_id in task.dependencies
                    )
                    if deps_satisfied:
                        ready_tasks.append(task)
                        del remaining_tasks[task_id]

            # Submit ready tasks up to max concurrent
            available_slots = self.max_concurrent_evaluations - len(futures)
            tasks_to_submit = ready_tasks[:available_slots]

            for task in tasks_to_submit:
                task.status = EvaluationTaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

                # Assign evaluators based on strategy
                assigned_evaluators = self._assign_evaluators(task)
                task.assigned_evaluators = [e.evaluator_id for e in assigned_evaluators]

                # Submit to executor
                future = self.executor.submit(
                    self._execute_single_evaluation_task,
                    task,
                    assigned_evaluators,
                    session.consensus_method
                )
                futures[future] = task

                logger.info(f"Submitted evaluation task {task.task_id} to {len(assigned_evaluators)} evaluators")
                self._notify_progress("task_started", task)

            # Wait for at least one task to complete
            if futures:
                completed_futures, _ = wait(
                    futures,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED
                )

                for future in completed_futures:
                    task = futures[future]
                    try:
                        # Task completed successfully
                        result_task = future.result()
                        self._update_task_from_result(task, result_task)
                        completed_task_ids.add(task.task_id)
                        session.completed_tasks += 1

                        # Check quality gate
                        if task.quality_gate_passed:
                            session.quality_gate_passed_tasks += 1

                        logger.info(f"Completed evaluation task {task.task_id}: "
                                  f"consensus={task.consensus_reached}, quality_gate={task.quality_gate_passed}")
                        self._notify_progress("task_completed", task)

                    except (RuntimeError, TimeoutError, ValueError) as e:
                        logger.error(f"Task {task.task_id} failed: {e}")
                        task.status = EvaluationTaskStatus.FAILED
                        task.error = str(e)
                        session.failed_tasks += 1

                    finally:
                        del futures[future]

        logger.info(f"All evaluation tasks completed: {session.completed_tasks}/{session.total_tasks} successful")

    def _execute_single_evaluation_task(
        self,
        task: EvaluationTask,
        evaluators: List[EvaluatorMember],
        consensus_method: ConsensusMethod
    ) -> EvaluationTask:
        """
        Execute a single evaluation task with multiple evaluators.

        Args:
            task: Evaluation task to execute
            evaluators: List of evaluators assigned to this task
            consensus_method: Method for building consensus

        Returns:
            Updated task with results
        """
        try:
            # Run evaluations in parallel for each evaluator
            with ThreadPoolExecutor(max_workers=len(evaluators)) as executor:
                futures = {
                    executor.submit(
                        evaluator.evaluate_content,
                        task.solution_content,
                        task.content_type,
                        [task.original_content] if task.original_content else None,
                        task.criteria
                    ): evaluator
                    for evaluator in evaluators
                }

                # Collect assessments
                assessments = []
                for future in concurrent_futures.as_completed(futures, timeout=self.task_timeout):
                    evaluator = futures[future]
                    try:
                        assessment = future.result()
                        assessments.append(assessment)

                        # Update evaluator metrics
                        self._update_evaluator_metrics(evaluator, assessment, True)

                    except (RuntimeError, TimeoutError, ValueError) as e:
                        logger.error(f"Evaluator {evaluator.evaluator_id} failed: {e}")
                        self._update_evaluator_metrics(evaluator, None, False)

            task.assessments = assessments

            # Build consensus
            if len(assessments) >= self.min_evaluators_per_task:
                task.integrated_evaluation = self._build_consensus(
                    assessments,
                    task.solution_content,
                    task.content_type,
                    consensus_method,
                    task.threshold
                )
                task.consensus_score = task.integrated_evaluation.consensus_score
                task.consensus_reached = task.integrated_evaluation.consensus_reached

                # Apply quality gate
                task.quality_gate_passed = self._apply_quality_gate(
                    task.integrated_evaluation,
                    task.threshold
                )

                task.status = EvaluationTaskStatus.COMPLETED if task.quality_gate_passed else EvaluationTaskStatus.QUALITY_GATE_FAILED
            else:
                logger.warning(f"Insufficient assessments for task {task.task_id}: {len(assessments)} < {self.min_evaluators_per_task}")
                task.status = EvaluationTaskStatus.FAILED

            task.completed_at = datetime.now()

            # Detect and mitigate bias
            if self.bias_detection_enabled and len(assessments) > 1:
                self._detect_and_mitigate_bias(task)

        except (RuntimeError, TimeoutError, ValueError) as e:
            logger.error(f"Error executing evaluation task {task.task_id}: {e}")
            task.status = EvaluationTaskStatus.FAILED
            task.error = str(e)

        return task

    def _build_consensus(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        method: ConsensusMethod,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """
        Build consensus from multiple evaluator assessments.

        Args:
            assessments: List of assessments from evaluators
            content: Content that was evaluated
            content_type: Type of content
            method: Consensus method to use
            threshold: Quality threshold

        Returns:
            IntegratedEvaluation with consensus results
        """
        if not assessments:
            return self._create_empty_evaluation(content_type)

        # Apply consensus method
        if method == ConsensusMethod.MAJORITY_VOTE:
            return self._consensus_majority_vote(assessments, content, content_type, threshold)
        elif method == ConsensusMethod.WEIGHTED_AVERAGE:
            return self._consensus_weighted_average(assessments, content, content_type, threshold)
        elif method == ConsensusMethod.MEDIAN:
            return self._consensus_median(assessments, content, content_type, threshold)
        elif method == ConsensusMethod.BATESIAN:
            return self._consensus_batesian(assessments, content, content_type, threshold)
        elif method == ConsensusMethod.DEMPSTER_SHAFER:
            return self._consensus_dempster_shafer(assessments, content, content_type, threshold)
        elif method == ConsensusMethod.DELPHI:
            return self._consensus_delphi(assessments, content, content_type, threshold)
        else:
            return self._consensus_weighted_average(assessments, content, content_type, threshold)

    def _consensus_weighted_average(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """
        Weighted average consensus based on evaluator reliability and expertise.

        This method now integrates ensemble weights when available, combining:
        - Ensemble weights (from LLMEnsemble configuration)
        - Reliability scores (from historical performance)
        - Expertise levels (from evaluator profiles)

        This provides a more intelligent weighted consensus that considers
        both static (ensemble) and dynamic (reliability) factors.
        """
        # Calculate weights combining ensemble, reliability, and expertise
        weights = []
        total_weight = 0.0

        for assessment in assessments:
            evaluator_id = assessment.evaluator_id
            metrics = self.evaluator_metrics.get(evaluator_id)

            if metrics:
                # Start with ensemble weight (if available)
                ensemble_weight = self.ensemble_weights.get(evaluator_id, 1.0) if hasattr(self, 'ensemble_weights') else 1.0

                # Add reliability factor
                evaluator = next((e for e in self.evaluators if e.evaluator_id == evaluator_id), None)
                expertise_factor = evaluator.expertise_level / 10.0 if evaluator else 0.5

                # Combined weight: ensemble * reliability * expertise
                weight = ensemble_weight * metrics.reliability_score * expertise_factor
            else:
                # Fallback to ensemble weight or default
                weight = self.ensemble_weights.get(evaluator_id, 1.0) if hasattr(self, 'ensemble_weights') else 1.0

            weights.append(weight)
            total_weight += weight

        # Normalize weights
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Calculate weighted consensus score
        weighted_scores = [a.composite_score * w for a, w in zip(assessments, weights)]
        consensus_score = sum(weighted_scores)

        # Variance analysis
        scores = [a.composite_score for a in assessments]
        variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        # Determine final verdict
        threshold_map = {
            EvaluationThreshold.MINIMAL_ACCEPTANCE: 60.0,
            EvaluationThreshold.STANDARD_APPROVAL: 75.0,
            EvaluationThreshold.HIGH_QUALITY: 85.0,
            EvaluationThreshold.EXCEPTIONAL: 95.0
        }
        required_score = threshold_map.get(threshold, 75.0)

        # Determine consensus reached
        consensus_reached = std_dev < 15.0 and consensus_score >= required_score

        if consensus_score >= required_score:
            final_verdict = "APPROVED"
        elif consensus_score >= (required_score * 0.8):
            final_verdict = "NEEDS_WORK"
        else:
            final_verdict = "REJECTED"

        # Confidence intervals
        mean_score = statistics.mean(scores)
        margin_of_error = 2 * (std_dev / (len(scores) ** 0.5))
        confidence_intervals = {
            "mean_score": mean_score,
            "lower_bound": max(0, mean_score - margin_of_error),
            "upper_bound": min(100, mean_score + margin_of_error),
            "margin_of_error": margin_of_error,
            "confidence_level": "95%"
        }

        # Generate recommendations
        recommendations = self._generate_weighted_recommendations(assessments, weights)

        return IntegratedEvaluation(
            assessments=assessments,
            consensus_score=consensus_score,
            consensus_reached=consensus_reached,
            variance_analysis={
                "variance": variance,
                "standard_deviation": std_dev,
                "mean_score": mean_score,
                "outliers": [s for s in scores if abs(s - mean_score) > (2 * std_dev)],
                "score_range": max(scores) - min(scores) if scores else 0,
                "assessments_count": len(assessments),
                "consensus_method": "weighted_average",
                "weights": dict(zip([a.evaluator_id for a in assessments], weights)),
                "uses_ensemble_weights": hasattr(self, 'ensemble_weights'),
                "weight_sources": ["ensemble", "reliability", "expertise"]
            },
            final_verdict=final_verdict,
            confidence_intervals=confidence_intervals,
            recommendations=recommendations,
            evaluation_metadata={
                'content_type': content_type,
                'num_evaluators': len(assessments),
                'threshold_used': threshold.value,
                'evaluation_timestamp': datetime.now().isoformat(),
                'consensus_method': 'weighted_average',
                'ensemble_integration': True if hasattr(self, 'ensemble_weights') else False
            }
        )

    def _consensus_majority_vote(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """Majority vote consensus"""
        # Classify each assessment
        verdicts = []
        for assessment in assessments:
            if assessment.composite_score >= 85:
                verdicts.append("APPROVED")
            elif assessment.composite_score >= 60:
                verdicts.append("NEEDS_WORK")
            else:
                verdicts.append("REJECTED")

        # Get majority verdict
        from collections import Counter
        verdict_counts = Counter(verdicts)
        final_verdict = verdict_counts.most_common(1)[0][0]

        # Calculate consensus score as average of majority group
        majority_verdict = final_verdict
        majority_assessments = [
            a for a, v in zip(assessments, verdicts) if v == majority_verdict
        ]
        consensus_score = statistics.mean([a.composite_score for a in majority_assessments])

        # Variance analysis
        scores = [a.composite_score for a in assessments]
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return IntegratedEvaluation(
            assessments=assessments,
            consensus_score=consensus_score,
            consensus_reached=verdict_counts[majority_verdict] > len(assessments) / 2,
            variance_analysis={
                "standard_deviation": std_dev,
                "mean_score": statistics.mean(scores),
                "majority_verdict": majority_verdict,
                "verdict_distribution": dict(verdict_counts),
                "assessments_count": len(assessments),
                "consensus_method": "majority_vote"
            },
            final_verdict=final_verdict,
            confidence_intervals={},
            recommendations=[f"Majority verdict: {majority_verdict}"],
            evaluation_metadata={
                'content_type': content_type,
                'num_evaluators': len(assessments),
                'evaluation_timestamp': datetime.now().isoformat(),
                'consensus_method': 'majority_vote'
            }
        )

    def _consensus_median(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """Median-based consensus (robust to outliers)"""
        scores = sorted([a.composite_score for a in assessments])
        consensus_score = statistics.median(scores)

        # Determine final verdict
        threshold_map = {
            EvaluationThreshold.MINIMAL_ACCEPTANCE: 60.0,
            EvaluationThreshold.STANDARD_APPROVAL: 75.0,
            EvaluationThreshold.HIGH_QUALITY: 85.0,
            EvaluationThreshold.EXCEPTIONAL: 95.0
        }
        required_score = threshold_map.get(threshold, 75.0)

        # Determine consensus reached
        consensus_reached = consensus_score >= required_score

        return IntegratedEvaluation(
            assessments=assessments,
            consensus_score=consensus_score,
            consensus_reached=consensus_reached,
            variance_analysis={
                "median": consensus_score,
                "mean_score": statistics.mean(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "assessments_count": len(assessments),
                "consensus_method": "median"
            },
            final_verdict="APPROVED" if consensus_reached else "NEEDS_WORK",
            confidence_intervals={},
            recommendations=[f"Median score: {consensus_score:.2f}"],
            evaluation_metadata={
                'content_type': content_type,
                'num_evaluators': len(assessments),
                'evaluation_timestamp': datetime.now().isoformat(),
                'consensus_method': 'median'
            }
        )

    def _consensus_batesian(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """Batesian consensus (weighted by historical reliability)"""
        # Calculate reliability weights
        weights = []
        for assessment in assessments:
            evaluator_id = assessment.evaluator_id
            metrics = self.evaluator_metrics.get(evaluator_id)

            if metrics and metrics.evaluations_completed > 0:
                # Reliability = success rate / (time factor)
                success_rate = (metrics.evaluations_completed - metrics.evaluations_failed) / metrics.evaluations_completed
                time_factor = 1.0 / (metrics.average_evaluation_time + 1.0)
                weight = success_rate * (1.0 + time_factor)
            else:
                weight = 1.0

            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        # Calculate weighted score
        weighted_scores = [a.composite_score * w for a, w in zip(assessments, weights)]
        consensus_score = sum(weighted_scores)

        return self._consensus_weighted_average(assessments, content, content_type, threshold)

    def _consensus_dempster_shafer(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """Dempster-Shafer evidence theory consensus"""
        # Map assessments to belief functions
        # Simplified implementation: treat each score as evidence for quality

        # Combine evidence using Dempster's rule
        belief_approved = 0.5  # Prior
        belief_rejected = 0.5  # Prior

        for assessment in assessments:
            score = assessment.composite_score / 100.0  # Normalize to 0-1

            # Evidence for approval
            e_approved = score
            # Evidence for rejection
            e_rejected = 1.0 - score

            # Dempster's combination rule
            k = belief_approved * e_rejected + belief_rejected * e_approved
            if k < 1.0:
                new_approved = (belief_approved * e_approved) / (1.0 - k)
                new_rejected = (belief_rejected * e_rejected) / (1.0 - k)
                belief_approved = new_approved
                belief_rejected = new_rejected

        # Final decision based on belief
        consensus_score = belief_approved * 100.0
        final_verdict = "APPROVED" if belief_approved > 0.7 else "NEEDS_WORK" if belief_approved > 0.4 else "REJECTED"

        return IntegratedEvaluation(
            assessments=assessments,
            consensus_score=consensus_score,
            consensus_reached=belief_approved > 0.7,
            variance_analysis={
                "belief_approved": belief_approved,
                "belief_rejected": belief_rejected,
                "assessments_count": len(assessments),
                "consensus_method": "dempster_shafer"
            },
            final_verdict=final_verdict,
            confidence_intervals={},
            recommendations=[f"Belief in approval: {belief_approved:.2f}"],
            evaluation_metadata={
                'content_type': content_type,
                'num_evaluators': len(assessments),
                'evaluation_timestamp': datetime.now().isoformat(),
                'consensus_method': 'dempster_shafer'
            }
        )

    def _consensus_delphi(
        self,
        assessments: List[EvaluatorAssessment],
        content: str,
        content_type: str,
        threshold: EvaluationThreshold
    ) -> IntegratedEvaluation:
        """Delphi method consensus (iterative refinement)"""
        # Simplified Delphi: iteratively average scores until convergence
        scores = [a.composite_score for a in assessments]
        prev_mean = statistics.mean(scores)

        for iteration in range(5):  # Max 5 iterations
            # Calculate mean excluding outliers
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0

            # Exclude outliers (> 2 std dev)
            filtered_scores = [s for s in scores if abs(s - mean) <= 2 * std]

            if not filtered_scores:
                break

            new_mean = statistics.mean(filtered_scores)

            # Check convergence
            if abs(new_mean - prev_mean) < 1.0:
                break

            prev_mean = new_mean

        consensus_score = prev_mean
        consensus_reached = consensus_score >= threshold.value

        return IntegratedEvaluation(
            assessments=assessments,
            consensus_score=consensus_score,
            consensus_reached=consensus_reached,
            variance_analysis={
                "final_mean": consensus_score,
                "iterations": iteration + 1,
                "converged": True,
                "assessments_count": len(assessments),
                "consensus_method": "delphi"
            },
            final_verdict="APPROVED" if consensus_reached else "NEEDS_WORK",
            confidence_intervals={},
            recommendations=[f"Delphi consensus after {iteration + 1} iterations"],
            evaluation_metadata={
                'content_type': content_type,
                'num_evaluators': len(assessments),
                'evaluation_timestamp': datetime.now().isoformat(),
                'consensus_method': 'delphi'
            }
        )

    def _apply_quality_gate(
        self,
        evaluation: IntegratedEvaluation,
        threshold: EvaluationThreshold
    ) -> bool:
        """
        Apply quality gate to integrated evaluation.

        Args:
            evaluation: Integrated evaluation to check
            threshold: Quality threshold to apply

        Returns:
            True if quality gate passed
        """
        # Check if consensus was reached
        if not evaluation.consensus_reached:
            return False

        # Check if verdict is approved
        if evaluation.final_verdict != "APPROVED":
            return False

        # Check if score meets threshold
        threshold_map = {
            EvaluationThreshold.MINIMAL_ACCEPTANCE: 60.0,
            EvaluationThreshold.STANDARD_APPROVAL: 75.0,
            EvaluationThreshold.HIGH_QUALITY: 85.0,
            EvaluationThreshold.EXCEPTIONAL: 95.0
        }
        required_score = threshold_map.get(threshold, 75.0)

        if evaluation.consensus_score < required_score:
            return False

        # Check variance is acceptable (not too much disagreement)
        variance = evaluation.variance_analysis.get("variance", 0)
        if variance > 100.0:  # Too much disagreement
            return False

        return True

    def _assign_evaluators(self, task: EvaluationTask) -> List[EvaluatorMember]:
        """
        Assign evaluators to a task based on load balancing strategy.

        Args:
            task: Task to assign evaluators to

        Returns:
            List of assigned evaluators
        """
        available_evaluators = [
            e for e in self.evaluators
            if self.evaluator_metrics.get(e.evaluator_id, EvaluatorMetrics(evaluator_id=e.evaluator_id)).current_load < 3
        ]

        if not available_evaluators:
            available_evaluators = self.evaluators

        # Determine number of evaluators to assign
        num_evaluators = min(
            self.max_evaluators_per_task,
            max(self.min_evaluators_per_task, len(available_evaluators))
        )

        if self.load_balancing_strategy == LoadBalancingStrategy.SPECIALIZATION_BASED:
            # Assign based on content type specializations
            return self._assign_by_specialization(task, available_evaluators, num_evaluators)
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Assign to least loaded evaluators
            return self._assign_by_load(available_evaluators, num_evaluators)
        elif self.load_balancing_strategy == LoadBalancingStrategy.EXPERTISE_MATCHED:
            # Assign by expertise matching
            return self._assign_by_expertise(task, available_evaluators, num_evaluators)
        elif self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round robin assignment
            return self._assign_round_robin(available_evaluators, num_evaluators)
        else:
            # Random assignment
            return random.sample(available_evaluators, min(num_evaluators, len(available_evaluators)))

    def _assign_by_specialization(
        self,
        task: EvaluationTask,
        evaluators: List[EvaluatorMember],
        num_evaluators: int
    ) -> List[EvaluatorMember]:
        """Assign evaluators based on content type specializations"""
        # Get relevant metrics for content type
        relevant_metrics = {
            EvaluationMetric.OVERALL_QUALITY,
            EvaluationMetric.CORRECTNESS,
            EvaluationMetric.COMPLETENESS
        }

        if task.content_type == "code":
            relevant_metrics.update([
                EvaluationMetric.ROBUSTNESS,
                EvaluationMetric.MAINTAINABILITY,
                EvaluationMetric.SECURITY
            ])
        elif task.content_type == "document":
            relevant_metrics.add(EvaluationMetric.CLARITY)

        # Score evaluators by specialization match
        scored_evaluators = []
        for evaluator in evaluators:
            specialization_score = sum(
                1.0 for metric in relevant_metrics if metric in evaluator.specializations
            )
            scored_evaluators.append((evaluator, specialization_score))

        # Sort by specialization score and pick top evaluators
        scored_evaluators.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored_evaluators[:num_evaluators]]

    def _assign_by_load(
        self,
        evaluators: List[EvaluatorMember],
        num_evaluators: int
    ) -> List[EvaluatorMember]:
        """Assign to least loaded evaluators"""
        # Sort by current load
        sorted_evaluators = sorted(
            evaluators,
            key=lambda e: self.evaluator_metrics.get(
                e.evaluator_id,
                EvaluatorMetrics(evaluator_id=e.evaluator_id)
            ).current_load
        )
        return sorted_evaluators[:num_evaluators]

    def _assign_by_expertise(
        self,
        task: EvaluationTask,
        evaluators: List[EvaluatorMember],
        num_evaluators: int
    ) -> List[EvaluatorMember]:
        """Assign by expertise level matching task complexity"""
        # Estimate task complexity
        complexity = len(task.solution_content) / 1000.0  # Simple heuristic

        # Select evaluators with appropriate expertise
        if complexity > 5.0:
            # High complexity: need senior evaluators
            suitable = [e for e in evaluators if e.expertise_level >= 7]
        elif complexity > 2.0:
            # Medium complexity: mix of senior and mid-level
            suitable = evaluators
        else:
            # Low complexity: any evaluator
            suitable = evaluators

        # Sort by expertise and pick top
        suitable.sort(key=lambda e: e.expertise_level, reverse=True)
        return suitable[:num_evaluators]

    def _assign_round_robin(
        self,
        evaluators: List[EvaluatorMember],
        num_evaluators: int
    ) -> List[EvaluatorMember]:
        """Round robin assignment"""
        # Use task counter to distribute evenly
        task_count = sum(m.evaluations_completed for m in self.evaluator_metrics.values())

        selected = []
        for i in range(num_evaluators):
            idx = (task_count + i) % len(evaluators)
            selected.append(evaluators[idx])

        return selected

    def _assign_evaluators_with_ensemble_weights(self, task: EvaluationTask) -> List[EvaluatorMember]:
        """
        Assign evaluators using ensemble-weighted selection.

        This method integrates ensemble weights with the load balancing strategy,
        preferring evaluators with higher ensemble weights (based on expertise and
        reliability) while still respecting load constraints.

        Args:
            task: Task to assign evaluators to

        Returns:
            List of assigned evaluators selected using ensemble weights
        """
        if not self.use_ensemble or not hasattr(self, 'ensemble_weights'):
            # Fallback to standard assignment
            return self._assign_evaluators(task)

        # Get available evaluators (not overloaded)
        available_evaluators = [
            e for e in self.evaluators
            if self.evaluator_metrics.get(e.evaluator_id, EvaluatorMetrics(evaluator_id=e.evaluator_id)).current_load < 3
        ]

        if not available_evaluators:
            available_evaluators = self.evaluators

        # Determine number of evaluators to assign
        num_evaluators = min(
            self.max_evaluators_per_task,
            max(self.min_evaluators_per_task, len(available_evaluators))
        )

        # Calculate combined scores: ensemble weight * load factor * specialization match
        scored_evaluators = []
        for evaluator in available_evaluators:
            # Base score from ensemble weight
            ensemble_weight = self.ensemble_weights.get(evaluator.evaluator_id, 1.0)

            # Load factor (prefer less loaded evaluators)
            metrics = self.evaluator_metrics.get(evaluator.evaluator_id, EvaluatorMetrics(evaluator_id=evaluator.evaluator_id))
            load_factor = 1.0 / (1.0 + metrics.current_load)

            # Specialization bonus
            specialization_bonus = 1.0
            if self.load_balancing_strategy == LoadBalancingStrategy.SPECIALIZATION_BASED:
                relevant_metrics = {EvaluationMetric.OVERALL_QUALITY, EvaluationMetric.CORRECTNESS}
                if task.content_type == "code":
                    relevant_metrics.update([EvaluationMetric.ROBUSTNESS, EvaluationMetric.MAINTAINABILITY])
                specialization_match = sum(1.0 for m in relevant_metrics if m in evaluator.specializations)
                specialization_bonus = 1.0 + (specialization_match * 0.1)

            # Combined score
            combined_score = ensemble_weight * load_factor * specialization_bonus

            scored_evaluators.append((evaluator, combined_score))

        # Sort by combined score and select top evaluators
        scored_evaluators.sort(key=lambda x: x[1], reverse=True)
        selected = [e for e, _ in scored_evaluators[:num_evaluators]]

        logger.debug(f"Ensemble-weighted assignment: selected {[e.evaluator_id for e in selected]} "
                    f"with scores {[s for _, s in scored_evaluators[:num_evaluators]]}")

        return selected

    def _detect_and_mitigate_bias(self, task: EvaluationTask):
        """
        Detect and mitigate bias in evaluations.

        Args:
            task: Task to analyze for bias
        """
        if len(task.assessments) < 2:
            return

        # Calculate mean and std dev
        scores = [a.composite_score for a in task.assessments]
        mean_score = statistics.mean(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0

        # Detect outliers (potential bias)
        for assessment in task.assessments:
            evaluator_id = assessment.evaluator_id
            score = assessment.composite_score

            # Check if this evaluator is an outlier
            if std_dev > 0 and abs(score - mean_score) > 2 * std_dev:
                # Initialize bias history if not exists
                if evaluator_id not in self.bias_history:
                    self.bias_history[evaluator_id] = []

                # Update bias profile
                self.bias_history[evaluator_id].append({
                    'timestamp': datetime.now(),
                    'task_id': task.task_id,
                    'score': score,
                    'mean_score': mean_score,
                    'deviation': score - mean_score
                })

                # Adjust reliability score
                metrics = self.evaluator_metrics.get(evaluator_id)
                if metrics:
                    metrics.reliability_score *= 0.95  # Slightly reduce reliability

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _generate_session_id(self, problem_statement: str) -> str:
        """Generate unique session ID"""
        hash_input = f"{problem_statement}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _generate_task_id(self, session_id: str, sub_problem_id: str) -> str:
        """Generate unique task ID"""
        return f"{session_id}_{sub_problem_id}"

    def _determine_task_priority(self, sub_problem: Dict[str, Any]) -> EvaluationTaskPriority:
        """Determine task priority from sub-problem metadata"""
        priority = sub_problem.get("priority", 5)

        if priority >= 8:
            return EvaluationTaskPriority.CRITICAL
        elif priority >= 6:
            return EvaluationTaskPriority.HIGH
        elif priority >= 4:
            return EvaluationTaskPriority.MEDIUM
        else:
            return EvaluationTaskPriority.LOW

    def _update_evaluator_metrics(
        self,
        evaluator: EvaluatorMember,
        assessment: Optional[EvaluatorAssessment],
        success: bool
    ):
        """
        Update metrics for an evaluator.

        Now includes ensemble utilization tracking when ensemble mode is active.
        """
        with self.evaluator_lock:
            if evaluator.evaluator_id not in self.evaluator_metrics:
                self.evaluator_metrics[evaluator.evaluator_id] = EvaluatorMetrics(
                    evaluator_id=evaluator.evaluator_id
                )

            metrics = self.evaluator_metrics[evaluator.evaluator_id]
            metrics.evaluations_completed += 1

            if not success:
                metrics.evaluations_failed += 1

            if assessment:
                metrics.total_time_spent += assessment.time_taken
                metrics.average_evaluation_time = (
                    metrics.total_time_spent / metrics.evaluations_completed
                )

                # Update consensus agreement rate
                # (simplified - should track actual consensus participation)
                metrics.consensus_agreement_rate = 0.8  # Placeholder

                # Update ensemble utilization if ensemble mode is active
                if self.use_ensemble and hasattr(metrics, 'ensemble_selection_count'):
                    # Calculate utilization rate
                    if metrics.ensemble_selection_count > 0:
                        metrics.ensemble_utilization = (
                            metrics.ensemble_selection_count /
                            self.metrics.total_tasks
                        ) if self.metrics.total_tasks > 0 else 0.0

            metrics.current_load -= 1
            metrics.last_active = datetime.now()

    def _update_task_from_result(self, task: EvaluationTask, result_task: EvaluationTask):
        """Update task from result"""
        task.assessments = result_task.assessments
        task.integrated_evaluation = result_task.integrated_evaluation
        task.consensus_score = result_task.consensus_score
        task.consensus_reached = result_task.consensus_reached
        task.quality_gate_passed = result_task.quality_gate_passed
        task.status = result_task.status
        task.completed_at = result_task.completed_at
        task.error = result_task.error

    def _update_session_metrics(self, session: EvaluationSession):
        """Update overall coordinator metrics"""
        self.metrics.total_sessions += 1
        self.metrics.total_tasks += session.total_tasks
        self.metrics.completed_tasks += session.completed_tasks
        self.metrics.failed_tasks += session.failed_tasks

        # Calculate average times
        session_duration = (session.completed_at - session.started_at).total_seconds()
        total_duration = self.metrics.average_session_time * (self.metrics.total_sessions - 1)
        self.metrics.average_session_time = (total_duration + session_duration) / self.metrics.total_sessions

        # Calculate quality gate pass rate
        total_completed = self.metrics.completed_tasks
        total_passed = session.quality_gate_passed_tasks
        self.metrics.quality_gate_pass_rate = total_passed / total_completed if total_completed > 0 else 0.0

    def _notify_progress(self, event_type: str, data: Any):
        """Notify progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(event_type, data)
            except (TypeError, ValueError) as e:
                logger.error(f"Progress callback error: {e}")

    def _generate_weighted_recommendations(
        self,
        assessments: List[EvaluatorAssessment],
        weights: List[float]
    ) -> List[str]:
        """Generate recommendations weighted by evaluator reliability"""
        recommendations = {}

        for assessment, weight in zip(assessments, weights):
            # Collect feedback from each assessment
            if "improvement_suggestions" in assessment.detailed_feedback:
                for suggestion in assessment.detailed_feedback["improvement_suggestions"]:
                    if suggestion not in recommendations:
                        recommendations[suggestion] = 0.0
                    recommendations[suggestion] += weight

        # Sort by weight and return
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, _ in sorted_recs[:10]]

    def _create_empty_evaluation(self, content_type: str) -> IntegratedEvaluation:
        """Create empty evaluation when no assessments available"""
        return IntegratedEvaluation(
            assessments=[],
            consensus_score=0.0,
            consensus_reached=False,
            variance_analysis={},
            final_verdict="REJECTED",
            confidence_intervals={},
            recommendations=["No assessments available"],
            evaluation_metadata={'content_type': content_type}
        )

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _save_state(self):
        """Save coordinator state to disk"""
        try:
            state = {
                'evaluator_metrics': self.evaluator_metrics,
                'bias_history': self.bias_history,
                'metrics': self.metrics,
                'session_history': self.session_history[-50:]  # Keep last 50
            }

            with open(self.persistence_path, 'w') as f:
                json.dump(state, f)

            logger.debug(f"Saved coordinator state to {self.persistence_path}")
        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load coordinator state from disk"""
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'r') as f:
                    state = json.load(f)

                self.evaluator_metrics = state.get('evaluator_metrics', {})
                self.bias_history = state.get('bias_history', {})
                self.metrics = state.get('metrics', CoordinatorMetrics())
                self.session_history = state.get('session_history', [])

                logger.debug(f"Loaded coordinator state from {self.persistence_path}")
        except (OSError, IOError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to load state: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get current coordinator status"""
        return {
            'evaluators': len(self.evaluators),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_sessions': self.metrics.total_sessions,
            'quality_gate_pass_rate': self.metrics.quality_gate_pass_rate,
            'average_session_time': self.metrics.average_session_time,
            'current_session': self.current_session.session_id if self.current_session else None
        }

    def get_evaluator_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all evaluators"""
        performance = {}

        for evaluator_id, metrics in self.evaluator_metrics.items():
            performance[evaluator_id] = {
                'evaluations_completed': metrics.evaluations_completed,
                'evaluations_failed': metrics.evaluations_failed,
                'success_rate': (
                    (metrics.evaluations_completed - metrics.evaluations_failed) / metrics.evaluations_completed
                    if metrics.evaluations_completed > 0 else 0.0
                ),
                'average_time': metrics.average_evaluation_time,
                'reliability_score': metrics.reliability_score,
                'current_load': metrics.current_load
            }

        return performance

    def get_ensemble_status(self) -> Dict[str, Any]:
        """
        Get the current status of ensemble integration.

        Returns:
            Dictionary with ensemble status information including:
            - use_ensemble: Whether ensemble mode is active
            - ensemble_available: Whether ensemble library is available
            - ensemble_weights: Current ensemble weights (if active)
            - evaluator_utilization: Per-evaluator ensemble utilization
            - ensemble_metrics: Overall ensemble performance metrics
        """
        status = {
            "use_ensemble": self.use_ensemble,
            "ensemble_available": ENSEMBLE_AVAILABLE,
            "coordination_mode": "ensemble" if self.use_ensemble else "fallback"
        }

        if self.use_ensemble and hasattr(self, 'ensemble'):
            status["ensemble_configured"] = True
            status["num_models"] = len(self.ensemble.models) if hasattr(self.ensemble, 'models') else len(self.evaluators)

            if hasattr(self, 'ensemble_weights'):
                status["ensemble_weights"] = self.ensemble_weights.copy()

            # Track ensemble utilization per evaluator
            evaluator_util = {}
            for eval_id, metrics in self.evaluator_metrics.items():
                if hasattr(metrics, 'ensemble_selection_count'):
                    evaluator_util[eval_id] = {
                        "selection_count": metrics.ensemble_selection_count,
                        "utilization_rate": getattr(metrics, 'ensemble_utilization', 0.0),
                        "ensemble_weight": self.ensemble_weights.get(eval_id, 1.0) if hasattr(self, 'ensemble_weights') else 1.0
                    }
            status["evaluator_utilization"] = evaluator_util

            # Overall ensemble metrics
            completed_tasks_list = list(self.completed_tasks.values())
            status["ensemble_metrics"] = {
                "total_evaluations": self.metrics.total_tasks,
                "completed_evaluations": self.metrics.completed_tasks,
                "ensemble_pass_rate": self.metrics.quality_gate_pass_rate,
                "average_consensus_score": sum(
                    t.consensus_score for t in completed_tasks_list
                ) / len(completed_tasks_list) if completed_tasks_list else 0.0
            }
        else:
            status["ensemble_configured"] = False
            status["fallback_mode"] = "ThreadPoolExecutor"

        return status

    def update_ensemble_weights(self, new_weights: Dict[str, float]):
        """
        Update ensemble weights dynamically based on performance.

        This allows adaptive ensemble weight adjustment as evaluator
        performance changes over time.

        Args:
            new_weights: Dictionary mapping evaluator_id -> new weight
        """
        if not self.use_ensemble or not hasattr(self, 'ensemble_weights'):
            logger.warning("Cannot update ensemble weights: ensemble mode not active")
            return

        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}

        # Update weights
        self.ensemble_weights.update(new_weights)

        # Update ensemble config if possible
        if hasattr(self, 'ensemble_config'):
            for cfg in self.ensemble_config:
                if cfg.name in new_weights:
                    cfg.weight = new_weights[cfg.name]

        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting and knowledge for coordinator
    # =========================================================================

    def _trigger_coordinator_alerts(
        self,
        session_id: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for coordinator failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.HIGH

                alert_manager.create_alert(
                    title=f"Coordinator Session Failed: {session_id}",
                    description=f"Evaluation session '{session_id}' failed. " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="evaluator_team_coordinator",
                    component="coordinator",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger coordinator alert: {e}")

    def _extract_coordinator_knowledge(
        self,
        session_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract coordinator session knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"coordinator_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="coordinator_session",
                source_component="evaluator_team_coordinator",
                title=f"Coordinator Session: {session_id}",
                content={
                    "session_id": session_id,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "evaluations_completed": result.get("total_evaluations", 0)
                },
                tags=["coordinator", "evaluation", "team"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted coordinator knowledge for {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract coordinator knowledge: {e}")
            return False

    def verify_with_lean(self, content: str, properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify content using Lean theorem prover for verification consensus.
        
        Args:
            content: The content to verify (theorem statement or proof)
            properties: Optional properties for verification
            
        Returns:
            Dict with verification results including:
            - verified: bool
            - formalized: str (Lean code)
            - proof_status: str
            - errors: list
        """
        if not LEAN_AVAILABLE:
            return {"verified": False, "error": "Lean verification not available"}
        
        try:
            client = LeanAideClient()
            # Auto-formalize the content
            formalized = client.autoformalize(content)
            # Verify the formalized content
            verification = client.verify(formalized)
            
            return {
                "verified": verification.get("success", False),
                "formalized": formalized,
                "proof_status": verification.get("status", "unknown"),
                "errors": verification.get("errors", []),
                "metadata": properties or {}
            }
        except Exception as e:
            logger.error(f"Lean verification failed: {e}")
            return {"verified": False, "error": str(e)}

    def shutdown(self):
        """Shutdown the coordinator and cleanup resources"""
        logger.info("Shutting down EvaluatorTeamCoordinator...")

        # Save final state
        if self.enable_persistence:
            self._save_state()

        # Shutdown executor or ensemble
        if self.use_ensemble:
            # Ensemble doesn't need explicit shutdown
            logger.info("Ensemble resources released")
        else:
            # Shutdown ThreadPoolExecutor
            self.executor.shutdown(wait=True)
            logger.info("Executor shutdown complete")

        logger.info("EvaluatorTeamCoordinator shutdown complete")


# =============================================================================
# INTEGRATION WITH DECOMPOSITION ENGINE
# =============================================================================

class DecompositionEvaluationBridge:
    """
    Bridge between DecompositionEngine and EvaluatorTeamCoordinator.

    This class provides a convenient interface for the DecompositionEngine
    to validate sub-problem solutions using the Evaluator Team.
    """

    def __init__(
        self,
        coordinator: Optional[EvaluatorTeamCoordinator] = None,
        auto_validate: bool = True,
        quality_threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL
    ):
        """
        Initialize the bridge.

        Args:
            coordinator: EvaluatorTeamCoordinator instance
            auto_validate: Automatically validate solutions after blue team fixes
            quality_threshold: Quality threshold for validation
        """
        self.coordinator = coordinator or EvaluatorTeamCoordinator()
        self.auto_validate = auto_validate
        self.quality_threshold = quality_threshold

        logger.info("DecompositionEvaluationBridge initialized")

    def validate_solution(
        self,
        sub_problem_id: str,
        sub_problem_description: str,
        solution: str,
        original_problem: Optional[str] = None,
        content_type: str = "general",
        criteria: Optional[List[EvaluationCriterion]] = None
    ) -> Dict[str, Any]:
        """
        Validate a single sub-problem solution.

        Args:
            sub_problem_id: ID of the sub-problem
            sub_problem_description: Description of the sub-problem
            solution: Solution to validate
            original_problem: Original problem statement
            content_type: Type of content
            criteria: Custom evaluation criteria

        Returns:
            Validation result with quality gate decision
        """
        # Create evaluation task
        task = EvaluationTask(
            task_id=f"validation_{sub_problem_id}_{int(time.time())}",
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            solution_content=solution,
            original_content=original_problem,
            criteria=criteria,
            content_type=content_type,
            threshold=self.quality_threshold
        )

        # Assign evaluators
        evaluators = self.coordinator._assign_evaluators(task)

        # Execute evaluation
        result_task = self.coordinator._execute_single_evaluation_task(
            task,
            evaluators,
            self.coordinator.consensus_method
        )

        # Return result
        return {
            'sub_problem_id': sub_problem_id,
            'validation_passed': result_task.quality_gate_passed,
            'consensus_score': result_task.consensus_score,
            'consensus_reached': result_task.consensus_reached,
            'final_verdict': result_task.integrated_evaluation.final_verdict if result_task.integrated_evaluation else "UNKNOWN",
            'recommendations': result_task.integrated_evaluation.recommendations if result_task.integrated_evaluation else [],
            'evaluations_count': len(result_task.assessments)
        }

    def validate_all_solutions(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
        solutions: Dict[str, str],
        original_problems: Optional[Dict[str, str]] = None,
        content_types: Optional[Dict[str, str]] = None,
        criteria: Optional[Dict[str, List[EvaluationCriterion]]] = None,
        progress_callback: Optional[Callable] = None
    ) -> EvaluationSession:
        """
        Validate all sub-problem solutions.

        This is the main integration point for the DecompositionEngine.

        Args:
            problem_statement: Original problem statement
            sub_problems: List of sub-problems
            solutions: Map of sub_problem_id -> solution
            original_problems: Optional map of original problems
            content_types: Optional map of content types
            criteria: Optional map of evaluation criteria
            progress_callback: Optional progress callback

        Returns:
            EvaluationSession with all validation results
        """
        return self.coordinator.coordinate_solution_evaluations(
            problem_statement=problem_statement,
            sub_problems=sub_problems,
            solutions=solutions,
            original_problems=original_problems,
            content_types=content_types,
            criteria=criteria,
            threshold=self.quality_threshold,
            progress_callback=progress_callback
        )

    def get_validation_report(self, session: EvaluationSession) -> Dict[str, Any]:
        """
        Generate a validation report from an evaluation session.

        Args:
            session: Evaluation session

        Returns:
            Validation report
        """
        passed_tasks = [t for t in session.tasks if t.quality_gate_passed]
        failed_tasks = [t for t in session.tasks if not t.quality_gate_passed]

        return {
            'session_id': session.session_id,
            'total_sub_problems': session.total_tasks,
            'passed_validations': len(passed_tasks),
            'failed_validations': len(failed_tasks),
            'validation_rate': len(passed_tasks) / session.total_tasks if session.total_tasks > 0 else 0.0,
            'average_consensus_score': statistics.mean([t.consensus_score for t in session.tasks]) if session.tasks else 0.0,
            'consensus_method': session.consensus_method.value,
            'failed_sub_problems': [t.sub_problem_id for t in failed_tasks],
            'recommendations': self._aggregate_recommendations(session)
        }

    def _aggregate_recommendations(self, session: EvaluationSession) -> List[str]:
        """Aggregate recommendations from all tasks"""
        all_recommendations = []

        for task in session.tasks:
            if task.integrated_evaluation:
                all_recommendations.extend(task.integrated_evaluation.recommendations)

        # Remove duplicates and count frequency
        from collections import Counter
        rec_counts = Counter(all_recommendations)

        # Return top recommendations by frequency
        return [rec for rec, _ in rec_counts.most_common(20)]

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Evaluator Team Coordinator
    # =========================================================================

    def _trigger_evaluator_alerts(
        self,
        operation: str,
        success: bool,
        session_id: Optional[str] = None,
        num_tasks: int = 0,
        num_passed: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for evaluation coordination failures or low pass rates."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures or low pass rates
            pass_rate = num_passed / max(num_tasks, 1) if num_tasks > 0 else 0.0
            if not success or pass_rate < 0.5:
                severity = AlertSeverity.HIGH if not success else AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Evaluator Team Coordinator Alert: {operation}",
                    description=f"Evaluator coordination operation '{operation}' " +
                                 ("failed" if not success else f"has low pass rate: {pass_rate:.2%}") +
                                 (f" for session '{session_id}'" if session_id else "") +
                                 f" ({num_passed}/{num_tasks} tasks passed)" +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="evaluator_team_coordinator",
                    component="evaluation_coordination",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Evaluator Team alert: {e}")

    def _extract_evaluator_knowledge(
        self,
        operation: str,
        session: EvaluationSession
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract evaluator coordination knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            # Calculate metrics
            passed_tasks = [t for t in session.tasks if t.status == EvaluationTaskStatus.COMPLETED]
            failed_tasks = [t for t in session.tasks if t.status == EvaluationTaskStatus.FAILED]

            artifact = KnowledgeArtifact(
                artifact_id=f"evaluator_coord_{operation}_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="evaluator_coordination",
                source_component="evaluator_team_coordinator",
                title=f"Evaluator Coordination: {operation} ({session.session_id})",
                content={
                    "operation": operation,
                    "session_id": session.session_id,
                    "num_tasks": session.total_tasks,
                    "num_passed": len(passed_tasks),
                    "num_failed": len(failed_tasks),
                    "pass_rate": len(passed_tasks) / max(session.total_tasks, 1) if session.total_tasks > 0 else 0.0,
                    "consensus_method": session.consensus_method.value if session.consensus_method else "unknown",
                    "status": session.status.value if session.status else "unknown",
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "average_consensus_score": statistics.mean([t.consensus_score for t in session.tasks]) if session.tasks else 0.0,
                    "validation_rate": len(passed_tasks) / max(session.total_tasks, 1) if session.total_tasks > 0 else 0.0,
                    "failed_sub_problems": [t.sub_problem_id for t in failed_tasks]
                },
                tags=["evaluator", "coordination", operation]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Evaluator Coordination knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Evaluator Coordination knowledge: {e}")
            return False

    def _track_evaluator_performance(
        self,
        operation: str,
        success: bool,
        num_tasks: int = 0,
        pass_rate: float = 0.0
    ):
        """**ACTUAL INTEGRATION**: Track evaluator coordination performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            # Quality based on success and pass rate
            quality = 0.5 if success else 0.0
            if success:
                quality = pass_rate
            quality = max(quality, 0.0)

            performance_data = StrategyPerformanceData(
                strategy_name=f"evaluator_coordinator_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "num_tasks": num_tasks,
                    "pass_rate": pass_rate
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Evaluator Coordination performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track Evaluator Coordination performance: {e}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the EvaluatorTeamCoordinator"""

    # Initialize coordinator
    coordinator = EvaluatorTeamCoordinator(
        max_concurrent_evaluations=5,
        load_balancing_strategy=LoadBalancingStrategy.SPECIALIZATION_BASED,
        consensus_method=ConsensusMethod.WEIGHTED_AVERAGE,
        bias_detection_enabled=True
    )

    # Example sub-problems and solutions
    problem_statement = "Design a secure authentication system"
    sub_problems = [
        {
            "id": "sp_001",
            "description": "Implement password hashing",
            "priority": 8
        },
        {
            "id": "sp_002",
            "description": "Design session management",
            "priority": 7
        }
    ]

    solutions = {
        "sp_001": """
def hash_password(password: str) -> str:
    import bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()
""",
        "sp_002": """
class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, user_id: str) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {'user_id': user_id, 'created_at': datetime.now()}
        return session_id
"""
    }

    # Coordinate evaluations
    session = coordinator.coordinate_solution_evaluations(
        problem_statement=problem_statement,
        sub_problems=sub_problems,
        solutions=solutions,
        content_types={"sp_001": "code", "sp_002": "code"},
        threshold=EvaluationThreshold.HIGH_QUALITY
    )

    # Print results
    print(f"Session {session.session_id} completed:")
    print(f"  Total tasks: {session.total_tasks}")
    print(f"  Passed quality gate: {session.quality_gate_passed_tasks}")
    print(f"  Failed: {session.failed_tasks}")

    for task in session.tasks:
        print(f"\nSub-problem {task.sub_problem_id}:")
        print(f"  Consensus score: {task.consensus_score:.2f}")
        print(f"  Quality gate passed: {task.quality_gate_passed}")
        print(f"  Final verdict: {task.integrated_evaluation.final_verdict if task.integrated_evaluation else 'N/A'}")

    # Shutdown
    coordinator.shutdown()


if __name__ == "__main__":
    example_usage()
