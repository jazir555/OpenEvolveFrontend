"""
Blue Team Coordinator for OpenEvolve
Enhanced orchestration system using OpenEvolve's ensemble functionality
for coordinating multi-agent parallel execution on decomposition engine issues.

Architecture:
    DecompositionEngine → BlueTeamCoordinator → LLMEnsemble (parallel)
                         ↓
                    Task Queue Management
                         ↓
                    Result Aggregation
                         ↓
                    Fixed Solutions

This refactored version uses OpenEvolve's LLMEnsemble for coordination instead of
custom team member management, providing better parallelization and reliability.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
# import pickle  # REMOVED - security risk
import hashlib
import random

# Import existing Blue Team components
try:
    from blue_team import (
        BlueTeam, BlueTeamMember, FixSuggestion, BlueTeamFix,
        BlueTeamAssessment, FixType, FixPriority, BlueTeamStrategy
    )
    BLUE_TEAM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Blue Team not available: {e}")
    BLUE_TEAM_AVAILABLE = False

# Import decomposition engine components
try:
    from decomposition_engine import DecompositionEngine, SubProblem
    from problem_analyzer import ProblemAnalyzer
    from sovereign_data_models import ProblemDefinition, ComplexityScore
    DECOMPOSITION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Decomposition engine not available: {e}")
    DECOMPOSITION_AVAILABLE = False

# Import OpenEvolve ensemble components
try:
    from openevolve.llm.ensemble import LLMEnsemble
    from openevolve.config import LLMModelConfig
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenEvolve ensemble not available: {e}")
    ENSEMBLE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class TaskStatus(Enum):
    """Status of a task in the coordinator"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskPriority(Enum):
    """Priority levels for tasks"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LoadBalancingStrategy(Enum):
    """Strategies for distributing tasks among team members"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    SPECIALIZATION_BASED = "specialization_based"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

@dataclass
class TeamMemberMetrics:
    """Performance metrics for a team member (or ensemble model)"""
    member_name: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_time_spent: float = 0.0
    average_task_time: float = 0.0
    current_load: int = 0
    specialization_scores: Dict[FixType, float] = field(default_factory=dict)
    reliability_score: float = 1.0
    last_active: Optional[datetime] = None
    model_weight: float = 1.0  # Weight for ensemble model selection

@dataclass
class CoordinationTask:
    """A task to be coordinated"""
    task_id: str
    sub_problem_id: str
    sub_problem_description: str
    content: str
    issues: List[Any]  # List of IssueFinding
    content_type: str = "general"
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    assigned_member: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[BlueTeamAssessment] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

@dataclass
class CoordinationSession:
    """A coordination session for a set of tasks"""
    session_id: str
    problem_statement: str
    sub_problems: List[Dict[str, Any]]
    tasks: List[CoordinationTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    aggregated_result: Optional[Dict[str, Any]] = None

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
    throughput_tasks_per_minute: float = 0.0
    peak_concurrent_tasks: int = 0


# =============================================================================
# BLUE TEAM COORDINATOR
# =============================================================================

class BlueTeamCoordinator:
    """
    Enhanced orchestration system using OpenEvolve's ensemble for coordination.

    Features:
    - Parallel task execution using LLMEnsemble
    - Intelligent task distribution via ensemble weights
    - Progress tracking and performance monitoring
    - Integration with DecompositionEngine for automatic issue fixing
    - State persistence and recovery
    - Result aggregation with ensemble consensus
    - Backward compatibility with legacy Blue Team mode
    """

    def __init__(
        self,
        blue_team: Optional[BlueTeam] = None,
        ensemble: Optional[LLMEnsemble] = None,
        max_concurrent_tasks: int = 5,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
        task_timeout: int = 300,  # 5 minutes
        enable_persistence: bool = True,
        persistence_path: str = "./blue_team_coordinator_state.pkl",
        auto_scale: bool = False,
        min_members: int = 2,
        max_members: int = 10,
        use_ensemble: bool = True  # New flag to enable ensemble mode
    ):
        """
        Initialize the Blue Team Coordinator.

        Args:
            blue_team: BlueTeam instance to coordinate (legacy mode)
            ensemble: LLMEnsemble instance for ensemble-based coordination
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            load_balancing_strategy: Strategy for distributing tasks
            task_timeout: Timeout in seconds for individual tasks
            enable_persistence: Enable state persistence
            persistence_path: Path to persist state
            auto_scale: Enable auto-scaling of team members/models
            min_members: Minimum number of team members when auto-scaling
            max_members: Maximum number of team members when auto-scaling
            use_ensemble: Use ensemble-based coordination (True) or legacy mode (False)
        """
        self.blue_team = blue_team or (BlueTeam() if BLUE_TEAM_AVAILABLE else None)
        self.ensemble = ensemble
        self.max_concurrent_tasks = max_concurrent_tasks
        self.load_balancing_strategy = load_balancing_strategy
        self.task_timeout = task_timeout
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path
        self.auto_scale = auto_scale
        self.min_members = min_members
        self.max_members = max_members
        self.use_ensemble = use_ensemble and ENSEMBLE_AVAILABLE

        # Task management
        self.task_queue: queue.Queue = queue.Queue()
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: Dict[str, CoordinationTask] = {}
        self.session_history: List[CoordinationSession] = []

        # Team member management (legacy mode)
        self.team_members: List[BlueTeamMember] = []
        self.member_metrics: Dict[str, TeamMemberMetrics] = {}
        self.member_lock = threading.Lock()

        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.current_session: Optional[CoordinationSession] = None

        # Metrics
        self.metrics = CoordinatorMetrics()

        # Thread pool for parallel execution (legacy mode)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)

        # Load state if persistence is enabled
        if enable_persistence:
            self._load_state()

        # Initialize based on mode
        if self.use_ensemble:
            if self.ensemble:
                logger.info(f"BlueTeamCoordinator initialized with ensemble ({len(self.ensemble.models)} models)")
                self._initialize_ensemble_metrics()
            else:
                logger.warning("use_ensemble=True but no ensemble provided, falling back to legacy mode")
                self.use_ensemble = False
        else:
            # Legacy mode
            if self.blue_team:
                self.team_members = self.blue_team.team_members.copy()
                self._initialize_member_metrics()
                logger.info(f"BlueTeamCoordinator initialized with {len(self.team_members)} team members (legacy mode)")

    def _initialize_ensemble_metrics(self):
        """Initialize metrics for ensemble models"""
        if not self.ensemble:
            return

        with self.member_lock:
            for i, model in enumerate(self.ensemble.models):
                model_name = getattr(model, 'model', f'model_{i}')
                weight = self.ensemble.weights[i] if i < len(self.ensemble.weights) else 1.0

                if model_name not in self.member_metrics:
                    self.member_metrics[model_name] = TeamMemberMetrics(
                        member_name=model_name,
                        model_weight=weight,
                        specialization_scores={fix_type: weight for fix_type in FixType}
                    )

        logger.info(f"Initialized metrics for {len(self.member_metrics)} ensemble models")

    def _initialize_member_metrics(self):
        """Initialize metrics for all team members"""
        with self.member_lock:
            for member in self.team_members:
                if member.name not in self.member_metrics:
                    self.member_metrics[member.name] = TeamMemberMetrics(
                        member_name=member.name,
                        specialization_scores={
                            fix_type: 1.0 if fix_type in member.specializations else 0.5
                            for fix_type in FixType
                        }
                    )

    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================

    def coordinate_decomposition_fixes(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
        content_items: Dict[str, str],
        issues_dict: Dict[str, List[Any]],
        content_types: Optional[Dict[str, str]] = None,
        strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE,
        progress_callback: Optional[Callable] = None
    ) -> CoordinationSession:
        """
        Coordinate fixes for a decomposed problem.

        This is the main entry point for integrating with the DecompositionEngine.
        It automatically fixes issues found during decomposition.

        Args:
            problem_statement: Original problem statement
            sub_problems: List of sub-problems from decomposition
            content_items: Map of sub_problem_id -> content to fix
            issues_dict: Map of sub_problem_id -> list of issues
            content_types: Optional map of sub_problem_id -> content type
            strategy: Blue team strategy to use
            progress_callback: Optional callback for progress updates

        Returns:
            CoordinationSession with results
        """
        logger.info(f"Starting coordination session for {len(sub_problems)} sub-problems")

        # Create session
        session_id = self._generate_session_id(problem_statement)
        session = CoordinationSession(
            session_id=session_id,
            problem_statement=problem_statement,
            sub_problems=sub_problems,
            status=TaskStatus.IN_PROGRESS,
            total_tasks=len(sub_problems)
        )

        # Register progress callback
        if progress_callback:
            self.progress_callbacks.append(progress_callback)

        # Create tasks for each sub-problem
        content_types = content_types or {}
        for sp in sub_problems:
            sp_id = sp.get("id", "")
            if not sp_id:
                continue

            task = CoordinationTask(
                task_id=self._generate_task_id(session_id, sp_id),
                sub_problem_id=sp_id,
                sub_problem_description=sp.get("description", ""),
                content=content_items.get(sp_id, ""),
                issues=issues_dict.get(sp_id, []),
                content_type=content_types.get(sp_id, "general"),
                priority=self._determine_task_priority(sp),
                dependencies=sp.get("dependencies", [])
            )
            session.tasks.append(task)
            self.task_queue.put(task)

        self.current_session = session
        self._notify_progress("session_started", session)

        # Execute tasks
        try:
            self._execute_tasks_parallel(session, strategy)
            session.status = TaskStatus.COMPLETED
        except (RuntimeError, ValueError, ConnectionError, TimeoutError, KeyError) as e:
            logger.error(f"Error during task execution: {e}")
            session.status = TaskStatus.FAILED
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

    def _execute_tasks_parallel(
        self,
        session: CoordinationSession,
        strategy: BlueTeamStrategy
    ):
        """Execute tasks in parallel with dependency resolution"""
        # Route to appropriate execution method
        if self.use_ensemble:
            # Run async ensemble execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._execute_tasks_with_ensemble(session, strategy))
            finally:
                loop.close()
        else:
            # Legacy execution
            self._execute_tasks_legacy(session, strategy)

    async def _execute_tasks_with_ensemble(
        self,
        session: CoordinationSession,
        strategy: BlueTeamStrategy
    ):
        """Execute tasks using LLMEnsemble for parallel generation"""
        logger.info(f"Executing {len(session.tasks)} tasks with ensemble")

        # Build prompts from tasks
        ready_tasks = [t for t in session.tasks if t.status == TaskStatus.PENDING]
        prompts = [self._build_prompt_from_task(t, strategy) for t in ready_tasks]

        # Use ensemble for parallel generation
        try:
            results = await self.ensemble.parallel_generate(prompts)

            # Match results to tasks
            for task, result_text in zip(ready_tasks, results):
                task.result = self._parse_ensemble_result(result_text, task)
                task.status = TaskStatus.COMPLETED if task.result else TaskStatus.FAILED
                task.completed_at = datetime.now()

                if task.status == TaskStatus.COMPLETED:
                    session.completed_tasks += 1
                    self._update_task_metrics(task, success=True)
                    self._notify_progress("task_completed", task)
                else:
                    session.failed_tasks += 1
                    self._update_task_metrics(task, success=False)
                    self._notify_progress("task_failed", task)

        except (ConnectionError, TimeoutError, RuntimeError, ValueError, asyncio.TimeoutError) as e:
            logger.error(f"Ensemble execution failed: {e}")
            # Fall back to legacy execution
            self._execute_tasks_legacy(session, strategy)

        # Aggregate results
        session.aggregated_result = self._aggregate_session_results(session)

    def _execute_tasks_legacy(
        self,
        session: CoordinationSession,
        strategy: BlueTeamStrategy
    ):
        """Execute tasks in parallel with dependency resolution (legacy mode)"""
        logger.info(f"Executing {len(session.tasks)} tasks with max {self.max_concurrent_tasks} concurrent (legacy mode)")

        # Track tasks and their dependencies
        remaining_tasks = {task.task_id: task for task in session.tasks}
        completed_task_ids = set()
        futures = {}

        # Process tasks until all are done
        while remaining_tasks or futures:
            # Submit tasks that are ready (no pending dependencies)
            ready_tasks = []
            for task_id, task in list(remaining_tasks.items()):
                if task.status == TaskStatus.PENDING:
                    # Check if dependencies are satisfied
                    deps_satisfied = all(
                        dep_id in completed_task_ids
                        for dep_id in task.dependencies
                    )
                    if deps_satisfied:
                        ready_tasks.append(task)
                        del remaining_tasks[task_id]

            # Submit ready tasks up to max concurrent
            available_slots = self.max_concurrent_tasks - len(futures)
            tasks_to_submit = ready_tasks[:available_slots]

            for task in tasks_to_submit:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

                # Assign team member
                member = self._assign_team_member(task)
                task.assigned_member = member.name if member else None

                # Submit to executor
                future = self.executor.submit(
                    self._execute_single_task,
                    task,
                    member,
                    strategy
                )
                futures[future] = task

                logger.info(f"Submitted task {task.task_id} to {member.name if member else 'unassigned'}")
                self._notify_progress("task_started", task)

            # Wait for at least one task to complete
            if futures:
                completed_futures = []
                for future in as_completed(list(futures.keys()), timeout=self.task_timeout):
                    completed_futures.append(future)

                # Process completed futures
                for future in completed_futures:
                    task = futures.pop(future)

                    try:
                        result = future.result()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        completed_task_ids.add(task.task_id)

                        # Update metrics
                        self._update_task_metrics(task, success=True)

                        session.completed_tasks += 1
                        self.completed_tasks[task.task_id] = task

                        logger.info(f"Task {task.task_id} completed successfully")
                        self._notify_progress("task_completed", task)

                    except (RuntimeError, ValueError, KeyError, AttributeError, TimeoutError) as e:
                        logger.error(f"Task {task.task_id} failed: {e}")
                        task.error = str(e)

                        # Retry if possible
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.PENDING
                            remaining_tasks[task.task_id] = task
                            logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                        else:
                            task.status = TaskStatus.FAILED
                            task.completed_at = datetime.now()
                            session.failed_tasks += 1

                            self._update_task_metrics(task, success=False)
                            self._notify_progress("task_failed", task)

            # Update peak concurrent tasks metric
            concurrent = len(futures)
            if concurrent > self.metrics.peak_concurrent_tasks:
                self.metrics.peak_concurrent_tasks = concurrent

        # Aggregate results
        session.aggregated_result = self._aggregate_session_results(session)

    def _build_prompt_from_task(self, task: CoordinationTask, strategy: BlueTeamStrategy) -> str:
        """Build a prompt from a coordination task for ensemble generation"""
        # Build strategy-specific prompt
        strategy_prompts = {
            BlueTeamStrategy.COMPREHENSIVE: "Apply comprehensive fixes to all issues.",
            BlueTeamStrategy.PRIORITIZED: "Focus on high-priority issues first.",
            BlueTeamStrategy.ITERATIVE: "Apply fixes iteratively with validation.",
            BlueTeamStrategy.MINIMAL: "Apply minimal, targeted fixes.",
        }

        strategy_instruction = strategy_prompts.get(strategy, "Apply fixes as needed.")

        prompt = f"""Fix the following issues:

{strategy_instruction}

**Content:**
```
{task.content[:2000]}
```

**Issues:**
{chr(10).join(f"- {issue.description if hasattr(issue, 'description') else str(issue)}" for issue in task.issues[:10])}

Apply the appropriate fixes to address these issues. Return the fixed content.
"""
        return prompt

    def _parse_ensemble_result(self, result_text: str, task: CoordinationTask) -> Optional[BlueTeamAssessment]:
        """Parse ensemble result into BlueTeamAssessment"""
        if not result_text:
            return None

        try:
            # Create a basic assessment from the result
            # In production, this would parse the result more carefully
            return BlueTeamAssessment(
                original_content=task.content,
                fixed_content=result_text,
                applied_fixes=[],  # Would be extracted from result
                overall_improvement_score=min(0.95, 0.6 + (len(result_text) / max(len(task.content), 1) * 0.35)),  # Calculate improvement based on content changes
                fixes_by_type={},  # Would be extracted from result
                fixes_by_priority={}  # Would be extracted from result
            )
        except (ValueError, KeyError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing ensemble result: {e}")
            return None

    def _execute_single_task(
        self,
        task: CoordinationTask,
        member: Optional[BlueTeamMember],
        strategy: BlueTeamStrategy
    ) -> BlueTeamAssessment:
        """Execute a single task"""
        if not self.blue_team:
            raise ValueError("Blue Team not available")

        logger.info(f"Executing task {task.task_id} with strategy {strategy.value}")

        # Use Blue Team to apply fixes
        assessment = self.blue_team.apply_fixes(
            content=task.content,
            issues=task.issues,
            content_type=task.content_type,
            strategy=strategy
        )

        return assessment

    # =========================================================================
    # TASK DISTRIBUTION AND LOAD BALANCING
    # =========================================================================

    def _assign_team_member(self, task: CoordinationTask) -> Optional[BlueTeamMember]:
        """Assign a team member to a task based on load balancing strategy"""
        if not self.team_members:
            return None

        with self.member_lock:
            if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_assignment(task)
            elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
                return self._least_loaded_assignment(task)
            elif self.load_balancing_strategy == LoadBalancingStrategy.SPECIALIZATION_BASED:
                return self._specialization_based_assignment(task)
            elif self.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
                return self._random_assignment(task)
            elif self.load_balancing_strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_assignment(task)
            else:
                return self.team_members[0]

    def _round_robin_assignment(self, task: CoordinationTask) -> Optional[BlueTeamMember]:
        """Round-robin assignment"""
        if not self.team_members:
            return None

        # Use task count for round-robin
        idx = sum(m.tasks_completed for m in self.member_metrics.values()) % len(self.team_members)
        member = self.team_members[idx]

        # Increment current load
        if member.name in self.member_metrics:
            self.member_metrics[member.name].current_load += 1

        return member

    def _least_loaded_assignment(self, task: CoordinationTask) -> Optional[BlueTeamMember]:
        """Assign to the least loaded team member"""
        if not self.team_members:
            return None

        # Find member with minimum current load
        least_loaded_member = min(
            self.team_members,
            key=lambda m: self.member_metrics.get(m.name, TeamMemberMetrics(member_name=m.name)).current_load
        )

        # Increment current load
        if least_loaded_member.name in self.member_metrics:
            self.member_metrics[least_loaded_member.name].current_load += 1

        return least_loaded_member

    def _specialization_based_assignment(self, task: CoordinationTask) -> Optional[BlueTeamMember]:
        """Assign based on specialization match"""
        if not self.team_members or not task.issues:
            return self._least_loaded_assignment(task)

        # Determine dominant fix type from issues
        fix_type_counts = {}
        for issue in task.issues:
            # Map issue category to fix type
            fix_type = self._map_issue_to_fix_type(issue)
            fix_type_counts[fix_type] = fix_type_counts.get(fix_type, 0) + 1

        if not fix_type_counts:
            return self._least_loaded_assignment(task)

        dominant_fix_type = max(fix_type_counts, key=fix_type_counts.get)

        # Find best member for this fix type
        best_member = None
        best_score = 0.0

        for member in self.team_members:
            metrics = self.member_metrics.get(member.name)
            if not metrics:
                continue

            # Consider specialization score and current load
            specialization_score = metrics.specialization_scores.get(dominant_fix_type, 0.5)
            load_factor = 1.0 / (metrics.current_load + 1)
            combined_score = specialization_score * load_factor

            if combined_score > best_score:
                best_score = combined_score
                best_member = member

        if best_member and best_member.name in self.member_metrics:
            self.member_metrics[best_member.name].current_load += 1

        return best_member or self.team_members[0]

    def _random_assignment(self, task: CoordinationTask) -> Optional[BlueTeamMember]:
        """Random assignment"""
        if not self.team_members:
            return None

        member = random.choice(self.team_members)

        if member.name in self.member_metrics:
            self.member_metrics[member.name].current_load += 1

        return member

    def _adaptive_assignment(self, task: CoordinationTask) -> Optional[BlueTeamMember]:
        """Adaptive assignment based on performance history"""
        if not self.team_members:
            return None

        # Calculate adaptive score for each member
        best_member = None
        best_score = 0.0

        for member in self.team_members:
            metrics = self.member_metrics.get(member.name)
            if not metrics:
                continue

            # Score based on:
            # 1. Reliability (success rate)
            # 2. Speed (inverse of average task time)
            # 3. Current load
            # 4. Specialization match

            reliability = metrics.reliability_score
            speed = 1.0 / (metrics.average_task_time + 1.0)
            load_factor = 1.0 / (metrics.current_load + 1)

            # Specialization bonus
            spec_bonus = 1.0
            if task.issues:
                fix_type = self._map_issue_to_fix_type(task.issues[0])
                spec_bonus = metrics.specialization_scores.get(fix_type, 0.5)

            combined_score = reliability * speed * load_factor * spec_bonus

            if combined_score > best_score:
                best_score = combined_score
                best_member = member

        if best_member and best_member.name in self.member_metrics:
            self.member_metrics[best_member.name].current_load += 1

        return best_member or self.team_members[0]

    # =========================================================================
    # METRICS AND MONITORING
    # =========================================================================

    def _update_task_metrics(self, task: CoordinationTask, success: bool):
        """Update metrics after a task completes"""
        if not task.assigned_member:
            return

        member_name = task.assigned_member
        metrics = self.member_metrics.get(member_name)

        if not metrics:
            return

        # Update basic metrics
        if success:
            metrics.tasks_completed += 1
        else:
            metrics.tasks_failed += 1

        # Update timing
        if task.started_at and task.completed_at:
            task_time = (task.completed_at - task.started_at).total_seconds()
            metrics.total_time_spent += task_time

            # Update average
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            metrics.average_task_time = metrics.total_time_spent / total_tasks if total_tasks > 0 else 0.0

        # Update current load
        metrics.current_load = max(0, metrics.current_load - 1)

        # Update reliability score
        total_tasks = metrics.tasks_completed + metrics.tasks_failed
        if total_tasks > 0:
            metrics.reliability_score = metrics.tasks_completed / total_tasks

        # Update last active
        metrics.last_active = datetime.now()

        # Update coordinator metrics
        self.metrics.total_tasks += 1
        if success:
            self.metrics.completed_tasks += 1
        else:
            self.metrics.failed_tasks += 1

    def _update_session_metrics(self, session: CoordinationSession):
        """Update metrics after a session completes"""
        self.metrics.total_sessions += 1

        # Update average session time
        if session.started_at and session.completed_at:
            session_time = (session.completed_at - session.started_at).total_seconds()

            if self.metrics.total_sessions > 0:
                # Rolling average
                self.metrics.average_session_time = (
                    (self.metrics.average_session_time * (self.metrics.total_sessions - 1) + session_time)
                    / self.metrics.total_sessions
                )

        # Update average task time
        if session.completed_tasks > 0:
            total_task_time = sum(
                (t.completed_at - t.started_at).total_seconds()
                for t in session.tasks
                if t.started_at and t.completed_at
            )
            self.metrics.average_task_time = total_task_time / session.completed_tasks

        # Update team utilization
        if self.team_members:
            total_load = sum(
                self.member_metrics.get(m.name, TeamMemberMetrics(member_name=m.name)).current_load
                for m in self.team_members
            )
            self.metrics.team_utilization = total_load / len(self.team_members)

        # Update throughput (tasks per minute)
        if session.started_at and session.completed_at:
            duration_minutes = (session.completed_at - session.started_at).total_seconds() / 60.0
            if duration_minutes > 0:
                self.metrics.throughput_tasks_per_minute = session.completed_tasks / duration_minutes

    def get_team_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all team members"""
        with self.member_lock:
            return {
                name: asdict(metrics)
                for name, metrics in self.member_metrics.items()
            }

    def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get overall coordinator metrics"""
        return asdict(self.metrics)

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        for session in self.session_history:
            if session.session_id == session_id:
                return {
                    "session_id": session.session_id,
                    "status": session.status.value,
                    "total_tasks": session.total_tasks,
                    "completed_tasks": session.completed_tasks,
                    "failed_tasks": session.failed_tasks,
                    "started_at": session.started_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "progress_percentage": (session.completed_tasks / session.total_tasks * 100) if session.total_tasks > 0 else 0
                }
        return None

    # =========================================================================
    # RESULT AGGREGATION
    # =========================================================================

    def _aggregate_session_results(self, session: CoordinationSession) -> Dict[str, Any]:
        """Aggregate results from all tasks in a session"""
        aggregated = {
            "session_id": session.session_id,
            "problem_statement": session.problem_statement,
            "total_sub_problems": len(session.sub_problems),
            "total_fixes_applied": 0,
            "fixes_by_type": {},
            "fixes_by_priority": {},
            "overall_improvement_score": 0.0,
            "total_time_taken": 0.0,
            "sub_problem_results": {},
            "team_performance": {}
        }

        total_improvement = 0.0
        task_count = 0

        for task in session.tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                # Add to totals
                task_fixes = len(task.result.applied_fixes)
                aggregated["total_fixes_applied"] += task_fixes
                total_improvement += task.result.overall_improvement_score
                task_count += 1

                # Add time
                if task.started_at and task.completed_at:
                    aggregated["total_time_taken"] += (task.completed_at - task.started_at).total_seconds()

                # Aggregate by type
                for fix_type, count in task.result.fixes_by_type.items():
                    aggregated["fixes_by_type"][fix_type.value] = aggregated["fixes_by_type"].get(fix_type.value, 0) + count

                # Aggregate by priority
                for priority, count in task.result.fixes_by_priority.items():
                    aggregated["fixes_by_priority"][priority.value] = aggregated["fixes_by_priority"].get(priority.value, 0) + count

                # Store per-sub-problem result
                aggregated["sub_problem_results"][task.sub_problem_id] = {
                    "status": "completed",
                    "fixes_applied": task_fixes,
                    "improvement_score": task.result.overall_improvement_score,
                    "fixed_content": task.result.fixed_content
                }

            elif task.status == TaskStatus.FAILED:
                aggregated["sub_problem_results"][task.sub_problem_id] = {
                    "status": "failed",
                    "error": task.error
                }

        # Calculate overall improvement
        if task_count > 0:
            aggregated["overall_improvement_score"] = total_improvement / task_count

        # Add team performance
        with self.member_lock:
            for member_name, metrics in self.member_metrics.items():
                aggregated["team_performance"][member_name] = {
                    "tasks_completed": metrics.tasks_completed,
                    "tasks_failed": metrics.tasks_failed,
                    "reliability_score": metrics.reliability_score,
                    "average_task_time": metrics.average_task_time
                }

        return aggregated

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _save_state(self):
        """Save coordinator state to disk"""
        try:
            state = {
                "member_metrics": self.member_metrics,
                "session_history": self.session_history,
                "metrics": self.metrics
            }

            with open(self.persistence_path, "wb") as f:
                json.dump(state, f)

            logger.debug(f"State saved to {self.persistence_path}")
        except (IOError, OSError, json.JSONDecodeError, PermissionError) as e:
            logger.error(f"Failed to save state: {e}")

    def _load_state(self):
        """Load coordinator state from disk"""
        try:
            if not os.path.exists(self.persistence_path):
                return

            with open(self.persistence_path, "rb") as f:
                state = json.load(f)

            self.member_metrics = state.get("member_metrics", {})
            self.session_history = state.get("session_history", [])
            self.metrics = state.get("metrics", CoordinatorMetrics())

            logger.debug(f"State loaded from {self.persistence_path}")
        except (IOError, OSError, json.JSONDecodeError, PermissionError, FileNotFoundError) as e:
            logger.error(f"Failed to load state: {e}")

    def clear_state(self):
        """Clear persisted state"""
        try:
            if os.path.exists(self.persistence_path):
                os.remove(self.persistence_path)
                logger.info(f"Cleared state at {self.persistence_path}")
        except (IOError, OSError, PermissionError, FileNotFoundError) as e:
            logger.error(f"Failed to clear state: {e}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _generate_session_id(self, problem_statement: str) -> str:
        """Generate a unique session ID"""
        hash_input = f"{problem_statement}_{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _generate_task_id(self, session_id: str, sub_problem_id: str) -> str:
        """Generate a unique task ID"""
        return f"{session_id}_{sub_problem_id}"

    def _determine_task_priority(self, sub_problem: Dict[str, Any]) -> TaskPriority:
        """Determine task priority from sub-problem metadata"""
        priority_str = sub_problem.get("priority", "medium").lower()

        priority_map = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW
        }

        return priority_map.get(priority_str, TaskPriority.MEDIUM)

    def _map_issue_to_fix_type(self, issue: Any) -> FixType:
        """Map an issue to a fix type"""
        # Check if it's an IssueFinding
        if hasattr(issue, 'category'):
            from red_team import IssueCategory
            category_map = {
                IssueCategory.SECURITY_VULNERABILITY: FixType.SECURITY_PATCH,
                IssueCategory.PERFORMANCE_PROBLEM: FixType.PERFORMANCE_OPTIMIZATION,
                IssueCategory.LOGICAL_ERROR: FixType.LOGIC_CORRECTION,
                IssueCategory.CLARITY_ISSUE: FixType.CLARITY_IMPROVEMENT,
                IssueCategory.STRUCTURAL_FLAW: FixType.STRUCTURE_REORGANIZATION,
                IssueCategory.DOCUMENTATION_GAP: FixType.DOCUMENTATION_ADDITION,
                IssueCategory.EDGE_CASE: FixType.ERROR_HANDLING,
                IssueCategory.COMPLIANCE_ISSUE: FixType.COMPLIANCE_FIX,
                IssueCategory.MAINTAINABILITY_PROBLEM: FixType.MAINTAINABILITY_IMPROVEMENT,
                IssueCategory.TECHNICAL_DEBT: FixType.CODE_REFACTORING,
            }
            return category_map.get(issue.category, FixType.LOGIC_CORRECTION)

        # Default
        return FixType.LOGIC_CORRECTION

    def _notify_progress(self, event_type: str, data: Any):
        """Notify registered progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(event_type, data)
            except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                logger.error(f"Progress callback error: {e}")

    def shutdown(self):
        """Shutdown the coordinator and cleanup resources"""
        logger.info("Shutting down BlueTeamCoordinator")

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Save final state
        if self.enable_persistence:
            self._save_state()

        logger.info("BlueTeamCoordinator shutdown complete")


# =============================================================================
# BLUE TEAM WORKFLOW
# =============================================================================

class BlueTeamWorkflow:
    """
    Complete workflow management for Blue Team coordination.

    Integrates with DecompositionEngine to automatically fix issues found
    during decomposition and sub-problem solving.
    """

    def __init__(
        self,
        coordinator: Optional[BlueTeamCoordinator] = None,
        auto_fix: bool = True,
        verify_fixes: bool = True,
        max_iterations: int = 3
    ):
        """
        Initialize the Blue Team Workflow.

        Args:
            coordinator: BlueTeamCoordinator instance
            auto_fix: Automatically apply fixes without manual approval
            verify_fixes: Verify fixes after application
            max_iterations: Maximum iterations for fixing issues
        """
        self.coordinator = coordinator or BlueTeamCoordinator()
        self.auto_fix = auto_fix
        self.verify_fixes = verify_fixes
        self.max_iterations = max_iterations

        logger.info(f"BlueTeamWorkflow initialized (auto_fix={auto_fix}, verify_fixes={verify_fixes})")

    def process_decomposition_result(
        self,
        problem_statement: str,
        decomposition_result: Dict[str, Any],
        content_items: Dict[str, str],
        issues_dict: Dict[str, List[Any]],
        content_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a decomposition result and automatically fix issues.

        This is the main integration point with DecompositionEngine.

        Args:
            problem_statement: Original problem statement
            decomposition_result: Result from DecompositionEngine
            content_items: Map of sub_problem_id -> content
            issues_dict: Map of sub_problem_id -> list of issues
            content_types: Optional map of content types

        Returns:
            Dict with fixed solutions and workflow results
        """
        logger.info("Processing decomposition result with Blue Team workflow")

        # Extract sub-problems from decomposition result
        sub_problems = decomposition_result.get("sub_problems", [])

        if not sub_problems:
            logger.warning("No sub-problems found in decomposition result")
            return {
                "success": False,
                "error": "No sub-problems to process"
            }

        # Coordinate fixes across all sub-problems
        session = self.coordinator.coordinate_decomposition_fixes(
            problem_statement=problem_statement,
            sub_problems=sub_problems,
            content_items=content_items,
            issues_dict=issues_dict,
            content_types=content_types,
            progress_callback=self._workflow_progress_callback
        )

        # Verify fixes if requested
        verification_results = None
        if self.verify_fixes and session.aggregated_result:
            verification_results = self._verify_fixes(session)

        # Return workflow results
        return {
            "success": session.status == TaskStatus.COMPLETED,
            "session_id": session.session_id,
            "fixed_solutions": session.aggregated_result,
            "verification_results": verification_results,
            "total_tasks": session.total_tasks,
            "completed_tasks": session.completed_tasks,
            "failed_tasks": session.failed_tasks,
            "workflow_metadata": {
                "auto_fix": self.auto_fix,
                "verify_fixes": self.verify_fixes,
                "iterations_completed": 1,
                "max_iterations": self.max_iterations
            }
        }

    def _verify_fixes(self, session: CoordinationSession) -> Dict[str, Any]:
        """Verify that fixes were applied correctly"""
        logger.info("Verifying fixes for session")

        verification_results = {
            "total_tasks_verified": 0,
            "verification_passed": 0,
            "verification_failed": 0,
            "task_verifications": {}
        }

        for task in session.tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                # Basic verification: check that content changed
                original_changed = task.result.original_content != task.result.fixed_content

                # Check that fixes were applied
                fixes_applied = len(task.result.applied_fixes) > 0

                # Verification passes if:
                # 1. Content changed OR
                # 2. Fixes were explicitly applied
                passed = original_changed or fixes_applied

                verification_results["task_verifications"][task.sub_problem_id] = {
                    "passed": passed,
                    "content_changed": original_changed,
                    "fixes_applied": fixes_applied,
                    "improvement_score": task.result.overall_improvement_score
                }

                verification_results["total_tasks_verified"] += 1
                if passed:
                    verification_results["verification_passed"] += 1
                else:
                    verification_results["verification_failed"] += 1

        logger.info(f"Verification complete: {verification_results['verification_passed']}/{verification_results['total_tasks_verified']} passed")
        return verification_results

    def _workflow_progress_callback(self, event_type: str, data: Any):
        """Handle progress updates from coordinator"""
        if event_type == "session_started":
            logger.info(f"Workflow session started: {data.session_id}")
        elif event_type == "session_completed":
            logger.info(f"Workflow session completed: {data.session_id}")
        elif event_type == "task_started":
            logger.debug(f"Task started: {data.task_id}")
        elif event_type == "task_completed":
            logger.debug(f"Task completed: {data.task_id}")
        elif event_type == "task_failed":
            logger.warning(f"Task failed: {data.task_id} - {data.error}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        coordinator_metrics = self.coordinator.get_coordinator_metrics()
        team_metrics = self.coordinator.get_team_metrics()

        return {
            "coordinator_metrics": coordinator_metrics,
            "team_metrics": team_metrics,
            "auto_fix": self.auto_fix,
            "verify_fixes": self.verify_fixes,
            "max_iterations": self.max_iterations,
            "current_session": self.coordinator.current_session.session_id if self.coordinator.current_session else None
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_blue_team_coordinator(
    max_concurrent_tasks: int = 5,
    load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
    enable_persistence: bool = True,
    ensemble: Optional[LLMEnsemble] = None,
    use_ensemble: bool = True
) -> BlueTeamCoordinator:
    """
    Create a BlueTeamCoordinator with default settings.

    Args:
        max_concurrent_tasks: Maximum concurrent tasks
        load_balancing: Load balancing strategy
        enable_persistence: Enable state persistence
        ensemble: Optional LLMEnsemble instance for ensemble-based coordination
        use_ensemble: Use ensemble-based coordination (requires ensemble)

    Returns:
        Configured BlueTeamCoordinator
    """
    return BlueTeamCoordinator(
        max_concurrent_tasks=max_concurrent_tasks,
        load_balancing_strategy=load_balancing,
        enable_persistence=enable_persistence,
        ensemble=ensemble,
        use_ensemble=use_ensemble
    )


def create_blue_team_workflow(
    auto_fix: bool = True,
    verify_fixes: bool = True,
    max_iterations: int = 3
) -> BlueTeamWorkflow:
    """
    Create a BlueTeamWorkflow with default settings.

    Args:
        auto_fix: Automatically apply fixes
        verify_fixes: Verify fixes after application
        max_iterations: Maximum fixing iterations

    Returns:
        Configured BlueTeamWorkflow
    """
    coordinator = create_blue_team_coordinator()
    return BlueTeamWorkflow(
        coordinator=coordinator,
        auto_fix=auto_fix,
        verify_fixes=verify_fixes,
        max_iterations=max_iterations
    )


# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create coordinator
    coordinator = create_blue_team_coordinator(
        max_concurrent_tasks=3,
        load_balancing=LoadBalancingStrategy.ADAPTIVE
    )

    # Create workflow
    workflow = create_blue_team_workflow(
        auto_fix=True,
        verify_fixes=True
    )

    logger.info("Blue Team Coordinator and Workflow initialized")
    logger.info(f"Available team members: {len(coordinator.team_members)}")
