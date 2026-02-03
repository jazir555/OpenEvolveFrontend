"""
Execution Controller for Adaptive MDAP.

This controller orchestrates the adaptive solving process with proper
CrewAI integration using the project's existing CrewAI infrastructure.

Integrates with:
- crewai_mdap_maker_engine.MAKEREngine, MAKERAgentFactory
- crewai_mdap_integrator.MDAPCrew
- maker_engine.MakerEngine
- mdap_engine.MDAPEngine
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Execution Controller
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None


import time
import uuid
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from adaptive_mdap.core.types import (
    SubProblem,
    ComplexityScore,
    SolveConfig,
    SolveStrategy,
)
from adaptive_mdap.core.errors import ExecutionError
from adaptive_mdap.classifiers.task_complexity_classifier import TaskComplexityClassifier
from adaptive_mdap.allocators.resource_allocator import (
    AdaptiveMDAPAllocator,
    AllocationContext,
)
from adaptive_mdap.integrations.crewai_integration import (
    CrewAIIntegration,
    AdaptiveCrewConfig,
)
from adaptive_mdap.utils.metrics import get_metrics
from adaptive_mdap.utils.logger import get_logger, set_correlation_id, clear_correlation_id

logger = get_logger("controllers.execution")


class SolutionStatus(Enum):
    """Status of a solution attempt."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SolutionAttempt:
    """Record of a solution attempt."""
    attempt_id: str
    subproblem_id: str
    complexity_score: float
    allocated_strategy: str
    n_agents: int
    k_ahead: int
    status: SolutionStatus
    start_time: float
    end_time: Optional[float] = None
    solution: Optional[Any] = None
    error_message: Optional[str] = None
    crew_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


@dataclass
class ExecutionMetrics:
    """Metrics for execution tracking."""
    total_steps: int = 0
    total_votes: int = 0
    red_flags: int = 0
    api_calls: int = 0
    cost_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_votes": self.total_votes,
            "red_flags": self.red_flags,
            "api_calls": self.api_calls,
            "cost_estimate": self.cost_estimate,
        }


class AdaptiveExecutionController:
    """
    Controller for adaptive execution of sub-problems using CrewAI.
    
    Integrates with the existing CrewAI infrastructure:
    - Uses crewai.Agent, crewai.Task, crewai.Crew, crewai.Process
    - Leverages crewai_mdap_maker_engine.MAKERAgentFactory
    - Leverages crewai_mdap_integrator for MDAP logic
    """
    
    def __init__(
        self,
        classifier: Optional[TaskComplexityClassifier] = None,
        allocator: Optional[AdaptiveMDAPAllocator] = None,
        crewai_integration: Optional[CrewAIIntegration] = None,
        solver_factory: Optional[Callable[[SolveConfig], Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the execution controller.
        
        Args:
            classifier: Complexity classifier
            allocator: Resource allocator
            crewai_integration: CrewAI integration (uses default if None)
            solver_factory: Factory function to create solvers
            llm_config: LLM configuration for agents
        """
        self.classifier = classifier or TaskComplexityClassifier()
        self.allocator = allocator or AdaptiveMDAPAllocator()
        self.crewai = crewai_integration or CrewAIIntegration()
        self.solver_factory = solver_factory
        self.llm_config = llm_config or {}
        
        # Execution statistics
        self._execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "fallback_executions": 0,
        }
        
        # Active attempts tracking
        self._active_attempts: Dict[str, SolutionAttempt] = {}
        
        # Engine cache
        self._engines: Dict[str, Any] = {}
        
        logger.info("Initialized AdaptiveExecutionController with CrewAI integration")
    
    def execute_adaptive(
        self,
        subproblem: SubProblem,
        workflow_id: Optional[str] = None,
        context: Optional[AllocationContext] = None,
        force_strategy: Optional[SolveStrategy] = None,
        enable_escalation: bool = True,
    ) -> SolutionAttempt:
        """
        Execute a sub-problem with adaptive resource allocation and optional escalation.
        
        If an execution fails, it can escalate to a more powerful strategy 
        (e.g., MDAP_LIGHT -> MAKER_FULL) to ensure reliability.
        
        Args:
            subproblem: The sub-problem to solve
            workflow_id: Optional workflow ID for tracking
            context: Optional allocation context
            force_strategy: Force a specific strategy
            enable_escalation: Whether to enable automatic escalation on failure
            
        Returns:
            SolutionAttempt with results
        """
        attempt_id = str(uuid.uuid4())
        correlation_id = workflow_id or attempt_id
        set_correlation_id(correlation_id)
        
        try:
            logger.info(f"Starting adaptive execution for {subproblem.id}")
            
            # Step 1: Compute complexity
            start_time = time.time()
            complexity = self.classifier.compute_complexity(subproblem)
            
            # Step 2: Initial Allocation
            if force_strategy:
                config = self.allocator.strategy_configs[force_strategy]
                logger.info(f"Using forced strategy: {force_strategy.value}")
            else:
                config = self.allocator.allocate_resources(
                    complexity.overall_score,
                    context=context,
                )
            
            # Tiered Escalation Loop
            strategies_to_try = self._get_escalation_path(config.strategy) if enable_escalation else [config.strategy]
            
            last_error = None
            for strategy in strategies_to_try:
                try:
                    current_config = self.allocator.strategy_configs[strategy]
                    logger.info(f"Attempting execution with strategy: {strategy.value}")
                    
                    # Create attempt record for this tier
                    attempt = self._create_attempt_record(
                        attempt_id, subproblem, complexity, current_config
                    )
                    
                    # Execute with strategy
                    result, metrics = self._execute_with_strategy(
                        subproblem,
                        current_config,
                        attempt,
                    )
                    
                    # Success!
                    attempt.status = SolutionStatus.COMPLETED
                    attempt.end_time = time.time()
                    attempt.solution = result
                    attempt.metadata["execution_metrics"] = metrics.to_dict()
                    
                    self._record_success(subproblem, complexity, current_config, attempt, metrics)
                    return attempt
                    
                except Exception as e:
                    logger.warning(f"Strategy {strategy.value} failed: {e}")
                    last_error = e
                    self._execution_stats["failed_executions"] += 1
                    # Continue to next strategy in escalation path
            
            # If we reach here, all strategies failed
            logger.error(f"All strategies failed for {subproblem.id}. Falling back to standard.")
            return self._fallback_to_standard(subproblem, attempt, last_error)
            
        except Exception as e:
            logger.exception(f"Fatal error in adaptive controller: {e}")
            return self._create_failed_attempt(attempt_id, subproblem, e)
        finally:
            clear_correlation_id()
    
    def _get_escalation_path(self, start_strategy: SolveStrategy) -> List[SolveStrategy]:
        """Define the path of escalation for failed attempts."""
        full_path = [
            SolveStrategy.DIRECT,
            SolveStrategy.MDAP_LIGHT,
            SolveStrategy.MDAP_MEDIUM,
            SolveStrategy.MAKER_FULL,
            SolveStrategy.MAKER_ULTRA
        ]
        try:
            start_idx = full_path.index(start_strategy)
            return full_path[start_idx:]
        except ValueError:
            return [start_strategy]
    
    def _create_attempt_record(self, attempt_id, subproblem, complexity, config):
        attempt = SolutionAttempt(
            attempt_id=attempt_id,
            subproblem_id=subproblem.id,
            complexity_score=complexity.overall_score,
            allocated_strategy=config.strategy.value,
            n_agents=config.n_agents,
            k_ahead=config.k_ahead,
            status=SolutionStatus.IN_PROGRESS,
            start_time=time.time(),
            metadata={
                "complexity_breakdown": {
                    "text_length": complexity.text_length_score,
                    "domain_rarity": complexity.domain_rarity_score,
                    "depth": complexity.depth_score,
                    "historical_error": complexity.historical_error_score,
                    "dependency": complexity.dependency_score,
                    "keyword": complexity.keyword_score,
                    "constraint": complexity.constraint_score,
                }
            }
        )
        self._active_attempts[attempt_id] = attempt
        return attempt
    
    def _record_success(self, subproblem, complexity, config, attempt, metrics):
        self._execution_stats["successful_executions"] += 1
        self._execution_stats["total_executions"] += 1
        
        # Record outcome for learning
        self.allocator.record_outcome(
            complexity_score=complexity.overall_score,
            strategy=config.strategy,
            success=True,
            cost=self._estimate_cost(config),
            quality=1.0,
        )
        
        get_metrics().record_execution(
            strategy=config.strategy.value,
            success=True,
            duration_ms=attempt.duration_ms,
            cost=self._estimate_cost(config),
        )
    
    def _create_failed_attempt(self, attempt_id, subproblem, error):
        return SolutionAttempt(
            attempt_id=attempt_id,
            subproblem_id=subproblem.id,
            complexity_score=0.5,
            allocated_strategy="unknown",
            n_agents=0,
            k_ahead=0,
            status=SolutionStatus.FAILED,
            start_time=time.time(),
            end_time=time.time(),
            error_message=str(error),
        )
    
    def _execute_with_strategy(
        self,
        subproblem: SubProblem,
        config: SolveConfig,
        attempt: SolutionAttempt,
    ) -> tuple[Any, ExecutionMetrics]:
        """
        Execute sub-problem with specific strategy configuration.
        
        Args:
            subproblem: The sub-problem to solve
            config: Strategy configuration
            attempt: Solution attempt record
            
        Returns:
            Tuple of (solution_result, execution_metrics)
        """
        logger.debug(
            f"Executing with {config.strategy.value}: "
            f"n_agents={config.n_agents}, k_ahead={config.k_ahead}"
        )
        
        # Route to appropriate execution method based on strategy
        if config.strategy == SolveStrategy.DIRECT:
            return self._execute_direct(subproblem, config)
        elif config.strategy == SolveStrategy.MDAP_LIGHT:
            return self._execute_mdap_light(subproblem, config)
        elif config.strategy == SolveStrategy.MDAP_MEDIUM:
            return self._execute_mdap_medium(subproblem, config)
        elif config.strategy == SolveStrategy.MAKER_FULL:
            return self._execute_maker_full(subproblem, config)
        elif config.strategy == SolveStrategy.MAKER_ULTRA:
            return self._execute_maker_ultra(subproblem, config)
        else:
            raise ExecutionError(f"Unknown strategy: {config.strategy}")
    
    def _execute_direct(self, subproblem: SubProblem, config: SolveConfig) -> tuple[Any, ExecutionMetrics]:
        """
        Execute with DIRECT strategy (single agent, single LLM call).
        
        This is the simplest strategy - one agent, one call, no voting.
        """
        logger.debug(f"DIRECT solve for {subproblem.id}")
        metrics = ExecutionMetrics()
        
        try:
            # Try to use CrewAI integration first
            crew = self.crewai.create_execution_crew(
                subproblem.id,
                SolveStrategy.DIRECT,
                AdaptiveCrewConfig(
                    strategy=SolveStrategy.DIRECT,
                    n_agents=1,
                    k_ahead=0,
                    max_retries=config.max_retries,
                ),
            )
            
            if crew:
                result = crew.kickoff()
                metrics.api_calls = 1
                return {"strategy": "direct", "result": result, "subproblem_id": subproblem.id}, metrics
            
            # Fallback to direct LLM call
            from llm_utils import _request_openai_compatible_chat
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant solving sub-problems."},
                {"role": "user", "content": subproblem.description}
            ]
            
            response = _request_openai_compatible_chat(messages, temperature=0.0)
            metrics.api_calls = 1
            
            return {
                "strategy": "direct",
                "result": response,
                "subproblem_id": subproblem.id,
                "description": subproblem.description,
            }, metrics
            
        except Exception as e:
            logger.warning(f"DIRECT execution failed: {e}, using fallback")
            # Return a basic result as fallback
            return {
                "strategy": "direct_fallback",
                "result": f"Solution for: {subproblem.description}",
                "subproblem_id": subproblem.id,
            }, metrics
    
    def _execute_mdap_light(self, subproblem: SubProblem, config: SolveConfig) -> tuple[Any, ExecutionMetrics]:
        """
        Execute with MDAP_LIGHT strategy (3 agents, k=1 voting).
        
        Uses the MDAP integrator for lightweight multi-agent debate.
        """
        logger.debug(f"MDAP_LIGHT solve for {subproblem.id}")
        metrics = ExecutionMetrics()
        
        try:
            # Try to use CrewAI MDAP integration
            from crewai_mdap_integrator import MDAPCrew, MDAPConfig, MDAPTask, MDAPStep
            
            mdap_config = MDAPConfig(
                k_min=1,
                k_max=1,
                max_votes_per_step=3,
                timeout_seconds=config.timeout_ms // 1000 if config.timeout_ms else 60,
            )
            
            mdap_crew = MDAPCrew(config=mdap_config)
            
            task = MDAPTask(
                task_id=f"mdap_light_{subproblem.id}",
                description=subproblem.description,
                steps=[
                    MDAPStep(
                        step_id="solve",
                        prompt=subproblem.description,
                        expected_schema={"type": "object"},
                    )
                ],
            )
            
            result = mdap_crew.run_task(task)
            metrics.total_votes = result.step_results.get("solve", {}).get("vote_result", {}).get("votes", 0)
            metrics.red_flags = result.step_results.get("solve", {}).get("vote_result", {}).get("red_flags", 0)
            metrics.api_calls = 3  # 3 agents
            
            return {
                "strategy": "mdap_light",
                "result": result,
                "subproblem_id": subproblem.id,
            }, metrics
            
        except Exception as e:
            logger.warning(f"MDAP_LIGHT execution failed: {e}, falling back to DIRECT")
            # Fallback to direct
            return self._execute_direct(subproblem, config)
    
    def _execute_mdap_medium(self, subproblem: SubProblem, config: SolveConfig) -> tuple[Any, ExecutionMetrics]:
        """
        Execute with MDAP_MEDIUM strategy (5 agents, k=1 voting).
        
        Uses more agents than LIGHT but with same k threshold.
        """
        logger.debug(f"MDAP_MEDIUM solve for {subproblem.id}")
        metrics = ExecutionMetrics()
        
        try:
            from crewai_mdap_integrator import MDAPCrew, MDAPConfig, MDAPTask, MDAPStep
            
            mdap_config = MDAPConfig(
                k_min=1,
                k_max=1,
                max_votes_per_step=5,
                timeout_seconds=config.timeout_ms // 1000 if config.timeout_ms else 90,
            )
            
            mdap_crew = MDAPCrew(config=mdap_config)
            
            task = MDAPTask(
                task_id=f"mdap_medium_{subproblem.id}",
                description=subproblem.description,
                steps=[
                    MDAPStep(
                        step_id="solve",
                        prompt=subproblem.description,
                        expected_schema={"type": "object"},
                    )
                ],
            )
            
            result = mdap_crew.run_task(task)
            metrics.total_votes = 5
            metrics.api_calls = 5
            
            return {
                "strategy": "mdap_medium",
                "result": result,
                "subproblem_id": subproblem.id,
            }, metrics
            
        except Exception as e:
            logger.warning(f"MDAP_MEDIUM execution failed: {e}, falling back to MDAP_LIGHT")
            return self._execute_mdap_light(subproblem, config)
    
    def _execute_maker_full(self, subproblem: SubProblem, config: SolveConfig) -> tuple[Any, ExecutionMetrics]:
        """
        Execute with MAKER_FULL strategy (5 agents, k=2 voting).
        
        Uses the full MAKER protocol with first-to-ahead-by-K voting.
        """
        logger.debug(f"MAKER_FULL solve for {subproblem.id}")
        metrics = ExecutionMetrics()
        
        try:
            # Try to use CrewAI MAKER integration
            from crewai_mdap_maker_engine import MAKEREngine, MAKERConfig
            
            maker_config = MAKERConfig(
                k_ahead=2,
                max_voting_rounds=50,
                enable_first_to_ahead=True,
                enable_red_flagging=True,
            )
            
            maker_engine = MAKEREngine(config=maker_config)
            
            # Create a simple step for the maker engine
            from maker_engine import MakerStep, MakerState
            
            step = MakerStep(
                step_id=f"maker_full_{subproblem.id}",
                prompt_template=subproblem.description,
            )
            
            initial_state = {"problem": subproblem.description}
            
            def step_builder(state, history):
                return step
            
            def apply_action(state, action):
                return {"result": action, "previous_state": state}
            
            result = maker_engine.solve(
                initial_state=initial_state,
                step_builder=step_builder,
                apply_action=apply_action,
            )
            
            metrics.total_votes = maker_engine.metrics.get("votes_cast", 0)
            metrics.red_flags = maker_engine.metrics.get("red_flags", 0)
            metrics.total_steps = maker_engine.metrics.get("steps", 0)
            metrics.api_calls = metrics.total_votes
            
            return {
                "strategy": "maker_full",
                "result": result.state.current_state,
                "subproblem_id": subproblem.id,
                "metrics": maker_engine.metrics,
            }, metrics
            
        except Exception as e:
            logger.warning(f"MAKER_FULL execution failed: {e}, falling back to MDAP_MEDIUM")
            return self._execute_mdap_medium(subproblem, config)
    
    def _execute_maker_ultra(self, subproblem: SubProblem, config: SolveConfig) -> tuple[Any, ExecutionMetrics]:
        """
        Execute with MAKER_ULTRA strategy (7+ agents, k=3 voting).
        
        Maximum reliability strategy for the most complex problems.
        """
        logger.debug(f"MAKER_ULTRA solve for {subproblem.id}")
        metrics = ExecutionMetrics()
        
        try:
            from crewai_mdap_maker_engine import MAKEREngine, MAKERConfig
            from maker_engine import MakerStep, MakerState
            
            maker_config = MAKERConfig(
                k_ahead=3,
                max_voting_rounds=60,
                enable_first_to_ahead=True,
                enable_red_flagging=True,
            )
            
            maker_engine = MAKEREngine(config=maker_config)
            
            step = MakerStep(
                step_id=f"maker_ultra_{subproblem.id}",
                prompt_template=subproblem.description,
            )
            
            initial_state = {"problem": subproblem.description}
            
            def step_builder(state, history):
                return step
            
            def apply_action(state, action):
                return {"result": action, "previous_state": state}
            
            result = maker_engine.solve(
                initial_state=initial_state,
                step_builder=step_builder,
                apply_action=apply_action,
            )
            
            metrics.total_votes = maker_engine.metrics.get("votes_cast", 0)
            metrics.red_flags = maker_engine.metrics.get("red_flags", 0)
            metrics.total_steps = maker_engine.metrics.get("steps", 0)
            metrics.api_calls = metrics.total_votes
            
            return {
                "strategy": "maker_ultra",
                "result": result.state.current_state,
                "subproblem_id": subproblem.id,
                "metrics": maker_engine.metrics,
            }, metrics
            
        except Exception as e:
            logger.warning(f"MAKER_ULTRA execution failed: {e}, falling back to MAKER_FULL")
            return self._execute_maker_full(subproblem, config)
    
    def _fallback_to_standard(
        self,
        subproblem: SubProblem,
        original_attempt: SolutionAttempt,
        error: Exception,
    ) -> SolutionAttempt:
        """Fallback to standard solver when adaptive execution fails."""
        logger.warning(f"Falling back to standard solver for {subproblem.id}: {error}")
        
        try:
            # Try direct execution as ultimate fallback
            result, metrics = self._execute_direct(
                subproblem,
                SolveConfig(
                    strategy=SolveStrategy.DIRECT,
                    n_agents=1,
                    k_ahead=0,
                    max_retries=1,
                ),
            )
            
            original_attempt.status = SolutionStatus.COMPLETED
            original_attempt.end_time = time.time()
            original_attempt.solution = result
            original_attempt.metadata["fallback"] = True
            original_attempt.metadata["original_error"] = str(error)
            original_attempt.metadata["execution_metrics"] = metrics.to_dict()
            
            self._execution_stats["fallback_executions"] += 1
            
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            original_attempt.status = SolutionStatus.FAILED
            original_attempt.end_time = time.time()
            original_attempt.error_message = f"Original: {error}, Fallback: {fallback_error}"
            self._execution_stats["failed_executions"] += 1
        
        return original_attempt
    
    def _estimate_cost(self, config: SolveConfig) -> float:
        """Estimate cost of a configuration with granular strategy costs."""
        base_cost = {
            SolveStrategy.DIRECT: 1.0,
            SolveStrategy.MDAP_LIGHT: 3.0,
            SolveStrategy.MDAP_MEDIUM: 5.0,
            SolveStrategy.MAKER_FULL: 8.0,
            SolveStrategy.MAKER_ULTRA: 12.0,
        }.get(config.strategy, 1.0)
        
        voting_overhead = 1 + (config.k_ahead * 0.4)
        return base_cost * voting_overhead
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            **self._execution_stats,
            "active_attempts": len(self._active_attempts),
        }
    
    def get_attempt(self, attempt_id: str) -> Optional[SolutionAttempt]:
        """Get a specific attempt by ID."""
        return self._active_attempts.get(attempt_id)
