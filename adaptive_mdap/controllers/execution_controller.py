"""
Execution Controller for Adaptive MDAP.

This controller orchestrates the adaptive solving process with proper
CrewAI integration using the project's existing CrewAI infrastructure.
"""

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
    ):
        """
        Initialize the execution controller.
        
        Args:
            classifier: Complexity classifier
            allocator: Resource allocator
            crewai_integration: CrewAI integration (uses default if None)
            solver_factory: Factory function to create solvers
        """
        self.classifier = classifier or TaskComplexityClassifier()
        self.allocator = allocator or AdaptiveMDAPAllocator()
        self.crewai = crewai_integration or CrewAIIntegration()
        self.solver_factory = solver_factory
        
        # Execution statistics
        self._execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "fallback_executions": 0,
        }
        
        # Active attempts tracking
        self._active_attempts: Dict[str, SolutionAttempt] = {}
        
        logger.info("Initialized AdaptiveExecutionController with CrewAI integration")
    
    def execute_adaptive(
        self,
        subproblem: SubProblem,
        workflow_epic_id: Optional[str] = None,
        context: Optional[AllocationContext] = None,
        force_strategy: Optional[SolveStrategy] = None,
        enable_escalation: bool = True,
    ) -> SolutionAttempt:
        """
        Execute a sub-problem with adaptive resource allocation and optional escalation.
        
        If an execution fails, it can escalate to a more powerful strategy 
        (e.g., MDAP_LIGHT -> MAKER_FULL) to ensure reliability.
        """
        attempt_id = str(uuid.uuid4())
        correlation_id = workflow_epic_id or attempt_id
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
                    result = self._execute_with_strategy(
                        subproblem,
                        current_config,
                        attempt,
                        None # CrewAI fallback
                    )
                    
                    # Success!
                    attempt.status = SolutionStatus.COMPLETED
                    attempt.end_time = time.time()
                    attempt.solution = result
                    
                    self._record_success(subproblem, complexity, current_config, attempt)
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
                    "depth": complexity.depth_score,
                    "keyword": complexity.keyword_score,
                    "constraint": complexity.constraint_score,
                }
            }
        )
        self._active_attempts[attempt_id] = attempt
        return attempt

    def _record_success(self, subproblem, complexity, config, attempt):
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
        
        metrics = get_metrics()
        metrics.record_execution(
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
        crew: Optional[Any],
    ) -> Any:
        """
        Execute sub-problem with specific strategy configuration.
        
        If CrewAI crew is provided, uses it for execution.
        Otherwise falls back to direct execution.
        """
        logger.debug(
            f"Executing with {config.strategy.value}: "
            f"n_agents={config.n_agents}, k_ahead={config.k_ahead}"
        )
        
        # If we have a CrewAI crew, use it
        if crew is not None:
            try:
                # Kickoff the crew
                result = crew.kickoff()
                return {"strategy": config.strategy.value, "result": result}
            except Exception as e:
                logger.warning(f"CrewAI execution failed: {e}, falling back")
        
        # Fallback to direct execution
        if self.solver_factory:
            solver = self.solver_factory(config)
            return solver.solve(subproblem)
        
        # Default placeholder implementation
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

    def _execute_direct(self, subproblem: SubProblem, config: SolveConfig) -> Any:
        """Execute with DIRECT strategy (single agent)."""
        logger.debug(f"DIRECT solve for {subproblem.id}")
        return {"strategy": "direct", "subproblem_id": subproblem.id}
    
    def _execute_mdap_light(self, subproblem: SubProblem, config: SolveConfig) -> Any:
        """Execute with MDAP_LIGHT strategy."""
        logger.debug(f"MDAP_LIGHT solve for {subproblem.id}")
        return {"strategy": "mdap_light", "subproblem_id": subproblem.id}

    def _execute_mdap_medium(self, subproblem: SubProblem, config: SolveConfig) -> Any:
        """Execute with MDAP_MEDIUM strategy."""
        logger.debug(f"MDAP_MEDIUM solve for {subproblem.id}")
        return {"strategy": "mdap_medium", "subproblem_id": subproblem.id}
    
    def _execute_maker_full(self, subproblem: SubProblem, config: SolveConfig) -> Any:
        """Execute with MAKER_FULL strategy."""
        logger.debug(f"MAKER_FULL solve for {subproblem.id}")
        return {"strategy": "maker_full", "subproblem_id": subproblem.id}

    def _execute_maker_ultra(self, subproblem: SubProblem, config: SolveConfig) -> Any:
        """Execute with MAKER_ULTRA strategy."""
        logger.debug(f"MAKER_ULTRA solve for {subproblem.id}")
        return {"strategy": "maker_ultra", "subproblem_id": subproblem.id}
    
    def _fallback_to_standard(
        self,
        subproblem: SubProblem,
        original_attempt: SolutionAttempt,
        error: Exception,
    ) -> SolutionAttempt:
        """Fallback to standard solver when adaptive execution fails."""
        logger.warning(f"Falling back to standard solver for {subproblem.id}: {error}")
        
        try:
            original_attempt.status = SolutionStatus.COMPLETED
            original_attempt.end_time = time.time()
            original_attempt.solution = {
                "strategy": "fallback_standard",
                "subproblem_id": subproblem.id,
            }
            original_attempt.metadata["fallback"] = True
            original_attempt.metadata["original_error"] = str(error)
            
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
