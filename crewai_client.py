"""
CrewAI Client - Local Execution Replacement for crewai API Client

This module provides the client interface for executing CrewAI workflows locally,
replacing the crewai HTTP API client with local CrewAI flow execution.

Key Differences from crewai:
- No HTTP API (local execution only)
- No remote database (local JSON state files)
- No ticket system (state-based workflow)
- MIT license (replaces AGPL crewai)

License: MIT
"""


import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Import CrewAI components
from crewai_state_management import (
    WorkflowState,
    WorkflowStatus,
    ExecutionMethod,
    StateManager,
    create_workflow_state,
    create_state_manager,
)
from crewai_unified_flow import CrewAIUnifiedFlow, ExecutionMethod as CrewAIExecutionMethod, create_unified_flow

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExecutionMetrics:
    """Metrics for workflow execution"""
    workflow_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_seconds: float = 0.0
    phases_completed: int = 0
    phases_total: int = 6
    tokens_used: int = 0
    agents_deployed: int = 0
    voting_rounds: int = 0
    red_flags: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration_seconds": self.total_duration_seconds,
            "phases_completed": self.phases_completed,
            "phases_total": self.phases_total,
            "tokens_used": self.tokens_used,
            "agents_deployed": self.agents_deployed,
            "voting_rounds": self.voting_rounds,
            "red_flags": self.red_flags,
            "errors": self.errors,
        }


@dataclass
class ExecutionResult:
    """Result from workflow execution"""
    workflow_id: str
    status: str
    final_solution: Optional[str] = None
    phase_results: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[ExecutionMetrics] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "final_solution": self.final_solution,
            "phase_results": self.phase_results,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "error": self.error,
        }


# =============================================================================
# CREWAI CLIENT
# =============================================================================

class CrewAIClient:
    """
    Client for executing CrewAI workflows locally.

    Replaces CrewAIClient with local execution:
    - No HTTP API calls
    - Local state management
    - Direct CrewAI flow execution
    - Result aggregation from multiple agents
    """

    def __init__(
        self,
        state_storage_dir: str = "./crewai_states",
        enable_persistence: bool = True,
        default_execution_method: ExecutionMethod = ExecutionMethod.AUTO,
    ):
        """
        Initialize CrewAI client.

        Args:
            state_storage_dir: Directory for state storage
            enable_persistence: Enable state persistence to disk
            default_execution_method: Default execution method
        """
        self.state_storage_dir = state_storage_dir
        self.enable_persistence = enable_persistence
        self.default_execution_method = default_execution_method

        # Create state manager if persistence enabled
        if enable_persistence:
            self.state_manager = create_state_manager(state_storage_dir)
        else:
            self.state_manager = None

        flow_method = self._map_to_flow_execution_method(default_execution_method)

        # Create unified flow
        self.unified_flow = create_unified_flow(
            default_execution_method=flow_method,
            enable_persistence=enable_persistence
        )

        # Active workflows tracking
        self.active_workflows: Dict[str, WorkflowState] = {}

        logger.info(f"CrewAIClient initialized with storage_dir={state_storage_dir}")

    def execute_workflow(
        self,
        problem_statement: str,
        execution_method: ExecutionMethod = ExecutionMethod.AUTO,
        workflow_id: Optional[str] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a complete CrewAI workflow.

        Args:
            problem_statement: The problem to solve
            execution_method: Execution method to use
            workflow_id: Optional custom workflow ID
            callback: Optional callback for phase updates
            **kwargs: Additional execution parameters

        Returns:
            ExecutionResult with final solution and metrics
        """
        # Generate workflow ID if not provided
        if not workflow_id:
            workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"

        logger.info(f"Executing workflow {workflow_id}: {problem_statement[:100]}...")

        # Initialize metrics
        metrics = ExecutionMetrics(
            workflow_id=workflow_id,
            start_time=time.time(),
        )

        try:
            # Create initial workflow state
            state = create_workflow_state(
                workflow_id=workflow_id,
                problem_statement=problem_statement,
                execution_method=execution_method,
            )

            # Store active workflow
            self.active_workflows[workflow_id] = state

            # Save initial state if persistence enabled
            if self.state_manager:
                self.state_manager.save_state(workflow_id, state)

            # Execute workflow through unified flow
            flow_result = self.unified_flow.execute_full_workflow(
                problem_statement=problem_statement,
                execution_method=self._map_to_flow_execution_method(execution_method),
                **kwargs
            )

            # Update metrics
            metrics.end_time = time.time()
            metrics.total_duration_seconds = metrics.end_time - metrics.start_time
            phases = flow_result.get("phases", {})
            metrics.phases_completed = sum(
                1 for phase in phases.values()
                if isinstance(phase, dict) and phase.get("status") == "completed"
            )

            # Extract final solution
            final_solution = None
            if flow_result.get("status") == "completed":
                # Try to get final solution from different possible locations
                phase5_result = phases.get("phase5", {}) if isinstance(phases, dict) else {}
                phase6_result = phases.get("phase6", {}) if isinstance(phases, dict) else {}
                final_solution = (
                    phase5_result.get("final_solution")
                    or phase5_result.get("reassembled_content")
                    or phase6_result.get("final_solution")
                    or flow_result.get("final_solution")
                )

                # If not in phase 6, check phase 5 (reassembly)
                if not final_solution:
                    final_solution = phase5_result.get("reassembled_content")

            # Update state
            if self.state_manager:
                updated_state = self.state_manager.load_state(workflow_id)
                if updated_state:
                    self.active_workflows[workflow_id] = updated_state

            # Create result
            result = ExecutionResult(
                workflow_id=workflow_id,
                status=flow_result.get("status", "unknown"),
                final_solution=final_solution,
                phase_results=flow_result.get("phases", {}),
                metrics=metrics,
            )

            # Callback if provided
            if callback:
                callback(workflow_id, result.to_dict())

            logger.info(f"Workflow {workflow_id} completed in {metrics.total_duration_seconds:.2f}s")
            return result

        except Exception as e:
            metrics.end_time = time.time()
            metrics.total_duration_seconds = metrics.end_time - metrics.start_time
            metrics.errors.append(str(e))

            logger.error(f"Workflow {workflow_id} failed: {e}")

            return ExecutionResult(
                workflow_id=workflow_id,
                status="failed",
                error=str(e),
                metrics=metrics,
            )

        finally:
            # Clean up active workflow
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

    def execute_phase(
        self,
        workflow_id: str,
        phase_number: int,
        phase_input: Dict[str, Any],
        execution_method: ExecutionMethod = ExecutionMethod.AUTO,
    ) -> Dict[str, Any]:
        """
        Execute a single phase of a workflow.

        Args:
            workflow_id: Workflow identifier
            phase_number: Phase number (1-6)
            phase_input: Input data for the phase
            execution_method: Execution method to use

        Returns:
            Phase execution result
        """
        logger.info(f"Executing Phase {phase_number} for workflow {workflow_id}")

        try:
            flow_method = self._map_to_flow_execution_method(execution_method)

            # Map phase numbers to unified flow methods
            if phase_number == 1:
                result = self.unified_flow.phase_1_setup(
                    problem_statement=phase_input.get("problem_statement"),
                    execution_method=flow_method,
                    **phase_input.get("kwargs", {})
                )
            elif phase_number == 2:
                result = self.unified_flow.phase_2_solve(
                    phase_1_result=phase_input,
                    **phase_input.get("kwargs", {})
                )
            elif phase_number == 3:
                result = self.unified_flow.phase_3_critique(
                    phase_2_result=phase_input,
                    execution_method=flow_method,
                    **phase_input.get("kwargs", {})
                )
            elif phase_number == 4:
                critiques = phase_input.get("critiques") or phase_input.get("phase3_result")
                result = self.unified_flow.phase_4_verify(
                    phase_2_result=phase_input,
                    critiques=critiques,
                    execution_method=flow_method,
                    **phase_input.get("kwargs", {})
                )
            elif phase_number == 5:
                phase_2_result = phase_input.get("phase_2_result", phase_input)
                problem_statement = (
                    phase_input.get("problem_statement")
                    or phase_2_result.get("problem_statement")
                    or phase_input.get("analysis", {}).get("problem_statement", "")
                )
                result = self.unified_flow.phase_5_reassemble(
                    phase_2_result=phase_2_result,
                    problem_statement=problem_statement,
                    execution_method=flow_method,
                    **phase_input.get("kwargs", {})
                )
            elif phase_number == 6:
                phase_5_result = phase_input.get("phase_5_result", phase_input)
                final_solution = (
                    phase_input.get("final_solution")
                    or phase_5_result.get("final_solution")
                    or phase_5_result.get("reassembled_content")
                    or ""
                )
                problem_statement = (
                    phase_input.get("problem_statement")
                    or phase_input.get("analysis", {}).get("problem_statement", "")
                )
                result = self.unified_flow.phase_6_final_validation(
                    final_solution=final_solution,
                    problem_statement=problem_statement,
                    execution_method=flow_method,
                    **phase_input.get("kwargs", {})
                )
            else:
                # Phase number not in valid range 1-6
                error_msg = f"Invalid phase number: {phase_number}. Valid phases are 1-6."
                logger.error(error_msg)
                return {
                    "phase": phase_number,
                    "status": "failed",
                    "error": error_msg,
                }

            # Update state if persistence enabled
            if self.state_manager and workflow_id:
                state = self.state_manager.load_state(workflow_id)
                if state:
                    state.phase = phase_number
                    self.state_manager.save_state(workflow_id, state)

            return result

        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            logger.error(f"Phase {phase_number} execution failed: {e}")
            return {
                "phase": phase_number,
                "status": "failed",
                "error": str(e),
            }

    def get_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Get the current state of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Current workflow state or None
        """
        # Check active workflows first
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]

        # Load from storage if persistence enabled
        if self.state_manager:
            return self.state_manager.load_state(workflow_id)

        return None

    def get_workflow_tickets(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Return ticket-like entries for a workflow.

        CrewAI no longer uses a ticket system, so this derives ticket data
        from the decomposition plan and solution attempts to preserve
        compatibility with UI integrations.
        """
        state = self.get_workflow_state(workflow_id)
        if not state:
            return []

        tickets: List[Dict[str, Any]] = []
        sub_problems = []
        if state.decomposition_plan and state.decomposition_plan.sub_problems:
            sub_problems = list(state.decomposition_plan.sub_problems)

        sub_solutions = state.sub_solutions or {}

        def normalize_status(value: Optional[str]) -> str:
            if not value:
                return "pending"
            status = value.lower()
            if status in {"completed", "complete", "verified", "solved", "done"}:
                return "completed"
            if status in {"in_progress", "running", "solving", "active"}:
                return "in_progress"
            if status in {"failed", "error"}:
                return "failed"
            if status in {"paused", "blocked"}:
                return "blocked"
            return "pending"

        for sub_problem in sub_problems:
            attempt = sub_solutions.get(sub_problem.id)
            status = None
            assignee = None
            created_at = None
            if attempt is not None:
                if isinstance(attempt, dict):
                    status = attempt.get("status")
                    assignee = attempt.get("agent_name") or attempt.get("generated_by_model")
                    created_at = attempt.get("created_at") or attempt.get("timestamp")
                else:
                    status = getattr(attempt, "status", None)
                    assignee = getattr(attempt, "agent_name", None) or getattr(attempt, "generated_by_model", None)
                    created_at = getattr(attempt, "created_at", None) or getattr(attempt, "timestamp", None)

            tickets.append(
                {
                    "id": sub_problem.id,
                    "title": sub_problem.title,
                    "description": sub_problem.description,
                    "status": normalize_status(status),
                    "assigned_agent_id": assignee,
                    "created_at": created_at or state.created_at,
                    "updated_at": state.updated_at,
                    "sub_problem_id": sub_problem.id,
                    "dependencies": list(sub_problem.dependencies or []),
                    "priority": getattr(sub_problem, "priority", None),
                }
            )

        return tickets

    def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[str]:
        """
        List all workflow IDs.

        Args:
            status: Optional filter by workflow status

        Returns:
            List of workflow IDs
        """
        if self.state_manager:
            return self.state_manager.list_workflows(status=status)

        # Return active workflows if no persistence
        if status:
            return [
                wf_id for wf_id, state in self.active_workflows.items()
                if state.status == status
            ]
        return list(self.active_workflows.keys())

    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete a workflow and its state.

        Args:
            workflow_id: Workflow identifier

        Returns:
            True if deleted, False otherwise
        """
        # Remove from active workflows
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]

        # Delete from storage if persistence enabled
        if self.state_manager:
            return self.state_manager.delete_state(workflow_id)

        return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get client status and availability.

        Returns:
            Status information
        """
        flow_status = self.unified_flow.get_status()

        return {
            "client": "CrewAI",
            "version": "1.0.0",
            "state_storage_dir": self.state_storage_dir,
            "persistence_enabled": self.enable_persistence,
            "default_execution_method": self.default_execution_method,
            "active_workflows": len(self.active_workflows),
            "flow_status": flow_status,
        }

    def _map_to_flow_execution_method(
        self,
        method: ExecutionMethod
    ) -> CrewAIExecutionMethod:
        """Map state-management execution method to unified flow enum."""
        if isinstance(method, CrewAIExecutionMethod):
            return method
        if hasattr(method, "value"):
            method = method.value
        if isinstance(method, str):
            value = method.lower()
            if value in CrewAIExecutionMethod._value2member_map_:
                return CrewAIExecutionMethod(value)
        return CrewAIExecutionMethod.AUTO


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

class ResultAggregator:
    """
    Aggregates results from multiple agent executions.

    Used to combine results from:
    - Multiple voting rounds (MDAP)
    - Parallel agent executions
    - Phase results
    """

    @staticmethod
    def aggregate_votes(
        votes: Dict[str, int],
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """
        Aggregate voting results from multiple agents.

        Args:
            votes: Dictionary mapping candidate to vote count
            min_confidence: Minimum confidence threshold

        Returns:
            Aggregation result with winner and confidence
        """
        if not votes:
            return {
                "winner": None,
                "confidence": 0.0,
                "total_votes": 0,
                "candidates": 0,
            }

        total_votes = sum(votes.values())
        candidates = len(votes)

        # Find winner
        winner = max(votes.items(), key=lambda x: x[1])
        winner_name, winner_votes = winner

        # Calculate confidence
        confidence = winner_votes / total_votes if total_votes > 0 else 0.0

        return {
            "winner": winner_name,
            "winner_votes": winner_votes,
            "confidence": confidence,
            "total_votes": total_votes,
            "candidates": candidates,
            "meets_confidence_threshold": confidence >= min_confidence,
            "vote_distribution": votes,
        }

    @staticmethod
    def aggregate_phase_results(
        phase_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple phase executions.

        Args:
            phase_results: List of phase execution results

        Returns:
            Aggregated phase results
        """
        if not phase_results:
            return {
                "total_phases": 0,
                "successful_phases": 0,
                "failed_phases": 0,
                "results": {},
            }

        successful = sum(1 for r in phase_results if r.get("status") == "completed")
        failed = sum(1 for r in phase_results if r.get("status") == "failed")

        return {
            "total_phases": len(phase_results),
            "successful_phases": successful,
            "failed_phases": failed,
            "results": {r.get("phase", "unknown"): r for r in phase_results},
        }

    @staticmethod
    def aggregate_agent_outputs(
        outputs: List[Dict[str, Any]],
        aggregation_strategy: str = "first_to_head"
    ) -> Dict[str, Any]:
        """
        Aggregate outputs from multiple agents.

        Args:
            outputs: List of agent outputs
            aggregation_strategy: Strategy for aggregation (first_to_head, majority_vote, consensus)

        Returns:
            Aggregated output
        """
        if not outputs:
            return {
                "aggregated_output": None,
                "strategy": aggregation_strategy,
                "agent_count": 0,
            }

        if aggregation_strategy == "first_to_head":
            # First-to-Ahead-by-K: Find the first output to reach K votes
            vote_counts = {}
            for output in outputs:
                output_hash = hash(str(output))
                vote_counts[output_hash] = vote_counts.get(output_hash, 0) + 1

                # First to reach threshold wins
                if vote_counts[output_hash] >= 5:  # K=5 default
                    return {
                        "aggregated_output": output,
                        "strategy": aggregation_strategy,
                        "agent_count": len(outputs),
                        "winning_votes": vote_counts[output_hash],
                    }

            # If no clear winner, return most voted
            winner_hash = max(vote_counts.items(), key=lambda x: x[1])[0]
            winning_output = next(o for o in outputs if hash(str(o)) == winner_hash)

            return {
                "aggregated_output": winning_output,
                "strategy": aggregation_strategy,
                "agent_count": len(outputs),
                "winning_votes": vote_counts[winner_hash],
            }

        elif aggregation_strategy == "majority_vote":
            # Simple majority vote
            return ResultAggregator.aggregate_votes(
                {str(o): outputs.count(o) for o in set(outputs)}
            )

        elif aggregation_strategy == "consensus":
            # All agents must agree
            if len(set(str(o) for o in outputs)) == 1:
                return {
                    "aggregated_output": outputs[0],
                    "strategy": aggregation_strategy,
                    "agent_count": len(outputs),
                    "consensus_reached": True,
                }
            else:
                return {
                    "aggregated_output": None,
                    "strategy": aggregation_strategy,
                    "agent_count": len(outputs),
                    "consensus_reached": False,
                }

        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")


# =============================================================================
# MONITORING INTERFACE
# =============================================================================

class CrewAIMonitor:
    """
    Monitoring interface for CrewAI workflow execution.

    Provides:
    - Real-time workflow status tracking
    - Metrics collection
    - Logging
    - Event streaming (optional)
    """

    def __init__(
        self,
        client: CrewAIClient,
        enable_event_streaming: bool = False,
        metrics_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize CrewAI monitor.

        Args:
            client: CrewAI client to monitor
            enable_event_streaming: Enable event streaming
            metrics_callback: Optional callback for metrics updates
        """
        self.client = client
        self.enable_event_streaming = enable_event_streaming
        self.metrics_callback = metrics_callback

        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []

        logger.info("CrewAIMonitor initialized")

    def track_workflow(
        self,
        workflow_id: str,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Track a workflow execution in real-time.

        Args:
            workflow_id: Workflow to track
            callback: Optional callback for status updates

        Returns:
            Current workflow status
        """
        state = self.client.get_workflow_state(workflow_id)

        if not state:
            return {
                "workflow_id": workflow_id,
                "status": "not_found",
                "message": "Workflow not found",
            }

        status = {
            "workflow_id": workflow_id,
            "phase": state.phase,
            "status": state.status,
            "execution_method": state.execution_method,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

        # Add phase-specific metrics
        if state.phase == 2:
            status["solving_progress"] = state.solving_progress
            status["solutions_count"] = len(state.sub_solutions)
        elif state.phase == 3:
            status["critiques_count"] = len(state.critique_reports)
        elif state.phase == 4:
            status["verifications_count"] = len(state.verification_results)
        elif state.phase == 5:
            status["reassembly_complete"] = state.reassembly_result is not None
        elif state.phase == 6:
            status["final_validation_complete"] = state.final_validation is not None
            status["overall_score"] = state.overall_score

        # Callback if provided
        if callback:
            callback(workflow_id, status)

        # Add to metrics history
        if self.enable_event_streaming:
            self.metrics_history.append({
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "status": status,
            })

        return status

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tracked metrics.

        Returns:
            Metrics summary
        """
        # Get client status
        client_status = self.client.get_status()

        # Aggregate workflow metrics
        workflow_ids = self.client.list_workflows()

        total_phases = 0
        successful_workflows = 0
        failed_workflows = 0

        for wf_id in workflow_ids:
            state = self.client.get_workflow_state(wf_id)
            if state:
                total_phases = max(total_phases, state.phase)
                if state.status == WorkflowStatus.COMPLETED:
                    successful_workflows += 1
                elif state.status == WorkflowStatus.FAILED:
                    failed_workflows += 1

        return {
            "client_status": client_status,
            "total_workflows": len(workflow_ids),
            "active_workflows": client_status.get("active_workflows", 0),
            "successful_workflows": successful_workflows,
            "failed_workflows": failed_workflows,
            "total_phases_completed": total_phases,
            "metrics_history_entries": len(self.metrics_history),
        }

    def stream_events(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Get event stream for a workflow.

        Args:
            workflow_id: Workflow to get events for

        Returns:
            List of events
        """
        return [
            event for event in self.metrics_history
            if event["workflow_id"] == workflow_id
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_crewai_client(
    state_storage_dir: str = "./crewai_states",
    enable_persistence: bool = True,
    default_execution_method: ExecutionMethod = ExecutionMethod.AUTO,
) -> CrewAIClient:
    """
    Factory function to create CrewAI client.

    Args:
        state_storage_dir: Directory for state storage
        enable_persistence: Enable state persistence
        default_execution_method: Default execution method

    Returns:
        Configured CrewAIClient instance
    """
    return CrewAIClient(
        state_storage_dir=state_storage_dir,
        enable_persistence=enable_persistence,
        default_execution_method=default_execution_method,
    )


def create_crewai_monitor(
    client: Optional[CrewAIClient] = None,
    enable_event_streaming: bool = False,
    metrics_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> CrewAIMonitor:
    """
    Factory function to create CrewAI monitor.

    Args:
        client: CrewAI client to monitor (creates new if None)
        enable_event_streaming: Enable event streaming
        metrics_callback: Optional callback for metrics updates

    Returns:
        Configured CrewAIMonitor instance
    """
    if client is None:
        client = create_crewai_client()

    return CrewAIMonitor(
        client=client,
        enable_event_streaming=enable_event_streaming,
        metrics_callback=metrics_callback,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example CrewAI client usage
    print("CrewAI Client Example")
    print("=" * 50)

    # Create client
    client = create_crewai_client()

    # Execute a workflow
    result = client.execute_workflow(
        problem_statement="Design a zero-error distributed database system",
        execution_method=ExecutionMethod.ROMA_MDAP_MAKER
    )

    print(f"Workflow result: {result.status}")
    print(f"Duration: {result.metrics.total_duration_seconds:.2f}s")
    print(f"Phases completed: {result.metrics.phases_completed}")

    # Get workflow state
    state = client.get_workflow_state(result.workflow_id)
    if state:
        print(f"Current phase: {state.phase}")
        print(f"Current status: {state.status}")

    # List all workflows
    workflows = client.list_workflows()
    print(f"Total workflows: {len(workflows)}")

    # Monitor workflow
    monitor = create_crewai_monitor(client=client)
    metrics_summary = monitor.get_metrics_summary()
    print(f"Metrics summary: {metrics_summary}")
