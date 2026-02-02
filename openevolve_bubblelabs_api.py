"""
OpenEvolve-BubbleLabs API Integration

This module provides a comprehensive API integration between OpenEvolve and BubbleLabs,
enabling full control of OpenEvolve workflows through the BubbleLabs interface.
"""

import json
import asyncio
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

from workflow_structures import WorkflowState
from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from workflow_engine import run_sovereign_workflow
from evolution import run_evolution_loop
# from adversarial_testing import run_adversarial_process  # COMMENTED: Function may not exist
from api_server import team_manager, gauntlet_manager  # Import managers only
from parameter_manager import ParameterManager
from analytics_manager import AnalyticsManager

# Import state machine validation
try:
    import sys
    import os
    # Add parent directory to path to import bubblelabs_crewai_bridge
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bubblelabs_crewai_bridge import (
        ExtendedWorkflowStatus,
        validate_workflow_transition,
        get_valid_workflow_transitions,
        is_terminal_workflow_status
    )
    STATE_VALIDATION_AVAILABLE = True
except ImportError:
    STATE_VALIDATION_AVAILABLE = False
    ExtendedWorkflowStatus = None
    validate_workflow_transition = None
    get_valid_workflow_transitions = None
    is_terminal_workflow_status = None


# =============================================================================
# SECURITY: VALIDATION FUNCTIONS AND WHITELISTS
# =============================================================================

# Whitelist of safe workflow types
ALLOWED_WORKFLOW_TYPES: Set[str] = {
    "evolution",
    "adversarial",
    "sovereign",
    "default"
}

# Whitelist of safe parameters that can be set via setattr
SAFE_PARAMETERS: Set[str] = {
    # Evolution parameters
    "max_iterations",
    "population_size",
    "temperature",
    "top_p",
    "max_tokens",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "num_islands",
    "migration_rate",
    "feature_dimensions",
    "feature_bins",
    "diversity_metric",
    "early_stopping_patience",
    "convergence_threshold",
    "memory_limit_mb",
    "cpu_limit",
    # Workflow parameters
    "problem_statement",
    "content",
    "max_refinement_loops",
    "entanglement_strict_mode",
    # State parameters
    "progress",
    "start_time",
    "end_time",
    "error_message",
    "execution_time"
}


def validate_workflow_type(workflow_type: str) -> str:
    """
    Validate workflow type against whitelist to prevent code injection.

    Args:
        workflow_type: The workflow type to validate

    Returns:
        Validated workflow type

    Raises:
        ValueError: If workflow type is not allowed
    """
    if not workflow_type or not isinstance(workflow_type, str):
        raise ValueError("Workflow type must be a non-empty string")

    workflow_type = workflow_type.strip().lower()

    if workflow_type not in ALLOWED_WORKFLOW_TYPES:
        raise ValueError(
            f"Invalid workflow type: '{workflow_type}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_WORKFLOW_TYPES))}"
        )

    return workflow_type


def validate_parameter_name(param_name: str) -> bool:
    """
    Validate parameter name against whitelist to prevent unsafe attribute manipulation.

    Args:
        param_name: The parameter name to validate

    Returns:
        True if parameter is safe

    Raises:
        ValueError: If parameter is not in whitelist
    """
    if not param_name or not isinstance(param_name, str):
        raise ValueError("Parameter name must be a non-empty string")

    if param_name not in SAFE_PARAMETERS:
        raise ValueError(
            f"Parameter '{param_name}' is not allowed. "
            f"Only whitelisted parameters can be set via user input."
        )

    return True


def validate_parameter_value(param_name: str, param_value: Any) -> Any:
    """
    Validate and sanitize parameter value.

    Args:
        param_name: Name of the parameter
        param_value: Value to validate

    Returns:
        Sanitized value

    Raises:
        ValueError: If value is invalid
    """
    # String parameters: limit length and sanitize
    if isinstance(param_value, str):
        # Limit string length
        if len(param_value) > 100000:
            raise ValueError(f"Parameter '{param_name}' value too long (max 100000 characters)")

        # Check for null bytes
        if "\x00" in param_value:
            raise ValueError(f"Parameter '{param_name}' cannot contain null bytes")

    # List parameters: validate elements
    elif isinstance(param_value, list):
        if len(param_value) > 1000:
            raise ValueError(f"Parameter '{param_name}' list too long (max 1000 elements)")

    # Dict parameters: validate keys and values
    elif isinstance(param_value, dict):
        if len(param_value) > 100:
            raise ValueError(f"Parameter '{param_name}' dict too large (max 100 keys)")

    return param_value


class WorkflowStatus(Enum):
    """Enumeration of possible workflow statuses."""
    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowMetrics:
    """Data class to hold workflow execution metrics."""
    execution_time: float = 0.0
    tokens_used: int = 0
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    diversity: Optional[float] = None
    convergence: Optional[float] = None
    population_size: int = 0
    iterations_completed: int = 0
    total_iterations: int = 0


class OpenEvolveBubbleLabsIntegration:
    """
    Comprehensive integration between OpenEvolve and BubbleLabs platforms.
    Provides full control over OpenEvolve workflows through BubbleLabs interface.
    """

    def __init__(self):
        self.workflow_instances: Dict[str, WorkflowState] = {}
        self.workflow_definitions: Dict[str, Dict[str, Any]] = {}
        self.running_threads: Dict[str, threading.Thread] = {}
        self.team_manager = TeamManager()
        self.gauntlet_manager = GauntletManager()
        self.parameter_manager = ParameterManager()
        self.analytics_manager = AnalyticsManager()
        self.event_callbacks: Dict[str, List[Callable]] = {}

    def register_event_callback(self, event_type: str, callback: Callable):
        """Register a callback for specific events."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)

    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger all callbacks for a specific event."""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                    print(f"Error in event callback: {e}")

    def create_workflow_definition(self,
                                name: str,
                                description: str,
                                workflow_type: str,
                                parameters: Dict[str, Any]) -> str:
        """
        Create a new workflow definition with full OpenEvolve parameter control.

        Args:
            name: Name of the workflow
            description: Description of the workflow
            workflow_type: Type of workflow (evolution, adversarial, etc.)
            parameters: Complete OpenEvolve parameters

        Returns:
            ID of the created workflow definition

        Raises:
            ValueError: If workflow type is invalid
        """
        # Security: Validate workflow type to prevent code injection
        validated_type = validate_workflow_type(workflow_type)

        definition_id = str(uuid.uuid4())

        definition = {
            "id": definition_id,
            "name": name,
            "description": description,
            "workflow_type": validated_type,  # Use validated type
            "parameters": parameters,
            "created_at": time.time(),
            "nodes": self._generate_nodes_for_workflow_type(validated_type, parameters),
            "edges": self._generate_edges_for_workflow_type(validated_type)
        }

        self.workflow_definitions[definition_id] = definition
        return definition_id

    def _generate_nodes_for_workflow_type(self, workflow_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate workflow nodes based on workflow type."""
        if workflow_type == "evolution":
            return [
                {
                    "id": "evolution_content_input",
                    "type": "input",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Content Input",
                        "parameter": "content",
                        "description": "Input content for evolution"
                    }
                },
                {
                    "id": "evolution_content_analysis",
                    "type": "content_analyzer",
                    "position": {"x": 200, "y": 0},
                    "data": {
                        "label": "Content Analysis",
                        "parameter": "content_analyzer_team",
                        "description": "Analyze content and extract context",
                        "team": parameters.get("content_analyzer_team", "")
                    }
                },
                {
                    "id": "evolution_process",
                    "type": "evolution",
                    "position": {"x": 400, "y": 0},
                    "data": {
                        "label": "Evolution Process",
                        "parameter": "evolution_params",
                        "description": "Execute evolution with specified parameters",
                        "max_iterations": parameters.get("max_iterations", 100),
                        "population_size": parameters.get("population_size", 50),
                        "temperature": parameters.get("temperature", 0.7)
                    }
                },
                {
                    "id": "evolution_output",
                    "type": "output",
                    "position": {"x": 600, "y": 0},
                    "data": {
                        "label": "Evolution Output",
                        "parameter": "output",
                        "description": "Final evolved content"
                    }
                }
            ]
        elif workflow_type == "adversarial":
            return [
                {
                    "id": "adversarial_content_input",
                    "type": "input",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Content Input",
                        "parameter": "content",
                        "description": "Input content for adversarial testing"
                    }
                },
                {
                    "id": "adversarial_red_team",
                    "type": "red_team",
                    "position": {"x": 200, "y": -50},
                    "data": {
                        "label": "Red Team",
                        "parameter": "red_team_models",
                        "description": "Attack the content to find vulnerabilities",
                        "models": parameters.get("red_team_models", []),
                        "samples": parameters.get("red_team_samples", 5)
                    }
                },
                {
                    "id": "adversarial_blue_team",
                    "type": "blue_team",
                    "position": {"x": 200, "y": 50},
                    "data": {
                        "label": "Blue Team",
                        "parameter": "blue_team_models",
                        "description": "Defend against red team attacks",
                        "models": parameters.get("blue_team_models", []),
                        "samples": parameters.get("blue_team_samples", 5)
                    }
                },
                {
                    "id": "adversarial_evaluator",
                    "type": "evaluator",
                    "position": {"x": 400, "y": 0},
                    "data": {
                        "label": "Evaluator",
                        "parameter": "evaluator_models",
                        "description": "Judge red vs blue team performance",
                        "models": parameters.get("evaluator_models", []),
                        "samples": parameters.get("evaluator_samples", 5)
                    }
                },
                {
                    "id": "adversarial_output",
                    "type": "output",
                    "position": {"x": 600, "y": 0},
                    "data": {
                        "label": "Adversarial Output",
                        "parameter": "output",
                        "description": "Final hardened content"
                    }
                }
            ]
        elif workflow_type == "sovereign":
            return [
                {
                    "id": "sovereign_input",
                    "type": "input",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Problem Input",
                        "parameter": "problem_statement",
                        "description": "Problem to be solved by sovereign decomposition"
                    }
                },
                {
                    "id": "sovereign_content_analysis",
                    "type": "content_analyzer",
                    "position": {"x": 150, "y": -100},
                    "data": {
                        "label": "Content Analysis",
                        "parameter": "content_analyzer_team",
                        "description": "Analyze problem statement",
                        "team": parameters.get("content_analyzer_team", "")
                    }
                },
                {
                    "id": "sovereign_decomposition",
                    "type": "decomposer",
                    "position": {"x": 150, "y": 0},
                    "data": {
                        "label": "Problem Decomposition",
                        "parameter": "planner_team",
                        "description": "Break down problem into sub-problems",
                        "team": parameters.get("planner_team", "")
                    }
                },
                {
                    "id": "sovereign_verification",
                    "type": "verifier",
                    "position": {"x": 150, "y": 100},
                    "data": {
                        "label": "Sub-problem Verification",
                        "parameter": "assembler_team",
                        "description": "Verify sub-problem solutions",
                        "team": parameters.get("assembler_team", "")
                    }
                },
                {
                    "id": "sovereign_solver",
                    "type": "solver",
                    "position": {"x": 350, "y": 0},
                    "data": {
                        "label": "Sub-problem Solving",
                        "parameter": "solver_team",
                        "description": "Solve individual sub-problems",
                        "team": parameters.get("solver_team", ""),
                        "gauntlet": parameters.get("sub_problem_red_gauntlet", "")
                    }
                },
                {
                    "id": "sovereign_assembly",
                    "type": "assembler",
                    "position": {"x": 550, "y": 0},
                    "data": {
                        "label": "Solution Assembly",
                        "parameter": "assembler_team",
                        "description": "Assemble sub-solutions into final solution",
                        "team": parameters.get("assembler_team", "")
                    }
                }
            ]
        else:
            # Default evolution workflow
            return [
                {
                    "id": "default_input",
                    "type": "input",
                    "position": {"x": 0, "y": 0},
                    "data": {
                        "label": "Content Input",
                        "parameter": "content",
                        "description": "Input content for processing"
                    }
                },
                {
                    "id": "default_processing",
                    "type": "processing",
                    "position": {"x": 200, "y": 0},
                    "data": {
                        "label": "Processing",
                        "parameter": "process_params",
                        "description": "Process content with OpenEvolve"
                    }
                },
                {
                    "id": "default_output",
                    "type": "output",
                    "position": {"x": 400, "y": 0},
                    "data": {
                        "label": "Output",
                        "parameter": "output",
                        "description": "Processed content output"
                    }
                }
            ]

    def _generate_edges_for_workflow_type(self, workflow_type: str) -> List[Dict[str, Any]]:
        """Generate workflow edges based on workflow type."""
        if workflow_type == "evolution":
            return [
                {"id": "edge_1", "source": "evolution_content_input", "target": "evolution_content_analysis"},
                {"id": "edge_2", "source": "evolution_content_analysis", "target": "evolution_process"},
                {"id": "edge_3", "source": "evolution_process", "target": "evolution_output"}
            ]
        elif workflow_type == "adversarial":
            return [
                {"id": "edge_1", "source": "adversarial_content_input", "target": "adversarial_red_team"},
                {"id": "edge_2", "source": "adversarial_content_input", "target": "adversarial_blue_team"},
                {"id": "edge_3", "source": "adversarial_red_team", "target": "adversarial_evaluator"},
                {"id": "edge_4", "source": "adversarial_blue_team", "target": "adversarial_evaluator"},
                {"id": "edge_5", "source": "adversarial_evaluator", "target": "adversarial_output"}
            ]
        elif workflow_type == "sovereign":
            return [
                {"id": "edge_1", "source": "sovereign_input", "target": "sovereign_content_analysis"},
                {"id": "edge_2", "source": "sovereign_input", "target": "sovereign_decomposition"},
                {"id": "edge_3", "source": "sovereign_input", "target": "sovereign_verification"},
                {"id": "edge_4", "source": "sovereign_content_analysis", "target": "sovereign_solver"},
                {"id": "edge_5", "source": "sovereign_decomposition", "target": "sovereign_solver"},
                {"id": "edge_6", "source": "sovereign_verification", "target": "sovereign_solver"},
                {"id": "edge_7", "source": "sovereign_solver", "target": "sovereign_assembly"}
            ]
        else:
            return [
                {"id": "edge_1", "source": "default_input", "target": "default_processing"},
                {"id": "edge_2", "source": "default_processing", "target": "default_output"}
            ]

    def create_workflow_instance(self,
                               definition_id: str,
                               instance_name: str,
                               inputs: Dict[str, Any],
                               parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new workflow instance from a definition.

        Args:
            definition_id: ID of the workflow definition
            instance_name: Name for the instance (NOTE: Currently not used, kept for API compatibility)
            inputs: Input parameters for the workflow
            parameters: Optional override parameters

        Returns:
            ID of the created workflow instance

        Raises:
            ValueError: If workflow definition does not exist
        """
        if definition_id not in self.workflow_definitions:
            raise ValueError(f"Workflow definition {definition_id} not found")
        
        definition = self.workflow_definitions[definition_id]
        
        # Merge definition parameters with instance parameters
        final_parameters = {**definition.get("parameters", {})}
        if parameters:
            final_parameters.update(parameters)

        # Create a new WorkflowState object
        instance_id = str(uuid.uuid4())

        workflow_state = WorkflowState(
            workflow_id=instance_id,
            workflow_type=definition["workflow_type"],
            problem_statement=inputs.get("problem_statement", inputs.get("content", "Default problem")),
            current_stage="created",
            status="created"
        )

        # SECURITY: Apply parameters using whitelist to prevent unsafe attribute manipulation
        for param_name, param_value in final_parameters.items():
            # Validate parameter name against whitelist
            if param_name in SAFE_PARAMETERS and hasattr(workflow_state, param_name):
                # Validate parameter value
                validated_value = validate_parameter_value(param_name, param_value)
                setattr(workflow_state, param_name, validated_value)
            elif param_name in ["openevolve_parameters"] or param_name.startswith(("formal_", "z3_", "leanaide_")):
                continue
            elif param_name not in SAFE_PARAMETERS:
                # Log warning for non-whitelisted parameters
                logger.warning(f"Skipping non-whitelisted parameter: {param_name}")

        # Merge OpenEvolve parameters if provided
        openevolve_params = final_parameters.get("openevolve_parameters")
        if isinstance(openevolve_params, dict):
            workflow_state.openevolve_parameters.update(openevolve_params)
            if "entanglement_strict_mode" in openevolve_params and hasattr(workflow_state, "entanglement_strict_mode"):
                workflow_state.entanglement_strict_mode = bool(openevolve_params.get("entanglement_strict_mode"))

        # Allow root-level formal/z3/leanaide params to be forwarded into openevolve_parameters
        for param_name, param_value in final_parameters.items():
            if param_name.startswith(("formal_", "z3_", "leanaide_")):
                workflow_state.openevolve_parameters[param_name] = param_value
        
        # Set up teams and gauntlets if specified in parameters
        if "content_analyzer_team" in final_parameters:
            workflow_state.content_analyzer_team = self.team_manager.get_team(final_parameters["content_analyzer_team"])
        if "planner_team" in final_parameters:
            workflow_state.planner_team = self.team_manager.get_team(final_parameters["planner_team"])
        if "solver_team" in final_parameters:
            workflow_state.solver_team = self.team_manager.get_team(final_parameters["solver_team"])
        if "patcher_team" in final_parameters:
            workflow_state.patcher_team = self.team_manager.get_team(final_parameters["patcher_team"])
        if "assembler_team" in final_parameters:
            workflow_state.assembler_team = self.team_manager.get_team(final_parameters["assembler_team"])
        
        # Set up gauntlets if specified
        if "sub_problem_red_gauntlet" in final_parameters:
            workflow_state.sub_problem_red_gauntlet = self.gauntlet_manager.get_gauntlet(final_parameters["sub_problem_red_gauntlet"])
        if "sub_problem_gold_gauntlet" in final_parameters:
            workflow_state.sub_problem_gold_gauntlet = self.gauntlet_manager.get_gauntlet(final_parameters["sub_problem_gold_gauntlet"])
        if "final_red_gauntlet" in final_parameters:
            workflow_state.final_red_gauntlet = self.gauntlet_manager.get_gauntlet(final_parameters["final_red_gauntlet"])
        if "final_gold_gauntlet" in final_parameters:
            workflow_state.final_gold_gauntlet = self.gauntlet_manager.get_gauntlet(final_parameters["final_gold_gauntlet"])
            
        # Set up additional inputs
        for input_name, input_value in inputs.items():
            if hasattr(workflow_state, input_name):
                setattr(workflow_state, input_name, input_value)
        
        self.workflow_instances[instance_id] = workflow_state
        
        # Trigger event
        self._trigger_event("workflow_instance_created", {
            "instance_id": instance_id,
            "definition_id": definition_id,
            "status": "created"
        })
        
        return instance_id

    def start_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Start executing a workflow instance with state machine validation.

        Args:
            instance_id: ID of the workflow instance to start

        Returns:
            Dictionary containing:
            - message: Success message
            - instance_id: ID of the workflow instance
            - status: New workflow status
            - error: Error message (if failed)

        Raises:
            KeyError: If instance_id not found (converted to error dict)
            ValueError: If state transition is invalid

        Side Effects:
            - Updates workflow state in memory
            - Starts background thread for workflow execution
            - Triggers workflow_instance_started event
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        workflow_state = self.workflow_instances[instance_id]
        current_status = workflow_state.status
        new_status = WorkflowStatus.PENDING.value

        if workflow_state.status in [WorkflowStatus.RUNNING.value, WorkflowStatus.PAUSED.value]:
            return {"error": f"Workflow already {workflow_state.status}"}

        # Validate state transition if state validation is available
        if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
            valid_transitions = get_valid_workflow_transitions(current_status)
            logger.error(
                f"Invalid workflow state transition: {current_status} -> {new_status}. "
                f"Valid transitions from {current_status}: {valid_transitions}"
            )
            return {
                "error": f"Invalid state transition: {current_status} -> {new_status}",
                "valid_transitions": list(valid_transitions)
            }

        # Update status
        workflow_state.status = new_status
        workflow_state.start_time = time.time()

        # Execute workflow in a background thread
        thread = threading.Thread(target=self._execute_workflow_thread, args=(workflow_state,))
        thread.daemon = True
        thread.start()

        self.running_threads[instance_id] = thread

        # Trigger event
        self._trigger_event("workflow_instance_started", {
            "instance_id": instance_id,
            "status": workflow_state.status
        })

        return {"message": "Workflow started", "instance_id": instance_id, "status": workflow_state.status}

    def _execute_workflow_thread(self, workflow_state: WorkflowState):
        """
        Execute the workflow in a background thread.

        Thread Safety:
            This method runs in a separate daemon thread. Access to workflow_state
            should be thread-safe, and modifications should be atomic where possible.

        Args:
            workflow_state: The workflow state to execute

        Raises:
            ImportError: If required workflow execution functions are not available
            Exception: For workflow execution errors (caught and logged)
        """
        try:
            workflow_state.status = WorkflowStatus.RUNNING.value
            workflow_state.current_stage = "initializing"

            # Record start time for metrics
            start_time = time.time()

            # Execute based on workflow type
            if workflow_state.workflow_type == "evolution":
                # Import with graceful fallback
                try:
                    from evolution import run_evolution_process
                    run_evolution_process(
                        content=workflow_state.problem_statement,
                        workflow_state=workflow_state,
                        max_iterations=workflow_state.max_iterations,
                        population_size=workflow_state.population_size,
                        temperature=workflow_state.temperature,
                        top_p=workflow_state.top_p,
                        max_tokens=workflow_state.max_tokens,
                        frequency_penalty=workflow_state.frequency_penalty,
                        presence_penalty=workflow_state.presence_penalty,
                        seed=workflow_state.seed
                    )
                except ImportError as e:
                    logger.error(f"Evolution module not available: {e}")
                    raise
            elif workflow_state.workflow_type == "adversarial":
                # Import with graceful fallback
                try:
                    from adversarial import run_adversarial_process
                    run_adversarial_process(
                        content=workflow_state.problem_statement,
                        workflow_state=workflow_state
                    )
                except ImportError as e:
                    logger.error(f"Adversarial module not available: {e}")
                    raise
            elif workflow_state.workflow_type == "sovereign":
                run_sovereign_workflow(
                    workflow_state=workflow_state,
                    content_analyzer_team=workflow_state.content_analyzer_team,
                    planner_team=workflow_state.planner_team,
                    solver_team=workflow_state.solver_team,
                    patcher_team=workflow_state.patcher_team,
                    assembler_team=workflow_state.assembler_team,
                    sub_problem_red_gauntlet=workflow_state.sub_problem_red_gauntlet,
                    sub_problem_gold_gauntlet=workflow_state.sub_problem_gold_gauntlet,
                    final_red_gauntlet=workflow_state.final_red_gauntlet,
                    final_gold_gauntlet=workflow_state.final_gold_gauntlet,
                    max_refinement_loops=workflow_state.max_refinement_loops
                )
            else:
                # Default evolution workflow
                run_evolution_process(
                    content=workflow_state.problem_statement,
                    workflow_state=workflow_state
                )
            
            # Calculate final metrics
            workflow_state.execution_time = time.time() - start_time
            workflow_state.status = WorkflowStatus.COMPLETED.value
            workflow_state.end_time = time.time()
            
            # Trigger completion event
            self._trigger_event("workflow_instance_completed", {
                "instance_id": workflow_state.workflow_id,
                "status": workflow_state.status,
                "execution_time": workflow_state.execution_time
            })
            
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            workflow_state.status = WorkflowStatus.FAILED.value
            workflow_state.error_message = str(e)
            
            # Trigger failure event
            self._trigger_event("workflow_instance_failed", {
                "instance_id": workflow_state.workflow_id,
                "status": workflow_state.status,
                "error": str(e)
            })

    def pause_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Pause a running workflow instance with state machine validation.

        Args:
            instance_id: ID of the workflow instance to pause

        Returns:
            Dictionary containing:
            - message: Success message
            - instance_id: ID of the workflow instance
            - status: New workflow status (should be "paused")
            - error: Error message (if failed)

        Raises:
            KeyError: If instance_id not found (converted to error dict)
            ValueError: If workflow is not in running state (converted to error dict)

        Side Effects:
            - Updates workflow state in memory
            - Triggers workflow_instance_paused event
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        workflow_state = self.workflow_instances[instance_id]
        current_status = workflow_state.status

        if workflow_state.status != WorkflowStatus.RUNNING.value:
            return {"error": f"Cannot pause workflow in status: {workflow_state.status}"}

        new_status = WorkflowStatus.PAUSED.value

        # Validate state transition if state validation is available
        if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
            valid_transitions = get_valid_workflow_transitions(current_status)
            logger.error(
                f"Invalid workflow state transition: {current_status} -> {new_status}. "
                f"Valid transitions from {current_status}: {valid_transitions}"
            )
            return {
                "error": f"Invalid state transition: {current_status} -> {new_status}",
                "valid_transitions": list(valid_transitions)
            }

        # Update status
        workflow_state.status = new_status

        # Trigger event
        self._trigger_event("workflow_instance_paused", {
            "instance_id": instance_id,
            "status": workflow_state.status
        })

        return {"message": "Workflow paused", "instance_id": instance_id, "status": workflow_state.status}

    def resume_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Resume a paused workflow instance with state machine validation.

        Args:
            instance_id: ID of the workflow instance to resume

        Returns:
            Status of the resume operation
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        workflow_state = self.workflow_instances[instance_id]
        current_status = workflow_state.status

        if workflow_state.status != WorkflowStatus.PAUSED.value:
            return {"error": f"Cannot resume workflow in status: {workflow_state.status}"}

        new_status = WorkflowStatus.RUNNING.value

        # Validate state transition if state validation is available
        if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
            valid_transitions = get_valid_workflow_transitions(current_status)
            logger.error(
                f"Invalid workflow state transition: {current_status} -> {new_status}. "
                f"Valid transitions from {current_status}: {valid_transitions}"
            )
            return {
                "error": f"Invalid state transition: {current_status} -> {new_status}",
                "valid_transitions": list(valid_transitions)
            }

        # Update status and restart execution
        workflow_state.status = new_status

        # Restart execution in a background thread
        thread = threading.Thread(target=self._execute_workflow_thread, args=(workflow_state,))
        thread.daemon = True
        thread.start()

        self.running_threads[instance_id] = thread

        # Trigger event
        self._trigger_event("workflow_instance_resumed", {
            "instance_id": instance_id,
            "status": workflow_state.status
        })

        return {"message": "Workflow resumed", "instance_id": instance_id, "status": workflow_state.status}

    def stop_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Stop a running workflow instance gracefully with state machine validation.

        Args:
            instance_id: ID of the workflow instance to stop

        Returns:
            Dictionary containing:
            - message: Success message
            - instance_id: ID of the workflow instance
            - status: New workflow status (should be "stopped")
            - error: Error message (if failed)

        Raises:
            KeyError: If instance_id not found (converted to error dict)
            ValueError: If workflow is already stopped (converted to error dict)

        Side Effects:
            - Updates workflow state in memory
            - Cleans up background thread if exists
            - Triggers workflow_instance_stopping and workflow_instance_stopped events
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        workflow_state = self.workflow_instances[instance_id]
        current_status = workflow_state.status

        if workflow_state.status in [WorkflowStatus.STOPPING.value, WorkflowStatus.STOPPED.value,
                                   WorkflowStatus.CANCELLED.value, WorkflowStatus.COMPLETED.value]:
            return {"error": f"Workflow already {workflow_state.status}"}

        # Validate state transition if state validation is available
        stopping_status = WorkflowStatus.STOPPING.value
        if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, stopping_status):
            valid_transitions = get_valid_workflow_transitions(current_status)
            logger.error(
                f"Invalid workflow state transition: {current_status} -> {stopping_status}. "
                f"Valid transitions from {current_status}: {valid_transitions}"
            )
            return {
                "error": f"Invalid state transition: {current_status} -> {stopping_status}",
                "valid_transitions": list(valid_transitions)
            }

        # Update status
        workflow_state.status = stopping_status

        # Trigger event
        self._trigger_event("workflow_instance_stopping", {
            "instance_id": instance_id,
            "status": workflow_state.status
        })

        # For now, we'll mark it as stopped since Python threading doesn't allow easy interruption
        stopped_status = WorkflowStatus.STOPPED.value
        workflow_state.status = stopped_status
        workflow_state.end_time = time.time()

        # Clean up thread if exists
        if instance_id in self.running_threads:
            del self.running_threads[instance_id]

        # Trigger event
        self._trigger_event("workflow_instance_stopped", {
            "instance_id": instance_id,
            "status": workflow_state.status
        })

        return {"message": "Workflow stopped", "instance_id": instance_id, "status": workflow_state.status}

    def cancel_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Cancel a running workflow instance immediately with state machine validation.

        Args:
            instance_id: ID of the workflow instance to cancel

        Returns:
            Dictionary containing:
            - message: Success message
            - instance_id: ID of the workflow instance
            - status: New workflow status (should be "cancelled")
            - error: Error message (if failed)

        Raises:
            KeyError: If instance_id not found (converted to error dict)

        Side Effects:
            - Updates workflow state in memory
            - Cleans up background thread if exists
            - Triggers workflow_instance_cancelled event
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        workflow_state = self.workflow_instances[instance_id]
        current_status = workflow_state.status
        new_status = WorkflowStatus.CANCELLED.value

        # Validate state transition if state validation is available
        if STATE_VALIDATION_AVAILABLE and not validate_workflow_transition(current_status, new_status):
            valid_transitions = get_valid_workflow_transitions(current_status)
            logger.error(
                f"Invalid workflow state transition: {current_status} -> {new_status}. "
                f"Valid transitions from {current_status}: {valid_transitions}"
            )
            return {
                "error": f"Invalid state transition: {current_status} -> {new_status}",
                "valid_transitions": list(valid_transitions)
            }

        # Update status
        workflow_state.status = new_status
        workflow_state.end_time = time.time()

        # Clean up thread if exists
        if instance_id in self.running_threads:
            del self.running_threads[instance_id]

        # Trigger event
        self._trigger_event("workflow_instance_cancelled", {
            "instance_id": instance_id,
            "status": workflow_state.status
        })

        return {"message": "Workflow cancelled", "instance_id": instance_id, "status": workflow_state.status}

    def restart_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Restart a workflow instance with same parameters.

        Args:
            instance_id: ID of the workflow instance to restart

        Returns:
            Dictionary containing:
            - message: Success message
            - original_instance_id: Original instance ID
            - new_instance_id: New instance ID
            - status: New workflow status
            - error: Error message (if failed)

        Raises:
            KeyError: If instance_id not found (converted to error dict)

        Side Effects:
            - Creates new workflow instance in memory
            - Starts background thread for new instance
            - Triggers workflow_instance_restarted event
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        original_workflow_state = self.workflow_instances[instance_id]

        # Create a new instance with the same parameters
        new_instance_id = str(uuid.uuid4())

        # Copy the workflow state to a new instance
        workflow_state = WorkflowState(
            workflow_id=new_instance_id,
            workflow_type=original_workflow_state.workflow_type,
            problem_statement=original_workflow_state.problem_statement,
            current_stage="created",
            status="created"
        )

        # SECURITY: Copy only whitelisted safe attributes
        SAFE_COPY_ATTRIBUTES = {
            # Evolution parameters
            "max_iterations", "population_size", "temperature", "top_p",
            "max_tokens", "frequency_penalty", "presence_penalty", "seed",
            "num_islands", "migration_rate", "feature_dimensions",
            "feature_bins", "diversity_metric", "early_stopping_patience",
            "convergence_threshold", "memory_limit_mb", "cpu_limit",
            # Workflow parameters
            "max_refinement_loops",
            # Teams and gauntlets
            "content_analyzer_team", "planner_team", "solver_team",
            "patcher_team", "assembler_team", "sub_problem_red_gauntlet",
            "sub_problem_gold_gauntlet", "final_red_gauntlet",
            "final_gold_gauntlet"
        }

        # Copy only safe, whitelisted attributes
        for attr_name in SAFE_COPY_ATTRIBUTES:
            if hasattr(original_workflow_state, attr_name) and hasattr(workflow_state, attr_name):
                setattr(workflow_state, attr_name, getattr(original_workflow_state, attr_name))
        
        # Reset workflow-specific attributes
        workflow_state.workflow_id = new_instance_id
        workflow_state.current_stage = "created"
        workflow_state.status = "created"
        workflow_state.start_time = None
        workflow_state.end_time = None
        workflow_state.error_message = None
        
        self.workflow_instances[new_instance_id] = workflow_state
        
        # Start the new instance
        result = self.start_workflow_instance(new_instance_id)
        
        # Trigger event
        self._trigger_event("workflow_instance_restarted", {
            "original_instance_id": instance_id,
            "new_instance_id": new_instance_id,
            "status": workflow_state.status
        })
        
        return {
            "message": "Workflow restarted",
            "original_instance_id": instance_id,
            "new_instance_id": new_instance_id,
            "status": workflow_state.status
        }

    def get_workflow_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow instance.
        
        Args:
            instance_id: ID of the workflow instance
            
        Returns:
            Status information of the workflow instance
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}
        
        workflow_state = self.workflow_instances[instance_id]
        
        status_info = {
            "instance_id": instance_id,
            "status": workflow_state.status,
            "current_stage": workflow_state.current_stage,
            "progress": getattr(workflow_state, 'progress', 0.0),
            "start_time": getattr(workflow_state, 'start_time', None),
            "end_time": getattr(workflow_state, 'end_time', None),
            "execution_time": getattr(workflow_state, 'execution_time', None),
            "error_message": getattr(workflow_state, 'error_message', None)
        }
        
        return status_info

    def list_workflow_instances(self) -> List[Dict[str, Any]]:
        """
        List all workflow instances.
        
        Returns:
            List of workflow instance information
        """
        instances = []
        
        for instance_id, workflow_state in self.workflow_instances.items():
            instance_info = {
                "instance_id": instance_id,
                "workflow_type": workflow_state.workflow_type,
                "status": workflow_state.status,
                "current_stage": workflow_state.current_stage,
                "problem_statement": workflow_state.problem_statement[:50] + "..." if len(workflow_state.problem_statement) > 50 else workflow_state.problem_statement,
                "start_time": getattr(workflow_state, 'start_time', None),
                "end_time": getattr(workflow_state, 'end_time', None),
                "progress": getattr(workflow_state, 'progress', 0.0)
            }
            instances.append(instance_info)
        
        return instances

    def list_workflow_definitions(self) -> List[Dict[str, Any]]:
        """
        List all workflow definitions.
        
        Returns:
            List of workflow definition information
        """
        return [
            {
                "id": def_id,
                "name": definition["name"],
                "description": definition["description"],
                "workflow_type": definition["workflow_type"],
                "created_at": definition["created_at"]
            }
            for def_id, definition in self.workflow_definitions.items()
        ]

    def get_workflow_definition(self, definition_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific workflow definition.
        
        Args:
            definition_id: ID of the workflow definition
            
        Returns:
            Workflow definition information or None if not found
        """
        return self.workflow_definitions.get(definition_id)

    def delete_workflow_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Delete a workflow instance.
        
        Args:
            instance_id: ID of the workflow instance to delete
            
        Returns:
            Status of the deletion operation
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}
        
        # Cancel if running
        if self.workflow_instances[instance_id].status == WorkflowStatus.RUNNING.value:
            self.cancel_workflow_instance(instance_id)
        
        del self.workflow_instances[instance_id]
        
        # Clean up thread if exists
        if instance_id in self.running_threads:
            del self.running_threads[instance_id]
        
        return {"message": "Workflow instance deleted", "instance_id": instance_id}

    def sync_parameters_to_workflow(self, instance_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize parameters to an active workflow instance.

        Args:
            instance_id: ID of the workflow instance
            parameters: Parameters to sync

        Returns:
            Status of the sync operation
        """
        if instance_id not in self.workflow_instances:
            return {"error": f"Workflow instance {instance_id} not found"}

        workflow_state = self.workflow_instances[instance_id]

        # SECURITY: Update parameters using whitelist to prevent unsafe attribute manipulation
        updated_count = 0
        for param_name, param_value in parameters.items():
            # Validate parameter name against whitelist
            if param_name in SAFE_PARAMETERS and hasattr(workflow_state, param_name):
                # Validate parameter value
                validated_value = validate_parameter_value(param_name, param_value)
                setattr(workflow_state, param_name, validated_value)
                updated_count += 1
            elif param_name in ["openevolve_parameters"] or param_name.startswith(("formal_", "z3_", "leanaide_")):
                continue
            elif param_name not in SAFE_PARAMETERS:
                # Log warning for non-whitelisted parameters
                logger.warning(f"Skipping non-whitelisted parameter in sync: {param_name}")

        openevolve_params = parameters.get("openevolve_parameters")
        if isinstance(openevolve_params, dict):
            workflow_state.openevolve_parameters.update(openevolve_params)
            updated_count += 1
            if "entanglement_strict_mode" in openevolve_params and hasattr(workflow_state, "entanglement_strict_mode"):
                workflow_state.entanglement_strict_mode = bool(openevolve_params.get("entanglement_strict_mode"))

        for param_name, param_value in parameters.items():
            if param_name.startswith(("formal_", "z3_", "leanaide_")):
                workflow_state.openevolve_parameters[param_name] = param_value
                updated_count += 1

        return {
            "message": f"Parameters synced successfully ({updated_count} updated)",
            "instance_id": instance_id,
            "updated_count": updated_count
        }


# Global instance of the integration
openevolve_bubblelabs_integration = OpenEvolveBubbleLabsIntegration()
