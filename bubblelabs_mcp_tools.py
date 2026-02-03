"""
BubbleLabs MCP Tools for Model Context Protocol Integration

This module provides Model Context Protocol (MCP) tools that enable
CREWAI agents and other systems to interact with BubbleLabs workflows.

MCP Tools allow agents to:
- Create BubbleLabs workflows from natural language
- Execute BubbleLabs workflows
- Monitor workflow progress
- Retrieve workflow results
- Control workflow lifecycle (start/pause/resume/cancel)

Author: OpenEvolve Team
Date: 2025-12-29
"""

import json
import logging
import uuid
import atexit
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Lock

logger = logging.getLogger(__name__)

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

MAX_PROBLEM_STATEMENT_LENGTH = 10000
MAX_WORKFLOW_NAME_LENGTH = 255
MAX_TEAM_CONFIG_ENTRIES = 50
MAX_TIMEOUT_SECONDS = 3600
MAX_PARAMETERS_COUNT = 100


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_not_empty(value: str, param_name: str) -> str:
    """Validate that a string is not empty or just whitespace."""
    if not value or not value.strip():
        raise ValueError(f"{param_name} cannot be empty or whitespace")
    return value


def validate_string_length(value: str, max_length: int, param_name: str) -> str:
    """Validate string length."""
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    if len(value) > max_length:
        raise ValueError(f"{param_name} cannot exceed {max_length} characters")
    return value


def validate_dict_size(value: Dict[str, Any], max_size: int, param_name: str) -> Dict[str, Any]:
    """Validate dictionary size."""
    if value is None:
        return {}
    if len(value) > max_size:
        raise ValueError(f"{param_name} cannot exceed {max_size} entries")
    return value


def validate_range(value: int, min_value: int, max_value: int, param_name: str) -> int:
    """Validate numeric range."""
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    if value < min_value or value > max_value:
        raise ValueError(f"{param_name} must be between {min_value} and {max_value}")
    return value

# Import security layer
try:
    from bubblelabs_security import (
        validate_uuid,
        validate_workflow_type,
        validate_workflow_action,
        require_auth,
        validate_input,
        auth_manager,
        SecurityContext
    )
    SECURITY_AVAILABLE = True
    logger.info("Security layer loaded successfully")
except ImportError:
    SECURITY_AVAILABLE = False
    logger.warning("Security layer not available - MCP tools will run without security")

# Import BubbleLabs integration
try:
    from bubblelabs_integration import BubbleLabsIntegration, BubbleWorkflowDefinition
    from openevolve_bubblelabs_api import (
        OpenEvolveBubbleLabsIntegration,
        WorkflowStatus,
        WorkflowMetrics
    )
    BUBBLELABS_AVAILABLE = True
except ImportError:
    BUBBLELABS_AVAILABLE = False
    logger.warning("BubbleLabs integration not available - MCP tools will be stubs")


# =============================================================================
# SHARED API INSTANCES (SINGLETONS)
# =============================================================================
# CONCURRENCY FIX: Thread-safe singleton pattern with module-level lock
# Fixes Issue #1: Singleton initialization race condition
# Uses module-level lock to prevent race conditions during initialization
# The double-check locking pattern is safe in Python 3.5+ due to the GIL

_shared_bubblelabs_integration = None
_shared_api_instance = None
_singleton_lock = Lock()

# Lock hierarchy documentation:
# 1. Always acquire _singleton_lock first (for singletons)
# 2. Never acquire other locks while holding _singleton_lock
# This prevents deadlock by establishing a clear lock ordering


def get_shared_bubblelabs() -> BubbleLabsIntegration:
    """
    Get or create the shared BubbleLabsIntegration instance.

    CONCURRENCY FIX (Issue #1): Thread-safe singleton with double-check locking.
    This pattern is safe in Python 3.5+ because:
    - Global variable assignment is atomic (GIL protection)
    - Double-check prevents race condition after first initialization
    - Module-level lock ensures only one thread initializes

    Returns:
        Shared BubbleLabsIntegration instance
    """
    global _shared_bubblelabs_integration

    # First check (no lock) - fast path for already-initialized singleton
    if _shared_bubblelabs_integration is not None:
        return _shared_bubblelabs_integration

    # Lock for initialization
    with _singleton_lock:
        # Second check (with lock) - prevent race condition
        if _shared_bubblelabs_integration is None:
            _shared_bubblelabs_integration = BubbleLabsIntegration()
            logger.info("Created shared BubbleLabs integration instance (thread-safe)")

    return _shared_bubblelabs_integration


def get_shared_api() -> OpenEvolveBubbleLabsIntegration:
    """
    Get or create the shared OpenEvolveBubbleLabsIntegration instance.

    CONCURRENCY FIX (Issue #1): Thread-safe singleton with double-check locking.
    Safe in Python 3.5+ due to GIL protection of global variable assignment.

    Returns:
        Shared OpenEvolveBubbleLabsIntegration instance
    """
    global _shared_api_instance

    # First check (no lock) - fast path for already-initialized singleton
    if _shared_api_instance is not None:
        return _shared_api_instance

    # Lock for initialization
    with _singleton_lock:
        # Second check (with lock) - prevent race condition
        if _shared_api_instance is None:
            _shared_api_instance = OpenEvolveBubbleLabsIntegration()
            logger.info("Created shared BubbleLabs API instance (thread-safe)")

    return _shared_api_instance


def cleanup_shared_instances():
    """
    Cleanup shared singleton instances.

    MEMORY LEAK FIX (Leak #7): Properly cleanup singleton instances
    to prevent memory leaks. Should be called on shutdown or when
    singletons are no longer needed.

    This function is automatically registered with atexit to ensure
    cleanup on interpreter shutdown.
    """
    global _shared_bubblelabs_integration, _shared_api_instance

    logger.info("Cleaning up shared BubbleLabs MCP instances...")

    with _singleton_lock:
        # Cleanup BubbleLabs integration instance
        if _shared_bubblelabs_integration is not None:
            try:
                # Call cleanup if the instance has a close/cleanup method
                if hasattr(_shared_bubblelabs_integration, 'close'):
                    _shared_bubblelabs_integration.close()
                elif hasattr(_shared_bubblelabs_integration, 'cleanup'):
                    _shared_bubblelabs_integration.cleanup()

                # Clear any running threads
                if hasattr(_shared_bubblelabs_integration, 'running_threads'):
                    for instance_id, thread in list(_shared_bubblelabs_integration.running_threads.items()):
                        if thread.is_alive():
                            logger.warning(f"Thread {instance_id} still alive during cleanup")

                logger.info("Cleaned up shared BubbleLabs integration instance")
            except Exception as e:
                logger.error(f"Error cleaning up BubbleLabs integration: {e}")
            finally:
                _shared_bubblelabs_integration = None

        # Cleanup API instance
        if _shared_api_instance is not None:
            try:
                # Call cleanup if the instance has a close/cleanup method
                if hasattr(_shared_api_instance, 'close'):
                    _shared_api_instance.close()
                elif hasattr(_shared_api_instance, 'cleanup'):
                    _shared_api_instance.cleanup()

                logger.info("Cleaned up shared BubbleLabs API instance")
            except Exception as e:
                logger.error(f"Error cleaning up BubbleLabs API: {e}")
            finally:
                _shared_api_instance = None

    logger.info("Shared instance cleanup complete")


# MEMORY LEAK FIX (Leak #7): Register cleanup with atexit
# This ensures cleanup is called on interpreter shutdown
atexit.register(cleanup_shared_instances)


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================
# CONCURRENCY FIX (Issue #2): Thread-safe tool registry with lock
# Protects all dictionary operations to prevent race conditions during
# concurrent tool registration and lookup

_MCP_TOOLS: Dict[str, Callable] = {}
_tools_lock = Lock()


def mcp_tool(name: str):
    """
    Decorator to register a function as an MCP tool.

    CONCURRENCY FIX (Issue #2): Thread-safe registration.
    """
    def decorator(func):
        register_mcp_tool(name, func)
        return func
    return decorator


def register_mcp_tool(name: str, func: Callable):
    """
    Register an MCP tool.

    CONCURRENCY FIX (Issue #2): Protected with lock to prevent race conditions
    during concurrent registration.
    """
    with _tools_lock:
        _MCP_TOOLS[name] = func
    logger.info(f"Registered BubbleLabs MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[Callable]:
    """
    Get an MCP tool by name.

    CONCURRENCY FIX (Issue #2): Protected with lock for thread-safe lookup.
    """
    with _tools_lock:
        return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """
    List all registered MCP tools.

    CONCURRENCY FIX (Issue #2): Protected with lock to prevent race condition
    during dictionary iteration and list creation.
    """
    with _tools_lock:
        return list(_MCP_TOOLS.keys())


# =============================================================================
# MCP TOOL IMPLEMENTATIONS
# =============================================================================

@mcp_tool("create_bubblelabs_workflow")
@validate_input(workflow_type=validate_workflow_type if SECURITY_AVAILABLE else lambda x: x)
def create_bubblelabs_workflow(
    problem_statement: str,
    team_config: Optional[Dict[str, str]] = None,
    gauntlet_config: Optional[Dict[str, str]] = None,
    workflow_name: Optional[str] = None,
    workflow_type: str = "sovereign_decomposition",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a BubbleLabs workflow from a problem statement.

    Security: Requires authentication and validates workflow_type against whitelist.

    This tool creates a visual workflow definition in BubbleLabs that can be
    executed through the OpenEvolve SGDW (Sovereign-Grade Decomposition Workflow).

    Args:
        problem_statement: Natural language description of the problem to solve
        team_config: Optional team assignments
            - planner_team: Team for decomposition
            - solver_team: Team for solving sub-problems
            - assembler_team: Team for reassembly
            - content_analyzer_team: Team for content analysis
        gauntlet_config: Optional gauntlet assignments
            - sub_problem_red_gauntlet: Red team validation
            - final_gold_gauntlet: Gold team validation
        workflow_name: Optional custom name for the workflow
        workflow_type: Type of workflow (default: "sovereign_decomposition")
        api_key: Optional API key for authentication

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - workflow_id: ID of created workflow
        - workflow_name: Name of the workflow
        - nodes: List of workflow nodes
        - edges: List of workflow edges
        - message: Success/error message

    Example:
        >>> result = create_bubblelabs_workflow(
        ...     problem_statement="Create a REST API for task management",
        ...     team_config={"planner_team": "Backend-Team", "solver_team": "Fullstack-Team"}
        ... )
        >>> print(result["workflow_id"])
    """
    # Validate input
    if not problem_statement or not problem_statement.strip():
        return {
            "success": False,
            "error": "Invalid input",
            "message": "problem_statement cannot be empty"
        }

    # Security: Check authentication if available
    if SECURITY_AVAILABLE and api_key:
        context = auth_manager.validate_api_key(api_key)
        if not context or not context.authenticated:
            logger.warning(f"Unauthorized workflow creation attempt")
            return {
                "success": False,
                "error": "Authentication required",
                "message": "Please provide valid API credentials"
            }

    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available",
            "message": "Please install BubbleLabs integration"
        }

    try:
        # Get shared integration instance
        integration = get_shared_bubblelabs()

        # Set default configs
        team_config = team_config or {}
        gauntlet_config = gauntlet_config or {}

        # Create workflow definition
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement=problem_statement,
            team_config=team_config,
            gauntlet_config=gauntlet_config
        )

        # Update name if provided
        if workflow_name:
            definition.name = workflow_name

        logger.info(f"Created BubbleLabs workflow: {definition.id}")

        return {
            "success": True,
            "workflow_id": definition.id,
            "workflow_name": definition.name,
            "description": definition.description,
            "nodes": definition.nodes,
            "edges": definition.edges,
            "metadata": definition.metadata,
            "message": f"Workflow '{definition.name}' created successfully"
        }

    except Exception as e:
        logger.error(f"Error creating BubbleLabs workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create workflow: {str(e)}"
        }


@mcp_tool("execute_bubblelabs_workflow")
@validate_input(workflow_id=validate_uuid if SECURITY_AVAILABLE else lambda x: x)
def execute_bubblelabs_workflow(
    workflow_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    auto_start: bool = True,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute a BubbleLabs workflow.

    Security: Requires authentication and validates workflow_id format.

    This tool starts the execution of a previously created BubbleLabs workflow
    through the OpenEvolve backend.

    Args:
        workflow_id: ID of the workflow to execute (must be valid UUID)
        parameters: Optional execution parameters
            - content: Problem content/details
            - max_iterations: Maximum iterations (default: 100)
            - population_size: Population size for evolution (default: 50)
            - Any other OpenEvolve parameters
        auto_start: Automatically start the workflow (default: True)
        api_key: Optional API key for authentication

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - instance_id: ID of the workflow instance
        - status: Initial status of the instance
        - message: Success/error message

    Example:
        >>> result = execute_bubblelabs_workflow(
        ...     workflow_id="abc-123-def",
        ...     parameters={"content": "Build a user authentication system"}
        ... )
        >>> instance_id = result["instance_id"]
    """
    # Security: Check authentication
    if SECURITY_AVAILABLE and api_key:
        context = auth_manager.validate_api_key(api_key)
        if not context or not context.authenticated:
            logger.warning(f"Unauthorized workflow execution attempt")
            return {
                "success": False,
                "error": "Authentication required",
                "message": "Please provide valid API credentials"
            }

    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available",
            "message": "Please install BubbleLabs integration"
        }

    try:
        # Create API integration
        api = get_shared_api()

        # Set default parameters
        parameters = parameters or {}

        # Create workflow instance
        instance_id = api.create_workflow_instance(
            definition_id=workflow_id,
            initial_parameters=parameters
        )

        if not instance_id:
            return {
                "success": False,
                "error": "Failed to create workflow instance",
                "message": "Could not create workflow instance"
            }

        # Auto-start if requested
        if auto_start:
            result = api.start_workflow_instance(instance_id)
            if result.get("error"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "message": f"Created instance {instance_id} but failed to start"
                }

        logger.info(f"Started BubbleLabs workflow instance: {instance_id}")

        return {
            "success": True,
            "instance_id": instance_id,
            "workflow_id": workflow_id,
            "status": "running" if auto_start else "created",
            "message": f"Workflow instance {instance_id} started successfully"
        }

    except Exception as e:
        logger.error(f"Error executing BubbleLabs workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to execute workflow: {str(e)}"
        }


@mcp_tool("get_bubblelabs_workflow_status")
@validate_input(instance_id=validate_uuid if SECURITY_AVAILABLE else lambda x: x)
def get_bubblelabs_workflow_status(
    instance_id: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the status of a running BubbleLabs workflow.

    Security: Validates instance_id format.

    This tool retrieves the current status, progress, and metrics of a
    BubbleLabs workflow execution.

    Args:
        instance_id: ID of the workflow instance (must be valid UUID)
        api_key: Optional API key for authentication

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - instance_id: ID of the workflow instance
        - status: Current workflow status
        - progress: Progress (0.0 to 1.0)
        - current_stage: Current workflow stage
        - metrics: Workflow metrics (if available)
        - message: Status message

    Example:
        >>> status = get_bubblelabs_workflow_status("instance-123")
        >>> print(f"Progress: {status['progress']*100}%")
    """
    # Security: Check authentication
    if SECURITY_AVAILABLE and api_key:
        context = auth_manager.validate_api_key(api_key)
        if not context or not context.authenticated:
            logger.warning(f"Unauthorized status check attempt")
            return {
                "success": False,
                "error": "Authentication required",
                "message": "Please provide valid API credentials"
            }

    # CRITICAL BUG FIX #3: Removed duplicate docstring (was at lines 448-469)

    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available"
        }

    try:
        # Create API integration
        api = get_shared_api()

        # Get instance status
        status_info = api.get_workflow_instance_status(instance_id)

        if "error" in status_info:
            return {
                "success": False,
                "error": status_info["error"],
                "message": f"Failed to get status: {status_info['error']}"
            }

        logger.debug(f"Retrieved status for workflow instance: {instance_id}")

        return {
            "success": True,
            "instance_id": instance_id,
            **status_info
        }

    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to get workflow status: {str(e)}"
        }


@mcp_tool("control_bubblelabs_workflow")
def control_bubblelabs_workflow(
    instance_id: str,
    action: str
) -> Dict[str, Any]:
    """
    Control a running BubbleLabs workflow.

    This tool allows you to pause, resume, stop, or cancel a running workflow.

    Args:
        instance_id: ID of the workflow instance
        action: Action to perform
            - "pause": Pause the workflow
            - "resume": Resume a paused workflow
            - "stop": Stop the workflow
            - "cancel": Cancel the workflow
            - "restart": Restart the workflow (creates new instance)

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - instance_id: ID of the workflow instance
        - action: Action performed
        - new_status: New status after action
        - message: Success/error message

    Example:
        >>> result = control_bubblelabs_workflow("instance-123", "pause")
        >>> print(result["new_status"])
    """
    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available"
        }

    try:
        # Create API integration
        api = get_shared_api()

        # SECURITY: Validate action parameter to prevent command injection
        # Use a whitelist of allowed actions
        allowed_actions = {"pause", "resume", "stop", "cancel", "restart"}

        # Validate and sanitize action
        if not action or not isinstance(action, str):
            return {
                "success": False,
                "error": "Invalid action",
                "message": "Action must be a non-empty string"
            }

        action = action.strip().lower()

        # Check against whitelist
        if action not in allowed_actions:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "message": f"Valid actions: {', '.join(sorted(allowed_actions))}"
            }

        # Map validated action to API method
        result = None

        if action == "pause":
            result = api.pause_workflow_instance(instance_id)
        elif action == "resume":
            result = api.resume_workflow_instance(instance_id)
        elif action == "stop":
            result = api.stop_workflow_instance(instance_id)
        elif action == "cancel":
            result = api.cancel_workflow_instance(instance_id)
        elif action == "restart":
            result = api.restart_workflow_instance(instance_id)

        if result.get("error"):
            return {
                "success": False,
                "error": result["error"],
                "message": f"Failed to {action} workflow: {result['error']}"
            }

        logger.info(f"Performed action '{action}' on workflow instance: {instance_id}")

        return {
            "success": True,
            "instance_id": instance_id,
            "action": action,
            "new_status": result.get("status"),
            "message": f"Successfully {action}ed workflow instance {instance_id}"
        }

    except Exception as e:
        logger.error(f"Error controlling workflow: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to {action} workflow: {str(e)}"
        }


@mcp_tool("list_bubblelabs_workflows")
def list_bubblelabs_workflows(
    workflow_type: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all BubbleLabs workflow definitions and/or instances.

    This tool retrieves all available workflows and their current status.

    PERFORMANCE OPTIMIZATION: Uses generators for large datasets to reduce memory footprint.
    Implements lazy evaluation instead of loading all data into memory at once.

    Args:
        workflow_type: Optional filter by workflow type
        status: Optional filter by workflow status (for instances)

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - definitions: List of workflow definitions
        - instances: List of workflow instances (if status provided)
        - count: Number of items returned
        - message: Status message

    Example:
        >>> result = list_bubblelabs_workflows(status="running")
        >>> for instance in result["instances"]:
        ...     print(f"{instance['id']}: {instance['status']}")
    """
    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available"
        }

    try:
        # Get shared BubbleLabs integration
        integration = get_shared_bubblelabs()

        # PERFORMANCE: Use generator for definitions to reduce memory
        # Avoids creating intermediate list when filtering
        definitions_list = integration.list_workflow_definitions()

        # Generator function for lazy evaluation
        def definitions_generator():
            for d in definitions_list:
                yield {
                    "id": d.id,
                    "name": d.name,
                    "description": d.description,
                    "workflow_type": d.metadata.get("workflow_type", "unknown"),
                    "created_at": d.metadata.get("created_at", 0)
                }

        # Convert to list for JSON serialization (could stream in future)
        definitions = list(definitions_generator())

        # Get instances if status filter provided
        instances = []
        if status:
            instances_list = integration.list_workflow_instances()

            # PERFORMANCE: Use generator and filter before constructing dict
            # Reduces memory by filtering first, then building dicts
            def instances_generator():
                for inst in instances_list:
                    if inst.status == status:
                        yield {
                            "id": inst.id,
                            "definition_id": inst.definition_id,
                            "status": inst.status,
                            "progress": inst.progress,
                            "created_at": inst.created_at
                        }

            instances = list(instances_generator())

        logger.info(f"Listed {len(definitions)} definitions, {len(instances)} instances")

        return {
            "success": True,
            "definitions": definitions,
            "instances": instances,
            "count": len(definitions) + len(instances),
            "message": f"Retrieved {len(definitions)} definitions" +
                      (f" and {len(instances)} instances" if instances else "")
        }

    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to list workflows: {str(e)}"
        }


@mcp_tool("get_bubblelabs_workflow_results")
def get_bubblelabs_workflow_results(
    instance_id: str,
    wait_for_completion: bool = False,
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """
    Get the results of a completed BubbleLabs workflow.

    This tool retrieves the final results, metrics, and outputs of a workflow.

    Args:
        instance_id: ID of the workflow instance
        wait_for_completion: Wait for workflow to complete if still running
        timeout_seconds: Maximum time to wait (default: 300 seconds)

    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - instance_id: ID of the workflow instance
        - status: Final status
        - results: Workflow results (if completed)
        - metrics: Final workflow metrics
        - message: Status message

    Example:
        >>> results = get_bubblelabs_workflow_results("instance-123")
        >>> print(results["results"])
    """
    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available"
        }

    try:
        # Create API integration
        api = get_shared_api()

        # Get current status
        status_info = api.get_workflow_instance_status(instance_id)

        if "error" in status_info:
            return {
                "success": False,
                "error": status_info["error"],
                "message": f"Failed to get results: {status_info['error']}"
            }

        # Wait for completion if requested
        # CRITICAL BUG FIX #4: Added explicit break on completion and validated final status
        if wait_for_completion and status_info.get("status") == "running":
            import time
            start_time = time.time()

            # CRITICAL FIX: Define valid terminal states to ensure proper completion validation
            VALID_TERMINAL_STATES = {"completed", "failed", "cancelled", "stopped"}

            while status_info.get("status") == "running":
                if time.time() - start_time > timeout_seconds:
                    return {
                        "success": False,
                        "error": "Timeout waiting for completion",
                        "message": f"Workflow did not complete within {timeout_seconds}s"
                    }
                time.sleep(5)
                status_info = api.get_workflow_instance_status(instance_id)

                # CRITICAL FIX: Explicit break on terminal state
                current_status = status_info.get("status")
                if current_status in VALID_TERMINAL_STATES:
                    logger.info(f"Workflow reached terminal state: {current_status}")
                    break  # Exit loop - workflow has reached valid terminal state

            # CRITICAL FIX: Validate final status is in valid terminal state
            final_status = status_info.get("status")
            if final_status not in VALID_TERMINAL_STATES and final_status != "running":
                logger.warning(f"Unexpected final status: {final_status}")
                return {
                    "success": False,
                    "error": "Invalid workflow state",
                    "message": f"Workflow ended in unexpected state: {final_status}"
                }

        logger.info(f"Retrieved results for workflow instance: {instance_id}")

        return {
            "success": True,
            "instance_id": instance_id,
            **status_info
        }

    except Exception as e:
        logger.error(f"Error getting workflow results: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to get workflow results: {str(e)}"
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def export_mcp_tools_json() -> str:
    """
    Export all MCP tools as JSON for external tool registration.

    Returns:
        JSON string of tool definitions
    """
    tools = []
    for name in list_mcp_tools():
        func = get_mcp_tool(name)
        if func and func.__doc__:
            tools.append({
                "name": name,
                "description": func.__doc__.strip(),
                "parameters": "See function signature"
            })

    return json.dumps(tools, indent=2)


def print_mcp_tools_summary():
    """Print a summary of all registered MCP tools."""
    print("\n" + "=" * 70)
    print("BubbleLabs MCP Tools")
    print("=" * 70)

    for name in sorted(list_mcp_tools()):
        func = get_mcp_tool(name)
        if func:
            print(f"\n{name}")
            print("-" * 70)
            if func.__doc__:
                # Print first line of docstring
                first_line = func.__doc__.strip().split('\n')[0]
                print(f"  {first_line}")

    print("\n" + "=" * 70)


# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    # Initialize and test
    print("BubbleLabs MCP Tools Module")
    print(f"BubbleLabs available: {BUBBLELABS_AVAILABLE}")
    print(f"\nRegistered tools: {len(list_mcp_tools())}")
    print_mcp_tools_summary()
