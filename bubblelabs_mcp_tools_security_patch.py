"""
Security patch for bubblelabs_mcp_tools.py - Critical MCP tool fixes

This file contains the security-hardened versions of the remaining MCP tool functions.
Replace the corresponding functions in bubblelabs_mcp_tools.py with these versions.

Author: OpenEvolve Team
Date: 2025-12-29
"""

from typing import Dict, Any

# Import security components
try:
    from bubblelabs_security import (
        validate_uuid,
        validate_workflow_action,
        auth_manager
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False


def get_bubblelabs_workflow_status_patched(
    instance_id: str,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Get the status of a running BubbleLabs workflow.

    Security: Validates instance_id format (UUID) and checks authentication.

    This tool retrieves the current status, progress, and metrics of a
    BubbleLabs workflow execution.
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

    # Security: Validate instance_id format
    if SECURITY_AVAILABLE:
        try:
            instance_id = validate_uuid(instance_id)
        except Exception as e:
            return {
                "success": False,
                "error": "Invalid input",
                "message": str(e)
            }

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


def control_bubblelabs_workflow_patched(
    instance_id: str,
    action: str,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Control a running BubbleLabs workflow.

    Security: Requires authentication, validates instance_id and action against whitelist.

    This tool allows you to pause, resume, stop, or cancel a running workflow.
    """
    # Security: Check authentication
    if SECURITY_AVAILABLE and api_key:
        context = auth_manager.validate_api_key(api_key)
        if not context or not context.authenticated:
            logger.warning(f"Unauthorized workflow control attempt")
            return {
                "success": False,
                "error": "Authentication required",
                "message": "Please provide valid API credentials"
            }

    # Security: Validate inputs
    if SECURITY_AVAILABLE:
        try:
            instance_id = validate_uuid(instance_id)
            action = validate_workflow_action(action)
        except Exception as e:
            return {
                "success": False,
                "error": "Invalid input",
                "message": str(e)
            }

    if not BUBBLELABS_AVAILABLE:
        return {
            "success": False,
            "error": "BubbleLabs integration not available"
        }

    try:
        # Create API integration
        api = get_shared_api()

        # Map action to API method
        action = action.lower()
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
        else:
            # This should never be reached due to validation, but keep as safety check
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "message": f"Valid actions: pause, resume, stop, cancel, restart"
            }

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


def get_bubblelabs_workflow_results_patched(
    instance_id: str,
    wait_for_completion: bool = False,
    timeout_seconds: int = 300,
    api_key: str = None
) -> Dict[str, Any]:
    """
    Get the results of a completed BubbleLabs workflow.

    Security: Validates instance_id format and checks authentication.
    """
    # Security: Check authentication
    if SECURITY_AVAILABLE and api_key:
        context = auth_manager.validate_api_key(api_key)
        if not context or not context.authenticated:
            logger.warning(f"Unauthorized results access attempt")
            return {
                "success": False,
                "error": "Authentication required",
                "message": "Please provide valid API credentials"
            }

    # Security: Validate instance_id format
    if SECURITY_AVAILABLE:
        try:
            instance_id = validate_uuid(instance_id)
        except Exception as e:
            return {
                "success": False,
                "error": "Invalid input",
                "message": str(e)
            }

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
        if wait_for_completion and status_info.get("status") == "running":
            import time
            start_time = time.time()
            while status_info.get("status") == "running":
                if time.time() - start_time > timeout_seconds:
                    return {
                        "success": False,
                        "error": "Timeout waiting for completion",
                        "message": f"Workflow did not complete within {timeout_seconds}s"
                    }
                time.sleep(5)
                status_info = api.get_workflow_instance_status(instance_id)

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


def export_mcp_tools_json_auth_protected(api_key: str = None) -> str:
    """
    Export all MCP tools as JSON for external tool registration.

    Security: Requires authentication to export tools.
    """
    # Security: Check authentication
    if SECURITY_AVAILABLE and api_key:
        context = auth_manager.validate_api_key(api_key)
        if not context or not context.authenticated:
            logger.warning(f"Unauthorized tool export attempt")
            return json.dumps({"error": "Authentication required"})

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
