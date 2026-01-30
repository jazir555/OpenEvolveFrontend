"""
ACE API Utilities - Standard API Response Formats and Constants

This module provides standardized utilities for all ACE-related API endpoints
to ensure consistency across ace_mcp_tools.py, ace_CREWAI_bridge.py, and
ace_stage6_integration.py.

Key Features:
- Standard error response format
- Parameter naming conventions
- Module-level constants for default values
- Type hints and docstring standards
"""

from typing import Any, Dict, Optional, Union


# ============================================================================
# PARAMETER NAMING CONVENTIONS
# ============================================================================
"""
Parameter Naming Conventions:
- skillbook_path: Path to skillbook JSON file
- storage_path: Path to analytics/performance data files
- checkpoint_dir: Directory for checkpoint files
- filepath: Generic file path
- model: LiteLLM model name (e.g., "gpt-4o-mini")
- workflow_id: Unique identifier for workflow
- problem_statement: The problem to solve
- context: Additional context data
- agent_id: Unique identifier for agent
"""


# ============================================================================
# MODULE CONSTANTS - Standard Default Values
# ============================================================================

# Model Configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_PROMPT_VERSION = "v2.1"

# Skillbook Configuration
DEFAULT_SKILLBOOK_DIR = "./ace_skillbooks"
DEFAULT_MAX_SKILLS = 1000
DEFAULT_MIN_HELPFUL = 5
DEFAULT_DEDUP_THRESHOLD = 0.85

# Analytics Configuration
DEFAULT_ANALYTICS_DIR = "./ace_analytics"
DEFAULT_CHECKPOINT_DIR = "./ace_checkpoints"

# Pattern Mining Configuration
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_MAX_PATTERNS = 10
DEFAULT_MAX_ARTIFACTS = 10000

# Performance Configuration
DEFAULT_MAX_REFLECTOR_WORKERS = 3
DEFAULT_CHECKPOINT_INTERVAL = 100


# ============================================================================
# STANDARD API RESPONSE FORMATTER
# ============================================================================

def create_api_response(
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
    available: bool = True,
) -> Dict[str, Any]:
    """
    Create a standardized API response dictionary.

    This function ensures all ACE API endpoints return consistent response
    formats with proper success/error handling.

    Args:
        success: Whether the operation succeeded
        data: Response data (included on success)
        error: Error message (included on failure)
        error_code: Optional error code for categorization
        available: Whether the service/component is available

    Returns:
        Dict with standardized structure:
        {
            "success": bool,
            "available": bool,
            "data": Any (if success),
            "error": str (if failure),
            "error_code": str (optional, if failure)
        }

    Examples:
        >>> # Success response
        >>> create_api_response(True, data={"key": "value"})
        {"success": True, "available": True, "data": {"key": "value"}}

        >>> # Error response
        >>> create_api_response(False, error="Invalid input", error_code="VAL_001")
        {"success": False, "available": True, "error": "Invalid input", "error_code": "VAL_001"}

        >>> # Service unavailable
        >>> create_api_response(False, error="ACE not available", available=False)
        {"success": False, "available": False, "error": "ACE not available"}
    """
    response: Dict[str, Any] = {
        "success": success,
        "available": available,
    }

    if success:
        if data is not None:
            response["data"] = data
    else:
        response["error"] = error or "Unknown error"
        if error_code:
            response["error_code"] = error_code

    return response


# ============================================================================
# RESPONSE HELPERS
# ============================================================================

def create_success_response(
    data: Any,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a success response with optional message.

    Args:
        data: Response data
        message: Optional success message

    Returns:
        Standardized success response dict
    """
    response = create_api_response(True, data=data)
    if message:
        response["message"] = message
    return response


def create_error_response(
    error: str,
    error_code: Optional[str] = None,
    available: bool = True,
    include_details: bool = False,
    exception: Optional[Exception] = None,
) -> Dict[str, Any]:
    """
    Create an error response with proper error handling.

    Args:
        error: User-facing error message
        error_code: Optional error code
        available: Whether service is available
        include_details: Whether to include exception details (use sparingly)
        exception: Optional exception object for details

    Returns:
        Standardized error response dict
    """
    response = create_api_response(
        success=False,
        error=error,
        error_code=error_code,
        available=available,
    )

    if include_details and exception:
        response["details"] = str(exception)

    return response


def create_unavailable_response(
    component_name: str,
    import_error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a response for unavailable components.

    Args:
        component_name: Name of unavailable component
        import_error: Optional import error message

    Returns:
        Standardized unavailable response dict
    """
    error = f"{component_name} not available"
    if import_error:
        error += f": {import_error}"

    return create_api_response(
        success=False,
        error=error,
        available=False,
    )


# ============================================================================
# DOCSTRING TEMPLATES
# ============================================================================

DOCSTRING_TEMPLATE = """
{one_line_summary}

{detailed_description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}

Examples:
{examples}
"""


def format_function_docstring(
    one_liner: str,
    description: str,
    args: Dict[str, str],
    returns: str,
    raises: Optional[Dict[str, str]] = None,
    examples: Optional[str] = None,
) -> str:
    """
    Format a function docstring with Google/NumPy style.

    Args:
        one_liner: One-line summary
        description: Detailed description
        args: Dict of arg_name -> description
        returns: Description of return value
        raises: Optional dict of exception -> condition
        examples: Optional examples section

    Returns:
        Formatted docstring
    """
    sections = []

    # Summary
    sections.append(one_liner)
    sections.append("")

    # Description
    if description:
        sections.append(description)
        sections.append("")

    # Args
    sections.append("Args:")
    for arg_name, arg_desc in args.items():
        sections.append(f"    {arg_name}: {arg_desc}")
    sections.append("")

    # Returns
    sections.append("Returns:")
    sections.append(f"    {returns}")
    sections.append("")

    # Raises
    if raises:
        sections.append("Raises:")
        for exc, condition in raises.items():
            sections.append(f"    {exc}: {condition}")
        sections.append("")

    # Examples
    if examples:
        sections.append("Examples:")
        sections.append(examples)

    return "\n".join(sections)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT_VERSION",
    "DEFAULT_SKILLBOOK_DIR",
    "DEFAULT_MAX_SKILLS",
    "DEFAULT_MIN_HELPFUL",
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_ANALYTICS_DIR",
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_MIN_CLUSTER_SIZE",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_MAX_PATTERNS",
    "DEFAULT_MAX_ARTIFACTS",
    "DEFAULT_MAX_REFLECTOR_WORKERS",
    "DEFAULT_CHECKPOINT_INTERVAL",
    # Response functions
    "create_api_response",
    "create_success_response",
    "create_error_response",
    "create_unavailable_response",
    # Docstring utilities
    "format_function_docstring",
]
