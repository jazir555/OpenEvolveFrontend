"""
LMQL MCP Tools for OpenEvolve Reliability Integration

This module provides Model Context Protocol (MCP) tools that enable CrewAI
agents to leverage LMQL's constrained generation capabilities for reliable
token-level control over LLM outputs.

MIGRATION NOTICE:
    Previous: CREWAI agent orchestration
    Current: CrewAI agent orchestration (MIT-licensed)

Architecture: CrewAI (Orchestrator) -> LMQL Adapter (Constrained Generation) -> LLM Providers

Key Features:
- Token-level constraints during generation
- Structured data generation with JSON schemas
- ROMA decomposition with LMQL constraints
- MDAP vote generation with confidence bounds
- Constraint validation and templates

Author: OpenEvolve
Version: 1.0.0
"""

import sys
import os
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from functools import wraps

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# LMQL ADAPTER IMPORT
# =============================================================================

LMQL_AVAILABLE = False
LMQL_IMPORT_ERROR = None
LMQL_ADAPTER = None

try:
    from reliability.lmql_adapter import (
        LMQLAdapter,
        get_default_adapter,
        ConstrainedGenerationResult,
        StructuredGenerationResult,
    )
    from reliability.config import get_config
    LMQL_AVAILABLE = True
    logger.info("LMQL adapter imported successfully")
except ImportError as e:
    LMQL_IMPORT_ERROR = str(e)
    logger.warning(f"LMQL adapter not available: {e}")
    # Create stubs for graceful degradation
    LMQLAdapter = None
    get_default_adapter = None
    ConstrainedGenerationResult = None
    StructuredGenerationResult = None

# =============================================================================
# MCP TOOL REGISTRY (Thread-Safe)
# =============================================================================

import threading

_MCP_TOOLS = {}
_MCP_TOOLS_LOCK = threading.Lock()

def mcp_tool(name: str):
    """
    Decorator to register MCP tools (thread-safe).

    Args:
        name: Name of the MCP tool

    Returns:
        Decorator function
    """
    def decorator(func):
        with _MCP_TOOLS_LOCK:
            _MCP_TOOLS[name] = func
            logger.info(f"Registered LMQL MCP tool: {name}")

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_all_tools() -> Dict[str, callable]:
    """
    Get all registered MCP tools (thread-safe).

    Returns:
        Dictionary of tool name to callable
    """
    with _MCP_TOOLS_LOCK:
        return _MCP_TOOLS.copy()


def list_mcp_tools() -> List[str]:
    """
    List names of all registered MCP tools (thread-safe).

    Returns:
        List of tool names
    """
    with _MCP_TOOLS_LOCK:
        return list(_MCP_TOOLS.keys())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracking."""
    return str(uuid.uuid4())


def _validate_constraints(constraints: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate constraint definitions.

    Args:
        constraints: List of constraint dictionaries

    Returns:
        Tuple of (is_valid, errors, normalized_constraints)
    """
    errors = []
    normalized = []

    valid_types = {
        "REGEX", "LENGTH", "FROM_LIST", "JSON_SCHEMA",
        "CUSTOM", "NUMERICAL", "FORMAT", "RANGE"
    }

    for i, constraint in enumerate(constraints):
        if not isinstance(constraint, dict):
            errors.append(f"Constraint {i}: must be a dictionary")
            continue

        # Check required fields
        if "type" not in constraint:
            errors.append(f"Constraint {i}: missing 'type' field")
            continue

        constraint_type = constraint.get("type")

        if constraint_type not in valid_types:
            errors.append(f"Constraint {i}: invalid type '{constraint_type}'. Must be one of {valid_types}")
            continue

        # Validate type-specific fields
        if constraint_type == "REGEX":
            if "value" not in constraint or not isinstance(constraint["value"], str):
                errors.append(f"Constraint {i}: REGEX constraints must have a string 'value' field")
                continue

        elif constraint_type == "LENGTH":
            if "min" in constraint:
                if not isinstance(constraint["min"], int) or constraint["min"] < 0:
                    errors.append(f"Constraint {i}: LENGTH 'min' must be a non-negative integer")
                    continue
            if "max" in constraint:
                if not isinstance(constraint["max"], int) or constraint["max"] < 0:
                    errors.append(f"Constraint {i}: LENGTH 'max' must be a non-negative integer")
                    continue

        elif constraint_type == "FROM_LIST":
            if "value" not in constraint or not isinstance(constraint["value"], list):
                errors.append(f"Constraint {i}: FROM_LIST constraints must have a list 'value' field")
                continue

        elif constraint_type == "JSON_SCHEMA":
            if "value" not in constraint or not isinstance(constraint["value"], dict):
                errors.append(f"Constraint {i}: JSON_SCHEMA constraints must have a dict 'value' field")
                continue

        elif constraint_type == "NUMERICAL":
            if "min" in constraint:
                try:
                    float(constraint["min"])
                except (ValueError, TypeError):
                    errors.append(f"Constraint {i}: NUMERICAL 'min' must be numeric")
                    continue
            if "max" in constraint:
                try:
                    float(constraint["max"])
                except (ValueError, TypeError):
                    errors.append(f"Constraint {i}: NUMERICAL 'max' must be numeric")
                    continue

        # Normalize constraint
        normalized_constraint = {
            "type": constraint_type,
            "field": constraint.get("field", None),
            "value": constraint.get("value", None),
            "description": constraint.get("description", ""),
        }

        # Add type-specific fields
        if constraint_type == "LENGTH":
            normalized_constraint["min"] = constraint.get("min", 0)
            normalized_constraint["max"] = constraint.get("max", 10000)
        elif constraint_type == "NUMERICAL":
            normalized_constraint["min"] = constraint.get("min", float('-inf'))
            normalized_constraint["max"] = constraint.get("max", float('inf'))

        normalized.append(normalized_constraint)

    return len(errors) == 0, errors, normalized


def _get_constraint_templates(category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get available constraint templates.

    Args:
        category: Optional category filter (roma, mdap, leanaide, generic)

    Returns:
        Dictionary of constraint templates
    """
    templates = {
        "roma_subtask_length": {
            "type": "LENGTH",
            "description": "Limit ROMA subtask description length",
            "parameters": {
                "min": 50,
                "max": 2000,
            },
            "category": "roma",
            "example": {
                "type": "LENGTH",
                "field": "subtask_description",
                "min": 50,
                "max": 2000,
                "description": "Subtask must be between 50 and 2000 characters",
            }
        },
        "roma_max_depth": {
            "type": "NUMERICAL",
            "description": "Limit ROMA decomposition depth",
            "parameters": {
                "min": 1,
                "max": 10,
            },
            "category": "roma",
            "example": {
                "type": "NUMERICAL",
                "field": "decomposition_depth",
                "min": 1,
                "max": 10,
                "description": "Decomposition depth must be between 1 and 10",
            }
        },
        "mdap_confidence": {
            "type": "NUMERICAL",
            "description": "MDAP vote confidence range",
            "parameters": {
                "min": 0.0,
                "max": 1.0,
            },
            "category": "mdap",
            "example": {
                "type": "NUMERICAL",
                "field": "confidence",
                "min": 0.0,
                "max": 1.0,
                "description": "Confidence must be between 0.0 and 1.0",
            }
        },
        "mdap_vote_format": {
            "type": "JSON_SCHEMA",
            "description": "MDAP vote JSON schema",
            "parameters": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "vote": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["vote", "confidence"],
                }
            },
            "category": "mdap",
            "example": {
                "type": "JSON_SCHEMA",
                "field": "vote",
                "value": {
                    "type": "object",
                    "properties": {
                        "vote": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["vote", "confidence"],
                },
                "description": "Vote must match JSON schema",
            }
        },
        "leanaide_proof_length": {
            "type": "LENGTH",
            "description": "LeanAide proof length limit",
            "parameters": {
                "min": 10,
                "max": 5000,
            },
            "category": "leanaide",
            "example": {
                "type": "LENGTH",
                "field": "proof",
                "min": 10,
                "max": 5000,
                "description": "Proof must be between 10 and 5000 characters",
            }
        },
        "generic_regex_pattern": {
            "type": "REGEX",
            "description": "Generic regex pattern constraint",
            "parameters": {
                "pattern": "^[a-zA-Z0-9_-]+$",
            },
            "category": "generic",
            "example": {
                "type": "REGEX",
                "field": "identifier",
                "value": "^[a-zA-Z0-9_-]+$",
                "description": "Identifier must match regex pattern",
            }
        },
        "generic_enum_values": {
            "type": "FROM_LIST",
            "description": "Generic enum constraint",
            "parameters": {
                "values": ["option1", "option2", "option3"],
            },
            "category": "generic",
            "example": {
                "type": "FROM_LIST",
                "field": "status",
                "value": ["pending", "in_progress", "completed"],
                "description": "Status must be one of the allowed values",
            }
        },
    }

    # Filter by category if specified
    if category:
        return {
            name: template
            for name, template in templates.items()
            if template.get("category") == category.lower()
        }

    return templates


# =============================================================================
# MCP TOOL 1: Constrained Generation
# =============================================================================

@mcp_tool("lmql_constrained_generation")
def lmql_constrained_generation(
    prompt: str,
    constraints: List[Dict[str, Any]],
    decoding: str = "argmax",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate text with LMQL token-level constraints.

    This tool enables constrained generation where the LLM's output is
    constrained at the token level to satisfy specified constraints.

    Args:
        prompt: The generation prompt
        constraints: List of constraint dictionaries with structure:
            [{
                "type": "REGEX|LENGTH|FROM_LIST|JSON_SCHEMA|CUSTOM|NUMERICAL",
                "field": "field_name",
                "value": constraint_value,
                "description": "optional description"
            }]
        decoding: Decoding method (argmax, beam, sample)
        model: Model to use (default: from config)
        temperature: Sampling temperature (for sample decoding)
        max_tokens: Maximum tokens to generate

    Returns:
    {
        "success": bool,
        "output": str or None,
        "error": str or None,
        "constraint_violations": List[str],
        "decoding_method": str,
        "tokens_used": int,
        "correlation_id": str,
        "fallback_used": bool
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] LMQL constrained generation requested", extra={
        "correlation_id": correlation_id,
        "prompt_length": len(prompt),
        "num_constraints": len(constraints),
        "decoding": decoding,
    })

    # Validate inputs
    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "success": False,
            "output": None,
            "error": "Prompt must be a non-empty string",
            "constraint_violations": [],
            "decoding_method": decoding,
            "tokens_used": 0,
            "correlation_id": correlation_id,
            "fallback_used": False,
        }

    if not isinstance(constraints, list) or not constraints:
        return {
            "success": False,
            "output": None,
            "error": "Constraints must be a non-empty list",
            "constraint_violations": [],
            "decoding_method": decoding,
            "tokens_used": 0,
            "correlation_id": correlation_id,
            "fallback_used": False,
        }

    # Validate constraints
    is_valid, errors, normalized_constraints = _validate_constraints(constraints)
    if not is_valid:
        return {
            "success": False,
            "output": None,
            "error": f"Invalid constraints: {', '.join(errors)}",
            "constraint_violations": [],
            "decoding_method": decoding,
            "tokens_used": 0,
            "correlation_id": correlation_id,
            "fallback_used": False,
        }

    if not LMQL_AVAILABLE:
        logger.warning(f"[{correlation_id}] LMQL not available, using fallback")
        # Fallback: return unconstrained generation
        return {
            "success": False,
            "output": None,
            "error": f"LMQL not available: {LMQL_IMPORT_ERROR}",
            "constraint_violations": [],
            "decoding_method": decoding,
            "tokens_used": 0,
            "correlation_id": correlation_id,
            "fallback_used": True,
        }

    try:
        # Get LMQL adapter
        adapter = get_default_adapter()
        if adapter is None:
            config = get_config()
            adapter = LMQLAdapter(config=config)

        # Perform constrained generation
        result = adapter.constrained_generation(
            prompt=prompt,
            constraints=normalized_constraints,
            decoding=decoding,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Log success
        logger.info(f"[{correlation_id}] LMQL constrained generation completed", extra={
            "correlation_id": correlation_id,
            "success": result.success,
            "tokens_used": getattr(result, 'tokens_used', 0),
            "constraint_violations": len(getattr(result, 'constraint_violations', [])),
        })

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error if not result.success else None,
            "constraint_violations": getattr(result, 'constraint_violations', []),
            "decoding_method": decoding,
            "tokens_used": getattr(result, 'tokens_used', 0),
            "correlation_id": correlation_id,
            "fallback_used": False,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] LMQL constrained generation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "success": False,
            "output": None,
            "error": f"LMQL generation failed: {str(e)}",
            "constraint_violations": [],
            "decoding_method": decoding,
            "tokens_used": 0,
            "correlation_id": correlation_id,
            "fallback_used": False,
        }


# =============================================================================
# MCP TOOL 2: Structured Generation
# =============================================================================

@mcp_tool("lmql_structured_generation")
def lmql_structured_generation(
    prompt: str,
    schema: Dict[str, Any],
    decoding: str = "argmax",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate structured data matching JSON schema.

    This tool ensures that the LLM output conforms to a specified JSON schema,
    enabling reliable structured data extraction.

    Args:
        prompt: The generation prompt
        schema: JSON schema for output structure
        decoding: Decoding method (argmax, beam, sample)

    Returns:
    {
        "success": bool,
        "output": dict or None,
        "error": str or None,
        "schema_valid": bool,
        "correlation_id": str
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] LMQL structured generation requested", extra={
        "correlation_id": correlation_id,
        "prompt_length": len(prompt),
    })

    # Validate inputs
    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "success": False,
            "output": None,
            "error": "Prompt must be a non-empty string",
            "schema_valid": False,
            "correlation_id": correlation_id,
        }

    if not isinstance(schema, dict) or not schema:
        return {
            "success": False,
            "output": None,
            "error": "Schema must be a non-empty dictionary",
            "schema_valid": False,
            "correlation_id": correlation_id,
        }

    if not LMQL_AVAILABLE:
        logger.warning(f"[{correlation_id}] LMQL not available, using fallback")
        return {
            "success": False,
            "output": None,
            "error": f"LMQL not available: {LMQL_IMPORT_ERROR}",
            "schema_valid": True,
            "correlation_id": correlation_id,
        }

    try:
        # Get LMQL adapter
        adapter = get_default_adapter()
        if adapter is None:
            config = get_config()
            adapter = LMQLAdapter(config=config)

        # Perform structured generation
        result = adapter.structured_generation(
            prompt=prompt,
            schema=schema,
            decoding=decoding,
        )

        # Validate schema
        schema_valid = True
        if result.success and result.output:
            # Basic validation: check if output is dict
            schema_valid = isinstance(result.output, dict)

        logger.info(f"[{correlation_id}] LMQL structured generation completed", extra={
            "correlation_id": correlation_id,
            "success": result.success,
            "schema_valid": schema_valid,
        })

        return {
            "success": result.success,
            "output": result.output,
            "error": result.error if not result.success else None,
            "schema_valid": schema_valid,
            "correlation_id": correlation_id,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] LMQL structured generation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "success": False,
            "output": None,
            "error": f"LMQL structured generation failed: {str(e)}",
            "schema_valid": False,
            "correlation_id": correlation_id,
        }


# =============================================================================
# MCP TOOL 3: ROMA Decomposition with Constraints
# =============================================================================

@mcp_tool("lmql_roma_decompose")
def lmql_roma_decompose(
    task: str,
    max_depth: int = 5,
    max_subtasks: int = 10,
    subtask_token_limit: int = 2000,
    dependency_max_depth: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Decompose task using ROMA with LMQL constraints.

    This tool combines ROMA's hierarchical decomposition with LMQL's
    constrained generation to ensure subtasks meet specified constraints.

    Args:
        task: The task to decompose
        max_depth: Maximum decomposition depth (1-10)
        max_subtasks: Maximum subtasks per node (2-50)
        subtask_token_limit: Max tokens per subtask description
        dependency_max_depth: Maximum dependency chain depth

    Returns:
    {
        "success": bool,
        "decomposition": dict or None,
        "error": str or None,
        "stats": {
            "total_nodes": int,
            "max_depth_reached": int,
            "total_tokens": int,
            "constraint_violations": List[str]
        }
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] LMQL ROMA decomposition requested", extra={
        "correlation_id": correlation_id,
        "task_length": len(task),
        "max_depth": max_depth,
        "max_subtasks": max_subtasks,
    })

    # Validate inputs
    if not isinstance(task, str) or not task.strip():
        return {
            "success": False,
            "decomposition": None,
            "error": "Task must be a non-empty string",
            "stats": {},
        }

    # Validate numeric ranges
    if not (1 <= max_depth <= 10):
        return {
            "success": False,
            "decomposition": None,
            "error": "max_depth must be between 1 and 10",
            "stats": {},
        }

    if not (2 <= max_subtasks <= 50):
        return {
            "success": False,
            "decomposition": None,
            "error": "max_subtasks must be between 2 and 50",
            "stats": {},
        }

    if not LMQL_AVAILABLE:
        logger.warning(f"[{correlation_id}] LMQL not available")
        return {
            "success": False,
            "decomposition": None,
            "error": f"LMQL not available: {LMQL_IMPORT_ERROR}",
            "stats": {},
        }

    try:
        # Get LMQL adapter
        adapter = get_default_adapter()
        if adapter is None:
            config = get_config()
            adapter = LMQLAdapter(config=config)

        # Build ROMA-specific constraints
        constraints = [
            {
                "type": "LENGTH",
                "field": "subtask_description",
                "min": 50,
                "max": subtask_token_limit * 4,  # Rough char estimate
                "description": f"Subtask description length constraint",
            },
            {
                "type": "NUMERICAL",
                "field": "decomposition_depth",
                "min": 1,
                "max": max_depth,
                "description": f"Decomposition depth must not exceed {max_depth}",
            },
        ]

        # Perform ROMA decomposition with constraints
        result = adapter.roma_decompose(
            task=task,
            max_depth=max_depth,
            max_subtasks=max_subtasks,
            constraints=constraints,
            dependency_max_depth=dependency_max_depth,
        )

        stats = {
            "total_nodes": 0,
            "max_depth_reached": 0,
            "total_tokens": 0,
            "constraint_violations": [],
        }

        if result.success and result.decomposition:
            # Extract stats
            stats["total_nodes"] = result.decomposition.get("total_subtasks", 0)
            stats["max_depth_reached"] = result.decomposition.get("max_depth", 0)
            stats["total_tokens"] = result.decomposition.get("total_tokens", 0)
            stats["constraint_violations"] = getattr(result, 'constraint_violations', [])

        logger.info(f"[{correlation_id}] LMQL ROMA decomposition completed", extra={
            "correlation_id": correlation_id,
            "success": result.success,
            "total_nodes": stats["total_nodes"],
        })

        return {
            "success": result.success,
            "decomposition": result.decomposition if result.success else None,
            "error": result.error if not result.success else None,
            "stats": stats,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] LMQL ROMA decomposition failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "success": False,
            "decomposition": None,
            "error": f"ROMA decomposition failed: {str(e)}",
            "stats": {},
        }


# =============================================================================
# MCP TOOL 4: MDAP Vote Generation with Constraints
# =============================================================================

@mcp_tool("lmql_generate_mdap_vote")
def lmql_generate_mdap_vote(
    prompt: str,
    vote_schema: Optional[Dict[str, Any]] = None,
    confidence_range: Tuple[float, float] = (0.0, 1.0),
    **kwargs
) -> Dict[str, Any]:
    """
    Generate MDAP vote with LMQL constraints.

    This tool generates MDAP votes with constrained confidence values to
    ensure reliable multi-agent decision making.

    Args:
        prompt: Vote generation prompt
        vote_schema: Optional schema for vote structure
        confidence_range: Valid confidence range (min, max)

    Returns:
    {
        "success": bool,
        "vote": dict or None,
        "confidence": float or None,
        "error": str or None
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] LMQL MDAP vote generation requested", extra={
        "correlation_id": correlation_id,
        "prompt_length": len(prompt),
        "confidence_range": confidence_range,
    })

    # Validate inputs
    if not isinstance(prompt, str) or not prompt.strip():
        return {
            "success": False,
            "vote": None,
            "confidence": None,
            "error": "Prompt must be a non-empty string",
        }

    if not isinstance(confidence_range, tuple) or len(confidence_range) != 2:
        return {
            "success": False,
            "vote": None,
            "confidence": None,
            "error": "confidence_range must be a tuple of (min, max)",
        }

    min_conf, max_conf = confidence_range
    try:
        min_conf = float(min_conf)
        max_conf = float(max_conf)
    except (ValueError, TypeError):
        return {
            "success": False,
            "vote": None,
            "confidence": None,
            "error": "confidence_range values must be numeric",
        }

    if not (0.0 <= min_conf <= max_conf <= 1.0):
        return {
            "success": False,
            "vote": None,
            "confidence": None,
            "error": "confidence_range must be between 0.0 and 1.0",
        }

    if not LMQL_AVAILABLE:
        logger.warning(f"[{correlation_id}] LMQL not available")
        return {
            "success": False,
            "vote": None,
            "confidence": None,
            "error": f"LMQL not available: {LMQL_IMPORT_ERROR}",
        }

    try:
        # Get LMQL adapter
        adapter = get_default_adapter()
        if adapter is None:
            config = get_config()
            adapter = LMQLAdapter(config=config)

        # Build confidence constraint
        confidence_constraint = {
            "type": "NUMERICAL",
            "field": "confidence",
            "min": min_conf,
            "max": max_conf,
            "description": f"Confidence must be between {min_conf} and {max_conf}",
        }

        # Generate vote with confidence constraint
        result = adapter.generate_mdap_vote(
            prompt=prompt,
            vote_schema=vote_schema,
            confidence_constraint=confidence_constraint,
        )

        logger.info(f"[{correlation_id}] LMQL MDAP vote generation completed", extra={
            "correlation_id": correlation_id,
            "success": result.success,
        })

        return {
            "success": result.success,
            "vote": result.vote if result.success else None,
            "confidence": result.confidence if result.success else None,
            "error": result.error if not result.success else None,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] LMQL MDAP vote generation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "success": False,
            "vote": None,
            "confidence": None,
            "error": f"MDAP vote generation failed: {str(e)}",
        }


# =============================================================================
# MCP TOOL 5: Constraint Validation
# =============================================================================

@mcp_tool("lmql_validate_constraints")
def lmql_validate_constraints(
    constraints: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate constraint definitions before use.

    This tool validates that constraint definitions are properly formatted
    and contain all required fields before using them in generation.

    Args:
        constraints: List of constraint dictionaries to validate

    Returns:
    {
        "valid": bool,
        "errors": List[str],
        "warnings": List[str],
        "normalized_constraints": List[Dict]
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] LMQL constraint validation requested", extra={
        "correlation_id": correlation_id,
        "num_constraints": len(constraints) if constraints else 0,
    })

    # Validate inputs
    if not isinstance(constraints, list):
        return {
            "valid": False,
            "errors": ["Constraints must be a list"],
            "warnings": [],
            "normalized_constraints": [],
        }

    if not constraints:
        return {
            "valid": False,
            "errors": ["Constraints list cannot be empty"],
            "warnings": [],
            "normalized_constraints": [],
        }

    # Validate constraints
    is_valid, errors, normalized = _validate_constraints(constraints)

    # Generate warnings
    warnings = []
    for i, constraint in enumerate(normalized):
        if not constraint.get("description"):
            warnings.append(f"Constraint {i}: missing description (recommended)")

        if constraint["type"] == "REGEX" and not constraint.get("field"):
            warnings.append(f"Constraint {i}: REGEX constraint missing 'field' (may not apply correctly)")

    logger.info(f"[{correlation_id}] LMQL constraint validation completed", extra={
        "correlation_id": correlation_id,
        "valid": is_valid,
        "error_count": len(errors),
        "warning_count": len(warnings),
    })

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "normalized_constraints": normalized,
    }


# =============================================================================
# MCP TOOL 6: Get Constraint Templates
# =============================================================================

@mcp_tool("lmql_get_constraint_templates")
def lmql_get_constraint_templates(
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get available constraint templates.

    This tool returns predefined constraint templates that can be used
    as starting points for common constraint patterns.

    Args:
        category: Optional category filter (roma, mdap, leanaide, generic)

    Returns:
    {
        "templates": {
            "template_name": {
                "type": str,
                "description": str,
                "parameters": Dict[str, Any],
                "example": Dict
            }
        }
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] LMQL get constraint templates requested", extra={
        "correlation_id": correlation_id,
        "category": category,
    })

    templates = _get_constraint_templates(category)

    logger.info(f"[{correlation_id}] LMQL constraint templates retrieved", extra={
        "correlation_id": correlation_id,
        "template_count": len(templates),
    })

    return {
        "templates": templates,
    }


# =============================================================================
# MCP TOOL 7: LMQL Status
# =============================================================================

@mcp_tool("lmql_status")
def lmql_status() -> Dict[str, Any]:
    """
    Get LMQL adapter status.

    This tool returns the current status of the LMQL adapter, including
    availability, configuration, and usage statistics.

    Returns:
    {
        "available": bool,
        "enabled": bool,
        "version": str or None,
        "model": str,
        "statistics": {
            "total_requests": int,
            "successful_requests": int,
            "failed_requests": int,
            "avg_latency_ms": float
        }
    }
    """
    logger.info("LMQL status requested")

    if not LMQL_AVAILABLE:
        return {
            "available": False,
            "enabled": False,
            "version": None,
            "model": None,
            "error": LMQL_IMPORT_ERROR,
            "statistics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_latency_ms": 0.0,
            },
        }

    try:
        adapter = get_default_adapter()
        if adapter is None:
            config = get_config()
            adapter = LMQLAdapter(config=config)

        # Get statistics from adapter
        stats = adapter.get_statistics() if hasattr(adapter, 'get_statistics') else {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency_ms": 0.0,
        }

        return {
            "available": True,
            "enabled": True,
            "version": getattr(adapter, 'version', '1.0.0'),
            "model": getattr(adapter, 'model', 'unknown'),
            "statistics": stats,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Failed to get LMQL status: {e}")
        return {
            "available": True,
            "enabled": False,
            "version": None,
            "model": None,
            "error": str(e),
            "statistics": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_latency_ms": 0.0,
            },
        }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all LMQL MCP tools."""
    logger.info("Initializing LMQL MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} LMQL MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
        "lmql_available": LMQL_AVAILABLE,
    }


# Auto-initialize on import
if __name__ != "__main__":
    initialize_mcp_tools()


if __name__ == "__main__":
    print("LMQL MCP Tools Module")
    print(f"LMQL Available: {LMQL_AVAILABLE}")
    print(f"Registered Tools: {len(_MCP_TOOLS)}")
    print("\nTools:")
    for tool_name in sorted(_MCP_TOOLS.keys()):
        print(f"  - {tool_name}")
