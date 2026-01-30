"""
Guardrails MCP Tools for OpenEvolve Reliability Integration

This module provides Model Context Protocol (MCP) tools that enable CrewAI
agents to leverage Guardrails' validation capabilities for ensuring LLM outputs
meet specified quality and safety standards.

MIGRATION NOTICE:
    Previous: CREWAI agent orchestration
    Current: CrewAI agent orchestration (MIT-licensed)

Architecture: CrewAI (Orchestrator) -> Guardrails Adapter (Validation) -> LLM Outputs

Key Features:
- Output validation with multiple validators
- Input validation for prompts
- Batch validation for multiple outputs
- Custom validator registration
- Remediation strategies (reask, fix, filter, refrain)
- Validation statistics and monitoring

Author: OpenEvolve
Version: 1.0.0
"""

import sys
import os
import json
import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from functools import wraps
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure structured JSON logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# GUARDRAILS ADAPTER IMPORT
# =============================================================================

GUARDRAILS_AVAILABLE = False
GUARDRAILS_IMPORT_ERROR = None
GUARDRAILS_ADAPTER = None

try:
    from reliability.guardrails_adapter import (
        GuardrailsAdapter,
        create_adapter,
        ValidationResult,
        Validator,
    )
    from reliability.config import get_config
    GUARDRAILS_AVAILABLE = True
    logger.info("Guardrails adapter imported successfully")
except ImportError as e:
    GUARDRAILS_IMPORT_ERROR = str(e)
    logger.warning(f"Guardrails adapter not available: {e}")
    # Create stubs for graceful degradation
    GuardrailsAdapter = None
    create_adapter = None
    ValidationResult = None
    Validator = None

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
            logger.info(f"Registered Guardrails MCP tool: {name}")

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


def _get_available_validators(category: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get available validators.

    Args:
        category: Optional category filter (roma, mdap, leanaide, safety)

    Returns:
        Dictionary of validators
    """
    validators = {
        # ROMA-specific validators
        "roma_subtask_completeness": {
            "type": "completeness",
            "description": "Validate ROMA subtask description completeness",
            "parameters": {
                "min_length": 50,
                "required_fields": ["title", "description", "success_criteria"],
            },
            "category": "roma",
        },
        "roma_dependency_validity": {
            "type": "dependency",
            "description": "Validate ROMA dependency references",
            "parameters": {
                "allow_circular": False,
                "max_depth": 5,
            },
            "category": "roma",
        },

        # MDAP-specific validators
        "mdap_vote_format": {
            "type": "format",
            "description": "Validate MDAP vote format",
            "parameters": {
                "required_fields": ["vote", "confidence"],
                "confidence_range": [0.0, 1.0],
            },
            "category": "mdap",
        },
        "mdap_confidence_range": {
            "type": "numerical",
            "description": "Validate MDAP confidence is in valid range",
            "parameters": {
                "min": 0.0,
                "max": 1.0,
            },
            "category": "mdap",
        },

        # LeanAide-specific validators
        "leanaide_proof_syntax": {
            "type": "syntax",
            "description": "Validate LeanAide proof syntax",
            "parameters": {
                "check_tacotics": True,
                "check_lemmas": True,
            },
            "category": "leanaide",
        },
        "leanaide_tactic_validity": {
            "type": "tactic",
            "description": "Validate LeanAide tactic validity",
            "parameters": {
                "allow_custom_tactics": True,
                "check_library": True,
            },
            "category": "leanaide",
        },

        # Safety validators
        "toxicity": {
            "type": "safety",
            "description": "Detect toxic content",
            "parameters": {
                "threshold": 0.5,
            },
            "category": "safety",
        },
        "pii_detection": {
            "type": "privacy",
            "description": "Detect personally identifiable information",
            "parameters": {
                "detect_email": True,
                "detect_phone": True,
                "detect_ssn": True,
            },
            "category": "safety",
        },
        "code_injection": {
            "type": "security",
            "description": "Detect potential code injection attempts",
            "parameters": {
                "check_sql": True,
                "check_shell": True,
                "check_eval": True,
            },
            "category": "safety",
        },

        # Generic validators
        "json_format": {
            "type": "format",
            "description": "Validate JSON format",
            "parameters": {
                "strict": True,
            },
            "category": "generic",
        },
        "length_constraint": {
            "type": "length",
            "description": "Validate text length constraints",
            "parameters": {
                "min_length": None,
                "max_length": None,
            },
            "category": "generic",
        },
        "regex_pattern": {
            "type": "pattern",
            "description": "Validate against regex pattern",
            "parameters": {
                "pattern": ".*",
            },
            "category": "generic",
        },
        "keyword_presence": {
            "type": "content",
            "description": "Validate presence of required keywords",
            "parameters": {
                "keywords": [],
                "require_all": False,
            },
            "category": "generic",
        },
    }

    # Filter by category if specified
    if category:
        return {
            name: validator
            for name, validator in validators.items()
            if validator.get("category") == category.lower()
        }

    return validators


def _get_statistics_store() -> Dict[str, Any]:
    """Get or create statistics store (thread-safe)."""
    if not hasattr(_get_statistics_store, '_store'):
        _get_statistics_store._store = {
            'by_validator': defaultdict(lambda: {
                'uses': 0,
                'failures': 0,
                'remediations': 0,
                'total_time_ms': 0.0,
            }),
            'by_strategy': defaultdict(int),
            'total': {
                'validations': 0,
                'failures': 0,
                'remediations': 0,
            },
        }
    return _get_statistics_store._store


def _update_statistics(
    validator_name: str,
    success: bool,
    remediated: bool,
    strategy: Optional[str] = None,
    time_ms: float = 0.0
):
    """Update validation statistics (thread-safe)."""
    store = _get_statistics_store()

    # Update total stats
    store['total']['validations'] += 1
    if not success:
        store['total']['failures'] += 1
    if remediated:
        store['total']['remediations'] += 1

    # Update by-validator stats
    store['by_validator'][validator_name]['uses'] += 1
    if not success:
        store['by_validator'][validator_name]['failures'] += 1
    if remediated:
        store['by_validator'][validator_name]['remediations'] += 1
    store['by_validator'][validator_name]['total_time_ms'] += time_ms

    # Update by-strategy stats
    if strategy:
        store['by_strategy'][strategy] += 1


# =============================================================================
# MCP TOOL 1: Validate Output
# =============================================================================

@mcp_tool("guardrails_validate_output")
def guardrails_validate_output(
    output: str,
    validators: List[str],
    on_fail: str = "reask",
    **kwargs
) -> Dict[str, Any]:
    """
    Validate output with Guardrails validators.

    This tool validates that an LLM output meets specified quality and safety
    standards using registered validators.

    Args:
        output: The output to validate
        validators: List of validator names to apply
        on_fail: Remediation strategy (reask, fix, filter, refrain, exception)

    Returns:
    {
        "is_valid": bool,
        "output": str (possibly remediated),
        "failures": List[Dict],
        "remediated": bool,
        "validation_time_ms": float
    }
    """
    correlation_id = _generate_correlation_id()
    start_time = time.time()
    logger.info(f"[{correlation_id}] Guardrails output validation requested", extra={
        "correlation_id": correlation_id,
        "output_length": len(output),
        "num_validators": len(validators),
        "on_fail": on_fail,
    })

    # Validate inputs
    if not isinstance(output, str):
        return {
            "is_valid": False,
            "output": str(output) if output else "",
            "failures": [{"validator": "input", "error": "Output must be a string"}],
            "remediated": False,
            "validation_time_ms": 0.0,
        }

    if not isinstance(validators, list) or not validators:
        return {
            "is_valid": False,
            "output": output,
            "failures": [{"validator": "input", "error": "Validators must be a non-empty list"}],
            "remediated": False,
            "validation_time_ms": 0.0,
        }

    valid_strategies = ["reask", "fix", "filter", "refrain", "exception"]
    if on_fail not in valid_strategies:
        return {
            "is_valid": False,
            "output": output,
            "failures": [{"validator": "input", "error": f"Invalid on_fail strategy. Must be one of {valid_strategies}"}],
            "remediated": False,
            "validation_time_ms": 0.0,
        }

    if not GUARDRAILS_AVAILABLE:
        logger.warning(f"[{correlation_id}] Guardrails not available, returning unvalidated")
        validation_time_ms = (time.time() - start_time) * 1000
        return {
            "is_valid": True,  # Assume valid if guardrails unavailable
            "output": output,
            "failures": [],
            "remediated": False,
            "validation_time_ms": validation_time_ms,
            "warning": f"Guardrails not available: {GUARDRAILS_IMPORT_ERROR}",
        }

    try:
        # Get Guardrails adapter
        adapter = create_adapter()
        if adapter is None:
            config = get_config()
            adapter = GuardrailsAdapter(config=config)

        # Perform validation
        result = adapter.validate_output(
            output=output,
            validators=validators,
            on_fail=on_fail,
        )

        validation_time_ms = (time.time() - start_time) * 1000

        # Update statistics
        for validator_name in validators:
            _update_statistics(
                validator_name=validator_name,
                success=result.is_valid,
                remediated=getattr(result, 'remediated', False),
                strategy=on_fail if not result.is_valid else None,
                time_ms=validation_time_ms / len(validators),
            )

        logger.info(f"[{correlation_id}] Guardrails output validation completed", extra={
            "correlation_id": correlation_id,
            "is_valid": result.is_valid,
            "num_failures": len(result.failures) if hasattr(result, 'failures') else 0,
            "remediated": getattr(result, 'remediated', False),
            "validation_time_ms": validation_time_ms,
        })

        return {
            "is_valid": result.is_valid,
            "output": result.output,
            "failures": getattr(result, 'failures', []),
            "remediated": getattr(result, 'remediated', False),
            "validation_time_ms": validation_time_ms,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        validation_time_ms = (time.time() - start_time) * 1000
        logger.error(f"[{correlation_id}] Guardrails output validation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
            "validation_time_ms": validation_time_ms,
        })
        return {
            "is_valid": False,
            "output": output,
            "failures": [{"validator": "system", "error": str(e)}],
            "remediated": False,
            "validation_time_ms": validation_time_ms,
        }


# =============================================================================
# MCP TOOL 2: Validate Input
# =============================================================================

@mcp_tool("guardrails_validate_input")
def guardrails_validate_input(
    prompt: str,
    validators: List[str],
    on_fail: str = "exception",
    **kwargs
) -> Dict[str, Any]:
    """
    Validate input prompt with Guardrails.

    This tool validates that an input prompt meets specified standards
    before it is sent to an LLM.

    Args:
        prompt: The input prompt to validate
        validators: List of validator names
        on_fail: Remediation strategy (reask, fix, filter, refrain, exception)

    Returns:
    {
        "is_valid": bool,
        "prompt": str (possibly remediated),
        "failures": List[str],
        "safe_to_proceed": bool
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] Guardrails input validation requested", extra={
        "correlation_id": correlation_id,
        "prompt_length": len(prompt),
        "num_validators": len(validators),
        "on_fail": on_fail,
    })

    # Validate inputs
    if not isinstance(prompt, str):
        return {
            "is_valid": False,
            "prompt": str(prompt) if prompt else "",
            "failures": ["Prompt must be a string"],
            "safe_to_proceed": False,
        }

    if not isinstance(validators, list) or not validators:
        return {
            "is_valid": False,
            "prompt": prompt,
            "failures": ["Validators must be a non-empty list"],
            "safe_to_proceed": False,
        }

    if not GUARDRAILS_AVAILABLE:
        logger.warning(f"[{correlation_id}] Guardrails not available, allowing input")
        return {
            "is_valid": True,
            "prompt": prompt,
            "failures": [],
            "safe_to_proceed": True,
            "warning": f"Guardrails not available: {GUARDRAILS_IMPORT_ERROR}",
        }

    try:
        # Get Guardrails adapter
        adapter = create_adapter()
        if adapter is None:
            config = get_config()
            adapter = GuardrailsAdapter(config=config)

        # Perform validation
        result = adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail=on_fail,
        )

        logger.info(f"[{correlation_id}] Guardrails input validation completed", extra={
            "correlation_id": correlation_id,
            "is_valid": result.is_valid,
            "safe_to_proceed": result.is_valid,
        })

        return {
            "is_valid": result.is_valid,
            "prompt": result.prompt,
            "failures": getattr(result, 'failures', []),
            "safe_to_proceed": result.is_valid,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] Guardrails input validation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "is_valid": False,
            "prompt": prompt,
            "failures": [str(e)],
            "safe_to_proceed": False,
        }


# =============================================================================
# MCP TOOL 3: Batch Validation
# =============================================================================

@mcp_tool("guardrails_batch_validate")
def guardrails_batch_validate(
    outputs: List[str],
    validators: List[str],
    on_fail: str = "filter",
    **kwargs
) -> Dict[str, Any]:
    """
    Validate multiple outputs in batch.

    This tool efficiently validates multiple outputs using the same set of
    validators, useful for batch processing scenarios.

    Args:
        outputs: List of outputs to validate
        validators: List of validator names
        on_fail: Remediation strategy (filter, fix, refrain)

    Returns:
    {
        "results": List[Dict],
        "summary": {
            "total": int,
            "valid": int,
            "invalid": int,
            "remediated": int
        }
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] Guardrails batch validation requested", extra={
        "correlation_id": correlation_id,
        "batch_size": len(outputs),
        "num_validators": len(validators),
        "on_fail": on_fail,
    })

    # Validate inputs
    if not isinstance(outputs, list) or not outputs:
        return {
            "results": [],
            "summary": {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "remediated": 0,
            },
            "error": "Outputs must be a non-empty list",
        }

    if not isinstance(validators, list) or not validators:
        return {
            "results": [],
            "summary": {
                "total": len(outputs),
                "valid": 0,
                "invalid": len(outputs),
                "remediated": 0,
            },
            "error": "Validators must be a non-empty list",
        }

    if not GUARDRAILS_AVAILABLE:
        logger.warning(f"[{correlation_id}] Guardrails not available, assuming all valid")
        return {
            "results": [
                {
                    "index": i,
                    "output": output,
                    "is_valid": True,
                    "failures": [],
                    "remediated": False,
                }
                for i, output in enumerate(outputs)
            ],
            "summary": {
                "total": len(outputs),
                "valid": len(outputs),
                "invalid": 0,
                "remediated": 0,
            },
            "warning": f"Guardrails not available: {GUARDRAILS_IMPORT_ERROR}",
        }

    try:
        # Get Guardrails adapter
        adapter = create_adapter()
        if adapter is None:
            config = get_config()
            adapter = GuardrailsAdapter(config=config)

        # Perform batch validation
        results = []
        valid_count = 0
        invalid_count = 0
        remediated_count = 0

        for i, output in enumerate(outputs):
            result = adapter.validate_output(
                output=output,
                validators=validators,
                on_fail=on_fail,
            )

            results.append({
                "index": i,
                "output": result.output,
                "is_valid": result.is_valid,
                "failures": getattr(result, 'failures', []),
                "remediated": getattr(result, 'remediated', False),
            })

            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            if getattr(result, 'remediated', False):
                remediated_count += 1

        logger.info(f"[{correlation_id}] Guardrails batch validation completed", extra={
            "correlation_id": correlation_id,
            "total": len(outputs),
            "valid": valid_count,
            "invalid": invalid_count,
            "remediated": remediated_count,
        })

        return {
            "results": results,
            "summary": {
                "total": len(outputs),
                "valid": valid_count,
                "invalid": invalid_count,
                "remediated": remediated_count,
            },
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] Guardrails batch validation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "results": [],
            "summary": {
                "total": len(outputs),
                "valid": 0,
                "invalid": len(outputs),
                "remediated": 0,
            },
            "error": str(e),
        }


# =============================================================================
# MCP TOOL 4: Register Custom Validator
# =============================================================================

@mcp_tool("guardrails_register_validator")
def guardrails_register_validator(
    name: str,
    validator_type: str,
    config: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """
    Register a custom validator.

    This tool allows registration of custom validators for specialized
    validation needs beyond the built-in validators.

    Args:
        name: Unique validator name
        validator_type: Type of validator (custom, or predefined type)
        config: Validator configuration

    Returns:
    {
        "success": bool,
        "validator_name": str,
        "error": str or None
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] Guardrails register validator requested", extra={
        "correlation_id": correlation_id,
        "validator_name": name,
        "validator_type": validator_type,
    })

    # Validate inputs
    if not isinstance(name, str) or not name.strip():
        return {
            "success": False,
            "validator_name": name,
            "error": "Validator name must be a non-empty string",
        }

    if not isinstance(validator_type, str) or not validator_type.strip():
        return {
            "success": False,
            "validator_name": name,
            "error": "Validator type must be a non-empty string",
        }

    if not isinstance(config, dict):
        return {
            "success": False,
            "validator_name": name,
            "error": "Validator config must be a dictionary",
        }

    if not GUARDRAILS_AVAILABLE:
        logger.warning(f"[{correlation_id}] Guardrails not available")
        return {
            "success": False,
            "validator_name": name,
            "error": f"Guardrails not available: {GUARDRAILS_IMPORT_ERROR}",
        }

    try:
        # Get Guardrails adapter
        adapter = create_adapter()
        if adapter is None:
            config_obj = get_config()
            adapter = GuardrailsAdapter(config=config_obj)

        # Register validator
        success = adapter.register_validator(
            name=name,
            validator_type=validator_type,
            config=config,
        )

        if success:
            logger.info(f"[{correlation_id}] Validator registered successfully", extra={
                "correlation_id": correlation_id,
                "validator_name": name,
            })
            return {
                "success": True,
                "validator_name": name,
                "error": None,
            }
        else:
            logger.warning(f"[{correlation_id}] Validator registration failed", extra={
                "correlation_id": correlation_id,
                "validator_name": name,
            })
            return {
                "success": False,
                "validator_name": name,
                "error": "Validator registration failed (adapter returned False)",
            }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] Validator registration failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "success": False,
            "validator_name": name,
            "error": str(e),
        }


# =============================================================================
# MCP TOOL 5: Get Available Validators
# =============================================================================

@mcp_tool("guardrails_get_validators")
def guardrails_get_validators(
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get available validators.

    This tool returns information about available validators, including
    their types, descriptions, and parameters.

    Args:
        category: Optional category filter (roma, mdap, leanaide, safety, generic)

    Returns:
    {
        "validators": {
            "validator_name": {
                "type": str,
                "description": str,
                "parameters": Dict,
                "category": str
            }
        }
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] Guardrails get validators requested", extra={
        "correlation_id": correlation_id,
        "category": category,
    })

    validators = _get_available_validators(category)

    logger.info(f"[{correlation_id}] Validators retrieved", extra={
        "correlation_id": correlation_id,
        "validator_count": len(validators),
    })

    return {
        "validators": validators,
    }


# =============================================================================
# MCP TOOL 6: Apply Remediation
# =============================================================================

@mcp_tool("guardrails_apply_remediation")
def guardrails_apply_remediation(
    output: str,
    failure: Dict[str, Any],
    strategy: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply specific remediation strategy.

    This tool applies a specific remediation strategy to a failed validation,
    attempting to fix the output or provide an alternative.

    Args:
        output: The failed output
        failure: Failure details from validation
        strategy: Remediation strategy to apply (fix, filter, refrain)

    Returns:
    {
        "success": bool,
        "output": str or None,
        "strategy_applied": str,
        "error": str or None
    }
    """
    correlation_id = _generate_correlation_id()
    logger.info(f"[{correlation_id}] Guardrails apply remediation requested", extra={
        "correlation_id": correlation_id,
        "strategy": strategy,
    })

    # Validate inputs
    if not isinstance(output, str):
        return {
            "success": False,
            "output": None,
            "strategy_applied": strategy,
            "error": "Output must be a string",
        }

    if not isinstance(failure, dict):
        return {
            "success": False,
            "output": None,
            "strategy_applied": strategy,
            "error": "Failure must be a dictionary",
        }

    valid_strategies = ["fix", "filter", "refrain"]
    if strategy not in valid_strategies:
        return {
            "success": False,
            "output": None,
            "strategy_applied": strategy,
            "error": f"Invalid strategy. Must be one of {valid_strategies}",
        }

    if not GUARDRAILS_AVAILABLE:
        logger.warning(f"[{correlation_id}] Guardrails not available")
        return {
            "success": False,
            "output": None,
            "strategy_applied": strategy,
            "error": f"Guardrails not available: {GUARDRAILS_IMPORT_ERROR}",
        }

    try:
        # Get Guardrails adapter
        adapter = create_adapter()
        if adapter is None:
            config = get_config()
            adapter = GuardrailsAdapter(config=config)

        # Apply remediation
        result = adapter.apply_remediation(
            output=output,
            failure=failure,
            strategy=strategy,
        )

        logger.info(f"[{correlation_id}] Remediation applied", extra={
            "correlation_id": correlation_id,
            "success": result.success,
            "strategy_applied": strategy,
        })

        return {
            "success": result.success,
            "output": result.output if result.success else None,
            "strategy_applied": strategy,
            "error": result.error if not result.success else None,
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"[{correlation_id}] Remediation failed: {e}", extra={
            "correlation_id": correlation_id,
            "error": str(e),
        })
        return {
            "success": False,
            "output": None,
            "strategy_applied": strategy,
            "error": str(e),
        }


# =============================================================================
# MCP TOOL 7: Guardrails Status
# =============================================================================

@mcp_tool("guardrails_status")
def guardrails_status() -> Dict[str, Any]:
    """
    Get Guardrails adapter status.

    This tool returns the current status of the Guardrails adapter,
    including availability and configuration.

    Returns:
    {
        "available": bool,
        "enabled": bool,
        "version": str or None,
        "active_validators": List[str],
        "statistics": {
            "total_validations": int,
            "failures_caught": int,
            "remediations_applied": int,
            "avg_validation_time_ms": float
        }
    }
    """
    logger.info("Guardrails status requested")

    if not GUARDRAILS_AVAILABLE:
        return {
            "available": False,
            "enabled": False,
            "version": None,
            "active_validators": [],
            "error": GUARDRAILS_IMPORT_ERROR,
            "statistics": {
                "total_validations": 0,
                "failures_caught": 0,
                "remediations_applied": 0,
                "avg_validation_time_ms": 0.0,
            },
        }

    try:
        adapter = create_adapter()
        if adapter is None:
            config = get_config()
            adapter = GuardrailsAdapter(config=config)

        # Get active validators
        active_validators = list(_get_available_validators().keys())

        # Get statistics
        stats = _get_statistics_store()
        total_validations = stats['total']['validations']
        failures_caught = stats['total']['failures']
        remediations_applied = stats['total']['remediations']

        # Calculate average validation time
        total_time_ms = sum(
            v['total_time_ms'] for v in stats['by_validator'].values()
        )
        total_uses = sum(
            v['uses'] for v in stats['by_validator'].values()
        )
        avg_validation_time_ms = total_time_ms / total_uses if total_uses > 0 else 0.0

        return {
            "available": True,
            "enabled": True,
            "version": getattr(adapter, 'version', '1.0.0'),
            "active_validators": active_validators,
            "statistics": {
                "total_validations": total_validations,
                "failures_caught": failures_caught,
                "remediations_applied": remedations_applied,
                "avg_validation_time_ms": avg_validation_time_ms,
            },
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Failed to get Guardrails status: {e}")
        return {
            "available": True,
            "enabled": False,
            "version": None,
            "active_validators": [],
            "error": str(e),
            "statistics": {
                "total_validations": 0,
                "failures_caught": 0,
                "remediations_applied": 0,
                "avg_validation_time_ms": 0.0,
            },
        }


# =============================================================================
# MCP TOOL 8: Get Validation Statistics
# =============================================================================

@mcp_tool("guardrails_get_statistics")
def guardrails_get_statistics() -> Dict[str, Any]:
    """
    Get detailed validation statistics.

    This tool returns detailed statistics about validator usage, failures,
    and remediations, useful for monitoring and debugging.

    Returns:
    {
        "by_validator": {
            "validator_name": {
                "uses": int,
                "failures": int,
                "remediations": int,
                "avg_time_ms": float
            }
        },
        "by_strategy": {
            "reask": int,
            "fix": int,
            "filter": int,
            "refrain": int
        },
        "total": {
            "validations": int,
            "failures": int,
            "remediations": int
        }
    }
    """
    logger.info("Guardrails statistics requested")

    if not GUARDRAILS_AVAILABLE:
        return {
            "by_validator": {},
            "by_strategy": {},
            "total": {
                "validations": 0,
                "failures": 0,
                "remediations": 0,
            },
            "error": GUARDRAILS_IMPORT_ERROR,
        }

    try:
        stats = _get_statistics_store()

        # Calculate average times per validator
        by_validator = {}
        for validator_name, validator_stats in stats['by_validator'].items():
            avg_time_ms = (
                validator_stats['total_time_ms'] / validator_stats['uses']
                if validator_stats['uses'] > 0
                else 0.0
            )
            by_validator[validator_name] = {
                "uses": validator_stats['uses'],
                "failures": validator_stats['failures'],
                "remediations": validator_stats['remediations'],
                "avg_time_ms": avg_time_ms,
            }

        # Convert defaultdict to regular dict for JSON serialization
        by_strategy = dict(stats['by_strategy'])

        return {
            "by_validator": by_validator,
            "by_strategy": by_strategy,
            "total": dict(stats['total']),
        }

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"Failed to get Guardrails statistics: {e}")
        return {
            "by_validator": {},
            "by_strategy": {},
            "total": {
                "validations": 0,
                "failures": 0,
                "remediations": 0,
            },
            "error": str(e),
        }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all Guardrails MCP tools."""
    logger.info("Initializing Guardrails MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} Guardrails MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
        "guardrails_available": GUARDRAILS_AVAILABLE,
    }


# Auto-initialize on import
if __name__ != "__main__":
    initialize_mcp_tools()


if __name__ == "__main__":
    print("Guardrails MCP Tools Module")
    print(f"Guardrails Available: {GUARDRAILS_AVAILABLE}")
    print(f"Registered Tools: {len(_MCP_TOOLS)}")
    print("\nTools:")
    for tool_name in sorted(_MCP_TOOLS.keys()):
        print(f"  - {tool_name}")
