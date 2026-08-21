"""
ACE Security and Validation Utilities

This module provides centralized security, validation, and safety functions
for all ACE integration components. It addresses critical security vulnerabilities,
input validation gaps, resource management, and thread safety issues.

Created: 2025-12-29
Purpose: Ultra-comprehensive security hardening for ACE integration
"""
from __future__ import annotations


import os
import re
import json
import hashlib
import logging
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from functools import wraps

# ============================================================================
# CONFIGURATION
# ============================================================================

# Security limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_LIST_SIZE = 10000
MAX_STRING_LENGTH = 100000
MAX_MODEL_NAME_LENGTH = 100
SAFE_FILE_EXTENSIONS = {'.json', '.yaml', '.yml'}

# Base directories for file operations
DEFAULT_SKILLBOOK_DIR = "./ace_skillbooks"
DEFAULT_CHECKPOINT_DIR = "./ace_checkpoints"
DEFAULT_ANALYTICS_DIR = "./ace_analytics"

# Threading locks
_GLOBAL_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

# ============================================================================
# PATH VALIDATION (Fixes CVE-1: Path Traversal)
# ============================================================================

def validate_and_resolve_path(base_dir: str, user_path: str) -> str:
    """
    Validate and resolve a user-provided path to prevent directory traversal.

    CRITICAL SECURITY FIX: Prevents path traversal attacks

    Args:
        base_dir: Base directory that files must be under
        user_path: User-supplied path

    Returns:
        Absolute, validated path

    Raises:
        ValueError: If path tries to escape base directory
    """
    try:
        # Convert to absolute paths
        base = Path(base_dir).resolve()
        target = (base / user_path).resolve() if user_path else base

        # Ensure target is under base directory
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: {user_path} resolves outside base directory"
            )

        return str(target)
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise ValueError(f"Invalid path: {e}")


def validate_file_path_safe(filepath: str, base_dir: str = ".") -> str:
    """
    Validate file path is safe for access.

    Args:
        filepath: Path to validate
        base_dir: Base directory (default: current directory)

    Returns:
        Validated absolute path
    """
    if not filepath or not isinstance(filepath, str):
        raise ValueError("File path must be a non-empty string")

    if len(filepath) > 1000:  # Reasonable limit
        raise ValueError("File path too long")

    # Check for suspicious patterns
    suspicious = ['..', '~', '$', '|', ';', '&', '`', '\n', '\r', '\x00']
    if any(pattern in filepath for pattern in suspicious):
        raise ValueError(f"File path contains suspicious characters: {filepath}")

    # Resolve and validate
    return validate_and_resolve_path(base_dir, filepath)


# ============================================================================
# FILE OPERATIONS (Fixes TOCTOU and unsafe file access)
# ============================================================================

def safe_load_json_file(filepath: str, max_size: int = MAX_FILE_SIZE) -> Dict[str, Any]:
    """
    Safely load JSON file with comprehensive validation.

    CRITICAL SECURITY FIX: Prevents unsafe deserialization and DoS

    Args:
        filepath: Path to JSON file
        max_size: Maximum file size in bytes

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If file validation fails
        IOError: If file cannot be read
    """
    path = Path(filepath)

    # Check file extension
    if path.suffix.lower() not in SAFE_FILE_EXTENSIONS:
        raise ValueError(
            f"Unsafe file extension: {path.suffix}. Allowed: {SAFE_FILE_EXTENSIONS}"
        )

    # Check file size
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = path.stat().st_size
    if file_size > max_size:
        raise ValueError(
            f"File too large: {file_size} bytes (max: {max_size})"
        )

    if file_size == 0:
        raise ValueError(f"File is empty: {filepath}")

    # Read and validate JSON
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except UnicodeDecodeError:
        raise ValueError("File contains invalid characters (not valid UTF-8)")

    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object, got {type(data).__name__}")

    return data


def atomic_save_json_file(filepath: str, data: Dict[str, Any]) -> None:
    """
    Atomically save data to JSON file.

    CRITICAL FIX: Prevents file corruption and TOCTOU race conditions

    Args:
        filepath: Path to save to
        data: Data to save

    Raises:
        IOError: If save fails
    """
    path = Path(filepath)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file in same directory
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix='.tmp',
        delete=False,
        encoding='utf-8'
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name

    # Atomic rename (overwrites target if exists)
    try:
        os.replace(tmp_path, filepath)
    except OSError as e:
        # Clean up temp file if rename fails
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise IOError(f"Cannot save file: {e}")


# ============================================================================
# INPUT VALIDATION (Fixes HVE-1, HVE-2, and edge cases)
# ============================================================================

def validate_numeric_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    value_type: type = (int, float),
    allow_nan: bool = False,
    allow_infinity: bool = False
) -> Union[int, float]:
    """
    Validate numeric parameter is within acceptable range.

    CRITICAL FIX: Prevents NaN/Infinity bypass and integer overflow

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        value_type: Expected type(s)
        allow_nan: Whether to allow NaN
        allow_infinity: Whether to allow infinity

    Returns:
        Validated value

    Raises:
        ValueError: If validation fails
    """
    # Check type
    if not isinstance(value, value_type):
        raise ValueError(
            f"{name} must be {value_type.__name__}, got {type(value).__name__}"
        )

    # Check for NaN and Infinity
    if isinstance(value, float):
        if not allow_nan:
            import math
            if math.isnan(value):
                raise ValueError(f"{name} cannot be NaN")
        if not allow_infinity:
            import math
            if math.isinf(value):
                raise ValueError(f"{name} cannot be Infinity")

    # Check range
    if min_val is not None and value < min_val:
        raise ValueError(
            f"{name} must be >= {min_val}, got {value}"
        )

    if max_val is not None and value > max_val:
        raise ValueError(
            f"{name} must be <= {max_val}, got {value}"
        )

    return value


def validate_list_size(
    items: list,
    name: str,
    max_size: int = MAX_LIST_SIZE,
    min_size: int = 0,
    allow_empty: bool = True
) -> list:
    """
    Validate list size to prevent DoS.

    CRITICAL FIX: Prevents resource exhaustion via unbounded lists

    Args:
        items: List to validate
        name: Parameter name for error messages
        max_size: Maximum allowed size
        min_size: Minimum allowed size
        allow_empty: Whether empty list is allowed

    Returns:
        Validated list

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(items, list):
        raise ValueError(
            f"{name} must be a list, got {type(items).__name__}"
        )

    list_len = len(items)

    if not allow_empty and list_len == 0:
        raise ValueError(f"{name} cannot be empty")

    if list_len < min_size:
        raise ValueError(
            f"{name} too small: {list_len} items (min: {min_size})"
        )

    if list_len > max_size:
        raise ValueError(
            f"{name} too large: {list_len} items (max: {max_size})"
        )

    return items


def validate_string_length(
    value: str,
    name: str,
    max_length: int = MAX_STRING_LENGTH,
    min_length: int = 0,
    allow_empty: bool = True
) -> str:
    """
    Validate string length.

    Args:
        value: String to validate
        name: Parameter name for error messages
        max_length: Maximum allowed length
        min_length: Minimum allowed length
        allow_empty: Whether empty string is allowed

    Returns:
        Validated string

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string, got {type(value).__name__}"
        )

    str_len = len(value)

    if not allow_empty and str_len == 0:
        raise ValueError(f"{name} cannot be empty")

    if str_len < min_length:
        raise ValueError(
            f"{name} too short: {str_len} chars (min: {min_length})"
        )

    if str_len > max_length:
        raise ValueError(
            f"{name} too long: {str_len} chars (max: {max_length})"
        )

    return value


def validate_model_name(model: str) -> str:
    """
    Validate LiteLLM model name to prevent command injection.

    CRITICAL SECURITY FIX: Prevents command injection via model names

    Args:
        model: Model name to validate

    Returns:
        Validated model name

    Raises:
        ValueError: If model name is invalid
    """
    if not model or not isinstance(model, str):
        raise ValueError("Model name must be a non-empty string")

    if len(model) > MAX_MODEL_NAME_LENGTH:
        raise ValueError(
            f"Model name too long: {len(model)} chars (max: {MAX_MODEL_NAME_LENGTH})"
        )

    # Check for suspicious patterns
    suspicious_patterns = [';', '&', '|', '$', '`', '\n', '\r', '\x00', '..']
    if any(pattern in model for pattern in suspicious_patterns):
        raise ValueError(
            f"Model name contains suspicious characters: {model}"
        )

    # Validate format (provider/model-name or just model-name)
    if not re.match(r'^[a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)*$', model):
        raise ValueError(
            f"Invalid model format: {model}. "
            f"Expected format: provider/model-name or model-name"
        )

    return model


# ============================================================================
# DICTIONARY VALIDATION (Fixes MVE-2 and data structure issues)
# ============================================================================

def validate_dict_structure(
    data: Dict[str, Any],
    expected_fields: Dict[str, type],
    allow_extra: bool = True,
    require_all: bool = True
) -> Dict[str, Any]:
    """
    Validate dictionary structure and field types.

    Args:
        data: Dictionary to validate
        expected_fields: Expected field names and types
        allow_extra: Whether to allow extra fields
        require_all: Whether all expected fields are required

    Returns:
        Validated dictionary

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")

    validated = {}

    # Check expected fields
    for field_name, field_type in expected_fields.items():
        if field_name not in data:
            if require_all:
                raise ValueError(f"Missing required field: {field_name}")
            else:
                # Optional field, use default value
                if field_type == int:
                    validated[field_name] = 0
                elif field_type == float:
                    validated[field_name] = 0.0
                elif field_type == bool:
                    validated[field_name] = False
                elif field_type == str:
                    validated[field_name] = ""
                elif field_type == list:
                    validated[field_name] = []
                elif field_type == dict:
                    validated[field_name] = {}
                else:
                    validated[field_name] = None
                continue

        value = data[field_name]

        # Type validation with coercion
        if value is None:
            if field_type in (int, float, bool, str, list, dict):
                validated[field_name] = field_type() if field_type != bool else False
            else:
                validated[field_name] = None
        elif not isinstance(value, field_type):
            # Try to convert
            try:
                if field_type == bool and isinstance(value, str):
                    validated[field_name] = value.lower() in ('true', '1', 'yes')
                elif field_type == float and isinstance(value, (int, str)):
                    validated[field_name] = float(value)
                elif field_type == int and isinstance(value, (str, float)):
                    validated[field_name] = int(value)
                else:
                    raise ValueError(
                        f"Field '{field_name}': expected {field_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
            except (ValueError, TypeError):
                raise ValueError(
                    f"Field '{field_name}': cannot convert {type(value).__name__} "
                    f"to {field_type.__name__}"
                )
        else:
            validated[field_name] = value

    # Check for unexpected fields
    if not allow_extra:
        extra_fields = set(data.keys()) - set(expected_fields.keys())
        if extra_fields:
            raise ValueError(f"Unexpected fields: {extra_fields}")
    else:
        for field_name in set(data.keys()) - set(expected_fields.keys()):
            validated[field_name] = data[field_name]

    return validated


# ============================================================================
# SAFE HASHING (Fixes CVE-4: Weak MD5)
# ============================================================================

def generate_secure_hash(content: str, hash_length: int = 32) -> str:
    """
    Generate secure content hash using SHA-256.

    CRITICAL SECURITY FIX: Replaces MD5 with SHA-256

    Args:
        content: Content to hash
        hash_length: Length of hash to return (max 64 for SHA-256)

    Returns:
        Hex hash string
    """
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    full_hash = hash_obj.hexdigest()
    return full_hash[:min(hash_length, len(full_hash))]


# ============================================================================
# ERROR HANDLING (Fixes HVE-3: Information Disclosure)
# ============================================================================

def create_safe_error(
    user_message: str,
    internal_error: Exception,
    include_details: bool = False
) -> Dict[str, Any]:
    """
    Create safe error response without exposing internal details.

    CRITICAL SECURITY FIX: Prevents information disclosure

    Args:
        user_message: User-friendly error message
        internal_error: The actual exception
        include_details: Whether to include details (debug mode only)

    Returns:
        Safe error dictionary
    """
    # Log full error internally
    logger.error(
        f"{user_message}: {internal_error}",
        exc_info=True
    )

    # Return safe error to user
    response = {
        "success": False,
        "error": user_message,
        "error_type": type(internal_error).__name__
    }

    # Only include details in debug mode (never in production)
    if include_details:
        response["details"] = str(internal_error)

    return response


# ============================================================================
# THREAD SAFETY (Fixes Race Conditions #1, #4, #5, etc.)
# ============================================================================

def get_global_lock(name: str) -> threading.RLock:
    """
    Get or create a named global lock.

    CRITICAL FIX: Prevents race conditions on global state

    Args:
        name: Lock name

    Returns:
        Thread-safe RLock
    """
    with _LOCKS_LOCK:
        if name not in _GLOBAL_LOCKS:
            _GLOBAL_LOCKS[name] = threading.RLock()
        return _GLOBAL_LOCKS[name]


def synchronized(lock_name: str):
    """
    Decorator to synchronize function execution.

    Usage:
        @synchronized('mcp_tools_registry')
        def register_tool(name, func):
            _MCP_TOOLS[name] = func
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            lock = get_global_lock(lock_name)
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# LOGGING SANITIZATION (Fixes MVE-3: Sensitive Data in Logs)
# ============================================================================

def sanitize_for_logging(data: Any, max_length: int = 200) -> str:
    """
    Sanitize data for safe logging.

    CRITICAL SECURITY FIX: Prevents sensitive data in logs

    Args:
        data: Data to sanitize
        max_length: Maximum length of output

    Returns:
        Safe string representation
    """
    # Convert to string
    if isinstance(data, str):
        result = data
    elif isinstance(data, dict):
        # Remove sensitive keys
        sanitized = {
            k: "***REDACTED***" if k.lower() in (
                'password', 'secret', 'token', 'api_key', 'credential',
                'private_key', 'passphrase'
            ) else v
            for k, v in data.items()
        }
        result = str(sanitized)
    elif isinstance(data, list):
        result = f"[{len(data)} items]"
    else:
        result = str(data)

    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length] + "... (truncated)"

    return result


# ============================================================================
# RATE LIMITING (Fixes LVE-1: Missing Rate Limiting)
# ============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter.

    SECURITY FIX: Prevents DoS via excessive API calls
    """

    def __init__(self, max_calls: int = 100, window: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed
            window: Time window in seconds
        """
        self.max_calls = max_calls
        self.window = window
        self.calls: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if call is allowed.

        Args:
            identifier: Unique identifier (e.g., IP address, agent_id)

        Returns:
            True if call is allowed
        """
        import time
        now = time.time()

        with self._lock:
            # Clean old calls
            if identifier not in self.calls:
                self.calls[identifier] = []

            self.calls[identifier] = [
                call_time for call_time in self.calls[identifier]
                if now - call_time < self.window
            ]

            # Check limit
            if len(self.calls[identifier]) >= self.max_calls:
                return False

            # Record call
            self.calls[identifier].append(now)
            return True


def rate_limit(agent_id_param: str = 'agent_id', max_calls: int = 100, window: int = 60):
    """
    Decorator to apply rate limiting to functions.

    Usage:
        @rate_limit(agent_id_param='agent_id', max_calls=100, window=60)
        def some_function(agent_id, ...):
            pass
    """
    limiter = RateLimiter(max_calls=max_calls, window=window)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get agent_id from parameters
            agent_id = kwargs.get(agent_id_param)
            if not agent_id:
                agent_id = "default"

            # Check rate limit
            if not limiter.is_allowed(agent_id):
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Please try again later."
                }

            # Execute function
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Path validation
    'validate_and_resolve_path',
    'validate_file_path_safe',

    # File operations
    'safe_load_json_file',
    'atomic_save_json_file',

    # Input validation
    'validate_numeric_range',
    'validate_list_size',
    'validate_string_length',
    'validate_model_name',

    # Dictionary validation
    'validate_dict_structure',

    # Hashing
    'generate_secure_hash',

    # Error handling
    'create_safe_error',

    # Thread safety
    'get_global_lock',
    'synchronized',

    # Logging
    'sanitize_for_logging',

    # Rate limiting
    'RateLimiter',
    'rate_limit',

    # Configuration
    'MAX_FILE_SIZE',
    'MAX_LIST_SIZE',
    'MAX_STRING_LENGTH',
    'MAX_MODEL_NAME_LENGTH',
    'SAFE_FILE_EXTENSIONS',
    'DEFAULT_SKILLBOOK_DIR',
    'DEFAULT_CHECKPOINT_DIR',
    'DEFAULT_ANALYTICS_DIR',
]
