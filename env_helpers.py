"""
Environment Variable Helper Functions

Provides secure, validated environment variable handling with type conversion,
format validation, and graceful error handling.
"""

import os
import re
import logging
from typing import Optional, Union, List, Any
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when environment variable validation fails."""
    pass


def check_required_env_vars(required_vars: List[str]) -> None:
    """
    Check that all required environment variables are set.

    Args:
        required_vars: List of required environment variable names

    Raises:
        ValidationError: If any required environment variables are missing
    """
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValidationError(
            f"Missing required environment variables:\n"
            f"{', '.join(missing)}\n\n"
            f"Please set these environment variables before running the application.\n"
            f"See .env.example for a list of all required variables."
        )


def env_var_str(
    name: str,
    default: Optional[str] = None,
    required: bool = False,
    allow_empty: bool = False,
    pattern: Optional[str] = None,
) -> Optional[str]:
    """
    Get string environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        allow_empty: Whether empty string is allowed
        pattern: Optional regex pattern to validate against

    Returns:
        Environment variable value or default

    Raises:
        ValidationError: If validation fails
    """
    value = os.getenv(name, default)

    if value is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return None

    if not allow_empty and value == "":
        raise ValidationError(f"Environment variable '{name}' cannot be empty")

    if pattern and not re.match(pattern, value):
        raise ValidationError(
            f"Environment variable '{name}' does not match required pattern: {pattern}"
        )

    return value


def env_var_int(
    name: str,
    default: Optional[int] = None,
    required: bool = False,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
) -> Optional[int]:
    """
    Get integer environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Environment variable value as integer or default

    Raises:
        ValidationError: If validation fails
    """
    value_str = os.getenv(name)

    if value_str is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return default

    try:
        value = int(value_str)
    except ValueError:
        raise ValidationError(
            f"Environment variable '{name}' must be an integer, got: {value_str}"
        )

    if min_val is not None and value < min_val:
        raise ValidationError(
            f"Environment variable '{name}' must be >= {min_val}, got: {value}"
        )

    if max_val is not None and value > max_val:
        raise ValidationError(
            f"Environment variable '{name}' must be <= {max_val}, got: {value}"
        )

    return value


def env_var_float(
    name: str,
    default: Optional[float] = None,
    required: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Optional[float]:
    """
    Get float environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Environment variable value as float or default

    Raises:
        ValidationError: If validation fails
    """
    value_str = os.getenv(name)

    if value_str is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return default

    try:
        value = float(value_str)
    except ValueError:
        raise ValidationError(
            f"Environment variable '{name}' must be a float, got: {value_str}"
        )

    if min_val is not None and value < min_val:
        raise ValidationError(
            f"Environment variable '{name}' must be >= {min_val}, got: {value}"
        )

    if max_val is not None and value > max_val:
        raise ValidationError(
            f"Environment variable '{name}' must be <= {max_val}, got: {value}"
        )

    return value


def env_var_bool(
    name: str,
    default: Optional[bool] = None,
    required: bool = False,
) -> Optional[bool]:
    """
    Get boolean environment variable with validation.

    Accepts: true, false, 1, 0, yes, no (case-insensitive)

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required

    Returns:
        Environment variable value as boolean or default

    Raises:
        ValidationError: If validation fails
    """
    value_str = os.getenv(name)

    if value_str is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return default

    value_str_lower = value_str.lower().strip()

    if value_str_lower in ("true", "1", "yes", "on"):
        return True
    elif value_str_lower in ("false", "0", "no", "off"):
        return False
    else:
        raise ValidationError(
            f"Environment variable '{name}' must be a boolean value "
            f"(true/false, 1/0, yes/no), got: {value_str}"
        )


def env_var_list(
    name: str,
    default: Optional[List[str]] = None,
    required: bool = False,
    separator: str = ",",
    strip_whitespace: bool = True,
) -> Optional[List[str]]:
    """
    Get list environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        separator: Separator to split on (default: comma)
        strip_whitespace: Whether to strip whitespace from items

    Returns:
        Environment variable value as list of strings or default

    Raises:
        ValidationError: If validation fails
    """
    value_str = os.getenv(name)

    if value_str is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return default

    items = value_str.split(separator)

    if strip_whitespace:
        items = [item.strip() for item in items]

    # Remove empty strings
    items = [item for item in items if item]

    return items


def env_var_path(
    name: str,
    default: Optional[Union[str, Path]] = None,
    required: bool = False,
    must_exist: bool = False,
    create_if_missing: bool = False,
) -> Optional[Path]:
    """
    Get path environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        must_exist: Whether the path must exist
        create_if_missing: Whether to create the path if it doesn't exist

    Returns:
        Environment variable value as Path object or default

    Raises:
        ValidationError: If validation fails
    """
    value_str = os.getenv(name)

    if value_str is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return Path(default) if default else None

    path = Path(value_str).expanduser().resolve()

    if must_exist and not path.exists():
        raise ValidationError(
            f"Environment variable '{name}' points to non-existent path: {path}"
        )

    if create_if_missing and not path.exists():
        logger.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)

    return path


def env_var_url(
    name: str,
    default: Optional[str] = None,
    required: bool = False,
    allowed_schemes: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Get URL environment variable with validation.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        allowed_schemes: List of allowed URL schemes (e.g., ['http', 'https'])

    Returns:
        Environment variable value as URL string or default

    Raises:
        ValidationError: If validation fails
    """
    value = os.getenv(name, default)

    if value is None:
        if required:
            raise ValidationError(f"Required environment variable '{name}' is not set")
        return None

    try:
        parsed = urlparse(value)
    except Exception as e:
        raise ValidationError(
            f"Environment variable '{name}' is not a valid URL: {value}"
        )

    if not parsed.scheme or not parsed.netloc:
        raise ValidationError(
            f"Environment variable '{name}' is not a valid URL: {value}"
        )

    if allowed_schemes and parsed.scheme not in allowed_schemes:
        raise ValidationError(
            f"Environment variable '{name}' must use one of these schemes: "
            f"{', '.join(allowed_schemes)}, got: {parsed.scheme}"
        )

    return value


def env_var_api_key(
    name: str,
    default: Optional[str] = None,
    required: bool = False,
    provider: Optional[str] = None,
) -> Optional[str]:
    """
    Get API key environment variable with validation.

    Checks for placeholder values and warns about insecure defaults.

    Args:
        name: Environment variable name
        default: Default value if not set
        required: Whether this variable is required
        provider: Provider name for better error messages (e.g., "OpenAI", "Anthropic")

    Returns:
        Environment variable value or default

    Raises:
        ValidationError: If validation fails
    """
    value = os.getenv(name, default)

    if value is None:
        if required:
            provider_msg = f" for {provider}" if provider else ""
            raise ValidationError(
                f"Required environment variable '{name}'{provider_msg} is not set. "
                f"Please set your API key in the environment variables."
            )
        return None

    # Check for placeholder/insecure values
    insecure_patterns = [
        "your-api-key",
        "your-api-key-here",
        "your-production-api-key",
        "sk-1234567890",
        "demo_key",
        "placeholder",
        "change-me",
        "secret-key",
        "your-secret-key",
    ]

    value_lower = value.lower()
    for pattern in insecure_patterns:
        if pattern in value_lower:
            provider_msg = f" for {provider}" if provider else ""
            logger.warning(
                f"SECURITY WARNING: Environment variable '{name}'{provider_msg} "
                f"appears to contain a placeholder/insecure value. "
                f"This should not be used in production!"
            )
            break

    # Check for specific API key formats
    if provider:
        if provider.lower() == "openai" and not value.startswith("sk-"):
            logger.warning(
                f"Environment variable '{name}' for OpenAI doesn't match expected format"
            )
        elif provider.lower() == "anthropic" and not value.startswith("sk-ant-"):
            logger.warning(
                f"Environment variable '{name}' for Anthropic doesn't match expected format"
            )

    return value


def is_production() -> bool:
    """Check if running in production environment."""
    return env_var_bool("PRODUCTION", default=False) or env_var_bool("ENV", default="") == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return env_var_bool("DEVELOPMENT", default=True) or env_var_bool("ENV", default="") == "development"


def get_env() -> str:
    """Get current environment (development, staging, production)."""
    return env_var_str("ENV", default="development").lower()


def validate_all_api_keys() -> None:
    """
    Validate all commonly used API keys are properly set.

    Logs warnings for missing keys but doesn't raise exceptions
    unless in production mode.
    """
    api_key_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "CREWAI_API_KEY",
        "BUBBLELABS_API_KEY",
    ]

    missing_keys = []
    for var in api_key_vars:
        if not os.getenv(var):
            missing_keys.append(var)

    if missing_keys:
        msg = f"The following API keys are not set: {', '.join(missing_keys)}"
        if is_production():
            raise ValidationError(msg)
        else:
            logger.warning(f"{msg} - Some features may not work properly.")


def generate_secure_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure random key.

    Args:
        length: Length of key in bytes

    Returns:
        Hex-encoded secure key
    """
    import secrets

    return secrets.token_hex(length)


def get_or_generate_secret_key(
    env_var_name: str = "SECRET_KEY",
    min_length: int = 32,
) -> str:
    """
    Get secret key from environment or generate a secure one for development.

    Args:
        env_var_name: Environment variable name
        min_length: Minimum required length

    Returns:
        Secret key string

    Raises:
        ValidationError: In production if key is missing or too short
    """
    key = os.getenv(env_var_name)

    if not key:
        if is_production():
            raise ValidationError(
                f"Environment variable '{env_var_name}' must be set in production"
            )
        else:
            logger.warning(
                f"Environment variable '{env_var_name}' not set. "
                f"Generating a temporary key for development. "
                f"This key will change on restart!"
            )
            key = generate_secure_key(min_length)
    elif len(key) < min_length:
        if is_production():
            raise ValidationError(
                f"Environment variable '{env_var_name}' must be at least "
                f"{min_length} characters long"
            )
        else:
            logger.warning(
                f"Environment variable '{env_var_name}' is shorter than "
                f"recommended ({len(key)} < {min_length})"
            )

    return key
