"""
ROMA Reliability SSOT (Single Source of Truth)
Provides reliable validation and configuration for ROMA components
"""

from typing import Dict, Any, Optional, List


# Default ROMA validation configuration
DEFAULT_VALIDATION_CONFIG = {
    "strict_mode": False,
    "max_retry_attempts": 3,
    "timeout_seconds": 30,
    "validate_outputs": True,
    "log_validations": True
}


def get_validation_config(config_id: str = "default") -> Dict[str, Any]:
    """
    Get validation configuration for ROMA components.
    
    Args:
        config_id: Configuration identifier (default: "default")
    
    Returns:
        Configuration dictionary
    """
    configs = {
        "default": DEFAULT_VALIDATION_CONFIG.copy(),
        "strict": {
            **DEFAULT_VALIDATION_CONFIG,
            "strict_mode": True,
            "max_retry_attempts": 5
        },
        "lenient": {
            **DEFAULT_VALIDATION_CONFIG,
            "strict_mode": False,
            "max_retry_attempts": 1
        }
    }
    
    return configs.get(config_id, DEFAULT_VALIDATION_CONFIG.copy())


def validate_roma_output(output: Any, config: Dict[str, Any] = None) -> bool:
    """
    Validate ROMA component output.
    
    Args:
        output: Output to validate
        config: Validation configuration
    
    Returns:
        True if valid, False otherwise
    """
    if config is None:
        config = get_validation_config()
    
    if not config["validate_outputs"]:
        return True
    
    # Basic validation
    if output is None:
        return False
    
    return True
