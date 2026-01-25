"""
ACE + Steer Configuration Manager

Centralized configuration for ACE (Agentic Context Engine) and Steer (Reliability Layer).

This module provides:
- Environment variable-based configuration
- Graceful degradation when components unavailable
- Per-component enable/disable controls
- Configuration validation
- Status monitoring

Configuration Hierarchy (highest priority first):
1. Environment variables (ACE_ENABLED, STEER_ENABLED, etc.)
2. Configuration dict parameters
3. Default values

All functions have comprehensive error handling and never raise exceptions.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import ACE and Steer to check availability
try:
    from ace_mcp_tools import ACE_AVAILABLE
    ACE_IMPORT_SUCCESS = True
except ImportError:
    ACE_AVAILABLE = False
    ACE_IMPORT_SUCCESS = False
except Exception as e:  # TODO: Catch specific exception instead of Exception
    logger.warning(f"⚠️ Error importing ACE: {e}")
    ACE_AVAILABLE = False
    ACE_IMPORT_SUCCESS = False

try:
    from steer_mcp_tools import STEER_AVAILABLE
    STEER_IMPORT_SUCCESS = True
except ImportError:
    STEER_AVAILABLE = False
    STEER_IMPORT_SUCCESS = False
except Exception as e:  # TODO: Catch specific exception instead of Exception
    logger.warning(f"⚠️ Error importing Steer: {e}")
    STEER_AVAILABLE = False
    STEER_IMPORT_SUCCESS = False


# Default configuration values
DEFAULT_CONFIG = {
    # Component control
    'ace_enabled': True,  # Enable ACE learning system
    'steer_enabled': True,  # Enable Steer verification system

    # ACE configuration
    'ace_skillbook_path': './ace_skillbook.json',
    'ace_agent_id': 'openevolve_agent',
    'ace_learning_enabled': True,  # Enable ACE learning from feedback
    'ace_skill_injection_enabled': True,  # Enable ACE skill injection in prompts

    # Steer configuration
    'steer_verifications': ['json', 'slop'],  # Default verifications to run
    'steer_halt_on_failure': False,  # Whether to raise exception on verification failure
    'steer_json_strict': True,  # Strict JSON validation
    'steer_slop_threshold': 3.5,  # Slop detection threshold

    # Unified integration
    'unified_bridge_enabled': True,  # Use AceSteerBridge or handle separately

    # Fallback behavior
    'fallback_on_error': True,  # Use fallback when components fail
    'log_fallbacks': True,  # Log when using fallback behavior
}


def get_config_from_env() -> Dict[str, Any]:
    """
    Load ACE+Steer configuration from environment variables.

    All errors are caught and logged, never raises exceptions.

    Environment Variables:
        ACE_ENABLED: "true" or "false" - Enable ACE system
        STEER_ENABLED: "true" or "false" - Enable Steer system
        ACE_SKILLBOOK_PATH: Path to ACE skillbook file
        ACE_AGENT_ID: Agent ID for ACE learning
        STEER_VERIFICATIONS: Comma-separated list of verifications (e.g., "json,slop,pii")
        STEER_HALT_ON_FAILURE: "true" or "false" - Raise exception on verification failure
        STEER_JSON_STRICT: "true" or "false" - Strict JSON validation
        STEER_SLOP_THRESHOLD: Float threshold for slop detection
        UNIFIED_BRIDGE_ENABLED: "true" or "false" - Use unified AceSteerBridge

    Returns:
        Configuration dict from environment variables
    """
    config = {}

    try:
        # Component control
        if 'ACE_ENABLED' in os.environ:
            try:
                config['ace_enabled'] = os.environ['ACE_ENABLED'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing ACE_ENABLED: {e}")

        if 'STEER_ENABLED' in os.environ:
            try:
                config['steer_enabled'] = os.environ['STEER_ENABLED'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing STEER_ENABLED: {e}")

        # ACE configuration
        if 'ACE_SKILLBOOK_PATH' in os.environ:
            try:
                config['ace_skillbook_path'] = os.environ['ACE_SKILLBOOK_PATH']
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error reading ACE_SKILLBOOK_PATH: {e}")

        if 'ACE_AGENT_ID' in os.environ:
            try:
                config['ace_agent_id'] = os.environ['ACE_AGENT_ID']
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error reading ACE_AGENT_ID: {e}")

        if 'ACE_LEARNING_ENABLED' in os.environ:
            try:
                config['ace_learning_enabled'] = os.environ['ACE_LEARNING_ENABLED'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing ACE_LEARNING_ENABLED: {e}")

        if 'ACE_SKILL_INJECTION_ENABLED' in os.environ:
            try:
                config['ace_skill_injection_enabled'] = os.environ['ACE_SKILL_INJECTION_ENABLED'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing ACE_SKILL_INJECTION_ENABLED: {e}")

        # Steer configuration
        if 'STEER_VERIFICATIONS' in os.environ:
            try:
                verifications_str = os.environ['STEER_VERIFICATIONS']
                config['steer_verifications'] = [v.strip() for v in verifications_str.split(',') if v.strip()]
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing STEER_VERIFICATIONS: {e}")

        if 'STEER_HALT_ON_FAILURE' in os.environ:
            try:
                config['steer_halt_on_failure'] = os.environ['STEER_HALT_ON_FAILURE'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing STEER_HALT_ON_FAILURE: {e}")

        if 'STEER_JSON_STRICT' in os.environ:
            try:
                config['steer_json_strict'] = os.environ['STEER_JSON_STRICT'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing STEER_JSON_STRICT: {e}")

        if 'STEER_SLOP_THRESHOLD' in os.environ:
            try:
                config['steer_slop_threshold'] = float(os.environ['STEER_SLOP_THRESHOLD'])
            except ValueError as e:
                logger.warning(f"Invalid STEER_SLOP_THRESHOLD: {os.environ.get('STEER_SLOP_THRESHOLD')}, using default")
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing STEER_SLOP_THRESHOLD: {e}")

        # Unified integration
        if 'UNIFIED_BRIDGE_ENABLED' in os.environ:
            try:
                config['unified_bridge_enabled'] = os.environ['UNIFIED_BRIDGE_ENABLED'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing UNIFIED_BRIDGE_ENABLED: {e}")

        # Fallback behavior
        if 'FALLBACK_ON_ERROR' in os.environ:
            try:
                config['fallback_on_error'] = os.environ['FALLBACK_ON_ERROR'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing FALLBACK_ON_ERROR: {e}")

        if 'LOG_FALLBACKS' in os.environ:
            try:
                config['log_fallbacks'] = os.environ['LOG_FALLBACKS'].lower() in ('true', '1', 'yes', 'on')
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Error parsing LOG_FALLBACKS: {e}")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"⚠️ Unexpected error reading environment variables: {e}")

    return config


def get_ace_steer_config(
    user_config: Optional[Dict[str, Any]] = None,
    use_env: bool = True
) -> Dict[str, Any]:
    """
    Get complete ACE+Steer configuration with validation.

    Args:
        user_config: Optional user-provided configuration dict or BaseConfiguration (highest priority)
        use_env: Whether to load configuration from environment variables

    Returns:
        Complete configuration dict with defaults, env vars, and user config merged
    """
    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Apply environment variables
    if use_env:
        env_config = get_config_from_env()
        config.update(env_config)

    # Apply user config (highest priority)
    # Handle both dict and BaseConfiguration objects
    if user_config:
        if isinstance(user_config, dict):
            # Plain dict - can update directly
            config.update(user_config)
        elif hasattr(user_config, 'parameters'):
            # BaseConfiguration object - extract from parameters dict
            # Access parameters without triggering __getattr__
            try:
                params = object.__getattribute__(user_config, 'parameters')
                if isinstance(params, dict):
                    config.update(params)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Failed to extract parameters from config object: {e}")
        elif hasattr(user_config, 'keys'):
            # Dict-like object - try to convert to dict
            try:
                user_dict = dict(user_config)
                config.update(user_dict)
            except Exception as e:  # TODO: Catch specific exception instead of Exception
                logger.warning(f"⚠️ Failed to convert config to dict: {e}")

    # Apply availability constraints
    config['ace_available'] = ACE_AVAILABLE and config['ace_enabled']
    config['steer_available'] = STEER_AVAILABLE and config['steer_enabled']
    config['both_available'] = config['ace_available'] and config['steer_available']

    # Validate paths
    if 'ace_skillbook_path' in config:
        path = Path(config['ace_skillbook_path'])
        if not path.exists():
            logger.warning(f"ACE skillbook path does not exist: {config['ace_skillbook_path']}")

    # Log status
    logger.info(f"ACE Status: available={ACE_AVAILABLE}, enabled={config['ace_enabled']}, effective={config['ace_available']}")
    logger.info(f"Steer Status: available={STEER_AVAILABLE}, enabled={config['steer_enabled']}, effective={config['steer_available']}")

    return config


def is_ace_enabled(user_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if ACE is effectively enabled (available + enabled in config).

    Args:
        user_config: Optional user configuration to check

    Returns:
        True if ACE is available and enabled
    """
    config = get_ace_steer_config(user_config)
    return config['ace_available']


def is_steer_enabled(user_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if Steer is effectively enabled (available + enabled in config).

    Args:
        user_config: Optional user configuration to check

    Returns:
        True if Steer is available and enabled
    """
    config = get_ace_steer_config(user_config)
    return config['steer_available']


def is_unified_bridge_enabled(user_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if unified AceSteerBridge should be used.

    Args:
        user_config: Optional user configuration to check

    Returns:
        True if both ACE and Steer are available and unified bridge is enabled
    """
    config = get_ace_steer_config(user_config)
    return config['both_available'] and config['unified_bridge_enabled']


def get_status() -> Dict[str, Any]:
    """
    Get comprehensive status of ACE and Steer integration.

    All errors are caught and logged, never raises exceptions.

    Returns:
        Status dict with availability, configuration, and recommendations
    """
    try:
        config = get_ace_steer_config()
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"⚠️ Failed to get config for status: {e}")
        config = DEFAULT_CONFIG.copy()

    status = {
        'ace': {
            'import_success': ACE_IMPORT_SUCCESS,
            'available': ACE_AVAILABLE,
            'enabled': config.get('ace_enabled', True),
            'effective': config.get('ace_available', False),
            'skillbook_path': config.get('ace_skillbook_path', ''),
        },
        'steer': {
            'import_success': STEER_IMPORT_SUCCESS,
            'available': STEER_AVAILABLE,
            'enabled': config.get('steer_enabled', True),
            'effective': config.get('steer_available', False),
            'verifications': config.get('steer_verifications', []),
        },
        'unified_bridge': {
            'available': config.get('both_available', False),
            'enabled': config.get('unified_bridge_enabled', True),
            'effective': False,
        },
        'recommendations': []
    }

    try:
        status['unified_bridge']['effective'] = is_unified_bridge_enabled()
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.warning(f"⚠️ Error checking unified bridge status: {e}")

    # Add recommendations
    try:
        if not ACE_AVAILABLE and config.get('ace_enabled', True):
            status['recommendations'].append(
                "ACE is enabled but not available. Install ACE or set ACE_ENABLED=false"
            )

        if not STEER_AVAILABLE and config.get('steer_enabled', True):
            status['recommendations'].append(
                "Steer is enabled but not available. Install Steer or set STEER_ENABLED=false"
            )

        if not config.get('both_available', False) and config.get('unified_bridge_enabled', True):
            status['recommendations'].append(
                "Unified bridge enabled but both ACE and Steer must be available"
            )
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.warning(f"⚠️ Error generating recommendations: {e}")

    return status


def validate_config(config: Dict[str, Any]) -> Tuple[bool, list[str]]:
    """
    Validate ACE+Steer configuration.

    All errors are caught, never raises exceptions.

    Args:
        config: Configuration dict to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        if not isinstance(config, dict):
            return False, [f"config must be dict, got {type(config)}"]

        # Validate ace_enabled
        if 'ace_enabled' in config:
            if not isinstance(config['ace_enabled'], bool):
                errors.append(f"ace_enabled must be bool, got {type(config['ace_enabled'])}")

        # Validate steer_enabled
        if 'steer_enabled' in config:
            if not isinstance(config['steer_enabled'], bool):
                errors.append(f"steer_enabled must be bool, got {type(config['steer_enabled'])}")

        # Validate ace_skillbook_path
        if 'ace_skillbook_path' in config:
            if not isinstance(config['ace_skillbook_path'], str):
                errors.append(f"ace_skillbook_path must be str, got {type(config['ace_skillbook_path'])}")

        # Validate steer_verifications
        if 'steer_verifications' in config:
            if not isinstance(config['steer_verifications'], list):
                errors.append(f"steer_verifications must be list, got {type(config['steer_verifications'])}")
            else:
                valid_verifications = ['json', 'slop', 'pii', 'citations', 'sql']
                for v in config['steer_verifications']:
                    if v not in valid_verifications:
                        errors.append(f"Invalid verification: {v}. Valid options: {valid_verifications}")

        # Validate steer_slop_threshold
        if 'steer_slop_threshold' in config:
            if not isinstance(config['steer_slop_threshold'], (int, float)):
                errors.append(f"steer_slop_threshold must be numeric, got {type(config['steer_slop_threshold'])}")
            elif not (0.0 <= config['steer_slop_threshold'] <= 10.0):
                errors.append(f"steer_slop_threshold must be between 0.0 and 10.0, got {config['steer_slop_threshold']}")
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        logger.error(f"⚠️ Unexpected error validating config: {e}")
        errors.append(f"Validation error: {e}")

    return len(errors) == 0, errors


# Convenience functions for backward compatibility
def is_available() -> bool:
    """Check if either ACE or Steer is available (backward compatibility)"""
    return ACE_AVAILABLE or STEER_AVAILABLE


def get_availability() -> Dict[str, bool]:
    """Get availability status of both components (backward compatibility)"""
    return {
        'ace': ACE_AVAILABLE,
        'steer': STEER_AVAILABLE,
        'bridge': ACE_AVAILABLE and STEER_AVAILABLE,
    }


__all__ = [
    'get_ace_steer_config',
    'is_ace_enabled',
    'is_steer_enabled',
    'is_unified_bridge_enabled',
    'get_status',
    'validate_config',
    'is_available',
    'get_availability',
    'ACE_AVAILABLE',
    'STEER_AVAILABLE',
    'DEFAULT_CONFIG',
]
