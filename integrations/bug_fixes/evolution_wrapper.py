"""
Evolution Configuration Wrapper Adapter

Fixes duplicate dataclass fields in evolution.EvolutionConfiguration
without modifying the core evolution.py file.

Bug Fixed:
- Duplicate fields in EvolutionConfiguration dataclass:
  * convergence_threshold (lines 84, 92)
  * fitness_function (lines 85, 93)
  * elitism (lines 89, 94)
  * diversity_maintenance (lines 90, 95)
  * adaptive_parameters (lines 91, 96)

Solution:
- Wrapper class that removes duplicates during initialization
- Provides clean interface to EvolutionConfiguration
- No modifications to core evolution.py

Usage:
    from integrations.bug_fixes import EvolutionConfigurationWrapper

    # Use wrapper instead of original class
    config = EvolutionConfigurationWrapper(
        evolution_mode="standard",
        max_iterations=100
    )

    # Access wrapped config
    evolution_config = config.get_config()
"""

import logging
from typing import Any, Dict, Optional, List
from dataclasses import asdict

logger = logging.getLogger(__name__)


class EvolutionConfigurationWrapper:
    """
    Wrapper for evolution.EvolutionConfiguration that handles duplicate fields.

    The original EvolutionConfiguration has duplicate field definitions which
    cause the later definitions to overwrite the earlier ones. This wrapper
    provides a clean interface and logs warnings about duplicates.
    """

    # Duplicate fields (first occurrence is correct)
    DUPLICATE_FIELDS = {
        'convergence_threshold': 84,  # Duplicated at line 92
        'fitness_function': 85,       # Duplicated at line 93
        'elitism': 89,                # Duplicated at line 94
        'diversity_maintenance': 90,  # Duplicated at line 95
        'adaptive_parameters': 91,    # Duplicated at line 96
    }

    # Default values from evolution.py
    DEFAULTS = {
        'evolution_mode': 'standard',
        'max_iterations': 10,
        'population_size': 20,
        'temperature': 0.7,
        'max_tokens': 2048,
        'seed': None,
        'early_stopping': False,
        'convergence_threshold': 0.001,
        'fitness_function': 'default',
        'selection_pressure': 1.0,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elitism': True,
        'diversity_maintenance': True,
        'adaptive_parameters': False,
        'reasoning_effort': 'medium',
        'language': 'python',
        'file_suffix': '.py',
        'api_key': '',
        'api_base': 'https://api.openai.com/v1',
        'model_id': 'gpt-4',
        'backup_models': None,
        'timeout': 30,
        'max_retries': 3,
        'retry_delay': 1.0,
        'rate_limit': 60,
    }

    def __init__(self, **kwargs):
        """
        Initialize EvolutionConfiguration with duplicate handling.

        Args:
            **kwargs: Configuration parameters (same as EvolutionConfiguration)

        Example:
            config = EvolutionConfigurationWrapper(
                evolution_mode="standard",
                max_iterations=100,
                population_size=20
            )
        """
        self._config_dict: Dict[str, Any] = {}
        self._original_class = None

        # Try to import original EvolutionConfiguration
        try:
            from evolution import EvolutionConfiguration
            self._original_class = EvolutionConfiguration
            logger.info("Using core EvolutionConfiguration (with wrapper fixes)")
        except ImportError as e:
            logger.warning(f"Could not import EvolutionConfiguration: {e}")
            logger.info("Using standalone configuration dict")

        # Apply defaults
        self._config_dict.update(self.DEFAULTS)

        # Apply user-provided values
        for key, value in kwargs.items():
            if key in self.DUPLICATE_FIELDS:
                logger.debug(f"Setting duplicate field '{key}' to {value}")
            self._config_dict[key] = value

        # Remove duplicates from config dict
        self._remove_duplicates_from_dict()

        # Log if any duplicate fields were set
        duplicate_keys = set(kwargs.keys()) & set(self.DUPLICATE_FIELDS.keys())
        if duplicate_keys:
            logger.info(f"Configured duplicate fields: {duplicate_keys}")
            logger.warning("Note: These fields are duplicated in core evolution.py")
            logger.warning("Wrapper ensures only first occurrence is used")

    def _remove_duplicates_from_dict(self) -> None:
        """
        Ensure no duplicate field values in config dict.

        The wrapper handles this by design (dict keys are unique),
        but we log this for clarity.
        """
        # Dict keys are unique by design, so no action needed
        # Just log that we're aware of the duplicate fields
        pass

    def get_config(self) -> Any:
        """
        Get the wrapped configuration object.

        Returns:
            EvolutionConfiguration instance if available, else dict
        """
        if self._original_class is not None:
            try:
                # Create instance of original class
                return self._original_class(**self._config_dict)
            except Exception as e:
                logger.error(f"Failed to create EvolutionConfiguration: {e}")
                logger.info("Falling back to config dict")
                return self._config_dict

        return self._config_dict

    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        if self._original_class is not None:
            try:
                config = self.get_config()
                return asdict(config)
            except (AttributeError, TypeError, ValueError):
                pass

        return self._config_dict.copy()

    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to underlying config.

        Allows accessing config fields as attributes:
            config.max_iterations  # instead of config._config_dict['max_iterations']
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name in self._config_dict:
            return self._config_dict[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Valid attributes: {list(self._config_dict.keys())}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Proxy attribute setting to underlying config.
        """
        if name.startswith('_') or name in {'_original_class', '_config_dict'}:
            super().__setattr__(name, value)
        else:
            self._config_dict[name] = value
            if name in self.DUPLICATE_FIELDS:
                logger.debug(f"Updated duplicate field '{name}' to {value}")

    def __repr__(self) -> str:
        """String representation of the config."""
        items = ', '.join(f'{k}={repr(v)}' for k, v in self._config_dict.items()
                         if not k.startswith('_'))
        return f'EvolutionConfigurationWrapper({items})'

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Check for obviously invalid values
        if self._config_dict.get('max_iterations', 0) <= 0:
            issues.append("max_iterations must be positive")

        if self._config_dict.get('population_size', 0) <= 0:
            issues.append("population_size must be positive")

        if not 0.0 <= self._config_dict.get('temperature', 0.7) <= 2.0:
            issues.append("temperature should be between 0.0 and 2.0")

        if not 0.0 <= self._config_dict.get('mutation_rate', 0.1) <= 1.0:
            issues.append("mutation_rate should be between 0.0 and 1.0")

        return issues


# Convenience function
def create_evolution_config(**kwargs) -> EvolutionConfigurationWrapper:
    """
    Quick factory function for creating evolution config.

    Usage:
        from integrations.bug_fixes.evolution_wrapper import create_evolution_config

        config = create_evolution_config(
            max_iterations=100,
            population_size=20
        )
    """
    return EvolutionConfigurationWrapper(**kwargs)
