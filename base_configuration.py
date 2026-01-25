"""
BaseConfiguration - Foundation for OpenEvolve Configuration Classes
====================================================================

This module provides the BaseConfiguration class which serves as the foundation
for all OpenEvolve configuration classes, eliminating parameter duplication
across EvolutionConfiguration, AdversarialConfiguration, and other config classes.

Benefits:
- Single source of truth for all 272 parameters
- No need to redefine parameters in each configuration class
- Consistent validation, merging, and conversion across all configs
- Automatic integration with UnifiedConfiguration
- Eliminates ~800+ lines of duplicate parameter definitions

Usage:
    class MyConfiguration(BaseConfiguration):
        # No need to redefine parameters!
        # Just specify which categories you need
        pass

    config = MyConfiguration({'max_iterations': 20, 'temperature': 0.8})

Migration Guide:
    OLD (EvolutionConfiguration in evolution.py):
        class EvolutionConfiguration:
            max_iterations: int = 10
            temperature: float = 0.7
            # ... 270 more parameters duplicated

    NEW (with BaseConfiguration):
        from base_configuration import BaseConfiguration
        from unified_configuration import UnifiedConfiguration

        class EvolutionConfiguration(BaseConfiguration):
            # No parameter duplication needed!
            pass

        # Use UnifiedConfiguration for all 272 parameters
        unified = UnifiedConfiguration({'max_iterations': 20})
        evolution_config = EvolutionConfiguration.from_unified_config(unified)
"""

from typing import Dict, Any, Optional, List, Type, TypeVar, get_type_hints, TYPE_CHECKING
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

# Import ParameterManager with backward compatibility
try:
    from parameter_manager import ParameterManager
    PARAMETER_MANAGER_AVAILABLE = True
except ImportError:
    PARAMETER_MANAGER_AVAILABLE = False
    # Create a dummy type for type hints
    if TYPE_CHECKING:
        from parameter_manager import ParameterManager

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar('T', bound='BaseConfiguration')


class ConfigurationError(Exception):
    """Base exception for configuration errors"""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Exception raised when configuration validation fails"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed with {len(errors)} errors")


class BaseConfiguration(ABC):
    """
    Base class for all OpenEvolve configuration classes.

    This class provides a foundation for configuration management that:
    - Eliminates parameter duplication across config classes
    - Integrates seamlessly with UnifiedConfiguration
    - Provides consistent validation and conversion methods
    - Supports merging, cloning, and serialization

    Instead of redefining 272 parameters in each configuration class,
    subclasses simply inherit from BaseConfiguration and use the
    UnifiedConfiguration system internally.

    Attributes:
        _unified_config: The internal UnifiedConfiguration instance
        _manager: The ParameterManager instance used for validation (if available)
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        manager: Optional['ParameterManager'] = None,
        validate: bool = True
    ):
        """
        Initialize BaseConfiguration.

        Args:
            parameters: Dictionary of configuration parameters
            manager: Optional ParameterManager instance (creates default if None and available)
            validate: Whether to validate the configuration

        Raises:
            ConfigurationValidationError: If validation fails and validate=True
        """
        # Import here to avoid circular dependency
        from unified_configuration import UnifiedConfiguration

        # Initialize with empty config if none provided
        parameters = parameters or {}

        # Create UnifiedConfiguration with parameters
        if validate:
            self._unified_config = UnifiedConfiguration(parameters, manager, validate=True)
        else:
            self._unified_config = UnifiedConfiguration(parameters, manager, validate=False)

        # Store manager reference if available
        self._manager = getattr(self._unified_config, '_manager', None)

        logger.debug(f"Initialized {self.__class__.__name__} with {len(parameters)} parameters")

    # =========================================================================
    # CORE PROPERTIES
    # =========================================================================

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get all configuration parameters as dictionary"""
        # UnifiedConfiguration stores parameters in _parameters (private attribute)
        if hasattr(self._unified_config, '_parameters'):
            return self._unified_config._parameters.copy()
        # Fallback: try to access as property
        elif hasattr(self._unified_config, 'parameters'):
            return self._unified_config.parameters.copy()
        else:
            # Last resort: try to convert to dict
            try:
                return dict(self._unified_config)
            except (TypeError, ValueError, AttributeError) as e:
                logger.error(f"Failed to convert config to dict: {type(e).__name__}: {e}", exc_info=True)
                return {}

    @property
    def manager(self) -> Optional['ParameterManager']:
        """Get the ParameterManager instance (if available)"""
        return self._manager

    # =========================================================================
    # UNIVERSAL PARAMETER ACCESS (All 272 parameters available)
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """
        Get parameter value by name.

        This allows accessing any of the 272 parameters as attributes,
        even if they're not explicitly defined in subclasses.

        Args:
            name: Parameter name

        Returns:
            Parameter value

        Raises:
            AttributeError: If parameter doesn't exist
        """
        # Check if it's a real attribute (not a parameter)
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # Try to get from unified config
        try:
            return self._unified_config.get(name)
        except (KeyError, AttributeError):
            raise AttributeError(
                f"Parameter '{name}' not found in configuration. "
                f"Use .parameters.keys() to see available parameters."
            )

    def get(self, name: str, default: Any = None) -> Any:
        """
        Safely get parameter value with default fallback.

        Args:
            name: Parameter name
            default: Default value if parameter not found

        Returns:
            Parameter value or default
        """
        try:
            return self._unified_config.get(name, default)
        except (KeyError, AttributeError):
            return default

    def set(self, name: str, value: Any, validate: bool = True) -> None:
        """
        Set parameter value.

        Args:
            name: Parameter name
            value: New value
            validate: Whether to validate after setting
        """
        self._unified_config.set(name, value, validate)

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate(self) -> 'ValidationResult':
        """
        Validate the configuration.

        Returns:
            ValidationResult with validation status

        Example:
            result = config.validate()
            if not result.valid:
                print(f"Errors: {result.errors}")
        """
        return self._unified_config.validate()

    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return self.validate().valid

    # =========================================================================
    # CONVERSION METHODS
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary of all parameters
        """
        return self.parameters

    def to_json(self) -> str:
        """
        Convert configuration to JSON string.

        Returns:
            JSON representation of configuration
        """
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary of parameters

        Returns:
            Configuration instance

        Example:
            config = MyConfiguration.from_dict({'max_iterations': 20})
        """
        return cls(parameters=config_dict)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON string

        Returns:
            Configuration instance

        Example:
            config = MyConfiguration.from_json('{"max_iterations": 20}')
        """
        import json
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_unified_config(cls: Type[T], unified_config: 'UnifiedConfiguration') -> T:
        """
        Create configuration from UnifiedConfiguration.

        This is the primary integration point with the unified configuration system.

        Args:
            unified_config: UnifiedConfiguration instance

        Returns:
            Configuration instance

        Example:
            from unified_configuration import create_unified_config
            unified = create_unified_config({'evolution_mode': 'standard'})
            evolution_config = EvolutionConfiguration.from_unified_config(unified)
        """
        # Create instance with unified config's parameters
        return cls(parameters=unified_config.parameters, validate=False)

    # =========================================================================
    # MERGE AND CLONE
    # =========================================================================

    def merge(self, *others: Dict[str, Any], validate: bool = True) -> 'BaseConfiguration':
        """
        Merge this configuration with others.

        Later configurations override earlier ones.

        Args:
            *others: Other configuration dictionaries to merge
            validate: Whether to validate the merged configuration

        Returns:
            New configuration instance with merged parameters

        Example:
            merged = config1.merge(config2, {'max_iterations': 50})
        """
        merged_params = self.to_dict()

        for other in others:
            merged_params.update(other)

        # Create new instance of same class
        return self.__class__(parameters=merged_params, validate=validate)

    def clone(self) -> 'BaseConfiguration':
        """
        Create a deep copy of this configuration.

        Returns:
            New configuration instance with same parameters

        Example:
            config_copy = config.clone()
        """
        import copy
        cloned_params = copy.deepcopy(self.to_dict())
        return self.__class__(parameters=cloned_params, validate=False)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the configuration.

        Returns:
            Dictionary with configuration summary
        """
        return self._manager.get_config_summary(self.to_dict())

    def print_summary(self) -> None:
        """Print a formatted summary of the configuration"""
        summary = self.get_summary()

        print(f"\n{'='*60}")
        print(f"Configuration Summary: {self.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Total Parameters: {summary['total_parameters']}")
        print(f"Categories: {len(summary['categories'])}")
        print(f"Overrides: {summary['overrides']}")
        print(f"Defaults Used: {summary['defaults_used']}")

        if summary['categories']:
            print(f"\nCategories:")
            for category, count in summary['categories'].items():
                print(f"  • {category}: {count} parameters")

        if summary['missing_required']:
            print(f"\n⚠️  Missing Required: {', '.join(summary['missing_required'])}")

        print(f"{'='*60}\n")

    def export_typescript(self, filepath: str) -> None:
        """
        Export parameter schema as TypeScript types.

        Args:
            filepath: Path to output .ts file

        Example:
            config.export_typescript('openevolve_parameters.ts')
        """
        self._manager.export_typescript_types(filepath)

    # =========================================================================
    # STRING REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        params_count = len(self.to_dict())
        return f"{self.__class__.__name__}(parameters={params_count})"

    def __str__(self) -> str:
        summary = self.get_summary()
        return (
            f"{self.__class__.__name__}:\n"
            f"  Parameters: {summary['total_parameters']}\n"
            f"  Categories: {len(summary['categories'])}\n"
            f"  Overrides: {summary['overrides']}"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_config_from_parameter_manager(
    manager: 'ParameterManager',
    session_state: Optional[Dict[str, Any]] = None,
    config_class: Type[BaseConfiguration] = BaseConfiguration
) -> BaseConfiguration:
    """
    Create configuration from ParameterManager.

    This provides backward compatibility with existing code that uses
    ParameterManager and session_state pattern.

    Args:
        manager: ParameterManager instance
        session_state: Optional session state dictionary
        config_class: Configuration class to instantiate

    Returns:
        Configuration instance

    Example:
        from parameter_manager import ParameterManager
        manager = ParameterManager()
        config = create_config_from_parameter_manager(manager, st.session_state)

    Note:
        This function is deprecated. Use UnifiedConfiguration directly:
        from unified_configuration import create_unified_config
        config = create_unified_config(session_state)
    """
    # Get defaults from manager if available
    if PARAMETER_MANAGER_AVAILABLE and manager:
        defaults = manager.get_defaults()

        # Merge with session state if provided
        if session_state:
            defaults.update(session_state)

        # Create configuration
        return config_class(parameters=defaults, manager=manager, validate=True)
    else:
        # Fallback when ParameterManager is not available
        logger.warning("ParameterManager not available - creating config from session state only")
        parameters = session_state or {}
        return config_class(parameters=parameters, manager=None, validate=False)


# =============================================================================
# SPECIALIZED CONFIGURATION CLASSES
# =============================================================================

class EvolutionConfiguration(BaseConfiguration):
    """
    Evolution-specific configuration.

    Inherits all 272 parameters from BaseConfiguration,
    with evolution-specific defaults and validation.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None, validate: bool = True):
        # Set evolution-specific defaults
        evolution_defaults = {
            'evolution_mode': 'standard',
            'max_iterations': 10,
            'population_size': 20,
            'temperature': 0.7,
        }

        # Merge with provided parameters
        merged_params = evolution_defaults.copy()
        if parameters:
            merged_params.update(parameters)

        super().__init__(parameters=merged_params, validate=validate)


class AdversarialConfiguration(BaseConfiguration):
    """
    Adversarial-specific configuration.

    Inherits all 272 parameters from BaseConfiguration,
    with adversarial-specific defaults and validation.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None, validate: bool = True):
        # Set adversarial-specific defaults
        adversarial_defaults = {
            'evolution_mode': 'adversarial',
            'adversarial_rounds': 5,
            'attack_strength': 0.5,
            'defense_strategy': 'reactive',
        }

        # Merge with provided parameters
        merged_params = adversarial_defaults.copy()
        if parameters:
            merged_params.update(parameters)

        super().__init__(parameters=merged_params, validate=validate)


class QualityDiversityConfiguration(BaseConfiguration):
    """
    Quality Diversity (MAP-Elites) configuration.

    Inherits all 272 parameters from BaseConfiguration,
    with QD-specific defaults and validation.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None, validate: bool = True):
        # Set QD-specific defaults
        qd_defaults = {
            'evolution_mode': 'quality_diversity',
            'archive_size': 100,
            'feature_bins': 10,
            'diversity_weight': 0.5,
        }

        # Merge with provided parameters
        merged_params = qd_defaults.copy()
        if parameters:
            merged_params.update(parameters)

        super().__init__(parameters=merged_params, validate=validate)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    'BaseConfiguration',

    # Specialized configurations
    'EvolutionConfiguration',
    'AdversarialConfiguration',
    'QualityDiversityConfiguration',

    # Factory functions
    'create_config_from_parameter_manager',

    # Exceptions
    'ConfigurationError',
    'ConfigurationValidationError',
]


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    print("Testing BaseConfiguration")
    print("="*60)

    # Test basic creation
    config = EvolutionConfiguration({
        'max_iterations': 20,
        'temperature': 0.8
    })

    print(f"Created: {config}")
    print(f"Max iterations: {config.get('max_iterations')}")
    print(f"Temperature: {config.get('temperature')}")

    # Test parameter access
    print(f"\nEvolution mode: {config.evolution_mode}")

    # Test merge
    merged = config.merge({'max_iterations': 50})
    print(f"Merged max_iterations: {merged.get('max_iterations')}")

    # Test clone
    cloned = config.clone()
    print(f"Cloned: {cloned}")

    # Test validation
    validation = config.validate()
    print(f"\nValidation: {validation.valid}")

    # Test summary
    config.print_summary()

    print("\n✓ BaseConfiguration testing complete!")
