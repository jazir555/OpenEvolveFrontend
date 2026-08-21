"""
Unified Configuration - Single Source of Truth for All OpenEvolve Parameters

This module provides the UnifiedConfiguration class which serves as the single
source of truth for all 272 OpenEvolve parameters across all modules.

Key Features:
- Validates against parameter_manager's 272-parameter schema
- Provides convenience properties for commonly-accessed parameters
- Supports conversion to module-specific configurations
- Enables parameter merging with proper precedence
- Eliminates duplication across EvolutionConfiguration, AdversarialConfiguration, etc.

Usage:
    from unified_configuration import create_unified_config
    unified = create_unified_config({
        'evolution_mode': 'standard',
        'max_iterations': 10,
        'temperature': 0.7
    })

    # Convert to module-specific configs
    evo_config = unified.to_evolution_config()
    adv_config = unified.to_adversarial_config()
"""
from __future__ import annotations


from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import asdict
import logging

# Import ParameterManager components with backward compatibility
try:
    from parameter_manager import ParameterManager, ValidationResult
    PARAMETER_MANAGER_AVAILABLE = True
except ImportError:
    # Fallback for when parameter_manager is not available
    PARAMETER_MANAGER_AVAILABLE = False

    class ValidationResult:
        """Fallback ValidationResult for when parameter_manager is unavailable"""
        def __init__(self, valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
            self.valid = valid
            self.errors = errors or []
            self.warnings = warnings or []

logger = logging.getLogger(__name__)

# =============================================================================
# LAZY IMPORT HELPER FUNCTIONS
# =============================================================================

def _get_evolution_config():
    """
    Lazy import of EvolutionConfiguration to avoid circular dependencies.

    Returns:
        EvolutionConfiguration class or None if not available
    """
    try:
        from evolution import EvolutionConfiguration
        return EvolutionConfiguration
    except ImportError:
        return None

def _get_adversarial_config():
    """
    Lazy import of AdversarialConfiguration to avoid circular dependencies.

    Returns:
        AdversarialConfiguration class or None if not available
    """
    try:
        from adversarial import AdversarialConfiguration
        return AdversarialConfiguration
    except ImportError:
        return None

# Type variables for generic config conversion
T = TypeVar('T', bound='BaseConfiguration')


class UnifiedConfigurationError(Exception):
    """Base error for UnifiedConfiguration operations"""
    pass


class ConfigurationValidationError(UnifiedConfigurationError):
    """Raised when configuration validation fails"""

    def __init__(self, errors: List[str], warnings: List[str] = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__(f"Configuration validation failed with {len(errors)} errors")


class UnifiedConfiguration:
    """
    Single configuration class for ALL OpenEvolve modules.

    Uses ParameterManager's 272-parameter schema as the source of truth.
    Eliminates the need to redefine parameters in multiple configuration classes.

    This class stores parameters internally as a validated dict and provides:
    - Type-safe access to parameters via properties
    - Dynamic parameter access via get() method
    - Conversion to module-specific configurations
    - Parameter merging capabilities

    Attributes:
        _parameters: Validated parameter dictionary
        _manager: ParameterManager instance for validation
        _cache: Cache for converted configurations
    """

    def __init__(
        self,
        parameters: Dict[str, Any],
        manager: Optional['ParameterManager'] = None,
        validate: bool = True
    ):
        """
        Initialize UnifiedConfiguration with parameter validation.

        Args:
            parameters: Dictionary of parameter values
            manager: ParameterManager instance (creates new if None and available)
            validate: Whether to validate parameters (default: True)

        Raises:
            ConfigurationValidationError: If validation fails and validate=True
        """
        # Only create ParameterManager if it's available
        if PARAMETER_MANAGER_AVAILABLE:
            self._manager = manager or ParameterManager()
        else:
            self._manager = None
            if validate:
                logger.warning("ParameterManager not available - skipping validation")

        self._cache: Dict[str, Any] = {}

        # Validate parameters if requested and manager is available
        if validate and self._manager:
            validation_result = self._manager.validate(parameters)

            if not validation_result.valid:
                raise ConfigurationValidationError(
                    validation_result.errors,
                    validation_result.warnings
                )

            if validation_result.warnings:
                logger.warning(
                    f"Configuration validation warnings: {validation_result.warnings}"
                )

        # Store parameters (with defaults applied for missing values)
        self._parameters = self._apply_defaults(parameters)

        logger.debug(
            f"UnifiedConfiguration created with {len(self._parameters)} parameters"
        )

    def _apply_defaults(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values for missing parameters from ParameterManager schema.

        Args:
            parameters: User-provided parameters (may be incomplete)

        Returns:
            Complete parameter dict with defaults applied
        """
        result = {}

        # Only apply defaults if ParameterManager is available
        if self._manager and PARAMETER_MANAGER_AVAILABLE:
            defaults = self._manager.get_defaults()
            # Start with defaults
            result.update(defaults)

        # Override with user-provided values
        result.update(parameters)

        return result

    # =========================================================================
    # CORE PROPERTIES
    # =========================================================================

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Get all configuration parameters as dictionary.

        Returns:
            Dictionary of all parameters with defaults applied
        """
        return self._parameters.copy() if self._parameters else {}

    @property
    def parameters_fast(self) -> Dict[str, Any]:
        """
        Get all configuration parameters as dictionary WITHOUT copy.

        PERFORMANCE OPTIMIZATION: Use this in performance-critical code
        when you promise not to modify the returned dictionary.

        Returns:
            Direct reference to internal parameters dict (faster, no copy overhead)

        Warning: Do not modify the returned dictionary!
        """
        return self._parameters

    # =========================================================================
    # CONVENIENCE PROPERTIES - Commonly accessed parameters
    # =========================================================================

    @property
    def evolution_mode(self) -> str:
        """Get evolution mode parameter"""
        return self._parameters.get('evolution_mode', 'standard')

    @property
    def max_iterations(self) -> int:
        """Get max iterations parameter"""
        return self._parameters.get('max_iterations', 10)

    @property
    def population_size(self) -> int:
        """Get population size parameter"""
        return self._parameters.get('population_size', 20)

    @property
    def temperature(self) -> float:
        """Get LLM temperature parameter"""
        return self._parameters.get('temperature', 0.7)

    @property
    def max_tokens(self) -> int:
        """Get max tokens parameter"""
        return self._parameters.get('max_tokens', 2048)

    @property
    def seed(self) -> Optional[int]:
        """Get random seed parameter"""
        return self._parameters.get('seed')

    @property
    def api_key(self) -> str:
        """Get API key parameter"""
        return self._parameters.get('api_key', '')

    @property
    def api_base(self) -> str:
        """Get API base URL parameter"""
        return self._parameters.get('api_base', 'https://api.openai.com/v1')

    @property
    def model_id(self) -> str:
        """Get model ID parameter"""
        return self._parameters.get('model_id', 'gpt-4')

    # =========================================================================
    # ADVERSARIAL-SPECIFIC PROPERTIES
    # =========================================================================

    @property
    def adversarial_rounds(self) -> int:
        """Get adversarial rounds parameter"""
        return self._parameters.get('adversarial_rounds', 5)

    @property
    def attack_strength(self) -> float:
        """Get attack strength parameter"""
        return self._parameters.get('attack_strength', 0.5)

    @property
    def defense_strategy(self) -> str:
        """Get defense strategy parameter"""
        return self._parameters.get('defense_strategy', 'reactive')

    # =========================================================================
    # DYNAMIC PARAMETER ACCESS
    # =========================================================================

    def get(self, name: str, default: Any = None) -> Any:
        """
        Get any parameter by name.

        Args:
            name: Parameter name
            default: Default value if parameter not found

        Returns:
            Parameter value or default
        """
        return self._parameters.get(name, default)

    def get_category_params(self, category: str) -> Dict[str, Any]:
        """
        Get all parameters for a specific category.

        Args:
            category: Category name (e.g., 'core_evolution', 'adversarial')

        Returns:
            Dictionary of parameters in that category
        """
        category_params = {}
        for param_name, param_def in self._manager.schema.parameters.items():
            if param_def.category == category:
                category_params[param_name] = self._parameters.get(param_name)

        return category_params

    def set(self, name: str, value: Any, validate: bool = False) -> None:
        """
        Set a parameter value.

        Args:
            name: Parameter name
            value: New value
            validate: Whether to validate the new value

        Raises:
            ConfigurationValidationError: If validation fails
        """
        if validate:
            # Validate single parameter
            test_params = self._parameters.copy()
            test_params[name] = value
            validation = self._manager.validate(test_params)
            if not validation.valid:
                raise ConfigurationValidationError(validation.errors)

        self._parameters[name] = value
        # Clear cache when parameters change
        self._cache.clear()

    def update(self, parameters: Dict[str, Any], validate: bool = True) -> None:
        """
        Update multiple parameters at once.

        Args:
            parameters: Dictionary of parameters to update
            validate: Whether to validate the updated configuration

        Raises:
            ConfigurationValidationError: If validation fails
        """
        if validate:
            # Validate merged parameters
            test_params = self._parameters.copy()
            test_params.update(parameters)
            validation = self._manager.validate(test_params)
            if not validation.valid:
                raise ConfigurationValidationError(validation.errors)

        self._parameters.update(parameters)
        # Clear cache when parameters change
        self._cache.clear()

    # =========================================================================
    # PARAMETER MERGING
    # =========================================================================

    def merge(self, *others: Dict[str, Any], validate: bool = True) -> 'UnifiedConfiguration':
        """
        Merge this configuration with other parameter dictionaries.

        Later dictionaries take precedence over earlier ones.

        Args:
            *others: Variable number of parameter dictionaries to merge
            validate: Whether to validate the merged result

        Returns:
            New UnifiedConfiguration with merged parameters

        Raises:
            ConfigurationValidationError: If validation fails
        """
        # Start with current parameters
        merged = self._parameters.copy()

        # Apply each dict in sequence (later ones override)
        for other in others:
            merged.update(other)

        # Create new UnifiedConfiguration with merged params
        return UnifiedConfiguration(merged, self._manager, validate=validate)

    # =========================================================================
    # CONVERSION TO MODULE-SPECIFIC CONFIGURATIONS
    # =========================================================================

    def to_evolution_config(self) -> 'EvolutionConfiguration':
        """
        Convert to EvolutionConfiguration for evolution module.

        Returns:
            EvolutionConfiguration instance with all parameters

        Note:
            This method imports EvolutionConfiguration lazily to avoid
            circular import issues.
        """
        # Check cache first
        cache_key = 'evolution_config'
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Lazy import to avoid circular dependency
        EvolutionConfiguration = _get_evolution_config()
        if EvolutionConfiguration is None:
            raise ImportError("EvolutionConfiguration not available")

        if self._manager:
            config = EvolutionConfiguration.from_parameter_manager(
                self._manager,
                self._parameters
            )
        else:
            # Fallback: create config directly from parameters
            config = EvolutionConfiguration()
            for key, value in self._parameters.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Cache the result
        self._cache[cache_key] = config
        return config

    def to_adversarial_config(self) -> 'AdversarialConfiguration':
        """
        Convert to AdversarialConfiguration for adversarial module.

        Returns:
            AdversarialConfiguration instance with all parameters
        """
        # Check cache first
        cache_key = 'adversarial_config'
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Lazy import to avoid circular dependency
        AdversarialConfiguration = _get_adversarial_config()
        if AdversarialConfiguration is None:
            raise ImportError("AdversarialConfiguration not available")

        if self._manager:
            config = AdversarialConfiguration.from_parameter_manager(
                self._manager,
                self._parameters
            )
        else:
            # Fallback: create config directly from parameters
            config = AdversarialConfiguration()
            for key, value in self._parameters.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Cache the result
        self._cache[cache_key] = config
        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Complete parameter dictionary
        """
        return self._parameters.copy()

    def to_dict_fast(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary WITHOUT copy.

        PERFORMANCE OPTIMIZATION: Faster alternative to to_dict() for read-only access.

        Returns:
            Direct reference to internal parameters dict

        Warning: Do not modify the returned dictionary!
        """
        return self._parameters

    def bulk_get(self, *names: str) -> Dict[str, Any]:
        """
        Get multiple parameters efficiently in a single call.

        PERFORMANCE OPTIMIZATION: Faster than multiple individual get() calls.

        Args:
            *names: Variable number of parameter names to retrieve

        Returns:
            Dictionary mapping parameter names to their values

        Example:
            # Get multiple params at once
            params = config.bulk_get('max_iterations', 'temperature', 'population_size')
            max_iter = params['max_iterations']
            temp = params['temperature']
            pop = params['population_size']
        """
        return {name: self._parameters.get(name) for name in names}

    def cache_frequently_used(self, *names: str) -> 'CachedConfigView':
        """
        Create a cached view of frequently used parameters.

        PERFORMANCE OPTIMIZATION: Use this to cache parameter values that are
        accessed in tight loops or performance-critical code.

        Args:
            *names: Variable number of parameter names to cache

        Returns:
            CachedConfigView object with cached parameter values

        Example:
            # Cache parameters before entering loop
            cached = config.cache_frequently_used('max_iterations', 'temperature')
            for i in range(cached.max_iterations):
                # Access cached.temperature without property overhead
                result = process(i * cached.temperature)
        """
        return CachedConfigView(self, names)

    def validate(self) -> ValidationResult:
        """
        Validate the current configuration.

        Returns:
            ValidationResult with validation status
        """
        return self._manager.validate(self._parameters)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def __repr__(self) -> str:
        """String representation showing key parameters"""
        return (
            f"UnifiedConfiguration("
            f"mode={self.evolution_mode}, "
            f"iterations={self.max_iterations}, "
            f"temp={self.temperature}, "
            f"{len(self._parameters)} params total)"
        )

    def __len__(self) -> int:
        """Return number of parameters"""
        return len(self._parameters)

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists"""
        return name in self._parameters

    def __getitem__(self, name: str) -> Any:
        """Allow dict-style access: config['temperature']"""
        return self._parameters[name]

    def __setitem__(self, name: str, value: Any) -> None:
        """Allow dict-style setting: config['temperature'] = 0.8"""
        self.set(name, value, validate=True)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_unified_config(
    parameters: Optional[Dict[str, Any]] = None,
    manager: Optional['ParameterManager'] = None,
    validate: bool = True
) -> UnifiedConfiguration:
    """
    Factory function to create UnifiedConfiguration with defaults.

    Args:
        parameters: Initial parameters (uses all defaults if None)
        manager: ParameterManager instance (creates new if None and available)
        validate: Whether to validate parameters (default: True)

    Returns:
        UnifiedConfiguration instance
    """
    # Only create manager if ParameterManager is available
    if PARAMETER_MANAGER_AVAILABLE and manager is None:
        manager = ParameterManager()

    params = parameters or {}

    return UnifiedConfiguration(params, manager, validate=validate)


def merge_configs(
    *configs: Dict[str, Any],
    manager: Optional['ParameterManager'] = None
) -> UnifiedConfiguration:
    """
    Merge multiple configuration dictionaries into UnifiedConfiguration.

    Args:
        *configs: Variable number of config dicts to merge
        manager: ParameterManager for validation (creates new if None and available)

    Returns:
        UnifiedConfiguration with merged parameters

    Note:
        Later configs override earlier ones (last one wins)
    """
    # Only create manager if ParameterManager is available
    if PARAMETER_MANAGER_AVAILABLE and manager is None:
        manager = ParameterManager()

    # Merge all configs (later ones take precedence)
    merged = {}
    for config in reversed(configs):  # Reverse so first config has priority
        merged.update(config)

    return UnifiedConfiguration(merged, manager)


def load_unified_config_from_file(
    filepath: str,
    manager: Optional['ParameterManager'] = None
) -> UnifiedConfiguration:
    """
    Load UnifiedConfiguration from JSON file.

    Args:
        filepath: Path to JSON configuration file
        manager: ParameterManager instance (creates new if None and available)

    Returns:
        UnifiedConfiguration instance

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ConfigurationValidationError: If config is invalid
    """
    import json

    # Only create manager if ParameterManager is available
    if PARAMETER_MANAGER_AVAILABLE and manager is None:
        manager = ParameterManager()

    with open(filepath, 'r') as f:
        parameters = json.load(f)

    return UnifiedConfiguration(parameters, manager)


def save_unified_config_to_file(
    config: UnifiedConfiguration,
    filepath: str,
    pretty: bool = True
) -> None:
    """
    Save UnifiedConfiguration to JSON file.

    Args:
        config: UnifiedConfiguration to save
        filepath: Path to save configuration
        pretty: Whether to format JSON prettily (default: True)
    """
    import json

    with open(filepath, 'w') as f:
        if pretty:
            json.dump(config.to_dict(), f, indent=2)
        else:
            json.dump(config.to_dict(), f)

    logger.info(f"Configuration saved to {filepath}")


# =============================================================================
# CONVENIENCE PRESETS
# =============================================================================

def create_standard_evolution_config(
    max_iterations: int = 10,
    population_size: int = 20,
    temperature: float = 0.7,
    **kwargs
) -> UnifiedConfiguration:
    """
    Create UnifiedConfiguration with standard evolution presets.

    Args:
        max_iterations: Maximum iterations
        population_size: Population size
        temperature: LLM temperature
        **kwargs: Additional parameters to override

    Returns:
        UnifiedConfiguration for standard evolution
    """
    params = {
        'evolution_mode': 'standard',
        'max_iterations': max_iterations,
        'population_size': population_size,
        'temperature': temperature,
        **kwargs
    }

    return create_unified_config(params)


def create_adversarial_testing_config(
    adversarial_rounds: int = 5,
    attack_strength: float = 0.5,
    defense_strategy: str = 'reactive',
    **kwargs
) -> UnifiedConfiguration:
    """
    Create UnifiedConfiguration with adversarial testing presets.

    Args:
        adversarial_rounds: Number of adversarial rounds
        attack_strength: Strength of attacks (0.0-1.0)
        defense_strategy: Defense strategy to use
        **kwargs: Additional parameters to override

    Returns:
        UnifiedConfiguration for adversarial testing
    """
    params = {
        'evolution_mode': 'adversarial',
        'adversarial_rounds': adversarial_rounds,
        'attack_strength': attack_strength,
        'defense_strategy': defense_strategy,
        **kwargs
    }

    return create_unified_config(params)


def create_quality_diversity_config(
    archive_size: int = 100,
    feature_bins: int = 10,
    diversity_weight: float = 0.5,
    **kwargs
) -> UnifiedConfiguration:
    """
    Create UnifiedConfiguration with quality diversity presets.

    Args:
        archive_size: Size of archive for MAP-Elites
        feature_bins: Number of bins per feature dimension
        diversity_weight: Weight of diversity vs quality (0.0-1.0)
        **kwargs: Additional parameters to override

    Returns:
        UnifiedConfiguration for quality diversity evolution
    """
    params = {
        'evolution_mode': 'quality_diversity',
        'archive_size': archive_size,
        'feature_bins': feature_bins,
        'diversity_weight': diversity_weight,
        **kwargs
    }

    return create_unified_config(params)


# =============================================================================
# PERFORMANCE OPTIMIZATION CLASSES
# =============================================================================

class CachedConfigView:
    """
    Cached view of frequently accessed configuration parameters.

    PERFORMANCE OPTIMIZATION: Use this to cache parameter values for tight loops
    or performance-critical code paths. Eliminates property access overhead.

    Usage:
        # Cache parameters before loop
        cached = config.cache_frequently_used('max_iterations', 'temperature')

        # Access in loop without overhead
        for i in range(cached.max_iterations):
            result = process(i * cached.temperature)
    """

    def __init__(self, config: UnifiedConfiguration, param_names: tuple):
        """
        Initialize cached view with parameter values.

        Args:
            config: UnifiedConfiguration instance
            param_names: Tuple of parameter names to cache
        """
        self._cached_values = {}
        for name in param_names:
            self._cached_values[name] = config._parameters.get(name)

        # Create attributes for fast access
        for name, value in self._cached_values.items():
            setattr(self, name, value)

    def get(self, name: str, default: Any = None) -> Any:
        """Get cached parameter value"""
        return self._cached_values.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        """Get all cached values as dictionary"""
        return self._cached_values.copy()

    def __repr__(self) -> str:
        """String representation"""
        return f"CachedConfigView({len(self._cached_values)} params)"

