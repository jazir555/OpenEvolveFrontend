"""
Comprehensive Configuration System for OpenEvolve Gauntlet System

Provides centralized configuration management with validation,
environment variable support, and runtime configuration updates.

Key Features:
- Environment-based configuration
- Configuration validation
- Runtime configuration updates
- Profile-based configuration
- Configuration serialization/deserialization
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import logging
import json
from pathlib import Path
from datetime import timedelta
import copy

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


class CacheType(Enum):
    """Cache backend types"""
    MEMORY = "memory"
    REDIS = "redis"
    NONE = "none"


class CheckpointFrequency(Enum):
    """Checkpoint frequency levels"""
    MAJOR = "major"  # Only at major milestones
    MINOR = "minor"  # At minor milestones
    ALL = "all"  # After every operation


class CircuitBreakerStrategy(Enum):
    """Circuit breaker strategies"""
    INDIVIDUAL = "individual"  # Per-problem breaker
    HIERARCHICAL = "hierarchical"  # Per-level breaker
    GLOBAL = "global"  # Single breaker for all


class DifficultyLevel(Enum):
    """Difficulty levels for problems"""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


class StrategyProfile(Enum):
    """Predefined strategy profiles"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    FAST = "fast"
    THOROUGH = "thorough"


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = True
    cache_type: CacheType = CacheType.MEMORY
    ttl_seconds: int = 3600
    max_size: int = 1000
    redis_url: Optional[str] = None

    def validate(self) -> tuple[bool, List[str]]:
        """Validate cache configuration"""
        errors = []

        if self.ttl_seconds < 0:
            errors.append("TTL must be non-negative")

        if self.max_size < 0:
            errors.append("Max size must be non-negative")

        if self.cache_type == CacheType.REDIS and not self.redis_url:
            errors.append("Redis URL required when cache_type is REDIS")

        return (len(errors) == 0, errors)


@dataclass
class CheckpointConfig:
    """Checkpointing configuration"""
    enabled: bool = True
    storage_path: str = "./gauntlet_checkpoints"
    compression: bool = True
    frequency: CheckpointFrequency = CheckpointFrequency.MAJOR
    retention_count: int = 5
    auto_cleanup: bool = True

    def validate(self) -> tuple[bool, List[str]]:
        """Validate checkpoint configuration"""
        errors = []

        if self.retention_count < 0:
            errors.append("Retention count must be non-negative")

        return (len(errors) == 0, errors)


@dataclass
class ParallelExecutionConfig:
    """Parallel execution configuration"""
    enabled: bool = True
    max_parallelism: int = 10
    timeout_seconds: int = 300
    use_worker_pool: bool = False
    worker_pool_size: int = 4

    def validate(self) -> tuple[bool, List[str]]:
        """Validate parallel execution configuration"""
        errors = []

        if self.max_parallelism < 1:
            errors.append("Max parallelism must be at least 1")

        if self.timeout_seconds < 0:
            errors.append("Timeout must be non-negative")

        if self.worker_pool_size < 1:
            errors.append("Worker pool size must be at least 1")

        return (len(errors) == 0, errors)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    enabled: bool = True
    strategy: CircuitBreakerStrategy = CircuitBreakerStrategy.HIERARCHICAL
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3

    def validate(self) -> tuple[bool, List[str]]:
        """Validate circuit breaker configuration"""
        errors = []

        if self.failure_threshold < 1:
            errors.append("Failure threshold must be at least 1")

        if self.recovery_timeout_seconds < 0:
            errors.append("Recovery timeout must be non-negative")

        if self.half_open_max_calls < 1:
            errors.append("Half-open max calls must be at least 1")

        return (len(errors) == 0, errors)


@dataclass
class FuzzingConfig:
    """Fuzzing configuration"""
    enabled: bool = False
    max_iterations: int = 1000
    timeout_seconds: int = 30
    crash_analysis_enabled: bool = True
    input_types: List[str] = field(default_factory=lambda: ["auto"])

    def validate(self) -> tuple[bool, List[str]]:
        """Validate fuzzing configuration"""
        errors = []

        if self.max_iterations < 1:
            errors.append("Max iterations must be at least 1")

        if self.timeout_seconds < 0:
            errors.append("Timeout must be non-negative")

        return (len(errors) == 0, errors)


@dataclass
class DifficultyConfig:
    """Dynamic difficulty configuration"""
    enabled: bool = True
    initial_level: DifficultyLevel = DifficultyLevel.MEDIUM
    adjustment_window: int = 10
    success_threshold: float = 0.7
    failure_threshold: float = 0.3

    def validate(self) -> tuple[bool, List[str]]:
        """Validate difficulty configuration"""
        errors = []

        if self.adjustment_window < 1:
            errors.append("Adjustment window must be at least 1")

        if not 0 <= self.success_threshold <= 1:
            errors.append("Success threshold must be between 0 and 1")

        if not 0 <= self.failure_threshold <= 1:
            errors.append("Failure threshold must be between 0 and 1")

        if self.success_threshold <= self.failure_threshold:
            errors.append("Success threshold must be greater than failure threshold")

        return (len(errors) == 0, errors)


@dataclass
class MLDecompositionConfig:
    """ML-based decomposition configuration"""
    enabled: bool = False
    model_path: Optional[str] = None
    data_collection_path: str = "./data/decomposition"
    auto_train: bool = False
    training_threshold: int = 100

    def validate(self) -> tuple[bool, List[str]]:
        """Validate ML decomposition configuration"""
        errors = []

        if self.training_threshold < 1:
            errors.append("Training threshold must be at least 1")

        return (len(errors) == 0, errors)


@dataclass
class PluginConfig:
    """Plugin system configuration"""
    enabled: bool = False
    plugin_dir: str = "./plugins"
    sandbox_enabled: bool = True
    sandbox_timeout: float = 30.0
    auto_load: bool = False

    def validate(self) -> tuple[bool, List[str]]:
        """Validate plugin configuration"""
        errors = []

        if self.sandbox_timeout < 0:
            errors.append("Sandbox timeout must be non-negative")

        return (len(errors) == 0, errors)


@dataclass
class GauntletConfig:
    """Main Gauntlet system configuration"""

    # Sub-configurations
    cache: CacheConfig = field(default_factory=CacheConfig)
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    parallel_execution: ParallelExecutionConfig = field(default_factory=ParallelExecutionConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    fuzzing: FuzzingConfig = field(default_factory=FuzzingConfig)
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    ml_decomposition: MLDecompositionConfig = field(default_factory=MLDecompositionConfig)
    plugin: PluginConfig = field(default_factory=PluginConfig)

    # General settings
    max_gauntlet_rounds: int = 3
    pass_threshold: float = 0.75
    max_decomposition_depth: int = 3
    log_level: str = "INFO"

    # Strategy profile
    strategy_profile: StrategyProfile = StrategyProfile.BALANCED

    def validate(self) -> tuple[bool, List[str]]:
        """Validate entire configuration"""
        all_errors = []

        # Validate each sub-config
        for config_field in [
            'cache', 'checkpointing', 'parallel_execution',
            'circuit_breaker', 'fuzzing', 'difficulty',
            'ml_decomposition', 'plugin'
        ]:
            config = getattr(self, config_field)
            if hasattr(config, 'validate'):
                valid, errors = config.validate()
                if not valid:
                    all_errors.extend([
                        f"{config_field}.{error}" for error in errors
                    ])

        # Validate general settings
        if self.max_gauntlet_rounds < 1:
            all_errors.append("max_gauntlet_rounds must be at least 1")

        if not 0 <= self.pass_threshold <= 1:
            all_errors.append("pass_threshold must be between 0 and 1")

        if self.max_decomposition_depth < 0:
            all_errors.append("max_decomposition_depth must be non-negative")

        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_log_levels:
            all_errors.append(f"log_level must be one of {valid_log_levels}")

        return (len(all_errors) == 0, all_errors)

    @classmethod
    def from_env(cls: Type[T]) -> T:
        """Create configuration from environment variables"""
        config = cls()

        # Cache settings
        config.cache.enabled = _env_bool('CACHE_ENABLED', config.cache.enabled)
        config.cache.cache_type = CacheType(
            _env_str('CACHE_TYPE', config.cache.cache_type.value)
        )
        config.cache.ttl_seconds = _env_int('CACHE_TTL_SECONDS', config.cache.ttl_seconds)
        config.cache.max_size = _env_int('CACHE_MAX_SIZE', config.cache.max_size)
        config.cache.redis_url = _env_str('CACHE_REDIS_URL', config.cache.redis_url)

        # Checkpointing settings
        config.checkpointing.enabled = _env_bool('CHECKPOINTING_ENABLED', config.checkpointing.enabled)
        config.checkpointing.storage_path = _env_str(
            'CHECKPOINTING_STORAGE_PATH',
            config.checkpointing.storage_path
        )
        config.checkpointing.compression = _env_bool(
            'CHECKPOINTING_COMPRESSION',
            config.checkpointing.compression
        )
        config.checkpointing.frequency = CheckpointFrequency(
            _env_str('CHECKPOINTING_FREQUENCY', config.checkpointing.frequency.value)
        )
        config.checkpointing.retention_count = _env_int(
            'CHECKPOINTING_RETENTION_COUNT',
            config.checkpointing.retention_count
        )

        # Parallel execution settings
        config.parallel_execution.enabled = _env_bool(
            'PARALLEL_EXECUTION_ENABLED',
            config.parallel_execution.enabled
        )
        config.parallel_execution.max_parallelism = _env_int(
            'PARALLEL_EXECUTION_MAX_PARALLELISM',
            config.parallel_execution.max_parallelism
        )
        config.parallel_execution.timeout_seconds = _env_int(
            'PARALLEL_EXECUTION_TIMEOUT_SECONDS',
            config.parallel_execution.timeout_seconds
        )

        # Circuit breaker settings
        config.circuit_breaker.enabled = _env_bool(
            'CIRCUIT_BREAKER_ENABLED',
            config.circuit_breaker.enabled
        )
        config.circuit_breaker.strategy = CircuitBreakerStrategy(
            _env_str('CIRCUIT_BREAKER_STRATEGY', config.circuit_breaker.strategy.value)
        )
        config.circuit_breaker.failure_threshold = _env_int(
            'CIRCUIT_BREAKER_FAILURE_THRESHOLD',
            config.circuit_breaker.failure_threshold
        )

        # Fuzzing settings
        config.fuzzing.enabled = _env_bool('FUZZING_ENABLED', config.fuzzing.enabled)
        config.fuzzing.max_iterations = _env_int(
            'FUZZING_MAX_ITERATIONS',
            config.fuzzing.max_iterations
        )

        # Difficulty settings
        config.difficulty.enabled = _env_bool(
            'DIFFICULTY_ADJUSTMENT_ENABLED',
            config.difficulty.enabled
        )
        config.difficulty.initial_level = DifficultyLevel(
            _env_int('DIFFICULTY_INITIAL_LEVEL', config.difficulty.initial_level.value)
        )

        # ML decomposition settings
        config.ml_decomposition.enabled = _env_bool(
            'ML_DECOMPOSITION_ENABLED',
            config.ml_decomposition.enabled
        )
        config.ml_decomposition.model_path = _env_str(
            'ML_DECOMPOSITION_MODEL_PATH',
            config.ml_decomposition.model_path
        )

        # Plugin settings
        config.plugin.enabled = _env_bool('PLUGIN_ENABLED', config.plugin.enabled)
        config.plugin.plugin_dir = _env_str('PLUGIN_DIR', config.plugin.plugin_dir)
        config.plugin.sandbox_enabled = _env_bool(
            'PLUGIN_SANDBOX_ENABLED',
            config.plugin.sandbox_enabled
        )

        # General settings
        config.max_gauntlet_rounds = _env_int(
            'MAX_GAUNTLET_ROUNDS',
            config.max_gauntlet_rounds
        )
        config.pass_threshold = _env_float(
            'PASS_THRESHOLD',
            config.pass_threshold
        )
        config.max_decomposition_depth = _env_int(
            'MAX_DECOMPOSITION_DEPTH',
            config.max_decomposition_depth
        )
        config.log_level = _env_str('LOG_LEVEL', config.log_level)
        config.strategy_profile = StrategyProfile(
            _env_str('STRATEGY_PROFILE', config.strategy_profile.value)
        )

        return config

    def apply_profile(self, profile: StrategyProfile) -> 'GauntletConfig':
        """Apply a strategy profile to the configuration"""
        config = copy.deepcopy(self)
        config.strategy_profile = profile

        if profile == StrategyProfile.CONSERVATIVE:
            config.max_gauntlet_rounds = 5
            config.pass_threshold = 0.85
            config.max_decomposition_depth = 5
            config.difficulty.initial_level = DifficultyLevel.EASY

        elif profile == StrategyProfile.BALANCED:
            config.max_gauntlet_rounds = 3
            config.pass_threshold = 0.75
            config.max_decomposition_depth = 3
            config.difficulty.initial_level = DifficultyLevel.MEDIUM

        elif profile == StrategyProfile.AGGRESSIVE:
            config.max_gauntlet_rounds = 2
            config.pass_threshold = 0.65
            config.max_decomposition_depth = 2
            config.difficulty.initial_level = DifficultyLevel.HARD

        elif profile == StrategyProfile.FAST:
            config.max_gauntlet_rounds = 1
            config.pass_threshold = 0.60
            config.max_decomposition_depth = 2
            config.parallel_execution.enabled = True
            config.parallel_execution.max_parallelism = 20
            config.cache.enabled = True

        elif profile == StrategyProfile.THOROUGH:
            config.max_gauntlet_rounds = 5
            config.pass_threshold = 0.90
            config.max_decomposition_depth = 5
            config.fuzzing.enabled = True
            config.circuit_breaker.enabled = True

        return config

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = self._to_dict()

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_from_file(cls: Type[T], filepath: str) -> T:
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls._from_dict(config_dict)

    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def convert_enum(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_enum(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enum(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return {k: convert_enum(v) for k, v in asdict(obj).items()}
            return obj

        return convert_enum(asdict(self))

    @classmethod
    def _from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create configuration from dictionary"""
        def parse_enum(value: Any, enum_type: Type[Enum]):
            if isinstance(value, str):
                return enum_type(value)
            return value

        config = cls()

        # Parse cache config
        if 'cache' in config_dict:
            cache_dict = config_dict['cache']
            config.cache.cache_type = parse_enum(
                cache_dict.get('cache_type', config.cache.cache_type.value),
                CacheType
            )

        # Parse checkpointing config
        if 'checkpointing' in config_dict:
            cp_dict = config_dict['checkpointing']
            config.checkpointing.frequency = parse_enum(
                cp_dict.get('frequency', config.checkpointing.frequency.value),
                CheckpointFrequency
            )

        # Parse circuit breaker config
        if 'circuit_breaker' in config_dict:
            cb_dict = config_dict['circuit_breaker']
            config.circuit_breaker.strategy = parse_enum(
                cb_dict.get('strategy', config.circuit_breaker.strategy.value),
                CircuitBreakerStrategy
            )

        # Parse difficulty config
        if 'difficulty' in config_dict:
            diff_dict = config_dict['difficulty']
            config.difficulty.initial_level = DifficultyLevel(
                diff_dict.get('initial_level', config.difficulty.initial_level.value)
            )

        # Parse strategy profile
        if 'strategy_profile' in config_dict:
            config.strategy_profile = parse_enum(
                config_dict['strategy_profile'],
                StrategyProfile
            )

        # Update all other fields
        for key, value in config_dict.items():
            if hasattr(config, key) and not key.endswith('_config'):
                if key not in ['cache', 'checkpointing', 'circuit_breaker',
                               'difficulty', 'strategy_profile']:
                    setattr(config, key, value)

        return config


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable"""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


def _env_str(key: str, default: Optional[str]) -> Optional[str]:
    """Get string environment variable"""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Get integer environment variable"""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer value for {key}: {value}, using default {default}")
        return default


def _env_float(key: str, default: float) -> float:
    """Get float environment variable"""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float value for {key}: {value}, using default {default}")
        return default


def create_config(
    config_file: Optional[str] = None,
    profile: Optional[StrategyProfile] = None,
    from_env: bool = True
) -> GauntletConfig:
    """
    Create a Gauntlet configuration.

    Args:
        config_file: Optional path to configuration JSON file
        profile: Optional strategy profile to apply
        from_env: Whether to load configuration from environment variables

    Returns:
        GauntletConfig instance
    """
    # Start with environment-based config
    config = GauntletConfig.from_env() if from_env else GauntletConfig()

    # Load from file if provided
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            file_config = GauntletConfig.load_from_file(config_file)
            # Merge configs (file config takes precedence)
            for key, value in file_config.__dict__.items():
                setattr(config, key, value)
            logger.info(f"Loaded configuration from {config_file}")
        else:
            logger.warning(f"Configuration file not found: {config_file}")

    # Apply profile if specified
    if profile:
        config = config.apply_profile(profile)
        logger.info(f"Applied strategy profile: {profile.value}")

    # Validate configuration
    valid, errors = config.validate()
    if not valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Configuration validated successfully")
    return config


# Example usage
async def demo_config():
    """Demonstration of configuration system"""

    print("\n" + "=" * 60)
    print("Gauntlet Configuration System Demo")
    print("=" * 60)

    # Example 1: Default configuration
    print("\n1. Default Configuration:")
    config = GauntletConfig()
    print(f"   Max gauntlet rounds: {config.max_gauntlet_rounds}")
    print(f"   Pass threshold: {config.pass_threshold}")
    print(f"   Cache enabled: {config.cache.enabled}")

    # Example 2: Environment-based configuration
    print("\n2. Environment-based Configuration:")
    os.environ['CACHE_ENABLED'] = 'true'
    os.environ['MAX_GAUNTLET_ROUNDS'] = '5'
    env_config = GauntletConfig.from_env()
    print(f"   Max gauntlet rounds: {env_config.max_gauntlet_rounds}")
    print(f"   Cache enabled: {env_config.cache.enabled}")

    # Example 3: Profile-based configuration
    print("\n3. Profile-based Configuration:")
    conservative_config = config.apply_profile(StrategyProfile.CONSERVATIVE)
    print(f"   Profile: {conservative_config.strategy_profile.value}")
    print(f"   Max gauntlet rounds: {conservative_config.max_gauntlet_rounds}")
    print(f"   Pass threshold: {conservative_config.pass_threshold}")

    # Example 4: Save and load configuration
    print("\n4. Save and Load Configuration:")
    config.save_to_file("./gauntlet_config.json")
    loaded_config = GauntletConfig.load_from_file("./gauntlet_config.json")
    print(f"   Loaded config matches: {loaded_config.max_gauntlet_rounds == config.max_gauntlet_rounds}")

    # Example 5: Validation
    print("\n5. Configuration Validation:")
    valid, errors = config.validate()
    print(f"   Valid: {valid}")
    if not valid:
        print(f"   Errors: {errors}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_config())
