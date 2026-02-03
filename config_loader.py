"""
Unified Configuration Loader

Provides single source of truth for all configuration with proper precedence:
1. Environment variables (highest priority)
2. Configuration files (config.yaml, parameter_settings.json)
3. Default values (lowest priority)

Performs validation, conflict detection, and logging.
"""

import os
import logging
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

from env_helpers import (
    env_var_str,
    env_var_int,
    env_var_float,
    env_var_bool,
    env_var_list,
    env_var_path,
    env_var_url,
    env_var_api_key,
    check_required_env_vars,
    is_production,
    is_development,
    get_env,
    ValidationError,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """LLM generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.95  # Resolved from config conflict
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 4096
    seed: int = 42
    reasoning_effort: str = "medium"


@dataclass
class EvolutionConfig:
    """Evolutionary algorithm parameters."""
    max_iterations: int = 100
    population_size: int = 10
    num_islands: int = 1
    migration_interval: int = 50
    migration_rate: float = 0.1
    archive_size: int = 100
    elite_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    checkpoint_interval: int = 10
    language: str = "python"
    file_suffix: str = ".py"
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: int = 10
    diversity_metric: str = "edit_distance"


@dataclass
class CachingConfig:
    """Caching configuration."""
    enabled: bool = True
    cache_dir: str = "./llm_cache"
    ttl_hours: int = 24
    max_cache_size_mb: int = 100


@dataclass
class ParallelizationConfig:
    """Parallelization configuration."""
    enabled: bool = True
    max_workers: int = 8


@dataclass
class AsyncProcessingConfig:
    """Async processing configuration."""
    enabled: bool = True
    max_concurrent_tasks: int = 100


@dataclass
class MemoryManagementConfig:
    """Memory management configuration."""
    enabled: bool = True
    max_pool_size: int = 100
    memory_threshold_mb: int = 100
    gc_frequency: int = 10


@dataclass
class PerformanceOptimizationConfig:
    """Performance optimization settings."""
    caching: CachingConfig = field(default_factory=CachingConfig)
    parallelization: ParallelizationConfig = field(default_factory=ParallelizationConfig)
    async_processing: AsyncProcessingConfig = field(default_factory=AsyncProcessingConfig)
    memory_management: MemoryManagementConfig = field(default_factory=MemoryManagementConfig)


@dataclass
class RetryConfig:
    """Retry configuration."""
    enabled: bool = True
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    enabled: bool = True
    failure_threshold: int = 5
    timeout: float = 60.0


@dataclass
class RateLimiterConfig:
    """Rate limiter configuration."""
    enabled: bool = True
    max_requests: int = 10
    time_window: float = 60.0


@dataclass
class ReliabilityConfig:
    """Reliability settings."""
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    rate_limiter: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class OpenEvolveConfig:
    """OpenEvolve API configuration."""
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    model_name: str = "gpt-4"
    api_base: str = "https://api.openai.com/v1"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000  # Resolved from config conflict
    debug: bool = False
    workers: int = 1


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: Optional[str] = None
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    enable_encryption: bool = True
    key_encryption_key: Optional[str] = None


@dataclass
class AdaptiveMDAPConfig:
    """Adaptive MDAP/MAKER configuration."""
    enabled: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_dir: str = "./cache/adaptive_mdap"
    
    # Feature weights for complexity classification
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "text_length": 0.15,
        "domain_rarity": 0.20,
        "depth": 0.15,
        "historical_error": 0.20,
        "dependency": 0.10,
        "keyword_complexity": 0.10,
        "constraint_density": 0.10,
    })
    
    # Allocator configuration
    thresholds: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    enable_learning: bool = False
    enable_context_aware: bool = False
    
    # Strategy configurations
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "direct": {"n_agents": 1, "k_ahead": 0, "max_retries": 1, "timeout_ms": 30000},
        "mdap_light": {"n_agents": 3, "k_ahead": 1, "max_retries": 2, "timeout_ms": 60000},
        "mdap_medium": {"n_agents": 5, "k_ahead": 1, "max_retries": 2, "timeout_ms": 90000},
        "maker_full": {"n_agents": 5, "k_ahead": 2, "max_retries": 3, "timeout_ms": 120000},
        "maker_ultra": {"n_agents": 7, "k_ahead": 3, "max_retries": 4, "timeout_ms": 180000},
    })
    
    # Monitoring
    log_all_decisions: bool = True
    track_complexity_scores: bool = True
    compute_savings_metrics: bool = True


@dataclass
class Config:
    """Main configuration container."""
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    performance_optimization: PerformanceOptimizationConfig = field(
        default_factory=PerformanceOptimizationConfig
    )
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    openevolve: OpenEvolveConfig = field(default_factory=OpenEvolveConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    adaptive_mdap: AdaptiveMDAPConfig = field(default_factory=AdaptiveMDAPConfig)

    # Environment
    environment: str = "development"
    log_level: str = "INFO"


class ConfigLoader:
    """
    Unified configuration loader with precedence and validation.

    Precedence order:
    1. Environment variables (highest)
    2. Configuration files (config.yaml, parameter_settings.json)
    3. Default values (lowest)
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files. If None, uses current directory.
        """
        self.config_dir = config_dir or Path.cwd()
        self.config_file = self.config_dir / "config.yaml"
        self.params_file = self.config_dir / "parameter_settings.json"
        self._raw_config: Dict[str, Any] = {}
        self._conflicts: List[str] = []

    def load_all(self) -> Config:
        """
        Load configuration from all sources and apply precedence.

        Returns:
            Validated Config object

        Raises:
            ValidationError: If configuration is invalid
        """
        logger.info("Loading configuration from all sources...")

        # Load from files
        self._load_from_files()

        # Detect conflicts
        self._detect_conflicts()

        # Create config with precedence
        config = self._create_config()

        # Validate final config
        self._validate_config(config)

        # Log configuration source
        self._log_config_source(config)

        return config

    def _load_from_files(self) -> None:
        """Load configuration from YAML and JSON files."""
        # Load config.yaml
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self._raw_config.update(yaml.safe_load(f) or {})
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load {self.config_file}: {e}")
        else:
            logger.warning(f"Configuration file not found: {self.config_file}")

        # Load parameter_settings.json
        if self.params_file.exists():
            try:
                with open(self.params_file, "r") as f:
                    params = json.load(f)
                    # Merge global params
                    if "global" in params:
                        self._raw_config.update(params["global"])
                logger.info(f"Loaded configuration from {self.params_file}")
            except Exception as e:
                logger.warning(f"Failed to load {self.params_file}: {e}")
        else:
            logger.warning(f"Configuration file not found: {self.params_file}")

    def _detect_conflicts(self) -> None:
        """Detect conflicts between configuration sources."""
        # Check for top_p conflict (0.95 in config.yaml vs 1.0 in parameter_settings.json)
        # We're resolving to 0.95 as it's more conservative
        if "top_p" in self._raw_config:
            if self._raw_config["top_p"] == 1.0:
                logger.warning(
                    "Configuration conflict detected: top_p=1.0 in parameter_settings.json "
                    "conflicts with 0.95 in config.yaml. Using 0.95 for safety."
                )
                self._raw_config["top_p"] = 0.95
                self._conflicts.append("top_p")

        # Check for port conflict
        if self._conflicts:
            logger.warning(f"Resolved {len(self._conflicts)} configuration conflict(s)")

    def _create_config(self) -> Config:
        """Create Config object with proper precedence."""
        # Environment (highest priority)
        environment = env_var_str("ENV", default="development").lower()
        log_level = env_var_str("LOG_LEVEL", default="INFO").upper()

        # Server config
        server = ServerConfig(
            host=env_var_str("SERVER_HOST", default=self._raw_config.get("server_host", "0.0.0.0")),
            port=env_var_int("SERVER_PORT", default=self._raw_config.get("port", 8000), min_val=1024, max_val=65535),
            debug=env_var_bool("DEBUG", default=self._raw_config.get("debug", is_development())),
            workers=env_var_int("WORKERS", default=self._raw_config.get("workers", 1), min_val=1, max_val=32),
        )

        # OpenEvolve config
        openevolve = OpenEvolveConfig(
            base_url=env_var_url(
                "OPENEVOLVE_BASE_URL",
                default=self._raw_config.get("openevolve_base_url", "http://localhost:8000"),
                allowed_schemes=["http", "https"]
            ),
            api_key=env_var_api_key(
                "OPENEVOLVE_API_KEY",
                default=self._raw_config.get("openevolve_api_key"),
                provider="OpenEvolve"
            ),
            model_name=env_var_str(
                "MODEL_NAME",
                default=self._raw_config.get("model_name", "gpt-4")
            ),
            api_base=env_var_url(
                "API_BASE",
                default=self._raw_config.get("api_base", "https://api.openai.com/v1"),
                allowed_schemes=["http", "https"]
            ),
        )

        # Generation config
        generation = GenerationConfig(
            temperature=env_var_float(
                "TEMPERATURE",
                default=self._raw_config.get("temperature", 0.7),
                min_val=0.0,
                max_val=2.0
            ),
            top_p=self._raw_config.get("top_p", 0.95),  # Already resolved from conflict
            frequency_penalty=env_var_float(
                "FREQUENCY_PENALTY",
                default=self._raw_config.get("frequency_penalty", 0.0),
                min_val=-2.0,
                max_val=2.0
            ),
            presence_penalty=env_var_float(
                "PRESENCE_PENALTY",
                default=self._raw_config.get("presence_penalty", 0.0),
                min_val=-2.0,
                max_val=2.0
            ),
            max_tokens=env_var_int(
                "MAX_TOKENS",
                default=self._raw_config.get("max_tokens", 4096),
                min_val=1,
                max_val=128000
            ),
            seed=env_var_int(
                "SEED",
                default=self._raw_config.get("seed", 42)
            ),
            reasoning_effort=env_var_str(
                "REASONING_EFFORT",
                default=self._raw_config.get("reasoning_effort", "medium")
            ),
        )

        # Evolution config
        evolution = EvolutionConfig(
            max_iterations=env_var_int(
                "MAX_ITERATIONS",
                default=self._raw_config.get("max_iterations", 100),
                min_val=1
            ),
            population_size=env_var_int(
                "POPULATION_SIZE",
                default=self._raw_config.get("population_size", 10),
                min_val=2
            ),
            num_islands=env_var_int(
                "NUM_ISLANDS",
                default=self._raw_config.get("num_islands", 1),
                min_val=1
            ),
            migration_interval=env_var_int(
                "MIGRATION_INTERVAL",
                default=self._raw_config.get("migration_interval", 50),
                min_val=1
            ),
            migration_rate=env_var_float(
                "MIGRATION_RATE",
                default=self._raw_config.get("migration_rate", 0.1),
                min_val=0.0,
                max_val=1.0
            ),
            archive_size=env_var_int(
                "ARCHIVE_SIZE",
                default=self._raw_config.get("archive_size", 100),
                min_val=1
            ),
            elite_ratio=env_var_float(
                "ELITE_RATIO",
                default=self._raw_config.get("elite_ratio", 0.1),
                min_val=0.0,
                max_val=1.0
            ),
            exploration_ratio=env_var_float(
                "EXPLORATION_RATIO",
                default=self._raw_config.get("exploration_ratio", 0.2),
                min_val=0.0,
                max_val=1.0
            ),
            exploitation_ratio=env_var_float(
                "EXPLOITATION_RATIO",
                default=self._raw_config.get("exploitation_ratio", 0.7),
                min_val=0.0,
                max_val=1.0
            ),
            checkpoint_interval=env_var_int(
                "CHECKPOINT_INTERVAL",
                default=self._raw_config.get("checkpoint_interval", 10),
                min_val=1
            ),
            language=env_var_str(
                "LANGUAGE",
                default=self._raw_config.get("language", "python")
            ),
            file_suffix=env_var_str(
                "FILE_SUFFIX",
                default=self._raw_config.get("file_suffix", ".py")
            ),
            feature_dimensions=self._raw_config.get("feature_dimensions", ["complexity", "diversity"]),
            feature_bins=env_var_int(
                "FEATURE_BINS",
                default=self._raw_config.get("feature_bins", 10),
                min_val=2
            ),
            diversity_metric=env_var_str(
                "DIVERSITY_METRIC",
                default=self._raw_config.get("diversity_metric", "edit_distance")
            ),
        )

        # Performance optimization config
        perf_opts = self._raw_config.get("performance_optimization", {})
        performance_optimization = PerformanceOptimizationConfig(
            caching=CachingConfig(
                enabled=env_var_bool("CACHE_ENABLED", default=perf_opts.get("caching", {}).get("enabled", True)),
                cache_dir=env_var_str("CACHE_DIR", default=perf_opts.get("caching", {}).get("cache_dir", "./llm_cache")),
                ttl_hours=env_var_int("CACHE_TTL_HOURS", default=perf_opts.get("caching", {}).get("ttl_hours", 24), min_val=1),
                max_cache_size_mb=env_var_int(
                    "CACHE_MAX_SIZE_MB",
                    default=perf_opts.get("caching", {}).get("max_cache_size_mb", 100),
                    min_val=1
                ),
            ),
            parallelization=ParallelizationConfig(
                enabled=env_var_bool("PARALLELIZATION_ENABLED", default=perf_opts.get("parallelization", {}).get("enabled", True)),
                max_workers=env_var_int(
                    "MAX_WORKERS",
                    default=perf_opts.get("parallelization", {}).get("max_workers", 8),
                    min_val=1,
                    max_val=128
                ),
            ),
            async_processing=AsyncProcessingConfig(
                enabled=env_var_bool("ASYNC_PROCESSING_ENABLED", default=perf_opts.get("async_processing", {}).get("enabled", True)),
                max_concurrent_tasks=env_var_int(
                    "MAX_CONCURRENT_TASKS",
                    default=perf_opts.get("async_processing", {}).get("max_concurrent_tasks", 100),
                    min_val=1,
                    max_val=1000
                ),
            ),
            memory_management=MemoryManagementConfig(
                enabled=env_var_bool("MEMORY_MANAGEMENT_ENABLED", default=perf_opts.get("memory_management", {}).get("enabled", True)),
                max_pool_size=env_var_int(
                    "MAX_POOL_SIZE",
                    default=perf_opts.get("memory_management", {}).get("max_pool_size", 100),
                    min_val=1
                ),
                memory_threshold_mb=env_var_int(
                    "MEMORY_THRESHOLD_MB",
                    default=perf_opts.get("memory_management", {}).get("memory_threshold_mb", 100),
                    min_val=10
                ),
                gc_frequency=env_var_int(
                    "GC_FREQUENCY",
                    default=perf_opts.get("memory_management", {}).get("gc_frequency", 10),
                    min_val=1
                ),
            ),
        )

        # Reliability config
        reliability_config = self._raw_config.get("reliability", {})
        reliability = ReliabilityConfig(
            retry=RetryConfig(
                enabled=env_var_bool("RETRY_ENABLED", default=reliability_config.get("retry", {}).get("enabled", True)),
                max_attempts=env_var_int(
                    "RETRY_MAX_ATTEMPTS",
                    default=reliability_config.get("retry", {}).get("max_attempts", 3),
                    min_val=1,
                    max_val=10
                ),
                initial_delay=env_var_float(
                    "RETRY_INITIAL_DELAY",
                    default=reliability_config.get("retry", {}).get("initial_delay", 1.0),
                    min_val=0.1
                ),
                max_delay=env_var_float(
                    "RETRY_MAX_DELAY",
                    default=reliability_config.get("retry", {}).get("max_delay", 60.0),
                    min_val=1.0
                ),
                exponential_base=env_var_float(
                    "RETRY_EXPONENTIAL_BASE",
                    default=reliability_config.get("retry", {}).get("exponential_base", 2.0),
                    min_val=1.0,
                    max_val=10.0
                ),
                jitter=env_var_bool("RETRY_JITTER", default=reliability_config.get("retry", {}).get("jitter", True)),
            ),
            circuit_breaker=CircuitBreakerConfig(
                enabled=env_var_bool(
                    "CIRCUIT_BREAKER_ENABLED",
                    default=reliability_config.get("circuit_breaker", {}).get("enabled", True)
                ),
                failure_threshold=env_var_int(
                    "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
                    default=reliability_config.get("circuit_breaker", {}).get("failure_threshold", 5),
                    min_val=1
                ),
                timeout=env_var_float(
                    "CIRCUIT_BREAKER_TIMEOUT",
                    default=reliability_config.get("circuit_breaker", {}).get("timeout", 60.0),
                    min_val=1.0
                ),
            ),
            rate_limiter=RateLimiterConfig(
                enabled=env_var_bool(
                    "RATE_LIMITER_ENABLED",
                    default=reliability_config.get("rate_limiter", {}).get("enabled", True)
                ),
                max_requests=env_var_int(
                    "RATE_LIMITER_MAX_REQUESTS",
                    default=reliability_config.get("rate_limiter", {}).get("max_requests", 10),
                    min_val=1
                ),
                time_window=env_var_float(
                    "RATE_LIMITER_TIME_WINDOW",
                    default=reliability_config.get("rate_limiter", {}).get("time_window", 60.0),
                    min_val=1.0
                ),
            ),
        )

        # Security config
        security = SecurityConfig(
            secret_key=env_var_str("SECRET_KEY"),
            algorithm=env_var_str("JWT_ALGORITHM", default="HS256"),
            access_token_expire_minutes=env_var_int(
                "ACCESS_TOKEN_EXPIRE_MINUTES",
                default=30,
                min_val=1
            ),
            enable_encryption=env_var_bool("ENABLE_ENCRYPTION", default=True),
            key_encryption_key=env_var_str("KEY_ENCRYPTION_KEY"),
        )

        # Adaptive MDAP config
        adaptive_mdap_raw = self._raw_config.get("adaptive_mdap", {})
        adaptive_mdap = AdaptiveMDAPConfig(
            enabled=env_var_bool("ADAPTIVE_MDAP_ENABLED", default=adaptive_mdap_raw.get("enabled", True)),
            embedding_model=env_var_str(
                "ADAPTIVE_MDAP_EMBEDDING_MODEL",
                default=adaptive_mdap_raw.get("embedding_model", "all-MiniLM-L6-v2")
            ),
            cache_dir=env_var_str(
                "ADAPTIVE_MDAP_CACHE_DIR",
                default=adaptive_mdap_raw.get("cache_dir", "./cache/adaptive_mdap")
            ),
            feature_weights=adaptive_mdap_raw.get("feature_weights", {
                "text_length": 0.15,
                "domain_rarity": 0.20,
                "depth": 0.15,
                "historical_error": 0.20,
                "dependency": 0.10,
                "keyword_complexity": 0.10,
                "constraint_density": 0.10,
            }),
            thresholds=adaptive_mdap_raw.get("thresholds", [0.2, 0.4, 0.6, 0.8]),
            enable_learning=env_var_bool(
                "ADAPTIVE_MDAP_ENABLE_LEARNING",
                default=adaptive_mdap_raw.get("enable_learning", False)
            ),
            enable_context_aware=env_var_bool(
                "ADAPTIVE_MDAP_ENABLE_CONTEXT_AWARE",
                default=adaptive_mdap_raw.get("enable_context_aware", False)
            ),
            strategy_configs=adaptive_mdap_raw.get("strategy_configs", {
                "direct": {"n_agents": 1, "k_ahead": 0, "max_retries": 1, "timeout_ms": 30000},
                "mdap_light": {"n_agents": 3, "k_ahead": 1, "max_retries": 2, "timeout_ms": 60000},
                "mdap_medium": {"n_agents": 5, "k_ahead": 1, "max_retries": 2, "timeout_ms": 90000},
                "maker_full": {"n_agents": 5, "k_ahead": 2, "max_retries": 3, "timeout_ms": 120000},
                "maker_ultra": {"n_agents": 7, "k_ahead": 3, "max_retries": 4, "timeout_ms": 180000},
            }),
            log_all_decisions=adaptive_mdap_raw.get("log_all_decisions", True),
            track_complexity_scores=adaptive_mdap_raw.get("track_complexity_scores", True),
            compute_savings_metrics=adaptive_mdap_raw.get("compute_savings_metrics", True),
        )

        return Config(
            generation=generation,
            evolution=evolution,
            performance_optimization=performance_optimization,
            reliability=reliability,
            openevolve=openevolve,
            server=server,
            security=security,
            adaptive_mdap=adaptive_mdap,
            environment=environment,
            log_level=log_level,
        )

    def _validate_config(self, config: Config) -> None:
        """Validate final configuration."""
        logger.info("Validating configuration...")

        # Check for suspicious combinations
        if config.generation.top_p > 0.99 and config.generation.temperature < 0.1:
            logger.warning(
                "Unusual configuration: very high top_p with very low temperature "
                "may result in deterministic but unusual outputs"
            )

        # Check ratio constraints
        total_ratio = (
            config.evolution.elite_ratio
            + config.evolution.exploration_ratio
            + config.evolution.exploitation_ratio
        )
        if abs(total_ratio - 1.0) > 0.1:
            logger.warning(
                f"Evolution ratios don't sum to 1.0 (got {total_ratio:.2f}). "
                f"This may affect algorithm behavior."
            )

        # Validate production settings
        if is_production():
            if config.server.debug:
                logger.error("DEBUG mode should not be enabled in production")
                raise ValidationError("DEBUG mode must be disabled in production")

            if not config.security.secret_key:
                logger.error("SECRET_KEY must be set in production")
                raise ValidationError("SECRET_KEY environment variable must be set in production")

            if config.security.secret_key and len(config.security.secret_key) < 32:
                logger.error("SECRET_KEY must be at least 32 characters in production")
                raise ValidationError("SECRET_KEY must be at least 32 characters in production")

        # Check API keys in production
        if is_production() and not config.openevolve.api_key:
            logger.error("OPENEVOLVE_API_KEY must be set in production")
            raise ValidationError("OPENEVOLVE_API_KEY must be set in production")

        logger.info("Configuration validation passed")

    def _log_config_source(self, config: Config) -> None:
        """Log which configuration sources were used."""
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Log level: {config.log_level}")
        logger.info(f"Server: {config.server.host}:{config.server.port}")
        logger.info(f"Debug mode: {config.server.debug}")
        logger.info(f"Workers: {config.server.workers}")

        if config.openevolve.api_key:
            logger.info(f"OpenEvolve API key: Set (ending with ...{config.openevolve.api_key[-4:]})")
        else:
            logger.warning("OpenEvolve API key: Not set - some features may not work")
        
        # Log Adaptive MDAP configuration
        logger.info(f"Adaptive MDAP enabled: {config.adaptive_mdap.enabled}")
        if config.adaptive_mdap.enabled:
            logger.info(f"Adaptive MDAP thresholds: {config.adaptive_mdap.thresholds}")
            logger.info(f"Adaptive MDAP learning: {config.adaptive_mdap.enable_learning}")
            logger.info(f"Adaptive MDAP context-aware: {config.adaptive_mdap.enable_context_aware}")


# Global config instance
_config: Optional[Config] = None
_config_loader: Optional[ConfigLoader] = None


def load_config(config_dir: Optional[Path] = None, force_reload: bool = False) -> Config:
    """
    Load configuration from all sources.

    Args:
        config_dir: Directory containing config files
        force_reload: Force reload even if already loaded

    Returns:
        Validated Config object
    """
    global _config, _config_loader

    if _config is None or force_reload:
        _config_loader = ConfigLoader(config_dir)
        _config = _config_loader.load_all()

    return _config


def get_config() -> Config:
    """
    Get the currently loaded configuration.

    Returns:
        Config object (loads if not already loaded)

    Raises:
        ValidationError: If configuration is invalid
    """
    global _config

    if _config is None:
        return load_config()

    return _config


def reload_config() -> Config:
    """
    Force reload configuration from all sources.

    Returns:
        Reloaded Config object
    """
    return load_config(force_reload=True)


def get_config_summary() -> Dict[str, Any]:
    """
    Get a summary of the current configuration (safe for logging).

    Returns:
        Dictionary with non-sensitive configuration info
    """
    config = get_config()

    return {
        "environment": config.environment,
        "log_level": config.log_level,
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "debug": config.server.debug,
            "workers": config.server.workers,
        },
        "openevolve": {
            "base_url": config.openevolve.base_url,
            "model_name": config.openevolve.model_name,
            "api_key_set": bool(config.openevolve.api_key),
        },
        "generation": {
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "max_tokens": config.generation.max_tokens,
        },
        "performance": {
            "caching_enabled": config.performance_optimization.caching.enabled,
            "parallelization_enabled": config.performance_optimization.parallelization.enabled,
            "async_processing_enabled": config.performance_optimization.async_processing.enabled,
        },
    }
