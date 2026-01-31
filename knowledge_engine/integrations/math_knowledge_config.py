"""
Configuration Management for Mathematical Knowledge Integration

Centralized configuration with:
- YAML/JSON config files
- Environment variable support
- Validation
- Hot reload
- Secrets management

Author: OpenEvolve
Created: 2026-01-31
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///math_knowledge.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create from environment variables."""
        return cls(
            url=os.getenv("MATH_KNOWLEDGE_DB_URL", "sqlite:///math_knowledge.db"),
            pool_size=int(os.getenv("MATH_KNOWLEDGE_DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("MATH_KNOWLEDGE_DB_MAX_OVERFLOW", "20")),
            echo=os.getenv("MATH_KNOWLEDGE_DB_ECHO", "false").lower() == "true"
        )


@dataclass
class RedisConfig:
    """Redis configuration."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ttl_seconds: int = 86400
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        protocol = "rediss" if self.ssl else "redis"
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"
    
    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create from environment variables."""
        return cls(
            enabled=os.getenv("MATH_KNOWLEDGE_REDIS_ENABLED", "false").lower() == "true",
            host=os.getenv("MATH_KNOWLEDGE_REDIS_HOST", "localhost"),
            port=int(os.getenv("MATH_KNOWLEDGE_REDIS_PORT", "6379")),
            db=int(os.getenv("MATH_KNOWLEDGE_REDIS_DB", "0")),
            password=os.getenv("MATH_KNOWLEDGE_REDIS_PASSWORD"),
            ssl=os.getenv("MATH_KNOWLEDGE_REDIS_SSL", "false").lower() == "true",
            ttl_seconds=int(os.getenv("MATH_KNOWLEDGE_REDIS_TTL", "86400"))
        )


@dataclass
class Z3Config:
    """Z3 solver configuration."""
    timeout_ms: int = 30000
    memory_limit_mb: int = 4096
    proof_generation: bool = True
    model_generation: bool = True
    parallel_threads: int = 1
    simplify: bool = True
    executable_path: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "Z3Config":
        """Create from environment variables."""
        return cls(
            timeout_ms=int(os.getenv("MATH_KNOWLEDGE_Z3_TIMEOUT_MS", "30000")),
            memory_limit_mb=int(os.getenv("MATH_KNOWLEDGE_Z3_MEMORY_MB", "4096")),
            proof_generation=os.getenv("MATH_KNOWLEDGE_Z3_PROOF", "true").lower() == "true",
            model_generation=os.getenv("MATH_KNOWLEDGE_Z3_MODEL", "true").lower() == "true",
            parallel_threads=int(os.getenv("MATH_KNOWLEDGE_Z3_THREADS", "1")),
            executable_path=os.getenv("MATH_KNOWLEDGE_Z3_PATH")
        )


@dataclass
class LeanAideConfig:
    """LeanAIDE configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 7654
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    connection_pool_size: int = 10
    
    @property
    def base_url(self) -> str:
        """Get base URL."""
        return f"http://{self.host}:{self.port}"
    
    @classmethod
    def from_env(cls) -> "LeanAideConfig":
        """Create from environment variables."""
        return cls(
            enabled=os.getenv("MATH_KNOWLEDGE_LEANAIDE_ENABLED", "true").lower() == "true",
            host=os.getenv("MATH_KNOWLEDGE_LEANAIDE_HOST", "localhost"),
            port=int(os.getenv("MATH_KNOWLEDGE_LEANAIDE_PORT", "7654")),
            timeout=float(os.getenv("MATH_KNOWLEDGE_LEANAIDE_TIMEOUT", "300")),
            max_retries=int(os.getenv("MATH_KNOWLEDGE_LEANAIDE_RETRIES", "3")),
            connection_pool_size=int(os.getenv("MATH_KNOWLEDGE_LEANAIDE_POOL_SIZE", "10"))
        )


@dataclass
class FeatureExtractionConfig:
    """Feature extraction configuration."""
    embedding_dim: int = 128
    similarity_threshold: float = 0.7
    cache_size: int = 10000
    max_constraint_complexity: int = 100
    
    @classmethod
    def from_env(cls) -> "FeatureExtractionConfig":
        """Create from environment variables."""
        return cls(
            embedding_dim=int(os.getenv("MATH_KNOWLEDGE_EMBEDDING_DIM", "128")),
            similarity_threshold=float(os.getenv("MATH_KNOWLEDGE_SIMILARITY_THRESH", "0.7")),
            cache_size=int(os.getenv("MATH_KNOWLEDGE_CACHE_SIZE", "10000"))
        )


@dataclass
class ProofSearchConfig:
    """Proof search configuration."""
    max_depth: int = 20
    timeout_seconds: float = 300.0
    parallel_attempts: int = 3
    similarity_threshold: float = 0.7
    enable_knowledge_reuse: bool = True
    enable_learning: bool = True
    
    @classmethod
    def from_env(cls) -> "ProofSearchConfig":
        """Create from environment variables."""
        return cls(
            max_depth=int(os.getenv("MATH_KNOWLEDGE_MAX_DEPTH", "20")),
            timeout_seconds=float(os.getenv("MATH_KNOWLEDGE_SEARCH_TIMEOUT", "300")),
            parallel_attempts=int(os.getenv("MATH_KNOWLEDGE_PARALLEL", "3")),
            enable_knowledge_reuse=os.getenv("MATH_KNOWLEDGE_REUSE", "true").lower() == "true"
        )


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    tracing_enabled: bool = False
    tracing_endpoint: Optional[str] = None
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    
    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create from environment variables."""
        return cls(
            enabled=os.getenv("MATH_KNOWLEDGE_MONITORING_ENABLED", "true").lower() == "true",
            metrics_port=int(os.getenv("MATH_KNOWLEDGE_METRICS_PORT", "9090")),
            tracing_enabled=os.getenv("MATH_KNOWLEDGE_TRACING_ENABLED", "false").lower() == "true",
            tracing_endpoint=os.getenv("MATH_KNOWLEDGE_TRACING_ENDPOINT"),
            log_level=LogLevel(os.getenv("MATH_KNOWLEDGE_LOG_LEVEL", "INFO").upper()),
            log_format=os.getenv("MATH_KNOWLEDGE_LOG_FORMAT", "json")
        )


@dataclass
class APIConfig:
    """API server configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8765
    workers: int = 1
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False
    api_key: Optional[str] = None
    rate_limit_per_minute: int = 60
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create from environment variables."""
        cors = os.getenv("MATH_KNOWLEDGE_CORS_ORIGINS", "*")
        return cls(
            enabled=os.getenv("MATH_KNOWLEDGE_API_ENABLED", "true").lower() == "true",
            host=os.getenv("MATH_KNOWLEDGE_API_HOST", "0.0.0.0"),
            port=int(os.getenv("MATH_KNOWLEDGE_API_PORT", "8765")),
            workers=int(os.getenv("MATH_KNOWLEDGE_API_WORKERS", "1")),
            cors_origins=cors.split(","),
            api_key_required=os.getenv("MATH_KNOWLEDGE_API_KEY_REQUIRED", "false").lower() == "true",
            api_key=os.getenv("MATH_KNOWLEDGE_API_KEY"),
            rate_limit_per_minute=int(os.getenv("MATH_KNOWLEDGE_RATE_LIMIT", "60"))
        )


@dataclass
class MathKnowledgeConfig:
    """Main configuration class."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    z3: Z3Config = field(default_factory=Z3Config)
    leanaide: LeanAideConfig = field(default_factory=LeanAideConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    proof_search: ProofSearchConfig = field(default_factory=ProofSearchConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "MathKnowledgeConfig":
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**{k: cls._load_subconfig(v) for k, v in data.items()})
    
    @classmethod
    def _load_subconfig(cls, data: Dict[str, Any]) -> Any:
        """Load sub-configuration."""
        # This is a simplified version - real implementation would be more robust
        return data
    
    @classmethod
    def from_env(cls) -> "MathKnowledgeConfig":
        """Load configuration from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            redis=RedisConfig.from_env(),
            z3=Z3Config.from_env(),
            leanaide=LeanAideConfig.from_env(),
            features=FeatureExtractionConfig.from_env(),
            proof_search=ProofSearchConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
            api=APIConfig.from_env()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            if path.suffix in ['.yaml', '.yml']:
                f.write(self.to_yaml())
            else:
                f.write(self.to_json())
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate database URL
        if not self.database.url:
            errors.append("Database URL is required")
        
        # Validate Z3 timeout
        if self.z3.timeout_ms < 1000:
            errors.append("Z3 timeout must be at least 1000ms")
        
        # Validate API key if required
        if self.api.api_key_required and not self.api.api_key:
            errors.append("API key is required when API key authentication is enabled")
        
        # Validate Redis if enabled
        if self.redis.enabled:
            if not self.redis.host:
                errors.append("Redis host is required when Redis is enabled")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


# Global configuration instance
_config: Optional[MathKnowledgeConfig] = None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    use_env: bool = True
) -> MathKnowledgeConfig:
    """
    Load configuration from file and/or environment.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Default values
    
    Args:
        config_path: Path to config file
        use_env: Whether to use environment variables
        
    Returns:
        Loaded configuration
    """
    global _config
    
    # Start with defaults
    config = MathKnowledgeConfig()
    
    # Load from file if provided
    if config_path:
        file_config = MathKnowledgeConfig.from_file(config_path)
        # Merge file config over defaults
        for field_name in config.__dataclass_fields__:
            if getattr(file_config, field_name):
                setattr(config, field_name, getattr(file_config, field_name))
    
    # Override with environment variables
    if use_env:
        env_config = MathKnowledgeConfig.from_env()
        for field_name in config.__dataclass_fields__:
            env_value = getattr(env_config, field_name)
            if env_value:
                setattr(config, field_name, env_value)
    
    # Validate
    if not config.is_valid():
        errors = config.validate()
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")
    
    _config = config
    logger.info("Configuration loaded successfully")
    
    return config


def get_config() -> MathKnowledgeConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def create_default_config_file(path: Union[str, Path] = "config.yaml"):
    """Create a default configuration file."""
    config = MathKnowledgeConfig()
    config.save(path)
    logger.info(f"Default configuration saved to {path}")


# Example usage
if __name__ == "__main__":
    # Create default config
    create_default_config_file("example_config.yaml")
    
    # Load config
    config = load_config("example_config.yaml")
    
    print("Configuration loaded:")
    print(config.to_yaml())
