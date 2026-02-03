"""
Configuration management for Graphiti integration.

Following CLAUDE.md:
- LAW OF CONFIGURATION EXPLICITNESS: All config via environment variables
- Crash if missing required configuration
- No magic defaults
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class GraphitiConfig:
    """
    Graphiti integration configuration.

    All configuration values come from environment variables.
    Required values will cause startup failure if missing.
    """

    # Required: Graph Database Configuration
    graphiti_provider: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_PROVIDER", "neo4j")
    )
    graphiti_uri: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_URI", "")
    )
    graphiti_user: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_USER", "")
    )
    graphiti_password: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_PASSWORD", "")
    )
    graphiti_database: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_DATABASE", "neo4j")
    )

    # Required: LLM Configuration
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    llm_provider: str = field(
        default_factory=lambda: os.environ.get("LLM_PROVIDER", "openai")
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )

    # Optional: Advanced Configuration
    max_episode_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("GRAPHITI_MAX_EPISODE_RETRIES", "3")
        )
    )
    episode_timeout_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("GRAPHITI_EPISODE_TIMEOUT_MS", "30000")
        )
    )
    search_timeout_ms: int = field(
        default_factory=lambda: int(os.environ.get("GRAPHITI_SEARCH_TIMEOUT_MS", "5000"))
    )
    max_concurrent_episodes: int = field(
        default_factory=lambda: int(
            os.environ.get("GRAPHITI_MAX_CONCURRENT_EPISODES", "10")
        )
    )

    # Contradiction Detection Configuration
    contradiction_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "GRAPHITI_CONTRADICTION_ENABLED", "true"
        ).lower()
        == "true"
    )
    contradiction_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("GRAPHITI_CONTRADICTION_THRESHOLD", "0.7")
        )
    )

    # Agent Memory Configuration
    agent_memory_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "GRAPHITI_AGENT_MEMORY_ENABLED", "true"
        ).lower()
        == "true"
    )
    memory_summarization_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get("GRAPHITI_MEMORY_SUMMARIZATION_THRESHOLD", "100")
        )
    )

    # Incremental Update Configuration
    incremental_updates_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "GRAPHITI_INCREMENTAL_UPDATES_ENABLED", "true"
        ).lower()
        == "true"
    )
    entity_merge_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("GRAPHITI_ENTITY_MERGE_THRESHOLD", "0.85")
        )
    )

    # Monitoring Configuration
    telemetry_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "GRAPHITI_TELEMETRY_ENABLED", "false"
        ).lower()
        == "true"
    )
    metrics_collection_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "GRAPHITI_METRICS_ENABLED", "true"
        ).lower()
        == "true"
    )

    # Index Configuration
    entity_index_name: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_ENTITY_INDEX", "entities")
    )
    episode_index_name: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_EPISODE_INDEX", "episodes")
    )
    community_index_name: str = field(
        default_factory=lambda: os.environ.get("GRAPHITI_COMMUNITY_INDEX", "communities")
    )

    def validate(self) -> None:
        """
        Validate configuration.

        Raises:
            ConfigurationError: If required configuration is missing
        """
        errors = []
        missing_keys = []

        # Required graph database configuration
        if not self.graphiti_uri:
            missing_keys.append("GRAPHITI_URI")
            errors.append("Graph database URI is required")

        if not self.graphiti_user:
            missing_keys.append("GRAPHITI_USER")
            errors.append("Graph database user is required")

        if not self.graphiti_password:
            missing_keys.append("GRAPHITI_PASSWORD")
            errors.append("Graph database password is required")

        # Required LLM configuration
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
            errors.append("OpenAI API key is required")

        # Validate numeric ranges
        if self.max_episode_retries < 0:
            errors.append("max_episode_retries must be >= 0")

        if self.episode_timeout_ms < 1000:
            errors.append("episode_timeout_ms must be >= 1000")

        if self.search_timeout_ms < 500:
            errors.append("search_timeout_ms must be >= 500")

        if self.max_concurrent_episodes < 1:
            errors.append("max_concurrent_episodes must be >= 1")

        if not 0.0 <= self.contradiction_threshold <= 1.0:
            errors.append("contradiction_threshold must be between 0.0 and 1.0")

        if not 0.0 <= self.entity_merge_threshold <= 1.0:
            errors.append("entity_merge_threshold must be between 0.0 and 1.0")

        if self.memory_summarization_threshold < 10:
            errors.append("memory_summarization_threshold must be >= 10")

        # Validate provider
        valid_providers = ["neo4j", "falkordb", "kuzu", "neptune"]
        if self.graphiti_provider not in valid_providers:
            errors.append(
                f"Invalid provider: {self.graphiti_provider}. Must be one of {valid_providers}"
            )

        if errors:
            raise ConfigurationError(
                message=f"Configuration validation failed: {'; '.join(errors)}",
                missing_keys=missing_keys,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary (sans sensitive data).

        Returns:
            Configuration dictionary with passwords masked
        """
        config_dict = {
            "graphiti_provider": self.graphiti_provider,
            "graphiti_uri": self.graphiti_uri,
            "graphiti_user": self.graphiti_user,
            "graphiti_password": "***REDACTED***",
            "graphiti_database": self.graphiti_database,
            "openai_api_key": "***REDACTED***",
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "max_episode_retries": self.max_episode_retries,
            "episode_timeout_ms": self.episode_timeout_ms,
            "search_timeout_ms": self.search_timeout_ms,
            "max_concurrent_episodes": self.max_concurrent_episodes,
            "contradiction_enabled": self.contradiction_enabled,
            "contradiction_threshold": self.contradiction_threshold,
            "agent_memory_enabled": self.agent_memory_enabled,
            "memory_summarization_threshold": self.memory_summarization_threshold,
            "incremental_updates_enabled": self.incremental_updates_enabled,
            "entity_merge_threshold": self.entity_merge_threshold,
            "telemetry_enabled": self.telemetry_enabled,
            "metrics_collection_enabled": self.metrics_collection_enabled,
            "entity_index_name": self.entity_index_name,
            "episode_index_name": self.episode_index_name,
            "community_index_name": self.community_index_name,
        }
        return config_dict

    @classmethod
    def from_file(cls, config_path: str) -> "GraphitiConfig":
        """
        Load configuration from environment variables with optional override file.

        Args:
            config_path: Path to optional YAML configuration file

        Returns:
            Validated GraphitiConfig instance

        Note:
            Environment variables take precedence over file configuration
        """
        # Environment variables are already loaded by dataclass defaults
        config = cls()

        # Optional: Load from file for development (not for production)
        if config_path and Path(config_path).exists():
            import yaml

            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)

            # Apply file config only where environment vars are not set
            for key, value in file_config.items():
                env_key = key.upper()
                if env_key not in os.environ:
                    setattr(config, key, value)

        config.validate()
        return config


def validate_config(config: Optional[GraphitiConfig] = None) -> GraphitiConfig:
    """
    Validate and return configuration.

    Args:
        config: Optional configuration instance. If None, creates from environment.

    Returns:
        Validated GraphitiConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
    """
    if config is None:
        config = GraphitiConfig()

    config.validate()
    return config
