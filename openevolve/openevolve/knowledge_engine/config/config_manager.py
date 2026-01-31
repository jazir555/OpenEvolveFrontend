"""
OpenEvolve Knowledge Engine - Production Configuration

This module provides the core configuration and infrastructure for a production-ready
knowledge engine system with proper data storage, caching, and server configuration.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
import asyncio
import aiofiles
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for the knowledge engine."""
    # Recommended: postgresql (PostgreSQL License - permissive)
    # Recommended: memgraph (Apache 2.0 - permissive, for graph operations)
    # Legacy (not recommended): mongodb (SSPL - copyleft), neo4j (GPL - copyleft)
    type: str = "postgresql"
    host: str = "localhost"
    port: int = 5432  # 5432 for PostgreSQL, 7687 for Memgraph
    username: str = "openevolve"
    password: str = ""
    database: str = "openevolve_kg"
    connection_pool_size: int = 20
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Vector store configuration for embeddings and similarity search."""
    type: str = "qdrant"  # qdrant, chroma, weaviate, pgvector
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge_artifacts"
    distance_metric: str = "cosine"  # cosine, euclidean, dot
    vector_size: int = 1536  # Default for OpenAI embeddings
    recreate_collection: bool = False


@dataclass
class CacheConfig:
    """Cache configuration for performance optimization."""
    type: str = "redis"  # redis, memory, memcached
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl_seconds: int = 3600  # 1 hour default TTL
    max_items: int = 10000


@dataclass
class ServerConfig:
    """Server configuration for the knowledge engine."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 300  # 5 minutes
    max_connections: int = 1000
    cors_enabled: bool = True
    cors_origins: List[str] = None
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]


@dataclass
class LLMConfig:
    """LLM configuration for AI components."""
    provider: str = "openai"  # openai, anthropic, huggingface, azure, custom
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    request_timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class KnowledgeEngineConfig:
    """Main configuration for the knowledge engine."""
    # Core settings
    name: str = "OpenEvolve Knowledge Engine"
    version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    
    # Component configurations
    database: DatabaseConfig = None
    vector_store: VectorStoreConfig = None
    cache: CacheConfig = None
    server: ServerConfig = None
    llm: LLMConfig = None
    
    # Feature flags
    enable_temporal_graphs: bool = True
    enable_bilingual_extraction: bool = True
    enable_multi_agent_collaboration: bool = True
    enable_formal_verification: bool = True
    enable_retrieval_augmentation: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_queue_size: int = 1000
    response_cache_ttl: int = 300  # 5 minutes
    enable_request_logging: bool = True
    enable_response_caching: bool = True
    
    # Security settings
    enable_authentication: bool = True
    enable_authorization: bool = True
    jwt_secret: str = ""
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Monitoring and logging
    log_level: str = "INFO"
    log_format: str = "json"  # json, text
    enable_metrics: bool = True
    metrics_export_port: int = 9090
    enable_tracing: bool = False
    tracing_endpoint: Optional[str] = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.server is None:
            self.server = ServerConfig()
        if self.llm is None:
            self.llm = LLMConfig()


class ConfigManager:
    """
    Configuration manager for the knowledge engine.
    
    Handles loading, validating, and managing configuration for all components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Validate configuration
        self._validate_config()
        
        logger.info({
            "msg": "Configuration manager initialized",
            "config_path": config_path,
            "environment": self.config.environment,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _load_config(self) -> KnowledgeEngineConfig:
        """Load configuration from file or environment variables."""
        # First try to load from file
        if self.config_path and Path(self.config_path).exists():
            return self._load_from_file(self.config_path)
        
        # Then try to load from environment variables
        return self._load_from_environment()
    
    def _load_from_file(self, config_path: str) -> KnowledgeEngineConfig:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    data = json.load(f)
                elif config_path.endswith(('.yml', '.yaml')):
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
            
            # Convert to KnowledgeEngineConfig
            return self._dict_to_config(data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Fall back to environment variables
            return self._load_from_environment()
    
    def _load_from_environment(self) -> KnowledgeEngineConfig:
        """Load configuration from environment variables."""
        # Build config from environment variables
        config_data = {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "database": {
                "type": os.getenv("DB_TYPE", "postgresql"),
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "username": os.getenv("DB_USERNAME", "openevolve"),
                "password": os.getenv("DB_PASSWORD", ""),
                "database": os.getenv("DB_NAME", "openevolve_kg"),
                "ssl_enabled": os.getenv("DB_SSL_ENABLED", "false").lower() == "true"
            },
            "vector_store": {
                "type": os.getenv("VECTOR_STORE_TYPE", "qdrant"),
                "host": os.getenv("VECTOR_STORE_HOST", "localhost"),
                "port": int(os.getenv("VECTOR_STORE_PORT", "6333")),
                "collection_name": os.getenv("VECTOR_STORE_COLLECTION", "knowledge_artifacts"),
                "distance_metric": os.getenv("VECTOR_STORE_DISTANCE", "cosine"),
                "vector_size": int(os.getenv("VECTOR_STORE_SIZE", "1536"))
            },
            "cache": {
                "type": os.getenv("CACHE_TYPE", "redis"),
                "host": os.getenv("CACHE_HOST", "localhost"),
                "port": int(os.getenv("CACHE_PORT", "6379")),
                "db": int(os.getenv("CACHE_DB", "0")),
                "ttl_seconds": int(os.getenv("CACHE_TTL", "3600")),
                "max_items": int(os.getenv("CACHE_MAX_ITEMS", "10000"))
            },
            "server": {
                "host": os.getenv("SERVER_HOST", "0.0.0.0"),
                "port": int(os.getenv("SERVER_PORT", "8000")),
                "workers": int(os.getenv("SERVER_WORKERS", "4")),
                "timeout": int(os.getenv("SERVER_TIMEOUT", "300")),
                "max_connections": int(os.getenv("MAX_CONNECTIONS", "1000")),
                "cors_enabled": os.getenv("CORS_ENABLED", "true").lower() == "true",
                "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
                "ssl_enabled": os.getenv("SSL_ENABLED", "false").lower() == "true"
            },
            "llm": {
                "provider": os.getenv("LLM_PROVIDER", "openai"),
                "model": os.getenv("LLM_MODEL", "gpt-4o"),
                "api_key": os.getenv("LLM_API_KEY", ""),
                "base_url": os.getenv("LLM_BASE_URL"),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
                "top_p": float(os.getenv("LLM_TOP_P", "1.0")),
                "request_timeout": int(os.getenv("LLM_TIMEOUT", "120")),
                "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3"))
            },
            "max_concurrent_requests": int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
            "enable_authentication": os.getenv("ENABLE_AUTH", "true").lower() == "true",
            "jwt_secret": os.getenv("JWT_SECRET", ""),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true"
        }
        
        return self._dict_to_config(config_data)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> KnowledgeEngineConfig:
        """Convert dictionary to KnowledgeEngineConfig."""
        # Create nested config objects
        db_config = DatabaseConfig(**data.get("database", {}))
        vector_config = VectorStoreConfig(**data.get("vector_store", {}))
        cache_config = CacheConfig(**data.get("cache", {}))
        server_config = ServerConfig(**data.get("server", {}))
        llm_config = LLMConfig(**data.get("llm", {}))
        
        # Create main config
        config = KnowledgeEngineConfig(
            name=data.get("name", "OpenEvolve Knowledge Engine"),
            version=data.get("version", "1.0.0"),
            environment=data.get("environment", "development"),
            database=db_config,
            vector_store=vector_config,
            cache=cache_config,
            server=server_config,
            llm=llm_config,
            enable_temporal_graphs=data.get("enable_temporal_graphs", True),
            enable_bilingual_extraction=data.get("enable_bilingual_extraction", True),
            enable_multi_agent_collaboration=data.get("enable_multi_agent_collaboration", True),
            enable_formal_verification=data.get("enable_formal_verification", True),
            enable_retrieval_augmentation=data.get("enable_retrieval_augmentation", True),
            max_concurrent_requests=data.get("max_concurrent_requests", 100),
            request_queue_size=data.get("request_queue_size", 1000),
            response_cache_ttl=data.get("response_cache_ttl", 300),
            enable_request_logging=data.get("enable_request_logging", True),
            enable_response_caching=data.get("enable_response_caching", True),
            enable_authentication=data.get("enable_authentication", True),
            enable_authorization=data.get("enable_authorization", True),
            jwt_secret=data.get("jwt_secret", ""),
            rate_limit_requests=data.get("rate_limit_requests", 100),
            rate_limit_window=data.get("rate_limit_window", 60),
            log_level=data.get("log_level", "INFO"),
            log_format=data.get("log_format", "json"),
            enable_metrics=data.get("enable_metrics", True),
            metrics_export_port=data.get("metrics_export_port", 9090),
            enable_tracing=data.get("enable_tracing", False),
            tracing_endpoint=data.get("tracing_endpoint")
        )
        
        return config
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        errors = []
        
        # Validate database config
        if self.config.database.port <= 0 or self.config.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")
        
        # Validate vector store config
        if self.config.vector_store.port <= 0 or self.config.vector_store.port > 65535:
            errors.append("Vector store port must be between 1 and 65535")
        
        # Validate server config
        if self.config.server.port <= 0 or self.config.server.port > 65535:
            errors.append("Server port must be between 1 and 65535")
        
        if self.config.server.workers <= 0:
            errors.append("Server workers must be greater than 0")
        
        # Validate LLM config
        if not self.config.llm.api_key and self.config.llm.provider in ["openai", "anthropic"]:
            errors.append(f"API key required for {self.config.llm.provider} provider")
        
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"Configuration validation failed: {error_msg}")
            raise ValueError(f"Configuration validation failed: {error_msg}")
        
        logger.info("Configuration validation passed")
    
    async def save_config(self, config_path: str):
        """Save current configuration to file."""
        config_dict = {
            "name": self.config.name,
            "version": self.config.version,
            "environment": self.config.environment,
            "database": asdict(self.config.database),
            "vector_store": asdict(self.config.vector_store),
            "cache": asdict(self.config.cache),
            "server": asdict(self.config.server),
            "llm": asdict(self.config.llm),
            "enable_temporal_graphs": self.config.enable_temporal_graphs,
            "enable_bilingual_extraction": self.config.enable_bilingual_extraction,
            "enable_multi_agent_collaboration": self.config.enable_multi_agent_collaboration,
            "enable_formal_verification": self.config.enable_formal_verification,
            "enable_retrieval_augmentation": self.config.enable_retrieval_augmentation,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "request_queue_size": self.config.request_queue_size,
            "response_cache_ttl": self.config.response_cache_ttl,
            "enable_request_logging": self.config.enable_request_logging,
            "enable_response_caching": self.config.enable_response_caching,
            "enable_authentication": self.config.enable_authentication,
            "enable_authorization": self.config.enable_authorization,
            "jwt_secret": self.config.jwt_secret,
            "rate_limit_requests": self.config.rate_limit_requests,
            "rate_limit_window": self.config.rate_limit_window,
            "log_level": self.config.log_level,
            "log_format": self.config.log_format,
            "enable_metrics": self.config.enable_metrics,
            "metrics_export_port": self.config.metrics_export_port,
            "enable_tracing": self.config.enable_tracing,
            "tracing_endpoint": self.config.tracing_endpoint
        }
        
        async with aiofiles.open(config_path, 'w') as f:
            if config_path.endswith('.json'):
                await f.write(json.dumps(config_dict, indent=2))
            elif config_path.endswith(('.yml', '.yaml')):
                await f.write(yaml.dump(config_dict, default_flow_style=False))
        
        logger.info(f"Configuration saved to {config_path}")
    
    def get_component_config(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific component."""
        if hasattr(self.config, component_name):
            component = getattr(self.config, component_name)
            if hasattr(component, '__dict__'):
                return component.__dict__
            elif hasattr(component, '__dataclass_fields__'):
                return asdict(component)
        return None


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


async def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create the global configuration manager.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


async def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file."""
    global _config_manager
    
    if _config_manager:
        # Close any resources held by the old config manager
        pass
    
    _config_manager = ConfigManager(config_path)
    logger.info("Configuration reloaded")


def get_config_value(path: str, default=None):
    """
    Get a configuration value using dot notation (e.g., 'server.port').
    
    Args:
        path: Configuration path using dot notation
        default: Default value if path not found
        
    Returns:
        Configuration value or default
    """
    global _config_manager
    
    if not _config_manager:
        raise RuntimeError("Configuration manager not initialized")
    
    # Split the path and navigate the config
    parts = path.split('.')
    current = _config_manager.config
    
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict):
            current = current.get(part, default)
        else:
            return default
    
    return current if current is not None else default


# Default configuration file template
DEFAULT_CONFIG_TEMPLATE = {
    "name": "OpenEvolve Knowledge Engine",
    "version": "1.0.0",
    "environment": "development",
    "database": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "username": "openevolve",
        "password": "",
        "database": "openevolve_kg",
        "connection_pool_size": 20,
        "ssl_enabled": False
    },
    "vector_store": {
        "type": "qdrant",
        "host": "localhost",
        "port": 6333,
        "collection_name": "knowledge_artifacts",
        "distance_metric": "cosine",
        "vector_size": 1536,
        "recreate_collection": False
    },
    "cache": {
        "type": "redis",
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "ttl_seconds": 3600,
        "max_items": 10000
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "timeout": 300,
        "max_connections": 1000,
        "cors_enabled": True,
        "cors_origins": ["*"],
        "ssl_enabled": False
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "",
        "base_url": None,
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "request_timeout": 120,
        "max_retries": 3,
        "retry_delay": 1.0
    },
    "enable_temporal_graphs": True,
    "enable_bilingual_extraction": True,
    "enable_multi_agent_collaboration": True,
    "enable_formal_verification": True,
    "enable_retrieval_augmentation": True,
    "max_concurrent_requests": 100,
    "request_queue_size": 1000,
    "response_cache_ttl": 300,
    "enable_request_logging": True,
    "enable_response_caching": True,
    "enable_authentication": True,
    "enable_authorization": True,
    "jwt_secret": "",
    "rate_limit_requests": 100,
    "rate_limit_window": 60,
    "log_level": "INFO",
    "log_format": "json",
    "enable_metrics": True,
    "metrics_export_port": 9090,
    "enable_tracing": False,
    "tracing_endpoint": None
}


async def create_default_config(config_path: str = "config.yaml"):
    """
    Create a default configuration file.
    
    Args:
        config_path: Path to create the configuration file
    """
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(config_path, 'w') as f:
        await f.write(yaml.dump(DEFAULT_CONFIG_TEMPLATE, default_flow_style=False))
    
    logger.info(f"Default configuration created at {config_path}")


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create default config if it doesn't exist
        config_path = "config.yaml"
        if not Path(config_path).exists():
            await create_default_config(config_path)
        
        # Initialize config manager
        config_mgr = await get_config_manager(config_path)
        
        print(f"Environment: {config_mgr.config.environment}")
        print(f"Database: {config_mgr.config.database.type}@{config_mgr.config.database.host}:{config_mgr.config.database.port}")
        print(f"Vector Store: {config_mgr.config.vector_store.type}@{config_mgr.config.vector_store.host}:{config_mgr.config.vector_store.port}")
        print(f"LLM Provider: {config_mgr.config.llm.provider}, Model: {config_mgr.config.llm.model}")
        
        # Example of getting a specific config value
        server_port = get_config_value("server.port")
        print(f"Server Port: {server_port}")
    
    asyncio.run(main())