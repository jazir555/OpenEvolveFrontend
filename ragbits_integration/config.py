"""
RAGBits Configuration Module

Centralized configuration for RAGBits integration components.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store backend"""

    # Store type: "in_memory", "qdrant", "pgvector", "chroma"
    store_type: str = "in_memory"

    # Qdrant configuration
    qdrant_host: Optional[str] = None
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None

    # PGVector configuration
    pgvector_connection_string: Optional[str] = None
    pgvector_collection_name: str = "ragbits_artifacts"

    # Collection name
    collection_name: str = "workflow_artifacts"

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536


@dataclass
class DocumentSearchConfig:
    """Configuration for document search"""

    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Search settings
    default_top_k: int = 5
    similarity_threshold: float = 0.75

    # Ingestion settings
    ingest_batch_size: int = 100
    parallel_ingestion: bool = True
    max_ingestion_workers: int = 4


@dataclass
class StorageConfig:
    """Configuration for intermediary storage"""

    # Cache settings
    enable_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600

    # Versioning settings
    enable_versioning: bool = True
    max_versions_per_artifact: int = 10

    # Lifecycle settings
    enable_lifecycle_tracking: bool = True

    # Artifact retention
    retain_draft_artifacts: bool = True
    retain_rejected_artifacts: bool = True


@dataclass
class HephaestusIntegrationConfig:
    """Configuration for Hephaestus integration"""

    # Hephaestus endpoint
    hephaestus_endpoint: Optional[str] = None

    # Model mappings for different teams
    blue_team_model: str = "gpt-4"
    red_team_model: str = "claude-sonnet"
    gold_team_model: str = "gpt-4-turbo"

    # Fallback model
    fallback_model: str = "gpt-3.5-turbo"

    # Temperature settings
    blue_team_temperature: float = 0.7
    red_team_temperature: float = 0.5
    gold_team_temperature: float = 0.3


@dataclass
class RagbitsIntegrationConfig:
    """Main configuration for RAGBits integration"""

    # Sub-configurations
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    document_search: DocumentSearchConfig = field(default_factory=DocumentSearchConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    hephaestus: HephaestusIntegrationConfig = field(default_factory=HephaestusIntegrationConfig)

    # Feature flags
    enable_intermediary_storage: bool = True
    enable_semantic_search: bool = True
    enable_lifecycle_management: bool = True
    enable_hybrid_knowledge: bool = True

    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = False

    @classmethod
    def from_env(cls) -> "RagbitsIntegrationConfig":
        """
        Create configuration from environment variables.

        Environment variables:
        - RAGBITS_VECTOR_STORE_TYPE: Vector store type (in_memory, qdrant, pgvector)
        - RAGBITS_QDRANT_HOST: Qdrant host
        - RAGBITS_QDRANT_PORT: Qdrant port
        - RAGBITS_QDRANT_API_KEY: Qdrant API key
        - RAGBITS_PGVECTOR_CONN: PGVector connection string
        - RAGBITS_EMBEDDING_MODEL: Embedding model name
        - RAGBITS_HEPHAESTUS_ENDPOINT: Hephaestus endpoint
        - RAGBITS_LOG_LEVEL: Logging level
        """
        vector_store = VectorStoreConfig(
            store_type=os.getenv("RAGBITS_VECTOR_STORE_TYPE", "in_memory"),
            qdrant_host=os.getenv("RAGBITS_QDRANT_HOST"),
            qdrant_port=int(os.getenv("RAGBITS_QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("RAGBITS_QDRANT_API_KEY"),
            pgvector_connection_string=os.getenv("RAGBITS_PGVECTOR_CONN"),
            embedding_model=os.getenv("RAGBITS_EMBEDDING_MODEL", "text-embedding-3-small")
        )

        hephaestus = HephaestusIntegrationConfig(
            hephaestus_endpoint=os.getenv("RAGBITS_HEPHAESTUS_ENDPOINT")
        )

        return cls(
            vector_store=vector_store,
            hephaestus=hephaestus,
            log_level=os.getenv("RAGBITS_LOG_LEVEL", "INFO")
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RagbitsIntegrationConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            RagbitsIntegrationConfig instance
        """
        vector_store_config = VectorStoreConfig(**config_dict.get("vector_store", {}))
        document_search_config = DocumentSearchConfig(**config_dict.get("document_search", {}))
        storage_config = StorageConfig(**config_dict.get("storage", {}))
        hephaestus_config = HephaestusIntegrationConfig(**config_dict.get("hephaestus", {}))

        return cls(
            vector_store=vector_store_config,
            document_search=document_search_config,
            storage=storage_config,
            hephaestus=hephaestus_config,
            **{k: v for k, v in config_dict.items()
               if k not in ["vector_store", "document_search", "storage", "hephaestus"]}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "vector_store": {
                "store_type": self.vector_store.store_type,
                "qdrant_host": self.vector_store.qdrant_host,
                "qdrant_port": self.vector_store.qdrant_port,
                "qdrant_api_key": self.vector_store.qdrant_api_key,
                "pgvector_connection_string": self.vector_store.pgvector_connection_string,
                "collection_name": self.vector_store.collection_name,
                "embedding_model": self.vector_store.embedding_model,
                "embedding_dimension": self.vector_store.embedding_dimension
            },
            "document_search": {
                "chunk_size": self.document_search.chunk_size,
                "chunk_overlap": self.document_search.chunk_overlap,
                "default_top_k": self.document_search.default_top_k,
                "similarity_threshold": self.document_search.similarity_threshold,
                "ingest_batch_size": self.document_search.ingest_batch_size,
                "parallel_ingestion": self.document_search.parallel_ingestion,
                "max_ingestion_workers": self.document_search.max_ingestion_workers
            },
            "storage": {
                "enable_cache": self.storage.enable_cache,
                "cache_max_size": self.storage.cache_max_size,
                "cache_ttl_seconds": self.storage.cache_ttl_seconds,
                "enable_versioning": self.storage.enable_versioning,
                "max_versions_per_artifact": self.storage.max_versions_per_artifact,
                "enable_lifecycle_tracking": self.storage.enable_lifecycle_tracking
            },
            "hephaestus": {
                "hephaestus_endpoint": self.hephaestus.hephaestus_endpoint,
                "blue_team_model": self.hephaestus.blue_team_model,
                "red_team_model": self.hephaestus.red_team_model,
                "gold_team_model": self.hephaestus.gold_team_model,
                "fallback_model": self.hephaestus.fallback_model
            },
            "enable_intermediary_storage": self.enable_intermediary_storage,
            "enable_semantic_search": self.enable_semantic_search,
            "enable_lifecycle_management": self.enable_lifecycle_management,
            "enable_hybrid_knowledge": self.enable_hybrid_knowledge,
            "log_level": self.log_level,
            "enable_detailed_logging": self.enable_detailed_logging
        }

    def validate(self) -> bool:
        """
        Validate configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        errors = []

        # Validate vector store configuration
        if self.vector_store.store_type not in ["in_memory", "qdrant", "pgvector", "chroma"]:
            errors.append(f"Invalid vector store type: {self.vector_store.store_type}")

        if self.vector_store.store_type == "qdrant" and not self.vector_store.qdrant_host:
            errors.append("Qdrant host is required when using Qdrant")

        if self.vector_store.store_type == "pgvector" and not self.vector_store.pgvector_connection_string:
            errors.append("PGVector connection string is required when using PGVector")

        # Validate numeric ranges
        if not (0 <= self.document_search.similarity_threshold <= 1):
            errors.append("Similarity threshold must be between 0 and 1")

        if self.document_search.chunk_size <= 0:
            errors.append("Chunk size must be positive")

        if self.document_search.chunk_overlap >= self.document_search.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")

        # Validate model names
        if not all([
            self.hephaestus.blue_team_model,
            self.hephaestus.red_team_model,
            self.hephaestus.gold_team_model
        ]):
            errors.append("All team models must be specified")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")
        return True


def get_default_config() -> RagbitsIntegrationConfig:
    """Get default configuration for development"""
    return RagbitsIntegrationConfig(
        vector_store=VectorStoreConfig(
            store_type="in_memory",
            collection_name="dev_workflow_artifacts"
        ),
        document_search=DocumentSearchConfig(
            default_top_k=5,
            similarity_threshold=0.75
        ),
        storage=StorageConfig(
            enable_cache=True,
            cache_max_size=1000
        ),
        log_level="DEBUG",
        enable_detailed_logging=True
    )


def get_production_config() -> RagbitsIntegrationConfig:
    """Get production configuration template"""
    return RagbitsIntegrationConfig(
        vector_store=VectorStoreConfig(
            store_type="qdrant",  # or "pgvector"
            collection_name="prod_workflow_artifacts",
            embedding_model="text-embedding-3-small"
        ),
        document_search=DocumentSearchConfig(
            default_top_k=10,
            similarity_threshold=0.80,
            ingest_batch_size=200,
            parallel_ingestion=True,
            max_ingestion_workers=8
        ),
        storage=StorageConfig(
            enable_cache=True,
            cache_max_size=10000,
            cache_ttl_seconds=7200,
            enable_versioning=True,
            max_versions_per_artifact=20
        ),
        log_level="INFO",
        enable_detailed_logging=False
    )
