"""
Unit tests for RAGBits Configuration
"""

import pytest
import os
from ragbits_integration.config import (
    RagbitsIntegrationConfig,
    VectorStoreConfig,
    DocumentSearchConfig,
    StorageConfig,
    CrewAIIntegrationConfig,
    get_default_config,
    get_production_config
)


def test_default_config():
    """Test creating default configuration"""
    config = get_default_config()

    assert config.vector_store.store_type == "in_memory"
    assert config.storage.enable_cache is True
    assert config.log_level == "DEBUG"
    assert config.enable_detailed_logging is True


def test_production_config():
    """Test creating production configuration"""
    config = get_production_config()

    assert config.vector_store.store_type in ["qdrant", "pgvector"]
    assert config.log_level == "INFO"
    assert config.enable_detailed_logging is False


def test_config_validation():
    """Test configuration validation"""
    config = get_default_config()
    assert config.validate() is True


def test_config_validation_invalid_vector_store():
    """Test validation with invalid vector store type"""
    config = RagbitsIntegrationConfig(
        vector_store=VectorStoreConfig(store_type="invalid_type")
    )

    with pytest.raises(ValueError, match="Invalid vector store type"):
        config.validate()


def test_config_validation_missing_qdrant_host():
    """Test validation with missing Qdrant host"""
    config = RagbitsIntegrationConfig(
        vector_store=VectorStoreConfig(
            store_type="qdrant",
            qdrant_host=None  # Missing required field
        )
    )

    with pytest.raises(ValueError, match="Qdrant host is required"):
        config.validate()


def test_config_to_dict():
    """Test converting configuration to dictionary"""
    config = get_default_config()
    config_dict = config.to_dict()

    assert "vector_store" in config_dict
    assert "document_search" in config_dict
    assert "storage" in config_dict
    assert "CREWAI" in config_dict


def test_config_from_dict():
    """Test creating configuration from dictionary"""
    config_dict = {
        "vector_store": {
            "store_type": "in_memory"
        },
        "document_search": {
            "default_top_k": 10
        },
        "storage": {
            "enable_cache": True
        },
        "CREWAI": {
            "blue_team_model": "gpt-4"
        },
        "log_level": "INFO"
    }

    config = RagbitsIntegrationConfig.from_dict(config_dict)

    assert config.vector_store.store_type == "in_memory"
    assert config.document_search.default_top_k == 10
    assert config.storage.enable_cache is True
    assert config.CREWAI.blue_team_model == "gpt-4"


def test_config_from_env():
    """Test creating configuration from environment variables"""
    # Set environment variables
    original_env = {}
    env_vars = {
        "RAGBITS_VECTOR_STORE_TYPE": "qdrant",
        "RAGBITS_QDRANT_HOST": "localhost",
        "RAGBITS_QDRANT_PORT": "6333",
        "RAGBITS_LOG_LEVEL": "WARNING"
    }

    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        config = RagbitsIntegrationConfig.from_env()

        assert config.vector_store.store_type == "qdrant"
        assert config.vector_store.qdrant_host == "localhost"
        assert config.vector_store.qdrant_port == 6333
        assert config.log_level == "WARNING"

    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def test_vector_store_config():
    """Test vector store configuration"""
    config = VectorStoreConfig(
        store_type="qdrant",
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name="test_collection"
    )

    assert config.store_type == "qdrant"
    assert config.qdrant_host == "localhost"
    assert config.qdrant_port == 6333
    assert config.collection_name == "test_collection"


def test_document_search_config():
    """Test document search configuration"""
    config = DocumentSearchConfig(
        chunk_size=1000,
        chunk_overlap=100,
        default_top_k=10
    )

    assert config.chunk_size == 1000
    assert config.chunk_overlap == 100
    assert config.default_top_k == 10


def test_storage_config():
    """Test storage configuration"""
    config = StorageConfig(
        enable_cache=True,
        cache_max_size=5000,
        enable_versioning=True
    )

    assert config.enable_cache is True
    assert config.cache_max_size == 5000
    assert config.enable_versioning is True


def test_CREWAI_config():
    """Test CREWAI integration configuration"""
    config = CrewAIIntegrationConfig(
        blue_team_model="gpt-4",
        red_team_model="claude-opus",
        gold_team_model="gpt-4-turbo"
    )

    assert config.blue_team_model == "gpt-4"
    assert config.red_team_model == "claude-opus"
    assert config.gold_team_model == "gpt-4-turbo"
