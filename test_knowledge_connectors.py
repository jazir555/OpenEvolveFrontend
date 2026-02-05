"""Test knowledge source connectors."""

from external_knowledge_integration import (
    KnowledgeSourceConfig,
    KnowledgeSourceType,
    KnowledgeItem,
    DatabaseConnector,
    APIConnector,
    DocumentConnector,
    KnowledgeIntegrationManager
)


def test_knowledge_item():
    """Test KnowledgeItem dataclass."""
    print("Testing KnowledgeItem...")
    
    item = KnowledgeItem(
        source="test_source",
        content="Test content",
        relevance_score=0.85,
        metadata={"key": "value"}
    )
    
    assert item.source == "test_source"
    assert item.content == "Test content"
    assert item.relevance_score == 0.85
    assert item.metadata["key"] == "value"
    
    # Test to_dict
    item_dict = item.to_dict()
    assert item_dict["source"] == "test_source"
    assert item_dict["relevance_score"] == 0.85
    
    print("[OK] KnowledgeItem tests passed")


def test_knowledge_source_config():
    """Test KnowledgeSourceConfig."""
    print("\nTesting KnowledgeSourceConfig...")
    
    config = KnowledgeSourceConfig(
        name="test_db",
        source_type=KnowledgeSourceType.DATABASE,
        endpoint="postgresql://localhost:5432/knowledge",
        credentials={"username": "user", "password": "pass"},
        timeout=30,
        max_retries=3
    )
    
    assert config.name == "test_db"
    assert config.source_type == KnowledgeSourceType.DATABASE
    assert config.timeout == 30
    assert config.max_retries == 3
    
    print("[OK] KnowledgeSourceConfig tests passed")


def test_database_connector():
    """Test DatabaseConnector."""
    print("\nTesting DatabaseConnector...")
    
    config = KnowledgeSourceConfig(
        name="test_db",
        source_type=KnowledgeSourceType.DATABASE,
        endpoint="postgresql://localhost:5432/knowledge"
    )
    config.metadata = {"db_type": "postgresql"}
    
    connector = DatabaseConnector(config)
    
    assert connector.config.name == "test_db"
    assert connector.db_type == "postgresql"
    assert connector.is_available == True
    
    # Test metadata
    metadata = connector.get_metadata()
    assert metadata["name"] == "test_db"
    assert metadata["type"] == "database"
    assert metadata["is_available"] == True
    
    # Test query (will return empty list in placeholder implementation)
    context = {"query": "test query", "domain": "software"}
    results = connector.query(context)
    assert isinstance(results, list)
    
    print("[OK] DatabaseConnector tests passed")


def test_api_connector():
    """Test APIConnector."""
    print("\nTesting APIConnector...")
    
    config = KnowledgeSourceConfig(
        name="test_api",
        source_type=KnowledgeSourceType.API,
        endpoint="https://api.example.com",
        credentials={"api_key": "test_key"}
    )
    
    connector = APIConnector(config)
    
    assert connector.config.name == "test_api"
    assert connector.config.endpoint == "https://api.example.com"
    assert "Authorization" in connector.session.headers
    
    # Test metadata
    metadata = connector.get_metadata()
    assert metadata["name"] == "test_api"
    assert metadata["type"] == "api"
    
    print("[OK] APIConnector tests passed")


def test_document_connector():
    """Test DocumentConnector."""
    print("\nTesting DocumentConnector...")
    
    config = KnowledgeSourceConfig(
        name="test_docs",
        source_type=KnowledgeSourceType.DOCUMENT,
        endpoint="/path/to/documents"
    )
    config.metadata = {"doc_type": "pdf"}
    
    connector = DocumentConnector(config)
    
    assert connector.config.name == "test_docs"
    assert connector.doc_type == "pdf"
    assert connector.repository_path == "/path/to/documents"
    
    # Test metadata
    metadata = connector.get_metadata()
    assert metadata["name"] == "test_docs"
    assert metadata["type"] == "document"
    
    # Test query
    context = {"query": "test query", "keywords": ["test", "document"]}
    results = connector.query(context)
    assert isinstance(results, list)
    
    # Test index_documents
    doc_count = connector.index_documents(["/doc1.pdf", "/doc2.pdf"])
    assert doc_count == 2
    
    print("[OK] DocumentConnector tests passed")


def test_knowledge_integration_manager():
    """Test KnowledgeIntegrationManager with connectors."""
    print("\nTesting KnowledgeIntegrationManager...")
    
    manager = KnowledgeIntegrationManager()
    
    # Register connectors
    db_config = KnowledgeSourceConfig(
        name="db_source",
        source_type=KnowledgeSourceType.DATABASE
    )
    db_config.metadata = {"db_type": "postgresql"}
    db_connector = DatabaseConnector(db_config)
    manager.register_connector(db_connector)
    
    doc_config = KnowledgeSourceConfig(
        name="doc_source",
        source_type=KnowledgeSourceType.DOCUMENT
    )
    doc_config.metadata = {"doc_type": "text"}
    doc_connector = DocumentConnector(doc_config)
    manager.register_connector(doc_connector)
    
    # Check registered connectors
    assert len(manager.connectors) == 2
    assert "db_source" in manager.connectors
    assert "doc_source" in manager.connectors
    
    # Test get_connector_metadata
    metadata = manager.get_connector_metadata()
    assert len(metadata) == 2
    assert metadata["db_source"]["type"] == "database"
    assert metadata["doc_source"]["type"] == "document"
    
    # Test query_all_connectors
    context = {"query": "test", "domain": "software"}
    results = manager.query_all_connectors(context)
    assert len(results) == 2
    assert "db_source" in results
    assert "doc_source" in results
    
    # Test cache
    results2 = manager.query_all_connectors(context)
    assert results2 == results  # Should be from cache
    
    # Test clear cache
    manager.clear_cache()
    assert len(manager.cache) == 0
    
    print("[OK] KnowledgeIntegrationManager tests passed")


def test_connector_error_handling():
    """Test connector error handling."""
    print("\nTesting connector error handling...")
    
    config = KnowledgeSourceConfig(
        name="failing_api",
        source_type=KnowledgeSourceType.API,
        endpoint="https://invalid.example.com",
        fallback_enabled=True
    )
    
    connector = APIConnector(config)
    
    # Query should handle error gracefully with fallback
    context = {"query": "test"}
    results = connector.query(context)
    
    # Should return empty list due to fallback
    assert isinstance(results, list)
    assert len(results) == 0
    assert connector.last_error is not None
    
    print("[OK] Connector error handling tests passed")


if __name__ == "__main__":
    print("Running knowledge connector tests...\n")
    
    test_knowledge_item()
    test_knowledge_source_config()
    test_database_connector()
    test_api_connector()
    test_document_connector()
    test_knowledge_integration_manager()
    test_connector_error_handling()
    
    print("\n" + "="*50)
    print("All knowledge connector tests passed!")
    print("="*50)
