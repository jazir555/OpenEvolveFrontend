"""
Comprehensive tests for Outlines Knowledge Engine Integration.

Test categories:
- Unit tests for all adapter methods
- Integration tests with mock LLM responses
- End-to-end tests for KG workflows
- Performance benchmarks

Coverage target: >90%
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import Mock, patch, MagicMock

# Test imports
from integrations.outlines import (
    OutlinesAdapter,
    OutlinesConfig,
    OutlinesResult,
    ModelProvider,
    EntityExtractionSchema,
    RelationshipSchema,
    CypherQuerySchema,
    ValidationResultSchema,
    KnowledgeGraphConstraints,
    PromptTemplateManager,
    GenerationError,
    ValidationError,
)

from knowledge_engine.integrations.outlines import (
    OutlinesKGIntegration,
    KGExtractionResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Default test configuration."""
    return OutlinesConfig(
        model_provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        api_key="test-key",
        max_retries=1,
        enable_fallback=False,
    )


@pytest.fixture
def adapter(config):
    """Outlines adapter fixture."""
    with patch('integrations.outlines.adapter.openai') as mock_openai:
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        adapter = OutlinesAdapter(config)
        adapter._client = mock_client
        return adapter


@pytest.fixture
def kg_integration(config):
    """KG integration fixture."""
    with patch('integrations.outlines.adapter.openai') as mock_openai:
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        integration = OutlinesKGIntegration(config)
        return integration


@pytest.fixture
def sample_entity_schema():
    """Sample entity extraction result."""
    return {
        "entities": [
            {
                "name": "John Smith",
                "type": "PERSON",
                "confidence": 0.95,
                "properties": [
                    {"name": "title", "value": "CEO", "confidence": 0.9}
                ]
            },
            {
                "name": "Acme Corp",
                "type": "ORGANIZATION",
                "confidence": 0.92,
                "properties": []
            }
        ],
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": "gpt-4"
    }


@pytest.fixture
def sample_relationship_schema():
    """Sample relationship extraction result."""
    return {
        "relationships": [
            {
                "source": "John Smith",
                "target": "Acme Corp",
                "type": "WORKS_FOR",
                "confidence": 0.88,
                "properties": [
                    {"name": "since", "value": "2020", "confidence": 0.85}
                ],
                "directed": True
            }
        ],
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_used": "gpt-4"
    }


@pytest.fixture
def sample_cypher_schema():
    """Sample Cypher query result."""
    return {
        "query": "MATCH (p:PERSON {name: $name}) RETURN p",
        "parameters": {"name": "John Smith"},
        "explanation": "Find person by name",
        "query_type": "READ",
        "estimated_complexity": "LOW",
        "requires_index": True,
        "idempotent": True
    }


# =============================================================================
# UNIT TESTS - ADAPTER
# =============================================================================

class TestOutlinesAdapter:
    """Unit tests for OutlinesAdapter."""
    
    def test_adapter_initialization(self, config):
        """Test adapter initialization."""
        with patch('integrations.outlines.adapter.openai'):
            adapter = OutlinesAdapter(config)
            assert adapter.config == config
            assert adapter.circuit_breaker is not None
            assert adapter.grammar_cache is not None
    
    def test_adapter_initialization_without_api_key(self):
        """Test adapter initialization without API key."""
        config = OutlinesConfig(
            model_provider=ModelProvider.OPENAI,
            api_key=None,
        )
        with pytest.raises(GenerationError):
            OutlinesAdapter(config)
    
    def test_circuit_breaker(self, adapter):
        """Test circuit breaker functionality."""
        assert adapter.circuit_breaker.can_execute()
        
        # Simulate failures
        for _ in range(5):
            adapter.circuit_breaker.record_failure()
        
        assert not adapter.circuit_breaker.can_execute()
        
        # Reset
        adapter.circuit_breaker.record_success()
        assert adapter.circuit_breaker.can_execute()
    
    def test_grammar_cache(self, adapter):
        """Test grammar cache."""
        adapter.grammar_cache.set("test_key", "test_value")
        assert adapter.grammar_cache.get("test_key") == "test_value"
        
        # Test miss
        assert adapter.grammar_cache.get("nonexistent") is None
        
        # Test eviction
        adapter.grammar_cache.maxsize = 2
        adapter.grammar_cache.set("key1", "value1")
        adapter.grammar_cache.set("key2", "value2")
        adapter.grammar_cache.set("key3", "value3")
        
        # One of the early keys should be evicted
        assert adapter.grammar_cache.get("key3") == "value3"
    
    @patch('integrations.outlines.adapter.openai')
    def test_generate_json_success(self, mock_openai, adapter, sample_entity_schema):
        """Test successful JSON generation."""
        # Mock the response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(sample_entity_schema)))]
        adapter._client.chat.completions.create.return_value = mock_response
        
        schema = EntityExtractionSchema
        prompt = "Extract entities from: John Smith works at Acme Corp"
        
        result = adapter.generate_json(schema, prompt)
        
        assert result.success
        assert result.constraint_type == "json"
        assert "entities" in result.output
        assert len(result.output["entities"]) == 2
    
    @patch('integrations.outlines.adapter.openai')
    def test_generate_json_failure(self, mock_openai, adapter):
        """Test JSON generation failure with fallback."""
        adapter.config.fallback_to_unconstrained = True
        
        # Mock failure then success on fallback
        adapter._client.chat.completions.create.side_effect = [
            Exception("API Error"),
            Mock(choices=[Mock(message=Mock(content="fallback response"))])
        ]
        
        schema = EntityExtractionSchema
        prompt = "Extract entities"
        
        result = adapter.generate_json(schema, prompt)
        
        assert result.success  # Fallback succeeded
        assert result.constraint_type == "unconstrained"
    
    @patch('integrations.outlines.adapter.openai')
    def test_generate_regex_success(self, mock_openai, adapter):
        """Test successful regex generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="12345"))]
        adapter._client.chat.completions.create.return_value = mock_response
        
        pattern = r'^\d{5}$'
        prompt = "Generate a 5-digit number"
        
        result = adapter.generate_regex(pattern, prompt)
        
        assert result.success
        assert result.constraint_type == "regex"
        assert result.output == "12345"
    
    @patch('integrations.outlines.adapter.openai')
    def test_generate_choices_success(self, mock_openai, adapter):
        """Test successful choices generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="option_b"))]
        adapter._client.chat.completions.create.return_value = mock_response
        
        choices = ["option_a", "option_b", "option_c"]
        prompt = "Select the best option"
        
        result = adapter.generate_choices(choices, prompt)
        
        assert result.success
        assert result.constraint_type == "choices"
    
    @patch('integrations.outlines.adapter.openai')
    def test_batch_generate(self, mock_openai, adapter):
        """Test batch generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"result": "success"}'))]
        adapter._client.chat.completions.create.return_value = mock_response
        
        tasks = [
            {"type": "json", "constraint": {"type": "object"}, "prompt": "task 1"},
            {"type": "json", "constraint": {"type": "object"}, "prompt": "task 2"},
        ]
        
        results = adapter.batch_generate(tasks)
        
        assert len(results) == 2
        assert all(r.success for r in results)
    
    def test_validate_output(self, adapter):
        """Test output validation."""
        # JSON validation
        assert adapter.validate_output('{"key": "value"}', {"type": "object"})
        
        # Regex validation
        assert adapter.validate_output("12345", r'^\d+$')
        
        # Choices validation
        assert adapter.validate_output("choice_a", ["choice_a", "choice_b"])
        
        # Invalid cases
        assert not adapter.validate_output("invalid", r'^\d+$')
        assert not adapter.validate_output("unknown", ["choice_a", "choice_b"])


# =============================================================================
# UNIT TESTS - KG CONSTRAINTS
# =============================================================================

class TestKnowledgeGraphConstraints:
    """Unit tests for KG constraints."""
    
    def test_entity_extraction_schema(self, sample_entity_schema):
        """Test EntityExtractionSchema validation."""
        schema = EntityExtractionSchema(**sample_entity_schema)
        
        assert len(schema.entities) == 2
        assert schema.entities[0].name == "John Smith"
        assert schema.entities[0].type == "PERSON"
        assert schema.entities[0].confidence == 0.95
        
        # Test Memgraph conversion
        nodes = schema.to_memgraph_nodes()
        assert len(nodes) == 2
        assert nodes[0]["labels"] == ["PERSON"]
        assert nodes[0]["properties"]["name"] == "John Smith"
    
    def test_relationship_schema(self, sample_relationship_schema):
        """Test RelationshipSchema validation."""
        schema = RelationshipSchema(**sample_relationship_schema)
        
        assert len(schema.relationships) == 1
        rel = schema.relationships[0]
        assert rel.source == "John Smith"
        assert rel.target == "Acme Corp"
        assert rel.type == "WORKS_FOR"
        
        # Test Memgraph conversion
        edges = schema.to_memgraph_edges()
        assert len(edges) == 1
        assert edges[0]["type"] == "WORKS_FOR"
    
    def test_cypher_query_schema(self, sample_cypher_schema):
        """Test CypherQuerySchema validation."""
        schema = CypherQuerySchema(**sample_cypher_schema)
        
        assert "MATCH" in schema.query
        assert schema.query_type == CypherQuerySchema.QueryType.READ
        assert schema.idempotent is True
        
        # Test Memgraph compatibility
        query = schema.to_memgraph_query()
        assert "MATCH" in query
    
    def test_validation_result_schema(self):
        """Test ValidationResultSchema."""
        result = ValidationResultSchema(is_valid=True)
        assert result.is_valid is True
        assert result.confidence == 1.0
        
        # Test adding errors
        result.add_error("Test error", "field1", "Fix it")
        assert not result.is_valid
        assert len(result.errors) == 1
        assert len(result.issues) == 1
        
        # Test adding warnings
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
    
    def test_entity_types(self):
        """Test entity type enumeration."""
        types = KnowledgeGraphConstraints.get_entity_types()
        assert "PERSON" in types
        assert "ORGANIZATION" in types
        assert "LOCATION" in types
    
    def test_relation_types(self):
        """Test relation type enumeration."""
        types = KnowledgeGraphConstraints.get_relation_types()
        assert "WORKS_FOR" in types
        assert "LOCATED_IN" in types
        assert "RELATED_TO" in types
    
    def test_validate_entity_name(self):
        """Test entity name validation."""
        assert KnowledgeGraphConstraints.validate_entity_name("John Smith")
        assert KnowledgeGraphConstraints.validate_entity_name("Entity_123")
        assert not KnowledgeGraphConstraints.validate_entity_name("")  # Too short
    
    def test_validate_relation_type(self):
        """Test relation type validation."""
        assert KnowledgeGraphConstraints.validate_relation_type("WORKS_FOR")
        assert KnowledgeGraphConstraints.validate_relation_type("RELATED")
        assert not KnowledgeGraphConstraints.validate_relation_type("lowercase")  # Must be uppercase


# =============================================================================
# UNIT TESTS - PROMPT TEMPLATES
# =============================================================================

class TestPromptTemplateManager:
    """Unit tests for PromptTemplateManager."""
    
    def test_get_template(self):
        """Test getting templates."""
        template = PromptTemplateManager.get_template("entity_extraction")
        assert "ENTITY TYPES TO EXTRACT" in template
        
        with pytest.raises(ValueError):
            PromptTemplateManager.get_template("nonexistent")
    
    def test_list_templates(self):
        """Test listing available templates."""
        templates = PromptTemplateManager.list_templates()
        assert "entity_extraction" in templates
        assert "cypher_generation" in templates
    
    def test_create_entity_extraction_prompt(self):
        """Test entity extraction prompt creation."""
        prompt = PromptTemplateManager.create_entity_extraction_prompt(
            text="John Smith works at Acme Corp",
            entity_types=["PERSON", "ORGANIZATION"],
            model="gpt-4"
        )
        
        assert "John Smith works at Acme Corp" in prompt
        assert "PERSON" in prompt
        assert "ORGANIZATION" in prompt
    
    def test_create_relation_extraction_prompt(self):
        """Test relation extraction prompt creation."""
        prompt = PromptTemplateManager.create_relation_extraction_prompt(
            text="John Smith works at Acme Corp",
            relation_types=["WORKS_FOR"],
            entities=["John Smith", "Acme Corp"]
        )
        
        assert "John Smith works at Acme Corp" in prompt
        assert "WORKS_FOR" in prompt
        assert "John Smith" in prompt
    
    def test_create_cypher_generation_prompt(self):
        """Test Cypher generation prompt creation."""
        prompt = PromptTemplateManager.create_cypher_generation_prompt(
            query_intent="Find all persons",
            schema_description="Graph with PERSON nodes",
            node_labels=["PERSON", "COMPANY"],
            relationship_types=["WORKS_FOR"]
        )
        
        assert "Find all persons" in prompt
        assert "PERSON" in prompt
        assert "MATCH" in prompt or "Cypher" in prompt


# =============================================================================
# INTEGRATION TESTS - KG INTEGRATION
# =============================================================================

class TestOutlinesKGIntegration:
    """Integration tests for OutlinesKGIntegration."""
    
    @patch('integrations.outlines.adapter.openai')
    def test_extract_entities_constrained(self, mock_openai, kg_integration, sample_entity_schema):
        """Test constrained entity extraction."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(sample_entity_schema)))]
        kg_integration.adapter._client.chat.completions.create.return_value = mock_response
        
        result = kg_integration.extract_entities_constrained(
            text="John Smith works at Acme Corp",
            entity_types=["PERSON", "ORGANIZATION"]
        )
        
        assert isinstance(result, EntityExtractionSchema)
        assert len(result.entities) == 2
        assert result.entities[0].name == "John Smith"
    
    @patch('integrations.outlines.adapter.openai')
    def test_extract_relations_constrained(self, mock_openai, kg_integration, sample_relationship_schema):
        """Test constrained relationship extraction."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(sample_relationship_schema)))]
        kg_integration.adapter._client.chat.completions.create.return_value = mock_response
        
        result = kg_integration.extract_relations_constrained(
            text="John Smith works at Acme Corp",
            relation_types=["WORKS_FOR"],
            entities=["John Smith", "Acme Corp"]
        )
        
        assert isinstance(result, RelationshipSchema)
        assert len(result.relationships) == 1
        assert result.relationships[0].type == "WORKS_FOR"
    
    @patch('integrations.outlines.adapter.openai')
    def test_generate_cypher_constrained(self, mock_openai, kg_integration, sample_cypher_schema):
        """Test constrained Cypher generation."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(sample_cypher_schema)))]
        kg_integration.adapter._client.chat.completions.create.return_value = mock_response
        
        result = kg_integration.generate_cypher_constrained(
            query_intent="Find person by name",
            schema_description="Graph with PERSON nodes",
        )
        
        assert isinstance(result, CypherQuerySchema)
        assert "MATCH" in result.query
    
    def test_validate_kg_structure_valid(self, kg_integration):
        """Test KG validation with valid data."""
        kg_data = {
            "entities": [
                {"name": "John", "type": "PERSON"},
                {"name": "Acme", "type": "ORGANIZATION"},
            ],
            "relationships": [
                {"source": "John", "target": "Acme", "type": "WORKS_FOR"},
            ]
        }
        
        result = kg_integration.validate_kg_structure(kg_data)
        
        assert isinstance(result, ValidationResultSchema)
        assert result.is_valid is True
    
    def test_validate_kg_structure_invalid(self, kg_integration):
        """Test KG validation with invalid data."""
        kg_data = {
            "entities": [
                {"type": "PERSON"},  # Missing name
            ],
            "relationships": [
                {"source": "John"},  # Missing target and type
            ]
        }
        
        result = kg_integration.validate_kg_structure(kg_data)
        
        assert isinstance(result, ValidationResultSchema)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    @patch('integrations.outlines.adapter.openai')
    def test_batch_process_documents(self, mock_openai, kg_integration, sample_entity_schema, sample_relationship_schema):
        """Test batch document processing."""
        # Mock responses for multiple calls
        mock_response_entity = Mock()
        mock_response_entity.choices = [Mock(message=Mock(content=json.dumps(sample_entity_schema)))]
        
        mock_response_rel = Mock()
        mock_response_rel.choices = [Mock(message=Mock(content=json.dumps(sample_relationship_schema)))]
        
        kg_integration.adapter._client.chat.completions.create.side_effect = [
            mock_response_entity, mock_response_rel,  # Doc 1
            mock_response_entity, mock_response_rel,  # Doc 2
        ]
        
        docs = [
            {"id": "doc1", "text": "John Smith works at Acme Corp"},
            {"id": "doc2", "text": "Jane Doe works at Tech Inc"},
        ]
        
        results = kg_integration.batch_process_documents(docs)
        
        assert len(results) == 2
        assert all(isinstance(r, KGExtractionResult) for r in results)
        assert all(r.success for r in results)
    
    @patch('integrations.outlines.adapter.openai')
    def test_extract_and_build_kg(self, mock_openai, kg_integration, sample_entity_schema, sample_relationship_schema):
        """Test complete KG extraction and build."""
        mock_response_entity = Mock()
        mock_response_entity.choices = [Mock(message=Mock(content=json.dumps(sample_entity_schema)))]
        
        mock_response_rel = Mock()
        mock_response_rel.choices = [Mock(message=Mock(content=json.dumps(sample_relationship_schema)))]
        
        kg_integration.adapter._client.chat.completions.create.side_effect = [
            mock_response_entity,
            mock_response_rel,
        ]
        
        result = kg_integration.extract_and_build_kg(
            text="John Smith works at Acme Corp",
            entity_types=["PERSON", "ORGANIZATION"],
            relation_types=["WORKS_FOR"]
        )
        
        assert isinstance(result, KGExtractionResult)
        assert result.success is True
        assert len(result.entities) > 0
        assert len(result.relationships) > 0
        assert len(result.cypher_queries) > 0
    
    def test_get_status(self, kg_integration):
        """Test status retrieval."""
        status = kg_integration.get_status()
        
        assert "adapter_initialized" in status
        assert "hub_connected" in status
        assert "model_provider" in status
        assert "timestamp" in status


# =============================================================================
# END-TO-END TESTS
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    @patch('integrations.outlines.adapter.openai')
    def test_full_extraction_pipeline(self, mock_openai):
        """Test complete extraction pipeline."""
        # Setup mock
        entity_response = {
            "entities": [
                {"name": "Alice", "type": "PERSON", "confidence": 0.95, "properties": []},
                {"name": "Bob", "type": "PERSON", "confidence": 0.93, "properties": []},
                {"name": "TechCorp", "type": "ORGANIZATION", "confidence": 0.91, "properties": []},
            ]
        }
        
        relation_response = {
            "relationships": [
                {"source": "Alice", "target": "TechCorp", "type": "WORKS_FOR", "confidence": 0.88, "properties": [], "directed": True},
                {"source": "Bob", "target": "TechCorp", "type": "WORKS_FOR", "confidence": 0.85, "properties": [], "directed": True},
            ]
        }
        
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(entity_response)))]
        
        mock_response2 = Mock()
        mock_response2.choices = [Mock(message=Mock(content=json.dumps(relation_response)))]
        
        mock_client.chat.completions.create.side_effect = [mock_response, mock_response2]
        
        # Initialize and run
        config = OutlinesConfig(api_key="test-key")
        integration = OutlinesKGIntegration(config)
        integration.adapter._client = mock_client
        
        text = "Alice and Bob both work at TechCorp. They are software engineers."
        
        result = integration.extract_and_build_kg(
            text=text,
            entity_types=["PERSON", "ORGANIZATION"],
            relation_types=["WORKS_FOR"]
        )
        
        # Validate result
        assert result.success
        assert len(result.entities) == 3
        assert len(result.relationships) == 2
        
        # Validate Memgraph format
        memgraph_format = result.to_memgraph_format()
        assert "nodes" in memgraph_format
        assert "edges" in memgraph_format
        assert "queries" in memgraph_format
    
    @patch('integrations.outlines.adapter.openai')
    def test_cypher_generation_pipeline(self, mock_openai):
        """Test Cypher generation pipeline."""
        cypher_response = {
            "query": "MATCH (p:PERSON)-[:WORKS_FOR]->(c:ORGANIZATION) RETURN p, c",
            "parameters": {},
            "explanation": "Find all people who work for organizations",
            "query_type": "READ",
            "estimated_complexity": "MEDIUM",
            "requires_index": True,
            "idempotent": True
        }
        
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(cypher_response)))]
        mock_client.chat.completions.create.return_value = mock_response
        
        config = OutlinesConfig(api_key="test-key")
        integration = OutlinesKGIntegration(config)
        integration.adapter._client = mock_client
        
        result = integration.generate_cypher_constrained(
            query_intent="Find all employees and their companies",
            schema_description="Graph with PERSON and ORGANIZATION nodes, WORKS_FOR relationships",
            node_labels=["PERSON", "ORGANIZATION"],
            relationship_types=["WORKS_FOR"]
        )
        
        assert isinstance(result, CypherQuerySchema)
        assert "MATCH" in result.query
        assert result.query_type == CypherQuerySchema.QueryType.READ


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @patch('integrations.outlines.adapter.openai')
    def test_batch_processing_performance(self, mock_openai):
        """Benchmark batch processing performance."""
        entity_response = {
            "entities": [{"name": "Test", "type": "PERSON", "confidence": 0.9, "properties": []}]
        }
        relation_response = {"relationships": []}
        
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps(entity_response)))]
        
        mock_response2 = Mock()
        mock_response2.choices = [Mock(message=Mock(content=json.dumps(relation_response)))]
        
        mock_client.chat.completions.create.side_effect = [
            mock_response, mock_response2,
            mock_response, mock_response2,
            mock_response, mock_response2,
        ]
        
        config = OutlinesConfig(api_key="test-key", batch_max_workers=3)
        integration = OutlinesKGIntegration(config)
        integration.adapter._client = mock_client
        
        docs = [
            {"id": f"doc_{i}", "text": f"Test document {i}"}
            for i in range(3)
        ]
        
        start_time = time.time()
        results = integration.batch_process_documents(docs)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Should complete in reasonable time (parallel processing)
        # 3 docs with 2 calls each = 6 calls, but parallel so < 6 * mock_delay
        print(f"Batch processing time: {elapsed_time:.2f}s")
    
    def test_caching_performance(self, adapter):
        """Test that caching improves performance."""
        # Insert multiple items
        start_time = time.time()
        for i in range(100):
            adapter.grammar_cache.set(f"key_{i}", f"value_{i}")
        insert_time = time.time() - start_time
        
        # Retrieve items
        start_time = time.time()
        for i in range(100):
            adapter.grammar_cache.get(f"key_{i}")
        retrieve_time = time.time() - start_time
        
        # Retrieval should be fast
        assert retrieve_time < insert_time * 0.1  # At least 10x faster


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Error handling tests."""
    
    @patch('integrations.outlines.adapter.openai')
    def test_api_error_handling(self, mock_openai):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        config = OutlinesConfig(api_key="test-key", enable_fallback=False)
        adapter = OutlinesAdapter(config)
        adapter._client = mock_client
        
        result = adapter.generate_json({"type": "object"}, "test prompt")
        
        assert not result.success
        assert result.error is not None
        assert "API Error" in result.error
    
    @patch('integrations.outlines.adapter.openai')
    def test_fallback_generation(self, mock_openai):
        """Test fallback to unconstrained generation."""
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        # First call fails, second succeeds (fallback)
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="fallback output"))]
        
        mock_client.chat.completions.create.side_effect = [
            Exception("Constraint Error"),
            mock_response
        ]
        
        config = OutlinesConfig(api_key="test-key", fallback_to_unconstrained=True)
        adapter = OutlinesAdapter(config)
        adapter._client = mock_client
        
        result = adapter.generate_json({"type": "object"}, "test prompt")
        
        assert result.success
        assert result.constraint_type == "unconstrained"
        assert result.metadata.get("fallback") is True
    
    def test_invalid_schema_handling(self, kg_integration):
        """Test handling of invalid schema data."""
        invalid_data = {
            "entities": [
                {"invalid": "entity"}  # Missing required fields
            ]
        }
        
        result = kg_integration.validate_kg_structure(invalid_data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    @patch('integrations.outlines.adapter.openai')
    def test_timeout_handling(self, mock_openai):
        """Test timeout handling."""
        import concurrent.futures
        
        mock_client = Mock()
        mock_openai.OpenAI.return_value = mock_client
        
        def slow_call(*args, **kwargs):
            import time
            time.sleep(10)  # Simulate slow API
            return Mock(choices=[Mock(message=Mock(content="{}"))])
        
        mock_client.chat.completions.create.side_effect = slow_call
        
        config = OutlinesConfig(api_key="test-key", batch_timeout_seconds=0.1)
        adapter = OutlinesAdapter(config)
        adapter._client = mock_client
        
        tasks = [{"type": "json", "constraint": {}, "prompt": "test"}]
        
        with pytest.raises(concurrent.futures.TimeoutError):
            adapter.batch_generate(tasks)


# =============================================================================
# MEMGRAPH COMPATIBILITY TESTS
# =============================================================================

class TestMemgraphCompatibility:
    """Tests for Memgraph-compatible outputs."""
    
    def test_memgraph_node_format(self, sample_entity_schema):
        """Test entity conversion to Memgraph nodes."""
        schema = EntityExtractionSchema(**sample_entity_schema)
        nodes = schema.to_memgraph_nodes()
        
        for node in nodes:
            assert "labels" in node
            assert "properties" in node
            assert isinstance(node["labels"], list)
            assert isinstance(node["properties"], dict)
            assert "name" in node["properties"]
    
    def test_memgraph_edge_format(self, sample_relationship_schema):
        """Test relationship conversion to Memgraph edges."""
        schema = RelationshipSchema(**sample_relationship_schema)
        edges = schema.to_memgraph_edges()
        
        for edge in edges:
            assert "type" in edge
            assert "from" in edge
            assert "to" in edge
            assert "properties" in edge
    
    def test_cypher_memgraph_compatibility(self, sample_cypher_schema):
        """Test Cypher query Memgraph compatibility."""
        schema = CypherQuerySchema(**sample_cypher_schema)
        
        # Check for Neo4j-specific features
        neo4j_features = ["apoc.", "db.labels", "db.schema"]
        for feature in neo4j_features:
            assert feature not in schema.query.lower(), f"Query contains Neo4j-specific feature: {feature}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=knowledge_engine.integrations.outlines", "--cov=integrations.outlines"])
