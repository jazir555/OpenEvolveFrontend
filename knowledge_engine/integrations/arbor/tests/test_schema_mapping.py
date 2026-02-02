"""
Tests for Arbor Schema Mapping

Following CLAUDE.md principles:
- CONTRACT TESTS: Verify mapping contracts
- TYPE SAFETY: Validate conversions
"""

import pytest
from datetime import datetime

from knowledge_engine.integrations.arbor import (
    ArborSchemaMapper,
    convert_arbor_node,
    convert_arbor_edge,
    ARBOR_KIND_TO_ENTITY_TYPE,
    ARBOR_EDGE_TO_RELATIONSHIP_TYPE
)
from knowledge_engine.integrations.arbor.exceptions import ArborSchemaError


class TestArborSchemaMapper:
    """Test suite for ArborSchemaMapper."""
    
    @pytest.fixture
    def mapper(self):
        """Create test mapper."""
        return ArborSchemaMapper(storage_prefix="arbor")
    
    def test_namespace_id(self, mapper):
        """Test ID namespacing."""
        namespaced = mapper.namespace_id("node_123")
        assert namespaced == "arbor:node_123"
    
    def test_extract_arbor_id(self, mapper):
        """Test extracting original Arbor ID."""
        original = mapper.extract_arbor_id("arbor:node_123")
        assert original == "node_123"
        
        # Test non-namespaced ID
        assert mapper.extract_arbor_id("node_123") == "node_123"
    
    def test_map_node_kind_known(self, mapper):
        """Test mapping known node kinds."""
        assert mapper.map_node_kind("function") == "code_function"
        assert mapper.map_node_kind("class") == "code_class"
        assert mapper.map_node_kind("method") == "code_method"
    
    def test_map_node_kind_unknown(self, mapper):
        """Test mapping unknown node kinds."""
        result = mapper.map_node_kind("unknown_kind")
        assert result == "code_unknown_kind"
    
    def test_map_edge_kind_known(self, mapper):
        """Test mapping known edge kinds."""
        assert mapper.map_edge_kind("calls") == "code_calls"
        assert mapper.map_edge_kind("imports") == "code_imports"
        assert mapper.map_edge_kind("extends") == "code_extends"
    
    def test_map_edge_kind_unknown(self, mapper):
        """Test mapping unknown edge kinds."""
        result = mapper.map_edge_kind("unknown_relation")
        assert result == "code_unknown_relation"
    
    def test_convert_arbor_node_success(self, mapper):
        """Test successful node conversion."""
        arbor_node = {
            "id": "func_001",
            "name": "authenticate",
            "kind": "function",
            "file": "/src/auth.py",
            "lineStart": 10,
            "lineEnd": 25,
            "signature": "def authenticate(user, password)",
            "visibility": "public",
            "qualifiedName": "auth.authenticate"
        }
        
        entity = mapper.convert_arbor_node(arbor_node)
        
        assert entity.entity_id == "arbor:func_001"
        assert entity.name == "authenticate"
        assert entity.entity_type == "code_function"
        assert entity.properties["arbor_kind"] == "function"
        assert entity.properties["file_path"] == "/src/auth.py"
        assert entity.properties["qualified_name"] == "auth.authenticate"
        assert entity.properties["signature"] == "def authenticate(user, password)"
        assert entity.properties["visibility"] == "public"
        assert "location" in entity.properties
        assert entity.properties["location"]["line_start"] == 10
    
    def test_convert_arbor_node_minimal(self, mapper):
        """Test conversion with minimal fields."""
        arbor_node = {
            "id": "node_1",
            "name": "simple",
            "kind": "variable"
        }
        
        entity = mapper.convert_arbor_node(arbor_node)
        
        assert entity.entity_id == "arbor:node_1"
        assert entity.name == "simple"
        assert entity.entity_type == "code_variable"
        assert entity.properties["arbor_id"] == "node_1"
    
    def test_convert_arbor_node_missing_id(self, mapper):
        """Test conversion with missing ID."""
        arbor_node = {
            "name": "invalid",
            "kind": "function"
        }
        
        with pytest.raises(ArborSchemaError) as exc_info:
            mapper.convert_arbor_node(arbor_node)
        
        assert "missing 'id' field" in str(exc_info.value)
    
    def test_convert_arbor_node_language_detection(self, mapper):
        """Test language detection from file extension."""
        test_cases = [
            ({"file": "/src/main.py"}, "python"),
            ({"file": "/src/main.rs"}, "rust"),
            ({"file": "/src/main.ts"}, "typescript"),
            ({"file": "/src/main.tsx"}, "typescript"),
            ({"file": "/src/main.js"}, "javascript"),
            ({"file": "/src/main.go"}, "go"),
            ({"file": "/src/main.java"}, "java"),
            ({"file": "/src/main.cpp"}, "cpp"),
            ({"file": "/src/main.cs"}, "csharp"),
            ({"file": "/src/main.dart"}, "dart"),
        ]
        
        for file_info, expected_lang in test_cases:
            node = {
                "id": "test",
                "name": "test",
                "kind": "function",
                **file_info
            }
            entity = mapper.convert_arbor_node(node)
            assert entity.metadata.get("language") == expected_lang, f"Failed for {file_info}"
    
    def test_convert_arbor_edge_success(self, mapper):
        """Test successful edge conversion."""
        arbor_edge = {
            "from": "func_001",
            "to": "func_002",
            "kind": "calls",
            "location": {"line": 15, "column": 10}
        }
        
        rel = mapper.convert_arbor_edge(arbor_edge)
        
        assert rel.source_id == "arbor:func_001"
        assert rel.target_id == "arbor:func_002"
        assert rel.relationship_type == "code_calls"
        assert rel.properties["arbor_kind"] == "calls"
        assert rel.properties["location"]["line"] == 15
    
    def test_convert_arbor_edge_missing_from(self, mapper):
        """Test edge conversion with missing 'from'."""
        arbor_edge = {
            "to": "func_002",
            "kind": "calls"
        }
        
        with pytest.raises(ArborSchemaError) as exc_info:
            mapper.convert_arbor_edge(arbor_edge)
        
        assert "missing 'from' or 'to' field" in str(exc_info.value)
    
    def test_convert_arbor_graph(self, mapper):
        """Test full graph conversion."""
        arbor_graph = {
            "version": "1.0",
            "nodes": [
                {"id": "1", "name": "main", "kind": "function"},
                {"id": "2", "name": "helper", "kind": "function"}
            ],
            "edges": [
                {"from": "1", "to": "2", "kind": "calls"}
            ]
        }
        
        entities, relationships = mapper.convert_arbor_graph(arbor_graph)
        
        assert len(entities) == 2
        assert len(relationships) == 1
        
        assert entities[0].entity_id == "arbor:1"
        assert entities[1].entity_id == "arbor:2"
        
        assert relationships[0].source_id == "arbor:1"
        assert relationships[0].target_id == "arbor:2"
    
    def test_convert_arbor_graph_with_errors(self, mapper):
        """Test graph conversion with invalid nodes."""
        arbor_graph = {
            "nodes": [
                {"id": "1", "name": "valid", "kind": "function"},
                {"name": "invalid", "kind": "function"},  # Missing ID
            ],
            "edges": [
                {"from": "1", "to": "2", "kind": "calls"},
                {"from": "1", "kind": "calls"},  # Missing 'to'
            ]
        }
        
        entities, relationships = mapper.convert_arbor_graph(arbor_graph)
        
        # Should still get the valid node
        assert len(entities) == 1
        assert entities[0].name == "valid"
        
        # Should get no valid edges (both reference non-existent nodes)
        assert len(relationships) == 0
    
    def test_add_custom_mapping(self, mapper):
        """Test adding custom node kind mapping."""
        mapper.add_custom_mapping("custom_kind", "my_entity_type")
        
        result = mapper.map_node_kind("custom_kind")
        assert result == "my_entity_type"
    
    def test_add_custom_edge_mapping(self, mapper):
        """Test adding custom edge mapping."""
        mapper.add_custom_edge_mapping("custom_relation", "my_rel_type")
        
        result = mapper.map_edge_kind("custom_relation")
        assert result == "my_rel_type"


class TestMappingConstants:
    """Test suite for mapping constants."""
    
    def test_arbor_kind_mappings(self):
        """Verify all expected kind mappings exist."""
        expected_kinds = [
            "function", "method", "lambda",
            "class", "struct", "enum", "interface", "trait", "type_alias",
            "module", "namespace", "package",
            "variable", "constant", "field", "property",
            "import", "export", "use",
            "macro", "decorator", "attribute",
            "comment", "docstring"
        ]
        
        for kind in expected_kinds:
            assert kind in ARBOR_KIND_TO_ENTITY_TYPE, f"Missing mapping for {kind}"
            assert ARBOR_KIND_TO_ENTITY_TYPE[kind].startswith("code_")
    
    def test_arbor_edge_mappings(self):
        """Verify all expected edge mappings exist."""
        expected_edges = [
            "calls", "called_by", "imports", "exports",
            "extends", "implements", "uses_type", "references",
            "contains", "returns", "parameter", "field_of",
            "method_of", "overrides", "implements_trait"
        ]
        
        for edge in expected_edges:
            assert edge in ARBOR_EDGE_TO_RELATIONSHIP_TYPE, f"Missing mapping for {edge}"
            assert ARBOR_EDGE_TO_RELATIONSHIP_TYPE[edge].startswith("code_")


class TestConvenienceFunctions:
    """Test convenience conversion functions."""
    
    def test_convert_arbor_node_function(self):
        """Test convert_arbor_node convenience function."""
        arbor_node = {
            "id": "test",
            "name": "test_func",
            "kind": "function",
            "file": "/test.py"
        }
        
        entity = convert_arbor_node(arbor_node, prefix="test")
        
        assert entity.entity_id == "test:test"
        assert entity.name == "test_func"
    
    def test_convert_arbor_edge_function(self):
        """Test convert_arbor_edge convenience function."""
        arbor_edge = {
            "from": "a",
            "to": "b",
            "kind": "calls"
        }
        
        rel = convert_arbor_edge(arbor_edge, prefix="test")
        
        assert rel.source_id == "test:a"
        assert rel.target_id == "test:b"
