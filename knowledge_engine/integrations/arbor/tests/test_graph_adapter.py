"""
Tests for Arbor Graph Adapter

Following CLAUDE.md principles:
- CONTRACT TESTS: Verify adapter contracts
- ISOLATION: Mock Knowledge Engine
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock

from knowledge_engine.integrations.arbor import (
    ArborGraphAdapter,
    ArborClient,
    ArborConfig,
    MergeResult,
    GraphDelta
)
from knowledge_engine.integrations.arbor.exceptions import ArborSyncError
from knowledge_engine.schemas.base import Entity, Relationship


class TestMergeResult:
    """Test suite for MergeResult dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        result = MergeResult()
        
        assert not result.success
        assert result.nodes_imported == 0
        assert result.nodes_updated == 0
        assert result.nodes_skipped == 0
        assert result.edges_imported == 0
        assert result.edges_updated == 0
        assert result.edges_skipped == 0
        assert result.errors == []
        assert result.duration_seconds == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = MergeResult(
            success=True,
            nodes_imported=10,
            nodes_updated=5,
            edges_imported=20,
            duration_seconds=1.5
        )
        
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["nodes_imported"] == 10
        assert data["nodes_updated"] == 5
        assert data["edges_imported"] == 20
        assert data["duration_seconds"] == 1.5


class TestGraphDelta:
    """Test suite for GraphDelta dataclass."""
    
    def test_default_values(self):
        """Test default values."""
        delta = GraphDelta()
        
        assert delta.added_nodes == []
        assert delta.updated_nodes == []
        assert delta.removed_nodes == []
        assert delta.added_edges == []
        assert delta.removed_edges == []
    
    def test_has_changes_true(self):
        """Test has_changes with actual changes."""
        entity = Mock(spec=Entity)
        delta = GraphDelta(added_nodes=[entity])
        
        assert delta.has_changes
    
    def test_has_changes_false(self):
        """Test has_changes with no changes."""
        delta = GraphDelta()
        
        assert not delta.has_changes


class TestArborGraphAdapter:
    """Test suite for ArborGraphAdapter."""
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph."""
        kg = Mock()
        kg.add_entity_async = AsyncMock(return_value=True)
        kg.add_relationship_async = AsyncMock(return_value=True)
        kg.get_entity_async = AsyncMock(return_value=None)
        kg._entities = {}
        return kg
    
    @pytest.fixture
    def mock_arbor_client(self):
        """Create mock Arbor client."""
        client = Mock(spec=ArborClient)
        client.config = ArborConfig()
        return client
    
    @pytest.fixture
    def adapter(self, mock_knowledge_graph, mock_arbor_client):
        """Create test adapter."""
        return ArborGraphAdapter(
            knowledge_graph=mock_knowledge_graph,
            arbor_client=mock_arbor_client,
            storage_prefix="arbor"
        )
    
    @pytest.mark.asyncio
    async def test_merge_arbor_graph_success(self, adapter, mock_knowledge_graph):
        """Test successful graph merge."""
        arbor_graph = {
            "nodes": [
                {"id": "1", "name": "main", "kind": "function", "file": "/src/main.py"},
                {"id": "2", "name": "helper", "kind": "function", "file": "/src/main.py"}
            ],
            "edges": [
                {"from": "1", "to": "2", "kind": "calls"}
            ]
        }
        
        result = await adapter.merge_arbor_graph(arbor_graph)
        
        assert isinstance(result, MergeResult)
        assert result.success
        assert result.nodes_imported == 2
        assert result.edges_imported == 1
        assert result.duration_seconds > 0
        
        # Verify KG methods were called
        assert mock_knowledge_graph.add_entity_async.call_count == 2
        assert mock_knowledge_graph.add_relationship_async.call_count == 1
    
    @pytest.mark.asyncio
    async def test_merge_arbor_graph_with_updates(self, adapter, mock_knowledge_graph):
        """Test graph merge with existing entities."""
        # First entity already exists
        existing_entity = {"entity_id": "arbor:1", "name": "main"}
        mock_knowledge_graph.get_entity_async = AsyncMock(return_value=existing_entity)
        
        arbor_graph = {
            "nodes": [
                {"id": "1", "name": "main", "kind": "function"},
                {"id": "2", "name": "new_func", "kind": "function"}
            ],
            "edges": []
        }
        
        result = await adapter.merge_arbor_graph(arbor_graph)
        
        assert result.nodes_imported == 1  # Only the new one
        assert result.nodes_updated == 1   # The existing one
    
    @pytest.mark.asyncio
    async def test_merge_arbor_graph_empty(self, adapter):
        """Test merging empty graph."""
        arbor_graph = {"nodes": [], "edges": []}
        
        result = await adapter.merge_arbor_graph(arbor_graph)
        
        assert result.success
        assert result.nodes_imported == 0
        assert result.edges_imported == 0
    
    @pytest.mark.asyncio
    async def test_merge_arbor_graph_error_handling(self, adapter, mock_knowledge_graph):
        """Test error handling during merge."""
        # Make add_entity_async fail for second entity
        mock_knowledge_graph.add_entity_async = AsyncMock(side_effect=[
            True, Exception("DB error")
        ])
        
        arbor_graph = {
            "nodes": [
                {"id": "1", "name": "func1", "kind": "function"},
                {"id": "2", "name": "func2", "kind": "function"}
            ],
            "edges": []
        }
        
        result = await adapter.merge_arbor_graph(arbor_graph)
        
        assert not result.success
        assert result.nodes_imported == 1
        assert result.nodes_skipped == 1
        assert len(result.errors) == 1
        assert "DB error" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_apply_delta_add_nodes(self, adapter, mock_knowledge_graph):
        """Test applying delta with added nodes."""
        entity = Entity(
            entity_id="arbor:new",
            name="new_func",
            entity_type="code_function",
            properties={"arbor_id": "new"}
        )
        
        delta = GraphDelta(added_nodes=[entity])
        
        await adapter.apply_delta(delta)
        
        mock_knowledge_graph.add_entity_async.assert_called_once_with(
            name="arbor:new",
            entity_type="code_function",
            attributes=entity.properties
        )
    
    @pytest.mark.asyncio
    async def test_apply_delta_remove_nodes(self, adapter, mock_knowledge_graph):
        """Test applying delta with removed nodes."""
        mock_knowledge_graph.get_entity_async = AsyncMock(return_value={
            "properties": {}
        })
        
        delta = GraphDelta(removed_nodes=["arbor:old"])
        
        await adapter.apply_delta(delta)
        
        mock_knowledge_graph.get_entity_async.assert_called_once_with("arbor:old")
    
    @pytest.mark.asyncio
    async def test_apply_delta_add_edges(self, adapter, mock_knowledge_graph):
        """Test applying delta with added edges."""
        rel = Relationship(
            source_id="arbor:a",
            target_id="arbor:b",
            relationship_type="code_calls",
            properties={}
        )
        
        delta = GraphDelta(added_edges=[rel])
        
        await adapter.apply_delta(delta)
        
        mock_knowledge_graph.add_relationship_async.assert_called_once_with(
            source="arbor:a",
            target="arbor:b",
            relation_type="code_calls",
            attributes={}
        )
    
    @pytest.mark.asyncio
    async def test_handle_arbor_change_event_file_added(self, adapter, mock_knowledge_graph):
        """Test handling file added event."""
        event = {
            "type": "file_added",
            "nodes": [
                {"id": "1", "name": "new_func", "kind": "function"}
            ],
            "edges": []
        }
        
        await adapter.handle_arbor_change_event(event)
        
        mock_knowledge_graph.add_entity_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_arbor_change_event_file_removed(self, adapter, mock_knowledge_graph):
        """Test handling file removed event."""
        mock_knowledge_graph.get_entity_async = AsyncMock(return_value={
            "properties": {}
        })
        
        event = {
            "type": "file_removed",
            "node_ids": ["1", "2"]
        }
        
        await adapter.handle_arbor_change_event(event)
        
        assert mock_knowledge_graph.get_entity_async.call_count == 2
    
    @pytest.mark.asyncio
    async def test_handle_arbor_change_event_unknown_type(self, adapter):
        """Test handling unknown event type."""
        event = {"type": "unknown_event"}
        
        # Should not raise
        await adapter.handle_arbor_change_event(event)
    
    @pytest.mark.asyncio
    async def test_handle_arbor_change_event_error(self, adapter):
        """Test handling event processing error."""
        event = {"type": "file_added"}  # Missing nodes field
        
        with pytest.raises(ArborSyncError):
            await adapter.handle_arbor_change_event(event)
    
    def test_create_delta_from_arbor_export_full_import(self, adapter):
        """Test creating delta for full import."""
        new_graph = {
            "nodes": [
                {"id": "1", "name": "func1", "kind": "function"}
            ],
            "edges": [
                {"from": "1", "to": "2", "kind": "calls"}
            ]
        }
        
        delta = adapter.create_delta_from_arbor_export(None, new_graph)
        
        assert len(delta.added_nodes) == 1
        assert len(delta.added_edges) == 1
        assert not delta.removed_nodes
        assert not delta.updated_nodes
    
    def test_create_delta_from_arbor_export_incremental(self, adapter):
        """Test creating delta for incremental update."""
        old_graph = {
            "nodes": [
                {"id": "1", "name": "old_func", "kind": "function"},
                {"id": "2", "name": "keep_func", "kind": "function"}
            ],
            "edges": [
                {"from": "1", "to": "2", "kind": "calls"}
            ]
        }
        
        new_graph = {
            "nodes": [
                {"id": "2", "name": "keep_func", "kind": "function"},
                {"id": "3", "name": "new_func", "kind": "function"}
            ],
            "edges": [
                {"from": "2", "to": "3", "kind": "calls"}
            ]
        }
        
        delta = adapter.create_delta_from_arbor_export(old_graph, new_graph)
        
        # Node 1 removed, Node 3 added
        assert len(delta.removed_nodes) == 1
        assert "arbor:1" in delta.removed_nodes
        assert len(delta.added_nodes) == 1
        
        # Old edge removed, new edge added
        assert len(delta.removed_edges) == 1
        assert ("1", "2") in delta.removed_edges
        assert len(delta.added_edges) == 1
    
    def test_get_imported_arbor_ids(self, adapter):
        """Test getting imported Arbor IDs."""
        # Initially empty
        assert adapter.get_imported_arbor_ids() == set()
        
        # After adding some
        adapter._imported_arbor_ids.add("node_1")
        adapter._imported_arbor_ids.add("node_2")
        
        ids = adapter.get_imported_arbor_ids()
        assert ids == {"node_1", "node_2"}
        
        # Verify it's a copy
        ids.add("node_3")
        assert "node_3" not in adapter._imported_arbor_ids


class TestArborGraphAdapterQueries:
    """Test suite for adapter query methods."""
    
    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph with entities."""
        kg = Mock()
        
        # Create some test entities
        entity1 = Entity(
            entity_id="arbor:func1",
            name="func1",
            entity_type="code_function",
            properties={"file_path": "/src/main.py", "arbor_kind": "function"},
            metadata={"language": "python"}
        )
        entity2 = Entity(
            entity_id="arbor:class1",
            name="class1",
            entity_type="code_class",
            properties={"file_path": "/src/models.py", "arbor_kind": "class"},
            metadata={"language": "python"}
        )
        entity3 = Entity(
            entity_id="other:entity",
            name="other",
            entity_type="other_type",
            properties={}
        )
        
        kg._entities = {
            "arbor:func1": entity1,
            "arbor:class1": entity2,
            "other:entity": entity3
        }
        
        return kg
    
    @pytest.fixture
    def adapter(self, mock_knowledge_graph):
        """Create test adapter."""
        return ArborGraphAdapter(
            knowledge_graph=mock_knowledge_graph,
            storage_prefix="arbor"
        )
    
    @pytest.mark.asyncio
    async def test_get_code_entities_all(self, adapter):
        """Test getting all code entities."""
        results = await adapter.get_code_entities()
        
        # Should return only arbor-prefixed entities
        assert len(results) == 2
        assert all(e.entity_id.startswith("arbor:") for e in results)
    
    @pytest.mark.asyncio
    async def test_get_code_entities_by_file(self, adapter):
        """Test filtering by file path."""
        results = await adapter.get_code_entities(file_path="/src/main.py")
        
        assert len(results) == 1
        assert results[0].name == "func1"
    
    @pytest.mark.asyncio
    async def test_get_code_entities_by_type(self, adapter):
        """Test filtering by entity type."""
        results = await adapter.get_code_entities(entity_type="code_class")
        
        assert len(results) == 1
        assert results[0].name == "class1"
    
    @pytest.mark.asyncio
    async def test_get_code_entities_by_language(self, adapter):
        """Test filtering by language."""
        results = await adapter.get_code_entities(language="python")
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_get_code_entities_no_matches(self, adapter):
        """Test query with no matches."""
        results = await adapter.get_code_entities(language="rust")
        
        assert len(results) == 0
