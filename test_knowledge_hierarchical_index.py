"""
Tests for knowledge_hierarchical_index module.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime

from knowledge_hierarchical_index import (
    HierarchicalIndex,
    MemoryNode,
    ImportanceScorer,
    MemoryLevel,
    promote_demote_memories,
    create_hierarchical_index,
    convert_artifact_to_node,
)


class TestMemoryLevel:
    """Tests for MemoryLevel enum."""
    
    def test_memory_level_values(self):
        """Test that MemoryLevel has correct values."""
        assert MemoryLevel.CORE.value == 0
        assert MemoryLevel.IMPORTANT.value == 1
        assert MemoryLevel.CONTEXTUAL.value == 2
        assert MemoryLevel.GRANULAR.value == 3
    
    def test_memory_level_from_string(self):
        """Test parsing MemoryLevel from string."""
        assert MemoryLevel.from_string("core") == MemoryLevel.CORE
        assert MemoryLevel.from_string("important") == MemoryLevel.IMPORTANT
        assert MemoryLevel.from_string("CONTEXTUAL") == MemoryLevel.CONTEXTUAL
        assert MemoryLevel.from_string("Granular") == MemoryLevel.GRANULAR
    
    def test_memory_level_to_string(self):
        """Test converting MemoryLevel to string."""
        assert MemoryLevel.CORE.to_string() == "core"
        assert MemoryLevel.IMPORTANT.to_string() == "important"


class TestMemoryNode:
    """Tests for MemoryNode dataclass."""
    
    def test_memory_node_creation(self):
        """Test creating a MemoryNode."""
        node = MemoryNode(
            content="Test content",
            level=MemoryLevel.IMPORTANT,
            tags=["test", "example"],
            domain="testing"
        )
        assert node.content == "Test content"
        assert node.level == MemoryLevel.IMPORTANT
        assert node.tags == ["test", "example"]
        assert node.domain == "testing"
        assert 0.0 <= node.importance_score <= 1.0
    
    def test_memory_node_validation(self):
        """Test that scores are validated to [0, 1]."""
        node = MemoryNode(
            content="Test",
            importance_score=1.5,  # Should be clamped
            user_importance=-0.5    # Should be clamped
        )
        assert node.importance_score == 1.0
        assert node.user_importance == 0.0
    
    def test_memory_node_serialization(self):
        """Test to_dict and from_dict methods."""
        original = MemoryNode(
            content={"key": "value"},
            level=MemoryLevel.CORE,
            tags=["tag1"],
            domain="test"
        )
        
        data = original.to_dict()
        restored = MemoryNode.from_dict(data)
        
        assert restored.node_id == original.node_id
        assert restored.content == original.content
        assert restored.level == original.level
        assert restored.tags == original.tags
    
    def test_record_access(self):
        """Test access recording."""
        node = MemoryNode(content="Test")
        initial_count = node.access_count
        
        node.record_access()
        
        assert node.access_count == initial_count + 1
        assert (datetime.now() - node.last_accessed).total_seconds() < 1
    
    def test_add_remove_child(self):
        """Test child management."""
        node = MemoryNode(content="Test")
        
        node.add_child("child1")
        assert "child1" in node.child_ids
        
        node.add_child("child1")  # Duplicate should not be added
        assert node.child_ids.count("child1") == 1
        
        node.remove_child("child1")
        assert "child1" not in node.child_ids


class TestImportanceScorer:
    """Tests for ImportanceScorer."""
    
    def test_default_weights(self):
        """Test that default weights sum to 1.0."""
        scorer = ImportanceScorer()
        total = sum(scorer.weights.values())
        assert 0.99 <= total <= 1.01
    
    def test_frequency_score_calculation(self):
        """Test frequency score calculation."""
        scorer = ImportanceScorer()
        
        score_0 = scorer.calculate_frequency_score(0)
        score_10 = scorer.calculate_frequency_score(10)
        score_100 = scorer.calculate_frequency_score(100)
        
        assert 0.0 <= score_0 <= 1.0
        assert score_10 > score_0
        assert score_100 >= score_10
        assert score_100 <= 1.0
    
    def test_centrality_score_calculation(self):
        """Test centrality score calculation."""
        scorer = ImportanceScorer()
        
        score = scorer.calculate_centrality_score(5, 100)
        assert 0.0 <= score <= 1.0
        
        # More connections should give higher score
        score_0 = scorer.calculate_centrality_score(0, 100)
        score_10 = scorer.calculate_centrality_score(10, 100)
        assert score_10 >= score_0
    
    def test_decay_factor_calculation(self):
        """Test decay factor calculation."""
        scorer = ImportanceScorer()
        
        fresh = scorer.calculate_decay_factor(0)
        old = scorer.calculate_decay_factor(100)
        
        assert fresh > old
        assert 0.1 <= old <= 1.0
    
    def test_semantic_density_calculation(self):
        """Test semantic density calculation."""
        scorer = ImportanceScorer()
        
        # Short content should have lower density
        short = scorer.calculate_semantic_density("Hi")
        # Long content with structure should have higher density
        long_structured = scorer.calculate_semantic_density(
            "Implementation: 5 classes, 3 interfaces, 2 modules"
        )
        
        assert 0.0 <= short <= 1.0
        assert 0.0 <= long_structured <= 1.0


class TestHierarchicalIndex:
    """Tests for HierarchicalIndex."""
    
    @pytest.fixture
    def index(self):
        """Create an in-memory index for testing."""
        return HierarchicalIndex(storage_path=":memory:", use_sqlite=False)
    
    def test_index_creation(self, index):
        """Test creating an index."""
        assert len(index.nodes) == 0
        stats = index.get_statistics()
        assert stats["total_nodes"] == 0
    
    def test_add_memory(self, index):
        """Test adding memories."""
        node = index.add_memory(
            content="Test memory",
            level=MemoryLevel.CORE,
            tags=["test"],
            domain="testing"
        )
        
        assert node.node_id in index.nodes
        assert node.level == MemoryLevel.CORE
        assert node.domain == "testing"
    
    def test_get_memory(self, index):
        """Test retrieving memories."""
        node = index.add_memory(content="Test")
        
        # Get without recording access
        retrieved = index.get_memory(node.node_id, record_access=False)
        assert retrieved.node_id == node.node_id
        assert retrieved.access_count == 0
        
        # Get with recording access
        retrieved = index.get_memory(node.node_id, record_access=True)
        assert retrieved.access_count == 1
    
    def test_query_by_level(self, index):
        """Test querying by level."""
        index.add_memory(content="Core", level=MemoryLevel.CORE)
        index.add_memory(content="Important", level=MemoryLevel.IMPORTANT)
        index.add_memory(content="Important2", level=MemoryLevel.IMPORTANT)
        
        core_results = index.query_by_level(MemoryLevel.CORE)
        important_results = index.query_by_level(MemoryLevel.IMPORTANT)
        
        assert len(core_results) == 1
        assert len(important_results) == 2
    
    def test_query_by_domain(self, index):
        """Test querying by domain."""
        index.add_memory(content="A", domain="domain1")
        index.add_memory(content="B", domain="domain1")
        index.add_memory(content="C", domain="domain2")
        
        results = index.query_by_domain("domain1")
        assert len(results) == 2
    
    def test_query_by_tags(self, index):
        """Test querying by tags."""
        index.add_memory(content="A", tags=["tag1", "tag2"])
        index.add_memory(content="B", tags=["tag2", "tag3"])
        index.add_memory(content="C", tags=["tag3"])
        
        # Match any
        any_results = index.query_by_tags(["tag1", "tag3"], match_all=False)
        assert len(any_results) == 3
        
        # Match all
        all_results = index.query_by_tags(["tag2", "tag3"], match_all=True)
        assert len(all_results) == 1
    
    def test_update_memory(self, index):
        """Test updating memories."""
        node = index.add_memory(content="Original", user_importance=0.5)
        
        updated = index.update_memory(node.node_id, user_importance=0.9)
        
        assert updated.user_importance == 0.9
    
    def test_delete_memory(self, index):
        """Test deleting memories."""
        node = index.add_memory(content="To delete")
        node_id = node.node_id
        
        result = index.delete_memory(node_id)
        assert result is True
        assert node_id not in index.nodes
        
        # Deleting non-existent should return False
        result = index.delete_memory(node_id)
        assert result is False
    
    def test_promote_demote_memories(self, index):
        """Test promotion/demotion logic."""
        # Add memory with low importance to GRANULAR
        node = index.add_memory(
            content="Low importance",
            level=MemoryLevel.GRANULAR,
            user_importance=0.1
        )
        
        # Recalculate to get accurate scores
        index.recalculate_importance()
        
        # Check for changes
        changes = index.promote_demote_memories(dry_run=True)
        
        assert isinstance(changes, dict)
        assert "promotions" in changes
        assert "demotions" in changes
    
    def test_promote_node(self, index):
        """Test manual promotion."""
        node = index.add_memory(content="Test", level=MemoryLevel.CONTEXTUAL)
        
        promoted = index.promote_node(node.node_id, levels=1)
        
        assert promoted.level == MemoryLevel.IMPORTANT
    
    def test_demote_node(self, index):
        """Test manual demotion."""
        node = index.add_memory(content="Test", level=MemoryLevel.IMPORTANT)
        
        demoted = index.demote_node(node.node_id, levels=1)
        
        assert demoted.level == MemoryLevel.CONTEXTUAL
    
    def test_tree_structure(self, index):
        """Test tree structure retrieval."""
        parent = index.add_memory(content="Parent")
        child = index.add_memory(content="Child", parent_id=parent.node_id)
        
        tree = index.get_tree_structure()
        
        assert "roots" in tree
        assert tree["total_nodes"] == 2
    
    def test_statistics(self, index):
        """Test statistics generation."""
        index.add_memory(content="A", level=MemoryLevel.CORE, domain="d1")
        index.add_memory(content="B", level=MemoryLevel.GRANULAR, domain="d2")
        
        stats = index.get_statistics()
        
        assert stats["total_nodes"] == 2
        assert stats["by_level"]["CORE"] == 1
        assert stats["by_level"]["GRANULAR"] == 1
        assert "average_importance" in stats


class TestHierarchicalIndexPersistence:
    """Tests for persistence functionality."""
    
    def test_json_export_import(self):
        """Test JSON export and import."""
        index = HierarchicalIndex(storage_path=":memory:", use_sqlite=False)
        index.add_memory(content="Test1", level=MemoryLevel.CORE)
        index.add_memory(content="Test2", level=MemoryLevel.GRANULAR)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Export
            index.export_to_json(temp_path)
            assert os.path.exists(temp_path)
            
            # Import into new index
            new_index = HierarchicalIndex(storage_path=":memory:", use_sqlite=False)
            count = new_index.import_from_json(temp_path)
            
            assert count == 2
            assert new_index.get_statistics()["total_nodes"] == 2
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_file_based_sqlite(self):
        """Test file-based SQLite storage."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Create and populate index
            index1 = HierarchicalIndex(storage_path=db_path, use_sqlite=True)
            index1.add_memory(content="Test", level=MemoryLevel.CORE)
            
            # Create new index pointing to same file
            index2 = HierarchicalIndex(storage_path=db_path, use_sqlite=True)
            stats = index2.get_statistics()
            
            assert stats["total_nodes"] == 1
        finally:
            if os.path.exists(db_path):
                try:
                    os.unlink(db_path)
                except PermissionError:
                    pass  # Windows may hold the file


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_hierarchical_index(self):
        """Test factory function."""
        index = create_hierarchical_index(storage_path=":memory:", use_sqlite=False)
        assert isinstance(index, HierarchicalIndex)
    
    def test_promote_demote_memories_function(self):
        """Test standalone promote_demote function."""
        index = HierarchicalIndex(storage_path=":memory:", use_sqlite=False)
        index.add_memory(content="Test")
        
        changes = promote_demote_memories(index, dry_run=True)
        
        assert "promotions" in changes
        assert "demotions" in changes


class TestConvertArtifactToNode:
    """Tests for artifact conversion."""
    
    def test_convert_simple_artifact(self):
        """Test converting a simple artifact-like object."""
        class FakeArtifact:
            id = "test-id"
            content = "Test content"
            domain = "test-domain"
            tags = ["tag1"]
            timestamp = datetime.now()
            effectiveness_score = 0.8
        
        node = convert_artifact_to_node(FakeArtifact(), MemoryLevel.IMPORTANT)
        
        assert node.node_id == "test-id"
        assert node.content == "Test content"
        assert node.level == MemoryLevel.IMPORTANT
        assert node.domain == "test-domain"
    
    def test_convert_artifact_with_alt_fields(self):
        """Test converting artifact with alternative field names."""
        class FakeArtifact:
            artifact_id = "alt-id"
            content = {"key": "value"}
            domain = None
            tags = None
            created_at = None
            confidence = 0.7
        
        node = convert_artifact_to_node(FakeArtifact())
        
        assert node.node_id == "alt-id"
        assert node.user_importance == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
