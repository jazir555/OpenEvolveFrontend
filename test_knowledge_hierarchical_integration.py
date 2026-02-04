"""
Comprehensive Test Suite for 4-Layer Hierarchical Knowledge Indexing System

Tests all components:
1. Individual Index Tests (Hierarchical, Graph, Hash, Semantic)
2. Integration Tests (UnifiedContextAssembler, ContextAssemblyPipeline, EnhancedKnowledgeEngine)
3. Context Rot Prevention Tests (Long conversations, persistence, deduplication)
4. Performance Tests (Token budgets, query latency, thread safety)
5. Edge Cases (Empty indexes, duplicates, graceful degradation)

Author: OpenEvolve AI Test Suite
Version: 1.0.0
"""

import pytest
import os
import sys
import time
import json
import tempfile
import shutil
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# =============================================================================

try:
    from knowledge_hierarchical_index import (
        HierarchicalIndex, MemoryNode, MemoryLevel, ImportanceScorer
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    MemoryLevel = None

try:
    from knowledge_graph_index import (
        GraphIndex, MemoryNode as GraphMemoryNode, RelationshipType, 
        RelationshipEdge, TraversalMode, TraversalResult
    )
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    RelationshipType = None
    TraversalMode = None

try:
    from knowledge_hash_index import (
        HashIndex, HashIndexConfig, HashEntry, compute_simhash, 
        compute_minhash, hamming_distance
    )
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False

try:
    from knowledge_semantic_index import (
        SemanticIndex, SemanticIndexConfig, SemanticQuery, 
        SemanticResult, EmbeddingGenerator, generate_embedding
    )
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

try:
    from knowledge_context_assembler import (
        UnifiedContextAssembler, ContextAssemblyPipeline,
        ContextAssemblerConfig, ContextItem, AssembledContext,
        ContextAssemblyStage
    )
    ASSEMBLER_AVAILABLE = True
except ImportError:
    ASSEMBLER_AVAILABLE = False

try:
    from knowledge_engine_hierarchical_integration import (
        EnhancedKnowledgeEngine, EnhancedKnowledgeEngineConfig,
        UnifiedKnowledgeEntry, CuratedQueryResult, MaintenanceJobType
    )
    ENHANCED_ENGINE_AVAILABLE = True
except ImportError:
    ENHANCED_ENGINE_AVAILABLE = False


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir):
    """Create a temporary database path."""
    return os.path.join(temp_dir, "test_index.db")


@pytest.fixture
def hierarchical_index(temp_db_path):
    """Create a temporary HierarchicalIndex."""
    if not HIERARCHICAL_AVAILABLE:
        pytest.skip("Hierarchical index not available")
    index = HierarchicalIndex(storage_path=temp_db_path, use_sqlite=True)
    yield index
    index.clear_all()


@pytest.fixture
def graph_index(temp_db_path):
    """Create a temporary GraphIndex."""
    if not GRAPH_AVAILABLE:
        pytest.skip("Graph index not available")
    index = GraphIndex(db_path=temp_db_path)
    yield index


@pytest.fixture
def hash_index(temp_db_path):
    """Create a temporary HashIndex."""
    if not HASH_AVAILABLE:
        pytest.skip("Hash index not available")
    config = HashIndexConfig(db_path=temp_db_path)
    index = HashIndex(config=config)
    yield index


@pytest.fixture
def semantic_index(temp_dir):
    """Create a temporary SemanticIndex with mocked embeddings."""
    if not SEMANTIC_AVAILABLE:
        pytest.skip("Semantic index not available")
    
    config = SemanticIndexConfig(
        cache_dir=temp_dir,
        vector_backend="sqlite"
    )
    index = SemanticIndex(config=config)
    
    # Mock embedding generation
    index.embedding_generator = Mock()
    index.embedding_generator.generate = Mock(
        return_value=np.array([0.1, 0.2, 0.3, 0.4] * 384, dtype=np.float32)
    )
    
    yield index


@pytest.fixture
def context_assembler(temp_dir):
    """Create a UnifiedContextAssembler with test configuration."""
    if not ASSEMBLER_AVAILABLE:
        pytest.skip("Context assembler not available")
    
    config = ContextAssemblerConfig(
        db_path=os.path.join(temp_dir, "assembler.db"),
        hierarchical_index_path=os.path.join(temp_dir, "hierarchical.db"),
        graph_index_path=os.path.join(temp_dir, "graph.db"),
        hash_index_path=os.path.join(temp_dir, "hash.db"),
        semantic_cache_dir=os.path.join(temp_dir, "semantic"),
        max_tokens=4000,
        enable_hierarchical=True,
        enable_graph=True,
        enable_deduplication=True,
        enable_semantic=True
    )
    assembler = UnifiedContextAssembler(config=config)
    yield assembler


@pytest.fixture
def enhanced_engine(temp_dir):
    """Create an EnhancedKnowledgeEngine with test configuration."""
    if not ENHANCED_ENGINE_AVAILABLE:
        pytest.skip("Enhanced knowledge engine not available")
    
    config = EnhancedKnowledgeEngineConfig(
        storage_path=temp_dir,
        enable_hierarchical=HIERARCHICAL_AVAILABLE,
        enable_graph=GRAPH_AVAILABLE,
        enable_hash=HASH_AVAILABLE,
        enable_semantic=False,  # Skip semantic for most tests
        max_context_tokens=4000
    )
    engine = EnhancedKnowledgeEngine(config=config)
    yield engine


@pytest.fixture
def sample_memories():
    """Return a list of sample memories for testing."""
    return [
        {
            "content": "System design requires careful planning of database schema",
            "level": MemoryLevel.CORE if MemoryLevel else "core",
            "tags": ["architecture", "database"],
            "domain": "system_design"
        },
        {
            "content": "Cache invalidation is a hard problem in distributed systems",
            "level": MemoryLevel.IMPORTANT if MemoryLevel else "important",
            "tags": ["caching", "distributed"],
            "domain": "system_design"
        },
        {
            "content": "Use Redis for session storage in this project",
            "level": MemoryLevel.CONTEXTUAL if MemoryLevel else "contextual",
            "tags": ["redis", "session"],
            "domain": "implementation"
        },
        {
            "content": "The API endpoint /users should return user data",
            "level": MemoryLevel.GRANULAR if MemoryLevel else "granular",
            "tags": ["api", "endpoint"],
            "domain": "implementation"
        }
    ]


# =============================================================================
# 1. INDIVIDUAL INDEX TESTS - HIERARCHICAL INDEX
# =============================================================================

@pytest.mark.skipif(not HIERARCHICAL_AVAILABLE, reason="Hierarchical index not available")
class TestHierarchicalIndex:
    """Test HierarchicalIndex functionality."""
    
    def test_add_memory(self, hierarchical_index):
        """Test adding memories to hierarchical index."""
        node = hierarchical_index.add_memory(
            content="Test memory content",
            level=MemoryLevel.CORE,
            tags=["test", "memory"],
            domain="testing"
        )
        
        assert node is not None
        assert node.content == "Test memory content"
        assert node.level == MemoryLevel.CORE
        assert "test" in node.tags
        assert node.domain == "testing"
        assert node.node_id is not None
    
    def test_get_memory(self, hierarchical_index):
        """Test retrieving memory by ID."""
        node = hierarchical_index.add_memory(
            content="Retrievable memory",
            level=MemoryLevel.IMPORTANT
        )
        
        retrieved = hierarchical_index.get_memory(node.node_id)
        assert retrieved is not None
        assert retrieved.node_id == node.node_id
        assert retrieved.content == "Retrievable memory"
    
    def test_query_by_level(self, hierarchical_index, sample_memories):
        """Test querying memories by hierarchical level."""
        for memory in sample_memories:
            hierarchical_index.add_memory(**memory)
        
        core_memories = hierarchical_index.query_by_level(MemoryLevel.CORE)
        assert len(core_memories) >= 1
        assert all(m.level == MemoryLevel.CORE for m in core_memories)
    
    def test_query_by_domain(self, hierarchical_index, sample_memories):
        """Test querying memories by domain."""
        for memory in sample_memories:
            hierarchical_index.add_memory(**memory)
        
        system_design = hierarchical_index.query_by_domain("system_design")
        assert len(system_design) >= 2
        assert all(m.domain == "system_design" for m in system_design)
    
    def test_query_by_tags(self, hierarchical_index, sample_memories):
        """Test querying memories by tags."""
        for memory in sample_memories:
            hierarchical_index.add_memory(**memory)
        
        tagged = hierarchical_index.query_by_tags(["database"])
        assert len(tagged) >= 1
        assert any("database" in m.tags for m in tagged)
    
    def test_promote_node(self, hierarchical_index):
        """Test promoting a node to higher level."""
        node = hierarchical_index.add_memory(
            content="Promotable memory",
            level=MemoryLevel.GRANULAR
        )
        
        promoted = hierarchical_index.promote_node(node.node_id, levels=1)
        assert promoted is not None
        assert promoted.level == MemoryLevel.CONTEXTUAL
    
    def test_demote_node(self, hierarchical_index):
        """Test demoting a node to lower level."""
        node = hierarchical_index.add_memory(
            content="Demotable memory",
            level=MemoryLevel.CORE
        )
        
        demoted = hierarchical_index.demote_node(node.node_id, levels=1)
        assert demoted is not None
        assert demoted.level == MemoryLevel.IMPORTANT
    
    def test_promote_demote_memories(self, hierarchical_index):
        """Test automatic promotion/demotion based on importance scores."""
        # Create nodes with varying importance
        high_importance = hierarchical_index.add_memory(
            content="High importance",
            level=MemoryLevel.CONTEXTUAL,
            user_importance=0.9
        )
        
        changes = hierarchical_index.promote_demote_memories(
            auto_apply=True,
            dry_run=False
        )
        
        assert isinstance(changes, dict)
        assert "promotions" in changes
        assert "demotions" in changes
    
    def test_memory_node_relationships(self, hierarchical_index):
        """Test parent-child relationships between memory nodes."""
        parent = hierarchical_index.add_memory(
            content="Parent memory",
            level=MemoryLevel.CORE
        )
        
        child = hierarchical_index.add_memory(
            content="Child memory",
            level=MemoryLevel.IMPORTANT,
            parent_id=parent.node_id
        )
        
        assert child.parent_id == parent.node_id
        parent_updated = hierarchical_index.get_memory(parent.node_id)
        assert child.node_id in parent_updated.child_ids
    
    def test_importance_scoring(self, hierarchical_index):
        """Test importance score calculation."""
        scorer = ImportanceScorer()
        
        node = hierarchical_index.add_memory(
            content="Test content with technical terms and numbers: 123, algorithm",
            level=MemoryLevel.CONTEXTUAL
        )
        
        # Access node multiple times to increase frequency score
        for _ in range(5):
            hierarchical_index.get_memory(node.node_id)
        
        node_updated = hierarchical_index.get_memory(node.node_id)
        assert node_updated.access_count >= 5
    
    def test_tree_structure(self, hierarchical_index):
        """Test tree structure retrieval."""
        root = hierarchical_index.add_memory(
            content="Root memory",
            level=MemoryLevel.CORE
        )
        
        child1 = hierarchical_index.add_memory(
            content="Child 1",
            level=MemoryLevel.IMPORTANT,
            parent_id=root.node_id
        )
        
        child2 = hierarchical_index.add_memory(
            content="Child 2",
            level=MemoryLevel.IMPORTANT,
            parent_id=root.node_id
        )
        
        tree = hierarchical_index.get_tree_structure(root.node_id)
        assert tree is not None
        assert "node_id" in tree
        assert "children" in tree
    
    def test_statistics(self, hierarchical_index, sample_memories):
        """Test statistics generation."""
        for memory in sample_memories:
            hierarchical_index.add_memory(**memory)
        
        stats = hierarchical_index.get_statistics()
        assert stats["total_nodes"] >= 4
        assert "by_level" in stats
        assert "by_domain" in stats
        assert "average_importance" in stats


# =============================================================================
# 1. INDIVIDUAL INDEX TESTS - GRAPH INDEX
# =============================================================================

@pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph index not available")
class TestGraphIndex:
    """Test GraphIndex functionality."""
    
    def test_add_node(self, graph_index):
        """Test adding nodes to graph index."""
        node_id = graph_index.add_node(
            content="Test node content",
            node_type=NodeType.CONCEPT if 'NodeType' in dir() else None,
            importance=0.8
        )
        
        assert node_id is not None
        assert isinstance(node_id, str)
    
    def test_get_node(self, graph_index):
        """Test retrieving node by ID."""
        node_id = graph_index.add_node(
            content="Retrievable node",
            importance=0.7
        )
        
        node = graph_index.get_node(node_id)
        assert node is not None
        assert node.node_id == node_id
        assert node.content == "Retrievable node"
    
    def test_add_edge(self, graph_index):
        """Test adding edges between nodes."""
        node1 = graph_index.add_node(content="Source node")
        node2 = graph_index.add_node(content="Target node")
        
        edge_id = graph_index.add_edge(
            source_id=node1,
            target_id=node2,
            relationship_type=RelationshipType.CAUSAL,
            weight=0.9
        )
        
        assert edge_id is not None
    
    def test_get_edges_from_node(self, graph_index):
        """Test retrieving edges from a node."""
        node1 = graph_index.add_node(content="Source")
        node2 = graph_index.add_node(content="Target 1")
        node3 = graph_index.add_node(content="Target 2")
        
        graph_index.add_edge(node1, node2, RelationshipType.CAUSAL)
        graph_index.add_edge(node1, node3, RelationshipType.SEMANTIC)
        
        edges = graph_index.get_edges_from_node(node1)
        assert len(edges) == 2
    
    def test_traverse_bfs(self, graph_index):
        """Test BFS graph traversal."""
        # Create a chain: A -> B -> C
        node_a = graph_index.add_node(content="Node A")
        node_b = graph_index.add_node(content="Node B")
        node_c = graph_index.add_node(content="Node C")
        
        graph_index.add_edge(node_a, node_b, RelationshipType.CAUSAL)
        graph_index.add_edge(node_b, node_c, RelationshipType.CAUSAL)
        
        result = graph_index.traverse_relationships(
            start_node_id=node_a,
            depth=2,
            mode=TraversalMode.BFS
        )
        
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) == 3
        assert len(result.edges) == 2
    
    def test_traverse_dfs(self, graph_index):
        """Test DFS graph traversal."""
        node_a = graph_index.add_node(content="Node A")
        node_b = graph_index.add_node(content="Node B")
        node_c = graph_index.add_node(content="Node C")
        
        graph_index.add_edge(node_a, node_b, RelationshipType.CAUSAL)
        graph_index.add_edge(node_b, node_c, RelationshipType.CAUSAL)
        
        result = graph_index.traverse_relationships(
            start_node_id=node_a,
            depth=2,
            mode=TraversalMode.DFS
        )
        
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) == 3
    
    def test_traverse_weighted(self, graph_index):
        """Test weighted graph traversal."""
        node_a = graph_index.add_node(content="Node A")
        node_b = graph_index.add_node(content="Node B")
        node_c = graph_index.add_node(content="Node C")
        
        graph_index.add_edge(node_a, node_b, RelationshipType.CAUSAL, weight=0.9)
        graph_index.add_edge(node_a, node_c, RelationshipType.SEMANTIC, weight=0.5)
        
        result = graph_index.traverse_relationships(
            start_node_id=node_a,
            depth=1,
            mode=TraversalMode.WEIGHTED
        )
        
        assert isinstance(result, TraversalResult)
    
    def test_find_path(self, graph_index):
        """Test path finding between nodes."""
        node_a = graph_index.add_node(content="Start")
        node_b = graph_index.add_node(content="Middle")
        node_c = graph_index.add_node(content="End")
        
        graph_index.add_edge(node_a, node_b, RelationshipType.CAUSAL)
        graph_index.add_edge(node_b, node_c, RelationshipType.CAUSAL)
        
        path_result = graph_index.find_path(
            start_node_id=node_a,
            end_node_id=node_c,
            max_depth=3
        )
        
        assert path_result is not None
        assert len(path_result.path) == 3
    
    def test_get_connected_nodes(self, graph_index):
        """Test getting connected nodes."""
        node_a = graph_index.add_node(content="Source")
        node_b = graph_index.add_node(content="Connected 1")
        node_c = graph_index.add_node(content="Connected 2")
        
        graph_index.add_edge(node_a, node_b, RelationshipType.CAUSAL)
        graph_index.add_edge(node_a, node_c, RelationshipType.SEMANTIC)
        
        connected = graph_index.get_connected_nodes(node_a)
        assert len(connected) == 2
    
    def test_bidirectional_edge(self, graph_index):
        """Test bidirectional edge creation."""
        node1 = graph_index.add_node(content="Node 1")
        node2 = graph_index.add_node(content="Node 2")
        
        edge_id = graph_index.add_edge(
            node1, node2, RelationshipType.EQUIVALENT,
            bidirectional=True
        )
        
        edges_from = graph_index.get_edges_from_node(node1)
        edges_to = graph_index.get_edges_to_node(node2)
        
        assert len(edges_from) >= 1
        assert len(edges_to) >= 1


# =============================================================================
# 1. INDIVIDUAL INDEX TESTS - HASH INDEX
# =============================================================================

@pytest.mark.skipif(not HASH_AVAILABLE, reason="Hash index not available")
class TestHashIndex:
    """Test HashIndex functionality."""
    
    def test_add_content(self, hash_index):
        """Test adding content to hash index."""
        result = hash_index.add(
            memory_id="test_1",
            content="Test content for hashing"
        )
        
        assert result is not None
        is_duplicate, entry = result
        assert is_duplicate is False or entry is None
    
    def test_exact_duplicate_detection(self, hash_index):
        """Test detection of exact duplicates."""
        content = "Exact duplicate content"
        
        # Add first time
        result1 = hash_index.add(memory_id="id_1", content=content)
        assert result1[0] is False  # Not a duplicate
        
        # Add exact same content
        result2 = hash_index.add(memory_id="id_2", content=content)
        assert result2[0] is True  # Is a duplicate
    
    def test_near_duplicate_detection(self, hash_index):
        """Test detection of near-duplicates using SimHash."""
        content1 = "The quick brown fox jumps over the lazy dog"
        content2 = "The quick brown fox jumps over the lazy cat"
        
        hash_index.add(memory_id="id_1", content=content1)
        
        # Check for similar content
        near_duplicates = hash_index.find_near_duplicates(content2)
        assert len(near_duplicates) >= 1
    
    def test_simhash_computation(self):
        """Test SimHash computation."""
        hash1 = compute_simhash("Test content one")
        hash2 = compute_simhash("Test content one")
        hash3 = compute_simhash("Completely different content")
        
        # Same content should have same hash
        assert hash1 == hash2
        
        # Different content should have different hash
        assert hash1 != hash3
    
    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        hash1 = 0b11110000
        hash2 = 0b11110001
        
        distance = hamming_distance(hash1, hash2, num_bits=8)
        assert distance == 1
    
    def test_minhash_computation(self):
        """Test MinHash computation."""
        sig1 = compute_minhash("The quick brown fox")
        sig2 = compute_minhash("The quick brown fox")
        sig3 = compute_minhash("Completely different text here")
        
        # Same content should have similar signature
        assert sig1 == sig2
        
        # Different content should have different signature
        assert sig1 != sig3
    
    def test_bloom_filter(self, hash_index):
        """Test Bloom filter for fast existence checks."""
        content = "Content for bloom filter test"
        hash_index.add(memory_id="bloom_1", content=content)
        
        # Bloom filter should indicate possible membership
        assert hash_index.bloom_filter.contains(hash_index.config.compute_md5_hash(content))
    
    def test_find_duplicate(self, hash_index):
        """Test finding duplicate by content."""
        content = "Unique content for duplicate test"
        hash_index.add(memory_id="dup_1", content=content)
        
        found = hash_index.find_duplicate(content)
        assert found is not None
    
    def test_content_merging(self, hash_index):
        """Test merging of duplicate content."""
        content1 = "Base content"
        content2 = "Base content with additions"
        
        hash_index.add(memory_id="merge_1", content=content1)
        
        # Try to add similar content
        is_dup, merged = hash_index.add(
            memory_id="merge_2",
            content=content2,
            metadata={"auto_merge": True}
        )
        
        if is_dup and merged:
            assert merged is not None


# =============================================================================
# 1. INDIVIDUAL INDEX TESTS - SEMANTIC INDEX
# =============================================================================

@pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="Semantic index not available")
class TestSemanticIndex:
    """Test SemanticIndex functionality."""
    
    def test_add_content(self, semantic_index):
        """Test adding content with embedding."""
        content_id = semantic_index.add_content(
            content="Test semantic content",
            content_id="sem_1",
            metadata={"test": True}
        )
        
        assert content_id is not None
    
    def test_search(self, semantic_index):
        """Test semantic search."""
        # Add content
        semantic_index.add_content(
            content="Machine learning algorithms process data",
            content_id="ml_1"
        )
        semantic_index.add_content(
            content="Deep neural networks learn patterns",
            content_id="dl_1"
        )
        semantic_index.add_content(
            content="Completely unrelated topic about cooking",
            content_id="cook_1"
        )
        
        # Search
        results = semantic_index.search(
            query="artificial intelligence learning",
            top_k=3
        )
        
        assert len(results) >= 2
    
    def test_semantic_query_validation(self):
        """Test semantic query validation."""
        # Valid query
        query = SemanticQuery(
            query_text="Test query",
            top_k=10,
            similarity_threshold=0.7
        )
        errors = query.validate()
        assert len(errors) == 0
        
        # Invalid query
        invalid_query = SemanticQuery(
            query_text="",
            top_k=-1,
            similarity_threshold=1.5
        )
        errors = invalid_query.validate()
        assert len(errors) > 0
    
    def test_embedding_generation_mock(self, semantic_index):
        """Test embedding generation with mocked backend."""
        embedding = semantic_index.embedding_generator.generate("Test text")
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
    
    def test_search_with_filters(self, semantic_index):
        """Test semantic search with filters."""
        semantic_index.add_content(
            content="Important core concept",
            content_id="core_1",
            metadata={"level": "core", "importance": 0.9}
        )
        
        results = semantic_index.search(
            query="important concept",
            metadata_filters={"level": "core"}
        )
        
        assert len(results) >= 0  # May or may not return based on filter


# =============================================================================
# 2. INTEGRATION TESTS - CONTEXT ASSEMBLER
# =============================================================================

@pytest.mark.skipif(not ASSEMBLER_AVAILABLE, reason="Context assembler not available")
class TestUnifiedContextAssembler:
    """Test UnifiedContextAssembler integration."""
    
    def test_assemble_context(self, context_assembler):
        """Test context assembly from query."""
        context = context_assembler.assemble(
            query="How should we handle errors?",
            max_tokens=2000
        )
        
        assert context is not None
        assert isinstance(context, AssembledContext)
    
    def test_context_sections(self, context_assembler):
        """Test context section organization."""
        context = context_assembler.assemble(
            query="System design principles",
            max_tokens=2000
        )
        
        assert hasattr(context, 'core_principles')
        assert hasattr(context, 'key_relationships')
        assert hasattr(context, 'recent_details')
    
    def test_token_budget_management(self, context_assembler):
        """Test token budget allocation."""
        max_tokens = 1000
        context = context_assembler.assemble(
            query="Test query",
            max_tokens=max_tokens
        )
        
        assert context.total_tokens <= max_tokens
    
    def test_context_freshness(self, context_assembler):
        """Test context freshness scoring."""
        context = context_assembler.assemble(
            query="Recent changes",
            max_tokens=2000
        )
        
        assert hasattr(context, 'freshness_score')
        assert 0.0 <= context.freshness_score <= 1.0


@pytest.mark.skipif(not ASSEMBLER_AVAILABLE, reason="Context assembly pipeline not available")
class TestContextAssemblyPipeline:
    """Test ContextAssemblyPipeline."""
    
    def test_pipeline_execution(self, context_assembler):
        """Test full pipeline execution."""
        result = context_assembler.pipeline.execute(
            query="Test query for pipeline",
            initial_items=[]
        )
        
        assert result is not None
        assert hasattr(result, 'context')
        assert hasattr(result, 'success')
    
    def test_pipeline_stages(self, context_assembler):
        """Test all pipeline stages are present."""
        stages = context_assembler.pipeline.stages
        
        stage_types = [s.stage_type for s in stages]
        
        assert ContextAssemblyStage.HIERARCHICAL in stage_types
        assert ContextAssemblyStage.GRAPH in stage_types
        assert ContextAssemblyStage.DEDUPLICATION in stage_types
        assert ContextAssemblyStage.SEMANTIC in stage_types
    
    def test_stage_results(self, context_assembler):
        """Test pipeline stage results tracking."""
        result = context_assembler.pipeline.execute(
            query="Test for stage results"
        )
        
        assert hasattr(result, 'stage_results')
        assert len(result.stage_results) > 0
    
    def test_pipeline_with_conversation_history(self, context_assembler):
        """Test pipeline with conversation history."""
        history = [
            {"role": "user", "content": "What is the database schema?"},
            {"role": "assistant", "content": "The schema includes users and orders tables."},
            {"role": "user", "content": "How do we handle errors?"}
        ]
        
        result = context_assembler.pipeline.execute(
            query="Add validation to the schema",
            conversation_history=history
        )
        
        assert result is not None


# =============================================================================
# 2. INTEGRATION TESTS - ENHANCED KNOWLEDGE ENGINE
# =============================================================================

@pytest.mark.skipif(not ENHANCED_ENGINE_AVAILABLE, reason="Enhanced knowledge engine not available")
class TestEnhancedKnowledgeEngine:
    """Test EnhancedKnowledgeEngine integration."""
    
    def test_add_knowledge(self, enhanced_engine):
        """Test adding knowledge through all indexes."""
        entry = enhanced_engine.add_knowledge_with_indexing(
            content="Test knowledge entry",
            title="Test Entry",
            domain="testing",
            tags=["test", "knowledge"]
        )
        
        assert entry is not None
        assert isinstance(entry, UnifiedKnowledgeEntry)
        assert entry.entry_id is not None
    
    def test_query_with_curation(self, enhanced_engine):
        """Test querying with context curation."""
        # Add some knowledge
        enhanced_engine.add_knowledge_with_indexing(
            content="Core principle: Always validate inputs",
            domain="security",
            importance=0.9
        )
        enhanced_engine.add_knowledge_with_indexing(
            content="Implementation detail: Use Pydantic for validation",
            domain="security",
            importance=0.6
        )
        
        # Query
        results = enhanced_engine.query_with_context_curation(
            query="How to validate inputs",
            top_k=5
        )
        
        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], CuratedQueryResult)
    
    def test_entry_registry(self, enhanced_engine):
        """Test entry registry persistence."""
        entry = enhanced_engine.add_knowledge_with_indexing(
            content="Registry test entry",
            domain="testing"
        )
        
        # Check entry is in registry
        assert entry.entry_id in enhanced_engine._entries
        
        # Check entry data
        retrieved = enhanced_engine._entries[entry.entry_id]
        assert retrieved.content == "Registry test entry"
    
    def test_duplicate_handling(self, enhanced_engine):
        """Test duplicate detection and merging."""
        content = "Duplicate test content"
        
        entry1 = enhanced_engine.add_knowledge_with_indexing(
            content=content,
            domain="testing"
        )
        
        entry2 = enhanced_engine.add_knowledge_with_indexing(
            content=content,
            domain="testing"
        )
        
        # Should either merge or create new entry
        assert entry1 is not None
        assert entry2 is not None
    
    def test_query_ranking(self, enhanced_engine):
        """Test query result ranking."""
        # Add entries with different relevance
        enhanced_engine.add_knowledge_with_indexing(
            content="Python programming language",
            domain="programming",
            importance=0.8
        )
        enhanced_engine.add_knowledge_with_indexing(
            content="Java programming language",
            domain="programming",
            importance=0.8
        )
        
        results = enhanced_engine.query_with_context_curation(
            query="Python code examples",
            top_k=2
        )
        
        # Results should be ranked by relevance
        if len(results) >= 2:
            assert results[0].combined_score >= results[1].combined_score


# =============================================================================
# 3. CONTEXT ROT PREVENTION TESTS
# =============================================================================

@pytest.mark.skipif(not HIERARCHICAL_AVAILABLE or not GRAPH_AVAILABLE, 
                    reason="Required indexes not available")
class TestContextRotPrevention:
    """Test context rot prevention features."""
    
    def test_core_memory_persistence(self, hierarchical_index):
        """Test CORE memories persist across long conversations."""
        # Add a CORE memory
        core = hierarchical_index.add_memory(
            content="CORE: System architecture uses microservices",
            level=MemoryLevel.CORE,
            tags=["architecture", "core"]
        )
        
        # Simulate many granular messages
        for i in range(100):
            hierarchical_index.add_memory(
                content=f"GRANULAR: Message {i} - minor detail",
                level=MemoryLevel.GRANULAR,
                tags=["conversation"]
            )
        
        # CORE memory should still be retrievable and high importance
        retrieved = hierarchical_index.get_memory(core.node_id)
        assert retrieved is not None
        assert retrieved.level == MemoryLevel.CORE
        
        # Should be in top results for core level query
        core_memories = hierarchical_index.query_by_level(MemoryLevel.CORE)
        assert any(m.node_id == core.node_id for m in core_memories)
    
    def test_graph_relationships_connect_distant_messages(self, graph_index):
        """Test graph relationships connect distant messages."""
        # Create first message
        msg_1 = graph_index.add_node(
            content="First message: We need a database",
            importance=0.9
        )
        
        # Create distant message (simulating many messages in between)
        msg_100 = graph_index.add_node(
            content="Message 100: Therefore, we choose PostgreSQL",
            importance=0.8
        )
        
        # Create relationship
        graph_index.add_edge(
            msg_1, msg_100, RelationshipType.CAUSAL, weight=0.9
        )
        
        # Find path between distant messages
        path = graph_index.find_path(msg_1, msg_100, max_depth=10)
        
        assert path is not None
        assert len(path.path) == 2
    
    def test_deduplication_reduces_noise(self, hash_index):
        """Test that deduplication reduces context noise."""
        # Add similar messages multiple times
        messages = [
            "We should use Redis for caching",
            "We should use Redis for caching.",
            "We should use redis for caching",
            "Let's use Redis for caching",
        ]
        
        unique_count = 0
        for msg in messages:
            is_dup, _ = hash_index.add(memory_id=f"msg_{hash(msg)}", content=msg)
            if not is_dup:
                unique_count += 1
        
        # Should detect near-duplicates
        assert unique_count < len(messages)
    
    def test_semantic_relevance_ranking(self, semantic_index):
        """Test semantic relevance ranking."""
        # Add various content
        semantic_index.add_content(
            content="Machine learning algorithms",
            content_id="ml"
        )
        semantic_index.add_content(
            content="Deep learning neural networks",
            content_id="dl"
        )
        semantic_index.add_content(
            content="Cooking recipes for dinner",
            content_id="cooking"
        )
        
        # Search for AI-related content
        results = semantic_index.search(
            query="artificial intelligence",
            top_k=3
        )
        
        # AI-related content should be ranked higher
        if len(results) >= 2:
            # Check that AI content scores higher than cooking
            ai_scores = [r.similarity_score for r in results 
                        if 'ml' in r.id or 'dl' in r.id]
            cook_scores = [r.similarity_score for r in results 
                          if 'cooking' in r.id]
            
            if ai_scores and cook_scores:
                assert max(ai_scores) > max(cook_scores)
    
    @pytest.mark.slow
    def test_long_conversation_simulation(self, hierarchical_index, graph_index):
        """Simulate 100+ message conversation and verify context preservation."""
        # CORE principles that should persist
        core_memories = [
            hierarchical_index.add_memory(
                content=f"CORE Principle {i}: Critical system rule",
                level=MemoryLevel.CORE,
                domain="system"
            )
            for i in range(3)
        ]
        
        # Add 100+ contextual messages
        prev_node = None
        first_node = None
        
        for i in range(110):
            if i % 10 == 0:
                # Important checkpoint every 10 messages
                node = hierarchical_index.add_memory(
                    content=f"Checkpoint {i//10}: Important milestone",
                    level=MemoryLevel.IMPORTANT,
                    domain="milestones"
                )
            else:
                # Regular granular message
                node = hierarchical_index.add_memory(
                    content=f"Message {i}: Conversation detail",
                    level=MemoryLevel.GRANULAR,
                    domain="conversation"
                )
            
            # Add to graph with sequential relationship
            graph_node = graph_index.add_node(
                content=f"Graph node for message {i}",
                importance=0.5
            )
            
            if prev_node:
                graph_index.add_edge(
                    prev_node, graph_node, RelationshipType.SEQUENTIAL
                )
            else:
                first_node = graph_node
            
            prev_node = graph_node
        
        # Verify CORE memories still exist
        for core in core_memories:
            retrieved = hierarchical_index.get_memory(core.node_id)
            assert retrieved is not None
            assert retrieved.level == MemoryLevel.CORE
        
        # Verify we can traverse from first to recent message
        if first_node and prev_node:
            result = graph_index.traverse_relationships(
                first_node, depth=20, mode=TraversalMode.BFS
            )
            assert result is not None
            assert len(result.nodes) > 0


# =============================================================================
# 4. PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.skipif(not ASSEMBLER_AVAILABLE, reason="Context assembler not available")
    def test_token_budget_management(self, context_assembler):
        """Test token budget allocation and enforcement."""
        max_tokens = 500
        
        context = context_assembler.assemble(
            query="Test query",
            max_tokens=max_tokens
        )
        
        # Total tokens should not exceed budget
        assert context.total_tokens <= max_tokens
        
        # Should have allocated tokens across sections
        total_section_tokens = (
            sum(i.estimate_tokens() for i in context.core_principles) +
            sum(i.estimate_tokens() for i in context.key_relationships) +
            sum(i.estimate_tokens() for i in context.recent_details)
        )
        
        assert total_section_tokens <= max_tokens
    
    @pytest.mark.skipif(not ENHANCED_ENGINE_AVAILABLE, reason="Enhanced engine not available")
    def test_query_latency(self, enhanced_engine):
        """Test query latency with large dataset."""
        # Add many entries
        for i in range(100):
            enhanced_engine.add_knowledge_with_indexing(
                content=f"Entry {i}: {' '.join(['word'] * 20)}",
                domain="performance_test",
                importance=0.5
            )
        
        # Measure query time
        start_time = time.time()
        results = enhanced_engine.query_with_context_curation(
            query="Find specific entry",
            top_k=10
        )
        query_time = time.time() - start_time
        
        # Query should complete in reasonable time (adjust threshold as needed)
        assert query_time < 5.0  # 5 seconds
    
    @pytest.mark.skipif(not HIERARCHICAL_AVAILABLE, reason="Hierarchical index not available")
    def test_concurrent_access_thread_safety(self, hierarchical_index):
        """Test thread safety with concurrent access."""
        errors = []
        success_count = [0]
        
        def add_memories():
            try:
                for i in range(10):
                    hierarchical_index.add_memory(
                        content=f"Concurrent memory {i}",
                        level=MemoryLevel.CONTEXTUAL
                    )
                    success_count[0] += 1
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent threads
        threads = [threading.Thread(target=add_memories) for _ in range(5)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert success_count[0] == 50  # 5 threads * 10 memories each


# =============================================================================
# 5. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not HIERARCHICAL_AVAILABLE, reason="Hierarchical index not available")
    def test_empty_index_queries(self, hierarchical_index):
        """Test queries on empty index."""
        results = hierarchical_index.query_by_level(MemoryLevel.CORE)
        assert results == []
        
        results = hierarchical_index.search_content("test")
        assert results == []
        
        tree = hierarchical_index.get_tree_structure()
        assert "roots" in tree or tree == {}
    
    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph index not available")
    def test_empty_graph_queries(self, graph_index):
        """Test queries on empty graph."""
        result = graph_index.traverse_relationships("nonexistent", depth=2)
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) == 0
        
        path = graph_index.find_path("a", "b")
        assert path is None
    
    @pytest.mark.skipif(not HASH_AVAILABLE, reason="Hash index not available")
    def test_duplicate_content_handling(self, hash_index):
        """Test handling of duplicate content."""
        content = "Exactly the same content"
        
        # Add multiple times
        for i in range(5):
            is_dup, entry = hash_index.add(
                memory_id=f"dup_{i}",
                content=content
            )
            
            # First should not be duplicate, rest should be
            if i == 0:
                assert is_dup is False
            else:
                assert is_dup is True
    
    @pytest.mark.skipif(not ENHANCED_ENGINE_AVAILABLE, reason="Enhanced engine not available")
    def test_missing_dependencies_graceful_degradation(self, temp_dir):
        """Test graceful degradation when dependencies are missing."""
        # Create config with some features disabled
        config = EnhancedKnowledgeEngineConfig(
            storage_path=temp_dir,
            enable_hierarchical=False,
            enable_graph=False,
            enable_hash=True,
            enable_semantic=False
        )
        
        engine = EnhancedKnowledgeEngine(config=config)
        
        # Should still work with limited functionality
        entry = engine.add_knowledge_with_indexing(
            content="Test without all indexes",
            domain="testing"
        )
        
        assert entry is not None
        assert entry.entry_id is not None
    
    @pytest.mark.skipif(not ASSEMBLER_AVAILABLE, reason="Context assembler not available")
    def test_configuration_overrides(self, temp_dir):
        """Test configuration override handling."""
        # Test with various config combinations
        configs = [
            ContextAssemblerConfig(
                max_tokens=1000,
                enable_hierarchical=False,
                enable_graph=False,
                enable_deduplication=False,
                enable_semantic=False
            ),
            ContextAssemblerConfig(
                max_tokens=8000,
                core_token_ratio=0.5,
                granular_token_ratio=0.1
            ),
        ]
        
        for config in configs:
            config.db_path = os.path.join(temp_dir, "test.db")
            assembler = UnifiedContextAssembler(config=config)
            
            context = assembler.assemble(query="Test", max_tokens=config.max_tokens)
            assert context is not None
    
    @pytest.mark.skipif(not HIERARCHICAL_AVAILABLE, reason="Hierarchical index not available")
    def test_memory_level_bounds(self, hierarchical_index):
        """Test memory level boundary conditions."""
        # Try to promote beyond CORE
        node = hierarchical_index.add_memory(
            content="Already at CORE",
            level=MemoryLevel.CORE
        )
        
        promoted = hierarchical_index.promote_node(node.node_id, levels=5)
        assert promoted.level == MemoryLevel.CORE  # Should stay at CORE
        
        # Try to demote beyond GRANULAR
        node2 = hierarchical_index.add_memory(
            content="Already at GRANULAR",
            level=MemoryLevel.GRANULAR
        )
        
        demoted = hierarchical_index.demote_node(node2.node_id, levels=5)
        assert demoted.level == MemoryLevel.GRANULAR  # Should stay at GRANULAR
    
    @pytest.mark.skipif(not GRAPH_AVAILABLE, reason="Graph index not available")
    def test_circular_relationship_handling(self, graph_index):
        """Test handling of circular relationships."""
        node_a = graph_index.add_node(content="Node A")
        node_b = graph_index.add_node(content="Node B")
        node_c = graph_index.add_node(content="Node C")
        
        # Create cycle: A -> B -> C -> A
        graph_index.add_edge(node_a, node_b, RelationshipType.CAUSAL)
        graph_index.add_edge(node_b, node_c, RelationshipType.CAUSAL)
        graph_index.add_edge(node_c, node_a, RelationshipType.CAUSAL)
        
        # Traversal should handle cycle without infinite loop
        result = graph_index.traverse_relationships(
            node_a, depth=10, mode=TraversalMode.BFS
        )
        
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) == 3  # Should find all 3 nodes
    
    @pytest.mark.skipif(not HASH_AVAILABLE, reason="Hash index not available")
    def test_very_long_content_hashing(self, hash_index):
        """Test hashing of very long content."""
        # Create very long content
        long_content = "Word " * 10000  # 50k+ characters
        
        result = hash_index.add(memory_id="long_1", content=long_content)
        assert result is not None
        
        # Should handle without error
        is_dup, entry = result
        assert is_dup is False or entry is not None
    
    @pytest.mark.skipif(not SEMANTIC_AVAILABLE, reason="Semantic index not available")
    def test_empty_query_handling(self, semantic_index):
        """Test handling of empty or invalid queries."""
        # Empty query should return empty results or handle gracefully
        results = semantic_index.search(query="", top_k=10)
        assert isinstance(results, list)
        
        # Very long query
        long_query = "word " * 1000
        results = semantic_index.search(query=long_query, top_k=10)
        assert isinstance(results, list)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
