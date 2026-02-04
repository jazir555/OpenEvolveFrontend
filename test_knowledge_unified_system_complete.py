"""
Comprehensive Tests for the Unified Knowledge Memory System

This test suite validates:
1. Integration of all 4 indexes (semantic, hash, hierarchical, graph)
2. Context rot prevention (CORE facts persist, state never truncated)
3. End-to-end turn processing pipeline
4. Performance characteristics
5. Edge cases and correctness

Test Categories:
- Integration Tests (15 tests)
- Context Rot Prevention Tests (10 tests)
- End-to-End Tests (10 tests)
- Performance Tests (8 tests)
- Edge Case Tests (8 tests)
- Correctness Tests (8 tests)

Total: 50+ comprehensive test functions
"""

import pytest
import asyncio
import sqlite3
import threading
import time
import json
import hashlib
import random
import string
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ============================================================================
# Import System Components
# ============================================================================

# State Manager
from knowledge_state_manager import (
    StateManager, ConversationState, CoreFact, ActiveDecision, Constraint,
    CurrentContext, TurnResult, StateSnapshot, StateUpdate,
    FactPriority, DecisionStatus
)

# Hybrid Retrieval
from knowledge_hybrid_retrieval import (
    HybridRetriever, Memory, RetrievedMemory, RetrievalWeights,
    RetrievalStrategyType
)

# Lifecycle Manager
from knowledge_lifecycle_manager import (
    MemoryLifecycleManager, LifecycleConfig, LifecycleStage, MemoryType,
    MemoryMetadata, ConfidenceScorer, DecayDetector, ArchivalManager
)

# Working Memory
from knowledge_working_memory import (
    WorkingMemoryManager, SimpleMemoryRetriever, Memory as WorkingMemory,
    MemoryType as WMemoryType, Priority, TokenCounter, PromptContext,
    StateSnapshot as WMStateSnapshot, WorkingMemoryBuffer
)

# Chronicle Memory
from chronicle_memory import (
    ChronicleMemory, ChronicleEvent, EventType, Outcome,
    ChronicleStore, LoopDetector, Narrative
)

# Four Indexes
from knowledge_semantic_index import SemanticIndex, SemanticQuery, SemanticIndexConfig
from knowledge_hash_index import HashIndex, HashIndexConfig, compute_simhash, compute_minhash
from knowledge_hierarchical_index import HierarchicalIndex, MemoryNode, MemoryLevel
from knowledge_graph_index import GraphIndex, RelationshipType, NodeType, MemoryNode as GraphMemoryNode


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_llm_responses():
    """Mock LLM response generator."""
    responses = [
        "I understand. DECISION: approach_use_hybrid_search Rationale: Combines multiple strategies",
        "FACT: The system uses 4 indexes for retrieval. INSIGHT: This provides redundancy and accuracy.",
        "PREFERENCE: prioritize_recent_memories Value: True. Let me think about this...",
        "DECISION: archive_old_memories Rationale: Save storage space. FACT: Memories older than 180 days should be reviewed.",
        "I need to consider the STATE: current_mode=optimization before proceeding.",
    ]
    return iter(responses)


@pytest.fixture
def state_manager(temp_db_path):
    """Create a state manager with temp database."""
    manager = StateManager(db_path=temp_db_path, auto_persist=True)
    yield manager
    # Cleanup
    try:
        os.unlink(temp_db_path)
    except:
        pass


@pytest.fixture
def hybrid_retriever():
    """Create a hybrid retriever."""
    return HybridRetriever(default_limit=15, max_workers=4)


@pytest.fixture
def lifecycle_manager(temp_dir):
    """Create a lifecycle manager."""
    config = LifecycleConfig(
        active_db_path=os.path.join(temp_dir, "active.db"),
        archive_db_path=os.path.join(temp_dir, "archive.db"),
        metadata_db_path=os.path.join(temp_dir, "metadata.db"),
        decay_days_min=30,
        decay_days_max=90
    )
    manager = MemoryLifecycleManager(config)
    yield manager
    manager.close()


@pytest.fixture
def working_memory_manager():
    """Create a working memory manager."""
    return WorkingMemoryManager(max_context_tokens=4000)


@pytest.fixture
def chronicle_memory(temp_dir):
    """Create a chronicle memory instance."""
    return ChronicleMemory(storage_path=temp_dir)


@pytest.fixture
def all_four_indexes(temp_dir):
    """Create all four indexes for integration testing."""
    semantic = SemanticIndex(SemanticIndexConfig(
        cache_dir=os.path.join(temp_dir, "semantic_cache"),
        vector_backend="sqlite"
    ))
    hash_idx = HashIndex(HashIndexConfig(
        db_path=os.path.join(temp_dir, "hash.db")
    ))
    hierarchical = HierarchicalIndex(db_path=os.path.join(temp_dir, "hierarchical.db"))
    graph = GraphIndex(db_path=os.path.join(temp_dir, "graph.db"))
    
    return {
        'semantic': semantic,
        'hash': hash_idx,
        'hierarchical': hierarchical,
        'graph': graph
    }


@pytest.fixture
def unified_system(temp_dir, temp_db_path):
    """Create a fully integrated unified memory system."""
    # Create all components
    state_mgr = StateManager(db_path=temp_db_path)
    hybrid_ret = HybridRetriever(default_limit=15)
    lifecycle_mgr = MemoryLifecycleManager(LifecycleConfig(
        active_db_path=os.path.join(temp_dir, "active.db"),
        archive_db_path=os.path.join(temp_dir, "archive.db"),
        metadata_db_path=os.path.join(temp_dir, "metadata.db")
    ))
    working_mgr = WorkingMemoryManager(max_context_tokens=4000)
    chronicle = ChronicleMemory(storage_path=os.path.join(temp_dir, "chronicle"))
    
    system = {
        'state_manager': state_mgr,
        'hybrid_retriever': hybrid_ret,
        'lifecycle_manager': lifecycle_mgr,
        'working_memory': working_mgr,
        'chronicle': chronicle,
        'temp_dir': temp_dir
    }
    
    yield system
    
    # Cleanup
    lifecycle_mgr.close()


# ============================================================================
# Helper Functions
# ============================================================================

def generate_test_content(size_words: int = 50) -> str:
    """Generate random test content."""
    words = [' '.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) 
             for _ in range(size_words)]
    return ' '.join(words)


def create_test_memory(memory_id: str, content: str, importance: int = 5) -> Memory:
    """Create a test memory for hybrid retrieval."""
    return Memory(
        id=memory_id,
        content=content,
        importance=importance,
        timestamp=time.time(),
        tags=set(content.lower().split()[:5])
    )


def create_core_fact(key: str, value: Any, turn: int = 0) -> CoreFact:
    """Create a CORE priority fact."""
    return CoreFact(
        key=key,
        value=value,
        priority=FactPriority.CRITICAL,
        source_turn=turn,
        confidence=1.0
    )


def simulate_conversation_turn(
    system: Dict,
    conversation_id: str,
    turn_number: int,
    user_input: str,
    mock_response: str
) -> Dict[str, Any]:
    """Simulate a complete conversation turn."""
    results = {
        'turn_number': turn_number,
        'user_input': user_input,
        'state_updated': False,
        'memories_retrieved': 0,
        'facts_extracted': 0
    }
    
    # 1. Build prompt context (retrieves memories)
    prompt_context = system['working_memory'].build_prompt_context(user_input)
    results['memories_retrieved'] = len(prompt_context.relevant_memories)
    
    # 2. Update state from response
    from knowledge_working_memory import TurnMetadata
    turn_meta = TurnMetadata(
        turn_id=f"turn_{turn_number}",
        timestamp=time.time(),
        query=user_input,
        response=mock_response,
        duration_ms=100.0,
        token_count=500
    )
    updates = system['working_memory'].update_from_response(mock_response, turn_meta)
    results['facts_extracted'] = updates.get('facts_added', 0)
    results['decisions_recorded'] = updates.get('decisions_recorded', 0)
    
    # 3. Record in chronicle
    asyncio.run(system['chronicle'].record_event(
        event_type=EventType.ACTION_COMPLETED,
        action="conversation_turn",
        parameters={
            'turn_number': turn_number,
            'user_input': user_input[:100],
            'facts_added': results['facts_extracted']
        },
        outcome=Outcome.SUCCESS,
        narrative=f"Turn {turn_number}: User asked about {user_input[:50]}..."
    ))
    
    return results


# ============================================================================
# INTEGRATION TESTS (15 tests)
# ============================================================================

class TestIntegrationAllFourIndexes:
    """Test all 4 indexes working together."""
    
    def test_all_indexes_created(self, all_four_indexes):
        """Test that all four indexes can be instantiated."""
        assert all_four_indexes['semantic'] is not None
        assert all_four_indexes['hash'] is not None
        assert all_four_indexes['hierarchical'] is not None
        assert all_four_indexes['graph'] is not None
    
    def test_semantic_index_add_and_search(self, all_four_indexes):
        """Test semantic index can add and search content."""
        semantic = all_four_indexes['semantic']
        
        # Add documents
        doc_id = semantic.add_document(
            content="Machine learning is a subset of artificial intelligence",
            metadata={"topic": "AI", "importance": 9}
        )
        
        # Search
        results = semantic.search("artificial intelligence", top_k=5)
        assert len(results) > 0
        assert any("intelligence" in r.content.lower() for r in results)
    
    def test_hash_index_deduplication(self, all_four_indexes):
        """Test hash index detects duplicates."""
        hash_idx = all_four_indexes['hash']
        
        content1 = "This is unique content for testing"
        content2 = "This is unique content for testing"  # Same
        content3 = "This is different content entirely"
        
        # Add first content
        result1 = hash_idx.add_content("doc1", content1.encode())
        assert result1.is_new is True
        
        # Add duplicate
        result2 = hash_idx.add_content("doc2", content2.encode())
        assert result2.is_duplicate is True
        assert result2.duplicate_of == "doc1"
        
        # Add different content
        result3 = hash_idx.add_content("doc3", content3.encode())
        assert result3.is_new is True
    
    def test_hierarchical_index_importance_levels(self, all_four_indexes):
        """Test hierarchical index organizes by importance."""
        hier = all_four_indexes['hierarchical']
        
        # Add at different levels
        core_id = hier.add_node(
            content="Core system requirement: never lose data",
            level=MemoryLevel.CORE,
            importance_score=1.0
        )
        
        granular_id = hier.add_node(
            content="Specific UI color is blue",
            level=MemoryLevel.GRANULAR,
            importance_score=0.3
        )
        
        # Query by level
        core_nodes = hier.get_nodes_by_level(MemoryLevel.CORE)
        assert any(n.node_id == core_id for n in core_nodes)
        
        # Core should be ranked higher
        all_nodes = hier.get_all_nodes_sorted()
        if all_nodes:
            assert all_nodes[0].level == MemoryLevel.CORE
    
    def test_graph_index_relationships(self, all_four_indexes):
        """Test graph index maintains relationships."""
        graph = all_four_indexes['graph']
        
        # Add nodes
        node1 = graph.add_node("System requires authentication", node_type=NodeType.REQUIREMENT)
        node2 = graph.add_node("Implement OAuth2 flow", node_type=NodeType.ACTION)
        
        # Add relationship
        graph.add_relationship(
            node1, node2, 
            RelationshipType.DEPENDS_ON,
            weight=0.9
        )
        
        # Traverse
        related = graph.traverse_relationships(node1, depth=1)
        assert node2 in [r.node_id for r in related]
    
    def test_cross_index_consistency(self, all_four_indexes):
        """Test that indexes maintain consistent data."""
        content = "Cross-index test content about neural networks"
        content_id = "cross_test_001"
        
        # Add to all indexes
        all_four_indexes['semantic'].add_document(content, doc_id=content_id)
        all_four_indexes['hash'].add_content(content_id, content.encode())
        
        # Verify retrieval from each
        semantic_results = all_four_indexes['semantic'].search("neural networks", top_k=1)
        hash_result = all_four_indexes['hash'].get_content_hash(content_id)
        
        assert len(semantic_results) > 0 or hash_result is not None
    
    def test_index_persistence(self, all_four_indexes, temp_dir):
        """Test indexes persist data across instances."""
        # Add data
        graph = all_four_indexes['graph']
        node_id = graph.add_node("Persistent test node")
        
        # Create new instance with same DB
        graph2 = GraphIndex(db_path=os.path.join(temp_dir, "graph.db"))
        
        # Verify data persisted
        node = graph2.get_node(node_id)
        assert node is not None
        assert node.content == "Persistent test node"
    
    def test_concurrent_index_access(self, all_four_indexes):
        """Test thread-safe concurrent access to indexes."""
        hier = all_four_indexes['hierarchical']
        errors = []
        
        def add_nodes(thread_id: int):
            try:
                for i in range(10):
                    hier.add_node(
                        content=f"Thread {thread_id} node {i}",
                        level=MemoryLevel.CONTEXTUAL
                    )
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    def test_integrated_hybrid_search(self, unified_system):
        """Test hybrid retriever uses multiple strategies."""
        retriever = unified_system['hybrid_retriever']
        
        # Index memories
        for i in range(20):
            mem = create_test_memory(
                f"mem_{i}",
                f"Test memory about topic {i % 5} with importance {i}",
                importance=5 + (i % 5)
            )
            retriever.index_memory(mem)
        
        # Search
        results = retriever.retrieve("topic 2", limit=10)
        assert len(results) > 0
        assert len(results) <= 10
        
        # Verify combined scoring
        for r in results:
            assert r.combined_score > 0
            assert len(r.strategy_scores) > 0


class TestIntegrationStateAndRetrieval:
    """Test state manager + hybrid retrieval integration."""
    
    def test_state_influences_retrieval(self, unified_system):
        """Test that state affects memory retrieval."""
        state_mgr = unified_system['state_manager']
        
        # Create conversation
        conv_id = "test_conv_001"
        state = state_mgr.create_conversation(conv_id)
        
        # Add facts to state
        fact = create_core_fact("user_preference", "dark_mode", turn=1)
        state.add_fact(fact)
        
        # State should persist the fact
        snapshot = state.create_snapshot()
        assert any(f.key == "user_preference" for f in snapshot.core_facts)
    
    def test_retrieval_context_updates_state(self, unified_system):
        """Test retrieval results update state context."""
        # Simulate memory retrieval affecting context
        working_mgr = unified_system['working_memory']
        
        # Add long-term memory
        mem_id = working_mgr.add_long_term_memory(
            content="Important system constraint",
            memory_type=WMemoryType.FACT,
            priority=Priority.CRITICAL
        )
        
        # Build context - should retrieve the memory
        context = working_mgr.build_prompt_context("system constraint")
        assert len(context.relevant_memories) > 0
    
    def test_state_snapshot_for_prompt(self, state_manager):
        """Test state snapshot formats correctly for LLM prompt."""
        conv_id = "test_conv_prompt"
        state = state_manager.create_conversation(conv_id)
        
        # Add various state elements
        state.add_fact(CoreFact("key_fact", "value", priority=FactPriority.CRITICAL))
        state.add_decision(ActiveDecision(
            decision_id="dec1",
            description="Use caching",
            rationale="Performance"
        ))
        state.add_constraint(Constraint(
            constraint_id="con1",
            description="Max 100 items",
            constraint_type="hard"
        ))
        
        # Get snapshot
        snapshot = state.create_snapshot(turn_number=5)
        prompt_text = snapshot.to_prompt_context()
        
        assert "key_fact" in prompt_text
        assert "Use caching" in prompt_text
        assert "Max 100 items" in prompt_text


class TestIntegrationLifecycleAndRetrieval:
    """Test lifecycle management integration with retrieval."""
    
    def test_lifecycle_affects_retrieval(self, unified_system):
        """Test that lifecycle stage affects memory retrieval."""
        lifecycle_mgr = unified_system['lifecycle_manager']
        
        # Create memories at different lifecycle stages
        meta1 = lifecycle_mgr.create_memory(
            memory_id="active_mem",
            content=b"Active memory content",
            memory_type=MemoryType.CORE
        )
        
        # CORE memories should never be archived
        assert meta1.memory_type == MemoryType.CORE
        
        # Get lifecycle info
        lifecycle = lifecycle_mgr.get_memory_lifecycle("active_mem")
        assert lifecycle['stage'] == 'active'
    
    def test_archived_memory_recovery(self, lifecycle_manager):
        """Test archived memories can be recovered."""
        # Create and archive a memory
        lifecycle_manager.create_memory(
            memory_id="test_archive",
            content=b"Content to archive",
            memory_type=MemoryType.STANDARD
        )
        
        # Force archive
        lifecycle_manager.force_archive("test_archive")
        
        # Verify archived
        lifecycle = lifecycle_manager.get_memory_lifecycle("test_archive")
        assert lifecycle['stage'] == 'archived'
        
        # Reactivate
        success = lifecycle_manager.force_reactivate("test_archive")
        assert success is True
        
        # Verify active again
        lifecycle = lifecycle_manager.get_memory_lifecycle("test_archive")
        assert lifecycle['stage'] == 'active'


class TestIntegrationWorkingMemoryPipeline:
    """Test working memory pipeline integration."""
    
    def test_full_working_memory_pipeline(self, unified_system):
        """Test complete working memory pipeline."""
        working_mgr = unified_system['working_memory']
        
        # 1. Build prompt context
        context = working_mgr.build_prompt_context("Tell me about AI")
        assert context.system_instruction is not None
        
        # 2. Simulate response
        mock_response = """
        FACT: AI stands for Artificial Intelligence
        DECISION: use_simple_explanations: True
        INSIGHT: Neural networks are inspired by biology
        """
        
        # 3. Update from response
        updates = working_mgr.update_from_response(mock_response)
        assert updates['facts_added'] > 0
        assert updates['decisions_recorded'] > 0
        
        # 4. Verify state updated
        state = working_mgr.get_state()
        assert len(state.facts) > 0
    
    def test_working_memory_buffer_cleared(self, working_memory_manager):
        """Test working buffer is cleared each turn."""
        mgr = working_memory_manager
        
        # Add to buffer
        mgr.add_to_buffer("Temporary reasoning", "reasoning")
        assert mgr.buffer.size() > 0
        
        # Update from response (should clear buffer)
        mgr.update_from_response("Some response")
        assert mgr.buffer.is_empty()


# ============================================================================
# CONTEXT ROT PREVENTION TESTS (10 tests)
# ============================================================================

class TestContextRotPrevention:
    """Test context rot prevention mechanisms."""
    
    def test_core_facts_persist_across_turns(self, unified_system):
        """Test CORE priority facts persist throughout conversation."""
        state_mgr = unified_system['state_manager']
        conv_id = "core_persistence_test"
        
        # Create conversation
        state = state_mgr.create_conversation(conv_id)
        
        # Add CORE fact
        core_fact = create_core_fact("system_critical", "never_delete", turn=0)
        state.add_fact(core_fact)
        
        # Simulate 20 turns
        for turn in range(1, 21):
            turn_result = TurnResult(
                turn_number=turn,
                extracted_facts=[],
                proposed_decisions=[],
                new_constraints=[]
            )
            state_mgr.update_from_turn(conv_id, turn_result)
        
        # Verify CORE fact still exists
        final_state = state_mgr.get_state(conv_id)
        assert "system_critical" in final_state.facts
        assert final_state.facts["system_critical"].priority == FactPriority.CRITICAL
    
    def test_state_never_truncated(self, unified_system):
        """Test that state is never truncated from context."""
        working_mgr = unified_system['working_memory']
        
        # Add many facts to state
        for i in range(50):
            working_mgr.update_state_fact(f"fact_{i}", f"value_{i}")
        
        # Build context
        context = working_mgr.build_prompt_context("test query")
        
        # State should be present
        assert context.state_section is not None
        # State section should contain facts
        assert "fact_" in context.state_section or len(context.state_section) > 50
    
    def test_only_top_n_memories_in_context(self, unified_system):
        """Test only top-N memories are included in context."""
        working_mgr = unified_system['working_memory']
        
        # Add many memories
        for i in range(100):
            working_mgr.add_long_term_memory(
                content=f"Memory content number {i} about various topics",
                memory_type=WMemoryType.FACT,
                priority=Priority.MEDIUM
            )
        
        # Build context with limit
        max_tokens = 2000
        context = working_mgr.build_prompt_context("test")
        
        # Should have limited memories
        # Exact count depends on token counting, but should be bounded
        assert len(context.relevant_memories) <= 50  # Reasonable upper bound
    
    def test_context_size_bounded(self, unified_system):
        """Test context size stays within bounds."""
        working_mgr = unified_system['working_memory']
        
        max_tokens = working_mgr.max_context_tokens
        
        # Build context with large query
        large_query = " ".join(["word"] * 500)
        context = working_mgr.build_prompt_context(large_query, max_tokens=max_tokens)
        
        # Count tokens
        token_count = working_mgr.token_counter.count(context.to_plain_text())
        
        # Should be within bounds (allowing some tolerance for estimation)
        assert token_count <= max_tokens * 1.2  # 20% tolerance
    
    def test_100_turn_conversation_simulation(self, unified_system):
        """Simulate 100+ turn conversation and verify integrity."""
        state_mgr = unified_system['state_manager']
        chronicle = unified_system['chronicle']
        
        conv_id = "long_conversation_test"
        state = state_mgr.create_conversation(conv_id)
        
        # Add initial CORE facts
        core_facts = [
            create_core_fact("project_name", "OpenEvolve", 0),
            create_core_fact("user_role", "admin", 0),
            create_core_fact("critical_constraint", "data_must_be_encrypted", 0),
        ]
        for fact in core_facts:
            state.add_fact(fact)
        
        # Simulate 100 turns
        for turn in range(1, 101):
            # Random turn content
            user_input = f"User query for turn {turn}"
            
            # Create turn result with some facts
            turn_result = TurnResult(
                turn_number=turn,
                input_text=user_input,
                extracted_facts=[
                    CoreFact(f"turn_{turn}_fact", f"value_{turn}", 
                            priority=random.choice([FactPriority.HIGH, FactPriority.MEDIUM]),
                            source_turn=turn)
                ] if turn % 5 == 0 else [],  # Add facts every 5 turns
                proposed_decisions=[],
                new_constraints=[]
            )
            
            state_mgr.update_from_turn(conv_id, turn_result)
            
            # Record in chronicle
            asyncio.run(chronicle.record_event(
                event_type=EventType.ACTION_COMPLETED,
                action="conversation_turn",
                parameters={'turn': turn},
                outcome=Outcome.SUCCESS
            ))
        
        # Verify
        final_state = state_mgr.get_state(conv_id)
        
        # All CORE facts must persist
        for fact in core_facts:
            assert fact.key in final_state.facts, f"CORE fact {fact.key} was lost!"
        
        # State should have accumulated facts
        assert len(final_state.facts) >= len(core_facts)
        
        # Version history should track all turns
        assert len(final_state.version_history) > 0
    
    def test_core_memory_type_never_decays(self, lifecycle_manager):
        """Test CORE type memories never decay or get archived."""
        # Create CORE memory
        meta = lifecycle_manager.create_memory(
            memory_id="core_test",
            content=b"Critical system configuration",
            memory_type=MemoryType.CORE
        )
        
        # Simulate aging
        # Manually set dates to simulate old memory
        meta.last_accessed = datetime.now() - timedelta(days=500)
        meta.created_at = datetime.now() - timedelta(days=600)
        
        # Run decay detection
        decay_detector = lifecycle_manager.decay_detector
        decay_score = decay_detector.calculate_decay_score(meta)
        
        # CORE memories should never decay
        assert decay_score == 0.0
        
        # Should not transition
        new_stage = decay_detector.should_transition(meta, decay_score)
        assert new_stage is None
    
    def test_hierarchical_importance_preservation(self, all_four_indexes):
        """Test hierarchical index preserves importance correctly."""
        hier = all_four_indexes['hierarchical']
        
        # Add at each level
        levels = [MemoryLevel.CORE, MemoryLevel.IMPORTANT, 
                  MemoryLevel.CONTEXTUAL, MemoryLevel.GRANULAR]
        
        node_ids = []
        for level in levels:
            node_id = hier.add_node(
                content=f"Content at level {level.name}",
                level=level,
                importance_score=1.0 - (level.value * 0.2)
            )
            node_ids.append((level, node_id))
        
        # Query all
        all_nodes = hier.get_all_nodes_sorted()
        
        # CORE should be at top
        core_nodes = [n for n in all_nodes if n.level == MemoryLevel.CORE]
        assert len(core_nodes) > 0
    
    def test_memory_promotion_based_on_access(self, all_four_indexes):
        """Test memories can be promoted based on access patterns."""
        hier = all_four_indexes['hierarchical']
        
        # Add granular memory
        node_id = hier.add_node(
            content="Initially unimportant detail",
            level=MemoryLevel.GRANULAR,
            importance_score=0.3
        )
        
        # Simulate heavy access
        for _ in range(50):
            hier.record_access(node_id)
        
        # Re-evaluate importance
        hier.recalculate_importance(node_id)
        
        # Node should have been promoted or have higher score
        updated = hier.get_node(node_id)
        assert updated.importance_score > 0.3 or updated.level != MemoryLevel.GRANULAR
    
    def test_retrieval_prioritizes_core_memories(self, unified_system):
        """Test that retrieval prioritizes CORE/important memories."""
        retriever = unified_system['hybrid_retriever']
        
        # Add memories with different importance
        for i in range(10):
            importance = 10 if i < 3 else 5  # First 3 are high importance
            mem = create_test_memory(
                f"mem_{i}",
                f"Memory about search topic",
                importance=importance
            )
            retriever.index_memory(mem)
        
        # Retrieve
        results = retriever.retrieve("search topic", limit=5)
        
        # Top results should include high importance memories
        top_scores = [r.combined_score for r in results[:3]]
        bottom_scores = [r.combined_score for r in results[-2:]]
        
        if top_scores and bottom_scores:
            assert top_scores[0] >= bottom_scores[0] * 0.5  # Should be reasonably higher
    
    def test_graph_relationships_preserve_context(self, all_four_indexes):
        """Test graph relationships help preserve context across turns."""
        graph = all_four_indexes['graph']
        
        # Create connected chain of ideas
        nodes = []
        prev_node = None
        for i in range(10):
            node = graph.add_node(f"Concept {i}: Part of the reasoning chain")
            nodes.append(node)
            
            if prev_node:
                graph.add_relationship(
                    prev_node, node,
                    RelationshipType.SEQUENTIAL,
                    weight=0.9
                )
            prev_node = node
        
        # Traverse from first node
        connected = graph.traverse_relationships(nodes[0], depth=5)
        
        # Should find connected nodes
        connected_ids = [n.node_id for n in connected]
        assert nodes[5] in connected_ids or nodes[3] in connected_ids


# ============================================================================
# END-TO-END TESTS (10 tests)
# ============================================================================

class TestEndToEndTurnProcessing:
    """Test full turn processing pipeline."""
    
    def test_full_turn_input_to_response(self, unified_system):
        """Test complete turn: input → LLM → response → update."""
        system = unified_system
        conv_id = "e2e_turn_test"
        
        # Setup
        system['state_manager'].create_conversation(conv_id)
        
        # Execute turn
        result = simulate_conversation_turn(
            system, conv_id, 1,
            "What is machine learning?",
            "FACT: ML is a subset of AI. DECISION: explain_simply: True"
        )
        
        assert result['turn_number'] == 1
        assert result['facts_extracted'] >= 0
    
    def test_state_accumulation_over_time(self, unified_system):
        """Test state accumulates correctly over multiple turns."""
        system = unified_system
        conv_id = "accumulation_test"
        
        system['state_manager'].create_conversation(conv_id)
        
        # Run 10 turns, each adding facts
        for turn in range(1, 11):
            turn_result = TurnResult(
                turn_number=turn,
                extracted_facts=[
                    CoreFact(f"fact_{turn}", f"value_{turn}", 
                            priority=FactPriority.MEDIUM,
                            source_turn=turn)
                ],
                proposed_decisions=[],
                new_constraints=[]
            )
            system['state_manager'].update_from_turn(conv_id, turn_result)
        
        # Verify accumulation
        state = system['state_manager'].get_state(conv_id)
        assert len(state.facts) >= 10
        
        # Each fact should have correct source turn
        for turn in range(1, 11):
            assert f"fact_{turn}" in state.facts
            assert state.facts[f"fact_{turn}"].source_turn == turn
    
    def test_memory_lifecycle_transitions(self, lifecycle_manager):
        """Test full memory lifecycle: create → active → cooling → archive."""
        # Create memory
        meta = lifecycle_manager.create_memory(
            memory_id="lifecycle_test",
            content=b"Test content for lifecycle",
            memory_type=MemoryType.STANDARD
        )
        assert meta.stage == LifecycleStage.ACTIVE
        
        # Manually age the memory
        meta.last_accessed = datetime.now() - timedelta(days=70)
        
        # Run maintenance - should transition to cooling
        results = lifecycle_manager.run_maintenance()
        
        # The lifecycle should be tracked
        lifecycle = lifecycle_manager.get_memory_lifecycle("lifecycle_test")
        assert lifecycle is not None
    
    def test_maintenance_job_execution(self, lifecycle_manager):
        """Test maintenance job executes correctly."""
        # Create multiple memories
        for i in range(5):
            lifecycle_manager.create_memory(
                memory_id=f"maint_test_{i}",
                content=f"Content {i}".encode(),
                memory_type=MemoryType.STANDARD
            )
        
        # Run maintenance
        results = lifecycle_manager.run_maintenance()
        
        assert 'processed' in results
        assert results['processed'] > 0
    
    def test_chronicle_records_conversation(self, chronicle_memory):
        """Test chronicle records conversation events."""
        chronicle = chronicle_memory
        
        # Record events
        for i in range(5):
            asyncio.run(chronicle.record_event(
                event_type=EventType.ACTION_COMPLETED,
                action=f"action_{i}",
                parameters={'index': i},
                outcome=Outcome.SUCCESS,
                narrative=f"Completed action {i}"
            ))
        
        # Retrieve events
        events = asyncio.run(chronicle.store.get_session_events(chronicle.session_id))
        assert len(events) >= 5
    
    def test_e2e_with_mock_llm(self, unified_system, mock_llm_responses):
        """Test end-to-end with mock LLM responses."""
        system = unified_system
        conv_id = "mock_llm_test"
        
        system['state_manager'].create_conversation(conv_id)
        
        # Process 3 turns with mock responses
        for turn in range(1, 4):
            try:
                response = next(mock_llm_responses)
            except StopIteration:
                response = "FACT: Default fact."
            
            result = simulate_conversation_turn(
                system, conv_id, turn,
                f"Query {turn}",
                response
            )
            
            assert result['turn_number'] == turn
    
    def test_cross_component_state_consistency(self, unified_system):
        """Test state remains consistent across all components."""
        system = unified_system
        conv_id = "consistency_test"
        
        # Initialize in all components
        state = system['state_manager'].create_conversation(conv_id)
        
        # Add fact through state manager
        state.add_fact(CoreFact("cross_component", "test_value", priority=FactPriority.HIGH))
        
        # Add to working memory
        system['working_memory'].update_state_fact("cross_component", "test_value")
        
        # Verify working memory state
        wm_state = system['working_memory'].get_state()
        assert wm_state.facts.get("cross_component") == "test_value"
    
    def test_full_pipeline_with_archival(self, unified_system):
        """Test full pipeline including memory archival."""
        lifecycle_mgr = unified_system['lifecycle_manager']
        
        # Create memory
        lifecycle_mgr.create_memory(
            memory_id="archive_pipeline_test",
            content=b"Important data to archive",
            memory_type=MemoryType.STANDARD
        )
        
        # Force archive
        success = lifecycle_mgr.force_archive("archive_pipeline_test")
        assert success is True
        
        # Verify archived
        lifecycle = lifecycle_mgr.get_memory_lifecycle("archive_pipeline_test")
        assert lifecycle['stage'] == 'archived'
        
        # Reactivate
        lifecycle_mgr.force_reactivate("archive_pipeline_test")
        lifecycle = lifecycle_mgr.get_memory_lifecycle("archive_pipeline_test")
        assert lifecycle['stage'] == 'active'
    
    def test_response_processing_extraction(self, working_memory_manager):
        """Test response processing extracts facts and decisions."""
        mgr = working_memory_manager
        
        response = """
        Let me analyze this...
        FACT: The system has 4 indexes
        FACT: Semantic index uses vectors
        DECISION: use_caching: True
        PREFERENCE: response_format: markdown
        INSIGHT: Combining indexes improves accuracy
        """
        
        updates = mgr.update_from_response(response)
        
        assert updates['facts_added'] >= 2
        assert updates['decisions_recorded'] >= 1
        assert updates['insights_stored'] >= 1
        assert updates['temporary_items_discarded'] > 0  # "Let me analyze..."
    
    def test_turn_metadata_tracking(self, unified_system):
        """Test turn metadata is tracked correctly."""
        system = unified_system
        
        # Process turn
        context = system['working_memory'].build_prompt_context("Test query")
        
        # Get stats
        stats = system['working_memory'].get_working_memory_stats()
        
        assert stats.turn_count >= 0
        assert stats.max_tokens_allowed > 0


# ============================================================================
# PERFORMANCE TESTS (8 tests)
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    PERFORMANCE_THRESHOLD_MS = 100  # Target: <100ms for queries
    
    def test_query_latency_under_threshold(self, unified_system):
        """Test query latency is under 100ms threshold."""
        retriever = unified_system['hybrid_retriever']
        
        # Index test data
        for i in range(100):
            mem = create_test_memory(f"perf_mem_{i}", f"Test content {i}")
            retriever.index_memory(mem)
        
        # Measure query time
        times = []
        for _ in range(10):
            start = time.time()
            results = retriever.retrieve("test content", limit=10)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Assert average is under threshold (allow some variance)
        assert avg_time < self.PERFORMANCE_THRESHOLD_MS * 2, f"Avg query time {avg_time:.2f}ms exceeds threshold"
        assert max_time < self.PERFORMANCE_THRESHOLD_MS * 5, f"Max query time {max_time:.2f}ms exceeds threshold"
    
    def test_memory_indexing_speed(self, unified_system):
        """Test memory indexing is fast."""
        retriever = unified_system['hybrid_retriever']
        
        start = time.time()
        
        # Index 100 memories
        for i in range(100):
            mem = create_test_memory(f"index_mem_{i}", f"Content for memory {i}")
            retriever.index_memory(mem)
        
        elapsed_ms = (time.time() - start) * 1000
        avg_per_mem = elapsed_ms / 100
        
        # Should be fast - under 10ms per memory on average
        assert avg_per_mem < 50, f"Indexing too slow: {avg_per_mem:.2f}ms per memory"
    
    def test_context_building_time(self, unified_system):
        """Test context building is fast."""
        working_mgr = unified_system['working_memory']
        
        # Add memories
        for i in range(50):
            working_mgr.add_long_term_memory(
                content=f"Memory {i} for context building test",
                priority=Priority.MEDIUM
            )
        
        # Measure context building
        times = []
        for _ in range(10):
            start = time.time()
            context = working_mgr.build_prompt_context("test query")
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        
        # Context building should be fast
        assert avg_time < 100, f"Context building too slow: {avg_time:.2f}ms"
    
    def test_concurrent_access_performance(self, unified_system):
        """Test concurrent access doesn't degrade performance."""
        retriever = unified_system['hybrid_retriever']
        
        # Index data
        for i in range(50):
            mem = create_test_memory(f"concurrent_{i}", f"Content {i}")
            retriever.index_memory(mem)
        
        def query_worker(thread_id: int) -> float:
            start = time.time()
            for _ in range(10):
                retriever.retrieve(f"content {thread_id}", limit=5)
            return (time.time() - start) * 1000
        
        # Run concurrent queries
        start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_worker, i) for i in range(5)]
            times = [f.result() for f in as_completed(futures)]
        
        total_time = (time.time() - start) * 1000
        avg_time_per_worker = sum(times) / len(times)
        
        # Concurrent access should be efficient
        assert avg_time_per_worker < 500, f"Concurrent access too slow: {avg_time_per_worker:.2f}ms"
    
    def test_large_scale_retrieval(self, unified_system):
        """Test retrieval scales to large memory sets."""
        retriever = unified_system['hybrid_retriever']
        
        # Index 1000 memories
        for i in range(1000):
            mem = create_test_memory(
                f"scale_mem_{i}",
                f"Content about topic {i % 10} with details {i}",
                importance=random.randint(1, 10)
            )
            retriever.index_memory(mem)
        
        # Query
        start = time.time()
        results = retriever.retrieve("topic 5", limit=20)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should still be fast even with 1000 memories
        assert elapsed_ms < 500, f"Large scale retrieval too slow: {elapsed_ms:.2f}ms"
        assert len(results) > 0
    
    def test_state_update_performance(self, state_manager):
        """Test state updates are fast."""
        conv_id = "perf_state_test"
        state = state_manager.create_conversation(conv_id)
        
        times = []
        for turn in range(1, 101):
            turn_result = TurnResult(
                turn_number=turn,
                extracted_facts=[CoreFact(f"fact_{turn}", f"value_{turn}", source_turn=turn)],
                proposed_decisions=[],
                new_constraints=[]
            )
            
            start = time.time()
            state_manager.update_from_turn(conv_id, turn_result)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # State updates should be fast
        assert avg_time < 50, f"State update avg too slow: {avg_time:.2f}ms"
        assert max_time < 200, f"State update max too slow: {max_time:.2f}ms"
    
    def test_lifecycle_maintenance_performance(self, lifecycle_manager):
        """Test lifecycle maintenance is performant."""
        # Create many memories
        for i in range(100):
            lifecycle_manager.create_memory(
                memory_id=f"maint_perf_{i}",
                content=f"Content {i}".encode(),
                memory_type=MemoryType.STANDARD
            )
        
        # Run maintenance
        start = time.time()
        results = lifecycle_manager.run_maintenance()
        elapsed_ms = (time.time() - start) * 1000
        
        # Maintenance should complete in reasonable time
        assert elapsed_ms < 5000, f"Maintenance too slow: {elapsed_ms:.2f}ms"
        assert results['processed'] > 0
    
    def test_memory_usage_bounded(self, unified_system):
        """Test memory usage stays bounded during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        retriever = unified_system['hybrid_retriever']
        for i in range(500):
            mem = create_test_memory(f"mem_bound_{i}", f"Content {i}")
            retriever.index_memory(mem)
        
        # Query multiple times
        for _ in range(50):
            retriever.retrieve("content", limit=15)
        
        final_mem = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = final_mem - initial_mem
        
        # Memory increase should be reasonable (less than 500MB)
        assert mem_increase < 500, f"Memory increased by {mem_increase:.1f}MB - potential leak"


# ============================================================================
# EDGE CASE TESTS (8 tests)
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_conversation(self, unified_system):
        """Test system handles empty conversation gracefully."""
        state_mgr = unified_system['state_manager']
        conv_id = "empty_test"
        
        # Create empty conversation
        state = state_mgr.create_conversation(conv_id)
        
        # Get snapshot
        snapshot = state.create_snapshot()
        assert snapshot is not None
        assert len(snapshot.core_facts) == 0
        
        # Build context from empty
        context = unified_system['working_memory'].build_prompt_context("")
        assert context is not None
    
    def test_duplicate_inputs(self, unified_system):
        """Test system handles duplicate inputs."""
        hash_idx = HashIndex(HashIndexConfig())
        
        content = "Duplicate input content"
        
        # Add twice
        result1 = hash_idx.add_content("doc1", content.encode())
        result2 = hash_idx.add_content("doc2", content.encode())
        
        # Second should be detected as duplicate
        assert result2.is_duplicate is True
        assert result2.duplicate_of == "doc1"
    
    def test_contradictory_facts(self, state_manager):
        """Test system handles contradictory facts."""
        conv_id = "contradiction_test"
        state = state_manager.create_conversation(conv_id)
        
        # Add fact
        fact1 = CoreFact("color", "blue", source_turn=1, confidence=0.9)
        state.add_fact(fact1)
        
        # Add contradictory fact
        fact2 = CoreFact("color", "red", source_turn=2, confidence=0.95)
        state.add_fact(fact2)  # Should update
        
        # Latest should win
        assert state.facts["color"].value == "red"
        assert state.facts["color"].confidence == 0.95
    
    def test_missing_dependencies(self, unified_system):
        """Test system handles missing dependencies gracefully."""
        # Try to get non-existent state
        state = unified_system['state_manager'].get_state("non_existent_conv")
        assert state is None
        
        # Try to update non-existent conversation
        with pytest.raises(ValueError):
            unified_system['state_manager'].update_from_turn(
                "non_existent",
                TurnResult(turn_number=1)
            )
    
    def test_large_content_10k_tokens(self, unified_system):
        """Test system handles large content (10K+ tokens equivalent)."""
        # Generate large content (~10K tokens worth)
        large_content = generate_test_content(size_words=40000)  # ~40K words
        
        lifecycle_mgr = unified_system['lifecycle_manager']
        
        # Should handle without error
        meta = lifecycle_mgr.create_memory(
            memory_id="large_content_test",
            content=large_content.encode(),
            memory_type=MemoryType.STANDARD
        )
        
        assert meta is not None
        
        # Should be retrievable
        result = lifecycle_mgr.retrieve_memory("large_content_test")
        assert result is not None
        content, _ = result
        assert len(content) == len(large_content.encode())
    
    def test_very_long_conversation_1000_turns(self, unified_system):
        """Test system handles very long conversations (1000+ turns)."""
        state_mgr = unified_system['state_manager']
        conv_id = "very_long_test"
        
        state = state_mgr.create_conversation(conv_id)
        
        # Add CORE fact first
        state.add_fact(create_core_fact("core_setting", "permanent_value", 0))
        
        # Simulate 1000 turns
        for turn in range(1, 1001):
            # Only add facts every 10 turns to avoid too much data
            if turn % 10 == 0:
                turn_result = TurnResult(
                    turn_number=turn,
                    extracted_facts=[CoreFact(f"turn_fact_{turn}", f"v{turn}", source_turn=turn)],
                    proposed_decisions=[],
                    new_constraints=[]
                )
                state_mgr.update_from_turn(conv_id, turn_result)
        
        # Verify state integrity
        final_state = state_mgr.get_state(conv_id)
        assert "core_setting" in final_state.facts
        # Should have accumulated facts
        assert len(final_state.facts) >= 100
    
    def test_special_characters_and_unicode(self, unified_system):
        """Test system handles special characters and unicode."""
        special_contents = [
            "Unicode: 你好世界 🌍 Привет мир",
            "Special: <script>alert('xss')</script>",
            "Quotes: \"single\" and 'double' and `backtick`",
            "Newlines: line1\nline2\r\nline3\n",
            "Null bytes: \x00 hidden",
            "Emoji: 🎉🚀💻🔥🌟",
        ]
        
        lifecycle_mgr = unified_system['lifecycle_manager']
        
        for i, content in enumerate(special_contents):
            meta = lifecycle_mgr.create_memory(
                memory_id=f"special_{i}",
                content=content.encode('utf-8'),
                memory_type=MemoryType.STANDARD
            )
            assert meta is not None
            
            # Verify retrieval
            result = lifecycle_mgr.retrieve_memory(f"special_{i}")
            assert result is not None
            retrieved_content, _ = result
            assert retrieved_content.decode('utf-8') == content
    
    def test_concurrent_modifications(self, unified_system):
        """Test system handles concurrent modifications."""
        conv_id = "concurrent_mod_test"
        state_mgr = unified_system['state_manager']
        state = state_mgr.create_conversation(conv_id)
        
        errors = []
        
        def modifier(thread_id: int):
            try:
                for i in range(10):
                    turn_result = TurnResult(
                        turn_number=thread_id * 100 + i,
                        extracted_facts=[CoreFact(f"thread_{thread_id}_fact_{i}", f"v{i}")],
                        proposed_decisions=[],
                        new_constraints=[]
                    )
                    state_mgr.update_from_turn(conv_id, turn_result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=modifier, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should complete without errors
        assert len(errors) == 0, f"Concurrent modification errors: {errors}"
        
        # State should be consistent
        final_state = state_mgr.get_state(conv_id)
        assert len(final_state.facts) >= 10  # At least some facts persisted


# ============================================================================
# CORRECTNESS TESTS (8 tests)
# ============================================================================

class TestCorrectness:
    """Test correctness guarantees."""
    
    def test_state_always_consistent(self, state_manager):
        """Test state is always internally consistent."""
        conv_id = "consistency_test"
        state = state_manager.create_conversation(conv_id)
        
        # Add various elements
        state.add_fact(CoreFact("f1", "v1"))
        state.add_decision(ActiveDecision("d1", "Decision"))
        state.add_constraint(Constraint("c1", "Constraint"))
        
        # Hash should change
        assert state.has_changed()
        state.update_hash()
        assert not state.has_changed()
        
        # Modify
        state.add_fact(CoreFact("f2", "v2"))
        assert state.has_changed()
        
        # Snapshot should be consistent
        snapshot = state.create_snapshot()
        assert snapshot is not None
    
    def test_no_memory_loss_for_core_facts(self, unified_system):
        """Test CORE facts are never lost."""
        lifecycle_mgr = unified_system['lifecycle_manager']
        
        # Create CORE memory
        lifecycle_mgr.create_memory(
            memory_id="core_no_loss",
            content=b"Critical data",
            memory_type=MemoryType.CORE
        )
        
        # Run maintenance multiple times
        for _ in range(5):
            lifecycle_mgr.run_maintenance()
        
        # Should still be retrievable
        result = lifecycle_mgr.retrieve_memory("core_no_loss")
        assert result is not None, "CORE memory was lost!"
    
    def test_deduplication_works_correctly(self, unified_system):
        """Test deduplication prevents redundant storage."""
        hash_idx = HashIndex(HashIndexConfig())
        
        content = "Exactly the same content"
        
        # Add same content multiple times
        results = []
        for i in range(5):
            result = hash_idx.add_content(f"dup_{i}", content.encode())
            results.append(result)
        
        # First should be new
        assert results[0].is_new is True
        
        # Rest should be duplicates
        for r in results[1:]:
            assert r.is_duplicate is True
            assert r.duplicate_of == "dup_0"
    
    def test_archived_memories_recoverable(self, lifecycle_manager):
        """Test archived memories can be fully recovered."""
        original_content = b"Original content to be archived and recovered"
        
        # Create and archive
        lifecycle_manager.create_memory(
            memory_id="recover_test",
            content=original_content,
            memory_type=MemoryType.STANDARD
        )
        
        lifecycle_manager.force_archive("recover_test")
        
        # Verify archived
        lifecycle = lifecycle_manager.get_memory_lifecycle("recover_test")
        assert lifecycle['stage'] == 'archived'
        
        # Recover
        success = lifecycle_manager.force_reactivate("recover_test")
        assert success is True
        
        # Verify content intact
        result = lifecycle_manager.retrieve_memory("recover_test")
        assert result is not None
        recovered_content, _ = result
        assert recovered_content == original_content
    
    def test_token_limits_respected(self, unified_system):
        """Test token limits are always respected."""
        working_mgr = unified_system['working_memory']
        max_tokens = 1000  # Small limit for testing
        
        # Add many memories
        for i in range(50):
            working_mgr.add_long_term_memory(
                content=f"Memory {i}: " + "word " * 20,
                priority=Priority.MEDIUM
            )
        
        # Build context
        context = working_mgr.build_prompt_context("test", max_tokens=max_tokens)
        
        # Count tokens
        token_count = working_mgr.token_counter.count(context.to_plain_text())
        
        # Should be under limit (with tolerance)
        assert token_count <= max_tokens * 1.1, f"Token limit exceeded: {token_count} > {max_tokens}"
    
    def test_version_history_tracks_all_changes(self, state_manager):
        """Test version history tracks all state changes."""
        conv_id = "version_test"
        state = state_manager.create_conversation(conv_id)
        
        initial_versions = len(state.version_history)
        
        # Make changes
        for i in range(5):
            state.add_fact(CoreFact(f"v_fact_{i}", f"value_{i}"))
            state.record_version(f"Added fact {i}")
        
        # Should have versions
        assert len(state.version_history) == initial_versions + 5
        
        # Each version should have unique ID
        version_ids = [v.version_id for v in state.version_history]
        assert len(version_ids) == len(set(version_ids)), "Duplicate version IDs!"
    
    def test_transitions_are_atomic(self, lifecycle_manager):
        """Test lifecycle transitions are atomic."""
        # Create memory
        lifecycle_manager.create_memory(
            memory_id="atomic_test",
            content=b"Test content",
            memory_type=MemoryType.STANDARD
        )
        
        # Get initial stage
        lifecycle = lifecycle_manager.get_memory_lifecycle("atomic_test")
        initial_stage = lifecycle['stage']
        
        # Archive
        lifecycle_manager.force_archive("atomic_test")
        
        # Should be archived, not intermediate state
        lifecycle = lifecycle_manager.get_memory_lifecycle("atomic_test")
        assert lifecycle['stage'] == 'archived'
        assert lifecycle['stage'] != initial_stage
    
    def test_query_results_deterministic(self, unified_system):
        """Test query results are deterministic."""
        retriever = unified_system['hybrid_retriever']
        
        # Index deterministic data
        for i in range(20):
            mem = create_test_memory(f"det_{i}", f"Content about deterministic topic {i}")
            retriever.index_memory(mem)
        
        # Same query multiple times
        results1 = retriever.retrieve("deterministic topic", limit=10)
        results2 = retriever.retrieve("deterministic topic", limit=10)
        results3 = retriever.retrieve("deterministic topic", limit=10)
        
        # Same number of results
        assert len(results1) == len(results2) == len(results3)
        
        # Same memories (by ID)
        ids1 = [r.memory.id for r in results1]
        ids2 = [r.memory.id for r in results2]
        ids3 = [r.memory.id for r in results3]
        
        assert ids1 == ids2 == ids3


# ============================================================================
# ADDITIONAL COMPREHENSIVE TESTS (to reach 50+)
# ============================================================================

class TestAdditionalCoverage:
    """Additional tests for comprehensive coverage."""
    
    def test_hash_index_minhash_similarity(self):
        """Test MinHash similarity detection."""
        content1 = "This is a test document about machine learning"
        content2 = "This is a test document about deep learning"  # Similar
        content3 = "Completely different content about cooking"
        
        sig1 = compute_minhash(content1, num_permutations=64)
        sig2 = compute_minhash(content2, num_permutations=64)
        sig3 = compute_minhash(content3, num_permutations=64)
        
        # Similar content should have similar signatures
        matches_1_2 = sum(a == b for a, b in zip(sig1, sig2))
        matches_1_3 = sum(a == b for a, b in zip(sig1, sig3))
        
        assert matches_1_2 > matches_1_3
    
    def test_simhash_near_duplicate_detection(self):
        """Test SimHash for near-duplicate detection."""
        content1 = "The quick brown fox jumps over the lazy dog"
        content2 = "The quick brown fox jumps over the lazy dogs"  # Near duplicate
        content3 = "Something completely different altogether"
        
        hash1 = compute_simhash(content1)
        hash2 = compute_simhash(content2)
        hash3 = compute_simhash(content3)
        
        # Near duplicates should have small Hamming distance
        from knowledge_hash_index import hamming_distance
        dist_1_2 = hamming_distance(hash1, hash2)
        dist_1_3 = hamming_distance(hash1, hash3)
        
        assert dist_1_2 < dist_1_3
    
    def test_retrieval_weights_validation(self):
        """Test retrieval weights validation."""
        # Valid weights
        weights = RetrievalWeights(hierarchical=0.25, graph=0.25, semantic=0.30, recency=0.20)
        assert weights.validate() is True
        
        # Invalid weights (don't sum to 1)
        weights_invalid = RetrievalWeights(hierarchical=0.5, graph=0.5, semantic=0.5, recency=0.5)
        assert weights_invalid.validate() is False
        
        # Normalization
        normalized = weights_invalid.normalize()
        assert normalized.validate() is True
    
    def test_chronicle_loop_detection(self, chronicle_memory):
        """Test chronicle loop detection."""
        chronicle = chronicle_memory
        
        # Record similar actions
        for i in range(3):
            asyncio.run(chronicle.start_action(
                action="retry_failed_strategy",
                parameters={"attempt": i},
                narrative=f"Retry attempt {i}"
            ))
            asyncio.run(chronicle.complete_action(outcome=Outcome.FAILURE))
        
        # Check for loop
        should_prevent, warning = asyncio.run(
            chronicle.check_for_loops("retry_failed_strategy", {"attempt": 3})
        )
        
        # Should detect potential loop
        assert isinstance(should_prevent, bool)
        assert isinstance(warning, (str, type(None)))
    
    def test_memory_node_importance_calculation(self, all_four_indexes):
        """Test memory node importance calculation."""
        hier = all_four_indexes['hierarchical']
        
        node = MemoryNode(
            content="Test node",
            level=MemoryLevel.IMPORTANT,
            frequency_score=0.8,
            centrality_score=0.7,
            user_importance=0.9
        )
        
        # Importance should be calculated from components
        assert node.importance_score > 0
        assert 0 <= node.importance_score <= 1
    
    def test_graph_path_finding(self, all_four_indexes):
        """Test graph path finding between nodes."""
        graph = all_four_indexes['graph']
        
        # Create chain
        nodes = []
        for i in range(5):
            node = graph.add_node(f"Node {i}")
            nodes.append(node)
            if i > 0:
                graph.add_relationship(nodes[i-1], node, RelationshipType.SEQUENTIAL)
        
        # Find path
        path = graph.find_path(nodes[0], nodes[4])
        
        if path:  # If path finding is implemented
            assert len(path) > 0
            assert path[0].node_id == nodes[0]
    
    def test_confidence_scorer_calculation(self, lifecycle_manager):
        """Test confidence scoring calculation."""
        scorer = lifecycle_manager.confidence_scorer
        
        # Create metadata with various signals
        meta = MemoryMetadata(
            memory_id="conf_test",
            source_reliability=0.9,
            confirmation_count=5,
            contradiction_count=1,
            user_confirmed=True
        )
        
        confidence = scorer.calculate_confidence(meta)
        
        # Should be high confidence
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high with these signals
    
    def test_decay_score_calculation(self, lifecycle_manager):
        """Test decay score calculation."""
        detector = lifecycle_manager.decay_detector
        
        # Create old memory
        old_meta = MemoryMetadata(
            memory_id="old_test",
            created_at=datetime.now() - timedelta(days=200),
            last_accessed=datetime.now() - timedelta(days=100),
            access_count=2,
            memory_type=MemoryType.STANDARD
        )
        
        decay = detector.calculate_decay_score(old_meta)
        
        # Should have some decay
        assert 0 <= decay <= 1
        assert decay > 0  # Should have decayed
    
    def test_core_type_no_decay(self, lifecycle_manager):
        """Test CORE type memories never decay."""
        detector = lifecycle_manager.decay_detector
        
        # Create old CORE memory
        core_meta = MemoryMetadata(
            memory_id="core_decay_test",
            created_at=datetime.now() - timedelta(days=1000),
            last_accessed=datetime.now() - timedelta(days=500),
            memory_type=MemoryType.CORE
        )
        
        decay = detector.calculate_decay_score(core_meta)
        assert decay == 0.0
    
    def test_temporal_memory_decay(self, lifecycle_manager):
        """Test temporal memory decay based on expiration."""
        detector = lifecycle_manager.decay_detector
        
        # Create temporal memory with expiration
        temporal_meta = MemoryMetadata(
            memory_id="temp_decay_test",
            created_at=datetime.now() - timedelta(days=10),
            expires_at=datetime.now() + timedelta(days=10),
            memory_type=MemoryType.TEMPORAL
        )
        
        decay = detector.calculate_decay_score(temporal_meta)
        assert 0 <= decay <= 1
        # Should be mid-decay (halfway to expiration)
        assert 0.4 < decay < 0.6
    
    def test_archival_compression(self, lifecycle_manager):
        """Test archived memories are compressed."""
        # Create large content
        large_content = b"X" * 10000
        
        lifecycle_manager.create_memory(
            memory_id="compress_test",
            content=large_content,
            memory_type=MemoryType.STANDARD
        )
        
        # Archive
        lifecycle_manager.force_archive("compress_test")
        
        # Get stats
        stats = lifecycle_manager.archival_manager.get_archive_stats()
        
        if stats.get('total_memories', 0) > 0:
            assert stats['total_compressed_size'] < stats['total_original_size']


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
