"""
Comprehensive tests for EntityKnowledgeGraph and KnowledgeState

Following CLAUDE.md principles:
- RUNTIME TRUTH: Test actual behavior, not assumptions
- IDEMPOTENCY: Verify operations are safe to retry
- ZERO TRUST: Validate all inputs and outputs
- STRUCTURED LOGGING: Verify logging with correlation IDs

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

# Import core components
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_engine.core import (
    EntityKnowledgeGraph,
    KnowledgeState,
    Entity,
    Relationship,
    KnowledgeTriple,
    StateSnapshot
)


class TestEntityKnowledgeGraph:
    """Test suite for EntityKnowledgeGraph."""

    def test_initialization(self):
        """Test graph initialization with correlation ID."""
        graph = EntityKnowledgeGraph(correlation_id="test-123")

        stats = graph.get_statistics()
        assert stats['entity_count'] == 0
        assert stats['relationship_count'] == 0
        assert stats['correlation_id'] == "test-123"
        print("[OK] Graph initialization successful")

    def test_add_entity(self):
        """Test adding entities to the graph."""
        graph = EntityKnowledgeGraph()

        # Add single entity
        result = graph.add_entity(
            name="Alice",
            entity_type="Person",
            attributes={"age": 30, "city": "NYC"}
        )

        assert result is True
        entity = graph.get_entity("Alice")
        assert entity is not None
        assert entity['name'] == "Alice"
        assert entity['entity_type'] == "Person"
        assert entity['attributes']['age'] == 30
        print("[OK] Entity added successfully")

    def test_add_entity_idempotent(self):
        """Test that adding same entity twice is idempotent."""
        graph = EntityKnowledgeGraph()

        # Add entity first time
        graph.add_entity("Bob", "Person", {"age": 25})
        entity_v1 = graph.get_entity("Bob")

        # Add same entity again with different attributes
        graph.add_entity("Bob", "Person", {"city": "LA"})
        entity_v2 = graph.get_entity("Bob")

        # Should have merged attributes
        assert entity_v2['attributes']['age'] == 25
        assert entity_v2['attributes']['city'] == "LA"
        assert entity_v2['updated_at'] >= entity_v1['updated_at']
        print("[OK] Entity addition is idempotent")

    def test_add_entity_validation(self):
        """Test input validation for entity addition."""
        graph = EntityKnowledgeGraph()

        # Test invalid inputs
        result = graph.add_entity("", "Person")
        assert result is False

        result = graph.add_entity("Test", "")
        assert result is False

        result = graph.add_entity(None, "Person")
        assert result is False

        print("[OK] Entity validation works correctly")

    def test_add_relationship(self):
        """Test adding relationships between entities."""
        graph = EntityKnowledgeGraph()

        # Add entities
        graph.add_entity("Alice", "Person")
        graph.add_entity("Bob", "Person")

        # Add relationship
        result = graph.add_relationship(
            source="Alice",
            target="Bob",
            relation_type="KNOWS",
            attributes={"since": "2020"}
        )

        assert result is True

        # Check relationship
        relationships = graph.get_relationships("Alice")
        assert len(relationships) == 1
        assert relationships[0]['source'] == "Alice"
        assert relationships[0]['target'] == "Bob"
        assert relationships[0]['relation_type'] == "KNOWS"
        print("[OK] Relationship added successfully")

    def test_add_relationship_idempotent(self):
        """Test that adding same relationship twice is idempotent."""
        graph = EntityKnowledgeGraph()

        graph.add_entity("A", "Entity")
        graph.add_entity("B", "Entity")

        # Add relationship first time
        result1 = graph.add_relationship("A", "B", "LINKS_TO")
        relationships_v1 = graph.get_relationships("A")

        # Add same relationship again
        result2 = graph.add_relationship("A", "B", "LINKS_TO")
        relationships_v2 = graph.get_relationships("A")

        assert result1 is True
        assert result2 is True
        assert len(relationships_v1) == len(relationships_v2) == 1
        print("[OK] Relationship addition is idempotent")

    def test_find_entities_by_type(self):
        """Test finding entities by type."""
        graph = EntityKnowledgeGraph()

        graph.add_entity("Alice", "Person", {"role": "engineer"})
        graph.add_entity("Bob", "Person", {"role": "manager"})
        graph.add_entity("CompanyX", "Organization")

        # Find all persons
        persons = graph.find_entities(entity_type="Person")
        assert len(persons) == 2
        assert all(p['entity_type'] == "Person" for p in persons)
        print("[OK] Finding entities by type works")

    def test_find_entities_by_attributes(self):
        """Test finding entities by attributes."""
        graph = EntityKnowledgeGraph()

        graph.add_entity("Alice", "Person", {"role": "engineer", "level": 5})
        graph.add_entity("Bob", "Person", {"role": "manager", "level": 6})
        graph.add_entity("Charlie", "Person", {"role": "engineer", "level": 4})

        # Find engineers at level 5
        results = graph.find_entities(
            entity_type="Person",
            attributes={"role": "engineer", "level": 5}
        )
        assert len(results) == 1
        assert results[0]['name'] == "Alice"
        print("[OK] Finding entities by attributes works")

    def test_search_entities(self):
        """Test searching entities by query."""
        graph = EntityKnowledgeGraph()

        graph.add_entity("MachineLearning", "Topic", {"description": "AI algorithms"})
        graph.add_entity("Python", "Language", {"description": "Programming language"})

        # Search for "machine"
        results = graph.search_entities("machine")
        assert len(results) == 1
        assert results[0]['name'] == "MachineLearning"

        # Search for "programming"
        results = graph.search_entities("programming")
        assert len(results) == 1
        assert results[0]['name'] == "Python"
        print("[OK] Entity search works correctly")

    def test_to_json(self):
        """Test serializing graph to JSON."""
        graph = EntityKnowledgeGraph(correlation_id="test-json")

        graph.add_entity("Alice", "Person", {"age": 30})
        graph.add_entity("Bob", "Person", {"age": 25})
        graph.add_relationship("Alice", "Bob", "KNOWS")

        json_str = graph.to_json()
        data = json.loads(json_str)

        assert data['metadata']['entity_count'] == 2
        assert data['metadata']['relationship_count'] == 1
        assert len(data['entities']) == 2
        assert len(data['relationships']) == 1
        assert data['metadata']['correlation_id'] == "test-json"
        print("[OK] JSON serialization works")

    def test_from_json(self):
        """Test loading graph from JSON."""
        graph1 = EntityKnowledgeGraph()

        graph1.add_entity("Alice", "Person", {"age": 30})
        graph1.add_entity("Bob", "Person", {"age": 25})
        graph1.add_relationship("Alice", "Bob", "KNOWS")

        json_str = graph1.to_json()

        # Load into new graph
        graph2 = EntityKnowledgeGraph()
        result = graph2.from_json(json_str)

        assert result is True
        stats = graph2.get_statistics()
        assert stats['entity_count'] == 2
        assert stats['relationship_count'] == 1

        # Verify entities
        alice = graph2.get_entity("Alice")
        assert alice is not None
        assert alice['attributes']['age'] == 30
        print("[OK] JSON deserialization works")

    def test_get_statistics(self):
        """Test getting graph statistics."""
        graph = EntityKnowledgeGraph()

        graph.add_entity("Alice", "Person")
        graph.add_entity("Bob", "Person")
        graph.add_entity("Company", "Organization")
        graph.add_relationship("Alice", "Bob", "KNOWS")

        stats = graph.get_statistics()
        assert stats['entity_count'] == 3
        assert stats['relationship_count'] == 1
        assert 'Person' in stats['entity_types']
        assert stats['entity_types']['Person'] == 2
        print("[OK] Statistics calculation works")

    def test_clear(self):
        """Test clearing the graph."""
        graph = EntityKnowledgeGraph()

        graph.add_entity("Alice", "Person")
        graph.add_relationship("Alice", "Bob", "KNOWS")

        # Clear graph
        graph.clear()

        stats = graph.get_statistics()
        assert stats['entity_count'] == 0
        assert stats['relationship_count'] == 0
        print("[OK] Graph clearing works")

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test asynchronous graph operations."""
        graph = EntityKnowledgeGraph()

        # Add entities asynchronously
        await graph.add_entity_async("Alice", "Person", {"age": 30})
        await graph.add_entity_async("Bob", "Person", {"age": 25})

        # Add relationship asynchronously
        await graph.add_relationship_async("Alice", "Bob", "KNOWS")

        # Verify
        entity = await graph.get_entity_async("Alice")
        assert entity is not None
        assert entity['name'] == "Alice"

        relationships = await graph.get_relationships_async("Alice")
        assert len(relationships) == 1

        # Test async JSON
        json_str = await graph.to_json_async()
        data = json.loads(json_str)
        assert data['metadata']['entity_count'] == 2

        print("[OK] Async operations work correctly")


class TestKnowledgeState:
    """Test suite for KnowledgeState."""

    def test_initialization(self):
        """Test state initialization."""
        state = KnowledgeState(
            query="What is AI?",
            correlation_id="test-state-123"
        )

        stats = state.get_statistics()
        assert stats['query'] == "What is AI?"
        assert stats['triple_count'] == 0
        assert stats['correlation_id'] == "test-state-123"
        print("[OK] State initialization successful")

    def test_add_knowledge(self):
        """Test adding knowledge triples."""
        state = KnowledgeState(query="Test query")

        triples = [
            ("AI", "is", "technology"),
            ("AI", "includes", "machine learning"),
            ("machine learning", "is", "subset of AI")
        ]

        result = state.add_knowledge(triples, source="test")
        assert result is True

        current = state.get_current_state()
        assert current['triple_count'] == 3
        print("[OK] Knowledge added successfully")

    def test_add_knowledge_idempotent(self):
        """Test that adding same knowledge twice is idempotent."""
        state = KnowledgeState(query="Test query")

        triples = [("AI", "is", "technology")]

        # Add first time
        state.add_knowledge(triples)
        stats_v1 = state.get_statistics()

        # Add same knowledge again
        state.add_knowledge(triples)
        stats_v2 = state.get_statistics()

        # Should not duplicate
        assert stats_v1['triple_count'] == stats_v2['triple_count'] == 1
        print("[OK] Knowledge addition is idempotent")

    def test_add_fact(self):
        """Test adding facts."""
        state = KnowledgeState(query="Test query")

        result = state.add_fact("AI is a branch of computer science")
        assert result is True

        result = state.add_fact("AI is a branch of computer science")  # Duplicate
        assert result is False  # Should return False for duplicate

        current = state.get_current_state()
        assert len(current['facts']) == 1
        print("[OK] Facts added successfully")

    def test_add_uncertainty(self):
        """Test adding uncertainties."""
        state = KnowledgeState(query="Test query")

        result = state.add_uncertainty("It is unclear when AGI will be achieved")
        assert result is True

        current = state.get_current_state()
        assert len(current['uncertainties']) == 1
        print("[OK] Uncertainties added successfully")

    def test_snapshots(self):
        """Test temporal snapshots."""
        state = KnowledgeState(query="Test query")

        # Add knowledge at different times
        timestamp1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        state.add_knowledge([("AI", "is", "technology")], timestamp=timestamp1)

        timestamp2 = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        state.add_knowledge([("AI", "includes", "ML")], timestamp=timestamp2)

        # Get history
        history = state.get_history()
        assert len(history) == 2
        assert history[0]['timestamp'] == timestamp1
        assert history[1]['timestamp'] == timestamp2
        print("[OK] Temporal snapshots work correctly")

    def test_get_state_at_time(self):
        """Test getting state at specific time."""
        state = KnowledgeState(query="Test query")

        timestamp1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        state.add_knowledge([("A", "relates", "B")], timestamp=timestamp1)

        timestamp2 = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc).isoformat()
        state.add_knowledge([("B", "relates", "C")], timestamp=timestamp2)

        # Query state at timestamp1
        state_at_t1 = state.get_state_at_time(timestamp1)
        assert state_at_t1 is not None
        assert len(state_at_t1['triples']) == 1
        assert state_at_t1['triples'][0]['subject'] == "A"

        # Query state at timestamp2
        state_at_t2 = state.get_state_at_time(timestamp2)
        assert state_at_t2 is not None
        assert len(state_at_t2['triples']) == 2
        print("[OK] Getting state at time works")

    def test_search_triples(self):
        """Test searching triples by pattern."""
        state = KnowledgeState(query="Test query")

        triples = [
            ("AI", "is", "technology"),
            ("AI", "includes", "ML"),
            ("ML", "is", "subset of AI"),
            ("Python", "is", "language")
        ]
        state.add_knowledge(triples)

        # Search by subject
        results = state.search_triples(subject="AI")
        assert len(results) == 2

        # Search by predicate
        results = state.search_triples(predicate="is")
        assert len(results) == 3

        # Search by object
        results = state.search_triples(obj="ML")
        assert len(results) == 2
        print("[OK] Triple search works correctly")

    def test_to_dict(self):
        """Test serializing state to dictionary."""
        state = KnowledgeState(query="Test query", correlation_id="test-dict")

        state.add_knowledge([("AI", "is", "technology")])
        state.add_fact("AI is important")
        state.add_uncertainty("AI future is unclear")

        data = state.to_dict()
        assert data['query'] == "Test query"
        assert data['triple_count'] == 1
        assert len(data['facts']) == 1
        assert len(data['uncertainties']) == 1
        assert data['correlation_id'] == "test-dict"
        print("[OK] Dictionary serialization works")

    def test_from_dict(self):
        """Test loading state from dictionary."""
        state1 = KnowledgeState(query="Test query")

        state1.add_knowledge([("AI", "is", "technology")])
        state1.add_fact("AI is important")

        data = state1.to_dict()

        # Load into new state
        state2 = KnowledgeState.from_dict(data)

        assert state2.query == "Test query"
        assert state2._version == state1._version
        assert len(state2._triples) == 1
        assert len(state2._facts) == 1
        print("[OK] Dictionary deserialization works")

    def test_to_json(self):
        """Test serializing state to JSON."""
        state = KnowledgeState(query="Test query")

        state.add_knowledge([("AI", "is", "technology")])
        state.add_fact("AI is important")

        json_str = state.to_json()
        data = json.loads(json_str)

        assert data['query'] == "Test query"
        assert data['triple_count'] == 1
        assert len(data['facts']) == 1
        print("[OK] JSON serialization works")

    def test_from_json(self):
        """Test loading state from JSON."""
        state1 = KnowledgeState(query="Test query")

        state1.add_knowledge([("AI", "is", "technology")])
        state1.add_fact("AI is important")

        json_str = state1.to_json()

        # Load into new state
        state2 = KnowledgeState.from_json(json_str)

        assert state2.query == "Test query"
        assert len(state2._triples) == 1
        assert len(state2._facts) == 1
        print("[OK] JSON deserialization works")

    def test_clear(self):
        """Test clearing state."""
        state = KnowledgeState(query="Test query")

        state.add_knowledge([("AI", "is", "technology")])
        state.add_fact("AI is important")

        # Clear state
        state.clear()

        stats = state.get_statistics()
        assert stats['triple_count'] == 0
        assert stats['fact_count'] == 0
        assert stats['version'] == 0
        print("[OK] State clearing works")

    def test_get_statistics(self):
        """Test getting state statistics."""
        state = KnowledgeState(query="What is AI?")

        state.add_knowledge([("AI", "is", "technology")])
        state.add_fact("AI is a branch of CS")
        state.add_uncertainty("AGI timeline unknown")

        stats = state.get_statistics()
        assert stats['triple_count'] == 1
        assert stats['fact_count'] == 1
        assert stats['uncertainty_count'] == 1
        assert stats['query'] == "What is AI?"
        print("[OK] Statistics calculation works")

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test asynchronous state operations."""
        state = KnowledgeState(query="Test query")

        # Add knowledge asynchronously
        await state.add_knowledge_async([("AI", "is", "technology")])
        await state.add_fact_async("AI is important")

        # Get current state asynchronously
        current = await state.get_current_state_async()
        assert current['triple_count'] == 1
        assert len(current['facts']) == 1

        # Test async JSON
        json_str = await state.to_json_async()
        data = json.loads(json_str)
        assert data['query'] == "Test query"

        # Test async statistics
        stats = await state.get_statistics_async()
        assert stats['triple_count'] == 1

        print("[OK] Async operations work correctly")


class TestIntegration:
    """Integration tests for EntityKnowledgeGraph and KnowledgeState."""

    def test_graph_and_state_integration(self):
        """Test using graph and state together."""
        graph = EntityKnowledgeGraph()
        state = KnowledgeState(query="Build knowledge about AI")

        # Extract entities from triples and add to graph
        triples = [
            ("AI", "is", "technology"),
            ("AI", "includes", "Machine Learning"),
            ("Machine Learning", "uses", "Neural Networks")
        ]

        state.add_knowledge(triples)

        # Add entities to graph
        graph.add_entity("AI", "Topic", {"description": "Artificial Intelligence"})
        graph.add_entity("Machine Learning", "Topic", {"description": "ML"})
        graph.add_entity("Neural Networks", "Topic", {"description": "NN"})

        # Add relationships
        graph.add_relationship("AI", "Machine Learning", "INCLUDES")
        graph.add_relationship("Machine Learning", "Neural Networks", "USES")

        # Verify both are consistent
        state_triples = state.search_triples(subject="AI")
        graph_rels = graph.get_relationships("AI")

        assert len(state_triples) == 2
        assert len(graph_rels) == 1
        print("[OK] Graph and state integration works")

    def test_serialization_roundtrip(self):
        """Test complete serialization roundtrip."""
        # Create and populate graph
        graph1 = EntityKnowledgeGraph(correlation_id="test-roundtrip")
        graph1.add_entity("Alice", "Person", {"age": 30})
        graph1.add_entity("Bob", "Person", {"age": 25})
        graph1.add_relationship("Alice", "Bob", "KNOWS")

        # Create and populate state
        state1 = KnowledgeState(query="Social network", correlation_id="test-roundtrip")
        state1.add_knowledge([("Alice", "knows", "Bob")])
        state1.add_fact("Alice and Bob are colleagues")

        # Serialize both
        graph_json = graph1.to_json()
        state_json = state1.to_json()

        # Load into new instances
        graph2 = EntityKnowledgeGraph()
        graph2.from_json(graph_json)

        state2 = KnowledgeState.from_json(state_json)

        # Verify
        assert graph2.get_statistics()['entity_count'] == 2
        assert state2.get_statistics()['triple_count'] == 1
        assert graph2._correlation_id == "test-roundtrip"
        assert state2._correlation_id == "test-roundtrip"

        print("[OK] Serialization roundtrip successful")


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing EntityKnowledgeGraph and KnowledgeState")
    print("="*60 + "\n")

    # Test EntityKnowledgeGraph
    print("Testing EntityKnowledgeGraph...")
    graph_tests = TestEntityKnowledgeGraph()

    graph_tests.test_initialization()
    graph_tests.test_add_entity()
    graph_tests.test_add_entity_idempotent()
    graph_tests.test_add_entity_validation()
    graph_tests.test_add_relationship()
    graph_tests.test_add_relationship_idempotent()
    graph_tests.test_find_entities_by_type()
    graph_tests.test_find_entities_by_attributes()
    graph_tests.test_search_entities()
    graph_tests.test_to_json()
    graph_tests.test_from_json()
    graph_tests.test_get_statistics()
    graph_tests.test_clear()

    # Run async test
    asyncio.run(graph_tests.test_async_operations())

    print("\n[OK] All EntityKnowledgeGraph tests passed!\n")

    # Test KnowledgeState
    print("Testing KnowledgeState...")
    state_tests = TestKnowledgeState()

    state_tests.test_initialization()
    state_tests.test_add_knowledge()
    state_tests.test_add_knowledge_idempotent()
    state_tests.test_add_fact()
    state_tests.test_add_uncertainty()
    state_tests.test_snapshots()
    state_tests.test_get_state_at_time()
    state_tests.test_search_triples()
    state_tests.test_to_dict()
    state_tests.test_from_dict()
    state_tests.test_to_json()
    state_tests.test_from_json()
    state_tests.test_clear()
    state_tests.test_get_statistics()

    # Run async test
    asyncio.run(state_tests.test_async_operations())

    print("\n[OK] All KnowledgeState tests passed!\n")

    # Test Integration
    print("Testing Integration...")
    integration_tests = TestIntegration()

    integration_tests.test_graph_and_state_integration()
    integration_tests.test_serialization_roundtrip()

    print("\n[OK] All Integration tests passed!\n")

    print("="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_tests()
