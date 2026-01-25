"""
Simple test runner for core graph modules
"""
import sys
sys.path.insert(0, '.')

from knowledge_engine.core import EntityKnowledgeGraph, KnowledgeState
import json
import asyncio

def test_entity_knowledge_graph():
    """Test EntityKnowledgeGraph functionality."""
    print("="*60)
    print("Testing EntityKnowledgeGraph")
    print("="*60)

    # Test 1: Initialization
    print("\n1. Testing initialization...")
    graph = EntityKnowledgeGraph(correlation_id="test-123")
    stats = graph.get_statistics()
    assert stats['entity_count'] == 0
    assert stats['relationship_count'] == 0
    print("   PASS: Graph initialization")

    # Test 2: Add entity
    print("\n2. Testing add_entity...")
    result = graph.add_entity("Alice", "Person", {"age": 30, "city": "NYC"})
    assert result is True
    entity = graph.get_entity("Alice")
    assert entity is not None
    assert entity['name'] == "Alice"
    assert entity['entity_type'] == "Person"
    print("   PASS: Entity added")

    # Test 3: Idempotent entity addition
    print("\n3. Testing idempotent entity addition...")
    graph.add_entity("Bob", "Person", {"age": 25})
    graph.add_entity("Bob", "Person", {"city": "LA"})
    entity = graph.get_entity("Bob")
    assert entity['attributes']['age'] == 25
    assert entity['attributes']['city'] == "LA"
    print("   PASS: Idempotent addition works")

    # Test 4: Add relationship
    print("\n4. Testing add_relationship...")
    graph.add_relationship("Alice", "Bob", "KNOWS", {"since": "2020"})
    relationships = graph.get_relationships("Alice")
    assert len(relationships) == 1
    assert relationships[0]['relation_type'] == "KNOWS"
    print("   PASS: Relationship added")

    # Test 5: Find entities
    print("\n5. Testing find_entities...")
    graph.add_entity("Charlie", "Person", {"role": "engineer"})
    persons = graph.find_entities(entity_type="Person")
    assert len(persons) == 3
    print("   PASS: Finding entities by type works")

    # Test 6: Search entities
    print("\n6. Testing search_entities...")
    results = graph.search_entities("Alice")
    assert len(results) == 1
    assert results[0]['name'] == "Alice"
    print("   PASS: Entity search works")

    # Test 7: JSON serialization
    print("\n7. Testing JSON serialization...")
    json_str = graph.to_json()
    data = json.loads(json_str)
    assert data['metadata']['entity_count'] == 3
    assert data['metadata']['relationship_count'] == 1
    print("   PASS: JSON serialization works")

    # Test 8: JSON deserialization
    print("\n8. Testing JSON deserialization...")
    graph2 = EntityKnowledgeGraph()
    result = graph2.from_json(json_str)
    assert result is True
    stats = graph2.get_statistics()
    assert stats['entity_count'] == 3
    assert stats['relationship_count'] == 1
    print("   PASS: JSON deserialization works")

    # Test 9: Async operations
    print("\n9. Testing async operations...")
    import asyncio
    async def test_async():
        graph3 = EntityKnowledgeGraph()
        await graph3.add_entity_async("AsyncEntity", "Test", {"key": "value"})
        entity = await graph3.get_entity_async("AsyncEntity")
        assert entity is not None
        assert entity['name'] == "AsyncEntity"
        return True

    result = asyncio.run(test_async())
    assert result is True
    print("   PASS: Async operations work")

    print("\n" + "="*60)
    print("ALL EntityKnowledgeGraph TESTS PASSED!")
    print("="*60)

def test_knowledge_state():
    """Test KnowledgeState functionality."""
    print("\n" + "="*60)
    print("Testing KnowledgeState")
    print("="*60)

    # Test 1: Initialization
    print("\n1. Testing initialization...")
    state = KnowledgeState(query="What is AI?", correlation_id="test-state")
    stats = state.get_statistics()
    assert stats['query'] == "What is AI?"
    assert stats['triple_count'] == 0
    print("   PASS: State initialization")

    # Test 2: Add knowledge
    print("\n2. Testing add_knowledge...")
    triples = [
        ("AI", "is", "technology"),
        ("AI", "includes", "machine learning")
    ]
    result = state.add_knowledge(triples, source="test")
    assert result is True
    current = state.get_current_state()
    assert len(current['triples']) == 2
    print("   PASS: Knowledge added")

    # Test 3: Idempotent knowledge addition
    print("\n3. Testing idempotent knowledge addition...")
    state.add_knowledge([("AI", "is", "technology")])
    stats = state.get_statistics()
    assert stats['triple_count'] == 2  # No duplicate
    print("   PASS: Idempotent addition works")

    # Test 4: Add fact
    print("\n4. Testing add_fact...")
    result = state.add_fact("AI is a branch of computer science")
    assert result is True
    result = state.add_fact("AI is a branch of computer science")  # Duplicate
    assert result is False
    print("   PASS: Fact addition works")

    # Test 5: Add uncertainty
    print("\n5. Testing add_uncertainty...")
    result = state.add_uncertainty("AGI timeline is unknown")
    assert result is True
    print("   PASS: Uncertainty addition works")

    # Test 6: Snapshots
    print("\n6. Testing temporal snapshots...")
    from datetime import datetime, timezone
    timestamp1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    state.add_knowledge([("ML", "is", "subset of AI")], timestamp=timestamp1)

    history = state.get_history()
    assert len(history) >= 2
    print("   PASS: Temporal snapshots work")

    # Test 7: Search triples
    print("\n7. Testing search_triples...")
    results = state.search_triples(subject="AI")
    assert len(results) == 2
    results = state.search_triples(predicate="is")
    assert len(results) >= 2
    print("   PASS: Triple search works")

    # Test 8: Get state at time
    print("\n8. Testing get_state_at_time...")
    state_at_t = state.get_state_at_time(timestamp1)
    assert state_at_t is not None
    assert len(state_at_t['triples']) >= 2
    print("   PASS: Get state at time works")

    # Test 9: JSON serialization
    print("\n9. Testing JSON serialization...")
    json_str = state.to_json()
    data = json.loads(json_str)
    assert data['query'] == "What is AI?"
    assert len(data['triples']) >= 3
    print("   PASS: JSON serialization works")

    # Test 10: JSON deserialization
    print("\n10. Testing JSON deserialization...")
    state2 = KnowledgeState.from_json(json_str)
    assert state2.query == "What is AI?"
    assert state2._version == state._version
    print("   PASS: JSON deserialization works")

    # Test 11: Async operations
    print("\n11. Testing async operations...")
    async def test_async():
        state3 = KnowledgeState(query="Async test")
        await state3.add_knowledge_async([("Test", "is", "async")])
        current = await state3.get_current_state_async()
        assert len(current['triples']) == 1
        return True

    result = asyncio.run(test_async())
    assert result is True
    print("   PASS: Async operations work")

    print("\n" + "="*60)
    print("ALL KnowledgeState TESTS PASSED!")
    print("="*60)

def test_integration():
    """Test integration between graph and state."""
    print("\n" + "="*60)
    print("Testing Integration")
    print("="*60)

    # Test 1: Combined usage
    print("\n1. Testing combined graph and state usage...")
    graph = EntityKnowledgeGraph()
    state = KnowledgeState(query="Build knowledge graph")

    # Extract from triples to graph
    triples = [
        ("Python", "is", "language"),
        ("Python", "used_for", "data_science"),
        ("data_science", "uses", "machine_learning")
    ]

    state.add_knowledge(triples)

    # Add entities to graph
    graph.add_entity("Python", "Language", {"paradigm": "multi"})
    graph.add_entity("data_science", "Field", {"interdisciplinary": True})
    graph.add_entity("machine_learning", "Technology", {})

    # Add relationships
    graph.add_relationship("Python", "data_science", "USED_FOR")
    graph.add_relationship("data_science", "machine_learning", "USES")

    # Verify
    state_triples = state.search_triples(subject="Python")
    graph_rels = graph.get_relationships("Python")

    assert len(state_triples) == 2
    assert len(graph_rels) == 1
    print("   PASS: Combined usage works")

    # Test 2: Serialization roundtrip
    print("\n2. Testing serialization roundtrip...")
    graph_json = graph.to_json()
    state_json = state.to_json()

    graph2 = EntityKnowledgeGraph()
    graph2.from_json(graph_json)

    state2 = KnowledgeState.from_json(state_json)

    assert graph2.get_statistics()['entity_count'] == 3
    assert state2.get_statistics()['triple_count'] == 3
    print("   PASS: Roundtrip serialization works")

    print("\n" + "="*60)
    print("ALL Integration TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CORE GRAPH MODULES TEST SUITE")
    print("="*60)

    try:
        test_entity_knowledge_graph()
        test_knowledge_state()
        test_integration()

        print("\n" + "="*60)
        print("SUCCESS: ALL TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print("\n" + "="*60)
        print(f"FAILURE: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
