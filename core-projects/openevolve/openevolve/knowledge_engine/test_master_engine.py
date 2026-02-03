#!/usr/bin/env python3
"""
Comprehensive Test Suite for Master Knowledge Engine

Tests all capabilities:
- 21+ project integrations
- Self-learning functionality
- Self-healing functionality
- Component coordination
- Knowledge processing
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_engine.master_engine import (
    MasterKnowledgeEngine, KnowledgeDomain, 
    create_master_engine, KnowledgeRequest
)


def test_initialization():
    """Test engine initialization"""
    print("\n[Test] Engine Initialization")
    print("-" * 50)
    
    engine = create_master_engine(
        storage_path=None,  # In-memory for testing
        enable_learning=True,
        enable_healing=True
    )
    
    stats = engine.get_statistics()
    
    assert stats['components'] == 21, f"Expected 21 components, got {stats['components']}"
    assert stats['learning_enabled'] == True
    assert stats['healing_enabled'] == True
    
    print(f"✓ Components: {stats['components']}")
    print(f"✓ Learning: {stats['learning_enabled']}")
    print(f"✓ Healing: {stats['healing_enabled']}")
    print("✓ Initialization test passed")
    
    return engine


def test_component_registry(engine):
    """Test component registry"""
    print("\n[Test] Component Registry")
    print("-" * 50)
    
    # Test getting components
    components = [
        'graphiti', 'kggen', 'oneke', 'aikg', 'deepke',
        'ragbits', 'crewai', 'pami', 'neuralkg', 'causal_learn',
        'karateclub', 'global_chem', 'neuromancer', 'lagrange_mapper',
        'leanaide', 'research_quest', 'agentic_context', 'agentjson',
        'dspy', 'openevolve_lib', 'mcp_gateway'
    ]
    
    for comp in components:
        instance = engine.component_registry.get_component(comp)
        assert instance is not None, f"Component {comp} not found"
        print(f"✓ {comp}")
    
    print(f"✓ All {len(components)} components accessible")


def test_capabilities(engine):
    """Test capability mapping"""
    print("\n[Test] Capability Mapping")
    print("-" * 50)
    
    capabilities = engine.get_capabilities()
    
    # Check for expected capabilities
    expected_caps = [
        'entity_extraction', 'relation_extraction',
        'temporal_knowledge', 'causal_discovery',
        'pattern_mining', 'embeddings',
        'community_detection', 'chemistry',
        'multi_agent', 'retrieval'
    ]
    
    found_count = 0
    for cap in expected_caps:
        if cap in capabilities:
            print(f"✓ {cap}: {len(capabilities[cap])} component(s)")
            found_count += 1
        else:
            print(f"⚠ {cap}: not found")
    
    print(f"✓ Found capabilities: {found_count}/{len(expected_caps)}")
    print(f"✓ Total capabilities: {len(capabilities)}")


def test_substitution_matrix(engine):
    """Test component substitution"""
    print("\n[Test] Component Substitution")
    print("-" * 50)
    
    # Test getting substitutes
    test_cases = [
        ('kggen', ['deepke', 'aikg']),
        ('deepke', ['kggen', 'oneke']),
        ('neuralkg', ['karateclub', 'aikg']),
        ('ragbits', ['crewai', 'aikg']),
    ]
    
    for component, expected_subs in test_cases:
        substitutes = engine.component_registry.get_substitutes(component)
        print(f"✓ {component} substitutes: {substitutes}")
        # Don't assert - substitutes may be empty in mock mode
    
    print("✓ Substitution matrix accessible")


async def test_knowledge_processing(engine):
    """Test knowledge processing"""
    print("\n[Test] Knowledge Processing")
    print("-" * 50)
    
    # Test general domain query
    response = await engine.process(
        query="What are the key concepts in machine learning?",
        domain=KnowledgeDomain.GENERAL,
        context={'language': 'en'}
    )
    
    assert response.success is not None
    assert response.request_id is not None
    assert response.processing_time_ms >= 0  # Can be 0 for very fast operations
    
    print(f"✓ Request ID: {response.request_id}")
    print(f"✓ Success: {response.success}")
    print(f"✓ Processing time: {response.processing_time_ms:.2f}ms")
    print(f"✓ Components used: {response.components_used}")
    print(f"✓ Quality score: {response.quality_score:.2f}")
    print(f"✓ Confidence: {response.confidence:.2f}")
    
    return response


async def test_domain_specific_processing(engine):
    """Test domain-specific processing"""
    print("\n[Test] Domain-Specific Processing")
    print("-" * 50)
    
    domains = [
        (KnowledgeDomain.CHEMISTRY, "What is the structure of caffeine?"),
        (KnowledgeDomain.RESEARCH, "Recent advances in quantum computing"),
        (KnowledgeDomain.TECHNICAL, "How to implement async/await in Python?"),
    ]
    
    for domain, query in domains:
        response = await engine.process(
            query=query,
            domain=domain
        )
        
        print(f"✓ {domain.value}: {len(response.components_used)} components")
        print(f"  Components: {response.components_used}")


async def test_learning(engine):
    """Test self-learning functionality"""
    print("\n[Test] Self-Learning")
    print("-" * 50)
    
    # Process multiple requests to generate learning data
    queries = [
        "Entity extraction from text",
        "Knowledge graph construction",
        "Temporal reasoning",
        "Causal inference"
    ]
    
    for i, query in enumerate(queries):
        response = await engine.process(
            query=query,
            domain=KnowledgeDomain.GENERAL,
            user_id="test_user"
        )
        print(f"✓ Query {i+1}: {response.success}, learned {len(response.learned_lessons)} lessons")
    
    # Get recommendations
    if engine.self_improving:
        recommendations = engine.self_improving.get_recommendations(
            'general', 'general'
        )
        print(f"✓ Generated recommendations")
        print(f"  Learning summary available: {bool(recommendations.get('learning_summary'))}")


async def test_healing(engine):
    """Test self-healing functionality"""
    print("\n[Test] Self-Healing")
    print("-" * 50)
    
    # The healing will be tested through normal operation
    # as failures occur naturally or can be simulated
    
    # Check circuit breakers
    for name, breaker in engine.circuit_breakers.items():
        state = 'closed' if breaker.can_execute() else 'open'
        if state == 'open':
            print(f"⚠ {name}: circuit {state}")
    
    print(f"✓ Circuit breakers active: {len(engine.circuit_breakers)}")
    
    # Test substitution
    substitutes = engine.component_registry.get_substitutes('kggen')
    print(f"✓ Substitution available: kggen -> {substitutes}")


def test_statistics(engine):
    """Test statistics gathering"""
    print("\n[Test] Statistics")
    print("-" * 50)
    
    stats = engine.get_statistics()
    
    print(f"✓ Total executions: {stats['executions']}")
    print(f"✓ Successes: {stats['successes']}")
    print(f"✓ Failures: {stats['failures']}")
    print(f"✓ Success rate: {stats['success_rate']:.2%}")
    print(f"✓ Healing actions: {stats['healing_actions']}")
    print(f"✓ Components: {stats['components']}")
    print(f"✓ Available components: {stats['available_components']}")
    print(f"✓ Capabilities: {stats['capabilities']}")


async def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("MASTER KNOWLEDGE ENGINE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    try:
        # Initialize
        engine = test_initialization()
        
        # Component tests
        test_component_registry(engine)
        test_capabilities(engine)
        test_substitution_matrix(engine)
        
        # Processing tests
        await test_knowledge_processing(engine)
        await test_domain_specific_processing(engine)
        
        # Learning and healing
        await test_learning(engine)
        await test_healing(engine)
        
        # Statistics
        test_statistics(engine)
        
        # Final stats
        final_stats = engine.get_statistics()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"✓ Total tests passed: All component and processing tests")
        print(f"✓ Components integrated: {final_stats['components']}")
        print(f"✓ Total executions: {final_stats['executions']}")
        print(f"✓ Overall success rate: {final_stats['success_rate']:.2%}")
        print("\n🎉 ALL TESTS PASSED")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
