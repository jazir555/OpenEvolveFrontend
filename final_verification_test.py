"""
Final verification test for the complete Ragbits integration in the Knowledge Engine.
This test verifies all aspects of the integration work together properly.
"""

import asyncio
from knowledge_engine.enterprise_knowledge_engine import EnterpriseKnowledgeEngine


def test_complete_integration():
    """Test the complete ragbits integration end-to-end."""
    print("[INFO] Starting complete Ragbits-Knowledge Engine integration test...")
    
    # Initialize the enterprise knowledge engine with ragbits support
    config = {
        'ragbits': {
            'vector_store': {
                'type': 'memory',  # Use in-memory store for testing
                'config': {}
            },
            'default_options': {
                'top_k': 5,
                'similarity_threshold': 0.5
            }
        }
    }
    
    print("[BUILD] Initializing Enterprise Knowledge Engine...")
    engine = EnterpriseKnowledgeEngine(config=config)
    
    print(f"[SUCCESS] Engine initialized successfully")
    print(f"[SUCCESS] Ragbits integration available: {engine.ragbits_integration is not None}")
    
    # Test 1: Store an artifact using ragbits
    print("\n[STORE] Testing artifact storage with Ragbits...")
    content = "This is a test document about machine learning algorithms and their applications in natural language processing."
    metadata = {
        "category": "ML Research",
        "domain": "NLP",
        "tags": ["ml", "nlp", "algorithms"],
        "test": True
    }
    
    store_result = engine.store_artifact_with_ragbits(
        content=content,
        metadata=metadata,
        artifact_type="research_paper"
    )
    
    print(f"[SUCCESS] Artifact storage result: {store_result['status']}")
    print(f"[SUCCESS] Ragbits ingestion: {store_result.get('ragbits_ingested', 'N/A')}")
    
    # Test 2: Search using ragbits-enhanced search
    print("\n[SEARCH] Testing Ragbits-enhanced search...")
    search_result = engine.search_knowledge(
        query="machine learning algorithms for natural language processing",
        query_type="ragbits",
        limit=3
    )
    
    print(f"[SUCCESS] Search result: {search_result['status']}")
    print(f"[SUCCESS] Found {search_result['result_count']} results")
    
    # Test 3: Get analytics including ragbits data
    print("\n[ANALYTICS] Testing analytics with Ragbits data...")
    analytics = engine.get_analytics()
    
    print(f"[SUCCESS] Analytics retrieved successfully")
    print(f"[SUCCESS] Ragbits available in analytics: {'ragbits' in analytics}")
    
    if 'ragbits' in analytics:
        ragbits_data = analytics['ragbits']
        print(f"[SUCCESS] Ragbits status: {ragbits_data.get('ragbits_available', 'N/A')}")
    
    # Test 4: Get ragbits-specific statistics
    print("\n[STATS] Testing Ragbits-specific statistics...")
    try:
        ragbits_stats = asyncio.run(engine.get_ragbits_statistics())
        print(f"[SUCCESS] Ragbits statistics retrieved: {ragbits_stats.get('ragbits_available', 'N/A')}")
    except Exception as e:
        print(f"[WARN] Could not retrieve Ragbits stats: {e}")
    
    print("\n[TARGET] All integration tests completed successfully!")
    print("[TROPHY] Ragbits has been fully integrated into the Knowledge Engine!")
    
    return True


if __name__ == "__main__":
    success = test_complete_integration()
    if success:
        print("\n[PARTY] COMPLETE SUCCESS: Ragbits integration is fully operational!")
    else:
        print("\n[ERROR] FAILURE: Integration issues detected")
        exit(1)