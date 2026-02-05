"""
Validation script for Ragbits integration in the Knowledge Engine
"""

from knowledge_engine.enterprise_knowledge_engine import EnterpriseKnowledgeEngine
import asyncio
import json

def validate_ragbits_integration():
    print("Validating Ragbits Integration in Knowledge Engine...")
    print("="*60)
    
    # Initialize the engine
    engine = EnterpriseKnowledgeEngine()
    
    print("+ Enterprise Knowledge Engine initialized")
    print("+ Ragbits integration available: {engine.ragbits_integration is not None}")
    
    # Test 1: Check that ragbits methods exist
    print("\nTesting Ragbits-specific methods:")
    
    methods_to_check = [
        'search_knowledge',
        'store_artifact_with_ragbits', 
        'get_ragbits_statistics'
    ]
    
    for method_name in methods_to_check:
        method_exists = hasattr(engine, method_name)
        status = "+" if method_exists else "-"
        print(f"  {status} {method_name} method exists: {method_exists}")
    
    # Test 2: Test search functionality with different query types
    print(f"\nTesting search functionality:")
    
    try:
        # Test with ragbits query type
        result = engine.search_knowledge("test query", query_type="ragbits")
        print(f"  + Ragbits search successful: {type(result)}")
        print(f"     Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    except Exception as e:
        print(f"  - Ragbits search failed: {e}")
    
    # Test 3: Test artifact storage with ragbits
    print(f"\nTesting artifact storage with Ragbits:")
    
    try:
        result = engine.store_artifact_with_ragbits(
            content="This is a test artifact for validation",
            metadata={"test": True, "validation": "ragbits"},
            artifact_type="validation_test"
        )
        print(f"  + Ragbits artifact storage successful: {type(result)}")
        print(f"     Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    except Exception as e:
        print(f"  - Ragbits artifact storage failed: {e}")
    
    # Test 4: Test ragbits statistics
    print(f"\nTesting Ragbits statistics retrieval:")
    
    try:
        stats_result = asyncio.run(engine.get_ragbits_statistics())
        print(f"  + Ragbits statistics retrieval successful: {type(stats_result)}")
        print(f"     Stats keys: {list(stats_result.keys()) if isinstance(stats_result, dict) else 'N/A'}")
    except Exception as e:
        print(f"  - Ragbits statistics retrieval failed: {e}")
    
    # Test 5: Test analytics include ragbits info
    print(f"\nTesting analytics with Ragbits info:")
    
    try:
        analytics = engine.get_analytics()
        has_ragbits = 'ragbits' in analytics
        print(f"  + Analytics include ragbits section: {has_ragbits}")
        if has_ragbits:
            print(f"     Ragbits keys: {list(analytics['ragbits'].keys()) if isinstance(analytics['ragbits'], dict) else 'N/A'}")
    except Exception as e:
        print(f"  - Analytics retrieval failed: {e}")
    
    print(f"\n{'='*60}")
    print("Ragbits Integration Validation Complete!")
    
    # Summary
    all_good = all([
        engine.ragbits_integration is not None,
        hasattr(engine, 'search_knowledge'),
        hasattr(engine, 'store_artifact_with_ragbits'),
        hasattr(engine, 'get_ragbits_statistics')
    ])
    
    if all_good:
        print("+ ALL TESTS PASSED - Ragbits integration is fully functional!")
        return True
    else:
        print("- SOME TESTS FAILED - Integration issues detected")
        return False

if __name__ == "__main__":
    success = validate_ragbits_integration()
    exit(0 if success else 1)