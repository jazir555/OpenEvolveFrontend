"""
Test script to identify and verify bugs in Phase 3 implementation
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all imports work correctly"""
    print("🔍 Testing imports...")
    
    # Test core imports
    from knowledge_extractor import KnowledgeArtifact, KnowledgeExtractor
    from knowledge_storage import KnowledgeStorage
    from knowledge_retriever import KnowledgeRetriever
    from integrated_engine import IntegratedKnowledgeEngine
    
    # Test Phase 2 imports
    from embedding_generator import EmbeddingGenerator
    from enhanced_storage import EnhancedKnowledgeStorage
    from enhanced_retriever import EnhancedKnowledgeRetriever
    
    # Test Phase 3 imports
    from real_database_integration import RealDatabaseIntegrator
    from production_engine import ProductionKnowledgeEngine
    
    print("[OK] All imports successful")
    return True

def test_database_integrator():
    """Test database integrator functionality"""
    print("\n🔍 Testing database integrator...")
    
    from real_database_integration import RealDatabaseIntegrator
    
    integrator = RealDatabaseIntegrator()
    
    # Test availability
    availability = integrator.get_availability()
    print(f"Database availability: {availability}")
    assert isinstance(availability, dict)
    
    # Test health status
    health = integrator.get_health_status()
    print(f"Health status: {health['overall_status']}")
    assert 'overall_status' in health
    
    # Test production readiness
    production_ready = integrator.is_production_ready()
    print(f"Production ready: {production_ready}")
    
    print("[OK] Database integrator test completed")
    return True

def test_production_engine():
    """Test production engine functionality"""
    print("\n🔍 Testing production engine...")
    
    from production_engine import ProductionKnowledgeEngine
    
    engine = ProductionKnowledgeEngine()
    
    # Test system status
    status = engine.get_system_status()
    print(f"System status: {status['status']}")
    print(f"Production ready: {status['production_ready']}")
    assert 'status' in status
    
    # Test health report
    health_report = engine.get_production_health_report()
    print(f"Health report status: {health_report['overall_status']}")
    assert 'overall_status' in health_report
    
    print("[OK] Production engine test completed")
    return True

def test_workflow_processing():
    """Test workflow processing with error handling"""
    print("\n🔍 Testing workflow processing...")
    
    from production_engine import ProductionKnowledgeEngine
    
    engine = ProductionKnowledgeEngine()
    
    # Test with standard workflow data
    workflow_data = {
        'workflow_id': 'test_workflow_001',
        'domain': 'ai',
        'solutions': [{'id': 'sol_1', 'approach': 'test'}],
        'timestamp': datetime.now().isoformat()
    }
    
    result = engine.process_workflow_production(workflow_data)
    print(f"Workflow processing result: {result['status']}")
    assert result['status'] in ['processed', 'partial_success']
    
    # Test with minimal workflow data
    minimal_workflow = {
        'workflow_id': 'minimal_test'
    }
    
    minimal_result = engine.process_workflow_production(minimal_workflow)
    print(f"Minimal workflow result: {minimal_result['status']}")
    assert 'status' in minimal_result
    
    print("[OK] Workflow processing test completed")
    return True

def test_search_functionality():
    """Test search functionality"""
    print("\n🔍 Testing search functionality...")
    
    from production_engine import ProductionKnowledgeEngine
    
    engine = ProductionKnowledgeEngine()
    
    # Test basic search
    search_result = engine.production_search("test query")
    print(f"Search result: {search_result['status']}")
    assert 'status' in search_result
    
    # Test with different query types
    hybrid_result = engine.production_search("test", query_type="hybrid")
    vector_result = engine.production_search("test", query_type="vector")
    keyword_result = engine.production_search("test", query_type="keyword")
    
    print(f"Hybrid: {hybrid_result['status']}, Vector: {vector_result['status']}, Keyword: {keyword_result['status']}")
    assert hybrid_result['status'] == 'success'
    
    print("[OK] Search functionality test completed")
    return True

def test_recommendations():
    """Test recommendation functionality"""
    print("\n🔍 Testing recommendations...")
    
    from production_engine import ProductionKnowledgeEngine
    
    engine = ProductionKnowledgeEngine()
    
    # Test basic recommendations
    context = {'problem_type': 'decomposition'}
    recommendations = engine.get_production_recommendations(context)
    print(f"Recommendations result: {recommendations['status']}")
    assert 'status' in recommendations
    
    # Test with user profile
    user_profile = {'expertise_level': 'intermediate'}
    personalized_recs = engine.get_production_recommendations(context, user_profile)
    print(f"Personalized recommendations: {personalized_recs['status']}")
    assert personalized_recs['status'] == 'success'
    
    print("[OK] Recommendations test completed")
    return True

def test_analytics():
    """Test analytics functionality"""
    print("\n🔍 Testing analytics...")
    
    from production_engine import ProductionKnowledgeEngine
    
    engine = ProductionKnowledgeEngine()
    
    # Test comprehensive analytics
    analytics = engine.get_comprehensive_analytics()
    print(f"Analytics generated: {len(analytics) > 0}")
    assert isinstance(analytics, dict)
    
    # Test health report
    health_report = engine.get_production_health_report()
    print(f"Health report generated: {len(health_report) > 0}")
    assert 'overall_status' in health_report
    
    print("[OK] Analytics test completed")
    return True

def test_error_handling():
    """Test error handling and robustness"""
    print("\n🔍 Testing error handling...")
    
    from production_engine import ProductionKnowledgeEngine
    
    engine = ProductionKnowledgeEngine()
    
    # Test with invalid input
    try:
        result = engine.process_workflow_production(None)
        print(f"Invalid input result: {result['status']}")
        assert result['status'] == 'error'
    except Exception as e:
        print(f"Caught expected error or handled gracefully: {e}")
    
    print("[OK] Error handling test completed")
    return True

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_imports()
    test_database_integrator()
    test_production_engine()
    test_workflow_processing()
    test_search_functionality()
    test_recommendations()
    test_analytics()
    test_error_handling()
