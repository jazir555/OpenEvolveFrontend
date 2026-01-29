"""
Simple test script for OpenEvolve Knowledge Engine

This script provides a basic test of the knowledge engine components
without requiring complex test frameworks.
"""

import sys
import os
from datetime import datetime

# Add the knowledge_engine directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_engine'))

# Import knowledge engine components
from knowledge_extractor import KnowledgeExtractor, KnowledgeArtifact
from knowledge_storage import KnowledgeStorage
from knowledge_retriever import KnowledgeRetriever
from integrated_engine import IntegratedKnowledgeEngine

def test_knowledge_extractor():
    """Test the KnowledgeExtractor component"""
    print("🧪 Testing KnowledgeExtractor...")
    
    # Create extractor
    extractor = KnowledgeExtractor()
    
    # Sample workflow data
    workflow_data = {
        'workflow_id': 'test_workflow_001',
        'timestamp': datetime.now().isoformat(),
        'execution_data': {
            'problem_type': 'decomposition',
            'complexity': 'high',
            'team_size': 5,
            'success': True,
            'execution_time': 3600
        },
        'solution_patterns': [
            {
                'pattern': 'hierarchical_task_analysis',
                'effectiveness': 0.95,
                'context': 'complex_decomposition'
            }
        ],
        'critique_patterns': [
            {
                'pattern': 'resource_allocation',
                'issue': 'suboptimal_distribution',
                'severity': 'medium'
            }
        ],
        'team_performance': {
            'efficiency': 0.87,
            'collaboration': 0.92,
            'adaptability': 0.85
        },
        'gauntlet_effectiveness': {
            'completion_rate': 0.90,
            'quality_score': 0.88,
            'iteration_count': 3
        }
    }
    
    # Test extraction
    artifacts = extractor.extract_from_workflow(workflow_data)
    
    print(f"✅ Extracted {len(artifacts)} knowledge artifacts")
    for i, artifact in enumerate(artifacts, 1):
        print(f"  {i}. {artifact.artifact_type}: {artifact.content[:50]}...")
    
    return True

def test_knowledge_storage():
    """Test the KnowledgeStorage component"""
    print("\n🧪 Testing KnowledgeStorage...")
    
    # Create storage
    storage = KnowledgeStorage()
    
    # Create sample artifact
    sample_artifact = {
        'type': 'solution_pattern',
        'source': 'test',
        'content': 'Test solution pattern for decomposition problems',
        'context': {'problem_type': 'decomposition'},
        'metadata': {'workflow_id': 'test_001'},
        'embeddings': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 96  # 768-dim
    }
    
    # Test storage and retrieval
    artifact_id = storage.store_knowledge_artifact(sample_artifact)
    print(f"✅ Stored artifact with ID: {artifact_id}")
    
    retrieved = storage.get_artifact_by_id(artifact_id)
    if retrieved and retrieved['content'] == sample_artifact['content']:
        print("✅ Successfully retrieved stored artifact")
    else:
        print("❌ Failed to retrieve artifact")
        return False
    
    # Test search
    search_results = storage.search_similar_artifacts(sample_artifact['embeddings'])
    print(f"✅ Found {len(search_results)} similar artifacts")
    
    # Test statistics
    stats = storage.get_statistics()
    print(f"✅ Knowledge base statistics: {stats['total_artifacts']} total artifacts")
    
    return True

def test_knowledge_retriever():
    """Test the KnowledgeRetriever component"""
    print("\n🧪 Testing KnowledgeRetriever...")
    
    # Create storage and retriever
    storage = KnowledgeStorage()
    retriever = KnowledgeRetriever(storage)
    
    # Store some test data
    for i in range(5):
        artifact = {
            'type': 'solution_pattern',
            'source': 'test',
            'content': f"Test knowledge artifact {i} for decomposition problems",
            'context': {'problem_type': 'decomposition', 'complexity': 'high'},
            'metadata': {'workflow_id': f'test_{i:03d}'},
            'embeddings': [0.1 + i*0.01, 0.2 + i*0.01, 0.3 + i*0.01, 0.4 + i*0.01,
                          0.5 + i*0.01, 0.6 + i*0.01, 0.7 + i*0.01, 0.8 + i*0.01] * 96
        }
        storage.store_knowledge_artifact(artifact)
    
    # Test search
    search_results = retriever.search_knowledge(
        query="decomposition",
        query_type="hybrid",
        limit=3
    )
    print(f"✅ Found {len(search_results)} search results")
    
    # Test recommendations
    context = {'problem_type': 'decomposition', 'complexity': 'high'}
    recommendations = retriever.get_recommendations(context, limit=2)
    print(f"✅ Got {len(recommendations)} recommendations")
    
    # Test quality metrics
    quality = retriever.get_knowledge_quality_metrics()
    print(f"✅ Overall quality score: {quality['overall_quality_score']:.2f}")
    
    return True

def test_integrated_engine():
    """Test the IntegratedKnowledgeEngine"""
    print("\n🧪 Testing IntegratedKnowledgeEngine...")
    
    # Create integrated engine
    engine = IntegratedKnowledgeEngine()
    
    # Sample workflow data
    workflow_data = {
        'workflow_id': 'integration_test_001',
        'timestamp': datetime.now().isoformat(),
        'execution_data': {
            'problem_type': 'decomposition',
            'complexity': 'high',
            'team_size': 3,
            'success': True,
            'execution_time': 1800
        },
        'solution_patterns': [
            {
                'pattern': 'modular_decomposition',
                'effectiveness': 0.92,
                'context': 'medium_complexity'
            }
        ]
    }
    
    # Test workflow processing
    processing_result = engine.process_workflow_data(workflow_data)
    print(f"✅ Processed workflow: {processing_result['status']}")
    print(f"✅ Extracted {processing_result['knowledge_extracted']} knowledge artifacts")
    
    # Test search
    search_results = engine.search_knowledge(
        query="decomposition",
        query_type="hybrid",
        limit=3
    )
    print(f"✅ Found {len(search_results)} search results")
    
    # Test recommendations
    context = {'problem_type': 'decomposition', 'complexity': 'high'}
    recommendations = engine.get_recommendations(context, limit=2)
    print(f"✅ Got {len(recommendations)} recommendations")
    
    # Test statistics
    stats = engine.get_knowledge_statistics()
    print(f"✅ Knowledge base contains {stats['total_artifacts']} artifacts")
    
    # Test quality metrics
    quality = engine.get_knowledge_quality()
    print(f"✅ Overall quality score: {quality['overall_quality_score']:.2f}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting OpenEvolve Knowledge Engine Tests\n")
    
    tests = [
        ("Knowledge Extractor", test_knowledge_extractor),
        ("Knowledge Storage", test_knowledge_storage),
        ("Knowledge Retriever", test_knowledge_retriever),
        ("Integrated Engine", test_integrated_engine)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"Running {test_name} Test")
            print(f"{'='*50}")
            
            if test_func():
                print(f"✅ {test_name} Test PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} Test FAILED")
                failed += 1
                
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"❌ {test_name} Test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Knowledge Engine implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)