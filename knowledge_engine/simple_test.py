"""
Simple test script for OpenEvolve Knowledge Engine

This script provides a basic test of the knowledge engine components
without requiring complex test frameworks.
"""

import sys
import os
from datetime import datetime

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import knowledge engine components
from knowledge_engine.knowledge_extractor import KnowledgeExtractor, KnowledgeArtifact
from knowledge_engine.knowledge_storage import KnowledgeStorage
from knowledge_engine.knowledge_retriever import KnowledgeRetriever
from knowledge_engine.integrated_engine import IntegratedKnowledgeEngine

def test_knowledge_extractor():
    """Test the KnowledgeExtractor component"""
    print("🧪 Testing KnowledgeExtractor...")
    
    # Create extractor
    extractor = KnowledgeExtractor()
    
    # Sample workflow data matching actual implementation requirements
    workflow_data = {
        'workflow_id': 'test_workflow_001',
        'domain': 'software_engineering',
        'success': True,
        'solutions': [
            {
                'id': 'sol_1',
                'approach': 'hierarchical_task_analysis',
                'success_rate': 0.95,
                'complexity': 7,
                'domain': 'ai',
                'problem_type': 'decomposition'
            }
        ],
        'critiques': [
            {
                'id': 'crit_1',
                'pattern': 'resource_allocation',
                'issue': 'suboptimal_distribution',
                'severity': 'medium',
                'content': 'Resource allocation issue'
            }
        ],
        'teams': [
            {
                'id': 'team_1',
                'name': 'Alpha Team',
                'role': 'solver',
                'success_rate': 0.87,
                'avg_response_time': 1.2,
                'completion_rate': 0.92,
                'quality_score': 0.85
            }
        ],
        'gauntlets': [
            {
                'id': 'gaunt_1',
                'name': 'Standard Gauntlet',
                'detection_rate': 0.90,
                'true_positive_rate': 0.88,
                'false_positive_rate': 0.05,
                'average_score': 0.88
            }
        ],
        'timestamp': datetime.now().isoformat()
    }
    
    # Test extraction
    artifacts = extractor.extract_from_workflow(workflow_data)
    
    print(f"[OK] Extracted {len(artifacts)} knowledge artifacts")
    for i, artifact in enumerate(artifacts, 1):
        content_summary = str(artifact.content)[:50]
        print(f"  {i}. {artifact.artifact_type}: {content_summary}...")
    
    return len(artifacts) >= 4

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
    print(f"[OK] Stored artifact with ID: {artifact_id}")
    
    retrieved = storage.get_artifact_by_id(artifact_id)
    if retrieved and retrieved['content'] == sample_artifact['content']:
        print("[OK] Successfully retrieved stored artifact")
    else:
        print("[FAIL] Failed to retrieve artifact")
        return False
    
    # Test search
    search_results = storage.search_similar_artifacts(sample_artifact['embeddings'])
    print(f"[OK] Found {len(search_results)} similar artifacts")
    
    # Test statistics
    stats = storage.get_statistics()
    print(f"[OK] Knowledge base statistics: {stats['total_artifacts']} total artifacts")
    
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
            'problem_type': 'decomposition',
            'context': {'complexity': 'high'},
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
    print(f"[OK] Found {len(search_results)} search results")
    
    # Test recommendations
    context = {'problem_type': 'decomposition'}
    recommendations = retriever.get_recommendations(context, limit=2)
    print(f"[OK] Got {len(recommendations)} recommendations")
    
    # Test quality metrics
    quality = retriever.get_knowledge_quality_metrics()
    print(f"[OK] Overall quality score: {quality['overall_quality_score']:.2f}")
    
    return True

def test_integrated_engine():
    """Test the IntegratedKnowledgeEngine"""
    print("\n🧪 Testing IntegratedKnowledgeEngine...")
    
    # Create integrated engine
    engine = IntegratedKnowledgeEngine()
    
    # Sample workflow data matching actual implementation requirements
    workflow_data = {
        'workflow_id': 'integration_test_001',
        'timestamp': datetime.now().isoformat(),
        'domain': 'general',
        'solutions': [
            {
                'id': 'sol_1',
                'approach': 'modular_decomposition',
                'success_rate': 0.92,
                'complexity': 5,
                'domain': 'general',
                'problem_type': 'decomposition'
            }
        ],
        'critiques': [],
        'teams': [],
        'gauntlets': []
    }
    
    # Test workflow processing
    processing_result = engine.process_workflow_data(workflow_data)
    print(f"[OK] Processed workflow: {processing_result['status']}")
    print(f"[OK] Extracted {processing_result['knowledge_extracted']} knowledge artifacts")
    
    # Test search
    search_results = engine.search_knowledge(
        query="decomposition",
        query_type="hybrid",
        limit=3
    )
    print(f"[OK] Found {len(search_results)} search results")
    
    # Test recommendations
    context = {'problem_type': 'decomposition'}
    recommendations = engine.get_recommendations(context, limit=2)
    print(f"[OK] Got {len(recommendations)} recommendations")
    
    # Test statistics
    stats = engine.get_knowledge_statistics()
    print(f"[OK] Knowledge base contains {stats['total_artifacts']} artifacts")
    
    # Test quality metrics
    quality = engine.get_knowledge_quality()
    print(f"[OK] Overall quality score: {quality['overall_quality_score']:.2f}")
    
    return True

def test_backup_restore():
    """Test backup and restore functionality"""
    print("\n🧪 Testing Backup/Restore...")
    
    import tempfile
    
    storage = KnowledgeStorage()
    
    # Store some data
    storage.store_knowledge_artifact({
        'type': 'solution_pattern',
        'content': 'Data to backup'
    })
    
    initial_count = storage.get_statistics()['total_artifacts']
    
    # Create backup
    fd, backup_path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    
    try:
        storage.backup_knowledge_base(backup_path)
        print(f"[OK] Created backup at {backup_path}")
        
        # New storage instance (to simulate clean state)
        new_storage = KnowledgeStorage()
        new_storage.restore_knowledge_base(backup_path)
        
        restored_count = new_storage.get_statistics()['total_artifacts']
        print(f"[OK] Restored {restored_count} artifacts")
        
        return initial_count == restored_count
    finally:
        if os.path.exists(backup_path):
            os.remove(backup_path)

if __name__ == "__main__":
    test_knowledge_extractor()
    test_knowledge_storage()
    test_knowledge_retriever()
    test_integrated_engine()
    test_backup_restore()
