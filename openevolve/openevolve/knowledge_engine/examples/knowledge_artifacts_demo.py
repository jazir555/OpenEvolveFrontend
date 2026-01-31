#!/usr/bin/env python3
"""
Knowledge Artifacts Demo

Demonstrates the complete knowledge extraction, storage, and retrieval pipeline.

Usage:
    python knowledge_artifacts_demo.py
"""

import json
import logging
from datetime import datetime, timezone

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_knowledge_artifacts():
    """Demonstrate complete knowledge artifacts pipeline"""

    print("\n" + "="*80)
    print("KNOWLEDGE ARTIFACTS DEMO")
    print("="*80 + "\n")

    # Import components
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from knowledge_engine.knowledge_extractor import KnowledgeExtractor
    from knowledge_engine.knowledge_storage import KnowledgeStorage
    from knowledge_engine.knowledge_retriever import KnowledgeRetriever

    # Sample workflow data
    workflow_data = {
        'workflow_id': 'demo_workflow_001',
        'domain': 'mathematical_optimization',
        'complexity': 'high',
        'execution_time': 1800,
        'success': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'solutions': [
            {
                'id': 'sol_001',
                'problem_type': 'optimization',
                'domain': 'algebra',
                'approach': 'hierarchical gradient descent with adaptive learning rate',
                'implementation': 'vectorized implementation with parallel processing',
                'success_rate': 0.95,
                'complexity': 8,
                'code': 'def optimize(function, initial_guess):\n    # Hierarchical optimization\n    pass',
                'documentation': 'Uses hierarchical decomposition for complex optimization',
                'performance': {
                    'convergence_rate': 0.92,
                    'iterations': 150,
                    'execution_time': 2.8
                }
            },
            {
                'id': 'sol_002',
                'problem_type': 'constraint_satisfaction',
                'domain': 'algebra',
                'approach': 'divide and conquer with constraint propagation',
                'implementation': 'recursive constraint partitioning',
                'success_rate': 0.88,
                'complexity': 7,
                'code': 'def solve_constraints(constraints):\n    # Divide and conquer\n    pass',
                'documentation': 'Efficient constraint satisfaction using decomposition',
                'performance': {
                    'satisfaction_rate': 0.85,
                    'backtracking_steps': 42,
                    'execution_time': 1.2
                }
            }
        ],
        'critiques': [
            {
                'id': 'crit_001',
                'issue_type': 'resource allocation inefficiency',
                'root_cause': 'suboptimal workload distribution across processing units',
                'prevention_strategy': 'implement dynamic resource allocation with load balancing',
                'severity': 'high',
                'affected_components': ['gradient_computation', 'constraint_propagation']
            }
        ],
        'teams': [
            {
                'name': 'optimization_team',
                'role': 'Blue',
                'domain': 'mathematical_optimization',
                'specialization': 'nonlinear_problems',
                'success_rate': 0.92,
                'avg_response_time': 1.5,
                'completion_rate': 0.94,
                'quality_score': 0.88,
                'performance_trends': [0.85, 0.88, 0.90, 0.92, 0.94]
            }
        ],
        'gauntlets': [
            {
                'name': 'quality_gauntlet',
                'type': 'Gold',
                'domain': 'validation',
                'problem_type': 'solution_quality',
                'detection_rate': 0.88,
                'false_positive_rate': 0.05,
                'true_positive_rate': 0.85,
                'average_score': 0.87,
                'performance_trends': [0.82, 0.84, 0.86, 0.88, 0.89]
            }
        ]
    }

    # Initialize components
    print("1. Initializing Knowledge Components...")
    extractor = KnowledgeExtractor({
        'quality_thresholds': {
            'high': 0.85,
            'medium': 0.65,
            'low': 0.40
        }
    })
    storage = KnowledgeStorage()
    retriever = KnowledgeRetriever(storage=storage)
    print("   [OK] Components initialized\n")

    # Extract knowledge
    print("2. Extracting Knowledge from Workflow...")
    print(f"   Workflow ID: {workflow_data['workflow_id']}")
    print(f"   Domain: {workflow_data['domain']}")
    print(f"   Solutions: {len(workflow_data['solutions'])}")

    artifacts = extractor.extract_from_workflow(workflow_data)

    print(f"   [OK] Extracted {len(artifacts)} artifacts\n")

    # Show extracted artifacts
    print("3. Extracted Artifacts Overview:")
    artifact_types = {}
    for artifact in artifacts:
        artifact_types[artifact.artifact_type] = artifact_types.get(artifact.artifact_type, 0) + 1

    for artifact_type, count in artifact_types.items():
        print(f"   - {artifact_type}: {count}")
    print()

    # Show quality scores
    print("4. Artifact Quality Scores:")
    for i, artifact in enumerate(artifacts[:5], 1):
        quality = artifact.calculate_quality_score()
        category = 'HIGH' if quality >= 0.85 else 'MEDIUM' if quality >= 0.65 else 'LOW'
        print(f"   {i}. {artifact.artifact_type}: {quality:.2f} ({category})")
    print()

    # Store artifacts
    print("5. Storing Artifacts in Knowledge Base...")
    stored_ids = []
    for artifact in artifacts:
        artifact_dict = artifact.to_dict()
        artifact_dict['type'] = artifact.artifact_type
        artifact_dict['source'] = artifact.source_workflow_id
        artifact_dict['content'] = json.dumps(artifact.content)

        artifact_id = storage.store_knowledge_artifact(artifact_dict)
        stored_ids.append(artifact_id)

    print(f"   [OK] Stored {len(stored_ids)} artifacts\n")

    # Get statistics
    print("6. Knowledge Base Statistics:")
    stats = storage.get_statistics()
    print(f"   Total artifacts: {stats['total_artifacts']}")
    print(f"   Artifact types: {json.dumps(stats['artifact_types'], indent=6)}")
    print(f"   Storage size: {stats['storage_size']} bytes")
    print()

    # Search knowledge
    print("7. Searching Knowledge Base...")
    search_results = retriever.search_knowledge(
        query='optimization',
        query_type='hybrid',
        limit=5
    )
    print(f"   Found {len(search_results)} results for 'optimization'\n")

    # Get recommendations
    print("8. Getting Context-Aware Recommendations...")
    recommendations = retriever.get_recommendations(
        context={
            'problem_type': 'optimization',
            'complexity': 'high'
        },
        recommendation_type='solution_pattern',
        limit=3
    )
    print(f"   Got {len(recommendations)} recommendations\n")

    # Show extraction stats
    print("9. Extraction Statistics:")
    extraction_stats = extractor.get_extraction_stats()
    print(f"   Total extractions: {extraction_stats['total_extractions']}")
    print(f"   Success rate: {extraction_stats['success_rate']:.2%}")
    print(f"   Average time: {extraction_stats['average_extraction_time']:.3f}s")
    print(f"   Quality distribution: {json.dumps(extraction_stats['quality_distribution'], indent=6)}")
    print()

    # Get quality metrics
    print("10. Knowledge Base Quality Metrics:")
    quality_metrics = retriever.get_knowledge_quality_metrics()
    quality = quality_metrics['quality_metrics']
    print(f"    Completeness: {quality['completeness']:.2f}")
    print(f"    Consistency: {quality['consistency']:.2f}")
    print(f"    Relevance: {quality['relevance']:.2f}")
    print(f"    Timeliness: {quality['timeliness']:.2f}")
    print(f"    Diversity: {quality['diversity']:.2f}")
    print(f"    Overall Score: {quality_metrics['overall_quality_score']:.2f}\n")

    # Show pattern recognition
    print("11. Pattern Recognition Results:")
    solution_artifacts = [a for a in artifacts if a.artifact_type == 'solution_pattern']
    for solution in solution_artifacts:
        pattern_info = solution.metadata.get('pattern_recognition', {})
        if pattern_info:
            print(f"   Solution: {solution.content.get('solution_id', 'unknown')}")
            print(f"   Pattern Type: {pattern_info.get('pattern_type', 'generic')}")
            print(f"   Match Score: {pattern_info.get('match_score', 0):.2f}")
            print(f"   Confidence: {pattern_info.get('confidence', 0):.2f}")
            print()

    # Summary
    print("="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print(f"[OK] Extracted {len(artifacts)} knowledge artifacts")
    print(f"[OK] Stored {len(stored_ids)} artifacts in knowledge base")
    print(f"[OK] Performed {len(search_results)} searches")
    print(f"[OK] Generated {len(recommendations)} recommendations")
    print(f"[OK] Overall quality score: {quality_metrics['overall_quality_score']:.2f}")
    print("\nAll components working correctly! [OK]\n")

    return artifacts, stored_ids, search_results


def demo_advanced_features():
    """Demonstrate advanced features"""

    print("\n" + "="*80)
    print("ADVANCED FEATURES DEMO")
    print("="*80 + "\n")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from knowledge_engine.knowledge_extractor import KnowledgeExtractor
    from knowledge_engine.knowledge_storage import KnowledgeStorage
    from knowledge_engine.knowledge_retriever import KnowledgeRetriever

    extractor = KnowledgeExtractor()
    storage = KnowledgeStorage()
    retriever = KnowledgeRetriever(storage=storage)

    # Create large workflow for performance testing
    print("1. Performance Test - Large Workflow Processing...")
    large_workflow = {
        'workflow_id': 'perf_test_001',
        'domain': 'performance_test',
        'complexity': 'high',
        'execution_time': 5000,
        'success': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'solutions': [
            {
                'id': f'sol_{i:03d}',
                'problem_type': 'optimization',
                'domain': 'test',
                'approach': f'Approach {i}',
                'implementation': f'Implementation {i}',
                'success_rate': 0.8 + (i % 20) * 0.01,
                'complexity': 5 + (i % 5),
                'code': f'def test_{i}(): pass',
                'documentation': f'Documentation {i}',
                'performance': {'iterations': i * 10}
            }
            for i in range(50)
        ]
    }

    import time
    start = time.time()
    artifacts = extractor.extract_from_workflow(large_workflow)
    elapsed = time.time() - start

    print(f"   Processed {len(artifacts)} artifacts from 50 solutions")
    print(f"   Time: {elapsed:.3f}s")
    rate = len(artifacts)/elapsed if elapsed > 0 else 0
    print(f"   Rate: {rate:.1f} artifacts/second\n")

    # Trend analysis
    print("2. Knowledge Trend Analysis...")
    trends = retriever.get_knowledge_trends(time_range='30d')
    print(f"   Trend: {trends['trend_analysis']['trend']}")
    print(f"   Change: {trends['trend_analysis']['change_percentage']:.1f}%")
    print(f"   Average daily: {trends['trend_analysis']['average_daily']:.1f}\n")

    # Advanced search
    print("3. Advanced Search with Faceting...")
    results = retriever.advanced_search({
        'query': 'test',
        'filters': {},
        'sort_by': 'timestamp',
        'sort_order': 'desc',
        'facets': ['type'],
        'page': 1,
        'page_size': 10
    })

    print(f"   Total results: {results['total_results']}")
    print(f"   Page: {results['page']} of {results['total_pages']}")
    if results.get('facets'):
        print(f"   Facets: {json.dumps(results['facets'], indent=6)}")
    print()

    print("Advanced features demo complete! [OK]\n")


if __name__ == '__main__':
    try:
        # Run main demo
        artifacts, stored_ids, results = demo_knowledge_artifacts()

        # Run advanced demo
        demo_advanced_features()

        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        print(f"\n[X] Error: {str(e)}")
        print("Check logs for details.\n")
        raise
