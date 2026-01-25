#!/usr/bin/env python3
"""
Quick Start: Knowledge Artifacts

This script demonstrates the easiest way to get started with knowledge artifacts.
Run this to see the system in action.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime, timezone
from knowledge_engine.knowledge_extractor import KnowledgeExtractor
from knowledge_engine.knowledge_storage import KnowledgeStorage
from knowledge_engine.knowledge_retriever import KnowledgeRetriever
import json

def quick_start():
    """Quick start example"""

    print("\n" + "="*70)
    print("KNOWLEDGE ARTIFACTS - QUICK START")
    print("="*70 + "\n")

    # 1. Create sample workflow data
    print("Step 1: Creating sample workflow data...")
    workflow_data = {
        'workflow_id': 'quickstart_001',
        'domain': 'optimization',
        'complexity': 'high',
        'execution_time': 1200,
        'success': True,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'solutions': [
            {
                'id': 'sol_001',
                'problem_type': 'optimization',
                'domain': 'math',
                'approach': 'gradient descent',
                'implementation': 'iterative approach',
                'success_rate': 0.92,
                'complexity': 7,
                'code': 'def optimize(): pass',
                'documentation': 'Standard gradient descent optimization',
                'performance': {'iterations': 100}
            }
        ],
        'critiques': [
            {
                'id': 'crit_001',
                'issue_type': 'convergence',
                'root_cause': 'learning rate too high',
                'prevention_strategy': 'adaptive learning rate',
                'severity': 'medium',
                'affected_components': ['optimizer']
            }
        ],
        'teams': [
            {
                'name': 'blue_team',
                'role': 'Blue',
                'domain': 'optimization',
                'success_rate': 0.90,
                'avg_response_time': 1.5,
                'completion_rate': 0.93,
                'quality_score': 0.88,
                'performance_trends': [0.85, 0.87, 0.88, 0.89, 0.90]
            }
        ],
        'gauntlets': [
            {
                'name': 'quality_gauntlet',
                'type': 'Gold',
                'domain': 'validation',
                'problem_type': 'quality',
                'detection_rate': 0.88,
                'false_positive_rate': 0.05,
                'true_positive_rate': 0.85,
                'average_score': 0.87,
                'performance_trends': [0.83, 0.85, 0.86, 0.87, 0.88]
            }
        ]
    }
    print("  [OK] Workflow data created\n")

    # 2. Initialize components
    print("Step 2: Initializing knowledge components...")
    extractor = KnowledgeExtractor()
    storage = KnowledgeStorage()
    retriever = KnowledgeRetriever(storage=storage)
    print("  [OK] Components initialized\n")

    # 3. Extract knowledge
    print("Step 3: Extracting knowledge from workflow...")
    artifacts = extractor.extract_from_workflow(workflow_data)
    print(f"  [OK] Extracted {len(artifacts)} artifacts\n")

    # 4. Show extracted artifacts
    print("Step 4: Extracted artifact types:")
    artifact_types = {}
    for artifact in artifacts:
        artifact_types[artifact.artifact_type] = artifact_types.get(artifact.artifact_type, 0) + 1
    for artifact_type, count in artifact_types.items():
        print(f"  - {artifact_type}: {count}")
    print()

    # 5. Store artifacts
    print("Step 5: Storing artifacts in knowledge base...")
    for artifact in artifacts:
        artifact_dict = artifact.to_dict()
        artifact_dict['type'] = artifact.artifact_type
        artifact_dict['source'] = artifact.source_workflow_id
        artifact_dict['content'] = json.dumps(artifact.content)
        storage.store_knowledge_artifact(artifact_dict)
    print("  [OK] All artifacts stored\n")

    # 6. Search knowledge
    print("Step 6: Searching knowledge base...")
    results = retriever.search_knowledge('optimization', limit=5)
    print(f"  [OK] Found {len(results)} results\n")

    # 7. Get quality metrics
    print("Step 7: Knowledge base quality metrics...")
    metrics = retriever.get_knowledge_quality_metrics()
    quality = metrics['quality_metrics']
    print(f"  - Overall quality: {metrics['overall_quality_score']:.2f}")
    print(f"  - Completeness: {quality['completeness']:.2f}")
    print(f"  - Consistency: {quality['consistency']:.2f}")
    print(f"  - Relevance: {quality['relevance']:.2f}")
    print()

    # 8. Summary
    print("="*70)
    print("QUICK START COMPLETE!")
    print("="*70)
    print("\nWhat you just did:")
    print("  1. Created sample workflow data")
    print("  2. Initialized knowledge components")
    print("  3. Extracted knowledge artifacts")
    print("  4. Stored artifacts in knowledge base")
    print("  5. Searched the knowledge base")
    print("  6. Retrieved quality metrics")
    print("\nNext steps:")
    print("  - Read KNOWLEDGE_ARTIFACTS_GUIDE.md for detailed documentation")
    print("  - Run knowledge_artifacts_demo.py for advanced examples")
    print("  - Check test_knowledge_artifacts.py for more usage patterns")
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    try:
        quick_start()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
