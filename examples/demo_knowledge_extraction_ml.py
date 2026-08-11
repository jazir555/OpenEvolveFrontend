"""
Demo: Stage 6 Knowledge Extraction with ML Clustering

This demo showcases the ML-based pattern clustering capabilities:
1. ML pattern discovery from workflow traces
2. Entity and relation extraction
3. Temporal knowledge graph construction
4. Knowledge validation with Z3
5. Hybrid semantic + keyword retrieval

Author: OpenEvolve AI
License: Apache 2.0
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import Stage 6 components
try:
    from stage6_knowledge_extraction import (
        Stage6KnowledgeExtraction,
        ExecutionTrace,
        HybridRetrievalSystem
    )
    STAGE6_AVAILABLE = True
except ImportError as e:
    STAGE6_AVAILABLE = False
    print(f"Stage 6 not available: {e}")

# Import ML clustering
try:
    from ml_pattern_clustering import (
        MLKnowledgeExtraction,
        MLPatternClustering,
        TemporalKnowledgeGraph,
        KnowledgeValidator
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"ML clustering not available: {e}")

# Import ACE workflow extractor
try:
    from ace_workflow_knowledge_extractor import WorkflowKnowledgeExtractor
    ACE_AVAILABLE = True
except ImportError as e:
    ACE_AVAILABLE = False
    print(f"ACE extractor not available: {e}")


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """Print section header."""
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print("-" * 70)


def demo_ml_pattern_clustering():
    """Demo 1: ML Pattern Clustering"""
    print_header("DEMO 1: ML Pattern Clustering")
    
    if not ML_AVAILABLE:
        print("[X] ML clustering not available - skipping demo")
        return
    
    # Sample workflow problem descriptions
    texts = [
        # Neural networks / Deep learning cluster
        "Optimize neural network architecture for image classification",
        "Apply deep learning to computer vision problems",
        "Neural network architectures for visual recognition tasks",
        "Improve CNN performance on image datasets",
        "Fine-tune ResNet model for object detection",
        
        # Decision trees / Traditional ML cluster
        "Implement decision trees for tabular data classification",
        "Random forest classifier for structured datasets",
        "Gradient boosting on tabular features with XGBoost",
        "Use LightGBM for efficient tabular learning",
        
        # Hyperparameter optimization cluster
        "Optimize hyperparameters using grid search",
        "Hyperparameter tuning with Bayesian optimization",
        "AutoML for automated hyperparameter selection",
        "Use Optuna for efficient hyperparameter search"
    ]
    
    print(f"\n[*] Clustering {len(texts)} workflow descriptions...")
    print("   Using: Sentence Transformers + DBSCAN")
    
    # Initialize clustering
    clustering = MLPatternClustering(
        model_name='all-MiniLM-L6-v2',
        clustering_algorithm='dbscan'
    )
    
    # Perform clustering
    patterns = clustering.cluster_patterns(texts)
    
    print(f"\n[OK] Discovered {len(patterns)} patterns:")
    print()
    
    for i, pattern in enumerate(patterns, 1):
        print(f"  Pattern {i}: {pattern.pattern_id}")
        print(f"    Type: {pattern.pattern_type}")
        print(f"    Confidence: {pattern.confidence:.2f}")
        print(f"    Cluster Size: {pattern.cluster_size}")
        if pattern.silhouette_score:
            print(f"    Silhouette Score: {pattern.silhouette_score:.3f}")
        print(f"    Description: {pattern.description[:80]}...")
        print(f"    Tags: {', '.join(pattern.tags)}")
        print()
    
    # Show cluster members for first pattern
    if patterns:
        print("  [FOLDER] Sample Cluster Members:")
        for j, member in enumerate(patterns[0].cluster_members[:3], 1):
            print(f"    {j}. {member[:70]}...")
        print()


def demo_entity_relation_extraction():
    """Demo 2: Entity and Relation Extraction"""
    print_header("DEMO 2: Entity & Relation Extraction")
    
    if not ML_AVAILABLE:
        print("[X] ML clustering not available - skipping demo")
        return
    
    from ml_pattern_clustering import EntityExtractor, RelationExtractor
    
    # Sample text
    text = """
    We implemented a neural network architecture that solves image classification 
    problems. The ResNet model improves accuracy on the CIFAR-10 dataset. 
    Our solution depends on PyTorch framework and requires GPU acceleration.
    """
    
    print(f"\n[DOC] Input Text:\n   {text[:100]}...")
    print("\n[SEARCH] Extracting entities and relations...")
    
    # Extract entities
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract_entities(text)
    
    print(f"\n[OK] Found {len(entities)} entities:")
    for entity in entities:
        print(f"   * {entity.entity_type.upper()}: '{entity.text}'")
        print(f"     Confidence: {entity.confidence:.2f}")
    
    # Extract relations
    if entities:
        relation_extractor = RelationExtractor()
        relations = relation_extractor.extract_relations(text, entities)
        
        print(f"\n[LINK] Found {len(relations)} relations:")
        for relation in relations:
            print(f"   * {relation.relation_type}: {relation.source_entity_id} -> {relation.target_entity_id}")
            print(f"     Evidence: '{relation.evidence[:50]}...'" if len(relation.evidence) > 50 else f"     Evidence: '{relation.evidence}'")


def demo_temporal_knowledge_graph():
    """Demo 3: Temporal Knowledge Graph"""
    print_header("DEMO 3: Temporal Knowledge Graph")
    
    if not ML_AVAILABLE:
        print("[X] ML clustering not available - skipping demo")
        return
    
    print("\n[CLOCK] Constructing temporal knowledge graph...")
    
    graph = TemporalKnowledgeGraph()
    
    # Add permanent knowledge
    node1 = graph.add_node(
        content="Neural networks are universal function approximators",
        node_type="fact",
        confidence=0.95
    )
    print(f"\n   [OK] Added permanent fact: {node1.node_id}")
    
    # Add time-bounded knowledge
    node2 = graph.add_node(
        content="BERT model achieves SOTA on GLUE benchmark",
        node_type="fact",
        confidence=0.9,
        valid_from=datetime(2026, 1, 1),
        valid_until=datetime(2026, 12, 31)
    )
    print(f"   [OK] Added time-bounded fact: {node2.node_id}")
    
    # Add expired knowledge
    node3 = graph.add_node(
        content="RNNs are the best for NLP tasks",
        node_type="fact",
        confidence=0.7,
        valid_until=datetime(2020, 1, 1)  # Expired
    )
    print(f"   [OK] Added expired fact: {node3.node_id}")
    
    # Create version
    node4 = graph.create_version(
        node_id=node2.node_id,
        new_content="BERT-large achieves improved SOTA on GLUE with 90.5%",
        confidence=0.92
    )
    if node4:
        print(f"   [OK] Created new version: {node4}")
    
    # Query valid knowledge
    valid = graph.get_valid_knowledge(
        at_time=datetime(2026, 6, 15),
        min_confidence=0.8
    )
    
    print(f"\n[LIST] Valid knowledge (as of 2026-06-15, confidence >= 0.8):")
    for node in valid:
        print(f"   * {node.content[:60]}...")
        print(f"     Confidence: {node.confidence}")
    
    print(f"\n[STATS] Graph Statistics:")
    print(f"   Total nodes: {len(graph.nodes)}")
    print(f"   Total edges: {len(graph.edges)}")
    print(f"   Valid nodes: {len(valid)}")


def demo_knowledge_validation():
    """Demo 4: Knowledge Validation with Z3"""
    print_header("DEMO 4: Knowledge Validation")
    
    if not ML_AVAILABLE:
        print("[X] ML clustering not available - skipping demo")
        return
    
    validator = KnowledgeValidator()
    
    # Validate a pattern
    print("\n[LAB] Validating discovered patterns...")
    
    from ml_pattern_clustering import MLPattern
    
    pattern = MLPattern(
        pattern_id="demo_pattern_001",
        pattern_type="semantic",
        description="Neural networks for computer vision tasks",
        confidence=0.85,
        cluster_size=5,
        silhouette_score=0.67,
        cluster_members=["text1", "text2", "text3", "text4", "text5"]
    )
    
    result = validator.validate_pattern(pattern)
    
    print(f"\n   Pattern: {pattern.pattern_id}")
    print(f"   Valid: {'[OK] Yes' if result['valid'] else '[X] No'}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"\n   Checks performed:")
    for check_name, check_result in result['checks'].items():
        if isinstance(check_result, dict):
            passed = "[OK]" if check_result.get('passed') else "[X]"
            print(f"     {passed} {check_name}: {check_result.get('value', 'N/A')}")
    
    # Check consistency
    print("\n[PUZZLE] Checking consistency between patterns...")
    
    patterns = [
        MLPattern(
            pattern_id="p1",
            pattern_type="semantic",
            description="Neural networks solve vision tasks",
            confidence=0.8,
            cluster_size=3
        ),
        MLPattern(
            pattern_id="p2",
            pattern_type="semantic",
            description="CNNs improve image recognition",
            confidence=0.75,
            cluster_size=4
        )
    ]
    
    consistency = validator.validate_consistency([
        "Neural networks are effective for vision",
        "CNNs are a type of neural network",
        "CNNs work well for images"
    ])
    
    print(f"\n   Consistency check result:")
    print(f"   Consistent: {'[OK] Yes' if consistency.get('consistent') else '[X] No'}")
    print(f"   Message: {consistency.get('message')}")


def demo_hybrid_retrieval():
    """Demo 5: Hybrid Retrieval (Semantic + Keyword)"""
    print_header("DEMO 5: Hybrid Retrieval System")
    
    if not STAGE6_AVAILABLE:
        print("[X] Stage 6 not available - skipping demo")
        return
    
    print("\n[SEARCH] Setting up hybrid retrieval...")
    
    retriever = HybridRetrievalSystem()
    
    # Add knowledge items
    knowledge_items = [
        {
            'id': 'k1',
            'description': 'Neural networks for computer vision and image classification',
            'content': 'Use CNN architectures like ResNet and EfficientNet for visual tasks',
            'tags': ['deep_learning', 'vision']
        },
        {
            'id': 'k2',
            'description': 'Transformer models for natural language processing',
            'content': 'BERT and GPT models excel at NLP tasks like classification and generation',
            'tags': ['nlp', 'transformers']
        },
        {
            'id': 'k3',
            'description': 'Decision trees and ensemble methods for tabular data',
            'content': 'Random Forest and XGBoost work well on structured datasets',
            'tags': ['traditional_ml', 'tabular']
        },
        {
            'id': 'k4',
            'description': 'Hyperparameter optimization techniques',
            'content': 'Bayesian optimization and grid search for tuning model parameters',
            'tags': ['optimization', 'automl']
        },
        {
            'id': 'k5',
            'description': 'Convolutional neural network architectures',
            'content': 'VGG, ResNet, DenseNet for image recognition and object detection',
            'tags': ['cnn', 'vision']
        }
    ]
    
    for item in knowledge_items:
        retriever.add_knowledge(item)
    
    print(f"   [OK] Added {len(knowledge_items)} knowledge items")
    
    # Test queries
    queries = [
        "neural network for vision",
        "NLP text classification",
        "tree-based models"
    ]
    
    for query in queries:
        print(f"\n   [QUERY] Query: '{query}'")
        
        # Semantic search (70% weight)
        results_semantic = retriever.retrieve(query, top_k=3, semantic_weight=0.7)
        print(f"      Semantic results:")
        for i, r in enumerate(results_semantic[:2], 1):
            print(f"        {i}. {r['description'][:50]}... (score: {r['retrieval_score']:.3f})")


def demo_stage6_integration():
    """Demo 6: Complete Stage 6 Integration"""
    print_header("DEMO 6: Complete Stage 6 Integration")
    
    if not STAGE6_AVAILABLE:
        print("[X] Stage 6 not available - skipping demo")
        return
    
    print("\n[GEAR] Initializing Stage 6 Knowledge Extraction Engine...")
    
    engine = Stage6KnowledgeExtraction(enable_ml=True)
    
    # Create sample workflow traces
    print("\n[DOC] Creating sample workflow traces...")
    
    traces = [
        ExecutionTrace(
            trace_id="trace_001",
            workflow_id="wf_001",
            problem_description="Optimize neural network for image classification",
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'hybrid'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 100}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'accuracy': 0.94, 'model': 'ResNet50'},
            execution_time_ms=5000.0,
            timestamp=datetime.now()
        ),
        ExecutionTrace(
            trace_id="trace_002",
            workflow_id="wf_002",
            problem_description="Fine-tune transformer for NLP classification",
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'semantic'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 50}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'accuracy': 0.91, 'model': 'BERT'},
            execution_time_ms=3500.0,
            timestamp=datetime.now()
        ),
        ExecutionTrace(
            trace_id="trace_003",
            workflow_id="wf_003",
            problem_description="Optimize CNN architecture for object detection",
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'hybrid'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 80}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'mAP': 0.78, 'model': 'YOLO'},
            execution_time_ms=4200.0,
            timestamp=datetime.now()
        ),
        ExecutionTrace(
            trace_id="trace_004",
            workflow_id="wf_004",
            problem_description="Tune hyperparameters for deep learning model",
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'parametric'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 30}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'best_score': 0.89, 'method': 'Bayesian'},
            execution_time_ms=2800.0,
            timestamp=datetime.now()
        ),
        ExecutionTrace(
            trace_id="trace_005",
            workflow_id="wf_005",
            problem_description="Improve vision transformer for image recognition",
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'hybrid'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 120}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'accuracy': 0.93, 'model': 'ViT'},
            execution_time_ms=5500.0,
            timestamp=datetime.now()
        )
    ]
    
    print(f"   [OK] Created {len(traces)} workflow traces")
    
    # Process traces
    print("\n[REFRESH] Processing traces (this may take a moment)...")
    
    async def process_all():
        results = []
        for trace in traces:
            result = await engine.process_trace(trace)
            results.append(result)
        return results
    
    results = asyncio.run(process_all())
    
    print("\n   Processing results:")
    for i, result in enumerate(results, 1):
        print(f"      Trace {i}: {result['patterns_extracted']} patterns, "
              f"{result['patterns_validated']} validated, "
              f"{result['artifacts_generated']} artifacts")
    
    # Show statistics
    print("\n[STATS] Final Statistics:")
    stats = engine.get_statistics()
    print(f"   Total traces processed: {stats['traces_processed']}")
    print(f"   Patterns extracted: {stats['patterns_extracted']}")
    print(f"   Pattern types: {stats['pattern_types']}")
    print(f"   Artifacts generated: {stats['artifacts_generated']}")
    print(f"   ML clustered patterns: {stats.get('ml_clustered_patterns', 0)}")
    print(f"   Avg ML silhouette score: {stats.get('avg_ml_silhouette_score', 0):.3f}")
    print(f"   ML available: {'[OK]' if stats['ml_available'] else '[X]'}")
    print(f"   Z3 available: {'[OK]' if stats['z3_available'] else '[X]'}")
    
    # Test retrieval
    print("\n[SEARCH] Testing knowledge retrieval:")
    retrieval_results = engine.retrieve_knowledge("neural network optimization", top_k=5)
    print(f"   Retrieved {len(retrieval_results)} relevant items")
    for i, item in enumerate(retrieval_results[:3], 1):
        print(f"      {i}. {item.get('description', 'N/A')[:50]}...")


def demo_ace_integration():
    """Demo 7: ACE Workflow Extractor Integration"""
    print_header("DEMO 7: ACE Workflow Extractor Integration")
    
    if not ACE_AVAILABLE:
        print("[X] ACE workflow extractor not available - skipping demo")
        return
    
    print("\n[GEAR] Initializing ACE Workflow Knowledge Extractor...")
    
    extractor = WorkflowKnowledgeExtractor(enable_learning=True)
    
    # Show ML stats
    ml_stats = extractor.get_ml_extraction_stats()
    print("\n   ML Components Status:")
    for component, available in ml_stats.items():
        if isinstance(available, bool):
            status = "[OK] Available" if available else "[X] Not Available"
            print(f"      {component}: {status}")
    
    # Extract from sample workflow
    print("\n[DOC] Extracting knowledge from workflow...")
    
    workflow_results = {
        'phases': {
            'phase_1_decomposition': {
                'success': True,
                'analysis': 'Decomposed into 3 sub-problems using hybrid strategy',
                'learning': {
                    'reflection_summary': 'Hybrid decomposition effective for ML optimization'
                }
            },
            'phase_2_evolution': {
                'success': True,
                'solutions': [
                    {'solution': 'ResNet-50 architecture', 'fitness': 0.94},
                    {'solution': 'EfficientNet-B3', 'fitness': 0.93}
                ]
            },
            'phase_3_verification': {
                'success': True,
                'verifications': [
                    {'verified': True, 'method': 'cross_validation'}
                ]
            }
        },
        'teams': {
            'blue_team': {
                'name': 'Blue Team',
                'type': 'blue_team',
                'tasks_completed': 10,
                'tasks_succeeded': 9,
                'avg_quality_score': 0.92
            },
            'red_team': {
                'name': 'Red Team',
                'type': 'red_team',
                'issues_found': 2,
                'true_positives': 2
            }
        }
    }
    
    result = extractor.extract_from_workflow(
        workflow_id="demo_workflow_001",
        problem_statement="Optimize neural network architecture for vision tasks",
        workflow_results=workflow_results
    )
    
    print(f"\n   [OK] Extraction complete:")
    print(f"      Workflow ID: {result.workflow_id}")
    print(f"      Total artifacts: {result.total_artifacts}")
    print(f"      Pattern artifacts: {result.pattern_count}")
    print(f"      Solution patterns: {result.solution_count}")
    print(f"      Anti-patterns: {result.anti_pattern_count}")


def print_summary():
    """Print demo summary."""
    print_header("DEMO SUMMARY")
    
    print("""
[TARGET] Stage 6 Knowledge Extraction - ML Clustering Implementation Complete!

[OK] Implemented Components:
   1. ML Pattern Clustering (Sentence Transformers + scikit-learn)
   2. Entity & Relation Extraction
   3. Temporal Knowledge Graph
   4. Knowledge Validation (Z3)
   5. Hybrid Retrieval (Semantic + Keyword)
   6. ACE Workflow Integration

[CHART] Key Features:
   * DBSCAN, KMeans, Hierarchical clustering algorithms
   * Automatic cluster quality evaluation (silhouette score)
   * Representative example selection
   * Time-aware knowledge storage with versioning
   * Z3-based logical consistency checking
   * Semantic search using embeddings
   * Fallback to rule-based when ML unavailable

[WRENCH] Libraries Used:
   * sentence-transformers (Apache 2.0) - Embeddings
   * scikit-learn (BSD) - Clustering algorithms
   * z3-solver (MIT) - Formal validation
   * networkx (BSD) - Graph operations
   * numpy (BSD) - Numerical operations

[DOC] Next Steps:
   * Run tests: pytest test_knowledge_extraction_comprehensive.py
   * See documentation: STAGE6_IMPLEMENTATION_COMPLETE.md
   * Integrate with your workflows

""")


async def main():
    """Main demo function."""
    print("""
+======================================================================+
|                                                                      |
|     Stage 6 Knowledge Extraction - ML Clustering Demo                |
|                                                                      |
|     Demonstrates ML-based pattern discovery and knowledge mgmt       |
|                                                                      |
+======================================================================+
    """)
    
    print(f"\n[CLOCK] Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all demos
    demo_ml_pattern_clustering()
    demo_entity_relation_extraction()
    demo_temporal_knowledge_graph()
    demo_knowledge_validation()
    demo_hybrid_retrieval()
    demo_stage6_integration()
    demo_ace_integration()
    
    # Summary
    print_summary()
    
    print(f"\n[CLOCK] Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
