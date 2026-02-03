"""
Example Usage of Knowledge Engine Integrations

This file demonstrates how to use all the integrated projects:
- PAMI (Pattern Mining)
- NeuralKG (KG Embeddings)
- Causal-Learn (Causal Discovery)
- Lagrange-Mapper (Topological Analysis)
"""

import numpy as np
from typing import Dict, Any


def example_pami_pattern_mining():
    """Example: Pattern Mining with PAMI"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Pattern Mining with PAMI")
    print("="*60)
    
    from knowledge_engine.integrations import PAMIPatternMiner
    
    miner = PAMIPatternMiner()
    
    if not miner.is_available():
        print("PAMI not available, skipping example")
        return
    
    # Sample transaction data (shopping baskets)
    transactions = [
        ['bread', 'milk', 'eggs'],
        ['bread', 'butter', 'milk'],
        ['milk', 'eggs', 'cheese'],
        ['bread', 'milk', 'butter', 'eggs'],
        ['bread', 'eggs'],
        ['milk', 'butter', 'cheese'],
        ['bread', 'milk', 'eggs', 'cheese'],
        ['eggs', 'cheese'],
        ['bread', 'milk'],
        ['butter', 'cheese']
    ]
    
    print("\nTransaction data:")
    for i, t in enumerate(transactions[:3], 1):
        print(f"  Transaction {i}: {t}")
    print("  ...")
    
    # Mine frequent patterns
    print("\n1. Frequent Pattern Mining:")
    result = miner.mine_frequent_patterns(
        transactions=transactions,
        min_support=0.2,
        algorithm='fpgrowth'
    )
    
    if result['status'] == 'success':
        print(f"   Found {result['statistics']['total_patterns']} patterns")
        print("   Top patterns:")
        for pattern in result['patterns'][:5]:
            print(f"     {pattern['pattern']}: support={pattern['support_ratio']:.2f}")
    
    # Discover association rules
    print("\n2. Association Rule Discovery:")
    result = miner.discover_association_rules(
        transactions=transactions,
        min_support=0.2,
        min_confidence=0.5
    )
    
    if result['status'] == 'success':
        print(f"   Found {result['statistics']['total_rules']} rules")
        print("   Top rules:")
        for rule in result['rules'][:3]:
            ant = ', '.join(rule['antecedent'])
            con = ', '.join(rule['consequent'])
            print(f"     {ant} -> {con}: conf={rule['confidence']:.2f}, supp={rule['support']:.2f}")
    
    # Graph pattern analysis
    print("\n3. Knowledge Graph Pattern Analysis:")
    graph_data = {
        'nodes': [
            {'id': 'Alice', 'type': 'Person'},
            {'id': 'Bob', 'type': 'Person'},
            {'id': 'Charlie', 'type': 'Person'},
            {'id': 'Dave', 'type': 'Person'},
            {'id': 'AcmeCorp', 'type': 'Organization'},
            {'id': 'TechInc', 'type': 'Organization'},
            {'id': 'StartupX', 'type': 'Organization'}
        ],
        'edges': [
            {'source': 'Alice', 'target': 'Bob', 'type': 'knows'},
            {'source': 'Bob', 'target': 'Charlie', 'type': 'knows'},
            {'source': 'Charlie', 'target': 'Dave', 'type': 'knows'},
            {'source': 'Alice', 'target': 'AcmeCorp', 'type': 'works_for'},
            {'source': 'Bob', 'target': 'TechInc', 'type': 'works_for'},
            {'source': 'Charlie', 'target': 'TechInc', 'type': 'works_for'},
            {'source': 'Dave', 'target': 'StartupX', 'type': 'works_for'},
            {'source': 'AcmeCorp', 'target': 'TechInc', 'type': 'partner'},
            {'source': 'TechInc', 'target': 'StartupX', 'type': 'invested_in'}
        ]
    }
    
    result = miner.analyze_knowledge_graph_patterns(graph_data, min_support=0.1)
    
    if result['status'] == 'success':
        stats = result['statistics']
        print(f"   Nodes: {stats['total_nodes']}, Edges: {stats['total_edges']}")
        print(f"   Entity types: {stats['unique_entity_types']}")
        print(f"   Triple patterns: {len(result['patterns'].get('triple_patterns', []))}")


def example_neuralkg_embeddings():
    """Example: Knowledge Graph Embeddings with NeuralKG"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Knowledge Graph Embeddings with NeuralKG")
    print("="*60)
    
    from knowledge_engine.integrations import NeuralKGEmbedder
    
    embedder = NeuralKGEmbedder()
    
    if not embedder.is_available():
        print("NeuralKG not available, skipping example")
        return
    
    # Sample knowledge graph triples
    triples = [
        ('Alice', 'knows', 'Bob'),
        ('Bob', 'knows', 'Charlie'),
        ('Charlie', 'knows', 'Dave'),
        ('Alice', 'works_for', 'AcmeCorp'),
        ('Bob', 'works_for', 'TechInc'),
        ('Charlie', 'works_for', 'TechInc'),
        ('Dave', 'works_for', 'StartupX'),
        ('AcmeCorp', 'partner', 'TechInc'),
        ('TechInc', 'invested_in', 'StartupX'),
        ('Alice', 'lives_in', 'NewYork'),
        ('Bob', 'lives_in', 'SanFrancisco'),
        ('Charlie', 'lives_in', 'SanFrancisco'),
        ('Dave', 'lives_in', 'Boston')
    ]
    
    print(f"\nKnowledge graph: {len(triples)} triples")
    print("Sample triples:")
    for t in triples[:3]:
        print(f"  ({t[0]}, {t[1]}, {t[2]})")
    
    # Generate embeddings
    print("\n1. Generating TransE Embeddings:")
    result = embedder.generate_embeddings(
        triples=triples,
        model_name='transe',
        embedding_dim=50
    )
    
    if result['status'] == 'success':
        embeddings = result['embeddings']
        entities = embeddings['entities']
        relations = embeddings['relations']
        
        print(f"   Entity embeddings: {len(entities)} entities")
        print(f"   Relation embeddings: {len(relations)} relations")
        print(f"   Embedding dimension: {result['metadata']['embedding_dim']}")
        
        # Show sample embeddings
        print("\n   Sample entity embeddings:")
        for entity in list(entities.keys())[:3]:
            emb = entities[entity]
            print(f"     {entity}: [{emb[0]:.3f}, {emb[1]:.3f}, ..., {emb[-1]:.3f}]")
    
    # Link prediction
    print("\n2. Link Prediction:")
    predictions = embedder.predict_links(
        head='Alice',
        relation='knows',
        candidate_tails=['Charlie', 'Dave', 'AcmeCorp', 'TechInc'],
        embeddings=embeddings,
        top_k=3
    )
    
    if predictions['status'] == 'success':
        print(f"   Predictions for 'Alice knows ?':")
        for pred in predictions['predictions']:
            print(f"     {pred['tail']}: score={pred['score']:.3f}, prob={pred['probability']:.3f}")
    
    # Find similar entities
    print("\n3. Finding Similar Entities:")
    similar = embedder.find_similar_entities(
        entity='TechInc',
        embeddings=embeddings,
        top_k=3
    )
    
    if similar['status'] == 'success':
        print(f"   Entities most similar to 'TechInc':")
        for sim in similar['similar_entities']:
            print(f"     {sim['entity']}: similarity={sim['similarity']:.3f}")
    
    # Ensemble embeddings
    print("\n4. Ensemble Embeddings (averaging multiple models):")
    ensemble = embedder.ensemble_embeddings(
        triples=triples,
        models=['transe'],  # Add more models if available
        embedding_dim=50
    )
    
    if ensemble['status'] == 'success':
        print(f"   Ensemble generated with {ensemble['metadata']['num_models']} models")


def example_causal_discovery():
    """Example: Causal Discovery with Causal-Learn"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Causal Discovery with Causal-Learn")
    print("="*60)
    
    from knowledge_engine.integrations import CausalDiscoveryEngine
    
    engine = CausalDiscoveryEngine()
    
    if not engine.is_available():
        print("Causal-Learn not available, skipping example")
        return
    
    # Generate synthetic data with known causal structure
    # X -> Y -> Z (chain)
    # X -> Z (direct)
    print("\nGenerating synthetic data with causal structure:")
    print("  X -> Y -> Z")
    print("  X -> Z")
    
    np.random.seed(42)
    n_samples = 500
    
    X = np.random.randn(n_samples)
    Y = 2.0 * X + np.random.randn(n_samples) * 0.5
    Z = 1.5 * Y + 0.5 * X + np.random.randn(n_samples) * 0.5
    
    data = np.column_stack([X, Y, Z])
    variable_names = ['X', 'Y', 'Z']
    
    print(f"\nData shape: {data.shape}")
    
    # PC Algorithm
    print("\n1. PC Algorithm (constraint-based):")
    result = engine.discover_causal_structure(
        data=data,
        variable_names=variable_names,
        algorithm='pc',
        alpha=0.05,
        independence_test='fisherz'
    )
    
    if result['status'] == 'success':
        graph = result['graph']
        print(f"   Discovered {len(graph['edges'])} edges:")
        for edge in graph['edges'][:5]:
            print(f"     {edge['source']} -> {edge['target']}: {edge['type']}")
    
    # Analyze causal graph
    print("\n2. Causal Graph Analysis:")
    analysis = engine.analyze_causal_graph(graph)
    
    if analysis['status'] == 'success':
        stats = analysis['analysis']
        print(f"   Nodes: {stats['num_nodes']}")
        print(f"   Edges: {stats['num_edges']}")
        print(f"   Roots (no parents): {stats['roots']}")
        print(f"   Leaves (no children): {stats['leaves']}")
        print(f"   Average out-degree: {stats['avg_out_degree']:.2f}")
    
    # Confounder identification example
    print("\n3. Confounder Identification:")
    
    # Create a graph with confounder W
    # W -> X, W -> Y, X -> Y
    graph_with_confounder = {
        'nodes': ['W', 'X', 'Y'],
        'edges': [
            {'source': 'W', 'target': 'X', 'type': 'directed'},
            {'source': 'W', 'target': 'Y', 'type': 'directed'},
            {'source': 'X', 'target': 'Y', 'type': 'directed'}
        ]
    }
    
    confounders = engine.identify_confounders(
        graph_data=graph_with_confounder,
        target_x='X',
        target_y='Y'
    )
    
    if confounders['status'] == 'success':
        conf = confounders['confounders']
        print(f"   Common causes: {conf['common_causes']}")
        print(f"   Mediators: {conf['mediators']}")
        print(f"   Adjustment set: {conf['adjustment_set']}")


def example_lagrange_mapper():
    """Example: Topological Analysis with Lagrange-Mapper"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Topological Analysis with Lagrange-Mapper")
    print("="*60)
    
    from knowledge_engine.integrations import LagrangeAttractorAnalyzer
    
    analyzer = LagrangeAttractorAnalyzer()
    
    if not analyzer.is_available():
        print("Lagrange-Mapper not available, skipping example")
        return
    
    # Generate embeddings with known cluster structure
    print("\nGenerating synthetic embeddings with 3 clusters...")
    np.random.seed(42)
    
    # Cluster 1
    cluster1 = np.random.randn(30, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # Cluster 2
    cluster2 = np.random.randn(30, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    # Cluster 3
    cluster3 = np.random.randn(40, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    labels = [f'point_{i}' for i in range(100)]
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Analyze embedding landscape
    print("\n1. Embedding Landscape Analysis:")
    result = analyzer.analyze_embedding_landscape(
        embeddings=embeddings,
        labels=labels,
        n_clusters=3,
        reduction_method='pca',
        reduction_dims=2
    )
    
    if result['status'] == 'success':
        landscape = result['landscape']
        print(f"   Found {landscape['n_clusters']} clusters")
        print(f"   Samples: {landscape['n_samples']}")
        
        print("\n   Cluster details:")
        for cluster in landscape['clusters']:
            print(f"     Cluster {cluster['cluster_id']}: "
                  f"size={cluster['size']}, "
                  f"density={cluster['density']:.3f}")
        
        print("\n   Attractor strengths:")
        for att in landscape['attractors'][:3]:
            print(f"     Attractor {att['cluster_id']}: "
                  f"strength={att['strength']:.3f}, "
                  f"tightness={att['tightness']:.3f}")
    
    # Knowledge graph topology
    print("\n2. Knowledge Graph Topology Analysis:")
    
    graph_data = {
        'nodes': [{'id': f'node_{i}', 'type': 'entity'} for i in range(20)],
        'edges': [
            {'source': f'node_{i}', 'target': f'node_{(i+1) % 20}', 'type': 'connected'}
            for i in range(20)
        ] + [
            {'source': f'node_{i}', 'target': f'node_{(i+5) % 20}', 'type': 'related'}
            for i in range(0, 20, 2)
        ]
    }
    
    result = analyzer.analyze_knowledge_topology(
        graph_data=graph_data,
        embedding_dim=10
    )
    
    if result['status'] == 'success':
        print(f"   Topology analyzed successfully")
        if 'landscape' in result and 'graph_metrics' in result['landscape']:
            metrics = result['landscape']['graph_metrics']
            print(f"   Nodes: {metrics['num_nodes']}")
            print(f"   Edges: {metrics['num_edges']}")
            print(f"   Density: {metrics['density']:.3f}")
            print(f"   Connected components: {metrics['connected_components']}")


def example_unified_extractor():
    """Example: Using the Unified Knowledge Extractor"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Unified Knowledge Extractor")
    print("="*60)
    
    from knowledge_engine.integrations.unified_knowledge_extraction import (
        UnifiedKnowledgeExtractor, extract_knowledge
    )
    
    extractor = UnifiedKnowledgeExtractor()
    
    print("\n1. Extractor Status:")
    status = extractor.get_status()
    print(f"   Available modules: {status['available_modules']}")
    print(f"   Capabilities: {status['capabilities']}")
    
    # Run extraction pipeline
    print("\n2. Running Extraction Pipeline:")
    
    input_data = {
        'text': 'Alice works at AcmeCorp. Bob knows Alice. Charlie works at TechInc.',
        'graph': {
            'nodes': [
                {'id': 'Alice', 'type': 'Person'},
                {'id': 'Bob', 'type': 'Person'},
                {'id': 'Charlie', 'type': 'Person'},
                {'id': 'AcmeCorp', 'type': 'Organization'},
                {'id': 'TechInc', 'type': 'Organization'}
            ],
            'edges': [
                {'source': 'Alice', 'target': 'Bob', 'type': 'colleague'},
                {'source': 'Bob', 'target': 'Charlie', 'type': 'knows'},
                {'source': 'Alice', 'target': 'AcmeCorp', 'type': 'works_for'},
                {'source': 'Charlie', 'target': 'TechInc', 'type': 'works_for'}
            ]
        },
        'triples': [
            ('Alice', 'knows', 'Bob'),
            ('Bob', 'knows', 'Charlie'),
            ('Alice', 'works_for', 'AcmeCorp')
        ]
    }
    
    result = extractor.run_extraction_pipeline(
        input_data=input_data,
        pipeline_config={
            'extract_text': True,
            'analyze_graph': True,
            'generate_embeddings': True
        }
    )
    
    print(f"   Pipeline status: {result.status}")
    print(f"   Stages completed: {result.data.get('pipeline_stages', [])}")
    
    if result.errors:
        print(f"   Errors: {result.errors}")
    
    # Quick extraction function
    print("\n3. Quick Extraction Function:")
    quick_result = extract_knowledge(
        data={'text': 'Sample text for quick extraction'},
        operations=['text']
    )
    
    print(f"   Quick extraction status: {quick_result['status']}")


def example_integrator_combined():
    """Example: Combined Usage with AIKnowledgeGraphIntegrator"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Combined Usage (AIKnowledgeGraphIntegrator)")
    print("="*60)
    
    from knowledge_engine.integrations import AIKnowledgeGraphIntegrator
    
    integrator = AIKnowledgeGraphIntegrator()
    
    print("\n1. Integration Status:")
    status = integrator.get_integration_status()
    for module, available in status.items():
        if module != 'timestamp':
            print(f"   {module}: {'✓' if available else '✗'}")
    
    # Mine patterns
    print("\n2. Pattern Mining:")
    transactions = [
        ['Person', 'Organization', 'Location'],
        ['Person', 'Organization'],
        ['Person', 'Person', 'Organization'],
        ['Organization', 'Location'],
        ['Person', 'Location']
    ]
    
    result = integrator.mine_patterns_with_pami(
        {'transactions': transactions},
        config={'mining_type': 'frequent_patterns', 'min_support': 0.2}
    )
    
    if result['status'] == 'success':
        print(f"   Found {len(result.get('patterns', []))} patterns")
    
    # Generate embeddings
    print("\n3. Embedding Generation:")
    triples = [
        ('Alice', 'type', 'Person'),
        ('Bob', 'type', 'Person'),
        ('AcmeCorp', 'type', 'Organization'),
        ('Alice', 'works_for', 'AcmeCorp'),
        ('Bob', 'works_for', 'AcmeCorp')
    ]
    
    result = integrator.embed_knowledge_graph_with_neuralkg(
        triples,
        model='transe',
        config={'embedding_dim': 50}
    )
    
    if result['status'] == 'success':
        entities = result.get('embeddings', {}).get('entities', {})
        print(f"   Generated embeddings for {len(entities)} entities")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("KNOWLEDGE ENGINE INTEGRATIONS - EXAMPLES")
    print("="*60)
    
    examples = [
        ("Pattern Mining (PAMI)", example_pami_pattern_mining),
        ("KG Embeddings (NeuralKG)", example_neuralkg_embeddings),
        ("Causal Discovery (Causal-Learn)", example_causal_discovery),
        ("Topological Analysis (Lagrange-Mapper)", example_lagrange_mapper),
        ("Unified Extractor", example_unified_extractor),
        ("Combined Usage", example_integrator_combined)
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()
