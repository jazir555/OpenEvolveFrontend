"""
KarateClub Analytics - Quick Start Guide

Run this script to verify your KarateClub installation and see all 51 algorithms in action.
"""

import asyncio
import networkx as nx
from knowledge_engine.integrations.karateclub_analytics import KarateClubAnalytics
from knowledge_engine.integrations.karateclub_algorithms import KarateClubAlgorithmRegistry


async def quick_start():
    """Quick start demonstration"""
    print("\n" + "=" * 80)
    print("KarateClub Analytics - Quick Start")
    print("=" * 80)

    # Show algorithm counts
    counts = KarateClubAlgorithmRegistry.get_total_count()
    print(f"\n📊 Algorithm Coverage:")
    print(f"   Community Detection: {counts['community']} algorithms")
    print(f"   Node Embeddings: {counts['node_embedding']} algorithms")
    print(f"   Graph Embeddings: {counts['graph_embedding']} algorithms")
    print(f"   Total: {counts['total']} algorithms [OK]")

    # Initialize analytics
    print(f"\n🚀 Initializing KarateClub Analytics...")
    analytics = KarateClubAnalytics()
    print("   [OK] Analytics engine ready")

    # Create sample graph
    print(f"\n📈 Creating sample graph...")
    graph = nx.karate_club_graph()
    print(f"   [OK] Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Example 1: Community Detection
    print(f"\n🔍 Community Detection (Label Propagation)...")
    communities = await analytics.detect_communities(graph, algorithm='label_propagation')
    print(f"   [OK] Found {communities.num_communities} communities")
    print(f"   [OK] Modularity: {communities.modularity:.3f}")
    print(f"   [OK] Time: {communities.execution_time_ms:.2f}ms")

    # Example 2: Node Embeddings
    print(f"\n🎯 Node Embeddings (Node2Vec)...")
    embeddings = await analytics.generate_node_embeddings(
        graph,
        algorithm='node2vec',
        dimensions=128
    )
    print(f"   [OK] Embedded {embeddings.num_nodes} nodes")
    print(f"   [OK] Dimensions: {embeddings.embedding_dim}")
    print(f"   [OK] Time: {embeddings.execution_time_ms:.2f}ms")

    # Example 3: Graph Metrics
    print(f"\n📊 Graph Metrics...")
    metrics = await analytics.compute_graph_metrics(graph)
    print(f"   [OK] Nodes: {metrics.num_nodes}")
    print(f"   [OK] Edges: {metrics.num_edges}")
    print(f"   [OK] Density: {metrics.density:.3f}")
    print(f"   [OK] Clustering: {metrics.avg_clustering:.3f}")
    print(f"   [OK] Connected: {metrics.is_connected}")
    print(f"   [OK] Time: {metrics.execution_time_ms:.2f}ms")

    # Example 4: Complete Structure Analysis
    print(f"\n🔬 Complete Structure Analysis...")
    structure = await analytics.analyze_graph_structure(graph)
    print(f"   [OK] Communities: {structure.communities.num_communities}")
    print(f"   [OK] Modularity: {structure.communities.modularity:.3f}")
    print(f"   [OK] Density: {structure.metrics.density:.3f}")
    print(f"   [OK] Time: {structure.execution_time_ms:.2f}ms")

    # Show available algorithms
    print(f"\n📚 Available Algorithms:")
    print(f"\n   Community Detection (10):")
    community_algos = list(KarateClubAlgorithmRegistry.get_algorithms_by_category('community').keys())
    for algo in community_algos[:5]:
        print(f"      - {algo}")
    print(f"      ... ({len(community_algos) - 5} more)")

    print(f"\n   Node Embeddings (32):")
    node_algos = list(KarateClubAlgorithmRegistry.get_algorithms_by_category('node_embedding').keys())
    for algo in node_algos[:5]:
        print(f"      - {algo}")
    print(f"      ... ({len(node_algos) - 5} more)")

    print(f"\n   Graph Embeddings (10):")
    graph_algos = list(KarateClubAlgorithmRegistry.get_algorithms_by_category('graph_embedding').keys())
    for algo in graph_algos[:5]:
        print(f"      - {algo}")
    print(f"      ... ({len(graph_algos) - 5} more)")

    # Success!
    print(f"\n" + "=" * 80)
    print("[OK] SUCCESS! KarateClub Analytics is working correctly!")
    print("=" * 80)

    print(f"\n📖 Next Steps:")
    print(f"   1. Read the documentation: knowledge_engine/integrations/KARATECLUB_README.md")
    print(f"   2. Run examples: python -m knowledge_engine.integrations.example_karateclub")
    print(f"   3. Run tests: pytest knowledge_engine/integrations/test_karateclub.py -v")
    print(f"   4. Integrate with Knowledge Engine: See unified_knowledge_graph.py")

    print(f"\n🎉 Happy Analyzing!")
    print()


if __name__ == '__main__':
    try:
        asyncio.run(quick_start())
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        print(f"\n💡 Make sure KarateClub is installed:")
        print(f"   pip install karateclub networkx numpy scipy")
