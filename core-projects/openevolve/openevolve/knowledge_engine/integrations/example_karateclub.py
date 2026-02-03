"""
KarateClub Analytics Usage Examples

Comprehensive examples demonstrating all 51 KarateClub algorithms:
- Community detection (10 algorithms)
- Node embeddings (32 algorithms)
- Graph embeddings (10 algorithms)
- Graph metrics and analysis
- Retrieval with embeddings
- Workflow integration
"""

import asyncio
import networkx as nx
import numpy as np

from knowledge_engine.integrations.karateclub_analytics import KarateClubAnalytics
from knowledge_engine.integrations.karateclub_algorithms import KarateClubAlgorithmRegistry
from knowledge_engine.integrations.karateclub_retrieval import KarateClubRetrieval
from knowledge_engine.integrations.karateclub_workflow import KarateClubWorkflowIntegration


async def example_community_detection():
    """Example: Community detection algorithms"""
    print("\n" + "=" * 80)
    print("Community Detection Examples")
    print("=" * 80)

    # Create sample graph
    graph = nx.karate_club_graph()

    # Initialize analytics
    analytics = KarateClubAnalytics()

    # Example 1: Label Propagation (fast)
    print("\n1. Label Propagation (fast)")
    communities = await analytics.detect_communities(
        graph,
        algorithm='label_propagation'
    )
    print(f"   Found {communities.num_communities} communities")
    print(f"   Modularity: {communities.modularity:.3f}")
    print(f"   Execution time: {communities.execution_time_ms:.2f}ms")

    # Example 2: GEMSEC (with embeddings)
    print("\n2. GEMSEC (with embeddings)")
    communities = await analytics.detect_communities(
        graph,
        algorithm='gemsec',
        dimensions=32
    )
    print(f"   Found {communities.num_communities} communities")
    print(f"   Algorithm: {communities.algorithm}")

    # Example 3: EdMot (edge motif)
    print("\n3. EdMot (edge motif)")
    communities = await analytics.detect_communities(
        graph,
        algorithm='edmot',
        components=8
    )
    print(f"   Found {communities.num_communities} communities")

    # List all community algorithms
    print("\nAll 10 community detection algorithms:")
    community_algos = KarateClubAlgorithmRegistry.get_algorithms_by_category('community')
    for name, info in list(community_algos.items())[:5]:
        print(f"   - {name}: {info['name']}")
    print("   ... (5 more)")


async def example_node_embeddings():
    """Example: Node embedding algorithms"""
    print("\n" + "=" * 80)
    print("Node Embedding Examples")
    print("=" * 80)

    # Create sample graph
    graph = nx.karate_club_graph()

    # Initialize analytics
    analytics = KarateClubAnalytics()

    # Example 1: DeepWalk
    print("\n1. DeepWalk (random walks)")
    result = await analytics.generate_node_embeddings(
        graph,
        algorithm='deepwalk',
        dimensions=128,
        walk_length=80,
        walk_number=10
    )
    print(f"   Embedded {result.num_nodes} nodes")
    print(f"   Dimensions: {result.embedding_dim}")
    print(f"   Execution time: {result.execution_time_ms:.2f}ms")

    # Example 2: Node2Vec (biased walks)
    print("\n2. Node2Vec (biased walks)")
    result = await analytics.generate_node_embeddings(
        graph,
        algorithm='node2vec',
        dimensions=128,
        p=1.0,  # Return parameter
        q=2.0   # In-out parameter
    )
    print(f"   Embedded {result.num_nodes} nodes")
    print(f"   Sample embedding: {list(result.embeddings.values())[0][:5]}...")

    # Example 3: GraRep (k-step loss)
    print("\n3. GraRep (k-step loss)")
    result = await analytics.generate_node_embeddings(
        graph,
        algorithm='grarep',
        dimensions=128,
        order=5
    )
    print(f"   Embedded {result.num_nodes} nodes")

    # Example 4: GraphWave (structural roles)
    print("\n4. GraphWave (structural roles)")
    result = await analytics.generate_node_embeddings(
        graph,
        algorithm='graphwave',
        dimensions=128,
        scales=[5, 10, 15]
    )
    print(f"   Structural embeddings for {result.num_nodes} nodes")

    # Example 5: HOPE (high-order proximities)
    print("\n5. HOPE (high-order proximities)")
    result = await analytics.generate_node_embeddings(
        graph,
        algorithm='hope',
        dimensions=128
    )
    print(f"   Preserved high-order proximities for {result.num_nodes} nodes")

    # List all node embedding algorithms
    print("\n32 Node embedding algorithms available:")
    node_algos = KarateClubAlgorithmRegistry.get_algorithms_by_category('node_embedding')
    categories = {
        'Neighbourhood': ['deepwalk', 'node2vec', 'walklets', 'grarep', 'hope', 'netmf'],
        'Structural': ['graphwave', 'role2vec', 'sinr'],
        'Attributed': ['feather_n', 'tadw', 'musae', 'ae', 'fscnmf']
    }
    for category, algos in categories.items():
        print(f"   {category}: {', '.join(algos[:3])}...")
    print(f"   Total: {len(node_algos)} algorithms")


async def example_graph_embeddings():
    """Example: Graph embedding algorithms"""
    print("\n" + "=" * 80)
    print("Graph Embedding Examples")
    print("=" * 80)

    # Create multiple graphs
    graphs = [
        nx.karate_club_graph(),
        nx.erdos_renyi_graph(30, 0.1),
        nx.barabasi_albert_graph(30, 3),
        nx.watts_strogatz_graph(30, 4, 0.1)
    ]

    # Initialize analytics
    analytics = KarateClubAnalytics()

    # Example 1: Graph2Vec
    print("\n1. Graph2Vec (Weisfeiler-Lehman)")
    result = await analytics.generate_graph_embeddings(
        graphs,
        algorithm='graph2vec',
        dimensions=128,
        wl_iterations=5
    )
    print(f"   Embedded {result.num_graphs} graphs")
    print(f"   Dimensions: {result.embedding_dim}")
    print(f"   Execution time: {result.execution_time_ms:.2f}ms")

    # Example 2: NetLSD (wave kernel)
    print("\n2. NetLSD (wave kernel signature)")
    result = await analytics.generate_graph_embeddings(
        graphs,
        algorithm='netlsd'
    )
    print(f"   Generated signatures for {result.num_graphs} graphs")

    # Example 3: Feather-G
    print("\n3. Feather-G (feature-based)")
    result = await analytics.generate_graph_embeddings(
        graphs,
        algorithm='feather_g',
        dimensions=128
    )
    print(f"   Feature-based embeddings for {result.num_graphs} graphs")

    # List all graph embedding algorithms
    print("\n10 Graph embedding algorithms available:")
    graph_algos = KarateClubAlgorithmRegistry.get_algorithms_by_category('graph_embedding')
    for name, info in list(graph_algos.items())[:5]:
        print(f"   - {name}: {info['name']}")


async def example_graph_metrics():
    """Example: Graph metrics computation"""
    print("\n" + "=" * 80)
    print("Graph Metrics Examples")
    print("=" * 80)

    # Create sample graph
    graph = nx.karate_club_graph()

    # Initialize analytics
    analytics = KarateClubAnalytics()

    # Compute graph metrics
    print("\nGraph-level metrics:")
    metrics = await analytics.compute_graph_metrics(graph)
    print(f"   Nodes: {metrics.num_nodes}")
    print(f"   Edges: {metrics.num_edges}")
    print(f"   Density: {metrics.density:.3f}")
    print(f"   Avg clustering: {metrics.avg_clustering:.3f}")
    print(f"   Is connected: {metrics.is_connected}")
    print(f"   Components: {metrics.num_components}")
    if metrics.diameter:
        print(f"   Diameter: {metrics.diameter}")
    if metrics.avg_path_length:
        print(f"   Avg path length: {metrics.avg_path_length:.2f}")
    if metrics.assortativity:
        print(f"   Assortativity: {metrics.assortativity:.3f}")

    # Compute node metrics
    print("\nNode-level metrics (for node 0):")
    node_metrics = await analytics.compute_node_metrics(graph, 0)
    print(f"   Degree centrality: {node_metrics.degree_centrality:.3f}")
    print(f"   Betweenness centrality: {node_metrics.betweenness_centrality:.3f}")
    print(f"   Eigenvector centrality: {node_metrics.eigenvector_centrality:.3f}")
    print(f"   Closeness centrality: {node_metrics.closeness_centrality:.3f}")
    print(f"   PageRank: {node_metrics.pagerank:.3f}")
    print(f"   Clustering coefficient: {node_metrics.clustering_coefficient:.3f}")
    print(f"   Degree: {node_metrics.degree}")


async def example_structure_analysis():
    """Example: Complete structure analysis"""
    print("\n" + "=" * 80)
    print("Complete Structure Analysis")
    print("=" * 80)

    # Create sample graph
    graph = nx.karate_club_graph()

    # Initialize analytics
    analytics = KarateClubAnalytics()

    # Perform complete analysis
    print("\nRunning comprehensive structure analysis...")
    analysis = await analytics.analyze_graph_structure(graph)

    print(f"\nAnalysis completed in {analysis.execution_time_ms:.2f}ms")

    print(f"\nCommunities: {analysis.communities.num_communities}")
    print(f"   Modularity: {analysis.communities.modularity:.3f}")

    print(f"\nGraph Metrics:")
    print(f"   Nodes: {analysis.metrics.num_nodes}")
    print(f"   Density: {analysis.metrics.density:.3f}")
    print(f"   Clustering: {analysis.metrics.avg_clustering:.3f}")

    print(f"\nTop 5 nodes by PageRank:")
    pagerank = analysis.centrality['pagerank']
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, score in top_nodes:
        print(f"   Node {node}: {score:.3f}")

    if analysis.roles:
        print(f"\nStructural roles detected: {len(analysis.roles)}")


async def example_graph_comparison():
    """Example: Graph comparison"""
    print("\n" + "=" * 80)
    print("Graph Comparison Examples")
    print("=" * 80)

    # Create multiple graphs
    graphs = [
        nx.karate_club_graph(),
        nx.erdos_renyi_graph(34, 0.1),
        nx.barabasi_albert_graph(34, 3)
    ]

    # Initialize analytics
    analytics = KarateClubAnalytics()

    # Compare using embeddings
    print("\n1. Comparison using embeddings:")
    comparison = await analytics.compare_graphs(graphs, method='embeddings')

    print(f"   Most similar pairs:")
    for graph1, graph2, sim in comparison.most_similar[:3]:
        print(f"      Graph {graph1} <-> Graph {graph2}: {sim:.3f}")

    # Compare using metrics
    print("\n2. Comparison using metrics:")
    comparison = await analytics.compare_graphs(graphs, method='metrics')

    print(f"   Most similar pairs:")
    for graph1, graph2, sim in comparison.most_similar[:3]:
        print(f"      Graph {graph1} <-> Graph {graph2}: {sim:.3f}")


async def example_retrieval():
    """Example: Embedding-based retrieval"""
    print("\n" + "=" * 80)
    print("Embedding-based Retrieval Examples")
    print("=" * 80)

    # Create sample graph
    graph = nx.karate_club_graph()

    # Initialize analytics and retrieval
    analytics = KarateClubAnalytics()
    retrieval = KarateClubRetrieval(analytics)

    # Generate embeddings
    print("\nGenerating embeddings for knowledge graph...")
    index = await retrieval.generate_embeddings_for_kg(
        graph,
        index_name='karate_club',
        algorithm='node2vec',
        dimensions=128
    )
    print(f"   Generated embeddings for {len(index.embeddings)} nodes")
    print(f"   Embedding dimension: {index.embedding_dim}")

    # Retrieve similar nodes
    query_node = str(list(graph.nodes())[0])
    print(f"\nFinding nodes similar to '{query_node}':")

    similar_nodes = await retrieval.retrieve_similar_nodes(
        query_node,
        index_name='karate_club',
        top_k=5
    )

    for node in similar_nodes:
        print(f"   {node.node}: similarity = {node.similarity:.3f}")

    # Hybrid retrieval
    print("\nHybrid retrieval (embeddings + keywords):")
    result = await retrieval.hybrid_retrieval(
        query='person',
        graph=graph,
        index_name='karate_club',
        alpha=0.5,
        top_k=5
    )

    print(f"   Query: '{result.query}'")
    print(f"   Alpha: {result.alpha}")
    print(f"   Results: {len(result.combined_results)} nodes")

    for node in result.combined_results[:3]:
        print(f"      {node.node}: {node.similarity:.3f}")


async def example_workflow_analysis():
    """Example: Workflow analysis"""
    print("\n" + "=" * 80)
    print("Workflow Analysis Examples")
    print("=" * 80)

    # Initialize workflow integration
    workflow = KarateClubWorkflowIntegration(None)

    # Analyze workflow execution
    print("\n1. Workflow execution analysis:")

    workflow_data = {
        'workflow_id': 'example_workflow',
        'tasks': [
            {'id': 'task1', 'type': 'data_ingestion', 'agent': 'agent1', 'duration': 10, 'status': 'completed'},
            {'id': 'task2', 'type': 'processing', 'agent': 'agent2', 'duration': 15, 'status': 'completed'},
            {'id': 'task3', 'type': 'analysis', 'agent': 'agent3', 'duration': 20, 'status': 'completed'},
            {'id': 'task4', 'type': 'output', 'agent': 'agent1', 'duration': 5, 'status': 'completed'},
        ],
        'dependencies': [
            {'source': 'task1', 'target': 'task2'},
            {'source': 'task2', 'target': 'task3'},
            {'source': 'task3', 'target': 'task4'},
        ]
    }

    analysis = await workflow.analyze_workflow_execution(workflow_data)

    print(f"   Workflow: {analysis.workflow_id}")
    print(f"   Agent communities: {analysis.agent_communities.num_communities}")
    print(f"   Critical path: {len(analysis.critical_path)} tasks")
    print(f"   Bottlenecks: {len(analysis.bottlenecks)}")
    print(f"   Insights:")
    for insight in analysis.insights[:3]:
        print(f"      - {insight}")

    # Analyze team performance
    print("\n2. Team performance analysis:")

    team_data = {
        'team_id': 'example_team',
        'members': [
            {'id': 'alice', 'name': 'Alice', 'role': 'Developer', 'contributions': 15},
            {'id': 'bob', 'name': 'Bob', 'role': 'Designer', 'contributions': 12},
            {'id': 'charlie', 'name': 'Charlie', 'role': 'Developer', 'contributions': 10},
            {'id': 'diana', 'name': 'Diana', 'role': 'QA', 'contributions': 8},
        ],
        'collaborations': [
            {'member1': 'alice', 'member2': 'bob', 'frequency': 8},
            {'member1': 'alice', 'member2': 'charlie', 'frequency': 12},
            {'member1': 'bob', 'member2': 'diana', 'frequency': 5},
        ]
    }

    analysis = await workflow.analyze_team_performance(team_data)

    print(f"   Team: {analysis.team_id}")
    print(f"   Sub-communities: {analysis.sub_communities.num_communities}")
    print(f"   Key contributors:")
    for contributor in analysis.key_contributors[:3]:
        print(f"      {contributor['name']} ({contributor['role']}): {contributor['score']:.3f}")
    print(f"   Recommendations:")
    for rec in analysis.recommendations[:2]:
        print(f"      - {rec}")


async def example_knowledge_graph_analysis():
    """Example: Knowledge graph analysis"""
    print("\n" + "=" * 80)
    print("Knowledge Graph Analysis Examples")
    print("=" * 80)

    # Create knowledge graph
    graph = nx.karate_club_graph()
    graph.graph['id'] = 'example_kg'

    # Add metadata
    for node in graph.nodes():
        graph.nodes[node]['type'] = 'concept'
        graph.nodes[node]['importance'] = np.random.rand()

    # Initialize workflow with analytics
    analytics = KarateClubAnalytics()
    workflow = KarateClubWorkflowIntegration(None, analytics)

    # Analyze knowledge graph
    print("\nAnalyzing knowledge graph structure...")
    analysis = await workflow.analyze_knowledge_graph(
        graph,
        analysis_depth='standard'
    )

    print(f"\nKnowledge Domains: {analysis.knowledge_domains.num_communities}")
    print(f"   Modularity: {analysis.knowledge_domains.modularity:.3f}")

    print(f"\nKey Concepts (top 5):")
    for concept in analysis.key_concepts[:5]:
        print(f"   {concept['concept']}: {concept['score']:.3f}")

    print(f"\nTopic Density:")
    for domain, density in list(analysis.topic_density.items())[:3]:
        print(f"   Domain {domain}: {density:.3f}")

    print(f"\nStructural Insights:")
    for insight in analysis.structural_insights.get('key_findings', [])[:3]:
        print(f"   - {insight}")


async def main():
    """Run all examples"""
    print("=" * 80)
    print("KarateClub Analytics - Complete Usage Examples")
    print("=" * 80)

    # Show algorithm counts
    counts = KarateClubAlgorithmRegistry.get_total_count()
    print(f"\nTotal Algorithms: {counts['total']}")
    print(f"  - Community Detection: {counts['community']}")
    print(f"  - Node Embedding: {counts['node_embedding']}")
    print(f"  - Graph Embedding: {counts['graph_embedding']}")

    # Run examples
    await example_community_detection()
    await example_node_embeddings()
    await example_graph_embeddings()
    await example_graph_metrics()
    await example_structure_analysis()
    await example_graph_comparison()
    await example_retrieval()
    await example_workflow_analysis()
    await example_knowledge_graph_analysis()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
