"""
Example: Complete Visualization Workflow

This example demonstrates all major visualization features.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from knowledge_engine.visualization import (
    GraphExplorer,
    TemporalVisualizer,
    CommunityVisualizer,
    ExportHandler,
    VisualizationOptions,
    NodeFilter,
    EdgeFilter,
    CommunityVisualizationOptions,
    TemporalVisualizationOptions
)


# Sample data
SAMPLE_TRIPLES = [
    # Social network
    {'subject': 'Alice', 'predicate': 'knows', 'object': 'Bob', 'confidence': 0.95},
    {'subject': 'Bob', 'predicate': 'knows', 'object': 'Charlie', 'confidence': 0.90},
    {'subject': 'Charlie', 'predicate': 'knows', 'object': 'David', 'confidence': 0.85},
    {'subject': 'David', 'predicate': 'knows', 'object': 'Eve', 'confidence': 0.80},
    {'subject': 'Eve', 'predicate': 'knows', 'object': 'Alice', 'confidence': 0.75},

    # Work relationships
    {'subject': 'Alice', 'predicate': 'works_with', 'object': 'Charlie', 'confidence': 0.95},
    {'subject': 'Bob', 'predicate': 'works_with', 'object': 'David', 'confidence': 0.90},
    {'subject': 'Charlie', 'predicate': 'manages', 'object': 'Eve', 'confidence': 0.85},

    # Additional connections
    {'subject': 'Alice', 'predicate': 'reports_to', 'object': 'Bob', 'confidence': 0.80},
    {'subject': 'Charlie', 'predicate': 'collaborates_with', 'object': 'David', 'confidence': 0.88},
]

SAMPLE_ENTITIES = [
    {'name': 'Alice', 'type': 'Person', 'department': 'Engineering'},
    {'name': 'Bob', 'type': 'Person', 'department': 'Management'},
    {'name': 'Charlie', 'type': 'Person', 'department': 'Engineering'},
    {'name': 'David', 'type': 'Person', 'department': 'Sales'},
    {'name': 'Eve', 'type': 'Person', 'department': 'Marketing'},
]


async def example_1_basic_graph_visualization():
    """Example 1: Basic graph visualization."""
    print("\n=== Example 1: Basic Graph Visualization ===\n")

    explorer = GraphExplorer()

    result = await explorer.visualize(
        triples=SAMPLE_TRIPLES,
        entities=SAMPLE_ENTITIES,
        options=VisualizationOptions(
            width=1200,
            height=800,
            show_labels=True,
            enable_zoom=True
        )
    )

    print(f"[OK] Graph visualization created")
    print(f"  Nodes: {result.node_count}")
    print(f"  Edges: {result.edge_count}")
    print(f"  Communities: {result.community_count}")
    print(f"  Output: {result.output_path}")
    print(f"  Generated in: {result.generation_time:.2f}s")


async def example_2_filtered_visualization():
    """Example 2: Visualization with filters."""
    print("\n=== Example 2: Filtered Visualization ===\n")

    explorer = GraphExplorer()

    # Apply filters
    node_filter = NodeFilter(
        search_query="Alice",
        min_degree=2
    )

    edge_filter = EdgeFilter(
        relationship_types=["knows", "works_with"],
        min_confidence=0.85
    )

    result = await explorer.visualize(
        triples=SAMPLE_TRIPLES,
        entities=SAMPLE_ENTITIES,
        node_filter=node_filter,
        edge_filter=edge_filter,
        options=VisualizationOptions(
            width=1200,
            height=800,
            show_labels=True
        )
    )

    print(f"[OK] Filtered visualization created")
    print(f"  Nodes after filtering: {result.node_count}")
    print(f"  Edges after filtering: {result.edge_count}")
    print(f"  Output: {result.output_path}")


async def example_3_temporal_visualization():
    """Example 3: Temporal visualization."""
    print("\n=== Example 3: Temporal Visualization ===\n")

    temporal_viz = TemporalVisualizer()

    # Create timestamps (one per day for last week)
    timestamps = [
        datetime.utcnow() - timedelta(days=i)
        for i in range(len(SAMPLE_TRIPLES) - 1, -1, -1)
    ]

    result = await temporal_viz.visualize_temporal(
        triples=SAMPLE_TRIPLES,
        timestamps=timestamps,
        options=TemporalVisualizationOptions(
            width=1200,
            height=800,
            show_timeline=True,
            enable_animation=True
        )
    )

    print(f"[OK] Temporal visualization created")
    print(f"  Snapshots: {result['snapshots']}")
    print(f"  Time span: {result['statistics'].get('time_span_hours', 0):.1f} hours")
    print(f"  Output: {result['output_path']}")


async def example_4_community_visualization():
    """Example 4: Community-based visualization."""
    print("\n=== Example 4: Community Visualization ===\n")

    community_viz = CommunityVisualizer()

    result = await community_viz.visualize_communities(
        triples=SAMPLE_TRIPLES,
        entities=SAMPLE_ENTITIES,
        options=CommunityVisualizationOptions(
            width=1200,
            height=800,
            layout_algorithm="force_community",
            show_inter_community_edges=True
        )
    )

    print(f"[OK] Community visualization created")
    print(f"  Communities detected: {result['num_communities']}")
    print(f"  Output: {result['output_path']}")


async def example_5_export_visualizations():
    """Example 5: Export in multiple formats."""
    print("\n=== Example 5: Export Visualizations ===\n")

    explorer = GraphExplorer()
    exporter = ExportHandler()

    # First create a visualization
    graph = explorer._build_graph(SAMPLE_TRIPLES)
    communities = await explorer._detect_communities(graph)
    centrality = await explorer._compute_centrality(graph)

    graph_data = explorer._prepare_graph_data(
        graph, SAMPLE_TRIPLES, communities, centrality,
        VisualizationOptions()
    )

    # Export in different formats
    formats_to_export = ['svg', 'html', 'json']

    for fmt in formats_to_export:
        output_path = f"data/visualizations/exports/example.{fmt}"

        if fmt == 'svg':
            result = await exporter.export_svg(
                graph_data, output_path, width=1200, height=800
            )
        elif fmt == 'html':
            result = await exporter.export_html(
                graph_data, output_path, width=1200, height=800
            )
        elif fmt == 'json':
            result = await exporter.export_json(
                graph_data, output_path, pretty=True
            )

        print(f"[OK] Exported {fmt.upper()}: {result}")


async def example_6_comparison_view():
    """Example 6: Before/after comparison."""
    print("\n=== Example 6: Comparison View ===\n")

    temporal_viz = TemporalVisualizer()

    # Split data into "before" and "after"
    before_triples = SAMPLE_TRIPLES[:5]
    after_triples = SAMPLE_TRIPLES

    result = await temporal_viz.create_comparison_view(
        triples_before=before_triples,
        triples_after=after_triples
    )

    print(f"[OK] Comparison view created")
    print(f"  Added nodes: {result['added_nodes']}")
    print(f"  Removed nodes: {result['removed_nodes']}")
    print(f"  Added edges: {result['added_edges']}")
    print(f"  Removed edges: {result['removed_edges']}")
    print(f"  Output: {result['output_path']}")


async def example_7_subgraph_extraction():
    """Example 7: Extract subgraph around a node."""
    print("\n=== Example 7: Subgraph Extraction ===\n")

    from knowledge_engine.visualization.api import extract_subgraph

    # Create subgraph request
    request_data = {
        'triples': SAMPLE_TRIPLES,
        'center_node': 'Alice',
        'radius': 2,
        'min_degree': 1
    }

    result = await extract_subgraph(request_data)

    print(f"[OK] Subgraph extracted")
    print(f"  Center node: {result['center_node']}")
    print(f"  Radius: {result['radius']}")
    print(f"  Nodes in subgraph: {result['node_count']}")
    print(f"  Edges in subgraph: {result['edge_count']}")


async def example_8_graph_statistics():
    """Example 8: Compute comprehensive statistics."""
    print("\n=== Example 8: Graph Statistics ===\n")

    from knowledge_engine.visualization.api import get_graph_statistics

    result = await get_graph_statistics(SAMPLE_TRIPLES)

    print(f"[OK] Statistics computed")
    print(f"  Nodes: {result.node_count}")
    print(f"  Edges: {result.edge_count}")
    print(f"  Communities: {result.communities}")
    print(f"  Density: {result.density:.4f}")
    print(f"  Connected: {result.is_connected}")
    print(f"  Avg clustering: {result.avg_clustering:.4f}")
    print(f"  Diameter: {result.diameter}")


async def example_9_generate_embedding_url():
    """Example 9: Generate embeddable URL."""
    print("\n=== Example 9: Embedding URL ===\n")

    explorer = GraphExplorer()
    exporter = ExportHandler()

    # Create graph data
    graph = explorer._build_graph(SAMPLE_TRIPLES)
    communities = await explorer._detect_communities(graph)
    centrality = await explorer._compute_centrality(graph)

    graph_data = explorer._prepare_graph_data(
        graph, SAMPLE_TRIPLES, communities, centrality,
        VisualizationOptions()
    )

    # Generate embedding URL
    embed_url = exporter.generate_embedding_url(
        graph_data=graph_data,
        base_url="https://example.com/visualizations",
        config={
            "width": 800,
            "height": 600,
            "show_labels": True
        }
    )

    print(f"[OK] Embedding URL generated")
    print(f"  URL: {embed_url[:100]}...")
    print(f"  Use this URL to embed in external sites")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Knowledge Graph Visualization - Complete Examples")
    print("=" * 60)

    try:
        await example_1_basic_graph_visualization()
        await example_2_filtered_visualization()
        await example_3_temporal_visualization()
        await example_4_community_visualization()
        await example_5_export_visualizations()
        await example_6_comparison_view()
        await example_7_subgraph_extraction()
        await example_8_graph_statistics()
        await example_9_generate_embedding_url()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
