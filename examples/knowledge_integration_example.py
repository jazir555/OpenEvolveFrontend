"""
BubbleLabs Knowledge Integration - Complete Example

This example demonstrates how to use all components of the BubbleLabs
Knowledge Engine integration together in a real-world workflow.

Workflow:
1. Initialize the knowledge engine
2. Query multiple knowledge sources
3. Extract knowledge from documents
4. Build and visualize knowledge graph
5. Analyze statistics

Author: OpenEvolve Integration Team
Created: 2026-01-03
"""

import asyncio
import json
from pathlib import Path
from bubblelabs_knowledge_integration import (
    BubbleLabsKnowledgeUI,
    KnowledgeGraphVisualizer,
    KnowledgeQueryInterface,
    KnowledgeExtractionWorkflow
)
from knowledge_engine.engine import KnowledgeEngine


async def example_1_basic_query():
    """
    Example 1: Basic knowledge query across multiple sources
    """
    print("\n" + "="*60)
    print("Example 1: Multi-Source Knowledge Query")
    print("="*60)

    # Initialize UI
    ui = BubbleLabsKnowledgeUI()
    ui.initialize_engine()

    # Execute unified query
    results = await ui.query_interface.unified_query(
        query="How does MCTS improve adversarial validation in decomposition workflows?",
        sources=['bedrock', 'graphiti', 'local'],
        bedrock_kb_id="YOUR_KB_ID",
        index_path="knowledge_index"
    )

    # Display results
    print("\n--- Query Results ---")
    for source, data in results['sources'].items():
        print(f"\n{source.upper()}:")
        if isinstance(data, dict) and 'error' not in data:
            if 'merged_context' in data:
                print(f"  Answer: {data['merged_context'][:200]}...")
            if 'nodes' in data:
                print(f"  Entities: {len(data['nodes'])}")
            if 'edges' in data:
                print(f"  Relationships: {len(data['edges'])}")
        elif isinstance(data, list):
            print(f"  Files found: {len(data)}")

    return results


async def example_2_knowledge_extraction():
    """
    Example 2: Extract knowledge from a research paper
    """
    print("\n" + "="*60)
    print("Example 2: Knowledge Extraction from Document")
    print("="*60)

    # Initialize
    ui = BubbleLabsKnowledgeUI()
    ui.initialize_engine()

    # Extract from document
    results = await ui.extraction_workflow.extract_from_document(
        document_path_or_url="https://arxiv.org/pdf/2301.07041",
        extraction_config={
            "extract_entities": True,
            "extract_relationships": True,
            "min_confidence": 0.7
        }
    )

    if 'error' not in results:
        print(f"\nExtracted {results['statistics']['total_entities']} entities")
        print(f"Extracted {results['statistics']['total_relationships']} relationships")

        # Display sample entities
        print("\n--- Sample Entities ---")
        for entity in results['entities'][:5]:
            print(f"  - {entity['name']} ({entity.get('type', 'Unknown')})")

        # Display sample relationships
        print("\n--- Sample Relationships ---")
        for rel in results['relationships'][:5]:
            print(f"  - {rel['source']} -> {rel['relation']} -> {rel['target']}")

        return results
    else:
        print(f"Extraction failed: {results['error']}")
        return None


async def example_3_graph_visualization():
    """
    Example 3: Build and visualize knowledge graph
    """
    print("\n" + "="*60)
    print("Example 3: Knowledge Graph Visualization")
    print("="*60)

    # Create visualizer
    visualizer = KnowledgeGraphVisualizer()

    # Sample data (in real use, this comes from extraction)
    entities = [
        {"name": "MCTS", "type": "algorithm", "attributes": {"confidence": 0.95}},
        {"name": "MDAP", "type": "framework", "attributes": {}},
        {"name": "Adversarial", "type": "technique", "attributes": {"confidence": 0.9}},
        {"name": "Decomposition", "type": "workflow", "attributes": {}},
    ]

    relationships = [
        {"source": "MCTS", "relation": "optimizes", "target": "MDAP", "attributes": {"confidence": 0.95}},
        {"source": "MCTS", "relation": "improves", "target": "Adversarial", "attributes": {"confidence": 0.9}},
        {"source": "MDAP", "relation": "uses", "target": "Decomposition", "attributes": {}},
        {"source": "Adversarial", "relation": "validates", "target": "Decomposition", "attributes": {}},
    ]

    # Build graph
    visualizer.build_graph_from_data(entities, relationships)

    # Get statistics
    stats = visualizer.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Connected: {stats['is_connected']}")

    # Create visualization
    fig = visualizer.create_interactive_plot(
        layout='spring',
        node_size_multiplier=1.5,
        show_labels=True
    )

    # In Streamlit, you would do:
    # st.plotly_chart(fig, use_container_width=True)

    print("\nVisualization created (use Streamlit to display)")

    # Find shortest path
    path = visualizer.find_shortest_path("MCTS", "Decomposition")
    if path:
        print(f"\nShortest path from MCTS to Decomposition:")
        print(f"  {' -> '.join(path)}")

    # Get neighbors
    neighbors = visualizer.get_entity_neighbors("MCTS")
    print(f"\nMCTS neighbors:")
    print(f"  Predecessors: {neighbors['predecessors']}")
    print(f"  Successors: {neighbors['successors']}")

    return visualizer


async def example_4_complete_workflow():
    """
    Example 4: Complete end-to-end workflow
    """
    print("\n" + "="*60)
    print("Example 4: Complete Knowledge Workflow")
    print("="*60)

    # Initialize
    ui = BubbleLabsKnowledgeUI()
    ui.initialize_engine()

    # Step 1: Query existing knowledge
    print("\n[Step 1] Querying knowledge bases...")
    query_results = await ui.query_interface.unified_query(
        query="MCTS optimization strategies for adversarial validation",
        sources=['bedrock', 'graphiti'],
        bedrock_kb_id="YOUR_KB_ID"
    )

    print(f"  Found results from {len(query_results['sources'])} sources")

    # Step 2: Extract new knowledge from document
    print("\n[Step 2] Extracting knowledge from document...")
    extraction_results = await ui.extraction_workflow.extract_from_document(
        document_path_or_url="https://arxiv.org/pdf/2301.07041"
    )

    if extraction_results and 'error' not in extraction_results:
        print(f"  Extracted {extraction_results['statistics']['total_entities']} entities")

        # Step 3: Build knowledge graph
        print("\n[Step 3] Building knowledge graph...")
        ui.visualizer.build_graph_from_data(
            extraction_results['entities'],
            extraction_results['relationships']
        )

        stats = ui.visualizer.get_graph_statistics()
        print(f"  Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

        # Step 4: Visualize
        print("\n[Step 4] Creating visualization...")
        fig = ui.visualizer.create_interactive_plot(layout='spring')
        print("  Visualization ready for display")

        # Step 5: Export results
        print("\n[Step 5] Exporting results...")

        # Save to JSON
        output_dir = Path("knowledge_output")
        output_dir.mkdir(exist_ok=True)

        with open(output_dir / "extracted_knowledge.json", "w") as f:
            json.dump({
                'entities': extraction_results['entities'],
                'relationships': extraction_results['relationships'],
                'statistics': extraction_results['statistics']
            }, f, indent=2)

        print(f"  Saved to {output_dir / 'extracted_knowledge.json'}")

        print("\n[OK] Complete workflow finished successfully!")
        return {
            'query_results': query_results,
            'extraction_results': extraction_results,
            'graph_stats': stats
        }
    else:
        print("[FAIL] Workflow failed at extraction step")
        return None


async def example_5_bubblelab_ui():
    """
    Example 5: Using the BubbleLab UI (TypeScript)
    """
    print("\n" + "="*60)
    print("Example 5: BubbleLab UI Usage")
    print("="*60)

    print("""
To use the BubbleLab UI:

1. Build the BubbleLab components:
   npm install
   npm run build

2. Load the OpenEvolve components in your BubbleLab host app
   (package: @openevolve/bubblelab-components).

3. Point the host app at the OpenEvolve API server.

4. Use the four main tabs:

   🔍 Query Knowledge Tab:
   - Enter your query in the text box
   - Select knowledge sources (Bedrock, Graphiti, Local)
   - Configure source-specific settings
   - Click "Execute Query"
   - Review results from each source
   - Explore query history

   📊 Knowledge Graph Tab:
   - View interactive network graph
   - Select layout algorithm
   - Filter entities by type and confidence
   - Click nodes to see details
   - Explore neighbors and paths
   - Review graph statistics

   📄 Extract Knowledge Tab:
   - Choose input source (URL, File, Text)
   - Upload document or enter text
   - Click "Extract Knowledge"
   - Review extracted entities and relationships
   - Export results

   📈 Statistics Tab:
   - View entity/relationship counts
   - Explore type distributions
   - Review query statistics
   - Analyze graph metrics

5. Use the sidebar for additional settings

For more details, see the documentation.
    """)


async def example_6_batch_processing():
    """
    Example 6: Batch knowledge extraction from multiple documents
    """
    print("\n" + "="*60)
    print("Example 6: Batch Knowledge Extraction")
    print("="*60)

    # Initialize
    ui = BubbleLabsKnowledgeUI()
    ui.initialize_engine()

    # List of documents to process
    documents = [
        "https://arxiv.org/pdf/2301.07041",
        "https://arxiv.org/pdf/2301.07042",
        # Add more documents...
    ]

    all_entities = []
    all_relationships = []

    print(f"\nProcessing {len(documents)} documents...")

    for i, doc_url in enumerate(documents, 1):
        print(f"\n[{i}/{len(documents)}] Processing: {doc_url}")

        results = await ui.extraction_workflow.extract_from_document(doc_url)

        if results and 'error' not in results:
            all_entities.extend(results['entities'])
            all_relationships.extend(results['relationships'])
            print(f"  [OK] Extracted {len(results['entities'])} entities")
        else:
            print(f"  [FAIL] Failed: {results.get('error', 'Unknown error')}")

    # Build combined knowledge graph
    print(f"\nBuilding combined knowledge graph...")
    ui.visualizer.build_graph_from_data(all_entities, all_relationships)

    stats = ui.visualizer.get_graph_statistics()
    print(f"Total entities: {len(all_entities)}")
    print(f"Total relationships: {len(all_relationships)}")
    print(f"Graph nodes: {stats['total_nodes']}")
    print(f"Graph edges: {stats['total_edges']}")

    # Save combined results
    output_dir = Path("knowledge_output")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "batch_extraction.json", "w") as f:
        json.dump({
            'entities': all_entities,
            'relationships': all_relationships,
            'statistics': {
                'total_entities': len(all_entities),
                'total_relationships': len(all_relationships),
                'documents_processed': len(documents)
            }
        }, f, indent=2)

    print(f"\n[OK] Batch processing complete!")
    print(f"Results saved to: {output_dir / 'batch_extraction.json'}")


async def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("  BubbleLabs Knowledge Integration - Complete Examples")
    print("="*70)

    examples = [
        ("Basic Query", example_1_basic_query),
        ("Knowledge Extraction", example_2_knowledge_extraction),
        ("Graph Visualization", example_3_graph_visualization),
        ("Complete Workflow", example_4_complete_workflow),
        ("BubbleLab UI", example_5_bubblelab_ui),
        ("Batch Processing", example_6_batch_processing),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning examples...")
    print("Note: Some examples require valid API keys and configuration")

    # Uncomment the examples you want to run:

    # await example_1_basic_query()
    # await example_2_knowledge_extraction()
    # await example_3_graph_visualization()
    # await example_4_complete_workflow()
    await example_5_bubblelab_ui()  # This one just prints instructions
    # await example_6_batch_processing()

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70)


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())

    # Or run individual example:
    # asyncio.run(example_1_basic_query())
    # asyncio.run(example_2_knowledge_extraction())
    # etc.
