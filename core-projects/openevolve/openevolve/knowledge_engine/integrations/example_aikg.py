"""
AI-Knowledge-Graph Integration Usage Examples

This module provides comprehensive examples of using the AIKG integration
for entity standardization, relationship inference, and visualization.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.engine import KnowledgeEngine
from knowledge_engine.integrations.aikg_standardization import Entity, Triple
from knowledge_engine.integrations.aikg_integration import AIKGIntegration


async def example_1_complete_pipeline():
    """
    Example 1: Complete AIKG pipeline processing.

    This demonstrates the full pipeline:
    1. Text processing
    2. Entity standardization
    3. Relationship inference
    4. Visualization generation
    """
    print("\n" + "="*80)
    print("Example 1: Complete AIKG Pipeline")
    print("="*80)

    # Initialize Knowledge Engine
    engine = KnowledgeEngine()

    # Sample text
    text = """
    Python is a high-level programming language used for web development,
    data science, and artificial intelligence. Django is a Python web framework.
    Machine learning is a subset of artificial intelligence that uses Python.
    JavaScript is also used for web development. TensorFlow is a machine learning library.
    """

    print(f"\nProcessing text: {text[:100]}...")

    try:
        # Process with AIKG
        result = await engine.process_with_aikg(
            text=text,
            enable_standardization=True,
            enable_inference=True,
            generate_visualization=True,
            output_path="example_graph_1.html"
        )

        # Print results
        print(f"\nResults:")
        print(f"  Original entities: {result.original_triple_count}")
        print(f"  Standardized entities: {len(result.standardized_entities)}")
        print(f"  Entity reduction: {result.entity_reduction_rate:.1f}%")
        print(f"  Original triples: {result.original_triple_count}")
        print(f"  Inferred triples: {result.inferred_triple_count}")
        print(f"  Total triples: {result.total_triple_count}")
        print(f"  Inference rate: {result.inferred_triple_count / result.original_triple_count * 100:.1f}%")
        print(f"  Visualization: {result.visualization_path}")

        # Print detailed summary
        summary = result.get_summary()
        print(f"\nDetailed Summary:")
        print(f"  Communities detected: {summary['visualization']['communities']}")
        print(f"  Graph density: {summary['inference'].get('avg_confidence', 0):.3f}")

    except Exception as e:
        print(f"Error: {e}")


async def example_2_entity_standardization():
    """
    Example 2: Entity standardization only.

    This demonstrates entity deduplication and canonicalization.
    """
    print("\n" + "="*80)
    print("Example 2: Entity Standardization")
    print("="*80)

    # Create sample entities with duplicates
    entities = [
        Entity("Python"),
        Entity("python"),
        Entity("PYTHON"),
        Entity("Machine Learning"),
        Entity("machine learning"),
        Entity("ML"),
        Entity("JavaScript"),
        Entity("javascript"),
        Entity("JS")
    ]

    # Create sample triples
    triples = [
        Triple("Python", "used_for", "Web Development"),
        Triple("python", "related_to", "Django"),
        Triple("PYTHON", "related_to", "Data Science"),
        Triple("Machine Learning", "subset_of", "Artificial Intelligence"),
        Triple("machine learning", "used_for", "Data Analysis"),
        Triple("JavaScript", "used_for", "Web Development"),
        Triple("javascript", "related_to", "React")
    ]

    print(f"\nInput: {len(entities)} entities, {len(triples)} triples")

    # Initialize Knowledge Engine
    engine = KnowledgeEngine()

    try:
        # Standardize entities
        result = await engine.standardize_entities_with_aikg(entities, triples)

        print(f"\nStandardization Results:")
        print(f"  Canonical entities: {len(result.canonical_entities)}")
        print(f"  Reduction: {result.statistics['duplicates_found']} duplicates found")
        print(f"  Self-references removed: {result.removed_self_refs}")

        print(f"\nCanonical Entities:")
        for entity in result.canonical_entities:
            print(f"  - {entity.name}")

        print(f"\nVariant Mappings:")
        for canonical, variants in result.variant_mappings.items():
            print(f"  {canonical} <- {', '.join(variants)}")

    except Exception as e:
        print(f"Error: {e}")


async def example_3_relationship_inference():
    """
    Example 3: Relationship inference only.

    This demonstrates inferring new relationships from existing ones.
    """
    print("\n" + "="*80)
    print("Example 3: Relationship Inference")
    print("="*80)

    # Create sample entities
    entities = [
        Entity("Python"),
        Entity("Django"),
        Entity("Flask"),
        Entity("Web Development"),
        Entity("Machine Learning"),
        Entity("TensorFlow"),
        Entity("PyTorch")
    ]

    # Create sample triples
    triples = [
        Triple("Python", "used_for", "Web Development"),
        Triple("Django", "framework_of", "Python"),
        Triple("Flask", "framework_of", "Python"),
        Triple("Python", "used_for", "Machine Learning"),
        Triple("TensorFlow", "library_of", "Python"),
        Triple("PyTorch", "library_of", "Python")
    ]

    print(f"\nInput: {len(triples)} triples")

    # Initialize Knowledge Engine
    engine = KnowledgeEngine()

    try:
        # Infer relationships
        result = await engine.infer_relationships_with_aikg(triples, entities)

        print(f"\nInference Results:")
        print(f"  Original triples: {len(result.original_triples)}")
        print(f"  Inferred triples: {len(result.inferred_triples)}")
        print(f"  Total triples: {len(result.all_triples)}")

        stats = result.get_statistics()
        print(f"  Average confidence: {stats['avg_confidence']:.3f}")
        print(f"  Inference methods: {', '.join(stats['inference_methods'])}")

        print(f"\nInferred Triples:")
        for triple in result.inferred_triples[:10]:  # Show first 10
            print(f"  - {triple.subject} | {triple.predicate} | {triple.object} "
                  f"(confidence: {triple.confidence:.2f}, source: {triple.source})")

    except Exception as e:
        print(f"Error: {e}")


async def example_4_visualization():
    """
    Example 4: Knowledge graph visualization.

    This demonstrates generating D3.js interactive visualizations.
    """
    print("\n" + "="*80)
    print("Example 4: Knowledge Graph Visualization")
    print("="*80)

    # Create sample entities
    entities = [
        Entity("Python"),
        Entity("JavaScript"),
        Entity("Django"),
        Entity("React"),
        Entity("Web Development"),
        Entity("Machine Learning"),
        Entity("Artificial Intelligence"),
        Entity("TensorFlow"),
        Entity("PyTorch")
    ]

    # Create sample triples
    triples = [
        Triple("Python", "used_for", "Web Development"),
        Triple("JavaScript", "used_for", "Web Development"),
        Triple("Django", "framework_of", "Python"),
        Triple("React", "library_of", "JavaScript"),
        Triple("Machine Learning", "subset_of", "Artificial Intelligence"),
        Triple("Python", "used_for", "Machine Learning"),
        Triple("TensorFlow", "library_of", "Python"),
        Triple("PyTorch", "library_of", "Python"),
        Triple("TensorFlow", "used_for", "Machine Learning"),
        Triple("PyTorch", "used_for", "Machine Learning")
    ]

    print(f"\nInput: {len(entities)} entities, {len(triples)} triples")

    # Initialize Knowledge Engine
    engine = KnowledgeEngine()

    try:
        # Generate visualization
        result = await engine.visualize_knowledge_graph(
            triples=triples,
            entities=entities,
            output_path="example_graph_4.html",
            width=1200,
            height=800
        )

        print(f"\nVisualization Results:")
        print(f"  Output path: {result.output_path}")
        print(f"  Nodes: {result.node_count}")
        print(f"  Edges: {result.edge_count}")
        print(f"  Communities: {result.community_count}")

        print(f"\nGraph Statistics:")
        stats = result.statistics
        print(f"  Average community size: {stats['avg_community_size']:.2f}")
        print(f"  Max centrality: {stats['max_centrality']:.3f}")
        print(f"  Graph density: {stats['graph_density']:.3f}")
        print(f"  Is connected: {stats['is_connected']}")

        print(f"\nOpen {result.output_path} in a web browser to view the visualization.")

    except Exception as e:
        print(f"Error: {e}")


async def example_5_direct_integration():
    """
    Example 5: Using AIKG integration directly.

    This demonstrates using the AIKG integration without going through KnowledgeEngine.
    """
    print("\n" + "="*80)
    print("Example 5: Direct AIKG Integration Usage")
    print("="*80)

    # Create AIKG integration directly
    config = {
        'standardization': {
            'enabled': True,
            'use_llm_for_entities': False
        },
        'inference': {
            'enabled': True,
            'apply_transitive': True,
            'use_llm_for_inference': False
        },
        'visualization': {
            'enabled': True,
            'output_dir': '.'
        }
    }

    aikg = AIKGIntegration(config)

    # Create sample data
    entities = [
        Entity("Python"),
        Entity("python"),
        Entity("Django"),
        Entity("Flask")
    ]

    triples = [
        Triple("Python", "used_for", "Web Development"),
        Triple("python", "related_to", "Django"),
        Triple("Django", "framework_of", "Python"),
        Triple("Flask", "framework_of", "Python")
    ]

    print(f"\nInput: {len(entities)} entities, {len(triples)} triples")

    try:
        # Process pre-extracted data
        result = await aikg.process_preextracted_data(
            entities=entities,
            triples=triples,
            enable_standardization=True,
            enable_inference=True,
            generate_visualization=True,
            output_path="example_graph_5.html"
        )

        print(f"\nResults:")
        print(f"  Entity reduction: {result.entity_reduction_rate:.1f}%")
        print(f"  Inference rate: {result.inferred_triple_count / result.original_triple_count * 100:.1f}%")
        print(f"  Visualization: {result.visualization_path}")

    except Exception as e:
        print(f"Error: {e}")


async def example_6_export():
    """
    Example 6: Export knowledge graph data.

    This demonstrates exporting graph data in various formats.
    """
    print("\n" + "="*80)
    print("Example 6: Export Knowledge Graph Data")
    print("="*80)

    # Create sample triples
    triples = [
        Triple("Python", "used_for", "Web Development"),
        Triple("Django", "framework_of", "Python"),
        Triple("JavaScript", "used_for", "Web Development")
    ]

    # Initialize Knowledge Engine
    engine = KnowledgeEngine()

    try:
        # Export as JSON
        json_data = await engine.export_knowledge_graph(triples, format="json")
        print(f"\nJSON Export: {len(json_data)} characters")

        # Export as CSV
        csv_data = await engine.export_knowledge_graph(triples, format="csv")
        print(f"CSV Export: {len(csv_data.split(chr(10)))} lines")

        # Print sample CSV
        print(f"\nCSV Preview:")
        print(csv_data[:200])

    except Exception as e:
        print(f"Error: {e}")


async def example_7_variant_mappings():
    """
    Example 7: Working with variant mappings.

    This demonstrates accessing and using entity variant mappings.
    """
    print("\n" + "="*80)
    print("Example 7: Entity Variant Mappings")
    print("="*80)

    # Create sample entities with variants
    entities = [
        Entity("Python"),
        Entity("python"),
        Entity("PYTHON"),
        Entity("Machine Learning"),
        Entity("machine learning"),
        Entity("ML")
    ]

    triples = [
        Triple("Python", "used_for", "Web Development"),
        Triple("python", "related_to", "Django")
    ]

    # Initialize Knowledge Engine
    engine = KnowledgeEngine()

    try:
        # Standardize entities
        result = await engine.standardize_entities_with_aikg(entities, triples)

        # Get variant mappings
        mappings = engine.get_aikg_variant_mappings()

        print(f"\nVariant Mappings:")
        for canonical, variants in mappings.items():
            print(f"  {canonical}")
            for variant in variants:
                print(f"    <- {variant}")

    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AI-Knowledge-Graph Integration Examples")
    print("="*80)

    # Run examples
    await example_1_complete_pipeline()
    await example_2_entity_standardization()
    await example_3_relationship_inference()
    await example_4_visualization()
    await example_5_direct_integration()
    await example_6_export()
    await example_7_variant_mappings()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
