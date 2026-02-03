"""
KG-Gen Pipeline Usage Examples

This file provides comprehensive examples of using the kg-gen pipeline integration
for knowledge graph extraction, processing, and Neo4j upload.
"""

import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example 1: Simple Knowledge Graph Extraction
async def example_1_simple_extraction():
    """
    Extract a knowledge graph from a simple text.
    """
    print("\n=== Example 1: Simple Knowledge Graph Extraction ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    # Initialize the knowledge engine
    engine = KnowledgeEngine()

    # Sample text
    text = """
    Python is a high-level programming language created by Guido van Rossum in 1991.
    Python emphasizes code readability with its notable use of significant whitespace.
    It is widely used for web development, data science, machine learning, and automation.
    The Python Software Foundation manages the language development.

    Machine learning is a subset of artificial intelligence that focuses on
    building systems that can learn from data. Python is the most popular
    programming language for machine learning due to libraries like TensorFlow
    and PyTorch.
    """

    # Extract knowledge graph
    graph = await engine.extract_knowledge_graph(
        text=text,
        context="Programming languages and machine learning",
        upload_to_neo4j=False  # Set to True to upload to Neo4j
    )

    # Display results
    print(f"Extracted {len(graph.entities)} entities:")
    for entity in graph.entities[:10]:  # Show first 10
        print(f"  - {entity}")

    print(f"\nExtracted {len(graph.relationships)} relationships:")
    for subject, predicate, obj in graph.relationships[:10]:  # Show first 10
        print(f"  - {subject} --[{predicate}]--> {obj}")

    print(f"\nMetadata: {graph.metadata}")

    # Cleanup
    await engine.cleanup_kggen_pipeline()


# Example 2: Large Document Processing
async def example_2_large_document():
    """
    Process a large document with parallel chunking.
    """
    print("\n=== Example 2: Large Document Processing ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    engine = KnowledgeEngine()

    # Load a large document
    document_path = "path/to/large_document.txt"

    # For this example, create a sample large document
    sample_text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science,
    and artificial intelligence concerned with the interactions between computers and
    human language, in particular how to program computers to process and analyze
    large amounts of natural language data.

    Machine learning is a field of inquiry devoted to understanding and building
    methods that 'learn', that is, methods that leverage data to improve performance
    on some set of tasks. It is seen as a part of artificial intelligence.

    Deep learning is part of a broader family of machine learning methods based on
    artificial neural networks with representation learning. Learning can be supervised,
    semi-supervised or unsupervised.

    """ * 100  # Repeat to make it large

    # Extract knowledge graph with chunking
    graph = await engine.extract_from_document(
        document_path=document_path if Path(document_path).exists() else None,
        chunk_size=5000  # Process in chunks of 5000 characters
    )

    print(f"Processed large document:")
    print(f"  - Total entities: {len(graph.entities)}")
    print(f"  - Total relationships: {len(graph.relationships)}")
    print(f"  - Entity clusters: {len(graph.entity_clusters)}")

    await engine.cleanup_kggen_pipeline()


# Example 3: Batch Processing
async def example_3_batch_processing():
    """
    Process multiple texts in batch.
    """
    print("\n=== Example 3: Batch Processing ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    engine = KnowledgeEngine()

    # Multiple texts to process
    texts = [
        "Python is a programming language used for web development.",
        "JavaScript is primarily used for frontend web development.",
        "Java is widely used in enterprise applications.",
        "C++ is used in system programming and game development.",
        "Go is designed for simplicity and concurrency.",
    ]

    # Process in batch
    graphs = await engine.extract_batch_knowledge_graphs(texts)

    print(f"Processed {len(graphs)} texts in batch:")
    for i, graph in enumerate(graphs):
        print(f"\nText {i+1}:")
        print(f"  - Entities: {len(graph.entities)}")
        print(f"  - Relationships: {len(graph.relationships)}")

    await engine.cleanup_kggen_pipeline()


# Example 4: Using Custom Context
async def example_4_custom_context():
    """
    Extract knowledge graph with custom context.
    """
    print("\n=== Example 4: Custom Context ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    engine = KnowledgeEngine()

    text = """
    The code implements an async function that fetches data from an API.
    It uses the aiohttp library for making asynchronous HTTP requests.
    Error handling is implemented using try-except blocks.
    """

    # Extract with specific context
    graph = await engine.extract_knowledge_graph(
        text=text,
        context="Python async programming and web development",
        upload_to_neo4j=False
    )

    print(f"Extraction with context '{graph.metadata.get('context')}':")
    print(f"  - Entities: {len(graph.entities)}")
    print(f"  - Relationships: {len(graph.relationships)}")

    await engine.cleanup_kggen_pipeline()


# Example 5: Neo4j Integration
async def example_5_neo4j_integration():
    """
    Extract knowledge graph and upload to Neo4j.
    """
    print("\n=== Example 5: Neo4j Integration ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    # Make sure Neo4j is running and environment variables are set
    # NEO4J_URI=bolt://localhost:7687
    # NEO4J_USER=neo4j
    # NEO4J_PASSWORD=your_password

    engine = KnowledgeEngine()

    text = """
    React is a JavaScript library for building user interfaces.
    It was developed by Facebook and is maintained by Meta.
    React uses a virtual DOM for efficient updates.
    Components are the building blocks of React applications.
    """

    # Extract and upload to Neo4j
    try:
        graph = await engine.extract_knowledge_graph(
            text=text,
            context="Frontend web development",
            upload_to_neo4j=True  # Upload to Neo4j
        )

        print(f"Knowledge graph uploaded to Neo4j:")
        print(f"  - Entities uploaded: {len(graph.entities)}")
        print(f"  - Relationships uploaded: {len(graph.relationships)}")

        # Query Neo4j statistics
        if engine.neo4j_backend:
            stats = await engine.get_neo4j_statistics()
            print(f"\nNeo4j Statistics:")
            print(f"  - Total entities: {stats.get('entity_count', 0)}")
            print(f"  - Total relationships: {stats.get('relationship_count', 0)}")

    except Exception as e:
        print(f"Neo4j integration failed: {e}")
        print("Make sure Neo4j is running and credentials are configured.")

    await engine.cleanup_kggen_pipeline()


# Example 6: Export Knowledge Graph
async def example_6_export_graph():
    """
    Export knowledge graph from Neo4j in various formats.
    """
    print("\n=== Example 6: Export Knowledge Graph ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    engine = KnowledgeEngine()

    if not engine.neo4j_backend:
        print("Neo4j backend not initialized. Skipping export example.")
        return

    try:
        # Export as JSON
        json_export = await engine.export_neo4j_graph(format='json')
        print("Exported as JSON (first 500 chars):")
        print(json_export[:500])

        # Export as CSV
        csv_export = await engine.export_neo4j_graph(format='csv')
        print("\nExported as CSV (first 500 chars):")
        print(csv_export[:500])

        # Save to file
        output_file = Path("knowledge_graph_export.json")
        output_file.write_text(json_export)
        print(f"\nSaved to {output_file}")

    except Exception as e:
        print(f"Export failed: {e}")

    await engine.cleanup_kggen_pipeline()


# Example 7: Query Specific Entity
async def example_7_query_entity():
    """
    Query a specific entity from Neo4j.
    """
    print("\n=== Example 7: Query Specific Entity ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    engine = KnowledgeEngine()

    if not engine.neo4j_backend:
        print("Neo4j backend not initialized. Skipping query example.")
        return

    # First, upload some data
    text = """
    Python is a programming language.
    Django is a Python web framework.
    Flask is another Python web framework.
    """

    try:
        await engine.extract_knowledge_graph(text, upload_to_neo4j=True)

        # Query for a specific entity
        entity_name = "Python"
        entity_data = await engine.query_neo4j_entity(entity_name)

        if entity_data:
            print(f"Found entity '{entity_name}':")
            print(f"  - Entity: {entity_data.get('entity')}")
            print(f"  - Relationships: {len(entity_data.get('relationships', []))}")
            for rel in entity_data.get('relationships', [])[:5]:
                print(f"    - [{rel.get('predicate')}] --> {rel.get('target')}")
        else:
            print(f"Entity '{entity_name}' not found in Neo4j")

    except Exception as e:
        print(f"Query failed: {e}")

    await engine.cleanup_kggen_pipeline()


# Example 8: Advanced Chunking Strategies
async def example_8_advanced_chunking():
    """
    Use different chunking strategies for document processing.
    """
    print("\n=== Example 8: Advanced Chunking Strategies ===\n")

    from knowledge_engine.integrations.kggen_chunking import DocumentChunker

    # Sample document
    document = """
    # Chapter 1: Introduction

    This is the introduction to the document. It provides an overview
    of the topics that will be covered.

    # Chapter 2: Background

    This chapter covers the background information. It includes
    historical context and related work.

    ## Section 2.1: History

    The history of the field dates back to the 1950s.

    ## Section 2.2: Related Work

    Related work includes various approaches to solving similar problems.

    # Chapter 3: Methodology

    This chapter describes the methodology used in this work.
    """ * 5  # Repeat to make it longer

    # Sentence-based chunking
    chunker = DocumentChunker(chunk_size=500, overlap=50)
    chunks_sentence = chunker.chunk_with_preservation(document, preserve_sentences=True)
    print(f"Sentence-based chunking: {len(chunks_sentence)} chunks")

    # Size-based chunking
    chunks_size = chunker.chunk_with_preservation(document, preserve_sentences=False)
    print(f"Size-based chunking: {len(chunks_size)} chunks")

    # Paragraph-based chunking
    chunks_paragraph = chunker.chunk_by_paragraphs(document, max_paragraphs_per_chunk=3)
    print(f"Paragraph-based chunking: {len(chunks_paragraph)} chunks")

    # Semantic unit chunking
    chunks_semantic = chunker.chunk_by_semantic_units(document)
    print(f"Semantic unit chunking: {len(chunks_semantic)} chunks")

    # Get statistics
    stats = chunker.get_chunk_statistics(chunks_sentence)
    print(f"\nChunk statistics:")
    print(f"  - Total chunks: {stats['total_chunks']}")
    print(f"  - Total length: {stats['total_length']}")
    print(f"  - Average length: {stats['avg_length']:.2f}")
    print(f"  - Min length: {stats['min_length']}")
    print(f"  - Max length: {stats['max_length']}")


# Example 9: Progress Tracking
async def example_9_progress_tracking():
    """
    Process chunks with progress tracking.
    """
    print("\n=== Example 9: Progress Tracking ===\n")

    from knowledge_engine.integrations.kggen_chunking import DocumentChunker
    from knowledge_engine.integrations.kggen_parallel import ParallelChunkProcessor

    # Create sample chunks
    document = " ".join([f"Sentence {i}. " for i in range(1000)])
    chunker = DocumentChunker(chunk_size=200, overlap=20)
    chunks = chunker.chunk_document(document)

    print(f"Processing {len(chunks)} chunks with progress tracking...")

    processor = ParallelChunkProcessor(max_workers=4)

    # Progress callback
    def progress_callback(progress):
        print(
            f"Progress: {progress.completed_chunks}/{progress.total_chunks} "
            f"({progress.completion_percentage:.1f}%) - "
            f"Elapsed: {progress.elapsed_time:.1f}s"
        )

    # Process with progress
    def process_func(chunk):
        # Simulate some work
        import time
        time.sleep(0.01)
        return f"Processed chunk {chunk.chunk_id}"

    results = await processor.process_with_progress(
        chunks,
        process_func,
        progress_callback=progress_callback,
        log_interval=1.0
    )

    print(f"\nCompleted processing {len(results)} chunks")


# Example 10: Complete Workflow
async def example_10_complete_workflow():
    """
    Complete workflow: document → chunking → extraction → upload → query.
    """
    print("\n=== Example 10: Complete Workflow ===\n")

    from knowledge_engine.engine import KnowledgeEngine

    engine = KnowledgeEngine()

    # Step 1: Load document
    document = """
    Kubernetes is an open-source container orchestration platform.
    It was originally developed by Google and is now maintained by the CNCF.
    Kubernetes automates deployment, scaling, and management of containerized applications.

    Docker is a platform for developing, shipping, and running applications in containers.
    Containers are lightweight, standalone, and executable software packages.
    Docker containers can run on any system that has Docker installed.

    Microservices architecture breaks applications into smaller, independent services.
    Each microservice runs in its own container and communicates via APIs.
    Kubernetes is commonly used to manage microservices deployments.
    """ * 10  # Make it larger

    print("Step 1: Document loaded")
    print(f"  - Length: {len(document)} characters")

    # Step 2: Chunk document
    from knowledge_engine.integrations.kggen_chunking import DocumentChunker
    chunker = DocumentChunker(chunk_size=1000, overlap=100)
    chunks = chunker.chunk_document(document)

    print(f"\nStep 2: Document chunked")
    print(f"  - Chunks: {len(chunks)}")

    # Step 3: Extract knowledge graph
    graph = await engine.kggen_pipeline.extract_from_large_document(
        document=document,
        chunk_size=1000,
        parallel_chunks=4
    )

    print(f"\nStep 3: Knowledge graph extracted")
    print(f"  - Entities: {len(graph.entities)}")
    print(f"  - Relationships: {len(graph.relationships)}")

    # Step 4: Upload to Neo4j (if available)
    if engine.neo4j_backend:
        result = await engine.kggen_pipeline.upload_to_neo4j(graph)

        if result.success:
            print(f"\nStep 4: Uploaded to Neo4j")
            print(f"  - Entities uploaded: {result.entities_uploaded}")
            print(f"  - Relationships uploaded: {result.relationships_uploaded}")

            # Step 5: Query Neo4j
            stats = await engine.get_neo4j_statistics()
            print(f"\nStep 5: Neo4j statistics")
            print(f"  - Total entities: {stats.get('entity_count', 0)}")
            print(f"  - Total relationships: {stats.get('relationship_count', 0)}")
        else:
            print(f"\nStep 4: Upload failed - {result.error}")
    else:
        print("\nStep 4: Neo4j not available, skipping upload")

    await engine.cleanup_kggen_pipeline()


# Main function to run examples
async def main():
    """Run all examples."""
    examples = [
        ("Simple Extraction", example_1_simple_extraction),
        ("Large Document", example_2_large_document),
        ("Batch Processing", example_3_batch_processing),
        ("Custom Context", example_4_custom_context),
        ("Neo4j Integration", example_5_neo4j_integration),
        ("Export Graph", example_6_export_graph),
        ("Query Entity", example_7_query_entity),
        ("Advanced Chunking", example_8_advanced_chunking),
        ("Progress Tracking", example_9_progress_tracking),
        ("Complete Workflow", example_10_complete_workflow),
    ]

    print("KG-Gen Pipeline Examples")
    print("=" * 60)

    # Run specific example or all
    import sys

    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        if 1 <= example_num <= len(examples):
            name, func = examples[example_num - 1]
            print(f"\nRunning Example {example_num}: {name}")
            try:
                await func()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Invalid example number. Choose 1-{len(examples)}")
    else:
        # Run a subset of examples
        for i, (name, func) in enumerate(examples[:5], 1):
            print(f"\n{'=' * 60}")
            print(f"Example {i}: {name}")
            print('=' * 60)
            try:
                await func()
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
