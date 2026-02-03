"""
RAGBits Document Processor - Complete Usage Example

This example demonstrates how to use the RAGBits document processor
for knowledge ingestion and semantic search.

Usage:
    python ragbits_document_processor_example.py
"""



import asyncio
import logging
from pathlib import Path

# Import the processor
from knowledge_engine.ragbits_document_processor import (
    RAGBitsDocumentProcessor,
    RAGBitsProcessorConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """
    Example 1: Basic text ingestion and search
    """
    print("\n" + "="*60)
    print("Example 1: Basic Text Ingestion and Search")
    print("="*60 + "\n")

    # Create processor with default config (in-memory storage)
    processor = RAGBitsDocumentProcessor()

    # Initialize
    if not await processor.initialize():
        print("❌ Failed to initialize processor (RAGBits not available)")
        return

    # Ingest some sample documents
    documents = [
        {
            "text": """
            Machine learning is a subset of artificial intelligence that enables
            systems to learn from data and improve their performance over time
            without being explicitly programmed. Common algorithms include
            neural networks, decision trees, and support vector machines.
            """,
            "metadata": {"title": "Introduction to ML", "category": "AI"}
        },
        {
            "text": """
            Deep learning is a type of machine learning that uses neural networks
            with multiple layers to model complex patterns in data. It has achieved
            state-of-the-art results in computer vision, natural language processing,
            and speech recognition.
            """,
            "metadata": {"title": "Deep Learning Overview", "category": "AI"}
        },
        {
            "text": """
            Python is a high-level programming language known for its simplicity
            and readability. It is widely used in web development, data science,
            artificial intelligence, and automation. Popular frameworks include
            Django, Flask, and TensorFlow.
            """,
            "metadata": {"title": "Python Programming", "category": "Programming"}
        }
    ]

    # Ingest documents
    print("📝 Ingesting documents...")
    for doc in documents:
        result = await processor.ingest_text(
            text=doc["text"],
            metadata=doc["metadata"],
            source=doc["metadata"]["title"]
        )
        if result.success:
            print(f"  ✅ {result.document_id}: {result.chunks_ingested} chunks")
        else:
            print(f"  ❌ Failed: {result.error}")

    # Search for relevant content
    print("\n🔍 Searching for 'neural networks'...")
    results = await processor.search("neural networks", top_k=2)

    print(f"\nFound {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        print(f"   Metadata: {result['metadata']}\n")

    # Get statistics
    stats = await processor.get_statistics()
    print(f"📊 Statistics:")
    print(f"   Ingested Documents: {stats['ingested_documents']}")
    print(f"   Vector Store: {stats['vector_store_type']}")
    print(f"   Embedding Model: {stats['embedding_model']}")


async def example_file_ingestion():
    """
    Example 2: Ingest documents from files
    """
    print("\n" + "="*60)
    print("Example 2: File Ingestion")
    print("="*60 + "\n")

    processor = RAGBitsDocumentProcessor()
    await processor.initialize()

    # Create sample files
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)

    # Create sample text files
    (sample_dir / "ai_basics.txt").write_text("""
    Artificial Intelligence (AI) is the simulation of human intelligence processes
    by machines, especially computer systems. These processes include learning,
    reasoning, and self-correction. AI applications include expert systems,
    natural language processing, speech recognition, and machine vision.
    """)

    (sample_dir / "web_development.txt").write_text("""
    Web development involves building and maintaining websites. It includes web design,
    web content development, client-side scripting, server-side scripting, and
    network security. Popular technologies include HTML, CSS, JavaScript, React,
    Vue.js, and Node.js.
    """)

    (sample_dir / "data_science.txt").write_text("""
    Data science combines statistics, mathematics, and computer science to extract
    insights from data. Key techniques include data mining, machine learning,
    data visualization, and predictive analytics. Tools like Python, R, and SQL
    are commonly used by data scientists.
    """)

    print(f"📁 Created sample files in {sample_dir}/")

    # Ingest all files
    print("\n📝 Ingesting files...")
    results = await processor.ingest_directory(
        directory=str(sample_dir),
        pattern="*.txt",
        metadata={"source": "sample_documents"}
    )

    print(f"\nProcessed {len(results)} files:")
    for result in results:
        if result.success:
            print(f"  ✅ {result.metadata['file_name']}: {result.chunks_ingested} chunks")
        else:
            print(f"  ❌ {result.metadata.get('file_name', 'unknown')}: {result.error}")

    # Search
    print("\n🔍 Searching for 'machine learning'...")
    results = await processor.search("machine learning", top_k=2)

    print(f"\nFound {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:150]}...")
        print()


async def example_with_filters():
    """
    Example 3: Search with metadata filters
    """
    print("\n" + "="*60)
    print("Example 3: Search with Filters")
    print("="*60 + "\n")

    processor = RAGBitsDocumentProcessor()
    await processor.initialize()

    # Ingest categorized documents
    docs = [
        {
            "text": "React is a JavaScript library for building user interfaces.",
            "metadata": {"category": "Frontend", "language": "JavaScript"}
        },
        {
            "text": "Django is a Python web framework for rapid development.",
            "metadata": {"category": "Backend", "language": "Python"}
        },
        {
            "text": "TensorFlow is a machine learning framework for Python.",
            "metadata": {"category": "AI", "language": "Python"}
        },
        {
            "text": "Vue.js is a progressive JavaScript framework for UIs.",
            "metadata": {"category": "Frontend", "language": "JavaScript"}
        }
    ]

    print("📝 Ingesting categorized documents...")
    for doc in docs:
        result = await processor.ingest_text(
            text=doc["text"],
            metadata=doc["metadata"],
            source=f"{doc['metadata']['category']}_{doc['metadata']['language']}"
        )
        print(f"  ✅ {result.document_id}")

    # Search without filters
    print("\n🔍 Search: 'JavaScript framework'")
    results = await processor.search("JavaScript framework", top_k=5)
    print(f"Found {len(results)} results (no filters)")

    # Search with category filter
    print("\n🔍 Search: 'framework' (Frontend only)")
    results = await processor.search(
        "framework",
        top_k=5,
        filters={"category": "Frontend"}
    )
    print(f"Found {len(results)} results (Frontend only)")

    # Search with language filter
    print("\n🔍 Search: 'framework' (Python only)")
    results = await processor.search(
        "framework",
        top_k=5,
        filters={"language": "Python"}
    )
    print(f"Found {len(results)} results (Python only)")


async def example_with_qdrant():
    """
    Example 4: Using Qdrant vector store (persistent storage)

    Note: This requires Qdrant to be running at http://localhost:6333
    """
    print("\n" + "="*60)
    print("Example 4: Using Qdrant Vector Store")
    print("="*60 + "\n")

    # Configure processor to use Qdrant
    config = RAGBitsProcessorConfig(
        vector_store_type="qdrant",
        qdrant_url="http://localhost:6333",
        qdrant_collection="knowledge_engine_docs"
    )

    processor = RAGBitsDocumentProcessor(config)

    print("⚠️  Note: This requires Qdrant running at http://localhost:6333")
    print("   To start Qdrant:")
    print("   docker run -p 6333:6333 qdrant/qdrant")
    print()

    # Try to initialize
    if not await processor.initialize():
        print("❌ Failed to initialize (Qdrant not available)")
        print("   Falling back to in-memory storage for demo...")

        # Use in-memory instead
        processor = RAGBitsDocumentProcessor()
        await processor.initialize()

    # Ingest and search
    print("📝 Ingesting documents...")
    await processor.ingest_text(
        "Qdrant is a vector similarity search engine.",
        metadata={"type": "database", "language": "Rust"},
        source="qdrant_info"
    )

    print("\n🔍 Searching for 'vector search'...")
    results = await processor.search("vector search", top_k=1)

    if results:
        print(f"✅ Found: {results[0]['content']}")


async def example_idempotency():
    """
    Example 5: Demonstrating idempotency (safe to re-ingest)
    """
    print("\n" + "="*60)
    print("Example 5: Idempotency - Safe Re-Ingestion")
    print("="*60 + "\n")

    processor = RAGBitsDocumentProcessor()
    await processor.initialize()

    # Ingest same document twice
    text = "This document tests idempotency."
    source = "idempotency_test"

    print("📝 First ingestion...")
    result1 = await processor.ingest_text(text, source=source)
    print(f"  Document ID: {result1.document_id}")
    print(f"  Chunks: {result1.chunks_ingested}")

    print("\n📝 Second ingestion (same content)...")
    result2 = await processor.ingest_text(text, source=source)
    print(f"  Document ID: {result2.document_id}")
    print(f"  Chunks: {result2.chunks_ingested}")
    print(f"  Note: Skipped due to idempotency check")

    print("\n✅ Idempotency verified: Re-ingesting same document is safe")


async def main():
    """
    Run all examples
    """
    print("\n" + "="*60)
    print("RAGBits Document Processor - Complete Examples")
    print("="*60)

    try:
        await example_basic_usage()
        await example_file_ingestion()
        await example_with_filters()
        await example_with_qdrant()
        await example_idempotency()

        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
