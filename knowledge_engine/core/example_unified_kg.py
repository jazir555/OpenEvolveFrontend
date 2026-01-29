"""
Example usage of Unified Knowledge Graph Manager.

Demonstrates how to use the unified interface across different backends.
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage example with in-memory backend"""
    from knowledge_engine.core import UnifiedKnowledgeGraph

    # Create manager with default config (memory backend)
    kg = UnifiedKnowledgeGraph()

    # Connect to backends
    await kg.connect_all()

    try:
        # Add knowledge
        entry_id = await kg.add_knowledge(
            source="example",
            content="The Unified Knowledge Graph Manager provides a consistent interface.",
            metadata={"tags": ["demo", "introduction"], "priority": "high"}
        )
        logger.info(f"Added knowledge entry: {entry_id}")

        # Add more knowledge
        await kg.add_knowledge(
            source="documentation",
            content="Neo4j is used for graph storage and relationship queries.",
            metadata={"tags": ["neo4j", "graph"]}
        )

        await kg.add_knowledge(
            source="documentation",
            content="Qdrant provides fast vector similarity search.",
            metadata={"tags": ["qdrant", "search"]}
        )

        # Search knowledge
        results = await kg.search("graph database")
        logger.info(f"Search found {results.total_count} results:")
        for result in results.results:
            logger.info(f"  - {result['source']}: {result['content'][:50]}...")

        # Analyze graph
        analysis = await kg.analyze("source_distribution")
        logger.info(f"Source distribution: {analysis.results}")

        # Get statistics
        stats = await kg.get_graph_stats()
        logger.info(f"Graph statistics: {stats}")

        # Visualize
        visualization = await kg.visualize("html")
        logger.info(f"Generated visualization ({len(visualization)} characters)")

    finally:
        await kg.disconnect_all()


async def example_with_neo4j():
    """Example using Neo4j backend"""
    from knowledge_engine.core import UnifiedKnowledgeGraph
    import yaml

    # Configuration with Neo4j
    config = {
        "backends": {
            "neo4j": {
                "enabled": True,
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "password",  # In production, use environment variables
                "database": "neo4j"
            },
            "memory": {
                "enabled": True
            }
        },
        "fallback_chain": ["neo4j", "memory"],
        "operations": {
            "add_knowledge": ["neo4j"],
            "search": ["neo4j"],
            "analyze": ["neo4j"],
            "visualize": ["neo4j"]
        }
    }

    kg = UnifiedKnowledgeGraph()
    kg.config = config
    kg._initialize_backends()

    await kg.connect_all()

    try:
        # Add knowledge to Neo4j
        entry_id = await kg.add_knowledge(
            source="neo4j_example",
            content="Neo4j is a native graph database with efficient relationship traversal.",
            metadata={"type": "technical"}
        )
        logger.info(f"Added to Neo4j: {entry_id}")

        # Search
        results = await kg.search("graph traversal")
        logger.info(f"Neo4j search: {results.total_count} results in {results.search_time_ms:.2f}ms")

        # Analyze
        analysis = await kg.analyze("entity_connections")
        logger.info(f"Entity analysis: {analysis.results}")

    finally:
        await kg.disconnect_all()


async def example_batch_operations():
    """Example of batch operations"""
    from knowledge_engine.core import UnifiedKnowledgeGraph

    kg = UnifiedKnowledgeGraph()
    await kg.connect_all()

    try:
        # Batch add knowledge
        entries = [
            {
                "source": "batch_example",
                "content": "Batch operations are more efficient than individual adds.",
                "metadata": {"batch": 1}
            },
            {
                "source": "batch_example",
                "content": "Vector embeddings can be pre-computed for faster indexing.",
                "metadata": {"batch": 2}
            },
            {
                "source": "batch_example",
                content": "Graph analytics reveal hidden patterns in knowledge connections.",
                "metadata": {"batch": 3}
            }
        ]

        ids = await kg.batch_add_knowledge(entries)
        logger.info(f"Batch added {len(ids)} entries")

        # Search across batch
        results = await kg.search("batch operations")
        logger.info(f"Found {results.total_count} results from batch")

    finally:
        await kg.disconnect_all()


async def example_multi_backend():
    """Example using multiple backends"""
    from knowledge_engine.core import UnifiedKnowledgeGraph

    # Configuration with multiple backends
    config = {
        "backends": {
            "memory": {
                "enabled": True
            },
            "mongodb": {
                "enabled": True,
                "uri": "mongodb://localhost:27017",
                "database": "knowledge_graph",
                "collection": "knowledge"
            }
        },
        "fallback_chain": ["mongodb", "memory"],
        "operations": {
            "add_knowledge": ["mongodb", "memory"],
            "search": ["mongodb", "memory"],
            "analyze": ["mongodb", "memory"],
            "visualize": ["memory"]
        }
    }

    kg = UnifiedKnowledgeGraph()
    kg.config = config
    kg._initialize_backends()

    # Check health
    await kg.connect_all()
    health = await kg.health_check()
    logger.info(f"Backend health: {health}")

    try:
        # Add knowledge - will use first healthy backend
        entry_id = await kg.add_knowledge(
            source="multi_backend",
            content="Data is automatically routed to the best available backend."
        )
        logger.info(f"Added entry: {entry_id}")

        # Get statistics from all backends
        stats = await kg.get_graph_stats()
        logger.info(f"Multi-backend stats: {stats}")

    finally:
        await kg.disconnect_all()


async def example_error_handling():
    """Example demonstrating error handling and fallback"""
    from knowledge_engine.core import UnifiedKnowledgeGraph, KnowledgeGraphError

    # Configuration with fallback
    config = {
        "backends": {
            "neo4j": {
                "enabled": True,
                "uri": "bolt://invalid-host:7687",  # Invalid - will fail
                "user": "neo4j",
                "password": "password"
            },
            "memory": {
                "enabled": True
            }
        },
        "fallback_chain": ["neo4j", "memory"],
        "operations": {
            "add_knowledge": ["neo4j", "memory"],
            "search": ["neo4j", "memory"]
        }
    }

    kg = UnifiedKnowledgeGraph()
    kg.config = config
    kg._initialize_backends()

    # Connect - Neo4j will fail, but memory will work
    connection_results = await kg.connect_all()
    logger.info(f"Connection results: {connection_results}")

    try:
        # This will automatically fall back to memory backend
        entry_id = await kg.add_knowledge(
            source="fallback_example",
            content="The system automatically falls back to healthy backends."
        )
        logger.info(f"Successfully added using fallback: {entry_id}")

    except KnowledgeGraphError as e:
        logger.error(f"All backends failed: {e}")

    finally:
        await kg.disconnect_all()


async def example_async_context_manager():
    """Example using async context manager"""
    from knowledge_engine.core import UnifiedKnowledgeGraph

    # Using context manager for automatic cleanup
    async with UnifiedKnowledgeGraph() as kg:
        # Add knowledge
        await kg.add_knowledge(
            source="context_manager",
            content="Async context managers ensure proper resource cleanup."
        )

        # Search
        results = await kg.search("context manager")
        logger.info(f"Found {results.total_count} results")

        # Resources are automatically cleaned up on exit


async def main():
    """Run all examples"""
    logger.info("=" * 80)
    logger.info("Unified Knowledge Graph Manager - Example Usage")
    logger.info("=" * 80)

    # Run examples
    logger.info("\n1. Basic Usage Example")
    await example_basic_usage()

    logger.info("\n2. Neo4j Backend Example")
    try:
        await example_with_neo4j()
    except Exception as e:
        logger.warning(f"Neo4j example failed (Neo4j not running?): {e}")

    logger.info("\n3. Batch Operations Example")
    await example_batch_operations()

    logger.info("\n4. Multi-Backend Example")
    try:
        await example_multi_backend()
    except Exception as e:
        logger.warning(f"Multi-backend example failed (MongoDB not running?): {e}")

    logger.info("\n5. Error Handling Example")
    await example_error_handling()

    logger.info("\n6. Async Context Manager Example")
    await example_async_context_manager()

    logger.info("\n" + "=" * 80)
    logger.info("All examples completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
