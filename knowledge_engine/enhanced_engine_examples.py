"""
Enhanced Knowledge Engine - Usage Examples

This file demonstrates key features of the Enhanced Knowledge Engine:
1. Basic CRUD operations
2. Semantic search
3. Knowledge graph operations
4. Active learning and feedback
5. Analytics and reporting
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_knowledge_core import (
    KnowledgeType, RelationType,
    KnowledgeItem, KnowledgeRelation
)
from enhanced_knowledge_engine import create_knowledge_engine
from knowledge_analytics import KnowledgeAnalyticsEngine


async def example_basic_operations():
    """Example: Basic CRUD operations."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic CRUD Operations")
    print("="*60)
    
    # Create engine
    engine = await create_knowledge_engine(
        storage_path="./example_storage",
        cache_size=1000
    )
    
    try:
        # Create knowledge items
        print("\n1. Adding knowledge items...")
        
        python_intro = await engine.add_knowledge(
            content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
            knowledge_type=KnowledgeType.TEXT,
            metadata={
                "author": "Documentation Team",
                "difficulty": "beginner",
                "estimated_read_time": 5
            },
            tags={"python", "programming", "introduction"},
            source="official_docs"
        )
        print(f"   Added: {python_intro.id} - Python Introduction")
        
        python_async = await engine.add_knowledge(
            content="""
            Asyncio is a library to write concurrent code using the async/await syntax.
            It is used as a foundation for multiple Python asynchronous frameworks.
            """,
            knowledge_type=KnowledgeType.TEXT,
            metadata={
                "author": "Core Developers",
                "difficulty": "intermediate",
                "prerequisites": ["python_basics"]
            },
            tags={"python", "async", "concurrency"},
            source="python_docs"
        )
        print(f"   Added: {python_async.id} - Python Asyncio")
        
        code_example = await engine.add_knowledge(
            content="""
            import asyncio
            
            async def main():
                print('Hello')
                await asyncio.sleep(1)
                print('World')
            
            asyncio.run(main())
            """,
            knowledge_type=KnowledgeType.CODE,
            metadata={
                "language": "python",
                "runnable": True
            },
            tags={"python", "async", "example"},
            source="code_repository"
        )
        print(f"   Added: {code_example.id} - Async Code Example")
        
        # Retrieve
        print("\n2. Retrieving knowledge...")
        retrieved = await engine.get_knowledge(python_intro.id)
        if retrieved:
            print(f"   Retrieved: {retrieved.content[:50]}...")
        
        # Update
        print("\n3. Updating knowledge...")
        updated = await engine.update_knowledge(
            python_intro.id,
            new_content="Python is a versatile, high-level programming language with elegant syntax.",
            confidence=0.95,
            metadata_updates={"updated_by": "admin"}
        )
        print(f"   Updated to version {updated.version}")
        
        # Delete
        print("\n4. Deleting knowledge...")
        deleted = await engine.delete_knowledge(code_example.id)
        print(f"   Deleted: {deleted}")
        
    finally:
        await engine.shutdown()


async def example_semantic_search():
    """Example: Semantic search capabilities."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Semantic Search")
    print("="*60)
    
    engine = await create_knowledge_engine(cache_size=1000)
    
    try:
        # Add diverse knowledge
        print("\n1. Adding diverse knowledge...")
        
        items = [
            ("Machine learning is a subset of artificial intelligence", 
             ["ml", "ai", "data_science"]),
            ("Neural networks are computing systems inspired by biological brains",
             ["neural_networks", "deep_learning"]),
            ("Python is great for data analysis and machine learning",
             ["python", "data_science", "ml"]),
            ("Docker containers help deploy applications consistently",
             ["docker", "devops", "deployment"]),
            ("REST APIs use HTTP methods to interact with resources",
             ["api", "rest", "web_development"]),
        ]
        
        for content, tags in items:
            item = await engine.add_knowledge(
                content=content,
                knowledge_type=KnowledgeType.TEXT,
                tags=set(tags)
            )
            print(f"   Added: {content[:40]}...")
        
        # Perform searches
        print("\n2. Searching with different modes...")
        
        # Keyword search
        print("\n   Keyword search for 'machine learning':")
        results = await engine.search(
            query="machine learning",
            search_mode="keyword",
            max_results=3
        )
        for r in results:
            print(f"      [{r.relevance_score:.2f}] {r.item.content[:50]}...")
        
        # Semantic search
        print("\n   Semantic search for 'AI and data science':")
        results = await engine.search(
            query="AI and data science",
            search_mode="semantic",
            max_results=3
        )
        for r in results:
            print(f"      [{r.relevance_score:.2f}] {r.item.content[:50]}...")
        
        # Hybrid search with filters
        print("\n   Hybrid search with tag filter 'ml':")
        results = await engine.search(
            query="programming",
            search_mode="hybrid",
            tags=["ml"],
            max_results=3
        )
        for r in results:
            print(f"      [{r.relevance_score:.2f}] {r.item.content[:50]}...")
            
    finally:
        await engine.shutdown()


async def example_knowledge_graph():
    """Example: Knowledge graph operations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Knowledge Graph")
    print("="*60)
    
    engine = await create_knowledge_engine(
        enable_graph=True,
        cache_size=1000
    )
    
    try:
        print("\n1. Creating knowledge hierarchy...")
        
        # Create parent topics
        ai = await engine.add_knowledge(
            content="Artificial Intelligence",
            knowledge_type=KnowledgeType.TEXT,
            tags={"ai", "computer_science"}
        )
        print(f"   Created: AI ({ai.id})")
        
        # Create subtopics
        ml = await engine.add_knowledge(
            content="Machine Learning",
            knowledge_type=KnowledgeType.TEXT,
            tags={"ml", "ai"}
        )
        print(f"   Created: ML ({ml.id})")
        
        dl = await engine.add_knowledge(
            content="Deep Learning",
            knowledge_type=KnowledgeType.TEXT,
            tags={"dl", "ml", "ai"}
        )
        print(f"   Created: DL ({dl.id})")
        
        nlp = await engine.add_knowledge(
            content="Natural Language Processing",
            knowledge_type=KnowledgeType.TEXT,
            tags={"nlp", "ai", "ml"}
        )
        print(f"   Created: NLP ({nlp.id})")
        
        # Create relations
        print("\n2. Creating relationships...")
        
        await engine.create_relation(
            ai.id, ml.id, RelationType.PART_OF, weight=1.0
        )
        print(f"   AI --[part_of]--> ML")
        
        await engine.create_relation(
            ml.id, dl.id, RelationType.PART_OF, weight=0.9
        )
        print(f"   ML --[part_of]--> DL")
        
        await engine.create_relation(
            ai.id, nlp.id, RelationType.PART_OF, weight=0.8
        )
        print(f"   AI --[part_of]--> NLP")
        
        await engine.create_relation(
            ml.id, nlp.id, RelationType.DEPENDS_ON, weight=0.7
        )
        print(f"   NLP --[depends_on]--> ML")
        
        # Find related
        print("\n3. Finding related to AI...")
        related = await engine.find_related(ai.id)
        for item, relation in related:
            print(f"   {relation.relation_type.value}: {item.content}")
        
        # Find paths
        print("\n4. Finding path from AI to DL...")
        paths = await engine.find_path(ai.id, dl.id, max_depth=3)
        for path in paths:
            print("   Path: " + " -> ".join(item.content for item in path))
        
        # Get graph stats
        print("\n5. Graph statistics:")
        stats = await engine.get_knowledge_graph_stats()
        print(f"   Nodes: {stats.get('nodes', 0)}")
        print(f"   Edges: {stats.get('edges', 0)}")
        print(f"   Components: {stats.get('connected_components', 0)}")
        
    finally:
        await engine.shutdown()


async def example_active_learning():
    """Example: Active learning and feedback."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Active Learning")
    print("="*60)
    
    engine = await create_knowledge_engine(
        enable_learning=True,
        cache_size=1000
    )
    
    try:
        print("\n1. Creating knowledge items...")
        
        items = []
        for i in range(3):
            item = await engine.add_knowledge(
                content=f"Tutorial topic {i+1}: Advanced concepts in async programming",
                knowledge_type=KnowledgeType.TEXT,
                metadata={"topic_number": i+1}
            )
            items.append(item)
            print(f"   Created item {i+1}: {item.id}")
        
        # Simulate user feedback
        print("\n2. Recording user feedback...")
        
        feedback_data = [
            (items[0].id, "positive", 0.9, "user-1"),
            (items[0].id, "positive", 0.85, "user-2"),
            (items[0].id, "positive", 0.95, "user-3"),
            (items[1].id, "neutral", 0.6, "user-1"),
            (items[1].id, "negative", 0.3, "user-2"),
            (items[1].id, "neutral", 0.5, "user-3"),
            (items[2].id, "positive", 0.8, "user-1"),
            (items[2].id, "positive", 0.75, "user-2"),
        ]
        
        for item_id, feedback_type, score, user_id in feedback_data:
            await engine.record_feedback(item_id, feedback_type, score, user_id)
        
        print(f"   Recorded {len(feedback_data)} feedback entries")
        
        # Get quality metrics
        print("\n3. Quality metrics by item:")
        for item in items:
            quality = await engine.get_item_quality(item.id)
            print(f"   Item {item.id}:")
            print(f"      Average score: {quality['average_score']:.2f}")
            print(f"      Feedback count: {quality['feedback_count']}")
            if 'trend' in quality:
                print(f"      Trend: {quality['trend']}")
        
        # Get recommendations
        print("\n4. Learning recommendations:")
        recommendations = await engine.get_learning_recommendations()
        for rec in recommendations:
            print(f"   [{rec['priority']}] {rec['type']}: {rec['message']}")
        
    finally:
        await engine.shutdown()


async def example_analytics():
    """Example: Analytics and reporting."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Analytics and Reporting")
    print("="*60)
    
    engine = await create_knowledge_engine(cache_size=1000)
    analytics = KnowledgeAnalyticsEngine(knowledge_engine=engine)
    
    try:
        print("\n1. Creating test data...")
        
        # Create items with varying quality
        for i in range(10):
            confidence = 0.5 + (i * 0.05)  # Increasing confidence
            item = await engine.add_knowledge(
                content=f"Knowledge item {i+1} with varying quality",
                knowledge_type=KnowledgeType.TEXT if i % 2 == 0 else KnowledgeType.CODE,
                metadata={
                    "index": i,
                    "category": "test"
                },
                tags={"test", f"tag_{i % 3}"},
                confidence=confidence
            )
        
        # Simulate usage
        print("\n2. Simulating usage patterns...")
        
        for i in range(20):
            # Log searches
            queries = ["python", "async", "machine learning", "tutorial", "api"]
            query = queries[i % len(queries)]
            analytics.usage_analytics.log_search(query, results_count=(i % 5) + 1)
            
            # Log item access
            item_id = list(engine._items.keys())[i % len(engine._items)]
            analytics.usage_analytics.log_access(item_id, user_id=f"user_{i % 3}")
        
        # Record trends
        print("\n3. Recording trend metrics...")
        for i in range(30):
            analytics.record_metric("daily_active_users", 50 + i * 2)
            analytics.record_metric("new_knowledge_items", 5 + i % 10)
            analytics.record_metric("search_queries", 100 + i * 5)
        
        # Generate report
        print("\n4. Generating comprehensive report...")
        items = list(engine._items.values())
        report = analytics.generate_comprehensive_report(items)
        
        print("\n   Quality Summary:")
        quality = report['quality']
        print(f"      Total items: {quality['total_items']}")
        print(f"      Average score: {quality['average_overall_score']:.2f}")
        print(f"      Quality distribution: {quality['quality_distribution']}")
        
        print("\n   Trends:")
        for metric, trend in list(report['trends'].items())[:3]:
            print(f"      {metric}: {trend.direction} ({trend.change_percent:+.1f}%)")
        
        print("\n   Usage Analytics:")
        usage = report['usage']
        search_stats = usage.get('search', {})
        print(f"      Total searches: {search_stats.get('total_searches', 0)}")
        print(f"      Unique queries: {search_stats.get('unique_queries', 0)}")
        
        print("\n   Insights:")
        for insight in report['insights'][:3]:
            print(f"      [{insight['severity']}] {insight['category']}: {insight['message']}")
        
        # Dashboard data
        print("\n5. Dashboard Summary:")
        dashboard = analytics.get_dashboard_data(items)
        summary = dashboard['summary']
        print(f"      Total items: {summary['total_items']}")
        print(f"      Average quality: {summary['avg_quality']:.2f}")
        print(f"      Active trends: {summary['active_trends']}")
        print(f"      Open issues: {summary['open_issues']}")
        
    finally:
        await engine.shutdown()


async def example_event_handling():
    """Example: Event handling."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Event Handling")
    print("="*60)
    
    engine = await create_knowledge_engine(cache_size=1000)
    events_received = []
    
    try:
        # Define event handler
        def on_event(event):
            events_received.append({
                "type": event.event_type,
                "item_id": event.item_id,
                "timestamp": event.timestamp
            })
            print(f"   Event: {event.event_type} for {event.item_id}")
        
        # Register handler
        engine.add_event_handler(on_event)
        print("\n1. Registered event handler")
        
        # Trigger events
        print("\n2. Triggering events...")
        
        item = await engine.add_knowledge(
            content="Test event content",
            knowledge_type=KnowledgeType.TEXT
        )
        
        await engine.update_knowledge(item.id, "Updated content")
        
        await engine.delete_knowledge(item.id)
        
        # Wait for event processing
        await asyncio.sleep(0.5)
        
        print(f"\n3. Total events received: {len(events_received)}")
        for event in events_received:
            print(f"   - {event['type']}: {event['item_id']}")
        
    finally:
        await engine.shutdown()


async def example_performance_monitoring():
    """Example: Performance monitoring."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Performance Monitoring")
    print("="*60)
    
    engine = await create_knowledge_engine(cache_size=1000)
    
    try:
        print("\n1. Health check:")
        health = engine.get_health_check()
        print(f"   Status: {health['status']}")
        print(f"   Initialized: {health['initialized_at']}")
        print("   Components:")
        for component, status in health['components'].items():
            print(f"      {component}: {'✓' if status else '✗'}")
        
        print("\n2. Adding test data for stats...")
        for i in range(100):
            await engine.add_knowledge(
                content=f"Performance test item {i}",
                knowledge_type=KnowledgeType.TEXT,
                generate_embedding=(i % 10 == 0)  # Only some with embeddings
            )
        
        print("\n3. Performing searches...")
        for i in range(50):
            await engine.search(f"query {i}", max_results=10)
        
        print("\n4. Statistics:")
        stats = engine.get_stats()
        print(f"   Total items: {stats['total_items']}")
        print(f"   Items created: {stats['items_created']}")
        print(f"   Searches performed: {stats['searches_performed']}")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100:.1f}%")
        
        print("\n5. Cache statistics:")
        cache_stats = await engine.cache.get_stats()
        print(f"   Total items: {cache_stats['total_items']}")
        print(f"   Active items: {cache_stats['active_items']}")
        print(f"   Max size: {cache_stats['max_size']}")
        
        print("\n6. Search index statistics:")
        search_stats = stats['search']
        print(f"   Vector index: {search_stats['vector_index_size']}")
        print(f"   Cached items: {search_stats['cached_items']}")
        
    finally:
        await engine.shutdown()


async def run_all_examples():
    """Run all examples."""
    print("\n" + "="*60)
    print("ENHANCED KNOWLEDGE ENGINE - EXAMPLES")
    print("="*60)
    
    examples = [
        ("Basic Operations", example_basic_operations),
        ("Semantic Search", example_semantic_search),
        ("Knowledge Graph", example_knowledge_graph),
        ("Active Learning", example_active_learning),
        ("Analytics", example_analytics),
        ("Event Handling", example_event_handling),
        ("Performance Monitoring", example_performance_monitoring),
    ]
    
    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n   ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run examples
    asyncio.run(run_all_examples())
