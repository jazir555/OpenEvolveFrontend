"""
CodeIndexer Usage Examples

This script demonstrates various CodeIndexer capabilities:
1. Repository indexing
2. Code search
3. Similarity detection
4. Incremental updates
5. Statistics
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import CodeIndexer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_engine.indexer import CodeIndexer


async def example_1_basic_indexing():
    """Example 1: Basic repository indexing"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Repository Indexing")
    print("="*80)

    # Initialize indexer with mock mode for demonstration
    indexer = CodeIndexer(
        code_base_path="../knowledge_engine",  # Analyze this codebase
        target_structure="Knowledge Engine with code indexing",
        output_dir="example_indexes",
        indexer_config_path="../indexer_config.yaml",
        enable_pre_filtering=False
    )

    # Index the repository
    print(f"\n📁 Indexing repository: knowledge_engine")
    output_files = await indexer.build_all_indexes()

    print(f"\n✅ Created {len(output_files)} index file(s):")
    for repo_name, file_path in output_files.items():
        print(f"  - {repo_name}: {file_path}")


async def example_2_code_search():
    """Example 2: Natural language code search"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Natural Language Code Search")
    print("="*80)

    indexer = CodeIndexer(
        code_base_path=".",
        output_dir="example_indexes"
    )

    # Search for database-related code
    queries = [
        "database connection handling",
        "LLM client initialization",
        "file parsing utilities"
    ]

    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = await indexer.search_code(query)

        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results[:5], 1):
                print(f"  {i}. {result['file_path']}")
                print(f"     Relevance: {result.get('relevance_score', 0):.2f}")
                print(f"     Reason: {result.get('reason', 'N/A')}")
        else:
            print("  No results found")


async def example_3_similar_files():
    """Example 3: Find similar files"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Finding Similar Files")
    print("="*80)

    indexer = CodeIndexer(
        code_base_path=".",
        output_dir="example_indexes"
    )

    # Find files similar to a reference file
    reference_file = "knowledge_engine/indexer.py"

    if Path(reference_file).exists():
        print(f"\n📋 Finding files similar to: {reference_file}")

        similar = await indexer.find_similar_files(
            file_path=reference_file,
            threshold=0.5,
            top_k=5
        )

        if similar:
            print(f"\nFound {len(similar)} similar files:")
            for i, sim in enumerate(similar, 1):
                print(f"\n  {i}. {sim['file_path']}")
                print(f"     Similarity: {sim['similarity_score']:.2f}")
                print(f"     Reason: {sim['reason']}")
                if sim.get('shared_aspects'):
                    print(f"     Shared: {', '.join(sim['shared_aspects'])}")
        else:
            print("No similar files found")
    else:
        print(f"Reference file not found: {reference_file}")


async def example_4_file_relationships():
    """Example 4: Get file relationships"""
    print("\n" + "="*80)
    print("EXAMPLE 4: File Relationships")
    print("="*80)

    indexer = CodeIndexer(
        code_base_path=".",
        output_dir="example_indexes"
    )

    # Get relationships for a specific file
    file_path = "knowledge_engine/llm_utils.py"

    print(f"\n🔗 Analyzing relationships for: {file_path}")

    relationships = await indexer.get_file_relationships(file_path)

    if relationships:
        file_summary = relationships.get('file_summary', {})
        print(f"\nFile Summary:")
        print(f"  Type: {file_summary.get('file_type', 'Unknown')}")
        print(f"  Functions: {', '.join(file_summary.get('main_functions', []))}")
        print(f"  Summary: {file_summary.get('summary', 'N/A')}")

        rels = relationships.get('relationships', [])
        metadata = relationships.get('metadata', {})

        print(f"\nRelationships:")
        print(f"  Total: {metadata.get('total_relationships', 0)}")
        print(f"  High confidence: {metadata.get('high_confidence_count', 0)}")

        for rel in rels[:3]:
            print(f"\n  → {rel['target_file_path']}")
            print(f"    Type: {rel['relationship_type']}")
            print(f"    Confidence: {rel['confidence_score']:.2f}")
            print(f"    Usage: {rel['usage_suggestions'][:80]}...")
    else:
        print("No relationships found")


async def example_5_incremental_updates():
    """Example 5: Incremental index updates"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Incremental Index Updates")
    print("="*80)

    indexer = CodeIndexer(
        code_base_path=".",
        output_dir="example_indexes"
    )

    # Update index for a specific file
    test_file = "knowledge_engine/indexer.py"

    if Path(test_file).exists():
        print(f"\n🔄 Updating index for: {test_file}")

        success = await indexer.update_index(
            file_path=test_file,
            repo_name="knowledge_engine"
        )

        if success:
            print("✅ Index updated successfully")
        else:
            print("❌ Update failed")
    else:
        print(f"File not found: {test_file}")


async def example_6_statistics():
    """Example 6: Get index statistics"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Index Statistics")
    print("="*80)

    indexer = CodeIndexer(
        code_base_path=".",
        output_dir="example_indexes"
    )

    print("\n📊 Gathering index statistics...")

    stats = indexer.get_index_statistics()

    print(f"\nTotal repositories indexed: {stats['total_repositories']}")

    for repo in stats['repositories']:
        print(f"\n📁 {repo['repo_name']}:")
        print(f"  Total files: {repo['total_files']}")
        print(f"  Lines of code: {repo['total_lines_of_code']:,}")
        print(f"  Relationships: {repo['total_relationships']}")
        print(f"  Last analyzed: {repo['analysis_date']}")

        if repo.get('file_types'):
            print(f"  File types:")
            for file_type, count in sorted(repo['file_types'].items(),
                                          key=lambda x: x[1],
                                          reverse=True)[:5]:
                print(f"    - {file_type}: {count}")


async def example_7_advanced_search():
    """Example 7: Advanced search with filters"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Advanced Search with Filters")
    print("="*80)

    indexer = CodeIndexer(
        code_base_path=".",
        output_dir="example_indexes"
    )

    # Search with filters
    print("\n🔍 Advanced search for: 'knowledge graph'")

    results = await indexer.search_code(
        query="knowledge graph implementation",
        filters={
            "file_type": "python",
            "min_confidence": 0.5
        }
    )

    if results:
        print(f"\nFound {len(results)} Python files with high confidence:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  {i}. {result['file_path']}")
            print(f"     Relevance: {result['relevance_score']:.2f}")
            print(f"     Aspects: {', '.join(result.get('relevant_snippets', [])[:3])}")
    else:
        print("No results found with specified filters")


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("CodeIndexer - Usage Examples")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Run examples
        await example_1_basic_indexing()
        await example_2_code_search()
        await example_3_similar_files()
        await example_4_file_relationships()
        await example_5_incremental_updates()
        await example_6_statistics()
        await example_7_advanced_search()

        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
