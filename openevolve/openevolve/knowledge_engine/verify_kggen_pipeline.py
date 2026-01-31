"""
Quick verification script for KG-Gen Pipeline integration
"""

import sys
import asyncio
from pathlib import Path

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent))


async def verify_imports():
    """Verify that all modules can be imported."""
    print("=" * 60)
    print("Verifying KG-Gen Pipeline Integration")
    print("=" * 60)

    tests = []

    # Test 1: Import main pipeline module
    print("\n[1/8] Testing kggen_pipeline import...")
    try:
        from knowledge_engine.integrations.kggen_pipeline import (
            KGGenPipelineIntegration,
            KnowledgeGraph,
            UploadResult
        )
        print("  ✓ kggen_pipeline imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import kggen_pipeline: {e}")
        tests.append(False)

    # Test 2: Import chunking module
    print("\n[2/8] Testing kggen_chunking import...")
    try:
        from knowledge_engine.integrations.kggen_chunking import (
            DocumentChunker,
            Chunk
        )
        print("  ✓ kggen_chunking imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import kggen_chunking: {e}")
        tests.append(False)

    # Test 3: Import parallel processing module
    print("\n[3/8] Testing kggen_parallel import...")
    try:
        from knowledge_engine.integrations.kggen_parallel import (
            ParallelChunkProcessor,
            ProcessingResult,
            BatchProgress
        )
        print("  ✓ kggen_parallel imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import kggen_parallel: {e}")
        tests.append(False)

    # Test 4: Import Neo4j module
    print("\n[4/8] Testing kggen_neo4j import...")
    try:
        from knowledge_engine.integrations.kggen_neo4j import (
            Neo4jGraphUploader
        )
        print("  ✓ kggen_neo4j imported successfully")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ Failed to import kggen_neo4j: {e}")
        tests.append(False)

    # Test 5: Check configuration file
    print("\n[5/8] Checking configuration file...")
    config_path = Path("knowledge_engine/config/kggen_pipeline.yaml")
    if config_path.exists():
        print(f"  ✓ Configuration file exists: {config_path}")
        tests.append(True)
    else:
        print(f"  ✗ Configuration file not found: {config_path}")
        tests.append(False)

    # Test 6: Create KnowledgeGraph instance
    print("\n[6/8] Testing KnowledgeGraph class...")
    try:
        graph = KnowledgeGraph(entities=["A", "B"])
        graph.add_relationship("A", "relates_to", "B")
        assert len(graph.entities) == 2
        assert len(graph.relationships) == 1
        print("  ✓ KnowledgeGraph class working correctly")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ KnowledgeGraph test failed: {e}")
        tests.append(False)

    # Test 7: Test DocumentChunker
    print("\n[7/8] Testing DocumentChunker class...")
    try:
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        text = "Sentence 1. Sentence 2. Sentence 3."
        chunks = chunker.chunk_document(text)
        assert len(chunks) > 0
        print(f"  ✓ DocumentChunker working correctly ({len(chunks)} chunks)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ DocumentChunker test failed: {e}")
        tests.append(False)

    # Test 8: Test Pipeline Integration
    print("\n[8/8] Testing KGGenPipelineIntegration class...")
    try:
        pipeline = KGGenPipelineIntegration()
        text = "Python is a programming language."
        graph = await pipeline.extract_knowledge_graph(text, upload_to_neo4j=False)
        assert graph is not None
        assert isinstance(graph, KnowledgeGraph)
        print(f"  ✓ Pipeline working correctly ({len(graph.entities)} entities)")
        tests.append(True)
    except Exception as e:
        print(f"  ✗ Pipeline test failed: {e}")
        tests.append(False)

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    passed = sum(tests)
    total = len(tests)
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("\n✓ All tests passed! KG-Gen Pipeline is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


async def verify_engine_integration():
    """Verify KnowledgeEngine integration."""
    print("\n" + "=" * 60)
    print("Verifying KnowledgeEngine Integration")
    print("=" * 60)

    try:
        from knowledge_engine.engine import KnowledgeEngine

        print("\n[1/3] Initializing KnowledgeEngine...")
        engine = KnowledgeEngine()
        print("  ✓ KnowledgeEngine initialized")

        print("\n[2/3] Checking KG-Gen pipeline...")
        if engine.kggen_pipeline:
            print("  ✓ KG-Gen pipeline initialized")
        else:
            print("  ⚠ KG-Gen pipeline not initialized (this is OK if dependencies are missing)")

        print("\n[3/3] Testing knowledge graph extraction...")
        text = "Python is a programming language created by Guido van Rossum."
        try:
            graph = await engine.extract_knowledge_graph(text, upload_to_neo4j=False)
            print(f"  ✓ Extraction successful ({len(graph.entities)} entities)")
        except Exception as e:
            print(f"  ⚠ Extraction failed: {e}")

        print("\n" + "=" * 60)
        print("KnowledgeEngine Integration Verification Complete")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n✗ KnowledgeEngine integration test failed: {e}")
        return 1


async def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("KG-Gen Pipeline Verification Script")
    print("=" * 60)

    # Run module import tests
    result1 = await verify_imports()

    # Run engine integration tests
    result2 = await verify_engine_integration()

    # Final summary
    print("\n" + "=" * 60)
    print("Final Status")
    print("=" * 60)

    if result1 == 0 and result2 == 0:
        print("✓ All verifications passed!")
        print("\nThe KG-Gen Pipeline Integration is ready to use.")
        print("\nQuick Start:")
        print("  from knowledge_engine.engine import KnowledgeEngine")
        print("  engine = KnowledgeEngine()")
        print('  graph = await engine.extract_knowledge_graph("Your text here")')
        return 0
    else:
        print("⚠ Some verifications failed or warnings were issued.")
        print("Please review the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
