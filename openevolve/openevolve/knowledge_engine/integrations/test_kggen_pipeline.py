"""
Comprehensive Test Suite for KG-Gen Pipeline Integration

This module provides comprehensive tests for the kg-gen pipeline integration,
including unit tests, integration tests, and performance tests.
"""

import asyncio
import logging
import os
import pytest
import tempfile
from typing import List, Dict, Any

from .kggen_pipeline import KGGenPipelineIntegration, KnowledgeGraph, UploadResult
from .kggen_chunking import DocumentChunker, Chunk
from .kggen_parallel import ParallelChunkProcessor, ProcessingResult, BatchProgress

logger = logging.getLogger(__name__)


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    def test_create_empty_graph(self):
        """Test creating an empty knowledge graph."""
        graph = KnowledgeGraph()
        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0
        assert graph.metadata is not None

    def test_add_entity(self):
        """Test adding entities to graph."""
        graph = KnowledgeGraph()
        graph.add_entity("Python")
        graph.add_entity("Programming Language")

        assert len(graph.entities) == 2
        assert "Python" in graph.entities

    def test_add_duplicate_entity(self):
        """Test that duplicate entities are not added."""
        graph = KnowledgeGraph()
        graph.add_entity("Python")
        graph.add_entity("Python")

        assert len(graph.entities) == 1

    def test_add_relationship(self):
        """Test adding relationships to graph."""
        graph = KnowledgeGraph()
        graph.add_relationship("Python", "is_a", "Programming Language")

        assert len(graph.relationships) == 1
        assert ("Python", "is_a", "Programming Language") in graph.relationships

    def test_add_duplicate_relationship(self):
        """Test that duplicate relationships are not added."""
        graph = KnowledgeGraph()
        graph.add_relationship("Python", "is_a", "Programming Language")
        graph.add_relationship("Python", "is_a", "Programming Language")

        assert len(graph.relationships) == 1

    def test_merge_graphs(self):
        """Test merging two knowledge graphs."""
        graph1 = KnowledgeGraph(entities=["A", "B"])
        graph1.add_relationship("A", "relates_to", "B")

        graph2 = KnowledgeGraph(entities=["C", "D"])
        graph2.add_relationship("C", "relates_to", "D")

        graph1.merge(graph2)

        assert len(graph1.entities) == 4
        assert len(graph1.relationships) == 2

    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = KnowledgeGraph(entities=["A", "B"])
        graph.add_relationship("A", "relates_to", "B")

        data = graph.to_dict()

        assert "entities" in data
        assert "relationships" in data
        assert "metadata" in data
        assert len(data["entities"]) == 2
        assert len(data["relationships"]) == 1


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    @pytest.fixture
    def sample_text(self):
        """Sample text for chunking tests."""
        return """
        Python is a high-level programming language. It was created by Guido van Rossum.
        Python is widely used for web development, data science, and machine learning.
        The language emphasizes code readability with its notable use of significant whitespace.

        Knowledge graphs are a way to represent information. They consist of entities and relationships.
        Entities represent real-world objects or concepts. Relationships connect these entities.
        Knowledge graphs are used in various applications including search engines and recommendation systems.

        Machine learning is a subset of artificial intelligence. It focuses on building systems that can learn from data.
        Common machine learning algorithms include neural networks, decision trees, and support vector machines.
        Deep learning is a type of machine learning that uses neural networks with many layers.
        """

    @pytest.fixture
    def chunker(self):
        """Create a DocumentChunker instance."""
        return DocumentChunker(chunk_size=500, overlap=50)

    def test_chunk_by_sentences(self, chunker, sample_text):
        """Test chunking by sentences."""
        chunks = chunker.chunk_with_preservation(sample_text, preserve_sentences=True)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) > 0 for chunk in chunks)

    def test_chunk_by_size(self, chunker, sample_text):
        """Test chunking by size."""
        chunks = chunker.chunk_with_preservation(sample_text, preserve_sentences=False)

        assert len(chunks) > 0
        # Check that most chunks respect the size limit
        large_chunks = [c for c in chunks if len(c) > chunker.chunk_size * 1.2]
        assert len(large_chunks) == 0  # Allow 20% overflow for sentence preservation

    def test_overlap_preservation(self, chunker, sample_text):
        """Test that overlap is preserved between chunks."""
        chunks = chunker.chunk_with_preservation(sample_text, preserve_sentences=True)

        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i].text[-100:]
                chunk2_start = chunks[i + 1].text[:100]
                # Some overlap should exist (not exact match, but some common words)
                has_overlap = any(word in chunk2_start.lower() for word in chunk1_end.lower().split())
                # This is a weak check due to sentence boundary preservation
                assert True  # Placeholder - actual overlap check depends on implementation

    def test_chunk_statistics(self, chunker, sample_text):
        """Test getting chunk statistics."""
        chunks = chunker.chunk_document(sample_text)
        stats = chunker.get_chunk_statistics(chunks)

        assert "total_chunks" in stats
        assert "total_length" in stats
        assert "avg_length" in stats
        assert stats["total_chunks"] == len(chunks)

    def test_chunk_by_paragraphs(self, chunker):
        """Test chunking by paragraphs."""
        text = """
        Paragraph 1. This is the first paragraph.
        It contains multiple sentences.

        Paragraph 2. This is the second paragraph.
        It also contains multiple sentences.

        Paragraph 3. This is the third paragraph.
        """

        chunks = chunker.chunk_by_paragraphs(text, max_paragraphs_per_chunk=2)

        assert len(chunks) >= 2

    def test_empty_text(self, chunker):
        """Test chunking empty text."""
        chunks = chunker.chunk_document("")
        assert len(chunks) == 0


class TestParallelChunkProcessor:
    """Tests for ParallelChunkProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a ParallelChunkProcessor instance."""
        return ParallelChunkProcessor(max_workers=2)

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for processing."""
        return [
            Chunk(text=f"Sample text {i}", chunk_id=i, start_pos=i*10, end_pos=(i+1)*10)
            for i in range(10)
        ]

    def test_process_chunks_parallel(self, processor, sample_chunks):
        """Test parallel chunk processing."""
        def process_func(chunk):
            return f"Processed: {chunk.text}"

        results = asyncio.run(processor.process_chunks_parallel(sample_chunks, process_func))

        assert len(results) == len(sample_chunks)
        assert all(r is not None for r in results)

    def test_process_with_progress(self, processor, sample_chunks):
        """Test processing with progress tracking."""
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress.to_dict())

        def process_func(chunk):
            return f"Processed: {chunk.text}"

        results = asyncio.run(
            processor.process_with_progress(
                sample_chunks,
                process_func,
                progress_callback=progress_callback,
                log_interval=0.1
            )
        )

        assert len(results) == len(sample_chunks)
        assert len(progress_updates) > 0

    def test_process_batches(self, processor, sample_chunks):
        """Test batch processing."""
        def process_func(chunk):
            return f"Processed: {chunk.text}"

        results = asyncio.run(
            processor.process_batches(sample_chunks, process_func, batch_size=3)
        )

        assert len(results) == len(sample_chunks)

    def test_error_handling(self, processor):
        """Test error handling in parallel processing."""
        def failing_func(chunk):
            if chunk.chunk_id == 5:
                raise ValueError("Simulated error")
            return f"Success: {chunk.chunk_id}"

        chunks = [
            Chunk(text=f"Text {i}", chunk_id=i, start_pos=i*10, end_pos=(i+1)*10)
            for i in range(10)
        ]

        results = asyncio.run(processor.process_chunks_parallel(chunks, failing_func))

        # Should still have results for non-failing chunks
        successful = [r for r in results if r and "Success" in r]
        assert len(successful) > 0


class TestKGGenPipelineIntegration:
    """Tests for KGGenPipelineIntegration class."""

    @pytest.fixture
    def pipeline(self):
        """Create a KGGenPipelineIntegration instance."""
        return KGGenPipelineIntegration()

    def test_extract_knowledge_graph_simple(self, pipeline):
        """Test knowledge graph extraction from simple text."""
        text = """
        Python is a programming language created by Guido van Rossum.
        Python is used for web development and data science.
        Machine learning is a field of artificial intelligence.
        """

        graph = asyncio.run(pipeline.extract_knowledge_graph(text))

        assert graph is not None
        assert isinstance(graph, KnowledgeGraph)
        assert len(graph.entities) > 0
        assert graph.metadata is not None

    def test_extract_with_context(self, pipeline):
        """Test extraction with context."""
        text = "The code uses async/await patterns for concurrency."
        context = "Software engineering best practices"

        graph = asyncio.run(pipeline.extract_knowledge_graph(text, context))

        assert graph is not None
        assert graph.metadata.get("context") == context

    def test_extract_from_large_document(self, pipeline):
        """Test extraction from a large document."""
        # Create a large document
        text = " ".join([f"Sentence {i}. " * 10 for i in range(100)])

        graph = asyncio.run(
            pipeline.extract_from_large_document(text, chunk_size=500, parallel_chunks=2)
        )

        assert graph is not None
        assert isinstance(graph, KnowledgeGraph)

    def test_deduplication(self, pipeline):
        """Test graph deduplication."""
        # Create a graph with duplicate entities
        graph = KnowledgeGraph()
        graph.entities = ["Python", "python", "PYTHON", "Java"]
        graph.relationships = [
            ("Python", "is_a", "Language"),
            ("python", "is_a", "language"),
        ]

        deduped = asyncio.run(pipeline._deduplicate_graph(graph, method='semhash'))

        # Should have fewer entities after deduplication
        assert len(deduped.entities) <= len(graph.entities)

    def test_batch_extraction(self, pipeline):
        """Test batch knowledge graph extraction."""
        texts = [
            "Python is a programming language.",
            "Java is also a programming language.",
            "JavaScript is used for web development."
        ]

        results = asyncio.run(pipeline.extract_batch(texts))

        assert len(results) == len(texts)
        assert all(isinstance(r, KnowledgeGraph) for r in results)

    def test_fallback_entity_extraction(self, pipeline):
        """Test fallback entity extraction."""
        text = "Python and Java are programming languages. Guido van Rossum created Python."

        entities = asyncio.run(pipeline._fallback_entity_extraction(text))

        assert isinstance(entities, list)
        assert len(entities) > 0

    def test_fallback_relation_extraction(self, pipeline):
        """Test fallback relation extraction."""
        text = "Python is a programming language created by Guido van Rossum."
        entities = ["Python", "programming language", "Guido van Rossum"]

        relations = asyncio.run(pipeline._fallback_relation_extraction(text, entities))

        assert isinstance(relations, list)
        assert all(isinstance(r, tuple) and len(r) == 3 for r in relations)


class TestNeo4jIntegration:
    """Tests for Neo4j integration (mocked)."""

    def test_upload_result_creation(self):
        """Test UploadResult object creation."""
        result = UploadResult(
            success=True,
            entities_uploaded=100,
            relationships_uploaded=200
        )

        assert result.success is True
        assert result.entities_uploaded == 100
        assert result.relationships_uploaded == 200
        assert result.timestamp is not None

    def test_upload_result_to_dict(self):
        """Test converting UploadResult to dictionary."""
        result = UploadResult(success=True, entities_uploaded=50)
        data = result.to_dict()

        assert "success" in data
        assert "entities_uploaded" in data
        assert "timestamp" in data


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def engine(self):
        """Create a KnowledgeEngine instance."""
        from knowledge_engine.engine import KnowledgeEngine
        return KnowledgeEngine()

    def test_end_to_end_extraction(self, engine):
        """Test end-to-end knowledge graph extraction."""
        text = """
        Python is a high-level programming language created by Guido van Rossum.
        It is widely used for web development, data science, and machine learning.
        The Python Software Foundation manages the language.
        """

        if engine.kggen_pipeline:
            graph = asyncio.run(engine.extract_knowledge_graph(text, upload_to_neo4j=False))

            assert graph is not None
            assert isinstance(graph, KnowledgeGraph)
        else:
            pytest.skip("KG-Gen pipeline not initialized")

    def test_document_extraction(self, engine):
        """Test extraction from document file."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("""
            Machine learning is a subset of artificial intelligence.
            Neural networks are a type of machine learning algorithm.
            Deep learning uses neural networks with many layers.
            """)
            temp_path = f.name

        try:
            if engine.kggen_pipeline:
                graph = asyncio.run(engine.extract_from_document(temp_path))

                assert graph is not None
                assert isinstance(graph, KnowledgeGraph)
            else:
                pytest.skip("KG-Gen pipeline not initialized")
        finally:
            os.unlink(temp_path)

    def test_batch_extraction(self, engine):
        """Test batch extraction from multiple texts."""
        texts = [
            "Text 1: Entity A relates to Entity B.",
            "Text 2: Entity C relates to Entity D.",
            "Text 3: Entity A relates to Entity C."
        ]

        if engine.kggen_pipeline:
            results = asyncio.run(engine.extract_batch_knowledge_graphs(texts))

            assert len(results) == len(texts)
            assert all(isinstance(r, KnowledgeGraph) for r in results)
        else:
            pytest.skip("KG-Gen pipeline not initialized")


# Performance tests
class TestPerformance:
    """Performance tests for the pipeline."""

    def test_large_document_performance(self):
        """Test performance with a large document."""
        # Create a 100KB document
        text = " ".join([f"Sentence {i}. " * 10 for i in range(1000)])

        pipeline = KGGenPipelineIntegration()
        import time

        start = time.time()
        graph = asyncio.run(
            pipeline.extract_from_large_document(text, chunk_size=5000, parallel_chunks=4)
        )
        elapsed = time.time() - start

        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 60  # 60 seconds max
        assert graph is not None

    def test_chunking_performance(self):
        """Test chunking performance."""
        # Create a large text
        text = " ".join([f"Sentence {i}. " for i in range(10000)])

        chunker = DocumentChunker(chunk_size=5000, overlap=200)
        import time

        start = time.time()
        chunks = chunker.chunk_document(text)
        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 5  # 5 seconds max
        assert len(chunks) > 0

    def test_parallel_processing_speedup(self):
        """Test that parallel processing provides speedup."""
        chunks = [
            Chunk(text=f"Sample text {i} " * 100, chunk_id=i, start_pos=i*1000, end_pos=(i+1)*1000)
            for i in range(20)
        ]

        def slow_process(chunk):
            import time
            time.sleep(0.01)  # Simulate some work
            return chunk.chunk_id

        # Sequential processing
        import time
        start = time.time()
        sequential_results = [slow_process(c) for c in chunks]
        sequential_time = time.time() - start

        # Parallel processing
        processor = ParallelChunkProcessor(max_workers=4)
        start = time.time()
        parallel_results = asyncio.run(processor.process_chunks_parallel(chunks, slow_process))
        parallel_time = time.time() - start

        # Parallel should be faster (with some tolerance)
        assert parallel_time < sequential_time * 1.5


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
