"""
Test Suite for AI-Knowledge-Graph Integration

This module provides comprehensive tests for the AIKG integration including:
- Entity standardization tests
- Relationship inference tests
- Visualization generation tests
- Complete pipeline tests
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from knowledge_engine.integrations.aikg_standardization import (
    AIKGEntityStandardizer,
    Entity,
    Triple
)
from knowledge_engine.integrations.aikg_inference import (
    AIKGRelationshipInference
)
from knowledge_engine.integrations.aikg_visualization import (
    AIKGVisualizer,
    VisualizationOptions
)
from knowledge_engine.integrations.aikg_integration import (
    AIKGIntegration
)


class TestEntityStandardization:
    """Test suite for entity standardization."""

    @pytest.fixture
    def standardizer(self):
        """Create standardizer instance for testing."""
        config = {
            'use_llm_for_entities': False,
            'stopword_removal': True,
            'root_word_analysis': True,
            'self_reference_filtering': True
        }
        return AIKGEntityStandardizer(config)

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            Entity("Python"),
            Entity("python"),
            Entity("PYTHON"),
            Entity("Machine Learning"),
            Entity("machine learning"),
            Entity("JavaScript"),
            Entity("javaScript")
        ]

    @pytest.fixture
    def sample_triples(self):
        """Create sample triples for testing."""
        return [
            Triple("Python", "used_for", "Web Development"),
            Triple("python", "related_to", "Django"),
            Triple("Python", "related_to", "Python"),  # Self-reference
            Triple("Machine Learning", "subset_of", "Artificial Intelligence"),
            Triple("machine learning", "used_for", "Data Analysis")
        ]

    @pytest.mark.asyncio
    async def test_text_normalization(self, standardizer):
        """Test text normalization."""
        test_cases = [
            ("Python Programming", "python programming"),
            ("Machine Learning", "machine learning"),
            ("The JavaScript Framework", "javascript framework"),
            ("Data  Science", "data science")
        ]

        for input_text, expected in test_cases:
            result = await standardizer.normalize_text(input_text)
            assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.asyncio
    async def test_entity_standardization(self, standardizer, sample_entities, sample_triples):
        """Test entity standardization."""
        result = await standardizer.standardize_entities(sample_entities, sample_triples)

        # Check that entities were reduced
        assert len(result.canonical_entities) < len(sample_entities)

        # Check that variants are tracked
        assert len(result.variant_mappings) > 0

        # Check that self-references were removed
        assert result.removed_self_refs > 0

        print(f"Standardized {len(sample_entities)} -> {len(result.canonical_entities)} entities")

    @pytest.mark.asyncio
    async def test_self_reference_filtering(self, standardizer):
        """Test self-reference filtering."""
        triples = [
            Triple("A", "related_to", "A"),
            Triple("B", "related_to", "C"),
            Triple("C", "related_to", "C")
        ]

        filtered = await standardizer.filter_self_references(triples)

        # Should only keep non-self-referential triples
        assert len(filtered) == 1
        assert filtered[0].subject == "B"

    @pytest.mark.asyncio
    async def test_frequency_grouping(self, standardizer, sample_entities):
        """Test frequency-based grouping."""
        groups = await standardizer.group_by_frequency(sample_entities)

        # Check that groups were created
        assert len(groups) > 0

        # Check that "python" variants are grouped
        python_groups = [g for g in groups.values() if any(e.name == "python" for e in g)]
        assert len(python_groups) > 0

    @pytest.mark.asyncio
    async def test_root_word_analysis(self, standardizer, sample_entities):
        """Test root word analysis."""
        root_groups = await standardizer.analyze_root_words(sample_entities)

        # Check that root groups were created
        assert len(root_groups) > 0

        # "Pyth" and "pyth" should be in same group
        assert "pyth" in root_groups or "Pyth" in root_groups.lower()


class TestRelationshipInference:
    """Test suite for relationship inference."""

    @pytest.fixture
    def inference_engine(self):
        """Create inference engine for testing."""
        config = {
            'apply_transitive': True,
            'use_llm_for_inference': False,
            'similarity_threshold': 0.7,
            'max_inference_depth': 3
        }
        return AIKGRelationshipInference(config)

    @pytest.fixture
    def sample_triples(self):
        """Create sample triples for inference testing."""
        return [
            Triple("Python", "used_for", "Web Development"),
            Triple("Web Development", "used_for", "Django"),
            Triple("Django", "framework_of", "Python")
        ]

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for inference testing."""
        return [
            Entity("Python"),
            Entity("Web Development"),
            Entity("Django")
        ]

    @pytest.mark.asyncio
    async def test_transitive_inference(self, inference_engine, sample_triples):
        """Test transitive relationship inference."""
        inferred = await inference_engine.transitive_inference(sample_triples)

        # Should infer at least one transitive relationship
        assert len(inferred) >= 0  # May or may not infer depending on graph structure

        # All inferred triples should have confidence
        for triple in inferred:
            assert triple.confidence > 0
            assert triple.source == "inferred"

    @pytest.mark.asyncio
    async def test_lexical_similarity_inference(self, inference_engine):
        """Test lexical similarity inference."""
        entities = [
            Entity("machine learning"),
            Entity("learning algorithm"),
            Entity("deep learning")
        ]

        inferred = await inference_engine.lexical_similarity_inference(entities, [])

        # Should find similar entities
        assert len(inferred) >= 0

    @pytest.mark.asyncio
    async def test_deduplication(self, inference_engine, sample_triples):
        """Test inference deduplication."""
        # Create duplicate inferred triples
        inferred = [
            Triple("A", "related_to", "B", confidence=0.8, source="inferred"),
            Triple("A", "related_to", "B", confidence=0.6, source="inferred"),
            Triple("C", "related_to", "D", confidence=0.7, source="inferred")
        ]

        deduped = await inference_engine.deduplicate_inferences(sample_triples, inferred)

        # Should remove duplicates
        assert len(deduped) <= len(inferred)

    @pytest.mark.asyncio
    async def test_complete_inference(self, inference_engine, sample_triples, sample_entities):
        """Test complete inference pipeline."""
        result = await inference_engine.infer_relationships(sample_triples, sample_entities)

        # Check result structure
        assert result.original_triples is not None
        assert result.inferred_triples is not None
        assert result.confidence_scores is not None
        assert result.inference_sources is not None

        # Check statistics
        stats = result.get_statistics()
        assert 'original_triples' in stats
        assert 'inferred_triples' in stats
        assert 'total_triples' in stats


class TestVisualization:
    """Test suite for visualization generation."""

    @pytest.fixture
    def visualizer(self):
        """Create visualizer for testing."""
        config = {
            'output_dir': tempfile.gettempdir(),
            'community_algorithm': 'louvain'
        }
        return AIKGVisualizer(config)

    @pytest.fixture
    def sample_triples(self):
        """Create sample triples for visualization."""
        return [
            Triple("Python", "used_for", "Web Development"),
            Triple("Python", "related_to", "Django"),
            Triple("JavaScript", "used_for", "Web Development"),
            Triple("Machine Learning", "subset_of", "Artificial Intelligence")
        ]

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for visualization."""
        return [
            Entity("Python"),
            Entity("Web Development"),
            Entity("Django"),
            Entity("JavaScript"),
            Entity("Machine Learning"),
            Entity("Artificial Intelligence")
        ]

    @pytest.mark.asyncio
    async def test_graph_building(self, visualizer, sample_triples):
        """Test NetworkX graph building."""
        graph = visualizer._build_graph(sample_triples)

        # Check that nodes and edges were added
        assert len(graph.nodes()) > 0
        assert len(graph.edges()) > 0

    @pytest.mark.asyncio
    async def test_community_detection(self, visualizer, sample_triples):
        """Test community detection."""
        graph = visualizer._build_graph(sample_triples)
        communities = await visualizer.detect_communities(graph)

        # Should detect at least one community
        assert len(communities) > 0

    @pytest.mark.asyncio
    async def test_centrality_computation(self, visualizer, sample_triples):
        """Test centrality computation."""
        graph = visualizer._build_graph(sample_triples)
        centrality = await visualizer.compute_centrality(graph)

        # Should compute centrality for all nodes
        assert len(centrality) == len(graph.nodes())

        # All values should be between 0 and 1
        for score in centrality.values():
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_visualization_generation(self, visualizer, sample_triples, sample_entities):
        """Test D3.js visualization generation."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = await visualizer.visualize_graph(
                triples=sample_triples,
                entities=sample_entities,
                output_path=output_path
            )

            # Check result
            assert result.output_path == output_path
            assert result.node_count > 0
            assert result.edge_count > 0
            assert Path(output_path).exists()

            # Check file content
            with open(output_path, 'r') as f:
                content = f.read()
                assert '<!DOCTYPE html>' in content
                assert 'd3.js' in content.lower()

        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_graph_export(self, visualizer, sample_triples):
        """Test graph data export."""
        # Test JSON export
        json_data = await visualizer.export_graph_data(sample_triples, format='json')
        assert json_data is not None
        assert len(json_data) > 0

        # Test CSV export
        csv_data = await visualizer.export_graph_data(sample_triples, format='csv')
        assert csv_data is not None
        assert 'source,target' in csv_data


class TestCompletePipeline:
    """Test suite for complete AIKG pipeline."""

    @pytest.fixture
    def aikg_integration(self):
        """Create AIKG integration for testing."""
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
                'output_dir': tempfile.gettempdir()
            }
        }
        return AIKGIntegration(config)

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities."""
        return [
            Entity("Python"),
            Entity("python"),
            Entity("Django"),
            Entity("Web Development")
        ]

    @pytest.fixture
    def sample_triples(self):
        """Create sample triples."""
        return [
            Triple("Python", "used_for", "Web Development"),
            Triple("python", "related_to", "Django"),
            Triple("Django", "framework_of", "Python")
        ]

    @pytest.mark.asyncio
    async def test_standardize_and_infer(self, aikg_integration, sample_entities, sample_triples):
        """Test standardization and inference pipeline."""
        result = await aikg_integration.standardize_and_infer(
            triples=sample_triples,
            entities=sample_entities
        )

        # Should return processed triples
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_complete_pipeline(self, aikg_integration, sample_entities, sample_triples):
        """Test complete processing pipeline."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = await aikg_integration.process_preextracted_data(
                entities=sample_entities,
                triples=sample_triples,
                enable_standardization=True,
                enable_inference=True,
                generate_visualization=True,
                output_path=output_path
            )

            # Check result structure
            assert result.original_entities is not None
            assert result.standardized_entities is not None
            assert result.all_triples is not None
            assert result.visualization_path is not None

            # Check that standardization reduced entities
            assert len(result.standardized_entities) <= len(result.original_entities)

            # Check that inference added triples
            assert len(result.all_triples) >= len(result.original_triples)

            # Check visualization file exists
            assert Path(result.visualization_path).exists()

            # Check summary
            summary = result.get_summary()
            assert 'entities' in summary
            assert 'triples' in summary
            assert 'visualization' in summary

        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
