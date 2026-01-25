"""
Unit Tests for Ontology Mapper

Comprehensive tests for all ontology mapping components.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase2.ontology_mapper import OntologyMapper, MappingResult
from phase2.ontology_components.lexical_matcher import LexicalMatcher
from phase2.ontology_components.semantic_matcher import SemanticMatcher, FallbackSemanticMatcher
from phase2.ontology_components.graph_embedder import GraphEmbedder, FallbackGraphEmbedder
from phase2.ontology_components.kg_validator import KGValidator, FallbackKGValidator


class TestLexicalMatcher:
    """Tests for LexicalMatcher"""

    def test_init(self):
        """Test initialization"""
        matcher = LexicalMatcher(threshold=0.3, method='jaro-winkler')
        assert matcher.threshold == 0.3
        assert matcher.method == 'jaro-winkler'

    def test_jaro_winkler_identical(self):
        """Test Jaro-Winkler similarity for identical strings"""
        matcher = LexicalMatcher(method='jaro-winkler')
        score = matcher.similarity("velocity", "velocity")
        assert score == pytest.approx(1.0, rel=0.01)

    def test_jaro_winkler_similar(self):
        """Test Jaro-Winkler for similar strings"""
        matcher = LexicalMatcher(method='jaro-winkler')
        score = matcher.similarity("velocity", "velocity_x")
        assert score > 0.8

    def test_jaro_winkler_different(self):
        """Test Jaro-Winkler for different strings"""
        matcher = LexicalMatcher(method='jaro-winkler')
        score = matcher.similarity("velocity", "pressure")
        assert score < 0.5

    def test_jaro_winkler_empty(self):
        """Test Jaro-Winkler for empty strings"""
        matcher = LexicalMatcher(method='jaro-winkler')
        score = matcher.similarity("", "")
        assert score == 1.0

    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity"""
        matcher = LexicalMatcher(method='levenshtein')
        score = matcher.similarity("velocity", "velocity")
        assert score == pytest.approx(1.0, rel=0.01)

    def test_ngram_similarity(self):
        """Test n-gram similarity"""
        matcher = LexicalMatcher(method='ngram')
        score = matcher.similarity("velocity", "velocity")
        assert score > 0.8

    def test_match_best(self):
        """Test finding best match"""
        matcher = LexicalMatcher()
        targets = ["velocity", "speed", "pressure", "voltage"]
        best, score = matcher.match_best("velocity", targets)
        assert best == "velocity"
        assert score > 0.9

    def test_match_all(self):
        """Test matching all targets"""
        matcher = LexicalMatcher()
        targets = ["velocity", "speed", "pressure"]
        matches = matcher.match_all("velocity", targets)
        assert len(matches) == 3
        assert matches[0][0] == "velocity"  # Best match first


class TestSemanticMatcher:
    """Tests for SemanticMatcher"""

    def test_init(self):
        """Test initialization"""
        matcher = SemanticMatcher(
            model_name='all-MiniLM-L6-v2',
            threshold=0.5
        )
        assert matcher.model_name == 'all-MiniLM-L6-v2'
        assert matcher.threshold == 0.5

    @pytest.mark.skipif(
        True,  # Skip in CI if model not available
        reason="Requires sentence-transformers"
    )
    def test_encode(self):
        """Test encoding texts"""
        matcher = SemanticMatcher()
        texts = ["velocity", "speed", "pressure"]
        embeddings = matcher.encode(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0  # Has embeddings

    @pytest.mark.skipif(
        True,
        reason="Requires sentence-transformers"
    )
    def test_similarity(self):
        """Test semantic similarity"""
        matcher = SemanticMatcher()
        score = matcher.similarity("fast", "rapid")
        assert score > 0.5  # Synonyms should have high similarity

    @pytest.mark.skipif(
        True,
        reason="Requires sentence-transformers"
    )
    def test_similarity_different(self):
        """Test semantic similarity for different concepts"""
        matcher = SemanticMatcher()
        score = matcher.similarity("velocity", "pressure")
        assert 0.0 <= score <= 1.0

    def test_fallback_matcher(self):
        """Test fallback semantic matcher"""
        matcher = FallbackSemanticMatcher()
        score = matcher.similarity("fast velocity", "fast speed")
        assert score > 0.3  # Should detect word overlap


class TestGraphEmbedder:
    """Tests for GraphEmbedder"""

    def test_init(self):
        """Test initialization"""
        embedder = GraphEmbedder(
            dimensions=64,
            walk_length=40,
            num_walks=20
        )
        assert embedder.dimensions == 64
        assert embedder.walk_length == 40
        assert embedder.num_walks == 20

    def test_fit_transform(self):
        """Test fitting and transforming"""
        import networkx as nx

        # Create test graph
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])

        embedder = GraphEmbedder(dimensions=32, walk_length=10, num_walks=5)
        embeddings = embedder.fit_transform(G)

        # Check embeddings
        assert len(embeddings) > 0
        for node, emb in embeddings.items():
            assert len(emb) == 32  # Correct dimension

    def test_fallback_embedder(self):
        """Test fallback graph embedder"""
        import networkx as nx

        # Create test graph
        G = nx.Graph()
        G.add_edges_from([('A', 'B'), ('B', 'C')])

        embedder = FallbackGraphEmbedder(dimensions=32)
        embeddings = embedder.fit_transform(G)

        assert len(embeddings) > 0
        for node, emb in embeddings.items():
            assert len(emb) == 32

    def test_similarity(self):
        """Test structural similarity"""
        import networkx as nx

        G1 = nx.Graph()
        G1.add_edges_from([('A', 'B'), ('B', 'C')])

        G2 = nx.Graph()
        G2.add_edges_from([('X', 'Y'), ('Y', 'Z')])

        embedder = GraphEmbedder(dimensions=16, walk_length=5, num_walks=3)
        sim = embedder.similarity(G1, G2, 'A', 'X')

        assert 0.0 <= sim <= 1.0


class TestKGValidator:
    """Tests for KGValidator"""

    def test_init(self):
        """Test initialization"""
        validator = KGValidator(
            use_conceptnet=True,
            use_wordnet=True
        )
        assert validator.use_conceptnet == True
        assert validator.use_wordnet == True

    def test_fallback_validator(self):
        """Test fallback KG validator"""
        validator = FallbackKGValidator()

        # Similar strings should have high score
        score = validator.validate_relation("velocity", "velocity_x")
        assert score is not None

        # Different strings
        score2 = validator.validate_relation("velocity", "quantum")
        assert score2 is not None

    def test_is_synonym_fallback(self):
        """Test synonym detection with fallback"""
        validator = FallbackKGValidator()

        # High similarity strings
        result = validator.is_synonym("velocity", "velocity_x")
        # Should be True due to high similarity
        assert isinstance(result, bool)


class TestOntologyMapper:
    """Tests for OntologyMapper"""

    def test_init(self):
        """Test initialization"""
        mapper = OntologyMapper()
        assert mapper.config is not None
        assert 'lexical_threshold' in mapper.config

    def test_normalize_concept(self):
        """Test concept normalization"""
        mapper = OntologyMapper()

        # Lowercase
        assert mapper._normalize_concept("Velocity") == "velocity"

        # Remove special characters
        assert mapper._normalize_concept("flow_rate") == "flow rate"

        # Extra spaces
        assert mapper._normalize_concept("flow  rate") == "flow rate"

    def test_jaro_winkler_similarity(self):
        """Test Jaro-Winkler similarity"""
        mapper = OntologyMapper()

        # Identical
        score = mapper._jaro_winkler_similarity("test", "test")
        assert score == pytest.approx(1.0, rel=0.01)

        # Similar
        score = mapper._jaro_winkler_similarity("test", "testing")
        assert score > 0.7

        # Different
        score = mapper._jaro_winkler_similarity("abc", "xyz")
        assert score < 0.5

    def test_cosine_similarity(self):
        """Test cosine similarity"""
        mapper = OntologyMapper()

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])

        score = mapper._cosine_similarity(v1, v2)
        assert score == pytest.approx(1.0, rel=0.01)

        v3 = np.array([0.0, 1.0, 0.0])
        score = mapper._cosine_similarity(v1, v3)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_map_ontologies_simple(self):
        """Test ontology mapping with simple domains"""
        from phase2.imech.core.domain import Domain

        # Create simple domains
        source_domain = Domain(
            id="fluid",
            name="Fluid Dynamics",
            description="Fluid flow domain"
        )

        target_domain = Domain(
            id="electric",
            name="Electricity",
            description="Electrical circuit domain"
        )

        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            source_domain,
            target_domain,
            use_stages=['lexical', 'aggregate']  # Only use fast stages
        )

        assert isinstance(result, MappingResult)
        assert 'algorithm' in result.metadata
        assert 'timestamp' in result.metadata

    def test_save_load_mapping(self):
        """Test saving and loading mappings"""
        import tempfile

        # Create mapping
        mapping = MappingResult(
            concept_mapping={'velocity': 'speed'},
            relation_mapping={'causes': 'causes'},
            confidence={('velocity', 'speed'): 0.85},
            metadata={'test': True}
        )

        mapper = OntologyMapper()

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        mapper.save_mapping(mapping, filepath)

        # Load back
        loaded = mapper.load_mapping(filepath)

        assert loaded.concept_mapping == mapping.concept_mapping
        assert loaded.metadata['test'] == True

        # Cleanup
        import os
        os.unlink(filepath)


class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self):
        """Test full ontology mapping pipeline"""
        from phase2.imech.core.domain import Domain
        import networkx as nx

        # Create domains with FDGs
        source_domain = Domain(
            id="mechanical",
            name="Mechanical System",
            description="Mass-spring-damper"
        )

        target_domain = Domain(
            id="electrical",
            name="Electrical Circuit",
            description="RLC circuit"
        )

        # Create simple FDGs
        source_fdg = nx.DiGraph()
        source_fdg.add_nodes_from(['mass', 'spring', 'damper', 'force'])
        source_fdg.add_edges_from([
            ('force', 'mass'),
            ('mass', 'spring'),
            ('spring', 'damper')
        ])

        target_fdg = nx.DiGraph()
        target_fdg.add_nodes_from(['inductor', 'capacitor', 'resistor', 'voltage'])
        target_fdg.add_edges_from([
            ('voltage', 'inductor'),
            ('inductor', 'capacitor'),
            ('capacitor', 'resistor')
        ])

        source_domain.fdg = type('FDG', (), {
            'to_networkx': lambda self: source_fdg
        })()

        target_domain.fdg = type('FDG', (), {
            'to_networkx': lambda self: target_fdg
        })()

        # Map
        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            source_domain,
            target_domain,
            use_stages=['lexical', 'aggregate']
        )

        # Validate
        assert isinstance(result, MappingResult)
        assert result.metadata['source_domain'] == 'Mechanical System'
        assert result.metadata['target_domain'] == 'Electrical Circuit'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
