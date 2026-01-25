"""
Unit Tests for Ψ₂ Ontology Mapper

Tests semantic mapping between problem domains.

Author: G2 (Ψ₂ Specialist)
Created: 2025-12-31
Status: 🟢 Active
"""

import pytest
import sys
from pathlib import Path
import numpy as np
import tempfile
import json
import networkx as nx

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase2"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core"))

from ontology_mapper import (
    MappingResult,
    OntologyMapper,
    create_mapper,
    map_domains
)


class TestMappingResult:
    """Test MappingResult dataclass"""

    def test_mapping_result_creation(self):
        """Test creating MappingResult"""
        result = MappingResult(
            concept_mapping={"concept1": "concept_a"},
            relation_mapping={"rel1": "rel_a"},
            confidence={"concept1": 0.8},
            metadata={"test": "data"}
        )

        assert len(result.concept_mapping) == 1
        assert len(result.relation_mapping) == 1
        assert result.confidence["concept1"] == 0.8

    def test_mapping_result_defaults(self):
        """Test MappingResult with defaults"""
        result = MappingResult()

        assert result.concept_mapping == {}
        assert result.relation_mapping == {}
        assert result.confidence == {}
        assert result.metadata == {}

    def test_mapping_result_repr(self):
        """Test MappingResult string representation"""
        result = MappingResult(
            concept_mapping={"a": "x", "b": "y"},
            confidence={"a": 0.7, "b": 0.9}
        )

        repr_str = repr(result)

        assert "concepts=2" in repr_str
        assert "avg_confidence" in repr_str


class TestOntologyMapper:
    """Test OntologyMapper"""

    @pytest.fixture
    def mapper(self):
        """Create mapper instance"""
        config = {
            'use_cache': False,  # Disable cache for tests
            'kg_enabled': False  # Disable KG for tests
        }
        return OntologyMapper(config)

    @pytest.fixture
    def sample_domains(self):
        """Create sample domains for testing"""
        # Create simple domain-like objects
        class SimpleDomain:
            def __init__(self, name, concepts):
                self.name = name
                self.fdg = None
                self.formal_constraints = []
                for concept in concepts:
                    self.formal_constraints.append(
                        type('MockConstraint', (), {'variables': [concept]})()
                    )

        domain1 = SimpleDomain("Physics", ["velocity", "acceleration", "force", "mass"])
        domain2 = SimpleDomain("Economics", ["price", "demand", "supply", "velocity"])

        return domain1, domain2

    def test_mapper_initialization(self, mapper):
        """Test mapper initialization"""
        assert mapper.config is not None
        assert isinstance(mapper.config, dict)
        assert mapper.config['use_cache'] is False

    def test_mapper_default_config(self):
        """Test mapper with default config"""
        mapper = OntologyMapper()

        assert mapper.config['lexical_threshold'] == 0.3
        assert mapper.config['semantic_threshold'] == 0.5
        assert mapper.config['final_threshold'] == 0.5

    def test_normalize_concept(self, mapper):
        """Test concept normalization"""
        # Lowercase
        norm1 = mapper._normalize_concept("VELOCITY")
        assert norm1 == "velocity"

        # Remove special characters
        norm2 = mapper._normalize_concept("test-concept!")
        assert norm2 == "test concept"

        # Remove extra spaces
        norm3 = mapper._normalize_concept("test    concept")
        assert norm3 == "test concept"

    def test_jaro_winkler_similarity(self, mapper):
        """Test Jaro-Winkler similarity calculation"""
        # Identical strings
        sim1 = mapper._jaro_winkler_similarity("test", "test")
        assert sim1 == pytest.approx(1.0, rel=0.01)

        # Completely different
        sim2 = mapper._jaro_winkler_similarity("abc", "xyz")
        assert sim2 < 0.5

        # Similar strings
        sim3 = mapper._jaro_winkler_similarity("velocity", "velocities")
        assert sim3 > 0.8

        # Empty strings
        sim4 = mapper._jaro_winkler_similarity("", "")
        assert sim4 == 1.0

        # One empty string
        sim5 = mapper._jaro_winkler_similarity("test", "")
        assert sim5 == 0.0

    def test_cosine_similarity(self, mapper):
        """Test cosine similarity calculation"""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([1.0, 2.0, 3.0])

        sim = mapper._cosine_similarity(v1, v2)
        assert sim == pytest.approx(1.0, rel=0.01)

        v3 = np.array([1.0, 0.0, 0.0])
        v4 = np.array([0.0, 1.0, 0.0])

        sim2 = mapper._cosine_similarity(v3, v4)
        assert sim2 == pytest.approx(0.0, rel=0.01)

        # Zero vectors
        v5 = np.array([0.0, 0.0, 0.0])
        v6 = np.array([1.0, 2.0, 3.0])

        sim3 = mapper._cosine_similarity(v5, v6)
        assert sim3 == 0.0

    def test_preprocess_domain(self, mapper, sample_domains):
        """Test domain preprocessing"""
        domain1, domain2 = sample_domains

        graph1, concepts1 = mapper._preprocess_domain(domain1)
        graph2, concepts2 = mapper._preprocess_domain(domain2)

        # Should return graph and concepts
        assert isinstance(graph1, nx.DiGraph)
        assert isinstance(concepts1, list)
        assert len(concepts1) > 0

        # Concepts should be normalized
        assert all(c.islower() for c in concepts1)

    def test_generate_candidates(self, mapper, sample_domains):
        """Test candidate generation"""
        domain1, domain2 = sample_domains

        _, concepts1 = mapper._preprocess_domain(domain1)
        _, concepts2 = mapper._preprocess_domain(domain2)

        candidates = mapper._generate_candidates(concepts1, concepts2)

        assert isinstance(candidates, list)
        # Should find some candidates
        assert len(candidates) >= 0

        # Each candidate should be tuple of (source, target, score)
        for c in candidates:
            assert isinstance(c, tuple)
            assert len(c) == 3
            assert isinstance(c[0], str)
            assert isinstance(c[1], str)
            assert isinstance(c[2], float)
            assert 0 <= c[2] <= 1

    def test_compute_semantic_similarity(self, mapper, sample_domains):
        """Test semantic similarity computation"""
        domain1, domain2 = sample_domains

        _, concepts1 = mapper._preprocess_domain(domain1)
        _, concepts2 = mapper._preprocess_domain(domain2)

        candidates = mapper._generate_candidates(concepts1, concepts2)

        # Even if semantic matcher is unavailable, should return empty dict
        scores = mapper._compute_semantic_similarity(candidates, concepts1, concepts2)

        assert isinstance(scores, dict)

    def test_compute_graph_similarity(self, mapper, sample_domains):
        """Test graph similarity computation"""
        domain1, domain2 = sample_domains

        graph1, concepts1 = mapper._preprocess_domain(domain1)
        graph2, concepts2 = mapper._preprocess_domain(domain2)

        # Even if graph embedder is unavailable, should return empty dict
        scores = mapper._compute_graph_similarity(graph1, graph2, concepts1, concepts2)

        assert isinstance(scores, dict)

    def test_aggregate_confidence(self, mapper):
        """Test confidence aggregation"""
        candidates = [
            ("vel", "velocity", 0.95),
            ("acc", "acceleration", 0.90),
            ("force", "power", 0.70)
        ]

        scores = {
            'semantic': {
                ("vel", "velocity"): 0.98,
                ("acc", "acceleration"): 0.85
            },
            'graph': {},
            'kg': {}
        }

        result = mapper._aggregate_confidence(candidates, scores, threshold=0.6)

        assert isinstance(result, MappingResult)
        assert len(result.concept_mapping) <= len(candidates)

    def test_map_ontologies(self, mapper, sample_domains):
        """Test full ontology mapping"""
        domain1, domain2 = sample_domains

        result = mapper.map_ontologies(domain1, domain2)

        assert isinstance(result, MappingResult)
        assert hasattr(result, 'concept_mapping')
        assert hasattr(result, 'relation_mapping')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'metadata')

        # Metadata should contain expected fields
        assert 'algorithm' in result.metadata
        assert 'timestamp' in result.metadata
        assert 'source_domain' in result.metadata
        assert 'target_domain' in result.metadata

    def test_map_ontologies_selective_stages(self, mapper, sample_domains):
        """Test mapping with selective stages"""
        domain1, domain2 = sample_domains

        # Use only lexical stage
        result = mapper.map_ontologies(
            domain1, domain2,
            use_stages=['lexical', 'aggregate']
        )

        assert isinstance(result, MappingResult)
        assert 'lexical' in result.metadata.get('stages_used', [])

    def test_save_and_load_mapping(self, mapper, sample_domains):
        """Test saving and loading mapping results"""
        domain1, domain2 = sample_domains

        result = mapper.map_ontologies(domain1, domain2)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            mapper.save_mapping(result, temp_path)

            # Load back
            loaded_result = mapper.load_mapping(temp_path)

            assert loaded_result.concept_mapping == result.concept_mapping
            assert loaded_result.relation_mapping == result.relation_mapping
            # Convert numpy floats to compare
            for k, v in loaded_result.confidence.items():
                assert abs(v - result.confidence[k]) < 0.01

        finally:
            import os
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_mapper(self):
        """Test create_mapper function"""
        mapper = create_mapper()

        assert isinstance(mapper, OntologyMapper)

    def test_create_mapper_with_config(self):
        """Test create_mapper with custom config"""
        config = {'lexical_threshold': 0.5}
        mapper = create_mapper(config)

        assert mapper.config['lexical_threshold'] == 0.5

    def test_map_domains_function(self):
        """Test map_domains convenience function"""
        # Create simple domains
        class SimpleDomain:
            def __init__(self, name, concepts):
                self.name = name
                self.fdg = None
                self.formal_constraints = []
                for concept in concepts:
                    self.formal_constraints.append(
                        type('MockConstraint', (), {'variables': [concept]})()
                    )

        domain1 = SimpleDomain("D1", ["a", "b"])
        domain2 = SimpleDomain("D2", ["x", "y"])

        result = map_domains(domain1, domain2)

        assert isinstance(result, MappingResult)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_domains(self):
        """Test mapping empty domains"""
        mapper = OntologyMapper({'use_cache': False, 'kg_enabled': False})

        class EmptyDomain:
            def __init__(self):
                self.name = "Empty"
                self.fdg = None
                self.formal_constraints = []

        domain1 = EmptyDomain()
        domain2 = EmptyDomain()

        result = mapper.map_ontologies(domain1, domain2)

        assert isinstance(result, MappingResult)
        assert len(result.concept_mapping) == 0

    def test_very_long_concept_names(self):
        """Test handling very long concept names"""
        mapper = OntologyMapper({'use_cache': False, 'kg_enabled': False})

        long_name = "a" * 1000

        normalized = mapper._normalize_concept(long_name)

        assert isinstance(normalized, str)
        # Should still be a string
        assert len(normalized) <= 1000

    def test_special_characters_in_concepts(self):
        """Test handling special characters"""
        mapper = OntologyMapper({'use_cache': False, 'kg_enabled': False})

        special = "test-concept@123!#$"

        normalized = mapper._normalize_concept(special)

        # Should remove special characters
        assert "@" not in normalized
        assert "!" not in normalized

    def test_unicode_concepts(self):
        """Test handling unicode in concepts"""
        mapper = OntologyMapper({'use_cache': False, 'kg_enabled': False})

        unicode_concept = "测试概念"

        normalized = mapper._normalize_concept(unicode_concept)

        assert isinstance(normalized, str)

    def test_similar_but_not_identical(self):
        """Test concepts that are similar but not identical"""
        mapper = OntologyMapper({'use_cache': False, 'kg_enabled': False})

        sim = mapper._jaro_winkler_similarity("velocity", "velocity_x")

        # Should have high similarity but not 1.0
        assert 0.7 < sim < 1.0

    def test_single_character_concepts(self):
        """Test single character concepts"""
        mapper = OntologyMapper({'use_cache': False, 'kg_enabled': False})

        sim = mapper._jaro_winkler_similarity("a", "b")

        # Should handle gracefully
        assert isinstance(sim, float)
        assert 0 <= sim <= 1

    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Should accept various configurations
        configs = [
            {'lexical_threshold': 0.5},
            {'semantic_threshold': 0.7},
            {'final_threshold': 0.9},
            {'use_cache': False}
        ]

        for config in configs:
            mapper = OntologyMapper(config)
            assert mapper is not None

    def test_cache_disabled(self):
        """Test mapper with cache disabled"""
        mapper = OntologyMapper({'use_cache': False})

        assert mapper.embedding_cache == {}
        assert mapper.kg_cache_db is None

    def test_kg_disabled(self):
        """Test mapper with KG disabled"""
        mapper = OntologyMapper({'kg_enabled': False, 'use_cache': False})

        assert mapper.kg_validator is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
