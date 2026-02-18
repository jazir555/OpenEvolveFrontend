"""
Comprehensive Tests for Knowledge Engine Completion

Tests all the new completion modules:
1. Embedding Service
2. Cloud Storage Backends
3. Full-Featured Backends
4. Confidence Scorer
5. Strategy Recommender
6. Complete Integration
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Test embedding service
from knowledge_engine.embedding_service import (
    EmbeddingService,
    EmbeddingConfig,
    create_embedding_service,
    get_default_embedding_service
)

# Test confidence scorer
from knowledge_engine.confidence_scorer import (
    ConfidenceScorer,
    ConfidenceFactors,
    calculate_confidence,
    get_confidence_scorer
)

# Test strategy recommender
from knowledge_engine.core.strategy_recommender_complete import (
    StrategyRecommendation,
    KeywordBasedRecommender,
    DomainBasedRecommender,
    ComplexityBasedRecommender,
    HistoricalPerformanceRecommender,
    EnsembleStrategySelector,
    recommend_strategy
)

# Test full-featured backends
from knowledge_engine.core.backends.full_featured_backends import (
    FullFeaturedInMemoryBackend,
    create_full_featured_backend
)

# Test complete integration
from knowledge_engine.__complete__ import (
    CompletedKnowledgeEngine,
    create_complete_knowledge_engine
)


class TestEmbeddingService(unittest.TestCase):
    """Test the embedding service."""
    
    def setUp(self):
        self.service = EmbeddingService()
    
    def test_embed_text(self):
        """Test text embedding generation."""
        text = "This is a test sentence."
        embedding = self.service.embed_text(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), self.service.config.dimensions)
        
        # Check normalization
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_embed_empty_text(self):
        """Test embedding of empty text."""
        embedding = self.service.embed_text("")
        self.assertEqual(np.linalg.norm(embedding), 0.0)
    
    def test_embed_batch(self):
        """Test batch embedding."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence."
        ]
        embeddings = self.service.embed_batch(texts)
        
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertEqual(embeddings.shape[1], self.service.config.dimensions)
    
    def test_similarity(self):
        """Test similarity computation."""
        text1 = "Machine learning is fascinating."
        text2 = "Deep learning uses neural networks."
        text3 = "The weather is nice today."
        
        emb1 = self.service.embed_text(text1)
        emb2 = self.service.embed_text(text2)
        emb3 = self.service.embed_text(text3)
        
        sim12 = self.service.compute_similarity(emb1, emb2)
        sim13 = self.service.compute_similarity(emb1, emb3)
        
        # Related texts should have higher similarity
        self.assertGreater(sim12, sim13)
    
    def test_caching(self):
        """Test embedding caching."""
        text = "Test caching functionality."
        
        # First call should miss cache
        emb1 = self.service.embed_text(text)
        misses_before = self.service._cache_misses
        
        # Second call should hit cache
        emb2 = self.service.embed_text(text)
        hits_after = self.service._cache_hits
        
        self.assertEqual(misses_before, self.service._cache_misses)
        self.assertGreater(hits_after, 0)
        np.testing.assert_array_equal(emb1, emb2)
    
    def test_stats(self):
        """Test statistics collection."""
        stats = self.service.get_stats()
        
        self.assertIn('model', stats)
        self.assertIn('dimensions', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('cache_hit_rate', stats)


class TestConfidenceScorer(unittest.TestCase):
    """Test the confidence scoring system."""
    
    def setUp(self):
        self.scorer = ConfidenceScorer()
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        confidence, factors = self.scorer.calculate_confidence(
            similarity_score=0.9,
            source="verified_database",
            metadata={"verified": True}
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(factors, ConfidenceFactors)
    
    def test_source_reliability(self):
        """Test source reliability scoring."""
        # High reliability source
        conf1, _ = self.scorer.calculate_confidence(
            similarity_score=0.8,
            source="verified_database"
        )
        
        # Low reliability source
        conf2, _ = self.scorer.calculate_confidence(
            similarity_score=0.8,
            source="unverified"
        )
        
        self.assertGreater(conf1, conf2)
    
    def test_confidence_levels(self):
        """Test confidence level classification."""
        self.assertEqual(self.scorer.get_confidence_level(0.95), "Very High")
        self.assertEqual(self.scorer.get_confidence_level(0.8), "High")
        self.assertEqual(self.scorer.get_confidence_level(0.65), "Medium")
        self.assertEqual(self.scorer.get_confidence_level(0.45), "Low")
        self.assertEqual(self.scorer.get_confidence_level(0.3), "Very Low")
    
    def test_explain_confidence(self):
        """Test confidence explanation."""
        factors = ConfidenceFactors(
            similarity_score=0.9,
            source_reliability=0.8,
            consistency_score=0.7,
            recency_score=0.6,
            coverage_score=0.5
        )
        
        explanation = self.scorer.explain_confidence(0.75, factors)
        self.assertIsInstance(explanation, str)
        self.assertIn("confidence", explanation.lower())


class TestStrategyRecommender(unittest.TestCase):
    """Test the strategy recommendation system."""
    
    def test_keyword_recommender(self):
        """Test keyword-based recommender."""
        recommender = KeywordBasedRecommender()
        
        rec = recommender.recommend_strategy(
            "I need to understand the semantic relationships between entities",
            domain="general"
        )
        
        self.assertIsInstance(rec, StrategyRecommendation)
        self.assertIn(rec.strategy_name, KeywordBasedRecommender.STRATEGY_KEYWORDS)
        self.assertGreaterEqual(rec.confidence, 0.0)
        self.assertLessEqual(rec.confidence, 1.0)
    
    def test_domain_recommender(self):
        """Test domain-based recommender."""
        recommender = DomainBasedRecommender()
        
        rec = recommender.recommend_strategy(
            "Analyze financial data",
            domain="finance"
        )
        
        self.assertIsInstance(rec, StrategyRecommendation)
        self.assertGreater(len(rec.reasoning), 0)
    
    def test_complexity_recommender(self):
        """Test complexity-based recommender."""
        recommender = ComplexityBasedRecommender()
        
        # Simple problem
        rec1 = recommender.recommend_strategy("Simple query")
        
        # Complex problem
        rec2 = recommender.recommend_strategy(
            "This is a very complex problem with multiple interconnected components "
            "that require careful analysis and sophisticated decomposition strategies"
        )
        
        self.assertIsInstance(rec1, StrategyRecommendation)
        self.assertIsInstance(rec2, StrategyRecommendation)
    
    def test_historical_recommender(self):
        """Test historical performance recommender."""
        recommender = HistoricalPerformanceRecommender()
        
        rec = recommender.recommend_strategy("Test problem")
        
        self.assertIsInstance(rec, StrategyRecommendation)
        self.assertIn("success rate", rec.reasoning.lower())
    
    def test_ensemble_selector(self):
        """Test ensemble strategy selector."""
        selector = EnsembleStrategySelector()
        
        rec = selector.recommend_strategy(
            "Optimize a complex system with multiple constraints",
            domain="engineering"
        )
        
        self.assertIsInstance(rec, StrategyRecommendation)
        self.assertGreaterEqual(rec.confidence, 0.0)
        self.assertLessEqual(rec.confidence, 1.0)
        self.assertGreater(len(rec.alternatives), 0)
    
    def test_recommend_strategy_convenience(self):
        """Test convenience function."""
        rec = recommend_strategy(
            "Analyze semantic dependencies in code",
            domain="engineering"
        )
        
        self.assertIsInstance(rec, StrategyRecommendation)


class TestFullFeaturedBackends(unittest.IsolatedAsyncioTestCase):
    """Test full-featured backends with CRUD operations."""
    
    async def test_inmemory_crud(self):
        """Test in-memory backend CRUD operations."""
        backend = FullFeaturedInMemoryBackend({})
        await backend.connect()
        
        try:
            # Create
            entry = KnowledgeEntry(
                source="test",
                content="Test content",
                metadata={"test": True}
            )
            entry_id = await backend.add_knowledge(entry)
            self.assertIsNotNone(entry_id)
            
            # Read
            results = await backend.search("test")
            self.assertGreater(results.total_count, 0)
            
            # Update
            updated = await backend.update_knowledge(
                entry_id,
                {"content": "Updated content"}
            )
            self.assertTrue(updated)
            
            # Delete
            deleted = await backend.delete_knowledge(entry_id)
            self.assertTrue(deleted)
            
            # Verify deletion
            results_after = await backend.search("updated")
            self.assertEqual(results_after.total_count, 0)
            
        finally:
            await backend.disconnect()
    
    async def test_inmemory_clear_all(self):
        """Test clearing all knowledge."""
        backend = FullFeaturedInMemoryBackend({})
        await backend.connect()
        
        try:
            # Add some entries
            for i in range(5):
                entry = KnowledgeEntry(
                    source="test",
                    content=f"Content {i}"
                )
                await backend.add_knowledge(entry)
            
            # Clear all
            cleared = await backend.clear_all()
            self.assertEqual(cleared, 5)
            
            # Verify
            stats = await backend.get_statistics()
            self.assertEqual(stats.node_count, 0)
            
        finally:
            await backend.disconnect()


class TestCompleteIntegration(unittest.IsolatedAsyncioTestCase):
    """Test the complete integration."""
    
    async def test_create_complete_engine(self):
        """Test creating complete knowledge engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_complete_knowledge_engine(
                storage_path=tmpdir,
                enable_learning=False
            )
            
            self.assertIsInstance(engine, CompletedKnowledgeEngine)
            self.assertIsNotNone(engine.embedding_service)
            self.assertIsNotNone(engine.confidence_scorer)
            self.assertIsNotNone(engine.strategy_selector)
    
    def test_generate_embedding(self):
        """Test embedding generation through complete engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_complete_knowledge_engine(
                storage_path=tmpdir,
                enable_learning=False
            )
            
            embedding = engine.generate_embedding("Test text")
            self.assertIsInstance(embedding, list)
            self.assertEqual(len(embedding), engine.embedding_service.config.dimensions)
    
    def test_recommend_strategy(self):
        """Test strategy recommendation through complete engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_complete_knowledge_engine(
                storage_path=tmpdir,
                enable_learning=False
            )
            
            rec = engine.recommend_strategy(
                "Analyze dependencies in a complex system",
                domain="engineering"
            )
            
            self.assertIsInstance(rec, StrategyRecommendation)
            self.assertGreater(len(rec.strategy_name), 0)
    
    def test_get_stats(self):
        """Test getting engine statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = create_complete_knowledge_engine(
                storage_path=tmpdir,
                enable_learning=False
            )
            
            stats = engine.get_stats()
            
            self.assertIn('embedding_service', stats)
            self.assertIn('confidence_scorer_enabled', stats)
            self.assertIn('strategy_selector', stats)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_embedding_similarity_workflow(self):
        """Test complete embedding and similarity workflow."""
        service = create_embedding_service()
        
        # Generate embeddings for related texts
        texts = [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Python is a programming language",
            "Neural networks can learn patterns"
        ]
        
        embeddings = service.embed_batch(texts)
        
        # ML/DL texts should be similar
        ml_dl_sim = service.compute_similarity(embeddings[0], embeddings[1])
        
        # ML/Python texts should be less similar
        ml_py_sim = service.compute_similarity(embeddings[0], embeddings[2])
        
        # Verify ML/DL is more similar than ML/Python
        self.assertGreater(ml_dl_sim, ml_py_sim)
    
    def test_confidence_strategy_integration(self):
        """Test integration of confidence and strategy systems."""
        # Get strategy recommendation
        strategy_rec = recommend_strategy(
            "Optimize machine learning model performance",
            domain="optimization"
        )
        
        # Get confidence for the strategy
        scorer = get_confidence_scorer()
        confidence = scorer.calculate_confidence(
            similarity_score=0.8,
            source="verified_database",
            metadata={"strategy": strategy_rec.strategy_name}
        )
        
        self.assertIsInstance(strategy_rec, StrategyRecommendation)
        self.assertGreaterEqual(confidence[0], 0.0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddingService))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategyRecommender))
    suite.addTests(loader.loadTestsFromTestCase(TestFullFeaturedBackends))
    suite.addTests(loader.loadTestsFromTestCase(TestCompleteIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
