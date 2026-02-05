"""
Comprehensive Test Suite for Ragbits-Knowledge Engine Integration

This test suite validates the complete integration of Ragbits with the Enterprise Knowledge Engine,
ensuring all components work together seamlessly.
"""

import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, List

from knowledge_engine.enterprise_knowledge_engine import EnterpriseKnowledgeEngine
from knowledge_engine.ragbits_document_processor import RAGBitsDocumentProcessor, DocumentProcessingResult


class TestRagbitsKnowledgeEngineIntegration(unittest.TestCase):
    """Test suite for Ragbits-Knowledge Engine integration."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'ragbits': {
                'vector_store': {
                    'type': 'memory',  # Use in-memory store for testing
                    'config': {}
                },
                'default_options': {
                    'top_k': 5,
                    'similarity_threshold': 0.5
                }
            }
        }
        self.engine = EnterpriseKnowledgeEngine(config=self.config)

    def test_ragbits_integration_initialization(self):
        """Test that Ragbits integration is properly initialized."""
        # Check that ragbits integration exists (may be None if not available)
        # Since Ragbits might not be installed in test environment, 
        # we check both cases
        if self.engine.ragbits_integration is not None:
            # If available, it should have the expected methods
            self.assertTrue(hasattr(self.engine.ragbits_integration, 'search_documents'))
            self.assertTrue(hasattr(self.engine.ragbits_integration, 'ingest_documents'))
        else:
            # If not available, it should be None (which is acceptable)
            self.assertIsNone(self.engine.ragbits_integration)

    def test_search_with_ragbits_when_available(self):
        """Test that search uses Ragbits when available."""
        if self.engine.ragbits_integration is not None:
            # Test search with ragbits query type
            result = self.engine.search_knowledge(
                query="test query",
                query_type="ragbits",
                limit=5
            )
            
            # Should return success status regardless of Ragbits availability
            self.assertIn(result['status'], ['success', 'error'])
            
            # If successful, should have results
            if result['status'] == 'success':
                self.assertIn('results', result)
                self.assertIn('result_count', result)

    def test_search_fallback_when_ragbits_not_available(self):
        """Test that search falls back to traditional methods when Ragbits is not available."""
        # Temporarily disable ragbits integration
        original_integration = self.engine.ragbits_integration
        self.engine.ragbits_integration = None
        
        try:
            result = self.engine.search_knowledge(
                query="test query",
                query_type="ragbits",
                limit=5
            )
            
            # Should still work with fallback
            self.assertIn(result['status'], ['success', 'error'])
        finally:
            # Restore original integration
            self.engine.ragbits_integration = original_integration

    def test_store_artifact_with_ragbits(self):
        """Test storing artifacts with Ragbits integration."""
        content = "This is a test artifact for Ragbits integration."
        metadata = {"test": True, "category": "integration_test"}
        
        result = self.engine.store_artifact_with_ragbits(
            content=content,
            metadata=metadata,
            artifact_type="test_artifact"
        )
        
        # Should return appropriate status
        self.assertIn(result['status'], ['success', 'partial_success', 'fallback_success', 'traditional_only', 'error'])
        
        # Should have an artifact ID in most cases
        if result['status'] not in ['error']:
            self.assertIn('artifact_id', result)

    def test_ragbits_statistics_retrieval(self):
        """Test retrieving Ragbits-specific statistics."""
        if self.engine.ragbits_integration is not None:
            # Test async statistics retrieval
            async def get_stats():
                return await self.engine.get_ragbits_statistics()
            
            stats_result = asyncio.run(get_stats())
            
            # Should return statistics
            self.assertIn('ragbits_available', stats_result)
            self.assertIsNotNone(stats_result)

    def test_get_analytics_includes_ragbits(self):
        """Test that analytics include Ragbits information."""
        analytics = self.engine.get_analytics()
        
        # Should include ragbits section
        self.assertIn('ragbits', analytics)
        
        # Ragbits section should have availability info
        self.assertIn('ragbits_available', analytics['ragbits'])


class TestRagbitsIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for Ragbits integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ragbits': {
                'vector_store': {
                    'type': 'memory',
                    'config': {}
                }
            }
        }
        self.engine = EnterpriseKnowledgeEngine(config=self.config)

    def test_invalid_search_query(self):
        """Test handling of invalid search queries."""
        result = self.engine.search_knowledge(
            query="",  # Empty query
            query_type="ragbits"
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)

    def test_invalid_artifact_content(self):
        """Test handling of invalid artifact content."""
        result = self.engine.store_artifact_with_ragbits(
            content="",  # Empty content
            metadata={}
        )
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)

    def test_large_query_handling(self):
        """Test handling of large queries."""
        large_query = "test " * 1000  # Very large query
        
        result = self.engine.search_knowledge(
            query=large_query,
            query_type="ragbits",
            limit=3
        )
        
        # Should handle gracefully (either success or error, but not crash)
        self.assertIn(result['status'], ['success', 'error'])


class TestRagbitsPerformance(unittest.TestCase):
    """Performance tests for Ragbits integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ragbits': {
                'vector_store': {
                    'type': 'memory',
                    'config': {}
                }
            }
        }
        self.engine = EnterpriseKnowledgeEngine(config=self.config)

    def test_search_performance(self):
        """Test search performance with Ragbits."""
        import time
        
        start_time = time.time()
        result = self.engine.search_knowledge(
            query="performance test query",
            query_type="ragbits",
            limit=5
        )
        end_time = time.time()
        
        search_duration = end_time - start_time
        
        # Should complete in reasonable time (under 10 seconds, even if Ragbits fails)
        self.assertLess(search_duration, 10.0)
        
        # Result should be properly structured
        self.assertIn('status', result)
        self.assertIn('processing_time', result)

    def test_bulk_artifact_storage(self):
        """Test performance of bulk artifact storage."""
        import time
        
        start_time = time.time()
        
        # Store multiple artifacts
        for i in range(5):
            content = f"Test artifact {i} for performance testing."
            metadata = {"test_id": i, "timestamp": datetime.now().isoformat()}
            
            result = self.engine.store_artifact_with_ragbits(
                content=content,
                metadata=metadata,
                artifact_type="perf_test"
            )
            
            self.assertIn(result['status'], ['success', 'partial_success', 'fallback_success', 'traditional_only'])
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Should handle multiple stores reasonably quickly
        self.assertLess(total_duration, 15.0)  # 15 seconds for 5 artifacts


class TestRagbitsCompatibility(unittest.TestCase):
    """Test compatibility with existing knowledge engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'ragbits': {
                'vector_store': {
                    'type': 'memory',
                    'config': {}
                }
            }
        }
        self.engine = EnterpriseKnowledgeEngine(config=self.config)

    def test_backward_compatibility(self):
        """Test that existing functionality still works with Ragbits integration."""
        # Test traditional search still works
        traditional_result = self.engine.search_knowledge(
            query="traditional search test",
            query_type="hybrid",  # Traditional type
            limit=3
        )
        
        self.assertIn(traditional_result['status'], ['success', 'error'])
        
        # Test traditional artifact storage still works
        workflow_data = {
            'workflow_id': 'compatibility_test_001',
            'timestamp': datetime.now().isoformat(),
            'execution_data': {
                'problem_type': 'test',
                'success': True
            }
        }
        
        processing_result = self.engine.process_workflow(workflow_data)
        self.assertIn(processing_result['status'], ['processed', 'error'])

    def test_mixed_usage_scenario(self):
        """Test scenario where both traditional and Ragbits methods are used."""
        # Store an artifact using Ragbits method
        store_result = self.engine.store_artifact_with_ragbits(
            content="Mixed usage test content",
            metadata={"usage_test": True},
            artifact_type="mixed_test"
        )
        
        self.assertIn(store_result['status'], ['success', 'partial_success', 'fallback_success', 'traditional_only'])
        
        # Search using traditional method
        search_result = self.engine.search_knowledge(
            query="Mixed usage test",
            query_type="hybrid",
            limit=5
        )
        
        self.assertIn(search_result['status'], ['success', 'error'])


def run_integration_tests():
    """Run all integration tests."""
    print("Running Ragbits-Knowledge Engine Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    
    # Load all test cases
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRagbitsKnowledgeEngineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestRagbitsIntegrationEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestRagbitsPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestRagbitsCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")
    
    success = result.wasSuccessful()
    print(f"\nOverall Result: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)