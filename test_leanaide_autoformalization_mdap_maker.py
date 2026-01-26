"""
Test Suite for LeanAide Autoformalization MDAP/MAKER Integration

This module provides comprehensive tests for the autoformalization system
that integrates MDAP and MAKER capabilities.
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import json
import time

from leanaide_autoformalization_mdap_maker import (
    LeanAideAutoformalizationEngine,
    AutoformalizationResult,
    AutoformalizationStrategy,
    create_leanaide_autoformalization_engine,
    autoformalize_with_mdap_maker
)


class TestLeanAideAutoformalizationEngine(unittest.TestCase):
    """Test the LeanAide autoformalization engine with MDAP/MAKER integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock clients
        self.mock_leanaide_client = Mock()
        self.mock_leanaide_client.cache = Mock()

        # Create the engine directly without mocking the AutoformalizationEngine
        # since it's imported from lean4_integration
        self.engine = LeanAideAutoformalizationEngine(
            leanaide_client=self.mock_leanaide_client,
            enable_caching=False
        )

    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.enable_caching, False)
        self.assertEqual(self.engine.cache_ttl_seconds, 3600)

    @patch('lean4_integration.AutoformalizationEngine')
    def test_create_engine_function(self, mock_auto_engine_class):
        """Test the create_engine convenience function."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by trivial"
        mock_result.theorem_name = "test"
        mock_result.errors = []
        mock_result.warnings = []

        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_auto_engine_class.return_value = mock_engine_instance

        engine = create_leanaide_autoformalization_engine(
            leanaide_client=self.mock_leanaide_client,
            enable_caching=True
        )

        self.assertIsInstance(engine, LeanAideAutoformalizationEngine)
        self.assertTrue(engine.enable_caching)

    @patch('lean4_integration.AutoformalizationEngine')
    def test_direct_autoformalize(self, mock_auto_engine_class):
        """Test direct autoformalization strategy."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by trivial"
        mock_result.theorem_name = "test"
        mock_result.errors = []
        mock_result.warnings = []
        
        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_auto_engine_class.return_value = mock_engine_instance

        async def run_test():
            result = await self.engine.autoformalize(
                natural_language="Prove that true is true",
                statement_type="theorem",
                name="test",
                strategy=AutoformalizationStrategy.DIRECT
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.strategy_used, "direct")
            self.assertEqual(result.lean_code, "theorem test : True := by trivial")

        asyncio.run(run_test())

    @patch('lean4_integration.AutoformalizationEngine')
    def test_cache_functionality(self, mock_auto_engine_class):
        """Test caching functionality."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by sorry"
        mock_result.theorem_name = "test"
        mock_result.errors = []
        mock_result.warnings = []

        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_auto_engine_class.return_value = mock_engine_instance

        # Create engine with caching enabled
        engine = LeanAideAutoformalizationEngine(
            leanaide_client=self.mock_leanaide_client,
            enable_caching=True,
            cache_ttl_seconds=1000  # Long TTL for testing
        )

        async def run_test():
            # First call
            result1 = await engine.autoformalize(
                natural_language="test theorem",
                statement_type="theorem",
                strategy=AutoformalizationStrategy.DIRECT
            )

            # Second call with same parameters should use cache
            result2 = await engine.autoformalize(
                natural_language="test theorem",
                statement_type="theorem",
                strategy=AutoformalizationStrategy.DIRECT
            )

            self.assertEqual(result1.lean_code, result2.lean_code)
            self.assertEqual(len(engine.cache), 1)  # One cached entry

        asyncio.run(run_test())

    @patch('lean4_integration.AutoformalizationEngine')
    def test_adaptive_strategy_selection(self, mock_auto_engine_class):
        """Test adaptive strategy selection."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by trivial"
        mock_result.theorem_name = "test"
        mock_result.errors = []
        mock_result.warnings = []

        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_auto_engine_class.return_value = mock_engine_instance

        engine = LeanAideAutoformalizationEngine(
            leanaide_client=self.mock_leanaide_client
        )

        # Test complex theorem selection
        complex_theorem = "Prove by induction that for all natural numbers n, sum of first n numbers is n*(n+1)/2"
        strategy = engine._select_adaptive_strategy(complex_theorem, {})
        # Should select direct since no MDAP orchestrator
        self.assertEqual(strategy, AutoformalizationStrategy.DIRECT)

        # Test simple theorem selection
        simple_theorem = "Prove that basic arithmetic holds"
        strategy = engine._select_adaptive_strategy(simple_theorem, {})
        self.assertEqual(strategy, AutoformalizationStrategy.DIRECT)

    @patch('lean4_integration.AutoformalizationEngine')
    def test_domain_inference(self, mock_auto_engine_class):
        """Test domain inference from natural language."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by trivial"
        mock_result.theorem_name = "test"
        mock_result.errors = []
        mock_result.warnings = []

        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_auto_engine_class.return_value = mock_engine_instance

        engine = LeanAideAutoformalizationEngine(
            leanaide_client=self.mock_leanaide_client
        )

        # Test algebra domain
        algebra_stmt = "Prove that for any group G, the identity element is unique"
        domain = engine._infer_domain(algebra_stmt)
        # Since we're importing from leanaide_mdap, the domain should be from that module
        from leanaide_mdap import LeanDomain
        self.assertEqual(domain, LeanDomain.ALGEBRA)

        # Test analysis domain
        analysis_stmt = "Prove the limit definition of derivatives"
        domain = engine._infer_domain(analysis_stmt)
        self.assertEqual(domain, LeanDomain.ANALYSIS)

        # Test general domain
        general_stmt = "Prove something mathematical"
        domain = engine._infer_domain(general_stmt)
        self.assertEqual(domain, LeanDomain.GENERAL)

    def test_system_status(self):
        """Test system status reporting."""
        status = self.engine.get_system_status()
        
        self.assertTrue(status["autoformalization_engine"])
        self.assertFalse(status["mdap_available"])  # No orchestrator provided
        self.assertFalse(status["maker_available"])  # No maker engine provided
        self.assertFalse(status["caching_enabled"])  # Set to False in setup
        self.assertIn("adaptive", status["available_strategies"])


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_leanaide_client = Mock()
        self.mock_leanaide_client.cache = Mock()

    @patch('leanaide_autoformalization_mdap_maker.LeanAideAutoformalizationEngine')
    def test_autoformalize_with_mdap_maker(self, mock_engine_class):
        """Test the convenience autoformalize function."""
        # Set up mock
        mock_result = AutoformalizationResult(
            success=True,
            lean_code="theorem test : True := by trivial",
            confidence=0.9,
            verification_status="not_verified"
        )
        
        mock_instance = Mock()
        mock_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_instance

        async def run_test():
            result = await autoformalize_with_mdap_maker(
                natural_language="Prove that true is true",
                leanaide_client=self.mock_leanaide_client
            )
            
            self.assertTrue(result.success)
            self.assertEqual(result.confidence, 0.9)
            self.assertEqual(result.lean_code, "theorem test : True := by trivial")

        asyncio.run(run_test())


def run_comprehensive_tests():
    """Run all tests."""
    print("Running comprehensive tests for LeanAide Autoformalization MDAP/MAKER...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests
    test_suite.addTest(unittest.makeSuite(TestLeanAideAutoformalizationEngine))
    test_suite.addTest(unittest.makeSuite(TestConvenienceFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)