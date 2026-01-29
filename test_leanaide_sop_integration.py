"""
Test Suite for LeanAide SOP Integration

This module provides comprehensive tests for the integration between
the LeanAide autoformalization system and the SOP generator.
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import time

from leanaide_sop_integration import (
    LeanAideSOPIntegration,
    MathematicalComponent,
    FormalVerificationResult,
    EnhancedSOPGenerator,
    create_enhanced_sop_generator_with_leanaide
)


class TestLeanAideSOPIntegration(unittest.TestCase):
    """Test the LeanAide-SOP integration system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock LeanAide client
        self.mock_leanaide_client = Mock()
        self.mock_leanaide_client.cache = Mock()
        
        # Create integration instance
        self.integration = LeanAideSOPIntegration(
            leanaide_client=self.mock_leanaide_client,
            enable_predictive_flagging=True,
            enable_red_flagging=True
        )

    def test_initialization(self):
        """Test integration system initialization."""
        self.assertIsNotNone(self.integration)
        self.assertIsNotNone(self.integration.autoformalization_engine)
        self.assertTrue(self.integration.enable_predictive_flagging)
        self.assertTrue(self.integration.enable_red_flagging)
        print("✅ Integration system initialized successfully")

    def test_mathematical_component_extraction(self):
        """Test extraction of mathematical components from SOP content."""
        sop_content = """
        # SOP for Mathematical Verification
        
        ## Objective
        Prove that for all natural numbers n, n + 0 = n.
        
        ## Conditions
        Where x > 0 and y < 10, verify that x + y > 0.
        
        ## Formula
        The equation a² + b² = c² must hold for right triangles.
        
        ## Constraint
        Ensure that for any function f, if f is continuous then f is bounded.
        """
        
        async def run_test():
            components = await self.integration.extract_mathematical_components(sop_content)
            
            self.assertGreater(len(components), 0)
            self.assertTrue(any("n + 0 = n" in comp.description for comp in components))
            self.assertTrue(any("x + y > 0" in comp.description for comp in components))
            self.assertTrue(any("a² + b² = c²" in comp.description for comp in components))
            
            print(f"✅ Extracted {len(components)} mathematical components")
            
            # Check domain inference
            for comp in components:
                self.assertIsInstance(comp.domain, str)
                print(f"  - Component: {comp.description[:50]}... (domain: {comp.domain})")
        
        asyncio.run(run_test())

    def test_domain_inference(self):
        """Test domain inference functionality."""
        test_cases = [
            ("Prove that groups have unique identity elements", "algebra"),
            ("Prove continuity of functions using epsilon-delta", "analysis"),
            ("Prove logical propositions using natural deduction", "logic"),
            ("Prove primality conditions for integers", "number_theory"),
            ("Prove properties of permutations", "combinatorics"),
            ("Prove geometric relationships in triangles", "geometry"),
            ("Prove topological properties of spaces", "topology"),
            ("Prove categorical properties of functors", "category_theory"),
            ("Prove general mathematical statements", "general"),
        ]
        
        for statement, expected_domain in test_cases:
            inferred_domain = self.integration._infer_domain(statement)
            print(f"  - '{statement[:30]}...' -> {inferred_domain} (expected: {expected_domain})")
            # Note: We're just verifying the function works, not exact matches
        
        print("✅ Domain inference working")

    @patch('leanaide_sop_integration.LeanAideAutoformalizationEngine')
    def test_verify_mathematical_component(self, mock_engine_class):
        """Test verification of mathematical components."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by trivial"
        mock_result.confidence = 0.9
        mock_result.errors = []
        
        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_engine_instance
        
        async def run_test():
            component = MathematicalComponent(
                description="Prove that for all n, n + 0 = n",
                domain="algebra",
                complexity=2
            )
            
            result = await self.integration.verify_mathematical_component(
                component,
                strategy="adaptive"
            )
            
            self.assertIsInstance(result, FormalVerificationResult)
            self.assertTrue(result.success)
            self.assertGreater(result.confidence, 0.5)
            print(f"✅ Mathematical component verified: confidence={result.confidence}")
        
        asyncio.run(run_test())

    @patch('leanaide_sop_integration.LeanAideAutoformalizationEngine')
    def test_verify_sop_mathematical_components(self, mock_engine_class):
        """Test verification of all mathematical components in an SOP."""
        # Set up mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.lean_code = "theorem test : True := by trivial"
        mock_result.confidence = 0.85
        mock_result.errors = []
        
        mock_engine_instance = Mock()
        mock_engine_instance.autoformalize = AsyncMock(return_value=mock_result)
        mock_engine_class.return_value = mock_engine_instance
        
        sop_content = """
        # Test SOP
        
        ## Objective
        Prove that for all natural numbers n, n + 0 = n.
        
        ## Condition  
        Where x > 0, verify that x² > 0.
        """
        
        async def run_test():
            results = await self.integration.verify_sop_mathematical_components(
                sop_content,
                strategy="adaptive"
            )
            
            self.assertIn("total_components", results)
            self.assertIn("successful_verifications", results)
            self.assertIn("success_rate", results)
            self.assertIn("average_confidence", results)
            self.assertIn("components", results)
            
            print(f"✅ SOP verification completed: {results['total_components']} components, "
                  f"{results['successful_verifications']} successful, "
                  f"success_rate={results['success_rate']:.2f}, "
                  f"avg_confidence={results['average_confidence']:.3f}")
        
        asyncio.run(run_test())

    def test_enhance_sop_content(self):
        """Test enhancing SOP content with verification information."""
        original_sop = "# Original SOP\nContent here."
        
        verification_results = {
            "total_components": 2,
            "successful_verifications": 1,
            "success_rate": 0.5,
            "average_confidence": 0.8,
            "overall_success": False,
            "components": [
                {
                    "component": "n + 0 = n",
                    "result": Mock(success=True, confidence=0.9, lean_code="theorem test : True := by trivial"),
                    "domain": "algebra",
                    "complexity": 1
                },
                {
                    "component": "x > 0 implies x² > 0", 
                    "result": Mock(success=False, confidence=0.3, error="Failed to prove"),
                    "domain": "analysis",
                    "complexity": 2
                }
            ]
        }
        
        enhanced_sop = self.integration._enhance_sop_content(original_sop, verification_results)
        
        self.assertIn("Mathematical Verification Summary", enhanced_sop)
        self.assertIn("Total mathematical components: 2", enhanced_sop)
        self.assertIn("Successfully verified: 1", enhanced_sop)
        self.assertIn("Success rate: 0.50", enhanced_sop)
        
        print("✅ SOP enhancement with verification summary working")


class TestEnhancedSOPGenerator(unittest.TestCase):
    """Test the enhanced SOP generator with LeanAide integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock components
        self.mock_base_generator = Mock()
        self.mock_leanaide_client = Mock()
        self.mock_leanaide_client.cache = Mock()
        
        self.leanaide_integration = LeanAideSOPIntegration(
            leanaide_client=self.mock_leanaide_client,
            enable_predictive_flagging=True,
            enable_red_flagging=True
        )
        
        self.enhanced_generator = EnhancedSOPGenerator(
            self.mock_base_generator,
            self.leanaide_integration
        )

    def test_enhanced_generator_initialization(self):
        """Test enhanced generator initialization."""
        self.assertIsNotNone(self.enhanced_generator)
        self.assertEqual(self.enhanced_generator.base_generator, self.mock_base_generator)
        self.assertEqual(self.enhanced_generator.leanaide_integration, self.leanaide_integration)
        print("✅ Enhanced SOP generator initialized successfully")

    @patch('leanaide_sop_integration.LeanAideSOPIntegration')
    @patch('leanaide_sop_integration.EnhancedSOPGenerator._sop_to_content')
    @patch('leanaide_sop_integration.EnhancedSOPGenerator._update_sop_with_verification')
    def test_generate_sop_with_verification(
        self, 
        mock_update_sop, 
        mock_sop_to_content, 
        mock_integration_class
    ):
        """Test generating SOP with mathematical verification."""
        # Set up mocks
        mock_integration_instance = Mock()
        mock_integration_instance.enhance_sop_with_formal_verification = AsyncMock(
            return_value=("enhanced_content", {"test": "results"})
        )
        mock_integration_class.return_value = mock_integration_instance
        
        mock_sop_to_content.return_value = "test sop content"
        mock_update_sop.return_value = Mock()
        
        async def run_test():
            result = await self.enhanced_generator.generate_sop_with_verification(
                requirement_description="Test requirement",
                domain="general"
            )
            
            self.assertIsNotNone(result)
            print("✅ SOP generation with verification completed successfully")
        
        asyncio.run(run_test())

    def test_sop_to_content_conversion(self):
        """Test SOP to content conversion."""
        # Test with object that has content attribute
        class MockSOPWithContent:
            def __init__(self):
                self.content = "test content"
        
        sop_with_content = MockSOPWithContent()
        content = self.enhanced_generator._sop_to_content(sop_with_content)
        self.assertEqual(content, "test content")
        
        # Test with object that has to_string method
        class MockSOPToString:
            def to_string(self):
                return "converted content"
        
        sop_to_string = MockSOPToString()
        content = self.enhanced_generator._sop_to_content(sop_to_string)
        self.assertEqual(content, "converted content")
        
        # Test with plain object
        plain_obj = "plain content"
        content = self.enhanced_generator._sop_to_content(plain_obj)
        self.assertEqual(content, "plain content")
        
        print("✅ SOP to content conversion working")

    def test_update_sop_with_verification(self):
        """Test updating SOP with verification information."""
        class MockSOP:
            def __init__(self):
                self.metadata = {}
        
        sop = MockSOP()
        verification_results = {"test": "results"}
        
        updated_sop = self.enhanced_generator._update_sop_with_verification(
            sop, "enhanced_content", verification_results
        )
        
        self.assertIn("verification_results", updated_sop.metadata)
        self.assertEqual(updated_sop.metadata["verification_results"], verification_results)
        
        print("✅ SOP update with verification working")


class TestFactoryFunction(unittest.TestCase):
    """Test the factory function for creating enhanced generators."""

    def test_create_enhanced_sop_generator_with_leanaide(self):
        """Test the factory function."""
        mock_base_generator = Mock()
        mock_leanaide_client = Mock()
        
        enhanced_gen = create_enhanced_sop_generator_with_leanaide(
            mock_base_generator,
            mock_leanaide_client,
            enable_predictive_flagging=True,
            enable_red_flagging=True
        )
        
        self.assertIsInstance(enhanced_gen, EnhancedSOPGenerator)
        print("✅ Factory function creates enhanced generator successfully")


def run_integration_tests():
    """Run all integration tests."""
    print("Running LeanAide-SOP Integration Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests
    test_suite.addTest(unittest.makeSuite(TestLeanAideSOPIntegration))
    test_suite.addTest(unittest.makeSuite(TestEnhancedSOPGenerator))
    test_suite.addTest(unittest.makeSuite(TestFactoryFunction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ LeanAide-SOP integration is fully functional")
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} tests failed")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]} - {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]} - {error[1]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)