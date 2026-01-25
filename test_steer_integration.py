"""
Test suite for STEER (Safety, Trust, Evaluation, Error Reduction) integration.

This module tests the complete integration of the STEER reliability layer
into the OpenEvolve platform.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

from steer_context_engine import (
    SteerContextEngine,
    get_steer_engine,
    with_steer_verification,
    verify_output,
    get_reliable_prompt,
    STEER_AVAILABLE,
    CORE_MODULES_AVAILABLE,
)


class TestSteerContextEngine(unittest.TestCase):
    """Test cases for STEER Context Engine integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        os.rmdir(self.temp_dir)
    
    def test_steer_engine_initialization(self):
        """Test STEER Context Engine initialization."""
        engine = SteerContextEngine()
        
        self.assertIsNotNone(engine)
        status = engine.get_status()
        self.assertIsInstance(status, dict)
        self.assertEqual(status["available"], STEER_AVAILABLE)
        self.assertEqual(status["core_modules_available"], CORE_MODULES_AVAILABLE)
    
    def test_context_enhanced_with_rules(self):
        """Test context-enhanced prompt generation with rules."""
        engine = SteerContextEngine()
        
        base_prompt = "Solve this math problem"
        enhanced_prompt = engine.get_context_enhanced_with_rules(
            base_prompt=base_prompt,
            agent_name="test_agent",
            include_json_rules=True,
            include_slop_rules=True,
            include_pii_rules=True,
            include_citation_rules=True,
        )
        
        self.assertIn("TASK:", enhanced_prompt)
        self.assertIn("Solve this math problem", enhanced_prompt)
        # The rules might not appear if STEER isn't available, but the function should still work
    
    def test_verify_json_output(self):
        """Test JSON output verification."""
        engine = SteerContextEngine()
        
        # Test valid JSON
        result = engine.verify_json_output('{"test": "value"}')
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
    
    def test_verify_slop_filter(self):
        """Test slop filter verification."""
        engine = SteerContextEngine()
        
        # Test with some text
        result = engine.verify_slop_filter("This is a test output")
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
    
    def test_verify_pii_safety(self):
        """Test PII safety verification."""
        engine = SteerContextEngine()
        
        # Test with text that shouldn't contain PII
        result = engine.verify_pii_safety("This is a test output")
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
    
    def test_verify_citations(self):
        """Test citation verification."""
        engine = SteerContextEngine()
        
        # Test with text that has citations
        result = engine.verify_citations("This is a fact [doc 1]")
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
    
    def test_verify_sql_security(self):
        """Test SQL security verification."""
        engine = SteerContextEngine()
        
        # Test with a safe SQL query
        result = engine.verify_sql_security("SELECT * FROM users")
        self.assertIsInstance(result, dict)
        self.assertIn("passed", result)
    
    def test_run_all_verifications(self):
        """Test running all verifications."""
        engine = SteerContextEngine()
        
        # Test with a simple output and minimal verifications
        result = engine.run_all_verifications(
            output="Test output",
            verifications=["slop"] if STEER_AVAILABLE else []  # Only test if STEER is available
        )
        self.assertIsInstance(result, dict)
        self.assertIn("all_passed", result)
        self.assertIn("results", result)
    
    def test_global_engine_access(self):
        """Test access to global STEER engine."""
        engine1 = get_steer_engine()
        engine2 = get_steer_engine()
        
        # Both should return the same instance (singleton pattern)
        self.assertIs(engine1, engine2)
        self.assertIsNotNone(engine1)
    
    def test_add_custom_rule(self):
        """Test adding a custom rule."""
        engine = SteerContextEngine()
        
        def dummy_verification(output):
            return {"passed": True, "reason": "dummy", "suggested_fixes": [], "judge": "dummy"}
        
        engine.add_custom_rule("dummy_rule", "A dummy rule", dummy_verification)
        
        # Check that the rule was added
        self.assertIn("dummy_rule", engine.rules)
    
    def test_module_availability(self):
        """Test that module availability flags are set correctly."""
        self.assertIsInstance(STEER_AVAILABLE, bool)
        self.assertIsInstance(CORE_MODULES_AVAILABLE, bool)


class TestSTEERIntegration(unittest.TestCase):
    """Test STEER integration with other components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.rmdir(self.temp_dir)
    
    def test_crewai_bridge_integration(self):
        """Test integration with CrewAI bridge."""
        engine = SteerContextEngine()
        
        # Check that CrewAI bridge status is properly reported
        status = engine.get_status()
        self.assertIsInstance(status["crewai_bridge_available"], bool)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test verify_output
        result = verify_output(
            output="Test output", 
            verifications=[]  # Empty list to avoid actual verification
        )
        self.assertIsInstance(result, dict)
        
        # Test get_reliable_prompt
        prompt = get_reliable_prompt("Test prompt", "test_agent")
        self.assertIsInstance(prompt, str)
        self.assertIn("Test prompt", prompt)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests from our classes
    suite.addTests(loader.loadTestsFromTestCase(TestSteerContextEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestSTEERIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running STEER Context Engine Integration Tests...")
    print(f"STEER Available: {STEER_AVAILABLE}")
    print(f"Core Modules Available: {CORE_MODULES_AVAILABLE}")
    print("-" * 50)
    
    success = run_tests()
    
    print("-" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    if not STEER_AVAILABLE:
        print("\nNote: STEER is not available in this environment.")
        print("Install steer to enable full functionality:")
        print("  pip install steer")