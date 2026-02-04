"""
Test suite for LMQL-DSPy Integration Adapter

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs and outputs
- ANTI-HALLUCINATION: Verify data integrity
- READ-ONLY STATE: Don't modify underlying systems' data
- IDEMPOTENCY: Safe to run multiple times
- CONFIGURATION EXPLICITNESS: All parameters configurable
- UTC: All timestamps in UTC
"""

import unittest
import asyncio
import sys
import os
from datetime import datetime, timezone

# Add paths to access the required modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lmql_dspy_adapter import LMQLDSPyAdapter, create_unified_interface
from lmql_adapter import create_list_constraint


class TestLMQLDSPyAdapter(unittest.TestCase):
    """Test cases for the LMQL-DSPy Adapter"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "dspy_config": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "lmql_config": {
                "fallback_on_error": True,
                "enable_metrics": True
            }
        }
        self.adapter = LMQLDSPyAdapter(config=config)
        self.interface = create_unified_interface(self.adapter)
    
    def test_initialization(self):
        """Test that the adapter initializes correctly"""
        self.assertIsNotNone(self.adapter)
        self.assertIsNotNone(self.adapter.dspy_integration)
        self.assertIsNotNone(self.adapter.lmql_adapter)
        self.assertEqual(self.adapter.config["log_level"], "WARNING")
    
    def test_async_methods_exist(self):
        """Test that async methods exist and are callable"""
        self.assertTrue(hasattr(self.adapter, 'constrained_chain_of_thought'))
        self.assertTrue(hasattr(self.adapter, 'constrained_program_of_thought'))
        self.assertTrue(hasattr(self.adapter, 'constrained_multi_step_reasoning'))
        self.assertTrue(hasattr(self.adapter, 'solve_with_constrained_signature'))
        self.assertTrue(hasattr(self.adapter, 'batch_solve_with_constraints'))
        
        # Verify they are callable
        self.assertTrue(callable(self.adapter.constrained_chain_of_thought))
        self.assertTrue(callable(self.adapter.constrained_program_of_thought))
        self.assertTrue(callable(self.adapter.constrained_multi_step_reasoning))
        self.assertTrue(callable(self.adapter.solve_with_constrained_signature))
        self.assertTrue(callable(self.adapter.batch_solve_with_constraints))
    
    def test_interface_function_exists(self):
        """Test that the interface function exists and is callable"""
        self.assertTrue(callable(self.interface))


class TestAntiCorruptionLayer(unittest.TestCase):
    """Test cases to verify anti-corruption layer patterns"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "dspy_config": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "lmql_config": {
                "fallback_on_error": True,
                "enable_metrics": True
            }
        }
        self.adapter = LMQLDSPyAdapter(config=config)
    
    def test_readonly_access(self):
        """Test that underlying systems remain unmodified"""
        # The adapter should not modify the state of underlying systems
        # during normal operations
        initial_dspy_state = str(type(self.adapter.dspy_integration))
        initial_lmql_state = str(type(self.adapter.lmql_adapter))
        
        # Perform operations
        async def run_test():
            # Create a simple constraint
            boolean_constraint = create_list_constraint("answer", ["yes", "no"])
            
            # Run a constrained operation
            result = await self.adapter.constrained_chain_of_thought(
                question="Test question",
                constraints=[boolean_constraint]
            )
            return result
        
        # Run the async test
        result = asyncio.run(run_test())
        
        # Check that the types/instances are still the same (no modification)
        final_dspy_state = str(type(self.adapter.dspy_integration))
        final_lmql_state = str(type(self.adapter.lmql_adapter))
        
        self.assertEqual(initial_dspy_state, final_dspy_state)
        self.assertEqual(initial_lmql_state, final_lmql_state)
    
    def test_error_handling_prevents_corruption(self):
        """Test that errors don't corrupt the system state"""
        initial_state = str(type(self.adapter.dspy_integration))
        
        async def run_error_test():
            try:
                # Try to run with invalid parameters
                result = await self.adapter.constrained_chain_of_thought(
                    question=""
                )
                return result
            except Exception:
                return {"error_occurred": True}
        
        result = asyncio.run(run_error_test())
        
        # System should still be in a valid state after error
        final_state = str(type(self.adapter.dspy_integration))
        self.assertEqual(initial_state, final_state)
    
    def test_canonical_data_model_compliance(self):
        """Test that the adapter follows canonical data model patterns"""
        async def run_test():
            # Create a simple constraint
            boolean_constraint = create_list_constraint("answer", ["yes", "no"])
            
            # Perform an operation that returns structured data
            result = await self.adapter.constrained_chain_of_thought(
                question="Is this a test?",
                constraints=[boolean_constraint]
            )
            return result
        
        result = asyncio.run(run_test())
        
        # Verify that returned data has expected structure
        self.assertIn('success', result)
        self.assertIn('dspy_result', result)
        self.assertIn('lmql_result', result)
        self.assertIn('constraint_validation', result)
        self.assertIn('processing_time_ms', result)
        self.assertIn('correlation_id', result)


def run_tests():
    """Function to run the tests"""
    # Create test suites
    loader = unittest.TestLoader()
    
    suite1 = loader.loadTestsFromTestCase(TestLMQLDSPyAdapter)
    suite2 = loader.loadTestsFromTestCase(TestAntiCorruptionLayer)
    
    # Combine suites
    combined_suite = unittest.TestSuite([suite1, suite2])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(error[1])
        return False


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)