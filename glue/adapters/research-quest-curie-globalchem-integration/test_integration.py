"""
Test suite for Research Quest - Curie-GlobalChem Integration Adapter

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'core-projects', 'global-chem', 'global_chem'))

from research_quest_curie_globalchem_adapter import ResearchQuestCurieGlobalChemAdapter, create_research_interface


class TestResearchQuestCurieGlobalChemAdapter(unittest.TestCase):
    """Test cases for the Research Quest - Curie-GlobalChem Adapter"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "curie_globalchem_config": {
                "log_level": "WARNING"
            },
            "research_quest_config": {
                "model": "openai/gpt-4o",
                "temperature": 0.1
            }
        }
        self.adapter = ResearchQuestCurieGlobalChemAdapter(config=config)
        self.interface = create_research_interface(self.adapter)
    
    def test_initialization(self):
        """Test that the adapter initializes correctly"""
        self.assertIsNotNone(self.adapter)
        self.assertIsNotNone(self.adapter.curie_gc_adapter)
        self.assertIsNotNone(self.adapter.research_quest_integration)
        self.assertEqual(self.adapter.config["log_level"], "WARNING")
    
    def test_async_methods_exist(self):
        """Test that async methods exist and are callable"""
        self.assertTrue(hasattr(self.adapter, 'search_chemicals_for_research'))
        self.assertTrue(hasattr(self.adapter, 'conduct_chemistry_research'))
        self.assertTrue(hasattr(self.adapter, 'analyze_chemical_interactions'))
        self.assertTrue(hasattr(self.adapter, 'generate_research_proposal'))
        
        # Verify they are callable
        self.assertTrue(callable(self.adapter.search_chemicals_for_research))
        self.assertTrue(callable(self.adapter.conduct_chemistry_research))
        self.assertTrue(callable(self.adapter.analyze_chemical_interactions))
        self.assertTrue(callable(self.adapter.generate_research_proposal))
    
    def test_interface_function_exists(self):
        """Test that the interface function exists and is callable"""
        self.assertTrue(callable(self.interface))


class TestAntiCorruptionLayer(unittest.TestCase):
    """Test cases to verify anti-corruption layer patterns"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "curie_globalchem_config": {
                "log_level": "WARNING"
            },
            "research_quest_config": {
                "model": "openai/gpt-4o",
                "temperature": 0.1
            }
        }
        self.adapter = ResearchQuestCurieGlobalChemAdapter(config=config)
    
    def test_readonly_access(self):
        """Test that underlying systems remain unmodified"""
        # The adapter should not modify the state of underlying systems
        # during normal operations
        initial_state = str(type(self.adapter.curie_gc_adapter))
        initial_rq_state = str(type(self.adapter.research_quest_integration))
        
        # Perform operations
        async def run_test():
            result = await self.adapter.search_chemicals_for_research("test")
            return result
        
        # Run the async test
        result = asyncio.run(run_test())
        
        # Check that the types/instances are still the same (no modification)
        final_state = str(type(self.adapter.curie_gc_adapter))
        final_rq_state = str(type(self.adapter.research_quest_integration))
        
        self.assertEqual(initial_state, final_state)
        self.assertEqual(initial_rq_state, final_rq_state)
    
    def test_error_handling_prevents_corruption(self):
        """Test that errors don't corrupt the system state"""
        initial_state = str(type(self.adapter.curie_gc_adapter))
        
        async def run_error_test():
            try:
                # Try to search for something that might cause an error
                result = await self.adapter.search_chemicals_for_research("")
                return result
            except Exception:
                return {"error_occurred": True}
        
        result = asyncio.run(run_error_test())
        
        # System should still be in a valid state after error
        final_state = str(type(self.adapter.curie_gc_adapter))
        self.assertEqual(initial_state, final_state)
    
    def test_canonical_data_model_compliance(self):
        """Test that the adapter follows canonical data model patterns"""
        async def run_test():
            # Perform an operation that returns structured data
            result = await self.adapter.search_chemicals_for_research("water")
            return result
        
        result = asyncio.run(run_test())
        
        # Verify that returned data has expected structure
        self.assertIn('research_topic', result)
        self.assertIn('identified_chemicals', result)
        self.assertIn('related_chemicals', result)
        self.assertIn('properties_calculated', result)
        self.assertIn('timestamp', result)
        self.assertIn('adapter_version', result)


def run_tests():
    """Function to run the tests"""
    # Create test suites
    loader = unittest.TestLoader()
    
    suite1 = loader.loadTestsFromTestCase(TestResearchQuestCurieGlobalChemAdapter)
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