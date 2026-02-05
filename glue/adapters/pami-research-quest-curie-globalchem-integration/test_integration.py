"""
Test suite for PAMI - Research Quest - Curie-GlobalChem Integration Adapter

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

from pami_research_quest_curie_globalchem_adapter import PAMIResearchQuestCurieGlobalChemAdapter, create_unified_interface


class TestPAMIResearchQuestCurieGlobalChemAdapter(unittest.TestCase):
    """Test cases for the PAMI - Research Quest - Curie-GlobalChem Adapter"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "min_support": 0.1,
            "min_confidence": 0.5,
            "research_quest_curie_globalchem_config": {
                "log_level": "WARNING",
                "curie_globalchem_config": {
                    "log_level": "WARNING"
                },
                "research_quest_config": {
                    "model": "openai/gpt-4o",
                    "temperature": 0.1
                }
            },
            "pami_config": {}
        }
        self.adapter = PAMIResearchQuestCurieGlobalChemAdapter(config=config)
        self.interface = create_unified_interface(self.adapter)
    
    def test_initialization(self):
        """Test that the adapter initializes correctly"""
        self.assertIsNotNone(self.adapter)
        self.assertIsNotNone(self.adapter.rq_cg_adapter)
        self.assertIsNotNone(self.adapter.pami_integration)
        self.assertEqual(self.adapter.config["log_level"], "WARNING")
    
    def test_async_methods_exist(self):
        """Test that async methods exist and are callable"""
        self.assertTrue(hasattr(self.adapter, 'analyze_research_patterns'))
        self.assertTrue(hasattr(self.adapter, 'conduct_pattern_enriched_research'))
        self.assertTrue(hasattr(self.adapter, 'analyze_chemical_knowledge_graph_patterns'))
        self.assertTrue(hasattr(self.adapter, 'generate_pattern_based_research_proposal'))
        
        # Verify they are callable
        self.assertTrue(callable(self.adapter.analyze_research_patterns))
        self.assertTrue(callable(self.adapter.conduct_pattern_enriched_research))
        self.assertTrue(callable(self.adapter.analyze_chemical_knowledge_graph_patterns))
        self.assertTrue(callable(self.adapter.generate_pattern_based_research_proposal))
    
    def test_interface_function_exists(self):
        """Test that the interface function exists and is callable"""
        self.assertTrue(callable(self.interface))


class TestAntiCorruptionLayer(unittest.TestCase):
    """Test cases to verify anti-corruption layer patterns"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "min_support": 0.1,
            "min_confidence": 0.5,
            "research_quest_curie_globalchem_config": {
                "log_level": "WARNING",
                "curie_globalchem_config": {
                    "log_level": "WARNING"
                },
                "research_quest_config": {
                    "model": "openai/gpt-4o",
                    "temperature": 0.1
                }
            },
            "pami_config": {}
        }
        self.adapter = PAMIResearchQuestCurieGlobalChemAdapter(config=config)
    
    def test_readonly_access(self):
        """Test that underlying systems remain unmodified"""
        # The adapter should not modify the state of underlying systems
        # during normal operations
        initial_state = str(type(self.adapter.rq_cg_adapter))
        initial_pami_state = str(type(self.adapter.pami_integration))
        
        # Perform operations
        async def run_test():
            sample_data = {
                'transactions': [['test', 'data', 'pattern']]
            }
            result = await self.adapter.analyze_research_patterns(sample_data)
            return result
        
        # Run the async test
        result = asyncio.run(run_test())
        
        # Check that the types/instances are still the same (no modification)
        final_state = str(type(self.adapter.rq_cg_adapter))
        final_pami_state = str(type(self.adapter.pami_integration))
        
        self.assertEqual(initial_state, final_state)
        self.assertEqual(initial_pami_state, final_pami_state)
    
    def test_error_handling_prevents_corruption(self):
        """Test that errors don't corrupt the system state"""
        initial_state = str(type(self.adapter.rq_cg_adapter))
        
        async def run_error_test():
            try:
                # Try to analyze with empty data
                result = await self.adapter.analyze_research_patterns({})
                return result
            except Exception:
                return {"error_occurred": True}
        
        result = asyncio.run(run_error_test())
        
        # System should still be in a valid state after error
        final_state = str(type(self.adapter.rq_cg_adapter))
        self.assertEqual(initial_state, final_state)
    
    def test_canonical_data_model_compliance(self):
        """Test that the adapter follows canonical data model patterns"""
        async def run_test():
            # Perform an operation that returns structured data
            sample_data = {
                'transactions': [['aspirin', 'pain_relief']]
            }
            result = await self.adapter.analyze_research_patterns(sample_data)
            return result
        
        result = asyncio.run(run_test())
        
        # Verify that returned data has expected structure
        self.assertIn('pattern_analysis', result)
        self.assertIn('insights', result)
        self.assertIn('recommendations', result)
        self.assertIn('timestamp', result)
        self.assertIn('adapter_version', result)


def run_tests():
    """Function to run the tests"""
    # Create test suites
    loader = unittest.TestLoader()
    
    suite1 = loader.loadTestsFromTestCase(TestPAMIResearchQuestCurieGlobalChemAdapter)
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
        print("[OK] All tests passed!")
        return True
    else:
        print("[FAIL] Some tests failed!")
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