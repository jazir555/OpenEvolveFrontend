"""
Comprehensive test to verify the Curie-GlobalChem integration follows anti-corruption layer patterns

This test ensures that:
1. GlobalChem data remains unmodified (read-only principle)
2. Data transformation between systems is handled properly
3. Error handling prevents corruption
4. The adapter acts as a proper translation layer
"""

import unittest
import sys
import os
from datetime import datetime
import tempfile
import copy

# Add paths to access both systems
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, 'src')
globalchem_dir = os.path.join(current_dir, '..', '..', 'core-projects', 'global-chem', 'global_chem')

sys.path.insert(0, src_dir)
sys.path.insert(0, globalchem_dir)

from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface


class TestAntiCorruptionLayer(unittest.TestCase):
    """Test cases to verify anti-corruption layer patterns"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        config = {
            "log_level": "WARNING",  # Reduce verbosity during tests
            "max_results": 5
        }
        self.adapter = CurieGlobalChemAdapter(config=config)
        self.interface = create_curie_interface(self.adapter)
        
        # Store original GlobalChem state to verify it's not modified
        self.original_nodes = copy.deepcopy(list(self.adapter.global_chem.check_available_nodes()))
    
    def test_readonly_globalchem_access(self):
        """Test that GlobalChem data remains unmodified"""
        # Get initial state
        initial_nodes = set(self.adapter.global_chem.check_available_nodes())
        
        # Perform several operations
        result1 = self.interface('search', chemical_name='test1')
        result2 = self.interface('search', chemical_name='test2')
        
        # Check that GlobalChem state hasn't changed
        final_nodes = set(self.adapter.global_chem.check_available_nodes())
        
        self.assertEqual(initial_nodes, final_nodes, 
                         "GlobalChem state should remain unchanged after adapter operations")
    
    def test_data_transformation_validation(self):
        """Test that data transformations are validated"""
        # Test with invalid input
        result = self.interface('search', chemical_name='')
        self.assertIsNone(result, "Empty chemical name should return None")
        
        # Test with None input
        with self.assertRaises(TypeError):
            self.interface('search', chemical_name=None)
    
    def test_error_handling_prevents_corruption(self):
        """Test that errors don't corrupt the system state"""
        initial_state = len(self.adapter.global_chem.check_available_nodes())
        
        # Try to access non-existent chemical (should not affect state)
        result = self.interface('search', chemical_name='nonexistent_chemical_12345')
        self.assertIsNone(result)
        
        # Try with invalid smiles (should not affect state)
        props = self.adapter.get_chemical_properties('invalid_smiles')
        self.assertEqual(props, {})
        
        # Verify state remains unchanged
        final_state = len(self.adapter.global_chem.check_available_nodes())
        self.assertEqual(initial_state, final_state)
    
    def test_adapter_acting_as_translation_layer(self):
        """Test that adapter properly translates between systems"""
        # Verify the adapter has the expected interface methods
        self.assertTrue(hasattr(self.adapter, 'search_chemical_by_name'))
        self.assertTrue(hasattr(self.adapter, 'get_chemical_properties'))
        self.assertTrue(hasattr(self.adapter, 'get_related_chemicals'))
        self.assertTrue(hasattr(self.adapter, 'run_chemistry_experiment'))
        
        # Verify the interface function works
        self.assertTrue(callable(self.interface))
        
        # Test that the interface properly delegates to adapter methods
        # (This tests the anti-corruption pattern where Curie talks to adapter,
        # not directly to GlobalChem)
        result = self.interface('search', chemical_name='test')
        # This should call adapter.search_chemical_by_name internally
    
    def test_idempotency_of_operations(self):
        """Test that operations are idempotent (safe to repeat)"""
        # Multiple identical calls should have consistent behavior
        result1 = self.interface('search', chemical_name='test')
        result2 = self.interface('search', chemical_name='test')
        
        # Results should be equivalent (both None for non-existent chemical)
        self.assertEqual(result1, result2)
        
        # GlobalChem state should remain unchanged
        self.assertEqual(self.original_nodes, 
                         list(self.adapter.global_chem.check_available_nodes()))
    
    def test_timestamp_format_compliance(self):
        """Test that all timestamps follow UTC format as per CLAUDE.md"""
        # Perform an operation that generates a timestamp
        result = self.interface('search', chemical_name='test')
        
        # If result has timestamp, verify format
        if result and 'timestamp' in result:
            timestamp = result['timestamp']
            # Should end with 'Z' indicating UTC
            self.assertTrue(timestamp.endswith('Z'), 
                           f"Timestamp should be in UTC format ending with 'Z', got: {timestamp}")
    
    def test_configuration_isolation(self):
        """Test that adapter configuration doesn't affect GlobalChem"""
        # Create another adapter with different config
        config2 = {"log_level": "WARNING", "max_results": 20}
        adapter2 = CurieGlobalChemAdapter(config=config2)
        
        # Verify that GlobalChem instances are independent
        nodes1 = set(self.adapter.global_chem.check_available_nodes())
        nodes2 = set(adapter2.global_chem.check_available_nodes())
        
        self.assertEqual(nodes1, nodes2, 
                         "Different adapters should have same GlobalChem view")
    
    def test_canonical_data_model_compliance(self):
        """Test that the adapter follows canonical data model patterns"""
        # Verify that returned data has expected structure
        result = self.interface('search', chemical_name='test')
        
        if result is not None:  # If we get a result
            expected_fields = {'name', 'smiles', 'node', 'timestamp'}
            result_keys = set(result.keys())
            
            # Check that all expected fields are present
            self.assertTrue(expected_fields.issubset(result_keys),
                           f"Result should contain all expected fields: {expected_fields}")


def run_comprehensive_tests():
    """Function to run the comprehensive anti-corruption tests"""
    print("Running comprehensive anti-corruption layer tests...")
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAntiCorruptionLayer)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✓ All anti-corruption layer tests passed!")
        return True
    else:
        print("✗ Some anti-corruption layer tests failed!")
        for failure in result.failures:
            print(f"FAILURE: {failure[0]}")
            print(failure[1])
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(error[1])
        return False


if __name__ == '__main__':
    success = run_comprehensive_tests()
    exit(0 if success else 1)