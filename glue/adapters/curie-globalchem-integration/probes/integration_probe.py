#!/usr/bin/env python3
"""
Probe script to verify Curie-GlobalChem integration works correctly

This script tests that the integration can access both systems and perform basic operations.
"""

import sys
import os
from datetime import datetime

# Add the core projects to the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, '..', '..', 'core-projects', 'global-chem'))
sys.path.insert(0, os.path.join(cur_dir, '..', '..', 'core-projects', 'Curie'))

def test_globalchem_access():
    """Test that we can access GlobalChem"""
    print("Testing GlobalChem access...")
    try:
        # Add the GlobalChem path to sys.path
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "core-projects", "global-chem", "global_chem"))

        from global_chem import GlobalChem
        gc = GlobalChem()
        gc.build_global_chem_network()
        nodes = gc.check_available_nodes()
        print(f"[SUCCESS] Successfully accessed GlobalChem with {len(nodes)} available nodes")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to access GlobalChem: {e}")
        return False

def test_curie_access():
    """Test that we can access Curie"""
    print("Testing Curie access...")
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "core-projects", "Curie"))

        import curie
        print("[SUCCESS] Successfully accessed Curie module")
        return True
    except ImportError:
        try:
            from curie import experiment
            print("[SUCCESS] Successfully accessed Curie experiment module")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to access Curie: {e}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to access Curie: {e}")
        return False

def test_adapter_creation():
    """Test that we can create the adapter"""
    print("Testing adapter creation...")
    try:
        import sys
        import os
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(cur_dir, '..', 'src'))
        from curie_globalchem_adapter import CurieGlobalChemAdapter
        adapter = CurieGlobalChemAdapter()
        print("[SUCCESS] Successfully created Curie-GlobalChem adapter")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create adapter: {e}")
        return False

def test_basic_integration():
    """Test basic integration functionality"""
    print("Testing basic integration...")
    try:
        import sys
        import os
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(cur_dir, '..', 'src'))
        from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface

        # Create adapter
        adapter = CurieGlobalChemAdapter()
        interface = create_curie_interface(adapter)

        # Test a basic operation (this might return None if the chemical doesn't exist,
        # but shouldn't throw an exception)
        result = interface('search', chemical_name='benzene')

        print("[SUCCESS] Basic integration test completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Basic integration test failed: {e}")
        return False

def main():
    """Main probe function"""
    print("=== Curie-GlobalChem Integration Probe ===")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()

    tests = [
        test_globalchem_access,
        test_curie_access,
        test_adapter_creation,
        test_basic_integration
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    passed = sum(results)
    total = len(results)

    print(f"=== Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("[SUCCESS] All probes successful! Integration is working correctly.")
        return 0
    else:
        print("[ERROR] Some probes failed. Please check the integration setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())