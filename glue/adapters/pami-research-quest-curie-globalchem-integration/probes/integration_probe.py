#!/usr/bin/env python3
"""
Probe script to verify PAMI - Research Quest - Curie-GlobalChem integration works correctly

This script tests that the integration can access all systems and perform basic operations.
"""

import sys
import os
from datetime import datetime, timezone
import asyncio

# Add the core projects and glue directories to the path
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, '..', 'core-projects', 'global-chem', 'global_chem'))
sys.path.insert(0, os.path.join(cur_dir, '..', 'core-projects', 'Curie'))
sys.path.insert(0, os.path.join(cur_dir, '..', 'knowledge_engine', 'integrations'))
sys.path.insert(0, os.path.join(cur_dir, 'src'))


def test_globalchem_access():
    """Test that we can access GlobalChem"""
    print("Testing GlobalChem access...")
    try:
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
        sys.path.insert(0, os.path.join(cur_dir, '..', 'core-projects', 'Curie'))
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


def test_research_quest_access():
    """Test that we can access Research Quest"""
    print("Testing Research Quest access...")
    try:
        sys.path.insert(0, os.path.join(cur_dir, '..', 'knowledge_engine', 'integrations'))
        from research_quest_integration import ResearchQuestIntegration
        rq = ResearchQuestIntegration()
        print("[SUCCESS] Successfully accessed Research Quest integration")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to access Research Quest: {e}")
        return False


def test_pami_access():
    """Test that we can access PAMI"""
    print("Testing PAMI access...")
    try:
        sys.path.insert(0, os.path.join(cur_dir, '..', 'knowledge_engine', 'integrations'))
        from pami_integration import PAMIIntegration
        pami = PAMIIntegration()
        print("[SUCCESS] Successfully accessed PAMI integration")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to access PAMI: {e}")
        return False


def test_adapter_creation():
    """Test that we can create the combined adapter"""
    print("Testing combined adapter creation...")
    try:
        sys.path.insert(0, os.path.join(cur_dir, 'src'))
        from pami_research_quest_curie_globalchem_adapter import PAMIResearchQuestCurieGlobalChemAdapter
        adapter = PAMIResearchQuestCurieGlobalChemAdapter()
        print("[SUCCESS] Successfully created PAMI - Research Quest - Curie-GlobalChem adapter")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create combined adapter: {e}")
        return False


async def test_basic_integration():
    """Test basic integration functionality"""
    print("Testing basic integration...")
    try:
        sys.path.insert(0, os.path.join(cur_dir, 'src'))
        from pami_research_quest_curie_globalchem_adapter import PAMIResearchQuestCurieGlobalChemAdapter, create_unified_interface
        
        # Create adapter
        adapter = PAMIResearchQuestCurieGlobalChemAdapter()
        interface = create_unified_interface(adapter)
        
        # Test a basic operation (this might return results or empty dicts,
        # but shouldn't throw an exception)
        sample_data = {
            'transactions': [['test', 'data', 'pattern']]
        }
        result = await interface('pattern_analysis', research_data=sample_data)
        
        print("[SUCCESS] Basic integration test completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Basic integration test failed: {e}")
        return False


async def main():
    """Main probe function"""
    print("=== PAMI - Research Quest - Curie-GlobalChem Integration Probe ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    
    # Define the tests
    sync_tests = [
        test_globalchem_access,
        test_curie_access,
        test_research_quest_access,
        test_pami_access,
        test_adapter_creation
    ]
    
    # Run synchronous tests
    results = []
    for test in sync_tests:
        result = test()
        results.append(result)
        print()
    
    # Run asynchronous tests
    async_result = await test_basic_integration()
    results.append(async_result)
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
    exit_code = asyncio.run(main())
    sys.exit(exit_code)