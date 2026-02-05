"""
Test script to verify circular import issues are fixed.

This script tests that all 4 files in the circular import chain can be imported
without errors:
1. z3_api_server.py
2. z3_leanaide_openevolve_integration.py  
3. bubblelabs_integration.py
4. api_server.py

Usage:
    python test_imports_fixed.py
"""

import sys
import traceback


def test_import(module_name: str, description: str) -> bool:
    """Test importing a module and report success/failure."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Module: {module_name}")
    print(f"{'='*60}")
    
    try:
        __import__(module_name)
        print(f"[PASS] SUCCESS: {module_name} imported without errors")
        return True
    except Exception as e:
        print(f"[FAIL] FAILED: {module_name}")
        print(f"   Error: {type(e).__name__}: {e}")
        print(f"\n   Traceback:")
        traceback.print_exc()
        return False


def test_import_chain():
    """Test the full import chain to ensure circular imports are resolved."""
    
    results = {
        "api_server": False,
        "bubblelabs_integration": False,
        "z3_leanaide_openevolve_integration": False,
        "z3_api_server": False,
    }
    
    print("\n" + "="*70)
    print("CIRCULAR IMPORT FIX VERIFICATION")
    print("="*70)
    print("""
Import Chain Being Tested:
    1. z3_api_server.py 
       -> imports -> z3_leanaide_openevolve_integration.py
       -> imports -> bubblelabs_integration.py  
       -> imports -> api_server.py (LAZY IMPORT FIX)
    
The circular import is fixed by using lazy imports in bubblelabs_integration.py
""")
    
    # Test 1: Import api_server first (this should work)
    results["api_server"] = test_import(
        "api_server", 
        "API Server (base module)"
    )
    
    # Test 2: Import bubblelabs_integration (should use lazy imports)
    results["bubblelabs_integration"] = test_import(
        "bubblelabs_integration",
        "BubbleLabs Integration (uses lazy imports from api_server)"
    )
    
    # Test 3: Import z3_leanaide_openevolve_integration
    results["z3_leanaide_openevolve_integration"] = test_import(
        "z3_leanaide_openevolve_integration",
        "Z3-LeanAIDE-OpenEvolve Integration"
    )
    
    # Test 4: Import z3_api_server (top of the chain)
    results["z3_api_server"] = test_import(
        "z3_api_server",
        "Z3 API Server (top of import chain)"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for module, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {module}")
    
    print(f"\n{'='*70}")
    if all_passed:
        print("ALL IMPORTS SUCCESSFUL - Circular import issue is FIXED!")
        print("="*70)
        return 0
    else:
        print("SOME IMPORTS FAILED - Issue may still exist")
        print("="*70)
        return 1


def test_cross_imports():
    """Test that cross-imports between modules work correctly."""
    print("\n" + "="*70)
    print("CROSS-IMPORT VERIFICATION")
    print("="*70)
    
    try:
        # Test that we can import all modules and access their attributes
        print("\n1. Testing api_server imports...")
        import api_server
        print(f"   - team_manager: {type(api_server.team_manager).__name__}")
        print(f"   - gauntlet_manager: {type(api_server.gauntlet_manager).__name__}")
        
        print("\n2. Testing bubblelabs_integration lazy imports...")
        import bubblelabs_integration
        # Trigger the lazy import
        team_mgr, gauntlet_mgr = bubblelabs_integration._get_api_server_managers()
        print(f"   - Lazy loaded team_manager: {type(team_mgr).__name__}")
        print(f"   - Lazy loaded gauntlet_manager: {type(gauntlet_mgr).__name__}")
        
        print("\n3. Testing BubbleLabsIntegration class...")
        integration = bubblelabs_integration.BubbleLabsIntegration()
        print(f"   - Integration created: {type(integration).__name__}")
        print(f"   - team_manager: {type(integration.team_manager).__name__}")
        print(f"   - gauntlet_manager: {type(integration.gauntlet_manager).__name__}")
        
        print("\n4. Testing z3_leanaide_openevolve_integration...")
        import z3_leanaide_openevolve_integration
        print(f"   - Module loaded: z3_leanaide_openevolve_integration")
        
        print("\n5. Testing z3_api_server...")
        import z3_api_server
        print(f"   - Module loaded: z3_api_server")
        print(f"   - INTEGRATION_AVAILABLE: {z3_api_server.INTEGRATION_AVAILABLE}")
        
        print("\n[PASS] ALL CROSS-IMPORTS WORKING CORRECTLY!")
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] CROSS-IMPORT FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


def main():
    """Main test runner."""
    exit_code = test_import_chain()
    exit_code = max(exit_code, test_cross_imports())
    
    if exit_code == 0:
        print("\n" + "="*70)
        print("ALL TESTS PASSED - CIRCULAR IMPORT ISSUE IS FIXED!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED")
        print("="*70)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
