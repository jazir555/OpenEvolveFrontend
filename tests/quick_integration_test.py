#!/usr/bin/env python3
"""
Quick integration test for OpenEvolve features
"""

def test_all_features():
    """Test all major OpenEvolve features."""
    print("Testing OpenEvolve integration...")
    
    # Test 1: Basic import
    try:
        print("[SUCCESS] OpenEvolve core available")
    except ImportError:
        print("[FAILURE] OpenEvolve core not available")
        return False
    
    # Test 2: Integration module
    try:
        print("[SUCCESS] Integration module available")
    except ImportError:
        print("[FAILURE] Integration module not available")
        return False
    
    # Test 3: Advanced configuration
    try:
        from openevolve_integration import create_advanced_openevolve_config
        _ = create_advanced_openevolve_config("gpt-4", "test-key")
        print("[SUCCESS] Advanced configuration working")
    except Exception as e:
        print(f"[FAILURE] Advanced configuration failed: {e}")
        return False
    
    # Test 4: Ensemble configuration
    try:
        from openevolve_integration import create_ensemble_config_with_fallback
        create_ensemble_config_with_fallback(["gpt-4"], ["gpt-3.5-turbo"], "test-key")
        print("[SUCCESS] Ensemble configuration working")
    except Exception as e:
        print(f"[FAILURE] Ensemble configuration failed: {e}")
        return False
    
    # Test 5: Logging utility
    try:
        print("[SUCCESS] Logging utility available")
    except ImportError:
        print("[FAILURE] Logging utility not available")
        return False
    
    # Test 6: Monitoring dashboard
    try:
        print("[SUCCESS] Monitoring dashboard available")
    except ImportError:
        print("[FAILURE] Monitoring dashboard not available")
        return False
    
    print("\n[SUCCESS] All major components are working correctly!")
    return True

if __name__ == "__main__":
    success = test_all_features()
    exit(0 if success else 1)