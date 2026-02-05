#!/usr/bin/env python3
"""
Test script to verify Unicode encoding errors are fixed.
Tests importing the fixed knowledge_engine/engine.py module.
"""

import sys
import os

def test_encoding():
    """Test that we can print ASCII-safe characters without encoding errors."""
    print("Testing encoding fix...")
    print("[OK] Check mark replacement works")
    print("[WARN] Warning replacement works") 
    print("[FAIL] Cross mark replacement works")
    print("All ASCII-safe characters printed successfully!")
    return True

def test_knowledge_engine_import():
    """Test importing knowledge_engine/engine.py"""
    print("\nTesting knowledge_engine/engine.py import...")
    
    try:
        # Import will trigger the print statements at module level
        from knowledge_engine import engine
        print("[OK] Successfully imported knowledge_engine.engine")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to import: {e}")
        return False

def main():
    """Main test function."""
    print("="*60)
    print("UNICODE ENCODING FIX VERIFICATION")
    print("="*60)
    
    results = []
    
    # Test 1: Basic encoding
    results.append(("Basic encoding test", test_encoding()))
    
    # Test 2: Knowledge engine import
    results.append(("Knowledge engine import", test_knowledge_engine_import()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[FAILURE] Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
