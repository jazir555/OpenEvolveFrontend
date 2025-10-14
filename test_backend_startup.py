#!/usr/bin/env python3
"""
Test script to verify OpenEvolve backend startup functionality
"""

import os
import sys
import logging
import requests
import time

# Add current directory to path to import main module
sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_backend_health_check():
    """Test if LLM backend health check works"""
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            print("✓ LLM backend is already running on port 8000")
            return True
        else:
            print(f"✗ LLM backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ LLM backend is not running (connection refused)")
        print("  Note: OpenEvolve requires an LLM server (like OptiLLM) on port 8000")
        return False
    except requests.exceptions.Timeout:
        print("✗ LLM backend health check timed out")
        return False
    except Exception as e:
        print(f"✗ Error during LLM backend health check: {e}")
        return False

def test_backend_script_exists():
    """Test if OpenEvolve CLI script exists"""
    backend_script = "openevolve/openevolve-run.py"
    if os.path.exists(backend_script):
        print(f"✓ OpenEvolve CLI script found: {backend_script}")
        return True
    else:
        print(f"✗ OpenEvolve CLI script not found: {backend_script}")
        return False

def test_project_root():
    """Test project root detection"""
    try:
        from main import get_project_root
        root = get_project_root()
        print(f"✓ Project root detected: {root}")
        return True
    except Exception as e:
        print(f"✗ Error detecting project root: {e}")
        return False

if __name__ == "__main__":
    print("Testing OpenEvolve Backend Startup Functionality")
    print("=" * 50)
    
    tests = [
        ("Project Root Detection", test_project_root),
        ("OpenEvolve CLI Script", test_backend_script_exists),
        ("LLM Backend Health Check", test_backend_health_check),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Backend startup functionality is working correctly.")
    else:
        print("✗ Some tests failed. Check the implementation.")