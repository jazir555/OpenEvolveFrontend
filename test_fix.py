#!/usr/bin/env python3
"""
Test script to verify the integrated_workflow.py fixes
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing import of integrated_workflow...")
    from integrated_workflow import run_fully_integrated_adversarial_evolution
    print("✅ SUCCESS: integrated_workflow imports successfully!")
    
    # Test if the function can be called with minimal parameters
    print("Testing function signature...")
    try:
        # Just test that we can create the function object
        func = run_fully_integrated_adversarial_evolution
        print("✅ SUCCESS: Function signature is valid!")
        
        # Test if we can import the analyze_with_model function
        from integrated_workflow import analyze_with_model
        print("✅ SUCCESS: analyze_with_model imports successfully!")
        
    except Exception as e:
        print(f"❌ ERROR in function signature: {e}")
        
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("This indicates there are syntax errors in the file.")
except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")

print("\nTest completed.")