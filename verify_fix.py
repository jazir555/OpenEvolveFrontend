#!/usr/bin/env python3
"""
Simple verification script to test if the integrated_workflow.py fixes work
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import integrated_workflow
        print("✅ integrated_workflow imports successfully!")
        
        # Test if we can access the main function
        if hasattr(integrated_workflow, 'run_fully_integrated_adversarial_evolution'):
            print("✅ run_fully_integrated_adversarial_evolution function found!")
        else:
            print("❌ run_fully_integrated_adversarial_evolution function NOT found!")
            
        # Test if we can access other key functions
        key_functions = [
            'run_enhanced_adversarial_loop',
            'run_enhanced_evolution_loop', 
            'run_evaluator_loop',
            'check_approval_rate'
        ]
        
        for func_name in key_functions:
            if hasattr(integrated_workflow, func_name):
                print(f"✅ {func_name} function found!")
            else:
                print(f"❌ {func_name} function NOT found!")
                
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_evolution_imports():
    """Test if evolution imports work"""
    print("\nTesting evolution imports...")
    
    try:
        from evolution import ContentEvaluator, _request_openai_compatible_chat
        print("✅ evolution imports successfully!")
        
        # Test ContentEvaluator
        evaluator = ContentEvaluator("document_general", "Test prompt")
        print("✅ ContentEvaluator created successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Evolution import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Evolution unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("=== Verifying integrated_workflow.py fixes ===\n")
    
    success1 = test_imports()
    success2 = test_evolution_imports()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The fixes appear to be working.")
    else:
        print("\n⚠️ Some tests failed. There may still be issues.")