#!/usr/bin/env python3
"""
Test script to verify OpenEvolve integration in the frontend files.
This script tests that the changes we made work properly.
"""

import sys
import os
import tempfile
import traceback
from datetime import datetime

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_evolution_module():
    """Test the evolution module with OpenEvolve integration"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing evolution module...")
    
    try:
        from evolution import ContentEvaluator, run_evolution_loop
        print("[SUCCESS] Successfully imported evolution module")
        
        # Test ContentEvaluator
        evaluator = ContentEvaluator("general", "Test evaluator prompt")
        print("[SUCCESS] Successfully created ContentEvaluator instance")
        
        # Create a temporary file for testing
        test_content = "This is a test content for evolution."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        try:
            # Test the evaluator
            result = evaluator.evaluate(temp_file_path)
            print(f"[SUCCESS] ContentEvaluator returned result with score: {result.get('score', 'N/A')}")
            print(f"[SUCCESS] Additional metrics: {list(result.keys())}")
            
            # Verify OpenEvolve-compatible metrics are present
            assert 'combined_score' in result, "Missing combined_score for OpenEvolve compatibility"
            assert 'complexity' in result, "Missing complexity metric for OpenEvolve"
            assert 'diversity' in result, "Missing diversity metric for OpenEvolve"
            print("[SUCCESS] All OpenEvolve-compatible metrics present")
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
        
        print("[SUCCESS] Evolution module tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Evolution module test failed: {e}")
        traceback.print_exc()
        return False

def test_adversarial_module():
    """Test the adversarial module with OpenEvolve integration"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing adversarial module...")
    
    try:
        from adversarial import run_adversarial_testing
        print("[SUCCESS] Successfully imported adversarial module")
        
        # Check if OpenEvolve is available
        try:
            from openevolve.api import run_evolution as openevolve_run_evolution
            print("[SUCCESS] OpenEvolve API available")
        except ImportError:
            print("[WARNING] OpenEvolve API not available - fallback to API-based testing expected")
        
        # Check for the necessary components used in integrated functionality
        from adversarial import _run_adversarial_testing_with_openevolve_backend
        print("[SUCCESS] OpenEvolve backend function available")
        
        print("[SUCCESS] Adversarial module tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Adversarial module test failed: {e}")
        traceback.print_exc()
        return False

def test_integrated_workflow_module():
    """Test the integrated workflow module with OpenEvolve integration"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing integrated workflow module...")
    
    try:
        from integrated_workflow import run_fully_integrated_adversarial_evolution
        print("[SUCCESS] Successfully imported integrated workflow module")
        
        # Check for the enhanced evolution loop function
        from integrated_workflow import run_enhanced_evolution_loop
        print("[SUCCESS] Enhanced evolution loop function available")
        
        print("[SUCCESS] Integrated workflow module tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Integrated workflow module test failed: {e}")
        traceback.print_exc()
        return False

def test_openevolve_evaluator_integration():
    """Test that OpenEvolve evaluators return proper metrics"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing OpenEvolve evaluator integration...")
    
    try:
        from evolution import ContentEvaluator
        
        # Create evaluators for different content types
        content_types = ["code_python", "document_general", "document_legal", "document_medical", "document_technical"]
        test_content = {
            "code_python": "def hello():\n    print('Hello, World!')\n    return True",
            "document_general": "This is a general document for testing purposes.",
            "document_legal": "This contract is between parties A and B for legal services.",
            "document_medical": "Patient diagnosis: hypertension, treatment: medication.",
            "document_technical": "API documentation for the service including endpoints and examples."
        }
        
        for content_type in content_types:
            evaluator = ContentEvaluator(content_type, "Test evaluator prompt")
            test_content_str = test_content[content_type]
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
                temp_file.write(test_content_str)
                temp_file_path = temp_file.name
            
            try:
                result = evaluator.evaluate(temp_file_path)
                
                # Check if essential OpenEvolve metrics are present
                essential_metrics = ['score', 'combined_score', 'complexity', 'diversity', 'length']
                missing_metrics = [metric for metric in essential_metrics if metric not in result]
                
                if missing_metrics:
                    print(f"[ERROR] {content_type} evaluator missing metrics: {missing_metrics}")
                    return False
                else:
                    print(f"[SUCCESS] {content_type} evaluator has all essential metrics")
                    
            finally:
                os.unlink(temp_file_path)
        
        print("[SUCCESS] OpenEvolve evaluator integration tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenEvolve evaluator integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing OpenEvolve Integration in Frontend Files")
    print("=" * 60)
    
    tests = [
        test_evolution_module,
        test_adversarial_module,
        test_integrated_workflow_module,
        test_openevolve_evaluator_integration
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
        print()  # Add blank line between tests
    
    print("=" * 60)
    print("Test Summary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("[SUCCESS] All tests passed! OpenEvolve integration is working properly.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)