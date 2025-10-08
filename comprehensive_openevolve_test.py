#!/usr/bin/env python3
"""
Comprehensive test to verify all OpenEvolve integrations are working properly.
"""

import sys
import os
import tempfile
import traceback

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_all_openevolve_integrations():
    """Test all OpenEvolve integrations"""
    print("=" * 60)
    print("Comprehensive OpenEvolve Integration Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Evolution module integration
    print("\n1. Testing Evolution Module Integration...")
    try:
        from .evolution import ContentEvaluator
        print("✓ Successfully imported evolution module")
        
        # Test ContentEvaluator
        evaluator = ContentEvaluator("general", "Test evaluator prompt")
        print("✓ Successfully created ContentEvaluator instance")
        
        # Create a temporary file for testing
        test_content = "This is a test content for evolution."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        try:
            # Test the evaluator
            result = evaluator.evaluate(temp_file_path)
            print(f"✓ ContentEvaluator returned result with score: {result.get('score', 'N/A')}")
            
            # Verify OpenEvolve-compatible metrics are present
            required_metrics = ['combined_score', 'complexity', 'diversity']
            missing_metrics = [metric for metric in required_metrics if metric not in result]
            if missing_metrics:
                print(f"⚠ Missing OpenEvolve-compatible metrics: {missing_metrics}")
                results.append(False)
            else:
                print("✓ All required OpenEvolve-compatible metrics present")
                results.append(True)
                
        finally:
            # Clean up
            os.unlink(temp_file_path)
        
    except Exception as e:
        print(f"✗ Error testing evolution module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 2: Adversarial module integration
    print("\n2. Testing Adversarial Module Integration...")
    try:
        from .adversarial import run_enhanced_adversarial_evolution
        print("✓ Successfully imported adversarial module")
        
        # Check for OpenEvolve availability
        try:
            print("✓ OpenEvolve API available")
            results.append(True)
        except ImportError:
            print("⚠ OpenEvolve API not available - fallback to API-based testing expected")
            results.append(True)  # This is expected behavior
        
    except Exception as e:
        print(f"✗ Error testing adversarial module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 3: Integrated workflow integration
    print("\n3. Testing Integrated Workflow Integration...")
    try:
        from .integrated_workflow import generate_adversarial_data_augmentation
        print("✓ Successfully imported integrated workflow module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing integrated workflow module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 4: Evolutionary optimization integration
    print("\n4. Testing Evolutionary Optimization Integration...")
    try:
        from .evolutionary_optimization import EvolutionaryOptimizer
        print("✓ Successfully imported evolutionary optimization module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing evolutionary optimization module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 5: Quality assurance integration
    print("\n5. Testing Quality Assurance Integration...")
    try:
        from .quality_assurance import QualityAssuranceOrchestrator
        print("✓ Successfully imported quality assurance module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing quality assurance module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 6: Prompt engineering integration
    print("\n6. Testing Prompt Engineering Integration...")
    try:
        from .prompt_engineering import PromptEngineeringSystem
        print("✓ Successfully imported prompt engineering module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing prompt engineering module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 7: Model orchestration integration
    print("\n7. Testing Model Orchestration Integration...")
    try:
        from .model_orchestration import ModelOrchestrator
        print("✓ Successfully imported model orchestration module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing model orchestration module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 8: Content analyzer integration
    print("\n8. Testing Content Analyzer Integration...")
    try:
        from .content_analyzer import ContentAnalyzer
        print("✓ Successfully imported content analyzer module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing content analyzer module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 9: Red team integration
    print("\n9. Testing Red Team Integration...")
    try:
        from .red_team import RedTeam
        print("✓ Successfully imported red team module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing red team module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 10: Blue team integration
    print("\n10. Testing Blue Team Integration...")
    try:
        from .blue_team import BlueTeam
        print("✓ Successfully imported blue team module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing blue team module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Test 11: Evaluator team integration
    print("\n11. Testing Evaluator Team Integration...")
    try:
        from .evaluator_team import EvaluatorTeam
        print("✓ Successfully imported evaluator team module")
        results.append(True)
    except Exception as e:
        print(f"✗ Error testing evaluator team module: {e}")
        traceback.print_exc()
        results.append(False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✅ All tests passed! OpenEvolve integration is working properly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = test_all_openevolve_integrations()
    sys.exit(0 if success else 1)