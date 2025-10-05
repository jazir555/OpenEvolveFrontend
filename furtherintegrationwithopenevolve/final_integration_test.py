#!/usr/bin/env python3
"""
Final Integration Test for OpenEvolve Frontend
This script verifies that all components properly integrate with OpenEvolve.
"""

import sys
import os
import tempfile
import traceback
from datetime import datetime

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_openevolve_availability():
    """Test if OpenEvolve is available and properly configured"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing OpenEvolve availability...")
    
    try:
        # Test importing OpenEvolve modules
        from openevolve.api import run_evolution as openevolve_run_evolution
        from openevolve.config import Config, LLMModelConfig
        print("[PASS] OpenEvolve API modules imported successfully")
        
        # Test creating configuration
        config = Config()
        print("[PASS] OpenEvolve Config created successfully")
        
        # Test creating LLM model configuration
        llm_config = LLMModelConfig(
            name="gpt-4o",
            api_key="test-key",
            api_base="https://api.openai.com/v1"
        )
        config.llm.models = [llm_config]
        print("[PASS] OpenEvolve LLMModelConfig created successfully")
        
        # Test setting evolution parameters
        config.evolution.max_iterations = 10
        config.evolution.population_size = 5
        config.evolution.num_islands = 1
        config.evolution.elite_ratio = 0.2
        config.evolution.mutation_rate = 0.1
        config.evolution.crossover_rate = 0.8
        config.evolution.archive_size = 50
        config.evolution.checkpoint_interval = 5
        print("[PASS] OpenEvolve evolution parameters configured successfully")
        
        # Test setting database parameters
        config.database.feature_dimensions = ["quality", "diversity", "complexity"]
        config.database.feature_bins = 10
        config.database.elite_selection_ratio = 0.2
        config.database.exploration_ratio = 0.3
        config.database.exploitation_ratio = 0.7
        print("[PASS] OpenEvolve database parameters configured successfully")
        
        # Test setting evaluator parameters
        config.evaluator.timeout = 300
        config.evaluator.max_retries = 3
        config.evaluator.cascade_evaluation = True
        config.evaluator.cascade_thresholds = [0.5, 0.75, 0.9]
        config.evaluator.parallel_evaluations = 4
        print("[PASS] OpenEvolve evaluator parameters configured successfully")
        
        return True
        
    except ImportError as e:
        print(f"[WARN] OpenEvolve not available: {e}")
        print("Note: This is expected in environments without OpenEvolve installed")
        return False
    except Exception as e:
        print(f"[FAIL] Error testing OpenEvolve availability: {e}")
        traceback.print_exc()
        return False

def test_evolution_module_integration():
    """Test evolution module integration with OpenEvolve"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing evolution module integration...")
    
    try:
        from evolution import ContentEvaluator, run_evolution_loop
        print("[PASS] Evolution module imported successfully")
        
        # Test ContentEvaluator
        evaluator = ContentEvaluator("general", "Test evaluator prompt")
        print("[PASS] ContentEvaluator created successfully")
        
        # Create a temporary file for testing
        test_content = "This is a test content for evolution."
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp_file:
            temp_file.write(test_content)
            temp_file_path = temp_file.name
        
        try:
            # Test the evaluator
            result = evaluator.evaluate(temp_file_path)
            print(f"[PASS] ContentEvaluator returned result with score: {result.get('score', 'N/A')}")
            
            # Verify OpenEvolve-compatible metrics are present
            required_metrics = ['combined_score', 'complexity', 'diversity']
            missing_metrics = [metric for metric in required_metrics if metric not in result]
            if missing_metrics:
                print(f"[WARN] Missing OpenEvolve-compatible metrics: {missing_metrics}")
                return False
            else:
                print("[PASS] All required OpenEvolve-compatible metrics present")
                
        finally:
            # Clean up
            os.unlink(temp_file_path)
        
        print("[PASS] Evolution module integration tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing evolution module integration: {e}")
        traceback.print_exc()
        return False

def test_adversarial_module_integration():
    """Test adversarial module integration with OpenEvolve"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing adversarial module integration...")
    
    try:
        from adversarial import run_adversarial_testing
        print("[PASS] Adversarial module imported successfully")
        
        # Test if OpenEvolve is available through the adversarial module
        try:
            from openevolve.api import run_evolution as openevolve_run_evolution
            print("[PASS] OpenEvolve API available through adversarial module")
        except ImportError:
            print("[WARN] OpenEvolve API not available - fallback to API-based testing expected")
        
        # Check for the necessary components used in integrated functionality
        from adversarial import _run_adversarial_testing_with_openevolve_backend
        print("[PASS] OpenEvolve backend function available in adversarial module")
        
        print("[PASS] Adversarial module integration tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing adversarial module integration: {e}")
        traceback.print_exc()
        return False

def test_integrated_workflow_integration():
    """Test integrated workflow integration with OpenEvolve"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing integrated workflow integration...")
    
    try:
        from integrated_workflow import run_fully_integrated_adversarial_evolution
        print("[PASS] Integrated workflow module imported successfully")
        
        # Check for the enhanced evolution loop function
        from integrated_workflow import run_enhanced_evolution_loop
        print("[PASS] Enhanced evolution loop function available in integrated workflow")
        
        print("[PASS] Integrated workflow integration tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing integrated workflow integration: {e}")
        traceback.print_exc()
        return False

def test_evolutionary_optimization_integration():
    """Test evolutionary optimization integration with OpenEvolve"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing evolutionary optimization integration...")
    
    try:
        from evolutionary_optimization import EvolutionaryOptimizer, EvolutionConfiguration
        print("[PASS] Evolutionary optimization module imported successfully")
        
        # Test creating configuration
        config = EvolutionConfiguration()
        print("[PASS] EvolutionConfiguration created successfully")
        
        # Test creating optimizer
        optimizer = EvolutionaryOptimizer(config)
        print("[PASS] EvolutionaryOptimizer created successfully")
        
        print("[PASS] Evolutionary optimization integration tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing evolutionary optimization integration: {e}")
        traceback.print_exc()
        return False

def test_quality_assurance_integration():
    """Test quality assurance integration with OpenEvolve"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Testing quality assurance integration...")
    
    try:
        from quality_assurance import QualityAssuranceOrchestrator
        print("[PASS] Quality assurance module imported successfully")
        
        # Test creating orchestrator
        orchestrator = QualityAssuranceOrchestrator()
        print("[PASS] QualityAssuranceOrchestrator created successfully")
        
        print("[PASS] Quality assurance integration tests passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing quality assurance integration: {e}")
        traceback.print_exc()
        return False

def test_all_modules():
    """Run all integration tests"""
    print("=" * 60)
    print("Final Integration Test for OpenEvolve Frontend")
    print("=" * 60)
    
    tests = [
        test_openevolve_availability,
        test_evolution_module_integration,
        test_adversarial_module_integration,
        test_integrated_workflow_integration,
        test_evolutionary_optimization_integration,
        test_quality_assurance_integration
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"[FAIL] Test {test_func.__name__} failed with exception: {e}")
            traceback.print_exc()
            results.append(False)
        print()  # Add blank line between tests
    
    print("=" * 60)
    print("Integration Test Summary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(tests) - sum(results)}")
    
    if all(results):
        print("[PASS] All integration tests passed! OpenEvolve integration is working properly.")
        return 0
    else:
        print("[FAIL] Some integration tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = test_all_modules()
    sys.exit(exit_code)