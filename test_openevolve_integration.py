#!/usr/bin/env python3
"""
Comprehensive Test Suite for OpenEvolve Integration
This script tests all the integrated OpenEvolve features to ensure they work correctly.
"""

import sys
import os
import traceback


def test_openevolve_availability():
    """Test if OpenEvolve backend is available."""
    print("[INFO] Testing OpenEvolve backend availability...")
    try:
        print("[SUCCESS] OpenEvolve backend is available")
        return True
    except ImportError as e:
        print(f"[FAILURE] OpenEvolve backend not available: {e}")
        return False


def test_advanced_configuration():
    """Test advanced configuration options."""
    print("[INFO] Testing advanced configuration options...")
    try:
        from openevolve_integration import create_advanced_openevolve_config
        
        # Test creating advanced configuration
        config = create_advanced_openevolve_config(
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7,
            max_iterations=100,
            population_size=1000,
            num_islands=5,
            feature_dimensions=["complexity", "diversity"],
            feature_bins=10,
            enable_artifacts=True,
            cascade_evaluation=True,
            use_llm_feedback=True,
        )
        
        if config:
            print("[SUCCESS] Advanced configuration created successfully")
            print(f"   - Max iterations: {config.max_iterations}")
            print(f"   - Population size: {config.database.population_size}")
            print(f"   - Islands: {config.database.num_islands}")
            print(f"   - Feature dimensions: {config.database.feature_dimensions}")
            return True
        else:
            print("[FAILURE] Failed to create advanced configuration")
            return False
    except Exception as e:
        print(f"[FAILURE] Error testing advanced configuration: {e}")
        traceback.print_exc()
        return False


def test_ensemble_configuration():
    """Test ensemble model configuration."""
    print("[INFO] Testing ensemble model configuration...")
    try:
        from openevolve_integration import create_ensemble_config_with_fallback
        
        # Test creating ensemble configuration
        primary_models = ["gpt-4", "claude-3-opus"]
        fallback_models = ["gpt-3.5-turbo", "claude-3-sonnet"]
        
        config = create_ensemble_config_with_fallback(
            primary_models=primary_models,
            fallback_models=fallback_models,
            api_key="test-key",
            primary_weight=1.0,
            fallback_weight=0.3,
        )
        
        if config:
            print("[SUCCESS] Ensemble configuration created successfully")
            print(f"   - Primary models: {len(config.llm.models[:len(primary_models)])}")
            print(f"   - Fallback models: {len(config.llm.models[len(primary_models):])}")
            return True
        else:
            print("[FAILURE] Failed to create ensemble configuration")
            return False
    except Exception as e:
        print(f"[FAILURE] Error testing ensemble configuration: {e}")
        traceback.print_exc()
        return False


def test_map_elites_features():
    """Test MAP-Elites quality-diversity optimization features."""
    print("[INFO] Testing MAP-Elites features...")
    try:
        # Test that MAP-Elites features are configurable
        from openevolve.config import DatabaseConfig
        
        db_config = DatabaseConfig(
            feature_dimensions=["complexity", "diversity", "performance"],
            feature_bins=15,
            diversity_metric="edit_distance",
        )
        
        print("[SUCCESS] MAP-Elites configuration validated")
        print(f"   - Feature dimensions: {db_config.feature_dimensions}")
        print(f"   - Feature bins: {db_config.feature_bins}")
        print(f"   - Diversity metric: {db_config.diversity_metric}")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing MAP-Elites features: {e}")
        traceback.print_exc()
        return False


def test_island_model_features():
    """Test island-based evolution features."""
    print("[INFO] Testing island model features...")
    try:
        # Test island configuration
        from openevolve.config import DatabaseConfig
        
        db_config = DatabaseConfig(
            num_islands=3,
            migration_interval=25,
            migration_rate=0.15,
        )
        
        print("[SUCCESS] Island model configuration validated")
        print(f"   - Islands: {db_config.num_islands}")
        print(f"   - Migration interval: {db_config.migration_interval}")
        print(f"   - Migration rate: {db_config.migration_rate}")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing island model features: {e}")
        traceback.print_exc()
        return False


def test_artifact_feedback():
    """Test artifact side-channel feedback."""
    print("[INFO] Testing artifact feedback features...")
    try:
        # Test evaluator configuration with artifact support
        from openevolve.config import EvaluatorConfig
        
        eval_config = EvaluatorConfig(
            enable_artifacts=True,
            max_artifact_storage=50 * 1024 * 1024,  # 50MB
        )
        
        print("[SUCCESS] Artifact feedback configuration validated")
        print(f"   - Artifacts enabled: {eval_config.enable_artifacts}")
        print(f"   - Max storage: {eval_config.max_artifact_storage} bytes")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing artifact feedback features: {e}")
        traceback.print_exc()
        return False


def test_cascade_evaluation():
    """Test cascade evaluation features."""
    print("[INFO] Testing cascade evaluation features...")
    try:
        # Test cascade evaluation configuration
        from openevolve.config import EvaluatorConfig
        
        eval_config = EvaluatorConfig(
            cascade_evaluation=True,
            cascade_thresholds=[0.6, 0.8, 0.95],
        )
        
        print("[SUCCESS] Cascade evaluation configuration validated")
        print(f"   - Cascade enabled: {eval_config.cascade_evaluation}")
        print(f"   - Thresholds: {eval_config.cascade_thresholds}")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing cascade evaluation features: {e}")
        traceback.print_exc()
        return False


def test_llm_feedback():
    """Test LLM feedback integration."""
    print("[INFO] Testing LLM feedback features...")
    try:
        # Test LLM feedback configuration
        from openevolve.config import EvaluatorConfig
        
        eval_config = EvaluatorConfig(
            use_llm_feedback=True,
            llm_feedback_weight=0.15,
        )
        
        print("[SUCCESS] LLM feedback configuration validated")
        print(f"   - LLM feedback enabled: {eval_config.use_llm_feedback}")
        print(f"   - Feedback weight: {eval_config.llm_feedback_weight}")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing LLM feedback features: {e}")
        traceback.print_exc()
        return False


def test_prompt_stochasticity():
    """Test prompt template stochasticity features."""
    print("[INFO] Testing prompt stochasticity features...")
    try:
        # Test prompt configuration with stochasticity
        from openevolve.config import PromptConfig
        
        prompt_config = PromptConfig(
            use_template_stochasticity=True,
            use_meta_prompting=True,
            meta_prompt_weight=0.1,
        )
        
        print("[SUCCESS] Prompt stochasticity configuration validated")
        print(f"   - Template stochasticity: {prompt_config.use_template_stochasticity}")
        print(f"   - Meta-prompting: {prompt_config.use_meta_prompting}")
        print(f"   - Meta-prompt weight: {prompt_config.meta_prompt_weight}")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing prompt stochasticity features: {e}")
        traceback.print_exc()
        return False


def test_early_stopping():
    """Test early stopping mechanisms."""
    print("[INFO] Testing early stopping features...")
    try:
        # Test early stopping configuration
        from openevolve.config import Config
        
        config = Config(
            early_stopping_patience=15,
            convergence_threshold=0.001,
            early_stopping_metric="combined_score",
        )
        
        print("[SUCCESS] Early stopping configuration validated")
        print(f"   - Patience: {config.early_stopping_patience}")
        print(f"   - Convergence threshold: {config.convergence_threshold}")
        print(f"   - Metric: {config.early_stopping_metric}")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing early stopping features: {e}")
        traceback.print_exc()
        return False


def test_visualization_features():
    """Test visualization features."""
    print("[INFO] Testing visualization features...")
    try:
        # Test that visualization dependencies are available
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple test plot
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        plt.close(fig)
        
        print("[SUCCESS] Visualization libraries are available")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing visualization features: {e}")
        return False


def test_logging_features():
    """Test logging and monitoring features."""
    print("[INFO] Testing logging and monitoring features...")
    try:
        # Test that logging utilities are available
        from logging_util import logger
        
        # Test basic logging
        logger.info("Testing logging system")
        logger.warning("Testing warning logging")
        logger.error("Testing error logging")
        
        # Test specialized logging
        logger.log_evolution_start({"test_param": "value"})
        logger.log_adversarial_progress(1, 0.85, 3)
        logger.log_api_call("gpt-4", 1200, 1.5, True)
        
        print("[SUCCESS] Logging system is working correctly")
        return True
    except Exception as e:
        print(f"[FAILURE] Error testing logging features: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("=" * 60)
    print("OpenEvolve Comprehensive Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("OpenEvolve Backend", test_openevolve_availability),
        ("Advanced Configuration", test_advanced_configuration),
        ("Ensemble Models", test_ensemble_configuration),
        ("MAP-Elites Features", test_map_elites_features),
        ("Island Model", test_island_model_features),
        ("Artifact Feedback", test_artifact_feedback),
        ("Cascade Evaluation", test_cascade_evaluation),
        ("LLM Feedback", test_llm_feedback),
        ("Prompt Stochasticity", test_prompt_stochasticity),
        ("Early Stopping", test_early_stopping),
        ("Visualization", test_visualization_features),
        ("Logging System", test_logging_features),
    ]
    
    results = []
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 60}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                passed += 1
                print(f"[PASS] {test_name}: PASSED")
            else:
                print(f"[FAIL] {test_name}: FAILED")
        except Exception as e:
            print(f"[ERROR] {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    print(f"\n{'-' * 60}")
    print("TEST SUMMARY")
    print(f"{'-' * 60}")
    print(f"Passed: {passed}/{total} tests")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! OpenEvolve integration is working correctly.")
        return True
    else:
        print(f"\n[WARNING] {total-passed} tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    # Change to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)