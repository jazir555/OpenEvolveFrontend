"""
Test script for the updated Model Orchestration System
Verifies that all advanced features from model_orchestration1.py are properly integrated
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_model_registration():
    """Test basic model registration functionality"""
    print("Testing basic model registration...")
    
    try:
        from model_orchestration import ModelOrchestrator, ModelRole, ModelTeam
        
        # Create orchestrator
        orchestrator = ModelOrchestrator()
        
        # Register models
        orchestrator.register_model(
            model_name="gpt-4o",
            role=ModelRole.RED_TEAM,
            weight=1.0,
            api_key="test-key-1",
            api_base="https://api.openai.com/v1"
        )
        
        orchestrator.register_model(
            model_name="claude-3-opus",
            role=ModelRole.BLUE_TEAM,
            weight=1.2,
            api_key="test-key-2",
            api_base="https://api.anthropic.com/v1"
        )
        
        orchestrator.register_model(
            model_name="gemini-1.5-pro",
            role=ModelRole.EVALUATOR,
            weight=0.8,
            api_key="test-key-3",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        # Verify registration
        assert len(orchestrator.models) == 3, f"Expected 3 models, got {len(orchestrator.models)}"
        assert len(orchestrator.get_models_by_role(ModelRole.RED_TEAM)) == 1, "Red team model not registered"
        assert len(orchestrator.get_models_by_role(ModelRole.BLUE_TEAM)) == 1, "Blue team model not registered"
        assert len(orchestrator.get_models_by_role(ModelRole.EVALUATOR)) == 1, "Evaluator model not registered"
        
        # Verify advanced registry
        assert len(orchestrator.registry.models) == 3, f"Expected 3 models in registry, got {len(orchestrator.registry.models)}"
        assert len(orchestrator.registry.get_models_by_team(ModelTeam.RED)) == 1, "Red team model not in registry"
        assert len(orchestrator.registry.get_models_by_team(ModelTeam.BLUE)) == 1, "Blue team model not in registry"
        assert len(orchestrator.registry.get_models_by_team(ModelTeam.EVALUATOR)) == 1, "Evaluator model not in registry"
        
        print("[PASS] Basic model registration test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic model registration test failed: {e}")
        return False

def test_advanced_model_config():
    """Test advanced model configuration features"""
    print("Testing advanced model configuration...")
    
    try:
        from model_orchestration import ModelConfig, ModelTeam, ModelProvider
        
        # Create model config with advanced parameters
        config = ModelConfig(
            model_id="test-model-advanced",
            provider=ModelProvider.OPENAI,
            api_key="test-key",
            api_base="https://api.openai.com/v1",
            temperature=0.8,
            top_p=0.9,
            max_tokens=2048,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            seed=42,
            team=ModelTeam.RED,
            weight=1.5,
            enabled=True,
            performance_score=0.95
        )
        
        # Verify all parameters
        assert config.model_id == "test-model-advanced", "Model ID not set correctly"
        assert config.provider == ModelProvider.OPENAI, "Provider not set correctly"
        assert config.temperature == 0.8, "Temperature not set correctly"
        assert config.top_p == 0.9, "Top-p not set correctly"
        assert config.max_tokens == 2048, "Max tokens not set correctly"
        assert config.frequency_penalty == 0.1, "Frequency penalty not set correctly"
        assert config.presence_penalty == 0.1, "Presence penalty not set correctly"
        assert config.seed == 42, "Seed not set correctly"
        assert config.team == ModelTeam.RED, "Team not set correctly"
        assert config.weight == 1.5, "Weight not set correctly"
        assert config.enabled, "Enabled flag not set correctly"
        assert config.performance_score == 0.95, "Performance score not set correctly"
        
        print("[PASS] Advanced model configuration test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Advanced model configuration test failed: {e}")
        return False

def test_model_performance_tracking():
    """Test model performance tracking functionality"""
    print("Testing model performance tracking...")
    
    try:
        from model_orchestration import ModelRegistry, ModelPerformance
        
        # Create registry and performance tracker
        registry = ModelRegistry()
        performance = ModelPerformance("test-model")
        
        # Verify initial state
        assert performance.model_id == "test-model", "Model ID not set correctly"
        assert performance.total_requests == 0, "Initial total requests not zero"
        assert performance.success_requests == 0, "Initial success requests not zero"
        assert performance.avg_response_time == 0.0, "Initial avg response time not zero"
        assert performance.avg_token_usage == 0, "Initial avg token usage not zero"
        assert performance.avg_cost == 0.0, "Initial avg cost not zero"
        assert performance.issues_found == 0, "Initial issues found not zero"
        assert performance.issues_resolved == 0, "Initial issues resolved not zero"
        assert performance.evaluation_score == 0.0, "Initial evaluation score not zero"
        assert performance.performance_history == [], "Initial performance history not empty"
        
        # Test performance history tracking
        test_metrics = {
            "issues_found": 5,
            "issues_resolved": 3,
            "evaluation_score": 0.85
        }
        
        registry.update_model_performance(
            model_id="test-model",
            response_success=True,
            response_time=1.5,
            token_usage=100,
            cost=0.002,
            additional_metrics=test_metrics
        )
        
        # Verify performance was updated
        perf = registry.model_performance["test-model"]
        assert perf.total_requests == 1, "Total requests not incremented"
        assert perf.success_requests == 1, "Success requests not incremented"
        assert perf.avg_response_time > 0, "Avg response time not updated"
        assert perf.avg_token_usage > 0, "Avg token usage not updated"
        assert perf.avg_cost > 0, "Avg cost not updated"
        assert perf.issues_found == 5, "Issues found not updated"
        assert perf.issues_resolved == 3, "Issues resolved not updated"
        assert perf.evaluation_score == 0.85, "Evaluation score not updated"
        assert len(perf.performance_history) == 1, "Performance history not updated"
        
        print("[PASS] Model performance tracking test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Model performance tracking test failed: {e}")
        return False

def test_load_balancing():
    """Test load balancing functionality"""
    print("Testing load balancing...")
    
    try:
        from model_orchestration import ModelRegistry, ModelConfig, ModelTeam, ModelProvider, LoadBalancer, OrchestrationStrategy
        
        # Create registry and load balancer
        registry = ModelRegistry()
        load_balancer = LoadBalancer(registry)
        
        # Register multiple models for the same team
        for i in range(3):
            config = ModelConfig(
                model_id=f"test-model-{i}",
                provider=ModelProvider.OPENAI,
                api_key=f"test-key-{i}",
                api_base="https://api.openai.com/v1",
                team=ModelTeam.RED
            )
            registry.register_model(config)
        
        # Test round-robin selection
        model1 = load_balancer.get_next_model(ModelTeam.RED, OrchestrationStrategy.ROUND_ROBIN)
        model2 = load_balancer.get_next_model(ModelTeam.RED, OrchestrationStrategy.ROUND_ROBIN)
        model3 = load_balancer.get_next_model(ModelTeam.RED, OrchestrationStrategy.ROUND_ROBIN)
        model4 = load_balancer.get_next_model(ModelTeam.RED, OrchestrationStrategy.ROUND_ROBIN)
        
        # Should cycle through models
        assert model1 == "test-model-0", "Round-robin selection 1 failed"
        assert model2 == "test-model-1", "Round-robin selection 2 failed"
        assert model3 == "test-model-2", "Round-robin selection 3 failed"
        assert model4 == "test-model-0", "Round-robin selection 4 failed"
        
        # Test random selection
        random_model = load_balancer.get_next_model(ModelTeam.RED, OrchestrationStrategy.RANDOM_SAMPLING)
        assert random_model in ["test-model-0", "test-model-1", "test-model-2"], "Random selection failed"
        
        print("[PASS] Load balancing test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Load balancing test failed: {e}")
        return False

def test_ensemble_execution():
    """Test ensemble execution functionality"""
    print("Testing ensemble execution...")
    
    try:
        from model_orchestration import ModelOrchestrator, ModelRole
        
        # Create orchestrator
        orchestrator = ModelOrchestrator()
        
        # Register multiple models for the same role
        orchestrator.register_model(
            model_name="gpt-4o",
            role=ModelRole.RED_TEAM,
            weight=1.0,
            api_key="test-key-1"
        )
        
        orchestrator.register_model(
            model_name="claude-3-opus",
            role=ModelRole.RED_TEAM,
            weight=1.2,
            api_key="test-key-2"
        )
        
        # Test ensemble execution (this would normally make API calls)
        messages = [{"role": "user", "content": "Test content for ensemble"}]
        
        # This won't actually execute since we're not making real API calls,
        # but we can verify the method works without errors
        responses = orchestrator.execute_with_ensemble(
            messages=messages,
            role=ModelRole.RED_TEAM,
            selection_strategy="round_robin",
            temperature=0.7,
            max_tokens=1000,
            num_responses=2
        )
        
        # Should return list (even if empty due to no real API calls)
        assert isinstance(responses, list), "Ensemble execution didn't return list"
        
        print("[PASS] Ensemble execution test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Ensemble execution test failed: {e}")
        return False

def test_triad_interaction():
    """Test triad interaction functionality"""
    print("Testing triad interaction...")
    
    try:
        from model_orchestration import ModelOrchestrator, ModelRole
        
        # Create orchestrator
        orchestrator = ModelOrchestrator()
        
        # Register models for all roles
        orchestrator.register_model("gpt-4o", ModelRole.RED_TEAM, api_key="test-1")
        orchestrator.register_model("claude-3-opus", ModelRole.BLUE_TEAM, api_key="test-2")
        orchestrator.register_model("gemini-1.5-pro", ModelRole.EVALUATOR, api_key="test-3")
        
        # Test triad interaction (this would normally make API calls)
        content = "Sample content to analyze and improve"
        content_type = "document"
        
        # This won't actually execute since we're not making real API calls,
        # but we can verify the method works without errors
        results = orchestrator.execute_triad_interaction(
            content=content,
            content_type=content_type,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Should return dict with expected keys
        assert isinstance(results, dict), "Triad interaction didn't return dict"
        assert "red_team_analysis" in results, "Red team analysis missing from results"
        assert "blue_team_resolution" in results, "Blue team resolution missing from results"
        assert "evaluator_assessment" in results, "Evaluator assessment missing from results"
        
        print("[PASS] Triad interaction test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Triad interaction test failed: {e}")
        return False

def test_performance_reports():
    """Test performance reporting functionality"""
    print("Testing performance reporting...")
    
    try:
        from model_orchestration import ModelOrchestrator, ModelRole, ModelTeam
        
        # Create orchestrator and register models
        orchestrator = ModelOrchestrator()
        orchestrator.register_model("gpt-4o", ModelRole.RED_TEAM, api_key="test")
        orchestrator.register_model("claude-3-opus", ModelRole.BLUE_TEAM, api_key="test")
        orchestrator.register_model("gemini-1.5-pro", ModelRole.EVALUATOR, api_key="test")
        
        # Test model performance metrics
        metrics = orchestrator.get_model_performance_metrics()
        assert isinstance(metrics, dict), "Performance metrics didn't return dict"
        
        # Test specific model metrics
        model_metrics = orchestrator.get_model_performance_metrics("gpt-4o")
        assert isinstance(model_metrics, dict), "Specific model metrics didn't return dict"
        assert model_metrics["model"] == "gpt-4o", "Model name not correct in metrics"
        
        # Test team performance report
        team_report = orchestrator.get_team_performance_report(ModelTeam.RED)
        assert isinstance(team_report, dict), "Team performance report didn't return dict"
        assert team_report["team"] == "red", "Team name not correct in report"
        
        # Test orchestration efficiency metrics
        efficiency_metrics = orchestrator.get_orchestration_efficiency_metrics()
        assert isinstance(efficiency_metrics, dict), "Efficiency metrics didn't return dict"
        
        print("[PASS] Performance reporting test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Performance reporting test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("Model Orchestration System Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_model_registration,
        test_advanced_model_config,
        test_model_performance_tracking,
        test_load_balancing,
        test_ensemble_execution,
        test_triad_interaction,
        test_performance_reports
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    print()
    
    if failed == 0:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("Model Orchestration System is fully integrated with advanced features.")
        return True
    else:
        print(f"[WARNING] {failed} TEST(S) FAILED")
        print("Some features may not be working correctly.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)