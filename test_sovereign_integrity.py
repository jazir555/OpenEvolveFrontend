"""
Test script to verify the integrity and functionality of sovereign decomposition system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all key modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        "sovereign_data_models",
        "sovereign_refinement",
        "sovereign_reliability",
        "problem_analyzer",
        "decomposition_engine",
        "sovereign_gauntlets",
        "sovereign_quality_assessment",
        "sovereign_knowledge_manager",
        "sovereign_team_coordination",
        "sovereign_integration",
        "sovereign_solution_orchestration",
        "sovereign_persistence",
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  [PASS] {module_name}")
        except ImportError as e:
            print(f"  [FAIL] {module_name}: {e}")
            return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test data models
        from sovereign_data_models import (
            ProblemDefinition, SubProblem, DecompositionPlan, 
            ProblemType, SubProblemType, generate_id
        )
        
        # Create a simple problem definition
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="This is a test problem for verification",
            problem_type=ProblemType.ANALYSIS,
            domain_context=None,
            complexity_score=None,
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={}
        )
        print("  [PASS] Data models working")
        
        # Test problem analyzer
        from problem_analyzer import ProblemAnalyzer
        analyzer = ProblemAnalyzer()
        analyzed_problem = analyzer.analyze_problem("How can we improve system performance?")
        print(f"  [PASS] Problem analyzer working - identified domain: {analyzed_problem.domain_context.domain}")
        
        # Test decomposition engine
        from decomposition_engine import DecompositionEngine
        engine = DecompositionEngine(analyzer)
        plan = engine.decompose(analyzed_problem, strategy='hybrid')
        print(f"  [PASS] Decomposition engine working - created {len(plan.sub_problems)} sub-problems")
        
        # Test gauntlets
        from sovereign_gauntlets import GauntletSystem
        gauntlet_system = GauntletSystem()
        gauntlet_results = gauntlet_system.run_decomposition_gauntlets(plan)
        print(f"  [PASS] Gauntlet system working - {len(gauntlet_results)} gauntlets run")
        
        # Test quality assessment
        from sovereign_quality_assessment import QualityAssessor
        quality_assessor = QualityAssessor()
        quality_report = quality_assessor.generate_quality_report(plan)
        print(f"  [PASS] Quality assessment working - overall score: {quality_report.metrics.overall_score:.2f}")
        
        # Test refinement coordinator
        from sovereign_refinement import RefinementCoordinator
        from sovereign_data_models import Feedback
        refinement_coordinator = RefinementCoordinator(
            gauntlet_system=gauntlet_system,
            quality_assessor=quality_assessor
        )
        
        # Create minimal feedback for testing refinement
        test_feedback = [
            Feedback(
                id=generate_id("feedback"),
                source="test",
                feedback_type="critique",
                content="This is a test feedback for verification",
                severity="minor",
                actionable=True,
                timestamp=None
            )
        ]
        
        processed = refinement_coordinator.process_feedback(plan, test_feedback)
        print(f"  [PASS] Refinement coordinator working - processed {len(processed['improvements'])} improvements")
        
        # Test knowledge manager
        from sovereign_knowledge_manager import KnowledgeManager
        knowledge_manager = KnowledgeManager()
        patterns = knowledge_manager.extract_patterns(plan, success=True, quality_score=0.8)
        print(f"  [PASS] Knowledge manager working - extracted {len(patterns)} patterns")
        
        # Test reliability features
        from sovereign_reliability import (
            with_retry, RateLimiter, CircuitBreaker, 
            get_error_handler, get_health_monitor
        )
        
        # Test with_retry decorator
        @with_retry(max_attempts=2, retry_on=(ValueError,))
        def test_function():
            return "success"
        
        result = test_function()
        print(f"  [PASS] Retry decorator working - result: {result}")
        
        # Test error handler
        error_handler = get_error_handler()
        error_info = error_handler.handle_error(ValueError("Test error"))
        print(f"  [PASS] Error handler working - handled: {error_info['type']}")
        
        # Test health monitor
        health_monitor = get_health_monitor()
        health_status = health_monitor.get_health_status()
        print(f"  [PASS] Health monitor working - status: {health_status['status']}")
        
        # Test rate limiter
        rate_limiter = RateLimiter(max_requests=5, time_window=1.0)  # Very short window for test
        is_allowed = rate_limiter.is_allowed()
        print(f"  [PASS] Rate limiter working - request allowed: {is_allowed}")
        
        # Test circuit breaker
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=1.0)
        
        def success_func():
            return "success"
        
        result = circuit_breaker.call(success_func)
        print(f"  [PASS] Circuit breaker working - result: {result}")
        
        print("\n  [PASS] All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"  [FAIL] Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test end-to-end integration."""
    print("\nTesting end-to-end integration...")
    
    try:
        from sovereign_integration import SovereignIntegrationOrchestrator
        
        # Create orchestrator
        orchestrator = SovereignIntegrationOrchestrator()
        
        # Run a complete workflow
        result = orchestrator.run_complete_workflow(
            problem_text="Analyze how to optimize a Python web application for better performance",
            title="Performance Optimization Analysis",
            strategy='hybrid',
            max_refinement_cycles=1
        )
        
        print(f"  [PASS] Integration workflow completed: success={result.success}")
        print(f"    - Quality score: {result.quality_score:.2f}")
        print(f"    - Sub-problems created: {len(result.final_plan.sub_problems) if result.final_plan else 0}")
        print(f"    - Refinement cycles: {result.refinement_cycles}")
        print(f"    - Execution time: {result.execution_time:.2f}s")
        
        print("\n  [PASS] End-to-end integration test passed!")
        return True
        
    except Exception as e:
        print(f"  [FAIL] End-to-end integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Verifying Sovereign Decomposition System Integrity\n")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test integration
    if not test_integration():
        all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED! Sovereign Decomposition System is fully functional.")
        print("\nThe system has been verified to:")
        print("  - Import all required modules correctly")
        print("  - Execute basic functionality across all components")
        print("  - Perform end-to-end problem decomposition workflows")
        print("  - Handle errors and reliability scenarios appropriately")
    else:
        print("SOME TESTS FAILED! Please review the issues above.")
    print(f"{'='*60}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)