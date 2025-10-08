"""
Integration test to verify that components work together.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_component_integration():
    """Test that components can work together"""
    print("OpenEvolve Integration Test")
    print("=" * 40)
    
    try:
        # Import all components
        from content_analyzer import ContentAnalyzer
        from quality_assessment import QualityAssessmentEngine
        from red_team import RedTeam
        from blue_team import BlueTeam
        from evaluator_team import EvaluatorTeam
        from configuration_system import ConfigurationManager
        from quality_assurance import QualityAssuranceOrchestrator
        from performance_optimization import PerformanceOptimizationOrchestrator
        
        print("[PASS] All components imported successfully")
        
        # Initialize components
        content_analyzer = ContentAnalyzer()
        quality_assessment = QualityAssessmentEngine()
        red_team = RedTeam()
        blue_team = BlueTeam()
        evaluator_team = EvaluatorTeam()
        config_manager = ConfigurationManager()
        qa_orchestrator = QualityAssuranceOrchestrator()
        perf_optimizer = PerformanceOptimizationOrchestrator()
        
        print("[PASS] All components instantiated successfully")
        
        # Test a simple workflow
        print("\nTesting workflow...")
        
        # 1. Analyze content
        sample_content = """
        # Sample Security Protocol
        
        This document describes a security protocol for user authentication.
        
        ## Authentication Process
        1. User enters username and password
        2. System checks credentials
        3. If valid, grant access
        4. If invalid, deny access
        
        ## Issues
        - Passwords are stored in plain text
        - No password complexity requirements
        - No account lockout mechanism
        - No multi-factor authentication
        """
        
        analysis = content_analyzer.analyze_content(sample_content)
        print(f"[PASS] Content analyzed - Type: {analysis.semantic_understanding.get('content_type', 'unknown')}")
        
        # 2. Assess quality
        quality_result = quality_assessment.assess_quality(sample_content, "document")
        print(f"[PASS] Quality assessed - Score: {quality_result.composite_score:.2f}")
        
        # 3. Red team assessment
        red_result = red_team.assess_content(sample_content, "document")
        print(f"[PASS] Red team assessment - Issues found: {len(red_result.findings)}")
        
        # 4. Blue team fixes
        blue_result = blue_team.apply_fixes(sample_content, red_result.findings, "document")
        print(f"[PASS] Blue team fixes applied - Fixes: {len(blue_result.applied_fixes)}")
        
        # 5. Evaluator team assessment
        eval_result = evaluator_team.evaluate_content(blue_result.fixed_content, "document")
        print(f"[PASS] Evaluator team assessment - Consensus: {eval_result.consensus_reached}")
        
        # 6. Configuration system
        params = config_manager.list_parameters()
        print(f"[PASS] Configuration system - Parameters: {len(params)}")
        
        # 7. Quality assurance
        qa_gates = len(qa_orchestrator.gates)
        print(f"[PASS] Quality assurance - Gates: {qa_gates}")
        
        # 8. Performance optimization
        perf_optimizers = len(perf_optimizer.optimizers)
        print(f"[PASS] Performance optimization - Optimizers: {perf_optimizers}")
        
        print("\n[SUCCESS] All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_component_integration()
    
    if success:
        print("\n[V] Integration test completed successfully!")
        print("All components are working together correctly.")
    else:
        print("\n[X] Integration test failed.")
        print("Some components are not working together correctly.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)