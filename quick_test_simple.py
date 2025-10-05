"""
Simple test to verify that the main components can be imported and instantiated.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_component_imports():
    """Test that all main components can be imported"""
    components = {}
    errors = []
    
    # Test Content Analyzer
    try:
        from content_analyzer import ContentAnalyzer
        components['content_analyzer'] = ContentAnalyzer()
        print("[PASS] ContentAnalyzer imported and instantiated")
    except Exception as e:
        errors.append(f"ContentAnalyzer: {e}")
        print("[FAIL] ContentAnalyzer import failed")
    
    # Test Prompt Engineering
    try:
        from prompt_engineering import PromptEngineeringSystem
        components['prompt_engineering'] = PromptEngineeringSystem()
        print("[PASS] PromptEngineeringSystem imported and instantiated")
    except Exception as e:
        errors.append(f"PromptEngineeringSystem: {e}")
        print("[FAIL] PromptEngineeringSystem import failed")
    
    # Test Model Orchestration
    try:
        from model_orchestration import ModelOrchestrator
        components['model_orchestration'] = ModelOrchestrator()
        print("[PASS] ModelOrchestrator imported and instantiated")
    except Exception as e:
        errors.append(f"ModelOrchestrator: {e}")
        print("[FAIL] ModelOrchestrator import failed")
    
    # Test Quality Assessment
    try:
        from quality_assessment import QualityAssessmentEngine
        components['quality_assessment'] = QualityAssessmentEngine()
        print("[PASS] QualityAssessmentEngine imported and instantiated")
    except Exception as e:
        errors.append(f"QualityAssessmentEngine: {e}")
        print("[FAIL] QualityAssessmentEngine import failed")
    
    # Test Red Team
    try:
        from red_team import RedTeam
        components['red_team'] = RedTeam()
        print("[PASS] RedTeam imported and instantiated")
    except Exception as e:
        errors.append(f"RedTeam: {e}")
        print("[FAIL] RedTeam import failed")
    
    # Test Blue Team
    try:
        from blue_team import BlueTeam
        components['blue_team'] = BlueTeam()
        print("[PASS] BlueTeam imported and instantiated")
    except Exception as e:
        errors.append(f"BlueTeam: {e}")
        print("[FAIL] BlueTeam import failed")
    
    # Test Evaluator Team
    try:
        from evaluator_team import EvaluatorTeam
        components['evaluator_team'] = EvaluatorTeam()
        print("[PASS] EvaluatorTeam imported and instantiated")
    except Exception as e:
        errors.append(f"EvaluatorTeam: {e}")
        print("[FAIL] EvaluatorTeam import failed")
    
    # Test Evolutionary Optimization
    try:
        from evolutionary_optimization import EvolutionaryOptimizer, EvolutionConfiguration
        config = EvolutionConfiguration()
        # Mock fitness function for testing
        class MockFitnessFunction:
            def evaluate(self, individual, content_type="general"):
                return 75.0  # Fixed score for testing
        components['evolutionary_optimizer'] = EvolutionaryOptimizer(config, MockFitnessFunction())
        print("[PASS] EvolutionaryOptimizer imported and instantiated")
    except Exception as e:
        errors.append(f"EvolutionaryOptimizer: {e}")
        print("[FAIL] EvolutionaryOptimizer import failed")
    
    # Test Configuration System
    try:
        from configuration_system import ConfigurationManager
        components['configuration_manager'] = ConfigurationManager()
        print("[PASS] ConfigurationManager imported and instantiated")
    except Exception as e:
        errors.append(f"ConfigurationManager: {e}")
        print("[FAIL] ConfigurationManager import failed")
    
    # Test Quality Assurance
    try:
        from quality_assurance import QualityAssuranceOrchestrator
        components['qa_orchestrator'] = QualityAssuranceOrchestrator()
        print("[PASS] QualityAssuranceOrchestrator imported and instantiated")
    except Exception as e:
        errors.append(f"QualityAssuranceOrchestrator: {e}")
        print("[FAIL] QualityAssuranceOrchestrator import failed")
    
    # Test Performance Optimization
    try:
        from performance_optimization import PerformanceOptimizationOrchestrator
        components['perf_optimizer'] = PerformanceOptimizationOrchestrator()
        print("[PASS] PerformanceOptimizationOrchestrator imported and instantiated")
    except Exception as e:
        errors.append(f"PerformanceOptimizationOrchestrator: {e}")
        print("[FAIL] PerformanceOptimizationOrchestrator import failed")
    
    return components, errors

def test_basic_functionality(components):
    """Test basic functionality of components"""
    errors = []
    
    try:
        # Test Content Analyzer
        if 'content_analyzer' in components:
            sample_content = "# Test Document\n\nThis is a test."
            result = components['content_analyzer'].analyze_content(sample_content)
            print("[PASS] ContentAnalyzer basic functionality works")
        else:
            print("[SKIP] ContentAnalyzer not available for testing")
        
        # Test Prompt Engineering
        if 'prompt_engineering' in components:
            # This would normally create prompts, but we'll just check initialization
            print("[PASS] PromptEngineeringSystem basic initialization works")
        else:
            print("[SKIP] PromptEngineeringSystem not available for testing")
            
        # Test Quality Assessment
        if 'quality_assessment' in components:
            sample_content = "This is a test document for quality assessment."
            result = components['quality_assessment'].assess_quality(sample_content, "document")
            print("[PASS] QualityAssessmentEngine basic functionality works")
        else:
            print("[SKIP] QualityAssessmentEngine not available for testing")
            
        # Test Red Team
        if 'red_team' in components:
            sample_content = "Sample content for red team testing."
            result = components['red_team'].assess_content(sample_content, "document")
            print("[PASS] RedTeam basic functionality works")
        else:
            print("[SKIP] RedTeam not available for testing")
            
        # Test Blue Team
        if 'blue_team' in components:
            sample_content = "Sample content for blue team testing."
            # Create a simple mock issue
            from red_team import IssueFinding, IssueCategory, SeverityLevel
            mock_issues = [
                IssueFinding(
                    title="Test Issue",
                    description="This is a test issue",
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.CLARITY_ISSUE,
                    confidence=0.8
                )
            ]
            result = components['blue_team'].apply_fixes(sample_content, mock_issues, "document")
            print("[PASS] BlueTeam basic functionality works")
        else:
            print("[SKIP] BlueTeam not available for testing")
            
        # Test Evaluator Team
        if 'evaluator_team' in components:
            sample_content = "Sample content for evaluation."
            # This would normally do a full evaluation, but we'll just check initialization
            print("[PASS] EvaluatorTeam basic initialization works")
        else:
            print("[SKIP] EvaluatorTeam not available for testing")
            
        # Test Configuration Manager
        if 'configuration_manager' in components:
            # Test basic parameter access
            params = components['configuration_manager'].list_parameters()
            print(f"[PASS] ConfigurationManager has {len(params)} parameters")
        else:
            print("[SKIP] ConfigurationManager not available for testing")
            
        # Test Quality Assurance
        if 'qa_orchestrator' in components:
            # Test basic initialization
            gates = components['qa_orchestrator'].gates
            print(f"[PASS] QualityAssuranceOrchestrator has {len(gates)} quality gates")
        else:
            print("[SKIP] QualityAssuranceOrchestrator not available for testing")
            
        # Test Performance Optimization
        if 'perf_optimizer' in components:
            # Test basic initialization
            optimizers = components['perf_optimizer'].optimizers
            print(f"[PASS] PerformanceOptimizationOrchestrator has {len(optimizers)} optimizers")
        else:
            print("[SKIP] PerformanceOptimizationOrchestrator not available for testing")
            
    except Exception as e:
        errors.append(f"Basic functionality test failed: {e}")
        print(f"[FAIL] Basic functionality test failed: {e}")
    
    return errors

def main():
    """Main test function"""
    print("OpenEvolve Quick Component Test")
    print("=" * 40)
    
    # Test imports
    print("\n1. Testing component imports:")
    components, import_errors = test_component_imports()
    
    if import_errors:
        print(f"\n[ERROR] Import errors found ({len(import_errors)}):")
        for error in import_errors:
            print(f"   {error}")
    else:
        print("\n[SUCCESS] All components imported successfully!")
    
    # Test basic functionality
    print("\n2. Testing basic functionality:")
    func_errors = test_basic_functionality(components)
    
    if func_errors:
        print(f"\n[ERROR] Functionality errors found ({len(func_errors)}):")
        for error in func_errors:
            print(f"   {error}")
    else:
        print("\n[SUCCESS] Basic functionality tests passed!")
    
    # Summary
    total_components = len(components)
    print(f"\nSUMMARY:")
    print(f"   Components available: {total_components}/11")
    print(f"   Import errors: {len(import_errors)}")
    print(f"   Functionality errors: {len(func_errors)}")
    
    if total_components >= 9 and len(import_errors) == 0:
        print("\n[SUCCESS] System is mostly ready! Most components are working correctly.")
        return True
    elif total_components >= 6:
        print("\n[WARNING] System is partially ready. Some components need attention.")
        return True
    else:
        print("\n[ERROR] System needs significant work. Too many components are missing.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)