"""
System Test for OpenEvolve
Tests the complete integrated system functionality.
"""
import time
from typing import Dict, Any
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all components
try:
    from content_analyzer import ContentAnalyzer
    from prompt_engineering import PromptEngineeringSystem
    from model_orchestration import ModelOrchestrator
    from quality_assessment import QualityAssessmentEngine
    from red_team import RedTeam
    from blue_team import BlueTeam
    from evaluator_team import EvaluatorTeam
    from evolutionary_optimization import EvolutionaryOptimizer, EvolutionConfiguration
    from configuration_system import ConfigurationManager
    from quality_assurance import QualityAssuranceOrchestrator
    from performance_optimization import PerformanceOptimizationOrchestrator
    
    # Import enums and data classes
    from evolutionary_optimization import EvolutionStrategy, SelectionMethod, CrossoverOperator, MutationOperator
    from quality_assessment import SeverityLevel
    from red_team import RedTeamStrategy
    from blue_team import FixType
    from quality_assurance import QualityGateType
    
    logger.info("All components imported successfully")
    
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    raise

class SystemTester:
    """Complete system tester for OpenEvolve"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        # Initialize configuration
        self.config_manager = ConfigurationManager()
        
        # Initialize performance optimization
        self.perf_optimizer = PerformanceOptimizationOrchestrator()
        
        # Initialize quality assurance
        self.qa_orchestrator = QualityAssuranceOrchestrator()
        
        # Initialize core components
        self.content_analyzer = ContentAnalyzer()
        self.prompt_engineering = PromptEngineeringSystem()
        self.model_orchestrator = ModelOrchestrator()
        self.quality_assessment = QualityAssessmentEngine()
        self.red_team = RedTeam()
        self.blue_team = BlueTeam()
        self.evaluator_team = EvaluatorTeam()
        
        # Initialize evolutionary optimizer
        evolution_config = EvolutionConfiguration(
            population_size=20,
            num_generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism_rate=0.1,
            selection_method=SelectionMethod.TOURNAMENT,
            tournament_size=3,
            crossover_operator=CrossoverOperator.SINGLE_POINT,
            mutation_operator=MutationOperator.BIT_FLIP,
            strategy=EvolutionStrategy.GENERATIONAL
        )
        
        # Mock fitness function for testing
        class MockFitnessFunction:
            def evaluate(self, individual, content_type="general"):
                # Simple fitness based on content length and quality indicators
                content = individual.genome
                word_count = len(content.split())
                sentence_count = len(content.split('.')) + len(content.split('!')) + len(content.split('?'))
                
                # Base score
                score = 50.0
                
                # Adjust based on content characteristics
                if word_count > 100:
                    score += 20
                if sentence_count > 5:
                    score += 15
                if 'introduction' in content.lower() or 'conclusion' in content.lower():
                    score += 15
                
                return min(100.0, score)
        
        self.evolutionary_optimizer = EvolutionaryOptimizer(
            config=evolution_config,
            fitness_function=MockFitnessFunction()
        )
        
        logger.info("All components initialized successfully")
    
    def test_content_analyzer(self) -> Dict[str, Any]:
        """Test the Content Analyzer component"""
        logger.info("Testing Content Analyzer...")
        start_time = time.time()
        
        try:
            # Test with sample content
            sample_content = """
            # Sample Technical Document
            
            ## Introduction
            This is a sample technical document for testing the content analyzer.
            
            ## Main Content
            The document contains multiple sections and subsections to demonstrate 
            the analyzer's capabilities.
            
            ## Conclusion
            This concludes our sample document for testing purposes.
            """
            
            # Analyze content
            analysis_result = self.content_analyzer.analyze_content(sample_content)
            
            # Verify results
            assert analysis_result is not None
            assert hasattr(analysis_result, 'input_parsing')
            assert hasattr(analysis_result, 'semantic_understanding')
            assert hasattr(analysis_result, 'pattern_recognition')
            assert hasattr(analysis_result, 'metadata_extraction')
            
            execution_time = time.time() - start_time
            logger.info(f"Content Analyzer test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "ContentAnalyzer",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "word_count": analysis_result.input_parsing.get('word_count', 0),
                    "sentence_count": analysis_result.input_parsing.get('sentence_count', 0),
                    "content_type": analysis_result.semantic_understanding.get('content_type', 'unknown'),
                    "overall_score": analysis_result.overall_score
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Content Analyzer test failed: {e}")
            return {
                "component": "ContentAnalyzer",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_prompt_engineering(self) -> Dict[str, Any]:
        """Test the Prompt Engineering System"""
        logger.info("Testing Prompt Engineering System...")
        start_time = time.time()
        
        try:
            # Test template instantiation
            variables = {
                'content': 'Sample content for testing',
                'content_type': 'document'
            }
            
            context = {
                'target_audience': 'developers'
            }
            
            # Instantiate critique prompt
            critique_prompt = self.prompt_engineering.prompt_manager.instantiate_prompt(
                'red_team_critique', variables, context
            )
            
            # Verify prompt was created
            assert critique_prompt is not None
            assert hasattr(critique_prompt, 'rendered_prompt')
            assert len(critique_prompt.rendered_prompt) > 0
            
            execution_time = time.time() - start_time
            logger.info(f"Prompt Engineering test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "PromptEngineering",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "template_name": critique_prompt.template_name,
                    "prompt_length": len(critique_prompt.rendered_prompt),
                    "variables_used": list(critique_prompt.variables_used.keys())
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Prompt Engineering test failed: {e}")
            return {
                "component": "PromptEngineering",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_quality_assessment(self) -> Dict[str, Any]:
        """Test the Quality Assessment Engine"""
        logger.info("Testing Quality Assessment Engine...")
        start_time = time.time()
        
        try:
            # Test with sample content
            sample_content = """
            # Quality Assessment Test Document
            
            This document is used to test the quality assessment engine.
            It contains various sections to evaluate different quality dimensions.
            
            ## Structure
            The document has a clear structure with headers and content.
            
            ## Clarity
            The content should be clear and easy to understand.
            """
            
            # Assess quality
            assessment_result = self.quality_assessment.assess_quality(
                sample_content, "document"
            )
            
            # Verify results
            assert assessment_result is not None
            assert hasattr(assessment_result, 'scores')
            assert hasattr(assessment_result, 'composite_score')
            assert isinstance(assessment_result.scores, dict)
            
            execution_time = time.time() - start_time
            logger.info(f"Quality Assessment test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "QualityAssessment",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "composite_score": assessment_result.composite_score,
                    "dimensions_assessed": len(assessment_result.scores),
                    "issues_found": len(assessment_result.issues),
                    "recommendations_count": len(assessment_result.recommendations)
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quality Assessment test failed: {e}")
            return {
                "component": "QualityAssessment",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_red_team(self) -> Dict[str, Any]:
        """Test the Red Team (Critics) functionality"""
        logger.info("Testing Red Team functionality...")
        start_time = time.time()
        
        try:
            # Test with sample code that has issues
            sample_code = """
            def authenticate_user(username, password):
                # This function has security vulnerabilities
                if username == "admin" and password == "password123":
                    return True
                return False
            
            def process_data(data):
                # This function uses eval() which is dangerous
                result = eval(data)
                return result
            """
            
            # Assess content with red team
            assessment = self.red_team.assess_content(
                sample_code, "code", 
                strategy=RedTeamStrategy.SYSTEMATIC
            )
            
            # Verify results
            assert assessment is not None
            assert hasattr(assessment, 'findings')
            assert isinstance(assessment.findings, list)
            
            execution_time = time.time() - start_time
            logger.info(f"Red Team test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "RedTeam",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "findings_count": len(assessment.findings),
                    "confidence_score": assessment.confidence_score,
                    "critical_issues": len([f for f in assessment.findings 
                                           if f.severity == SeverityLevel.CRITICAL]),
                    "high_issues": len([f for f in assessment.findings 
                                      if f.severity == SeverityLevel.HIGH])
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Red Team test failed: {e}")
            return {
                "component": "RedTeam",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_blue_team(self) -> Dict[str, Any]:
        """Test the Blue Team (Fixers) functionality"""
        logger.info("Testing Blue Team functionality...")
        start_time = time.time()
        
        try:
            # Create mock issues for blue team to fix
            from red_team import IssueFinding, IssueCategory, SeverityLevel
            
            mock_issues = [
                IssueFinding(
                    title="Hardcoded credentials",
                    description="Password is hardcoded in the authenticate function",
                    severity=SeverityLevel.CRITICAL,
                    category=IssueCategory.SECURITY_VULNERABILITY,
                    confidence=0.95
                ),
                IssueFinding(
                    title="Use of eval() function",
                    description="Using eval() is dangerous and can lead to code execution",
                    severity=SeverityLevel.CRITICAL,
                    category=IssueCategory.SECURITY_VULNERABILITY,
                    confidence=0.9
                )
            ]
            
            # Sample vulnerable code
            vulnerable_code = """
            def authenticate_user(username, password):
                if username == "admin" and password == "password123":
                    return True
                return False
            
            def process_data(data):
                result = eval(data)
                return result
            """
            
            # Apply fixes with blue team
            assessment = self.blue_team.apply_fixes(
                vulnerable_code, mock_issues, "code"
            )
            
            # Verify results
            assert assessment is not None
            assert hasattr(assessment, 'applied_fixes')
            assert isinstance(assessment.applied_fixes, list)
            
            execution_time = time.time() - start_time
            logger.info(f"Blue Team test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "BlueTeam",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "fixes_applied": len(assessment.applied_fixes),
                    "improvement_score": assessment.overall_improvement_score,
                    "fix_suggestions": len(assessment.fix_suggestions),
                    "suggested_fixes": len([f for f in assessment.fix_suggestions 
                                          if f.fix_type in [FixType.SECURITY_PATCH, FixType.INPUT_VALIDATION]])
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Blue Team test failed: {e}")
            return {
                "component": "BlueTeam",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_evaluator_team(self) -> Dict[str, Any]:
        """Test the Evaluator Team (Judges) functionality"""
        logger.info("Testing Evaluator Team functionality...")
        start_time = time.time()
        
        try:
            # Test with sample content
            sample_content = """
            # Sample Document for Evaluation
            
            This is a sample document that demonstrates various content qualities.
            It has a clear structure with introduction, body, and conclusion sections.
            
            ## Introduction
            This section introduces the topic and provides context.
            
            ## Body
            The main content discusses the key points in detail.
            
            ## Conclusion
            This section summarizes the main points and provides closure.
            """
            
            # Evaluate content
            evaluation = self.evaluator_team.evaluate_content(
                sample_content, "document"
            )
            
            # Verify results
            assert evaluation is not None
            assert hasattr(evaluation, 'assessments')
            assert isinstance(evaluation.assessments, list)
            assert len(evaluation.assessments) > 0
            
            execution_time = time.time() - start_time
            logger.info(f"Evaluator Team test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "EvaluatorTeam",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "assessments_count": len(evaluation.assessments),
                    "consensus_score": evaluation.consensus_score,
                    "consensus_reached": evaluation.consensus_reached,
                    "final_verdict": evaluation.final_verdict,
                    "evaluators_involved": len(evaluation.assessments)
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Evaluator Team test failed: {e}")
            return {
                "component": "EvaluatorTeam",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_evolutionary_optimization(self) -> Dict[str, Any]:
        """Test the Evolutionary Optimization framework"""
        logger.info("Testing Evolutionary Optimization framework...")
        start_time = time.time()
        
        try:
            # Test with sample content
            sample_content = """
            # Sample Document for Evolution
            
            This is a basic document that will be evolved through the 
            evolutionary optimization process. It contains multiple sections
            and should improve through successive generations.
            
            ## Section 1
            This is the first section of the document.
            
            ## Section 2
            This is the second section with more detailed information.
            
            ## Conclusion
            This concludes our sample document.
            """
            
            # Run evolution
            evolution_result = self.evolutionary_optimizer.evolve(
                sample_content, "document", max_generations=3
            )
            
            # Verify results
            assert evolution_result is not None
            assert hasattr(evolution_result, 'best_individual')
            assert hasattr(evolution_result, 'final_population')
            
            execution_time = time.time() - start_time
            logger.info(f"Evolutionary Optimization test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "EvolutionaryOptimization",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "generations_completed": len(evolution_result.evolution_history),
                    "final_fitness": evolution_result.best_individual.fitness,
                    "population_size": len(evolution_result.final_population.individuals),
                    "fitness_improvement": evolution_result.fitness_improvement,
                    "diversity_score": evolution_result.final_population.diversity_score
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Evolutionary Optimization test failed: {e}")
            return {
                "component": "EvolutionaryOptimization",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_configuration_system(self) -> Dict[str, Any]:
        """Test the Configuration Parameters System"""
        logger.info("Testing Configuration System...")
        start_time = time.time()
        
        try:
            # Test parameter retrieval
            model_temp = self.config_manager.get_parameter_value("model_temperature")
            pop_size = self.config_manager.get_parameter_value("evolution_population_size")
            
            # Test parameter setting
            self.config_manager.set_parameter_value("model_temperature", 0.8)
            new_temp = self.config_manager.get_parameter_value("model_temperature")
            
            # Create test profile
            test_profile = self.config_manager.create_profile(
                name="test_profile",
                description="Test configuration profile",
                parameters={"model_temperature": 0.9, "model_max_tokens": 8192},
                tags=["test", "temporary"]
            )
            
            # Verify operations
            assert model_temp is not None
            assert pop_size is not None
            assert new_temp == 0.8
            assert test_profile is not None
            
            execution_time = time.time() - start_time
            logger.info(f"Configuration System test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "ConfigurationSystem",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "parameters_tested": 3,
                    "profiles_created": 1,
                    "model_temperature_initial": model_temp,
                    "model_temperature_updated": new_temp,
                    "population_size": pop_size
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Configuration System test failed: {e}")
            return {
                "component": "ConfigurationSystem",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_quality_assurance(self) -> Dict[str, Any]:
        """Test the Quality Assurance Mechanisms"""
        logger.info("Testing Quality Assurance Mechanisms...")
        start_time = time.time()
        
        try:
            # Test with content that has issues
            problematic_content = "<script>alert('XSS')</script> Please review this document."
            
            # Validate through input gate
            qa_result = self.qa_orchestrator.validate_through_gate(
                QualityGateType.MODEL_INPUT,
                problematic_content,
                {"content_type": "document", "source": "user_input"}
            )
            
            # Verify results
            assert qa_result is not None
            assert hasattr(qa_result, 'overall_status')
            assert hasattr(qa_result, 'blocking_issues')
            
            execution_time = time.time() - start_time
            logger.info(f"Quality Assurance test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "QualityAssurance",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "gate_results": len(qa_result.results),
                    "blocking_issues_found": len(qa_result.blocking_issues),
                    "overall_status": qa_result.overall_status.value,
                    "recommendations_generated": len(qa_result.recommendations),
                    "security_issues": len([r for r in qa_result.results 
                                          if 'security' in str(r.issues).lower()])
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quality Assurance test failed: {e}")
            return {
                "component": "QualityAssurance",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_performance_optimization(self) -> Dict[str, Any]:
        """Test the Performance Optimization Techniques"""
        logger.info("Testing Performance Optimization Techniques...")
        start_time = time.time()
        
        try:
            # Test with large dataset
            large_data = list(range(10000))
            context = {"operation": "performance_test", "chunk_size": 1000}
            
            # Apply optimizations
            optimized_data = self.perf_optimizer.apply_optimizations(
                large_data, context
            )
            
            # Measure performance
            perf_metrics = self.perf_optimizer.measure_overall_performance(
                large_data, context
            )
            
            # Verify results
            assert optimized_data is not None
            assert len(optimized_data) == len(large_data)
            assert isinstance(perf_metrics, dict)
            
            execution_time = time.time() - start_time
            logger.info(f"Performance Optimization test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "PerformanceOptimization",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "optimizers_tested": len(perf_metrics),
                    "data_size": len(large_data),
                    "optimization_applied": optimized_data == large_data,  # Should be same data
                    "performance_metrics": len(perf_metrics),
                    "memory_optimized": "MemoryManagement" in str(perf_metrics.keys())
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance Optimization test failed: {e}")
            return {
                "component": "PerformanceOptimization",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def test_integrated_workflow(self) -> Dict[str, Any]:
        """Test the complete integrated workflow"""
        logger.info("Testing Integrated Workflow...")
        start_time = time.time()
        
        try:
            # Sample content that needs improvement
            initial_content = """
            # Security Protocol
            
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
            
            # Step 1: Analyze content
            analysis = self.content_analyzer.analyze_content(initial_content)
            
            # Step 2: Assess quality
            quality_assessment = self.quality_assessment.assess_quality(
                initial_content, "document"
            )
            
            # Step 3: Red team assessment
            red_assessment = self.red_team.assess_content(
                initial_content, "document"
            )
            
            # Step 4: Blue team fixes
            blue_assessment = self.blue_team.apply_fixes(
                initial_content, red_assessment.findings, "document"
            )
            
            # Step 5: Evaluator team assessment
            evaluator_assessment = self.evaluator_team.evaluate_content(
                blue_assessment.fixed_content, "document"
            )
            
            # Step 6: Evolutionary optimization
            evolution_result = self.evolutionary_optimizer.evolve(
                blue_assessment.fixed_content, "document", max_generations=2
            )
            
            # Verify all steps completed successfully
            assert analysis is not None
            assert quality_assessment is not None
            assert red_assessment is not None
            assert blue_assessment is not None
            assert evaluator_assessment is not None
            assert evolution_result is not None
            
            execution_time = time.time() - start_time
            logger.info(f"Integrated Workflow test completed in {execution_time:.2f} seconds")
            
            return {
                "component": "IntegratedWorkflow",
                "status": "PASSED",
                "execution_time": execution_time,
                "details": {
                    "steps_completed": 6,
                    "initial_content_length": len(initial_content),
                    "final_content_length": len(evolution_result.best_individual.genome),
                    "quality_improvement": evolution_result.best_individual.fitness > 50,
                    "issues_identified": len(red_assessment.findings),
                    "fixes_applied": len(blue_assessment.applied_fixes),
                    "evaluator_consensus": evaluator_assessment.consensus_reached
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Integrated Workflow test failed: {e}")
            return {
                "component": "IntegratedWorkflow",
                "status": "FAILED",
                "execution_time": execution_time,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        logger.info("Starting complete system test suite...")
        
        # Run individual component tests
        tests = [
            self.test_content_analyzer,
            self.test_prompt_engineering,
            self.test_quality_assessment,
            self.test_red_team,
            self.test_blue_team,
            self.test_evaluator_team,
            self.test_evolutionary_optimization,
            self.test_configuration_system,
            self.test_quality_assurance,
            self.test_performance_optimization,
            self.test_integrated_workflow
        ]
        
        results = []
        passed_count = 0
        failed_count = 0
        
        for test_func in tests:
            try:
                result = test_func()
                results.append(result)
                
                if result["status"] == "PASSED":
                    passed_count += 1
                    logger.info(f"✓ {result['component']}: {result['status']}")
                else:
                    failed_count += 1
                    logger.error(f"✗ {result['component']}: {result['status']} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_count += 1
                error_result = {
                    "component": test_func.__name__,
                    "status": "ERROR",
                    "execution_time": 0,
                    "error": str(e)
                }
                results.append(error_result)
                logger.error(f"✗ {test_func.__name__}: ERROR - {e}")
        
        # Calculate overall statistics
        total_time = time.time() - self.start_time
        overall_status = "PASSED" if failed_count == 0 else "FAILED"
        
        summary = {
            "overall_status": overall_status,
            "total_tests": len(results),
            "passed_tests": passed_count,
            "failed_tests": failed_count,
            "error_tests": len([r for r in results if r["status"] == "ERROR"]),
            "pass_rate": (passed_count / max(1, len(results))) * 100,
            "total_execution_time": total_time,
            "test_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"System test suite completed: {overall_status}")
        logger.info(f"Passed: {passed_count}/{len(results)} ({summary['pass_rate']:.1f}%)")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        return summary
    
    def generate_test_report(self, summary: Dict[str, Any]) -> str:
        """Generate a detailed test report"""
        report_lines = [
            "=" * 80,
            "OPENEVOLVE COMPLETE SYSTEM TEST REPORT",
            "=" * 80,
            f"Generated: {summary['timestamp']}",
            f"Overall Status: {summary['overall_status']}",
            f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['pass_rate']:.1f}%)",
            f"Total Execution Time: {summary['total_execution_time']:.2f} seconds",
            "-" * 80,
            "DETAILED RESULTS:",
            "-" * 80
        ]
        
        for result in summary["test_results"]:
            status_icon = "✓" if result["status"] == "PASSED" else "✗" if result["status"] == "FAILED" else "!"
            report_lines.append(f"{status_icon} {result['component']:<25} {result['status']:<10} {result['execution_time']:>8.2f}s")
            
            if result["status"] == "PASSED" and "details" in result:
                for key, value in result["details"].items():
                    report_lines.append(f"    {key}: {value}")
            elif result["status"] in ["FAILED", "ERROR"] and "error" in result:
                report_lines.append(f"    Error: {result['error']}")
        
        report_lines.extend([
            "-" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)

def main():
    """Main function to run system tests"""
    print("OpenEvolve Complete System Test")
    print("=" * 50)
    
    # Create system tester
    tester = SystemTester()
    
    # Run all tests
    print("Running comprehensive system tests...")
    summary = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_test_report(summary)
    print("\n" + report)
    
    # Save report to file
    report_filename = f"system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {report_filename}")
    except Exception as e:
        print(f"\nWarning: Could not save report to file: {e}")
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  Status: {summary['overall_status']}")
    print(f"  Passed: {summary['passed_tests']}/{summary['total_tests']} tests")
    print(f"  Time: {summary['total_execution_time']:.2f} seconds")
    
    if summary['overall_status'] == 'PASSED':
        print("\n🎉 All tests passed! The system is functioning correctly.")
    else:
        print(f"\n⚠️  {summary['failed_tests']} tests failed. Please review the report above.")
    
    return summary['overall_status'] == 'PASSED'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)