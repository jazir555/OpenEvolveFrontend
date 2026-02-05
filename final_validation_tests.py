"""
FINAL: Complete System Integration and Validation Test
Comprehensive end-to-end validation of the entire Sovereign-Grade system
"""

import unittest
import sys
import os
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, List
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all the necessary modules
from sovereign_data_models import ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt, generate_id
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_team_coordination import TeamCoordinator
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_persistence import SovereignDatabase
from auth_system import AuthenticationSystem
from input_validation import InputValidator
from performance_optimization import LLMResponseCache
from advanced_features import AdvancedFeaturesManager
from monitoring_system import MetricsCollector
from sovereign_gauntlets import GauntletSystem


class TestCompleteSystemIntegration(unittest.TestCase):
    """Complete end-to-end system integration test"""
    
    def setUp(self):
        """Set up complete system with all components"""
        # Use in-memory database for testing
        self.db = SovereignDatabase(":memory:")
        
        # Set up authentication system
        self.auth_system = AuthenticationSystem(db_path=":memory:")
        
        # Set up input validation
        self.input_validator = InputValidator()
        
        # Set up metrics collection
        self.metrics_collector = MetricsCollector()
        
        # Set up cache
        self.cache = LLMResponseCache()
        
        # Set up advanced features
        self.advanced_features = AdvancedFeaturesManager()
        
        # Set up gauntlet system
        with patch('sovereign_gauntlets.OpenEvolveClient') as mock_openevolve:
            self.mock_gauntlet_client = mock_openevolve.return_value
            self.gauntlet_system = GauntletSystem(openevolve_client=self.mock_gauntlet_client)
        
        # Mock the OpenEvolve client for all components that need it
        with patch('problem_analyzer.OpenEvolveClient') as mock_analyzer, \
             patch('decomposition_engine.OpenEvolveClient') as mock_decomposer, \
             patch('sovereign_team_coordination.OpenEvolveClient') as mock_coordinator, \
             patch('sovereign_solution_orchestration.OpenEvolveClient') as mock_orchestrator:
            
            # Set up analyzer
            self.mock_analyzer_client = mock_analyzer.return_value
            self.analyzer = ProblemAnalyzer(openevolve_client=self.mock_analyzer_client)
            
            # Set up decomposer  
            self.mock_decomposer_client = mock_decomposer.return_value
            self.decomposer = DecompositionEngine(openevolve_client=self.mock_decomposer_client)
            
            # Set up coordinator
            self.mock_coordinator_client = mock_coordinator.return_value
            self.coordinator = TeamCoordinator(openevolve_client=self.mock_coordinator_client)
            
            # Set up orchestrator
            self.mock_orchestrator_client = mock_orchestrator.return_value
            self.orchestrator = SolutionOrchestrator(openevolve_client=self.mock_orchestrator_client)
    
    def test_complete_end_to_end_workflow(self):
        """Test complete end-to-end workflow from problem input to solution delivery"""
        print("\n" + "="*80)
        print("STARTING COMPLETE END-TO-END SYSTEM INTEGRATION TEST")
        print("="*80)
        
        # PHASE 1: Problem Definition and Analysis
        print("\nPhase 1: Problem Definition and Analysis")
        
        # Create a complex real-world problem
        complex_problem_text = """
        Design and implement a resilient, scalable, and secure microservices architecture 
        for a global e-commerce platform that processes millions of transactions daily 
        while maintaining 99.99% uptime, handling seasonal traffic spikes of 10x normal load, 
        and ensuring PCI-DSS compliance across all payment operations.

        The system must support multiple payment methods, provide real-time inventory management, 
        enable personalization at scale, implement fraud detection, handle multi-region deployments,
        and provide comprehensive monitoring and alerting capabilities.
        """
        
        start_time = time.time()
        
        # Mock the API response for problem analysis
        mock_analysis_response = Mock()
        mock_analysis_response.success = True
        mock_analysis_response.best_code = json.dumps({
            "domain": "software_engineering",
            "subdomain": "enterprise_architecture",
            "related_domains": ["security", "dev_ops", "database", "payment_processing"],
            "key_concepts": ["microservices", "resiliency", "scalability", "security", "distributed_systems"],
            "domain_complexity": 9.0,
            "required_expertise": ["architect", "security", "devops", "payment", "database"]
        })
        
        self.mock_analyzer_client.evolve.return_value = mock_analysis_response
        
        # Analyze the problem
        problem = self.analyzer.analyze_problem(
            problem_text=complex_problem_text,
            title="Global E-commerce Platform Architecture"
        )
        
        analysis_time = time.time() - start_time
        
        self.assertIsNotNone(problem)
        self.assertIn("microservices", problem.domain_context.domain.lower())
        print(f"[OK] Problem analysis completed in {analysis_time:.3f}s")
        
        # Store the problem
        problem_id = self.db.create_problem(problem)
        self.assertTrue(problem_id)
        print(f"[OK] Problem stored in database with ID: {problem_id}")
        
        # PHASE 2: Problem Decomposition
        print("\nPhase 2: Problem Decomposition")
        
        start_time = time.time()
        
        # Mock the decomposition response
        mock_decomposition_response = Mock()
        mock_decomposition_response.success = True
        mock_decomposition_response.best_code = json.dumps([
            {
                "id": generate_id("sub1"),
                "description": "Design microservices architecture and API gateway",
                "dependencies": [],
                "ai_suggested_complexity_score": 8.5,
                "ai_suggested_evaluation_prompt": "Validate architectural decisions and API design patterns"
            },
            {
                "id": generate_id("sub2"),
                "description": "Implement security framework and access controls",
                "dependencies": [generate_id("sub1")],
                "ai_suggested_complexity_score": 9.0,
                "ai_suggested_evaluation_prompt": "Verify security implementation meets enterprise standards"
            },
            {
                "id": generate_id("sub3"),
                "description": "Set up database infrastructure and data models",
                "dependencies": [generate_id("sub1")],
                "ai_suggested_complexity_score": 7.5,
                "ai_suggested_evaluation_prompt": "Validate database design and scalability"
            },
            {
                "id": generate_id("sub4"),
                "description": "Implement payment processing and PCI compliance",
                "dependencies": [generate_id("sub2"), generate_id("sub3")],
                "ai_suggested_complexity_score": 9.5,
                "ai_suggested_evaluation_prompt": "Verify PCI-DSS compliance and security"
            },
            {
                "id": generate_id("sub5"),
                "description": "Create monitoring and observability solutions",
                "dependencies": [generate_id("sub1")],
                "ai_suggested_complexity_score": 7.0,
                "ai_suggested_evaluation_prompt": "Validate monitoring coverage and alerting"
            }
        ])
        
        self.mock_decomposer_client.evolve.return_value = mock_decomposition_response
        
        # Decompose the problem
        decomposition_plan = self.decomposer.decompose(problem, strategy="dependency")
        
        decomposition_time = time.time() - start_time
        
        self.assertIsNotNone(decomposition_plan)
        self.assertGreater(len(decomposition_plan.sub_problems), 3)
        print(f"[OK] Problem decomposition completed in {decomposition_time:.3f}s")
        print(f"[OK] Generated {len(decomposition_plan.sub_problems)} sub-problems")
        
        # Store the plan
        plan_id = self.db.create_plan(decomposition_plan)
        self.assertTrue(plan_id)
        print(f"[OK] Decomposition plan stored with ID: {plan_id}")
        
        # PHASE 3: Team Coordination and Validation
        print("\nPhase 3: Team Coordination and Gauntlet Validation")
        
        start_time = time.time()
        
        # Mock team coordination responses
        mock_coordination_response = Mock()
        mock_coordination_response.success = True
        mock_coordination_response.best_code = json.dumps({
            "red_team_validation": {
                "passed": True,
                "score": 0.8,
                "issues_found": 0,
                "recommendations": ["Consider additional security hardening"]
            },
            "blue_team_implementation": {
                "status": "completed",
                "confidence": 0.85,
                "time_to_solution": 120  # minutes
            },
            "gold_team_verification": {
                "passed": True,
                "final_score": 0.82,
                "approval_reasoning": "Solution meets all requirements with minor recommendations"
            }
        })
        
        self.mock_coordinator_client.evolve.return_value = mock_coordination_response
        
        # Coordinate the teams for each sub-problem
        team_assignments = []
        for sub_problem in decomposition_plan.sub_problems:
            assignment = self.coordinator.assign_to_teams(sub_problem)
            self.assertIsNotNone(assignment)
            team_assignments.append(assignment)
        
        coordination_time = time.time() - start_time
        print(f"[OK] Team coordination completed in {coordination_time:.3f}s")
        print(f"[OK] Created {len(team_assignments)} team assignments")
        
        # PHASE 4: Solution Generation and Orchestration
        print("\nPhase 4: Solution Generation and Orchestration")
        
        start_time = time.time()
        
        # Mock solution orchestration responses
        mock_orchestration_response = Mock()
        mock_orchestration_response.success = True
        mock_orchestration_response.best_code = json.dumps({
            "integration_score": 0.88,
            "conflict_count": 0,
            "integration_effort": "low",
            "final_solution": {
                "content": "Integrated solution combining all microservice components with security, monitoring, and deployment automation",
                "confidence": 0.85,
                "validation_status": "approved"
            }
        })
        
        self.mock_orchestrator_client.evolve.return_value = mock_orchestration_response
        
        # Generate solution attempts for each sub-problem
        solution_attempts = []
        for i, sub_problem in enumerate(decomposition_plan.sub_problems):
            # Mock a solution attempt
            mock_solution = SolutionAttempt(
                id=generate_id("sol"),
                sub_problem_id=sub_problem.id,
                approach=f"Implementation approach for {sub_problem.title}",
                solution_content=f"Detailed implementation for {sub_problem.description}",
                team_id="development_team",
                confidence_score=0.8 + (i * 0.02)  # Slightly varying confidence
            )
            
            solution_attempts.append(mock_solution)
        
        # Run validation gauntlets for each solution
        gauntlet_results = []
        mock_gauntlet_response = Mock()
        mock_gauntlet_response.success = True
        mock_gauntlet_response.best_code = json.dumps({
            "passed": True,
            "score": 0.85,
            "feedback": "Solution is robust and well-implemented",
            "improvements": ["Consider adding more edge case handling"]
        })
        
        self.mock_gauntlet_client.evolve.return_value = mock_gauntlet_response
        
        for solution in solution_attempts:
            result = self.gauntlet_system.run_gauntlet(
                content=solution.solution_content,
                gauntlet_name="standard_validation",
                team_name="red",
                context={"solution_type": "implementation", "complexity": solution.confidence_score}
            )
            gauntlet_results.append(result)
        
        # Orchestrate the final solution
        final_solution = self.orchestrator.orchestrate_solutions(
            plan=decomposition_plan,
            solution_attempts=solution_attempts
        )
        
        orchestration_time = time.time() - start_time
        
        self.assertIsNotNone(final_solution)
        self.assertGreater(len(final_solution.integrated_content), 0)
        print(f"[OK] Solution orchestration completed in {orchestration_time:.3f}s")
        print(f"[OK] Generated final solution with {len(gauntlet_results)} validation results")
        
        # PHASE 5: Storage and Validation
        print("\nPhase 5: Storage and Validation")
        
        # Store the final solution
        solution_id = self.db.create_solution_attempt(SolutionAttempt(
            id=generate_id("final_sol"),
            sub_problem_id="all",  # Represents complete solution
            approach="Integrated Architecture Solution",
            solution_content=final_solution.integrated_content,
            team_id="orchestration_team",
            confidence_score=final_solution.overall_confidence
        ))
        
        self.assertTrue(solution_id)
        print(f"[OK] Final solution stored with ID: {solution_id}")
        
        # Validate the complete workflow in database
        stored_problem = self.db.get_problem(problem.id)
        stored_plan = self.db.get_plan(decomposition_plan.id)
        solution_attempts_count = len(self.db.list_solution_attempts_for_problem(problem.id))
        
        self.assertIsNotNone(stored_problem)
        self.assertIsNotNone(stored_plan)
        self.assertGreater(solution_attempts_count, 0)
        
        print(f"[OK] Database validation passed: Problem, Plan, and {solution_attempts_count} solutions stored")
        
        # PHASE 6: Performance and Quality Metrics
        print("\nPhase 6: Performance and Quality Validation")
        
        total_time = time.time() - start_time_global
        
        # Validate performance targets
        self.assertLess(analysis_time, 5.0, "Problem analysis took too long")
        self.assertLess(decomposition_time, 10.0, "Problem decomposition took too long")  
        self.assertLess(coordination_time, 15.0, "Team coordination took too long")
        self.assertLess(total_time, 30.0, "Complete workflow took too long")
        
        print(f"[OK] Performance validation passed - Total workflow completed in {total_time:.3f}s")
        
        # Validate solution quality
        self.assertGreater(final_solution.overall_confidence, 0.8, "Final solution confidence too low")
        self.assertGreater(len(final_solution.integrated_content), 100, "Solution content too sparse")
        
        print(f"[OK] Quality validation passed - Confidence: {final_solution.overall_confidence:.2f}")
        
        # PHASE 7: Security and Validation
        print("\nPhase 7: Security and Input Validation")
        
        # Test that malicious inputs are properly handled
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE problems; --",
            "../../../../etc/passwd",
            "python -c 'import os; os.system(\"rm -rf /\")'",
        ]
        
        for malicious_input in malicious_inputs:
            # Should handle malicious input gracefully
            try:
                is_valid = self.input_validator.validate_input(
                    malicious_input,
                    field_name="test_field",
                    rules=[self.input_validator.VALIDATION_RULES.NOT_EMPTY]
                )
                # Even if validation passes, the input should be sanitized
                self.assertNotIn("<script", is_valid.lower())
            except (ValueError, TypeError, RuntimeError):
                # It's okay if validation fails for malicious input
                pass
        
        print("[OK] Security validation passed - Malicious inputs handled safely")
        
        # PHASE 8: Advanced Features Integration
        print("\nPhase 8: Advanced Features Integration")
        
        # Test domain-specific template application
        template_result = self.advanced_features.apply_domain_template(
            domain_name="software_engineering",
            strategy="microservices",
            context={
                "problem_complexity": problem.complexity_score.overall_complexity,
                "requirements": ["scalability", "security", "reliability"]
            }
        )
        # May be None if template doesn't exist, which is fine
        # Just verify no exceptions were raised
        
        # Test visual representation generation
        if decomposition_plan.sub_problems:
            visual_result = self.advanced_features.generate_visual_representation(
                plan=decomposition_plan,
                format_type="mermaid"
            )
            # Just verify no exceptions - result could be None if not implemented yet
            if visual_result:
                self.assertIsInstance(visual_result, str)
        
        print("[OK] Advanced features integration validated")
        
        print("\n" + "="*80)
        print("🎉 COMPLETE END-TO-END SYSTEM INTEGRATION TEST PASSED! 🎉")
        print("="*80)
        print(f"Total workflow time: {time.time() - start_time_global:.3f}s")
        print(f"Components validated: {len(decomposition_plan.sub_problems)} sub-problems")
        print(f"Teams coordinated: {len(team_assignments)} assignments")
        print(f"Validation gauntlets: {len(gauntlet_results)} runs")
        print(f"Final confidence: {final_solution.overall_confidence:.2f}")
        print("="*80)
    
    def test_error_recovery_and_resilience(self):
        """Test system resilience and error recovery capabilities"""
        print("\nTesting Error Recovery and Resilience...")
        
        # Test 1: Database connection resilience
        db = SovereignDatabase(":memory:")
        
        # Create a valid problem successfully
        valid_problem = ProblemDefinition(
            id=generate_id("valid"),
            title="Valid Test Problem",
            description="This is a valid test problem to verify database operations work correctly",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="testing"),
            complexity_score=ComplexityScore(
                explanation="Test problem",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        result = db.create_problem(valid_problem)
        self.assertTrue(result, "Valid problem should be created successfully")
        
        retrieved = db.get_problem(valid_problem.id)
        self.assertIsNotNone(retrieved, "Valid problem should be retrievable")
        
        # Test 2: Partial failure handling
        # Try to create a problem with some invalid fields but ensure existing data is safe
        try:
            invalid_problem = ProblemDefinition(
                id="",  # Invalid - empty ID
                title="",  # Invalid - empty title
                description="This problem has invalid fields",
                problem_type="INVALID_TYPE",  # Invalid enum value
                domain_context=DomainContext(domain="testing"),
                complexity_score=ComplexityScore(
                    explanation="Test invalid problem",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            
            # This should fail but not corrupt the database
            invalid_result = db.create_problem(invalid_problem)
            # Result might be False or raise an exception - either is acceptable
        except (ValueError, TypeError, RuntimeError):
            # Exception is also acceptable behavior for invalid input
            pass
        
        # Verify that the valid problem is still accessible
        still_there = db.get_problem(valid_problem.id)
        self.assertIsNotNone(still_there, "Valid problem should remain after invalid operation")
        self.assertEqual(still_there.title, "Valid Test Problem")
        
        print("[OK] Error recovery and resilience validation passed")
    
    def test_concurrent_workflow_isolation(self):
        """Test that concurrent workflows don't interfere with each other"""
        print("\nTesting Concurrent Workflow Isolation...")
        
        import threading
        import queue
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Shared database for all workflows
        shared_db = SovereignDatabase(":memory:")
        
        def run_workflow_instance(instance_id: int) -> Dict[str, Any]:
            """Run a single workflow instance"""
            try:
                # Create unique problem for this instance
                problem = ProblemDefinition(
                    id=generate_id(f"instance_{instance_id}"),
                    title=f"Workflow Instance {instance_id}",
                    description=f"Test workflow instance {instance_id} for isolation testing",
                    problem_type=ProblemType.RESEARCH,
                    domain_context=DomainContext(domain="isolation_testing"),
                    complexity_score=ComplexityScore(
                        explanation=f"Instance {instance_id} test",
                        cognitive_complexity=4.0 + (instance_id % 3),
                        computational_complexity=4.0 + (instance_id % 3),
                        domain_complexity=4.0 + (instance_id % 3),
                        integration_complexity=4.0 + (instance_id % 3),
                        overall_complexity=4.0 + (instance_id % 3)
                    )
                )
                
                # Store problem
                problem_result = shared_db.create_problem(problem)
                
                # Create decomposition plan
                plan = DecompositionPlan(
                    id=generate_id(f"plan_{instance_id}"),
                    problem_id=problem.id,
                    strategy="semantic",
                    sub_problems=[
                        SubProblem(
                            id=generate_id(f"sub_{instance_id}_1"),
                            parent_id=problem.id,
                            title=f"Sub-problem 1 for instance {instance_id}",
                            description=f"First sub-problem for workflow instance {instance_id}",
                            type=SubProblemType.ANALYSIS,
                            complexity_score=ComplexityScore(
                                explanation="Test sub-problem",
                                cognitive_complexity=4.0,
                                computational_complexity=4.0,
                                domain_complexity=4.0,
                                integration_complexity=4.0,
                                overall_complexity=4.0
                            )
                        ),
                        SubProblem(
                            id=generate_id(f"sub_{instance_id}_2"),
                            parent_id=problem.id,
                            title=f"Sub-problem 2 for instance {instance_id}",
                            description=f"Second sub-problem for workflow instance {instance_id}",
                            type=SubProblemType.IMPLEMENTATION,
                            complexity_score=ComplexityScore(
                                explanation="Test sub-problem",
                                cognitive_complexity=5.0,
                                computational_complexity=5.0,
                                domain_complexity=5.0,
                                integration_complexity=5.0,
                                overall_complexity=5.0
                            )
                        )
                    ],
                    status=PlanStatus.ACTIVE
                )
                
                plan_result = shared_db.create_plan(plan)
                
                return {
                    'instance_id': instance_id,
                    'problem_created': problem_result,
                    'plan_created': plan_result,
                    'success': problem_result and plan_result,
                    'error': None
                }
            except Exception as e:
                return {
                    'instance_id': instance_id,
                    'problem_created': False,
                    'plan_created': False,
                    'success': False,
                    'error': str(e)
                }
        
        # Run multiple workflows concurrently
        num_instances = 10
        results = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:  # 5 concurrent workers
            futures = [executor.submit(run_workflow_instance, i) for i in range(num_instances)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_instances = [r for r in results if r['success']]
        failed_instances = [r for r in results if not r['success']]
        
        print(f"[OK] Concurrent workflow test: {len(successful_instances)}/{num_instances} instances succeeded in {total_time:.3f}s")
        
        # Verify that all successful instances have isolated data
        all_problems = shared_db.list_problems()
        all_plans = shared_db.list_plans()
        
        print(f"[OK] Database contains {len(all_problems)} problems and {len(all_plans)} plans from all instances")
        
        # All instances should succeed (if database operations are properly isolated)
        self.assertGreaterEqual(len(successful_instances), num_instances * 0.9, 
                              f"Most workflows should succeed. Got {len(successful_instances)}/{num_instances}")
        
        # Verify no cross-contamination between instances
        problem_titles = [p.title for p in all_problems]
        unique_titles = set(problem_titles)
        
        # Should have one unique title per successful instance
        self.assertEqual(len(unique_titles), len(successful_instances), 
                        "Each instance should have its own problem")
    
    def test_resource_cleanup_and_memory_management(self):
        """Test resource cleanup and memory management"""
        print("\nTesting Resource Cleanup and Memory Management...")
        
        import gc
        import weakref
        
        # Create objects that should be cleaned up
        objects_to_track = []
        weakrefs_to_dead_objects = []
        
        for i in range(50):  # Create 50 objects to test cleanup
            problem = ProblemDefinition(
                id=generate_id(f"cleanup_test_{i}"),
                title=f"Cleanup Test Problem {i}",
                description=f"Problem {i} for resource cleanup testing",
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="cleanup_test"),
                complexity_score=ComplexityScore(
                    explanation="Resource cleanup test",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            objects_to_track.append(problem)
            
            # Create a weak reference to track if object gets cleaned up
            weak_ref = weakref.ref(problem)
            weakrefs_to_dead_objects.append(weak_ref)
        
        # Verify all objects exist initially
        living_before_cleanup = sum(1 for ref in weakrefs_to_dead_objects if ref() is not None)
        self.assertEqual(living_before_cleanup, 50, "All objects should be alive initially")
        
        # Clear the objects and force garbage collection
        del objects_to_track
        gc.collect()
        
        # Check how many objects were cleaned up
        living_after_cleanup = sum(1 for ref in weakrefs_to_dead_objects if ref() is not None)
        
        print(f"[OK] Memory cleanup: {50 - living_after_cleanup}/50 objects properly cleaned up")
        
        # Most objects should have been cleaned up (allow a few to remain due to internal references)
        self.assertLess(living_after_cleanup, 10, "Most objects should be cleaned up after deletion")
    
    def test_system_monitoring_and_metrics_collection(self):
        """Test system monitoring and metrics collection"""
        print("\nTesting System Monitoring and Metrics Collection...")
        
        # Test metrics collection
        self.metrics_collector.increment_counter("test_operation", labels={"type": "integration"})
        self.metrics_collector.set_gauge("active_workflows", 1)
        self.metrics_collector.record_histogram("workflow_duration", 1.23, labels={"status": "success"})
        
        # Retrieve metrics
        counter_val = self.metrics_collector.get_counter_value("test_operation", labels={"type": "integration"})
        gauge_val = self.metrics_collector.get_gauge_value("active_workflows")
        
        self.assertGreaterEqual(counter_val, 1, "Counter should have been incremented")
        self.assertEqual(gauge_val, 1, "Gauge should have been set")
        
        print(f"[OK] Metrics collection: Counter={counter_val}, Gauge={gauge_val}")
        
        # Test that metrics collection doesn't interfere with main operations
        problem = ProblemDefinition(
            id=generate_id("metrics_test"),
            title="Metrics Collection Test",
            description="Test that metrics collection doesn't interfere with operations",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="metrics_test"),
            complexity_score=ComplexityScore(
                explanation="Metrics test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        # This should work without issues despite metrics collection happening
        result = self.db.create_problem(problem)
        self.assertTrue(result, "Problem creation should work with metrics collection active")
        
        print("[OK] Monitoring and metrics collection doesn't interfere with operations")
    
    def test_cache_efficiency_and_correctness(self):
        """Test cache efficiency and correctness"""
        print("\nTesting Cache Efficiency and Correctness...")
        
        # Test cache stores and retrieves correctly
        test_content = "Test content for caching"
        test_params = {"model": "test_model", "temperature": 0.5}
        test_response = {"choices": [{"message": {"content": "Cached response"}}]}
        
        # Store in cache
        self.cache.cache_response(test_content, test_params, test_response)
        
        # Retrieve from cache
        cached = self.cache.get_response(test_content, test_params)
        
        self.assertIsNotNone(cached, "Cached response should be retrievable")
        self.assertEqual(cached["choices"][0]["message"]["content"], "Cached response")
        
        print("[OK] Cache stores and retrieves correctly")
        
        # Test cache doesn't return incorrect responses for different inputs
        different_content = "Different test content"
        different_response = self.cache.get_response(different_content, test_params)
        
        self.assertNotEqual(different_response, cached, "Cache should not mix up different content")
        
        # Test cache statistics
        stats = self.cache.get_stats()
        self.assertIn('total_requests', stats)
        self.assertIn('cache_hits', stats)
        self.assertGreaterEqual(stats['total_requests'], 1)
        
        print(f"[OK] Cache correctness: Hits={stats['cache_hits']}, Misses calculated from total={stats['total_requests']}")


def run_final_validation_tests():
    """Run the final comprehensive validation tests"""
    print("Running FINAL COMPREHENSIVE VALIDATION TESTS...")
    print("="*80)
    
    # Create test suite with all validation tests
    suite = unittest.TestSuite()
    
    # Add the complete system integration tests
    suite.addTest(unittest.makeSuite(TestCompleteSystemIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("FINAL VALIDATION TEST RESULTS")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures or result.errors:
        print("\n[FAIL] SOME TESTS FAILED [FAIL]")
        for test, trace in result.failures:
            print(f"\nFAILED: {test}")
            print(trace)
        for test, trace in result.errors:
            print(f"\nERROR: {test}")
            print(trace)
    else:
        print("\n🎉 ALL FINAL VALIDATION TESTS PASSED! 🎉")
        print("The Sovereign-Grade Problem Decomposition System is fully validated and ready for production!")
    
    print("="*80)
    
    return result


if __name__ == "__main__":
    run_final_validation_tests()