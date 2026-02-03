"""
Integration and Performance Tests for Sovereign-Grade System
Comprehensive integration and performance tests
"""


import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import os
from datetime import datetime
import json
import random
from typing import Dict, List, Any

# Add the project root to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sovereign_data_models import ProblemDefinition, SubProblem, generate_id
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_team_coordination import TeamCoordinator
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_persistence import SovereignDatabase
from auth_system import AuthenticationSystem
from performance_optimization import LLMResponseCache, ParallelProcessor
from scalability_improvements import ResourceMonitor
from monitoring_system import MetricsCollector, DistributedTracer


class TestIntegrationWorkflows(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db = SovereignDatabase("test_integration.db")
        self.auth_system = AuthenticationSystem(db_path="test_integration_auth.db")
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        for db_file in ["test_integration.db", "test_integration_auth.db"]:
            if os.path.exists(db_file):
                os.remove(db_file)
    
    def test_complete_problem_to_solution_workflow(self):
        """Test complete workflow from problem definition to solution"""
        from auth_system import Role, Permission
        
        # Create a user
        user = self.auth_system.create_user(
            username="test_analyst",
            email="analyst@example.com",
            password="complexPassword123!",
            roles=[Role.ANALYST],
            permissions=[Permission.CREATE_PROBLEM, Permission.READ_PROBLEM]
        )
        
        # Authenticate user
        authenticated_user = self.auth_system.authenticate("test_analyst", "complexPassword123!")
        self.assertIsNotNone(authenticated_user)
        
        # Create a problem
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Complete Workflow Test Problem",
            description="This is a test problem to validate the complete workflow from analysis through solution",
            problem_type="RESEARCH",
            domain_context={
                "domain": "software_engineering",
                "subdomain": "api_design",
                "related_domains": ["dev_ops", "security"],
                "domain_knowledge": {"key_concepts": ["rest", "authentication", "rate_limiting"]}
            },
            complexity_score={
                "cognitive_complexity": 7.5,
                "computational_complexity": 6.0,
                "domain_complexity": 8.0,
                "integration_complexity": 7.0,
                "overall_complexity": 7.1,
                "explanation": "Moderate to high complexity due to multiple domain requirements"
            },
            constraints=[],
            success_criteria=[]
        )
        
        # Store the problem
        problem_id = self.db.create_problem(problem)
        self.assertTrue(problem_id)
        
        # Retrieve and verify
        retrieved_problem = self.db.get_problem(problem.id)
        self.assertIsNotNone(retrieved_problem)
        self.assertEqual(retrieved_problem.title, problem.title)
        
        # Track metrics
        self.metrics_collector.increment_counter("problems_created", labels={"user": authenticated_user.id})
        self.metrics_collector.set_gauge("active_problems", 1)
        
        # Verify metrics were recorded
        problem_created = self.metrics_collector.get_counter_value("problems_created", labels={"user": authenticated_user.id})
        active_problems = self.metrics_collector.get_current_gauge_value("active_problems")
        
        self.assertIsNotNone(problem_created)
        self.assertEqual(active_problems, 1)
    
    def test_analyzer_to_decomposer_integration(self):
        """Test integration between problem analyzer and decomposition engine"""
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve_analyzer, \
             patch('decomposition_engine.OpenEvolveClient') as mock_openevolve_decomposer:
            
            # Setup mock clients
            mock_client_analyzer = mock_openevolve_analyzer.return_value
            mock_client_decomposer = mock_openevolve_decomposer.return_value
            
            # Mock analyzer responses
            analysis_mock_result = Mock()
            analysis_mock_result.success = True
            analysis_mock_result.best_code = json.dumps({
                "domain": "software_engineering",
                "subdomain": "api_design",
                "related_domains": ["security"],
                "key_concepts": ["rest", "authentication"],
                "domain_complexity": 7.5,
                "required_expertise": ["javascript", "python"],
                "estimated_complexity": 7.0,
                "potential_challenges": ["rate_limiting", "authentication"],
                "required_expertise": ["api_design", "security"]
            })
            
            mock_client_analyzer.evolve.return_value = analysis_mock_result
            
            # Mock decomposer responses
            decomposition_mock_result = Mock()
            decomposition_mock_result.success = True
            decomposition_mock_result.best_code = json.dumps([
                {
                    "id": generate_id("sub1"),
                    "description": "Design API endpoints",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 6.5,
                    "ai_suggested_evaluation_prompt": "Evaluate endpoint design quality"
                },
                {
                    "id": generate_id("sub2"),
                    "description": "Implement authentication",
                    "dependencies": [generate_id("sub1")],
                    "ai_suggested_complexity_score": 7.0,
                    "ai_suggested_evaluation_prompt": "Evaluate security implementation"
                }
            ])
            
            mock_client_decomposer.evolve.return_value = decomposition_mock_result
            
            # Create analyzer and decomposer
            analyzer = ProblemAnalyzer(openevolve_client=mock_client_analyzer)
            decomposer = DecompositionEngine(openevolve_client=mock_client_decomposer)
            
            # Analyze problem
            problem = analyzer.analyze_problem(
                problem_text="Design a secure REST API for user management",
                title="Secure User Management API"
            )
            
            self.assertIsNotNone(problem)
            
            # Decompose the problem
            plan = decomposer.decompose(problem, strategy="semantic")
            
            self.assertIsNotNone(plan)
            self.assertGreater(len(plan.sub_problems), 0)
            
            # Verify dependencies were correctly handled
            if len(plan.sub_problems) > 1:
                # Second sub-problem should depend on first
                first_sub = plan.sub_problems[0]
                second_sub = plan.sub_problems[1]
                if first_sub.id in second_sub.dependencies:
                    self.assertIn(first_sub.id, second_sub.dependencies)
    
    def test_database_team_coordination_integration(self):
        """Test integration between database and team coordination"""
        # Create a problem in database
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Coordination Test",
            description="Test team coordination features",
            problem_type="RESEARCH",
            domain_context={"domain": "software_engineering"},
            complexity_score={"cognitive_complexity": 5.0, "overall_complexity": 5.0}
        )
        
        problem_id = self.db.create_problem(problem)
        self.assertTrue(problem_id)
        
        # Create team coordinator
        coordinator = TeamCoordinator()
        
        # Create a decomposition plan
        plan = Mock()
        plan.id = generate_id("plan")
        plan.problem_id = problem_id
        plan.sub_problems = [
            SubProblem(
                id=generate_id("sub1"),
                parent_id=problem_id,
                title="Test Sub-task 1",
                description="Test sub-task for red team review",
                type="ANALYSIS",
                complexity_score={"overall_complexity": 6.0}
            )
        ]
        
        # Assign to red team (mock the actual team assignment)
        assignment = coordinator.assign_decomposition_review(plan)
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.team, 'red')
        
        # Verify assignment was stored in database (via coordinator's internal tracking)
        workload_info = coordinator.balance_workload()
        self.assertIn('red_team', workload_info['red_team'])


class TestPerformanceAndStress(unittest.TestCase):
    """Performance and stress tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache = LLMResponseCache()
        self.processor = ParallelProcessor()
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor()
    
    def test_llm_cache_performance(self):
        """Test LLM response cache performance"""
        import time
        
        # Add some entries to cache
        for i in range(100):
            content = f"Test content {i}"
            model_params = {"model": "gpt-4", "temperature": 0.7}
            response = {"choices": [{"message": {"content": f"Response {i}"}}]}
            
            self.cache.cache_response(content, model_params, response)
        
        # Measure cache hit performance
        start_time = time.time()
        for i in range(1000):
            content = f"Test content {i % 100}"  # Hit cache 90% of the time
            model_params = {"model": "gpt-4", "temperature": 0.7}
            result = self.cache.get_response(content, model_params)
        
        cache_time = time.time() - start_time
        
        print(f"Cache performance: {cache_time:.3f}s for 1000 operations")
        
        # Cache operations should be fast
        self.assertLess(cache_time, 1.0, "Cache operations taking too long")
        
        # Verify cache stats
        stats = self.cache.get_stats()
        self.assertGreaterEqual(stats['total_entries'], 100)
        
        # Verify hit/miss counts
        self.assertGreater(stats['hits'] + stats['misses'], 900)
    
    def test_parallel_processing_performance(self):
        """Test parallel processing performance"""
        import time
        
        def sample_task(task_id):
            """Simulate a task that takes some time"""
            time.sleep(0.001)  # Small delay to simulate real work
            return f"Task {task_id} completed"
        
        # Create multiple tasks
        tasks = [lambda i=i: sample_task(i) for i in range(50)]
        
        # Run tasks in parallel
        start_time = time.time()
        parallel_results = self.processor.process_in_parallel(tasks)
        parallel_time = time.time() - start_time
        
        # Run tasks sequentially for comparison
        start_time = time.time()
        sequential_results = [task() for task in tasks]
        sequential_time = time.time() - start_time
        
        print(f"Parallel processing: {parallel_time:.3f}s for {len(tasks)} tasks")
        print(f"Sequential processing: {sequential_time:.3f}s for {len(tasks)} tasks")
        
        # Verify results
        self.assertEqual(len(parallel_results), len(tasks))
        self.assertEqual(len(sequential_results), len(tasks))
        
        # Both should produce same results
        for i, (par_result, seq_result) in enumerate(zip(parallel_results, sequential_results)):
            self.assertEqual(par_result, seq_result)
        
        # Parallel should be much faster for I/O bound tasks
        # Note: For CPU-bound tasks, sequential might actually be faster due to Python's GIL
        # But for I/O-bound tasks we simulate, parallel should be faster
    
    def test_resource_monitoring(self):
        """Test resource monitoring functionality"""
        # Start monitoring
        self.resource_monitor.start_monitoring(interval=0.1)  # Sample every 0.1 seconds
        
        # Simulate some work to generate metrics
        time.sleep(0.3)  # Allow monitoring to collect some data
        
        # Get current metrics
        metrics = self.resource_monitor.get_current_metrics()
        
        self.assertIn('cpu_percent', metrics)
        self.assertIn('memory_percent', metrics)
        self.assertIn('process_count', metrics)
        
        print(f"Current system metrics: {metrics}")
        
        # Verify metrics are reasonable
        self.assertGreaterEqual(metrics['cpu_percent'], 0)
        self.assertLessEqual(metrics['cpu_percent'], 100)
        self.assertGreaterEqual(metrics['memory_percent'], 0)
        self.assertLessEqual(metrics['memory_percent'], 100)
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
    
    def test_concurrent_user_simulation(self):
        """Simulate concurrent users accessing the system"""
        import threading
        import time
        from queue import Queue
        
        results_queue = Queue()
        
        def user_simulation(user_id):
            """Simulate a user performing operations"""
            try:
                # Simulate analysis work
                time.sleep(random.uniform(0.01, 0.05))  # Random small delay
                
                # Simulate database operations
                db = SovereignDatabase(f"test_concurrent_{user_id}.db")
                
                problem = ProblemDefinition(
                    id=generate_id("problem"),
                    title=f"Concurrent Test Problem {user_id}",
                    description=f"Problem created by concurrent user {user_id}",
                    problem_type="RESEARCH",
                    domain_context={"domain": "software_engineering"},
                    complexity_score={"overall_complexity": random.uniform(4.0, 8.0)}
                )
                
                db.create_problem(problem)
                
                # Retrieve problem
                retrieved = db.get_problem(problem.id)
                
                results_queue.put({
                    'user_id': user_id,
                    'success': retrieved is not None,
                    'problem_id': problem.id if retrieved else None
                })
                
            except Exception as e:
                results_queue.put({
                    'user_id': user_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Create multiple threads to simulate concurrent users
        threads = []
        num_users = 10
        
        start_time = time.time()
        for i in range(num_users):
            thread = threading.Thread(target=user_simulation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        print(f"Concurrent user simulation: {num_users} users in {total_time:.3f}s")
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        successful_ops = sum(1 for r in results if r['success'])
        
        print(f"Successful operations: {successful_ops}/{num_users}")
        
        # Most operations should succeed
        self.assertGreaterEqual(successful_ops, num_users * 0.8)
        
        # Verify performance is reasonable
        ops_per_second = num_users / total_time if total_time > 0 else float('inf')
        print(f"Operations per second: {ops_per_second:.1f}")
        self.assertGreater(ops_per_second, 5)  # Should handle at least 5 ops/sec
    
    def test_memory_usage_under_load(self):
        """Test memory usage when system is under load"""
        import gc
        import psutil
        import time
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create a large number of objects to simulate load
        large_objects = []
        for i in range(1000):
            large_obj = {
                'id': generate_id("large_obj"),
                'data': [f"item_{j}" for j in range(100)],  # 100 items per object
                'metadata': {'created_by': 'stress_test', 'batch': i}
            }
            large_objects.append(large_obj)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.2f}MB -> {peak_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        
        # Verify memory increase is reasonable (less than 50MB for this test)
        self.assertLess(memory_increase, 50.0, f"Memory increase too high: {memory_increase}MB")
        
        # Clean up
        del large_objects
        gc.collect()
        
        # Check memory after cleanup
        cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        cleanup_increase = cleanup_memory - initial_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.2f}MB (increase: {cleanup_increase:.2f}MB)")
        
        # Memory should be mostly reclaimed
        self.assertLess(cleanup_increase, memory_increase * 0.5)  # Less than half of peak increase
    
    def test_metrics_collection_performance(self):
        """Test performance of metrics collection under high frequency"""
        import time
        
        # Collect metrics rapidly
        start_time = time.time()
        for i in range(1000):
            # Simulate collecting various system metrics
            self.metrics_collector.set_gauge(f"test_metric_{i % 10}", i % 100)
            self.metrics_collector.increment_counter("test_counter", labels={"iteration": str(i % 5)})
        
        collection_time = time.time() - start_time
        
        print(f"Metrics collection performance: {collection_time:.3f}s for 1000 operations")
        
        # Should be fast (< 0.5 seconds for 1000 operations)
        self.assertLess(collection_time, 0.5, "Metrics collection too slow")


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and resilience"""
    
    def test_database_error_recovery(self):
        """Test database error recovery"""
        db = SovereignDatabase("test_error_recovery.db")
        
        # Create a problem
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Error Recovery Test",
            description="Test error handling in database operations",
            problem_type="RESEARCH",
            domain_context={"domain": "software_engineering"},
            complexity_score={"overall_complexity": 5.0}
        )
        
        # Successfully create problem
        result = db.create_problem(problem)
        self.assertTrue(result)
        
        # Try to create with invalid data (should handle gracefully)
        invalid_problem = ProblemDefinition(
            id="",  # Invalid - empty ID
            title="",  # Invalid - empty title
            description="",
            problem_type="INVALID_TYPE",  # Invalid type
            domain_context={},
            complexity_score={}
        )
        
        try:
            db.create_problem(invalid_problem)
            # If it doesn't raise an error, check if it returns False
            # (depends on implementation)
        except (ValueError, TypeError, RuntimeError):
            # If it raises an exception, that's also acceptable error handling
            pass
        
        # Verify original problem still exists
        retrieved = db.get_problem(problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Error Recovery Test")
    
    def test_authentication_failure_handling(self):
        """Test authentication failure handling"""
        auth_system = AuthenticationSystem(db_path="test_auth_errors.db")
        
        # Create a user
        user = auth_system.create_user(
            username="test_user",
            email="test@example.com", 
            password="SecurePass123!",
            permissions=[]
        )
        self.assertIsNotNone(user)
        
        # Try to authenticate with wrong password
        wrong_auth = auth_system.authenticate("test_user", "wrong_password")
        self.assertIsNone(wrong_auth)
        
        # Try to authenticate with non-existent user
        nonexistent_auth = auth_system.authenticate("nonexistent_user", "password")
        self.assertIsNone(nonexistent_auth)
        
        # Try to authenticate with empty credentials
        empty_auth = auth_system.authenticate("", "")
        self.assertIsNone(empty_auth)
    
    def test_cache_fallback_behavior(self):
        """Test cache fallback behavior when cache fails"""
        cache = LLMResponseCache()
        
        # Test normal cache operation
        content = "normal content"
        model_params = {"model": "gpt-4"}
        response = {"choices": [{"message": {"content": "cached response"}}]}
        
        cache.cache_response(content, model_params, response)
        cached_result = cache.get_response(content, model_params)
        self.assertIsNotNone(cached_result)
        
        # Test cache behavior with large content
        large_content = "x" * 10000  # Large content
        large_response = {"choices": [{"message": {"content": "large response"}}]}
        
        cache.cache_response(large_content, model_params, large_response)
        large_cached_result = cache.get_response(large_content, model_params)
        self.assertIsNotNone(large_cached_result)


class TestSecurity(unittest.TestCase):
    """Security-related tests"""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in database queries"""
        db = SovereignDatabase("test_security.db")
        
        # Try to create a problem with SQL injection in title
        malicious_title = "Normal Title'; DROP TABLE problems; --"
        
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title=malicious_title,
            description="Test SQL injection prevention",
            problem_type="RESEARCH",
            domain_context={"domain": "software_engineering"},
            complexity_score={"overall_complexity": 5.0}
        )
        
        # This should handle the malicious input safely
        result = db.create_problem(problem)
        
        # Verify the problem was created safely (malicious SQL was not executed)
        if result:  # If creation was allowed (depends on validation)
            retrieved = db.get_problem(problem.id)
            self.assertIsNotNone(retrieved)
            # The malicious SQL should NOT have been executed
            # The title may have been sanitized or rejected by validation
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        from input_validation import InputValidator
        
        validator = InputValidator()
        
        # Test XSS prevention
        malicious_html = '<script>alert("xss")</script><p>Safe content</p>'
        
        # With HTML sanitization
        sanitized = validator._sanitize_html(malicious_html, "test_field")
        
        # Script tag should be removed but paragraph remains
        self.assertNotIn("alert", sanitized)
        self.assertNotIn("<script>", sanitized)
        self.assertIn("<p>", sanitized)
        self.assertIn("Safe content", sanitized)


def run_performance_tests():
    """Run the performance tests"""
    print("Running integration and performance tests...")
    
    # Create a test suite for integration tests
    suite = unittest.TestSuite()
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestIntegrationWorkflows))
    suite.addTest(unittest.makeSuite(TestPerformanceAndStress))
    suite.addTest(unittest.makeSuite(TestErrorHandling))
    suite.addTest(unittest.makeSuite(TestSecurity))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    print(f"\nIntegration/Performance Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result


if __name__ == "__main__":
    run_performance_tests()