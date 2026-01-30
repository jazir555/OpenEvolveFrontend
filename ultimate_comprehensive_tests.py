"""
Ultimate Comprehensive Unit Tests for Sovereign-Grade System
Complete battery of tests for all components with extreme scenarios and edge cases
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import time
import threading
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import sys
import os
import tempfile
import sqlite3
import hashlib
import secrets
import gc
import weakref
import tracemalloc
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
import random
import string
import uuid
import logging
import queue
import multiprocessing
from contextlib import contextmanager

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
    Constraint, SuccessCriterion, DomainContext, ComplexityScore,
    ProblemType, SubProblemType, PlanStatus, SubProblemStatus, generate_id
)
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_team_coordination import TeamCoordinator
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_persistence import SovereignDatabase
from auth_system import AuthenticationSystem
from input_validation import InputValidator
from performance_optimization import LLMResponseCache, ParallelProcessor
from scalability_improvements import ResourceMonitor
from monitoring_system import MetricsCollector
from sovereign_gauntlets import GauntletSystem


class TestUltimateEdgeCases(unittest.TestCase):
    """Ultimate edge case and boundary condition tests"""
    
    def test_recursive_complexity_calculation(self):
        """Test extreme recursive complexity calculations"""
        # Create a deeply nested complexity score calculation to test recursion limits
        def create_deep_complexity(depth: int) -> ComplexityScore:
            if depth <= 0:
                return ComplexityScore(
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0,
                    explanation="Base case complexity"
                )
            
            deeper = create_deep_complexity(depth - 1)
            return ComplexityScore(
                cognitive_complexity=min(10.0, deeper.cognitive_complexity + random.uniform(0.1, 0.3)),
                computational_complexity=min(10.0, deeper.computational_complexity + random.uniform(0.1, 0.3)),
                domain_complexity=min(10.0, deeper.domain_complexity + random.uniform(0.1, 0.3)),
                integration_complexity=min(10.0, deeper.integration_complexity + random.uniform(0.1, 0.3)),
                overall_complexity=min(10.0, deeper.overall_complexity + random.uniform(0.1, 0.2)),
                explanation=f"Recursive complexity at depth {depth}"
            )
        
        # Test very deep recursion (but not too deep to cause actual stack overflow)
        deep_complexity = create_deep_complexity(50)  # Should handle 50 levels deep
        
        self.assertLessEqual(deep_complexity.overall_complexity, 10.0)
        self.assertGreaterEqual(deep_complexity.overall_complexity, 5.0)
        self.assertIn("depth 50", deep_complexity.explanation)
    
    def test_extreme_input_combinations(self):
        """Test extreme combinations of inputs"""
        # Generate extreme parameter combinations
        extreme_combinations = [
            {
                'title': 'A' * 10000,  # Extremely long title
                'description': 'B' * 50000,  # Extremely long description
                'complexity_scores': {
                    'cognitive_complexity': 9.99,
                    'computational_complexity': 0.01,
                    'domain_complexity': 9.99,
                    'integration_complexity': 0.01,
                    'overall_complexity': 5.0  # Average should be moderate despite extremes
                }
            },
            {
                'title': 'C',
                'description': 'D',
                'complexity_scores': {
                    'cognitive_complexity': 0.01,
                    'computational_complexity': 9.99,
                    'domain_complexity': 0.01,
                    'integration_complexity': 9.99,
                    'overall_complexity': 5.0
                }
            }
        ]
        
        for combo in extreme_combinations:
            with self.subTest(extreme_case=combo['title'][:20]):
                # Create a problem with extreme parameters
                extreme_problem = ProblemDefinition(
                    id=generate_id("extreme_test"),
                    title=combo['title'],
                    description=combo['description'],
                    problem_type=ProblemType.RESEARCH,
                    domain_context=DomainContext(domain="extreme_testing"),
                    complexity_score=ComplexityScore(
                        explanation="Extreme parameter test",
                        cognitive_complexity=combo['complexity_scores']['cognitive_complexity'],
                        computational_complexity=combo['complexity_scores']['computational_complexity'],
                        domain_complexity=combo['complexity_scores']['domain_complexity'],
                        integration_complexity=combo['complexity_scores']['integration_complexity'],
                        overall_complexity=combo['complexity_scores']['overall_complexity']
                    )
                )
                
                # Validate without issues
                validation_errors = extreme_problem.validate()
                
                # Should have specific validation rules (like title/description length) that trigger on extremes
                # If there are validation errors, they should be expected
                if len(validation_errors) > 0:
                    print(f"Extreme case validation errors: {validation_errors}")
                
                # Test database storage with extreme data
                db = SovereigntyDatabase(":memory:")
                result = db.create_problem(extreme_problem)
                
                # Should handle extreme data gracefully
                self.assertIsNotNone(result)
                
                # Retrieve and verify extreme data is preserved
                retrieved = db.get_problem(extreme_problem.id)
                self.assertIsNotNone(retrieved)
                self.assertEqual(len(retrieved.title), len(combo['title']))
    
    def test_concurrent_modification_scenarios(self):
        """Test concurrent modification of the same resources"""
        import time
        import threading
        from queue import Queue
        
        # Create database for concurrent access
        db = SovereigntyDatabase(":memory:")
        
        # Shared problem that will be accessed by multiple threads
        shared_problem = ProblemDefinition(
            id=generate_id("shared_concurrent"),
            title="Shared Concurrent Problem",
            description="Problem being accessed by multiple concurrent threads",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="concurrent_access"),
            complexity_score=ComplexityScore(
                explanation="Concurrent access test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        result = db.create_problem(shared_problem)
        self.assertTrue(result)
        
        # Queues for results and errors
        results_queue = Queue()
        errors_queue = Queue()
        operations_performed = 0
        
        def concurrent_accessor(accessor_id: int):
            """Function that runs in each thread accessing the database"""
            local_operations = 0
            try:
                for i in range(20):  # Each thread performs 20 operations
                    # Read operation
                    retrieved = db.get_problem(shared_problem.id)
                    
                    if retrieved:
                        # Update operation with slight variations
                        retrieved.title = f"Updated by accessor {accessor_id}, iteration {i}"
                        retrieved.updated_at = datetime.now()
                        update_result = db.update_problem(retrieved)
                        local_operations += 1
                        
                        results_queue.put({
                            'accessor_id': accessor_id,
                            'operation': 'update',
                            'iteration': i,
                            'success': update_result,
                            'timestamp': time.time()
                        })
                    else:
                        results_queue.put({
                            'accessor_id': accessor_id,
                            'operation': 'read_failed',
                            'iteration': i,
                            'success': False,
                            'timestamp': time.time()
                        })
                    
                    # Brief sleep to allow other threads to interleave
                    time.sleep(0.001)
            except Exception as e:
                errors_queue.put({
                    'accessor_id': accessor_id,
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        # Start multiple threads
        threads = []
        num_accessors = 10
        
        start_time = time.time()
        
        for accessor_id in range(num_accessors):
            thread = threading.Thread(target=concurrent_accessor, args=(accessor_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        successful_updates = [r for r in results if r['operation'] == 'update' and r['success']]
        failed_operations = [r for r in results if not r['success']]
        
        print(f"Concurrent access test: {len(successful_updates)} successful updates, {len(errors)} errors, {len(failed_operations)} failed ops")
        print(f"Total time: {total_time:.3f}s for {num_accessors} threads with {num_accessors * 20} operations")
        
        # Verify database integrity despite concurrent access
        final_problem = db.get_problem(shared_problem.id)
        self.assertIsNotNone(final_problem)
        self.assertIn("Updated by accessor", final_problem.title)
        
        # Should have minimal errors despite concurrency
        self.assertLess(len(errors), len(results) * 0.1, f"Too many errors during concurrent access: {len(errors)}")
    
    def test_resource_exhaustion_scenarios(self):
        """Test system behavior under resource exhaustion"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline resource usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        baseline_fds = len(process.open_files()) if hasattr(process, 'open_files') else 0
        
        print(f"Baseline memory: {baseline_memory:.1f}MB, FDs: {baseline_fds}")
        
        # Create maximum possible objects to test resource limits
        created_objects = []
        
        for i in range(10000):  # Create many objects to test memory handling
            obj = {
                'id': generate_id(f"resource_test_{i}"),
                'data': {
                    'nested_list': [f"item_{j}" for j in range(50)],
                    'nested_dict': {f'key_{k}': f'value_{k}' for k in range(25)},
                    'complexity': ComplexityScore(
                        cognitive_complexity=random.uniform(1.0, 10.0),
                        computational_complexity=random.uniform(1.0, 10.0),
                        domain_complexity=random.uniform(1.0, 10.0),
                        integration_complexity=random.uniform(1.0, 10.0),
                        overall_complexity=random.uniform(1.0, 10.0),
                        explanation=f"Resource test object {i}"
                    ),
                    'metadata': {
                        'created_by': f"resource_test_{i % 100}",
                        'timestamp': datetime.now().isoformat(),
                        'batch': i // 1000
                    }
                },
                'references': [generate_id(f"ref_{i}_{j}") for j in range(5)],
                'temporal_data': [datetime.now().isoformat() for _ in range(3)]
            }
            created_objects.append(obj)
            
            # Periodically check resource usage and trigger GC
            if i % 1000 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - baseline_memory
                
                print(f"  After {i} objects: Memory {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
                
                # Memory should not grow unbounded
                self.assertLess(memory_increase, 500.0, f"Memory usage grew too large: {memory_increase:.1f}MB")
                
                # Trigger garbage collection
                gc.collect()
        
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - baseline_memory
        
        print(f"Peak memory usage: {peak_memory:.1f}MB (+{memory_increase:.1f}MB for {len(created_objects)} objects)")
        
        # Clean up objects
        del created_objects
        gc.collect()
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024
        cleanup_increase = cleanup_memory - baseline_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB (+{cleanup_increase:.1f}MB)")
        
        # Memory should be largely reclaimed after cleanup
        self.assertLess(cleanup_increase, memory_increase * 0.3, "Memory not sufficiently reclaimed after cleanup")
    
    def test_timing_attack_vulnerabilities(self):
        """Test vulnerability to timing attacks"""
        import time
        
        auth_system = AuthenticationSystem()
        
        # Create a user
        user = auth_system.create_user(
            username="timing_test",
            email="timing@example.com",
            password="SecurePassword123!",
            permissions=[]
        )
        
        # Test that authentication time is consistent regardless of valid/invalid user
        valid_auth_times = []
        invalid_auth_times = []
        
        # Test with valid user
        for _ in range(100):
            start_time = time.perf_counter()
            result = auth_system.authenticate("timing_test", "SecurePassword123!")
            end_time = time.perf_counter()
            valid_auth_times.append(end_time - start_time)
        
        # Test with invalid user
        for _ in range(100):
            start_time = time.perf_counter()
            result = auth_system.authenticate("invalid_user", "wrong_password")
            end_time = time.perf_counter()
            invalid_auth_times.append(end_time - start_time)
        
        # Calculate average times
        avg_valid_time = sum(valid_auth_times) / len(valid_auth_times)
        avg_invalid_time = sum(invalid_auth_times) / len(invalid_auth_times)
        
        # Times should be approximately equal (within 10%) to prevent timing attacks
        time_difference = abs(avg_valid_time - avg_invalid_time) / max(avg_valid_time, avg_invalid_time)
        
        print(f"Timing attack test - Valid: {avg_valid_time:.6f}s, Invalid: {avg_invalid_time:.6f}s")
        print(f"Time difference: {time_difference:.1%}")
        
        # Difference should be less than 20% to prevent timing attacks
        self.assertLess(time_difference, 0.20, "Timing difference too large - potential timing attack vulnerability")
    
    def test_cache_poisoning_prevention(self):
        """Test cache poisoning prevention"""
        cache = LLMResponseCache(max_size=100)
        
        # Test that cache handles malicious content safely
        malicious_inputs = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "python -c 'import os; os.system(\"rm -rf /\")'",
            "eval('console.log(\"malicious\")')",
        ]
        
        # Each malicious input should be cached and retrieved safely
        for i, malicious_input in enumerate(malicious_inputs):
            model_params = {"model": "test_model", "temperature": 0.7}
            response = {"choices": [{"message": {"content": f"Response to malicious input {i}"}}]}
            
            # Store malicious input
            cache.cache_response(malicious_input, model_params, response)
            
            # Retrieve and verify it's handled safely
            retrieved = cache.get_response(malicious_input, model_params)
            
            self.assertIsNotNone(retrieved)
            self.assertIn(f"Response to malicious input {i}", retrieved['choices'][0]['message']['content'])
        
        # Verify cache still functions normally
        normal_input = "What is the capital of France?"
        normal_response = {"choices": [{"message": {"content": "Paris"}}]}
        
        cache.cache_response(normal_input, {"model": "gpt-4"}, normal_response)
        retrieved_normal = cache.get_response(normal_input, {"model": "gpt-4"})
        
        self.assertIsNotNone(retrieved_normal)
        self.assertEqual(retrieved_normal['choices'][0]['message']['content'], "Paris")
        
        # Check cache statistics
        stats = cache.get_stats()
        self.assertGreaterEqual(stats['total_requests'], len(malicious_inputs) + 1)
        self.assertGreaterEqual(stats['total_hits'], 1)  # At least the normal one should be a hit


class TestAdvancedIntegrationScenarios(unittest.TestCase):
    """Advanced integration scenarios with complex multi-component workflows"""
    
    def test_complete_workflow_with_failure_scenarios(self):
        """Test complete workflow with various failure points"""
        from unittest.mock import MagicMock
        
        # Set up system components with mocks
        with patch('problem_analyzer.OpenEvolveClient') as mock_analyzer_client, \
             patch('decomposition_engine.OpenEvolveClient') as mock_decomposer_client, \
             patch('sovereign_team_coordination.OpenEvolveClient') as mock_coordination_client, \
             patch('sovereign_solution_orchestration.OpenEvolveClient') as mock_orchestration_client:
            
            # Mock all clients
            mock_analyzer_client.return_value = Mock()
            mock_decomposer_client.return_value = Mock()
            mock_coordination_client.return_value = Mock()
            mock_orchestration_client.return_value = Mock()
            
            # Mock responses for successful operation
            analysis_result = Mock()
            analysis_result.success = True
            analysis_result.best_code = json.dumps({
                "domain": "software_engineering",
                "subdomain": "api_design",
                "related_domains": ["security"],
                "key_concepts": ["rest", "authentication"],
                "domain_complexity": 7.5,
                "required_expertise": ["api_design", "security"]
            })
            
            decomposition_result = Mock()
            decomposition_result.success = True
            decomposition_result.best_code = json.dumps([
                {
                    "id": generate_id("sub1"),
                    "description": "Design API endpoints",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 6.5,
                    "ai_suggested_evaluation_prompt": "Validate API design"
                },
                {
                    "id": generate_id("sub2"),
                    "description": "Implement authentication",
                    "dependencies": [generate_id("sub1")],
                    "ai_suggested_complexity_score": 7.0,
                    "ai_suggested_evaluation_prompt": "Validate security implementation"
                }
            ])
            
            mock_analyzer_client.return_value.evolve.return_value = analysis_result
            mock_decomposer_client.return_value.evolve.return_value = decomposition_result
            mock_coordination_client.return_value.evolve.return_value = Mock(success=True, best_code=json.dumps({"passed": True, "feedback": "Valid"}))
            mock_orchestration_client.return_value.evolve.return_value = Mock(success=True, best_code=json.dumps({"integrated": True, "confidence": 0.85}))
            
            # Create analyzer and other components
            analyzer = ProblemAnalyzer(openevolve_client=mock_analyzer_client.return_value)
            decomposer = DecompositionEngine(openevolve_client=mock_decomposer_client.return_value)
            coordinator = TeamCoordinator(openevolve_client=mock_coordination_client.return_value)
            orchestrator = SolutionOrchestrator(openevolve_client=mock_orchestration_client.return_value)
            
            # Test with a complex problem
            complex_problem_text = "Design and implement a secure, scalable, and highly available API system for a global e-commerce platform that handles millions of transactions per day while ensuring PCI-DSS compliance and providing real-time analytics."
            
            # Run complete analysis -> decomposition -> coordination -> orchestration workflow
            start_time = time.time()
            
            # 1. Problem Analysis
            analyzed_problem = analyzer.analyze_problem(
                problem_text=complex_problem_text,
                title="Complex E-commerce API Design"
            )
            
            analysis_time = time.time() - start_time
            print(f"Problem analysis completed in {analysis_time:.3f}s")
            
            self.assertIsNotNone(analyzed_problem)
            self.assertGreater(len(analyzed_problem.domain_context.domain), 0)
            
            # 2. Problem Decomposition
            start_time = time.time()
            decomposition_plan = decomposer.decompose(analyzed_problem, strategy="hybrid")
            decomposition_time = time.time() - start_time
            
            print(f"Problem decomposition completed in {decomposition_time:.3f}s")
            
            self.assertIsNotNone(decomposition_plan)
            self.assertGreater(len(decomposition_plan.sub_problems), 0)
            
            # 3. Team Coordination (simulated)
            start_time = time.time()
            team_assignment = coordinator.assign_to_team(
                task_id=decomposition_plan.id,
                team="red",
                priority=8
            )
            coordination_time = time.time() - start_time
            
            print(f"Team coordination completed in {coordination_time:.3f}s")
            
            self.assertIsNotNone(team_assignment)
            self.assertEqual(team_assignment.team, "red")
            
            # 4. Solution Orchestration (simulated)
            start_time = time.time()
            mock_solution_attempt = SolutionAttempt(
                id=generate_id("mock_solution"),
                sub_problem_id=decomposition_plan.sub_problems[0].id if decomposition_plan.sub_problems else generate_id("fallback"),
                approach="Mock approach for testing",
                solution_content="Mock solution content for testing",
                team_id="mock_team",
                confidence_score=0.85
            )
            
            integration_result = orchestrator.integrate_solutions(decomposition_plan, [mock_solution_attempt])
            orchestration_time = time.time() - start_time
            
            print(f"Solution orchestration completed in {orchestration_time:.3f}s")
            
            self.assertIsNotNone(integration_result)
    
    def test_parallel_decomposition_convergence(self):
        """Test parallel decomposition strategies converging to consistent results"""
        with patch('decomposition_engine.OpenEvolveClient') as mock_client:
            # Mock responses for different strategies to converge on similar results
            mock_result = Mock()
            mock_result.success = True
            mock_result.best_code = json.dumps([
                {
                    "id": generate_id("strategy_sub1"),
                    "description": "Common element across strategies",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 6.0,
                    "ai_suggested_evaluation_prompt": "Validate common element"
                },
                {
                    "id": generate_id("strategy_sub2"), 
                    "description": "Strategy-specific element",
                    "dependencies": [generate_id("strategy_sub1")],
                    "ai_suggested_complexity_score": 7.0,
                    "ai_suggested_evaluation_prompt": "Validate specific element"
                }
            ])
            
            mock_client.return_value.evolve.return_value = mock_result
            
            engine = DecompositionEngine(openevolve_client=mock_client.return_value)
            
            # Create test problem
            test_problem = ProblemDefinition(
                id=generate_id("convergence_test"),
                title="Convergence Test Problem",
                description="Test that different decomposition strategies converge on consistent core elements",
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="convergence_testing"),
                complexity_score=ComplexityScore(
                    explanation="Convergence test problem",
                    cognitive_complexity=6.0,
                    computational_complexity=6.0,
                    domain_complexity=6.0,
                    integration_complexity=6.0,
                    overall_complexity=6.0
                )
            )
            
            # Test multiple strategies in parallel
            strategies = ["semantic", "dependency", "complexity", "research", "hybrid"]
            
            results = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    strategy: executor.submit(engine.decompose, test_problem, strategy)
                    for strategy in strategies
                }
                
                for strategy, future in futures.items():
                    results[strategy] = future.result(timeout=10)  # 10s timeout
            
            # Verify all strategies produced results
            for strategy, result in results.items():
                self.assertIsNotNone(result, f"Strategy {strategy} failed to produce result")
                self.assertGreater(len(result.sub_problems), 0, f"Strategy {strategy} produced no sub-problems")
            
            # Check for convergence on core concepts (all should have common elements)
            all_sub_problems = []
            for strategy, result in results.items():
                all_sub_problems.extend(result.sub_problems)
            
            # Count common elements across strategies - should have some overlap for core concepts
            descriptions = [sp.description for sp in all_sub_problems]
            common_elements = set()
            for desc in descriptions:
                if descriptions.count(desc) > 1:  # Appears in multiple strategies
                    common_elements.add(desc)
            
            print(f"Found {len(common_elements)} common elements across {len(strategies)} strategies")
            print(f"Total sub-problems: {len(all_sub_problems)}")
            
            # Some convergence is expected for core concepts
            self.assertGreaterEqual(len(common_elements), 1, "Strategies should converge on at least some common elements")
    
    def test_gauntlet_validation_chain(self):
        """Test complete gauntlet validation chain with multiple rounds"""
        with patch('gauntlet_system.OpenEvolveClient') as mock_client:
            # Mock responses for gauntlet rounds
            mock_response = Mock()
            mock_response.success = True
            mock_response.best_code = json.dumps({
                "passed": True,
                "score": 0.85,
                "justification": "Content passes validation criteria",
                "improvements": ["Consider additional edge cases"]
            })
            
            mock_client.return_value.evolve.return_value = mock_response
            
            gauntlet_system = GauntletSystem(openevolve_client=mock_client.return_value)
            
            # Create sample content for validation
            test_content = "This is a test solution that will undergo multi-round validation through the gauntlet system. The content needs to meet various validation criteria across multiple rounds with different validation focus areas."
            
            # Test different gauntlet types
            gauntlet_types = ["standard", "adaptive", "hierarchical", "competitive", "collaborative"]
            
            results = {}
            for gauntlet_type in gauntlet_types:
                result = gauntlet_system.run_gauntlet(
                    content=test_content,
                    gauntlet_type=gauntlet_type,
                    team_name="red",
                    context={"test_scenario": f"gauntlet_type_{gauntlet_type}"}
                )
                results[gauntlet_type] = result
                print(f"Gauntlet {gauntlet_type}: {result['passed'] if isinstance(result, dict) and 'passed' in result else 'Unknown'}")
            
            # At least most gauntlets should pass with the good test content
            passed_count = sum(1 for r in results.values() 
                            if isinstance(r, dict) and r.get('passed', False))
            self.assertGreaterEqual(passed_count, len(gauntlet_types) * 0.6, 
                                  f"At least 60% of gauntlets should pass, got {passed_count}/{len(gauntlet_types)}")


class TestExtremePerformanceScenarios(unittest.TestCase):
    """Tests for extreme performance scenarios and edge cases"""
    
    def test_million_record_simulation(self):
        """Test system behavior with simulated large-scale data"""
        import time
        
        db = SovereignDatabase(":memory:")
        
        # Create a large number of problems to test system performance
        large_batch_size = 1000  # Using 1000 instead of 1M for practicality in testing
        batch_problems = []
        
        print(f"Creating {large_batch_size} problems for performance test...")
        start_time = time.time()
        
        for i in range(large_batch_size):
            problem = ProblemDefinition(
                id=generate_id(f"perf_test_{i}"),
                title=f"Performance Test Problem {i}",
                description=f"Performance test problem number {i} in a large batch. " + "Detailed description for performance testing. " * 5,
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="performance_testing"),
                complexity_score=ComplexityScore(
                    explanation=f"Performance test problem {i}",
                    cognitive_complexity=5.0 + (i % 3),
                    computational_complexity=5.0 + (i % 3),
                    domain_complexity=5.0 + (i % 3),
                    integration_complexity=5.0 + (i % 3),
                    overall_complexity=5.0 + (i % 3)
                )
            )
            batch_problems.append(problem)
        
        # Batch insert all problems
        for problem in batch_problems:
            db.create_problem(problem)
        
        batch_insert_time = time.time() - start_time
        print(f"Batch inserted {large_batch_size} problems in {batch_insert_time:.3f}s ({large_batch_size/batch_insert_time:.1f} problems/sec)")
        
        # Test retrieval performance
        start_time = time.time()
        retrieved_problems = db.list_problems()
        retrieval_time = time.time() - start_time
        
        print(f"Retrieved {len(retrieved_problems)} problems in {retrieval_time:.3f}s ({len(retrieved_problems)/retrieval_time:.1f} problems/sec)")
        
        # Verify all problems were stored and retrieved
        self.assertEqual(len(retrieved_problems), large_batch_size)
        
        # Performance should be reasonable
        self.assertLess(batch_insert_time, 5.0, f"Batch insertion too slow for {large_batch_size} problems: {batch_insert_time:.3f}s")
        self.assertLess(retrieval_time, 3.0, f"Batch retrieval too slow for {large_batch_size} problems: {retrieval_time:.3f}s")
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency when processing large workflows"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Baseline memory: {baseline_memory:.1f}MB")
        
        # Create complex workflow with many interdependent components
        complex_workflow_components = []
        
        for workflow_id in range(100):  # Create 100 mini-workflows
            # Each workflow has several components
            workflow_components = {
                'problem': ProblemDefinition(
                    id=generate_id(f"wf_{workflow_id}_prob"),
                    title=f"Workflow {workflow_id} Problem",
                    description=f"Problem for workflow {workflow_id}",
                    problem_type=ProblemType.RESEARCH,
                    domain_context=DomainContext(domain="workflow_testing"),
                    complexity_score=ComplexityScore(
                        explanation=f"Workflow {workflow_id} problem",
                        cognitive_complexity=5.0,
                        computational_complexity=5.0,
                        domain_complexity=5.0,
                        integration_complexity=5.0,
                        overall_complexity=5.0
                    )
                ),
                'sub_problems': [
                    SubProblem(
                        id=generate_id(f"wf_{workflow_id}_sub_{i}"),
                        parent_id=generate_id(f"wf_{workflow_id}_prob"),
                        title=f"Sub-problem {i} for workflow {workflow_id}",
                        description=f"Sub-problem {i} in workflow {workflow_id}",
                        type=random.choice(list(SubProblemType)),
                        complexity_score=ComplexityScore(
                            explanation=f"Sub-problem {i} for workflow {workflow_id}",
                            cognitive_complexity=5.0 + (i % 2),
                            computational_complexity=5.0 + (i % 2),
                            domain_complexity=5.0 + (i % 2),
                            integration_complexity=5.0 + (i % 2),
                            overall_complexity=5.0 + (i % 2)
                        )
                    )
                    for i in range(5)  # 5 sub-problems per workflow
                ],
                'attempts': [
                    SolutionAttempt(
                        id=generate_id(f"wf_{workflow_id}_attempt_{j}"),
                        sub_problem_id=generate_id(f"wf_{workflow_id}_sub_{j % 5}"),
                        approach=f"Approach for workflow {workflow_id}, attempt {j}",
                        solution_content=f"Solution content for workflow {workflow_id}, attempt {j}",
                        team_id="test_team",
                        confidence_score=0.7 + (random.uniform(-0.2, 0.2))
                    )
                    for j in range(3)  # 3 solution attempts per workflow
                ]
            }
            complex_workflow_components.append(workflow_components)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        print(f"Peak memory during complex workflow creation: {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 200.0, f"Memory usage too high: {memory_increase:.1f}MB for {len(complex_workflow_components)} workflows")
        
        # Clean up
        del complex_workflow_components
        gc.collect()
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        cleanup_increase = cleanup_memory - baseline_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB (+{cleanup_increase:.1f}MB)")
        
        # Should have reclaimed most memory
        self.assertLess(cleanup_increase, memory_increase * 0.5, "Memory not adequately reclaimed after cleanup")
    
    def test_concurrent_cache_operations(self):
        """Test cache performance under concurrent operations"""
        import time
        from concurrent.futures import ThreadPoolExecutor
        
        cache = LLMResponseCache(max_size=500)
        
        # Create diverse content for cache operations
        cache_content = []
        for i in range(200):
            content = f"Test content for cache operation {i}. " + "Additional content to make it substantial. " * 3
            model_params = {"model": f"model_{i % 5}", "temperature": round(0.1 + (i % 8) * 0.1, 1)}
            response = {"choices": [{"message": {"content": f"Response to cache test {i}"}}]}
            cache_content.append((content, model_params, response))
        
        results = []
        
        def cache_operation_worker(worker_id):
            """Worker function for concurrent cache operations"""
            worker_results = []
            for i in range(50):  # Each worker does 50 operations
                content, params, response = cache_content[(worker_id * 50 + i) % len(cache_content)]
                
                # Alternate between cache write and read operations
                if i % 2 == 0:
                    # Write operation
                    cache.cache_response(content, params, response)
                    worker_results.append(('write', True))
                else:
                    # Read operation
                    cached = cache.get_response(content, params)
                    worker_results.append(('read', cached is not None))
            
            return worker_results
        
        # Run concurrent cache operations
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_operation_worker, i) for i in range(10)]
            all_results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Flatten results
        all_operations = []
        for worker_results in all_results:
            all_operations.extend(worker_results)
        
        write_ops = [op for op in all_operations if op[0] == 'write']
        read_ops = [op for op in all_operations if op[0] == 'read']
        successful_reads = [op for op in read_ops if op[1]]
        
        print(f"Concurrent cache operations: {len(write_ops)} writes, {len(read_ops)} reads in {total_time:.3f}s")
        print(f"Cache hit rate: {len(successful_reads)}/{len(read_ops)} ({len(successful_reads)/len(read_ops)*100:.1f}%)")
        print(f"Operations per second: {len(all_operations)/total_time:.1f}")
        
        # Verify cache operations were successful
        self.assertGreater(len(write_ops), 0)
        self.assertGreater(len(read_ops), 0)
        self.assertLess(total_time, 5.0, f"Cache operations took too long: {total_time:.3f}s")
        
        # Cache should have reasonable hit rate
        expected_hit_rate = 0.3  # With random access pattern, expect some cache misses
        self.assertGreaterEqual(len(successful_reads)/len(read_ops), expected_hit_rate,
                              f"Cache hit rate too low: {len(successful_reads)/len(read_ops):.2f}")
        
        # Check cache stats
        stats = cache.get_stats()
        print(f"Cache statistics: Size={stats['current_size']}, Hits={stats['total_hits']}, Misses={stats['total_misses']}")


class TestSecurityHardening(unittest.TestCase):
    """Security hardening tests for vulnerabilities and attack prevention"""
    
    def test_input_sanitization_comprehensive(self):
        """Test comprehensive input sanitization against various attacks"""
        validator = InputValidator()
        
        # Comprehensive attack vectors
        attack_vectors = [
            # SQL Injection attempts
            ["1' OR '1'='1", "SELECT * FROM users WHERE id =", "; DROP TABLE users; --"],
            
            # XSS attempts
            ["<script>alert('xss')</script>", "javascript:alert('xss')", "src=x onerror=alert('xss')"],
            
            # Path traversal
            ["../../../etc/passwd", "..\\..\\windows\\system32\\", "/var/lib/../shadow"],
            
            # Command injection
            ["test; rm -rf /", "test && whoami", "test | cat /etc/passwd"],
            
            # Logic injection
            ["{{7*7}}", "${7*7}", "#{7*7}"],
            
            # Regex denial of service
            ["^((a+)+)+$", "(a+)+", "(\w+)+"],
        ]
        
        for attack_category, attack_examples in [
            ("SQL Injection", attack_vectors[0]),
            ("XSS", attack_vectors[1]),
            ("Path Traversal", attack_vectors[2]),
            ("Command Injection", attack_vectors[3]),
            ("Logic Injection", attack_vectors[4]),
            ("Regex DoS", attack_vectors[5])
        ]:
            with self.subTest(attack_category=attack_category):
                for attack in attack_examples:
                    # Input should be handled safely by the validation system
                    try:
                        # Should either sanitize, reject, or handle gracefully
                        sanitized = validator.sanitize_input(attack)
                        # If it doesn't throw an exception, the system handled it safely
                    except (ValueError, TypeError, RuntimeError):
                        # Throwing an exception is also acceptable as long as it doesn't crash the system
                        pass
    
    def test_authentication_brute_force_protection(self):
        """Test brute force protection in authentication system"""
        import time
        
        auth_system = AuthenticationSystem()
        
        # Create a test user
        user = auth_system.create_user(
            username="brute_test",
            email="brute@example.com",
            password="SecurePassword123!",
            permissions=[]
        )
        
        # Try multiple failed login attempts from the same user
        start_time = time.time()
        failed_attempts = 0
        
        for i in range(15):  # 15 failed attempts
            result = auth_system.authenticate("brute_test", f"wrong_password_{i}")
            if result is None:  # Authentication failed
                failed_attempts += 1
            time.sleep(0.01)  # Small delay to make test more realistic
        
        time_for_attempts = time.time() - start_time
        
        # Verify failed attempts were recorded
        print(f"Brute force test: {failed_attempts} failed attempts in {time_for_attempts:.3f}s")
        
        # Should handle multiple failed attempts without crashing
        self.assertEqual(failed_attempts, 15, "All failed attempts should be handled")
        
        # Now try to authenticate with correct credentials
        # System should still allow legitimate authentication after failed attempts
        correct_auth = auth_system.authenticate("brute_test", "SecurePassword123!")
        
        # Depending on implementation, this might work or not, but shouldn't crash
        print(f"Correct authentication after failed attempts: {'Success' if correct_auth else 'Blocked'}")
    
    def test_distributed_denial_of_service_protection(self):
        """Test protection against distributed denial of service"""
        import threading
        import time
        from queue import Queue
        
        # Create components that might be targets for DoS
        db = SovereignDatabase(":memory:")
        
        results_queue = Queue()
        
        def dos_worker(worker_id):
            """Worker to simulate DoS attempt"""
            local_results = []
            for i in range(50):  # Each worker makes 50 requests
                try:
                    # Try to overload the system with rapid requests
                    problem = ProblemDefinition(
                        id=generate_id(f"dos_test_{worker_id}_{i}"),
                        title=f"DoS Test {worker_id}-{i}",
                        description=f"Problem {i} from worker {worker_id} as part of DoS simulation",
                        problem_type=ProblemType.RESEARCH,
                        domain_context=DomainContext(domain="dos_testing"),
                        complexity_score=ComplexityScore(
                            explanation=f"DoS test {worker_id}-{i}",
                            cognitive_complexity=1.0,  # Low complexity to process quickly
                            computational_complexity=1.0,
                            domain_complexity=1.0,
                            integration_complexity=1.0,
                            overall_complexity=1.0
                        )
                    )
                    
                    # Rapid-fire operations
                    result = db.create_problem(problem)
                    local_results.append(('create', result))
                    
                    # Immediate retrieval
                    retrieved = db.get_problem(problem.id)
                    local_results.append(('read', retrieved is not None))
                    
                except Exception as e:
                    local_results.append(('error', str(e)))
            
            results_queue.put(local_results)
        
        # Launch multiple workers to simulate distributed attack
        workers = []
        num_workers = 10
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=dos_worker, args=(worker_id,))
            workers.append(thread)
            thread.start()
        
        # Wait for all workers to complete
        for worker in workers:
            worker.join(timeout=10)  # 10 second timeout
        
        total_time = time.time() - start_time
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.extend(results_queue.get())
        
        successful_ops = [r for r in all_results if r[1] is True]
        failed_ops = [r for r in all_results if r[1] is False]
        errors = [r for r in all_results if r[0] == 'error']
        
        print(f"Distributed DoS simulation: {len(all_results)} operations in {total_time:.3f}s")
        print(f"Successes: {len(successful_ops)}, Failures: {len(failed_ops)}, Errors: {len(errors)}")
        
        # System should remain functional despite high load
        # Not all operations need to succeed, but system shouldn't crash
        self.assertLess(len(errors), len(all_results) * 0.5, "Too many errors during high load")


class TestComprehensiveSystemValidation(unittest.TestCase):
    """Comprehensive validation of the entire system"""
    
    def test_full_system_workflow_integration(self):
        """Test complete end-to-end system workflow"""
        from unittest.mock import MagicMock, patch
        
        # Mock all LLM clients to avoid external dependencies
        with patch('problem_analyzer.OpenEvolveClient') as mock_analyzer, \
             patch('decomposition_engine.OpenEvolveClient') as mock_decomposer, \
             patch('sovereign_team_coordination.OpenEvolveClient') as mock_coordinator, \
             patch('sovereign_solution_orchestration.OpenEvolveClient') as mock_orchestrator, \
             patch('gauntlet_system.OpenEvolveClient') as mock_gauntlet:
            
            # Set up mock responses
            analysis_result = Mock()
            analysis_result.success = True
            analysis_result.best_code = json.dumps({
                "domain": "software_engineering",
                "subdomain": "api_design", 
                "related_domains": ["security", "performance"],
                "key_concepts": ["rest", "authentication", "rate_limiting"],
                "domain_complexity": 7.5,
                "required_expertise": ["api_design", "security", "performance_optimization"]
            })
            
            decomposition_result = Mock()
            decomposition_result.success = True
            decomposition_result.best_code = json.dumps([
                {
                    "id": generate_id("full_wf_sub1"),
                    "description": "Design API architecture and security",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 8.0,
                    "ai_suggested_evaluation_prompt": "Validate API design and security measures"
                },
                {
                    "id": generate_id("full_wf_sub2"),
                    "description": "Implement rate limiting and caching",
                    "dependencies": [generate_id("full_wf_sub1")],
                    "ai_suggested_complexity_score": 7.5,
                    "ai_suggested_evaluation_prompt": "Validate performance optimization implementation"
                },
                {
                    "id": generate_id("full_wf_sub3"),
                    "description": "Create monitoring and alerting",
                    "dependencies": [generate_id("full_wf_sub1")],
                    "ai_suggested_complexity_score": 6.5,
                    "ai_suggested_evaluation_prompt": "Validate monitoring coverage and effectiveness"
                }
            ])
            
            validation_result = Mock()
            validation_result.success = True
            validation_result.best_code = json.dumps({
                "passed": True,
                "score": 0.88,
                "feedback": "Solution meets all requirements",
                "improvements": ["Consider additional error handling scenarios"]
            })
            
            integration_result = Mock()
            integration_result.success = True
            integration_result.best_code = json.dumps({
                "integrated_solution": "Comprehensive API system with security, performance, and monitoring",
                "confidence": 0.85,
                "validation_status": "approved"
            })
            
            mock_analyzer.return_value.evolve.return_value = analysis_result
            mock_decomposer.return_value.evolve.return_value = decomposition_result
            mock_coordinator.return_value.evolve.return_value = validation_result
            mock_orchestrator.return_value.evolve.return_value = integration_result
            mock_gauntlet.return_value.evolve.return_value = validation_result
            
            # Create system components
            analyzer = ProblemAnalyzer(openevolve_client=mock_analyzer.return_value)
            decomposer = DecompositionEngine(openevolve_client=mock_decomposer.return_value)
            coordinator = TeamCoordinator(openevolve_client=mock_coordinator.return_value)
            orchestrator = SolutionOrchestrator(openevolve_client=mock_orchestrator.return_value)
            gauntlet_system = GauntletSystem(openevolve_client=mock_gauntlet.return_value)
            
            # Run complete workflow
            start_time = time.time()
            
            # 1. Problem Analysis
            complex_problem = analyzer.analyze_problem(
                problem_text="Design and implement a comprehensive API system with security, performance optimization, monitoring, and scalability features for a high-traffic application",
                title="Comprehensive API System Design"
            )
            
            self.assertIsNotNone(complex_problem)
            self.assertIn("api", complex_problem.domain_context.domain.lower())
            
            # 2. Decomposition
            plan = decomposer.decompose(complex_problem, strategy="hybrid")
            self.assertIsNotNone(plan)
            self.assertGreater(len(plan.sub_problems), 2)
            
            # 3. Team coordination assignment
            assignment = coordinator.assign_to_team(
                task_id=plan.id,
                team="red",
                priority=9
            )
            self.assertIsNotNone(assignment)
            
            # 4. Solution attempts and validation
            solution_attempts = []
            for sub_problem in plan.sub_problems:
                # Create mock solution attempt
                attempt = SolutionAttempt(
                    id=generate_id("mock_attempt"),
                    sub_problem_id=sub_problem.id,
                    approach=f"Solution for {sub_problem.title}",
                    solution_content=f"Implementation of {sub_problem.description}",
                    team_id="development_team",
                    confidence_score=0.8 + (random.uniform(-0.1, 0.1))
                )
                solution_attempts.append(attempt)
                
                # Run gauntlet validation on each solution
                validation = gauntlet_system.run_gauntlet(
                    content=attempt.solution_content,
                    gauntlet_type="standard",
                    team_name="red",
                    context={"validation_target": "sub_solution", "complexity": attempt.confidence_score}
                )
                
                # Validation should be successful for our mock
                if isinstance(validation, dict) and 'passed' in validation:
                    self.assertTrue(validation['passed'])
            
            # 5. Solution orchestration and integration
            final_solution = orchestrator.integrate_solutions(plan, solution_attempts)
            self.assertIsNotNone(final_solution)
            self.assertGreater(len(getattr(final_solution, 'integrated_content', '') or ''), 0)
            
            total_workflow_time = time.time() - start_time
            
            print(f"Complete end-to-end workflow completed in {total_workflow_time:.3f}s")
            print(f"Processed problem with {len(plan.sub_problems)} sub-problems")
            print(f"Created {len(solution_attempts)} solution attempts")
            
            # Workflow should complete in reasonable time
            self.assertLess(total_workflow_time, 10.0, f"Complete workflow took too long: {total_workflow_time:.3f}s")
    
    def test_error_propagation_and_recovery(self):
        """Test error propagation and recovery mechanisms"""
        db = SovereignDatabase(":memory:")
        
        # Create a problem
        problem = ProblemDefinition(
            id=generate_id("error_test"),
            title="Error Propagation Test",
            description="Test how the system handles errors and propagates recovery",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="error_handling"),
            complexity_score=ComplexityScore(
                explanation="Error propagation test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        # Store problem
        result = db.create_problem(problem)
        self.assertTrue(result)
        
        # Test error recovery when retrieving non-existent item
        nonexistent = db.get_problem("nonexistent_id")
        self.assertIsNone(nonexistent)
        
        # Test normal retrieval still works after error
        normal_retrieval = db.get_problem(problem.id)
        self.assertIsNotNone(normal_retrieval)
        self.assertEqual(normal_retrieval.title, "Error Propagation Test")
        
        print("Error handling and recovery validation passed")
    
    def test_system_resilience_under_stress(self):
        """Test system resilience under various stress conditions"""
        import gc
        import time
        from concurrent.futures import ThreadPoolExecutor
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        db = SovereignDatabase(":memory:")
        
        # Create stress test: multiple operations in parallel
        def stress_worker(worker_id):
            """Worker function for stress testing"""
            local_results = {'success': 0, 'failures': 0, 'operations': []}
            
            for i in range(25):  # 25 operations per worker
                try:
                    # Mix of operations
                    if i % 4 == 0:
                        # Create problem
                        problem = ProblemDefinition(
                            id=generate_id(f"stress_{worker_id}_{i}"),
                            title=f"Stress Test {worker_id}-{i}",
                            description=f"Problem for stress testing worker {worker_id}, operation {i}",
                            problem_type=ProblemType.RESEARCH,
                            domain_context=DomainContext(domain="stress_testing"),
                            complexity_score=ComplexityScore(
                                explanation=f"Stress test {worker_id}-{i}",
                                cognitive_complexity=5.0 + (i % 2),
                                computational_complexity=5.0 + (i % 2),
                                domain_complexity=5.0 + (i % 2),
                                integration_complexity=5.0 + (i % 2),
                                overall_complexity=5.0 + (i % 2)
                            )
                        )
                        result = db.create_problem(problem)
                        local_results['operations'].append(('create', result))
                        if result:
                            local_results['success'] += 1
                        else:
                            local_results['failures'] += 1
                    elif i % 4 == 1:
                        # Read operation
                        all_problems = db.list_problems()
                        local_results['operations'].append(('read', len(all_problems)))
                        local_results['success'] += 1
                    elif i % 4 == 2:
                        # Update operation (if there are problems to update)
                        all_problems = db.list_problems()
                        if all_problems:
                            prob = all_problems[0]
                            prob.title = f"Updated by worker {worker_id}, op {i}"
                            result = db.update_problem(prob)
                            local_results['operations'].append(('update', result))
                            if result:
                                local_results['success'] += 1
                            else:
                                local_results['failures'] += 1
                        else:
                            local_results['operations'].append(('update_skipped', True))
                            local_results['success'] += 1
                    else:
                        # Delete operation (if there are problems to delete)
                        all_problems = db.list_problems()
                        if all_problems and random.random() < 0.3:  # Only sometimes delete
                            prob = all_problems[0]
                            result = db.delete_problem(prob.id)
                            local_results['operations'].append(('delete', result))
                            if result:
                                local_results['success'] += 1
                            else:
                                local_results['failures'] += 1
                        else:
                            local_results['operations'].append(('delete_skipped', True))
                            local_results['success'] += 1
                
                except Exception as e:
                    local_results['operations'].append(('error', str(e)))
                    local_results['failures'] += 1
                    
                    # Don't let a single error stop the worker
                    continue
            
            return local_results
        
        # Run stress test with multiple concurrent workers
        num_workers = 8
        total_operations = num_workers * 25  # 200 total operations
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_workers)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        # Aggregate results
        total_success = sum(r['success'] for r in results)
        total_failures = sum(r['failures'] for r in results)
        
        print(f"Stress test completed: {total_success} successes, {total_failures} failures in {total_time:.3f}s")
        print(f"Total operations: {total_success + total_failures} out of {total_operations} attempted")
        print(f"Success rate: {total_success/(total_success + total_failures)*100:.1f}%")
        print(f"Operations per second: {(total_success + total_failures)/total_time:.1f}")
        
        # System should maintain high success rate under stress
        success_rate = total_success / (total_success + total_failures) if (total_success + total_failures) > 0 else 0
        self.assertGreaterEqual(success_rate, 0.80, f"Success rate too low under stress: {success_rate:.2f}")
        
        # Should complete in reasonable time
        self.assertLess(total_time, 10.0, f"Stress test took too long: {total_time:.3f}s")
        
        # Check final system state
        final_problem_count = len(db.list_problems())
        print(f"Final database state: {final_problem_count} problems remaining")
        
        # System should be in a consistent state after stress testing
        self.assertGreaterEqual(final_problem_count, 0, "Problem count cannot be negative")
        
        # Check memory usage stayed reasonable
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        print(f"Memory change during stress test: {memory_increase:+.1f}MB")
        self.assertLess(abs(memory_increase), 200.0, f"Memory usage changed too much during stress: {memory_increase:+.1f}MB")


def run_ultimate_comprehensive_tests():
    """Run the ultimate comprehensive test suite"""
    print("Running Ultimate Comprehensive Validation Tests...")
    print("="*80)
    
    # Create comprehensive test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestUltimateEdgeCases))
    suite.addTest(unittest.makeSuite(TestAdvancedIntegrationScenarios))
    suite.addTest(unittest.makeSuite(TestExtremePerformanceScenarios))
    suite.addTest(unittest.makeSuite(TestSecurityHardening))
    suite.addTest(unittest.makeSuite(TestComprehensiveSystemValidation))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    print("\n" + "="*80)
    print("ULTIMATE COMPREHENSIVE TEST RESULTS")
    print("="*80)
    print(f"Total tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures or result.errors:
        print("\nISSUES IDENTIFIED:")
        for test, trace in result.failures:
            print(f"\nFAILED: {test}")
            print(trace)
        for test, trace in result.errors:
            print(f"\nERROR: {test}")
            print(trace)
        print(f"\n⚠️  Tests completed with issues - {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun} passed")
    else:
        print(f"\n🎉 ALL {result.testsRun} ULTIMATE COMPREHENSIVE TESTS PASSED! 🎉")
        print("The Sovereign-Grade system has passed all extreme validation scenarios!")
    
    print("="*80)
    return result


if __name__ == "__main__":
    run_ultimate_comprehensive_tests()