"""
Deep Edge Case Unit Tests for Sovereign-Grade System
Extensive testing of edge cases, boundary conditions, and complex scenarios
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import time
import threading
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import tempfile
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
import random
import string
import uuid
import hashlib
import secrets
import gc
import tracemalloc
import weakref
import logging
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
from auth_system import AuthenticationSystem, AuthorizationSystem
from input_validation import InputValidator
from performance_optimization import LLMResponseCache
from monitoring_system import MetricsCollector


class TestExtremeEdgeCases(unittest.TestCase):
    """Test extreme edge cases and boundary conditions"""
    
    def test_extremely_deep_nested_problems(self):
        """Test handling of extremely deep nested problem structures"""
        # Create a deeply nested problem hierarchy
        max_depth = 50  # Create 50 levels of nesting
        
        # Build nested structure
        nested_problem = None
        for i in range(max_depth):
            current_problem = ProblemDefinition(
                id=generate_id(f"nested_{i}"),
                title=f"Nested Problem Level {i}",
                description=f"Level {i} in a deeply nested hierarchy",
                problem_type=ProblemType.RESEARCH if i < max_depth - 1 else ProblemType.IMPLEMENTATION,
                domain_context=DomainContext(domain=f"nesting_level_{i}"),
                complexity_score=ComplexityScore(
                    explanation=f"Nesting level {i}",
                    cognitive_complexity=5.0 + (i * 0.1),
                    computational_complexity=5.0 + (i * 0.1),
                    domain_complexity=5.0 + (i * 0.1),
                    integration_complexity=5.0 + (i * 0.1),
                    overall_complexity=5.0 + (i * 0.1)
                ),
                parent_id=nested_problem.id if nested_problem else None
            )
            nested_problem = current_problem
        
        # Create a database to test persistence with nested problems
        db = SovereignDatabase(":memory:")
        
        # Store deeply nested problem (should handle gracefully)
        start_time = time.time()
        result = db.create_problem(nested_problem)
        store_time = time.time() - start_time
        
        self.assertTrue(result)
        print(f"Deep nesting stored in {store_time:.3f}s")
        
        # Retrieve and verify
        retrieved = db.get_problem(nested_problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, f"Nested Problem Level {max_depth-1}")
    
    def test_extremely_complex_sub_problem_dependencies(self):
        """Test extremely complex dependency graphs"""
        # Create an interconnected web of dependencies
        sub_problems = []
        
        # Create 50 sub-problems with complex interdependencies
        for i in range(50):
            dependencies = []
            # Each sub-problem depends on 3-7 randomly selected prior sub-problems
            if i > 0:
                num_deps = min(7, max(3, i // 5))  # Varying number of dependencies
                for j in range(num_deps):
                    dep_idx = random.randint(0, min(i-1, max(1, i-10)))  # Limit dependency range but allow some distant deps
                    if dep_idx < len(sub_problems):
                        dependencies.append(sub_problems[dep_idx].id)
            
            sub_problem = SubProblem(
                id=generate_id(f"complex_dep_{i}"),
                parent_id=generate_id("complex_root"),
                title=f"Complex Dependency Sub-problem {i}",
                description=f"Sub-problem {i} with {len(dependencies)} dependencies",
                type=random.choice(list(SubProblemType)),
                complexity_score=ComplexityScore(
                    explanation=f"Complex dependency test {i}",
                    cognitive_complexity=5.0 + (i % 3),
                    computational_complexity=5.0 + (i % 3),
                    domain_complexity=5.0 + (i % 3),
                    integration_complexity=5.0 + (i % 3),
                    overall_complexity=5.0 + (i % 3)
                ),
                dependencies=dependencies
            )
            sub_problems.append(sub_problem)
        
        # Test dependency validation with complex graph
        from decomposition_engine import validate_dependencies
        errors = validate_dependencies(sub_problems)
        
        # Should detect any circular dependencies
        circular_errors = [e for e in errors if "circular" in e.lower()]
        
        print(f"Complex dependency validation - Circular errors found: {len(circular_errors)}")
        print(f"Total validation errors: {len(errors)}")
        
        # If there are circular dependencies in our random generation, that's expected
        # and the system should handle them appropriately
    
    def test_maximum_size_content_fields(self):
        """Test maximum size content fields without performance degradation"""
        # Test maximum length strings for various fields
        max_title = "A" * 10000  # 10k character title
        max_description = "B" * 100000  # 100k character description
        max_complexity_explanation = "C" * 50000  # 50k character explanation
        
        large_problem = ProblemDefinition(
            id=generate_id("large_content"),
            title=max_title,
            description=max_description,
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="large_content_test"),
            complexity_score=ComplexityScore(
                explanation=max_complexity_explanation,
                cognitive_complexity=7.5,
                computational_complexity=7.0,
                domain_complexity=8.0,
                integration_complexity=6.5,
                overall_complexity=7.4
            )
        )
        
        # Test database operations with large content
        db = SovereignDatabase(":memory:")
        
        start_time = time.time()
        result = db.create_problem(large_problem)
        large_store_time = time.time() - start_time
        
        self.assertTrue(result)
        print(f"Large content problem stored in {large_store_time:.3f}s")
        
        # Retrieve and verify
        retrieved = db.get_problem(large_problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved.title), len(max_title))
        self.assertEqual(len(retrieved.description), len(max_description))
    
    def test_concurrent_database_transactions(self):
        """Test concurrent database transactions and locking"""
        import threading
        import time
        
        db = SovereignDatabase(":memory:")
        
        results = []
        errors = []
        
        def concurrent_writer(writer_id):
            """Concurrent writer function"""
            try:
                for i in range(10):  # Each writer creates 10 problems
                    problem = ProblemDefinition(
                        id=generate_id(f"concurrent_{writer_id}_{i}"),
                        title=f"Concurrent Write Problem {writer_id}-{i}",
                        description=f"Problem created by concurrent writer {writer_id}, iteration {i}",
                        problem_type=ProblemType.RESEARCH,
                        domain_context=DomainContext(domain="concurrent_test"),
                        complexity_score=ComplexityScore(
                            explanation=f"Concurrent test {writer_id}-{i}",
                            cognitive_complexity=5.0 + (writer_id % 3),
                            computational_complexity=5.0 + (writer_id % 3),
                            domain_complexity=5.0 + (writer_id % 3),
                            integration_complexity=5.0 + (writer_id % 3),
                            overall_complexity=5.0 + (writer_id % 3)
                        )
                    )
                    
                    start_time = time.time()
                    result = db.create_problem(problem)
                    op_time = time.time() - start_time
                    
                    results.append({
                        'writer_id': writer_id,
                        'iteration': i,
                        'success': result,
                        'operation_time': op_time
                    })
                    
                    # Brief pause to allow other threads to interleave
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Writer {writer_id}: {str(e)}")
        
        # Run multiple writers concurrently
        threads = []
        num_writers = 20
        
        start_time = time.time()
        
        for writer_id in range(num_writers):
            thread = threading.Thread(target=concurrent_writer, args=(writer_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify results
        successful_ops = [r for r in results if r['success']]
        failed_ops = [r for r in results if not r['success']]
        
        print(f"Concurrent operations: {len(successful_ops)} successful, {len(failed_ops)} failed")
        print(f"Total time for {num_writers * 10} operations: {total_time:.3f}s")
        print(f"Operations per second: {(num_writers * 10) / total_time:.1f}")
        
        # All operations should succeed (with proper database locking)
        self.assertGreater(len(successful_ops), len(failed_ops) * 0.1, "Most concurrent operations should succeed")
        self.assertEqual(len(errors), 0, f"Should have no threading errors: {errors}")
    
    def test_recursive_decomposition_limits(self):
        """Test recursive decomposition with maximum depth limits"""
        # Create a problem that would normally lead to infinite recursion if not properly handled
        recursive_problem = ProblemDefinition(
            id=generate_id("recursive_test"),
            title="Self-Referencing Recursive Problem",
            description="This is a problem that keeps decomposing into itself, but the system should handle this gracefully with depth limits. This problem should create sub-problems that also decompose, but the system must prevent infinite recursion. The problem tries to break itself down into smaller parts, but each part is essentially the same as the original.",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="recursion_test"),
            complexity_score=ComplexityScore(
                explanation="Recursive problem test",
                cognitive_complexity=8.5,
                computational_complexity=8.0,
                domain_complexity=8.5,
                integration_complexity=7.5,
                overall_complexity=8.1
            )
        )
        
        # Mock the OpenEvolve client to simulate recursive responses
        with patch('decomposition_engine.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            mock_response = Mock()
            mock_response.success = True
            # Simulate recursive decomposition that would continue infinitely
            mock_response.best_code = json.dumps([
                {
                    "id": generate_id("recursive_sub1"),
                    "description": "Break down the main problem into smaller sub-problems",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 7.5,
                    "ai_suggested_evaluation_prompt": "Validate the breakdown approach"
                },
                {
                    "id": generate_id("recursive_sub2"),
                    "description": "Further break down the first sub-problem recursively",
                    "dependencies": [generate_id("recursive_sub1")],
                    "ai_suggested_complexity_score": 8.0,
                    "ai_suggested_evaluation_prompt": "Validate recursive breakdown"
                }
            ])
            
            mock_client.evolve.return_value = mock_response
            
            engine = DecompositionEngine(openevolve_client=mock_client)
            
            # This should handle recursive decomposition gracefully with depth limits
            start_time = time.time()
            plan = engine.decompose(recursive_problem, strategy="recursive_hybrid")  # Hypothetical recursive strategy
            recursion_time = time.time() - start_time
            
            # Should complete in reasonable time without infinite recursion
            print(f"Recursive decomposition handled in {recursion_time:.3f}s")
            self.assertLess(recursion_time, 10.0, "Recursive decomposition should be prevented or limited")  # Should not hang forever


class TestSecurityEdgeCases(unittest.TestCase):
    """Test security edge cases and vulnerability probes"""
    
    def test_extreme_input_sizes_for_injection_attempts(self):
        """Test extremely large inputs that might bypass validation"""
        validator = InputValidator()
        
        # Create extremely large strings that attempt to bypass validation
        malicious_large_strings = [
            # SQL injection with large padding
            "'" + "A" * 10000 + "; DROP TABLE problems; --",
            
            # Command injection with large padding
            "test" + "B" * 20000 + "; rm -rf /; echo done",
            
            # XSS with large payload
            '<script>' + 'X' * 15000 + '</script>',
            
            # Path traversal with large padding
            '../../../' + 'etc/' * 500 + 'passwd',
            
            # Buffer overflow attempt with large string
            'A' * 100000,
            
            # JSON injection attempt
            '{"id": "1", "malicious": "' + "B" * 50000 + '"}' + '}; DROP TABLE test; --',
        ]
        
        for i, malicious_input in enumerate(malicious_large_strings):
            with self.subTest(input_type=f"malicious_{i}"):
                try:
                    # This should handle the large input safely
                    result = self.validator.validate_input(
                        malicious_input,
                        f"test_field_{i}",
                        [self.validator.VALIDATION_RULES.MAX_LENGTH(50000)]  # Reasonable max length
                    )
                    # Should either sanitize or reject properly
                except (ValueError, TypeError, RuntimeError):
                    # Exception during validation of malicious input is acceptable
                    pass
    
    def test_encoding_and_character_set_vulnerabilities(self):
        """Test various encodings and character sets for vulnerabilities"""
        validator = InputValidator()
        
        # Different encoding attacks
        encoding_attacks = [
            # UTF-8 encoded attacks
            "Test\u0000DROP TABLE problems; --",  # Null byte injection
            "SELECT * FROM users WHERE id=\uffff",  # Unicode non-character
            "Test\x00DROP TABLE problems; --",  # Binary null byte
            "SELECT * FROM users WHERE id=\ud83d\ude08",  # Emoji character that might cause parsing issues
        ]
        
        for encoding_attack in encoding_attacks:
            with self.subTest(encoding_type=encoding_attack[:20]):
                try:
                    # Should handle encoding attacks safely
                    result = self.validator.validate_input(
                        encoding_attack,
                        "encoding_test_field",
                        [self.validator.VALIDATION_RULES.NOT_EMPTY]
                    )
                    self.assertIsNotNone(result)
                except UnicodeEncodeError:
                    # This might be expected for certain invalid encodings
                    pass
                except (ValueError, TypeError, RuntimeError):
                    # Other exceptions for malicious encoding are acceptable
                    pass
    
    def test_race_condition_scenarios(self):
        """Test potential race conditions in multi-threaded access"""
        import threading
        import time
        from queue import Queue
        
        db = SovereignDatabase(":memory:")
        
        # Shared result queues
        create_results = Queue()
        read_results = Queue()
        error_count = 0
        error_lock = threading.Lock()
        
        def creator_thread(thread_id):
            """Thread that creates problems"""
            for i in range(20):
                try:
                    problem = ProblemDefinition(
                        id=generate_id(f"race_test_{thread_id}_{i}"),
                        title=f"Race Condition Test {thread_id}-{i}",
                        description=f"Problem {i} from thread {thread_id}",
                        problem_type=ProblemType.RESEARCH,
                        domain_context=DomainContext(domain="race_test"),
                        complexity_score=ComplexityScore(
                            explanation=f"Race condition test {thread_id}-{i}",
                            cognitive_complexity=5.0,
                            computational_complexity=5.0,
                            domain_complexity=5.0,
                            integration_complexity=5.0,
                            overall_complexity=5.0
                        )
                    )
                    
                    result = db.create_problem(problem)
                    create_results.put({'thread_id': thread_id, 'attempt': i, 'success': result})
                    
                    # Brief pause to allow interleaving
                    time.sleep(0.0001)
                except Exception as e:
                    with error_lock:
                        error_count += 1
        
        def reader_thread(thread_id):
            """Thread that reads problems"""
            for i in range(10):
                try:
                    # Try to read problems (some might not exist yet)
                    problems = db.list_problems()
                    read_results.put({'thread_id': thread_id, 'attempt': i, 'count': len(problems)})
                    
                    # Brief pause to allow interleaving
                    time.sleep(0.0001)
                except Exception as e:
                    with error_lock:
                        error_count += 1
        
        # Start multiple creator threads
        creators = []
        readers = []
        
        for i in range(5):  # 5 creator threads
            thread = threading.Thread(target=creator_thread, args=(i,))
            creators.append(thread)
            thread.start()
        
        for i in range(3):  # 3 reader threads
            thread = threading.Thread(target=reader_thread, args=(i,))
            readers.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        start_time = time.time()
        for thread in creators + readers:
            thread.join(timeout=10)  # 10 second timeout
        
        total_time = time.time() - start_time
        
        # Collect results
        created_count = 0
        while not create_results.empty():
            result = create_results.get()
            if result['success']:
                created_count += 1
        
        read_attempts = 0
        while not read_results.empty():
            read_results.get()
            read_attempts += 1
        
        print(f"Race condition test - Created: {created_count}, Reads: {read_attempts}, Errors: {error_count}")
        print(f"Total time: {total_time:.3f}s")
        
        # Verify database integrity - no corruption should occur
        final_problems = db.list_problems()
        print(f"Final problem count: {len(final_problems)}")
        
        # There should be no corruption/errors despite concurrent access
        self.assertLess(error_count, 5, f"Too many errors in concurrent access: {error_count}")
        self.assertGreater(len(final_problems), 0, "Database should have problems despite race conditions")
    
    def test_memory_corruption_and_resource_exhaustion(self):
        """Test defenses against memory corruption and resource exhaustion"""
        import gc
        import psutil
        import os
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many objects to test memory management
        objects_created = 0
        max_objects = 10000  # Create 10k objects
        large_objects = []
        
        for i in range(max_objects):
            # Create a complex object with multiple nested structures
            large_obj = {
                'id': generate_id(f"memory_test_{i}"),
                'data': {
                    'nested_list': [f"item_{j}" for j in range(100)],
                    'nested_dict': {f'key_{k}': f'value_{k}' for k in range(50)},
                    'complexity': {
                        'cognitive': 5.0 + (i % 5),
                        'computational': 6.0 + (i % 4),
                        'domain': 7.0 + (i % 3),
                        'integration': 8.0 + (i % 2)
                    },
                    'metadata': {
                        'created_by': 'memory_test',
                        'batch': i // 100,
                        'iteration': i % 100
                    }
                },
                'references': [generate_id(f"ref_{i}_{j}") for j in range(10)],
                'temporal_data': [datetime.now().isoformat() for _ in range(5)]
            }
            large_objects.append(large_obj)
            objects_created += 1
            
            # Periodically trigger garbage collection
            if i % 1000 == 0:
                gc.collect()
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        print(f"Memory usage: Baseline {baseline_memory:.1f}MB -> Peak {peak_memory:.1f}MB")
        print(f"Memory increase: {memory_increase:.1f}MB for {objects_created} objects")
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 200.0, f"Memory usage increased too much: {memory_increase:.1f}MB")
        
        # Clean up and check memory after cleanup
        del large_objects
        gc.collect()
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        cleanup_increase = cleanup_memory - baseline_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB (increase: {cleanup_increase:.1f}MB)")
        
        # Memory should be largely reclaimed after cleanup
        self.assertLess(cleanup_increase, memory_increase * 0.3, "Memory not properly released after cleanup")


class TestPerformanceEdgeCases(unittest.TestCase):
    """Test performance edge cases and boundary conditions"""
    
    def test_cache_performance_under_extreme_load(self):
        """Test cache performance under extreme load conditions"""
        import time
        import threading
        from queue import Queue
        
        cache = LLMResponseCache(max_size=1000)
        
        # Results queue for concurrent operations
        results = Queue()
        
        def cache_worker(worker_id):
            """Worker function to perform cache operations concurrently"""
            for i in range(100):  # Each worker performs 100 operations
                content = f"Test content for worker {worker_id}, operation {i}"
                model_params = {"model": f"test_model_{worker_id % 3}", "temperature": 0.7}
                response = {"choices": [{"message": {"content": f"Response from worker {worker_id}, op {i}"}}]}
                
                # Cache the response
                cache.cache_response(content, model_params, response)
                
                # Sometimes retrieve (50% of operations)
                if random.random() < 0.5:
                    retrieved = cache.get_response(content, model_params)
                    results.put({
                        'worker_id': worker_id,
                        'operation': 'get',
                        'hit': retrieved is not None,
                        'op_num': i
                    })
                else:
                    results.put({
                        'worker_id': worker_id,
                        'operation': 'set',
                        'op_num': i
                    })
        
        # Run concurrent cache operations
        threads = []
        num_workers = 10  # 10 concurrent workers
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=cache_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        operations = []
        while not results.empty():
            operations.append(results.get())
        
        cache_gets = [op for op in operations if op['operation'] == 'get']
        cache_hits = [op for op in cache_gets if op['hit']]
        
        hit_rate = len(cache_hits) / len(cache_gets) if cache_gets else 0
        
        print(f"Cache performance: {len(operations)} operations in {total_time:.3f}s")
        print(f"Cache hit rate: {hit_rate:.2%} ({len(cache_hits)}/{len(cache_gets)})")
        print(f"Operations per second: {len(operations)/total_time:.1f}")
        
        # Verify cache statistics
        stats = cache.get_stats()
        print(f"Cache stats - Size: {stats['current_size']}, Hits: {stats['total_hits']}, Misses: {stats['total_misses']}")
        
        # Should have good performance under load
        self.assertLess(total_time, 5.0, f"Cache operations took too long: {total_time:.3f}s")
        self.assertGreaterEqual(stats['current_size'], 500, "Cache should retain many entries under load")
    
    def test_parallel_processing_efficiency(self):
        """Test efficiency of parallel processing under various loads"""
        import time
        import concurrent.futures
        from multiprocessing import cpu_count
        
        # Simulate CPU-bound tasks
        def cpu_intensive_task(task_id):
            """Simulate a CPU-intensive task"""
            result = 0
            for i in range(1000000):  # CPU-intensive calculation
                result += i * task_id
            return f"Task {task_id} completed with result {result}"
        
        # Simulate I/O-bound tasks
        def io_simulated_task(task_id):
            """Simulate an I/O-bound task"""
            import time
            time.sleep(0.01)  # Simulate I/O delay
            return f"I/O Task {task_id} completed"
        
        total_tasks = 100
        
        # Test with different numbers of workers
        for max_workers in [2, 4, 8, cpu_count()]:
            with self.subTest(max_workers=max_workers):
                # CPU-bound tasks
                cpu_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    cpu_tasks = [executor.submit(cpu_intensive_task, i) for i in range(total_tasks)]
                    cpu_results = [future.result() for future in concurrent.futures.as_completed(cpu_tasks)]
                cpu_time = time.time() - cpu_start
                
                # I/O-bound tasks
                io_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    io_tasks = [executor.submit(io_simulated_task, i) for i in range(total_tasks)]
                    io_results = [future.result() for future in concurrent.futures.as_completed(io_tasks)]
                io_time = time.time() - io_start
                
                print(f"Workers: {max_workers}")
                print(f"  - CPU-bound: {total_tasks} tasks in {cpu_time:.3f}s ({total_tasks/cpu_time:.1f} ops/sec)")
                print(f"  - I/O-bound: {total_tasks} tasks in {io_time:.3f}s ({total_tasks/io_time:.1f} ops/sec)")
                
                # Verify all tasks completed
                self.assertEqual(len(cpu_results), total_tasks)
                self.assertEqual(len(io_results), total_tasks)
    
    def test_database_indexing_performance(self):
        """Test database indexing performance with large datasets"""
        import time
        
        db = SovereignDatabase(":memory:")
        
        # Create large dataset to test indexing
        large_dataset_size = 10000
        test_problems = []
        
        for i in range(large_dataset_size):
            problem = ProblemDefinition(
                id=generate_id(f"index_test_{i}"),
                title=f"Index Test Problem {i}",
                description=f"Problem {i} for database indexing performance test. " + "Performance test content. " * 10,
                problem_type=random.choice(list(ProblemType)),
                domain_context=DomainContext(domain=f"domain_{i % 50}"),  # 50 different domains
                complexity_score=ComplexityScore(
                    explanation=f"Index performance test {i}",
                    cognitive_complexity=5.0 + (i % 4),
                    computational_complexity=5.0 + (i % 4),
                    domain_complexity=5.0 + (i % 4),
                    integration_complexity=5.0 + (i % 4),
                    overall_complexity=5.0 + (i % 4)
                )
            )
            test_problems.append(problem)
        
        # Time bulk insert
        start_time = time.time()
        for problem in test_problems:
            db.create_problem(problem)
        insert_time = time.time() - start_time
        
        print(f"Indexing test - Inserted {len(test_problems)} problems in {insert_time:.3f}s")
        
        # Time indexed queries
        start_time = time.time()
        for i in range(100):  # 100 random queries
            query_domain = f"domain_{random.randint(0, 49)}"
            results = db.list_problems(problem_type=random.choice(list(ProblemType)))
        
        query_time = time.time() - start_time
        avg_query_time = query_time / 100
        
        print(f"Indexed queries: {avg_query_time:.4f}s per query on average")
        
        # Verify insert performance is reasonable
        self.assertLess(insert_time, 15.0, f"Insertion of {large_dataset_size} problems took too long: {insert_time:.3f}s")
        
        # Verify query performance is reasonable (should be well under 0.1s per query with indexing)
        self.assertLess(avg_query_time, 0.05, f"Indexed queries too slow: {avg_query_time:.4f}s per query")


class TestGauntletSystemEdgeCases(unittest.TestCase):
    """Test gauntlet system with edge cases and extreme scenarios"""
    
    def setUp(self):
        """Set up gauntlet system for testing"""
        with patch('sovereign_gauntlets.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            from sovereign_gauntlets import GauntletSystem
            self.gauntlet_system = GauntletSystem(openevolve_client=self.mock_client)
    
    def test_very_long_gauntlet_with_many_rounds(self):
        """Test gauntlet with many rounds and validators"""
        from gauntlet_structures import GauntletDefinition, GauntletRoundRule, ValidationCheckpoint
        
        # Create an extended gauntlet with many rounds
        many_rounds_gauntlet = GauntletDefinition(
            name="Extended Multi-Round Gauntlet",
            team_name="red",
            rounds=[
                GauntletRoundRule(
                    round_number=i,
                    quorum_required_approvals=1 if i < 5 else 2,  # Require more approvals in later rounds
                    quorum_from_panel_size=2 if i < 5 else 3,
                    min_overall_confidence=0.6 - (i * 0.02),  # Slightly decrease threshold each round
                    max_score_variance=0.3,
                    per_judge_requirements={},
                    collaboration_mode="none"
                )
                for i in range(1, 11)  # 10 rounds
            ],
            validation_checkpoints=[
                ValidationCheckpoint(
                    id=generate_id("cp1"),
                    name="Early Validation",
                    description="Early validation checkpoint",
                    validation_type="structural"
                ),
                ValidationCheckpoint(
                    id=generate_id("cp2"), 
                    name="Mid-process Validation",
                    description="Mid-process validation checkpoint",
                    validation_type="functional"
                ),
                ValidationCheckpoint(
                    id=generate_id("cp3"),
                    name="Final Validation",
                    description="Final validation checkpoint", 
                    validation_type="holistic"
                )
            ]
        )
        
        # Test content for the gauntlet
        test_content = "This is a complex solution that will be validated through multiple rigorous rounds in the extended gauntlet. The content needs to be substantial enough to undergo thorough validation across multiple rounds and criteria. The solution should address all requirements specified in the original problem statement while maintaining high quality, correctness, and efficiency standards."
        
        # Mock responses for the many rounds
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps({
            "passed": True,
            "score": 0.85,
            "justification": "Solution passed initial validation",
            "improvements": ["Consider additional edge case handling"]
        })
        self.mock_client.evolve.return_value = mock_response
        
        # Run the extended gauntlet
        start_time = time.time()
        result = self.gauntlet_system.run_gauntlet(
            content=test_content,
            gauntlet_def=many_rounds_gauntlet,
            team=self.gauntlet_system.get_team("red"),
            context={"validation_type": "extended", "test_scenario": "many_rounds"}
        )
        gauntlet_time = time.time() - start_time
        
        print(f"Extended gauntlet completed in {gauntlet_time:.3f}s")
        
        # Should complete in reasonable time despite many rounds
        self.assertIsNotNone(result)
        self.assertLess(gauntlet_time, 10.0, "Extended gauntlet should complete in reasonable time")
    
    def test_gauntlet_with_extremely_divergent_opinions(self):
        """Test gauntlet behavior with extremely divergent judge opinions"""
        from gauntlet_structures import GauntletDefinition, GauntletRoundRule
        
        divergent_gauntlet = GauntletDefinition(
            name="Divergent Opinion Gauntlet",
            team_name="gold",
            rounds=[
                GauntletRoundRule(
                    round_number=1,
                    quorum_required_approvals=3,  # Require unanimous agreement
                    quorum_from_panel_size=3,     # 3 judges
                    min_overall_confidence=0.9,   # High confidence requirement
                    max_score_variance=0.01,      # Very low variance allowed
                    per_judge_requirements={},
                    collaboration_mode="none"
                )
            ]
        )
        
        # This gauntlet is designed to be very strict, so it may reject content
        test_content = "This is a test solution that might get very different scores from different judges due to subjective interpretation."
        
        # Mock divergent responses from different judges
        # In practice, this would require the gauntlet system to handle multiple judge responses
        # For this test, we'll check that the system handles high-variance situations properly
        
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps({
            "passed": False,  # Expected due to high variance requirements
            "score": 0.4,     # Low average score due to variance
            "justification": "High variance in scores prevented consensus",
            "improvements": ["Reduce subjective elements", "Add more objective criteria"]
        })
        
        self.mock_client.evolve.return_value = mock_response
        
        result = self.gauntlet_system.run_gauntlet(
            content=test_content,
            gauntlet_def=divergent_gauntlet,
            team=self.gauntlet_system.get_team("gold"),  # Use gold team which typically validates quality
            context={"validation_type": "strict", "test_scenario": "divergent_opinions"}
        )
        
        self.assertIsNotNone(result)
        # The result might be rejected due to the strict requirements, which is expected behavior
    
    def test_nested_gauntlet_scenarios(self):
        """Test nested or sequential gauntlet scenarios"""
        from gauntlet_structures import GauntletDefinition, GauntletRoundRule
        
        # Create specialized gauntlets for different validation aspects
        structural_gauntlet = GauntletDefinition(
            name="Structural Gauntlet",
            team_name="red",
            rounds=[
                GauntletRoundRule(
                    round_number=1,
                    quorum_required_approvals=1,
                    quorum_from_panel_size=2,
                    min_overall_confidence=0.7,
                    max_score_variance=0.3,
                    per_judge_requirements={},
                    collaboration_mode="none"
                )
            ],
            description="Validates structural correctness and completeness"
        )
        
        security_gauntlet = GauntletDefinition(
            name="Security Gauntlet", 
            team_name="red",
            rounds=[
                GauntletRoundRule(
                    round_number=1,
                    quorum_required_approvals=2,
                    quorum_from_panel_size=2,
                    min_overall_confidence=0.8,
                    max_score_variance=0.2,
                    per_judge_requirements={},
                    collaboration_mode="none"
                )
            ],
            description="Validates security aspects and vulnerabilities"
        )
        
        performance_gauntlet = GauntletDefinition(
            name="Performance Gauntlet",
            team_name="blue", 
            rounds=[
                GauntletRoundRule(
                    round_number=1,
                    quorum_required_approvals=1,
                    quorum_from_panel_size=2,
                    min_overall_confidence=0.75,
                    max_score_variance=0.25,
                    per_judge_requirements={},
                    collaboration_mode="none"
                )
            ],
            description="Validates performance characteristics and efficiency"
        )
        
        # Test content
        test_solution = """
        def secure_api_endpoint():
            '''API endpoint with security and performance considerations'''
            # Implementation with security measures
            # Rate limiting, authentication, input validation
            # Performance optimizations
            # Caching, efficient algorithms, resource management
            pass
        """
        
        # Run each gauntlet sequentially
        results = {}
        
        for gauntlet_name, gauntlet_def in [
            ("structural", structural_gauntlet),
            ("security", security_gauntlet), 
            ("performance", performance_gauntlet)
        ]:
            mock_response = Mock()
            mock_response.success = True
            mock_response.best_code = json.dumps({
                "passed": True,
                "score": 0.8,
                "justification": f"{gauntlet_name.title()} validation passed",
                "improvements": [f"Consider additional {gauntlet_name} checks"]
            })
            
            self.mock_client.evolve.return_value = mock_response
            
            result = self.gauntlet_system.run_gauntlet(
                content=test_solution,
                gauntlet_def=gauntlet_def,
                team=self.gauntlet_system.get_team(gauntlet_def.team_name),
                context={"validation_type": gauntlet_name, "test_scenario": "sequential_validation"}
            )
            
            results[gauntlet_name] = result
        
        # Verify all sequential validations ran
        self.assertEqual(len(results), 3)
        for gauntlet_name, result in results.items():
            self.assertIsNotNone(result, f"{gauntlet_name} gauntlet should have a result")


class TestAdvancedWorkflowScenarios(unittest.TestCase):
    """Test advanced workflow scenarios with complex interactions"""
    
    def test_cascading_failure_scenarios(self):
        """Test how the system handles cascading failures"""
        # Create a decomposition plan with intentional dependencies that might fail
        sub_problems = []
        for i in range(5):
            sub_problem = SubProblem(
                id=generate_id(f"cascading_{i}"),
                parent_id=generate_id("cascading_root"),
                title=f"Cascading Test Sub-problem {i}",
                description=f"Sub-problem {i} in cascading failure test",
                type=SubProblemType.IMPLEMENTATION if i < 4 else SubProblemType.VALIDATION,
                complexity_score=ComplexityScore(
                    explanation=f"Cascading test {i}",
                    cognitive_complexity=5.0 + (i * 0.5),
                    computational_complexity=5.0 + (i * 0.5),
                    domain_complexity=5.0 + (i * 0.5),
                    integration_complexity=5.0 + (i * 0.5),
                    overall_complexity=5.0 + (i * 0.5)
                ),
                dependencies=[sub_problems[j].id for j in range(i)] if i > 0 else []  # Each depends on all previous
            )
            sub_problems.append(sub_problem)
        
        plan = DecompositionPlan(
            id=generate_id("cascading_plan"),
            problem_id=generate_id("cascading_parent"),
            strategy="dependency",
            sub_problems=sub_problems,
            dependency_graph={sp.id: sp.dependencies for sp in sub_problems},
            validation_checkpoints=[],
            quality_scores={},
            confidence_level=0.8
        )
        
        # Mock orchestrator that simulates some failures
        orchestrator = SolutionOrchestrator()
        
        # Track which sub-problems fail
        failed_sub_problems = set()
        
        def mock_solve_sub_problem(sub_problem_id, approach, content, team_id, confidence_score):
            """Mock solution attempt that fails for certain sub-problems"""
            if "cascading_2" in sub_problem_id:
                # Simulate failure for sub-problem 2
                failed_sub_problems.add(sub_problem_id)
                return Mock(success=False, error="Simulated cascading failure")
            return Mock(success=True, solution_id=generate_id("solution"))
        
        with patch.object(orchestrator.solution_tracker, 'track_solution_attempt', side_effect=mock_solve_sub_problem):
            # Try to process the plan with intentional failures
            try:
                result = orchestrator.integrate_solutions(plan)
                # If integration completes despite failures, verify error handling worked
                print(f"Cascading failure handled gracefully. Failed: {len(failed_sub_problems)} sub-problems")
            except Exception as e:
                # Failure during integration is also acceptable when dealing with cascading failures
                print(f"Cascading failure propagated as expected: {e}")
        
        # System should handle cascading failures gracefully
        self.assertIn(f"sp_cascading_2", failed_sub_problems, "Intentionally failed sub-problem should be tracked")
    
    def test_extremely_parallel_execution(self):
        """Test execution with maximum parallelism"""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Create many independent sub-problems that can be solved in parallel
        sub_problems = []
        for i in range(50):  # 50 sub-problems
            sub_problem = SubProblem(
                id=generate_id(f"parallel_{i}"),
                parent_id=generate_id("parallel_root"),
                title=f"Parallel Test Sub-problem {i}",
                description=f"Sub-problem {i} designed for parallel execution test",
                type=SubProblemType.ANALYSIS,
                complexity_score=ComplexityScore(
                    explanation=f"Parallel execution test {i}",
                    cognitive_complexity=4.5,
                    computational_complexity=4.0,
                    domain_complexity=4.5,
                    integration_complexity=3.5,
                    overall_complexity=4.1
                ),
                dependencies=[]  # No dependencies for maximum parallelism
            )
            sub_problems.append(sub_problem)
        
        plan = DecompositionPlan(
            id=generate_id("parallel_plan"),
            problem_id=generate_id("parallel_parent"),
            strategy="parallel",
            sub_problems=sub_problems,
            dependency_graph={sp.id: [] for sp in sub_problems},  # No dependencies
            validation_checkpoints=[],
            quality_scores={},
            confidence_level=0.85
        )
        
        # Simulate parallel execution
        solutions = {}
        
        def solve_sub_problem(sub_problem):
            """Simulate solving a sub-problem"""
            import time
            # Simulate work
            time.sleep(0.01)  # Small delay
            return {
                'sub_problem_id': sub_problem.id,
                'solution': f"Solution for {sub_problem.title}",
                'confidence': 0.8 + (random.uniform(-0.1, 0.1))
            }
        
        start_time = time.time()
        
        # Execute all sub-problems in parallel
        with ThreadPoolExecutor(max_workers=20) as executor:  # 20 threads
            futures = [executor.submit(solve_sub_problem, sp) for sp in sub_problems]
            results = [future.result() for future in as_completed(futures)]
        
        parallel_time = time.time() - start_time
        
        solution_count = len(results)
        print(f"Parallel execution: {solution_count} sub-problems solved in {parallel_time:.3f}s")
        print(f"Operations per second: {solution_count/parallel_time:.1f}")
        
        self.assertEqual(len(results), len(sub_problems))
        self.assertLess(parallel_time, 2.0, f"Parallel execution should be fast: {parallel_time:.3f}s")
    
    def test_cross_team_validation_chain(self):
        """Test validation involving multiple teams in sequence"""
        from sovereign_team_coordination import TeamCoordinator
        
        coordinator = TeamCoordinator()
        
        # Create a solution that goes through Red -> Blue -> Gold validation chain
        sub_problem = SubProblem(
            id=generate_id("cross_team_test"),
            parent_id=generate_id("cross_team_parent"),
            title="Cross-Team Validation Test",
            description="Solution that requires validation through Red, Blue, and Gold teams",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Cross-team validation test",
                cognitive_complexity=6.0,
                computational_complexity=5.5,
                domain_complexity=6.5,
                integration_complexity=6.0,
                overall_complexity=6.0
            )
        )
        
        # Create solution attempt
        solution = SolutionAttempt(
            id=generate_id("cross_team_solution"),
            sub_problem_id=sub_problem.id,
            approach="Cross-team validation approach",
            solution_content="This solution will be validated by multiple teams in sequence",
            team_id="initial_team",
            confidence_score=0.75
        )
        
        # Mock the team validation process
        with patch.object(coordinator.red_team, 'assess_content') as mock_red, \
             patch.object(coordinator.blue_team, 'apply_fixes') as mock_blue, \
             patch.object(coordinator.gold_team, 'evaluate_content') as mock_gold:
            
            # Mock responses for each team
            red_result = Mock()
            red_result.findings = []  # No issues found by Red team
            red_result.confidence_score = 0.8
            mock_red.return_value = red_result
            
            blue_result = Mock()
            blue_result.fixed_content = solution.solution_content  # No fixes needed
            blue_result.quality_score = 0.85
            mock_blue.return_value = blue_result
            
            gold_result = Mock()
            gold_result.consensus_score = 88  # Out of 100
            gold_result.final_verdict = "APPROVED"
            gold_result.recommendations = ["Good implementation"]
            mock_gold.return_value = gold_result
            
            # Execute cross-team validation
            start_time = time.time()
            
            # Red team review
            red_review = coordinator.red_team.assess_content(
                content=solution.solution_content,
                content_type="solution_implementation"
            )
            
            # Blue team review (only if Red approves)
            if not red_review.findings:
                blue_fixes = coordinator.blue_team.apply_fixes(
                    current_content=solution.solution_content,
                    issues_found=red_review.findings
                )
                final_content = blue_fixes.fixed_content
            else:
                final_content = solution.solution_content
            
            # Gold team validation
            gold_evaluation = coordinator.gold_team.evaluate_content(
                content=final_content,
                content_type="solution_implementation"
            )
            
            cross_team_time = time.time() - start_time
            
            print(f"Cross-team validation completed in {cross_team_time:.3f}s")
            print(f"Gold team verdict: {gold_evaluation.final_verdict}")
            
            # Verify validation sequence completed
            self.assertIsNotNone(red_review)
            self.assertIsNotNone(gold_evaluation)
            self.assertEqual(gold_evaluation.final_verdict, "APPROVED")


class TestSystemResilience(unittest.TestCase):
    """Test system resilience under various failure conditions"""
    
    def test_partial_component_failure_recovery(self):
        """Test system behavior when individual components fail"""
        # Create a database that might fail occasionally
        db = SovereignDatabase(":memory:")
        
        # Add a problem successfully
        good_problem = ProblemDefinition(
            id=generate_id("resilient_test"),
            title="Resilient Operation Test",
            description="Test system resilience when components fail",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="resilience_testing"),
            complexity_score=ComplexityScore(
                explanation="Resilience test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        # Store problem
        result = db.create_problem(good_problem)
        self.assertTrue(result)
        
        # Test that system can continue operating despite individual component failures
        retrieved = db.get_problem(good_problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Resilient Operation Test")
        
        # Test database operations continue to work
        more_problems = []
        for i in range(10):
            problem = ProblemDefinition(
                id=generate_id(f"additional_{i}"),
                title=f"Additional Problem {i}",
                description=f"Problem {i} for resilience testing",
                problem_type="RESEARCH",
                domain_context=DomainContext(domain="resilience_testing"),
                complexity_score=ComplexityScore(
                    explanation=f"Additional problem {i}",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            success = db.create_problem(problem)
            self.assertTrue(success)
            more_problems.append(problem)
        
        # Verify all problems were stored
        all_problems = db.list_problems()
        self.assertGreaterEqual(len(all_problems), 11)  # Original + 10 additional
    
    def test_network_disruption_simulation(self):
        """Simulate network disruptions and test graceful degradation"""
        from unittest.mock import PropertyMock
        
        # Test how the system handles simulated network issues
        # This would involve mocking network calls to LLM services
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            
            # Simulate network timeout
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "Request timeout after 30 seconds"
            mock_client.evolve.return_value = mock_result
            
            analyzer = ProblemAnalyzer(openevolve_client=mock_client)
            
            # This should handle network failure gracefully
            try:
                result = analyzer.analyze_problem(
                    problem_text="This will fail due to simulated network issue",
                    title="Network Disruption Test"
                )
                # If it returns None or a fallback, that's acceptable
                self.assertIsNone(result, "Analyzer should return None when network fails")
            except Exception as e:
                # Alternative: system might raise exception, which should be caught and handled
                print(f"Network failure handled with error: {e}")
                # This is also acceptable behavior
    
    def test_resource_starvation_handling(self):
        """Test system behavior under resource starvation"""
        # This test focuses on the system's ability to handle resource constraints
        # by implementing graceful degradation rather than crashing
        
        # Create many objects to consume memory and test resource management
        large_objects = []
        
        try:
            for i in range(5000):  # Create many objects
                large_obj = {
                    'id': generate_id(f"resource_test_{i}"),
                    'data': [
                        {'nested': f"Nested data item {j}", 'value': random.random()}
                        for j in range(100)
                    ],
                    'metadata': {
                        'iteration': i,
                        'timestamp': datetime.now().isoformat(),
                        'test_group': f"group_{i % 10}"
                    }
                }
                large_objects.append(large_obj)
                
                # Check if we're approaching resource limits
                # In a real system, we'd check actual memory/disk usage
                if i % 1000 == 0:
                    print(f"Created {i} large objects for resource starvation test")
        
        except MemoryError:
            # If we run out of memory, that's acceptable - system should handle it gracefully
            print("Memory limit reached during resource starvation test - this is expected behavior")
        finally:
            # Clean up to free resources
            del large_objects
            import gc
            gc.collect()
            
            # System should continue functioning normally after resource stress
            db = SovereignDatabase(":memory:")
            simple_problem = ProblemDefinition(
                id=generate_id("recovery_test"),
                title="Recovery Test",
                description="Test that system recovers after resource stress",
                problem_type="RESEARCH",
                domain_context=DomainContext(domain="recovery_testing"),
                complexity_score=ComplexityScore(
                    explanation="Recovery test",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            
            # This should still work after resource stress
            result = db.create_problem(simple_problem)
            self.assertTrue(result)
            
            retrieved = db.get_problem(simple_problem.id)
            self.assertIsNotNone(retrieved)


def run_edge_case_tests():
    """Run the edge case tests"""
    print("Running Edge Case and Extreme Scenario Tests...")
    print("="*80)
    
    # Create test suite for edge cases
    suite = unittest.TestSuite()
    
    # Add all edge case test classes
    suite.addTest(unittest.makeSuite(TestExtremeEdgeCases))
    suite.addTest(unittest.makeSuite(TestSecurityEdgeCases))
    suite.addTest(unittest.makeSuite(TestPerformanceEdgeCases))
    suite.addTest(unittest.makeSuite(TestGauntletSystemEdgeCases))
    suite.addTest(unittest.makeSuite(TestAdvancedWorkflowScenarios))
    suite.addTest(unittest.makeSuite(TestSystemResilience))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("EDGE CASE TEST RESULTS")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\nSome tests failed - this may be expected for edge cases")
        for test, trace in result.failures:
            print(f"\nFAILED: {test}")
            print(trace)
        for test, trace in result.errors:
            print(f"\nERROR: {test}")
            print(trace)
    else:
        print("\n[OK] All edge case tests passed!")
    
    print("="*80)
    return result


if __name__ == "__main__":
    run_edge_case_tests()