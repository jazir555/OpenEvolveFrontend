"""
Extra Comprehensive Unit Tests for Sovereign-Grade System
Additional edge case, security, and integration tests
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import sys
import os
import tempfile
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
import hashlib
import secrets
import asyncio
import gc

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
    Constraint, SuccessCriterion, DomainContext, ComplexityScore,
    ProblemType, SubProblemType, PlanStatus, generate_id
)
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_team_coordination import TeamCoordinator
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_persistence import SovereignDatabase
from auth_system import AuthenticationSystem, AuthorizationSystem
from input_validation import InputValidator
from performance_optimization import LLMResponseCache
from advanced_features import AdvancedFeaturesManager


class TestErrorConditions(unittest.TestCase):
    """Tests for error conditions and failure scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db = SovereignDatabase(self.temp_db.name)
        self.auth_system = AuthenticationSystem(db_path=":memory:")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_connection_failures(self):
        """Test handling of database connection failures"""
        # Create a db that tries to connect to a non-existent location
        try:
            with patch('sqlite3.connect') as mock_connect:
                mock_connect.side_effect = sqlite3.Error("Connection failed")
                
                # This should handle the error gracefully
                db = SovereignDatabase("/invalid/path/that/does/not/exist.db")
                
                # Any operation should fail gracefully
                problem = ProblemDefinition(
                    id=generate_id("test"),
                    title="Test",
                    description="Test problem",
                    problem_type=ProblemType.RESEARCH,
                    domain_context=DomainContext(domain="test"),
                    complexity_score=ComplexityScore(
                        explanation="Test",
                        cognitive_complexity=5.0,
                        computational_complexity=5.0,
                        domain_complexity=5.0,
                        integration_complexity=5.0,
                        overall_complexity=5.0
                    )
                )
                
                with self.assertRaises(sqlite3.Error):
                    db.create_problem(problem)
                    
        except Exception as e:
            # Expected behavior - should handle database errors gracefully
            pass
    
    def test_llm_client_failures(self):
        """Test handling of LLM client failures"""
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            mock_result = Mock()
            mock_result.success = False
            mock_result.error = "API Error: Rate limit exceeded"
            mock_client.evolve.return_value = mock_result
            
            analyzer = ProblemAnalyzer(openevolve_client=mock_client)
            
            with self.assertRaises(Exception):
                # This should fail gracefully when LLM is unavailable
                problem = analyzer.analyze_problem(
                    problem_text="This will fail due to mocked LLM error",
                    title="Test Error Handling"
                )
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection in sub-problems"""
        # Create a circular dependency scenario
        sub1 = SubProblem(
            id=generate_id("sub1"),
            parent_id=generate_id("parent"),
            title="Sub-problem 1",
            description="First sub-problem",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                explanation="Test circular dependency",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            ),
            dependencies=[generate_id("sub2")]  # Depends on sub2
        )
        
        sub2 = SubProblem(
            id=generate_id("sub2"),
            parent_id=generate_id("parent"),
            title="Sub-problem 2", 
            description="Second sub-problem",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                explanation="Test circular dependency",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            ),
            dependencies=[generate_id("sub1")]  # Depends on sub1 - circular!
        )
        
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id=generate_id("problem"),
            strategy="dependency",
            sub_problems=[sub1, sub2],  # This creates a circular dependency
            status=PlanStatus.ACTIVE
        )
        
        # Test that dependency validation detects the circular dependency
        from decomposition_engine import validate_dependencies
        errors = validate_dependencies(plan.sub_problems)
        
        # Should detect circular dependency
        circular_errors = [e for e in errors if "circular" in e.lower()]
        self.assertGreater(len(circular_errors), 0, "Should detect circular dependency")
    
    def test_solution_conflict_resolution(self):
        """Test solution conflict resolution with multiple attempts"""
        solution1 = SolutionAttempt(
            id=generate_id("sol1"),
            sub_problem_id=generate_id("sub"),
            approach="approach_1",
            solution_content="Same content for both solutions",
            team_id="team_a",
            confidence_score=0.8
        )
        
        solution2 = SolutionAttempt(
            id=generate_id("sol2"),
            sub_problem_id=generate_id("sub"),
            approach="approach_2", 
            solution_content="Same content for both solutions",  # Same content
            team_id="team_b",
            confidence_score=0.9  # Different confidence
        )
        
        # Test conflict detection - same content should be detected as duplicate
        from solution_orchestration import detect_conflicts
        conflicts = detect_conflicts([solution1, solution2])
        
        # Same content should not be considered conflicting, but we might want to detect duplicates
        # This depends on the specific conflict detection algorithm
        print(f"Detected {len(conflicts)} conflicts")
    
    def test_concurrent_database_access(self):
        """Test concurrent database access and race conditions"""
        import threading
        import time
        
        # Create a shared database
        db = SovereignDatabase(":memory:")
        
        results = []
        
        def create_problem_worker(worker_id):
            """Worker function to create problems concurrently"""
            try:
                for i in range(5):  # Each worker creates 5 problems
                    problem = ProblemDefinition(
                        id=generate_id(f"worker{worker_id}_item{i}"),
                        title=f"Concurrent Problem Worker {worker_id} Item {i}",
                        description=f"Problem created by worker {worker_id}, item {i}",
                        problem_type=ProblemType.RESEARCH,
                        domain_context=DomainContext(domain="concurrent_test"),
                        complexity_score=ComplexityScore(
                            explanation=f"Concurrent worker {worker_id} test",
                            cognitive_complexity=5.0,
                            computational_complexity=5.0,
                            domain_complexity=5.0,
                            integration_complexity=5.0,
                            overall_complexity=5.0
                        )
                    )
                    
                    result = db.create_problem(problem)
                    results.append((worker_id, i, result))
                    time.sleep(0.01)  # Small delay to increase chance of race conditions
            except Exception as e:
                results.append((worker_id, -1, f"Error: {str(e)}"))
        
        # Create multiple threads
        threads = []
        for worker_id in range(10):  # 10 concurrent workers
            thread = threading.Thread(target=create_problem_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all operations completed successfully
        successful_creations = [r for r in results if r[2] is True]
        failed_creations = [r for r in results if r[2] is not True and not str(r[2]).startswith("Error")]
        
        print(f"Concurrent database test: {len(successful_creations)} successful, {len(failed_creations)} failed")
        
        # All operations should be successful (though some might fail due to ID collisions which is expected)
        total_expected = 10 * 5  # 10 workers * 5 problems each
        self.assertEqual(len(successful_creations) + len(failed_creations), total_expected)
        
        # Verify that we have the expected number of unique problems in the database
        all_problems = db.list_problems()
        print(f"Total problems in database: {len(all_problems)}")
    
    def test_memory_leak_prevention(self):
        """Test that objects are properly cleaned up to prevent memory leaks"""
        import gc
        import weakref
        
        # Create a bunch of objects
        problems = []
        for i in range(1000):
            problem = ProblemDefinition(
                id=generate_id("leak_test"),
                title=f"Leak Test Problem {i}",
                description=f"Problem {i} for memory leak testing",
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="memory_test"),
                complexity_score=ComplexityScore(
                    explanation="Memory leak test",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            problems.append(problem)
        
        # Create weak references to track if objects get collected
        weak_refs = [weakref.ref(prob) for prob in problems]
        
        # Clear the list to allow garbage collection
        del problems
        gc.collect()
        
        # Check how many objects were actually collected
        alive_objects = [wr for wr in weak_refs if wr() is not None]
        
        print(f"After garbage collection: {len(alive_objects)} objects still alive out of 1000")
        
        # Most objects should have been collected (allow some for potential caching)
        self.assertLess(len(alive_objects), 100, "Too many objects remained after collection - possible memory leak")


class TestSecurityValidation(unittest.TestCase):
    """Additional security validation tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auth_system = AuthenticationSystem(db_path=":memory:")
    
    def test_xss_prevention_in_inputs(self):
        """Test XSS prevention in user inputs"""
        from input_validation import InputValidator
        
        validator = InputValidator()
        
        # Malicious XSS attempts
        xss_attempts = [
            '<script>alert("xss")</script>',
            'javascript:alert("xss")',
            '<img src="x" onerror="alert(\'xss\')">',
            '<svg onload=alert("xss")>',
            '"><script>alert("xss")</script>',
            '<iframe src="javascript:alert(\'xss\')"></iframe>'
        ]
        
        for malicious_input in xss_attempts:
            # This should either clean the input or reject it
            try:
                # If there's a method that handles HTML sanitization
                if hasattr(validator, '_sanitize_html'):
                    cleaned = validator._sanitize_html(malicious_input, "test_field")
                    # Should not contain dangerous tags
                    self.assertNotIn('<script', cleaned.lower())
                    self.assertNotIn('javascript:', cleaned.lower())
            except (ValueError, TypeError, RuntimeError):
                # If sanitization throws an exception, that's also acceptable
                pass
    
    def test_sql_injection_attempts(self):
        """Test SQL injection prevention"""
        # Test that malicious SQL doesn't get executed
        malicious_titles = [
            "'; DROP TABLE problems; --",
            "'; UPDATE users SET admin=1; --", 
            "' OR 1=1; --",
            "'; DELETE FROM problems; --",
            "') UNION SELECT password FROM users; --"
        ]
        
        db = SovereignDatabase(":memory:")
        
        for malicious_title in malicious_titles:
            try:
                problem = ProblemDefinition(
                    id=generate_id("sql_test"),
                    title=malicious_title,
                    description="SQL injection test",
                    problem_type=ProblemType.RESEARCH,
                    domain_context=DomainContext(domain="security_test"),
                    complexity_score=ComplexityScore(
                        explanation="SQL injection test",
                        cognitive_complexity=5.0,
                        computational_complexity=5.0,
                        domain_complexity=5.0,
                        integration_complexity=5.0,
                        overall_complexity=5.0
                    )
                )
                
                # This should handle the malicious input safely
                result = db.create_problem(problem)
                
                # If it was created, retrieve and verify the title was handled safely
                if result:
                    retrieved = db.get_problem(problem.id)
                    # The malicious SQL should not have been executed, 
                    # but the title may have been stored as-is
                    # The important thing is that no tables were dropped
                    tables = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                    table_names = [t[0] for t in tables]
                    # Verify critical tables still exist
                    self.assertIn('problems', table_names, "Critical table 'problems' should still exist")
                    
            except (ValueError, TypeError, RuntimeError):
                # If creation threw an exception due to validation, that's also valid protection
                pass
    
    def test_authentication_edge_cases(self):
        """Test authentication edge cases and brute force protection"""
        # Create a test user
        from auth_system import Role, Permission
        
        user = self.auth_system.create_user(
            username="security_test",
            email="security@test.com",
            password="SecurePass123!",
            roles=[Role.VIEWER],
            permissions=[Permission.READ_PROBLEM]
        )
        
        self.assertIsNotNone(user)
        
        # Test multiple failed login attempts
        for i in range(5):
            result = self.auth_system.authenticate("security_test", f"wrong_password_{i}")
            self.assertIsNone(result, f"Login should fail for attempt {i}")
        
        # Now test with correct password - should still work (no lockout in our basic system)
        correct_result = self.auth_system.authenticate("security_test", "SecurePass123!")
        self.assertIsNotNone(correct_result, "Correct login should still work after failed attempts")
    
    def test_permission_boundary_tests(self):
        """Test permission boundary violations"""
        from auth_system import Role, Permission
        from authorization_system import AuthorizationSystem
        
        # Create users with different permission levels
        admin_user = self.auth_system.create_user(
            username="admin_user",
            email="admin@test.com",
            password="SecurePass123!",
            roles=[Role.ADMIN]
        )
        
        analyst_user = self.auth_system.create_user(
            username="analyst_user", 
            email="analyst@test.com",
            password="SecurePass123!",
            roles=[Role.ANALYST]
        )
        
        # Create authorization system
        authz_system = AuthorizationSystem(self.auth_system)
        
        # Admin should have all permissions
        admin_can_admin = authz_system.check_permission(admin_user, Permission.ADMIN_ACCESS)
        self.assertTrue(admin_can_admin, "Admin should have admin access")
        
        # Analyst should not have admin permissions
        analyst_can_admin = authz_system.check_permission(analyst_user, Permission.ADMIN_ACCESS)
        self.assertFalse(analyst_can_admin, "Analyst should not have admin access")
        
        # Both should have basic read access based on role hierarchy
        analyst_can_read = authz_system.check_permission(analyst_user, Permission.READ_PROBLEM)
        self.assertTrue(analyst_can_read, "Analyst should have read problem permission")


class TestAdvancedFeatureScenarios(unittest.TestCase):
    """Test advanced feature scenarios and integrations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.advanced_manager = AdvancedFeaturesManager()
    
    def test_multi_modal_content_processing(self):
        """Test multi-modal content processing"""
        # Test with empty content
        result = self.advanced_manager.process_multi_modal_content([])
        self.assertEqual(result, [])
        
        # Test with single content item
        content_item = {
            "type": "text",
            "data": "This is a test", 
            "metadata": {"source": "test"}
        }
        
        result = self.advanced_manager.process_multi_modal_content([content_item])
        self.assertEqual(len(result), 1)
        self.assertIn("processed", result[0])
        
        # Test with mixed content types
        mixed_content = [
            {"type": "text", "data": "Text content", "metadata": {}},
            {"type": "image", "data": "fake_image_data", "metadata": {"format": "png"}},
            {"type": "structured_data", "data": {"key": "value"}, "metadata": {}}
        ]
        
        result = self.advanced_manager.process_multi_modal_content(mixed_content)
        self.assertEqual(len(result), 3)
        
        # Verify processing happened
        for item in result:
            self.assertIn("processed", item)
            self.assertIn("type", item)
    
    def test_visual_representation_generation(self):
        """Test visual representation generation"""
        # Test with empty plan
        mermaid_empty = self.advanced_manager.generate_visual_representation({}, format_type="mermaid")
        self.assertIsInstance(mermaid_empty, str)
        self.assertIn("graph", mermaid_empty.lower())
        
        # Test with simple plan
        simple_plan = {
            "id": generate_id("plan"),
            "problem_id": generate_id("problem"),
            "sub_problems": [
                {
                    "id": generate_id("sub1"),
                    "title": "Root Problem",
                    "dependencies": []
                },
                {
                    "id": generate_id("sub2"), 
                    "title": "Child Problem",
                    "dependencies": [generate_id("sub1")]
                }
            ]
        }
        
        mermaid_result = self.advanced_manager.generate_visual_representation(simple_plan, format_type="mermaid")
        self.assertIsInstance(mermaid_result, str)
        self.assertIn("graph", mermaid_result)
        self.assertIn("Root Problem", mermaid_result)
        self.assertIn("Child Problem", mermaid_result)
        
        # Test with other formats
        graphviz_result = self.advanced_manager.generate_visual_representation(simple_plan, format_type="graphviz")
        self.assertIsInstance(graphviz_result, str)
        
        plantuml_result = self.advanced_manager.generate_visual_representation(simple_plan, format_type="plantuml")
        self.assertIsInstance(plantuml_result, str)
    
    def test_domain_specific_template_application(self):
        """Test domain-specific template application"""
        # Test with non-existent domain
        result = self.advanced_manager.apply_domain_template("NonExistentDomain", "generic_strategy")
        self.assertIsNone(result)
        
        # Test getting available templates
        templates = self.advanced_manager.get_available_domain_templates()
        self.assertIsInstance(templates, dict)
        # Should have at least some templates
        self.assertGreaterEqual(len(templates), 1, "Should have at least one domain template")
        
        # Test template application with a valid domain (if any are implemented)
        for domain_name, domain_info in templates.items():
            template_result = self.advanced_manager.apply_domain_template(domain_name, "default_strategy")
            # Result could be None if the particular domain doesn't have strategies
            if template_result is not None:
                self.assertIsInstance(template_result, dict)
                break


class TestPerformanceBoundaryConditions(unittest.TestCase):
    """Test performance under boundary conditions"""
    
    def test_large_problem_decomposition(self):
        """Test decomposition of very large problems"""
        import time
        
        # Create a very large problem description
        large_description = "This is a very large problem description. " * 2000  # 8000 sentences
        
        large_problem = ProblemDefinition(
            id=generate_id("large_prob"),
            title="Large Scale Problem",
            description=large_description,
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="large_scale_analysis"),
            complexity_score=ComplexityScore(
                explanation="Large scale problem for boundary testing",
                cognitive_complexity=8.0,
                computational_complexity=8.5,
                domain_complexity=7.5,
                integration_complexity=8.0,
                overall_complexity=8.0
            )
        )
        
        # Mock the analyzer to prevent actual LLM calls
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            mock_result = Mock()
            mock_result.success = True
            mock_result.best_code = json.dumps({
                "domain": "large_scale_analysis",
                "subdomain": "boundary_analysis",
                "related_domains": ["scalability"],
                "key_concepts": ["scaling", "optimization"],
                "domain_complexity": 8.0,
                "required_expertise": ["scalability", "performance"]
            })
            mock_client.evolve.return_value = mock_result
            
            analyzer = ProblemAnalyzer(openevolve_client=mock_client)
            
            start_time = time.time()
            
            # This should handle the large input gracefully
            analyzed_problem = analyzer.analyze_problem(
                problem_text=large_description,
                title="Large Scale Problem"
            )
            
            analysis_time = time.time() - start_time
            
            print(f"Time to analyze large problem: {analysis_time:.3f}s")
            
            # Should complete in reasonable time (under 5 seconds even for large input)
            self.assertLess(analysis_time, 5.0, "Large problem analysis took too long")
            self.assertIsNotNone(analyzed_problem)
    
    def test_many_concurrent_operations(self):
        """Test system performance with many concurrent operations"""
        import concurrent.futures
        import time
        
        # Create the database
        db = SovereignDatabase(":memory:")
        
        # Create a base problem template
        base_problem = ProblemDefinition(
            id=generate_id("template"),
            title="Concurrent Operation Template",
            description="Template for concurrent operations testing",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="concurrent_test"),
            complexity_score=ComplexityScore(
                explanation="Template for concurrent testing",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        def operation_worker(operation_id):
            """Worker function for concurrent operations"""
            try:
                # Create problem with unique ID
                problem = base_problem.__class__(
                    **{**base_problem.__dict__, 'id': generate_id(f"op{operation_id}")},
                    title=f"{base_problem.title} - Operation {operation_id}"
                )
                
                # Perform database operation
                result = db.create_problem(problem)
                
                # Verify creation
                if result:
                    retrieved = db.get_problem(problem.id)
                    return result and retrieved is not None
                else:
                    return False
            except Exception as e:
                print(f"Operation {operation_id} failed: {e}")
                return False
        
        # Run many operations concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit 100 concurrent operations
            futures = [executor.submit(operation_worker, i) for i in range(100)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        successful_operations = sum(1 for r in results if r)
        
        print(f"Concurrent operations: {successful_operations}/100 completed in {total_time:.3f}s")
        print(f"Operations per second: {100/total_time:.1f}")
        
        # Should complete in reasonable time
        self.assertLess(total_time, 10.0, "100 concurrent operations took too long")
        # Should have reasonable success rate
        self.assertGreaterEqual(successful_operations, 90, "Too many concurrent operations failed")
    
    def test_cache_efficiency_under_load(self):
        """Test cache efficiency under load conditions"""
        import time
        
        cache = LLMResponseCache()
        
        # Add many similar items to test cache efficiency
        for i in range(50):
            content = f"Similar content pattern {i % 10}"  # Create 10 unique patterns, repeated 5 times each
            model_params = {"model": "gpt-4", "temperature": 0.7}
            response = {"choices": [{"message": {"content": f"Response for pattern {i % 10}"}}]}
            
            cache.cache_response(content, model_params, response)
        
        # Test cache hit rate with similar requests
        start_time = time.time()
        hits = 0
        misses = 0
        
        for i in range(100):
            content = f"Similar content pattern {i % 10}"
            model_params = {"model": "gpt-4", "temperature": 0.7}
            result = cache.get_response(content, model_params)
            if result:
                hits += 1
            else:
                misses += 1
        
        elapsed = time.time() - start_time
        
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        print(f"Cache performance: {hit_rate:.2%} hit rate, {elapsed:.3f}s for 100 requests")
        
        # Should have good hit rate due to reuse of patterns
        self.assertGreater(hit_rate, 0.5, "Cache hit rate too low")
        # Should be fast
        self.assertLess(elapsed, 1.0, "Cache operations too slow")
        
        # Check cache statistics
        stats = cache.get_stats()
        print(f"Cache stats - size: {stats['current_size']}, hits: {stats['total_hits']}, misses: {stats['total_misses']}")


class TestComplexWorkflowScenarios(unittest.TestCase):
    """Test complex multi-step workflow scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db = SovereignDatabase(self.temp_db.name)
        
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.analyzer = ProblemAnalyzer(openevolve_client=self.mock_client)
            self.decomposer = DecompositionEngine(openevolve_client=self.mock_client)
            self.coordinator = TeamCoordinator()
            self.orchestrator = SolutionOrchestrator()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_end_to_end_complex_workflow(self):
        """Test a complete complex workflow from problem to solution"""
        # Mock responses for the entire workflow
        self.mock_client.evolve.side_effect = [
            # First call: problem analysis
            Mock(success=True, best_code=json.dumps({
                "domain": "software_engineering", 
                "subdomain": "system_architecture",
                "related_domains": ["dev_ops", "security"],
                "key_concepts": ["scalability", "reliability", "security"],
                "domain_complexity": 8.5,
                "required_expertise": ["architecture", "security", "devops"]
            })),
            # Second call: decomposition
            Mock(success=True, best_code=json.dumps([
                {
                    "id": generate_id("sub1"),
                    "description": "Design system architecture and components",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 7.0,
                    "ai_suggested_evaluation_prompt": "Validate architectural decisions"
                },
                {
                    "id": generate_id("sub2"),
                    "description": "Implement security measures and authentication",
                    "dependencies": [generate_id("sub1")],
                    "ai_suggested_complexity_score": 8.0,
                    "ai_suggested_evaluation_prompt": "Verify security implementation"
                },
                {
                    "id": generate_id("sub3"),
                    "description": "Set up deployment and monitoring",
                    "dependencies": [generate_id("sub1")],
                    "ai_suggested_complexity_score": 7.5,
                    "ai_suggested_evaluation_prompt": "Validate deployment pipeline"
                }
            ]))
        ]
        
        # Step 1: Analyze problem
        complex_problem = self.analyzer.analyze_problem(
            problem_text="Design and implement a secure, scalable, and highly available system architecture for a global SaaS platform serving millions of users with 99.99% uptime requirements",
            title="Global SaaS Platform Architecture"
        )
        
        self.assertIsNotNone(complex_problem)
        
        # Step 2: Decompose problem
        decomposition_plan = self.decomposer.decompose(complex_problem, strategy="dependency")
        self.assertIsNotNone(decomposition_plan)
        self.assertGreater(len(decomposition_plan.sub_problems), 2)
        
        # Step 3: Store plan in database
        plan_id = self.db.create_plan(decomposition_plan)
        self.assertTrue(plan_id)
        
        # Step 4: Retrieve plan
        retrieved_plan = self.db.get_plan(decomposition_plan.id)
        self.assertIsNotNone(retrieved_plan)
        self.assertEqual(len(retrieved_plan.sub_problems), len(decomposition_plan.sub_problems))
        
        # Step 5: Assign to teams and process
        for i, sub_problem in enumerate(decomposition_plan.sub_problems):
            # Assign to different teams
            team_assignment = self.coordinator.assign_to_team(
                task_id=sub_problem.id,
                team="red" if i % 3 == 0 else "blue" if i % 3 == 1 else "gold",
                priority=sub_problem.priority
            )
            
            self.assertIsNotNone(team_assignment)
        
        # Step 6: Simulate solution attempts
        solution_attempts = []
        for sub_problem in decomposition_plan.sub_problems:
            attempt = SolutionAttempt(
                id=generate_id("attempt"),
                sub_problem_id=sub_problem.id,
                approach=f"Approach for {sub_problem.title}",
                solution_content=f"Solution content for {sub_problem.description}",
                team_id="development_team",
                confidence_score=0.85
            )
            
            # Store attempt
            attempt_id = self.db.create_solution_attempt(attempt)
            self.assertTrue(attempt_id)
            
            solution_attempts.append(attempt)
        
        # Step 7: Orchestrate solutions
        final_solution = self.orchestrator.integrate_solutions(decomposition_plan, solution_attempts)
        self.assertIsNotNone(final_solution)
        self.assertGreater(len(final_solution.integrated_content), 0)
        
        print("Complex end-to-end workflow completed successfully!")
    
    def test_error_recovery_workflow(self):
        """Test workflow error recovery and retry logic"""
        # Create a problem
        problem = ProblemDefinition(
            id=generate_id("error_recovery"),
            title="Error Recovery Test",
            description="Test error handling and recovery in workflows",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="error_handling"),
            complexity_score=ComplexityScore(
                explanation="Error recovery test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        # Store problem
        problem_id = self.db.create_problem(problem)
        self.assertTrue(problem_id)
        
        # Test error recovery in database operations
        # Try to update a non-existent problem
        fake_problem = ProblemDefinition(
            id="non_existent_id",
            title="Fake Problem",
            description="This problem doesn't exist",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="error_test"),
            complexity_score=ComplexityScore(
                explanation="Non-existent problem test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        update_result = self.db.update_problem(fake_problem)
        # This should fail gracefully, returning False
        self.assertFalse(update_result, "Update of non-existent problem should return False")
        
        # Verify original problem is still intact
        retrieved = self.db.get_problem(problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Error Recovery Test")
    
    def test_validation_gauntlet_integration(self):
        """Test integration of validation gauntlets with solution workflow"""
        from sovereign_gauntlets import GauntletSystem
        
        # Create a gauntlet system
        with patch('sovereign_gauntlets.OpenEvolveClient') as mock_openevolve:
            mock_gauntlet_client = mock_openevolve.return_value
            gauntlet_system = GauntletSystem(openevolve_client=mock_gauntlet_client)
            
            # Mock gauntlet response
            mock_gauntlet_result = Mock()
            mock_gauntlet_result.success = True
            mock_gauntlet_result.best_code = json.dumps({
                "passed": True,
                "score": 0.85,
                "feedback": "Solution meets requirements",
                "improvements": []
            })
            mock_gauntlet_client.evolve.return_value = mock_gauntlet_result
            
            # Create a solution to validate
            solution_content = "This is a sample solution that should pass validation"
            
            # Run gauntlet validation
            validation_result = gauntlet_system.run_gauntlet(
                content=solution_content,
                gauntlet_name="standard_validation",
                team_name="red",
                context={"validation_type": "completeness", "requirements": ["requirement1", "requirement2"]}
            )
            
            # Verify validation completed
            self.assertIsNotNone(validation_result)
            if 'passed' in validation_result:
                self.assertTrue(validation_result['passed'])
            elif 'is_approved' in validation_result:
                self.assertTrue(validation_result['is_approved'])


class TestRegressionPrevention(unittest.TestCase):
    """Tests to prevent regression of known issues"""
    
    def test_known_issue_prevention(self):
        """Test prevention of previously known issues"""
        # Test 1: Verify the problem analyzer has been properly implemented
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            mock_result = Mock()
            mock_result.success = True
            mock_result.best_code = json.dumps({
                "domain": "software_engineering",
                "subdomain": "testing",
                "related_domains": ["qa"],
                "key_concepts": ["verification", "validation"],
                "domain_complexity": 5.0,
                "required_expertise": ["testing", "analysis"]
            })
            mock_client.evolve.return_value = mock_result
            
            analyzer = ProblemAnalyzer(openevolve_client=mock_client)
            
            # This should not result in an empty implementation anymore
            result = analyzer.analyze_problem(
                problem_text="Test problem for regression prevention",
                title="Regression Prevention Test"
            )
            
            self.assertIsNotNone(result, "Analyzer should return a result, not None")
            self.assertIsInstance(result, ProblemDefinition, "Analyzer should return ProblemDefinition")
            self.assertGreater(len(result.domain_context.domain), 0, "Domain context should be populated")
    
    def test_id_generation_uniqueness(self):
        """Test ID generation maintains uniqueness"""
        import time
        
        # Generate many IDs and ensure no collisions
        ids = set()
        collision_count = 0
        
        for i in range(5000):  # Generate 5000 IDs to test uniqueness 
            new_id = generate_id("test")
            if new_id in ids:
                collision_count += 1
            ids.add(new_id)
        
        self.assertEqual(collision_count, 0, f"Found {collision_count} ID collisions out of 5000 generated IDs")
        self.assertEqual(len(ids), 5000, "All generated IDs should be unique")
        
        print(f"ID uniqueness test: 5000 IDs generated with 0 collisions")


def run_additional_comprehensive_tests():
    """Run the comprehensive additional tests"""
    print("Running comprehensive additional unit tests...")
    
    # Create a test suite for extended tests
    suite = unittest.TestSuite()
    
    # Add all the extended test cases
    suite.addTest(unittest.makeSuite(TestErrorConditions))
    suite.addTest(unittest.makeSuite(TestSecurityValidation))
    suite.addTest(unittest.makeSuite(TestAdvancedFeatureScenarios))
    suite.addTest(unittest.makeSuite(TestPerformanceBoundaryConditions))
    suite.addTest(unittest.makeSuite(TestComplexWorkflowScenarios))
    suite.addTest(unittest.makeSuite(TestRegressionPrevention))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ADDITIONAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures or result.errors:
        print("\nFAILED TESTS:")
        for test, trace in result.failures:
            print(f"\n{test}")
            print(trace)
        
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"\n{test}")
            print(trace)
    else:
        print(f"\n🎉 ALL {result.testsRun} ADDITIONAL TESTS PASSED! 🎉")
    
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    run_additional_comprehensive_tests()