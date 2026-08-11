"""
Additional Unit Tests for Sovereign-Grade Problem Decomposition System
Comprehensive unit tests for core modules and functions
"""


import unittest
from unittest.mock import Mock, MagicMock, patch
import json
import uuid
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, Any, List

# Add the project root to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sovereign_data_models import (
    ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
    Constraint, SuccessCriterion, DomainContext, ComplexityScore, generate_id
)
from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_team_coordination import TeamCoordinator
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_persistence import SovereignDatabase
from auth_system import AuthenticationSystem, AuthorizationSystem
from input_validation import InputValidator


class TestDataModels(unittest.TestCase):
    """Unit tests for data models"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_generate_id_uniqueness(self):
        """Test that generated IDs are unique"""
        ids = set()
        for _ in range(100):
            new_id = generate_id("test")
            self.assertNotIn(new_id, ids)
            ids.add(new_id)
    
    def test_complexity_score_validation(self):
        """Test ComplexityScore validation"""
        # Valid scores
        valid_score = ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=6.0,
            domain_complexity=7.0,
            integration_complexity=8.0,
            overall_complexity=6.5,
            explanation="Test explanation"
        )
        
        errors = valid_score.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid scores (out of range)
        invalid_score = ComplexityScore(
            cognitive_complexity=15.0,  # Out of range
            computational_complexity=-5.0,  # Out of range
            domain_complexity=5.0,
            integration_complexity=5.0,
            overall_complexity=5.0,
            explanation="Test explanation"
        )
        
        errors = invalid_score.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("cognitive_complexity", str(errors[0]))
        self.assertIn("computational_complexity", str(errors[1]))
    
    def test_constraint_validation(self):
        """Test Constraint validation"""
        # Valid constraint
        valid_constraint = Constraint(
            id=generate_id("constraint"),
            description="Time constraint",
            type="time",
            severity="hard"
        )
        
        errors = valid_constraint.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid constraint
        invalid_constraint = Constraint(
            id=generate_id("constraint"),
            description="Test",
            type="invalid_type",
            severity="invalid_severity"
        )
        
        errors = invalid_constraint.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("constraint type", errors[0])
        self.assertIn("constraint severity", errors[1])
    
    def test_success_criterion_validation(self):
        """Test SuccessCriterion validation"""
        # Valid criterion
        valid_criterion = SuccessCriterion(
            id=generate_id("criterion"),
            description="Test criterion",
            metric="accuracy",
            threshold=0.8,
            validation_method="test"
        )
        
        errors = valid_criterion.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid criterion (threshold out of range)
        invalid_criterion = SuccessCriterion(
            id=generate_id("criterion"),
            description="Test criterion",
            metric="accuracy",
            threshold=1.5,  # Above 1.0
            validation_method="test"
        )
        
        errors = invalid_criterion.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("threshold", errors[0])
    
    def test_problem_definition_validation(self):
        """Test ProblemDefinition validation"""
        # Valid problem definition
        valid_problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="Test description",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            ),
            constraints=[],
            success_criteria=[]
        )
        
        errors = valid_problem.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid problem (missing title)
        invalid_problem = ProblemDefinition(
            id=generate_id("problem"),
            title="",  # Empty title
            description="Test description",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        errors = invalid_problem.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("title", errors[0])
    
    def test_sub_problem_validation(self):
        """Test SubProblem validation"""
        # Valid sub-problem
        valid_sub = SubProblem(
            id=generate_id("sub"),
            parent_id=generate_id("parent"),
            title="Test Sub-problem",
            description="Test description",
            type="IMPLEMENTATION",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        errors = valid_sub.validate()
        self.assertEqual(len(errors), 0)
        
        # Invalid sub-problem (missing parent_id)
        invalid_sub = SubProblem(
            id=generate_id("sub"),
            parent_id="",  # Empty parent_id
            title="Test Sub-problem",
            description="Test description",
            type="IMPLEMENTATION",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        errors = invalid_sub.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("parent_id", errors[0])


class TestAnalyzer(unittest.TestCase):
    """Unit tests for Problem Analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.analyzer = ProblemAnalyzer(openevolve_client=self.mock_client)
    
    def test_analyze_problem(self):
        """Test problem analysis method"""
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = json.dumps({
            "domain": "software_engineering",
            "subdomain": "web_development",
            "related_domains": ["dev_ops"],
            "key_concepts": ["api", "rest"],
            "domain_complexity": 7.0,
            "required_expertise": ["javascript", "python"]
        })
        
        self.mock_client.evolve.return_value = mock_result
        
        problem = self.analyzer.analyze_problem(
            problem_text="Build a REST API for user management",
            title="User Management API"
        )
        
        self.assertIsNotNone(problem)
        self.assertEqual(problem.title, "User Management API")
        self.assertIn("user management", problem.description.lower())
    
    def test_extract_domain_context(self):
        """Test domain context extraction"""
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = json.dumps({
            "domain": "software_engineering",
            "subdomain": "web_development", 
            "related_domains": ["dev_ops"],
            "key_concepts": ["api", "rest"],
            "domain_complexity": 7.0,
            "required_expertise": ["javascript", "python"]
        })
        
        self.mock_client.evolve.return_value = mock_result
        
        context = self.analyzer._extract_domain_context_llm("Build a REST API")
        
        self.assertEqual(context.domain, "software_engineering")
        self.assertEqual(context.subdomain, "web_development")
        self.assertIn("dev_ops", context.related_domains)
    
    def test_classify_problem_type(self):
        """Test problem type classification"""
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = "IMPLEMENTATION"
        
        self.mock_client.evolve.return_value = mock_result
        
        problem_type = self.analyzer._classify_problem_type_llm("Implement a feature")
        
        self.assertEqual(problem_type.value, "IMPLEMENTATION")
    
    def test_assess_complexity(self):
        """Test complexity assessment"""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test",
            description="Analyze and implement a complex system",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = """Cognitive: 7.5
Computational: 6.5
Domain: 8.0
Integration: 7.0
Explanation: Complex multi-component system with difficult algorithms"""
        
        self.mock_client.evolve.return_value = mock_result
        
        complexity = self.analyzer._assess_complexity_llm(problem)
        
        self.assertGreaterEqual(complexity.cognitive_complexity, 7.0)
        self.assertGreaterEqual(complexity.domain_complexity, 7.5)
    
    def test_identify_constraints(self):
        """Test constraint identification"""
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = """[time] Must be completed in 3 months | hard
[resource] Budget limited to $50,000 | soft
[quality] Accuracy must exceed 95% | hard"""
        
        self.mock_client.evolve.return_value = mock_result
        
        constraints = self.analyzer._identify_constraints_llm("Analyze with budget and timeline constraints")
        
        self.assertGreater(len(constraints), 0)
        time_constraint = next((c for c in constraints if c.type == "time"), None)
        self.assertIsNotNone(time_constraint)
        self.assertIn("months", time_constraint.description)


class TestDecompositionEngine(unittest.TestCase):
    """Unit tests for Decomposition Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('decomposition_engine.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.engine = DecompositionEngine(openevolve_client=self.mock_client)
    
    def test_create_semantic_decomposition(self):
        """Test semantic decomposition strategy"""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Semantic Test",
            description="Perform semantic analysis of this complex problem",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = json.dumps([
            {
                "id": generate_id("sub"),
                "description": "Analyze the semantic components",
                "dependencies": [],
                "ai_suggested_complexity_score": 6.5,
                "ai_suggested_evaluation_prompt": "Evaluate semantic analysis quality"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_result
        
        sub_problems = self.engine._create_semantic_decomposition(problem)
        
        self.assertGreater(len(sub_problems), 0)
        self.assertIn("semantic", sub_problems[0].description)
    
    def test_apply_decomposition_strategy(self):
        """Test applying a specific decomposition strategy"""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Strategy Test",
            description="Decompose this problem using dependency strategy",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        # Mock OpenEvolve response
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = json.dumps([
            {
                "id": generate_id("sub1"),
                "description": "First dependent component",
                "dependencies": [],
                "ai_suggested_complexity_score": 5.0,
                "ai_suggested_evaluation_prompt": "Check dependency resolution"
            },
            {
                "id": generate_id("sub2"), 
                "description": "Second component depending on first",
                "dependencies": [generate_id("sub1")],
                "ai_suggested_complexity_score": 6.0,
                "ai_suggested_evaluation_prompt": "Check dependency chain"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_result
        
        plan = self.engine.apply_decomposition_strategy(problem, "dependency")
        
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.sub_problems), 1)
        
        # Check dependencies
        sub2 = next(sp for sp in plan.sub_problems if len(sp.dependencies) > 0)
        self.assertIsNotNone(sub2)


class TestPersistence(unittest.TestCase):
    """Unit tests for persistence layer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_path = "test_sovereign.db"
        self.db = SovereignDatabase(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def test_create_and_retrieve_problem(self):
        """Test creating and retrieving a problem"""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Test Problem",
            description="Test problem for persistence",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        # Create problem
        result = self.db.create_problem(problem)
        self.assertTrue(result)
        
        # Retrieve problem
        retrieved = self.db.get_problem(problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, problem.title)
        self.assertEqual(retrieved.problem_type, problem.problem_type)
    
    def test_update_problem(self):
        """Test updating a problem"""
        problem = ProblemDefinition(
            id=generate_id("problem"),
            title="Original Title",
            description="Original description",
            problem_type="RESEARCH",
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        # Create problem
        self.db.create_problem(problem)
        
        # Update problem
        problem.title = "Updated Title"
        problem.description = "Updated description"
        updated = self.db.update_problem(problem)
        self.assertTrue(updated)
        
        # Retrieve updated problem
        retrieved = self.db.get_problem(problem.id)
        self.assertEqual(retrieved.title, "Updated Title")
        self.assertEqual(retrieved.description, "Updated description")
    
    def test_create_and_retrieve_sub_problem(self):
        """Test creating and retrieving a sub-problem"""
        parent_id = generate_id("parent")
        
        sub_problem = SubProblem(
            id=generate_id("sub"),
            parent_id=parent_id,
            title="Test Sub-problem",
            description="Test sub-problem description",
            type="IMPLEMENTATION",
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test"
            )
        )
        
        # Create sub-problem
        result = self.db.create_subproblem(sub_problem)
        self.assertTrue(result)
        
        # Retrieve sub-problem
        retrieved = self.db.get_subproblem(sub_problem.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, sub_problem.title)
        self.assertEqual(retrieved.type, sub_problem.type)
    
    def test_list_sub_problems_by_parent(self):
        """Test listing sub-problems for a parent"""
        parent_id = generate_id("parent")
        
        # Create multiple sub-problems
        sub1 = SubProblem(
            id=generate_id("sub1"),
            parent_id=parent_id,
            title="Sub-problem 1",
            description="First sub-problem",
            type="ANALYSIS",
            complexity_score=ComplexityScore(
                cognitive_complexity=4.0, computational_complexity=4.0,
                domain_complexity=4.0, integration_complexity=4.0, overall_complexity=4.0,
                explanation="Test"
            )
        )
        
        sub2 = SubProblem(
            id=generate_id("sub2"),
            parent_id=parent_id,
            title="Sub-problem 2", 
            description="Second sub-problem",
            type="IMPLEMENTATION",
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0, computational_complexity=6.0,
                domain_complexity=6.0, integration_complexity=6.0, overall_complexity=6.0,
                explanation="Test"
            )
        )
        
        self.db.create_subproblem(sub1)
        self.db.create_subproblem(sub2)
        
        # List sub-problems by parent
        sub_problems = self.db.list_subproblems(parent_id)
        self.assertEqual(len(sub_problems), 2)
        
        titles = {sp.title for sp in sub_problems}
        self.assertIn("Sub-problem 1", titles)
        self.assertIn("Sub-problem 2", titles)
    
    def test_create_and_retrieve_solution_attempt(self):
        """Test creating and retrieving a solution attempt"""
        sub_problem_id = generate_id("sub")
        
        attempt = SolutionAttempt(
            id=generate_id("attempt"),
            sub_problem_id=sub_problem_id,
            approach="Testing approach",
            solution_content="This is a test solution",
            team_id="test_team",
            confidence_score=0.85
        )
        
        # Create solution attempt
        result = self.db.create_solution_attempt(attempt)
        self.assertTrue(result)
        
        # Retrieve solution attempt
        retrieved = self.db.get_solution_attempt(attempt.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.solution_content, attempt.solution_content)
        self.assertEqual(retrieved.confidence_score, attempt.confidence_score)


class TestInputValidation(unittest.TestCase):
    """Unit tests for input validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = InputValidator()
    
    def test_validate_not_empty_field(self):
        """Test not-empty validation rule"""
        rules = [self.validator.VALIDATION_RULES.NOT_EMPTY]
        
        # Valid data
        result = self.validator.validate("valid text", "test_field", rules)
        self.assertEqual(result, "valid text")
        
        # Invalid data (empty)
        with self.assertRaises(Exception):
            self.validator.validate("", "test_field", rules)
    
    def test_validate_min_length(self):
        """Test minimum length validation rule"""
        rules = [self.validator.VALIDATION_RULES.MIN_LENGTH(5)]
        
        # Valid data
        result = self.validator.validate("valid text", "test_field", rules)
        self.assertEqual(result, "valid text")
        
        # Invalid data (too short)
        with self.assertRaises(Exception):
            self.validator.validate("shrt", "test_field", rules)
    
    def test_validate_max_length(self):
        """Test maximum length validation rule"""
        rules = [self.validator.VALIDATION_RULES.MAX_LENGTH(10)]
        
        # Valid data
        result = self.validator.validate("short", "test_field", rules)
        self.assertEqual(result, "short")
        
        # Invalid data (too long)
        with self.assertRaises(Exception):
            self.validator.validate("this text is too long", "test_field", rules)
    
    def test_validate_pattern(self):
        """Test pattern validation rule"""
        # Email pattern
        rules = [self.validator.VALIDATION_RULES.PATTERN(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
        
        # Valid email
        result = self.validator.validate("test@example.com", "email_field", rules)
        self.assertEqual(result, "test@example.com")
        
        # Invalid email
        with self.assertRaises(Exception):
            self.validator.validate("invalid-email", "email_field", rules)
    
    def test_validate_type(self):
        """Test type validation rule"""
        rules = [self.validator.VALIDATION_RULES.TYPE(int)]
        
        # Valid integer
        result = self.validator.validate("42", "number_field", rules)
        self.assertEqual(result, 42)
        
        # Valid integer (already int)
        result = self.validator.validate(42, "number_field", rules)
        self.assertEqual(result, 42)
        
        # Invalid type
        with self.assertRaises(Exception):
            self.validator.validate("not a number", "number_field", rules)
    
    def test_validate_range(self):
        """Test range validation rule"""
        rules = [self.validator.VALIDATION_RULES.RANGE(min_val=1, max_val=10)]
        
        # Valid value in range
        result = self.validator.validate(5, "range_field", rules)
        self.assertEqual(result, 5)
        
        # Valid boundary values
        result = self.validator.validate(1, "range_field", rules)
        self.assertEqual(result, 1)
        
        result = self.validator.validate(10, "range_field", rules)
        self.assertEqual(result, 10)
        
        # Invalid value (too low)
        with self.assertRaises(Exception):
            self.validator.validate(0, "range_field", rules)
        
        # Invalid value (too high)
        with self.assertRaises(Exception):
            self.validator.validate(15, "range_field", rules)
    
    def test_validate_email(self):
        """Test email validation rule"""
        rules = [self.validator.VALIDATION_RULES.EMAIL]
        
        # Valid email
        result = self.validator.validate("user@example.com", "email_field", rules)
        self.assertEqual(result, "user@example.com")
        
        # Invalid email
        with self.assertRaises(Exception):
            self.validator.validate("invalid-email", "email_field", rules)
    
    def test_validate_url(self):
        """Test URL validation rule"""
        rules = [self.validator.VALIDATION_RULES.URL]
        
        # Valid URL
        result = self.validator.validate("https://example.com", "url_field", rules)
        self.assertEqual(result, "https://example.com")
        
        # Invalid URL
        with self.assertRaises(Exception):
            self.validator.validate("not-a-url", "url_field", rules)
    
    def test_validate_html_sanitization(self):
        """Test HTML sanitization validation rule"""
        rules = [self.validator.VALIDATION_RULES.SANITIZE_HTML]
        
        # Malicious HTML
        malicious_html = '<p>Safe text</p><script>alert("xss")</script><p>More text</p>'
        sanitized = self.validator.validate(malicious_html, "html_field", rules)
        
        # Script tag should be removed
        self.assertNotIn("<script>", sanitized)
        self.assertIn("<p>Safe text</p>", sanitized)
        self.assertIn("<p>More text</p>", sanitized)
    
    def test_validate_no_script_tags(self):
        """Test script tag removal validation rule"""
        rules = [self.validator.VALIDATION_RULES.NO_SCRIPT]
        
        # Content with script tags
        content_with_script = 'Safe text<script>alert("xss")</script>More safe text'
        cleaned = self.validator.validate(content_with_script, "content_field", rules)
        
        # Script tags should be removed
        self.assertNotIn("<script>", cleaned)
        self.assertIn("Safe text", cleaned)
        self.assertIn("More safe text", cleaned)


class TestAuthSystem(unittest.TestCase):
    """Unit tests for authentication system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auth_system = AuthenticationSystem(db_path="test_auth.db")
        self.authz_system = AuthorizationSystem(self.auth_system)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import os
        if os.path.exists("test_auth.db"):
            os.remove("test_auth.db")
    
    def test_hash_password_and_verify(self):
        """Test password hashing and verification"""
        password = "test_password_123!"
        hashed = self.auth_system.hash_password(password)
        
        # Verify correct password
        self.assertTrue(self.auth_system.verify_password(password, hashed))
        
        # Verify incorrect password
        self.assertFalse(self.auth_system.verify_password("wrong_password", hashed))
    
    def test_create_and_authenticate_user(self):
        """Test user creation and authentication"""
        from auth_system import Role, Permission
        
        user = self.auth_system.create_user(
            username="test_user",
            email="test@example.com",
            password="secure_password_123!",
            roles=[Role.ANALYST],
            permissions=[Permission.CREATE_PROBLEM, Permission.READ_PROBLEM]
        )
        
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "test_user")
        
        # Authenticate user
        authenticated_user = self.auth_system.authenticate("test_user", "secure_password_123!")
        self.assertIsNotNone(authenticated_user)
        self.assertEqual(authenticated_user.username, "test_user")
        
        # Try with wrong password
        failed_auth = self.auth_system.authenticate("test_user", "wrong_password")
        self.assertIsNone(failed_auth)
    
    def test_authorization_check(self):
        """Test authorization permission checking"""
        from auth_system import Role, Permission
        
        user = self.auth_system.create_user(
            username="authz_test",
            email="authz@example.com",
            password="password_123!",
            roles=[Role.WORKFLOW_MANAGER]
        )
        
        # Check role-based permission
        has_permission = self.authz_system.check_permission(user, Permission.CREATE_PROBLEM)
        self.assertTrue(has_permission)
        
        # Check direct permission
        user_with_direct_perm = self.auth_system.create_user(
            username="direct_perm",
            email="direct@example.com",
            password="password_123!",
            roles=[Role.VIEWER],
            permissions=[Permission.CREATE_PROBLEM]  # Direct permission
        )
        
        has_direct_permission = self.authz_system.check_permission(
            user_with_direct_perm, 
            Permission.CREATE_PROBLEM
        )
        self.assertTrue(has_direct_permission)


class TestSolutionOrchestrator(unittest.TestCase):
    """Unit tests for solution orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = SolutionOrchestrator()
    
    def test_track_solution_attempt(self):
        """Test tracking solution attempts"""
        attempt = SolutionAttempt(
            id=generate_id("attempt"),
            sub_problem_id=generate_id("sub"),
            approach="Test approach",
            solution_content="This is a test solution",
            team_id="test_team",
            confidence_score=0.85
        )
        
        # Track solution attempt
        tracked_id = self.orchestrator.track_solution_attempt(
            attempt.sub_problem_id,
            attempt.approach,
            attempt.solution_content,
            attempt.team_id,
            attempt.confidence_score
        )
        
        self.assertIsNotNone(tracked_id)
        
        # Check if stored correctly
        stored = self.orchestrator.get_solution_attempts(attempt.sub_problem_id)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].content, "This is a test solution")
    
    def test_integrate_solutions(self):
        """Test solution integration"""
        # Create mock solutions
        mock_plan = Mock()
        mock_plan.sub_problems = [
            Mock(id="sp1", solution="Solution 1"),
            Mock(id="sp2", solution="Solution 2")
        ]
        
        # Since integration requires complex dependencies, mock the integration
        with patch.object(self.orchestrator, '_integrate_solution_parts') as mock_integration:
            mock_integration.return_value = "Integrated solution"
            
            result = self.orchestrator.integrate_solutions(mock_plan)
            
            self.assertIsNotNone(result)
            mock_integration.assert_called_once()


class TestTeamCoordination(unittest.TestCase):
    """Unit tests for team coordination"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = TeamCoordinator()
    
    def test_create_team_assignment(self):
        """Test team assignment creation"""
        assignment = self.coordinator.assign_to_team(
            task_id=generate_id("task"),
            team="red",
            priority=7,
            due_hours=2.0
        )
        
        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.team, "red")
        self.assertEqual(assignment.status, "assigned")
        
        # Verify assignment was stored
        capacity_info = self.coordinator.track_team_capacity("red")
        self.assertEqual(capacity_info.current_tasks, 1)
    
    def test_process_red_team_feedback(self):
        """Test processing red team feedback"""
        from sovereign_data_models import Feedback
        
        feedback_list = [
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="critique",
                content="Issue found with implementation",
                severity="high",
                actionable=True,
                timestamp=datetime.now()
            )
        ]
        
        request = self.coordinator.process_red_team_feedback(
            plan_id=generate_id("plan"),
            feedback=feedback_list
        )
        
        self.assertIsNotNone(request)
        self.assertEqual(request.priority, 10)  # High issues should get high priority
        self.assertEqual(len(request.feedback), 1)
    
    def test_load_balancing(self):
        """Test team load balancing"""
        for i in range(10):
            self.coordinator.assign_to_team(
                task_id=generate_id("task"),
                team="blue",
                priority=5
            )
        
        # Check team workload
        balance_info = self.coordinator.balance_workload()
        
        self.assertIn('red_team', balance_info)
        self.assertIn('blue_team', balance_info)
        self.assertIn('gold_team', balance_info)
        self.assertIn('balance_score', balance_info)


def run_additional_tests():
    """Run the additional unit tests"""
    print("Running additional unit tests...")
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestDataModels))
    suite.addTest(unittest.makeSuite(TestAnalyzer))
    suite.addTest(unittest.makeSuite(TestDecompositionEngine))
    suite.addTest(unittest.makeSuite(TestPersistence))
    suite.addTest(unittest.makeSuite(TestInputValidation))
    suite.addTest(unittest.makeSuite(TestAuthSystem))
    suite.addTest(unittest.makeSuite(TestSolutionOrchestrator))
    suite.addTest(unittest.makeSuite(TestTeamCoordination))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result


if __name__ == "__main__":
    run_additional_tests()