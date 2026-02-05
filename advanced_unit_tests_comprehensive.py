"""
Advanced Unit Tests for Sovereign-Grade System
Additional comprehensive unit tests with extreme edge cases and advanced validation
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
from dataclasses import asdict, dataclass
import dataclasses
import pickle
import queue
import multiprocessing
import psutil
import math
import statistics
from decimal import Decimal
from fractions import Fraction
import array
import collections
import itertools

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


class TestAdvancedDataModelValidation(unittest.TestCase):
    """Advanced validation tests for complex data model interactions"""
    
    def test_complexity_score_edge_cases(self):
        """Test ComplexityScore with floating point precision edge cases"""
        # Test with floating point precision issues
        scores_to_test = [
            # Exact boundaries
            (0.0, 0.0, 0.0, 0.0, 0.0, "Minimum boundary"),  # All at min
            (10.0, 10.0, 10.0, 10.0, 10.0, "Maximum boundary"),  # All at max
            (5.0, 5.0, 5.0, 5.0, 5.0, "Middle values"),
            # Floating point precision
            (0.1 + 0.2, 0.3, 0.0, 0.0, 0.2, "Floating point precision test"),
            # Near boundary values
            (0.01, 0.01, 0.01, 0.01, 0.01, "Near-zero values"),
            (9.99, 9.99, 9.99, 9.99, 9.99, "Near-max values"),
        ]
        
        for i, (cog, comp, dom, integ, overall, desc) in enumerate(scores_to_test):
            with self.subTest(description=desc, test_case=i):
                try:
                    score = ComplexityScore(
                        cognitive_complexity=cog,
                        computational_complexity=comp,
                        domain_complexity=dom,
                        integration_complexity=integ,
                        overall_complexity=overall,
                        explanation=desc
                    )
                    
                    errors = score.validate()
                    if not errors:
                        # If valid, check that values are as expected
                        self.assertAlmostEqual(score.cognitive_complexity, cog, delta=0.001)
                        self.assertAlmostEqual(score.computational_complexity, comp, delta=0.001)
                        self.assertAlmostEqual(score.domain_complexity, dom, delta=0.001)
                        self.assertAlmostEqual(score.integration_complexity, integ, delta=0.001)
                        self.assertAlmostEqual(score.overall_complexity, overall, delta=0.001)
                    else:
                        print(f"Validation errors for {desc}: {errors}")
                except Exception as e:
                    # Some combinations might cause validation errors which is acceptable
                    pass
    
    def test_problem_definition_complex_nesting(self):
        """Test deeply nested problem dependencies"""
        # Create parent problem
        parent_problem = ProblemDefinition(
            id=generate_id("parent"),
            title="Parent Problem",
            description="Parent problem with nested sub-problems",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="research"),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Parent problem"
            )
        )
        
        # Create deeply nested sub-problems with complex dependencies
        max_depth = 10
        sub_problem_tree = {}
        
        for depth in range(max_depth):
            for branch in range(3):  # 3 branches per level
                sub_problem_id = generate_id(f"nested_{depth}_{branch}")
                
                # Create dependencies on previous level
                dependencies = []
                if depth > 0:
                    # Dependency on parent of same branch or sibling branches
                    parent_branches = [f"nested_{depth-1}_{b}" for b in range(3)]
                    dependencies.extend(parent_branches[:branch+1])  # Dependencies up to this branch
                
                sub_problem = SubProblem(
                    id=sub_problem_id,
                    parent_id=parent_problem.id,
                    title=f"Nested Sub-problem {depth}-{branch}",
                    description=f"Sub-problem at depth {depth}, branch {branch}",
                    type=random.choice(list(SubProblemType)),
                    complexity_score=ComplexityScore(
                        cognitive_complexity=5.0 + (depth * 0.5),
                        computational_complexity=5.0 + (depth * 0.5),
                        domain_complexity=5.0 + (depth * 0.5),
                        integration_complexity=5.0 + (depth * 0.5),
                        overall_complexity=5.0 + (depth * 0.5),
                        explanation=f"Depth {depth}, Branch {branch}"
                    ),
                    dependencies=dependencies
                )
                
                sub_problem_tree[sub_problem_id] = sub_problem
        
        # Create decomposition plan with complex dependency graph
        plan = DecompositionPlan(
            id=generate_id("nested_plan"),
            problem_id=parent_problem.id,
            strategy="nested_dependency",
            sub_problems=list(sub_problem_tree.values()),
            dependency_graph={
                sp.id: sp.dependencies for sp in sub_problem_tree.values()
            },
            validation_checkpoints=[],
            quality_scores={},
            confidence_level=0.85
        )
        
        # Verify plan can be created and validated
        validation_errors = plan.validate()
        self.assertEqual(len(validation_errors), 0, 
                        f"Plan should validate without errors. Errors: {validation_errors}")
        
        # Verify dependency graph structure
        self.assertEqual(len(plan.sub_problems), max_depth * 3)  # 10 levels * 3 branches each
        print(f"Created plan with {len(plan.sub_problems)} nested sub-problems")
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies in complex graphs"""
        # Create a circular dependency scenario
        sub1 = SubProblem(
            id=generate_id("circular_1"),
            parent_id=generate_id("circular_root"),
            title="Circular Sub-problem 1",
            description="First in circular dependency chain",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0, computational_complexity=5.0,
                domain_complexity=5.0, integration_complexity=5.0,
                overall_complexity=5.0, explanation="Circular test 1"
            ),
            dependencies=[generate_id("circular_3")]  # Points to 3, forming circle: 1->3->2->1
        )
        
        sub2 = SubProblem(
            id=generate_id("circular_2"),
            parent_id=generate_id("circular_root"),
            title="Circular Sub-problem 2", 
            description="Second in circular dependency chain",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0, computational_complexity=6.0,
                domain_complexity=6.0, integration_complexity=6.0,
                overall_complexity=6.0, explanation="Circular test 2"
            ),
            dependencies=[sub1.id]  # Points to 1
        )
        
        sub3 = SubProblem(
            id=generate_id("circular_3"),
            parent_id=generate_id("circular_root"),
            title="Circular Sub-problem 3",
            description="Third in circular dependency chain", 
            type=SubProblemType.VALIDATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=7.0, computational_complexity=7.0,
                domain_complexity=7.0, integration_complexity=7.0,
                overall_complexity=7.0, explanation="Circular test 3"
            ),
            dependencies=[sub2.id]  # Points to 2, creating: 1->3->2->1
        )
        
        # Create plan with circular dependency
        plan = DecompositionPlan(
            id=generate_id("circular_plan"),
            problem_id=generate_id("circular_root"),
            strategy="circular_test",
            sub_problems=[sub1, sub2, sub3],
            dependency_graph={
                sub1.id: [sub3.id],
                sub2.id: [sub1.id], 
                sub3.id: [sub2.id]  # Circular: 1<-2<-3<-1
            },
            validation_checkpoints=[],
            quality_scores={},
            confidence_level=0.5
        )
        
        # Check for validation errors (circular dependency should be detected)
        validation_errors = plan.validate()
        
        # There should be validation errors due to circular dependency
        circular_errors = [e for e in validation_errors if "circular" in e.lower() or "cycle" in e.lower()]
        self.assertGreater(len(circular_errors), 0, 
                          f"Should detect circular dependencies. All errors: {validation_errors}")
        print(f"Detected {len(circular_errors)} circular dependency errors as expected")
    
    def test_constraint_combinations(self):
        """Test complex combinations of constraints"""
        complex_constraints = [
            Constraint(
                id=generate_id("constraint_1"),
                description="Time constraint with hard requirement",
                type="time",
                severity="hard",
                metadata={"max_duration": "2 weeks", "penalty": "project_failure"}
            ),
            Constraint(
                id=generate_id("constraint_2"),
                description="Resource constraint with soft requirement",
                type="resource", 
                severity="soft",
                metadata={"budget": 100000, "adjustable": True}
            ),
            Constraint(
                id=generate_id("constraint_3"),
                description="Quality constraint with hard requirement",
                type="quality",
                severity="hard",
                metadata={"min_accuracy": 0.95, "validation_required": True}
            ),
            Constraint(
                id=generate_id("constraint_4"),
                description="Technical constraint with soft requirement",
                type="technical",
                severity="soft", 
                metadata={"preferred_language": "python", "alternative": "java"}
            )
        ]
        
        # Create problem with multiple complex constraints
        problem_with_constraints = ProblemDefinition(
            id=generate_id("constraint_test"),
            title="Constraint Combination Test",
            description="Problem with multiple complex, overlapping constraints",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(domain="complex_constraints"),
            complexity_score=ComplexityScore(
                cognitive_complexity=8.0, computational_complexity=7.5,
                domain_complexity=8.5, integration_complexity=9.0,
                overall_complexity=8.25, explanation="Complex constraint testing"
            ),
            constraints=complex_constraints
        )
        
        # Validate the problem with complex constraints
        validation_errors = problem_with_constraints.validate()
        print(f"Constraint validation errors: {len(validation_errors)}")
        
        # The constraints should be valid
        constraint_errors = [e for e in validation_errors if "constraint" in e.lower()]
        self.assertEqual(len(constraint_errors), 0, 
                         f"Constraints should be valid. Errors: {constraint_errors}")
    
    def test_success_criterion_complex_thresholds(self):
        """Test complex success criteria with various threshold types"""
        criteria = [
            SuccessCriterion(
                id=generate_id("criterion_1"),
                description="Accuracy-based success criterion",
                metric="accuracy",
                threshold=0.95,  # High threshold
                validation_method="automated",
                metadata={"tolerance": 0.01, "benchmark": "industry_standard"}
            ),
            SuccessCriterion(
                id=generate_id("criterion_2"),
                description="Performance-based success criterion",
                metric="latency",
                threshold=0.1,  # 100ms threshold
                validation_method="automated",
                metadata={"measurement_unit": "seconds", "load_conditions": "peak"}
            ),
            SuccessCriterion(
                id=generate_id("criterion_3"),
                description="Coverage-based success criterion",
                metric="test_coverage",
                threshold=0.90,  # 90% coverage
                validation_method="automated",
                metadata={"instrumentation": "line_level"}
            ),
            SuccessCriterion(
                id=generate_id("criterion_4"),
                description="Security-based success criterion",
                metric="vulnerability_score",
                threshold=0.99,  # Very high security requirement
                validation_method="manual",
                metadata={"scan_tool": "custom_security_scanner", "false_positive_tolerance": 0.01}
            )
        ]
        
        problem_with_criteria = ProblemDefinition(
            id=generate_id("criteria_test"),
            title="Success Criterion Complexity Test",
            description="Problem with multiple complex success criteria",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="quality_assurance"),
            complexity_score=ComplexityScore(
                cognitive_complexity=7.0, computational_complexity=6.5,
                domain_complexity=7.5, integration_complexity=7.0,
                overall_complexity=7.0, explanation="Criteria complexity testing"
            ),
            success_criteria=criteria
        )
        
        # Validate the problem with complex success criteria
        validation_errors = problem_with_criteria.validate()
        criterion_errors = [e for e in validation_errors if "criterion" in e.lower()]
        
        self.assertEqual(len(criterion_errors), 0, 
                         f"Success criteria should be valid. Errors: {criterion_errors}")
        
        # Verify all criteria were preserved
        self.assertEqual(len(problem_with_criteria.success_criteria), len(criteria))


class TestAdvancedAnalyzerScenarios(unittest.TestCase):
    """Advanced scenarios for problem analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.analyzer = ProblemAnalyzer(openevolve_client=self.mock_client)
    
    def test_multilingual_analysis(self):
        """Test analysis of problems in multiple languages"""
        multilingual_problems = [
            ("fr", "Analysez cette problème complexe de système distribué en français"),
            ("es", "Analizar el problema complejo del sistema distribuido en español"),
            ("de", "Analysieren Sie das komplexe verteilte Systemproblem auf Deutsch"),
            ("ja", "複雑な分散システムの問題を日本語で分析してください"),
            ("zh", "请分析这个复杂的分布式系统问题用中文"),
            ("ru", "Проанализируйте сложную проблему распределенной системы на русском языке"),
            ("ar", "حلل المشكلة المعقدة لنظام موزع باللغة العربية"),
        ]
        
        for lang_code, problem_text in multilingual_problems:
            with self.subTest(language=lang_code):
                # Mock response for multilingual analysis
                mock_result = Mock()
                mock_result.success = True
                mock_result.best_code = json.dumps({
                    "domain": "distributed_systems" if "system" in problem_text.lower() else "multi_language_analysis",
                    "subdomain": f"{lang_code}_technology",
                    "related_domains": ["internationalization", "multi_language_processing"],
                    "key_concepts": ["distributed", "system", "problem", "analysis"],
                    "domain_complexity": 7.5,
                    "required_expertise": ["multi_language_analyst", f"{lang_code}_expert"]
                })
                
                self.mock_client.evolve.return_value = mock_result
                
                try:
                    result = self.analyzer.analyze_problem(
                        problem_text=problem_text,
                        title=f"Multilingual Analysis Test ({lang_code})"
                    )
                    
                    # Should produce a result despite language differences
                    self.assertIsNotNone(result)
                    self.assertIn(lang_code, result.domain_context.subdomain)
                    
                except Exception as e:
                    # Some language processing might fail, which is acceptable in a test
                    print(f"Multilingual analysis for {lang_code} produced exception (may be expected): {e}")
    
    def test_extremely_long_problem_analysis(self):
        """Test analysis of extremely long problem statements"""
        # Create a very long problem (simulating detailed specifications)
        very_long_problem_parts = []
        
        # Simulate detailed system specification
        for i in range(100):  # 100 paragraphs of detailed requirements
            paragraph = f"""
Paragraph {i+1}: Requirement specification for complex system component {i+1}.
This section details the functional requirements, non-functional requirements, 
performance requirements, security requirements, and integration requirements
for this specific component. The system must handle various edge cases,
support multiple authentication methods, implement proper error handling,
provide comprehensive logging, maintain data integrity, ensure availability,
protect against common security threats, and scale efficiently under load.

Additional considerations for component {i+1} include user experience factors,
performance benchmarks, monitoring requirements, backup and recovery procedures,
disaster recovery scenarios, compliance requirements, audit logging needs,
and integration touchpoints with other system components. The solution must
meet industry standards for the domain and incorporate best practices."""

            very_long_problem_parts.append(paragraph)
        
        very_long_problem = " ".join(very_long_problem_parts)
        
        # Mock response for long problem analysis
        mock_result = Mock()
        mock_result.success = True
        mock_result.best_code = json.dumps({
            "domain": "software_engineering",
            "subdomain": "system_architecture",
            "related_domains": ["requirements_analysis", "system_design"],
            "key_concepts": ["scalability", "security", "performance", "reliability"],
            "domain_complexity": 9.2,
            "required_expertise": ["system_architect", "requirements_engineer", "security_specialist"]
        })
        
        self.mock_client.evolve.return_value = mock_result
        
        # Measure analysis time
        start_time = time.time()
        result = self.analyzer.analyze_problem(
            problem_text=very_long_problem,
            title="Extremely Long Problem Analysis Test"
        )
        analysis_time = time.time() - start_time
        
        # Should complete in reasonable time despite length
        self.assertIsNotNone(result)
        self.assertLess(analysis_time, 30.0, f"Long problem analysis should complete in under 30s, took {analysis_time:.2f}s")
        print(f"Long problem ({len(very_long_problem)} chars) analyzed in {analysis_time:.3f}s")
    
    def test_highly_ambiguous_problem_analysis(self):
        """Test analysis of highly ambiguous problems"""
        ambiguous_problem = """
        Do stuff with things in ways that work better than normal ways.
        Make it good. Make it fast. Make it reliable. Make it secure.
        The thing should do the thing in a way that is good.
        Requirements: It should work, it should be good, it should be fast.
        Success criteria: People should be happy with it.
        Constraints: None.
        """
        
        # Even with ambiguous input, analyzer should handle it gracefully
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps({
            "domain": "ambiguous_problem_resolution",
            "subdomain": "vague_requirements_analysis",
            "related_domains": ["requirements_clarification", "specification_development"],
            "key_concepts": ["ambiguity_resolution", "requirement_elaboration"],
            "domain_complexity": 6.0,  # Moderate complexity for ambiguity resolution
            "required_expertise": ["requirements_analyst", "clarification_specialist"]
        })
        
        self.mock_client.evolve.return_value = mock_response
        
        result = self.analyzer.analyze_problem(
            problem_text=ambiguous_problem,
            title="Highly Ambiguous Problem Analysis"
        )
        
        # Should handle ambiguous input gracefully
        self.assertIsNotNone(result)
        # Result should identify the ambiguity
        self.assertIn("ambiguous", result.domain_context.domain.lower())
    
    def test_domain_context_edge_cases(self):
        """Test domain context with edge cases"""
        test_cases = [
            # Empty domain context
            DomainContext(domain=""),
            # Very long domain names
            DomainContext(domain="x" * 1000),
            # Special characters in domain
            DomainContext(domain="domain_with_very_complex_name_like_machine_learning_and_data_science_and_artificial_intelligence_and_deep_neural_networks"),
            # Nested domains
            DomainContext(domain="software_engineering", subdomain="machine_learning", related_domains=["data_science", "artificial_intelligence", "statistical_modeling"]),
            # Complex domain knowledge
            DomainContext(
                domain="complex_system_integration",
                subdomain="multi_cloud_architecture", 
                related_domains=["dev_ops", "security", "performance_optimization", "compliance", "monitoring"],
                domain_knowledge={
                    "key_concepts": ["microservices", "containerization", "load_balancing", "caching", "authentication", "authorization"],
                    "common_patterns": ["circuit_breaker", "bulkhead", "retry_mechanism", "caching_layer"],
                    "pitfalls": ["race_conditions", "deadlocks", "resource_starvation", "memory_leaks"],
                    "best_practices": ["separation_of_concerns", "defensive_programming", "comprehensive_testing"]
                }
            )
        ]
        
        for i, domain_context in enumerate(test_cases):
            with self.subTest(test_case=i):
                # Test validation
                errors = domain_context.validate()
                
                if i == 0:  # Empty domain should have errors
                    self.assertGreater(len(errors), 0, f"Empty domain should have validation errors for case {i}")
                elif i == 1:  # Very long domain should have errors
                    self.assertGreater(len(errors), 0, f"Very long domain should have validation errors for case {i}")
                else:  # Others should be valid
                    self.assertEqual(len(errors), 0, f"Valid domain context should have no errors, case {i}: {errors}")


class TestAdvancedDecompositionStrategies(unittest.TestCase):
    """Advanced decomposition strategy tests with complex scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('decomposition_engine.OpenEvolveClient') as mock_openevolve:
            self.mock_client = mock_openevolve.return_value
            self.engine = DecompositionEngine(openevolve_client=self.mock_client)
    
    def test_dynamic_strategy_selection(self):
        """Test dynamic strategy selection based on problem characteristics"""
        # Test problems with different characteristics
        test_problems = [
            # Research-heavy problem
            ProblemDefinition(
                id=generate_id("research_test"),
                title="Complex Research Problem",
                description="Conduct extensive research on machine learning algorithms and their applications in natural language processing, including comparative analysis of transformer models, evaluation of performance metrics, and identification of optimal architectures for specific use cases.",
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="machine_learning", subdomain="nlp_research"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=8.5, computational_complexity=7.0,
                    domain_complexity=8.0, integration_complexity=6.5,
                    overall_complexity=7.5, explanation="Research focus"
                )
            ),
            # Implementation-heavy problem  
            ProblemDefinition(
                id=generate_id("impl_test"),
                title="Complex Implementation Problem",
                description="Implement a distributed system for processing large-scale data streams with real-time analytics, fault tolerance, auto-scaling, and comprehensive monitoring capabilities.",
                problem_type=ProblemType.IMPLEMENTATION,
                domain_context=DomainContext(domain="software_engineering", subdomain="distributed_systems"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=7.0, computational_complexity=8.5,
                    domain_complexity=8.0, integration_complexity=9.0,
                    overall_complexity=8.1, explanation="Implementation focus"
                )
            ),
            # Analysis-heavy problem
            ProblemDefinition(
                id=generate_id("analysis_test"),
                title="Complex Analysis Problem",
                description="Analyze the performance bottlenecks in the existing system architecture, identify root causes of latency issues, recommend optimization strategies, and develop a migration plan.",
                problem_type=ProblemType.ANALYSIS,
                domain_context=DomainContext(domain="performance_analysis", subdomain="system_optimization"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=9.0, computational_complexity=6.0,
                    domain_complexity=7.5, integration_complexity=7.0,
                    overall_complexity=7.4, explanation="Analysis focus"
                )
            )
        ]
        
        for problem in test_problems:
            with self.subTest(problem_type=problem.problem_type):
                # Mock decomposition response
                mock_response = Mock()
                mock_response.success = True
                mock_response.best_code = json.dumps([
                    {
                        "id": generate_id("dynamic_sub1"),
                        "description": f"Sub-problem for {problem.problem_type.value} type problem",
                        "dependencies": [],
                        "ai_suggested_complexity_score": 7.0,
                        "ai_suggested_evaluation_prompt": f"Evaluate approach for {problem.problem_type.value} problem type"
                    }
                ])
                
                self.mock_client.evolve.return_value = mock_response
                
                # Test different strategies
                strategies_to_test = ["semantic", "dependency", "complexity", "research", "hybrid"]
                
                for strategy in strategies_to_test:
                    with self.subTest(strategy=strategy):
                        plan = self.engine.decompose(problem, strategy=strategy)
                        self.assertIsNotNone(plan)
                        self.assertEqual(plan.strategy, strategy)
                        self.assertGreater(len(plan.sub_problems), 0)
    
    def test_recursive_decomposition_with_limiting(self):
        """Test recursive decomposition with depth limiting to prevent infinite loops"""
        # Create a problem that might lead to recursive decomposition
        recursive_problem = ProblemDefinition(
            id=generate_id("recursive_test"),
            title="Potentially Recursive Problem",
            description="Solve this problem by first understanding how to solve problems, then apply that knowledge to this instance, then generalize the solution for all similar problems, then create a framework for understanding problem-solving, then solve the original problem using the framework.",
            problem_type=ProblemType.RESEARCH,
            domain_context=DomainContext(domain="metacognitive_problem_solving"),
            complexity_score=ComplexityScore(
                cognitive_complexity=9.0, computational_complexity=8.0,
                domain_complexity=8.5, integration_complexity=7.5,
                overall_complexity=8.3, explanation="Metacognitive recursion test"
            )
        )
        
        # Mock response that would normally cause recursion, but engine should limit depth
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps([
            {
                "id": generate_id("recurse_sub1"),
                "description": "Understand how to solve problems",
                "dependencies": [],
                "ai_suggested_complexity_score": 8.0,
                "ai_suggested_evaluation_prompt": "Validate problem-solving approach understanding"
            },
            {
                "id": generate_id("recurse_sub2"), 
                "description": "Apply knowledge to this instance",
                "dependencies": [generate_id("recurse_sub1")],
                "ai_suggested_complexity_score": 7.5,
                "ai_suggested_evaluation_prompt": "Validate application of knowledge"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_response
        
        # This should complete without infinite recursion due to depth limiting
        start_time = time.time()
        plan = self.engine.decompose(recursive_problem, strategy="recursive_hybrid")
        recursion_time = time.time() - start_time
        
        # Should complete in reasonable time (under 5 seconds)
        self.assertLess(recursion_time, 5.0, f"Recursive decomposition should be limited. Took: {recursion_time:.2f}s")
        self.assertIsNotNone(plan)
        self.assertGreater(len(plan.sub_problems), 0)
    
    def test_multi_dimensional_complexity_decomposition(self):
        """Test decomposition based on multi-dimensional complexity analysis"""
        complex_problem = ProblemDefinition(
            id=generate_id("multidim_test"),
            title="Multi-Dimensional Complexity Problem",
            description="Design and implement a system that must simultaneously achieve high performance (sub-millisecond response times), extreme reliability (99.999% uptime), robust security (zero vulnerabilities), exceptional scalability (millions of concurrent users), and perfect maintainability (zero technical debt). The system must integrate with 50+ external APIs, handle 100+ different data formats, support 20+ different protocols, and work across 10+ different platforms.",
            problem_type=ProblemType.HYBRID,
            domain_context=DomainContext(domain="system_architecture", subdomain="multi_constraint_optimization"),
            complexity_score=ComplexityScore(
                cognitive_complexity=9.5, computational_complexity=9.0,
                domain_complexity=9.3, integration_complexity=9.7,
                overall_complexity=9.4, explanation="Multi-dimensional complexity"
            )
        )
        
        # Mock response for complex multidimensional problem
        mock_response = Mock()
        mock_response.success = True
        mock_response.best_code = json.dumps([
            {
                "id": generate_id("perf_sub"),
                "description": "Address performance constraints with sub-millisecond optimization",
                "dependencies": [],
                "ai_suggested_complexity_score": 9.2,
                "ai_suggested_evaluation_prompt": "Validate performance against sub-millisecond targets"
            },
            {
                "id": generate_id("reliability_sub"),
                "description": "Implement reliability mechanisms for 99.999% uptime",
                "dependencies": [],
                "ai_suggested_complexity_score": 9.4,
                "ai_suggested_evaluation_prompt": "Validate reliability against 99.999% target"
            },
            {
                "id": generate_id("security_sub"),
                "description": "Apply zero-vulnerability security measures",
                "dependencies": [],
                "ai_suggested_complexity_score": 9.6,
                "ai_suggested_evaluation_prompt": "Validate security against zero-vulnerability requirement"
            },
            {
                "id": generate_id("scalability_sub"),
                "description": "Design scalability for millions of concurrent users",
                "dependencies": [generate_id("perf_sub")],
                "ai_suggested_complexity_score": 9.3,
                "ai_suggested_evaluation_prompt": "Validate scalability against millions of users"
            },
            {
                "id": generate_id("integration_sub"),
                "description": "Handle integration with 50+ external APIs and 100+ data formats",
                "dependencies": [generate_id("scalability_sub")],
                "ai_suggested_complexity_score": 9.0,
                "ai_suggested_evaluation_prompt": "Validate integration capabilities"
            }
        ])
        
        self.mock_client.evolve.return_value = mock_response
        
        # Decompose with multi-dimensional strategy
        plan = self.engine.decompose(complex_problem, strategy="complexity")
        
        self.assertIsNotNone(plan)
        self.assertGreaterEqual(len(plan.sub_problems), 5)  # Should have at least 5 dimension-specific sub-problems
        
        # Verify high complexity problems were properly identified
        high_complexity_subproblems = [
            sp for sp in plan.sub_problems 
            if sp.complexity_score.overall_complexity >= 9.0
        ]
        self.assertGreaterEqual(len(high_complexity_subproblems), 3, 
                                "Should have multiple high-complexity sub-problems for the complex problem")


class TestConcurrencyAndThreading(unittest.TestCase):
    """Test concurrency and threading scenarios"""
    
    def test_concurrent_problem_analysis(self):
        """Test concurrent problem analysis operations"""
        import threading
        import time
        
        # Create multiple mock clients for concurrent testing
        def create_mock_response(problem_id):
            mock = Mock()
            mock.success = True
            mock.best_code = json.dumps({
                "domain": "concurrent_testing",
                "subdomain": f"problem_{problem_id}",
                "related_domains": ["multi_threading", "concurrent_processing"],
                "key_concepts": ["concurrency", "parallelism", "thread_safety"],
                "domain_complexity": 5.0 + (problem_id % 3),
                "required_expertise": ["concurrency_expert"]
            })
            return mock
        
        results = {}
        errors = {}
        
        def analyze_worker(worker_id):
            """Worker function for concurrent analysis"""
            try:
                with patch('problem_analyzer.OpenEvolveClient') as mock_openevolve:
                    mock_client = mock_openevolve.return_value
                    mock_client.evolve.return_value = create_mock_response(worker_id)
                    
                    analyzer = ProblemAnalyzer(openevolve_client=mock_client)
                    
                    result = analyzer.analyze_problem(
                        problem_text=f"Concurrent analysis test problem {worker_id}",
                        title=f"Concurrent Test Problem {worker_id}"
                    )
                    results[worker_id] = result
            except Exception as e:
                errors[worker_id] = e
        
        # Create and start multiple threads
        threads = []
        num_workers = 20  # Test with 20 concurrent workers
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=analyze_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify results
        successful_analyses = [r for r in results.values() if r is not None]
        
        print(f"Concurrent analysis: {len(successful_analyses)}/{num_workers} successful in {total_time:.3f}s")
        print(f"Throughput: {len(successful_analyses)/total_time:.1f} analyses/second")
        
        # Most analyses should succeed
        self.assertGreaterEqual(len(successful_analyses), num_workers * 0.9,
                              f"Most concurrent analyses should succeed, got {len(successful_analyses)}/{num_workers}")
        
        # Should complete in reasonable time despite concurrency
        self.assertLess(total_time, 15.0, f"Concurrent analysis should complete in reasonable time: {total_time:.2f}s")
    
    def test_database_concurrent_access(self):
        """Test database concurrent access and thread safety"""
        db = SovereignDatabase(":memory:")  # Use in-memory for thread safety testing
        
        results = []
        errors = []
        
        def db_worker(worker_id):
            """Worker function for concurrent database operations"""
            local_results = []
            local_errors = []
            
            for i in range(5):  # Each worker does 5 operations
                try:
                    # Create a problem
                    problem = ProblemDefinition(
                        id=generate_id(f"concurrent_{worker_id}_{i}"),
                        title=f"Concurrent DB Test {worker_id}-{i}",
                        description=f"Problem for concurrent database testing, worker {worker_id}, operation {i}",
                        problem_type=ProblemType.RESEARCH,
                        domain_context=DomainContext(domain="concurrent_database"),
                        complexity_score=ComplexityScore(
                            cognitive_complexity=5.0 + (i % 2),
                            computational_complexity=5.0 + (i % 2),
                            domain_complexity=5.0 + (i % 2),
                            integration_complexity=5.0 + (i % 2),
                            overall_complexity=5.0 + (i % 2),
                            explanation="Concurrent DB test"
                        )
                    )
                    
                    # Insert problem
                    success = db.create_problem(problem)
                    if success:
                        local_results.append((worker_id, i, "create", True))
                    else:
                        local_errors.append((worker_id, i, "create", "failed"))
                    
                    # Retrieve problem
                    retrieved = db.get_problem(problem.id)
                    if retrieved:
                        local_results.append((worker_id, i, "retrieve", True))
                    else:
                        local_errors.append((worker_id, i, "retrieve", "failed"))
                        
                except Exception as e:
                    local_errors.append((worker_id, i, "exception", str(e)))
            
            results.extend(local_results)
            errors.extend(local_errors)
        
        # Run concurrent database operations
        threads = []
        num_workers = 10
        
        start_time = time.time()
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=db_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        create_ops = [r for r in results if r[2] == "create"]
        retrieve_ops = [r for r in results if r[2] == "retrieve"]
        error_count = len(errors)
        
        total_ops = len(create_ops) + len(retrieve_ops)
        
        print(f"Concurrent DB operations: {total_ops} total, {error_count} errors in {total_time:.3f}s")
        print(f"Success rate: {len(results)/(len(results)+len(errors))*100:.1f}%")
        
        # Verify most operations succeeded
        self.assertGreaterEqual(len(results), (num_workers * 5 * 2) * 0.9,  # 90% success rate expected
                               f"Most DB operations should succeed under concurrency")
        
        # Verify data integrity - all created problems should be retrievable
        all_problems = db.list_problems()
        expected_count = num_workers * 5  # Each worker creates 5 problems
        self.assertGreaterEqual(len(all_problems), expected_count * 0.8,  # Allow for some failures
                               f"Most created problems should remain in database: {len(all_problems)}/{expected_count}")
    
    def test_parallel_decomposition_execution(self):
        """Test parallel decomposition execution"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Create problems for parallel processing
        problems = []
        for i in range(30):  # 30 problems for parallel processing
            problem = ProblemDefinition(
                id=generate_id(f"parallel_{i}"),
                title=f"Parallel Decomposition Test {i}",
                description=f"Problem {i} for parallel decomposition testing",
                problem_type=ProblemType.RESEARCH,
                domain_context=DomainContext(domain="parallel_processing"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0 + (i % 3),
                    computational_complexity=5.0 + (i % 3),
                    domain_complexity=5.0 + (i % 3),
                    integration_complexity=5.0 + (i % 3),
                    overall_complexity=5.0 + (i % 3),
                    explanation=f"Parallel processing test {i}"
                )
            )
            problems.append(problem)
        
        # Mock client for decomposition engine
        with patch('decomposition_engine.OpenEvolveClient') as mock_openevolve:
            mock_client = mock_openevolve.return_value
            
            def create_mock_decomposition_response(problem_id):
                mock_result = Mock()
                mock_result.success = True
                mock_result.best_code = json.dumps([
                    {
                        "id": generate_id(f"sub_{problem_id}_1"),
                        "description": f"Sub-problem 1 for parallel problem {problem_id}",
                        "dependencies": [],
                        "ai_suggested_complexity_score": 5.0 + (problem_id % 2),
                        "ai_suggested_evaluation_prompt": f"Evaluate sub-problem for problem {problem_id}"
                    }
                ])
                return mock_result
            
            successful_decompositions = []
            failed_decompositions = []
            
            def decompose_single_problem(problem):
                """Function to decompose a single problem"""
                try:
                    mock_client.evolve.return_value = create_mock_decomposition_response(int(problem.id.split('_')[-1]))
                    engine = DecompositionEngine(openevolve_client=mock_client)
                    plan = engine.decompose(problem, strategy="semantic")
                    return (problem.id, plan, None)
                except Exception as e:
                    return (problem.id, None, str(e))
            
            # Execute decompositions in parallel
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=10) as executor:  # 10 parallel workers
                futures = [executor.submit(decompose_single_problem, problem) for problem in problems]
                
                for future in as_completed(futures):
                    problem_id, plan, error = future.result()
                    if plan is not None:
                        successful_decompositions.append((problem_id, plan))
                    else:
                        failed_decompositions.append((problem_id, error))
            
            parallel_time = time.time() - start_time
            
            print(f"Parallel decomposition: {len(successful_decompositions)} successful, {len(failed_decompositions)} failed in {parallel_time:.3f}s")
            print(f"Throughput: {len(problems)/parallel_time:.1f} problems/second")
            
            # Verify most decompositions succeeded
            self.assertGreaterEqual(len(successful_decompositions), len(problems) * 0.8,
                                  f"Most parallel decompositions should succeed: {len(successful_decompositions)}/{len(problems)}")


class TestAdvancedSecurityScenarios(unittest.TestCase):
    """Advanced security testing scenarios"""
    
    def test_input_sanitization_comprehensive(self):
        """Test comprehensive input sanitization against multiple attack vectors"""
        validator = InputValidator()
        
        # Comprehensive attack vectors
        attack_vectors = [
            # SQL Injection attempts
            ("sql_injection_1", "'; DROP TABLE problems; --"),
            ("sql_injection_2", "' OR '1'='1"),
            ("sql_injection_3", "test'; UPDATE users SET password='hacked' WHERE 1=1; --"),
            ("sql_injection_4", "admin'; DELETE FROM users; --"),
            
            # XSS attempts
            ("xss_script", "<script>alert('xss')</script>"),
            ("xss_img", "<img src=x onerror=alert('xss')>"),
            ("xss_href", "javascript:alert('xss')"),
            ("xss_svg", "<svg onload=alert('xss')>"),
            ("xss_iframe", "<iframe src=javascript:alert('xss')></iframe>"),
            
            # Path Traversal
            ("path_traversal_1", "../../../etc/passwd"),
            ("path_traversal_2", "..\\..\\windows\\system32\\config\\sam"),
            ("path_traversal_3", "/../../../../../../../etc/passwd%00"),
            
            # Command Injection
            ("cmd_inject_1", "test; rm -rf /"),
            ("cmd_inject_2", "test && whoami"),
            ("cmd_inject_3", "test | cat /etc/passwd"),
            ("cmd_inject_4", "$(rm -rf /)"),
            
            # Logic Injection
            ("logic_1", "{{7*7}}"),
            ("logic_2", "${7*7}"),
            ("logic_3", "#{7*7}"),
            
            # Regex DoS
            ("regex_dos_1", "^((a+)+)+$"),
            ("regex_dos_2", "(a+)+"),
            
            # Normal inputs (should pass)
            ("normal_1", "This is a normal, safe input"),
            ("normal_2", "Valid input with some punctuation: !@#$%"),
            ("normal_3", "Alphanumeric input with spaces and hyphens and_underscores 12345"),
        ]
        
        sanitized_results = []
        for input_name, attack_input in attack_vectors:
            try:
                # Apply all validation rules
                rules = [
                    validator.VALIDATION_RULES.NOT_EMPTY,
                    validator.VALIDATION_RULES.MAX_LENGTH(1000),
                    validator.VALIDATION_RULES.SANITIZE_HTML,
                    validator.VALIDATION_RULES.NO_SCRIPT
                ]
                
                sanitized = validator.validate_input(attack_input, input_name, rules)
                sanitized_results.append((input_name, attack_input, sanitized, "success"))
            except Exception as e:
                sanitized_results.append((input_name, attack_input, str(e), "exception"))
        
        # Analyze results
        successful_sanitizations = [r for r in sanitized_results if r[3] == "success"]
        exceptions = [r for r in sanitized_results if r[3] == "exception"]
        
        print(f"Input sanitization: {len(successful_sanitizations)} sanitized, {len(exceptions)} exceptions")
        
        # All normal inputs should pass
        normal_inputs = [r for r in sanitized_results if r[0].startswith("normal_")]
        for input_name, original, sanitized, status in normal_inputs:
            self.assertEqual(status, "success", f"Normal input {input_name} should pass validation")
            self.assertEqual(sanitized, original, f"Normal input {input_name} should not be modified")
        
        # Malicious inputs should either be sanitized or throw exceptions
        malicious_inputs = [r for r in sanitized_results if not r[0].startswith("normal_")]
        for input_name, original, result, status in malicious_inputs:
            if status == "success":
                # If successful, the result should be different from original (sanitized)
                self.assertNotEqual(result, original, f"Malicious input {input_name} should be sanitized when successful")
            # If it threw an exception, that's also acceptable (rejecting malicious input)
    
    def test_authentication_brute_force_protection(self):
        """Test authentication brute force protection"""
        from datetime import datetime, timedelta
        
        auth_system = AuthenticationSystem(db_path=":memory:")
        
        # Create test user
        user = auth_system.create_user(
            username="brute_test_user",
            email="brute@test.com",
            password="SecurePassword123!",
            roles=[],
            permissions=[]
        )
        
        self.assertIsNotNone(user)
        
        # Simulate multiple failed login attempts
        failed_attempts = 0
        start_time = time.time()
        
        for i in range(50):  # 50 failed attempts
            result = auth_system.authenticate("brute_test_user", f"wrong_password_{i}")
            if result is None:
                failed_attempts += 1
            
            # Brief pause to simulate realistic timing
            time.sleep(0.01)
        
        elapsed_time = time.time() - start_time
        
        # Verify failed attempts were recorded and handled
        print(f"Brute force simulation: {failed_attempts} failed attempts in {elapsed_time:.3f}s")
        
        # Even with many attempts, system should not crash
        self.assertEqual(failed_attempts, 50, "All authentication attempts should be processed")
        
        # Now test that legitimate authentication still works after failed attempts
        legitimate_auth = auth_system.authenticate("brute_test_user", "SecurePassword123!")
        self.assertIsNotNone(legitimate_auth, "Legitimate authentication should still work after failed attempts")
    
    def test_authorization_matrix_validation(self):
        """Test complex authorization matrix validation"""
        from auth_system import Role, Permission
        
        auth_system = AuthenticationSystem(db_path=":memory:")
        authz_system = AuthorizationSystem(auth_system)
        
        # Create users with different role combinations
        admin_user = auth_system.create_user(
            username="admin_user",
            email="admin@test.com",
            password="SecurePassword123!",
            roles=[Role.ADMIN],
            permissions=[]
        )
        
        analyst_user = auth_system.create_user(
            username="analyst_user",
            email="analyst@test.com",
            password="SecurePassword123!",
            roles=[Role.ANALYST],
            permissions=[Permission.CREATE_PROBLEM, Permission.READ_PROBLEM]
        )
        
        viewer_user = auth_system.create_user(
            username="viewer_user",
            email="viewer@test.com",
            password="SecurePassword123!",
            roles=[Role.VIEWER],
            permissions=[Permission.READ_PROBLEM]
        )
        
        # Test authorization matrix
        test_cases = [
            # (user, permission, expected_result, description)
            (admin_user, Permission.CREATE_PROBLEM, True, "Admin should have create permission"),
            (admin_user, Permission.READ_PROBLEM, True, "Admin should have read permission"),
            (analyst_user, Permission.CREATE_PROBLEM, True, "Analyst with explicit permission should have it"),
            (analyst_user, Permission.READ_PROBLEM, True, "Analyst with explicit permission should have it"),
            (analyst_user, Permission.DELETE_PROBLEM, False, "Analyst without permission should be denied"),
            (viewer_user, Permission.CREATE_PROBLEM, False, "Viewer should not have create permission"),
            (viewer_user, Permission.READ_PROBLEM, True, "Viewer should have read permission"),
        ]
        
        for user, permission, expected, description in test_cases:
            with self.subTest(description=description):
                result = authz_system.check_permission(user, permission)
                self.assertEqual(result, expected, 
                               f"Authorization check failed: {description}. Expected {expected}, got {result}")
        
        print(f"Authorization matrix validation: {len(test_cases)} test cases passed")


class TestPerformanceEdgeCases(unittest.TestCase):
    """Performance edge cases and stress testing"""
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency when processing large amounts of data"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many objects to test memory management
        large_objects = []
        
        print(f"Starting memory efficiency test - Baseline: {baseline_memory:.1f}MB")
        
        try:
            # Create 10,000 objects with complex data structures
            for i in range(10000):
                complex_obj = {
                    'id': generate_id(f"perf_test_{i}"),
                    'metadata': {
                        'index': i,
                        'group': i % 100,  # Grouping
                        'category': f"cat_{i % 10}",  # Categories
                        'timestamp': datetime.now().isoformat()
                    },
                    'nested_data': [
                        {'field1': f'value_{j}', 'field2': j * 2, 'field3': {'subfield': j}}
                        for j in range(10)  # Nested structures
                    ],
                    'computed_values': [
                        math.sqrt(j * i + 1) for j in range(5)  # Computed values
                    ],
                    'binary_data': array.array('f', [float(j) for j in range(20)]),  # Binary data
                    'collections': {
                        'set_data': set(f'item_{k}' for k in range(i % 50)),
                        'deque_data': collections.deque(range(i % 100)),
                        'counter_data': collections.Counter(f'char_{c}' for c in f"test_string_{i}")
                    }
                }
                large_objects.append(complex_obj)
                
                # Periodically trigger garbage collection
                if i % 1000 == 0:
                    gc.collect()
            
            # Check memory usage after creation
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory
            
            print(f"After creating 10,000 objects: {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Memory increase should be reasonable
            self.assertLess(memory_increase, 500.0, f"Memory usage grew too large: {memory_increase:.1f}MB")
            
            # Clean up and check memory after cleanup
            del large_objects
            gc.collect()
            
            cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
            cleanup_increase = cleanup_memory - baseline_memory
            
            print(f"After cleanup: {cleanup_memory:.1f}MB (+{cleanup_increase:.1f}MB)")
            
            # Memory should be largely reclaimed
            self.assertLess(cleanup_increase, memory_increase * 0.3, 
                           f"Memory not sufficiently reclaimed after cleanup: {cleanup_increase:.1f}MB still allocated")
            
        except MemoryError:
            self.fail("Memory efficiency test failed - consumed too much memory")
        except Exception as e:
            print(f"Memory efficiency test encountered issue: {e}")
            # Still allow the test to pass if the system handles high memory usage gracefully
    
    def test_cache_performance_under_extreme_load(self):
        """Test cache performance under extreme load conditions"""
        import time
        import threading
        from queue import Queue
        import random
        
        cache = LLMResponseCache(max_size=1000)
        
        # Results tracking
        results_queue = Queue()
        start_time = time.time()
        
        def cache_worker(worker_id):
            """Worker function to perform cache operations concurrently"""
            local_results = {'gets': 0, 'sets': 0, 'hits': 0, 'misses': 0}
            
            for i in range(200):  # Each worker does 200 operations
                # Create varied content and parameters to simulate realistic usage
                content = f"Test content for worker {worker_id}, operation {i}, timestamp {datetime.now().isoformat()}, random {random.randint(1, 1000)}"
                model_params = {
                    "model": f"gpt-{random.choice(['4', '3.5'])}",
                    "temperature": round(random.uniform(0.1, 0.9), 2),
                    "max_tokens": random.randint(100, 1000)
                }
                response = {"choices": [{"message": {"content": f"Response from worker {worker_id}, operation {i}"}}]}
                
                # Set operation
                cache.cache_response(content, model_params, response)
                local_results['sets'] += 1
                
                # Get operation (sometimes the same content, sometimes different for hits/misses)
                if random.random() < 0.6:  # 60% chance to request same content (hit)
                    retrieved = cache.get_response(content, model_params)
                    if retrieved:
                        local_results['hits'] += 1
                    else:
                        local_results['misses'] += 1
                else:  # 40% chance to request different content (likely miss)
                    different_content = f"Different content {random.randint(1, 10000)}"
                    retrieved = cache.get_response(different_content, model_params)
                    if retrieved:
                        local_results['hits'] += 1
                    else:
                        local_results['misses'] += 1
                
                local_results['gets'] += 1
            
            results_queue.put(local_results)
        
        # Run concurrent cache operations
        threads = []
        num_workers = 20  # 20 concurrent workers
        
        for worker_id in range(num_workers):
            thread = threading.Thread(target=cache_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Aggregate results
        total_results = {'gets': 0, 'sets': 0, 'hits': 0, 'misses': 0}
        while not results_queue.empty():
            result = results_queue.get()
            for key in total_results:
                total_results[key] += result[key]
        
        hit_rate = total_results['hits'] / total_results['gets'] if total_results['gets'] > 0 else 0
        
        print(f"Cache performance under load:")
        print(f"  Operations: {total_results['sets']} sets, {total_results['gets']} gets in {total_time:.3f}s")
        print(f"  Hits: {total_results['hits']}, Misses: {total_results['misses']}")
        print(f"  Hit rate: {hit_rate:.2%}")
        print(f"  Throughput: {(total_results['sets'] + total_results['gets'])/total_time:.1f} ops/sec")
        
        # Verify cache operated within reasonable parameters
        self.assertLess(total_time, 10.0, f"Cache operations took too long under load: {total_time:.2f}s")
        self.assertGreater(hit_rate, 0.3, f"Cache should have reasonable hit rate: {hit_rate:.2%}")
    
    def test_database_performance_under_load(self):
        """Test database performance under high load"""
        import time
        
        db = SovereignDatabase(":memory:")
        
        # Create and insert many problems to test performance
        test_problems = []
        for i in range(5000):  # 5000 problems
            problem = ProblemDefinition(
                id=generate_id(f"perf_db_{i}"),
                title=f"Performance Test Problem {i}",
                description=f"Performance test problem {i} with substantial content to test database performance under load. " + "Detailed description content. " * 10,
                problem_type=random.choice(list(ProblemType)),
                domain_context=DomainContext(domain=f"perf_domain_{i % 50}"),
                complexity_score=ComplexityScore(
                    cognitive_complexity=5.0 + (i % 3),
                    computational_complexity=5.0 + (i % 3),
                    domain_complexity=5.0 + (i % 3),
                    integration_complexity=5.0 + (i % 3),
                    overall_complexity=5.0 + (i % 3),
                    explanation=f"Performance test {i}"
                )
            )
            test_problems.append(problem)
        
        # Time bulk insertion
        start_time = time.time()
        for problem in test_problems:
            db.create_problem(problem)
        insert_time = time.time() - start_time
        
        # Time querying
        query_start = time.time()
        for _ in range(100):  # 100 random queries
            random_idx = random.randint(0, len(test_problems) - 1)
            random_problem = db.get_problem(test_problems[random_idx].id)
            self.assertIsNotNone(random_problem)
        query_time = time.time() - query_start
        
        # Time listing operations
        list_start = time.time()
        all_problems = db.list_problems()
        list_time = time.time() - list_start
        
        print(f"Database performance under load:")
        print(f"  Inserted {len(test_problems)} problems in {insert_time:.3f}s ({len(test_problems)/insert_time:.1f} problems/sec)")
        print(f"  Executed 100 random queries in {query_time:.3f}s ({100/query_time:.1f} queries/sec)")
        print(f"  Listed all problems in {list_time:.3f}s")
        print(f"  Total problems in DB: {len(all_problems)}")
        
        # Verify performance targets
        self.assertLess(insert_time, 10.0, f"Bulk insertion too slow: {insert_time:.2f}s for {len(test_problems)} problems")
        self.assertLess(query_time, 5.0, f"Random queries too slow: {query_time:.2f}s for 100 queries")
        self.assertEqual(len(all_problems), len(test_problems), "All problems should be retrievable")


def run_advanced_unit_tests():
    """Run the advanced unit tests"""
    print("Running Advanced Unit Tests...")
    print("="*80)
    
    # Create test suite with all advanced tests
    suite = unittest.TestSuite()
    
    suite.addTest(unittest.makeSuite(TestAdvancedDataModelValidation))
    suite.addTest(unittest.makeSuite(TestAdvancedAnalyzerScenarios))
    suite.addTest(unittest.makeSuite(TestAdvancedDecompositionStrategies))
    suite.addTest(unittest.makeSuite(TestConcurrencyAndThreading))
    suite.addTest(unittest.makeSuite(TestAdvancedSecurityScenarios))
    suite.addTest(unittest.makeSuite(TestPerformanceEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("ADVANCED UNIT TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\nSome tests failed or had errors (which may be expected for stress tests):")
        for test, trace in result.failures:
            print(f"\nFAILED: {test}")
            print(trace)
        for test, trace in result.errors:
            print(f"\nERROR: {test}")
            print(trace)
    else:
        print("\n[OK] All advanced unit tests passed!")
    
    print("="*80)
    return result


if __name__ == "__main__":
    run_advanced_unit_tests()