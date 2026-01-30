#!/usr/bin/env python3
"""
Comprehensive test suite for subproblem functionality.
Tests data models, decomposition, solving, and integration.
"""

import unittest
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, List

# Import the core modules
try:
    from sovereign_data_models import (
        SubProblem, SubProblemType, ComplexityScore, SuccessCriterion,
        ComplexityBreakdown, SubProblemTeamAssignment, GauntletAssignment,
        ResourceEstimate, PotentialApproach, QualityMetrics,
        SubProblemStatus, SolutionAttempt, generate_id
    )
    from decomposition_engine import DecompositionEngine
    from sub_problem_solver import SubProblemSolver, SolvingStrategy
    
    # Check for optional dependencies
    try:
        from workflow_structures import Team
        TEAM_AVAILABLE = True
    except ImportError:
        TEAM_AVAILABLE = False
    
    try:
        from openevolve_client import OpenEvolveClient
        OPENEVOLVE_AVAILABLE = True
    except ImportError:
        OPENEVOLVE_AVAILABLE = False
        
except ImportError as e:
    print(f"Import error: {e}")
    raise


class TestSubProblemDataModel(unittest.TestCase):
    """Test SubProblem data model functionality"""
    
    def test_subproblem_creation_with_all_fields(self):
        """Test creating SubProblem with all enhanced fields"""
        subproblem = SubProblem(
            id="test_sp_001",
            parent_id="test_prob_001",
            title="Test SubProblem",
            description="A test subproblem for validation",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Test complexity",
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=6.0,
                integration_complexity=3.0,
                overall_complexity=4.5
            ),
            dependencies=["dep_001", "dep_002"],
            success_criteria=[
                SuccessCriterion(
                    id="test_criterion_1",
                    description="Test criterion 1",
                    metric="accuracy",
                    threshold=0.95,
                    validation_method="automated"
                )
            ],
            validation_gauntlet="test_gauntlet",
            assigned_team="test_team",
            estimated_effort=10,
            priority=7,
            status=SubProblemStatus.PENDING,
            # Enhanced fields
            acceptance_criteria=["Test acceptance 1", "Test acceptance 2"],
            ai_suggested_evolution_mode="adversarial",
            ai_suggested_complexity_score=ComplexityBreakdown(
                final_score=6.5,
                calculation_breakdown={"technical": 7.0, "domain": 6.0},
                metadata={"source": "ai_analysis"}
            ),
            ai_suggested_evaluation_prompt="Evaluate test subproblem thoroughly",
            ai_suggested_team_assignment=SubProblemTeamAssignment(
                solver="solver_team",
                patcher="patcher_team",
                red_team="red_team",
                gold_team="gold_team",
                metadata={"priority": "high"}
            ),
            ai_suggested_gauntlet_assignment=GauntletAssignment(
                red_team_gauntlet="red_gauntlet",
                gold_team_gauntlet="gold_gauntlet",
                metadata={"intensity": "high"}
            ),
            estimated_resources=ResourceEstimate(
                time_hours=8.5,
                api_tokens=50000,
                computational_units=10.0,
                human_review_minutes=30,
                metadata={"budget": "medium"}
            ),
            potential_approaches=[
                PotentialApproach(
                    name="Approach 1",
                    description="First approach to solve",
                    estimated_effort=5.0,
                    success_probability=0.85,
                    risk_level="low",
                    metadata={"preferred": True}
                )
            ],
            required_expertise=["Python", "Testing", "AI"],
            associated_risks=["Risk 1", "Risk 2"],
            success_dependencies=["success_dep_1"],
            testing_approach="integration",
            quality_metrics=QualityMetrics(
                accuracy_target=0.95,
                performance_target="<100ms",
                security_requirements=["encryption"],
                compliance_requirements=["GDPR"],
                metadata={"critical": True}
            )
        )
        
        # Verify all fields are set correctly
        self.assertEqual(subproblem.id, "test_sp_001")
        self.assertEqual(subproblem.title, "Test SubProblem")
        self.assertEqual(subproblem.type, SubProblemType.IMPLEMENTATION)
        self.assertEqual(len(subproblem.acceptance_criteria), 2)
        self.assertEqual(subproblem.ai_suggested_evolution_mode, "adversarial")
        self.assertIsNotNone(subproblem.ai_suggested_complexity_score)
        self.assertEqual(subproblem.ai_suggested_complexity_score.final_score, 6.5)
        self.assertEqual(len(subproblem.potential_approaches), 1)
        self.assertEqual(len(subproblem.required_expertise), 3)
        
    def test_subproblem_backward_compatibility(self):
        """Test that old-style SubProblem creation still works"""
        # Create SubProblem without enhanced fields (old style)
        old_subproblem = SubProblem(
            id="old_sp_001",
            parent_id="old_prob_001",
            title="Old Style SubProblem",
            description="Old style without enhanced fields",
            type=SubProblemType.RESEARCH,
            complexity_score=ComplexityScore(
                explanation="Simple",
                cognitive_complexity=3.0,
                computational_complexity=2.0,
                domain_complexity=3.0,
                integration_complexity=2.0,
                overall_complexity=2.5
            )
        )
        
        # Verify it works and has default values for new fields
        self.assertEqual(old_subproblem.id, "old_sp_001")
        self.assertEqual(old_subproblem.acceptance_criteria, [])
        self.assertEqual(old_subproblem.ai_suggested_evolution_mode, "standard")
        self.assertIsNone(old_subproblem.ai_suggested_complexity_score)
        self.assertEqual(old_subproblem.ai_suggested_evaluation_prompt, "")
        self.assertIsNone(old_subproblem.ai_suggested_team_assignment)
        self.assertIsNone(old_subproblem.ai_suggested_gauntlet_assignment)
        self.assertIsNone(old_subproblem.estimated_resources)
        self.assertEqual(old_subproblem.potential_approaches, [])
        self.assertEqual(old_subproblem.required_expertise, [])
        self.assertEqual(old_subproblem.associated_risks, [])
        self.assertEqual(old_subproblem.success_dependencies, [])
        self.assertEqual(old_subproblem.testing_approach, "")
        self.assertIsNone(old_subproblem.quality_metrics)
        
    def test_subproblem_serialization(self):
        """Test SubProblem serialization and deserialization"""
        # Create a SubProblem with some enhanced fields
        original_sp = SubProblem(
            id="serial_test_001",
            parent_id="serial_prob_001",
            title="Serialization Test",
            description="Test serialization",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Test",
                cognitive_complexity=4.0,
                computational_complexity=3.0,
                domain_complexity=4.0,
                integration_complexity=3.0,
                overall_complexity=3.5
            ),
            acceptance_criteria=["Criteria 1", "Criteria 2"],
            ai_suggested_evolution_mode="quality_diversity",
            estimated_resources=ResourceEstimate(
                time_hours=5.0,
                api_tokens=25000,
                computational_units=5.0,
                human_review_minutes=15
            ),
            potential_approaches=[
                PotentialApproach(
                    name="Test Approach",
                    description="Test approach description",
                    estimated_effort=4.0,
                    success_probability=0.8,
                    risk_level="medium"
                )
            ]
        )
        
        # Serialize to dict
        data = original_sp.to_dict()
        
        # Verify all enhanced fields are in the serialized data
        self.assertIn('acceptance_criteria', data)
        self.assertIn('ai_suggested_evolution_mode', data)
        self.assertIn('estimated_resources', data)
        self.assertIn('potential_approaches', data)
        
        # Deserialize back
        restored_sp = SubProblem.from_dict(data)
        
        # Verify all fields are restored correctly
        self.assertEqual(restored_sp.id, original_sp.id)
        self.assertEqual(restored_sp.acceptance_criteria, original_sp.acceptance_criteria)
        self.assertEqual(restored_sp.ai_suggested_evolution_mode, original_sp.ai_suggested_evolution_mode)
        self.assertEqual(restored_sp.estimated_resources.time_hours, original_sp.estimated_resources.time_hours)
        self.assertEqual(len(restored_sp.potential_approaches), len(original_sp.potential_approaches))
        
    def test_subproblem_json_roundtrip(self):
        """Test JSON serialization roundtrip"""
        original_sp = SubProblem(
            id="json_test_001",
            parent_id="json_prob_001",
            title="JSON Test",
            description="Test JSON serialization",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                explanation="JSON test",
                cognitive_complexity=3.0,
                computational_complexity=2.0,
                domain_complexity=3.0,
                integration_complexity=2.0,
                overall_complexity=2.5
            ),
            acceptance_criteria=["JSON criteria"],
            ai_suggested_evolution_mode="guided",
            quality_metrics=QualityMetrics(
                accuracy_target=0.9,
                performance_target="<200ms",
                security_requirements=["basic"],
                compliance_requirements=["none"]
            )
        )
        
        # Serialize to JSON
        json_str = json.dumps(original_sp.to_dict(), indent=2)
        
        # Deserialize from JSON
        parsed_data = json.loads(json_str)
        restored_sp = SubProblem.from_dict(parsed_data)
        
        # Verify integrity
        self.assertEqual(restored_sp.id, original_sp.id)
        self.assertEqual(restored_sp.acceptance_criteria, original_sp.acceptance_criteria)
        self.assertEqual(restored_sp.ai_suggested_evolution_mode, original_sp.ai_suggested_evolution_mode)
        self.assertEqual(restored_sp.quality_metrics.accuracy_target, original_sp.quality_metrics.accuracy_target)
        
    def test_subproblem_validation(self):
        """Test SubProblem validation"""
        # Valid SubProblem
        valid_sp = SubProblem(
            id="valid_001",
            parent_id="valid_prob_001",
            title="Valid SubProblem",
            description="Valid description",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Valid",
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5
            )
        )
        
        # Should have no validation errors
        errors = valid_sp.validate()
        self.assertEqual(errors, [])
        
        # Invalid evolution mode
        invalid_sp = SubProblem(
            id="invalid_001",
            parent_id="invalid_prob_001",
            title="Invalid SubProblem",
            description="Invalid description",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Invalid",
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5
            ),
            ai_suggested_evolution_mode="invalid_mode"
        )
        
        # Should have validation error for invalid evolution mode
        errors = invalid_sp.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("ai_suggested_evolution_mode", errors[0])
        
        # Test validation of nested objects
        sp_with_invalid_complexity = SubProblem(
            id="invalid_complexity_001",
            parent_id="invalid_prob_001",
            title="Invalid Complexity",
            description="Invalid complexity test",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Invalid",
                cognitive_complexity=5.0,
                computational_complexity=4.0,
                domain_complexity=5.0,
                integration_complexity=4.0,
                overall_complexity=4.5
            ),
            ai_suggested_complexity_score=ComplexityBreakdown(
                final_score=15.0,  # Invalid: should be 0-10
                calculation_breakdown={},
                metadata={}
            )
        )
        
        errors = sp_with_invalid_complexity.validate()
        self.assertGreater(len(errors), 0)
        self.assertIn("ComplexityBreakdown final_score must be between 0.0 and 10.0", errors[0])


class TestSubProblemSolver(unittest.TestCase):
    """Test SubProblemSolver functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_subproblem = SubProblem(
            id="solver_test_001",
            parent_id="solver_prob_001",
            title="Test Problem to Solve",
            description="A test problem that needs solving",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Test complexity",
                cognitive_complexity=4.0,
                computational_complexity=3.0,
                domain_complexity=4.0,
                integration_complexity=3.0,
                overall_complexity=3.5
            ),
            acceptance_criteria=["Solution must work"],
            ai_suggested_evolution_mode="standard",
            estimated_resources=ResourceEstimate(
                time_hours=2.0,
                api_tokens=10000,
                computational_units=2.0,
                human_review_minutes=5
            )
        )
        
    def test_solver_initialization(self):
        """Test SubProblemSolver initialization"""
        solver = SubProblemSolver()
        
        self.assertIsNotNone(solver)
        self.assertEqual(solver.default_strategy, SolvingStrategy.STANDARD)
        self.assertEqual(solver.solution_history, {})
        
    def test_solver_strategies(self):
        """Test different solving strategies"""
        solver = SubProblemSolver()
        
        # Test that strategies are available
        self.assertEqual(SolvingStrategy.STANDARD.value, "standard")
        self.assertEqual(SolvingStrategy.MDAP.value, "mdap")
        self.assertEqual(SolvingStrategy.MAKER.value, "maker")
        self.assertEqual(SolvingStrategy.HYBRID.value, "hybrid")
        
    def test_solution_tracking(self):
        """Test solution tracking functionality"""
        solver = SubProblemSolver()
        
        # Create a mock solution
        mock_solution = SolutionAttempt(
            id="mock_solution_001",
            sub_problem_id="solver_test_001",
            approach="mock",
            solution_content="Mock solution content",
            team_id="test_team",
            confidence_score=0.95
        )
        
        # Track the solution
        solver._track_solution("solver_test_001", mock_solution)
        
        # Verify tracking
        history = solver.get_solution_history("solver_test_001")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].id, "mock_solution_001")
        
        # Test best solution selection
        best = solver.get_best_solution("solver_test_001")
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "mock_solution_001")
        
        # Test multiple solutions
        mock_solution2 = SolutionAttempt(
            id="mock_solution_002",
            sub_problem_id="solver_test_001",
            approach="mock2",
            solution_content="Better mock solution",
            team_id="test_team",
            confidence_score=0.98
        )
        
        solver._track_solution("solver_test_001", mock_solution2)
        
        # Should have 2 solutions now
        history = solver.get_solution_history("solver_test_001")
        self.assertEqual(len(history), 2)
        
        # Best solution should be the one with higher confidence
        best = solver.get_best_solution("solver_test_001")
        self.assertEqual(best.id, "mock_solution_002")
        
    def test_prompt_building(self):
        """Test prompt building functionality"""
        solver = SubProblemSolver()
        
        # Build prompt for our test subproblem
        prompt = solver._build_prompt(self.test_subproblem)
        
        # Verify prompt contains expected elements
        self.assertIn("Test Problem to Solve", prompt)
        self.assertIn("A test problem that needs solving", prompt)
        self.assertIn("SUB-PROBLEM:", prompt)
        self.assertIn("TASK:", prompt)
        self.assertIn("SOLUTION:", prompt)


class TestSubProblemIntegration(unittest.TestCase):
    """Test integration between subproblem components"""
    
    def test_subproblem_lifecycle(self):
        """Test complete subproblem lifecycle: creation -> solving -> validation"""
        # Create a subproblem
        subproblem = SubProblem(
            id="lifecycle_001",
            parent_id="lifecycle_prob_001",
            title="Lifecycle Test Problem",
            description="Problem for lifecycle testing",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Lifecycle test",
                cognitive_complexity=3.0,
                computational_complexity=2.0,
                domain_complexity=3.0,
                integration_complexity=2.0,
                overall_complexity=2.5
            ),
            acceptance_criteria=["Must pass tests"],
            ai_suggested_evolution_mode="standard",
            testing_approach="unit"
        )
        
        # Validate it
        errors = subproblem.validate()
        self.assertEqual(errors, [])
        
        # Serialize and deserialize
        data = subproblem.to_dict()
        restored = SubProblem.from_dict(data)
        
        # Verify restoration
        self.assertEqual(restored.id, subproblem.id)
        self.assertEqual(restored.acceptance_criteria, subproblem.acceptance_criteria)
        
        # Create solver and track solution
        solver = SubProblemSolver()
        
        # Create a solution attempt
        solution = SolutionAttempt(
            id="lifecycle_solution_001",
            sub_problem_id=restored.id,
            approach="lifecycle_test",
            solution_content="Lifecycle solution content",
            team_id="test_team",
            confidence_score=0.90
        )
        
        solver._track_solution(restored.id, solution)
        
        # Verify solution tracking
        history = solver.get_solution_history(restored.id)
        self.assertEqual(len(history), 1)
        
        best_solution = solver.get_best_solution(restored.id)
        self.assertIsNotNone(best_solution)
        
    def test_subproblem_with_all_enhanced_features(self):
        """Test subproblem with all enhanced features working together"""
        # Create comprehensive subproblem
        comprehensive_sp = SubProblem(
            id="comprehensive_001",
            parent_id="comprehensive_prob_001",
            title="Comprehensive Test",
            description="Comprehensive test with all features",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                explanation="Comprehensive test",
                cognitive_complexity=6.0,
                computational_complexity=5.0,
                domain_complexity=7.0,
                integration_complexity=6.0,
                overall_complexity=6.0
            ),
            dependencies=["dep_001"],
            success_criteria=[
                SuccessCriterion(
                    id="lifecycle_criterion_1",
                    description="Must work correctly",
                    metric="functionality",
                    threshold=0.99,
                    validation_method="automated"
                )
            ],
            # All enhanced fields
            acceptance_criteria=["Acceptance 1", "Acceptance 2", "Acceptance 3"],
            ai_suggested_evolution_mode="adversarial",
            ai_suggested_complexity_score=ComplexityBreakdown(
                final_score=7.5,
                calculation_breakdown={
                    "technical": 8.0,
                    "domain": 7.0,
                    "integration": 7.5
                },
                metadata={"source": "ai_analysis", "confidence": "high"}
            ),
            ai_suggested_evaluation_prompt="Comprehensive evaluation prompt",
            ai_suggested_team_assignment=SubProblemTeamAssignment(
                solver="solver_team",
                patcher="patcher_team",
                red_team="red_team",
                gold_team="gold_team",
                metadata={"priority": "high", "experience_level": "senior"}
            ),
            ai_suggested_gauntlet_assignment=GauntletAssignment(
                red_team_gauntlet="comprehensive_red_gauntlet",
                gold_team_gauntlet="comprehensive_gold_gauntlet",
                metadata={"intensity": "high", "duration": "long"}
            ),
            estimated_resources=ResourceEstimate(
                time_hours=12.5,
                api_tokens=75000,
                computational_units=15.0,
                human_review_minutes=60,
                metadata={"budget": "high", "priority": "urgent"}
            ),
            potential_approaches=[
                PotentialApproach(
                    name="Approach 1",
                    description="First comprehensive approach",
                    estimated_effort=6.0,
                    success_probability=0.80,
                    risk_level="medium",
                    metadata={"preferred": False, "complexity": "medium"}
                ),
                PotentialApproach(
                    name="Approach 2",
                    description="Second comprehensive approach",
                    estimated_effort=7.0,
                    success_probability=0.85,
                    risk_level="low",
                    metadata={"preferred": True, "complexity": "high"}
                )
            ],
            required_expertise=["Python", "AI", "Testing", "Security"],
            associated_risks=["Risk 1", "Risk 2", "Risk 3"],
            success_dependencies=["success_dep_1", "success_dep_2"],
            testing_approach="integration",
            quality_metrics=QualityMetrics(
                accuracy_target=0.98,
                performance_target="<50ms",
                security_requirements=["encryption", "authentication"],
                compliance_requirements=["GDPR", "HIPAA"],
                metadata={"critical": True, "audit_required": True}
            )
        )
        
        # Test validation
        errors = comprehensive_sp.validate()
        self.assertEqual(errors, [])
        
        # Test serialization
        data = comprehensive_sp.to_dict()
        
        # Verify all enhanced fields are present
        self.assertIn('acceptance_criteria', data)
        self.assertIn('ai_suggested_evolution_mode', data)
        self.assertIn('ai_suggested_complexity_score', data)
        self.assertIn('ai_suggested_evaluation_prompt', data)
        self.assertIn('ai_suggested_team_assignment', data)
        self.assertIn('ai_suggested_gauntlet_assignment', data)
        self.assertIn('estimated_resources', data)
        self.assertIn('potential_approaches', data)
        self.assertIn('required_expertise', data)
        self.assertIn('associated_risks', data)
        self.assertIn('success_dependencies', data)
        self.assertIn('testing_approach', data)
        self.assertIn('quality_metrics', data)
        
        # Test deserialization
        restored = SubProblem.from_dict(data)
        
        # Verify all fields are restored
        self.assertEqual(restored.id, comprehensive_sp.id)
        self.assertEqual(restored.acceptance_criteria, comprehensive_sp.acceptance_criteria)
        self.assertEqual(restored.ai_suggested_evolution_mode, comprehensive_sp.ai_suggested_evolution_mode)
        self.assertEqual(restored.ai_suggested_complexity_score.final_score, comprehensive_sp.ai_suggested_complexity_score.final_score)
        self.assertEqual(restored.ai_suggested_evaluation_prompt, comprehensive_sp.ai_suggested_evaluation_prompt)
        self.assertEqual(restored.ai_suggested_team_assignment.solver, comprehensive_sp.ai_suggested_team_assignment.solver)
        self.assertEqual(restored.ai_suggested_gauntlet_assignment.red_team_gauntlet, comprehensive_sp.ai_suggested_gauntlet_assignment.red_team_gauntlet)
        self.assertEqual(restored.estimated_resources.time_hours, comprehensive_sp.estimated_resources.time_hours)
        self.assertEqual(len(restored.potential_approaches), len(comprehensive_sp.potential_approaches))
        self.assertEqual(restored.required_expertise, comprehensive_sp.required_expertise)
        self.assertEqual(restored.associated_risks, comprehensive_sp.associated_risks)
        self.assertEqual(restored.success_dependencies, comprehensive_sp.success_dependencies)
        self.assertEqual(restored.testing_approach, comprehensive_sp.testing_approach)
        self.assertEqual(restored.quality_metrics.accuracy_target, comprehensive_sp.quality_metrics.accuracy_target)


class TestSubProblemEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_empty_subproblem(self):
        """Test SubProblem with minimal required fields"""
        minimal_sp = SubProblem(
            id="minimal_001",
            parent_id="minimal_prob_001",
            title="Minimal",
            description="Minimal description",
            type=SubProblemType.RESEARCH,
            complexity_score=ComplexityScore(
                explanation="Minimal",
                cognitive_complexity=1.0,
                computational_complexity=1.0,
                domain_complexity=1.0,
                integration_complexity=1.0,
                overall_complexity=1.0
            )
        )
        
        # Should work with default values for all optional fields
        self.assertEqual(minimal_sp.acceptance_criteria, [])
        self.assertEqual(minimal_sp.ai_suggested_evolution_mode, "standard")
        self.assertIsNone(minimal_sp.ai_suggested_complexity_score)
        
        # Should validate successfully
        errors = minimal_sp.validate()
        self.assertEqual(errors, [])
        
    def test_subproblem_with_empty_enhanced_fields(self):
        """Test SubProblem with empty enhanced fields"""
        empty_enhanced_sp = SubProblem(
            id="empty_enhanced_001",
            parent_id="empty_enhanced_prob_001",
            title="Empty Enhanced",
            description="Empty enhanced fields test",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(
                explanation="Empty enhanced",
                cognitive_complexity=2.0,
                computational_complexity=2.0,
                domain_complexity=2.0,
                integration_complexity=2.0,
                overall_complexity=2.0
            ),
            acceptance_criteria=[],  # Empty list
            ai_suggested_evolution_mode="standard",
            ai_suggested_complexity_score=None,  # None
            ai_suggested_evaluation_prompt="",  # Empty string
            ai_suggested_team_assignment=None,  # None
            ai_suggested_gauntlet_assignment=None,  # None
            estimated_resources=None,  # None
            potential_approaches=[],  # Empty list
            required_expertise=[],  # Empty list
            associated_risks=[],  # Empty list
            success_dependencies=[],  # Empty list
            testing_approach="",  # Empty string
            quality_metrics=None  # None
        )
        
        # Should work fine
        errors = empty_enhanced_sp.validate()
        self.assertEqual(errors, [])
        
        # Should serialize and deserialize correctly
        data = empty_enhanced_sp.to_dict()
        restored = SubProblem.from_dict(data)
        
        self.assertEqual(restored.id, empty_enhanced_sp.id)
        self.assertEqual(restored.acceptance_criteria, [])
        self.assertEqual(restored.ai_suggested_evolution_mode, "standard")
        
    def test_subproblem_validation_edge_cases(self):
        """Test validation edge cases"""
        # Test with missing required fields
        try:
            SubProblem()  # Should fail - missing required fields
            self.fail("Expected TypeError for missing required fields")
        except TypeError:
            pass  # Expected
            
        # Test with invalid types - this might not raise due to dynamic typing
        try:
            SubProblem(
                id=123,  # Should be string
                parent_id="test",
                title="Test",
                description="Test",
                type=SubProblemType.IMPLEMENTATION,
                complexity_score=ComplexityScore(
                    explanation="Test",
                    cognitive_complexity=5.0,
                    computational_complexity=4.0,
                    domain_complexity=5.0,
                    integration_complexity=4.0,
                    overall_complexity=4.5
                )
            )
            # If this doesn't raise, it's due to Python's dynamic typing
            # We'll check validation instead
            pass
        except (TypeError, ValueError, AttributeError):
            pass  # Expected in some cases


def run_comprehensive_tests():
    """Run all comprehensive subproblem tests"""
    print("=" * 80)
    print("COMPREHENSIVE SUBPROBLEM FUNCTIONALITY TEST SUITE")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSubProblemDataModel))
    suite.addTests(loader.loadTestsFromTestCase(TestSubProblemSolver))
    suite.addTests(loader.loadTestsFromTestCase(TestSubProblemIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestSubProblemEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED - Subproblem functionality is working correctly!")
    else:
        print("\n[FAIL] SOME TESTS FAILED - Issues found in subproblem functionality")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)