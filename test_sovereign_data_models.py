"""
Tests for Sovereign Data Models
Task 1.4: Unit tests for data models
"""

import pytest
from datetime import datetime
import json

from sovereign_data_models import (
    ProblemType, SubProblemType, DecompositionStrategy, SubProblemStatus, PlanStatus,
    Constraint, SuccessCriterion, DomainContext, ComplexityScore,
    ProblemDefinition, SubProblem, DecompositionPlan, SolutionAttempt,
    Pattern, TeamAssignment, Feedback, ValidationResult, QualityScores,
    DependencyGraph, ValidationCheckpoint
)


class TestEnums:
    """Test enum classes"""
    
    def test_problem_type_enum(self):
        assert ProblemType.RESEARCH.value == "research"
        assert ProblemType.IMPLEMENTATION.value == "implementation"
    
    def test_decomposition_strategy_enum(self):
        assert DecompositionStrategy.SEMANTIC.value == "semantic"
        assert DecompositionStrategy.HYBRID.value == "hybrid"
    
    def test_status_enums(self):
        assert SubProblemStatus.PENDING.value == "pending"
        assert PlanStatus.DRAFT.value == "draft"


class TestConstraint:
    """Test Constraint model"""
    
    def test_create_constraint(self):
        constraint = Constraint(
            id="c1",
            description="Must complete in 2 weeks",
            type="time",
            severity="hard"
        )
        assert constraint.id == "c1"
        assert constraint.type == "time"
    
    def test_constraint_serialization(self):
        constraint = Constraint(
            id="c1",
            description="Test constraint",
            type="resource",
            severity="soft",
            metadata={"key": "value"}
        )
        data = constraint.to_dict()
        assert data['id'] == "c1"
        assert data['metadata']['key'] == "value"
        
        restored = Constraint.from_dict(data)
        assert restored.id == constraint.id
        assert restored.metadata == constraint.metadata


class TestSuccessCriterion:
    """Test SuccessCriterion model"""
    
    def test_create_success_criterion(self):
        criterion = SuccessCriterion(
            id="sc1",
            description="Accuracy > 95%",
            metric="accuracy",
            threshold=0.95,
            validation_method="automated_test"
        )
        assert criterion.threshold == 0.95
        assert criterion.metric == "accuracy"
    
    def test_success_criterion_serialization(self):
        criterion = SuccessCriterion(
            id="sc1",
            description="Test criterion",
            metric="performance",
            threshold=100.0,
            validation_method="benchmark"
        )
        data = criterion.to_dict()
        restored = SuccessCriterion.from_dict(data)
        assert restored.threshold == criterion.threshold


class TestComplexityScore:
    """Test ComplexityScore model"""
    
    def test_create_complexity_score(self):
        score = ComplexityScore(
            cognitive_complexity=7.5,
            computational_complexity=6.0,
            domain_complexity=8.0,
            integration_complexity=5.5,
            overall_complexity=6.75,
            explanation="High cognitive load due to abstract concepts"
        )
        assert score.overall_complexity == 6.75
        assert score.cognitive_complexity == 7.5
    
    def test_complexity_score_serialization(self):
        score = ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=5.0,
            domain_complexity=5.0,
            integration_complexity=5.0,
            overall_complexity=5.0,
            explanation="Medium complexity"
        )
        data = score.to_dict()
        restored = ComplexityScore.from_dict(data)
        assert restored.overall_complexity == score.overall_complexity


class TestDomainContext:
    """Test DomainContext model"""
    
    def test_create_domain_context(self):
        context = DomainContext(
            domain="machine_learning",
            subdomain="natural_language_processing",
            related_domains=["linguistics", "statistics"],
            domain_knowledge={"frameworks": ["pytorch", "tensorflow"]}
        )
        assert context.domain == "machine_learning"
        assert len(context.related_domains) == 2
    
    def test_domain_context_serialization(self):
        context = DomainContext(
            domain="test_domain",
            subdomain="test_subdomain"
        )
        data = context.to_dict()
        restored = DomainContext.from_dict(data)
        assert restored.domain == context.domain


class TestProblemDefinition:
    """Test ProblemDefinition model"""
    
    def test_create_problem_definition(self):
        domain = DomainContext(domain="software_engineering")
        complexity = ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=5.0,
            domain_complexity=5.0,
            integration_complexity=5.0,
            overall_complexity=5.0,
            explanation="Medium"
        )
        
        problem = ProblemDefinition(
            id="p1",
            title="Test Problem",
            description="A test problem",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=domain,
            complexity_score=complexity
        )
        
        assert problem.id == "p1"
        assert problem.problem_type == ProblemType.IMPLEMENTATION
        assert problem.domain_context.domain == "software_engineering"
    
    def test_problem_definition_serialization(self):
        domain = DomainContext(domain="test")
        complexity = ComplexityScore(
            cognitive_complexity=1.0,
            computational_complexity=1.0,
            domain_complexity=1.0,
            integration_complexity=1.0,
            overall_complexity=1.0,
            explanation="Low"
        )
        
        problem = ProblemDefinition(
            id="p1",
            title="Test",
            description="Test problem",
            problem_type=ProblemType.RESEARCH,
            domain_context=domain,
            complexity_score=complexity,
            stakeholders=["user1", "user2"]
        )
        
        data = problem.to_dict()
        assert data['problem_type'] == "research"
        assert len(data['stakeholders']) == 2
        
        restored = ProblemDefinition.from_dict(data)
        assert restored.id == problem.id
        assert restored.problem_type == problem.problem_type
        assert len(restored.stakeholders) == 2


class TestSubProblem:
    """Test SubProblem model"""
    
    def test_create_sub_problem(self):
        complexity = ComplexityScore(
            cognitive_complexity=3.0,
            computational_complexity=3.0,
            domain_complexity=3.0,
            integration_complexity=3.0,
            overall_complexity=3.0,
            explanation="Low-medium"
        )
        
        sub_problem = SubProblem(
            id="sp1",
            parent_id="p1",
            title="Sub-problem 1",
            description="First sub-problem",
            type=SubProblemType.ANALYSIS,
            complexity_score=complexity,
            dependencies=["sp2", "sp3"]
        )
        
        assert sub_problem.id == "sp1"
        assert sub_problem.type == SubProblemType.ANALYSIS
        assert len(sub_problem.dependencies) == 2
        assert sub_problem.status == SubProblemStatus.PENDING
    
    def test_sub_problem_serialization(self):
        complexity = ComplexityScore(
            cognitive_complexity=2.0,
            computational_complexity=2.0,
            domain_complexity=2.0,
            integration_complexity=2.0,
            overall_complexity=2.0,
            explanation="Low"
        )
        
        sub_problem = SubProblem(
            id="sp1",
            parent_id="p1",
            title="Test",
            description="Test sub-problem",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=complexity
        )
        
        data = sub_problem.to_dict()
        assert data['type'] == "implementation"
        assert data['status'] == "pending"
        
        restored = SubProblem.from_dict(data)
        assert restored.id == sub_problem.id
        assert restored.type == sub_problem.type


class TestDecompositionPlan:
    """Test DecompositionPlan model"""
    
    def test_create_decomposition_plan(self):
        plan = DecompositionPlan(
            id="plan1",
            problem_id="p1",
            strategy=DecompositionStrategy.SEMANTIC,
            confidence_level=0.85
        )
        
        assert plan.id == "plan1"
        assert plan.strategy == DecompositionStrategy.SEMANTIC
        assert plan.status == PlanStatus.DRAFT
        assert plan.confidence_level == 0.85
    
    def test_decomposition_plan_with_sub_problems(self):
        complexity = ComplexityScore(
            cognitive_complexity=2.0,
            computational_complexity=2.0,
            domain_complexity=2.0,
            integration_complexity=2.0,
            overall_complexity=2.0,
            explanation="Low"
        )
        
        sub_problem = SubProblem(
            id="sp1",
            parent_id="p1",
            title="Test",
            description="Test",
            type=SubProblemType.ANALYSIS,
            complexity_score=complexity
        )
        
        plan = DecompositionPlan(
            id="plan1",
            problem_id="p1",
            strategy=DecompositionStrategy.DEPENDENCY,
            sub_problems=[sub_problem]
        )
        
        assert len(plan.sub_problems) == 1
        assert plan.sub_problems[0].id == "sp1"
    
    def test_decomposition_plan_serialization(self):
        plan = DecompositionPlan(
            id="plan1",
            problem_id="p1",
            strategy=DecompositionStrategy.COMPLEXITY,
            confidence_level=0.9
        )
        
        data = plan.to_dict()
        assert data['strategy'] == "complexity"
        assert data['status'] == "draft"
        
        restored = DecompositionPlan.from_dict(data)
        assert restored.id == plan.id
        assert restored.strategy == plan.strategy


class TestDependencyGraph:
    """Test DependencyGraph model"""
    
    def test_create_dependency_graph(self):
        graph = DependencyGraph(
            edges={"sp1": ["sp2", "sp3"], "sp2": [], "sp3": []},
            critical_path=["sp1", "sp2"],
            execution_order=["sp2", "sp3", "sp1"]
        )
        
        assert len(graph.edges) == 3
        assert len(graph.critical_path) == 2
        assert graph.execution_order[0] == "sp2"
    
    def test_dependency_graph_serialization(self):
        graph = DependencyGraph(
            edges={"a": ["b"], "b": []},
            execution_order=["b", "a"]
        )
        
        data = graph.to_dict()
        assert "edges" in data
        assert "execution_order" in data


class TestQualityScores:
    """Test QualityScores model"""
    
    def test_create_quality_scores(self):
        scores = QualityScores(
            coherence_score=0.9,
            completeness_score=0.85,
            feasibility_score=0.8,
            integration_score=0.88,
            overall_score=0.86,
            meets_thresholds=True
        )
        
        assert scores.overall_score == 0.86
        assert scores.meets_thresholds is True
    
    def test_quality_scores_serialization(self):
        scores = QualityScores(
            coherence_score=0.7,
            completeness_score=0.7,
            feasibility_score=0.7,
            integration_score=0.7,
            overall_score=0.7,
            meets_thresholds=False
        )
        
        data = scores.to_dict()
        restored = QualityScores.from_dict(data)
        assert restored.overall_score == scores.overall_score


class TestPattern:
    """Test Pattern model"""
    
    def test_create_pattern(self):
        pattern = Pattern(
            id="pat1",
            problem_type=ProblemType.RESEARCH,
            strategy=DecompositionStrategy.SEMANTIC,
            pattern_description="Semantic clustering for research problems",
            success_rate=0.85,
            usage_count=10,
            avg_quality_score=0.82,
            applicable_domains=["machine_learning", "data_science"]
        )
        
        assert pattern.success_rate == 0.85
        assert pattern.usage_count == 10
        assert len(pattern.applicable_domains) == 2
    
    def test_pattern_serialization(self):
        pattern = Pattern(
            id="pat1",
            problem_type=ProblemType.IMPLEMENTATION,
            strategy=DecompositionStrategy.DEPENDENCY,
            pattern_description="Test pattern",
            success_rate=0.9,
            usage_count=5,
            avg_quality_score=0.88
        )
        
        data = pattern.to_dict()
        assert data['problem_type'] == "implementation"
        assert data['strategy'] == "dependency"
        
        restored = Pattern.from_dict(data)
        assert restored.success_rate == pattern.success_rate


class TestValidationResult:
    """Test ValidationResult model"""
    
    def test_create_validation_result(self):
        result = ValidationResult(
            validator="coherence_gauntlet",
            passed=True,
            score=0.92,
            feedback="Excellent coherence",
            improvements=["Consider edge case X"]
        )
        
        assert result.passed is True
        assert result.score == 0.92
        assert len(result.improvements) == 1
    
    def test_validation_result_serialization(self):
        result = ValidationResult(
            validator="test_gauntlet",
            passed=False,
            score=0.5,
            feedback="Needs improvement"
        )
        
        data = result.to_dict()
        restored = ValidationResult.from_dict(data)
        assert restored.passed == result.passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
