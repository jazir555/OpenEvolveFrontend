"""
Tests for Sovereign Gauntlet System
Task 5.7: Unit tests for gauntlet integration
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from sovereign_data_models import (
    DecompositionPlan, SubProblem, DecompositionStrategy, SubProblemType,
    ComplexityScore, SuccessCriterion, DependencyGraph, generate_id
)
from sovereign_gauntlets import (
    CoherenceGauntlet, CompletenessGauntlet, FeasibilityGauntlet,
    DependencyGauntlet, GauntletSystem, ValidationResult
)


@pytest.fixture
def sample_complexity():
    """Create a sample complexity score."""
    return ComplexityScore(
        cognitive_complexity=5.0,
        computational_complexity=5.0,
        domain_complexity=5.0,
        integration_complexity=5.0,
        overall_complexity=5.0,
        explanation="Medium complexity"
    )


@pytest.fixture
def sample_sub_problem(sample_complexity):
    """Create a sample sub-problem."""
    return SubProblem(
        id=generate_id("subproblem"),
        parent_id="problem1",
        title="Test Sub-Problem",
        description="A test sub-problem for validation",
        type=SubProblemType.ANALYSIS,
        complexity_score=sample_complexity,
        dependencies=[],
        success_criteria=[
            SuccessCriterion(
                id=generate_id("criterion"),
                description="Test criterion",
                metric="completion",
                threshold=1.0,
                validation_method="review"
            )
        ],
        validation_gauntlet="coherence",
        priority=5,
        estimated_effort=8
    )


@pytest.fixture
def good_plan(sample_sub_problem, sample_complexity):
    """Create a good decomposition plan that should pass gauntlets."""
    sub_problems = []
    
    # Create diverse sub-problems
    for i in range(4):
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title=f"Sub-Problem {i+1}",
            description=f"Description for sub-problem {i+1} with sufficient detail",
            type=[SubProblemType.RESEARCH, SubProblemType.ANALYSIS, 
                  SubProblemType.IMPLEMENTATION, SubProblemType.VALIDATION][i],
            complexity_score=sample_complexity,
            dependencies=[sub_problems[i-1].id] if i > 0 else [],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description=f"Criterion for sub-problem {i+1}",
                    metric="completion",
                    threshold=0.9,
                    validation_method="review"
                )
            ],
            validation_gauntlet="coherence",
            priority=8 - i,
            estimated_effort=10
        )
        sub_problems.append(sp)
    
    return DecompositionPlan(
        id=generate_id("plan"),
        problem_id="problem1",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=sub_problems,
        dependency_graph=DependencyGraph(
            nodes={sp.id: sp for sp in sub_problems},
            edges={sp.id: sp.dependencies for sp in sub_problems},
            execution_order=[sp.id for sp in sub_problems]
        ),
        confidence_level=0.85
    )


@pytest.fixture
def poor_plan():
    """Create a poor decomposition plan that should fail gauntlets."""
    complexity = ComplexityScore(
        cognitive_complexity=9.5,
        computational_complexity=9.5,
        domain_complexity=9.5,
        integration_complexity=9.5,
        overall_complexity=9.5,
        explanation="Very high complexity"
    )
    
    sp = SubProblem(
        id=generate_id("subproblem"),
        parent_id="problem1",
        title="Vague Problem",
        description="Short",  # Too short
        type=SubProblemType.ANALYSIS,
        complexity_score=complexity,  # Too complex
        dependencies=[],
        success_criteria=[],  # No criteria
        validation_gauntlet="",
        priority=0,
        estimated_effort=100  # Too much effort
    )
    
    return DecompositionPlan(
        id=generate_id("plan"),
        problem_id="problem1",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=[sp],
        confidence_level=0.5
    )


class TestCoherenceGauntlet:
    """Test CoherenceGauntlet."""
    
    def test_good_plan_passes(self, good_plan):
        gauntlet = CoherenceGauntlet()
        # Mock the LLM check to return a passing result
        gauntlet._check_coherence_with_llm = Mock(return_value={
            'score': 0.85,
            'feedback': 'Good coherence',
            'improvements': []
        })
        result = gauntlet.run(good_plan)
        
        assert result.passed is True
        assert result.score >= 0.7
        assert result.validator == "coherence_gauntlet"
    
    def test_empty_plan_fails(self):
        gauntlet = CoherenceGauntlet()
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=[],
            confidence_level=0.0
        )
        
        result = gauntlet.run(plan)
        assert result.passed is False
        assert result.score == 0.0
    
    def test_poor_plan_fails(self, poor_plan):
        gauntlet = CoherenceGauntlet()
        # Mock the LLM check to return a low score
        gauntlet._check_coherence_with_llm = Mock(return_value={
            'score': 0.5,
            'feedback': 'Poor coherence',
            'improvements': ['Add more detail']
        })
        result = gauntlet.run(poor_plan)
        
        assert result.score < 0.9  # Should have some issues


class TestCompletenessGauntlet:
    """Test CompletenessGauntlet."""
    
    def test_good_plan_passes(self, good_plan):
        gauntlet = CompletenessGauntlet()
        # Mock the LLM check
        gauntlet._check_completeness_with_llm = Mock(return_value={
            'score': 0.85,
            'feedback': 'Good completeness',
            'improvements': []
        })
        result = gauntlet.run(good_plan)
        
        assert result.passed is True
        assert result.score >= 0.75
        assert result.validator == "completeness_gauntlet"
    
    def test_single_subproblem_gets_penalty(self, sample_sub_problem):
        gauntlet = CompletenessGauntlet()
        # Mock the LLM check
        gauntlet._check_completeness_with_llm = Mock(return_value={
            'score': 0.7,
            'feedback': 'Single subproblem',
            'improvements': ['Consider more sub-problems']
        })
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=[sample_sub_problem],
            confidence_level=0.7
        )
        
        result = gauntlet.run(plan)
        assert result.score < 1.0  # Should be penalized
    
    def test_diverse_types_score_higher(self, good_plan):
        gauntlet = CompletenessGauntlet()
        # Mock the LLM check
        gauntlet._check_completeness_with_llm = Mock(return_value={
            'score': 0.9,
            'feedback': 'Good diversity',
            'improvements': []
        })
        result = gauntlet.run(good_plan)
        
        # Good plan has 4 different types
        assert result.score >= 0.8


class TestFeasibilityGauntlet:
    """Test FeasibilityGauntlet."""
    
    def test_good_plan_passes(self, good_plan):
        gauntlet = FeasibilityGauntlet()
        # Mock the LLM check
        gauntlet._check_feasibility_with_llm = Mock(return_value={
            'score': 0.85,
            'feedback': 'Feasible plan',
            'improvements': []
        })
        result = gauntlet.run(good_plan)
        
        assert result.passed is True
        assert result.score >= 0.7
        assert result.validator == "feasibility_gauntlet"
    
    def test_high_complexity_fails(self, poor_plan):
        gauntlet = FeasibilityGauntlet()
        # Mock the LLM check
        gauntlet._check_feasibility_with_llm = Mock(return_value={
            'score': 0.4,
            'feedback': 'Too complex',
            'improvements': ['Reduce complexity']
        })
        result = gauntlet.run(poor_plan)
        
        # Poor plan has complexity 9.5 (above max of 8.0)
        assert result.score < 0.9
    
    def test_excessive_effort_penalized(self, sample_complexity):
        gauntlet = FeasibilityGauntlet()
        # Mock the LLM check
        gauntlet._check_feasibility_with_llm = Mock(return_value={
            'score': 0.6,
            'feedback': 'Excessive effort',
            'improvements': ['Break down further']
        })
        
        # Create plan with excessive effort
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title="Huge Task",
            description="A task requiring too much effort",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=sample_complexity,
            dependencies=[],
            success_criteria=[],
            estimated_effort=100  # Way too much
        )
        
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=[sp],
            confidence_level=0.5
        )
        
        result = gauntlet.run(plan)
        assert result.score < 1.0


class TestDependencyGauntlet:
    """Test DependencyGauntlet."""
    
    def test_good_plan_passes(self, good_plan):
        gauntlet = DependencyGauntlet()
        # Mock the LLM check
        gauntlet._check_dependency_with_llm = Mock(return_value={
            'score': 0.9,
            'feedback': 'Good dependencies',
            'improvements': []
        })
        result = gauntlet.run(good_plan)
        
        assert result.passed is True
        assert result.score >= 0.8
        assert result.validator == "dependency_gauntlet"
    
    def test_circular_dependency_fails(self, sample_complexity):
        gauntlet = DependencyGauntlet()
        # Mock the LLM check to detect cycles
        gauntlet._check_dependency_with_llm = Mock(return_value={
            'score': 0.3,
            'feedback': 'Circular dependency detected',
            'improvements': ['Remove circular dependency between sp1 and sp2']
        })
        
        # Create circular dependency: A -> B -> A
        sp1 = SubProblem(
            id="sp1",
            parent_id="problem1",
            title="Sub-Problem 1",
            description="First sub-problem",
            type=SubProblemType.ANALYSIS,
            complexity_score=sample_complexity,
            dependencies=["sp2"],  # Depends on sp2
            success_criteria=[]
        )
        
        sp2 = SubProblem(
            id="sp2",
            parent_id="problem1",
            title="Sub-Problem 2",
            description="Second sub-problem",
            type=SubProblemType.ANALYSIS,
            complexity_score=sample_complexity,
            dependencies=["sp1"],  # Depends on sp1 - circular!
            success_criteria=[]
        )
        
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.DEPENDENCY,
            sub_problems=[sp1, sp2],
            confidence_level=0.5
        )
        
        result = gauntlet.run(plan)
        assert result.passed is False
        assert result.score < 0.8
    
    def test_invalid_dependency_reference(self, sample_complexity):
        gauntlet = DependencyGauntlet()
        # Mock the LLM check
        gauntlet._check_dependency_with_llm = Mock(return_value={
            'score': 0.7,
            'feedback': 'Invalid dependency reference',
            'improvements': ['Fix dependency reference']
        })
        
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title="Sub-Problem",
            description="A sub-problem with invalid dependency",
            type=SubProblemType.ANALYSIS,
            complexity_score=sample_complexity,
            dependencies=["nonexistent_id"],  # Invalid reference
            success_criteria=[]
        )
        
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.DEPENDENCY,
            sub_problems=[sp],
            confidence_level=0.5
        )
        
        result = gauntlet.run(plan)
        assert result.score < 1.0


class TestGauntletSystem:
    """Test GauntletSystem orchestrator."""
    
    def test_run_all_gauntlets(self, good_plan):
        system = GauntletSystem()
        # Mock all gauntlet runs
        for name, gauntlet in system.gauntlets.items():
            gauntlet.run = Mock(return_value=ValidationResult(
                validator=name,
                passed=True,
                score=0.85,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ))
        
        results = system.run_decomposition_gauntlets(good_plan)
        
        # Check that we have results for core gauntlets
        assert 'coherence' in results
        assert 'completeness' in results
        assert 'feasibility' in results
        assert 'dependency' in results
    
    def test_run_specific_gauntlets(self, good_plan):
        system = GauntletSystem()
        results = system.run_decomposition_gauntlets(
            good_plan, 
            gauntlets=['coherence', 'completeness']
        )
        
        assert len(results) == 2
        assert 'coherence' in results
        assert 'completeness' in results
        assert 'feasibility' not in results
    
    def test_process_feedback(self, good_plan):
        system = GauntletSystem()
        # Mock results
        mock_results = {
            'coherence': ValidationResult(
                validator='coherence',
                passed=True,
                score=0.85,
                feedback="Good coherence",
                improvements=[],
                timestamp=datetime.now()
            ),
            'completeness': ValidationResult(
                validator='completeness',
                passed=True,
                score=0.9,
                feedback="Good completeness",
                improvements=[],
                timestamp=datetime.now()
            ),
            'feasibility': ValidationResult(
                validator='feasibility',
                passed=True,
                score=0.8,
                feedback="Good feasibility",
                improvements=[],
                timestamp=datetime.now()
            ),
            'dependency': ValidationResult(
                validator='dependency',
                passed=True,
                score=0.85,
                feedback="Good dependencies",
                improvements=[],
                timestamp=datetime.now()
            )
        }
        feedback = system.process_gauntlet_feedback(mock_results)
        
        assert len(feedback) == 4
        # Check that feedback sources match gauntlet names
        sources = {f.source for f in feedback}
        assert sources == {'coherence', 'completeness', 'feasibility', 'dependency'}
    
    def test_overall_quality(self, good_plan):
        system = GauntletSystem()
        # Mock results
        mock_results = {
            'coherence': ValidationResult(
                validator='coherence',
                passed=True,
                score=0.85,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'completeness': ValidationResult(
                validator='completeness',
                passed=True,
                score=0.9,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'feasibility': ValidationResult(
                validator='feasibility',
                passed=True,
                score=0.8,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'dependency': ValidationResult(
                validator='dependency',
                passed=True,
                score=0.85,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            )
        }
        quality = system.get_overall_quality(mock_results)
        
        assert 0.0 <= quality <= 1.0
        assert quality >= 0.7  # Good plan should have good quality
    
    def test_all_passed(self, good_plan, poor_plan):
        system = GauntletSystem()
        
        # Mock good results - all passed
        good_results = {
            'coherence': ValidationResult(
                validator='coherence',
                passed=True,
                score=0.85,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'completeness': ValidationResult(
                validator='completeness',
                passed=True,
                score=0.9,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'feasibility': ValidationResult(
                validator='feasibility',
                passed=True,
                score=0.8,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'dependency': ValidationResult(
                validator='dependency',
                passed=True,
                score=0.85,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            )
        }
        assert system.all_passed(good_results) is True
        
        # Mock poor results - some failed
        poor_results = {
            'coherence': ValidationResult(
                validator='coherence',
                passed=False,
                score=0.5,
                feedback="Poor",
                improvements=["Fix this"],
                timestamp=datetime.now()
            ),
            'completeness': ValidationResult(
                validator='completeness',
                passed=False,
                score=0.4,
                feedback="Poor",
                improvements=["Fix this"],
                timestamp=datetime.now()
            ),
            'feasibility': ValidationResult(
                validator='feasibility',
                passed=True,
                score=0.8,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            ),
            'dependency': ValidationResult(
                validator='dependency',
                passed=True,
                score=0.85,
                feedback="Good",
                improvements=[],
                timestamp=datetime.now()
            )
        }
        assert system.all_passed(poor_results) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
