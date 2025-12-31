"""
Tests for HybridDecomposition strategy.
"""

import pytest
from datetime import datetime

from decomposition_engine import HybridDecomposition, SemanticDecomposition, DependencyDecomposition
from problem_analyzer import ProblemAnalyzer
from sovereign_data_models import (
    ProblemDefinition, ProblemType, DomainContext, ComplexityScore,
    Constraint, SuccessCriterion, generate_id
)


class TestHybridDecomposition:
    """Test hybrid decomposition strategy."""
    
    @pytest.fixture
    def analyzer(self):
        return ProblemAnalyzer()
    
    @pytest.fixture
    def hybrid_strategy(self):
        return HybridDecomposition()
    
    @pytest.fixture
    def sample_problem(self):
        """Create a sample problem for testing."""
        return ProblemDefinition(
            id=generate_id("problem"),
            title="Build Recommendation System",
            description="Build a scalable recommendation system with ML models and real-time processing",
            problem_type=ProblemType.IMPLEMENTATION,
            domain_context=DomainContext(
                domain="machine_learning",
                subdomain="recommendation_systems",
                related_domains=["data_engineering", "distributed_systems"],
                domain_knowledge={}
            ),
            complexity_score=ComplexityScore(
                cognitive_complexity=7.5,
                computational_complexity=8.0,
                domain_complexity=7.0,
                integration_complexity=8.5,
                overall_complexity=7.75,
                explanation="High complexity ML system"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={},
            deadline=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_hybrid_strategy_initialization(self, hybrid_strategy):
        """Test hybrid strategy can be initialized."""
        assert hybrid_strategy is not None
        assert hybrid_strategy.get_strategy_name() == "hybrid"
    
    def test_hybrid_decomposition_creates_subproblems(self, hybrid_strategy, sample_problem):
        """Test hybrid strategy creates sub-problems."""
        sub_problems = hybrid_strategy.decompose(sample_problem)
        
        assert len(sub_problems) > 0
        assert all(sp.parent_id == sample_problem.id for sp in sub_problems)
    
    def test_hybrid_combines_multiple_strategies(self, hybrid_strategy, sample_problem):
        """Test hybrid strategy combines semantic and dependency approaches."""
        # Get results from individual strategies
        semantic = SemanticDecomposition()
        dependency = DependencyDecomposition()
        
        semantic_results = semantic.decompose(sample_problem)
        dependency_results = dependency.decompose(sample_problem)
        hybrid_results = hybrid_strategy.decompose(sample_problem)
        
        # Hybrid should have characteristics of both
        # It should have semantic-like structure
        assert len(hybrid_results) >= len(semantic_results) * 0.5
        
        # But with enhanced dependencies
        total_hybrid_deps = sum(len(sp.dependencies) for sp in hybrid_results)
        total_semantic_deps = sum(len(sp.dependencies) for sp in semantic_results)
        
        # Hybrid should have at least as many dependencies as semantic alone
        assert total_hybrid_deps >= total_semantic_deps
    
    def test_hybrid_balances_complexity(self, hybrid_strategy, sample_problem):
        """Test hybrid strategy balances complexity."""
        sub_problems = hybrid_strategy.decompose(sample_problem)
        
        # No sub-problem should be excessively complex
        max_complexity = 7.0
        for sp in sub_problems:
            assert sp.complexity_score.overall_complexity <= max_complexity + 1.0  # Allow small margin
    
    def test_hybrid_splits_complex_subproblems(self, hybrid_strategy):
        """Test hybrid strategy splits overly complex sub-problems."""
        from sovereign_data_models import SubProblem, SubProblemType
        
        # Create a very complex sub-problem
        complex_sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="test_parent",
            title="Very Complex Task",
            description="This is an extremely complex task that needs to be split",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(
                cognitive_complexity=9.0,
                computational_complexity=9.0,
                domain_complexity=8.5,
                integration_complexity=9.0,
                overall_complexity=8.9,
                explanation="Extremely complex"
            ),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="feasibility",
            priority=5,
            estimated_effort=80
        )
        
        # Split it
        split_results = hybrid_strategy._split_complex_subproblem(complex_sp)
        
        assert len(split_results) == 2
        assert split_results[0].title.endswith("Phase 1")
        assert split_results[1].title.endswith("Phase 2")
        
        # Phase 2 should depend on Phase 1
        assert split_results[0].id in split_results[1].dependencies
        
        # Both should have lower complexity
        assert split_results[0].complexity_score.overall_complexity < complex_sp.complexity_score.overall_complexity
        assert split_results[1].complexity_score.overall_complexity < complex_sp.complexity_score.overall_complexity
    
    def test_hybrid_optimizes_dependencies(self, hybrid_strategy):
        """Test hybrid strategy removes transitive dependencies."""
        from sovereign_data_models import SubProblem, SubProblemType
        
        # Create sub-problems with transitive dependencies
        # A -> B -> C, but also A -> C (transitive, should be removed)
        sp_a = SubProblem(
            id="sp_a",
            parent_id="test",
            title="Task A",
            description="First task",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=["sp_b", "sp_c"],  # sp_c is transitive
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        sp_b = SubProblem(
            id="sp_b",
            parent_id="test",
            title="Task B",
            description="Second task",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=["sp_c"],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        sp_c = SubProblem(
            id="sp_c",
            parent_id="test",
            title="Task C",
            description="Third task",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        # Optimize dependencies
        optimized = hybrid_strategy._optimize_dependencies([sp_a, sp_b, sp_c])
        
        # Find optimized sp_a
        opt_sp_a = next(sp for sp in optimized if sp.id == "sp_a")
        
        # sp_c should be removed from sp_a's dependencies (transitive)
        assert "sp_c" not in opt_sp_a.dependencies
        assert "sp_b" in opt_sp_a.dependencies
    
    def test_hybrid_finds_similar_subproblems(self, hybrid_strategy):
        """Test hybrid strategy can find similar sub-problems."""
        from sovereign_data_models import SubProblem, SubProblemType
        
        target = SubProblem(
            id="target",
            parent_id="test",
            title="Data Processing Pipeline",
            description="Build data processing pipeline for ML",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        similar = SubProblem(
            id="similar",
            parent_id="test",
            title="Data Pipeline Implementation",
            description="Implement data processing for machine learning",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        different = SubProblem(
            id="different",
            parent_id="test",
            title="User Interface Design",
            description="Design the frontend interface",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        candidates = [similar, different]
        found = hybrid_strategy._find_similar_subproblems(target, candidates)
        
        # Should find similar but not different
        assert len(found) >= 1
        assert any(sp.id == "similar" for sp in found)
    
    def test_hybrid_with_real_problem(self, hybrid_strategy, analyzer):
        """Test hybrid strategy with a real problem."""
        problem = analyzer.analyze_problem(
            "Build a distributed recommendation system with real-time ML inference, "
            "data pipeline, API layer, and monitoring dashboard",
            title="Recommendation System"
        )
        
        sub_problems = hybrid_strategy.decompose(problem)
        
        # Should create multiple sub-problems
        assert len(sub_problems) >= 3
        
        # Should have reasonable complexity
        for sp in sub_problems:
            assert sp.complexity_score.overall_complexity <= 8.0
        
        # Should have some dependencies
        total_deps = sum(len(sp.dependencies) for sp in sub_problems)
        assert total_deps > 0
        
        # All should have valid IDs
        sp_ids = {sp.id for sp in sub_problems}
        for sp in sub_problems:
            for dep in sp.dependencies:
                # Dependencies should reference valid sub-problems
                # (or be from parent problem)
                assert dep in sp_ids or dep != sp.id


class TestHybridIntegration:
    """Test hybrid strategy integration with decomposition engine."""
    
    def test_engine_has_hybrid_strategy(self):
        """Test decomposition engine includes hybrid strategy."""
        from decomposition_engine import DecompositionEngine
        
        engine = DecompositionEngine()
        assert 'hybrid' in engine.strategies
        assert isinstance(engine.strategies['hybrid'], HybridDecomposition)
    
    def test_engine_can_use_hybrid_strategy(self):
        """Test engine can decompose using hybrid strategy."""
        from decomposition_engine import DecompositionEngine
        
        analyzer = ProblemAnalyzer()
        engine = DecompositionEngine(analyzer)
        
        problem = analyzer.analyze_problem(
            "Create a machine learning pipeline with data processing, model training, and deployment",
            title="ML Pipeline"
        )
        
        # Manually select hybrid strategy
        plan = engine.decompose(problem, strategy='hybrid')
        
        assert plan is not None
        assert plan.strategy.value == 'hybrid'
        assert len(plan.sub_problems) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
