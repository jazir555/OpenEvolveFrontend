"""
Tests for Sovereign Knowledge Management System
Task 9.6: Unit tests for knowledge management
"""

import pytest
from datetime import datetime

from sovereign_data_models import (
    DecompositionPlan, SubProblem, Pattern, ProblemType, DecompositionStrategy,
    SubProblemType, ComplexityScore, generate_id
)
from sovereign_knowledge_manager import KnowledgeManager


@pytest.fixture
def sample_plan():
    """Create a sample decomposition plan."""
    complexity = ComplexityScore(
        cognitive_complexity=5.0,
        computational_complexity=5.0,
        domain_complexity=5.0,
        integration_complexity=5.0,
        overall_complexity=5.0,
        explanation="Medium"
    )
    
    sub_problems = []
    for i in range(4):
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title=f"Sub-Problem {i+1}",
            description=f"Description {i+1}",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=complexity,
            dependencies=[],
            success_criteria=[],
            estimated_effort=10
        )
        sub_problems.append(sp)
    
    return DecompositionPlan(
        id=generate_id("plan"),
        problem_id="problem1",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=sub_problems,
        confidence_level=0.85
    )


@pytest.fixture
def sample_pattern():
    """Create a sample pattern."""
    return Pattern(
        id=generate_id("pattern"),
        problem_type=ProblemType.IMPLEMENTATION,
        strategy=DecompositionStrategy.SEMANTIC,
        pattern_description="Use semantic decomposition for implementation problems",
        success_rate=0.85,
        usage_count=5,
        avg_quality_score=0.82,
        applicable_domains=["software_engineering"],
        created_at=datetime.now(),
        last_used=datetime.now()
    )


class TestKnowledgeManager:
    """Test KnowledgeManager class."""
    
    def test_initialization(self):
        manager = KnowledgeManager()
        assert manager.database is not None
        assert manager.strategy_performance is not None
    
    def test_extract_patterns_success(self, sample_plan):
        manager = KnowledgeManager()
        
        patterns = manager.extract_patterns(
            plan=sample_plan,
            success=True,
            quality_score=0.85
        )
        
        assert isinstance(patterns, list)
        assert len(patterns) >= 1  # At least strategy pattern
    
    def test_extract_patterns_failure(self, sample_plan):
        manager = KnowledgeManager()
        
        # Low quality should not extract patterns
        patterns = manager.extract_patterns(
            plan=sample_plan,
            success=False,
            quality_score=0.4
        )
        
        assert len(patterns) == 0
    
    def test_store_pattern(self, sample_pattern):
        manager = KnowledgeManager()
        
        result = manager.store_pattern(sample_pattern)
        
        assert result is True
    
    def test_retrieve_patterns(self, sample_pattern):
        manager = KnowledgeManager()
        
        # Store a pattern first
        manager.store_pattern(sample_pattern)
        
        # Retrieve patterns
        patterns = manager.retrieve_patterns(
            problem_type=ProblemType.IMPLEMENTATION,
            min_success_rate=0.7
        )
        
        assert isinstance(patterns, list)
        # Should find at least the one we stored
        assert len(patterns) >= 1
    
    def test_retrieve_patterns_with_domain_filter(self, sample_pattern):
        manager = KnowledgeManager()
        
        manager.store_pattern(sample_pattern)
        
        # Retrieve with matching domain
        patterns = manager.retrieve_patterns(
            problem_type=ProblemType.IMPLEMENTATION,
            domain="software_engineering"
        )
        
        assert len(patterns) >= 1
    
    def test_apply_pattern(self, sample_pattern):
        manager = KnowledgeManager()
        
        guidance = manager.apply_pattern(
            pattern=sample_pattern,
            problem_description="Build a new system"
        )
        
        assert 'pattern_id' in guidance
        assert 'strategy' in guidance
        assert 'success_rate' in guidance
        assert 'recommendations' in guidance
        assert guidance['applicable'] is True
    
    def test_track_strategy_performance(self):
        manager = KnowledgeManager()
        
        # Track some performance data
        manager.track_strategy_performance(DecompositionStrategy.SEMANTIC, 0.85)
        manager.track_strategy_performance(DecompositionStrategy.SEMANTIC, 0.90)
        manager.track_strategy_performance(DecompositionStrategy.SEMANTIC, 0.88)
        
        # Check it was tracked
        assert len(manager.strategy_performance[DecompositionStrategy.SEMANTIC.value]) == 3
    
    def test_get_strategy_performance(self):
        manager = KnowledgeManager()
        
        # Track performance
        manager.track_strategy_performance(DecompositionStrategy.DEPENDENCY, 0.80)
        manager.track_strategy_performance(DecompositionStrategy.DEPENDENCY, 0.85)
        manager.track_strategy_performance(DecompositionStrategy.DEPENDENCY, 0.90)
        
        # Get performance metrics
        metrics = manager.get_strategy_performance(DecompositionStrategy.DEPENDENCY)
        
        assert 'avg_score' in metrics
        assert 'min_score' in metrics
        assert 'max_score' in metrics
        assert 'usage_count' in metrics
        assert metrics['usage_count'] == 3
        assert metrics['avg_score'] == pytest.approx(0.85, rel=0.01)
    
    def test_get_strategy_performance_no_data(self):
        manager = KnowledgeManager()
        
        # Get performance for unused strategy
        metrics = manager.get_strategy_performance(DecompositionStrategy.COMPLEXITY)
        
        assert metrics['avg_score'] == 0.0
        assert metrics['usage_count'] == 0
    
    def test_adapt_strategies(self):
        manager = KnowledgeManager()
        
        # Track improving strategy
        for score in [0.7, 0.75, 0.8, 0.85, 0.9]:
            manager.track_strategy_performance(DecompositionStrategy.SEMANTIC, score)
        
        # Get adaptation recommendations
        recommendations = manager.adapt_strategies()
        
        assert isinstance(recommendations, dict)
        if DecompositionStrategy.SEMANTIC.value in recommendations:
            rec = recommendations[DecompositionStrategy.SEMANTIC.value]
            assert 'trend' in rec
            assert 'action' in rec
    
    def test_get_best_strategy(self, sample_pattern):
        manager = KnowledgeManager()
        
        # Store some patterns
        manager.store_pattern(sample_pattern)
        
        # Get best strategy
        best = manager.get_best_strategy(
            problem_type=ProblemType.IMPLEMENTATION
        )
        
        # Should return a strategy (or None if no patterns)
        assert best is None or isinstance(best, DecompositionStrategy)


class TestPatternExtraction:
    """Test pattern extraction logic."""
    
    def test_extract_strategy_pattern(self, sample_plan):
        manager = KnowledgeManager()
        
        pattern = manager._extract_strategy_pattern(sample_plan, 0.85)
        
        assert pattern is not None
        assert pattern.strategy == sample_plan.strategy
        assert pattern.avg_quality_score == 0.85
    
    def test_extract_structural_patterns(self, sample_plan):
        manager = KnowledgeManager()
        
        patterns = manager._extract_structural_patterns(sample_plan, 0.85)
        
        assert isinstance(patterns, list)
        # Should extract structural pattern for good decomposition
        assert len(patterns) >= 1


class TestPatternSimilarity:
    """Test pattern similarity detection."""
    
    def test_patterns_similar(self):
        manager = KnowledgeManager()
        
        p1 = Pattern(
            id="p1",
            problem_type=ProblemType.IMPLEMENTATION,
            strategy=DecompositionStrategy.SEMANTIC,
            pattern_description="Pattern 1",
            success_rate=0.85,
            usage_count=1,
            avg_quality_score=0.80,
            applicable_domains=[],
            created_at=datetime.now(),
            last_used=datetime.now()
        )
        
        p2 = Pattern(
            id="p2",
            problem_type=ProblemType.IMPLEMENTATION,
            strategy=DecompositionStrategy.SEMANTIC,
            pattern_description="Pattern 2",
            success_rate=0.90,
            usage_count=1,
            avg_quality_score=0.82,  # Similar quality
            applicable_domains=[],
            created_at=datetime.now(),
            last_used=datetime.now()
        )
        
        assert manager._patterns_similar(p1, p2) is True
    
    def test_patterns_not_similar(self):
        manager = KnowledgeManager()
        
        p1 = Pattern(
            id="p1",
            problem_type=ProblemType.IMPLEMENTATION,
            strategy=DecompositionStrategy.SEMANTIC,
            pattern_description="Pattern 1",
            success_rate=0.85,
            usage_count=1,
            avg_quality_score=0.80,
            applicable_domains=[],
            created_at=datetime.now(),
            last_used=datetime.now()
        )
        
        p2 = Pattern(
            id="p2",
            problem_type=ProblemType.RESEARCH,  # Different type
            strategy=DecompositionStrategy.SEMANTIC,
            pattern_description="Pattern 2",
            success_rate=0.90,
            usage_count=1,
            avg_quality_score=0.80,
            applicable_domains=[],
            created_at=datetime.now(),
            last_used=datetime.now()
        )
        
        assert manager._patterns_similar(p1, p2) is False


class TestRecommendations:
    """Test recommendation generation."""
    
    def test_generate_recommendations(self, sample_pattern):
        manager = KnowledgeManager()
        
        recommendations = manager._generate_recommendations(sample_pattern)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should mention the strategy
        assert any('semantic' in r.lower() for r in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
