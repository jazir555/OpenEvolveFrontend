"""
Unit tests for TaskComplexityClassifier.
"""

import pytest
import math
from adaptive_mdap.core.types import SubProblem
from adaptive_mdap.classifiers.task_complexity_classifier import (
    TaskComplexityClassifier,
    ClassifierConfig,
)
from adaptive_mdap.core.errors import ClassificationError


class TestTextLengthFeature:
    """Tests for text length feature computation."""
    
    def test_empty_description(self, classifier):
        """Test empty description returns 0."""
        subproblem = SubProblem(
            id="test",
            description="",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_text_length_feature(subproblem)
        assert score == 0.0
    
    def test_short_description(self, classifier):
        """Test short description returns low score."""
        subproblem = SubProblem(
            id="test",
            description="Short task",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_text_length_feature(subproblem)
        assert 0.0 <= score < 0.3
    
    def test_medium_description(self, classifier):
        """Test medium description returns medium score."""
        description = "This is a " + "medium length " * 50 + "description."
        subproblem = SubProblem(
            id="test",
            description=description,
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_text_length_feature(subproblem)
        assert 0.3 <= score <= 0.7
    
    def test_long_description(self, classifier):
        """Test long description returns high score."""
        description = "This is a " + "very long " * 500 + "description."
        subproblem = SubProblem(
            id="test",
            description=description,
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_text_length_feature(subproblem)
        assert score > 0.7
    
    def test_very_long_description_capped(self, classifier):
        """Test very long description is capped at 1.0."""
        description = "x" * 10000
        subproblem = SubProblem(
            id="test",
            description=description,
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_text_length_feature(subproblem)
        assert score <= 1.0


class TestDepthFeature:
    """Tests for depth feature computation."""
    
    def test_depth_zero(self, classifier):
        """Test depth 0 returns 0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_depth_feature(subproblem)
        assert score == 0.0
    
    def test_depth_five(self, classifier):
        """Test depth 5 returns 0.5."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=5,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_depth_feature(subproblem)
        assert abs(score - 0.5) < 0.01
    
    def test_depth_ten(self, classifier):
        """Test depth 10 returns 1.0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=10,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_depth_feature(subproblem)
        assert score == 1.0
    
    def test_depth_twenty_capped(self, classifier):
        """Test depth 20 is capped at 1.0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=20,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_depth_feature(subproblem)
        assert score == 1.0
    
    def test_negative_depth(self, classifier):
        """Test negative depth returns 0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=-5,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_depth_feature(subproblem)
        assert score == 0.0


class TestDependencyFeature:
    """Tests for dependency feature computation."""
    
    def test_zero_dependencies(self, classifier):
        """Test 0 dependencies returns 0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_dependency_feature(subproblem)
        assert score == 0.0
    
    def test_five_dependencies(self, classifier):
        """Test 5 dependencies returns 0.5."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=0,
            dependencies=["d1", "d2", "d3", "d4", "d5"],
            metadata={},
        )
        score = classifier.compute_dependency_feature(subproblem)
        assert abs(score - 0.5) < 0.01
    
    def test_ten_dependencies(self, classifier):
        """Test 10 dependencies returns 1.0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=0,
            dependencies=[f"d{i}" for i in range(10)],
            metadata={},
        )
        score = classifier.compute_dependency_feature(subproblem)
        assert score == 1.0
    
    def test_fifteen_dependencies_capped(self, classifier):
        """Test 15 dependencies is capped at 1.0."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="test",
            depth=0,
            dependencies=[f"d{i}" for i in range(15)],
            metadata={},
        )
        score = classifier.compute_dependency_feature(subproblem)
        assert score == 1.0


class TestHistoricalErrorFeature:
    """Tests for historical error feature computation."""
    
    def test_cold_start(self, classifier):
        """Test cold start returns default 0.4."""
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="unknown_domain",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_historical_error_feature(subproblem)
        assert abs(score - 0.4) < 0.01
    
    def test_perfect_domain(self, classifier):
        """Test domain with 0% errors returns low score (smoothed)."""
        # Update stats for a domain with 10 samples
        for _ in range(10):
            classifier.update_historical_stats("perfect_domain", success=True, complexity=0.5)
        
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="perfect_domain",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_historical_error_feature(subproblem)
        # Smoothed from 0.0 with prior 0.4
        assert 0.0 < score < 0.2
    
    def test_terrible_domain(self, classifier):
        """Test domain with 100% errors returns high score (smoothed)."""
        # Update stats for a domain with 10 samples
        for _ in range(10):
            classifier.update_historical_stats("terrible_domain", success=False, complexity=0.5)
        
        subproblem = SubProblem(
            id="test",
            description="Test",
            domain="terrible_domain",
            depth=0,
            dependencies=[],
            metadata={},
        )
        score = classifier.compute_historical_error_feature(subproblem)
        # Smoothed from 1.0 with prior 0.4
        assert 0.6 < score < 1.0


class TestComplexityCombination:
    """Tests for complexity score combination."""
    
    def test_equal_weights(self, classifier, sample_subproblem):
        """Test granular weights produce average of features."""
        complexity = classifier.compute_complexity(sample_subproblem)
        
        # Check all components are in valid range
        assert 0.0 <= complexity.overall_score <= 1.0
        assert 0.0 <= complexity.text_length_score <= 1.0
        assert 0.0 <= complexity.domain_rarity_score <= 1.0
        assert 0.0 <= complexity.depth_score <= 1.0
        assert 0.0 <= complexity.historical_error_score <= 1.0
        assert 0.0 <= complexity.dependency_score <= 1.0
        assert complexity.keyword_score >= 0.0
        assert complexity.constraint_score >= 0.0
        
        # Check weights sum to 1
        total_weight = sum(complexity.feature_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_simple_problem_low_complexity(self, classifier, simple_subproblem):
        """Test simple problem has low complexity."""
        complexity = classifier.compute_complexity(simple_subproblem)
        assert complexity.overall_score < 0.5
    
    def test_complex_problem_high_complexity(self, classifier, complex_subproblem):
        """Test complex problem has high complexity."""
        complexity = classifier.compute_complexity(complex_subproblem)
        assert complexity.overall_score > 0.5


class TestClassifierConfig:
    """Tests for classifier configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ClassifierConfig()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert len(config.feature_weights) == 7
        assert abs(sum(config.feature_weights.values()) - 1.0) < 0.01
    
    def test_invalid_weights(self):
        """Test invalid weights raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ClassifierConfig(
                feature_weights={
                    "text_length": 0.5,
                    "domain_rarity": 0.5,
                    "depth": 0.5,
                    "historical_error": 0.5,
                    "dependency": 0.5,
                }
            )
