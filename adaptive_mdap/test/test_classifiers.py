"""Tests for TaskComplexityClassifier."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from adaptive_mdap.classifiers.task_complexity_classifier import (
    TaskComplexityClassifier,
    ClassifierConfig,
)
from adaptive_mdap.core.types import SubProblem, ComplexityScore


class TestClassifierConfig:
    """Tests for ClassifierConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClassifierConfig()
        
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.max_text_length == 5000
        assert config.max_depth == 10
        assert config.max_dependencies == 10
        assert sum(config.feature_weights.values()) == 1.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ClassifierConfig(
            embedding_model="custom-model",
            max_text_length=1000,
            feature_weights={
                "text_length": 0.3,
                "domain_rarity": 0.2,
                "depth": 0.2,
                "historical_error": 0.1,
                "dependency": 0.1,
                "keyword_complexity": 0.05,
                "constraint_density": 0.05,
            }
        )
        
        assert config.embedding_model == "custom-model"
        assert config.max_text_length == 1000
    
    def test_invalid_feature_weights(self):
        """Test that invalid feature weights raise an error."""
        with pytest.raises(ValueError, match="Feature weights must sum to 1.0"):
            ClassifierConfig(
                feature_weights={
                    "text_length": 0.5,
                    "domain_rarity": 0.5,
                    "depth": 0.5,  # Sum > 1.0
                    "historical_error": 0.0,
                    "dependency": 0.0,
                    "keyword_complexity": 0.0,
                    "constraint_density": 0.0,
                }
            )


class TestTaskComplexityClassifier:
    """Tests for TaskComplexityClassifier."""
    
    def test_classifier_initialization(self):
        """Test classifier can be initialized."""
        classifier = TaskComplexityClassifier()
        
        assert classifier.config is not None
        assert classifier._embedding_model is None
    
    def test_compute_text_length_score(self):
        """Test text length score computation."""
        classifier = TaskComplexityClassifier()
        
        # Short text should have low score
        short_text = "Solve x + 2 = 4"
        score = classifier._compute_text_length_score(short_text)
        assert 0.0 <= score <= 1.0
        
        # Long text should have higher score
        long_text = "a" * 1000
        long_score = classifier._compute_text_length_score(long_text)
        assert long_score >= score
    
    def test_compute_depth_score(self):
        """Test depth score computation."""
        classifier = TaskComplexityClassifier()
        
        # Zero depth
        score0 = classifier._compute_depth_score(0)
        assert score0 == 0.0
        
        # Max depth
        score_max = classifier._compute_depth_score(10)
        assert score_max == 1.0
    
    def test_compute_dependency_score(self):
        """Test dependency score computation."""
        classifier = TaskComplexityClassifier()
        
        # No dependencies
        score0 = classifier._compute_dependency_score([])
        assert score0 == 0.0
        
        # Many dependencies
        deps = ["dep1", "dep2", "dep3", "dep4", "dep5"]
        score = classifier._compute_dependency_score(deps)
        assert 0.0 <= score <= 1.0
    
    def test_compute_keyword_score(self):
        """Test keyword complexity score."""
        classifier = TaskComplexityClassifier()
        
        # Simple text - low score
        simple = "Write a function"
        simple_score = classifier._compute_keyword_score(simple)
        
        # Complex text - high score
        complex_text = "Optimize the distributed cryptographic algorithm"
        complex_score = classifier._compute_keyword_score(complex_text)
        
        assert complex_score >= simple_score
    
    def test_classify_subproblem(self):
        """Test subproblem classification."""
        classifier = TaskComplexityClassifier()
        
        subproblem = SubProblem(
            id="test-1",
            description="Implement a simple addition function",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        # Should not raise an error
        score = classifier.classify(subproblem)
        
        assert isinstance(score, ComplexityScore)
        assert 0.0 <= score.overall_score <= 1.0
    
    def test_classify_with_cache(self):
        """Test that classification uses cache."""
        classifier = TaskComplexityClassifier()
        
        subproblem = SubProblem(
            id="test-2",
            description="Solve a mathematical problem",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        # First call
        score1 = classifier.classify(subproblem)
        
        # Second call should hit cache
        score2 = classifier.classify(subproblem)
        
        # Scores should be the same
        assert score1.overall_score == score2.overall_score
    
    def test_subproblem_too_long(self):
        """Test handling of very long subproblems."""
        classifier = TaskComplexityClassifier()
        
        long_desc = "a" * 10000  # Exceeds max_text_length
        
        subproblem = SubProblem(
            id="test-long",
            description=long_desc,
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        # Should handle gracefully
        score = classifier.classify(subproblem)
        assert 0.0 <= score.overall_score <= 1.0
    
    def test_empty_description(self):
        """Test handling of empty description."""
        classifier = TaskComplexityClassifier()
        
        subproblem = SubProblem(
            id="test-empty",
            description="",
            domain="mathematics",
            depth=1,
            dependencies=[],
            metadata={},
        )
        
        score = classifier.classify(subproblem)
        assert 0.0 <= score.overall_score <= 1.0
    
    def test_high_depth_subproblem(self):
        """Test classification of deeply nested subproblem."""
        classifier = TaskComplexityClassifier()
        
        subproblem = SubProblem(
            id="test-deep",
            description="Implement a complex recursive algorithm",
            domain="computer_science",
            depth=10,  # Max depth
            dependencies=["dep1", "dep2", "dep3"],
            metadata={},
        )
        
        score = classifier.classify(subproblem)
        assert score.depth_score == 1.0  # Max depth should give max score


class TestComplexityScore:
    """Tests for ComplexityScore validation."""
    
    def test_valid_score(self):
        """Test creating a valid complexity score."""
        score = ComplexityScore(
            overall_score=0.5,
            text_length_score=0.3,
            domain_rarity_score=0.4,
            depth_score=0.5,
            historical_error_score=0.2,
            dependency_score=0.3,
            feature_weights={"text_length": 0.2, "domain_rarity": 0.2},
        )
        
        assert score.overall_score == 0.5
    
    def test_score_boundary_values(self):
        """Test boundary values for scores."""
        # Min boundary
        score_min = ComplexityScore(
            overall_score=0.0,
            text_length_score=0.0,
            domain_rarity_score=0.0,
            depth_score=0.0,
            historical_error_score=0.0,
            dependency_score=0.0,
            feature_weights={},
        )
        assert score_min.overall_score == 0.0
        
        # Max boundary
        score_max = ComplexityScore(
            overall_score=1.0,
            text_length_score=1.0,
            domain_rarity_score=1.0,
            depth_score=1.0,
            historical_error_score=1.0,
            dependency_score=1.0,
            feature_weights={},
        )
        assert score_max.overall_score == 1.0
    
    def test_invalid_score_raises_error(self):
        """Test that invalid scores raise ValueError."""
        with pytest.raises(ValueError, match="overall_score must be in"):
            ComplexityScore(
                overall_score=1.5,  # Invalid
                text_length_score=0.5,
                domain_rarity_score=0.5,
                depth_score=0.5,
                historical_error_score=0.5,
                dependency_score=0.5,
                feature_weights={},
            )
    
    def test_small_floating_point_error_clamped(self):
        """Test that small floating point errors are clamped."""
        # Slightly negative value
        score = ComplexityScore(
            overall_score=-0.0001,
            text_length_score=0.5,
            domain_rarity_score=0.5,
            depth_score=0.5,
            historical_error_score=0.5,
            dependency_score=0.5,
            feature_weights={},
        )
        assert score.overall_score == 0.0
        
        # Slightly > 1.0 value
        score = ComplexityScore(
            overall_score=1.0001,
            text_length_score=0.5,
            domain_rarity_score=0.5,
            depth_score=0.5,
            historical_error_score=0.5,
            dependency_score=0.5,
            feature_weights={},
        )
        assert score.overall_score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
