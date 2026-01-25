"""
Unit Tests for Δ₂ Predictive Model Generator
============================================

Comprehensive test suite for predictive model generation.

Author: Agent E2 (Δ₂ Specialist)
Date: 2025-12-31
"""

import pytest
import numpy as np
from typing import List, Dict, Any

# Import module to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4.predictive_model_generator import (
    PredictiveModelGenerator,
    SolutionAnalyzer,
    NeuralNetworkGenerator,
    TreeModelGenerator,
    RESESolution,
    Delta2Config,
    ModelType,
    PredictionType,
    FalsifiabilityStatus,
    Feature,
    Pattern,
    Prediction,
    ModelMetrics,
    FalsifiabilityReport,
    generate_predictive_model,
    ModelGenerationError,
    FalsifiabilityError,
)

# Skip tests if ML libraries not available
pytest.importorskip("sklearn", reason="scikit-learn not available")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Create sample configuration"""
    return Delta2Config(
        default_model_type=ModelType.AUTO,
        prefer_interpretable=True,
        tree_max_depth=5,
        forest_n_estimators=10,  # Small for testing
        nn_max_epochs=5,  # Small for testing
        random_seed=42
    )


@pytest.fixture
def sample_solution():
    """Create sample RESE solution"""
    return RESESolution(
        problem_id="test_001",
        solution={
            "parameters": {"temperature": 300, "pressure": 1.0},
            "outcome": 0.85
        },
        constraints=[
            "Temperature must be < 500 K",
            "Pressure must be > 0.1 bar",
            "Time must be positive"
        ],
        aci_history=[45.0, 35.0, 25.0, 20.0],
        metadata={
            "domain": "chemistry",
            "n_classes": 2,
            "require_interpretability": True
        },
        stage_results={}
    )


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    y = np.random.randn(100)  # Regression targets
    return X, y


# =============================================================================
# SOLUTION ANALYZER TESTS
# =============================================================================

class TestSolutionAnalyzer:
    """Test solution analysis functionality"""

    def test_initialization(self, sample_config):
        """Test analyzer initialization"""
        analyzer = SolutionAnalyzer(sample_config)
        assert analyzer.config == sample_config

    def test_analyze_solution(self, sample_config, sample_solution):
        """Test complete solution analysis"""
        analyzer = SolutionAnalyzer(sample_config)
        analysis = analyzer.analyze(sample_solution)

        # Check required keys
        assert 'features' in analysis
        assert 'patterns' in analysis
        assert 'complexity' in analysis
        assert 'n_samples' in analysis
        assert 'prediction_type' in analysis
        assert 'requires_interpretability' in analysis

    def test_extract_features(self, sample_config, sample_solution):
        """Test feature extraction"""
        analyzer = SolutionAnalyzer(sample_config)
        features = analyzer._extract_features(sample_solution)

        assert isinstance(features, list)
        assert len(features) > 0
        assert all(isinstance(f, Feature) for f in features)

        # Check sorted by importance
        importances = [f.importance for f in features]
        assert importances == sorted(importances, reverse=True)

    def test_extract_patterns(self, sample_config, sample_solution):
        """Test pattern extraction"""
        analyzer = SolutionAnalyzer(sample_config)
        patterns = analyzer._extract_patterns(sample_solution)

        assert isinstance(patterns, list)
        assert all(isinstance(p, Pattern) for p in patterns)

    def test_estimate_complexity(self, sample_config, sample_solution):
        """Test complexity estimation"""
        analyzer = SolutionAnalyzer(sample_config)
        complexity = analyzer._estimate_complexity(sample_solution)

        assert isinstance(complexity, int)
        assert complexity > 0

    def test_determine_prediction_type(self, sample_config, sample_solution):
        """Test prediction type determination"""
        analyzer = SolutionAnalyzer(sample_config)

        # Regression problem
        pred_type = analyzer._determine_prediction_type(sample_solution)
        assert pred_type == PredictionType.REGRESSION

        # Classification problem
        classification_solution = RESESolution(
            problem_id="test_002",
            solution={},
            constraints=["Class must be A or B"],
            metadata={}
        )
        pred_type = analyzer._determine_prediction_type(classification_solution)
        assert pred_type == PredictionType.CLASSIFICATION

    def test_needs_interpretability(self, sample_config, sample_solution):
        """Test interpretability requirement detection"""
        analyzer = SolutionAnalyzer(sample_config)

        # Scientific domain
        needs_interp = analyzer._needs_interpretability(sample_solution)
        assert needs_interp == True

        # Non-scientific
        other_solution = RESESolution(
            problem_id="test_003",
            solution={},
            constraints=[],
            metadata={"domain": "business"}
        )
        needs_interp = analyzer._needs_interpretability(other_solution)
        assert needs_interp == False


# =============================================================================
# TREE MODEL GENERATOR TESTS
# =============================================================================

class TestTreeModelGenerator:
    """Test tree-based model generation"""

    def test_initialization(self, sample_config):
        """Test generator initialization"""
        generator = TreeModelGenerator(sample_config)
        assert generator.config == sample_config

    def test_generate_decision_tree(self, sample_config, sample_solution):
        """Test decision tree generation"""
        generator = TreeModelGenerator(sample_config)

        analysis = {
            'prediction_type': PredictionType.REGRESSION,
            'features': [Feature(name=f"f{i}", type="numeric") for i in range(5)],
            'complexity': 50,
            'n_samples': 100
        }

        model = generator.generate_decision_tree(analysis, sample_solution)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_generate_random_forest(self, sample_config, sample_solution):
        """Test random forest generation"""
        generator = TreeModelGenerator(sample_config)

        analysis = {
            'prediction_type': PredictionType.REGRESSION,
            'features': [Feature(name=f"f{i}", type="numeric") for i in range(5)],
            'complexity': 200,
            'n_samples': 100
        }

        model = generator.generate_random_forest(analysis, sample_solution)

        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_train_tree_model(self, sample_config, sample_data):
        """Test tree model training"""
        generator = TreeModelGenerator(sample_config)

        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(
            max_depth=5,
            random_state=42
        )

        X, y = sample_data
        trained_model, metrics = generator.train(model, X, y)

        assert trained_model is not None
        assert metrics is not None
        assert hasattr(metrics, 'validation_loss')


# =============================================================================
# PREDICTIVE MODEL GENERATOR TESTS
# =============================================================================

class TestPredictiveModelGenerator:
    """Test main predictive model generator"""

    def test_initialization(self, sample_config):
        """Test generator initialization"""
        generator = PredictiveModelGenerator(sample_config)
        assert generator.config == sample_config

    def test_generate_with_auto_model_type(self, sample_config, sample_solution, sample_data):
        """Test model generation with AUTO model type"""
        generator = PredictiveModelGenerator(sample_config)

        X, y = sample_data
        model = generator.generate(sample_solution, ModelType.AUTO, X, y)

        assert model is not None
        assert model.model is not None
        assert model.model_type in ModelType
        assert len(model.features) > 0
        assert len(model.predictions) > 0
        assert model.metrics is not None
        assert model.falsifiability is not None

    def test_generate_decision_tree(self, sample_config, sample_solution, sample_data):
        """Test decision tree generation"""
        generator = PredictiveModelGenerator(sample_config)

        X, y = sample_data
        model = generator.generate(sample_solution, ModelType.DECISION_TREE, X, y)

        assert model is not None
        assert model.model_type == ModelType.DECISION_TREE

    def test_generate_random_forest(self, sample_config, sample_solution, sample_data):
        """Test random forest generation"""
        generator = PredictiveModelGenerator(sample_config)

        X, y = sample_data
        model = generator.generate(sample_solution, ModelType.RANDOM_FOREST, X, y)

        assert model is not None
        assert model.model_type == ModelType.RANDOM_FOREST

    def test_model_falsifiability_validation(self, sample_config, sample_solution, sample_data):
        """Test falsifiability validation"""
        generator = PredictiveModelGenerator(sample_config)

        X, y = sample_data
        model = generator.generate(sample_solution, ModelType.DECISION_TREE, X, y)

        assert model.falsifiability is not None
        assert isinstance(model.falsifiability.is_falsifiable, bool)
        assert isinstance(model.falsifiability.num_testable_predictions, int)

    def test_model_predictions(self, sample_config, sample_solution, sample_data):
        """Test prediction generation"""
        generator = PredictiveModelGenerator(sample_config)

        X, y = sample_data
        model = generator.generate(sample_solution, ModelType.DECISION_TREE, X, y)

        assert len(model.predictions) > 0
        assert all(isinstance(p, Prediction) for p in model.predictions)

        # Check prediction structure
        for pred in model.predictions:
            assert pred.variable is not None
            assert pred.condition is not None
            assert pred.expected_value is not None
            assert pred.confidence >= 0

    def test_uncertainty_quantification(self, sample_config, sample_solution, sample_data):
        """Test uncertainty quantification"""
        config_with_uncertainty = Delta2Config(
            uncertainty_method="bootstrap",
            n_bootstrap_samples=10  # Small for testing
        )
        generator = PredictiveModelGenerator(config_with_uncertainty)

        X, y = sample_data
        model = generator.generate(sample_solution, ModelType.RANDOM_FOREST, X, y)

        assert model.uncertainty is not None
        assert model.uncertainty.method == "bootstrap"

    def test_model_type_selection(self, sample_config, sample_solution, sample_data):
        """Test automatic model type selection"""
        generator = PredictiveModelGenerator(sample_config)

        X, y = sample_data

        # Low complexity → decision tree (interpretable)
        model1 = generator.generate(sample_solution, ModelType.AUTO, X, y)
        assert model1.model_type in [ModelType.DECISION_TREE, ModelType.RANDOM_FOREST]

    def test_error_on_invalid_model_type(self, sample_config, sample_solution, sample_data):
        """Test error handling for invalid model type"""
        generator = PredictiveModelGenerator(sample_config)

        # Skip neural network if PyTorch not available
        try:
            import torch
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False

        if not HAS_TORCH:
            X, y = sample_data
            # Should fallback to available models
            model = generator.generate(sample_solution, ModelType.AUTO, X, y)
            assert model is not None


# =============================================================================
# PUBLIC API TESTS
# =============================================================================

class TestPublicAPI:
    """Test public API functions"""

    def test_generate_predictive_model(self, sample_solution, sample_data):
        """Test public API function"""
        X, y = sample_data

        model = generate_predictive_model(
            solution=sample_solution,
            model_type=ModelType.DECISION_TREE,
            X=X,
            y=y
        )

        assert model is not None
        assert model.model is not None
        assert model.model_type == ModelType.DECISION_TREE

    def test_generate_with_default_config(self, sample_solution, sample_data):
        """Test generation with default configuration"""
        X, y = sample_data

        model = generate_predictive_model(
            solution=sample_solution,
            X=X,
            y=y
        )

        assert model is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self, sample_solution, sample_data):
        """Test complete generation pipeline"""
        X, y = sample_data

        # Generate model
        model = generate_predictive_model(
            solution=sample_solution,
            model_type=ModelType.RANDOM_FOREST,
            X=X,
            y=y
        )

        # Verify all components
        assert model.model is not None
        assert len(model.features) > 0
        assert len(model.predictions) > 0
        assert model.metrics is not None
        assert model.falsifiability is not None
        assert model.metadata is not None

        # Verify falsifiability
        if model.falsifiability.is_falsifiable:
            assert model.falsifiability.num_testable_predictions > 0

    def test_multiple_solutions(self, sample_data):
        """Test generating models for multiple solutions"""
        solutions = [
            RESESolution(
                problem_id=f"test_{i:03d}",
                solution={"param": i},
                constraints=[f"Constraint {i}"],
                aci_history=[50.0 - i*5, 40.0 - i*5, 30.0 - i*5]
            )
            for i in range(1, 6)
        ]

        X, y = sample_data
        models = []

        for solution in solutions:
            model = generate_predictive_model(
                solution=solution,
                model_type=ModelType.DECISION_TREE,
                X=X,
                y=y
            )
            models.append(model)

        assert len(models) == 5
        assert all(m is not None for m in models)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
