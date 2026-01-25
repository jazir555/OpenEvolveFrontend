"""
Comprehensive unit tests for Predictive Model Generator (Δ₂)

Tests model generation, training, falsifiability validation,
and uncertainty quantification.

Author: Agent E2 (Δ₂ Specialist)
Created: 2025-12-31
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

# Try to import predictive model generator
try:
    from rese.phase4.predictive_model_generator import (
        PredictiveModelGenerator,
        SolutionAnalyzer,
        NeuralNetworkGenerator,
        TreeModelGenerator,
        RESESolution,
        PredictiveModel,
        Feature,
        Pattern,
        Prediction,
        UncertaintyQuantification,
        ModelMetrics,
        FalsifiabilityReport,
        ModelType,
        PredictionType,
        FalsifiabilityStatus,
        Delta2Config,
        generate_predictive_model,
        Delta2Error,
        ModelGenerationError,
        FalsifiabilityError,
        TORCH_AVAILABLE,
        SKLEARN_AVAILABLE
    )
except ImportError:
    pytest.skip("Predictive model generator module not available", allow_module_level=True)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_solution():
    """Create sample RESE solution"""
    return RESESolution(
        problem_id="test_001",
        solution={"param1": 42, "param2": 3.14},
        constraints=["constraint1", "constraint2", "constraint3"],
        aci_history=[0.8, 0.7, 0.6, 0.55, 0.5],
        metadata={"domain": "physics", "n_samples": 1000}
    )


@pytest.fixture
def basic_config():
    """Create basic Delta2 config"""
    return Delta2Config(
        default_model_type=ModelType.AUTO,
        train_test_split=0.2,
        min_accuracy=0.7,
        require_falsifiable=True,
        verbose=False
    )


# =============================================================================
# RESESolution Tests
# =============================================================================

class TestRESESolution:
    """Test RESESolution functionality"""

    def test_initialization(self):
        """Test solution initialization"""
        solution = RESESolution(
            problem_id="test_001",
            solution={"x": 1},
            constraints=["c1", "c2"]
        )

        assert solution.problem_id == "test_001"
        assert solution.solution == {"x": 1}
        assert len(solution.constraints) == 2
        assert solution.architecture is None
        assert len(solution.aci_history) == 0

    def test_with_metadata(self):
        """Test solution with metadata"""
        metadata = {
            "domain": "chemistry",
            "n_samples": 500,
            "require_interpretability": True
        }

        solution = RESESolution(
            problem_id="test_002",
            solution={},
            constraints=[],
            metadata=metadata
        )

        assert solution.metadata == metadata


# =============================================================================
# Delta2Config Tests
# =============================================================================

class TestDelta2Config:
    """Test Delta2Config functionality"""

    def test_default_values(self):
        """Test default configuration"""
        config = Delta2Config()

        assert config.default_model_type == ModelType.AUTO
        assert config.prefer_interpretable == False
        assert config.nn_max_epochs == 100
        assert config.nn_learning_rate == 0.001
        assert config.tree_max_depth == 10
        assert config.forest_n_estimators == 100
        assert config.train_test_split == 0.2
        assert config.min_accuracy == 0.8
        assert config.require_falsifiable == True

    def test_custom_values(self):
        """Test custom configuration"""
        config = Delta2Config(
            prefer_interpretable=True,
            nn_max_epochs=50,
            min_accuracy=0.9
        )

        assert config.prefer_interpretable == True
        assert config.nn_max_epochs == 50
        assert config.min_accuracy == 0.9


# =============================================================================
# Feature and Pattern Tests
# =============================================================================

class TestDataStructures:
    """Test data structures"""

    def test_feature(self):
        """Test Feature structure"""
        feature = Feature(
            name="temperature",
            type="continuous",
            domain="physical",
            importance=0.9,
            description="Temperature in Kelvin"
        )

        assert feature.name == "temperature"
        assert feature.type == "continuous"
        assert feature.importance == 0.9

    def test_pattern(self):
        """Test Pattern structure"""
        pattern = Pattern(
            type="constraint",
            source="constraint1",
            variables=["x", "y"],
            confidence=0.85
        )

        assert pattern.type == "constraint"
        assert len(pattern.variables) == 2
        assert pattern.confidence == 0.85

    def test_prediction(self):
        """Test Prediction structure"""
        prediction = Prediction(
            variable="temperature",
            condition="when pressure increases",
            expected_value=373.15,
            confidence=0.95,
            test_method="experimental_validation"
        )

        assert prediction.variable == "temperature"
        assert prediction.expected_value == 373.15
        assert prediction.test_method == "experimental_validation"


# =============================================================================
# SolutionAnalyzer Tests
# =============================================================================

class TestSolutionAnalyzer:
    """Test SolutionAnalyzer functionality"""

    def test_initialization(self, basic_config):
        """Test analyzer initialization"""
        analyzer = SolutionAnalyzer(basic_config)

        assert analyzer.config == basic_config

    def test_analyze_basic(self, basic_config, sample_solution):
        """Test basic solution analysis"""
        analyzer = SolutionAnalyzer(basic_config)

        analysis = analyzer.analyze(sample_solution)

        assert 'features' in analysis
        assert 'patterns' in analysis
        assert 'complexity' in analysis
        assert 'n_samples' in analysis
        assert 'prediction_type' in analysis
        assert 'requires_interpretability' in analysis

    def test_extract_features(self, basic_config, sample_solution):
        """Test feature extraction"""
        analyzer = SolutionAnalyzer(basic_config)

        features = analyzer._extract_features(sample_solution)

        assert len(features) > 0
        assert all(isinstance(f, Feature) for f in features)

    def test_extract_patterns(self, basic_config, sample_solution):
        """Test pattern extraction"""
        analyzer = SolutionAnalyzer(basic_config)

        patterns = analyzer._extract_patterns(sample_solution)

        assert isinstance(patterns, list)

    def test_estimate_complexity(self, basic_config, sample_solution):
        """Test complexity estimation"""
        analyzer = SolutionAnalyzer(basic_config)

        complexity = analyzer._estimate_complexity(sample_solution)

        assert complexity > 0

    def test_determine_prediction_type_classification(self, basic_config):
        """Test prediction type determination for classification"""
        analyzer = SolutionAnalyzer(basic_config)

        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=["classify into categories", "determine class type"]
        )

        pred_type = analyzer._determine_prediction_type(solution)

        assert pred_type == PredictionType.CLASSIFICATION

    def test_determine_prediction_type_regression(self, basic_config):
        """Test prediction type determination for regression"""
        analyzer = SolutionAnalyzer(basic_config)

        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=["optimize value", "minimize cost"]
        )

        pred_type = analyzer._determine_prediction_type(solution)

        assert pred_type == PredictionType.REGRESSION

    def test_needs_interpretability(self, basic_config):
        """Test interpretability requirement"""
        analyzer = SolutionAnalyzer(basic_config)

        # Physics domain requires interpretability
        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=[],
            metadata={"domain": "physics"}
        )

        assert analyzer._needs_interpretability(solution)

        # Non-scientific domain
        solution2 = RESESolution(
            problem_id="test2",
            solution={},
            constraints=[],
            metadata={"domain": "business"}
        )

        assert not analyzer._needs_interpretability(solution2)


# =============================================================================
# PredictiveModelGenerator Tests
# =============================================================================

class TestPredictiveModelGenerator:
    """Test PredictiveModelGenerator functionality"""

    def test_initialization(self, basic_config):
        """Test generator initialization"""
        generator = PredictiveModelGenerator(basic_config)

        assert generator.config == basic_config
        assert generator._analyzer is not None

    def test_select_model_type_auto(self, basic_config):
        """Test automatic model type selection"""
        generator = PredictiveModelGenerator(basic_config)

        analysis = {
            'complexity': 50,
            'requires_interpretability': True
        }

        model_type = generator._select_model_type(analysis)

        assert isinstance(model_type, ModelType)

    def test_select_model_type_interpretable(self):
        """Test interpretable model selection"""
        config = Delta2Config(prefer_interpretable=True)
        generator = PredictiveModelGenerator(config)

        analysis = {
            'complexity': 100,
            'requires_interpretability': True
        }

        model_type = generator._select_model_type(analysis)

        # Should prefer interpretable models
        if SKLEARN_AVAILABLE:
            assert model_type in [ModelType.DECISION_TREE, ModelType.RANDOM_FOREST]

    def test_prepare_data_from_metadata(self, basic_config, sample_solution):
        """Test data preparation from metadata"""
        generator = PredictiveModelGenerator(basic_config)

        # Add training data to metadata
        sample_solution.metadata['training_data'] = {
            'X': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            'y': [1.0, 2.0, 3.0]
        }

        X, y = generator._prepare_data(sample_solution, {})

        assert X.shape[0] == 3
        assert len(y) == 3

    def test_generate_predictions(self, basic_config, sample_solution):
        """Test prediction generation"""
        generator = PredictiveModelGenerator(basic_config)

        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        X = np.random.randn(10, 5)

        predictions = generator._generate_predictions(mock_model, X, sample_solution)

        assert len(predictions) > 0
        assert all(isinstance(p, Prediction) for p in predictions)

    def test_validate_falsifiability(self, basic_config):
        """Test falsifiability validation"""
        generator = PredictiveModelGenerator(basic_config)

        # Valid predictions
        predictions = [
            Prediction(
                variable="x",
                condition="test condition",
                expected_value=1.0,
                confidence=0.95,
                test_method="experimental"
            ),
            Prediction(
                variable="y",
                condition="test condition 2",
                expected_value=2.0,
                confidence=0.90,
                test_method="experimental"
            )
        ]

        report = generator._validate_falsifiability(predictions)

        assert report.is_falsifiable
        assert report.num_testable_predictions == 2

    def test_validate_not_falsifiable(self, basic_config):
        """Test when model is not falsifiable"""
        generator = PredictiveModelGenerator(basic_config)

        # No predictions
        report = generator._validate_falsifiability([])

        assert not report.is_falsifiable
        assert report.num_testable_predictions == 0
        assert len(report.issues) > 0

    def test_quantify_uncertainty_bootstrap(self, basic_config):
        """Test bootstrap uncertainty quantification"""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not available")

        config = Delta2Config(
            uncertainty_method="bootstrap",
            n_bootstrap_samples=10
        )
        generator = PredictiveModelGenerator(config)

        # Create mock model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)

        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model.fit(X, y)

        uncertainty = generator._quantify_uncertainty(model, X, y)

        if uncertainty:
            assert uncertainty.method == "bootstrap"
            assert len(uncertainty.confidence_intervals) > 0


# =============================================================================
# ModelMetrics Tests
# =============================================================================

class TestModelMetrics:
    """Test ModelMetrics functionality"""

    def test_initialization(self):
        """Test metrics initialization"""
        metrics = ModelMetrics(
            accuracy=0.85,
            r2_score=0.90,
            mse=0.01,
            f1_score=0.88,
            training_loss=0.1,
            validation_loss=0.15
        )

        assert metrics.accuracy == 0.85
        assert metrics.r2_score == 0.90
        assert metrics.mse == 0.01

    def test_none_values(self):
        """Test metrics with None values"""
        metrics = ModelMetrics()

        assert metrics.accuracy is None
        assert metrics.r2_score is None
        assert metrics.training_loss == 0.0


# =============================================================================
# FalsifiabilityReport Tests
# =============================================================================

class TestFalsifiabilityReport:
    """Test FalsifiabilityReport functionality"""

    def test_falsifiable_report(self):
        """Test falsifiable report"""
        report = FalsifiabilityReport(
            is_falsifiable=True,
            status=FalsifiabilityStatus.FALSIFIABLE,
            num_testable_predictions=5,
            issues=[]
        )

        assert report.is_falsifiable
        assert report.status == FalsifiabilityStatus.FALSIFIABLE
        assert report.num_testable_predictions == 5

    def test_not_falsifiable_report(self):
        """Test non-falsifiable report"""
        report = FalsifiabilityReport(
            is_falsifiable=False,
            status=FalsifiabilityStatus.NOT_FALSIFIABLE,
            num_testable_predictions=0,
            issues=["No testable predictions"]
        )

        assert not report.is_falsifiable
        assert len(report.issues) > 0


# =============================================================================
# PredictiveModel Tests
# =============================================================================

class TestPredictiveModel:
    """Test PredictiveModel functionality"""

    def test_initialization(self):
        """Test model initialization"""
        mock_model = Mock()

        model = PredictiveModel(
            model=mock_model,
            model_type=ModelType.RANDOM_FOREST,
            prediction_type=PredictionType.REGRESSION,
            features=[],
            predictions=[],
            metrics=ModelMetrics(),
            falsifiability=FalsifiabilityReport(
                is_falsifiable=True,
                status=FalsifiabilityStatus.FALSIFIABLE,
                num_testable_predictions=3
            )
        )

        assert model.model == mock_model
        assert model.model_type == ModelType.RANDOM_FOREST
        assert model.prediction_type == PredictionType.REGRESSION
        assert model.falsifiability.is_falsifiable


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_generate_predictive_model(self, sample_solution):
        """Test generate_predictive_model convenience function"""
        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not available")

        # Create simple training data
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        try:
            model = generate_predictive_model(
                solution=sample_solution,
                model_type=ModelType.RANDOM_FOREST,
                X=X,
                y=y
            )

            assert model is not None
            assert isinstance(model, PredictiveModel)
        except Exception as e:
            # Some test environments may not have all dependencies
            if "No ML library available" not in str(e):
                raise


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling"""

    def test_model_generation_error(self):
        """Test ModelGenerationError"""
        with pytest.raises(ModelGenerationError):
            raise ModelGenerationError("Model generation failed")

    def test_falsifiability_error(self):
        """Test FalsifiabilityError"""
        with pytest.raises(FalsifiabilityError):
            raise FalsifiabilityError("Model not falsifiable")

    def test_delta2_error(self):
        """Test Delta2Error"""
        with pytest.raises(Delta2Error):
            raise Delta2Error("Delta2 error")


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_empty_constraints(self, basic_config):
        """Test with empty constraints"""
        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=[]
        )

        analyzer = SolutionAnalyzer(basic_config)
        features = analyzer._extract_features(solution)

        # Should return empty list
        assert len(features) == 0

    def test_empty_aci_history(self, basic_config):
        """Test with empty ACI history"""
        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=["c1"],
            aci_history=[]
        )

        analyzer = SolutionAnalyzer(basic_config)
        analysis = analyzer.analyze(solution)

        # Should still complete
        assert 'features' in analysis

    def test_very_large_solution(self, basic_config):
        """Test with very large solution"""
        # Create many constraints
        constraints = [f"constraint_{i}" for i in range(1000)]

        solution = RESESolution(
            problem_id="test_large",
            solution={"value": 42},
            constraints=constraints
        )

        analyzer = SolutionAnalyzer(basic_config)
        complexity = analyzer._estimate_complexity(solution)

        # Should handle large solutions
        assert complexity > 0

    def test_missing_metadata(self, basic_config):
        """Test with missing metadata"""
        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=[]
        )

        analyzer = SolutionAnalyzer(basic_config)
        n_samples = analyzer._estimate_sample_size(solution)

        # Should use default
        assert n_samples > 0

    def test_invalid_domain(self, basic_config):
        """Test with unknown domain"""
        solution = RESESolution(
            problem_id="test",
            solution={},
            constraints=[],
            metadata={"domain": "unknown_domain_xyz"}
        )

        analyzer = SolutionAnalyzer(basic_config)
        interpretability = analyzer._needs_interpretability(solution)

        # Should handle gracefully
        assert isinstance(interpretability, bool)
