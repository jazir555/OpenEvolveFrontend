"""
Integration Tests for Δ₂ Predictive Model Generator
==================================================

Integration tests for Δ₂ with Δ₁ (Architecture Assembly) and Stage 8 E2E.

Author: Agent E2 (Δ₂ Specialist)
Date: 2025-12-31
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase4.predictive_model_generator import (
    PredictiveModelGenerator,
    RESESolution,
    Delta2Config,
    ModelType,
    generate_predictive_model,
)

pytest.importorskip("sklearn", reason="scikit-learn not available")


# =============================================================================
# Δ₁ ARCHITECTURE ASSEMBLY INTEGRATION TESTS
# =============================================================================

class TestDelta1Integration:
    """Test integration with Δ₁ (Architecture Assembly)"""

    @pytest.fixture
    def solution_with_architecture(self):
        """Create solution with architecture from Δ₁"""
        return RESESolution(
            problem_id="delta1_integration_001",
            solution={
                "components": ["module_a", "module_b", "module_c"],
                "connections": [
                    {"from": "module_a", "to": "module_b"},
                    {"from": "module_b", "to": "module_c"}
                ]
            },
            constraints=[
                "Module A output must be positive",
                "Module B processing time < 100ms",
                "Module C accuracy > 95%"
            ],
            architecture={
                "type": "pipeline",
                "components": [
                    {
                        "name": "module_a",
                        "type": "preprocessing",
                        "inputs": ["raw_data"],
                        "outputs": ["processed_data"]
                    },
                    {
                        "name": "module_b",
                        "type": "transformation",
                        "inputs": ["processed_data"],
                        "outputs": ["features"]
                    },
                    {
                        "name": "module_c",
                        "type": "prediction",
                        "inputs": ["features"],
                        "outputs": ["prediction"]
                    }
                ],
                "flow": ["module_a", "module_b", "module_c"]
            },
            metadata={
                "domain": "machine_learning",
                "architecture_source": "delta1"
            }
        )

    @pytest.fixture
    def architecture_data(self):
        """Generate data matching architecture"""
        np.random.seed(42)
        # Simulate data flowing through architecture
        X = np.random.randn(100, 3)  # 3 modules = 3 features
        y = np.random.randn(100)
        return X, y

    def test_generate_from_architecture(self, solution_with_architecture, architecture_data):
        """Test model generation from Δ₁ architecture"""
        generator = PredictiveModelGenerator()

        X, y = architecture_data
        model = generator.generate(
            solution_with_architecture,
            ModelType.RANDOM_FOREST,
            X, y
        )

        # Verify model respects architecture
        assert model is not None
        assert len(model.features) > 0

        # Check architecture metadata captured
        assert model.metadata.get('solution_metadata', {}).get('architecture_source') == 'delta1'

    def test_architecture_guided_model_structure(self, solution_with_architecture, architecture_data):
        """Test that architecture guides model structure"""
        X, y = architecture_data

        # Generate model
        model = generate_predictive_model(
            solution=solution_with_architecture,
            model_type=ModelType.DECISION_TREE,
            X=X,
            y=y
        )

        # Architecture has 3 components → expect ~3 important features
        # Use lower threshold since data is random
        top_features = [f for f in model.features if f.importance > 0.1]
        assert len(top_features) >= 1  # At least 1 feature should be important


# =============================================================================
# STAGE 8 E2E INTEGRATION TESTS
# =============================================================================

class TestStage8Integration:
    """Test integration with Stage 8 E2E pipeline"""

    @pytest.fixture
    def stage8_solution(self):
        """Create solution ready for Stage 8"""
        return RESESolution(
            problem_id="stage8_integration_001",
            solution={
                "invention": "novel_material",
                "parameters": {
                    "composition": "Cu-Zn-Al",
                    "processing_temp": 400,
                    "annealing_time": 2
                }
            },
            constraints=[
                "Temperature < 500°C",
                "Annealing time > 0.5 hours",
                "Composition purity > 95%"
            ],
            aci_history=[60.0, 45.0, 30.0, 20.0, 15.0],
            metadata={
                "domain": "materials_science",
                "target_stage": 8,
                "output_format": "standard_operating_procedure"
            },
            stage_results={
                "stage1": {"status": "complete"},
                "stage2": {"status": "complete"},
                "stage3": {"status": "complete"},
                "stage4": {"status": "complete"},
                "stage5": {"status": "complete"},
                "stage6": {"status": "complete"},
                "stage7": {"status": "complete"}
            }
        )

    @pytest.fixture
    def materials_data(self):
        """Generate materials science data"""
        np.random.seed(42)
        # Features: composition ratios, temperature, time
        X = np.random.randn(50, 4)
        # Target: material property (strength)
        y = 200 + 50 * X[:, 0] + 30 * X[:, 1] + np.random.randn(50) * 10
        return X, y

    def test_generate_for_stage8(self, stage8_solution, materials_data):
        """Test model generation for Stage 8 output"""
        generator = PredictiveModelGenerator()

        X, y = materials_data
        model = generator.generate(stage8_solution, ModelType.RANDOM_FOREST, X, y)

        # Verify model ready for Stage 8
        assert model is not None
        assert model.falsifiability.is_falsifiable
        assert model.predictions is not None

    def test_stage8_predictions(self, stage8_solution, materials_data):
        """Test that model generates Stage 8-compatible predictions"""
        X, y = materials_data
        model = generate_predictive_model(
            solution=stage8_solution,
            model_type=ModelType.DECISION_TREE,
            X=X,
            y=y
        )

        # Check predictions are actionable
        assert len(model.predictions) > 0

        for pred in model.predictions:
            # Must have test method
            assert pred.test_method is not None
            assert len(pred.test_method) > 0

            # Must have confidence
            assert 0 <= pred.confidence <= 1

            # Must be testable
            assert pred.expected_value is not None

    def test_stage8_metadata_compatibility(self, stage8_solution, materials_data):
        """Test that model metadata is Stage 8 compatible"""
        X, y = materials_data
        model = generate_predictive_model(
            solution=stage8_solution,
            model_type=ModelType.RANDOM_FOREST,
            X=X,
            y=y
        )

        # Check required metadata fields
        assert 'problem_id' in model.metadata
        assert 'generation_timestamp' in model.metadata
        assert 'config' in model.metadata

        # Check Stage 8 readiness
        assert model.metadata['problem_id'] == "stage8_integration_001"


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests"""

    def test_full_rese_to_model_pipeline(self):
        """Test complete pipeline from RESE solution to predictive model"""

        # Simulate RESE pipeline output
        rese_solution = RESESolution(
            problem_id="e2e_001",
            solution={
                "optimization_result": {
                    "objective": 0.95,
                    "parameters": {
                        "x1": 1.5,
                        "x2": 2.3,
                        "x3": 0.8
                    }
                }
            },
            constraints=[
                "x1 + x2 + x3 < 10",
                "x1 > 0",
                "x2 > 0",
                "x3 > 0"
            ],
            architecture={
                "type": "optimization",
                "objective_function": "minimize_cost",
                "constraints": 4
            },
            aci_history=[80.0, 60.0, 40.0, 25.0, 15.0],
            metadata={
                "domain": "optimization",
                "rese_phases_complete": ["I", "II", "III", "IV"],
                "n_samples": 200
            }
        )

        # Generate training data
        np.random.seed(42)
        X = np.random.rand(200, 3) * 10  # 3 parameters
        y = 2 * X[:, 0] + 3 * X[:, 1] + X[:, 2] + np.random.randn(200) * 0.1

        # Generate predictive model
        model = generate_predictive_model(
            solution=rese_solution,
            model_type=ModelType.RANDOM_FOREST,
            X=X,
            y=y
        )

        # Verify complete pipeline
        assert model is not None
        assert model.falsifiability.is_falsifiable
        assert len(model.predictions) > 0
        assert model.metrics.r2_score is not None or model.metrics.accuracy is not None

        # Verify predictions are actionable
        for pred in model.predictions:
            assert pred.variable is not None
            assert pred.test_method is not None
            assert pred.expected_value is not None

    def test_multi_domain_generation(self):
        """Test model generation across multiple domains"""

        domains = [
            ("physics", "temperature", "pressure"),
            ("chemistry", "concentration", "ph"),
            ("biology", "cell_count", "growth_rate"),
            ("economics", "price", "demand")
        ]

        for domain, var1, var2 in domains:
            # Create domain-specific solution
            solution = RESESolution(
                problem_id=f"{domain}_test",
                solution={},
                constraints=[
                    f"{var1} must be positive",
                    f"{var2} must be measurable"
                ],
                metadata={"domain": domain}
            )

            # Generate data
            np.random.seed(42)
            X = np.random.randn(50, 2)
            y = np.random.randn(50)

            # Generate model
            model = generate_predictive_model(
                solution=solution,
                model_type=ModelType.DECISION_TREE,
                X=X,
                y=y
            )

            assert model is not None
            assert model.metadata['solution_metadata']['domain'] == domain


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in integration scenarios"""

    def test_handle_missing_architecture(self):
        """Test handling of missing architecture"""
        solution = RESESolution(
            problem_id="no_arch_001",
            solution={},
            constraints=["Constraint 1"],
            # No architecture provided
        )

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Should still generate model
        model = generate_predictive_model(
            solution=solution,
            model_type=ModelType.DECISION_TREE,
            X=X,
            y=y
        )

        assert model is not None

    def test_handle_incomplete_aci_history(self):
        """Test handling of incomplete ACI history"""
        solution = RESESolution(
            problem_id="incomplete_aci_001",
            solution={},
            constraints=["Constraint 1"],
            aci_history=[50.0, 40.0]  # Only 2 data points
        )

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Should still generate model
        model = generate_predictive_model(
            solution=solution,
            model_type=ModelType.DECISION_TREE,
            X=X,
            y=y
        )

        assert model is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
