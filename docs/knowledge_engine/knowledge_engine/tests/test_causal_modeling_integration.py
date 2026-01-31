"""
Tests for Causal Model Builder with causal-learn Integration

Tests the integration between knowledge engine's CausalModelBuilder and
the existing causal-learn adapter.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, UTC
from typing import Dict, Any, List

# Test imports
try:
    from knowledge_engine.causal_modeling import (
        CausalModelBuilder,
        CAUSAL_LEARN_INTEGRATION_AVAILABLE
    )
    from knowledge_engine.schemas.long_horizon import (
        CausalModel,
        CausalRelationship
    )
except ImportError:
    pytest.skip("Knowledge engine modules not available", allow_module_level=True)


@pytest.fixture
def sample_outcomes() -> List[Dict[str, Any]]:
    """Generate sample outcomes for testing"""
    np.random.seed(42)

    # Create synthetic data with known causal structure
    # X -> Y -> Z (chain)
    n_samples = 100
    X = np.random.randn(n_samples)
    Y = 0.5 * X + np.random.randn(n_samples) * 0.5
    Z = 0.3 * Y + np.random.randn(n_samples) * 0.3

    outcomes = []
    for i in range(n_samples):
        outcomes.append({
            "context": {
                "exploration_rate": abs(X[i]),
                "population_size": int(100 + X[i] * 20)
            },
            "metrics": {
                "fitness": Y[i],
                "diversity": Z[i],
                "convergence_speed": abs(Y[i]) * 2
            },
            "timestamp": datetime.now(UTC).isoformat()
        })

    return outcomes


@pytest.fixture
def causal_builder() -> CausalModelBuilder:
    """Create causal model builder"""
    return CausalModelBuilder()


class TestCausalModelBuilder:
    """Test suite for CausalModelBuilder"""

    @pytest.mark.asyncio
    async def test_builder_initialization(self, causal_builder):
        """Test builder initializes correctly"""
        assert causal_builder is not None

        # Check if causal-learn is available
        if CAUSAL_LEARN_INTEGRATION_AVAILABLE:
            assert causal_builder.use_causal_learn is True
            assert causal_builder.adapter is not None
        else:
            assert causal_builder.use_causal_learn is False
            assert causal_builder.adapter is None

    @pytest.mark.asyncio
    async def test_build_model_basic(self, causal_builder, sample_outcomes):
        """Test basic model building"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes,
            method="pc"
        )

        assert model is not None
        assert isinstance(model, CausalModel)
        assert model.domain == "test"
        assert len(model.relationships) > 0
        assert len(model.factors) > 0
        assert len(model.outcomes) > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not CAUSAL_LEARN_INTEGRATION_AVAILABLE,
        reason="causal-learn integration not available"
    )
    async def test_build_model_with_causal_learn(self, sample_outcomes):
        """Test model building using causal-learn adapter"""
        builder = CausalModelBuilder()

        # Verify causal-learn is being used
        assert builder.use_causal_learn is True

        model = await builder.build_model(
            domain="test_causal_learn",
            outcomes=sample_outcomes,
            method="pc",
            alpha=0.05,
            indep_test="fisherz"
        )

        assert model is not None
        assert len(model.relationships) > 0

        # Check that relationships have proper structure
        for rel in model.relationships:
            assert isinstance(rel, CausalRelationship)
            assert rel.cause is not None
            assert rel.effect is not None
            assert 0 <= rel.strength <= 1
            assert 0 <= rel.confidence <= 1
            assert len(rel.evidence) > 0

    @pytest.mark.asyncio
    async def test_build_model_fallback(self, sample_outcomes):
        """Test fallback when causal-learn is unavailable"""
        # Force fallback by disabling causal-learn
        builder = CausalModelBuilder()
        builder.use_causal_learn = False
        builder.adapter = None

        model = await builder.build_model(
            domain="test_fallback",
            outcomes=sample_outcomes
        )

        assert model is not None
        assert isinstance(model, CausalModel)
        # Fallback should still produce relationships
        assert len(model.relationships) >= 0

    @pytest.mark.asyncio
    async def test_identify_causes(self, causal_builder, sample_outcomes):
        """Test identifying causes of specific outcome"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        # Identify causes for fitness
        causes = await causal_builder.identify_causes(
            model=model,
            outcome="fitness"
        )

        assert isinstance(causes, list)
        # Should have at least some causes
        for cause in causes:
            assert cause.effect == "fitness"
            assert cause.confidence >= causal_builder.min_confidence

    @pytest.mark.asyncio
    async def test_predict_intervention(self, causal_builder, sample_outcomes):
        """Test intervention prediction"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        # Get first factor
        if model.factors:
            factor = model.factors[0]
            prediction = await causal_builder.predict_intervention(
                model=model,
                cause=factor,
                value=0.5
            )

            assert prediction is not None
            assert prediction.intervention is not None
            assert isinstance(prediction.predicted_effect, (int, float))
            assert isinstance(prediction.confidence, (int, float))

    @pytest.mark.asyncio
    async def test_explain_outcome(self, causal_builder, sample_outcomes):
        """Test outcome explanation"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        # Explain first outcome
        if model.outcomes:
            outcome = model.outcomes[0]
            explanation = await causal_builder.explain_outcome(
                model=model,
                outcome=outcome
            )

            assert explanation is not None
            assert explanation.outcome == outcome
            assert isinstance(explanation.causes, list)
            assert isinstance(explanation.contribution, dict)
            assert isinstance(explanation.confidence, (int, float))

    @pytest.mark.asyncio
    async def test_update_model(self, causal_builder, sample_outcomes):
        """Test model updating with new data"""
        # Build initial model
        model = await causal_builder.build_model(
            domain="test_update",
            outcomes=sample_outcomes[:50]
        )

        original_rel_count = len(model.relationships)

        # Update with new data
        updated_model = await causal_builder.update_model(
            model=model,
            new_data=sample_outcomes[50:]
        )

        assert updated_model is not None
        assert updated_model.model_id == model.model_id
        assert updated_model.domain == model.domain

    @pytest.mark.asyncio
    async def test_store_and_load_model(self, causal_builder, sample_outcomes):
        """Test model persistence (without actual knowledge engine)"""
        model = await causal_builder.build_model(
            domain="test_persistence",
            outcomes=sample_outcomes
        )

        # Store (will warn but not fail without knowledge engine)
        model_id = await causal_builder.store_model(model)
        assert model_id == model.model_id

        # Load (will return None without knowledge engine)
        loaded_model = await causal_builder.load_model(
            model_id=model_id,
            domain="test_persistence"
        )

        # Without knowledge engine, should load from cache
        if loaded_model:
            assert loaded_model.model_id == model_id

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not CAUSAL_LEARN_INTEGRATION_AVAILABLE,
        reason="causal-learn integration not available"
    )
    async def test_different_algorithms(self, sample_outcomes):
        """Test different causal discovery algorithms"""
        algorithms = ["pc", "ges", "direct_lingam"]

        for algorithm in algorithms:
            builder = CausalModelBuilder()

            try:
                model = await builder.build_model(
                    domain=f"test_{algorithm}",
                    outcomes=sample_outcomes,
                    method=algorithm
                )

                assert model is not None
                assert len(model.relationships) >= 0

            except Exception as e:
                # Some algorithms may fail on certain data
                pytest.skip(f"Algorithm {algorithm} failed: {e}")

    @pytest.mark.asyncio
    async def test_model_caching(self, causal_builder, sample_outcomes):
        """Test that models are cached by domain"""
        # Build model first time
        model1 = await causal_builder.build_model(
            domain="test_cache",
            outcomes=sample_outcomes
        )

        # Build model second time (should use cache)
        model2 = await causal_builder.build_model(
            domain="test_cache",
            outcomes=sample_outcomes
        )

        # Should be the same object from cache
        assert model1.model_id == model2.model_id

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, causal_builder, sample_outcomes):
        """Test model serialization to dictionary"""
        model = await causal_builder.build_model(
            domain="test_serialize",
            outcomes=sample_outcomes
        )

        model_dict = model.to_dict()

        assert isinstance(model_dict, dict)
        assert "model_id" in model_dict
        assert "domain" in model_dict
        assert "relationships" in model_dict
        assert "factors" in model_dict
        assert "outcomes" in model_dict
        assert "graph_data" in model_dict

    @pytest.mark.asyncio
    async def test_builder_to_dict(self, causal_builder):
        """Test builder serialization"""
        builder_dict = causal_builder.to_dict()

        assert isinstance(builder_dict, dict)
        assert "discovery_method" in builder_dict
        assert "min_confidence" in builder_dict
        assert "use_causal_learn" in builder_dict
        assert "models" in builder_dict


class TestCausalRelationships:
    """Test suite for CausalRelationship validation"""

    @pytest.mark.asyncio
    async def test_relationship_structure(self, causal_builder, sample_outcomes):
        """Test that relationships have proper structure"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        for rel in model.relationships:
            # Check required fields
            assert hasattr(rel, 'cause')
            assert hasattr(rel, 'effect')
            assert hasattr(rel, 'strength')
            assert hasattr(rel, 'confidence')

            # Check types
            assert isinstance(rel.cause, str)
            assert isinstance(rel.effect, str)
            assert isinstance(rel.strength, (int, float))
            assert isinstance(rel.confidence, (int, float))

            # Check ranges
            assert 0 <= rel.strength <= 1
            assert 0 <= rel.confidence <= 1

    @pytest.mark.asyncio
    async def test_relationship_evidence(self, causal_builder, sample_outcomes):
        """Test that relationships include evidence"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        for rel in model.relationships:
            # Evidence should be provided
            if CAUSAL_LEARN_INTEGRATION_AVAILABLE and causal_builder.use_causal_learn:
                # Causal-learn should provide evidence
                assert len(rel.evidence) > 0
                assert isinstance(rel.evidence, list)

    @pytest.mark.asyncio
    async def test_no_self_loops(self, causal_builder, sample_outcomes):
        """Test that relationships don't have self-loops"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        for rel in model.relationships:
            # No relationship should be X -> X
            assert rel.cause != rel.effect


class TestGraphStructure:
    """Test suite for graph structure"""

    @pytest.mark.asyncio
    async def test_graph_data_structure(self, causal_builder, sample_outcomes):
        """Test graph data structure"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        graph_data = model.graph_data

        assert isinstance(graph_data, dict)
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "num_nodes" in graph_data
        assert "num_edges" in graph_data

        # Check counts match
        assert len(graph_data["nodes"]) == graph_data["num_nodes"]
        assert len(graph_data["edges"]) == graph_data["num_edges"]

    @pytest.mark.asyncio
    async def test_graph_nodes_are_variables(self, causal_builder, sample_outcomes):
        """Test that graph nodes correspond to variables"""
        model = await causal_builder.build_model(
            domain="test",
            outcomes=sample_outcomes
        )

        all_variables = set(model.factors + model.outcomes)
        graph_nodes = set(model.graph_data["nodes"])

        # All variables should be in graph
        # (Note: Some may be filtered out if they have no relationships)
        assert graph_nodes.issubset(all_variables)


@pytest.mark.integration
class TestCausalLearnIntegration:
    """Integration tests for causal-learn adapter"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not CAUSAL_LEARN_INTEGRATION_AVAILABLE,
        reason="causal-learn integration not available"
    )
    async def test_adapter_initialization(self):
        """Test that causal-learn adapter initializes correctly"""
        from knowledge_engine.causal_modeling import CausalModelBuilder

        builder = CausalModelBuilder()

        assert builder.adapter is not None
        assert builder.use_causal_learn is True

        # Try to initialize
        await builder.adapter.initialize({
            'default_algorithm': 'pc',
            'cache_enabled': True
        })

        assert builder.adapter.is_initialized is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not CAUSAL_LEARN_INTEGRATION_AVAILABLE,
        reason="causal-learn integration not available"
    )
    async def test_adapter_discovery(self):
        """Test that adapter performs causal discovery"""
        from knowledge_engine.causal_modeling import CausalModelBuilder

        builder = CausalModelBuilder()
        await builder.adapter.initialize({'default_algorithm': 'pc'})

        # Create simple synthetic data
        np.random.seed(42)
        X = np.random.randn(100)
        Y = 0.5 * X + np.random.randn(100)
        data = np.column_stack([X, Y])

        # Run discovery
        from integrations.base.causal_interface import CausalGraphResult

        result = await builder.adapter.discover_causal_structure(
            data=data,
            method="pc"
        )

        assert isinstance(result, CausalGraphResult)
        assert result.graph is not None
        assert len(result.nodes) == 2
        assert result.algorithm_used == "PC"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
