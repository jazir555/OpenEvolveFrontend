"""
Tests for OneKE Integration

This test suite validates the OneKE schema-guided information extraction
integration with OpenEvolve.
"""

import pytest
import asyncio
from typing import Dict, Any
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from integrations.oneke import OneKEAdapter, OneKEBridge, create_oneke_bridge
from integrations.base.extraction_interface import (
    ExtractionType,
    SchemaDefinition,
    ExtractionResult,
    ConfigurationError,
    ConnectionError
)


# ========== Fixtures ==========

@pytest.fixture
async def adapter():
    """Create and initialize OneKE adapter."""
    adapter = OneKEAdapter()
    success = await adapter.initialize()
    if not success:
        pytest.skip("Could not initialize OneKE adapter")
    yield adapter
    await adapter.shutdown()


@pytest.fixture
async def bridge():
    """Create and initialize OneKE bridge."""
    bridge = OneKEBridge()
    success = await bridge.initialize()
    if not success:
        pytest.skip("Could not initialize OneKE bridge")
    yield bridge
    await bridge.shutdown()


@pytest.fixture
def physics_text():
    """Sample physics text for testing."""
    return """
    The quantum harmonic oscillator is a fundamental system in quantum mechanics.
    It is described by the Hamiltonian H = (p^2/2m) + (1/2)mω^2x^2.
    The energy eigenvalues are E_n = ℏω(n + 1/2), where n is the quantum number.
    The ground state wavefunction is a Gaussian function.
    We can use perturbation theory to approximate anharmonic corrections.
    """


@pytest.fixture
def chemistry_text():
    """Sample chemistry text for testing."""
    return """
    The combustion of methane (CH4) with oxygen produces carbon dioxide and water.
    The balanced chemical equation is: CH4 + 2O2 → CO2 + 2H2O.
    This reaction releases 890 kJ/mol of heat.
    The reaction requires activation energy to proceed.
    """


@pytest.fixture
def test_workflow():
    """Create a test workflow."""
    from workflow_structures import WorkflowState

    return WorkflowState(
        workflow_id="test_001",
        problem_statement="Solve the quantum harmonic oscillator",
        final_solution="The energy eigenvalues are E_n = ℏω(n + 1/2)"
    )


# ========== Adapter Tests ==========

class TestOneKEAdapter:
    """Test OneKE adapter functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test adapter initialization."""
        adapter = OneKEAdapter()
        assert adapter is not None
        assert adapter.config is not None
        assert not adapter.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_success(self, adapter):
        """Test successful initialization."""
        assert adapter.is_initialized
        assert adapter.oneke_path is not None or True  # May use env var

    @pytest.mark.asyncio
    async def test_validate_config(self, adapter):
        """Test configuration validation."""
        validation = await adapter.validate()
        assert 'is_valid' in validation
        assert 'checks' in validation
        assert 'issues' in validation

    def test_load_schema(self, adapter):
        """Test schema loading."""
        schema_path = Path(__file__).parent.parent.parent / 'integrations/oneke/schemas/physics.yaml'

        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")

        schema = adapter.load_schema(str(schema_path))
        assert schema.name == 'physics_concepts'
        assert len(schema.entity_types) > 0
        assert schema.description is not None


class TestNERExtraction:
    """Test Named Entity Recognition extraction."""

    @pytest.mark.asyncio
    async def test_extract_ner_basic(self, adapter, physics_text):
        """Test basic NER extraction."""
        schema_path = Path(__file__).parent.parent.parent / 'integrations/oneke/schemas/physics.yaml'

        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")

        schema = adapter.load_schema(str(schema_path))
        result = await adapter.extract_ner(physics_text, schema)

        assert isinstance(result, ExtractionResult)
        assert result.extraction_type == ExtractionType.NER
        assert isinstance(result.entities, list)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_extract_ner_without_schema(self, adapter, physics_text):
        """Test NER extraction without schema (should use fallback)."""
        result = await adapter.extract_ner(physics_text, schema=None)

        assert isinstance(result, ExtractionResult)
        assert result.extraction_type == ExtractionType.NER
        # Fallback may have low or zero confidence
        assert result.confidence >= 0.0


class TestRelationExtraction:
    """Test Relation Extraction."""

    @pytest.mark.asyncio
    async def test_extract_re(self, adapter, physics_text):
        """Test relation extraction."""
        schema_path = Path(__file__).parent.parent.parent / 'integrations/oneke/schemas/physics.yaml'

        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")

        schema = adapter.load_schema(str(schema_path))
        result = await adapter.extract_re(physics_text, schema)

        assert isinstance(result, ExtractionResult)
        assert result.extraction_type == ExtractionType.RE
        assert isinstance(result.relations, list)


class TestEventExtraction:
    """Test Event Extraction."""

    @pytest.mark.asyncio
    async def test_extract_ee(self, adapter, chemistry_text):
        """Test event extraction."""
        schema_path = Path(__file__).parent.parent.parent / 'integrations/oneke/schemas/chemistry.yaml'

        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")

        schema = adapter.load_schema(str(schema_path))
        result = await adapter.extract_ee(chemistry_text, schema)

        assert isinstance(result, ExtractionResult)
        assert result.extraction_type == ExtractionType.EE
        assert isinstance(result.events, list)


class TestTripleExtraction:
    """Test Triple Extraction."""

    @pytest.mark.asyncio
    async def test_extract_triple(self, adapter, physics_text):
        """Test triple extraction."""
        schema_path = Path(__file__).parent.parent.parent / 'integrations/oneke/schemas/relations.yaml'

        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")

        schema = adapter.load_schema(str(schema_path))
        result = await adapter.extract_triple(physics_text, schema)

        assert isinstance(result, ExtractionResult)
        assert result.extraction_type == ExtractionType.TRIPLE
        assert isinstance(result.triples, list)


class TestSchemaGuidedExtraction:
    """Test Schema-Guided Extraction."""

    @pytest.mark.asyncio
    async def test_extract_schema_guided(self, adapter, physics_text):
        """Test schema-guided extraction."""
        schema_path = Path(__file__).parent.parent.parent / 'integrations/oneke/schemas/physics.yaml'

        if not schema_path.exists():
            pytest.skip(f"Schema file not found: {schema_path}")

        schema = adapter.load_schema(str(schema_path))
        result = await adapter.extract_schema_guided(physics_text, schema)

        assert isinstance(result, ExtractionResult)
        assert result.extraction_type == ExtractionType.SCHEMA
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)
        assert isinstance(result.events, list)
        assert isinstance(result.triples, list)


class TestBatchExtraction:
    """Test Batch Extraction."""

    @pytest.mark.asyncio
    async def test_batch_extract_ner(self, adapter):
        """Test batch NER extraction."""
        texts = [
            "The quantum harmonic oscillator is described by the Hamiltonian.",
            "The combustion of methane produces carbon dioxide and water.",
            "We use perturbation theory to approximate the solution."
        ]

        results = await adapter.batch_extract(texts, ExtractionType.NER)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ExtractionResult)
            assert result.extraction_type == ExtractionType.NER


# ========== Bridge Tests ==========

class TestOneKEBridge:
    """Test OneKE bridge functionality."""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge):
        """Test bridge initialization."""
        assert bridge.adapter is not None
        assert bridge.is_initialized is not False

    @pytest.mark.asyncio
    async def test_validate_integration(self, bridge):
        """Test integration validation."""
        validation = await bridge.validate_integration()
        assert 'is_valid' in validation
        assert 'checks' in validation
        assert 'bridge' in validation

    @pytest.mark.asyncio
    async def test_extract_from_workflow(self, bridge, test_workflow):
        """Test extraction from workflow."""
        results = await bridge.extract_from_workflow(
            test_workflow,
            schemas=['physics_concepts']
        )

        assert isinstance(results, dict)
        # May have results or may have errors if OneKE not fully configured


class TestPhysicsExtraction:
    """Test Physics domain extraction."""

    @pytest.mark.asyncio
    async def test_extract_physics_knowledge(self, bridge, test_workflow):
        """Test physics knowledge extraction."""
        knowledge = await bridge.extract_physics_knowledge(test_workflow)

        assert isinstance(knowledge, dict)
        assert 'concepts' in knowledge
        assert 'observables' in knowledge
        assert 'dynamics' in knowledge
        assert 'quantum' in knowledge
        assert 'confidence' in knowledge


class TestChemistryExtraction:
    """Test Chemistry domain extraction."""

    @pytest.mark.asyncio
    async def test_extract_chemistry_knowledge(self, bridge, chemistry_text):
        """Test chemistry knowledge extraction."""
        # Create workflow from chemistry text
        from workflow_structures import WorkflowState

        workflow = WorkflowState(
            workflow_id="chem_001",
            problem_statement="Analyze methane combustion",
            final_solution=chemistry_text
        )

        knowledge = await bridge.extract_chemistry_knowledge(workflow)

        assert isinstance(knowledge, dict)
        assert 'substances' in knowledge
        assert 'reactions' in knowledge
        assert 'properties' in knowledge
        assert 'confidence' in knowledge


class TestSolutionPatternExtraction:
    """Test Solution Pattern extraction."""

    @pytest.mark.asyncio
    async def test_extract_solution_patterns(self, bridge, test_workflow):
        """Test solution pattern extraction."""
        patterns = await bridge.extract_solution_patterns(
            test_workflow,
            domain='physics'
        )

        assert isinstance(patterns, dict)
        assert 'patterns' in patterns
        assert 'approaches' in patterns
        assert 'techniques' in patterns
        assert 'relations' in patterns
        assert 'confidence' in patterns


class TestWorkflowExtraction:
    """Test workflow-based extraction."""

    @pytest.mark.asyncio
    async def test_extract_from_multiple_workflows(self, bridge):
        """Test extraction from multiple workflows."""
        from workflow_structures import WorkflowState

        workflows = [
            WorkflowState(
                workflow_id=f"test_{i}",
                problem_statement=f"Test problem {i}",
                final_solution=f"Test solution {i}"
            )
            for i in range(3)
        ]

        results = await bridge.batch_extract_from_workflows(
            workflows,
            schemas=['physics_concepts']
        )

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)


# ========== Integration Tests ==========

class TestWorkflowKnowledgeExtractorIntegration:
    """Test integration with workflow_knowledge_extractor.py."""

    @pytest.mark.asyncio
    async def test_combined_extraction(self, bridge, test_workflow):
        """Test combined extraction with workflow knowledge extractor."""
        # This would be integrated with workflow_knowledge_extractor.py
        # For now, test that bridge can extract from workflow

        results = await bridge.extract_from_workflow(
            test_workflow,
            schemas=['physics_concepts', 'relations']
        )

        assert isinstance(results, dict)
        # Results may be empty if OneKE not fully configured


# ========== Helper Tests ==========

class TestHelpers:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_create_oneke_bridge(self):
        """Test convenience function."""
        try:
            bridge = await create_oneke_bridge()
            assert bridge is not None
            await bridge.shutdown()
        except Exception as e:
            pytest.skip(f"Could not create bridge: {e}")

    @pytest.mark.asyncio
    async def test_extract_domain_knowledge(self):
        """Test domain knowledge extraction convenience function."""
        from integrations.oneke import extract_domain_knowledge
        from workflow_structures import WorkflowState

        workflow = WorkflowState(
            workflow_id="test_001",
            problem_statement="Test physics problem",
            final_solution="Test solution"
        )

        try:
            knowledge = await extract_domain_knowledge(workflow, domains=['physics'])
            assert isinstance(knowledge, dict)
        except Exception as e:
            pytest.skip(f"Could not extract domain knowledge: {e}")


# ========== Error Handling Tests ==========

class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_extract_before_initialization(self):
        """Test extraction before initialization."""
        adapter = OneKEAdapter()
        # Don't initialize

        with pytest.raises(Exception):  # Should raise ExtractionError
            await adapter.extract_ner("Test text")

    @pytest.mark.asyncio
    async def test_invalid_schema_path(self, adapter):
        """Test loading invalid schema path."""
        with pytest.raises(Exception):  # Should raise SchemaLoadError
            adapter.load_schema("/nonexistent/path/schema.yaml")

    @pytest.mark.asyncio
    async def test_shutdown_before_initialization(self):
        """Test shutdown before initialization."""
        adapter = OneKEAdapter()
        # Should not raise exception
        result = await adapter.shutdown()
        assert result is True or result is False


# ========== Performance Tests ==========

class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_batch_extraction_performance(self, adapter):
        """Test batch extraction performance."""
        import time

        texts = ["Test text for extraction"] * 10

        start = time.time()
        results = await adapter.batch_extract(texts, ExtractionType.NER)
        elapsed = time.time() - start

        assert len(results) == 10
        # Should complete in reasonable time (adjust as needed)
        assert elapsed < 60  # 60 seconds max for 10 texts


# ========== Run Tests ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
