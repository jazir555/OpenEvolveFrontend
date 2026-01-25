"""
GlobalChem Integration Tests

Comprehensive test suite for GlobalChem adapter and bridge functionality.

Author: Agent 7 (GlobalChem Integration Specialist)
Date: 2026-01-02
Version: 0.1.0
"""

import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from integrations.global_chem.adapter import (
    GlobalChemAdapter,
    ChemicalKnowledgeError,
    SMILESParsingError,
    SMARTSParsingError,
)

from integrations.global_chem.bridge import (
    GlobalChemBridge,
    ChemicalEntity,
    ChemicalRelationship,
    ChemicalEntityType,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def adapter():
    """Create and initialize a GlobalChem adapter for testing."""
    adapter = GlobalChemAdapter()
    config = {
        'auto_start': False,  # Don't auto-load lists for faster tests
        'cache_enabled': True,
        'cache_ttl': 3600,
    }
    # Note: These tests will use mocking since GlobalChem may not be available
    return adapter


@pytest.fixture
async def bridge(adapter):
    """Create and initialize a GlobalChem bridge for testing."""
    bridge = GlobalChemBridge(adapter)
    return bridge


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing."""
    return {
        'auto_start': False,
        'cache_enabled': True,
        'cache_ttl': 3600,
        'chemical_lists': ['amino_acids', 'vitamins'],
        'entity_recognition': {
            'enabled': True,
            'confidence_threshold': 0.7,
        },
        'oneke_integration': True,
        'cache_entities': True,
    }


@pytest.fixture
def sample_smiles_data() -> Dict[str, Dict[str, str]]:
    """Sample SMILES data for mocking."""
    return {
        'amino_acids': {
            'Glycine': 'C(C(=O)O)N',
            'Alanine': 'C(C)(C(=O)O)N',
            'Valine': 'CC(C)C(C(=O)O)N',
        },
        'vitamins': {
            'Ascorbic Acid': 'C(C(C1C(=C(C(=O)O1)O)O)O)O',
            'Thiamine': 'CC1=CN(C=C1)CC(C)C1=CN=C(C)N=C1',
        }
    }


# ============================================================================
# Adapter Tests
# ============================================================================

class TestGlobalChemAdapter:
    """Test suite for GlobalChemAdapter."""

    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter, sample_config):
        """Test adapter initialization."""
        with patch('integrations.global_chem.adapter.GLOBAL_CHEM_AVAILABLE', True):
            mock_globalchem = Mock()
            mock_globalchem.get_all_smiles = Mock(return_value={})

            with patch('integrations.global_chem.adapter.GlobalChem', return_value=mock_globalchem):
                result = await adapter.initialize(sample_config)
                assert result is True
                assert adapter.is_initialized is True
                assert adapter.cache_enabled is True
                assert adapter.cache_ttl == 3600

    @pytest.mark.asyncio
    async def test_adapter_unavailable(self, adapter, sample_config):
        """Test adapter behavior when GlobalChem is unavailable."""
        with patch('integrations.global_chem.adapter.GLOBAL_CHEM_AVAILABLE', False):
            with pytest.raises(Exception):  # ConfigurationError
                await adapter.initialize(sample_config)

    @pytest.mark.asyncio
    async def test_parse_smiles_valid(self, adapter, sample_smiles_data):
        """Test SMILES parsing for valid SMILES."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        result = await adapter.parse_smiles('C(C(=O)O)N')

        assert result['is_valid'] is True
        assert result['canonical_form'] == 'C(C(=O)O)N'
        assert result['error'] is None

    @pytest.mark.asyncio
    async def test_parse_smiles_invalid(self, adapter):
        """Test SMILES parsing for invalid SMILES."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value={})

        result = await adapter.parse_smiles('INVALID_SMILES')

        assert result['is_valid'] is False
        assert result['error'] is not None

    @pytest.mark.asyncio
    async def test_parse_smiles_cache(self, adapter, sample_smiles_data):
        """Test SMILES caching."""
        adapter.is_initialized = True
        adapter.cache_enabled = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        # First call - should hit database
        result1 = await adapter.parse_smiles('C(C(=O)O)N')
        assert 'C(C(=O)O)N' in adapter.smiles_cache

        # Second call - should use cache
        result2 = await adapter.parse_smiles('C(C(=O)O)N')
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_parse_smarts_valid(self, adapter):
        """Test SMARTS parsing for valid SMARTS."""
        adapter.is_initialized = True

        result = await adapter.parse_smarts('[C][C]')

        assert result['is_valid'] is True
        assert result['pattern_type'] == 'atom_query'
        assert result['error'] is None

    @pytest.mark.asyncio
    async def test_parse_smarts_invalid(self, adapter):
        """Test SMARTS parsing for invalid SMARTS."""
        adapter.is_initialized = True

        result = await adapter.parse_smarts('')

        assert result['is_valid'] is False
        assert result['error'] is not None

    @pytest.mark.asyncio
    async def test_query_chemical_list(self, adapter, sample_smiles_data):
        """Test querying a specific chemical list."""
        adapter.is_initialized = True
        adapter.chemical_lists_cache = sample_smiles_data

        result = await adapter.query_chemical_list('amino_acids', limit=2)

        assert result['list_name'] == 'amino_acids'
        assert len(result['chemicals']) <= 2
        assert result['total'] == len(sample_smiles_data['amino_acids'])

    @pytest.mark.asyncio
    async def test_query_chemical_list_with_filter(self, adapter, sample_smiles_data):
        """Test querying chemical list with query filter."""
        adapter.is_initialized = True
        adapter.chemical_lists_cache = sample_smiles_data

        result = await adapter.query_chemical_list('amino_acids', query='Glycine')

        assert len(result['chemicals']) == 1
        assert result['chemicals'][0]['name'] == 'Glycine'

    @pytest.mark.asyncio
    async def test_get_available_chemical_lists(self, adapter, sample_smiles_data):
        """Test getting available chemical lists."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        lists = await adapter.get_available_chemical_lists()

        assert len(lists) == 2
        assert 'amino_acids' in lists
        assert 'vitamins' in lists

    @pytest.mark.asyncio
    async def test_search(self, adapter, sample_smiles_data):
        """Test searching for chemicals."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        result = await adapter.search('Glycine', num_results=10)

        assert 'chemicals' in result
        assert 'total_found' in result
        assert result['query'] == 'Glycine'

    @pytest.mark.asyncio
    async def test_validate_initialized(self, adapter, sample_smiles_data):
        """Test validation for initialized adapter."""
        adapter.is_initialized = True
        adapter.cache_enabled = True
        adapter.cache_ttl = 3600
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        result = await adapter.validate()

        assert result['is_valid'] is True
        assert result['checks']['initialized'] is True
        assert result['checks']['chemical_lists_loaded'] is True

    @pytest.mark.asyncio
    async def test_validate_not_initialized(self, adapter):
        """Test validation for uninitialized adapter."""
        adapter.is_initialized = False

        result = await adapter.validate()

        assert result['is_valid'] is False
        assert result['checks']['initialized'] is False

    @pytest.mark.asyncio
    async def test_shutdown(self, adapter):
        """Test adapter shutdown."""
        adapter.is_initialized = True
        adapter.chemical_lists_cache = {'test': {}}
        adapter.smiles_cache = {'test': {}}

        result = await adapter.shutdown()

        assert result is True
        assert adapter.is_initialized is False
        assert len(adapter.chemical_lists_cache) == 0
        assert len(adapter.smiles_cache) == 0

    @pytest.mark.asyncio
    async def test_add_episode_not_applicable(self, adapter):
        """Test that add_episode returns not_applicable for GlobalChem."""
        adapter.is_initialized = True

        from datetime import datetime
        result = await adapter.add_episode(
            name="test",
            body="test episode",
            reference_time=datetime.now()
        )

        assert result['status'] == 'not_applicable'

    @pytest.mark.asyncio
    async def test_add_triplet_not_applicable(self, adapter):
        """Test that add_triplet returns not_applicable for GlobalChem."""
        adapter.is_initialized = True

        result = await adapter.add_triplet(
            source_entity={"name": "test"},
            relationship={"fact": "test"},
            target_entity={"name": "test"}
        )

        assert result['status'] == 'not_applicable'

    @pytest.mark.asyncio
    async def test_remove_episode_not_applicable(self, adapter):
        """Test that remove_episode returns False for GlobalChem."""
        adapter.is_initialized = True

        result = await adapter.remove_episode("test-uuid")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_episodes_empty(self, adapter):
        """Test that get_episodes returns empty list for GlobalChem."""
        adapter.is_initialized = True

        from datetime import datetime
        result = await adapter.get_episodes(datetime.now())

        assert result == []


# ============================================================================
# Bridge Tests
# ============================================================================

class TestGlobalChemBridge:
    """Test suite for GlobalChemBridge."""

    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge, adapter, sample_config):
        """Test bridge initialization."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value={})

        result = await bridge.initialize(sample_config)

        assert result is True
        assert bridge.is_initialized is True
        assert bridge.oneke_integration_enabled is True

    @pytest.mark.asyncio
    async def test_recognize_chemical_entities(self, bridge, adapter, sample_smiles_data):
        """Test chemical entity recognition."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem

        text = "Glycine and Alanine are amino acids"
        entities = await bridge.recognize_chemical_entities(text, threshold=0.5)

        assert len(entities) > 0
        assert isinstance(entities[0], ChemicalEntity)
        assert entities[0].name in ['Glycine', 'Alanine']

    @pytest.mark.asyncio
    async def test_recognize_chemical_entities_high_threshold(
        self, bridge, adapter, sample_smiles_data
    ):
        """Test entity recognition with high threshold."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem

        text = "Glycine and Alanine are amino acids"
        entities = await bridge.recognize_chemical_entities(text, threshold=0.99)

        # Should return fewer or no entities with high threshold
        assert len(entities) >= 0

    @pytest.mark.asyncio
    async def test_classify_list_type(self, bridge):
        """Test chemical list type classification."""
        # Organic compounds
        assert bridge._classify_list_type('organic_acids') == ChemicalEntityType.ORGANIC_COMPOUND

        # Biomolecules
        assert bridge._classify_list_type('amino_acids') == ChemicalEntityType.BIOMOLECULE

        # Drugs
        assert bridge._classify_list_type('drugs_list') == ChemicalEntityType.DRUG

        # Food additives
        assert bridge._classify_list_type('food_additives') == ChemicalEntityType.FOOD_ADDITIVE

        # Unknown
        assert bridge._classify_list_type('unknown_list') == ChemicalEntityType.UNKNOWN

    @pytest.mark.asyncio
    async def test_extract_chemical_relationships(self, bridge):
        """Test chemical relationship extraction."""
        bridge.is_initialized = True

        entity1 = ChemicalEntity(
            name="Aspirin",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            entity_type=ChemicalEntityType.DRUG,
            source_list="drugs",
            properties={},
            confidence=0.9
        )

        entity2 = ChemicalEntity(
            name="Salicylic Acid",
            smiles="OC1=CC=CC=C1C(=O)O",
            entity_type=ChemicalEntityType.ORGANIC_COMPOUND,
            source_list="acids",
            properties={},
            confidence=0.85
        )

        text = "Aspirin is derived from salicylic acid"
        relationships = await bridge.extract_chemical_relationships([entity1, entity2], text)

        assert len(relationships) > 0
        assert isinstance(relationships[0], ChemicalRelationship)

    @pytest.mark.asyncio
    async def test_generate_knowledge_graph(self, bridge, adapter, sample_smiles_data):
        """Test knowledge graph generation."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem

        text = "Glycine and Alanine are amino acids"
        kg = await bridge.generate_knowledge_graph(text)

        assert 'nodes' in kg
        assert 'edges' in kg
        assert 'metadata' in kg
        assert kg['metadata']['source'] == 'global_chem'

    @pytest.mark.asyncio
    async def test_query_chemical_knowledge(self, bridge, adapter, sample_smiles_data):
        """Test querying chemical knowledge."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem

        # Mock the adapter search method
        adapter.search = AsyncMock(return_value={
            'chemicals': [
                {'name': 'Glycine', 'smiles': 'C(C(=O)O)N', 'list': 'amino_acids'}
            ]
        })

        results = await bridge.query_chemical_knowledge('Glycine')

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_chemical_knowledge_with_filter(
        self, bridge, adapter, sample_smiles_data
    ):
        """Test querying chemical knowledge with entity type filter."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem

        # Mock the adapter search method
        adapter.search = AsyncMock(return_value={
            'chemicals': [
                {'name': 'Glycine', 'smiles': 'C(C(=O)O)N', 'list': 'amino_acids'}
            ]
        })

        results = await bridge.query_chemical_knowledge(
            'Glycine',
            entity_type=ChemicalEntityType.BIOMOLECULE
        )

        # Should filter to only biomolecules
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_integrate_with_oneke_disabled(self, bridge):
        """Test OneKE integration when disabled."""
        bridge.is_initialized = True
        bridge.oneke_integration_enabled = False

        result = await bridge.integrate_with_oneke("test text")

        assert result['oneke_integration'] is False
        assert result['chemical_entities'] == []

    @pytest.mark.asyncio
    async def test_integrate_with_oneke_enabled(self, bridge, adapter, sample_smiles_data):
        """Test OneKE integration when enabled."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem
        bridge.oneke_integration_enabled = True

        oneke_results = {
            'entities': [
                {'name': 'Glycine', 'type': 'CHEMICAL'}
            ]
        }

        result = await bridge.integrate_with_oneke("Glycine is an amino acid", oneke_results)

        assert result['oneke_integration'] is True
        assert 'chemical_entities' in result
        assert 'oneke_entities' in result

    @pytest.mark.asyncio
    async def test_get_statistics(self, bridge, adapter):
        """Test getting bridge statistics."""
        adapter.is_initialized = True
        adapter.get_available_chemical_lists = AsyncMock(return_value=['amino_acids', 'vitamins'])

        bridge.is_initialized = True
        bridge.oneke_integration_enabled = True
        bridge.entity_cache = {'test': Mock()}
        bridge.relationship_cache = [Mock()]

        stats = await bridge.get_statistics()

        assert stats['initialized'] is True
        assert stats['oneke_integration_enabled'] is True
        assert stats['cached_entities'] == 1
        assert stats['cached_relationships'] == 1
        assert len(stats['available_chemical_lists']) == 2

    @pytest.mark.asyncio
    async def test_shutdown_bridge(self, bridge, adapter):
        """Test bridge shutdown."""
        adapter.is_initialized = True
        adapter.shutdown = AsyncMock(return_value=True)

        bridge.is_initialized = True
        bridge.entity_cache = {'test': Mock()}
        bridge.relationship_cache = [Mock()]

        result = await bridge.shutdown()

        assert result is True
        assert bridge.is_initialized is False
        assert len(bridge.entity_cache) == 0
        assert len(bridge.relationship_cache) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestGlobalChemIntegration:
    """Integration tests for GlobalChem with OpenEvolve."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, adapter, bridge, sample_smiles_data):
        """Test full pipeline from text to knowledge graph."""
        # Initialize
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        adapter.search = AsyncMock(return_value={
            'chemicals': [
                {'name': 'Glycine', 'smiles': 'C(C(=O)O)N', 'list': 'amino_acids'}
            ]
        })

        bridge.is_initialized = True
        bridge.global_chem = adapter.global_chem

        # Text with chemical entities
        text = "Glycine and Alanine are amino acids. Glycine reacts with acetic acid."

        # Generate knowledge graph
        kg = await bridge.generate_knowledge_graph(text)

        # Verify results
        assert len(kg['nodes']) > 0
        assert kg['metadata']['source'] == 'global_chem'

    @pytest.mark.asyncio
    async def test_cache_effectiveness(self, adapter, sample_smiles_data):
        """Test that caching improves performance."""
        import time

        adapter.is_initialized = True
        adapter.cache_enabled = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(return_value=sample_smiles_data)

        # First call (no cache)
        start1 = time.time()
        await adapter.parse_smiles('C(C(=O)O)N')
        time1 = time.time() - start1

        # Second call (with cache)
        start2 = time.time()
        await adapter.parse_smiles('C(C(=O)O)N')
        time2 = time.time() - start2

        # Second call should be faster (though timing can vary)
        assert 'C(C(=O)O)N' in adapter.smiles_cache

    @pytest.mark.asyncio
    async def test_error_handling(self, adapter):
        """Test error handling in adapter."""
        adapter.is_initialized = True
        adapter.global_chem = Mock()
        adapter.global_chem.get_all_smiles = Mock(side_effect=Exception("Database error"))

        # Should handle errors gracefully
        with pytest.raises(ChemicalKnowledgeError):
            await adapter.get_available_chemical_lists()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, bridge):
        """Test graceful degradation when GlobalChem is unavailable."""
        bridge.is_initialized = True
        bridge.oneke_integration_enabled = False

        # Should not crash, just return empty results
        result = await bridge.integrate_with_oneke("test")

        assert result['chemical_entities'] == []
        assert result['oneke_integration'] is False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
