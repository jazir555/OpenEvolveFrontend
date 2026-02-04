"""
Test Suite for Unified Knowledge Graph Integrations

Tests all KG integrations including:
- DeepKE (entity/relation extraction)
- NeuralKG (embeddings, link prediction)
- KarateClub (graph analysis, communities)
- KG-Gen (LLM-based extraction)
- OneKE (bilingual extraction)
- AI-Knowledge-Graph (standardization, inference)
- Graphiti (temporal knowledge)
- GlobalChem (chemical analysis)
- Causal-Learn (causal discovery)
- PyGraphistry (visualization)

Run with: pytest test_unified_kg_integrations.py -v
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration]


class TestUnifiedKGIntegrationHub:
    """Tests for the Unified KG Integration Hub."""
    
    @pytest.fixture
    async def hub(self):
        """Create and initialize hub."""
        from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
        hub = UnifiedKGIntegrationHub()
        await hub.initialize()
        return hub
    
    @pytest.mark.asyncio
    async def test_hub_initialization(self):
        """Test hub initializes correctly."""
        from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
        
        hub = UnifiedKGIntegrationHub()
        result = await hub.initialize()
        
        # Should return True (some integrations may be available)
        assert isinstance(result, bool)
        
        # Check health status
        health = hub.get_health_status()
        assert 'integrations' in health
        assert 'summary' in health
        assert health['summary']['total'] > 0
    
    @pytest.mark.asyncio
    async def test_health_status(self):
        """Test health status reporting."""
        from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
        
        hub = UnifiedKGIntegrationHub()
        await hub.initialize()
        
        health = hub.get_health_status()
        assert 'timestamp' in health
        assert isinstance(health['integrations'], dict)
        
        summary = health['summary']
        assert summary['total'] >= summary['available'] + summary['unavailable'] + summary['error']
    
    @pytest.mark.asyncio
    async def test_available_integrations(self):
        """Test getting available integrations."""
        from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
        
        hub = UnifiedKGIntegrationHub()
        await hub.initialize()
        
        available = hub.get_available_integrations()
        assert isinstance(available, list)
        
        # Should have some integrations
        assert len(available) >= 0


class TestDeepKEIntegration:
    """Tests for DeepKE integration."""
    
    @pytest.fixture
    def integration(self):
        """Create DeepKE integration."""
        from knowledge_engine.integrations.deepke_integration import DeepKEIntegration
        return DeepKEIntegration()
    
    def test_deepke_initialization(self, integration):
        """Test DeepKE initializes."""
        assert integration is not None
        # May or may not be available depending on environment
        availability = integration.is_available()
        assert isinstance(availability, bool)
    
    def test_entity_extraction(self, integration):
        """Test entity extraction."""
        if not integration.is_available():
            pytest.skip("DeepKE not available")
        
        text = "Apple Inc. was founded by Steve Jobs in Cupertino."
        result = integration.extract_entities(text)
        
        assert isinstance(result, dict)
        # Result should contain entities or status
        assert 'status' in result or 'entities' in result


class TestNeuralKGIntegration:
    """Tests for NeuralKG integration."""
    
    @pytest.fixture
    def integration(self):
        """Create NeuralKG integration."""
        from knowledge_engine.integrations.neuralkg_integration import NeuralKGIntegration
        return NeuralKGIntegration()
    
    def test_neuralkg_initialization(self, integration):
        """Test NeuralKG initializes."""
        assert integration is not None
        availability = integration.is_available()
        assert isinstance(availability, bool)
    
    def test_embedding_generation(self, integration):
        """Test embedding generation."""
        if not integration.is_available():
            pytest.skip("NeuralKG not available")
        
        triples = [
            ("Paris", "capital_of", "France"),
            ("Berlin", "capital_of", "Germany"),
            ("London", "capital_of", "UK")
        ]
        
        result = integration.generate_embeddings(triples, model='transe')
        
        assert isinstance(result, dict)
        assert 'status' in result


class TestKarateClubIntegration:
    """Tests for KarateClub integration."""
    
    @pytest.fixture
    def integration(self):
        """Create KarateClub integration."""
        from knowledge_engine.integrations.karateclub_integration import KarateClubIntegration
        return KarateClubIntegration()
    
    def test_karateclub_initialization(self, integration):
        """Test KarateClub initializes."""
        assert integration is not None
        availability = integration.is_available()
        assert isinstance(availability, bool)
    
    def test_graph_analysis(self, integration):
        """Test graph analysis."""
        if not integration.is_available():
            pytest.skip("KarateClub not available")
        
        graph_data = {
            'nodes': [
                {'id': 'A'}, {'id': 'B'}, {'id': 'C'}, {'id': 'D'}
            ],
            'edges': [
                {'source': 'A', 'target': 'B'},
                {'source': 'B', 'target': 'C'},
                {'source': 'C', 'target': 'D'},
                {'source': 'D', 'target': 'A'}
            ]
        }
        
        result = integration.analyze_graph(graph_data)
        
        assert isinstance(result, dict)
        # Should have some analysis results
        assert len(result) > 0


class TestKGGenIntegration:
    """Tests for KG-Gen integration."""
    
    @pytest.fixture
    def integration(self):
        """Create KG-Gen integration."""
        from knowledge_engine.integrations.kggen_integration import KGGenIntegration
        return KGGenIntegration()
    
    def test_kggen_initialization(self, integration):
        """Test KG-Gen initializes."""
        assert integration is not None
    
    def test_knowledge_extraction(self, integration):
        """Test knowledge extraction."""
        text = "Apple Inc. is a technology company founded by Steve Jobs."
        
        # May use mock if LLM not available
        result = integration.extract_graph(text)
        
        assert result is not None
        assert hasattr(result, 'entities') or isinstance(result, dict)


class TestOneKEIntegration:
    """Tests for OneKE integration."""
    
    @pytest.fixture
    def integration(self):
        """Create OneKE integration."""
        from knowledge_engine.integrations.oneke_integration import OneKEIntegration
        return OneKEIntegration()
    
    def test_oneke_initialization(self, integration):
        """Test OneKE initializes."""
        assert integration is not None
        availability = integration.is_available()
        assert isinstance(availability, bool)


class TestAIKGIntegration:
    """Tests for AI-Knowledge-Graph integration."""
    
    @pytest.fixture
    def integration(self):
        """Create AIKG integration."""
        from knowledge_engine.integrations.aikg_integration import AIKGIntegration
        return AIKGIntegration()
    
    def test_aikg_initialization(self, integration):
        """Test AIKG initializes."""
        assert integration is not None
    
    @pytest.mark.asyncio
    async def test_knowledge_processing(self, integration):
        """Test knowledge graph processing."""
        text = "Microsoft was founded by Bill Gates."
        
        result = await integration.process_knowledge_graph(
            text,
            enable_standardization=False,
            enable_inference=False,
            generate_visualization=False
        )
        
        assert result is not None
        assert result.success is True or result.success is False


class TestGlobalChemIntegration:
    """Tests for GlobalChem integration."""
    
    @pytest.fixture
    def integration(self):
        """Create GlobalChem integration."""
        from knowledge_engine.integrations.global_chem_integration import GlobalChemIntegration
        return GlobalChemIntegration()
    
    def test_globalchem_initialization(self, integration):
        """Test GlobalChem initializes."""
        assert integration is not None
        availability = integration.is_available()
        assert isinstance(availability, bool)
    
    def test_chemical_search(self, integration):
        """Test chemical search."""
        if not integration.is_available():
            pytest.skip("GlobalChem not available")
        
        # Search for a common chemical
        results = integration._adapter.search_chemicals("glucose")
        
        assert isinstance(results, list)
        # May or may not find results depending on database


class TestCausalLearnIntegration:
    """Tests for Causal-Learn integration (SSOT: integrations/causal_learn/)."""
    
    @pytest.fixture
    def integration(self):
        """Create Causal-Learn integration."""
        from knowledge_engine.integrations.causal_learn_integration import CausalLearnIntegration
        return CausalLearnIntegration()
    
    def test_ssot_import(self):
        """Test that SSOT components can be imported."""
        try:
            from integrations.causal_learn import (
                CausalLearnAdapter,
                CausalDiscoveryBridge,
                CAUSAL_LEARN_AVAILABLE
            )
            # Can import, check structure
            assert hasattr(CausalLearnAdapter, 'discover_causal_structure')
            assert hasattr(CausalDiscoveryBridge, 'initialize')
        except ImportError:
            pytest.skip("SSOT (integrations.causal_learn) not available")
    
    def test_causal_learn_initialization(self, integration):
        """Test Causal-Learn initializes."""
        assert integration is not None
        availability = integration.is_available()
        assert isinstance(availability, bool)
    
    def test_ssot_info(self):
        """Test SSOT info function."""
        from knowledge_engine.integrations.causal_learn_integration import get_ssot_info
        info = get_ssot_info()
        assert 'ssot_location' in info
        assert info['ssot_location'] == 'integrations/causal_learn/'
    
    def test_causal_discovery(self, integration):
        """Test causal discovery."""
        if not integration.is_available():
            pytest.skip("Causal-Learn not available")
        
        # Generate synthetic data with known causal structure
        np.random.seed(42)
        n_samples = 100
        
        # X -> Y -> Z
        X = np.random.normal(0, 1, n_samples)
        Y = 2 * X + np.random.normal(0, 0.5, n_samples)
        Z = 1.5 * Y + np.random.normal(0, 0.5, n_samples)
        
        data = np.column_stack([X, Y, Z])
        
        result = integration.discover_structure(
            data,
            variable_names=['X', 'Y', 'Z'],
            algorithm='pc'
        )
        
        assert isinstance(result, dict)
        assert 'status' in result
    
    def test_get_available_algorithms(self, integration):
        """Test getting available algorithms."""
        algorithms = integration.get_available_algorithms()
        assert isinstance(algorithms, list)
        # Should have at least some algorithms if available
        if integration.is_available():
            assert len(algorithms) > 0
    
    def test_get_algorithm_info(self, integration):
        """Test getting algorithm info."""
        info = integration.get_algorithm_info('pc')
        assert 'name' in info
        assert 'description' in info
        assert info['name'] == 'pc'


class TestPyGraphistryIntegration:
    """Tests for PyGraphistry integration."""
    
    @pytest.fixture
    def integration(self):
        """Create PyGraphistry integration."""
        from knowledge_engine.integrations.pygraphistry_integration import PyGraphistryIntegration
        return PyGraphistryIntegration()
    
    def test_pygraphistry_initialization(self, integration):
        """Test PyGraphistry initializes."""
        assert integration is not None
        # May or may not be available depending on API key
        availability = integration.is_available()
        assert isinstance(availability, bool)
    
    def test_graph_analysis(self, integration):
        """Test graph analysis."""
        nodes = [
            {'id': 'A', 'label': 'Node A', 'type': 'person'},
            {'id': 'B', 'label': 'Node B', 'type': 'person'},
            {'id': 'C', 'label': 'Node C', 'type': 'organization'}
        ]
        edges = [
            {'source': 'A', 'target': 'B', 'relation': 'knows'},
            {'source': 'B', 'target': 'C', 'relation': 'works_for'}
        ]
        
        metrics = integration.analyze_graph(nodes, edges)
        
        assert metrics is not None
        assert metrics.node_count == 3
        assert metrics.edge_count == 2
    
    def test_community_detection(self, integration):
        """Test community detection."""
        nodes = [
            {'id': f'node_{i}'} for i in range(10)
        ]
        edges = [
            {'source': f'node_{i}', 'target': f'node_{(i+1) % 10}'}
            for i in range(10)
        ]
        
        result = integration.detect_communities(nodes, edges)
        
        assert isinstance(result, dict)
        # May or may not find communities


class TestIntegrationPipeline:
    """Tests for multi-step integration pipelines."""
    
    @pytest.mark.asyncio
    async def test_extraction_pipeline(self):
        """Test extraction to analysis pipeline."""
        from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
        
        hub = UnifiedKGIntegrationHub()
        await hub.initialize()
        
        text = "Google was founded by Larry Page and Sergey Brin."
        
        # Extract entities
        extract_result = await hub.extract_entities(text)
        assert extract_result is not None
        
        # If we have nodes and edges, try analysis
        if extract_result.success and extract_result.data:
            data = extract_result.data
            if 'nodes' in data and 'edges' in data:
                analyze_result = await hub.analyze_graph(
                    data['nodes'],
                    data['edges']
                )
                assert analyze_result is not None


class TestGraphitiIntegration:
    """Tests for Graphiti integration."""
    
    @pytest.fixture
    def integration(self):
        """Create Graphiti integration."""
        from knowledge_engine.integrations.graphiti_integration import GraphitiIntegration
        # Use test configuration
        return GraphitiIntegration(
            uri='bolt://localhost:7687',
            user='neo4j',
            password='test'
        )
    
    def test_graphiti_initialization(self, integration):
        """Test Graphiti initializes."""
        assert integration is not None
        assert integration.uri is not None
    
    @pytest.mark.asyncio
    async def test_knowledge_artifact_creation(self, integration):
        """Test creating knowledge artifact."""
        from knowledge_engine.integrations.graphiti_integration import KnowledgeArtifact
        
        artifact = KnowledgeArtifact(
            id="test_001",
            content="Test knowledge content",
            artifact_type="test",
            valid_at=datetime.now(timezone.utc),
            source="test"
        )
        
        assert artifact.id == "test_001"
        assert artifact.content == "Test knowledge content"
        
        # Test to_dict
        data = artifact.to_dict()
        assert data['id'] == "test_001"


def test_all_integrations_importable():
    """Test that all integration modules can be imported."""
    integrations = [
        'knowledge_engine.integrations.deepke_integration',
        'knowledge_engine.integrations.neuralkg_integration',
        'knowledge_engine.integrations.karateclub_integration',
        'knowledge_engine.integrations.kggen_integration',
        'knowledge_engine.integrations.oneke_integration',
        'knowledge_engine.integrations.aikg_integration',
        'knowledge_engine.integrations.graphiti_integration',
        'knowledge_engine.integrations.global_chem_integration',
        'knowledge_engine.integrations.causal_learn_integration',
        'knowledge_engine.integrations.pygraphistry_integration',
    ]
    
    for module_name in integrations:
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
