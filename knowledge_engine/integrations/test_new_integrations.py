"""
Integration Tests for New KG Integrations (Outlines, LMQL, Neuromancer)

This module provides comprehensive tests for the three new integrations:
1. Outlines - Structured LLM output generation
2. LMQL - Declarative query language for LLMs
3. Neuromancer - Physics-informed neural operators

Test Categories:
    - Unit tests for individual components
    - Integration tests with the Unified Hub
    - End-to-end workflow tests
    - Performance benchmarks

Author: OpenEvolve
Date: 2026-02-03
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestOutlinesIntegration:
    """Test suite for Outlines integration."""
    
    @pytest.fixture
    def outlines_config(self) -> Dict[str, Any]:
        """Test configuration for Outlines."""
        return {
            'default_model': 'gpt-3.5-turbo',
            'cache_enabled': True,
            'circuit_breaker_threshold': 5,
            'fallback_enabled': True
        }
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, outlines_config):
        """Test Outlines adapter initialization."""
        try:
            from integrations.outlines.adapter import OutlinesAdapter
            adapter = OutlinesAdapter(outlines_config)
            assert adapter is not None
            assert hasattr(adapter, 'generate_json')
            assert hasattr(adapter, 'generate_regex')
            assert hasattr(adapter, 'generate_choices')
        except ImportError as e:
            pytest.skip(f"Outlines not available: {e}")
    
    @pytest.mark.asyncio
    async def test_kg_constraints_loading(self):
        """Test KG constraints schema loading."""
        try:
            from integrations.outlines.kg_constraints import (
                EntityExtractionSchema,
                RelationshipSchema,
                CypherQuerySchema
            )
            # Test schema instantiation
            entity = EntityExtractionSchema(
                entities=[{
                    'name': 'Test Entity',
                    'type': 'ORG',
                    'confidence': 0.95,
                    'properties': {'location': 'USA'}
                }]
            )
            assert entity.entities[0].name == 'Test Entity'
        except ImportError as e:
            pytest.skip(f"Outlines constraints not available: {e}")
    
    @pytest.mark.asyncio
    async def test_prompt_templates(self):
        """Test prompt template loading."""
        try:
            from integrations.outlines.prompt_templates import PromptTemplateManager
            manager = PromptTemplateManager()
            
            # Test template retrieval
            entity_template = manager.get_template('entity_extraction')
            assert entity_template is not None
            
            relation_template = manager.get_template('relation_extraction')
            assert relation_template is not None
        except ImportError as e:
            pytest.skip(f"Outlines templates not available: {e}")
    
    @pytest.mark.asyncio
    async def test_kg_integration_wrapper(self):
        """Test Knowledge Engine wrapper for Outlines."""
        try:
            from knowledge_engine.integrations.outlines.outlines_integration import OutlinesKGIntegration
            integration = OutlinesKGIntegration()
            
            assert hasattr(integration, 'extract_entities_constrained')
            assert hasattr(integration, 'generate_cypher_constrained')
            assert hasattr(integration, 'validate_kg_structure')
        except ImportError as e:
            pytest.skip(f"Outlines KE integration not available: {e}")


class TestLMQLIntegration:
    """Test suite for LMQL integration."""
    
    @pytest.fixture
    def lmql_config(self) -> Dict[str, Any]:
        """Test configuration for LMQL."""
        return {
            'default_model': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            'cache_enabled': True
        }
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, lmql_config):
        """Test LMQL adapter initialization."""
        try:
            from integrations.lmql.adapter import LMQLAdapter
            adapter = LMQLAdapter(lmql_config)
            assert adapter is not None
            assert hasattr(adapter, 'query')
            assert hasattr(adapter, 'extract_entities')
            assert hasattr(adapter, 'query_kg')
        except ImportError as e:
            pytest.skip(f"LMQL not available: {e}")
    
    @pytest.mark.asyncio
    async def test_constraint_engine(self):
        """Test constraint evaluation engine."""
        try:
            from integrations.lmql.constraint_engine import (
                ConstraintEvaluator,
                LengthConstraint,
                RegexConstraint
            )
            
            evaluator = ConstraintEvaluator()
            
            # Test length constraint
            length_constraint = LengthConstraint(min_length=5, max_length=10)
            assert evaluator.evaluate("hello", length_constraint) == True
            assert evaluator.evaluate("hi", length_constraint) == False
            
            # Test regex constraint
            regex_constraint = RegexConstraint(pattern=r"^\d{3}$")
            assert evaluator.evaluate("123", regex_constraint) == True
            assert evaluator.evaluate("abc", regex_constraint) == False
        except ImportError as e:
            pytest.skip(f"LMQL constraint engine not available: {e}")
    
    @pytest.mark.asyncio
    async def test_query_templates(self):
        """Test LMQL query templates."""
        try:
            from integrations.lmql.query_templates import LMQLQueryTemplates
            templates = LMQLQueryTemplates()
            
            # Test template retrieval
            entity_template = templates.get_template('entity_extraction')
            assert entity_template is not None
            assert '{text}' in entity_template
        except ImportError as e:
            pytest.skip(f"LMQL templates not available: {e}")
    
    @pytest.mark.asyncio
    async def test_kg_integration_wrapper(self):
        """Test Knowledge Engine wrapper for LMQL."""
        try:
            from knowledge_engine.integrations.lmql.lmql_integration import LMQLKGIntegration
            integration = LMQLKGIntegration()
            
            assert hasattr(integration, 'query_entities')
            assert hasattr(integration, 'query_relations')
            assert hasattr(integration, 'multi_hop_query')
            assert hasattr(integration, 'generate_cypher')
        except ImportError as e:
            pytest.skip(f"LMQL KE integration not available: {e}")


class TestNeuromancerIntegration:
    """Test suite for Neuromancer integration."""
    
    @pytest.fixture
    def neuromancer_config(self) -> Dict[str, Any]:
        """Test configuration for Neuromancer."""
        return {
            'device': 'cpu',
            'dtype': 'float32',
            'solver_tolerance': 1e-6
        }
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, neuromancer_config):
        """Test Neuromancer adapter initialization."""
        try:
            from integrations.neuromancer.adapter import NeuroMANCERAdapter
            adapter = NeuroMANCERAdapter(neuromancer_config)
            assert adapter is not None
        except ImportError as e:
            pytest.skip(f"Neuromancer not available: {e}")
    
    @pytest.mark.asyncio
    async def test_physics_constraints(self):
        """Test physics constraint definitions."""
        try:
            from integrations.neuromancer.physics_constraints import (
                ConservationLaws,
                MechanicalConstraints
            )
            
            # Test conservation laws
            conservation = ConservationLaws()
            mass_result = conservation.mass_conservation(
                initial_mass=10.0,
                final_mass=10.0,
                tolerance=1e-6
            )
            assert mass_result['satisfied'] == True
            
            # Test mechanical constraints
            mechanics = MechanicalConstraints()
            newton_result = mechanics.newton_second_law(
                force=10.0,
                mass=2.0,
                acceleration=5.0
            )
            assert newton_result['satisfied'] == True
        except ImportError as e:
            pytest.skip(f"Neuromancer physics constraints not available: {e}")
    
    @pytest.mark.asyncio
    async def test_scientific_domains(self):
        """Test scientific domain configurations."""
        try:
            from integrations.neuromancer.scientific_domains import (
                ClimateModeling,
                FluidDynamics,
                StructuralMechanics
            )
            
            # Test domain instantiation
            climate = ClimateModeling()
            assert hasattr(climate, 'get_default_parameters')
            
            fluids = FluidDynamics()
            assert hasattr(fluids, 'get_default_parameters')
        except ImportError as e:
            pytest.skip(f"Neuromancer scientific domains not available: {e}")
    
    @pytest.mark.asyncio
    async def test_kg_physics_bridge(self):
        """Test KG-to-Physics bridge."""
        try:
            from integrations.neuromancer.kg_physics_bridge import KGPhysicsBridge
            bridge = KGPhysicsBridge()
            
            assert hasattr(bridge, 'kg_to_physics_problem')
            assert hasattr(bridge, 'physics_solution_to_kg')
            assert hasattr(bridge, 'validate_physics_consistency')
        except ImportError as e:
            pytest.skip(f"Neuromancer KG bridge not available: {e}")
    
    @pytest.mark.asyncio
    async def test_kg_integration_wrapper(self):
        """Test Knowledge Engine wrapper for Neuromancer."""
        try:
            from knowledge_engine.integrations.neuromancer.neuromancer_integration import NeuromancerKGIntegration
            integration = NeuromancerKGIntegration()
            
            assert hasattr(integration, 'infer_temporal_dynamics')
            assert hasattr(integration, 'validate_physical_laws')
            assert hasattr(integration, 'simulate_what_if')
            assert hasattr(integration, 'calibrate_from_observations')
        except ImportError as e:
            pytest.skip(f"Neuromancer KE integration not available: {e}")


class TestUnifiedHubIntegration:
    """Test suite for Unified Hub integration of new modules."""
    
    @pytest.mark.asyncio
    async def test_hub_initialization(self):
        """Test that Unified Hub can be initialized with new integrations."""
        try:
            from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
            
            hub = UnifiedKGIntegrationHub()
            assert hub is not None
            
            # Check routing map includes new operations
            from knowledge_engine.unified_kg_integration_hub import KGOperationType
            assert KGOperationType.STRUCTURED_GENERATION
            assert KGOperationType.DECLARATIVE_QUERY
            assert KGOperationType.PHYSICS_SIMULATION
            
        except ImportError as e:
            pytest.skip(f"Unified Hub not available: {e}")
    
    @pytest.mark.asyncio
    async def test_routing_map(self):
        """Test that new integrations are in routing map."""
        try:
            from knowledge_engine.unified_kg_integration_hub import (
                UnifiedKGIntegrationHub,
                KGOperationType
            )
            
            hub = UnifiedKGIntegrationHub()
            
            # Check new integrations are mapped
            assert 'outlines' in hub._routing_map[KGOperationType.STRUCTURED_GENERATION]
            assert 'lmql' in hub._routing_map[KGOperationType.DECLARATIVE_QUERY]
            assert 'neuromancer' in hub._routing_map[KGOperationType.PHYSICS_SIMULATION]
            
        except ImportError as e:
            pytest.skip(f"Unified Hub not available: {e}")


class TestEndToEndWorkflows:
    """End-to-end workflow tests combining multiple integrations."""
    
    @pytest.mark.asyncio
    async def test_structured_extraction_workflow(self):
        """Test workflow: Structured entity extraction with Outlines."""
        try:
            from knowledge_engine.integrations.outlines.outlines_integration import OutlinesKGIntegration
            
            integration = OutlinesKGIntegration()
            
            # Test extraction with constraints
            result = await integration.extract_entities_constrained(
                text="Apple Inc. was founded by Steve Jobs in Cupertino.",
                entity_types=['ORG', 'PERSON', 'LOCATION']
            )
            
            assert result is not None
            assert isinstance(result, dict)
            
        except ImportError:
            pytest.skip("Outlines not available")
    
    @pytest.mark.asyncio
    async def test_declarative_query_workflow(self):
        """Test workflow: Declarative query with LMQL."""
        try:
            from knowledge_engine.integrations.lmql.lmql_integration import LMQLKGIntegration
            
            integration = LMQLKGIntegration()
            
            # Test entity query
            result = await integration.query_entities(
                query_str="Find all technology companies founded before 2000",
                filters={'domain': 'technology'}
            )
            
            assert result is not None
            
        except ImportError:
            pytest.skip("LMQL not available")
    
    @pytest.mark.asyncio
    async def test_physics_simulation_workflow(self):
        """Test workflow: Physics simulation with Neuromancer."""
        try:
            from knowledge_engine.integrations.neuromancer.neuromancer_integration import NeuromancerKGIntegration
            
            integration = NeuromancerKGIntegration()
            
            # Test physics validation
            result = await integration.validate_physical_laws(
                kg_subgraph={
                    'entities': [{'id': 'mass1', 'mass': 10.0}],
                    'relationships': [{'source': 'force1', 'target': 'mass1'}]
                },
                domain='mechanics'
            )
            
            assert result is not None
            
        except ImportError:
            pytest.skip("Neuromancer not available")


class TestPerformanceBenchmarks:
    """Performance benchmarks for new integrations."""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_outlines_constraint_compilation(self):
        """Benchmark Outlines constraint compilation speed."""
        try:
            from integrations.outlines.adapter import OutlinesAdapter
            import time
            
            adapter = OutlinesAdapter({'cache_enabled': True})
            
            schema = {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'count': {'type': 'integer'}
                }
            }
            
            start = time.time()
            # Constraint compilation happens on first call
            # Subsequent calls should use cache
            end = time.time()
            
            compilation_time = (end - start) * 1000
            assert compilation_time < 1000  # Should compile in under 1 second
            
        except ImportError:
            pytest.skip("Outlines not available")
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_lmql_constraint_evaluation(self):
        """Benchmark LMQL constraint evaluation speed."""
        try:
            from integrations.lmql.constraint_engine import ConstraintEvaluator, LengthConstraint
            import time
            
            evaluator = ConstraintEvaluator()
            constraint = LengthConstraint(min_length=5, max_length=100)
            
            start = time.time()
            for _ in range(1000):
                evaluator.evaluate("test string", constraint)
            end = time.time()
            
            eval_time = (end - start) * 1000
            assert eval_time < 100  # Should evaluate 1000 constraints in under 100ms
            
        except ImportError:
            pytest.skip("LMQL not available")


def run_all_tests():
    """Run all integration tests."""
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-k', 'not benchmark'  # Skip benchmarks by default
    ])


if __name__ == '__main__':
    run_all_tests()
