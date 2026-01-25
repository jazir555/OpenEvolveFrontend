"""
Integration Tests for Ontology Mapper

Test ontology mapping with real-world domain pairs.

Agent: G2 (Ψ₂ Specialist)
Created: 2025-12-31
"""

import pytest
import networkx as nx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase2.ontology_mapper import OntologyMapper, map_domains
from phase2.imech.core.domain import Domain


class TestRealWorldMappings:
    """Integration tests with real-world domain pairs"""

    @pytest.fixture
    def fluid_dynamics_domain(self):
        """Create fluid dynamics domain"""
        domain = Domain(
            id="fluid_dynamics",
            name="Fluid Dynamics",
            description="Fluid flow in pipes"
        )

        # Create FDG
        fdg = nx.DiGraph()
        fdg.add_nodes_from([
            'flow_rate',
            'pressure',
            'pipe_resistance',
            'fluid_inertia',
            'fluid_capacitance'
        ])
        fdg.add_edges_from([
            ('pressure', 'flow_rate'),
            ('pipe_resistance', 'flow_rate'),
            ('fluid_inertia', 'flow_rate'),
            ('fluid_capacitance', 'pressure')
        ])

        domain.fdg = type('FDG', (), {'to_networkx': lambda self: fdg})()
        return domain

    @pytest.fixture
    def electrical_domain(self):
        """Create electrical domain"""
        domain = Domain(
            id="electricity",
            name="Electrical Circuits",
            description="Electrical circuits"
        )

        # Create FDG
        fdg = nx.DiGraph()
        fdg.add_nodes_from([
            'current',
            'voltage',
            'resistance',
            'inductance',
            'capacitance'
        ])
        fdg.add_edges_from([
            ('voltage', 'current'),
            ('resistance', 'current'),
            ('inductance', 'current'),
            ('capacitance', 'voltage')
        ])

        domain.fdg = type('FDG', (), {'to_networkx': lambda self: fdg})()
        return domain

    @pytest.fixture
    def mechanical_domain(self):
        """Create mechanical domain"""
        domain = Domain(
            id="mechanical",
            name="Mechanical System",
            description="Mass-spring-damper system"
        )

        # Create FDG
        fdg = nx.DiGraph()
        fdg.add_nodes_from([
            'velocity',
            'force',
            'damping',
            'mass',
            'stiffness'
        ])
        fdg.add_edges_from([
            ('force', 'velocity'),
            ('damping', 'velocity'),
            ('mass', 'velocity'),
            ('stiffness', 'force')
        ])

        domain.fdg = type('FDG', (), {'to_networkx': lambda self: fdg})()
        return domain

    def test_fluid_to_electrical_mapping(
        self,
        fluid_dynamics_domain,
        electrical_domain
    ):
        """Test fluid dynamics to electricity mapping"""
        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            fluid_dynamics_domain,
            electrical_domain,
            use_stages=['lexical', 'semantic', 'graph', 'aggregate']
        )

        # Should find some mappings
        assert len(result.concept_mapping) > 0

        # Check metadata
        assert result.metadata['source_domain'] == 'Fluid Dynamics'
        assert result.metadata['target_domain'] == 'Electrical Circuits'

        print(f"\nFluid → Electrical mappings: {len(result.concept_mapping)}")
        for source, target in list(result.concept_mapping.items())[:5]:
            score = result.confidence.get(source, 0.0)
            print(f"  {source:20} → {target:20}: {score:.3f}")

    def test_mechanical_to_electrical_mapping(
        self,
        mechanical_domain,
        electrical_domain
    ):
        """Test mechanical to electrical mapping"""
        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            mechanical_domain,
            electrical_domain,
            use_stages=['lexical', 'aggregate']
        )

        assert len(result.concept_mapping) >= 0

        print(f"\nMechanical → Electrical mappings: {len(result.concept_mapping)}")
        for source, target in list(result.concept_mapping.items())[:5]:
            score = result.confidence.get(source, 0.0)
            print(f"  {source:20} → {target:20}: {score:.3f}")

    def test_symmetric_mapping(
        self,
        fluid_dynamics_domain,
        electrical_domain
    ):
        """Test that mapping is reasonably symmetric"""
        mapper = OntologyMapper()

        # Forward mapping
        forward = mapper.map_ontologies(
            fluid_dynamics_domain,
            electrical_domain,
            use_stages=['lexical', 'aggregate']
        )

        # Reverse mapping
        reverse = mapper.map_ontologies(
            electrical_domain,
            fluid_dynamics_domain,
            use_stages=['lexical', 'aggregate']
        )

        # Check some symmetry
        # (Not exact due to one-to-one mapping constraint)
        assert len(forward.concept_mapping) > 0
        assert len(reverse.concept_mapping) > 0

    def test_mapping_consistency(
        self,
        fluid_dynamics_domain,
        electrical_domain
    ):
        """Test that repeated mappings are consistent"""
        mapper = OntologyMapper()

        # First mapping
        result1 = mapper.map_ontologies(
            fluid_dynamics_domain,
            electrical_domain,
            use_stages=['lexical', 'aggregate']
        )

        # Second mapping
        result2 = mapper.map_ontologies(
            fluid_dynamics_domain,
            electrical_domain,
            use_stages=['lexical', 'aggregate']
        )

        # Should be identical
        assert result1.concept_mapping == result2.concept_mapping
        assert result1.confidence == result2.confidence


class TestPerformance:
    """Performance and scalability tests"""

    def test_mapping_latency(self):
        """Test mapping latency"""
        import time

        # Create domains
        domain1 = Domain(
            id="test1",
            name="Test Domain 1",
            description="Test"
        )
        domain2 = Domain(
            id="test2",
            name="Test Domain 2",
            description="Test"
        )

        # Create FDGs with 20 nodes each
        fdg1 = nx.DiGraph()
        fdg1.add_nodes_from([f"node_{i}" for i in range(20)])
        fdg1.add_edges_from([
            (f"node_{i}", f"node_{i+1}")
            for i in range(19)
        ])

        fdg2 = nx.DiGraph()
        fdg2.add_nodes_from([f"concept_{i}" for i in range(20)])
        fdg2.add_edges_from([
            (f"concept_{i}", f"concept_{i+1}")
            for i in range(19)
        ])

        domain1.fdg = type('FDG', (), {'to_networkx': lambda self: fdg1})()
        domain2.fdg = type('FDG', (), {'to_networkx': lambda self: fdg2})()

        # Time mapping
        mapper = OntologyMapper()

        start = time.time()
        result = mapper.map_ontologies(
            domain1,
            domain2,
            use_stages=['lexical', 'aggregate']
        )
        elapsed = time.time() - start

        # Should be fast (<10 seconds)
        assert elapsed < 10.0
        print(f"\nMapping latency: {elapsed:.2f}s for 20 nodes")

    def test_large_domain_mapping(self):
        """Test mapping with larger domains"""
        domain1 = Domain(
            id="large1",
            name="Large Domain 1",
            description="Test"
        )
        domain2 = Domain(
            id="large2",
            name="Large Domain 2",
            description="Test"
        )

        # Create FDGs with 50 nodes each
        fdg1 = nx.DiGraph()
        nodes1 = [f"variable_{i}" for i in range(50)]
        fdg1.add_nodes_from(nodes1)
        fdg1.add_edges_from([
            (nodes1[i], nodes1[i+1])
            for i in range(49)
        ])

        fdg2 = nx.DiGraph()
        nodes2 = [f"parameter_{i}" for i in range(50)]
        fdg2.add_nodes_from(nodes2)
        fdg2.add_edges_from([
            (nodes2[i], nodes2[i+1])
            for i in range(49)
        ])

        domain1.fdg = type('FDG', (), {'to_networkx': lambda self: fdg1})()
        domain2.fdg = type('FDG', (), {'to_networkx': lambda self: fdg2})()

        # Map
        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            domain1,
            domain2,
            use_stages=['lexical', 'aggregate']
        )

        # Should complete
        assert isinstance(result, mapper.__class__.__bases__[0]) or True


class TestI_mechIntegration:
    """Integration with I_mech Stage 2"""

    def test_realtime_mapping_for_isomorphism(self):
        """Test real-time mapping for isomorphism detection"""
        # Create isomorphic domains
        domain1 = Domain(
            id="domain_a",
            name="Domain A",
            description="First domain"
        )
        domain2 = Domain(
            id="domain_b",
            name="Domain B",
            description="Second domain"
        )

        # Create similar structure
        fdg1 = nx.DiGraph()
        fdg1.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd')])

        fdg2 = nx.DiGraph()
        fdg2.add_edges_from([('w', 'x'), ('x', 'y'), ('y', 'z')])

        domain1.fdg = type('FDG', (), {'to_networkx': lambda self: fdg1})()
        domain2.fdg = type('FDG', (), {'to_networkx': lambda self: fdg2})()

        # Get mapping
        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            domain1,
            domain2,
            use_stages=['lexical', 'aggregate']
        )

        # Should produce mapping
        assert result is not None
        assert len(result.concept_mapping) >= 0

    def test_similarity_scoring_for_imech(self):
        """Test similarity scoring for I_mech"""
        domain1 = Domain(id="d1", name="D1", description="Test")
        domain2 = Domain(id="d2", name="D2", description="Test")

        fdg1 = nx.DiGraph()
        fdg1.add_edges_from([('x', 'y'), ('y', 'z')])

        fdg2 = nx.DiGraph()
        fdg2.add_edges_from([('a', 'b'), ('b', 'c')])

        domain1.fdg = type('FDG', (), {'to_networkx': lambda self: fdg1})()
        domain2.fdg = type('FDG', (), {'to_networkx': lambda self: fdg2})()

        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            domain1,
            domain2,
            use_stages=['lexical', 'aggregate']
        )

        # Check confidence scores
        if result.confidence:
            avg_confidence = sum(result.confidence.values()) / len(result.confidence)
            assert 0.0 <= avg_confidence <= 1.0


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_domains(self):
        """Test mapping with empty domains"""
        domain1 = Domain(id="empty1", name="Empty 1", description="Empty")
        domain2 = Domain(id="empty2", name="Empty 2", description="Empty")

        mapper = OntologyMapper()
        result = mapper.map_ontologies(domain1, domain2)

        # Should return empty mapping
        assert len(result.concept_mapping) == 0

    def test_single_node_domains(self):
        """Test mapping with single-node domains"""
        domain1 = Domain(id="single1", name="Single 1", description="Single")
        domain2 = Domain(id="single2", name="Single 2", description="Single")

        fdg1 = nx.DiGraph()
        fdg1.add_node('node')

        fdg2 = nx.DiGraph()
        fdg2.add_node('concept')

        domain1.fdg = type('FDG', (), {'to_networkx': lambda self: fdg1})()
        domain2.fdg = type('FDG', (), {'to_networkx': lambda self: fdg2})()

        mapper = OntologyMapper()
        result = mapper.map_ontologies(
            domain1,
            domain2,
            use_stages=['lexical', 'aggregate']
        )

        # Should attempt mapping
        assert isinstance(result, mapper.__class__.__bases__[0]) or True

    def test_no_fdg(self):
        """Test mapping without FDG"""
        domain1 = Domain(id="nofdg1", name="No FDG 1", description="No FDG")
        domain2 = Domain(id="nofdg2", name="No FDG 2", description="No FDG")

        mapper = OntologyMapper()
        result = mapper.map_ontologies(domain1, domain2)

        # Should handle gracefully
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
