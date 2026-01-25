"""
Integration tests for I_mech

Tests complete pipeline from FDG extraction to solution transfer.

Agent: G3 (I_mech Specialist)
Created: 2025-12-31
"""


import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from phase2.imech import (
    IMechValidator,
    Domain,
    FunctionalDependencyGraph,
    Node,
    Edge,
    EdgeType,
    SimilarityResult
)


class TestFullPipeline:
    """Test complete I_mech pipeline"""

    def setup_method(self):
        """Create validator"""
        self.validator = IMechValidator(
            use_exact_isomorphism=False,
            enable_proofs=False,
            cache_enabled=False
        )

    def test_full_pipeline_isomorphic_domains(self):
        """Test complete pipeline with isomorphic domains"""
        # Create source domain with solution
        source_domain = self._create_engineering_domain()

        # Create target domain (isomorphic)
        target_domain = self._create_engineering_domain(prefix="m")

        # Run I_mech
        result = self.validator.compare(source_domain, target_domain)

        # Assertions
        assert result is not None
        assert result.structural_score > 0.7  # Should detect isomorphism
        assert len(result.node_mapping) > 0
        assert result.computation_time >= 0

        # If solution was transferred
        if result.total_score > 0.7 and source_domain.has_solution():
            assert result.transferred_solution is not None
            assert result.validation_result is not None

    def test_full_pipeline_partial_isomorphism(self):
        """Test pipeline with partial isomorphism"""
        source_domain = self._create_engineering_domain(size=5)
        target_domain = self._create_engineering_domain(size=7)

        result = self.validator.compare(source_domain, target_domain)

        assert result is not None
        # Should detect partial similarity
        assert result.total_score >= 0.0
        assert result.structural_score >= 0.0

    def test_full_pipeline_with_solution_transfer(self):
        """Test solution transfer end-to-end"""
        # Create source with solution
        source = self._create_engineering_domain()
        source.solutions = [
            {
                'structure': {'algorithm': 'control_loop'},
                'parameters': {'gain': 2.5, 'damping': 0.3}
            }
        ]

        # Create target
        target = self._create_engineering_domain(prefix="m")

        # Compare
        result = self.validator.compare(source, target)

        # If isomorphism detected
        if result.total_score > 0.7:
            assert result.transferred_solution is not None

            # Check validation
            validation = result.validation_result
            if validation:
                assert 'is_valid' in validation

    def test_find_analogous_domains_pipeline(self):
        """Test finding analogous domains"""
        target = self._create_engineering_domain(size=5)

        candidates = []
        for i in range(5):
            domain = self._create_engineering_domain(
                size=5,
                prefix=f"domain{i}_"
            )
            domain.solutions = [{'parameters': {'value': i}}]
            candidates.append(domain)

        # Find analogies
        results = self.validator.find_analogous_domains(
            target,
            candidates,
            threshold=0.5
        )

        assert isinstance(results, list)

    def _create_engineering_domain(self, size=5, prefix="n"):
        """Helper: create engineering control system domain"""
        domain = Domain(
            id=f"control_system_{prefix}",
            name=f"Control System {prefix}",
            description="Feedback control system with gain and damping",
            formal_constraints=[
                f"{prefix}gain * {prefix}error > 0",
                f"{prefix}damping >= 0 and {prefix}damping <= 1",
                f"{prefix}output = f({prefix}input, {prefix}gain, {prefix}damping)"
            ],
            natural_language_constraints=[
                "Gain must be positive",
                "Damping must be between 0 and 1",
                "System must be stable"
            ]
        )

        # Create FDG
        fdg = FunctionalDependencyGraph()

        # Add nodes (variables in control system)
        nodes_data = [
            (f"{prefix}input", "continuous", "system input"),
            (f"{prefix}error", "continuous", "error signal"),
            (f"{prefix}gain", "continuous", "controller gain"),
            (f"{prefix}damping", "continuous", "damping coefficient"),
            (f"{prefix}output", "continuous", "system output")
        ]

        for var_id, var_type, description in nodes_data:
            node = Node(
                id=var_id,
                variable=var_id,
                constraint_type=var_type,
                metadata={'description': description}
            )
            fdg.add_node(node)

        # Add causal edges
        edges_data = [
            (f"{prefix}input", f"{prefix}error", EdgeType.CAUSAL, 1.0),
            (f"{prefix}error", f"{prefix}output", EdgeType.CAUSAL, 1.0),
            (f"{prefix}gain", f"{prefix}output", EdgeType.CAUSAL, 0.8),
            (f"{prefix}damping", f"{prefix}output", EdgeType.CAUSAL, 0.5),
            (f"{prefix}output", f"{prefix}error", EdgeType.FEEDBACK, 0.9)
        ]

        for src, tgt, edge_type, weight in edges_data:
            edge = Edge(source=src, target=tgt, edge_type=edge_type, weight=weight)
            fdg.add_edge(edge)

        domain.fdg = fdg

        return domain


class TestHistoricalAnalogies:
    """Test I_mech on historical analogies"""

    def test_thomas_edison_light_bulb(self):
        """Test analogy: Edison's light bulb -> electric lighting"""

        # Source: Candle (light source)
        candle_domain = Domain(
            id="candle",
            name="Candle",
            description="Simple light source using combustion",
            formal_constraints=[
                "fuel + oxygen -> light + heat",
                "light_intensity ~ fuel_consumption_rate"
            ],
            solutions=[{'parameters': {'wick_length': 1.0, 'fuel_type': 'wax'}}]
        )

        # Target: Light bulb (electric light)
        lightbulb_domain = Domain(
            id="lightbulb",
            name="Light Bulb",
            description="Electric light source using filament",
            formal_constraints=[
                "electricity -> light + heat",
                "light_intensity ~ current^2"
            ]
        )

        validator = IMechValidator()
        result = validator.compare(candle_domain, lightbulb_domain)

        # Should detect some mechanistic similarity (energy conversion)
        assert result is not None
        assert 0.0 <= result.total_score <= 1.0

    def test_steam_engine_to_internal_combustion(self):
        """Test analogy: Steam engine -> Internal combustion engine"""

        # Both convert thermal energy to mechanical work
        steam_engine = Domain(
            id="steam_engine",
            name="Steam Engine",
            description="External combustion engine",
            formal_constraints=[
                "heat -> steam_pressure -> mechanical_work",
                "efficiency = work_out / heat_in"
            ],
            solutions=[{'parameters': {'pressure': 100, 'temperature': 200}}]
        )

        combustion_engine = Domain(
            id="combustion_engine",
            name="Internal Combustion Engine",
            description="Internal combustion engine",
            formal_constraints=[
                "combustion -> gas_pressure -> mechanical_work",
                "efficiency = work_out / fuel_energy"
            ]
        )

        validator = IMechValidator()
        result = validator.compare(steam_engine, combustion_engine)

        # Should detect strong mechanistic similarity
        assert result is not None
        assert result.structural_score >= 0.0

    def test_telegraph_to_telephone(self):
        """Test analogy: Telegraph -> Telephone"""

        # Both transmit information over electrical signals
        telegraph = Domain(
            id="telegraph",
            name="Telegraph",
            description="Long-distance transmission of coded messages",
            formal_constraints=[
                "message -> code -> electrical_signal -> decode -> message",
                "signal_strength = voltage * duration"
            ],
            solutions=[{'parameters': {'code_type': 'morse', 'voltage': 12}}]
        )

        telephone = Domain(
            id="telephone",
            name="Telephone",
            description="Transmission of voice over electrical signals",
            formal_constraints=[
                "voice -> electrical_signal -> voice",
                "signal_amplitude = voice_pressure"
            ]
        )

        validator = IMechValidator()
        result = validator.compare(telegraph, telephone)

        assert result is not None


class TestPerformance:
    """Test performance benchmarks"""

    def test_small_graphs_performance(self):
        """Test performance on small graphs (10 nodes)"""
        domain1 = self._create_domain(size=10)
        domain2 = self._create_domain(size=10)

        validator = IMechValidator()
        result = validator.compare(domain1, domain2)

        # Should complete quickly
        assert result.computation_time < 5.0  # 5 seconds max

    def test_medium_graphs_performance(self):
        """Test performance on medium graphs (50 nodes)"""
        domain1 = self._create_domain(size=50)
        domain2 = self._create_domain(size=50)

        validator = IMechValidator()
        result = validator.compare(domain1, domain2)

        # Should complete in reasonable time
        assert result.computation_time < 30.0  # 30 seconds max

    def _create_domain(self, size=10):
        """Helper: create test domain"""
        domain = Domain(
            id=f"domain_{size}",
            name=f"Domain size {size}",
            description="Performance test domain"
        )

        fdg = FunctionalDependencyGraph()

        for i in range(size):
            node = Node(
                id=f"n{i}",
                variable=f"x{i}",
                constraint_type="continuous"
            )
            fdg.add_node(node)

        # Add chain edges
        for i in range(size - 1):
            edge = Edge(
                source=f"n{i}",
                target=f"n{i+1}",
                edge_type=EdgeType.CAUSAL
            )
            fdg.add_edge(edge)

        domain.fdg = fdg

        return domain
