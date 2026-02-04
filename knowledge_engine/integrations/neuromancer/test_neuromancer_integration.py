"""Tests for Neuromancer Knowledge Engine Integration.

Unit tests for physics constraints, KG bridge, ODE/PDE solutions,
and GPU/CPU compatibility.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
import sys

# Check for torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import modules under test
from integrations.neuromancer import (
    # Physics constraints
    ConservationLawConstraint,
    ThermodynamicConstraint,
    MechanicalConstraint,
    ElectromagneticConstraint,
    ChemicalConstraint,
    ConstraintLibrary,
    ConstraintConfig,
    ConservationQuantity,
    ConstraintType,
    create_physics_loss,
    
    # Scientific domains
    ClimateModeling,
    FluidDynamics,
    StructuralMechanics,
    ChemicalKinetics,
    BiologicalSystems,
    DomainLibrary,
    
    # KG Physics bridge
    KGPhysicsBridge,
    PhysicsProblem,
    KGUpdates,
    ConsistencyReport,
    
    # Neural operators
    NeuromancerAdapter,
    NeuralOperatorConfig,
    NeuralOperatorType,
    SolutionResult
)

from knowledge_engine.integrations.neuromancer import (
    NeuromancerKGIntegration,
    PredictionResult,
    ValidationResult,
    CalibrationResult,
    DiscoveredEquation,
    PhysicsAwareEmbedding
)


# =============================================================================
# Physics Constraint Tests
# =============================================================================

class TestConservationLawConstraint:
    """Tests for conservation law constraints."""
    
    def test_mass_conservation_creation(self):
        """Test creation of mass conservation constraint."""
        config = ConstraintConfig(weight=1.0, tolerance=1e-6)
        constraint = ConservationLawConstraint(
            quantity=ConservationQuantity.MASS,
            config=config
        )
        
        assert constraint.name == "mass_conservation"
        assert constraint.constraint_type == ConstraintType.CONSERVATION
        assert constraint.quantity == ConservationQuantity.MASS
    
    def test_energy_conservation_creation(self):
        """Test creation of energy conservation constraint."""
        config = ConstraintConfig(weight=1.0, tolerance=1e-4)
        constraint = ConservationLawConstraint(
            quantity=ConservationQuantity.ENERGY,
            config=config
        )
        
        assert constraint.name == "energy_conservation"
        assert constraint.quantity == ConservationQuantity.ENERGY
    
    def test_momentum_conservation_creation(self):
        """Test creation of momentum conservation constraint."""
        constraint = ConservationLawConstraint(
            quantity=ConservationQuantity.MOMENTUM
        )
        
        assert constraint.name == "momentum_conservation"
        assert constraint.quantity == ConservationQuantity.MOMENTUM
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_conservation_loss_computation(self):
        """Test conservation loss computation with torch."""
        constraint = ConservationLawConstraint(
            quantity=ConservationQuantity.MASS,
            config=ConstraintConfig(device="cpu")
        )
        
        # Create dummy predictions and coordinates
        predictions = torch.ones(10, 1, requires_grad=True)
        time = torch.linspace(0, 1, 10)
        context = {'time': time}
        
        loss = constraint.compute_loss(predictions, context)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_conservation_validation(self):
        """Test conservation constraint validation."""
        constraint = ConservationLawConstraint(
            quantity=ConservationQuantity.ENERGY,
            config=ConstraintConfig(tolerance=0.1)
        )
        
        solution = np.array([1.0, 1.01, 0.99, 1.0])
        violation = constraint.validate(solution, {})
        
        assert violation.constraint_name == "energy_conservation"
        assert isinstance(violation.violated, bool)
        assert hasattr(violation, 'timestamp')


class TestThermodynamicConstraint:
    """Tests for thermodynamic constraints."""
    
    def test_entropy_production_constraint(self):
        """Test entropy production constraint."""
        constraint = ThermodynamicConstraint(
            constraint_name="entropy_production",
            constraint_type="entropy_production"
        )
        
        assert constraint.name == "entropy_production"
        assert constraint.constraint_type == ConstraintType.THERMODYNAMIC
    
    def test_positive_temperature_constraint(self):
        """Test positive temperature constraint."""
        constraint = ThermodynamicConstraint(
            constraint_name="positive_temperature",
            constraint_type="temperature_positive"
        )
        
        assert constraint.thermo_type == "temperature_positive"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_positive_temperature_loss(self):
        """Test positive temperature loss computation."""
        constraint = ThermodynamicConstraint(
            constraint_name="positive_temperature",
            constraint_type="temperature_positive",
            config=ConstraintConfig(device="cpu")
        )
        
        # Test with negative temperature (should give high loss)
        predictions = torch.tensor([[-100.0], [200.0], [300.0]])
        loss = constraint.compute_loss(predictions, {})
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0  # Should penalize negative temperature


class TestMechanicalConstraint:
    """Tests for mechanical constraints."""
    
    def test_newton_second_law_constraint(self):
        """Test Newton's second law constraint."""
        constraint = MechanicalConstraint(
            constraint_name="newton_second_law",
            constraint_type="newton_second_law"
        )
        
        assert constraint.name == "newton_second_law"
        assert constraint.mechanical_type == "newton_second_law"
    
    def test_hooke_law_constraint(self):
        """Test Hooke's law constraint."""
        constraint = MechanicalConstraint(
            constraint_name="hooke_law",
            constraint_type="hooke_law"
        )
        
        assert constraint.name == "hooke_law"
        assert constraint.constraint_type == ConstraintType.MECHANICAL
    
    def test_equilibrium_constraint(self):
        """Test equilibrium constraint."""
        constraint = MechanicalConstraint(
            constraint_name="equilibrium",
            constraint_type="equilibrium"
        )
        
        assert constraint.mechanical_type == "equilibrium"


class TestElectromagneticConstraint:
    """Tests for electromagnetic constraints."""
    
    def test_gauss_law_constraint(self):
        """Test Gauss's law constraint."""
        constraint = ElectromagneticConstraint(
            constraint_name="gauss_law",
            constraint_type="gauss_law"
        )
        
        assert constraint.name == "gauss_law"
        assert constraint.em_type == "gauss_law"
    
    def test_faraday_law_constraint(self):
        """Test Faraday's law constraint."""
        constraint = ElectromagneticConstraint(
            constraint_name="faraday_law",
            constraint_type="faraday_law"
        )
        
        assert constraint.constraint_type == ConstraintType.ELECTROMAGNETIC
    
    def test_no_magnetic_monopoles_constraint(self):
        """Test no magnetic monopoles constraint."""
        constraint = ElectromagneticConstraint(
            constraint_name="no_monopoles",
            constraint_type="no_magnetic_monopoles"
        )
        
        assert constraint.em_type == "no_magnetic_monopoles"


class TestChemicalConstraint:
    """Tests for chemical constraints."""
    
    def test_mass_action_kinetics_constraint(self):
        """Test mass action kinetics constraint."""
        constraint = ChemicalConstraint(
            constraint_name="mass_action",
            constraint_type="mass_action"
        )
        
        assert constraint.name == "mass_action"
        assert constraint.chemical_type == "mass_action"
    
    def test_chemical_equilibrium_constraint(self):
        """Test chemical equilibrium constraint."""
        constraint = ChemicalConstraint(
            constraint_name="equilibrium",
            constraint_type="equilibrium"
        )
        
        assert constraint.chemical_type == "equilibrium"
    
    def test_atom_conservation_constraint(self):
        """Test atom conservation constraint."""
        constraint = ChemicalConstraint(
            constraint_name="atom_conservation",
            constraint_type="conservation_of_atoms"
        )
        
        assert constraint.constraint_type == ConstraintType.CHEMICAL


class TestConstraintLibrary:
    """Tests for constraint library factory methods."""
    
    def test_conservation_of_mass_factory(self):
        """Test mass conservation factory."""
        constraint = ConstraintLibrary.conservation_of_mass()
        assert isinstance(constraint, ConservationLawConstraint)
        assert constraint.quantity == ConservationQuantity.MASS
    
    def test_conservation_of_energy_factory(self):
        """Test energy conservation factory."""
        constraint = ConstraintLibrary.conservation_of_energy()
        assert isinstance(constraint, ConservationLawConstraint)
        assert constraint.quantity == ConservationQuantity.ENERGY
    
    def test_entropy_production_factory(self):
        """Test entropy production factory."""
        constraint = ConstraintLibrary.entropy_production()
        assert isinstance(constraint, ThermodynamicConstraint)
    
    def test_newtons_second_law_factory(self):
        """Test Newton's second law factory."""
        constraint = ConstraintLibrary.newtons_second_law()
        assert isinstance(constraint, MechanicalConstraint)
    
    def test_gauss_law_factory(self):
        """Test Gauss's law factory."""
        constraint = ConstraintLibrary.gauss_law()
        assert isinstance(constraint, ElectromagneticConstraint)
    
    def test_mass_action_kinetics_factory(self):
        """Test mass action kinetics factory."""
        constraint = ConstraintLibrary.mass_action_kinetics()
        assert isinstance(constraint, ChemicalConstraint)


# =============================================================================
# Scientific Domain Tests
# =============================================================================

class TestClimateModeling:
    """Tests for climate modeling domain."""
    
    def test_climate_domain_creation(self):
        """Test climate domain initialization."""
        domain = ClimateModeling()
        
        assert domain.config.name == "ClimateModeling"
        assert len(domain.constraints) > 0
        assert 'gravity' in domain.config.default_parameters
    
    def test_climate_domain_constraints(self):
        """Test climate domain constraints setup."""
        domain = ClimateModeling()
        
        constraint_names = [c.name for c in domain.constraints]
        assert any('energy' in name for name in constraint_names)
        assert any('mass' in name for name in constraint_names)
    
    def test_climate_solve_steady_state(self):
        """Test steady-state climate solve."""
        domain = ClimateModeling()
        
        problem = {
            'type': 'steady_state',
            'boundary_conditions': {'temperature': 300}
        }
        
        result = domain.solve(problem)
        
        assert result.domain == "ClimateModeling"
        assert hasattr(result, 'constraints_satisfied')
        assert hasattr(result, 'timestamp')
    
    def test_climate_validate_solution(self):
        """Test climate solution validation."""
        domain = ClimateModeling()
        
        # Test with valid temperature
        valid_solution = {'temperature_field': np.array([250, 280, 300])}
        errors = domain.validate_solution(valid_solution, {})
        assert len(errors) == 0
        
        # Test with invalid (too high) temperature
        invalid_solution = {'temperature_field': np.array([250, 500, 300])}
        errors = domain.validate_solution(invalid_solution, {})
        assert len(errors) > 0


class TestFluidDynamics:
    """Tests for fluid dynamics domain."""
    
    def test_fluid_domain_creation(self):
        """Test fluid dynamics domain initialization."""
        domain = FluidDynamics()
        
        assert domain.config.name == "FluidDynamics"
        assert 'density' in domain.config.default_parameters
    
    def test_fluid_reynolds_number_computation(self):
        """Test Reynolds number computation."""
        domain = FluidDynamics()
        
        problem = {
            'velocity_scale': 10.0,
            'length_scale': 1.0
        }
        params = domain.config.default_parameters
        
        re = domain._compute_reynolds_number(problem, params)
        
        assert re > 0
        assert isinstance(re, float)
    
    def test_fluid_validate_solution(self):
        """Test fluid dynamics solution validation."""
        domain = FluidDynamics()
        
        # Test with negative density
        invalid_solution = {'density_field': np.array([1.0, -0.5, 1.0])}
        errors = domain.validate_solution(invalid_solution, {})
        assert len(errors) > 0
        assert any('negative density' in e.lower() for e in errors)


class TestStructuralMechanics:
    """Tests for structural mechanics domain."""
    
    def test_structural_domain_creation(self):
        """Test structural mechanics domain initialization."""
        domain = StructuralMechanics()
        
        assert domain.config.name == "StructuralMechanics"
        assert 'youngs_modulus' in domain.config.default_parameters
    
    def test_structural_safety_factor(self):
        """Test safety factor computation."""
        domain = StructuralMechanics()
        
        solution = {'max_stress': 100e6}  # 100 MPa
        params = {'yield_strength': 250e6}  # 250 MPa
        
        sf = domain._compute_safety_factor(solution, params)
        
        assert sf == 2.5  # 250/100
    
    def test_structural_validate_stress(self):
        """Test stress validation."""
        domain = StructuralMechanics()
        
        # Test with stress exceeding yield
        solution = {'max_stress': 300e6}
        context = {'yield_strength': 250e6}
        errors = domain.validate_solution(solution, context)
        
        assert len(errors) > 0
        assert any('yield' in e.lower() for e in errors)


class TestChemicalKinetics:
    """Tests for chemical kinetics domain."""
    
    def test_chemical_domain_creation(self):
        """Test chemical kinetics domain initialization."""
        domain = ChemicalKinetics()
        
        assert domain.config.name == "ChemicalKinetics"
        assert 'universal_gas_constant' in domain.config.default_parameters
    
    def test_chemical_validate_concentration(self):
        """Test concentration validation."""
        domain = ChemicalKinetics()
        
        # Test with negative concentration
        solution = {'concentrations': np.array([1.0, -0.1, 0.5])}
        errors = domain.validate_solution(solution, {})
        
        assert len(errors) > 0
        assert any('negative' in e.lower() for e in errors)


class TestBiologicalSystems:
    """Tests for biological systems domain."""
    
    def test_biological_domain_creation(self):
        """Test biological systems domain initialization."""
        domain = BiologicalSystems()
        
        assert domain.config.name == "BiologicalSystems"
        assert 'carrying_capacity' in domain.config.default_parameters
    
    def test_logistic_solve(self):
        """Test logistic growth model."""
        domain = BiologicalSystems()
        
        problem = {
            'model_type': 'logistic',
            'initial_population': 100
        }
        
        result = domain.solve(problem)
        
        assert result.domain == "BiologicalSystems"
        if result.success and result.solution:
            assert result.solution.get('carrying_capacity') == 1000
    
    def test_r0_computation(self):
        """Test basic reproduction number computation."""
        domain = BiologicalSystems()
        
        params = {
            'transmission_rate': 0.5,
            'recovery_rate': 0.1
        }
        
        r0 = domain._compute_r0({}, params, 'sir')
        
        assert r0 == 5.0  # 0.5/0.1
    
    def test_biological_validate_population(self):
        """Test population validation."""
        domain = BiologicalSystems()
        
        # Test with negative population
        solution = {'population_trajectory': np.array([100, -50, 200])}
        errors = domain.validate_solution(solution, {})
        
        assert len(errors) > 0
        assert any('negative' in e.lower() for e in errors)


class TestDomainLibrary:
    """Tests for domain library."""
    
    def test_get_all_domains(self):
        """Test getting all domains."""
        domains = DomainLibrary.get_all_domains()
        
        assert 'climate' in domains
        assert 'fluid_dynamics' in domains
        assert 'structural_mechanics' in domains
        assert 'chemical_kinetics' in domains
        assert 'biological_systems' in domains
    
    def test_climate_factory(self):
        """Test climate domain factory."""
        domain = DomainLibrary.climate_modeling()
        assert isinstance(domain, ClimateModeling)
    
    def test_fluid_factory(self):
        """Test fluid dynamics factory."""
        domain = DomainLibrary.fluid_dynamics()
        assert isinstance(domain, FluidDynamics)


# =============================================================================
# KG Physics Bridge Tests
# =============================================================================

class TestKGPhysicsBridge:
    """Tests for KG-Physics bridge."""
    
    def test_bridge_creation(self):
        """Test bridge initialization."""
        bridge = KGPhysicsBridge()
        
        assert bridge is not None
        assert len(bridge.domains) > 0
    
    def test_kg_to_physics_problem(self):
        """Test KG to physics problem conversion."""
        bridge = KGPhysicsBridge()
        
        kg_subgraph = {
            'entities': [
                {
                    'id': 'fluid_1',
                    'type': 'fluid',
                    'physics_properties': {'density': 1.225, 'velocity': 10.0}
                }
            ],
            'relationships': []
        }
        
        problem = bridge.kg_to_physics_problem(kg_subgraph)
        
        assert isinstance(problem, PhysicsProblem)
        assert problem.domain == 'fluid_dynamics'
        assert 'density' in str(problem.parameters)
    
    def test_physics_solution_to_kg(self):
        """Test physics solution to KG conversion."""
        from integrations.neuromancer.scientific_domains import SimulationResult
        
        bridge = KGPhysicsBridge()
        
        solution = SimulationResult(
            success=True,
            domain="FluidDynamics",
            solution={'velocity_field': np.zeros((10, 2))},
            metadata={},
            constraints_satisfied={'mass_conservation': True},
            validation_errors=[],
            computation_time=1.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        updates = bridge.physics_solution_to_kg(solution)
        
        assert isinstance(updates, KGUpdates)
        assert updates.metadata['domain'] == "FluidDynamics"
    
    def test_validate_physics_consistency(self):
        """Test physics consistency validation."""
        bridge = KGPhysicsBridge()
        
        kg_data = {
            'entities': [
                {
                    'id': 'entity_1',
                    'type': 'fluid',
                    'physics_properties': {'temperature': 300, 'pressure': 101325}
                }
            ],
            'relationships': []
        }
        
        report = bridge.validate_physics_consistency(kg_data)
        
        assert isinstance(report, ConsistencyReport)
        assert hasattr(report, 'is_consistent')
        assert hasattr(report, 'confidence')
    
    def test_infer_missing_properties(self):
        """Test property inference."""
        bridge = KGPhysicsBridge()
        
        entity = {
            'id': 'fluid_1',
            'type': 'fluid',
            'physics_properties': {
                'density': 1.225,
                'velocity': 10.0
            }
        }
        
        inferred = bridge.infer_missing_properties(entity, "physics_informed")
        
        assert inferred.entity_id == 'fluid_1'
        assert hasattr(inferred, 'inferred_values')
        assert hasattr(inferred, 'confidence')
    
    def test_simulate_system_behavior(self):
        """Test system behavior simulation."""
        bridge = KGPhysicsBridge()
        
        entities = [
            {
                'id': 'fluid_1',
                'type': 'fluid',
                'physics_properties': {'density': 1.0}
            }
        ]
        
        result = bridge.simulate_system_behavior(entities, time_horizon=10.0)
        
        assert result.simulation_id.startswith('sim_')
        assert hasattr(result, 'solution_data')
        assert hasattr(result, 'kg_updates')


# =============================================================================
# Neural Operator Tests
# =============================================================================

class TestNeuromancerAdapter:
    """Tests for Neuromancer neural operator adapter."""
    
    def test_adapter_creation(self):
        """Test adapter initialization."""
        adapter = NeuromancerAdapter(device="cpu")
        
        assert adapter.device == "cpu"
        assert not adapter.initialized
    
    def test_adapter_initialize(self):
        """Test adapter initialization."""
        adapter = NeuromancerAdapter(device="cpu")
        result = adapter.initialize({})
        
        assert result is True
        assert adapter.initialized
    
    def test_solve_ode(self):
        """Test ODE solving."""
        adapter = NeuromancerAdapter(device="cpu")
        adapter.initialize({})
        
        result = adapter.solve_ode(
            system="dy/dt = -y",
            initial_conditions={'y': 1.0},
            t_span=(0, 1),
            num_points=10
        )
        
        assert isinstance(result, SolutionResult)
        assert hasattr(result, 'solution')
        assert hasattr(result, 'coordinates')
    
    def test_solve_pde(self):
        """Test PDE solving."""
        adapter = NeuromancerAdapter(device="cpu")
        adapter.initialize({})
        
        result = adapter.solve_pde(
            equation="laplace",
            domain={
                'dimensions': 2,
                'x_min': 0, 'x_max': 1,
                'y_min': 0, 'y_max': 1,
                'resolution': 20
            },
            boundary_conditions={'dirichlet': {'value': 0}}
        )
        
        assert isinstance(result, SolutionResult)
        assert hasattr(result, 'solution')
    
    def test_learn_dynamics(self):
        """Test dynamics learning."""
        adapter = NeuromancerAdapter(device="cpu")
        adapter.initialize({})
        
        data = np.random.randn(100, 3)
        model = adapter.learn_dynamics(
            data=data,
            variable_names=['x', 'y', 'z'],
            domain_type="generic"
        )
        
        assert model.model_id.startswith('dynamics_')
        assert model.domain == "generic"
        assert model.variable_names == ['x', 'y', 'z']
    
    def test_predict_trajectory(self):
        """Test trajectory prediction."""
        adapter = NeuromancerAdapter(device="cpu")
        adapter.initialize({})
        
        # Create a mock model
        model = adapter.learn_dynamics(
            data=np.random.randn(50, 2),
            variable_names=['x', 'y']
        )
        
        result = adapter.predict_trajectory(
            model=model,
            horizon=10,
            initial_state=np.array([1.0, 0.0])
        )
        
        assert hasattr(result, 'trajectory')
        assert hasattr(result, 'time_points')
    
    def test_calibrate_physics_model(self):
        """Test physics model calibration."""
        adapter = NeuromancerAdapter(device="cpu")
        adapter.initialize({})
        
        observations = [
            {'time': 0, 'value': 1.0},
            {'time': 1, 'value': 0.9},
            {'time': 2, 'value': 0.8}
        ]
        
        params = {'param1': 1.0, 'param2': 0.5}
        
        result = adapter.calibrate_physics_model(observations, params)
        
        assert hasattr(result, 'calibrated_parameters')
        assert hasattr(result, 'calibration_error')


# =============================================================================
# GPU/CPU Compatibility Tests
# =============================================================================

class TestGPUCompatibility:
    """Tests for GPU/CPU compatibility."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_device_selection_cpu(self):
        """Test CPU device selection."""
        adapter = NeuromancerAdapter(device="cpu")
        assert adapter.device == "cpu"
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                        reason="CUDA not available")
    def test_device_selection_cuda(self):
        """Test CUDA device selection."""
        adapter = NeuromancerAdapter(device="cuda")
        assert "cuda" in adapter.device
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_operator_device_placement(self):
        """Test operator device placement."""
        from integrations.neuromancer.neural_operators import PINNOperator
        
        config = NeuralOperatorConfig(
            input_dim=1,
            output_dim=1,
            device="cpu"
        )
        
        operator = PINNOperator(config)
        assert operator._device == "cpu"


# =============================================================================
# Knowledge Engine Integration Tests
# =============================================================================

class TestNeuromancerKGIntegration:
    """Tests for Neuromancer KG Integration."""
    
    def test_integration_creation(self):
        """Test integration initialization."""
        integration = NeuromancerKGIntegration(device="cpu")
        
        assert integration.device == "cpu"
        assert not integration.initialized
    
    def test_integration_initialize(self):
        """Test integration initialization."""
        integration = NeuromancerKGIntegration(device="cpu")
        result = integration.initialize({})
        
        assert result is True
        assert integration.initialized
    
    def test_infer_temporal_dynamics(self):
        """Test temporal dynamics inference."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        # Provide historical data to avoid KG dependency
        historical_data = np.array([[1.0], [1.1], [1.2], [1.15], [1.25]])
        
        result = integration.infer_temporal_dynamics(
            entity_id="test_entity",
            property_name="value",
            horizon=5,
            historical_data=historical_data
        )
        
        assert isinstance(result, PredictionResult)
        assert result.entity_id == "test_entity"
        assert result.horizon == 5
    
    def test_validate_physical_laws(self):
        """Test physical law validation."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        kg_subgraph = {
            'entities': [
                {'id': 'e1', 'type': 'fluid', 'physics_properties': {}}
            ],
            'relationships': []
        }
        
        result = integration.validate_physical_laws(
            kg_subgraph=kg_subgraph,
            domain="fluid_dynamics"
        )
        
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'constraint_scores')
    
    def test_simulate_what_if(self):
        """Test what-if simulation."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        scenario = {
            'entities': [
                {'id': 'f1', 'type': 'fluid', 'physics_properties': {'velocity': 10.0}}
            ],
            'time_horizon': 5.0
        }
        
        result = integration.simulate_what_if(
            scenario=scenario,
            constraints=["conservation_of_mass"]
        )
        
        assert hasattr(result, 'simulation_id')
        assert hasattr(result, 'success')
        assert hasattr(result, 'kg_updates')
    
    def test_calibrate_from_observations(self):
        """Test calibration from observations."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        observations = [
            {'time': 0, 'value': 1.0},
            {'time': 1, 'value': 0.95},
            {'time': 2, 'value': 0.90}
        ]
        
        result = integration.calibrate_from_observations(
            entity_id="sensor_1",
            observations=observations,
            physics_params={'param1': 1.0}
        )
        
        assert isinstance(result, CalibrationResult)
        assert result.entity_id == "sensor_1"
        assert hasattr(result, 'calibrated_model')
    
    def test_discover_equations(self):
        """Test equation discovery."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        data = [
            {'t': 0, 'y': 1.0, 'dy': -0.1},
            {'t': 1, 'y': 0.9, 'dy': -0.09},
            {'t': 2, 'y': 0.81, 'dy': -0.081}
        ]
        
        result = integration.discover_equations(
            data=data,
            candidate_terms=["y", "y^2"],
            entity_id="system_1"
        )
        
        assert isinstance(result, DiscoveredEquation)
        assert result.entity_id == "system_1"
        assert hasattr(result, 'equation_form')
    
    def test_physics_enriched_embedding(self):
        """Test physics-enriched embedding."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        entity = {
            'id': 'test_entity',
            'physics_properties': {
                'temperature': 300,
                'pressure': 101325
            }
        }
        
        result = integration.physics_enriched_embedding(entity)
        
        assert isinstance(result, PhysicsAwareEmbedding)
        assert result.entity_id == "test_entity"
        assert len(result.physics_features) > 0


# =============================================================================
# Integration and End-to-End Tests
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end workflow tests."""
    
    def test_climate_simulation_workflow(self):
        """Test complete climate simulation workflow."""
        # Setup
        kg_bridge = KGPhysicsBridge()
        domain = DomainLibrary.climate_modeling()
        
        # Create KG subgraph
        kg_subgraph = {
            'entities': [
                {
                    'id': 'atmosphere',
                    'type': 'atmosphere',
                    'physics_properties': {
                        'temperature': 288,
                        'pressure': 101325
                    }
                }
            ],
            'relationships': []
        }
        
        # Convert to physics problem
        physics_problem = kg_bridge.kg_to_physics_problem(kg_subgraph)
        
        # Solve
        problem = {
            'type': 'steady_state',
            'boundary_conditions': {'temperature': 300}
        }
        result = domain.solve(problem)
        
        # Convert back to KG
        kg_updates = kg_bridge.physics_solution_to_kg(result)
        
        # Verify
        assert result.domain == "ClimateModeling"
        assert kg_updates.metadata['domain'] == "ClimateModeling"
    
    def test_fluid_dynamics_with_constraints(self):
        """Test fluid dynamics with physics constraints."""
        # Setup domain
        domain = DomainLibrary.fluid_dynamics()
        
        # Get constraints
        constraints = domain.get_constraints()
        
        # Verify constraints are set up
        assert len(constraints) > 0
        
        # Run simulation
        problem = {
            'flow_type': 'incompressible',
            'velocity_scale': 10.0,
            'length_scale': 1.0
        }
        result = domain.solve(problem)
        
        # Verify constraints were checked
        assert len(result.constraints_satisfied) > 0
    
    def test_structural_analysis_with_safety(self):
        """Test structural analysis with safety factor."""
        domain = DomainLibrary.structural_mechanics()
        
        problem = {
            'analysis_type': 'static',
            'loads': {'force': 1000}
        }
        result = domain.solve(problem)
        
        if result.success and result.solution:
            # Check safety factor in metadata
            assert 'safety_factor' in result.metadata
    
    def test_physics_constraint_combination(self):
        """Test combining multiple physics constraints."""
        constraints = [
            ConstraintLibrary.conservation_of_mass(),
            ConstraintLibrary.conservation_of_energy(),
            ConstraintLibrary.entropy_production()
        ]
        
        assert len(constraints) == 3
        
        # Test with dummy data
        if TORCH_AVAILABLE:
            predictions = torch.ones(10, 1)
            context = {'coordinates': torch.randn(10, 2)}
            
            total_loss = create_physics_loss(constraints, predictions, context)
            assert isinstance(total_loss, torch.Tensor)


# =============================================================================
# Performance and Memory Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_large_domain_simulation(self):
        """Test simulation with larger domains."""
        adapter = NeuromancerAdapter(device="cpu")
        adapter.initialize({})
        
        # Larger domain
        result = adapter.solve_pde(
            equation="laplace",
            domain={
                'dimensions': 2,
                'x_min': 0, 'x_max': 10,
                'y_min': 0, 'y_max': 10,
                'resolution': 50
            },
            boundary_conditions={'value': 0}
        )
        
        assert result.success
        # Should complete within reasonable time
    
    def test_model_caching(self):
        """Test model caching behavior."""
        integration = NeuromancerKGIntegration(device="cpu")
        integration.initialize({})
        
        # Learn model
        data = np.random.randn(100, 2)
        model = integration.neural_adapter.learn_dynamics(
            data=data,
            variable_names=['x', 'y']
        )
        
        # Cache model
        model_key = "test_model"
        integration.learned_models[model_key] = model
        
        # Retrieve from cache
        cached_model = integration.learned_models.get(model_key)
        assert cached_model is not None
        assert cached_model.model_id == model.model_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
