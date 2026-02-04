"""Scientific domain configurations for Neuromancer.

Pre-configured models for different scientific domains.
Each domain includes: default parameters, typical constraints, validation rules.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .physics_constraints import (
    ConstraintLibrary, ConservationLawConstraint, ThermodynamicConstraint,
    MechanicalConstraint, ElectromagneticConstraint, ChemicalConstraint,
    ConstraintConfig, PhysicsConstraint
)

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Types of scientific domains."""
    CLIMATE = "climate"
    FLUID_DYNAMICS = "fluid_dynamics"
    STRUCTURAL_MECHANICS = "structural_mechanics"
    CHEMICAL_KINETICS = "chemical_kinetics"
    BIOLOGICAL_SYSTEMS = "biological_systems"
    ELECTROMAGNETIC = "electromagnetic"


@dataclass
class DomainConfig:
    """Configuration for a scientific domain."""
    name: str
    domain_type: DomainType
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    typical_constraints: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)
    characteristic_scales: Dict[str, float] = field(default_factory=dict)
    numerical_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of a domain simulation."""
    success: bool
    domain: str
    solution: Any
    metadata: Dict[str, Any]
    constraints_satisfied: Dict[str, bool]
    validation_errors: List[str]
    computation_time: float
    timestamp: str


class ScientificDomain(ABC):
    """Base class for scientific domain implementations."""

    def __init__(self, config: DomainConfig):
        self.config = config
        self.constraints: List[PhysicsConstraint] = []
        self._setup_constraints()
        logger.info(f"Initialized {config.name} domain with {len(self.constraints)} constraints")

    @abstractmethod
    def _setup_constraints(self):
        """Set up domain-specific physics constraints."""
        raise NotImplementedError

    @abstractmethod
    def solve(
        self,
        problem: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """Solve a problem in this domain."""
        raise NotImplementedError

    @abstractmethod
    def validate_solution(self, solution: Any, context: Dict[str, Any]) -> List[str]:
        """Validate solution against domain-specific rules."""
        raise NotImplementedError

    def get_constraints(self) -> List[PhysicsConstraint]:
        """Get all physics constraints for this domain."""
        return self.constraints

    def normalize_units(self, value: float, quantity: str) -> float:
        """Normalize value to characteristic scale."""
        scale = self.config.characteristic_scales.get(quantity, 1.0)
        return value / scale

    def denormalize_units(self, normalized_value: float, quantity: str) -> float:
        """Convert normalized value back to physical units."""
        scale = self.config.characteristic_scales.get(quantity, 1.0)
        return normalized_value * scale


class ClimateModeling(ScientificDomain):
    """Climate and weather modeling domain."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        config = DomainConfig(
            name="ClimateModeling",
            domain_type=DomainType.CLIMATE,
            default_parameters={
                "gravity": 9.81,  # m/s²
                "earth_radius": 6.371e6,  # m
                "rotation_rate": 7.292e-5,  # rad/s
                "specific_heat_air": 1004,  # J/(kg·K)
                "gas_constant": 287,  # J/(kg·K)
                "reference_temperature": 288,  # K
                "reference_pressure": 101325,  # Pa
            },
            typical_constraints=[
                "energy_conservation",
                "mass_conservation",
                "positive_temperature",
                "entropy_production"
            ],
            validation_rules=[
                "temperature_within_physical_limits",
                "pressure_positive",
                "wind_speed_reasonable"
            ],
            units={
                "temperature": "K",
                "pressure": "Pa",
                "velocity": "m/s",
                "density": "kg/m³"
            },
            characteristic_scales={
                "temperature": 50.0,  # Temperature variation scale (K)
                "pressure": 10000.0,  # Pressure variation scale (Pa)
                "velocity": 50.0,  # Wind speed scale (m/s)
                "time": 86400.0,  # Daily scale (s)
                "length": 1e6  # 1000 km scale (m)
            },
            numerical_settings={
                "time_step": 3600,  # 1 hour
                "horizontal_resolution": 100000,  # 100 km
                "vertical_levels": 20
            }
        )
        
        if custom_config:
            config.default_parameters.update(custom_config.get('parameters', {}))
            config.numerical_settings.update(custom_config.get('numerical', {}))
        
        super().__init__(config)

    def _setup_constraints(self):
        """Set up climate modeling constraints."""
        # Energy conservation (first law of thermodynamics)
        energy_config = ConstraintConfig(weight=1.0, tolerance=1e-4)
        self.constraints.append(ConstraintLibrary.conservation_of_energy(energy_config))
        
        # Mass conservation (continuity equation)
        mass_config = ConstraintConfig(weight=1.0, tolerance=1e-5)
        self.constraints.append(ConstraintLibrary.conservation_of_mass(mass_config))
        
        # Positive temperature (thermodynamics)
        temp_config = ConstraintConfig(weight=0.5, tolerance=1e-6)
        self.constraints.append(ConstraintLibrary.positive_temperature(temp_config))
        
        # Entropy production (second law)
        entropy_config = ConstraintConfig(weight=0.3, tolerance=1e-5)
        self.constraints.append(ConstraintLibrary.entropy_production(entropy_config))

    def solve(
        self,
        problem: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """Solve climate modeling problem."""
        from datetime import datetime, timezone
        import time
        
        start_time = time.time()
        params = {**self.config.default_parameters, **(parameters or {})}
        
        problem_type = problem.get('type', 'evolution')
        
        try:
            if problem_type == 'evolution':
                solution = self._solve_time_evolution(problem, params)
            elif problem_type == 'steady_state':
                solution = self._solve_steady_state(problem, params)
            elif problem_type == 'ensemble':
                solution = self._solve_ensemble(problem, params)
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")
            
            # Validate solution
            validation_errors = self.validate_solution(solution, problem)
            
            # Check constraints
            constraints_satisfied = {}
            for constraint in self.constraints:
                violation = constraint.validate(solution, problem)
                constraints_satisfied[constraint.name] = not violation.violated
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                success=len(validation_errors) == 0,
                domain=self.config.name,
                solution=solution,
                metadata={
                    "problem_type": problem_type,
                    "parameters": params,
                    "computation_time": computation_time
                },
                constraints_satisfied=constraints_satisfied,
                validation_errors=validation_errors,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Climate modeling solve failed: {e}")
            return SimulationResult(
                success=False,
                domain=self.config.name,
                solution=None,
                metadata={"error": str(e)},
                constraints_satisfied={},
                validation_errors=[str(e)],
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _solve_time_evolution(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve time evolution of climate state."""
        # Simplified implementation - would use full atmospheric equations
        initial_state = problem.get('initial_state', {})
        time_span = problem.get('time_span', (0, 86400))
        
        # Placeholder for actual solver
        solution = {
            "type": "time_evolution",
            "time_span": time_span,
            "initial_state": initial_state,
            "final_state": initial_state,  # Would be computed
            "trajectory": []  # Would contain time series
        }
        
        return solution

    def _solve_steady_state(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve for steady-state climate."""
        boundary_conditions = problem.get('boundary_conditions', {})
        
        solution = {
            "type": "steady_state",
            "boundary_conditions": boundary_conditions,
            "temperature_field": None,  # Would be computed
            "pressure_field": None,
            "wind_field": None
        }
        
        return solution

    def _solve_ensemble(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve ensemble of climate scenarios."""
        n_members = problem.get('ensemble_size', 10)
        
        solution = {
            "type": "ensemble",
            "ensemble_size": n_members,
            "members": [],  # Would contain individual solutions
            "statistics": {}
        }
        
        return solution

    def validate_solution(self, solution: Any, context: Dict[str, Any]) -> List[str]:
        """Validate climate solution."""
        errors = []
        
        if solution is None:
            errors.append("Solution is None")
            return errors
        
        # Check temperature bounds
        temp = solution.get('temperature_field') if isinstance(solution, dict) else None
        if temp is not None:
            if TORCH_AVAILABLE and isinstance(temp, torch.Tensor):
                temp = temp.detach().cpu().numpy()
            if np.any(temp < 150):  # Below coldest Earth temperature
                errors.append("Temperature below physical limit (150K)")
            if np.any(temp > 400):  # Above hottest Earth temperature
                errors.append("Temperature above physical limit (400K)")
        
        # Check pressure bounds
        pressure = solution.get('pressure_field') if isinstance(solution, dict) else None
        if pressure is not None:
            if TORCH_AVAILABLE and isinstance(pressure, torch.Tensor):
                pressure = pressure.detach().cpu().numpy()
            if np.any(pressure < 0):
                errors.append("Negative pressure detected")
        
        return errors


class FluidDynamics(ScientificDomain):
    """Fluid dynamics and flow simulation domain."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        config = DomainConfig(
            name="FluidDynamics",
            domain_type=DomainType.FLUID_DYNAMICS,
            default_parameters={
                "density": 1.225,  # kg/m³ (air at sea level)
                "viscosity": 1.81e-5,  # Pa·s (dynamic viscosity)
                "kinematic_viscosity": 1.48e-5,  # m²/s
                "bulk_modulus": 1.42e5,  # Pa (for air)
                "surface_tension": 0.072,  # N/m (water-air)
            },
            typical_constraints=[
                "mass_conservation",
                "momentum_conservation",
                "energy_conservation",
                "incompressibility"
            ],
            validation_rules=[
                "velocity_divergence_zero_for_incompressible",
                "pressure_gradient_physical",
                "no_negative_density"
            ],
            units={
                "velocity": "m/s",
                "pressure": "Pa",
                "density": "kg/m³",
                "vorticity": "1/s"
            },
            characteristic_scales={
                "velocity": 10.0,  # m/s
                "pressure": 1000.0,  # Pa
                "length": 1.0,  # m
                "time": 0.1,  # s
                "density": 1.0  # kg/m³
            },
            numerical_settings={
                "courant_number": 0.5,
                "time_step": 0.01,
                "grid_resolution": 100
            }
        )
        
        if custom_config:
            config.default_parameters.update(custom_config.get('parameters', {}))
            config.numerical_settings.update(custom_config.get('numerical', {}))
        
        super().__init__(config)

    def _setup_constraints(self):
        """Set up fluid dynamics constraints."""
        # Mass conservation (continuity)
        mass_config = ConstraintConfig(weight=1.0, tolerance=1e-6)
        self.constraints.append(ConstraintLibrary.conservation_of_mass(mass_config))
        
        # Momentum conservation (Navier-Stokes)
        momentum_config = ConstraintConfig(weight=1.0, tolerance=1e-5)
        self.constraints.append(ConstraintLibrary.conservation_of_momentum(momentum_config))
        
        # Energy conservation
        energy_config = ConstraintConfig(weight=0.8, tolerance=1e-4)
        self.constraints.append(ConstraintLibrary.conservation_of_energy(energy_config))

    def solve(
        self,
        problem: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """Solve fluid dynamics problem."""
        from datetime import datetime, timezone
        import time
        
        start_time = time.time()
        params = {**self.config.default_parameters, **(parameters or {})}
        
        flow_type = problem.get('flow_type', 'incompressible')
        
        try:
            if flow_type == 'incompressible':
                solution = self._solve_incompressible(problem, params)
            elif flow_type == 'compressible':
                solution = self._solve_compressible(problem, params)
            elif flow_type == 'turbulent':
                solution = self._solve_turbulent(problem, params)
            else:
                raise ValueError(f"Unknown flow type: {flow_type}")
            
            validation_errors = self.validate_solution(solution, problem)
            
            constraints_satisfied = {}
            for constraint in self.constraints:
                violation = constraint.validate(solution, problem)
                constraints_satisfied[constraint.name] = not violation.violated
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                success=len(validation_errors) == 0,
                domain=self.config.name,
                solution=solution,
                metadata={
                    "flow_type": flow_type,
                    "reynolds_number": self._compute_reynolds_number(problem, params),
                    "parameters": params
                },
                constraints_satisfied=constraints_satisfied,
                validation_errors=validation_errors,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Fluid dynamics solve failed: {e}")
            return SimulationResult(
                success=False,
                domain=self.config.name,
                solution=None,
                metadata={"error": str(e)},
                constraints_satisfied={},
                validation_errors=[str(e)],
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _compute_reynolds_number(self, problem: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Compute Reynolds number for the flow."""
        velocity_scale = problem.get('velocity_scale', 1.0)
        length_scale = problem.get('length_scale', 1.0)
        nu = params.get('kinematic_viscosity', 1e-5)
        return velocity_scale * length_scale / nu

    def _solve_incompressible(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve incompressible flow (pressure-velocity coupling)."""
        solution = {
            "type": "incompressible",
            "velocity_field": None,
            "pressure_field": None,
            "stream_function": None,
            "vorticity_field": None
        }
        return solution

    def _solve_compressible(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve compressible flow."""
        solution = {
            "type": "compressible",
            "velocity_field": None,
            "pressure_field": None,
            "density_field": None,
            "temperature_field": None,
            "mach_number": None
        }
        return solution

    def _solve_turbulent(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve turbulent flow."""
        solution = {
            "type": "turbulent",
            "velocity_field": None,
            "pressure_field": None,
            "turbulent_kinetic_energy": None,
            "dissipation_rate": None,
            "eddy_viscosity": None
        }
        return solution

    def validate_solution(self, solution: Any, context: Dict[str, Any]) -> List[str]:
        """Validate fluid dynamics solution."""
        errors = []
        
        if solution is None:
            errors.append("Solution is None")
            return errors
        
        # Check for negative density in compressible flow
        density = solution.get('density_field') if isinstance(solution, dict) else None
        if density is not None:
            if TORCH_AVAILABLE and isinstance(density, torch.Tensor):
                density = density.detach().cpu().numpy()
            if np.any(density < 0):
                errors.append("Negative density detected")
        
        return errors


class StructuralMechanics(ScientificDomain):
    """Structural mechanics and stress analysis domain."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        config = DomainConfig(
            name="StructuralMechanics",
            domain_type=DomainType.STRUCTURAL_MECHANICS,
            default_parameters={
                "youngs_modulus": 200e9,  # Pa (steel)
                "poisson_ratio": 0.3,
                "density": 7850,  # kg/m³ (steel)
                "yield_strength": 250e6,  # Pa
                "shear_modulus": 79e9,  # Pa
            },
            typical_constraints=[
                "newton_second_law",
                "equilibrium",
                "hooke_law"
            ],
            validation_rules=[
                "stress_below_yield",
                "strain_compatibility",
                "displacement_continuity"
            ],
            units={
                "stress": "Pa",
                "strain": "dimensionless",
                "displacement": "m",
                "force": "N"
            },
            characteristic_scales={
                "stress": 1e8,  # 100 MPa
                "strain": 0.001,
                "displacement": 0.01,  # 1 cm
                "length": 1.0,  # m
                "force": 1e6  # MN
            },
            numerical_settings={
                "element_type": "tetrahedral",
                "element_order": 2,
                "num_gauss_points": 4
            }
        )
        
        if custom_config:
            config.default_parameters.update(custom_config.get('parameters', {}))
            config.numerical_settings.update(custom_config.get('numerical', {}))
        
        super().__init__(config)

    def _setup_constraints(self):
        """Set up structural mechanics constraints."""
        # Force equilibrium
        equilibrium_config = ConstraintConfig(weight=1.0, tolerance=1e-8)
        self.constraints.append(ConstraintLibrary.equilibrium(equilibrium_config))
        
        # Constitutive relation (Hooke's law)
        hooke_config = ConstraintConfig(weight=1.0, tolerance=1e-6)
        self.constraints.append(ConstraintLibrary.hooke_law(hooke_config))
        
        # Newton's laws
        newton_config = ConstraintConfig(weight=1.0, tolerance=1e-8)
        self.constraints.append(ConstraintLibrary.newtons_second_law(newton_config))

    def solve(
        self,
        problem: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """Solve structural mechanics problem."""
        from datetime import datetime, timezone
        import time
        
        start_time = time.time()
        params = {**self.config.default_parameters, **(parameters or {})}
        
        analysis_type = problem.get('analysis_type', 'static')
        
        try:
            if analysis_type == 'static':
                solution = self._solve_static(problem, params)
            elif analysis_type == 'dynamic':
                solution = self._solve_dynamic(problem, params)
            elif analysis_type == 'modal':
                solution = self._solve_modal(problem, params)
            elif analysis_type == 'buckling':
                solution = self._solve_buckling(problem, params)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            validation_errors = self.validate_solution(solution, problem)
            
            constraints_satisfied = {}
            for constraint in self.constraints:
                violation = constraint.validate(solution, problem)
                constraints_satisfied[constraint.name] = not violation.violated
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                success=len(validation_errors) == 0,
                domain=self.config.name,
                solution=solution,
                metadata={
                    "analysis_type": analysis_type,
                    "safety_factor": self._compute_safety_factor(solution, params),
                    "parameters": params
                },
                constraints_satisfied=constraints_satisfied,
                validation_errors=validation_errors,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Structural mechanics solve failed: {e}")
            return SimulationResult(
                success=False,
                domain=self.config.name,
                solution=None,
                metadata={"error": str(e)},
                constraints_satisfied={},
                validation_errors=[str(e)],
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _solve_static(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve static structural analysis."""
        solution = {
            "type": "static",
            "displacement_field": None,
            "stress_field": None,
            "strain_field": None,
            "reaction_forces": None,
            "max_stress": None,
            "max_displacement": None
        }
        return solution

    def _solve_dynamic(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve dynamic structural analysis."""
        solution = {
            "type": "dynamic",
            "displacement_history": None,
            "velocity_history": None,
            "acceleration_history": None,
            "stress_history": None,
            "natural_frequencies": None
        }
        return solution

    def _solve_modal(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve modal analysis (natural frequencies/modes)."""
        solution = {
            "type": "modal",
            "natural_frequencies": None,
            "mode_shapes": None,
            "modal_masses": None,
            "participation_factors": None
        }
        return solution

    def _solve_buckling(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve buckling analysis."""
        solution = {
            "type": "buckling",
            "critical_loads": None,
            "buckling_modes": None,
            "safety_margin": None
        }
        return solution

    def _compute_safety_factor(self, solution: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Compute safety factor."""
        max_stress = solution.get('max_stress', 0)
        yield_strength = params.get('yield_strength', 250e6)
        if max_stress > 0:
            return yield_strength / max_stress
        return float('inf')

    def validate_solution(self, solution: Any, context: Dict[str, Any]) -> List[str]:
        """Validate structural mechanics solution."""
        errors = []
        
        if solution is None:
            errors.append("Solution is None")
            return errors
        
        if isinstance(solution, dict):
            max_stress = solution.get('max_stress')
            yield_strength = context.get('yield_strength', 250e6)
            
            if max_stress is not None and max_stress > yield_strength:
                errors.append(f"Maximum stress ({max_stress:.2e} Pa) exceeds yield strength")
        
        return errors


class ChemicalKinetics(ScientificDomain):
    """Chemical reaction kinetics domain."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        config = DomainConfig(
            name="ChemicalKinetics",
            domain_type=DomainType.CHEMICAL_KINETICS,
            default_parameters={
                "universal_gas_constant": 8.314,  # J/(mol·K)
                "avogadro_number": 6.022e23,  # 1/mol
                "boltzmann_constant": 1.381e-23,  # J/K
                "faraday_constant": 96485,  # C/mol
            },
            typical_constraints=[
                "mass_action_kinetics",
                "atom_conservation",
                "chemical_equilibrium"
            ],
            validation_rules=[
                "concentration_positive",
                "reaction_rates_physical",
                "mass_balance"
            ],
            units={
                "concentration": "mol/m³",
                "reaction_rate": "mol/(m³·s)",
                "temperature": "K",
                "activation_energy": "J/mol"
            },
            characteristic_scales={
                "concentration": 1.0,  # mol/m³
                "time": 1.0,  # s
                "temperature": 300,  # K
                "activation_energy": 50000  # J/mol
            },
            numerical_settings={
                "integrator": "cvode",
                "relative_tolerance": 1e-6,
                "absolute_tolerance": 1e-10
            }
        )
        
        if custom_config:
            config.default_parameters.update(custom_config.get('parameters', {}))
            config.numerical_settings.update(custom_config.get('numerical', {}))
        
        super().__init__(config)

    def _setup_constraints(self):
        """Set up chemical kinetics constraints."""
        # Mass action kinetics
        mass_action_config = ConstraintConfig(weight=1.0, tolerance=1e-6)
        self.constraints.append(ConstraintLibrary.mass_action_kinetics(mass_action_config))
        
        # Atom conservation
        atom_config = ConstraintConfig(weight=1.0, tolerance=1e-8)
        self.constraints.append(ConstraintLibrary.atom_conservation(atom_config))
        
        # Chemical equilibrium
        equilibrium_config = ConstraintConfig(weight=0.5, tolerance=1e-4)
        self.constraints.append(ConstraintLibrary.chemical_equilibrium(equilibrium_config))

    def solve(
        self,
        problem: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """Solve chemical kinetics problem."""
        from datetime import datetime, timezone
        import time
        
        start_time = time.time()
        params = {**self.config.default_parameters, **(parameters or {})}
        
        problem_type = problem.get('type', 'transient')
        
        try:
            if problem_type == 'transient':
                solution = self._solve_transient(problem, params)
            elif problem_type == 'steady_state':
                solution = self._solve_steady_state(problem, params)
            elif problem_type == 'sensitivity':
                solution = self._solve_sensitivity(problem, params)
            else:
                raise ValueError(f"Unknown problem type: {problem_type}")
            
            validation_errors = self.validate_solution(solution, problem)
            
            constraints_satisfied = {}
            for constraint in self.constraints:
                violation = constraint.validate(solution, problem)
                constraints_satisfied[constraint.name] = not violation.violated
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                success=len(validation_errors) == 0,
                domain=self.config.name,
                solution=solution,
                metadata={
                    "problem_type": problem_type,
                    "reaction_extent": self._compute_reaction_extent(solution),
                    "parameters": params
                },
                constraints_satisfied=constraints_satisfied,
                validation_errors=validation_errors,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Chemical kinetics solve failed: {e}")
            return SimulationResult(
                success=False,
                domain=self.config.name,
                solution=None,
                metadata={"error": str(e)},
                constraints_satisfied={},
                validation_errors=[str(e)],
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _solve_transient(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve transient chemical kinetics."""
        solution = {
            "type": "transient",
            "concentration_history": None,
            "reaction_rates_history": None,
            "time_points": None,
            "final_conversion": None
        }
        return solution

    def _solve_steady_state(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve for steady-state concentrations."""
        solution = {
            "type": "steady_state",
            "concentrations": None,
            "reaction_rates": None,
            "equilibrium_constants": None
        }
        return solution

    def _solve_sensitivity(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve sensitivity analysis."""
        solution = {
            "type": "sensitivity",
            "sensitivity_matrix": None,
            "important_parameters": None,
            "local_sensitivities": None
        }
        return solution

    def _compute_reaction_extent(self, solution: Dict[str, Any]) -> float:
        """Compute overall reaction extent."""
        # Placeholder implementation
        return 0.5

    def validate_solution(self, solution: Any, context: Dict[str, Any]) -> List[str]:
        """Validate chemical kinetics solution."""
        errors = []
        
        if solution is None:
            errors.append("Solution is None")
            return errors
        
        # Check for negative concentrations
        if isinstance(solution, dict):
            conc = solution.get('concentrations') or solution.get('final_concentrations')
            if conc is not None:
                if TORCH_AVAILABLE and isinstance(conc, torch.Tensor):
                    conc = conc.detach().cpu().numpy()
                if isinstance(conc, np.ndarray) and np.any(conc < 0):
                    errors.append("Negative concentration detected")
        
        return errors


class BiologicalSystems(ScientificDomain):
    """Biological systems domain (population dynamics, epidemiology)."""

    def __init__(self, custom_config: Optional[Dict[str, Any]] = None):
        config = DomainConfig(
            name="BiologicalSystems",
            domain_type=DomainType.BIOLOGICAL_SYSTEMS,
            default_parameters={
                "carrying_capacity": 1000,
                "growth_rate": 0.1,
                "death_rate": 0.05,
                "interaction_coefficient": 0.001,
            },
            typical_constraints=[
                "population_positive",
                "carrying_capacity_limit",
                "mass_conservation"
            ],
            validation_rules=[
                "population_non_negative",
                "extinction_realistic",
                "equilibrium_stability"
            ],
            units={
                "population": "individuals",
                "rate": "1/time",
                "time": "days"
            },
            characteristic_scales={
                "population": 1000,
                "time": 30,  # days
                "rate": 0.1  # per day
            },
            numerical_settings={
                "time_step": 0.1,
                "integration_method": "rk4",
                "population_threshold": 1e-6
            }
        )
        
        if custom_config:
            config.default_parameters.update(custom_config.get('parameters', {}))
            config.numerical_settings.update(custom_config.get('numerical', {}))
        
        super().__init__(config)

    def _setup_constraints(self):
        """Set up biological system constraints."""
        # Mass conservation (total biomass)
        mass_config = ConstraintConfig(weight=0.8, tolerance=1e-6)
        self.constraints.append(ConstraintLibrary.conservation_of_mass(mass_config))

    def solve(
        self,
        problem: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> SimulationResult:
        """Solve biological system problem."""
        from datetime import datetime, timezone
        import time
        
        start_time = time.time()
        params = {**self.config.default_parameters, **(parameters or {})}
        
        model_type = problem.get('model_type', 'logistic')
        
        try:
            if model_type == 'logistic':
                solution = self._solve_logistic(problem, params)
            elif model_type == 'lotka_volterra':
                solution = self._solve_lotka_volterra(problem, params)
            elif model_type == 'sir':
                solution = self._solve_sir(problem, params)
            elif model_type == 'seir':
                solution = self._solve_seir(problem, params)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            validation_errors = self.validate_solution(solution, problem)
            
            constraints_satisfied = {}
            for constraint in self.constraints:
                violation = constraint.validate(solution, problem)
                constraints_satisfied[constraint.name] = not violation.violated
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                success=len(validation_errors) == 0,
                domain=self.config.name,
                solution=solution,
                metadata={
                    "model_type": model_type,
                    "basic_reproduction_number": self._compute_r0(solution, params, model_type),
                    "parameters": params
                },
                constraints_satisfied=constraints_satisfied,
                validation_errors=validation_errors,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Biological systems solve failed: {e}")
            return SimulationResult(
                success=False,
                domain=self.config.name,
                solution=None,
                metadata={"error": str(e)},
                constraints_satisfied={},
                validation_errors=[str(e)],
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _solve_logistic(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve logistic growth model."""
        K = params.get('carrying_capacity', 1000)
        r = params.get('growth_rate', 0.1)
        N0 = problem.get('initial_population', 100)
        
        solution = {
            "type": "logistic",
            "carrying_capacity": K,
            "growth_rate": r,
            "initial_population": N0,
            "population_trajectory": None,
            "equilibrium_population": K,
            "doubling_time": np.log(2) / r if r > 0 else None
        }
        return solution

    def _solve_lotka_volterra(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve Lotka-Volterra predator-prey model."""
        solution = {
            "type": "lotka_volterra",
            "prey_population": None,
            "predator_population": None,
            "phase_portrait": None,
            "equilibrium_points": None,
            "period": None
        }
        return solution

    def _solve_sir(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve SIR epidemiological model."""
        solution = {
            "type": "sir",
            "susceptible": None,
            "infected": None,
            "recovered": None,
            "peak_infection": None,
            "final_epidemic_size": None,
            "herd_immunity_threshold": None
        }
        return solution

    def _solve_seir(self, problem: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Solve SEIR epidemiological model."""
        solution = {
            "type": "seir",
            "susceptible": None,
            "exposed": None,
            "infected": None,
            "recovered": None,
            "incubation_period": params.get('incubation_period', 5),
            "infectious_period": params.get('infectious_period', 7)
        }
        return solution

    def _compute_r0(self, solution: Dict[str, Any], params: Dict[str, Any], model_type: str) -> Optional[float]:
        """Compute basic reproduction number."""
        if model_type in ['sir', 'seir']:
            beta = params.get('transmission_rate', 0.5)
            gamma = params.get('recovery_rate', 0.1)
            return beta / gamma if gamma > 0 else None
        return None

    def validate_solution(self, solution: Any, context: Dict[str, Any]) -> List[str]:
        """Validate biological system solution."""
        errors = []
        
        if solution is None:
            errors.append("Solution is None")
            return errors
        
        # Check for negative populations
        if isinstance(solution, dict):
            for key in ['population_trajectory', 'prey_population', 'predator_population',
                       'susceptible', 'infected', 'recovered', 'exposed']:
                pop = solution.get(key)
                if pop is not None:
                    if TORCH_AVAILABLE and isinstance(pop, torch.Tensor):
                        pop = pop.detach().cpu().numpy()
                    if isinstance(pop, np.ndarray) and np.any(pop < -1e-10):
                        errors.append(f"Negative population in {key}")
        
        return errors


class DomainLibrary:
    """Library of pre-configured scientific domains."""

    @staticmethod
    def climate_modeling(config: Optional[Dict[str, Any]] = None) -> ClimateModeling:
        """Create climate modeling domain."""
        return ClimateModeling(config)

    @staticmethod
    def fluid_dynamics(config: Optional[Dict[str, Any]] = None) -> FluidDynamics:
        """Create fluid dynamics domain."""
        return FluidDynamics(config)

    @staticmethod
    def structural_mechanics(config: Optional[Dict[str, Any]] = None) -> StructuralMechanics:
        """Create structural mechanics domain."""
        return StructuralMechanics(config)

    @staticmethod
    def chemical_kinetics(config: Optional[Dict[str, Any]] = None) -> ChemicalKinetics:
        """Create chemical kinetics domain."""
        return ChemicalKinetics(config)

    @staticmethod
    def biological_systems(config: Optional[Dict[str, Any]] = None) -> BiologicalSystems:
        """Create biological systems domain."""
        return BiologicalSystems(config)

    @staticmethod
    def get_all_domains() -> Dict[str, ScientificDomain]:
        """Get all available domains."""
        return {
            "climate": DomainLibrary.climate_modeling(),
            "fluid_dynamics": DomainLibrary.fluid_dynamics(),
            "structural_mechanics": DomainLibrary.structural_mechanics(),
            "chemical_kinetics": DomainLibrary.chemical_kinetics(),
            "biological_systems": DomainLibrary.biological_systems()
        }


__all__ = [
    "DomainType",
    "DomainConfig",
    "SimulationResult",
    "ScientificDomain",
    "ClimateModeling",
    "FluidDynamics",
    "StructuralMechanics",
    "ChemicalKinetics",
    "BiologicalSystems",
    "DomainLibrary"
]
