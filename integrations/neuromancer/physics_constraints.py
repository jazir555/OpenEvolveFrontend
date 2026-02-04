"""Physics constraint definitions for knowledge graphs.

Common physical laws as differentiable constraints for neural operators.
Enables physics-informed machine learning with enforceable conservation laws.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
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

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of physics constraints."""
    CONSERVATION = "conservation"
    THERMODYNAMIC = "thermodynamic"
    MECHANICAL = "mechanical"
    ELECTROMAGNETIC = "electromagnetic"
    CHEMICAL = "chemical"
    BOUNDARY = "boundary"
    INITIAL = "initial"


class ConservationQuantity(Enum):
    """Physical quantities subject to conservation laws."""
    MASS = "mass"
    ENERGY = "energy"
    MOMENTUM = "momentum"
    ANGULAR_MOMENTUM = "angular_momentum"
    CHARGE = "charge"


@dataclass
class ConstraintViolation:
    """Result of constraint validation."""
    constraint_name: str
    violated: bool
    violation_magnitude: float
    violation_relative: float
    details: Dict[str, Any]
    timestamp: str


@dataclass
class ConstraintConfig:
    """Configuration for physics constraints."""
    weight: float = 1.0
    tolerance: float = 1e-6
    hard_constraint: bool = False
    penalty_type: str = "l2"  # "l1", "l2", "huber"
    normalize: bool = True
    device: str = "cpu"


class PhysicsConstraint(ABC):
    """Base class for physics constraints."""

    def __init__(self, name: str, constraint_type: ConstraintType, config: Optional[ConstraintConfig] = None):
        self.name = name
        self.constraint_type = constraint_type
        self.config = config or ConstraintConfig()
        self._device = self.config.device
        logger.info(f"Initialized {constraint_type.value} constraint: {name}")

    @abstractmethod
    def compute_loss(self, predictions: Any, context: Dict[str, Any]) -> Union[float, torch.Tensor]:
        """Compute differentiable constraint loss.
        
        Args:
            predictions: Model predictions
            context: Additional context (spatial coordinates, time, etc.)
            
        Returns:
            Constraint loss value (scalar)
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, solution: Any, context: Dict[str, Any]) -> ConstraintViolation:
        """Validate solution against constraint.
        
        Args:
            solution: Solution to validate
            context: Validation context
            
        Returns:
            ConstraintViolation with validation results
        """
        raise NotImplementedError

    def to_device(self, device: str):
        """Move constraint to device."""
        self._device = device
        return self

    def _ensure_tensor(self, data: Any, dtype=torch.float32) -> torch.Tensor:
        """Convert data to tensor on correct device."""
        if not TORCH_AVAILABLE:
            return data
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        return torch.tensor(data, dtype=dtype, device=self._device)


class ConservationLawConstraint(PhysicsConstraint):
    """Conservation law constraints (mass, energy, momentum)."""

    def __init__(
        self,
        quantity: ConservationQuantity,
        domain_volume: Optional[float] = None,
        config: Optional[ConstraintConfig] = None
    ):
        super().__init__(
            name=f"{quantity.value}_conservation",
            constraint_type=ConstraintType.CONSERVATION,
            config=config
        )
        self.quantity = quantity
        self.domain_volume = domain_volume or 1.0

    def compute_loss(self, predictions: Any, context: Dict[str, Any]) -> Union[float, torch.Tensor]:
        """Compute conservation law violation loss.
        
        For time-dependent problems, enforces d(quantity)/dt = 0 (or source/sink terms).
        For steady-state, enforces divergence = 0.
        """
        if not TORCH_AVAILABLE:
            return 0.0

        # Extract quantity from predictions
        quantity_field = self._ensure_tensor(predictions)
        
        # Get time or spatial coordinates
        coords = context.get('coordinates')
        if coords is None:
            logger.warning("No coordinates provided for conservation constraint")
            return torch.tensor(0.0, device=self._device)
        
        coords = self._ensure_tensor(coords)
        
        # Compute time derivative or divergence
        time_coord = context.get('time')
        if time_coord is not None:
            # Time-dependent: compute d(quantity)/dt
            loss = self._compute_time_derivative_loss(quantity_field, time_coord, context)
        else:
            # Steady-state: compute divergence
            loss = self._compute_divergence_loss(quantity_field, coords, context)
        
        # Apply weight
        loss = loss * self.config.weight
        
        # Apply penalty type
        if self.config.penalty_type == "l1":
            loss = torch.abs(loss)
        elif self.config.penalty_type == "huber":
            delta = 0.1
            loss = torch.where(
                torch.abs(loss) < delta,
                0.5 * loss ** 2 / delta,
                torch.abs(loss) - 0.5 * delta
            )
        else:  # l2
            loss = loss ** 2
            
        return loss.mean()

    def _compute_time_derivative_loss(
        self,
        quantity: torch.Tensor,
        time: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute time derivative of conserved quantity."""
        # Compute gradient with respect to time
        quantity_flat = quantity.reshape(-1)
        time_flat = time.reshape(-1)
        
        # Sort by time for proper finite differences
        sorted_indices = torch.argsort(time_flat)
        quantity_sorted = quantity_flat[sorted_indices]
        time_sorted = time_flat[sorted_indices]
        
        # Compute time derivative using finite differences
        dt = time_sorted[1:] - time_sorted[:-1]
        dquantity = quantity_sorted[1:] - quantity_sorted[:-1]
        
        # Avoid division by zero
        dt = torch.clamp(dt, min=1e-10)
        time_derivative = dquantity / dt
        
        # Expected change from sources/sinks
        source_term = context.get('source_term', torch.zeros_like(time_derivative))
        
        # Conservation: d(quantity)/dt = source_term
        violation = time_derivative - source_term.to(self._device)
        
        return violation

    def _compute_divergence_loss(
        self,
        quantity: torch.Tensor,
        coords: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute divergence of flux for steady-state conservation."""
        # For scalar fields, compute gradient divergence
        # For vector fields (like momentum), compute div(vector)
        
        if quantity.dim() == 1:
            # Scalar field - compute Laplacian as proxy for divergence
            # Use autograd for second derivatives
            coords.requires_grad_(True)
            quantity.requires_grad_(True)
            
            # Compute gradient
            grad = torch.autograd.grad(
                quantity.sum(),
                coords,
                create_graph=True
            )[0]
            
            # Compute divergence of gradient (Laplacian)
            divergence = torch.zeros(quantity.shape[0], device=self._device)
            for i in range(coords.shape[-1]):
                div_component = torch.autograd.grad(
                    grad[:, i].sum(),
                    coords,
                    create_graph=True,
                    retain_graph=True
                )[0][:, i]
                divergence = divergence + div_component
        else:
            # Vector field - compute divergence directly
            divergence = torch.zeros(quantity.shape[0], device=self._device)
            for i in range(quantity.shape[-1]):
                coords.requires_grad_(True)
                grad = torch.autograd.grad(
                    quantity[:, i].sum(),
                    coords,
                    create_graph=True
                )[0]
                divergence = divergence + grad[:, i]
        
        # Source/sink term
        source = context.get('source_term', torch.zeros_like(divergence))
        
        return divergence - source.to(self._device)

    def validate(self, solution: Any, context: Dict[str, Any]) -> ConstraintViolation:
        """Validate conservation law satisfaction."""
        from datetime import datetime, timezone
        
        loss = self.compute_loss(solution, context)
        
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            loss_value = loss.detach().cpu().item()
        else:
            loss_value = float(loss)
        
        # Compute relative violation
        solution_mag = np.mean(np.abs(solution)) if isinstance(solution, np.ndarray) else 1.0
        relative_violation = loss_value / (solution_mag + 1e-10)
        
        violated = relative_violation > self.config.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            violated=violated,
            violation_magnitude=loss_value,
            violation_relative=relative_violation,
            details={
                "quantity": self.quantity.value,
                "tolerance": self.config.tolerance,
                "domain_volume": self.domain_volume
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class ThermodynamicConstraint(PhysicsConstraint):
    """Thermodynamic constraints (entropy, temperature relationships)."""

    def __init__(
        self,
        constraint_name: str,
        constraint_type: str = "entropy_production",
        config: Optional[ConstraintConfig] = None
    ):
        super().__init__(
            name=constraint_name,
            constraint_type=ConstraintType.THERMODYNAMIC,
            config=config
        )
        self.thermo_type = constraint_type

    def compute_loss(self, predictions: Any, context: Dict[str, Any]) -> Union[float, torch.Tensor]:
        """Compute thermodynamic constraint loss."""
        if not TORCH_AVAILABLE:
            return 0.0

        if self.thermo_type == "entropy_production":
            return self._entropy_production_loss(predictions, context)
        elif self.thermo_type == "temperature_positive":
            return self._temperature_positive_loss(predictions, context)
        elif self.thermo_type == "second_law":
            return self._second_law_loss(predictions, context)
        else:
            logger.warning(f"Unknown thermodynamic constraint type: {self.thermo_type}")
            return torch.tensor(0.0, device=self._device)

    def _entropy_production_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Entropy must be produced (not destroyed) in irreversible processes."""
        # For time-dependent problems
        time = context.get('time')
        if time is None:
            return torch.tensor(0.0, device=self._device)
        
        entropy = self._ensure_tensor(predictions)
        time_tensor = self._ensure_tensor(time)
        
        # Compute entropy rate of change
        sorted_indices = torch.argsort(time_tensor.reshape(-1))
        entropy_sorted = entropy.reshape(-1)[sorted_indices]
        time_sorted = time_tensor.reshape(-1)[sorted_indices]
        
        dt = time_sorted[1:] - time_sorted[:-1]
        dS = entropy_sorted[1:] - entropy_sorted[:-1]
        
        dt = torch.clamp(dt, min=1e-10)
        dS_dt = dS / dt
        
        # Entropy production principle: dS/dt >= 0 for isolated systems
        # Penalize negative entropy production (entropy destruction)
        violation = torch.relu(-dS_dt)
        
        return (violation ** 2).mean() * self.config.weight

    def _temperature_positive_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Temperature must be positive (absolute temperature)."""
        temperature = self._ensure_tensor(predictions)
        
        # Penalize negative or near-zero temperatures
        violation = torch.relu(-temperature + 0.01)  # Soft constraint at 0.01K
        
        return (violation ** 2).mean() * self.config.weight

    def _second_law_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Second law of thermodynamics (heat flows from hot to cold)."""
        temperature = self._ensure_tensor(predictions)
        heat_flux = context.get('heat_flux')
        
        if heat_flux is None:
            return torch.tensor(0.0, device=self._device)
        
        heat_flux = self._ensure_tensor(heat_flux)
        
        # Fourier's law: heat flux proportional to negative temperature gradient
        # Heat flows from high T to low T (opposite to gradient)
        coords = context.get('coordinates')
        if coords is None:
            return torch.tensor(0.0, device=self._device)
        
        coords = self._ensure_tensor(coords)
        coords.requires_grad_(True)
        
        # Compute temperature gradient
        grad_T = torch.autograd.grad(
            temperature.sum(),
            coords,
            create_graph=True
        )[0]
        
        # Heat flux should be opposite to temperature gradient
        # q = -k * grad(T), so q * grad(T) should be negative
        dot_product = (heat_flux * grad_T).sum(dim=-1)
        
        # Penalize positive values (heat flowing against gradient)
        violation = torch.relu(dot_product)
        
        return (violation ** 2).mean() * self.config.weight

    def validate(self, solution: Any, context: Dict[str, Any]) -> ConstraintViolation:
        """Validate thermodynamic constraint satisfaction."""
        from datetime import datetime, timezone
        
        loss = self.compute_loss(solution, context)
        
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            loss_value = loss.detach().cpu().item()
        else:
            loss_value = float(loss)
        
        violated = loss_value > self.config.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            violated=violated,
            violation_magnitude=loss_value,
            violation_relative=loss_value / (self.config.tolerance + 1e-10),
            details={
                "thermodynamic_type": self.thermo_type,
                "tolerance": self.config.tolerance
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class MechanicalConstraint(PhysicsConstraint):
    """Mechanical constraints (Newton's laws, Hooke's law)."""

    def __init__(
        self,
        constraint_name: str,
        constraint_type: str = "newton_second_law",
        config: Optional[ConstraintConfig] = None
    ):
        super().__init__(
            name=constraint_name,
            constraint_type=ConstraintType.MECHANICAL,
            config=config
        )
        self.mechanical_type = constraint_type

    def compute_loss(self, predictions: Any, context: Dict[str, Any]) -> Union[float, torch.Tensor]:
        """Compute mechanical constraint loss."""
        if not TORCH_AVAILABLE:
            return 0.0

        if self.mechanical_type == "newton_second_law":
            return self._newton_second_law_loss(predictions, context)
        elif self.mechanical_type == "hooke_law":
            return self._hooke_law_loss(predictions, context)
        elif self.mechanical_type == "equilibrium":
            return self._equilibrium_loss(predictions, context)
        else:
            logger.warning(f"Unknown mechanical constraint type: {self.mechanical_type}")
            return torch.tensor(0.0, device=self._device)

    def _newton_second_law_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """F = ma constraint."""
        # predictions: acceleration field
        # context['force']: force field
        # context['mass']: mass field
        
        acceleration = self._ensure_tensor(predictions)
        force = self._ensure_tensor(context.get('force', torch.zeros_like(acceleration)))
        mass = self._ensure_tensor(context.get('mass', torch.ones(acceleration.shape[0])))
        
        # Newton's second law: F = ma
        expected_force = mass.unsqueeze(-1) * acceleration if acceleration.dim() > mass.dim() else mass * acceleration
        
        violation = force - expected_force
        
        return (violation ** 2).mean() * self.config.weight

    def _hooke_law_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """F = -kx constraint for springs."""
        displacement = self._ensure_tensor(predictions)
        force = self._ensure_tensor(context.get('force', torch.zeros_like(displacement)))
        spring_constant = context.get('spring_constant', 1.0)
        
        # Hooke's law: F = -k * x
        expected_force = -spring_constant * displacement
        
        violation = force - expected_force
        
        return (violation ** 2).mean() * self.config.weight

    def _equilibrium_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Sum of forces = 0 at equilibrium."""
        # predictions: displacement field
        # Sum of all forces should be zero
        
        forces = context.get('forces')
        if forces is None:
            return torch.tensor(0.0, device=self._device)
        
        # Sum all forces
        total_force = sum(self._ensure_tensor(f) for f in forces)
        
        # At equilibrium, total force should be zero
        violation = total_force
        
        return (violation ** 2).mean() * self.config.weight

    def validate(self, solution: Any, context: Dict[str, Any]) -> ConstraintViolation:
        """Validate mechanical constraint satisfaction."""
        from datetime import datetime, timezone
        
        loss = self.compute_loss(solution, context)
        
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            loss_value = loss.detach().cpu().item()
        else:
            loss_value = float(loss)
        
        violated = loss_value > self.config.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            violated=violated,
            violation_magnitude=loss_value,
            violation_relative=loss_value / (self.config.tolerance + 1e-10),
            details={
                "mechanical_type": self.mechanical_type,
                "tolerance": self.config.tolerance
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class ElectromagneticConstraint(PhysicsConstraint):
    """Electromagnetic constraints (Maxwell's equations)."""

    def __init__(
        self,
        constraint_name: str,
        constraint_type: str = "gauss_law",
        config: Optional[ConstraintConfig] = None
    ):
        super().__init__(
            name=constraint_name,
            constraint_type=ConstraintType.ELECTROMAGNETIC,
            config=config
        )
        self.em_type = constraint_type

    def compute_loss(self, predictions: Any, context: Dict[str, Any]) -> Union[float, torch.Tensor]:
        """Compute electromagnetic constraint loss."""
        if not TORCH_AVAILABLE:
            return 0.0

        if self.em_type == "gauss_law":
            return self._gauss_law_loss(predictions, context)
        elif self.em_type == "faraday_law":
            return self._faraday_law_loss(predictions, context)
        elif self.em_type == "ampere_law":
            return self._ampere_law_loss(predictions, context)
        elif self.em_type == "no_magnetic_monopoles":
            return self._no_monopoles_loss(predictions, context)
        else:
            logger.warning(f"Unknown EM constraint type: {self.em_type}")
            return torch.tensor(0.0, device=self._device)

    def _gauss_law_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """∇·E = ρ/ε₀ (Gauss's law for electricity)."""
        electric_field = self._ensure_tensor(predictions)
        charge_density = self._ensure_tensor(context.get('charge_density', torch.zeros(electric_field.shape[0])))
        epsilon_0 = context.get('epsilon_0', 8.854e-12)
        
        coords = context.get('coordinates')
        if coords is None:
            return torch.tensor(0.0, device=self._device)
        
        coords = self._ensure_tensor(coords)
        coords.requires_grad_(True)
        
        # Compute divergence of E
        divergence = torch.zeros(electric_field.shape[0], device=self._device)
        for i in range(electric_field.shape[-1]):
            grad = torch.autograd.grad(
                electric_field[:, i].sum(),
                coords,
                create_graph=True,
                retain_graph=True
            )[0]
            divergence = divergence + grad[:, i]
        
        # Expected divergence: ρ/ε₀
        expected_div = charge_density / epsilon_0
        
        violation = divergence - expected_div
        
        return (violation ** 2).mean() * self.config.weight

    def _faraday_law_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """∇×E = -∂B/∂t (Faraday's law of induction)."""
        # Simplified implementation for 2D/3D
        electric_field = self._ensure_tensor(predictions)
        magnetic_field = self._ensure_tensor(context.get('magnetic_field', torch.zeros_like(electric_field)))
        time = context.get('time')
        
        coords = context.get('coordinates')
        if coords is None or time is None:
            return torch.tensor(0.0, device=self._device)
        
        coords = self._ensure_tensor(coords)
        time = self._ensure_tensor(time)
        
        # Compute curl of E (simplified for 2D)
        # ∇×E = (∂Ez/∂y - ∂Ey/∂z, ∂Ex/∂z - ∂Ez/∂x, ∂Ey/∂x - ∂Ex/∂y)
        
        # Compute dB/dt
        sorted_indices = torch.argsort(time.reshape(-1))
        B_sorted = magnetic_field.reshape(-1, magnetic_field.shape[-1])[sorted_indices]
        t_sorted = time.reshape(-1)[sorted_indices]
        
        dt = t_sorted[1:] - t_sorted[:-1]
        dB = B_sorted[1:] - B_sorted[:-1]
        dt = torch.clamp(dt, min=1e-10).unsqueeze(-1)
        dB_dt = dB / dt
        
        # For simplified case, assume curl is approximated
        # Full curl computation would require spatial derivatives
        violation = dB_dt  # Simplified: penalize any change in B
        
        return (violation ** 2).mean() * self.config.weight

    def _ampere_law_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """∇×B = μ₀J + μ₀ε₀∂E/∂t (Ampère-Maxwell law)."""
        magnetic_field = self._ensure_tensor(predictions)
        current_density = context.get('current_density')
        
        # Simplified version
        if current_density is not None:
            current_density = self._ensure_tensor(current_density)
            mu_0 = context.get('mu_0', 4 * np.pi * 1e-7)
            expected = mu_0 * current_density
            violation = magnetic_field - expected
            return (violation ** 2).mean() * self.config.weight
        
        return torch.tensor(0.0, device=self._device)

    def _no_monopoles_loss(self, predictions: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """∇·B = 0 (No magnetic monopoles)."""
        magnetic_field = self._ensure_tensor(predictions)
        
        coords = context.get('coordinates')
        if coords is None:
            return torch.tensor(0.0, device=self._device)
        
        coords = self._ensure_tensor(coords)
        coords.requires_grad_(True)
        
        # Compute divergence of B
        divergence = torch.zeros(magnetic_field.shape[0], device=self._device)
        for i in range(magnetic_field.shape[-1]):
            grad = torch.autograd.grad(
                magnetic_field[:, i].sum(),
                coords,
                create_graph=True,
                retain_graph=True
            )[0]
            divergence = divergence + grad[:, i]
        
        # Should be zero
        violation = divergence
        
        return (violation ** 2).mean() * self.config.weight

    def validate(self, solution: Any, context: Dict[str, Any]) -> ConstraintViolation:
        """Validate electromagnetic constraint satisfaction."""
        from datetime import datetime, timezone
        
        loss = self.compute_loss(solution, context)
        
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            loss_value = loss.detach().cpu().item()
        else:
            loss_value = float(loss)
        
        violated = loss_value > self.config.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            violated=violated,
            violation_magnitude=loss_value,
            violation_relative=loss_value / (self.config.tolerance + 1e-10),
            details={
                "em_type": self.em_type,
                "tolerance": self.config.tolerance
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class ChemicalConstraint(PhysicsConstraint):
    """Chemical constraints (reaction kinetics, equilibrium)."""

    def __init__(
        self,
        constraint_name: str,
        constraint_type: str = "mass_action",
        config: Optional[ConstraintConfig] = None
    ):
        super().__init__(
            name=constraint_name,
            constraint_type=ConstraintType.CHEMICAL,
            config=config
        )
        self.chemical_type = constraint_type

    def compute_loss(self, predictions: Any, context: Dict[str, Any]) -> Union[float, torch.Tensor]:
        """Compute chemical constraint loss."""
        if not TORCH_AVAILABLE:
            return 0.0

        if self.chemical_type == "mass_action":
            return self._mass_action_loss(predictions, context)
        elif self.chemical_type == "equilibrium":
            return self._equilibrium_loss(predictions, context)
        elif self.chemical_type == "conservation_of_atoms":
            return self._atom_conservation_loss(predictions, context)
        else:
            logger.warning(f"Unknown chemical constraint type: {self.chemical_type}")
            return torch.tensor(0.0, device=self._device)

    def _mass_action_loss(self, concentrations: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Mass action kinetics constraint."""
        concentrations = self._ensure_tensor(concentrations)
        reaction_rates = context.get('reaction_rates')
        stoichiometry = context.get('stoichiometry')
        
        if reaction_rates is None or stoichiometry is None:
            return torch.tensor(0.0, device=self._device)
        
        reaction_rates = self._ensure_tensor(reaction_rates)
        stoichiometry = self._ensure_tensor(stoichiometry)
        
        # Compute reaction rates from mass action
        # r = k * [A]^a * [B]^b
        computed_rates = torch.ones_like(reaction_rates)
        for i, stoich in enumerate(stoichiometry):
            if stoich > 0:  # Reactant
                computed_rates = computed_rates * (concentrations[:, i] ** stoich)
        
        violation = reaction_rates - computed_rates
        
        return (violation ** 2).mean() * self.config.weight

    def _equilibrium_loss(self, concentrations: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Chemical equilibrium constraint (Q = K_eq)."""
        concentrations = self._ensure_tensor(concentrations)
        equilibrium_constant = context.get('equilibrium_constant', 1.0)
        
        # Compute reaction quotient Q
        # Q = [products] / [reactants]
        product_conc = concentrations.prod(dim=-1)
        
        Q = product_conc
        
        # At equilibrium: Q = K_eq
        violation = Q - equilibrium_constant
        
        return (violation ** 2).mean() * self.config.weight

    def _atom_conservation_loss(self, concentrations: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Conservation of atoms in reactions."""
        concentrations = self._ensure_tensor(concentrations)
        atom_matrix = context.get('atom_matrix')
        
        if atom_matrix is None:
            return torch.tensor(0.0, device=self._device)
        
        atom_matrix = self._ensure_tensor(atom_matrix)
        
        # Total atoms of each type should be conserved
        # atom_matrix: (n_species, n_elements)
        # concentrations: (n_points, n_species)
        
        total_atoms = torch.matmul(concentrations, atom_matrix)
        
        # Expected total atoms (should be constant)
        expected_atoms = context.get('expected_atoms')
        if expected_atoms is not None:
            expected_atoms = self._ensure_tensor(expected_atoms)
            violation = total_atoms - expected_atoms
        else:
            # Check variance is small (conservation)
            violation = total_atoms.var(dim=0)
        
        return (violation ** 2).mean() * self.config.weight

    def validate(self, solution: Any, context: Dict[str, Any]) -> ConstraintViolation:
        """Validate chemical constraint satisfaction."""
        from datetime import datetime, timezone
        
        loss = self.compute_loss(solution, context)
        
        if TORCH_AVAILABLE and isinstance(loss, torch.Tensor):
            loss_value = loss.detach().cpu().item()
        else:
            loss_value = float(loss)
        
        violated = loss_value > self.config.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            violated=violated,
            violation_magnitude=loss_value,
            violation_relative=loss_value / (self.config.tolerance + 1e-10),
            details={
                "chemical_type": self.chemical_type,
                "tolerance": self.config.tolerance
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )


class ConstraintLibrary:
    """Library of pre-defined physics constraints."""

    @staticmethod
    def conservation_of_mass(config: Optional[ConstraintConfig] = None) -> ConservationLawConstraint:
        """Create mass conservation constraint."""
        return ConservationLawConstraint(
            quantity=ConservationQuantity.MASS,
            config=config
        )

    @staticmethod
    def conservation_of_energy(config: Optional[ConstraintConfig] = None) -> ConservationLawConstraint:
        """Create energy conservation constraint."""
        return ConservationLawConstraint(
            quantity=ConservationQuantity.ENERGY,
            config=config
        )

    @staticmethod
    def conservation_of_momentum(config: Optional[ConstraintConfig] = None) -> ConservationLawConstraint:
        """Create momentum conservation constraint."""
        return ConservationLawConstraint(
            quantity=ConservationQuantity.MOMENTUM,
            config=config
        )

    @staticmethod
    def entropy_production(config: Optional[ConstraintConfig] = None) -> ThermodynamicConstraint:
        """Create entropy production constraint."""
        return ThermodynamicConstraint(
            constraint_name="entropy_production",
            constraint_type="entropy_production",
            config=config
        )

    @staticmethod
    def positive_temperature(config: Optional[ConstraintConfig] = None) -> ThermodynamicConstraint:
        """Create positive temperature constraint."""
        return ThermodynamicConstraint(
            constraint_name="positive_temperature",
            constraint_type="temperature_positive",
            config=config
        )

    @staticmethod
    def newtons_second_law(config: Optional[ConstraintConfig] = None) -> MechanicalConstraint:
        """Create Newton's second law constraint (F=ma)."""
        return MechanicalConstraint(
            constraint_name="newton_second_law",
            constraint_type="newton_second_law",
            config=config
        )

    @staticmethod
    def hooke_law(config: Optional[ConstraintConfig] = None) -> MechanicalConstraint:
        """Create Hooke's law constraint (F=-kx)."""
        return MechanicalConstraint(
            constraint_name="hooke_law",
            constraint_type="hooke_law",
            config=config
        )

    @staticmethod
    def equilibrium(config: Optional[ConstraintConfig] = None) -> MechanicalConstraint:
        """Create equilibrium constraint (sum of forces = 0)."""
        return MechanicalConstraint(
            constraint_name="mechanical_equilibrium",
            constraint_type="equilibrium",
            config=config
        )

    @staticmethod
    def gauss_law(config: Optional[ConstraintConfig] = None) -> ElectromagneticConstraint:
        """Create Gauss's law constraint."""
        return ElectromagneticConstraint(
            constraint_name="gauss_law",
            constraint_type="gauss_law",
            config=config
        )

    @staticmethod
    def faraday_law(config: Optional[ConstraintConfig] = None) -> ElectromagneticConstraint:
        """Create Faraday's law constraint."""
        return ElectromagneticConstraint(
            constraint_name="faraday_law",
            constraint_type="faraday_law",
            config=config
        )

    @staticmethod
    def no_magnetic_monopoles(config: Optional[ConstraintConfig] = None) -> ElectromagneticConstraint:
        """Create no magnetic monopoles constraint."""
        return ElectromagneticConstraint(
            constraint_name="no_magnetic_monopoles",
            constraint_type="no_magnetic_monopoles",
            config=config
        )

    @staticmethod
    def mass_action_kinetics(config: Optional[ConstraintConfig] = None) -> ChemicalConstraint:
        """Create mass action kinetics constraint."""
        return ChemicalConstraint(
            constraint_name="mass_action",
            constraint_type="mass_action",
            config=config
        )

    @staticmethod
    def chemical_equilibrium(config: Optional[ConstraintConfig] = None) -> ChemicalConstraint:
        """Create chemical equilibrium constraint."""
        return ChemicalConstraint(
            constraint_name="chemical_equilibrium",
            constraint_type="equilibrium",
            config=config
        )

    @staticmethod
    def atom_conservation(config: Optional[ConstraintConfig] = None) -> ChemicalConstraint:
        """Create atom conservation constraint."""
        return ChemicalConstraint(
            constraint_name="atom_conservation",
            constraint_type="conservation_of_atoms",
            config=config
        )


def create_physics_loss(
    constraints: List[PhysicsConstraint],
    predictions: Any,
    context: Dict[str, Any]
) -> Union[float, torch.Tensor]:
    """Create combined physics-informed loss from multiple constraints.
    
    Args:
        constraints: List of physics constraints to apply
        predictions: Model predictions
        context: Context dictionary with coordinates, time, etc.
        
    Returns:
        Combined physics loss
    """
    total_loss = 0.0 if not TORCH_AVAILABLE else torch.tensor(0.0, device=constraints[0]._device if constraints else 'cpu')
    
    for constraint in constraints:
        try:
            loss = constraint.compute_loss(predictions, context)
            if TORCH_AVAILABLE and isinstance(total_loss, torch.Tensor):
                total_loss = total_loss + loss
            else:
                total_loss = total_loss + float(loss)
        except Exception as e:
            logger.error(f"Error computing loss for {constraint.name}: {e}")
    
    return total_loss


__all__ = [
    "ConstraintType",
    "ConservationQuantity",
    "ConstraintViolation",
    "ConstraintConfig",
    "PhysicsConstraint",
    "ConservationLawConstraint",
    "ThermodynamicConstraint",
    "MechanicalConstraint",
    "ElectromagneticConstraint",
    "ChemicalConstraint",
    "ConstraintLibrary",
    "create_physics_loss"
]
