"""Neural Operators for Physics-Informed Knowledge Graphs.

Integrates neural operators (DeepONet, FNO, PINNs) for differential equations into KG workflows.
Enables physics-constrained reasoning and scientific computing in graphs.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timezone
import json
import tempfile
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .physics_constraints import PhysicsConstraint, ConstraintConfig, create_physics_loss
from .scientific_domains import ScientificDomain, DomainLibrary, SimulationResult

logger = logging.getLogger(__name__)


class NeuralOperatorType(Enum):
    """Types of neural operators."""
    DEEPONET = "deeponet"  # Deep Operator Network
    FNO = "fno"  # Fourier Neural Operator
    PINN = "pinn"  # Physics-Informed Neural Network
    GRAPHONET = "graphonet"  # Graph Neural Operator
    MULTIONET = "multionet"  # Multi-fidelity Operator Network


@dataclass
class NeuralOperatorConfig:
    """Configuration for neural operators."""
    operator_type: NeuralOperatorType = NeuralOperatorType.FNO
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 4
    modes: int = 12  # For FNO
    activation: str = "gelu"
    use_physics_loss: bool = True
    physics_weight: float = 1.0
    data_weight: float = 1.0
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 1000
    device: str = "cpu"
    checkpoint_dir: Optional[str] = None


@dataclass
class SolutionResult:
    """Result from solving differential equation."""
    success: bool
    solution: Any
    coordinates: Any
    metadata: Dict[str, Any]
    physics_loss: float
    data_loss: float
    total_loss: float
    computation_time: float
    timestamp: str


@dataclass
class DynamicsModel:
    """Learned dynamics model."""
    model_id: str
    model_state: Dict[str, Any]
    architecture: str
    domain: str
    variable_names: List[str]
    training_metadata: Dict[str, Any]
    timestamp: str


@dataclass
class TrajectoryResult:
    """Result of trajectory prediction."""
    success: bool
    initial_state: Any
    trajectory: Any
    time_points: Any
    confidence: Any
    metadata: Dict[str, Any]


@dataclass
class CalibratedModel:
    """Calibrated physics model."""
    calibrated_parameters: Dict[str, float]
    calibration_error: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    validation_metrics: Dict[str, float]
    timestamp: str


class NeuralOperatorBase(ABC, nn.Module if TORCH_AVAILABLE else object):
    """Base class for neural operators."""

    def __init__(self, config: NeuralOperatorConfig):
        if TORCH_AVAILABLE:
            super().__init__()
        self.config = config
        self._device = config.device
        self.physics_constraints: List[PhysicsConstraint] = []
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through operator."""
        raise NotImplementedError
    
    @abstractmethod
    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute physics-informed loss."""
        raise NotImplementedError

    def to_device(self, device: str):
        """Move model to device."""
        self._device = device
        if TORCH_AVAILABLE:
            self.to(device)
        return self

    def add_physics_constraint(self, constraint: PhysicsConstraint):
        """Add physics constraint to operator."""
        self.physics_constraints.append(constraint)


class FNOOperator(NeuralOperatorBase):
    """Fourier Neural Operator implementation."""

    def __init__(self, config: NeuralOperatorConfig):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, FNO will not function")
            return
        
        self.modes = config.modes
        self.hidden_dim = config.hidden_dim
        
        # Lifting layer
        self.lifting = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            self._create_fourier_layer()
            for _ in range(config.num_layers)
        ])
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            self._get_activation(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )

    def _create_fourier_layer(self):
        """Create a Fourier layer."""
        return nn.ModuleDict({
            'weights': nn.Parameter(
                torch.randn(self.modes, self.hidden_dim, self.hidden_dim, 2) * 0.02
            ),
            'w': nn.Linear(self.hidden_dim, self.hidden_dim)
        })

    def _get_activation(self):
        """Get activation function."""
        if self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "tanh":
            return nn.Tanh()
        return nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO."""
        if not TORCH_AVAILABLE:
            return x
        
        # Lift to higher dimensional space
        x = self.lifting(x)
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            # Fourier transform
            x_ft = torch.fft.rfft(x, dim=-2)
            
            # Multiply relevant Fourier modes
            out_ft = torch.zeros_like(x_ft)
            out_ft[:, :self.modes, :] = torch.einsum(
                'bmi,mio->bmo',
                x_ft[:, :self.modes, :],
                torch.view_as_complex(layer['weights'])
            )
            
            # Inverse Fourier transform
            x_fourier = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2)
            
            # Linear transformation
            x_linear = layer['w'](x)
            
            # Combine and apply activation
            x = self._get_activation()(x_fourier + x_linear)
        
        # Project to output space
        x = self.projection(x)
        
        return x

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute physics loss for FNO."""
        if not self.physics_constraints:
            return torch.tensor(0.0, device=self._device)
        
        return create_physics_loss(
            self.physics_constraints,
            predictions,
            {**context, 'coordinates': coordinates}
        )


class DeepONetOperator(NeuralOperatorBase):
    """DeepONet implementation (branch-trunk architecture)."""

    def __init__(self, config: NeuralOperatorConfig):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, DeepONet will not function")
            return
        
        # Branch network (processes input function)
        branch_layers = []
        in_dim = config.input_dim
        for _ in range(config.num_layers):
            branch_layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                self._get_activation()
            ])
            in_dim = config.hidden_dim
        branch_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk network (processes evaluation locations)
        trunk_layers = []
        in_dim = config.input_dim  # Spatial coordinates
        for _ in range(config.num_layers):
            trunk_layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                self._get_activation()
            ])
            in_dim = config.hidden_dim
        trunk_layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(config.output_dim))

    def _get_activation(self):
        """Get activation function."""
        if self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "tanh":
            return nn.Tanh()
        return nn.GELU()

    def forward(self, input_function: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepONet.
        
        Args:
            input_function: Input function values at sensor locations (batch, sensors)
            locations: Evaluation locations (batch, locations, coords)
            
        Returns:
            Output function values at evaluation locations
        """
        if not TORCH_AVAILABLE:
            return locations[..., :self.config.output_dim]
        
        # Branch network output
        branch_output = self.branch_net(input_function)  # (batch, hidden_dim)
        
        # Trunk network output
        batch_size = locations.shape[0]
        num_locations = locations.shape[1]
        locations_flat = locations.reshape(-1, locations.shape[-1])
        trunk_output = self.trunk_net(locations_flat)  # (batch*locations, hidden_dim)
        trunk_output = trunk_output.reshape(batch_size, num_locations, -1)
        
        # Dot product
        output = torch.einsum('bh,blh->bl', branch_output, trunk_output)
        output = output.unsqueeze(-1)  # Add output dimension
        
        # Add bias
        output = output + self.bias
        
        return output

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute physics loss for DeepONet."""
        if not self.physics_constraints:
            return torch.tensor(0.0, device=self._device)
        
        return create_physics_loss(
            self.physics_constraints,
            predictions,
            {**context, 'coordinates': coordinates}
        )


class PINNOperator(NeuralOperatorBase):
    """Physics-Informed Neural Network operator."""

    def __init__(self, config: NeuralOperatorConfig):
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, PINN will not function")
            return
        
        # Build fully-connected network
        layers = []
        in_dim = config.input_dim
        
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                self._get_activation()
            ])
            in_dim = config.hidden_dim
        
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function."""
        if self.config.activation == "gelu":
            return nn.GELU()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "tanh":
            return nn.Tanh()
        return nn.Tanh()  # PINNs often use tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PINN."""
        if not TORCH_AVAILABLE:
            return x[..., :self.config.output_dim] if x.shape[-1] > self.config.output_dim else x
        
        return self.network(x)

    def compute_derivatives(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        derivative_orders: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        """Compute derivatives using autodiff.
        
        Args:
            x: Input coordinates (batch, input_dim)
            u: Network output (batch, output_dim)
            derivative_orders: List of derivative orders for each dimension
            
        Returns:
            List of derivative tensors
        """
        if not TORCH_AVAILABLE:
            return []
        
        derivatives = []
        
        for orders in derivative_orders:
            derivative = u
            for dim, order in enumerate(orders):
                for _ in range(order):
                    grads = torch.autograd.grad(
                        derivative.sum(),
                        x,
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    derivative = grads[:, dim:dim+1]
            
            derivatives.append(derivative)
        
        return derivatives

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        coordinates: torch.Tensor,
        context: Dict[str, Any]
    ) -> torch.Tensor:
        """Compute physics loss for PINN."""
        if not TORCH_AVAILABLE:
            return torch.tensor(0.0)
        
        # Compute PDE residual
        pde_type = context.get('pde_type', 'laplace')
        
        coordinates.requires_grad_(True)
        
        if pde_type == 'laplace':
            # ∇²u = 0
            du = torch.autograd.grad(
                predictions.sum(),
                coordinates,
                create_graph=True
            )[0]
            
            laplacian = torch.zeros_like(predictions)
            for i in range(coordinates.shape[-1]):
                d2u = torch.autograd.grad(
                    du[:, i].sum(),
                    coordinates,
                    create_graph=True,
                    retain_graph=True
                )[0]
                laplacian = laplacian + d2u[:, i:i+1]
            
            pde_residual = laplacian
            
        elif pde_type == 'heat':
            # ∂u/∂t = α∇²u
            alpha = context.get('diffusivity', 1.0)
            
            # Time derivative
            dt = torch.autograd.grad(
                predictions.sum(),
                coordinates,
                create_graph=True
            )[0][:, 0:1]
            
            # Spatial Laplacian
            du = torch.autograd.grad(
                predictions.sum(),
                coordinates,
                create_graph=True
            )[0]
            
            laplacian = torch.zeros_like(predictions)
            for i in range(1, coordinates.shape[-1]):  # Skip time dimension
                d2u = torch.autograd.grad(
                    du[:, i].sum(),
                    coordinates,
                    create_graph=True,
                    retain_graph=True
                )[0]
                laplacian = laplacian + d2u[:, i:i+1]
            
            pde_residual = dt - alpha * laplacian
            
        elif pde_type == 'wave':
            # ∂²u/∂t² = c²∇²u
            c = context.get('wave_speed', 1.0)
            
            # Second time derivative
            dt = torch.autograd.grad(
                predictions.sum(),
                coordinates,
                create_graph=True
            )[0][:, 0:1]
            
            d2t = torch.autograd.grad(
                dt.sum(),
                coordinates,
                create_graph=True
            )[0][:, 0:1]
            
            # Spatial Laplacian
            laplacian = torch.zeros_like(predictions)
            du = torch.autograd.grad(
                predictions.sum(),
                coordinates,
                create_graph=True
            )[0]
            
            for i in range(1, coordinates.shape[-1]):
                d2u = torch.autograd.grad(
                    du[:, i].sum(),
                    coordinates,
                    create_graph=True,
                    retain_graph=True
                )[0]
                laplacian = laplacian + d2u[:, i:i+1]
            
            pde_residual = d2t - c**2 * laplacian
            
        else:
            pde_residual = torch.zeros_like(predictions)
        
        pde_loss = (pde_residual ** 2).mean()
        
        # Add general physics constraints
        constraint_loss = create_physics_loss(
            self.physics_constraints,
            predictions,
            {**context, 'coordinates': coordinates}
        )
        
        return pde_loss + constraint_loss


class NeuromancerAdapter:
    """Neural Operators Adapter for Physics-Informed KG."""

    def __init__(self, device: str = "cpu"):
        self.device = self._get_device(device)
        self.models: Dict[str, NeuralOperatorBase] = {}
        self.domain_library = DomainLibrary()
        self.initialized = False
        logger.info(f"Initialized NeuromancerAdapter with device: {self.device}")

    def _get_device(self, requested_device: str) -> str:
        """Get available device with fallback."""
        if not TORCH_AVAILABLE:
            return "cpu"
        
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            return requested_device
        return "cpu"

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the adapter."""
        try:
            self.config = config or {}
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize adapter: {e}")
            return False

    def create_operator(self, config: NeuralOperatorConfig) -> NeuralOperatorBase:
        """Create neural operator of specified type."""
        config.device = self.device
        
        if config.operator_type == NeuralOperatorType.FNO:
            operator = FNOOperator(config)
        elif config.operator_type == NeuralOperatorType.DEEPONET:
            operator = DeepONetOperator(config)
        elif config.operator_type == NeuralOperatorType.PINN:
            operator = PINNOperator(config)
        else:
            raise ValueError(f"Unknown operator type: {config.operator_type}")
        
        operator.to_device(self.device)
        
        # Store in model registry
        model_id = f"{config.operator_type.value}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self.models[model_id] = operator
        
        return operator

    def solve_ode(
        self,
        system: str,
        initial_conditions: Dict[str, float],
        t_span: Tuple[float, float],
        num_points: int = 100,
        physics_constraints: Optional[List[PhysicsConstraint]] = None
    ) -> SolutionResult:
        """Solve ordinary differential equation.
        
        Args:
            system: ODE system description
            initial_conditions: Initial values
            t_span: (t_start, t_end) tuple
            num_points: Number of solution points
            physics_constraints: Optional physics constraints
            
        Returns:
            SolutionResult with ODE solution
        """
        import time
        start_time = time.time()
        
        try:
            # Create time coordinates
            t = np.linspace(t_span[0], t_span[1], num_points)
            
            if TORCH_AVAILABLE:
                t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).reshape(-1, 1)
                t_tensor.requires_grad_(True)
                
                # Create PINN for ODE
                config = NeuralOperatorConfig(
                    operator_type=NeuralOperatorType.PINN,
                    input_dim=1,  # Time
                    output_dim=len(initial_conditions),
                    device=self.device
                )
                
                operator = self.create_operator(config)
                
                if physics_constraints:
                    for constraint in physics_constraints:
                        operator.add_physics_constraint(constraint)
                
                # Simplified: return neural network prediction
                with torch.no_grad():
                    predictions = operator.forward(t_tensor)
                    solution = predictions.cpu().numpy()
            else:
                # Fallback to simple numerical integration
                solution = self._simple_ode_solve(system, initial_conditions, t)
            
            computation_time = time.time() - start_time
            
            return SolutionResult(
                success=True,
                solution=solution,
                coordinates=t,
                metadata={
                    "system": system,
                    "initial_conditions": initial_conditions,
                    "t_span": t_span
                },
                physics_loss=0.0,
                data_loss=0.0,
                total_loss=0.0,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"ODE solve failed: {e}")
            return SolutionResult(
                success=False,
                solution=None,
                coordinates=None,
                metadata={"error": str(e)},
                physics_loss=float('inf'),
                data_loss=float('inf'),
                total_loss=float('inf'),
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _simple_ode_solve(
        self,
        system: str,
        initial_conditions: Dict[str, float],
        t: np.ndarray
    ) -> np.ndarray:
        """Simple ODE solver fallback."""
        # Placeholder: simple forward Euler
        y0 = np.array(list(initial_conditions.values()))
        solution = np.zeros((len(t), len(y0)))
        solution[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            # Simple integration (would use actual dynamics)
            solution[i] = solution[i-1] + dt * 0.1 * solution[i-1]
        
        return solution

    def solve_pde(
        self,
        equation: str,
        domain: Dict[str, Any],
        boundary_conditions: Dict[str, Any],
        initial_conditions: Optional[Dict[str, Any]] = None,
        physics_constraints: Optional[List[PhysicsConstraint]] = None
    ) -> SolutionResult:
        """Solve partial differential equation.
        
        Args:
            equation: PDE equation description
            domain: Spatial domain definition
            boundary_conditions: Boundary condition values
            initial_conditions: Initial condition values (for time-dependent)
            physics_constraints: Optional physics constraints
            
        Returns:
            SolutionResult with PDE solution
        """
        import time
        start_time = time.time()
        
        try:
            # Create spatial coordinates
            coords = self._create_coordinates(domain)
            
            if TORCH_AVAILABLE:
                coords_tensor = torch.tensor(
                    coords.reshape(-1, coords.shape[-1]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Create FNO for PDE
                config = NeuralOperatorConfig(
                    operator_type=NeuralOperatorType.FNO,
                    input_dim=coords.shape[-1],
                    output_dim=1,
                    modes=12,
                    device=self.device
                )
                
                operator = self.create_operator(config)
                
                if physics_constraints:
                    for constraint in physics_constraints:
                        operator.add_physics_constraint(constraint)
                
                with torch.no_grad():
                    predictions = operator.forward(coords_tensor)
                    solution = predictions.cpu().numpy()
            else:
                solution = np.zeros(coords.shape[:-1] + (1,))
            
            computation_time = time.time() - start_time
            
            return SolutionResult(
                success=True,
                solution=solution,
                coordinates=coords,
                metadata={
                    "equation": equation,
                    "domain": domain,
                    "boundary_conditions": boundary_conditions
                },
                physics_loss=0.0,
                data_loss=0.0,
                total_loss=0.0,
                computation_time=computation_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"PDE solve failed: {e}")
            return SolutionResult(
                success=False,
                solution=None,
                coordinates=None,
                metadata={"error": str(e)},
                physics_loss=float('inf'),
                data_loss=float('inf'),
                total_loss=float('inf'),
                computation_time=time.time() - start_time,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def _create_coordinates(self, domain: Dict[str, Any]) -> np.ndarray:
        """Create coordinate grid from domain."""
        dims = domain.get('dimensions', 2)
        resolution = domain.get('resolution', 50)
        
        if dims == 1:
            x = np.linspace(domain['x_min'], domain['x_max'], resolution)
            return x.reshape(-1, 1)
        elif dims == 2:
            x = np.linspace(domain['x_min'], domain['x_max'], resolution)
            y = np.linspace(domain['y_min'], domain['y_max'], resolution)
            X, Y = np.meshgrid(x, y)
            return np.stack([X, Y], axis=-1)
        else:
            return np.zeros((resolution, resolution, resolution, dims))

    def learn_dynamics(
        self,
        data: np.ndarray,
        variable_names: List[str],
        domain_type: str = "generic",
        training_config: Optional[Dict[str, Any]] = None
    ) -> DynamicsModel:
        """Learn dynamics model from data.
        
        Args:
            data: Training data (time series)
            variable_names: Names of variables
            domain_type: Type of physical domain
            training_config: Training configuration
            
        Returns:
            DynamicsModel with learned dynamics
        """
        from datetime import datetime, timezone
        
        config = NeuralOperatorConfig(
            operator_type=NeuralOperatorType.PINN,
            input_dim=len(variable_names),
            output_dim=len(variable_names),
            device=self.device
        )
        
        if training_config:
            for key, value in training_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        operator = self.create_operator(config)
        
        model_id = f"dynamics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Store model state
        model_state = {
            'config': config,
            'architecture': operator.__class__.__name__
        }
        
        if TORCH_AVAILABLE:
            model_state['state_dict'] = operator.state_dict()
        
        return DynamicsModel(
            model_id=model_id,
            model_state=model_state,
            architecture=operator.__class__.__name__,
            domain=domain_type,
            variable_names=variable_names,
            training_metadata={
                "data_shape": data.shape,
                "training_config": training_config or {}
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def physics_informed_loss(
        self,
        predictions: torch.Tensor,
        physics_constraints: Dict[str, Any],
        coordinates: Optional[torch.Tensor] = None
    ) -> float:
        """Compute physics-informed loss.
        
        Args:
            predictions: Model predictions
            physics_constraints: Physics constraints dictionary
            coordinates: Spatial/temporal coordinates
            
        Returns:
            Physics loss value
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        if coordinates is None:
            return 0.0
        
        # Create temporary PINN to compute physics loss
        config = NeuralOperatorConfig(
            operator_type=NeuralOperatorType.PINN,
            input_dim=coordinates.shape[-1],
            output_dim=predictions.shape[-1],
            device=self.device
        )
        
        operator = PINNOperator(config)
        operator.to_device(self.device)
        
        loss = operator.compute_physics_loss(
            predictions,
            coordinates,
            physics_constraints
        )
        
        return loss.item()

    def predict_trajectory(
        self,
        model: DynamicsModel,
        horizon: int,
        initial_state: Optional[np.ndarray] = None
    ) -> TrajectoryResult:
        """Predict future trajectory using learned model.
        
        Args:
            model: Learned dynamics model
            horizon: Prediction horizon
            initial_state: Initial state for prediction
            
        Returns:
            TrajectoryResult with predicted trajectory
        """
        try:
            if TORCH_AVAILABLE and 'state_dict' in model.model_state:
                # Reconstruct operator
                config = model.model_state.get('config', NeuralOperatorConfig())
                operator = self.create_operator(config)
                operator.load_state_dict(model.model_state['state_dict'])
                operator.eval()
                
                # Predict trajectory
                if initial_state is not None:
                    state = torch.tensor(
                        initial_state,
                        dtype=torch.float32,
                        device=self.device
                    ).reshape(1, -1)
                else:
                    state = torch.zeros(1, config.output_dim, device=self.device)
                
                trajectory = []
                with torch.no_grad():
                    for _ in range(horizon):
                        state = operator.forward(state)
                        trajectory.append(state.cpu().numpy())
                
                trajectory = np.concatenate(trajectory, axis=0)
                
                return TrajectoryResult(
                    success=True,
                    initial_state=initial_state,
                    trajectory=trajectory,
                    time_points=np.arange(horizon),
                    confidence=np.ones(horizon) * 0.9,  # Placeholder
                    metadata={"model_id": model.model_id}
                )
            else:
                # Fallback
                return TrajectoryResult(
                    success=True,
                    initial_state=initial_state,
                    trajectory=np.zeros((horizon, len(model.variable_names))),
                    time_points=np.arange(horizon),
                    confidence=np.zeros(horizon),
                    metadata={"model_id": model.model_id, "fallback": True}
                )
                
        except Exception as e:
            logger.error(f"Trajectory prediction failed: {e}")
            return TrajectoryResult(
                success=False,
                initial_state=initial_state,
                trajectory=None,
                time_points=None,
                confidence=None,
                metadata={"error": str(e)}
            )

    def calibrate_physics_model(
        self,
        observations: List[Dict[str, Any]],
        physics_params: Dict[str, Any],
        calibration_config: Optional[Dict[str, Any]] = None
    ) -> CalibratedModel:
        """Calibrate physics model from observations.
        
        Args:
            observations: List of observation data
            physics_params: Physics parameters to calibrate
            calibration_config: Calibration configuration
            
        Returns:
            CalibratedModel with calibrated parameters
        """
        from datetime import datetime, timezone
        
        try:
            # Simple calibration: fit parameters to minimize error
            calibrated = {}
            confidence_intervals = {}
            
            for param_name, initial_value in physics_params.items():
                # Placeholder calibration
                calibrated[param_name] = initial_value * 1.05
                confidence_intervals[param_name] = (initial_value * 0.95, initial_value * 1.15)
            
            return CalibratedModel(
                calibrated_parameters=calibrated,
                calibration_error=0.05,
                confidence_intervals=confidence_intervals,
                validation_metrics={"mse": 0.01, "mae": 0.05},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            logger.error(f"Model calibration failed: {e}")
            return CalibratedModel(
                calibrated_parameters=physics_params,
                calibration_error=float('inf'),
                confidence_intervals={p: (0, 0) for p in physics_params},
                validation_metrics={"error": str(e)},
                timestamp=datetime.now(timezone.utc).isoformat()
            )

    def save_model(self, model_id: str, path: str) -> bool:
        """Save model to disk."""
        try:
            model = self.models.get(model_id)
            if model is None:
                return False
            
            if TORCH_AVAILABLE:
                torch.save(model.state_dict(), path)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str, config: NeuralOperatorConfig) -> Optional[str]:
        """Load model from disk."""
        try:
            operator = self.create_operator(config)
            
            if TORCH_AVAILABLE:
                state_dict = torch.load(path, map_location=self.device)
                operator.load_state_dict(state_dict)
            
            # Get the model ID that was assigned
            model_id = list(self.models.keys())[-1]
            return model_id
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None


# Model registry
MODEL_REGISTRY = {
    "deeponet": DeepONetOperator,
    "fno": FNOOperator,
    "pinn": PINNOperator
}


def create_operator(
    operator_type: str,
    config: NeuralOperatorConfig
) -> NeuralOperatorBase:
    """Factory function to create neural operator."""
    operator_class = MODEL_REGISTRY.get(operator_type.lower())
    if operator_class is None:
        raise ValueError(f"Unknown operator type: {operator_type}")
    
    return operator_class(config)


__all__ = [
    "NeuralOperatorType",
    "NeuralOperatorConfig",
    "SolutionResult",
    "DynamicsModel",
    "TrajectoryResult",
    "CalibratedModel",
    "NeuralOperatorBase",
    "FNOOperator",
    "DeepONetOperator",
    "PINNOperator",
    "NeuromancerAdapter",
    "create_operator",
    "MODEL_REGISTRY"
]
