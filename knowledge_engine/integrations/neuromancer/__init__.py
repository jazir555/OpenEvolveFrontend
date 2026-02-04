"""Knowledge Engine Neuromancer Integration.

Physics-informed knowledge graph capabilities.

This module provides a thin wrapper around the primary Neuromancer integration
located at `integrations.neuromancer`. It follows the SSOT (Single Source of Truth)
pattern where all core business logic is in the primary integration.

Usage:
    from knowledge_engine.integrations.neuromancer import NeuromancerKGIntegration
    
    # Initialize integration
    integration = NeuromancerKGIntegration(
        kg_integration_hub=hub,
        memgraph_client=client,
        device="cuda"
    )
    integration.initialize(config)
    
    # Infer temporal dynamics
    result = integration.infer_temporal_dynamics(
        entity_id="weather_station_1",
        property_name="temperature",
        horizon=24
    )
    
    # Validate physical laws
    validation = integration.validate_physical_laws(
        kg_subgraph=subgraph,
        domain="climate"
    )
    
    # Simulate what-if scenario
    simulation = integration.simulate_what_if(
        scenario=scenario,
        constraints=["conservation_of_energy"]
    )
    
    # Calibrate model from observations
    calibration = integration.calibrate_from_observations(
        entity_id="sensor_1",
        observations=obs_data
    )
    
    # Discover equations
    equation = integration.discover_equations(
        data=training_data,
        candidate_terms=["y", "y^2", "sin(y)"]
    )
    
    # Create physics-enriched embedding
    embedding = integration.physics_enriched_embedding(entity)

Note:
    This is a wrapper module. For core functionality, use:
    - integrations.neuromancer (primary implementation)
    
    Old stub imports are deprecated and will be removed in v2.0.
"""

import warnings
from typing import Any

# Primary integration - SSOT
from .neuromancer_integration import (
    NeuromancerKGIntegration,
    UnifiedKGIntegrationHub,
    PredictionResult,
    ValidationResult,
    CalibrationResult,
    DiscoveredEquation,
    PhysicsAwareEmbedding
)

# Re-export from primary implementation
from integrations.neuromancer import (
    # Core adapters
    NeuromancerAdapter,
    NeuralOperatorConfig,
    NeuralOperatorType,
    SolutionResult,
    DynamicsModel,
    TrajectoryResult,
    CalibratedModel,
    
    # Neural operators
    NeuralOperatorBase,
    FNOOperator,
    DeepONetOperator,
    PINNOperator,
    create_operator,
    MODEL_REGISTRY,
    
    # Physics constraints
    ConstraintType,
    ConservationQuantity,
    ConstraintViolation,
    ConstraintConfig,
    PhysicsConstraint,
    ConservationLawConstraint,
    ThermodynamicConstraint,
    MechanicalConstraint,
    ElectromagneticConstraint,
    ChemicalConstraint,
    ConstraintLibrary,
    create_physics_loss,
    
    # Scientific domains
    DomainType,
    DomainConfig,
    SimulationResult,
    ScientificDomain,
    ClimateModeling,
    FluidDynamics,
    StructuralMechanics,
    ChemicalKinetics,
    BiologicalSystems,
    DomainLibrary,
    
    # KG Physics bridge
    EntityPhysicsType,
    RelationshipPhysicsType,
    PhysicsProblem,
    KGUpdates,
    ConsistencyReport,
    InferredProperties,
    SimulationResultKG,
    KGPhysicsBridge,
    
    # Legacy adapters
    NeuroMANCERAdapter,
    HybridSolver,
    LeanAideNeuroMANCERBridge
)

__version__ = "1.0.0"
__author__ = "OpenEvolve Team"


# Deprecation warnings for old stub imports
class _DeprecatedImport:
    """Helper class for deprecated imports."""
    
    def __init__(self, name: str, replacement: str):
        self.name = name
        self.replacement = replacement
    
    def __call__(self, *args, **kwargs):
        warnings.warn(
            f"{self.name} is deprecated. Use {self.replacement} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return None


def _deprecated_import(name: str, replacement: str):
    """Create deprecated import warning."""
    warnings.warn(
        f"Importing {name} from knowledge_engine.integrations.neuromancer is deprecated. "
        f"Use {replacement} from integrations.neuromancer instead.",
        DeprecationWarning,
        stacklevel=2
    )


__all__ = [
    # Knowledge Engine integration (primary)
    "NeuromancerKGIntegration",
    "UnifiedKGIntegrationHub",
    
    # Result types
    "PredictionResult",
    "ValidationResult",
    "CalibrationResult",
    "DiscoveredEquation",
    "PhysicsAwareEmbedding",
    
    # Neural operators
    "NeuromancerAdapter",
    "NeuralOperatorConfig",
    "NeuralOperatorType",
    "SolutionResult",
    "DynamicsModel",
    "TrajectoryResult",
    "CalibratedModel",
    "NeuralOperatorBase",
    "FNOOperator",
    "DeepONetOperator",
    "PINNOperator",
    "create_operator",
    "MODEL_REGISTRY",
    
    # Physics constraints
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
    "create_physics_loss",
    
    # Scientific domains
    "DomainType",
    "DomainConfig",
    "SimulationResult",
    "ScientificDomain",
    "ClimateModeling",
    "FluidDynamics",
    "StructuralMechanics",
    "ChemicalKinetics",
    "BiologicalSystems",
    "DomainLibrary",
    
    # KG Physics bridge
    "EntityPhysicsType",
    "RelationshipPhysicsType",
    "PhysicsProblem",
    "KGUpdates",
    "ConsistencyReport",
    "InferredProperties",
    "SimulationResultKG",
    "KGPhysicsBridge",
    
    # Legacy adapters
    "NeuroMANCERAdapter",
    "HybridSolver",
    "LeanAideNeuroMANCERBridge"
]


# Convenience function for creating integration
def create_neuromancer_kg_integration(
    kg_integration_hub=None,
    memgraph_client=None,
    device: str = "cpu",
    config: dict = None
) -> NeuromancerKGIntegration:
    """
    Factory function to create Neuromancer KG Integration.
    
    Args:
        kg_integration_hub: UnifiedKGIntegrationHub instance
        memgraph_client: Memgraph client
        device: Device for computation
        config: Configuration dictionary
        
    Returns:
        Initialized NeuromancerKGIntegration
        
    Example:
        >>> integration = create_neuromancer_kg_integration(
        ...     kg_integration_hub=hub,
        ...     device="cuda",
        ...     config={"neural_operator": {"learning_rate": 1e-4}}
        ... )
    """
    integration = NeuromancerKGIntegration(
        kg_integration_hub=kg_integration_hub,
        memgraph_client=memgraph_client,
        device=device
    )
    integration.initialize(config)
    return integration
