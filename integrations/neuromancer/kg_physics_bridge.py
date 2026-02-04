"""Bridge between Knowledge Graph and Physics simulations.

Maps KG entities/relationships to physics simulations.
Enables physics-informed knowledge graphs where relationships follow physical laws.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from enum import Enum
from datetime import datetime, timezone
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .physics_constraints import (
    PhysicsConstraint, ConstraintLibrary, ConstraintViolation,
    ConservationQuantity, ConstraintType
)
from .scientific_domains import (
    ScientificDomain, DomainLibrary, SimulationResult, DomainType
)

logger = logging.getLogger(__name__)


class EntityPhysicsType(Enum):
    """Physics types for KG entities."""
    CONTINUUM_FIELD = "continuum_field"  # Fluid, solid, temperature field
    PARTICLE = "particle"  # Point mass, electron, atom
    SYSTEM = "system"  # Collection of interacting entities
    BOUNDARY = "boundary"  # Domain boundary
    SOURCE = "source"  # External forcing, heat source
    OBSERVER = "observer"  # Measurement device


class RelationshipPhysicsType(Enum):
    """Physics types for KG relationships."""
    CAUSAL = "causal"  # A causes B
    CONSERVES = "conserves"  # Conserves quantity
    FLOWS_TO = "flows_to"  # Mass/energy flow
    INTERACTS_WITH = "interacts_with"  # Force interaction
    CONSTRAINS = "constrains"  # Boundary constraint
    MEASURES = "measures"  # Observation
    EMBEDS_IN = "embeds_in"  # Spatial embedding


@dataclass
class PhysicsProblem:
    """Physics problem extracted from KG."""
    problem_type: str
    domain: str
    equations: List[str]
    variables: Dict[str, Any]
    parameters: Dict[str, float]
    initial_conditions: Dict[str, Any]
    boundary_conditions: Dict[str, Any]
    constraints: List[str]
    geometry: Optional[Dict[str, Any]] = None
    temporal_settings: Optional[Dict[str, Any]] = None


@dataclass
class KGUpdates:
    """Updates to apply to knowledge graph from physics solution."""
    new_entities: List[Dict[str, Any]] = field(default_factory=list)
    new_relationships: List[Dict[str, Any]] = field(default_factory=list)
    property_updates: List[Dict[str, Any]] = field(default_factory=list)
    temporal_data: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """Report on physics consistency of KG."""
    is_consistent: bool
    violations: List[ConstraintViolation]
    warnings: List[str]
    suggestions: List[str]
    confidence: float
    timestamp: str


@dataclass
class InferredProperties:
    """Properties inferred from physics model."""
    entity_id: str
    inferred_values: Dict[str, Any]
    confidence: Dict[str, float]
    inference_method: str
    timestamp: str


@dataclass
class SimulationResultKG:
    """Simulation result with KG integration."""
    simulation_id: str
    success: bool
    solution_data: Any
    kg_updates: KGUpdates
    visualizations: List[str]
    metrics: Dict[str, float]
    timestamp: str


class KGPhysicsBridge:
    """Bridge between Knowledge Graph and Physics simulations."""

    # Mapping of entity types to physics domains
    ENTITY_DOMAIN_MAP = {
        "atmosphere": DomainType.CLIMATE,
        "ocean": DomainType.CLIMATE,
        "fluid": DomainType.FLUID_DYNAMICS,
        "air": DomainType.FLUID_DYNAMICS,
        "water": DomainType.FLUID_DYNAMICS,
        "structure": DomainType.STRUCTURAL_MECHANICS,
        "beam": DomainType.STRUCTURAL_MECHANICS,
        "plate": DomainType.STRUCTURAL_MECHANICS,
        "chemical_species": DomainType.CHEMICAL_KINETICS,
        "reactor": DomainType.CHEMICAL_KINETICS,
        "population": DomainType.BIOLOGICAL_SYSTEMS,
        "species": DomainType.BIOLOGICAL_SYSTEMS,
        "disease": DomainType.BIOLOGICAL_SYSTEMS,
    }

    # Mapping of relationship types to physics constraints
    RELATIONSHIP_CONSTRAINT_MAP = {
        "conserves_mass": "conservation_of_mass",
        "conserves_energy": "conservation_of_energy",
        "conserves_momentum": "conservation_of_momentum",
        "heats": "second_law",
        "cools": "second_law",
        "flows": "mass_conservation",
        "applies_force": "newton_second_law",
        "deforms": "hooke_law",
        "reacts": "mass_action_kinetics",
        "catalyzes": "mass_action_kinetics",
    }

    def __init__(self):
        self.domains = DomainLibrary.get_all_domains()
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        self.simulation_cache: Dict[str, SimulationResultKG] = {}
        logger.info("Initialized KG-Physics Bridge")

    def kg_to_physics_problem(self, kg_subgraph: Dict[str, Any]) -> PhysicsProblem:
        """Convert KG subgraph to physics problem.
        
        Args:
            kg_subgraph: Knowledge graph subgraph with entities and relationships
            
        Returns:
            PhysicsProblem extracted from KG
        """
        entities = kg_subgraph.get('entities', [])
        relationships = kg_subgraph.get('relationships', [])
        
        # Determine domain from entities
        domain = self._infer_domain(entities)
        
        # Extract variables from entity properties
        variables = self._extract_variables(entities)
        
        # Extract parameters
        parameters = self._extract_parameters(entities)
        
        # Extract equations from relationships
        equations = self._extract_equations(relationships, domain)
        
        # Extract initial conditions
        initial_conditions = self._extract_initial_conditions(entities)
        
        # Extract boundary conditions
        boundary_conditions = self._extract_boundary_conditions(entities, relationships)
        
        # Extract constraints
        constraints = self._extract_constraints(relationships)
        
        # Extract geometry
        geometry = self._extract_geometry(entities)
        
        # Determine problem type
        problem_type = self._determine_problem_type(kg_subgraph)
        
        return PhysicsProblem(
            problem_type=problem_type,
            domain=domain.value if isinstance(domain, DomainType) else domain,
            equations=equations,
            variables=variables,
            parameters=parameters,
            initial_conditions=initial_conditions,
            boundary_conditions=boundary_conditions,
            constraints=constraints,
            geometry=geometry
        )

    def physics_solution_to_kg(self, solution: SimulationResult) -> KGUpdates:
        """Convert physics solution to KG updates.
        
        Args:
            solution: Physics simulation result
            
        Returns:
            KGUpdates to apply to knowledge graph
        """
        updates = KGUpdates()
        
        if not solution.success:
            updates.metadata['error'] = solution.validation_errors
            return updates
        
        solution_data = solution.solution
        
        # Create temporal entities for time-dependent solutions
        if 'time_points' in solution_data or 'trajectory' in solution_data:
            temporal_entities = self._create_temporal_entities(solution)
            updates.new_entities.extend(temporal_entities)
            updates.temporal_data.extend(self._extract_temporal_data(solution))
        
        # Create field entities
        if 'velocity_field' in solution_data or 'temperature_field' in solution_data:
            field_entities = self._create_field_entities(solution)
            updates.new_entities.extend(field_entities)
        
        # Create relationships based on physics interactions
        relationships = self._create_physics_relationships(solution)
        updates.new_relationships.extend(relationships)
        
        # Update entity properties with computed values
        property_updates = self._create_property_updates(solution)
        updates.property_updates.extend(property_updates)
        
        # Add metadata
        updates.metadata = {
            'domain': solution.domain,
            'computation_time': solution.computation_time,
            'constraints_satisfied': solution.constraints_satisfied,
            'timestamp': solution.timestamp
        }
        
        return updates

    def validate_physics_consistency(self, kg_data: Dict[str, Any]) -> ConsistencyReport:
        """Validate physics consistency of KG data.
        
        Args:
            kg_data: Knowledge graph data to validate
            
        Returns:
            ConsistencyReport with validation results
        """
        violations = []
        warnings = []
        suggestions = []
        
        entities = kg_data.get('entities', [])
        relationships = kg_data.get('relationships', [])
        
        # Check entity physics types
        for entity in entities:
            entity_type = entity.get('type', '').lower()
            physics_props = entity.get('physics_properties', {})
            
            # Check for required physics properties
            if entity_type in self.ENTITY_DOMAIN_MAP:
                domain = self.ENTITY_DOMAIN_MAP[entity_type]
                required_props = self._get_required_properties(domain)
                
                missing_props = [p for p in required_props if p not in physics_props]
                if missing_props:
                    warnings.append(
                        f"Entity {entity.get('id')} missing physics properties: {missing_props}"
                    )
                    suggestions.append(
                        f"Add properties {missing_props} to entity {entity.get('id')}"
                    )
        
        # Check conservation law consistency
        conservation_violations = self._check_conservation_laws(entities, relationships)
        violations.extend(conservation_violations)
        
        # Check causal consistency
        causal_issues = self._check_causal_consistency(relationships)
        warnings.extend(causal_issues)
        
        # Check unit consistency
        unit_issues = self._check_unit_consistency(entities)
        warnings.extend(unit_issues)
        
        # Calculate overall confidence
        total_checks = len(entities) + len(relationships)
        violation_count = len(violations)
        confidence = max(0.0, 1.0 - (violation_count / max(total_checks, 1)))
        
        is_consistent = len(violations) == 0 and len(warnings) < len(entities) * 0.5
        
        return ConsistencyReport(
            is_consistent=is_consistent,
            violations=violations,
            warnings=warnings,
            suggestions=suggestions,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def infer_missing_properties(
        self,
        entity: Dict[str, Any],
        physics_model: str
    ) -> InferredProperties:
        """Infer missing properties using physics model.
        
        Args:
            entity: Entity with possibly missing properties
            physics_model: Physics model to use for inference
            
        Returns:
            InferredProperties with inferred values
        """
        entity_id = entity.get('id', 'unknown')
        entity_type = entity.get('type', '').lower()
        existing_props = entity.get('physics_properties', {})
        
        inferred_values = {}
        confidence = {}
        
        # Get domain for entity type
        domain_type = self.ENTITY_DOMAIN_MAP.get(entity_type)
        if domain_type is None:
            return InferredProperties(
                entity_id=entity_id,
                inferred_values={},
                confidence={},
                inference_method="none",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        domain = self.domains.get(domain_type.value)
        if domain is None:
            return InferredProperties(
                entity_id=entity_id,
                inferred_values={},
                confidence={},
                inference_method="none",
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Infer properties based on domain
        if domain_type == DomainType.FLUID_DYNAMICS:
            if 'density' in existing_props and 'velocity' in existing_props:
                # Infer momentum
                inferred_values['momentum'] = (
                    existing_props['density'] * existing_props['velocity']
                )
                confidence['momentum'] = 0.95
            
            if 'pressure' in existing_props and 'density' in existing_props:
                # Infer temperature (ideal gas law)
                gas_constant = domain.config.default_parameters.get('gas_constant', 287)
                inferred_values['temperature'] = (
                    existing_props['pressure'] / (existing_props['density'] * gas_constant)
                )
                confidence['temperature'] = 0.8
        
        elif domain_type == DomainType.STRUCTURAL_MECHANICS:
            if 'stress' in existing_props:
                # Infer strain from Hooke's law
                E = domain.config.default_parameters.get('youngs_modulus', 200e9)
                inferred_values['strain'] = existing_props['stress'] / E
                confidence['strain'] = 0.9
            
            if 'force' in existing_props and 'area' in existing_props:
                # Infer stress
                inferred_values['stress'] = existing_props['force'] / existing_props['area']
                confidence['stress'] = 0.85
        
        elif domain_type == DomainType.CHEMICAL_KINETICS:
            if 'concentration' in existing_props and 'rate_constant' in existing_props:
                # Infer reaction rate
                inferred_values['reaction_rate'] = (
                    existing_props['rate_constant'] * existing_props['concentration']
                )
                confidence['reaction_rate'] = 0.9
        
        return InferredProperties(
            entity_id=entity_id,
            inferred_values=inferred_values,
            confidence=confidence,
            inference_method=physics_model,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def simulate_system_behavior(
        self,
        system_entities: List[Dict[str, Any]],
        time_horizon: float,
        physics_params: Optional[Dict[str, Any]] = None
    ) -> SimulationResultKG:
        """Simulate behavior of system of entities.
        
        Args:
            system_entities: Entities forming the system
            time_horizon: Time to simulate
            physics_params: Additional physics parameters
            
        Returns:
            SimulationResultKG with simulation results
        """
        simulation_id = f"sim_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Create KG subgraph from entities
        kg_subgraph = {
            'entities': system_entities,
            'relationships': []  # Would be extracted from KG
        }
        
        # Convert to physics problem
        physics_problem = self.kg_to_physics_problem(kg_subgraph)
        
        # Get domain solver
        domain = self.domains.get(physics_problem.domain)
        if domain is None:
            return SimulationResultKG(
                simulation_id=simulation_id,
                success=False,
                solution_data=None,
                kg_updates=KGUpdates(),
                visualizations=[],
                metrics={},
                timestamp=datetime.now(timezone.utc).isoformat()
            )
        
        # Solve
        problem = {
            'type': 'transient',
            'time_span': (0, time_horizon),
            'initial_state': physics_problem.initial_conditions
        }
        
        result = domain.solve(problem, physics_params)
        
        # Convert to KG updates
        kg_updates = self.physics_solution_to_kg(result)
        
        # Create result
        sim_result = SimulationResultKG(
            simulation_id=simulation_id,
            success=result.success,
            solution_data=result.solution,
            kg_updates=kg_updates,
            visualizations=[],  # Would be generated
            metrics={
                'computation_time': result.computation_time,
                'constraint_satisfaction': sum(result.constraints_satisfied.values()) / max(len(result.constraints_satisfied), 1)
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Cache result
        self.simulation_cache[simulation_id] = sim_result
        
        return sim_result

    # Helper methods

    def _infer_domain(self, entities: List[Dict[str, Any]]) -> DomainType:
        """Infer physics domain from entities."""
        domain_counts = {}
        
        for entity in entities:
            entity_type = entity.get('type', '').lower()
            domain = self.ENTITY_DOMAIN_MAP.get(entity_type)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if domain_counts:
            return max(domain_counts.items(), key=lambda x: x[1])[0]
        
        return DomainType.FLUID_DYNAMICS  # Default

    def _extract_variables(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract physics variables from entities."""
        variables = {}
        
        for entity in entities:
            physics_props = entity.get('physics_properties', {})
            for key, value in physics_props.items():
                if isinstance(value, (int, float)):
                    variables[f"{entity.get('id')}_{key}"] = {
                        'value': value,
                        'entity': entity.get('id'),
                        'property': key
                    }
        
        return variables

    def _extract_parameters(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract physics parameters from entities."""
        parameters = {}
        
        for entity in entities:
            physics_props = entity.get('physics_properties', {})
            for key, value in physics_props.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    parameters[f"{entity.get('id')}_{key}"] = float(value)
        
        return parameters

    def _extract_equations(self, relationships: List[Dict[str, Any]], domain: DomainType) -> List[str]:
        """Extract physics equations from relationships."""
        equations = []
        
        for rel in relationships:
            rel_type = rel.get('type', '').lower()
            equation_template = self.RELATIONSHIP_CONSTRAINT_MAP.get(rel_type)
            if equation_template:
                equations.append(f"{equation_template}: {rel.get('from')} -> {rel.get('to')}")
        
        # Add default domain equations
        if domain == DomainType.FLUID_DYNAMICS:
            equations.extend([
                "continuity: ∇·v = 0",
                "navier_stokes: ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f"
            ])
        elif domain == DomainType.CLIMATE:
            equations.extend([
                "primitive_equations: atmospheric_dynamics",
                "thermodynamic_equation: energy_conservation"
            ])
        elif domain == DomainType.STRUCTURAL_MECHANICS:
            equations.extend([
                "equilibrium: ∇·σ + f = 0",
                "constitutive: σ = C:ε"
            ])
        
        return equations

    def _extract_initial_conditions(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract initial conditions from entities."""
        initial_conditions = {}
        
        for entity in entities:
            initial_state = entity.get('initial_state', {})
            if initial_state:
                initial_conditions[entity.get('id')] = initial_state
        
        return initial_conditions

    def _extract_boundary_conditions(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract boundary conditions from entities and relationships."""
        boundary_conditions = {
            'dirichlet': {},
            'neumann': {},
            'robin': {}
        }
        
        for entity in entities:
            if entity.get('type', '').lower() == 'boundary':
                bc_props = entity.get('physics_properties', {})
                bc_type = bc_props.get('bc_type', 'dirichlet')
                bc_value = bc_props.get('value')
                
                if bc_value is not None:
                    boundary_conditions[bc_type][entity.get('id')] = bc_value
        
        return boundary_conditions

    def _extract_constraints(self, relationships: List[Dict[str, Any]]) -> List[str]:
        """Extract physics constraints from relationships."""
        constraints = []
        
        for rel in relationships:
            rel_type = rel.get('type', '').lower()
            constraint = self.RELATIONSHIP_CONSTRAINT_MAP.get(rel_type)
            if constraint:
                constraints.append(constraint)
        
        return list(set(constraints))  # Remove duplicates

    def _extract_geometry(self, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract geometry from entities."""
        for entity in entities:
            if 'geometry' in entity:
                return entity['geometry']
        
        return None

    def _determine_problem_type(self, kg_subgraph: Dict[str, Any]) -> str:
        """Determine physics problem type from KG subgraph."""
        entities = kg_subgraph.get('entities', [])
        
        # Check for temporal markers
        for entity in entities:
            if 'time_dependent' in entity.get('tags', []):
                return 'transient'
            if 'steady_state' in entity.get('tags', []):
                return 'steady_state'
        
        return 'steady_state'  # Default

    def _get_required_properties(self, domain: DomainType) -> List[str]:
        """Get required physics properties for a domain."""
        required_props = {
            DomainType.FLUID_DYNAMICS: ['density', 'viscosity'],
            DomainType.CLIMATE: ['temperature', 'pressure'],
            DomainType.STRUCTURAL_MECHANICS: ['youngs_modulus', 'poisson_ratio'],
            DomainType.CHEMICAL_KINETICS: ['concentration', 'rate_constant'],
            DomainType.BIOLOGICAL_SYSTEMS: ['population', 'growth_rate']
        }
        
        return required_props.get(domain, [])

    def _check_conservation_laws(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> List[ConstraintViolation]:
        """Check if conservation laws are satisfied."""
        violations = []
        
        # Check mass conservation
        mass_in = 0
        mass_out = 0
        
        for rel in relationships:
            if rel.get('type') == 'flows_to':
                flow_rate = rel.get('properties', {}).get('flow_rate', 0)
                mass_out += flow_rate
                mass_in += flow_rate  # Simplified
        
        # In a closed system, mass should be conserved
        # This is a simplified check
        
        return violations

    def _check_causal_consistency(self, relationships: List[Dict[str, Any]]) -> List[str]:
        """Check causal consistency in relationships."""
        warnings = []
        
        # Check for cycles in causal relationships
        causal_rels = [r for r in relationships if r.get('type') == 'causes']
        # Simplified check - full implementation would use graph cycle detection
        
        return warnings

    def _check_unit_consistency(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Check unit consistency across entities."""
        warnings = []
        
        # Simplified unit check
        for entity in entities:
            props = entity.get('physics_properties', {})
            # Would check unit compatibility in full implementation
        
        return warnings

    def _create_temporal_entities(self, solution: SimulationResult) -> List[Dict[str, Any]]:
        """Create temporal entities from simulation result."""
        entities = []
        
        solution_data = solution.solution
        if not isinstance(solution_data, dict):
            return entities
        
        time_points = solution_data.get('time_points', [])
        
        for i, t in enumerate(time_points):
            entity = {
                'id': f"state_{i}",
                'type': 'system_state',
                'timestamp': t,
                'properties': {
                    'time_index': i,
                    'domain': solution.domain
                }
            }
            entities.append(entity)
        
        return entities

    def _extract_temporal_data(self, solution: SimulationResult) -> List[Dict[str, Any]]:
        """Extract temporal data from simulation."""
        temporal_data = []
        
        solution_data = solution.solution
        if not isinstance(solution_data, dict):
            return temporal_data
        
        # Extract trajectory data
        trajectory = solution_data.get('trajectory', [])
        for i, state in enumerate(trajectory):
            temporal_data.append({
                'time_index': i,
                'state': state,
                'domain': solution.domain
            })
        
        return temporal_data

    def _create_field_entities(self, solution: SimulationResult) -> List[Dict[str, Any]]:
        """Create field entities from simulation result."""
        entities = []
        
        solution_data = solution.solution
        if not isinstance(solution_data, dict):
            return entities
        
        # Create field entity for velocity
        if 'velocity_field' in solution_data:
            entities.append({
                'id': f"velocity_field_{solution.timestamp}",
                'type': 'vector_field',
                'field_type': 'velocity',
                'data': 'velocity_field_reference',
                'domain': solution.domain
            })
        
        # Create field entity for temperature
        if 'temperature_field' in solution_data:
            entities.append({
                'id': f"temperature_field_{solution.timestamp}",
                'type': 'scalar_field',
                'field_type': 'temperature',
                'data': 'temperature_field_reference',
                'domain': solution.domain
            })
        
        return entities

    def _create_physics_relationships(self, solution: SimulationResult) -> List[Dict[str, Any]]:
        """Create physics-based relationships from solution."""
        relationships = []
        
        # Create satisfies_constraint relationships
        for constraint, satisfied in solution.constraints_satisfied.items():
            relationships.append({
                'from': 'simulation',
                'to': constraint,
                'type': 'satisfies' if satisfied else 'violates',
                'properties': {'satisfied': satisfied}
            })
        
        return relationships

    def _create_property_updates(self, solution: SimulationResult) -> List[Dict[str, Any]]:
        """Create property updates from solution."""
        updates = []
        
        solution_data = solution.solution
        if not isinstance(solution_data, dict):
            return updates
        
        # Update computed metrics
        if 'max_stress' in solution_data:
            updates.append({
                'entity_id': 'structure',
                'property': 'max_stress',
                'value': solution_data['max_stress'],
                'source': 'simulation'
            })
        
        if 'max_displacement' in solution_data:
            updates.append({
                'entity_id': 'structure',
                'property': 'max_displacement',
                'value': solution_data['max_displacement'],
                'source': 'simulation'
            })
        
        return updates


__all__ = [
    "EntityPhysicsType",
    "RelationshipPhysicsType",
    "PhysicsProblem",
    "KGUpdates",
    "ConsistencyReport",
    "InferredProperties",
    "SimulationResultKG",
    "KGPhysicsBridge"
]
