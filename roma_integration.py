"""
ROMA Integration Module

Reasoning on Modular Architectures integration.

Author: OpenEvolve Team
Date: 2026-02-06
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class ROMAIntegration:
    """ROMA Integration class"""

    def __init__(self):
        logger.info("ROMA Integration initialized")
        self.components = {}
        self.integration_history = []
        self.entanglement_matrix = {}

    def integrate(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Integrate components using ROMA's associative integration methodology.
        
        Args:
            components: List of components to integrate
            
        Returns:
            Dict with integration results including success status, 
            integrated components, dependencies, and entanglement metrics
        """
        logger.info(f"Starting ROMA integration for {len(components)} components")
        
        try:
            # Validate input components
            if not components:
                return {
                    "success": False,
                    "error": "No components provided for integration",
                    "integrated_components": [],
                    "dependencies": {},
                    "entanglement_score": 0.0
                }
            
            # Process each component and build integration plan
            processed_components = []
            dependencies = {}
            entanglements = {}
            
            for i, component in enumerate(components):
                # Generate unique ID for component if not present
                if "id" not in component:
                    component_id = f"comp_{hashlib.md5(json.dumps(component, sort_keys=True).encode()).hexdigest()[:8]}"
                    component["id"] = component_id
                else:
                    component_id = component["id"]
                
                # Analyze component for dependencies and entanglements
                deps, entanglements_for_comp = self._analyze_component_relationships(component, components)
                dependencies[component_id] = deps
                entanglements[component_id] = entanglements_for_comp
                
                # Process component
                processed_component = self._process_component(component)
                processed_components.append(processed_component)
                
                # Store component in internal registry
                self.components[component_id] = processed_component
            
            # Build execution order based on dependencies
            execution_order = self._build_execution_order(dependencies)
            
            # Calculate entanglement score
            entanglement_score = self._calculate_entanglement_score(entanglements)
            
            # Create integration result
            integration_result = {
                "success": True,
                "integrated_components": processed_components,
                "dependencies": dependencies,
                "entanglements": entanglements,
                "execution_order": execution_order,
                "entanglement_score": entanglement_score,
                "total_components": len(processed_components),
                "integration_timestamp": datetime.utcnow().isoformat(),
                "message": f"Successfully integrated {len(processed_components)} components"
            }
            
            # Add to integration history
            self.integration_history.append(integration_result)
            
            logger.info(f"ROMA integration completed successfully with {len(processed_components)} components")
            return integration_result
            
        except Exception as e:
            logger.error(f"ROMA integration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "integrated_components": [],
                "dependencies": {},
                "entanglement_score": 0.0,
                "message": f"Integration failed: {str(e)}"
            }
    
    def _analyze_component_relationships(self, component: Dict[str, Any], all_components: List[Dict[str, Any]]) -> tuple:
        """Analyze component for dependencies and entanglements with other components."""
        dependencies = []
        entanglements = []
        
        # Extract component properties for analysis
        comp_props = component.get("properties", {})
        comp_interfaces = component.get("interfaces", [])
        comp_requirements = component.get("requirements", [])
        
        # Check for dependencies based on requirements
        for req in comp_requirements:
            for other_comp in all_components:
                if other_comp != component:
                    other_outputs = other_comp.get("outputs", [])
                    if req in other_outputs:
                        dependencies.append(other_comp.get("id", f"unknown_{hash(str(other_comp)) % 10000}"))
        
        # Check for entanglements based on shared interfaces or properties
        for other_comp in all_components:
            if other_comp != component:
                other_interfaces = other_comp.get("interfaces", [])
                other_props = other_comp.get("properties", {})
                
                # Interface entanglement
                shared_interfaces = set(comp_interfaces) & set(other_interfaces)
                if shared_interfaces:
                    entanglements.append({
                        "with": other_comp.get("id", f"unknown_{hash(str(other_comp)) % 10000}"),
                        "type": "interface",
                        "strength": len(shared_interfaces) / max(len(comp_interfaces), len(other_interfaces), 1)
                    })
                
                # Property entanglement
                shared_props = set(comp_props.keys()) & set(other_props.keys())
                if shared_props:
                    entanglements.append({
                        "with": other_comp.get("id", f"unknown_{hash(str(other_comp)) % 10000}"),
                        "type": "property",
                        "strength": len(shared_props) / max(len(comp_props), len(other_props), 1)
                    })
        
        return dependencies, entanglements
    
    def _process_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance a component with ROMA-specific metadata."""
        processed = component.copy()
        
        # Add ROMA-specific metadata
        processed["metadata"] = processed.get("metadata", {})
        processed["metadata"]["roma_processed"] = True
        processed["metadata"]["processing_timestamp"] = datetime.utcnow().isoformat()
        
        # Calculate component complexity
        complexity_score = self._calculate_component_complexity(component)
        processed["metadata"]["complexity_score"] = complexity_score
        
        # Add entanglement awareness
        processed["metadata"]["entanglement_aware"] = True
        
        return processed
    
    def _calculate_component_complexity(self, component: Dict[str, Any]) -> float:
        """Calculate complexity score for a component."""
        # Base complexity on number of properties, interfaces, and requirements
        props_count = len(component.get("properties", {}))
        interfaces_count = len(component.get("interfaces", []))
        requirements_count = len(component.get("requirements", []))
        dependencies_count = len(component.get("dependencies", []))
        
        # Normalize to 0-1 scale
        total_elements = props_count + interfaces_count + requirements_count + dependencies_count
        complexity = min(1.0, total_elements / 50.0)  # Scale factor of 50 elements = 1.0 complexity
        
        return complexity
    
    def _build_execution_order(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Build execution order based on dependency graph using topological sort."""
        # Build adjacency list and in-degree count
        all_nodes = set(dependencies.keys())
        for deps in dependencies.values():
            all_nodes.update(deps)
        
        # Initialize in-degrees
        in_degree = {node: 0 for node in all_nodes}
        for node, deps in dependencies.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 1
        
        # Topological sort using Kahn's algorithm
        from collections import deque
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        order = []
        
        while queue:
            current = queue.popleft() # Optimized O(1) removal
            order.append(current)
            
            # Find nodes that depend on current
            for node, deps in dependencies.items():
                if current in deps:
                    in_degree[node] -= 1
                    if in_degree[node] == 0:
                        queue.append(node)
        
        # Check for cycles
        if len(order) != len(all_nodes):
            logger.warning("Cycle detected in dependency graph")
            # Return all nodes if cycle detected
            return list(all_nodes)
        
        return order
    
    def _calculate_entanglement_score(self, entanglements: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate overall entanglement score from component entanglements."""
        if not entanglements:
            return 0.0
        
        total_strength = 0.0
        total_connections = 0
        
        for comp_id, ents in entanglements.items():
            for ent in ents:
                total_strength += ent.get("strength", 0.0)
                total_connections += 1
        
        if total_connections == 0:
            return 0.0
        
        # Normalize to 0-1 scale
        avg_strength = total_strength / total_connections
        return min(1.0, avg_strength * 2.0)  # Boost slightly to reflect entanglement importance
    
    def _calculate_entanglement_strength(self, relationship_type: str) -> float:
        """Calculate entanglement strength based on relationship type."""
        # Define relationship type strengths
        strength_map = {
            "dependency": 0.9,
            "part_of": 0.8,
            "contains": 0.8,
            "inherits": 0.7,
            "uses": 0.6,
            "related_to": 0.5,
            "similar_to": 0.4,
            "connected_to": 0.3,
            "associated_with": 0.2
        }
        
        return strength_map.get(relationship_type, 0.1)
