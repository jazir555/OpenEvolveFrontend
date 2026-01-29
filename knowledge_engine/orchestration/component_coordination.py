"""
Component Coordination System

Enables intelligent coordination between components where:
- Components communicate results to each other
- Gaps in one component are filled by another
- Results are cross-validated between components
- Confidence scores are aggregated
- The best results are selected from multiple sources

Gap Coverage Strategy:
1. Each component declares its capabilities and gaps
2. The coordinator matches gap fillers to gaps
3. Results flow between components as needed
4. Cross-validation ensures quality
5. Final output is fused from multiple sources
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
import copy

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of capabilities components can have"""
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    GRAPH_CONSTRUCTION = "graph_construction"
    PATTERN_DETECTION = "pattern_detection"
    EMBEDDING_GENERATION = "embedding_generation"
    CAUSAL_DISCOVERY = "causal_discovery"
    TOPOLOGICAL_ANALYSIS = "topological_analysis"
    DOMAIN_SPECIFIC = "domain_specific"
    TEMPORAL_MODELING = "temporal_modeling"


class GapType(Enum):
    """Types of gaps components might have"""
    NO_CHEMISTRY = "no_chemistry"
    NO_TEMPORAL = "no_temporal"
    NO_CAUSAL = "no_causal"
    NO_TOPOLOGICAL = "no_topological"
    NO_DOMAIN_KNOWLEDGE = "no_domain_knowledge"
    LIMITED_SCALE = "limited_scale"
    HIGH_MEMORY = "high_memory"
    SLOW_EXECUTION = "slow_execution"


@dataclass
class ComponentCapabilities:
    """Capabilities and gaps of a component"""
    component_type: str
    
    # What it can do
    capabilities: Set[CapabilityType] = field(default_factory=set)
    
    # What it cannot do / needs help with
    gaps: Set[GapType] = field(default_factory=set)
    
    # Input/output formats
    input_formats: List[str] = field(default_factory=list)
    output_formats: List[str] = field(default_factory=list)
    
    # Quality metrics
    typical_accuracy: float = 0.0
    typical_speed_ms: float = 0.0
    
    # Dependencies on other components
    optimal_predecessors: List[str] = field(default_factory=list)
    provides_to: List[str] = field(default_factory=list)


@dataclass
class CoordinationContext:
    """Context for component coordination"""
    data_type: str
    domain: str
    input_data: Dict[str, Any]
    
    # Intermediate results
    component_results: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-validation results
    validation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Confidence scores
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Gap assignments
    gap_fillers: Dict[str, str] = field(default_factory=dict)


@dataclass
class GapFillingAssignment:
    """Assignment of a gap filler to a gap"""
    gap: GapType
    gap_owner: str  # Component with the gap
    filler: str  # Component filling the gap
    filler_confidence: float
    method: str  # How the gap is filled


class ComponentCapabilityRegistry:
    """Registry of component capabilities and gaps"""
    
    def __init__(self):
        self.capabilities: Dict[str, ComponentCapabilities] = {}
        self._initialize_default_capabilities()
    
    def _initialize_default_capabilities(self):
        """Initialize default capabilities for all components"""
        
        # DeepKE - Entity extraction
        self.capabilities['deepke'] = ComponentCapabilities(
            component_type='deepke',
            capabilities={
                CapabilityType.ENTITY_EXTRACTION,
                CapabilityType.RELATION_EXTRACTION
            },
            gaps={
                GapType.NO_CAUSAL,
                GapType.NO_TOPOLOGICAL,
                GapType.NO_CHEMISTRY
            },
            input_formats=['text'],
            output_formats=['entities', 'relations', 'triples'],
            typical_accuracy=0.85,
            typical_speed_ms=2000,
            provides_to=['kg_gen', 'karate_club', 'pami']
        )
        
        # KG-Gen - Graph construction
        self.capabilities['kg_gen'] = ComponentCapabilities(
            component_type='kg_gen',
            capabilities={
                CapabilityType.GRAPH_CONSTRUCTION
            },
            gaps={
                GapType.NO_ENTITY_EXTRACTION,
                GapType.NO_CAUSAL
            },
            input_formats=['entities', 'relations', 'text'],
            output_formats=['graph', 'nodes', 'edges'],
            typical_accuracy=0.80,
            typical_speed_ms=3000,
            optimal_predecessors=['deepke'],
            provides_to=['karate_club', 'neuralkg', 'pami']
        )
        
        # Karate Club - Graph analysis
        self.capabilities['karate_club'] = ComponentCapabilities(
            component_type='karate_club',
            capabilities={
                CapabilityType.PATTERN_DETECTION,
                CapabilityType.TOPOLOGICAL_ANALYSIS
            },
            gaps={
                GapType.NO_CAUSAL,
                GapType.NO_EMBEDDING_GENERATION
            },
            input_formats=['graph'],
            output_formats=['communities', 'centralities', 'patterns'],
            typical_accuracy=0.90,
            typical_speed_ms=1500,
            optimal_predecessors=['kg_gen'],
            provides_to=['neuralkg', 'lagrange_mapper']
        )
        
        # PAMI - Pattern mining
        self.capabilities['pami'] = ComponentCapabilities(
            component_type='pami',
            capabilities={
                CapabilityType.PATTERN_DETECTION
            },
            gaps={
                GapType.NO_GRAPH_CONSTRUCTION,
                GapType.NO_CAUSAL
            },
            input_formats=['transactions', 'entities'],
            output_formats=['patterns', 'rules', 'frequent_itemsets'],
            typical_accuracy=0.88,
            typical_speed_ms=5000,
            optimal_predecessors=['deepke'],
            provides_to=[]
        )
        
        # NeuralKG - Embeddings
        self.capabilities['neuralkg'] = ComponentCapabilities(
            component_type='neuralkg',
            capabilities={
                CapabilityType.EMBEDDING_GENERATION
            },
            gaps={
                GapType.NO_CAUSAL,
                GapType.HIGH_MEMORY,
                GapType.SLOW_EXECUTION
            },
            input_formats=['triples', 'graph'],
            output_formats=['embeddings', 'vectors'],
            typical_accuracy=0.82,
            typical_speed_ms=10000,
            optimal_predecessors=['kg_gen', 'karate_club'],
            provides_to=['lagrange_mapper']
        )
        
        # Causal-Learn - Causal discovery
        self.capabilities['causal_learn'] = ComponentCapabilities(
            component_type='causal_learn',
            capabilities={
                CapabilityType.CAUSAL_DISCOVERY
            },
            gaps={
                GapType.NO_ENTITY_EXTRACTION,
                GapType.NO_GRAPH_CONSTRUCTION,
                GapType.HIGH_MEMORY
            },
            input_formats=['data_matrix', 'time_series'],
            output_formats=['causal_graph', 'causal_relations'],
            typical_accuracy=0.75,
            typical_speed_ms=8000,
            optimal_predecessors=['deepke'],
            provides_to=[]
        )
        
        # Lagrange-Mapper - Topological analysis
        self.capabilities['lagrange_mapper'] = ComponentCapabilities(
            component_type='lagrange_mapper',
            capabilities={
                CapabilityType.TOPOLOGICAL_ANALYSIS
            },
            gaps={
                GapType.NO_EMBEDDING_GENERATION,
                GapType.NO_CAUSAL
            },
            input_formats=['embeddings', 'vectors'],
            output_formats=['attractors', 'landscapes', 'clusters'],
            typical_accuracy=0.85,
            typical_speed_ms=4000,
            optimal_predecessors=['neuralkg'],
            provides_to=[]
        )
        
        # GlobalChem - Chemistry
        self.capabilities['global_chem'] = ComponentCapabilities(
            component_type='global_chem',
            capabilities={
                CapabilityType.ENTITY_EXTRACTION,
                CapabilityType.DOMAIN_SPECIFIC
            },
            gaps={
                GapType.NO_CAUSAL,
                GapType.NO_TOPOLOGICAL,
                GapType.LIMITED_SCALE
            },
            input_formats=['text', 'smiles'],
            output_formats=['chemical_entities', 'compounds'],
            typical_accuracy=0.92,
            typical_speed_ms=1000,
            provides_to=['deepke', 'kg_gen']
        )
        
        # Neuromancer - Temporal modeling
        self.capabilities['neuromancer'] = ComponentCapabilities(
            component_type='neuromancer',
            capabilities={
                CapabilityType.TEMPORAL_MODELING
            },
            gaps={
                GapType.NO_ENTITY_EXTRACTION,
                GapType.HIGH_MEMORY,
                GapType.SLOW_EXECUTION
            },
            input_formats=['time_series', 'temporal_data'],
            output_formats=['models', 'predictions', 'dynamics'],
            typical_accuracy=0.78,
            typical_speed_ms=15000,
            optimal_predecessors=['deepke'],
            provides_to=[]
        )
    
    def get_capabilities(self, component: str) -> Optional[ComponentCapabilities]:
        """Get capabilities for a component"""
        return self.capabilities.get(component)
    
    def find_gap_fillers(self, gap: GapType) -> List[Tuple[str, float]]:
        """
        Find components that can fill a specific gap.
        
        Returns list of (component, confidence) tuples sorted by confidence.
        """
        fillers = []
        
        for comp_type, caps in self.capabilities.items():
            if gap == GapType.NO_CHEMISTRY and CapabilityType.DOMAIN_SPECIFIC in caps.capabilities:
                if comp_type == 'global_chem':
                    fillers.append((comp_type, 1.0))
            
            elif gap == GapType.NO_CAUSAL and CapabilityType.CAUSAL_DISCOVERY in caps.capabilities:
                fillers.append((comp_type, caps.typical_accuracy))
            
            elif gap == GapType.NO_TOPOLOGICAL and CapabilityType.TOPOLOGICAL_ANALYSIS in caps.capabilities:
                fillers.append((comp_type, caps.typical_accuracy))
            
            elif gap == GapType.NO_TEMPORAL and CapabilityType.TEMPORAL_MODELING in caps.capabilities:
                fillers.append((comp_type, caps.typical_accuracy))
            
            elif gap == GapType.NO_ENTITY_EXTRACTION and CapabilityType.ENTITY_EXTRACTION in caps.capabilities:
                fillers.append((comp_type, caps.typical_accuracy))
            
            elif gap == GapType.NO_EMBEDDING_GENERATION and CapabilityType.EMBEDDING_GENERATION in caps.capabilities:
                fillers.append((comp_type, caps.typical_accuracy))
        
        # Sort by confidence
        fillers.sort(key=lambda x: x[1], reverse=True)
        return fillers
    
    def find_optimal_sequence(self, required_capabilities: Set[CapabilityType]) -> List[str]:
        """
        Find optimal component sequence for required capabilities.
        
        Args:
            required_capabilities: Set of capabilities needed
            
        Returns:
            List of component types in optimal order
        """
        sequence = []
        remaining = required_capabilities.copy()
        used_components = set()
        
        while remaining:
            best_component = None
            best_coverage = 0
            best_predecessors_match = 0
            
            for comp_type, caps in self.capabilities.items():
                if comp_type in used_components:
                    continue
                
                # Check coverage of remaining capabilities
                coverage = len(remaining & caps.capabilities)
                
                # Check predecessor match
                pred_match = sum(1 for p in caps.optimal_predecessors if p in used_components)
                
                if coverage > best_coverage or (coverage == best_coverage and pred_match > best_predecessors_match):
                    best_coverage = coverage
                    best_component = comp_type
                    best_predecessors_match = pred_match
            
            if best_component is None or best_coverage == 0:
                break
            
            sequence.append(best_component)
            used_components.add(best_component)
            remaining -= self.capabilities[best_component].capabilities
        
        return sequence


class ComponentCoordinator:
    """
    Coordinates component execution with gap filling and cross-validation.
    
    Responsibilities:
    - Match components to fill each other's gaps
    - Route data between components optimally
    - Cross-validate results from multiple components
    - Fuse results for best quality
    """
    
    def __init__(self, learning_engine=None):
        """
        Initialize the coordinator.
        
        Args:
            learning_engine: Optional learning engine for adaptive coordination
        """
        self.capability_registry = ComponentCapabilityRegistry()
        self.learning_engine = learning_engine
        
        # Coordination state
        self.coordination_history = []
        
        logger.info({
            "msg": "ComponentCoordinator initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def coordinate_pipeline(self, components: List[str], 
                           input_data: Dict[str, Any],
                           data_type: str,
                           domain: str) -> Dict[str, Any]:
        """
        Coordinate a multi-component pipeline with gap filling.
        
        Args:
            components: List of component types to coordinate
            input_data: Input data
            data_type: Type of data
            domain: Domain
            
        Returns:
            Coordination plan with gap fillers and routing
        """
        context = CoordinationContext(
            data_type=data_type,
            domain=domain,
            input_data=input_data
        )
        
        # Identify gaps in components
        gaps = self._identify_gaps(components)
        
        # Find gap fillers
        gap_assignments = self._assign_gap_fillers(gaps)
        context.gap_fillers = {a.gap.value: a.filler for a in gap_assignments}
        
        # Build coordination plan
        plan = {
            'primary_components': components,
            'gap_assignments': [
                {
                    'gap': a.gap.value,
                    'component_with_gap': a.gap_owner,
                    'filler_component': a.filler,
                    'confidence': a.filler_confidence,
                    'method': a.method
                }
                for a in gap_assignments
            ],
            'data_routing': self._build_data_routing(components, gap_assignments),
            'cross_validation_points': self._identify_validation_points(components),
            'expected_confidence': self._calculate_expected_confidence(
                components, gap_assignments
            )
        }
        
        return plan
    
    def _identify_gaps(self, components: List[str]) -> List[Tuple[str, GapType]]:
        """Identify gaps in the selected components"""
        gaps = []
        
        for comp in components:
            caps = self.capability_registry.get_capabilities(comp)
            if caps:
                for gap in caps.gaps:
                    gaps.append((comp, gap))
        
        return gaps
    
    def _assign_gap_fillers(self, gaps: List[Tuple[str, GapType]]) -> List[GapFillingAssignment]:
        """Assign components to fill identified gaps"""
        assignments = []
        
        for component, gap in gaps:
            # Find potential fillers
            fillers = self.capability_registry.find_gap_fillers(gap)
            
            if fillers:
                # Select best filler
                best_filler, confidence = fillers[0]
                
                # Determine method
                method = self._determine_filling_method(gap, best_filler)
                
                assignment = GapFillingAssignment(
                    gap=gap,
                    gap_owner=component,
                    filler=best_filler,
                    filler_confidence=confidence,
                    method=method
                )
                
                assignments.append(assignment)
        
        return assignments
    
    def _determine_filling_method(self, gap: GapType, filler: str) -> str:
        """Determine how a gap should be filled"""
        method_map = {
            GapType.NO_CHEMISTRY: f"{filler}_recognizes_chemical_entities",
            GapType.NO_CAUSAL: f"{filler}_discovers_causal_relations",
            GapType.NO_TOPOLOGICAL: f"{filler}_analyzes_structure",
            GapType.NO_ENTITY_EXTRACTION: f"{filler}_extracts_entities",
            GapType.NO_EMBEDDING_GENERATION: f"{filler}_generates_embeddings",
            GapType.NO_TEMPORAL: f"{filler}_models_dynamics",
        }
        
        return method_map.get(gap, f"{filler}_provides_capability")
    
    def _build_data_routing(self, components: List[str],
                           gap_assignments: List[GapFillingAssignment]) -> Dict[str, Any]:
        """Build optimal data routing between components"""
        routing = {
            'stages': [],
            'data_flows': []
        }
        
        # Build stages based on dependencies
        executed = set()
        remaining = set(components + [a.filler for a in gap_assignments])
        
        stage_num = 0
        while remaining:
            stage_num += 1
            stage_components = []
            
            for comp in list(remaining):
                caps = self.capability_registry.get_capabilities(comp)
                if caps:
                    # Check if all predecessors are executed
                    preds = set(caps.optimal_predecessors)
                    if preds <= executed or not preds:
                        stage_components.append(comp)
                        executed.add(comp)
                        remaining.remove(comp)
            
            if not stage_components:
                # Break deadlock - add remaining anyway
                stage_components = list(remaining)
                executed.update(remaining)
                remaining.clear()
            
            routing['stages'].append({
                'stage': stage_num,
                'components': stage_components
            })
        
        # Define data flows
        for comp in components:
            caps = self.capability_registry.get_capabilities(comp)
            if caps:
                for pred in caps.optimal_predecessors:
                    if pred in components:
                        routing['data_flows'].append({
                            'from': pred,
                            'to': comp,
                            'data_type': 'output_to_input'
                        })
        
        return routing
    
    def _identify_validation_points(self, components: List[str]) -> List[Dict[str, Any]]:
        """Identify points where cross-validation can occur"""
        validation_points = []
        
        # Find components with overlapping capabilities
        capability_components = {}
        for comp in components:
            caps = self.capability_registry.get_capabilities(comp)
            if caps:
                for cap in caps.capabilities:
                    if cap not in capability_components:
                        capability_components[cap] = []
                    capability_components[cap].append(comp)
        
        # Create validation points for overlapping capabilities
        for cap, comps in capability_components.items():
            if len(comps) > 1:
                validation_points.append({
                    'capability': cap.value,
                    'components': comps,
                    'validation_type': 'result_comparison'
                })
        
        return validation_points
    
    def _calculate_expected_confidence(self, components: List[str],
                                      gap_assignments: List[GapFillingAssignment]) -> float:
        """Calculate expected confidence for the coordination plan"""
        if not components:
            return 0.0
        
        # Base confidence from component accuracy
        base_confidences = []
        for comp in components:
            caps = self.capability_registry.get_capabilities(comp)
            if caps:
                base_confidences.append(caps.typical_accuracy)
        
        if not base_confidences:
            return 0.0
        
        avg_base = sum(base_confidences) / len(base_confidences)
        
        # Adjust for gap fillers
        gap_penalty = 0.0
        for assignment in gap_assignments:
            # Small penalty for needing gap filler
            gap_penalty += 0.05 * (1 - assignment.filler_confidence)
        
        return max(0.0, min(1.0, avg_base - gap_penalty))
    
    def cross_validate_results(self, component_results: Dict[str, Any],
                               validation_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cross-validate results from multiple components.
        
        Args:
            component_results: Results from each component
            validation_points: Points to validate
            
        Returns:
            Validation report with confidence scores
        """
        validation_report = {
            'validations': [],
            'overall_confidence': 0.0,
            'inconsistencies': []
        }
        
        for point in validation_points:
            capability = point['capability']
            components = point['components']
            
            # Get results for this capability
            results = {}
            for comp in components:
                if comp in component_results:
                    results[comp] = component_results[comp]
            
            if len(results) < 2:
                continue
            
            # Compare results
            comparison = self._compare_results(results, capability)
            
            validation_report['validations'].append({
                'capability': capability,
                'components': list(results.keys()),
                'agreement_score': comparison['agreement_score'],
                'consistency': comparison['consistent']
            })
            
            if not comparison['consistent']:
                validation_report['inconsistencies'].append({
                    'capability': capability,
                    'details': comparison['differences']
                })
        
        # Calculate overall confidence
        if validation_report['validations']:
            scores = [v['agreement_score'] for v in validation_report['validations']]
            validation_report['overall_confidence'] = sum(scores) / len(scores)
        
        return validation_report
    
    def _compare_results(self, results: Dict[str, Any], 
                        capability: str) -> Dict[str, Any]:
        """Compare results from multiple components for the same capability"""
        # Simplified comparison - would be more sophisticated in production
        
        result_sizes = {}
        for comp, result in results.items():
            if isinstance(result, dict):
                result_sizes[comp] = len(str(result))
            elif isinstance(result, list):
                result_sizes[comp] = len(result)
            else:
                result_sizes[comp] = 1
        
        if not result_sizes:
            return {'agreement_score': 0.0, 'consistent': False, 'differences': 'no_data'}
        
        # Check if sizes are similar (simple heuristic)
        sizes = list(result_sizes.values())
        if len(sizes) < 2:
            return {'agreement_score': 1.0, 'consistent': True, 'differences': []}
        
        avg_size = sum(sizes) / len(sizes)
        max_deviation = max(abs(s - avg_size) for s in sizes) / avg_size if avg_size > 0 else 0
        
        # Agreement score decreases with deviation
        agreement_score = max(0.0, 1.0 - max_deviation)
        
        return {
            'agreement_score': agreement_score,
            'consistent': agreement_score > 0.7,
            'differences': {
                'size_deviation': max_deviation,
                'component_sizes': result_sizes
            }
        }
    
    def fuse_results(self, component_results: Dict[str, Any],
                    validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse results from multiple components into unified output.
        
        Uses validation scores to weight contributions.
        """
        fused = {
            'sources': list(component_results.keys()),
            'confidence': validation_report.get('overall_confidence', 0.5),
            'results': {}
        }
        
        # Simple fusion - merge all results
        for comp, result in component_results.items():
            if isinstance(result, dict):
                for key, value in result.items():
                    if key not in fused['results']:
                        fused['results'][key] = []
                    fused['results'][key].append({
                        'source': comp,
                        'value': value
                    })
        
        # For each key, select best result or merge
        for key, values in fused['results'].items():
            if len(values) == 1:
                fused['results'][key] = values[0]['value']
            else:
                # Multiple sources - merge or select best
                fused['results'][key] = self._merge_values(values)
        
        return fused
    
    def _merge_values(self, values: List[Dict[str, Any]]) -> Any:
        """Merge values from multiple sources"""
        # For lists, concatenate and deduplicate
        all_items = []
        for v in values:
            val = v['value']
            if isinstance(val, list):
                all_items.extend(val)
            else:
                all_items.append(val)
        
        # Simple deduplication for string items
        if all_items and isinstance(all_items[0], str):
            return list(set(all_items))
        
        return all_items
    
    def get_coordination_report(self) -> Dict[str, Any]:
        """Get report on coordination activities"""
        return {
            'registered_components': len(self.capability_registry.capabilities),
            'coordination_history': len(self.coordination_history),
            'capability_coverage': self._analyze_capability_coverage()
        }
    
    def _analyze_capability_coverage(self) -> Dict[str, int]:
        """Analyze coverage of capabilities across components"""
        coverage = {}
        
        for cap in CapabilityType:
            count = sum(
                1 for c in self.capability_registry.capabilities.values()
                if cap in c.capabilities
            )
            coverage[cap.value] = count
        
        return coverage


# Convenience function for gap analysis
def analyze_pipeline_gaps(components: List[str]) -> Dict[str, Any]:
    """
    Analyze gaps in a proposed pipeline.
    
    Args:
        components: List of component types
        
    Returns:
        Gap analysis report
    """
    registry = ComponentCapabilityRegistry()
    coordinator = ComponentCoordinator()
    
    gaps = coordinator._identify_gaps(components)
    assignments = coordinator._assign_gap_fillers(gaps)
    
    return {
        'components': components,
        'gaps_identified': [
            {
                'component': g[0],
                'gap': g[1].value
            }
            for g in gaps
        ],
        'gap_fillers': [
            {
                'gap': a.gap.value,
                'filled_by': a.filler,
                'confidence': a.filler_confidence
            }
            for a in assignments
        ],
        'recommendations': [
            f"Add {a.filler} to fill {a.gap.value} gap in {a.gap_owner}"
            for a in assignments
        ]
    }
