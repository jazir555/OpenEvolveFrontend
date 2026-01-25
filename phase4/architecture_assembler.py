"""
Δ₁ Architecture Assembly System

Assembles validated RESE components from Phases I-III into complete architectures.

Author: Agent E1 (Δ₁ Specialist)
Created: 2025-12-31
Status: Implementation Phase
Dependencies:
    - rese.core.symbolic_constraint_engine (Constraint foundation)
    - rese.phase1.* (Φ₁.₅, Φ₂, Φ₃)
    - rese.phase2.* (Ψ₃, I_mech)
    - rese.phase3.* (Γ₁, Γ₂, Γ₃)
    - rese.gamma1.core.aci_calculator (ACI calculation)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set, Callable
from enum import Enum
import time
import hashlib
from datetime import datetime
from collections import defaultdict, deque
import json

# Try to import RESE components
try:
    from core.symbolic_constraint_engine import (
        SymbolicConstraintEngine, Constraint, ConstraintType
    )
except ImportError:
    SymbolicConstraintEngine = None
    Constraint = None
    ConstraintType = None

try:
    from gamma1.core.aci_calculator import ACICalculator, ACIResult
    from gamma1.core.csp_models import CSPInstance
except ImportError:
    ACICalculator = None
    ACIResult = None
    CSPInstance = None


# =============================================================================
# Data Structures
# =============================================================================

class AssemblyPattern(Enum):
    """Architecture assembly patterns"""
    SEQUENTIAL = "sequential"  # Linear pipeline
    PARALLEL = "parallel"  # Independent components
    HIERARCHICAL = "hierarchical"  # Nested components
    FEEDBACK = "feedback"  # Loops with convergence
    HYBRID = "hybrid"  # Mixed patterns


class ACIChange(Enum):
    """Expected ACI change from component"""
    DECREASE = "decrease"  # Reduces solvability
    NEUTRAL = "neutral"  # No change
    INCREASE = "increase"  # Improves solvability


class PhaseType(Enum):
    """RESE Phase types"""
    CORE = "core"
    PHASE_I = "phase_i"  # Epistemic Audit
    PHASE_II = "phase_ii"  # Isomorphic Resonance
    PHASE_III = "phase_iii"  # Monte Carlo Refinement


class SideEffect(Enum):
    """Component side effects"""
    READ_ONLY = "read_only"
    UPDATES_DATABASE = "updates_database"
    UPDATES_CACHE = "updates_cache"
    GENERATES_PROOF = "generates_proof"
    SENDS_TO_STAGE = "sends_to_stage"


@dataclass
class ComponentInterface:
    """
    Formal interface contract for RESE components
    """
    # Identification
    component_id: str
    component_name: str
    phase: PhaseType

    # Input/Output types
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)

    # Preconditions and postconditions
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)

    # Side effects
    side_effects: List[SideEffect] = field(default_factory=list)

    # Dependencies
    requires: List[str] = field(default_factory=list)  # Component IDs
    provides: List[str] = field(default_factory=list)  # Capabilities

    # ACI specifications
    min_input_aci: float = 0.0
    max_input_aci: float = 1.0
    expected_aci_change: ACIChange = ACIChange.NEUTRAL

    # Performance
    time_complexity: str = "O(n)"
    space_complexity: str = "O(n)"

    # Validation
    is_validated: bool = False
    validation_score: float = 0.0


@dataclass
class Architecture:
    """
    Complete assembled architecture
    """
    # Identification
    architecture_id: str
    name: str
    description: str

    # Components
    components: List[ComponentInterface] = field(default_factory=list)

    # Structure
    assembly_pattern: AssemblyPattern = AssemblyPattern.SEQUENTIAL
    connections: List[Tuple[str, str]] = field(default_factory=list)  # (from, to)

    # Dependencies
    dependency_layers: List[List[str]] = field(default_factory=list)

    # Validation
    validation_score: float = 0.0
    component_validations: Dict[str, float] = field(default_factory=dict)

    # ACI
    expected_aci_improvement: float = 0.0
    actual_aci_improvement: float = 0.0

    # Performance
    estimated_runtime: float = 0.0
    actual_runtime: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "delta1_assembler"
    version: str = "1.0"

    def add_component(self, component: ComponentInterface) -> bool:
        """Add component if compatible"""
        if self.is_compatible(component):
            self.components.append(component)
            self.component_validations[component.component_id] = component.validation_score
            return True
        return False

    def is_compatible(self, component: ComponentInterface) -> bool:
        """Check if component compatible with existing architecture"""
        # Check for duplicates
        if any(c.component_id == component.component_id for c in self.components):
            return False

        # Check dependencies satisfied
        component_ids = {c.component_id for c in self.components}
        if not all(dep in component_ids for dep in component.requires):
            return False

        return True

    def has_component(self, component_id: str) -> bool:
        """Check if component exists in architecture"""
        return any(c.component_id == component_id for c in self.components)

    def get_component(self, component_id: str) -> Optional[ComponentInterface]:
        """Get component by ID"""
        for c in self.components:
            if c.component_id == component_id:
                return c
        return None

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'architecture_id': self.architecture_id,
            'name': self.name,
            'description': self.description,
            'components': [c.component_id for c in self.components],
            'assembly_pattern': self.assembly_pattern.value,
            'validation_score': self.validation_score,
            'expected_aci_improvement': self.expected_aci_improvement,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class AssemblyResult:
    """
    Result of architecture assembly
    """
    architecture: Architecture
    success: bool
    message: str
    assembly_time: float = 0.0
    aci_score: float = 0.0

    # Diagnostics
    components_considered: int = 0
    components_added: int = 0
    components_rejected: int = 0

    # Validation
    is_validated: bool = False
    validation_score: float = 0.0


@dataclass
class AssemblyConfig:
    """
    Configuration for architecture assembly
    """
    # Assembly strategy
    strategy: str = "greedy"  # greedy, beam, mcts
    beam_width: int = 5
    mcts_iterations: int = 1000

    # ACI guidance
    target_aci: float = 0.8
    min_aci_improvement: float = 0.05

    # Validation
    require_validation: bool = True
    min_validation_score: float = 0.6

    # Performance
    max_components: int = 20
    max_assembly_time: float = 60.0

    # Debugging
    verbose: bool = False
    log_interval: int = 10


# =============================================================================
# Main Architecture Assembler
# =============================================================================

class ArchitectureAssembler:
    """
    Architecture Assembly System (Δ₁)

    Assembles validated RESE components into complete architectures.

    Usage:
        assembler = ArchitectureAssembler()
        result = assembler.assemble(components, problem)
        architecture = result.architecture
    """

    def __init__(
        self,
        config: AssemblyConfig = None,
        aci_calculator: ACICalculator = None
    ):
        """
        Initialize architecture assembler

        Args:
            config: Assembly configuration
            aci_calculator: ACI calculator for guidance
        """
        self.config = config or AssemblyConfig()
        self.aci_calculator = aci_calculator or (ACICalculator() if ACICalculator else None)

        # Component registry
        self.available_components: Dict[str, ComponentInterface] = {}

        # Assembly cache
        self.assembly_cache: Dict[str, Architecture] = {}

        # Statistics
        self.assemblies_created = 0
        self.assemblies_validated = 0

        # Register default components
        self._register_default_components()

    def _register_default_components(self):
        """Register default RESE components"""

        # Core: Symbolic Constraint Engine
        self.register_component(ComponentInterface(
            component_id="sce",
            component_name="Symbolic Constraint Engine",
            phase=PhaseType.CORE,
            input_types=["Problem"],
            output_types=["ConstraintSet"],
            preconditions=["problem is not None"],
            postconditions=["constraints are well-formed"],
            side_effects=[SideEffect.READ_ONLY],
            requires=[],
            provides=["constraint_management"],
            expected_aci_change=ACIChange.INCREASE,
            is_validated=True,
            validation_score=1.0  # Always validated
        ))

        # Phase I: Φ₁.₅ Tacit Assumption Miner
        self.register_component(ComponentInterface(
            component_id="phi15",
            component_name="Tacit Assumption Miner",
            phase=PhaseType.PHASE_I,
            input_types=["NullResult"],
            output_types=["TacitAssumption", "ParadigmShift"],
            preconditions=["len(null_results) >= 1"],
            postconditions=["all assumptions have confidence in [0,1]"],
            side_effects=[SideEffect.UPDATES_DATABASE],
            requires=["sce"],
            provides=["assumption_mining", "paradigm_shift_detection"],
            expected_aci_change=ACIChange.INCREASE,
            time_complexity="O(n log n)",
            is_validated=True,
            validation_score=0.75
        ))

        # Phase I: Φ₂ Cognitive Debiasing
        self.register_component(ComponentInterface(
            component_id="phi2",
            component_name="Cognitive Debiasing",
            phase=PhaseType.PHASE_I,
            input_types=["Problem"],
            output_types=["DebiasedProblem"],
            requires=["sce"],
            provides=["bias_detection", "bias_correction"],
            expected_aci_change=ACIChange.INCREASE,
            is_validated=False,  # Not implemented yet
            validation_score=0.0
        ))

        # Phase II: Ψ₃ Constraint Inversion
        self.register_component(ComponentInterface(
            component_id="psi3",
            component_name="Constraint Inversion",
            phase=PhaseType.PHASE_II,
            input_types=["ConstraintSet"],
            output_types=["InvertedConstraints"],
            requires=["sce"],
            provides=["constraint_inversion"],
            expected_aci_change=ACIChange.INCREASE,
            time_complexity="O(2^(n/10))",  # Exponential reduction
            is_validated=True,
            validation_score=0.80
        ))

        # Phase II: I_mech Isomorphism Validator
        self.register_component(ComponentInterface(
            component_id="imech",
            component_name="Isomorphism Validator",
            phase=PhaseType.PHASE_II,
            input_types=["Domain", "Domain"],
            output_types=["SimilarityResult", "TransferredSolution"],
            preconditions=["both domains have FDG", "source has solution"],
            postconditions=["similarity in [0,1]"],
            requires=["sce"],
            provides=["isomorphism_validation", "solution_transfer"],
            expected_aci_change=ACIChange.INCREASE,
            time_complexity="O(n^2) for subgraph, O(n) for exact",
            is_validated=True,
            validation_score=0.80
        ))

        # Phase III: Γ₁ ACI Analyzer
        self.register_component(ComponentInterface(
            component_id="gamma1",
            component_name="ACI Analyzer",
            phase=PhaseType.PHASE_III,
            input_types=["CSPInstance", "ProblemState"],
            output_types=["ACIResult"],
            preconditions=["csp is not None"],
            postconditions=["ACI in [0,1]", "confidence in [0,1]"],
            side_effects=[SideEffect.READ_ONLY],
            requires=[],
            provides=["aci_calculation", "solvability_assessment"],
            expected_aci_change=ACIChange.NEUTRAL,
            time_complexity="O(V + E)",
            is_validated=True,
            validation_score=0.85
        ))

        # Phase III: Γ₂ MCTS Search
        self.register_component(ComponentInterface(
            component_id="gamma2",
            component_name="MCTS Search",
            phase=PhaseType.PHASE_III,
            input_types=["Problem", "ACIResult"],
            output_types=["Solution"],
            requires=["gamma1"],
            provides=["mcts_search", "solution_finding"],
            expected_aci_change=ACIChange.INCREASE,
            time_complexity="O(iterations * branching_factor)",
            is_validated=True,
            validation_score=0.75
        ))

    def register_component(self, component: ComponentInterface):
        """Register a component for assembly"""
        self.available_components[component.component_id] = component

    def assemble(
        self,
        component_ids: List[str] = None,
        problem: Any = None,
        strategy: str = None
    ) -> AssemblyResult:
        """
        Assemble architecture from components

        Args:
            component_ids: List of component IDs to assemble (None = auto-select)
            problem: Target problem (for ACI guidance)
            strategy: Assembly strategy (greedy, beam, mcts)

        Returns:
            AssemblyResult with architecture and diagnostics
        """
        start_time = time.time()
        strategy = strategy or self.config.strategy

        # Determine components to use
        if component_ids is None:
            # Auto-select components using ACI guidance
            component_ids = self._select_components(problem)
        else:
            # Validate requested components
            for cid in component_ids:
                if cid not in self.available_components:
                    return AssemblyResult(
                        architecture=None,
                        success=False,
                        message=f"Unknown component: {cid}",
                        assembly_time=time.time() - start_time
                    )

        # Resolve dependencies
        try:
            ordered_ids = self._resolve_dependencies(component_ids)
        except ValueError as e:
            return AssemblyResult(
                architecture=None,
                success=False,
                message=f"Dependency resolution failed: {e}",
                assembly_time=time.time() - start_time
            )

        # Build architecture
        architecture = self._build_architecture(ordered_ids, problem)

        # Validate architecture
        if self.config.require_validation:
            validation_score = self._validate_architecture(architecture)
            architecture.validation_score = validation_score

            if validation_score < self.config.min_validation_score:
                return AssemblyResult(
                    architecture=architecture,
                    success=False,
                    message=f"Validation score too low: {validation_score:.2f}",
                    assembly_time=time.time() - start_time,
                    validation_score=validation_score
                )

        assembly_time = time.time() - start_time

        return AssemblyResult(
            architecture=architecture,
            success=True,
            message="Architecture assembled successfully",
            assembly_time=assembly_time,
            components_considered=len(component_ids),
            components_added=len(architecture.components),
            is_validated=True,
            validation_score=architecture.validation_score
        )

    def _select_components(self, problem: Any) -> List[str]:
        """
        Auto-select components using ACI guidance

        Strategy: Greedy selection to maximize ACI improvement
        """
        if problem is None:
            # Default: Use all validated components
            return [
                cid for cid, comp in self.available_components.items()
                if comp.is_validated and comp.validation_score >= 0.7
            ]

        # ACI-guided selection
        selected = []
        remaining = set(self.available_components.keys())

        # Start with core components
        for cid in ["sce", "gamma1"]:
            if cid in remaining:
                selected.append(cid)
                remaining.remove(cid)

        # Greedy selection
        current_aci = 0.0
        while current_aci < self.config.target_aci and remaining:
            best_component = None
            best_improvement = -float('inf')

            for cid in remaining:
                comp = self.available_components[cid]

                # Check if compatible
                if not all(dep in selected for dep in comp.requires):
                    continue

                # Estimate improvement
                if comp.expected_aci_change == ACIChange.INCREASE:
                    improvement = 0.15  # Estimated improvement
                else:
                    improvement = 0.0

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_component = cid

            if best_component:
                selected.append(best_component)
                remaining.remove(best_component)
                current_aci += best_improvement
            else:
                break

        return selected

    def _resolve_dependencies(self, component_ids: List[str]) -> List[str]:
        """
        Resolve component dependencies using topological sort

        Returns:
            List of component IDs in dependency order

        Raises:
            ValueError: If cyclic dependency detected
        """
        # Build dependency graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        all_components = set(component_ids)
        for cid in component_ids:
            if cid not in self.available_components:
                continue

            comp = self.available_components[cid]

            # Add dependencies
            for dep in comp.requires:
                if dep in self.available_components:
                    graph[dep].append(cid)
                    in_degree[cid] += 1
                    all_components.add(dep)

            # Ensure component is in graph
            if cid not in in_degree:
                in_degree[cid] = 0

        # Topological sort (Kahn's algorithm)
        queue = deque([cid for cid in all_components if in_degree[cid] == 0])
        result = []

        while queue:
            cid = queue.popleft()
            result.append(cid)

            for neighbor in graph[cid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(all_components):
            raise ValueError("Cyclic dependency detected")

        return result

    def _build_architecture(
        self,
        component_ids: List[str],
        problem: Any
    ) -> Architecture:
        """Build architecture from ordered component IDs"""

        arch_id = hashlib.sha256(
            "|".join(component_ids).encode()
        ).hexdigest()[:16]

        architecture = Architecture(
            architecture_id=f"arch_{arch_id}",
            name=f"Assembly_{self.assemblies_created}",
            description=f"Auto-assembled architecture with {len(component_ids)} components"
        )

        # Add components in dependency order
        for cid in component_ids:
            if cid in self.available_components:
                component = self.available_components[cid]
                architecture.add_component(component)

        # Determine assembly pattern
        architecture.assembly_pattern = self._determine_pattern(architecture)

        # Build dependency layers
        architecture.dependency_layers = self._build_layers(architecture)

        # Estimate ACI improvement
        architecture.expected_aci_improvement = self._estimate_aci_improvement(architecture)

        # Estimate runtime
        architecture.estimated_runtime = self._estimate_runtime(architecture)

        self.assemblies_created += 1

        return architecture

    def _determine_pattern(self, architecture: Architecture) -> AssemblyPattern:
        """Determine assembly pattern from component dependencies"""

        # Count dependencies
        total_deps = sum(len(c.requires) for c in architecture.components)

        # Check for feedback loops
        has_feedback = self._has_feedback_loops(architecture)

        if has_feedback:
            return AssemblyPattern.FEEDBACK
        elif total_deps == 0:
            return AssemblyPattern.PARALLEL
        elif total_deps == len(architecture.components) - 1:
            return AssemblyPattern.SEQUENTIAL
        else:
            return AssemblyPattern.HYBRID

    def _has_feedback_loops(self, architecture: Architecture) -> bool:
        """Check if architecture has feedback loops"""
        # Build graph
        component_ids = {c.component_id for c in architecture.components}
        graph = {}

        for comp in architecture.components:
            graph[comp.component_id] = [
                dep for dep in comp.requires
                if dep in component_ids
            ]

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        return any(dfs(cid) for cid in graph if cid not in visited)

    def _build_layers(self, architecture: Architecture) -> List[List[str]]:
        """Build dependency layers for parallel execution"""

        component_ids = {c.component_id for c in architecture.components}
        in_degree = {cid: 0 for cid in component_ids}

        # Calculate in-degrees
        for comp in architecture.components:
            for dep in comp.requires:
                if dep in component_ids:
                    in_degree[comp.component_id] += 1

        # Build layers
        layers = []
        remaining = set(component_ids)

        while remaining:
            # Find components with no dependencies within remaining
            ready = [
                cid for cid in remaining
                if all(dep not in remaining for dep in
                      self.available_components[cid].requires)
            ]

            if not ready:
                # Circular dependency - break cycle
                ready = [list(remaining)[0]]

            layers.append(ready)
            remaining -= set(ready)

        return layers

    def _estimate_aci_improvement(self, architecture: Architecture) -> float:
        """Estimate total ACI improvement from architecture"""

        total_improvement = 0.0

        for comp in architecture.components:
            if comp.expected_aci_change == ACIChange.INCREASE:
                # Base improvement on validation score
                improvement = 0.15 * comp.validation_score
                total_improvement += improvement

        # Cap at 1.0
        return min(total_improvement, 1.0)

    def _estimate_runtime(self, architecture: Architecture) -> float:
        """Estimate architecture runtime (simplified)"""

        # This is a rough estimate - actual runtime depends on problem
        base_time = 1.0  # Base time per component
        parallel_bonus = 0.7  # Parallel execution bonus

        total_time = 0.0
        for layer in architecture.dependency_layers:
            # Components in same layer can run in parallel
            layer_time = base_time * (1 + len(layer) * parallel_bonus)
            total_time += layer_time

        return total_time

    def _validate_architecture(self, architecture: Architecture) -> float:
        """
        Validate assembled architecture

        Returns:
            Validation score [0, 1]
        """
        # Check if architecture has any components
        if not architecture.components:
            return 0.0

        # Aggregate component validations (weighted average)
        total_weight = 0.0
        weighted_score = 0.0

        has_sce = False

        for comp in architecture.components:
            weight = 1.0
            if comp.phase == PhaseType.CORE:
                weight = 2.0  # Core components more important

            if comp.component_id == "sce":
                has_sce = True

            if comp.is_validated:
                weighted_score += weight * comp.validation_score
                total_weight += weight

        if total_weight == 0:
            return 0.0

        base_score = weighted_score / total_weight

        # Bonus for having constraint engine
        if has_sce:
            base_score = min(1.0, base_score + 0.1)

        # Bonus for ACI guidance
        if architecture.has_component("gamma1"):
            base_score += 0.05

        # Bonus for multiple phases
        phases_present = {c.phase for c in architecture.components}
        if len(phases_present) >= 3:
            base_score += 0.1

        return min(base_score, 1.0)

    def generate_fingerprint(self, architecture: Architecture) -> str:
        """Generate unique fingerprint for architecture"""

        sorted_components = sorted([c.component_id for c in architecture.components])
        fingerprint_str = "|".join([
            architecture.assembly_pattern.value,
            ",".join(sorted_components),
            str(architecture.expected_aci_improvement)
        ])

        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    def get_available_components(self) -> List[ComponentInterface]:
        """Get list of available components"""
        return list(self.available_components.values())

    def get_component(self, component_id: str) -> Optional[ComponentInterface]:
        """Get component interface by ID"""
        return self.available_components.get(component_id)


# =============================================================================
# Assembly Strategies
# =============================================================================

class BeamSearchAssembler(ArchitectureAssembler):
    """
    Beam search assembly strategy

    Maintains top-k partial architectures and expands all.
    """

    def assemble(
        self,
        component_ids: List[str] = None,
        problem: Any = None,
        strategy: str = None
    ) -> AssemblyResult:
        """Beam search assembly"""
        start_time = time.time()

        # Get candidate components
        if component_ids is None:
            component_ids = list(self.available_components.keys())

        # Initialize beam with empty architectures
        beam = [self._create_empty_arch()]

        for _ in range(len(component_ids)):
            candidates = []

            # Expand all architectures in beam
            for arch in beam:
                for cid in component_ids:
                    if cid not in [c.component_id for c in arch.components]:
                        # Try adding component
                        comp = self.available_components[cid]
                        if arch.is_compatible(comp):
                            new_arch = self._copy_architecture(arch)
                            new_arch.add_component(comp)
                            new_arch.dependency_layers = self._build_layers(new_arch)

                            # Score
                            score = self._score_architecture(new_arch)
                            candidates.append((score, new_arch))

            # Keep top-k
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = [arch for _, arch in candidates[:self.config.beam_width]]

        # Return best
        if not beam:
            return AssemblyResult(
                architecture=None,
                success=False,
                message="No valid architectures found",
                assembly_time=time.time() - start_time
            )

        best_arch = max(beam, key=lambda a: self._score_architecture(a))

        return AssemblyResult(
            architecture=best_arch,
            success=True,
            message="Beam search assembly complete",
            assembly_time=time.time() - start_time,
            validation_score=best_arch.validation_score
        )

    def _create_empty_arch(self) -> Architecture:
        """Create empty architecture"""
        return Architecture(
            architecture_id="empty",
            name="Empty",
            description="Empty architecture"
        )

    def _copy_architecture(self, arch: Architecture) -> Architecture:
        """Create copy of architecture"""
        new_arch = Architecture(
            architecture_id=f"{arch.architecture_id}_copy",
            name=arch.name,
            description=arch.description
        )
        new_arch.components = list(arch.components)
        new_arch.assembly_pattern = arch.assembly_pattern
        new_arch.dependency_layers = list(arch.dependency_layers)
        new_arch.validation_score = arch.validation_score
        new_arch.expected_aci_improvement = arch.expected_aci_improvement
        return new_arch

    def _score_architecture(self, arch: Architecture) -> float:
        """Score architecture for beam search"""
        # Combine validation and ACI improvement
        return 0.6 * arch.validation_score + 0.4 * arch.expected_aci_improvement


# =============================================================================
# Utility Functions
# =============================================================================

def are_compatible(
    comp1: ComponentInterface,
    comp2: ComponentInterface
) -> bool:
    """Check if two components are compatible"""

    # Check for circular dependencies
    if comp1.component_id in comp2.requires and comp2.component_id in comp1.requires:
        return False

    # Check type compatibility (simplified)
    # In full implementation, would check input/output type matching

    return True


def fingerprint_architecture(architecture: Architecture) -> str:
    """
    Generate unique fingerprint for architecture

    Convenience function that creates a temporary assembler
    """
    assembler = ArchitectureAssembler()
    return assembler.generate_fingerprint(architecture)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Demonstration
    print("=" * 60)
    print("Delta-1 Architecture Assembly System")
    print("=" * 60)

    # Create assembler
    assembler = ArchitectureAssembler()

    print("\nAvailable Components:")
    for comp in assembler.get_available_components():
        status = "[OK]" if comp.is_validated else "[--]"
        print(f"  {status} {comp.component_id:10s} - {comp.component_name}")

    # Assemble architecture
    print("\n" + "=" * 60)
    print("Assembling Architecture...")
    print("=" * 60)

    result = assembler.assemble(
        component_ids=None,  # Auto-select
        strategy="greedy"
    )

    if result.success:
        arch = result.architecture
        print(f"\n[SUCCESS] Assembly successful!")
        print(f"  Architecture ID: {arch.architecture_id}")
        print(f"  Components: {len(arch.components)}")
        print(f"  Pattern: {arch.assembly_pattern.value}")
        print(f"  Validation Score: {arch.validation_score:.2f}")
        print(f"  Expected ACI Improvement: {arch.expected_aci_improvement:.2f}")
        print(f"  Estimated Runtime: {arch.estimated_runtime:.2f}s")

        print("\nComponents:")
        for comp in arch.components:
            print(f"  - {comp.component_id} ({comp.component_name})")

        print("\nDependency Layers:")
        for i, layer in enumerate(arch.dependency_layers):
            print(f"  Layer {i}: {', '.join(layer)}")

        print(f"\nFingerprint: {assembler.generate_fingerprint(arch)}")
    else:
        print(f"\n[FAILED] Assembly failed: {result.message}")

    print("\n" + "=" * 60)
    print("Delta-1 Architecture Assembly Complete")
    print("=" * 60)
