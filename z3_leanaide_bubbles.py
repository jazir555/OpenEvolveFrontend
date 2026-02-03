"""
Z3 and LeanAIDE Bubbles for BubbleLab

This module provides BubbleLab workflow nodes (bubbles) for Z3 SMT solving
and LeanAIDE theorem proving operations. These bubbles enable visualization
and control of formal verification workflows through the BubbleLabs UI.

Features:
- Z3 constraint solving bubbles
- Z3 theorem proving bubbles  
- LeanAIDE proof visualization bubbles
- Cross-verification bubbles
- Sub-problem loop bubbles for entangled workflows
- Entanglement matrix integration and visualization
- Flexible workflow builder for arbitrary patterns

Usage:
    from z3_leanaide_bubbles import (
        create_z3_solver_bubble,
        create_z3_prover_bubble,
        create_leanaide_proof_bubble,
        create_cross_verification_bubble,
        create_z3_workflow,
        create_subproblem_loop_bubble,
        create_entanglement_visualization_bubble,
        create_z3_workflow_with_entanglement
    )
"""

import uuid
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Bubble Configuration Constants
# =============================================================================

Z3_NODE_POSITIONS = {
    "input": {"x": 0, "y": 0},
    "classification": {"x": 150, "y": 0},
    "z3_solver": {"x": 300, "y": 0},
    "z3_prover": {"x": 300, "y": 100},
    "leanaide_proof": {"x": 500, "y": 0},
    "cross_verify": {"x": 650, "y": 0},
    "result": {"x": 800, "y": 0},
}

Z3_NODE_COLORS = {
    "z3_solver": "#FF6B6B",
    "z3_prover": "#E17055",
    "leanaide_proof": "#00B894",
    "cross_verify": "#6C5CE7",
    "classification": "#0984E3",
    "input": "#74B9FF",
    "result": "#00B894",
    "subproblem_loop": "#FDCB6E",
    "entanglement_viz": "#E84393",
    "subproblem": "#81ECEC",
    "super_node": "#A29BFE",
}

Z3_NODE_ICONS = {
    "z3_solver": "🔐",
    "z3_prover": "📐",
    "leanaide_proof": "📚",
    "cross_verify": "⚖️",
    "classification": "🔍",
    "input": "📥",
    "result": "✅",
    "subproblem_loop": "🔄",
    "entanglement_viz": "🕸️",
    "subproblem": "📦",
    "super_node": "🔗",
}


# =============================================================================
# Bubble Data Classes
# =============================================================================

@dataclass
class Z3SolverBubbleConfig:
    """Configuration for a Z3 constraint solver bubble."""
    problem_text: str
    variables: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    timeout_seconds: int = 30
    strategy: str = "auto"


@dataclass
class Z3ProverBubbleConfig:
    """Configuration for a Z3 theorem prover bubble."""
    theorem_statement: str
    proof_strategy: str = "default"
    timeout_seconds: int = 60


@dataclass
class LeanAideProofBubbleConfig:
    """Configuration for a LeanAIDE proof visualization bubble."""
    theorem_name: str
    proof_type: str = "theorem"  # theorem, definition, lemma
    mcts_enabled: bool = True
    timeout_seconds: int = 120


@dataclass
class CrossVerificationBubbleConfig:
    """Configuration for cross-verification bubble."""
    problem_text: str
    z3_strategy: str = "adaptive"
    lean_strategy: str = "auto"
    timeout_seconds: int = 60


@dataclass
class ProblemClassificationBubbleConfig:
    """Configuration for problem classification bubble."""
    problem_text: str
    auto_classify: bool = True


@dataclass
class SubProblemLoopBubbleConfig:
    """Configuration for sub-problem loop bubble (handles entangled sub-problems)."""
    sub_problems: List[Dict[str, Any]]
    entanglement_matrix: Dict[str, List[str]] = field(default_factory=dict)
    loop_strategy: str = "sequential"  # sequential, parallel, super_node
    max_iterations: int = 10
    convergence_threshold: float = 0.95
    
    def get_sub_problem_ids(self) -> List[str]:
        """Extract sub-problem IDs from the list."""
        return [sp.get("id", f"sp_{i}") for i, sp in enumerate(self.sub_problems)]
    
    def get_entangled_pairs(self) -> List[Tuple[str, str]]:
        """Get all entangled pairs from the matrix."""
        pairs = []
        for source, targets in self.entanglement_matrix.items():
            for target in targets:
                pairs.append((source, target))
        return pairs


@dataclass
class EntanglementVisualizationConfig:
    """Configuration for entanglement matrix visualization bubble."""
    entanglement_matrix: Dict[str, List[str]]
    sub_problems: List[Dict[str, Any]] = field(default_factory=list)
    show_coupling_strength: bool = True
    highlight_super_nodes: bool = True
    
    def get_coupling_density(self) -> float:
        """Calculate coupling density (entanglements / max possible)."""
        n = len(self.sub_problems)
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) / 2
        actual_edges = sum(len(targets) for targets in self.entanglement_matrix.values()) // 2
        return actual_edges / max_edges if max_edges > 0 else 0.0


@dataclass
class SubProblemBubbleConfig:
    """Configuration for an individual sub-problem bubble."""
    sub_problem_id: str
    problem_text: str
    entangled_with: List[str] = field(default_factory=list)
    entanglement_source: str = "symbolic_overlap"
    is_super_node: bool = False
    super_node_partner: Optional[str] = None
    

# =============================================================================
# Entanglement Matrix Utilities (compatible with utils/entanglement_utils)
# =============================================================================

def normalize_entanglement_matrix_z3(
    matrix: Dict[str, Any],
    allowed_ids: Optional[List[str]] = None,
    enforce_symmetry: bool = True,
    strict: bool = False,
) -> Dict[str, Set[str]]:
    """
    Normalize entanglement matrices to Dict[str, Set[str]].
    Compatible with utils/entanglement_utils.normalize_entanglement_matrix.
    
    Args:
        matrix: Raw entanglement matrix
        allowed_ids: Allowed sub-problem IDs
        enforce_symmetry: Ensure bidirectional entanglements
        strict: Raise on validation errors
    
    Returns:
        Normalized matrix
    """
    allowed_set = set(allowed_ids or [])
    raw_map: Dict[str, Set[str]] = {}
    
    if matrix:
        for key, value in matrix.items():
            if allowed_set and key not in allowed_set:
                if strict:
                    raise ValueError(f"Entanglement matrix key not allowed: {key}")
                continue
            if isinstance(value, (set, list, tuple)):
                items = value
            elif value is None:
                items = []
            else:
                items = [value]
            
            raw_set: Set[str] = set()
            for item in items:
                if item is None:
                    continue
                if item == key:
                    if strict:
                        raise ValueError(f"Self-entanglement detected for {key}")
                    continue
                if allowed_set and item not in allowed_set:
                    if strict:
                        raise ValueError(f"Entanglement partner not allowed: {item}")
                    continue
                raw_set.add(item)
            raw_map[key] = raw_set
    
    if not allowed_set:
        allowed_set = set(raw_map.keys())
    
    normalized: Dict[str, Set[str]] = {key: set() for key in allowed_set}
    for key, partners in raw_map.items():
        if allowed_set and key not in allowed_set:
            continue
        normalized.setdefault(key, set()).update(partners)
    
    if enforce_symmetry:
        for key, partners in list(normalized.items()):
            for partner in list(partners):
                normalized.setdefault(partner, set()).add(key)
    
    for key in normalized:
        normalized[key].discard(key)
    
    return normalized


def serialize_entanglement_matrix_z3(matrix: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Serialize normalized matrix to JSON-safe format."""
    return {key: sorted(list(value)) for key, value in matrix.items()}


def build_entanglement_from_subproblems(sub_problems: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Build entanglement matrix from sub-problem shared symbols.
    
    Args:
        sub_problems: List of sub-problem dicts with optional 'shared_symbols' field
    
    Returns:
        Entanglement matrix
    """
    matrix: Dict[str, Set[str]] = {}
    
    # Initialize empty sets for all sub-problems
    ids = [sub_problems[i].get("id", f"sp_{i}") for i in range(len(sub_problems))]
    for sp_id in ids:
        matrix[sp_id] = set()
    
    # Find shared symbols and create entanglements
    for i, sp1 in enumerate(sub_problems):
        id1 = sp1.get("id", f"sp_{i}")
        symbols1 = set(sp1.get("shared_symbols", []))
        
        for j, sp2 in enumerate(sub_problems):
            if i >= j:
                continue
            id2 = sp2.get("id", f"sp_{j}")
            symbols2 = set(sp2.get("shared_symbols", []))
            
            # Check for symbol overlap
            overlap = symbols1 & symbols2
            if overlap:
                matrix[id1].add(id2)
                matrix[id2].add(id1)
    
    return serialize_entanglement_matrix_z3(matrix)


def create_z3_solver_bubble(
    config: Z3SolverBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Z3 Solver"
) -> Dict[str, Any]:
    """
    Create a Z3 constraint solver bubble.
    
    Args:
        config: Z3SolverBubbleConfig with solver configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a Z3 solver bubble
    """
    position = position or Z3_NODE_POSITIONS.get("z3_solver", {"x": 300, "y": 0})
    icon = Z3_NODE_ICONS.get("z3_solver", "🔐")
    color = Z3_NODE_COLORS.get("z3_solver", "#FF6B6B")
    
    bubble = {
        "id": f"z3_solver_{uuid.uuid4().hex[:8]}",
        "type": "z3_solver",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "problem_text": config.problem_text,
            "variables": config.variables,
            "constraints": config.constraints,
            "timeout_seconds": config.timeout_seconds,
            "strategy": config.strategy,
            "status": "pending",
            "node_color": color,
            "result": None,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created Z3 solver bubble: {bubble['id']}")
    return bubble


def create_z3_prover_bubble(
    config: Z3ProverBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Z3 Prover"
) -> Dict[str, Any]:
    """
    Create a Z3 theorem prover bubble.
    
    Args:
        config: Z3ProverBubbleConfig with prover configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a Z3 prover bubble
    """
    position = position or Z3_NODE_POSITIONS.get("z3_prover", {"x": 300, "y": 100})
    icon = Z3_NODE_ICONS.get("z3_prover", "📐")
    color = Z3_NODE_COLORS.get("z3_prover", "#E17055")
    
    bubble = {
        "id": f"z3_prover_{uuid.uuid4().hex[:8]}",
        "type": "z3_prover",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "theorem_statement": config.theorem_statement,
            "proof_strategy": config.proof_strategy,
            "timeout_seconds": config.timeout_seconds,
            "status": "pending",
            "node_color": color,
            "proven": False,
            "proof_steps": [],
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created Z3 prover bubble: {bubble['id']}")
    return bubble


def create_leanaide_proof_bubble(
    config: LeanAideProofBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a LeanAIDE proof visualization bubble.
    
    Args:
        config: LeanAideProofBubbleConfig with proof configuration
        position: Optional position override
        label: Display label (defaults to theorem name)
    
    Returns:
        Dict representing a LeanAIDE proof bubble
    """
    position = position or Z3_NODE_POSITIONS.get("leanaide_proof", {"x": 500, "y": 0})
    icon = Z3_NODE_ICONS.get("leanaide_proof", "📚")
    color = Z3_NODE_COLORS.get("leanaide_proof", "#00B894")
    label = label or f"{icon} {config.theorem_name}"
    
    bubble = {
        "id": f"leanaide_proof_{uuid.uuid4().hex[:8]}",
        "type": "leanaide_proof",
        "position": position,
        "data": {
            "label": label,
            "theorem_name": config.theorem_name,
            "proof_type": config.proof_type,
            "mcts_enabled": config.mcts_enabled,
            "timeout_seconds": config.timeout_seconds,
            "status": "pending",
            "node_color": color,
            "proof_steps": [],
            "proven": False,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created LeanAIDE proof bubble: {bubble['id']}")
    return bubble


def create_cross_verification_bubble(
    config: CrossVerificationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Cross Verification"
) -> Dict[str, Any]:
    """
    Create a cross-verification bubble (Z3 + LeanAIDE).
    
    Args:
        config: CrossVerificationBubbleConfig with verification configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a cross-verification bubble
    """
    position = position or Z3_NODE_POSITIONS.get("cross_verify", {"x": 650, "y": 0})
    icon = Z3_NODE_ICONS.get("cross_verify", "⚖️")
    color = Z3_NODE_COLORS.get("cross_verify", "#6C5CE7")
    
    bubble = {
        "id": f"cross_verify_{uuid.uuid4().hex[:8]}",
        "type": "cross_verification",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "problem_text": config.problem_text,
            "z3_strategy": config.z3_strategy,
            "lean_strategy": config.lean_strategy,
            "timeout_seconds": config.timeout_seconds,
            "status": "pending",
            "node_color": color,
            "z3_status": None,
            "lean_status": None,
            "agreement": None,
            "confidence_score": 0.0,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created cross-verification bubble: {bubble['id']}")
    return bubble


def create_problem_classification_bubble(
    config: ProblemClassificationBubbleConfig,
    position: Dict[str, float] = None,
    label: str = "Problem Classification"
) -> Dict[str, Any]:
    """
    Create a problem classification bubble.
    
    Args:
        config: ProblemClassificationBubbleConfig with classification configuration
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a classification bubble
    """
    position = position or Z3_NODE_POSITIONS.get("classification", {"x": 150, "y": 0})
    icon = Z3_NODE_ICONS.get("classification", "🔍")
    color = Z3_NODE_COLORS.get("classification", "#0984E3")
    
    bubble = {
        "id": f"classification_{uuid.uuid4().hex[:8]}",
        "type": "problem_classification",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "problem_text": config.problem_text,
            "auto_classify": config.auto_classify,
            "status": "pending",
            "node_color": color,
            "classification": None,
            "confidence": 0.0,
            "recommended_solver": None,
            "execution_time": 0.0
        }
    }
    
    logger.debug(f"Created classification bubble: {bubble['id']}")
    return bubble


def create_z3_result_bubble(
    result_status: str = "pending",
    position: Dict[str, float] = None,
    label: str = "Result"
) -> Dict[str, Any]:
    """
    Create a result bubble for Z3/LeanAIDE workflows.
    
    Args:
        result_status: Result status (pending, success, failed)
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing a result bubble
    """
    position = position or Z3_NODE_POSITIONS.get("result", {"x": 800, "y": 0})
    icon = Z3_NODE_ICONS.get("result", "✅")
    color = Z3_NODE_COLORS.get("result", "#00B894")
    
    status_colors = {
        "pending": "#FDCB6E",
        "success": "#00B894",
        "failed": "#FF7675",
        "verified": "#00B894",
        "unverified": "#FF7675"
    }
    color = status_colors.get(result_status, color)
    
    bubble = {
        "id": f"z3_result_{uuid.uuid4().hex[:8]}",
        "type": "z3_result",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "status": result_status,
            "node_color": color,
            "summary": None,
            "details": {}
        }
    }
    
    logger.debug(f"Created Z3 result bubble: {bubble['id']}")
    return bubble


# =============================================================================
# Sub-Problem Loop and Entanglement Bubbles
# =============================================================================

def create_subproblem_loop_bubble(
    config: SubProblemLoopBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create a sub-problem loop bubble for handling entangled sub-problems.
    
    This bubble manages iterative solving of entangled sub-problems,
    supporting convergence-based refinement.
    
    Args:
        config: SubProblemLoopBubbleConfig with loop configuration
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a sub-problem loop bubble
    """
    position = position or {"x": 300, "y": 200}
    icon = Z3_NODE_ICONS.get("subproblem_loop", "🔄")
    color = Z3_NODE_COLORS.get("subproblem_loop", "#FDCB6E")
    
    sub_problem_ids = config.get_sub_problem_ids()
    entangled_pairs = config.get_entangled_pairs()
    
    label = label or f"{icon} Sub-Problem Loop ({len(sub_problem_ids)} problems)"
    
    bubble = {
        "id": f"subproblem_loop_{uuid.uuid4().hex[:8]}",
        "type": "subproblem_loop",
        "position": position,
        "data": {
            "label": label,
            "sub_problems": config.sub_problems,
            "sub_problem_ids": sub_problem_ids,
            "entanglement_matrix": config.entanglement_matrix,
            "entangled_pairs": entangled_pairs,
            "loop_strategy": config.loop_strategy,
            "max_iterations": config.max_iterations,
            "convergence_threshold": config.convergence_threshold,
            "status": "pending",
            "node_color": color,
            "current_iteration": 0,
            "converged": False,
            "refined_sub_problems": []
        }
    }
    
    logger.debug(f"Created sub-problem loop bubble: {bubble['id']}")
    return bubble


def create_entanglement_visualization_bubble(
    config: EntanglementVisualizationConfig,
    position: Dict[str, float] = None,
    label: str = "Entanglement Matrix"
) -> Dict[str, Any]:
    """
    Create a bubble for visualizing the entanglement matrix.
    
    This bubble displays the coupling between sub-problems and
    highlights super-nodes (tightly coupled groups).
    
    Args:
        config: EntanglementVisualizationConfig with visualization settings
        position: Optional position override
        label: Display label for the bubble
    
    Returns:
        Dict representing an entanglement visualization bubble
    """
    position = position or {"x": 150, "y": 200}
    icon = Z3_NODE_ICONS.get("entanglement_viz", "🕸️")
    color = Z3_NODE_COLORS.get("entanglement_viz", "#E84393")
    
    coupling_density = config.get_coupling_density()
    
    # Identify super-nodes (nodes with high degree)
    super_nodes = []
    if config.highlight_super_nodes:
        matrix = normalize_entanglement_matrix_z3(config.entanglement_matrix)
        avg_degree = sum(len(neighbors) for neighbors in matrix.values()) / max(len(matrix), 1)
        for sp_id, neighbors in matrix.items():
            if len(neighbors) > avg_degree * 1.5:
                super_nodes.append(sp_id)
    
    bubble = {
        "id": f"entanglement_viz_{uuid.uuid4().hex[:8]}",
        "type": "entanglement_viz",
        "position": position,
        "data": {
            "label": f"{icon} {label}",
            "entanglement_matrix": config.entanglement_matrix,
            "sub_problems": config.sub_problems,
            "coupling_density": coupling_density,
            "super_nodes": super_nodes,
            "show_coupling_strength": config.show_coupling_strength,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created entanglement visualization bubble: {bubble['id']}")
    return bubble


def create_subproblem_bubble(
    config: SubProblemBubbleConfig,
    position: Dict[str, float] = None,
    label: str = None
) -> Dict[str, Any]:
    """
    Create an individual sub-problem bubble for detailed viewing.
    
    Args:
        config: SubProblemBubbleConfig with sub-problem details
        position: Optional position override
        label: Display label (auto-generated if not provided)
    
    Returns:
        Dict representing a sub-problem bubble
    """
    position = position or {"x": 400, "y": 200}
    icon = Z3_NODE_ICONS.get("super_node", "📦") if config.is_super_node else Z3_NODE_ICONS.get("subproblem", "📦")
    color = Z3_NODE_COLORS.get("super_node", "#A29BFE") if config.is_super_node else Z3_NODE_COLORS.get("subproblem", "#81ECEC")
    
    label = label or f"{icon} {config.sub_problem_id}"
    
    bubble = {
        "id": f"subproblem_{config.sub_problem_id}_{uuid.uuid4().hex[:8]}",
        "type": "subproblem",
        "sub_problem_id": config.sub_problem_id,
        "position": position,
        "data": {
            "label": label,
            "problem_text": config.problem_text,
            "entangled_with": config.entangled_with,
            "entanglement_source": config.entanglement_source,
            "is_super_node": config.is_super_node,
            "super_node_partner": config.super_node_partner,
            "status": "pending",
            "node_color": color
        }
    }
    
    logger.debug(f"Created sub-problem bubble: {bubble['id']}")
    return bubble


def create_super_node_bubble(
    sub_problem_ids: List[str],
    problem_text: str = "Super Node",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a super-node bubble for tightly coupled sub-problems.
    
    Super-nodes are groups of sub-problems that are highly entangled
    and should be solved together.
    
    Args:
        sub_problem_ids: List of sub-problem IDs in the super node
        problem_text: Description of the super node
        position: Optional position override
    
    Returns:
        Dict representing a super-node bubble
    """
    position = position or {"x": 500, "y": 200}
    icon = Z3_NODE_ICONS.get("super_node", "🔗")
    color = Z3_NODE_COLORS.get("super_node", "#A29BFE")
    
    bubble = {
        "id": f"super_node_{uuid.uuid4().hex[:8]}",
        "type": "super_node",
        "position": position,
        "data": {
            "label": f"{icon} Super Node ({len(sub_problem_ids)} problems)",
            "sub_problem_ids": sub_problem_ids,
            "problem_text": problem_text,
            "status": "pending",
            "node_color": color,
            "member_count": len(sub_problem_ids)
        }
    }
    
    logger.debug(f"Created super-node bubble: {bubble['id']}")
    return bubble


# =============================================================================
# Edge Creation Functions
# =============================================================================

def create_z3_edge(
    source_id: str,
    target_id: str,
    edge_type: str = "default",
    source_handle: str = "output",
    target_handle: str = "input"
) -> Dict[str, Any]:
    """
    Create an edge connecting Z3/LeanAIDE bubbles.
    
    Args:
        source_id: ID of the source bubble
        target_id: ID of the target bubble
        edge_type: Type of edge (default, conditional, feedback)
        source_handle: Handle on source bubble
        target_handle: Handle on target bubble
    
    Returns:
        Dict representing an edge
    """
    edge = {
        "id": f"edge_{source_id}_{target_id}_{uuid.uuid4().hex[:8]}",
        "source": source_id,
        "target": target_id,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
        "type": edge_type,
        "animated": edge_type == "default",
        "style": {
            "stroke": get_z3_edge_color(edge_type),
            "strokeWidth": 2
        }
    }
    
    return edge


def get_z3_edge_color(edge_type: str) -> str:
    """Get color for edge type."""
    colors = {
        "default": "#888888",
        "conditional": "#FF6B6B",
        "feedback": "#9B59B6",
        "success": "#00B894",
        "error": "#FF7675",
    }
    return colors.get(edge_type, "#888888")


def create_conditional_z3_edge(
    source_id: str,
    target_id: str,
    condition: str
) -> Dict[str, Any]:
    """Create a conditional edge with labeled condition."""
    edge = create_z3_edge(source_id, target_id, "conditional")
    edge["label"] = condition
    edge["labelStyle"] = {"fill": "#FF6B6B", "fontSize": 12}
    return edge


def create_feedback_z3_edge(
    source_id: str,
    target_id: str
) -> Dict[str, Any]:
    """Create a feedback edge for iterative verification."""
    return create_z3_edge(
        source_id, target_id, "feedback",
        "feedback", "retry"
    )


# =============================================================================
# Complete Z3 Workflow Creation
# =============================================================================

def create_z3_solver_workflow(
    problem_text: str,
    workflow_name: str = "Z3 Solver Workflow",
    variables: List[Dict[str, Any]] = None,
    constraints: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a complete Z3 constraint solving workflow.
    
    Args:
        problem_text: The constraint problem to solve
        workflow_name: Name of the workflow
        variables: Optional list of variables
        constraints: Optional list of constraints
    
    Returns:
        Dict with complete workflow definition
    """
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"z3_input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": Z3_NODE_POSITIONS["input"],
        "data": {
            "label": "📥 Input",
            "problem_text": problem_text,
            "status": "pending",
            "node_color": Z3_NODE_COLORS["input"]
        }
    }
    nodes.append(input_bubble)
    
    # Create solver bubble
    solver_config = Z3SolverBubbleConfig(
        problem_text=problem_text,
        variables=variables or [],
        constraints=constraints or []
    )
    solver_bubble = create_z3_solver_bubble(solver_config)
    nodes.append(solver_bubble)
    
    # Connect input to solver
    edges.append(create_z3_edge(input_bubble["id"], solver_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_z3_result_bubble()
    nodes.append(result_bubble)
    
    # Connect solver to result
    edges.append(create_z3_edge(solver_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Z3 solver workflow for: {problem_text[:50]}...",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_text": problem_text,
            "workflow_type": "z3_solver",
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    logger.info(f"Created Z3 workflow: {workflow['id']}")
    return workflow


def create_z3_leanaide_workflow(
    problem_text: str,
    workflow_name: str = "Z3-LeanAIDE Verification",
    include_proof: bool = True,
    include_cross_verify: bool = True
) -> Dict[str, Any]:
    """
    Create a complete Z3 + LeanAIDE verification workflow.
    
    Args:
        problem_text: The problem to verify
        workflow_name: Name of the workflow
        include_proof: Whether to include LeanAIDE proof
        include_cross_verify: Whether to include cross-verification
    
    Returns:
        Dict with complete workflow definition
    """
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"z3_input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": Z3_NODE_POSITIONS["input"],
        "data": {
            "label": "📥 Input",
            "problem_text": problem_text,
            "status": "pending",
            "node_color": Z3_NODE_COLORS["input"]
        }
    }
    nodes.append(input_bubble)
    
    # Create classification bubble
    class_config = ProblemClassificationBubbleConfig(problem_text=problem_text)
    class_bubble = create_problem_classification_bubble(class_config)
    nodes.append(class_bubble)
    edges.append(create_z3_edge(input_bubble["id"], class_bubble["id"]))
    
    # Create Z3 solver bubble
    solver_config = Z3SolverBubbleConfig(problem_text=problem_text)
    solver_bubble = create_z3_solver_bubble(solver_config)
    nodes.append(solver_bubble)
    edges.append(create_z3_edge(class_bubble["id"], solver_bubble["id"]))
    
    # Create LeanAIDE proof bubble (optional)
    if include_proof:
        proof_config = LeanAideProofBubbleConfig(
            theorem_name=workflow_name,
            proof_type="theorem"
        )
        proof_bubble = create_leanaide_proof_bubble(proof_config)
        nodes.append(proof_bubble)
        edges.append(create_z3_edge(solver_bubble["id"], proof_bubble["id"]))
    
    # Create cross-verification bubble (optional)
    if include_cross_verify:
        cross_config = CrossVerificationBubbleConfig(problem_text=problem_text)
        cross_bubble = create_cross_verification_bubble(cross_config)
        nodes.append(cross_bubble)
        
        last_node = proof_bubble["id"] if include_proof else solver_bubble["id"]
        edges.append(create_z3_edge(last_node, cross_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_z3_result_bubble()
    nodes.append(result_bubble)
    
    # Connect last node to result
    if include_cross_verify:
        edges.append(create_z3_edge(cross_bubble["id"], result_bubble["id"]))
    elif include_proof:
        edges.append(create_z3_edge(proof_bubble["id"], result_bubble["id"]))
    else:
        edges.append(create_z3_edge(solver_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Z3-LeanAIDE verification for: {problem_text[:50]}...",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_text": problem_text,
            "workflow_type": "z3_leanaide",
            "include_proof": include_proof,
            "include_cross_verify": include_cross_verify,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    logger.info(f"Created Z3-LeanAIDE workflow: {workflow['id']}")
    return workflow


def create_z3_workflow_with_entanglement(
    problem_text: str,
    sub_problems: List[Dict[str, Any]],
    entanglement_matrix: Dict[str, List[str]] = None,
    workflow_name: str = "Entangled Z3 Workflow",
    loop_strategy: str = "sequential"
) -> Dict[str, Any]:
    """
    Create a Z3 workflow with sub-problem loops and entanglement matrix integration.
    
    This workflow supports:
    - Iterative solving of entangled sub-problems
    - Convergence-based refinement
    - Super-node detection for tightly coupled problems
    
    Args:
        problem_text: The overall problem description
        sub_problems: List of sub-problem dicts with 'id', 'problem_text', optional 'shared_symbols'
        entanglement_matrix: Dict mapping sub-problem IDs to lists of entangled IDs
        workflow_name: Name of the workflow
        loop_strategy: Strategy for sub-problem loops (sequential, parallel, super_node)
    
    Returns:
        Dict with complete workflow definition including entanglement visualization
    """
    # Build entanglement matrix if not provided
    if entanglement_matrix is None:
        entanglement_matrix = build_entanglement_from_subproblems(sub_problems)
    
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"z3_input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": {"x": 0, "y": 0},
        "data": {
            "label": "📥 Input",
            "problem_text": problem_text,
            "status": "pending",
            "node_color": Z3_NODE_COLORS["input"]
        }
    }
    nodes.append(input_bubble)
    
    # Create entanglement visualization bubble
    ent_viz_config = EntanglementVisualizationConfig(
        entanglement_matrix=entanglement_matrix,
        sub_problems=sub_problems,
        show_coupling_strength=True,
        highlight_super_nodes=True
    )
    ent_viz_bubble = create_entanglement_visualization_bubble(ent_viz_config)
    nodes.append(ent_viz_bubble)
    edges.append(create_z3_edge(input_bubble["id"], ent_viz_bubble["id"]))
    
    # Create sub-problem loop bubble
    loop_config = SubProblemLoopBubbleConfig(
        sub_problems=sub_problems,
        entanglement_matrix=entanglement_matrix,
        loop_strategy=loop_strategy,
        max_iterations=10,
        convergence_threshold=0.95
    )
    loop_bubble = create_subproblem_loop_bubble(loop_config)
    nodes.append(loop_bubble)
    edges.append(create_z3_edge(ent_viz_bubble["id"], loop_bubble["id"]))
    
    # Create individual sub-problem bubbles
    sub_problem_bubbles = []
    for i, sp in enumerate(sub_problems):
        sp_id = sp.get("id", f"sp_{i}")
        entangled_with = entanglement_matrix.get(sp_id, [])
        
        # Check if this is a super-node
        matrix = normalize_entanglement_matrix_z3(entanglement_matrix)
        avg_degree = sum(len(neighbors) for neighbors in matrix.values()) / max(len(matrix), 1)
        is_super_node = len(matrix.get(sp_id, set())) > avg_degree * 1.5
        
        sp_config = SubProblemBubbleConfig(
            sub_problem_id=sp_id,
            problem_text=sp.get("problem_text", f"Sub-problem {sp_id}"),
            entangled_with=entangled_with,
            entanglement_source=sp.get("entanglement_source", "symbolic_overlap"),
            is_super_node=is_super_node
        )
        sp_bubble = create_subproblem_bubble(
            sp_config,
            position={"x": 400, "y": 200 + i * 80}
        )
        nodes.append(sp_bubble)
        sub_problem_bubbles.append(sp_bubble)
        
        # Connect loop to each sub-problem
        edges.append(create_z3_edge(loop_bubble["id"], sp_bubble["id"], "feedback"))
    
    # Create cross-verification for entangled pairs
    cross_bubble = create_cross_verification_bubble(
        CrossVerificationBubbleConfig(problem_text=problem_text)
    )
    nodes.append(cross_bubble)
    
    # Connect all sub-problems to cross-verification
    for sp_bubble in sub_problem_bubbles:
        edges.append(create_z3_edge(sp_bubble["id"], cross_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_z3_result_bubble()
    nodes.append(result_bubble)
    edges.append(create_z3_edge(cross_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Entangled Z3 workflow with {len(sub_problems)} sub-problems",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_text": problem_text,
            "sub_problems": sub_problems,
            "entanglement_matrix": entanglement_matrix,
            "workflow_type": "z3_entangled",
            "loop_strategy": loop_strategy,
            "coupling_density": ent_viz_config.get_coupling_density(),
            "super_nodes": ent_viz_bubble["data"]["super_nodes"],
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    }
    
    logger.info(f"Created entangled Z3 workflow: {workflow['id']}")
    return workflow


# =============================================================================
# Flexible Workflow Builder for Z3/LeanAIDE
# =============================================================================

@dataclass
class Z3BubbleDefinition:
    """Definition of a Z3/LeanAIDE bubble for user-defined workflows."""
    bubble_type: str  # z3_solver, z3_prover, leanaide_proof, cross_verification, classification
    label: str
    position: Dict[str, float] = None
    config: Dict[str, Any] = field(default_factory=dict)
    node_color: str = "#888888"


@dataclass  
class Z3EdgeDefinition:
    """Definition of an edge for Z3/LeanAIDE workflows."""
    source_label: str
    target_label: str
    edge_type: str = "default"
    condition: str = ""
    source_handle: str = "output"
    target_handle: str = "input"


class Z3FlexibleWorkflowBuilder:
    """Builder for creating arbitrary Z3/LeanAIDE workflow patterns."""
    
    def __init__(self):
        self.bubbles: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, Any]] = []
        self.bubble_map: Dict[str, Dict[str, Any]] = {}
    
    def add_bubble(self, bubble_def: Z3BubbleDefinition) -> str:
        """Add a bubble to the workflow."""
        bubble_id = f"{bubble_def.bubble_type}_{uuid.uuid4().hex[:8]}"
        
        position = bubble_def.position or {"x": len(self.bubbles) * 150, "y": 0}
        
        bubble = {
            "id": bubble_id,
            "type": bubble_def.bubble_type,
            "position": position,
            "data": {
                "label": bubble_def.label,
                "status": "pending",
                "node_color": bubble_def.node_color,
                **bubble_def.config
            }
        }
        
        self.bubbles.append(bubble)
        self.bubble_map[bubble_def.label] = bubble
        
        return bubble_id
    
    def add_edge(self, edge_def: Z3EdgeDefinition) -> str:
        """Add an edge connecting two bubbles."""
        source_bubble = self.bubble_map.get(edge_def.source_label)
        target_bubble = self.bubble_map.get(edge_def.target_label)
        
        if not source_bubble:
            raise ValueError(f"Source bubble not found: {edge_def.source_label}")
        if not target_bubble:
            raise ValueError(f"Target bubble not found: {edge_def.target_label}")
        
        edge_id = f"edge_{source_bubble['id']}_{target_bubble['id']}_{uuid.uuid4().hex[:8]}"
        
        edge = {
            "id": edge_id,
            "source": source_bubble["id"],
            "target": target_bubble["id"],
            "sourceHandle": edge_def.source_handle,
            "targetHandle": edge_def.target_handle,
            "type": edge_def.edge_type,
            "animated": edge_def.edge_type == "default",
            "style": {
                "stroke": get_z3_edge_color(edge_def.edge_type),
                "strokeWidth": 2
            }
        }
        
        if edge_def.condition:
            edge["label"] = edge_def.condition
            edge["labelStyle"] = {"fill": "#FF6B6B", "fontSize": 12}
        
        self.edges.append(edge)
        return edge_id
    
    def build(self, workflow_name: str, problem_text: str = "") -> Dict[str, Any]:
        """Build the complete workflow."""
        return {
            "id": str(uuid.uuid4()),
            "name": workflow_name,
            "description": problem_text or f"Z3/LeanAIDE workflow: {workflow_name}",
            "nodes": self.bubbles,
            "edges": self.edges,
            "metadata": {
                "workflow_type": "z3_leanaide_custom",
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        }
    
    def reset(self):
        """Reset the builder."""
        self.bubbles = []
        self.edges = []
        self.bubble_map = {}


def create_custom_z3_workflow(
    workflow_name: str,
    problem_text: str,
    bubble_labels: List[str],
    bubble_types: List[str],
    team_config: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create a custom Z3/LeanAIDE workflow from label/type lists.
    
    Args:
        workflow_name: Name of the workflow
        problem_text: Problem description
        bubble_labels: Ordered list of bubble labels
        bubble_types: Ordered list of bubble types
        team_config: Optional team mapping
    
    Returns:
        Dict with workflow definition
    """
    builder = Z3FlexibleWorkflowBuilder()
    team_config = team_config or {}
    
    for i, (label, btype) in enumerate(zip(bubble_labels, bubble_types)):
        color = Z3_NODE_COLORS.get(btype, "#888888")
        config = {"problem_text": problem_text} if i == 0 else {}
        
        if btype == "z3_solver":
            config = {"problem_text": problem_text, "variables": [], "constraints": []}
        elif btype == "classification":
            config = {"problem_text": problem_text, "auto_classify": True}
        
        bubble_def = Z3BubbleDefinition(
            bubble_type=btype,
            label=label,
            node_color=color,
            config=config
        )
        builder.add_bubble(bubble_def)
    
    # Create sequential edges
    for i in range(len(bubble_labels) - 1):
        edge_def = Z3EdgeDefinition(
            source_label=bubble_labels[i],
            target_label=bubble_labels[i + 1]
        )
        builder.add_edge(edge_def)
    
    return builder.build(workflow_name, problem_text)


# =============================================================================
# Bubble Update Functions
# =============================================================================

def update_z3_bubble_status(
    bubble: Dict[str, Any],
    status: str,
    additional_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Update the status of a Z3/LeanAIDE bubble."""
    bubble["data"]["status"] = status
    
    if additional_data:
        bubble["data"].update(additional_data)
    
    status_colors = {
        "pending": "#FDCB6E",
        "running": "#74B9FF",
        "success": "#00B894",
        "failed": "#FF7675",
        "verified": "#00B894",
    }
    
    if status in status_colors:
        bubble["data"]["node_color"] = status_colors[status]
    
    return bubble


def add_z3_result_to_bubble(
    bubble: Dict[str, Any],
    success: bool,
    result_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Add result data to a Z3/LeanAIDE bubble."""
    bubble["data"]["result"] = result_data
    bubble["data"]["status"] = "success" if success else "failed"
    bubble["data"]["node_color"] = Z3_NODE_COLORS.get("result", "#00B894")
    
    return bubble


# =============================================================================
# Serialization and Export
# =============================================================================

def serialize_z3_bubble(bubble: Dict[str, Any]) -> str:
    """Serialize a Z3 bubble to JSON string."""
    import json
    return json.dumps(bubble, indent=2)


def serialize_z3_workflow(workflow: Dict[str, Any]) -> str:
    """Serialize a Z3 workflow to JSON string."""
    import json
    return json.dumps(workflow, indent=2)


def export_z3_workflow_to_json(
    workflow: Dict[str, Any],
    output_path: str
) -> bool:
    """Export a Z3 workflow to a JSON file."""
    import json
    import os
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        logger.info(f"Exported Z3 workflow to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export workflow: {e}")
        return False


# =============================================================================
# Example Usage
# =============================================================================

def example_z3_workflow():
    """Example: Create and export a Z3 solver workflow."""
    workflow = create_z3_solver_workflow(
        problem_text="Find x, y such that: x + y = 10, x * y = 24",
        workflow_name="Equation Solver"
    )
    
    export_z3_workflow_to_json(workflow, "z3_workflow_example.json")
    
    return workflow


def example_z3_leanaide_workflow():
    """Example: Create and export a Z3-LeanAIDE workflow."""
    workflow = create_z3_leanaide_workflow(
        problem_text="Prove that for all natural numbers n, n^2 >= n",
        workflow_name="Theorem Verification",
        include_proof=True,
        include_cross_verify=True
    )
    
    export_z3_workflow_to_json(workflow, "z3_leanaide_workflow_example.json")
    
    return workflow


def example_entangled_z3_workflow():
    """Example: Create and export an entangled Z3 workflow with sub-problem loops."""
    # Define sub-problems with shared symbols for entanglement
    sub_problems = [
        {
            "id": "sp_A",
            "problem_text": "Solve for x: x + y = 10",
            "shared_symbols": ["x", "y"]
        },
        {
            "id": "sp_B", 
            "problem_text": "Solve for y: x * y = 24",
            "shared_symbols": ["x", "y"]
        },
        {
            "id": "sp_C",
            "problem_text": "Verify x > 0 and y > 0",
            "shared_symbols": ["x", "y"]
        },
        {
            "id": "sp_D",
            "problem_text": "Calculate x^2 + y^2",
            "shared_symbols": ["x", "y"]
        }
    ]
    
    # Entanglement matrix (automatically computed from shared_symbols)
    workflow = create_z3_workflow_with_entanglement(
        problem_text="System of equations with entangled variables",
        sub_problems=sub_problems,
        workflow_name="Entangled Equation Solver",
        loop_strategy="sequential"
    )
    
    export_z3_workflow_to_json(workflow, "z3_entangled_workflow_example.json")
    
    return workflow

if __name__ == "__main__":
    # Run examples
    workflow = example_z3_workflow()
    print(f"Created workflow: {workflow['name']}")
    print(f"Nodes: {len(workflow['nodes'])}")
    print(f"Edges: {len(workflow['edges'])}")
    
    # Run entanglement example
    ent_workflow = example_entangled_z3_workflow()
    print(f"\nCreated entangled workflow: {ent_workflow['name']}")
    print(f"Nodes: {len(ent_workflow['nodes'])}")
    print(f"Edges: {len(ent_workflow['edges'])}")
    print(f"Coupling density: {ent_workflow['metadata'].get('coupling_density', 'N/A')}")
    print(f"Super nodes: {ent_workflow['metadata'].get('super_nodes', [])}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config classes
    'Z3SolverBubbleConfig',
    'Z3ProverBubbleConfig',
    'LeanAideProofBubbleConfig',
    'CrossVerificationBubbleConfig',
    'ProblemClassificationBubbleConfig',
    'SubProblemLoopBubbleConfig',
    'EntanglementVisualizationConfig',
    'SubProblemBubbleConfig',
    
    # Builder definition classes
    'Z3BubbleDefinition',
    'Z3EdgeDefinition',
    
    # Entanglement utilities
    'normalize_entanglement_matrix_z3',
    'serialize_entanglement_matrix_z3',
    'build_entanglement_from_subproblems',
    
    # Bubble creation
    'create_z3_solver_bubble',
    'create_z3_prover_bubble',
    'create_leanaide_proof_bubble',
    'create_cross_verification_bubble',
    'create_problem_classification_bubble',
    'create_z3_result_bubble',
    'create_subproblem_loop_bubble',
    'create_entanglement_visualization_bubble',
    'create_subproblem_bubble',
    'create_super_node_bubble',
    
    # Edge creation
    'create_z3_edge',
    'create_conditional_z3_edge',
    'create_feedback_z3_edge',
    
    # Workflow creation
    'create_z3_solver_workflow',
    'create_z3_leanaide_workflow',
    'create_z3_workflow_with_entanglement',
    
    # Flexible builder
    'Z3FlexibleWorkflowBuilder',
    'create_custom_z3_workflow',
    
    # Updates
    'update_z3_bubble_status',
    'add_z3_result_to_bubble',
    
    # Serialization
    'serialize_z3_bubble',
    'serialize_z3_workflow',
    'export_z3_workflow_to_json',
]
