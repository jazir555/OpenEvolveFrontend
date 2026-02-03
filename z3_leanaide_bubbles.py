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
- Flexible workflow builder for arbitrary patterns

Usage:
    from z3_leanaide_bubbles import (
        create_z3_solver_bubble,
        create_z3_prover_bubble,
        create_leanaide_proof_bubble,
        create_cross_verification_bubble,
        create_z3_workflow
    )
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
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
}

Z3_NODE_ICONS = {
    "z3_solver": "🔐",
    "z3_prover": "📐",
    "leanaide_proof": "📚",
    "cross_verify": "⚖️",
    "classification": "🔍",
    "input": "📥",
    "result": "✅",
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


# =============================================================================
# Bubble Creation Functions
# =============================================================================

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


if __name__ == "__main__":
    # Run examples
    workflow = example_z3_workflow()
    print(f"Created workflow: {workflow['name']}")
    print(f"Nodes: {len(workflow['nodes'])}")
    print(f"Edges: {len(workflow['edges'])}")


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
    'Z3BubbleDefinition',
    'Z3EdgeDefinition',
    
    # Bubble creation
    'create_z3_solver_bubble',
    'create_z3_prover_bubble',
    'create_leanaide_proof_bubble',
    'create_cross_verification_bubble',
    'create_problem_classification_bubble',
    'create_z3_result_bubble',
    
    # Edge creation
    'create_z3_edge',
    'create_conditional_z3_edge',
    'create_feedback_z3_edge',
    
    # Workflow creation
    'create_z3_solver_workflow',
    'create_z3_leanaide_workflow',
    
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
