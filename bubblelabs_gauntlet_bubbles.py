"""
BubbleLabs Gauntlet Bubbles for OpenEvolve

This module provides BubbleLab workflow nodes (bubbles) specifically designed for gauntlet
operations in the OpenEvolve system. These bubbles enable visualization and control of
gauntlet workflows through the BubbleLabs UI.

Gauntlet Types Supported:
- Red Team Gauntlets (Adversarial Testing)
- Blue Team Gauntlets (Fix Generation)
- Gold Team Gauntlets (Consensus Verification)
- 3-Round Gauntlet System (LoongFlow AI Eval → Red Team → Gold Team)

Usage:
    from bubblelabs_gauntlet_bubbles import (
        create_gauntlet_execution_bubble,
        create_gauntlet_round_bubble,
        create_gauntlet_validation_bubble,
        create_gauntlet_result_bubble,
        create_gauntlet_workflow_definition
    )
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Bubble Configuration Constants
# =============================================================================

GAUNTLET_NODE_POSITIONS = {
    "start": {"x": 0, "y": 0},
    "evaluation": {"x": 200, "y": 0},
    "red_team": {"x": 400, "y": 0},
    "gold_team": {"x": 600, "y": 0},
    "result": {"x": 800, "y": 0},
}

GAUNTLET_NODE_COLORS = {
    "red_team": "#FF6B6B",
    "blue_team": "#4ECDC4",
    "gold_team": "#FFE66D",
    "evaluation": "#95E1D3",
    "result": "#A8E6CF",
    "input": "#DDA0DD",
    "output": "#98D8C8",
}

GAUNTLET_NODE_ICONS = {
    "red_team": "🛡️",
    "blue_team": "🔧",
    "gold_team": "✨",
    "evaluation": "📊",
    "result": "✅",
    "input": "📥",
    "output": "📤",
}


# =============================================================================
# Gauntlet Bubble Data Classes
# =============================================================================

@dataclass
class GauntletBubbleConfig:
    """Configuration for a gauntlet bubble."""
    gauntlet_name: str
    gauntlet_type: str
    team_name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    priority: int = 1


@dataclass
class GauntletRoundBubbleConfig:
    """Configuration for a gauntlet round bubble."""
    round_name: str
    round_order: int
    gauntlet_types: List[str]
    pass_threshold: float = 0.7
    requires_consensus: bool = True
    max_iterations: int = 5


@dataclass
class GauntletValidationBubbleConfig:
    """Configuration for a gauntlet validation bubble."""
    validation_type: str
    criteria: Dict[str, float]
    weight: float = 1.0
    required_score: float = 0.8
    feedback_mode: str = "detailed"


# =============================================================================
# Bubble Creation Functions
# =============================================================================

def create_gauntlet_execution_bubble(
    config: GauntletBubbleConfig,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a gauntlet execution bubble for BubbleLab workflow.
    
    Args:
        config: GauntletBubbleConfig with gauntlet configuration
        position: Optional position override (defaults to GAUNTLET_NODE_POSITIONS)
    
    Returns:
        Dict representing a BubbleLab node for gauntlet execution
    """
    position = position or GAUNTLET_NODE_POSITIONS.get(
        config.gauntlet_type, 
        {"x": 400, "y": 0}
    )
    
    icon = GAUNTLET_NODE_ICONS.get(config.gauntlet_type, "⚙️")
    color = GAUNTLET_NODE_COLORS.get(config.gauntlet_type, "#888888")
    
    bubble = {
        "id": f"gauntlet_{config.gauntlet_type}_{uuid.uuid4().hex[:8]}",
        "type": "gauntlet_execution",
        "position": position,
        "data": {
            "label": f"{icon} {config.gauntlet_name}",
            "gauntlet_type": config.gauntlet_type,
            "team_name": config.team_name,
            "description": config.description,
            "parameters": config.parameters,
            "timeout_seconds": config.timeout_seconds,
            "retry_count": config.retry_count,
            "priority": config.priority,
            "status": "pending",
            "node_color": color,
        }
    }
    
    logger.debug(f"Created gauntlet execution bubble: {bubble['id']}")
    return bubble


def create_gauntlet_round_bubble(
    config: GauntletRoundBubbleConfig,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a gauntlet round bubble for the 3-Round Gauntlet System.
    
    Args:
        config: GauntletRoundBubbleConfig with round configuration
        position: Optional position override
    
    Returns:
        Dict representing a BubbleLab node for gauntlet round
    """
    position = position or {"x": 200 + config.round_order * 200, "y": 0}
    
    icon = "🔄"
    color = "#6C5CE7"
    
    bubble = {
        "id": f"gauntlet_round_{config.round_order}_{uuid.uuid4().hex[:8]}",
        "type": "gauntlet_round",
        "position": position,
        "data": {
            "label": f"{icon} Round {config.round_order}: {config.round_name}",
            "round_name": config.round_name,
            "round_order": config.round_order,
            "gauntlet_types": config.gauntlet_types,
            "pass_threshold": config.pass_threshold,
            "requires_consensus": config.requires_consensus,
            "max_iterations": config.max_iterations,
            "status": "pending",
            "node_color": color,
        }
    }
    
    logger.debug(f"Created gauntlet round bubble: {bubble['id']}")
    return bubble


def create_gauntlet_validation_bubble(
    config: GauntletValidationBubbleConfig,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a gauntlet validation bubble for quality assessment.
    
    Args:
        config: GauntletValidationBubbleConfig with validation configuration
        position: Optional position override
    
    Returns:
        Dict representing a BubbleLab node for gauntlet validation
    """
    position = position or GAUNTLET_NODE_POSITIONS.get("evaluation", {"x": 200, "y": 0})
    
    icon = "📋"
    color = GAUNTLET_NODE_COLORS.get("evaluation", "#95E1D3")
    
    bubble = {
        "id": f"gauntlet_validation_{config.validation_type}_{uuid.uuid4().hex[:8]}",
        "type": "gauntlet_validation",
        "position": position,
        "data": {
            "label": f"{icon} Validation: {config.validation_type}",
            "validation_type": config.validation_type,
            "criteria": config.criteria,
            "weight": config.weight,
            "required_score": config.required_score,
            "feedback_mode": config.feedback_mode,
            "status": "pending",
            "node_color": color,
        }
    }
    
    logger.debug(f"Created gauntlet validation bubble: {bubble['id']}")
    return bubble


def create_gauntlet_result_bubble(
    gauntlet_name: str,
    result_status: str = "pending",
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a gauntlet result bubble for displaying outcomes.
    
    Args:
        gauntlet_name: Name of the gauntlet
        result_status: Result status (pending, passed, failed, partial)
        position: Optional position override
    
    Returns:
        Dict representing a BubbleLab node for gauntlet result
    """
    position = position or GAUNTLET_NODE_POSITIONS.get("result", {"x": 800, "y": 0})
    
    status_icons = {
        "pending": "⏳",
        "passed": "✅",
        "failed": "❌",
        "partial": "⚠️",
    }
    status_colors = {
        "pending": "#FFEAA7",
        "passed": "#00B894",
        "failed": "#FF7675",
        "partial": "#FDCB6E",
    }
    
    icon = status_icons.get(result_status, "📊")
    color = status_colors.get(result_status, "#888888")
    
    bubble = {
        "id": f"gauntlet_result_{gauntlet_name}_{uuid.uuid4().hex[:8]}",
        "type": "gauntlet_result",
        "position": position,
        "data": {
            "label": f"{icon} Result: {gauntlet_name}",
            "gauntlet_name": gauntlet_name,
            "status": result_status,
            "score": None,
            "feedback": None,
            "improvements": [],
            "node_color": color,
        }
    }
    
    logger.debug(f"Created gauntlet result bubble: {bubble['id']}")
    return bubble


def create_red_team_bubble(
    team_name: str = "Red Team",
    attack_modes: List[str] = None,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Red Team adversarial testing bubble.
    
    Args:
        team_name: Name of the Red Team
        attack_modes: List of attack modes to use
        position: Optional position override
    
    Returns:
        Dict representing a Red Team bubble
    """
    position = position or GAUNTLET_NODE_POSITIONS.get("red_team", {"x": 400, "y": 0})
    
    config = GauntletBubbleConfig(
        gauntlet_name=f"Red Team: {team_name}",
        gauntlet_type="red_team",
        team_name=team_name,
        description="Adversarial testing and attack simulation",
        parameters={"attack_modes": attack_modes or ["prompt_injection", "logic_bomb", "edge_case"]},
        timeout_seconds=600,
        retry_count=2,
        priority=2
    )
    
    return create_gauntlet_execution_bubble(config, position)


def create_blue_team_bubble(
    team_name: str = "Blue Team",
    fix_types: List[str] = None,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Blue Team fix generation bubble.
    
    Args:
        team_name: Name of the Blue Team
        fix_types: Types of fixes to generate
        position: Optional position override
    
    Returns:
        Dict representing a Blue Team bubble
    """
    position = position or GAUNTLET_NODE_POSITIONS.get("blue_team", {"x": 500, "y": 100})
    
    config = GauntletBubbleConfig(
        gauntlet_name=f"Blue Team: {team_name}",
        gauntlet_type="blue_team",
        team_name=team_name,
        description="Automated fix generation and improvement",
        parameters={"fix_types": fix_types or ["correctness", "performance", "robustness"]},
        timeout_seconds=300,
        retry_count=3,
        priority=1
    )
    
    return create_gauntlet_execution_bubble(config, position)


def create_gold_team_bubble(
    team_name: str = "Gold Team",
    verification_modes: List[str] = None,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a Gold Team consensus verification bubble.
    
    Args:
        team_name: Name of the Gold Team
        verification_modes: Modes of verification to perform
        position: Optional position override
    
    Returns:
        Dict representing a Gold Team bubble
    """
    position = position or GAUNTLET_NODE_POSITIONS.get("gold_team", {"x": 600, "y": 0})
    
    config = GauntletBubbleConfig(
        gauntlet_name=f"Gold Team: {team_name}",
        gauntlet_type="gold_team",
        team_name=team_name,
        description="Consensus verification and quality assurance",
        parameters={"verification_modes": verification_modes or ["consistency", "completeness", "correctness"]},
        timeout_seconds=300,
        retry_count=3,
        priority=3
    )
    
    return create_gauntlet_execution_bubble(config, position)


def create_loongeval_bubble(
    evaluation_criteria: Dict[str, float] = None,
    position: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Create a LoongFlow AI Evaluation bubble.
    
    Args:
        evaluation_criteria: Criteria for AI evaluation
        position: Optional position override
    
    Returns:
        Dict representing a LoongFlow evaluation bubble
    """
    position = position or GAUNTLET_NODE_POSITIONS.get("evaluation", {"x": 200, "y": 0})
    
    config = GauntletValidationBubbleConfig(
        validation_type="loongeval",
        criteria=evaluation_criteria or {
            "relevance": 0.3,
            "correctness": 0.4,
            "completeness": 0.3
        },
        weight=1.0,
        required_score=0.7,
        feedback_mode="detailed"
    )
    
    return create_gauntlet_validation_bubble(config, position)


# =============================================================================
# Bubble Edge Creation Functions
# =============================================================================

def create_bubble_edge(
    source_id: str,
    target_id: str,
    edge_type: str = "default",
    source_handle: str = "output",
    target_handle: str = "input"
) -> Dict[str, Any]:
    """
    Create an edge connecting two bubbles.
    
    Args:
        source_id: ID of the source bubble
        target_id: ID of the target bubble
        edge_type: Type of edge (default, conditional, feedback)
        source_handle: Handle on source bubble
        target_handle: Handle on target bubble
    
    Returns:
        Dict representing a BubbleLab edge
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
            "stroke": get_edge_color(edge_type),
            "strokeWidth": 2
        }
    }
    
    return edge


def get_edge_color(edge_type: str) -> str:
    """Get color for edge type."""
    colors = {
        "default": "#888888",
        "conditional": "#FF6B6B",
        "feedback": "#9B59B6",
        "success": "#00B894",
        "error": "#FF7675",
    }
    return colors.get(edge_type, "#888888")


def create_conditional_edge(
    source_id: str,
    target_id: str,
    condition: str,
    source_handle: str = "output",
    target_handle: str = "input"
) -> Dict[str, Any]:
    """Create a conditional edge with labeled condition."""
    edge = create_bubble_edge(
        source_id, target_id, "conditional", 
        source_handle, target_handle
    )
    edge["label"] = condition
    edge["labelStyle"] = {"fill": "#FF6B6B", "fontSize": 12}
    return edge


def create_feedback_edge(
    source_id: str,
    target_id: str,
    source_handle: str = "feedback",
    target_handle: str = "input"
) -> Dict[str, Any]:
    """Create a feedback edge for iterative improvement."""
    return create_bubble_edge(
        source_id, target_id, "feedback",
        source_handle, target_handle
    )


# =============================================================================
# Complete Gauntlet Workflow Definition Creation
# =============================================================================

def create_gauntlet_workflow_definition(
    workflow_name: str,
    problem_statement: str,
    gauntlet_config: Dict[str, Any],
    team_config: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create a complete gauntlet workflow definition with all necessary bubbles.
    
    Args:
        workflow_name: Name of the workflow
        problem_statement: Problem being solved
        gauntlet_config: Configuration for gauntlets
        team_config: Team configuration mapping
    
    Returns:
        Dict with 'nodes', 'edges', and 'metadata' for BubbleLab workflow
    """
    team_config = team_config or {}
    
    nodes = []
    edges = []
    
    # Create input bubble
    input_bubble = {
        "id": f"input_{uuid.uuid4().hex[:8]}",
        "type": "input",
        "position": GAUNTLET_NODE_POSITIONS["start"],
        "data": {
            "label": "📥 Input",
            "problem_statement": problem_statement,
            "status": "pending",
            "node_color": GAUNTLET_NODE_COLORS["input"],
        }
    }
    nodes.append(input_bubble)
    
    # Create LoongFlow AI Evaluation bubble
    loongeval_bubble = create_loongeval_bubble()
    nodes.append(loongeval_bubble)
    
    # Connect input to evaluation
    edges.append(create_bubble_edge(input_bubble["id"], loongeval_bubble["id"]))
    
    # Create Red Team bubble
    red_team_bubble = create_red_team_bubble(
        team_name=team_config.get("red_team", "Red Team"),
        attack_modes=gauntlet_config.get("attack_modes")
    )
    nodes.append(red_team_bubble)
    
    # Connect evaluation to Red Team
    edges.append(create_bubble_edge(loongeval_bubble["id"], red_team_bubble["id"]))
    
    # Create Blue Team bubble (for fixes)
    if gauntlet_config.get("include_blue_team", True):
        blue_team_bubble = create_blue_team_bubble(
            team_name=team_config.get("blue_team", "Blue Team")
        )
        nodes.append(blue_team_bubble)
        
        # Connect Red Team to Blue Team
        edges.append(create_bubble_edge(red_team_bubble["id"], blue_team_bubble["id"]))
        
        # Create feedback edge from Blue back to Red for iterations
        if gauntlet_config.get("max_iterations", 3) > 1:
            edges.append(create_feedback_edge(
                blue_team_bubble["id"], 
                red_team_bubble["id"],
                "feedback",
                "retry"
            ))
    
    # Create Gold Team bubble
    gold_team_bubble = create_gold_team_bubble(
        team_name=team_config.get("gold_team", "Gold Team")
    )
    nodes.append(gold_team_bubble)
    
    # Determine last team before Gold
    last_team = blue_team_bubble["id"] if gauntlet_config.get("include_blue_team", True) else red_team_bubble["id"]
    edges.append(create_bubble_edge(last_team, gold_team_bubble["id"]))
    
    # Create result bubble
    result_bubble = create_gauntlet_result_bubble(workflow_name)
    nodes.append(result_bubble)
    
    # Connect Gold Team to result
    edges.append(create_bubble_edge(gold_team_bubble["id"], result_bubble["id"]))
    
    workflow = {
        "id": str(uuid.uuid4()),
        "name": workflow_name,
        "description": f"Gauntlet workflow for: {problem_statement[:50]}...",
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "problem_statement": problem_statement,
            "gauntlet_config": gauntlet_config,
            "team_config": team_config,
            "created_at": datetime.now().isoformat(),
            "workflow_type": "gauntlet_3_round",
            "version": "1.0.0",
        }
    }
    
    logger.info(f"Created gauntlet workflow: {workflow['id']}")
    return workflow


def create_3_round_gauntlet_workflow(
    problem_statement: str,
    gauntlet_name: str = "Default Gauntlet",
    team_config: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create a complete 3-Round Gauntlet System workflow.
    
    Round 1: LoongFlow AI Evaluation
    Round 2: Red Team Attack
    Round 3: Gold Team Verification
    
    Args:
        problem_statement: The problem to solve
        gauntlet_name: Name of the gauntlet
        team_config: Team configuration
    
    Returns:
        Dict with complete workflow definition
    """
    gauntlet_config = {
        "attack_modes": ["prompt_injection", "logic_bomb", "edge_case"],
        "verification_modes": ["consistency", "completeness", "correctness"],
        "include_blue_team": True,
        "max_iterations": 3,
        "pass_threshold": 0.7,
    }
    
    return create_gauntlet_workflow_definition(
        workflow_name=gauntlet_name,
        problem_statement=problem_statement,
        gauntlet_config=gauntlet_config,
        team_config=team_config or {}
    )


# =============================================================================
# Bubble Update Functions
# =============================================================================

def update_bubble_status(
    bubble: Dict[str, Any],
    status: str,
    additional_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Update the status of a gauntlet bubble.
    
    Args:
        bubble: The bubble to update
        status: New status (pending, running, passed, failed, partial)
        additional_data: Additional data to merge into bubble data
    
    Returns:
        Updated bubble
    """
    bubble["data"]["status"] = status
    
    if additional_data:
        bubble["data"].update(additional_data)
    
    # Update color based on status
    status_colors = {
        "pending": "#FFEAA7",
        "running": "#74B9FF",
        "passed": "#00B894",
        "failed": "#FF7675",
        "partial": "#FDCB6E",
    }
    
    if status in status_colors:
        bubble["data"]["node_color"] = status_colors[status]
    
    return bubble


def add_bubble_result(
    bubble: Dict[str, Any],
    score: float,
    feedback: str,
    improvements: List[str] = None
) -> Dict[str, Any]:
    """
    Add result data to a gauntlet bubble.
    
    Args:
        bubble: The bubble to update
        score: Result score (0.0 to 1.0)
        feedback: Feedback message
        improvements: List of improvement suggestions
    
    Returns:
        Updated bubble
    """
    bubble["data"]["score"] = score
    bubble["data"]["feedback"] = feedback
    bubble["data"]["improvements"] = improvements or []
    
    # Determine status based on score
    if score >= 0.9:
        status = "passed"
    elif score >= 0.7:
        status = "partial"
    else:
        status = "failed"
    
    bubble["data"]["status"] = status
    
    return update_bubble_status(bubble, status)


# =============================================================================
# Bubble Serialization and Export
# =============================================================================

def serialize_bubble(bubble: Dict[str, Any]) -> str:
    """Serialize a bubble to JSON string."""
    import json
    return json.dumps(bubble, indent=2)


def serialize_workflow(workflow: Dict[str, Any]) -> str:
    """Serialize a workflow to JSON string."""
    import json
    return json.dumps(workflow, indent=2)


def export_workflow_to_json(
    workflow: Dict[str, Any],
    output_path: str
) -> bool:
    """
    Export a workflow to a JSON file.
    
    Args:
        workflow: Workflow definition to export
        output_path: Path to save the JSON file
    
    Returns:
        True if export was successful
    """
    import json
    import os
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(workflow, f, indent=2)
        
        logger.info(f"Exported workflow to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export workflow: {e}")
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def create_simple_gauntlet_bubble(
    gauntlet_type: str,
    label: str,
    team: str = "Team"
) -> Dict[str, Any]:
    """
    Create a simple gauntlet bubble with minimal configuration.
    
    Args:
        gauntlet_type: Type of gauntlet (red_team, blue_team, gold_team, evaluation)
        label: Display label for the bubble
        team: Team name
    
    Returns:
        Dict representing a gauntlet bubble
    """
    config = GauntletBubbleConfig(
        gauntlet_name=label,
        gauntlet_type=gauntlet_type,
        team_name=team,
        description=f"Simple {gauntlet_type} bubble"
    )
    
    return create_gauntlet_execution_bubble(config)


# =============================================================================
# Example Usage
# =============================================================================

def example_gauntlet_workflow():
    """Example: Create and export a complete gauntlet workflow."""
    workflow = create_3_round_gauntlet_workflow(
        problem_statement="Design a REST API for task management with authentication",
        gauntlet_name="API Design Gauntlet",
        team_config={
            "red_team": "Security Team",
            "blue_team": "Backend Team",
            "gold_team": "Architecture Team"
        }
    )
    
    # Export to file
    export_workflow_to_json(workflow, "gauntlet_workflow_example.json")
    
    return workflow


if __name__ == "__main__":
    # Run example
    workflow = example_gauntlet_workflow()
    print(f"Created workflow: {workflow['name']}")
    print(f"Nodes: {len(workflow['nodes'])}")
    print(f"Edges: {len(workflow['edges'])}")
