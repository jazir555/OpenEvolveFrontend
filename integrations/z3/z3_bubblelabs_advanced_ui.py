"""
Advanced Z3 BubbleLabs UI Components

Rich, interactive visualization components for Z3 integration in BubbleLabs:
- Interactive constraint solver visualizer
- Proof tree explorer
- Optimization landscape viewer
- Real-time solving progress
- Comparative analysis views
- Export capabilities

Author: OpenEvolve
Created: 2026-01-31
"""


import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# Import base components
try:
    from z3_leanaide_bubblelabs_ui import (
        Z3BubbleLabsUIManager, NodeStatus,
        Z3SolverNodeState, Z3TheoremProverNodeState
    )
    BASE_UI_AVAILABLE = True
except ImportError:
    BASE_UI_AVAILABLE = False
    logger.warning("Base BubbleLabs UI not available")

try:
    from z3prover_integration import Z3SolverResult, Z3ResultStatus
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# =============================================================================
# Data Classes for Advanced Visualization
# =============================================================================

@dataclass
class ConstraintVisualization:
    """Visual representation of a constraint."""
    constraint_id: str
    expression: str
    status: str  # "active", "satisfied", "violated", "relaxed"
    color: str = "#3b82f6"  # Default blue
    importance: float = 1.0
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.constraint_id,
            "expression": self.expression,
            "status": self.status,
            "color": self.color,
            "importance": self.importance
        }


@dataclass
class VariableVisualization:
    """Visual representation of a variable."""
    var_name: str
    var_type: str
    current_value: Optional[Any] = None
    domain_min: Optional[float] = None
    domain_max: Optional[float] = None
    is_integer: bool = True
    color: str = "#10b981"  # Default green
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.var_name,
            "type": self.var_type,
            "value": self.current_value,
            "domain": [self.domain_min, self.domain_max] if self.domain_min and self.domain_max else None,
            "color": self.color
        }


@dataclass
class ProofTreeNode:
    """Node in proof tree visualization."""
    node_id: str
    tactic: str
    goal: str
    status: str  # "open", "closed", "error"
    children: List['ProofTreeNode'] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "tactic": self.tactic,
            "goal": self.goal[:100] + "..." if len(self.goal) > 100 else self.goal,
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
            "depth": self.depth,
            "execution_time": self.execution_time
        }


@dataclass
class OptimizationPoint:
    """Point in optimization landscape."""
    coordinates: Dict[str, float]
    objective_value: float
    is_feasible: bool
    is_optimal: bool = False
    iteration: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates,
            "objective": self.objective_value,
            "feasible": self.is_feasible,
            "optimal": self.is_optimal,
            "iteration": self.iteration
        }


@dataclass
class SolvingProgress:
    """Progress information for solving."""
    stage: str
    percent_complete: float
    current_iteration: int
    total_iterations: int
    current_objective: Optional[float] = None
    best_objective: Optional[float] = None
    messages: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "percent": self.percent_complete,
            "iteration": f"{self.current_iteration}/{self.total_iterations}",
            "current_obj": self.current_objective,
            "best_obj": self.best_objective,
            "messages": self.messages[-5:],  # Last 5 messages
            "timestamp": self.timestamp
        }


# =============================================================================
# Advanced UI Manager
# =============================================================================

class Z3AdvancedBubbleLabsUI:
    """
    Advanced UI components for Z3 in BubbleLabs.
    
    Provides rich visualizations:
    - Interactive constraint graphs
    - Proof tree explorers
    - Optimization landscapes
    - Real-time progress tracking
    - CAV-NLP enhanced solving options
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_ui = None
        if BASE_UI_AVAILABLE:
            from z3_leanaide_bubblelabs_ui import get_z3_bubblelabs_ui
            self.base_ui = get_z3_bubblelabs_ui()
        
        # CAV-NLP integration
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for BubbleLabs Advanced UI")
        
        # State storage for visualizations
        self._constraint_viz: Dict[str, List[ConstraintVisualization]] = {}
        self._variable_viz: Dict[str, List[VariableVisualization]] = {}
        self._proof_trees: Dict[str, ProofTreeNode] = {}
        self._optimization_landscapes: Dict[str, List[OptimizationPoint]] = {}
        self._progress: Dict[str, SolvingProgress] = {}
    
    # =====================================================================
    # Interactive Constraint Visualizer
    # =====================================================================
    
    def create_constraint_visualization(
        self,
        node_id: str,
        variables: List[VariableVisualization],
        constraints: List[ConstraintVisualization]
    ) -> Dict[str, Any]:
        """
        Create interactive constraint visualization.
        
        Returns visualization data for a constraint network graph.
        """
        self._variable_viz[node_id] = variables
        self._constraint_viz[node_id] = constraints
        
        # Build graph structure
        nodes = []
        edges = []
        
        # Add variable nodes
        for var in variables:
            nodes.append({
                "id": f"var_{var.var_name}",
                "type": "variable",
                "label": f"{var.var_name} = {var.current_value}",
                "color": var.color,
                "data": var.to_dict()
            })
        
        # Add constraint nodes and edges
        for constraint in constraints:
            constraint_node_id = f"constraint_{constraint.constraint_id}"
            nodes.append({
                "id": constraint_node_id,
                "type": "constraint",
                "label": constraint.expression,
                "color": constraint.color,
                "status": constraint.status,
                "data": constraint.to_dict()
            })
            
            # Connect to variables (simplified - would parse expression)
            for var in variables:
                if var.var_name in constraint.expression:
                    edges.append({
                        "source": f"var_{var.var_name}",
                        "target": constraint_node_id,
                        "type": "participates_in"
                    })
        
        return {
            "node_id": node_id,
            "graph": {
                "nodes": nodes,
                "edges": edges
            },
            "statistics": {
                "variable_count": len(variables),
                "constraint_count": len(constraints),
                "satisfied": sum(1 for c in constraints if c.status == "satisfied"),
                "violated": sum(1 for c in constraints if c.status == "violated")
            }
        }
    
    def update_constraint_status(
        self,
        node_id: str,
        constraint_id: str,
        new_status: str,
        color: Optional[str] = None
    ):
        """Update constraint status in visualization."""
        if node_id in self._constraint_viz:
            for constraint in self._constraint_viz[node_id]:
                if constraint.constraint_id == constraint_id:
                    constraint.status = new_status
                    if color:
                        constraint.color = color
                    break
    
    # =====================================================================
    # Proof Tree Explorer
    # =====================================================================
    
    def create_proof_tree(
        self,
        node_id: str,
        root: ProofTreeNode
    ) -> Dict[str, Any]:
        """
        Create proof tree visualization.
        
        Returns hierarchical tree structure for proof exploration.
        """
        self._proof_trees[node_id] = root
        
        def count_nodes(node: ProofTreeNode) -> Tuple[int, int, int]:
            """Count total, closed, and open nodes."""
            total = 1
            closed = 1 if node.status == "closed" else 0
            open_nodes = 1 if node.status == "open" else 0
            
            for child in node.children:
                t, c, o = count_nodes(child)
                total += t
                closed += c
                open_nodes += o
            
            return total, closed, open_nodes
        
        total, closed, open_nodes = count_nodes(root)
        
        return {
            "node_id": node_id,
            "tree": root.to_dict(),
            "statistics": {
                "total_nodes": total,
                "closed_nodes": closed,
                "open_nodes": open_nodes,
                "completion": closed / total if total > 0 else 0
            },
            "interaction": {
                "expandable": True,
                "collapsible": True,
                "searchable": True,
                "exportable": ["json", "pdf", "png"]
            }
        }
    
    def expand_proof_node(
        self,
        node_id: str,
        tree_node_id: str
    ) -> Optional[ProofTreeNode]:
        """Get children of a proof tree node for lazy loading."""
        def find_node(node: ProofTreeNode, target_id: str) -> Optional[ProofTreeNode]:
            if node.node_id == target_id:
                return node
            for child in node.children:
                found = find_node(child, target_id)
                if found:
                    return found
            return None
        
        if node_id in self._proof_trees:
            return find_node(self._proof_trees[node_id], tree_node_id)
        return None
    
    # =====================================================================
    # Optimization Landscape Viewer
    # =====================================================================
    
    def create_optimization_landscape(
        self,
        node_id: str,
        points: List[OptimizationPoint],
        dimensions: List[str]
    ) -> Dict[str, Any]:
        """
        Create optimization landscape visualization.
        
        Returns data for 2D/3D landscape visualization.
        """
        self._optimization_landscapes[node_id] = points
        
        # Calculate bounds
        if points:
            obj_values = [p.objective_value for p in points]
            feasible_points = [p for p in points if p.is_feasible]
            optimal_points = [p for p in points if p.is_optimal]
        else:
            obj_values = []
            feasible_points = []
            optimal_points = []
        
        return {
            "node_id": node_id,
            "dimensions": dimensions,
            "points": [p.to_dict() for p in points],
            "visualization": {
                "type": "scatter3d" if len(dimensions) >= 2 else "scatter",
                "color_by": "objective",
                "size_by": "feasibility"
            },
            "statistics": {
                "total_points": len(points),
                "feasible_points": len(feasible_points),
                "optimal_points": len(optimal_points),
                "objective_range": [min(obj_values), max(obj_values)] if obj_values else [0, 0]
            },
            "interaction": {
                "rotatable": True,
                "zoomable": True,
                "selectable": True,
                "brush_select": True
            }
        }
    
    def add_optimization_point(
        self,
        node_id: str,
        point: OptimizationPoint
    ):
        """Add point to optimization landscape (for incremental updates)."""
        if node_id not in self._optimization_landscapes:
            self._optimization_landscapes[node_id] = []
        self._optimization_landscapes[node_id].append(point)
    
    # =====================================================================
    # Real-time Progress Tracking
    # =====================================================================
    
    def create_progress_tracker(
        self,
        node_id: str,
        total_iterations: int,
        stage_name: str = "Solving"
    ) -> SolvingProgress:
        """Create progress tracker for real-time updates."""
        progress = SolvingProgress(
            stage=stage_name,
            percent_complete=0.0,
            current_iteration=0,
            total_iterations=total_iterations
        )
        self._progress[node_id] = progress
        return progress
    
    def update_progress(
        self,
        node_id: str,
        iteration: Optional[int] = None,
        percent: Optional[float] = None,
        message: Optional[str] = None,
        current_objective: Optional[float] = None,
        best_objective: Optional[float] = None
    ) -> Optional[SolvingProgress]:
        """Update progress tracker."""
        if node_id not in self._progress:
            return None
        
        progress = self._progress[node_id]
        
        if iteration is not None:
            progress.current_iteration = iteration
            progress.percent_complete = (iteration / progress.total_iterations) * 100
        
        if percent is not None:
            progress.percent_complete = percent
        
        if message:
            progress.messages.append(message)
        
        if current_objective is not None:
            progress.current_objective = current_objective
        
        if best_objective is not None:
            progress.best_objective = best_objective
        
        progress.timestamp = datetime.utcnow().isoformat()
        return progress
    
    def get_progress(self, node_id: str) -> Optional[SolvingProgress]:
        """Get current progress."""
        return self._progress.get(node_id)
    
    # =====================================================================
    # Comparative Analysis Views
    # =====================================================================
    
    def create_comparison_view(
        self,
        comparison_id: str,
        results: List[Dict[str, Any]],
        labels: List[str]
    ) -> Dict[str, Any]:
        """
        Create side-by-side comparison of multiple results.
        
        Useful for comparing:
        - Different solver strategies
        - Z3 vs Lean results
        - Multiple solutions
        """
        if len(results) != len(labels):
            raise ValueError("Results and labels must have same length")
        
        # Extract common metrics
        metrics = {}
        for result in results:
            for key, value in result.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Create comparison table
        table = []
        for key, values in metrics.items():
            row = {"metric": key}
            for label, value in zip(labels, values):
                row[label] = value
            table.append(row)
        
        return {
            "comparison_id": comparison_id,
            "type": "side_by_side",
            "results": [
                {"label": label, "data": result}
                for label, result in zip(labels, results)
            ],
            "table": table,
            "visualization": {
                "chart_types": ["bar", "radar", "parallel"],
                "metrics": list(metrics.keys())
            }
        }
    
    # =====================================================================
    # Export Capabilities
    # =====================================================================
    
    def export_visualization(
        self,
        node_id: str,
        viz_type: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export visualization data.
        
        Supported formats: json, csv, svg, png (for charts)
        """
        data = None
        
        if viz_type == "constraints" and node_id in self._constraint_viz:
            data = {
                "variables": [v.to_dict() for v in self._variable_viz.get(node_id, [])],
                "constraints": [c.to_dict() for c in self._constraint_viz.get(node_id, [])]
            }
        
        elif viz_type == "proof" and node_id in self._proof_trees:
            data = self._proof_trees[node_id].to_dict()
        
        elif viz_type == "optimization" and node_id in self._optimization_landscapes:
            data = [p.to_dict() for p in self._optimization_landscapes[node_id]]
        
        elif viz_type == "progress" and node_id in self._progress:
            data = self._progress[node_id].to_dict()
        
        return {
            "node_id": node_id,
            "type": viz_type,
            "format": format,
            "data": data,
            "exported_at": datetime.utcnow().isoformat()
        }
    
    # =====================================================================
    # CAV-NLP Options UI
    # =====================================================================
    
    def create_cav_nlp_options_panel(self, node_id: str) -> Dict[str, Any]:
        """
        Create CAV-NLP options panel for the UI.
        
        Returns configuration options for CAV-NLP enhanced solving.
        """
        return {
            "node_id": node_id,
            "type": "cav_nlp_options",
            "title": "CAV-NLP Enhanced Solving",
            "enabled": self.use_cav_nlp,
            "available": CAV_NLP_AVAILABLE,
            "options": [
                {
                    "id": "use_natural_language_input",
                    "label": "Enable Natural Language Input",
                    "type": "toggle",
                    "default": True,
                    "description": "Allow problems to be specified in natural language"
                },
                {
                    "id": "auto_formalization",
                    "label": "Auto-Formalization",
                    "type": "toggle",
                    "default": True,
                    "description": "Automatically convert NL to formal constraints"
                },
                {
                    "id": "formalization_cache",
                    "label": "Cache Formalizations",
                    "type": "toggle",
                    "default": True,
                    "description": "Cache formalization results for reuse"
                },
                {
                    "id": "verification_mode",
                    "label": "Verification Mode",
                    "type": "select",
                    "options": ["standard", "enhanced", "strict"],
                    "default": "enhanced",
                    "description": "Level of verification to apply"
                },
                {
                    "id": "constraint_extraction",
                    "label": "Extract Implicit Constraints",
                    "type": "toggle",
                    "default": True,
                    "description": "Automatically detect implicit constraints from text"
                }
            ],
            "actions": [
                {
                    "id": "test_cav_nlp",
                    "label": "Test CAV-NLP Connection",
                    "type": "button",
                    "enabled": CAV_NLP_AVAILABLE
                },
                {
                    "id": "view_formalization_history",
                    "label": "View Formalization History",
                    "type": "button",
                    "enabled": self.use_cav_nlp
                }
            ]
        }
    
    def get_cav_nlp_status(self, node_id: str) -> Dict[str, Any]:
        """Get current CAV-NLP status for display in UI."""
        return {
            "node_id": node_id,
            "available": CAV_NLP_AVAILABLE,
            "enabled": self.use_cav_nlp,
            "solver_ready": hasattr(self, 'enhanced_solver') and self.enhanced_solver is not None,
            "service_ready": hasattr(self, 'math_service') and self.math_service is not None,
            "capabilities": [
                "natural_language_formalization",
                "constraint_extraction",
                "auto_verification"
            ] if self.use_cav_nlp else []
        }
    
    # =====================================================================
    # Dashboard Components
    # =====================================================================
    
    def create_solver_dashboard(
        self,
        node_id: str,
        solver_state: Any
    ) -> Dict[str, Any]:
        """
        Create comprehensive solver dashboard.
        
        Combines multiple visualizations into unified dashboard.
        """
        dashboard = {
            "node_id": node_id,
            "layout": "grid",
            "components": [],
            "refresh_rate_ms": 1000
        }
        
        # Add progress widget if available
        if node_id in self._progress:
            dashboard["components"].append({
                "type": "progress",
                "title": "Solving Progress",
                "data": self._progress[node_id].to_dict(),
                "position": {"x": 0, "y": 0, "w": 6, "h": 2}
            })
        
        # Add constraint graph if available
        if node_id in self._constraint_viz:
            dashboard["components"].append({
                "type": "graph",
                "title": "Constraint Network",
                "data": self.create_constraint_visualization(
                    node_id,
                    self._variable_viz.get(node_id, []),
                    self._constraint_viz.get(node_id, [])
                ),
                "position": {"x": 6, "y": 0, "w": 6, "h": 4}
            })
        
        # Add statistics widget
        dashboard["components"].append({
            "type": "stats",
            "title": "Solver Statistics",
            "data": {
                "status": getattr(solver_state, 'status', 'unknown'),
                "execution_time": getattr(solver_state, 'execution_time', 0),
                "result": getattr(solver_state, 'result_status', 'unknown')
            },
            "position": {"x": 0, "y": 2, "w": 6, "h": 2}
        })
        
        return dashboard


# =============================================================================
# Global Instance
# =============================================================================

_advanced_ui: Optional[Z3AdvancedBubbleLabsUI] = None


def get_z3_advanced_bubblelabs_ui() -> Z3AdvancedBubbleLabsUI:
    """Get global advanced UI instance."""
    global _advanced_ui
    if _advanced_ui is None:
        _advanced_ui = Z3AdvancedBubbleLabsUI()
    return _advanced_ui


# =============================================================================
# Example Usage
# =============================================================================

def example_constraint_visualization():
    """Example: Constraint visualization."""
    ui = get_z3_advanced_bubblelabs_ui()
    
    variables = [
        VariableVisualization("x", "Int", 5, 0, 10),
        VariableVisualization("y", "Int", 8, 0, 10)
    ]
    
    constraints = [
        ConstraintVisualization("c1", "(> x 0)", "satisfied", "#10b981"),
        ConstraintVisualization("c2", "(< x 10)", "satisfied", "#10b981"),
        ConstraintVisualization("c3", "(= y (+ x 5))", "satisfied", "#10b981")
    ]
    
    viz = ui.create_constraint_visualization("node_1", variables, constraints)
    
    print("Constraint Visualization:")
    print(f"  Nodes: {len(viz['graph']['nodes'])}")
    print(f"  Edges: {len(viz['graph']['edges'])}")
    print(f"  Satisfied: {viz['statistics']['satisfied']}")


def example_proof_tree():
    """Example: Proof tree visualization."""
    ui = get_z3_advanced_bubblelabs_ui()
    
    root = ProofTreeNode(
        node_id="root",
        tactic="intro",
        goal="forall x, x > 0 -> x + 1 > 0",
        status="closed"
    )
    
    child1 = ProofTreeNode(
        node_id="child1",
        tactic="intro",
        goal="x > 0 -> x + 1 > 0",
        status="closed",
        parent_id="root",
        depth=1
    )
    
    child2 = ProofTreeNode(
        node_id="child2",
        tactic="linarith",
        goal="x + 1 > 0",
        status="closed",
        parent_id="child1",
        depth=2
    )
    
    child1.children.append(child2)
    root.children.append(child1)
    
    viz = ui.create_proof_tree("proof_1", root)
    
    print("\nProof Tree:")
    print(f"  Total nodes: {viz['statistics']['total_nodes']}")
    print(f"  Completion: {viz['statistics']['completion']:.1%}")


def example_optimization_landscape():
    """Example: Optimization landscape."""
    ui = get_z3_advanced_bubblelabs_ui()
    
    points = [
        OptimizationPoint({"x": 0, "y": 0}, 10.0, True),
        OptimizationPoint({"x": 1, "y": 1}, 8.0, True),
        OptimizationPoint({"x": 2, "y": 2}, 6.0, True, is_optimal=True),
        OptimizationPoint({"x": 3, "y": 3}, 15.0, False),
    ]
    
    viz = ui.create_optimization_landscape("opt_1", points, ["x", "y"])
    
    print("\nOptimization Landscape:")
    print(f"  Points: {viz['statistics']['total_points']}")
    print(f"  Feasible: {viz['statistics']['feasible_points']}")
    print(f"  Optimal: {viz['statistics']['optimal_points']}")


if __name__ == "__main__":
    print("Z3 Advanced BubbleLabs UI")
    print("=" * 50)
    
    example_constraint_visualization()
    example_proof_tree()
    example_optimization_landscape()
