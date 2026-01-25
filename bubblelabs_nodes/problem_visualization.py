"""
Problem Hierarchy Visualization System

Provides comprehensive visualization capabilities for problem decomposition
hierarchies with multiple output formats (ASCII, HTML, Graphviz/DOT).
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ProblemStatus(Enum):
    """Problem execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class OutputFormat(Enum):
    """Visualization output formats"""
    ASCII = "ascii"
    HTML = "html"
    DOT = "dot"  # Graphviz


@dataclass
class TreeNode:
    """Represents a node in the problem tree"""
    problem_id: str
    status: ProblemStatus
    score: Optional[float] = None  # 0-100
    teams: List[str] = field(default_factory=list)  # Team history
    timing_ms: Optional[float] = None
    attempt_count: int = 0
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def depth(self) -> int:
        """Calculate depth of this node in tree"""
        if self.parent is None:
            return 0
        return 1 + self.parent.depth()

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0

    def subtree_size(self) -> int:
        """Calculate size of subtree including this node"""
        size = 1
        for child in self.children:
            size += child.subtree_size()
        return size


class ProblemTreeBuilder:
    """
    Builds tree representation of problem hierarchy.
    """

    def __init__(self):
        self.node_map: Dict[str, TreeNode] = {}

    def build_tree(self, problem: Dict[str, Any], parent: Optional[TreeNode] = None) -> TreeNode:
        """
        Build tree from problem hierarchy.

        Args:
            problem: Problem definition
            parent: Parent node (for recursion)

        Returns:
            TreeNode root
        """
        # Extract problem properties
        problem_id = problem.get('id', f"problem_{id(problem)}")
        status = ProblemStatus(problem.get('status', 'pending'))
        score = problem.get('score')
        teams = problem.get('teams', [])
        timing_ms = problem.get('timing_ms')
        attempt_count = problem.get('attempt_count', 0)
        metadata = {k: v for k, v in problem.items()
                   if k not in ['id', 'status', 'score', 'teams', 'timing_ms', 'attempt_count', 'subproblems']}

        # Create node
        node = TreeNode(
            problem_id=problem_id,
            status=status,
            score=score,
            teams=teams,
            timing_ms=timing_ms,
            attempt_count=attempt_count,
            metadata=metadata,
            parent=parent
        )

        # Register node
        self.node_map[problem_id] = node

        # Check for circular references
        if self._detect_circular_reference(node):
            logger.warning(f"Circular reference detected at {problem_id}")
            return node

        # Recursively build children
        subproblems = problem.get('subproblems', [])
        if subproblems:
            for subproblem in subproblems:
                child = self.build_tree(subproblem, parent=node)
                node.children.append(child)

        return node

    def _detect_circular_reference(self, node: TreeNode, visited: Optional[set] = None) -> bool:
        """Detect circular references in tree"""
        if visited is None:
            visited = set()

        if node.problem_id in visited:
            return True

        visited.add(node.problem_id)

        if node.parent:
            return self._detect_circular_reference(node.parent, visited)

        return False

    def validate_tree(self, root: TreeNode) -> Tuple[bool, List[str]]:
        """
        Validate tree structure.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Check for circular references
        if self._detect_circular_reference(root):
            errors.append("Circular reference detected")

        # Check all nodes have valid IDs
        for node_id, node in self.node_map.items():
            if not node_id:
                errors.append(f"Node has empty ID")

            if node.status not in ProblemStatus:
                errors.append(f"Node {node_id} has invalid status")

        # Check parent-child consistency
        for node_id, node in self.node_map.items():
            if node.parent:
                if node not in node.parent.children:
                    errors.append(f"Node {node_id} parent doesn't reference it as child")

            for child in node.children:
                if child.parent is not node:
                    errors.append(f"Node {node_id} child doesn't reference it as parent")

        return len(errors) == 0, errors


class ASCIITreeRenderer:
    """
    Renders problem tree as ASCII art with box-drawing characters.
    """

    def __init__(self, show_metadata: bool = True, show_timing: bool = True, show_teams: bool = True):
        self.show_metadata = show_metadata
        self.show_timing = show_timing
        self.show_teams = show_teams

    def render(self, root: TreeNode) -> str:
        """
        Render tree as ASCII art.

        Args:
            root: Root node of tree

        Returns:
            ASCII string representation
        """
        lines = []
        lines.append(self._render_header(root))
        lines.append(self._render_node(root, "", is_last=True))
        return "\n".join(lines)

    def _render_header(self, root: TreeNode) -> str:
        """Render tree header"""
        size = root.subtree_size()
        depth = self._max_depth(root)
        return f"Problem Hierarchy Tree ({size} nodes, {depth} levels)"

    def _render_node(
        self,
        node: TreeNode,
        prefix: str,
        is_last: bool
    ) -> str:
        """Render single node and its children"""
        lines = []

        # Build node representation
        node_str = self._node_to_string(node)

        # Add prefix with tree connector
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node_str}")

        # Render children
        if node.children:
            child_prefix = prefix + ("    " if is_last else "│   ")

            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                child_str = self._render_node(child, child_prefix, is_last_child)
                lines.append(child_str)

        return "\n".join(lines)

    def _node_to_string(self, node: TreeNode) -> str:
        """Convert node to string representation"""
        parts = [node.problem_id]

        # Add status with color indicator
        status_symbol = self._status_symbol(node.status)
        parts.append(f"[{status_symbol}{node.status.value}]")

        # Add score if available
        if node.score is not None:
            parts.append(f"Score: {node.score:.0f}/100")

        # Add timing
        if self.show_timing and node.timing_ms is not None:
            parts.append(f"({node.timing_ms:.0f}ms)")

        # Add teams
        if self.show_teams and node.teams:
            parts.append(f"Teams: {'→'.join(node.teams)}")

        # Add attempt count
        if node.attempt_count > 0:
            parts.append(f"Attempt #{node.attempt_count}")

        return " ".join(parts)

    def _status_symbol(self, status: ProblemStatus) -> str:
        """Get symbol for status"""
        symbols = {
            ProblemStatus.PENDING: "⏳",
            ProblemStatus.IN_PROGRESS: "🔄",
            ProblemStatus.COMPLETE: "✅",
            ProblemStatus.FAILED: "❌",
        }
        return symbols.get(status, "?")

    def _max_depth(self, node: TreeNode) -> int:
        """Calculate maximum depth of tree"""
        if not node.children:
            return node.depth()
        return max(child.depth() for child in node.children)


class HTMLTreeRenderer:
    """
    Renders problem tree as interactive HTML with CSS styling.
    """

    def __init__(self, show_metadata: bool = True, show_timing: bool = True, show_teams: bool = True):
        self.show_metadata = show_metadata
        self.show_timing = show_timing
        self.show_teams = show_teams

    def render_html(self, root: TreeNode) -> str:
        """
        Render tree as HTML.

        Args:
            root: Root node of tree

        Returns:
            HTML string
        """
        html_parts = []

        # HTML header
        html_parts.append(self._html_header(root))

        # Tree container
        html_parts.append('<div class="problem-tree">')
        html_parts.append(self._render_node_html(root, 0))
        html_parts.append('</div>')

        # HTML footer with script
        html_parts.append(self._html_footer())

        return "\n".join(html_parts)

    def _html_header(self, root: TreeNode) -> str:
        """Generate HTML header with CSS"""
        size = root.subtree_size()
        depth = self._max_depth(root)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Problem Hierarchy Tree</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}

        .problem-tree {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}

        .tree-header {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }}

        .tree-node {{
            margin-left: 20px;
            padding: 10px;
            border-left: 2px solid #ddd;
            transition: all 0.3s;
        }}

        .tree-node:hover {{
            background-color: #f9f9f9;
            border-left-color: #007bff;
        }}

        .node-header {{
            cursor: pointer;
            user-select: none;
        }}

        .node-id {{
            font-weight: bold;
            color: #007bff;
        }}

        .node-status {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 8px;
        }}

        .status-pending {{ background-color: #ffc107; color: #000; }}
        .status-in_progress {{ background-color: #17a2b8; color: #fff; }}
        .status-complete {{ background-color: #28a745; color: #fff; }}
        .status-failed {{ background-color: #dc3545; color: #fff; }}

        .node-score {{
            margin-left: 8px;
            font-weight: bold;
        }}

        .node-timing {{
            color: #6c757d;
            font-size: 14px;
            margin-left: 8px;
        }}

        .node-teams {{
            color: #495057;
            font-size: 14px;
            margin-left: 8px;
        }}

        .node-metadata {{
            margin-top: 8px;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 14px;
        }}

        .node-children {{
            display: none;
            margin-top: 8px;
        }}

        .node-children.expanded {{
            display: block;
        }}

        .expand-icon {{
            display: inline-block;
            width: 16px;
            margin-right: 4px;
            transition: transform 0.3s;
        }}

        .expand-icon.expanded {{
            transform: rotate(90deg);
        }}
    </style>
</head>
<body>
    <div class="tree-header">Problem Hierarchy Tree ({size} nodes, {depth} levels)</div>
"""

    def _render_node_html(self, node: TreeNode, level: int) -> str:
        """Render single node as HTML"""
        has_children = len(node.children) > 0

        html = f'<div class="tree-node">\n'
        html += '  <div class="node-header" onclick="toggleNode(this)">\n'

        # Expand icon
        if has_children:
            html += '    <span class="expand-icon">▶</span>\n'
        else:
            html += '    <span class="expand-icon">●</span>\n'

        # Node ID
        html += f'    <span class="node-id">{node.problem_id}</span>\n'

        # Status badge
        status_class = f"status-{node.status.value}"
        html += f'    <span class="node-status {status_class}">{node.status.value}</span>\n'

        # Score
        if node.score is not None:
            score_color = self._score_color(node.score)
            html += f'    <span class="node-score" style="color: {score_color}">{node.score:.0f}/100</span>\n'

        # Timing
        if self.show_timing and node.timing_ms is not None:
            html += f'    <span class="node-timing">⏱ {node.timing_ms:.0f}ms</span>\n'

        # Teams
        if self.show_teams and node.teams:
            html += f'    <span class="node-teams">👥 {" → ".join(node.teams)}</span>\n'

        # Attempt count
        if node.attempt_count > 0:
            html += f'    <span class="node-timing">Attempt #{node.attempt_count}</span>\n'

        html += '  </div>\n'

        # Metadata tooltip
        if self.show_metadata and node.metadata:
            html += '  <div class="node-metadata">\n'
            for key, value in node.metadata.items():
                html += f'    <div><strong>{key}:</strong> {value}</div>\n'
            html += '  </div>\n'

        # Children
        if has_children:
            html += '  <div class="node-children">\n'
            for child in node.children:
                html += self._render_node_html(child, level + 1)
            html += '  </div>\n'

        html += '</div>\n'

        return html

    def _score_color(self, score: float) -> str:
        """Get color for score"""
        if score >= 80:
            return "#28a745"  # Green
        elif score >= 60:
            return "#ffc107"  # Yellow
        elif score >= 40:
            return "#fd7e14"  # Orange
        else:
            return "#dc3545"  # Red

    def _html_footer(self) -> str:
        """Generate HTML footer with JavaScript"""
        return """<script>
    function toggleNode(header) {
        const icon = header.querySelector('.expand-icon');
        const children = header.parentElement.querySelector('.node-children');

        if (children) {
            children.classList.toggle('expanded');
            icon.classList.toggle('expanded');
        }
    }

    // Auto-expand first level
    document.addEventListener('DOMContentLoaded', function() {
        const rootChildren = document.querySelector('.node-children');
        if (rootChildren) {
            rootChildren.classList.add('expanded');
            const rootIcon = document.querySelector('.expand-icon');
            if (rootIcon) {
                rootIcon.classList.add('expanded');
            }
        }
    });
</script>
</body>
</html>"""

    def _max_depth(self, node: TreeNode) -> int:
        """Calculate maximum depth of tree"""
        if not node.children:
            return node.depth()
        return max(child.depth() for child in node.children)


class GraphvizTreeRenderer:
    """
    Renders problem tree as Graphviz DOT format.
    """

    def __init__(self, show_metadata: bool = False, show_timing: bool = True, show_teams: bool = True):
        self.show_metadata = show_metadata
        self.show_timing = show_timing
        self.show_teams = show_teams
        self.node_counter = 0

    def render_dot(self, root: TreeNode) -> str:
        """
        Render tree as Graphviz DOT format.

        Args:
            root: Root node of tree

        Returns:
            DOT format string
        """
        lines = []

        # DOT header
        lines.append("digraph ProblemHierarchy {")
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box, style=rounded];")
        lines.append("")

        # Render nodes
        self.node_counter = 0
        self._render_nodes_dot(root, lines)

        # Render edges
        self._render_edges_dot(root, lines)

        # DOT footer
        lines.append("}")

        return "\n".join(lines)

    def _render_nodes_dot(self, node: TreeNode, lines: List[str]):
        """Render single node in DOT format"""
        node_id = f'node_{self.node_counter}'
        self.node_counter += 1

        # Build label
        label_parts = [node.problem_id]

        # Status
        label_parts.append(f"\\nStatus: {node.status.value}")

        # Score
        if node.score is not None:
            label_parts.append(f"\\nScore: {node.score:.0f}/100")

        # Timing
        if self.show_timing and node.timing_ms is not None:
            label_parts.append(f"\\nTime: {node.timing_ms:.0f}ms")

        # Teams
        if self.show_teams and node.teams:
            label_parts.append(f"\\nTeams: {' → '.join(node.teams)}")

        label = '"'.join(label_parts) + '"'

        # Node style based on status
        color = self._status_color(node.status)
        style = "filled,rounded" if color else "rounded"
        fillcolor = color if color else "white"

        lines.append(f'  {node_id} [label={label}, style="{style}", fillcolor="{fillcolor}"];')

        # Store node ID in metadata for edge rendering
        node.metadata['_dot_id'] = node_id

        # Render children
        for child in node.children:
            self._render_nodes_dot(child, lines)

    def _render_edges_dot(self, node: TreeNode, lines: List[str]):
        """Render edges in DOT format"""
        parent_id = node.metadata.get('_dot_id')

        for child in node.children:
            child_id = child.metadata.get('_dot_id')
            if parent_id and child_id:
                lines.append(f'  {parent_id} -> {child_id};')

            # Recursively render child edges
            self._render_edges_dot(child, lines)

    def _status_color(self, status: ProblemStatus) -> str:
        """Get fill color for status"""
        colors = {
            ProblemStatus.PENDING: "#fff3cd",
            ProblemStatus.IN_PROGRESS: "#d1ecf1",
            ProblemStatus.COMPLETE: "#d4edda",
            ProblemStatus.FAILED: "#f8d7da",
        }
        return colors.get(status, "")


class VisualizationAPI:
    """
    High-level API for visualizing problem hierarchies.
    """

    def __init__(self):
        self.builder = ProblemTreeBuilder()

    def visualize_problem(
        self,
        problem: Dict[str, Any],
        format: OutputFormat = OutputFormat.ASCII,
        show_metadata: bool = True,
        show_timing: bool = True,
        show_teams: bool = True
    ) -> str:
        """
        Visualize a problem hierarchy.

        Args:
            problem: Problem definition
            format: Output format (ascii, html, dot)
            show_metadata: Include metadata in output
            show_timing: Include timing information
            show_teams: Include team history

        Returns:
            Formatted visualization string
        """
        # Build tree
        root = self.builder.build_tree(problem)

        # Validate tree
        is_valid, errors = self.builder.validate_tree(root)
        if not is_valid:
            logger.warning(f"Tree validation errors: {errors}")

        # Render based on format
        if format == OutputFormat.ASCII:
            renderer = ASCIITreeRenderer(show_metadata, show_timing, show_teams)
            return renderer.render(root)
        elif format == OutputFormat.HTML:
            renderer = HTMLTreeRenderer(show_metadata, show_timing, show_teams)
            return renderer.render_html(root)
        elif format == OutputFormat.DOT:
            renderer = GraphvizTreeRenderer(show_metadata, show_timing, show_teams)
            return renderer.render_dot(root)
        else:
            raise ValueError(f"Unsupported format: {format}")


def visualize_problem(
    problem: Dict[str, Any],
    format: str = "ascii",
    show_metadata: bool = True,
    show_timing: bool = True,
    show_teams: bool = True
) -> str:
    """
    Convenience function to visualize a problem hierarchy.

    Args:
        problem: Problem definition
        format: Output format ("ascii", "html", "dot")
        show_metadata: Include metadata in output
        show_timing: Include timing information
        show_teams: Include team history

    Returns:
        Formatted visualization string

    Example:
        >>> problem = {
        ...     'id': 'root',
        ...     'status': 'complete',
        ...     'score': 85,
        ...     'subproblems': [
        ...         {'id': 'child1', 'status': 'complete', 'score': 90},
        ...         {'id': 'child2', 'status': 'failed'}
        ...     ]
        ... }
        >>> print(visualize_problem(problem, format='ascii'))
    """
    api = VisualizationAPI()
    output_format = OutputFormat(format.lower())
    return api.visualize_problem(problem, output_format, show_metadata, show_timing, show_teams)
