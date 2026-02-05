"""
Problem Hierarchy Visualization for OpenEvolve Gauntlet System

Generates visual tree diagrams showing problem decomposition, solution status,
and team contributions for better debugging and understanding.

Key Features:
- ASCII art rendering for terminal output
- HTML renderer for web UI
- DOT/Graphviz renderer for diagrams
- Metadata display (status, score, teams, timing)
- Multiple output formats
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProblemNode:
    """Represents a node in the problem hierarchy tree"""
    problem_id: str
    statement: str
    status: str  # pending, in_progress, complete, failed
    score: Optional[float] = None
    teams: List[str] = None
    timing: Optional[float] = None
    level: int = 0
    children: List['ProblemNode'] = None
    parent: Optional['ProblemNode'] = None
    attempt_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.teams is None:
            self.teams = []
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}


class ProblemTreeBuilder:
    """
    Builds problem hierarchy trees from execution data.
    """

    def build_tree(self, problem: Dict[str, Any], level: int = 0) -> ProblemNode:
        """
        Build a tree from a problem and its subproblems.

        Args:
            problem: Problem definition
            level: Current hierarchy level

        Returns:
            ProblemNode representing the tree
        """
        node = ProblemNode(
            problem_id=problem.get('id', f'problem_{level}'),
            statement=problem.get('statement', 'Unknown Problem'),
            status=problem.get('status', 'pending'),
            score=problem.get('score'),
            teams=problem.get('teams', []),
            timing=problem.get('timing'),
            level=level,
            metadata=problem.get('metadata', {})
        )

        # Recursively build children
        subproblems = problem.get('subproblems', [])
        for subproblem in subproblems:
            child_node = self.build_tree(subproblem, level + 1)
            child_node.parent = node
            node.children.append(child_node)

        return node


class ASCIITreeRenderer:
    """
    Renders problem trees as ASCII art for terminal display.
    """

    # Box drawing characters
    BOX_CHARS = {
        'vertical': '│',
        'branch': '├',
        'last_branch': '└',
        'horizontal': '─',
        'cross': '┼',
    }

    def render(self, tree: ProblemNode) -> str:
        """
        Render tree as ASCII art.

        Args:
            tree: ProblemNode to render

        Returns:
            ASCII art string
        """
        lines = []
        self._render_node(tree, lines, "", True)
        return '\n'.join(lines)

    def _render_node(
        self,
        node: ProblemNode,
        lines: List[str],
        prefix: str,
        is_last: bool
    ):
        """Render a node and its children"""
        # Add status indicator
        status_icon = self._get_status_icon(node.status)
        score_display = f" [{node.score:.0f}/100]" if node.score else ""

        # Add this node
        connector = self.BOX_CHARS['last_branch'] if is_last else self.BOX_CHARS['branch']
        lines.append(f"{prefix}{connector} {status_icon} {node.statement}{score_display}")

        # Add metadata
        indent = '  ' * (node.level + 1)

        # Team history
        if node.teams:
            teams_str = ' -> '.join(node.teams)
            lines.append(f"{indent}Teams: {teams_str}")

        # Timing
        if node.timing:
            lines.append(f"{indent}Time: {node.timing:.2f}s")

        # Attempts
        if node.attempt_count > 1:
            lines.append(f"{indent}Attempts: {node.attempt_count}")

        # Render children
        child_count = len(node.children)
        for i, child in enumerate(node.children):
            is_last_child = (i == child_count - 1)
            child_prefix = f"{prefix}{'    ' if is_last else prefix}{self.BOX_CHARS['vertical']}"
            self._render_node(child, lines, child_prefix, is_last_child)

    def _get_status_icon(self, status: str) -> str:
        """Get icon for problem status"""
        icons = {
            'pending': '⏳',
            'in_progress': '🔄',
            'complete': '[OK]',
            'failed': '[FAIL]',
            'approved': '🏆',
        }
        return icons.get(status, '❓')


class HTMLTreeRenderer:
    """
    Renders problem trees as interactive HTML.
    """

    def render(self, tree: ProblemNode) -> str:
        """
        Render tree as HTML.

        Args:
            tree: ProblemNode to render

        Returns:
            HTML string
        """
        lines = ['<!DOCTYPE html>', '<html>', '<head>']
        lines.append('  <meta charset="UTF-8">')
        lines.append('  <title>Problem Tree</title>')
        lines.append('  <style>')
        lines.append(CSS_STYLES)
        lines.append('  </style>')
        lines.append('</head>')
        lines.append('<body>')
        lines.append('  <div class="tree">')
        self._render_node(tree, lines, 0)
        lines.append('  </div>')
        lines.append('</body>')
        lines.append('</html>')

        return '\n'.join(lines)

    def _render_node(self, node: ProblemNode, lines: List[str], depth: int):
        """Render node in HTML"""
        status_class = f"status-{node.status}"

        lines.append(f'    <div class="node {status_class}" data-depth="{depth}">')

        # Statement and status
        lines.append(f'      <div class="statement">{self._escape_html(node.statement)}</div>')
        lines.append(f'      <div class="status">{node.status}</div>')

        # Score
        if node.score is not None:
            lines.append(f'      <div class="score">{node.score:.0f}/100</div>')

        # Teams
        if node.teams:
            lines.append('      <div class="teams">')
            for team in node.teams:
                lines.append(f'        <span class="team">{team}</span>')
            lines.append('      </div>')

        # Timing
        if node.timing:
            lines.append(f'      <div class="timing">{node.timing:.2f}s</div>')

        # Children
        if node.children:
            lines.append('      <div class="children">')
            for child in node.children:
                self._render_node(child, lines, depth + 1)
            lines.append('      </div>')

        lines.append('    </div>')

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))


class GraphvizTreeRenderer:
    """
    Renders problem trees in DOT format for Graphviz.
    """

    def render(self, tree: ProblemNode) -> str:
        """
        Render tree as DOT format.

        Args:
            tree: ProblemNode to render

        Returns:
            DOT format string
        """
        lines = ['digraph ProblemTree {', '  node [shape=box, style=rounded];', '']

        # Add nodes and edges
        node_id = 0
        self._add_node(tree, lines, node_id)
        node_id = self._add_edges(tree, lines, node_id)

        lines.append('}')
        return '\n'.join(lines)

    def _add_node(self, node: ProblemNode, lines: List[str], node_id: int) -> int:
        """Add node to DOT output"""
        # Create label
        label = node.statement
        if node.score is not None:
            label += f"\\n{node.score:.0f}/100"

        # Color by status
        colors = {
            'pending': 'lightgray',
            'in_progress': 'lightblue',
            'complete': 'lightgreen',
            'failed': 'lightcoral',
            'approved': 'gold',
        }

        color = colors.get(node.status, 'white')
        lines.append(
            f'  node_{node_id} [label="{label}", fillcolor="{color}"];'
        )

        # Add children
        child_id = node_id + 1
        for child in node.children:
            child_id = self._add_node(child, lines, child_id)
            child_id += 1

        return child_id

    def _add_edges(self, node: ProblemNode, lines: List[str], parent_id: int) -> int:
        """Add edges to DOT output"""
        current_id = parent_id + 1

        for child in node.children:
            lines.append(f'  node_{parent_id} -> node_{current_id};')
            current_id = self._add_edges(child, lines, current_id)
            current_id += 1

        return current_id


CSS_STYLES = """
    <style>
      .tree {
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 14px;
        line-height: 1.5;
      }

      .node {
        margin: 8px 0;
        padding: 8px;
        border-left: 2px solid #ccc;
        padding-left: 20px;
      }

      .statement {
        font-weight: bold;
      }

      .status {
        font-size: 12px;
        margin: 4px 0;
      }

      .score {
        font-size: 12px;
        margin: 4px 0;
      }

      .teams {
        margin: 4px 0;
      }

      .team {
        display: inline-block;
        padding: 2px 6px;
        margin: 2px;
        border-radius: 3px;
        font-size: 11px;
        background: #f0f0f0;
      }

      .timing {
        font-size: 11px;
        color: #666;
      }

      .status-pending { color: #666; }
      .status-in_progress { color: #007bff; }
      .status-complete { color: #28a745; }
      .status-failed { color: #dc3545; }
      .status-approved { color: #ffc107; }
    </style>
"""


def visualize_problem(
    problem: Dict[str, Any],
    format: str = 'ascii',
    output_file: Optional[str] = None
) -> str:
    """
    Visualize a problem hierarchy.

    Args:
        problem: Problem definition with subproblems
        format: Output format ('ascii', 'html', 'dot')
        output_file: Optional file path to save output

    Returns:
        Rendered visualization string
    """
    builder = ProblemTreeBuilder()
    tree = builder.build_tree(problem)

    if format == 'ascii':
        renderer = ASCIITreeRenderer()
        output = renderer.render(tree)
    elif format == 'html':
        renderer = HTMLTreeRenderer()
        output = renderer.render(tree)
    elif format == 'dot':
        renderer = GraphvizTreeRenderer()
        output = renderer.render(tree)
    else:
        raise ValueError(f"Unknown format: {format}")

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        logger.info(f"Visualization saved to {output_file}")

    return output


# Convenience functions for common use cases
def visualize_ascii(problem: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Visualize problem as ASCII art"""
    return visualize_problem(problem, 'ascii', output_file)


def visualize_html(problem: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Visualize problem as HTML"""
    return visualize_problem(problem, 'html', output_file)


def visualize_dot(problem: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Visualize problem as DOT (Graphviz) format"""
    return visualize_problem(problem, 'dot', output_file)
