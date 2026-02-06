"""
TODO/FIXME Tracker and Resolution System

This module provides tools for tracking, categorizing, and resolving TODOs
and FIXMEs throughout the OpenEvolve Frontend codebase.

Features:
- Scan codebase for TODO/FIXME comments
- Categorize by priority and component
- Generate resolution reports
- Track progress over time
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class TodoItem:
    """Represents a single TODO/FIXME item."""
    file_path: str
    line_number: int
    type: str  # TODO, FIXME, HACK, XXX, NOTE
    priority: str  # critical, high, medium, low
    component: str
    description: str
    context: str  # Surrounding code
    assigned_to: Optional[str] = None
    status: str = "open"  # open, in_progress, resolved, deferred
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class TodoStats:
    """Statistics for TODO items."""
    total: int
    by_type: Dict[str, int]
    by_priority: Dict[str, int]
    by_component: Dict[str, int]
    by_status: Dict[str, int]
    estimated_hours: Optional[Dict[str, float]] = None


class TodoScanner:
    """Scans codebase for TODO/FIXME comments."""

    # Patterns for TODO comments
    PATTERNS = {
        'TODO': r'(?i)TODO\(?([a-z]*)\)?:?\s*(.+)',
        'FIXME': r'(?i)FIXME\(?([a-z]*)\)?:?\s*(.+)',
        'HACK': r'(?i)HACK\(?([a-z]*)\)?:?\s*(.+)',
        'XXX': r'(?i)XXX\(?([a-z]*)\)?:?\s*(.+)',
        'NOTE': r'(?i)NOTE\(?([a-z]*)\)?:?\s*(.+)',
    }

    # Priority keywords
    PRIORITY_KEYWORDS = {
        'critical': ['critical', 'urgent', 'security', 'blocker', 'crash'],
        'high': ['important', 'high', 'priority', 'must'],
        'medium': ['medium', 'should', 'improve', 'enhance'],
        'low': ['low', 'minor', 'nice_to_have', 'maybe', 'someday'],
    }

    # File extensions to scan
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.tsx', '.jsx',
        '.java', '.cpp', '.c', '.h', '.cs',
        '.go', '.rs', '.rb', '.php',
        '.md', '.rst', '.txt'
    }

    # Component detection patterns
    COMPONENT_PATTERNS = {
        'verification': r'verification|z3|lean|prove',
        'decomposition': r'decomposition|roma|mdap',
        'icr': r'icr|refinement|iterative',
        'crewai': r'crewai|workflow|bridge',
        'ace': r'ace|agentic|learning',
        'claudiomiro': r'claudiomiro|mcp',
        'datapizza': r'datapizza|pipeline|chunk',
        'bubblelabs': r'bubblelab|grpc|knowledge',
        'c2c': r'c2c|cache|ensemble',
        'ui': r'ui|react|frontend',
        'testing': r'test|spec|mock',
    }

    def __init__(self, root_dir: str = "."):
        """
        Initialize TODO scanner.

        Args:
            root_dir: Root directory to scan
        """
        self.root_dir = Path(root_dir)
        self.todos: List[TodoItem] = []
        self.stats: Optional[TodoStats] = None

    def scan_all(
        self,
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None
    ) -> List[TodoItem]:
        """
        Scan all files for TODOs.

        Args:
            exclude_dirs: Directories to exclude (default: ['node_modules', '.git', '__pycache__'])
            exclude_files: File patterns to exclude

        Returns:
            List of TodoItem objects
        """
        if exclude_dirs is None:
            exclude_dirs = ['node_modules', '.git', '__pycache__', '.pytest_cache',
                          'venv', 'env', 'dist', 'build', '.next', 'out']

        exclude_dirs = set(exclude_dirs)
        self.todos = []

        for file_path in self.root_dir.rglob('*'):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue

            # Skip non-code files
            if file_path.suffix not in self.CODE_EXTENSIONS:
                continue

            # Scan file
            try:
                file_todos = self.scan_file(file_path)
                self.todos.extend(file_todos)
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")

        # Calculate statistics
        self.stats = self._calculate_stats()
        logger.info(f"Scanned {len(self.todos)} TODOs across codebase")

        return self.todos

    def scan_file(self, file_path: Path) -> List[TodoItem]:
        """Scan a single file for TODOs."""
        todos = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return todos

        for line_num, line in enumerate(lines, start=1):
            for todo_type, pattern in self.PATTERNS.items():
                match = re.search(pattern, line)
                if match:
                    priority_hint = match.group(1)
                    description = match.group(2).strip()

                    # Determine priority
                    priority = self._determine_priority(line, priority_hint)

                    # Determine component
                    component = self._determine_component(file_path, line)

                    # Extract context
                    context = self._extract_context(lines, line_num)

                    todo = TodoItem(
                        file_path=str(file_path.relative_to(self.root_dir)),
                        line_number=line_num,
                        type=todo_type,
                        priority=priority,
                        component=component,
                        description=description,
                        context=context,
                    )
                    todos.append(todo)

        return todos

    def _determine_priority(self, line: str, hint: str) -> str:
        """Determine priority from line content and hint."""
        line_lower = line.lower()

        # Check explicit hint
        if hint and hint in ['critical', 'high', 'medium', 'low']:
            return hint

        # Check keywords
        for priority, keywords in self.PRIORITY_KEYWORDS.items():
            if any(keyword in line_lower for keyword in keywords):
                return priority

        return 'medium'  # Default

    def _determine_component(self, file_path: Path, line: str) -> str:
        """Determine component from file path and content."""
        # Check file path first
        path_lower = str(file_path).lower()
        for component, pattern in self.COMPONENT_PATTERNS.items():
            if re.search(pattern, path_lower):
                return component

        # Check line content
        line_lower = line.lower()
        for component, pattern in self.COMPONENT_PATTERNS.items():
            if re.search(pattern, line_lower):
                return component

        return 'general'

    def _extract_context(self, lines: List[str], line_num: int) -> str:
        """Extract surrounding code context."""
        start = max(0, line_num - 2)
        end = min(len(lines), line_num + 2)
        context_lines = lines[start:end]
        return ''.join(context_lines).strip()

    def _calculate_stats(self) -> TodoStats:
        """Calculate statistics from scanned TODOs."""
        type_counts = Counter(t.type for t in self.todos)
        priority_counts = Counter(t.priority for t in self.todos)
        component_counts = Counter(t.component for t in self.todos)
        status_counts = Counter(t.status for t in self.todos)

        return TodoStats(
            total=len(self.todos),
            by_type=dict(type_counts),
            by_priority=dict(priority_counts),
            by_component=dict(component_counts),
            by_status=dict(status_counts),
        )

    def get_high_priority(self) -> List[TodoItem]:
        """Get high and critical priority TODOs."""
        return [t for t in self.todos if t.priority in ['critical', 'high']]

    def get_by_component(self, component: str) -> List[TodoItem]:
        """Get TODOs for a specific component."""
        return [t for t in self.todos if t.component == component]

    def get_by_type(self, todo_type: str) -> List[TodoItem]:
        """Get TODOs of a specific type."""
        return [t for t in self.todos if t.type == todo_type]

    def generate_report(self) -> str:
        """Generate human-readable report."""
        if not self.stats:
            return "No TODOs scanned yet"

        report = []
        report.append("=" * 80)
        report.append("TODO/FIXME REPORT")
        report.append("=" * 80)
        report.append(f"Total TODOs: {self.stats.total}")
        report.append("")

        # By priority
        report.append("BY PRIORITY:")
        for priority in ['critical', 'high', 'medium', 'low']:
            count = self.stats.by_priority.get(priority, 0)
            if count > 0:
                report.append(f"  {priority.capitalize()}: {count}")
        report.append("")

        # By type
        report.append("BY TYPE:")
        for todo_type, count in sorted(self.stats.by_type.items()):
            report.append(f"  {todo_type}: {count}")
        report.append("")

        # By component
        report.append("BY COMPONENT:")
        for component, count in sorted(self.stats.by_component.items(),
                                       key=lambda x: x[1], reverse=True):
            report.append(f"  {component}: {count}")
        report.append("")

        # High priority items
        high_priority = self.get_high_priority()
        if high_priority:
            report.append("HIGH PRIORITY ITEMS:")
            for todo in high_priority[:20]:  # Limit to 20
                report.append(f"  [{todo.priority.upper()}] {todo.file_path}:{todo.line_number}")
                report.append(f"    {todo.description[:80]}")
            if len(high_priority) > 20:
                report.append(f"  ... and {len(high_priority) - 20} more")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def export_json(self, output_path: str = "todos.json"):
        """Export todos to JSON file."""
        data = {
            'scan_date': datetime.now().isoformat(),
            'total': len(self.todos),
            'stats': {
                'by_type': self.stats.by_type if self.stats else {},
                'by_priority': self.stats.by_priority if self.stats else {},
                'by_component': self.stats.by_component if self.stats else {},
            },
            'todos': [
                {
                    'file': t.file_path,
                    'line': t.line_number,
                    'type': t.type,
                    'priority': t.priority,
                    'component': t.component,
                    'description': t.description,
                    'context': t.context,
                }
                for t in self.todos
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self.todos)} TODOs to {output_path}")


# Convenience functions
def scan_codebase(root_dir: str = ".") -> TodoScanner:
    """Scan codebase and return scanner with results."""
    scanner = TodoScanner(root_dir)
    scanner.scan_all()
    return scanner


def quick_report(root_dir: str = ".") -> str:
    """Generate quick TODO report."""
    scanner = scan_codebase(root_dir)
    return scanner.generate_report()


__all__ = [
    'TodoItem',
    'TodoStats',
    'TodoScanner',
    'scan_codebase',
    'quick_report',
]
