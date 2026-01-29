"""
Traceability Matrix for OpenEvolve Gauntlet System

Tracks all changes made to solutions throughout the Blue→Red→Gold
workflow, providing a complete audit trail and debugging capabilities.

Key Features:
- Change tracking for all team modifications
- Diff generation for solution changes
- Timeline visualization of changes
- Team contribution tracking
- Full history queries
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class Change:
    """Represents a single change to a solution"""
    change_id: str
    timestamp: datetime
    team: str  # 'blue', 'red', 'gold'
    author: Optional[str]
    change_type: str  # 'create', 'modify', 'approve', 'reject'
    description: str
    before: Optional[Any] = None
    after: Optional[Any] = None
    diff: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.change_id is None:
            self.change_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique change ID"""
        content = f"{self.team}:{self.timestamp}:{self.description}"
        hash_obj = hashlib.sha256(content.encode())
        return f"change_{hash_obj.hexdigest()[:16]}"


@dataclass
class Modification:
    """Represents a specific modification within a change"""
    modification_id: str
    change_id: str
    section: str  # What part of the solution was modified
    operation: str  # 'add', 'remove', 'replace', 'move'
    before_value: Any = None
    after_value: Any = None
    line_number: Optional[int] = None
    reason: Optional[str] = None


@dataclass
class ChangeTrace:
    """Complete trace of changes for a problem"""
    problem_id: str
    changes: List[Change] = field(default_factory=list)
    modifications: List[Modification] = field(default_factory=list)
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()


class ChangeTracker:
    """
    Tracks all changes made to solutions by different teams.
    """

    def __init__(self):
        self.traces: Dict[str, ChangeTrace] = {}

    def track_change(
        self,
        problem_id: str,
        team: str,
        change_type: str,
        description: str,
        before: Any = None,
        after: Any = None,
        author: str = None,
        metadata: Dict[str, Any] = None
    ) -> Change:
        """
        Track a change to a solution.

        Args:
            problem_id: Problem identifier
            team: Team making the change ('blue', 'red', 'gold')
            change_type: Type of change ('create', 'modify', 'approve', 'reject')
            description: Human-readable description
            before: State before change
            after: State after change
            author: Optional author identifier
            metadata: Additional metadata

        Returns:
            Change object
        """
        # Generate diff if both before and after are provided
        diff = None
        if before is not None and after is not None:
            diff = self._generate_diff(before, after)

        change = Change(
            change_id=None,  # Will be generated in __post_init__
            timestamp=datetime.utcnow(),
            team=team,
            author=author,
            change_type=change_type,
            description=description,
            before=before,
            after=after,
            diff=diff,
            metadata=metadata or {}
        )

        # Get or create trace
        if problem_id not in self.traces:
            self.traces[problem_id] = ChangeTrace(problem_id=problem_id)

        trace = self.traces[problem_id]
        trace.changes.append(change)
        trace.updated_at = datetime.utcnow()

        logger.info(
            f"Tracked change: {team} team {change_type} - {description} "
            f"(problem: {problem_id})"
        )

        return change

    def track_modification(
        self,
        problem_id: str,
        change_id: str,
        section: str,
        operation: str,
        before_value: Any = None,
        after_value: Any = None,
        line_number: int = None,
        reason: str = None
    ) -> Modification:
        """
        Track a specific modification within a change.

        Args:
            problem_id: Problem identifier
            change_id: Parent change ID
            section: Section being modified
            operation: Type of operation ('add', 'remove', 'replace', 'move')
            before_value: Value before modification
            after_value: Value after modification
            line_number: Optional line number
            reason: Reason for modification

        Returns:
            Modification object
        """
        mod = Modification(
            modification_id=f"mod_{hashlib.sha256(change_id.encode()).hexdigest()[:16]}",
            change_id=change_id,
            section=section,
            operation=operation,
            before_value=before_value,
            after_value=after_value,
            line_number=line_number,
            reason=reason
        )

        if problem_id in self.traces:
            self.traces[problem_id].modifications.append(mod)

        return mod

    def get_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Get complete trace for a problem"""
        return self.traces.get(problem_id)

    def get_changes_by_team(
        self,
        problem_id: str,
        team: str
    ) -> List[Change]:
        """Get all changes by a specific team"""
        trace = self.traces.get(problem_id)
        if not trace:
            return []

        return [c for c in trace.changes if c.team == team]

    def get_changes_by_time_range(
        self,
        problem_id: str,
        start: datetime,
        end: datetime
    ) -> List[Change]:
        """Get changes within a time range"""
        trace = self.traces.get(problem_id)
        if not trace:
            return []

        return [
            c for c in trace.changes
            if start <= c.timestamp <= end
        ]

    def get_full_history(self, problem_id: str) -> List[Change]:
        """Get full change history for a problem"""
        trace = self.traces.get(problem_id)
        if not trace:
            return []

        # Return sorted by timestamp
        return sorted(trace.changes, key=lambda c: c.timestamp)

    def _generate_diff(self, before: Any, after: Any) -> str:
        """Generate a diff between two states"""
        # Handle different types
        if isinstance(before, str) and isinstance(after, str):
            return self._string_diff(before, after)
        elif isinstance(before, dict) and isinstance(after, dict):
            return self._dict_diff(before, after)
        elif isinstance(before, list) and isinstance(after, list):
            return self._list_diff(before, after)
        else:
            # Simple representation
            return f"Before: {repr(before)[:100]}\nAfter: {repr(after)[:100]}"

    def _string_diff(self, before: str, after: str) -> str:
        """Generate diff for strings"""
        lines = []
        lines.append(f"- {before}")
        lines.append(f"+ {after}")
        return "\n".join(lines)

    def _dict_diff(self, before: dict, after: dict) -> str:
        """Generate diff for dictionaries"""
        lines = []

        all_keys = set(before.keys()) | set(after.keys())

        for key in sorted(all_keys):
            before_val = before.get(key, "<missing>")
            after_val = after.get(key, "<missing>")

            if before_val != after_val:
                lines.append(f"  {key}:")
                lines.append(f"  - {repr(before_val)[:50]}")
                lines.append(f"  + {repr(after_val)[:50]}")

        return "\n".join(lines)

    def _list_diff(self, before: list, after: list) -> str:
        """Generate diff for lists"""
        lines = []

        max_len = max(len(before), len(after))

        for i in range(max_len):
            before_val = before[i] if i < len(before) else "<missing>"
            after_val = after[i] if i < len(after) else "<missing>"

            if before_val != after_val:
                lines.append(f"  [{i}]:")
                lines.append(f"  - {repr(before_val)[:50]}")
                lines.append(f"  + {repr(after_val)[:50]}")

        return "\n".join(lines)


class TraceStorage:
    """
    Storage backend for traceability data.

    Can be extended to support database storage.
    """

    def __init__(self, storage_type: str = 'memory'):
        self.storage_type = storage_type
        self.traces: Dict[str, Dict[str, Any]] = {}

    async def save_trace(self, trace: ChangeTrace) -> bool:
        """Save a trace to storage"""
        try:
            trace_dict = {
                'problem_id': trace.problem_id,
                'changes': [
                    {
                        'change_id': c.change_id,
                        'timestamp': c.timestamp.isoformat(),
                        'team': c.team,
                        'author': c.author,
                        'change_type': c.change_type,
                        'description': c.description,
                        'before': c.before,
                        'after': c.after,
                        'diff': c.diff,
                        'metadata': c.metadata,
                    }
                    for c in trace.changes
                ],
                'modifications': [
                    {
                        'modification_id': m.modification_id,
                        'change_id': m.change_id,
                        'section': m.section,
                        'operation': m.operation,
                        'before_value': m.before_value,
                        'after_value': m.after_value,
                        'line_number': m.line_number,
                        'reason': m.reason,
                    }
                    for m in trace.modifications
                ],
                'created_at': trace.created_at.isoformat(),
                'updated_at': trace.updated_at.isoformat(),
            }

            self.traces[trace.problem_id] = trace_dict
            return True

        except Exception as e:
            logger.error(f"Failed to save trace: {e}")
            return False

    async def load_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Load a trace from storage"""
        trace_dict = self.traces.get(problem_id)
        if not trace_dict:
            return None

        changes = [
            Change(
                change_id=c['change_id'],
                timestamp=datetime.fromisoformat(c['timestamp']),
                team=c['team'],
                author=c['author'],
                change_type=c['change_type'],
                description=c['description'],
                before=c['before'],
                after=c['after'],
                diff=c['diff'],
                metadata=c['metadata'],
            )
            for c in trace_dict['changes']
        ]

        modifications = [
            Modification(
                modification_id=m['modification_id'],
                change_id=m['change_id'],
                section=m['section'],
                operation=m['operation'],
                before_value=m['before_value'],
                after_value=m['after_value'],
                line_number=m['line_number'],
                reason=m['reason'],
            )
            for m in trace_dict['modifications']
        ]

        return ChangeTrace(
            problem_id=trace_dict['problem_id'],
            changes=changes,
            modifications=modifications,
            created_at=datetime.fromisoformat(trace_dict['created_at']),
            updated_at=datetime.fromisoformat(trace_dict['updated_at']),
        )

    async def list_problems(self) -> List[str]:
        """List all problems with traces"""
        return list(self.traces.keys())


class TraceVisualizer:
    """
    Generates visual representations of change traces.
    """

    def generate_timeline(self, trace: ChangeTrace) -> str:
        """Generate timeline visualization"""
        lines = []
        lines.append(f"Timeline for Problem: {trace.problem_id}")
        lines.append(f"Created: {trace.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Changes: {len(trace.changes)}")
        lines.append("")

        for i, change in enumerate(trace.changes, 1):
            team_icon = {
                'blue': '🔵',
                'red': '🔴',
                'gold': '🟡',
            }.get(change.team, '⚪')

            lines.append(
                f"{i}. {team_icon} {change.team.upper()} - {change.change_type}: "
                f"{change.description}"
            )
            lines.append(f"   Time: {change.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if change.author:
                lines.append(f"   Author: {change.author}")

            if change.diff:
                lines.append(f"   Diff:")
                for diff_line in change.diff.split('\n')[:3]:  # First 3 lines
                    lines.append(f"     {diff_line}")

            lines.append("")

        return "\n".join(lines)

    def generate_team_contributions(self, trace: ChangeTrace) -> Dict[str, Dict[str, Any]]:
        """Generate team contribution statistics"""
        stats = {
            'blue': {'count': 0, 'types': {}},
            'red': {'count': 0, 'types': {}},
            'gold': {'count': 0, 'types': {}},
        }

        for change in trace.changes:
            if change.team in stats:
                stats[change.team]['count'] += 1

                change_type = change.change_type
                stats[change.team]['types'][change_type] = \
                    stats[change.team]['types'].get(change_type, 0) + 1

        return stats

    def generate_diff_view(
        self,
        before: Any,
        after: Any,
        context_lines: int = 3
    ) -> str:
        """Generate detailed diff view"""
        tracker = ChangeTracker()
        diff = tracker._generate_diff(before, after)

        lines = [
            "=== DIFF VIEW ===",
            "",
            "BEFORE:",
            str(before)[:200],
            "",
            "AFTER:",
            str(after)[:200],
            "",
            "DIFF:",
            diff,
        ]

        return "\n".join(lines)


class TraceabilityMatrix:
    """
    Main interface for traceability functionality.

    Integrates change tracking, storage, and visualization.
    """

    def __init__(self, storage_type: str = 'memory'):
        self.tracker = ChangeTracker()
        self.storage = TraceStorage(storage_type=storage_type)
        self.visualizer = TraceVisualizer()

    async def record_change(
        self,
        problem_id: str,
        team: str,
        change_type: str,
        description: str,
        before: Any = None,
        after: Any = None,
        author: str = None
    ) -> Change:
        """Record a change and persist it"""
        change = self.tracker.track_change(
            problem_id=problem_id,
            team=team,
            change_type=change_type,
            description=description,
            before=before,
            after=after,
            author=author
        )

        # Persist to storage
        trace = self.tracker.get_trace(problem_id)
        if trace:
            await self.storage.save_trace(trace)

        return change

    async def get_full_trace(self, problem_id: str) -> Optional[ChangeTrace]:
        """Get full trace including from storage"""
        # Check storage first
        trace = await self.storage.load_trace(problem_id)
        if trace:
            return trace

        # Fall back to in-memory tracker
        return self.tracker.get_trace(problem_id)

    async def generate_timeline(self, problem_id: str) -> Optional[str]:
        """Generate timeline visualization for a problem"""
        trace = await self.get_full_trace(problem_id)
        if not trace:
            return None

        return self.visualizer.generate_timeline(trace)

    async def get_team_stats(self, problem_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get team contribution statistics"""
        trace = await self.get_full_trace(problem_id)
        if not trace:
            return None

        return self.visualizer.generate_team_contributions(trace)


# Convenience functions
async def track_solution_change(
    problem_id: str,
    team: str,
    description: str,
    before: Any = None,
    after: Any = None
) -> Change:
    """Convenience function to track a solution change"""
    matrix = TraceabilityMatrix()
    return await matrix.record_change(
        problem_id=problem_id,
        team=team,
        change_type='modify',
        description=description,
        before=before,
        after=after
    )


async def get_problem_timeline(problem_id: str) -> Optional[str]:
    """Convenience function to get problem timeline"""
    matrix = TraceabilityMatrix()
    return await matrix.generate_timeline(problem_id)


# Example usage
async def demo_traceability():
    """Demonstration of traceability system"""

    matrix = TraceabilityMatrix()

    # Simulate Blue Team creating a solution
    await matrix.record_change(
        problem_id='problem_123',
        team='blue',
        change_type='create',
        description='Initial solution created',
        after={'code': 'def solve(): return "solution"'}
    )

    # Simulate Blue Team modifying
    await matrix.record_change(
        problem_id='problem_123',
        team='blue',
        change_type='modify',
        description='Added error handling',
        before={'code': 'def solve(): return "solution"'},
        after={'code': 'def solve():\n  try:\n    return "solution"\n  except Exception as e:\n    return None'},
        author='developer_1'
    )

    # Simulate Red Team attacking
    await matrix.record_change(
        problem_id='problem_123',
        team='red',
        change_type='modify',
        description='Found buffer overflow vulnerability',
        before={'code': 'def solve():\n  return "solution"'},
        after={'code': 'def solve():\n  # VULNERABLE: Buffer overflow possible\n  return "solution"'},
        author='security_researcher'
    )

    # Simulate Gold Team approval
    await matrix.record_change(
        problem_id='problem_123',
        team='gold',
        change_type='approve',
        description='Solution approved after fixes',
        author='reviewer_1'
    )

    # Get timeline
    timeline = await matrix.generate_timeline('problem_123')
    print(timeline)

    # Get team stats
    stats = await matrix.get_team_stats('problem_123')
    print(f"\nTeam Stats:")
    for team, team_stats in stats.items():
        print(f"  {team.title()}: {team_stats['count']} changes")
        print(f"    Types: {team_stats['types']}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(demo_traceability())
