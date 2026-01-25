"""
Unit Tests for Traceability Components

Tests for the change tracking and traceability matrix systems.
"""

import pytest
from bubblelabs_nodes import (
    ChangeTracker,
    Change,
    TraceabilityMatrix,
    get_change_tracker,
)
from datetime import datetime


class TestChangeTracker:
    """Tests for ChangeTracker"""

    def test_track_change(self):
        """Test basic change tracking"""
        tracker = ChangeTracker()

        change = tracker.track_change(
            problem_id='problem_123',
            team='blue_team',
            change_type='solution_update',
            description='Improved algorithm',
            before={'version': 1},
            after={'version': 2}
        )

        assert change.problem_id == 'problem_123'
        assert change.team == 'blue_team'
        assert change.change_type == 'solution_update'

    def test_get_changes_for_problem(self):
        """Test retrieving changes for a problem"""
        tracker = ChangeTracker()

        tracker.track_change('p1', 'team1', 'update', 'Change 1', {}, {})
        tracker.track_change('p1', 'team1', 'update', 'Change 2', {}, {})
        tracker.track_change('p2', 'team1', 'update', 'Change 3', {}, {})

        changes = tracker.get_changes_for_problem('p1')

        assert len(changes) == 2
        assert all(c.problem_id == 'p1' for c in changes)

    def test_get_changes_by_team(self):
        """Test retrieving changes by team"""
        tracker = ChangeTracker()

        tracker.track_change('p1', 'blue_team', 'update', 'C1', {}, {})
        tracker.track_change('p2', 'red_team', 'update', 'C2', {}, {})

        blue_changes = tracker.get_changes_by_team('blue_team')

        assert len(blue_changes) == 1
        assert blue_changes[0].team == 'blue_team'

    def test_get_changes_by_type(self):
        """Test retrieving changes by type"""
        tracker = ChangeTracker()

        tracker.track_change('p1', 'team1', 'solution_update', 'C1', {}, {})
        tracker.track_change('p1', 'team1', 'validation', 'C2', {}, {})

        updates = tracker.get_changes_by_type('solution_update')

        assert len(updates) == 1
        assert updates[0].change_type == 'solution_update'

    def test_get_timeline(self):
        """Test timeline retrieval"""
        tracker = ChangeTracker()

        tracker.track_change('p1', 'team1', 'update', 'C1', {}, {})
        tracker.track_change('p1', 'team1', 'update', 'C2', {}, {})

        timeline = tracker.get_timeline('p1')

        assert len(timeline) == 2
        assert timeline[0].timestamp <= timeline[1].timestamp


class TestTraceabilityMatrix:
    """Tests for TraceabilityMatrix"""

    def test_add_change(self):
        """Test adding change to matrix"""
        matrix = TraceabilityMatrix()

        change = Change(
            change_id='c1',
            problem_id='p1',
            team='team1',
            change_type='update',
            description='Change 1',
            before={},
            after={},
            timestamp=datetime.utcnow()
        )

        matrix.add_change(change)

        assert len(matrix.get_all_changes()) == 1

    def test_get_audit_trail(self):
        """Test audit trail generation"""
        matrix = TraceabilityMatrix()

        change1 = Change(
            change_id='c1',
            problem_id='p1',
            team='team1',
            change_type='update',
            description='Change 1',
            before={},
            after={},
            timestamp=datetime.utcnow()
        )

        matrix.add_change(change1)

        trail = matrix.get_audit_trail('p1')

        assert len(trail) == 1
        assert trail[0]['change_id'] == 'c1'

    def test_filter_by_criteria(self):
        """Test filtering changes"""
        matrix = TraceabilityMatrix()

        change1 = Change(
            change_id='c1',
            problem_id='p1',
            team='blue_team',
            change_type='update',
            description='Change 1',
            before={},
            after={},
            timestamp=datetime.utcnow()
        )

        change2 = Change(
            change_id='c2',
            problem_id='p1',
            team='red_team',
            change_type='validation',
            description='Change 2',
            before={},
            after={},
            timestamp=datetime.utcnow()
        )

        matrix.add_change(change1)
        matrix.add_change(change2)

        # Filter by team
        blue_changes = matrix.filter_by_team('blue_team')
        assert len(blue_changes) == 1

        # Filter by type
        updates = matrix.filter_by_type('update')
        assert len(updates) == 1


class TestChangeValidation:
    """Tests for change validation"""

    def test_validate_before_after(self):
        """Test before/after validation"""
        tracker = ChangeTracker()

        # Valid change
        change = tracker.track_change(
            problem_id='p1',
            team='team1',
            change_type='update',
            description='Test',
            before={'version': 1},
            after={'version': 2}
        )

        # Should have diff
        assert change.before is not None
        assert change.after is not None

    def test_diff_generation(self):
        """Test diff generation"""
        tracker = ChangeTracker()

        change = tracker.track_change(
            problem_id='p1',
            team='team1',
            change_type='update',
            description='Test',
            before={'field': 'old_value'},
            after={'field': 'new_value'}
        )

        # Change should have diff
        assert change.diff is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
