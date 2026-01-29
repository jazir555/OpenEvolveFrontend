"""
Integration Tests for Traceability System

Comprehensive integration tests for the change tracking,
audit trail, and traceability matrix system.
"""

import pytest
from datetime import datetime, timedelta
from bubblelabs_nodes.traceability import (
    Change,
    ChangeTrace,
    ChangeTracker,
    Modification,
    TraceabilityMatrix,
)


class TestChangeTrackerIntegration:
    """Integration tests for ChangeTracker"""

    def test_complete_workflow_tracking(self):
        """Test tracking changes through complete workflow"""
        tracker = ChangeTracker()

        # Blue Team creates initial solution
        change1 = tracker.track_change(
            problem_id='problem_123',
            team='blue',
            change_type='create',
            description='Created initial solution',
            before=None,
            after={'algorithm': 'v1', 'score': 0.75}
        )

        # Red Team finds issues
        change2 = tracker.track_change(
            problem_id='problem_123',
            team='red',
            change_type='modify',
            description='Fixed buffer overflow bug',
            before={'code': 'unsafe'},
            after={'code': 'safe'}
        )

        # Gold Team approves
        change3 = tracker.track_change(
            problem_id='problem_123',
            team='gold',
            change_type='approve',
            description='Solution approved for production',
            before={'validated': False},
            after={'validated': True}
        )

        # Get complete trace
        trace = tracker.get_trace('problem_123')

        assert trace is not None
        assert len(trace.changes) == 3
        assert trace.changes[0].team == 'blue'
        assert trace.changes[1].team == 'red'
        assert trace.changes[2].team == 'gold'

    def test_team_contributions(self):
        """Test tracking contributions by team"""
        tracker = ChangeTracker()

        # Multiple changes by different teams
        tracker.track_change('p1', 'blue', 'create', 'Created solution', {}, {})
        tracker.track_change('p1', 'blue', 'modify', 'Optimized algorithm', {}, {})
        tracker.track_change('p1', 'red', 'modify', 'Found edge case', {}, {})
        tracker.track_change('p1', 'red', 'modify', 'Fixed edge case', {}, {})
        tracker.track_change('p1', 'gold', 'approve', 'Approved solution', {}, {})

        # Get changes by team
        blue_changes = tracker.get_changes_by_team('p1', 'blue')
        red_changes = tracker.get_changes_by_team('p1', 'red')
        gold_changes = tracker.get_changes_by_team('p1', 'gold')

        assert len(blue_changes) == 2
        assert len(red_changes) == 2
        assert len(gold_changes) == 1

    def test_modification_tracking(self):
        """Test tracking specific modifications"""
        tracker = ChangeTracker()

        # Track a change
        change = tracker.track_change(
            problem_id='p1',
            team='blue',
            change_type='modify',
            description='Updated algorithm',
            before={'version': 1},
            after={'version': 2}
        )

        # Track specific modifications
        mod1 = tracker.track_modification(
            problem_id='p1',
            change_id=change.change_id,
            section='algorithm',
            operation='replace',
            before_value='v1',
            after_value='v2',
            reason='Optimization'
        )

        mod2 = tracker.track_modification(
            problem_id='p1',
            change_id=change.change_id,
            section='config',
            operation='add',
            after_value={'timeout': 30},
            reason='Added configuration'
        )

        trace = tracker.get_trace('p1')
        assert len(trace.modifications) == 2

    def test_temporal_queries(self):
        """Test querying changes by time range"""
        tracker = ChangeTracker()

        # Track changes at different times
        base_time = datetime.utcnow()

        # Create changes with specific timestamps (simulated)
        tracker.track_change('p1', 'blue', 'create', 'Change 1', {}, {})

        # Simulate time passing
        import time
        time.sleep(0.1)

        tracker.track_change('p1', 'red', 'modify', 'Change 2', {}, {})

        # Get recent changes
        recent_changes = tracker.get_recent_changes('p1', hours=1)

        assert len(recent_changes) >= 2

    def test_change_aggregation(self):
        """Test aggregating changes across problems"""
        tracker = ChangeTracker()

        # Track changes across multiple problems
        tracker.track_change('p1', 'blue', 'create', 'Solution 1', {}, {})
        tracker.track_change('p2', 'blue', 'create', 'Solution 2', {}, {})
        tracker.track_change('p3', 'red', 'modify', 'Fix 1', {}, {})

        # Get all changes
        all_traces = tracker.get_all_traces()

        assert len(all_traces) == 3

    def test_metadata_tracking(self):
        """Test tracking metadata with changes"""
        tracker = ChangeTracker()

        change = tracker.track_change(
            problem_id='p1',
            team='blue',
            change_type='modify',
            description='Performance improvement',
            before={'time': 100},
            after={'time': 50},
            metadata={
                'improvement_factor': 2,
                'optimization_technique': 'caching',
                'tested': True
            }
        )

        assert change.metadata['improvement_factor'] == 2
        assert change.metadata['optimization_technique'] == 'caching'
        assert change.metadata['tested'] is True


class TestTraceabilityMatrix:
    """Integration tests for TraceabilityMatrix"""

    def test_matrix_building(self):
        """Test building complete traceability matrix"""
        matrix = TraceabilityMatrix()

        # Add changes from multiple teams
        change1 = Change(
            change_id='c1',
            timestamp=datetime.utcnow(),
            team='blue',
            author='alice',
            change_type='create',
            description='Created solution',
            before=None,
            after={'solution': 'code'}
        )

        change2 = Change(
            change_id='c2',
            timestamp=datetime.utcnow() + timedelta(seconds=1),
            team='red',
            author='bob',
            change_type='modify',
            description='Fixed bug',
            before={'bug': 'present'},
            after={'bug': 'fixed'}
        )

        matrix.add_change(change1)
        matrix.add_change(change2)

        # Get all changes
        all_changes = matrix.get_all_changes()
        assert len(all_changes) == 2

    def test_trace_queries(self):
        """Test querying traces from matrix"""
        matrix = TraceabilityMatrix()

        # Add changes for multiple problems
        for i in range(3):
            change = Change(
                change_id=f'c{i}',
                timestamp=datetime.utcnow(),
                team='blue',
                change_type='create',
                description=f'Solution {i}',
                before=None,
                after={'id': i}
            )
            matrix.add_change(change, problem_id=f'problem_{i}')

        # Query specific problem
        trace = matrix.get_trace('problem_1')
        assert trace is not None
        assert len(trace.changes) == 1

    def test_audit_trail_generation(self):
        """Test generating complete audit trail"""
        matrix = TraceabilityMatrix()

        # Create audit trail
        changes = []
        teams = ['blue', 'red', 'gold']
        types = ['create', 'modify', 'approve']

        for i, (team, change_type) in enumerate(zip(teams, types)):
            change = Change(
                change_id=f'c{i}',
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                team=team,
                change_type=change_type,
                description=f'Step {i+1}',
                before={'step': i},
                after={'step': i+1}
            )
            changes.append(change)
            matrix.add_change(change, problem_id='audit_test')

        # Generate audit trail
        trail = matrix.get_audit_trail('audit_test')

        assert len(trail) == 3
        assert trail[0]['team'] == 'blue'
        assert trail[1]['team'] == 'red'
        assert trail[2]['team'] == 'gold'

    def test_filtering_by_criteria(self):
        """Test filtering changes by various criteria"""
        matrix = TraceabilityMatrix()

        # Add changes with different attributes
        changes = [
            Change('c1', datetime.utcnow(), 'blue', 'alice', 'create', 'Desc 1', None, {}),
            Change('c2', datetime.utcnow(), 'red', 'bob', 'modify', 'Desc 2', {}, {}),
            Change('c3', datetime.utcnow(), 'blue', 'alice', 'modify', 'Desc 3', {}, {}),
            Change('c4', datetime.utcnow(), 'gold', 'charlie', 'approve', 'Desc 4', {}, {}),
        ]

        for change in changes:
            matrix.add_change(change, problem_id='filter_test')

        # Filter by team
        blue_changes = matrix.filter_by_team('filter_test', 'blue')
        assert len(blue_changes) == 2

        # Filter by type
        modify_changes = matrix.filter_by_type('filter_test', 'modify')
        assert len(modify_changes) == 2

        # Filter by author
        alice_changes = matrix.filter_by_author('filter_test', 'alice')
        assert len(alice_changes) == 2

    def test_diff_generation(self):
        """Test generating diffs for changes"""
        matrix = TraceabilityMatrix()

        change = Change(
            change_id='c1',
            timestamp=datetime.utcnow(),
            team='blue',
            change_type='modify',
            description='Updated configuration',
            before={'timeout': 10, 'retries': 3},
            after={'timeout': 30, 'retries': 5, 'cache': True}
        )

        matrix.add_change(change, problem_id='diff_test')

        # Get diff
        trace = matrix.get_trace('diff_test')
        diff = trace.changes[0].diff

        assert diff is not None
        assert 'timeout' in diff or 'cache' in diff

    def test_export_import(self):
        """Test exporting and importing traceability data"""
        matrix = TraceabilityMatrix()

        # Add sample data
        change = Change(
            change_id='c1',
            timestamp=datetime.utcnow(),
            team='blue',
            change_type='create',
            description='Test',
            before=None,
            after={'data': 'value'}
        )
        matrix.add_change(change, problem_id='export_test')

        # Export
        exported = matrix.export_to_dict()

        # Verify export structure
        assert 'traces' in exported
        assert 'exported_at' in exported

        # Import into new matrix
        new_matrix = TraceabilityMatrix()
        new_matrix.import_from_dict(exported)

        # Verify import
        imported_trace = new_matrix.get_trace('export_test')
        assert imported_trace is not None
        assert len(imported_trace.changes) == 1

    def test_statistics_generation(self):
        """Test generating statistics from traceability data"""
        matrix = TraceabilityMatrix()

        # Add changes
        for i in range(10):
            change = Change(
                change_id=f'c{i}',
                timestamp=datetime.utcnow(),
                team=['blue', 'red', 'gold'][i % 3],
                change_type=['create', 'modify', 'approve'][i % 3],
                description=f'Change {i}',
                before=None,
                after={}
            )
            matrix.add_change(change, problem_id=f'problem_{i % 3}')

        # Generate statistics
        stats = matrix.generate_statistics()

        assert stats['total_changes'] == 10
        assert stats['total_problems'] == 3
        assert 'by_team' in stats
        assert 'by_type' in stats

    def test_compliance_reporting(self):
        """Test generating compliance reports"""
        matrix = TraceabilityMatrix()

        # Add changes with approval chain
        changes = [
            Change('c1', datetime.utcnow(), 'blue', 'alice', 'create', 'Created', None, {}),
            Change('c2', datetime.utcnow(), 'red', 'bob', 'modify', 'Tested', {}, {}),
            Change('c3', datetime.utcnow(), 'gold', 'charlie', 'approve', 'Approved', {}, {}),
        ]

        for change in changes:
            matrix.add_change(change, problem_id='compliance_test')

        # Generate compliance report
        report = matrix.generate_compliance_report('compliance_test')

        assert report['has_blue_team_approval'] is True
        assert report['has_red_team_testing'] is True
        assert report['has_gold_team_approval'] is True
        assert report['is_compliant'] is True

    def test_cross_problem_tracking(self):
        """Test tracking changes across related problems"""
        matrix = TraceabilityMatrix()

        # Create parent-child problem relationship
        parent_change = Change(
            'c_parent',
            datetime.utcnow(),
            'blue',
            'alice',
            'create',
            'Parent problem',
            None,
            {'subproblems': ['child1', 'child2']}
        )

        child1_change = Change(
            'c_child1',
            datetime.utcnow(),
            'blue',
            'bob',
            'create',
            'Child 1',
            None,
            {}
        )

        child2_change = Change(
            'c_child2',
            datetime.utcnow(),
            'blue',
            'charlie',
            'create',
            'Child 2',
            None,
            {}
        )

        matrix.add_change(parent_change, problem_id='parent')
        matrix.add_change(child1_change, problem_id='child1')
        matrix.add_change(child2_change, problem_id='child2')

        # Link related problems
        matrix.link_problems('parent', ['child1', 'child2'])

        # Get related changes
        related = matrix.get_related_changes('parent')
        assert len(related) >= 1


class TestTraceabilityVisualization:
    """Integration tests for traceability visualization"""

    def test_timeline_generation(self):
        """Test generating timeline visualization"""
        tracker = ChangeTracker()

        # Add changes over time
        for i in range(5):
            tracker.track_change(
                problem_id='timeline_test',
                team=['blue', 'red', 'gold'][i % 3],
                change_type='modify',
                description=f'Change {i}',
                before={'step': i},
                after={'step': i+1}
            )

        # Get timeline
        timeline = tracker.get_timeline('timeline_test')

        assert len(timeline) == 5
        # Verify chronological order
        for i in range(1, len(timeline)):
            assert timeline[i].timestamp >= timeline[i-1].timestamp

    def test_team_contribution_visualization(self):
        """Test visualizing team contributions"""
        tracker = ChangeTracker()

        # Add changes from different teams
        team_counts = {'blue': 5, 'red': 3, 'gold': 2}
        for team, count in team_counts.items():
            for _ in range(count):
                tracker.track_change(
                    problem_id='contribution_test',
                    team=team,
                    change_type='modify',
                    description=f'{team} change',
                    before={},
                    after={}
                )

        # Get contribution stats
        stats = tracker.get_contribution_stats('contribution_test')

        assert stats['blue'] == 5
        assert stats['red'] == 3
        assert stats['gold'] == 2

    def test_change_flow_visualization(self):
        """Test visualizing change flow through teams"""
        tracker = ChangeTracker()

        # Simulate workflow: Blue -> Red -> Gold
        tracker.track_change('flow_test', 'blue', 'create', 'Created', None, {})
        tracker.track_change('flow_test', 'red', 'modify', 'Tested', {}, {})
        tracker.track_change('flow_test', 'gold', 'approve', 'Approved', {}, {})

        # Get flow
        flow = tracker.get_change_flow('flow_test')

        assert flow[0] == 'blue'
        assert flow[1] == 'red'
        assert flow[2] == 'gold'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
