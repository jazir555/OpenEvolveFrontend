"""
Unit and Integration Tests for Problem Visualization System

Comprehensive test coverage for visualization functionality including
tree building, rendering (ASCII, HTML, DOT), and API integration.
"""

import pytest
from bubblelabs_nodes.problem_visualization import (
    ProblemTreeBuilder,
    ASCIITreeRenderer,
    HTMLTreeRenderer,
    GraphvizTreeRenderer,
    VisualizationAPI,
    OutputFormat,
    ProblemStatus,
    TreeNode,
    visualize_problem
)


class TestProblemTreeBuilder:
    """Tests for ProblemTreeBuilder"""

    def test_build_simple_tree(self):
        """Test building a simple tree without subproblems"""
        builder = ProblemTreeBuilder()
        problem = {'id': 'root', 'status': 'complete', 'score': 85}

        root = builder.build_tree(problem)

        assert root.problem_id == 'root'
        assert root.status == ProblemStatus.COMPLETE
        assert root.score == 85
        assert len(root.children) == 0
        assert root.is_leaf()

    def test_build_hierarchy(self):
        """Test building a multi-level hierarchy"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'status': 'complete',
            'subproblems': [
                {
                    'id': 'child1',
                    'status': 'complete',
                    'subproblems': [
                        {'id': 'grandchild1', 'status': 'complete'}
                    ]
                },
                {'id': 'child2', 'status': 'failed'}
            ]
        }

        root = builder.build_tree(problem)

        assert root.problem_id == 'root'
        assert len(root.children) == 2
        assert root.children[0].problem_id == 'child1'
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].problem_id == 'grandchild1'
        assert root.children[1].problem_id == 'child2'

    def test_depth_calculation(self):
        """Test depth calculation for nodes"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'subproblems': [
                {
                    'id': 'child1',
                    'subproblems': [
                        {'id': 'grandchild1'}
                    ]
                }
            ]
        }

        root = builder.build_tree(problem)

        assert root.depth() == 0
        assert root.children[0].depth() == 1
        assert root.children[0].children[0].depth() == 2

    def test_subtree_size(self):
        """Test subtree size calculation"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'subproblems': [
                {'id': 'child1'},
                {'id': 'child2'}
            ]
        }

        root = builder.build_tree(problem)

        assert root.subtree_size() == 3  # root + 2 children
        assert root.children[0].subtree_size() == 1  # just child1

    def test_circular_reference_detection(self):
        """Test detection of circular references"""
        builder = ProblemTreeBuilder()

        # Create a node with a circular parent reference
        problem = {'id': 'root'}
        root = builder.build_tree(problem)

        # Manually create circular reference
        root.children.append(root)

        is_valid, errors = builder.validate_tree(root)

        assert is_valid is False
        assert any("circular" in error.lower() for error in errors)

    def test_tree_validation(self):
        """Test tree validation"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'status': 'complete',
            'subproblems': [
                {'id': 'child1', 'status': 'complete'},
                {'id': 'child2', 'status': 'failed'}
            ]
        }

        root = builder.build_tree(problem)
        is_valid, errors = builder.validate_tree(root)

        assert is_valid is True
        assert len(errors) == 0


class TestASCIITreeRenderer:
    """Tests for ASCIITreeRenderer"""

    def test_render_simple_tree(self):
        """Test rendering a simple tree"""
        builder = ProblemTreeBuilder()
        problem = {'id': 'root', 'status': 'complete', 'score': 85}

        root = builder.build_tree(problem)
        renderer = ASCIITreeRenderer()

        output = renderer.render(root)

        assert 'root' in output
        assert 'complete' in output
        assert '85' in output
        assert '[OK]' in output or 'complete' in output

    def test_render_hierarchy(self):
        """Test rendering a hierarchy"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'status': 'complete',
            'subproblems': [
                {'id': 'child1', 'status': 'complete', 'score': 90},
                {'id': 'child2', 'status': 'failed', 'score': 30}
            ]
        }

        root = builder.build_tree(problem)
        renderer = ASCIITreeRenderer()

        output = renderer.render(root)

        assert 'root' in output
        assert 'child1' in output
        assert 'child2' in output
        assert '├──' in output or '└──' in output

    def test_status_symbols(self):
        """Test status symbol rendering"""
        renderer = ASCIITreeRenderer()

        assert renderer._status_symbol(ProblemStatus.PENDING) == "⏳"
        assert renderer._status_symbol(ProblemStatus.IN_PROGRESS) == "🔄"
        assert renderer._status_symbol(ProblemStatus.COMPLETE) == "[OK]"
        assert renderer._status_symbol(ProblemStatus.FAILED) == "[FAIL]"

    def test_timing_display(self):
        """Test timing information display"""
        builder = ProblemTreeBuilder()
        problem = {'id': 'root', 'status': 'complete', 'timing_ms': 1500}

        root = builder.build_tree(problem)
        renderer = ASCIITreeRenderer(show_timing=True)

        output = renderer.render(root)

        assert '1500ms' in output or '1.5' in output

    def test_teams_display(self):
        """Test team history display"""
        builder = ProblemTreeBuilder()
        problem = {'id': 'root', 'status': 'complete', 'teams': ['Blue', 'Red', 'Gold']}

        root = builder.build_tree(problem)
        renderer = ASCIITreeRenderer(show_teams=True)

        output = renderer.render(root)

        assert 'Blue' in output
        assert 'Red' in output or '->' in output


class TestHTMLTreeRenderer:
    """Tests for HTMLTreeRenderer"""

    def test_render_html(self):
        """Test HTML rendering"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'status': 'complete',
            'score': 85,
            'subproblems': [
                {'id': 'child1', 'status': 'complete'}
            ]
        }

        root = builder.build_tree(problem)
        renderer = HTMLTreeRenderer()

        html = renderer.render_html(root)

        assert '<!DOCTYPE html>' in html
        assert '<html' in html
        assert 'root' in html
        assert 'child1' in html
        assert 'status-complete' in html

    def test_css_styling(self):
        """Test CSS classes are applied"""
        builder = ProblemTreeBuilder()
        problem = {'id': 'root', 'status': 'complete'}

        root = builder.build_tree(problem)
        renderer = HTMLTreeRenderer()

        html = renderer.render_html(root)

        assert 'class="problem-tree"' in html
        assert 'class="tree-node"' in html
        assert 'class="node-status"' in html

    def test_javascript_functionality(self):
        """Test JavaScript for interactivity"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'subproblems': [
                {'id': 'child1'}
            ]
        }

        root = builder.build_tree(problem)
        renderer = HTMLTreeRenderer()

        html = renderer.render_html(root)

        assert 'function toggleNode' in html
        assert 'onclick' in html
        assert 'expand-icon' in html

    def test_score_coloring(self):
        """Test score-based color coding"""
        renderer = HTMLTreeRenderer()

        assert renderer._score_color(90) == "#28a745"  # Green
        assert renderer._score_color(70) == "#ffc107"  # Yellow
        assert renderer._score_color(50) == "#fd7e14"  # Orange
        assert renderer._score_color(30) == "#dc3545"  # Red


class TestGraphvizTreeRenderer:
    """Tests for GraphvizTreeRenderer"""

    def test_render_dot(self):
        """Test DOT format rendering"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'status': 'complete',
            'subproblems': [
                {'id': 'child1', 'status': 'complete'}
            ]
        }

        root = builder.build_tree(problem)
        renderer = GraphvizTreeRenderer()

        dot = renderer.render_dot(root)

        assert 'digraph ProblemHierarchy' in dot
        assert 'node_0' in dot
        assert '->' in dot
        assert '}' in dot

    def test_node_labels(self):
        """Test node labels in DOT output"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'status': 'complete',
            'score': 85,
            'timing_ms': 1500,
            'teams': ['Blue', 'Red']
        }

        root = builder.build_tree(problem)
        renderer = GraphvizTreeRenderer(show_timing=True, show_teams=True)

        dot = renderer.render_dot(root)

        assert 'root' in dot
        assert 'complete' in dot
        assert '85' in dot
        assert '1500' in dot
        assert 'Blue' in dot

    def test_edge_rendering(self):
        """Test edge rendering between nodes"""
        builder = ProblemTreeBuilder()
        problem = {
            'id': 'root',
            'subproblems': [
                {'id': 'child1'},
                {'id': 'child2'}
            ]
        }

        root = builder.build_tree(problem)
        renderer = GraphvizTreeRenderer()

        dot = renderer.render_dot(root)

        # Should have edges from parent to children
        assert '->' in dot
        # Count edges (should be 2)
        edge_count = dot.count('->')
        assert edge_count >= 2

    def test_status_colors(self):
        """Test status-based fill colors"""
        renderer = GraphvizTreeRenderer()

        assert renderer._status_color(ProblemStatus.PENDING) == "#fff3cd"
        assert renderer._status_color(ProblemStatus.IN_PROGRESS) == "#d1ecf1"
        assert renderer._status_color(ProblemStatus.COMPLETE) == "#d4edda"
        assert renderer._status_color(ProblemStatus.FAILED) == "#f8d7da"


class TestVisualizationAPI:
    """Tests for VisualizationAPI"""

    def test_visualize_ascii(self):
        """Test ASCII visualization through API"""
        api = VisualizationAPI()
        problem = {
            'id': 'root',
            'status': 'complete',
            'score': 85,
            'subproblems': [
                {'id': 'child1', 'status': 'complete', 'score': 90}
            ]
        }

        result = api.visualize_problem(problem, OutputFormat.ASCII)

        assert 'root' in result
        assert 'child1' in result
        assert 'complete' in result

    def test_visualize_html(self):
        """Test HTML visualization through API"""
        api = VisualizationAPI()
        problem = {'id': 'root', 'status': 'complete'}

        result = api.visualize_problem(problem, OutputFormat.HTML)

        assert '<!DOCTYPE html>' in result
        assert 'root' in result

    def test_visualize_dot(self):
        """Test DOT visualization through API"""
        api = VisualizationAPI()
        problem = {
            'id': 'root',
            'subproblems': [
                {'id': 'child1'}
            ]
        }

        result = api.visualize_problem(problem, OutputFormat.DOT)

        assert 'digraph ProblemHierarchy' in result
        assert 'root' in result

    def test_visualize_with_options(self):
        """Test visualization with display options"""
        api = VisualizationAPI()
        problem = {
            'id': 'root',
            'status': 'complete',
            'score': 85,
            'timing_ms': 1500,
            'teams': ['Blue', 'Red'],
            'metadata': {'key': 'value'}
        }

        # All options enabled
        result = api.visualize_problem(
            problem,
            OutputFormat.ASCII,
            show_metadata=True,
            show_timing=True,
            show_teams=True
        )

        assert 'root' in result
        assert '1500' in result or '1.5' in result
        assert 'Blue' in result or 'Red' in result

        # Timing disabled
        result_no_timing = api.visualize_problem(
            problem,
            OutputFormat.ASCII,
            show_timing=False
        )

        # Should still have content but less detail
        assert 'root' in result_no_timing


class TestConvenienceFunction:
    """Tests for visualize_problem convenience function"""

    def test_convenience_function(self):
        """Test the convenience function"""
        problem = {
            'id': 'root',
            'status': 'complete',
            'subproblems': [
                {'id': 'child1', 'status': 'complete'}
            ]
        }

        result = visualize_problem(problem, format='ascii')

        assert 'root' in result
        assert 'child1' in result

    def test_format_string_conversion(self):
        """Test format string to enum conversion"""
        problem = {'id': 'root'}

        # Test all formats
        for format_str in ['ascii', 'html', 'dot']:
            result = visualize_problem(problem, format=format_str)
            assert result is not None
            assert len(result) > 0

    def test_invalid_format(self):
        """Test invalid format handling"""
        problem = {'id': 'root'}

        with pytest.raises(ValueError):
            visualize_problem(problem, format='invalid')


class TestVisualizationIntegration:
    """Integration tests for complete visualization workflow"""

    def test_complex_hierarchy(self):
        """Test visualization of complex multi-level hierarchy"""
        problem = {
            'id': 'root_problem',
            'status': 'complete',
            'score': 85,
            'timing_ms': 5000,
            'teams': ['Blue', 'Red', 'Gold'],
            'attempt_count': 2,
            'metadata': {'domain': 'math', 'difficulty': 'hard'},
            'subproblems': [
                {
                    'id': 'subproblem_1',
                    'status': 'complete',
                    'score': 90,
                    'timing_ms': 2000,
                    'teams': ['Blue', 'Red'],
                    'subproblems': [
                        {
                            'id': 'subproblem_1_a',
                            'status': 'complete',
                            'score': 95,
                            'timing_ms': 500,
                            'teams': ['Blue']
                        },
                        {
                            'id': 'subproblem_1_b',
                            'status': 'complete',
                            'score': 85,
                            'timing_ms': 1500,
                            'teams': ['Blue', 'Red']
                        }
                    ]
                },
                {
                    'id': 'subproblem_2',
                    'status': 'complete',
                    'score': 80,
                    'timing_ms': 3000,
                    'teams': ['Blue', 'Red']
                },
                {
                    'id': 'subproblem_3',
                    'status': 'failed',
                    'score': 30,
                    'timing_ms': 1000,
                    'teams': ['Blue', 'Red']
                }
            ]
        }

        # Test ASCII
        ascii_result = visualize_problem(problem, format='ascii')
        assert 'root_problem' in ascii_result
        assert 'subproblem_1' in ascii_result
        assert 'subproblem_1_a' in ascii_result
        assert '95' in ascii_result  # score
        assert '[OK]' in ascii_result  # complete status
        assert '[FAIL]' in ascii_result  # failed status

        # Test HTML
        html_result = visualize_problem(problem, format='html')
        assert 'root_problem' in html_result
        assert 'status-complete' in html_result
        assert 'status-failed' in html_result

        # Test DOT
        dot_result = visualize_problem(problem, format='dot')
        assert 'digraph ProblemHierarchy' in dot_result
        assert 'root_problem' in dot_result

    def test_wide_hierarchy(self):
        """Test visualization of wide hierarchy (many siblings)"""
        problem = {
            'id': 'root',
            'subproblems': [
                {'id': f'child_{i}', 'status': 'complete'}
                for i in range(10)
            ]
        }

        result = visualize_problem(problem, format='ascii')

        assert 'root' in result
        for i in range(10):
            assert f'child_{i}' in result

    def test_deep_hierarchy(self):
        """Test visualization of deep hierarchy (many levels)"""
        # Create a deep hierarchy
        problem = {'id': 'level_0'}
        current = problem

        for i in range(1, 6):  # 6 levels total
            child = {'id': f'level_{i}'}
            current['subproblems'] = [child]
            current = child

        result = visualize_problem(problem, format='ascii')

        for i in range(6):
            assert f'level_{i}' in result

    def test_mixed_status(self):
        """Test visualization with mixed problem statuses"""
        problem = {
            'id': 'root',
            'subproblems': [
                {'id': 'pending', 'status': 'pending'},
                {'id': 'in_progress', 'status': 'in_progress'},
                {'id': 'complete', 'status': 'complete'},
                {'id': 'failed', 'status': 'failed'}
            ]
        }

        result = visualize_problem(problem, format='ascii')

        assert 'pending' in result
        assert 'in_progress' in result
        assert 'complete' in result
        assert 'failed' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
