"""
Test problem hierarchy visualization.

Verifies that:
- Tree builder works
- ASCII renderer works
- HTML renderer works
- Graphviz renderer works
- Visualization API works
"""
import sys

# Run from package root
sys.path.insert(0, '.')

from bubblelabs_nodes.problem_visualization import (
    visualize_problem,
    ProblemTreeBuilder,
    ASCIITreeRenderer,
    HTMLTreeRenderer,
    GraphvizTreeRenderer,
    VisualizationAPI,
    TreeNode,
    ProblemStatus,
    OutputFormat
)


def create_test_tree():
    """Create a test problem tree."""
    # Create root node
    root = TreeNode(
        problem_id='root_problem',
        status=ProblemStatus.COMPLETE,
        score=95.0,
        teams=['Blue', 'Red', 'Gold'],
        timing_ms=1250.0,
        attempt_count=1
    )

    # Add child nodes
    child1 = TreeNode(
        problem_id='subproblem_1',
        status=ProblemStatus.COMPLETE,
        score=88.0,
        teams=['Blue', 'Red', 'Gold'],
        timing_ms=450.0,
        attempt_count=1,
        parent=root
    )

    child2 = TreeNode(
        problem_id='subproblem_2',
        status=ProblemStatus.COMPLETE,
        score=92.0,
        teams=['Blue', 'Red', 'Gold'],
        timing_ms=380.0,
        attempt_count=1,
        parent=root
    )

    child3 = TreeNode(
        problem_id='subproblem_3',
        status=ProblemStatus.FAILED,
        score=45.0,
        teams=['Blue', 'Red'],
        timing_ms=420.0,
        attempt_count=3,
        parent=root
    )

    # Add grandchildren
    grandchild1 = TreeNode(
        problem_id='subproblem_1_1',
        status=ProblemStatus.COMPLETE,
        score=90.0,
        teams=['Blue', 'Red', 'Gold'],
        timing_ms=200.0,
        attempt_count=1,
        parent=child1
    )

    grandchild2 = TreeNode(
        problem_id='subproblem_1_2',
        status=ProblemStatus.COMPLETE,
        score=85.0,
        teams=['Blue', 'Red', 'Gold'],
        timing_ms=250.0,
        attempt_count=1,
        parent=child1
    )

    # Build tree structure
    root.children = [child1, child2, child3]
    child1.children = [grandchild1, grandchild2]

    return root


def test_ascii_renderer():
    """Test ASCII tree renderer."""
    print("\n" + "=" * 60)
    print("TEST 1: ASCII Tree Renderer")
    print("=" * 60)

    tree = create_test_tree()
    renderer = ASCIITreeRenderer()

    output = renderer.render(tree)

    # Save output to file for inspection
    with open('test_ascii_debug.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print("\n  Saved debug output to: test_ascii_debug.txt")

    # Check for expected content (may be in different format)
    assert 'root_problem' in output, "Should contain root problem ID"
    assert 'subproblem_1' in output, "Should contain child problem ID"
    assert 'subproblem_1_1' in output, "Should contain grandchild problem ID"
    # Status may be in different format
    has_status = 'COMPLETE' in output or 'complete' in output or '[OK]' in output
    assert has_status, "Should show status"
    # Score may be in different format
    has_score = '95' in output or '95.0' in output
    assert has_score, "Should show score"

    print("\n  ASCII output generated (length: {} chars)".format(len(output)))
    print("  First 200 chars:")
    try:
        print("  " + output[:200])
    except UnicodeEncodeError:
        print("  [Unicode box-drawing characters - cannot display on Windows console]")
        print("  Output is valid and can be saved to file")

    print("\n[PASS] ASCII renderer working")


def test_html_renderer():
    """Test HTML tree renderer."""
    print("\n" + "=" * 60)
    print("TEST 2: HTML Tree Renderer")
    print("=" * 60)

    tree = create_test_tree()
    renderer = HTMLTreeRenderer()

    output = renderer.render_html(tree)

    # Check for HTML structure
    assert '<!DOCTYPE html>' in output or '<html' in output, "Should be HTML"
    assert 'root_problem' in output, "Should contain root problem ID"
    assert 'subproblem_1' in output, "Should contain child problem ID"
    # Status may be in different format
    has_status = 'COMPLETE' in output or 'complete' in output
    assert has_status, "Should show status"
    # Score may be in different format
    has_score = '95' in output or '95.0' in output
    assert has_score, "Should show score"
    assert '<div' in output or '<ul' in output, "Should have HTML structure"

    # Save to file for manual inspection
    with open('test_tree_output.html', 'w', encoding='utf-8') as f:
        f.write(output)
    print("\n  Saved to: test_tree_output.html")

    print("[PASS] HTML renderer working")


def test_graphviz_renderer():
    """Test Graphviz DOT renderer."""
    print("\n" + "=" * 60)
    print("TEST 3: Graphviz DOT Renderer")
    print("=" * 60)

    tree = create_test_tree()
    renderer = GraphvizTreeRenderer()

    output = renderer.render_dot(tree)

    assert 'digraph' in output, "Should be DOT format"
    assert 'root_problem' in output, "Should contain root problem ID"
    assert 'subproblem_1' in output, "Should contain child problem ID"
    assert '->' in output, "Should have edges"

    # Save to file for manual inspection
    with open('test_tree_output.dot', 'w', encoding='utf-8') as f:
        f.write(output)
    print("\n  Saved to: test_tree_output.dot")
    print("  Render with: dot -Tpng test_tree_output.dot -o tree.png")

    print("[PASS] Graphviz renderer working")


def test_visualization_api():
    """Test VisualizationAPI convenience interface."""
    print("\n" + "=" * 60)
    print("TEST 4: Visualization API")
    print("=" * 60)

    # Create test problem dict
    problem = {
        'id': 'root_problem',
        'statement': 'Test problem',
        'subproblems': [
            {
                'id': 'child_1',
                'statement': 'First subproblem',
                'subproblems': [
                    {'id': 'grandchild_1', 'statement': 'Deep subproblem'}
                ]
            },
            {'id': 'child_2', 'statement': 'Second subproblem'}
        ]
    }

    api = VisualizationAPI()

    # Test ASCII format
    ascii_output = api.visualize_problem(problem, OutputFormat.ASCII)
    assert 'root_problem' in ascii_output
    print("\n  ASCII format: OK")

    # Test HTML format
    html_output = api.visualize_problem(problem, OutputFormat.HTML)
    assert '<!DOCTYPE html>' in html_output or '<html' in html_output
    print("  HTML format: OK")

    # Test DOT format
    dot_output = api.visualize_problem(problem, OutputFormat.DOT)
    assert 'digraph' in dot_output
    print("  DOT format: OK")

    print("\n[PASS] Visualization API working")


def test_problem_from_dict():
    """Test building tree from problem dictionary."""
    print("\n" + "=" * 60)
    print("TEST 5: Build Tree from Problem Dict")
    print("=" * 60)

    problem = {
        'id': 'test_problem',
        'statement': 'Solve this complex problem',
        'type': 'test',
        'subproblems': [
            {
                'id': 'child_1',
                'statement': 'First subproblem',
                'subproblems': [
                    {'id': 'grandchild_1', 'statement': 'Deep subproblem'}
                ]
            },
            {
                'id': 'child_2',
                'statement': 'Second subproblem'
            }
        ]
    }

    builder = ProblemTreeBuilder()
    tree = builder.build_tree(problem)

    assert tree.problem_id == 'test_problem'
    assert len(tree.children) == 2
    assert tree.children[0].problem_id == 'child_1'
    assert len(tree.children[0].children) == 1
    assert tree.children[0].children[0].problem_id == 'grandchild_1'

    # Render it
    renderer = ASCIITreeRenderer()
    output = renderer.render(tree)
    print("\n  ASCII output generated successfully")
    try:
        print("\n" + output[:300])
    except UnicodeEncodeError:
        print("  [Unicode output - see test_tree_ascii.txt for full output]")
        with open('test_tree_ascii.txt', 'w', encoding='utf-8') as f:
            f.write(output)
        print("  Saved to: test_tree_ascii.txt")

    print("\n[PASS] Tree builder working with problem dicts")


def test_visualize_problem_function():
    """Test the convenience function."""
    print("\n" + "=" * 60)
    print("TEST 6: visualize_problem() Function")
    print("=" * 60)

    problem = {
        'id': 'convenience_test',
        'statement': 'Test the convenience function',
        'type': 'test',
        'subproblems': [
            {'id': 'child_1', 'statement': 'Child 1'},
            {'id': 'child_2', 'statement': 'Child 2'}
        ]
    }

    # Test ASCII format (default) - pass string not enum
    ascii_output = visualize_problem(problem, 'ascii')
    assert 'convenience_test' in ascii_output
    print("\n  ASCII output: OK")

    # Test HTML format
    html_output = visualize_problem(problem, 'html')
    assert '<!DOCTYPE html>' in html_output or '<html' in html_output
    print("  HTML output: OK")

    # Test DOT format
    dot_output = visualize_problem(problem, 'dot')
    assert 'digraph' in dot_output
    print("  DOT output: OK")

    print("\n[PASS] Convenience function working")


def main():
    print("=" * 60)
    print("PROBLEM HIERARCHY VISUALIZATION TESTS")
    print("=" * 60)

    test_ascii_renderer()
    test_html_renderer()
    test_graphviz_renderer()
    test_visualization_api()
    test_problem_from_dict()
    test_visualize_problem_function()

    print("\n" + "=" * 60)
    print("[SUCCESS] All visualization tests passed!")
    print("=" * 60)

    print("\n[VISUALIZATION CAPABILITIES]")
    print("  - ASCII tree rendering with box-drawing characters")
    print("  - HTML interactive tree with collapsible nodes")
    print("  - Graphviz DOT format for PNG/SVG rendering")
    print("  - Automatic tree building from problem dicts")
    print("  - Metadata display (status, score, teams, timing)")
    print("  - Convenience API for easy visualization")


if __name__ == '__main__':
    main()
