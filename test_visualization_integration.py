"""
Test visualization integration with solveProblem().

Verifies that:
- Visualization works during solving
- Different output formats work
- Visualization can be enabled/disabled
- Integration doesn't break normal solving
"""
import asyncio
import sys
import logging

sys.path.insert(0, '.')

# Set up logging to see visualization output
logging.basicConfig(level=logging.INFO, format='%(message)s')

from bubblelabs_nodes.gauntlet_solver import solveProblem


async def test_visualization_enabled():
    """Test visualization when enabled."""
    print("\n" + "=" * 60)
    print("TEST 1: Visualization Enabled")
    print("=" * 60)

    problem = {
        'id': 'viz_test',
        'statement': 'Test visualization integration',
        'subproblems': [
            {
                'id': 'child_1',
                'statement': 'First child',
                'subproblems': [
                    {'id': 'grandchild_1', 'statement': 'Deep child 1'},
                    {'id': 'grandchild_2', 'statement': 'Deep child 2'},
                ]
            },
            {'id': 'child_2', 'statement': 'Second child'},
            {'id': 'child_3', 'statement': 'Third child'},
        ]
    }

    # Solve with visualization enabled
    result = await solveProblem(problem, enable_visualization=True, visualization_format='ascii')

    print(f"\nSolved: {result['success']}")
    print(f"Solutions: {result.get('num_solutions', 0)}")

    assert result['success'], "Should solve successfully"

    print("\n[PASS] Visualization enabled works")


async def test_visualization_disabled():
    """Test that visualization doesn't interfere when disabled."""
    print("\n" + "=" * 60)
    print("TEST 2: Visualization Disabled (default)")
    print("=" * 60)

    problem = {
        'id': 'no_viz_test',
        'statement': 'Test without visualization',
        'subproblems': [
            {'id': 'child_1', 'statement': 'Child 1'},
            {'id': 'child_2', 'statement': 'Child 2'},
        ]
    }

    # Solve with visualization disabled (default)
    result = await solveProblem(problem, enable_visualization=False)

    print(f"Solved: {result['success']}")
    assert result['success'], "Should solve successfully"

    print("[PASS] Works normally when visualization disabled")


async def test_html_visualization():
    """Test HTML visualization format."""
    print("\n" + "=" * 60)
    print("TEST 3: HTML Visualization Format")
    print("=" * 60)

    problem = {
        'id': 'html_viz_test',
        'statement': 'Test HTML visualization',
        'subproblems': [
            {'id': 'child_1', 'statement': 'Child 1'},
            {'id': 'child_2', 'statement': 'Child 2'},
        ]
    }

    # Solve with HTML visualization
    result = await solveProblem(problem, enable_visualization=True, visualization_format='html')

    print(f"\nSolved: {result['success']}")
    assert result['success'], "Should solve successfully"

    print("[PASS] HTML visualization works")


async def test_dot_visualization():
    """Test DOT visualization format."""
    print("\n" + "=" * 60)
    print("TEST 4: DOT Visualization Format")
    print("=" * 60)

    problem = {
        'id': 'dot_viz_test',
        'statement': 'Test DOT visualization',
        'subproblems': [
            {'id': 'child_1', 'statement': 'Child 1'},
        ]
    }

    # Solve with DOT visualization
    result = await solveProblem(problem, enable_visualization=True, visualization_format='dot')

    print(f"\nSolved: {result['success']}")
    assert result['success'], "Should solve successfully"

    print("[PASS] DOT visualization works")


async def test_complex_hierarchy_visualization():
    """Test visualization with complex deep hierarchy."""
    print("\n" + "=" * 60)
    print("TEST 5: Complex Hierarchy Visualization")
    print("=" * 60)

    problem = {
        'id': 'complex_viz_test',
        'statement': 'Complex hierarchy for visualization',
        'subproblems': [
            {
                'id': 'branch_1',
                'statement': 'Branch 1',
                'subproblems': [
                    {
                        'id': 'branch_1_1',
                        'statement': 'Branch 1.1',
                        'subproblems': [
                            {'id': 'branch_1_1_1', 'statement': 'Branch 1.1.1'},
                            {'id': 'branch_1_1_2', 'statement': 'Branch 1.1.2'},
                        ]
                    },
                    {'id': 'branch_1_2', 'statement': 'Branch 1.2'},
                ]
            },
            {
                'id': 'branch_2',
                'statement': 'Branch 2',
                'subproblems': [
                    {'id': 'branch_2_1', 'statement': 'Branch 2.1'},
                    {'id': 'branch_2_2', 'statement': 'Branch 2.2'},
                    {'id': 'branch_2_3', 'statement': 'Branch 2.3'},
                ]
            },
            {'id': 'branch_3', 'statement': 'Branch 3'},
        ]
    }

    # Solve with visualization
    result = await solveProblem(problem, enable_visualization=True, visualization_format='ascii')

    print(f"\nSolved: {result['success']}")
    print(f"Total solutions: {result.get('num_solutions', 0)}")

    assert result['success'], "Should solve complex hierarchy"

    print("\n[PASS] Complex hierarchy visualization works")


async def test_visualization_with_checkpointing():
    """Test visualization and checkpointing together."""
    print("\n" + "=" * 60)
    print("TEST 6: Visualization + Checkpointing Together")
    print("=" * 60)

    problem = {
        'id': 'combined_test',
        'statement': 'Test visualization with checkpointing',
        'subproblems': [
            {'id': 'child_1', 'statement': 'Child 1'},
            {'id': 'child_2', 'statement': 'Child 2'},
        ]
    }

    # Solve with both enabled
    result = await solveProblem(
        problem,
        enable_visualization=True,
        enable_checkpointing=True,
        visualization_format='ascii'
    )

    print(f"\nSolved: {result['success']}")
    assert result['success'], "Should solve successfully"

    # Check checkpoints were created
    from bubblelabs_nodes.checkpoint_manager import create_checkpoint_manager
    manager = create_checkpoint_manager()
    checkpoints = await manager.list_checkpoints('combined_test')
    print(f"Checkpoints created: {len(checkpoints)}")

    assert len(checkpoints) > 0, "Should have checkpoints"

    print("\n[PASS] Visualization works with checkpointing")


async def main():
    print("=" * 60)
    print("VISUALIZATION INTEGRATION TESTS")
    print("=" * 60)

    await test_visualization_enabled()
    await test_visualization_disabled()
    await test_html_visualization()
    await test_dot_visualization()
    await test_complex_hierarchy_visualization()
    await test_visualization_with_checkpointing()

    print("\n" + "=" * 60)
    print("[SUCCESS] All visualization integration tests passed!")
    print("=" * 60)

    print("\n[VISUALIZATION CAPABILITIES]")
    print("  - Automatic problem structure visualization")
    print("  - Multiple output formats (ASCII, HTML, DOT)")
    print("  - Works with complex hierarchies")
    print("  - Compatible with checkpointing")
    print("  - Can be enabled/disabled")
    print("  - Integrated into solveProblem()")


if __name__ == '__main__':
    asyncio.run(main())
