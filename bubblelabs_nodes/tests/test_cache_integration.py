"""
Test cache integration with solveProblem
"""
import pytest
import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports to avoid complex __init__.py
import gauntlet_solver


@pytest.mark.asyncio
async def test_solveProblem_basic():
    """Test that solveProblem works with a basic problem"""
    problem = {
        'id': 'test_problem_1',
        'type': 'test',
        'value': 42
    }

    result = await gauntlet_solver.solveProblem(problem)

    assert result is not None
    assert 'problem_id' in result
    print(f"[OK] solveProblem basic test passed: {result}")


@pytest.mark.asyncio
async def test_solveProblem_caching():
    """Test that cache actually works with solveProblem"""
    problem = {
        'id': 'cache_test_problem',
        'type': 'test',
        'value': 123
    }

    # First call - cache miss
    result1 = await gauntlet_solver.solveProblem(problem)
    assert result1 is not None

    # Second call - should hit cache (same problem definition)
    result2 = await gauntlet_solver.solveProblem(problem)
    assert result2 is not None

    # Results should be identical
    assert result1 == result2
    print(f"[OK] Cache integration test passed")


@pytest.mark.asyncio
async def test_solveProblem_with_subproblems():
    """Test solveProblem with subproblems"""
    problem = {
        'id': 'parent_problem',
        'type': 'test',
        'subproblems': [
            {'id': 'child1', 'value': 1},
            {'id': 'child2', 'value': 2},
            {'id': 'child3', 'value': 3},
        ]
    }

    result = await gauntlet_solver.solveProblem(problem)

    assert result is not None
    assert 'success' in result
    print(f"[OK] Subproblem test passed: {result}")


if __name__ == '__main__':
    asyncio.run(test_solveProblem_basic())
    asyncio.run(test_solveProblem_caching())
    asyncio.run(test_solveProblem_with_subproblems())
    print("\n[SUCCESS] All integration tests passed!")
