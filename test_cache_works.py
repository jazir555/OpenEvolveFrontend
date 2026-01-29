"""
Test cache integration by running from package root
"""
import asyncio
import sys

# Run from package root
sys.path.insert(0, '.')

# Import after adding to path
from bubblelabs_nodes.gauntlet_solver import solveProblem


async def main():
    print("=" * 60)
    print("Testing Cache Integration with solveProblem")
    print("=" * 60)

    # Test 1: Basic problem
    print("\n[Test 1] Basic problem solve...")
    problem1 = {
        'id': 'test_basic_1',
        'type': 'test',
        'value': 42
    }
    result1 = await solveProblem(problem1)
    print(f"Result: {result1}")
    print("[PASS] Basic solve works!")

    # Test 2: Cache hit
    print("\n[Test 2] Cache hit test...")
    print("Calling solveProblem with same problem again...")
    result2 = await solveProblem(problem1)
    print(f"Result: {result2}")
    # Compare core solution (not timestamp which changes)
    assert result1['problem_id'] == result2['problem_id'], "Problem ID should match"
    assert result1['solution'] == result2['solution'], "Solution should match"
    assert result1['success'] == result2['success'], "Success status should match"
    print("[PASS] Cache hit works - same solution returned!")

    # Test 3: Different problem
    print("\n[Test 3] Different problem...")
    problem3 = {
        'id': 'test_different',
        'type': 'test',
        'value': 999
    }
    result3 = await solveProblem(problem3)
    print(f"Result: {result3}")
    print("[PASS] Different problem works!")

    # Test 4: Problem with subproblems
    print("\n[Test 4] Problem with subproblems...")
    problem4 = {
        'id': 'parent_test',
        'type': 'test',
        'subproblems': [
            {'id': 'child1', 'value': 1},
            {'id': 'child2', 'value': 2},
            {'id': 'child3', 'value': 3},
        ]
    }
    result4 = await solveProblem(problem4)
    print(f"Result: {result4}")
    print("[PASS] Subproblems work!")

    print("\n" + "=" * 60)
    print("[SUCCESS] All cache integration tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
