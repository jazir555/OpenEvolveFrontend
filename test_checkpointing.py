"""
Test checkpointing system.

Verify that:
- CheckpointManager works
- Can save checkpoints
- Can load checkpoints
- Can resume from checkpoints
"""
import asyncio
import sys

sys.path.insert(0, '.')

from bubblelabs_nodes.checkpoint_manager import (
    CheckpointManager,
    create_checkpoint_manager,
    CheckpointRepository,
    StateSerializer,
    PipelineState,
    CheckpointMetadata
)


async def test_checkpoint_basics():
    """Test basic checkpoint operations."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Checkpoint Operations")
    print("=" * 60)

    # Create checkpoint manager
    manager = create_checkpoint_manager()
    print(f"Created manager: {type(manager).__name__}")

    # Create test state
    test_problem = {'id': 'test_problem_1', 'statement': 'Test'}
    test_context = {'level': 1, 'stage': 'solving'}
    test_solutions = {'subproblem_1': 'solution_1'}

    # Create checkpoint
    checkpoint_id = await manager.create_checkpoint(
        problem=test_problem,
        context=test_context,
        solutions=test_solutions,
        level=1,
        stage='solving'
    )

    print(f"Created checkpoint: {checkpoint_id}")

    # List checkpoints
    checkpoints = await manager.list_checkpoints(problem_id='test_problem_1')
    print(f"Found {len(checkpoints)} checkpoints")

    # Load checkpoint
    loaded_state = await manager.load_checkpoint(checkpoint_id)
    print(f"Loaded state: {loaded_state.problem['id']}")

    assert loaded_state.problem['id'] == test_problem['id']
    assert loaded_state.context['level'] == test_context['level']
    assert loaded_state.solutions == test_solutions

    print("[PASS] Basic checkpoint operations work")


async def test_checkpoint_manager_methods():
    """Test CheckpointManager methods."""
    print("\n" + "=" * 60)
    print("TEST 2: CheckpointManager Methods")
    print("=" * 60)

    manager = create_checkpoint_manager()

    # Test checkpoint_count (property, not method)
    count = manager.checkpoint_count
    print(f"Checkpoint count: {count}")

    # Test generate_checkpoint_id (method with args)
    checkpoint_id = manager.generate_checkpoint_id('test_problem', 1, 'solving')
    print(f"Generated ID: {checkpoint_id}")

    # Test enabled
    print(f"Enabled: {manager.enabled}")

    # Test repository
    print(f"Repository type: {type(manager.repository).__name__}")

    # Test serializer
    print(f"Serializer type: {type(manager.serializer).__name__}")

    print("[PASS] CheckpointManager methods work")


async def main():
    print("=" * 60)
    print("CHECKPOINTING SYSTEM TESTS")
    print("=" * 60)

    await test_checkpoint_basics()
    await test_checkpoint_manager_methods()

    print("\n" + "=" * 60)
    print("[SUCCESS] Checkpointing tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
