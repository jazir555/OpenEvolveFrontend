"""
Example Usage of Long-Horizon Agentic Framework

This example demonstrates:
1. Creating a long-running workflow
2. Saving and loading state
3. Learning from outcomes
4. Temporal context tracking
5. Checkpoint creation and replay

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
from datetime import datetime, timezone, timedelta
from openvolve.long_horizon import (
    create_framework,
    ExplorationStrategy
)
from openvolve.long_horizon.schemas import (
    TemporalEvent,
    TimeWindow,
    StateLevel
)


async def example_state_management():
    """Example: Persistent state management"""
    print("\n=== State Management Example ===\n")

    framework = create_framework()
    state_manager = framework['state_manager']

    # Save initial state
    snapshot = await state_manager.save_snapshot(
        state_data={
            'model_version': '1.0',
            'training_data': [1, 2, 3, 4, 5],
            'accuracy': 0.85
        },
        level=StateLevel.WORKFLOW,
        workflow_id='training_workflow',
        is_checkpoint=True,
        checkpoint_name='initial_model',
        created_by='example_agent'
    )

    print(f"Saved snapshot: {snapshot.snapshot_id}")

    # Load state
    loaded = await state_manager.load_snapshot(snapshot.snapshot_id)
    print(f"Loaded state: {loaded.state_data}")

    # Update state (versioning)
    snapshot2 = await state_manager.save_snapshot(
        state_data={
            'model_version': '2.0',
            'training_data': [1, 2, 3, 4, 5, 6, 7],
            'accuracy': 0.92
        },
        level=StateLevel.WORKFLOW,
        workflow_id='training_workflow',
        parent_snapshot_id=snapshot.snapshot_id,
        created_by='example_agent'
    )

    print(f"Created new version: {snapshot2.snapshot_id}")

    # Get history
    history = await state_manager.get_history(snapshot2.snapshot_id)
    print(f"Version history: {len(history)} versions")

    state_manager.close()


async def example_workflow_execution():
    """Example: Workflow orchestration"""
    print("\n=== Workflow Execution Example ===\n")

    framework = create_framework()
    orchestrator = framework['workflow_orchestrator']

    # Create workflow
    await orchestrator.create_workflow(
        workflow_id='data_pipeline',
        name='Data Processing Pipeline',
        description='Processes data in three stages',
        steps=[
            {'step_id': 'fetch', 'type': 'data_fetch', 'source': 'api'},
            {'step_id': 'transform', 'type': 'data_transform', 'method': 'normalize'},
            {'step_id': 'load', 'type': 'data_load', 'destination': 'database'}
        ],
        created_by='example_agent'
    )

    print("Created workflow: data_pipeline")

    # Register step handlers
    async def fetch_handler(workflow_def, execution, step):
        print(f"Fetching data from {step['source']}...")
        await asyncio.sleep(0.5)  # Simulate work
        return {'records': 1000, 'source': step['source']}

    async def transform_handler(workflow_def, execution, step):
        print(f"Transforming data with {step['method']}...")
        await asyncio.sleep(0.5)
        return {'transformed': 1000, 'method': step['method']}

    async def load_handler(workflow_def, execution, step):
        print(f"Loading data to {step['destination']}...")
        await asyncio.sleep(0.5)
        return {'loaded': 1000, 'destination': step['destination']}

    orchestrator.register_step_handler('data_fetch', fetch_handler)
    orchestrator.register_step_handler('data_transform', transform_handler)
    orchestrator.register_step_handler('data_load', load_handler)

    # Start workflow
    execution = await orchestrator.start_workflow(
        workflow_id='data_pipeline',
        input_parameters={'batch_size': 100}
    )

    print(f"Started execution: {execution.execution_id}")
    print(f"Status: {execution.status}")

    # Wait a bit for execution
    await asyncio.sleep(2)

    # Close
    await orchestrator.shutdown()


async def example_learning():
    """Example: Learning and adaptation"""
    print("\n=== Learning Example ===\n")

    framework = create_framework()
    learning_engine = framework['learning_engine']

    # Simulate multiple executions with different strategies
    strategies = ['conservative', 'moderate', 'aggressive']

    for i, strategy in enumerate(strategies * 10):
        # Simulate execution outcome
        import random
        success = random.random() > 0.3  # 70% success rate
        performance = random.uniform(0.5, 0.95) if success else random.uniform(0.1, 0.4)

        # Record outcome
        await learning_engine.record_outcome(
            workflow_id='trading_workflow',
            execution_id=f'exec_{i}',
            lesson_type='success' if success else 'failure',
            lesson_description=f'Execution with {strategy} strategy',
            success=success,
            performance_score=performance,
            strategy_used=strategy,
            parameters={'risk_level': strategy},
            learned_by='trading_agent'
        )

    print("Recorded 30 learning outcomes")

    # Get strategy recommendations
    recommendations = await learning_engine.get_strategy_recommendations()
    print("\nStrategy Recommendations:")
    for rec in recommendations[:3]:
        print(f"  {rec['strategy_id']}: "
              f"avg_performance={rec['avg_performance']:.2f}, "
              f"success_rate={rec['success_rate']:.2f}, "
              f"trend={rec['trend']}")

    # Select optimal strategy
    selected = await learning_engine.select_strategy(
        available_strategies=strategies,
        exploration_strategy=ExplorationStrategy.EPSILON_GREEDY
    )
    print(f"\nSelected strategy: {selected}")

    # Get learning summary
    summary = await learning_engine.get_learning_summary()
    print(f"\nLearning Summary:")
    print(f"  Total outcomes: {summary['total_outcomes']}")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Strategies tracked: {summary['strategies_tracked']}")


async def example_temporal_context():
    """Example: Temporal context management"""
    print("\n=== Temporal Context Example ===\n")

    framework = create_framework()
    temporal_context = framework['temporal_context']

    # Add temporal events
    now = datetime.now(timezone.utc)

    events = [
        TemporalEvent(
            event_id='event_1',
            event_type='model_update',
            timestamp=now - timedelta(hours=2),
            event_data={'version': '1.0', 'accuracy': 0.85},
            source='system',
            importance=0.7
        ),
        TemporalEvent(
            event_id='event_2',
            event_type='model_update',
            timestamp=now - timedelta(hours=1),
            event_data={'version': '1.1', 'accuracy': 0.88},
            source='system',
            importance=0.8
        ),
        TemporalEvent(
            event_id='event_3',
            event_type='model_update',
            timestamp=now,
            event_data={'version': '1.2', 'accuracy': 0.91},
            source='system',
            importance=0.9
        )
    ]

    for event in events:
        await temporal_context.add_event(event)

    print("Added 3 temporal events")

    # Query events in time window
    time_window = TimeWindow(
        window_id='last_3_hours',
        start_time=now - timedelta(hours=3),
        end_time=now + timedelta(hours=1)
    )

    recent_events = await temporal_context.get_events(time_window)
    print(f"\nEvents in last 3 hours: {len(recent_events)}")

    # Analyze trend
    trend = await temporal_context.analyze_trend(
        metric_name='accuracy',
        time_window=time_window
    )

    print(f"\nAccuracy Trend:")
    print(f"  Type: {trend.trend_type}")
    print(f"  Slope: {trend.slope:.4f}")
    print(f"  Correlation: {trend.correlation:.4f}")
    print(f"  Is anomaly: {trend.is_anomaly}")
    print(f"  Impact: {trend.impact_level}")


async def example_checkpoint_replay():
    """Example: Checkpoint and replay"""
    print("\n=== Checkpoint & Replay Example ===\n")

    framework = create_framework()
    state_manager = framework['state_manager']
    checkpoint_manager = framework['checkpoint_manager']

    # Create snapshot
    snapshot = await state_manager.save_snapshot(
        state_data={'epoch': 10, 'loss': 0.123, 'accuracy': 0.95},
        level=StateLevel.WORKFLOW,
        workflow_id='training_workflow',
        created_by='trainer'
    )

    print(f"Created snapshot: {snapshot.snapshot_id}")

    # Create checkpoint
    metadata = await checkpoint_manager.create_checkpoint(
        snapshot_id=snapshot.snapshot_id,
        checkpoint_name='epoch_10',
        checkpoint_type='milestone',
        workflow_id='training_workflow',
        created_by='trainer',
        description='Completed training epoch 10'
    )

    print(f"Created checkpoint: {metadata.checkpoint_name}")

    # Get checkpoints for workflow
    checkpoints = await state_manager.get_checkpoints('training_workflow')
    print(f"\nTotal checkpoints: {len(checkpoints)}")

    # Demonstrate replay (conceptual)
    print("\nCheckpoint can be used for:")
    print("  - Rollback to previous state")
    print("  - Debug with replay")
    print("  - Analyze execution")
    print("  - Retry with modifications")

    state_manager.close()


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Long-Horizon Agentic Framework - Examples")
    print("=" * 60)

    # Note: These examples require MongoDB and Neo4j to be running
    # Uncomment to run:

    # await example_state_management()
    # await example_workflow_execution()
    # await example_learning()
    # await example_temporal_context()
    # await example_checkpoint_replay()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

    print("\nNote: To run these examples, ensure MongoDB and Neo4j are running:")
    print("  docker-compose up -d")


if __name__ == '__main__':
    asyncio.run(main())
