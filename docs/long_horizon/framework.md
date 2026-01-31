# Long-Horizon Agentic Framework Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Components](#core-components)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Overview

The Long-Horizon Agentic Framework enables AI agents to maintain state, learn, and operate across days, weeks, and months. It provides production-ready infrastructure for persistent, stateful agent workflows.

### Key Capabilities

- **Persistent State Management**: Git-like versioning with checkpoint/rollback
- **Time-Aware Orchestration**: Scheduling, deadlines, and temporal reasoning
- **Online Learning**: Continuous adaptation and strategy optimization
- **Checkpoint & Replay**: Debug, analyze, and retry workflows
- **Temporal Context**: Causal chains, pattern detection, trend analysis
- **Production-Ready**: Battle-tested, scalable, fault-tolerant

### Design Principles

All components follow the **Federation Constitution** (CLAUDE.md):

- **Law of Runtime Truth**: Verify everything with execution
- **Law of Idempotency**: All operations are replay-safe
- **Law of UTC**: All timestamps in UTC
- **Law of Configuration Explicitness**: All settings via environment variables
- **Anti-Corruption Layer**: Canonical schemas for all data

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Long-Horizon Framework                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Workflow    │  │   State      │  │   Temporal      │  │
│  │ Orchestrator  │◄─┤  Manager     │◄─┤   Context       │  │
│  └───────┬───────┘  └──────┬───────┘  └─────────────────┘  │
│          │                  │                                  │
│          │                  │                                  │
│  ┌───────▼──────────────────▼───────┐  ┌─────────────────┐  │
│  │      Checkpoint & Replay          │  │   Learning      │  │
│  │           System                   │  │   Engine        │  │
│  └────────────────────────────────────┘  └─────────────────┘  │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Storage Backends                          │
│  ┌──────────────┐           ┌──────────────┐               │
│  │   MongoDB    │           │    Neo4j     │               │
│  │  (Documents) │           │  (Graph)     │               │
│  └──────────────┘           └──────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **State Manager** | Persistent storage, versioning, compression |
| **Workflow Orchestrator** | Execution, scheduling, resumption |
| **Temporal Context** | Time-aware reasoning, patterns, trends |
| **Learning Engine** | Strategy optimization, adaptation |
| **Checkpoint Manager** | Automatic checkpointing, cleanup |
| **Replay Engine** | Rollback, debugging, analysis |

### State Machine Diagrams

#### Workflow Execution States

```
┌─────────┐
│ PENDING │
└────┬────┘
     │
     ▼
┌─────────┐
│ RUNNING │◄────────┐
└────┬────┘         │
     │              │
     ▼              │ Paused
┌─────────┐         │
│ WAITING │─────────┘
└────┬────┘
     │
     ├───────────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│COMPLETED│ │  FAILED │
└─────────┘ └─────────┘
```

#### Checkpoint Lifecycle

```
Snapshot Created
       │
       ▼
Validate Integrity
       │
       ├─ Invalid → Discard
       │
       ▼
   Valid
       │
       ▼
Create Checkpoint
       │
       ├─────────────┐
       ▼             ▼
   Store        Index for Search
       │             │
       └──────┬──────┘
              ▼
         Ready for Replay
```

---

## Installation

### Prerequisites

- Python 3.9+
- MongoDB 4.4+
- Neo4j 4.4+

### Install Dependencies

```bash
# Core dependencies
pip install pymongo neo4j structlog pydantic numpy scipy

# Optional: For specific use cases
pip install redis  # For distributed state
pip import celery  # For distributed task queue
```

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# Required
MONGODB_URL=mongodb://localhost:27017/openevolve
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Optional - State Management
STATE_COMPRESSION_ENABLED=true
STATE_MAX_VERSIONS=1000

# Optional - Workflow
WORKFLOW_TIMEOUT_DEFAULT=3600
WORKFLOW_MAX_RETRIES=3
WORKFLOW_HEARTBEAT_INTERVAL=30
WORKFLOW_PARALLEL_WORKERS=4

# Optional - Learning
LEARNING_EXPLORATION_RATE=0.1
LEARNING_EXPLORATION_DECAY=0.995
LEARNING_MIN_EXPLORATION_RATE=0.01

# Optional - Checkpoint
CHECKPOINT_AUTO_MILESTONES=start,complete,error
CHECKPOINT_MAX_PER_WORKFLOW=50
CHECKPOINT_INTERVAL_SECONDS=300
```

### Docker Setup (Recommended)

```yaml
# docker-compose.yml
version: '3.8'

services:
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  neo4j:
    image: neo4j:5.0
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/your_password
    volumes:
      - neo4j_data:/data

volumes:
  mongodb_data:
  neo4j_data:
```

Start services:

```bash
docker-compose up -d
```

---

## Quick Start

### Basic Usage

```python
import asyncio
from openvolve.long_horizon import create_framework

async def main():
    # Create framework
    framework = create_framework()

    # Access components
    state_manager = framework['state_manager']
    orchestrator = framework['workflow_orchestrator']
    learning_engine = framework['learning_engine']
    temporal_context = framework['temporal_context']

    # Save state
    snapshot = await state_manager.save_snapshot(
        state_data={'counter': 0, 'status': 'initialized'},
        level='session',
        workflow_id='my_workflow',
        is_checkpoint=True,
        checkpoint_name='start'
    )

    # Create workflow
    await orchestrator.create_workflow(
        workflow_id='my_workflow',
        name='My Long-Running Workflow',
        description='Processes data continuously',
        steps=[
            {'step_id': 'fetch', 'type': 'data_fetch', 'source': 'api'},
            {'step_id': 'process', 'type': 'data_process', 'algorithm': 'ml'},
            {'step_id': 'store', 'type': 'data_store', 'destination': 'db'}
        ],
        schedule_type='cron',
        schedule_expression='0 */6 * * *',  # Every 6 hours
        timeout_seconds=3600
    )

    # Start execution
    execution = await orchestrator.start_workflow(
        workflow_id='my_workflow',
        input_parameters={'batch_size': 100}
    )

    print(f"Workflow started: {execution.execution_id}")

if __name__ == '__main__':
    asyncio.run(main())
```

### Learning and Adaptation

```python
# Record learning outcomes
await learning_engine.record_outcome(
    workflow_id='my_workflow',
    execution_id=execution.execution_id,
    lesson_type='success',
    lesson_description='Batch size of 100 optimal',
    success=True,
    performance_score=0.95,
    strategy_used='batch_100',
    parameters={'batch_size': 100},
    learned_by='my_agent'
)

# Select optimal strategy (with exploration)
strategies = ['batch_50', 'batch_100', 'batch_200']
selected = await learning_engine.select_strategy(
    available_strategies=strategies,
    exploration_strategy=ExplorationStrategy.EPSILON_GREEDY
)

print(f"Selected strategy: {selected}")
```

### Temporal Context

```python
from datetime import datetime, timezone, timedelta

# Add temporal event
event = TemporalEvent(
    event_id='event_1',
    event_type='data_processed',
    timestamp=datetime.now(timezone.utc),
    event_data={'records': 1000, 'duration_ms': 500},
    source='my_agent',
    importance=0.8
)

await temporal_context.add_event(event)

# Analyze trends
time_window = TimeWindow(
    window_id='last_24h',
    start_time=datetime.now(timezone.utc) - timedelta(hours=24),
    end_time=datetime.now(timezone.utc)
)

trend = await temporal_context.analyze_trend(
    metric_name='records',
    time_window=time_window
)

print(f"Trend: {trend.trend_type}")
print(f"Slope: {trend.slope}")
```

### Checkpoint and Replay

```python
# Create checkpoint at milestone
await state_manager.create_checkpoint(
    snapshot_id=snapshot.snapshot_id,
    checkpoint_name='after_data_processing',
    checkpoint_type='milestone',
    workflow_id='my_workflow',
    created_by='my_agent',
    description='Completed data processing phase'
)

# Later: replay from checkpoint
replay_session = await replay_engine.start_replay(
    checkpoint_id='checkpoint_id',
    replay_reason='Debug performance issue',
    replay_type='debug',
    replayed_by='developer'
)
```

---

## Core Components

### State Manager

Manages persistent state with git-like versioning.

**Key Features:**
- Multi-level state (session, workflow, agent, global)
- Delta compression for efficiency
- Version history with branching
- Automatic cleanup

**Example:**

```python
# Save state
snapshot = await state_manager.save_snapshot(
    state_data={'model_weights': [0.1, 0.2, 0.3]},
    level='workflow',
    workflow_id='training_workflow',
    parent_snapshot_id='previous_version'
)

# Load state
loaded = await state_manager.load_snapshot(snapshot.snapshot_id)

# Get history
history = await state_manager.get_history(snapshot.snapshot_id)
for h in history:
    print(f"Version {h.version}: {h.created_at}")
```

### Workflow Orchestrator

Orchestrates long-running workflow execution.

**Key Features:**
- Time-aware scheduling (cron-like)
- Human-in-the-loop support
- Checkpoint-based resumption
- Dependency management

**Example:**

```python
# Create workflow
await orchestrator.create_workflow(
    workflow_id='daily_report',
    name='Daily Analytics Report',
    description='Generates daily analytics report',
    steps=[
        {'step_id': 'gather', 'type': 'data_gathering'},
        {'step_id': 'analyze', 'type': 'analysis', 'is_checkpoint': True},
        {'step_id': 'report', 'type': 'reporting'}
    ],
    schedule_type='cron',
    schedule_expression='0 9 * * *',  # 9 AM daily
    human_handoff_points=['report']
)

# Register step handler
async def analysis_handler(workflow_def, execution, step):
    # Perform analysis
    return {'metrics': {'accuracy': 0.95}}

orchestrator.register_step_handler('analysis', analysis_handler)

# Start workflow
execution = await orchestrator.start_workflow(
    workflow_id='daily_report'
)
```

### Temporal Context Manager

Manages time-aware reasoning and context.

**Key Features:**
- Temporal event tracking
- Causal relationship graphs
- Pattern detection
- Trend analysis

**Example:**

```python
# Add event with causal link
event1 = TemporalEvent(
    event_id='model_updated',
    event_type='model_change',
    timestamp=datetime.now(timezone.utc),
    event_data={'version': '2.0'},
    source='system'
)

event2 = TemporalEvent(
    event_id='accuracy_improved',
    event_type='performance_change',
    timestamp=datetime.now(timezone.utc) + timedelta(minutes=5),
    event_data={'accuracy': 0.95},
    source='system'
)

await temporal_context.add_event(event1)
await temporal_context.add_event(event2)

# Link events causally
link = CausalLink(
    link_id='link1',
    cause_event_id='model_updated',
    effect_event_id='accuracy_improved',
    causal_type='direct',
    strength=0.8
)

await temporal_context.add_causal_link(link)

# Detect patterns
patterns = await temporal_context.detect_patterns(
    event_type='performance_change',
    time_window=TimeWindow(
        window_id='analysis_window',
        start_time=datetime.now(timezone.utc) - timedelta(days=7),
        end_time=datetime.now(timezone.utc)
    )
)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Period: {pattern.period_seconds}s")
```

### Learning Engine

Online learning and strategy adaptation.

**Key Features:**
- Strategy performance tracking
- Exploration vs exploitation
- A/B testing
- Adaptive learning rates

**Example:**

```python
# Record outcomes
for i in range(100):
    success = await execute_strategy(strategy='strategy_a')
    await learning_engine.record_outcome(
        workflow_id='my_workflow',
        execution_id=f'exec_{i}',
        lesson_type='success' if success else 'failure',
        lesson_description='Execution result',
        success=success,
        performance_score=0.9 if success else 0.3,
        strategy_used='strategy_a',
        parameters={},
        learned_by='my_agent'
    )

# Run A/B test
ab_result = await learning_engine.run_ab_test(
    test_name='Strategy A vs B',
    hypothesis='Strategy B performs better',
    control_strategy='strategy_a',
    treatment_strategy='strategy_b',
    test_context={'environment': 'production'}
)

print(f"Winner: {ab_result.recommended_strategy}")
print(f"Confidence: {ab_result.recommendation_confidence}")
```

### Checkpoint Manager

Automatic checkpoint creation and lifecycle management.

**Key Features:**
- Automatic milestone checkpointing
- Retention policies
- Integrity validation
- Compression support

**Example:**

```python
# Checkpoint at milestone
metadata = await checkpoint_manager.create_checkpoint(
    snapshot_id=snapshot.snapshot_id,
    checkpoint_name='epoch_10',
    checkpoint_type='milestone',
    workflow_id='training_workflow',
    created_by='trainer',
    description='Completed training epoch 10',
    validate=True  # Validate integrity
)

# Check if should checkpoint
should_checkpoint = await checkpoint_manager.should_create_checkpoint(
    workflow_id='training_workflow',
    step_number=10,
    milestone_type='epoch_complete'
)

# Cleanup old checkpoints
deleted = await checkpoint_manager.cleanup_old_checkpoints(
    workflow_id='training_workflow',
    keep_count=50
)
```

### Replay Engine

Rollback and replay for debugging and analysis.

**Key Features:**
- Rollback to checkpoints
- Replay with modifications
- Execution comparison
- Branching from checkpoints

**Example:**

```python
# Start replay session
replay = await replay_engine.start_replay(
    checkpoint_id='checkpoint_epoch_10',
    replay_reason='Investigate overfitting',
    replay_type='debug',
    replayed_by='developer',
    modifications=[
        {'parameter': 'learning_rate', 'value': 0.001}
    ]
)

# Execute replay
results = await replay_engine.execute_replay(
    replay_id=replay.replay_id,
    workflow_orchestrator=orchestrator,
    modifications=replay.modifications
)

# Compare with original
print(f"Original accuracy: {results['original']['accuracy']}")
print(f"Replay accuracy: {results['replay']['accuracy']}")

# Rollback to checkpoint
restored = await replay_engine.rollback_to_checkpoint(
    checkpoint_id='checkpoint_epoch_10'
)
```

---

## API Reference

### StateManager

```python
class StateManager:
    async def save_snapshot(
        self,
        state_data: Dict[str, Any],
        level: StateLevel,
        workflow_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_snapshot_id: Optional[str] = None,
        is_checkpoint: bool = False,
        checkpoint_name: Optional[str] = None,
        created_by: str = "system"
    ) -> StateSnapshot

    async def load_snapshot(
        self,
        snapshot_id: str,
        decompress: bool = True
    ) -> StateSnapshot

    async def create_checkpoint(
        self,
        snapshot_id: str,
        checkpoint_name: str,
        checkpoint_type: str,
        workflow_id: str,
        created_by: str,
        description: str
    ) -> StateCheckpoint

    async def get_history(
        self,
        snapshot_id: str,
        max_depth: int = 100
    ) -> List[StateSnapshot]
```

### WorkflowOrchestrator

```python
class WorkflowOrchestrator:
    async def create_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        dependencies: Optional[List[str]] = None,
        schedule_type: str = "manual",
        schedule_expression: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        human_handoff_points: Optional[List[str]] = None,
        created_by: str = "system"
    ) -> WorkflowDefinition

    async def start_workflow(
        self,
        workflow_id: str,
        input_parameters: Optional[Dict[str, Any]] = None,
        execution_agent: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> WorkflowExecution

    async def pause_workflow(self, execution_id: str) -> None
    async def resume_workflow(self, execution_id: str) -> None
    async def cancel_workflow(self, execution_id: str) -> None

    def register_step_handler(
        self,
        step_type: str,
        handler: Callable
    ) -> None
```

### TemporalContextManager

```python
class TemporalContextManager:
    async def add_event(self, event: TemporalEvent) -> None
    async def add_causal_link(self, link: CausalLink) -> None

    async def get_events(
        self,
        time_window: TimeWindow,
        event_types: Optional[List[str]] = None,
        importance_threshold: float = 0.0,
        workflow_id: Optional[str] = None
    ) -> List[TemporalEvent]

    async def detect_patterns(
        self,
        event_type: str,
        time_window: TimeWindow
    ) -> List[TemporalPattern]

    async def analyze_trend(
        self,
        metric_name: str,
        time_window: TimeWindow,
        workflow_id: Optional[str] = None
    ) -> TrendAnalysis
```

### LearningEngine

```python
class LearningEngine:
    async def record_outcome(
        self,
        workflow_id: str,
        execution_id: str,
        lesson_type: str,
        lesson_description: str,
        success: bool,
        performance_score: float,
        strategy_used: str,
        parameters: Dict[str, Any],
        environmental_factors: Optional[Dict[str, Any]] = None,
        causal_factors: Optional[List[str]] = None,
        learned_by: str = "system"
    ) -> LearningOutcome

    async def select_strategy(
        self,
        available_strategies: List[str],
        context: Optional[Dict[str, Any]] = None,
        exploration_strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY
    ) -> str

    async def run_ab_test(
        self,
        test_name: str,
        hypothesis: str,
        control_strategy: str,
        treatment_strategy: str,
        test_context: Dict[str, Any],
        sample_size: Optional[int] = None
    ) -> ABTestResult
```

---

## Best Practices

### 1. State Management

**DO:**
- Use appropriate state levels (session → workflow → agent → global)
- Create checkpoints at milestones
- Enable compression for large states
- Clean up old states regularly

**DON'T:**
- Store massive binary blobs in state (use references instead)
- Create checkpoints too frequently (performance impact)
- Ignore version chains (they're useful for debugging)

### 2. Workflow Design

**DO:**
- Break long workflows into logical steps
- Add human-in-the-loop points for critical decisions
- Set appropriate timeouts
- Use checkpoints for long-running steps

**DON'T:**
- Create monolithic workflows (hard to debug)
- Ignore error handling (always have retry logic)
- Hardcode values (use parameters)

### 3. Learning and Adaptation

**DO:**
- Record detailed outcomes (context matters)
- Use A/B testing for significant changes
- Monitor exploration rate decay
- Regularize to prevent overfitting

**DON'T:**
- Overfit to recent data (maintain long-term view)
- Ignore statistical significance
- Change strategies too frequently

### 4. Temporal Context

**DO:**
- Record all events with accurate timestamps
- Establish causal links when possible
- Analyze trends regularly
- Use time windows appropriately

**DON'T:**
- Mix timezones (always use UTC)
- Ignore causal relationships
- Create too many events (performance)

### 5. Checkpoint Strategy

**DO:**
- Checkpoint before risky operations
- Validate checkpoint integrity
- Use descriptive checkpoint names
- Implement retention policies

**DON'T:**
- Keep all checkpoints forever (storage costs)
- Skip validation (corrupted checkpoints are useless)
- Create checkpoints mid-transaction (consistency issues)

---

## Troubleshooting

### Common Issues

#### 1. Connection Errors

**Problem:** Cannot connect to MongoDB/Neo4j

**Solution:**
```python
# Verify environment variables
import os
print(f"MONGODB_URL: {os.getenv('MONGODB_URL')}")
print(f"NEO4J_URL: {os.getenv('NEO4J_URL')}")

# Test connections
from pymongo import MongoClient
from neo4j import GraphDatabase

# Test MongoDB
client = MongoClient(os.getenv('MONGODB_URL'))
client.admin.command('ping')  # Should succeed

# Test Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URL'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
)
driver.verify_connectivity()  # Should succeed
```

#### 2. Checkpoint Corruption

**Problem:** Checkpoint validation fails

**Solution:**
```python
# Check integrity report
integrity = await validator.validate_checkpoint(snapshot, state_manager)
print(f"Errors: {integrity.validation_errors}")

# Common causes:
# 1. Incomplete write (check logs)
# 2. Disk full (check disk space)
# 3. Concurrent modification (use locks)

# If corrupted, rollback to parent
parent = await state_manager.load_snapshot(snapshot.parent_snapshot_id)
```

#### 3. Memory Issues

**Problem:** Out of memory with large states

**Solution:**
```python
# Enable compression
snapshot = await state_manager.save_snapshot(
    state_data=large_data,
    level='workflow',
    workflow_id='workflow_id',
    is_compressed=True  # Enable compression
)

# Or store references instead of actual data
snapshot = await state_manager.save_snapshot(
    state_data={
        'data_ref': 's3://bucket/large_data.parquet',
        'metadata': {'rows': 1000000, 'columns': 100}
    },
    level='workflow',
    workflow_id='workflow_id'
)
```

#### 4. Workflow Stuck

**Problem:** Workflow status stuck at RUNNING

**Solution:**
```python
# Check heartbeat
execution = await get_execution(execution_id)
if (datetime.now(timezone.utc) - execution.last_heartbeat).total_seconds() > 300:
    # No heartbeat for 5 minutes - likely dead
    await orchestrator.cancel_workflow(execution_id)

# Or pause and resume
await orchestrator.pause_workflow(execution_id)
# Investigate
await orchestrator.resume_workflow(execution_id)
```

#### 5. Learning Not Converging

**Problem:** Strategy performance not improving

**Solution:**
```python
# Check exploration rate
print(f"Exploration rate: {learning_engine._exploration_rate}")

# If too low, reset
learning_engine._exploration_rate = 0.1

# Check if enough data
summary = await learning_engine.get_learning_summary()
print(f"Total outcomes: {summary['total_outcomes']}")

# If insufficient, continue exploring
# Or check if strategies are actually different
```

---

## Advanced Topics

### Distributed Execution

For distributed workflows across multiple machines:

```python
# Use Redis for distributed state
from openvolve.long_horizon import RedisStateManager

state_manager = RedisStateManager(
    redis_url='redis://localhost:6379/0'
)

# Use Celery for distributed task queue
from celery import Celery

app = Celery('workflows', broker='redis://localhost:6379/0')

@app.task
def execute_workflow_step(workflow_id, step_id, state):
    # Execute step
    result = process_step(state)
    return result
```

### Custom Exploration Strategies

```python
from openvolve.long_horizon.learning_engine import ExplorationStrategy

# Implement custom strategy
class BayesianExploration(ExplorationStrategy):
    def __init__(self):
        super().__init__("bayesian")

    def select(self, strategies, performances):
        # Implement Bayesian optimization
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor

        # Fit GP to performances
        gp = GaussianProcessRegressor()
        # ... fit and predict

        return best_strategy

# Use custom strategy
selected = await learning_engine.select_strategy(
    available_strategies=strategies,
    exploration_strategy=BayesianExploration()
)
```

### Temporal Causal Inference

```python
# Discover causal relationships automatically
from openvolve.long_horizon.temporal_context import CausalInferenceEngine

causal_engine = CausalInferenceEngine()

# Learn causal structure
causal_graph = await causal_engine.learn_structure(
    events=events,
    method='pc'  # Peter-Clark algorithm
)

# Visualize graph
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
for link in causal_graph.links:
    G.add_edge(link.cause, link.effect)

nx.draw(G, with_labels=True)
plt.show()
```

### Multi-Agent Coordination

```python
# Coordinate multiple agents with shared state
framework = create_framework()

# Agent 1: Data collection
agent1_state = await framework['state_manager'].save_snapshot(
    state_data={'collected': []},
    level='agent',
    agent_id='collector'
)

# Agent 2: Processing
agent2_state = await framework['state_manager'].save_snapshot(
    state_data={'processed': []},
    level='agent',
    agent_id='processor'
)

# Shared workflow coordinates both
await orchestrator.create_workflow(
    workflow_id='multi_agent_pipeline',
    name='Multi-Agent Pipeline',
    steps=[
        {'step_id': 'collect', 'agent': 'collector'},
        {'step_id': 'process', 'agent': 'processor'}
    ]
)
```

---

## Performance Tuning

### State Management

- **Compression Ratio**: Monitor and tune compression threshold
- **Batch Size**: Save states in batches for better throughput
- **Indexing**: Add MongoDB indexes for frequently queried fields

### Workflow Execution

- **Parallel Workers**: Adjust based on CPU cores
- **Heartbeat Interval**: Balance between responsiveness and overhead
- **Checkpoint Frequency**: Tradeoff between safety and performance

### Learning Engine

- **Exploration Rate**: Start high, decay slowly
- **Sample Size**: Ensure statistical power for A/B tests
- **Memory Limit**: Prune old outcomes periodically

---

## Monitoring and Observability

### Structured Logging

All components use structured logging:

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "workflow_started",
    workflow_id='my_workflow',
    execution_id='exec_123',
    estimated_duration=3600
)
```

### Metrics to Track

- State save/load latency
- Workflow execution duration
- Checkpoint size and frequency
- Learning convergence rate
- Strategy distribution
- Temporal event rate

### Example Dashboard (Grafana)

```promql
# Workflow success rate
sum(rate(workflow_completed_total{status="success"}[5m])) /
sum(rate(workflow_completed_total[5m]))

# Average checkpoint size
avg(state_size_bytes{type="checkpoint"})

# Learning rate over time
avg(exploration_rate)
```

---

## Security Considerations

1. **State Encryption**: Encrypt sensitive state data at rest
2. **Access Control**: Implement role-based access control
3. **Audit Trail**: Log all state changes and checkpoint access
4. **Credential Management**: Use secrets management (HashiCorp Vault, AWS Secrets Manager)
5. **Network Security**: Use TLS for database connections

---

## Contributing

When contributing to the framework:

1. Follow CLAUDE.md principles
2. Add tests for new features
3. Update documentation
4. Ensure all timestamps are UTC
5. Make operations idempotent
6. Use canonical schemas

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues, questions, or contributions:
- GitHub: [repository URL]
- Documentation: [docs URL]
- Discord/Slack: [community URL]

---

**Last Updated**: January 30, 2026
**Version**: 1.0.0
