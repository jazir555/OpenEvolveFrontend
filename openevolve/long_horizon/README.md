# Long-Horizon Agentic Framework

> Production-ready infrastructure for AI agents that maintain state, learn, and operate across days, weeks, and months.

## Overview

The Long-Horizon Agentic Framework provides the foundational infrastructure for building persistent, stateful AI agents that can execute complex workflows over extended time periods. It enables agents to:

- **Maintain state** across days, weeks, and months with git-like versioning
- **Learn and adapt** strategies based on outcomes
- **Understand time** through temporal reasoning and causal chains
- **Recover gracefully** with checkpointing and rollback
- **Orchestrate workflows** with time-aware scheduling

## Key Features

### Persistent State Management
- Multi-level state hierarchy (session → workflow → agent → global)
- Git-like versioning with branching and merging
- Delta compression for efficient storage
- Automatic checkpoint creation at milestones
- Rollback to any previous state

### Time-Aware Orchestration
- Cron-like scheduling with smart dependencies
- Human-in-the-loop handoff points
- Resume from checkpoints after failures
- Workflow state machine (pending, running, paused, completed, failed)
- Distributed execution support

### Temporal Reasoning
- Event tracking with causal relationships
- Pattern detection (periodic, sequential, trends)
- Trend analysis with anomaly detection
- Time-aware context retrieval
- Recurring event recognition

### Online Learning
- Strategy performance tracking
- Exploration vs exploitation balance (ε-greedy, UCB, Thompson sampling)
- A/B testing framework with statistical validation
- Adaptation actions with rollback capability
- Meta-learning across workflow instances

### Checkpoint & Replay
- Automatic checkpoint creation
- Integrity validation (SHA-256 checksums)
- Rollback for debugging and recovery
- Replay with modifications
- Execution comparison and analysis

## Quick Start

### Installation

```bash
# Install dependencies
pip install pymongo neo4j structlog pydantic numpy scipy

# Set environment variables
export MONGODB_URL="mongodb://localhost:27017/openevolve"
export NEO4J_URL="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
```

### Basic Usage

```python
import asyncio
from openvolve.long_horizon import create_framework

async def main():
    # Create framework with all components
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
        workflow_id='my_workflow'
    )

    # Create and start workflow
    await orchestrator.create_workflow(
        workflow_id='my_workflow',
        name='My Workflow',
        description='A long-running workflow',
        steps=[
            {'step_id': 'step1', 'type': 'process', 'action': 'analyze'},
            {'step_id': 'step2', 'type': 'output', 'action': 'report'}
        ]
    )

    execution = await orchestrator.start_workflow(
        workflow_id='my_workflow'
    )

    print(f"Workflow started: {execution.execution_id}")

if __name__ == '__main__':
    asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Long-Horizon Framework                   │
├─────────────────────────────────────────────────────────────┤
│  Workflow Orchestrator  │  State Manager  │  Learning Engine │
│  Temporal Context       │  Checkpoint      │  Replay System  │
├─────────────────────────────────────────────────────────────┤
│                    Storage Backends                          │
│  MongoDB (Documents)    │    Neo4j (Graph)                  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### StateManager
Persistent state storage with versioning, compression, and checkpointing.

### WorkflowOrchestrator
Time-aware workflow execution with scheduling, dependencies, and human-in-the-loop support.

### TemporalContextManager
Time-aware reasoning with event tracking, causal chains, and pattern detection.

### LearningEngine
Online learning with strategy optimization, A/B testing, and adaptation.

### CheckpointManager
Automatic checkpoint creation with integrity validation and cleanup.

### ReplayEngine
Rollback and replay for debugging, analysis, and retry.

## Documentation

Comprehensive documentation is available in `docs/long_horizon/framework.md`:

- Architecture overview
- API reference
- Best practices
- Troubleshooting guide
- Advanced topics
- Performance tuning

## Testing

Run the comprehensive test suite:

```bash
pytest tests/long_horizon/test_framework.py -v
```

Test coverage includes:
- State persistence and recovery
- Workflow execution and resumption
- Temporal context accuracy
- Learning convergence
- Checkpoint integrity
- Concurrent execution safety

## Design Principles

All components follow the **Federation Constitution** (CLAUDE.md):

- **Law of Runtime Truth**: Verify everything with execution, not documentation
- **Law of Idempotency**: All operations are replay-safe
- **Law of UTC**: All timestamps in UTC ISO-8601 format
- **Law of Configuration Explicitness**: All settings via environment variables
- **Anti-Corruption Layer**: Canonical schemas for all data representations

## Configuration

Required environment variables:
```bash
MONGODB_URL=mongodb://localhost:27017/openevolve
NEO4J_URL=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

Optional configuration:
```bash
# State Management
STATE_COMPRESSION_ENABLED=true
STATE_MAX_VERSIONS=1000

# Workflow
WORKFLOW_TIMEOUT_DEFAULT=3600
WORKFLOW_MAX_RETRIES=3
WORKFLOW_PARALLEL_WORKERS=4

# Learning
LEARNING_EXPLORATION_RATE=0.1
LEARNING_EXPLORATION_DECAY=0.995

# Checkpoint
CHECKPOINT_MAX_PER_WORKFLOW=50
CHECKPOINT_INTERVAL_SECONDS=300
```

## Integration with OpenEvolve

This framework integrates seamlessly with:
- **OpenEvolve Evolution API**: For evolutionary code generation
- **Knowledge Engine**: For persistent learning
- **LoongFlow PES**: For Plan-Execute-Summarize workflows

## Production Considerations

### Monitoring
Track these metrics:
- State save/load latency
- Workflow execution duration
- Checkpoint size and frequency
- Learning convergence rate
- Strategy distribution

### Security
- Encrypt state data at rest
- Use TLS for database connections
- Implement role-based access control
- Audit all state changes

### Performance Tuning
- Enable compression for large states
- Adjust checkpoint frequency based on workflow duration
- Tune exploration rate for learning convergence
- Use appropriate state levels to minimize overhead

## Examples

See `docs/long_horizon/framework.md` for detailed examples:
- Basic workflow execution
- Learning and adaptation
- Temporal context management
- Checkpoint and replay
- A/B testing
- Multi-agent coordination

## License

MIT License

## Support

For issues, questions, or contributions, please refer to the main OpenEvolve repository.

---

**Version**: 1.0.0
**Last Updated**: January 30, 2026
**Total Lines of Code**: ~4,350
**Components**: 6 core modules + canonical schemas
