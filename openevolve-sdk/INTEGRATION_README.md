# OpenEvolve-BubbleLab Integration

## Overview

The OpenEvolve-BubbleLab Integration provides a complete plugin architecture and workflow orchestration system for the OpenEvolve platform. It enables seamless integration between BubbleLab plugins (RAGBits, Datapizza) and OpenEvolve's core capabilities.

## Features

### ✅ Completed Features

- **Plugin Registry** (`plugin-registry.ts`)
  - Centralized plugin lifecycle management
  - Circuit breaker protection per plugin
  - Health monitoring and automatic recovery
  - Plugin discovery and capability querying

- **Workflow Orchestrator** (`workflow-orchestrator.ts`)
  - Multi-step workflow execution
  - Dependency-aware step ordering (topological sort)
  - Parallel and sequential execution support
  - Conditional step execution
  - Error handling strategies (stop, continue, retry)
  - Input variable resolution and output mapping

- **Workflow Templates** (`workflow-templates.ts`)
  - Research Assistant: Search, analyze, and summarize
  - Data Analysis Pipeline: ETL and analytics
  - Proof Verification: Multi-prover formal verification
  - Knowledge Extraction: Structured data extraction
  - Problem Solving: ROMA-powered problem analysis

- **Event Bus Integration** (`plugin-events.ts`)
  - Cross-plugin event communication
  - Event subscriptions and publishing
  - Pre-configured cross-plugin handlers
  - Event correlation and tracing

- **Monitoring & Telemetry** (`workflow-monitoring.ts`)
  - Workflow execution metrics
  - Step-level performance tracking
  - Aggregate statistics and percentiles
  - Error tracking and summarization
  - LocalStorage persistence

- **UI Components**
  - WorkflowExecutionTab: Complete workflow UI
  - BubbleLabsIntegrationTab: Updated with workflow section
  - Plugin configuration panels

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   OpenEvolve Application                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Plugin Registry (Central Hub)             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │   │
│  │  │  RAGBits     │  │  Datapizza   │  │  Custom    │ │   │
│  │  │  Plugin      │  │  Plugin      │  │  Plugins   │ │   │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Workflow Orchestrator                       │   │
│  │  - Execute workflows across plugins                 │   │
│  │  - Handle dependencies and parallelism              │   │
│  │  - Retry logic and error handling                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Workflow Templates                          │   │
│  │  - Research Assistant                               │   │
│  │  - Data Analysis Pipeline                           │   │
│  │  - Proof Verification                               │   │
│  │  - Knowledge Extraction                             │   │
│  │  - Problem Solving                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Event Bus Integration                       │   │
│  │  - Cross-plugin communication                       │   │
│  │  - Event subscriptions                              │   │
│  │  - Automatic event handlers                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Monitoring & Telemetry                     │   │
│  │  - Performance metrics                              │   │
│  │  - Error tracking                                   │   │
│  │  - Aggregate statistics                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Initialize the Integration

```typescript
import { initializeBubbleLabIntegration } from '@/lib/plugin-integration';

// Initialize with configuration
await initializeBubbleLabIntegration({
  ragbits: {
    serverUrl: 'http://localhost:3000/ragbits',
    apiKey: process.env.RAGBITS_API_KEY,
    enabled: true
  },
  datapizza: {
    serverUrl: 'http://localhost:3000/datapizza',
    apiKey: process.env.DATAPIZZA_API_KEY,
    enabled: true
  },
  autoStart: true
});
```

### 2. Execute a Workflow

```typescript
import { getWorkflowOrchestrator } from '@/lib/workflow-orchestrator';
import { RESEARCH_ASSISTANT_WORKFLOW } from '@/lib/workflow-templates';

const orchestrator = getWorkflowOrchestrator();

const result = await orchestrator.executeWorkflow(
  RESEARCH_ASSISTANT_WORKFLOW,
  {
    query: 'machine learning algorithms for NLP'
  }
);

console.log('Results:', result.results);
```

### 3. Monitor Performance

```typescript
import { getWorkflowMonitor } from '@/lib/workflow-monitoring';

const monitor = getWorkflowMonitor();
const stats = monitor.getAggregateStats();
console.log('Workflow statistics:', stats);
```

## File Structure

```
bubblelab-converted/src/
├── lib/
│   ├── plugin-registry.ts          # Plugin management system
│   ├── workflow-orchestrator.ts    # Workflow execution engine
│   ├── workflow-templates.ts       # Pre-built workflow templates
│   ├── workflow-monitoring.ts      # Telemetry and monitoring
│   ├── plugin-events.ts            # Event bus integration
│   └── plugin-integration.ts       # Main integration module
├── components/openevolve/main/
│   ├── BubbleLabsIntegrationTab.tsx  # Plugin management UI
│   ├── WorkflowExecutionTab.tsx      # Workflow executor UI
│   └── OpenEvolveApp.tsx             # Main app (updated)
└── tests/contract/
    └── workflow-orchestrator.test.ts # Contract tests
```

## Usage Examples

### Execute a Custom Workflow

```typescript
import type { WorkflowDefinition } from '@/lib/workflow-orchestrator';

const workflow: WorkflowDefinition = {
  id: 'my-workflow',
  name: 'My Custom Workflow',
  steps: [
    {
      id: 'search',
      name: 'Search Knowledge',
      plugin: 'ragbits',
      action: 'search',
      input: { query: '$query', topK: 10 },
      outputMapping: { results: 'search_results' }
    },
    {
      id: 'analyze',
      name: 'Analyze Results',
      plugin: 'datapizza',
      action: 'processData',
      input: { data: '$search_results' },
      dependsOn: ['search']
    }
  ]
};

const orchestrator = getWorkflowOrchestrator();
const result = await orchestrator.executeWorkflow(workflow, {
  query: 'your query'
});
```

### Create a Custom Plugin

```typescript
import { getPluginRegistry } from '@/lib/plugin-registry';
import type { PluginInterface } from '@/lib/plugin-registry';

class MyPlugin implements PluginInterface {
  metadata = {
    name: 'my-plugin',
    version: '1.0.0',
    description: 'My custom plugin',
    author: 'Me',
    enabled: true
  };

  capabilities = { processing: true };

  async initialize() { /* ... */ }
  // ... implement other required methods
}

const registry = getPluginRegistry();
await registry.registerPlugin(new MyPlugin());
```

### Subscribe to Events

```typescript
import { getPluginEventIntegration } from '@/lib/plugin-events';

const events = getPluginEventIntegration();
events.subscribePlugin(plugin, ['workflow.completed'], async (event) => {
  console.log('Workflow completed:', event.data);
});
```

## Workflow Templates

### Research Assistant
Searches knowledge base, analyzes results, and generates insights.

```typescript
import { RESEARCH_ASSISTANT_WORKFLOW } from '@/lib/workflow-templates';
await orchestrator.executeWorkflow(RESEARCH_ASSISTANT_WORKFLOW, {
  query: 'blockchain scalability'
});
```

### Data Analysis Pipeline
ETL pipeline for processing and analyzing data.

```typescript
import { DATA_ANALYSIS_PIPELINE } from '@/lib/workflow-templates';
await orchestrator.executeWorkflow(DATA_ANALYSIS_PIPELINE, {
  data_source: 's3://bucket/data.csv',
  analysis_query: 'Calculate statistics'
});
```

### Proof Verification
Multi-prover formal verification system.

```typescript
import { PROOF_VERIFICATION_WORKFLOW } from '@/lib/workflow-templates';
await orchestrator.executeWorkflow(PROOF_VERIFICATION_WORKFLOW, {
  theorem: 'forall x. x > 0 => x + 1 > 0'
});
```

## Testing

Run contract tests:

```bash
npm run test:contract
```

Run specific test suite:

```bash
npm test -- workflow-orchestrator
```

## Federation Constitution Compliance

This implementation follows all principles from the Federation Constitution:

### ✅ Law of Air Gap (Source Code Isolation)
- Plugins are isolated modules with no direct core project imports
- Each plugin implements the `PluginInterface` contract

### ✅ Law of Runtime Truth (Anti-Hallucination)
- All plugin capabilities verified at runtime via `healthCheck()`
- Plugin discovery through capability queries, not documentation

### ✅ Law of Configuration Explicitness
- All configuration via environment variables or explicit parameters
- No magic defaults - system fails loudly if config missing

### ✅ Law of Idempotency
- Workflow executions are idempotent
- Plugin operations can be safely retried

### ✅ Circuit Breaker Protection
- Each plugin has dedicated circuit breaker
- Prevents cascading failures across plugins

### ✅ UTC Timestamps
- All timestamps in UTC ISO-8601 format
- Consistent timezone handling across system

## Monitoring

### View Workflow Statistics

```typescript
const monitor = getWorkflowMonitor();
const stats = monitor.getAggregateStats();
console.log(stats);
// {
//   totalExecutions: 150,
//   successfulExecutions: 142,
//   failedExecutions: 8,
//   averageDuration: 2341,
//   errorRate: 0.053
// }
```

### View Performance Percentiles

```typescript
const percentiles = monitor.getPerformancePercentiles();
console.log(percentiles);
// { p50: 1500, p90: 3500, p95: 4200, p99: 5800 }
```

### Export Metrics

```typescript
const json = monitor.exportMetrics();
console.log(json);
```

## Documentation

- [WORKFLOWS.md](./WORKFLOWS.md) - Complete workflow guide with examples
- [Plugin Interface](./lib/plugin-registry.ts) - Plugin API reference
- [Workflow Templates](./lib/workflow-templates.ts) - Available templates

## API Reference

### PluginRegistry
- `registerPlugin(plugin)` - Register a new plugin
- `getPlugin(name)` - Get plugin by name
- `getPluginsByCapability(capability)` - Filter plugins by capability
- `healthCheckAll()` - Check health of all plugins
- `getStatistics()` - Get registry statistics

### WorkflowOrchestrator
- `executeWorkflow(workflow, input)` - Execute a workflow
- `validateWorkflow(workflow)` - Validate workflow definition
- `cancelWorkflow(executionId)` - Cancel running workflow
- `getActiveWorkflows()` - Get active executions

### WorkflowMonitor
- `getWorkflowMetrics(executionId)` - Get metrics for execution
- `getAggregateStats()` - Get aggregate statistics
- `getPerformancePercentiles()` - Get performance percentiles
- `exportMetrics()` - Export all metrics as JSON

## Error Handling

Workflows support three error handling strategies:

1. **stop**: Stop immediately on error (default)
2. **continue**: Continue with remaining steps
3. **retry**: Retry failed steps up to maxRetries

```typescript
const workflow: WorkflowDefinition = {
  id: 'retry-workflow',
  steps: [
    {
      id: 'step1',
      retryOnFailure: true,
      // ...
    }
  ],
  onError: 'retry',
  maxRetries: 3
};
```

## Performance Optimization

- Use `dependsOn` to enable parallel execution
- Set appropriate timeouts to prevent hanging
- Use circuit breakers to prevent cascading failures
- Monitor performance metrics and optimize slow steps

## Future Enhancements

Potential future improvements:

- Workflow visual editor
- Workflow versioning and rollback
- Workflow scheduling and cron jobs
- Plugin hot-reloading
- Distributed workflow execution
- Workflow debugging and tracing UI
- Custom plugin marketplace

## Support

For issues, questions, or contributions, please contact the OpenEvolve team.

## License

MIT License - See LICENSE file for details
