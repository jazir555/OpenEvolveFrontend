# OpenEvolve-BubbleLab Integration Guide

Complete guide for using the OpenEvolve-BubbleLab plugin integration system with workflow orchestration.

## Overview

The OpenEvolve-BubbleLab integration provides:

- **Plugin Registry**: Centralized management of all plugins
- **Workflow Orchestrator**: Execute complex multi-step workflows
- **Event Bus Integration**: Cross-plugin communication
- **Pre-built Templates**: Common workflow patterns

## Quick Start

### 1. Initialize the Integration

```typescript
import { initializeBubbleLabIntegration } from '@/lib/plugin-integration';

// Initialize with plugin configuration
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

### 2. Execute a Workflow Template

```typescript
import { getWorkflowOrchestrator } from '@/lib/workflow-orchestrator';
import { RESEARCH_ASSISTANT_WORKFLOW } from '@/lib/workflow-templates';

const orchestrator = getWorkflowOrchestrator();

const result = await orchestrator.executeWorkflow(
  RESEARCH_ASSISTANT_WORKFLOW,
  {
    query: 'machine learning algorithms for natural language processing'
  }
);

console.log('Results:', result.results);
```

### 3. Create Custom Workflows

```typescript
import type { WorkflowDefinition } from '@/lib/workflow-orchestrator';

const customWorkflow: WorkflowDefinition = {
  id: 'my-custom-workflow',
  name: 'My Custom Workflow',
  description: 'Does something amazing',
  version: '1.0.0',
  steps: [
    {
      id: 'search',
      name: 'Search Knowledge',
      plugin: 'ragbits',
      action: 'search',
      input: {
        query: '$query',
        topK: 10
      },
      outputMapping: {
        results: 'search_results'
      }
    },
    {
      id: 'process',
      name: 'Process Results',
      plugin: 'datapizza',
      action: 'processData',
      input: {
        data: '$search_results',
        processingType: 'analysis'
      },
      dependsOn: ['search'],
      outputMapping: {
        analysis: 'final_analysis'
      }
    }
  ],
  onError: 'continue',
  maxRetries: 2
};

const orchestrator = getWorkflowOrchestrator();
const result = await orchestrator.executeWorkflow(customWorkflow, {
  query: 'your search query here'
});
```

## Workflow Templates

### Research Assistant

Searches knowledge base, analyzes results, and generates insights.

**Use Cases:**
- Literature review
- Competitive analysis
- Market research

**Input Parameters:**
- `query`: Search query string

**Output:**
- `search_results`: Raw search results
- `categories`: Categorized results
- `final_summary`: Generated summary

```typescript
import { RESEARCH_ASSISTANT_WORKFLOW } from '@/lib/workflow-templates';

const result = await orchestrator.executeWorkflow(
  RESEARCH_ASSISTANT_WORKFLOW,
  { query: 'blockchain scalability solutions' }
);
```

### Data Analysis Pipeline

Ingests data, processes it, and generates analytics.

**Use Cases:**
- ETL workflows
- Data preprocessing
- Analytics pipelines

**Input Parameters:**
- `data_source`: Path or URL to data source
- `analysis_query`: Analysis query to execute

**Output:**
- `processed_data`: Cleaned and processed data
- `analytics_insights`: Statistical analysis results

```typescript
import { DATA_ANALYSIS_PIPELINE } from '@/lib/workflow-templates';

const result = await orchestrator.executeWorkflow(
  DATA_ANALYSIS_PIPELINE,
  {
    data_source: 's3://my-bucket/data.csv',
    analysis_query: 'Calculate average, median, and standard deviation'
  }
);
```

### Proof Verification

Verifies mathematical proofs using multiple provers.

**Use Cases:**
- Formal verification
- Mathematical proofs
- Theorem proving

**Input Parameters:**
- `theorem`: Mathematical theorem to verify

**Output:**
- `z3_verification_result`: Z3 verification output
- `lean_verification_result`: LeanAide verification output
- `cross_validation_result`: Comparison of both results

```typescript
import { PROOF_VERIFICATION_WORKFLOW } from '@/lib/workflow-templates';

const result = await orchestrator.executeWorkflow(
  PROOF_VERIFICATION_WORKFLOW,
  { theorem: 'forall x. x > 0 => x + 1 > 0' }
);
```

### Knowledge Extraction

Extracts structured knowledge from unstructured documents.

**Use Cases:**
- Document processing
- Knowledge base creation
- Information extraction

**Input Parameters:**
- `source_type`: Type of source (file, url, text)
- `source_value`: The source content or location

**Output:**
- `extracted_knowledge`: Structured knowledge entities
- `enriched_knowledge`: Knowledge with added metadata

```typescript
import { KNOWLEDGE_EXTRACTION_WORKFLOW } from '@/lib/workflow-templates';

const result = await orchestrator.executeWorkflow(
  KNOWLEDGE_EXTRACTION_WORKFLOW,
  {
    source_type: 'text',
    source_value: 'The Eiffel Tower is located in Paris, France.'
  }
);
```

### Problem Solving

Analyzes problems using ROMA and generates solutions.

**Use Cases:**
- Complex problem decomposition
- Solution generation
- Decision support

**Input Parameters:**
- `problem`: Problem description

**Output:**
- `problem_decomposition`: Breakdown of problem components
- `similar_solutions`: Solutions from similar problems
- `generated_solution`: AI-generated solution

```typescript
import { PROBLEM_SOLVING_WORKFLOW } from '@/lib/workflow-templates';

const result = await orchestrator.executeWorkflow(
  PROBLEM_SOLVING_WORKFLOW,
  { problem: 'How to reduce database query time by 50%' }
);
```

## Workflow Definition Reference

### Workflow Structure

```typescript
interface WorkflowDefinition {
  id: string;                  // Unique workflow identifier
  name: string;                // Human-readable name
  description?: string;        // Optional description
  version?: string;            // Version string
  steps: WorkflowStep[];       // Array of workflow steps
  onError?: 'stop' | 'continue' | 'retry';  // Error handling
  maxRetries?: number;         // Maximum retry attempts
  timeout?: number;            // Overall timeout in ms
  metadata?: Record<string, unknown>;  // Additional metadata
}
```

### Step Structure

```typescript
interface WorkflowStep {
  id: string;                  // Unique step identifier
  name: string;                // Step name
  description?: string;        // Optional description
  plugin: string;              // Plugin name to execute
  action: string;              // Plugin method to call
  input: Record<string, unknown>;  // Input parameters
  outputMapping?: Record<string, string>;  // Map output to workflow output
  condition?: (context: WorkflowContext) => boolean;  // Conditional execution
  retryOnFailure?: boolean;    // Retry on failure
  timeout?: number;            // Step timeout in ms
  dependsOn?: string[];        // Step IDs that must complete first
}
```

### Input References

Steps can reference inputs from:
- Workflow input: `$parameter_name`
- Previous step outputs: `$step_id.output_field`

```typescript
{
  id: 'step2',
  name: 'Use Previous Output',
  plugin: 'datapizza',
  action: 'processData',
  input: {
    data: '$step1.output'  // References step1's output
  }
}
```

### Error Handling Strategies

**stop**: Stop workflow immediately on error
```typescript
{
  onError: 'stop',
  steps: [...]
}
```

**continue**: Continue executing remaining steps on error
```typescript
{
  onError: 'continue',
  steps: [...]
}
```

**retry**: Retry failed steps up to maxRetries times
```typescript
{
  onError: 'retry',
  maxRetries: 3,
  steps: [
    {
      id: 'step1',
      retryOnFailure: true,  // This step will be retried
      ...
    }
  ]
}
```

### Conditional Execution

Execute steps based on workflow context:

```typescript
{
  id: 'conditional-step',
  name: 'Conditional Step',
  plugin: 'datapizza',
  action: 'processData',
  input: {...},
  condition: (context) => {
    // Only execute if previous step succeeded
    const prevResult = context.stepResults.get('previous-step');
    return prevResult?.success === true;
  }
}
```

### Dependencies

Execute steps in specific order using dependencies:

```typescript
{
  steps: [
    {
      id: 'parallel-1',
      name: 'Parallel Task 1',
      plugin: 'plugin-a',
      action: 'action1',
      input: {}
    },
    {
      id: 'parallel-2',
      name: 'Parallel Task 2',
      plugin: 'plugin-b',
      action: 'action2',
      input: {}
    },
    {
      id: 'final',
      name: 'Final Task',
      plugin: 'plugin-c',
      action: 'action3',
      input: {},
      dependsOn: ['parallel-1', 'parallel-2']  // Waits for both
    }
  ]
}
```

## Plugin API

### Registering Custom Plugins

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

  capabilities = {
    processing: true
  };

  status = 'idle';

  async initialize(config?: Record<string, unknown>): Promise<void> {
    // Initialize plugin
    this.status = 'ready';
  }

  async updateConfig(config: Record<string, unknown>): Promise<void> {
    // Update configuration
  }

  async resetConfig(): Promise<void> {
    // Reset to defaults
  }

  async healthCheck(): Promise<boolean> {
    return this.status === 'ready';
  }

  getContext() {
    return { config: {}, state: {} };
  }

  getStatus() {
    return this.status;
  }

  async destroy(): Promise<void> {
    this.status = 'idle';
  }

  // Custom actions
  async myAction(input: { param: string }): Promise<{ result: string }> {
    return { result: `processed: ${input.param}` };
  }
}

const registry = getPluginRegistry();
const plugin = new MyPlugin();
await registry.registerPlugin(plugin);
```

### Using Plugins in Workflows

```typescript
const workflow: WorkflowDefinition = {
  id: 'use-custom-plugin',
  name: 'Use Custom Plugin',
  steps: [
    {
      id: 'step1',
      name: 'Execute Custom Action',
      plugin: 'my-plugin',  // Uses your custom plugin
      action: 'myAction',   // Calls the method
      input: {
        param: 'hello'
      },
      outputMapping: {
        result: 'my_result'
      }
    }
  ]
};
```

## Event Bus Integration

### Subscribing to Events

```typescript
import { getPluginEventIntegration } from '@/lib/plugin-events';

const eventIntegration = getPluginEventIntegration();

eventIntegration.subscribePlugin(
  plugin,
  ['workflow.completed', 'data.processed'],
  async (event) => {
    console.log('Event received:', event.type, event.data);
  }
);
```

### Publishing Events

```typescript
await eventIntegration.publishEvent(
  plugin,
  'custom.event',
  { data: 'value' }
);
```

### Standard Events

- `plugin.initialized`: Plugin has been initialized
- `plugin.error`: Plugin encountered an error
- `plugin.status.changed`: Plugin status changed
- `workflow.started`: Workflow execution started
- `workflow.completed`: Workflow execution completed
- `workflow.failed`: Workflow execution failed
- `data.processed`: Data has been processed
- `knowledge.indexed`: Knowledge has been indexed
- `search.executed`: Search has been executed

## Monitoring and Observability

### Get Workflow Statistics

```typescript
const orchestrator = getWorkflowOrchestrator();
const activeWorkflows = orchestrator.getActiveWorkflows();
console.log('Active workflows:', activeWorkflows.length);
```

### Get Plugin Statistics

```typescript
import { getPluginRegistry } from '@/lib/plugin-registry';

const registry = getPluginRegistry();
const stats = registry.getStatistics();
console.log('Plugin statistics:', stats);
```

### Health Checks

```typescript
// Check all plugins
const healthResults = await registry.healthCheckAll();
console.log('Plugin health:', healthResults);

// Check specific plugin
const isHealthy = await registry.healthCheck('ragbits');
console.log('RAGBits healthy:', isHealthy);
```

## Best Practices

### 1. Workflow Design

- **Keep steps focused**: Each step should do one thing well
- **Use dependencies**: Explicitly define step dependencies
- **Handle errors**: Choose appropriate error handling strategy
- **Set timeouts**: Prevent workflows from hanging indefinitely

### 2. Error Handling

```typescript
// Good: Explicit error handling
{
  onError: 'retry',
  maxRetries: 3,
  steps: [
    {
      id: 'critical-step',
      retryOnFailure: true,
      timeout: 30000,
      ...
    }
  ]
}

// Good: Graceful degradation
{
  onError: 'continue',
  steps: [
    {
      id: 'optional-step',
      condition: (ctx) => ctx.input.runOptional === true,
      ...
    }
  ]
}
```

### 3. Resource Management

```typescript
// Always cleanup when done
const integration = await initializeBubbleLabIntegration(config);
try {
  // Use the integration
  await orchestrator.executeWorkflow(workflow);
} finally {
  await integration.destroy();
}
```

### 4. Circuit Breaker Protection

The plugin registry includes circuit breaker protection to prevent cascading failures. Plugins that fail repeatedly will be temporarily disabled.

## Troubleshooting

### Workflow Fails Immediately

Check that all required plugins are registered and initialized:

```typescript
const registry = getPluginRegistry();
const stats = registry.getStatistics();
console.log('Total plugins:', stats.totalPlugins);
console.log('Healthy plugins:', stats.healthyPlugins);
```

### Step Not Executing

Verify:
1. Plugin name is correct
2. Action exists on plugin
3. Dependencies are satisfied
4. Condition evaluates to true (if specified)

### Performance Issues

1. Check step execution times
2. Adjust timeouts appropriately
3. Consider parallel execution for independent steps
4. Monitor plugin health status

## API Reference

See generated TypeScript definitions for complete API documentation:

- `PluginInterface`: Interface all plugins must implement
- `WorkflowDefinition`: Workflow structure
- `WorkflowStep`: Step structure
- `WorkflowExecutionResult`: Execution result structure
- `PluginRegistry`: Registry methods
- `WorkflowOrchestrator`: Orchestrator methods

## Examples

See the `/examples` directory for complete working examples:

- `basic-workflow.ts`: Simple workflow example
- `conditional-workflow.ts`: Conditional execution
- `parallel-workflow.ts`: Parallel step execution
- `error-handling.ts`: Error handling strategies
- `custom-plugin.ts`: Creating custom plugins
