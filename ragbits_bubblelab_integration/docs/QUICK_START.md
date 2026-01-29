# Quick Start Guide: Ragbits + BubbleLab Integration

Get up and running with the Ragbits + BubbleLab Integration in minutes!

## Prerequisites

- Node.js 16+
- npm or yarn package manager
- Basic understanding of RAG (Retrieval-Augmented Generation) concepts

## Installation

Install the integration package:

```bash
npm install @openevolve/ragbits-bubblelab-integration
```

Or with yarn:

```bash
yarn add @openevolve/ragbits-bubblelab-integration
```

## Basic Example

Here's a minimal example to create and execute your first RAG workflow:

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

// Define your RAG workflow
const myWorkflow: BubbleLabWorkflowConfig = {
  id: 'my-first-workflow',
  name: 'My First RAG Workflow',
  description: 'A simple RAG workflow example',
  nodes: [
    // Document ingestion node
    {
      id: 'ingest-node',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './documents/', // Path to your documents
        chunkSize: 1000,
        chunkOverlap: 200
      }
    },
    // Semantic search node
    {
      id: 'search-node',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 5, // Return top 5 results
        scoreThreshold: 0.7 // Minimum relevance score
      }
    },
    // Response generation node
    {
      id: 'generation-node',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o', // LLM model to use
        temperature: 0.7, // Creativity level
        maxTokens: 1000 // Max tokens in response
      }
    }
  ],
  edges: [
    // Connect ingestion to search
    { source: 'ingest-node', target: 'search-node' },
    // Connect search to generation
    { source: 'search-node', target: 'generation-node' }
  ],
  metadata: {}
};

// Initialize the integration
const integration = RagbitsBubbleLabIntegration.getInstance();

// Create and initialize the workflow engine
const workflowEngine = integration.createWorkflowEngine(myWorkflow);
await workflowEngine.initialize();

// Execute the workflow
const results = await workflowEngine.executeWorkflow({
  query: 'What are the key points in these documents?'
});

console.log('Workflow results:', results);

// Clean up resources
await workflowEngine.dispose();
```

## Understanding the Components

### Workflow Configuration

The `BubbleLabWorkflowConfig` defines your RAG workflow:

- **nodes**: Individual processing steps in your workflow
- **edges**: Connections between nodes
- **id, name, description**: Metadata about the workflow

### Node Types

The integration provides four main node types:

1. **ragbits-ingest**: Handles document ingestion
2. **ragbits-search**: Performs semantic search
3. **ragbits-generation**: Generates responses
4. **ragbits-index**: Manages vector index

## Advanced Example with Monitoring

For production use, add monitoring to track workflow performance:

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

const monitoredWorkflow: BubbleLabWorkflowConfig = {
  id: 'monitored-workflow',
  name: 'Monitored RAG Workflow',
  description: 'RAG workflow with monitoring',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './docs/',
        chunkSize: 1000
      }
    },
    {
      id: 'search',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 5,
        scoreThreshold: 0.7
      }
    },
    {
      id: 'generate',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.7
      }
    }
  ],
  edges: [
    { source: 'ingest', target: 'search' },
    { source: 'search', target: 'generate' }
  ],
  metadata: {}
};

const integration = RagbitsBubbleLabIntegration.getInstance();

// Create monitoring service
const monitoring = integration.createMonitoringService({
  enableRealTimeMonitoring: true,
  enablePerformanceTracking: true,
  logLevel: 'info'
});

// Add event listener for monitoring
monitoring.addEventListener((event) => {
  console.log(`Event: ${event.eventType} for ${event.nodeId || event.workflowId}`);
});

// Create and run workflow
const engine = integration.createWorkflowEngine(monitoredWorkflow);
await engine.initialize();

const results = await engine.executeWorkflow({
  query: 'Analyze the documents and provide insights'
});

// Get performance metrics
const metrics = monitoring.getPerformanceMetrics();
console.log('Performance metrics:', metrics);

// Clean up
await engine.dispose();
```

## Configuration Options

### Workflow Engine Options

```typescript
const engine = integration.createWorkflowEngine(workflowConfig, {
  timeout: 30000,        // 30 seconds timeout per node
  maxRetries: 3,         // Max retries for failed nodes
  enableLogging: true,   // Enable logging
  logLevel: 'info'       // Log level: debug, info, warn, error
});
```

### Monitoring Options

```typescript
const monitoring = integration.createMonitoringService({
  enableRealTimeMonitoring: true,
  enablePerformanceTracking: true,
  enableErrorTracking: true,
  logLevel: 'info',
  retentionPeriod: 7,    // Days to retain logs
  samplingRate: 1.0,     // Sample all events (0.0 to 1.0)
  enableAlerting: true,
  alertThresholds: {
    executionTime: 30000, // 30 seconds
    errorRate: 5,         // 5%
    memoryUsage: 1024     // 1GB
  }
});
```

## Next Steps

1. **Explore Examples**: Check out the [examples directory](./examples/) for more use cases
2. **Custom Nodes**: Learn how to create custom bubble nodes
3. **Production Deployment**: See the deployment guide for production considerations
4. **API Reference**: Browse the full API documentation

## Troubleshooting

### Common Issues

**Q: My workflow is timing out**
A: Increase the timeout option in the workflow engine:
```typescript
const engine = integration.createWorkflowEngine(workflowConfig, {
  timeout: 60000 // 60 seconds instead of default 30
});
```

**Q: Search results are not relevant**
A: Adjust the search parameters:
```typescript
{
  topK: 10,              // Return more results
  scoreThreshold: 0.5,   // Lower threshold for more results
  enableHybridSearch: true // Enable hybrid search
}
```

**Q: Getting memory errors with large documents**
A: Use Qdrant for vector storage instead of memory:
```typescript
{
  id: 'index',
  type: 'ragbits-index',
  data: {
    vectorStoreType: 'qdrant', // Use Qdrant instead of memory
    embeddingModel: 'text-embedding-3-large'
  }
}
```

Need more help? Check out our [full documentation](./README.md) or join our community forums.