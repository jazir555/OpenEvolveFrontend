# Ragbits + BubbleLab Integration

The Ragbits + BubbleLab Integration provides a powerful visual interface for creating, managing, and executing Retrieval-Augmented Generation (RAG) workflows. This integration combines the advanced RAG capabilities of Ragbits with the intuitive workflow builder of BubbleLab.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The Ragbits + BubbleLab Integration enables users to:
- Visually configure RAG workflows using BubbleLab's drag-and-drop interface
- Leverage Ragbits' document processing and search capabilities
- Monitor and debug RAG workflows through BubbleLab's observability features
- Export RAG workflows as production-ready configurations

### Key Features
- Visual RAG workflow configuration
- Real-time monitoring and debugging
- Scalable deployment options
- Intuitive user experience

## Architecture

The integration follows a modular architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BubbleLab UI  │◄──►│  Integration    │◄──►│   Ragbits Core  │
│   (Workflow     │    │     Layer       │    │   (RAG Engine)  │
│   Builder)      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  RAG Workflows  │    │  Configuration  │    │  RAG Components │
│  (Visual Flow)  │    │  Management     │    │  (Indexing,    │
│                 │    │                 │    │   Retrieval,    │
│                 │    │                 │    │   Generation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Integration Points
1. **BubbleLab Plugin System**: Uses existing plugin architecture
2. **Ragbits Document Processor**: Integrates with existing processor
3. **Component Library**: Adds RAG-specific Bubble components
4. **Monitoring Interface**: Provides real-time metrics and debugging

## Installation

To install the Ragbits + BubbleLab Integration:

```bash
npm install @openevolve/ragbits-bubblelab-integration
```

### Prerequisites
- Node.js 16+
- Ragbits framework
- BubbleLab workflow builder

## Quick Start

Here's a simple example to get started with the integration:

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

// Define a simple RAG workflow
const workflowConfig: BubbleLabWorkflowConfig = {
  id: 'simple-rag-workflow',
  name: 'Simple RAG Workflow',
  description: 'A basic RAG workflow for demonstration',
  nodes: [
    {
      id: 'ingest-node',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './documents/',
        chunkSize: 1000,
        chunkOverlap: 200
      }
    },
    {
      id: 'search-node',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 5,
        scoreThreshold: 0.7
      }
    },
    {
      id: 'generation-node',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.7,
        maxTokens: 1000
      }
    }
  ],
  edges: [
    {
      id: 'edge-1',
      source: 'ingest-node',
      target: 'search-node'
    },
    {
      id: 'edge-2',
      source: 'search-node',
      target: 'generation-node'
    }
  ],
  metadata: {}
};

// Get the integration instance
const integration = RagbitsBubbleLabIntegration.getInstance();

// Generate a Ragbits configuration from the BubbleLab workflow
const generatedConfig = integration.generateConfig(workflowConfig);

// Create and execute the workflow
const workflowEngine = integration.createWorkflowEngine(workflowConfig);
await workflowEngine.initialize();

const results = await workflowEngine.executeWorkflow({
  query: 'What is the capital of France?'
});

console.log('Workflow results:', results);
```

## Components

### Bubble Components

The integration provides several specialized Bubble components for RAG workflows:

#### RAGBitsIngestBubble
Handles document ingestion into the RAG system.

**Configuration Options:**
- `sourceType`: 'file' | 'url' | 'text'
- `sourcePath`: Path to the document source
- `metadata`: Additional metadata for the document
- `chunkSize`: Size of text chunks (default: 1000)
- `chunkOverlap`: Overlap between chunks (default: 200)

#### RAGBitsSearchBubble
Performs semantic search on indexed documents.

**Configuration Options:**
- `topK`: Number of results to return (default: 5)
- `scoreThreshold`: Minimum relevance score (default: 0.0)
- `enableHybridSearch`: Enable hybrid search (default: false)
- `defaultFilters`: Default filters for search

#### RAGBitsGenerationBubble
Generates responses using retrieved context.

**Configuration Options:**
- `llmModel`: LLM model to use (default: 'gpt-4o')
- `temperature`: Temperature for generation (default: 0.7)
- `maxTokens`: Maximum tokens in response (default: 1000)
- `systemPrompt`: System prompt for the LLM

#### RAGBitsIndexBubble
Manages the vector index for document storage.

**Configuration Options:**
- `vectorStoreType`: 'memory' | 'qdrant' (default: 'memory')
- `embeddingModel`: Embedding model to use (default: 'text-embedding-3-small')
- `autoRefresh`: Enable auto-refresh (default: false)
- `refreshInterval`: Refresh interval in seconds (default: 300)

## API Reference

### RagbitsBubbleLabIntegration

The main integration class that provides access to all functionality.

#### getInstance()
Returns the singleton instance of the integration.

```typescript
const integration = RagbitsBubbleLabIntegration.getInstance();
```

#### createWorkflowEngine(workflowConfig, options?)
Creates a new workflow engine instance.

```typescript
const engine = integration.createWorkflowEngine(workflowConfig, {
  timeout: 30000,
  maxRetries: 3
});
```

#### createProcessorIntegration(config?)
Creates a new processor integration instance.

```typescript
const processor = integration.createProcessorIntegration({
  enableAutoIndexing: true,
  batchSize: 10
});
```

#### createMonitoringService(config?)
Creates a new monitoring service instance.

```typescript
const monitoring = integration.createMonitoringService({
  enableRealTimeMonitoring: true,
  logLevel: 'info'
});
```

#### generateConfig(bubbleLabConfig, options?)
Generates a Ragbits configuration from a BubbleLab workflow.

```typescript
const config = integration.generateConfig(workflowConfig, {
  validate: true,
  targetEnvironment: 'production'
});
```

#### mapConfig(bubbleLabConfig)
Maps a BubbleLab workflow to a Ragbits configuration.

```typescript
const config = integration.mapConfig(workflowConfig);
```

### RAGBitsWorkflowEngine

Executes RAG workflows defined in BubbleLab.

#### initialize()
Initializes the workflow engine by creating instances of all nodes.

#### executeWorkflow(initialInput?)
Executes the entire workflow with optional initial input.

#### getExecutionHistory()
Returns the execution history of the workflow.

#### dispose()
Disposes of resources used by the engine.

### RagbitsProcessorIntegration

Integrates with the Ragbits document processor.

#### initialize()
Initializes the processor integration.

#### processDocument(source, content, metadata?)
Processes a single document.

#### addDocument(source, content, metadata?)
Adds a document to the processing queue.

#### search(query, topK?, filters?, minScore?)
Searches for documents using the Ragbits processor.

#### getStats()
Returns statistics about document processing.

#### clearStore()
Clears the vector store.

### MonitoringService

Provides real-time monitoring and debugging capabilities.

#### logWorkflowStart(workflowId, metadata?)
Logs a workflow start event.

#### logWorkflowComplete(workflowId, duration, metadata?)
Logs a workflow completion event.

#### logNodeStart(workflowId, nodeId, metadata?)
Logs a node start event.

#### logNodeComplete(workflowId, nodeId, duration, metadata?)
Logs a node completion event.

#### getPerformanceMetrics()
Returns performance metrics.

#### getWorkflowStats()
Returns workflow statistics.

## Configuration

### Environment-Specific Settings

The configuration generator supports different target environments:

- `development`: Optimized for development with memory storage
- `staging`: Balanced settings for staging environments
- `production`: Optimized for production with Qdrant storage

### Configuration Validation

All generated configurations are validated to ensure they meet requirements:

- Document processor settings are valid
- Search parameters are within acceptable ranges
- Generation parameters are properly configured
- Workflow structure is valid

## Monitoring

The integration provides comprehensive monitoring capabilities:

- Real-time workflow execution visualization
- Performance metrics dashboard
- Debugging tools and logs viewer
- Component state monitoring

### Alerting System

The monitoring service includes an alerting system that triggers when:

- Execution time exceeds threshold
- Error rate exceeds threshold
- Memory usage exceeds threshold

## Examples

### Basic RAG Workflow

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

const basicWorkflow: BubbleLabWorkflowConfig = {
  id: 'basic-rag',
  name: 'Basic RAG Workflow',
  description: 'A simple RAG workflow',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './docs/',
        chunkSize: 1000,
        chunkOverlap: 200
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
const engine = integration.createWorkflowEngine(basicWorkflow);
await engine.initialize();

const results = await engine.executeWorkflow({
  query: 'Summarize the key points in the documents'
});
```

### Advanced RAG Workflow with Monitoring

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

const advancedWorkflow: BubbleLabWorkflowConfig = {
  id: 'advanced-rag',
  name: 'Advanced RAG Workflow',
  description: 'A complex RAG workflow with monitoring',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './docs/',
        chunkSize: 500,
        chunkOverlap: 100
      }
    },
    {
      id: 'index',
      type: 'ragbits-index',
      position: { x: 200, y: 0 },
      data: {
        vectorStoreType: 'qdrant',
        embeddingModel: 'text-embedding-3-large'
      }
    },
    {
      id: 'search',
      type: 'ragbits-search',
      position: { x: 400, y: 0 },
      data: {
        topK: 10,
        scoreThreshold: 0.8,
        enableHybridSearch: true
      }
    },
    {
      id: 'generate',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4-turbo',
        temperature: 0.3,
        maxTokens: 2000
      }
    }
  ],
  edges: [
    { source: 'ingest', target: 'index' },
    { source: 'index', target: 'search' },
    { source: 'search', target: 'generate' }
  ],
  metadata: {}
};

const integration = RagbitsBubbleLabIntegration.getInstance();

// Set up monitoring
const monitoring = integration.createMonitoringService({
  enableRealTimeMonitoring: true,
  enablePerformanceTracking: true
});

// Create and execute workflow
const engine = integration.createWorkflowEngine(advancedWorkflow);
await engine.initialize();

// Add monitoring to workflow execution
monitoring.addEventListener((event) => {
  console.log(`[${event.eventType}] ${event.nodeId || event.workflowId}`);
});

const results = await engine.executeWorkflow({
  query: 'Provide a detailed analysis of the documents'
});

// Get performance metrics
const metrics = monitoring.getPerformanceMetrics();
console.log('Performance metrics:', metrics);
```

## Troubleshooting

### Common Issues

#### Workflow Execution Timeout
If workflows are timing out, try increasing the timeout:

```typescript
const engine = integration.createWorkflowEngine(workflowConfig, {
  timeout: 60000 // 60 seconds instead of default 30
});
```

#### Memory Issues
For large document sets, consider using Qdrant instead of memory storage:

```typescript
const workflowWithQdrant = {
  // ... other config
  nodes: [
    // ... other nodes
    {
      id: 'index',
      type: 'ragbits-index',
      data: {
        vectorStoreType: 'qdrant', // Use Qdrant instead of memory
        // ... other index config
      }
    }
  ]
};
```

#### Search Quality Issues
Adjust search parameters for better results:

```typescript
{
  id: 'search',
  type: 'ragbits-search',
  data: {
    topK: 10, // Return more results
    scoreThreshold: 0.5, // Lower threshold for more lenient matching
    enableHybridSearch: true // Enable hybrid search
  }
}
```

### Debugging Tips

Enable detailed logging to troubleshoot issues:

```typescript
const engine = integration.createWorkflowEngine(workflowConfig, {
  enableLogging: true,
  logLevel: 'debug'
});
```

Monitor specific nodes for debugging:

```typescript
const monitoring = integration.createMonitoringService({
  enableRealTimeMonitoring: true,
  logLevel: 'debug'
});

monitoring.addEventListener((event) => {
  if (event.eventType.includes('error')) {
    console.error('Error in workflow:', event);
  }
});
```

## Contributing

See the [contributing guidelines](CONTRIBUTING.md) for information on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.