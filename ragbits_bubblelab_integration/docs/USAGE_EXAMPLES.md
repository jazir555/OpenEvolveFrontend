# Usage Examples: Ragbits + BubbleLab Integration

This document provides practical examples of how to use the Ragbits + BubbleLab Integration for various RAG (Retrieval-Augmented Generation) scenarios.

## Table of Contents
- [Basic Setup](#basic-setup)
- [Simple RAG Workflow](#simple-rag-workflow)
- [Advanced RAG Workflow](#advanced-rag-workflow)
- [Document Processing](#document-processing)
- [Search and Retrieval](#search-and-retrieval)
- [Response Generation](#response-generation)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Configuration Management](#configuration-management)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)

## Basic Setup

### Initializing the Integration

```typescript
import { RagbitsBubbleLabIntegration } from '@openevolve/ragbits-bubblelab-integration';

// Get the singleton instance
const integration = RagbitsBubbleLabIntegration.getInstance();
```

### Creating a Simple Workflow Configuration

```typescript
import { type BubbleLabWorkflowConfig } from '@openevolve/ragbits-bubblelab-integration';

const simpleWorkflow: BubbleLabWorkflowConfig = {
  id: 'simple-rag-workflow',
  name: 'Simple RAG Workflow',
  description: 'A basic RAG workflow for getting started',
  nodes: [
    {
      id: 'ingest-node',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './documents/',
        chunkSize: 1000,
        chunkOverlap: 200,
        metadata: { source: 'user-uploaded' }
      }
    },
    {
      id: 'search-node',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 5,
        scoreThreshold: 0.7,
        enableHybridSearch: false
      }
    },
    {
      id: 'generation-node',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.7,
        maxTokens: 1000,
        systemPrompt: 'You are a helpful assistant that answers questions based on the provided context.'
      }
    }
  ],
  edges: [
    { source: 'ingest-node', target: 'search-node' },
    { source: 'search-node', target: 'generation-node' }
  ],
  metadata: {}
};
```

## Simple RAG Workflow

### Creating and Executing a Basic Workflow

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

// Define a simple RAG workflow
const basicWorkflow: BubbleLabWorkflowConfig = {
  id: 'basic-rag',
  name: 'Basic RAG Workflow',
  description: 'A simple RAG workflow for demonstration',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'text',
        sourcePath: 'inline',
        chunkSize: 1000,
        chunkOverlap: 200
      }
    },
    {
      id: 'search',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 3,
        scoreThreshold: 0.5
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

// Get integration instance
const integration = RagbitsBubbleLabIntegration.getInstance();

// Create and initialize workflow engine
const engine = integration.createWorkflowEngine(basicWorkflow);
await engine.initialize();

// Execute the workflow
const results = await engine.executeWorkflow({
  query: 'What is the meaning of life?',
  context: 'Philosophical concepts'
});

console.log('Workflow results:', results);

// Clean up
await engine.dispose();
```

## Advanced RAG Workflow

### Complex Workflow with Multiple Document Sources

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

const advancedWorkflow: BubbleLabWorkflowConfig = {
  id: 'advanced-rag-workflow',
  name: 'Advanced RAG Workflow',
  description: 'A complex RAG workflow with multiple document sources and conditional logic',
  nodes: [
    // Multiple ingestion nodes for different sources
    {
      id: 'pdf-ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './pdfs/',
        fileType: 'pdf',
        chunkSize: 800,
        chunkOverlap: 100
      }
    },
    {
      id: 'web-ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 200 },
      data: {
        sourceType: 'url',
        sourcePath: 'https://example.com/docs',
        chunkSize: 1000,
        chunkOverlap: 200
      }
    },
    // Index management node
    {
      id: 'index-manager',
      type: 'ragbits-index',
      position: { x: 200, y: 100 },
      data: {
        vectorStoreType: 'memory',
        embeddingModel: 'text-embedding-3-small',
        autoRefresh: true,
        refreshInterval: 300
      }
    },
    // Multiple search strategies
    {
      id: 'semantic-search',
      type: 'ragbits-search',
      position: { x: 400, y: 0 },
      data: {
        topK: 5,
        scoreThreshold: 0.7,
        enableHybridSearch: true
      }
    },
    {
      id: 'keyword-search',
      type: 'ragbits-search',
      position: { x: 400, y: 200 },
      data: {
        topK: 3,
        scoreThreshold: 0.5,
        enableHybridSearch: false
      }
    },
    // Result aggregator
    {
      id: 'result-aggregator',
      type: 'ragbits-generation', // Using generation node for aggregation
      position: { x: 600, y: 100 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.3,
        maxTokens: 1500,
        systemPrompt: 'Aggregate and synthesize information from multiple sources. Provide a comprehensive answer based on all available information.'
      }
    }
  ],
  edges: [
    { source: 'pdf-ingest', target: 'index-manager' },
    { source: 'web-ingest', target: 'index-manager' },
    { source: 'index-manager', target: 'semantic-search' },
    { source: 'index-manager', target: 'keyword-search' },
    { source: 'semantic-search', target: 'result-aggregator' },
    { source: 'keyword-search', target: 'result-aggregator' }
  ],
  metadata: {
    complexity: 'high',
    useCase: 'multi-source-research'
  }
};

const integration = RagbitsBubbleLabIntegration.getInstance();
const engine = integration.createWorkflowEngine(advancedWorkflow);
await engine.initialize();

const results = await engine.executeWorkflow({
  query: 'Compare and contrast the approaches to artificial intelligence in the provided documents'
});

console.log('Advanced workflow results:', results);
await engine.dispose();
```

## Document Processing

### Processing Documents with the Integration

```typescript
import { RagbitsProcessorIntegration } from '@openevolve/ragbits-bubblelab-integration';

// Create and initialize processor integration
const processor = new RagbitsProcessorIntegration();
await processor.initialize();

// Process a single document
const docResult = await processor.processDocument(
  './research-paper.pdf',
  'Content of the research paper...',
  { 
    title: 'Research Paper Title',
    author: 'Author Name',
    year: 2023,
    category: 'academic'
  }
);

console.log('Document processing result:', docResult);

// Process multiple documents
const documents = [
  { source: './doc1.txt', content: 'Content 1...', metadata: { type: 'report' } },
  { source: './doc2.txt', content: 'Content 2...', metadata: { type: 'manual' } },
  { source: './doc3.txt', content: 'Content 3...', metadata: { type: 'guide' } }
];

for (const doc of documents) {
  const result = await processor.processDocument(doc.source, doc.content, doc.metadata);
  console.log(`Processed ${doc.source}:`, result.success ? 'SUCCESS' : 'FAILED');
}

// Search in processed documents
const searchResults = await processor.search('artificial intelligence', 10, { type: 'report' });
console.log('Search results:', searchResults);

// Get processing statistics
const stats = processor.getStats();
console.log('Processing statistics:', stats);

await processor.dispose();
```

### Batch Document Processing

```typescript
import { RagbitsProcessorIntegration } from '@openevolve/ragbits-bubblelab-integration';

const processor = new RagbitsProcessorIntegration({
  enableAutoIndexing: true,
  batchSize: 5,
  enableCaching: true
});

await processor.initialize();

// Add multiple documents to processing queue
const processingPromises = [];
for (let i = 0; i < 20; i++) {
  processingPromises.push(
    processor.addDocument(
      `./document-${i}.txt`,
      `Content of document ${i} for batch processing. This is sample content for testing batch processing capabilities.`,
      { batchId: 'batch-001', index: i }
    )
  );
}

// Wait for all documents to be added to queue
const addedResults = await Promise.all(processingPromises);
console.log(`Added ${addedResults.length} documents to processing queue`);

// Process the queue
await processor.processQueue();

// Check queue status
console.log(`Queue size: ${processor.getQueueSize()}`);
console.log(`Processing stats:`, processor.getStats());

await processor.dispose();
```

## Search and Retrieval

### Performing Semantic Search

```typescript
import { RagbitsProcessorIntegration } from '@openevolve/ragbits-bubblelab-integration';

const processor = new RagbitsProcessorIntegration();
await processor.initialize();

// Process a document first
await processor.processDocument(
  './technical-manual.txt',
  'This document contains technical specifications for the AI system. It covers architecture, implementation details, and performance characteristics.',
  { category: 'technical', type: 'manual' }
);

// Perform semantic search
const searchResults = await processor.search(
  'AI system architecture',  // Query
  5,                         // topK
  { category: 'technical' }, // Filters
  0.6                        // Minimum score
);

console.log('Search results:', searchResults);

// Perform multiple searches
const queries = [
  'system architecture',
  'implementation details',
  'performance characteristics'
];

for (const query of queries) {
  const results = await processor.search(query, 3, { type: 'manual' });
  console.log(`Results for "${query}":`, results);
}

await processor.dispose();
```

### Advanced Search with Filtering

```typescript
import { RagbitsProcessorIntegration } from '@openevolve/ragbits-bubblelab-integration';

const processor = new RagbitsProcessorIntegration();
await processor.initialize();

// Process documents with various metadata
const docs = [
  { 
    source: './finance-report-q1.txt', 
    content: 'Financial report for Q1 2023...', 
    metadata: { category: 'finance', quarter: 'Q1', year: 2023 } 
  },
  { 
    source: './finance-report-q2.txt', 
    content: 'Financial report for Q2 2023...', 
    metadata: { category: 'finance', quarter: 'Q2', year: 2023 } 
  },
  { 
    source: './tech-specs-v1.txt', 
    content: 'Technical specifications v1...', 
    metadata: { category: 'technology', version: '1.0', type: 'specification' } 
  }
];

for (const doc of docs) {
  await processor.processDocument(doc.source, doc.content, doc.metadata);
}

// Search with complex filters
const financeResults = await processor.search(
  'revenue growth',
  10,
  { category: 'finance', year: 2023 },
  0.5
);

const techResults = await processor.search(
  'system requirements',
  5,
  { category: 'technology', type: 'specification' },
  0.7
);

console.log('Finance results:', financeResults);
console.log('Technology results:', techResults);

await processor.dispose();
```

## Response Generation

### Generating Responses with Context

```typescript
import { RagbitsProcessorIntegration } from '@openevolve/ragbits-bubblelab-integration';

const processor = new RagbitsProcessorIntegration();
await processor.initialize();

// Process some documents
await processor.processDocument(
  './company-handbook.txt',
  'Our company values include innovation, collaboration, and customer focus. We strive to create products that solve real problems for our customers.',
  { department: 'hr', type: 'handbook' }
);

// Search for relevant information
const searchResults = await processor.search('company values', 3);
console.log('Search results:', searchResults);

// Generate a response based on the search results
const context = searchResults.map(result => ({
  content: result.content,
  metadata: result.metadata
}));

// Simulate generation (in a real implementation, this would connect to an LLM)
const response = `Based on the company handbook, our core values are: ${context.map(c => c.content).join(' ')}`;

console.log('Generated response:', response);

await processor.dispose();
```

## Monitoring and Debugging

### Setting Up Monitoring

```typescript
import {
  RagbitsBubbleLabIntegration,
  MonitoringService
} from '@openevolve/ragbits-bubblelab-integration';

// Create monitoring service
const monitoring = new MonitoringService({
  enableRealTimeMonitoring: true,
  enablePerformanceTracking: true,
  enableErrorTracking: true,
  logLevel: 'info'
});

// Add event listener
monitoring.addEventListener((event) => {
  console.log(`[${event.eventType}] ${event.nodeId || event.workflowId} - ${event.duration || 'N/A'}ms`);
});

// Add alert callback
monitoring.addAlertCallback((alert) => {
  console.warn(`MONITORING ALERT: ${alert}`);
});

// Create workflow with monitoring
const integration = RagbitsBubbleLabIntegration.getInstance();
const workflowConfig = {
  id: 'monitored-workflow',
  name: 'Monitored Workflow',
  description: 'Workflow with monitoring',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: { sourceType: 'text', sourcePath: 'inline' }
    }
  ],
  edges: [],
  metadata: {}
};

const engine = integration.createWorkflowEngine(workflowConfig);
await engine.initialize();

// Execute workflow
const results = await engine.executeWorkflow({ content: 'Test content for monitoring' });

// Get performance metrics
const metrics = monitoring.getPerformanceMetrics();
console.log('Performance metrics:', metrics);

// Get workflow statistics
const stats = monitoring.getWorkflowStats();
console.log('Workflow stats:', stats);

await engine.dispose();
```

### Advanced Monitoring with Custom Metrics

```typescript
import { MonitoringService } from '@openevolve/ragbits-bubblelab-integration';

const monitoring = new MonitoringService({
  enableRealTimeMonitoring: true,
  enablePerformanceTracking: true,
  enableTokenTracking: true,
  logLevel: 'debug'
});

// Track custom metrics
let totalQueries = 0;
let successfulQueries = 0;

monitoring.addEventListener((event) => {
  if (event.eventType === 'node_complete' && event.nodeId.includes('search')) {
    totalQueries++;
    if (!event.error) {
      successfulQueries++;
    }
  }
  
  // Log detailed information for debugging
  if (event.eventType.includes('error')) {
    console.error(`Error in ${event.nodeId || event.workflowId}:`, event.error);
  }
});

// Simulate workflow execution
monitoring.logWorkflowStart('analytics-workflow');
monitoring.logNodeStart('analytics-workflow', 'ingest-node');

// Simulate processing time
await new Promise(resolve => setTimeout(resolve, 500));

monitoring.logNodeComplete('analytics-workflow', 'ingest-node', 500);
monitoring.logWorkflowComplete('analytics-workflow', 600);

// Get detailed metrics
const allMetrics = monitoring.getPerformanceMetrics();
const workflowEvents = monitoring.getWorkflowEvents('analytics-workflow');
const nodeEvents = monitoring.getNodeEvents('ingest-node');

console.log('All metrics:', allMetrics);
console.log('Workflow events:', workflowEvents);
console.log('Node events:', nodeEvents);
console.log(`Query success rate: ${(successfulQueries / totalQueries * 100).toFixed(2)}%`);
```

## Configuration Management

### Generating Ragbits Configuration

```typescript
import {
  RagbitsBubbleLabIntegration,
  ConfigGenerator
} from '@openevolve/ragbits-bubblelab-integration';

const integration = RagbitsBubbleLabIntegration.getInstance();

// Define a BubbleLab workflow
const bubbleLabWorkflow = {
  id: 'config-generation-test',
  name: 'Config Generation Test',
  description: 'Workflow for testing config generation',
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
    }
  ],
  edges: [
    { source: 'ingest', target: 'search' }
  ],
  metadata: {}
};

// Generate Ragbits configuration
const generatedConfig = integration.generateConfig(bubbleLabWorkflow, {
  validate: true,
  format: 'json',
  targetEnvironment: 'production'
});

console.log('Generated Ragbits config:', generatedConfig.ragbitsConfig);

if (generatedConfig.validationErrors) {
  console.error('Validation errors:', generatedConfig.validationErrors);
}

// Generate TypeScript configuration
const tsConfig = ConfigGenerator.formatConfig(generatedConfig, 'typescript');
console.log('TypeScript config:', tsConfig);
```

### Environment-Specific Configuration

```typescript
import { ConfigGenerator } from '@openevolve/ragbits-bubblelab-integration';

const workflowConfig = {
  id: 'env-specific-workflow',
  name: 'Environment Specific Workflow',
  description: 'Workflow with environment-specific settings',
  nodes: [
    {
      id: 'index',
      type: 'ragbits-index',
      position: { x: 0, y: 0 },
      data: {
        vectorStoreType: 'memory', // Will be overridden in production
        embeddingModel: 'text-embedding-3-small'
      }
    }
  ],
  edges: [],
  metadata: {}
};

// Generate configuration for different environments
const devConfig = ConfigGenerator.generate(workflowConfig, {
  targetEnvironment: 'development'
});

const prodConfig = ConfigGenerator.generate(workflowConfig, {
  targetEnvironment: 'production'
});

console.log('Development config vector store:', devConfig.ragbitsConfig.documentProcessor.vector_store_type);
console.log('Production config vector store:', prodConfig.ragbitsConfig.documentProcessor.vector_store_type);

// Development typically uses memory store, production uses Qdrant
console.assert(devConfig.ragbitsConfig.documentProcessor.vector_store_type === 'memory');
console.assert(prodConfig.ragbitsConfig.documentProcessor.vector_store_type === 'qdrant');
```

## Error Handling

### Handling Workflow Execution Errors

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

const errorHandlingWorkflow: BubbleLabWorkflowConfig = {
  id: 'error-handling-workflow',
  name: 'Error Handling Workflow',
  description: 'Workflow designed to test error handling',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './nonexistent-file.txt' // This will cause an error
      }
    }
  ],
  edges: [],
  metadata: {}
};

const integration = RagbitsBubbleLabIntegration.getInstance();
const engine = integration.createWorkflowEngine(errorHandlingWorkflow);

try {
  await engine.initialize();
  
  const results = await engine.executeWorkflow({ content: 'Test content' });
  
  // Check for errors in results
  const errorResults = results.filter(result => !result.success);
  if (errorResults.length > 0) {
    console.warn('Workflow execution had errors:');
    errorResults.forEach(result => {
      console.warn(`  Node ${result.nodeId}: ${result.error}`);
    });
  }
  
  console.log('Execution completed with results:', results);
} catch (error) {
  console.error('Workflow execution failed:', error);
} finally {
  await engine.dispose();
}
```

### Robust Error Handling Pattern

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

async function executeWorkflowWithRetry(
  workflowConfig: BubbleLabWorkflowConfig,
  maxRetries: number = 3
) {
  const integration = RagbitsBubbleLabIntegration.getInstance();
  const engine = integration.createWorkflowEngine(workflowConfig, {
    timeout: 30000,
    maxRetries: 0 // We'll handle retries manually
  });

  try {
    await engine.initialize();

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        console.log(`Attempt ${attempt} of ${maxRetries}`);
        const results = await engine.executeWorkflow({ content: 'Test content' });
        
        // Check if any nodes failed
        const failedNodes = results.filter(result => !result.success);
        if (failedNodes.length === 0) {
          console.log('Workflow executed successfully');
          return results;
        } else {
          console.warn(`Attempt ${attempt} had ${failedNodes.length} failed nodes`);
          if (attempt === maxRetries) {
            console.error('Max retries reached, workflow failed');
            return results;
          }
          // Wait before retrying
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
        }
      } catch (error) {
        console.error(`Attempt ${attempt} failed:`, error);
        if (attempt === maxRetries) {
          throw error;
        }
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
  } finally {
    await engine.dispose();
  }
}

// Example usage
const robustWorkflow: BubbleLabWorkflowConfig = {
  id: 'robust-workflow',
  name: 'Robust Workflow',
  description: 'Workflow with robust error handling',
  nodes: [
    {
      id: 'ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './docs/'
      }
    }
  ],
  edges: [],
  metadata: {}
};

try {
  const results = await executeWorkflowWithRetry(robustWorkflow, 3);
  console.log('Final results:', results);
} catch (error) {
  console.error('Workflow ultimately failed:', error);
}
```

## Performance Optimization

### Optimizing for Large Document Sets

```typescript
import {
  RagbitsProcessorIntegration,
  type ProcessorIntegrationConfig
} from '@openevolve/ragbits-bubblelab-integration';

// Configuration optimized for large document sets
const largeScaleConfig: ProcessorIntegrationConfig = {
  enableAutoIndexing: true,
  autoIndexInterval: 600, // 10 minutes
  batchSize: 20, // Larger batch size for efficiency
  enableCaching: true,
  cacheTTL: 7200, // 2 hour cache TTL
  enableMonitoring: true,
  maxConcurrentProcesses: 10 // More concurrent processes
};

const processor = new RagbitsProcessorIntegration(largeScaleConfig);
await processor.initialize();

// Process documents in batches
const documents = Array.from({ length: 100 }, (_, i) => ({
  source: `./doc-${i.toString().padStart(3, '0')}.txt`,
  content: `Content of document ${i} for performance testing. `.repeat(50),
  metadata: { batch: Math.floor(i / 10), source: 'bulk-import' }
}));

// Add all documents to queue first
const addPromises = documents.map(doc => 
  processor.addDocument(doc.source, doc.content, doc.metadata)
);

await Promise.all(addPromises);
console.log(`Added ${documents.length} documents to processing queue`);

// Process in batches
while (processor.getQueueSize() > 0) {
  await processor.processQueue();
  console.log(`Processed batch, remaining in queue: ${processor.getQueueSize()}`);
  // Small delay to prevent overwhelming the system
  await new Promise(resolve => setTimeout(resolve, 100));
}

// Get final statistics
const stats = processor.getStats();
console.log('Processing completed with stats:', stats);

await processor.dispose();
```

### Optimizing Workflow Execution

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

const optimizedWorkflow: BubbleLabWorkflowConfig = {
  id: 'optimized-workflow',
  name: 'Optimized Workflow',
  description: 'Workflow optimized for performance',
  nodes: [
    {
      id: 'optimized-ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'text',
        sourcePath: 'inline',
        chunkSize: 2000, // Larger chunks for better performance
        chunkOverlap: 200,
        metadata: { optimized: true }
      }
    },
    {
      id: 'optimized-search',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 3, // Fewer results for better performance
        scoreThreshold: 0.6, // Reasonable threshold
        enableHybridSearch: false // Disable if not needed
      }
    }
  ],
  edges: [
    { source: 'optimized-ingest', target: 'optimized-search' }
  ],
  metadata: {}
};

const integration = RagbitsBubbleLabIntegration.getInstance();

// Create engine with performance optimizations
const engine = integration.createWorkflowEngine(optimizedWorkflow, {
  timeout: 60000, // Longer timeout for large operations
  maxRetries: 1, // Reduce retries to fail fast
  enableLogging: false, // Disable logging for performance
  logLevel: 'error' // Only log errors
});

await engine.initialize();

// Measure execution time
const startTime = Date.now();
const results = await engine.executeWorkflow({ 
  content: 'Performance test content for optimization'.repeat(100) 
});
const endTime = Date.now();

console.log(`Workflow executed in ${endTime - startTime}ms with ${results.length} results`);

await engine.dispose();
```

## Real-World Use Cases

### Document Question Answering System

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

// Create a workflow for document question answering
const qaWorkflow: BubbleLabWorkflowConfig = {
  id: 'document-qa-system',
  name: 'Document Question Answering System',
  description: 'A system for answering questions about documents',
  nodes: [
    {
      id: 'document-loader',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './knowledge-base/',
        chunkSize: 1000,
        chunkOverlap: 200,
        metadata: { 
          source: 'knowledge-base',
          processingStage: 'ingestion'
        }
      }
    },
    {
      id: 'vector-indexer',
      type: 'ragbits-index',
      position: { x: 200, y: 0 },
      data: {
        vectorStoreType: 'memory',
        embeddingModel: 'text-embedding-3-small',
        autoRefresh: true,
        refreshInterval: 300
      }
    },
    {
      id: 'semantic-searcher',
      type: 'ragbits-search',
      position: { x: 400, y: 0 },
      data: {
        topK: 8,
        scoreThreshold: 0.75,
        enableHybridSearch: true,
        defaultFilters: { 
          processingStage: 'indexed',
          source: 'knowledge-base'
        }
      }
    },
    {
      id: 'answer-generator',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.3, // Lower temperature for more consistent answers
        maxTokens: 1000,
        systemPrompt: `You are a document question answering assistant. 
        Answer questions based only on the provided context. 
        If the answer is not in the context, say "I don't know based on the provided documents."
        Be concise but complete in your answers.`
      }
    }
  ],
  edges: [
    { source: 'document-loader', target: 'vector-indexer' },
    { source: 'vector-indexer', target: 'semantic-searcher' },
    { source: 'semantic-searcher', target: 'answer-generator' }
  ],
  metadata: {
    useCase: 'document-qa',
    domain: 'enterprise-knowledge',
    createdAt: new Date().toISOString()
  }
};

const integration = RagbitsBubbleLabIntegration.getInstance();

// Create and initialize the system
const qaEngine = integration.createWorkflowEngine(qaWorkflow);
await qaEngine.initialize();

// Example questions to test the system
const questions = [
  'What are the main policies outlined in the documents?',
  'How does the company approach customer service?',
  'What are the key performance indicators mentioned?'
];

for (const question of questions) {
  console.log(`\n❓ Question: ${question}`);
  
  const results = await qaEngine.executeWorkflow({ query: question });
  
  // Find the generation node result
  const generationResult = results.find(r => r.nodeId === 'answer-generator');
  
  if (generationResult && generationResult.success) {
    console.log(`💡 Answer: ${generationResult.output.response || generationResult.output}`);
  } else {
    console.log('❌ Could not generate an answer for this question');
    if (generationResult?.error) {
      console.log(`   Error: ${generationResult.error}`);
    }
  }
}

await qaEngine.dispose();
```

### Content Analysis Pipeline

```typescript
import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '@openevolve/ragbits-bubblelab-integration';

// Create a workflow for content analysis
const analysisWorkflow: BubbleLabWorkflowConfig = {
  id: 'content-analysis-pipeline',
  name: 'Content Analysis Pipeline',
  description: 'Pipeline for analyzing content and extracting insights',
  nodes: [
    {
      id: 'content-ingest',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'text',
        sourcePath: 'inline',
        chunkSize: 500, // Smaller chunks for detailed analysis
        chunkOverlap: 100,
        metadata: { analysisType: 'content-insights' }
      }
    },
    {
      id: 'content-search',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 10,
        scoreThreshold: 0.5,
        enableHybridSearch: true
      }
    },
    {
      id: 'insight-generator',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.5, // Balanced creativity and consistency
        maxTokens: 1500,
        systemPrompt: `You are an analytical assistant. Analyze the provided content and extract key insights including:
        - Main themes and topics
        - Key entities mentioned
        - Important facts and figures
        - Trends or patterns
        - Recommendations or implications
        
        Format your response in a structured way with clear sections.`
      }
    }
  ],
  edges: [
    { source: 'content-ingest', target: 'content-search' },
    { source: 'content-search', target: 'insight-generator' }
  ],
  metadata: {
    useCase: 'content-analysis',
    analysisDepth: 'deep',
    outputFormat: 'structured-insights'
  }
};

const integration = RagbitsBubbleLabIntegration.getInstance();
const analysisEngine = integration.createWorkflowEngine(analysisWorkflow);
await analysisEngine.initialize();

// Analyze sample content
const sampleContent = `
Artificial Intelligence is transforming industries at an unprecedented pace. 
Recent studies show that 78% of enterprises are now implementing some form of AI technology. 
Key areas of impact include automation of routine tasks, enhanced decision-making capabilities, 
and improved customer experiences. However, challenges remain in areas such as data privacy, 
algorithmic bias, and workforce displacement concerns. Organizations that successfully navigate 
these challenges are seeing productivity gains of 20-40%.
`;

const analysisResults = await analysisEngine.executeWorkflow({
  content: sampleContent,
  query: 'Analyze the content and extract key insights'
});

const insightResult = analysisResults.find(r => r.nodeId === 'insight-generator');
if (insightResult && insightResult.success) {
  console.log('Content Analysis Results:');
  console.log(insightResult.output.response || insightResult.output);
}

await analysisEngine.dispose();
```

This comprehensive set of examples demonstrates various ways to use the Ragbits + BubbleLab Integration for different RAG scenarios, from simple workflows to complex, real-world applications.