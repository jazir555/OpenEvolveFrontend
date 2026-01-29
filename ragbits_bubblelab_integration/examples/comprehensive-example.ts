/**
 * Comprehensive Example: Ragbits + BubbleLab Integration
 *
 * This example demonstrates a complete RAG workflow with monitoring, error handling,
 * and configuration management.
 */

import {
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig,
  type RAGBitsIngestConfig,
  type RAGBitsSearchConfig,
  type RAGBitsGenerationConfig,
  type RAGBitsIndexConfig
} from '../index';

// Define a comprehensive RAG workflow configuration
const comprehensiveWorkflow: BubbleLabWorkflowConfig = {
  id: 'comprehensive-rag-workflow',
  name: 'Comprehensive RAG Workflow',
  description: 'A complete RAG workflow with all components',
  nodes: [
    // Document ingestion node
    {
      id: 'document-ingestion',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'file',
        sourcePath: './documents/',
        chunkSize: 1000,
        chunkOverlap: 200,
        metadata: {
          source: 'comprehensive-example',
          category: 'documentation'
        }
      } as RAGBitsIngestConfig
    },
    // Index management node
    {
      id: 'index-management',
      type: 'ragbits-index',
      position: { x: 200, y: 0 },
      data: {
        vectorStoreType: 'memory',
        embeddingModel: 'text-embedding-3-small',
        autoRefresh: true,
        refreshInterval: 300 // 5 minutes
      } as RAGBitsIndexConfig
    },
    // Semantic search node
    {
      id: 'semantic-search',
      type: 'ragbits-search',
      position: { x: 400, y: 0 },
      data: {
        topK: 5,
        scoreThreshold: 0.7,
        enableHybridSearch: true,
        defaultFilters: {
          category: 'documentation'
        }
      } as RAGBitsSearchConfig
    },
    // Response generation node
    {
      id: 'response-generation',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o',
        temperature: 0.7,
        maxTokens: 1000,
        systemPrompt: 'You are a helpful assistant that answers questions based on the provided context. Be concise but informative.'
      } as RAGBitsGenerationConfig
    }
  ],
  edges: [
    // Connect ingestion to index
    {
      id: 'ingest-to-index',
      source: 'document-ingestion',
      target: 'index-management'
    },
    // Connect index to search
    {
      id: 'index-to-search',
      source: 'index-management',
      target: 'semantic-search'
    },
    // Connect search to generation
    {
      id: 'search-to-generation',
      source: 'semantic-search',
      target: 'response-generation'
    }
  ],
  metadata: {
    createdBy: 'comprehensive-example',
    createdAt: new Date().toISOString(),
    version: '1.0.0'
  }
};

// Error handling wrapper for async operations
async function safeExecute<T>(operation: () => Promise<T>, operationName: string): Promise<T | null> {
  try {
    console.log(`\n🚀 Starting ${operationName}...`);
    const result = await operation();
    console.log(`✅ ${operationName} completed successfully`);
    return result;
  } catch (error) {
    console.error(`❌ Error in ${operationName}:`, error);
    return null;
  }
}

// Main execution function
async function runComprehensiveExample() {
  console.log('🌟 Starting Comprehensive Ragbits + BubbleLab Integration Example');

  // 1. Get the integration instance
  const integration = RagbitsBubbleLabIntegration.getInstance();
  console.log('✅ Integration instance acquired');

  // 2. Generate Ragbits configuration from BubbleLab workflow
  const generatedConfig = await safeExecute(
    () => Promise.resolve(integration.generateConfig(comprehensiveWorkflow, {
      validate: true,
      format: 'json',
      targetEnvironment: 'development'
    })),
    'Configuration Generation'
  );

  if (generatedConfig) {
    if (generatedConfig.validationErrors && generatedConfig.validationErrors.length > 0) {
      console.error('❌ Configuration validation errors:', generatedConfig.validationErrors);
      return;
    }
    console.log('✅ Configuration generated and validated successfully');
  }

  // 3. Create and initialize workflow engine
  const workflowEngine = await safeExecute(
    async () => {
      const engine = integration.createWorkflowEngine(comprehensiveWorkflow, {
        timeout: 60000, // 1 minute timeout
        maxRetries: 2,
        enableLogging: true,
        logLevel: 'info'
      });
      await engine.initialize();
      return engine;
    },
    'Workflow Engine Initialization'
  );

  if (!workflowEngine) {
    console.error('❌ Failed to initialize workflow engine');
    return;
  }

  // 4. Create and configure monitoring service
  const monitoringService = await safeExecute(
    () => {
      const monitoring = integration.createMonitoringService({
        enableRealTimeMonitoring: true,
        enablePerformanceTracking: true,
        enableErrorTracking: true,
        logLevel: 'info',
        retentionPeriod: 7,
        samplingRate: 1.0,
        enableAlerting: true,
        alertThresholds: {
          executionTime: 30000, // 30 seconds
          errorRate: 5, // 5%
          memoryUsage: 1024 // 1GB
        }
      });

      // Add event listener for real-time monitoring
      monitoring.addEventListener((event) => {
        const timestamp = new Date(event.timestamp).toLocaleTimeString();
        console.log(`📊 [${timestamp}] ${event.eventType} - ${event.nodeId || event.workflowId}`);
      });

      // Add alert callback
      monitoring.addAlertCallback((alert) => {
        console.warn(`🚨 MONITORING ALERT: ${alert}`);
      });

      return monitoring;
    },
    'Monitoring Service Setup'
  );

  if (!monitoringService) {
    console.error('❌ Failed to set up monitoring service');
    return;
  }

  // 5. Execute the workflow with sample queries
  const sampleQueries = [
    'What are the key concepts discussed in these documents?',
    'Summarize the main points from the documentation',
    'Find information about advanced features'
  ];

  for (const [index, query] of sampleQueries.entries()) {
    console.log(`\n🔍 Executing query ${index + 1}/${sampleQueries.length}: "${query}"`);

    const results = await safeExecute(
      () => workflowEngine.executeWorkflow({ query }),
      `Workflow Execution for Query ${index + 1}`
    );

    if (results) {
      console.log(`✅ Query ${index + 1} executed with ${results.length} node results`);
      
      // Display results
      for (const result of results) {
        if (result.success) {
          console.log(`  🟢 Node ${result.nodeId}: SUCCESS (${result.executionTime}ms)`);
        } else {
          console.log(`  🔴 Node ${result.nodeId}: FAILED - ${result.error}`);
        }
      }
    }
  }

  // 6. Retrieve and display performance metrics
  console.log('\n📈 Performance Metrics:');
  const metrics = monitoringService.getPerformanceMetrics();
  console.log(`  Workflow Execution Time: ${metrics.workflowExecutionTime}ms`);
  console.log(`  Node Execution Times:`, metrics.nodeExecutionTimes);
  console.log(`  Error Rate: ${metrics.errorRate}%`);
  console.log(`  Throughput: ${metrics.throughput} ops/min`);

  // 7. Retrieve and display workflow statistics
  console.log('\n📊 Workflow Statistics:');
  const stats = monitoringService.getWorkflowStats();
  console.log(`  Total Workflows: ${stats.totalWorkflows}`);
  console.log(`  Active Workflows: ${stats.activeWorkflows}`);
  console.log(`  Completed Workflows: ${stats.completedWorkflows}`);
  console.log(`  Error Workflows: ${stats.errorWorkflows}`);
  console.log(`  Average Execution Time: ${stats.averageExecutionTime}ms`);

  // 8. Export monitoring data for analysis
  const exportData = await safeExecute(
    () => Promise.resolve(monitoringService.exportData('json')),
    'Monitoring Data Export'
  );

  if (exportData) {
    console.log('\n📥 Monitoring data exported successfully');
    // In a real application, you might save this to a file or send to a monitoring service
  }

  // 9. Clean up resources
  await safeExecute(
    () => workflowEngine.dispose(),
    'Workflow Engine Disposal'
  );

  console.log('\n🎉 Comprehensive example completed successfully!');
}

// Advanced example with custom error handling and retry logic
async function runAdvancedExample() {
  console.log('\n\n⚙️  Starting Advanced Example with Custom Error Handling');

  const integration = RagbitsBubbleLabIntegration.getInstance();

  // Create a workflow with potential error scenarios
  const errorProneWorkflow: BubbleLabWorkflowConfig = {
    id: 'error-prone-workflow',
    name: 'Error Prone Workflow',
    description: 'Workflow designed to test error handling',
    nodes: [
      {
        id: 'ingest-node',
        type: 'ragbits-ingest',
        position: { x: 0, y: 0 },
        data: {
          sourceType: 'file',
          sourcePath: './potentially-missing-file.txt', // This might not exist
          chunkSize: 1000
        }
      },
      {
        id: 'search-node',
        type: 'ragbits-search',
        position: { x: 300, y: 0 },
        data: {
          topK: 5,
          scoreThreshold: 0.9 // High threshold might yield no results
        }
      }
    ],
    edges: [
      { source: 'ingest-node', target: 'search-node' }
    ],
    metadata: {}
  };

  // Create workflow engine with custom error handling
  const engine = integration.createWorkflowEngine(errorProneWorkflow, {
    timeout: 10000, // Shorter timeout for this example
    maxRetries: 1,
    enableLogging: true,
    logLevel: 'debug'
  });

  try {
    await engine.initialize();

    // Execute with error handling
    const results = await engine.executeWorkflow({
      query: 'Test query for error handling'
    });

    console.log('Results from error-prone workflow:', results);

    // Check for errors and handle them appropriately
    const failedNodes = results.filter(result => !result.success);
    if (failedNodes.length > 0) {
      console.log(`⚠️  ${failedNodes.length} nodes failed:`);
      failedNodes.forEach(node => {
        console.log(`  - ${node.nodeId}: ${node.error}`);
      });
    } else {
      console.log('✅ All nodes executed successfully');
    }
  } catch (error) {
    console.error('❌ Error executing error-prone workflow:', error);
  } finally {
    await engine.dispose();
  }

  console.log('✅ Advanced example completed');
}

// Run the examples
async function runAllExamples() {
  try {
    await runComprehensiveExample();
    await runAdvancedExample();
    
    console.log('\n🏁 All examples completed successfully!');
  } catch (error) {
    console.error('💥 Error running examples:', error);
  }
}

// Export for use as a module or for direct execution
export {
  comprehensiveWorkflow,
  runComprehensiveExample,
  runAdvancedExample,
  runAllExamples
};

// Execute if this file is run directly
if (require.main === module) {
  runAllExamples();
}