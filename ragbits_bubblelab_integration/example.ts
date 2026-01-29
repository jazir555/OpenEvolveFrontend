/**
 * Example usage of Ragbits + BubbleLab Integration
 * 
 * This example demonstrates how to use the integration to create and execute a RAG workflow
 */

import { 
  RagbitsBubbleLabIntegration,
  RAGBitsIngestBubble,
  RAGBitsSearchBubble,
  RAGBitsGenerationBubble,
  type BubbleLabWorkflowConfig
} from './index';

// Define a sample BubbleLab workflow configuration
const sampleWorkflow: BubbleLabWorkflowConfig = {
  id: 'sample-rag-workflow',
  name: 'Sample RAG Workflow',
  description: 'A simple RAG workflow for demonstration',
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
        metadata: { source: 'example' }
      }
    },
    {
      id: 'search-node',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 5,
        scoreThreshold: 0.7,
        enableHybridSearch: true
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
  metadata: {
    createdBy: 'example-user',
    createdAt: new Date().toISOString()
  }
};

async function runExample() {
  console.log('Starting Ragbits + BubbleLab Integration Example');
  
  try {
    // Get the integration instance
    const integration = RagbitsBubbleLabIntegration.getInstance();
    
    // Generate a Ragbits configuration from the BubbleLab workflow
    console.log('\n1. Generating Ragbits configuration from BubbleLab workflow...');
    const generatedConfig = integration.generateConfig(sampleWorkflow, {
      validate: true,
      format: 'json',
      targetEnvironment: 'development'
    });
    
    if (generatedConfig.validationErrors && generatedConfig.validationErrors.length > 0) {
      console.error('Configuration validation errors:', generatedConfig.validationErrors);
    } else {
      console.log('✓ Configuration generated successfully');
    }
    
    // Create a workflow engine
    console.log('\n2. Creating workflow engine...');
    const workflowEngine = integration.createWorkflowEngine(sampleWorkflow, {
      timeout: 60000, // 1 minute timeout
      maxRetries: 2,
      enableLogging: true,
      logLevel: 'info'
    });
    
    // Initialize the workflow engine
    console.log('3. Initializing workflow engine...');
    await workflowEngine.initialize();
    console.log('✓ Workflow engine initialized');
    
    // Create and initialize monitoring service
    console.log('\n4. Setting up monitoring...');
    const monitoringService = integration.createMonitoringService({
      enableRealTimeMonitoring: true,
      enablePerformanceTracking: true,
      enableErrorTracking: true,
      logLevel: 'info'
    });
    
    // Add event listener for monitoring
    monitoringService.addEventListener((event) => {
      console.log(`[MONITORING] ${event.eventType} - ${event.nodeId || event.workflowId}`);
    });
    
    // Add the monitoring service to the workflow engine (in a real implementation)
    // This would be done by connecting the workflow engine to the monitoring service
    
    // Execute the workflow
    console.log('\n5. Executing workflow...');
    const results = await workflowEngine.executeWorkflow({
      query: 'What is the capital of France?',
      context: 'Geography knowledge base'
    });
    
    console.log('✓ Workflow executed successfully');
    console.log(`Results: ${results.length} nodes executed`);
    
    // Print results
    for (const result of results) {
      console.log(`- Node ${result.nodeId}: ${result.success ? 'SUCCESS' : 'FAILED'}`);
      if (!result.success) {
        console.log(`  Error: ${result.error}`);
      }
    }
    
    // Get and print performance metrics
    console.log('\n6. Performance metrics:');
    const metrics = monitoringService.getPerformanceMetrics();
    console.log(`- Workflow execution time: ${metrics.workflowExecutionTime}ms`);
    console.log(`- Average node execution time: ${Object.values(metrics.nodeExecutionTimes).reduce((a, b) => a + b, 0) / Object.keys(metrics.nodeExecutionTimes).length || 0}ms`);
    console.log(`- Error rate: ${metrics.errorRate}%`);
    console.log(`- Throughput: ${metrics.throughput} ops/min`);
    
    // Get workflow statistics
    const stats = monitoringService.getWorkflowStats();
    console.log('\n7. Workflow statistics:');
    console.log(`- Total workflows: ${stats.totalWorkflows}`);
    console.log(`- Active workflows: ${stats.activeWorkflows}`);
    console.log(`- Completed workflows: ${stats.completedWorkflows}`);
    console.log(`- Error workflows: ${stats.errorWorkflows}`);
    console.log(`- Average execution time: ${stats.averageExecutionTime}ms`);
    
    console.log('\n✓ Example completed successfully!');
    
  } catch (error) {
    console.error('Example failed with error:', error);
  }
}

// Run the example
if (require.main === module) {
  runExample();
}

export { sampleWorkflow, runExample };