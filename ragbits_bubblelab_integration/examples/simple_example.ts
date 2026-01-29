/**
 * Simple Example of Ragbits + BubbleLab Integration
 * 
 * This example demonstrates basic usage of the integration
 */

import { 
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig
} from '../index';

// Define a simple BubbleLab workflow configuration
const simpleWorkflow: BubbleLabWorkflowConfig = {
  id: 'simple-rag-workflow',
  name: 'Simple RAG Workflow',
  description: 'A basic RAG workflow for demonstration',
  nodes: [
    {
      id: 'ingest-node',
      type: 'ragbits-ingest',
      position: { x: 0, y: 0 },
      data: {
        sourceType: 'text',
        sourcePath: 'example-source',
        chunkSize: 500,
        chunkOverlap: 50,
        metadata: { 
          source: 'simple_example', 
          created_at: new Date().toISOString() 
        }
      }
    },
    {
      id: 'search-node',
      type: 'ragbits-search',
      position: { x: 300, y: 0 },
      data: {
        topK: 3,
        scoreThreshold: 0.5,
        enableHybridSearch: false,
        defaultFilters: {}
      }
    },
    {
      id: 'generation-node',
      type: 'ragbits-generation',
      position: { x: 600, y: 0 },
      data: {
        llmModel: 'gpt-4o-mini',
        temperature: 0.7,
        maxTokens: 500,
        systemPrompt: 'You are a helpful assistant that answers questions based on the provided context.'
      }
    }
  ],
  edges: [
    {
      id: 'ingest-to-search',
      source: 'ingest-node',
      target: 'search-node'
    },
    {
      id: 'search-to-generation',
      source: 'search-node',
      target: 'generation-node'
    }
  ],
  metadata: {
    category: 'examples',
    complexity: 'simple',
    created_by: 'integration-demo'
  }
};

async function runSimpleExample() {
  console.log('Starting Ragbits + BubbleLab Integration - Simple Example');
  console.log('=' .repeat(60));
  
  try {
    // Get the integration instance
    const integration = RagbitsBubbleLabIntegration.getInstance();
    
    // Generate a Ragbits configuration from the BubbleLab workflow
    console.log('\n1. Generating Ragbits configuration from BubbleLab workflow...');
    const generatedConfig = integration.generateConfig(simpleWorkflow, {
      validate: true,
      format: 'json',
      targetEnvironment: 'development'
    });
    
    if (generatedConfig.validationErrors && generatedConfig.validationErrors.length > 0) {
      console.error('❌ Configuration validation errors:', generatedConfig.validationErrors);
      return;
    }
    
    console.log('✅ Configuration generated successfully');
    console.log(`   Workflow Name: ${generatedConfig.ragbitsConfig.workflow.name}`);
    console.log(`   Number of Nodes: ${generatedConfig.ragbitsConfig.workflow.nodes.length}`);
    
    // Map the BubbleLab workflow to Ragbits configuration
    console.log('\n2. Mapping BubbleLab workflow to Ragbits configuration...');
    const mappedConfig = integration.mapConfig(simpleWorkflow);
    console.log('✅ Workflow mapped successfully');
    
    // Note: In a real implementation, we would initialize and run the workflow engine
    // For this example, we'll just demonstrate the configuration generation
    
    console.log('\n3. Configuration Preview:');
    console.log(`   Document Processor Embedding Model: ${mappedConfig.documentProcessor.embedding_model}`);
    console.log(`   Search Default Top-K: ${mappedConfig.search.default_top_k}`);
    console.log(`   Generation Default Model: ${mappedConfig.generation.default_model}`);
    
    console.log('\n4. Example of creating individual components:');
    
    // Example of creating a monitoring service
    const monitoringService = integration.createMonitoringService({
      enableRealTimeMonitoring: true,
      logLevel: 'info',
      alertThresholds: {
        executionTime: 10000, // 10 seconds
        errorRate: 5, // 5%
        memoryUsage: 512 // 512MB
      }
    });
    console.log('✅ Monitoring service created');
    
    // Example of creating a processor integration
    const processorIntegration = integration.createProcessorIntegration({
      enableAutoIndexing: true,
      batchSize: 5,
      enableCaching: true
    });
    console.log('✅ Processor integration created');
    
    console.log('\n✅ Simple example completed successfully!');
    console.log('\nThe integration is properly set up and ready to use.');
    console.log('You can now create more complex workflows using the same patterns.');
    
  } catch (error) {
    console.error('❌ Example failed with error:', error);
  }
}

// Run the example if this file is executed directly
if (require.main === module) {
  runSimpleExample();
}

export { simpleWorkflow, runSimpleExample };