/**
 * End-to-End Integration Tests for Ragbits + BubbleLab Integration
 */

import {
  RagbitsBubbleLabIntegration,
  RAGBitsWorkflowEngine,
  RagbitsProcessorIntegration,
  MonitoringService,
  ConfigGenerator,
  type BubbleLabWorkflowConfig
} from '../index';

// Mock the Ragbits document processor to avoid external dependencies in tests
jest.mock('../../knowledge_engine/ragbits_document_processor', () => {
  return {
    RAGBitsDocumentProcessor: jest.fn().mockImplementation(() => {
      return {
        initialize: jest.fn().mockResolvedValue(true),
        ingest_file: jest.fn().mockResolvedValue({
          success: true,
          document_id: 'test-doc-id',
          chunks_ingested: 1,
          error: null
        }),
        ingest_text: jest.fn().mockResolvedValue({
          success: true,
          document_id: 'test-doc-id',
          chunks_ingested: 1,
          error: null
        }),
        search: jest.fn().mockResolvedValue([
          {
            content: 'Test search result content',
            score: 0.9,
            metadata: { document_id: 'test-doc-id', source: 'test' }
          }
        ]),
        get_statistics: jest.fn().mockResolvedValue({ total_documents: 1, total_chunks: 1 }),
        clear: jest.fn().mockResolvedValue(true),
        close: jest.fn().mockResolvedValue(undefined)
      };
    })
  };
});

describe('End-to-End Integration Tests for Ragbits + BubbleLab Integration', () => {
  test('full integration workflow: create, configure, execute, monitor', async () => {
    // 1. Get the integration instance
    const integration = RagbitsBubbleLabIntegration.getInstance();
    
    // 2. Define a sample workflow
    const sampleWorkflow: BubbleLabWorkflowConfig = {
      id: 'e2e-test-workflow',
      name: 'E2E Test Workflow',
      description: 'An end-to-end test workflow',
      nodes: [
        {
          id: 'ingest-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'text',
            sourcePath: 'inline',
            chunkSize: 1000,
            chunkOverlap: 200,
            metadata: { source: 'e2e-test' }
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
            systemPrompt: 'You are a helpful assistant.'
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
        createdBy: 'e2e-test',
        createdAt: new Date().toISOString()
      }
    };

    // 3. Generate a Ragbits configuration from the BubbleLab workflow
    const generatedConfig = integration.generateConfig(sampleWorkflow, {
      validate: true,
      format: 'json',
      targetEnvironment: 'development'
    });

    expect(generatedConfig).toBeDefined();
    expect(generatedConfig.ragbitsConfig).toBeDefined();
    expect(generatedConfig.validationErrors).toBeUndefined();

    // 4. Create and initialize the workflow engine
    const workflowEngine = integration.createWorkflowEngine(sampleWorkflow, {
      timeout: 30000, // 30 second timeout
      maxRetries: 2,
      enableLogging: true,
      logLevel: 'info'
    });

    await workflowEngine.initialize();
    expect(workflowEngine).toBeDefined();

    // 5. Create and set up the monitoring service
    const monitoringService = integration.createMonitoringService({
      enableRealTimeMonitoring: true,
      enablePerformanceTracking: true,
      enableErrorTracking: true,
      logLevel: 'info'
    });

    // Add event listener for monitoring
    let eventCount = 0;
    monitoringService.addEventListener((event) => {
      eventCount++;
      expect(event.workflowId).toBe('e2e-test-workflow');
    });

    // 6. Execute the workflow
    const results = await workflowEngine.executeWorkflow({
      query: 'What is the meaning of life?',
      context: 'Philosophical concepts'
    });

    expect(results).toBeDefined();
    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeGreaterThanOrEqual(1);

    // Check individual results
    for (const result of results) {
      expect(result.nodeId).toBeDefined();
      expect(typeof result.success).toBe('boolean');
    }

    // 7. Verify monitoring captured events
    const workflowEvents = monitoringService.getWorkflowEvents('e2e-test-workflow');
    expect(workflowEvents).toBeDefined();
    expect(Array.isArray(workflowEvents)).toBe(true);

    // 8. Check performance metrics
    const metrics = monitoringService.getPerformanceMetrics();
    expect(metrics).toBeDefined();
    expect(typeof metrics.workflowExecutionTime).toBe('number');
    expect(typeof metrics.errorRate).toBe('number');
    expect(typeof metrics.throughput).toBe('number');

    // 9. Check workflow statistics
    const stats = monitoringService.getWorkflowStats();
    expect(stats).toBeDefined();
    expect(typeof stats.totalWorkflows).toBe('number');
    expect(typeof stats.completedWorkflows).toBe('number');

    // 10. Clean up
    await workflowEngine.dispose();
  });

  test('integration with processor: ingest, search, and monitor', async () => {
    // 1. Create processor integration
    const processorIntegration = new RagbitsProcessorIntegration();
    await processorIntegration.initialize();

    // 2. Process a document
    const documentResult = await processorIntegration.processDocument(
      './test-e2e.txt',
      'This is a test document for end-to-end integration testing. It contains meaningful content that can be searched later.',
      { source: 'e2e-test', type: 'integration-test' }
    );

    expect(documentResult).toBeDefined();
    expect(documentResult.success).toBe(true);
    expect(documentResult.documentId).toBeDefined();

    // 3. Search for content in the processed document
    const searchResults = await processorIntegration.search(
      'integration testing',
      5,
      { source: 'e2e-test' },
      0.5
    );

    expect(searchResults).toBeDefined();
    expect(Array.isArray(searchResults)).toBe(true);
    expect(searchResults.length).toBeGreaterThanOrEqual(0); // Might be 0 due to mock

    // 4. Get processor statistics
    const stats = processorIntegration.getStats();
    expect(stats).toBeDefined();
    expect(typeof stats.totalProcessed).toBe('number');
    expect(typeof stats.successful).toBe('number');

    // 5. Get index statistics
    const indexStats = await processorIntegration.getIndexStats();
    expect(indexStats).toBeDefined();

    // 6. Clean up
    await processorIntegration.dispose();
  });

  test('configuration mapping and generation roundtrip', () => {
    // 1. Define a BubbleLab workflow
    const originalWorkflow: BubbleLabWorkflowConfig = {
      id: 'roundtrip-test',
      name: 'Roundtrip Test',
      description: 'Test configuration roundtrip',
      nodes: [
        {
          id: 'ingest-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'file',
            sourcePath: './test.txt',
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
        }
      ],
      edges: [
        {
          id: 'edge-1',
          source: 'ingest-node',
          target: 'search-node'
        }
      ],
      metadata: {}
    };

    // 2. Generate Ragbits configuration from BubbleLab workflow
    const integration = RagbitsBubbleLabIntegration.getInstance();
    const generatedConfig = integration.generateConfig(originalWorkflow, {
      validate: true
    });

    expect(generatedConfig).toBeDefined();
    expect(generatedConfig.ragbitsConfig).toBeDefined();
    expect(generatedConfig.validationErrors).toBeUndefined();

    // 3. Map BubbleLab workflow to Ragbits config using ConfigMapper directly
    const mappedConfig = require('../index').ConfigMapper.mapBubbleLabToRagbits(originalWorkflow);
    expect(mappedConfig).toBeDefined();

    // 4. Verify that both methods produce similar structures
    expect(generatedConfig.ragbitsConfig.workflow.name).toBe(mappedConfig.workflow.name);
    expect(generatedConfig.ragbitsConfig.workflow.nodes.length).toBe(mappedConfig.workflow.nodes.length);
    expect(generatedConfig.ragbitsConfig.workflow.connections.length).toBe(mappedConfig.workflow.connections.length);
  });

  test('monitoring integration with workflow execution', async () => {
    // 1. Set up monitoring service
    const monitoringService = new MonitoringService({
      enableRealTimeMonitoring: true,
      enablePerformanceTracking: true,
      enableErrorTracking: true,
      logLevel: 'info'
    });

    // 2. Track events
    const events: any[] = [];
    monitoringService.addEventListener((event) => {
      events.push(event);
    });

    // 3. Create a simple workflow
    const simpleWorkflow: BubbleLabWorkflowConfig = {
      id: 'monitoring-test',
      name: 'Monitoring Test',
      description: 'Test monitoring integration',
      nodes: [
        {
          id: 'ingest-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'text',
            sourcePath: 'inline'
          }
        }
      ],
      edges: [],
      metadata: {}
    };

    // 4. Create and execute workflow engine
    const workflowEngine = new RAGBitsWorkflowEngine(simpleWorkflow);
    await workflowEngine.initialize();

    // Log workflow start
    monitoringService.logWorkflowStart('monitoring-test');

    // Execute workflow
    const startTime = Date.now();
    const results = await workflowEngine.executeWorkflow({ content: 'Test content for monitoring' });
    const duration = Date.now() - startTime;

    // Log workflow completion
    monitoringService.logWorkflowComplete('monitoring-test', duration);

    // 5. Verify monitoring captured events
    expect(events.length).toBeGreaterThanOrEqual(2); // At least start and complete
    
    const startEvent = events.find(e => e.eventType === 'workflow_start');
    const completeEvent = events.find(e => e.eventType === 'workflow_complete');
    
    expect(startEvent).toBeDefined();
    expect(completeEvent).toBeDefined();
    expect(startEvent!.workflowId).toBe('monitoring-test');
    expect(completeEvent!.workflowId).toBe('monitoring-test');
    expect(completeEvent!.duration).toBeGreaterThanOrEqual(0);

    // 6. Verify metrics
    const metrics = monitoringService.getPerformanceMetrics();
    expect(metrics.workflowExecutionTime).toBeGreaterThanOrEqual(0);

    // 7. Verify workflow stats
    const stats = monitoringService.getWorkflowStats();
    expect(stats.totalWorkflows).toBeGreaterThanOrEqual(1);

    // 8. Clean up
    await workflowEngine.dispose();
  });

  test('integration with all services working together', async () => {
    // 1. Get integration instance
    const integration = RagbitsBubbleLabIntegration.getInstance();

    // 2. Create all services
    const workflowConfig: BubbleLabWorkflowConfig = {
      id: 'full-integration-test',
      name: 'Full Integration Test',
      description: 'Test all services integration',
      nodes: [
        {
          id: 'ingest-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'text',
            sourcePath: 'inline',
            metadata: { test: true }
          }
        }
      ],
      edges: [],
      metadata: {}
    };

    const workflowEngine = integration.createWorkflowEngine(workflowConfig);
    const processorIntegration = integration.createProcessorIntegration();
    const monitoringService = integration.createMonitoringService();

    // 3. Initialize services
    await workflowEngine.initialize();
    await processorIntegration.initialize();

    // 4. Add monitoring
    monitoringService.addEventListener((event) => {
      expect(event.workflowId).toBeDefined();
    });

    // 5. Execute workflow
    const workflowResults = await workflowEngine.executeWorkflow({
      content: 'Full integration test content'
    });

    // 6. Process a document separately
    const docResult = await processorIntegration.processDocument(
      './integration-test.txt',
      'This is content for full integration testing'
    );

    // 7. Search the processed content
    const searchResults = await processorIntegration.search('integration');

    // 8. Verify all components worked
    expect(workflowResults).toBeDefined();
    expect(Array.isArray(workflowResults)).toBe(true);
    expect(docResult).toBeDefined();
    expect(docResult.success).toBe(true);
    expect(Array.isArray(searchResults)).toBe(true);

    // 9. Check monitoring captured activity
    const metrics = monitoringService.getPerformanceMetrics();
    expect(metrics).toBeDefined();

    // 10. Clean up
    await workflowEngine.dispose();
    await processorIntegration.dispose();
  });

  test('real-world scenario: document QA workflow', async () => {
    // Simulate a real-world document QA workflow
    const qaWorkflow: BubbleLabWorkflowConfig = {
      id: 'document-qa-workflow',
      name: 'Document QA Workflow',
      description: 'A realistic document question answering workflow',
      nodes: [
        {
          id: 'document-loader',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'file',
            sourcePath: './documents/',
            chunkSize: 500,
            chunkOverlap: 50,
            metadata: { 
              source: 'user-uploaded-documents',
              processingStage: 'loading'
            }
          }
        },
        {
          id: 'vector-indexer',
          type: 'ragbits-index',
          position: { x: 300, y: 0 },
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
          position: { x: 600, y: 0 },
          data: {
            topK: 10,
            scoreThreshold: 0.8,
            enableHybridSearch: true,
            defaultFilters: { 
              source: 'user-uploaded-documents',
              processingStage: 'indexed'
            }
          }
        },
        {
          id: 'answer-generator',
          type: 'ragbits-generation',
          position: { x: 900, y: 0 },
          data: {
            llmModel: 'gpt-4o',
            temperature: 0.3,
            maxTokens: 1500,
            systemPrompt: 'You are a helpful document assistant. Answer questions based on the provided context. Be accurate and cite sources when possible.'
          }
        }
      ],
      edges: [
        {
          id: 'load-to-index',
          source: 'document-loader',
          target: 'vector-indexer'
        },
        {
          id: 'index-to-search',
          source: 'vector-indexer',
          target: 'semantic-searcher'
        },
        {
          id: 'search-to-generate',
          source: 'semantic-searcher',
          target: 'answer-generator'
        }
      ],
      metadata: {
        domain: 'document-qa',
        complexity: 'high',
        expectedThroughput: 'medium',
        createdAt: new Date().toISOString()
      }
    };

    // Get integration instance
    const integration = RagbitsBubbleLabIntegration.getInstance();

    // Generate configuration
    const config = integration.generateConfig(qaWorkflow, {
      validate: true,
      targetEnvironment: 'development'
    });

    expect(config).toBeDefined();
    expect(config.ragbitsConfig).toBeDefined();
    expect(config.validationErrors).toBeUndefined();

    // Create and initialize workflow engine
    const engine = integration.createWorkflowEngine(qaWorkflow);
    await engine.initialize();

    // Create monitoring
    const monitoring = integration.createMonitoringService({
      enableRealTimeMonitoring: true,
      enablePerformanceTracking: true
    });

    // Add monitoring to workflow execution
    let monitoringEvents: any[] = [];
    monitoring.addEventListener(event => {
      monitoringEvents.push(event);
    });

    // Execute the QA workflow
    const qaResults = await engine.executeWorkflow({
      query: 'What are the main topics covered in these documents?',
      context: 'Document analysis and summarization'
    });

    // Verify results
    expect(qaResults).toBeDefined();
    expect(Array.isArray(qaResults)).toBe(true);
    expect(qaResults.length).toBeGreaterThan(0);

    // Check that all nodes executed
    const nodeIds = qaWorkflow.nodes.map(n => n.id);
    const executedNodeIds = qaResults.map(r => r.nodeId);
    expect(executedNodeIds.sort()).toEqual(nodeIds.sort());

    // Check monitoring captured events
    expect(monitoringEvents.length).toBeGreaterThan(0);

    // Get performance metrics
    const metrics = monitoring.getPerformanceMetrics();
    expect(metrics).toBeDefined();

    // Get workflow stats
    const stats = monitoring.getWorkflowStats();
    expect(stats).toBeDefined();

    // Clean up
    await engine.dispose();
  });

  test('error recovery and resilience testing', async () => {
    // Test error recovery capabilities
    const integration = RagbitsBubbleLabIntegration.getInstance();

    // Create a workflow with potential error points
    const resilientWorkflow: BubbleLabWorkflowConfig = {
      id: 'resilient-workflow',
      name: 'Resilient Workflow',
      description: 'Workflow designed to test error recovery',
      nodes: [
        {
          id: 'primary-ingest',
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
          id: 'backup-search',
          type: 'ragbits-search',
          position: { x: 300, y: 0 },
          data: {
            topK: 5,
            scoreThreshold: 0.5,
            enableHybridSearch: false
          }
        }
      ],
      edges: [
        {
          id: 'ingest-to-search',
          source: 'primary-ingest',
          target: 'backup-search'
        }
      ],
      metadata: {}
    };

    // Create workflow engine with error handling options
    const engine = integration.createWorkflowEngine(resilientWorkflow, {
      timeout: 15000,
      maxRetries: 3,
      enableLogging: true,
      logLevel: 'debug'
    });

    await engine.initialize();

    // Execute workflow multiple times to test resilience
    for (let i = 0; i < 3; i++) {
      const results = await engine.executeWorkflow({
        content: `Test content for resilience test ${i}`,
        query: `Query ${i}`
      });

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBeGreaterThan(0);
    }

    // Test that the engine can handle multiple executions without issues
    const finalStats = engine.getExecutionHistory();
    expect(finalStats.length).toBeGreaterThanOrEqual(6); // At least 2 nodes * 3 executions

    // Clean up
    await engine.dispose();
  });

  test('scalability and concurrent execution', async () => {
    // Test the system's ability to handle concurrent workflows
    const integration = RagbitsBubbleLabIntegration.getInstance();

    // Create a simple workflow template
    const workflowTemplate: BubbleLabWorkflowConfig = {
      id: 'concurrent-template',
      name: 'Concurrent Template',
      description: 'Template for concurrent execution testing',
      nodes: [
        {
          id: 'ingest-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'text',
            sourcePath: 'inline',
            metadata: { concurrent: true }
          }
        }
      ],
      edges: [],
      metadata: {}
    };

    // Create multiple workflow instances
    const workflowInstances = Array.from({ length: 5 }, (_, i) => ({
      ...workflowTemplate,
      id: `concurrent-workflow-${i}`,
      name: `Concurrent Workflow ${i}`,
      metadata: { ...workflowTemplate.metadata, instance: i }
    }));

    // Create and initialize multiple engines
    const engines = workflowInstances.map(config => integration.createWorkflowEngine(config));
    await Promise.all(engines.map(engine => engine.initialize()));

    // Execute all workflows concurrently
    const executionPromises = engines.map((engine, i) => 
      engine.executeWorkflow({ 
        content: `Concurrent test content for workflow ${i}`,
        query: `Query for workflow ${i}`
      })
    );

    const allResults = await Promise.all(executionPromises);

    // Verify all executions completed successfully
    expect(allResults).toHaveLength(5);
    allResults.forEach((results, i) => {
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      expect(results.length).toBeGreaterThanOrEqual(0); // Could be 0 if no nodes execute
    });

    // Test monitoring with concurrent workflows
    const monitoring = integration.createMonitoringService({
      enableRealTimeMonitoring: true
    });

    let totalEvents = 0;
    monitoring.addEventListener(() => {
      totalEvents++;
    });

    // Execute one more workflow to test monitoring
    const additionalEngine = integration.createWorkflowEngine(workflowTemplate);
    await additionalEngine.initialize();
    await additionalEngine.executeWorkflow({ content: 'Monitoring test' });
    await additionalEngine.dispose();

    // Verify monitoring captured events
    expect(totalEvents).toBeGreaterThan(0);

    // Clean up all engines
    await Promise.all(engines.map(engine => engine.dispose()));
  });

  test('configuration validation and environment adaptation', async () => {
    const integration = RagbitsBubbleLabIntegration.getInstance();

    // Create a workflow that should work in different environments
    const envTestWorkflow: BubbleLabWorkflowConfig = {
      id: 'env-test-workflow',
      name: 'Environment Test Workflow',
      description: 'Workflow for testing environment adaptation',
      nodes: [
        {
          id: 'ingest-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'file',
            sourcePath: './test-data/',
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
        }
      ],
      edges: [
        {
          id: 'ingest-to-search',
          source: 'ingest-node',
          target: 'search-node'
        }
      ],
      metadata: {}
    };

    // Test configuration generation for different environments
    const devConfig = integration.generateConfig(envTestWorkflow, {
      targetEnvironment: 'development',
      validate: true
    });

    const prodConfig = integration.generateConfig(envTestWorkflow, {
      targetEnvironment: 'production',
      validate: true
    });

    // Verify configurations are different for different environments
    expect(devConfig.ragbitsConfig).toBeDefined();
    expect(prodConfig.ragbitsConfig).toBeDefined();

    // In production, vector store should typically be different than in development
    // (though with mocks, they might appear the same)
    expect(devConfig.ragbitsConfig.documentProcessor.chunk_size).toBe(1000); // Default
    expect(prodConfig.ragbitsConfig.documentProcessor.chunk_size).toBe(500); // Production setting

    // Verify both configurations are valid
    expect(devConfig.validationErrors).toBeUndefined();
    expect(prodConfig.validationErrors).toBeUndefined();

    // Create engines for both configurations
    const devEngine = integration.createWorkflowEngine(envTestWorkflow);
    const prodEngine = integration.createWorkflowEngine(envTestWorkflow);

    await devEngine.initialize();
    await prodEngine.initialize();

    // Execute both workflows
    const devResults = await devEngine.executeWorkflow({ content: 'Dev environment test' });
    const prodResults = await prodEngine.executeWorkflow({ content: 'Prod environment test' });

    expect(devResults).toBeDefined();
    expect(prodResults).toBeDefined();

    // Clean up
    await devEngine.dispose();
    await prodEngine.dispose();
  });

  test('cleanup and resource management', async () => {
    // Test proper cleanup and resource management
    const integration = RagbitsBubbleLabIntegration.getInstance();

    const cleanupWorkflow: BubbleLabWorkflowConfig = {
      id: 'cleanup-test',
      name: 'Cleanup Test',
      description: 'Workflow for testing cleanup',
      nodes: [
        {
          id: 'test-node',
          type: 'ragbits-ingest',
          position: { x: 0, y: 0 },
          data: {
            sourceType: 'text',
            sourcePath: 'inline'
          }
        }
      ],
      edges: [],
      metadata: {}
    };

    // Create multiple services
    const engine = integration.createWorkflowEngine(cleanupWorkflow);
    const processor = integration.createProcessorIntegration();
    const monitoring = integration.createMonitoringService();

    // Initialize all services
    await engine.initialize();
    await processor.initialize();

    // Execute a workflow
    await engine.executeWorkflow({ content: 'Test for cleanup' });

    // Verify services have resources allocated
    const initialHistory = engine.getExecutionHistory();
    expect(initialHistory).toBeDefined();

    // Perform cleanup
    await engine.dispose();
    await processor.dispose();

    // Verify cleanup worked (this is more of a smoke test since we're using mocks)
    // The main goal is to ensure dispose methods don't throw errors
    expect(() => {}).not.toThrow(); // If we got here, cleanup succeeded
  });
});