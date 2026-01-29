/**
 * Integration Tests for Ragbits + BubbleLab Integration
 */

import {
  RagbitsBubbleLabIntegration,
  RAGBitsIngestBubble,
  RAGBitsSearchBubble,
  RAGBitsGenerationBubble,
  RAGBitsIndexBubble,
  ConfigMapper,
  ConfigGenerator,
  RAGBitsWorkflowEngine,
  RagbitsProcessorIntegration,
  MonitoringService,
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

describe('Integration Tests for Ragbits + BubbleLab Integration', () => {
  describe('Core Integration Layer', () => {
    test('should create integration instance', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      expect(integration).toBeDefined();
    });

    test('should create all major components', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      
      // Create workflow engine
      const workflowConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };
      const engine = integration.createWorkflowEngine(workflowConfig);
      expect(engine).toBeDefined();

      // Create processor integration
      const processor = integration.createProcessorIntegration();
      expect(processor).toBeDefined();

      // Create monitoring service
      const monitoring = integration.createMonitoringService();
      expect(monitoring).toBeDefined();
    });
  });

  describe('RAGBits Bubble Components', () => {
    test('should create all bubble components', () => {
      const ingestConfig = {
        id: 'ingest-test',
        name: 'Test Ingest',
        description: 'Test ingest bubble',
        sourceType: 'file',
        sourcePath: './test.txt'
      };

      const searchConfig = {
        id: 'search-test',
        name: 'Test Search',
        description: 'Test search bubble',
        topK: 5
      };

      const generationConfig = {
        id: 'generation-test',
        name: 'Test Generation',
        description: 'Test generation bubble',
        llmModel: 'gpt-4o'
      };

      const indexConfig = {
        id: 'index-test',
        name: 'Test Index',
        description: 'Test index bubble',
        vectorStoreType: 'memory'
      };

      const ingestBubble = new RAGBitsIngestBubble(ingestConfig);
      const searchBubble = new RAGBitsSearchBubble(searchConfig);
      const generationBubble = new RAGBitsGenerationBubble(generationConfig);
      const indexBubble = new RAGBitsIndexBubble(indexConfig);

      expect(ingestBubble).toBeDefined();
      expect(searchBubble).toBeDefined();
      expect(generationBubble).toBeDefined();
      expect(indexBubble).toBeDefined();
    });

    test('should initialize all bubble components', async () => {
      const ingestConfig = {
        id: 'ingest-test',
        name: 'Test Ingest',
        description: 'Test ingest bubble',
        sourceType: 'file',
        sourcePath: './test.txt'
      };

      const searchConfig = {
        id: 'search-test',
        name: 'Test Search',
        description: 'Test search bubble',
        topK: 5
      };

      const generationConfig = {
        id: 'generation-test',
        name: 'Test Generation',
        description: 'Test generation bubble',
        llmModel: 'gpt-4o'
      };

      const indexConfig = {
        id: 'index-test',
        name: 'Test Index',
        description: 'Test index bubble',
        vectorStoreType: 'memory'
      };

      const ingestBubble = new RAGBitsIngestBubble(ingestConfig);
      const searchBubble = new RAGBitsSearchBubble(searchConfig);
      const generationBubble = new RAGBitsGenerationBubble(generationConfig);
      const indexBubble = new RAGBitsIndexBubble(indexConfig);

      await ingestBubble.initialize();
      await searchBubble.initialize();
      await generationBubble.initialize();
      await indexBubble.initialize();

      // Should not throw
    });
  });

  describe('Configuration Mapping System', () => {
    test('should map bubble lab config to ragbits config', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
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
              scoreThreshold: 0.7,
              enableHybridSearch: true
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

      const ragbitsConfig = ConfigMapper.mapBubbleLabToRagbits(bubbleLabConfig);
      
      expect(ragbitsConfig).toBeDefined();
      expect(ragbitsConfig.workflow.name).toBe('Test Workflow');
      expect(ragbitsConfig.workflow.nodes).toHaveLength(2);
      expect(ragbitsConfig.workflow.connections).toHaveLength(1);
    });

    test('should validate bubble lab config', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [
          {
            id: 'ingest-node',
            type: 'ragbits-ingest',
            position: { x: 0, y: 0 },
            data: {
              sourceType: 'file',
              sourcePath: './test.txt'
            }
          }
        ],
        edges: [],
        metadata: {}
      };

      const validation = ConfigMapper.validateBubbleLabConfig(bubbleLabConfig);
      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should generate config from bubble lab config', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
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
          }
        ],
        edges: [],
        metadata: {}
      };

      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig, {
        validate: true,
        format: 'json',
        targetEnvironment: 'development'
      });

      expect(generatedConfig).toBeDefined();
      expect(generatedConfig.ragbitsConfig).toBeDefined();
      expect(generatedConfig.validationErrors).toBeUndefined();
    });
  });

  describe('RAG Workflow Execution Engine', () => {
    test('should create and initialize workflow engine', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      await engine.initialize();

      expect(engine).toBeDefined();
    });

    test('should execute workflow with no nodes', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      await engine.initialize();

      const results = await engine.executeWorkflow();
      expect(results).toHaveLength(0);
    });

    test('should get execution history', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      await engine.initialize();

      const results = await engine.executeWorkflow();
      const history = engine.getExecutionHistory();

      expect(history).toEqual(results);
    });
  });

  describe('Ragbits Document Processor Integration', () => {
    test('should create and initialize processor integration', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      expect(processor).toBeDefined();
    });

    test('should process document', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const result = await processor.processDocument(
        './test.txt',
        'This is a test document for integration testing. It contains meaningful content that can be searched later.',
        { source: 'integration-test', type: 'test-document' }
      );

      expect(result).toBeDefined();
      expect(result.success).toBe(true);
      expect(result.documentId).toBeDefined();
    });

    test('should search documents', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const results = await processor.search('integration testing');
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });

    test('should get processor statistics', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const stats = processor.getStats();
      expect(stats).toBeDefined();
      expect(typeof stats.totalProcessed).toBe('number');
    });
  });

  describe('Enhanced Configuration Generator', () => {
    test('should generate config with environment settings', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig, {
        targetEnvironment: 'production'
      });

      expect(generatedConfig).toBeDefined();
      expect(generatedConfig.ragbitsConfig.documentProcessor.vector_store_type).toBe('qdrant');
    });

    test('should validate generated config', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig, {
        validate: true
      });

      expect(generatedConfig.validationErrors).toBeUndefined();
    });

    test('should format config in different formats', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig);

      const jsonFormat = ConfigGenerator.formatConfig(generatedConfig, 'json');
      expect(jsonFormat).toContain('"documentProcessor":');

      const tsFormat = ConfigGenerator.formatConfig(generatedConfig, 'typescript');
      expect(tsFormat).toContain('export const ragbitsConfig:');
    });
  });

  describe('Monitoring and Debugging Features', () => {
    test('should create monitoring service', () => {
      const monitoring = new MonitoringService();
      expect(monitoring).toBeDefined();
    });

    test('should track workflow events', () => {
      const monitoring = new MonitoringService();

      monitoring.logWorkflowStart('test-workflow');
      monitoring.logWorkflowComplete('test-workflow', 1000);

      const events = monitoring.getWorkflowEvents('test-workflow');
      expect(events).toHaveLength(2);
      expect(events[0].eventType).toBe('workflow_start');
      expect(events[1].eventType).toBe('workflow_complete');
    });

    test('should track node events', () => {
      const monitoring = new MonitoringService();

      monitoring.logNodeStart('test-workflow', 'test-node');
      monitoring.logNodeComplete('test-workflow', 'test-node', 500);

      const events = monitoring.getNodeEvents('test-node');
      expect(events).toHaveLength(2);
      expect(events[0].eventType).toBe('node_start');
      expect(events[1].eventType).toBe('node_complete');
    });

    test('should get performance metrics', () => {
      const monitoring = new MonitoringService();

      const metrics = monitoring.getPerformanceMetrics();
      expect(metrics).toBeDefined();
      expect(typeof metrics.workflowExecutionTime).toBe('number');
    });

    test('should get workflow statistics', () => {
      const monitoring = new MonitoringService();

      const stats = monitoring.getWorkflowStats();
      expect(stats).toBeDefined();
      expect(typeof stats.totalWorkflows).toBe('number');
    });
  });

  describe('Integration and Testing', () => {
    test('should integrate all components together', async () => {
      // Create integration instance
      const integration = RagbitsBubbleLabIntegration.getInstance();

      // Create a simple workflow
      const workflowConfig: BubbleLabWorkflowConfig = {
        id: 'integration-test-workflow',
        name: 'Integration Test Workflow',
        description: 'A workflow for integration testing',
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

      // Create and initialize components
      const engine = integration.createWorkflowEngine(workflowConfig);
      await engine.initialize();

      const processor = integration.createProcessorIntegration();
      await processor.initialize();

      const monitoring = integration.createMonitoringService();
      monitoring.addEventListener((event) => {
        expect(event.workflowId).toBe('integration-test-workflow');
      });

      // Execute workflow
      const results = await engine.executeWorkflow({
        content: 'Integration test content'
      });

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);

      // Verify monitoring captured events
      const workflowEvents = monitoring.getWorkflowEvents('integration-test-workflow');
      expect(workflowEvents).toBeDefined();
      expect(Array.isArray(workflowEvents)).toBe(true);

      // Clean up
      await engine.dispose();
      await processor.dispose();
    });

    test('should generate config and create workflow engine', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();

      const workflowConfig: BubbleLabWorkflowConfig = {
        id: 'config-test-workflow',
        name: 'Config Test Workflow',
        description: 'A workflow for config testing',
        nodes: [
          {
            id: 'ingest-node',
            type: 'ragbits-ingest',
            position: { x: 0, y: 0 },
            data: {
              sourceType: 'file',
              sourcePath: './test.txt'
            }
          }
        ],
        edges: [],
        metadata: {}
      };

      // Generate config
      const generatedConfig = integration.generateConfig(workflowConfig, {
        validate: true,
        targetEnvironment: 'development'
      });

      expect(generatedConfig).toBeDefined();
      expect(generatedConfig.ragbitsConfig).toBeDefined();

      // Create workflow engine with the same config
      const engine = integration.createWorkflowEngine(workflowConfig);
      expect(engine).toBeDefined();
    });

    test('should handle complex workflow with multiple nodes', async () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();

      // Create a more complex workflow
      const complexWorkflow: BubbleLabWorkflowConfig = {
        id: 'complex-workflow',
        name: 'Complex Workflow',
        description: 'A complex workflow with multiple nodes',
        nodes: [
          {
            id: 'ingest-node',
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
          createdBy: 'integration-test',
          createdAt: new Date().toISOString()
        }
      };

      // Create and initialize workflow engine
      const engine = integration.createWorkflowEngine(complexWorkflow);
      await engine.initialize();

      // Execute the workflow
      const results = await engine.executeWorkflow({
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

      // Clean up
      await engine.dispose();
    });
  });
});