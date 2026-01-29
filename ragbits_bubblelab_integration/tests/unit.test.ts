/**
 * Unit Tests for Ragbits + BubbleLab Integration
 */

import {
  BaseBubble,
  RAGBitsIngestBubble,
  RAGBitsSearchBubble,
  RAGBitsGenerationBubble,
  RAGBitsIndexBubble,
  ConfigMapper,
  ConfigGenerator,
  RAGBitsWorkflowEngine,
  RagbitsProcessorIntegration,
  MonitoringService,
  RagbitsBubbleLabIntegration,
  type BubbleLabWorkflowConfig,
  type RAGBitsIngestConfig,
  type RAGBitsSearchConfig,
  type RAGBitsGenerationConfig,
  type RAGBitsIndexConfig
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
        search: jest.fn().mockResolvedValue([]),
        get_statistics: jest.fn().mockResolvedValue({}),
        clear: jest.fn().mockResolvedValue(true),
        close: jest.fn().mockResolvedValue(undefined)
      };
    })
  };
});

describe('Unit Tests for Ragbits + BubbleLab Integration', () => {
  describe('BaseBubble', () => {
    class TestBubble extends BaseBubble<{ id: string; name: string; description: string }, any, any> {
      async action(input: any): Promise<any> {
        return { result: 'test' };
      }
    }

    test('should validate config correctly', () => {
      expect(() => {
        new TestBubble({ id: '', name: 'Test', description: 'Test' });
      }).toThrow('Bubble configuration must have an id');

      expect(() => {
        new TestBubble({ id: 'test', name: '', description: 'Test' });
      }).toThrow('Bubble configuration must have a name');

      const bubble = new TestBubble({ id: 'test', name: 'Test', description: 'Test' });
      expect(bubble).toBeDefined();
    });

    test('should initialize correctly', async () => {
      const bubble = new TestBubble({ id: 'test', name: 'Test', description: 'Test' });
      await bubble.initialize();
      // Should not throw
    });

    test('should dispose correctly', async () => {
      const bubble = new TestBubble({ id: 'test', name: 'Test', description: 'Test' });
      await bubble.dispose();
      // Should not throw
    });
  });

  describe('RAGBitsIngestBubble', () => {
    test('should create instance with valid config', () => {
      const config: RAGBitsIngestConfig = {
        id: 'ingest-test',
        name: 'Test Ingest',
        description: 'Test ingest bubble',
        sourceType: 'file',
        sourcePath: './test.txt',
        chunkSize: 1000,
        chunkOverlap: 200
      };

      const bubble = new RAGBitsIngestBubble(config);
      expect(bubble).toBeDefined();
    });

    test('should execute action successfully', async () => {
      const config: RAGBitsIngestConfig = {
        id: 'ingest-test',
        name: 'Test Ingest',
        description: 'Test ingest bubble',
        sourceType: 'file',
        sourcePath: './test.txt'
      };

      const bubble = new RAGBitsIngestBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        content: 'Test content',
        metadata: { test: true }
      });

      expect(result.success).toBe(true);
      expect(result.documentId).toBeDefined();
    });
  });

  describe('RAGBitsSearchBubble', () => {
    test('should create instance with valid config', () => {
      const config: RAGBitsSearchConfig = {
        id: 'search-test',
        name: 'Test Search',
        description: 'Test search bubble',
        topK: 5,
        scoreThreshold: 0.7
      };

      const bubble = new RAGBitsSearchBubble(config);
      expect(bubble).toBeDefined();
    });

    test('should execute action successfully', async () => {
      const config: RAGBitsSearchConfig = {
        id: 'search-test',
        name: 'Test Search',
        description: 'Test search bubble'
      };

      const bubble = new RAGBitsSearchBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        query: 'Test query',
        topK: 5
      });

      expect(result.success).toBe(true);
      expect(Array.isArray(result.results)).toBe(true);
    });
  });

  describe('RAGBitsGenerationBubble', () => {
    test('should create instance with valid config', () => {
      const config: RAGBitsGenerationConfig = {
        id: 'generation-test',
        name: 'Test Generation',
        description: 'Test generation bubble',
        llmModel: 'gpt-4o'
      };

      const bubble = new RAGBitsGenerationBubble(config);
      expect(bubble).toBeDefined();
    });

    test('should execute action successfully', async () => {
      const config: RAGBitsGenerationConfig = {
        id: 'generation-test',
        name: 'Test Generation',
        description: 'Test generation bubble',
        llmModel: 'gpt-4o'
      };

      const bubble = new RAGBitsGenerationBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        query: 'Test query',
        context: [{ content: 'Test context', metadata: {} }]
      });

      expect(result.success).toBe(true);
      expect(typeof result.response).toBe('string');
    });
  });

  describe('RAGBitsIndexBubble', () => {
    test('should create instance with valid config', () => {
      const config: RAGBitsIndexConfig = {
        id: 'index-test',
        name: 'Test Index',
        description: 'Test index bubble',
        vectorStoreType: 'memory'
      };

      const bubble = new RAGBitsIndexBubble(config);
      expect(bubble).toBeDefined();
    });

    test('should execute action successfully', async () => {
      const config: RAGBitsIndexConfig = {
        id: 'index-test',
        name: 'Test Index',
        description: 'Test index bubble',
        vectorStoreType: 'memory'
      };

      const bubble = new RAGBitsIndexBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        operation: 'stats'
      });

      expect(result.success).toBe(true);
      expect(result.operation).toBe('stats');
    });
  });

  describe('ConfigMapper', () => {
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

    test('should validate bubble lab config correctly', () => {
      const validConfig: BubbleLabWorkflowConfig = {
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

      const validation = ConfigMapper.validateBubbleLabConfig(validConfig);
      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);

      // Test with invalid config
      const invalidConfig = { ...validConfig };
      invalidConfig.nodes[0].id = '';
      const invalidValidation = ConfigMapper.validateBubbleLabConfig(invalidConfig);
      expect(invalidValidation.isValid).toBe(false);
      expect(invalidValidation.errors).toContain('Node missing ID');
    });
  });

  describe('ConfigGenerator', () => {
    test('should generate ragbits config from bubble lab config', () => {
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
      expect(generatedConfig.ragbitsConfig.workflow.name).toBe('Test Workflow');
      expect(generatedConfig.validationErrors).toBeUndefined();
    });

    test('should apply environment settings', () => {
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

      // In production, vector store should be qdrant instead of memory
      expect(generatedConfig.ragbitsConfig.documentProcessor.vector_store_type).toBe('qdrant');
    });

    test('should validate config correctly', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [
          {
            id: 'ingest-node',
            type: 'invalid-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig, {
        validate: true
      });

      expect(generatedConfig.validationErrors).toBeDefined();
      expect(generatedConfig.validationErrors!.length).toBeGreaterThan(0);
    });
  });

  describe('RAGBitsWorkflowEngine', () => {
    test('should create workflow engine instance', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      expect(engine).toBeDefined();
    });

    test('should initialize workflow engine', async () => {
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
      // Should not throw
    });

    test('should execute workflow', async () => {
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
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('RagbitsProcessorIntegration', () => {
    test('should create processor integration instance', () => {
      const processor = new RagbitsProcessorIntegration();
      expect(processor).toBeDefined();
    });

    test('should initialize processor integration', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();
      // Should not throw
    });

    test('should process document', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const result = await processor.processDocument(
        './test.txt',
        'Test content'
      );
      expect(result).toBeDefined();
      expect(result.success).toBe(true);
    });

    test('should search documents', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const results = await processor.search('Test query');
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('MonitoringService', () => {
    test('should create monitoring service instance', () => {
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

    test('should get workflow stats', () => {
      const monitoring = new MonitoringService();

      const stats = monitoring.getWorkflowStats();
      expect(stats).toBeDefined();
      expect(typeof stats.totalWorkflows).toBe('number');
    });
  });

  describe('RagbitsBubbleLabIntegration', () => {
    test('should be a singleton', () => {
      const instance1 = RagbitsBubbleLabIntegration.getInstance();
      const instance2 = RagbitsBubbleLabIntegration.getInstance();

      expect(instance1).toBe(instance2);
    });

    test('should create workflow engine', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const engine = integration.createWorkflowEngine(bubbleLabConfig);
      expect(engine).toBeDefined();
    });

    test('should create processor integration', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const processor = integration.createProcessorIntegration();
      expect(processor).toBeDefined();
    });

    test('should create monitoring service', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const monitoring = integration.createMonitoringService();
      expect(monitoring).toBeDefined();
    });

    test('should generate config', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const config = integration.generateConfig(bubbleLabConfig);
      expect(config).toBeDefined();
    });

    test('should map config', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const config = integration.mapConfig(bubbleLabConfig);
      expect(config).toBeDefined();
    });
  });
});