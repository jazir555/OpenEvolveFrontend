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

describe('Ragbits + BubbleLab Integration', () => {
  describe('Core Components', () => {
    test('should create integration instance', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      expect(integration).toBeDefined();
    });

    test('should create bubble instances', () => {
      const ingestBubble = new RAGBitsIngestBubble({
        id: 'test',
        name: 'Test Ingest',
        description: 'Test ingest bubble',
        sourceType: 'file',
        sourcePath: './test.txt'
      });
      expect(ingestBubble).toBeDefined();

      const searchBubble = new RAGBitsSearchBubble({
        id: 'test',
        name: 'Test Search',
        description: 'Test search bubble',
        topK: 5
      });
      expect(searchBubble).toBeDefined();

      const generationBubble = new RAGBitsGenerationBubble({
        id: 'test',
        name: 'Test Generation',
        description: 'Test generation bubble',
        llmModel: 'gpt-4o'
      });
      expect(generationBubble).toBeDefined();

      const indexBubble = new RAGBitsIndexBubble({
        id: 'test',
        name: 'Test Index',
        description: 'Test index bubble',
        vectorStoreType: 'memory'
      });
      expect(indexBubble).toBeDefined();
    });

    test('should create processor integration', () => {
      const processorIntegration = new RagbitsProcessorIntegration();
      expect(processorIntegration).toBeDefined();
    });

    test('should create monitoring service', () => {
      const monitoringService = new MonitoringService();
      expect(monitoringService).toBeDefined();
    });
  });

  describe('Configuration Mapping', () => {
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
  });

  describe('Configuration Generation', () => {
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
        targetEnvironment: 'development'
      });

      expect(generatedConfig).toBeDefined();
      expect(generatedConfig.ragbitsConfig).toBeDefined();
      expect(generatedConfig.ragbitsConfig.workflow.name).toBe('Test Workflow');
      expect(generatedConfig.validationErrors).toBeUndefined();
    });
  });

  describe('Workflow Engine', () => {
    test('should create workflow engine', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'test-workflow',
        name: 'Test Workflow',
        description: 'A test workflow',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const workflowEngine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      expect(workflowEngine).toBeDefined();
    });
  });

  describe('Monitoring Service', () => {
    test('should create monitoring service with config', () => {
      const monitoringService = new MonitoringService({
        enableRealTimeMonitoring: true,
        logLevel: 'debug',
        retentionPeriod: 14
      });
      expect(monitoringService).toBeDefined();
    });

    test('should track workflow events', () => {
      const monitoringService = new MonitoringService();
      
      monitoringService.logWorkflowStart('test-workflow');
      monitoringService.logWorkflowComplete('test-workflow', 1000);
      
      const events = monitoringService.getWorkflowEvents('test-workflow');
      expect(events).toHaveLength(2);
      expect(events[0].eventType).toBe('workflow_start');
      expect(events[1].eventType).toBe('workflow_complete');
    });
  });
});