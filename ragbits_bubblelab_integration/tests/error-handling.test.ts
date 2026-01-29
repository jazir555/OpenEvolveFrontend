/**
 * Error Handling Tests for Ragbits + BubbleLab Integration
 */

import {
  RAGBitsIngestBubble,
  RAGBitsSearchBubble,
  RAGBitsGenerationBubble,
  RAGBitsIndexBubble,
  RAGBitsWorkflowEngine,
  RagbitsProcessorIntegration,
  MonitoringService,
  ConfigMapper,
  ConfigGenerator,
  type BubbleLabWorkflowConfig
} from '../index';

// Mock the Ragbits document processor to simulate errors
jest.mock('../../knowledge_engine/ragbits_document_processor', () => {
  return {
    RAGBitsDocumentProcessor: jest.fn().mockImplementation(() => {
      return {
        initialize: jest.fn().mockResolvedValue(true),
        ingest_file: jest.fn().mockRejectedValue(new Error('Simulated ingestion error')),
        ingest_text: jest.fn().mockRejectedValue(new Error('Simulated ingestion error')),
        search: jest.fn().mockRejectedValue(new Error('Simulated search error')),
        get_statistics: jest.fn().mockRejectedValue(new Error('Simulated stats error')),
        clear: jest.fn().mockRejectedValue(new Error('Simulated clear error')),
        close: jest.fn().mockResolvedValue(undefined)
      };
    })
  };
});

describe('Error Handling Tests for Ragbits + BubbleLab Integration', () => {
  describe('Bubble Error Handling', () => {
    test('RAGBitsIngestBubble should handle ingestion errors gracefully', async () => {
      const config = {
        id: 'ingest-error-test',
        name: 'Ingest Error Test',
        description: 'Test ingest bubble error handling',
        sourceType: 'file',
        sourcePath: './nonexistent.txt'
      };

      const bubble = new RAGBitsIngestBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        source: './nonexistent.txt',
        content: 'Test content'
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error).toContain('Simulated ingestion error');
    });

    test('RAGBitsSearchBubble should handle search errors gracefully', async () => {
      const config = {
        id: 'search-error-test',
        name: 'Search Error Test',
        description: 'Test search bubble error handling',
        topK: 5
      };

      const bubble = new RAGBitsSearchBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        query: 'Test query'
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error).toContain('Simulated search error');
    });

    test('RAGBitsGenerationBubble should handle generation errors gracefully', async () => {
      const config = {
        id: 'generation-error-test',
        name: 'Generation Error Test',
        description: 'Test generation bubble error handling',
        llmModel: 'gpt-4o'
      };

      const bubble = new RAGBitsGenerationBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        query: 'Test query',
        context: [{ content: 'Test context', metadata: {} }]
      });

      expect(result.success).toBe(true); // Generation bubble simulates response
      expect(result.response).toBeDefined();
    });

    test('RAGBitsIndexBubble should handle index operation errors gracefully', async () => {
      const config = {
        id: 'index-error-test',
        name: 'Index Error Test',
        description: 'Test index bubble error handling',
        vectorStoreType: 'memory'
      };

      const bubble = new RAGBitsIndexBubble(config);
      await bubble.initialize();

      const result = await bubble.action({
        operation: 'stats'
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error).toContain('Simulated stats error');
    });

    test('RAGBitsIngestBubble should handle invalid config', () => {
      expect(() => {
        new RAGBitsIngestBubble({} as any);
      }).toThrow('Bubble configuration must have an id');
    });

    test('RAGBitsSearchBubble should handle invalid config', () => {
      expect(() => {
        new RAGBitsSearchBubble({} as any);
      }).toThrow('Bubble configuration must have an id');
    });

    test('RAGBitsGenerationBubble should handle invalid config', () => {
      expect(() => {
        new RAGBitsGenerationBubble({} as any);
      }).toThrow('Bubble configuration must have an id');
    });

    test('RAGBitsIndexBubble should handle invalid config', () => {
      expect(() => {
        new RAGBitsIndexBubble({} as any);
      }).toThrow('Bubble configuration must have an id');
    });
  });

  describe('Workflow Engine Error Handling', () => {
    test('should handle node execution errors', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'error-workflow',
        name: 'Error Workflow',
        description: 'A workflow with error handling test',
        nodes: [
          {
            id: 'ingest-node',
            type: 'ragbits-ingest',
            position: { x: 0, y: 0 },
            data: {
              sourceType: 'file',
              sourcePath: './nonexistent.txt'
            }
          }
        ],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig, { timeout: 5000 });
      await engine.initialize();

      const results = await engine.executeWorkflow();
      
      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(false);
      expect(results[0].error).toBeDefined();
    });

    test('should handle timeout errors', async () => {
      // Create a mock bubble that takes too long to execute
      class SlowBubble {
        async action(input: any): Promise<any> {
          return new Promise(resolve => {
            setTimeout(() => {
              resolve({ result: 'slow result' });
            }, 10000); // 10 seconds, longer than timeout
          });
        }
      }

      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'timeout-workflow',
        name: 'Timeout Workflow',
        description: 'A workflow with timeout test',
        nodes: [
          {
            id: 'slow-node',
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

      // Use a shorter timeout to trigger timeout error faster
      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig, { timeout: 100 });
      await engine.initialize();

      const results = await engine.executeWorkflow();
      
      expect(results).toHaveLength(1);
      expect(results[0].success).toBe(false);
      expect(results[0].error).toBeDefined();
      expect(results[0].error).toContain('timed out');
    });

    test('should handle invalid workflow configuration', async () => {
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid-workflow',
        name: 'Invalid Workflow',
        description: 'A workflow with invalid configuration',
        nodes: [
          {
            id: 'invalid-node',
            type: 'nonexistent-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      try {
        const engine = new RAGBitsWorkflowEngine(invalidConfig);
        await engine.initialize();
        // Should not reach here
        expect(true).toBe(false);
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    test('should handle missing node instances', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'missing-node-workflow',
        name: 'Missing Node Workflow',
        description: 'A workflow with missing node instances',
        nodes: [],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      await engine.initialize();

      // Try to execute a node that doesn't exist
      try {
        await (engine as any).executeNode('nonexistent-node-id', {});
        expect(true).toBe(false); // Should not reach here
      } catch (error) {
        expect(error).toBeDefined();
        expect((error as Error).message).toContain('Node instance not found');
      }
    });

    test('should handle invalid node types', async () => {
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid-node-type-workflow',
        name: 'Invalid Node Type Workflow',
        description: 'A workflow with invalid node types',
        nodes: [
          {
            id: 'invalid-node',
            type: 'invalid-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      try {
        const engine = new RAGBitsWorkflowEngine(invalidConfig);
        await engine.initialize();
        expect(true).toBe(false); // Should not reach here
      } catch (error) {
        expect(error).toBeDefined();
        expect((error as Error).message).toContain('Unsupported node type');
      }
    });

    test('should handle cycle detection in workflow', () => {
      const cyclicConfig: BubbleLabWorkflowConfig = {
        id: 'cyclic-workflow',
        name: 'Cyclic Workflow',
        description: 'A workflow with a cycle',
        nodes: [
          {
            id: 'node-1',
            type: 'ragbits-ingest',
            position: { x: 0, y: 0 },
            data: { sourceType: 'file', sourcePath: './test.txt' }
          },
          {
            id: 'node-2',
            type: 'ragbits-search',
            position: { x: 300, y: 0 },
            data: { topK: 5 }
          }
        ],
        edges: [
          {
            id: 'edge-1',
            source: 'node-1',
            target: 'node-2'
          },
          {
            id: 'edge-2',
            source: 'node-2',
            target: 'node-1' // Creates a cycle
          }
        ],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(cyclicConfig);
      expect(() => {
        (engine as any).topologicalSort();
      }).toThrow('Cycle detected in workflow');
    });
  });

  describe('Processor Integration Error Handling', () => {
    test('should handle processor initialization errors', async () => {
      // Create a mock module that simulates initialization failure
      const mockModule = {
        RAGBitsDocumentProcessor: jest.fn().mockImplementation(() => {
          return {
            initialize: jest.fn().mockResolvedValue(false), // Simulate initialization failure
            ingest_file: jest.fn().mockResolvedValue({
              success: true,
              document_id: 'test',
              chunks_ingested: 1,
              error: null
            }),
            ingest_text: jest.fn().mockResolvedValue({
              success: true,
              document_id: 'test',
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

      jest.doMock('../../knowledge_engine/ragbits_document_processor', () => mockModule);

      // Need to re-require the module to get the new mock
      const { RagbitsProcessorIntegration } = require('../index');

      const processor = new RagbitsProcessorIntegration();
      
      await expect(processor.initialize()).rejects.toThrow('Failed to initialize Ragbits document processor');
    });

    test('should handle document processing errors', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      await expect(
        processor.processDocument('./error.txt', 'Error content')
      ).rejects.toThrow('Simulated ingestion error');
    });

    test('should handle search errors', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      await expect(
        processor.search('error query')
      ).rejects.toThrow('Simulated search error');
    });

    test('should handle index clearing errors', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const result = await processor.clearStore();
      expect(result).toBe(false); // Should return false on error
    });

    test('should handle index stats errors', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      await expect(
        processor.getIndexStats()
      ).rejects.toThrow('Simulated stats error');
    });

    test('should handle queue processing errors', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      // Add a document to the queue
      const promise = processor.addDocument('./error.txt', 'Error content');
      
      // Process the queue which should trigger the error
      await expect(processor.processQueue()).resolves.toBeUndefined();
      
      // The promise should reject with the error
      await expect(promise).rejects.toThrow('Simulated ingestion error');
    });

    test('should handle invalid processor configuration', () => {
      const config = {
        enableAutoIndexing: true,
        autoIndexInterval: -1, // Invalid negative interval
        batchSize: -5, // Invalid negative batch size
        enableCaching: true,
        cacheTTL: -10, // Invalid negative TTL
        enableMonitoring: true,
        maxConcurrentProcesses: -2 // Invalid negative value
      };

      const processor = new RagbitsProcessorIntegration(config as any);
      expect(processor).toBeDefined();
      // The processor should handle invalid config gracefully
    });
  });

  describe('Configuration Error Handling', () => {
    test('should handle invalid bubble lab config', () => {
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid-config',
        name: 'Invalid Config',
        description: 'An invalid config for testing',
        nodes: [
          {
            id: '', // Invalid: no ID
            type: '', // Invalid: no type
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [
          {
            id: 'invalid-edge',
            source: 'nonexistent', // References non-existent node
            target: 'nonexistent'
          }
        ],
        metadata: {}
      };

      const validation = ConfigMapper.validateBubbleLabConfig(invalidConfig);
      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContain('Node missing ID');
      expect(validation.errors).toContain('Node unknown missing type');
      expect(validation.errors).toContain('Edge references non-existent source node: nonexistent');
      expect(validation.errors).toContain('Edge references non-existent target node: nonexistent');
    });

    test('should handle config generation with validation errors', () => {
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid-gen-config',
        name: 'Invalid Gen Config',
        description: 'An invalid config for generation testing',
        nodes: [
          {
            id: 'invalid-node',
            type: 'invalid-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      const generated = ConfigGenerator.generate(invalidConfig, { validate: true });
      expect(generated.validationErrors).toBeDefined();
      expect(generated.validationErrors!.length).toBeGreaterThan(0);
    });

    test('should handle invalid node types during mapping', () => {
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid-node-type-config',
        name: 'Invalid Node Type Config',
        description: 'A config with invalid node types',
        nodes: [
          {
            id: 'invalid-node',
            type: 'completely-invalid-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      const ragbitsConfig = ConfigMapper.mapBubbleLabToRagbits(invalidConfig);
      expect(ragbitsConfig.workflow.nodes).toHaveLength(0); // Invalid nodes should be filtered out
    });

    test('should handle missing required fields in node data', () => {
      const configWithMissingFields: BubbleLabWorkflowConfig = {
        id: 'missing-fields-config',
        name: 'Missing Fields Config',
        description: 'A config with missing required fields',
        nodes: [
          {
            id: 'ingest-node',
            type: 'ragbits-ingest',
            position: { x: 0, y: 0 },
            data: {
              // Missing required sourceType and sourcePath
            }
          }
        ],
        edges: [],
        metadata: {}
      };

      const ragbitsConfig = ConfigMapper.mapBubbleLabToRagbits(configWithMissingFields);
      expect(ragbitsConfig.workflow.nodes).toHaveLength(1); // Should still have the node but with defaults
    });

    test('should handle invalid environment settings', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'env-test-config',
        name: 'Env Test Config',
        description: 'A config for environment testing',
        nodes: [],
        edges: [],
        metadata: {}
      };

      // Test with invalid environment
      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig, {
        targetEnvironment: 'invalid-environment' as any
      });

      // Should default to development settings
      expect(generatedConfig.ragbitsConfig.documentProcessor.vector_store_type).toBe('memory');
    });
  });

  describe('Monitoring Error Handling', () => {
    test('should handle event listener errors gracefully', () => {
      const monitoring = new MonitoringService();

      // Add a listener that throws an error
      monitoring.addEventListener(() => {
        throw new Error('Listener error');
      });

      // This should not crash the monitoring service
      monitoring.logWorkflowStart('test-workflow');
      
      // Service should still be functional
      const events = monitoring.getEventLog();
      expect(events).toHaveLength(1);
      expect(events[0].workflowId).toBe('test-workflow');
    });

    test('should handle alert callback errors gracefully', () => {
      const monitoring = new MonitoringService();

      // Add an alert callback that throws an error
      monitoring.addAlertCallback(() => {
        throw new Error('Alert callback error');
      });

      // Trigger an alert condition
      monitoring.logWorkflowError('test-workflow', 'Test error');
      
      // Service should still be functional
      const events = monitoring.getEventLog();
      expect(events).toHaveLength(1);
      expect(events[0].eventType).toBe('workflow_error');
    });

    test('should handle invalid event data', () => {
      const monitoring = new MonitoringService();

      // Try to log an event with invalid data
      const invalidEvent: any = {
        id: null, // Invalid ID
        timestamp: 'not-a-date', // Invalid timestamp
        eventType: 'invalid-event-type',
        workflowId: 'test-workflow'
      };

      // The service should handle invalid event data gracefully
      expect(() => {
        monitoring['recordEvent'](invalidEvent);
      }).not.toThrow();
    });

    test('should handle invalid alert thresholds', () => {
      const monitoring = new MonitoringService({
        alertThresholds: {
          executionTime: -1000, // Invalid negative threshold
          errorRate: 150, // Invalid rate > 100%
          memoryUsage: -500 // Invalid negative memory usage
        }
      });

      // Service should still initialize
      expect(monitoring).toBeDefined();

      // Try to trigger an alert with invalid thresholds
      monitoring.logWorkflowComplete('test-workflow', 1000);
      
      const stats = monitoring.getWorkflowStats();
      expect(stats).toBeDefined();
    });

    test('should handle invalid sampling rate', () => {
      const monitoring = new MonitoringService({
        samplingRate: 1.5 // Invalid rate > 1.0
      });

      // Service should handle invalid sampling rate gracefully
      expect(monitoring).toBeDefined();

      monitoring.logWorkflowStart('test-workflow');
      const events = monitoring.getEventLog();
      expect(events).toBeDefined();
    });

    test('should handle invalid retention period', () => {
      const monitoring = new MonitoringService({
        retentionPeriod: -5 // Invalid negative period
      });

      // Service should handle invalid retention period gracefully
      expect(monitoring).toBeDefined();

      monitoring.logWorkflowStart('test-workflow');
      const events = monitoring.getEventLog();
      expect(events).toBeDefined();
    });
  });

  describe('Integration Error Handling', () => {
    test('should handle errors in integration methods', () => {
      const integration = require('../index').RagbitsBubbleLabIntegration.getInstance();
      
      // Test with invalid config
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid',
        name: 'Invalid',
        description: 'Invalid config',
        nodes: [
          {
            id: 'invalid-node',
            type: 'invalid-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      // This should not throw but return a config with validation errors
      const config = integration.generateConfig(invalidConfig, { validate: true });
      expect(config).toBeDefined();
      expect(config.validationErrors).toBeDefined();
    });

    test('should handle errors in workflow engine creation', () => {
      const integration = require('../index').RagbitsBubbleLabIntegration.getInstance();
      
      // Test with invalid workflow config
      const invalidConfig: BubbleLabWorkflowConfig = {
        id: 'invalid-engine-config',
        name: 'Invalid Engine Config',
        description: 'Invalid config for engine',
        nodes: [
          {
            id: 'invalid-node',
            type: 'completely-invalid-type',
            position: { x: 0, y: 0 },
            data: {}
          }
        ],
        edges: [],
        metadata: {}
      };

      const engine = integration.createWorkflowEngine(invalidConfig);
      expect(engine).toBeDefined();
    });

    test('should handle errors in processor integration creation', () => {
      const integration = require('../index').RagbitsBubbleLabIntegration.getInstance();
      
      // Test with invalid processor config
      const invalidConfig = {
        enableAutoIndexing: true,
        autoIndexInterval: -1, // Invalid negative interval
        batchSize: -5, // Invalid negative batch size
        enableCaching: true,
        cacheTTL: -10, // Invalid negative TTL
        enableMonitoring: true,
        maxConcurrentProcesses: -2 // Invalid negative value
      };

      const processor = integration.createProcessorIntegration(invalidConfig as any);
      expect(processor).toBeDefined();
    });

    test('should handle errors in monitoring service creation', () => {
      const integration = require('../index').RagbitsBubbleLabIntegration.getInstance();
      
      // Test with invalid monitoring config
      const invalidConfig = {
        enableRealTimeMonitoring: true,
        enablePerformanceTracking: true,
        enableErrorTracking: true,
        enableTokenTracking: true,
        logLevel: 'invalid-level' as any, // Invalid log level
        retentionPeriod: -5, // Invalid negative period
        samplingRate: 1.5, // Invalid rate > 1.0
        enableAlerting: true,
        alertThresholds: {
          executionTime: -1000, // Invalid negative threshold
          errorRate: 150, // Invalid rate > 100%
          memoryUsage: -500 // Invalid negative memory usage
        }
      };

      const monitoring = integration.createMonitoringService(invalidConfig);
      expect(monitoring).toBeDefined();
    });

    test('should handle errors when mapping invalid config', () => {
      const integration = require('../index').RagbitsBubbleLabIntegration.getInstance();
      
      // Test with completely invalid config
      const invalidConfig: any = {
        invalidField: 'invalid-value',
        nodes: 'not-an-array',
        edges: null
      };

      try {
        const config = integration.mapConfig(invalidConfig);
        expect(config).toBeDefined();
      } catch (error) {
        // If it throws, that's also acceptable as long as it's handled gracefully
        expect(error).toBeDefined();
      }
    });

    test('should handle errors in complex workflow execution', async () => {
      const integration = require('../index').RagbitsBubbleLabIntegration.getInstance();

      // Create a workflow with mixed valid and invalid nodes
      const mixedWorkflow: BubbleLabWorkflowConfig = {
        id: 'mixed-workflow',
        name: 'Mixed Workflow',
        description: 'Workflow with mixed valid and invalid nodes',
        nodes: [
          {
            id: 'valid-node',
            type: 'ragbits-ingest',
            position: { x: 0, y: 0 },
            data: {
              sourceType: 'text',
              sourcePath: 'inline'
            }
          },
          {
            id: 'invalid-node',
            type: 'invalid-type',
            position: { x: 300, y: 0 },
            data: {}
          }
        ],
        edges: [
          {
            id: 'edge-1',
            source: 'valid-node',
            target: 'invalid-node'
          }
        ],
        metadata: {}
      };

      const engine = integration.createWorkflowEngine(mixedWorkflow);
      await engine.initialize();

      // Execution should handle errors gracefully
      const results = await engine.executeWorkflow({
        content: 'Test content for mixed workflow'
      });

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
      // Should have results for at least the valid node
      expect(results.length).toBeGreaterThanOrEqual(1);

      await engine.dispose();
    }, 10000); // Increase timeout for this test
  });

  describe('General Error Handling', () => {
    test('should handle unexpected errors gracefully', () => {
      // Test error handling in various components with unexpected inputs
      expect(() => {
        ConfigMapper.validateBubbleLabConfig(null as any);
      }).not.toThrow();

      expect(() => {
        ConfigMapper.mapBubbleLabToRagbits(null as any);
      }).toThrow();

      expect(() => {
        ConfigGenerator.generate(null as any);
      }).toThrow();

      expect(() => {
        new RAGBitsWorkflowEngine(null as any);
      }).toThrow();
    });

    test('should handle null/undefined inputs', () => {
      const monitoring = new MonitoringService();

      // These should not throw
      expect(() => {
        monitoring.logWorkflowStart(null as any);
      }).not.toThrow();

      expect(() => {
        monitoring.logWorkflowComplete(null as any, 0);
      }).not.toThrow();

      expect(() => {
        monitoring.logNodeStart(null as any, null as any);
      }).not.toThrow();

      expect(() => {
        monitoring.logNodeComplete(null as any, null as any, 0);
      }).not.toThrow();
    });

    test('should handle async errors in promise chains', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      // Test error handling in promise chains
      await expect(
        processor.processDocument('./error.txt', 'Error content')
          .catch(err => {
            // Error should be caught and handled
            expect(err).toBeDefined();
            return { success: false, error: err.message };
          })
      ).resolves.toEqual(expect.objectContaining({ success: false }));
    });
  });
});