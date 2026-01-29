/**
 * Performance Tests for Ragbits + BubbleLab Integration
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

describe('Performance Tests for Ragbits + BubbleLab Integration', () => {
  describe('Workflow Execution Performance', () => {
    test('should execute simple workflow efficiently', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'simple-workflow',
        name: 'Simple Workflow',
        description: 'A simple workflow for performance testing',
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
          }
        ],
        edges: [],
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(bubbleLabConfig);
      await engine.initialize();

      const startTime = Date.now();
      const results = await engine.executeWorkflow({ content: 'Test content for performance' });
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Execution should be reasonably fast (under 5 seconds for simple workflow)
      expect(executionTime).toBeLessThan(5000);
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    }, 10000); // Increase timeout for this test

    test('should handle concurrent workflow executions', async () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'concurrent-workflow',
        name: 'Concurrent Workflow',
        description: 'A workflow for concurrent execution testing',
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

      // Create multiple engines to simulate concurrent execution
      const engines = Array.from({ length: 5 }, () => new RAGBitsWorkflowEngine(bubbleLabConfig));
      await Promise.all(engines.map(engine => engine.initialize()));

      const startTime = Date.now();
      const results = await Promise.all(
        engines.map(engine => engine.executeWorkflow({ content: 'Test content' }))
      );
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // All 5 concurrent executions should complete in reasonable time
      expect(executionTime).toBeLessThan(10000); // Less than 10 seconds for 5 concurrent
      expect(results).toHaveLength(5);
    }, 15000); // Increase timeout for this test

    test('should execute complex workflow efficiently', async () => {
      const complexWorkflow: BubbleLabWorkflowConfig = {
        id: 'complex-workflow',
        name: 'Complex Workflow',
        description: 'A complex workflow for performance testing',
        nodes: Array.from({ length: 10 }, (_, i) => ({
          id: `node-${i}`,
          type: i % 4 === 0 ? 'ragbits-ingest' : 
                i % 4 === 1 ? 'ragbits-search' : 
                i % 4 === 2 ? 'ragbits-generation' : 'ragbits-index',
          position: { x: i * 100, y: 0 },
          data: i % 4 === 0 ? { sourceType: 'file', sourcePath: `./test${i}.txt` } :
                i % 4 === 1 ? { topK: 5, scoreThreshold: 0.7 } :
                i % 4 === 2 ? { llmModel: 'gpt-4o' } :
                { vectorStoreType: 'memory' }
        })),
        edges: Array.from({ length: 9 }, (_, i) => ({
          id: `edge-${i}`,
          source: `node-${i}`,
          target: `node-${i + 1}`
        })),
        metadata: {}
      };

      const engine = new RAGBitsWorkflowEngine(complexWorkflow);
      await engine.initialize();

      const startTime = Date.now();
      const results = await engine.executeWorkflow({ query: 'Performance test query' });
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Execution of complex workflow should be reasonable
      expect(executionTime).toBeLessThan(15000); // Less than 15 seconds for 10-node workflow
      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    }, 20000); // Increase timeout for this test
  });

  describe('Configuration Generation Performance', () => {
    test('should generate config efficiently for simple workflow', () => {
      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'simple-config-workflow',
        name: 'Simple Config Workflow',
        description: 'A simple workflow for config generation performance testing',
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

      const startTime = Date.now();
      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig);
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Config generation should be very fast
      expect(executionTime).toBeLessThan(1000); // Less than 1 second
      expect(generatedConfig).toBeDefined();
    });

    test('should generate config efficiently for complex workflow', () => {
      // Create a more complex workflow with multiple nodes and connections
      const complexNodes = Array.from({ length: 20 }, (_, i) => ({
        id: `node-${i}`,
        type: i % 4 === 0 ? 'ragbits-ingest' : 
              i % 4 === 1 ? 'ragbits-search' : 
              i % 4 === 2 ? 'ragbits-generation' : 'ragbits-index',
        position: { x: i * 100, y: 0 },
        data: i % 4 === 0 ? { sourceType: 'file', sourcePath: `./test${i}.txt` } :
              i % 4 === 1 ? { topK: 5, scoreThreshold: 0.7 } :
              i % 4 === 2 ? { llmModel: 'gpt-4o' } :
              { vectorStoreType: 'memory' }
      }));

      const complexEdges = Array.from({ length: 19 }, (_, i) => ({
        id: `edge-${i}`,
        source: `node-${i}`,
        target: `node-${i + 1}`
      }));

      const bubbleLabConfig: BubbleLabWorkflowConfig = {
        id: 'complex-config-workflow',
        name: 'Complex Config Workflow',
        description: 'A complex workflow for config generation performance testing',
        nodes: complexNodes,
        edges: complexEdges,
        metadata: {}
      };

      const startTime = Date.now();
      const generatedConfig = ConfigGenerator.generate(bubbleLabConfig);
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Even complex config generation should be fast
      expect(executionTime).toBeLessThan(2000); // Less than 2 seconds for 20 nodes
      expect(generatedConfig).toBeDefined();
    });
  });

  describe('Processor Integration Performance', () => {
    test('should process documents efficiently', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const startTime = Date.now();
      const result = await processor.processDocument(
        './performance-test.txt',
        'This is a test document for performance evaluation. '.repeat(100)
      );
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Document processing should be reasonably fast
      expect(executionTime).toBeLessThan(5000); // Less than 5 seconds
      expect(result).toBeDefined();
      expect(result.success).toBe(true);
    }, 10000); // Increase timeout for this test

    test('should handle multiple concurrent document processing', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const documents = Array.from({ length: 10 }, (_, i) => ({
        source: `./doc-${i}.txt`,
        content: `Document ${i} content for performance testing. `.repeat(50)
      }));

      const startTime = Date.now();
      const results = await Promise.all(
        documents.map(doc => processor.processDocument(doc.source, doc.content))
      );
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Processing 10 documents concurrently should be reasonable
      expect(executionTime).toBeLessThan(15000); // Less than 15 seconds for 10 docs
      expect(results).toHaveLength(10);
      expect(results.every(r => r.success)).toBe(true);
    }, 20000); // Increase timeout for this test

    test('should search efficiently', async () => {
      const processor = new RagbitsProcessorIntegration();
      await processor.initialize();

      const startTime = Date.now();
      const results = await processor.search('performance test query', 10);
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Search should be fast even with mock
      expect(executionTime).toBeLessThan(2000); // Less than 2 seconds
      expect(Array.isArray(results)).toBe(true);
    });
  });

  describe('Monitoring Performance', () => {
    test('should track events efficiently', () => {
      const monitoring = new MonitoringService();

      const startTime = Date.now();
      
      // Simulate tracking many events
      for (let i = 0; i < 1000; i++) {
        monitoring.logWorkflowStart(`workflow-${i}`);
        monitoring.logWorkflowComplete(`workflow-${i}`, Math.floor(Math.random() * 1000));
      }

      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Tracking 1000 events should be fast
      expect(executionTime).toBeLessThan(2000); // Less than 2 seconds for 1000 events
      
      const stats = monitoring.getWorkflowStats();
      expect(stats.totalWorkflows).toBeGreaterThanOrEqual(1000);
    });

    test('should get metrics efficiently', () => {
      const monitoring = new MonitoringService();

      // Add some events to have data
      monitoring.logWorkflowStart('test-workflow');
      monitoring.logWorkflowComplete('test-workflow', 500);

      const startTime = Date.now();
      const metrics = monitoring.getPerformanceMetrics();
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Getting metrics should be very fast
      expect(executionTime).toBeLessThan(100); // Less than 100ms
      expect(metrics).toBeDefined();
    });

    test('should handle high-frequency event logging', () => {
      const monitoring = new MonitoringService();

      const startTime = Date.now();
      
      // Simulate high-frequency event logging
      for (let i = 0; i < 5000; i++) {
        monitoring.logNodeStart('high-freq-workflow', `node-${i % 10}`);
        monitoring.logNodeComplete('high-freq-workflow', `node-${i % 10}`, Math.floor(Math.random() * 100));
      }

      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Tracking 5000 events should be efficient
      expect(executionTime).toBeLessThan(5000); // Less than 5 seconds for 5000 events
      
      const stats = monitoring.getWorkflowStats();
      expect(stats.totalWorkflows).toBeGreaterThanOrEqual(1);
    }, 10000); // Increase timeout for this test
  });

  describe('Memory Usage Tests', () => {
    test('should not have significant memory leaks during repeated operations', () => {
      // Capture initial memory usage if available
      const initialMemory = (global as any).gc ? process.memoryUsage().heapUsed : 0;

      // Perform many operations
      for (let i = 0; i < 100; i++) {
        const bubbleLabConfig: BubbleLabWorkflowConfig = {
          id: `test-workflow-${i}`,
          name: `Test Workflow ${i}`,
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

        const generatedConfig = ConfigGenerator.generate(bubbleLabConfig);
        expect(generatedConfig).toBeDefined();
      }

      // Allow garbage collection
      if ((global as any).gc) {
        (global as any).gc();
        const finalMemory = process.memoryUsage().heapUsed;
        
        // Memory increase should be reasonable (less than 50MB for 100 operations)
        const memoryIncrease = finalMemory - initialMemory;
        expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024); // 50MB
      }
    });

    test('should handle large workflow configurations efficiently', () => {
      // Create a large workflow configuration
      const largeWorkflow: BubbleLabWorkflowConfig = {
        id: 'large-workflow',
        name: 'Large Workflow',
        description: 'A large workflow with many nodes',
        nodes: Array.from({ length: 50 }, (_, i) => ({
          id: `large-node-${i}`,
          type: i % 4 === 0 ? 'ragbits-ingest' : 
                i % 4 === 1 ? 'ragbits-search' : 
                i % 4 === 2 ? 'ragbits-generation' : 'ragbits-index',
          position: { x: i * 50, y: 0 },
          data: i % 4 === 0 ? { 
            sourceType: 'file', 
            sourcePath: `./large-test-${i}.txt`,
            metadata: { large: true, index: i }
          } :
          i % 4 === 1 ? { 
            topK: 5, 
            scoreThreshold: 0.7,
            defaultFilters: { category: `category-${i % 5}` }
          } :
          i % 4 === 2 ? { 
            llmModel: 'gpt-4o',
            temperature: 0.7,
            maxTokens: 1000,
            systemPrompt: `System prompt for node ${i}`
          } :
          { 
            vectorStoreType: 'memory',
            embeddingModel: 'text-embedding-3-small',
            autoRefresh: i % 10 === 0 // Every 10th node has auto-refresh enabled
          }
        })),
        edges: Array.from({ length: 49 }, (_, i) => ({
          id: `large-edge-${i}`,
          source: `large-node-${i}`,
          target: `large-node-${i + 1}`
        })),
        metadata: {
          large: true,
          nodeCount: 50,
          edgeCount: 49,
          generatedAt: new Date().toISOString()
        }
      };

      const startTime = Date.now();
      const generatedConfig = ConfigGenerator.generate(largeWorkflow);
      const endTime = Date.now();

      const executionTime = endTime - startTime;
      
      // Generation of large workflow should be efficient
      expect(executionTime).toBeLessThan(5000); // Less than 5 seconds for 50 nodes
      expect(generatedConfig).toBeDefined();
      expect(generatedConfig.ragbitsConfig.workflow.nodes).toHaveLength(50);
    }, 10000); // Increase timeout for this test
  });

  describe('Integration Performance', () => {
    test('should maintain performance with full integration', async () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();

      // Create a moderately complex workflow
      const workflowConfig: BubbleLabWorkflowConfig = {
        id: 'integration-performance-test',
        name: 'Integration Performance Test',
        description: 'A workflow for integration performance testing',
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
              metadata: { source: 'integration-performance-test' }
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
          performanceTest: true,
          timestamp: new Date().toISOString()
        }
      };

      // Create all components
      const engine = integration.createWorkflowEngine(workflowConfig);
      await engine.initialize();

      const processor = integration.createProcessorIntegration();
      await processor.initialize();

      const monitoring = integration.createMonitoringService();
      let eventCount = 0;
      monitoring.addEventListener(() => {
        eventCount++;
      });

      // Execute workflow multiple times to test sustained performance
      const executionTimes: number[] = [];
      for (let i = 0; i < 10; i++) {
        const startTime = Date.now();
        const results = await engine.executeWorkflow({
          query: `Performance test query ${i}`,
          context: 'Performance testing context'
        });
        const endTime = Date.now();
        
        executionTimes.push(endTime - startTime);
        expect(results).toBeDefined();
      }

      // Calculate average execution time
      const avgExecutionTime = executionTimes.reduce((sum, time) => sum + time, 0) / executionTimes.length;
      
      // Average execution time should be reasonable
      expect(avgExecutionTime).toBeLessThan(3000); // Less than 3 seconds average
      
      // Should have captured events
      expect(eventCount).toBeGreaterThan(0);

      // Clean up
      await engine.dispose();
      await processor.dispose();
    }, 30000); // Increase timeout for this test
  });
});