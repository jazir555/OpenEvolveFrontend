/**
 * Contract Tests for Workflow Orchestrator
 *
 * Tests the workflow execution engine with various scenarios
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { WorkflowOrchestrator, type WorkflowDefinition } from '../../lib/workflow-orchestrator';
import { PluginRegistry, type PluginInterface } from '../../lib/plugin-registry';

// Mock plugin for testing
class MockPlugin implements PluginInterface {
  metadata = {
    name: 'mock-plugin',
    version: '1.0.0',
    description: 'Mock plugin for testing',
    author: 'Test',
    enabled: true
  };

  capabilities = {
    processing: true
  };

  status: 'idle' | 'initializing' | 'ready' | 'busy' | 'error' = 'idle';

  async initialize(): Promise<void> {
    this.status = 'ready';
  }

  async updateConfig(): Promise<void> {
    // Mock implementation
  }

  async resetConfig(): Promise<void> {
    // Mock implementation
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }

  getContext() {
    return {
      config: {},
      state: {}
    };
  }

  getStatus() {
    return this.status;
  }

  async destroy(): Promise<void> {
    this.status = 'idle';
  }

  // Custom actions
  async processData(input: { data: string }): Promise<{ result: string }> {
    this.status = 'busy';
    const result = { result: `processed: ${input.data}` };
    this.status = 'ready';
    return result;
  }

  async transformData(input: { value: number }): Promise<{ doubled: number }> {
    this.status = 'busy';
    const result = { doubled: input.value * 2 };
    this.status = 'ready';
    return result;
  }

  async failAction(): Promise<never> {
    this.status = 'busy';
    this.status = 'error';
    throw new Error('Intentional failure');
  }
}

describe('Workflow Orchestrator', () => {
  let orchestrator: WorkflowOrchestrator;
  let registry: PluginRegistry;
  let mockPlugin: MockPlugin;

  beforeEach(async () => {
    registry = new PluginRegistry({ autoInitialize: false });
    orchestrator = new WorkflowOrchestrator(registry);
    mockPlugin = new MockPlugin();
    await registry.registerPlugin(mockPlugin);
  });

  afterEach(async () => {
    await registry.destroy();
  });

  describe('Workflow Validation', () => {
    it('should validate a correct workflow', () => {
      const workflow: WorkflowDefinition = {
        id: 'test-workflow',
        name: 'Test Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Step 1',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' }
          }
        ]
      };

      const result = orchestrator.validateWorkflow(workflow);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject workflow without ID', () => {
      const workflow: WorkflowDefinition = {
        id: '',
        name: 'Test Workflow',
        steps: []
      };

      const result = orchestrator.validateWorkflow(workflow);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Workflow ID is required');
    });

    it('should reject workflow without steps', () => {
      const workflow: WorkflowDefinition = {
        id: 'test',
        name: 'Test Workflow',
        steps: []
      };

      const result = orchestrator.validateWorkflow(workflow);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Workflow must have at least one step');
    });

    it('should reject workflow with circular dependencies', async () => {
      const workflow: WorkflowDefinition = {
        id: 'test',
        name: 'Test Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Step 1',
            plugin: 'mock-plugin',
            action: 'processData',
            input: {},
            dependsOn: ['step2']
          },
          {
            id: 'step2',
            name: 'Step 2',
            plugin: 'mock-plugin',
            action: 'processData',
            input: {},
            dependsOn: ['step1']
          }
        ]
      };

      await expect(orchestrator.executeWorkflow(workflow)).rejects.toThrow('Circular dependency');
    });
  });

  describe('Workflow Execution', () => {
    it('should execute a simple workflow', async () => {
      const workflow: WorkflowDefinition = {
        id: 'simple-workflow',
        name: 'Simple Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Process Data',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'hello' },
            outputMapping: { result: 'output' }
          }
        ]
      };

      const result = await orchestrator.executeWorkflow(workflow);

      expect(result.status).toBe('completed');
      expect(result.executionId).toBeDefined();
      expect(result.stepResults.size).toBe(1);
      expect(result.results).toHaveProperty('output');
    });

    it('should execute workflow with dependencies', async () => {
      const workflow: WorkflowDefinition = {
        id: 'dependent-workflow',
        name: 'Dependent Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Process Data',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'first' }
          },
          {
            id: 'step2',
            name: 'Transform Data',
            plugin: 'mock-plugin',
            action: 'transformData',
            input: { value: 10 },
            dependsOn: ['step1']
          }
        ]
      };

      const result = await orchestrator.executeWorkflow(workflow);

      expect(result.status).toBe('completed');
      expect(result.stepResults.size).toBe(2);
    });

    it('should handle workflow errors with stop policy', async () => {
      const workflow: WorkflowDefinition = {
        id: 'failing-workflow',
        name: 'Failing Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Process Data',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' }
          },
          {
            id: 'step2',
            name: 'Fail Step',
            plugin: 'mock-plugin',
            action: 'failAction',
            input: {}
          },
          {
            id: 'step3',
            name: 'Should Not Execute',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' }
          }
        ],
        onError: 'stop'
      };

      const result = await orchestrator.executeWorkflow(workflow);

      expect(result.status).toBe('failed');
      expect(result.errors).toHaveLength(1);
      expect(result.errors[0].stepId).toBe('step2');
      expect(result.stepResults.size).toBe(1); // Only step1 completed
    });

    it('should handle workflow errors with continue policy', async () => {
      const workflow: WorkflowDefinition = {
        id: 'continue-workflow',
        name: 'Continue on Error',
        steps: [
          {
            id: 'step1',
            name: 'Process Data',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' }
          },
          {
            id: 'step2',
            name: 'Fail Step',
            plugin: 'mock-plugin',
            action: 'failAction',
            input: {}
          },
          {
            id: 'step3',
            name: 'Execute Anyway',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' }
          }
        ],
        onError: 'continue'
      };

      const result = await orchestrator.executeWorkflow(workflow);

      expect(result.status).toBe('failed'); // A failed step means the workflow failed
      expect(result.errors).toHaveLength(1);
      expect(result.stepResults.size).toBe(2); // step1 and step3
    });

    it('should execute steps in topological order', async () => {
      const executionOrder: string[] = [];

      // Track execution order by modifying the mock plugin
      const originalProcess = mockPlugin.processData.bind(mockPlugin);
      mockPlugin.processData = async (input) => {
        executionOrder.push('process');
        return originalProcess(input);
      };

      const originalTransform = mockPlugin.transformData.bind(mockPlugin);
      mockPlugin.transformData = async (input) => {
        executionOrder.push('transform');
        return originalTransform(input);
      };

      const workflow: WorkflowDefinition = {
        id: 'order-workflow',
        name: 'Order Test',
        steps: [
          {
            id: 'step3',
            name: 'Transform',
            plugin: 'mock-plugin',
            action: 'transformData',
            input: { value: 5 },
            dependsOn: ['step1', 'step2']
          },
          {
            id: 'step1',
            name: 'Process 1',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'a' }
          },
          {
            id: 'step2',
            name: 'Process 2',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'b' }
          }
        ]
      };

      await orchestrator.executeWorkflow(workflow);

      // step1 and step2 should execute before step3
      expect(executionOrder).toEqual(['process', 'process', 'transform']);
    });
  });

  describe('Workflow Cancellation', () => {
    it('should cancel a running workflow', async () => {
      // Create a workflow with delays
      let shouldBlock = true;

      const workflow: WorkflowDefinition = {
        id: 'long-workflow',
        name: 'Long Running Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Block Step',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' }
          }
        ]
      };

      // Start execution
      const executionPromise = orchestrator.executeWorkflow(workflow);

      // Get active workflows
      const activeWorkflows = orchestrator.getActiveWorkflows();
      expect(activeWorkflows.length).toBeGreaterThan(0);

      // Cancel the workflow
      const cancelled = await orchestrator.cancelWorkflow(activeWorkflows[0].executionId);
      expect(cancelled).toBe(true);
    });
  });

  describe('Output Mapping', () => {
    it('should map step outputs to workflow output', async () => {
      const workflow: WorkflowDefinition = {
        id: 'mapping-workflow',
        name: 'Mapping Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Process Data',
            plugin: 'mock-plugin',
            action: 'processData',
            input: { data: 'test' },
            outputMapping: { result: 'firstOutput' }
          },
          {
            id: 'step2',
            name: 'Transform Data',
            plugin: 'mock-plugin',
            action: 'transformData',
            input: { value: 10 },
            outputMapping: { doubled: 'secondOutput' }
          }
        ]
      };

      const result = await orchestrator.executeWorkflow(workflow);

      expect(result.results).toHaveProperty('firstOutput');
      expect(result.results).toHaveProperty('secondOutput');
      expect(result.results.firstOutput).toContain('processed: test');
      expect(result.results.secondOutput).toBe(20);
    });
  });
});
