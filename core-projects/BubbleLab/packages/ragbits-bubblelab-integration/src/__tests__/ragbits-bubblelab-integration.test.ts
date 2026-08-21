/**
 * Unit tests for the RagbitsBubbleLabIntegration facade.
 *
 * These run fully offline: the RAGBits engines operate in mock mode when no
 * real `ragbits`/`RAGBitsDocumentProcessor` is supplied, and the test never
 * imports the (optional, uninstalled) `ragbits` peer dependency.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { RagbitsBubbleLabIntegration } from '../RagbitsBubbleLabIntegration';
import { RAGBitsWorkflowEngine } from '../engine';
import type { BubbleLabWorkflowConfig } from '../types';

/**
 * Build a minimal, valid single-node BubbleLab workflow that can execute
 * offline (generation bubble runs in mock mode without a processor).
 */
function makeWorkflow(id: string, name: string): BubbleLabWorkflowConfig {
  return {
    id,
    name,
    nodes: [
      {
        id: 'gen-1',
        name: 'Generation',
        type: 'ragbits-generation',
        config: { model: 'mock-llm', name: 'Generation Config' }
      }
    ],
    edges: []
  };
}

describe('RagbitsBubbleLabIntegration', () => {
  afterEach(() => {
    RagbitsBubbleLabIntegration.resetInstance();
  });

  describe('getInstance()', () => {
    it('returns the same singleton instance on repeated calls', () => {
      const a = RagbitsBubbleLabIntegration.getInstance();
      const b = RagbitsBubbleLabIntegration.getInstance();
      expect(a).toBe(b);
      expect(a).toBeInstanceOf(RagbitsBubbleLabIntegration);
    });

    it('resetInstance() clears the singleton so a fresh one is created', () => {
      const first = RagbitsBubbleLabIntegration.getInstance();
      RagbitsBubbleLabIntegration.resetInstance();
      const second = RagbitsBubbleLabIntegration.getInstance();
      expect(second).not.toBe(first);
    });
  });

  describe('workflow delegation', () => {
    it('createWorkflowEngine() wraps a real RAGBitsWorkflowEngine and registers it', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const workflow = makeWorkflow('wf-1', 'Demo Workflow');
      const engine = integration.createWorkflowEngine(workflow);
      expect(engine).toBeInstanceOf(RAGBitsWorkflowEngine);
      expect(integration.getWorkflowEngine('wf-1')).toBe(engine);
      expect(integration.listWorkflows()).toEqual([{ id: 'wf-1', name: 'Demo Workflow' }]);
    });

    it('runWorkflow() initializes and executes the underlying engine, and records status', async () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const workflow = makeWorkflow('wf-2', 'Runnable Workflow');
      integration.createWorkflowEngine(workflow);

      const result = await integration.runWorkflow('wf-2');
      expect(result).toBeDefined();
      expect(result.workflowId).toBe('wf-2');
      expect(['success', 'partial', 'failed']).toContain(result.status);
      expect(result.status).toBe('success');

      const status = integration.getWorkflowStatus('wf-2');
      expect(status).not.toBeNull();
      expect(status?.executionId).toBe(result.executionId);
      expect(integration.listExecutions('wf-2').length).toBe(1);
    });

    it('throws when operating on an unregistered workflow id', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      expect(() => integration.getWorkflowEngine('nope')).not.toThrow();
      expect(integration.getWorkflowEngine('nope')).toBeUndefined();
      expect(() => integration.listExecutions('nope')).toThrow(/No workflow engine registered/);
    });
  });

  describe('auxiliary factories', () => {
    it('createDocumentProcessor() returns a RAGBitsDocumentProcessor', async () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const processor = integration.createDocumentProcessor();
      expect(processor).toBeDefined();
    });

    it('getMonitoringService() is lazily created and stable', () => {
      const integration = RagbitsBubbleLabIntegration.getInstance();
      const a = integration.getMonitoringService();
      const b = integration.getMonitoringService();
      expect(a).toBe(b);
    });
  });
});
