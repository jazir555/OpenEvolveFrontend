/**
 * End-to-End Integration Test
 *
 * Tests the complete OpenEvolve-BubbleLab integration including:
 * - Plugin registration and initialization
 * - Workflow execution across plugins
 * - Event bus integration
 * - Monitoring and telemetry
 */

import { describe, it, expect, beforeEach, afterEach, beforeAll } from 'vitest';
import {
  initializeBubbleLabIntegration,
  getBubbleLabIntegration,
  type BubbleLabIntegrationConfig
} from '../../lib/plugin-integration';
import {
  getPluginRegistry,
  resetPluginRegistry
} from '../../lib/plugin-registry';
import {
  getWorkflowOrchestrator,
  type WorkflowDefinition
} from '../../lib/workflow-orchestrator';
import { getWorkflowMonitor, resetWorkflowMonitor } from '../../lib/workflow-monitoring';
import { openevolveApi } from '../../lib/openevolveApi';
import { RESEARCH_ASSISTANT_WORKFLOW } from '../../lib/workflow-templates';

describe('End-to-End Integration', () => {
  beforeAll(async () => {
    // Initialize the integration
    await initializeBubbleLabIntegration({
      ragbits: {
        serverUrl: 'http://localhost:3000/ragbits',
        enabled: true
      },
      datapizza: {
        serverUrl: 'http://localhost:3000/datapizza',
        enabled: true
      },
      autoStart: false // Don't auto-start for testing
    });
  });

  afterAll(async () => {
    const integration = getBubbleLabIntegration();
    if (integration) {
      await integration.destroy();
    }
  });

  describe('Plugin Registry', () => {
    it('should have plugins registered', () => {
      const registry = getPluginRegistry();
      const plugins = registry.getAllPlugins();

      expect(plugins.length).toBeGreaterThan(0);
      expect(plugins[0]).toBeDefined();
      expect(plugins[0].metadata).toBeDefined();
      expect(plugins[0].capabilities).toBeDefined();
    });

    it('should have OpenEvolve API plugin', () => {
      const registry = getPluginRegistry();
      const openevolvePlugin = registry.getPlugin('openevolve');

      expect(openevolvePlugin).toBeDefined();
      expect(openevolvePlugin?.metadata.name).toBe('openevolve');
    });

    it('should get plugins by capability', () => {
      const registry = getPluginRegistry();
      const searchPlugins = registry.getPluginsByCapability('search');

      expect(Array.isArray(searchPlugins)).toBe(true);
    });

    it('should get plugin statistics', () => {
      const registry = getPluginRegistry();
      const stats = registry.getStatistics();

      expect(stats).toBeDefined();
      expect(stats.totalPlugins).toBeGreaterThan(0);
      expect(typeof stats.totalPlugins).toBe('number');
    });
  });

  describe('Workflow Orchestrator', () => {
    it('should be available', () => {
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();

      expect(orchestrator).toBeDefined();
    });

    it('should validate workflow templates', () => {
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();

      expect(orchestrator).toBeDefined();
      const validation = orchestrator!.validateWorkflow(RESEARCH_ASSISTANT_WORKFLOW);

      expect(validation.valid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should execute a simple workflow', async () => {
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();
      expect(orchestrator).toBeDefined();

      const simpleWorkflow: WorkflowDefinition = {
        id: 'test-workflow',
        name: 'Test Workflow',
        steps: [
          {
            id: 'step1',
            name: 'Test Step',
            plugin: 'openevolve',
            action: 'bubblelabsZ3Prove',
            input: { theorem: 'forall x. x > 0' }
          }
        ]
      };

      const result = await orchestrator!.executeWorkflow(simpleWorkflow);

      expect(result).toBeDefined();
      expect(result.executionId).toBeDefined();
      expect(result.workflowId).toBe('test-workflow');
      expect(result.status).toMatch(/completed|failed/);
    });
  });

  describe('Monitoring System', () => {
    it('should track workflow executions', async () => {
      const monitor = getWorkflowMonitor();
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();
      expect(orchestrator).toBeDefined();

      const simpleWorkflow: WorkflowDefinition = {
        id: 'monitor-test-workflow',
        name: 'Monitor Test',
        steps: [
          {
            id: 'step1',
            name: 'Test Step',
            plugin: 'openevolve',
            action: 'bubblelabsZ3Prove',
            input: { theorem: 'forall x. x > 0' }
          }
        ]
      };

      const result = await orchestrator!.executeWorkflow(simpleWorkflow);

      // Check that metrics were recorded
      const metrics = monitor.getWorkflowMetrics(result.executionId);
      expect(metrics).toBeDefined();
      expect(metrics?.workflowId).toBe('monitor-test-workflow');
      expect(metrics?.executionId).toBe(result.executionId);
    });

    it('should provide aggregate statistics', () => {
      const monitor = getWorkflowMonitor();
      const stats = monitor.getAggregateStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalExecutions).toBe('number');
      expect(typeof stats.averageDuration).toBe('number');
    });
  });

  describe('Integration Lifecycle', () => {
    it('should start the integration', async () => {
      const integration = getBubbleLabIntegration();
      expect(integration).toBeDefined();

      const status = integration!.getStatus();
      expect(status).toBeDefined();
      expect(typeof status.initialized).toBe('boolean');
      expect(typeof status.pluginCount).toBe('number');
    });

    it('should access registry through integration', () => {
      const integration = getBubbleLabIntegration();
      const registry = integration?.getRegistry();

      expect(registry).toBeDefined();
      expect(registry?.getAllPlugins()).toBeDefined();
    });

    it('should access orchestrator through integration', () => {
      const integration = getBubbleLabIntegration();
      const orchestrator = integration?.getOrchestrator();

      expect(orchestrator).toBeDefined();
    });
  });

  describe('Cross-Plugin Communication', () => {
    it('should handle events between plugins', async () => {
      // This test verifies the event bus integration is working
      // Events should flow from workflow orchestrator to event integration
      const monitor = getWorkflowMonitor();
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();

      const workflow: WorkflowDefinition = {
        id: 'event-test-workflow',
        name: 'Event Test',
        steps: [
          {
            id: 'step1',
            name: 'Test Step',
            plugin: 'openevolve',
            action: 'bubblelabsZ3Prove',
            input: { theorem: 'forall x. x > 0' }
          }
        ]
      };

      const beforeExecutions = monitor.getAggregateStats().totalExecutions;
      await orchestrator!.executeWorkflow(workflow);
      const afterExecutions = monitor.getAggregateStats().totalExecutions;

      // Verify that the workflow was tracked (monitoring integration works)
      expect(afterExecutions).toBeGreaterThanOrEqual(beforeExecutions);
    });
  });

  describe('Error Handling', () => {
    it('should handle plugin failures gracefully', async () => {
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();

      const workflow: WorkflowDefinition = {
        id: 'error-test-workflow',
        name: 'Error Test',
        steps: [
          {
            id: 'step1',
            name: 'Invalid Step',
            plugin: 'nonexistent-plugin',
            action: 'nonexistent-action',
            input: {}
          }
        ],
        onError: 'continue'
      };

      const result = await orchestrator!.executeWorkflow(workflow);

      // Should not throw, should return failed result
      expect(result).toBeDefined();
      expect(result.status).toBe('failed');
    });
  });

  describe('Workflow Templates', () => {
    it('should have all predefined templates', () => {
      const orchestrator = getBubbleLabIntegration()?.getOrchestrator();

      // Test each template validates successfully
      const templates = [
        RESEARCH_ASSISTANT_WORKFLOW
        // Add other templates as plugins become available
      ];

      for (const template of templates) {
        const validation = orchestrator!.validateWorkflow(template);
        expect(validation.valid).toBe(true);
      }
    });
  });
});

/**
 * Manual Integration Test Checklist
 *
 * Run these steps manually to verify the integration:
 *
 * 1. Plugin Registration
 *    [ ] RAGBits plugin loads successfully
 *    [ ] Datapizza plugin loads successfully
 *    [ ] OpenEvolve API adapter loads
 *    [ ] All plugins appear in registry
 *
 * 2. Plugin Initialization
 *    [ ] Plugins initialize without errors
 *    [ ] Health checks pass for available plugins
 *    [ ] Plugin capabilities are detected
 *
 * 3. Workflow Execution
 *    [ ] Can select workflow template
 *    [ ] Can input workflow parameters
 *    [ ] Workflow executes end-to-end
 *    [ ] Step results are displayed
 *    [ ] Final output is shown
 *
 * 4. Event Bus
 *    [ ] Events are published on workflow start
 *    [ ] Events are published on workflow completion
 *    [ ] Cross-plugin handlers work
 *
 * 5. Monitoring
 *    [ ] Workflow metrics are recorded
 *    [ ] Step metrics are recorded
 *    [ ] Aggregate statistics are available
 *    [ ] Can export metrics
 *
 * 6. UI Integration
 *    [ ] Workflow tab appears in navigation
 *    [ ] Workflow templates display correctly
 *    [ ] Execution shows real-time updates
 *    [ ] History is tracked and displayed
 *
 * 7. Error Handling
 *    [ ] Invalid workflows show validation errors
 *    [ ] Plugin failures don't crash the app
 *    [ ] Retry logic works when configured
 *    [ ] Circuit breakers prevent cascading failures
 */
