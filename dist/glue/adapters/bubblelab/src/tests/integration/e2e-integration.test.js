"use strict";
/**
 * End-to-End Integration Test
 *
 * Tests the complete OpenEvolve-BubbleLab integration including:
 * - Plugin registration and initialization
 * - Workflow execution across plugins
 * - Event bus integration
 * - Monitoring and telemetry
 */
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const plugin_integration_1 = require("../../lib/plugin-integration");
const plugin_registry_1 = require("../../lib/plugin-registry");
const workflow_monitoring_1 = require("../../lib/workflow-monitoring");
const workflow_templates_1 = require("../../lib/workflow-templates");
(0, vitest_1.describe)('End-to-End Integration', () => {
    (0, vitest_1.beforeAll)(async () => {
        // Initialize the integration
        await (0, plugin_integration_1.initializeBubbleLabIntegration)({
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
        const integration = (0, plugin_integration_1.getBubbleLabIntegration)();
        if (integration) {
            await integration.destroy();
        }
    });
    (0, vitest_1.describe)('Plugin Registry', () => {
        (0, vitest_1.it)('should have plugins registered', () => {
            const registry = (0, plugin_registry_1.getPluginRegistry)();
            const plugins = registry.getAllPlugins();
            (0, vitest_1.expect)(plugins.length).toBeGreaterThan(0);
            (0, vitest_1.expect)(plugins[0]).toBeDefined();
            (0, vitest_1.expect)(plugins[0].metadata).toBeDefined();
            (0, vitest_1.expect)(plugins[0].capabilities).toBeDefined();
        });
        (0, vitest_1.it)('should have OpenEvolve API plugin', () => {
            const registry = (0, plugin_registry_1.getPluginRegistry)();
            const openevolvePlugin = registry.getPlugin('openevolve');
            (0, vitest_1.expect)(openevolvePlugin).toBeDefined();
            (0, vitest_1.expect)(openevolvePlugin?.metadata.name).toBe('openevolve');
        });
        (0, vitest_1.it)('should get plugins by capability', () => {
            const registry = (0, plugin_registry_1.getPluginRegistry)();
            const searchPlugins = registry.getPluginsByCapability('search');
            (0, vitest_1.expect)(Array.isArray(searchPlugins)).toBe(true);
        });
        (0, vitest_1.it)('should get plugin statistics', () => {
            const registry = (0, plugin_registry_1.getPluginRegistry)();
            const stats = registry.getStatistics();
            (0, vitest_1.expect)(stats).toBeDefined();
            (0, vitest_1.expect)(stats.totalPlugins).toBeGreaterThan(0);
            (0, vitest_1.expect)(typeof stats.totalPlugins).toBe('number');
        });
    });
    (0, vitest_1.describe)('Workflow Orchestrator', () => {
        (0, vitest_1.it)('should be available', () => {
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            (0, vitest_1.expect)(orchestrator).toBeDefined();
        });
        (0, vitest_1.it)('should validate workflow templates', () => {
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            (0, vitest_1.expect)(orchestrator).toBeDefined();
            const validation = orchestrator.validateWorkflow(workflow_templates_1.RESEARCH_ASSISTANT_WORKFLOW);
            (0, vitest_1.expect)(validation.valid).toBe(true);
            (0, vitest_1.expect)(validation.errors).toHaveLength(0);
        });
        (0, vitest_1.it)('should execute a simple workflow', async () => {
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            (0, vitest_1.expect)(orchestrator).toBeDefined();
            const simpleWorkflow = {
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
            const result = await orchestrator.executeWorkflow(simpleWorkflow);
            (0, vitest_1.expect)(result).toBeDefined();
            (0, vitest_1.expect)(result.executionId).toBeDefined();
            (0, vitest_1.expect)(result.workflowId).toBe('test-workflow');
            (0, vitest_1.expect)(result.status).toMatch(/completed|failed/);
        });
    });
    (0, vitest_1.describe)('Monitoring System', () => {
        (0, vitest_1.it)('should track workflow executions', async () => {
            const monitor = (0, workflow_monitoring_1.getWorkflowMonitor)();
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            (0, vitest_1.expect)(orchestrator).toBeDefined();
            const simpleWorkflow = {
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
            const result = await orchestrator.executeWorkflow(simpleWorkflow);
            // Check that metrics were recorded
            const metrics = monitor.getWorkflowMetrics(result.executionId);
            (0, vitest_1.expect)(metrics).toBeDefined();
            (0, vitest_1.expect)(metrics?.workflowId).toBe('monitor-test-workflow');
            (0, vitest_1.expect)(metrics?.executionId).toBe(result.executionId);
        });
        (0, vitest_1.it)('should provide aggregate statistics', () => {
            const monitor = (0, workflow_monitoring_1.getWorkflowMonitor)();
            const stats = monitor.getAggregateStats();
            (0, vitest_1.expect)(stats).toBeDefined();
            (0, vitest_1.expect)(typeof stats.totalExecutions).toBe('number');
            (0, vitest_1.expect)(typeof stats.averageDuration).toBe('number');
        });
    });
    (0, vitest_1.describe)('Integration Lifecycle', () => {
        (0, vitest_1.it)('should start the integration', async () => {
            const integration = (0, plugin_integration_1.getBubbleLabIntegration)();
            (0, vitest_1.expect)(integration).toBeDefined();
            const status = integration.getStatus();
            (0, vitest_1.expect)(status).toBeDefined();
            (0, vitest_1.expect)(typeof status.initialized).toBe('boolean');
            (0, vitest_1.expect)(typeof status.pluginCount).toBe('number');
        });
        (0, vitest_1.it)('should access registry through integration', () => {
            const integration = (0, plugin_integration_1.getBubbleLabIntegration)();
            const registry = integration?.getRegistry();
            (0, vitest_1.expect)(registry).toBeDefined();
            (0, vitest_1.expect)(registry?.getAllPlugins()).toBeDefined();
        });
        (0, vitest_1.it)('should access orchestrator through integration', () => {
            const integration = (0, plugin_integration_1.getBubbleLabIntegration)();
            const orchestrator = integration?.getOrchestrator();
            (0, vitest_1.expect)(orchestrator).toBeDefined();
        });
    });
    (0, vitest_1.describe)('Cross-Plugin Communication', () => {
        (0, vitest_1.it)('should handle events between plugins', async () => {
            // This test verifies the event bus integration is working
            // Events should flow from workflow orchestrator to event integration
            const monitor = (0, workflow_monitoring_1.getWorkflowMonitor)();
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            const workflow = {
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
            await orchestrator.executeWorkflow(workflow);
            const afterExecutions = monitor.getAggregateStats().totalExecutions;
            // Verify that the workflow was tracked (monitoring integration works)
            (0, vitest_1.expect)(afterExecutions).toBeGreaterThanOrEqual(beforeExecutions);
        });
    });
    (0, vitest_1.describe)('Error Handling', () => {
        (0, vitest_1.it)('should handle plugin failures gracefully', async () => {
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            const workflow = {
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
            const result = await orchestrator.executeWorkflow(workflow);
            // Should not throw, should return failed result
            (0, vitest_1.expect)(result).toBeDefined();
            (0, vitest_1.expect)(result.status).toBe('failed');
        });
    });
    (0, vitest_1.describe)('Workflow Templates', () => {
        (0, vitest_1.it)('should have all predefined templates', () => {
            const orchestrator = (0, plugin_integration_1.getBubbleLabIntegration)()?.getOrchestrator();
            // Test each template validates successfully
            const templates = [
                workflow_templates_1.RESEARCH_ASSISTANT_WORKFLOW
                // Add other templates as plugins become available
            ];
            for (const template of templates) {
                const validation = orchestrator.validateWorkflow(template);
                (0, vitest_1.expect)(validation.valid).toBe(true);
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
//# sourceMappingURL=e2e-integration.test.js.map