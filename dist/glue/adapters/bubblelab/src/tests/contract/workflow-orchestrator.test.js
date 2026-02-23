"use strict";
/**
 * Contract Tests for Workflow Orchestrator
 *
 * Tests the workflow execution engine with various scenarios
 */
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const workflow_orchestrator_1 = require("../../lib/workflow-orchestrator");
const plugin_registry_1 = require("../../lib/plugin-registry");
// Mock plugin for testing
class MockPlugin {
    constructor() {
        this.metadata = {
            name: 'mock-plugin',
            version: '1.0.0',
            description: 'Mock plugin for testing',
            author: 'Test',
            enabled: true
        };
        this.capabilities = {
            processing: true
        };
        this.status = 'idle';
    }
    async initialize() {
        this.status = 'ready';
    }
    async updateConfig() {
        // Mock implementation
    }
    async resetConfig() {
        // Mock implementation
    }
    async healthCheck() {
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
    async destroy() {
        this.status = 'idle';
    }
    // Custom actions
    async processData(input) {
        this.status = 'busy';
        const result = { result: `processed: ${input.data}` };
        this.status = 'ready';
        return result;
    }
    async transformData(input) {
        this.status = 'busy';
        const result = { doubled: input.value * 2 };
        this.status = 'ready';
        return result;
    }
    async failAction() {
        this.status = 'busy';
        this.status = 'error';
        throw new Error('Intentional failure');
    }
}
(0, vitest_1.describe)('Workflow Orchestrator', () => {
    let orchestrator;
    let registry;
    let mockPlugin;
    (0, vitest_1.beforeEach)(async () => {
        registry = new plugin_registry_1.PluginRegistry({ autoInitialize: false });
        orchestrator = new workflow_orchestrator_1.WorkflowOrchestrator(registry);
        mockPlugin = new MockPlugin();
        await registry.registerPlugin(mockPlugin);
    });
    (0, vitest_1.afterEach)(async () => {
        await registry.destroy();
    });
    (0, vitest_1.describe)('Workflow Validation', () => {
        (0, vitest_1.it)('should validate a correct workflow', () => {
            const workflow = {
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
            (0, vitest_1.expect)(result.valid).toBe(true);
            (0, vitest_1.expect)(result.errors).toHaveLength(0);
        });
        (0, vitest_1.it)('should reject workflow without ID', () => {
            const workflow = {
                id: '',
                name: 'Test Workflow',
                steps: []
            };
            const result = orchestrator.validateWorkflow(workflow);
            (0, vitest_1.expect)(result.valid).toBe(false);
            (0, vitest_1.expect)(result.errors).toContain('Workflow ID is required');
        });
        (0, vitest_1.it)('should reject workflow without steps', () => {
            const workflow = {
                id: 'test',
                name: 'Test Workflow',
                steps: []
            };
            const result = orchestrator.validateWorkflow(workflow);
            (0, vitest_1.expect)(result.valid).toBe(false);
            (0, vitest_1.expect)(result.errors).toContain('Workflow must have at least one step');
        });
        (0, vitest_1.it)('should reject workflow with circular dependencies', async () => {
            const workflow = {
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
            await (0, vitest_1.expect)(orchestrator.executeWorkflow(workflow)).rejects.toThrow('Circular dependency');
        });
    });
    (0, vitest_1.describe)('Workflow Execution', () => {
        (0, vitest_1.it)('should execute a simple workflow', async () => {
            const workflow = {
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
            (0, vitest_1.expect)(result.status).toBe('completed');
            (0, vitest_1.expect)(result.executionId).toBeDefined();
            (0, vitest_1.expect)(result.stepResults.size).toBe(1);
            (0, vitest_1.expect)(result.results).toHaveProperty('output');
        });
        (0, vitest_1.it)('should execute workflow with dependencies', async () => {
            const workflow = {
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
            (0, vitest_1.expect)(result.status).toBe('completed');
            (0, vitest_1.expect)(result.stepResults.size).toBe(2);
        });
        (0, vitest_1.it)('should handle workflow errors with stop policy', async () => {
            const workflow = {
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
            (0, vitest_1.expect)(result.status).toBe('failed');
            (0, vitest_1.expect)(result.errors).toHaveLength(1);
            (0, vitest_1.expect)(result.errors[0].stepId).toBe('step2');
            (0, vitest_1.expect)(result.stepResults.size).toBe(1); // Only step1 completed
        });
        (0, vitest_1.it)('should handle workflow errors with continue policy', async () => {
            const workflow = {
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
            (0, vitest_1.expect)(result.status).toBe('completed'); // Workflow completed despite error
            (0, vitest_1.expect)(result.errors).toHaveLength(1);
            (0, vitest_1.expect)(result.stepResults.size).toBe(2); // step1 and step3
        });
        (0, vitest_1.it)('should execute steps in topological order', async () => {
            const executionOrder = [];
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
            const workflow = {
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
            (0, vitest_1.expect)(executionOrder).toEqual(['process', 'process', 'transform']);
        });
    });
    (0, vitest_1.describe)('Workflow Cancellation', () => {
        (0, vitest_1.it)('should cancel a running workflow', async () => {
            // Create a workflow with delays
            let shouldBlock = true;
            const workflow = {
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
            (0, vitest_1.expect)(activeWorkflows.length).toBeGreaterThan(0);
            // Cancel the workflow
            const cancelled = await orchestrator.cancelWorkflow(activeWorkflows[0].executionId);
            (0, vitest_1.expect)(cancelled).toBe(true);
        });
    });
    (0, vitest_1.describe)('Output Mapping', () => {
        (0, vitest_1.it)('should map step outputs to workflow output', async () => {
            const workflow = {
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
            (0, vitest_1.expect)(result.results).toHaveProperty('firstOutput');
            (0, vitest_1.expect)(result.results).toHaveProperty('secondOutput');
            (0, vitest_1.expect)(result.results.firstOutput).toContain('processed: test');
            (0, vitest_1.expect)(result.results.secondOutput).toBe(20);
        });
    });
});
//# sourceMappingURL=workflow-orchestrator.test.js.map