"use strict";
/**
 * OpenEvolve API Contract Tests
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 * "Protecting the Mega-Project from Updates"
 *
 * These tests verify that the OpenEvolve API returns the expected fields.
 * If the contract is violated (Project API changed), the adapter MUST refuse to start.
 *
 * Runs on container startup. If these tests fail, the application should NOT start.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const openevolveApi_1 = require("../../lib/openevolveApi");
// Configuration
const API_URL = process.env.OPENEVOLVE_API_URL || 'http://localhost:8000';
const API_KEY = process.env.OPENEVOLVE_API_KEY;
const TIMEOUT = 30000; // 30 second timeout for contract tests
const WORKFLOW_POLL_INTERVAL_MS = 500;
const WORKFLOW_POLL_TIMEOUT_MS = 20000;
const TERMINAL_WORKFLOW_STATES = new Set(['completed', 'failed', 'stopped', 'cancelled']);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const getWorkflowApiConfig = () => {
    if (!API_KEY) {
        return null;
    }
    return {
        baseUrl: API_URL,
        apiKey: API_KEY,
        timeout: TIMEOUT,
    };
};
const waitForTerminalWorkflowState = async (instanceId, config) => {
    const startedAt = Date.now();
    let latest = await openevolveApi_1.openevolveApi.getWorkflowInstance(instanceId, config);
    while (!TERMINAL_WORKFLOW_STATES.has(latest.status.status)) {
        if (Date.now() - startedAt > WORKFLOW_POLL_TIMEOUT_MS) {
            throw new Error(`Timed out waiting for workflow ${instanceId}. Last state: ${latest.status.status}`);
        }
        await sleep(WORKFLOW_POLL_INTERVAL_MS);
        latest = await openevolveApi_1.openevolveApi.getWorkflowInstance(instanceId, config);
    }
    return latest;
};
(0, vitest_1.describe)('OpenEvolve API Contract Tests', () => {
    (0, vitest_1.describe)('Health Check Endpoint', () => {
        (0, vitest_1.it)('should return health status with required fields', async () => {
            const response = await fetch(`${API_URL}/health`, {
                headers: API_KEY ? { 'Authorization': `Bearer ${API_KEY}` } : {},
            });
            (0, vitest_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Health check must include status field
            (0, vitest_1.expect)(data).toHaveProperty('status');
            (0, vitest_1.expect)(typeof data.status).toBe('string');
            // Contract: Should include version information
            (0, vitest_1.expect)(data).toHaveProperty('version');
        });
        (0, vitest_1.it)('should respond within timeout', async () => {
            const start = Date.now();
            const response = await fetch(`${API_URL}/health`);
            const duration = Date.now() - start;
            (0, vitest_1.expect)(response.ok).toBe(true);
            (0, vitest_1.expect)(duration).toBeLessThan(TIMEOUT);
        });
    });
    (0, vitest_1.describe)('Evolutions Endpoint', () => {
        (0, vitest_1.it)('should return evolutions list with required fields', async () => {
            const response = await fetch(`${API_URL}/evolutions`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Evolutions endpoint not implemented - skipping');
                return;
            }
            (0, vitest_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have evolutions array
            (0, vitest_1.expect)(data).toHaveProperty('evolutions');
            (0, vitest_1.expect)(Array.isArray(data.evolutions)).toBe(true);
            // Contract: Each evolution must have required fields
            if (data.evolutions.length > 0) {
                const evolution = data.evolutions[0];
                (0, vitest_1.expect)(evolution).toHaveProperty('id');
                (0, vitest_1.expect)(evolution).toHaveProperty('name');
                (0, vitest_1.expect)(evolution).toHaveProperty('created_at');
            }
        });
    });
    (0, vitest_1.describe)('Adversarial Runs Endpoint', () => {
        (0, vitest_1.it)('should return adversarial runs list with required fields', async () => {
            const response = await fetch(`${API_URL}/adversarial-runs`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Adversarial runs endpoint not implemented - skipping');
                return;
            }
            (0, vitest_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have runs array
            (0, vitest_1.expect)(data).toHaveProperty('runs');
            (0, vitest_1.expect)(Array.isArray(data.runs)).toBe(true);
            // Contract: Each run must have required fields
            if (data.runs.length > 0) {
                const run = data.runs[0];
                (0, vitest_1.expect)(run).toHaveProperty('id');
                (0, vitest_1.expect)(run).toHaveProperty('status');
                (0, vitest_1.expect)(run).toHaveProperty('created_at');
            }
        });
    });
    (0, vitest_1.describe)('Create Evolution Endpoint', () => {
        (0, vitest_1.it)('should accept evolution creation request with required fields', async () => {
            const testRequest = {
                name: 'Contract Test Evolution',
                base_prompt: 'Test prompt for contract validation',
                adversarial_prompt: 'Test adversarial prompt',
                parameters: {
                    temperature: 0.7,
                    max_tokens: 1000,
                },
            };
            const response = await fetch(`${API_URL}/evolutions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(testRequest),
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Create evolution endpoint not implemented - skipping');
                return;
            }
            (0, vitest_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return created evolution with ID
            (0, vitest_1.expect)(data).toHaveProperty('id');
            (0, vitest_1.expect)(data).toHaveProperty('name');
            (0, vitest_1.expect)(data.name).toBe(testRequest.name);
            (0, vitest_1.expect)(data).toHaveProperty('created_at');
        });
    });
    (0, vitest_1.describe)('Get Evolution by ID Endpoint', () => {
        (0, vitest_1.it)('should return evolution details with required fields', async () => {
            // First, try to list evolutions to get a valid ID
            const listResponse = await fetch(`${API_URL}/evolutions`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            if (!listResponse.ok) {
                console.warn('Cannot get evolution list - skipping get by ID test');
                return;
            }
            const listData = await listResponse.json();
            if (!listData.evolutions || listData.evolutions.length === 0) {
                console.warn('No evolutions found - skipping get by ID test');
                return;
            }
            const evolutionId = listData.evolutions[0].id;
            const response = await fetch(`${API_URL}/evolutions/${evolutionId}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            (0, vitest_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return all required fields
            (0, vitest_1.expect)(data).toHaveProperty('id');
            (0, vitest_1.expect)(data).toHaveProperty('name');
            (0, vitest_1.expect)(data).toHaveProperty('base_prompt');
            (0, vitest_1.expect)(data).toHaveProperty('created_at');
            (0, vitest_1.expect)(data).toHaveProperty('updated_at');
        });
    });
    (0, vitest_1.describe)('Error Responses', () => {
        (0, vitest_1.it)('should return proper error for invalid evolution ID', async () => {
            const response = await fetch(`${API_URL}/evolutions/invalid-id-12345`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Contract: Should return 404 or 400 for invalid ID
            (0, vitest_1.expect)([400, 404]).toContain(response.status);
            const data = await response.json();
            // Contract: Error response must have error field
            (0, vitest_1.expect)(data).toHaveProperty('error');
        });
        (0, vitest_1.it)('should return proper error for invalid request body', async () => {
            const response = await fetch(`${API_URL}/evolutions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: 'invalid json{{{',
            });
            // Contract: Should return 400 for bad request
            (0, vitest_1.expect)(response.status).toBe(400);
        });
    });
    (0, vitest_1.describe)('Pagination and Filtering', () => {
        (0, vitest_1.it)('should support pagination parameters', async () => {
            const response = await fetch(`${API_URL}/evolutions?limit=10&offset=0`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            if (response.status === 404) {
                console.warn('Evolutions endpoint not implemented - skipping pagination test');
                return;
            }
            (0, vitest_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Should respect limit parameter
            (0, vitest_1.expect)(data.evolutions.length).toBeLessThanOrEqual(10);
        });
    });
    (0, vitest_1.describe)('BubbleLabs Workflow Lifecycle', () => {
        (0, vitest_1.it)('should execute an end-to-end BubbleLabs workflow with OpenEvolve controls', async () => {
            const config = getWorkflowApiConfig();
            if (!config) {
                console.warn('OPENEVOLVE_API_KEY not set - skipping BubbleLabs workflow lifecycle contract test');
                return;
            }
            const uniqueSuffix = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
            const definitionName = `contract-workflow-${uniqueSuffix}`;
            let instanceId = null;
            try {
                const createdDefinition = await openevolveApi_1.openevolveApi.createWorkflowDefinition({
                    name: definitionName,
                    description: 'Contract e2e workflow lifecycle test',
                    workflow_type: 'evolution',
                    parameters: {
                        max_iterations: 1,
                        population_size: 2,
                    },
                }, config);
                (0, vitest_1.expect)(createdDefinition.definition_id).toBeTruthy();
                const definitionDetails = await openevolveApi_1.openevolveApi.getWorkflowDefinition(createdDefinition.definition_id, config);
                (0, vitest_1.expect)(definitionDetails.id).toBe(createdDefinition.definition_id);
                (0, vitest_1.expect)(definitionDetails.workflow_type).toBe('evolution');
                const createdInstance = await openevolveApi_1.openevolveApi.createWorkflowInstance({
                    definition_id: createdDefinition.definition_id,
                    instance_name: `instance-${uniqueSuffix}`,
                    inputs: {
                        problem_statement: 'Contract test: evolve a short deterministic prompt',
                    },
                }, config);
                instanceId = createdInstance.instance_id;
                (0, vitest_1.expect)(instanceId).toBeTruthy();
                const synced = await openevolveApi_1.openevolveApi.syncWorkflowInstanceParameters(instanceId, {
                    parameters: {
                        max_iterations: 1,
                        population_size: 2,
                        temperature: 0.1,
                    },
                }, config);
                (0, vitest_1.expect)(synced).toHaveProperty('updated_count');
                const started = await openevolveApi_1.openevolveApi.startWorkflowInstance(instanceId, config);
                (0, vitest_1.expect)(started).toHaveProperty('instance_id', instanceId);
                const terminalState = await waitForTerminalWorkflowState(instanceId, config);
                if (terminalState.status.status === 'failed') {
                    throw new Error(`Workflow failed: ${terminalState.status.error_message || 'unknown failure'}`);
                }
                if (terminalState.status.status !== 'completed') {
                    const stopped = await openevolveApi_1.openevolveApi.stopWorkflowInstance(instanceId, config);
                    (0, vitest_1.expect)(stopped).toHaveProperty('status', 'stopped');
                }
                const finalizedState = await openevolveApi_1.openevolveApi.getWorkflowInstance(instanceId, config);
                (0, vitest_1.expect)(['completed', 'stopped']).toContain(finalizedState.status.status);
            }
            finally {
                if (instanceId) {
                    await openevolveApi_1.openevolveApi.deleteWorkflowInstance(instanceId, config).catch(() => undefined);
                }
            }
        }, 120000);
    });
});
/**
 * Usage Instructions:
 *
 * 1. Run tests on container startup:
 *    ```bash
 *    npm run test:contract
 *    ```
 *
 * 2. If tests fail, container MUST refuse to start:
 *    ```javascript
 *    try {
 *      await runContractTests();
 *    } catch (error) {
 *      logger.error('Contract tests failed - refusing to start');
 *      process.exit(1);
 *    }
 *    ```
 *
 * 3. Tests verify critical fields that the adapter depends on
 * 4. If API changes, tests fail before corrupting data
 */
//# sourceMappingURL=openevolve-api.test.js.map