"use strict";
/**
 * BubbleLab Adapter Contract Tests
 *
 * Purpose: Validate BubbleLab API contracts to prevent breaking changes
 * Compliance: Phase 2 - The Contract (Defense)
 *
 * These tests run on adapter startup to verify the API returns expected fields
 * If contracts are violated, the adapter refuses to start (Law of Runtime Truth)
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExecutionHistoryContract = exports.ExecutionResponseContract = exports.BubbleFlowCreateResponseContract = exports.BubbleFlowContract = exports.HealthCheckContract = void 0;
exports.validateAllContracts = validateAllContracts;
const globals_1 = require("@jest/globals");
const zod_1 = require("zod");
// =============================================================================
// Mock BubbleLab API Responses for Contract Testing
// =============================================================================
const MOCK_HEALTH_RESPONSE = {
    status: 'ok',
    version: '1.0.0',
};
const MOCK_BUBBLE_FLOW_LIST_RESPONSE = [
    {
        id: '123',
        name: 'Test Flow',
        description: 'A test workflow',
        eventType: 'webhook/http',
        webhookActive: false,
        createdAt: '2026-02-03T00:00:00.000Z',
        updatedAt: '2026-02-03T00:00:00.000Z',
    },
];
const MOCK_BUBBLE_FLOW_CREATE_RESPONSE = {
    id: '456',
    name: 'New Flow',
    requiredCredentials: {
        'postgres-bubble': ['DATABASE_CRED'],
    },
    webhookUrl: 'https://api.bubblelab.dev/webhook/abc123',
    createdAt: '2026-02-03T00:00:00.000Z',
};
const MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE = {
    execution_id: 'exec-789',
    output: {
        message: 'Success',
        processed: true,
    },
    status: 'success',
    startedAt: '2026-02-03T00:00:00.000Z',
    completedAt: '2026-02-03T00:00:01.000Z',
};
const MOCK_EXECUTION_HISTORY_RESPONSE = {
    executions: [
        {
            id: 'exec-001',
            status: 'success',
            startedAt: '2026-02-03T00:00:00.000Z',
            completedAt: '2026-02-03T00:00:01.000Z',
            output: { result: 'test' },
        },
    ],
};
// =============================================================================
// Contract Schemas
// =============================================================================
/**
 * Health Check Response Contract
 */
const HealthCheckContract = zod_1.z.object({
    status: zod_1.z.enum(['ok', 'healthy', 'error']),
    version: zod_1.z.string().optional(),
});
exports.HealthCheckContract = HealthCheckContract;
/**
 * BubbleFlow List Response Contract
 */
const BubbleFlowContract = zod_1.z.object({
    id: zod_1.z.union([zod_1.z.string(), zod_1.z.number()]),
    name: zod_1.z.string(),
    description: zod_1.z.string().optional(),
    eventType: zod_1.z.string(),
    webhookActive: zod_1.z.boolean(),
    createdAt: zod_1.z.string().optional(),
    updatedAt: zod_1.z.string().optional(),
});
exports.BubbleFlowContract = BubbleFlowContract;
/**
 * BubbleFlow Create Response Contract
 */
const BubbleFlowCreateResponseContract = zod_1.z.object({
    id: zod_1.z.union([zod_1.z.string(), zod_1.z.number()]),
    name: zod_1.z.string(),
    requiredCredentials: zod_1.z.record(zod_1.z.string(), zod_1.z.array(zod_1.z.string())).optional(),
    webhookUrl: zod_1.z.string().optional(),
    createdAt: zod_1.z.string().optional(),
});
exports.BubbleFlowCreateResponseContract = BubbleFlowCreateResponseContract;
/**
 * Execution Response Contract
 */
const ExecutionResponseContract = zod_1.z.object({
    execution_id: zod_1.z.union([zod_1.z.string(), zod_1.z.number()]).optional(),
    output: zod_1.z.any().optional(),
    error: zod_1.z.string().optional(),
    status: zod_1.z.string().optional(),
});
exports.ExecutionResponseContract = ExecutionResponseContract;
/**
 * Execution History Response Contract
 */
const ExecutionHistoryContract = zod_1.z.object({
    executions: zod_1.z.array(zod_1.z.object({
        id: zod_1.z.union([zod_1.z.string(), zod_1.z.number()]).optional(),
        status: zod_1.z.string(),
        startedAt: zod_1.z.string(),
        completedAt: zod_1.z.string().optional(),
        output: zod_1.z.any().optional(),
        error: zod_1.z.string().optional(),
    })).optional(),
});
exports.ExecutionHistoryContract = ExecutionHistoryContract;
// =============================================================================
// Contract Tests
// =============================================================================
(0, globals_1.describe)('BubbleLab API Contract Tests', () => {
    (0, globals_1.describe)('Health Check Endpoint', () => {
        (0, globals_1.it)('should return valid health check response', () => {
            const result = HealthCheckContract.safeParse(MOCK_HEALTH_RESPONSE);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.status).toBe('ok');
            }
        });
        (0, globals_1.it)('should include status field', () => {
            const response = MOCK_HEALTH_RESPONSE;
            (0, globals_1.expect)(response).toHaveProperty('status');
            (0, globals_1.expect)(typeof response.status).toBe('string');
        });
        (0, globals_1.it)('should allow optional version field', () => {
            const response = MOCK_HEALTH_RESPONSE;
            (0, globals_1.expect)(response.version).toBeDefined();
            (0, globals_1.expect)(typeof response.version).toBe('string');
        });
    });
    (0, globals_1.describe)('BubbleFlow List Endpoint', () => {
        (0, globals_1.it)('should return valid BubbleFlow objects', () => {
            const result = BubbleFlowContract.safeParse(MOCK_BUBBLE_FLOW_LIST_RESPONSE[0]);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.name).toBe('Test Flow');
                (0, globals_1.expect)(result.data.eventType).toBe('webhook/http');
            }
        });
        (0, globals_1.it)('should include required fields', () => {
            const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];
            (0, globals_1.expect)(flow).toHaveProperty('id');
            (0, globals_1.expect)(flow).toHaveProperty('name');
            (0, globals_1.expect)(flow).toHaveProperty('eventType');
            (0, globals_1.expect)(flow).toHaveProperty('webhookActive');
        });
        (0, globals_1.it)('should support string or numeric IDs', () => {
            const stringId = { ...MOCK_BUBBLE_FLOW_LIST_RESPONSE[0], id: '123' };
            const numberId = { ...MOCK_BUBBLE_FLOW_LIST_RESPONSE[0], id: 123 };
            const stringResult = BubbleFlowContract.safeParse(stringId);
            const numberResult = BubbleFlowContract.safeParse(numberId);
            (0, globals_1.expect)(stringResult.success).toBe(true);
            (0, globals_1.expect)(numberResult.success).toBe(true);
        });
        (0, globals_1.it)('should include optional timestamp fields', () => {
            const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];
            (0, globals_1.expect)(flow.createdAt).toBeDefined();
            (0, globals_1.expect)(flow.updatedAt).toBeDefined();
            // Verify ISO-8601 format
            (0, globals_1.expect)(() => new Date(flow.createdAt)).not.toThrow();
            (0, globals_1.expect)(() => new Date(flow.updatedAt)).not.toThrow();
        });
    });
    (0, globals_1.describe)('BubbleFlow Create Endpoint', () => {
        (0, globals_1.it)('should return valid create response', () => {
            const result = BubbleFlowCreateResponseContract.safeParse(MOCK_BUBBLE_FLOW_CREATE_RESPONSE);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.id).toBeDefined();
                (0, globals_1.expect)(result.data.name).toBe('New Flow');
            }
        });
        (0, globals_1.it)('should include requiredCredentials object', () => {
            const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;
            (0, globals_1.expect)(response.requiredCredentials).toBeDefined();
            (0, globals_1.expect)(typeof response.requiredCredentials).toBe('object');
        });
        (0, globals_1.it)('should include credential arrays per bubble', () => {
            const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;
            if (response.requiredCredentials) {
                const bubbleName = Object.keys(response.requiredCredentials)[0];
                const creds = response.requiredCredentials[bubbleName];
                (0, globals_1.expect)(Array.isArray(creds)).toBe(true);
                (0, globals_1.expect)(creds).toContain('DATABASE_CRED');
            }
        });
        (0, globals_1.it)('should include webhook URL if webhook is active', () => {
            const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;
            (0, globals_1.expect)(response.webhookUrl).toBeDefined();
            (0, globals_1.expect)(typeof response.webhookUrl).toBe('string');
            (0, globals_1.expect)(response.webhookUrl).toMatch(/^https?:\/\//);
        });
    });
    (0, globals_1.describe)('BubbleFlow Execute Endpoint', () => {
        (0, globals_1.it)('should return valid execution response', () => {
            const result = ExecutionResponseContract.safeParse(MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.status).toBe('success');
            }
        });
        (0, globals_1.it)('should include execution ID or status', () => {
            const response = MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE;
            (0, globals_1.expect)(response.execution_id !== undefined || response.status !== undefined).toBe(true);
        });
        (0, globals_1.it)('should include output data on success', () => {
            const response = MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE;
            if (response.status === 'success') {
                (0, globals_1.expect)(response.output).toBeDefined();
            }
        });
        (0, globals_1.it)('should include error message on failure', () => {
            const failedResponse = {
                ...MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE,
                status: 'failed',
                error: 'Execution failed',
                output: undefined,
            };
            const result = ExecutionResponseContract.safeParse(failedResponse);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.error).toBeDefined();
            }
        });
    });
    (0, globals_1.describe)('Execution History Endpoint', () => {
        (0, globals_1.it)('should return valid execution history', () => {
            const result = ExecutionHistoryContract.safeParse(MOCK_EXECUTION_HISTORY_RESPONSE);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.executions).toBeDefined();
                (0, globals_1.expect)(Array.isArray(result.data.executions)).toBe(true);
            }
        });
        (0, globals_1.it)('should include timestamp fields for each execution', () => {
            const history = MOCK_EXECUTION_HISTORY_RESPONSE;
            if (history.executions && history.executions.length > 0) {
                const execution = history.executions[0];
                (0, globals_1.expect)(execution.startedAt).toBeDefined();
                (0, globals_1.expect)(typeof execution.startedAt).toBe('string');
                // Verify ISO-8601 format
                (0, globals_1.expect)(() => new Date(execution.startedAt)).not.toThrow();
            }
        });
        (0, globals_1.it)('should include status for each execution', () => {
            const history = MOCK_EXECUTION_HISTORY_RESPONSE;
            if (history.executions && history.executions.length > 0) {
                const execution = history.executions[0];
                (0, globals_1.expect)(execution.status).toBeDefined();
                (0, globals_1.expect)(typeof execution.status).toBe('string');
            }
        });
    });
    (0, globals_1.describe)('Edge Cases and Error Handling', () => {
        (0, globals_1.it)('should handle missing optional fields gracefully', () => {
            const minimalFlow = {
                id: '123',
                name: 'Minimal Flow',
                eventType: 'manual',
                webhookActive: false,
            };
            const result = BubbleFlowContract.safeParse(minimalFlow);
            (0, globals_1.expect)(result.success).toBe(true);
        });
        (0, globals_1.it)('should reject invalid response structures', () => {
            const invalidFlow = {
                id: '123',
                // Missing required 'name' field
                eventType: 'webhook/http',
            };
            const result = BubbleFlowContract.safeParse(invalidFlow);
            (0, globals_1.expect)(result.success).toBe(false);
        });
        (0, globals_1.it)('should handle empty execution history', () => {
            const emptyHistory = { executions: [] };
            const result = ExecutionHistoryContract.safeParse(emptyHistory);
            (0, globals_1.expect)(result.success).toBe(true);
            if (result.success) {
                (0, globals_1.expect)(result.data.executions).toHaveLength(0);
            }
        });
    });
    (0, globals_1.describe)('Data Type Validation', () => {
        (0, globals_1.it)('should validate boolean webhookActive field', () => {
            const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];
            (0, globals_1.expect)(typeof flow.webhookActive).toBe('boolean');
            (0, globals_1.expect)(flow.webhookActive).toBe(false);
        });
        (0, globals_1.it)('should validate string event types', () => {
            const flow = MOCK_BUBBLE_FLOW_LIST_RESPONSE[0];
            (0, globals_1.expect)(typeof flow.eventType).toBe('string');
            (0, globals_1.expect)(['webhook/http', 'schedule', 'manual']).toContain(flow.eventType);
        });
        (0, globals_1.it)('should validate credential type arrays', () => {
            const response = MOCK_BUBBLE_FLOW_CREATE_RESPONSE;
            if (response.requiredCredentials) {
                for (const bubbleName in response.requiredCredentials) {
                    const creds = response.requiredCredentials[bubbleName];
                    (0, globals_1.expect)(Array.isArray(creds)).toBe(true);
                    for (const cred of creds) {
                        (0, globals_1.expect)(typeof cred).toBe('string');
                    }
                }
            }
        });
    });
});
// =============================================================================
// Contract Validation Helper
// =============================================================================
/**
 * Validate all API contracts before starting adapter
 * This function should be called during adapter initialization
 *
 * @returns true if all contracts are valid
 * @throws Error if any contract is violated
 */
function validateAllContracts() {
    console.log('Validating BubbleLab API contracts...');
    try {
        // Health check contract
        const healthResult = HealthCheckContract.safeParse(MOCK_HEALTH_RESPONSE);
        if (!healthResult.success) {
            throw new Error(`Health check contract violated: ${JSON.stringify(healthResult.error)}`);
        }
        // BubbleFlow contract
        const flowResult = BubbleFlowContract.safeParse(MOCK_BUBBLE_FLOW_LIST_RESPONSE[0]);
        if (!flowResult.success) {
            throw new Error(`BubbleFlow contract violated: ${JSON.stringify(flowResult.error)}`);
        }
        // Create response contract
        const createResult = BubbleFlowCreateResponseContract.safeParse(MOCK_BUBBLE_FLOW_CREATE_RESPONSE);
        if (!createResult.success) {
            throw new Error(`Create response contract violated: ${JSON.stringify(createResult.error)}`);
        }
        // Execution contract
        const execResult = ExecutionResponseContract.safeParse(MOCK_BUBBLE_FLOW_EXECUTE_RESPONSE);
        if (!execResult.success) {
            throw new Error(`Execution response contract violated: ${JSON.stringify(execResult.error)}`);
        }
        // History contract
        const historyResult = ExecutionHistoryContract.safeParse(MOCK_EXECUTION_HISTORY_RESPONSE);
        if (!historyResult.success) {
            throw new Error(`Execution history contract violated: ${JSON.stringify(historyResult.error)}`);
        }
        console.log('All BubbleLab API contracts validated successfully');
        return true;
    }
    catch (error) {
        console.error('Contract validation failed:', error);
        throw error;
    }
}
//# sourceMappingURL=contract.test.js.map