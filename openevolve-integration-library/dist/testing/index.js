"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createMockClient = createMockClient;
const client_1 = require("../api/client");
const client_2 = require("../api/client");
function createMockClient(mockResponses = {}, mockErrors = {}) {
    const client = new client_1.OpenEvolveClient({
        baseUrl: 'http://mock-client',
        enableWebSocket: false
    });
    const backend = client.getBackend();
    jest.spyOn(backend, 'post').mockImplementation(async (endpoint, data) => {
        const integration = endpoint.split('/').find(p => Object.values(client_2.IntegrationName).includes(p));
        if (integration && mockErrors[integration]) {
            throw mockErrors[integration];
        }
        if (integration && mockResponses[integration]) {
            return mockResponses[integration];
        }
        return { success: true, mock: true, endpoint, data };
    });
    jest.spyOn(backend, 'get').mockImplementation(async (endpoint) => {
        const integration = endpoint.split('/').find(p => Object.values(client_2.IntegrationName).includes(p));
        if (integration && mockErrors[integration]) {
            throw mockErrors[integration];
        }
        return { success: true, mock: true, endpoint };
    });
    jest.spyOn(backend, 'ping').mockResolvedValue(true);
    jest.spyOn(backend, 'getStatus').mockResolvedValue({
        online: true,
        version: '1.0.0-mock',
        uptime: 1000,
        activeConnections: 0,
        memory: { used: 0, total: 0, percentage: 0 },
        cpu: 0
    });
    jest.spyOn(client, 'execute').mockImplementation(async (integration, inputs, options) => {
        if (mockErrors[integration]) {
            throw mockErrors[integration];
        }
        if (mockResponses[integration]) {
            return mockResponses[integration];
        }
        return { success: true, mock: true, integration, inputs, executionId: options?.executionId };
    });
    jest.spyOn(client, 'executeStream').mockImplementation(async (integration, inputs, onProgress, options) => {
        const executionId = options?.executionId || 'mock-id';
        if (mockErrors[integration]) {
            throw mockErrors[integration];
        }
        onProgress({
            integration,
            executionId,
            progress: 100,
            message: 'Complete',
            timestamp: new Date().toISOString()
        });
        if (mockResponses[integration]) {
            return mockResponses[integration];
        }
        return { success: true, mock: true, integration, inputs, executionId };
    });
    return client;
}
//# sourceMappingURL=index.js.map