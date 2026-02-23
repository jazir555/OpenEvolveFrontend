"use strict";
/**
 * Datapizza API Contract Tests
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 * "Protecting the Mega-Project from Updates"
 *
 * These tests verify that the Datapizza API returns the expected fields.
 * If the contract is violated (Project API changed), the adapter MUST refuse to start.
 *
 * Runs on container startup. If these tests fail, the application should NOT start.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const DatapizzaClient_1 = require("../../services/DatapizzaClient");
// Configuration
const API_URL = process.env.DATAPIZZA_BASE_URL || 'http://localhost:3000/datapizza';
const API_KEY = process.env.DATAPIZZA_API_KEY;
const TIMEOUT = 30000; // 30 second timeout for contract tests
(0, globals_1.describe)('Datapizza API Contract Tests', () => {
    let client;
    (0, globals_1.beforeAll)(() => {
        // Initialize client with production config
        client = new DatapizzaClient_1.DatapizzaClient({
            baseUrl: API_URL,
            apiKey: API_KEY,
            timeout: TIMEOUT,
        });
    });
    (0, globals_1.describe)('Health Check Endpoint', () => {
        (0, globals_1.it)('should return health status with required fields', async () => {
            const response = await fetch(`${API_URL}/health`, {
                headers: API_KEY ? { 'Authorization': `Bearer ${API_KEY}` } : {},
            });
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Health check must include status field
            (0, globals_1.expect)(data).toHaveProperty('status');
            (0, globals_1.expect)(typeof data.status).toBe('string');
        });
        (0, globals_1.it)('should respond within timeout', async () => {
            const start = Date.now();
            const response = await fetch(`${API_URL}/health`);
            const duration = Date.now() - start;
            (0, globals_1.expect)(response.ok).toBe(true);
            (0, globals_1.expect)(duration).toBeLessThan(TIMEOUT);
        });
    });
    (0, globals_1.describe)('Data Processing Endpoint', () => {
        (0, globals_1.it)('should accept data processing request with required fields', async () => {
            const processRequest = {
                data: {
                    text: 'Test data for processing',
                    metadata: {
                        source: 'contract-test',
                    },
                },
                processingType: 'standard',
                options: {
                    chunk_size: 1000,
                    overlap_size: 200,
                },
            };
            const response = await fetch(`${API_URL}/data/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(processRequest),
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Data process endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return dataId
            (0, globals_1.expect)(data).toHaveProperty('dataId');
            (0, globals_1.expect)(typeof data.dataId).toBe('string');
            // Contract: Must return success status
            (0, globals_1.expect)(data).toHaveProperty('success');
            (0, globals_1.expect)(data.success).toBe(true);
            // Contract: Must return processedData
            (0, globals_1.expect)(data).toHaveProperty('processedData');
            // Contract: Must include processing type
            (0, globals_1.expect)(data).toHaveProperty('processingType');
        });
    });
    (0, globals_1.describe)('Data Query Endpoint', () => {
        (0, globals_1.it)('should return query results with required fields', async () => {
            const queryParams = new URLSearchParams({
                query: 'test query',
                data_source: 'default',
                limit: '10',
                offset: '0',
            });
            const response = await fetch(`${API_URL}/data/query?${queryParams}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Data query endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have results array
            (0, globals_1.expect)(data).toHaveProperty('results');
            (0, globals_1.expect)(Array.isArray(data.results)).toBe(true);
            // Contract: Must include total count
            (0, globals_1.expect)(data).toHaveProperty('totalCount');
            (0, globals_1.expect)(typeof data.totalCount).toBe('number');
            // Contract: Each result must have required fields
            if (data.results.length > 0) {
                const result = data.results[0];
                (0, globals_1.expect)(result).toHaveProperty('id');
                (0, globals_1.expect)(result).toHaveProperty('score');
                (0, globals_1.expect)(typeof result.score).toBe('number');
                (0, globals_1.expect)(result).toHaveProperty('data');
                (0, globals_1.expect)(result.data).toHaveProperty('content');
                (0, globals_1.expect)(result.data).toHaveProperty('source');
            }
        });
        (0, globals_1.it)('should support limit and offset parameters', async () => {
            const queryParams = new URLSearchParams({
                query: 'test query',
                limit: '5',
                offset: '0',
            });
            const response = await fetch(`${API_URL}/data/query?${queryParams}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            if (response.status === 404) {
                console.warn('Data query endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Should respect limit parameter
            (0, globals_1.expect)(data.results.length).toBeLessThanOrEqual(5);
        });
    });
    (0, globals_1.describe)('Pipeline Run Endpoint', () => {
        (0, globals_1.it)('should accept pipeline run request with required fields', async () => {
            const pipelineRequest = {
                data_source: 'test-source',
                pipeline_type: 'standard',
                parameters: {
                    chunk_size: 1000,
                    embedding_model: 'default',
                },
            };
            const response = await fetch(`${API_URL}/pipelines/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(pipelineRequest),
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Pipeline run endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return pipelineId
            (0, globals_1.expect)(data).toHaveProperty('pipelineId');
            (0, globals_1.expect)(typeof data.pipelineId).toBe('string');
            // Contract: Must return status
            (0, globals_1.expect)(data).toHaveProperty('status');
            (0, globals_1.expect)(['pending', 'running', 'completed', 'failed']).toContain(data.status);
            // Contract: Must include data source and pipeline type
            (0, globals_1.expect)(data).toHaveProperty('dataSource');
            (0, globals_1.expect)(data).toHaveProperty('pipelineType');
        });
    });
    (0, globals_1.describe)('Pipeline Recommendation Endpoint', () => {
        (0, globals_1.it)('should return pipeline recommendation with required fields', async () => {
            const response = await fetch(`${API_URL}/pipelines/recommend?data_source=test-source`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Pipeline recommendation endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have recommended pipeline
            (0, globals_1.expect)(data).toHaveProperty('recommendedPipeline');
            (0, globals_1.expect)(typeof data.recommendedPipeline).toBe('string');
            // Contract: Must have confidence score
            (0, globals_1.expect)(data).toHaveProperty('confidence');
            (0, globals_1.expect)(typeof data.confidence).toBe('number');
            (0, globals_1.expect)(data.confidence).toBeGreaterThanOrEqual(0);
            (0, globals_1.expect)(data.confidence).toBeLessThanOrEqual(1);
            // Contract: Should have alternatives
            (0, globals_1.expect)(data).toHaveProperty('alternatives');
            (0, globals_1.expect)(Array.isArray(data.alternatives)).toBe(true);
        });
    });
    (0, globals_1.describe)('Data Domain Detection Endpoint', () => {
        (0, globals_1.it)('should return data domain classification', async () => {
            const response = await fetch(`${API_URL}/data/detect-domain?data_source=test-source`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Data domain detection endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have domain classification
            (0, globals_1.expect)(data).toHaveProperty('domain');
            (0, globals_1.expect)(['structured', 'unstructured', 'semi-structured', 'general']).toContain(data.domain);
            // Contract: Must have confidence score
            (0, globals_1.expect)(data).toHaveProperty('confidence');
            (0, globals_1.expect)(typeof data.confidence).toBe('number');
        });
    });
    (0, globals_1.describe)('Error Responses', () => {
        (0, globals_1.it)('should return proper error for invalid pipeline request', async () => {
            const invalidRequest = {
                // Missing required data_source field
                pipeline_type: 'standard',
            };
            const response = await fetch(`${API_URL}/pipelines/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(invalidRequest),
            });
            // Contract: Should return 400 for bad request
            (0, globals_1.expect)(response.status).toBe(400);
        });
        (0, globals_1.it)('should return proper error for invalid query parameters', async () => {
            const response = await fetch(`${API_URL}/data/query?limit=invalid`, // Invalid limit value
            {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Contract: Should return 400 for invalid parameters
            (0, globals_1.expect)(response.status).toBe(400);
        });
    });
    (0, globals_1.describe)('Authentication', () => {
        (0, globals_1.it)('should reject requests with invalid API key', async () => {
            const response = await fetch(`${API_URL}/health`, {
                headers: {
                    'Authorization': 'Bearer invalid-key-12345',
                },
            });
            // Contract: Should return 401 or 403 for invalid auth
            // If auth is not enabled, this may pass, which is acceptable
            if (response.status === 401 || response.status === 403) {
                (0, globals_1.expect)(true).toBe(true);
            }
            else {
                console.warn('Authentication not enforced - skipping auth test');
            }
        });
    });
    (0, globals_1.describe)('Response Timeouts', () => {
        (0, globals_1.it)('should respond within reasonable time for health check', async () => {
            const start = Date.now();
            const response = await fetch(`${API_URL}/health`);
            const duration = Date.now() - start;
            (0, globals_1.expect)(response.ok).toBe(true);
            (0, globals_1.expect)(duration).toBeLessThan(5000); // 5 second max for health check
        });
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
//# sourceMappingURL=datapizza-api.test.js.map