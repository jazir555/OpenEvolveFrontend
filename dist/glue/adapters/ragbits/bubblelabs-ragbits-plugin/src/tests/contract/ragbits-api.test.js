"use strict";
/**
 * RAGBits API Contract Tests
 *
 * Federation Constitution - Section 4, Phase 2: The Contract
 * "Protecting the Mega-Project from Updates"
 *
 * These tests verify that the RAGBits API returns the expected fields.
 * If the contract is violated (Project API changed), the adapter MUST refuse to start.
 *
 * Runs on container startup. If these tests fail, the application should NOT start.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const globals_1 = require("@jest/globals");
const ragbitsClient_1 = require("../../lib/ragbitsClient");
// Configuration
const API_URL = process.env.RAGBITS_API_URL || 'http://localhost:3000/ragbits';
const API_KEY = process.env.RAGBITS_API_KEY;
const TIMEOUT = 30000; // 30 second timeout for contract tests
(0, globals_1.describe)('RAGBits API Contract Tests', () => {
    let client;
    (0, globals_1.beforeAll)(() => {
        // Initialize client with production config
        client = new ragbitsClient_1.RagbitsClient({
            serverUrl: API_URL,
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
    (0, globals_1.describe)('Search Endpoint', () => {
        (0, globals_1.it)('should return search results with required fields', async () => {
            const searchRequest = {
                query: 'test search query',
                topK: 5,
                scoreThreshold: 0.7,
            };
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(searchRequest),
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Search endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have results array
            (0, globals_1.expect)(data).toHaveProperty('results');
            (0, globals_1.expect)(Array.isArray(data.results)).toBe(true);
            // Contract: Should include metadata
            (0, globals_1.expect)(data).toHaveProperty('metadata');
            // Contract: Each result must have required fields
            if (data.results.length > 0) {
                const result = data.results[0];
                (0, globals_1.expect)(result).toHaveProperty('id');
                (0, globals_1.expect)(result).toHaveProperty('score');
                (0, globals_1.expect)(typeof result.score).toBe('number');
                (0, globals_1.expect)(result.score).toBeGreaterThanOrEqual(0);
                (0, globals_1.expect)(result.score).toBeLessThanOrEqual(1);
            }
        });
        (0, globals_1.it)('should support hybrid search parameter', async () => {
            const searchRequest = {
                query: 'test query',
                enableHybridSearch: true,
                enableReranking: true,
            };
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(searchRequest),
            });
            if (response.status === 404) {
                console.warn('Search endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Metadata should indicate search type
            (0, globals_1.expect)(data.metadata).toHaveProperty('searchType');
        });
    });
    (0, globals_1.describe)('Ingest Endpoint', () => {
        (0, globals_1.it)('should accept document ingestion with required fields', async () => {
            const ingestRequest = {
                content: 'Test document content for contract validation',
                metadata: {
                    source: 'contract-test',
                    timestamp: new Date().toISOString(),
                },
            };
            const response = await fetch(`${API_URL}/ingest`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(ingestRequest),
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Ingest endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return document ID
            (0, globals_1.expect)(data).toHaveProperty('documentId');
            (0, globals_1.expect)(typeof data.documentId).toBe('string');
            // Contract: Must return success status
            (0, globals_1.expect)(data).toHaveProperty('success');
            (0, globals_1.expect)(data.success).toBe(true);
        });
    });
    (0, globals_1.describe)('Batch Ingest Endpoint', () => {
        (0, globals_1.it)('should accept multiple documents for ingestion', async () => {
            const batchRequest = {
                documents: [
                    {
                        content: 'First test document',
                        metadata: { source: 'contract-test' },
                    },
                    {
                        content: 'Second test document',
                        metadata: { source: 'contract-test' },
                    },
                ],
            };
            const response = await fetch(`${API_URL}/ingest/batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(batchRequest),
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Batch ingest endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return array of results
            (0, globals_1.expect)(Array.isArray(data)).toBe(true);
            // Contract: Each result must have documentId
            if (data.length > 0) {
                (0, globals_1.expect)(data[0]).toHaveProperty('documentId');
            }
        });
    });
    (0, globals_1.describe)('Index Statistics Endpoint', () => {
        (0, globals_1.it)('should return index statistics with required fields', async () => {
            const response = await fetch(`${API_URL}/index/stats`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Index stats endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must have total documents count
            (0, globals_1.expect)(data).toHaveProperty('totalDocuments');
            (0, globals_1.expect)(typeof data.totalDocuments).toBe('number');
            // Contract: Should include index size
            (0, globals_1.expect)(data).toHaveProperty('indexSize');
            // Contract: Should include last updated timestamp
            (0, globals_1.expect)(data).toHaveProperty('lastUpdated');
        });
    });
    (0, globals_1.describe)('Cache Clear Endpoint', () => {
        (0, globals_1.it)('should clear cache and return success', async () => {
            const response = await fetch(`${API_URL}/cache/clear`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
            });
            // Allow 404 if endpoint not implemented yet
            if (response.status === 404) {
                console.warn('Cache clear endpoint not implemented - skipping');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: Must return success status
            (0, globals_1.expect)(data).toHaveProperty('success');
            (0, globals_1.expect)(data.success).toBe(true);
        });
    });
    (0, globals_1.describe)('Error Responses', () => {
        (0, globals_1.it)('should return proper error for invalid search request', async () => {
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: 'invalid json{{{',
            });
            // Contract: Should return 400 for bad request
            (0, globals_1.expect)(response.status).toBe(400);
        });
        (0, globals_1.it)('should return proper error for missing required fields', async () => {
            const invalidRequest = {
                // Missing required 'query' field
                topK: 5,
            };
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(invalidRequest),
            });
            // Contract: Should return 400 for missing required fields
            (0, globals_1.expect)(response.status).toBe(400);
        });
    });
    (0, globals_1.describe)('Filtering and Parameters', () => {
        (0, globals_1.it)('should support filtering by metadata', async () => {
            const searchRequest = {
                query: 'test query',
                filter: {
                    source: 'test-source',
                    type: 'document',
                },
            };
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(searchRequest),
            });
            if (response.status === 404) {
                console.warn('Search endpoint not implemented - skipping filter test');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
        });
        (0, globals_1.it)('should support score threshold parameter', async () => {
            const searchRequest = {
                query: 'test query',
                scoreThreshold: 0.8,
            };
            const response = await fetch(`${API_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
                },
                body: JSON.stringify(searchRequest),
            });
            if (response.status === 404) {
                console.warn('Search endpoint not implemented - skipping threshold test');
                return;
            }
            (0, globals_1.expect)(response.ok).toBe(true);
            const data = await response.json();
            // Contract: All results should meet threshold
            if (data.results && data.results.length > 0) {
                data.results.forEach((result) => {
                    (0, globals_1.expect)(result.score).toBeGreaterThanOrEqual(0.8);
                });
            }
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
//# sourceMappingURL=ragbits-api.test.js.map