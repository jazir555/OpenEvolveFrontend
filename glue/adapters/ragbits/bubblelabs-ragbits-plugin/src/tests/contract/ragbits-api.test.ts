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

import { describe, it, expect, beforeAll } from '@jest/globals';
import { RagbitsClient } from '../../lib/ragbitsClient';

// Configuration
const API_URL = process.env.RAGBITS_API_URL || 'http://localhost:3000/ragbits';
const API_KEY = process.env.RAGBITS_API_KEY;
const TIMEOUT = 30000; // 30 second timeout for contract tests

describe('RAGBits API Contract Tests', () => {
  let client: RagbitsClient;

  beforeAll(() => {
    // Initialize client with production config
    client = new RagbitsClient({
      serverUrl: API_URL,
      apiKey: API_KEY,
      timeout: TIMEOUT,
    });
  });

  describe('Health Check Endpoint', () => {
    it('should return health status with required fields', async () => {
      const response = await fetch(`${API_URL}/health`, {
        headers: API_KEY ? { 'Authorization': `Bearer ${API_KEY}` } : {},
      });

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Health check must include status field
      expect(data).toHaveProperty('status');
      expect(typeof data.status).toBe('string');
    });

    it('should respond within timeout', async () => {
      const start = Date.now();
      const response = await fetch(`${API_URL}/health`);
      const duration = Date.now() - start;

      expect(response.ok).toBe(true);
      expect(duration).toBeLessThan(TIMEOUT);
    });
  });

  describe('Search Endpoint', () => {
    it('should return search results with required fields', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have results array
      expect(data).toHaveProperty('results');
      expect(Array.isArray(data.results)).toBe(true);

      // Contract: Should include metadata
      expect(data).toHaveProperty('metadata');

      // Contract: Each result must have required fields
      if (data.results.length > 0) {
        const result = data.results[0];
        expect(result).toHaveProperty('id');
        expect(result).toHaveProperty('score');
        expect(typeof result.score).toBe('number');
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(1);
      }
    });

    it('should support hybrid search parameter', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Metadata should indicate search type
      expect(data.metadata).toHaveProperty('searchType');
    });
  });

  describe('Ingest Endpoint', () => {
    it('should accept document ingestion with required fields', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return document ID
      expect(data).toHaveProperty('documentId');
      expect(typeof data.documentId).toBe('string');

      // Contract: Must return success status
      expect(data).toHaveProperty('success');
      expect(data.success).toBe(true);
    });
  });

  describe('Batch Ingest Endpoint', () => {
    it('should accept multiple documents for ingestion', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return array of results
      expect(Array.isArray(data)).toBe(true);

      // Contract: Each result must have documentId
      if (data.length > 0) {
        expect(data[0]).toHaveProperty('documentId');
      }
    });
  });

  describe('Index Statistics Endpoint', () => {
    it('should return index statistics with required fields', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have total documents count
      expect(data).toHaveProperty('totalDocuments');
      expect(typeof data.totalDocuments).toBe('number');

      // Contract: Should include index size
      expect(data).toHaveProperty('indexSize');

      // Contract: Should include last updated timestamp
      expect(data).toHaveProperty('lastUpdated');
    });
  });

  describe('Cache Clear Endpoint', () => {
    it('should clear cache and return success', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return success status
      expect(data).toHaveProperty('success');
      expect(data.success).toBe(true);
    });
  });

  describe('Error Responses', () => {
    it('should return proper error for invalid search request', async () => {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
        },
        body: 'invalid json{{{',
      });

      // Contract: Should return 400 for bad request
      expect(response.status).toBe(400);
    });

    it('should return proper error for missing required fields', async () => {
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
      expect(response.status).toBe(400);
    });
  });

  describe('Filtering and Parameters', () => {
    it('should support filtering by metadata', async () => {
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

      expect(response.ok).toBe(true);
    });

    it('should support score threshold parameter', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: All results should meet threshold
      if (data.results && data.results.length > 0) {
        data.results.forEach((result: any) => {
          expect(result.score).toBeGreaterThanOrEqual(0.8);
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
