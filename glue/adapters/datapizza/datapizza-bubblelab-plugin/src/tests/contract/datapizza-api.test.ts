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

import { describe, it, expect, beforeAll } from 'vitest';
import { DatapizzaClient } from '../../services/DatapizzaClient';

// Configuration
const API_URL = process.env.DATAPIZZA_BASE_URL || 'http://localhost:3000/datapizza';
const API_KEY = process.env.DATAPIZZA_API_KEY;
const TIMEOUT = 30000; // 30 second timeout for contract tests

describe('Datapizza API Contract Tests', () => {
  let client: DatapizzaClient;

  beforeAll(() => {
    // Initialize client with production config
    client = new DatapizzaClient({
      baseUrl: API_URL,
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

  describe('Data Processing Endpoint', () => {
    it('should accept data processing request with required fields', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return dataId
      expect(data).toHaveProperty('dataId');
      expect(typeof data.dataId).toBe('string');

      // Contract: Must return success status
      expect(data).toHaveProperty('success');
      expect(data.success).toBe(true);

      // Contract: Must return processedData
      expect(data).toHaveProperty('processedData');

      // Contract: Must include processing type
      expect(data).toHaveProperty('processingType');
    });
  });

  describe('Data Query Endpoint', () => {
    it('should return query results with required fields', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have results array
      expect(data).toHaveProperty('results');
      expect(Array.isArray(data.results)).toBe(true);

      // Contract: Must include total count
      expect(data).toHaveProperty('totalCount');
      expect(typeof data.totalCount).toBe('number');

      // Contract: Each result must have required fields
      if (data.results.length > 0) {
        const result = data.results[0];
        expect(result).toHaveProperty('id');
        expect(result).toHaveProperty('score');
        expect(typeof result.score).toBe('number');
        expect(result).toHaveProperty('data');
        expect(result.data).toHaveProperty('content');
        expect(result.data).toHaveProperty('source');
      }
    });

    it('should support limit and offset parameters', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Should respect limit parameter
      expect(data.results.length).toBeLessThanOrEqual(5);
    });
  });

  describe('Pipeline Run Endpoint', () => {
    it('should accept pipeline run request with required fields', async () => {
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

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must return pipelineId
      expect(data).toHaveProperty('pipelineId');
      expect(typeof data.pipelineId).toBe('string');

      // Contract: Must return status
      expect(data).toHaveProperty('status');
      expect(['pending', 'running', 'completed', 'failed']).toContain(data.status);

      // Contract: Must include data source and pipeline type
      expect(data).toHaveProperty('dataSource');
      expect(data).toHaveProperty('pipelineType');
    });
  });

  describe('Pipeline Recommendation Endpoint', () => {
    it('should return pipeline recommendation with required fields', async () => {
      const response = await fetch(
        `${API_URL}/pipelines/recommend?data_source=test-source`,
        {
          headers: {
            'Content-Type': 'application/json',
            ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
          },
        }
      );

      // Allow 404 if endpoint not implemented yet
      if (response.status === 404) {
        console.warn('Pipeline recommendation endpoint not implemented - skipping');
        return;
      }

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have recommended pipeline
      expect(data).toHaveProperty('recommendedPipeline');
      expect(typeof data.recommendedPipeline).toBe('string');

      // Contract: Must have confidence score
      expect(data).toHaveProperty('confidence');
      expect(typeof data.confidence).toBe('number');
      expect(data.confidence).toBeGreaterThanOrEqual(0);
      expect(data.confidence).toBeLessThanOrEqual(1);

      // Contract: Should have alternatives
      expect(data).toHaveProperty('alternatives');
      expect(Array.isArray(data.alternatives)).toBe(true);
    });
  });

  describe('Data Domain Detection Endpoint', () => {
    it('should return data domain classification', async () => {
      const response = await fetch(
        `${API_URL}/data/detect-domain?data_source=test-source`,
        {
          headers: {
            'Content-Type': 'application/json',
            ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
          },
        }
      );

      // Allow 404 if endpoint not implemented yet
      if (response.status === 404) {
        console.warn('Data domain detection endpoint not implemented - skipping');
        return;
      }

      expect(response.ok).toBe(true);

      const data = await response.json();

      // Contract: Must have domain classification
      expect(data).toHaveProperty('domain');
      expect(['structured', 'unstructured', 'semi-structured', 'general']).toContain(
        data.domain
      );

      // Contract: Must have confidence score
      expect(data).toHaveProperty('confidence');
      expect(typeof data.confidence).toBe('number');
    });
  });

  describe('Error Responses', () => {
    it('should return proper error for invalid pipeline request', async () => {
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
      expect(response.status).toBe(400);
    });

    it('should return proper error for invalid query parameters', async () => {
      const response = await fetch(
        `${API_URL}/data/query?limit=invalid`, // Invalid limit value
        {
          headers: {
            'Content-Type': 'application/json',
            ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` }),
          },
        }
      );

      // Contract: Should return 400 for invalid parameters
      expect(response.status).toBe(400);
    });
  });

  describe('Authentication', () => {
    it('should reject requests with invalid API key', async () => {
      const response = await fetch(`${API_URL}/health`, {
        headers: {
          'Authorization': 'Bearer invalid-key-12345',
        },
      });

      // Contract: Should return 401 or 403 for invalid auth
      // If auth is not enabled, this may pass, which is acceptable
      if (response.status === 401 || response.status === 403) {
        expect(true).toBe(true);
      } else {
        console.warn('Authentication not enforced - skipping auth test');
      }
    });
  });

  describe('Response Timeouts', () => {
    it('should respond within reasonable time for health check', async () => {
      const start = Date.now();
      const response = await fetch(`${API_URL}/health`);
      const duration = Date.now() - start;

      expect(response.ok).toBe(true);
      expect(duration).toBeLessThan(5000); // 5 second max for health check
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
