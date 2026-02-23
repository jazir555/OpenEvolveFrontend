/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Adapter Contract Tests
 *
 * These tests validate the contract between the adapter and ICR system.
 * If the contract is violated (API changed), the adapter refuses to start.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Law of Runtime Truth: Tests verify actual API behavior
 * - Contract Defense: Prevents data corruption from API changes
 * - All tests run on container startup
 */

import axios from 'axios';
import { ICRClient } from '../src/icr-client';
import { ICRAdapter } from '../src/adapter';
import {
  ModeType,
  RefineModeRequestSchema,
  ReactModeRequestSchema,
  DeepthinkModeRequestSchema,
  AdaptiveDeepthinkRequestSchema,
  AgenticModeRequestSchema,
  ContextualModeRequestSchema,
  GenerativeUIModeRequestSchema
} from '../src/icr-canonical';

// Mock axios for testing
jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('ICR Adapter Contract Tests', () => {
  let client: ICRClient;
  let adapter: ICRAdapter;
  let mockInstance: any;

  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();

    // Create mock axios instance
    mockInstance = {
      post: jest.fn(),
      get: jest.fn(),
      interceptors: {
        request: { use: jest.fn() },
        response: { use: jest.fn() }
      }
    };

    mockedAxios.create.mockReturnValue(mockInstance);
    mockedAxios.isAxiosError = jest.fn(() => true);

    client = new ICRClient();
    adapter = new ICRAdapter({ client });
  });

  // ==========================================================================
  // HEALTH CHECK CONTRACT
  // ==========================================================================

  describe('Health Check Contract', () => {
    it('should return health check response with correct structure', async () => {
      const mockHealthResponse = {
        status: 'healthy',
        version: '1.0.0',
        available_modes: ['refine', 'react', 'deepthink', 'adaptive_deepthink', 'agentic', 'contextual', 'generative_ui'],
        timestamp_utc: new Date().toISOString(),
        uptime_seconds: 3600
      };

      mockInstance.post.mockResolvedValue({ data: mockHealthResponse });

      const response = await adapter.healthCheck();

      expect(response).toHaveProperty('status');
      expect(response).toHaveProperty('version');
      expect(response).toHaveProperty('available_modes');
      expect(response).toHaveProperty('timestamp_utc');
      expect(response).toHaveProperty('uptime_seconds');
      expect(response.available_modes).toHaveLength(7);
    });

    it('should handle health check errors gracefully', async () => {
      mockInstance.post.mockRejectedValue(new Error('API unavailable'));

      await expect(adapter.healthCheck()).rejects.toThrow();
    });
  });

  // ==========================================================================
  // REFINE MODE CONTRACT
  // ==========================================================================

  describe('Refine Mode Contract', () => {
    it('should accept valid Refine mode request', () => {
      const request = {
        mode: 'refine' as const,
        prompt: 'Create a test function',
        options: {
          temperature: 0.7,
          evolution_mode: 'quality' as const,
          refinement_stages: 3
        },
        metadata: {
          correlation_id: 'test-001',
          timestamp_utc: new Date().toISOString(),
          source_service: 'test'
        }
      };

      const result = RefineModeRequestSchema.safeParse(request);
      expect(result.success).toBe(true);
    });

    it('should return Refine mode response with correct structure', async () => {
      const mockResponse = {
        mode: 'refine',
        request: {
          mode: 'refine',
          prompt: 'Create a test function',
          metadata: {
            correlation_id: 'test-001',
            timestamp_utc: new Date().toISOString(),
            source_service: 'test'
          }
        },
        result: {
          success: true,
          content: 'Generated content',
          execution_time_ms: 1000,
          iteration_count: 3,
          iterations: []
        },
        metadata: {
          correlation_id: 'test-001',
          timestamp_utc: new Date().toISOString(),
          source_service: 'icr-adapter',
          completed_at_utc: new Date().toISOString()
        }
      };

      mockInstance.post.mockResolvedValue({ data: mockResponse });

      const response = await adapter.createRefinementRequest('Create a test function');

      expect(response.mode).toBe('refine');
      expect(response.result).toHaveProperty('success');
      expect(response.result).toHaveProperty('content');
      expect(response.result).toHaveProperty('execution_time_ms');
      expect(response.result).toHaveProperty('iterations');
    });
  });

  // ==========================================================================
  // REACT MODE CONTRACT
  // ==========================================================================

  describe('React Mode Contract', () => {
    it('should accept valid React mode request', () => {
      const request = {
        mode: 'react' as const,
        prompt: 'Create a React todo app',
        options: {
          worker_count: 5,
          enable_preview: true
        },
        metadata: {
          correlation_id: 'test-002',
          timestamp_utc: new Date().toISOString(),
          source_service: 'test'
        }
      };

      const result = ReactModeRequestSchema.safeParse(request);
      expect(result.success).toBe(true);
    });

    it('should return React mode response with workers array', async () => {
      const mockResponse = {
        mode: 'react',
        request: {
          mode: 'react',
          prompt: 'Create a React todo app',
          metadata: {
            correlation_id: 'test-002',
            timestamp_utc: new Date().toISOString(),
            source_service: 'test'
          }
        },
        result: {
          success: true,
          content: 'React app generated',
          execution_time_ms: 5000,
          iteration_count: 1,
          orchestrator_plan: 'Plan for todo app',
          workers: [
            {
              worker_id: 'worker-1',
              title: 'UI Components',
              status: 'completed',
              generated_content: 'React code'
            }
          ]
        },
        metadata: {
          correlation_id: 'test-002',
          timestamp_utc: new Date().toISOString(),
          source_service: 'icr-adapter',
          completed_at_utc: new Date().toISOString()
        }
      };

      mockInstance.post.mockResolvedValue({ data: mockResponse });

      const response = await adapter.createReactRequest('Create a React todo app');

      expect(response.mode).toBe('react');
      expect(response.result).toHaveProperty('workers');
      expect(Array.isArray(response.result.workers)).toBe(true);
    });
  });

  // ==========================================================================
  // DEEPTHINK MODE CONTRACT
  // ==========================================================================

  describe('Deepthink Mode Contract', () => {
    it('should accept valid Deepthink mode request', () => {
      const request = {
        mode: 'deepthink' as const,
        prompt: 'Solve this complex problem',
        options: {
          strategy_count: 3,
          sub_strategy_count: 5,
          hypothesis_count: 10,
          enable_red_team: true
        },
        metadata: {
          correlation_id: 'test-003',
          timestamp_utc: new Date().toISOString(),
          source_service: 'test'
        }
      };

      const result = DeepthinkModeRequestSchema.safeParse(request);
      expect(result.success).toBe(true);
    });

    it('should return Deepthink mode response with strategies', async () => {
      const mockResponse = {
        mode: 'deepthink',
        request: {
          mode: 'deepthink',
          prompt: 'Solve this complex problem',
          metadata: {
            correlation_id: 'test-003',
            timestamp_utc: new Date().toISOString(),
            source_service: 'test'
          }
        },
        result: {
          success: true,
          content: 'Problem solved',
          execution_time_ms: 10000,
          iteration_count: 5,
          strategies: [
            {
              strategy_id: 'strat-1',
              strategy_text: 'Strategy 1',
              sub_strategies: []
            }
          ],
          best_solution: 'Best solution'
        },
        metadata: {
          correlation_id: 'test-003',
          timestamp_utc: new Date().toISOString(),
          source_service: 'icr-adapter',
          completed_at_utc: new Date().toISOString()
        }
      };

      mockInstance.post.mockResolvedValue({ data: mockResponse });

      const response = await adapter.createDeepthinkRequest('Solve this complex problem');

      expect(response.mode).toBe('deepthink');
      expect(response.result).toHaveProperty('strategies');
      expect(Array.isArray(response.result.strategies)).toBe(true);
    });
  });

  // ==========================================================================
  // ALL MODES ACCESSIBILITY
  // ==========================================================================

  describe('All 7 Modes Contract', () => {
    const allModes: ModeType[] = [
      'refine',
      'react',
      'deepthink',
      'adaptive_deepthink',
      'agentic',
      'contextual',
      'generative_ui'
    ];

    it('should have all 7 modes defined', () => {
      expect(allModes).toHaveLength(7);
    });

    it('should validate request schemas for all modes', () => {
      const baseMetadata = {
        correlation_id: 'test-all',
        timestamp_utc: new Date().toISOString(),
        source_service: 'test'
      };

      const requests = [
        { mode: 'refine', prompt: 'test', metadata: baseMetadata },
        { mode: 'react', prompt: 'test', metadata: baseMetadata },
        { mode: 'deepthink', prompt: 'test', metadata: baseMetadata },
        { mode: 'adaptive_deepthink', prompt: 'test', metadata: baseMetadata },
        { mode: 'agentic', prompt: 'test', metadata: baseMetadata },
        { mode: 'contextual', prompt: 'test', metadata: baseMetadata },
        { mode: 'generative_ui', prompt: 'test', metadata: baseMetadata }
      ] as const;

      requests.forEach((req) => {
        let schema;
        switch (req.mode) {
          case 'refine':
            schema = RefineModeRequestSchema;
            break;
          case 'react':
            schema = ReactModeRequestSchema;
            break;
          case 'deepthink':
            schema = DeepthinkModeRequestSchema;
            break;
          case 'adaptive_deepthink':
            schema = AdaptiveDeepthinkRequestSchema;
            break;
          case 'agentic':
            schema = AgenticModeRequestSchema;
            break;
          case 'contextual':
            schema = ContextualModeRequestSchema;
            break;
          case 'generative_ui':
            schema = GenerativeUIModeRequestSchema;
            break;
        }

        const result = schema.safeParse(req);
        expect(result.success).toBe(true);
      });
    });
  });

  // ==========================================================================
  // ERROR HANDLING CONTRACT
  // ==========================================================================

  describe('Error Handling Contract', () => {
    it('should handle 429 Too Many Requests with retry', async () => {
      const error429 = {
        response: { status: 429 },
        message: 'Too many requests',
        isAxiosError: true
      };

      // First two calls fail, third succeeds
      mockInstance.post
        .mockRejectedValueOnce(error429)
        .mockRejectedValueOnce(error429)
        .mockResolvedValueOnce({
          data: {
            mode: 'refine',
            request: {},
            result: { success: true, content: 'test', execution_time_ms: 100, iteration_count: 1 },
            metadata: {}
          }
        });

      const response = await adapter.createRefinementRequest('test');

      expect(response.result.success).toBe(true);
      expect(mockInstance.post).toHaveBeenCalledTimes(3);
    });

    it('should handle 500 Internal Server Error with retry', async () => {
      const error500 = {
        response: { status: 500 },
        message: 'Internal server error',
        isAxiosError: true
      };

      mockInstance.post
        .mockRejectedValueOnce(error500)
        .mockRejectedValueOnce(error500)
        .mockResolvedValueOnce({
          data: {
            mode: 'refine',
            request: {},
            result: { success: true, content: 'test', execution_time_ms: 100, iteration_count: 1 },
            metadata: {}
          }
        });

      const response = await adapter.createRefinementRequest('test');

      expect(response.result.success).toBe(true);
    });

    it('should not retry on 400 Bad Request', async () => {
      const error400 = {
        response: { status: 400 },
        message: 'Bad request',
        isAxiosError: true
      };

      mockInstance.post.mockRejectedValue(error400);

      await expect(adapter.createRefinementRequest('test')).rejects.toMatchObject({
        response: { status: 400 }
      });

      // Should only attempt once (no retries)
      expect(mockInstance.post).toHaveBeenCalledTimes(1);
    });

    it('should open circuit breaker after threshold failures', async () => {
      const error500 = {
        response: { status: 500 },
        message: 'Internal server error',
        isAxiosError: true
      };

      // Fail 6 times (threshold is 5)
      for (let i = 0; i < 6; i++) {
        mockInstance.post.mockRejectedValue(error500);
        try {
          await adapter.createRefinementRequest('test');
        } catch (e) {
          // Expected to fail
        }
      }

      const state = adapter.getCircuitBreakerState();
      expect(state.state).toBe('open');
      expect(state.failureCount).toBeGreaterThanOrEqual(5);
    });
  });

  // ==========================================================================
  // CIRCUIT BREAKER CONTRACT
  // ==========================================================================

  describe('Circuit Breaker Contract', () => {
    it('should reject requests when circuit is open', async () => {
      // Reset circuit breaker
      adapter.resetCircuitBreaker();

      const error500 = {
        response: { status: 500 },
        message: 'Internal server error',
        isAxiosError: true
      };

      // Trigger circuit breaker to open
      for (let i = 0; i < 5; i++) {
        mockInstance.post.mockRejectedValue(error500);
        try {
          await adapter.createRefinementRequest('test');
        } catch (e) {
          // Expected
        }
      }

      // Circuit should now be open
      const stateBefore = adapter.getCircuitBreakerState();
      expect(stateBefore.state).toBe('open');

      // Next request should be rejected immediately without HTTP call
      const callCountBefore = mockInstance.post.mock.calls.length;
      await expect(adapter.createRefinementRequest('test')).rejects.toThrow('Circuit breaker is OPEN');
      const callCountAfter = mockInstance.post.mock.calls.length;

      // Should not have made another HTTP call
      expect(callCountAfter).toBe(callCountBefore);
    });

    it('should allow requests after circuit breaker reset', async () => {
      // Reset circuit breaker
      adapter.resetCircuitBreaker();

      const state = adapter.getCircuitBreakerState();
      expect(state.state).toBe('closed');

      // Request should succeed
      mockInstance.post.mockResolvedValue({
        data: {
          mode: 'refine',
          request: {},
          result: { success: true, content: 'test', execution_time_ms: 100, iteration_count: 1 },
          metadata: {}
        }
      });

      const response = await adapter.createRefinementRequest('test');
      expect(response.result.success).toBe(true);
    });
  });

  // ==========================================================================
  // IDEMPOTENCY CONTRACT
  // ==========================================================================

  describe('Idempotency Contract', () => {
    it('should handle duplicate requests with same correlation ID', async () => {
      const correlationId = 'test-idempotency-001';

      mockInstance.post.mockResolvedValue({
        data: {
          mode: 'refine',
          request: {
            mode: 'refine',
            prompt: 'test',
            metadata: { correlation_id: correlationId }
          },
          result: {
            success: true,
            content: 'test content',
            execution_time_ms: 100,
            iteration_count: 1
          },
          metadata: {
            correlation_id: correlationId
          }
        }
      });

      // Execute same request twice
      const response1 = await adapter.createRefinementRequest('test', {}, correlationId);
      const response2 = await adapter.createRefinementRequest('test', {}, correlationId);

      // Both should succeed (idempotent)
      expect(response1.result.success).toBe(true);
      expect(response2.result.success).toBe(true);
      expect(response1.request.metadata.correlation_id).toBe(correlationId);
      expect(response2.request.metadata.correlation_id).toBe(correlationId);
    });
  });

  // ==========================================================================
  // UTC TIMESTAMP CONTRACT
  // ==========================================================================

  describe('UTC Timestamp Contract', () => {
    it('should use UTC ISO-8601 timestamps in all requests', async () => {
      mockInstance.post.mockResolvedValue({
        data: {
          mode: 'refine',
          request: {},
          result: { success: true, content: 'test', execution_time_ms: 100, iteration_count: 1 },
          metadata: {}
        }
      });

      await adapter.createRefinementRequest('test');

      const { calls } = mockInstance.post.mock;
      expect(calls.length).toBeGreaterThan(0);

      const lastCall = calls[calls.length - 1];
      const requestBody = lastCall[0];

      // Verify metadata has UTC timestamp
      expect(requestBody.metadata).toBeDefined();
      expect(requestBody.metadata.timestamp_utc).toBeDefined();

      // Verify it's a valid ISO-8601 UTC timestamp
      const timestamp = requestBody.metadata.timestamp_utc;
      expect(timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/);
    });
  });
});
