/**
 * BubbleLab API Client
 *
 * Purpose: Direct API client for BubbleLab endpoints
 * Compliance: Law of Runtime Truth - wraps actual API calls
 *
 * Features:
 * - Timeout enforcement (Law of Configuration Explicitness)
 * - Retry logic for transient failures
 * - Structured error responses
 * - Idempotent operations where possible
 */

import { logger } from '../../../lib/logger';
import { retryWithBackoff } from '../../../lib/retry';

// =============================================================================
// Configuration
// =============================================================================

export interface BubbleLabClientConfig {
  api_url: string;
  timeout_ms: number;
  auth_token?: string;
  max_retries?: number;
}

// =============================================================================
// Type Definitions
// =============================================================================

export interface BubbleLabResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  status?: number;
}

export interface BubbleFlowListResponse {
  flows: Array<{
    id: string | number;
    name: string;
    description?: string;
    eventType: string;
    webhookActive: boolean;
    createdAt?: string;
    updatedAt?: string;
  }>;
}

export interface BubbleFlowCreateRequest {
  name: string;
  description?: string;
  code: string;
  eventType: string;
  webhookActive?: boolean;
}

export interface BubbleFlowCreateResponse {
  id: string | number;
  name: string;
  requiredCredentials?: Record<string, string[]>;
  webhookUrl?: string;
  createdAt?: string;
}

export interface BubbleFlowExecuteRequest {
  payload?: any;
  credentials?: Record<string, number>;
}

export interface BubbleFlowExecuteResponse {
  execution_id?: string;
  output?: any;
  error?: string;
  status?: string;
}

// =============================================================================
// BubbleLab API Client
// =============================================================================

export class BubbleLabClient {
  private readonly config: BubbleLabClientConfig;

  constructor(config: BubbleLabClientConfig) {
    this.config = {
      max_retries: 3,
      ...config,
    };

    // Validate required configuration (Law of Configuration Explicitness)
    if (!this.config.api_url) {
      throw new Error('BUBBLELAB_API_URL is required');
    }

    if (!this.config.timeout_ms || this.config.timeout_ms <= 0) {
      throw new Error('TIMEOUT_MS must be a positive number');
    }

    logger.info('BubbleLabClient initialized', {
      api_url: this.config.api_url,
      timeout_ms: this.config.timeout_ms,
      has_auth: !!this.config.auth_token,
    });
  }

  // ==========================================================================
  // Health Check
  // ==========================================================================

  /**
   * Check BubbleLab API health
   */
  async healthCheck(): Promise<BubbleLabResponse<{ status: string; version?: string }>> {
    try {
      const response = await this.makeRequest<{ status: string; version?: string }>(
        '/health',
        'GET'
      );

      return { success: true, data: response };
    } catch (error) {
      return this.handleError('Health check failed', error);
    }
  }

  // ==========================================================================
  // BubbleFlow Operations
  // ==========================================================================

  /**
   * List all BubbleFlows
   * Idempotent: GET operation, safe to retry
   */
  async listBubbleFlows(): Promise<BubbleLabResponse<BubbleFlowListResponse>> {
    try {
      const flows = await this.makeRequest<any[]>('/bubble-flow', 'GET');

      return {
        success: true,
        data: { flows: flows || [] },
      };
    } catch (error) {
      return this.handleError('Failed to list BubbleFlows', error);
    }
  }

  /**
   * Get a specific BubbleFlow by ID
   * Idempotent: GET operation, safe to retry
   */
  async getBubbleFlow(flowId: string): Promise<BubbleLabResponse<any>> {
    try {
      const flow = await this.makeRequest<any>(`/bubble-flow/${flowId}`, 'GET');

      return { success: true, data: flow };
    } catch (error) {
      return this.handleError(`Failed to get BubbleFlow ${flowId}`, error);
    }
  }

  /**
   * Create a new BubbleFlow
   * NOT idempotent: will create multiple flows on retry
   * Caller should implement deduplication logic
   */
  async createBubbleFlow(
    request: BubbleFlowCreateRequest
  ): Promise<BubbleLabResponse<BubbleFlowCreateResponse>> {
    try {
      logger.info('Creating BubbleFlow', {
        name: request.name,
        event_type: request.eventType,
      });

      const response = await this.makeRequest<BubbleFlowCreateResponse>(
        '/bubble-flow',
        'POST',
        request
      );

      logger.info('BubbleFlow created successfully', {
        flow_id: response.id,
        flow_name: response.name,
      });

      return { success: true, data: response };
    } catch (error) {
      return this.handleError('Failed to create BubbleFlow', error);
    }
  }

  /**
   * Update a BubbleFlow
   * NOT idempotent: multiple updates with same data are OK but not guaranteed
   */
  async updateBubbleFlow(
    flowId: string,
    updates: Partial<BubbleFlowCreateRequest>
  ): Promise<BubbleLabResponse<any>> {
    try {
      logger.info('Updating BubbleFlow', { flow_id: flowId });

      const response = await this.makeRequest<any>(
        `/bubble-flow/${flowId}`,
        'PUT',
        updates
      );

      logger.info('BubbleFlow updated successfully', { flow_id: flowId });

      return { success: true, data: response };
    } catch (error) {
      return this.handleError(`Failed to update BubbleFlow ${flowId}`, error);
    }
  }

  /**
   * Delete a BubbleFlow
   * Idempotent with check: verify flow doesn't exist after deletion
   */
  async deleteBubbleFlow(flowId: string): Promise<BubbleLabResponse<void>> {
    try {
      logger.info('Deleting BubbleFlow', { flow_id: flowId });

      await this.makeRequest<void>(`/bubble-flow/${flowId}`, 'DELETE');

      // Verify deletion (idempotency check)
      try {
        await this.getBubbleFlow(flowId);
        logger.warn('BubbleFlow still exists after deletion', { flow_id: flowId });
      } catch {
        // Expected - flow should not exist
        logger.info('BubbleFlow deletion verified', { flow_id: flowId });
      }

      return { success: true };
    } catch (error) {
      return this.handleError(`Failed to delete BubbleFlow ${flowId}`, error);
    }
  }

  // ==========================================================================
  // Execution Operations
  // ==========================================================================

  /**
   * Execute a BubbleFlow
   * NOT idempotent: each execution creates a new run
   */
  async executeBubbleFlow(
    flowId: string,
    request: BubbleFlowExecuteRequest
  ): Promise<BubbleLabResponse<BubbleFlowExecuteResponse>> {
    try {
      logger.info('Executing BubbleFlow', {
        flow_id: flowId,
        has_payload: !!request.payload,
        has_credentials: !!request.credentials,
      });

      const response = await this.makeRequest<BubbleFlowExecuteResponse>(
        `/bubble-flow/${flowId}/execute`,
        'POST',
        request,
        this.config.timeout_ms  // Use configured timeout
      );

      logger.info('BubbleFlow execution completed', {
        flow_id: flowId,
        execution_id: response.execution_id,
        status: response.status,
      });

      return { success: true, data: response };
    } catch (error) {
      return this.handleError(`Failed to execute BubbleFlow ${flowId}`, error);
    }
  }

  /**
   * Get execution history for a BubbleFlow
   * Idempotent: GET operation, safe to retry
   */
  async getExecutionHistory(
    flowId: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<BubbleLabResponse<any>> {
    try {
      const queryParams = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
      });

      const history = await this.makeRequest<any>(
        `/bubble-flow/${flowId}/executions?${queryParams}`,
        'GET'
      );

      return { success: true, data: history };
    } catch (error) {
      return this.handleError(`Failed to get execution history for ${flowId}`, error);
    }
  }

  // ==========================================================================
  // Validation Operations
  // ==========================================================================

  /**
   * Validate BubbleFlow code without creating
   * Idempotent: Validation operation, no side effects
   */
  async validateCode(code: string): Promise<BubbleLabResponse<any>> {
    try {
      const result = await this.makeRequest<any>(
        '/bubble-flow/validate',
        'POST',
        { code }
      );

      return { success: true, data: result };
    } catch (error) {
      return this.handleError('Code validation failed', error);
    }
  }

  // ==========================================================================
  // Private Helper Methods
  // ==========================================================================

  /**
   * Make HTTP request to BubbleLab API with retry logic
   */
  private async makeRequest<T>(
    endpoint: string,
    method: string = 'GET',
    body?: any,
    timeout_ms?: number
  ): Promise<T> {
    const url = `${this.config.api_url}${endpoint}`;
    const timeout = timeout_ms || this.config.timeout_ms;

    // Wrap the actual request in retry logic
    return retryWithBackoff(
      async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        try {
          const headers: Record<string, string> = {
            'Content-Type': 'application/json',
          };

          if (this.config.auth_token) {
            headers['Authorization'] = `Bearer ${this.config.auth_token}`;
          }

          const response = await fetch(url, {
            method,
            headers,
            body: body ? JSON.stringify(body) : undefined,
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            const errorText = await response.text();
            throw new Error(
              `HTTP ${response.status}: ${errorText || response.statusText}`
            );
          }

          // Handle empty responses (e.g., DELETE)
          const contentType = response.headers.get('content-type');
          if (!contentType || !contentType.includes('application/json')) {
            return undefined as T;
          }

          return await response.json();
        } catch (error) {
          clearTimeout(timeoutId);

          // Re-throw timeout errors
          if (error instanceof Error && error.name === 'AbortError') {
            throw new Error(`Request timeout after ${timeout}ms`);
          }

          throw error;
        }
      },
      {
        max_retries: this.config.max_retries || 3,
        base_delay_ms: 1000,
        max_delay_ms: 10000,
        jitter_ms: 500,
      }
    );
  }

  /**
   * Handle errors and return structured error response
   */
  private handleError(message: string, error: unknown): BubbleLabResponse {
    const errorMessage = error instanceof Error ? error.message : String(error);

    logger.error(message, error instanceof Error ? error : undefined, {
      error_message: errorMessage,
    });

    return {
      success: false,
      error: `${message}: ${errorMessage}`,
    };
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a BubbleLab client from environment variables
 */
export function createBubbleLabClient(): BubbleLabClient {
  const api_url = process.env.BUBBLELAB_API_URL;
  const timeout_ms = process.env.TIMEOUT_MS
    ? parseInt(process.env.TIMEOUT_MS, 10)
    : undefined;
  const auth_token = process.env.BUBBLELAB_AUTH_TOKEN;

  if (!api_url) {
    throw new Error('BUBBLELAB_API_URL environment variable is required');
  }

  if (!timeout_ms) {
    throw new Error('TIMEOUT_MS environment variable is required');
  }

  return new BubbleLabClient({
    api_url,
    timeout_ms,
    auth_token,
  });
}
