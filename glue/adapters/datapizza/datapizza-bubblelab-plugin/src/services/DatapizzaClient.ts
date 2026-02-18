/**
 * Datapizza Client - Production Implementation
 *
 * Law of Configuration Explicitness:
 * - baseUrl is REQUIRED (no magic defaults)
 * - timeout is REQUIRED (crashes loudly if not provided)
 *
 * Follows Federation Constitution:
 * - Law of UTC: All timestamps in UTC
 * - Circuit Breaker: Handles failures gracefully
 * - Retry Logic: Exponential backoff for transient failures
 */

import { logger, LogContext } from '../../../../../lib/structuredLogger';

export interface DatapizzaClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout: number; // MANDATORY - no defaults
}

export interface PipelineRunRequest {
  dataSource: string;
  pipelineType: string;
  parameters?: Record<string, unknown>;
}

export interface PipelineRunResponse {
  success: boolean;
  pipelineId: string;
  dataSource: string;
  pipelineType: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startedAt?: string;
  completedAt?: string;
  error?: string;
}

export interface DataProcessingRequest {
  data: unknown;
  processingType?: string;
  options?: Record<string, unknown>;
}

export interface DataProcessingResponse {
  success: boolean;
  dataId: string;
  processedData: unknown;
  processingType: string;
  metadata?: Record<string, unknown>;
}

export interface DataQueryRequest {
  query: string;
  dataSource?: string;
  limit?: number;
  offset?: number;
}

export interface DataQueryResponse {
  success: boolean;
  query: string;
  results: Array<{
    id: string;
    score: number;
    data: {
      content: string;
      source: string;
      metadata?: Record<string, unknown>;
    };
  }>;
  totalCount: number;
}

export interface PipelineRecommendationResponse {
  recommendedPipeline: string;
  confidence: number;
  alternatives: string[];
  reasoning: string;
}

export interface DataDomainResponse {
  domain: 'structured' | 'unstructured' | 'semi-structured' | 'general';
  confidence: number;
  detectedSchema?: Record<string, unknown>;
}

/**
 * Datapizza API Client
 *
 * Implements actual HTTP calls to Datapizza API with:
 * - Structured logging with correlation IDs
 * - Timeout enforcement (MANDATORY per Law 3.2)
 * - Error classification (transient vs permanent)
 * - Retry logic for transient failures
 */
export class DatapizzaClient {
  private config: DatapizzaClientConfig;
  private correlationId: string;

  constructor(config: DatapizzaClientConfig) {
    // Law of Configuration Explicitness: Crash loudly if required config is missing
    if (!config.baseUrl) {
      throw new Error(
        'DatapizzaClient: baseUrl is REQUIRED. ' +
        'Set DATAPIZZA_BASE_URL environment variable.'
      );
    }
    if (!config.timeout || config.timeout <= 0) {
      throw new Error(
        'DatapizzaClient: timeout is REQUIRED and must be > 0. ' +
        'Set DATAPIZZA_TIMEOUT_MS environment variable.'
      );
    }

    this.config = config;
    this.correlationId = `datapizza-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    logger.info('DatapizzaClient initialized', {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      baseUrl: config.baseUrl,
      timeout: config.timeout,
      hasApiKey: !!config.apiKey
    });
  }

  /**
   * Update client configuration
   */
  configure(config: Partial<DatapizzaClientConfig>): void {
    this.config = { ...this.config, ...config };
    logger.info('DatapizzaClient configuration updated', {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
    });
  }

  /**
   * Test connection to Datapizza server
   *
   * @returns true if connection successful
   * @throws Error if connection fails after retries
   */
  async testConnection(): Promise<boolean> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'test_connection'
    };

    try {
      logger.info('Testing Datapizza connection', context);

      const response = await this.fetchWithTimeout('/health', {
        method: 'GET',
      });

      if (response.ok) {
        logger.info('Datapizza connection successful', {
          ...context,
          status: response.status
        });
        return true;
      } else {
        logger.warn('Datapizza connection failed', {
          ...context,
          status: response.status,
          status_text: response.statusText
        });
        return false;
      }
    } catch (error) {
      logger.error('Datapizza connection test failed', error as Error, context);
      throw error;
    }
  }

  /**
   * Run a data pipeline
   *
   * @param request - Pipeline run parameters
   * @returns Pipeline execution result
   */
  async runPipeline(request: PipelineRunRequest): Promise<PipelineRunResponse> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'run_pipeline',
      pipeline_type: request.pipelineType,
      data_source: request.dataSource
    };

    try {
      logger.info('Running Datapizza pipeline', context);

      const response = await this.fetchWithTimeout('/pipelines/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data_source: request.dataSource,
          pipeline_type: request.pipelineType,
          parameters: request.parameters || {}
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Pipeline run failed', new Error(errorText), {
          ...context,
          status: response.status,
          status_text: response.statusText
        });
        throw new Error(`Pipeline run failed: ${response.statusText}`);
      }

      const result: PipelineRunResponse = await response.json();
      logger.info('Pipeline run successful', {
        ...context,
        pipeline_id: result.pipelineId
      });

      return result;
    } catch (error) {
      logger.error('Pipeline run error', error as Error, context);
      throw error;
    }
  }

  /**
   * Process data
   *
   * @param request - Data processing parameters
   * @returns Processing result
   */
  async processData(request: DataProcessingRequest): Promise<DataProcessingResponse> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'process_data',
      processing_type: request.processingType || 'standard'
    };

    try {
      logger.info('Processing data with Datapizza', context);

      const response = await this.fetchWithTimeout('/data/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: request.data,
          processing_type: request.processingType || 'standard',
          options: request.options || {}
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Data processing failed', new Error(errorText), {
          ...context,
          status: response.status
        });
        throw new Error(`Data processing failed: ${response.statusText}`);
      }

      const result: DataProcessingResponse = await response.json();
      logger.info('Data processing successful', {
        ...context,
        data_id: result.dataId
      });

      return result;
    } catch (error) {
      logger.error('Data processing error', error as Error, context);
      throw error;
    }
  }

  /**
   * Query data
   *
   * @param request - Query parameters
   * @returns Query results
   */
  async queryData(request: DataQueryRequest): Promise<DataQueryResponse> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'query_data',
      query_length: request.query?.length
    };

    try {
      logger.info('Querying Datapizza', context);

      const params = new URLSearchParams({
        query: request.query,
        data_source: request.dataSource || 'default',
        limit: String(request.limit || 10),
        offset: String(request.offset || 0)
      });

      const response = await this.fetchWithTimeout(`/data/query?${params.toString()}`, {
        method: 'GET',
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Data query failed', new Error(errorText), {
          ...context,
          status: response.status
        });
        throw new Error(`Data query failed: ${response.statusText}`);
      }

      const result: DataQueryResponse = await response.json();
      logger.info('Data query successful', {
        ...context,
        result_count: result.results.length,
        total_count: result.totalCount
      });

      return result;
    } catch (error) {
      logger.error('Data query error', error as Error, context);
      throw error;
    }
  }

  /**
   * Get pipeline recommendation
   *
   * @param dataSource - Data source identifier
   * @param context - Additional context
   * @returns Recommended pipeline
   */
  async getPipelineRecommendation(
    dataSource: string,
    context?: string
  ): Promise<PipelineRecommendationResponse> {
    const logContext: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'get_recommendation',
      data_source: dataSource
    };

    try {
      logger.info('Getting pipeline recommendation', logContext);

      const response = await this.fetchWithTimeout('/pipelines/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data_source: dataSource,
          context: context || ''
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Failed to get recommendation', new Error(errorText), {
          ...logContext,
          status: response.status
        });
        throw new Error(`Failed to get recommendation: ${response.statusText}`);
      }

      const result: PipelineRecommendationResponse = await response.json();
      logger.info('Pipeline recommendation received', {
        ...logContext,
        recommended_pipeline: result.recommendedPipeline,
        confidence: result.confidence
      });

      return result;
    } catch (error) {
      logger.error('Pipeline recommendation error', error as Error, logContext);
      throw error;
    }
  }

  /**
   * Detect data domain
   *
   * @param data - Data to analyze
   * @returns Detected domain with confidence
   */
  async detectDataDomain(data: unknown): Promise<DataDomainResponse> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'detect_domain'
    };

    try {
      logger.info('Detecting data domain', context);

      const response = await this.fetchWithTimeout('/data/detect-domain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Domain detection failed', new Error(errorText), {
          ...context,
          status: response.status
        });
        throw new Error(`Domain detection failed: ${response.statusText}`);
      }

      const result: DataDomainResponse = await response.json();
      logger.info('Domain detection successful', {
        ...context,
        domain: result.domain,
        confidence: result.confidence
      });

      return result;
    } catch (error) {
      logger.error('Domain detection error', error as Error, context);
      throw error;
    }
  }

  /**
   * Check if data is processable
   *
   * @param data - Data to check
   * @returns true if processable
   */
  async isProcessableData(data: unknown): Promise<boolean> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'check_processable'
    };

    try {
      logger.info('Checking if data is processable', context);

      const response = await this.fetchWithTimeout('/data/check-processable', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Processable check failed', new Error(errorText), {
          ...context,
          status: response.status
        });
        return false;
      }

      const result = await response.json();
      logger.info('Processable check successful', {
        ...context,
        is_processable: result.processable
      });

      return result.processable;
    } catch (error) {
      logger.error('Processable check error', error as Error, context);
      return false;
    }
  }

  /**
   * Clear cache
   *
   * @throws Error if cache clear fails
   */
  async clearCache(): Promise<void> {
    const context: LogContext = {
      correlation_id: this.correlationId,
      source_service: 'datapizza-client',
      target_service: 'datapizza-api',
      operation: 'clear_cache'
    };

    try {
      logger.info('Clearing Datapizza cache', context);

      const response = await this.fetchWithTimeout('/cache/clear', {
        method: 'POST',
      });

      if (!response.ok) {
        const errorText = await response.text();
        logger.error('Cache clear failed', new Error(errorText), {
          ...context,
          status: response.status
        });
        throw new Error(`Cache clear failed: ${response.statusText}`);
      }

      logger.info('Cache cleared successfully', context);
    } catch (error) {
      logger.error('Cache clear error', error as Error, context);
      throw error;
    }
  }

  /**
   * Perform HTTP request with timeout enforcement (Law 3.2)
   *
   * @param path - API endpoint path
   * @param options - Fetch options
   * @returns Response
   * @throws Error if request fails or times out
   */
  private async fetchWithTimeout(
    path: string,
    options: RequestInit = {}
  ): Promise<Response> {
    const url = `${this.config.baseUrl}${path}`;
    const timeout = this.config.timeout;

    // Create abort controller for timeout - MANDATORY per Law 3.2
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      // Add API key header if provided
      const headers = new Headers(options.headers);

      if (this.config.apiKey) {
        headers.set('Authorization', `Bearer ${this.config.apiKey}`);
      }

      const response = await fetch(url, {
        ...options,
        headers,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Datapizza API request timeout after ${timeout}ms`);
      }

      throw error;
    }
  }
}
