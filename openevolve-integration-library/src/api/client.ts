/**
 * Unified API Client for OpenEvolve Integration Library
 *
 * This is the main client that provides unified access to all OpenEvolve integrations.
 * It handles HTTP communication, WebSocket connections, error handling, and provides
 * a consistent interface for working with all integrations.
 */

import { v4 as uuidv4 } from 'uuid';
import {
  BackendClient,
  WebSocketHandlers,
} from './backend';
import {
  ClientConfig,
  ExecutionOptions,
  ProgressUpdate,
  BatchRequest,
  BatchResult,
  HealthStatus,
  IntegrationAdapter,
  ConnectionState,
  RequestMetrics,
  RetryConfig,
  CircuitBreakerConfig,
  WebSocketMessage,
  Middleware,
} from './types';

import {
  IntegrationError,
  ConnectionError,
  ValidationError,
  createIntegrationError,
} from './errors';

import {
  LeanAideIntegration,
  EvolutionIntegration,
  KnowledgeIntegration,
  MakerIntegration,
  CrewAIIntegration,
  DecompositionIntegration,
  VerificationIntegration,
  AssemblyIntegration,
  SolutionIntegration,
} from '../integrations';

/**
 * Integration registry - typed access to all integrations
 */
export interface IntegrationRegistry {
  leanaide: LeanAideIntegration;
  evolution: EvolutionIntegration;
  knowledge: KnowledgeIntegration;
  maker: MakerIntegration;
  crewai: CrewAIIntegration;
  decomposition: DecompositionIntegration;
  verification: VerificationIntegration;
  assembly: AssemblyIntegration;
  solution: SolutionIntegration;
}

/**
 * Integration enum
 */
export enum IntegrationName {
  LEANAIDE = 'leanaide',
  EVOLUTION = 'evolution',
  KNOWLEDGE = 'knowledge',
  MAKER = 'maker',
  CREWAI = 'crewai',
  DECOMPOSITION = 'decomposition',
  VERIFICATION = 'verification',
  ASSEMBLY = 'assembly',
  SOLUTION = 'solution',
}

/**
 * Default retry configuration
 */
const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxAttempts: 3,
  initialDelay: 1000,
  maxDelay: 10000,
  backoffMultiplier: 2,
  retryOn4xx: false,
  retryOn5xx: true,
  retryableStatusCodes: [408, 429, 500, 502, 503, 504],
};

const MAX_METRICS_SIZE = 1000;

/**
 * Unified client for all OpenEvolve integrations
 *
 * @example
 * ```typescript
 * const client = new OpenEvolveClient({
 *   baseUrl: 'http://localhost:8000',
 *   debug: true
 * });
 *
 * // Execute an integration
 * const result = await client.execute(
 *   IntegrationName.LEANAIDE,
 *   { problem: 'Solve x^2 + 2x + 1 = 0' },
 *   { onProgress: (update) => console.log(update.progress) }
 * );
 *
 * // Use integration-specific methods
 * const proof = await client.integrations.leanaide.execute({
 *   type: 'theorem_proving',
 *   statement: '...'
 * });
 * ```
 */
export class OpenEvolveClient {
  public static readonly VERSION = '1.1.0';
  private backend: BackendClient;
  private config: ClientConfig;
  private integrationAdapters: Map<string, IntegrationAdapter>;
  private connectionState: ConnectionState = 'disconnected';
  private executionMetrics: Map<string, RequestMetrics> = new Map();
  private progressCallbacks: Map<string, (update: ProgressUpdate) => void> = new Map();
  private errorHandlers: Set<(error: IntegrationError) => void> = new Set();
  private retryConfig: RetryConfig;
  private circuitBreakerConfig: Partial<CircuitBreakerConfig>;
  private middleware: Middleware[] = [];
  private debug: boolean;
  private healthCheckTimer: any = null;


  /**
   * Create a new OpenEvolve client
   *
   * @param config - Client configuration
   */
  constructor(config: ClientConfig) {
    this.integrationAdapters = new Map();
    this.executionMetrics = new Map();
    this.progressCallbacks = new Map();
    this.errorHandlers = new Set();
    
    if (config.onError) {
      this.errorHandlers.add(config.onError);
    }

    this.middleware = config.middleware || [];

    this.retryConfig = DEFAULT_RETRY_CONFIG;
    this.circuitBreakerConfig = config.circuitBreakerConfig || {};
    this.debug = config.debug || false;
    this.config = config;


    try {
      // Initialize backend client
      this.backend = new BackendClient({
        baseUrl: config.baseUrl,
        timeout: config.timeout || 30000,
        apiKey: config.apiKey,
        debug: this.debug,
        headers: config.headers,
        requestTransform: config.requestTransform,
        responseTransform: config.responseTransform,
      });

      // Load integration adapters
      this.loadIntegrations();

      // Setup WebSocket if enabled
      if (config.enableWebSocket !== false) {
        try {
          this.setupWebSocket();
        } catch (wsError) {
          this.log('Failed to setup WebSocket, client will continue in HTTP-only mode', wsError);
        }
      }

      this.log('OpenEvolve client initialized');

      // Start background health check if interval is provided
      if (config.healthCheckInterval && config.healthCheckInterval > 0) {
        this.startHealthCheck(config.healthCheckInterval);
      }
    } catch (error) {

      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`[OpenEvolveClient] Critical failure during initialization: ${errorMessage}`);
      
      throw createIntegrationError('client_init', error);
    }
  }



  /**
   * Load integration adapters
   */
  private loadIntegrations(): void {
    const retryConfig = this.config.retryConfig || DEFAULT_RETRY_CONFIG;
    const cbConfig = this.circuitBreakerConfig;

    const integrations = [
      { name: IntegrationName.LEANAIDE, adapter: new LeanAideIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.EVOLUTION, adapter: new EvolutionIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.KNOWLEDGE, adapter: new KnowledgeIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.MAKER, adapter: new MakerIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.CREWAI, adapter: new CrewAIIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.DECOMPOSITION, adapter: new DecompositionIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.VERIFICATION, adapter: new VerificationIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.ASSEMBLY, adapter: new AssemblyIntegration(this.backend, retryConfig, cbConfig) },
      { name: IntegrationName.SOLUTION, adapter: new SolutionIntegration(this.backend, retryConfig, cbConfig) },
    ];

    integrations.forEach(({ name, adapter }) => {
      // Set global error handler for each adapter to report back to this client
      adapter.setGlobalErrorHandler((error) => {



        // We use a dummy executionId if we don't have one (though most errors from adapters will have been caught by client.execute)
        // This is for direct calls like client.integrations.leanaide.execute()
        this.handleExecutionError('direct-call', error, name);
      });
      this.integrationAdapters.set(name, adapter);
    });

    this.log('Integration adapters loaded');
  }



  /**
   * Setup WebSocket connection
   */
  private setupWebSocket(): void {
    const handlers: WebSocketHandlers = {
      onConnect: () => {
        this.connectionState = 'connected';
        this.log('WebSocket connected');
      },
      onDisconnect: (reason) => {
        this.connectionState = 'disconnected';
        this.log('WebSocket disconnected:', { reason });
      },
      onError: (error) => {
        this.log('WebSocket error:', error);
      },
      onMessage: (message: WebSocketMessage) => {
        this.handleWebSocketMessage(message);
      },
      onReconnect: (attemptNumber) => {
        this.connectionState = 'reconnecting';
        this.log('WebSocket reconnecting:', { attemptNumber });
      },
    };

    this.backend.websocket('/ws', handlers);
    this.connectionState = 'connecting';
  }

  /**
   * Start periodic health checks
   */
  public startHealthCheck(interval: number): void {
    this.stopHealthCheck();
    this.healthCheckTimer = setInterval(async () => {
      try {
        const health = await this.healthCheck();
        if (health.status === 'unhealthy') {
          this.handleExecutionError('background-health', new Error('Backend reported unhealthy status'), 'backend');
        }
      } catch (error) {
        this.log('Background health check failed', error);
      }
    }, interval);
    
    // Ensure timer doesn't keep the process alive in Node.js
    if (this.healthCheckTimer.unref) {
      this.healthCheckTimer.unref();
    }
  }

  /**
   * Stop periodic health checks
   */
  public stopHealthCheck(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }
  }

  /**
   * Add a global error handler

   */
  addErrorHandler(handler: (error: IntegrationError) => void): void {
    this.errorHandlers.add(handler);
  }

  /**
   * Remove a global error handler
   */
  removeErrorHandler(handler: (error: IntegrationError) => void): void {
    this.errorHandlers.delete(handler);
  }

  /**
   * Handle WebSocket message
   */

  private handleWebSocketMessage(message: WebSocketMessage): void {
    try {
      this.log('WebSocket message received:', message);

      // Handle different message types
      switch (message.type) {
        case 'progress':
          this.handleProgressUpdate(message.data as ProgressUpdate);
          break;
        case 'complete':
          this.handleExecutionComplete(message.executionId!, message.data);
          break;
        case 'error':
          this.handleExecutionError(message.executionId!, message.data, message.integration);
          break;

        case 'status':
          // Handle status updates
          break;
      }
    } catch (error) {
      this.log('Error handling WebSocket message:', error);
    }
  }

  /**
   * Handle progress update
   */
  private handleProgressUpdate(update: ProgressUpdate): void {
    try {
      this.log('Progress update:', update);
      const callback = this.progressCallbacks.get(update.executionId);
      if (callback) {
        callback(update);
      }
    } catch (error) {
      this.log('Error in progress callback:', error);
    }
  }

  /**
   * Handle execution complete
   */
  private handleExecutionComplete(executionId: string, result: any): void {
    try {
      this.log('Execution complete:', { executionId, result });

      const metrics = this.executionMetrics.get(executionId);
      if (metrics) {
        metrics.endTime = new Date().toISOString();
        metrics.duration = Date.now() - new Date(metrics.startTime).getTime();
        metrics.success = true;
        this.executionMetrics.set(executionId, metrics);
      }
    } catch (error) {
      this.log('Error handling execution complete:', error);
    }
  }

  /**
   * Handle successful execution
   */
  private handleExecutionSuccess(executionId: string): void {
    try {
      this.log('Execution success:', { executionId });

      const metrics = this.executionMetrics.get(executionId);
      if (metrics) {
        metrics.endTime = new Date().toISOString();
        metrics.duration = Date.now() - new Date(metrics.startTime).getTime();
        metrics.success = true;
        this.executionMetrics.set(executionId, metrics);
      }
    } catch (error) {
      this.log('Error handling execution success:', error);
    }
  }

        /**

         * Handle execution error

         */

        private handleExecutionError(executionId: string, error: any, integration?: string): void {

          try {

            const integrationError = createIntegrationError(integration || 'unknown', error);

      

            // Track metrics only once per execution

            if (!error || !(error as any)._isMetricsTracked) {

              const metrics = this.executionMetrics.get(executionId);

              if (metrics) {

                metrics.endTime = new Date().toISOString();

                metrics.duration = Date.now() - new Date(metrics.startTime).getTime();

                metrics.success = false;

                metrics.error = integrationError.message;

                this.executionMetrics.set(executionId, metrics);

              }

              if (error && typeof error === 'object') (error as any)._isMetricsTracked = true;

            }

      

            this.log('Execution error:', { executionId, error: integrationError });

      

            // Trigger global error handlers

            this.errorHandlers.forEach(handler => {

              try {

                handler(integrationError);

              } catch (cbError) {

                this.log('Error in global error handler:', cbError);

              }

            });

          } catch (err) {

            this.log('Error handling execution error:', err);

          }

        }

      

    


  /**
   * Generic execution method with full typing
   *
   * @param integration - Integration name
   * @param inputs - Input data
   * @param options - Execution options
   * @returns Execution result
   *
   * @example
   * ```typescript
   * const result = await client.execute(
   *   IntegrationName.LEANAIDE,
   *   { problem: 'Solve x^2 + 2x + 1 = 0' },
   *   { timeout: 60000 }
   * );
   * ```
   */
  async execute<TIntegration extends string, TInputs, TResult>(
    integration: TIntegration,
    inputs: TInputs,
    options?: ExecutionOptions
  ): Promise<TResult> {
    const executionId = options?.executionId || uuidv4();
    const startTime = new Date().toISOString();

    // Track metrics
    this.executionMetrics.set(executionId, {
      requestId: executionId,
      integration: integration as string,
      startTime,
      endTime: '',
      duration: 0,
      retries: 0,
      success: false,
    });

    this.log('Executing integration:', { integration, executionId, inputs });
    this.clearOldMetrics();

    try {
      // Run through middleware pipeline
      const result = await this.runMiddleware<TResult>(
        { integration: integration as string, inputs, options, executionId },
        async () => {
          // Validate inputs
          await this.validateInputs(integration as string, inputs);

          // Get adapter
          const adapter = this.getIntegration(integration as string);

          // Pass onRetry to adapter via options if provided
          const executionOptions = { ...options, executionId };

          // Execute with options
          const execResult = await adapter.execute<TInputs, TResult>(
            inputs,
            executionOptions
          );

          // Call completion callback
          if (options?.onComplete) {
            try {
              options.onComplete(execResult);
            } catch (cbError) {
              this.log('Error in onComplete callback:', cbError);
            }
          }

          return execResult;
        }
      );

      this.handleExecutionSuccess(executionId);
      return result;
    } catch (error) {
      const integrationError = createIntegrationError(integration as string, error);
      this.handleExecutionError(executionId, integrationError, integration as string);

      // Call error callback
      if (options?.onError) {
        try {
          options.onError(integrationError);
        } catch (cbError) {
          this.log('Error in onError callback:', cbError);
        }
      }

      throw integrationError;
    }
  }



    

  /**
   * Run the middleware pipeline
   */
  private async runMiddleware<T>(
    context: { integration: string; inputs: any; options?: ExecutionOptions; executionId: string },
    finalAction: () => Promise<T>
  ): Promise<T> {
    let index = -1;

    const dispatch = async (i: number): Promise<T> => {
      if (i <= index) {
        throw new Error('next() called multiple times in middleware pipeline');
      }
      index = i;

      try {
        const fn = i === this.middleware.length ? finalAction : this.middleware[i];
        
        if (!fn) {
          // If no middleware and no finalAction (shouldn't happen)
          if (i === this.middleware.length) {
             throw new Error('Final action missing in middleware pipeline');
          }
          return await dispatch(i + 1);
        }

        if (i === this.middleware.length) {
          return await (fn as () => Promise<T>)();
        }

        return await (fn as Middleware)(context, () => dispatch(i + 1));
      } catch (error) {
        // Wrap any middleware-originating error to ensure consistency
        throw createIntegrationError(context.integration, error);
      }
    };

    return await dispatch(0);
  }


  /**
   * Stream execution with progress updates
   *
   * @param integration - Integration name
   * @param inputs - Input data
   * @param onProgress - Progress callback
   * @param options - Execution options
   * @returns Execution result
   *
   * @example
   * ```typescript
   * const result = await client.executeStream(
   *   IntegrationName.EVOLUTION,
   *   { prompt: 'Evolve a solution for...' },
   *   (update) => {
   *     console.log(`Progress: ${update.progress}%`);
   *   }
   * );
   * ```
   */
  async executeStream<TInputs, TResult>(
    integration: string,
    inputs: TInputs,
    onProgress: (update: ProgressUpdate) => void,
    options?: ExecutionOptions
  ): Promise<TResult> {
    const executionId = options?.executionId || uuidv4();
    const startTime = new Date().toISOString();

    // Track metrics
    this.executionMetrics.set(executionId, {
      requestId: executionId,
      integration: integration as string,
      startTime,
      endTime: '',
      duration: 0,
      retries: 0,
      success: false,
    });

    this.log('Executing integration with streaming:', { integration, executionId });
    this.clearOldMetrics();

    try {
      // Run through middleware pipeline
      const result = await this.runMiddleware<TResult>(
        { integration, inputs, options, executionId },
        async () => {
          // Validate inputs
          await this.validateInputs(integration, inputs);

          // Get the adapter
          const adapter = this.getIntegration(integration);

          // Register progress callback for the global socket
          this.progressCallbacks.set(executionId, onProgress);

          try {
            // Execute with adapter's streaming support
            return await adapter.executeStream<TInputs, TResult>(
              inputs,
              onProgress,
              { ...options, executionId }
            );
          } finally {
            this.progressCallbacks.delete(executionId);
          }
        }
      );

      this.handleExecutionSuccess(executionId);
      return result;
    } catch (error) {
      this.handleExecutionError(executionId, error, integration);
      throw createIntegrationError(integration, error);
    }



  }

  /**
   * Batch execution
   *
   * @param requests - Array of batch requests
   * @returns Array of batch results
   *
   * @example
   * ```typescript
   * const results = await client.executeBatch([
   *   {
   *     integration: IntegrationName.LEANAIDE,
   *     id: 'req1',
   *     inputs: { problem: 'Problem 1' }
   *   },
   *   {
   *     integration: IntegrationName.MAKER,
   *     id: 'req2',
   *     inputs: { prompt: 'Create...' }
   *   }
   * ]);
   * ```
   */
  async executeBatch<TInputs, TResult>(
    requests: BatchRequest<TInputs>[]
  ): Promise<BatchResult<TResult>[]> {
    if (!Array.isArray(requests)) {
      this.log('Invalid batch request: expected array');
      return [];
    }

    this.log('Executing batch:', { count: requests.length });

    const results: BatchResult<TResult>[] = await Promise.allSettled(
      requests.map(async (request) => {
        const startTime = Date.now();
        const requestId = request.id || uuidv4();

        try {
          if (!request.integration) {
            throw new Error('Integration name is required for batch request');
          }

          const result = await this.execute<string, TInputs, TResult>(
            request.integration,
            request.inputs,
            { ...request.options, executionId: requestId }
          );

          return {
            id: requestId,
            result,
            error: null,
            executionTime: Date.now() - startTime,
            success: true,
          };
        } catch (error) {
          return {
            id: requestId,
            result: null,
            error: createIntegrationError(request.integration || 'batch', error),
            executionTime: Date.now() - startTime,
            success: false,
          };
        }
      })
    ).then((settledResults) =>
      settledResults.map((result, index) => {
        if (result.status === 'fulfilled') {
          return result.value;
        } else {
          // This case should be rare as the map handles its own errors
          const request = requests[index];
          return {
            id: request?.id || `failed-${index}`,
            result: null,
            error: createIntegrationError(request?.integration || 'batch', result.reason),
            executionTime: 0,
            success: false,
          };
        }
      })
    );

    return results;
  }


  /**
   * Health check
   *
   * @returns Health status
   */
  async healthCheck(): Promise<HealthStatus> {
    this.log('Performing health check');

    const backendStatus = await this.backend.getStatus();
    
    // Check all integrations in parallel
    const integrationNames = Array.from(this.integrationAdapters.keys());
    const healthResults = await Promise.all(
      Array.from(this.integrationAdapters.values()).map(integration => 
        integration.healthCheck().catch(error => ({
          name: integration.name,
          status: 'unavailable' as const,
          responseTime: 0,
          lastError: error instanceof Error ? error.message : String(error),
          endpoints: [],
        }))
      )
    );

    const integrationHealth: Record<string, any> = {};
    healthResults.forEach((result, index) => {
      integrationHealth[integrationNames[index]] = result;
    });

    return {
      status: backendStatus.online ? 'healthy' : 'unhealthy',
      backend: backendStatus,
      integrations: integrationHealth,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Connect to backend
   */
  async connect(): Promise<void> {
    this.log('Connecting to backend');
    this.connectionState = 'connecting';

    try {
      // Ping backend to verify connection
      const isOnline = await this.backend.ping();

      if (!isOnline) {
        throw new ConnectionError('backend', 'Backend is not responding');
      }

      // Setup WebSocket if enabled
      if (this.config.enableWebSocket !== false && !this.backend.isWebSocketConnected()) {
        this.setupWebSocket();
      }

      this.connectionState = 'connected';
      this.log('Connected successfully');
    } catch (error) {
      this.connectionState = 'disconnected';
      throw createIntegrationError('backend', error);
    }
  }

  /**
   * Disconnect from backend
   */
  async disconnect(): Promise<void> {
    this.log('Disconnecting from backend');
    this.connectionState = 'disconnecting';
    this.stopHealthCheck();

    try {

      this.backend.disconnectWebSocket();
      this.connectionState = 'disconnected';
      this.log('Disconnected successfully');
    } catch (error) {
      throw createIntegrationError('backend', error);
    }
  }

  /**
   * Get integration versions
   */
  getVersions(): Record<string, string> {
    const versions: Record<string, string> = {};
    for (const [name, adapter] of this.integrationAdapters.entries()) {
      versions[name] = adapter.getVersion();
    }
    return versions;
  }

  /**
   * Check if connected
   *
   * @returns Connection status
   */
  isConnected(): boolean {
    return this.connectionState === 'connected';
  }

  /**
   * Get connection state
   *
   * @returns Current connection state
   */
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  /**
   * Validate inputs for an integration
   */
  private async validateInputs(integration: string, inputs: any): Promise<void> {
    const adapter = this.integrationAdapters.get(integration);
    if (!adapter) {
      throw new IntegrationError(
        integration,
        'INTEGRATION_NOT_FOUND',
        `Integration '${integration}' not found in registry`
      );
    }
    
    const result = await adapter.validate(inputs);
    if (!result.valid) {
      throw new ValidationError(integration, result.errors);
    }
  }

  /**
   * Get integration adapter
   */
  private getIntegration(name: string): IntegrationAdapter {
    const integration = this.integrationAdapters.get(name);
    if (!integration) {
      throw new IntegrationError(
        name,
        'INTEGRATION_NOT_FOUND',
        `Integration '${name}' not found`
      );
    }
    return integration;
  }

  /**
   * Integration-specific accessors
   * Provides typed access to all integrations
   *
   * @example
   * ```typescript
   * // Use LeanAide integration
   * const proof = await client.integrations.leanaide.execute({
   *   type: 'theorem_proving',
   *   statement: '...'
   * });
   *
   * // Use Evolution integration
   * const evolved = await client.integrations.evolution.execute({
   *   prompt: 'Evolve...'
   * });
   * ```
   */
  get integrations(): IntegrationRegistry {
    return {
      leanaide: this.getIntegration(IntegrationName.LEANAIDE) as LeanAideIntegration,
      evolution: this.getIntegration(IntegrationName.EVOLUTION) as EvolutionIntegration,
      knowledge: this.getIntegration(IntegrationName.KNOWLEDGE) as KnowledgeIntegration,
      maker: this.getIntegration(IntegrationName.MAKER) as MakerIntegration,
      crewai: this.getIntegration(IntegrationName.CREWAI) as CrewAIIntegration,
      decomposition: this.getIntegration(IntegrationName.DECOMPOSITION) as DecompositionIntegration,
      verification: this.getIntegration(IntegrationName.VERIFICATION) as VerificationIntegration,
      assembly: this.getIntegration(IntegrationName.ASSEMBLY) as AssemblyIntegration,
      solution: this.getIntegration(IntegrationName.SOLUTION) as SolutionIntegration,
    };
  }

  /**
   * Get execution metrics
   *
   * @param executionId - Execution ID
   * @returns Metrics or null if not found
   */
  getMetrics(executionId: string): RequestMetrics | null {
    return this.executionMetrics.get(executionId) || null;
  }

  /**
   * Get all execution metrics
   *
   * @returns Map of execution metrics
   */
  getAllMetrics(): Map<string, RequestMetrics> {
    return this.executionMetrics;
  }

  /**
   * Get a summary of execution metrics
   */
  getMetricsSummary(): {
    totalRequests: number;
    successRate: number;
    averageDuration: number;
    totalRetries: number;
  } {
    const metrics = Array.from(this.executionMetrics.values());
    const totalRequests = metrics.length;
    
    if (totalRequests === 0) {
      return { totalRequests: 0, successRate: 0, averageDuration: 0, totalRetries: 0 };
    }

    const successes = metrics.filter(m => m.success).length;
    const totalDuration = metrics.reduce((sum, m) => sum + m.duration, 0);
    const totalRetries = metrics.reduce((sum, m) => sum + (m.retries || 0), 0);

    return {
      totalRequests,
      successRate: successes / totalRequests,
      averageDuration: totalDuration / totalRequests,
      totalRetries,
    };
  }

  /**
   * Clear execution metrics
   */
  clearMetrics(): void {
    this.executionMetrics.clear();
  }

  /**
   * Update retry configuration
   *
   * @param config - New retry configuration
   */
  updateRetryConfig(config: Partial<RetryConfig>): void {
    this.retryConfig = { ...this.retryConfig, ...config };
    this.log('Retry configuration updated:', this.retryConfig);
  }

  /**
   * Log debug message
   */
  private log(message: string, data?: any): void {
    if (this.debug) {
      if (data) {
        console.log(`[OpenEvolveClient] ${message}`, data);
      } else {
        console.log(`[OpenEvolveClient] ${message}`);
      }
    }
  }

  /**
   * Clear old metrics if size limit is reached
   */
  private clearOldMetrics(): void {
    if (this.executionMetrics.size >= MAX_METRICS_SIZE) {
      // Remove first 100 entries (oldest)
      const keysToRemove = Array.from(this.executionMetrics.keys()).slice(0, 100);
      for (const key of keysToRemove) {
        this.executionMetrics.delete(key);
      }
    }
  }

  /**
   * Get the backend client
   *
   * @returns Backend client instance
   */
  getBackend(): BackendClient {
    return this.backend;
  }
}

/**
 * Create an OpenEvolve client with default configuration
 *
 * @param baseUrl - Backend base URL
 * @returns Client instance
 *
 * @example
 * ```typescript
 * const client = createOpenEvolveClient('http://localhost:8000');
 * const result = await client.execute(IntegrationName.LEANAIDE, { ... });
 * ```
 */
export function createOpenEvolveClient(
  baseUrl: string
): OpenEvolveClient {
  return new OpenEvolveClient({
    baseUrl,
    timeout: 30000,
    retryAttempts: 3,
    enableWebSocket: true,
    debug: false,
  });
}

/**
 * Re-export types and enums
 */
export * from './types';
export * from './errors';
export { BackendClient } from './backend';
