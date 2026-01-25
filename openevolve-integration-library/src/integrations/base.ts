/**
 * Base Integration Adapter
 *
 * Provides common functionality for all integration adapters:
 * - Error handling
 * - Validation
 * - Execution lifecycle
 * - Progress tracking
 * - Request/response transformation
 */

import type { BackendClient } from '../api/backend';
import type {
  IntegrationAdapter,
  ValidationResult,
  ParameterSchema,
  ProgressUpdate,
  ExecutionOptions,
  IntegrationHealth,
  RetryConfig,
  CircuitBreakerConfig,
  CircuitState,
} from '../api/types';
import {
  IntegrationError,
  ValidationError as ValidationErrorClass,
  TimeoutError,
  CancellationError,
  CircuitBreakerError,
  ParseError,
  createIntegrationError,
} from '../api/errors';




import { generateId, isPlainObject, validateInputs, retryWithBackoff } from '../utils/helpers';

/**
 * Base class for all integration adapters
 */
export abstract class BaseIntegrationAdapter implements IntegrationAdapter {
  protected client: BackendClient;
  public name: string;
  protected version: string;
  protected description: string;
  protected retryConfig?: Partial<RetryConfig>;
  
  // Circuit Breaker State
  private circuitState: CircuitState = 'closed';
  private failureCount: number = 0;
  private successCount: number = 0;
  private lastFailureTime: number = 0;
  protected circuitBreakerConfig: CircuitBreakerConfig;
  protected onGlobalError?: (error: IntegrationError) => void;


  /**
   * Create a new integration adapter
   *
   * @param client - Backend client instance
   * @param name - Integration name
   * @param version - Integration version
   * @param description - Integration description
   * @param retryConfig - Optional retry configuration
   * @param circuitBreakerConfig - Optional circuit breaker configuration
   */
  constructor(
    client: BackendClient,
    name: string,
    version: string,
    description: string,
    retryConfig?: Partial<RetryConfig>,
    circuitBreakerConfig?: Partial<CircuitBreakerConfig>
  ) {
    this.client = client;
    this.name = name;
    this.version = version;
    this.description = description;
    this.retryConfig = retryConfig;
    
    // Default circuit breaker config
    this.circuitBreakerConfig = {
      enabled: circuitBreakerConfig?.enabled ?? true,
      failureThreshold: circuitBreakerConfig?.failureThreshold ?? 5,
      resetTimeout: circuitBreakerConfig?.resetTimeout ?? 30000,
      successThreshold: circuitBreakerConfig?.successThreshold ?? 2,
    };
  }

  /**
   * Set global error handler
   */
  public setGlobalErrorHandler(handler: (error: IntegrationError) => void): void {
    this.onGlobalError = handler;
  }


  /**
   * Check if the circuit is open and throw if it is
   */
  protected checkCircuit(): void {
    if (!this.circuitBreakerConfig.enabled) return;

    if (this.circuitState === 'open') {
      const now = Date.now();
      if (now - this.lastFailureTime > this.circuitBreakerConfig.resetTimeout) {
        this.client.log?.(`[${this.name}] Circuit breaker moving to half-open state`);
        this.circuitState = 'half-open';
        this.successCount = 0;
      } else {
        throw new CircuitBreakerError(this.name, {
          cooldownRemaining: this.circuitBreakerConfig.resetTimeout - (now - this.lastFailureTime)
        });
      }
    }
  }

  /**
   * Record a successful call for circuit breaker
   */
  protected recordSuccess(): void {
    if (!this.circuitBreakerConfig.enabled) return;

    if (this.circuitState === 'half-open') {
      this.successCount++;
      if (this.successCount >= this.circuitBreakerConfig.successThreshold) {
        this.client.log?.(`[${this.name}] Circuit breaker closed`);
        this.circuitState = 'closed';
        this.failureCount = 0;
      }
    } else if (this.circuitState === 'closed') {
      this.failureCount = 0;
    }
  }

  /**
   * Record a failed call for circuit breaker
   */
  protected recordFailure(error: any): void {
    if (!this.circuitBreakerConfig.enabled) return;

    // Only count certain types of errors as circuit-breaking failures
    // (e.g., connection errors, timeouts, server errors)
    const integrationError = error instanceof IntegrationError ? error : this.handleError(error);
    const criticalCodes = ['CONNECTION_ERROR', 'TIMEOUT_ERROR', 'NETWORK_ERROR', 'EXECUTION_ERROR'];
    
    if (!criticalCodes.includes(integrationError.code)) {
      return;
    }

    this.failureCount++;
    this.lastFailureTime = Date.now();

    if (this.circuitState === 'closed' && this.failureCount >= this.circuitBreakerConfig.failureThreshold) {
      this.client.log?.(`[${this.name}] Circuit breaker opened for ${this.name}`);
      this.circuitState = 'open';
    } else if (this.circuitState === 'half-open') {
      this.client.log?.(`[${this.name}] Circuit breaker re-opened for ${this.name}`);
      this.circuitState = 'open';
    }
  }

  /**
   * Get current circuit state
   */
  getCircuitState(): CircuitState {
    return this.circuitState;
  }


  /**
   * Get integration name
   */
  getName(): string {
    return this.name;
  }

  /**
   * Get integration version
   */
  getVersion(): string {
    return this.version;
  }

  /**
   * Get integration description
   */
  getDescription(): string {
    return this.description;
  }

  /**
   * Execute integration - must be implemented by subclasses
   */
  abstract execute<TInputs, TResult>(
    inputs: TInputs,
    options?: ExecutionOptions
  ): Promise<TResult>;

  /**
   * Validate inputs using the integration schema
   */
  async validate<TInputs>(inputs: TInputs): Promise<ValidationResult> {
    return validateInputs(inputs, this.getSchema());
  }

  /**
   * Get parameter schema - must be implemented by subclasses
   */
  abstract getSchema(): ParameterSchema;

  /**
   * Stream execution with progress updates
   */
  async executeStream<TInputs, TResult>(
    inputs: TInputs,
    _onProgress: (update: ProgressUpdate) => void,
    options?: ExecutionOptions
  ): Promise<TResult> {
    const validation = await this.validate(inputs);
    if (!validation.valid) {
      throw new ValidationErrorClass(this.name, validation.errors);
    }
    // Default implementation doesn't stream via WebSocket, uses global client socket via execute
    return this.execute<TInputs, TResult>(inputs, options);
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<IntegrationHealth> {
    const startTime = Date.now();

    try {
      if (!this.client) {
        throw new Error('Backend client not initialized');
      }

      // Ping the backend
      const isOnline = await this.client.ping();

      return {
        name: this.name,
        status: isOnline ? 'available' : 'unavailable',
        responseTime: Date.now() - startTime,
        lastError: undefined,
        endpoints: this.getEndpoints(),
      };
    } catch (error) {
      return {
        name: this.name,
        status: 'unavailable',
        responseTime: Date.now() - startTime,
        lastError: error instanceof Error ? error.message : String(error || 'Unknown health check error'),
        endpoints: [],
      };
    }
  }


  /**
   * Get integration endpoints - to be implemented by subclasses
   */
  protected abstract getEndpoints(): string[];

  /**
   * Execute backend request with common error handling and transformation
   */
  protected async requestBackend<TResponse>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH',
    endpoint: string,
    data?: any,
    options?: ExecutionOptions
  ): Promise<TResponse> {
    this.checkCircuit();
    const maxRetries = options?.retries ?? this.retryConfig?.maxAttempts ?? 3;
    
    try {
      const result = await retryWithBackoff(
        async () => {
          const abortController = new AbortController();
          let timeoutId: any;

          if (options?.timeout) {
            timeoutId = setTimeout(() => {
              abortController.abort();
            }, options.timeout);
          }

          try {
            const transformedData = data ? this.transformRequest(data) : undefined;
            
            let response: any;
            const axiosConfig = {
              signal: options?.signal || abortController.signal,
              timeout: options?.timeout,
            };

            switch (method) {
              case 'GET':
                response = await this.client.get<TResponse>(endpoint, axiosConfig);
                break;
              case 'POST':
                response = await this.client.post<any, TResponse>(endpoint, transformedData, axiosConfig);
                break;
              case 'PUT':
                response = await this.client.put<any, TResponse>(endpoint, transformedData, axiosConfig);
                break;
              case 'DELETE':
                response = await this.client.delete<TResponse>(endpoint, axiosConfig);
                break;
              case 'PATCH':
                response = await this.client.patch<any, TResponse>(endpoint, transformedData, axiosConfig);
                break;
            }

            if (timeoutId) clearTimeout(timeoutId);
            
            try {
              const result = this.transformResponse<TResponse>(response);
              const validation = this.validateResponse(result);
              if (!validation.valid) {
                throw new ParseError(this.name, 'Response validation failed', { errors: validation.errors });
              }
              return result;
            } catch (parseError) {
              if (parseError instanceof IntegrationError) throw parseError;
              throw new ParseError(this.name, 'Failed to parse or transform backend response', { 
                originalError: parseError,
                responseData: response 
              });
            }
          } catch (error: any) {

            if (timeoutId) clearTimeout(timeoutId);
            
            if (error.name === 'AbortError' || error.code === 'ECONNABORTED') {
              throw new TimeoutError(this.name, options?.timeout || 0);
            }
            
            throw this.handleError(error);
          }
        },
        maxRetries > 0 ? maxRetries - 1 : 0,
        this.retryConfig?.initialDelay || 1000,
        (error) => {
          // Record failure for circuit breaker but only if it's not a validation/auth error
          // Note: we record it even if we are going to retry, because multiple retries 
          // that all fail should contribute to opening the circuit.
          
          const integrationError = error instanceof IntegrationError ? error : this.handleError(error);
          
          // Circular dependency check: We should ideally import isRetryableError but let's keep it simple
          const retryableCodes = ['NETWORK_ERROR', 'TIMEOUT_ERROR', 'RATE_LIMIT_ERROR', 'CONNECTION_ERROR'];
          const shouldRetry = retryableCodes.includes(integrationError.code) || 
                 (integrationError.code === 'EXECUTION_ERROR' && integrationError.message.includes('Server error'));

          return shouldRetry;
        },
        (error, attempt, delay) => {
          if (options?.onRetry) {
            try {
              options.onRetry(error instanceof IntegrationError ? error : this.handleError(error), attempt, delay);
            } catch (cbError) {
              // Ignore errors in callback
            }
          }
        }
      );


      this.recordSuccess();
      return result;
    } catch (error) {
      this.recordFailure(error);
      throw error;
    }
  }


  /**
   * Execute backend request with common error handling (POST legacy)
   */
  protected async executeBackend<TRequest, TResponse>(
    endpoint: string,
    request: TRequest,
    executionId?: string,
    options?: ExecutionOptions
  ): Promise<TResponse> {
    return this.requestBackend<TResponse>(
      'POST', 
      endpoint, 
      executionId ? { ...request as any, executionId } : request, 
      options
    );
  }

  /**
   * Stream execution via WebSocket
   */
  protected async streamExecute<TRequest, TResponse>(
    endpoint: string,
    request: TRequest,
    onProgress: (update: ProgressUpdate) => void,
    options?: ExecutionOptions
  ): Promise<TResponse> {
    this.checkCircuit();

    return new Promise((resolve, reject) => {
      const executionId = (request as any).executionId || generateId();
      let isFinalized = false;
      let timeoutId: any;

      const finalize = (fn: () => void) => {
        if (isFinalized) return;
        isFinalized = true;
        if (timeoutId) clearTimeout(timeoutId);
        if (ws) ws.disconnect();
        fn();
      };

      // Setup WebSocket handlers
      const handlers = {
        onConnect: () => {
          this.client.log?.(`[${this.name}] WebSocket connected for execution ${executionId}`);
        },
        onError: (error: Error) => {
          this.recordFailure(error);
          finalize(() => reject(this.handleError(error)));
        },
        onMessage: (message: any) => {
          if (message.executionId !== executionId) return;

          if (message.type === 'progress') {
            onProgress(message.data as ProgressUpdate);
          } else if (message.type === 'complete') {
            try {
              const result = this.transformResponse<TResponse>(message.data);
              const validation = this.validateResponse(result);
              if (!validation.valid) {
                throw new ParseError(this.name, 'Stream response validation failed', { errors: validation.errors });
              }
              this.recordSuccess();
              finalize(() => resolve(result));
            } catch (parseError) {
              const integrationError = parseError instanceof IntegrationError 
                ? parseError 
                : new ParseError(this.name, 'Failed to parse or transform stream response', { originalError: parseError });
              this.recordFailure(integrationError);
              finalize(() => reject(integrationError));
            }
          } else if (message.type === 'error') {

            const errorMessage = message.data?.message || 'Unknown stream error';
            const error = new Error(errorMessage);
            this.recordFailure(error);
            finalize(() => reject(this.handleError(error)));
          }
        }
      };


      // Connect to WebSocket
      const ws = this.client.websocket(`/ws/${this.name}/${executionId}`, handlers);

      // Send execution request
      this.client.post<TRequest, { executionId: string }>(
        endpoint,
        { ...request, executionId },
        { signal: options?.signal }
      ).catch((error) => {
        finalize(() => reject(this.handleError(error)));
      });

      // Handle external cancellation
      if (options?.signal) {
        options.signal.addEventListener('abort', () => {
          finalize(() => reject(new CancellationError(this.name, executionId)));
        });
      }

      // Set timeout
      if (options?.timeout) {
        timeoutId = setTimeout(() => {
          finalize(() => reject(new TimeoutError(this.name, options.timeout!)));
        }, options.timeout);
      }
    });
  }

  /**
   * Handle errors with integration-specific context
   */
  protected handleError(error: any): never {
    const integrationError = createIntegrationError(this.name, error);
    if (this.onGlobalError) {
      try {
        this.onGlobalError(integrationError);
      } catch (cbError) {
        // Ignore errors in error handler
      }
    }
    throw integrationError;
  }





  /**
   * Transform request to backend format
   * Override in subclasses for custom transformation
   */
  protected transformRequest<T>(data: T): any {
    return data;
  }

  /**
   * Transform response from backend format
   * Override in subclasses for custom transformation
   */
  protected transformResponse<T>(data: any): T {
    return data as T;
  }

  /**
   * Validate response data
   * Override in subclasses for custom validation
   */
  protected validateResponse<T>(_data: T): { valid: boolean; errors?: string[] } {
    return { valid: true };
  }


  /**
   * Validate required fields
   */
  protected validateRequired<T>(inputs: T, requiredFields: (keyof T)[]): string[] {
    const errors: string[] = [];

    if (!inputs || typeof inputs !== 'object') {
      return ['Inputs must be an object'];
    }

    for (const field of requiredFields) {
      if (inputs[field] === undefined || inputs[field] === null) {
        errors.push(`Required field '${String(field)}' is missing`);
      }
    }

    return errors;
  }

  /**
   * Validate field types
   */
  protected validateTypes<T>(
    inputs: T,
    typeDefinitions: Partial<Record<keyof T, string>>
  ): string[] {
    const errors: string[] = [];

    for (const [field, expectedType] of Object.entries(typeDefinitions)) {
      const value = inputs[field as keyof T];

      if (value !== undefined && value !== null) {
        let isValid = false;
        const actualType = typeof value;

        if (expectedType === 'array') {
          isValid = Array.isArray(value);
        } else if (expectedType === 'object') {
          isValid = actualType === 'object' && !Array.isArray(value);
        } else {
          isValid = actualType === expectedType;
        }

        if (!isValid) {
          errors.push(
            `Field '${field}' has invalid type: expected ${expectedType}, got ${
              Array.isArray(value) ? 'array' : actualType
            }`
          );
        }
      }
    }

    return errors;
  }

  /**
   * Validate enum values
   */
  protected validateEnum<T>(
    inputs: T,
    enumDefinitions: Partial<Record<keyof T, any[]>>
  ): string[] {
    const errors: string[] = [];

    if (!inputs || typeof inputs !== 'object') {
      return [];
    }

    for (const [field, validValues] of Object.entries(enumDefinitions)) {
      const values = validValues as any[];
      const value = inputs[field as keyof T];

      if (value !== undefined && !values.includes(value)) {
        const displayValue = isPlainObject(value) ? JSON.stringify(value) : String(value);
        errors.push(
          `Field '${field}' has invalid value: ${displayValue}. Valid values are: ${values.join(', ')}`
        );
      }
    }

    return errors;
  }

  /**
   * Validate numeric ranges
   */
  protected validateRanges<T>(
    inputs: T,
    rangeDefinitions: Partial<Record<
      keyof T,
      { min?: number; max?: number }
    >>
  ): string[] {
    const errors: string[] = [];

    for (const [field, rangeDef] of Object.entries(rangeDefinitions)) {
      const range = rangeDef as { min?: number; max?: number };
      const value = inputs[field as keyof T];

      if (typeof value === 'number') {
        if (range.min !== undefined && value < range.min) {
          errors.push(
            `Field '${field}' must be at least ${range.min}, got ${value}`
          );
        }

        if (range.max !== undefined && value > range.max) {
          errors.push(
            `Field '${field}' must be at most ${range.max}, got ${value}`
          );
        }
      }
    }

    return errors;
  }
}
