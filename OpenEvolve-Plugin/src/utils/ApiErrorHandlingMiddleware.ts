/**
 * API Error Handling Middleware
 * Provides centralized error handling for API requests with retry logic, caching, and graceful degradation
 */

import { errorLogger } from './errorLogging';
import { gracefulErrorHandler } from './gracefulErrorHandler';
import { toast } from 'react-toastify';

// Define API error handling options
interface ApiErrorHandlerOptions {
  retries?: number;
  retryDelay?: number;
  exponentialBackoff?: boolean;
  cacheEnabled?: boolean;
  cacheTtl?: number; // Time to live in milliseconds
  timeout?: number; // Request timeout in milliseconds
  fallbackResponse?: any;
  logErrors?: boolean;
  notifyUser?: boolean;
  transformError?: (error: any) => any;
  validateResponse?: (response: any) => boolean;
  onRetry?: (attempt: number, error: any) => void;
  onTimeout?: (request: any) => void;
}

// Define API request configuration
interface ApiRequestConfig {
  url: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  headers?: Record<string, string>;
  body?: any;
  options?: ApiErrorHandlerOptions;
}

// Define API response
interface ApiResponse<T = any> {
  data?: T;
  error?: any;
  success: boolean;
  statusCode?: number;
  retryCount?: number;
  cached?: boolean;
}

/**
 * API Error Handler Class
 * Provides centralized error handling for API requests
 */
export class ApiErrorHandler {
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private pendingRequests: Map<string, Promise<any>> = new Map();
  private defaultOptions: ApiErrorHandlerOptions = {
    retries: 3,
    retryDelay: 1000,
    exponentialBackoff: true,
    cacheEnabled: true,
    cacheTtl: 300000, // 5 minutes
    timeout: 30000, // 30 seconds
    logErrors: true,
    notifyUser: true,
  };

  constructor(options?: Partial<ApiErrorHandlerOptions>) {
    this.defaultOptions = { ...this.defaultOptions, ...options };
  }

  /**
   * Make an API request with error handling
   */
  async request<T = any>(config: ApiRequestConfig): Promise<ApiResponse<T>> {
    const options = { ...this.defaultOptions, ...config.options };
    const cacheKey = this.generateCacheKey(config);

    // Check cache first if enabled
    if (options.cacheEnabled && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey)!;
      if (Date.now() - cached.timestamp < cached.ttl) {
        return { data: cached.data, success: true, cached: true };
      } else {
        this.cache.delete(cacheKey); // Remove expired cache
      }
    }

    // Check if we already have a pending request for this key
    if (this.pendingRequests.has(cacheKey)) {
      try {
        const data = await this.pendingRequests.get(cacheKey);
        return { data, success: true };
      } catch (error) {
        // If pending request failed, continue with new request
      }
    }

    let retryCount = 0;
    let lastError: any;

    // Create abort controller for timeout
    const abortController = new AbortController();

    // Set timeout
    const timeoutId = setTimeout(() => {
      abortController.abort();
      if (options.onTimeout) {
        options.onTimeout(config);
      }
    }, options.timeout);

    while (retryCount <= (options.retries || 0)) {
      try {
        // Create a promise that will be added to pending requests
        const requestPromise = this.makeRequestWithTimeout(
          config,
          abortController,
          options
        );

        // Store the pending request
        this.pendingRequests.set(cacheKey, requestPromise);

        const response = await requestPromise;

        // Clear timeout and pending request
        clearTimeout(timeoutId);
        this.pendingRequests.delete(cacheKey);

        // Validate response if validator is provided
        if (options.validateResponse && !options.validateResponse(response)) {
          throw new Error('Invalid response format');
        }

        // Cache successful response if caching is enabled
        if (options.cacheEnabled) {
          this.cache.set(cacheKey, {
            data: response,
            timestamp: Date.now(),
            ttl: options.cacheTtl || 300000,
          });
        }

        return { data: response, success: true, retryCount };
      } catch (error) {
        lastError = error;
        retryCount++;

        // Log error if enabled
        if (options.logErrors) {
          errorLogger.logError(error, 'error', {
            component: 'ApiErrorHandler',
            function: 'request',
            additionalData: {
              url: config.url,
              method: config.method,
              retryCount,
              maxRetries: options.retries,
              cacheKey,
            },
          });
        }

        // Notify user if enabled
        if (options.notifyUser) {
          const errorMessage = this.formatErrorMessage(error, retryCount, options.retries || 0);
          if (retryCount <= (options.retries || 0)) {
            toast.info(errorMessage);
          } else {
            toast.error(errorMessage);
          }
        }

        // Call retry callback if provided
        if (options.onRetry) {
          options.onRetry(retryCount, error);
        }

        // If this isn't the last attempt, wait before retrying
        if (retryCount <= (options.retries || 0)) {
          const delay = this.calculateRetryDelay(
            options.retryDelay || 1000,
            retryCount,
            options.exponentialBackoff
          );
          await this.delay(delay);
        }
      }
    }

    // All retries exhausted, return error response
    clearTimeout(timeoutId);
    this.pendingRequests.delete(cacheKey);

    // Apply error transformation if provided
    const transformedError = options.transformError
      ? options.transformError(lastError)
      : lastError;

    return {
      error: transformedError,
      success: false,
      retryCount: retryCount - 1,
    };
  }

  /**
   * Make an API request with timeout handling
   */
  private async makeRequestWithTimeout(
    config: ApiRequestConfig,
    abortController: AbortController,
    options: ApiErrorHandlerOptions
  ): Promise<any> {
    const { url, method, headers = {}, body } = config;

    const fetchOptions: RequestInit = {
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
      signal: abortController.signal,
    };

    if (body && method !== 'GET') {
      fetchOptions.body = typeof body === 'string' ? body : JSON.stringify(body);
    }

    const response = await fetch(url, fetchOptions);

    if (!response.ok) {
      const errorText = await response.text();
      const error = new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
      (error as any).statusCode = response.status;
      throw error;
    }

    return response.json();
  }

  /**
   * Calculate retry delay with optional exponential backoff
   */
  private calculateRetryDelay(baseDelay: number, attempt: number, exponential: boolean): number {
    if (exponential) {
      return baseDelay * Math.pow(2, attempt - 1); // 1s, 2s, 4s, 8s...
    }
    return baseDelay;
  }

  /**
   * Format error message for user notification
   */
  private formatErrorMessage(error: any, retryCount: number, maxRetries: number): string {
    const message = error.message || 'An error occurred';
    
    if (retryCount <= maxRetries) {
      return `Request failed (attempt ${retryCount}/${maxRetries}): ${message}`;
    } else {
      return `Request failed after ${maxRetries} attempts: ${message}`;
    }
  }

  /**
   * Generate cache key for request
   */
  private generateCacheKey(config: ApiRequestConfig): string {
    const keyData = {
      url: config.url,
      method: config.method,
      body: config.method !== 'GET' ? config.body : undefined,
    };
    return JSON.stringify(keyData);
  }

  /**
   * Delay helper function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Clear pending requests
   */
  clearPendingRequests(): void {
    this.pendingRequests.clear();
  }

  /**
   * Get cache size
   */
  getCacheSize(): number {
    return this.cache.size;
  }

  /**
   * Get pending request count
   */
  getPendingRequestCount(): number {
    return this.pendingRequests.size;
  }

  /**
   * Update default options
   */
  updateDefaultOptions(options: Partial<ApiErrorHandlerOptions>): void {
    this.defaultOptions = { ...this.defaultOptions, ...options };
  }
}

/**
 * HTTP Client with built-in error handling
 */
export class HttpClient {
  private errorHandler: ApiErrorHandler;

  constructor(options?: Partial<ApiErrorHandlerOptions>) {
    this.errorHandler = new ApiErrorHandler(options);
  }

  /**
   * Perform GET request
   */
  async get<T = any>(url: string, options?: ApiErrorHandlerOptions): Promise<ApiResponse<T>> {
    return this.errorHandler.request<T>({ url, method: 'GET', options });
  }

  /**
   * Perform POST request
   */
  async post<T = any>(
    url: string,
    body?: any,
    options?: ApiErrorHandlerOptions
  ): Promise<ApiResponse<T>> {
    return this.errorHandler.request<T>({ url, method: 'POST', body, options });
  }

  /**
   * Perform PUT request
   */
  async put<T = any>(
    url: string,
    body?: any,
    options?: ApiErrorHandlerOptions
  ): Promise<ApiResponse<T>> {
    return this.errorHandler.request<T>({ url, method: 'PUT', body, options });
  }

  /**
   * Perform DELETE request
   */
  async delete<T = any>(url: string, options?: ApiErrorHandlerOptions): Promise<ApiResponse<T>> {
    return this.errorHandler.request<T>({ url, method: 'DELETE', options });
  }

  /**
   * Perform PATCH request
   */
  async patch<T = any>(
    url: string,
    body?: any,
    options?: ApiErrorHandlerOptions
  ): Promise<ApiResponse<T>> {
    return this.errorHandler.request<T>({ url, method: 'PATCH', body, options });
  }

  /**
   * Update error handler options
   */
  updateOptions(options: Partial<ApiErrorHandlerOptions>): void {
    this.errorHandler.updateDefaultOptions(options);
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.errorHandler.clearCache();
  }

  /**
   * Get cache size
   */
  getCacheSize(): number {
    return this.errorHandler.getCacheSize();
  }
}

/**
 * Create a configured HTTP client instance
 */
export function createHttpClient(options?: Partial<ApiErrorHandlerOptions>): HttpClient {
  return new HttpClient(options);
}

/**
 * Axios-style interceptors for API error handling
 */
export class ApiInterceptor {
  private requestInterceptors: Array<(config: ApiRequestConfig) => ApiRequestConfig> = [];
  private responseInterceptors: Array<(response: ApiResponse) => ApiResponse> = [];
  private errorInterceptors: Array<(error: any) => any> = [];

  /**
   * Add request interceptor
   */
  addRequestInterceptor(interceptor: (config: ApiRequestConfig) => ApiRequestConfig): number {
    this.requestInterceptors.push(interceptor);
    return this.requestInterceptors.length - 1;
  }

  /**
   * Add response interceptor
   */
  addResponseInterceptor(interceptor: (response: ApiResponse) => ApiResponse): number {
    this.responseInterceptors.push(interceptor);
    return this.responseInterceptors.length - 1;
  }

  /**
   * Add error interceptor
   */
  addErrorInterceptor(interceptor: (error: any) => any): number {
    this.errorInterceptors.push(interceptor);
    return this.errorInterceptors.length - 1;
  }

  /**
   * Apply request interceptors
   */
  applyRequestInterceptors(config: ApiRequestConfig): ApiRequestConfig {
    return this.requestInterceptors.reduce(
      (acc, interceptor) => interceptor(acc),
      config
    );
  }

  /**
   * Apply response interceptors
   */
  applyResponseInterceptors(response: ApiResponse): ApiResponse {
    return this.responseInterceptors.reduce(
      (acc, interceptor) => interceptor(acc),
      response
    );
  }

  /**
   * Apply error interceptors
   */
  applyErrorInterceptors(error: any): any {
    return this.errorInterceptors.reduce(
      (acc, interceptor) => interceptor(acc),
      error
    );
  }
}

// Create a default HTTP client instance
export const apiClient = createHttpClient();

/**
 * Higher-order function to wrap API calls with error handling
 */
export function withApiErrorHandling<T>(
  apiCall: (config: ApiRequestConfig) => Promise<T>,
  options?: ApiErrorHandlerOptions
) {
  return async (config: ApiRequestConfig): Promise<ApiResponse<T>> => {
    const errorHandler = new ApiErrorHandler(options);
    return errorHandler.request<T>(config);
  };
}