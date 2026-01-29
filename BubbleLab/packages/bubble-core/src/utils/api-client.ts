/**
 * HTTP API Client Utility
 *
 * Provides a consistent way to make HTTP requests across all bubbles
 * with built-in error handling, retries, and timeout management.
 */

import { Result, wrapAsync, retry as retryOperation } from './result.js';
import { HTTP_TIMEOUT_DEFAULT, RETRY_DEFAULT_ATTEMPTS } from './constants.js';

/**
 * Custom API Error class
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public responseBody?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * HTTP methods
 */
export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

/**
 * API Client configuration
 */
export interface ApiClientConfig {
  baseURL: string;
  timeout?: number;
  retryAttempts?: number;
  defaultHeaders?: Record<string, string>;
}

/**
 * Request options
 */
export interface RequestOptions {
  method?: HttpMethod;
  headers?: Record<string, string>;
  body?: unknown;
  timeout?: number;
  retry?: boolean;
  params?: Record<string, string | number>;
}

/**
 * API Response wrapper
 */
export interface ApiResponse<T = unknown> {
  data: T;
  status: number;
  headers: Headers;
}

/**
 * HTTP API Client class
 */
export class ApiClient {
  constructor(private config: ApiClientConfig) {}

  /**
   * Perform GET request
   */
  async get<T>(endpoint: string, options?: RequestOptions): Promise<Result<ApiResponse<T>>> {
    return this.request<T>(endpoint, { ...options, method: 'GET' });
  }

  /**
   * Perform POST request
   */
  async post<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<Result<ApiResponse<T>>> {
    return this.request<T>(endpoint, { ...options, method: 'POST', body: data });
  }

  /**
   * Perform PUT request
   */
  async put<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<Result<ApiResponse<T>>> {
    return this.request<T>(endpoint, { ...options, method: 'PUT', body: data });
  }

  /**
   * Perform PATCH request
   */
  async patch<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<Result<ApiResponse<T>>> {
    return this.request<T>(endpoint, { ...options, method: 'PATCH', body: data });
  }

  /**
   * Perform DELETE request
   */
  async delete<T>(endpoint: string, options?: RequestOptions): Promise<Result<ApiResponse<T>>> {
    return this.request<T>(endpoint, { ...options, method: 'DELETE' });
  }

  /**
   * Perform HTTP request with error handling and retries
   */
  protected async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<Result<ApiResponse<T>>> {
    const {
      method = 'GET',
      headers = {},
      body,
      timeout = this.config.timeout || HTTP_TIMEOUT_DEFAULT,
      retry = options.retry !== false,
      params,
    } = options;

    // Build URL with query parameters
    let url = `${this.config.baseURL}${endpoint}`;
    if (params && Object.keys(params).length > 0) {
      const searchParams = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => {
        searchParams.append(key, String(value));
      });
      url += `?${searchParams.toString()}`;
    }

    // Prepare request
    const requestInit: RequestInit = {
      method,
      headers: {
        ...this.config.defaultHeaders,
        ...headers,
      },
    };

    // Add body for methods that support it
    if (body && ['POST', 'PUT', 'PATCH'].includes(method)) {
      if (typeof body === 'string') {
        requestInit.body = body;
      } else {
        requestInit.headers = {
          ...requestInit.headers,
          'Content-Type': 'application/json',
        };
        requestInit.body = JSON.stringify(body);
      }
    }

    // Execute with retry if enabled
    const operation = async () => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      try {
        const response = await fetch(url, {
          ...requestInit,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          const responseBody = await this.tryParseResponse(response);
          throw new ApiError(
            `HTTP ${response.status}: ${response.statusText}`,
            response.status,
            responseBody
          );
        }

        const data = await this.parseResponse<T>(response);
        return {
          data,
          status: response.status,
          headers: response.headers,
        };
      } catch (error) {
        clearTimeout(timeoutId);

        if (error instanceof Error && error.name === 'AbortError') {
          throw new ApiError('Request timeout', undefined, { timeout });
        }

        throw error;
      }
    };

    if (retry && this.config.retryAttempts !== 0) {
      const maxAttempts = this.config.retryAttempts ?? RETRY_DEFAULT_ATTEMPTS;
      return retryOperation(async () => wrapAsync(operation), {
        maxAttempts,
        shouldRetry: (error) => this.shouldRetryRequest(error),
      });
    }

    return wrapAsync(operation);
  }

  /**
   * Parse response body
   */
  private async parseResponse<T>(response: Response): Promise<T> {
    const contentType = response.headers.get('content-type');

    if (contentType?.includes('application/json')) {
      return await response.json();
    }

    if (contentType?.includes('text/')) {
      return (await response.text()) as T;
    }

    return (await response.blob()) as T;
  }

  /**
   * Try to parse response body for error details
   */
  private async tryParseResponse(response: Response): Promise<unknown> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      }
      return await response.text();
    } catch {
      return null;
    }
  }

  /**
   * Determine if request should be retried
   */
  private shouldRetryRequest(error: unknown): boolean {
    if (error instanceof ApiError) {
      // Retry on server errors and rate limiting
      return !error.statusCode || error.statusCode >= 500 || error.statusCode === 429;
    }
    return false;
  }
}

/**
 * Create API client instance
 */
export function createApiClient(config: ApiClientConfig): ApiClient {
  return new ApiClient(config);
}

/**
 * Helper for authenticated requests
 */
export class AuthenticatedApiClient extends ApiClient {
  constructor(
    config: ApiClientConfig,
    private getAuthToken: () => string | Promise<string>
  ) {
    super(config);
  }

  /**
   * Override request to add authentication
   */
  protected async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<Result<ApiResponse<T>>> {
    const token = await this.getAuthToken();

    const headers = {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    };

    return super.request<T>(endpoint, { ...options, headers });
  }
}

/**
 * Create authenticated API client
 */
export function createAuthenticatedApiClient(
  config: ApiClientConfig,
  getAuthToken: () => string | Promise<string>
): AuthenticatedApiClient {
  return new AuthenticatedApiClient(config, getAuthToken);
}
