/**
 * HTTP API Client Utility
 *
 * Provides a consistent way to make HTTP requests across all bubbles
 * with built-in error handling, retries, and timeout management.
 */
import { Result } from './result.js';
/**
 * Custom API Error class
 */
export declare class ApiError extends Error {
    statusCode?: number | undefined;
    responseBody?: unknown | undefined;
    constructor(message: string, statusCode?: number | undefined, responseBody?: unknown | undefined);
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
export declare class ApiClient {
    private config;
    constructor(config: ApiClientConfig);
    /**
     * Perform GET request
     */
    get<T>(endpoint: string, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
    /**
     * Perform POST request
     */
    post<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
    /**
     * Perform PUT request
     */
    put<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
    /**
     * Perform PATCH request
     */
    patch<T>(endpoint: string, data?: unknown, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
    /**
     * Perform DELETE request
     */
    delete<T>(endpoint: string, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
    /**
     * Perform HTTP request with error handling and retries
     */
    protected request<T>(endpoint: string, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
    /**
     * Parse response body
     */
    private parseResponse;
    /**
     * Try to parse response body for error details
     */
    private tryParseResponse;
    /**
     * Determine if request should be retried
     */
    private shouldRetryRequest;
}
/**
 * Create API client instance
 */
export declare function createApiClient(config: ApiClientConfig): ApiClient;
/**
 * Helper for authenticated requests
 */
export declare class AuthenticatedApiClient extends ApiClient {
    private getAuthToken;
    constructor(config: ApiClientConfig, getAuthToken: () => string | Promise<string>);
    /**
     * Override request to add authentication
     */
    protected request<T>(endpoint: string, options?: RequestOptions): Promise<Result<ApiResponse<T>>>;
}
/**
 * Create authenticated API client
 */
export declare function createAuthenticatedApiClient(config: ApiClientConfig, getAuthToken: () => string | Promise<string>): AuthenticatedApiClient;
//# sourceMappingURL=api-client.d.ts.map