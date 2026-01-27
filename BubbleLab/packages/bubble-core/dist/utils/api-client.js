/**
 * HTTP API Client Utility
 *
 * Provides a consistent way to make HTTP requests across all bubbles
 * with built-in error handling, retries, and timeout management.
 */
import { wrapAsync, retry as retryOperation } from './result.js';
import { HTTP_TIMEOUT_DEFAULT, RETRY_DEFAULT_ATTEMPTS } from './constants.js';
/**
 * Custom API Error class
 */
export class ApiError extends Error {
    statusCode;
    responseBody;
    constructor(message, statusCode, responseBody) {
        super(message);
        this.statusCode = statusCode;
        this.responseBody = responseBody;
        this.name = 'ApiError';
    }
}
/**
 * HTTP API Client class
 */
export class ApiClient {
    config;
    constructor(config) {
        this.config = config;
    }
    /**
     * Perform GET request
     */
    async get(endpoint, options) {
        return this.request(endpoint, { ...options, method: 'GET' });
    }
    /**
     * Perform POST request
     */
    async post(endpoint, data, options) {
        return this.request(endpoint, { ...options, method: 'POST', body: data });
    }
    /**
     * Perform PUT request
     */
    async put(endpoint, data, options) {
        return this.request(endpoint, { ...options, method: 'PUT', body: data });
    }
    /**
     * Perform PATCH request
     */
    async patch(endpoint, data, options) {
        return this.request(endpoint, { ...options, method: 'PATCH', body: data });
    }
    /**
     * Perform DELETE request
     */
    async delete(endpoint, options) {
        return this.request(endpoint, { ...options, method: 'DELETE' });
    }
    /**
     * Perform HTTP request with error handling and retries
     */
    async request(endpoint, options = {}) {
        const { method = 'GET', headers = {}, body, timeout = this.config.timeout || HTTP_TIMEOUT_DEFAULT, retry = options.retry !== false, params, } = options;
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
        const requestInit = {
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
            }
            else {
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
                    throw new ApiError(`HTTP ${response.status}: ${response.statusText}`, response.status, responseBody);
                }
                const data = await this.parseResponse(response);
                return {
                    data,
                    status: response.status,
                    headers: response.headers,
                };
            }
            catch (error) {
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
    async parseResponse(response) {
        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
            return await response.json();
        }
        if (contentType?.includes('text/')) {
            return (await response.text());
        }
        return (await response.blob());
    }
    /**
     * Try to parse response body for error details
     */
    async tryParseResponse(response) {
        try {
            const contentType = response.headers.get('content-type');
            if (contentType?.includes('application/json')) {
                return await response.json();
            }
            return await response.text();
        }
        catch {
            return null;
        }
    }
    /**
     * Determine if request should be retried
     */
    shouldRetryRequest(error) {
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
export function createApiClient(config) {
    return new ApiClient(config);
}
/**
 * Helper for authenticated requests
 */
export class AuthenticatedApiClient extends ApiClient {
    getAuthToken;
    constructor(config, getAuthToken) {
        super(config);
        this.getAuthToken = getAuthToken;
    }
    /**
     * Override request to add authentication
     */
    async request(endpoint, options = {}) {
        const token = await this.getAuthToken();
        const headers = {
            ...options.headers,
            Authorization: `Bearer ${token}`,
        };
        return super.request(endpoint, { ...options, headers });
    }
}
/**
 * Create authenticated API client
 */
export function createAuthenticatedApiClient(config, getAuthToken) {
    return new AuthenticatedApiClient(config, getAuthToken);
}
//# sourceMappingURL=api-client.js.map