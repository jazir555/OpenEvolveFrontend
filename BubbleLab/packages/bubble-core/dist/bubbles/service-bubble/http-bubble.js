import { z } from 'zod';
import { ServiceBubble } from '../../types/service-bubble-class.js';
import { CredentialType } from '@bubblelab/shared-schemas';
// ============================================================================
// PARAMETER SCHEMAS
// ============================================================================
const HttpMethodSchema = z.enum([
    'GET',
    'POST',
    'PUT',
    'PATCH',
    'DELETE',
    'HEAD',
    'OPTIONS',
]);
const RetryStrategySchema = z.enum(['exponential', 'linear', 'none']);
const HttpBubbleParamsSchema = z.object({
    operation: z.enum([
        'request',
        'get',
        'post',
        'put',
        'patch',
        'delete',
        'head',
        'options',
    ]),
    url: z.string().url('Must be a valid URL').describe('The URL to make the HTTP request to'),
    method: HttpMethodSchema.optional().describe('HTTP method (overrides operation default)'),
    headers: z.record(z.string()).optional().describe('HTTP headers to include in the request'),
    body: z
        .union([z.string(), z.record(z.unknown()), z.instanceof(FormData), z.instanceof(URLSearchParams)])
        .optional()
        .describe('Request body (string, JSON, FormData, or URLSearchParams)'),
    queryParams: z.record(z.union([z.string(), z.number(), z.boolean()])).optional().describe('Query parameters to append to URL'),
    timeout: z
        .number()
        .min(100)
        .max(300000)
        .default(30000)
        .describe('Request timeout in milliseconds (default: 30000, max: 300000)'),
    followRedirects: z.boolean().default(true).describe('Whether to follow HTTP redirects (default: true)'),
    maxRedirects: z.number().int().min(0).max(20).default(20).describe('Maximum number of redirects to follow'),
    // Retry configuration
    retryEnabled: z.boolean().default(true).describe('Enable automatic retry on failures (default: true)'),
    maxRetries: z.number().int().min(0).max(10).default(3).describe('Maximum number of retry attempts (default: 3)'),
    retryStrategy: RetryStrategySchema.default('exponential').describe('Retry strategy: exponential, linear, or none (default: exponential)'),
    retryDelay: z.number().min(0).default(1000).describe('Initial retry delay in milliseconds (default: 1000)'),
    retryMultiplier: z.number().min(1).default(2).describe('Multiplier for exponential backoff (default: 2)'),
    retryableStatusCodes: z.array(z.number()).default([408, 429, 500, 502, 503, 504]).describe('HTTP status codes that trigger retry'),
    retryableErrors: z.array(z.string()).default(['ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', 'EAI_AGAIN']).describe('Error codes that trigger retry'),
    // Circuit breaker configuration
    circuitBreakerEnabled: z.boolean().default(false).describe('Enable circuit breaker pattern (default: false)'),
    circuitBreakerThreshold: z.number().int().min(1).default(5).describe('Number of failures before opening circuit (default: 5)'),
    circuitBreakerTimeout: z.number().min(1000).default(60000).describe('Time in milliseconds to keep circuit open (default: 60000)'),
    circuitBreakerHalfOpenAttempts: z.number().int().min(1).default(1).describe('Number of successful requests to close circuit (default: 1)'),
    // Authentication
    authType: z
        .enum(['none', 'bearer', 'basic', 'api-key', 'api-key-header', 'custom'])
        .default('none')
        .describe('Authentication type'),
    authHeader: z.string().optional().describe('Custom header name when authType is "custom"'),
    credentials: z.record(z.nativeEnum(CredentialType), z.string()).optional(),
    // Response handling
    responseType: z.enum(['json', 'text', 'blob', 'arraybuffer']).default('json').describe('Expected response type (default: json)'),
    validateStatus: z.boolean().default(true).describe('Validate HTTP status codes (default: true)'),
    successStatusCodes: z.array(z.number()).default([200, 201, 202, 204]).describe('HTTP status codes considered successful'),
    // SSL/TLS
    rejectUnauthorized: z.boolean().default(true).describe('Reject unauthorized SSL certificates (default: true)'),
});
// ============================================================================
// RESULT SCHEMA
// ============================================================================
const HttpBubbleResultSchema = z.object({
    success: z.boolean().describe('Whether the request was ultimately successful'),
    data: z.unknown().optional().describe('Parsed response data'),
    json: z.unknown().optional().describe('JSON parsed response (alias for data)'),
    status: z.number().describe('HTTP status code'),
    statusText: z.string().describe('HTTP status text'),
    headers: z.record(z.string()).describe('Response headers'),
    body: z.string().describe('Response body as string'),
    contentType: z.string().optional().describe('Content-Type header'),
    contentLength: z.number().optional().describe('Content-Length header'),
    error: z.string().describe('Error message if request failed'),
    errorCode: z.string().optional().describe('Error code for network/timeout errors'),
    // Metrics
    metrics: z.object({
        totalAttempts: z.number().describe('Total number of attempts made'),
        responseTime: z.number().describe('Total response time in milliseconds'),
        lastAttemptTime: z.number().describe('Last attempt response time in milliseconds'),
        retryCount: z.number().describe('Number of retries performed'),
        fromCache: z.boolean().optional().describe('Whether response came from cache'),
        circuitBreakerTripped: z.boolean().optional().describe('Whether circuit breaker was triggered'),
    }),
    // Request info
    request: z.object({
        url: z.string().describe('Final URL after redirects'),
        method: z.string().describe('HTTP method used'),
        headers: z.record(z.string()).optional().describe('Request headers sent'),
    }),
});
// ============================================================================
// MAIN BUBBLE CLASS
// ============================================================================
export class HttpBubble extends ServiceBubble {
    static service = 'nodex-core';
    static authType = 'apikey';
    static bubbleName = 'http';
    static type = 'service';
    static schema = HttpBubbleParamsSchema;
    static resultSchema = HttpBubbleResultSchema;
    static shortDescription = 'Production-ready HTTP client with retry, circuit breaker, and advanced features';
    static longDescription = `
    Advanced HTTP client service bubble with enterprise-grade features.

    Features:
    - All HTTP methods (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS)
    - Automatic retry with exponential backoff
    - Circuit breaker pattern for fault tolerance
    - Comprehensive error handling
    - Timeout handling
    - Request/response validation
    - Query parameters and headers support
    - Multiple body types (JSON, text, FormData, URLSearchParams)
    - Response parsing (JSON, text, blob, arraybuffer)
    - HTTP status code handling
    - Request/response metrics
    - SSL/TLS configuration

    Use cases:
    - Calling external REST APIs
    - Webhook requests
    - Data fetching from web services
    - Integration with third-party services
    - Health checks and monitoring
    - Microservice communication
  `;
    static alias = 'http';
    // Circuit breaker state (shared across all instances for same URL)
    static circuitBreakerStates = new Map();
    constructor(params, context, instanceId) {
        super(params, context, instanceId);
    }
    chooseCredential() {
        const credentials = this.params.credentials;
        if (!credentials || typeof credentials !== 'object') {
            return undefined;
        }
        return credentials[CredentialType.CUSTOM_AUTH_KEY];
    }
    async testCredential() {
        // For HTTP bubble, credentials are optional
        // Test by making a simple request if URL is provided
        try {
            if (this.params.url) {
                const result = await this.executeRequest(1);
                return result.success;
            }
            return true;
        }
        catch {
            return false;
        }
    }
    async performAction(context) {
        void context;
        // Check circuit breaker state first
        if (this.params.circuitBreakerEnabled) {
            const circuitState = HttpBubble.getCircuitBreakerState(this.params.url);
            if (circuitState.isOpen) {
                const now = Date.now();
                if (now < circuitState.nextAttemptTime) {
                    // Circuit is still open
                    return this.createCircuitBreakerResult();
                }
                // Circuit half-open - allow one attempt
            }
        }
        // Execute request with retry logic
        return this.executeWithRetry();
    }
    async executeWithRetry() {
        const maxAttempts = this.params.retryEnabled ? this.params.maxRetries + 1 : 1;
        let lastError = null;
        let lastResult = null;
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const startTime = Date.now();
            try {
                const result = await this.executeRequest(attempt);
                // Check if we should retry on this status code
                if (this.shouldRetry(result, attempt, maxAttempts)) {
                    lastError = new Error(result.error || `HTTP ${result.status}`);
                    await this.delay(this.calculateRetryDelay(attempt));
                    continue;
                }
                // Success or non-retryable error
                if (result.success) {
                    this.handleCircuitBreakerSuccess();
                }
                else {
                    this.handleCircuitBreakerFailure();
                }
                return result;
            }
            catch (error) {
                lastError = error;
                const errorMessage = lastError.message;
                // Check if error is retryable
                if (this.isRetryableError(errorMessage) && attempt < maxAttempts) {
                    await this.delay(this.calculateRetryDelay(attempt));
                    continue;
                }
                // Non-retryable error or max attempts reached
                this.handleCircuitBreakerFailure();
                return this.createErrorResult(lastError, attempt, Date.now() - startTime);
            }
        }
        // All retries exhausted
        this.handleCircuitBreakerFailure();
        return lastResult || this.createErrorResult(lastError || new Error('Max retries exceeded'), maxAttempts, 0);
    }
    async executeRequest(attempt) {
        const startTime = Date.now();
        const url = this.buildUrl();
        try {
            console.log(`[HttpBubble] Attempt ${attempt}: ${this.params.method || this.getOperationMethod()} ${url}`);
            // Build request options
            const options = this.buildRequestOptions();
            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), this.params.timeout);
            options.signal = controller.signal;
            // Execute fetch
            const response = await fetch(url, options);
            clearTimeout(timeoutId);
            const responseTime = Date.now() - startTime;
            // Parse response
            const responseBody = await this.parseResponse(response, this.params.responseType);
            const responseHeaders = this.parseHeaders(response.headers);
            // Determine success
            const isSuccess = this.isSuccessStatus(response.status);
            const result = {
                success: isSuccess,
                data: responseBody,
                json: responseBody, // Alias for data
                status: response.status,
                statusText: response.statusText,
                headers: responseHeaders,
                body: typeof responseBody === 'string' ? responseBody : JSON.stringify(responseBody),
                contentType: response.headers.get('content-type') || undefined,
                contentLength: response.headers.get('content-length')
                    ? parseInt(response.headers.get('content-length'), 10)
                    : undefined,
                error: isSuccess ? '' : `HTTP ${response.status}: ${response.statusText}`,
                metrics: {
                    totalAttempts: attempt,
                    responseTime: responseTime,
                    lastAttemptTime: responseTime,
                    retryCount: attempt - 1,
                },
                request: {
                    url: response.url || url,
                    method: options.method || 'GET',
                    headers: options.headers,
                },
            };
            console.log(`[HttpBubble] Request completed: ${response.status} (${responseTime}ms)`);
            return result;
        }
        catch (error) {
            const responseTime = Date.now() - startTime;
            const errorObj = error;
            // Handle AbortError (timeout)
            if (errorObj.name === 'AbortError') {
                throw new Error(`Request timeout after ${this.params.timeout}ms`);
            }
            throw errorObj;
        }
    }
    buildUrl() {
        let url = this.params.url;
        // Add query parameters
        if (this.params.queryParams && Object.keys(this.params.queryParams).length > 0) {
            const searchParams = new URLSearchParams();
            for (const [key, value] of Object.entries(this.params.queryParams)) {
                searchParams.append(key, String(value));
            }
            const queryString = searchParams.toString();
            url += url.includes('?') ? `&${queryString}` : `?${queryString}`;
        }
        return url;
    }
    buildRequestOptions() {
        const method = this.params.method || this.getOperationMethod();
        const options = {
            method,
            headers: this.buildHeaders(),
            redirect: this.params.followRedirects ? 'follow' : 'manual',
        };
        // Add body for appropriate methods
        if (this.params.body && !['GET', 'HEAD'].includes(method)) {
            options.body = this.serializeBody();
        }
        return options;
    }
    buildHeaders() {
        const headers = {
            'User-Agent': 'BubbleLab-HttpBubble/2.0',
            'Accept': this.getAcceptHeader(),
        };
        // Add custom headers
        if (this.params.headers) {
            Object.assign(headers, this.params.headers);
        }
        // Add authentication
        const credential = this.chooseCredential();
        if (credential && this.params.authType !== 'none') {
            this.addAuthHeaders(headers, credential);
        }
        // Add Content-Type if body is object and not already set
        if (this.params.body &&
            typeof this.params.body === 'object' &&
            !(this.params.body instanceof FormData) &&
            !(this.params.body instanceof URLSearchParams) &&
            !headers['Content-Type']) {
            headers['Content-Type'] = 'application/json';
        }
        return headers;
    }
    addAuthHeaders(headers, credential) {
        switch (this.params.authType) {
            case 'bearer':
                headers['Authorization'] = `Bearer ${credential}`;
                break;
            case 'basic':
                headers['Authorization'] = `Basic ${credential}`;
                break;
            case 'api-key':
                headers['X-API-Key'] = credential;
                break;
            case 'api-key-header':
                headers['Api-Key'] = credential;
                break;
            case 'custom':
                if (this.params.authHeader) {
                    headers[this.params.authHeader] = credential;
                }
                break;
        }
    }
    getAcceptHeader() {
        switch (this.params.responseType) {
            case 'json':
                return 'application/json, */*';
            case 'text':
                return 'text/plain, */*';
            case 'blob':
                return '*/*';
            case 'arraybuffer':
                return '*/*';
            default:
                return '*/*';
        }
    }
    serializeBody() {
        if (!this.params.body) {
            return null;
        }
        if (typeof this.params.body === 'string') {
            return this.params.body;
        }
        if (this.params.body instanceof FormData || this.params.body instanceof URLSearchParams) {
            return this.params.body;
        }
        return JSON.stringify(this.params.body);
    }
    async parseResponse(response, responseType) {
        switch (responseType) {
            case 'json':
                const text = await response.text();
                try {
                    return JSON.parse(text);
                }
                catch {
                    return text; // Return as text if not valid JSON
                }
            case 'text':
                return await response.text();
            case 'blob':
                return await response.blob();
            case 'arraybuffer':
                return await response.arrayBuffer();
            default:
                return await response.text();
        }
    }
    parseHeaders(headers) {
        const result = {};
        headers.forEach((value, key) => {
            result[key] = value;
        });
        return result;
    }
    getOperationMethod() {
        const methodMap = {
            request: 'GET',
            get: 'GET',
            post: 'POST',
            put: 'PUT',
            patch: 'PATCH',
            delete: 'DELETE',
            head: 'HEAD',
            options: 'OPTIONS',
        };
        return methodMap[this.params.operation] || 'GET';
    }
    isSuccessStatus(status) {
        if (!this.params.validateStatus) {
            return true;
        }
        return this.params.successStatusCodes.includes(status);
    }
    shouldRetry(result, attempt, maxAttempts) {
        if (attempt >= maxAttempts) {
            return false;
        }
        if (!this.params.retryEnabled) {
            return false;
        }
        // Retry on specific status codes
        if (this.params.retryableStatusCodes.includes(result.status)) {
            return true;
        }
        return false;
    }
    isRetryableError(errorMessage) {
        return this.params.retryableErrors.some(code => errorMessage.includes(code));
    }
    calculateRetryDelay(attempt) {
        if (!this.params.retryEnabled) {
            return 0;
        }
        switch (this.params.retryStrategy) {
            case 'exponential':
                return this.params.retryDelay * Math.pow(this.params.retryMultiplier, attempt - 1);
            case 'linear':
                return this.params.retryDelay * attempt;
            case 'none':
                return 0;
            default:
                return this.params.retryDelay;
        }
    }
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    // ============================================================================
    // CIRCUIT BREAKER METHODS
    // ============================================================================
    static getCircuitBreakerState(url) {
        const normalizedUrl = this.normalizeUrl(url);
        return HttpBubble.circuitBreakerStates.get(normalizedUrl) || {
            isOpen: false,
            failureCount: 0,
            lastFailureTime: 0,
            nextAttemptTime: 0,
        };
    }
    static setCircuitBreakerState(url, state) {
        const normalizedUrl = this.normalizeUrl(url);
        HttpBubble.circuitBreakerStates.set(normalizedUrl, state);
    }
    static normalizeUrl(url) {
        try {
            const parsed = new URL(url);
            return `${parsed.protocol}//${parsed.host}`;
        }
        catch {
            return url;
        }
    }
    handleCircuitBreakerSuccess() {
        if (!this.params.circuitBreakerEnabled) {
            return;
        }
        const state = HttpBubble.getCircuitBreakerState(this.params.url);
        if (state.isOpen) {
            // Circuit is half-open, successful request closes it
            state.failureCount = 0;
            state.isOpen = false;
            HttpBubble.setCircuitBreakerState(this.params.url, state);
            console.log(`[HttpBubble] Circuit breaker closed for ${this.params.url}`);
        }
        else {
            // Reset failure count on successful request
            state.failureCount = 0;
            HttpBubble.setCircuitBreakerState(this.params.url, state);
        }
    }
    handleCircuitBreakerFailure() {
        if (!this.params.circuitBreakerEnabled) {
            return;
        }
        const state = HttpBubble.getCircuitBreakerState(this.params.url);
        const now = Date.now();
        state.failureCount++;
        state.lastFailureTime = now;
        if (state.failureCount >= this.params.circuitBreakerThreshold) {
            state.isOpen = true;
            state.nextAttemptTime = now + this.params.circuitBreakerTimeout;
            console.warn(`[HttpBubble] Circuit breaker opened for ${this.params.url} (failures: ${state.failureCount})`);
        }
        HttpBubble.setCircuitBreakerState(this.params.url, state);
    }
    createCircuitBreakerResult() {
        const state = HttpBubble.getCircuitBreakerState(this.params.url);
        const timeUntilRetry = Math.max(0, state.nextAttemptTime - Date.now());
        return {
            success: false,
            data: undefined,
            status: 503,
            statusText: 'Service Unavailable',
            headers: {},
            body: '',
            error: `Circuit breaker is open. Retry after ${timeUntilRetry}ms`,
            errorCode: 'CIRCUIT_BREAKER_OPEN',
            metrics: {
                totalAttempts: 1,
                responseTime: 0,
                lastAttemptTime: 0,
                retryCount: 0,
                circuitBreakerTripped: true,
            },
            request: {
                url: this.params.url,
                method: this.params.method || this.getOperationMethod(),
            },
        };
    }
    createErrorResult(error, attempt, responseTime) {
        return {
            success: false,
            data: undefined,
            status: 0,
            statusText: 'Request Failed',
            headers: {},
            body: '',
            error: error.message,
            errorCode: error.name,
            metrics: {
                totalAttempts: attempt,
                responseTime,
                lastAttemptTime: responseTime,
                retryCount: attempt - 1,
            },
            request: {
                url: this.params.url,
                method: this.params.method || this.getOperationMethod(),
            },
        };
    }
}
//# sourceMappingURL=http-bubble.js.map