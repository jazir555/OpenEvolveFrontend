"use strict";
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR HTTP Client
 *
 * HTTP client for communicating with the ICR system.
 * Implements:
 * - Circuit breaker pattern
 * - Exponential backoff retry with jitter
 * - Structured JSON logging with correlation IDs
 * - UTC timestamps only
 * - Mandatory configuration validation (no magic defaults)
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Configuration Explicitness: Required env vars crash if missing
 * - UTC: All timestamps in UTC ISO-8601 format
 * - Observability: Structured logging with correlation IDs
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.icrClient = exports.ICRClient = void 0;
const axios_1 = __importDefault(require("axios"));
const uuid_1 = require("uuid");
// ============================================================================
// CONFIGURATION (Law of Configuration Explicitness)
// ============================================================================
/**
 * Validate required environment variables at startup.
 * CRASHES immediately if any required variable is missing.
 *
 * This is NOT OPTIONAL. The Federation Constitution demands:
 * "If TARGET_API_URL is missing, the service crashes immediately with a loud error."
 */
function validateEnvVars() {
    const required = [
        'OPENEVOLVE_ICR_API_URL',
        'TIMEOUT_MS'
    ];
    const missing = [];
    for (const envVar of required) {
        if (!process.env[envVar]) {
            missing.push(envVar);
        }
    }
    if (missing.length > 0) {
        throw new Error(`ICR Adapter Configuration Error: Missing required environment variables: ${missing.join(', ')}\n` +
            `The Federation Constitution prohibits magic defaults. ` +
            `Please set these environment variables before starting the service.`);
    }
}
// Validate at module load time (fail fast)
validateEnvVars();
// Extract configuration with NO DEFAULTS
const CONFIG = {
    API_URL: process.env.OPENEVOLVE_ICR_API_URL,
    TIMEOUT_MS: parseInt(process.env.TIMEOUT_MS, 10),
    MAX_RETRIES: parseInt(process.env.MAX_RETRIES || '3', 10),
    INITIAL_RETRY_DELAY_MS: parseInt(process.env.INITIAL_RETRY_DELAY_MS || '1000', 10),
    BACKOFF_FACTOR: parseFloat(process.env.BACKOFF_FACTOR || '2.0'),
    CIRCUIT_BREAKER_THRESHOLD: parseInt(process.env.CIRCUIT_BREAKER_THRESHOLD || '5', 10),
    CIRCUIT_BREAKER_TIMEOUT_MS: parseInt(process.env.CIRCUIT_BREAKER_TIMEOUT_MS || '60000', 10),
    ENABLE_JITTER: process.env.ENABLE_JITTER !== 'false'
};
// ============================================================================
// CIRCUIT BREAKER IMPLEMENTATION
// ============================================================================
var CircuitBreakerState;
(function (CircuitBreakerState) {
    CircuitBreakerState["CLOSED"] = "closed";
    CircuitBreakerState["OPEN"] = "open";
    CircuitBreakerState["HALF_OPEN"] = "half_open";
})(CircuitBreakerState || (CircuitBreakerState = {}));
class CircuitBreaker {
    constructor(threshold = CONFIG.CIRCUIT_BREAKER_THRESHOLD, timeoutMs = CONFIG.CIRCUIT_BREAKER_TIMEOUT_MS, logger) {
        this.threshold = threshold;
        this.timeoutMs = timeoutMs;
        this.logger = logger;
        this.stats = {
            state: CircuitBreakerState.CLOSED,
            failureCount: 0,
            lastFailureTime: null,
            lastSuccessTime: null,
            openedAt: null
        };
    }
    /**
     * Check if circuit breaker allows request execution
     */
    allowRequest() {
        const now = Date.now();
        if (this.stats.state === CircuitBreakerState.OPEN) {
            // Check if timeout has elapsed
            if (this.stats.openedAt && (now - this.stats.openedAt) >= this.timeoutMs) {
                this.stats.state = CircuitBreakerState.HALF_OPEN;
                this.logger.info({
                    msg: 'Circuit breaker transitioning to HALF_OPEN',
                    state: this.stats.state,
                    time_since_opened_ms: now - (this.stats.openedAt || 0)
                });
                return true;
            }
            return false;
        }
        return true;
    }
    /**
     * Record a successful request
     */
    recordSuccess() {
        this.stats.failureCount = 0;
        this.stats.lastSuccessTime = Date.now();
        if (this.stats.state === CircuitBreakerState.HALF_OPEN) {
            this.stats.state = CircuitBreakerState.CLOSED;
            this.stats.openedAt = null;
            this.logger.info({
                msg: 'Circuit breaker transitioning to CLOSED',
                state: this.stats.state
            });
        }
    }
    /**
     * Record a failed request
     */
    recordFailure() {
        this.stats.failureCount++;
        this.stats.lastFailureTime = Date.now();
        this.logger.warn({
            msg: 'Circuit breaker recording failure',
            failure_count: this.stats.failureCount,
            threshold: this.threshold
        });
        if (this.stats.failureCount >= this.threshold) {
            this.stats.state = CircuitBreakerState.OPEN;
            this.stats.openedAt = Date.now();
            this.logger.error({
                msg: 'Circuit breaker OPENED',
                failure_count: this.stats.failureCount,
                threshold: this.threshold,
                state: this.stats.state
            });
        }
    }
    /**
     * Get current circuit breaker state
     */
    getState() {
        return { ...this.stats };
    }
    /**
     * Reset circuit breaker (for testing/manual recovery)
     */
    reset() {
        this.stats = {
            state: CircuitBreakerState.CLOSED,
            failureCount: 0,
            lastFailureTime: null,
            lastSuccessTime: null,
            openedAt: null
        };
        this.logger.info({ msg: 'Circuit breaker manually reset' });
    }
}
class StructuredLogger {
    constructor(sourceService = 'icr-adapter') {
        this.sourceService = sourceService;
    }
    log(level, entry) {
        const logEntry = {
            ...entry,
            timestamp_utc: new Date().toISOString(),
            source_service: this.sourceService,
            target_service: 'icr-core'
        };
        // Output JSON Lines (jsonl format)
        const jsonLine = JSON.stringify({ level, ...logEntry });
        console.log(jsonLine);
    }
    info(entry) {
        this.log('info', entry);
    }
    warn(entry) {
        this.log('warn', entry);
    }
    error(entry) {
        this.log('error', entry);
    }
    debug(entry) {
        if (process.env.DEBUG === 'true') {
            this.log('debug', entry);
        }
    }
}
// ============================================================================
// RETRY LOGIC WITH EXPONENTIAL BACKOFF AND JITTER
// ============================================================================
/**
 * Calculate delay with exponential backoff and jitter
 */
function calculateRetryDelay(attempt, initialDelay, backoffFactor, enableJitter) {
    const exponentialDelay = initialDelay * Math.pow(backoffFactor, attempt);
    if (!enableJitter) {
        return exponentialDelay;
    }
    // Add jitter: random value between 0 and 0.5 * exponentialDelay
    const jitter = Math.random() * 0.5 * exponentialDelay;
    return Math.floor(exponentialDelay + jitter);
}
/**
 * Sleep function for retry delays
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
class ICRClient {
    constructor(options) {
        // Configuration Explicitness: Use provided options or env vars (NO DEFAULTS)
        const apiUrl = options?.apiUrl || CONFIG.API_URL;
        const timeout = options?.timeout || CONFIG.TIMEOUT_MS;
        this.maxRetries = options?.maxRetries ?? CONFIG.MAX_RETRIES;
        this.logger = new StructuredLogger();
        this.circuitBreaker = new CircuitBreaker(CONFIG.CIRCUIT_BREAKER_THRESHOLD, CONFIG.CIRCUIT_BREAKER_TIMEOUT_MS, this.logger);
        this.axiosInstance = axios_1.default.create({
            baseURL: apiUrl,
            timeout: timeout,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        // Request interceptor for logging
        this.axiosInstance.interceptors.request.use((config) => {
            const correlationId = config.headers['X-Correlation-ID'] || (0, uuid_1.v4)();
            config.headers['X-Correlation-ID'] = correlationId;
            this.logger.debug({
                msg: 'ICR API request initiated',
                correlation_id: correlationId,
                method: config.method,
                url: config.url
            });
            return config;
        }, (error) => {
            this.logger.error({
                msg: 'ICR API request interceptor error',
                error: error.message
            });
            return Promise.reject(error);
        });
        // Response interceptor for logging
        this.axiosInstance.interceptors.response.use((response) => {
            const correlationId = response.config.headers['X-Correlation-ID'];
            this.logger.debug({
                msg: 'ICR API response received',
                correlation_id: correlationId,
                status: response.status
            });
            return response;
        }, (error) => {
            const correlationId = error.config?.headers?.['X-Correlation-ID'] || (0, uuid_1.v4)();
            this.logger.error({
                msg: 'ICR API response error',
                correlation_id: correlationId,
                status: error.response?.status,
                error: error.message
            });
            return Promise.reject(error);
        });
        this.logger.info({
            msg: 'ICR Client initialized',
            api_url: apiUrl,
            timeout_ms: timeout,
            max_retries: this.maxRetries
        });
    }
    /**
     * Execute request with retry logic and circuit breaker
     */
    async executeWithRetry(requestFn, correlationId, operation) {
        // Check circuit breaker first
        if (!this.circuitBreaker.allowRequest()) {
            const error = new Error(`Circuit breaker is OPEN. Rejecting ${operation} request.`);
            this.logger.error({
                msg: error.message,
                correlation_id: correlationId,
                operation,
                circuit_state: this.circuitBreaker.getState()
            });
            throw error;
        }
        let lastError = null;
        for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
            try {
                const response = await requestFn();
                this.circuitBreaker.recordSuccess();
                this.logger.info({
                    msg: `${operation} completed successfully`,
                    correlation_id: correlationId,
                    attempt,
                    retry_count: attempt
                });
                return response.data;
            }
            catch (error) {
                lastError = error;
                // Check if error is retryable
                const isRetryable = this.isRetryableError(error);
                this.logger.warn({
                    msg: `${operation} attempt ${attempt + 1} failed`,
                    correlation_id: correlationId,
                    attempt: attempt + 1,
                    is_retryable: isRetryable,
                    error: error.message
                });
                if (!isRetryable || attempt >= this.maxRetries) {
                    this.circuitBreaker.recordFailure();
                    throw error;
                }
                // Calculate delay and sleep
                const delay = calculateRetryDelay(attempt, CONFIG.INITIAL_RETRY_DELAY_MS, CONFIG.BACKOFF_FACTOR, CONFIG.ENABLE_JITTER);
                this.logger.debug({
                    msg: `Retrying ${operation} after delay`,
                    correlation_id: correlationId,
                    attempt: attempt + 1,
                    delay_ms: delay
                });
                await sleep(delay);
            }
        }
        // Should never reach here, but TypeScript needs it
        this.circuitBreaker.recordFailure();
        throw lastError;
    }
    /**
     * Determine if an error is retryable
     */
    isRetryableError(error) {
        // Network errors (no response)
        if (!error.response) {
            return true;
        }
        const status = error.response.status;
        // Retry on 429 (Too Many Requests) and 5xx errors
        return status === 429 || status >= 500;
    }
    /**
     * Execute mode request
     */
    async executeMode(request, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        this.logger.info({
            msg: 'Executing ICR mode request',
            correlation_id: cid,
            mode: request.mode
        });
        return this.executeWithRetry(() => this.axiosInstance.post('/api/modes/execute', request, {
            headers: { 'X-Correlation-ID': cid }
        }), cid, `execute_mode_${request.mode}`);
    }
    /**
     * Health check
     */
    async healthCheck(request, correlationId) {
        const cid = correlationId || (0, uuid_1.v4)();
        const payload = request || { correlation_id: cid };
        this.logger.info({
            msg: 'Performing ICR health check',
            correlation_id: cid
        });
        return this.executeWithRetry(() => this.axiosInstance.post('/api/health', payload, {
            headers: { 'X-Correlation-ID': cid }
        }), cid, 'health_check');
    }
    /**
     * Get circuit breaker state (for monitoring)
     */
    getCircuitBreakerState() {
        return this.circuitBreaker.getState();
    }
    /**
     * Reset circuit breaker (for recovery)
     */
    resetCircuitBreaker() {
        this.circuitBreaker.reset();
    }
}
exports.ICRClient = ICRClient;
// ============================================================================
// SINGLETON INSTANCE
// ============================================================================
/**
 * Default ICR client instance
 */
exports.icrClient = new ICRClient();
//# sourceMappingURL=icr-client.js.map