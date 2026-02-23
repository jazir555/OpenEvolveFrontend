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
import { ICRModeRequest, ICRModeResponse, ICRHealthCheckRequest, ICRHealthCheckResponse } from './icr-canonical';
declare enum CircuitBreakerState {
    CLOSED = "closed",
    OPEN = "open",
    HALF_OPEN = "half_open"
}
interface CircuitBreakerStats {
    state: CircuitBreakerState;
    failureCount: number;
    lastFailureTime: number | null;
    lastSuccessTime: number | null;
    openedAt: number | null;
}
export interface ICRClientOptions {
    apiUrl?: string;
    timeout?: number;
    maxRetries?: number;
}
export declare class ICRClient {
    private readonly axiosInstance;
    private readonly circuitBreaker;
    private readonly logger;
    private readonly maxRetries;
    constructor(options?: ICRClientOptions);
    /**
     * Execute request with retry logic and circuit breaker
     */
    private executeWithRetry;
    /**
     * Determine if an error is retryable
     */
    private isRetryableError;
    /**
     * Execute mode request
     */
    executeMode(request: ICRModeRequest, correlationId?: string): Promise<ICRModeResponse>;
    /**
     * Health check
     */
    healthCheck(request?: ICRHealthCheckRequest, correlationId?: string): Promise<ICRHealthCheckResponse>;
    /**
     * Get circuit breaker state (for monitoring)
     */
    getCircuitBreakerState(): CircuitBreakerStats;
    /**
     * Reset circuit breaker (for recovery)
     */
    resetCircuitBreaker(): void;
}
/**
 * Default ICR client instance
 */
export declare const icrClient: ICRClient;
export {};
//# sourceMappingURL=icr-client.d.ts.map