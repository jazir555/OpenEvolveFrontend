/**
 * Bubble Error Handler Mixin
 *
 * Provides reusable error handling methods for all bubble types
 * Integrates with the error-handler utilities for consistent error responses
 */
import { CircuitBreaker, type ErrorResponse, type SuccessResponse } from './error-handler.js';
/**
 * Configuration for bubble error handling
 */
export interface BubbleErrorHandlerConfig {
    timeout?: number;
    retryAttempts?: number;
    retryDelay?: number;
    enableCircuitBreaker?: boolean;
    circuitBreakerThreshold?: number;
}
/**
 * Default configuration
 */
export declare const defaultBubbleErrorHandlerConfig: BubbleErrorHandlerConfig;
/**
 * Mixin class for adding error handling to bubbles
 */
export declare class BubbleErrorHandler {
    protected circuitBreaker?: CircuitBreaker;
    protected errorHandlerConfig: BubbleErrorHandlerConfig;
    constructor(config?: BubbleErrorHandlerConfig);
    /**
     * Wrap an operation with comprehensive error handling
     */
    protected executeWithErrorHandling<T>(operation: () => Promise<T>, operationName: string, context?: {
        timeout?: number;
        retry?: boolean;
        correlationId?: string;
    }): Promise<SuccessResponse<T> | ErrorResponse>;
    /**
     * Reset the circuit breaker (useful for testing or manual recovery)
     */
    protected resetCircuitBreaker(): void;
    /**
     * Get circuit breaker metrics (useful for monitoring)
     */
    protected getCircuitBreakerMetrics(): {
        enabled: boolean;
    } | {
        state: import("./error-handler.js").CircuitState;
        failureCount: number;
        successCount: number;
        lastFailureTime: number | undefined;
        nextAttemptTime: number | undefined;
        enabled: boolean;
    };
}
/**
 * Type guard to check if a response is an error response
 */
export declare function isErrorResponse<T>(response: SuccessResponse<T> | ErrorResponse): response is ErrorResponse;
/**
 * Type guard to check if a response is a success response
 */
export declare function isSuccessResponse<T>(response: SuccessResponse<T> | ErrorResponse): response is SuccessResponse<T>;
/**
 * Extract data from a success response, throwing if it's an error
 */
export declare function getDataOrThrow<T>(response: SuccessResponse<T> | ErrorResponse): T;
/**
 * Extract error from an error response, throwing if it's a success
 */
export declare function getErrorOrThrow<T>(response: SuccessResponse<T> | ErrorResponse): ErrorResponse;
//# sourceMappingURL=bubble-error-handler.d.ts.map