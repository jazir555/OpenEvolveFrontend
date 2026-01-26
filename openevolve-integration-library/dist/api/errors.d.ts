export declare class IntegrationError extends Error {
    integration: string;
    code: string;
    details?: any | undefined;
    constructor(integration: string, code: string, message: string, details?: any | undefined);
    readonly timestamp: string;
    getRetryAfterMs(): number | undefined;
    toJSON(): {
        name: string;
        integration: string;
        code: string;
        message: string;
        details: any;
        timestamp: string;
    };
    static fromJSON(obj: any): IntegrationError;
}
export declare class ConnectionError extends IntegrationError {
    constructor(integration: string, message: string, details?: any);
}
export declare class AuthenticationError extends IntegrationError {
    constructor(integration: string, message?: string, details?: any);
}
export declare class AuthorizationError extends IntegrationError {
    constructor(integration: string, message?: string, details?: any);
}
export declare class ValidationError extends IntegrationError {
    errors: Array<{
        field: string;
        message: string;
        code: string;
    }>;
    constructor(integration: string, errors: Array<{
        field: string;
        message: string;
        code: string;
    }>, details?: any);
    getErrorMessages(): string[];
    getErrorForField(field: string): string | undefined;
}
export declare class ExecutionError extends IntegrationError {
    constructor(integration: string, message: string, details?: any);
}
export declare class TimeoutError extends IntegrationError {
    constructor(integration: string, timeout: number, details?: any);
}
export declare class RateLimitError extends IntegrationError {
    constructor(integration: string, retryAfter: number, details?: any);
    getRetryAfterMs(): number;
}
export declare class NotFoundError extends IntegrationError {
    constructor(integration: string, resource: string, identifier: string, details?: any);
}
export declare class ConfigurationError extends IntegrationError {
    constructor(integration: string, message: string, details?: any);
}
export declare class NetworkError extends IntegrationError {
    constructor(integration: string, message: string, details?: any);
}
export declare class CancellationError extends IntegrationError {
    constructor(integration: string, executionId: string, details?: any);
}
export declare class ParseError extends IntegrationError {
    constructor(integration: string, message: string, details?: any);
}
export declare class RetryError extends IntegrationError {
    constructor(integration: string, originalError: Error, attempts: number, details?: any);
}
export declare class CircuitBreakerError extends IntegrationError {
    constructor(integration: string, details?: any);
}
export declare class ServiceUnavailableError extends IntegrationError {
    constructor(integration: string, message?: string, details?: any);
}
export declare class UnknownError extends IntegrationError {
    constructor(integration: string, message: string, details?: any);
}
export declare function isIntegrationError(error: any): error is IntegrationError;
export declare function isRetryableError(error: any): boolean;
export declare function isCriticalError(error: any): boolean;
export declare function createIntegrationError(integration: string, error: any): IntegrationError;
//# sourceMappingURL=errors.d.ts.map