import type { BackendClient } from '../api/backend';
import type { IntegrationAdapter, ValidationResult, ParameterSchema, ProgressUpdate, ExecutionOptions, IntegrationHealth, RetryConfig, CircuitBreakerConfig, CircuitState } from '../api/types';
import { IntegrationError } from '../api/errors';
export declare abstract class BaseIntegrationAdapter implements IntegrationAdapter {
    protected client: BackendClient;
    name: string;
    protected version: string;
    protected description: string;
    protected retryConfig?: Partial<RetryConfig>;
    private circuitState;
    private failureCount;
    private successCount;
    private lastFailureTime;
    protected circuitBreakerConfig: CircuitBreakerConfig;
    protected onGlobalError?: (error: IntegrationError) => void;
    constructor(client: BackendClient, name: string, version: string, description: string, retryConfig?: Partial<RetryConfig>, circuitBreakerConfig?: Partial<CircuitBreakerConfig>);
    setGlobalErrorHandler(handler: (error: IntegrationError) => void): void;
    protected checkCircuit(): void;
    protected recordSuccess(): void;
    protected recordFailure(error: any): void;
    getCircuitState(): CircuitState;
    getName(): string;
    getVersion(): string;
    getDescription(): string;
    abstract execute<TInputs, TResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    validate<TInputs>(inputs: TInputs): Promise<ValidationResult>;
    abstract getSchema(): ParameterSchema;
    executeStream<TInputs, TResult>(inputs: TInputs, _onProgress: (update: ProgressUpdate) => void, options?: ExecutionOptions): Promise<TResult>;
    healthCheck(): Promise<IntegrationHealth>;
    protected abstract getEndpoints(): string[];
    protected requestBackend<TResponse>(method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH', endpoint: string, data?: any, options?: ExecutionOptions): Promise<TResponse>;
    protected executeBackend<TRequest, TResponse>(endpoint: string, request: TRequest, executionId?: string, options?: ExecutionOptions): Promise<TResponse>;
    protected streamExecute<TRequest, TResponse>(endpoint: string, request: TRequest, onProgress: (update: ProgressUpdate) => void, options?: ExecutionOptions): Promise<TResponse>;
    protected handleError(error: any): never;
    protected transformRequest<T>(data: T): any;
    protected transformResponse<T>(data: any): T;
    protected validateResponse<T>(_data: T): {
        valid: boolean;
        errors?: string[];
    };
    protected validateRequired<T>(inputs: T, requiredFields: (keyof T)[]): string[];
    protected validateTypes<T>(inputs: T, typeDefinitions: Partial<Record<keyof T, string>>): string[];
    protected validateEnum<T>(inputs: T, enumDefinitions: Partial<Record<keyof T, any[]>>): string[];
    protected validateRanges<T>(inputs: T, rangeDefinitions: Partial<Record<keyof T, {
        min?: number;
        max?: number;
    }>>): string[];
}
//# sourceMappingURL=base.d.ts.map