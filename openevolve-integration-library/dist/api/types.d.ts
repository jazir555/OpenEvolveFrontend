import { IntegrationError } from './errors';
export type Middleware = (context: {
    integration: string;
    inputs: any;
    options?: ExecutionOptions;
    executionId: string;
}, next: () => Promise<any>) => Promise<any>;
export interface CircuitBreakerConfig {
    enabled: boolean;
    failureThreshold: number;
    resetTimeout: number;
    successThreshold: number;
}
export type CircuitState = 'closed' | 'open' | 'half-open';
export interface ClientConfig {
    baseUrl: string;
    timeout?: number;
    retryAttempts?: number;
    retryConfig?: Partial<RetryConfig>;
    circuitBreakerConfig?: Partial<CircuitBreakerConfig>;
    healthCheckInterval?: number;
    enableWebSocket?: boolean;
    debug?: boolean;
    apiKey?: string;
    headers?: Record<string, string>;
    middleware?: Middleware[];
    requestTransform?: RequestTransform;
    responseTransform?: ResponseTransform;
    onError?: (error: IntegrationError) => void;
}
export interface ExecutionOptions {
    executionId?: string;
    timeout?: number;
    retries?: number;
    retryConfig?: Partial<RetryConfig>;
    stream?: boolean;
    onProgress?: (update: ProgressUpdate) => void;
    onComplete?: (result: any) => void;
    onError?: (error: IntegrationError) => void;
    onRetry?: (error: IntegrationError, attempt: number, delay: number) => void;
    fallback?: any;
    signal?: AbortSignal;
    metadata?: Record<string, any>;
    bypassCache?: boolean;
}
export interface ProgressUpdate {
    integration: string;
    executionId: string;
    progress: number;
    message: string;
    timestamp: string;
    data?: any;
    stage?: string;
    eta?: number;
}
export interface BatchRequest<TInputs> {
    integration: string;
    id: string;
    inputs: TInputs;
    options?: ExecutionOptions;
}
export interface BatchResult<TResult> {
    id: string;
    result: TResult | null;
    error: IntegrationError | null;
    executionTime: number;
    success: boolean;
}
export interface HealthStatus {
    status: 'healthy' | 'degraded' | 'unhealthy';
    backend: BackendStatus;
    integrations: Record<string, IntegrationHealth>;
    timestamp: string;
}
export interface BackendStatus {
    online: boolean;
    version: string;
    uptime: number;
    activeConnections: number;
    memory: {
        used: number;
        total: number;
        percentage: number;
    };
    cpu: number;
}
export interface IntegrationHealth {
    name: string;
    status: 'available' | 'unavailable' | 'degraded';
    responseTime: number;
    lastError?: string;
    endpoints: string[];
}
export interface WebSocketMessage {
    type: 'progress' | 'complete' | 'error' | 'status';
    integration?: string;
    executionId?: string;
    data: any;
    timestamp: string;
}
export type RequestTransform = (data: any) => any;
export type ResponseTransform = (data: any) => any;
export type ErrorTransform = (error: any) => IntegrationError;
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'disconnecting';
export interface IntegrationAdapter {
    name: string;
    getVersion(): string;
    execute<TInputs, TResult>(inputs: TInputs, options?: ExecutionOptions): Promise<TResult>;
    executeStream<TInputs, TResult>(inputs: TInputs, onProgress: (update: ProgressUpdate) => void, options?: ExecutionOptions): Promise<TResult>;
    healthCheck(): Promise<IntegrationHealth>;
    validate<TInputs>(inputs: TInputs): Promise<ValidationResult>;
}
export interface ValidationResult {
    valid: boolean;
    errors: ValidationErrorItem[];
    warnings: ValidationWarning[];
}
export interface ValidationErrorItem {
    field: string;
    message: string;
    code: string;
    name?: string;
}
export interface ParameterSchema {
    type: string;
    properties?: Record<string, any>;
    required?: string[];
    description?: string;
    enum?: any[];
    [key: string]: any;
}
export interface ValidationWarning {
    field: string;
    message: string;
    code: string;
}
export interface RetryConfig {
    maxAttempts: number;
    initialDelay: number;
    maxDelay: number;
    backoffMultiplier: number;
    retryOn4xx: boolean;
    retryOn5xx: boolean;
    retryableStatusCodes: number[];
}
export interface RequestMetrics {
    requestId: string;
    integration: string;
    startTime: string;
    endTime: string;
    duration: number;
    retries: number;
    success: boolean;
    statusCode?: number;
    error?: string;
}
export interface ApiResponse<T> {
    data: T;
    meta: ResponseMeta;
}
export interface ResponseMeta {
    requestId: string;
    timestamp: string;
    executionTime: number;
    cached: boolean;
    rateLimit?: {
        limit: number;
        remaining: number;
        reset: string;
    };
}
//# sourceMappingURL=types.d.ts.map