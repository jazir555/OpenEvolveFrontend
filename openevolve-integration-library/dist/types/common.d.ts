export interface ValidationResult {
    valid: boolean;
    errors?: string[];
    warnings?: string[];
}
export interface ParameterSchema {
    name: string;
    version: string;
    parameters: Record<string, ParameterDefinition>;
    required: string[];
}
export interface ParameterDefinition {
    type: 'string' | 'number' | 'boolean' | 'object' | 'array' | 'enum';
    description: string;
    required: boolean;
    default?: any;
    enum?: any[];
    min?: number;
    max?: number;
    schema?: any;
}
export interface ProgressUpdate {
    progress: number;
    message: string;
    currentStep?: string;
    totalSteps?: number;
    stepNumber?: number;
    metadata?: Record<string, any>;
}
export interface ExecutionResult<T = any> {
    success: boolean;
    data?: T;
    error?: string;
    executionTime?: number;
    metadata?: Record<string, any>;
}
export interface APIResponse<T = any> {
    status: number;
    data: T;
    message?: string;
    metadata?: Record<string, any>;
}
export interface ExecutionConfig {
    stream?: boolean;
    onProgress?: (update: ProgressUpdate) => void;
    timeout?: number;
    retries?: number;
    options?: Record<string, any>;
}
export declare class IntegrationError extends Error {
    code: string;
    details?: any | undefined;
    constructor(message: string, code: string, details?: any | undefined);
}
export declare class ValidationError extends IntegrationError {
    errors: string[];
    constructor(message: string, errors: string[]);
}
export declare class ExecutionError extends IntegrationError {
    executionId?: string | undefined;
    constructor(message: string, executionId?: string | undefined);
}
export declare class TimeoutError extends IntegrationError {
    timeout: number;
    constructor(message: string, timeout: number);
}
//# sourceMappingURL=common.d.ts.map