import { ExecutionConfig } from './common';
export interface MakerInputs {
    operation: 'create' | 'execute' | 'validate' | 'list';
    input: ToolConfig | ToolExecutionInput | ValidationInput;
    config?: ExecutionConfig;
}
export interface ToolConfig {
    name: string;
    description: string;
    type: 'function' | 'api' | 'workflow' | 'custom';
    inputSchema: Record<string, any>;
    outputSchema?: Record<string, any>;
    implementation: ToolImplementation;
    config?: Record<string, any>;
}
export interface ToolImplementation {
    type: 'python' | 'javascript' | 'api' | 'workflow' | 'composite';
    code?: string;
    endpoint?: string;
    workflow?: any;
}
export interface ToolExecutionInput {
    toolId: string;
    parameters: Record<string, any>;
    options?: {
        timeout?: number;
        async?: boolean;
        callbackUrl?: string;
    };
}
export interface ValidationInput {
    toolId: string;
    validationType: 'syntax' | 'semantic' | 'execution' | 'all';
    testInputs?: Record<string, any>[];
}
export interface ToolResult {
    toolId: string;
    name: string;
    status: 'created' | 'updated' | 'failed';
    validation: ValidationResult;
    metadata: ToolMetadata;
}
export interface ValidationResult {
    valid: boolean;
    checks: ValidationCheck[];
    score: number;
    issues: ValidationIssue[];
}
export interface ValidationCheck {
    name: string;
    status: 'passed' | 'failed' | 'skipped';
    message: string;
    executionTime: number;
}
export interface ValidationIssue {
    severity: 'error' | 'warning' | 'info';
    type: string;
    message: string;
    location?: string;
    suggestion?: string;
}
export interface ExecutionResult {
    executionId: string;
    toolId: string;
    status: 'success' | 'error' | 'timeout';
    result?: any;
    error?: string;
    metadata: ExecutionMetadata;
}
export interface ExecutionMetadata {
    executionTime: number;
    memoryUsage?: number;
    cpuUsage?: number;
    timestamp: string;
    environment: string;
}
export interface ToolMetadata {
    version: string;
    created: string;
    modified: string;
    createdBy: string;
    tags?: string[];
    category?: string;
}
export interface ToolListResult {
    tools: ToolInfo[];
    total: number;
}
export interface ToolInfo {
    toolId: string;
    name: string;
    description: string;
    type: string;
    status: 'active' | 'inactive' | 'deprecated';
    version: string;
    lastExecuted?: string;
    executionCount: number;
}
export interface MakerResult {
    type: 'create' | 'execute' | 'validate' | 'list';
    result: ToolResult | ExecutionResult | ValidationResult | ToolListResult;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=maker.d.ts.map