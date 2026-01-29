import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Hephaestus task types
 */
export type HephaestusTaskType = 'generate' | 'execute' | 'delegate' | 'optimize';
/**
 * Code language types
 */
export type CodeLanguage = 'python' | 'javascript' | 'typescript' | 'java' | 'cpp' | 'go' | 'rust';
/**
 * Hephaestus node configuration
 */
export interface HephaestusNodeConfig {
    taskType?: HephaestusTaskType;
    language?: CodeLanguage;
    enableExecution?: boolean;
    enableOptimization?: boolean;
    timeoutMs?: number;
}
/**
 * Code generation result
 */
export interface CodeGenerationResult {
    code: string;
    language: CodeLanguage;
    quality: number;
    dependencies: string[];
    documentation: string;
    tests?: string;
}
/**
 * Code execution result
 */
export interface CodeExecutionResult {
    success: boolean;
    output: string;
    error?: string;
    executionTime: number;
    memory: number;
    cpu: number;
}
/**
 * Delegation result
 */
export interface DelegationResult {
    delegatedTo: string;
    taskId: string;
    status: string;
    result?: any;
    metadata: {
        delegatedAt: Date;
        completedAt?: Date;
        executionTime: number;
    };
}
/**
 * Optimization result
 */
export interface OptimizationResult {
    originalCode: string;
    optimizedCode: string;
    improvements: Array<{
        type: string;
        description: string;
        impact: string;
    }>;
    performance: {
        speedup: number;
        memoryReduction: number;
        timeComplexity: string;
        spaceComplexity: string;
    };
}
/**
 * Hephaestus result
 */
export interface HephaestusResult {
    taskId: string;
    taskType: HephaestusTaskType;
    language: CodeLanguage;
    input: string;
    output: CodeGenerationResult | CodeExecutionResult | DelegationResult | OptimizationResult;
    metadata: {
        executedAt: Date;
        executionTime: number;
        parameters: {
            language: CodeLanguage;
            enableExecution: boolean;
            enableOptimization: boolean;
        };
    };
}
/**
 * Hephaestus Node
 *
 * Bridges to Hephaestus for code generation and execution.
 * Supports delegation, optimization, and cross-service integration.
 */
export declare class HephaestusNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Hephaestus Bridge";
    static readonly DESCRIPTION = "Code generation and execution bridge with delegation and optimization capabilities";
    static readonly ICON = "hephaestus";
    static readonly CATEGORY = "integration";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: HephaestusNodeConfig);
    /**
     * Execute Hephaestus task
     *
     * @param inputs - Must contain 'input' and optionally 'taskType'
     * @param context - Execution context
     * @returns Promise resolving to Hephaestus result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Generate code from natural language
     *
     * @param input - Natural language description
     * @param language - Target programming language
     * @param context - Execution context
     * @returns Promise resolving to generated code
     */
    private generateCode;
    /**
     * Execute code
     *
     * @param code - Code to execute
     * @param language - Programming language
     * @param context - Execution context
     * @returns Promise resolving to execution result
     */
    private executeCode;
    /**
     * Delegate task to another service
     *
     * @param input - Task input
     * @param delegateTo - Target service
     * @param context - Execution context
     * @returns Promise resolving to delegation result
     */
    private delegateTask;
    /**
     * Optimize code
     *
     * @param code - Code to optimize
     * @param language - Programming language
     * @param context - Execution context
     * @returns Promise resolving to optimization result
     */
    private optimizeCode;
    /**
     * Monitor delegation progress
     *
     * @param taskId - Task ID to monitor
     * @param context - Execution context
     * @returns Promise resolving to delegation status
     */
    private monitorDelegation;
    /**
     * Validate input data
     *
     * @param inputs - Input data to validate
     * @returns Array of validation errors
     */
    validateInputs(inputs: NodeInputs): ValidationError[];
    /**
     * Get JSON Schema for configuration parameters
     *
     * @returns Parameter schema
     */
    getParameterSchema(): ParameterSchema;
    /**
     * Get supported languages
     *
     * @returns Array of supported languages
     */
    getSupportedLanguages(): CodeLanguage[];
    /**
     * Get available services for delegation
     *
     * @returns Promise resolving to available services
     */
    getAvailableServices(): Promise<NodeResult>;
    /**
     * Get code quality metrics
     *
     * @param code - Code to analyze
     * @param language - Programming language
     * @returns Promise resolving to quality metrics
     */
    getQualityMetrics(code: string, language: CodeLanguage): Promise<NodeResult>;
}
export default HephaestusNode;
