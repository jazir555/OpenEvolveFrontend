import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Proof task types
 */
export type ProofTaskType = 'generate' | 'verify' | 'repair' | 'synthesis';
/**
 * LeanAide node configuration
 */
export interface LeanAideNodeConfig {
    taskType?: ProofTaskType;
    model?: string;
    temperature?: number;
    enableAutoRepair?: boolean;
    maxRepairAttempts?: number;
}
/**
 * Lean code output
 */
export interface LeanCodeOutput {
    code: string;
    theorem: string;
    proof: string;
    tactics: string[];
    verified: boolean;
    errors: string[];
    warnings: string[];
}
/**
 * Verification result
 */
export interface VerificationResult {
    verified: boolean;
    errors: Array<{
        line: number;
        column: number;
        message: string;
        severity: 'error' | 'warning';
    }>;
    warnings: string[];
    tacticsUsed: string[];
    proofSteps: number;
}
/**
 * Proof repair result
 */
export interface ProofRepairResult {
    originalCode: string;
    repairedCode: string;
    repairs: Array<{
        line: number;
        issue: string;
        fix: string;
    }>;
    verified: boolean;
    attempts: number;
}
/**
 * LeanAide result
 */
export interface LeanAideResult {
    taskId: string;
    taskType: ProofTaskType;
    theorem: string;
    output: LeanCodeOutput | VerificationResult | ProofRepairResult;
    model: string;
    metadata: {
        executedAt: Date;
        executionTime: number;
        temperature: number;
        parameters: {
            model: string;
            temperature: number;
            enableAutoRepair: boolean;
        };
    };
}
/**
 * LeanAide Node
 *
 * Generates and verifies Lean 4 formal proofs.
 * Provides automated proof repair and synthesis capabilities.
 */
export declare class LeanAideNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "LeanAide Verification";
    static readonly DESCRIPTION = "Lean 4 formal proof generation and verification with automated repair";
    static readonly ICON = "leanaide";
    static readonly CATEGORY = "verification";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: LeanAideNodeConfig);
    /**
     * Execute LeanAide task
     *
     * @param inputs - Must contain 'theorem' and optionally 'proof_attempt'
     * @param context - Execution context
     * @returns Promise resolving to LeanAide result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Generate Lean 4 proof
     *
     * @param theorem - Theorem statement
     * @param model - Model to use
     * @param temperature - Temperature for generation
     * @param context - Execution context
     * @returns Promise resolving to generated proof
     */
    private generateProof;
    /**
     * Verify Lean 4 proof
     *
     * @param code - Lean 4 code to verify
     * @param context - Execution context
     * @returns Promise resolving to verification result
     */
    private verifyProof;
    /**
     * Repair Lean 4 proof
     *
     * @param theorem - Theorem statement
     * @param proofAttempt - Proof attempt to repair
     * @param model - Model to use
     * @param temperature - Temperature for generation
     * @param context - Execution context
     * @returns Promise resolving to repair result
     */
    private repairProof;
    /**
     * Synthesize proof from natural language
     *
     * @param theorem - Theorem statement in natural language
     * @param model - Model to use
     * @param temperature - Temperature for generation
     * @param context - Execution context
     * @returns Promise resolving to synthesized proof
     */
    private synthesizeProof;
    /**
     * Attempt automatic repair of generated proof
     *
     * @param theorem - Theorem statement
     * @param code - Generated code with errors
     * @param model - Model to use
     * @param temperature - Temperature for generation
     * @param context - Execution context
     * @returns Promise resolving to repaired code
     */
    private attemptAutoRepair;
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
     * Get supported models
     *
     * @returns Promise resolving to list of supported models
     */
    getSupportedModels(): Promise<NodeResult>;
    /**
     * Run benchmark
     *
     * @param dataset - Benchmark dataset
     * @param model - Model to use
     * @param evaluator - Evaluator to use
     * @returns Promise resolving to benchmark result
     */
    runBenchmark(dataset: any[], model: string, evaluator: string): Promise<NodeResult>;
    /**
     * Get benchmark results
     *
     * @param benchmarkId - Benchmark ID
     * @returns Promise resolving to benchmark results
     */
    getBenchmarkResults(benchmarkId: string): Promise<NodeResult>;
}
export default LeanAideNode;
