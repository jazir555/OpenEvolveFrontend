import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Attack modes for adversarial testing
 */
export type AttackMode = 'prompt_injection' | 'jailbreak' | 'adversarial_examples' | 'data_poisoning' | 'model_extraction' | 'semantic_attacks';
/**
 * Adversarial test status
 */
export type TestStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
/**
 * Adversarial node configuration
 */
export interface AdversarialNodeConfig {
    attackModes?: AttackMode[];
    numRounds?: number;
    redTeamModels?: Array<{
        provider: string;
        model: string;
    }>;
    blueTeamModels?: Array<{
        provider: string;
        model: string;
    }>;
    timeoutMs?: number;
    enableAutoPatch?: boolean;
}
/**
 * Attack result from a single round
 */
export interface AttackResult {
    round: number;
    attackMode: AttackMode;
    success: boolean;
    promptUsed: string;
    responseReceived: string;
    vulnerability: string;
    confidence: number;
}
/**
 * Patch proposal
 */
export interface PatchProposal {
    round: number;
    attackMode: string;
    vulnerability: string;
    proposedFix: string;
    approved: boolean;
    feedback?: string;
}
/**
 * Adversarial test result
 */
export interface AdversarialTestResult {
    testId: string;
    status: TestStatus;
    content: string;
    attackModes: AttackMode[];
    rounds: number;
    attacks: AttackResult[];
    patches: PatchProposal[];
    summary: {
        totalAttacks: number;
        successfulAttacks: number;
        vulnerabilitiesFound: number;
        patchesProposed: number;
        patchesApplied: number;
        overallRobustness: number;
    };
    recommendations: string[];
    metadata: {
        startedAt: Date;
        completedAt?: Date;
        executionTime: number;
        redTeamModels: Array<{
            provider: string;
            model: string;
        }>;
        blueTeamModels: Array<{
            provider: string;
            model: string;
        }>;
    };
}
/**
 * Adversarial Node
 *
 * Executes red team/blue team adversarial testing to identify vulnerabilities.
 * Provides automated patching and comprehensive security analysis.
 */
export declare class AdversarialNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Adversarial Testing";
    static readonly DESCRIPTION = "Red team/blue team testing for identifying vulnerabilities and improving robustness";
    static readonly ICON = "adversarial";
    static readonly CATEGORY = "testing";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: AdversarialNodeConfig);
    /**
     * Execute adversarial testing
     *
     * @param inputs - Must contain 'content' to test
     * @param context - Execution context
     * @returns Promise resolving to adversarial test result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
    /**
     * Monitor adversarial test progress until completion
     *
     * @param testId - Test ID to monitor
     * @param attackModes - Attack modes being tested
     * @param context - Execution context
     * @returns Promise resolving to test status
     */
    private monitorTest;
    /**
     * Calculate overall robustness score
     *
     * @param result - Test result
     * @returns Robustness score (0-1)
     */
    private calculateRobustness;
    /**
     * Generate recommendations based on test results
     *
     * @param result - Test result
     * @returns Array of recommendations
     */
    private generateRecommendations;
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
     * Approve or reject a patch proposal
     *
     * @param testId - Test ID
     * @param round - Round number
     * @param approved - Whether to approve the patch
     * @param feedback - Optional feedback
     * @returns Promise resolving to approval result
     */
    approvePatch(testId: string, round: number, approved: boolean, feedback?: string): Promise<NodeResult>;
    /**
     * Stop running adversarial test
     *
     * @param testId - Test ID to stop
     * @returns Promise resolving to stop result
     */
    stopTest(testId: string): Promise<NodeResult>;
    /**
     * List all adversarial tests
     *
     * @param params - Optional query parameters
     * @returns Promise resolving to list of tests
     */
    listTests(params?: {
        status?: string;
        limit?: number;
        offset?: number;
    }): Promise<NodeResult>;
}
export default AdversarialNode;
