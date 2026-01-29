import { OpenEvolveBaseNode, NodeInputs, NodeResult, ExecutionContext, ValidationError, ParameterSchema } from './OpenEvolveBaseNode';
/**
 * Verification check types
 */
export type VerificationCheck = 'requirements' | 'quality' | 'completeness' | 'correctness' | 'consistency' | 'feasibility' | 'all';
/**
 * Verification result for a single check
 */
export interface CheckResult {
    check: VerificationCheck;
    passed: boolean;
    score: number;
    details: string[];
    severity: 'critical' | 'major' | 'minor' | 'info';
    suggestions: string[];
}
/**
 * Verification report interface
 */
export interface VerificationReport {
    solutionId: string;
    overallScore: number;
    passed: boolean;
    checks: CheckResult[];
    requirements: {
        specified: string[];
        met: string[];
        partiallyMet: string[];
        notMet: string[];
        coverage: number;
    };
    qualityMetrics: {
        completeness: number;
        correctness: number;
        clarity: number;
        consistency: number;
        feasibility: number;
    };
    issues: {
        critical: string[];
        major: string[];
        minor: string[];
    };
    suggestions: string[];
    metadata: {
        verifiedAt: Date;
        verificationTime: number;
        threshold: number;
        verifierVersion: string;
    };
}
/**
 * Verification node configuration
 */
export interface VerificationNodeConfig {
    threshold?: number;
    checks?: VerificationCheck[];
    strictMode?: boolean;
    generateSuggestions?: boolean;
    includeDetails?: boolean;
}
/**
 * Solution Verification Node
 *
 * Validates solutions against requirements and quality standards.
 * Generates detailed verification reports with actionable feedback.
 */
export declare class VerificationNode extends OpenEvolveBaseNode {
    static readonly DISPLAY_NAME = "Solution Verification";
    static readonly DESCRIPTION = "Verify solutions against requirements and quality standards with comprehensive reporting";
    static readonly ICON = "verification";
    static readonly CATEGORY = "verification";
    static readonly VERSION = "1.0.0";
    constructor(id: string, config?: VerificationNodeConfig);
    /**
     * Execute solution verification
     *
     * @param inputs - Must contain 'solution' and 'requirements'
     * @param context - Execution context
     * @returns Promise resolving to verification result
     */
    execute(inputs: NodeInputs, context: ExecutionContext): Promise<NodeResult>;
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
     * Perform all verification checks
     *
     * @param solution - Solution to verify
     * @param requirements - Requirements to verify against
     * @param originalProblem - Original problem statement
     * @param qualityStandards - Quality standards
     * @returns Array of check results
     */
    private performAllChecks;
    /**
     * Perform a single verification check
     *
     * @param check - Check to perform
     * @param solution - Solution to verify
     * @param requirements - Requirements
     * @param originalProblem - Original problem
     * @param qualityStandards - Quality standards
     * @returns Check result
     */
    private performCheck;
    /**
     * Check requirements coverage
     *
     * @param solution - Solution to check
     * @param requirements - Requirements to verify
     * @returns Check result
     */
    private checkRequirements;
    /**
     * Check quality standards
     *
     * @param solution - Solution to check
     * @param qualityStandards - Quality standards
     * @returns Check result
     */
    private checkQuality;
    /**
     * Check completeness
     *
     * @param solution - Solution to check
     * @param requirements - Requirements
     * @param originalProblem - Original problem
     * @returns Check result
     */
    private checkCompleteness;
    /**
     * Check correctness
     *
     * @param solution - Solution to check
     * @param originalProblem - Original problem
     * @returns Check result
     */
    private checkCorrectness;
    /**
     * Check consistency
     *
     * @param solution - Solution to check
     * @returns Check result
     */
    private checkConsistency;
    /**
     * Check feasibility
     *
     * @param solution - Solution to check
     * @returns Check result
     */
    private checkFeasibility;
    /**
     * Analyze requirements coverage
     *
     * @param solution - Solution to analyze
     * @param requirements - Requirements
     * @returns Requirements analysis
     */
    private analyzeRequirementsCoverage;
    /**
     * Calculate quality metrics
     *
     * @param solution - Solution
     * @param checks - Check results
     * @param requirementsAnalysis - Requirements analysis
     * @returns Quality metrics
     */
    private calculateQualityMetrics;
    /**
     * Identify issues from check results
     *
     * @param checks - Check results
     * @returns Issues organized by severity
     */
    private identifyIssues;
    /**
     * Generate improvement suggestions
     *
     * @param checks - Check results
     * @param requirementsAnalysis - Requirements analysis
     * @param qualityMetrics - Quality metrics
     * @returns Array of suggestions
     */
    private generateSuggestions;
    /**
     * Calculate overall verification score
     *
     * @param checks - Check results
     * @param qualityMetrics - Quality metrics
     * @returns Overall score (0-1)
     */
    private calculateOverallScore;
}
export default VerificationNode;
