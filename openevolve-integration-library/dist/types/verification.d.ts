import { ExecutionConfig } from './common';
export interface VerificationInputs {
    operation: 'verify' | 'checks' | 'validate';
    input: VerificationInput | ChecksInput | ValidationInput;
    config?: ExecutionConfig;
}
export interface VerificationInput {
    solution: any;
    requirements: string[];
    options?: VerificationOptions;
}
export interface VerificationOptions {
    level?: 'basic' | 'standard' | 'thorough' | 'exhaustive';
    checkTypes?: VerificationCheckType[];
}
export type VerificationCheckType = 'correctness' | 'completeness' | 'consistency' | 'performance' | 'security' | 'usability' | 'maintainability' | 'scalability';
export interface ChecksInput {
    solution: any;
    checkTypes?: VerificationCheckType[];
    options?: {
        parallel?: boolean;
        timeout?: number;
    };
}
export interface ValidationInput {
    solution: any;
    schema: any;
    options?: {
        strict?: boolean;
        additionalProperties?: boolean;
        customValidators?: Record<string, (value: any) => boolean>;
    };
}
export interface VerificationResult {
    status: 'passed' | 'failed' | 'partial';
    score: number;
    checks: CheckResult[];
}
export interface CheckResult {
    type: VerificationCheckType;
    name: string;
    status: 'passed' | 'failed' | 'warning' | 'skipped';
    score: number;
    message: string;
}
export interface CheckDetail {
    type: 'error' | 'warning' | 'info' | 'success';
    message: string;
    location?: string;
}
export interface RequirementsCoverage {
    total: number;
    covered: number;
    partial: number;
    notCovered: number;
    percentage: number;
}
export interface RequirementBreakdown {
    requirement: string;
    status: 'covered' | 'partial' | 'not-covered';
    score: number;
    checks: string[];
}
export interface Suggestion {
    priority: 'high' | 'medium' | 'low';
    category: string;
    description: string;
    rationale?: string;
    impact?: {
        improvement: string;
        effort: 'low' | 'medium' | 'high';
    };
}
export interface VerificationMetadata {
    verificationTime: number;
    checksPerformed: number;
    checksPassed: number;
    checksFailed: number;
    checksWarning: number;
    checksSkipped: number;
    timestamp: string;
    version: string;
}
export interface ValidationResult {
    valid: boolean;
    errors: ValidationError[];
    score: number;
    details: ValidationDetails;
}
export interface ValidationError {
    path: string;
    message: string;
    expected?: string;
    actual?: any;
    code?: string;
}
export interface ValidationDetails {
    fieldsValidated: number;
}
export interface VerificationExecutionResult {
    type: 'verify' | 'checks' | 'validate';
    result: VerificationResult | CheckResult[] | ValidationResult;
    metadata: {
        executionTime: number;
        timestamp: string;
        apiVersion: string;
    };
}
//# sourceMappingURL=verification.d.ts.map