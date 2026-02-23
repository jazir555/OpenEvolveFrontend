/**
 * CANONICAL DATA MODELS - Unified Verification Orchestrator
 *
 * Following the Anti-Corruption Layer (ACL) pattern:
 * - All external system outputs (Z3, LeanAide) are normalized to these schemas
 * - No external data structures pass through unnormalized
 * - These schemas are the source of truth for the glue layer
 */
import { z } from 'zod';
/**
 * Problem representation - canonical format
 */
export declare const ProblemSchema: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["SMT_CONSTRAINTS", "THEOREM_PROVING", "FORMAL_VERIFICATION", "CODE_CORRECTNESS", "MODEL_CHECKING", "SAT_SOLVING"]>;
    description: z.ZodString;
    statement: z.ZodString;
    variables: z.ZodOptional<z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        type: z.ZodString;
        domain: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        type: string;
        domain?: string | undefined;
    }, {
        name: string;
        type: string;
        domain?: string | undefined;
    }>, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
    id: string;
    description: string;
    statement: string;
    metadata?: Record<string, any> | undefined;
    variables?: {
        name: string;
        type: string;
        domain?: string | undefined;
    }[] | undefined;
}, {
    type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
    id: string;
    description: string;
    statement: string;
    metadata?: Record<string, any> | undefined;
    variables?: {
        name: string;
        type: string;
        domain?: string | undefined;
    }[] | undefined;
}>;
export type Problem = z.infer<typeof ProblemSchema>;
/**
 * Constraints specification
 */
export declare const ConstraintsSchema: z.ZodObject<{
    timeout: z.ZodDefault<z.ZodNumber>;
    memory: z.ZodOptional<z.ZodNumber>;
    maxIterations: z.ZodOptional<z.ZodNumber>;
    precision: z.ZodDefault<z.ZodEnum<["low", "medium", "high", "exact"]>>;
    allowedSystems: z.ZodDefault<z.ZodArray<z.ZodEnum<["z3", "leanaide", "both"]>, "many">>;
    requiredConfidence: z.ZodDefault<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    precision: "high" | "medium" | "exact" | "low";
    allowedSystems: ("z3" | "both" | "leanaide")[];
    requiredConfidence: number;
    memory?: number | undefined;
    maxIterations?: number | undefined;
}, {
    memory?: number | undefined;
    timeout?: number | undefined;
    maxIterations?: number | undefined;
    precision?: "high" | "medium" | "exact" | "low" | undefined;
    allowedSystems?: ("z3" | "both" | "leanaide")[] | undefined;
    requiredConfidence?: number | undefined;
}>;
export type Constraints = z.infer<typeof ConstraintsSchema>;
/**
 * Verification strategies
 */
export type VerificationStrategy = 'z3_only' | 'leanaide_only' | 'parallel' | 'sequential' | 'hybrid';
export declare const VerificationStrategySchema: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
/**
 * Main verification request
 */
export declare const VerificationRequestSchema: z.ZodObject<{
    requestId: z.ZodString;
    problem: z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["SMT_CONSTRAINTS", "THEOREM_PROVING", "FORMAL_VERIFICATION", "CODE_CORRECTNESS", "MODEL_CHECKING", "SAT_SOLVING"]>;
        description: z.ZodString;
        statement: z.ZodString;
        variables: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            type: z.ZodString;
            domain: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            type: string;
            domain?: string | undefined;
        }, {
            name: string;
            type: string;
            domain?: string | undefined;
        }>, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        id: string;
        description: string;
        statement: string;
        metadata?: Record<string, any> | undefined;
        variables?: {
            name: string;
            type: string;
            domain?: string | undefined;
        }[] | undefined;
    }, {
        type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        id: string;
        description: string;
        statement: string;
        metadata?: Record<string, any> | undefined;
        variables?: {
            name: string;
            type: string;
            domain?: string | undefined;
        }[] | undefined;
    }>;
    constraints: z.ZodObject<{
        timeout: z.ZodDefault<z.ZodNumber>;
        memory: z.ZodOptional<z.ZodNumber>;
        maxIterations: z.ZodOptional<z.ZodNumber>;
        precision: z.ZodDefault<z.ZodEnum<["low", "medium", "high", "exact"]>>;
        allowedSystems: z.ZodDefault<z.ZodArray<z.ZodEnum<["z3", "leanaide", "both"]>, "many">>;
        requiredConfidence: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        precision: "high" | "medium" | "exact" | "low";
        allowedSystems: ("z3" | "both" | "leanaide")[];
        requiredConfidence: number;
        memory?: number | undefined;
        maxIterations?: number | undefined;
    }, {
        memory?: number | undefined;
        timeout?: number | undefined;
        maxIterations?: number | undefined;
        precision?: "high" | "medium" | "exact" | "low" | undefined;
        allowedSystems?: ("z3" | "both" | "leanaide")[] | undefined;
        requiredConfidence?: number | undefined;
    }>;
    strategy: z.ZodOptional<z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>>;
    confidenceRequired: z.ZodDefault<z.ZodNumber>;
    timestamp: z.ZodString;
    correlationId: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    problem: {
        type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        id: string;
        description: string;
        statement: string;
        metadata?: Record<string, any> | undefined;
        variables?: {
            name: string;
            type: string;
            domain?: string | undefined;
        }[] | undefined;
    };
    constraints: {
        timeout: number;
        precision: "high" | "medium" | "exact" | "low";
        allowedSystems: ("z3" | "both" | "leanaide")[];
        requiredConfidence: number;
        memory?: number | undefined;
        maxIterations?: number | undefined;
    };
    requestId: string;
    confidenceRequired: number;
    strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
    correlationId?: string | undefined;
}, {
    timestamp: string;
    problem: {
        type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        id: string;
        description: string;
        statement: string;
        metadata?: Record<string, any> | undefined;
        variables?: {
            name: string;
            type: string;
            domain?: string | undefined;
        }[] | undefined;
    };
    constraints: {
        memory?: number | undefined;
        timeout?: number | undefined;
        maxIterations?: number | undefined;
        precision?: "high" | "medium" | "exact" | "low" | undefined;
        allowedSystems?: ("z3" | "both" | "leanaide")[] | undefined;
        requiredConfidence?: number | undefined;
    };
    requestId: string;
    strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
    correlationId?: string | undefined;
    confidenceRequired?: number | undefined;
}>;
export type VerificationRequest = z.infer<typeof VerificationRequestSchema>;
/**
 * Individual system result
 */
export declare const VerificationResultSchema: z.ZodObject<{
    system: z.ZodEnum<["z3", "leanaide"]>;
    verified: z.ZodBoolean;
    confidence: z.ZodNumber;
    output: z.ZodString;
    proof: z.ZodOptional<z.ZodString>;
    metadata: z.ZodObject<{
        executionTime: z.ZodNumber;
        memoryUsed: z.ZodOptional<z.ZodNumber>;
        iterations: z.ZodOptional<z.ZodNumber>;
        strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
        timestamp: z.ZodString;
        errorMessage: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        executionTime: number;
        errorMessage?: string | undefined;
        iterations?: number | undefined;
        memoryUsed?: number | undefined;
    }, {
        timestamp: string;
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        executionTime: number;
        errorMessage?: string | undefined;
        iterations?: number | undefined;
        memoryUsed?: number | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        timestamp: string;
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        executionTime: number;
        errorMessage?: string | undefined;
        iterations?: number | undefined;
        memoryUsed?: number | undefined;
    };
    output: string;
    system: "z3" | "leanaide";
    confidence: number;
    verified: boolean;
    proof?: string | undefined;
}, {
    metadata: {
        timestamp: string;
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        executionTime: number;
        errorMessage?: string | undefined;
        iterations?: number | undefined;
        memoryUsed?: number | undefined;
    };
    output: string;
    system: "z3" | "leanaide";
    confidence: number;
    verified: boolean;
    proof?: string | undefined;
}>;
export type VerificationResult = z.infer<typeof VerificationResultSchema>;
/**
 * System result for internal use
 */
export interface SystemResult {
    system: 'z3' | 'leanaide';
    verified: boolean;
    confidence: number;
    output: string;
    proof?: string;
    executionTime: number;
    memoryUsed?: number;
    errorMessage?: string;
    timestamp: string;
}
/**
 * Comparison report for cross-validation
 */
export declare const ComparisonReportSchema: z.ZodObject<{
    agreement: z.ZodBoolean;
    agreementType: z.ZodEnum<["full_agreement", "partial_agreement", "disagreement", "inconclusive"]>;
    confidenceAlignment: z.ZodNumber;
    verificationAlignment: z.ZodBoolean;
    details: z.ZodString;
}, "strip", z.ZodTypeAny, {
    details: string;
    agreement: boolean;
    agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
    confidenceAlignment: number;
    verificationAlignment: boolean;
}, {
    details: string;
    agreement: boolean;
    agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
    confidenceAlignment: number;
    verificationAlignment: boolean;
}>;
export type ComparisonReport = z.infer<typeof ComparisonReportSchema>;
/**
 * Disagreement detection
 */
export declare const DisagreementSchema: z.ZodObject<{
    type: z.ZodEnum<["verification_outcome", "confidence_level", "proof_structure", "timeout_mismatch"]>;
    systemA: z.ZodEnum<["z3", "leanaide"]>;
    systemB: z.ZodEnum<["z3", "leanaide"]>;
    description: z.ZodString;
    severity: z.ZodEnum<["low", "medium", "high", "critical"]>;
    resolution: z.ZodEnum<["trust_z3", "trust_leanaide", "trust_higher_confidence", "require_manual_review", "escalate"]>;
}, "strip", z.ZodTypeAny, {
    type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
    severity: "high" | "medium" | "low" | "critical";
    description: string;
    resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
    systemA: "z3" | "leanaide";
    systemB: "z3" | "leanaide";
}, {
    type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
    severity: "high" | "medium" | "low" | "critical";
    description: string;
    resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
    systemA: "z3" | "leanaide";
    systemB: "z3" | "leanaide";
}>;
export type Disagreement = z.infer<typeof DisagreementSchema>;
/**
 * Cross-validation result
 */
export declare const CrossValidationResultSchema: z.ZodObject<{
    requestId: z.ZodString;
    verified: z.ZodBoolean;
    agreement: z.ZodBoolean;
    agreementType: z.ZodEnum<["full_agreement", "partial_agreement", "disagreement", "inconclusive"]>;
    confidence: z.ZodNumber;
    systemResults: z.ZodArray<z.ZodObject<{
        system: z.ZodEnum<["z3", "leanaide"]>;
        verified: z.ZodBoolean;
        confidence: z.ZodNumber;
        output: z.ZodString;
        proof: z.ZodOptional<z.ZodString>;
        metadata: z.ZodObject<{
            executionTime: z.ZodNumber;
            memoryUsed: z.ZodOptional<z.ZodNumber>;
            iterations: z.ZodOptional<z.ZodNumber>;
            strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
            timestamp: z.ZodString;
            errorMessage: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        }, {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        };
        output: string;
        system: "z3" | "leanaide";
        confidence: number;
        verified: boolean;
        proof?: string | undefined;
    }, {
        metadata: {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        };
        output: string;
        system: "z3" | "leanaide";
        confidence: number;
        verified: boolean;
        proof?: string | undefined;
    }>, "many">;
    conflicts: z.ZodArray<z.ZodObject<{
        type: z.ZodEnum<["verification_outcome", "confidence_level", "proof_structure", "timeout_mismatch"]>;
        systemA: z.ZodEnum<["z3", "leanaide"]>;
        systemB: z.ZodEnum<["z3", "leanaide"]>;
        description: z.ZodString;
        severity: z.ZodEnum<["low", "medium", "high", "critical"]>;
        resolution: z.ZodEnum<["trust_z3", "trust_leanaide", "trust_higher_confidence", "require_manual_review", "escalate"]>;
    }, "strip", z.ZodTypeAny, {
        type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
        systemA: "z3" | "leanaide";
        systemB: "z3" | "leanaide";
    }, {
        type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
        systemA: "z3" | "leanaide";
        systemB: "z3" | "leanaide";
    }>, "many">;
    resolution: z.ZodEnum<["verified", "not_verified", "inconclusive", "requires_review", "escalated"]>;
    strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
    metadata: z.ZodObject<{
        correlationId: z.ZodOptional<z.ZodString>;
        totalExecutionTime: z.ZodNumber;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        totalExecutionTime: number;
        correlationId?: string | undefined;
    }, {
        timestamp: string;
        totalExecutionTime: number;
        correlationId?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        timestamp: string;
        totalExecutionTime: number;
        correlationId?: string | undefined;
    };
    confidence: number;
    strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
    resolution: "verified" | "inconclusive" | "not_verified" | "requires_review" | "escalated";
    verified: boolean;
    conflicts: {
        type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
        systemA: "z3" | "leanaide";
        systemB: "z3" | "leanaide";
    }[];
    requestId: string;
    agreement: boolean;
    agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
    systemResults: {
        metadata: {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        };
        output: string;
        system: "z3" | "leanaide";
        confidence: number;
        verified: boolean;
        proof?: string | undefined;
    }[];
}, {
    metadata: {
        timestamp: string;
        totalExecutionTime: number;
        correlationId?: string | undefined;
    };
    confidence: number;
    strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
    resolution: "verified" | "inconclusive" | "not_verified" | "requires_review" | "escalated";
    verified: boolean;
    conflicts: {
        type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
        systemA: "z3" | "leanaide";
        systemB: "z3" | "leanaide";
    }[];
    requestId: string;
    agreement: boolean;
    agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
    systemResults: {
        metadata: {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        };
        output: string;
        system: "z3" | "leanaide";
        confidence: number;
        verified: boolean;
        proof?: string | undefined;
    }[];
}>;
export type CrossValidationResult = z.infer<typeof CrossValidationResultSchema>;
/**
 * Confidence scoring
 */
export declare const ConfidenceScoreSchema: z.ZodObject<{
    combined: z.ZodNumber;
    individual: z.ZodRecord<z.ZodEnum<["z3", "leanaide"]>, z.ZodNumber>;
    weights: z.ZodRecord<z.ZodEnum<["z3", "leanaide"]>, z.ZodNumber>;
    evidence: z.ZodArray<z.ZodObject<{
        source: z.ZodString;
        weight: z.ZodNumber;
        description: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        source: string;
        description: string;
        weight: number;
    }, {
        source: string;
        description: string;
        weight: number;
    }>, "many">;
    meetsThreshold: z.ZodBoolean;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    combined: number;
    individual: Partial<Record<"z3" | "leanaide", number>>;
    weights: Partial<Record<"z3" | "leanaide", number>>;
    evidence: {
        source: string;
        description: string;
        weight: number;
    }[];
    meetsThreshold: boolean;
}, {
    timestamp: string;
    combined: number;
    individual: Partial<Record<"z3" | "leanaide", number>>;
    weights: Partial<Record<"z3" | "leanaide", number>>;
    evidence: {
        source: string;
        description: string;
        weight: number;
    }[];
    meetsThreshold: boolean;
}>;
export type ConfidenceScore = z.infer<typeof ConfidenceScoreSchema>;
/**
 * Verification options for the public API
 */
export declare const VerificationOptionsSchema: z.ZodObject<{
    strategy: z.ZodOptional<z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>>;
    confidenceRequired: z.ZodDefault<z.ZodNumber>;
    timeout: z.ZodDefault<z.ZodNumber>;
    crossValidate: z.ZodDefault<z.ZodBoolean>;
    storeResults: z.ZodDefault<z.ZodBoolean>;
    correlationId: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout: number;
    confidenceRequired: number;
    crossValidate: boolean;
    storeResults: boolean;
    strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
    correlationId?: string | undefined;
}, {
    timeout?: number | undefined;
    strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
    correlationId?: string | undefined;
    confidenceRequired?: number | undefined;
    crossValidate?: boolean | undefined;
    storeResults?: boolean | undefined;
}>;
export type VerificationOptions = z.infer<typeof VerificationOptionsSchema>;
/**
 * Learning feedback - for improving strategy selection
 */
export declare const StrategyEffectivenessSchema: z.ZodObject<{
    strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
    problemType: z.ZodEnum<["SMT_CONSTRAINTS", "THEOREM_PROVING", "FORMAL_VERIFICATION", "CODE_CORRECTNESS", "MODEL_CHECKING", "SAT_SOLVING"]>;
    successRate: z.ZodNumber;
    averageConfidence: z.ZodNumber;
    averageExecutionTime: z.ZodNumber;
    sampleSize: z.ZodNumber;
    lastUpdated: z.ZodString;
}, "strip", z.ZodTypeAny, {
    strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
    lastUpdated: string;
    problemType: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
    successRate: number;
    averageConfidence: number;
    averageExecutionTime: number;
    sampleSize: number;
}, {
    strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
    lastUpdated: string;
    problemType: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
    successRate: number;
    averageConfidence: number;
    averageExecutionTime: number;
    sampleSize: number;
}>;
export type StrategyEffectiveness = z.infer<typeof StrategyEffectivenessSchema>;
/**
 * Export all schemas for validation
 */
export declare const CanonicalSchemas: {
    Problem: z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["SMT_CONSTRAINTS", "THEOREM_PROVING", "FORMAL_VERIFICATION", "CODE_CORRECTNESS", "MODEL_CHECKING", "SAT_SOLVING"]>;
        description: z.ZodString;
        statement: z.ZodString;
        variables: z.ZodOptional<z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            type: z.ZodString;
            domain: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            type: string;
            domain?: string | undefined;
        }, {
            name: string;
            type: string;
            domain?: string | undefined;
        }>, "many">>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        id: string;
        description: string;
        statement: string;
        metadata?: Record<string, any> | undefined;
        variables?: {
            name: string;
            type: string;
            domain?: string | undefined;
        }[] | undefined;
    }, {
        type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        id: string;
        description: string;
        statement: string;
        metadata?: Record<string, any> | undefined;
        variables?: {
            name: string;
            type: string;
            domain?: string | undefined;
        }[] | undefined;
    }>;
    Constraints: z.ZodObject<{
        timeout: z.ZodDefault<z.ZodNumber>;
        memory: z.ZodOptional<z.ZodNumber>;
        maxIterations: z.ZodOptional<z.ZodNumber>;
        precision: z.ZodDefault<z.ZodEnum<["low", "medium", "high", "exact"]>>;
        allowedSystems: z.ZodDefault<z.ZodArray<z.ZodEnum<["z3", "leanaide", "both"]>, "many">>;
        requiredConfidence: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        precision: "high" | "medium" | "exact" | "low";
        allowedSystems: ("z3" | "both" | "leanaide")[];
        requiredConfidence: number;
        memory?: number | undefined;
        maxIterations?: number | undefined;
    }, {
        memory?: number | undefined;
        timeout?: number | undefined;
        maxIterations?: number | undefined;
        precision?: "high" | "medium" | "exact" | "low" | undefined;
        allowedSystems?: ("z3" | "both" | "leanaide")[] | undefined;
        requiredConfidence?: number | undefined;
    }>;
    VerificationRequest: z.ZodObject<{
        requestId: z.ZodString;
        problem: z.ZodObject<{
            id: z.ZodString;
            type: z.ZodEnum<["SMT_CONSTRAINTS", "THEOREM_PROVING", "FORMAL_VERIFICATION", "CODE_CORRECTNESS", "MODEL_CHECKING", "SAT_SOLVING"]>;
            description: z.ZodString;
            statement: z.ZodString;
            variables: z.ZodOptional<z.ZodArray<z.ZodObject<{
                name: z.ZodString;
                type: z.ZodString;
                domain: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                name: string;
                type: string;
                domain?: string | undefined;
            }, {
                name: string;
                type: string;
                domain?: string | undefined;
            }>, "many">>;
            metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        }, "strip", z.ZodTypeAny, {
            type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
            id: string;
            description: string;
            statement: string;
            metadata?: Record<string, any> | undefined;
            variables?: {
                name: string;
                type: string;
                domain?: string | undefined;
            }[] | undefined;
        }, {
            type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
            id: string;
            description: string;
            statement: string;
            metadata?: Record<string, any> | undefined;
            variables?: {
                name: string;
                type: string;
                domain?: string | undefined;
            }[] | undefined;
        }>;
        constraints: z.ZodObject<{
            timeout: z.ZodDefault<z.ZodNumber>;
            memory: z.ZodOptional<z.ZodNumber>;
            maxIterations: z.ZodOptional<z.ZodNumber>;
            precision: z.ZodDefault<z.ZodEnum<["low", "medium", "high", "exact"]>>;
            allowedSystems: z.ZodDefault<z.ZodArray<z.ZodEnum<["z3", "leanaide", "both"]>, "many">>;
            requiredConfidence: z.ZodDefault<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            timeout: number;
            precision: "high" | "medium" | "exact" | "low";
            allowedSystems: ("z3" | "both" | "leanaide")[];
            requiredConfidence: number;
            memory?: number | undefined;
            maxIterations?: number | undefined;
        }, {
            memory?: number | undefined;
            timeout?: number | undefined;
            maxIterations?: number | undefined;
            precision?: "high" | "medium" | "exact" | "low" | undefined;
            allowedSystems?: ("z3" | "both" | "leanaide")[] | undefined;
            requiredConfidence?: number | undefined;
        }>;
        strategy: z.ZodOptional<z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>>;
        confidenceRequired: z.ZodDefault<z.ZodNumber>;
        timestamp: z.ZodString;
        correlationId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        problem: {
            type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
            id: string;
            description: string;
            statement: string;
            metadata?: Record<string, any> | undefined;
            variables?: {
                name: string;
                type: string;
                domain?: string | undefined;
            }[] | undefined;
        };
        constraints: {
            timeout: number;
            precision: "high" | "medium" | "exact" | "low";
            allowedSystems: ("z3" | "both" | "leanaide")[];
            requiredConfidence: number;
            memory?: number | undefined;
            maxIterations?: number | undefined;
        };
        requestId: string;
        confidenceRequired: number;
        strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
        correlationId?: string | undefined;
    }, {
        timestamp: string;
        problem: {
            type: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
            id: string;
            description: string;
            statement: string;
            metadata?: Record<string, any> | undefined;
            variables?: {
                name: string;
                type: string;
                domain?: string | undefined;
            }[] | undefined;
        };
        constraints: {
            memory?: number | undefined;
            timeout?: number | undefined;
            maxIterations?: number | undefined;
            precision?: "high" | "medium" | "exact" | "low" | undefined;
            allowedSystems?: ("z3" | "both" | "leanaide")[] | undefined;
            requiredConfidence?: number | undefined;
        };
        requestId: string;
        strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
        correlationId?: string | undefined;
        confidenceRequired?: number | undefined;
    }>;
    VerificationResult: z.ZodObject<{
        system: z.ZodEnum<["z3", "leanaide"]>;
        verified: z.ZodBoolean;
        confidence: z.ZodNumber;
        output: z.ZodString;
        proof: z.ZodOptional<z.ZodString>;
        metadata: z.ZodObject<{
            executionTime: z.ZodNumber;
            memoryUsed: z.ZodOptional<z.ZodNumber>;
            iterations: z.ZodOptional<z.ZodNumber>;
            strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
            timestamp: z.ZodString;
            errorMessage: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        }, {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        };
        output: string;
        system: "z3" | "leanaide";
        confidence: number;
        verified: boolean;
        proof?: string | undefined;
    }, {
        metadata: {
            timestamp: string;
            strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
            executionTime: number;
            errorMessage?: string | undefined;
            iterations?: number | undefined;
            memoryUsed?: number | undefined;
        };
        output: string;
        system: "z3" | "leanaide";
        confidence: number;
        verified: boolean;
        proof?: string | undefined;
    }>;
    CrossValidationResult: z.ZodObject<{
        requestId: z.ZodString;
        verified: z.ZodBoolean;
        agreement: z.ZodBoolean;
        agreementType: z.ZodEnum<["full_agreement", "partial_agreement", "disagreement", "inconclusive"]>;
        confidence: z.ZodNumber;
        systemResults: z.ZodArray<z.ZodObject<{
            system: z.ZodEnum<["z3", "leanaide"]>;
            verified: z.ZodBoolean;
            confidence: z.ZodNumber;
            output: z.ZodString;
            proof: z.ZodOptional<z.ZodString>;
            metadata: z.ZodObject<{
                executionTime: z.ZodNumber;
                memoryUsed: z.ZodOptional<z.ZodNumber>;
                iterations: z.ZodOptional<z.ZodNumber>;
                strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
                timestamp: z.ZodString;
                errorMessage: z.ZodOptional<z.ZodString>;
            }, "strip", z.ZodTypeAny, {
                timestamp: string;
                strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
                executionTime: number;
                errorMessage?: string | undefined;
                iterations?: number | undefined;
                memoryUsed?: number | undefined;
            }, {
                timestamp: string;
                strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
                executionTime: number;
                errorMessage?: string | undefined;
                iterations?: number | undefined;
                memoryUsed?: number | undefined;
            }>;
        }, "strip", z.ZodTypeAny, {
            metadata: {
                timestamp: string;
                strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
                executionTime: number;
                errorMessage?: string | undefined;
                iterations?: number | undefined;
                memoryUsed?: number | undefined;
            };
            output: string;
            system: "z3" | "leanaide";
            confidence: number;
            verified: boolean;
            proof?: string | undefined;
        }, {
            metadata: {
                timestamp: string;
                strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
                executionTime: number;
                errorMessage?: string | undefined;
                iterations?: number | undefined;
                memoryUsed?: number | undefined;
            };
            output: string;
            system: "z3" | "leanaide";
            confidence: number;
            verified: boolean;
            proof?: string | undefined;
        }>, "many">;
        conflicts: z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["verification_outcome", "confidence_level", "proof_structure", "timeout_mismatch"]>;
            systemA: z.ZodEnum<["z3", "leanaide"]>;
            systemB: z.ZodEnum<["z3", "leanaide"]>;
            description: z.ZodString;
            severity: z.ZodEnum<["low", "medium", "high", "critical"]>;
            resolution: z.ZodEnum<["trust_z3", "trust_leanaide", "trust_higher_confidence", "require_manual_review", "escalate"]>;
        }, "strip", z.ZodTypeAny, {
            type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
            severity: "high" | "medium" | "low" | "critical";
            description: string;
            resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
            systemA: "z3" | "leanaide";
            systemB: "z3" | "leanaide";
        }, {
            type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
            severity: "high" | "medium" | "low" | "critical";
            description: string;
            resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
            systemA: "z3" | "leanaide";
            systemB: "z3" | "leanaide";
        }>, "many">;
        resolution: z.ZodEnum<["verified", "not_verified", "inconclusive", "requires_review", "escalated"]>;
        strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
        metadata: z.ZodObject<{
            correlationId: z.ZodOptional<z.ZodString>;
            totalExecutionTime: z.ZodNumber;
            timestamp: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            timestamp: string;
            totalExecutionTime: number;
            correlationId?: string | undefined;
        }, {
            timestamp: string;
            totalExecutionTime: number;
            correlationId?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            timestamp: string;
            totalExecutionTime: number;
            correlationId?: string | undefined;
        };
        confidence: number;
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        resolution: "verified" | "inconclusive" | "not_verified" | "requires_review" | "escalated";
        verified: boolean;
        conflicts: {
            type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
            severity: "high" | "medium" | "low" | "critical";
            description: string;
            resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
            systemA: "z3" | "leanaide";
            systemB: "z3" | "leanaide";
        }[];
        requestId: string;
        agreement: boolean;
        agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
        systemResults: {
            metadata: {
                timestamp: string;
                strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
                executionTime: number;
                errorMessage?: string | undefined;
                iterations?: number | undefined;
                memoryUsed?: number | undefined;
            };
            output: string;
            system: "z3" | "leanaide";
            confidence: number;
            verified: boolean;
            proof?: string | undefined;
        }[];
    }, {
        metadata: {
            timestamp: string;
            totalExecutionTime: number;
            correlationId?: string | undefined;
        };
        confidence: number;
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        resolution: "verified" | "inconclusive" | "not_verified" | "requires_review" | "escalated";
        verified: boolean;
        conflicts: {
            type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
            severity: "high" | "medium" | "low" | "critical";
            description: string;
            resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
            systemA: "z3" | "leanaide";
            systemB: "z3" | "leanaide";
        }[];
        requestId: string;
        agreement: boolean;
        agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
        systemResults: {
            metadata: {
                timestamp: string;
                strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
                executionTime: number;
                errorMessage?: string | undefined;
                iterations?: number | undefined;
                memoryUsed?: number | undefined;
            };
            output: string;
            system: "z3" | "leanaide";
            confidence: number;
            verified: boolean;
            proof?: string | undefined;
        }[];
    }>;
    ConfidenceScore: z.ZodObject<{
        combined: z.ZodNumber;
        individual: z.ZodRecord<z.ZodEnum<["z3", "leanaide"]>, z.ZodNumber>;
        weights: z.ZodRecord<z.ZodEnum<["z3", "leanaide"]>, z.ZodNumber>;
        evidence: z.ZodArray<z.ZodObject<{
            source: z.ZodString;
            weight: z.ZodNumber;
            description: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            source: string;
            description: string;
            weight: number;
        }, {
            source: string;
            description: string;
            weight: number;
        }>, "many">;
        meetsThreshold: z.ZodBoolean;
        timestamp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        combined: number;
        individual: Partial<Record<"z3" | "leanaide", number>>;
        weights: Partial<Record<"z3" | "leanaide", number>>;
        evidence: {
            source: string;
            description: string;
            weight: number;
        }[];
        meetsThreshold: boolean;
    }, {
        timestamp: string;
        combined: number;
        individual: Partial<Record<"z3" | "leanaide", number>>;
        weights: Partial<Record<"z3" | "leanaide", number>>;
        evidence: {
            source: string;
            description: string;
            weight: number;
        }[];
        meetsThreshold: boolean;
    }>;
    VerificationOptions: z.ZodObject<{
        strategy: z.ZodOptional<z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>>;
        confidenceRequired: z.ZodDefault<z.ZodNumber>;
        timeout: z.ZodDefault<z.ZodNumber>;
        crossValidate: z.ZodDefault<z.ZodBoolean>;
        storeResults: z.ZodDefault<z.ZodBoolean>;
        correlationId: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        timeout: number;
        confidenceRequired: number;
        crossValidate: boolean;
        storeResults: boolean;
        strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
        correlationId?: string | undefined;
    }, {
        timeout?: number | undefined;
        strategy?: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only" | undefined;
        correlationId?: string | undefined;
        confidenceRequired?: number | undefined;
        crossValidate?: boolean | undefined;
        storeResults?: boolean | undefined;
    }>;
    StrategyEffectiveness: z.ZodObject<{
        strategy: z.ZodEnum<["z3_only", "leanaide_only", "parallel", "sequential", "hybrid"]>;
        problemType: z.ZodEnum<["SMT_CONSTRAINTS", "THEOREM_PROVING", "FORMAL_VERIFICATION", "CODE_CORRECTNESS", "MODEL_CHECKING", "SAT_SOLVING"]>;
        successRate: z.ZodNumber;
        averageConfidence: z.ZodNumber;
        averageExecutionTime: z.ZodNumber;
        sampleSize: z.ZodNumber;
        lastUpdated: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        lastUpdated: string;
        problemType: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        successRate: number;
        averageConfidence: number;
        averageExecutionTime: number;
        sampleSize: number;
    }, {
        strategy: "hybrid" | "sequential" | "parallel" | "z3_only" | "leanaide_only";
        lastUpdated: string;
        problemType: "SMT_CONSTRAINTS" | "THEOREM_PROVING" | "FORMAL_VERIFICATION" | "CODE_CORRECTNESS" | "MODEL_CHECKING" | "SAT_SOLVING";
        successRate: number;
        averageConfidence: number;
        averageExecutionTime: number;
        sampleSize: number;
    }>;
    ComparisonReport: z.ZodObject<{
        agreement: z.ZodBoolean;
        agreementType: z.ZodEnum<["full_agreement", "partial_agreement", "disagreement", "inconclusive"]>;
        confidenceAlignment: z.ZodNumber;
        verificationAlignment: z.ZodBoolean;
        details: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        details: string;
        agreement: boolean;
        agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
        confidenceAlignment: number;
        verificationAlignment: boolean;
    }, {
        details: string;
        agreement: boolean;
        agreementType: "full_agreement" | "partial_agreement" | "disagreement" | "inconclusive";
        confidenceAlignment: number;
        verificationAlignment: boolean;
    }>;
    Disagreement: z.ZodObject<{
        type: z.ZodEnum<["verification_outcome", "confidence_level", "proof_structure", "timeout_mismatch"]>;
        systemA: z.ZodEnum<["z3", "leanaide"]>;
        systemB: z.ZodEnum<["z3", "leanaide"]>;
        description: z.ZodString;
        severity: z.ZodEnum<["low", "medium", "high", "critical"]>;
        resolution: z.ZodEnum<["trust_z3", "trust_leanaide", "trust_higher_confidence", "require_manual_review", "escalate"]>;
    }, "strip", z.ZodTypeAny, {
        type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
        systemA: "z3" | "leanaide";
        systemB: "z3" | "leanaide";
    }, {
        type: "confidence_level" | "verification_outcome" | "proof_structure" | "timeout_mismatch";
        severity: "high" | "medium" | "low" | "critical";
        description: string;
        resolution: "escalate" | "trust_z3" | "trust_leanaide" | "trust_higher_confidence" | "require_manual_review";
        systemA: "z3" | "leanaide";
        systemB: "z3" | "leanaide";
    }>;
};
//# sourceMappingURL=canonical.d.ts.map