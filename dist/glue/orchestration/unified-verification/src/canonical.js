"use strict";
/**
 * CANONICAL DATA MODELS - Unified Verification Orchestrator
 *
 * Following the Anti-Corruption Layer (ACL) pattern:
 * - All external system outputs (Z3, LeanAide) are normalized to these schemas
 * - No external data structures pass through unnormalized
 * - These schemas are the source of truth for the glue layer
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CanonicalSchemas = exports.StrategyEffectivenessSchema = exports.VerificationOptionsSchema = exports.ConfidenceScoreSchema = exports.CrossValidationResultSchema = exports.DisagreementSchema = exports.ComparisonReportSchema = exports.VerificationResultSchema = exports.VerificationRequestSchema = exports.VerificationStrategySchema = exports.ConstraintsSchema = exports.ProblemSchema = void 0;
const zod_1 = require("zod");
/**
 * Problem representation - canonical format
 */
exports.ProblemSchema = zod_1.z.object({
    id: zod_1.z.string().uuid(),
    type: zod_1.z.enum([
        'SMT_CONSTRAINTS',
        'THEOREM_PROVING',
        'FORMAL_VERIFICATION',
        'CODE_CORRECTNESS',
        'MODEL_CHECKING',
        'SAT_SOLVING'
    ]),
    description: zod_1.z.string(),
    statement: zod_1.z.string(), // The formal statement/theorem/constraints
    variables: zod_1.z.array(zod_1.z.object({
        name: zod_1.z.string(),
        type: zod_1.z.string(),
        domain: zod_1.z.string().optional()
    })).optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
});
/**
 * Constraints specification
 */
exports.ConstraintsSchema = zod_1.z.object({
    timeout: zod_1.z.number().positive().default(30000), // milliseconds
    memory: zod_1.z.number().positive().optional(), // MB
    maxIterations: zod_1.z.number().positive().optional(),
    precision: zod_1.z.enum(['low', 'medium', 'high', 'exact']).default('medium'),
    allowedSystems: zod_1.z.array(zod_1.z.enum(['z3', 'leanaide', 'both'])).default(['both']),
    requiredConfidence: zod_1.z.number().min(0).max(1).default(0.95)
});
exports.VerificationStrategySchema = zod_1.z.enum([
    'z3_only',
    'leanaide_only',
    'parallel',
    'sequential',
    'hybrid'
]);
/**
 * Main verification request
 */
exports.VerificationRequestSchema = zod_1.z.object({
    requestId: zod_1.z.string().uuid(),
    problem: exports.ProblemSchema,
    constraints: exports.ConstraintsSchema,
    strategy: exports.VerificationStrategySchema.optional(),
    confidenceRequired: zod_1.z.number().min(0).max(1).default(0.95),
    timestamp: zod_1.z.string().datetime(),
    correlationId: zod_1.z.string().optional()
});
/**
 * Individual system result
 */
exports.VerificationResultSchema = zod_1.z.object({
    system: zod_1.z.enum(['z3', 'leanaide']),
    verified: zod_1.z.boolean(),
    confidence: zod_1.z.number().min(0).max(1),
    output: zod_1.z.string(),
    proof: zod_1.z.string().optional(),
    metadata: zod_1.z.object({
        executionTime: zod_1.z.number(), // milliseconds
        memoryUsed: zod_1.z.number().optional(), // MB
        iterations: zod_1.z.number().optional(),
        strategy: exports.VerificationStrategySchema,
        timestamp: zod_1.z.string().datetime(),
        errorMessage: zod_1.z.string().optional()
    })
});
/**
 * Comparison report for cross-validation
 */
exports.ComparisonReportSchema = zod_1.z.object({
    agreement: zod_1.z.boolean(),
    agreementType: zod_1.z.enum([
        'full_agreement',
        'partial_agreement',
        'disagreement',
        'inconclusive'
    ]),
    confidenceAlignment: zod_1.z.number().min(0).max(1),
    verificationAlignment: zod_1.z.boolean(),
    details: zod_1.z.string()
});
/**
 * Disagreement detection
 */
exports.DisagreementSchema = zod_1.z.object({
    type: zod_1.z.enum([
        'verification_outcome',
        'confidence_level',
        'proof_structure',
        'timeout_mismatch'
    ]),
    systemA: zod_1.z.enum(['z3', 'leanaide']),
    systemB: zod_1.z.enum(['z3', 'leanaide']),
    description: zod_1.z.string(),
    severity: zod_1.z.enum(['low', 'medium', 'high', 'critical']),
    resolution: zod_1.z.enum([
        'trust_z3',
        'trust_leanaide',
        'trust_higher_confidence',
        'require_manual_review',
        'escalate'
    ])
});
/**
 * Cross-validation result
 */
exports.CrossValidationResultSchema = zod_1.z.object({
    requestId: zod_1.z.string().uuid(),
    verified: zod_1.z.boolean(),
    agreement: zod_1.z.boolean(),
    agreementType: zod_1.z.enum([
        'full_agreement',
        'partial_agreement',
        'disagreement',
        'inconclusive'
    ]),
    confidence: zod_1.z.number().min(0).max(1),
    systemResults: zod_1.z.array(exports.VerificationResultSchema),
    conflicts: zod_1.z.array(exports.DisagreementSchema),
    resolution: zod_1.z.enum([
        'verified',
        'not_verified',
        'inconclusive',
        'requires_review',
        'escalated'
    ]),
    strategy: exports.VerificationStrategySchema,
    metadata: zod_1.z.object({
        correlationId: zod_1.z.string().optional(),
        totalExecutionTime: zod_1.z.number(),
        timestamp: zod_1.z.string().datetime()
    })
});
/**
 * Confidence scoring
 */
exports.ConfidenceScoreSchema = zod_1.z.object({
    combined: zod_1.z.number().min(0).max(1),
    individual: zod_1.z.record(zod_1.z.enum(['z3', 'leanaide']), zod_1.z.number().min(0).max(1)),
    weights: zod_1.z.record(zod_1.z.enum(['z3', 'leanaide']), zod_1.z.number().min(0).max(1)),
    evidence: zod_1.z.array(zod_1.z.object({
        source: zod_1.z.string(),
        weight: zod_1.z.number().min(0).max(1),
        description: zod_1.z.string()
    })),
    meetsThreshold: zod_1.z.boolean(),
    timestamp: zod_1.z.string().datetime()
});
/**
 * Verification options for the public API
 */
exports.VerificationOptionsSchema = zod_1.z.object({
    strategy: exports.VerificationStrategySchema.optional(),
    confidenceRequired: zod_1.z.number().min(0).max(1).default(0.95),
    timeout: zod_1.z.number().positive().default(30000),
    crossValidate: zod_1.z.boolean().default(true),
    storeResults: zod_1.z.boolean().default(true),
    correlationId: zod_1.z.string().optional()
});
/**
 * Learning feedback - for improving strategy selection
 */
exports.StrategyEffectivenessSchema = zod_1.z.object({
    strategy: exports.VerificationStrategySchema,
    problemType: zod_1.z.enum([
        'SMT_CONSTRAINTS',
        'THEOREM_PROVING',
        'FORMAL_VERIFICATION',
        'CODE_CORRECTNESS',
        'MODEL_CHECKING',
        'SAT_SOLVING'
    ]),
    successRate: zod_1.z.number().min(0).max(1),
    averageConfidence: zod_1.z.number().min(0).max(1),
    averageExecutionTime: zod_1.z.number(),
    sampleSize: zod_1.z.number().positive(),
    lastUpdated: zod_1.z.string().datetime()
});
/**
 * Export all schemas for validation
 */
exports.CanonicalSchemas = {
    Problem: exports.ProblemSchema,
    Constraints: exports.ConstraintsSchema,
    VerificationRequest: exports.VerificationRequestSchema,
    VerificationResult: exports.VerificationResultSchema,
    CrossValidationResult: exports.CrossValidationResultSchema,
    ConfidenceScore: exports.ConfidenceScoreSchema,
    VerificationOptions: exports.VerificationOptionsSchema,
    StrategyEffectiveness: exports.StrategyEffectivenessSchema,
    ComparisonReport: exports.ComparisonReportSchema,
    Disagreement: exports.DisagreementSchema
};
//# sourceMappingURL=canonical.js.map