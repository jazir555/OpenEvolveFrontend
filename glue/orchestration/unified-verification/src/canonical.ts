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
export const ProblemSchema = z.object({
  id: z.string().uuid(),
  type: z.enum([
    'SMT_CONSTRAINTS',
    'THEOREM_PROVING',
    'FORMAL_VERIFICATION',
    'CODE_CORRECTNESS',
    'MODEL_CHECKING',
    'SAT_SOLVING'
  ]),
  description: z.string(),
  statement: z.string(), // The formal statement/theorem/constraints
  variables: z.array(z.object({
    name: z.string(),
    type: z.string(),
    domain: z.string().optional()
  })).optional(),
  metadata: z.record(z.any()).optional()
});

export type Problem = z.infer<typeof ProblemSchema>;

/**
 * Constraints specification
 */
export const ConstraintsSchema = z.object({
  timeout: z.number().positive().default(30000), // milliseconds
  memory: z.number().positive().optional(), // MB
  maxIterations: z.number().positive().optional(),
  precision: z.enum(['low', 'medium', 'high', 'exact']).default('medium'),
  allowedSystems: z.array(z.enum(['z3', 'leanaide', 'both'])).default(['both']),
  requiredConfidence: z.number().min(0).max(1).default(0.95)
});

export type Constraints = z.infer<typeof ConstraintsSchema>;

/**
 * Verification strategies
 */
export type VerificationStrategy =
  | 'z3_only'
  | 'leanaide_only'
  | 'parallel'
  | 'sequential'
  | 'hybrid';

export const VerificationStrategySchema = z.enum([
  'z3_only',
  'leanaide_only',
  'parallel',
  'sequential',
  'hybrid'
]);

/**
 * Main verification request
 */
export const VerificationRequestSchema = z.object({
  requestId: z.string().uuid(),
  problem: ProblemSchema,
  constraints: ConstraintsSchema,
  strategy: VerificationStrategySchema.optional(),
  confidenceRequired: z.number().min(0).max(1).default(0.95),
  timestamp: z.string().datetime(),
  correlationId: z.string().optional()
});

export type VerificationRequest = z.infer<typeof VerificationRequestSchema>;

/**
 * Individual system result
 */
export const VerificationResultSchema = z.object({
  system: z.enum(['z3', 'leanaide']),
  verified: z.boolean(),
  confidence: z.number().min(0).max(1),
  output: z.string(),
  proof: z.string().optional(),
  metadata: z.object({
    executionTime: z.number(), // milliseconds
    memoryUsed: z.number().optional(), // MB
    iterations: z.number().optional(),
    strategy: VerificationStrategySchema,
    timestamp: z.string().datetime(),
    errorMessage: z.string().optional()
  })
});

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
export const ComparisonReportSchema = z.object({
  agreement: z.boolean(),
  agreementType: z.enum([
    'full_agreement',
    'partial_agreement',
    'disagreement',
    'inconclusive'
  ]),
  confidenceAlignment: z.number().min(0).max(1),
  verificationAlignment: z.boolean(),
  details: z.string()
});

export type ComparisonReport = z.infer<typeof ComparisonReportSchema>;

/**
 * Disagreement detection
 */
export const DisagreementSchema = z.object({
  type: z.enum([
    'verification_outcome',
    'confidence_level',
    'proof_structure',
    'timeout_mismatch'
  ]),
  systemA: z.enum(['z3', 'leanaide']),
  systemB: z.enum(['z3', 'leanaide']),
  description: z.string(),
  severity: z.enum(['low', 'medium', 'high', 'critical']),
  resolution: z.enum([
    'trust_z3',
    'trust_leanaide',
    'trust_higher_confidence',
    'require_manual_review',
    'escalate'
  ])
});

export type Disagreement = z.infer<typeof DisagreementSchema>;

/**
 * Cross-validation result
 */
export const CrossValidationResultSchema = z.object({
  requestId: z.string().uuid(),
  verified: z.boolean(),
  agreement: z.boolean(),
  agreementType: z.enum([
    'full_agreement',
    'partial_agreement',
    'disagreement',
    'inconclusive'
  ]),
  confidence: z.number().min(0).max(1),
  systemResults: z.array(VerificationResultSchema),
  conflicts: z.array(DisagreementSchema),
  resolution: z.enum([
    'verified',
    'not_verified',
    'inconclusive',
    'requires_review',
    'escalated'
  ]),
  strategy: VerificationStrategySchema,
  metadata: z.object({
    correlationId: z.string().optional(),
    totalExecutionTime: z.number(),
    timestamp: z.string().datetime()
  })
});

export type CrossValidationResult = z.infer<typeof CrossValidationResultSchema>;

/**
 * Confidence scoring
 */
export const ConfidenceScoreSchema = z.object({
  combined: z.number().min(0).max(1),
  individual: z.record(z.enum(['z3', 'leanaide']), z.number().min(0).max(1)),
  weights: z.record(z.enum(['z3', 'leanaide']), z.number().min(0).max(1)),
  evidence: z.array(z.object({
    source: z.string(),
    weight: z.number().min(0).max(1),
    description: z.string()
  })),
  meetsThreshold: z.boolean(),
  timestamp: z.string().datetime()
});

export type ConfidenceScore = z.infer<typeof ConfidenceScoreSchema>;

/**
 * Verification options for the public API
 */
export const VerificationOptionsSchema = z.object({
  strategy: VerificationStrategySchema.optional(),
  confidenceRequired: z.number().min(0).max(1).default(0.95),
  timeout: z.number().positive().default(30000),
  crossValidate: z.boolean().default(true),
  storeResults: z.boolean().default(true),
  correlationId: z.string().optional()
});

export type VerificationOptions = z.infer<typeof VerificationOptionsSchema>;

/**
 * Learning feedback - for improving strategy selection
 */
export const StrategyEffectivenessSchema = z.object({
  strategy: VerificationStrategySchema,
  problemType: z.enum([
    'SMT_CONSTRAINTS',
    'THEOREM_PROVING',
    'FORMAL_VERIFICATION',
    'CODE_CORRECTNESS',
    'MODEL_CHECKING',
    'SAT_SOLVING'
  ]),
  successRate: z.number().min(0).max(1),
  averageConfidence: z.number().min(0).max(1),
  averageExecutionTime: z.number(),
  sampleSize: z.number().positive(),
  lastUpdated: z.string().datetime()
});

export type StrategyEffectiveness = z.infer<typeof StrategyEffectivenessSchema>;

/**
 * Export all schemas for validation
 */
export const CanonicalSchemas = {
  Problem: ProblemSchema,
  Constraints: ConstraintsSchema,
  VerificationRequest: VerificationRequestSchema,
  VerificationResult: VerificationResultSchema,
  CrossValidationResult: CrossValidationResultSchema,
  ConfidenceScore: ConfidenceScoreSchema,
  VerificationOptions: VerificationOptionsSchema,
  StrategyEffectiveness: StrategyEffectivenessSchema,
  ComparisonReport: ComparisonReportSchema,
  Disagreement: DisagreementSchema
};
