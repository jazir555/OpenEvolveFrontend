/**
 * LeanAide Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for LeanAide (Lean 4 proof assistant) interactions.
 * All adapters must normalize their data to/from this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for LeanAide data in the glue layer.
 * Do not pass raw LeanAide API responses between services.
 */

import { z } from 'zod';

/**
 * LeanAide Tactic Types
 *
 * Common tactics used in Lean 4 proofs.
 */
export const LeanTactic = z.enum([
  'intros',
  'apply',
  'exact',
  'refine',
  'by_cases',
  'constructor',
  'left',
  'right',
  'rw',
  'simp',
  'linarith',
  'ring',
  'norm_num',
  'assumption',
  'contradiction',
  'existsi',
  'use',
  'have',
  'let',
  'calc',
  'induction',
  'cases',
  'other',
]);

export type LeanTactic = z.infer<typeof LeanTactic>;

/**
 * Proof Severity Levels
 */
export const LeanSeverity = z.enum([
  'error',
  'warning',
  'info',
  'hint',
]);

export type LeanSeverity = z.infer<typeof LeanSeverity>;

/**
 * Lean Message Schema
 *
 * Represents a message or error from the Lean compiler.
 */
export const LeanMessage = z.object({
  severity: LeanSeverity.describe("Severity level of the message"),
  line: z.number().int().positive().optional()
    .describe("Line number where the message occurs"),
  column: z.number().int().nonnegative().optional()
    .describe("Column number where the message occurs"),
  end_line: z.number().int().positive().optional()
    .describe("End line number for multiline messages"),
  end_column: z.number().int().nonnegative().optional()
    .describe("End column number for multiline messages"),
  message: z.string().describe("The message text"),
  code: z.string().optional().describe("Error code if applicable"),
});

export type LeanMessage = z.infer<typeof LeanMessage>;

/**
 * Proof Verification Request Schema
 *
 * Represents a request to verify a Lean 4 proof.
 */
export const ProofVerificationRequest = z.object({
  // The Lean 4 proof code to verify
  proof_code: z.string()
    .min(1, "Proof code cannot be empty")
    .describe("The Lean 4 proof code to verify"),

  // The theorem statement being proved
  theorem: z.string()
    .min(1, "Theorem statement cannot be empty")
    .describe("The theorem statement being proved"),

  // Optional imports required for the proof
  imports: z.array(z.string()).optional()
    .describe("Required Lean imports for the proof"),

  // Timeout in milliseconds (MANDATORY - no infinite hangs)
  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(300000, "Timeout cannot exceed 5 minutes")
    .describe("Verification timeout in milliseconds"),

  // Optional metadata for tracking and correlation
  metadata: z.record(z.any()).optional()
    .describe("Optional metadata for observability and tracking"),

  // Optional correlation ID for distributed tracing
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type ProofVerificationRequest = z.infer<typeof ProofVerificationRequest>;

/**
 * Proof Verification Response Schema
 *
 * Represents the response from verifying a Lean 4 proof.
 */
export const ProofVerificationResponse = z.object({
  // Whether the proof was verified successfully
  verified: z.boolean().describe("Whether the proof was verified successfully"),

  // Tactics used during the proof (if successful)
  tactics_used: z.array(z.string()).optional()
    .describe("List of tactics used in the proof"),

  // Messages from the Lean compiler (errors, warnings, hints)
  messages: z.array(LeanMessage).optional()
    .describe("Messages from the Lean compiler"),

  // Error details (if verification failed)
  errors: z.array(LeanMessage).optional()
    .describe("Errors encountered during verification (deprecated, use messages)"),

  // Optional proof goals remaining (if partial proof)
  remaining_goals: z.array(z.string()).optional()
    .describe("Proof goals that remain to be solved"),

  // Execution metadata
  metadata: z.object({
    lean_version: z.string().optional().describe("Lean 4 version used"),
    verification_time_ms: z.number().optional()
      .describe("Actual verification time in milliseconds"),
    memory_used_mb: z.number().optional().describe("Memory usage in MB"),
    tactics_count: z.number().optional().describe("Number of tactics used"),
  }).optional().describe("Execution metadata"),

  // Original correlation ID for tracing
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  // Timestamp in UTC (Law of UTC)
  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type ProofVerificationResponse = z.infer<typeof ProofVerificationResponse>;

/**
 * Lean Compilation Request Schema
 *
 * Represents a request to compile Lean 4 code.
 */
export const LeanCompilationRequest = z.object({
  // The Lean 4 code to compile
  code: z.string()
    .min(1, "Code cannot be empty")
    .describe("The Lean 4 code to compile"),

  // Optional filename for the code
  filename: z.string().optional()
    .describe("Optional filename for the code (e.g., 'Main.lean')"),

  // Optional imports required for the code
  imports: z.array(z.string()).optional()
    .describe("Required Lean imports for the code"),

  // Timeout in milliseconds (MANDATORY - no infinite hangs)
  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(300000, "Timeout cannot exceed 5 minutes")
    .describe("Compilation timeout in milliseconds"),

  // Optional metadata for tracking and correlation
  metadata: z.record(z.any()).optional()
    .describe("Optional metadata for observability and tracking"),

  // Optional correlation ID for distributed tracing
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),
});

export type LeanCompilationRequest = z.infer<typeof LeanCompilationRequest>;

/**
 * Lean Compilation Response Schema
 *
 * Represents the response from compiling Lean 4 code.
 */
export const LeanCompilationResponse = z.object({
  // Whether the compilation was successful
  compiled: z.boolean().describe("Whether the code compiled successfully"),

  // Warnings from the compiler
  warnings: z.array(LeanMessage).optional()
    .describe("Warnings from the Lean compiler"),

  // Errors from the compiler
  errors: z.array(LeanMessage).optional()
    .describe("Errors from the Lean compiler"),

  // Compiled output (if successful and requested)
  output: z.string().optional()
    .describe("Compiled output or IR (if successful)"),

  // Execution metadata
  metadata: z.object({
    lean_version: z.string().optional().describe("Lean 4 version used"),
    compilation_time_ms: z.number().optional()
      .describe("Actual compilation time in milliseconds"),
    memory_used_mb: z.number().optional().describe("Memory usage in MB"),
    lines_of_code: z.number().optional().describe("Number of lines compiled"),
  }).optional().describe("Execution metadata"),

  // Original correlation ID for tracing
  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  // Timestamp in UTC (Law of UTC)
  timestamp: z.string().datetime()
    .describe("UTC timestamp of the response (ISO-8601)"),
});

export type LeanCompilationResponse = z.infer<typeof LeanCompilationResponse>;

/**
 * Transformation Functions
 *
 * Helper functions to convert between external formats and the canonical schema.
 */

/**
 * Transform raw LeanAide API response to canonical ProofVerificationResponse
 */
export function transformLeanAideResponseToCanonical(
  rawResponse: any,
  correlationId?: string
): ProofVerificationResponse {
  const timestamp = new Date().toISOString();

  // Convert raw messages to canonical format
  const messages: LeanMessage[] = (rawResponse.messages || []).map((msg: any) => ({
    severity: LeanSeverity.parse(msg.severity || 'info'),
    line: msg.line,
    column: msg.column,
    end_line: msg.endLine,
    end_column: msg.endColumn,
    message: msg.message || msg.text,
    code: msg.code,
  }));

  // Extract errors for backward compatibility
  const errors = messages.filter(m => m.severity === 'error');

  return {
    verified: rawResponse.verified || rawResponse.success || false,
    tactics_used: rawResponse.tacticsUsed || rawResponse.tactics || [],
    messages,
    errors: errors.length > 0 ? errors : undefined,
    remaining_goals: rawResponse.remainingGoals || rawResponse.goals,
    metadata: {
      lean_version: rawResponse.version,
      verification_time_ms: rawResponse.time,
      memory_used_mb: rawResponse.memory,
      tactics_count: rawResponse.tacticsCount,
    },
    correlation_id: correlationId,
    timestamp,
  };
}

/**
 * Transform canonical ProofVerificationRequest to LeanAide API format
 */
export function transformCanonicalToLeanAideRequest(
  canonicalRequest: ProofVerificationRequest
): any {
  return {
    code: canonicalRequest.proof_code,
    theorem: canonicalRequest.theorem,
    imports: canonicalRequest.imports || [],
    timeout: canonicalRequest.timeout_ms,
    metadata: canonicalRequest.metadata,
  };
}

/**
 * Transform raw Lean compilation response to canonical format
 */
export function transformCompilationResponseToCanonical(
  rawResponse: any,
  correlationId?: string
): LeanCompilationResponse {
  const timestamp = new Date().toISOString();

  // Convert raw messages to canonical format
  const warnings: LeanMessage[] = (rawResponse.warnings || []).map((msg: any) => ({
    severity: LeanSeverity.parse('warning'),
    line: msg.line,
    column: msg.column,
    message: msg.message || msg.text,
  }));

  const errors: LeanMessage[] = (rawResponse.errors || []).map((msg: any) => ({
    severity: LeanSeverity.parse('error'),
    line: msg.line,
    column: msg.column,
    message: msg.message || msg.text,
    code: msg.code,
  }));

  return {
    compiled: rawResponse.compiled || rawResponse.success || false,
    warnings: warnings.length > 0 ? warnings : undefined,
    errors: errors.length > 0 ? errors : undefined,
    output: rawResponse.output,
    metadata: {
      lean_version: rawResponse.version,
      compilation_time_ms: rawResponse.time,
      memory_used_mb: rawResponse.memory,
      lines_of_code: rawResponse.lines,
    },
    correlation_id: correlationId,
    timestamp,
  };
}

/**
 * Validation Functions
 */

/**
 * Validate a ProofVerificationRequest against the schema
 */
export function validateProofVerificationRequest(data: unknown): {
  success: boolean;
  data?: ProofVerificationRequest;
  errors?: string[];
} {
  const result = ProofVerificationRequest.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a ProofVerificationResponse against the schema
 */
export function validateProofVerificationResponse(data: unknown): {
  success: boolean;
  data?: ProofVerificationResponse;
  errors?: string[];
} {
  const result = ProofVerificationResponse.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a LeanCompilationRequest against the schema
 */
export function validateLeanCompilationRequest(data: unknown): {
  success: boolean;
  data?: LeanCompilationRequest;
  errors?: string[];
} {
  const result = LeanCompilationRequest.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Validate a LeanCompilationResponse against the schema
 */
export function validateLeanCompilationResponse(data: unknown): {
  success: boolean;
  data?: LeanCompilationResponse;
  errors?: string[];
} {
  const result = LeanCompilationResponse.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

/**
 * Example usage and validation examples
 */
export const LeanAideExamples = {
  validProofVerificationRequest: {
    proof_code: `
theorem example (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => rw [add_succ, ih]
    `,
    theorem: "∀ (n : Nat), n + 0 = n",
    imports: ["Init.Data.Nat.Basic"],
    timeout_ms: 10000,
    metadata: {
      source: "interactive_prover",
      difficulty: "beginner",
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  } as ProofVerificationRequest,

  validProofVerificationResponse: {
    verified: true,
    tactics_used: ["induction", "rfl", "rw", "add_succ"],
    messages: [],
    metadata: {
      lean_version: "4.7.0",
      verification_time_ms: 125,
      memory_used_mb: 45,
      tactics_count: 4,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as ProofVerificationResponse,

  validLeanCompilationRequest: {
    code: `
def add (m n : Nat) : Nat :=
  match m with
  | Nat.zero => n
  | Nat.succ m' => Nat.succ (add m' n)

theorem add_zero (n : Nat) : add n 0 = n := by
  cases n
  case zero => rfl
  case succ n_ih =>
    rw [add]
    rfl
    `,
    filename: "Example.lean",
    imports: ["Init.Data.Nat.Basic"],
    timeout_ms: 15000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440001",
  } as LeanCompilationRequest,

  validLeanCompilationResponse: {
    compiled: true,
    warnings: [
      {
        severity: "warning" as const,
        line: 8,
        column: 4,
        message: "unused variable n_ih",
      },
    ],
    errors: [],
    metadata: {
      lean_version: "4.7.0",
      compilation_time_ms: 342,
      memory_used_mb: 52,
      lines_of_code: 12,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440001",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as LeanCompilationResponse,

  proofVerificationWithError: {
    verified: false,
    messages: [
      {
        severity: "error" as const,
        line: 2,
        column: 8,
        message: "type mismatch",
        code: "type mismatch",
      },
      {
        severity: "hint" as const,
        line: 2,
        column: 8,
        message: "did you mean to use 'exact' instead of 'apply'?",
      },
    ],
    errors: [
      {
        severity: "error" as const,
        line: 2,
        column: 8,
        message: "type mismatch",
        code: "type mismatch",
      },
    ],
    metadata: {
      lean_version: "4.7.0",
      verification_time_ms: 45,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440002",
    timestamp: "2025-02-03T12:34:56.789Z",
  } as ProofVerificationResponse,
};
