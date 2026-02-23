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
export declare const LeanTactic: z.ZodEnum<["intros", "apply", "exact", "refine", "by_cases", "constructor", "left", "right", "rw", "simp", "linarith", "ring", "norm_num", "assumption", "contradiction", "existsi", "use", "have", "let", "calc", "induction", "cases", "other"]>;
export type LeanTactic = z.infer<typeof LeanTactic>;
/**
 * Proof Severity Levels
 */
export declare const LeanSeverity: z.ZodEnum<["error", "warning", "info", "hint"]>;
export type LeanSeverity = z.infer<typeof LeanSeverity>;
/**
 * Lean Message Schema
 *
 * Represents a message or error from the Lean compiler.
 */
export declare const LeanMessage: z.ZodObject<{
    severity: z.ZodEnum<["error", "warning", "info", "hint"]>;
    line: z.ZodOptional<z.ZodNumber>;
    column: z.ZodOptional<z.ZodNumber>;
    end_line: z.ZodOptional<z.ZodNumber>;
    end_column: z.ZodOptional<z.ZodNumber>;
    message: z.ZodString;
    code: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    severity: "info" | "error" | "warning" | "hint";
    code?: string | undefined;
    line?: number | undefined;
    column?: number | undefined;
    end_line?: number | undefined;
    end_column?: number | undefined;
}, {
    message: string;
    severity: "info" | "error" | "warning" | "hint";
    code?: string | undefined;
    line?: number | undefined;
    column?: number | undefined;
    end_line?: number | undefined;
    end_column?: number | undefined;
}>;
export type LeanMessage = z.infer<typeof LeanMessage>;
/**
 * Proof Verification Request Schema
 *
 * Represents a request to verify a Lean 4 proof.
 */
export declare const ProofVerificationRequest: z.ZodObject<{
    proof_code: z.ZodString;
    theorem: z.ZodString;
    imports: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    timeout_ms: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    theorem: string;
    proof_code: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    imports?: string[] | undefined;
}, {
    timeout_ms: number;
    theorem: string;
    proof_code: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    imports?: string[] | undefined;
}>;
export type ProofVerificationRequest = z.infer<typeof ProofVerificationRequest>;
/**
 * Proof Verification Response Schema
 *
 * Represents the response from verifying a Lean 4 proof.
 */
export declare const ProofVerificationResponse: z.ZodObject<{
    verified: z.ZodBoolean;
    tactics_used: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    messages: z.ZodOptional<z.ZodArray<z.ZodObject<{
        severity: z.ZodEnum<["error", "warning", "info", "hint"]>;
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        end_line: z.ZodOptional<z.ZodNumber>;
        end_column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }>, "many">>;
    errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        severity: z.ZodEnum<["error", "warning", "info", "hint"]>;
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        end_line: z.ZodOptional<z.ZodNumber>;
        end_column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }>, "many">>;
    remaining_goals: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodObject<{
        lean_version: z.ZodOptional<z.ZodString>;
        verification_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        tactics_count: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        verification_time_ms?: number | undefined;
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
    }, {
        verification_time_ms?: number | undefined;
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    verified: boolean;
    correlation_id?: string | undefined;
    metadata?: {
        verification_time_ms?: number | undefined;
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
    } | undefined;
    errors?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
    tactics_used?: string[] | undefined;
    messages?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
    remaining_goals?: string[] | undefined;
}, {
    timestamp: string;
    verified: boolean;
    correlation_id?: string | undefined;
    metadata?: {
        verification_time_ms?: number | undefined;
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        tactics_count?: number | undefined;
    } | undefined;
    errors?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
    tactics_used?: string[] | undefined;
    messages?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
    remaining_goals?: string[] | undefined;
}>;
export type ProofVerificationResponse = z.infer<typeof ProofVerificationResponse>;
/**
 * Lean Compilation Request Schema
 *
 * Represents a request to compile Lean 4 code.
 */
export declare const LeanCompilationRequest: z.ZodObject<{
    code: z.ZodString;
    filename: z.ZodOptional<z.ZodString>;
    imports: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    timeout_ms: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    code: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    imports?: string[] | undefined;
    filename?: string | undefined;
}, {
    timeout_ms: number;
    code: string;
    correlation_id?: string | undefined;
    metadata?: Record<string, any> | undefined;
    imports?: string[] | undefined;
    filename?: string | undefined;
}>;
export type LeanCompilationRequest = z.infer<typeof LeanCompilationRequest>;
/**
 * Lean Compilation Response Schema
 *
 * Represents the response from compiling Lean 4 code.
 */
export declare const LeanCompilationResponse: z.ZodObject<{
    compiled: z.ZodBoolean;
    warnings: z.ZodOptional<z.ZodArray<z.ZodObject<{
        severity: z.ZodEnum<["error", "warning", "info", "hint"]>;
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        end_line: z.ZodOptional<z.ZodNumber>;
        end_column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }>, "many">>;
    errors: z.ZodOptional<z.ZodArray<z.ZodObject<{
        severity: z.ZodEnum<["error", "warning", "info", "hint"]>;
        line: z.ZodOptional<z.ZodNumber>;
        column: z.ZodOptional<z.ZodNumber>;
        end_line: z.ZodOptional<z.ZodNumber>;
        end_column: z.ZodOptional<z.ZodNumber>;
        message: z.ZodString;
        code: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }, {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }>, "many">>;
    output: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodObject<{
        lean_version: z.ZodOptional<z.ZodString>;
        compilation_time_ms: z.ZodOptional<z.ZodNumber>;
        memory_used_mb: z.ZodOptional<z.ZodNumber>;
        lines_of_code: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        compilation_time_ms?: number | undefined;
        lines_of_code?: number | undefined;
    }, {
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        compilation_time_ms?: number | undefined;
        lines_of_code?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    compiled: boolean;
    correlation_id?: string | undefined;
    metadata?: {
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        compilation_time_ms?: number | undefined;
        lines_of_code?: number | undefined;
    } | undefined;
    errors?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
    output?: string | undefined;
    warnings?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
}, {
    timestamp: string;
    compiled: boolean;
    correlation_id?: string | undefined;
    metadata?: {
        lean_version?: string | undefined;
        memory_used_mb?: number | undefined;
        compilation_time_ms?: number | undefined;
        lines_of_code?: number | undefined;
    } | undefined;
    errors?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
    output?: string | undefined;
    warnings?: {
        message: string;
        severity: "info" | "error" | "warning" | "hint";
        code?: string | undefined;
        line?: number | undefined;
        column?: number | undefined;
        end_line?: number | undefined;
        end_column?: number | undefined;
    }[] | undefined;
}>;
export type LeanCompilationResponse = z.infer<typeof LeanCompilationResponse>;
/**
 * Transformation Functions
 *
 * Helper functions to convert between external formats and the canonical schema.
 */
/**
 * Transform raw LeanAide API response to canonical ProofVerificationResponse
 */
export declare function transformLeanAideResponseToCanonical(rawResponse: any, correlationId?: string): ProofVerificationResponse;
/**
 * Transform canonical ProofVerificationRequest to LeanAide API format
 */
export declare function transformCanonicalToLeanAideRequest(canonicalRequest: ProofVerificationRequest): any;
/**
 * Transform raw Lean compilation response to canonical format
 */
export declare function transformCompilationResponseToCanonical(rawResponse: any, correlationId?: string): LeanCompilationResponse;
/**
 * Validation Functions
 */
/**
 * Validate a ProofVerificationRequest against the schema
 */
export declare function validateProofVerificationRequest(data: unknown): {
    success: boolean;
    data?: ProofVerificationRequest;
    errors?: string[];
};
/**
 * Validate a ProofVerificationResponse against the schema
 */
export declare function validateProofVerificationResponse(data: unknown): {
    success: boolean;
    data?: ProofVerificationResponse;
    errors?: string[];
};
/**
 * Validate a LeanCompilationRequest against the schema
 */
export declare function validateLeanCompilationRequest(data: unknown): {
    success: boolean;
    data?: LeanCompilationRequest;
    errors?: string[];
};
/**
 * Validate a LeanCompilationResponse against the schema
 */
export declare function validateLeanCompilationResponse(data: unknown): {
    success: boolean;
    data?: LeanCompilationResponse;
    errors?: string[];
};
/**
 * Example usage and validation examples
 */
export declare const LeanAideExamples: {
    validProofVerificationRequest: ProofVerificationRequest;
    validProofVerificationResponse: ProofVerificationResponse;
    validLeanCompilationRequest: LeanCompilationRequest;
    validLeanCompilationResponse: LeanCompilationResponse;
    proofVerificationWithError: ProofVerificationResponse;
};
//# sourceMappingURL=leanaide-canonical.d.ts.map