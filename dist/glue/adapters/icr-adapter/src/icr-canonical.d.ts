/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 *
 * ICR Canonical Schemas
 *
 * Canonical data models for all 7 ICR modes.
 * These schemas define the Anti-Corruption Layer (ACL) contract.
 * All data entering/leaving the ICR system MUST conform to these schemas.
 *
 * FEDERATION CONSTITUTION COMPLIANCE:
 * - Air Gap: No imports from core-projects
 * - Runtime Truth: Schemas reflect actual API behavior
 * - Configuration Explicitness: All fields required (no magic defaults)
 * - UTC: All timestamps in UTC ISO-8601 format
 */
import { z } from 'zod';
/**
 * Mode type enum for all 7 ICR modes
 */
export declare const ModeTypeSchema: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
export type ModeType = z.infer<typeof ModeTypeSchema>;
/**
 * Base metadata schema included in all requests/responses
 */
export declare const ICRMetadataSchema: z.ZodObject<{
    correlation_id: z.ZodString;
    timestamp_utc: z.ZodString;
    source_service: z.ZodDefault<z.ZodString>;
    mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
    request_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    correlation_id: string;
    source_service: string;
    mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
    timestamp_utc: string;
    request_id?: string | undefined;
}, {
    correlation_id: string;
    mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
    timestamp_utc: string;
    source_service?: string | undefined;
    request_id?: string | undefined;
}>;
export type ICRMetadata = z.infer<typeof ICRMetadataSchema>;
/**
 * Base result schema for all mode responses
 */
export declare const ICRResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    content: z.ZodString;
    error: z.ZodOptional<z.ZodString>;
    execution_time_ms: z.ZodNumber;
    iteration_count: z.ZodDefault<z.ZodNumber>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    success: boolean;
    content: string;
    execution_time_ms: number;
    iteration_count: number;
    error?: string | undefined;
    metadata?: Record<string, any> | undefined;
}, {
    success: boolean;
    content: string;
    execution_time_ms: number;
    error?: string | undefined;
    metadata?: Record<string, any> | undefined;
    iteration_count?: number | undefined;
}>;
export type ICRResult = z.infer<typeof ICRResultSchema>;
/**
 * Mode options schema
 */
export declare const ModeOptionsSchema: z.ZodObject<{
    temperature: z.ZodOptional<z.ZodNumber>;
    top_p: z.ZodOptional<z.ZodNumber>;
    max_iterations: z.ZodOptional<z.ZodNumber>;
    model_name: z.ZodOptional<z.ZodString>;
    provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
}, "strip", z.ZodTypeAny, {
    provider?: "google" | "openai" | "anthropic" | undefined;
    max_iterations?: number | undefined;
    temperature?: number | undefined;
    top_p?: number | undefined;
    model_name?: string | undefined;
}, {
    provider?: "google" | "openai" | "anthropic" | undefined;
    max_iterations?: number | undefined;
    temperature?: number | undefined;
    top_p?: number | undefined;
    model_name?: string | undefined;
}>;
export type ModeOptions = z.infer<typeof ModeOptionsSchema>;
/**
 * Refine Mode Request Schema
 * Mode: Traditional iterative refinements with automated feature suggestion
 */
export declare const RefineModeRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"refine">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        evolution_mode: z.ZodOptional<z.ZodEnum<["novelty", "quality", "off"]>>;
        refinement_stages: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "refine";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "refine";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    } | undefined;
}>;
export type RefineModeRequest = z.infer<typeof RefineModeRequestSchema>;
/**
 * Refine Mode Response Schema
 */
export declare const RefineModeResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"refine">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"refine">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            evolution_mode: z.ZodOptional<z.ZodEnum<["novelty", "quality", "off"]>>;
            refinement_stages: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        iterations: z.ZodArray<z.ZodObject<{
            iteration_number: z.ZodNumber;
            content: z.ZodString;
            suggested_features: z.ZodOptional<z.ZodString>;
            bug_fixes: z.ZodOptional<z.ZodString>;
            status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
    };
    mode: "refine";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
    };
    mode: "refine";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    };
}>;
export type RefineModeResponse = z.infer<typeof RefineModeResponseSchema>;
/**
 * React Mode Request Schema
 * Mode: React application development with orchestrator-coordination
 */
export declare const ReactModeRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"react">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        worker_count: z.ZodOptional<z.ZodNumber>;
        enable_preview: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "react";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "react";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    } | undefined;
}>;
export type ReactModeRequest = z.infer<typeof ReactModeRequestSchema>;
/**
 * React Mode Response Schema
 */
export declare const ReactModeResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"react">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"react">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            worker_count: z.ZodOptional<z.ZodNumber>;
            enable_preview: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        orchestrator_plan: z.ZodOptional<z.ZodString>;
        workers: z.ZodArray<z.ZodObject<{
            worker_id: z.ZodString;
            title: z.ZodString;
            system_instruction: z.ZodOptional<z.ZodString>;
            user_prompt: z.ZodOptional<z.ZodString>;
            generated_content: z.ZodOptional<z.ZodString>;
            status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }>, "many">;
        preview_url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    };
    mode: "react";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    };
    mode: "react";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    };
}>;
export type ReactModeResponse = z.infer<typeof ReactModeResponseSchema>;
/**
 * Deepthink Mode Request Schema
 * Mode: Complex problem-solving through strategic decomposition
 */
export declare const DeepthinkModeRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"deepthink">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        strategy_count: z.ZodOptional<z.ZodNumber>;
        sub_strategy_count: z.ZodOptional<z.ZodNumber>;
        hypothesis_count: z.ZodOptional<z.ZodNumber>;
        enable_iterative_corrections: z.ZodOptional<z.ZodBoolean>;
        enable_red_team: z.ZodOptional<z.ZodBoolean>;
        red_team_aggressiveness: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    } | undefined;
}>;
export type DeepthinkModeRequest = z.infer<typeof DeepthinkModeRequestSchema>;
/**
 * Deepthink Mode Response Schema
 */
export declare const DeepthinkModeResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"deepthink">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"deepthink">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            strategy_count: z.ZodOptional<z.ZodNumber>;
            sub_strategy_count: z.ZodOptional<z.ZodNumber>;
            hypothesis_count: z.ZodOptional<z.ZodNumber>;
            enable_iterative_corrections: z.ZodOptional<z.ZodBoolean>;
            enable_red_team: z.ZodOptional<z.ZodBoolean>;
            red_team_aggressiveness: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        strategies: z.ZodArray<z.ZodObject<{
            strategy_id: z.ZodString;
            strategy_text: z.ZodString;
            sub_strategies: z.ZodArray<z.ZodObject<{
                sub_strategy_id: z.ZodString;
                sub_strategy_text: z.ZodString;
                solution: z.ZodOptional<z.ZodString>;
                critique: z.ZodOptional<z.ZodString>;
                refined_solution: z.ZodOptional<z.ZodString>;
                status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
            }, "strip", z.ZodTypeAny, {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }, {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }>, "many">;
        }, "strip", z.ZodTypeAny, {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }, {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }>, "many">;
        hypotheses: z.ZodOptional<z.ZodArray<z.ZodObject<{
            hypothesis_id: z.ZodString;
            hypothesis_text: z.ZodString;
            test_result: z.ZodOptional<z.ZodString>;
            status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }>, "many">>;
        best_solution: z.ZodOptional<z.ZodString>;
        red_team_evaluations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            strategy_id: z.ZodString;
            evaluation: z.ZodString;
            killed: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }, {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    };
    mode: "deepthink";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    };
    mode: "deepthink";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    };
}>;
export type DeepthinkModeResponse = z.infer<typeof DeepthinkModeResponseSchema>;
/**
 * Adaptive Deepthink Mode Request Schema
 * Mode: Full deepthink mode access to an agent
 */
export declare const AdaptiveDeepthinkRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"adaptive_deepthink">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        enable_streaming: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "adaptive_deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "adaptive_deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    } | undefined;
}>;
export type AdaptiveDeepthinkRequest = z.infer<typeof AdaptiveDeepthinkRequestSchema>;
/**
 * Adaptive Deepthink Mode Response Schema
 */
export declare const AdaptiveDeepthinkResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"adaptive_deepthink">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"adaptive_deepthink">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            conversation_id: z.ZodOptional<z.ZodString>;
            enable_streaming: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        tool_calls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            tool_name: z.ZodString;
            parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
            result: z.ZodAny;
        }, "strip", z.ZodTypeAny, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }>, "many">>;
        reasoning_trace: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    };
    mode: "adaptive_deepthink";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    };
    mode: "adaptive_deepthink";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    };
}>;
export type AdaptiveDeepthinkResponse = z.infer<typeof AdaptiveDeepthinkResponseSchema>;
/**
 * Agentic Mode Request Schema
 * Mode: General-purpose iterative refinement with tool-based manipulation
 */
export declare const AgenticModeRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"agentic">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        enable_diff_tools: z.ZodOptional<z.ZodBoolean>;
        enable_file_tools: z.ZodOptional<z.ZodBoolean>;
        enable_web_search: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "agentic";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "agentic";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    } | undefined;
}>;
export type AgenticModeRequest = z.infer<typeof AgenticModeRequestSchema>;
/**
 * Agentic Mode Response Schema
 */
export declare const AgenticModeResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"agentic">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"agentic">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            conversation_id: z.ZodOptional<z.ZodString>;
            enable_diff_tools: z.ZodOptional<z.ZodBoolean>;
            enable_file_tools: z.ZodOptional<z.ZodBoolean>;
            enable_web_search: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        tool_calls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            tool_name: z.ZodString;
            parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
            result: z.ZodAny;
        }, "strip", z.ZodTypeAny, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }>, "many">>;
        diff_operations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["search_and_replace", "delete", "insert_before", "insert_after"]>;
            params: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }, {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    };
    mode: "agentic";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    };
    mode: "agentic";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    };
}>;
export type AgenticModeResponse = z.infer<typeof AgenticModeResponseSchema>;
/**
 * Contextual Mode Request Schema
 * Mode: Iterative refinement through specialized agent collaboration
 */
export declare const ContextualModeRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"contextual">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        enable_memory_agent: z.ZodOptional<z.ZodBoolean>;
        memory_compression_threshold: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "contextual";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "contextual";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    } | undefined;
}>;
export type ContextualModeRequest = z.infer<typeof ContextualModeRequestSchema>;
/**
 * Contextual Mode Response Schema
 */
export declare const ContextualModeResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"contextual">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"contextual">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            conversation_id: z.ZodOptional<z.ZodString>;
            enable_memory_agent: z.ZodOptional<z.ZodBoolean>;
            memory_compression_threshold: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        agent_interactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            agent_type: z.ZodEnum<["main_generator", "iterative_agent", "memory_agent"]>;
            content: z.ZodString;
            timestamp_utc: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }, {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }>, "many">>;
        memory_compression_events: z.ZodOptional<z.ZodArray<z.ZodObject<{
            timestamp_utc: z.ZodString;
            compressed_message_count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            timestamp_utc: string;
            compressed_message_count: number;
        }, {
            timestamp_utc: string;
            compressed_message_count: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    };
    mode: "contextual";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    };
    mode: "contextual";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    };
}>;
export type ContextualModeResponse = z.infer<typeof ContextualModeResponseSchema>;
/**
 * Generative UI Mode Request Schema
 * Mode: Interactive UI development with user interaction capture
 */
export declare const GenerativeUIModeRequestSchema: z.ZodObject<{
    mode: z.ZodLiteral<"generative_ui">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        enable_interaction_capture: z.ZodOptional<z.ZodBoolean>;
        quality_threshold: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "generative_ui";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "generative_ui";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    } | undefined;
}>;
export type GenerativeUIModeRequest = z.infer<typeof GenerativeUIModeRequestSchema>;
/**
 * Generative UI Mode Response Schema
 */
export declare const GenerativeUIModeResponseSchema: z.ZodObject<{
    mode: z.ZodLiteral<"generative_ui">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"generative_ui">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            enable_interaction_capture: z.ZodOptional<z.ZodBoolean>;
            quality_threshold: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        ui_structure: z.ZodOptional<z.ZodString>;
        html_content: z.ZodOptional<z.ZodString>;
        css_content: z.ZodOptional<z.ZodString>;
        js_content: z.ZodOptional<z.ZodString>;
        quality_score: z.ZodOptional<z.ZodNumber>;
        interactions_captured: z.ZodOptional<z.ZodArray<z.ZodObject<{
            interaction_type: z.ZodEnum<["click", "input", "hover", "submit"]>;
            element_id: z.ZodString;
            timestamp_utc: z.ZodString;
            value: z.ZodOptional<z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }, {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        iteration_count?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    };
    mode: "generative_ui";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        iteration_count?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    };
    mode: "generative_ui";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    };
}>;
export type GenerativeUIModeResponse = z.infer<typeof GenerativeUIModeResponseSchema>;
/**
 * Union of all mode request types
 */
export declare const ICRModeRequestSchema: z.ZodDiscriminatedUnion<"mode", [z.ZodObject<{
    mode: z.ZodLiteral<"refine">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        evolution_mode: z.ZodOptional<z.ZodEnum<["novelty", "quality", "off"]>>;
        refinement_stages: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "refine";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "refine";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        evolution_mode?: "novelty" | "quality" | "off" | undefined;
        refinement_stages?: number | undefined;
    } | undefined;
}>, z.ZodObject<{
    mode: z.ZodLiteral<"react">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        worker_count: z.ZodOptional<z.ZodNumber>;
        enable_preview: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "react";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "react";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        worker_count?: number | undefined;
        enable_preview?: boolean | undefined;
    } | undefined;
}>, z.ZodObject<{
    mode: z.ZodLiteral<"deepthink">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        strategy_count: z.ZodOptional<z.ZodNumber>;
        sub_strategy_count: z.ZodOptional<z.ZodNumber>;
        hypothesis_count: z.ZodOptional<z.ZodNumber>;
        enable_iterative_corrections: z.ZodOptional<z.ZodBoolean>;
        enable_red_team: z.ZodOptional<z.ZodBoolean>;
        red_team_aggressiveness: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        strategy_count?: number | undefined;
        sub_strategy_count?: number | undefined;
        hypothesis_count?: number | undefined;
        enable_iterative_corrections?: boolean | undefined;
        enable_red_team?: boolean | undefined;
        red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
    } | undefined;
}>, z.ZodObject<{
    mode: z.ZodLiteral<"adaptive_deepthink">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        enable_streaming: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "adaptive_deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "adaptive_deepthink";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_streaming?: boolean | undefined;
    } | undefined;
}>, z.ZodObject<{
    mode: z.ZodLiteral<"agentic">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        enable_diff_tools: z.ZodOptional<z.ZodBoolean>;
        enable_file_tools: z.ZodOptional<z.ZodBoolean>;
        enable_web_search: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "agentic";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "agentic";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_diff_tools?: boolean | undefined;
        enable_file_tools?: boolean | undefined;
        enable_web_search?: boolean | undefined;
    } | undefined;
}>, z.ZodObject<{
    mode: z.ZodLiteral<"contextual">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        enable_memory_agent: z.ZodOptional<z.ZodBoolean>;
        memory_compression_threshold: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "contextual";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "contextual";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        conversation_id?: string | undefined;
        enable_memory_agent?: boolean | undefined;
        memory_compression_threshold?: number | undefined;
    } | undefined;
}>, z.ZodObject<{
    mode: z.ZodLiteral<"generative_ui">;
    prompt: z.ZodString;
    options: z.ZodOptional<z.ZodObject<{
        temperature: z.ZodOptional<z.ZodNumber>;
        top_p: z.ZodOptional<z.ZodNumber>;
        model_name: z.ZodOptional<z.ZodString>;
        provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
    } & {
        enable_interaction_capture: z.ZodOptional<z.ZodBoolean>;
        quality_threshold: z.ZodOptional<z.ZodNumber>;
        max_iterations: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    }, {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    }>>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "generative_ui";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    } | undefined;
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    prompt: string;
    mode: "generative_ui";
    options?: {
        provider?: "google" | "openai" | "anthropic" | undefined;
        max_iterations?: number | undefined;
        temperature?: number | undefined;
        top_p?: number | undefined;
        model_name?: string | undefined;
        enable_interaction_capture?: boolean | undefined;
        quality_threshold?: number | undefined;
    } | undefined;
}>]>;
export type ICRModeRequest = z.infer<typeof ICRModeRequestSchema>;
/**
 * Union of all mode response types
 */
export declare const ICRModeResponseSchema: z.ZodDiscriminatedUnion<"mode", [z.ZodObject<{
    mode: z.ZodLiteral<"refine">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"refine">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            evolution_mode: z.ZodOptional<z.ZodEnum<["novelty", "quality", "off"]>>;
            refinement_stages: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        iterations: z.ZodArray<z.ZodObject<{
            iteration_number: z.ZodNumber;
            content: z.ZodString;
            suggested_features: z.ZodOptional<z.ZodString>;
            bug_fixes: z.ZodOptional<z.ZodString>;
            status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }>, "many">;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
    };
    mode: "refine";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iterations: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            content: string;
            iteration_number: number;
            error?: string | undefined;
            suggested_features?: string | undefined;
            bug_fixes?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
    };
    mode: "refine";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "refine";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            evolution_mode?: "novelty" | "quality" | "off" | undefined;
            refinement_stages?: number | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"react">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"react">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            worker_count: z.ZodOptional<z.ZodNumber>;
            enable_preview: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        orchestrator_plan: z.ZodOptional<z.ZodString>;
        workers: z.ZodArray<z.ZodObject<{
            worker_id: z.ZodString;
            title: z.ZodString;
            system_instruction: z.ZodOptional<z.ZodString>;
            user_prompt: z.ZodOptional<z.ZodString>;
            generated_content: z.ZodOptional<z.ZodString>;
            status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
            error: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }>, "many">;
        preview_url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    };
    mode: "react";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        workers: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            title: string;
            worker_id: string;
            error?: string | undefined;
            system_instruction?: string | undefined;
            user_prompt?: string | undefined;
            generated_content?: string | undefined;
        }[];
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        orchestrator_plan?: string | undefined;
        preview_url?: string | undefined;
    };
    mode: "react";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "react";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            worker_count?: number | undefined;
            enable_preview?: boolean | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"deepthink">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"deepthink">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            strategy_count: z.ZodOptional<z.ZodNumber>;
            sub_strategy_count: z.ZodOptional<z.ZodNumber>;
            hypothesis_count: z.ZodOptional<z.ZodNumber>;
            enable_iterative_corrections: z.ZodOptional<z.ZodBoolean>;
            enable_red_team: z.ZodOptional<z.ZodBoolean>;
            red_team_aggressiveness: z.ZodOptional<z.ZodEnum<["low", "medium", "high"]>>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        strategies: z.ZodArray<z.ZodObject<{
            strategy_id: z.ZodString;
            strategy_text: z.ZodString;
            sub_strategies: z.ZodArray<z.ZodObject<{
                sub_strategy_id: z.ZodString;
                sub_strategy_text: z.ZodString;
                solution: z.ZodOptional<z.ZodString>;
                critique: z.ZodOptional<z.ZodString>;
                refined_solution: z.ZodOptional<z.ZodString>;
                status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
            }, "strip", z.ZodTypeAny, {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }, {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }>, "many">;
        }, "strip", z.ZodTypeAny, {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }, {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }>, "many">;
        hypotheses: z.ZodOptional<z.ZodArray<z.ZodObject<{
            hypothesis_id: z.ZodString;
            hypothesis_text: z.ZodString;
            test_result: z.ZodOptional<z.ZodString>;
            status: z.ZodEnum<["pending", "processing", "completed", "error", "cancelled"]>;
        }, "strip", z.ZodTypeAny, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }, {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }>, "many">>;
        best_solution: z.ZodOptional<z.ZodString>;
        red_team_evaluations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            strategy_id: z.ZodString;
            evaluation: z.ZodString;
            killed: z.ZodBoolean;
        }, "strip", z.ZodTypeAny, {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }, {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    };
    mode: "deepthink";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        strategies: {
            strategy_id: string;
            strategy_text: string;
            sub_strategies: {
                status: "error" | "processing" | "completed" | "pending" | "cancelled";
                sub_strategy_id: string;
                sub_strategy_text: string;
                solution?: string | undefined;
                critique?: string | undefined;
                refined_solution?: string | undefined;
            }[];
        }[];
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        hypotheses?: {
            status: "error" | "processing" | "completed" | "pending" | "cancelled";
            hypothesis_id: string;
            hypothesis_text: string;
            test_result?: string | undefined;
        }[] | undefined;
        best_solution?: string | undefined;
        red_team_evaluations?: {
            strategy_id: string;
            evaluation: string;
            killed: boolean;
        }[] | undefined;
    };
    mode: "deepthink";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            strategy_count?: number | undefined;
            sub_strategy_count?: number | undefined;
            hypothesis_count?: number | undefined;
            enable_iterative_corrections?: boolean | undefined;
            enable_red_team?: boolean | undefined;
            red_team_aggressiveness?: "high" | "medium" | "low" | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"adaptive_deepthink">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"adaptive_deepthink">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            conversation_id: z.ZodOptional<z.ZodString>;
            enable_streaming: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        tool_calls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            tool_name: z.ZodString;
            parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
            result: z.ZodAny;
        }, "strip", z.ZodTypeAny, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }>, "many">>;
        reasoning_trace: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    };
    mode: "adaptive_deepthink";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        reasoning_trace?: string | undefined;
    };
    mode: "adaptive_deepthink";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "adaptive_deepthink";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_streaming?: boolean | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"agentic">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"agentic">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            conversation_id: z.ZodOptional<z.ZodString>;
            enable_diff_tools: z.ZodOptional<z.ZodBoolean>;
            enable_file_tools: z.ZodOptional<z.ZodBoolean>;
            enable_web_search: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        tool_calls: z.ZodOptional<z.ZodArray<z.ZodObject<{
            tool_name: z.ZodString;
            parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
            result: z.ZodAny;
        }, "strip", z.ZodTypeAny, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }, {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }>, "many">>;
        diff_operations: z.ZodOptional<z.ZodArray<z.ZodObject<{
            type: z.ZodEnum<["search_and_replace", "delete", "insert_before", "insert_after"]>;
            params: z.ZodArray<z.ZodString, "many">;
        }, "strip", z.ZodTypeAny, {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }, {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    };
    mode: "agentic";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        tool_calls?: {
            parameters: Record<string, any>;
            tool_name: string;
            result?: any;
        }[] | undefined;
        diff_operations?: {
            type: "delete" | "search_and_replace" | "insert_before" | "insert_after";
            params: string[];
        }[] | undefined;
    };
    mode: "agentic";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "agentic";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_diff_tools?: boolean | undefined;
            enable_file_tools?: boolean | undefined;
            enable_web_search?: boolean | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"contextual">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"contextual">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            conversation_id: z.ZodOptional<z.ZodString>;
            enable_memory_agent: z.ZodOptional<z.ZodBoolean>;
            memory_compression_threshold: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        conversation_id: z.ZodOptional<z.ZodString>;
        agent_interactions: z.ZodOptional<z.ZodArray<z.ZodObject<{
            agent_type: z.ZodEnum<["main_generator", "iterative_agent", "memory_agent"]>;
            content: z.ZodString;
            timestamp_utc: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }, {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }>, "many">>;
        memory_compression_events: z.ZodOptional<z.ZodArray<z.ZodObject<{
            timestamp_utc: z.ZodString;
            compressed_message_count: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            timestamp_utc: string;
            compressed_message_count: number;
        }, {
            timestamp_utc: string;
            compressed_message_count: number;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    };
    mode: "contextual";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        iteration_count?: number | undefined;
        conversation_id?: string | undefined;
        agent_interactions?: {
            content: string;
            timestamp_utc: string;
            agent_type: "main_generator" | "iterative_agent" | "memory_agent";
        }[] | undefined;
        memory_compression_events?: {
            timestamp_utc: string;
            compressed_message_count: number;
        }[] | undefined;
    };
    mode: "contextual";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "contextual";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            conversation_id?: string | undefined;
            enable_memory_agent?: boolean | undefined;
            memory_compression_threshold?: number | undefined;
        } | undefined;
    };
}>, z.ZodObject<{
    mode: z.ZodLiteral<"generative_ui">;
    request: z.ZodObject<{
        mode: z.ZodLiteral<"generative_ui">;
        prompt: z.ZodString;
        options: z.ZodOptional<z.ZodObject<{
            temperature: z.ZodOptional<z.ZodNumber>;
            top_p: z.ZodOptional<z.ZodNumber>;
            model_name: z.ZodOptional<z.ZodString>;
            provider: z.ZodOptional<z.ZodEnum<["google", "openai", "anthropic"]>>;
        } & {
            enable_interaction_capture: z.ZodOptional<z.ZodBoolean>;
            quality_threshold: z.ZodOptional<z.ZodNumber>;
            max_iterations: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        }, {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        }>>;
        metadata: z.ZodObject<{
            correlation_id: z.ZodString;
            timestamp_utc: z.ZodString;
            source_service: z.ZodDefault<z.ZodString>;
            mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
            request_id: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        }, {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        }>;
    }, "strip", z.ZodTypeAny, {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    }, {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    }>;
    result: z.ZodObject<{
        success: z.ZodBoolean;
        content: z.ZodString;
        error: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodNumber;
        iteration_count: z.ZodDefault<z.ZodNumber>;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    } & {
        ui_structure: z.ZodOptional<z.ZodString>;
        html_content: z.ZodOptional<z.ZodString>;
        css_content: z.ZodOptional<z.ZodString>;
        js_content: z.ZodOptional<z.ZodString>;
        quality_score: z.ZodOptional<z.ZodNumber>;
        interactions_captured: z.ZodOptional<z.ZodArray<z.ZodObject<{
            interaction_type: z.ZodEnum<["click", "input", "hover", "submit"]>;
            element_id: z.ZodString;
            timestamp_utc: z.ZodString;
            value: z.ZodOptional<z.ZodAny>;
        }, "strip", z.ZodTypeAny, {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }, {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }>, "many">>;
    }, "strip", z.ZodTypeAny, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    }, {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        iteration_count?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    }>;
    metadata: z.ZodObject<{
        correlation_id: z.ZodString;
        timestamp_utc: z.ZodString;
        source_service: z.ZodDefault<z.ZodString>;
        mode: z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>;
        request_id: z.ZodOptional<z.ZodString>;
    } & {
        completed_at_utc: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    }, {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    metadata: {
        correlation_id: string;
        source_service: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        iteration_count: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    };
    mode: "generative_ui";
    request: {
        metadata: {
            correlation_id: string;
            source_service: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    };
}, {
    metadata: {
        correlation_id: string;
        mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
        timestamp_utc: string;
        completed_at_utc: string;
        source_service?: string | undefined;
        request_id?: string | undefined;
    };
    result: {
        success: boolean;
        content: string;
        execution_time_ms: number;
        error?: string | undefined;
        metadata?: Record<string, any> | undefined;
        quality_score?: number | undefined;
        iteration_count?: number | undefined;
        ui_structure?: string | undefined;
        html_content?: string | undefined;
        css_content?: string | undefined;
        js_content?: string | undefined;
        interactions_captured?: {
            timestamp_utc: string;
            interaction_type: "input" | "click" | "submit" | "hover";
            element_id: string;
            value?: any;
        }[] | undefined;
    };
    mode: "generative_ui";
    request: {
        metadata: {
            correlation_id: string;
            mode: "refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui";
            timestamp_utc: string;
            source_service?: string | undefined;
            request_id?: string | undefined;
        };
        prompt: string;
        mode: "generative_ui";
        options?: {
            provider?: "google" | "openai" | "anthropic" | undefined;
            max_iterations?: number | undefined;
            temperature?: number | undefined;
            top_p?: number | undefined;
            model_name?: string | undefined;
            enable_interaction_capture?: boolean | undefined;
            quality_threshold?: number | undefined;
        } | undefined;
    };
}>]>;
export type ICRModeResponse = z.infer<typeof ICRModeResponseSchema>;
/**
 * Health check request schema
 */
export declare const ICRHealthCheckRequestSchema: z.ZodObject<{
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    correlation_id?: string | undefined;
}, {
    correlation_id?: string | undefined;
}>;
export type ICRHealthCheckRequest = z.infer<typeof ICRHealthCheckRequestSchema>;
/**
 * Health check response schema
 */
export declare const ICRHealthCheckResponseSchema: z.ZodObject<{
    status: z.ZodEnum<["healthy", "degraded", "unhealthy"]>;
    version: z.ZodString;
    available_modes: z.ZodArray<z.ZodEnum<["refine", "react", "deepthink", "adaptive_deepthink", "agentic", "contextual", "generative_ui"]>, "many">;
    timestamp_utc: z.ZodString;
    uptime_seconds: z.ZodNumber;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    version: string;
    status: "healthy" | "unhealthy" | "degraded";
    timestamp_utc: string;
    available_modes: ("refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui")[];
    uptime_seconds: number;
    metadata?: Record<string, any> | undefined;
}, {
    version: string;
    status: "healthy" | "unhealthy" | "degraded";
    timestamp_utc: string;
    available_modes: ("refine" | "react" | "deepthink" | "adaptive_deepthink" | "agentic" | "contextual" | "generative_ui")[];
    uptime_seconds: number;
    metadata?: Record<string, any> | undefined;
}>;
export type ICRHealthCheckResponse = z.infer<typeof ICRHealthCheckResponseSchema>;
//# sourceMappingURL=icr-canonical.d.ts.map