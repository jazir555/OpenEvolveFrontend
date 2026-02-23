/**
 * MAKER Canonical Schema - Anti-Corruption Layer (ACL)
 *
 * Federation Constitution Compliant Canonical Schema for MAKER Engine integration.
 * All external data must be transformed to/from this format.
 *
 * Law 6: All timestamps are UTC ISO-8601 strings
 */
import { z } from 'zod';
/**
 * Voting Mode Enum
 */
export declare const VotingModeSchema: z.ZodEnum<["simple", "k_ahead", "weighted", "consensus"]>;
export type VotingMode = z.infer<typeof VotingModeSchema>;
/**
 * Red Flag Severity Enum
 */
export declare const RedFlagSeveritySchema: z.ZodEnum<["low", "medium", "high", "critical"]>;
export type RedFlagSeverity = z.infer<typeof RedFlagSeveritySchema>;
/**
 * MAKER Configuration Schema
 */
export declare const MakerConfigSchema: z.ZodObject<{
    k_min: z.ZodDefault<z.ZodNumber>;
    k_max: z.ZodDefault<z.ZodNumber>;
    max_votes_per_step: z.ZodDefault<z.ZodNumber>;
    max_steps: z.ZodDefault<z.ZodNumber>;
    timeout_seconds: z.ZodDefault<z.ZodNumber>;
    checkpoint_interval: z.ZodDefault<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    checkpoint_interval: number;
    k_min: number;
    k_max: number;
    max_votes_per_step: number;
    max_steps: number;
    timeout_seconds: number;
}, {
    checkpoint_interval?: number | undefined;
    k_min?: number | undefined;
    k_max?: number | undefined;
    max_votes_per_step?: number | undefined;
    max_steps?: number | undefined;
    timeout_seconds?: number | undefined;
}>;
export type MakerConfig = z.infer<typeof MakerConfigSchema>;
/**
 * MAKER Step Schema
 */
export declare const MakerStepSchema: z.ZodObject<{
    step_id: z.ZodString;
    prompt_template: z.ZodString;
    task_type: z.ZodDefault<z.ZodString>;
    priority: z.ZodDefault<z.ZodNumber>;
    system_prompt: z.ZodOptional<z.ZodString>;
    expected_schema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    stop_sequences: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    step_id: string;
    priority: number;
    task_type: string;
    prompt_template: string;
    metadata?: Record<string, any> | undefined;
    system_prompt?: string | undefined;
    expected_schema?: Record<string, any> | undefined;
    stop_sequences?: string[] | undefined;
}, {
    step_id: string;
    prompt_template: string;
    metadata?: Record<string, any> | undefined;
    priority?: number | undefined;
    system_prompt?: string | undefined;
    task_type?: string | undefined;
    expected_schema?: Record<string, any> | undefined;
    stop_sequences?: string[] | undefined;
}>;
export type MakerStep = z.infer<typeof MakerStepSchema>;
/**
 * Agent Vote Schema
 */
export declare const AgentVoteSchema: z.ZodObject<{
    agent_id: z.ZodString;
    vote: z.ZodAny;
    raw_text: z.ZodString;
    timestamp: z.ZodString;
    red_flags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    agent_id: string;
    raw_text: string;
    red_flags?: string[] | undefined;
    vote?: any;
}, {
    timestamp: string;
    agent_id: string;
    raw_text: string;
    red_flags?: string[] | undefined;
    vote?: any;
}>;
export type AgentVote = z.infer<typeof AgentVoteSchema>;
/**
 * MAKER Run Result Schema
 */
export declare const MakerRunResultSchema: z.ZodObject<{
    success: z.ZodBoolean;
    steps_completed: z.ZodNumber;
    votes_cast: z.ZodNumber;
    red_flags_detected: z.ZodNumber;
    final_action: z.ZodOptional<z.ZodAny>;
    agent_votes: z.ZodOptional<z.ZodArray<z.ZodObject<{
        agent_id: z.ZodString;
        vote: z.ZodAny;
        raw_text: z.ZodString;
        timestamp: z.ZodString;
        red_flags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        agent_id: string;
        raw_text: string;
        red_flags?: string[] | undefined;
        vote?: any;
    }, {
        timestamp: string;
        agent_id: string;
        raw_text: string;
        red_flags?: string[] | undefined;
        vote?: any;
    }>, "many">>;
    red_flags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    metrics: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    terminated_reason: z.ZodString;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
    execution_time_ms: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    success: boolean;
    steps_completed: number;
    votes_cast: number;
    red_flags_detected: number;
    terminated_reason: string;
    correlation_id?: string | undefined;
    metrics?: Record<string, any> | undefined;
    red_flags?: string[] | undefined;
    execution_time_ms?: number | undefined;
    final_action?: any;
    agent_votes?: {
        timestamp: string;
        agent_id: string;
        raw_text: string;
        red_flags?: string[] | undefined;
        vote?: any;
    }[] | undefined;
}, {
    timestamp: string;
    success: boolean;
    steps_completed: number;
    votes_cast: number;
    red_flags_detected: number;
    terminated_reason: string;
    correlation_id?: string | undefined;
    metrics?: Record<string, any> | undefined;
    red_flags?: string[] | undefined;
    execution_time_ms?: number | undefined;
    final_action?: any;
    agent_votes?: {
        timestamp: string;
        agent_id: string;
        raw_text: string;
        red_flags?: string[] | undefined;
        vote?: any;
    }[] | undefined;
}>;
export type MakerRunResult = z.infer<typeof MakerRunResultSchema>;
/**
 * Validation Functions
 */
export declare function validateMakerConfig(data: unknown): {
    success: boolean;
    data?: MakerConfig;
    errors?: string[];
};
export declare function validateMakerStep(data: unknown): {
    success: boolean;
    data?: MakerStep;
    errors?: string[];
};
export declare function validateMakerRunResult(data: unknown): {
    success: boolean;
    data?: MakerRunResult;
    errors?: string[];
};
/**
 * Type Guards
 */
export declare function isMakerConfig(data: unknown): data is MakerConfig;
export declare function isMakerStep(data: unknown): data is MakerStep;
/**
 * Example Usage
 */
export declare const MakerExamples: {
    config: MakerConfig;
    step: MakerStep;
    result: MakerRunResult;
};
//# sourceMappingURL=maker-canonical.d.ts.map