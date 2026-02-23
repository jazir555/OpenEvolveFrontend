"use strict";
/**
 * MAKER Canonical Schema - Anti-Corruption Layer (ACL)
 *
 * Federation Constitution Compliant Canonical Schema for MAKER Engine integration.
 * All external data must be transformed to/from this format.
 *
 * Law 6: All timestamps are UTC ISO-8601 strings
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MakerExamples = exports.MakerRunResultSchema = exports.AgentVoteSchema = exports.MakerStepSchema = exports.MakerConfigSchema = exports.RedFlagSeveritySchema = exports.VotingModeSchema = void 0;
exports.validateMakerConfig = validateMakerConfig;
exports.validateMakerStep = validateMakerStep;
exports.validateMakerRunResult = validateMakerRunResult;
exports.isMakerConfig = isMakerConfig;
exports.isMakerStep = isMakerStep;
const zod_1 = require("zod");
/**
 * Voting Mode Enum
 */
exports.VotingModeSchema = zod_1.z.enum([
    'simple',
    'k_ahead',
    'weighted',
    'consensus',
]);
/**
 * Red Flag Severity Enum
 */
exports.RedFlagSeveritySchema = zod_1.z.enum([
    'low',
    'medium',
    'high',
    'critical',
]);
/**
 * MAKER Configuration Schema
 */
exports.MakerConfigSchema = zod_1.z.object({
    k_min: zod_1.z.number().int().min(1).default(2),
    k_max: zod_1.z.number().int().min(1).default(8),
    max_votes_per_step: zod_1.z.number().int().min(1).default(60),
    max_steps: zod_1.z.number().int().min(1).default(1000),
    timeout_seconds: zod_1.z.number().int().min(1).default(90),
    checkpoint_interval: zod_1.z.number().int().min(1).default(25),
});
/**
 * MAKER Step Schema
 */
exports.MakerStepSchema = zod_1.z.object({
    step_id: zod_1.z.string().min(1, 'Step ID cannot be empty'),
    prompt_template: zod_1.z.string().min(1, 'Prompt template cannot be empty'),
    task_type: zod_1.z.string().default('general'),
    priority: zod_1.z.number().int().min(0).default(0),
    system_prompt: zod_1.z.string().optional(),
    expected_schema: zod_1.z.record(zod_1.z.any()).optional(),
    stop_sequences: zod_1.z.array(zod_1.z.string()).optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Agent Vote Schema
 */
exports.AgentVoteSchema = zod_1.z.object({
    agent_id: zod_1.z.string(),
    vote: zod_1.z.any(),
    raw_text: zod_1.z.string(),
    timestamp: zod_1.z.string().datetime('UTC timestamp required'),
    red_flags: zod_1.z.array(zod_1.z.string()).optional(),
});
/**
 * MAKER Run Result Schema
 */
exports.MakerRunResultSchema = zod_1.z.object({
    success: zod_1.z.boolean(),
    steps_completed: zod_1.z.number().int().min(0),
    votes_cast: zod_1.z.number().int().min(0),
    red_flags_detected: zod_1.z.number().int().min(0),
    final_action: zod_1.z.any().optional(),
    agent_votes: zod_1.z.array(exports.AgentVoteSchema).optional(),
    red_flags: zod_1.z.array(zod_1.z.string()).optional(),
    metrics: zod_1.z.record(zod_1.z.any()).optional(),
    terminated_reason: zod_1.z.string(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime('UTC timestamp required'),
    execution_time_ms: zod_1.z.number().int().min(0).optional(),
});
/**
 * Validation Functions
 */
function validateMakerConfig(data) {
    const result = exports.MakerConfigSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateMakerStep(data) {
    const result = exports.MakerStepSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateMakerRunResult(data) {
    const result = exports.MakerRunResultSchema.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
/**
 * Type Guards
 */
function isMakerConfig(data) {
    return typeof data === 'object' && data !== null &&
        'k_min' in data && 'k_max' in data;
}
function isMakerStep(data) {
    return typeof data === 'object' && data !== null &&
        'step_id' in data && 'prompt_template' in data;
}
/**
 * Example Usage
 */
exports.MakerExamples = {
    config: {
        k_min: 2,
        k_max: 7,
        max_votes_per_step: 30,
        max_steps: 100,
        timeout_seconds: 60,
        checkpoint_interval: 20,
    },
    step: {
        step_id: 'maker-step-001',
        prompt_template: 'Analyze this: {state}',
        task_type: 'analysis',
        priority: 1,
        system_prompt: 'You are a helpful assistant',
        expected_schema: { type: 'object' },
        metadata: { domain: 'general' },
    },
    result: {
        success: true,
        steps_completed: 5,
        votes_cast: 15,
        red_flags_detected: 0,
        final_action: { action: 'continue', reason: 'Consensus reached' },
        agent_votes: [],
        red_flags: [],
        metrics: { total_votes: 15, unique_agents: 3 },
        terminated_reason: 'completed',
        timestamp: '2025-02-17T12:30:45.000Z',
        execution_time_ms: 1500,
    },
};
//# sourceMappingURL=maker-canonical.js.map