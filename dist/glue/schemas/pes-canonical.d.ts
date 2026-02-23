/**
 * PES Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Plan-Execute-Summarize (PES)
 * pattern interactions. PES is a fundamental paradigm in AI problem-solving where:
 * 1. PLAN: Generate a strategy for solving the problem
 * 2. EXECUTE: Carry out the plan to produce a solution
 * 3. SUMMARIZE: Reflect on the result and extract insights
 *
 * This is the generic PES schema that can be used by any system implementing
 * the PES pattern, including LoongFlow and custom hybrid workflows.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for PES data in the glue layer.
 * Do not pass raw PES API responses between services.
 */
import { z } from 'zod';
/**
 * Problem Type Enum
 *
 * Categorizes the type of problem being solved.
 */
export declare const ProblemType: z.ZodEnum<["optimization", "reasoning", "generation", "validation", "classification", "prediction", "synthesis"]>;
export type ProblemType = z.infer<typeof ProblemType>;
/**
 * Execution Step Type Enum
 *
 * Defines the type of each step in an execution plan.
 */
export declare const ExecutionStepType: z.ZodEnum<["plan", "execute", "summarize", "validate", "transform", "aggregate"]>;
export type ExecutionStepType = z.infer<typeof ExecutionStepType>;
/**
 * Execution State Enum
 *
 * Tracks the state of an execution workflow.
 */
export declare const ExecutionState: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "timeout", "paused"]>;
export type ExecutionState = z.infer<typeof ExecutionState>;
/**
 * ============================================================================
 * CORE PES SCHEMAS
 * ============================================================================
 */
/**
 * Problem Schema
 *
 * Represents a problem to be solved using the PES paradigm.
 * This is the input to any PES-based workflow.
 */
export declare const Problem: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["optimization", "reasoning", "generation", "validation", "classification", "prediction", "synthesis"]>;
    description: z.ZodString;
    context: z.ZodRecord<z.ZodString, z.ZodAny>;
    constraints: z.ZodArray<z.ZodString, "many">;
    success_criteria: z.ZodArray<z.ZodString, "many">;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
    priority: z.ZodOptional<z.ZodNumber>;
    tags: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    type: "classification" | "validation" | "synthesis" | "optimization" | "reasoning" | "generation" | "prediction";
    id: string;
    constraints: string[];
    created_at: string;
    description: string;
    context: Record<string, any>;
    success_criteria: string[];
    metadata?: Record<string, any> | undefined;
    priority?: number | undefined;
    tags?: string[] | undefined;
}, {
    type: "classification" | "validation" | "synthesis" | "optimization" | "reasoning" | "generation" | "prediction";
    id: string;
    constraints: string[];
    created_at: string;
    description: string;
    context: Record<string, any>;
    success_criteria: string[];
    metadata?: Record<string, any> | undefined;
    priority?: number | undefined;
    tags?: string[] | undefined;
}>;
export type Problem = z.infer<typeof Problem>;
/**
 * Execution Step Schema
 *
 * Represents a single step in an execution plan.
 */
export declare const ExecutionStep: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodEnum<["plan", "execute", "summarize", "validate", "transform", "aggregate"]>;
    description: z.ZodString;
    worker_type: z.ZodString;
    parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
    timeout_ms: z.ZodNumber;
    retry_config: z.ZodOptional<z.ZodObject<{
        max_retries: z.ZodOptional<z.ZodNumber>;
        backoff_ms: z.ZodOptional<z.ZodNumber>;
        max_backoff_ms: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        max_retries?: number | undefined;
        backoff_ms?: number | undefined;
        max_backoff_ms?: number | undefined;
    }, {
        max_retries?: number | undefined;
        backoff_ms?: number | undefined;
        max_backoff_ms?: number | undefined;
    }>>;
    depends_on: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    type: "summarize" | "validate" | "plan" | "transform" | "execute" | "aggregate";
    id: string;
    description: string;
    parameters: Record<string, any>;
    worker_type: string;
    retry_config?: {
        max_retries?: number | undefined;
        backoff_ms?: number | undefined;
        max_backoff_ms?: number | undefined;
    } | undefined;
    depends_on?: string[] | undefined;
}, {
    timeout_ms: number;
    type: "summarize" | "validate" | "plan" | "transform" | "execute" | "aggregate";
    id: string;
    description: string;
    parameters: Record<string, any>;
    worker_type: string;
    retry_config?: {
        max_retries?: number | undefined;
        backoff_ms?: number | undefined;
        max_backoff_ms?: number | undefined;
    } | undefined;
    depends_on?: string[] | undefined;
}>;
export type ExecutionStep = z.infer<typeof ExecutionStep>;
/**
 * Execution Plan Schema
 *
 * Represents a complete plan for solving a problem.
 */
export declare const ExecutionPlan: z.ZodObject<{
    id: z.ZodString;
    problem_id: z.ZodString;
    steps: z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        type: z.ZodEnum<["plan", "execute", "summarize", "validate", "transform", "aggregate"]>;
        description: z.ZodString;
        worker_type: z.ZodString;
        parameters: z.ZodRecord<z.ZodString, z.ZodAny>;
        timeout_ms: z.ZodNumber;
        retry_config: z.ZodOptional<z.ZodObject<{
            max_retries: z.ZodOptional<z.ZodNumber>;
            backoff_ms: z.ZodOptional<z.ZodNumber>;
            max_backoff_ms: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
            max_backoff_ms?: number | undefined;
        }, {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
            max_backoff_ms?: number | undefined;
        }>>;
        depends_on: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        timeout_ms: number;
        type: "summarize" | "validate" | "plan" | "transform" | "execute" | "aggregate";
        id: string;
        description: string;
        parameters: Record<string, any>;
        worker_type: string;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
            max_backoff_ms?: number | undefined;
        } | undefined;
        depends_on?: string[] | undefined;
    }, {
        timeout_ms: number;
        type: "summarize" | "validate" | "plan" | "transform" | "execute" | "aggregate";
        id: string;
        description: string;
        parameters: Record<string, any>;
        worker_type: string;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
            max_backoff_ms?: number | undefined;
        } | undefined;
        depends_on?: string[] | undefined;
    }>, "many">;
    dependencies: z.ZodOptional<z.ZodArray<z.ZodObject<{
        step_id: z.ZodString;
        depends_on: z.ZodArray<z.ZodString, "many">;
    }, "strip", z.ZodTypeAny, {
        step_id: string;
        depends_on: string[];
    }, {
        step_id: string;
        depends_on: string[];
    }>, "many">>;
    resource_requirements: z.ZodOptional<z.ZodObject<{
        estimated_duration_ms: z.ZodOptional<z.ZodNumber>;
        required_workers: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        memory_mb: z.ZodOptional<z.ZodNumber>;
        cpu_cores: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        estimated_duration_ms?: number | undefined;
        required_workers?: string[] | undefined;
        memory_mb?: number | undefined;
        cpu_cores?: number | undefined;
    }, {
        estimated_duration_ms?: number | undefined;
        required_workers?: string[] | undefined;
        memory_mb?: number | undefined;
        cpu_cores?: number | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
    planner_type: z.ZodOptional<z.ZodString>;
    confidence_score: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    id: string;
    steps: {
        timeout_ms: number;
        type: "summarize" | "validate" | "plan" | "transform" | "execute" | "aggregate";
        id: string;
        description: string;
        parameters: Record<string, any>;
        worker_type: string;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
            max_backoff_ms?: number | undefined;
        } | undefined;
        depends_on?: string[] | undefined;
    }[];
    created_at: string;
    problem_id: string;
    metadata?: Record<string, any> | undefined;
    dependencies?: {
        step_id: string;
        depends_on: string[];
    }[] | undefined;
    confidence_score?: number | undefined;
    resource_requirements?: {
        estimated_duration_ms?: number | undefined;
        required_workers?: string[] | undefined;
        memory_mb?: number | undefined;
        cpu_cores?: number | undefined;
    } | undefined;
    planner_type?: string | undefined;
}, {
    id: string;
    steps: {
        timeout_ms: number;
        type: "summarize" | "validate" | "plan" | "transform" | "execute" | "aggregate";
        id: string;
        description: string;
        parameters: Record<string, any>;
        worker_type: string;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
            max_backoff_ms?: number | undefined;
        } | undefined;
        depends_on?: string[] | undefined;
    }[];
    created_at: string;
    problem_id: string;
    metadata?: Record<string, any> | undefined;
    dependencies?: {
        step_id: string;
        depends_on: string[];
    }[] | undefined;
    confidence_score?: number | undefined;
    resource_requirements?: {
        estimated_duration_ms?: number | undefined;
        required_workers?: string[] | undefined;
        memory_mb?: number | undefined;
        cpu_cores?: number | undefined;
    } | undefined;
    planner_type?: string | undefined;
}>;
export type ExecutionPlan = z.infer<typeof ExecutionPlan>;
/**
 * Log Entry Schema
 *
 * Represents a single log entry during execution.
 */
export declare const LogEntry: z.ZodObject<{
    timestamp: z.ZodString;
    level: z.ZodEnum<["debug", "info", "warn", "error", "fatal"]>;
    message: z.ZodString;
    context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    step_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    level: "debug" | "info" | "warn" | "error" | "fatal";
    message: string;
    correlation_id?: string | undefined;
    step_id?: string | undefined;
    context?: Record<string, any> | undefined;
}, {
    timestamp: string;
    level: "debug" | "info" | "warn" | "error" | "fatal";
    message: string;
    correlation_id?: string | undefined;
    step_id?: string | undefined;
    context?: Record<string, any> | undefined;
}>;
export type LogEntry = z.infer<typeof LogEntry>;
/**
 * Artifact Schema
 *
 * Represents an artifact produced during execution.
 */
export declare const Artifact: z.ZodObject<{
    type: z.ZodString;
    uri: z.ZodString;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    size_bytes: z.ZodOptional<z.ZodNumber>;
    checksum: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: string;
    uri: string;
    metadata?: Record<string, any> | undefined;
    size_bytes?: number | undefined;
    checksum?: string | undefined;
}, {
    type: string;
    uri: string;
    metadata?: Record<string, any> | undefined;
    size_bytes?: number | undefined;
    checksum?: string | undefined;
}>;
export type Artifact = z.infer<typeof Artifact>;
/**
 * Execution Metrics Schema
 *
 * Represents metrics collected during execution.
 */
export declare const ExecutionMetrics: z.ZodObject<{
    duration_ms: z.ZodNumber;
    success: z.ZodBoolean;
    score: z.ZodOptional<z.ZodNumber>;
    error_count: z.ZodNumber;
    steps_completed: z.ZodOptional<z.ZodNumber>;
    steps_failed: z.ZodOptional<z.ZodNumber>;
    memory_peak_mb: z.ZodOptional<z.ZodNumber>;
    cpu_time_ms: z.ZodOptional<z.ZodNumber>;
    iterations: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    duration_ms: number;
    success: boolean;
    error_count: number;
    steps_completed?: number | undefined;
    steps_failed?: number | undefined;
    score?: number | undefined;
    iterations?: number | undefined;
    memory_peak_mb?: number | undefined;
    cpu_time_ms?: number | undefined;
}, {
    duration_ms: number;
    success: boolean;
    error_count: number;
    steps_completed?: number | undefined;
    steps_failed?: number | undefined;
    score?: number | undefined;
    iterations?: number | undefined;
    memory_peak_mb?: number | undefined;
    cpu_time_ms?: number | undefined;
}>;
export type ExecutionMetrics = z.infer<typeof ExecutionMetrics>;
/**
 * Execution Result Schema
 *
 * Represents the result of executing a plan.
 */
export declare const ExecutionResult: z.ZodObject<{
    id: z.ZodString;
    problem_id: z.ZodString;
    plan_id: z.ZodString;
    state: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "timeout", "paused"]>;
    outputs: z.ZodRecord<z.ZodString, z.ZodAny>;
    metrics: z.ZodObject<{
        duration_ms: z.ZodNumber;
        success: z.ZodBoolean;
        score: z.ZodOptional<z.ZodNumber>;
        error_count: z.ZodNumber;
        steps_completed: z.ZodOptional<z.ZodNumber>;
        steps_failed: z.ZodOptional<z.ZodNumber>;
        memory_peak_mb: z.ZodOptional<z.ZodNumber>;
        cpu_time_ms: z.ZodOptional<z.ZodNumber>;
        iterations: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        duration_ms: number;
        success: boolean;
        error_count: number;
        steps_completed?: number | undefined;
        steps_failed?: number | undefined;
        score?: number | undefined;
        iterations?: number | undefined;
        memory_peak_mb?: number | undefined;
        cpu_time_ms?: number | undefined;
    }, {
        duration_ms: number;
        success: boolean;
        error_count: number;
        steps_completed?: number | undefined;
        steps_failed?: number | undefined;
        score?: number | undefined;
        iterations?: number | undefined;
        memory_peak_mb?: number | undefined;
        cpu_time_ms?: number | undefined;
    }>;
    artifacts: z.ZodOptional<z.ZodArray<z.ZodObject<{
        type: z.ZodString;
        uri: z.ZodString;
        metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        size_bytes: z.ZodOptional<z.ZodNumber>;
        checksum: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        uri: string;
        metadata?: Record<string, any> | undefined;
        size_bytes?: number | undefined;
        checksum?: string | undefined;
    }, {
        type: string;
        uri: string;
        metadata?: Record<string, any> | undefined;
        size_bytes?: number | undefined;
        checksum?: string | undefined;
    }>, "many">>;
    logs: z.ZodOptional<z.ZodArray<z.ZodObject<{
        timestamp: z.ZodString;
        level: z.ZodEnum<["debug", "info", "warn", "error", "fatal"]>;
        message: z.ZodString;
        context: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        correlation_id: z.ZodOptional<z.ZodString>;
        step_id: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        timestamp: string;
        level: "debug" | "info" | "warn" | "error" | "fatal";
        message: string;
        correlation_id?: string | undefined;
        step_id?: string | undefined;
        context?: Record<string, any> | undefined;
    }, {
        timestamp: string;
        level: "debug" | "info" | "warn" | "error" | "fatal";
        message: string;
        correlation_id?: string | undefined;
        step_id?: string | undefined;
        context?: Record<string, any> | undefined;
    }>, "many">>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        stack_trace: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        stack_trace?: string | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        stack_trace?: string | undefined;
    }>>;
    created_at: z.ZodString;
    completed_at: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    state: "running" | "completed" | "failed" | "pending" | "cancelled" | "timeout" | "paused";
    metrics: {
        duration_ms: number;
        success: boolean;
        error_count: number;
        steps_completed?: number | undefined;
        steps_failed?: number | undefined;
        score?: number | undefined;
        iterations?: number | undefined;
        memory_peak_mb?: number | undefined;
        cpu_time_ms?: number | undefined;
    };
    id: string;
    created_at: string;
    problem_id: string;
    plan_id: string;
    outputs: Record<string, any>;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        stack_trace?: string | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    completed_at?: string | undefined;
    logs?: {
        timestamp: string;
        level: "debug" | "info" | "warn" | "error" | "fatal";
        message: string;
        correlation_id?: string | undefined;
        step_id?: string | undefined;
        context?: Record<string, any> | undefined;
    }[] | undefined;
    artifacts?: {
        type: string;
        uri: string;
        metadata?: Record<string, any> | undefined;
        size_bytes?: number | undefined;
        checksum?: string | undefined;
    }[] | undefined;
}, {
    state: "running" | "completed" | "failed" | "pending" | "cancelled" | "timeout" | "paused";
    metrics: {
        duration_ms: number;
        success: boolean;
        error_count: number;
        steps_completed?: number | undefined;
        steps_failed?: number | undefined;
        score?: number | undefined;
        iterations?: number | undefined;
        memory_peak_mb?: number | undefined;
        cpu_time_ms?: number | undefined;
    };
    id: string;
    created_at: string;
    problem_id: string;
    plan_id: string;
    outputs: Record<string, any>;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        stack_trace?: string | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    completed_at?: string | undefined;
    logs?: {
        timestamp: string;
        level: "debug" | "info" | "warn" | "error" | "fatal";
        message: string;
        correlation_id?: string | undefined;
        step_id?: string | undefined;
        context?: Record<string, any> | undefined;
    }[] | undefined;
    artifacts?: {
        type: string;
        uri: string;
        metadata?: Record<string, any> | undefined;
        size_bytes?: number | undefined;
        checksum?: string | undefined;
    }[] | undefined;
}>;
export type ExecutionResult = z.infer<typeof ExecutionResult>;
/**
 * Performance Assessment Schema
 *
 * Represents an assessment of execution performance.
 */
export declare const PerformanceAssessment: z.ZodObject<{
    success_criteria_met: z.ZodBoolean;
    quality_score: z.ZodNumber;
    efficiency_score: z.ZodNumber;
    criteria_scores: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
    improvement_suggestions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
}, "strip", z.ZodTypeAny, {
    quality_score: number;
    success_criteria_met: boolean;
    efficiency_score: number;
    criteria_scores?: Record<string, number> | undefined;
    improvement_suggestions?: string[] | undefined;
}, {
    quality_score: number;
    success_criteria_met: boolean;
    efficiency_score: number;
    criteria_scores?: Record<string, number> | undefined;
    improvement_suggestions?: string[] | undefined;
}>;
export type PerformanceAssessment = z.infer<typeof PerformanceAssessment>;
/**
 * Summary Schema
 *
 * Represents a summary/evaluation of an execution result.
 * This is the "Summarize" phase of the PES paradigm.
 */
export declare const Summary: z.ZodObject<{
    id: z.ZodString;
    result_id: z.ZodString;
    evaluation: z.ZodString;
    insights: z.ZodArray<z.ZodString, "many">;
    performance_assessment: z.ZodObject<{
        success_criteria_met: z.ZodBoolean;
        quality_score: z.ZodNumber;
        efficiency_score: z.ZodNumber;
        criteria_scores: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodNumber>>;
        improvement_suggestions: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        quality_score: number;
        success_criteria_met: boolean;
        efficiency_score: number;
        criteria_scores?: Record<string, number> | undefined;
        improvement_suggestions?: string[] | undefined;
    }, {
        quality_score: number;
        success_criteria_met: boolean;
        efficiency_score: number;
        criteria_scores?: Record<string, number> | undefined;
        improvement_suggestions?: string[] | undefined;
    }>;
    recommendations: z.ZodArray<z.ZodString, "many">;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    created_at: z.ZodString;
    summarizer_type: z.ZodOptional<z.ZodString>;
    confidence_score: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    id: string;
    insights: string[];
    created_at: string;
    recommendations: string[];
    evaluation: string;
    result_id: string;
    performance_assessment: {
        quality_score: number;
        success_criteria_met: boolean;
        efficiency_score: number;
        criteria_scores?: Record<string, number> | undefined;
        improvement_suggestions?: string[] | undefined;
    };
    metadata?: Record<string, any> | undefined;
    confidence_score?: number | undefined;
    summarizer_type?: string | undefined;
}, {
    id: string;
    insights: string[];
    created_at: string;
    recommendations: string[];
    evaluation: string;
    result_id: string;
    performance_assessment: {
        quality_score: number;
        success_criteria_met: boolean;
        efficiency_score: number;
        criteria_scores?: Record<string, number> | undefined;
        improvement_suggestions?: string[] | undefined;
    };
    metadata?: Record<string, any> | undefined;
    confidence_score?: number | undefined;
    summarizer_type?: string | undefined;
}>;
export type Summary = z.infer<typeof Summary>;
/**
 * ============================================================================
 * TRANSFORMATION FUNCTIONS
 * ============================================================================
 */
/**
 * Transform raw PES data to canonical Problem format
 */
export declare function transformProblemToCanonical(rawProblem: any, generatedId?: string): Problem;
/**
 * Transform canonical Problem to external format
 */
export declare function transformCanonicalToProblem(canonical: Problem): Record<string, any>;
/**
 * Transform raw execution result to canonical ExecutionResult format
 */
export declare function transformExecutionResultToCanonical(rawResult: any, generatedId?: string): ExecutionResult;
/**
 * Transform canonical Summary to external format
 */
export declare function transformCanonicalToSummary(canonical: Summary): Record<string, any>;
/**
 * ============================================================================
 * VALIDATION FUNCTIONS
 * ============================================================================
 */
/**
 * Validate a Problem object
 */
export declare function validateProblem(data: unknown): {
    success: boolean;
    data?: Problem;
    errors?: string[];
};
/**
 * Validate an ExecutionPlan object
 */
export declare function validateExecutionPlan(data: unknown): {
    success: boolean;
    data?: ExecutionPlan;
    errors?: string[];
};
/**
 * Validate an ExecutionResult object
 */
export declare function validateExecutionResult(data: unknown): {
    success: boolean;
    data?: ExecutionResult;
    errors?: string[];
};
/**
 * Validate a Summary object
 */
export declare function validateSummary(data: unknown): {
    success: boolean;
    data?: Summary;
    errors?: string[];
};
/**
 * ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================
 */
/**
 * Create a UTC timestamp (Law of UTC)
 */
export declare function createPESUTCTimestamp(): string;
/**
 * Create a correlation ID
 */
export declare function createPESCorrelationId(): string;
/**
 * Check if data is a valid Problem
 */
export declare function isProblem(data: unknown): data is Problem;
/**
 * Check if data is a valid ExecutionResult
 */
export declare function isExecutionResult(data: unknown): data is ExecutionResult;
/**
 * Check if data is a valid Summary
 */
export declare function isSummary(data: unknown): data is Summary;
/**
 * ============================================================================
 * EXAMPLES
 * ============================================================================
 */
/**
 * Example usage:
 * @example
 * ```typescript
 * import { validateProblem, createPESUTCTimestamp } from './pes-canonical';
 *
 * const problemData = {
 *   type: 'optimization',
 *   description: 'Find optimal hyperparameters for ML model',
 *   context: { model: 'neural_network', dataset: 'imagenet' },
 *   constraints: ['max_epochs <= 100', 'batch_size in [32, 64, 128]'],
 *   success_criteria: ['accuracy > 0.9', 'training_time < 3600s'],
 *   created_at: createPESUTCTimestamp(),
 * };
 *
 * const validation = validateProblem(problemData);
 * if (!validation.success) {
 *   console.error('Invalid problem:', validation.errors);
 * }
 * ```
 */
//# sourceMappingURL=pes-canonical.d.ts.map