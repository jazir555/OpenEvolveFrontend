/**
 * LoongFlow Adapter Contract Tests
 *
 * Comprehensive contract tests validating API contracts between the LoongFlow
 * adapter and the LoongFlow core system.
 *
 * Purpose: Phase 2 - The Contract (Defense)
 * Law of Runtime Truth: Tests execute against real LoongFlow API
 * Law of Configuration Explicitness: API URL from environment
 * Law of UTC: All timestamps validated as UTC ISO-8601
 *
 * These tests run on adapter startup. If contracts are violated, the adapter
 * refuses to start to prevent data corruption from API changes.
 *
 * Environment Variables:
 *   LOONGFLOW_API_URL - Base URL of LoongFlow sidecar (required)
 *   LOONGFLOW_TIMEOUT_MS - Request timeout in ms (default: 30000)
 *   SKIP_CONTRACT_TESTS - Skip contract tests if true (for development only)
 */
import { z } from 'zod';
/**
 * Health Check Response Contract
 */
declare const HealthCheckContract: z.ZodObject<{
    status: z.ZodEnum<["healthy", "unhealthy", "ok", "error"]>;
    version: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: "error" | "healthy" | "unhealthy" | "ok";
    timestamp?: string | undefined;
    version?: string | undefined;
}, {
    status: "error" | "healthy" | "unhealthy" | "ok";
    timestamp?: string | undefined;
    version?: string | undefined;
}>;
/**
 * Problem Submission Request Contract
 */
declare const ProblemSubmissionRequestContract: z.ZodObject<{
    task: z.ZodString;
    max_iterations: z.ZodOptional<z.ZodNumber>;
    target_score: z.ZodOptional<z.ZodNumber>;
    concurrency: z.ZodOptional<z.ZodNumber>;
    initial_code: z.ZodOptional<z.ZodString>;
    initial_score: z.ZodOptional<z.ZodNumber>;
    initial_evaluation: z.ZodOptional<z.ZodString>;
    workspace_path: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    task: string;
    metadata?: Record<string, any> | undefined;
    max_iterations?: number | undefined;
    target_score?: number | undefined;
    concurrency?: number | undefined;
    initial_code?: string | undefined;
    initial_score?: number | undefined;
    initial_evaluation?: string | undefined;
    workspace_path?: string | undefined;
}, {
    task: string;
    metadata?: Record<string, any> | undefined;
    max_iterations?: number | undefined;
    target_score?: number | undefined;
    concurrency?: number | undefined;
    initial_code?: string | undefined;
    initial_score?: number | undefined;
    initial_evaluation?: string | undefined;
    workspace_path?: string | undefined;
}>;
/**
 * Problem Submission Response Contract
 */
declare const ProblemSubmissionResponseContract: z.ZodObject<{
    agent_id: z.ZodString;
    status: z.ZodString;
    message: z.ZodString;
    timestamp: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    message: string;
    status: string;
    agent_id: string;
    timestamp?: string | undefined;
}, {
    message: string;
    status: string;
    agent_id: string;
    timestamp?: string | undefined;
}>;
/**
 * PES Agent State Contract
 */
declare const PESAgentStateContract: z.ZodObject<{
    agent_id: z.ZodString;
    status: z.ZodEnum<["idle", "running", "interrupted", "completed", "failed"]>;
    current_iteration: z.ZodNumber;
    max_iterations: z.ZodNumber;
    target_score: z.ZodNumber;
    best_score: z.ZodNumber;
    start_time: z.ZodString;
    end_time: z.ZodOptional<z.ZodString>;
    completion_count: z.ZodNumber;
    total_prompt_tokens: z.ZodNumber;
    total_completion_tokens: z.ZodNumber;
    total_cost: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    status: "idle" | "running" | "completed" | "failed" | "interrupted";
    start_time: string;
    max_iterations: number;
    best_score: number;
    agent_id: string;
    target_score: number;
    current_iteration: number;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
    end_time?: string | undefined;
}, {
    status: "idle" | "running" | "completed" | "failed" | "interrupted";
    start_time: string;
    max_iterations: number;
    best_score: number;
    agent_id: string;
    target_score: number;
    current_iteration: number;
    completion_count: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    total_cost: number;
    end_time?: string | undefined;
}>;
/**
 * Execution Result Contract
 */
declare const ExecutionResultContract: z.ZodObject<{
    agent_id: z.ZodString;
    status: z.ZodString;
    final_solution: z.ZodOptional<z.ZodString>;
    final_score: z.ZodOptional<z.ZodNumber>;
    best_solutions: z.ZodOptional<z.ZodArray<z.ZodObject<{
        solution: z.ZodString;
        solution_id: z.ZodString;
        generate_plan: z.ZodString;
        parent_id: z.ZodString;
        island_id: z.ZodNumber;
        iteration: z.ZodNumber;
        timestamp: z.ZodNumber;
        generation: z.ZodNumber;
        sample_cnt: z.ZodNumber;
        sample_weight: z.ZodNumber;
        score: z.ZodNumber;
        evaluation: z.ZodString;
        summary: z.ZodString;
        metadata: z.ZodRecord<z.ZodString, z.ZodAny>;
    }, "strip", z.ZodTypeAny, {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }, {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }>, "many">>;
    total_iterations: z.ZodNumber;
    total_tokens: z.ZodNumber;
    total_cost: z.ZodNumber;
    was_interrupted: z.ZodBoolean;
    start_time: z.ZodString;
    end_time: z.ZodString;
    error: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    status: string;
    start_time: string;
    end_time: string;
    total_tokens: number;
    total_iterations: number;
    agent_id: string;
    total_cost: number;
    was_interrupted: boolean;
    error?: string | undefined;
    final_score?: number | undefined;
    best_solutions?: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }[] | undefined;
    final_solution?: string | undefined;
}, {
    status: string;
    start_time: string;
    end_time: string;
    total_tokens: number;
    total_iterations: number;
    agent_id: string;
    total_cost: number;
    was_interrupted: boolean;
    error?: string | undefined;
    final_score?: number | undefined;
    best_solutions?: {
        timestamp: number;
        metadata: Record<string, any>;
        summary: string;
        solution: string;
        score: number;
        evaluation: string;
        iteration: number;
        island_id: number;
        solution_id: string;
        parent_id: string;
        generate_plan: string;
        generation: number;
        sample_cnt: number;
        sample_weight: number;
    }[] | undefined;
    final_solution?: string | undefined;
}>;
/**
 * Database Status Contract
 */
declare const DatabaseStatusContract: z.ZodObject<{
    global_status: z.ZodObject<{
        current_iteration: z.ZodNumber;
        best_score: z.ZodNumber;
        total_solutions: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        best_score: number;
        current_iteration: number;
        total_solutions: number;
    }, {
        best_score: number;
        current_iteration: number;
        total_solutions: number;
    }>;
    island_status: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodObject<{
        best_score: z.ZodNumber;
        total_solutions: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        best_score: number;
        total_solutions: number;
    }, {
        best_score: number;
        total_solutions: number;
    }>>>;
}, "strip", z.ZodTypeAny, {
    global_status: {
        best_score: number;
        current_iteration: number;
        total_solutions: number;
    };
    island_status?: Record<string, {
        best_score: number;
        total_solutions: number;
    }> | undefined;
}, {
    global_status: {
        best_score: number;
        current_iteration: number;
        total_solutions: number;
    };
    island_status?: Record<string, {
        best_score: number;
        total_solutions: number;
    }> | undefined;
}>;
/**
 * Checkpoint Info Contract
 */
declare const CheckpointInfoContract: z.ZodObject<{
    checkpoint_path: z.ZodString;
    tag: z.ZodString;
    created_at: z.ZodString;
    iteration: z.ZodNumber;
    completion_count: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    created_at: string;
    iteration: number;
    checkpoint_path: string;
    tag: string;
    completion_count: number;
}, {
    created_at: string;
    iteration: number;
    checkpoint_path: string;
    tag: string;
    completion_count: number;
}>;
/**
 * Error Response Contract
 */
declare const ErrorResponseContract: z.ZodObject<{
    detail: z.ZodString;
    error_code: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    detail: string;
    timestamp?: string | undefined;
    error_code?: string | undefined;
}, {
    detail: string;
    timestamp?: string | undefined;
    error_code?: string | undefined;
}>;
/**
 * Validate all API contracts before starting adapter
 * This function should be called during adapter initialization
 *
 * @returns true if all contracts are valid
 * @throws Error if any contract is violated
 */
export declare function validateAllContracts(): boolean;
export { HealthCheckContract, ProblemSubmissionRequestContract, ProblemSubmissionResponseContract, PESAgentStateContract, ExecutionResultContract, DatabaseStatusContract, CheckpointInfoContract, ErrorResponseContract, };
//# sourceMappingURL=contract.test.d.ts.map