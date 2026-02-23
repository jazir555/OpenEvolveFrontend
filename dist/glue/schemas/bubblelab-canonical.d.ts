/**
 * BubbleLab Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for BubbleLab workflow and
 * bubble management interactions. All adapters must normalize their data to/from
 * this format.
 *
 * Law of the "Air Gap": This is the ONLY acceptable format for BubbleLab data
 * in the glue layer. Do not pass raw BubbleLab API responses between services.
 */
import { z } from 'zod';
/**
 * Bubble Type Enum
 *
 * Defines the types of bubbles available in the BubbleLab system.
 */
export declare const BubbleType: z.ZodEnum<["workflow", "data_processing", "analysis", "visualization", "notification", "integration", "custom"]>;
export type BubbleType = z.infer<typeof BubbleType>;
/**
 * Bubble Status Enum
 *
 * Defines the possible states of a bubble.
 */
export declare const BubbleStatus: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "paused"]>;
export type BubbleStatus = z.infer<typeof BubbleStatus>;
/**
 * Bubble Request Schema
 *
 * Represents a request to create or execute a bubble.
 */
export declare const BubbleRequest: z.ZodObject<{
    workspace_id: z.ZodString;
    bubble_type: z.ZodEnum<["workflow", "data_processing", "analysis", "visualization", "notification", "integration", "custom"]>;
    name: z.ZodString;
    description: z.ZodOptional<z.ZodString>;
    data: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    config: z.ZodOptional<z.ZodObject<{
        priority: z.ZodOptional<z.ZodEnum<["low", "normal", "high", "urgent"]>>;
        retry_count: z.ZodOptional<z.ZodNumber>;
        timeout_ms: z.ZodNumber;
        dependencies: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
        notification_settings: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        timeout_ms: number;
        retry_count?: number | undefined;
        dependencies?: string[] | undefined;
        priority?: "high" | "low" | "normal" | "urgent" | undefined;
        notification_settings?: Record<string, any> | undefined;
    }, {
        timeout_ms: number;
        retry_count?: number | undefined;
        dependencies?: string[] | undefined;
        priority?: "high" | "low" | "normal" | "urgent" | undefined;
        notification_settings?: Record<string, any> | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    name: string;
    workspace_id: string;
    bubble_type: "custom" | "analysis" | "integration" | "workflow" | "notification" | "data_processing" | "visualization";
    correlation_id?: string | undefined;
    config?: {
        timeout_ms: number;
        retry_count?: number | undefined;
        dependencies?: string[] | undefined;
        priority?: "high" | "low" | "normal" | "urgent" | undefined;
        notification_settings?: Record<string, any> | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    data?: Record<string, any> | undefined;
    description?: string | undefined;
}, {
    name: string;
    workspace_id: string;
    bubble_type: "custom" | "analysis" | "integration" | "workflow" | "notification" | "data_processing" | "visualization";
    correlation_id?: string | undefined;
    config?: {
        timeout_ms: number;
        retry_count?: number | undefined;
        dependencies?: string[] | undefined;
        priority?: "high" | "low" | "normal" | "urgent" | undefined;
        notification_settings?: Record<string, any> | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    data?: Record<string, any> | undefined;
    description?: string | undefined;
}>;
export type BubbleRequest = z.infer<typeof BubbleRequest>;
/**
 * Bubble Response Schema
 *
 * Represents the response after creating or executing a bubble.
 */
export declare const BubbleResponse: z.ZodObject<{
    bubble_id: z.ZodString;
    workspace_id: z.ZodString;
    bubble_type: z.ZodEnum<["workflow", "data_processing", "analysis", "visualization", "notification", "integration", "custom"]>;
    status: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "paused"]>;
    result: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        created_at: z.ZodOptional<z.ZodString>;
        started_at: z.ZodOptional<z.ZodString>;
        completed_at: z.ZodOptional<z.ZodString>;
        execution_time_ms: z.ZodOptional<z.ZodNumber>;
        retry_count: z.ZodOptional<z.ZodNumber>;
        logs: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        retry_count?: number | undefined;
        created_at?: string | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
        logs?: string[] | undefined;
        execution_time_ms?: number | undefined;
    }, {
        retry_count?: number | undefined;
        created_at?: string | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
        logs?: string[] | undefined;
        execution_time_ms?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
    workspace_id: string;
    bubble_type: "custom" | "analysis" | "integration" | "workflow" | "notification" | "data_processing" | "visualization";
    bubble_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        retry_count?: number | undefined;
        created_at?: string | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
        logs?: string[] | undefined;
        execution_time_ms?: number | undefined;
    } | undefined;
    result?: Record<string, any> | undefined;
}, {
    timestamp: string;
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
    workspace_id: string;
    bubble_type: "custom" | "analysis" | "integration" | "workflow" | "notification" | "data_processing" | "visualization";
    bubble_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
    } | undefined;
    metadata?: {
        retry_count?: number | undefined;
        created_at?: string | undefined;
        started_at?: string | undefined;
        completed_at?: string | undefined;
        logs?: string[] | undefined;
        execution_time_ms?: number | undefined;
    } | undefined;
    result?: Record<string, any> | undefined;
}>;
export type BubbleResponse = z.infer<typeof BubbleResponse>;
/**
 * Workflow Request Schema
 *
 * Represents a request to execute a workflow.
 */
export declare const WorkflowRequest: z.ZodObject<{
    workflow_id: z.ZodString;
    workspace_id: z.ZodString;
    parameters: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    config: z.ZodOptional<z.ZodObject<{
        timeout_ms: z.ZodNumber;
        stop_on_error: z.ZodOptional<z.ZodBoolean>;
        parallel_execution: z.ZodOptional<z.ZodBoolean>;
        retry_config: z.ZodOptional<z.ZodObject<{
            max_retries: z.ZodOptional<z.ZodNumber>;
            backoff_ms: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
        }, {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
        }>>;
    }, "strip", z.ZodTypeAny, {
        timeout_ms: number;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
        } | undefined;
        parallel_execution?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    }, {
        timeout_ms: number;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
        } | undefined;
        parallel_execution?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    metadata: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
}, "strip", z.ZodTypeAny, {
    workflow_id: string;
    workspace_id: string;
    correlation_id?: string | undefined;
    config?: {
        timeout_ms: number;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
        } | undefined;
        parallel_execution?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    parameters?: Record<string, any> | undefined;
}, {
    workflow_id: string;
    workspace_id: string;
    correlation_id?: string | undefined;
    config?: {
        timeout_ms: number;
        retry_config?: {
            max_retries?: number | undefined;
            backoff_ms?: number | undefined;
        } | undefined;
        parallel_execution?: boolean | undefined;
        stop_on_error?: boolean | undefined;
    } | undefined;
    metadata?: Record<string, any> | undefined;
    parameters?: Record<string, any> | undefined;
}>;
export type WorkflowRequest = z.infer<typeof WorkflowRequest>;
/**
 * Workflow Response Schema
 *
 * Represents the response after executing a workflow.
 */
export declare const WorkflowResponse: z.ZodObject<{
    execution_id: z.ZodString;
    workflow_id: z.ZodString;
    workspace_id: z.ZodString;
    status: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "paused"]>;
    results: z.ZodArray<z.ZodObject<{
        bubble_id: z.ZodString;
        status: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "paused"]>;
        result: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
        error: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
        bubble_id: string;
        error?: Record<string, any> | undefined;
        result?: Record<string, any> | undefined;
    }, {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
        bubble_id: string;
        error?: Record<string, any> | undefined;
        result?: Record<string, any> | undefined;
    }>, "many">;
    error: z.ZodOptional<z.ZodObject<{
        code: z.ZodString;
        message: z.ZodString;
        failed_bubble_id: z.ZodOptional<z.ZodString>;
        details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    }, "strip", z.ZodTypeAny, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        failed_bubble_id?: string | undefined;
    }, {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        failed_bubble_id?: string | undefined;
    }>>;
    metadata: z.ZodOptional<z.ZodObject<{
        started_at: z.ZodOptional<z.ZodString>;
        completed_at: z.ZodOptional<z.ZodString>;
        total_execution_time_ms: z.ZodOptional<z.ZodNumber>;
        bubbles_executed: z.ZodOptional<z.ZodNumber>;
        bubbles_succeeded: z.ZodOptional<z.ZodNumber>;
        bubbles_failed: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        started_at?: string | undefined;
        completed_at?: string | undefined;
        total_execution_time_ms?: number | undefined;
        bubbles_executed?: number | undefined;
        bubbles_succeeded?: number | undefined;
        bubbles_failed?: number | undefined;
    }, {
        started_at?: string | undefined;
        completed_at?: string | undefined;
        total_execution_time_ms?: number | undefined;
        bubbles_executed?: number | undefined;
        bubbles_succeeded?: number | undefined;
        bubbles_failed?: number | undefined;
    }>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    workflow_id: string;
    timestamp: string;
    execution_id: string;
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
    results: {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
        bubble_id: string;
        error?: Record<string, any> | undefined;
        result?: Record<string, any> | undefined;
    }[];
    workspace_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        failed_bubble_id?: string | undefined;
    } | undefined;
    metadata?: {
        started_at?: string | undefined;
        completed_at?: string | undefined;
        total_execution_time_ms?: number | undefined;
        bubbles_executed?: number | undefined;
        bubbles_succeeded?: number | undefined;
        bubbles_failed?: number | undefined;
    } | undefined;
}, {
    workflow_id: string;
    timestamp: string;
    execution_id: string;
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
    results: {
        status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
        bubble_id: string;
        error?: Record<string, any> | undefined;
        result?: Record<string, any> | undefined;
    }[];
    workspace_id: string;
    correlation_id?: string | undefined;
    error?: {
        message: string;
        code: string;
        details?: Record<string, any> | undefined;
        failed_bubble_id?: string | undefined;
    } | undefined;
    metadata?: {
        started_at?: string | undefined;
        completed_at?: string | undefined;
        total_execution_time_ms?: number | undefined;
        bubbles_executed?: number | undefined;
        bubbles_succeeded?: number | undefined;
        bubbles_failed?: number | undefined;
    } | undefined;
}>;
export type WorkflowResponse = z.infer<typeof WorkflowResponse>;
/**
 * Bubble Status Request Schema
 *
 * Represents a request to check the status of a bubble.
 */
export declare const BubbleStatusRequest: z.ZodObject<{
    bubble_id: z.ZodString;
    include_logs: z.ZodOptional<z.ZodBoolean>;
    include_result: z.ZodOptional<z.ZodBoolean>;
    timeout_ms: z.ZodNumber;
    correlation_id: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    timeout_ms: number;
    bubble_id: string;
    correlation_id?: string | undefined;
    include_logs?: boolean | undefined;
    include_result?: boolean | undefined;
}, {
    timeout_ms: number;
    bubble_id: string;
    correlation_id?: string | undefined;
    include_logs?: boolean | undefined;
    include_result?: boolean | undefined;
}>;
export type BubbleStatusRequest = z.infer<typeof BubbleStatusRequest>;
/**
 * Bubble Status Response Schema
 *
 * Represents the status response for a bubble.
 */
export declare const BubbleStatusResponse: z.ZodObject<{
    bubble_id: z.ZodString;
    status: z.ZodEnum<["pending", "running", "completed", "failed", "cancelled", "paused"]>;
    progress: z.ZodOptional<z.ZodNumber>;
    result: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    error: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    logs: z.ZodOptional<z.ZodArray<z.ZodString, "many">>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
    bubble_id: string;
    correlation_id?: string | undefined;
    error?: Record<string, any> | undefined;
    result?: Record<string, any> | undefined;
    logs?: string[] | undefined;
    progress?: number | undefined;
}, {
    timestamp: string;
    status: "running" | "completed" | "failed" | "pending" | "cancelled" | "paused";
    bubble_id: string;
    correlation_id?: string | undefined;
    error?: Record<string, any> | undefined;
    result?: Record<string, any> | undefined;
    logs?: string[] | undefined;
    progress?: number | undefined;
}>;
export type BubbleStatusResponse = z.infer<typeof BubbleStatusResponse>;
/**
 * Error Model
 *
 * Represents errors that can occur during BubbleLab operations.
 */
export declare const BubbleLabError: z.ZodObject<{
    code: z.ZodEnum<["BUBBLE_NOT_FOUND", "WORKFLOW_NOT_FOUND", "WORKSPACE_NOT_FOUND", "INVALID_BUBBLE_TYPE", "INVALID_WORKFLOW_DEFINITION", "EXECUTION_TIMEOUT", "DEPENDENCY_FAILED", "INSUFFICIENT_PERMISSIONS", "QUOTA_EXCEEDED", "VALIDATION_ERROR", "UNKNOWN_ERROR"]>;
    message: z.ZodString;
    details: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodAny>>;
    correlation_id: z.ZodOptional<z.ZodString>;
    timestamp: z.ZodString;
}, "strip", z.ZodTypeAny, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "QUOTA_EXCEEDED" | "BUBBLE_NOT_FOUND" | "WORKFLOW_NOT_FOUND" | "WORKSPACE_NOT_FOUND" | "INVALID_BUBBLE_TYPE" | "INVALID_WORKFLOW_DEFINITION" | "EXECUTION_TIMEOUT" | "DEPENDENCY_FAILED" | "INSUFFICIENT_PERMISSIONS";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}, {
    timestamp: string;
    message: string;
    code: "VALIDATION_ERROR" | "UNKNOWN_ERROR" | "QUOTA_EXCEEDED" | "BUBBLE_NOT_FOUND" | "WORKFLOW_NOT_FOUND" | "WORKSPACE_NOT_FOUND" | "INVALID_BUBBLE_TYPE" | "INVALID_WORKFLOW_DEFINITION" | "EXECUTION_TIMEOUT" | "DEPENDENCY_FAILED" | "INSUFFICIENT_PERMISSIONS";
    correlation_id?: string | undefined;
    details?: Record<string, any> | undefined;
}>;
export type BubbleLabError = z.infer<typeof BubbleLabError>;
/**
 * Transformation Functions
 */
/**
 * Transform raw BubbleLab API response to canonical BubbleResponse
 */
export declare function transformBubbleResponseToCanonical(rawResponse: any, correlationId?: string): BubbleResponse;
/**
 * Transform canonical BubbleRequest to BubbleLab API format
 */
export declare function transformCanonicalToBubbleRequest(canonicalRequest: BubbleRequest): any;
/**
 * Transform raw Workflow API response to canonical WorkflowResponse
 */
export declare function transformWorkflowResponseToCanonical(rawResponse: any, correlationId?: string): WorkflowResponse;
/**
 * Transform canonical WorkflowRequest to BubbleLab API format
 */
export declare function transformCanonicalToWorkflowRequest(canonicalRequest: WorkflowRequest): any;
/**
 * Validation Functions
 */
export declare function validateBubbleRequest(data: unknown): {
    success: boolean;
    data?: BubbleRequest;
    errors?: string[];
};
export declare function validateBubbleResponse(data: unknown): {
    success: boolean;
    data?: BubbleResponse;
    errors?: string[];
};
export declare function validateWorkflowRequest(data: unknown): {
    success: boolean;
    data?: WorkflowRequest;
    errors?: string[];
};
export declare function validateWorkflowResponse(data: unknown): {
    success: boolean;
    data?: WorkflowResponse;
    errors?: string[];
};
/**
 * Type Guards
 */
export declare function isBubbleRequest(data: unknown): data is BubbleRequest;
export declare function isWorkflowRequest(data: unknown): data is WorkflowRequest;
/**
 * Example usage and validation examples
 */
export declare const BubbleLabExamples: {
    validBubbleRequest: BubbleRequest;
    validBubbleResponse: BubbleResponse;
    validWorkflowRequest: WorkflowRequest;
    validWorkflowResponse: WorkflowResponse;
    validBubbleLabError: BubbleLabError;
};
//# sourceMappingURL=bubblelab-canonical.d.ts.map