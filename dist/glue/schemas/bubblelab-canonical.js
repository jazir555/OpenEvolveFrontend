"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.BubbleLabExamples = exports.BubbleLabError = exports.BubbleStatusResponse = exports.BubbleStatusRequest = exports.WorkflowResponse = exports.WorkflowRequest = exports.BubbleResponse = exports.BubbleRequest = exports.BubbleStatus = exports.BubbleType = void 0;
exports.transformBubbleResponseToCanonical = transformBubbleResponseToCanonical;
exports.transformCanonicalToBubbleRequest = transformCanonicalToBubbleRequest;
exports.transformWorkflowResponseToCanonical = transformWorkflowResponseToCanonical;
exports.transformCanonicalToWorkflowRequest = transformCanonicalToWorkflowRequest;
exports.validateBubbleRequest = validateBubbleRequest;
exports.validateBubbleResponse = validateBubbleResponse;
exports.validateWorkflowRequest = validateWorkflowRequest;
exports.validateWorkflowResponse = validateWorkflowResponse;
exports.isBubbleRequest = isBubbleRequest;
exports.isWorkflowRequest = isWorkflowRequest;
const zod_1 = require("zod");
/**
 * Bubble Type Enum
 *
 * Defines the types of bubbles available in the BubbleLab system.
 */
exports.BubbleType = zod_1.z.enum([
    'workflow',
    'data_processing',
    'analysis',
    'visualization',
    'notification',
    'integration',
    'custom',
]);
/**
 * Bubble Status Enum
 *
 * Defines the possible states of a bubble.
 */
exports.BubbleStatus = zod_1.z.enum([
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled',
    'paused',
]);
/**
 * Bubble Request Schema
 *
 * Represents a request to create or execute a bubble.
 */
exports.BubbleRequest = zod_1.z.object({
    workspace_id: zod_1.z.string()
        .min(1, "Workspace ID cannot be empty")
        .describe("Identifier of the workspace where the bubble will be created"),
    bubble_type: exports.BubbleType.describe("Type of bubble to create"),
    name: zod_1.z.string()
        .min(1, "Bubble name cannot be empty")
        .max(255, "Bubble name cannot exceed 255 characters")
        .describe("Name of the bubble"),
    description: zod_1.z.string()
        .max(1000, "Description cannot exceed 1000 characters")
        .optional()
        .describe("Optional description of the bubble"),
    data: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Input data or parameters for the bubble"),
    config: zod_1.z.object({
        priority: zod_1.z.enum(['low', 'normal', 'high', 'urgent']).optional()
            .describe("Execution priority"),
        retry_count: zod_1.z.number().int().min(0).max(10).optional()
            .describe("Number of retry attempts on failure"),
        timeout_ms: zod_1.z.number()
            .int("Timeout must be an integer")
            .positive("Timeout must be positive")
            .max(3600000, "Timeout cannot exceed 1 hour")
            .describe("Execution timeout in milliseconds (MANDATORY)"),
        dependencies: zod_1.z.array(zod_1.z.string()).optional()
            .describe("List of bubble IDs this bubble depends on"),
        notification_settings: zod_1.z.record(zod_1.z.any()).optional()
            .describe("Notification configuration"),
    }).optional().describe("Bubble configuration"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata for observability and tracking"),
});
/**
 * Bubble Response Schema
 *
 * Represents the response after creating or executing a bubble.
 */
exports.BubbleResponse = zod_1.z.object({
    bubble_id: zod_1.z.string().uuid().describe("Unique identifier for the bubble"),
    workspace_id: zod_1.z.string().describe("Workspace containing the bubble"),
    bubble_type: exports.BubbleType.describe("Type of the bubble"),
    status: exports.BubbleStatus.describe("Current status of the bubble"),
    result: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Result data returned by the bubble (if completed)"),
    error: zod_1.z.object({
        code: zod_1.z.string().describe("Error code"),
        message: zod_1.z.string().describe("Error message"),
        details: zod_1.z.record(zod_1.z.any()).optional().describe("Additional error details"),
    }).optional().describe("Error information (if failed)"),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional()
            .describe("UTC timestamp when bubble was created (ISO-8601)"),
        started_at: zod_1.z.string().datetime().optional()
            .describe("UTC timestamp when bubble started execution (ISO-8601)"),
        completed_at: zod_1.z.string().datetime().optional()
            .describe("UTC timestamp when bubble completed (ISO-8601)"),
        execution_time_ms: zod_1.z.number().optional()
            .describe("Actual execution time in milliseconds"),
        retry_count: zod_1.z.number().optional()
            .describe("Number of retries attempted"),
        logs: zod_1.z.array(zod_1.z.string()).optional()
            .describe("Execution logs"),
    }).optional().describe("Execution metadata"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Workflow Request Schema
 *
 * Represents a request to execute a workflow.
 */
exports.WorkflowRequest = zod_1.z.object({
    workflow_id: zod_1.z.string()
        .min(1, "Workflow ID cannot be empty")
        .describe("Identifier of the workflow to execute"),
    workspace_id: zod_1.z.string()
        .min(1, "Workspace ID cannot be empty")
        .describe("Identifier of the workspace containing the workflow"),
    parameters: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Input parameters for the workflow"),
    config: zod_1.z.object({
        timeout_ms: zod_1.z.number()
            .int("Timeout must be an integer")
            .positive("Timeout must be positive")
            .max(3600000, "Timeout cannot exceed 1 hour")
            .describe("Execution timeout in milliseconds (MANDATORY)"),
        stop_on_error: zod_1.z.boolean().optional()
            .describe("Whether to stop workflow on first error"),
        parallel_execution: zod_1.z.boolean().optional()
            .describe("Whether to execute bubbles in parallel where possible"),
        retry_config: zod_1.z.object({
            max_retries: zod_1.z.number().int().min(0).max(10).optional()
                .describe("Maximum number of retries"),
            backoff_ms: zod_1.z.number().int().min(0).optional()
                .describe("Backoff delay between retries"),
        }).optional().describe("Retry configuration"),
    }).optional().describe("Workflow execution configuration"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    metadata: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Optional metadata for observability and tracking"),
});
/**
 * Workflow Response Schema
 *
 * Represents the response after executing a workflow.
 */
exports.WorkflowResponse = zod_1.z.object({
    execution_id: zod_1.z.string().uuid().describe("Unique identifier for this execution"),
    workflow_id: zod_1.z.string().describe("Identifier of the executed workflow"),
    workspace_id: zod_1.z.string().describe("Workspace containing the workflow"),
    status: exports.BubbleStatus.describe("Status of the workflow execution"),
    results: zod_1.z.array(zod_1.z.object({
        bubble_id: zod_1.z.string().describe("ID of the bubble"),
        status: exports.BubbleStatus.describe("Status of the bubble"),
        result: zod_1.z.record(zod_1.z.any()).optional().describe("Bubble result"),
        error: zod_1.z.record(zod_1.z.any()).optional().describe("Bubble error"),
    })).describe("Results of each bubble in the workflow"),
    error: zod_1.z.object({
        code: zod_1.z.string().describe("Error code"),
        message: zod_1.z.string().describe("Error message"),
        failed_bubble_id: zod_1.z.string().optional().describe("ID of the bubble that caused the failure"),
        details: zod_1.z.record(zod_1.z.any()).optional().describe("Additional error details"),
    }).optional().describe("Workflow-level error (if failed)"),
    metadata: zod_1.z.object({
        started_at: zod_1.z.string().datetime().optional()
            .describe("UTC timestamp when workflow started (ISO-8601)"),
        completed_at: zod_1.z.string().datetime().optional()
            .describe("UTC timestamp when workflow completed (ISO-8601)"),
        total_execution_time_ms: zod_1.z.number().optional()
            .describe("Total execution time in milliseconds"),
        bubbles_executed: zod_1.z.number().optional()
            .describe("Number of bubbles executed"),
        bubbles_succeeded: zod_1.z.number().optional()
            .describe("Number of bubbles that succeeded"),
        bubbles_failed: zod_1.z.number().optional()
            .describe("Number of bubbles that failed"),
    }).optional().describe("Execution metadata"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Bubble Status Request Schema
 *
 * Represents a request to check the status of a bubble.
 */
exports.BubbleStatusRequest = zod_1.z.object({
    bubble_id: zod_1.z.string().uuid().describe("ID of the bubble to check"),
    include_logs: zod_1.z.boolean().optional()
        .describe("Whether to include execution logs"),
    include_result: zod_1.z.boolean().optional()
        .describe("Whether to include the result (if completed)"),
    timeout_ms: zod_1.z.number()
        .int("Timeout must be an integer")
        .positive("Timeout must be positive")
        .max(30000, "Timeout cannot exceed 30 seconds")
        .describe("Request timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
});
/**
 * Bubble Status Response Schema
 *
 * Represents the status response for a bubble.
 */
exports.BubbleStatusResponse = zod_1.z.object({
    bubble_id: zod_1.z.string().uuid().describe("ID of the bubble"),
    status: exports.BubbleStatus.describe("Current status"),
    progress: zod_1.z.number().min(0).max(100).optional()
        .describe("Progress percentage (0-100)"),
    result: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Result data (if completed and requested)"),
    error: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Error information (if failed)"),
    logs: zod_1.z.array(zod_1.z.string()).optional()
        .describe("Execution logs (if requested)"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for distributed tracing"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp of the response (ISO-8601)"),
});
/**
 * Error Model
 *
 * Represents errors that can occur during BubbleLab operations.
 */
exports.BubbleLabError = zod_1.z.object({
    code: zod_1.z.enum([
        'BUBBLE_NOT_FOUND',
        'WORKFLOW_NOT_FOUND',
        'WORKSPACE_NOT_FOUND',
        'INVALID_BUBBLE_TYPE',
        'INVALID_WORKFLOW_DEFINITION',
        'EXECUTION_TIMEOUT',
        'DEPENDENCY_FAILED',
        'INSUFFICIENT_PERMISSIONS',
        'QUOTA_EXCEEDED',
        'VALIDATION_ERROR',
        'UNKNOWN_ERROR',
    ]).describe("Error code for categorization"),
    message: zod_1.z.string().describe("Human-readable error message"),
    details: zod_1.z.record(zod_1.z.any()).optional()
        .describe("Additional error details"),
    correlation_id: zod_1.z.string().uuid().optional()
        .describe("Correlation ID for tracing the error"),
    timestamp: zod_1.z.string().datetime()
        .describe("UTC timestamp when the error occurred (ISO-8601)"),
});
/**
 * Transformation Functions
 */
/**
 * Transform raw BubbleLab API response to canonical BubbleResponse
 */
function transformBubbleResponseToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        bubble_id: rawResponse.id || rawResponse.bubble_id,
        workspace_id: rawResponse.workspace_id,
        bubble_type: exports.BubbleType.parse(rawResponse.type.toLowerCase()),
        status: exports.BubbleStatus.parse(rawResponse.status.toLowerCase()),
        result: rawResponse.result,
        error: rawResponse.error ? {
            code: rawResponse.error.code,
            message: rawResponse.error.message,
            details: rawResponse.error.details,
        } : undefined,
        metadata: {
            created_at: rawResponse.created_at,
            started_at: rawResponse.started_at,
            completed_at: rawResponse.completed_at,
            execution_time_ms: rawResponse.execution_time,
            retry_count: rawResponse.retry_count,
            logs: rawResponse.logs,
        },
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * Transform canonical BubbleRequest to BubbleLab API format
 */
function transformCanonicalToBubbleRequest(canonicalRequest) {
    return {
        workspace_id: canonicalRequest.workspace_id,
        type: canonicalRequest.bubble_type,
        name: canonicalRequest.name,
        description: canonicalRequest.description,
        data: canonicalRequest.data,
        config: canonicalRequest.config,
        metadata: canonicalRequest.metadata,
    };
}
/**
 * Transform raw Workflow API response to canonical WorkflowResponse
 */
function transformWorkflowResponseToCanonical(rawResponse, correlationId) {
    const timestamp = new Date().toISOString();
    return {
        execution_id: rawResponse.execution_id || rawResponse.id,
        workflow_id: rawResponse.workflow_id,
        workspace_id: rawResponse.workspace_id,
        status: exports.BubbleStatus.parse(rawResponse.status.toLowerCase()),
        results: (rawResponse.results || rawResponse.bubbles || []).map((bubble) => ({
            bubble_id: bubble.bubble_id,
            status: exports.BubbleStatus.parse(bubble.status.toLowerCase()),
            result: bubble.result,
            error: bubble.error,
        })),
        error: rawResponse.error ? {
            code: rawResponse.error.code,
            message: rawResponse.error.message,
            failed_bubble_id: rawResponse.error.failed_bubble_id,
            details: rawResponse.error.details,
        } : undefined,
        metadata: {
            started_at: rawResponse.started_at,
            completed_at: rawResponse.completed_at,
            total_execution_time_ms: rawResponse.total_time,
            bubbles_executed: rawResponse.bubbles_executed,
            bubbles_succeeded: rawResponse.bubbles_succeeded,
            bubbles_failed: rawResponse.bubbles_failed,
        },
        correlation_id: correlationId,
        timestamp,
    };
}
/**
 * Transform canonical WorkflowRequest to BubbleLab API format
 */
function transformCanonicalToWorkflowRequest(canonicalRequest) {
    return {
        workflow_id: canonicalRequest.workflow_id,
        workspace_id: canonicalRequest.workspace_id,
        parameters: canonicalRequest.parameters,
        config: canonicalRequest.config,
        metadata: canonicalRequest.metadata,
    };
}
/**
 * Validation Functions
 */
function validateBubbleRequest(data) {
    const result = exports.BubbleRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateBubbleResponse(data) {
    const result = exports.BubbleResponse.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateWorkflowRequest(data) {
    const result = exports.WorkflowRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function validateWorkflowResponse(data) {
    const result = exports.WorkflowResponse.safeParse(data);
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
function isBubbleRequest(data) {
    return typeof data === 'object' && data !== null &&
        'workspace_id' in data && 'bubble_type' in data && 'name' in data;
}
function isWorkflowRequest(data) {
    return typeof data === 'object' && data !== null &&
        'workflow_id' in data && 'workspace_id' in data;
}
/**
 * Example usage and validation examples
 */
exports.BubbleLabExamples = {
    validBubbleRequest: {
        workspace_id: "workspace_abc123",
        bubble_type: "data_processing",
        name: "Process Customer Data",
        description: "Clean and transform customer data",
        data: {
            source: "database",
            table: "customers",
            transformations: ["clean", "normalize"],
        },
        config: {
            priority: "high",
            retry_count: 3,
            timeout_ms: 60000,
            dependencies: [],
            notification_settings: {
                on_success: true,
                on_failure: true,
            },
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        metadata: {
            requested_by: "user123",
            project: "data_pipeline",
        },
    },
    validBubbleResponse: {
        bubble_id: "550e8400-e29b-41d4-a716-446655440001",
        workspace_id: "workspace_abc123",
        bubble_type: "data_processing",
        status: "completed",
        result: {
            rows_processed: 10000,
            rows_cleaned: 9500,
            output_table: "customers_clean",
        },
        metadata: {
            created_at: "2025-02-03T12:30:00.000Z",
            started_at: "2025-02-03T12:30:01.000Z",
            completed_at: "2025-02-03T12:30:45.000Z",
            execution_time_ms: 44000,
            retry_count: 0,
            logs: [
                "Starting data processing...",
                "Connected to database",
                "Processing 10000 rows...",
                "Completed successfully",
            ],
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        timestamp: "2025-02-03T12:30:45.000Z",
    },
    validWorkflowRequest: {
        workflow_id: "workflow_xyz789",
        workspace_id: "workspace_abc123",
        parameters: {
            input_data: "/data/input.csv",
            output_format: "json",
        },
        config: {
            timeout_ms: 300000,
            stop_on_error: false,
            parallel_execution: true,
            retry_config: {
                max_retries: 2,
                backoff_ms: 5000,
            },
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        metadata: {
            requested_by: "user456",
        },
    },
    validWorkflowResponse: {
        execution_id: "550e8400-e29b-41d4-a716-446655440002",
        workflow_id: "workflow_xyz789",
        workspace_id: "workspace_abc123",
        status: "completed",
        results: [
            {
                bubble_id: "bubble_001",
                status: "completed",
                result: { records_processed: 1000 },
            },
            {
                bubble_id: "bubble_002",
                status: "completed",
                result: { report_generated: true, report_path: "/reports/output.json" },
            },
        ],
        metadata: {
            started_at: "2025-02-03T12:00:00.000Z",
            completed_at: "2025-02-03T12:05:00.000Z",
            total_execution_time_ms: 300000,
            bubbles_executed: 2,
            bubbles_succeeded: 2,
            bubbles_failed: 0,
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        timestamp: "2025-02-03T12:05:00.000Z",
    },
    validBubbleLabError: {
        code: 'BUBBLE_NOT_FOUND',
        message: "The specified bubble does not exist",
        details: {
            bubble_id: "invalid_bubble_id",
            workspace_id: "workspace_abc123",
        },
        correlation_id: "550e8400-e29b-41d4-a716-446655440000",
        timestamp: "2025-02-03T12:34:56.789Z",
    },
};
//# sourceMappingURL=bubblelab-canonical.js.map