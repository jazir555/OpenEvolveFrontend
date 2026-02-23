"use strict";
/**
 * AgentJSON Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AgentJSON
 * (JSON-based agent protocol and serialization) interactions.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AgentJsonExamples = exports.AgentJsonError = exports.AgentJsonBatchResponse = exports.AgentJsonBatchRequest = exports.AgentJsonResponse = exports.AgentJsonRequest = exports.AgentDefinition = exports.AgentMessage = exports.AgentState = exports.MessageType = exports.AgentType = void 0;
exports.validateAgentJsonRequest = validateAgentJsonRequest;
exports.isAgentJsonRequest = isAgentJsonRequest;
const zod_1 = require("zod");
/**
 * Agent Type Enum
 */
exports.AgentType = zod_1.z.enum([
    'reactive',
    'proactive',
    'deliberative',
    'hybrid',
]);
/**
 * Message Type Enum
 */
exports.MessageType = zod_1.z.enum([
    'request',
    'response',
    'notification',
    'error',
    'heartbeat',
]);
/**
 * Agent State Enum
 */
exports.AgentState = zod_1.z.enum([
    'idle',
    'processing',
    'waiting',
    'error',
    'terminated',
]);
/**
 * Agent Message Schema
 */
exports.AgentMessage = zod_1.z.object({
    message_id: zod_1.z.string().uuid().describe("Unique message identifier"),
    message_type: exports.MessageType.describe("Type of message"),
    sender_id: zod_1.z.string().describe("Sender agent identifier"),
    receiver_id: zod_1.z.string().optional().describe("Receiver agent identifier"),
    payload: zod_1.z.record(zod_1.z.any()).describe("Message payload (JSON)"),
    timestamp: zod_1.z.string().datetime().describe("UTC message timestamp"),
    correlation_id: zod_1.z.string().uuid().optional().describe("Correlation for request-response"),
    reply_to: zod_1.z.string().uuid().optional().describe("Message ID this is a reply to"),
    expires_at: zod_1.z.string().datetime().optional().describe("Message expiration time"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Agent Definition Schema
 */
exports.AgentDefinition = zod_1.z.object({
    agent_id: zod_1.z.string().describe("Unique agent identifier"),
    agent_type: exports.AgentType.describe("Type of agent"),
    name: zod_1.z.string().describe("Agent name"),
    description: zod_1.z.string().optional().describe("Agent description"),
    version: zod_1.z.string().optional().describe("Agent version"),
    capabilities: zod_1.z.array(zod_1.z.string()).describe("Agent capabilities"),
    config: zod_1.z.object({
        max_concurrent_tasks: zod_1.z.number().int().positive().optional(),
        timeout_ms: zod_1.z.number().int().positive().optional(),
        retry_policy: zod_1.z.enum(['none', 'fixed', 'exponential']).optional(),
        max_retries: zod_1.z.number().int().min(0).optional(),
    }).optional().describe("Agent configuration"),
    communication: zod_1.z.object({
        protocol: zod_1.z.enum(['http', 'websocket', 'amqp', 'mqtt']).optional(),
        endpoint: zod_1.z.string().optional().describe("Communication endpoint"),
        auth_required: zod_1.z.boolean().optional(),
    }).optional().describe("Communication settings"),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Agent JSON Request Schema
 */
exports.AgentJsonRequest = zod_1.z.object({
    agent_id: zod_1.z.string().describe("Target agent ID"),
    action: zod_1.z.enum([
        'send_message',
        'create_agent',
        'update_agent',
        'delete_agent',
        'query_state',
        'execute_task',
    ]).describe("Action to perform"),
    message: exports.AgentMessage.optional().describe("Message to send (for send_message)"),
    definition: exports.AgentDefinition.optional().describe("Agent definition (for create/update)"),
    task: zod_1.z.object({
        task_id: zod_1.z.string().optional(),
        task_type: zod_1.z.string(),
        parameters: zod_1.z.record(zod_1.z.any()),
        timeout_ms: zod_1.z.number().int().positive().max(3600000).optional(),
    }).optional().describe("Task to execute"),
    timeout_ms: zod_1.z.number()
        .int().positive().max(60000)
        .describe("Request timeout (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Agent JSON Response Schema
 */
exports.AgentJsonResponse = zod_1.z.object({
    request_id: zod_1.z.string().describe("Request identifier"),
    action: zod_1.z.enum([
        'send_message',
        'create_agent',
        'update_agent',
        'delete_agent',
        'query_state',
        'execute_task',
    ]).describe("Action that was performed"),
    status: zod_1.z.enum([
        'success',
        'failed',
        'timeout',
        'partial',
    ]).describe("Action status"),
    result: zod_1.z.object({
        message_id: zod_1.z.string().optional().describe("Sent message ID"),
        agent_id: zod_1.z.string().optional().describe("Created/updated agent ID"),
        state: exports.AgentState.optional().describe("Agent state (for query_state)"),
        task_result: zod_1.z.record(zod_1.z.any()).optional().describe("Task execution result"),
    }).optional().describe("Action result"),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }).optional().describe("Error information"),
    metadata: zod_1.z.object({
        processing_time_ms: zod_1.z.number().optional(),
        timestamp: zod_1.z.string().datetime().optional(),
    }).optional().describe("Response metadata"),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Batch Message Request Schema
 */
exports.AgentJsonBatchRequest = zod_1.z.object({
    batch_id: zod_1.z.string().describe("Batch identifier"),
    messages: zod_1.z.array(exports.AgentMessage)
        .min(1, "Batch must contain at least one message")
        .describe("Messages to send"),
    config: zod_1.z.object({
        parallel: zod_1.z.boolean().optional().describe("Send messages in parallel"),
        stop_on_error: zod_1.z.boolean().optional().describe("Stop on first error"),
        timeout_ms: zod_1.z.number().int().positive().max(300000).optional(),
    }).optional(),
    timeout_ms: zod_1.z.number().int().positive().max(60000),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Batch Message Response Schema
 */
exports.AgentJsonBatchResponse = zod_1.z.object({
    batch_id: zod_1.z.string().describe("Batch identifier"),
    status: zod_1.z.enum([
        'completed',
        'partially_completed',
        'failed',
    ]).describe("Batch status"),
    results: zod_1.z.array(zod_1.z.object({
        message_id: zod_1.z.string(),
        status: zod_1.z.enum(['sent', 'failed', 'timeout']),
        error: zod_1.z.record(zod_1.z.any()).optional(),
    })).describe("Individual message results"),
    summary: zod_1.z.object({
        total: zod_1.z.number(),
        sent: zod_1.z.number(),
        failed: zod_1.z.number(),
    }).describe("Batch summary"),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Error Model
 */
exports.AgentJsonError = zod_1.z.object({
    code: zod_1.z.enum([
        'AGENT_NOT_FOUND',
        'INVALID_MESSAGE',
        'AGENT_UNAVAILABLE',
        'TIMEOUT',
        'AUTHORIZATION_FAILED',
        'VALIDATION_ERROR',
        'UNKNOWN_ERROR',
    ]),
    message: zod_1.z.string(),
    details: zod_1.z.record(zod_1.z.any()).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Validation Functions
 */
function validateAgentJsonRequest(data) {
    const result = exports.AgentJsonRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function isAgentJsonRequest(data) {
    return typeof data === 'object' && data !== null &&
        'agent_id' in data && 'action' in data;
}
/**
 * Examples
 */
exports.AgentJsonExamples = {
    validRequest: {
        agent_id: "agent_001",
        action: "send_message",
        message: {
            message_id: "550e8400-e29b-41d4-a716-446655440000",
            message_type: "request",
            sender_id: "agent_002",
            receiver_id: "agent_001",
            payload: { query: "What is the weather?" },
            timestamp: "2025-02-03T12:30:00.000Z",
        },
        timeout_ms: 5000,
    },
    validResponse: {
        request_id: "req_001",
        action: "send_message",
        status: "success",
        result: {
            message_id: "550e8400-e29b-41d4-a716-446655440000",
        },
        timestamp: "2025-02-03T12:30:01.000Z",
    },
};
//# sourceMappingURL=agentjson-canonical.js.map