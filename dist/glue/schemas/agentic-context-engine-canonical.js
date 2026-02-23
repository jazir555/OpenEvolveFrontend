"use strict";
/**
 * Agentic Context Engine Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Agentic Context Engine
 * interactions. All adapters must normalize their data to/from this format.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AgenticContextExamples = exports.AgenticContextError = exports.AgenticContextUpdateRequest = exports.AgenticContextQueryResponse = exports.AgenticContextQueryRequest = exports.AgenticContextResponse = exports.AgenticContextRequest = exports.ContextEntry = exports.ContextState = exports.ContextType = void 0;
exports.validateAgenticContextRequest = validateAgenticContextRequest;
exports.isAgenticContextRequest = isAgenticContextRequest;
const zod_1 = require("zod");
/**
 * Context Type Enum
 */
exports.ContextType = zod_1.z.enum([
    'conversation',
    'document',
    'session',
    'user_profile',
    'domain_knowledge',
    'working_memory',
    'episodic',
]);
/**
 * Context State Enum
 */
exports.ContextState = zod_1.z.enum([
    'active',
    'archived',
    'expired',
    'locked',
]);
/**
 * Context Entry Schema
 */
exports.ContextEntry = zod_1.z.object({
    key: zod_1.z.string().min(1).describe("Unique key for the context entry"),
    value: zod_1.z.any().describe("Context value (any JSON-serializable data)"),
    type: zod_1.z.enum([
        'string',
        'number',
        'boolean',
        'object',
        'array',
        'null',
    ]).optional().describe("Data type for validation"),
    timestamp: zod_1.z.string().datetime().optional().describe("UTC timestamp when entry was created"),
    ttl_seconds: zod_1.z.number().int().positive().optional().describe("Time-to-live in seconds"),
    metadata: zod_1.z.record(zod_1.z.any()).optional().describe("Additional metadata"),
});
/**
 * Context Store Request Schema
 */
exports.AgenticContextRequest = zod_1.z.object({
    context_id: zod_1.z.string().optional().describe("Context identifier (optional for creation)"),
    context_type: exports.ContextType.describe("Type of context"),
    entries: zod_1.z.array(exports.ContextEntry).optional().describe("Context entries to store"),
    state: exports.ContextState.optional().describe("Context state"),
    config: zod_1.z.object({
        ttl_seconds: zod_1.z.number().int().positive().optional()
            .describe("Default time-to-live for context"),
        max_entries: zod_1.z.number().int().positive().optional()
            .describe("Maximum number of entries"),
        eviction_policy: zod_1.z.enum(['lru', 'fifo', 'lfu']).optional()
            .describe("Cache eviction policy"),
        compression: zod_1.z.boolean().optional()
            .describe("Whether to compress stored data"),
    }).optional().describe("Context configuration"),
    timeout_ms: zod_1.z.number()
        .int().positive().max(60000)
        .describe("Request timeout in milliseconds (MANDATORY)"),
    correlation_id: zod_1.z.string().uuid().optional(),
    metadata: zod_1.z.record(zod_1.z.any()).optional(),
});
/**
 * Context Store Response Schema
 */
exports.AgenticContextResponse = zod_1.z.object({
    context_id: zod_1.z.string().describe("Context identifier"),
    context_type: exports.ContextType.describe("Type of context"),
    state: exports.ContextState.describe("Current state"),
    entries: zod_1.z.array(exports.ContextEntry).describe("Stored context entries"),
    metadata: zod_1.z.object({
        created_at: zod_1.z.string().datetime().optional().describe("UTC timestamp of creation"),
        updated_at: zod_1.z.string().datetime().optional().describe("UTC timestamp of last update"),
        expires_at: zod_1.z.string().datetime().optional().describe("UTC timestamp of expiration"),
        entry_count: zod_1.z.number().optional().describe("Number of entries"),
        size_bytes: zod_1.z.number().optional().describe("Approximate size in bytes"),
    }).optional().describe("Context metadata"),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
        details: zod_1.z.record(zod_1.z.any()).optional(),
    }).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Context Query Request Schema
 */
exports.AgenticContextQueryRequest = zod_1.z.object({
    context_id: zod_1.z.string().describe("Context to query"),
    query: zod_1.z.object({
        keys: zod_1.z.array(zod_1.z.string()).optional().describe("Specific keys to retrieve"),
        pattern: zod_1.z.string().optional().describe("Pattern match for keys"),
        filter: zod_1.z.record(zod_1.z.any()).optional().describe("Filter criteria"),
        limit: zod_1.z.number().int().positive().optional().describe("Maximum results"),
        offset: zod_1.z.number().int().min(0).optional().describe("Result offset"),
    }).describe("Query criteria"),
    timeout_ms: zod_1.z.number().int().positive().max(30000),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Context Query Response Schema
 */
exports.AgenticContextQueryResponse = zod_1.z.object({
    context_id: zod_1.z.string().describe("Context identifier"),
    results: zod_1.z.array(exports.ContextEntry).describe("Matching entries"),
    metadata: zod_1.z.object({
        total_count: zod_1.z.number().optional().describe("Total matching entries"),
        returned_count: zod_1.z.number().describe("Number of entries returned"),
        query_time_ms: zod_1.z.number().optional().describe("Query execution time"),
    }).optional(),
    error: zod_1.z.object({
        code: zod_1.z.string(),
        message: zod_1.z.string(),
    }).optional(),
    correlation_id: zod_1.z.string().uuid().optional(),
    timestamp: zod_1.z.string().datetime(),
});
/**
 * Context Update Request Schema
 */
exports.AgenticContextUpdateRequest = zod_1.z.object({
    context_id: zod_1.z.string().describe("Context to update"),
    updates: zod_1.z.object({
        add_entries: zod_1.z.array(exports.ContextEntry).optional().describe("Entries to add"),
        update_entries: zod_1.z.array(zod_1.z.object({
            key: zod_1.z.string(),
            value: zod_1.z.any(),
            merge: zod_1.z.boolean().optional().describe("Whether to merge objects"),
        })).optional().describe("Entries to update"),
        delete_keys: zod_1.z.array(zod_1.z.string()).optional().describe("Keys to delete"),
        set_state: exports.ContextState.optional().describe("New state"),
    }).describe("Update operations"),
    timeout_ms: zod_1.z.number().int().positive().max(60000),
    correlation_id: zod_1.z.string().uuid().optional(),
});
/**
 * Error Model
 */
exports.AgenticContextError = zod_1.z.object({
    code: zod_1.z.enum([
        'CONTEXT_NOT_FOUND',
        'CONTEXT_EXPIRED',
        'CONTEXT_LOCKED',
        'INVALID_KEY',
        'QUOTA_EXCEEDED',
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
function validateAgenticContextRequest(data) {
    const result = exports.AgenticContextRequest.safeParse(data);
    if (result.success) {
        return { success: true, data: result.data };
    }
    return {
        success: false,
        errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
    };
}
function isAgenticContextRequest(data) {
    return typeof data === 'object' && data !== null &&
        'context_type' in data;
}
/**
 * Examples
 */
exports.AgenticContextExamples = {
    validRequest: {
        context_type: "conversation",
        entries: [
            {
                key: "last_message",
                value: "Hello, world!",
                type: "string",
                ttl_seconds: 3600,
            },
        ],
        timeout_ms: 5000,
    },
    validResponse: {
        context_id: "ctx_123",
        context_type: "conversation",
        state: "active",
        entries: [
            {
                key: "last_message",
                value: "Hello, world!",
                type: "string",
                timestamp: "2025-02-03T12:30:00.000Z",
            },
        ],
        metadata: {
            created_at: "2025-02-03T12:30:00.000Z",
            entry_count: 1,
        },
        timestamp: "2025-02-03T12:30:05.000Z",
    },
};
//# sourceMappingURL=agentic-context-engine-canonical.js.map