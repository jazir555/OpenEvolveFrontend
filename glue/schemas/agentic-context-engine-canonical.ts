/**
 * Agentic Context Engine Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Agentic Context Engine
 * interactions. All adapters must normalize their data to/from this format.
 */

import { z } from 'zod';

/**
 * Context Type Enum
 */
export const ContextType = z.enum([
  'conversation',
  'document',
  'session',
  'user_profile',
  'domain_knowledge',
  'working_memory',
  'episodic',
]);

export type ContextType = z.infer<typeof ContextType>;

/**
 * Context State Enum
 */
export const ContextState = z.enum([
  'active',
  'archived',
  'expired',
  'locked',
]);

export type ContextState = z.infer<typeof ContextState>;

/**
 * Context Entry Schema
 */
export const ContextEntry = z.object({
  key: z.string().min(1).describe("Unique key for the context entry"),
  value: z.any().describe("Context value (any JSON-serializable data)"),
  type: z.enum([
    'string',
    'number',
    'boolean',
    'object',
    'array',
    'null',
  ]).optional().describe("Data type for validation"),
  timestamp: z.string().datetime().optional().describe("UTC timestamp when entry was created"),
  ttl_seconds: z.number().int().positive().optional()
    .describe("Time-to-live in seconds"),
  metadata: z.record(z.any()).optional().describe("Additional metadata"),
});

export type ContextEntry = z.infer<typeof ContextEntry>;

/**
 * Context Store Request Schema
 */
export const AgenticContextRequest = z.object({
  context_id: z.string().optional().describe("Context identifier (optional for creation)"),

  context_type: ContextType.describe("Type of context"),

  entries: z.array(ContextEntry).optional().describe("Context entries to store"),

  state: ContextState.optional().describe("Context state"),

  config: z.object({
    ttl_seconds: z.number().int().positive().optional()
      .describe("Default time-to-live for context"),
    max_entries: z.number().int().positive().optional()
      .describe("Maximum number of entries"),
    eviction_policy: z.enum(['lru', 'fifo', 'lfu']).optional()
      .describe("Cache eviction policy"),
    compression: z.boolean().optional()
      .describe("Whether to compress stored data"),
  }).optional().describe("Context configuration"),

  timeout_ms: z.number()
    .int().positive().max(60000)
    .describe("Request timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),
  metadata: z.record(z.any()).optional(),
});

export type AgenticContextRequest = z.infer<typeof AgenticContextRequest>;

/**
 * Context Store Response Schema
 */
export const AgenticContextResponse = z.object({
  context_id: z.string().describe("Context identifier"),

  context_type: ContextType.describe("Type of context"),

  state: ContextState.describe("Current state"),

  entries: z.array(ContextEntry).describe("Stored context entries"),

  metadata: z.object({
    created_at: z.string().datetime().optional().describe("UTC timestamp of creation"),
    updated_at: z.string().datetime().optional().describe("UTC timestamp of last update"),
    expires_at: z.string().datetime().optional().describe("UTC timestamp of expiration"),
    entry_count: z.number().optional().describe("Number of entries"),
    size_bytes: z.number().optional().describe("Approximate size in bytes"),
  }).optional().describe("Context metadata"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional(),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AgenticContextResponse = z.infer<typeof AgenticContextResponse>;

/**
 * Context Query Request Schema
 */
export const AgenticContextQueryRequest = z.object({
  context_id: z.string().describe("Context to query"),

  query: z.object({
    keys: z.array(z.string()).optional().describe("Specific keys to retrieve"),
    pattern: z.string().optional().describe("Pattern match for keys"),
    filter: z.record(z.any()).optional().describe("Filter criteria"),
    limit: z.number().int().positive().optional()
      .describe("Maximum results"),
    offset: z.number().int().min(0).optional()
      .describe("Result offset"),
  }).describe("Query criteria"),

  timeout_ms: z.number().int().positive().max(30000),
  correlation_id: z.string().uuid().optional(),
});

export type AgenticContextQueryRequest = z.infer<typeof AgenticContextQueryRequest>;

/**
 * Context Query Response Schema
 */
export const AgenticContextQueryResponse = z.object({
  context_id: z.string().describe("Context identifier"),

  results: z.array(ContextEntry).describe("Matching entries"),

  metadata: z.object({
    total_count: z.number().optional().describe("Total matching entries"),
    returned_count: z.number().describe("Number of entries returned"),
    query_time_ms: z.number().optional().describe("Query execution time"),
  }).optional(),

  error: z.object({
    code: z.string(),
    message: z.string(),
  }).optional(),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AgenticContextQueryResponse = z.infer<typeof AgenticContextQueryResponse>;

/**
 * Context Update Request Schema
 */
export const AgenticContextUpdateRequest = z.object({
  context_id: z.string().describe("Context to update"),

  updates: z.object({
    add_entries: z.array(ContextEntry).optional().describe("Entries to add"),
    update_entries: z.array(z.object({
      key: z.string(),
      value: z.any(),
      merge: z.boolean().optional().describe("Whether to merge objects"),
    })).optional().describe("Entries to update"),
    delete_keys: z.array(z.string()).optional().describe("Keys to delete"),
    set_state: ContextState.optional().describe("New state"),
  }).describe("Update operations"),

  timeout_ms: z.number().int().positive().max(60000),
  correlation_id: z.string().uuid().optional(),
});

export type AgenticContextUpdateRequest = z.infer<typeof AgenticContextUpdateRequest>;

/**
 * Error Model
 */
export const AgenticContextError = z.object({
  code: z.enum([
    'CONTEXT_NOT_FOUND',
    'CONTEXT_EXPIRED',
    'CONTEXT_LOCKED',
    'INVALID_KEY',
    'QUOTA_EXCEEDED',
    'VALIDATION_ERROR',
    'UNKNOWN_ERROR',
  ]),
  message: z.string(),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AgenticContextError = z.infer<typeof AgenticContextError>;

/**
 * Validation Functions
 */
export function validateAgenticContextRequest(data: unknown): {
  success: boolean;
  data?: AgenticContextRequest;
  errors?: string[];
} {
  const result = AgenticContextRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function isAgenticContextRequest(data: unknown): data is AgenticContextRequest {
  return typeof data === 'object' && data !== null
    && 'context_type' in data;
}

/**
 * Examples
 */
export const AgenticContextExamples = {
  validRequest: {
    context_type: "conversation" as const,
    entries: [
      {
        key: "last_message",
        value: "Hello, world!",
        type: "string" as const,
        ttl_seconds: 3600,
      },
    ],
    timeout_ms: 5000,
  } as AgenticContextRequest,

  validResponse: {
    context_id: "ctx_123",
    context_type: "conversation" as const,
    state: "active" as const,
    entries: [
      {
        key: "last_message",
        value: "Hello, world!",
        type: "string" as const,
        timestamp: "2025-02-03T12:30:00.000Z",
      },
    ],
    metadata: {
      created_at: "2025-02-03T12:30:00.000Z",
      entry_count: 1,
    },
    timestamp: "2025-02-03T12:30:05.000Z",
  } as AgenticContextResponse,
};
