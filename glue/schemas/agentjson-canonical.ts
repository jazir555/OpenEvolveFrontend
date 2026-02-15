/**
 * AgentJSON Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for AgentJSON
 * (JSON-based agent protocol and serialization) interactions.
 */

import { z } from 'zod';

/**
 * Agent Type Enum
 */
export const AgentType = z.enum([
  'reactive',
  'proactive',
  'deliberative',
  'hybrid',
]);

export type AgentType = z.infer<typeof AgentType>;

/**
 * Message Type Enum
 */
export const MessageType = z.enum([
  'request',
  'response',
  'notification',
  'error',
  'heartbeat',
]);

export type MessageType = z.infer<typeof MessageType>;

/**
 * Agent State Enum
 */
export const AgentState = z.enum([
  'idle',
  'processing',
  'waiting',
  'error',
  'terminated',
]);

export type AgentState = z.infer<typeof AgentState>;

/**
 * Agent Message Schema
 */
export const AgentMessage = z.object({
  message_id: z.string().uuid().describe("Unique message identifier"),

  message_type: MessageType.describe("Type of message"),

  sender_id: z.string().describe("Sender agent identifier"),

  receiver_id: z.string().optional().describe("Receiver agent identifier"),

  payload: z.record(z.any()).describe("Message payload (JSON)"),

  timestamp: z.string().datetime().describe("UTC message timestamp"),

  correlation_id: z.string().uuid().optional().describe("Correlation for request-response"),

  reply_to: z.string().uuid().optional().describe("Message ID this is a reply to"),

  expires_at: z.string().datetime().optional().describe("Message expiration time"),

  metadata: z.record(z.any()).optional(),
});

export type AgentMessage = z.infer<typeof AgentMessage>;

/**
 * Agent Definition Schema
 */
export const AgentDefinition = z.object({
  agent_id: z.string().describe("Unique agent identifier"),

  agent_type: AgentType.describe("Type of agent"),

  name: z.string().describe("Agent name"),

  description: z.string().optional().describe("Agent description"),

  version: z.string().optional().describe("Agent version"),

  capabilities: z.array(z.string()).describe("Agent capabilities"),

  config: z.object({
    max_concurrent_tasks: z.number().int().positive().optional(),
    timeout_ms: z.number().int().positive().optional(),
    retry_policy: z.enum(['none', 'fixed', 'exponential']).optional(),
    max_retries: z.number().int().min(0).optional(),
  }).optional().describe("Agent configuration"),

  communication: z.object({
    protocol: z.enum(['http', 'websocket', 'amqp', 'mqtt']).optional(),
    endpoint: z.string().optional().describe("Communication endpoint"),
    auth_required: z.boolean().optional(),
  }).optional().describe("Communication settings"),

  metadata: z.record(z.any()).optional(),
});

export type AgentDefinition = z.infer<typeof AgentDefinition>;

/**
 * Agent JSON Request Schema
 */
export const AgentJsonRequest = z.object({
  agent_id: z.string().describe("Target agent ID"),

  action: z.enum([
    'send_message',
    'create_agent',
    'update_agent',
    'delete_agent',
    'query_state',
    'execute_task',
  ]).describe("Action to perform"),

  message: AgentMessage.optional().describe("Message to send (for send_message)"),

  definition: AgentDefinition.optional().describe("Agent definition (for create/update)"),

  task: z.object({
    task_id: z.string().optional(),
    task_type: z.string(),
    parameters: z.record(z.any()),
    timeout_ms: z.number().int().positive().max(3600000).optional(),
  }).optional().describe("Task to execute"),

  timeout_ms: z.number()
    .int().positive().max(60000)
    .describe("Request timeout (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),

  metadata: z.record(z.any()).optional(),
});

export type AgentJsonRequest = z.infer<typeof AgentJsonRequest>;

/**
 * Agent JSON Response Schema
 */
export const AgentJsonResponse = z.object({
  request_id: z.string().describe("Request identifier"),

  action: z.enum([
    'send_message',
    'create_agent',
    'update_agent',
    'delete_agent',
    'query_state',
    'execute_task',
  ]).describe("Action that was performed"),

  status: z.enum([
    'success',
    'failed',
    'timeout',
    'partial',
  ]).describe("Action status"),

  result: z.object({
    message_id: z.string().optional().describe("Sent message ID"),
    agent_id: z.string().optional().describe("Created/updated agent ID"),
    state: AgentState.optional().describe("Agent state (for query_state)"),
    task_result: z.record(z.any()).optional().describe("Task execution result"),
  }).optional().describe("Action result"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional().describe("Error information"),

  metadata: z.object({
    processing_time_ms: z.number().optional(),
    timestamp: z.string().datetime().optional(),
  }).optional().describe("Response metadata"),

  correlation_id: z.string().uuid().optional(),

  timestamp: z.string().datetime(),
});

export type AgentJsonResponse = z.infer<typeof AgentJsonResponse>;

/**
 * Batch Message Request Schema
 */
export const AgentJsonBatchRequest = z.object({
  batch_id: z.string().describe("Batch identifier"),

  messages: z.array(AgentMessage)
    .min(1, "Batch must contain at least one message")
    .describe("Messages to send"),

  config: z.object({
    parallel: z.boolean().optional().describe("Send messages in parallel"),
    stop_on_error: z.boolean().optional().describe("Stop on first error"),
    timeout_ms: z.number().int().positive().max(300000).optional(),
  }).optional(),

  timeout_ms: z.number().int().positive().max(60000),
  correlation_id: z.string().uuid().optional(),
});

export type AgentJsonBatchRequest = z.infer<typeof AgentJsonBatchRequest>;

/**
 * Batch Message Response Schema
 */
export const AgentJsonBatchResponse = z.object({
  batch_id: z.string().describe("Batch identifier"),

  status: z.enum([
    'completed',
    'partially_completed',
    'failed',
  ]).describe("Batch status"),

  results: z.array(z.object({
    message_id: z.string(),
    status: z.enum(['sent', 'failed', 'timeout']),
    error: z.record(z.any()).optional(),
  })).describe("Individual message results"),

  summary: z.object({
    total: z.number(),
    sent: z.number(),
    failed: z.number(),
  }).describe("Batch summary"),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AgentJsonBatchResponse = z.infer<typeof AgentJsonBatchResponse>;

/**
 * Error Model
 */
export const AgentJsonError = z.object({
  code: z.enum([
    'AGENT_NOT_FOUND',
    'INVALID_MESSAGE',
    'AGENT_UNAVAILABLE',
    'TIMEOUT',
    'AUTHORIZATION_FAILED',
    'VALIDATION_ERROR',
    'UNKNOWN_ERROR',
  ]),
  message: z.string(),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AgentJsonError = z.infer<typeof AgentJsonError>;

/**
 * Validation Functions
 */
export function validateAgentJsonRequest(data: unknown): {
  success: boolean;
  data?: AgentJsonRequest;
  errors?: string[];
} {
  const result = AgentJsonRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function isAgentJsonRequest(data: unknown): data is AgentJsonRequest {
  return typeof data === 'object' && data !== null &&
    'agent_id' in data && 'action' in data;
}

/**
 * Examples
 */
export const AgentJsonExamples = {
  validRequest: {
    agent_id: "agent_001",
    action: "send_message" as const,
    message: {
      message_id: "550e8400-e29b-41d4-a716-446655440000",
      message_type: "request" as const,
      sender_id: "agent_002",
      receiver_id: "agent_001",
      payload: { query: "What is the weather?" },
      timestamp: "2025-02-03T12:30:00.000Z",
    },
    timeout_ms: 5000,
  } as AgentJsonRequest,

  validResponse: {
    request_id: "req_001",
    action: "send_message" as const,
    status: "success" as const,
    result: {
      message_id: "550e8400-e29b-41d4-a716-446655440000",
    },
    timestamp: "2025-02-03T12:30:01.000Z",
  } as AgentJsonResponse,
};
