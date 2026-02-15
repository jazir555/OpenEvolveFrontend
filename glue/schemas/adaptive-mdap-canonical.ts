/**
 * Adaptive MDAP Canonical Schema - Anti-Corruption Layer
 *
 * This schema defines the canonical data models for Adaptive Multi-Domain
 * Adaptive Processing (MDAP) interactions. All adapters must normalize their
 * data to/from this format.
 */

import { z } from 'zod';

/**
 * Processing Domain Enum
 */
export const ProcessingDomain = z.enum([
  'text',
  'image',
  'audio',
  'video',
  'multimodal',
  'structured_data',
]);

export type ProcessingDomain = z.infer<typeof ProcessingDomain>;

/**
 * Adaptation Mode Enum
 */
export const AdaptationMode = z.enum([
  'static',
  'dynamic',
  'incremental',
  'continual',
]);

export type AdaptationMode = z.infer<typeof AdaptationMode>;

/**
 * MDAP Processing Request Schema
 */
export const AdaptiveMdapRequest = z.object({
  task_id: z.string()
    .min(1, "Task ID cannot be empty")
    .describe("Unique identifier for the processing task"),

  domain: ProcessingDomain.describe("Processing domain"),

  input_data: z.union([
    z.string(),
    z.record(z.any()),
    z.array(z.any()),
  ]).describe("Input data to process"),

  adaptation_config: z.object({
    mode: AdaptationMode.optional().describe("Adaptation learning mode"),
    learning_rate: z.number().positive().optional().describe("Learning rate for adaptation"),
    batch_size: z.number().int().positive().optional().describe("Batch size for incremental updates"),
    threshold: z.number().min(0).max(1).optional().describe("Confidence threshold for adaptation"),
  }).optional().describe("Adaptation configuration"),

  model_config: z.object({
    base_model: z.string().optional().describe("Base model identifier"),
    fine_tuned: z.boolean().optional().describe("Whether to use fine-tuned model"),
    parameters: z.record(z.any()).optional().describe("Additional model parameters"),
  }).optional().describe("Model configuration"),

  timeout_ms: z.number()
    .int("Timeout must be an integer")
    .positive("Timeout must be positive")
    .max(3600000, "Timeout cannot exceed 1 hour")
    .describe("Processing timeout in milliseconds (MANDATORY)"),

  correlation_id: z.string().uuid().optional()
    .describe("Correlation ID for distributed tracing"),

  metadata: z.record(z.any()).optional()
    .describe("Optional metadata"),
});

export type AdaptiveMdapRequest = z.infer<typeof AdaptiveMdapRequest>;

/**
 * MDAP Processing Response Schema
 */
export const AdaptiveMdapResponse = z.object({
  task_id: z.string().describe("Task identifier"),

  status: z.enum([
    'pending',
    'processing',
    'completed',
    'failed',
    'timeout',
  ]).describe("Processing status"),

  result: z.union([
    z.string(),
    z.record(z.any()),
    z.array(z.any()),
  ]).optional().describe("Processing result"),

  adaptations: z.object({
    adaptations_made: z.number().optional().describe("Number of adaptations applied"),
    adaptation_history: z.array(z.object({
      timestamp: z.string().datetime(),
      change_type: z.string(),
      performance_delta: z.number().optional(),
    })).optional().describe("History of adaptations"),
    model_version: z.string().optional().describe("Current model version"),
  }).optional().describe("Adaptation information"),

  performance: z.object({
    accuracy: z.number().min(0).max(1).optional().describe("Accuracy score"),
    latency_ms: z.number().optional().describe("Processing latency"),
    throughput: z.number().optional().describe("Throughput metrics"),
    resource_usage: z.record(z.number()).optional().describe("Resource utilization"),
  }).optional().describe("Performance metrics"),

  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.any()).optional(),
  }).optional().describe("Error information"),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime().describe("UTC timestamp (ISO-8601)"),
});

export type AdaptiveMdapResponse = z.infer<typeof AdaptiveMdapResponse>;

/**
 * Batch Processing Request Schema
 */
export const AdaptiveMdapBatchRequest = z.object({
  batch_id: z.string().describe("Batch identifier"),

  tasks: z.array(AdaptiveMdapRequest.omit({ timeout_ms: true, correlation_id: true }))
    .min(1, "Batch must contain at least one task")
    .describe("Tasks to process"),

  config: z.object({
    parallelism: z.number().int().min(1).max(100).optional()
      .describe("Number of parallel tasks"),
    stop_on_error: z.boolean().optional()
      .describe("Whether to stop on first error"),
    timeout_ms: z.number().int().positive().max(7200000).optional()
      .describe("Overall batch timeout (max 2 hours)"),
  }).optional().describe("Batch configuration"),

  timeout_ms: z.number().int().positive().max(3600000)
    .describe("Default timeout for individual tasks (MANDATORY)"),

  correlation_id: z.string().uuid().optional(),
  metadata: z.record(z.any()).optional(),
});

export type AdaptiveMdapBatchRequest = z.infer<typeof AdaptiveMdapBatchRequest>;

/**
 * Batch Processing Response Schema
 */
export const AdaptiveMdapBatchResponse = z.object({
  batch_id: z.string().describe("Batch identifier"),

  status: z.enum([
    'pending',
    'processing',
    'completed',
    'partially_completed',
    'failed',
  ]).describe("Batch status"),

  results: z.array(AdaptiveMdapResponse).describe("Individual task results"),

  summary: z.object({
    total_tasks: z.number().describe("Total number of tasks"),
    completed: z.number().describe("Successfully completed tasks"),
    failed: z.number().describe("Failed tasks"),
    total_processing_time_ms: z.number().optional().describe("Total processing time"),
    average_latency_ms: z.number().optional().describe("Average latency per task"),
  }).describe("Batch summary"),

  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AdaptiveMdapBatchResponse = z.infer<typeof AdaptiveMdapBatchResponse>;

/**
 * Error Model
 */
export const AdaptiveMdapError = z.object({
  code: z.enum([
    'INVALID_INPUT',
    'DOMAIN_NOT_SUPPORTED',
    'MODEL_NOT_AVAILABLE',
    'ADAPTATION_FAILED',
    'PROCESSING_TIMEOUT',
    'RESOURCE_EXHAUSTED',
    'VALIDATION_ERROR',
    'UNKNOWN_ERROR',
  ]).describe("Error code"),

  message: z.string().describe("Human-readable error message"),
  details: z.record(z.any()).optional(),
  correlation_id: z.string().uuid().optional(),
  timestamp: z.string().datetime(),
});

export type AdaptiveMdapError = z.infer<typeof AdaptiveMdapError>;

/**
 * Validation Functions
 */
export function validateAdaptiveMdapRequest(data: unknown): {
  success: boolean;
  data?: AdaptiveMdapRequest;
  errors?: string[];
} {
  const result = AdaptiveMdapRequest.safeParse(data);
  if (result.success) {
    return { success: true, data: result.data };
  }
  return {
    success: false,
    errors: result.error.errors.map(e => `${e.path.join('.')}: ${e.message}`),
  };
}

export function validateAdaptiveMdapResponse(data: unknown): {
  success: boolean;
  data?: AdaptiveMdapResponse;
  errors?: string[];
} {
  const result = AdaptiveMdapResponse.safeParse(data);
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
export function isAdaptiveMdapRequest(data: unknown): data is AdaptiveMdapRequest {
  return typeof data === 'object' && data !== null &&
    'task_id' in data && 'domain' in data && 'input_data' in data;
}

/**
 * Example usage
 */
export const AdaptiveMdapExamples = {
  validRequest: {
    task_id: "task_001",
    domain: "text" as const,
    input_data: "Sample text for processing",
    adaptation_config: {
      mode: "incremental" as const,
      learning_rate: 0.001,
      threshold: 0.85,
    },
    timeout_ms: 30000,
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
  } as AdaptiveMdapRequest,

  validResponse: {
    task_id: "task_001",
    status: "completed" as const,
    result: { processed: true, output: "Processed result" },
    adaptations: {
      adaptations_made: 5,
      model_version: "v1.2.3",
    },
    performance: {
      accuracy: 0.92,
      latency_ms: 1250,
    },
    correlation_id: "550e8400-e29b-41d4-a716-446655440000",
    timestamp: "2025-02-03T12:30:45.000Z",
  } as AdaptiveMdapResponse,
};
