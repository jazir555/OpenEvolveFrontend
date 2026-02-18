/**
 * BubbleLab Canonical Schema
 *
 * Purpose: Define the canonical data model for BubbleLab integration
 * Compliance: Anti-Corruption Layer - normalize BubbleLab data to canonical form
 *
 * This schema maps BubbleLab's bubble/workflow concepts to the OpenEvolve canonical model
 */

import { z } from 'zod';

// =============================================================================
// BubbleLab-Specific Canonical Types
// =============================================================================

/**
 * Bubble Type Enumeration
 * Maps to different bubble types in BubbleLab (PostgreSQL, Slack, AI Agent, etc.)
 */
export enum BubbleType {
  POSTGRESQL = 'postgresql',
  SLACK = 'slack',
  AI_AGENT = 'ai_agent',
  DATABASE_ANALYZER = 'database_analyzer',
  SLACK_NOTIFIER = 'slack_notifier',
  WEBHOOK = 'webhook',
  CUSTOM = 'custom',
}

/**
 * Credential Type Enumeration
 * Maps to BubbleLab credential types
 */
export enum CredentialType {
  DATABASE_CRED = 'DATABASE_CRED',
  SLACK_CRED = 'SLACK_CRED',
  FIRECRAWL_API_KEY = 'FIRECRAWL_API_KEY',
  OPENAI_CRED = 'OPENAI_CRED',
  ANTHROPIC_CRED = 'ANTHROPIC_CRED',
  GOOGLE_GEMINI_CRED = 'GOOGLE_GEMINI_CRED',
}

/**
 * Event Type Enumeration
 * Maps to BubbleLab trigger event types
 */
export enum EventType {
  WEBHOOK_HTTP = 'webhook/http',
  SCHEDULE = 'schedule',
  MANUAL = 'manual',
}

/**
 * Workflow Execution Status
 */
export enum ExecutionStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  SUCCESS = 'success',
  FAILED = 'failed',
  TIMEOUT = 'timeout',
}

// =============================================================================
// Canonical Schemas (Zod)
// =============================================================================

/**
 * Canonical Bubble Definition
 * Represents a single bubble in a workflow
 */
export const CanonicalBubbleSchema = z.object({
  id: z.string().optional(),
  name: z.string(),
  type: z.nativeEnum(BubbleType),
  config: z.record(z.any()),
  required_credentials: z.array(z.nativeEnum(CredentialType)).optional(),
  metadata: z.record(z.any()).optional(),
});

export type CanonicalBubble = z.infer<typeof CanonicalBubbleSchema>;

/**
 * Canonical BubbleFlow (Workflow) Definition
 * Represents a complete BubbleLab workflow
 */
export const CanonicalBubbleFlowSchema = z.object({
  id: z.string().optional(),
  name: z.string().min(1, 'Workflow name is required'),
  description: z.string().optional(),
  event_type: z.nativeEnum(EventType),
  code: z.string().optional(),
  bubbles: z.array(CanonicalBubbleSchema).optional(),
  required_credentials: z.record(z.string(), z.array(z.nativeEnum(CredentialType))).optional(),
  webhook_active: z.boolean().default(false),
  webhook_url: z.string().url().optional(),
  created_at: z.string().datetime().optional(),
  updated_at: z.string().datetime().optional(),
});

export type CanonicalBubbleFlow = z.infer<typeof CanonicalBubbleFlowSchema>;

/**
 * Canonical Workflow Execution Result
 */
export const CanonicalExecutionResultSchema = z.object({
  execution_id: z.string().optional(),
  flow_id: z.string(),
  status: z.nativeEnum(ExecutionStatus),
  output: z.any().optional(),
  error: z.string().optional(),
  started_at: z.string().datetime(),
  completed_at: z.string().datetime().optional(),
  duration_ms: z.number().optional(),
  logs: z.array(z.object({
    timestamp: z.string().datetime(),
    level: z.string(),
    message: z.string(),
  })).optional(),
});

export type CanonicalExecutionResult = z.infer<typeof CanonicalExecutionResultSchema>;

/**
 * Canonical BubbleLab Event
 * Represents events from BubbleLab to be processed by the orchestration layer
 */
export const CanonicalBubbleLabEventSchema = z.object({
  event_id: z.string().uuid(),
  event_type: z.enum([
    'workflow.created',
    'workflow.updated',
    'workflow.deleted',
    'workflow.executed',
    'workflow.execution_failed',
    'bubble.created',
    'bubble.updated',
  ]),
  flow_id: z.string().optional(),
  execution_id: z.string().optional(),
  timestamp: z.string().datetime(),
  data: z.any(),
  correlation_id: z.string().uuid().optional(),
});

export type CanonicalBubbleLabEvent = z.infer<typeof CanonicalBubbleLabEventSchema>;

/**
 * Canonical Credential Mapping
 * Maps credential types to their IDs
 */
export const CanonicalCredentialMappingSchema = z.record(
  z.nativeEnum(CredentialType),
  z.number()  // credential ID
);

export type CanonicalCredentialMapping = z.infer<typeof CanonicalCredentialMappingSchema>;

// =============================================================================
// Mapping Functions: BubbleLab Native -> Canonical
// =============================================================================

/**
 * Map BubbleLab API response to Canonical BubbleFlow
 *
 * @param apiResponse - Raw API response from BubbleLab
 * @returns Canonical BubbleFlow
 */
export function mapToCanonicalBubbleFlow(apiResponse: any): CanonicalBubbleFlow {
  return {
    id: apiResponse.id?.toString(),
    name: apiResponse.name || 'Unnamed Flow',
    description: apiResponse.description,
    event_type: mapEventType(apiResponse.eventType),
    code: apiResponse.code,
    bubbles: apiResponse.bubbles || [],
    required_credentials: apiResponse.requiredCredentials || {},
    webhook_active: apiResponse.webhookActive || false,
    webhook_url: apiResponse.webhookUrl,
    created_at: apiResponse.createdAt
      ? new Date(apiResponse.createdAt).toISOString()
      : undefined,
    updated_at: apiResponse.updatedAt
      ? new Date(apiResponse.updatedAt).toISOString()
      : undefined,
  };
}

/**
 * Map BubbleLab event type string to canonical enum
 */
function mapEventType(eventType: string): EventType {
  const mapping: Record<string, EventType> = {
    'webhook/http': EventType.WEBHOOK_HTTP,
    'schedule': EventType.SCHEDULE,
    'manual': EventType.MANUAL,
  };

  return mapping[eventType] || EventType.MANUAL;
}

/**
 * Map BubbleLab execution result to canonical form
 */
export function mapToCanonicalExecutionResult(apiResponse: any, flowId: string): CanonicalExecutionResult {
  const startedAt = apiResponse.startedAt
    ? new Date(apiResponse.startedAt).toISOString()
    : new Date().toISOString();

  const completedAt = apiResponse.completedAt
    ? new Date(apiResponse.completedAt).toISOString()
    : undefined;

  return {
    execution_id: apiResponse.id?.toString(),
    flow_id: flowId,
    status: mapExecutionStatus(apiResponse.status),
    output: apiResponse.output,
    error: apiResponse.error,
    started_at: startedAt,
    completed_at: completedAt,
    duration_ms: completedAt && startedAt
      ? new Date(completedAt).getTime() - new Date(startedAt).getTime()
      : undefined,
    logs: apiResponse.logs || [],
  };
}

/**
 * Map execution status string to canonical enum
 */
function mapExecutionStatus(status: string): ExecutionStatus {
  const mapping: Record<string, ExecutionStatus> = {
    'pending': ExecutionStatus.PENDING,
    'running': ExecutionStatus.RUNNING,
    'success': ExecutionStatus.SUCCESS,
    'failed': ExecutionStatus.FAILED,
    'timeout': ExecutionStatus.TIMEOUT,
  };

  return mapping[status?.toLowerCase()] || ExecutionStatus.PENDING;
}

// =============================================================================
// Mapping Functions: Canonical -> BubbleLab Native
// =============================================================================

/**
 * Map Canonical BubbleFlow to BubbleLab API request format
 */
export function mapFromCanonicalBubbleFlow(canonical: CanonicalBubbleFlow): any {
  return {
    name: canonical.name,
    description: canonical.description,
    code: canonical.code,
    eventType: canonical.event_type,
    webhookActive: canonical.webhook_active,
  };
}

/**
 * Map Canonical Credential Mapping to BubbleLab format
 */
export function mapFromCanonicalCredentials(canonical: CanonicalCredentialMapping): Record<string, number> {
  // Convert enum keys to string keys
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(canonical)) {
    result[key] = Number(value);
  }
  return result;
}

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Validate and parse a CanonicalBubbleFlow
 */
export function validateCanonicalBubbleFlow(data: unknown): CanonicalBubbleFlow {
  return CanonicalBubbleFlowSchema.parse(data);
}

/**
 * Validate and parse a CanonicalExecutionResult
 */
export function validateCanonicalExecutionResult(data: unknown): CanonicalExecutionResult {
  return CanonicalExecutionResultSchema.parse(data);
}

/**
 * Validate and parse a CanonicalBubbleLabEvent
 */
export function validateCanonicalBubbleLabEvent(data: unknown): CanonicalBubbleLabEvent {
  return CanonicalBubbleLabEventSchema.parse(data);
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Generate a unique correlation ID for tracking
 */
export function generateCorrelationId(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Convert UTC Date to ISO-8601 string (Law of UTC)
 */
export function toUTCISOString(date: Date): string {
  return date.toISOString();
}

/**
 * Parse ISO-8601 string to UTC Date
 */
export function fromUTCISOString(isoString: string): Date {
  return new Date(isoString);
}
